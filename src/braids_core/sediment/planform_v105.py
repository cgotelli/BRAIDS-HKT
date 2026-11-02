"""Q8a noncohesive redistribution and planform transaction (v105).

This module keeps the four source-distinct operations separate while giving
them one atomic owner.  The numerical kernels are BRAIDS implementations from
equations audited in the locked, unmodified Delft3D4 FLOW checkout; Delft3D is
never linked, compiled, or modified here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HEADER = ROOT / "solver/wave/braids_planform_v105.h"
SOURCE = HEADER.with_suffix(".cpp")
SCOPE = "braids-q8a-planform-v105"

RELAXATION = {
    "off",
    "classic_wet_slope_flux",
    "classic_scour_bundle",
    "braids_conservative_wet_dry_avalanche",
}
SOURCE_ANCHORS = {
    "classic_wet_slope_flux": "adjust_bedload.f90:417-440",
    "classic_scour_bundle": "rdscour.f90:87-160; shearx.f90:82-104; avalan.f90:107-170",
    "dry_bank": "bott3d.f90:884-989",
    "bedslope_output": "wrsedm.f90:349-351,951-964",
    "braids_conservative_wet_dry_avalanche": "solver/cut_cells/cutcell_solver.f90:8378-8512",
}


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _array_digest(*values) -> str:
    h = hashlib.sha256()
    for value in values:
        a = np.ascontiguousarray(value)
        h.update(a.dtype.str.encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def transfer_token_v105(owner: str, proposal) -> str:
    return digest({"owner": owner, "proposal": _array_digest(proposal)})


def adapt_toe_collapse_trial_v105(reference_layers, area, trial_result: Mapping):
    """Bridge the qualified physical-toe/undercut operator into Q8a.

    The historical operator returns trial areal layer masses and histories.
    Q8a consumes one conservative *cell mass* delta, preventing either the toe
    operator or the planform owner from transferring the inventory twice.
    """
    if not isinstance(trial_result, Mapping) or "trial_layers" not in trial_result:
        raise ValueError("Toe/collapse trial result is incomplete")
    status=trial_result.get("status",0)
    status_host=np.asarray(status.get() if hasattr(status,"get") else status)
    if np.any(status_host):raise ValueError("Rejected toe/collapse trial cannot enter Q8a")
    module=type(reference_layers).__module__.split(".")[0]
    if module=="cupy":
        import cupy as xp
    else:xp=np
    reference=xp.asarray(reference_layers);trial=xp.asarray(trial_result["trial_layers"])
    if reference.shape!=trial.shape or reference.ndim not in (3,4):
        raise ValueError("Toe/collapse trial layer geometry differs")
    cell_area=xp.asarray(area)
    spatial=reference.shape[2:]
    if cell_area.shape!=spatial:raise ValueError("Toe/collapse area geometry differs")
    delta=(trial-reference)*cell_area
    axes=(0,)+tuple(range(2,reference.ndim))
    if bool(xp.any(xp.abs(xp.sum(delta,axis=axes))>1.e-8)):
        raise ValueError("Toe/collapse trial is not classwise conservative")
    history=trial_result.get("trial_history")
    if reference.ndim==3:
        delta=delta[:,:,None,:]
        if history is not None:
            history={k:(v[None,:] if getattr(v,"ndim",0)==1 else v) for k,v in history.items()}
    return {"delta_mass":xp.ascontiguousarray(delta),
            "history":history,
            "token":transfer_token_v105("redistribution.braids_toe_collapse",
                delta.get() if hasattr(delta,"get") else delta)}


@dataclass(frozen=True)
class PlanformMethodsV105:
    relaxation: str = "off"
    dry_bank: bool = False
    toe_collapse: bool = False
    integrator: str = "implicit_fwave_midpoint"

    def __post_init__(self):
        if self.relaxation not in RELAXATION:
            raise ValueError("Unknown Q8a redistribution identity")
        if self.integrator != "implicit_fwave_midpoint":
            raise ValueError("Q8a GPU composition requires implicit_fwave_midpoint")
        if self.toe_collapse and self.relaxation not in (
                "off", "braids_conservative_wet_dry_avalanche"):
            raise ValueError("BRAIDS toe/collapse cannot share legacy FLOW relaxation ownership")

    @property
    def method_ids(self) -> tuple[str, ...]:
        primary = {
            "off": "redistribution.off",
            "classic_wet_slope_flux": "redistribution.avalanche",
            "classic_scour_bundle": "redistribution.classic_scour_bundle",
            "braids_conservative_wet_dry_avalanche":
                "redistribution.braids_conservative_wet_dry_avalanche",
        }[self.relaxation]
        extra = (("redistribution.dry_bank",) if self.dry_bank else ()) + (
            ("redistribution.braids_toe_collapse",) if self.toe_collapse else ())
        return (primary,) + extra

    @property
    def fingerprint(self):
        return digest({"scope": SCOPE, **asdict(self), "method_ids": self.method_ids})


@dataclass(frozen=True)
class PlanformParametersV105:
    wet_slope: float = 10.0
    avalanche_time_s: float = 86400.0
    dry_slope: float = 1.0
    wet_depth_switch_m: float = 0.1
    relaxation_length_m: float = 1.0
    thetsd: float = 0.0
    hmaxth_m: float = 1.0
    sedthr_m: float = 0.0
    scour_slope: float = 1.0

    def __post_init__(self):
        values = np.asarray(tuple(asdict(self).values()), dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Finite Q8a controls required")
        if min(self.wet_slope, self.avalanche_time_s, self.dry_slope,
               self.relaxation_length_m, self.scour_slope) <= 0:
            raise ValueError("Positive Q8a slope/time/length controls required")
        if not 0 <= self.thetsd <= 1 or self.hmaxth_m < 0 or self.sedthr_m < 0 or \
                self.wet_depth_switch_m < 0:
            raise ValueError("Invalid Q8a dry-bank/wet-front control")

    @property
    def diffusivity_m2_s(self):
        return 0.5 * self.relaxation_length_m**2 / self.avalanche_time_s


@dataclass(frozen=True)
class ScourMapV105:
    reference: tuple[int, ...]
    application: tuple[int, ...]
    factor: tuple[float, ...]
    slope: float

    def __post_init__(self):
        if not (len(self.reference) == len(self.application) == len(self.factor)):
            raise ValueError("Scour mapping columns differ")
        if not np.isfinite(self.factor).all() or not np.isfinite(self.slope) or self.slope <= 0:
            raise ValueError("Invalid Scour factor/slope")

    @classmethod
    def read(cls, path, *, one_based_shape: tuple[int, int] | None = None):
        """Read the locked FLOW payload: flag, slope, then Mref Nref Mmap Nmap factor."""
        rows = [line.split("#", 1)[0].strip() for line in Path(path).read_text().splitlines()]
        rows = [line for line in rows if line]
        if len(rows) < 2:
            raise ValueError("Scour file requires flag and slope")
        flag = int(rows[0]); slope = float(rows[1])
        if flag == 0:
            return None
        if one_based_shape is None:
            raise ValueError("Scour M/N mapping requires the prepared structured-patch shape")
        ny, nx = one_based_shape
        reference, application, factor = [], [], []
        for line in rows[2:]:
            fields = line.split()
            if len(fields) != 5:
                raise ValueError("Each Scour map row requires Mref Nref Mmap Nmap factor")
            mr, nr, ma, na = map(int, fields[:4]); value = float(fields[4])
            if not (1 <= mr <= nx and 1 <= ma <= nx and 1 <= nr <= ny and 1 <= na <= ny):
                raise ValueError("Scour M/N point is outside the prepared grid")
            reference.append((nr-1)*nx+mr-1); application.append((na-1)*nx+ma-1)
            factor.append(value)
        return cls(tuple(reference), tuple(application), tuple(factor), slope)


def _real(value, shape, name, *, positive=False, nonnegative=False):
    a = np.ascontiguousarray(np.broadcast_to(np.asarray(value, np.float64), shape).copy())
    if not np.isfinite(a).all() or (positive and np.any(a <= 0)) or \
            (nonnegative and np.any(a < 0)):
        raise ValueError(f"Invalid {name}")
    return a


def _binary(value, shape, name):
    a = np.ascontiguousarray(np.broadcast_to(np.asarray(value, np.int32), shape).copy())
    if np.any((a != 0) & (a != 1)):
        raise ValueError(f"{name} must be binary")
    return a


@lru_cache(maxsize=1)
def _native_library_v105():
    compiler = shutil.which("g++")
    if not compiler:
        raise RuntimeError("Q8a native CPU route requires an OpenMP C++ compiler")
    flags = ["-O3", "-fopenmp", "-shared", "-ffp-contract=off", "-std=c++11"]
    # A PE build timestamp otherwise changes the binary identity in each new
    # process, making an unchanged-source checkpoint impossible to reopen.
    if os.name == "nt": flags += ["-static-libstdc++", "-static-libgcc", "-Wl,--no-insert-timestamp"]
    inputs = dict(source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                  equations_sha256=hashlib.sha256(HEADER.read_bytes()).hexdigest(),
                  compiler_sha256=hashlib.sha256(Path(compiler).read_bytes()).hexdigest(),flags=flags)
    cache = ROOT / 'build/planform_v105'; cache.mkdir(parents=True, exist_ok=True)
    directory = cache / digest(inputs)
    name = 'q8a.dll' if os.name == 'nt' else 'q8a.so'
    library = directory / name
    if not directory.exists():
        staging = Path(tempfile.mkdtemp(prefix='compile_', dir=cache))
        built = subprocess.run([compiler, *flags, str(SOURCE), '-o', str(staging/name)],
                               cwd=staging,capture_output=True,text=True,timeout=120)
        (staging/'build.log').write_text(built.stdout+built.stderr)
        if built.returncode:raise RuntimeError(built.stdout+built.stderr)
        receipt = dict(inputs,binary_sha256=hashlib.sha256((staging/name).read_bytes()).hexdigest())
        (staging/'identity.json').write_text(json.dumps(receipt,indent=2))
        try:staging.rename(directory)
        except OSError:
            # Another process may have published the same source/compiler key.
            # Keep our staging diagnostics; never overwrite a loaded library.
            if not directory.exists():raise
    receipt = json.loads((directory/'identity.json').read_text())
    if (any(receipt.get(k)!=v for k,v in inputs.items()) or
            hashlib.sha256(library.read_bytes()).hexdigest()!=receipt['binary_sha256']):
        raise ValueError('Cached planform runtime identity differs')
    dependency = os.add_dll_directory(str(Path(compiler).parent)) if os.name == "nt" else None
    dll = ctypes.CDLL(str(library))
    real = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
    integer = np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS")
    slopes = dll.braids_q8a_slopes_v105
    slopes.argtypes = [ctypes.c_int]*3 + [ctypes.c_double]*2 + [real]*4 + [integer]
    shear = dll.braids_q8a_shear_map_v105
    shear.argtypes = [ctypes.c_int]*3 + [real, integer, integer, real, real, integer]
    flux = dll.braids_q8a_classic_flux_v105
    flux.argtypes = [ctypes.c_int]*3 + [ctypes.c_double]*4 + [real]*7 + [integer]
    scour = dll.braids_q8a_scour_v105
    scour.argtypes = [ctypes.c_int]*4 + [ctypes.c_double]*3 + [real]*5 + [integer]
    dry = dll.braids_q8a_dry_bank_v105
    dry.argtypes = [ctypes.c_int]*4 + [ctypes.c_double]*3 + [real]*4 + [integer]*2 + [real]*6 + [integer]
    repose = dll.braids_q8a_repose_v105
    repose.argtypes = [ctypes.c_int]*4 + [ctypes.c_double]*7 + [real]*6 + [integer]*2
    return dll, slopes, shear, flux, scour, dry, repose, dependency, receipt


class NativePlanformRuntimeV105:
    backend = "native_cpu_openmp"
    xp = np

    def __init__(self, methods=PlanformMethodsV105(), parameters=PlanformParametersV105(), *, threads=4):
        if type(threads) is not int or not 1 <= threads <= 256:
            raise ValueError("Explicit CPU thread count in [1,256] required")
        if methods.dry_bank and parameters.thetsd <= 0:
            raise ValueError("redistribution.dry_bank requires positive ThetSD")
        self.methods, self.parameters, self.threads = methods, parameters, threads
        (self._dll, self._slopes, self._shear, self._flux, self._scour, self._dry,
         self._repose, self._dependency, self.receipt) = _native_library_v105()
        self.identity = digest({"scope": SCOPE, "backend": self.backend,
            "methods": methods.fingerprint, "parameters": asdict(parameters), "runtime": self.receipt})

    def slopes(self, bed, *, dx_m, dy_m):
        shape = np.asarray(bed).shape
        if len(shape) != 2: raise ValueError("Q8a requires a structured 2-D patch")
        a = _real(bed, shape, "bed"); out = np.empty((3, a.size)); status = np.empty(a.size, np.int32)
        self._slopes(shape[1], shape[0], self.threads, float(dx_m), float(dy_m), a.ravel(),
                     out[0], out[1], out[2], status)
        if np.any(status): raise ValueError("Invalid Q8a bed-slope state")
        return {"dzduu": out[0].reshape(shape), "dzdvv": out[1].reshape(shape),
                "magnitude": out[2].reshape(shape), "output_id": "output.bedslope"}

    def map_scour_shear(self, shear, mapping: ScourMapV105 | None):
        a = np.ascontiguousarray(np.asarray(shear, np.float64).ravel()); out = np.empty_like(a)
        status = np.empty(a.size, np.int32)
        if mapping is None:
            reference = application = np.empty(0, np.int32); factor = np.empty(0)
        else:
            reference = np.ascontiguousarray(mapping.reference, dtype=np.int32)
            application = np.ascontiguousarray(mapping.application, dtype=np.int32)
            factor = np.ascontiguousarray(mapping.factor, dtype=np.float64)
        if reference.size and (reference.min() < 0 or application.min() < 0 or
                               reference.max() >= a.size or application.max() >= a.size):
            raise ValueError("Scour mapping index outside shear state")
        self._shear(a.size, reference.size, self.threads, a, reference, application, factor, out, status)
        if np.any(status): raise ValueError("Invalid Scour shear mapping")
        return out.reshape(np.asarray(shear).shape)

    def _validated(self, layers, rock, depth, area, bulk, erosion, fraction, fixfac, wet, erodible):
        q = np.ascontiguousarray(layers, dtype=np.float64)
        if q.ndim != 4 or min(q.shape) < 1:
            raise ValueError("layers must have shape (layer,grain,y,x)")
        nl, ng, ny, nx = q.shape; shape=(ny,nx)
        bulk = _real(bulk, (ng,), "bulk density", positive=True)
        rock = _real(rock, shape, "bedrock")
        depth = _real(depth, shape, "water depth", nonnegative=True)
        area = _real(area, shape, "cell area", positive=True)
        wet = _binary(depth > 0 if wet is None else wet, shape, "wet mask")
        erodible = _binary(np.ones(shape) if erodible is None else erodible, shape, "erodible mask")
        erosion = _real(0. if erosion is None else erosion, (ng,ny,nx), "erosion mass")
        fraction = _real(1./ng if fraction is None else fraction, (ng,ny,nx), "surface fraction", nonnegative=True)
        total=fraction.sum(axis=0)
        if np.any(total <= 0): raise ValueError("Positive surface-fraction sum required")
        fraction /= total
        fixfac = _real(1. if fixfac is None else fixfac, (ng,ny,nx), "fixfac", nonnegative=True)
        if np.any(q < 0) or not np.isfinite(q).all(): raise ValueError("Finite nonnegative layer inventory required")
        return q, rock, depth, area, bulk, erosion, fraction, fixfac, wet, erodible

    def advance(self, layers, rock, depth, area, bulk_density, *, dt_s, dx_m, dy_m,
                erosion_mass_areal=None, surface_fraction=None, fixfac=None, wet=None,
                erodible=None, scour_map=None, shear=None, toe_face_delta_mass=None,
                toe_history=None, transfer_tokens=()):
        if not np.isfinite([dt_s,dx_m,dy_m]).all() or min(dt_s,dx_m,dy_m) <= 0:
            raise ValueError("Positive finite Q8a time/grid scales required")
        q,rock,depth,area,bulk,erosion,fraction,fixfac,wet,erodible=self._validated(
            layers,rock,depth,area,bulk_density,erosion_mass_areal,surface_fraction,fixfac,wet,erodible)
        nl,ng,ny,nx=q.shape;nc=nx*ny;accepted=q.copy();trial=q.copy();status=np.zeros(nc,np.int32)
        base=rock.ravel().copy()
        for l in range(1,nl):
            for g in range(ng): base += q[l,g].ravel()/bulk[g]
        bed=base.copy()
        for g in range(ng): bed += trial[0,g].ravel()/bulk[g]
        slopes=self.slopes(bed.reshape(ny,nx),dx_m=dx_m,dy_m=dy_m)
        classic_flux={"qx_m3_s":np.zeros((ny,nx)),"qy_m3_s":np.zeros((ny,nx))}
        substeps=0;moved=np.zeros((ng,ny,nx))
        if self.methods.relaxation == "classic_wet_slope_flux":
            self._flux(nx,ny,self.threads,float(dx_m),float(dy_m),self.parameters.wet_slope,
                self.parameters.avalanche_time_s,area.ravel(),bed,slopes["dzduu"].ravel(),
                slopes["dzdvv"].ravel(),slopes["magnitude"].ravel(),
                classic_flux["qx_m3_s"].ravel(),classic_flux["qy_m3_s"].ravel(),status)
        elif self.methods.relaxation == "classic_scour_bundle":
            slope=self.parameters.scour_slope if scour_map is None else scour_map.slope
            result=np.empty((ng,nc));self._scour(nx,ny,ng,self.threads,float(dx_m),float(dy_m),
                float(slope),area.ravel(),bulk,base,trial[0].reshape(ng,nc),result,status)
            trial[0]=result.reshape((ng,ny,nx))
        elif self.methods.relaxation == "braids_conservative_wet_dry_avalanche":
            result=np.empty((ng,nc));sub=np.zeros(1,np.int32)
            self._repose(nx,ny,ng,self.threads,float(dx_m),float(dy_m),float(dt_s),
                self.parameters.diffusivity_m2_s,self.parameters.wet_slope,self.parameters.dry_slope,
                self.parameters.wet_depth_switch_m,area.ravel(),bulk,base,depth.ravel(),
                trial[0].reshape(ng,nc),result,status,sub)
            trial[0]=result.reshape((ng,ny,nx));substeps=int(sub[0])
        if np.any(status): raise ValueError("Q8a primary redistribution failed")
        if self.methods.dry_bank:
            bed=base.copy()
            for g in range(ng):bed+=trial[0,g].ravel()/bulk[g]
            result=np.empty((ng,nc));delta=np.empty((ng,nc));status.fill(0)
            self._dry(nx,ny,ng,self.threads,self.parameters.thetsd,self.parameters.hmaxth_m,
                self.parameters.sedthr_m,area.ravel(),bulk,bed,depth.ravel(),wet.ravel(),
                erodible.ravel(),erosion.reshape(ng,nc),fraction.reshape(ng,nc),fixfac.reshape(ng,nc),
                trial[0].reshape(ng,nc),result,delta,status)
            if np.any(status): raise ValueError("Q8a dry-bank transfer failed")
            trial[0]=result.reshape((ng,ny,nx));moved+=delta.reshape((ng,ny,nx))
        tokens=list(transfer_tokens)
        if toe_face_delta_mass is not None:
            if not self.methods.toe_collapse:
                raise ValueError("Toe/collapse proposal supplied without its selected method")
            raw=np.asarray(toe_face_delta_mass,np.float64)
            if raw.shape==(ng,ny,nx):
                delta=np.zeros_like(trial);delta[0]=_real(raw,(ng,ny,nx),"toe/collapse cell mass delta")
            elif raw.shape==trial.shape:
                delta=_real(raw,trial.shape,"toe/collapse layer mass delta")
            else:raise ValueError("Toe/collapse proposal must resolve active or all bed layers")
            token=transfer_token_v105("redistribution.braids_toe_collapse",raw)
            if token in tokens: raise ValueError("Toe/collapse material transfer applied twice")
            if np.max(np.abs(np.sum(delta,axis=(0,2,3))))>1.e-9:
                raise ValueError("Toe/collapse proposal must conserve each sediment class")
            next_layers=trial+delta/area
            if np.any(next_layers < -1.e-11): raise ValueError("Toe/collapse proposal exceeds donor inventory")
            trial=np.maximum(next_layers,0.);moved+=np.sum(delta,axis=0);tokens.append(token)
        elif self.methods.toe_collapse:
            raise ValueError("Selected toe/collapse route requires an explicit conservative proposal")
        bed=rock.copy()
        for l in range(nl):
            for g in range(ng):bed+=trial[l,g]/bulk[g]
        if not np.isfinite(trial).all() or np.any(trial<0):raise ValueError("Invalid Q8a trial")
        return {"layers":trial,"accepted_unchanged":np.array_equal(accepted,q),"bed_elevation_m":bed,
                "bedslope":self.slopes(bed,dx_m=dx_m,dy_m=dy_m),"classic_flux":classic_flux,
                "mapped_shear":None if shear is None else self.map_scour_shear(shear,scour_map),
                "dry_bank_delta_mass_kg":moved,"substeps":substeps,"transfer_tokens":tuple(tokens),
                "method_ids":self.methods.method_ids,"runtime_identity":self.identity,"dt_s":float(dt_s),
                "toe_history":None if toe_history is None else {
                    k:np.ascontiguousarray(v).copy() for k,v in toe_history.items()}}


class PlanformOwnerV105:
    """Atomic accepted Q8a state with cancellation/restart identity."""
    def __init__(self, runtime, layers, rock, depth, area, bulk_density, *, time_s=0., updates=0,
                 auxiliary_state=None):
        self.runtime=runtime;self.layers=np.ascontiguousarray(layers,np.float64)
        self.rock=np.ascontiguousarray(rock,np.float64);self.depth=np.ascontiguousarray(depth,np.float64)
        self.area=np.ascontiguousarray(area,np.float64);self.bulk=np.ascontiguousarray(bulk_density,np.float64)
        self.time_s=float(time_s);self.updates=int(updates);self._trial=None
        self.auxiliary={} if auxiliary_state is None else {
            k:np.ascontiguousarray(v).copy() for k,v in auxiliary_state.items()}
        runtime._validated(self.layers,self.rock,self.depth,self.area,self.bulk,None,None,None,None,None)

    @property
    def state_sha256(self):
        values=[self.layers,self.rock,self.depth,self.area,self.bulk,
                np.asarray([self.time_s,self.updates],np.float64)]
        for key in sorted(self.auxiliary):values.extend((np.frombuffer(key.encode(),np.uint8),self.auxiliary[key]))
        return _array_digest(*values)

    def propose(self, **kwargs):
        if self._trial is not None:raise RuntimeError("Q8a trial already pending")
        self._trial=self.runtime.advance(self.layers,self.rock,self.depth,self.area,self.bulk,**kwargs)
        return self._trial

    def commit(self, *, cancelled=False, accept=True):
        if self._trial is None:raise RuntimeError("No Q8a trial pending")
        trial=self._trial;self._trial=None
        if cancelled:raise InterruptedError("Q8a transaction cancelled before publication")
        if not accept:return False
        self.layers=np.ascontiguousarray(trial["layers"]);self.time_s+=float(trial.get("dt_s",0.))
        if trial.get("toe_history") is not None:self.auxiliary=trial["toe_history"]
        self.updates+=1;return True

    def discard(self): self._trial=None

    def checkpoint(self):
        if self._trial is not None:raise RuntimeError("Cannot checkpoint a pending Q8a trial")
        return {"schema":"braids.planform.v105","runtime_identity":self.runtime.identity,
            "time_s":self.time_s,"updates":self.updates,"layers":self.layers.copy(),
            "rock":self.rock.copy(),"depth":self.depth.copy(),"area":self.area.copy(),
            "bulk":self.bulk.copy(),"auxiliary":{k:v.copy() for k,v in self.auxiliary.items()},
            "state_sha256":self.state_sha256}

    def restore(self, document: Mapping):
        if document.get("schema")!="braids.planform.v105" or document.get("runtime_identity")!=self.runtime.identity:
            raise ValueError("Q8a checkpoint identity differs")
        candidate=PlanformOwnerV105(self.runtime,document["layers"],document["rock"],
            document["depth"],document["area"],document["bulk"],time_s=document["time_s"],
            updates=document["updates"],auxiliary_state=document.get("auxiliary"))
        if candidate.state_sha256!=document.get("state_sha256"):raise ValueError("Q8a checkpoint digest differs")
        self.__dict__.update(candidate.__dict__)


def remap_planform_owner_v105(owner, offsets, sources, weights, target_area):
    """Parent-owned conservative remap; every source mass is transferred once."""
    offsets=np.asarray(offsets,np.int64);sources=np.asarray(sources,np.int64);weights=np.asarray(weights,float)
    target_area=np.asarray(target_area,float);nt=target_area.size;nl,ng,ny,nx=owner.layers.shape
    old_area=owner.area.ravel();old_layers=owner.layers.reshape(nl,ng,-1)
    if (offsets.shape!=(nt+1,) or offsets[0]!=0 or offsets[-1]!=len(sources) or
            len(weights)!=len(sources) or np.any(sources<0) or np.any(sources>=old_area.size) or
            not np.isfinite(target_area).all() or np.any(target_area<=0)):
        raise ValueError("Invalid Q8a remap CSR")
    mapped=np.zeros((nl,ng,nt));rock=np.zeros(nt);depth=np.zeros(nt)
    row=np.repeat(np.arange(nt),np.diff(offsets))
    if (not np.isfinite(weights).all() or np.any(weights<0) or
            not np.allclose(np.bincount(row,weights=weights,minlength=nt),1.,rtol=0,atol=1e-13) or
            not np.allclose(np.bincount(sources,weights=weights*target_area.ravel()[row],
                                        minlength=old_area.size),old_area,rtol=0,atol=1e-12)):
        raise ValueError("Q8a remap weights do not conserve area")
    for t in range(nt):
        sl=slice(offsets[t],offsets[t+1]);idx=sources[sl];w=weights[sl]
        mapped[:,:,t]=np.sum(old_layers[:,:,idx]*w,axis=2)
        rock[t]=np.sum(owner.rock.ravel()[idx]*w);depth[t]=np.sum(owner.depth.ravel()[idx]*w)
    before=np.sum(old_layers*old_area,axis=2);after=np.sum(mapped*target_area.ravel(),axis=2)
    if not np.allclose(before,after,rtol=0,atol=2e-10):raise ValueError("Q8a AMR remap does not conserve class/layer mass")
    shape=target_area.shape if target_area.ndim==2 else (1,nt)
    auxiliary={}
    for key,value in owner.auxiliary.items():
        a=np.asarray(value);leading=a.shape[:-2]
        if a.shape[-2:]!=owner.area.shape:raise ValueError(f"Q8a auxiliary {key} is not cell-owned")
        flat=a.reshape((-1,old_area.size));out=np.empty((flat.shape[0],nt),dtype=a.dtype)
        for t in range(nt):
            idx=sources[offsets[t]:offsets[t+1]];w=weights[offsets[t]:offsets[t+1]]
            if np.issubdtype(a.dtype,np.integer):out[:,t]=np.max(flat[:,idx],axis=1)
            else:out[:,t]=np.sum(flat[:,idx]*w,axis=1)
        auxiliary[key]=out.reshape(leading+shape)
    return PlanformOwnerV105(owner.runtime,mapped.reshape((nl,ng)+shape),rock.reshape(shape),
        depth.reshape(shape),target_area.reshape(shape),owner.bulk,time_s=owner.time_s,
        updates=owner.updates,auxiliary_state=auxiliary)

