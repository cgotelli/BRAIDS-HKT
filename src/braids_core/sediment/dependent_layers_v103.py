"""Q5b dependent layers and authoritative surface recomposition.

The accepted Q5a column equations remain frozen.  This module composes their
native column transaction with Delft3D4's coarse-layer sorting operator and
the Q7a accepted dune state.  It is an implementation route, not QGIS product
admission.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from functools import lru_cache
import ctypes
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np

from .bed_columns_v101 import (
    ColumnMethodsV101, NativeColumnRuntimeV101, array_sha, digest, host,
)

ROOT = Path(__file__).resolve().parents[2]
HEADER = ROOT / "solver/wave/braids_dependent_layers_v103.h"
SOURCE = HEADER.with_suffix(".cpp")
SCOPE = "braids-q5b-dependent-layers-v103"

SORTING_CODES = {
    "off": 0,
    "discrete_shields": 1,
    "continuous_shields": 2,
    "wilcock_mcardell": 3,
    "transport_hiding": 4,
}
Q3_SURFACE_CONSUMERS = (
    "capacity", "hiding_exposure", "bed_slope", "secondary_flow",
    "bedload_boundary", "bedload_adaptation",
)
Q4_SURFACE_CONSUMERS = ("suspended_erosion", "suspended_deposition")


def dependent_target_controls(methods, nl):
    # Topology validation once, not a GPU-to-host field download every update.
    methods.targets(np.zeros(1), nl, dune_height=np.zeros(1))
    q = methods.q5a
    return ({"constant": 1, "depth": 2, "dune_height": 3}[methods.transport_layer],
        int(methods.coarse_layer), int(q.exchange_thickness_m > 0), q.lagrangian_underlayers,
        q.transport_thickness_m, methods.ttl_alpha, methods.ttl_min_m, methods.coarse_thickness_m,
        q.exchange_thickness_m,
        q.underlayer_thickness_m if q.lagrangian_thickness_m is None else q.lagrangian_thickness_m,
        q.underlayer_thickness_m if q.eulerian_thickness_m is None else q.eulerian_thickness_m)


@dataclass(frozen=True)
class DependentLayerMethodsV103:
    q5a: ColumnMethodsV101
    coarse_layer: bool = False
    coarse_thickness_m: float = .02
    sorting: str = "off"
    transport_layer: str = "constant"
    ttl_alpha: float = .01
    ttl_min_m: float = .001
    a_max: float = 20.
    sink_fraction_max: float = .05
    asfm: float = .523
    bsfm: float = .67
    sigma_sfm: float = .2
    water_density_kg_m3: float = 1000.
    gravity_m_s2: float = 9.81
    viscosity_m2_s: float = 1.e-6
    hiding_scope: str = "active"

    def __post_init__(self):
        if self.sorting not in SORTING_CODES:
            raise ValueError("Unknown Delft3D sorting choice")
        if self.transport_layer not in ("constant", "depth", "dune_height"):
            raise ValueError("Transport layer must be constant, depth or dune_height")
        if self.sorting != "off" and not self.coarse_layer:
            raise ValueError("Sorting requires the coarse layer; it is never silently enabled")
        if self.hiding_scope not in ("active", "active_coarse"):
            raise ValueError("IHidExpTrCrs scope must be active or active_coarse")
        values = (self.coarse_thickness_m, self.ttl_alpha, self.ttl_min_m,
                  self.a_max, self.sink_fraction_max, self.asfm, self.bsfm,
                  self.sigma_sfm, self.water_density_kg_m3,
                  self.gravity_m_s2, self.viscosity_m2_s)
        if not np.isfinite(values).all() or min(values[:3]) <= 0 or self.a_max < 1:
            raise ValueError("Invalid dependent-layer controls")
        if not 0 <= self.sink_fraction_max <= 1 or min(values[5:]) <= 0:
            raise ValueError("Invalid sorting physics")
        if self.q5a.bed != "layered":
            raise ValueError("Dependent layers require a layered bed")
        if self.q5a.exchange_thickness_m > 0 and not self.coarse_layer:
            pass

    @property
    def fingerprint(self):
        return digest({"scope": SCOPE, **asdict(self)})

    @property
    def density(self):
        return self.q5a.density

    @property
    def fraction_ids(self):
        return self.q5a.fraction_ids

    def targets(self, depth, nl, *, dune_height=None):
        depth = np.asarray(depth, dtype=float)
        if depth.ndim != 1 or not np.isfinite(depth).all() or np.any(depth < 0):
            raise ValueError("Finite nonnegative depth required")
        if self.transport_layer == "constant":
            active = np.full_like(depth, self.q5a.transport_thickness_m)
        elif self.transport_layer == "depth":
            active = np.maximum(self.ttl_alpha * depth, self.ttl_min_m)
        else:
            if dune_height is None:
                raise ValueError("TTLForm=3 requires accepted Q7a dune height")
            dune_height = np.asarray(dune_height, dtype=float)
            if dune_height.shape != depth.shape or not np.isfinite(dune_height).all() or np.any(dune_height < 0):
                raise ValueError("Invalid accepted dune height")
            active = np.maximum(self.ttl_alpha * dune_height, self.ttl_min_m)
        required = 2 + int(self.coarse_layer) + int(self.q5a.exchange_thickness_m > 0)
        if nl < required:
            raise ValueError("Layer topology lacks an independent basal layer")
        eulerian = (self.q5a.underlayer_thickness_m if self.q5a.eulerian_thickness_m is None
                    else self.q5a.eulerian_thickness_m)
        target = np.full((nl, len(depth)), eulerian)
        target[0] = active
        k = 1
        if self.coarse_layer:
            target[k] = self.coarse_thickness_m; k += 1
        if self.q5a.exchange_thickness_m > 0:
            target[k] = self.q5a.exchange_thickness_m; k += 1
        nlag = self.q5a.lagrangian_underlayers
        lag = self.q5a.underlayer_thickness_m if self.q5a.lagrangian_thickness_m is None else self.q5a.lagrangian_thickness_m
        if k+nlag >= nl:
            raise ValueError("Lagrangian layers leave no independent basal layer")
        target[k:k+nlag] = lag
        return np.ascontiguousarray(target)


@dataclass(frozen=True)
class ResolvedDependentLayersV103:
    methods: DependentLayerMethodsV103
    reasons: tuple[str,...]


def resolve_dependent_layers_v103(q5a,*,sorting="off",transport_layer="constant",
                                  coarse_layer=None,q7a_dune_state=False,**controls):
    """Automatically resolve mandatory Q5b auxiliaries and record why."""
    reasons=[];requires_coarse=sorting!="off"
    if coarse_layer is None:
        coarse_layer=requires_coarse
        reasons.append("coarse layer enabled because the selected sorting method requires it" if requires_coarse
                       else "coarse layer left off because no selected method requires it")
    elif requires_coarse and not coarse_layer:
        raise ValueError("Selected sorting method physically requires the coarse layer")
    if transport_layer=="dune_height":
        if not q7a_dune_state: raise ValueError("TTLForm=3 requires an accepted Q7a dune state")
        reasons.append("dune-height transport layer consumes the accepted Q7a dune state")
    methods=DependentLayerMethodsV103(q5a,coarse_layer=bool(coarse_layer),sorting=sorting,
                                      transport_layer=transport_layer,**controls)
    return ResolvedDependentLayersV103(methods,tuple(reasons))


@lru_cache(maxsize=1)
def native_sort_runtime_v103():
    compiler = shutil.which("g++")
    if not compiler:
        raise RuntimeError("Q5b native route requires an OpenMP C++ compiler")
    directory = Path(tempfile.mkdtemp(prefix="braids_q5b_v103_"))
    library = directory / ("q5b.dll" if os.name == "nt" else "q5b.so")
    flags = ["-O3", "-fopenmp", "-shared", "-ffp-contract=off", "-std=c++11"]
    if os.name == "nt":
        flags += ["-static-libstdc++", "-static-libgcc"]
    build = subprocess.run([compiler, *flags, str(SOURCE), "-o", str(library)],
                           cwd=directory, capture_output=True, text=True, timeout=120)
    if build.returncode:
        raise RuntimeError(build.stdout + build.stderr)
    dependency = os.add_dll_directory(str(Path(compiler).parent)) if os.name == "nt" else None
    dll = ctypes.CDLL(str(library))
    fn = dll.braids_dependent_sort_v103
    real = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    integer = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
    fn.argtypes = ([ctypes.c_int]*7 + [ctypes.c_double]*12 + [real]*13 +
                   [integer] + [real]*4 + [integer])
    fn.restype = None
    receipt = {
        "source": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "equations": hashlib.sha256(HEADER.read_bytes()).hexdigest(),
        "compiler": hashlib.sha256(Path(compiler).read_bytes()).hexdigest(),
        "binary": hashlib.sha256(library.read_bytes()).hexdigest(),
        "flags": flags,
    }
    return fn, dll, dependency, receipt


class NativeDependentLayerRuntimeV103:
    backend = "native_cpu_openmp"

    def __init__(self, methods, nc, nl, *, threads=4, **q5a_options):
        if not isinstance(methods, DependentLayerMethodsV103):
            raise TypeError("Versioned Q5b methods required")
        base_methods = replace(methods.q5a, transport_layer="constant")
        self.methods, self.nc, self.nl, self.threads = methods, nc, nl, threads
        self.target_controls = dependent_target_controls(methods, nl)
        self.base = NativeColumnRuntimeV101(base_methods, nc, nl, threads=threads,
                                            **q5a_options)
        self.rho = np.ascontiguousarray(methods.density)
        self.solid_rho = np.ascontiguousarray(methods.q5a.solid_density_kg_m3)
        self.diameter = np.ascontiguousarray(methods.q5a.diameter_m)
        self.fn, _, _, self.receipt = native_sort_runtime_v103()
        self.identity = digest({"scope": SCOPE, "backend": self.backend,
            "methods": methods.fingerprint, "base": self.base.identity,
            "runtime": self.receipt, "nc": nc, "nl": nl, "threads": threads})

    def targets(self, depth, *, dune_height=None):
        depth = np.ascontiguousarray(depth, dtype=float)
        if self.methods.transport_layer == "dune_height" and dune_height is None:
            raise ValueError("TTLForm=3 requires accepted dune height")
        dune = np.zeros(self.nc) if dune_height is None else np.ascontiguousarray(dune_height, dtype=float)
        if any(x.shape != (self.nc,) or not np.isfinite(x).all() or np.any(x < 0)
               for x in (depth, dune)):
            raise ValueError("Finite nonnegative depth/dune fields required")
        fn = native_sort_runtime_v103()[1].braids_dependent_targets_v103
        real = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        fn.argtypes = [ctypes.c_int]*7 + [ctypes.c_double]*7 + [real]*3
        fn.restype = None
        out = np.empty((self.nl, self.nc))
        fn(self.nc, self.nl, self.threads, *self.target_controls, depth, dune, out)
        return out

    def evaluate(self, mass, sv, delta, depth, dt, *, dune_height=None,
                 dune_length=None, shear_pa=None, bedload_mass_flux=None,
                 hiding_factor=None):
        target = self.targets(depth, dune_height=dune_height)
        return self.evaluate_targets(mass, sv, delta, target, dt,
            dune_length=dune_length, shear_pa=shear_pa,
            bedload_mass_flux=bedload_mass_flux, hiding_factor=hiding_factor)

    def evaluate_targets(self, mass, sv, delta, target, dt, *, dune_length,
                         shear_pa, bedload_mass_flux, hiding_factor, sorting_dt=None):
        m = self.methods; ng = len(m.fraction_ids)
        target = np.ascontiguousarray(target, dtype=float)
        interval = dt if sorting_dt is None else sorting_dt
        if (target.shape != (self.nl, self.nc) or not np.isfinite(target).all()
                or np.any(target <= 0) or not np.isfinite(interval) or interval < 0):
            raise ValueError("Invalid dependent target or sorting interval")
        shape = (ng, self.nc)
        tau = np.zeros(self.nc) if shear_pa is None else np.ascontiguousarray(shear_pa, dtype=float)
        length = np.ones(self.nc) if dune_length is None else np.ascontiguousarray(dune_length, dtype=float)
        bedload = np.zeros(shape) if bedload_mass_flux is None else np.ascontiguousarray(bedload_mass_flux, dtype=float)
        hiding = np.ones(shape) if hiding_factor is None else np.ascontiguousarray(hiding_factor, dtype=float)
        if (tau.shape != (self.nc,) or length.shape != (self.nc,) or
                bedload.shape != shape or hiding.shape != shape or
                not all(np.isfinite(a).all() for a in (tau, length, bedload, hiding))):
            raise ValueError("Invalid Q5b sorting fields")
        mobility = np.empty(shape); transfer = np.empty(self.nc)
        status = np.empty(self.nc, dtype=np.int32)
        initial=np.ascontiguousarray(mass,dtype=float);initial_sv=np.ascontiguousarray(sv,dtype=float)
        pulse=np.ascontiguousarray(delta,dtype=float)
        if initial.shape!=(self.nl,ng,self.nc) or initial_sv.shape!=(self.nl,self.nc) or pulse.shape!=shape:
            raise ValueError("Invalid Q5b column/pulse shapes")
        out=np.empty_like(initial);sout=np.empty_like(initial_sv)
        q=m.q5a
        self.fn(self.nc, ng, self.nl, self.threads, SORTING_CODES[m.sorting],q.top_update,q.porosity,interval,
            m.water_density_kg_m3, m.gravity_m_s2, m.viscosity_m2_s, m.a_max,
            m.sink_fraction_max, m.asfm, m.bsfm, m.sigma_sfm,q.solid_fraction,
            q.sand_solid_fraction,q.mud_solid_fraction,tau,length,target[0],self.rho,
            self.solid_rho,self.diameter,hiding,bedload,initial,initial_sv,pulse,
            self.base.phi,self.base.sigma,self.base.mud,out,sout,mobility,transfer,status)
        if np.any(status):
            raise ValueError("Q5b native sorting rejected a nonphysical state")
        zero=np.zeros_like(pulse)
        final,final_sv,basal=self.base.evaluate(out,sout,zero,target,dt,
            coarse_mobility=mobility if m.coarse_layer else None)
        return final,final_sv,basal,mobility,transfer,target


def authoritative_surface_v103(methods, mass, *, hiding_factor=None):
    """Return the sole Q3/Q4 surface fractions and optional coarse hiding view."""
    mass = np.asarray(mass, dtype=float)
    if mass.ndim != 3 or mass.shape[1] != len(methods.fraction_ids):
        raise ValueError("Layer/class/cell mass shape differs")
    top = mass[0] / np.asarray(methods.density)[:, None]
    hiding = top.copy()
    if methods.hiding_scope == "active_coarse":
        if not methods.coarse_layer or mass.shape[0] < 2:
            raise ValueError("Active+coarse hiding requires the coarse layer")
        hiding += mass[1] / np.asarray(methods.density)[:, None]
    def normalized(a):
        total = a.sum(axis=0)
        return np.where(total[None, :] > 0, a/np.maximum(total[None, :], 1e-300), 0.)
    result = {"active_fraction": normalized(top), "hiding_fraction": normalized(hiding),
              "producer_scope": SCOPE, "consumers": Q3_SURFACE_CONSUMERS + Q4_SURFACE_CONSUMERS}
    if hiding_factor is not None:
        h = np.asarray(hiding_factor, dtype=float)
        if h.shape != top.shape or not np.isfinite(h).all() or np.any(h <= 0):
            raise ValueError("Invalid classwise HidExp output")
        result["HidExp"] = h.copy()
    return result


class DependentColumnOwnerV103:
    """Accepted Q5b inventory owner shared by native CPU and resident CUDA."""
    def __init__(self,runtime,mass,sv,rock,area,*,time_s=0.):
        self.runtime=runtime; self.xp=getattr(runtime,"cp",np)
        self.mass=self.xp.array(mass,dtype=self.xp.float64,order="C",copy=True)
        self.sv=self.xp.array(sv,dtype=self.xp.float64,order="C",copy=True)
        self.rock=self.xp.array(rock,dtype=self.xp.float64,order="C",copy=True)
        self.area=self.xp.array(area,dtype=self.xp.float64,order="C",copy=True)
        ng=len(runtime.methods.fraction_ids)
        if (self.mass.shape!=(runtime.nl,ng,runtime.nc) or self.sv.shape!=(runtime.nl,runtime.nc) or
                self.rock.shape!=(runtime.nc,) or self.area.shape!=(runtime.nc,) or
                not all(np.isfinite(host(x)).all() for x in (self.mass,self.sv,self.rock,self.area)) or
                np.any(host(self.mass)<0) or np.any(host(self.sv)<=0) or np.any(host(self.sv)>1) or
                np.any(host(self.area)<=0) or not np.isfinite(time_s) or time_s<0):
            raise ValueError("Invalid initial accepted Q5b column")
        self.time_s=float(time_s);self.updates=0
        self.reference=host((self.mass*self.area[None,None,:]).sum(axis=(0,2))).copy()
        self.external=np.zeros(ng);self.basal=np.zeros(ng);self.pending=None;self._seal()
    def _token(self):
        return digest({"runtime":self.runtime.identity,"updates":self.updates,"time":self.time_s,
            "arrays":array_sha(*(host(x) for x in (self.mass,self.sv,self.rock,self.area,
                                                    self.reference,self.external,self.basal)))})
    def _seal(self):
        if self.xp is np:
            for x in (self.mass,self.sv,self.rock,self.area): x.setflags(write=False)
        self.state_sha256=self._token()
    def validate(self):
        if self._token()!=self.state_sha256: raise ValueError("Accepted Q5b column was mutated")
    @property
    def total(self):
        """Total class stock exposed to the downstream Q4/Q5 transaction."""
        return self.mass.sum(axis=0)
    @property
    def bed(self):
        volume=(self.mass/self.xp.asarray(self.runtime.rho)[None,:,None]).sum(axis=1)/self.sv
        return self.rock+volume.sum(axis=0)
    def surface(self,*,hiding_factor=None):
        return authoritative_surface_v103(self.runtime.methods,host(self.mass),hiding_factor=hiding_factor)
    def begin(self,delta,depth,dt,*,dune_height=None,dune_length=None,shear_pa=None,
              bedload_mass_flux=None,hiding_factor=None,wet=None,cancelled=False):
        self.validate()
        if self.pending is not None: raise ValueError("An uncommitted Q5b trial exists")
        if cancelled: raise InterruptedError("Q5b update cancelled before launch")
        if self.updates>=20: raise ValueError("Bounded Q5b route permits at most 20 accepted updates")
        trial=delta
        if wet is not None:
            mask=self.xp.asarray(wet)
            if mask.shape!=(self.runtime.nc,): raise ValueError("Wet/dry mask shape differs")
            trial=self.xp.ascontiguousarray(self.xp.where(mask[None,:],delta,0.))
        result=self.runtime.evaluate(self.mass,self.sv,trial,depth,dt,dune_height=dune_height,
            dune_length=dune_length,shear_pa=shear_pa,bedload_mass_flux=bedload_mass_flux,
            hiding_factor=hiding_factor)
        out,solid,basal,mobility,transfer,target=result
        before=host((self.mass*self.area[None,None,:]).sum(axis=(0,2)))
        after=host((out*self.area[None,None,:]).sum(axis=(0,2)))
        ext=host((trial*self.area[None,:]).sum(axis=1)); base=host((basal*self.area[None,:]).sum(axis=1))
        residual=after-before-ext-base
        if np.max(np.abs(residual))>3e-11*max(1.,np.max(before)):
            raise ValueError("Q5b classwise column budget does not close")
        self.pending=(out,solid,ext,base,float(dt),residual,mobility,transfer,target)
        self.pending_sha256=digest({"arrays":array_sha(*(host(x) for x in self.pending[:4]+self.pending[5:])),
                                    "dt":self.pending[4]})
        return self.pending
    def validate_pending(self):
        """Authenticate an unpublished Q5b candidate without publishing it."""
        self.validate()
        if self.pending is None:
            raise ValueError("No Q5b trial")
        token=digest({"arrays":array_sha(*(host(x) for x in self.pending[:4]+self.pending[5:])),
                      "dt":self.pending[4]})
        if token!=self.pending_sha256:
            raise ValueError("Q5b trial was mutated")
        return self.pending
    def compose_pending_layers(self,mass,sv,*,cancelled=False):
        """Replace a Q5b trial by a downstream split-process layer result.

        The caller must start from this owner's authenticated pending layers.
        Only the unpublished trial is changed: the accepted state and clock
        remain untouched until :meth:`commit`.  The additional bed-reservoir
        change is added to the existing external ledger, so one later commit
        publishes the transport and split-process result together.

        This is the narrow ownership seam used by the recursive v115 bank/
        fines composition.  It deliberately accepts absolute layers instead
        of another class-total pulse because bank collapse, repose and fines
        exchange have already selected their layer destinations.
        """
        self.validate_pending()
        if cancelled:
            raise InterruptedError("Q5b split-process composition cancelled")
        xp=self.xp
        candidate=xp.ascontiguousarray(xp.asarray(mass,dtype=xp.float64))
        solid=xp.ascontiguousarray(xp.asarray(sv,dtype=xp.float64))
        if (candidate.shape!=self.mass.shape or solid.shape!=self.sv.shape or
                not bool(xp.isfinite(candidate).all()) or
                not bool(xp.isfinite(solid).all()) or
                bool((candidate<0).any()) or bool((solid<=0).any()) or
                bool((solid>1).any())):
            raise ValueError("Invalid composed Q5b layer state")
        out,old_solid,ext,base,dt,old_residual,mobility,transfer,target=self.pending
        before=host((self.mass*self.area[None,None,:]).sum(axis=(0,2)))
        after=host((candidate*self.area[None,None,:]).sum(axis=(0,2)))
        prior=host((out*self.area[None,None,:]).sum(axis=(0,2)))
        composed_change=after-prior
        composed_ext=np.asarray(ext,dtype=float)+composed_change
        residual=after-before-composed_ext-np.asarray(base,dtype=float)
        if np.max(np.abs(residual))>3e-11*max(1.,np.max(before)):
            raise ValueError("Composed Q5b classwise column budget does not close")
        self.pending=(candidate,solid,composed_ext,base,dt,residual,
                      mobility,transfer,target)
        self.pending_sha256=digest({
            "arrays":array_sha(*(host(x) for x in
                self.pending[:4]+self.pending[5:])),"dt":dt})
        return {"class_bed_change_kg":composed_change.copy(),
                "maximum_budget_residual_kg":float(np.max(np.abs(residual)))}
    def commit(self,*,cancelled=False):
        self.validate()
        if self.pending is None: raise ValueError("No Q5b trial")
        if cancelled:
            self.rollback();raise InterruptedError("Q5b update cancelled before publication")
        token=digest({"arrays":array_sha(*(host(x) for x in self.pending[:4]+self.pending[5:])),"dt":self.pending[4]})
        if token!=self.pending_sha256: raise ValueError("Q5b trial was mutated")
        out,solid,ext,base,dt,residual,*_=self.pending
        self.mass,self.sv=out,solid;self.external+=ext;self.basal+=base
        self.time_s+=dt;self.updates+=1;self.pending=None;self._seal();return residual
    def rollback(self): self.pending=None;self.pending_sha256=None
    def checkpoint(self):
        self.validate()
        if self.pending is not None: raise ValueError("Checkpoint requires committed Q5b state")
        data={"schema":1,"scope":SCOPE,"runtime":self.runtime.identity,"methods":self.runtime.methods.fingerprint,
              "time_s":self.time_s,"updates":self.updates,
              **{k:host(getattr(self,k)).tolist() for k in ("mass","sv","rock","area","reference","external","basal")},
              "state_sha256":self.state_sha256}
        return {**data,"checkpoint_sha256":digest(data)}
    def restore(self,data):
        raw={k:v for k,v in data.items() if k!="checkpoint_sha256"}
        if (digest(raw)!=data.get("checkpoint_sha256") or data.get("schema")!=1 or data.get("scope")!=SCOPE or
                data.get("runtime")!=self.runtime.identity or data.get("methods")!=self.runtime.methods.fingerprint):
            raise ValueError("Q5b restart identity differs")
        candidate=DependentColumnOwnerV103(self.runtime,data["mass"],data["sv"],data["rock"],data["area"],time_s=data["time_s"])
        candidate.updates=int(data["updates"])
        if not 0<=candidate.updates<=20: raise ValueError("Invalid Q5b restart count")
        for key in ("reference","external","basal"):
            value=np.asarray(data[key],dtype=float)
            if value.shape!=getattr(candidate,key).shape or not np.isfinite(value).all(): raise ValueError("Invalid Q5b ledger")
            setattr(candidate,key,value)
        candidate._seal()
        if candidate.state_sha256!=data["state_sha256"]: raise ValueError("Q5b restart state differs")
        self.__dict__.update(candidate.__dict__)


def remap_dependent_owner_v103(owner,target_runtime,offsets,indices,weights,target_area):
    """Conservative dynamic-AMR remap of authoritative Q5b layers."""
    owner.validate()
    if owner.pending is not None: raise ValueError("Q5b remap requires committed state")
    if (owner.runtime.methods.fingerprint!=target_runtime.methods.fingerprint or
            owner.runtime.nl!=target_runtime.nl or owner.runtime.backend!=target_runtime.backend):
        raise ValueError("Q5b remap cannot change physics or backend")
    offsets=np.ascontiguousarray(offsets,dtype=np.int32);indices=np.ascontiguousarray(indices,dtype=np.int32)
    weights=np.ascontiguousarray(weights,dtype=float);area=np.ascontiguousarray(target_area,dtype=float)
    nc=target_runtime.nc
    if (offsets.shape!=(nc+1,) or offsets[0]!=0 or offsets[-1]!=len(indices) or weights.shape!=indices.shape or
            np.any(np.diff(offsets)<=0) or np.any(indices<0) or np.any(indices>=owner.runtime.nc) or
            not np.isfinite(weights).all() or np.any(weights<0) or area.shape!=(nc,) or np.any(area<=0)):
        raise ValueError("Invalid Q5b conservative remap")
    row=np.repeat(np.arange(nc),np.diff(offsets))
    if (not np.allclose(np.bincount(row,weights=weights),1.,atol=1e-13,rtol=0) or
            not np.allclose(np.bincount(indices,weights=weights*area[row],minlength=owner.runtime.nc),host(owner.area),atol=1e-12,rtol=0)):
        raise ValueError("Q5b remap does not conserve area")
    xp=owner.xp
    mass,sv,rock=owner.runtime.base.remap(owner.mass,owner.sv,owner.rock,
        xp.asarray(offsets),xp.asarray(indices),xp.asarray(weights),nc)
    # Conservative interpolation may round an exact unit solid fraction one
    # ulp above one; clamp only that dimensionless roundoff, never mass.
    sv=xp.minimum(sv,1.)
    mapped=DependentColumnOwnerV103(target_runtime,mass,sv,rock,area,time_s=owner.time_s)
    mapped.updates=owner.updates
    for key in ("reference","external","basal"): setattr(mapped,key,getattr(owner,key).copy())
    mapped._seal()
    return mapped


__all__ = ["DependentColumnOwnerV103", "DependentLayerMethodsV103", "NativeDependentLayerRuntimeV103",
           "ResolvedDependentLayersV103",
           "Q3_SURFACE_CONSUMERS", "Q4_SURFACE_CONSUMERS", "SORTING_CODES",
           "authoritative_surface_v103", "native_sort_runtime_v103", "remap_dependent_owner_v103",
           "resolve_dependent_layers_v103"]
