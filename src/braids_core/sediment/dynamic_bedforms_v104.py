"""Q7b dynamic bedforms on the accepted Q7a/Q5b state.

The constitutive choices and source/sink ordering follow locked Delft3D4 FLOW
2DH sources.  BRAIDS uses a conservative TVD finite-volume advection operator
that is shared by OpenMP and CUDA; it does not modify or link Delft3D.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HEADER = ROOT / "solver/wave/braids_dynamic_bedforms_v104.h"
SOURCE = HEADER.with_suffix(".cpp")
SCOPE = "braids-q7b-dynamic-bedforms-v104"

RELAXATION = {
    "equilibrium": 0,
    "constant_time": 1,
    "constant_length": 2,
    "length_factor": 3,
    "depth_limited": 4,
}
ROUGHNESS = {"van_rijn_2007_static": 0, "van_rijn_1984": 1, "power": 2}
BOUNDARY = {"closed": 0, "open": 1, "periodic": 2}
FEEDBACK_CONSUMERS = ("hydraulic_friction", "transport_capacity", "q5b_transport_layer")


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def host(value):
    if type(value).__module__.split(".")[0] == "cupy":
        import cupy as cp
        return cp.asnumpy(value)
    return np.asarray(value)


def array_sha(*values) -> str:
    h = hashlib.sha256()
    for value in values:
        a = np.ascontiguousarray(host(value))
        h.update(a.dtype.str.encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class DynamicBedformMethodsV104:
    relaxation: str = "equilibrium"
    advection: bool = False
    clock: str = "hydrodynamic"
    cfl_control: bool = False
    roughness: str = "van_rijn_1984"
    integrator: str = "implicit_fwave_midpoint"

    def __post_init__(self):
        if self.relaxation not in RELAXATION:
            raise ValueError("Unknown Delft3D dynamic-bedform relaxation choice")
        if self.clock not in ("hydrodynamic", "morphological"):
            raise ValueError("Bedform clock must be hydrodynamic or morphological")
        if self.roughness not in ROUGHNESS:
            raise ValueError("Dynamic feedback supports Van Rijn 1984 or power roughness")
        if self.integrator != "implicit_fwave_midpoint":
            raise ValueError("Q7b GPU composition requires implicit_fwave_midpoint")
        if self.relaxation == "equilibrium" and (self.advection or self.cfl_control):
            raise ValueError("FLOW exposes bedform advection/CFL only with non-equilibrium relaxation")
        if self.cfl_control and not self.advection:
            raise ValueError("Bedform CFL control requires bedform advection")

    @property
    def fingerprint(self):
        return digest({"scope": SCOPE, **asdict(self)})


@dataclass(frozen=True)
class ResolvedDynamicBedformsV104:
    methods: DynamicBedformMethodsV104
    celerity: str
    active: bool
    automatic: tuple[str, ...]

    @property
    def fingerprint(self):
        return digest({"scope": SCOPE, "methods": asdict(self.methods),
                       "celerity": self.celerity, "active": self.active,
                       "automatic": self.automatic})


def resolve_dynamic_bedforms_v104(methods: DynamicBedformMethodsV104) -> ResolvedDynamicBedformsV104:
    active = methods.relaxation != "equilibrium"
    celerity = "bedload_froude" if methods.relaxation == "depth_limited" else "velocity_power"
    automatic = () if not active else (f"{celerity} celerity selected by {methods.relaxation}",)
    return ResolvedDynamicBedformsV104(methods, celerity, active, automatic)


@dataclass(frozen=True)
class DynamicBedformParametersV104:
    gravity_m_s2: float = 9.81
    constant_time_s: float = 3600.0
    constant_length_m: float = 100.0
    length_factor: float = 5.0
    maximum_depth_m: float = 10.0
    depth_length_exponent: float = 1.0
    maximum_length_factor: float = 20.0
    celerity_coefficient: float = 0.01
    celerity_exponent: float = 1.0
    gamma_exponent: float = 1.0
    bedload_celerity_coefficient: float = 1.0
    minimum_gamma: float = 0.05
    roughness_coefficient: float = 1.1
    roughness_exponent: float = 1.0
    dry_depth_m: float = 1.0e-6
    cfl_target: float = 0.8
    ripple_roughness_relaxation_s: float = 0.0
    mega_roughness_relaxation_s: float = 0.0
    dune_roughness_relaxation_s: float = 0.0

    def __post_init__(self):
        values = np.asarray(tuple(asdict(self).values()), dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Finite dynamic-bedform parameters required")
        positive = (self.gravity_m_s2, self.constant_time_s, self.constant_length_m,
                    self.length_factor, self.maximum_depth_m, self.maximum_length_factor,
                    self.celerity_coefficient, self.bedload_celerity_coefficient,
                    self.roughness_coefficient, self.cfl_target)
        if min(positive) <= 0 or self.minimum_gamma < 0 or self.minimum_gamma > 1:
            raise ValueError("Invalid positive dynamic-bedform parameter")
        if not 0 < self.cfl_target <= 1:
            raise ValueError("Bedform CFL target must lie in (0,1]")
        if min(self.ripple_roughness_relaxation_s,self.mega_roughness_relaxation_s,
               self.dune_roughness_relaxation_s) < 0:
            raise ValueError("Van Rijn 2007 roughness relaxation times cannot be negative")

    def wire(self):
        return np.ascontiguousarray([
            self.gravity_m_s2, self.constant_time_s, self.constant_length_m,
            self.length_factor, self.maximum_depth_m, self.depth_length_exponent,
            self.maximum_length_factor, self.celerity_coefficient,
            self.celerity_exponent, self.gamma_exponent,
            self.bedload_celerity_coefficient, self.minimum_gamma,
            self.roughness_coefficient, self.roughness_exponent, self.dry_depth_m,
            self.ripple_roughness_relaxation_s,self.mega_roughness_relaxation_s,
            self.dune_roughness_relaxation_s,
        ], dtype=np.float64)


def _modes(methods: DynamicBedformMethodsV104):
    return np.ascontiguousarray([
        RELAXATION[methods.relaxation], int(methods.advection),
        int(methods.clock == "morphological"), int(methods.cfl_control),
        ROUGHNESS[methods.roughness],
    ], dtype=np.int32)


def _boundary_codes(boundary):
    if isinstance(boundary, str): boundary = (boundary,)*4
    if len(boundary) != 4 or any(x not in BOUNDARY for x in boundary):
        raise ValueError("Four bedform boundaries (west,east,south,north) are required")
    code = np.ascontiguousarray([BOUNDARY[x] for x in boundary], dtype=np.int32)
    if (code[0] == 2) != (code[1] == 2) or (code[2] == 2) != (code[3] == 2):
        raise ValueError("Periodic bedform boundaries must be paired")
    return code


def _field(value, shape, name, *, positive=False, nonnegative=False, dtype=np.float64):
    a = np.broadcast_to(np.asarray(value, dtype=dtype), shape).copy()
    if dtype == np.float64 and (not np.isfinite(a).all() or (positive and np.any(a <= 0)) or
                               (nonnegative and np.any(a < 0))):
        raise ValueError(f"Invalid {name}")
    return np.ascontiguousarray(a)


@lru_cache(maxsize=1)
def _native_library_v104():
    compiler = shutil.which("g++")
    if not compiler: raise RuntimeError("Q7b native CPU route requires an OpenMP C++ compiler")
    directory = Path(tempfile.mkdtemp(prefix="braids_q7b_v104_"))
    library = directory / ("q7b.dll" if os.name == "nt" else "q7b.so")
    flags = ["-O3", "-fopenmp", "-shared", "-ffp-contract=off", "-std=c++11"]
    if os.name == "nt": flags += ["-static-libstdc++", "-static-libgcc"]
    build = subprocess.run([compiler, *flags, str(SOURCE), "-o", str(library)], cwd=directory,
                           capture_output=True, text=True, timeout=120)
    if build.returncode: raise RuntimeError(build.stdout + build.stderr)
    dependency = os.add_dll_directory(str(Path(compiler).parent)) if os.name == "nt" else None
    dll = ctypes.CDLL(str(library)); real=np.ctypeslib.ndpointer(np.float64,flags="C_CONTIGUOUS")
    integer=np.ctypeslib.ndpointer(np.int32,flags="C_CONTIGUOUS")
    diag=dll.braids_dynamic_diagnostics_v104
    diag.argtypes=[ctypes.c_int,ctypes.c_int,integer,real]+[real]*12+[integer]
    diag.restype=None
    relax=dll.braids_dynamic_roughness_relax_v104
    relax.argtypes=[ctypes.c_int,ctypes.c_int,real,ctypes.c_double]+[real]*7+[integer]
    relax.restype=None
    step=dll.braids_dynamic_step_v104
    step.argtypes=[ctypes.c_int]*4+[integer]*2+[ctypes.c_double]*3+[real]*5+[integer,real,integer]
    step.restype=None
    receipt={"source_sha256":hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
             "equations_sha256":hashlib.sha256(HEADER.read_bytes()).hexdigest(),
             "binary_sha256":hashlib.sha256(library.read_bytes()).hexdigest(),
             "compiler_sha256":hashlib.sha256(Path(compiler).read_bytes()).hexdigest(),"flags":flags}
    return diag,relax,step,dll,dependency,receipt


class NativeDynamicBedformRuntimeV104:
    backend="native_cpu_openmp"
    xp=np

    def __init__(self, methods=DynamicBedformMethodsV104(),
                 parameters=DynamicBedformParametersV104(), *, threads=4):
        if type(threads) is not int or not 1 <= threads <= 256:
            raise ValueError("Explicit CPU thread count in [1,256] required")
        self.resolved=resolve_dynamic_bedforms_v104(methods); self.methods=methods
        self.parameters=parameters; self.threads=threads; self.mode=_modes(methods); self.params=parameters.wire()
        self._diag,self._relax,self._step,self._dll,self._dependency,receipt=_native_library_v104()
        self.receipt=receipt
        self.identity=digest({"backend":self.backend,"methods":self.resolved.fingerprint,
                              "parameters":asdict(parameters),"runtime":receipt})

    def _diagnostics(self, shape, fields):
        nc=int(np.prod(shape)); out=np.empty((5,nc)); status=np.empty(nc,np.int32)
        order=(3,4,5,6,7,0,1,2,8,9,10)
        self._diag(nc,self.threads,self.mode,self.params,*(fields[i].ravel() for i in order),out,status)
        if np.any(status): raise ValueError(f"Nonphysical Q7b cells: {np.flatnonzero(status).tolist()}")
        return out

    def _roughness_relax(self,shape,current,target,dt_s):
        nc=int(np.prod(shape));out=np.empty((3,nc));status=np.empty(nc,np.int32)
        self._relax(nc,self.threads,self.params,float(dt_s),*(x.ravel() for x in current+target),out,status)
        if np.any(status):raise ValueError("Invalid Van Rijn 2007 roughness relaxation state")
        return [out[k].reshape(shape) for k in range(3)]

    def advance(self, height, equilibrium_height, length, depth, u, v, *, bedload_x=0.,
                bedload_y=0., ripple_roughness=0., mega_roughness=0., dune_roughness=0.,
                static_roughness_target=None, wet=None,
                dt_s, dx_m, dy_m, morfac=1., morphology_started=True,
                boundary=("open","open","open","open")):
        shape=np.asarray(height).shape
        if len(shape)!=2 or min(shape)<1: raise ValueError("Q7b requires a 2-D structured patch")
        ny,nx=shape; nc=nx*ny
        if not np.isfinite([dt_s,dx_m,dy_m,morfac]).all() or dt_s<=0 or dx_m<=0 or dy_m<=0 or morfac<0:
            raise ValueError("Invalid Q7b time/grid scale")
        names=("height","equilibrium_height","length","depth","u","v","bedload_x","bedload_y","ripple","mega")
        vals=(height,equilibrium_height,length,depth,u,v,bedload_x,bedload_y,
              ripple_roughness,mega_roughness,dune_roughness)
        names=names+("dune_roughness",)
        fields=[_field(x,shape,n,positive=n=="length",nonnegative=n in ("height","equilibrium_height","depth","ripple","mega","dune_roughness")) for x,n in zip(vals,names)]
        wet=_field(fields[3]>self.parameters.dry_depth_m if wet is None else wet,shape,"wet",dtype=np.int32)
        if np.any((wet!=0)&(wet!=1)): raise ValueError("Q7b wet mask must be binary")
        diag=self._diagnostics(shape,fields)
        effective_dt=dt_s*(morfac if self.methods.clock=="morphological" else 1.)
        if not morphology_started or (self.methods.clock=="morphological" and morfac==0): effective_dt=0.
        if self.methods.roughness=="van_rijn_2007_static":
            if not isinstance(static_roughness_target,Mapping) or set(static_roughness_target)!={"ripple","mega_ripple","dune"}:
                raise ValueError("Van Rijn 2007 static roughness requires all three Q7a target components")
            target=[_field(static_roughness_target[k],shape,f"target_{k}",nonnegative=True)
                    for k in ("ripple","mega_ripple","dune")]
            fields[8:11]=self._roughness_relax(shape,fields[8:11],target,dt_s)
        elif static_roughness_target is not None:
            raise ValueError("Static roughness target is only valid for Van Rijn 2007")
        if effective_dt and self.methods.relaxation != "equilibrium":
            diag[2]=np.where(diag[2]>0.,np.maximum(diag[2],effective_dt),0.)
        cfl=float(effective_dt*np.max(np.abs(diag[0])/dx_m+np.abs(diag[1])/dy_m)) if effective_dt else 0.
        if self.methods.advection and self.methods.cfl_control:
            nsub=max(1,int(math.ceil(cfl/self.parameters.cfl_target)))
        else:
            nsub=1
            if self.methods.advection and cfl>1.+1e-12:
                raise ValueError("Bedform advection CFL exceeds one while BdfCFL control is disabled")
        result=fields[0].copy().ravel(); status=np.zeros(nc,np.int32)
        if effective_dt:
            self._step(nx,ny,self.threads,nsub,self.mode,_boundary_codes(boundary),effective_dt,dx_m,dy_m,
                       fields[0].ravel(),fields[1].ravel(),diag[0],diag[1],diag[2],wet.ravel(),result,status)
            if np.any(status): raise ValueError(f"Unstable Q7b update cells: {np.flatnonzero(status).tolist()}")
        final_fields=list(fields); final_fields[0]=result.reshape(shape)
        final_diag=self._diagnostics(shape,final_fields)
        return {"height_m":result.reshape(shape),"equilibrium_height_m":fields[1],"length_m":fields[2],
                "celerity_x_m_s":diag[0].reshape(shape),"celerity_y_m_s":diag[1].reshape(shape),
                "relaxation_time_s":diag[2].reshape(shape),"dune_roughness_m":final_diag[3].reshape(shape),
                "roughness_m":final_diag[4].reshape(shape),"effective_dt_s":effective_dt,
                "ripple_roughness_m":fields[8],"mega_roughness_m":fields[9],
                "static_dune_roughness_m":fields[10],
                "hydraulic_dt_s":dt_s,
                "maximum_courant":cfl,"substeps":nsub,"wet":wet}


@dataclass(frozen=True)
class BedformFeedbackV104:
    accepted_update: int
    height_m: object
    length_m: object
    roughness_m: object
    token: str
    consumers: tuple[str,...]=FEEDBACK_CONSUMERS


class DynamicBedformOwnerV104:
    """Atomic accepted Q7b state; no backend switching or speculative publication."""
    def __init__(self,runtime,height,equilibrium,length,ripple,mega,area,*,dune_roughness=0.,time_s=0.,bedform_time_s=0.):
        self.runtime=runtime;self.xp=runtime.xp
        shape=host(height).shape
        if len(shape)!=2: raise ValueError("Q7b owner requires a structured patch")
        self.height=self.xp.asarray(_field(host(height),shape,"height",nonnegative=True))
        self.equilibrium=self.xp.asarray(_field(host(equilibrium),shape,"equilibrium",nonnegative=True))
        self.length=self.xp.asarray(_field(host(length),shape,"length",positive=True))
        self.ripple=self.xp.asarray(_field(host(ripple),shape,"ripple",nonnegative=True))
        self.mega=self.xp.asarray(_field(host(mega),shape,"mega",nonnegative=True))
        self.dune_component=self.xp.asarray(_field(host(dune_roughness),shape,"dune_roughness",nonnegative=True))
        self.area=self.xp.asarray(_field(host(area),shape,"area",positive=True))
        self.roughness=self._roughness(self.height,self.length)
        self.time_s=float(time_s);self.bedform_time_s=float(bedform_time_s);self.updates=0
        self.reference_integral=float(host((self.height*self.area).sum()));self.cumulative_change=0.
        self.pending=None;self.pending_sha256=None;self._seal()

    def _roughness(self,height,length):
        xp=self.xp;p=self.runtime.parameters
        if self.runtime.methods.roughness=="van_rijn_2007_static":dune=self.dune_component
        elif self.runtime.methods.roughness=="van_rijn_1984":
            dune=1.1*height*(1-xp.exp(-25*height/xp.maximum(length,1e-6)))
        else: dune=p.roughness_coefficient*xp.abs(height)**p.roughness_exponent
        return xp.sqrt(self.ripple*self.ripple+self.mega*self.mega+dune*dune)

    def _seal(self):
        self.state_sha256=array_sha(self.height,self.equilibrium,self.length,self.ripple,self.mega,
                                    self.dune_component,self.roughness,self.area)

    def validate(self):
        if self.state_sha256!=array_sha(self.height,self.equilibrium,self.length,self.ripple,self.mega,
                                        self.dune_component,self.roughness,self.area):
            raise ValueError("Accepted Q7b state was mutated")

    def propose(self,depth,u,v,*,bedload_x=0.,bedload_y=0.,wet=None,dt_s,dx_m,dy_m,
                morfac=1.,morphology_started=True,boundary=("open",)*4,static_roughness_target=None,
                equilibrium_height=None,length=None):
        self.validate()
        if self.pending is not None: raise ValueError("A Q7b trial is already pending")
        if self.updates>=20: raise ValueError("Q7b bounded owner supports at most 20 accepted updates")
        eq=self.equilibrium if equilibrium_height is None else equilibrium_height
        ln=self.length if length is None else length
        result=self.runtime.advance(self.height,eq,ln,depth,u,v,bedload_x=bedload_x,bedload_y=bedload_y,
            ripple_roughness=self.ripple,mega_roughness=self.mega,dune_roughness=self.dune_component,
            static_roughness_target=static_roughness_target,wet=wet,dt_s=dt_s,dx_m=dx_m,
            dy_m=dy_m,morfac=morfac,morphology_started=morphology_started,boundary=boundary)
        self.pending=result
        self.pending_sha256=array_sha(result["height_m"],result["equilibrium_height_m"],result["length_m"],
            result["ripple_roughness_m"],result["mega_roughness_m"],result["static_dune_roughness_m"],result["roughness_m"])
        return result

    def commit(self,*,cancelled=False):
        self.validate()
        if self.pending is None: raise ValueError("No Q7b trial")
        if cancelled: self.rollback();raise InterruptedError("Q7b update cancelled before publication")
        result=self.pending
        if self.pending_sha256!=array_sha(result["height_m"],result["equilibrium_height_m"],result["length_m"],
            result["ripple_roughness_m"],result["mega_roughness_m"],result["static_dune_roughness_m"],result["roughness_m"]):
            raise ValueError("Q7b trial was mutated")
        before=float(host((self.height*self.area).sum()))
        self.height=result["height_m"];self.equilibrium=result["equilibrium_height_m"]
        self.length=result["length_m"];self.roughness=result["roughness_m"]
        self.ripple=result["ripple_roughness_m"];self.mega=result["mega_roughness_m"]
        self.dune_component=result["static_dune_roughness_m"]
        self.time_s+=float(result["hydraulic_dt_s"])
        self.bedform_time_s+=float(result["effective_dt_s"]);self.updates+=1
        change=float(host((self.height*self.area).sum()))-before;self.cumulative_change+=change
        self.pending=None;self.pending_sha256=None;self._seal();return change

    def rollback(self): self.pending=None;self.pending_sha256=None

    def feedback(self):
        self.validate()
        token=digest({"scope":SCOPE,"runtime":self.runtime.identity,"update":self.updates,
                      "state":self.state_sha256})
        return BedformFeedbackV104(self.updates,self.height,self.length,self.roughness,token)

    def checkpoint(self):
        self.validate()
        if self.pending is not None: raise ValueError("Checkpoint requires committed Q7b state")
        data={"schema":104,"scope":SCOPE,"runtime":self.runtime.identity,"methods":self.runtime.resolved.fingerprint,
              "time_s":self.time_s,"bedform_time_s":self.bedform_time_s,"updates":self.updates,
              "reference_integral":self.reference_integral,"cumulative_change":self.cumulative_change,
              **{k:host(getattr(self,k)).tolist() for k in ("height","equilibrium","length","ripple","mega","dune_component","area")},
              "state_sha256":self.state_sha256}
        return {**data,"checkpoint_sha256":digest(data)}

    def restore(self,data):
        raw={k:v for k,v in data.items() if k!="checkpoint_sha256"}
        if digest(raw)!=data.get("checkpoint_sha256") or data.get("schema")!=104 or data.get("scope")!=SCOPE or data.get("runtime")!=self.runtime.identity or data.get("methods")!=self.runtime.resolved.fingerprint:
            raise ValueError("Q7b restart identity differs")
        candidate=DynamicBedformOwnerV104(self.runtime,data["height"],data["equilibrium"],data["length"],
            data["ripple"],data["mega"],data["area"],dune_roughness=data["dune_component"],
            time_s=data["time_s"],bedform_time_s=data["bedform_time_s"])
        candidate.updates=int(data["updates"]);candidate.reference_integral=float(data["reference_integral"])
        candidate.cumulative_change=float(data["cumulative_change"]);candidate._seal()
        if candidate.state_sha256!=data["state_sha256"] or not 0<=candidate.updates<=20:
            raise ValueError("Invalid Q7b restart state")
        self.__dict__.update(candidate.__dict__)


def compose_bedform_feedback_v104(feedback: BedformFeedbackV104, *, applied_tokens=()):
    if feedback.token in set(applied_tokens):
        raise ValueError("Accepted dynamic-bedform feedback would be applied twice")
    return {"hydraulic_roughness_m":feedback.roughness_m,
            "transport_roughness_m":feedback.roughness_m,
            "q5b_dune_length_m":feedback.length_m,"q5b_dune_height_m":feedback.height_m,
            "feedback_token":feedback.token,"consumers":feedback.consumers}


def resolve_main_channel_averaged_bed_level_v104(enabled: bool):
    if enabled:
        raise ValueError("MainChannelAveragedBedLevel is a D-Flow FM output; locked Delft3D4 FLOW has no consumer/writer")
    return None


def remap_dynamic_bedform_owner_v104(owner,target_runtime,offsets,indices,weights,target_area):
    owner.validate()
    if owner.pending is not None: raise ValueError("Q7b remap requires committed state")
    if owner.runtime.resolved.fingerprint!=target_runtime.resolved.fingerprint or owner.runtime.backend!=target_runtime.backend:
        raise ValueError("Q7b remap cannot change physics or backend")
    offsets=np.ascontiguousarray(offsets,np.int32);indices=np.ascontiguousarray(indices,np.int32)
    weights=np.ascontiguousarray(weights,float);area=np.ascontiguousarray(target_area,float)
    if area.ndim!=2: raise ValueError("Q7b target AMR patch must be two-dimensional")
    nc=area.size;old_area=host(owner.area).ravel()
    if offsets.shape!=(nc+1,) or offsets[0]!=0 or offsets[-1]!=len(indices) or np.any(np.diff(offsets)<=0) or weights.shape!=indices.shape or np.any(indices<0) or np.any(indices>=old_area.size) or not np.isfinite(weights).all() or np.any(weights<0) or np.any(area<=0):
        raise ValueError("Invalid Q7b conservative remap")
    row=np.repeat(np.arange(nc),np.diff(offsets))
    if not np.allclose(np.bincount(row,weights=weights),1.,atol=1e-13,rtol=0) or not np.allclose(np.bincount(indices,weights=weights*area.ravel()[row],minlength=old_area.size),old_area,atol=1e-12,rtol=0):
        raise ValueError("Q7b remap does not conserve area")
    def mapped(value): return np.add.reduceat(host(value).ravel()[indices]*weights,offsets[:-1])
    result=DynamicBedformOwnerV104(target_runtime,mapped(owner.height).reshape(area.shape),
        mapped(owner.equilibrium).reshape(area.shape),mapped(owner.length).reshape(area.shape),
        mapped(owner.ripple).reshape(area.shape),mapped(owner.mega).reshape(area.shape),area,
        dune_roughness=mapped(owner.dune_component).reshape(area.shape))
    result.time_s=owner.time_s;result.bedform_time_s=owner.bedform_time_s;result.updates=owner.updates
    result.reference_integral=owner.reference_integral;result.cumulative_change=owner.cumulative_change;result._seal()
    return result


__all__=["BedformFeedbackV104","DynamicBedformMethodsV104","DynamicBedformOwnerV104",
         "DynamicBedformParametersV104","NativeDynamicBedformRuntimeV104","ResolvedDynamicBedformsV104",
         "compose_bedform_feedback_v104","remap_dynamic_bedform_owner_v104",
         "resolve_dynamic_bedforms_v104","resolve_main_channel_averaged_bed_level_v104"]
