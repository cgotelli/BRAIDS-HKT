"""Additive v97 wet-active density projection primitive, not a solver.

Both matrix and right-hand side exclude exactly the same inactive faces.
Rank-one cells use the explicit saved minimum-norm policy; no diagonal
regularisation or CPU fallback is used. This does not qualify moving wet/dry
implicit integration or constitute another density/time/state owner.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np

from .densin_fwave_v85 import verify_locked_sources_v85

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/"solver/wave/braids_density_active_projection_v97_module.f90"
POLICY = "weighted_minimum_norm_active_set_v97"
RANK_TOL = 256.*np.finfo(float).eps
RESIDUAL_TOL = 2048.*np.finfo(float).eps
ABI = "braids-v97-native-density-active-projection-v1"
RANK_CODES = "-1=exact_zero_source_rank_not_evaluated;0=no_active_faces;1=normal_only;2=full_rank"


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _digest(*values):
    h = hashlib.sha256()
    for value in values:
        a = np.ascontiguousarray(value)
        h.update(a.dtype.str.encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _array(value, shape, name, *, integer=False):
    raw = np.asarray(value)
    if raw.shape != shape or raw.dtype.kind not in "fiub" or not np.isfinite(raw).all():
        raise ValueError(f"v97 {name} shape/type/finite identity differs")
    if integer and (np.any(raw != np.floor(raw)) or np.any(np.abs(raw)>2147483647)):
        raise ValueError(f"v97 {name} must be int32")
    a = np.array(raw, dtype=np.int32 if integer else np.float64, order="C", copy=True)
    a.setflags(write=False)
    return a


@dataclass(frozen=True)
class DensityProjectionGeometryV97:
    normal: np.ndarray
    length: np.ndarray
    offset: np.ndarray
    faces: np.ndarray
    signs: np.ndarray
    fingerprint: str

    @property
    def cell_count(self):
        return self.offset.size-1

    @property
    def face_count(self):
        return self.length.size


def _geometry_token(g):
    return _digest(g.normal,g.length,g.offset,g.faces,g.signs)


def resolve_density_projection_geometry_v97(normal, length, offset, faces, signs):
    raw = np.asarray(length)
    off = np.asarray(offset)
    fs = np.asarray(faces)
    if raw.ndim != 1 or raw.size<1 or off.ndim!=1 or off.size<2 or fs.ndim!=1:
        raise ValueError("v97 density geometry dimensions differ")
    nf,nc,nadj = raw.size,off.size-1,fs.size
    g = DensityProjectionGeometryV97(
        _array(normal,(2,nf),"normal"),_array(length,(nf,),"length"),
        _array(offset,(nc+1,),"offset",integer=True),
        _array(faces,(nadj,),"faces",integer=True),
        _array(signs,(nadj,),"signs",integer=True),"")
    if (np.any(g.length<=0) or
        np.any(np.abs(np.linalg.norm(g.normal,axis=0)-1)>2e-13) or
        g.offset[0]!=0 or g.offset[-1]!=nadj or np.any(np.diff(g.offset)<0) or
        np.any(g.faces<0) or np.any(g.faces>=nf) or
        np.any((g.signs!=-1)&(g.signs!=1))):
        raise ValueError("v97 density geometry connectivity/normal is invalid")
    return DensityProjectionGeometryV97(**{**g.__dict__,"fingerprint":_geometry_token(g)})


def validate_projection_inputs_v97(g,depth,acceleration,wet,active,*,policy,rank_tol,residual_tol):
    if type(g) is not DensityProjectionGeometryV97 or g.fingerprint!=_geometry_token(g):
        raise ValueError("v97 density geometry identity differs")
    if policy!=POLICY:
        raise ValueError("v97 minimum-norm density policy must be explicitly selected")
    for name,value in (("rank tolerance",rank_tol),("residual tolerance",residual_tol)):
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value) or not 0<value<1e-6:
            raise ValueError(f"v97 {name} differs")
    h = _array(depth,(g.cell_count,),"depth")
    a = _array(acceleration,(g.face_count,),"face acceleration")
    w = _array(wet,(g.cell_count,),"wet mask",integer=True)
    f = _array(active,(g.face_count,),"active face mask",integer=True)
    if np.any(h<0) or np.any((w!=0)&(w!=1)) or np.any((f!=0)&(f!=1)) or np.any((w==1)&(h<=0)):
        raise ValueError("v97 density depth/masks are invalid")
    slot_wet = np.repeat(w,np.diff(g.offset))
    if np.any((slot_wet==0)&(f[g.faces]!=0)):
        raise ValueError("v97 active density face touches an excluded dry cell")
    return h,a,w,f


def make_projection_result_v97(g,inputs,acceleration,rate,ranks,residual,*,policy,rank_tol,
                               residual_tol,runtime_sha,backend,threads):
    nc = g.cell_count
    values = (_array(acceleration,(2,nc),"cell acceleration"),
              _array(rate,(2,nc),"momentum rate"),
              _array(ranks,(nc,),"ranks",integer=True),
              _array(residual,(nc,),"projection residual"))
    result = dict(cell_acceleration_m_s2=values[0],momentum_rate_m2_s2=values[1],
        numerical_rank=values[2],scaled_projection_residual=values[3],
        numerical_rank_codes=RANK_CODES,
        policy=policy,rank_relative_tolerance=rank_tol,residual_relative_tolerance=residual_tol,
        geometry_sha256=g.fingerprint,inputs_sha256=_digest(*inputs),
        outputs_sha256=_digest(*values),native_runtime_sha256=runtime_sha,
        backend=backend,native_threads=threads,numerical_python_cell_face_loops=0,
        state_or_clock_publication_count=0,production_admission=False,
        moving_wet_dry_solver_qualified=False)
    result["receipt_sha256"] = _hash({k:v for k,v in result.items() if not isinstance(v,np.ndarray)})
    return result


def validate_projection_result_v97(result,g,depth,face_acceleration,wet,active,*,runtime_sha):
    """Authenticate a consumer handoff, including masks and actual outputs."""
    keys={"cell_acceleration_m_s2","momentum_rate_m2_s2","numerical_rank",
        "scaled_projection_residual","numerical_rank_codes","policy",
        "rank_relative_tolerance","residual_relative_tolerance","geometry_sha256",
        "inputs_sha256","outputs_sha256","native_runtime_sha256","backend",
        "native_threads","numerical_python_cell_face_loops","state_or_clock_publication_count",
        "production_admission","moving_wet_dry_solver_qualified","receipt_sha256"}
    if type(result) is not dict or set(result)!=keys:
        raise ValueError("v97 projection result type differs")
    if (result["backend"] not in ("native_cpu","actual_cuda") or
        result["numerical_rank_codes"]!=RANK_CODES or
        type(result["native_threads"]) is not int or not 1<=result["native_threads"]<=256 or
        result["numerical_python_cell_face_loops"]!=0 or result["state_or_clock_publication_count"]!=0 or
        result["production_admission"] is not False or result["moving_wet_dry_solver_qualified"] is not False):
        raise ValueError("v97 projection result semantic metadata differs")
    inputs=validate_projection_inputs_v97(g,depth,face_acceleration,wet,active,
        policy=result.get("policy"),rank_tol=result.get("rank_relative_tolerance"),
        residual_tol=result.get("residual_relative_tolerance"))
    arrays=tuple(result.get(k) for k in ("cell_acceleration_m_s2","momentum_rate_m2_s2",
                                       "numerical_rank","scaled_projection_residual"))
    if (result.get("receipt_sha256")!=_hash({k:v for k,v in result.items() if not isinstance(v,np.ndarray) and k!="receipt_sha256"}) or
        result.get("geometry_sha256")!=g.fingerprint or result.get("inputs_sha256")!=_digest(*inputs) or
        result.get("outputs_sha256")!=_digest(*arrays) or result.get("native_runtime_sha256")!=runtime_sha):
        raise ValueError("v97 projection result/input/runtime identity differs")
    if (arrays[0].shape!=(2,g.cell_count) or arrays[1].shape!=(2,g.cell_count) or
        arrays[2].shape!=(g.cell_count,) or arrays[3].shape!=(g.cell_count,) or
        not all(np.isfinite(a).all() for a in arrays) or
        np.any((arrays[2]<-1)|(arrays[2]>2)) or
        np.any(arrays[3]>result["residual_relative_tolerance"]) or
        not np.array_equal(arrays[1],inputs[0]*arrays[0])):
        raise ValueError("v97 projection result values differ")
    # Independent vector audit of the projection equations, not a second
    # physical application. No Python numerical cell/face loops or state write.
    nc=g.cell_count
    cell=np.repeat(np.arange(nc),np.diff(g.offset))
    face=g.faces
    chosen=(inputs[2][cell]!=0)&(inputs[3][face]!=0)
    c=cell[chosen]; f=face[chosen]; sign=g.signs[chosen]
    nx=sign*g.normal[0,f]; ny=sign*g.normal[1,f]
    weight=g.length[f]; directional=sign*inputs[1][f]
    def gather(value):
        return np.bincount(c,weights=value,minlength=nc)
    a=gather(weight*nx*nx); b=gather(weight*nx*ny); d=gather(weight*ny*ny)
    bx=gather(weight*nx*directional); by=gather(weight*ny*directional)
    count=np.bincount(c,minlength=nc)
    nonzero=gather((directional!=0).astype(float))
    ranks=np.zeros(nc,np.int32)
    ranks[(count>0)&(nonzero==0)]=-1
    solving=(count>0)&(nonzero>0)
    det=a*d-b*b; hi=.5*(a+d+np.hypot(a-d,2*b))
    lo=np.zeros(nc); np.divide(det,hi,out=lo,where=solving)
    ratio=np.zeros(nc); np.divide(lo,hi,out=ratio,where=solving)
    rtol=result["rank_relative_tolerance"]
    ambiguous=(ratio>.5*rtol)&(ratio<2*rtol)
    if np.any(solving&((hi<=0)|~np.isfinite(hi)|~np.isfinite(det)|(ratio< -rtol)|ambiguous)):
        raise ValueError("v97 projection result has invalid/ambiguous rank")
    one=solving&(ratio<2*rtol); two=solving&~one
    ranks[one]=1; ranks[two]=2
    expected=np.zeros((2,nc))
    expected[0,two]=(d[two]*bx[two]-b[two]*by[two])/det[two]
    expected[1,two]=(a[two]*by[two]-b[two]*bx[two])/det[two]
    denominator=(hi[one]-lo[one])*hi[one]
    expected[0,one]=((a[one]-lo[one])*bx[one]+b[one]*by[one])/denominator
    expected[1,one]=(b[one]*bx[one]+(d[one]-lo[one])*by[one])/denominator
    # A vector-scaled floating-point bound preserves the zero tangent while
    # allowing unavoidable CPU/CUDA arithmetic-order roundoff.
    scale=np.maximum(np.finfo(float).tiny,np.max(np.abs(expected)+np.abs(arrays[0]),axis=0))
    if (not np.array_equal(arrays[2],ranks) or
        np.any(np.abs(arrays[0]-expected)>512*np.finfo(float).eps*scale) or
        np.any(arrays[0][:,~solving]!=0)):
        raise ValueError("v97 projection result violates the active normal equations")
    rx=a*arrays[0][0]+b*arrays[0][1]-bx
    ry=b*arrays[0][0]+d*arrays[0][1]-by
    residual_scale=np.maximum(np.finfo(float).tiny,np.maximum(
        np.abs(bx)+np.abs(a*arrays[0][0])+np.abs(b*arrays[0][1]),
        np.abs(by)+np.abs(b*arrays[0][0])+np.abs(d*arrays[0][1])))
    computed=np.maximum(np.abs(rx),np.abs(ry))/residual_scale
    if np.any(computed>result["residual_relative_tolerance"]):
        raise ValueError("v97 projection result fails independent residual check")
    return result


class NativeDensityActiveProjectionRuntimeV97:
    def __init__(self,receipt,library,dependency,library_path):
        self.receipt,self.library,self.dependency = receipt,library,dependency
        self.library_path = Path(library_path)
        self.function = library.braids_density_active_projection_v97
        fa = np.ctypeslib.ndpointer(dtype=np.float64,flags="C_CONTIGUOUS")
        ia = np.ctypeslib.ndpointer(dtype=np.int32,flags="C_CONTIGUOUS")
        self.function.argtypes = [ctypes.c_int]*3+[fa]*4+[ia]*5+[
            ctypes.c_double,ctypes.c_double,ctypes.c_int,fa,fa,ia,fa,ia]
        self.function.restype = None

    def evaluate(self,geometry,depth,face_acceleration,wet,active,*,policy,
                 rank_tol=RANK_TOL,residual_tol=RESIDUAL_TOL,native_threads=4,cancelled=False):
        cancel = cancelled if callable(cancelled) else lambda: cancelled
        if not callable(cancelled) and type(cancelled) is not bool:
            raise ValueError("v97 cancellation must be bool/callable")
        if cancel():
            raise InterruptedError("v97 density projection cancelled before launch")
        if type(native_threads) is not int or not 1<=native_threads<=256:
            raise ValueError("v97 native thread count differs")
        if self.receipt.get("receipt_sha256")!=_hash({k:v for k,v in self.receipt.items() if k!="receipt_sha256"}):
            raise ValueError("v97 native runtime receipt differs")
        if (_sha(SOURCE)!=self.receipt["source_sha256"] or
            _sha(Path(__file__))!=self.receipt["wrapper_sha256"] or
            _sha(self.library_path)!=self.receipt["library_sha256"]):
            raise ValueError("v97 native source/library identity differs")
        inputs = validate_projection_inputs_v97(geometry,depth,face_acceleration,wet,active,
            policy=policy,rank_tol=rank_tol,residual_tol=residual_tol)
        g,nc,nf = geometry,geometry.cell_count,geometry.face_count
        cell = np.empty((2,nc)); rate=np.empty((2,nc)); rank=np.empty(nc,np.int32)
        residual=np.empty(nc); status=np.empty(nc,np.int32)
        self.function(nc,nf,len(g.faces),inputs[0],inputs[1],g.normal,g.length,g.offset,
            g.faces,g.signs,inputs[2],inputs[3],float(rank_tol),float(residual_tol),
            native_threads,cell,rate,rank,residual,status)
        if np.any(status):
            raise ValueError("v97 native density projection rejected rank/residual: "+str(status.tolist()))
        if cancel():
            raise InterruptedError("v97 density projection cancelled after launch")
        return make_projection_result_v97(g,inputs,cell,rate,rank,residual,policy=policy,
            rank_tol=rank_tol,residual_tol=residual_tol,runtime_sha=self.receipt["receipt_sha256"],
            backend="native_cpu",threads=native_threads)


def build_native_density_active_projection_v97(build_directory,*,compiler=None):
    locked=verify_locked_sources_v85(ROOT)
    chosen=compiler or shutil.which("gfortran")
    if chosen is None and os.name=="nt":
        p=Path(r"C:\Strawberry\c\bin\gfortran.exe")
        chosen=p if p.is_file() else None
    if chosen is None:
        raise ValueError("v97 native density projection requires gfortran")
    chosen=Path(chosen).resolve()
    flags=["-O3","-fopenmp","-shared","-ffree-line-length-none","-static-libgfortran","-static-libgcc"]
    root=Path(build_directory).resolve(); root.mkdir(parents=True,exist_ok=True)
    source_sha=_sha(SOURCE)
    build=Path(tempfile.mkdtemp(prefix="v97_density_",dir=root))
    name="density_active_v97.dll" if os.name=="nt" else "libdensity_active_v97.so"
    path=build/name; partial=build/(name+".partial")
    proc=subprocess.run([str(chosen),*flags,str(SOURCE),"-o",str(partial)],cwd=build,
                        capture_output=True,text=True,timeout=120)
    if proc.returncode or not partial.is_file():
        raise ValueError("v97 native build failed: "+proc.stdout+proc.stderr)
    if source_sha!=_sha(SOURCE):
        raise ValueError("v97 native source changed during compilation")
    os.replace(partial,path)
    version=subprocess.run([str(chosen),"--version"],capture_output=True,text=True,check=True,timeout=20).stdout.splitlines()[0]
    receipt=dict(abi=ABI,source_sha256=source_sha,wrapper_sha256=_sha(Path(__file__)),library_sha256=_sha(path),
        compiler_sha256=_sha(chosen),compiler_identity=version,compile_flags=flags,
        locked_source_sha256=locked,thread_policy="per_region_num_threads_no_nested_team",
        numerical_python_cell_face_loops=0,production_admission=False)
    receipt["receipt_sha256"]=_hash(receipt)
    dependency=os.add_dll_directory(str(chosen.parent)) if os.name=="nt" and hasattr(os,"add_dll_directory") else None
    return NativeDensityActiveProjectionRuntimeV97(receipt,ctypes.CDLL(str(path)),dependency,path)
