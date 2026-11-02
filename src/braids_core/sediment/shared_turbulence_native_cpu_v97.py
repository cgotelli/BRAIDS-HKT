"""Additive native wet-active v91 consumer for the v97 implicit owner.

The cap is the unchanged v96/v91 cap, evaluated once. Only the momentum
consumer gains explicit authenticated masks; it owns no clock or state.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np

from .shared_turbulence_native_cpu_v96 import (
    build_native_shared_turbulence_runtime_v96,PreparedNativeTurbulenceV96,
    _preparation_token as _base_preparation_token,_hash,_sha,_threads,_readonly,
)
from .shared_turbulence_consumers_v91 import execution_document_v91,_cap_token,verify_locked_sources_v91
from .density_active_projection_v97 import _digest

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/"solver/wave/braids_shared_turbulence_v97_module.f90"
ABI="braids-v97-native-wet-active-v91-momentum-v1"


@dataclass(frozen=True)
class PreparedNativeTurbulenceV97:
    base_preparation: PreparedNativeTurbulenceV96
    native_runtime_sha256: str
    preparation_sha256: str

    @property
    def execution(self): return self.base_preparation.execution
    @property
    def cap_receipt(self): return self.base_preparation.cap_receipt
    @property
    def native_threads(self): return self.base_preparation.native_threads


def _preparation_token(p):
    return _hash(dict(abi=ABI,base_preparation=p.base_preparation.preparation_sha256,
                      runtime=p.native_runtime_sha256))


@dataclass(frozen=True)
class NativeMomentumRateV97:
    momentum_rate_m2_s2: np.ndarray
    face_flux_m4_s2: np.ndarray
    cap_receipt: object
    parent_hles_state_sha256: str
    native_runtime_sha256: str
    native_threads: int
    wet_cell_mask_sha256: str
    active_wet_wet_face_mask_sha256: str
    midpoint_q_sha256: str
    result_sha256: str
    momentum_rate_evaluation_count: int=1
    depth_source_evaluation_count: int=0
    numerical_python_cell_face_loops: int=0
    state_or_clock_publication_count: int=0


def _result_token(r):
    return _hash(dict(abi=ABI,cap=r.cap_receipt.receipt_sha256,parent=r.parent_hles_state_sha256,
        runtime=r.native_runtime_sha256,threads=r.native_threads,wet=r.wet_cell_mask_sha256,
        faces=r.active_wet_wet_face_mask_sha256,q=r.midpoint_q_sha256,
        outputs=_digest(r.momentum_rate_m2_s2,r.face_flux_m4_s2),
        counts=[r.momentum_rate_evaluation_count,r.depth_source_evaluation_count,
                r.numerical_python_cell_face_loops,r.state_or_clock_publication_count]))


def _mask(value,size,name):
    a=np.asarray(value)
    if a.shape!=(size,) or a.dtype.kind not in "fiub" or not np.isfinite(a).all() or np.any((a!=0)&(a!=1)):
        raise ValueError(f"v97 momentum {name} must be a boolean mask")
    return np.ascontiguousarray(a,dtype=np.int32)


def validate_masked_momentum_inputs_v97(execution,q,wet,active):
    execution_document_v91(execution)
    base=execution.base_execution.base_execution
    nc,nf=execution.cell_count,execution.face_count
    state=np.asarray(q)
    if state.shape!=(3,nc) or state.dtype.kind not in "fiu" or not np.isfinite(state).all() or np.any(state[0]<0):
        raise ValueError("v97 momentum midpoint q is invalid")
    state=np.ascontiguousarray(state,dtype=np.float64)
    wet=_mask(wet,nc,"wet cells"); active=_mask(active,nf,"active faces")
    expected=(base.neighbour>=0)&(wet[base.owner]!=0)&(wet[np.maximum(base.neighbour,0)]!=0)
    if not np.array_equal(active!=0,expected):
        raise ValueError("v97 momentum face mask is not the exact wet-wet topology image")
    if np.any((wet!=0)&(state[0]<=0)):
        raise ValueError("v97 momentum frozen wet cell must retain positive current depth")
    return state,wet,active


class NativeSharedTurbulenceRuntimeV97:
    def __init__(self,receipt,library,dependency,path,cap_runtime):
        self.receipt,self.library,self.dependency=receipt,library,dependency
        self.library_path=Path(path); self.cap_runtime=cap_runtime
        self.function=library.braids_v97_masked_momentum
        fa=np.ctypeslib.ndpointer(dtype=np.float64,flags="C_CONTIGUOUS")
        ia=np.ctypeslib.ndpointer(dtype=np.int32,flags="C_CONTIGUOUS")
        self.function.argtypes=[ctypes.c_int]*3+[fa]*2+[ia]*2+[fa]*3+[ia]*5+[
            ctypes.c_int,fa,fa,ctypes.POINTER(ctypes.c_int)]
        self.function.restype=None

    def _validate_runtime(self):
        if (self.receipt.get("receipt_sha256")!=_hash({k:v for k,v in self.receipt.items() if k!="receipt_sha256"}) or
            _sha(SOURCE)!=self.receipt["source_sha256"] or
            _sha(Path(__file__))!=self.receipt["wrapper_sha256"] or
            _sha(self.library_path)!=self.receipt["library_sha256"] or
            self.cap_runtime.receipt["receipt_sha256"]!=self.receipt["cap_runtime_sha256"]):
            raise ValueError("v97 momentum runtime/source/cap identity differs")
        self.cap_runtime._validate_runtime()

    def prepare(self,execution,parent_hles,handoff,*,dt_s,native_threads=4):
        self._validate_runtime()
        base=self.cap_runtime.prepare(execution,parent_hles,handoff,dt_s=dt_s,native_threads=native_threads)
        p=PreparedNativeTurbulenceV97(base,self.receipt["receipt_sha256"],"")
        return PreparedNativeTurbulenceV97(base,p.native_runtime_sha256,_preparation_token(p))

    def evaluate(self,prepared,q,wet_cell_mask,active_wet_wet_face_mask,*,native_threads=None,cancelled=False):
        cancel=cancelled if callable(cancelled) else lambda:cancelled
        if not callable(cancelled) and type(cancelled) is not bool:
            raise ValueError("v97 momentum cancellation must be bool/callable")
        if cancel(): raise InterruptedError("v97 momentum cancelled before launch")
        self._validate_runtime()
        if (type(prepared) is not PreparedNativeTurbulenceV97 or
            prepared.preparation_sha256!=_preparation_token(prepared) or
            prepared.native_runtime_sha256!=self.receipt["receipt_sha256"] or
            prepared.base_preparation.preparation_sha256!=_base_preparation_token(prepared.base_preparation) or
            prepared.cap_receipt.receipt_sha256!=_cap_token(prepared.cap_receipt)):
            raise ValueError("v97 momentum preparation/cap identity differs")
        threads=_threads(prepared.native_threads if native_threads is None else native_threads)
        state,wet,active=validate_masked_momentum_inputs_v97(prepared.execution,q,wet_cell_mask,active_wet_wet_face_mask)
        ex=prepared.execution; base=ex.base_execution.base_execution; nc,nf=ex.cell_count,ex.face_count
        ints=[np.ascontiguousarray(v,dtype=np.int32) for v in
            (base.owner,base.neighbour,base.cell_face_offset,base.cell_faces,base.cell_face_sign)]
        rate=np.empty((3,nc)); flux=np.empty((2,nf)); status=ctypes.c_int()
        self.function(nc,nf,len(ints[3]),state,prepared.cap_receipt.capped_total_vicuv_m2_s,
            ints[0],ints[1],base.face_length_m,base.centre_distance_m,base.cell_area_m2,
            ints[2],ints[3],ints[4],wet,active,threads,flux,rate,ctypes.byref(status))
        if status.value or not np.isfinite(rate).all() or not np.isfinite(flux).all():
            raise ValueError("v97 masked momentum kernel rejected input")
        if cancel(): raise InterruptedError("v97 momentum cancelled after launch")
        r=NativeMomentumRateV97(_readonly(rate),_readonly(flux),prepared.cap_receipt,
            prepared.cap_receipt.parent_hles_state_sha256,self.receipt["receipt_sha256"],threads,
            _digest(wet),_digest(active),_digest(state),"")
        return NativeMomentumRateV97(**{**r.__dict__,"result_sha256":_result_token(r)})


def build_native_shared_turbulence_runtime_v97(build_directory,*,compiler=None):
    locked=verify_locked_sources_v91(ROOT)
    cap_runtime=build_native_shared_turbulence_runtime_v96(Path(build_directory)/"cap",compiler=compiler)
    chosen=compiler or shutil.which("gfortran")
    if chosen is None and os.name=="nt": chosen=Path(r"C:\Strawberry\c\bin\gfortran.exe")
    if chosen is None or not Path(chosen).is_file(): raise ValueError("v97 momentum requires gfortran")
    chosen=Path(chosen).resolve()
    flags=["-O3","-fopenmp","-shared","-ffree-line-length-none","-static-libgfortran","-static-libgcc"]
    root=Path(build_directory).resolve(); root.mkdir(parents=True,exist_ok=True)
    source_sha=_sha(SOURCE)
    build=Path(tempfile.mkdtemp(prefix="v97_momentum_",dir=root))
    name="momentum_wet_active_v97.dll" if os.name=="nt" else "libmomentum_wet_active_v97.so"
    path=build/name; partial=build/(name+".partial")
    proc=subprocess.run([str(chosen),*flags,str(SOURCE),"-o",str(partial)],cwd=build,
                        capture_output=True,text=True,timeout=120)
    if proc.returncode or not partial.is_file(): raise ValueError("v97 momentum build failed: "+proc.stdout+proc.stderr)
    if source_sha!=_sha(SOURCE): raise ValueError("v97 momentum source changed during compilation")
    os.replace(partial,path)
    receipt=dict(abi=ABI,source_sha256=source_sha,wrapper_sha256=_sha(Path(__file__)),library_sha256=_sha(path),
        compiler_sha256=_sha(chosen),compile_flags=flags,locked_source_sha256=locked,
        cap_runtime_sha256=cap_runtime.receipt["receipt_sha256"],cap_application_count=1,
        numerical_python_cell_face_loops=0,thread_policy="per_region_num_threads_no_nested_team",
        production_admission=False)
    receipt["receipt_sha256"]=_hash(receipt)
    dep=os.add_dll_directory(str(chosen.parent)) if os.name=="nt" and hasattr(os,"add_dll_directory") else None
    return NativeSharedTurbulenceRuntimeV97(receipt,ctypes.CDLL(str(path)),dep,path,cap_runtime)
