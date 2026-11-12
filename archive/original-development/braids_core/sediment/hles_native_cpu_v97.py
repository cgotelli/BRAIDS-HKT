"""Pure OpenMP peer of the frozen v90 HLES candidate, with v90 ownership.

No numerical Python cell/face loops and no second filter, closure or clock
owner.  The existing v90 state/receipt/checkpoint layer remains authoritative.
The optional begin helpers return that legacy trial plus native evidence;
they do not silently replace its backend/method identity.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np

from . import hles_producer_v90 as v90
from .settling_exchange_native_cpu_v97 import _digest, _hash, _poll, _sha

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/"solver/wave/braids_hles_candidate_v97_module.f90"
ABI_ID="braids-v97-native-v90-hles-candidate-v1"


@dataclass(frozen=True)
class NativeHlesCandidateV97:
    values: tuple
    native_runtime_sha256: str
    parent_state_sha256: str
    execution_fingerprint: str
    inputs_sha256: str
    result_sha256: str
    native_threads: int
    filter_owner_count: int=1
    reconstruction_owner_count: int=1
    protke_owner_count: int=1
    detvic_owner_count: int=1
    python_numerical_cell_face_loop_count: int=0
    state_or_clock_publication_count: int=0
    native_candidate_launch_count: int=1


class NativeHlesRuntimeV97:
    def __init__(self,receipt,library,dependency,library_path):
        self.receipt=receipt; self.library=library; self.dependency=dependency
        self.library_path=Path(library_path).resolve()
        self._function=library.braids_hles_candidate_v97
        fa=np.ctypeslib.ndpointer(dtype=np.float64,flags="C_CONTIGUOUS")
        ia=np.ctypeslib.ndpointer(dtype=np.int32,flags="C_CONTIGUOUS")
        self._function.argtypes=[ctypes.c_int]*4+[ia]*5+[fa]*6+[ia]+[fa]*2+[ia]+[
            ctypes.c_double]*9+[ctypes.c_int]*2+[fa]*4+[ia]*3+[fa]*5+[
            ctypes.POINTER(ctypes.c_int)]
        self._function.restype=None

    def evaluate(self,execution,parent,arrays,closure_interval_s,*,native_threads=4,cancelled=False):
        poll=_poll(cancelled)
        if poll(): raise InterruptedError("native HLES cancelled before launch")
        if type(native_threads) is not int or not 1<=native_threads<=256:
            raise ValueError("native HLES thread count must be 1..256")
        if (self.receipt.get("receipt_sha256")!=_hash({k:v for k,v in self.receipt.items() if k!="receipt_sha256"}) or
            _sha(SOURCE)!=self.receipt["source_sha256"] or
            _sha(Path(__file__))!=self.receipt["wrapper_sha256"] or
            _sha(self.library_path)!=self.receipt["library_sha256"]):
            raise ValueError("native HLES source/runtime identity differs")
        v90.execution_document_v90(execution)
        v90.validate_hles_producer_state_v90(parent,execution)
        if execution.backend!=v90.BACKEND_CPU: raise ValueError("native HLES requires explicit CPU execution")
        if not isinstance(arrays,tuple) or len(arrays)!=5: raise ValueError("native HLES input inventory differs")
        mean,active_face,depth,chezy,active_cell=v90._input_arrays(execution,
            accepted_face_normal_velocity_m_s=arrays[0],active_face=arrays[1],
            closure_depth_m=arrays[2],closure_chezy_mhalf_s=arrays[3],active_cell=arrays[4])
        interval=v90._finite(closure_interval_s,"closure interval",positive=True)
        coefficient=math.exp(-interval/(60.0*execution.relaxation_time_min)) if execution.relaxation_time_min>0 else 0.
        base=execution.closure_execution.base_execution.base_execution
        nc,nf=execution.cell_count,execution.face_count
        owner=v90._array(base.owner,(nf,),"owner",dtype=np.int32)
        neighbour=v90._array(base.neighbour,(nf,),"neighbour",dtype=np.int32)
        offset=v90._array(base.cell_face_offset,(nc+1,),"offset",dtype=np.int32)
        nadj=int(offset[-1])
        faces=v90._array(base.cell_faces,(nadj,),"faces",dtype=np.int32)
        signs=v90._array(base.cell_face_sign,(nadj,),"signs",dtype=np.int32)
        normal=v90._array(base.normal,(2,nf),"normal")
        length=v90._array(base.face_length_m,(nf,),"length",nonnegative=True)
        distance=v90._array(base.centre_distance_m,(nf,),"distance",nonnegative=True)
        area=v90._array(base.cell_area_m2,(nc,),"area",nonnegative=True)
        if (np.any(owner<0) or np.any(owner>=nc) or np.any(neighbour<-1) or np.any(neighbour>=nc) or
            offset[0]!=0 or np.any(np.diff(offset)<0) or np.any(faces<0) or np.any(faces>=nf) or
            np.any((signs!=-1)&(signs!=1)) or np.any(length<=0) or np.any(distance<=0) or np.any(area<=0)):
            raise ValueError("native HLES topology differs")
        filtered=np.empty(nf); fluct=np.empty(nf); mean_cell=np.empty((2,nc)); fluct_cell=np.empty((2,nc))
        complete=np.empty(nc,np.int32); boundary=np.empty(nc,np.int32); detvic_active=np.empty(nc,np.int32)
        ptke=np.empty(nc); hles=np.empty(nc); elder=np.empty(nc); vic=np.empty(nc); dic=np.empty(nc)
        closure=execution.closure_execution; status=ctypes.c_int()
        self._function(nc,nf,nadj,native_threads,owner,neighbour,offset,faces,signs,
            normal,length,distance,area,parent.accepted_filtered_face_normal_velocity_m_s,mean,
            active_face,depth,chezy,active_cell,coefficient,closure.gamma,closure.sigmat,closure.flp,
            closure.vicmol_m2_s,closure.dicmol_m2_s,closure.von_karman,closure.gravity_m_s2,
            base.dry_tolerance_m,int(closure.hles_enabled),int(closure.elder_enabled),
            filtered,fluct,mean_cell,fluct_cell,complete,boundary,detvic_active,ptke,hles,elder,vic,dic,
            ctypes.byref(status))
        outputs=(filtered,fluct,mean_cell,fluct_cell,complete,boundary,detvic_active,ptke,hles,elder,vic,dic)
        if status.value or not all(np.isfinite(a).all() for a in outputs): raise ValueError("native HLES candidate rejected fields")
        if poll(): raise InterruptedError("native HLES cancelled after launch")
        input_sha=_hash(dict(execution=execution.fingerprint,parent=parent.state_sha256,
            fields=_digest(mean,active_face,depth,chezy,active_cell),interval=interval))
        result_sha=_hash(dict(runtime=self.receipt["receipt_sha256"],inputs=input_sha,
            coefficient=coefficient,outputs=_digest(*outputs),native_threads=native_threads))
        for a in outputs: a.setflags(write=False)
        return NativeHlesCandidateV97((coefficient,*outputs),self.receipt["receipt_sha256"],
            parent.state_sha256,execution.fingerprint,input_sha,result_sha,native_threads)

    def begin_trial(self,execution,parent,*,accepted_step_receipt,hles_input_receipt,current_handoff,
                    accepted_face_normal_velocity_m_s,active_face,closure_depth_m,
                    closure_chezy_mhalf_s,active_cell,native_threads=4,cancelled=False):
        v90.validate_hles_producer_state_v90(parent,execution)
        if not parent.primed or parent.accepted_updates>=v90.MAX_ACCEPTED_UPDATES:
            raise ValueError("native HLES trial requires primed bounded parent")
        arrays=v90._input_arrays(execution,accepted_face_normal_velocity_m_s=accepted_face_normal_velocity_m_s,
            active_face=active_face,closure_depth_m=closure_depth_m,closure_chezy_mhalf_s=closure_chezy_mhalf_s,
            active_cell=active_cell)
        dt,interval=v90._validate_input_receipt(execution,parent,accepted_step_receipt,hles_input_receipt,arrays,current_handoff)
        result=self.evaluate(execution,parent,arrays,interval,native_threads=native_threads,cancelled=cancelled)
        names=("coefficient","filtered","fluctuation","mean_cell","fluct_cell","complete","boundary",
               "detvic_active","ptke","hles","elder","dynamic_vicuv","dynamic_dicuv")
        trial=v90._wrap_trial(execution,parent,accepted_step_receipt,hles_input_receipt,
            current_handoff=current_handoff,dt_s=dt,backend=v90.BACKEND_CPU,
            **dict(zip(names,result.values)))
        return trial,result

    def begin_prime(self,execution,parent,*,prime_receipt,closure_interval_s,
                    accepted_face_normal_velocity_m_s,active_face,closure_depth_m,
                    closure_chezy_mhalf_s,active_cell,native_threads=4,cancelled=False):
        v90.validate_hles_producer_state_v90(parent,execution)
        if parent.primed or parent.accepted_updates!=0 or parent.time_s!=0:
            raise ValueError("native HLES prime requires cold zero-time parent")
        arrays=v90._input_arrays(execution,accepted_face_normal_velocity_m_s=accepted_face_normal_velocity_m_s,
            active_face=active_face,closure_depth_m=closure_depth_m,closure_chezy_mhalf_s=closure_chezy_mhalf_s,
            active_cell=active_cell)
        interval=v90._finite(closure_interval_s,"prime closure interval",positive=True)
        v90._validate_prime_receipt(execution,parent,prime_receipt,arrays,interval)
        result=self.evaluate(execution,parent,arrays,interval,native_threads=native_threads,cancelled=cancelled)
        return v90._wrap_prime_trial(execution,parent,prime_receipt,result.values,v90.BACKEND_CPU),result


def build_native_hles_runtime_v97(build_directory,*,compiler=None):
    locked=v90.verify_locked_sources_v90(ROOT)
    selected=compiler or shutil.which("gfortran")
    if selected is None and os.name=="nt":
        fallback=Path(r"C:\Strawberry\c\bin\gfortran.exe")
        selected=fallback if fallback.is_file() else None
    if selected is None: raise ValueError("native HLES requires gfortran")
    selected=Path(selected).resolve()
    flags=["-O3","-fopenmp","-shared","-ffree-line-length-none","-static-libgfortran","-static-libgcc"]
    source_sha=_sha(SOURCE); wrapper_sha=_sha(Path(__file__))
    root=Path(build_directory).resolve(); root.mkdir(parents=True,exist_ok=True)
    build=Path(tempfile.mkdtemp(prefix="v97_hles_",dir=root))
    name="braids_hles_v97.dll" if os.name=="nt" else "libbraids_hles_v97.so"
    path=build/name; partial=build/(name+".partial")
    process=subprocess.run([str(selected),*flags,str(SOURCE),"-o",str(partial)],
                           cwd=build,capture_output=True,text=True,timeout=120)
    if process.returncode or not partial.is_file(): raise ValueError("native HLES build failed: "+process.stdout+process.stderr)
    if source_sha!=_sha(SOURCE) or wrapper_sha!=_sha(Path(__file__)): raise ValueError("native HLES source changed during build")
    os.replace(partial,path)
    identity=subprocess.run([str(selected),"--version"],capture_output=True,text=True,check=True,timeout=20).stdout.splitlines()[0]
    receipt=dict(abi=ABI_ID,source_sha256=source_sha,wrapper_sha256=wrapper_sha,library_sha256=_sha(path),
        compiler_identity=identity,compiler_sha256=_sha(selected),compile_flags=flags,locked_source_sha256=locked,
        mapping_id=v90.MAPPING_ID,producer_cadence=v90.PRODUCER_CADENCE,
        physical_owner_counts=dict(filter=1,reconstruction=1,protke=1,detvic=1),
        openmp_face_cell_loops=True,nested_teams=False,state_or_clock_publication_count=0,
        historical_v90_state_and_receipt_identity_preserved=True,production_admission=False)
    receipt["receipt_sha256"]=_hash(receipt)
    dependency=os.add_dll_directory(str(selected.parent)) if os.name=="nt" and hasattr(os,"add_dll_directory") else None
    return NativeHlesRuntimeV97(receipt,ctypes.CDLL(str(path)),dependency,path)


__all__=["NativeHlesCandidateV97","NativeHlesRuntimeV97","build_native_hles_runtime_v97"]
