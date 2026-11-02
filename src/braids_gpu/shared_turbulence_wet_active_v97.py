"""Resident actual-CUDA wet-active v91 momentum primitive.

Cap/time/state ownership remains with the coupled caller. This component
adds no accepted-state publication and never chooses a CPU fallback.
"""
from pathlib import Path
import numpy as np

from braids_core.sediment.shared_turbulence_consumers_v91 import execution_document_v91
from braids_core.sediment.hles_producer_v90 import BACKEND_GPU
from braids_core.sediment.shared_turbulence_native_cpu_v96 import _hash,_sha

ROOT=Path(__file__).resolve().parents[1]
CUDA_SOURCE=Path(__file__).with_suffix(".cu")
OPTIONS=("--std=c++11","--fmad=false","--prec-div=true","--prec-sqrt=true","--ftz=false")


class ResidentWetActiveMomentumOperatorV97:
    def __init__(self,execution):
        execution_document_v91(execution)
        if execution.backend!=BACKEND_GPU:
            raise ValueError("v97 momentum requires explicit implicit GPU backend")
        import cupy as cp
        self.cp=cp; self.execution=execution; self.device_id=int(cp.cuda.Device().id)
        self.nc,self.nf=execution.cell_count,execution.face_count
        base=execution.base_execution.base_execution
        self.host_geometry=tuple(base.__getattribute__(name) for name in (
            "owner","neighbour","face_length_m","centre_distance_m","cell_area_m2",
            "cell_face_offset","cell_faces","cell_face_sign"))
        self.geometry=tuple(cp.asarray(v) for v in self.host_geometry)
        source=CUDA_SOURCE.read_text(encoding="utf-8")
        names=("v97_momentum_face_flux","v97_momentum_gather")
        self.module=cp.RawModule(code=source,options=OPTIONS,name_expressions=names)
        self.kernels={name:self.module.get_function(name) for name in names}
        self.receipt=dict(backend=BACKEND_GPU,actual_cuda=True,automatic_cpu_fallback=False,
            execution_sha256=execution.fingerprint,source_sha256=_sha(CUDA_SOURCE),
            wrapper_sha256=_sha(Path(__file__)),cuda_options=list(OPTIONS),
            device_id=self.device_id,cap_owner_count=0,state_or_clock_publication_count=0,
            numerical_python_cell_face_loops=0,production_admission=False)
        self.receipt["receipt_sha256"]=_hash(self.receipt)

    def evaluate(self,q,capped_vicuv,wet_cell_mask,active_wet_wet_face_mask,*,cancelled=False):
        cp=self.cp
        cancel=cancelled if callable(cancelled) else lambda:cancelled
        if not callable(cancelled) and type(cancelled) is not bool:
            raise ValueError("v97 CUDA momentum cancellation must be bool/callable")
        if cancel(): raise InterruptedError("v97 CUDA momentum cancelled before launch")
        if (self.receipt.get("receipt_sha256")!=_hash({k:v for k,v in self.receipt.items() if k!="receipt_sha256"}) or
            _sha(CUDA_SOURCE)!=self.receipt["source_sha256"] or
            _sha(Path(__file__))!=self.receipt["wrapper_sha256"]):
            raise ValueError("v97 CUDA momentum runtime/source identity differs")
        execution_document_v91(self.execution)
        for value,shape,name in ((q,(3,self.nc),"q"),(capped_vicuv,(self.nc,),"vicuv")):
            if (not isinstance(value,cp.ndarray) or value.shape!=shape or value.dtype!=cp.float64 or
                not value.flags.c_contiguous or int(value.device.id)!=self.device_id or
                not bool(cp.isfinite(value).all().item())):
                raise ValueError(f"v97 CUDA momentum {name} must be resident finite FP64")
        wet,active=wet_cell_mask,active_wet_wet_face_mask
        for value,size in ((wet,self.nc),(active,self.nf)):
            if (not isinstance(value,cp.ndarray) or value.shape!=(size,) or
                value.dtype not in (cp.int32,cp.bool_) or int(value.device.id)!=self.device_id or
                not value.flags.c_contiguous or bool(cp.any((value!=0)&(value!=1)).item())):
                raise ValueError("v97 CUDA momentum masks must be resident boolean/int32")
        wet=wet.astype(cp.int32,copy=False); active=active.astype(cp.int32,copy=False)
        owner,neighbour,length,distance,area,offset,faces,signs=self.geometry
        for actual,expected in zip(self.geometry,self.host_geometry):
            if not bool(cp.array_equal(actual,cp.asarray(expected)).item()):
                raise ValueError("v97 CUDA resident momentum geometry differs")
        expected=(neighbour>=0)&(wet[owner]!=0)&(wet[cp.maximum(neighbour,0)]!=0)
        if not bool(cp.array_equal(active!=0,expected).item()):
            raise ValueError("v97 CUDA momentum face mask differs from wet-wet topology")
        if bool(cp.any(q[0]<0).item()) or bool(cp.any((wet!=0)&(q[0]<=0)).item()) or bool(cp.any(capped_vicuv<0).item()):
            raise ValueError("v97 CUDA momentum depth/vicuv is invalid")
        flux=cp.empty((2,self.nf)); rate=cp.empty((3,self.nc))
        fs=cp.empty(self.nf,cp.int32); cs=cp.empty(self.nc,cp.int32)
        self.kernels["v97_momentum_face_flux"](((self.nf+127)//128,),(128,),(
            np.int64(self.nc),np.int64(self.nf),q,capped_vicuv,owner,neighbour,
            length,distance,wet,active,flux,fs))
        self.kernels["v97_momentum_gather"](((self.nc+127)//128,),(128,),(
            np.int64(self.nc),np.int64(self.nf),area,offset,faces,signs,wet,flux,rate,cs))
        if bool(cp.any(fs).item()) or bool(cp.any(cs).item()):
            raise ValueError("v97 CUDA momentum kernel rejected input")
        if cancel(): raise InterruptedError("v97 CUDA momentum cancelled after launch")
        return dict(momentum_rate_m2_s2=rate,face_flux_m4_s2=flux,
            momentum_rate_evaluation_count=1,depth_source_evaluation_count=0,
            state_or_clock_publication_count=0,numerical_python_cell_face_loops=0,
            native_runtime_sha256=self.receipt["receipt_sha256"],backend="actual_cuda")
