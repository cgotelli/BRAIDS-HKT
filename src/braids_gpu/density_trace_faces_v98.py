"""Resident density face augmentation only; the hydraulic owner projects once."""
from pathlib import Path

import numpy as np

from braids_core.sediment.density_active_projection_v97 import _hash,_sha
from braids_core.sediment.density_trace_projection_v98 import validate_density_trace_inputs_v98
from braids_core.sediment.density_trace_faces_v98 import make_density_trace_faces_result_v98
from .density_trace_projection_v98 import ResidentDensityTraceProjectionV98

ROOT=Path(__file__).resolve().parents[1]


class ResidentDensityTraceFacesV98:
    def __init__(self,geometry,*,backend):
        self.base=ResidentDensityTraceProjectionV98(geometry,backend=backend)
        self.cp,self.geometry,self.device_id=self.base.cp,geometry,self.base.device_id
        self.receipt=dict(base_runtime_sha256=self.base.receipt["receipt_sha256"],
            wrapper_sha256=_sha(Path(__file__)),core_wrapper_sha256=_sha(ROOT/"braids_core/sediment/density_trace_faces_v98.py"),
            backend=backend,actual_cuda=True,automatic_cpu_fallback=False,
            density_projection_evaluation_count=0,production_admission=False)
        self.receipt["receipt_sha256"]=_hash(self.receipt)

    def _validate_runtime(self):
        if (self.receipt["receipt_sha256"]!=_hash({k:v for k,v in self.receipt.items() if k!="receipt_sha256"}) or
            _sha(Path(__file__))!=self.receipt["wrapper_sha256"] or
            _sha(ROOT/"braids_core/sediment/density_trace_faces_v98.py")!=self.receipt["core_wrapper_sha256"] or
            self.base.receipt["receipt_sha256"]!=self.receipt["base_runtime_sha256"]):
            raise ValueError("v98 CUDA density face-only runtime/source identity differs")
        self.base._validate_runtime()

    def launch_resident(self,inputs,lookup,*,reference_density,gravity):
        """Caller authenticates runtime/provenance before entering its residual."""
        cp,g=self.cp,self.geometry
        h,rho,distance,base,wet,mask,faces,th,tr,tw=inputs
        nc,nf,nt=g.cell_count,g.face_count,faces.size
        arrays=(*inputs,lookup)
        shapes=((nc,),(nc,),(nf,),(nf,),(nc,),(nf,),(nt,),(nt,),(nt,),(nt,),(nf,))
        integers={4,5,6,9,10}
        if any(not isinstance(a,cp.ndarray) or a.device.id!=self.device_id or not a.flags.c_contiguous or
            a.shape!=shape or a.dtype!=(cp.int32 if i in integers else cp.float64)
            for i,(a,shape) in enumerate(zip(arrays,shapes))):
            raise ValueError("v98 CUDA density face-only inputs must be selected-device resident arrays")
        acceleration,active,status=cp.empty(nf),cp.empty(nf,cp.int32),cp.empty(nf,cp.int32)
        self.base.kernel(((nf+127)//128,),(128,),(np.int64(nc),np.int64(nf),np.int64(nt),
            self.base.owner_device,lookup,h,rho,distance,base,mask,wet,th,tr,tw,
            np.float64(reference_density),np.float64(gravity),acceleration,active,status))
        return dict(face_acceleration=acceleration,density_mask=active,status=status)

    def evaluate(self,geometry,midpoint_depth,accepted_density_cell,**kwargs):
        inputs,lookup,provenance,cancel=validate_density_trace_inputs_v98(
            geometry,midpoint_depth,accepted_density_cell,**kwargs)
        if geometry.fingerprint!=self.geometry.fingerprint:
            raise ValueError("v98 CUDA density face-only call geometry differs")
        cp=self.cp
        with cp.cuda.Device(self.device_id):
            self._validate_runtime()
            result=self.launch_resident(tuple(cp.asarray(a) for a in inputs),cp.asarray(lookup),
                reference_density=provenance["reference_density"],gravity=provenance["gravity"])
            if np.any(cp.asnumpy(result["status"])):
                raise ValueError("v98 CUDA density face-only construction failed")
            accel,mask=cp.asnumpy(result["face_acceleration"]),cp.asnumpy(result["density_mask"])
        if cancel():raise InterruptedError("v98 CUDA density face-only cancelled after launch")
        return make_density_trace_faces_result_v98(geometry,inputs,provenance,accel,mask,
            runtime_sha=self.receipt["receipt_sha256"],backend="actual_cuda",threads=128)
