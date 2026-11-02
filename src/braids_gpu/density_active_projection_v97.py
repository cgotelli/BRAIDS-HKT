"""Actual CUDA wet-active projection primitive; not a wet/dry integrator."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from braids_core.sediment.density_active_projection_v97 import (
    POLICY,RANK_TOL,RESIDUAL_TOL,_hash,_sha,_geometry_token,
    DensityProjectionGeometryV97,validate_projection_inputs_v97,make_projection_result_v97,
)
from braids_core.sediment.densin_fwave_v85 import verify_locked_sources_v85

ROOT=Path(__file__).resolve().parents[1]
CUDA_SOURCE=Path(__file__).with_suffix(".cu")
OPTIONS=("--std=c++11","--fmad=false","--prec-div=true","--prec-sqrt=true","--ftz=false")


class ResidentDensityActiveProjectionV97:
    def __init__(self,geometry,*,backend):
        if backend!="implicit_fwave_midpoint_gpu":
            raise ValueError("v97 density projection requires explicit implicit GPU backend")
        if type(geometry) is not DensityProjectionGeometryV97 or geometry.fingerprint!=_geometry_token(geometry):
            raise ValueError("v97 density geometry identity differs")
        import cupy as cp
        self.cp=cp; self.geometry=geometry; self.device_id=int(cp.cuda.Device().id)
        self.source_sha=_sha(CUDA_SOURCE)
        self.module=cp.RawModule(code=CUDA_SOURCE.read_text(encoding="utf-8"),options=OPTIONS,
            name_expressions=("density_active_projection_v97",))
        self.kernel=self.module.get_function("density_active_projection_v97")
        self.geometry_device=tuple(cp.asarray(v) for v in
            (geometry.normal,geometry.length,geometry.offset,geometry.faces,geometry.signs))
        props=cp.cuda.runtime.getDeviceProperties(self.device_id)
        name=props["name"].decode() if isinstance(props["name"],bytes) else str(props["name"])
        self.receipt=dict(backend=backend,actual_cuda=True,automatic_cpu_fallback=False,
            source_sha256=self.source_sha,wrapper_sha256=_sha(Path(__file__)),
            core_wrapper_sha256=_sha(ROOT/"braids_core/sediment/density_active_projection_v97.py"),
            geometry_sha256=geometry.fingerprint,device_id=self.device_id,device_name=name,
            cuda_options=list(OPTIONS),locked_source_sha256=verify_locked_sources_v85(ROOT),
            numerical_python_cell_face_loops=0,production_admission=False)
        self.receipt["receipt_sha256"]=_hash(self.receipt)

    def evaluate(self,depth,face_acceleration,wet,active,*,policy,
                 rank_tol=RANK_TOL,residual_tol=RESIDUAL_TOL,cancelled=False):
        cancel=cancelled if callable(cancelled) else lambda: cancelled
        if not callable(cancelled) and type(cancelled) is not bool:
            raise ValueError("v97 CUDA cancellation must be bool/callable")
        if cancel():
            raise InterruptedError("v97 CUDA projection cancelled before launch")
        if (self.receipt.get("receipt_sha256")!=_hash({k:v for k,v in self.receipt.items() if k!="receipt_sha256"}) or
            _sha(CUDA_SOURCE)!=self.source_sha or _sha(Path(__file__))!=self.receipt["wrapper_sha256"] or
            _sha(ROOT/"braids_core/sediment/density_active_projection_v97.py")!=self.receipt["core_wrapper_sha256"]):
            raise ValueError("v97 CUDA runtime/source receipt differs")
        inputs=validate_projection_inputs_v97(self.geometry,depth,face_acceleration,wet,active,
            policy=policy,rank_tol=rank_tol,residual_tol=residual_tol)
        cp=self.cp; g=self.geometry; nc,nf=g.cell_count,g.face_count
        with cp.cuda.Device(self.device_id):
            if any(not np.array_equal(cp.asnumpy(actual),expected) for actual,expected in zip(
                    self.geometry_device,(g.normal,g.length,g.offset,g.faces,g.signs))):
                raise ValueError("v97 CUDA resident geometry differs")
            device_inputs=tuple(cp.asarray(v) for v in inputs)
            cell=cp.empty((2,nc)); rate=cp.empty((2,nc)); rank=cp.empty(nc,cp.int32)
            residual=cp.empty(nc); status=cp.empty(nc,cp.int32)
            self.kernel(((nc+127)//128,),(128,),(
                np.int64(nc),np.int64(nf),device_inputs[0],device_inputs[1],
                *self.geometry_device,device_inputs[2],device_inputs[3],
                np.float64(rank_tol),np.float64(residual_tol),cell,rate,rank,residual,status))
            code=cp.asnumpy(status)
            if np.any(code):
                raise ValueError("v97 CUDA density projection rejected rank/residual: "+str(code.tolist()))
            values=tuple(cp.asnumpy(v) for v in (cell,rate,rank,residual))
        if cancel():
            raise InterruptedError("v97 CUDA projection cancelled after launch")
        return make_projection_result_v97(g,inputs,*values,policy=policy,rank_tol=rank_tol,
            residual_tol=residual_tol,runtime_sha=self.receipt["receipt_sha256"],
            backend="actual_cuda",threads=128)
