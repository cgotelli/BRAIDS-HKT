"""Actual CUDA pre-capped internal-trace viscosity, no hydrodynamic owner."""
from pathlib import Path

import numpy as np

from braids_core.sediment.density_active_projection_v97 import _hash, _sha
from braids_core.sediment.density_trace_projection_v98 import DensityTraceGeometryV98, geometry_token_v98
from braids_core.sediment.momentum_trace_v98 import (
    validate_momentum_trace_inputs_v98, make_momentum_trace_result_v98,
)
from braids_core.sediment.shared_turbulence_consumers_v91 import verify_locked_sources_v91

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(__file__).with_suffix(".cu")
OPTIONS = ("--std=c++11", "--fmad=false", "--prec-div=true", "--prec-sqrt=true", "--ftz=false")


class ResidentMomentumTraceV98:
    def __init__(self, geometry, *, backend):
        if backend != "implicit_fwave_midpoint_gpu":
            raise ValueError("v98 momentum trace requires explicitly selected implicit GPU")
        if type(geometry) is not DensityTraceGeometryV98 or geometry.fingerprint != geometry_token_v98(geometry):
            raise ValueError("v98 momentum trace CUDA geometry identity differs")
        import cupy as cp
        self.cp, self.geometry = cp, geometry
        self.device_id = int(cp.cuda.runtime.getDevice())
        self.module = cp.RawModule(code=SOURCE.read_text(encoding="utf-8"), options=OPTIONS,
            name_expressions=("momentum_trace_faces_v98", "momentum_trace_cells_v98"))
        self.face_kernel = self.module.get_function("momentum_trace_faces_v98")
        self.cell_kernel = self.module.get_function("momentum_trace_cells_v98")
        g = geometry.projection
        self.geometry_host = (geometry.owner, g.length, g.offset, g.faces, g.signs)
        self.geometry_device = tuple(cp.asarray(a) for a in self.geometry_host)
        self.receipt = dict(backend=backend, actual_cuda=True, automatic_cpu_fallback=False,
            source_sha256=_sha(SOURCE), wrapper_sha256=_sha(Path(__file__)),
            core_wrapper_sha256=_sha(ROOT/"braids_core/sediment/momentum_trace_v98.py"),
            geometry_sha256=geometry.fingerprint, device_id=self.device_id,
            cuda_options=list(OPTIONS), locked_source_sha256=verify_locked_sources_v91(ROOT),
            cap_evaluation_count=0, numerical_python_cell_face_loops=0, production_admission=False)
        self.receipt["receipt_sha256"] = _hash(self.receipt)

    def _validate_runtime(self):
        if (self.receipt["receipt_sha256"] != _hash({k:v for k,v in self.receipt.items() if k != "receipt_sha256"}) or
                _sha(SOURCE) != self.receipt["source_sha256"] or
                _sha(Path(__file__)) != self.receipt["wrapper_sha256"] or
                _sha(ROOT/"braids_core/sediment/momentum_trace_v98.py") != self.receipt["core_wrapper_sha256"] or
                self.geometry.fingerprint != geometry_token_v98(self.geometry)):
            raise ValueError("v98 momentum trace CUDA runtime/source identity differs")
        if any(not np.array_equal(self.cp.asnumpy(a), b) for a,b in zip(self.geometry_device, self.geometry_host)):
            raise ValueError("v98 momentum trace CUDA resident geometry differs")

    def launch_resident(self, inputs, lookup):
        """Pure candidate seam; caller authenticates runtime/provenance at entry."""
        cp, g = self.cp, self.geometry
        q, vic, area, distance, wet, faces, tq, tv, tw = inputs
        nc, nf, nt = g.cell_count, g.face_count, faces.size
        arrays = (*inputs, lookup)
        shapes = ((3,nc),(nc,),(nc,),(nf,),(nc,),(nt,),(3,nt),(nt,),(nt,),(nf,))
        integers = {4,5,8,9}
        if any(not isinstance(a,cp.ndarray) or a.device.id != self.device_id or not a.flags.c_contiguous or
                a.shape != shape or a.dtype != (cp.int32 if i in integers else cp.float64)
                for i,(a,shape) in enumerate(zip(arrays,shapes))):
            raise ValueError("v98 momentum trace requires selected-device contiguous resident arrays")
        own,length,offset,adj,sign = self.geometry_device
        flux,rate = cp.empty((2,nf)),cp.empty((2,nc))
        fs,cs = cp.empty(nf,cp.int32),cp.empty(nc,cp.int32)
        self.face_kernel(((nf+127)//128,),(128,),(np.int64(nc),np.int64(nf),np.int64(nt),
            own,lookup,q,vic,length,distance,wet,tq,tv,tw,flux,fs))
        self.cell_kernel(((nc+127)//128,),(128,),(np.int64(nc),np.int64(nf),
            offset,adj,sign,wet,area,flux,rate,cs))
        return dict(face_flux=flux,momentum_rate=rate,face_status=fs,cell_status=cs)

    def evaluate(self, geometry, midpoint_q, capped_vicuv_cell, **kwargs):
        inputs,lookup,provenance,cancel = validate_momentum_trace_inputs_v98(
            geometry,midpoint_q,capped_vicuv_cell,**kwargs)
        if geometry.fingerprint != self.geometry.fingerprint:
            raise ValueError("v98 momentum trace CUDA call geometry differs")
        cp = self.cp
        with cp.cuda.Device(self.device_id):
            self._validate_runtime()
            result = self.launch_resident(tuple(cp.asarray(a) for a in inputs),cp.asarray(lookup))
            if np.any(cp.asnumpy(result["face_status"])) or np.any(cp.asnumpy(result["cell_status"])):
                raise ValueError("v98 momentum trace CUDA face/CSR computation failed")
            flux,rate = cp.asnumpy(result["face_flux"]),cp.asnumpy(result["momentum_rate"])
        if cancel():
            raise InterruptedError("v98 momentum trace cancelled after CUDA launch")
        return make_momentum_trace_result_v98(geometry,inputs,provenance,flux,rate,
            runtime_sha=self.receipt["receipt_sha256"],backend="actual_cuda",threads=128)
