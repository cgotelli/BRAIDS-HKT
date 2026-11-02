"""Actual-CUDA internal trace DensIn, with resident augmentation/projection.

The diagnostic evaluate API uploads inputs and collects authenticated outputs.
The resident kernel seam is exposed for the caller's already-authenticated true
residual, without a second hydraulic solve or CPU numerical fallback.
"""
from pathlib import Path

import numpy as np

from braids_core.sediment.density_active_projection_v97 import (
    POLICY, RANK_TOL, RESIDUAL_TOL, _hash, _sha, make_projection_result_v97,
)
from braids_core.sediment.density_trace_projection_v98 import (
    DensityTraceGeometryV98, geometry_token_v98, validate_density_trace_inputs_v98,
    make_density_trace_result_v98,
)
from braids_core.sediment.densin_fwave_v85 import verify_locked_sources_v85
from .density_active_projection_v97 import ResidentDensityActiveProjectionV97

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(__file__).with_suffix(".cu")
OPTIONS = ("--std=c++11", "--fmad=false", "--prec-div=true", "--prec-sqrt=true", "--ftz=false")


class ResidentDensityTraceProjectionV98:
    def __init__(self, geometry, *, backend):
        if backend != "implicit_fwave_midpoint_gpu":
            raise ValueError("v98 density trace requires explicitly selected implicit GPU")
        if (type(geometry) is not DensityTraceGeometryV98 or
                geometry.fingerprint != geometry_token_v98(geometry)):
            raise ValueError("v98 CUDA density trace geometry identity differs")
        import cupy as cp
        self.cp, self.geometry = cp, geometry
        self.projection = ResidentDensityActiveProjectionV97(geometry.projection, backend=backend)
        self.device_id = self.projection.device_id
        self.module = cp.RawModule(code=SOURCE.read_text(encoding="utf-8"), options=OPTIONS,
                                  name_expressions=("density_trace_faces_v98",))
        self.kernel = self.module.get_function("density_trace_faces_v98")
        self.owner_device = cp.asarray(geometry.owner)
        self.receipt = dict(backend=backend, actual_cuda=True, automatic_cpu_fallback=False,
            source_sha256=_sha(SOURCE), wrapper_sha256=_sha(Path(__file__)),
            core_wrapper_sha256=_sha(ROOT/"braids_core/sediment/density_trace_projection_v98.py"),
            geometry_sha256=geometry.fingerprint, device_id=self.device_id,
            projection_runtime_sha256=self.projection.receipt["receipt_sha256"],
            cuda_options=list(OPTIONS), locked_source_sha256=verify_locked_sources_v85(ROOT),
            numerical_python_cell_face_loops=0, production_admission=False)
        self.receipt["receipt_sha256"] = _hash(self.receipt)

    def _validate_runtime(self):
        if (self.receipt.get("receipt_sha256") != _hash({k:v for k,v in self.receipt.items() if k != "receipt_sha256"}) or
                _sha(SOURCE) != self.receipt["source_sha256"] or
                _sha(Path(__file__)) != self.receipt["wrapper_sha256"] or
                _sha(ROOT/"braids_core/sediment/density_trace_projection_v98.py") != self.receipt["core_wrapper_sha256"] or
                self.geometry.fingerprint != geometry_token_v98(self.geometry) or
                self.projection.receipt["receipt_sha256"] != self.receipt["projection_runtime_sha256"]):
            raise ValueError("v98 CUDA density trace runtime/source identity differs")
        cp, g = self.cp, self.geometry
        if (not np.array_equal(cp.asnumpy(self.owner_device), g.owner) or
                any(not np.array_equal(cp.asnumpy(actual), expected) for actual, expected in zip(
                    self.projection.geometry_device, (g.projection.normal, g.projection.length,
                    g.projection.offset, g.projection.faces, g.projection.signs)))):
            raise ValueError("v98 CUDA density trace resident geometry differs")

    def launch_resident(self, inputs, lookup, *, reference_density, gravity):
        """For authenticated caller inputs; returns only resident candidate arrays.

        `inputs` is the ten-array order from validate_density_trace_inputs_v98.
        Caller owns trace provenance, cancellation, nonlinear residual count and
        accepted publication. This method never advances a clock or solves fluxes.
        """
        cp, g = self.cp, self.geometry
        h, rho, distance, base, wet, base_mask, faces, th, tr, tw = inputs
        arrays = (*inputs, lookup)
        if any(not isinstance(a, cp.ndarray) or a.device.id != self.device_id or not a.flags.c_contiguous for a in arrays):
            raise ValueError("v98 CUDA trace inputs must be resident contiguous arrays on the selected device")
        nc, nf, nt = g.cell_count, g.face_count, faces.size
        expected = ((nc,), (nc,), (nf,), (nf,), (nc,), (nf,), (nt,), (nt,), (nt,), (nt,), (nf,))
        integers = {4, 5, 6, 9, 10}
        if any(a.shape != shape or a.dtype != (cp.int32 if i in integers else cp.float64)
               for i, (a, shape) in enumerate(zip(arrays, expected))):
            raise ValueError("v98 CUDA trace resident input shape/dtype differs")
        accel, mask, face_status = cp.empty(nf), cp.empty(nf, cp.int32), cp.empty(nf, cp.int32)
        cell, rate = cp.empty((2, nc)), cp.empty((2, nc))
        rank, residual, status = cp.empty(nc, cp.int32), cp.empty(nc), cp.empty(nc, cp.int32)
        self.kernel(((nf+127)//128,), (128,), (np.int64(nc), np.int64(nf), np.int64(nt),
            self.owner_device, lookup, h, rho, distance, base, base_mask, wet,
            th, tr, tw, np.float64(reference_density), np.float64(gravity), accel, mask, face_status))
        self.projection.kernel(((nc+127)//128,), (128,), (np.int64(nc), np.int64(nf),
            h, accel, *self.projection.geometry_device, wet, mask,
            np.float64(RANK_TOL), np.float64(RESIDUAL_TOL), cell, rate, rank, residual, status))
        return dict(face_acceleration=accel, density_mask=mask, face_status=face_status,
                    cell_acceleration=cell, momentum_rate=rate, rank=rank,
                    residual=residual, projection_status=status)

    def evaluate(self, geometry, midpoint_depth, accepted_density_cell, **kwargs):
        inputs, lookup, provenance, cancel = validate_density_trace_inputs_v98(
            geometry, midpoint_depth, accepted_density_cell, **kwargs)
        if geometry.fingerprint != self.geometry.fingerprint:
            raise ValueError("v98 CUDA density trace call geometry differs")
        cp = self.cp
        with cp.cuda.Device(self.device_id):
            self._validate_runtime()
            device_inputs = tuple(cp.asarray(a) for a in inputs)
            result = self.launch_resident(device_inputs, cp.asarray(lookup),
                reference_density=provenance["reference_density"], gravity=provenance["gravity"])
            if np.any(cp.asnumpy(result["face_status"])) or np.any(cp.asnumpy(result["projection_status"])):
                raise ValueError("v98 CUDA density trace rank/residual/face construction failed")
            values = {key: cp.asnumpy(value) for key, value in result.items()}
        if cancel():
            raise InterruptedError("v98 CUDA density trace cancelled after launch")
        h, _, _, _, wet, _, _, _, _, _ = inputs
        projected = make_projection_result_v97(geometry.projection,
            (h, values["face_acceleration"], wet, values["density_mask"]),
            values["cell_acceleration"], values["momentum_rate"], values["rank"], values["residual"],
            policy=POLICY, rank_tol=RANK_TOL, residual_tol=RESIDUAL_TOL,
            runtime_sha=self.projection.receipt["receipt_sha256"], backend="actual_cuda", threads=128)
        return make_density_trace_result_v98(geometry, inputs, provenance,
            values["face_acceleration"], values["density_mask"], projected,
            runtime_sha=self.receipt["receipt_sha256"],
            projection_runtime_sha=self.projection.receipt["receipt_sha256"],
            backend="actual_cuda", threads=128)
