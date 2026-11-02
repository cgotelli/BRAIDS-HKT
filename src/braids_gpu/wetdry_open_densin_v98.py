"""Resident wet-active physical-open DensIn augmentation, not an open solver.

Parent boundary density is accepted-state data. A sole hydraulic boundary
owner supplies the current physical trace and independent ownership/density
masks. This helper neither resolves a boundary nor changes its physical flux.
The caller applies the unchanged active projection exactly once afterwards.
"""
from pathlib import Path

import numpy as np

from braids_core.sediment.coupled_open_boundary_native_cpu_v96 import (
    validate_accepted_boundary_density_receipt_v96,
)
from braids_core.sediment.density_active_projection_v97 import _digest, _hash, _sha
from braids_core.sediment.density_trace_projection_v98 import _hex
from braids_core.sediment.hydraulic_boundary_trace_cpu_v96 import (
    OWNER_ACTUAL_CUDA, execution_document_hydraulic_boundary_trace_v96,
)

SOURCE = Path(__file__).with_suffix(".cu")
OPTIONS = ("--std=c++11", "--fmad=false", "--prec-div=true", "--prec-sqrt=true", "--ftz=false")


class ResidentWetDryOpenDensInGPUV98:
    def __init__(self, boundary_execution, *, backend, gravity_m_s2,
                 reference_density_kg_m3, dry_tolerance_m, ibaroc_open_boundary=True):
        execution_document_hydraulic_boundary_trace_v96(boundary_execution)
        if (backend != "implicit_fwave_midpoint_gpu"
                or boundary_execution.hydraulic_owner_kind != OWNER_ACTUAL_CUDA):
            raise ValueError("v98 wet/dry exterior density requires explicitly selected actual CUDA owner")
        controls = (gravity_m_s2, reference_density_kg_m3, dry_tolerance_m)
        if (type(ibaroc_open_boundary) is not bool or any(isinstance(v, bool)
                or not isinstance(v, (int, float)) or not np.isfinite(v) for v in controls)
                or gravity_m_s2 <= 0 or reference_density_kg_m3 <= 0 or dry_tolerance_m < 0):
            raise ValueError("v98 wet/dry exterior density controls differ")
        self.execution = boundary_execution
        self.nf, self.nc = boundary_execution.face_count, boundary_execution.cell_count
        self.controls = dict(gravity_m_s2=float(gravity_m_s2),
            reference_density_kg_m3=float(reference_density_kg_m3), dry_tolerance_m=float(dry_tolerance_m),
            ibaroc_open_boundary=ibaroc_open_boundary)
        self.geometry_host = tuple(np.array(a, copy=True, order="C") for a in (
            boundary_execution.owner, boundary_execution.face_distance_m,
            boundary_execution.open_mask.astype(np.int32)))
        for a in self.geometry_host:
            a.setflags(write=False)
        import cupy as cp
        self.cp, self.device_id = cp, int(cp.cuda.runtime.getDevice())
        self.geometry_device = tuple(cp.asarray(a) for a in self.geometry_host)
        self.module = cp.RawModule(code=SOURCE.read_text(encoding="utf-8"), options=OPTIONS,
            name_expressions=("wetdry_physical_densin_v98",))
        self.kernel = self.module.get_function("wetdry_physical_densin_v98")
        self.receipt = dict(backend=backend, actual_cuda=True,
            boundary_execution_fingerprint=boundary_execution.fingerprint,
            source_sha256=_sha(SOURCE), wrapper_sha256=_sha(Path(__file__)),
            geometry_sha256=_digest(*self.geometry_host), controls=self.controls.copy(),
            device_id=self.device_id, cuda_options=list(OPTIONS), automatic_cpu_fallback=False,
            production_admission=False)
        self.receipt["receipt_sha256"] = _hash(self.receipt)
        self.parent_density = None
        self.parent_density_device = None
        self.last_resident_result = None
        self.kernel_launch_count = 0

    def set_parent_boundary_density(self, receipt):
        """Bind the exact accepted density receipt once before residual stepping."""
        validate_accepted_boundary_density_receipt_v96(receipt, self.execution)
        self.parent_density = receipt
        with self.cp.cuda.Device(self.device_id):
            self.parent_density_device = self.cp.asarray(receipt.mixture_density_kg_m3).copy()
        self.last_resident_result = None

    def validate_runtime(self):
        """Sequence-entry authentication; no numerical residual is evaluated."""
        execution_document_hydraulic_boundary_trace_v96(self.execution)
        if (self.receipt["receipt_sha256"] != _hash({k: v for k, v in self.receipt.items()
                if k != "receipt_sha256"}) or _sha(SOURCE) != self.receipt["source_sha256"]
                or _sha(Path(__file__)) != self.receipt["wrapper_sha256"]
                or _digest(*self.geometry_host) != self.receipt["geometry_sha256"]
                or self.controls != self.receipt["controls"]
                or self.execution.fingerprint != self.receipt["boundary_execution_fingerprint"]):
            raise ValueError("v98 wet/dry exterior density runtime/source identity differs")
        if any(not np.array_equal(self.cp.asnumpy(a), b)
                for a, b in zip(self.geometry_device, self.geometry_host)):
            raise ValueError("v98 wet/dry exterior density resident geometry differs")
        if self.parent_density is None or self.parent_density_device is None:
            raise ValueError("v98 wet/dry exterior density has no accepted boundary density")
        validate_accepted_boundary_density_receipt_v96(self.parent_density, self.execution)
        if not np.array_equal(self.cp.asnumpy(self.parent_density_device),
                              self.parent_density.mixture_density_kg_m3):
            raise ValueError("v98 wet/dry exterior density accepted parent image differs")

    def launch_resident(self, depth, selected_density, wet_cell_mask, hydraulic_trace_q,
                        physical_owned_mask, exterior_density_request_mask, base_acceleration,
                        base_density_mask, *, current_trace_owner_sha256, cancelled=False):
        self.last_resident_result = None
        if not callable(cancelled) and type(cancelled) is not bool:
            raise ValueError("v98 wet/dry exterior density cancellation must be bool/callable")
        poll = cancelled if callable(cancelled) else lambda: cancelled
        if poll():
            raise InterruptedError("v98 wet/dry exterior density cancelled before launch")
        trace_sha = _hex(current_trace_owner_sha256, "current physical trace owner")
        if self.parent_density is None:
            raise ValueError("v98 wet/dry exterior density has no accepted boundary density")
        cp, nf, nc = self.cp, self.nf, self.nc
        inputs = (depth, selected_density, wet_cell_mask, hydraulic_trace_q,
            physical_owned_mask, exterior_density_request_mask, base_acceleration, base_density_mask)
        shapes = ((nc,), (nc,), (nc,), (3, nf), (nf,), (nf,), (nf,), (nf,))
        types = (cp.float64, cp.float64, cp.int32, cp.float64,
                 cp.int32, cp.int32, cp.float64, cp.int32)
        if any(not isinstance(a, cp.ndarray) or a.device.id != self.device_id
                or not a.flags.c_contiguous or a.shape != shape or a.dtype != dtype
                for a, shape, dtype in zip(inputs, shapes, types)):
            raise ValueError("v98 wet/dry exterior density requires contiguous selected-device inputs")
        out = dict(face_acceleration=cp.empty(nf, cp.float64), density_mask=cp.empty(nf, cp.int32),
            exterior_active=cp.empty(nf, cp.int32), status=cp.empty(nf, cp.int32))
        own, distance, geometry_open = self.geometry_device
        c = self.controls
        with cp.cuda.Device(self.device_id):
            self.kernel(((nf+127)//128,), (128,), (np.int32(nf), np.float64(c["gravity_m_s2"]),
                np.float64(c["reference_density_kg_m3"]), np.float64(c["dry_tolerance_m"]),
                np.int32(c["ibaroc_open_boundary"]), depth, selected_density, wet_cell_mask,
                hydraulic_trace_q, self.parent_density_device, own, distance, geometry_open,
                physical_owned_mask, exterior_density_request_mask, base_acceleration, base_density_mask,
                out["face_acceleration"], out["density_mask"], out["exterior_active"], out["status"]))
        self.kernel_launch_count += 1
        if poll():
            raise InterruptedError("v98 wet/dry exterior density cancelled after launch")
        out["audit"] = dict(runtime_sha256=self.receipt["receipt_sha256"],
            accepted_boundary_density_receipt_sha256=self.parent_density.receipt_sha256,
            current_trace_owner_sha256=trace_sha, density_face_augmentation_count=1,
            density_projection_count=0, physical_flux_replacement_count=0,
            riemann_evaluation_count=0, boundary_solver_evaluation_count=0,
            boundary_memory_publication_count=0, state_or_clock_publication_count=0,
            numerical_host_readback_count=0, automatic_cpu_fallback=False, production_admission=False)
        self.last_resident_result = out
        return out
