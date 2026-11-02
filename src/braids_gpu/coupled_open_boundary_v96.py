"""Resident actual-CUDA exterior-density operator for Q4/v96.

This additive operator consumes the physical trace produced by the existing
implicit-f-wave boundary owner.  It never resolves a boundary condition and
never feeds that trace to a second Riemann solve.  The unchanged v85 kernel
owns interior faces; this kernel overwrites only the authenticated active-open
face accelerations before the single v85 face-to-cell projection.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from braids_core.sediment.coupled_open_boundary_native_cpu_v96 import (
    AcceptedBoundaryDensityReceiptV96,
    validate_accepted_boundary_density_receipt_v96,
)
from braids_core.sediment.hydraulic_boundary_trace_cpu_v96 import (
    HydraulicBoundaryTraceExecutionV96,
    HydraulicBoundaryTraceReceiptV96,
    OWNER_ACTUAL_CUDA,
    execution_document_hydraulic_boundary_trace_v96,
    resolve_hydraulic_boundary_trace_execution_v96,
    validate_hydraulic_boundary_trace_receipt_v96,
)
from braids_core.sediment.densin_fwave_v85 import DensInExecutionV85


SCHEMA = 1
SCOPE_ID = "braids-q4-coupled-open-boundary-gpu-v96-v1"
SEMANTIC_ID = SCOPE_ID + ":sole-owner-trace-exterior-densin"
BACKEND = "actual_cuda"
CUDA_SOURCE = Path(__file__).with_suffix(".cu")
CUDA_OPTIONS = ("--std=c++11", "--fmad=false", "--prec-div=true",
                "--prec-sqrt=true", "--ftz=false")
KERNEL_NAME = "v96_open_densin_face"
THREADS = 128
_ROOT = Path(__file__).resolve().parents[1]


class CoupledOpenBoundaryGPUV96Error(ValueError):
    """The actual-CUDA open-boundary operator contract is invalid."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError, OverflowError) as error:
        raise CoupledOpenBoundaryGPUV96Error(
            "v96 CUDA open-boundary value is not canonical JSON") from error


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CoupledOpenBoundaryGPUV96Error(message)


@dataclass(frozen=True)
class CoupledOpenBoundaryGPUExecutionV96:
    boundary_trace_execution: HydraulicBoundaryTraceExecutionV96
    ibaroc_open_boundary: bool
    semantic_id: str
    fingerprint: str


def _execution_payload(value: CoupledOpenBoundaryGPUExecutionV96) -> dict:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "semantic_id": value.semantic_id,
        "backend": BACKEND,
        "boundary_trace_execution":
            execution_document_hydraulic_boundary_trace_v96(
                value.boundary_trace_execution),
        "ibaroc_open_boundary": value.ibaroc_open_boundary,
        "interior_density_owner": "unchanged_v85_face_kernel",
        "open_density_owner": "active_open_face_overwrite_before_one_projection",
        "trace_role": "physical_boundary_state_never_riemann_right_state",
        "boundary_solver_call_count": 0,
        "automatic_cpu_fallback": False,
        "product_admission": False,
    }


def resolve_coupled_open_boundary_gpu_execution_v96(
    boundary_trace_execution: HydraulicBoundaryTraceExecutionV96, *,
    ibaroc_open_boundary: bool = True,
) -> CoupledOpenBoundaryGPUExecutionV96:
    execution_document_hydraulic_boundary_trace_v96(
        boundary_trace_execution)
    _require(boundary_trace_execution.hydraulic_owner_kind ==
             OWNER_ACTUAL_CUDA,
             "v96 CUDA open density requires the actual controller trace owner")
    _require(type(ibaroc_open_boundary) is bool,
             "v96 CUDA IBaroc selection must be Boolean")
    placeholder = CoupledOpenBoundaryGPUExecutionV96(
        boundary_trace_execution, ibaroc_open_boundary, SEMANTIC_ID, "")
    return CoupledOpenBoundaryGPUExecutionV96(
        **{**placeholder.__dict__,
           "fingerprint": _fingerprint(_execution_payload(placeholder))})


def resolve_live_coupled_open_boundary_gpu_execution_v96(
    model: object,
    hydraulic_execution: DensInExecutionV85, *,
    ibaroc_open_boundary: bool = True,
) -> CoupledOpenBoundaryGPUExecutionV96:
    """Bind the typed trace execution to the actually loaded GPU owner."""
    controller = getattr(model, "implicit_midpoint", None)
    boundary = getattr(model, "boundary", None)
    _require(controller is not None and boundary is not None and
             getattr(controller, "solver", None) is model,
             "v96 CUDA live boundary owner is unavailable")
    cp = getattr(controller, "cp", None)
    _require(cp is not None and getattr(boundary, "cp", None) is cp,
             "v96 CUDA live boundary backend differs")
    owner = np.ascontiguousarray(cp.asnumpy(controller.owner), dtype=np.int32)
    neighbour = np.ascontiguousarray(
        cp.asnumpy(controller.neighbour), dtype=np.int32)
    bid = np.ascontiguousarray(cp.asnumpy(boundary.arrays["bid"]),
                              dtype=np.int32)
    mode = np.ascontiguousarray(cp.asnumpy(controller.boundary_mode),
                               dtype=np.int32)
    _require(np.array_equal(owner, hydraulic_execution.owner) and
             np.array_equal(neighbour, hydraulic_execution.neighbour) and
             np.any((neighbour < 0) & (bid >= 0)),
             "v96 CUDA live boundary topology differs")
    source_inventory: dict[str, str] = {}
    for name in ("_parent_source", "_g1_bridge_source", "_outlet_source"):
        path = getattr(boundary, name, None)
        if path is not None:
            resolved = Path(path).resolve()
            _require(resolved.is_file() and not resolved.is_symlink(),
                     "v96 CUDA boundary source path differs")
            source_inventory[name] = hashlib.sha256(
                resolved.read_bytes()).hexdigest()
    controller_source = Path(controller.source_path).resolve()
    source_inventory["implicit_midpoint_fwave_cuda"] = hashlib.sha256(
        controller_source.read_bytes()).hexdigest()
    _require(len(source_inventory) >= 2 and
             source_inventory["implicit_midpoint_fwave_cuda"] ==
                controller.source_sha256,
             "v96 CUDA live boundary source inventory differs")
    array_identity = {
        "bid": hashlib.sha256(bid.tobytes()).hexdigest(),
        "mode": hashlib.sha256(mode.tobytes()).hexdigest(),
        "boundary_faces": hashlib.sha256(np.ascontiguousarray(
            cp.asnumpy(boundary.arrays["boundary_faces"])).tobytes()).hexdigest(),
        "boundary_offset": hashlib.sha256(np.ascontiguousarray(
            cp.asnumpy(boundary.arrays["boundary_offset"])).tobytes()).hexdigest(),
        "forcing": hashlib.sha256(np.ascontiguousarray(
            cp.asnumpy(boundary.arrays["forcing"])).tobytes()).hexdigest(),
    }
    trace_execution = resolve_hydraulic_boundary_trace_execution_v96(
        owner=owner, neighbour=neighbour, boundary_index=bid,
        boundary_mode=mode,
        face_distance_m=hydraulic_execution.face_distance_m,
        hydraulic_owner_kind=OWNER_ACTUAL_CUDA,
        hydraulic_owner_execution_sha256=_fingerprint({
            "hydraulic_execution":
                hydraulic_execution.hydraulic_execution_sha256,
            "controller_source": controller.source_sha256,
            "boundary_arrays": array_identity,
        }),
        hydraulic_owner_source_sha256=_fingerprint(source_inventory),
        hydraulic_owner_runtime_sha256=_fingerprint({
            "source_inventory": source_inventory,
            "boundary_arrays": array_identity,
            "device_id": int(cp.cuda.Device().id),
            "owner": "implicit_midpoint_boundary_trace_adapter",
        }))
    return resolve_coupled_open_boundary_gpu_execution_v96(
        trace_execution,
        ibaroc_open_boundary=ibaroc_open_boundary)


def execution_document_coupled_open_boundary_gpu_v96(
    value: CoupledOpenBoundaryGPUExecutionV96,
) -> dict:
    _require(type(value) is CoupledOpenBoundaryGPUExecutionV96,
             "v96 CUDA open-boundary execution type differs")
    payload = _execution_payload(value)
    _require(value.semantic_id == SEMANTIC_ID and
             type(value.ibaroc_open_boundary) is bool and
             value.fingerprint == _fingerprint(payload),
             "v96 CUDA open-boundary execution fingerprint differs")
    return {**payload, "fingerprint": value.fingerprint}


class ResidentOpenBoundaryDensInGPUV96:
    """Overwrite active-open v85 face accelerations without a host fallback."""

    def __init__(self, execution: CoupledOpenBoundaryGPUExecutionV96, *,
                 cupy_module=None) -> None:
        execution_document_coupled_open_boundary_gpu_v96(execution)
        if cupy_module is None:
            import cupy as cp
        else:
            cp = cupy_module
        self.cp = cp
        self.execution = execution
        self.trace_execution = execution.boundary_trace_execution
        self.device_id = int(cp.cuda.Device().id)
        source = CUDA_SOURCE.read_text(encoding="utf-8")
        self.module = cp.RawModule(
            code=source, options=CUDA_OPTIONS,
            name_expressions=(KERNEL_NAME,))
        self.kernel = self.module.get_function(KERNEL_NAME)
        self.nf = self.trace_execution.face_count
        self.owner = cp.ascontiguousarray(cp.asarray(
            self.trace_execution.owner, dtype=cp.int32))
        self.distance = cp.ascontiguousarray(cp.asarray(
            self.trace_execution.face_distance_m, dtype=cp.float64))
        self.exterior_densin_active_face = cp.zeros(
            self.nf, dtype=cp.int32)
        self.boundary_mixture_density = cp.zeros(
            self.nf, dtype=cp.float64)
        self.hydraulic_trace_q = cp.zeros((3, self.nf), dtype=cp.float64)
        self.kernel_launch_count = 0
        properties = cp.cuda.runtime.getDeviceProperties(self.device_id)
        name = properties["name"]
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="replace")
        paths = (Path(__file__).resolve(), CUDA_SOURCE)
        self.runtime_receipt = {
            "scope_id": SCOPE_ID + ":runtime",
            "execution_fingerprint": execution.fingerprint,
            "backend": BACKEND,
            "actual_cuda_kernels": True,
            "boundary_solver_call_count": 0,
            "automatic_cpu_fallback": False,
            "device_id": self.device_id,
            "device_name": str(name),
            "threads_per_block": THREADS,
            "source_sha256": {
                str(path.relative_to(_ROOT)).replace("\\", "/"):
                    hashlib.sha256(path.read_bytes()).hexdigest()
                for path in paths
            },
        }
        self.runtime_receipt["receipt_sha256"] = _fingerprint(
            self.runtime_receipt)

    def _device(self, value, shape, dtype, name):
        cp = self.cp
        _require(isinstance(value, cp.ndarray) and value.shape == shape and
                 value.dtype == dtype and value.flags.c_contiguous and
                 int(value.device.id) == self.device_id,
                 f"v96 CUDA open-boundary {name} must be resident {shape}")
        return value

    def set_parent_boundary_density(
        self, receipt: AcceptedBoundaryDensityReceiptV96,
    ) -> None:
        validate_accepted_boundary_density_receipt_v96(
            receipt, self.trace_execution)
        self.boundary_mixture_density.set(receipt.mixture_density_kg_m3)

    def set_residual_trace(
        self, receipt: HydraulicBoundaryTraceReceiptV96, *,
        midpoint_q_host: np.ndarray, bed_elevation_host: np.ndarray,
    ) -> None:
        validate_hydraulic_boundary_trace_receipt_v96(
            receipt, self.trace_execution,
            midpoint_q=midpoint_q_host,
            bed_elevation_m=bed_elevation_host,
            hydraulic_request_sha256=receipt.hydraulic_request_sha256,
            parent_state_sha256=receipt.parent_state_sha256,
            evaluation_index=receipt.evaluation_index,
            evaluation_time_s=receipt.evaluation_time_s)
        self.hydraulic_trace_q.set(receipt.trace_q)
        self.exterior_densin_active_face.set(
            receipt.exterior_densin_active_mask.astype(np.int32))

    def overwrite_active_open_faces(
        self, *, cell_depth, cell_mixture_density, hydraulic_trace_q,
        active_open_face, face_acceleration, face_status,
        gravity_m_s2: float, reference_density_kg_m3: float,
    ) -> None:
        """Launch once after v85 faces and before the one v85 projection."""
        cp, nf = self.cp, self.nf
        depth = self._device(
            cell_depth, (self.trace_execution.cell_count,), cp.float64,
            "cell depth")
        mixture = self._device(
            cell_mixture_density, (self.trace_execution.cell_count,),
            cp.float64, "cell mixture density")
        trace = self._device(
            hydraulic_trace_q, (3, nf), cp.float64, "hydraulic trace")
        active = self._device(
            active_open_face, (nf,), cp.int32, "active-open mask")
        acceleration = self._device(
            face_acceleration, (nf,), cp.float64, "face acceleration")
        status = self._device(face_status, (nf,), cp.int32, "face status")
        _require(np.isfinite(gravity_m_s2) and gravity_m_s2 > 0.0 and
                 np.isfinite(reference_density_kg_m3) and
                 reference_density_kg_m3 > 0.0,
                 "v96 CUDA open-boundary constants are invalid")
        self.kernel_launch_count += 1
        self.kernel(((nf+THREADS-1)//THREADS,), (THREADS,), (
            depth, mixture, self.boundary_mixture_density, trace, active,
            self.owner, self.distance, np.int64(nf),
            np.float64(gravity_m_s2), np.float64(reference_density_kg_m3),
            np.int32(self.execution.ibaroc_open_boundary), acceleration,
            status))


def method_identity_coupled_open_boundary_gpu_v96() -> dict[str, object]:
    return {
        "scope_id": SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "backend": BACKEND,
        "interior_faces": "unchanged_v85",
        "open_faces": "same_owner_physical_trace_exterior_densin",
        "trace_is_riemann_right_state": False,
        "actual_cuda_kernel": True,
        "automatic_cpu_fallback": False,
        "product_admission": False,
    }


__all__ = [
    "BACKEND", "SEMANTIC_ID", "CoupledOpenBoundaryGPUExecutionV96",
    "CoupledOpenBoundaryGPUV96Error", "ResidentOpenBoundaryDensInGPUV96",
    "execution_document_coupled_open_boundary_gpu_v96",
    "method_identity_coupled_open_boundary_gpu_v96",
    "resolve_coupled_open_boundary_gpu_execution_v96",
    "resolve_live_coupled_open_boundary_gpu_execution_v96",
]
