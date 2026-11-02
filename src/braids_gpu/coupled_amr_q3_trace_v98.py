"""Authenticated coarse Q3 constitutive trace for actual-CUDA v98."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np

from .active_suspended_control_v94 import ResidentQ3V83ActiveFaceTrialV94
from .coupled_amr_trace_v98 import (
    CoupledAMRInternalTraceGPUV98,
    validate_coupled_amr_internal_trace_structure_gpu_v98,
)
from .coupled_wetdry_v97 import (
    CoupledWetDryGPUExecutionV97,
    CoupledWetDryGPUTrialV97,
)


SCHEMA = 1
SCOPE_ID = "braids.q4.coupled_amr.q3_trace.actual_cuda.v98"
BACKEND = "actual_cuda"


class CoupledAMRQ3TraceGPUV98Error(ValueError):
    """The coarse actual-CUDA Q3 trace is invalid or stale."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError, OverflowError) as error:
        raise CoupledAMRQ3TraceGPUV98Error(
            "v98 CUDA Q3 trace identity is not canonical JSON") from error


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _digest(*values: object) -> str:
    result = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(value)
        result.update(array.dtype.str.encode("ascii"))
        result.update(_canonical(list(array.shape)))
        result.update(array.tobytes(order="C"))
    return result.hexdigest()


def _readonly(value: object, shape: tuple[int, ...], name: str, *,
              nonnegative: bool = False) -> np.ndarray:
    raw = np.asarray(value)
    if (raw.shape != shape or raw.dtype.kind not in "fiu" or
            not np.isfinite(raw).all() or
            (nonnegative and np.any(raw < 0.0))):
        raise CoupledAMRQ3TraceGPUV98Error(
            f"v98 CUDA Q3 trace {name} differs")
    result = np.array(raw, dtype=np.float64, order="C", copy=True)
    result.setflags(write=False)
    return result


def _face_trial(
    trial: CoupledWetDryGPUTrialV97,
) -> ResidentQ3V83ActiveFaceTrialV94:
    try:
        face = trial.active_trial.base_v93_trial.q3_open_trial.face_trial
    except AttributeError as error:
        raise CoupledAMRQ3TraceGPUV98Error(
            "v98 CUDA Q3 trace lacks the coarse face trial") from error
    if type(face) is not ResidentQ3V83ActiveFaceTrialV94:
        raise CoupledAMRQ3TraceGPUV98Error(
            "v98 CUDA Q3 coarse face-trial type differs")
    return face


@dataclass(frozen=True)
class CoupledAMRQ3TraceGPUV98:
    internal_trace_receipt_sha256: str
    coarse_trial_sha256: str
    coarse_face_trial_sha256: str
    coarse_producer_receipt_sha256: str
    fine_execution_fingerprint: str
    coarse_cell: int
    interface_faces: tuple[int, int]
    interface_vector_kg_m_s: np.ndarray
    interface_taurat: np.ndarray
    interface_hiding: np.ndarray
    interface_availability: np.ndarray
    constitutive_recompute_count: int
    resident_readback_count: int
    physical_boundary_owner_count: int
    accepted_clock_increment_count: int
    backend: str
    receipt_sha256: str


def _payload(value: CoupledAMRQ3TraceGPUV98) -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "internal_trace": value.internal_trace_receipt_sha256,
        "coarse_trial": value.coarse_trial_sha256,
        "coarse_face_trial": value.coarse_face_trial_sha256,
        "coarse_producer": value.coarse_producer_receipt_sha256,
        "fine_execution": value.fine_execution_fingerprint,
        "coarse_cell": value.coarse_cell,
        "interface_faces": list(value.interface_faces),
        "arrays_sha256": _digest(
            value.interface_vector_kg_m_s, value.interface_taurat,
            value.interface_hiding, value.interface_availability),
        "constitutive_recompute_count": value.constitutive_recompute_count,
        "resident_readback_count": value.resident_readback_count,
        "physical_boundary_owner_count": value.physical_boundary_owner_count,
        "accepted_clock_increment_count": value.accepted_clock_increment_count,
        "source_order": "reuse_current_coarse_v83_then_fine_face_q3",
        "cumulative_external_stock_limit":
            "macro_owner_must_reject_negative_corrected_bed",
        "backend": value.backend,
        "automatic_cpu_fallback": False,
    }


def resolve_coupled_amr_q3_trace_gpu_v98(
    trace: CoupledAMRInternalTraceGPUV98,
    coarse_trial: CoupledWetDryGPUTrialV97,
    fine_execution: CoupledWetDryGPUExecutionV97,
) -> CoupledAMRQ3TraceGPUV98:
    validate_coupled_amr_internal_trace_structure_gpu_v98(
        trace, fine_execution)
    if type(coarse_trial) is not CoupledWetDryGPUTrialV97:
        raise CoupledAMRQ3TraceGPUV98Error(
            "v98 CUDA Q3 trace requires an exact coarse v97 trial")
    if (trace.coarse_provisional_trial_sha256 != coarse_trial.trial_sha256 or
            trace.coarse_provisional_state_sha256 !=
                coarse_trial.candidate.state_sha256):
        raise CoupledAMRQ3TraceGPUV98Error(
            "v98 CUDA Q3 trace and coarse trial differ")
    face = _face_trial(coarse_trial)
    producer = face.producer_receipt
    ng = fine_execution.fraction_count
    cell = trace.coarse_cell
    if (producer.bed_capacity_kg_m_s.shape[1:] !=
            (ng, coarse_trial.candidate.q.shape[1]) or
            not 0 <= cell < producer.bed_capacity_kg_m_s.shape[2]):
        raise CoupledAMRQ3TraceGPUV98Error(
            "v98 CUDA Q3 coarse producer dimensions differ")
    vector = np.repeat(
        np.asarray(producer.bed_capacity_kg_m_s[:, :, cell])[:, :, None],
        2, axis=2)
    taurat = np.repeat(
        np.asarray(producer.producer_diagnostics[2])[:, cell, None],
        2, axis=1)
    hiding = np.ones((ng, 2), dtype=np.float64)
    try:
        import cupy as cp
        availability_cell = np.ascontiguousarray(
            cp.asnumpy(face.accepted_cell_availability[:, cell]))
    except Exception as error:  # pragma: no cover - host dependent
        raise CoupledAMRQ3TraceGPUV98Error(
            "v98 CUDA Q3 trace resident availability readback failed") \
            from error
    availability = np.repeat(availability_cell[:, None], 2, axis=1)
    arrays = (
        _readonly(vector, (2, ng, 2), "capacity vector"),
        _readonly(taurat, (ng, 2), "taurat", nonnegative=True),
        _readonly(hiding, (ng, 2), "hiding", nonnegative=True),
        _readonly(availability, (ng, 2), "availability", nonnegative=True),
    )
    if np.any(arrays[3] > 1.0):
        raise CoupledAMRQ3TraceGPUV98Error(
            "v98 CUDA Q3 coarse availability exceeds one")
    placeholder = CoupledAMRQ3TraceGPUV98(
        trace.receipt_sha256, coarse_trial.trial_sha256,
        face.trial_sha256, producer.receipt_sha256,
        fine_execution.fingerprint, cell, tuple(trace.fine_faces),
        *arrays, 0, 1, 0, 0, BACKEND, "")
    result = CoupledAMRQ3TraceGPUV98(
        **{**placeholder.__dict__,
           "receipt_sha256": _hash(_payload(placeholder))})
    validate_coupled_amr_q3_trace_gpu_v98(
        result, trace, coarse_trial, fine_execution)
    return result


def validate_coupled_amr_q3_trace_gpu_v98(
    value: CoupledAMRQ3TraceGPUV98,
    trace: CoupledAMRInternalTraceGPUV98,
    coarse_trial: CoupledWetDryGPUTrialV97,
    fine_execution: CoupledWetDryGPUExecutionV97,
) -> None:
    if (type(value) is not CoupledAMRQ3TraceGPUV98 or
            type(trace) is not CoupledAMRInternalTraceGPUV98 or
            type(coarse_trial) is not CoupledWetDryGPUTrialV97 or
            type(fine_execution) is not CoupledWetDryGPUExecutionV97):
        raise CoupledAMRQ3TraceGPUV98Error(
            "v98 CUDA Q3 trace type differs")
    validate_coupled_amr_internal_trace_structure_gpu_v98(
        trace, fine_execution)
    face = _face_trial(coarse_trial)
    producer = face.producer_receipt
    ng = fine_execution.fraction_count
    arrays = (
        (value.interface_vector_kg_m_s, (2, ng, 2)),
        (value.interface_taurat, (ng, 2)),
        (value.interface_hiding, (ng, 2)),
        (value.interface_availability, (ng, 2)),
    )
    if (value.internal_trace_receipt_sha256 != trace.receipt_sha256 or
            value.coarse_trial_sha256 != coarse_trial.trial_sha256 or
            value.coarse_face_trial_sha256 != face.trial_sha256 or
            value.coarse_producer_receipt_sha256 !=
                producer.receipt_sha256 or
            value.fine_execution_fingerprint != fine_execution.fingerprint or
            value.coarse_cell != trace.coarse_cell or
            value.interface_faces != tuple(trace.fine_faces) or
            value.constitutive_recompute_count != 0 or
            value.resident_readback_count != 1 or
            value.physical_boundary_owner_count != 0 or
            value.accepted_clock_increment_count != 0 or
            value.backend != BACKEND or
            value.receipt_sha256 != _hash(_payload(value))):
        raise CoupledAMRQ3TraceGPUV98Error(
            "v98 CUDA Q3 trace identity differs")
    for array, shape in arrays:
        if (array.shape != shape or array.flags.writeable or
                not np.isfinite(array).all()):
            raise CoupledAMRQ3TraceGPUV98Error(
                "v98 CUDA Q3 trace array identity differs")
    cell = value.coarse_cell
    try:
        import cupy as cp
        resident_availability = np.repeat(
            cp.asnumpy(face.accepted_cell_availability[:, cell, None]),
            2, axis=1)
    except Exception as error:  # pragma: no cover - host dependent
        raise CoupledAMRQ3TraceGPUV98Error(
            "v98 CUDA Q3 trace validation readback failed") from error
    if (not np.array_equal(
            value.interface_vector_kg_m_s,
            np.repeat(
                producer.bed_capacity_kg_m_s[:, :, cell, None],
                2, axis=2)) or
            not np.array_equal(
                value.interface_taurat,
                np.repeat(
                    producer.producer_diagnostics[2][:, cell, None],
                    2, axis=1)) or
            not np.array_equal(
                value.interface_availability, resident_availability) or
            np.any(value.interface_hiding != 1.0) or
            np.any(value.interface_availability < 0.0) or
            np.any(value.interface_availability > 1.0)):
        raise CoupledAMRQ3TraceGPUV98Error(
            "v98 CUDA Q3 trace producer reconstruction differs")


__all__ = [
    "BACKEND", "CoupledAMRQ3TraceGPUV98",
    "CoupledAMRQ3TraceGPUV98Error", "SCOPE_ID",
    "resolve_coupled_amr_q3_trace_gpu_v98",
    "validate_coupled_amr_q3_trace_gpu_v98",
]
