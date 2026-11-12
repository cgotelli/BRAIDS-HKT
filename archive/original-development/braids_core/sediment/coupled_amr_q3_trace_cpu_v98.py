"""Authenticated coarse Q3 constitutive trace for the minimum v98 AMR gate."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np

from .active_suspended_control_v94 import CPUQ3V83ActiveFaceTrialV94
from .coupled_amr_trace_cpu_v98 import (
    CoupledAMRInternalTraceCPUV98,
    validate_coupled_amr_internal_trace_structure_cpu_v98,
)
from .coupled_wetdry_owner_cpu_v97 import (
    CoupledWetDryCPUExecutionV97,
    CoupledWetDryCPUTrialV97,
)


SCHEMA = 1
SCOPE_ID = "braids.coupled_amr.q3_trace.cpu.v98"


class CoupledAMRQ3TraceCPUV98Error(ValueError):
    """The coarse Q3 trace is invalid or stale."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False).encode("ascii")


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
        raise CoupledAMRQ3TraceCPUV98Error(
            f"v98 Q3 trace {name} differs")
    result = np.array(raw, dtype=np.float64, order="C", copy=True)
    result.setflags(write=False)
    return result


def _face_trial(trial: CoupledWetDryCPUTrialV97) -> CPUQ3V83ActiveFaceTrialV94:
    try:
        face = trial.active_trial.base_v93_trial.q3_open_trial.face_trial
    except AttributeError as error:
        raise CoupledAMRQ3TraceCPUV98Error(
            "v98 Q3 trace lacks the accepted coarse face trial") from error
    if type(face) is not CPUQ3V83ActiveFaceTrialV94:
        raise CoupledAMRQ3TraceCPUV98Error(
            "v98 Q3 trace coarse face-trial type differs")
    return face


@dataclass(frozen=True)
class CoupledAMRQ3TraceCPUV98:
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
    physical_boundary_owner_count: int
    accepted_clock_increment_count: int
    receipt_sha256: str


def _payload(value: CoupledAMRQ3TraceCPUV98) -> dict[str, object]:
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
        "physical_boundary_owner_count": value.physical_boundary_owner_count,
        "accepted_clock_increment_count": value.accepted_clock_increment_count,
        "source_order": "reuse_current_coarse_v83_then_fine_face_q3",
        "cumulative_external_stock_limit":
            "macro_owner_must_reject_negative_corrected_bed",
    }


def resolve_coupled_amr_q3_trace_cpu_v98(
    trace: CoupledAMRInternalTraceCPUV98,
    coarse_trial: CoupledWetDryCPUTrialV97,
    fine_execution: CoupledWetDryCPUExecutionV97,
) -> CoupledAMRQ3TraceCPUV98:
    validate_coupled_amr_internal_trace_structure_cpu_v98(
        trace, fine_execution)
    if type(coarse_trial) is not CoupledWetDryCPUTrialV97:
        raise CoupledAMRQ3TraceCPUV98Error(
            "v98 Q3 trace requires an exact coarse v97 trial")
    if (trace.coarse_provisional_trial_sha256 != coarse_trial.trial_sha256 or
            trace.coarse_provisional_state_sha256 !=
                coarse_trial.candidate.state_sha256):
        raise CoupledAMRQ3TraceCPUV98Error(
            "v98 Q3 trace and coarse provisional trial differ")
    face = _face_trial(coarse_trial)
    producer = face.producer_receipt
    ng = fine_execution.fraction_count
    cell = trace.coarse_cell
    if (producer.bed_capacity_kg_m_s.shape[1:] !=
            (ng, coarse_trial.candidate.q.shape[1]) or
            not 0 <= cell < producer.bed_capacity_kg_m_s.shape[2]):
        raise CoupledAMRQ3TraceCPUV98Error(
            "v98 Q3 coarse producer dimensions differ")
    vector = np.repeat(
        np.asarray(producer.bed_capacity_kg_m_s[:, :, cell])[:, :, None],
        2, axis=2)
    taurat = np.repeat(
        np.asarray(producer.producer_diagnostics[2])[:, cell, None],
        2, axis=1)
    hiding = np.ones((ng, 2), dtype=np.float64)
    availability = np.repeat(
        np.asarray(face.accepted_cell_availability)[:, cell, None],
        2, axis=1)
    arrays = (
        _readonly(vector, (2, ng, 2), "capacity vector"),
        _readonly(taurat, (ng, 2), "taurat", nonnegative=True),
        _readonly(hiding, (ng, 2), "hiding", nonnegative=True),
        _readonly(availability, (ng, 2), "availability", nonnegative=True),
    )
    if np.any(arrays[3] > 1.0):
        raise CoupledAMRQ3TraceCPUV98Error(
            "v98 Q3 coarse availability exceeds one")
    placeholder = CoupledAMRQ3TraceCPUV98(
        trace.receipt_sha256, coarse_trial.trial_sha256,
        face.trial_sha256, producer.receipt_sha256,
        fine_execution.fingerprint, cell, tuple(trace.fine_faces),
        *arrays, 0, 0, 0, "")
    result = CoupledAMRQ3TraceCPUV98(
        **{**placeholder.__dict__,
           "receipt_sha256": _hash(_payload(placeholder))})
    validate_coupled_amr_q3_trace_cpu_v98(
        result, trace, coarse_trial, fine_execution)
    return result


def validate_coupled_amr_q3_trace_cpu_v98(
    value: CoupledAMRQ3TraceCPUV98,
    trace: CoupledAMRInternalTraceCPUV98,
    coarse_trial: CoupledWetDryCPUTrialV97,
    fine_execution: CoupledWetDryCPUExecutionV97,
) -> None:
    if (type(value) is not CoupledAMRQ3TraceCPUV98 or
            type(trace) is not CoupledAMRInternalTraceCPUV98 or
            type(coarse_trial) is not CoupledWetDryCPUTrialV97 or
            type(fine_execution) is not CoupledWetDryCPUExecutionV97):
        raise CoupledAMRQ3TraceCPUV98Error(
            "v98 Q3 trace type differs")
    validate_coupled_amr_internal_trace_structure_cpu_v98(
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
            value.physical_boundary_owner_count != 0 or
            value.accepted_clock_increment_count != 0 or
            value.receipt_sha256 != _hash(_payload(value))):
        raise CoupledAMRQ3TraceCPUV98Error(
            "v98 Q3 trace identity differs")
    for array, shape in arrays:
        if (array.shape != shape or array.flags.writeable or
                not np.isfinite(array).all()):
            raise CoupledAMRQ3TraceCPUV98Error(
                "v98 Q3 trace array identity differs")
    cell = value.coarse_cell
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
                value.interface_availability,
                np.repeat(
                    face.accepted_cell_availability[:, cell, None],
                    2, axis=1)) or
            np.any(value.interface_hiding != 1.0) or
            np.any(value.interface_availability < 0.0) or
            np.any(value.interface_availability > 1.0)):
        raise CoupledAMRQ3TraceCPUV98Error(
            "v98 Q3 trace producer reconstruction differs")


__all__ = [
    "CoupledAMRQ3TraceCPUV98", "CoupledAMRQ3TraceCPUV98Error",
    "resolve_coupled_amr_q3_trace_cpu_v98",
    "validate_coupled_amr_q3_trace_cpu_v98",
]
