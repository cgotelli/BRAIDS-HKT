"""Combined suspended/T/S trace derived from a typed v98 AMR trace."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np

from .coupled_amr_trace_cpu_v98 import (
    CoupledAMRInternalTraceCPUV98,
    validate_coupled_amr_internal_trace_structure_cpu_v98,
)
from .coupled_wetdry_owner_cpu_v97 import (
    CoupledWetDryCPUExecutionV97,
    _v92_execution,
)
from .transported_fluid_settling_v92 import (
    FLUID_SALINITY,
    FLUID_TEMPERATURE,
    TEMPERATURE_FLOOR_C,
)


SCHEMA = 1
SCOPE_ID = "braids.coupled_amr.scalar_trace.cpu.v98"


class CoupledAMRScalarTraceCPUV98Error(ValueError):
    """The combined scalar trace is invalid or stale."""


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


@dataclass(frozen=True)
class CoupledAMRScalarTraceCPUV98:
    internal_trace_receipt_sha256: str
    fine_execution_fingerprint: str
    fraction_count: int
    active_fluid_lanes: tuple[int, ...]
    lane_semantics: tuple[str, ...]
    interface_faces: tuple[int, int]
    substep_concentration: np.ndarray
    internal_interface_owner_count: int
    physical_boundary_owner_count: int
    boundary_memory_consumer_count: int
    accepted_clock_increment_count: int
    receipt_sha256: str


def _payload(value: CoupledAMRScalarTraceCPUV98) -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "internal_trace": value.internal_trace_receipt_sha256,
        "fine_execution": value.fine_execution_fingerprint,
        "fraction_count": value.fraction_count,
        "active_fluid_lanes": list(value.active_fluid_lanes),
        "lane_semantics": list(value.lane_semantics),
        "interface_faces": list(value.interface_faces),
        "substep_concentration_sha256": _digest(
            value.substep_concentration),
        "internal_interface_owner_count":
            value.internal_interface_owner_count,
        "physical_boundary_owner_count": value.physical_boundary_owner_count,
        "boundary_memory_consumer_count": value.boundary_memory_consumer_count,
        "accepted_clock_increment_count":
            value.accepted_clock_increment_count,
        "temperature_representation": "affine_above_floor_c",
        "internal_face_only": True,
    }


def resolve_coupled_amr_scalar_trace_cpu_v98(
    trace: CoupledAMRInternalTraceCPUV98,
    fine_execution: CoupledWetDryCPUExecutionV97,
) -> CoupledAMRScalarTraceCPUV98:
    validate_coupled_amr_internal_trace_structure_cpu_v98(
        trace, fine_execution)
    v92 = _v92_execution(fine_execution.active_execution)
    active = tuple(map(int, np.flatnonzero(v92.transported_mask)))
    ng = fine_execution.fraction_count
    lanes = ng+len(active)
    concentration = np.zeros((2, lanes, 2), dtype=np.float64)
    semantics = [f"suspended_fraction_{index}" for index in range(ng)]
    semantics.extend(
        "temperature_affine_c" if lane == FLUID_TEMPERATURE else
        "salinity_ppt" for lane in active)
    for step in range(2):
        for slot in range(2):
            depth = float(trace.trace_q[step, 0, slot])
            wet = bool(trace.trace_wet_mask[step, slot])
            if wet:
                if depth <= 0.0:
                    raise CoupledAMRScalarTraceCPUV98Error(
                        "v98 wet scalar trace has nonpositive depth")
                concentration[step, :ng, slot] = (
                    trace.trace_suspended_kg_m2[step, :, slot]/depth)
                for position, lane in enumerate(active):
                    value = trace.trace_fluid_integral[step, lane, slot]/depth
                    if lane == FLUID_TEMPERATURE:
                        value -= TEMPERATURE_FLOOR_C
                    concentration[step, ng+position, slot] = value
            elif (np.any(trace.trace_suspended_kg_m2[step, :, slot] != 0.0)
                  or np.any(trace.trace_fluid_integral[step, :, slot] != 0.0)):
                raise CoupledAMRScalarTraceCPUV98Error(
                    "v98 dry scalar trace retains transported inventory")
    if (not np.isfinite(concentration).all() or
            np.any(concentration < 0.0)):
        raise CoupledAMRScalarTraceCPUV98Error(
            "v98 combined scalar trace is outside its affine domain")
    concentration.setflags(write=False)
    placeholder = CoupledAMRScalarTraceCPUV98(
        trace.receipt_sha256, fine_execution.fingerprint, ng, active,
        tuple(semantics), tuple(trace.fine_faces), concentration,
        1, 0, 0, 0, "")
    result = CoupledAMRScalarTraceCPUV98(
        **{**placeholder.__dict__,
           "receipt_sha256": _hash(_payload(placeholder))})
    validate_coupled_amr_scalar_trace_cpu_v98(result, trace, fine_execution)
    return result


def validate_coupled_amr_scalar_trace_cpu_v98(
    value: CoupledAMRScalarTraceCPUV98,
    trace: CoupledAMRInternalTraceCPUV98,
    fine_execution: CoupledWetDryCPUExecutionV97,
) -> None:
    if (type(value) is not CoupledAMRScalarTraceCPUV98 or
            type(trace) is not CoupledAMRInternalTraceCPUV98 or
            type(fine_execution) is not CoupledWetDryCPUExecutionV97):
        raise CoupledAMRScalarTraceCPUV98Error(
            "v98 scalar trace type differs")
    validate_coupled_amr_internal_trace_structure_cpu_v98(
        trace, fine_execution)
    v92 = _v92_execution(fine_execution.active_execution)
    active = tuple(map(int, np.flatnonzero(v92.transported_mask)))
    lanes = fine_execution.fraction_count+len(active)
    array = np.asarray(value.substep_concentration)
    if (value.internal_trace_receipt_sha256 != trace.receipt_sha256 or
            value.fine_execution_fingerprint != fine_execution.fingerprint or
            value.fraction_count != fine_execution.fraction_count or
            value.active_fluid_lanes != active or
            value.interface_faces != tuple(trace.fine_faces) or
            array.shape != (2, lanes, 2) or array.flags.writeable or
            not np.isfinite(array).all() or np.any(array < 0.0) or
            len(value.lane_semantics) != lanes or
            value.internal_interface_owner_count != 1 or
            value.physical_boundary_owner_count != 0 or
            value.boundary_memory_consumer_count != 0 or
            value.accepted_clock_increment_count != 0 or
            value.receipt_sha256 != _hash(_payload(value))):
        raise CoupledAMRScalarTraceCPUV98Error(
            "v98 scalar trace identity differs")
    # Independently recover every lane from the conserved trace image.
    rebuilt = np.zeros_like(array)
    for step in range(2):
        for slot in range(2):
            depth = float(trace.trace_q[step, 0, slot])
            if trace.trace_wet_mask[step, slot]:
                rebuilt[step, :value.fraction_count, slot] = (
                    trace.trace_suspended_kg_m2[step, :, slot]/depth)
                for position, lane in enumerate(active):
                    rebuilt[step, value.fraction_count+position, slot] = (
                        trace.trace_fluid_integral[step, lane, slot]/depth-
                        (TEMPERATURE_FLOOR_C
                         if lane == FLUID_TEMPERATURE else 0.0))
    if not np.array_equal(array, rebuilt):
        raise CoupledAMRScalarTraceCPUV98Error(
            "v98 scalar trace conserved reconstruction differs")


__all__ = [
    "CoupledAMRScalarTraceCPUV98",
    "CoupledAMRScalarTraceCPUV98Error",
    "resolve_coupled_amr_scalar_trace_cpu_v98",
    "validate_coupled_amr_scalar_trace_cpu_v98",
]
