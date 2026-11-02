"""Typed actual-CUDA coarse traces for the bounded Q4/v98 AMR seam.

The trace is formed from conserved coarse parent/provisional images at the
quarter and three-quarter points of a 2:1 coarse step.  It is an internal
coupling image, never a physical boundary state.  Temperature, salinity,
water density, mixture density, bed elevation and wetness are derived after
interpolation; density itself is never interpolated.

This module performs no CUDA numerical work.  It authenticates the immutable
host image that :mod:`braids_gpu.coupled_amr_fwave_v98` uploads once and
consumes from resident CUDA kernels during every true nonlinear residual.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Sequence

import numpy as np

from braids_core.sediment.densin_fwave_v85 import (
    mixture_density_kg_m3_v85,
)
from braids_core.sediment.dynamic_fluid_settling_v88 import (
    resolve_water_density_v88,
)
from braids_core.sediment.transported_fluid_settling_v92 import (
    MODE_TRANSPORTED,
)
from .coupled_wetdry_v97 import (
    CoupledWetDryGPUExecutionV97,
    CoupledWetDryGPUStateV97,
    CoupledWetDryGPUTrialV97,
    ResidentCoupledWetDryLifecycleV97,
    _grid,
    _v92_execution,
    _v92_state,
)


SCHEMA = 1
SCOPE_ID = "braids.q4.coupled_amr.internal_trace.actual_cuda.v98"
BACKEND = "actual_cuda"
ALPHAS = (0.25, 0.75)
SUBSTEP_COUNT = 2
INTERFACE_FACE_COUNT = 2


class CoupledAMRTraceGPUV98Error(ValueError):
    """The actual-CUDA coarse/fine trace is invalid or stale."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError, OverflowError) as error:
        raise CoupledAMRTraceGPUV98Error(
            "v98 CUDA trace identity is not canonical JSON") from error


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _digest(*values: object) -> str:
    digest = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(value)
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(_canonical(list(array.shape)))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _readonly(value: object, shape: tuple[int, ...], name: str, *,
              nonnegative: bool = False, integer: bool = False) -> np.ndarray:
    raw = np.asarray(value)
    if (raw.shape != shape or raw.dtype.kind not in "fiu" or
            not np.isfinite(raw).all() or
            (integer and np.any(raw != np.floor(raw))) or
            (nonnegative and np.any(raw < 0.0))):
        raise CoupledAMRTraceGPUV98Error(
            f"v98 CUDA internal trace {name} differs")
    result = np.array(
        raw, dtype=np.int32 if integer else np.float64,
        order="C", copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class CoupledAMRInternalTraceGPUV98:
    coarse_execution_fingerprint: str
    fine_execution_fingerprint: str
    coarse_parent_state_sha256: str
    coarse_provisional_trial_sha256: str
    coarse_provisional_state_sha256: str
    coarse_cell: int
    fine_faces: tuple[int, int]
    substep_alpha: np.ndarray
    substep_time_s: np.ndarray
    trace_q: np.ndarray
    trace_suspended_kg_m2: np.ndarray
    trace_bed_kg_m2: np.ndarray
    trace_fluid_integral: np.ndarray
    trace_bed_elevation_m: np.ndarray
    trace_water_density_kg_m3: np.ndarray
    trace_mixture_density_kg_m3: np.ndarray
    trace_wet_mask: np.ndarray
    physical_boundary_owner_count: int
    boundary_memory_consumer_count: int
    accepted_clock_increment_count: int
    density_interpolation_count: int
    eos_derivation_count: int
    backend: str
    receipt_sha256: str


def _payload(value: CoupledAMRInternalTraceGPUV98) -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "backend": value.backend,
        "coarse_execution": value.coarse_execution_fingerprint,
        "fine_execution": value.fine_execution_fingerprint,
        "coarse_parent": value.coarse_parent_state_sha256,
        "coarse_provisional_trial":
            value.coarse_provisional_trial_sha256,
        "coarse_provisional_state":
            value.coarse_provisional_state_sha256,
        "coarse_cell": value.coarse_cell,
        "fine_faces": list(value.fine_faces),
        "arrays_sha256": _digest(
            value.substep_alpha, value.substep_time_s, value.trace_q,
            value.trace_suspended_kg_m2, value.trace_bed_kg_m2,
            value.trace_fluid_integral, value.trace_bed_elevation_m,
            value.trace_water_density_kg_m3,
            value.trace_mixture_density_kg_m3, value.trace_wet_mask),
        "physical_boundary_owner_count":
            value.physical_boundary_owner_count,
        "boundary_memory_consumer_count":
            value.boundary_memory_consumer_count,
        "accepted_clock_increment_count":
            value.accepted_clock_increment_count,
        "density_interpolation_count": value.density_interpolation_count,
        "eos_derivation_count": value.eos_derivation_count,
        "interpolation": "conserved_coarse_parent_to_provisional",
        "substep_alpha_semantics": "coarse_step_quarter_three_quarter",
        "derived_after_interpolation": [
            "bed_elevation", "temperature", "salinity",
            "water_density", "mixture_density", "wet_mask"],
        "internal_face_only": True,
        "physical_boundary_surrogate": False,
        "automatic_cpu_fallback": False,
    }


def _compatible_executions(
    coarse: CoupledWetDryGPUExecutionV97,
    fine: CoupledWetDryGPUExecutionV97,
) -> None:
    if (type(coarse) is not CoupledWetDryGPUExecutionV97 or
            type(fine) is not CoupledWetDryGPUExecutionV97):
        raise CoupledAMRTraceGPUV98Error(
            "v98 CUDA trace requires exact v97 GPU executions")
    cv92, fv92 = (_v92_execution(coarse.active_execution),
                  _v92_execution(fine.active_execution))
    cg, fg = _grid(coarse.active_execution), _grid(fine.active_execution)
    if (coarse.fraction_count != fine.fraction_count or
            cg.profile.sediment_catalog_id !=
                fg.profile.sediment_catalog_id or
            tuple(cg.profile.sediment_d50_m) !=
                tuple(fg.profile.sediment_d50_m) or
            not np.array_equal(cg.profile.sediment_density_kg_m3,
                               fg.profile.sediment_density_kg_m3) or
            cv92.temperature_mode != fv92.temperature_mode or
            cv92.salinity_mode != fv92.salinity_mode or
            cv92.equilibrium_temperature_c !=
                fv92.equilibrium_temperature_c or
            cv92.equilibrium_salinity_ppt !=
                fv92.equilibrium_salinity_ppt or
            cv92.density_eos != fv92.density_eos or
            cg.profile.water_density_kg_m3 !=
                fg.profile.water_density_kg_m3):
        raise CoupledAMRTraceGPUV98Error(
            "v98 CUDA coarse/fine transported physics differ")


def _resolve(
    coarse_execution: CoupledWetDryGPUExecutionV97,
    coarse_parent: CoupledWetDryGPUStateV97,
    coarse_runtime: ResidentCoupledWetDryLifecycleV97,
    coarse_provisional: CoupledWetDryGPUTrialV97,
    fine_execution: CoupledWetDryGPUExecutionV97, *,
    coarse_cell: int,
    fine_faces: Sequence[int],
    validate_source: bool,
) -> CoupledAMRInternalTraceGPUV98:
    _compatible_executions(coarse_execution, fine_execution)
    if (type(coarse_parent) is not CoupledWetDryGPUStateV97 or
            type(coarse_runtime) is not ResidentCoupledWetDryLifecycleV97 or
            type(coarse_provisional) is not CoupledWetDryGPUTrialV97 or
            coarse_runtime.execution is not coarse_execution):
        raise CoupledAMRTraceGPUV98Error(
            "v98 CUDA trace parent/runtime/provisional type differs")
    if validate_source:
        coarse_runtime._validate_trial(coarse_parent, coarse_provisional)
    if (type(coarse_cell) is not int or
            not 0 <= coarse_cell < coarse_execution.cell_count):
        raise CoupledAMRTraceGPUV98Error(
            "v98 CUDA trace coarse cell is outside the execution")
    faces = tuple(fine_faces)
    fg = _grid(fine_execution.active_execution)
    if (len(faces) != INTERFACE_FACE_COUNT or len(set(faces)) != 2 or
            any(type(face) is not int or
                not 0 <= face < fine_execution.face_count
                for face in faces)):
        raise CoupledAMRTraceGPUV98Error(
            "v98 CUDA trace requires two distinct fine interface faces")
    for face in faces:
        if (int(fg.neighbour[face]) >= 0 or
                int(fg.boundary_index[face]) >= 0):
            raise CoupledAMRTraceGPUV98Error(
                "v98 CUDA trace face is not an internal negative-neighbour "
                "coupling face")

    cv92_execution = _v92_execution(coarse_execution.active_execution)
    parent = _v92_state(coarse_parent.active_state)
    candidate = _v92_state(coarse_provisional.candidate.active_state)
    parent_bed = np.asarray(
        coarse_parent.active_state.base_state.bed_mass_kg_m2)
    candidate_bed = np.asarray(
        coarse_provisional.candidate.active_state.base_state.
        bed_mass_kg_m2)
    q0 = np.asarray(parent.q[:, coarse_cell])
    q1 = np.asarray(candidate.q[:, coarse_cell])
    suspended0 = np.asarray(
        parent.suspended_state.suspended_mass_kg_m2[:, coarse_cell])
    suspended1 = np.asarray(
        candidate.suspended_state.suspended_mass_kg_m2[:, coarse_cell])
    bed0 = np.asarray(parent_bed[:, coarse_cell])
    bed1 = np.asarray(candidate_bed[:, coarse_cell])
    fluid0 = np.asarray((parent.h_temperature_c_m[coarse_cell],
                         parent.h_salinity_ppt_m[coarse_cell]))
    fluid1 = np.asarray((candidate.h_temperature_c_m[coarse_cell],
                         candidate.h_salinity_ppt_m[coarse_cell]))
    alpha = _readonly(ALPHAS, (2,), "substep alpha")
    dt = float(coarse_provisional.hydraulic_trial.dt_s)
    time = _readonly(
        coarse_parent.time_s+alpha*dt, (2,), "substep time")
    ng = coarse_execution.fraction_count
    trace_q = np.empty((2, 3, 2), dtype=np.float64)
    trace_suspended = np.empty((2, ng, 2), dtype=np.float64)
    trace_bed = np.empty((2, ng, 2), dtype=np.float64)
    trace_fluid = np.empty((2, 2, 2), dtype=np.float64)
    trace_z = np.empty((2, 2), dtype=np.float64)
    trace_water = np.empty((2, 2), dtype=np.float64)
    trace_mixture = np.empty((2, 2), dtype=np.float64)
    trace_wet = np.empty((2, 2), dtype=np.int32)
    cg = _grid(coarse_execution.active_execution)
    dry = float(cg.dry_tolerance_m)
    for step, weight in enumerate(ALPHAS):
        q = (1.0-weight)*q0+weight*q1
        suspended = (1.0-weight)*suspended0+weight*suspended1
        bed = (1.0-weight)*bed0+weight*bed1
        fluid = (1.0-weight)*fluid0+weight*fluid1
        if (not np.isfinite(q).all() or q[0] < 0.0 or
                np.any(suspended < 0.0) or np.any(bed < 0.0) or
                not np.isfinite(fluid).all() or fluid[1] < 0.0):
            raise CoupledAMRTraceGPUV98Error(
                "v98 CUDA interpolated trace left its physical domain")
        wet = bool(q[0] > dry)
        if not wet and (np.any(q[1:] != 0.0) or
                        np.any(suspended != 0.0) or
                        np.any(fluid != 0.0)):
            raise CoupledAMRTraceGPUV98Error(
                "v98 CUDA dry trace retains transported inventory")
        if wet:
            temperature = (float(fluid[0]/q[0])
                           if cv92_execution.temperature_mode ==
                           MODE_TRANSPORTED else
                           cv92_execution.equilibrium_temperature_c)
            salinity = (float(fluid[1]/q[0])
                        if cv92_execution.salinity_mode ==
                        MODE_TRANSPORTED else
                        cv92_execution.equilibrium_salinity_ppt)
            concentration = suspended[:, None]/q[0]
        else:
            temperature = cv92_execution.equilibrium_temperature_c
            salinity = cv92_execution.equilibrium_salinity_ppt
            concentration = np.zeros((ng, 1), dtype=np.float64)
        water = resolve_water_density_v88(
            np.asarray((temperature,)), np.asarray((salinity,)),
            eos=cv92_execution.density_eos,
            constituents_active=cv92_execution.constituents_active,
            reference_water_density_kg_m3=
                cg.profile.water_density_kg_m3)
        mixture = mixture_density_kg_m3_v85(
            concentration, water,
            cg.profile.sediment_density_kg_m3)
        elevation = (
            coarse_execution.initial_bed_elevation_m[coarse_cell]+np.sum(
                (bed-coarse_execution.reference_bed_mass_kg_m2[
                    :, coarse_cell])/
                coarse_execution.dry_bulk_density_kg_m3))
        for slot in range(2):
            trace_q[step, :, slot] = q
            trace_suspended[step, :, slot] = suspended
            trace_bed[step, :, slot] = bed
            trace_fluid[step, :, slot] = fluid
            trace_z[step, slot] = elevation
            trace_water[step, slot] = water[0]
            trace_mixture[step, slot] = mixture[0]
            trace_wet[step, slot] = int(wet)
    arrays = (
        _readonly(trace_q, (2, 3, 2), "q"),
        _readonly(trace_suspended, (2, ng, 2), "suspended",
                  nonnegative=True),
        _readonly(trace_bed, (2, ng, 2), "bed", nonnegative=True),
        _readonly(trace_fluid, (2, 2, 2), "fluid"),
        _readonly(trace_z, (2, 2), "bed elevation"),
        _readonly(trace_water, (2, 2), "water density",
                  nonnegative=True),
        _readonly(trace_mixture, (2, 2), "mixture density",
                  nonnegative=True),
        _readonly(trace_wet, (2, 2), "wet mask", integer=True),
    )
    placeholder = CoupledAMRInternalTraceGPUV98(
        coarse_execution.fingerprint, fine_execution.fingerprint,
        coarse_parent.state_sha256, coarse_provisional.trial_sha256,
        coarse_provisional.candidate.state_sha256, coarse_cell, faces,
        alpha, time, *arrays, 0, 0, 0, 0, 4, BACKEND, "")
    return CoupledAMRInternalTraceGPUV98(
        **{**placeholder.__dict__,
           "receipt_sha256": _hash(_payload(placeholder))})


def resolve_coupled_amr_internal_trace_gpu_v98(
    coarse_execution: CoupledWetDryGPUExecutionV97,
    coarse_parent: CoupledWetDryGPUStateV97,
    coarse_runtime: ResidentCoupledWetDryLifecycleV97,
    coarse_provisional: CoupledWetDryGPUTrialV97,
    fine_execution: CoupledWetDryGPUExecutionV97, *,
    coarse_cell: int,
    fine_faces: Sequence[int],
) -> CoupledAMRInternalTraceGPUV98:
    result = _resolve(
        coarse_execution, coarse_parent, coarse_runtime,
        coarse_provisional, fine_execution, coarse_cell=coarse_cell,
        fine_faces=fine_faces, validate_source=True)
    validate_coupled_amr_internal_trace_gpu_v98(
        result, coarse_execution, coarse_parent, coarse_runtime,
        coarse_provisional, fine_execution)
    return result


def validate_coupled_amr_internal_trace_structure_gpu_v98(
    value: CoupledAMRInternalTraceGPUV98,
    fine_execution: CoupledWetDryGPUExecutionV97,
) -> None:
    if (type(value) is not CoupledAMRInternalTraceGPUV98 or
            type(fine_execution) is not CoupledWetDryGPUExecutionV97):
        raise CoupledAMRTraceGPUV98Error(
            "v98 CUDA internal trace structure type differs")
    _compatible_executions(fine_execution, fine_execution)
    ng = fine_execution.fraction_count
    names_shapes = (
        ("substep_alpha", (2,)), ("substep_time_s", (2,)),
        ("trace_q", (2, 3, 2)),
        ("trace_suspended_kg_m2", (2, ng, 2)),
        ("trace_bed_kg_m2", (2, ng, 2)),
        ("trace_fluid_integral", (2, 2, 2)),
        ("trace_bed_elevation_m", (2, 2)),
        ("trace_water_density_kg_m3", (2, 2)),
        ("trace_mixture_density_kg_m3", (2, 2)),
        ("trace_wet_mask", (2, 2)),
    )
    if (value.fine_execution_fingerprint != fine_execution.fingerprint or
            value.backend != BACKEND or
            value.receipt_sha256 != _hash(_payload(value)) or
            tuple(value.substep_alpha) != ALPHAS or
            np.any(np.diff(value.substep_time_s) <= 0.0) or
            value.physical_boundary_owner_count != 0 or
            value.boundary_memory_consumer_count != 0 or
            value.accepted_clock_increment_count != 0 or
            value.density_interpolation_count != 0 or
            value.eos_derivation_count != 4):
        raise CoupledAMRTraceGPUV98Error(
            "v98 CUDA internal trace structure identity differs")
    for name, shape in names_shapes:
        array = np.asarray(getattr(value, name))
        if (array.shape != shape or array.flags.writeable or
                not np.isfinite(array).all()):
            raise CoupledAMRTraceGPUV98Error(
                f"v98 CUDA internal trace {name} structure differs")
    if (np.any(value.trace_q[:, 0] < 0.0) or
            np.any(value.trace_suspended_kg_m2 < 0.0) or
            np.any(value.trace_bed_kg_m2 < 0.0) or
            np.any(value.trace_water_density_kg_m3 <= 0.0) or
            np.any(value.trace_mixture_density_kg_m3 <
                   value.trace_water_density_kg_m3) or
            np.any((value.trace_wet_mask != 0) &
                   (value.trace_wet_mask != 1))):
        raise CoupledAMRTraceGPUV98Error(
            "v98 CUDA internal trace physical structure differs")
    dry = value.trace_wet_mask == 0
    if (np.any(value.trace_q[:, 0][dry] != 0.0) or
            np.any(np.moveaxis(value.trace_q[:, 1:], 1, -1)[dry] != 0.0) or
            np.any(np.moveaxis(
                value.trace_suspended_kg_m2, 1, -1)[dry] != 0.0) or
            np.any(np.moveaxis(
                value.trace_fluid_integral, 1, -1)[dry] != 0.0)):
        raise CoupledAMRTraceGPUV98Error(
            "v98 CUDA dry internal trace retains transported inventory")
    fg = _grid(fine_execution.active_execution)
    if (len(value.fine_faces) != INTERFACE_FACE_COUNT or
            len(set(value.fine_faces)) != INTERFACE_FACE_COUNT):
        raise CoupledAMRTraceGPUV98Error(
            "v98 CUDA internal trace face inventory differs")
    for face in value.fine_faces:
        if (type(face) is not int or
                not 0 <= face < fine_execution.face_count or
                int(fg.neighbour[face]) >= 0 or
                int(fg.boundary_index[face]) >= 0):
            raise CoupledAMRTraceGPUV98Error(
                "v98 CUDA internal trace topology differs")


def validate_coupled_amr_internal_trace_gpu_v98(
    value: CoupledAMRInternalTraceGPUV98,
    coarse_execution: CoupledWetDryGPUExecutionV97,
    coarse_parent: CoupledWetDryGPUStateV97,
    coarse_runtime: ResidentCoupledWetDryLifecycleV97,
    coarse_provisional: CoupledWetDryGPUTrialV97,
    fine_execution: CoupledWetDryGPUExecutionV97,
) -> None:
    validate_coupled_amr_internal_trace_structure_gpu_v98(
        value, fine_execution)
    rebuilt = _resolve(
        coarse_execution, coarse_parent, coarse_runtime,
        coarse_provisional, fine_execution,
        coarse_cell=value.coarse_cell, fine_faces=value.fine_faces,
        validate_source=True)
    if (value.receipt_sha256 != rebuilt.receipt_sha256 or
            value.coarse_execution_fingerprint !=
                coarse_execution.fingerprint or
            value.coarse_parent_state_sha256 !=
                coarse_parent.state_sha256 or
            value.coarse_provisional_trial_sha256 !=
                coarse_provisional.trial_sha256 or
            value.coarse_provisional_state_sha256 !=
                coarse_provisional.candidate.state_sha256 or
            any(not np.array_equal(getattr(value, name),
                                   getattr(rebuilt, name))
                for name in (
                    "substep_alpha", "substep_time_s", "trace_q",
                    "trace_suspended_kg_m2", "trace_bed_kg_m2",
                    "trace_fluid_integral", "trace_bed_elevation_m",
                    "trace_water_density_kg_m3",
                    "trace_mixture_density_kg_m3", "trace_wet_mask"))):
        raise CoupledAMRTraceGPUV98Error(
            "v98 CUDA rebuilt internal trace differs")


def method_identity_coupled_amr_trace_gpu_v98() -> dict[str, object]:
    return {
        "scope_id": SCOPE_ID,
        "backend": BACKEND,
        "substep_alpha": list(ALPHAS),
        "conserved_interpolation": True,
        "density_derived_after_interpolation": True,
        "physical_boundary_owner_count": 0,
        "boundary_memory_consumer_count": 0,
        "accepted_clock_increment_count": 0,
        "automatic_cpu_fallback": False,
        "production_admission": False,
    }


__all__ = [
    "ALPHAS", "BACKEND", "CoupledAMRInternalTraceGPUV98",
    "CoupledAMRTraceGPUV98Error", "INTERFACE_FACE_COUNT", "SCOPE_ID",
    "method_identity_coupled_amr_trace_gpu_v98",
    "resolve_coupled_amr_internal_trace_gpu_v98",
    "validate_coupled_amr_internal_trace_gpu_v98",
    "validate_coupled_amr_internal_trace_structure_gpu_v98",
]
