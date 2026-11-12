"""Typed coarse-state traces for the minimum 2:1 CPU v98 AMR gate.

The trace is an internal coupling input, never a physical boundary.  It is
formed from conserved coarse parent/provisional rows at the two fine-substep
midpoints (one quarter and three quarters of the coarse interval).  Bed,
temperature, salinity, water density and mixture density are derived *after*
interpolation; density is never interpolated.

This module only owns trace construction.  The additive native v98 residual
must consume the receipt at every nonlinear evaluation before it becomes AMR
solver evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Sequence

import numpy as np

from .coupled_wetdry_owner_cpu_v97 import (
    CoupledWetDryCPUExecutionV97,
    CoupledWetDryCPUStateV97,
    CoupledWetDryCPUTrialV97,
    _validate_trial as _validate_v97_trial,
    _v92_execution,
    _v92_state,
)
from .densin_fwave_v85 import mixture_density_kg_m3_v85
from .dynamic_fluid_settling_v88 import resolve_water_density_v88
from .transported_fluid_settling_v92 import MODE_TRANSPORTED
from .coupled_amr_geometry_v98 import (
    CartesianAMRPatchGeometryV98,
    CoupledAMRJoinedGeometryV98,
    validate_joined_amr_geometry_v98,
)


SCHEMA = 1
SCOPE_ID = "braids.coupled_amr.internal_trace.cpu.v98"
ALPHAS = (0.25, 0.75)
SUBSTEP_COUNT = 2
INTERFACE_FACE_COUNT = 2
_SHA = re.compile(r"[0-9a-f]{64}\Z")


class CoupledAMRTraceCPUV98Error(ValueError):
    """Fail-closed internal coarse/fine trace error."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


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
        raise CoupledAMRTraceCPUV98Error(
            f"v98 internal trace {name} differs")
    result = np.array(
        raw, dtype=np.int32 if integer else np.float64,
        order="C", copy=True)
    result.setflags(write=False)
    return result


def _grid(execution: CoupledWetDryCPUExecutionV97):
    return (execution.active_execution.base_execution.base_execution.
            base_execution.base_execution.base_execution)


@dataclass(frozen=True)
class CoupledAMRInternalTraceCPUV98:
    coarse_execution_fingerprint: str
    fine_execution_fingerprint: str
    coarse_parent_state_sha256: str
    coarse_provisional_trial_sha256: str
    coarse_provisional_state_sha256: str
    coarse_cell: int
    fine_faces: tuple[int, int]
    joined_geometry_receipt_sha256: str | None
    interface_centre_distance_m: np.ndarray
    physical_boundary_distance_reuse_count: int
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
    receipt_sha256: str


def _payload(value: CoupledAMRInternalTraceCPUV98) -> dict:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "coarse_execution": value.coarse_execution_fingerprint,
        "fine_execution": value.fine_execution_fingerprint,
        "coarse_parent": value.coarse_parent_state_sha256,
        "coarse_provisional_trial":
            value.coarse_provisional_trial_sha256,
        "coarse_provisional_state":
            value.coarse_provisional_state_sha256,
        "coarse_cell": value.coarse_cell,
        "fine_faces": list(value.fine_faces),
        "joined_geometry_receipt_sha256":
            value.joined_geometry_receipt_sha256,
        "interface_centre_distance_sha256":
            _digest(value.interface_centre_distance_m),
        "physical_boundary_distance_reuse_count":
            value.physical_boundary_distance_reuse_count,
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
        "derived_after_interpolation": [
            "bed_elevation", "temperature", "salinity",
            "water_density", "mixture_density", "wet_mask"],
        "internal_face_only": True,
        "physical_boundary_surrogate": False,
    }


def _compatible_executions(
    coarse: CoupledWetDryCPUExecutionV97,
    fine: CoupledWetDryCPUExecutionV97,
) -> None:
    if (type(coarse) is not CoupledWetDryCPUExecutionV97 or
            type(fine) is not CoupledWetDryCPUExecutionV97):
        raise CoupledAMRTraceCPUV98Error(
            "v98 trace requires exact CPU v97 executions")
    c_v92, f_v92 = (_v92_execution(coarse.active_execution),
                    _v92_execution(fine.active_execution))
    cg, fg = _grid(coarse), _grid(fine)
    if (coarse.fraction_count != fine.fraction_count or
            cg.profile.sediment_catalog_id !=
                fg.profile.sediment_catalog_id or
            tuple(cg.profile.sediment_d50_m) !=
                tuple(fg.profile.sediment_d50_m) or
            not np.array_equal(cg.profile.sediment_density_kg_m3,
                               fg.profile.sediment_density_kg_m3) or
            c_v92.temperature_mode != f_v92.temperature_mode or
            c_v92.salinity_mode != f_v92.salinity_mode or
            c_v92.equilibrium_temperature_c !=
                f_v92.equilibrium_temperature_c or
            c_v92.equilibrium_salinity_ppt !=
                f_v92.equilibrium_salinity_ppt or
            c_v92.density_eos != f_v92.density_eos or
            cg.profile.water_density_kg_m3 !=
                fg.profile.water_density_kg_m3):
        raise CoupledAMRTraceCPUV98Error(
            "v98 coarse/fine sediment or transported-fluid physics differ")


def _resolve(
    coarse_execution: CoupledWetDryCPUExecutionV97,
    coarse_parent: CoupledWetDryCPUStateV97,
    coarse_provider: object,
    coarse_provisional: CoupledWetDryCPUTrialV97,
    fine_execution: CoupledWetDryCPUExecutionV97, *,
    coarse_cell: int, fine_faces: Sequence[int],
    joined_geometry: CoupledAMRJoinedGeometryV98 | None,
    coarse_patch_geometry: CartesianAMRPatchGeometryV98 | None,
    fine_patch_geometry: CartesianAMRPatchGeometryV98 | None,
    validate_source: bool,
) -> CoupledAMRInternalTraceCPUV98:
    _compatible_executions(coarse_execution, fine_execution)
    if (type(coarse_parent) is not CoupledWetDryCPUStateV97 or
            type(coarse_provisional) is not CoupledWetDryCPUTrialV97):
        raise CoupledAMRTraceCPUV98Error(
            "v98 trace parent/provisional type differs")
    if validate_source:
        _validate_v97_trial(
            coarse_execution, coarse_parent, coarse_provider,
            coarse_provisional)
    if (type(coarse_cell) is not int or
            not 0 <= coarse_cell < coarse_execution.cell_count):
        raise CoupledAMRTraceCPUV98Error(
            "v98 trace coarse cell is outside the execution")
    faces = tuple(fine_faces)
    fg = _grid(fine_execution)
    if (len(faces) != INTERFACE_FACE_COUNT or len(set(faces)) != 2 or
            any(type(face) is not int or not 0 <= face <
                fine_execution.face_count for face in faces)):
        raise CoupledAMRTraceCPUV98Error(
            "v98 trace requires two distinct fine interface faces")
    for face in faces:
        if (int(fg.neighbour[face]) >= 0 or
                int(fg.boundary_index[face]) >= 0):
            raise CoupledAMRTraceCPUV98Error(
                "v98 trace face is internal-coupling topology, not a "
                "physical boundary or ordinary fine-fine face")

    geometry_values = (
        joined_geometry, coarse_patch_geometry, fine_patch_geometry)
    if all(item is None for item in geometry_values):
        # Retain the historical algebra-only trace constructor for isolated
        # operator tests.  A coupled fine-substep owner must reject this
        # explicitly: the negative-neighbour distance is only the fine
        # physical-boundary half-cell distance, not the coarse-donor to
        # fine-owner centre distance.
        interface_distance = _readonly(
            np.asarray(fg.centre_distance_m)[np.asarray(faces)], (2,),
            "unqualified interface centre distance", nonnegative=True)
        joined_receipt = None
        distance_reuse_count = INTERFACE_FACE_COUNT
    elif any(item is None for item in geometry_values):
        raise CoupledAMRTraceCPUV98Error(
            "v98 joined geometry requires both authenticated patch images")
    else:
        if (type(joined_geometry) is not CoupledAMRJoinedGeometryV98 or
                type(coarse_patch_geometry) is not
                    CartesianAMRPatchGeometryV98 or
                type(fine_patch_geometry) is not
                    CartesianAMRPatchGeometryV98):
            raise CoupledAMRTraceCPUV98Error(
                "v98 joined geometry image type differs")
        try:
            validate_joined_amr_geometry_v98(
                joined_geometry, _grid(coarse_execution), fg,
                coarse_patch_geometry, fine_patch_geometry)
        except (TypeError, ValueError) as error:
            raise CoupledAMRTraceCPUV98Error(
                "v98 joined geometry failed independent reconstruction") \
                from error
        if (joined_geometry.coarse_owner_cell != coarse_cell or
                tuple(joined_geometry.fine_faces) != faces):
            raise CoupledAMRTraceCPUV98Error(
                "v98 joined geometry selects another coarse/fine interface")
        interface_distance = _readonly(
            joined_geometry.normal_centre_distance_m, (2,),
            "joined interface centre distance", nonnegative=True)
        if np.any(interface_distance <= 0.0):
            raise CoupledAMRTraceCPUV98Error(
                "v98 joined interface centre distance must be positive")
        joined_receipt = joined_geometry.receipt_sha256
        distance_reuse_count = 0

    c_v92_execution = _v92_execution(coarse_execution.active_execution)
    parent = _v92_state(coarse_parent.active_state)
    candidate = _v92_state(coarse_provisional.candidate.active_state)
    parent_bed = np.asarray(
        coarse_parent.active_state.base_state.bed_mass_kg_m2)
    candidate_bed = np.asarray(
        coarse_provisional.candidate.active_state.base_state.
        bed_mass_kg_m2)
    h0 = float(parent.q[0, coarse_cell])
    h1 = float(candidate.q[0, coarse_cell])
    q0, q1 = parent.q[:, coarse_cell], candidate.q[:, coarse_cell]
    suspended0 = parent.suspended_state.suspended_mass_kg_m2[:, coarse_cell]
    suspended1 = candidate.suspended_state.suspended_mass_kg_m2[:, coarse_cell]
    bed0, bed1 = parent_bed[:, coarse_cell], candidate_bed[:, coarse_cell]
    fluid0 = np.array((parent.h_temperature_c_m[coarse_cell],
                       parent.h_salinity_ppt_m[coarse_cell]))
    fluid1 = np.array((candidate.h_temperature_c_m[coarse_cell],
                       candidate.h_salinity_ppt_m[coarse_cell]))
    alpha = _readonly(ALPHAS, (2,), "substep alpha")
    time = _readonly(
        coarse_parent.time_s+alpha*coarse_provisional.hydraulic_evidence.
        wetdry_trial.dt_s, (2,), "substep time")
    ng = coarse_execution.fraction_count
    trace_q = np.empty((2, 3, 2))
    trace_suspended = np.empty((2, ng, 2))
    trace_bed = np.empty((2, ng, 2))
    trace_fluid = np.empty((2, 2, 2))
    trace_z = np.empty((2, 2))
    trace_water = np.empty((2, 2))
    trace_mixture = np.empty((2, 2))
    trace_wet = np.empty((2, 2), dtype=np.int32)
    grid = _grid(coarse_execution)
    dry = float(grid.dry_tolerance_m)
    for step, weight in enumerate(ALPHAS):
        q = (1.0-weight)*q0+weight*q1
        suspended = (1.0-weight)*suspended0+weight*suspended1
        bed = (1.0-weight)*bed0+weight*bed1
        fluid = (1.0-weight)*fluid0+weight*fluid1
        if (not np.isfinite(q).all() or q[0] < 0.0 or
                np.any(suspended < 0.0) or np.any(bed < 0.0) or
                fluid[1] < 0.0):
            raise CoupledAMRTraceCPUV98Error(
                "v98 interpolated conserved trace is outside its domain")
        wet = q[0] > dry
        if not wet and (np.any(q[1:] != 0.0) or
                        np.any(suspended != 0.0) or
                        np.any(fluid != 0.0)):
            raise CoupledAMRTraceCPUV98Error(
                "v98 dry internal trace retains transported inventory")
        if wet:
            temperature = (float(fluid[0]/q[0])
                           if c_v92_execution.temperature_mode ==
                           MODE_TRANSPORTED else
                           c_v92_execution.equilibrium_temperature_c)
            salinity = (float(fluid[1]/q[0])
                        if c_v92_execution.salinity_mode == MODE_TRANSPORTED
                        else c_v92_execution.equilibrium_salinity_ppt)
            concentration = suspended[:, None]/q[0]
        else:
            temperature = c_v92_execution.equilibrium_temperature_c
            salinity = c_v92_execution.equilibrium_salinity_ppt
            concentration = np.zeros((ng, 1))
        water = resolve_water_density_v88(
            np.asarray((temperature,)), np.asarray((salinity,)),
            eos=c_v92_execution.density_eos,
            constituents_active=c_v92_execution.constituents_active,
            reference_water_density_kg_m3=
                grid.profile.water_density_kg_m3)
        mixture = mixture_density_kg_m3_v85(
            concentration, water,
            grid.profile.sediment_density_kg_m3)
        elevation = (coarse_execution.initial_bed_elevation_m[coarse_cell]+
                     np.sum((bed-
                         coarse_execution.reference_bed_mass_kg_m2[
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
    placeholder = CoupledAMRInternalTraceCPUV98(
        coarse_execution.fingerprint, fine_execution.fingerprint,
        coarse_parent.state_sha256, coarse_provisional.trial_sha256,
        coarse_provisional.candidate.state_sha256, coarse_cell, faces,
        joined_receipt, interface_distance, distance_reuse_count,
        alpha, time, *arrays, 0, 0, 0, 0, 4, "")
    return CoupledAMRInternalTraceCPUV98(
        **{**placeholder.__dict__,
           "receipt_sha256": _hash(_payload(placeholder))})


def resolve_coupled_amr_internal_trace_cpu_v98(
    coarse_execution: CoupledWetDryCPUExecutionV97,
    coarse_parent: CoupledWetDryCPUStateV97,
    coarse_provider: object,
    coarse_provisional: CoupledWetDryCPUTrialV97,
    fine_execution: CoupledWetDryCPUExecutionV97, *,
    coarse_cell: int, fine_faces: Sequence[int],
    joined_geometry: CoupledAMRJoinedGeometryV98 | None = None,
    coarse_patch_geometry: CartesianAMRPatchGeometryV98 | None = None,
    fine_patch_geometry: CartesianAMRPatchGeometryV98 | None = None,
) -> CoupledAMRInternalTraceCPUV98:
    result = _resolve(
        coarse_execution, coarse_parent, coarse_provider,
        coarse_provisional, fine_execution, coarse_cell=coarse_cell,
        fine_faces=fine_faces, joined_geometry=joined_geometry,
        coarse_patch_geometry=coarse_patch_geometry,
        fine_patch_geometry=fine_patch_geometry, validate_source=True)
    validate_coupled_amr_internal_trace_cpu_v98(
        result, coarse_execution, coarse_parent, coarse_provider,
        coarse_provisional, fine_execution,
        joined_geometry=joined_geometry,
        coarse_patch_geometry=coarse_patch_geometry,
        fine_patch_geometry=fine_patch_geometry)
    return result


def validate_coupled_amr_internal_trace_structure_cpu_v98(
    value: CoupledAMRInternalTraceCPUV98,
    fine_execution: CoupledWetDryCPUExecutionV97,
) -> None:
    """Validate the immutable trace image without replaying its coarse owner.

    Native residual adapters deliberately receive only the already-authenticated
    trace receipt.  The full validator below remains the authority that replays
    the coarse parent/provisional interpolation.  This structural validator is
    the narrower hand-off gate: it proves exact fine-execution binding, internal
    (not physical) face selection, immutable array shapes and the no-boundary /
    no-clock owner counts before a nonlinear residual may consume the image.
    """
    if (type(value) is not CoupledAMRInternalTraceCPUV98 or
            type(fine_execution) is not CoupledWetDryCPUExecutionV97):
        raise CoupledAMRTraceCPUV98Error(
            "v98 internal trace structure type differs")
    _compatible_executions(fine_execution, fine_execution)
    ng = fine_execution.fraction_count
    names_shapes = (
        ("substep_alpha", (2,)),
        ("substep_time_s", (2,)),
        ("interface_centre_distance_m", (2,)),
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
            value.receipt_sha256 != _hash(_payload(value)) or
            tuple(value.substep_alpha) != ALPHAS or
            np.any(np.diff(value.substep_time_s) <= 0.0) or
            value.physical_boundary_owner_count != 0 or
            value.boundary_memory_consumer_count != 0 or
            value.accepted_clock_increment_count != 0 or
            value.density_interpolation_count != 0 or
            value.eos_derivation_count != 4 or
            value.physical_boundary_distance_reuse_count not in
                (0, INTERFACE_FACE_COUNT) or
            ((value.joined_geometry_receipt_sha256 is None) !=
             (value.physical_boundary_distance_reuse_count ==
              INTERFACE_FACE_COUNT))):
        raise CoupledAMRTraceCPUV98Error(
            "v98 internal trace structure identity differs")
    for name, shape in names_shapes:
        array = np.asarray(getattr(value, name))
        if (array.shape != shape or array.flags.writeable or
                not np.isfinite(array).all()):
            raise CoupledAMRTraceCPUV98Error(
                f"v98 internal trace {name} structure differs")
    if (np.any(value.trace_q[:, 0] < 0.0) or
            np.any(value.interface_centre_distance_m <= 0.0) or
            np.any(value.trace_suspended_kg_m2 < 0.0) or
            np.any(value.trace_bed_kg_m2 < 0.0) or
            np.any(value.trace_water_density_kg_m3 <= 0.0) or
            np.any(value.trace_mixture_density_kg_m3 <
                   value.trace_water_density_kg_m3) or
            np.any((value.trace_wet_mask != 0) &
                   (value.trace_wet_mask != 1))):
        raise CoupledAMRTraceCPUV98Error(
            "v98 internal trace physical structure differs")
    dry = value.trace_wet_mask == 0
    if (np.any(value.trace_q[:, 0][dry] != 0.0) or
            np.any(np.moveaxis(value.trace_q[:, 1:], 1, -1)[dry] != 0.0) or
            np.any(np.moveaxis(value.trace_suspended_kg_m2, 1, -1)[dry] != 0.0) or
            np.any(np.moveaxis(value.trace_fluid_integral, 1, -1)[dry] != 0.0)):
        raise CoupledAMRTraceCPUV98Error(
            "v98 dry internal trace retains transported inventory")
    fg = _grid(fine_execution)
    if (len(value.fine_faces) != INTERFACE_FACE_COUNT or
            len(set(value.fine_faces)) != INTERFACE_FACE_COUNT):
        raise CoupledAMRTraceCPUV98Error(
            "v98 internal trace face inventory differs")
    for face in value.fine_faces:
        if (type(face) is not int or not 0 <= face < fine_execution.face_count or
                int(fg.neighbour[face]) >= 0 or
                int(fg.boundary_index[face]) >= 0):
            raise CoupledAMRTraceCPUV98Error(
                "v98 trace face is not an internal negative-neighbour face")
    for digest in (
            value.coarse_execution_fingerprint,
            value.fine_execution_fingerprint,
            value.coarse_parent_state_sha256,
            value.coarse_provisional_trial_sha256,
            value.coarse_provisional_state_sha256,
            value.receipt_sha256):
        if _SHA.fullmatch(digest) is None:
            raise CoupledAMRTraceCPUV98Error(
                "v98 internal trace SHA-256 identity differs")
    if (value.joined_geometry_receipt_sha256 is not None and
            _SHA.fullmatch(value.joined_geometry_receipt_sha256) is None):
        raise CoupledAMRTraceCPUV98Error(
            "v98 joined geometry SHA-256 identity differs")


def validate_coupled_amr_internal_trace_cpu_v98(
    value: CoupledAMRInternalTraceCPUV98,
    coarse_execution: CoupledWetDryCPUExecutionV97,
    coarse_parent: CoupledWetDryCPUStateV97,
    coarse_provider: object,
    coarse_provisional: CoupledWetDryCPUTrialV97,
    fine_execution: CoupledWetDryCPUExecutionV97,
    *,
    joined_geometry: CoupledAMRJoinedGeometryV98 | None = None,
    coarse_patch_geometry: CartesianAMRPatchGeometryV98 | None = None,
    fine_patch_geometry: CartesianAMRPatchGeometryV98 | None = None,
) -> None:
    if type(value) is not CoupledAMRInternalTraceCPUV98:
        raise CoupledAMRTraceCPUV98Error(
            "v98 internal trace receipt type differs")
    validate_coupled_amr_internal_trace_structure_cpu_v98(
        value, fine_execution)
    expected = _resolve(
        coarse_execution, coarse_parent, coarse_provider,
        coarse_provisional, fine_execution,
        coarse_cell=value.coarse_cell, fine_faces=value.fine_faces,
        joined_geometry=joined_geometry,
        coarse_patch_geometry=coarse_patch_geometry,
        fine_patch_geometry=fine_patch_geometry,
        validate_source=True)
    array_names = (
        "substep_alpha", "substep_time_s", "trace_q",
        "trace_suspended_kg_m2", "trace_bed_kg_m2",
        "trace_fluid_integral", "trace_bed_elevation_m",
        "trace_water_density_kg_m3", "trace_mixture_density_kg_m3",
        "trace_wet_mask", "interface_centre_distance_m")
    if (value.receipt_sha256 != _hash(_payload(value)) or
            value.receipt_sha256 != expected.receipt_sha256 or
            any(not np.array_equal(getattr(value, name),
                                   getattr(expected, name))
                for name in array_names) or
            any(np.asarray(getattr(value, name)).flags.writeable
                for name in array_names) or
            value.physical_boundary_owner_count != 0 or
            value.boundary_memory_consumer_count != 0 or
            value.accepted_clock_increment_count != 0 or
            value.density_interpolation_count != 0 or
            value.eos_derivation_count != 4):
        raise CoupledAMRTraceCPUV98Error(
            "v98 internal trace is stale, mutated, or recomputed differently")
    for digest in (
            value.coarse_execution_fingerprint,
            value.fine_execution_fingerprint,
            value.coarse_parent_state_sha256,
            value.coarse_provisional_trial_sha256,
            value.coarse_provisional_state_sha256,
            value.receipt_sha256):
        if _SHA.fullmatch(digest) is None:
            raise CoupledAMRTraceCPUV98Error(
                "v98 internal trace SHA-256 identity differs")


__all__ = [
    "ALPHAS", "CoupledAMRInternalTraceCPUV98",
    "CoupledAMRTraceCPUV98Error", "INTERFACE_FACE_COUNT", "SCHEMA",
    "SCOPE_ID", "SUBSTEP_COUNT",
    "resolve_coupled_amr_internal_trace_cpu_v98",
    "validate_coupled_amr_internal_trace_cpu_v98",
    "validate_coupled_amr_internal_trace_structure_cpu_v98",
]
