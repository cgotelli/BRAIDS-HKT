"""Q4/v82 FLOW 2DH suspended-support closure composition.

The locked classic-FLOW source makes a distinction which is easy to miss:
``SchmidtNumber`` is stored per sediment fraction in ``sigdif`` and is used by
the vertical turbulent-diffusion branch, while the ``KMAX=1`` horizontal
branch is constituent independent and uses the shared cell ``dicuv`` field
divided by the literal 0.7.  BRAIDS is a depth-averaged solver, so v82 records
the per-fraction sand values for restart/provenance but does not incorrectly
apply them to horizontal diffusion.

V82 closes the smaller, real 2DH gap left by v76/v78: a nonuniform,
authenticated cell ``dicuv`` field is carried through FCT advection,
donor-limited diffusion, SettleFrm=2, Galappatti exchange, boundary memory,
wet/dry cleanup and the classwise mass ledger.  The accepted v78 API remains
frozen.  A zero-diffusion v78 execution is required so this additive owner is
the only diffusion owner.  Trials remain limited to four accepted
``implicit_fwave_midpoint`` updates.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from .flow2dh_boundary_diffusion import (
    FLOW2DH_HORIZONTAL_SCHMIDT,
    flow2dh_horizontal_diffusive_face_mass,
)
from .flow2dh_exchange import advance_flow2dh_equilibrium_exchange
from .flow2dh_settling import settlefrm2_noncohesive
from .multifraction_suspended_lifecycle_v78 import (
    MAX_ACCEPTED_UPDATES,
    ROLE_CPU_ORACLE,
    ROLE_CUDA,
    MultifractionSuspendedExecutionV78,
    MultifractionSuspendedStateV78,
    MultifractionSuspendedTrialV78,
    begin_multifraction_cpu_oracle_trial_v78,
    build_multifraction_trial_v78,
    commit_multifraction_trial_v78,
    execution_document_v78,
    load_multifraction_checkpoint_v78,
    multifraction_checkpoint_v78,
    rollback_multifraction_trial_v78,
    validate_multifraction_suspended_state_v78,
)
from .suspended_lifecycle_live_v76 import (
    resolve_suspended_lifecycle_state_v76,
)


SCHEMA = 1
SCOPE_ID = "braids-q4-flow2dh-suspended-support-closures-v82-v1"
CHECKPOINT_SCOPE_ID = SCOPE_ID + ":checkpoint"
SEMANTIC_ID = hashlib.sha256(SCOPE_ID.encode("ascii")).hexdigest()
INTEGRATOR = "implicit_fwave_midpoint"
HORIZONTAL_SCHMIDT_BRANCH = (
    "d3d4_flow2dh_kmax1_constituent_independent_dicuv_over_0p7"
)
TPSNUMBER_ROLE = (
    "recorded_vertical_turbulent_diffusion_control_inactive_for_kmax1"
)
DICUV_PROVIDER = (
    "authenticated_cell_field_constant_over_bounded_execution_v1"
)
MAX_UPDATES = MAX_ACCEPTED_UPDATES


class SuspendedSupportV82Error(ValueError):
    """Fail-closed v82 configuration, transaction or ledger error."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, OverflowError) as error:
        raise SuspendedSupportV82Error(
            "v82 value is not finite canonical JSON") from error


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _array_digest(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(_canonical(list(value.shape)))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _frozen_array(value: object, shape: tuple[int, ...], name: str, *,
                  nonnegative: bool = False, positive: bool = False) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != shape or raw.dtype.kind not in "fiu":
        raise SuspendedSupportV82Error(
            f"v82 {name} must be a finite numeric array with shape {shape}")
    result = np.ascontiguousarray(raw, dtype=np.float64)
    if not np.isfinite(result).all():
        raise SuspendedSupportV82Error(f"v82 {name} must be finite")
    if nonnegative and np.any(result < 0.0):
        raise SuspendedSupportV82Error(f"v82 {name} must be nonnegative")
    if positive and np.any(result <= 0.0):
        raise SuspendedSupportV82Error(f"v82 {name} must be positive")
    result.setflags(write=False)
    return result


def resolve_sand_schmidt_numbers_v82(
    fraction_count: int, overrides: object | None = None,
) -> np.ndarray:
    """Resolve locked ``rdsed`` sand defaults or explicit per-fraction values.

    These values are intentionally provenance-only for this KMAX=1 slice.
    They must not be substituted for the horizontal 0.7 branch.
    """
    if (isinstance(fraction_count, bool) or type(fraction_count) is not int or
            fraction_count < 1):
        raise SuspendedSupportV82Error(
            "v82 Schmidt fraction count must be a positive integer")
    values = (np.ones(fraction_count, dtype=np.float64) if overrides is None
              else np.asarray(overrides))
    return _frozen_array(
        values, (fraction_count,), "per-fraction Schmidt number", positive=True)


@dataclass(frozen=True)
class SuspendedSupportExecutionV82:
    base_execution: MultifractionSuspendedExecutionV78
    dicuv_cell_m2_s: np.ndarray
    tpsnumber_by_fraction: np.ndarray
    semantic_id: str
    fingerprint: str

    @property
    def role(self) -> str:
        return self.base_execution.role

    @property
    def fraction_count(self) -> int:
        return self.base_execution.fraction_count

    @property
    def cell_count(self) -> int:
        return self.base_execution.cell_count

    @property
    def face_count(self) -> int:
        return self.base_execution.face_count

    @property
    def boundary_count(self) -> int:
        return self.base_execution.boundary_count


def _execution_payload(value: SuspendedSupportExecutionV82) -> dict:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "semantic_id": value.semantic_id,
        "base_v78_execution": execution_document_v78(value.base_execution),
        "integrator": INTEGRATOR,
        "kmax": 1,
        "dicuv_provider": DICUV_PROVIDER,
        "dicuv_cell_m2_s": value.dicuv_cell_m2_s.tolist(),
        "horizontal_schmidt_branch": HORIZONTAL_SCHMIDT_BRANCH,
        "horizontal_schmidt_value": FLOW2DH_HORIZONTAL_SCHMIDT,
        "tpsnumber_by_fraction": value.tpsnumber_by_fraction.tolist(),
        "tpsnumber_role": TPSNUMBER_ROLE,
        "accepted_update_limit": MAX_UPDATES,
        "automatic_cpu_fallback": False,
        "locked_delft3d_source_read_only": True,
    }


def resolve_suspended_support_execution_v82(
    base_execution: MultifractionSuspendedExecutionV78, *,
    dicuv_cell_m2_s: object, tpsnumber_by_fraction: object | None = None,
) -> SuspendedSupportExecutionV82:
    """Authenticate a bounded spatial-diffusivity lifecycle execution."""
    execution_document_v78(base_execution)
    if base_execution.role not in (ROLE_CPU_ORACLE, ROLE_CUDA):
        raise SuspendedSupportV82Error("v82 base role differs")
    # The frozen predecessor must contribute no diffusion; v82 owns the one
    # and only spatial field pass.
    if base_execution.base_execution.profile.horizontal_diffusivity_m2_s != 0.0:
        raise SuspendedSupportV82Error(
            "v82 requires a zero-dicuv v78 base to prevent duplicate diffusion")
    dicuv = _frozen_array(
        dicuv_cell_m2_s, (base_execution.cell_count,), "dicuv cell field",
        nonnegative=True)
    tps = resolve_sand_schmidt_numbers_v82(
        base_execution.fraction_count, tpsnumber_by_fraction)
    placeholder = SuspendedSupportExecutionV82(
        base_execution, dicuv, tps, SEMANTIC_ID, "")
    return SuspendedSupportExecutionV82(
        **{**placeholder.__dict__,
           "fingerprint": _fingerprint(_execution_payload(placeholder))})


def execution_document_v82(value: SuspendedSupportExecutionV82) -> dict:
    if not isinstance(value, SuspendedSupportExecutionV82):
        raise SuspendedSupportV82Error("v82 execution object differs")
    payload = _execution_payload(value)
    if value.fingerprint != _fingerprint(payload):
        raise SuspendedSupportV82Error("v82 execution fingerprint differs")
    return {**payload, "fingerprint": value.fingerprint}


@dataclass(frozen=True)
class SuspendedSupportStateV82:
    execution_fingerprint: str
    base_state: MultifractionSuspendedStateV78
    state_sha256: str

    @property
    def accepted_updates(self) -> int:
        return self.base_state.accepted_updates

    @property
    def time_s(self) -> float:
        return self.base_state.time_s

    @property
    def suspended_mass_kg_m2(self) -> np.ndarray:
        return self.base_state.suspended_mass_kg_m2

    @property
    def bed_mass_kg_m2(self) -> np.ndarray:
        return self.base_state.bed_mass_kg_m2


def _state_token(value: SuspendedSupportStateV82) -> str:
    return _fingerprint({
        "execution_fingerprint": value.execution_fingerprint,
        "base_state_sha256": value.base_state.state_sha256,
    })


def _wrap_state(
    execution: SuspendedSupportExecutionV82,
    base_state: MultifractionSuspendedStateV78,
) -> SuspendedSupportStateV82:
    execution_document_v82(execution)
    validate_multifraction_suspended_state_v78(
        base_state, execution.base_execution)
    placeholder = SuspendedSupportStateV82(
        execution.fingerprint, base_state, "")
    return SuspendedSupportStateV82(
        **{**placeholder.__dict__, "state_sha256": _state_token(placeholder)})


def resolve_suspended_support_state_v82(
    execution: SuspendedSupportExecutionV82, **state_fields,
) -> SuspendedSupportStateV82:
    from .multifraction_suspended_lifecycle_v78 import (
        resolve_multifraction_suspended_state_v78,
    )
    try:
        base = resolve_multifraction_suspended_state_v78(
            execution.base_execution, **state_fields)
    except (TypeError, ValueError) as error:
        raise SuspendedSupportV82Error(
            "v82 initial lifecycle state is invalid") from error
    return _wrap_state(execution, base)


def validate_suspended_support_state_v82(
    state: SuspendedSupportStateV82,
    execution: SuspendedSupportExecutionV82,
) -> None:
    if (not isinstance(state, SuspendedSupportStateV82) or
            state.execution_fingerprint != execution.fingerprint or
            state.state_sha256 != _state_token(state)):
        raise SuspendedSupportV82Error(
            "v82 state is stale, mutated, or execution-mismatched")
    validate_multifraction_suspended_state_v78(
        state.base_state, execution.base_execution)


def _advected_mass_from_flux(
    execution: SuspendedSupportExecutionV82,
    parent_mass: np.ndarray, face_flux: np.ndarray, dt_s: float,
) -> np.ndarray:
    base = execution.base_execution.base_execution
    result = np.array(parent_mass, dtype=np.float64, copy=True)
    for cell in range(base.cell_count):
        for slot in range(base.cell_face_offset[cell],
                          base.cell_face_offset[cell + 1]):
            face = base.cell_faces[slot]
            result[:, cell] -= (
                dt_s*base.cell_face_sign[slot]*base.face_length_m[face]*
                face_flux[:, face]/base.cell_area_m2[cell])
    scale = np.maximum(1.0, np.abs(parent_mass))
    tolerance = 8192.0*np.finfo(np.float64).eps*scale
    if not np.isfinite(result).all() or np.any(result < -tolerance):
        raise SuspendedSupportV82Error(
            "v82 reconstructed post-FCT mass is invalid")
    return np.maximum(result, 0.0)


def _cpu_spatial_diffusion(
    execution: SuspendedSupportExecutionV82, mass: np.ndarray,
    depth: np.ndarray, boundary: np.ndarray, exchange_area: np.ndarray,
    face_active: np.ndarray, dt_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base = execution.base_execution.base_execution
    ng, nc, nf, nb = (execution.fraction_count, execution.cell_count,
                      execution.face_count, execution.boundary_count)
    face_mass = np.zeros((ng, nf), dtype=np.float64)
    effective = np.zeros(nf, dtype=np.float64)
    for face in range(nf):
        owner = int(base.owner[face])
        neighbour = int(base.neighbour[face])
        boundary_id = int(base.boundary_index[face])
        if neighbour < 0 and boundary_id < 0:
            if face_active[face]:
                raise SuspendedSupportV82Error(
                    "v82 closed face cannot be diffusion-active")
            continue
        left_depth = depth[owner]
        right_depth = depth[neighbour] if neighbour >= 0 else left_depth
        left = (mass[:, owner]/left_depth if left_depth > 0.0
                else np.zeros(ng))
        if neighbour >= 0:
            right = (mass[:, neighbour]/right_depth if right_depth > 0.0
                     else np.zeros(ng))
            right_dicuv = execution.dicuv_cell_m2_s[neighbour]
        else:
            right = boundary[:, face]
            # The exterior boundary trace has no independently owned cell;
            # the adjacent FLOW cell coefficient is extended to the face.
            right_dicuv = execution.dicuv_cell_m2_s[owner]
        result = flow2dh_horizontal_diffusive_face_mass(
            left, right, execution.dicuv_cell_m2_s[owner], right_dicuv,
            exchange_area[face], base.centre_distance_m[face],
            left_depth, right_depth, dt_s=dt_s,
            dry_threshold_m=base.profile.diffusion_dry_threshold_m,
            transport_face_active=bool(face_active[face]))
        face_mass[:, face] = result.mass_left_to_right_kg
        if face_active[face]:
            effective[face] = float(np.asarray(
                result.effective_diffusivity_m2_s).reshape(-1)[0])

    theta = np.ones((ng, nc), dtype=np.float64)
    margin = 1.0-32.0*np.finfo(np.float64).eps
    for grain in range(ng):
        for cell in range(nc):
            outgoing = 0.0
            for slot in range(base.cell_face_offset[cell],
                              base.cell_face_offset[cell + 1]):
                face = base.cell_faces[slot]
                outgoing += max(
                    base.cell_face_sign[slot]*face_mass[grain, face], 0.0)
            inventory = mass[grain, cell]*base.cell_area_m2[cell]
            if outgoing > 0.0 and outgoing >= inventory:
                theta[grain, cell] = min(1.0, inventory/outgoing)*margin
    for grain in range(ng):
        for face in range(nf):
            value = face_mass[grain, face]
            donor = (int(base.owner[face]) if value > 0.0 else
                     int(base.neighbour[face]) if value < 0.0 else -1)
            if donor >= 0:
                face_mass[grain, face] *= theta[grain, donor]

    result_mass = np.array(mass, copy=True)
    for cell in range(nc):
        for slot in range(base.cell_face_offset[cell],
                          base.cell_face_offset[cell + 1]):
            face = base.cell_faces[slot]
            result_mass[:, cell] -= (
                base.cell_face_sign[slot]*face_mass[:, face]/
                base.cell_area_m2[cell])
    tolerance = 8192.0*np.finfo(np.float64).eps*np.maximum(1.0, np.abs(mass))
    if not np.isfinite(result_mass).all() or np.any(result_mass < -tolerance):
        raise SuspendedSupportV82Error("v82 spatial diffusion result is invalid")
    result_mass = np.maximum(result_mass, 0.0)
    ledger = np.zeros((ng, nb), dtype=np.float64)
    for face in range(nf):
        boundary_id = int(base.boundary_index[face])
        if boundary_id >= 0:
            ledger[:, boundary_id] -= face_mass[:, face]
    return result_mass, face_mass, ledger, effective


@dataclass(frozen=True)
class SuspendedSupportTrialV82:
    execution_fingerprint: str
    parent_state_sha256: str
    base_trial: MultifractionSuspendedTrialV78
    candidate: SuspendedSupportStateV82
    effective_horizontal_diffusivity_m2_s: np.ndarray
    trial_sha256: str


def _trial_token(value: SuspendedSupportTrialV82) -> str:
    return _fingerprint({
        "execution_fingerprint": value.execution_fingerprint,
        "parent_state_sha256": value.parent_state_sha256,
        "base_trial_sha256": value.base_trial.trial_sha256,
        "candidate_state_sha256": value.candidate.state_sha256,
        "effective_horizontal_diffusivity_sha256":
            _array_digest(value.effective_horizontal_diffusivity_m2_s),
    })


def _wrap_trial(
    execution: SuspendedSupportExecutionV82,
    parent: SuspendedSupportStateV82,
    base_trial: MultifractionSuspendedTrialV78,
    effective: object,
) -> SuspendedSupportTrialV82:
    candidate = _wrap_state(execution, base_trial.candidate)
    effective_a = _frozen_array(
        effective, (execution.face_count,), "effective face diffusivity",
        nonnegative=True)
    placeholder = SuspendedSupportTrialV82(
        execution.fingerprint, parent.state_sha256, base_trial, candidate,
        effective_a, "")
    return SuspendedSupportTrialV82(
        **{**placeholder.__dict__, "trial_sha256": _trial_token(placeholder)})


def begin_suspended_support_cpu_trial_v82(
    execution: SuspendedSupportExecutionV82,
    parent: SuspendedSupportStateV82, *,
    accepted_step_receipt: Mapping[str, object], cancelled: bool = False,
    **inputs,
) -> SuspendedSupportTrialV82:
    """Run one complete CPU-oracle lifecycle with spatial ``dicuv``."""
    if execution.role != ROLE_CPU_ORACLE:
        raise SuspendedSupportV82Error(
            "v82 CPU trial requires isolated_cpu_oracle role")
    validate_suspended_support_state_v82(parent, execution)
    if cancelled:
        raise SuspendedSupportV82Error(
            "v82 update cancelled before candidate publication")
    if parent.accepted_updates >= MAX_UPDATES:
        raise SuspendedSupportV82Error("v82 accepted update limit was reached")
    try:
        base_zero = begin_multifraction_cpu_oracle_trial_v78(
            execution.base_execution, parent.base_state,
            accepted_step_receipt=accepted_step_receipt, **inputs)
    except (TypeError, ValueError) as error:
        raise SuspendedSupportV82Error(
            "v82 base lifecycle rejected the CPU trial") from error

    base = execution.base_execution.base_execution
    dt = float(inputs["dt_s"])
    h0 = np.asarray(inputs["depth_before_m"], dtype=np.float64)
    h1 = np.asarray(inputs["depth_after_m"], dtype=np.float64)
    face_area = np.asarray(
        inputs["accepted_face_exchange_area_m2"], dtype=np.float64)
    face_active = np.asarray(inputs["transport_face_active"], dtype=bool)
    momentum = np.asarray(inputs["endpoint_momentum_m2_s"], dtype=np.float64)
    shear = np.asarray(inputs["shear_velocity_m_s"], dtype=np.float64)
    capacity = np.asarray(
        inputs["suspended_transport_capacity_kg_m_s"], dtype=np.float64)
    surface_fraction = np.asarray(inputs["surface_fraction"], dtype=np.float64)
    advected = _advected_mass_from_flux(
        execution, parent.suspended_mass_kg_m2,
        base_zero.advective_face_flux_kg_m_s, dt)
    diffused, face_mass, diffusive_ledger, effective = \
        _cpu_spatial_diffusion(
            execution, advected, h1,
            base_zero.boundary_concentration_kg_m3, face_area, face_active, dt)

    wet = h1 > base.dry_tolerance_m
    speed = np.zeros(base.cell_count, dtype=np.float64)
    speed[wet] = np.hypot(momentum[0, wet], momentum[1, wet])/h1[wet]
    concentration = np.zeros(
        (base.cell_count, execution.fraction_count), dtype=np.float64)
    concentration[wet] = (diffused[:, wet]/h1[wet]).T
    settling = settlefrm2_noncohesive(
        concentration, base.profile.sediment_d50_m,
        base.profile.settling_fac_dss,
        base.profile.sediment_density_kg_m3,
        base.profile.water_density_kg_m3,
        base.profile.temperature_value_c,
        settling_reference_concentration_kg_m3=
            base.profile.settling_reference_concentration_kg_m3,
        gravity_m_s2=base.profile.gravity_m_s2, wet_mask=wet)
    exchange = advance_flow2dh_equilibrium_exchange(
        diffused, parent.bed_mass_kg_m2, capacity, surface_fraction,
        speed, shear, h0, h1, settling.settling_velocity_m_s.T,
        base.cell_area_m2, dt_s=dt, fac_tsd=base.profile.fac_tsd,
        sediment_threshold_m=base.profile.sediment_threshold_m,
        minimum_reference_depth_m=base.profile.reference_depth_floor_m,
        flow_wet_mask=wet)
    cleanup = np.where(wet[None, :], 0.0,
                       exchange.suspended_mass_kg_m2)
    final_suspended = exchange.suspended_mass_kg_m2-cleanup
    final_bed = exchange.bed_mass_kg_m2+cleanup
    cleanup_water = -np.sum(
        cleanup*base.cell_area_m2[None, :], axis=1)
    cleanup_bed = -cleanup_water
    parent_lifecycle = parent.base_state.lifecycle
    zero_candidate = base_zero.candidate.lifecycle
    candidate_lifecycle = resolve_suspended_lifecycle_state_v76(
        base,
        suspended_mass_kg_m2=final_suspended,
        bed_mass_kg_m2=final_bed,
        boundary_last_outflow_concentration_kg_m3=
            zero_candidate.boundary_last_outflow_concentration_kg_m3,
        boundary_return_clock_s=zero_candidate.boundary_return_clock_s,
        advective_boundary_ledger_kg=
            zero_candidate.advective_boundary_ledger_kg,
        diffusive_boundary_ledger_kg=(
            parent_lifecycle.diffusive_boundary_ledger_kg+diffusive_ledger),
        water_exchange_ledger_kg=(
            parent_lifecycle.water_exchange_ledger_kg+
            exchange.suspended_exchange_ledger_kg),
        bed_exchange_ledger_kg=(
            parent_lifecycle.bed_exchange_ledger_kg+
            exchange.bed_exchange_ledger_kg),
        wetdry_water_ledger_kg=(
            parent_lifecycle.wetdry_water_ledger_kg+cleanup_water),
        wetdry_bed_ledger_kg=(
            parent_lifecycle.wetdry_bed_ledger_kg+cleanup_bed),
        reference_water_mass_kg=parent_lifecycle.reference_water_mass_kg,
        reference_bed_mass_kg=parent_lifecycle.reference_bed_mass_kg,
        accepted_updates=parent.accepted_updates+1,
        time_s=parent.time_s+dt)
    base_trial = build_multifraction_trial_v78(
        execution.base_execution, parent.base_state, accepted_step_receipt,
        candidate_lifecycle,
        boundary_concentration_kg_m3=
            base_zero.boundary_concentration_kg_m3,
        advective_face_flux_kg_m_s=base_zero.advective_face_flux_kg_m_s,
        diffusive_face_mass_owner_out_kg=face_mass,
        settling_velocity_m_s=settling.settling_velocity_m_s.T,
        equilibrium_concentration_kg_m3=
            exchange.equilibrium_concentration_kg_m3,
        transferred_to_suspension_kg_m2=
            exchange.transferred_to_suspension_kg_m2,
        wetdry_transfer_to_bed_kg_m2=cleanup,
        observed_parallel_workers=1, backend=ROLE_CPU_ORACLE)
    return _wrap_trial(execution, parent, base_trial, effective)


def _validate_trial(
    execution: SuspendedSupportExecutionV82,
    parent: SuspendedSupportStateV82,
    trial: SuspendedSupportTrialV82,
) -> None:
    validate_suspended_support_state_v82(parent, execution)
    if (not isinstance(trial, SuspendedSupportTrialV82) or
            trial.execution_fingerprint != execution.fingerprint or
            trial.parent_state_sha256 != parent.state_sha256 or
            trial.trial_sha256 != _trial_token(trial)):
        raise SuspendedSupportV82Error(
            "v82 trial is stale, mutated, or belongs to another parent")
    validate_suspended_support_state_v82(trial.candidate, execution)


def commit_suspended_support_trial_v82(
    execution: SuspendedSupportExecutionV82,
    parent: SuspendedSupportStateV82,
    trial: SuspendedSupportTrialV82,
) -> SuspendedSupportStateV82:
    _validate_trial(execution, parent, trial)
    # Exercise the frozen predecessor's transaction validation as well.
    committed = commit_multifraction_trial_v78(
        execution.base_execution, parent.base_state, trial.base_trial)
    if committed.state_sha256 != trial.candidate.base_state.state_sha256:
        raise SuspendedSupportV82Error("v82/base committed state differs")
    return trial.candidate


def rollback_suspended_support_trial_v82(
    execution: SuspendedSupportExecutionV82,
    parent: SuspendedSupportStateV82,
    trial: SuspendedSupportTrialV82,
) -> SuspendedSupportStateV82:
    _validate_trial(execution, parent, trial)
    rolled = rollback_multifraction_trial_v78(
        execution.base_execution, parent.base_state, trial.base_trial)
    if rolled is not parent.base_state:
        raise SuspendedSupportV82Error("v82/base rollback differs")
    return parent


def cancel_suspended_support_trial_v82(
    execution: SuspendedSupportExecutionV82,
    parent: SuspendedSupportStateV82,
    trial: SuspendedSupportTrialV82,
) -> SuspendedSupportStateV82:
    return rollback_suspended_support_trial_v82(execution, parent, trial)


def suspended_support_checkpoint_v82(
    execution: SuspendedSupportExecutionV82,
    state: SuspendedSupportStateV82,
) -> dict:
    validate_suspended_support_state_v82(state, execution)
    body = {
        "schema": SCHEMA,
        "scope_id": CHECKPOINT_SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "execution_fingerprint": execution.fingerprint,
        "integrator": INTEGRATOR,
        "base_v78_checkpoint": multifraction_checkpoint_v78(
            execution.base_execution, state.base_state),
        "state_sha256": state.state_sha256,
    }
    return {**body, "fingerprint": _fingerprint(body)}


def load_suspended_support_checkpoint_v82(
    execution: SuspendedSupportExecutionV82,
    document: Mapping[str, object],
) -> SuspendedSupportStateV82:
    expected = {
        "schema", "scope_id", "semantic_id", "execution_fingerprint",
        "integrator", "base_v78_checkpoint", "state_sha256", "fingerprint",
    }
    if not isinstance(document, Mapping) or set(document) != expected:
        raise SuspendedSupportV82Error("v82 checkpoint inventory differs")
    body = {key: document[key] for key in document if key != "fingerprint"}
    if (document["fingerprint"] != _fingerprint(body) or
            document["schema"] != SCHEMA or
            document["scope_id"] != CHECKPOINT_SCOPE_ID or
            document["semantic_id"] != SEMANTIC_ID or
            document["execution_fingerprint"] != execution.fingerprint or
            document["integrator"] != INTEGRATOR):
        raise SuspendedSupportV82Error("v82 checkpoint identity differs")
    try:
        base = load_multifraction_checkpoint_v78(
            execution.base_execution, document["base_v78_checkpoint"])
    except (TypeError, ValueError) as error:
        raise SuspendedSupportV82Error(
            "v82 base checkpoint is invalid") from error
    state = _wrap_state(execution, base)
    if document["state_sha256"] != state.state_sha256:
        raise SuspendedSupportV82Error("v82 checkpoint state digest differs")
    return state


def method_identity_v82() -> dict:
    return {
        "scope_id": SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "builds_additively_on":
            "braids-q4-multifraction-suspended-lifecycle-v78-v1",
        "integrator": INTEGRATOR,
        "backend_roles": [ROLE_CPU_ORACLE, ROLE_CUDA],
        "fixed_grid_only": True,
        "fraction_scope": "2..3 noncohesive sand/gravel classes",
        "dicuv_provider": DICUV_PROVIDER,
        "horizontal_schmidt_branch": HORIZONTAL_SCHMIDT_BRANCH,
        "horizontal_schmidt_value": FLOW2DH_HORIZONTAL_SCHMIDT,
        "tpsnumber_role": TPSNUMBER_ROLE,
        "tpsnumber_horizontal_effect": False,
        "accepted_update_limit": MAX_UPDATES,
        "automatic_cpu_fallback": False,
        "transaction": "immutable_commit_rollback_cancel_exact_restart_v1",
        "not_claimed": [
            "DensIn", "FactCr", "dynamic temperature or salinity",
            "dynamic vicuv-to-dicuv turbulence closure",
            "vertical sediment diffusion", "general AMR",
        ],
        "delft3d_source_modified": False,
        "long_run_claimed": False,
    }


FROZEN_PREDECESSOR_SHA256 = {
    "braids_core/sediment/suspended_lifecycle_live_v76.py":
        "20ba39ba7228ea6c50e2ea310777d59516c9c157c05d71c6bf3298fed07d4ee6",
    "braids_gpu/suspended_lifecycle_live_v76.py":
        "182355c4ef1184fb64b877ca71ef5ada344bd40eb27bc6c5c0968a0541661ff8",
    "braids_core/sediment/multifraction_suspended_lifecycle_v78.py":
        "eaba4861d38369eaf740803c43a722e10a0ee70bb9a2e1dcf56bddba2a7e10dd",
    "braids_gpu/multifraction_suspended_lifecycle_v78.py":
        "e4fdd4aa5244d0ceff7f7b5479cdfbe4bc2cd9c5344f731cecf5351180facc6b",
    "braids_gpu/flow2dh_boundary_diffusion.cu":
        "b61c2ffe2c4a4bc411ffc4c0ce0fc07ca9edd1d86a23d622d903cb5aa15e71bf",
    "braids_gpu/law11_soursin_exchange.cu":
        "26fd25cee3bbdfdced65690e9367e34daf8fa50ed260dd425b7a3b2248c2ff6d",
}


def verify_frozen_predecessors_v82(root: str | Path) -> dict[str, str]:
    root = Path(root)
    result = {}
    for relative, expected in FROZEN_PREDECESSOR_SHA256.items():
        actual = hashlib.sha256((root/relative).read_bytes()).hexdigest()
        if actual != expected:
            raise SuspendedSupportV82Error(
                "v82 frozen predecessor hash differs: " + relative)
        result[relative] = actual
    return result


def verify_sources_v82(root: str | Path) -> dict[str, str]:
    root = Path(root)
    paths = (
        "braids_core/sediment/suspended_support_closures_v82.py",
        "braids_gpu/suspended_support_closures_v82.py",
        "tests/test_q4_suspended_support_closures_v82.py",
        "tests/test_q4_suspended_support_closures_cuda_v82.py",
    )
    result = {}
    for relative in paths:
        path = root/relative
        if not path.is_file():
            raise SuspendedSupportV82Error(
                "v82 additive source is absent: " + relative)
        result[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


__all__ = [
    "CHECKPOINT_SCOPE_ID", "DICUV_PROVIDER", "HORIZONTAL_SCHMIDT_BRANCH",
    "INTEGRATOR", "MAX_UPDATES", "SCHEMA", "SCOPE_ID", "SEMANTIC_ID",
    "SuspendedSupportExecutionV82", "SuspendedSupportStateV82",
    "SuspendedSupportTrialV82", "SuspendedSupportV82Error",
    "TPSNUMBER_ROLE", "begin_suspended_support_cpu_trial_v82",
    "cancel_suspended_support_trial_v82",
    "commit_suspended_support_trial_v82", "execution_document_v82",
    "load_suspended_support_checkpoint_v82", "method_identity_v82",
    "resolve_sand_schmidt_numbers_v82",
    "resolve_suspended_support_execution_v82",
    "resolve_suspended_support_state_v82",
    "rollback_suspended_support_trial_v82",
    "suspended_support_checkpoint_v82",
    "validate_suspended_support_state_v82",
    "verify_frozen_predecessors_v82", "verify_sources_v82",
]
