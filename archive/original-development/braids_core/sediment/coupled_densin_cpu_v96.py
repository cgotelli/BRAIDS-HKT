"""Atomic CPU composition of native v95 hydraulics with accepted v94b physics.

V96 is deliberately narrow.  It authenticates concentration directly from an
accepted v94b state, evaluates the native OpenMP augmented-f-wave/DensIn
midpoint, and gives that one endpoint and one conservative water-face field to
the existing v94b scalar/Q3/Exner transaction.  The accepted hydraulic state is
the q already owned by v92 inside v94b; no shadow hydraulic state exists.

The admitted increment is a closed, flat, fully-wet, fixed Cartesian case with
uniform water density and an authenticated exact-zero turbulence/diffusion
operator.  It is an atomic validation owner, not product/plugin admission.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Callable, Mapping

import numpy as np

from . import active_suspended_control_v94 as _v94
from .active_suspended_control_v94 import (
    ActiveSuspendedControlExecutionV94,
    ActiveSuspendedControlStateV94,
    ActiveSuspendedControlTrialV94,
)
from .active_suspended_control_v94b import (
    SEMANTIC_ID_V94B,
    begin_active_suspended_control_cpu_trial_v94b,
    checkpoint_active_suspended_control_state_v94b,
    commit_active_suspended_control_trial_v94b,
    execution_document_v94b,
    load_active_suspended_control_checkpoint_v94b,
    rollback_active_suspended_control_trial_v94b,
    validate_active_suspended_control_state_v94b,
)
from .densin_fwave_v85 import (
    LOCKED_UPSTREAM_COMMIT,
    delft_kmax1_face_acceleration_v85,
    mixture_density_kg_m3_v85,
    project_face_acceleration_v85,
)
from .densin_residual_cpu_v95 import (
    DENSIN_SEDIMENT_INCREMENT,
    DensInResidualCPUExecutionV95,
    NativeDensInRuntimeV95,
    NativeRateResultV95,
    execution_document_v95,
)
from .hles_producer_v90 import (
    CurrentClosureHandoffV90,
    HlesProducerStateV90,
    HlesProducerTrialV90,
    begin_hles_cpu_trial_v90,
    commit_hles_trial_v90,
    hles_checkpoint_v90,
    load_hles_checkpoint_v90,
    make_current_closure_handoff_v90,
    make_hles_input_receipt_v90,
    rollback_hles_trial_v90,
    validate_hles_producer_state_v90,
)


SCHEMA = 1
SCOPE_ID = "braids-q4-coupled-densin-cpu-v96-v1"
SEMANTIC_ID = SCOPE_ID + ":atomic-native-v95-v94b-owner"
CHECKPOINT_SCOPE_ID = SCOPE_ID + ":checkpoint"
BACKEND = "native_cpu_openmp"
MAX_ACCEPTED_UPDATES = 4
_SHA256 = __import__("re").compile(r"[0-9a-f]{64}\Z")


class CoupledDensInCPUV96Error(ValueError):
    """The bounded v96 execution, receipt, state, or transaction is invalid."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False).encode("ascii")
    except (TypeError, ValueError, OverflowError) as error:
        raise CoupledDensInCPUV96Error(
            "v96 value is not finite canonical JSON") from error


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _digest(*values: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(value)
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(_canonical(list(array.shape)))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise CoupledDensInCPUV96Error(
            f"v96 {name} must be a lowercase SHA-256")
    return value


def _finite(value: object, name: str, *, positive: bool = False) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(float(value))):
        raise CoupledDensInCPUV96Error(f"v96 {name} must be finite numeric data")
    result = float(value)
    if positive and result <= 0.0:
        raise CoupledDensInCPUV96Error(f"v96 {name} must be positive")
    return result


def _array(value: object, shape: tuple[int, ...], name: str, *,
           positive: bool = False, nonnegative: bool = False,
           integer: bool = False) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != shape or raw.dtype.kind not in "fiu b":
        raise CoupledDensInCPUV96Error(
            f"v96 {name} must be numeric with shape {shape}")
    result = np.ascontiguousarray(
        raw, dtype=np.int32 if integer else np.float64)
    if not np.isfinite(result).all():
        raise CoupledDensInCPUV96Error(f"v96 {name} must be finite")
    if positive and np.any(result <= 0.0):
        raise CoupledDensInCPUV96Error(f"v96 {name} must be positive")
    if nonnegative and np.any(result < 0.0):
        raise CoupledDensInCPUV96Error(f"v96 {name} must be nonnegative")
    result.setflags(write=False)
    return result


def _cancelled(value: bool | Callable[[], bool]) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if not callable(value):
        raise CoupledDensInCPUV96Error(
            "v96 cancellation must be Boolean or a zero-argument callable")
    result = value()
    if not isinstance(result, (bool, np.bool_)):
        raise CoupledDensInCPUV96Error(
            "v96 cancellation callback must return Boolean data")
    return bool(result)


def _v92_execution(execution: ActiveSuspendedControlExecutionV94):
    return execution.base_execution.base_execution


def _v92_state(state: ActiveSuspendedControlStateV94):
    return state.base_state.base_state


def _grid_execution(execution: ActiveSuspendedControlExecutionV94):
    return _v92_execution(execution).base_execution.base_execution.base_execution


@dataclass(frozen=True)
class V94bAcceptedDensityReceiptV96:
    execution_fingerprint: str
    active_execution_fingerprint: str
    parent_state_sha256: str
    base_v92_state_sha256: str
    suspended_state_sha256: str
    accepted_update: int
    time_s: float
    depth_m: np.ndarray
    suspended_mass_kg_m2: np.ndarray
    concentration_kg_m3: np.ndarray
    water_density_kg_m3: np.ndarray
    mixture_density_kg_m3: np.ndarray
    receipt_sha256: str


def _density_receipt_token(value: V94bAcceptedDensityReceiptV96) -> str:
    return _fingerprint({
        "scope_id": SCOPE_ID + ":accepted-v94b-density",
        "source_kind": "typed_v94b_accepted_state",
        "execution_fingerprint": value.execution_fingerprint,
        "active_execution_fingerprint": value.active_execution_fingerprint,
        "parent_state_sha256": value.parent_state_sha256,
        "base_v92_state_sha256": value.base_v92_state_sha256,
        "suspended_state_sha256": value.suspended_state_sha256,
        "accepted_update": value.accepted_update,
        "time_s": value.time_s,
        "arrays_sha256": _digest(
            value.depth_m, value.suspended_mass_kg_m2,
            value.concentration_kg_m3, value.water_density_kg_m3,
            value.mixture_density_kg_m3),
        "call_order": "accepted_v94b_state_to_next_native_hydraulics",
    })


@dataclass(frozen=True)
class CoupledDensInCPUExecutionV96:
    active_execution: ActiveSuspendedControlExecutionV94
    hydraulic_execution: DensInResidualCPUExecutionV95
    locked_delft_commit: str
    locked_delft_source_sha256: tuple[tuple[str, str], ...]
    semantic_id: str
    fingerprint: str

    @property
    def fraction_count(self) -> int:
        return self.active_execution.fraction_count

    @property
    def cell_count(self) -> int:
        return self.active_execution.cell_count

    @property
    def face_count(self) -> int:
        return _v92_execution(self.active_execution).face_count


def _execution_payload(value: CoupledDensInCPUExecutionV96) -> dict:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "semantic_id": value.semantic_id,
        "backend": BACKEND,
        "active_v94b_execution": execution_document_v94b(
            value.active_execution),
        "native_v95_execution": execution_document_v95(
            value.hydraulic_execution),
        "accepted_density_source": "typed_v94b_accepted_state",
        "joint_state": "v94b_q_plus_hles_no_shadow_hydraulic_state",
        "hydraulic_endpoint_owner_count": 1,
        "water_face_flux_owner_count": 1,
        "scalar_lifecycle_owner_count": 1,
        "authoritative_exner_owner_count": 1,
        "hles_clock_owner_count": 1,
        "boundary_mode": "closed_reflective",
        "grid_scope": "flat_fully_wet_cartesian_fixed_grid",
        "water_density_scope": "uniform_accepted_v92_field",
        "turbulence_scope": "authenticated_exact_zero",
        "nested_midpoint_disposition": (
            "v92_exact_constant-rate_wrapper_midpoint_not_a_second_"
            "hydraulic_solve;native_v95_midpoint_is_authoritative"),
        "locked_delft_commit": value.locked_delft_commit,
        "locked_delft_source_sha256": dict(
            value.locked_delft_source_sha256),
        "accepted_update_limit": MAX_ACCEPTED_UPDATES,
        "automatic_cpu_fallback": False,
        "product_admission": False,
    }


def _topology_matches_v95(value: CoupledDensInCPUExecutionV96) -> bool:
    grid = _grid_execution(value.active_execution)
    base = value.hydraulic_execution.base_execution
    pairs = (
        (grid.owner, base.owner),
        (grid.neighbour, base.neighbour),
        (grid.normal, base.normal),
        (grid.face_length_m, base.face_length_m),
        (grid.centre_distance_m, base.face_distance_m),
        (grid.cell_area_m2, base.cell_area_m2),
        (grid.cell_face_offset, base.cell_face_offset),
        (grid.cell_faces, base.cell_faces),
        (grid.cell_face_sign, base.cell_face_sign),
    )
    return all(np.array_equal(left, right) for left, right in pairs)


def resolve_coupled_densin_cpu_execution_v96(
    active_execution: ActiveSuspendedControlExecutionV94,
    hydraulic_execution: DensInResidualCPUExecutionV95,
) -> CoupledDensInCPUExecutionV96:
    execution_document_v94b(active_execution)
    document = execution_document_v95(hydraulic_execution)
    if active_execution.semantic_id != SEMANTIC_ID_V94B:
        raise CoupledDensInCPUV96Error(
            "v96 requires the exact-zero v94b active route")
    if active_execution.backend != "cpu" or \
            hydraulic_execution.densin_mode != DENSIN_SEDIMENT_INCREMENT:
        raise CoupledDensInCPUV96Error(
            "v96 requires CPU v94b and enabled native CPU DensIn")
    v92 = _v92_execution(active_execution)
    v78 = v92.base_execution.base_execution
    profile = v78.base_execution.profile
    base = hydraulic_execution.base_execution
    if (hydraulic_execution.cell_count != active_execution.cell_count or
            hydraulic_execution.face_count != v92.face_count or
            base.fraction_count != active_execution.fraction_count or
            base.hydraulic_execution_sha256 !=
                v78.hydraulic_execution_sha256 or
            not np.array_equal(base.solid_density_kg_m3,
                               profile.sediment_density_kg_m3) or
            base.hydraulic_reference_density_kg_m3 !=
                profile.water_density_kg_m3 or
            base.gravity_m_s2 != profile.gravity_m_s2 or
            base.dry_tolerance_m != v78.base_execution.dry_tolerance_m or
            np.ptp(base.water_density_kg_m3) != 0.0):
        raise CoupledDensInCPUV96Error(
            "v96 hydraulic fractions, constants, or geometry scope differ")
    placeholder = CoupledDensInCPUExecutionV96(
        active_execution, hydraulic_execution, LOCKED_UPSTREAM_COMMIT,
        tuple(hydraulic_execution.locked_delft_source_sha256),
        SEMANTIC_ID, "")
    if not _topology_matches_v95(placeholder):
        raise CoupledDensInCPUV96Error(
            "v96 native hydraulic and v94b topologies differ")
    return CoupledDensInCPUExecutionV96(
        **{**placeholder.__dict__,
           "fingerprint": _fingerprint(_execution_payload(placeholder))})


def execution_document_v96(value: CoupledDensInCPUExecutionV96) -> dict:
    if type(value) is not CoupledDensInCPUExecutionV96:
        raise CoupledDensInCPUV96Error("v96 execution type differs")
    payload = _execution_payload(value)
    if (value.semantic_id != SEMANTIC_ID or
            value.active_execution.semantic_id != SEMANTIC_ID_V94B or
            value.locked_delft_commit != LOCKED_UPSTREAM_COMMIT or
            value.locked_delft_source_sha256 !=
                value.hydraulic_execution.locked_delft_source_sha256 or
            not _topology_matches_v95(value) or
            value.fingerprint != _fingerprint(payload)):
        raise CoupledDensInCPUV96Error(
            "v96 execution fingerprint or locked identity differs")
    return {**payload, "fingerprint": value.fingerprint}


@dataclass(frozen=True)
class CoupledDensInCPUStateV96:
    execution_fingerprint: str
    active_state: ActiveSuspendedControlStateV94
    hles_state: HlesProducerStateV90
    accepted_chain_sha256: str
    state_sha256: str

    @property
    def q(self) -> np.ndarray:
        return _v92_state(self.active_state).q

    @property
    def accepted_updates(self) -> int:
        return self.active_state.accepted_updates

    @property
    def time_s(self) -> float:
        return self.active_state.time_s


def _state_token(value: CoupledDensInCPUStateV96) -> str:
    return _fingerprint({
        "scope_id": SCOPE_ID + ":state",
        "execution_fingerprint": value.execution_fingerprint,
        "active_v94b_state_sha256": value.active_state.state_sha256,
        "hles_state_sha256": value.hles_state.state_sha256,
        "accepted_chain_sha256": value.accepted_chain_sha256,
        "shared_update": value.accepted_updates,
        "shared_time_s": value.time_s,
    })


def _zero_turbulence_scope(
    execution: CoupledDensInCPUExecutionV96,
    hles_state: HlesProducerStateV90,
) -> bool:
    hles_execution = _v92_execution(
        execution.active_execution).base_execution.hles_execution
    closure = hles_execution.closure_execution
    return bool(
        np.all(closure.background.vicuv_m2_s == 0.0) and
        np.all(closure.background.dicuv_m2_s == 0.0) and
        closure.vicmol_m2_s == 0.0 and closure.dicmol_m2_s == 0.0 and
        np.all(hles_state.accepted_dynamic_vicuv_m2_s == 0.0) and
        np.all(hles_state.accepted_dynamic_dicuv_m2_s == 0.0))


def _provider_is_closed(provider: object) -> bool:
    binding = getattr(provider, "binding", None)
    document = getattr(binding, "boundary_execution", None)
    if not isinstance(document, Mapping):
        return False
    rows = document.get("boundaries")
    return bool(isinstance(rows, list) and rows and all(
        isinstance(row, Mapping) and row.get("internal_mode") == 0 and
        row.get("public_mode_id") == "bed_transport.closed" and
        all(float(item) == 0.0 for item in row.get("source_rates", (1.0,)))
        for row in rows))


def _wrap_state(
    execution: CoupledDensInCPUExecutionV96, *,
    active_state: ActiveSuspendedControlStateV94,
    hles_state: HlesProducerStateV90,
    accepted_chain_sha256: str,
    q3_provider: object | None = None,
) -> CoupledDensInCPUStateV96:
    execution_document_v96(execution)
    validate_active_suspended_control_state_v94b(
        active_state, execution.active_execution, q3_provider)
    hles_execution = _v92_execution(
        execution.active_execution).base_execution.hles_execution
    validate_hles_producer_state_v90(hles_state, hles_execution)
    v92 = _v92_state(active_state)
    if (active_state.accepted_updates != hles_state.accepted_updates or
            active_state.time_s != hles_state.time_s or
            active_state.accepted_updates > MAX_ACCEPTED_UPDATES or
            np.any(v92.q[0] <=
                   execution.hydraulic_execution.base_execution.
                   dry_tolerance_m) or
            np.ptp(v92.water_density_cell_kg_m3) != 0.0 or
            not _zero_turbulence_scope(execution, hles_state) or
            (q3_provider is not None and not _provider_is_closed(q3_provider))):
        raise CoupledDensInCPUV96Error(
            "v96 parent clocks or narrow physical scope differ")
    chain = _sha(accepted_chain_sha256, "accepted chain")
    placeholder = CoupledDensInCPUStateV96(
        execution.fingerprint, active_state, hles_state, chain, "")
    return CoupledDensInCPUStateV96(
        **{**placeholder.__dict__, "state_sha256": _state_token(placeholder)})


def resolve_coupled_densin_cpu_state_v96(
    execution: CoupledDensInCPUExecutionV96, *,
    active_state: ActiveSuspendedControlStateV94,
    hles_state: HlesProducerStateV90,
    q3_provider: object,
) -> CoupledDensInCPUStateV96:
    if active_state.accepted_updates != 0 or hles_state.accepted_updates != 0:
        raise CoupledDensInCPUV96Error(
            "v96 fresh joint state must start at update zero")
    chain = _fingerprint({
        "scope_id": SCOPE_ID,
        "execution_fingerprint": execution.fingerprint,
        "initial_active_state_sha256": active_state.state_sha256,
        "initial_hles_state_sha256": hles_state.state_sha256,
    })
    return _wrap_state(
        execution, active_state=active_state, hles_state=hles_state,
        accepted_chain_sha256=chain, q3_provider=q3_provider)


def validate_coupled_densin_cpu_state_v96(
    state: CoupledDensInCPUStateV96,
    execution: CoupledDensInCPUExecutionV96,
    q3_provider: object | None = None,
) -> None:
    if (type(state) is not CoupledDensInCPUStateV96 or
            state.execution_fingerprint != execution.fingerprint or
            state.state_sha256 != _state_token(state)):
        raise CoupledDensInCPUV96Error(
            "v96 state is stale, mutated, or cross-execution")
    rebuilt = _wrap_state(
        execution, active_state=state.active_state,
        hles_state=state.hles_state,
        accepted_chain_sha256=state.accepted_chain_sha256,
        q3_provider=q3_provider)
    if rebuilt.state_sha256 != state.state_sha256:
        raise CoupledDensInCPUV96Error("v96 rebuilt state differs")


def make_v94b_accepted_density_receipt_v96(
    execution: CoupledDensInCPUExecutionV96,
    parent: CoupledDensInCPUStateV96,
    q3_provider: object,
) -> V94bAcceptedDensityReceiptV96:
    validate_coupled_densin_cpu_state_v96(parent, execution, q3_provider)
    v92 = _v92_state(parent.active_state)
    suspended = v92.suspended_state
    depth = _array(v92.q[0], (execution.cell_count,), "accepted depth",
                   positive=True)
    mass = _array(
        suspended.suspended_mass_kg_m2,
        (execution.fraction_count, execution.cell_count),
        "accepted suspended mass", nonnegative=True)
    concentration = np.ascontiguousarray(mass/depth[None, :])
    water = _array(
        v92.water_density_cell_kg_m3, (execution.cell_count,),
        "accepted water density", positive=True)
    mixture = mixture_density_kg_m3_v85(
        concentration, water,
        execution.hydraulic_execution.base_execution.solid_density_kg_m3)
    concentration.setflags(write=False)
    mixture.setflags(write=False)
    placeholder = V94bAcceptedDensityReceiptV96(
        execution.fingerprint, execution.active_execution.fingerprint,
        parent.state_sha256, v92.state_sha256, suspended.state_sha256,
        parent.accepted_updates, parent.time_s, depth, mass, concentration,
        water, mixture, "")
    return V94bAcceptedDensityReceiptV96(
        **{**placeholder.__dict__,
           "receipt_sha256": _density_receipt_token(placeholder)})


def validate_v94b_accepted_density_receipt_v96(
    value: V94bAcceptedDensityReceiptV96,
    execution: CoupledDensInCPUExecutionV96,
    parent: CoupledDensInCPUStateV96,
    q3_provider: object,
) -> None:
    if type(value) is not V94bAcceptedDensityReceiptV96:
        raise CoupledDensInCPUV96Error("v96 density receipt type differs")
    expected = make_v94b_accepted_density_receipt_v96(
        execution, parent, q3_provider)
    if (value.receipt_sha256 != _density_receipt_token(value) or
            value.receipt_sha256 != expected.receipt_sha256 or
            value.parent_state_sha256 != parent.state_sha256):
        raise CoupledDensInCPUV96Error(
            "v96 accepted v94b density receipt is stale or mismatched")


@dataclass(frozen=True)
class CoupledDensInOwnerCountsV96:
    native_hydraulic_owner_count: int
    water_face_flux_owner_count: int
    scalar_lifecycle_owner_count: int
    authoritative_exner_owner_count: int
    hles_clock_owner_count: int
    duplicate_hydraulic_owner_count: int
    duplicate_scalar_owner_count: int
    duplicate_exner_owner_count: int


@dataclass(frozen=True)
class CoupledDensInCPUTrialV96:
    execution_fingerprint: str
    parent_state_sha256: str
    accepted_density: V94bAcceptedDensityReceiptV96
    current_handoff: CurrentClosureHandoffV90
    dt_s: float
    midpoint_q: np.ndarray
    base_fwave_rate: np.ndarray
    density_momentum_rate: np.ndarray
    full_rate: np.ndarray
    final_residual: np.ndarray
    face_normal_acceleration_m_s2: np.ndarray
    cell_acceleration_m_s2: np.ndarray
    face_water_flux_m2_s: np.ndarray
    midpoint_depth_sha256_by_evaluation: tuple[str, ...]
    residual_evaluation_count: int
    density_source_evaluation_count: int
    augmented_riemann_call_count: int
    nonlinear_iterations: int
    nonlinear_residual_linf: float
    water_volume_residual_m3: float
    active_trial: ActiveSuspendedControlTrialV94
    hles_trial: HlesProducerTrialV90
    owner_counts: CoupledDensInOwnerCountsV96
    candidate: CoupledDensInCPUStateV96
    backend: str
    trial_sha256: str


def _counts_payload(value: CoupledDensInOwnerCountsV96) -> tuple[int, ...]:
    return tuple(value.__dict__.values())


def _trial_token(value: CoupledDensInCPUTrialV96) -> str:
    return _fingerprint({
        "scope_id": SCOPE_ID + ":trial",
        "execution_fingerprint": value.execution_fingerprint,
        "parent_state_sha256": value.parent_state_sha256,
        "accepted_density_receipt_sha256":
            value.accepted_density.receipt_sha256,
        "current_handoff_receipt_sha256":
            value.current_handoff.receipt_sha256,
        "dt_s": value.dt_s,
        "arrays_sha256": _digest(
            value.midpoint_q, value.base_fwave_rate,
            value.density_momentum_rate, value.full_rate,
            value.final_residual,
            value.face_normal_acceleration_m_s2,
            value.cell_acceleration_m_s2,
            value.face_water_flux_m2_s),
        "midpoint_depth_sha256_by_evaluation":
            list(value.midpoint_depth_sha256_by_evaluation),
        "counts": {
            "residual": value.residual_evaluation_count,
            "density": value.density_source_evaluation_count,
            "riemann": value.augmented_riemann_call_count,
            "nonlinear_iterations": value.nonlinear_iterations,
            "owners": list(_counts_payload(value.owner_counts)),
        },
        "nonlinear_residual_linf": value.nonlinear_residual_linf,
        "water_volume_residual_m3": value.water_volume_residual_m3,
        "active_trial_sha256": value.active_trial.trial_sha256,
        "hles_trial_sha256": value.hles_trial.trial_sha256,
        "candidate_state_sha256": value.candidate.state_sha256,
        "backend": value.backend,
    })


def _residual_norm(residual: np.ndarray, q: np.ndarray,
                   gravity: float) -> float:
    depth_scale = max(1.0, float(np.max(np.abs(q[0]))))
    momentum_scale = max(
        1.0, float(np.max(np.abs(q[1:]))),
        depth_scale*math.sqrt(gravity*depth_scale))
    scale = np.asarray([depth_scale, momentum_scale, momentum_scale])[:, None]
    return float(np.max(np.abs(residual)/scale))


def _closed_scalar_inputs(
    execution: CoupledDensInCPUExecutionV96,
    parent: CoupledDensInCPUStateV96,
    endpoint_q: np.ndarray,
    shear_velocity_m_s: object,
) -> dict[str, object]:
    v92_execution = _v92_execution(execution.active_execution)
    v92 = _v92_state(parent.active_state)
    grid = _grid_execution(execution.active_execution)
    ng, nc, nf, nb = (execution.fraction_count, execution.cell_count,
                      execution.face_count, v92_execution.boundary_count)
    shear = _array(
        shear_velocity_m_s, (nc,), "shear velocity", nonnegative=True)
    concentration = np.ascontiguousarray(
        v92.suspended_state.suspended_mass_kg_m2/v92.q[0][None, :])
    left = np.ascontiguousarray(concentration[:, grid.owner])
    right = np.array(left, copy=True, order="C")
    internal = grid.neighbour >= 0
    right[:, internal] = concentration[:, grid.neighbour[internal]]
    fluid = np.vstack((v92.temperature_cell_c, v92.salinity_cell_ppt))
    fluid_left = np.ascontiguousarray(fluid[:, grid.owner])
    fluid_right = np.array(fluid_left, copy=True, order="C")
    fluid_right[:, internal] = fluid[:, grid.neighbour[internal]]
    prescribed = np.zeros((ng, nb), dtype=np.float64)
    prescribed_fluid = np.broadcast_to(
        np.asarray((v92_execution.equilibrium_temperature_c,
                    v92_execution.equilibrium_salinity_ppt))[:, None],
        (2, nb)).copy()
    boundary_faces = np.flatnonzero(grid.boundary_index >= 0)
    seen: set[int] = set()
    for face in boundary_faces:
        boundary = int(grid.boundary_index[face])
        if boundary not in seen:
            owner = int(grid.owner[face])
            prescribed[:, boundary] = concentration[:, owner]
            prescribed_fluid[:, boundary] = fluid[:, owner]
            seen.add(boundary)
    face_depth = np.asarray(endpoint_q[0, grid.owner], dtype=np.float64).copy()
    face_depth[internal] = 0.5*(
        endpoint_q[0, grid.owner[internal]]+
        endpoint_q[0, grid.neighbour[internal]])
    return {
        "reconstructed_left_fluid": fluid_left,
        "reconstructed_right_fluid": fluid_right,
        "prescribed_boundary_fluid": prescribed_fluid,
        "reconstructed_left_concentration_kg_m3": left,
        "reconstructed_right_concentration_kg_m3": right,
        "prescribed_boundary_concentration_kg_m3": prescribed,
        "accepted_face_exchange_area_m2": np.ascontiguousarray(
            np.where(internal, face_depth*grid.face_length_m, 0.0)),
        "transport_face_active": np.ascontiguousarray(internal),
        "shear_velocity_m_s": shear,
        "hydro_constant": np.ones(nc, dtype=np.bool_),
    }


def _face_normal_velocity(
    execution: CoupledDensInCPUExecutionV96,
    endpoint_q: np.ndarray,
    face_water_flux_m2_s: np.ndarray,
) -> np.ndarray:
    grid = _grid_execution(execution.active_execution)
    internal = grid.neighbour >= 0
    depth = np.asarray(endpoint_q[0, grid.owner], dtype=np.float64).copy()
    depth[internal] = 0.5*(
        endpoint_q[0, grid.owner[internal]]+
        endpoint_q[0, grid.neighbour[internal]])
    if np.any(depth <= 0.0):
        raise CoupledDensInCPUV96Error(
            "v96 HLES face depth left fully-wet scope")
    return np.ascontiguousarray(face_water_flux_m2_s/depth)


def _restore_provider_after_failure(
    trial: ActiveSuspendedControlTrialV94 | None,
    provider: object,
    checkpoint: Mapping[str, object],
) -> None:
    if trial is not None:
        base = trial.base_v93_trial
        try:
            if getattr(provider, "_closed", None) is base.q3_prepared_commit:
                provider.abort_closed_commit(base.q3_prepared_commit)
            elif getattr(provider, "_open", None) is base.q3_open_trial:
                provider.rollback(base.q3_open_trial)
        except Exception:
            pass
    if (getattr(provider, "_open", None) is not None or
            getattr(provider, "_prepared", None) is not None or
            getattr(provider, "_closed", None) is not None or
            provider.checkpoint() != checkpoint):
        provider.restore_checkpoint(checkpoint)


def begin_coupled_densin_cpu_trial_v96(
    execution: CoupledDensInCPUExecutionV96,
    parent: CoupledDensInCPUStateV96,
    q3_provider: object, *,
    native_runtime: NativeDensInRuntimeV95,
    shear_velocity_m_s: object,
    q3_inputs: Mapping[str, object],
    dt_s: float,
    cancelled: bool | Callable[[], bool] = False,
) -> CoupledDensInCPUTrialV96:
    """Open one native-hydraulic plus scalar/Q3/Exner/HLES transaction."""
    execution_document_v96(execution)
    validate_coupled_densin_cpu_state_v96(parent, execution, q3_provider)
    if type(native_runtime) is not NativeDensInRuntimeV95 or \
            native_runtime.receipt != \
                execution.hydraulic_execution.native_build_receipt:
        raise CoupledDensInCPUV96Error("v96 native runtime identity differs")
    if parent.accepted_updates >= MAX_ACCEPTED_UPDATES:
        raise CoupledDensInCPUV96Error(
            "v96 accepted update limit was reached")
    if not isinstance(q3_inputs, Mapping):
        raise CoupledDensInCPUV96Error("v96 Q3 inputs must be a mapping")
    if _cancelled(cancelled):
        raise InterruptedError("v96 cancelled before native hydraulic work")
    dt = _finite(dt_s, "timestep", positive=True)
    hydraulic = execution.hydraulic_execution
    if dt > hydraulic.maximum_dt_s:
        raise CoupledDensInCPUV96Error(
            "v96 timestep exceeds the native v95 bound")
    density = make_v94b_accepted_density_receipt_v96(
        execution, parent, q3_provider)
    mixture = np.ascontiguousarray(density.mixture_density_kg_m3)
    mixture_sha = _digest(mixture)
    midpoint = np.array(parent.q, copy=True, order="C")
    depth_digests: list[str] = []
    riemann_calls = density_calls = nonlinear_iterations = 0
    result: NativeRateResultV95 | None = None
    residual: np.ndarray | None = None
    residual_norm = math.inf
    for evaluation in range(1, hydraulic.max_nonlinear_iterations+2):
        if _cancelled(cancelled):
            raise InterruptedError("v96 cancelled during native midpoint")
        result = native_runtime.evaluate(hydraulic, midpoint, mixture)
        if _digest(mixture) != mixture_sha:
            raise CoupledDensInCPUV96Error(
                "v96 native solve mutated accepted v94b density")
        depth_digests.append(_digest(midpoint[0]))
        riemann_calls += result.augmented_riemann_call_count
        density_calls += result.density_source_evaluation_count
        residual = np.ascontiguousarray(
            midpoint-parent.q-0.5*dt*result.full_rate)
        residual_norm = _residual_norm(
            residual, midpoint,
            hydraulic.base_execution.gravity_m_s2)
        if residual_norm <= hydraulic.nonlinear_tolerance:
            break
        if evaluation > hydraulic.max_nonlinear_iterations:
            raise CoupledDensInCPUV96Error(
                "v96 native implicit midpoint failed to converge")
        midpoint = np.ascontiguousarray(midpoint-residual)
        if (not np.isfinite(midpoint).all() or
                np.any(midpoint[0] <=
                       hydraulic.base_execution.dry_tolerance_m)):
            raise CoupledDensInCPUV96Error(
                "v96 midpoint left the fully-wet scope")
        nonlinear_iterations += 1
    if result is None or residual is None:
        raise CoupledDensInCPUV96Error("v96 native solve produced no result")
    endpoint = np.ascontiguousarray(parent.q+dt*result.full_rate)
    if np.any(endpoint[0] <= hydraulic.base_execution.dry_tolerance_m):
        raise CoupledDensInCPUV96Error(
            "v96 native endpoint left the fully-wet scope")
    boundary = hydraulic.base_execution.neighbour < 0
    boundary_outflow = float(np.sum(
        hydraulic.base_execution.face_length_m[boundary]*
        result.face_water_flux_m2_s[boundary]))
    water_residual = float(np.sum(
        hydraulic.base_execution.cell_area_m2*(endpoint[0]-parent.q[0]))+
        dt*boundary_outflow)
    water_scale = max(1.0, float(np.sum(
        hydraulic.base_execution.cell_area_m2*parent.q[0])))
    if abs(water_residual) > 128*np.finfo(float).eps*water_scale:
        raise CoupledDensInCPUV96Error(
            "v96 native closed water budget differs")
    request = _fingerprint({
        "scope_id": SCOPE_ID + ":hydraulic-request",
        "execution_fingerprint": execution.fingerprint,
        "parent_state_sha256": parent.state_sha256,
        "accepted_density_receipt_sha256": density.receipt_sha256,
        "dt_s": dt,
    })
    hles_execution = _v92_execution(
        execution.active_execution).base_execution.hles_execution
    handoff = make_current_closure_handoff_v90(
        hles_execution, parent.hles_state,
        hydraulic_trial_request_sha256=request)
    scalar = _closed_scalar_inputs(
        execution, parent, endpoint, shear_velocity_m_s)
    v92_inputs = {
        "augmented_fwave_base_rate_m2_s2": result.full_rate,
        "accepted_water_flux_m2_s": result.face_water_flux_m2_s,
        "dt_s": dt,
        **scalar,
    }
    provider_checkpoint = q3_provider.checkpoint()
    active_trial: ActiveSuspendedControlTrialV94 | None = None
    try:
        if _cancelled(cancelled):
            raise InterruptedError("v96 cancelled before v94b child work")
        active_trial = begin_active_suspended_control_cpu_trial_v94b(
            execution.active_execution, parent.active_state,
            parent.hles_state, handoff, q3_provider,
            v92_inputs=v92_inputs, q3_inputs=dict(q3_inputs),
            cancelled=(cancelled if callable(cancelled)
                       else lambda: bool(cancelled)))
        base_trial = active_trial.base_v93_trial.base_v92_trial
        endpoint_equal = np.array_equal(base_trial.candidate.q, endpoint)
        face_flux_equal = np.array_equal(
            base_trial.hydraulic_face_receipt.accepted_water_flux_m2_s,
            result.face_water_flux_m2_s)
        zero_vicuv = not np.any(
            base_trial.cap_receipt.capped_total_vicuv_m2_s != 0.0)
        zero_dicuv = not np.any(
            base_trial.cap_receipt.capped_total_dicuv_m2_s != 0.0)
        zero_rate = not np.any(base_trial.momentum_rate_m2_s2 != 0.0)
        zero_flux = not np.any(base_trial.momentum_face_flux_m4_s2 != 0.0)
        if (not endpoint_equal or
                not face_flux_equal or not zero_vicuv or not zero_dicuv or
                not zero_rate or not zero_flux):
            raise CoupledDensInCPUV96Error(
                "v96 nested v92 did not consume the sole native endpoint "
                f"(endpoint={endpoint_equal}, face_flux={face_flux_equal}, "
                f"vicuv={zero_vicuv}, "
                f"dicuv={zero_dicuv}, rate={zero_rate}, flux={zero_flux})")
        if _cancelled(cancelled):
            raise InterruptedError("v96 cancelled after v94b child work")
        face_velocity = _face_normal_velocity(
            execution, endpoint, result.face_water_flux_m2_s)
        inactive_face = np.zeros(execution.face_count, dtype=np.int32)
        inactive_cell = np.zeros(execution.cell_count, dtype=np.int32)
        closure_inputs = {
            "accepted_face_normal_velocity_m_s": face_velocity,
            "active_face": inactive_face,
            "closure_depth_m": endpoint[0],
            "closure_chezy_mhalf_s": np.ones(execution.cell_count),
            "active_cell": inactive_cell,
        }
        step = base_trial.hydraulic_receipt.base_step_receipt
        hles_input = make_hles_input_receipt_v90(
            hles_execution, parent.hles_state,
            accepted_step_receipt=step, current_handoff=handoff,
            closure_interval_s=dt, **closure_inputs)
        hles_trial = begin_hles_cpu_trial_v90(
            hles_execution, parent.hles_state,
            accepted_step_receipt=step,
            hles_input_receipt=hles_input, current_handoff=handoff,
            cancelled=False, **closure_inputs)
        if (np.any(hles_trial.candidate.accepted_dynamic_vicuv_m2_s != 0.0)
                or np.any(hles_trial.candidate.
                          accepted_dynamic_dicuv_m2_s != 0.0)):
            raise CoupledDensInCPUV96Error(
                "v96 zero turbulence candidate produced nonzero fields")
        chain = _fingerprint({
            "previous": parent.accepted_chain_sha256,
            "accepted_density": density.receipt_sha256,
            "native_build": hydraulic.native_build_receipt.receipt_sha256,
            "native_midpoint_sha256": _digest(midpoint),
            "native_full_rate_sha256": _digest(result.full_rate),
            "native_face_water_flux_sha256":
                _digest(result.face_water_flux_m2_s),
            "active_trial": active_trial.trial_sha256,
            "hles_trial": hles_trial.trial_sha256,
            "dt_s": dt,
        })
        candidate = _wrap_state(
            execution, active_state=active_trial.candidate,
            hles_state=hles_trial.candidate,
            accepted_chain_sha256=chain)
        counts = CoupledDensInOwnerCountsV96(1, 1, 1, 1, 1, 0, 0, 0)
        placeholder = CoupledDensInCPUTrialV96(
            execution.fingerprint, parent.state_sha256, density, handoff, dt,
            _array(midpoint, (3, execution.cell_count), "midpoint"),
            result.base_fwave_rate, result.density_momentum_rate,
            result.full_rate,
            _array(residual, (3, execution.cell_count), "final residual"),
            result.face_normal_acceleration_m_s2,
            result.cell_acceleration_m_s2,
            result.face_water_flux_m2_s, tuple(depth_digests),
            len(depth_digests), density_calls, riemann_calls,
            nonlinear_iterations, residual_norm, water_residual,
            active_trial, hles_trial, counts, candidate, BACKEND, "")
        trial = CoupledDensInCPUTrialV96(
            **{**placeholder.__dict__,
               "trial_sha256": _trial_token(placeholder)})
        _validate_trial(execution, parent, q3_provider, trial)
        return trial
    except Exception as original_error:
        try:
            _restore_provider_after_failure(
                active_trial, q3_provider, provider_checkpoint)
        except Exception as cleanup_error:
            raise RuntimeError(
                "v96 begin failed and provider restore also failed") \
                from cleanup_error
        raise original_error


def _validate_trial(
    execution: CoupledDensInCPUExecutionV96,
    parent: CoupledDensInCPUStateV96,
    q3_provider: object,
    trial: CoupledDensInCPUTrialV96,
) -> None:
    validate_coupled_densin_cpu_state_v96(parent, execution, q3_provider)
    if (type(trial) is not CoupledDensInCPUTrialV96 or
            trial.execution_fingerprint != execution.fingerprint or
            trial.parent_state_sha256 != parent.state_sha256 or
            trial.backend != BACKEND or
            trial.trial_sha256 != _trial_token(trial)):
        raise CoupledDensInCPUV96Error(
            "v96 trial is stale, mutated, or cross-parent")
    validate_v94b_accepted_density_receipt_v96(
        trial.accepted_density, execution, parent, q3_provider)
    hydraulic = execution.hydraulic_execution
    nc, nf = execution.cell_count, execution.face_count
    midpoint = _array(trial.midpoint_q, (3, nc), "trial midpoint")
    base_rate = _array(trial.base_fwave_rate, (3, nc), "base rate")
    density_rate = _array(
        trial.density_momentum_rate, (2, nc), "density rate")
    full_rate = _array(trial.full_rate, (3, nc), "full rate")
    stored_residual = _array(
        trial.final_residual, (3, nc), "stored residual")
    face_acceleration = _array(
        trial.face_normal_acceleration_m_s2, (nf,), "face acceleration")
    cell_acceleration = _array(
        trial.cell_acceleration_m_s2, (2, nc), "cell acceleration")
    face_flux = _array(
        trial.face_water_flux_m2_s, (nf,), "face water flux")
    dt = _finite(trial.dt_s, "trial timestep", positive=True)
    if dt > hydraulic.maximum_dt_s:
        raise CoupledDensInCPUV96Error("v96 trial timestep exceeds bound")
    expected_full = np.array(base_rate, copy=True)
    expected_full[1:] += density_rate
    residual = midpoint-parent.q-0.5*dt*full_rate
    endpoint = parent.q+dt*full_rate
    norm = _residual_norm(
        residual, midpoint, hydraulic.base_execution.gravity_m_s2)
    boundary = hydraulic.base_execution.neighbour < 0
    boundary_outflow = float(np.sum(
        hydraulic.base_execution.face_length_m[boundary]*face_flux[boundary]))
    water_residual = float(np.sum(
        hydraulic.base_execution.cell_area_m2*(endpoint[0]-parent.q[0]))+
        dt*boundary_outflow)
    expected_face = delft_kmax1_face_acceleration_v85(
        midpoint[0], trial.accepted_density.mixture_density_kg_m3,
        hydraulic.base_execution.owner, hydraulic.base_execution.neighbour,
        hydraulic.base_execution.face_distance_m,
        gravity_m_s2=hydraulic.base_execution.gravity_m_s2,
        hydraulic_reference_density_kg_m3=
            hydraulic.base_execution.hydraulic_reference_density_kg_m3)
    expected_cell = project_face_acceleration_v85(
        expected_face, hydraulic.base_execution.normal,
        hydraulic.base_execution.face_length_m,
        hydraulic.base_execution.cell_face_offset,
        hydraulic.base_execution.cell_faces,
        hydraulic.base_execution.cell_face_sign)
    expected_density = midpoint[0][None, :]*expected_cell
    expected_counts = CoupledDensInOwnerCountsV96(1, 1, 1, 1, 1, 0, 0, 0)
    if (trial.owner_counts != expected_counts or
            trial.candidate.accepted_updates != parent.accepted_updates+1 or
            trial.candidate.time_s != parent.time_s+dt or
            trial.residual_evaluation_count < 1 or
            trial.residual_evaluation_count >
                hydraulic.max_nonlinear_iterations+1 or
            trial.nonlinear_iterations != trial.residual_evaluation_count-1 or
            trial.density_source_evaluation_count !=
                trial.residual_evaluation_count or
            trial.augmented_riemann_call_count !=
                nf*trial.residual_evaluation_count or
            len(trial.midpoint_depth_sha256_by_evaluation) !=
                trial.residual_evaluation_count or
            trial.midpoint_depth_sha256_by_evaluation[-1] !=
                _digest(midpoint[0]) or
            any(_SHA256.fullmatch(item) is None for item in
                trial.midpoint_depth_sha256_by_evaluation) or
            not np.array_equal(expected_full, full_rate) or
            not np.array_equal(residual, stored_residual) or
            not np.array_equal(endpoint, trial.candidate.q) or
            trial.nonlinear_residual_linf != norm or
            norm > hydraulic.nonlinear_tolerance or
            trial.water_volume_residual_m3 != water_residual):
        raise CoupledDensInCPUV96Error(
            "v96 residual, endpoint, counts, or ledgers differ")
    scale = max(1.0, float(np.max(np.abs(expected_density))))
    if (not np.allclose(expected_face, face_acceleration,
                        rtol=8*np.finfo(float).eps, atol=1e-18) or
            not np.allclose(expected_cell, cell_acceleration,
                            rtol=16*np.finfo(float).eps, atol=1e-18) or
            not np.allclose(expected_density, density_rate,
                            rtol=16*np.finfo(float).eps,
                            atol=16*np.finfo(float).eps*scale)):
        raise CoupledDensInCPUV96Error(
            "v96 native density source differs from the v85 oracle")
    v92_trial = trial.active_trial.base_v93_trial.base_v92_trial
    if (v92_trial.candidate.q is not trial.active_trial.candidate.
            base_state.base_state.q and
            not np.array_equal(v92_trial.candidate.q,
                               trial.active_trial.candidate.
                               base_state.base_state.q)):
        raise CoupledDensInCPUV96Error(
            "v96 v92 and v94b hydraulic candidates differ")
    if (not np.array_equal(v92_trial.candidate.q, endpoint) or
            not np.array_equal(
                v92_trial.hydraulic_face_receipt.
                accepted_water_flux_m2_s, face_flux)):
        raise CoupledDensInCPUV96Error(
            "v96 nested scalar owner did not consume native hydraulics")
    _v94._validate_trial(
        execution.active_execution, parent.active_state, parent.hles_state,
        trial.current_handoff, trial.active_trial)
    if rollback_hles_trial_v90(
            _v92_execution(execution.active_execution).base_execution.
            hles_execution,
            parent.hles_state, trial.hles_trial) is not parent.hles_state:
        raise CoupledDensInCPUV96Error("v96 HLES rollback validation differs")
    expected_chain = _fingerprint({
        "previous": parent.accepted_chain_sha256,
        "accepted_density": trial.accepted_density.receipt_sha256,
        "native_build": hydraulic.native_build_receipt.receipt_sha256,
        "native_midpoint_sha256": _digest(midpoint),
        "native_full_rate_sha256": _digest(full_rate),
        "native_face_water_flux_sha256": _digest(face_flux),
        "active_trial": trial.active_trial.trial_sha256,
        "hles_trial": trial.hles_trial.trial_sha256,
        "dt_s": dt,
    })
    rebuilt = _wrap_state(
        execution, active_state=trial.active_trial.candidate,
        hles_state=trial.hles_trial.candidate,
        accepted_chain_sha256=expected_chain)
    if (trial.candidate.accepted_chain_sha256 != expected_chain or
            trial.candidate.state_sha256 != rebuilt.state_sha256):
        raise CoupledDensInCPUV96Error(
            "v96 joint candidate chain differs")


def commit_coupled_densin_cpu_trial_v96(
    execution: CoupledDensInCPUExecutionV96,
    parent: CoupledDensInCPUStateV96,
    q3_provider: object,
    trial: CoupledDensInCPUTrialV96,
) -> CoupledDensInCPUStateV96:
    # A successful begin deliberately leaves the exact Q3 child open, so the
    # provider cannot emit a new committed checkpoint here.  The authenticated
    # v93 child already carries the byte-for-byte parent checkpoint captured
    # before it opened.
    checkpoint = trial.active_trial.base_v93_trial.q3_parent_checkpoint
    try:
        _validate_trial(execution, parent, q3_provider, trial)
        hles = commit_hles_trial_v90(
            _v92_execution(execution.active_execution).base_execution.
            hles_execution,
            parent.hles_state, trial.hles_trial)
        if hles is not trial.hles_trial.candidate:
            raise CoupledDensInCPUV96Error(
                "v96 HLES commit returned a substituted candidate")
    except Exception as original_error:
        try:
            _restore_provider_after_failure(
                getattr(trial, "active_trial", None), q3_provider,
                checkpoint)
        except Exception as cleanup_error:
            raise RuntimeError(
                "v96 pre-publication commit failed and restore also failed") \
                from cleanup_error
        raise original_error
    # The v94b/Q3 provider publication is deliberately the last fallible
    # operation.  The joint candidate was completely authenticated above.
    commit_active_suspended_control_trial_v94b(
        execution.active_execution, parent.active_state, parent.hles_state,
        trial.current_handoff, trial.active_trial)
    return trial.candidate


def rollback_coupled_densin_cpu_trial_v96(
    execution: CoupledDensInCPUExecutionV96,
    parent: CoupledDensInCPUStateV96,
    q3_provider: object,
    trial: CoupledDensInCPUTrialV96,
) -> CoupledDensInCPUStateV96:
    _validate_trial(execution, parent, q3_provider, trial)
    if rollback_hles_trial_v90(
            _v92_execution(execution.active_execution).base_execution.
            hles_execution,
            parent.hles_state, trial.hles_trial) is not parent.hles_state:
        raise CoupledDensInCPUV96Error("v96 rollback changed HLES parent")
    active = rollback_active_suspended_control_trial_v94b(
        execution.active_execution, parent.active_state, parent.hles_state,
        trial.current_handoff, trial.active_trial)
    if active is not parent.active_state:
        raise CoupledDensInCPUV96Error("v96 rollback changed active parent")
    return parent


def cancel_coupled_densin_cpu_trial_v96(*args, **kwargs):
    return rollback_coupled_densin_cpu_trial_v96(*args, **kwargs)


def coupled_densin_cpu_checkpoint_v96(
    execution: CoupledDensInCPUExecutionV96,
    state: CoupledDensInCPUStateV96,
    q3_provider: object,
) -> dict[str, object]:
    validate_coupled_densin_cpu_state_v96(state, execution, q3_provider)
    hles_execution = _v92_execution(
        execution.active_execution).base_execution.hles_execution
    body = {
        "schema": SCHEMA,
        "scope_id": CHECKPOINT_SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "execution_fingerprint": execution.fingerprint,
        "active_v94b_checkpoint":
            checkpoint_active_suspended_control_state_v94b(
                execution.active_execution, state.active_state, q3_provider),
        "hles_checkpoint": hles_checkpoint_v90(
            hles_execution, state.hles_state),
        "accepted_chain_sha256": state.accepted_chain_sha256,
        "state_sha256": state.state_sha256,
    }
    return {**body, "fingerprint": _fingerprint(body)}


def load_coupled_densin_cpu_checkpoint_v96(
    execution: CoupledDensInCPUExecutionV96,
    document: Mapping[str, object],
    q3_provider: object,
) -> CoupledDensInCPUStateV96:
    expected = {
        "schema", "scope_id", "semantic_id", "execution_fingerprint",
        "active_v94b_checkpoint", "hles_checkpoint",
        "accepted_chain_sha256", "state_sha256", "fingerprint",
    }
    if not isinstance(document, Mapping) or set(document) != expected:
        raise CoupledDensInCPUV96Error("v96 checkpoint inventory differs")
    body = {key: document[key] for key in document if key != "fingerprint"}
    if (document["fingerprint"] != _fingerprint(body) or
            document["schema"] != SCHEMA or
            document["scope_id"] != CHECKPOINT_SCOPE_ID or
            document["semantic_id"] != SEMANTIC_ID or
            document["execution_fingerprint"] != execution.fingerprint):
        raise CoupledDensInCPUV96Error("v96 checkpoint identity differs")
    hles_execution = _v92_execution(
        execution.active_execution).base_execution.hles_execution
    hles = load_hles_checkpoint_v90(
        hles_execution, document["hles_checkpoint"])
    provider_checkpoint = q3_provider.checkpoint()
    try:
        active = load_active_suspended_control_checkpoint_v94b(
            execution.active_execution,
            document["active_v94b_checkpoint"], q3_provider)
        state = _wrap_state(
            execution, active_state=active, hles_state=hles,
            accepted_chain_sha256=document["accepted_chain_sha256"],
            q3_provider=q3_provider)
        if state.state_sha256 != document["state_sha256"]:
            raise CoupledDensInCPUV96Error(
                "v96 checkpoint state digest differs")
        return state
    except Exception as original_error:
        try:
            q3_provider.restore_checkpoint(provider_checkpoint)
        except Exception as cleanup_error:
            raise RuntimeError(
                "v96 checkpoint load failed and provider restore also failed") \
                from cleanup_error
        raise original_error


def method_identity_v96() -> dict[str, object]:
    return {
        "scope_id": SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "backend": BACKEND,
        "density_source": "genuine_accepted_v94b_state_receipt",
        "hydraulics": "native_v95_openmp_augmented_fwave_midpoint",
        "joint_state": "v94b_q_plus_hles_no_shadow_q",
        "native_endpoint_consumers": [
            "one_v92_scalar_lifecycle", "one_q3_exner_owner",
            "one_zero_hles_clock_owner"],
        "commit_order": "validate_all_then_hles_pure_then_q3_publish_last",
        "accepted_update_limit": MAX_ACCEPTED_UPDATES,
        "product_admission": False,
        "not_claimed": [
            "open_hydraulic_or_sediment_boundaries",
            "nonzero_turbulence_or_diffusion", "nonflat_bed",
            "spatial_water_density", "moving_wet_dry", "AMR",
            "cut_cells", "other_legal_Delft_component_combinations",
            "plugin_support", "long_trajectory_qualification",
        ],
    }


__all__ = [
    "BACKEND", "MAX_ACCEPTED_UPDATES", "SEMANTIC_ID",
    "CoupledDensInCPUExecutionV96", "CoupledDensInCPUStateV96",
    "CoupledDensInCPUTrialV96", "CoupledDensInCPUV96Error",
    "CoupledDensInOwnerCountsV96", "V94bAcceptedDensityReceiptV96",
    "begin_coupled_densin_cpu_trial_v96",
    "cancel_coupled_densin_cpu_trial_v96",
    "commit_coupled_densin_cpu_trial_v96",
    "coupled_densin_cpu_checkpoint_v96", "execution_document_v96",
    "load_coupled_densin_cpu_checkpoint_v96",
    "make_v94b_accepted_density_receipt_v96", "method_identity_v96",
    "resolve_coupled_densin_cpu_execution_v96",
    "resolve_coupled_densin_cpu_state_v96",
    "rollback_coupled_densin_cpu_trial_v96",
    "validate_coupled_densin_cpu_state_v96",
    "validate_v94b_accepted_density_receipt_v96",
]
