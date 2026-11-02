"""Bounded Q4 multi-fraction suspended lifecycle composition (v78).

This additive layer authenticates a two-or-more-class noncohesive lifecycle
without changing the accepted v74 route owner or v76 lifecycle.  The CPU path
is an isolated oracle built from the accepted v76 implementation.  The CUDA
owner lives in :mod:`braids_gpu.multifraction_suspended_lifecycle_v78`.

Every trial is tied to one accepted, fixed-grid
``implicit_fwave_midpoint`` hydraulic-step receipt.  The receipt, classwise
water/bed state, classwise operator evidence, and one aggregate bed-plus-water
ledger are committed as one immutable transaction.  Four accepted updates is
the deliberately small validation bound; this module is not a long-run or
product admission.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Mapping

import numpy as np

from .suspended_lifecycle_live_v76 import (
    BACKEND_CPU,
    BACKEND_CUDA,
    MAX_ACCEPTED_UPDATES as V76_MAX_ACCEPTED_UPDATES,
    SuspendedLifecycleExecutionV76,
    SuspendedLifecycleStateV76,
    SuspendedLifecycleTrialV76,
    begin_suspended_lifecycle_trial_v76,
    execution_document_v76,
    load_suspended_lifecycle_checkpoint_v76,
    resolve_suspended_lifecycle_state_v76,
    suspended_lifecycle_checkpoint_v76,
    validate_suspended_lifecycle_state_v76,
)


SCHEMA = 1
SCOPE_ID = "braids-q4-multifraction-suspended-lifecycle-v78-v1"
CHECKPOINT_SCOPE_ID = SCOPE_ID + ":checkpoint"
STEP_RECEIPT_SCOPE_ID = SCOPE_ID + ":accepted-hydraulic-step"
SEMANTIC_ID = hashlib.sha256(SCOPE_ID.encode("ascii")).hexdigest()
INTEGRATOR = "implicit_fwave_midpoint"
HYDRAULIC_BACKEND = "implicit_fwave_midpoint_gpu"
ROLE_CPU_ORACLE = "isolated_cpu_oracle"
ROLE_CUDA = "actual_cuda"
ROLES = (ROLE_CPU_ORACLE, ROLE_CUDA)
MIN_FRACTIONS = 2
MAX_FRACTIONS = 3
# Keep the accepted v76 bound.  It is stricter than the requested <=20 cap.
MAX_ACCEPTED_UPDATES = V76_MAX_ACCEPTED_UPDATES
if MAX_ACCEPTED_UPDATES > 20:  # pragma: no cover - import-time invariant
    raise RuntimeError("v78 bounded update limit exceeds twenty")

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class MultifractionSuspendedV78Error(ValueError):
    """Fail-closed v78 identity, receipt, transaction, or ledger error."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, OverflowError) as error:
        raise MultifractionSuspendedV78Error(
            "v78 value is not finite canonical JSON") from error


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _finite(value: object, name: str, *, positive: bool = False) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(float(value))):
        raise MultifractionSuspendedV78Error(
            f"v78 {name} must be finite")
    result = float(value)
    if positive and result <= 0.0:
        raise MultifractionSuspendedV78Error(
            f"v78 {name} must be positive")
    return result


def _digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(_canonical(list(array.shape)))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise MultifractionSuspendedV78Error(
            f"v78 {name} must be a lowercase SHA-256 digest")
    return value


def make_implicit_fwave_midpoint_step_receipt_v78(
    *, hydraulic_execution_sha256: str, accepted_step_sha256: str,
    accepted_update: int, time_before_s: float, dt_s: float,
) -> dict:
    """Create the only accepted v78 hydraulic handoff receipt.

    ``accepted_step_sha256`` is supplied by the hydraulic transaction owner;
    this helper does not infer acceptance from a stable-looking state.
    """
    execution_sha = _require_sha256(
        hydraulic_execution_sha256, "hydraulic execution digest")
    step_sha = _require_sha256(
        accepted_step_sha256, "accepted hydraulic step digest")
    if (isinstance(accepted_update, bool) or type(accepted_update) is not int or
            not 1 <= accepted_update <= MAX_ACCEPTED_UPDATES):
        raise MultifractionSuspendedV78Error(
            "v78 accepted hydraulic update is outside the bounded scope")
    start = _finite(time_before_s, "hydraulic receipt start time")
    if start < 0.0:
        raise MultifractionSuspendedV78Error(
            "v78 hydraulic receipt start time must be nonnegative")
    dt = _finite(dt_s, "hydraulic receipt timestep", positive=True)
    body = {
        "schema": SCHEMA,
        "scope_id": STEP_RECEIPT_SCOPE_ID,
        "integrator": INTEGRATOR,
        "hydraulic_backend": HYDRAULIC_BACKEND,
        "fixed_grid": True,
        "accepted": True,
        "automatic_cpu_fallback": False,
        "hydraulic_execution_sha256": execution_sha,
        "accepted_step_sha256": step_sha,
        "accepted_update": accepted_update,
        "time_before_s": start,
        "dt_s": dt,
        "time_after_s": start + dt,
    }
    return {**body, "fingerprint": _fingerprint(body)}


def validate_implicit_fwave_midpoint_step_receipt_v78(
    value: Mapping[str, object], *, hydraulic_execution_sha256: str,
    parent_accepted_updates: int, parent_time_s: float, dt_s: float,
) -> dict:
    expected = {
        "schema", "scope_id", "integrator", "hydraulic_backend",
        "fixed_grid", "accepted", "automatic_cpu_fallback",
        "hydraulic_execution_sha256", "accepted_step_sha256",
        "accepted_update", "time_before_s", "dt_s", "time_after_s",
        "fingerprint",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise MultifractionSuspendedV78Error(
            "v78 hydraulic step receipt inventory differs")
    body = {key: value[key] for key in value if key != "fingerprint"}
    if (value["fingerprint"] != _fingerprint(body) or
            value["schema"] != SCHEMA or
            value["scope_id"] != STEP_RECEIPT_SCOPE_ID or
            value["integrator"] != INTEGRATOR or
            value["hydraulic_backend"] != HYDRAULIC_BACKEND or
            value["fixed_grid"] is not True or
            value["accepted"] is not True or
            value["automatic_cpu_fallback"] is not False):
        raise MultifractionSuspendedV78Error(
            "v78 requires an accepted fixed-grid implicit_fwave_midpoint "
            "receipt without CPU fallback")
    _require_sha256(value["accepted_step_sha256"],
                    "accepted hydraulic step digest")
    if (value["hydraulic_execution_sha256"] !=
            _require_sha256(hydraulic_execution_sha256,
                            "hydraulic execution digest") or
            value["accepted_update"] != parent_accepted_updates + 1 or
            value["time_before_s"] != float(parent_time_s) or
            value["dt_s"] != float(dt_s) or
            value["time_after_s"] != float(parent_time_s) + float(dt_s)):
        raise MultifractionSuspendedV78Error(
            "v78 hydraulic step receipt is stale or belongs to another update")
    return dict(value)


@dataclass(frozen=True)
class MultifractionSuspendedExecutionV78:
    role: str
    base_execution: SuspendedLifecycleExecutionV76
    hydraulic_execution_sha256: str
    semantic_id: str
    fingerprint: str

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


def _execution_payload(value: MultifractionSuspendedExecutionV78) -> dict:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "semantic_id": value.semantic_id,
        "role": value.role,
        "base_v76_execution": execution_document_v76(value.base_execution),
        "hydraulic_execution_sha256": value.hydraulic_execution_sha256,
        "integrator": INTEGRATOR,
        "hydraulic_backend": HYDRAULIC_BACKEND,
        "fixed_grid": True,
        "automatic_cpu_fallback": False,
        "minimum_fraction_count": MIN_FRACTIONS,
        "maximum_fraction_count": MAX_FRACTIONS,
        "accepted_update_limit": MAX_ACCEPTED_UPDATES,
    }


def resolve_multifraction_suspended_execution_v78(
    base_execution: SuspendedLifecycleExecutionV76, *, role: str,
    hydraulic_execution_sha256: str,
) -> MultifractionSuspendedExecutionV78:
    execution_document_v76(base_execution)
    if role not in ROLES:
        raise MultifractionSuspendedV78Error(
            "v78 role must be isolated_cpu_oracle or actual_cuda")
    expected_backend = BACKEND_CPU if role == ROLE_CPU_ORACLE else BACKEND_CUDA
    if base_execution.backend != expected_backend:
        raise MultifractionSuspendedV78Error(
            "v78 role and accepted v76 backend differ")
    if not MIN_FRACTIONS <= base_execution.fraction_count <= MAX_FRACTIONS:
        raise MultifractionSuspendedV78Error(
            "v78 requires two or three authenticated sand/gravel fractions")
    if any(kind != "sand" for kind in
           base_execution.profile.sediment_fraction_classes):
        raise MultifractionSuspendedV78Error(
            "v78 admits noncohesive sand/gravel fractions only")
    hydraulic_sha = _require_sha256(
        hydraulic_execution_sha256, "hydraulic execution digest")
    placeholder = MultifractionSuspendedExecutionV78(
        role, base_execution, hydraulic_sha, SEMANTIC_ID, "")
    return MultifractionSuspendedExecutionV78(
        **{**placeholder.__dict__,
           "fingerprint": _fingerprint(_execution_payload(placeholder))})


def execution_document_v78(
    value: MultifractionSuspendedExecutionV78,
) -> dict:
    if not isinstance(value, MultifractionSuspendedExecutionV78):
        raise MultifractionSuspendedV78Error("v78 execution object differs")
    payload = _execution_payload(value)
    if value.fingerprint != _fingerprint(payload):
        raise MultifractionSuspendedV78Error(
            "v78 execution fingerprint differs")
    return {**payload, "fingerprint": value.fingerprint}


@dataclass(frozen=True)
class MultifractionSuspendedStateV78:
    execution_fingerprint: str
    lifecycle: SuspendedLifecycleStateV76
    combined_reference_mass_kg: float
    combined_boundary_ledger_kg: float
    combined_current_mass_kg: float
    combined_residual_kg: float
    state_sha256: str

    @property
    def accepted_updates(self) -> int:
        return self.lifecycle.accepted_updates

    @property
    def time_s(self) -> float:
        return self.lifecycle.time_s

    @property
    def suspended_mass_kg_m2(self) -> np.ndarray:
        return self.lifecycle.suspended_mass_kg_m2

    @property
    def bed_mass_kg_m2(self) -> np.ndarray:
        return self.lifecycle.bed_mass_kg_m2


def _combined_ledger(
    execution: MultifractionSuspendedExecutionV78,
    state: SuspendedLifecycleStateV76,
) -> tuple[float, float, float, float]:
    area = execution.base_execution.cell_area_m2[None, :]
    reference = float(np.sum(
        state.reference_water_mass_kg + state.reference_bed_mass_kg,
        dtype=np.float64))
    boundary = float(np.sum(
        state.advective_boundary_ledger_kg +
        state.diffusive_boundary_ledger_kg, dtype=np.float64))
    current = float(np.sum(
        (state.suspended_mass_kg_m2 + state.bed_mass_kg_m2) * area,
        dtype=np.float64))
    return reference, boundary, current, current-reference-boundary


def _state_token(value: MultifractionSuspendedStateV78) -> str:
    return _fingerprint({
        "execution_fingerprint": value.execution_fingerprint,
        "base_state_sha256": value.lifecycle.state_sha256,
        "combined_reference_mass_kg": value.combined_reference_mass_kg,
        "combined_boundary_ledger_kg": value.combined_boundary_ledger_kg,
        "combined_current_mass_kg": value.combined_current_mass_kg,
        "combined_residual_kg": value.combined_residual_kg,
    })


def _wrap_state(
    execution: MultifractionSuspendedExecutionV78,
    lifecycle: SuspendedLifecycleStateV76,
) -> MultifractionSuspendedStateV78:
    execution_document_v78(execution)
    validate_suspended_lifecycle_state_v76(
        lifecycle, execution.base_execution)
    if lifecycle.accepted_updates > MAX_ACCEPTED_UPDATES:
        raise MultifractionSuspendedV78Error(
            "v78 accepted update limit was exceeded")
    ledger = _combined_ledger(execution, lifecycle)
    scale = max(1.0, abs(ledger[0]) + abs(ledger[1]))
    if abs(ledger[3]) > 65536.0*np.finfo(np.float64).eps*scale:
        raise MultifractionSuspendedV78Error(
            "v78 aggregate bed-plus-water ledger did not close")
    placeholder = MultifractionSuspendedStateV78(
        execution.fingerprint, lifecycle, *ledger, "")
    return MultifractionSuspendedStateV78(
        **{**placeholder.__dict__, "state_sha256": _state_token(placeholder)})


def resolve_multifraction_suspended_state_v78(
    execution: MultifractionSuspendedExecutionV78, **state_fields,
) -> MultifractionSuspendedStateV78:
    try:
        lifecycle = resolve_suspended_lifecycle_state_v76(
            execution.base_execution, **state_fields)
    except (TypeError, ValueError) as error:
        raise MultifractionSuspendedV78Error(
            "v78 accepted lifecycle state is invalid") from error
    return _wrap_state(execution, lifecycle)


def validate_multifraction_suspended_state_v78(
    state: MultifractionSuspendedStateV78,
    execution: MultifractionSuspendedExecutionV78,
) -> None:
    if (not isinstance(state, MultifractionSuspendedStateV78) or
            state.execution_fingerprint != execution.fingerprint or
            state.state_sha256 != _state_token(state)):
        raise MultifractionSuspendedV78Error(
            "v78 state is stale, mutated, or belongs to another execution")
    expected = _combined_ledger(execution, state.lifecycle)
    actual = (
        state.combined_reference_mass_kg,
        state.combined_boundary_ledger_kg,
        state.combined_current_mass_kg,
        state.combined_residual_kg,
    )
    if actual != expected:
        raise MultifractionSuspendedV78Error(
            "v78 aggregate ledger fields differ from class state")
    validate_suspended_lifecycle_state_v76(
        state.lifecycle, execution.base_execution)


def _class_residuals(
    execution: MultifractionSuspendedExecutionV78,
    state: SuspendedLifecycleStateV76,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    area = execution.base_execution.cell_area_m2[None, :]
    water = np.sum(state.suspended_mass_kg_m2*area, axis=1)
    bed = np.sum(state.bed_mass_kg_m2*area, axis=1)
    boundary = np.sum(
        state.advective_boundary_ledger_kg+
        state.diffusive_boundary_ledger_kg, axis=1)
    water_expected = (
        state.reference_water_mass_kg+boundary+
        state.water_exchange_ledger_kg+state.wetdry_water_ledger_kg)
    bed_expected = (
        state.reference_bed_mass_kg+state.bed_exchange_ledger_kg+
        state.wetdry_bed_ledger_kg)
    return (water-water_expected, bed-bed_expected,
            water+bed-state.reference_water_mass_kg-
            state.reference_bed_mass_kg-boundary)


def _frozen(value: object) -> np.ndarray:
    result = np.ascontiguousarray(value)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class MultifractionSuspendedTrialV78:
    execution_fingerprint: str
    parent_state_sha256: str
    accepted_step_receipt_sha256: str
    candidate: MultifractionSuspendedStateV78
    boundary_concentration_kg_m3: np.ndarray
    advective_face_flux_kg_m_s: np.ndarray
    diffusive_face_mass_owner_out_kg: np.ndarray
    settling_velocity_m_s: np.ndarray
    equilibrium_concentration_kg_m3: np.ndarray
    transferred_to_suspension_kg_m2: np.ndarray
    wetdry_transfer_to_bed_kg_m2: np.ndarray
    water_ledger_residual_kg: np.ndarray
    bed_ledger_residual_kg: np.ndarray
    class_combined_ledger_residual_kg: np.ndarray
    total_combined_ledger_residual_kg: float
    observed_parallel_workers: int
    backend: str
    trial_sha256: str


def _trial_token(value: MultifractionSuspendedTrialV78) -> str:
    return _fingerprint({
        "execution_fingerprint": value.execution_fingerprint,
        "parent_state_sha256": value.parent_state_sha256,
        "accepted_step_receipt_sha256":
            value.accepted_step_receipt_sha256,
        "candidate_state_sha256": value.candidate.state_sha256,
        "operator_arrays_sha256": _fingerprint([
            _digest(array) for array in (
                value.boundary_concentration_kg_m3,
                value.advective_face_flux_kg_m_s,
                value.diffusive_face_mass_owner_out_kg,
                value.settling_velocity_m_s,
                value.equilibrium_concentration_kg_m3,
                value.transferred_to_suspension_kg_m2,
                value.wetdry_transfer_to_bed_kg_m2,
                value.water_ledger_residual_kg,
                value.bed_ledger_residual_kg,
                value.class_combined_ledger_residual_kg,
            )]),
        "total_combined_ledger_residual_kg":
            value.total_combined_ledger_residual_kg,
        "observed_parallel_workers": value.observed_parallel_workers,
        "backend": value.backend,
    })


def build_multifraction_trial_v78(
    execution: MultifractionSuspendedExecutionV78,
    parent: MultifractionSuspendedStateV78,
    accepted_step_receipt: Mapping[str, object],
    candidate_lifecycle: SuspendedLifecycleStateV76, *,
    boundary_concentration_kg_m3: object,
    advective_face_flux_kg_m_s: object,
    diffusive_face_mass_owner_out_kg: object,
    settling_velocity_m_s: object,
    equilibrium_concentration_kg_m3: object,
    transferred_to_suspension_kg_m2: object,
    wetdry_transfer_to_bed_kg_m2: object,
    observed_parallel_workers: int,
    backend: str,
) -> MultifractionSuspendedTrialV78:
    candidate = _wrap_state(execution, candidate_lifecycle)
    water, bed, combined = _class_residuals(execution, candidate_lifecycle)
    ng, nf = execution.fraction_count, execution.face_count
    nc = execution.cell_count
    shapes = (
        (boundary_concentration_kg_m3, (ng, nf), "boundary concentration"),
        (advective_face_flux_kg_m_s, (ng, nf), "advective face flux"),
        (diffusive_face_mass_owner_out_kg, (ng, nf),
         "diffusive face mass"),
        (settling_velocity_m_s, (ng, nc), "settling velocity"),
        (equilibrium_concentration_kg_m3, (ng, nc),
         "equilibrium concentration"),
        (transferred_to_suspension_kg_m2, (ng, nc),
         "exchange transfer"),
        (wetdry_transfer_to_bed_kg_m2, (ng, nc), "wet/dry transfer"),
    )
    arrays = []
    for raw, shape, name in shapes:
        value = np.asarray(raw)
        if value.shape != shape or value.dtype.kind not in "fiu" or \
                not np.isfinite(value).all():
            raise MultifractionSuspendedV78Error(
                f"v78 {name} evidence must be finite with shape {shape}")
        arrays.append(_frozen(value.astype(np.float64, copy=False)))
    residuals = [_frozen(value) for value in (water, bed, combined)]
    if (isinstance(observed_parallel_workers, bool) or
            type(observed_parallel_workers) is not int or
            observed_parallel_workers < 1):
        raise MultifractionSuspendedV78Error(
            "v78 observed worker count is invalid")
    placeholder = MultifractionSuspendedTrialV78(
        execution.fingerprint, parent.state_sha256,
        str(accepted_step_receipt["fingerprint"]), candidate,
        *arrays, *residuals, candidate.combined_residual_kg,
        observed_parallel_workers, backend, "")
    return MultifractionSuspendedTrialV78(
        **{**placeholder.__dict__, "trial_sha256": _trial_token(placeholder)})


def begin_multifraction_cpu_oracle_trial_v78(
    execution: MultifractionSuspendedExecutionV78,
    parent: MultifractionSuspendedStateV78, *,
    accepted_step_receipt: Mapping[str, object],
    cancelled: bool = False, **inputs,
) -> MultifractionSuspendedTrialV78:
    """Evaluate the accepted v76 class-general CPU path as an oracle."""
    if execution.role != ROLE_CPU_ORACLE:
        raise MultifractionSuspendedV78Error(
            "v78 CPU oracle requires the isolated_cpu_oracle role")
    validate_multifraction_suspended_state_v78(parent, execution)
    if cancelled:
        raise MultifractionSuspendedV78Error(
            "v78 update cancelled before candidate publication")
    if parent.accepted_updates >= MAX_ACCEPTED_UPDATES:
        raise MultifractionSuspendedV78Error(
            "v78 accepted update limit was reached")
    if "dt_s" not in inputs:
        raise MultifractionSuspendedV78Error("v78 timestep is absent")
    dt = _finite(inputs["dt_s"], "timestep", positive=True)
    receipt = validate_implicit_fwave_midpoint_step_receipt_v78(
        accepted_step_receipt,
        hydraulic_execution_sha256=execution.hydraulic_execution_sha256,
        parent_accepted_updates=parent.accepted_updates,
        parent_time_s=parent.time_s, dt_s=dt)
    try:
        base_trial: SuspendedLifecycleTrialV76 = \
            begin_suspended_lifecycle_trial_v76(
                execution.base_execution, parent.lifecycle, **inputs)
    except (TypeError, ValueError) as error:
        raise MultifractionSuspendedV78Error(
            "v78 CPU oracle rejected the lifecycle trial") from error
    return build_multifraction_trial_v78(
        execution, parent, receipt, base_trial.candidate,
        boundary_concentration_kg_m3=
            base_trial.boundary_concentration_kg_m3,
        advective_face_flux_kg_m_s=base_trial.advective_face_flux_kg_m_s,
        diffusive_face_mass_owner_out_kg=
            base_trial.diffusive_face_mass_owner_out_kg,
        settling_velocity_m_s=base_trial.settling_velocity_m_s,
        equilibrium_concentration_kg_m3=
            base_trial.equilibrium_concentration_kg_m3,
        transferred_to_suspension_kg_m2=
            base_trial.transferred_to_suspension_kg_m2,
        wetdry_transfer_to_bed_kg_m2=
            base_trial.wetdry_transfer_to_bed_kg_m2,
        observed_parallel_workers=1, backend=ROLE_CPU_ORACLE)


def _validate_trial(
    execution: MultifractionSuspendedExecutionV78,
    parent: MultifractionSuspendedStateV78,
    trial: MultifractionSuspendedTrialV78,
) -> None:
    validate_multifraction_suspended_state_v78(parent, execution)
    if (not isinstance(trial, MultifractionSuspendedTrialV78) or
            trial.execution_fingerprint != execution.fingerprint or
            trial.parent_state_sha256 != parent.state_sha256 or
            trial.trial_sha256 != _trial_token(trial)):
        raise MultifractionSuspendedV78Error(
            "v78 trial is stale, mutated, or belongs to another parent")
    validate_multifraction_suspended_state_v78(trial.candidate, execution)


def commit_multifraction_trial_v78(
    execution: MultifractionSuspendedExecutionV78,
    parent: MultifractionSuspendedStateV78,
    trial: MultifractionSuspendedTrialV78,
) -> MultifractionSuspendedStateV78:
    _validate_trial(execution, parent, trial)
    return trial.candidate


def rollback_multifraction_trial_v78(
    execution: MultifractionSuspendedExecutionV78,
    parent: MultifractionSuspendedStateV78,
    trial: MultifractionSuspendedTrialV78,
) -> MultifractionSuspendedStateV78:
    _validate_trial(execution, parent, trial)
    return parent


def cancel_multifraction_trial_v78(
    execution: MultifractionSuspendedExecutionV78,
    parent: MultifractionSuspendedStateV78,
    trial: MultifractionSuspendedTrialV78,
) -> MultifractionSuspendedStateV78:
    return rollback_multifraction_trial_v78(execution, parent, trial)


def multifraction_checkpoint_v78(
    execution: MultifractionSuspendedExecutionV78,
    state: MultifractionSuspendedStateV78,
) -> dict:
    validate_multifraction_suspended_state_v78(state, execution)
    body = {
        "schema": SCHEMA,
        "scope_id": CHECKPOINT_SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "execution_fingerprint": execution.fingerprint,
        "integrator": INTEGRATOR,
        "hydraulic_execution_sha256": execution.hydraulic_execution_sha256,
        "transaction_schema": "accepted_state_no_trial_arrays_v1",
        "base_v76_checkpoint": suspended_lifecycle_checkpoint_v76(
            execution.base_execution, state.lifecycle),
        "combined_ledger": {
            "reference_mass_kg": state.combined_reference_mass_kg,
            "boundary_ledger_kg": state.combined_boundary_ledger_kg,
            "current_mass_kg": state.combined_current_mass_kg,
            "residual_kg": state.combined_residual_kg,
        },
        "state_sha256": state.state_sha256,
    }
    return {**body, "fingerprint": _fingerprint(body)}


def load_multifraction_checkpoint_v78(
    execution: MultifractionSuspendedExecutionV78,
    document: Mapping[str, object],
) -> MultifractionSuspendedStateV78:
    expected = {
        "schema", "scope_id", "semantic_id", "execution_fingerprint",
        "integrator", "hydraulic_execution_sha256", "transaction_schema",
        "base_v76_checkpoint", "combined_ledger", "state_sha256",
        "fingerprint",
    }
    if not isinstance(document, Mapping) or set(document) != expected:
        raise MultifractionSuspendedV78Error(
            "v78 checkpoint inventory differs")
    body = {key: document[key] for key in document if key != "fingerprint"}
    if (document["fingerprint"] != _fingerprint(body) or
            document["schema"] != SCHEMA or
            document["scope_id"] != CHECKPOINT_SCOPE_ID or
            document["semantic_id"] != SEMANTIC_ID or
            document["execution_fingerprint"] != execution.fingerprint or
            document["integrator"] != INTEGRATOR or
            document["hydraulic_execution_sha256"] !=
            execution.hydraulic_execution_sha256 or
            document["transaction_schema"] !=
            "accepted_state_no_trial_arrays_v1"):
        raise MultifractionSuspendedV78Error(
            "v78 checkpoint identity differs")
    try:
        lifecycle = load_suspended_lifecycle_checkpoint_v76(
            execution.base_execution, document["base_v76_checkpoint"])
    except (TypeError, ValueError) as error:
        raise MultifractionSuspendedV78Error(
            "v78 base lifecycle checkpoint is invalid") from error
    state = _wrap_state(execution, lifecycle)
    ledger = document["combined_ledger"]
    expected_ledger = {
        "reference_mass_kg": state.combined_reference_mass_kg,
        "boundary_ledger_kg": state.combined_boundary_ledger_kg,
        "current_mass_kg": state.combined_current_mass_kg,
        "residual_kg": state.combined_residual_kg,
    }
    if ledger != expected_ledger or document["state_sha256"] != state.state_sha256:
        raise MultifractionSuspendedV78Error(
            "v78 checkpoint aggregate ledger or state digest differs")
    return state


def method_identity_v78() -> dict:
    return {
        "scope_id": SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "builds_additively_on": [
            "braids-q4-split-suspended-route-owner-v74-v1",
            "braids-q4-fixed-grid-suspended-lifecycle-v76-v1",
        ],
        "fraction_scope": "2..3 noncohesive sand/gravel classes",
        "integrator": INTEGRATOR,
        "hydraulic_backend": HYDRAULIC_BACKEND,
        "strict_accepted_step_receipt": True,
        "fixed_grid_only": True,
        "accepted_update_limit": MAX_ACCEPTED_UPDATES,
        "automatic_cpu_fallback": False,
        "cpu_role": ROLE_CPU_ORACLE,
        "cuda_role": ROLE_CUDA,
        "class_state": (
            "settling_exchange_advection_diffusion_water_bed_boundary_memory"),
        "aggregate_ledger": "one_all_class_bed_plus_water_boundary_ledger",
        "wet_dry_cleanup": (
            "deposit_each_class_entire_dry_water_inventory_to_same_class_bed"),
        "transaction": "immutable_commit_rollback_cancel_exact_restart_v1",
        "delft3d_source_modified": False,
        "long_run_claimed": False,
    }


FROZEN_PREDECESSOR_SHA256 = {
    "braids_core/sediment/suspended_route_live_v74.py":
        "938dffc627dc6ec1102aff13d8999f2a5410de9853026e5c7cbfcfa65c29f8dd",
    "braids_gpu/suspended_route_live_v74.py":
        "c13c9875b246dff65371810654cd5610bedd088cd7095a81bead942f2e0412cb",
    "braids_gpu/suspended_route_live_v74.cu":
        "b905edce1829fe65af1e3c2857c9d56b8e6f7c0b9940a7520a1f5889bf657a45",
    "braids_core/sediment/suspended_lifecycle_live_v76.py":
        "20ba39ba7228ea6c50e2ea310777d59516c9c157c05d71c6bf3298fed07d4ee6",
    "braids_gpu/suspended_lifecycle_live_v76.py":
        "182355c4ef1184fb64b877ca71ef5ada344bd40eb27bc6c5c0968a0541661ff8",
    "braids_gpu/law11_soursin_exchange.cu":
        "26fd25cee3bbdfdced65690e9367e34daf8fa50ed260dd425b7a3b2248c2ff6d",
    "braids_gpu/flow2dh_boundary_diffusion.cu":
        "b61c2ffe2c4a4bc411ffc4c0ce0fc07ca9edd1d86a23d622d903cb5aa15e71bf",
}


def verify_frozen_predecessors_v78(root: str | Path) -> dict[str, str]:
    root = Path(root)
    result = {}
    for relative, expected in FROZEN_PREDECESSOR_SHA256.items():
        path = root/relative
        if not path.is_file():
            raise MultifractionSuspendedV78Error(
                "v78 frozen predecessor is absent: " + relative)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise MultifractionSuspendedV78Error(
                "v78 frozen predecessor hash differs: " + relative)
        result[relative] = actual
    return result


def verify_sources_v78(root: str | Path) -> dict[str, str]:
    root = Path(root)
    paths = (
        "braids_core/sediment/multifraction_suspended_lifecycle_v78.py",
        "braids_gpu/multifraction_suspended_lifecycle_v78.py",
        "tests/test_q4_multifraction_suspended_lifecycle_v78.py",
        "tests/test_q4_multifraction_suspended_cuda_v78.py",
    )
    result = {}
    for relative in paths:
        path = root/relative
        if not path.is_file():
            raise MultifractionSuspendedV78Error(
                "v78 additive source is absent: " + relative)
        result[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


__all__ = [
    "CHECKPOINT_SCOPE_ID", "HYDRAULIC_BACKEND", "INTEGRATOR",
    "MAX_ACCEPTED_UPDATES", "MAX_FRACTIONS", "MIN_FRACTIONS",
    "MultifractionSuspendedExecutionV78", "MultifractionSuspendedStateV78",
    "MultifractionSuspendedTrialV78", "MultifractionSuspendedV78Error",
    "ROLE_CPU_ORACLE", "ROLE_CUDA", "SCHEMA", "SCOPE_ID", "SEMANTIC_ID",
    "STEP_RECEIPT_SCOPE_ID", "begin_multifraction_cpu_oracle_trial_v78",
    "build_multifraction_trial_v78", "cancel_multifraction_trial_v78",
    "commit_multifraction_trial_v78", "execution_document_v78",
    "load_multifraction_checkpoint_v78",
    "make_implicit_fwave_midpoint_step_receipt_v78", "method_identity_v78",
    "multifraction_checkpoint_v78", "resolve_multifraction_suspended_execution_v78",
    "resolve_multifraction_suspended_state_v78",
    "rollback_multifraction_trial_v78",
    "validate_implicit_fwave_midpoint_step_receipt_v78",
    "validate_multifraction_suspended_state_v78",
    "verify_frozen_predecessors_v78", "verify_sources_v78",
]
