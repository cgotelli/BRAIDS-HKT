"""Thin Q4/v94b admission for the locked FLOW exact-zero vector route.

V94b does not duplicate the v94 scalar, IopSus, Q3, or Exner lifecycle.  It
adds one execution identity for ``SusCor=false`` with
``NeglectEntrainment=true`` and delegates every transaction to the existing
single owners.  Historical v94 rows are returned byte-for-byte by the old
resolver and keep their old checkpoint identity.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
from typing import Callable, Mapping

from . import active_suspended_control_v94 as _v94
from .active_suspended_control_v94 import (
    FORMULA_CODE,
    ActiveSuspendedControlExecutionV94,
    ActiveSuspendedControlStateV94,
    ActiveSuspendedControlTrialV94,
)
from .factcr_lifecycle_v83 import FactCrExecutionV83
from .hles_producer_v90 import CurrentClosureHandoffV90, HlesProducerStateV90
from .suspended_exner_owner_v93 import (
    SUSPENDED_VECTOR_CONSTRUCTED_V93,
    SUSPENDED_VECTOR_EXACT_ZERO_V93,
    SuspendedExnerExecutionV93,
)


SCOPE_ID_V94B = _v94.SCOPE_ID + ":suscor-false-zero-vector-v94b"
SEMANTIC_ID_V94B = hashlib.sha256(SCOPE_ID_V94B.encode("ascii")).hexdigest()


class ActiveSuspendedControlV94bError(ValueError):
    """A v94b execution, state, checkpoint, or transaction is invalid."""


def _is_zero_mode(execution: ActiveSuspendedControlExecutionV94) -> bool:
    return (execution.base_execution.suspended_vector_mode ==
            SUSPENDED_VECTOR_EXACT_ZERO_V93)


def execution_document_v94b(
    execution: ActiveSuspendedControlExecutionV94,
) -> dict[str, object]:
    """Validate v94/v94b mode pairing without changing old documents."""
    document = _v94.execution_document_v94(execution)
    zero_mode = _is_zero_mode(execution)
    if execution.semantic_id == _v94.SEMANTIC_ID:
        if (zero_mode or
                execution.base_execution.suspended_vector_mode !=
                    SUSPENDED_VECTOR_CONSTRUCTED_V93 or
                (execution.base_execution.neglect_entrainment and
                 not execution.suscor)):
            raise ActiveSuspendedControlV94bError(
                "historical v94 identity cannot represent the exact-zero "
                "suspended-vector route")
    elif execution.semantic_id == SEMANTIC_ID_V94B:
        if (not zero_mode or execution.suscor or
                not execution.base_execution.neglect_entrainment):
            raise ActiveSuspendedControlV94bError(
                "v94b identity requires SusCor=false, "
                "NeglectEntrainment=true, and exact-zero vector mode")
    else:
        raise ActiveSuspendedControlV94bError(
            "active suspended-control semantic identity is not admitted")
    return document


def resolve_active_suspended_control_execution_v94b(
    base_execution: SuspendedExnerExecutionV93, *,
    factcr_execution: FactCrExecutionV83,
    live_taurat_execution: Mapping[str, object],
    iopsus_by_fraction: object = 1,
    suscor: bool = True,
    formula_code: int = FORMULA_CODE,
    initializer_water_density_kg_m3: object | None = None,
    initializer_kinematic_viscosity_m2_s: object = 1.0e-6,
) -> ActiveSuspendedControlExecutionV94:
    """Resolve old v94 rows exactly or the one additive v94b route."""
    mode = base_execution.suspended_vector_mode
    kwargs = dict(
        factcr_execution=factcr_execution,
        live_taurat_execution=live_taurat_execution,
        iopsus_by_fraction=iopsus_by_fraction,
        formula_code=formula_code,
        initializer_water_density_kg_m3=initializer_water_density_kg_m3,
        initializer_kinematic_viscosity_m2_s=
            initializer_kinematic_viscosity_m2_s)
    if mode == SUSPENDED_VECTOR_CONSTRUCTED_V93:
        # This is the historical resolver itself, so all old documents and
        # fingerprints remain exactly stable.
        execution = _v94.resolve_active_suspended_control_execution_v94(
            base_execution, suscor=suscor, **kwargs)
        execution_document_v94b(execution)
        return execution
    if mode != SUSPENDED_VECTOR_EXACT_ZERO_V93:
        raise ActiveSuspendedControlV94bError(
            "v94b suspended-vector mode is not admitted")
    if type(suscor) is not bool or suscor:
        raise ActiveSuspendedControlV94bError(
            "the exact-zero v94b route requires SusCor=false")

    # Reuse all old validation and frozen-threshold construction with the
    # admitted SusCor=true value, then replace only the additive identities.
    validated = _v94.resolve_active_suspended_control_execution_v94(
        base_execution, suscor=True, **kwargs)
    placeholder = replace(
        validated, suscor=False, semantic_id=SEMANTIC_ID_V94B,
        fingerprint="")
    execution = replace(
        placeholder,
        fingerprint=_v94._fingerprint(_v94._execution_payload(placeholder)))
    execution_document_v94b(execution)
    return execution


def resolve_initial_active_suspended_control_state_v94b(
    execution: ActiveSuspendedControlExecutionV94, **kwargs,
) -> ActiveSuspendedControlStateV94:
    execution_document_v94b(execution)
    return _v94.resolve_initial_active_suspended_control_state_v94(
        execution, **kwargs)


def validate_active_suspended_control_state_v94b(
    state: ActiveSuspendedControlStateV94,
    execution: ActiveSuspendedControlExecutionV94,
    q3_provider: object | None = None,
) -> None:
    execution_document_v94b(execution)
    _v94.validate_active_suspended_control_state_v94(
        state, execution, q3_provider)


def begin_active_suspended_control_cpu_trial_v94b(
    execution: ActiveSuspendedControlExecutionV94,
    parent: ActiveSuspendedControlStateV94,
    parent_hles: HlesProducerStateV90,
    handoff: CurrentClosureHandoffV90,
    q3_provider: object, *, v92_inputs: Mapping[str, object],
    q3_inputs: Mapping[str, object],
    cancelled: Callable[[], bool] = lambda: False,
) -> ActiveSuspendedControlTrialV94:
    execution_document_v94b(execution)
    return _v94.begin_active_suspended_control_cpu_trial_v94(
        execution, parent, parent_hles, handoff, q3_provider,
        v92_inputs=v92_inputs, q3_inputs=q3_inputs, cancelled=cancelled)


def commit_active_suspended_control_trial_v94b(
    execution: ActiveSuspendedControlExecutionV94,
    parent: ActiveSuspendedControlStateV94,
    parent_hles: HlesProducerStateV90,
    handoff: CurrentClosureHandoffV90,
    trial: ActiveSuspendedControlTrialV94,
) -> ActiveSuspendedControlStateV94:
    execution_document_v94b(execution)
    return _v94.commit_active_suspended_control_trial_v94(
        execution, parent, parent_hles, handoff, trial)


def rollback_active_suspended_control_trial_v94b(
    execution: ActiveSuspendedControlExecutionV94,
    parent: ActiveSuspendedControlStateV94,
    parent_hles: HlesProducerStateV90,
    handoff: CurrentClosureHandoffV90,
    trial: ActiveSuspendedControlTrialV94,
) -> ActiveSuspendedControlStateV94:
    execution_document_v94b(execution)
    return _v94.rollback_active_suspended_control_trial_v94(
        execution, parent, parent_hles, handoff, trial)


def cancel_active_suspended_control_trial_v94b(*args, **kwargs):
    return rollback_active_suspended_control_trial_v94b(*args, **kwargs)


def checkpoint_active_suspended_control_state_v94b(
    execution: ActiveSuspendedControlExecutionV94,
    state: ActiveSuspendedControlStateV94,
    q3_provider: object,
) -> dict[str, object]:
    execution_document_v94b(execution)
    document = _v94.checkpoint_active_suspended_control_state_v94(
        execution, state, q3_provider)
    if execution.semantic_id == _v94.SEMANTIC_ID:
        return document
    body = {key: value for key, value in document.items()
            if key != "fingerprint"}
    body["semantic_id"] = SEMANTIC_ID_V94B
    return {**body, "fingerprint": _v94._fingerprint(body)}


def load_active_suspended_control_checkpoint_v94b(
    execution: ActiveSuspendedControlExecutionV94,
    document: Mapping[str, object],
    q3_provider: object,
) -> ActiveSuspendedControlStateV94:
    execution_document_v94b(execution)
    if execution.semantic_id == _v94.SEMANTIC_ID:
        return _v94.load_active_suspended_control_checkpoint_v94(
            execution, document, q3_provider)
    if not isinstance(document, Mapping) or "fingerprint" not in document:
        raise ActiveSuspendedControlV94bError(
            "v94b checkpoint inventory differs")
    body = {key: value for key, value in document.items()
            if key != "fingerprint"}
    if (document["fingerprint"] != _v94._fingerprint(body) or
            body.get("semantic_id") != SEMANTIC_ID_V94B or
            body.get("execution_fingerprint") != execution.fingerprint):
        raise ActiveSuspendedControlV94bError(
            "v94b checkpoint identity differs")
    translated = dict(body)
    translated["semantic_id"] = _v94.SEMANTIC_ID
    translated["fingerprint"] = _v94._fingerprint(translated)
    return _v94.load_active_suspended_control_checkpoint_v94(
        execution, translated, q3_provider)


def method_identity_v94b() -> dict[str, object]:
    return {
        "scope_id": SCOPE_ID_V94B,
        "semantic_id": SEMANTIC_ID_V94B,
        "historical_v94_rows": "delegated_byte_identical",
        "added_route": {
            "suscor": False,
            "neglect_entrainment": True,
            "suspended_vector_mode": SUSPENDED_VECTOR_EXACT_ZERO_V93,
            "raw_v92_scalar_lifecycle": "unchanged",
            "selected_suspended_bed_delta": "exact_zero",
            "topology_divergence_kernel_count": 0,
            "zero_vector_materialization_count": 1,
            "q3_bed_delta": "already_timestep_integrated_applied_once",
        },
        "maximum_accepted_updates": _v94.MAX_ACCEPTED_UPDATES,
        "morphological_factor": 1.0,
        "automatic_cpu_fallback": False,
        "product_admission": False,
        "delft3d_source_modified": False,
    }


__all__ = [
    "ActiveSuspendedControlV94bError", "SCOPE_ID_V94B",
    "SEMANTIC_ID_V94B", "begin_active_suspended_control_cpu_trial_v94b",
    "cancel_active_suspended_control_trial_v94b",
    "checkpoint_active_suspended_control_state_v94b",
    "commit_active_suspended_control_trial_v94b", "execution_document_v94b",
    "load_active_suspended_control_checkpoint_v94b", "method_identity_v94b",
    "resolve_active_suspended_control_execution_v94b",
    "resolve_initial_active_suspended_control_state_v94b",
    "rollback_active_suspended_control_trial_v94b",
    "validate_active_suspended_control_state_v94b",
]
