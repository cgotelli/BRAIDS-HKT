"""Physical-open coarse leaf composed with one real 2:1 AMR interface.

This additive owner joins two already qualified v98 transactions without
turning the internal coarse/fine interface into a physical boundary.  The
coarse provisional update is produced by the cut-cell physical-boundary
owner; the joined-leaf owner then replaces only the authenticated internal
interface integrals and publishes the coarse Q3 provider once.  Accepted
physical-boundary scalar memory is carried to the corrected numerical parent
for the next macro step.

The scope remains one joined leaf and at most two macro updates.  It is not a
general regrid or multi-patch boundary admission.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Callable

import numpy as np

from .accepted_boundary_scalar_state_cpu_v96 import (
    accepted_boundary_scalar_checkpoint_v96,
    load_accepted_boundary_scalar_checkpoint_v96,
)
from .coupled_amr_corrected_images_native_v98 import (
    NativeAMRCorrectedImagesRuntimeV98,
)
from .coupled_amr_fine_substep_cpu_v98 import (
    CoupledAMRFineSubstepStateCPUV98,
)
from .coupled_amr_joined_leaf_cpu_v98 import (
    CoupledAMRJoinedLeafStateCPUV98,
    CoupledAMRJoinedLeafTrialCPUV98,
    checkpoint_coupled_amr_joined_leaf_cpu_v98,
    commit_coupled_amr_joined_leaf_cpu_v98,
    load_coupled_amr_joined_leaf_checkpoint_v98,
    prepare_coupled_amr_joined_leaf_cpu_v98,
    validate_coupled_amr_joined_leaf_state_v98,
    validate_coupled_amr_joined_leaf_trial_v98,
)
from .coupled_amr_q3_trace_cpu_v98 import CoupledAMRQ3TraceCPUV98
from .coupled_amr_register_cpu_v98 import (
    CoupledAMRCoarseCorrectionCPUV98,
    CoupledAMRInterfaceMapCPUV98,
)
from .coupled_amr_scalar_trace_cpu_v98 import CoupledAMRScalarTraceCPUV98
from .coupled_amr_trace_cpu_v98 import CoupledAMRInternalTraceCPUV98
from .coupled_amr_turbulence_trace_cpu_v98 import (
    CoupledAMRTurbulenceTraceCPUV98,
)
from .coupled_wetdry_open_owner_cpu_v98 import (
    CoupledWetDryOpenCPUExecutionV98,
    CoupledWetDryOpenCPUStateV98,
    CoupledWetDryOpenCPUTrialV98,
    _hash as _open_hash,
    _validate_trial as _validate_open_trial,
    _v92_execution,
    _wrap_state as _wrap_open_state,
    execution_document_coupled_wetdry_open_v98,
    validate_coupled_wetdry_open_cpu_state_v98,
)
from .coupled_wetdry_owner_cpu_v97 import (
    CoupledWetDryCPUExecutionV97,
    CoupledWetDryCPUStateV97,
)
from .hles_native_cpu_v97 import NativeHlesRuntimeV97
from .q3_wetdry_adapter_cpu_v97 import (
    CPUQ3V83ActiveWetDryExnerAdapterV97,
)


SCHEMA = 1
SCOPE_ID = "braids.coupled_amr.joined_open_leaf.cpu.v98"
CHECKPOINT_SCOPE_ID = SCOPE_ID + ":checkpoint"
SEMANTIC_ID = (
    "one_physical_cutcell_boundary_owner_plus_one_internal_2to1_register")
BACKEND = "native_cpu_physical_open_plus_private_fine_single_public_coarse"
_SHA = re.compile(r"[0-9a-f]{64}\Z")


class CoupledAMRJoinedOpenLeafCPUV98Error(ValueError):
    """The physical-open joined-leaf transaction differs."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("ascii")


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


@dataclass(frozen=True)
class CoupledAMRJoinedOpenLeafCountsCPUV98:
    physical_cutcell_boundary_owner_count: int = 1
    physical_boundary_scalar_memory_owner_count: int = 1
    internal_interface_register_owner_count: int = 1
    internal_interface_physical_boundary_surrogate_count: int = 0
    public_coarse_provider_publication_count: int = 1
    duplicate_owner_count: int = 0


@dataclass(frozen=True)
class CoupledAMRJoinedOpenLeafStateCPUV98:
    open_execution_fingerprint: str
    fine_execution_fingerprint: str
    interface_fingerprint: str
    joined_state: CoupledAMRJoinedLeafStateCPUV98
    coarse_open_numerical_parent: CoupledWetDryOpenCPUStateV98
    last_physical_trace_receipt_sha256: str
    last_open_hydraulic_trial_sha256: str
    last_physical_residual_evaluation_count: int
    last_physical_flux_consumption_count: int
    cumulative_physical_boundary_water_outflow_m3: float
    previous_state_sha256: str | None
    counts: CoupledAMRJoinedOpenLeafCountsCPUV98
    semantic_id: str
    backend: str
    state_sha256: str

    @property
    def accepted_macro_updates(self) -> int:
        return self.joined_state.accepted_macro_updates

    @property
    def time_s(self) -> float:
        return self.joined_state.time_s


def _state_payload(value: CoupledAMRJoinedOpenLeafStateCPUV98) -> dict:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID + ":state",
        "open_execution": value.open_execution_fingerprint,
        "fine_execution": value.fine_execution_fingerprint,
        "interface": value.interface_fingerprint,
        "joined_state": value.joined_state.state_sha256,
        "coarse_open_numerical_parent":
            value.coarse_open_numerical_parent.state_sha256,
        "last_physical_trace": value.last_physical_trace_receipt_sha256,
        "last_open_hydraulic_trial":
            value.last_open_hydraulic_trial_sha256,
        "last_physical_residual_evaluation_count":
            value.last_physical_residual_evaluation_count,
        "last_physical_flux_consumption_count":
            value.last_physical_flux_consumption_count,
        "cumulative_physical_boundary_water_outflow_m3":
            value.cumulative_physical_boundary_water_outflow_m3,
        "previous": value.previous_state_sha256,
        "counts": value.counts.__dict__,
        "semantic_id": value.semantic_id,
        "backend": value.backend,
        "scope_limit": "one_joined_leaf_one_physical_boundary_family",
        "internal_interface_physical_boundary_surrogate_count": 0,
    }


@dataclass(frozen=True)
class CoupledAMRJoinedOpenLeafTrialCPUV98:
    open_parent_state_sha256: str
    coarse_open_trial: CoupledWetDryOpenCPUTrialV98
    joined_trial: CoupledAMRJoinedLeafTrialCPUV98
    candidate: CoupledAMRJoinedOpenLeafStateCPUV98
    trial_sha256: str


def _trial_payload(value: CoupledAMRJoinedOpenLeafTrialCPUV98) -> dict:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID + ":trial",
        "open_parent": value.open_parent_state_sha256,
        "coarse_open_trial": value.coarse_open_trial.trial_sha256,
        "joined_trial": value.joined_trial.trial_sha256,
        "candidate": value.candidate.state_sha256,
        "publication_count": 0,
    }


def _physical_water_outflow_m3(
    execution: CoupledWetDryOpenCPUExecutionV98,
    trial: CoupledWetDryOpenCPUTrialV98,
) -> float:
    hydraulic = trial.open_hydraulic_trial.wetdry_trial
    mask = np.asarray(execution.boundary_execution.open_mask, dtype=bool)
    grid = (execution.base_execution.active_execution.base_execution.
            base_execution.base_execution.base_execution.base_execution)
    return float(hydraulic.dt_s * np.sum(
        hydraulic.accepted_limited_face_water_flux_m2_s[mask] *
        grid.face_length_m[mask]))


def _provider_snapshot(
    trial: CoupledWetDryOpenCPUTrialV98,
) -> dict[str, object]:
    try:
        return (trial.base_trial.active_trial.base_v93_trial.
                q3_parent_checkpoint)
    except (AttributeError, TypeError) as error:
        raise CoupledAMRJoinedOpenLeafCPUV98Error(
            "v98 joined-open trial has no authenticated provider parent") \
            from error


def _open_chain(
    parent: CoupledWetDryOpenCPUStateV98,
    open_trial: CoupledWetDryOpenCPUTrialV98,
    joined: CoupledAMRJoinedLeafStateCPUV98,
) -> str:
    return _open_hash({
        "scope_id": SCOPE_ID,
        "previous_open_chain": parent.accepted_chain_sha256,
        "physical_open_trial": open_trial.trial_sha256,
        "physical_trace":
            open_trial.open_hydraulic_trial.accepted_boundary_trace.
            receipt_sha256,
        "boundary_scalar": open_trial.boundary_scalar_candidate.state_sha256,
        "joined_state": joined.state_sha256,
    })


def _wrap_candidate(
    open_execution: CoupledWetDryOpenCPUExecutionV98,
    open_parent: CoupledWetDryOpenCPUStateV98,
    open_trial: CoupledWetDryOpenCPUTrialV98,
    joined: CoupledAMRJoinedLeafStateCPUV98,
    previous: CoupledAMRJoinedOpenLeafStateCPUV98 | None,
) -> CoupledAMRJoinedOpenLeafStateCPUV98:
    audit = open_trial.open_hydraulic_trial.residual_audit
    trace = open_trial.open_hydraulic_trial.accepted_boundary_trace
    open_numerical = _wrap_open_state(
        open_execution,
        base_state=joined.coarse_numerical_parent,
        accepted_boundary_scalar=open_trial.boundary_scalar_candidate,
        accepted_chain_sha256=_open_chain(
            open_parent, open_trial, joined),
        q3_provider=None)
    physical = _physical_water_outflow_m3(open_execution, open_trial)
    cumulative = physical if previous is None else (
        previous.cumulative_physical_boundary_water_outflow_m3 + physical)
    placeholder = CoupledAMRJoinedOpenLeafStateCPUV98(
        open_execution.fingerprint,
        joined.fine_execution_fingerprint,
        joined.interface_fingerprint,
        joined,
        open_numerical,
        trace.receipt_sha256,
        open_trial.open_hydraulic_trial.trial_sha256,
        audit.evaluation_count,
        audit.physical_flux_consumption_count,
        cumulative,
        None if previous is None else previous.state_sha256,
        CoupledAMRJoinedOpenLeafCountsCPUV98(),
        SEMANTIC_ID,
        BACKEND,
        "")
    return CoupledAMRJoinedOpenLeafStateCPUV98(
        **{**placeholder.__dict__,
           "state_sha256": _hash(_state_payload(placeholder))})


def validate_coupled_amr_joined_open_leaf_state_v98(
    value: CoupledAMRJoinedOpenLeafStateCPUV98,
    open_execution: CoupledWetDryOpenCPUExecutionV98,
    fine_execution: CoupledWetDryCPUExecutionV97,
    interface: CoupledAMRInterfaceMapCPUV98, *,
    coarse_provider: CPUQ3V83ActiveWetDryExnerAdapterV97 | None,
    fine_provider: CPUQ3V83ActiveWetDryExnerAdapterV97 | None,
) -> None:
    execution_document_coupled_wetdry_open_v98(open_execution)
    validate_coupled_amr_joined_leaf_state_v98(
        value.joined_state, open_execution.base_execution,
        fine_execution, interface, coarse_provider=coarse_provider,
        fine_provider=fine_provider)
    validate_coupled_wetdry_open_cpu_state_v98(
        value.coarse_open_numerical_parent, open_execution,
        coarse_provider)
    if (type(value) is not CoupledAMRJoinedOpenLeafStateCPUV98 or
            value.open_execution_fingerprint != open_execution.fingerprint or
            value.fine_execution_fingerprint != fine_execution.fingerprint or
            value.interface_fingerprint != interface.fingerprint or
            value.coarse_open_numerical_parent.base_state is not
                value.joined_state.coarse_numerical_parent or
            value.coarse_open_numerical_parent.accepted_updates !=
                value.joined_state.accepted_macro_updates or
            value.coarse_open_numerical_parent.time_s !=
                value.joined_state.time_s or
            _SHA.fullmatch(value.last_physical_trace_receipt_sha256) is None or
            _SHA.fullmatch(value.last_open_hydraulic_trial_sha256) is None or
            value.last_physical_residual_evaluation_count <= 0 or
            value.last_physical_flux_consumption_count <= 0 or
            not math.isfinite(
                value.cumulative_physical_boundary_water_outflow_m3) or
            value.counts != CoupledAMRJoinedOpenLeafCountsCPUV98() or
            value.semantic_id != SEMANTIC_ID or value.backend != BACKEND or
            value.state_sha256 != _hash(_state_payload(value))):
        raise CoupledAMRJoinedOpenLeafCPUV98Error(
            "v98 joined-open state identity or owner counts differ")


def validate_coupled_amr_joined_open_leaf_trial_v98(
    value: CoupledAMRJoinedOpenLeafTrialCPUV98,
    open_execution: CoupledWetDryOpenCPUExecutionV98,
    open_parent: CoupledWetDryOpenCPUStateV98,
    coarse_provider: CPUQ3V83ActiveWetDryExnerAdapterV97,
    fine_execution: CoupledWetDryCPUExecutionV97,
    fine_parent: CoupledWetDryCPUStateV97,
    fine_source_provider: CPUQ3V83ActiveWetDryExnerAdapterV97,
    fine_step1: CoupledAMRFineSubstepStateCPUV98,
    fine_provider1: CPUQ3V83ActiveWetDryExnerAdapterV97,
    fine_step2: CoupledAMRFineSubstepStateCPUV98,
    fine_provider2: CPUQ3V83ActiveWetDryExnerAdapterV97,
    interface: CoupledAMRInterfaceMapCPUV98, *,
    internal_trace: CoupledAMRInternalTraceCPUV98,
    scalar_trace: CoupledAMRScalarTraceCPUV98,
    q3_trace: CoupledAMRQ3TraceCPUV98,
    turbulence_trace: CoupledAMRTurbulenceTraceCPUV98,
    previous: CoupledAMRJoinedOpenLeafStateCPUV98 | None,
) -> None:
    if type(value) is not CoupledAMRJoinedOpenLeafTrialCPUV98:
        raise CoupledAMRJoinedOpenLeafCPUV98Error(
            "v98 joined-open trial type differs")
    _validate_open_trial(
        open_execution, open_parent, coarse_provider,
        value.coarse_open_trial)
    validate_coupled_amr_joined_leaf_trial_v98(
        value.joined_trial, open_execution.base_execution,
        open_parent.base_state, coarse_provider,
        value.coarse_open_trial.base_trial,
        fine_execution, fine_parent, fine_source_provider,
        fine_step1, fine_provider1, fine_step2, fine_provider2,
        interface, internal_trace=internal_trace,
        scalar_trace=scalar_trace, q3_trace=q3_trace,
        turbulence_trace=turbulence_trace,
        previous=None if previous is None else previous.joined_state)
    audit = value.coarse_open_trial.open_hydraulic_trial.residual_audit
    trace = (
        value.coarse_open_trial.open_hydraulic_trial.
        accepted_boundary_trace)
    physical = np.asarray(
        open_execution.boundary_execution.open_mask, dtype=bool)
    expected = _wrap_candidate(
        open_execution, open_parent, value.coarse_open_trial,
        value.joined_trial.candidate, previous)
    if (not np.any(physical) or physical[interface.coarse_face] or
            trace.open_face_count != int(np.count_nonzero(physical)) or
            audit.boundary_trace_owner_call_count !=
                audit.evaluation_count or
            audit.physical_flux_consumption_count !=
                trace.open_face_count * audit.evaluation_count or
            value.coarse_open_trial.open_hydraulic_trial.
                physical_boundary_surrogate_count != 0 or
            value.joined_trial.candidate.counts.
                physical_boundary_surrogate_count != 0 or
            value.open_parent_state_sha256 != open_parent.state_sha256 or
            value.joined_trial.coarse_provisional_trial_sha256 !=
                value.coarse_open_trial.base_trial.trial_sha256 or
            value.candidate.state_sha256 != expected.state_sha256 or
            value.trial_sha256 != _hash(_trial_payload(value))):
        raise CoupledAMRJoinedOpenLeafCPUV98Error(
            "v98 joined-open physical/internal ownership differs")


def prepare_coupled_amr_joined_open_leaf_cpu_v98(
    open_execution: CoupledWetDryOpenCPUExecutionV98,
    open_parent: CoupledWetDryOpenCPUStateV98,
    coarse_provider: CPUQ3V83ActiveWetDryExnerAdapterV97,
    coarse_open_trial: CoupledWetDryOpenCPUTrialV98,
    fine_execution: CoupledWetDryCPUExecutionV97,
    fine_parent: CoupledWetDryCPUStateV97,
    fine_source_provider: CPUQ3V83ActiveWetDryExnerAdapterV97,
    fine_step1: CoupledAMRFineSubstepStateCPUV98,
    fine_provider1: CPUQ3V83ActiveWetDryExnerAdapterV97,
    fine_step2: CoupledAMRFineSubstepStateCPUV98,
    fine_provider2: CPUQ3V83ActiveWetDryExnerAdapterV97,
    interface: CoupledAMRInterfaceMapCPUV98, *,
    internal_trace: CoupledAMRInternalTraceCPUV98,
    scalar_trace: CoupledAMRScalarTraceCPUV98,
    q3_trace: CoupledAMRQ3TraceCPUV98,
    turbulence_trace: CoupledAMRTurbulenceTraceCPUV98,
    hles_runtime: NativeHlesRuntimeV97,
    native_corrected_images_runtime: NativeAMRCorrectedImagesRuntimeV98,
    closure_chezy_mhalf_s: object,
    previous: CoupledAMRJoinedOpenLeafStateCPUV98 | None = None,
    cancelled: bool | Callable[[], bool] = False,
) -> CoupledAMRJoinedOpenLeafTrialCPUV98:
    """Prepare physical and internal-interface state without publication."""
    snapshot = _provider_snapshot(coarse_open_trial)
    try:
        _validate_open_trial(
            open_execution, open_parent, coarse_provider,
            coarse_open_trial)
        if callable(cancelled) and bool(cancelled()):
            raise InterruptedError(
                "v98 joined-open cancelled before internal register")
        if type(cancelled) is bool and cancelled:
            raise InterruptedError(
                "v98 joined-open cancelled before internal register")
        joined_trial = prepare_coupled_amr_joined_leaf_cpu_v98(
            open_execution.base_execution, open_parent.base_state,
            coarse_provider, coarse_open_trial.base_trial,
            fine_execution, fine_parent, fine_source_provider,
            fine_step1, fine_provider1, fine_step2, fine_provider2,
            interface, internal_trace=internal_trace,
            scalar_trace=scalar_trace, q3_trace=q3_trace,
            turbulence_trace=turbulence_trace,
            hles_runtime=hles_runtime,
            native_corrected_images_runtime=native_corrected_images_runtime,
            closure_chezy_mhalf_s=closure_chezy_mhalf_s,
            previous=None if previous is None else previous.joined_state,
            cancelled=cancelled)
        candidate = _wrap_candidate(
            open_execution, open_parent, coarse_open_trial,
            joined_trial.candidate, previous)
        placeholder = CoupledAMRJoinedOpenLeafTrialCPUV98(
            open_parent.state_sha256, coarse_open_trial, joined_trial,
            candidate, "")
        result = CoupledAMRJoinedOpenLeafTrialCPUV98(
            **{**placeholder.__dict__,
               "trial_sha256": _hash(_trial_payload(placeholder))})
        validate_coupled_amr_joined_open_leaf_trial_v98(
            result, open_execution, open_parent, coarse_provider,
            fine_execution, fine_parent, fine_source_provider,
            fine_step1, fine_provider1, fine_step2, fine_provider2,
            interface, internal_trace=internal_trace,
            scalar_trace=scalar_trace, q3_trace=q3_trace,
            turbulence_trace=turbulence_trace, previous=previous)
        return result
    except Exception as original_error:
        try:
            coarse_provider.restore_checkpoint(snapshot)
        except Exception as cleanup_error:  # pragma: no cover
            raise RuntimeError(
                "v98 joined-open prepare and provider restore both failed") \
                from cleanup_error
        raise original_error


def commit_coupled_amr_joined_open_leaf_cpu_v98(
    trial: CoupledAMRJoinedOpenLeafTrialCPUV98,
    open_execution: CoupledWetDryOpenCPUExecutionV98,
    open_parent: CoupledWetDryOpenCPUStateV98,
    coarse_provider: CPUQ3V83ActiveWetDryExnerAdapterV97,
    fine_execution: CoupledWetDryCPUExecutionV97,
    interface: CoupledAMRInterfaceMapCPUV98,
) -> CoupledAMRJoinedOpenLeafStateCPUV98:
    """Publish the corrected coarse provider exactly once."""
    snapshot = _provider_snapshot(trial.coarse_open_trial)
    try:
        accepted = commit_coupled_amr_joined_leaf_cpu_v98(
            trial.joined_trial, open_execution.base_execution,
            open_parent.base_state, coarse_provider,
            trial.coarse_open_trial.base_trial,
            fine_execution, interface)
        if (accepted is not trial.joined_trial.candidate or
                trial.candidate.joined_state is not accepted or
                coarse_provider.state.state_sha256 !=
                    accepted.coarse_provider_state_sha256):
            raise CoupledAMRJoinedOpenLeafCPUV98Error(
                "v98 joined-open commit substituted the candidate")
        return trial.candidate
    except Exception as original_error:
        try:
            coarse_provider.restore_checkpoint(snapshot)
        except Exception as cleanup_error:  # pragma: no cover
            raise RuntimeError(
                "v98 joined-open commit and provider restore both failed") \
                from cleanup_error
        raise original_error


def rollback_coupled_amr_joined_open_leaf_cpu_v98(
    trial: CoupledAMRJoinedOpenLeafTrialCPUV98,
    open_parent: CoupledWetDryOpenCPUStateV98,
    coarse_provider: CPUQ3V83ActiveWetDryExnerAdapterV97,
) -> CoupledWetDryOpenCPUStateV98:
    """Discard the private register and restore the exact public parent."""
    coarse_provider.restore_checkpoint(_provider_snapshot(
        trial.coarse_open_trial))
    return open_parent


cancel_coupled_amr_joined_open_leaf_cpu_v98 = \
    rollback_coupled_amr_joined_open_leaf_cpu_v98


def checkpoint_coupled_amr_joined_open_leaf_cpu_v98(
    state: CoupledAMRJoinedOpenLeafStateCPUV98,
    open_execution: CoupledWetDryOpenCPUExecutionV98,
    fine_execution: CoupledWetDryCPUExecutionV97,
    coarse_provider: CPUQ3V83ActiveWetDryExnerAdapterV97,
    fine_provider: CPUQ3V83ActiveWetDryExnerAdapterV97,
    interface: CoupledAMRInterfaceMapCPUV98,
    correction: CoupledAMRCoarseCorrectionCPUV98,
) -> dict[str, object]:
    validate_coupled_amr_joined_open_leaf_state_v98(
        state, open_execution, fine_execution, interface,
        coarse_provider=coarse_provider, fine_provider=fine_provider)
    body = {
        "schema": SCHEMA,
        "scope_id": CHECKPOINT_SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "backend": BACKEND,
        "open_execution_fingerprint": open_execution.fingerprint,
        "fine_execution_fingerprint": fine_execution.fingerprint,
        "interface_fingerprint": interface.fingerprint,
        "joined_checkpoint": checkpoint_coupled_amr_joined_leaf_cpu_v98(
            state.joined_state, open_execution.base_execution,
            fine_execution, coarse_provider, fine_provider,
            interface, correction),
        "boundary_checkpoint": accepted_boundary_scalar_checkpoint_v96(
            state.coarse_open_numerical_parent.accepted_boundary_scalar,
            _v92_execution(open_execution.active_execution),
            open_execution.boundary_execution),
        "open_accepted_chain_sha256":
            state.coarse_open_numerical_parent.accepted_chain_sha256,
        "last_physical_trace_receipt_sha256":
            state.last_physical_trace_receipt_sha256,
        "last_open_hydraulic_trial_sha256":
            state.last_open_hydraulic_trial_sha256,
        "last_physical_residual_evaluation_count":
            state.last_physical_residual_evaluation_count,
        "last_physical_flux_consumption_count":
            state.last_physical_flux_consumption_count,
        "cumulative_physical_boundary_water_outflow_m3":
            state.cumulative_physical_boundary_water_outflow_m3,
        "previous_state_sha256": state.previous_state_sha256,
        "counts": state.counts.__dict__,
        "state_sha256": state.state_sha256,
    }
    return {**body, "fingerprint": _hash(body)}


def load_coupled_amr_joined_open_leaf_checkpoint_v98(
    document: object,
    open_execution: CoupledWetDryOpenCPUExecutionV98,
    fine_execution: CoupledWetDryCPUExecutionV97,
    coarse_provider: CPUQ3V83ActiveWetDryExnerAdapterV97,
    fine_provider: CPUQ3V83ActiveWetDryExnerAdapterV97,
    interface: CoupledAMRInterfaceMapCPUV98,
    correction: CoupledAMRCoarseCorrectionCPUV98, *,
    native_corrected_images_runtime: NativeAMRCorrectedImagesRuntimeV98,
) -> CoupledAMRJoinedOpenLeafStateCPUV98:
    keys = {
        "schema", "scope_id", "semantic_id", "backend",
        "open_execution_fingerprint", "fine_execution_fingerprint",
        "interface_fingerprint", "joined_checkpoint",
        "boundary_checkpoint", "open_accepted_chain_sha256",
        "last_physical_trace_receipt_sha256",
        "last_open_hydraulic_trial_sha256",
        "last_physical_residual_evaluation_count",
        "last_physical_flux_consumption_count",
        "cumulative_physical_boundary_water_outflow_m3",
        "previous_state_sha256", "counts", "state_sha256", "fingerprint",
    }
    if not isinstance(document, dict) or set(document) != keys:
        raise CoupledAMRJoinedOpenLeafCPUV98Error(
            "v98 joined-open checkpoint inventory differs")
    body = {key: value for key, value in document.items()
            if key != "fingerprint"}
    if (document["fingerprint"] != _hash(body) or
            document["schema"] != SCHEMA or
            document["scope_id"] != CHECKPOINT_SCOPE_ID or
            document["semantic_id"] != SEMANTIC_ID or
            document["backend"] != BACKEND or
            document["open_execution_fingerprint"] !=
                open_execution.fingerprint or
            document["fine_execution_fingerprint"] !=
                fine_execution.fingerprint or
            document["interface_fingerprint"] != interface.fingerprint or
            document["counts"] !=
                CoupledAMRJoinedOpenLeafCountsCPUV98().__dict__):
        raise CoupledAMRJoinedOpenLeafCPUV98Error(
            "v98 joined-open checkpoint identity differs")
    coarse_before = coarse_provider.checkpoint()
    fine_before = fine_provider.checkpoint()
    try:
        joined = load_coupled_amr_joined_leaf_checkpoint_v98(
            document["joined_checkpoint"],
            open_execution.base_execution, fine_execution,
            coarse_provider, fine_provider, interface, correction,
            native_corrected_images_runtime=native_corrected_images_runtime)
        boundary = load_accepted_boundary_scalar_checkpoint_v96(
            document["boundary_checkpoint"],
            _v92_execution(open_execution.active_execution),
            open_execution.boundary_execution)
        open_parent = _wrap_open_state(
            open_execution, base_state=joined.coarse_numerical_parent,
            accepted_boundary_scalar=boundary,
            accepted_chain_sha256=
                str(document["open_accepted_chain_sha256"]),
            q3_provider=coarse_provider)
        placeholder = CoupledAMRJoinedOpenLeafStateCPUV98(
            open_execution.fingerprint, fine_execution.fingerprint,
            interface.fingerprint, joined, open_parent,
            str(document["last_physical_trace_receipt_sha256"]),
            str(document["last_open_hydraulic_trial_sha256"]),
            int(document["last_physical_residual_evaluation_count"]),
            int(document["last_physical_flux_consumption_count"]),
            float(document["cumulative_physical_boundary_water_outflow_m3"]),
            document["previous_state_sha256"],
            CoupledAMRJoinedOpenLeafCountsCPUV98(), SEMANTIC_ID, BACKEND,
            "")
        result = CoupledAMRJoinedOpenLeafStateCPUV98(
            **{**placeholder.__dict__,
               "state_sha256": _hash(_state_payload(placeholder))})
        validate_coupled_amr_joined_open_leaf_state_v98(
            result, open_execution, fine_execution, interface,
            coarse_provider=coarse_provider, fine_provider=fine_provider)
        if result.state_sha256 != document["state_sha256"]:
            raise CoupledAMRJoinedOpenLeafCPUV98Error(
                "v98 joined-open checkpoint state differs")
        return result
    except Exception as original_error:
        cleanup_error = None
        for provider, snapshot in (
                (fine_provider, fine_before),
                (coarse_provider, coarse_before)):
            try:
                provider.restore_checkpoint(snapshot)
            except Exception as error:  # pragma: no cover
                cleanup_error = error
        if cleanup_error is not None:
            raise RuntimeError(
                "v98 joined-open load and provider restore both failed") \
                from cleanup_error
        raise original_error


__all__ = [
    "BACKEND", "CHECKPOINT_SCOPE_ID", "SCHEMA", "SEMANTIC_ID", "SCOPE_ID",
    "CoupledAMRJoinedOpenLeafCPUV98Error",
    "CoupledAMRJoinedOpenLeafCountsCPUV98",
    "CoupledAMRJoinedOpenLeafStateCPUV98",
    "CoupledAMRJoinedOpenLeafTrialCPUV98",
    "cancel_coupled_amr_joined_open_leaf_cpu_v98",
    "checkpoint_coupled_amr_joined_open_leaf_cpu_v98",
    "commit_coupled_amr_joined_open_leaf_cpu_v98",
    "load_coupled_amr_joined_open_leaf_checkpoint_v98",
    "prepare_coupled_amr_joined_open_leaf_cpu_v98",
    "rollback_coupled_amr_joined_open_leaf_cpu_v98",
    "validate_coupled_amr_joined_open_leaf_state_v98",
    "validate_coupled_amr_joined_open_leaf_trial_v98",
]
