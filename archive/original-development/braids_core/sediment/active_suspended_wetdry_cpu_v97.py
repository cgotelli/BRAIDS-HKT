"""Compose one accepted v97 wet/dry solve with the v94b/Q3 owner.

The hydraulic active-set, draining limiter, density projection, and masked
v91 momentum source are supplied by one authenticated v97 trial.  This
coordinator runs the combined scalar lifecycle and the authoritative v93
Exner transaction once; it never reconstructs a shadow hydraulic state.
"""
from __future__ import annotations

from typing import Callable, Mapping

import numpy as np

from .recursive_surface_feedback_v114 import provider_surface_fraction_v114

from . import active_suspended_control_v94 as _v94
from .active_suspended_control_v94 import (
    ActiveSuspendedControlExecutionV94,
    ActiveSuspendedControlOwnerCountsV94,
    ActiveSuspendedControlStateV94,
    ActiveSuspendedControlTrialV94,
)
from .active_suspended_control_v94b import (
    execution_document_v94b,
    validate_active_suspended_control_state_v94b,
)
from .hles_producer_v90 import CurrentClosureHandoffV90, HlesProducerStateV90
from .factcr_native_cpu_v97 import NativeFactCrEvidenceV97
from .q3_wetdry_adapter_cpu_v97 import (
    CPUQ3V83ActiveWetDryExnerAdapterV97,
)
from .scalar_diffusion_native_cpu_v96 import NativeScalarDiffusionRuntimeV96
from .scalar_fct_native_cpu_v96 import NativeScalarFCTRuntimeV96
from .transported_fluid_native_scalar_cpu_v96 import (
    NativeCombinedScalarEvidenceV96,
)
from .suspended_exner_owner_v93 import (
    _build_trial_from_live_provider,
    select_suspended_bed_cpu_v93,
)
from .transported_fluid_settling_v92 import (
    SETTLING_DIAMETER_ACTIVE_IOPSUS_V94,
    make_settling_diameter_receipt_v92,
)
from .transported_fluid_wetdry_cpu_v97 import (
    WetDryHydraulicEvidenceV97,
    begin_transported_fluid_wetdry_cpu_trial_v97,
)


SCOPE_ID = "braids-q4-active-suspended-wetdry-cpu-v97-v1"


def begin_active_suspended_wetdry_cpu_trial_v97(
    execution: ActiveSuspendedControlExecutionV94,
    parent: ActiveSuspendedControlStateV94,
    parent_hles: HlesProducerStateV90,
    handoff: CurrentClosureHandoffV90,
    q3_provider: CPUQ3V83ActiveWetDryExnerAdapterV97, *,
    hydraulic_evidence: WetDryHydraulicEvidenceV97,
    reconstructed_left_fluid: object,
    reconstructed_right_fluid: object,
    prescribed_boundary_fluid: object,
    v92_scalar_inputs: Mapping[str, object],
    q3_inputs: Mapping[str, object],
    scalar_fct_runtime: NativeScalarFCTRuntimeV96,
    scalar_diffusion_runtime: NativeScalarDiffusionRuntimeV96,
    settling_exchange_runtime=None,
    scalar_boundary_runtime=None,
    native_scalar_threads: int = 4,
    return_native_scalar_evidence: bool = False,
    cancelled: Callable[[], bool] = lambda: False,
) -> (ActiveSuspendedControlTrialV94 |
      tuple[ActiveSuspendedControlTrialV94,
            NativeCombinedScalarEvidenceV96,
            NativeFactCrEvidenceV97]):
    """Begin one atomic v97 hydraulic/scalar/Q3 transaction."""
    execution_document_v94b(execution)
    if (execution.backend != "cpu" or
            type(q3_provider) is not
                CPUQ3V83ActiveWetDryExnerAdapterV97):
        raise _v94.ActiveSuspendedControlV94Error(
            "v97 wet/dry route requires its endpoint-masked CPU provider")
    if not callable(cancelled):
        raise TypeError("cancelled must be callable")
    validate_active_suspended_control_state_v94b(
        parent, execution, q3_provider)
    if parent.accepted_updates >= _v94.MAX_ACCEPTED_UPDATES:
        raise _v94.ActiveSuspendedControlV94Error(
            "v97 wet/dry accepted update limit was reached")
    if cancelled():
        raise InterruptedError("v97 wet/dry cancelled before child work")
    inputs = dict(v92_scalar_inputs)
    forbidden = {
        "settling_diameter_receipt", "suspended_capacity_resolver",
        "suspended_transport_capacity_kg_m_s", "surface_fraction",
        "accepted_water_flux_m2_s", "dt_s",
    }
    overlap = forbidden.intersection(inputs)
    if overlap:
        raise _v94.ActiveSuspendedControlV94Error(
            "v97 wet/dry caller duplicates owned inputs: " +
            ", ".join(sorted(overlap)))
    inputs["surface_fraction"] = provider_surface_fraction_v114(q3_provider)
    if np.any(execution.iopsus_by_fraction == 1):
        settling = make_settling_diameter_receipt_v92(
            execution.base_execution.base_execution,
            parent.base_state.base_state,
            suspended_diameter_m=parent.suspended_diameter_m,
            source_kind=SETTLING_DIAMETER_ACTIVE_IOPSUS_V94,
            source_state_sha256=parent.state_sha256)
    else:
        settling = make_settling_diameter_receipt_v92(
            execution.base_execution.base_execution,
            parent.base_state.base_state,
            suspended_diameter_m=parent.suspended_diameter_m)
    provider_checkpoint = q3_provider.checkpoint()
    base_trial = None
    open_trial = None
    try:
        q3_provider.arm_capacity(
            control_parent=parent, q3_inputs=q3_inputs,
            cancelled=cancelled)
        v92_trial, native_scalar_evidence = \
            begin_transported_fluid_wetdry_cpu_trial_v97(
            execution.base_execution.base_execution,
            parent.base_state.base_state, parent_hles, handoff,
            hydraulic_evidence=hydraulic_evidence,
            reconstructed_left_fluid=reconstructed_left_fluid,
            reconstructed_right_fluid=reconstructed_right_fluid,
            prescribed_boundary_fluid=prescribed_boundary_fluid,
            scalar_inputs=inputs, scalar_fct_runtime=scalar_fct_runtime,
            scalar_diffusion_runtime=scalar_diffusion_runtime,
            settling_exchange_runtime=settling_exchange_runtime,
            scalar_boundary_runtime=scalar_boundary_runtime,
            native_scalar_threads=native_scalar_threads,
            settling_diameter_receipt=settling,
            suspended_capacity_resolver=q3_provider.resolve_armed_capacity,
            cancelled=cancelled)
        suspended = select_suspended_bed_cpu_v93(
            execution.base_execution, parent.base_state,
            parent_hles, handoff, v92_trial)
        open_trial = q3_provider.begin(
            outer_parent=parent.base_state, parent_hles=parent_hles,
            handoff=handoff, v92_trial=v92_trial,
            cancelled=cancelled)
        native_factcr_evidence = q3_provider.native_factcr_evidence_v97
        base_trial = _build_trial_from_live_provider(
            execution.base_execution, parent.base_state, v92_trial,
            suspended, q3_provider, provider_checkpoint, open_trial,
            backend="cpu_oracle")
        face = base_trial.q3_open_trial.face_trial
        if not isinstance(face, _v94.CPUQ3V83ActiveFaceTrialV94):
            raise _v94.ActiveSuspendedControlV94Error(
                "v97 wet/dry provider returned a different face trial")
        chain = _v94._fingerprint({
            "previous": parent.accepted_iopsus_chain_sha256,
            "base_v93_trial": base_trial.trial_sha256,
            "parent_DSS_receipt": settling.receipt_sha256,
            "v83_producer": face.producer_receipt.receipt_sha256,
            "candidate_DSS_sha256": _v94._array_digest(
                face.candidate_suspended_diameter_m),
            "iopsus_by_fraction": execution.iopsus_by_fraction.tolist(),
        })
        candidate = _v94._wrap_state(
            execution, base_trial.candidate,
            suspended_diameter_m=face.candidate_suspended_diameter_m,
            accepted_iopsus_chain_sha256=chain)
        ng, nc = execution.fraction_count, execution.cell_count
        ones = _v94._readonly(
            np.ones((ng, nc), np.int32), (ng, nc), "owner ones",
            integer=True)
        zeros = _v94._readonly(
            np.zeros((ng, nc), np.int32), (ng, nc), "owner zeros",
            integer=True)
        counts = ActiveSuspendedControlOwnerCountsV94(
            ones, ones, face.factcr_application_count, ones, zeros, ones,
            face.iopsus_application_count,
            face.suscor_correction_application_count,
            face.suspended_vector_construction_owner_count)
        placeholder = ActiveSuspendedControlTrialV94(
            execution.fingerprint, parent.state_sha256, base_trial,
            settling, v92_trial.suspended_capacity_receipt,
            face.producer_receipt, face.live_taurat,
            face.transport_stage, face.candidate_suspended_diameter_m,
            counts, candidate, "cpu_oracle", "")
        trial = ActiveSuspendedControlTrialV94(
            **{**placeholder.__dict__,
               "trial_sha256": _v94._trial_token(placeholder)})
        _v94._validate_trial(
            execution, parent, parent_hles, handoff, trial)
        if type(return_native_scalar_evidence) is not bool:
            raise _v94.ActiveSuspendedControlV94Error(
                "v97 native scalar evidence selector must be Boolean")
        return ((trial, native_scalar_evidence, native_factcr_evidence)
                if return_native_scalar_evidence else trial)
    except Exception as original_error:
        try:
            if (base_trial is not None and
                    q3_provider._closed is base_trial.q3_prepared_commit):
                q3_provider.abort_closed_commit(
                    base_trial.q3_prepared_commit)
            elif open_trial is not None and q3_provider._open is open_trial:
                q3_provider.rollback(open_trial)
            if (q3_provider._open is not None or
                    q3_provider._prepared is not None or
                    q3_provider._closed is not None):
                raise RuntimeError(
                    "v97 wet/dry cleanup left an open Q3 transaction")
            q3_provider.restore_checkpoint(provider_checkpoint)
        except Exception as cleanup_error:
            raise RuntimeError(
                "v97 wet/dry begin failed and restore also failed") \
                from cleanup_error
        raise original_error


__all__ = ["SCOPE_ID", "begin_active_suspended_wetdry_cpu_trial_v97"]
