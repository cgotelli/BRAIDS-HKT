"""One v98 AMR-interface scalar transaction on an accepted hydraulic trial.

This additive owner keeps the accepted v92 lifecycle and its exact state
objects.  The historical v97 hydraulic evidence remains the compatibility
receipt for the already-converged active-set endpoint, while the authoritative
scalar producer is the v98 internal-interface FCT runtime.  The extra typed
evidence records the non-physical coarse/fine ledger; it owns no boundary
memory or clock.
"""
from __future__ import annotations

from typing import Callable, Mapping

import numpy as np

from .coupled_amr_scalar_trace_cpu_v98 import CoupledAMRScalarTraceCPUV98
from .scalar_diffusion_native_cpu_v96 import NativeScalarDiffusionRuntimeV96
from .scalar_fct_interface_native_cpu_v98 import (
    NativeScalarFCTInterfaceRuntimeV98,
)
from .shared_turbulence_consumers_v91 import (
    STATE_DOMAIN_MOVING_WETDRY_V97,
    make_hydraulic_turbulence_receipt_v91,
    validate_chkvic_receipt_v91,
)
from .suspended_support_closures_v82 import (
    resolve_suspended_support_execution_v82,
)
from .transported_fluid_amr_scalar_cpu_v98 import (
    NativeCombinedScalarInterfaceEvidenceV98,
    suspended_dynamic_trial_amr_scalar_cpu_v98,
    validate_native_combined_scalar_interface_evidence_v98,
)
from .transported_fluid_wetdry_cpu_v97 import (
    TransportedFluidWetDryCPUV97Error,
    WetDryHydraulicEvidenceV97,
    validate_wetdry_hydraulic_evidence_v97,
)


SCOPE_ID = "braids-q4-v92-amr-interface-wetdry-cpu-v98-v1"


def begin_transported_fluid_amr_wetdry_cpu_trial_v98(
    execution, parent, parent_hles, handoff, *,
    hydraulic_evidence: WetDryHydraulicEvidenceV97,
    scalar_trace: CoupledAMRScalarTraceCPUV98,
    substep_index: int,
    reconstructed_left_fluid: object,
    reconstructed_right_fluid: object,
    prescribed_boundary_fluid: object,
    scalar_inputs: Mapping[str, object],
    scalar_fct_runtime: NativeScalarFCTInterfaceRuntimeV98,
    scalar_diffusion_runtime: NativeScalarDiffusionRuntimeV96,
    settling_exchange_runtime=None,
    scalar_boundary_runtime=None,
    native_scalar_threads: int = 4,
    settling_diameter_receipt=None,
    suspended_capacity_resolver: Callable[..., object] | None = None,
    cancelled: bool | Callable[[], bool] = False,
):
    """Consume exactly one hydraulic endpoint and one AMR scalar trace."""
    from . import transported_fluid_settling_v92 as _v92
    from .hles_producer_v90 import (
        validate_current_closure_handoff_v90,
        validate_hles_producer_state_v90,
    )

    if (execution.backend != _v92.BACKEND_CPU or
            execution.base_execution.state_domain_mode !=
                STATE_DOMAIN_MOVING_WETDRY_V97 or
            type(hydraulic_evidence) is not WetDryHydraulicEvidenceV97 or
            type(scalar_trace) is not CoupledAMRScalarTraceCPUV98 or
            type(scalar_fct_runtime) is not
                NativeScalarFCTInterfaceRuntimeV98 or
            type(scalar_diffusion_runtime) is not
                NativeScalarDiffusionRuntimeV96 or
            type(substep_index) is not int or substep_index not in (0, 1)):
        raise TransportedFluidWetDryCPUV97Error(
            "v98 AMR scalar owner capability differs")
    _v92._validate_state(parent, execution)
    validate_hles_producer_state_v90(
        parent_hles, execution.base_execution.hles_execution)
    validate_current_closure_handoff_v90(
        handoff, execution.base_execution.hles_execution, parent_hles)
    if type(cancelled) is bool:
        poll = lambda: cancelled
    elif callable(cancelled):
        poll = cancelled
    else:
        raise TypeError("cancelled must be Boolean or callable")
    if poll():
        raise InterruptedError(
            "v98 AMR scalar owner cancelled before validation")
    if (parent.accepted_updates >= _v92.MAX_UPDATES or
            parent.accepted_updates != parent_hles.accepted_updates or
            parent.time_s != parent_hles.time_s):
        raise TransportedFluidWetDryCPUV97Error(
            "v98 AMR scalar parent clock/update bound differs")

    trial_h = hydraulic_evidence.wetdry_trial
    cap = trial_h.shared_turbulence_result.cap_receipt
    validate_chkvic_receipt_v91(
        cap, execution.base_execution, parent_hles, handoff)
    validate_wetdry_hydraulic_evidence_v97(
        hydraulic_evidence, execution.base_execution, parent, cap,
        midpoint_q=trial_h.midpoint_q, accepted_q=trial_h.endpoint_q,
        momentum_rate_m2_s2=trial_h.accepted_turbulence_momentum_rate,
        momentum_face_flux_m4_s2=trial_h.accepted_turbulence_face_flux)
    dt = trial_h.dt_s
    hydraulic = make_hydraulic_turbulence_receipt_v91(
        execution.base_execution, parent.base_state, parent_hles, handoff,
        cap, midpoint_q=trial_h.midpoint_q,
        accepted_q=trial_h.endpoint_q, dt_s=dt,
        residual_evaluation_count=trial_h.true_residual_evaluation_count,
        momentum_rate_evaluation_count=
            trial_h.true_residual_evaluation_count,
        backend="cpu_oracle", actual_implicit_controller=False)
    face = _v92.make_accepted_scalar_hydraulic_face_receipt_v92(
        execution, parent, hydraulic, accepted_q=trial_h.endpoint_q,
        accepted_water_flux_m2_s=
            trial_h.accepted_limited_face_water_flux_m2_s)
    if settling_diameter_receipt is None:
        settling_diameter_receipt = \
            _v92.make_settling_diameter_receipt_v92(execution, parent)
    _v92.validate_settling_diameter_receipt_v92(
        settling_diameter_receipt, execution, parent)
    scalar_execution = resolve_suspended_support_execution_v82(
        execution.base_execution.base_execution,
        dicuv_cell_m2_s=cap.capped_total_dicuv_m2_s)
    suspended, fluid, parent_ctot, settling_rho, settling_nu, \
        capacity, compatibility_evidence, interface_evidence = \
        suspended_dynamic_trial_amr_scalar_cpu_v98(
            execution, parent, scalar_execution, hydraulic,
            hydraulic_face_receipt=face,
            settling_diameter_receipt=settling_diameter_receipt,
            depth_before_m=np.asarray(parent.q[0]),
            depth_after_m=np.asarray(trial_h.endpoint_q[0]),
            endpoint_momentum_m2_s=np.asarray(trial_h.endpoint_q[1:]),
            reconstructed_left_fluid=reconstructed_left_fluid,
            reconstructed_right_fluid=reconstructed_right_fluid,
            prescribed_boundary_fluid=prescribed_boundary_fluid,
            dt_s=dt, scalar_inputs=dict(scalar_inputs),
            fct_runtime=scalar_fct_runtime,
            diffusion_runtime=scalar_diffusion_runtime,
            settling_exchange_runtime=settling_exchange_runtime,
            scalar_boundary_runtime=scalar_boundary_runtime,
            scalar_trace=scalar_trace, substep_index=substep_index,
            native_threads=native_scalar_threads,
            suspended_capacity_resolver=suspended_capacity_resolver,
            cancelled=poll)
    chain_v91 = _v92._fingerprint({
        "previous": parent.base_state.accepted_receipt_chain_sha256,
        "hydraulic": hydraulic.receipt_sha256,
        "wetdry_hydraulic_evidence": hydraulic_evidence.evidence_sha256,
        "scalar": suspended.trial_sha256,
        "amr_scalar": interface_evidence.evidence_sha256,
    })
    candidate_base = _v92._wrap_v91_state(
        execution.base_execution, q=trial_h.endpoint_q,
        suspended_state=suspended.candidate.base_state,
        receipt_chain=chain_v91)
    chain = _v92._fingerprint({
        "previous": parent.accepted_receipt_chain_sha256,
        "hydraulic": hydraulic.receipt_sha256,
        "hydraulic_face": face.receipt_sha256,
        "wetdry_hydraulic_evidence": hydraulic_evidence.evidence_sha256,
        "suspended": suspended.trial_sha256,
        "fluid": fluid.result_sha256,
        "native_scalar_evidence": compatibility_evidence.evidence_sha256,
        "amr_scalar_evidence": interface_evidence.evidence_sha256,
    })
    candidate = _v92._wrap_state(
        execution, candidate_base,
        h_temperature_c_m=fluid.h_temperature_c_m,
        h_salinity_ppt_m=fluid.h_salinity_ppt_m,
        boundary_last_outflow_fluid=fluid.boundary_last_outflow_fluid,
        boundary_return_clock_s=fluid.boundary_return_clock_s,
        reference_scalar_integral=parent.reference_scalar_integral,
        advective_boundary_ledger=(
            parent.advective_boundary_ledger+
            fluid.advective_boundary_ledger),
        diffusive_boundary_ledger=(
            parent.diffusive_boundary_ledger+
            fluid.diffusive_boundary_ledger),
        wetdry_scalar_ledger=(
            parent.wetdry_scalar_ledger+fluid.wetdry_scalar_ledger),
        accepted_receipt_chain_sha256=chain)
    trial = _v92._wrap_trial(
        execution, parent, cap_receipt=cap,
        hydraulic_receipt=hydraulic, hydraulic_face_receipt=face,
        settling_diameter_receipt=settling_diameter_receipt,
        suspended_capacity_receipt=capacity,
        scalar_execution=scalar_execution, suspended_trial=suspended,
        fluid_transport=fluid, midpoint_q=trial_h.midpoint_q,
        momentum_rate_m2_s2=
            trial_h.accepted_turbulence_momentum_rate,
        momentum_face_flux_m4_s2=
            trial_h.accepted_turbulence_face_flux,
        parent_total_suspended_concentration_kg_m3=parent_ctot,
        settling_water_density_kg_m3=settling_rho,
        settling_molecular_viscosity_m2_s=settling_nu,
        candidate=candidate, wetdry_hydraulic_evidence=hydraulic_evidence,
        native_scalar_evidence=compatibility_evidence)
    _v92._validate_trial(execution, parent, parent_hles, handoff, trial)
    validate_native_combined_scalar_interface_evidence_v98(
        interface_evidence, scalar_trace=scalar_trace,
        fct_runtime=scalar_fct_runtime, substep_index=substep_index,
        suspended_trial_sha256=suspended.trial_sha256,
        fluid_result_sha256=fluid.result_sha256)
    if poll():
        raise InterruptedError(
            "v98 AMR scalar owner cancelled before trial return")
    return trial, interface_evidence


__all__ = [
    "SCOPE_ID", "begin_transported_fluid_amr_wetdry_cpu_trial_v98",
]
