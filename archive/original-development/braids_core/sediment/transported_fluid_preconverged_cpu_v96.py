"""Additive v96 scalar phase for an already converged CPU hydraulic step.

The legacy v92 CPU entry point owns its own midpoint momentum solve.  A v96
owner that has already composed augmented f-wave, DensIn and native v91
diffusion must not call it.  This seam consumes one authenticated endpoint,
face-water field and native v91 cap, then runs only the existing suspended and
active-fluid scalar lifecycle.  The NumPy v91 rate is recomputed solely as a
strict equation oracle for the frozen v92 trial ABI; it is never added to the
hydraulic state a second time.
"""
from __future__ import annotations

from typing import Callable, Mapping

import numpy as np

from . import transported_fluid_settling_v92 as _v92
from .hles_producer_v90 import (
    CurrentClosureHandoffV90,
    HlesProducerStateV90,
    validate_current_closure_handoff_v90,
    validate_hles_producer_state_v90,
)
from .shared_turbulence_consumers_v91 import (
    ChkvicUsedTotalReceiptV91,
    make_hydraulic_turbulence_receipt_v91,
    momentum_diffusion_rate_v91,
    validate_chkvic_receipt_v91,
)
from .suspended_support_closures_v82 import (
    resolve_suspended_support_execution_v82,
)
from .transported_fluid_settling_v92 import (
    BACKEND_CPU,
    MAX_UPDATES,
    SettlingDiameterReceiptV92,
    TransportedFluidExecutionV92,
    TransportedFluidSettlingV92Error,
    TransportedFluidStateV92,
    TransportedFluidTrialV92,
    make_accepted_scalar_hydraulic_face_receipt_v92,
    make_settling_diameter_receipt_v92,
    validate_settling_diameter_receipt_v92,
)
from .scalar_diffusion_native_cpu_v96 import (
    NativeScalarDiffusionRuntimeV96,
)
from .scalar_fct_native_cpu_v96 import NativeScalarFCTRuntimeV96
from .transported_fluid_native_scalar_cpu_v96 import (
    NativeCombinedScalarEvidenceV96,
    suspended_dynamic_trial_native_cpu_v96,
)


SCOPE_ID = "braids-q4-v92-preconverged-scalar-cpu-v96-v1"


def begin_transported_fluid_preconverged_cpu_trial_v96(
    execution: TransportedFluidExecutionV92,
    parent: TransportedFluidStateV92,
    parent_hles: HlesProducerStateV90,
    handoff: CurrentClosureHandoffV90, *,
    cap_receipt: ChkvicUsedTotalReceiptV91,
    midpoint_q: object,
    accepted_q: object,
    native_momentum_rate_m2_s2: object,
    native_momentum_face_flux_m4_s2: object,
    residual_evaluation_count: int,
    reconstructed_left_fluid: object,
    reconstructed_right_fluid: object,
    prescribed_boundary_fluid: object,
    accepted_water_flux_m2_s: object,
    dt_s: float,
    settling_diameter_receipt: SettlingDiameterReceiptV92 | None = None,
    suspended_capacity_resolver: Callable[..., object] | None = None,
    scalar_fct_runtime: NativeScalarFCTRuntimeV96 | None = None,
    scalar_diffusion_runtime: NativeScalarDiffusionRuntimeV96 | None = None,
    native_scalar_threads: int = 4,
    return_native_scalar_evidence: bool = False,
    cancelled: bool | Callable[[], bool] = False,
    **scalar_inputs,
) -> (TransportedFluidTrialV92 |
      tuple[TransportedFluidTrialV92, NativeCombinedScalarEvidenceV96 | None]):
    """Run the v92 scalar/exchange phase without another hydraulic iterate."""
    if execution.backend != BACKEND_CPU:
        raise TransportedFluidSettlingV92Error(
            "v96 preconverged scalar route requires explicit CPU execution")
    _v92._validate_state(parent, execution)
    if parent.accepted_updates >= MAX_UPDATES:
        raise TransportedFluidSettlingV92Error(
            "v96 preconverged scalar accepted update limit was reached")
    validate_hles_producer_state_v90(
        parent_hles, execution.base_execution.hles_execution)
    validate_current_closure_handoff_v90(
        handoff, execution.base_execution.hles_execution, parent_hles)
    if type(cancelled) is bool:
        cancel_check = lambda: cancelled
    elif callable(cancelled):
        cancel_check = cancelled
    else:
        raise TypeError("cancelled must be Boolean or callable")
    if cancel_check():
        raise InterruptedError(
            "v96 preconverged scalar cancelled before validation")
    if (parent.accepted_updates != parent_hles.accepted_updates or
            parent.time_s != parent_hles.time_s):
        raise TransportedFluidSettlingV92Error(
            "v96 preconverged scalar parent clocks differ")
    dt = _v92._finite(dt_s, "preconverged timestep", positive=True)
    if (type(residual_evaluation_count) is not int or
            residual_evaluation_count < 1):
        raise TransportedFluidSettlingV92Error(
            "v96 preconverged residual count is invalid")
    validate_chkvic_receipt_v91(
        cap_receipt, execution.base_execution, parent_hles, handoff)
    midpoint = _v92._array(
        midpoint_q, (3, execution.cell_count), "preconverged midpoint")
    endpoint = _v92._array(
        accepted_q, (3, execution.cell_count), "preconverged endpoint")
    native_rate = _v92._array(
        native_momentum_rate_m2_s2, (3, execution.cell_count),
        "native momentum rate")
    native_flux = _v92._array(
        native_momentum_face_flux_m4_s2, (2, execution.face_count),
        "native momentum face flux")
    # Validation oracle only: the native result remains the physical rate in
    # the outer v96 residual.  Reusing the exact oracle arrays here preserves
    # all frozen v92/v93/v94 trial validation without a second source apply.
    oracle_rate, oracle_flux = momentum_diffusion_rate_v91(
        execution.base_execution, midpoint,
        cap_receipt.capped_total_vicuv_m2_s)
    rate_scale = max(1.0, float(np.max(np.abs(oracle_rate))))
    flux_scale = max(1.0, float(np.max(np.abs(oracle_flux))))
    if (not np.allclose(native_rate, oracle_rate, rtol=2e-14,
                        atol=2e-14*np.finfo(float).eps*rate_scale) or
            not np.allclose(native_flux, oracle_flux, rtol=2e-14,
                            atol=2e-14*np.finfo(float).eps*flux_scale)):
        raise TransportedFluidSettlingV92Error(
            "v96 native momentum differs from the v91 equation oracle")
    hydraulic = make_hydraulic_turbulence_receipt_v91(
        execution.base_execution, parent.base_state, parent_hles, handoff,
        cap_receipt, midpoint_q=midpoint, accepted_q=endpoint, dt_s=dt,
        residual_evaluation_count=residual_evaluation_count,
        momentum_rate_evaluation_count=residual_evaluation_count,
        backend="cpu_oracle", actual_implicit_controller=False)
    face = make_accepted_scalar_hydraulic_face_receipt_v92(
        execution, parent, hydraulic, accepted_q=endpoint,
        accepted_water_flux_m2_s=accepted_water_flux_m2_s)
    if settling_diameter_receipt is None:
        settling_diameter_receipt = make_settling_diameter_receipt_v92(
            execution, parent)
    validate_settling_diameter_receipt_v92(
        settling_diameter_receipt, execution, parent)
    scalar_execution = resolve_suspended_support_execution_v82(
        execution.base_execution.base_execution,
        dicuv_cell_m2_s=cap_receipt.capped_total_dicuv_m2_s)
    native_evidence: NativeCombinedScalarEvidenceV96 | None = None
    if scalar_fct_runtime is None and scalar_diffusion_runtime is None:
        suspended_trial, fluid, parent_ctot, settling_rho, settling_nu, \
            capacity_receipt = _v92._suspended_dynamic_trial_cpu(
                execution, parent, scalar_execution, hydraulic,
                hydraulic_face_receipt=face,
                settling_diameter_receipt=settling_diameter_receipt,
                depth_before_m=np.asarray(parent.q[0]),
                depth_after_m=np.asarray(endpoint[0]),
                endpoint_momentum_m2_s=np.asarray(endpoint[1:]),
                reconstructed_left_fluid=reconstructed_left_fluid,
                reconstructed_right_fluid=reconstructed_right_fluid,
                prescribed_boundary_fluid=prescribed_boundary_fluid,
                dt_s=dt, scalar_inputs=dict(scalar_inputs),
                suspended_capacity_resolver=suspended_capacity_resolver,
                return_capacity_receipt=True, cancelled=cancel_check)
    elif (type(scalar_fct_runtime) is NativeScalarFCTRuntimeV96 and
          type(scalar_diffusion_runtime) is
              NativeScalarDiffusionRuntimeV96):
        suspended_trial, fluid, parent_ctot, settling_rho, settling_nu, \
            capacity_receipt, native_evidence = \
            suspended_dynamic_trial_native_cpu_v96(
                execution, parent, scalar_execution, hydraulic,
                hydraulic_face_receipt=face,
                settling_diameter_receipt=settling_diameter_receipt,
                depth_before_m=np.asarray(parent.q[0]),
                depth_after_m=np.asarray(endpoint[0]),
                endpoint_momentum_m2_s=np.asarray(endpoint[1:]),
                reconstructed_left_fluid=reconstructed_left_fluid,
                reconstructed_right_fluid=reconstructed_right_fluid,
                prescribed_boundary_fluid=prescribed_boundary_fluid,
                dt_s=dt, scalar_inputs=dict(scalar_inputs),
                fct_runtime=scalar_fct_runtime,
                diffusion_runtime=scalar_diffusion_runtime,
                native_threads=native_scalar_threads,
                suspended_capacity_resolver=suspended_capacity_resolver,
                cancelled=cancel_check)
    else:
        raise TransportedFluidSettlingV92Error(
            "v96 native FCT and diffusion runtimes must be supplied together")
    chain_v91 = _v92._fingerprint({
        "previous": parent.base_state.accepted_receipt_chain_sha256,
        "hydraulic": hydraulic.receipt_sha256,
        "scalar": suspended_trial.trial_sha256,
    })
    candidate_base = _v92._wrap_v91_state(
        execution.base_execution, q=endpoint,
        suspended_state=suspended_trial.candidate.base_state,
        receipt_chain=chain_v91)
    chain = _v92._fingerprint({
        "previous": parent.accepted_receipt_chain_sha256,
        "hydraulic": hydraulic.receipt_sha256,
        "hydraulic_face": face.receipt_sha256,
        "suspended": suspended_trial.trial_sha256,
        "fluid": fluid.result_sha256,
        "preconverged_scalar_scope": SCOPE_ID,
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
        execution, parent, cap_receipt=cap_receipt,
        hydraulic_receipt=hydraulic, hydraulic_face_receipt=face,
        settling_diameter_receipt=settling_diameter_receipt,
        suspended_capacity_receipt=capacity_receipt,
        scalar_execution=scalar_execution, suspended_trial=suspended_trial,
        fluid_transport=fluid, midpoint_q=midpoint,
        momentum_rate_m2_s2=oracle_rate,
        momentum_face_flux_m4_s2=oracle_flux,
        parent_total_suspended_concentration_kg_m3=parent_ctot,
        settling_water_density_kg_m3=settling_rho,
        settling_molecular_viscosity_m2_s=settling_nu,
        candidate=candidate)
    _v92._validate_trial(execution, parent, parent_hles, handoff, trial)
    if type(return_native_scalar_evidence) is not bool:
        raise TransportedFluidSettlingV92Error(
            "v96 native scalar evidence selector must be Boolean")
    return (trial, native_evidence) if return_native_scalar_evidence else trial


__all__ = ["begin_transported_fluid_preconverged_cpu_trial_v96"]
