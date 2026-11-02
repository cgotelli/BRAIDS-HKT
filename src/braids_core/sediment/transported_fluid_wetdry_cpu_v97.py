"""Authenticated v97 wet/dry hydraulic handoff to the v92 scalar owner.

The v97 native active-set solve owns the physical hydraulic midpoint,
endpoint, limited water-face flux, density projection, and masked v91
momentum source.  This module does not run another hydraulic solve.  It binds
that exact result to one accepted v92 parent/cap so the established combined
sediment plus temperature/salinity scalar transaction can consume it once.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Callable, Mapping

import numpy as np

from .shared_turbulence_consumers_v91 import (
    STATE_DOMAIN_MOVING_WETDRY_V97,
    ChkvicUsedTotalReceiptV91,
    SharedTurbulenceExecutionV91,
    _cap_token,
    execution_document_v91,
    make_hydraulic_turbulence_receipt_v91,
    validate_chkvic_receipt_v91,
)
from .wetdry_fwave_native_cpu_v97 import (
    WetDryFwaveCPUExecutionV97,
    WetDryMidpointTrialV97,
    execution_document_wetdry_fwave_v97,
    validate_wetdry_midpoint_trial_v97,
)


SCHEMA = 1
SCOPE_ID = "braids-q4-v92-wetdry-hydraulic-handoff-cpu-v97-v1"
_SHA = re.compile(r"[0-9a-f]{64}\Z")


class TransportedFluidWetDryCPUV97Error(ValueError):
    """The v97-to-v92 physical hydraulic handoff is invalid."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError, OverflowError) as error:
        raise TransportedFluidWetDryCPUV97Error(
            "v97 hydraulic handoff is not finite canonical JSON") from error


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _digest(value: object) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(_canonical(list(array.shape)))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _readonly(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != shape or raw.dtype.kind not in "fiu" or \
            not np.isfinite(raw).all():
        raise TransportedFluidWetDryCPUV97Error(
            f"v97 hydraulic handoff {name} differs")
    result = np.array(raw, dtype=np.float64, order="C", copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class WetDryHydraulicEvidenceV97:
    wetdry_execution: WetDryFwaveCPUExecutionV97
    wetdry_trial: WetDryMidpointTrialV97
    v91_execution_fingerprint: str
    parent_v92_state_sha256: str
    parent_v91_state_sha256: str
    chkvic_receipt_sha256: str
    accepted_bed_elevation_m: np.ndarray
    accepted_water_density_kg_m3: np.ndarray
    accepted_mixture_density_kg_m3: np.ndarray
    evidence_sha256: str


def _evidence_payload(value: WetDryHydraulicEvidenceV97) -> dict:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "wetdry_execution_fingerprint": value.wetdry_execution.fingerprint,
        "wetdry_trial_sha256": value.wetdry_trial.trial_sha256,
        "v91_execution_fingerprint": value.v91_execution_fingerprint,
        "parent_v92_state_sha256": value.parent_v92_state_sha256,
        "parent_v91_state_sha256": value.parent_v91_state_sha256,
        "chkvic_receipt_sha256": value.chkvic_receipt_sha256,
        "physical_arrays_sha256": [
            _digest(value.accepted_bed_elevation_m),
            _digest(value.accepted_water_density_kg_m3),
            _digest(value.accepted_mixture_density_kg_m3),
        ],
        "hydraulic_owner_count": 1,
        "scalar_owner_count": 0,
        "state_or_clock_publication_count": 0,
    }


def _validate_topology(
    execution: SharedTurbulenceExecutionV91,
    wetdry_execution: WetDryFwaveCPUExecutionV97,
) -> None:
    execution_document_v91(execution)
    execution_document_wetdry_fwave_v97(wetdry_execution)
    if execution.state_domain_mode != STATE_DOMAIN_MOVING_WETDRY_V97:
        raise TransportedFluidWetDryCPUV97Error(
            "v97 hydraulic evidence requires the moving-wet/dry state mode")
    grid = execution.base_execution.base_execution
    hydraulic = wetdry_execution.base_execution.base_execution
    equal = (
        execution.cell_count == hydraulic.cell_count and
        execution.face_count == hydraulic.face_count and
        np.array_equal(grid.owner, hydraulic.owner) and
        np.array_equal(grid.neighbour, hydraulic.neighbour) and
        np.array_equal(grid.normal, hydraulic.normal) and
        np.array_equal(grid.face_length_m, hydraulic.face_length_m) and
        np.array_equal(grid.centre_distance_m,
                       hydraulic.face_distance_m) and
        np.array_equal(grid.cell_area_m2, hydraulic.cell_area_m2) and
        np.array_equal(grid.cell_face_offset,
                       hydraulic.cell_face_offset) and
        np.array_equal(grid.cell_faces, hydraulic.cell_faces) and
        np.array_equal(grid.cell_face_sign, hydraulic.cell_face_sign) and
        grid.dry_tolerance_m == hydraulic.dry_tolerance_m and
        grid.profile.gravity_m_s2 == hydraulic.gravity_m_s2 and
        execution.base_execution.hydraulic_execution_sha256 ==
            hydraulic.hydraulic_execution_sha256)
    if not equal:
        raise TransportedFluidWetDryCPUV97Error(
            "v97 wet/dry hydraulic and v91 scalar topologies differ")


def make_wetdry_hydraulic_evidence_v97(
    v91_execution: SharedTurbulenceExecutionV91,
    parent_v92_state: object,
    cap_receipt: ChkvicUsedTotalReceiptV91, *,
    wetdry_execution: WetDryFwaveCPUExecutionV97,
    wetdry_trial: WetDryMidpointTrialV97,
    accepted_bed_elevation_m: object,
    accepted_water_density_kg_m3: object,
    accepted_mixture_density_kg_m3: object,
) -> WetDryHydraulicEvidenceV97:
    _validate_topology(v91_execution, wetdry_execution)
    nc = v91_execution.cell_count
    if (not hasattr(parent_v92_state, "state_sha256") or
            not hasattr(parent_v92_state, "base_state") or
            not hasattr(parent_v92_state, "q") or
            type(cap_receipt) is not ChkvicUsedTotalReceiptV91 or
            cap_receipt.execution_fingerprint != v91_execution.fingerprint or
            cap_receipt.receipt_sha256 != _cap_token(cap_receipt)):
        raise TransportedFluidWetDryCPUV97Error(
            "v97 hydraulic evidence parent/cap identity differs")
    bed = _readonly(
        accepted_bed_elevation_m, (nc,), "accepted bed elevation")
    water = _readonly(
        accepted_water_density_kg_m3, (nc,), "accepted water density")
    mixture = _readonly(
        accepted_mixture_density_kg_m3, (nc,), "accepted mixture density")
    if np.any(water <= 0.0) or np.any(mixture < water):
        raise TransportedFluidWetDryCPUV97Error(
            "v97 hydraulic evidence density domain differs")
    validate_wetdry_midpoint_trial_v97(
        wetdry_trial, wetdry_execution, parent_q=parent_v92_state.q,
        bed_elevation_m=bed, accepted_water_density_kg_m3=water,
        accepted_mixture_density_kg_m3=mixture,
        dt_s=wetdry_trial.dt_s)
    turbulence = wetdry_trial.shared_turbulence_result
    if (turbulence is None or
            turbulence.cap_receipt.receipt_sha256 !=
                cap_receipt.receipt_sha256 or
            wetdry_trial.shared_turbulence_application_count !=
                wetdry_trial.true_residual_evaluation_count):
        raise TransportedFluidWetDryCPUV97Error(
            "v97 hydraulic evidence requires one masked-v91 consumer in "
            "every true residual")
    placeholder = WetDryHydraulicEvidenceV97(
        wetdry_execution, wetdry_trial, v91_execution.fingerprint,
        parent_v92_state.state_sha256,
        parent_v92_state.base_state.state_sha256,
        cap_receipt.receipt_sha256, bed, water, mixture, "")
    return WetDryHydraulicEvidenceV97(
        **{**placeholder.__dict__,
           "evidence_sha256": _fingerprint(
               _evidence_payload(placeholder))})


def validate_wetdry_hydraulic_evidence_v97(
    value: WetDryHydraulicEvidenceV97,
    v91_execution: SharedTurbulenceExecutionV91,
    parent_v92_state: object,
    cap_receipt: ChkvicUsedTotalReceiptV91, *,
    midpoint_q: object, accepted_q: object,
    momentum_rate_m2_s2: object, momentum_face_flux_m4_s2: object,
) -> None:
    if type(value) is not WetDryHydraulicEvidenceV97:
        raise TransportedFluidWetDryCPUV97Error(
            "v97 hydraulic evidence type differs")
    expected = make_wetdry_hydraulic_evidence_v97(
        v91_execution, parent_v92_state, cap_receipt,
        wetdry_execution=value.wetdry_execution,
        wetdry_trial=value.wetdry_trial,
        accepted_bed_elevation_m=value.accepted_bed_elevation_m,
        accepted_water_density_kg_m3=
            value.accepted_water_density_kg_m3,
        accepted_mixture_density_kg_m3=
            value.accepted_mixture_density_kg_m3)
    trial = value.wetdry_trial
    hashes = (
        value.evidence_sha256, value.v91_execution_fingerprint,
        value.parent_v92_state_sha256, value.parent_v91_state_sha256,
        value.chkvic_receipt_sha256)
    if (any(_SHA.fullmatch(item) is None for item in hashes) or
            value.evidence_sha256 != expected.evidence_sha256 or
            not np.array_equal(midpoint_q, trial.midpoint_q) or
            not np.array_equal(accepted_q, trial.endpoint_q) or
            not np.array_equal(momentum_rate_m2_s2,
                               trial.accepted_turbulence_momentum_rate) or
            not np.array_equal(momentum_face_flux_m4_s2,
                               trial.accepted_turbulence_face_flux)):
        raise TransportedFluidWetDryCPUV97Error(
            "v97 hydraulic evidence is stale, tampered or cross-trial")


def begin_transported_fluid_wetdry_cpu_trial_v97(
    execution, parent, parent_hles, handoff, *,
    hydraulic_evidence: WetDryHydraulicEvidenceV97,
    reconstructed_left_fluid: object,
    reconstructed_right_fluid: object,
    prescribed_boundary_fluid: object,
    scalar_inputs: Mapping[str, object],
    scalar_fct_runtime,
    scalar_diffusion_runtime,
    settling_exchange_runtime=None,
    scalar_boundary_runtime=None,
    native_scalar_threads: int = 4,
    settling_diameter_receipt=None,
    suspended_capacity_resolver: Callable[..., object] | None = None,
    cancelled: bool | Callable[[], bool] = False,
):
    """Consume one accepted v97 hydraulic trial in the v92 scalar owner.

    No hydraulic, density, limiter, or masked-v91 calculation is repeated.
    The only numerical work here is the existing combined scalar FCT,
    diffusion, settling/exchange and dry-transfer transaction.
    """
    from . import transported_fluid_settling_v92 as _v92
    from .hles_producer_v90 import (
        validate_current_closure_handoff_v90,
        validate_hles_producer_state_v90,
    )
    from .scalar_diffusion_native_cpu_v96 import (
        NativeScalarDiffusionRuntimeV96,
    )
    from .scalar_fct_native_cpu_v96 import NativeScalarFCTRuntimeV96
    from .suspended_support_closures_v82 import (
        resolve_suspended_support_execution_v82,
    )
    from .transported_fluid_native_scalar_cpu_v96 import (
        suspended_dynamic_trial_native_cpu_v96,
    )

    if execution.backend != _v92.BACKEND_CPU or \
            execution.base_execution.state_domain_mode != \
                STATE_DOMAIN_MOVING_WETDRY_V97:
        raise TransportedFluidWetDryCPUV97Error(
            "v97 scalar owner requires the explicit CPU moving-wet/dry "
            "execution")
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
            "v97 scalar owner cancelled before validation")
    if (parent.accepted_updates >= _v92.MAX_UPDATES or
            parent.accepted_updates != parent_hles.accepted_updates or
            parent.time_s != parent_hles.time_s):
        raise TransportedFluidWetDryCPUV97Error(
            "v97 scalar parent clock/update bound differs")
    if (type(scalar_fct_runtime) is not NativeScalarFCTRuntimeV96 or
            type(scalar_diffusion_runtime) is not
                NativeScalarDiffusionRuntimeV96):
        raise TransportedFluidWetDryCPUV97Error(
            "v97 scalar owner requires the exact native FCT/diffusion "
            "runtimes")
    validate_chkvic_receipt_v91(
        hydraulic_evidence.wetdry_trial.shared_turbulence_result.cap_receipt,
        execution.base_execution, parent_hles, handoff)
    cap = (
        hydraulic_evidence.wetdry_trial.shared_turbulence_result.
        cap_receipt)
    trial_h = hydraulic_evidence.wetdry_trial
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
        capacity, native_evidence = suspended_dynamic_trial_native_cpu_v96(
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
            native_threads=native_scalar_threads,
            suspended_capacity_resolver=suspended_capacity_resolver,
            cancelled=poll)
    chain_v91 = _v92._fingerprint({
        "previous": parent.base_state.accepted_receipt_chain_sha256,
        "hydraulic": hydraulic.receipt_sha256,
        "wetdry_hydraulic_evidence": hydraulic_evidence.evidence_sha256,
        "scalar": suspended.trial_sha256,
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
        "native_scalar_evidence": native_evidence.evidence_sha256,
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
        native_scalar_evidence=native_evidence)
    _v92._validate_trial(execution, parent, parent_hles, handoff, trial)
    if poll():
        raise InterruptedError(
            "v97 scalar owner cancelled before publication")
    return trial, native_evidence


__all__ = [
    "SCOPE_ID", "TransportedFluidWetDryCPUV97Error",
    "WetDryHydraulicEvidenceV97", "make_wetdry_hydraulic_evidence_v97",
    "validate_wetdry_hydraulic_evidence_v97",
    "begin_transported_fluid_wetdry_cpu_trial_v97",
]
