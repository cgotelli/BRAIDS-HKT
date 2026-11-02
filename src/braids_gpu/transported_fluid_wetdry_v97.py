"""Authenticated actual-CUDA v97 hydraulic handoff to the v92 owner.

This module carries one provisional resident wet/dry midpoint result into the
shared transported-fluid transaction.  It performs no hydraulic, limiter,
density, or v91 numerical work and publishes no state or clock.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re

import numpy as np

from braids_core.sediment.shared_turbulence_consumers_v91 import (
    STATE_DOMAIN_MOVING_WETDRY_V97,
    ChkvicUsedTotalReceiptV91,
    SharedTurbulenceExecutionV91,
    _cap_token,
    make_hydraulic_turbulence_receipt_v91,
    validate_chkvic_receipt_v91,
    execution_document_v91,
)
from braids_core.sediment.hles_producer_v90 import (
    validate_current_closure_handoff_v90,
    validate_hles_producer_state_v90,
)
from braids_core.sediment.transported_fluid_settling_v92 import (
    MAX_UPDATES,
    _validate_state as _validate_v92_state,
)
from .wetdry_fwave_v97 import (
    WetDryFwaveGPUExecutionV97,
    WetDryFwaveGPUTrialV97,
    execution_document_wetdry_fwave_gpu_v97,
    validate_wetdry_fwave_gpu_trial_v97,
)


SCOPE_ID = "braids-q4-v92-wetdry-hydraulic-handoff-actual-cuda-v97-v1"
BACKEND = "actual_cuda"
_SHA = re.compile(r"[0-9a-f]{64}\Z")


class TransportedFluidWetDryGPUV97Error(ValueError):
    """The exact CUDA wet/dry-to-v92 handoff differs or was tampered."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          allow_nan=False).encode("ascii")
    except (TypeError, ValueError, OverflowError) as error:
        raise TransportedFluidWetDryGPUV97Error(
            "v97 CUDA hydraulic evidence is not canonical JSON") from error


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _digest(value: object) -> str:
    array = np.ascontiguousarray(value)
    result = hashlib.sha256()
    result.update(array.dtype.str.encode("ascii"))
    result.update(_canonical(list(array.shape)))
    result.update(array.tobytes(order="C"))
    return result.hexdigest()


def _readonly(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    raw = np.asarray(value)
    if (raw.shape != shape or raw.dtype.kind not in "fiu" or
            not np.isfinite(raw).all()):
        raise TransportedFluidWetDryGPUV97Error(
            f"v97 CUDA hydraulic evidence {name} differs")
    result = np.array(raw, dtype=np.float64, order="C", copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class WetDryHydraulicEvidenceGPUV97:
    wetdry_execution: WetDryFwaveGPUExecutionV97
    wetdry_trial: WetDryFwaveGPUTrialV97
    v91_execution_fingerprint: str
    parent_v92_state_sha256: str
    parent_v91_state_sha256: str
    chkvic_receipt_sha256: str
    accepted_bed_elevation_m: np.ndarray
    accepted_water_density_kg_m3: np.ndarray
    accepted_mixture_density_kg_m3: np.ndarray
    backend: str
    evidence_sha256: str


def _payload(value: WetDryHydraulicEvidenceGPUV97) -> dict[str, object]:
    return {
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
        "backend": value.backend,
        "hydraulic_owner_count": 1,
        "scalar_owner_count": 0,
        "state_or_clock_publication_count": 0,
        "automatic_cpu_fallback": False,
    }


def _validate_topology(
    execution: SharedTurbulenceExecutionV91,
    wetdry: WetDryFwaveGPUExecutionV97,
) -> None:
    execution_document_v91(execution)
    execution_document_wetdry_fwave_gpu_v97(wetdry)
    if (execution.state_domain_mode != STATE_DOMAIN_MOVING_WETDRY_V97 or
            execution.backend != "implicit_fwave_midpoint_gpu" or
            not wetdry.shared_turbulence_enabled or
            wetdry.shared_turbulence_execution_fingerprint !=
                execution.fingerprint):
        raise TransportedFluidWetDryGPUV97Error(
            "v97 CUDA evidence execution modes differ")
    grid = execution.base_execution.base_execution
    hydraulic = wetdry.density_execution
    pairs = (
        (grid.owner, hydraulic.owner),
        (grid.neighbour, hydraulic.neighbour),
        (grid.normal, hydraulic.normal),
        (grid.face_length_m, hydraulic.face_length_m),
        (grid.centre_distance_m, hydraulic.face_distance_m),
        (grid.cell_area_m2, hydraulic.cell_area_m2),
        (grid.cell_face_offset, hydraulic.cell_face_offset),
        (grid.cell_faces, hydraulic.cell_faces),
        (grid.cell_face_sign, hydraulic.cell_face_sign),
    )
    if (execution.cell_count != hydraulic.cell_count or
            execution.face_count != hydraulic.face_count or
            any(not np.array_equal(left, right) for left, right in pairs) or
            grid.dry_tolerance_m != hydraulic.dry_tolerance_m or
            grid.profile.gravity_m_s2 != hydraulic.gravity_m_s2):
        raise TransportedFluidWetDryGPUV97Error(
            "v97 CUDA evidence topologies differ")


def make_wetdry_hydraulic_evidence_gpu_v97(
    v91_execution: SharedTurbulenceExecutionV91,
    parent_v92_state: object,
    cap_receipt: ChkvicUsedTotalReceiptV91, *,
    wetdry_execution: WetDryFwaveGPUExecutionV97,
    wetdry_trial: WetDryFwaveGPUTrialV97,
    accepted_bed_elevation_m: object,
    accepted_water_density_kg_m3: object,
    accepted_mixture_density_kg_m3: object,
) -> WetDryHydraulicEvidenceGPUV97:
    _validate_topology(v91_execution, wetdry_execution)
    nc = v91_execution.cell_count
    if (not hasattr(parent_v92_state, "state_sha256") or
            not hasattr(parent_v92_state, "base_state") or
            not hasattr(parent_v92_state, "q") or
            type(cap_receipt) is not ChkvicUsedTotalReceiptV91 or
            cap_receipt.execution_fingerprint != v91_execution.fingerprint or
            cap_receipt.receipt_sha256 != _cap_token(cap_receipt)):
        raise TransportedFluidWetDryGPUV97Error(
            "v97 CUDA evidence parent/cap identity differs")
    bed = _readonly(accepted_bed_elevation_m, (nc,), "accepted bed")
    water = _readonly(
        accepted_water_density_kg_m3, (nc,), "accepted water density")
    mixture = _readonly(
        accepted_mixture_density_kg_m3, (nc,), "accepted mixture density")
    if np.any(water <= 0.0) or np.any(mixture < water):
        raise TransportedFluidWetDryGPUV97Error(
            "v97 CUDA evidence density domain differs")
    validate_wetdry_fwave_gpu_trial_v97(
        wetdry_trial, wetdry_execution)
    if (wetdry_trial.parent_q_sha256 != _digest(parent_v92_state.q) or
            not np.array_equal(wetdry_trial.parent_q,
                               parent_v92_state.q) or
            wetdry_trial.accepted_bed_sha256 != _digest(bed) or
            wetdry_trial.accepted_water_density_sha256 != _digest(water) or
            wetdry_trial.accepted_mixture_density_sha256 !=
                _digest(mixture) or
            wetdry_trial.shared_turbulence_cap_receipt_sha256 !=
                cap_receipt.receipt_sha256 or
            wetdry_trial.shared_turbulence_application_count !=
                wetdry_trial.true_residual_evaluation_count):
        raise TransportedFluidWetDryGPUV97Error(
            "v97 CUDA evidence physical trial inputs differ")
    placeholder = WetDryHydraulicEvidenceGPUV97(
        wetdry_execution, wetdry_trial, v91_execution.fingerprint,
        parent_v92_state.state_sha256,
        parent_v92_state.base_state.state_sha256,
        cap_receipt.receipt_sha256, bed, water, mixture, BACKEND, "")
    return WetDryHydraulicEvidenceGPUV97(
        **{**placeholder.__dict__,
           "evidence_sha256": _hash(_payload(placeholder))})


def validate_wetdry_hydraulic_evidence_gpu_v97(
    value: WetDryHydraulicEvidenceGPUV97,
    v91_execution: SharedTurbulenceExecutionV91,
    parent_v92_state: object,
    cap_receipt: ChkvicUsedTotalReceiptV91, *,
    midpoint_q: object,
    accepted_q: object,
    momentum_rate_m2_s2: object,
    momentum_face_flux_m4_s2: object,
) -> None:
    if type(value) is not WetDryHydraulicEvidenceGPUV97:
        raise TransportedFluidWetDryGPUV97Error(
            "v97 CUDA hydraulic evidence concrete type differs")
    expected = make_wetdry_hydraulic_evidence_gpu_v97(
        v91_execution, parent_v92_state, cap_receipt,
        wetdry_execution=value.wetdry_execution,
        wetdry_trial=value.wetdry_trial,
        accepted_bed_elevation_m=value.accepted_bed_elevation_m,
        accepted_water_density_kg_m3=value.accepted_water_density_kg_m3,
        accepted_mixture_density_kg_m3=value.accepted_mixture_density_kg_m3)
    trial = value.wetdry_trial
    identities = (
        value.evidence_sha256, value.v91_execution_fingerprint,
        value.parent_v92_state_sha256, value.parent_v91_state_sha256,
        value.chkvic_receipt_sha256,
    )
    if (any(_SHA.fullmatch(item) is None for item in identities) or
            value.backend != BACKEND or
            value.evidence_sha256 != expected.evidence_sha256 or
            not np.array_equal(midpoint_q, trial.midpoint_q) or
            not np.array_equal(accepted_q, trial.endpoint_q) or
            not np.array_equal(momentum_rate_m2_s2,
                               trial.shared_turbulence_rate) or
            not np.array_equal(momentum_face_flux_m4_s2,
                               trial.shared_turbulence_face_flux)):
        raise TransportedFluidWetDryGPUV97Error(
            "v97 CUDA evidence is stale, tampered, or cross-trial")


def _cancelled(value) -> bool:
    if type(value) is bool:
        return value
    if callable(value):
        return bool(value())
    raise TypeError("cancelled must be Boolean or callable")


def begin_transported_fluid_wetdry_gpu_trial_v97(
    runtime,
    parent,
    parent_hles,
    handoff, *,
    cap_receipt: ChkvicUsedTotalReceiptV91,
    hydraulic_evidence: WetDryHydraulicEvidenceGPUV97,
    reconstructed_left_fluid,
    reconstructed_right_fluid,
    prescribed_boundary_fluid,
    settling_diameter_receipt=None,
    suspended_capacity_resolver=None,
    cancelled=False,
    **scalar_inputs,
):
    """Consume one v97 endpoint in the existing actual-CUDA scalar owner.

    No controller, augmented Riemann, draining, density, or v91 momentum
    kernel is relaunched here.  The function only arms the historical v92
    combined scalar/exchange owner with the exact authenticated endpoint and
    limited face-water flux produced by :class:`ResidentWetDryFwaveV97`.
    """
    from braids_gpu.transported_fluid_lifecycle_v92 import (
        ResidentTransportedFluidSettlingLifecycleV92,
    )

    if type(runtime) is not ResidentTransportedFluidSettlingLifecycleV92:
        raise TransportedFluidWetDryGPUV97Error(
            "v97 CUDA scalar handoff requires the exact v92 lifecycle")
    execution = runtime.execution
    _validate_v92_state(parent, execution)
    if execution.base_execution.fingerprint != \
            hydraulic_evidence.v91_execution_fingerprint:
        raise TransportedFluidWetDryGPUV97Error(
            "v97 CUDA scalar/v91 execution differs")
    validate_hles_producer_state_v90(
        parent_hles, execution.base_execution.hles_execution)
    validate_current_closure_handoff_v90(
        handoff, execution.base_execution.hles_execution, parent_hles)
    validate_chkvic_receipt_v91(
        cap_receipt, execution.base_execution, parent_hles, handoff)
    validate_wetdry_hydraulic_evidence_gpu_v97(
        hydraulic_evidence, execution.base_execution, parent, cap_receipt,
        midpoint_q=hydraulic_evidence.wetdry_trial.midpoint_q,
        accepted_q=hydraulic_evidence.wetdry_trial.endpoint_q,
        momentum_rate_m2_s2=
            hydraulic_evidence.wetdry_trial.shared_turbulence_rate,
        momentum_face_flux_m4_s2=
            hydraulic_evidence.wetdry_trial.shared_turbulence_face_flux)
    cancelled_now = _cancelled(cancelled)
    if (cancelled_now or parent.accepted_updates >= MAX_UPDATES or
            parent.accepted_updates != parent_hles.accepted_updates or
            parent.time_s != parent_hles.time_s):
        if cancelled_now:
            raise InterruptedError(
                "v97 CUDA scalar owner cancelled before child work")
        raise TransportedFluidWetDryGPUV97Error(
            "v97 CUDA scalar parent clock/update differs")
    if (runtime._parent is not None or runtime._trial is not None or
            runtime.base._open or runtime.base._pending is not None or
            runtime.combined._open):
        raise TransportedFluidWetDryGPUV97Error(
            "v97 CUDA previous scalar transaction is unfinished")

    trial_h = hydraulic_evidence.wetdry_trial
    hydraulic = make_hydraulic_turbulence_receipt_v91(
        execution.base_execution, parent.base_state, parent_hles, handoff,
        cap_receipt, midpoint_q=trial_h.midpoint_q,
        accepted_q=trial_h.endpoint_q, dt_s=trial_h.dt_s,
        residual_evaluation_count=trial_h.true_residual_evaluation_count,
        momentum_rate_evaluation_count=
            trial_h.true_residual_evaluation_count,
        backend=BACKEND, actual_implicit_controller=True)
    base = runtime.base
    try:
        base._parent = parent.base_state
        base._parent_hles = parent_hles
        base._handoff = handoff
        base._cap = cap_receipt
        base._dt = trial_h.dt_s
        base._time = parent.time_s
        base._residual_evaluations = trial_h.true_residual_evaluation_count
        base._cancelled_callback = cancelled
        base.capped_vic.set(cap_receipt.capped_total_vicuv_m2_s)
        base.capped_dic.set(cap_receipt.capped_total_dicuv_m2_s)
        base._pending = {
            "receipt": hydraulic,
            "midpoint": trial_h.midpoint_q,
            "accepted": trial_h.endpoint_q,
            "water": trial_h.accepted_limited_face_water_flux_m2_s,
            "rate": trial_h.shared_turbulence_rate,
            "flux": trial_h.shared_turbulence_face_flux,
        }
        runtime._parent = parent
        runtime._parent_hles = parent_hles
        runtime._handoff = handoff
        result = runtime.begin_scalar_trial(
            reconstructed_left_fluid=reconstructed_left_fluid,
            reconstructed_right_fluid=reconstructed_right_fluid,
            prescribed_boundary_fluid=prescribed_boundary_fluid,
            settling_diameter_receipt=settling_diameter_receipt,
            suspended_capacity_resolver=suspended_capacity_resolver,
            wetdry_hydraulic_evidence=hydraulic_evidence,
            cancelled=cancelled, **scalar_inputs)
        if result.wetdry_hydraulic_evidence is not hydraulic_evidence:
            raise TransportedFluidWetDryGPUV97Error(
                "v97 CUDA scalar trial lost hydraulic evidence identity")
        return result
    except Exception:
        runtime.abort_hydraulic_trial()
        raise


__all__ = [
    "BACKEND", "SCOPE_ID", "TransportedFluidWetDryGPUV97Error",
    "WetDryHydraulicEvidenceGPUV97",
    "make_wetdry_hydraulic_evidence_gpu_v97",
    "begin_transported_fluid_wetdry_gpu_trial_v97",
    "validate_wetdry_hydraulic_evidence_gpu_v97",
]
