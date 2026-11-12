"""Audited v97 midpoint solve with the additive v98 coarse/fine trace.

The native v98 rate adapter returns the exact frozen v97 rate-result type, so
the accepted active-set iteration remains the v97 owner.  This module is the
strict admission boundary: it binds every residual evaluation to one selected
trace slice and records the deliberately narrow first interface scope.

The additive density-trace face operator owns exterior DensIn on the two
coarse/fine faces during every true residual; the unchanged v97 owner then
projects the combined ordinary-internal plus trace system exactly once.
The historical internal-only v91 operator and a typed pre-capped trace operator
are evaluated separately and composed into the v98 residual without a second
cap or clock.  This is hydraulic evidence only; it is not scalar/Q3
interface qualification and it does not publish a clock or state.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Callable, Mapping

import numpy as np

from .coupled_amr_fwave_native_cpu_v98 import (
    AMRInterfaceResidualAuditV98,
    INTERFACE_FACE_COUNT,
    NativeAMRInterfaceFwaveRuntimeV98,
    _digest as _interface_digest,
    validate_amr_interface_residual_audit_v98,
)
from .coupled_geometry_native_cpu_v96 import DENSITY_WATER_PLUS_SEDIMENT
from .shared_turbulence_native_cpu_v97 import NativeMomentumRateV97
from .wetdry_fwave_native_cpu_v97 import (
    WetDryFwaveCPUExecutionV97,
    WetDryMidpointTrialV97,
    execution_document_wetdry_fwave_v97,
    solve_wetdry_midpoint_trial_v97,
    validate_wetdry_midpoint_trial_v97,
)


SCHEMA = 1
SCOPE_ID = "braids.q4.coupled_amr.wetdry.cpu.v98"


class CoupledAMRWetDryCPUV98Error(ValueError):
    """The external-trace midpoint trial or its audit is invalid."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError, OverflowError) as error:
        raise CoupledAMRWetDryCPUV98Error(
            "v98 external-trace identity is not canonical JSON") from error


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


@dataclass(frozen=True)
class WetDryAMRInterfaceTrialV98:
    execution_fingerprint: str
    trace_receipt_sha256: str
    substep_index: int
    wetdry_trial: WetDryMidpointTrialV97
    residual_audit: AMRInterfaceResidualAuditV98
    accepted_internal_turbulence_result: NativeMomentumRateV97
    accepted_trace_turbulence_result: Mapping[str, object]
    density_trace_interface_face_count: int
    density_trace_augmentation_application_count: int
    density_trace_projection_application_count: int
    interface_viscosity_face_count: int
    interface_turbulence_flux_linf_m4_s2: float
    scalar_interface_owner_count: int
    q3_interface_owner_count: int
    state_or_clock_publication_count: int
    trial_sha256: str


def _trial_payload(value: WetDryAMRInterfaceTrialV98) -> dict:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID + ":trial",
        "execution_fingerprint": value.execution_fingerprint,
        "trace_receipt_sha256": value.trace_receipt_sha256,
        "substep_index": value.substep_index,
        "wetdry_trial_sha256": value.wetdry_trial.trial_sha256,
        "residual_audit_sha256": value.residual_audit.audit_sha256,
        "accepted_internal_turbulence_result_sha256":
            value.accepted_internal_turbulence_result.result_sha256,
        "accepted_trace_turbulence_result_sha256":
            value.accepted_trace_turbulence_result["receipt_sha256"],
        "density_trace_interface_face_count":
            value.density_trace_interface_face_count,
        "density_trace_augmentation_application_count":
            value.density_trace_augmentation_application_count,
        "density_trace_projection_application_count":
            value.density_trace_projection_application_count,
        "interface_viscosity_face_count":
            value.interface_viscosity_face_count,
        "interface_turbulence_flux_linf_m4_s2":
            value.interface_turbulence_flux_linf_m4_s2,
        "scalar_interface_owner_count": value.scalar_interface_owner_count,
        "q3_interface_owner_count": value.q3_interface_owner_count,
        "state_or_clock_publication_count":
            value.state_or_clock_publication_count,
        "scope": (
            "typed_density_and_pre_capped_viscosity_trace_hydraulics_only"),
        "scalar_q3_interface_qualification": False,
    }


def _selected_density(
    execution: WetDryFwaveCPUExecutionV97,
    water: np.ndarray,
    mixture: np.ndarray,
) -> np.ndarray:
    return (mixture if execution.base_execution.density_mode ==
            DENSITY_WATER_PLUS_SEDIMENT else water)


def _validate_interface_trace(
    execution: WetDryFwaveCPUExecutionV97,
    runtime: NativeAMRInterfaceFwaveRuntimeV98,
    water: np.ndarray,
    mixture: np.ndarray,
) -> str:
    """Return the trace digest; DensIn now admits nonuniform density."""
    selected = _selected_density(execution, water, mixture)
    expanded = runtime._expanded_trace(
        density_reference=float(np.min(selected)))
    return _interface_digest(*expanded)


def validate_wetdry_amr_interface_trial_v98(
    value: WetDryAMRInterfaceTrialV98,
    execution: WetDryFwaveCPUExecutionV97, *,
    runtime: NativeAMRInterfaceFwaveRuntimeV98,
    parent_q: object,
    bed_elevation_m: object,
    accepted_water_density_kg_m3: object,
    accepted_mixture_density_kg_m3: object,
    dt_s: float,
) -> None:
    if (type(value) is not WetDryAMRInterfaceTrialV98 or
            type(runtime) is not NativeAMRInterfaceFwaveRuntimeV98 or
            runtime.execution is not execution or
            runtime.substep_index not in (0, 1)):
        raise CoupledAMRWetDryCPUV98Error(
            "v98 external-trace trial/runtime type differs")
    execution_document_wetdry_fwave_v97(execution)
    validate_wetdry_midpoint_trial_v97(
        value.wetdry_trial, execution, parent_q=parent_q,
        bed_elevation_m=bed_elevation_m,
        accepted_water_density_kg_m3=accepted_water_density_kg_m3,
        accepted_mixture_density_kg_m3=accepted_mixture_density_kg_m3,
        dt_s=dt_s)
    validate_amr_interface_residual_audit_v98(value.residual_audit)
    audit = value.residual_audit
    trial = value.wetdry_trial
    water = np.asarray(accepted_water_density_kg_m3, dtype=np.float64)
    mixture = np.asarray(
        accepted_mixture_density_kg_m3, dtype=np.float64)
    expected_slice = _validate_interface_trace(
        execution, runtime, water, mixture)
    faces = tuple(runtime.trace.fine_faces)
    receipts = audit.rate_receipts
    turbulence_results = runtime.turbulence_results()
    if not turbulence_results:
        raise CoupledAMRWetDryCPUV98Error(
            "v98 external-trace turbulence audit is empty")
    internal_turbulence, trace_turbulence = turbulence_results[-1]
    trace_flux = np.asarray(trace_turbulence["face_flux_m4_s2"])
    interface_flux_linf = float(np.max(np.abs(
        trace_flux[:, np.asarray(faces)])))
    if (not receipts or
            value.execution_fingerprint != execution.fingerprint or
            value.trace_receipt_sha256 != runtime.trace.receipt_sha256 or
            value.substep_index != runtime.substep_index or
            audit.execution_fingerprint != execution.fingerprint or
            audit.trace_receipt_sha256 != runtime.trace.receipt_sha256 or
            audit.substep_index != runtime.substep_index or
            audit.interface_faces != faces or
            audit.evaluation_count !=
                trial.true_residual_evaluation_count or
            audit.interface_riemann_call_count !=
                INTERFACE_FACE_COUNT*audit.evaluation_count or
            audit.discarded_reflective_interface_riemann_call_count !=
                INTERFACE_FACE_COUNT*audit.evaluation_count or
            audit.total_native_riemann_call_count !=
                (execution.face_count+INTERFACE_FACE_COUNT)*
                audit.evaluation_count or
            audit.density_trace_augmentation_application_count !=
                audit.evaluation_count or
            audit.density_trace_projection_application_count != 0 or
            audit.density_trace_face_evaluation_count !=
                INTERFACE_FACE_COUNT*audit.evaluation_count or
            audit.internal_turbulence_application_count !=
                audit.evaluation_count or
            audit.trace_turbulence_application_count !=
                audit.evaluation_count or
            audit.trace_viscosity_face_evaluation_count !=
                INTERFACE_FACE_COUNT*audit.evaluation_count or
            audit.turbulence_cap_application_count != 0 or
            len(turbulence_results) != audit.evaluation_count or
            any(receipt.trace_slice_sha256 != expected_slice or
                receipt.total_native_riemann_call_count !=
                    execution.face_count+INTERFACE_FACE_COUNT or
                receipt.density_trace_augmentation_application_count != 1 or
                receipt.density_trace_projection_application_count != 0 or
                receipt.density_trace_face_evaluation_count !=
                    INTERFACE_FACE_COUNT or
                receipt.internal_turbulence_result_sha256 !=
                    turbulence_results[index][0].result_sha256 or
                receipt.trace_turbulence_result_sha256 !=
                    turbulence_results[index][1]["receipt_sha256"] or
                receipt.internal_turbulence_application_count != 1 or
                receipt.trace_turbulence_application_count != 1 or
                receipt.trace_viscosity_face_evaluation_count !=
                    INTERFACE_FACE_COUNT or
                receipt.turbulence_cap_application_count != 0
                for index, receipt in enumerate(receipts)) or
            receipts[-1].v97_result_sha256 !=
                trial.native_rate_result.result_sha256 or
            receipts[-1].midpoint_sha256 !=
                trial.native_rate_result.midpoint_sha256 or
            trial.shared_turbulence_result is not None or
            trial.shared_turbulence_application_count != 0 or
            np.any(trial.accepted_turbulence_momentum_rate != 0.0) or
            value.accepted_internal_turbulence_result is not
                internal_turbulence or
            value.accepted_trace_turbulence_result is not trace_turbulence or
            np.any(np.asarray(internal_turbulence.face_flux_m4_s2)[
                :, np.asarray(faces)] != 0.0) or
            value.density_trace_interface_face_count !=
                INTERFACE_FACE_COUNT or
            value.density_trace_augmentation_application_count !=
                audit.evaluation_count or
            value.density_trace_projection_application_count !=
                trial.density_projection_application_count or
            trial.density_projection_application_count !=
                trial.true_residual_evaluation_count or
            value.interface_viscosity_face_count !=
                INTERFACE_FACE_COUNT or
            value.interface_turbulence_flux_linf_m4_s2 !=
                interface_flux_linf or
            value.scalar_interface_owner_count != 0 or
            value.q3_interface_owner_count != 0 or
            value.state_or_clock_publication_count != 0 or
            value.trial_sha256 != _fingerprint(_trial_payload(value))):
        raise CoupledAMRWetDryCPUV98Error(
            "v98 external-trace residual audit or bounded scope differs")


def solve_wetdry_midpoint_with_external_trace_v98(
    execution: WetDryFwaveCPUExecutionV97,
    parent_q: object, *,
    bed_elevation_m: object,
    accepted_water_density_kg_m3: object,
    accepted_mixture_density_kg_m3: object,
    dt_s: float,
    native_runtime: NativeAMRInterfaceFwaveRuntimeV98,
    substep_index: int,
    density_projection_runtime,
    shared_turbulence_runtime=None,
    shared_turbulence_prepared=None,
    cancelled: bool | Callable[[], bool] = False,
) -> WetDryAMRInterfaceTrialV98:
    """Run one audited fine hydraulic substep against a coarse trace.

    On cancellation or failure the runtime audit is cleared, so an exact
    retry starts from an empty residual-evidence inventory.
    """
    if (type(native_runtime) is not NativeAMRInterfaceFwaveRuntimeV98 or
            native_runtime.execution is not execution or
            type(substep_index) is not int or substep_index not in (0, 1) or
            native_runtime.substep_index != substep_index):
        raise CoupledAMRWetDryCPUV98Error(
            "v98 external-trace runtime/substep binding differs")
    execution_document_wetdry_fwave_v97(execution)
    water = np.asarray(accepted_water_density_kg_m3, dtype=np.float64)
    mixture = np.asarray(
        accepted_mixture_density_kg_m3, dtype=np.float64)
    _validate_interface_trace(
        execution, native_runtime, water, mixture)
    native_runtime.reset_residual_audit()
    try:
        trial = solve_wetdry_midpoint_trial_v97(
            execution, parent_q, bed_elevation_m=bed_elevation_m,
            accepted_water_density_kg_m3=water,
            accepted_mixture_density_kg_m3=mixture, dt_s=dt_s,
            native_runtime=native_runtime,
            density_projection_runtime=density_projection_runtime,
            shared_turbulence_runtime=shared_turbulence_runtime,
            shared_turbulence_prepared=shared_turbulence_prepared,
            cancelled=cancelled)
        audit = native_runtime.residual_audit()
        turbulence_results = native_runtime.turbulence_results()
        if len(turbulence_results) != audit.evaluation_count:
            raise CoupledAMRWetDryCPUV98Error(
                "v98 external-trace turbulence result count differs")
        internal_turbulence, trace_turbulence = turbulence_results[-1]
        interface_flux_linf = float(np.max(np.abs(
            np.asarray(trace_turbulence["face_flux_m4_s2"])[
                :, np.asarray(native_runtime.trace.fine_faces)])))
        placeholder = WetDryAMRInterfaceTrialV98(
            execution.fingerprint, native_runtime.trace.receipt_sha256,
            substep_index, trial, audit,
            internal_turbulence, trace_turbulence, INTERFACE_FACE_COUNT,
            audit.density_trace_augmentation_application_count,
            trial.density_projection_application_count,
            INTERFACE_FACE_COUNT, interface_flux_linf, 0, 0, 0, "")
        result = WetDryAMRInterfaceTrialV98(
            **{**placeholder.__dict__, "trial_sha256": _fingerprint(
                _trial_payload(placeholder))})
        validate_wetdry_amr_interface_trial_v98(
            result, execution, runtime=native_runtime,
            parent_q=parent_q, bed_elevation_m=bed_elevation_m,
            accepted_water_density_kg_m3=water,
            accepted_mixture_density_kg_m3=mixture, dt_s=dt_s)
        return result
    except Exception:
        native_runtime.reset_residual_audit()
        raise


def method_identity_coupled_amr_wetdry_cpu_v98() -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "backend": "native_cpu",
        "time_integrator": "frozen_v97_active_set_midpoint",
        "spatial_solver": "augmented_fwave_external_coarse_trace",
        "density_interface": (
            "typed_trace_face_augmentation_then_one_v97_projection"),
        "interface_shared_viscosity": "typed_pre_capped_trace_v91_law",
        "interior_shared_viscosity": (
            "existing_v97_native_operator_embedded_in_v98_rate"),
        "discarded_reflective_trace_calls": "authenticated",
        "scalar_interface_owner_count": 0,
        "q3_interface_owner_count": 0,
        "production_admission": False,
    }


__all__ = [
    "CoupledAMRWetDryCPUV98Error", "SCHEMA", "SCOPE_ID",
    "WetDryAMRInterfaceTrialV98",
    "method_identity_coupled_amr_wetdry_cpu_v98",
    "solve_wetdry_midpoint_with_external_trace_v98",
    "validate_wetdry_amr_interface_trial_v98",
]
