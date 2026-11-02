"""Private actual-CUDA v98 fine state and complete face-time integrals.

This module is the additive state boundary between one physical fine update
and the AMR register.  It never publishes a legacy v97 state: local v78-v94
ledger windows are rebased around the private candidate, while the signed
internal-interface transfers remain authoritative in the outer v98 receipt.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Callable

import numpy as np

from braids_core.sediment.active_suspended_control_v94 import (
    _wrap_state as _wrap_v94_state,
)
from braids_core.sediment.hles_producer_v90 import (
    CurrentClosureHandoffV90,
    HlesProducerTrialV90,
    make_hles_input_receipt_v90,
)
from braids_core.sediment.multifraction_suspended_lifecycle_v78 import (
    resolve_multifraction_suspended_state_v78,
)
from braids_core.sediment.shared_turbulence_consumers_v91 import (
    ChkvicUsedTotalReceiptV91,
    _wrap_state as _wrap_v91_state,
    make_hydraulic_turbulence_receipt_v91,
    validate_chkvic_receipt_v91,
)
from braids_core.sediment.suspended_exner_owner_v93 import (
    _wrap_state as _wrap_v93_state,
)
from braids_core.sediment.transported_fluid_settling_v92 import (
    _fluid_result_token,
    _wrap_state as _wrap_v92_state,
)
from braids_gpu.active_suspended_wetdry_v97 import (
    ResidentQ3V83ActiveWetDryExnerAdapterV97,
)
from braids_gpu.coupled_amr_fwave_v98 import (
    ResidentAMRInterfaceFwaveGPUV98,
    WetDryAMRInterfaceGPUTrialV98,
)
from braids_gpu.coupled_amr_q3_face_v98 import (
    AMRQ3FaceResultGPUV98,
    ResidentAMRQ3FaceGPUV98,
    validate_amr_q3_face_result_gpu_v98,
)
from braids_gpu.coupled_amr_register_v98 import (
    CoupledFaceTimeIntegralsGPUV98,
    SOURCE_PRIVATE_FINE,
    make_private_fine_face_time_integrals_gpu_v98,
    validate_coupled_face_time_integrals_gpu_v98,
)
from braids_gpu.coupled_amr_scalar_exchange_v98 import (
    AMRScalarExchangeResultGPUV98,
    ResidentAMRScalarExchangeGPUV98,
    validate_amr_scalar_exchange_result_gpu_v98,
)
from braids_gpu.coupled_amr_scalar_trace_v98 import (
    CoupledAMRScalarTraceGPUV98,
)
from braids_gpu.coupled_amr_trace_v98 import (
    CoupledAMRInternalTraceGPUV98,
)
from braids_gpu.coupled_wetdry_v97 import (
    CoupledWetDryGPUExecutionV97,
    CoupledWetDryGPUStateV97,
    _grid,
    _hles_inputs,
    _v91_execution,
    _v92_execution,
    _v92_state,
    _wrap_state as _wrap_v97_state,
    validate_coupled_wetdry_gpu_state_v97,
)
from braids_gpu.hles_producer_v90 import ResidentHlesProducerLifecycleV90
from braids_gpu.q3_exner_adapter_v93 import (
    resolve_resident_q3_v84_mixed_state_v93,
)


SCHEMA = 1
SCOPE_ID = "braids.q4.coupled_amr.fine_substep.actual_cuda.v98"
BACKEND = "actual_cuda_private_amr"


class CoupledAMRFineSubstepGPUV98Error(ValueError):
    """The private CUDA fine state or register image differs."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError, OverflowError) as error:
        raise CoupledAMRFineSubstepGPUV98Error(
            "v98 CUDA fine identity is not canonical JSON") from error


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _digest(*values: object) -> str:
    digest = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(value)
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(_canonical(list(array.shape)))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _poll(value: bool | Callable[[], bool]) -> Callable[[], bool]:
    if type(value) is bool:
        return lambda: value
    if callable(value):
        return lambda: bool(value())
    raise CoupledAMRFineSubstepGPUV98Error(
        "v98 CUDA fine cancellation must be bool or callable")


def _frozen(value: object, shape: tuple[int, ...], name: str,
            dtype=np.float64) -> np.ndarray:
    raw = np.asarray(value)
    if (raw.shape != shape or raw.dtype.kind not in "fiu" or
            not np.isfinite(raw).all()):
        raise CoupledAMRFineSubstepGPUV98Error(
            f"v98 CUDA fine {name} differs")
    result = np.array(raw, dtype=dtype, order="C", copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class CoupledAMRFineSubstepCountsGPUV98:
    hydraulic_interface_owner_count: int = 1
    scalar_interface_owner_count: int = 1
    factcr_owner_count: int = 1
    q3_interface_owner_count: int = 1
    exchange_owner_count: int = 1
    hles_owner_count: int = 1
    face_time_integral_owner_count: int = 1
    internal_interface_ledger_owner_count: int = 1
    physical_boundary_surrogate_count: int = 0
    accepted_state_publication_count: int = 0
    source_provider_mutation_count: int = 0


@dataclass(frozen=True)
class CoupledAMRFineSubstepStateGPUV98:
    execution_fingerprint: str
    parent_state_sha256: str
    source_provider_state_sha256: str
    internal_trace_receipt_sha256: str
    scalar_trace_receipt_sha256: str
    geometry_binding_receipt_sha256: str | None
    substep_index: int
    dt_s: float
    hydraulic_trial_sha256: str
    hydraulic_auxiliary_evidence_sha256: str | None
    scalar_exchange: AMRScalarExchangeResultGPUV98
    q3_interface_result: AMRQ3FaceResultGPUV98
    face_time_integrals: CoupledFaceTimeIntegralsGPUV98
    transport_stage: np.ndarray
    suspended_diameter_m: np.ndarray
    q3_state: object
    hles_trial: HlesProducerTrialV90
    numerical_parent: CoupledWetDryGPUStateV97
    scalar_internal_interface_ledger_kg: np.ndarray
    bed_internal_interface_ledger_kg: np.ndarray
    physical_suspended_boundary_increment_kg: np.ndarray
    physical_fluid_boundary_increment: np.ndarray
    counts: CoupledAMRFineSubstepCountsGPUV98
    backend: str
    state_sha256: str


def _state_payload(value: CoupledAMRFineSubstepStateGPUV98) -> dict:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID+":state",
        "execution": value.execution_fingerprint,
        "parent": value.parent_state_sha256,
        "source_provider": value.source_provider_state_sha256,
        "traces": [value.internal_trace_receipt_sha256,
                   value.scalar_trace_receipt_sha256],
        "geometry_binding": value.geometry_binding_receipt_sha256,
        "substep_index": value.substep_index,
        "dt_s": value.dt_s,
        "hydraulic": value.hydraulic_trial_sha256,
        "hydraulic_auxiliary_evidence":
            value.hydraulic_auxiliary_evidence_sha256,
        "scalar": value.scalar_exchange.result_sha256,
        "q3": value.q3_interface_result.result_sha256,
        "face_time_integrals": value.face_time_integrals.receipt_sha256,
        "q3_state": value.q3_state.state_sha256,
        "hles": value.hles_trial.trial_sha256,
        "numerical_parent": value.numerical_parent.state_sha256,
        "arrays_sha256": _digest(
            value.transport_stage, value.suspended_diameter_m,
            value.scalar_internal_interface_ledger_kg,
            value.bed_internal_interface_ledger_kg,
            value.physical_suspended_boundary_increment_kg,
            value.physical_fluid_boundary_increment),
        "counts": list(value.counts.__dict__.values()),
        "backend": value.backend,
        "internal_ledgers_authoritative": True,
        "numerical_parent_local_windows_rebased": True,
        "public_state_or_clock_publication_count": 0,
    }


def _build_private_parent(
    execution: CoupledWetDryGPUExecutionV97,
    parent: CoupledWetDryGPUStateV97,
    source_provider: ResidentQ3V83ActiveWetDryExnerAdapterV97,
    hydraulic: WetDryAMRInterfaceGPUTrialV98,
    scalar: AMRScalarExchangeResultGPUV98,
    q3: AMRQ3FaceResultGPUV98,
    handoff: CurrentClosureHandoffV90,
    cap: ChkvicUsedTotalReceiptV91,
    hles_runtime: ResidentHlesProducerLifecycleV90,
    closure_chezy_mhalf_s: object,
    poll: Callable[[], bool],
):
    v92_execution = _v92_execution(execution.active_execution)
    v91_execution = _v91_execution(execution.active_execution)
    v78_execution = v91_execution.base_execution
    grid = _grid(execution.active_execution)
    area = np.asarray(grid.cell_area_m2)
    bed = np.ascontiguousarray(
        scalar.final_bed_mass_kg_m2+q3.cell_delta_kg_m2)
    if np.any(bed < 0.0):
        raise CoupledAMRFineSubstepGPUV98Error(
            "v98 CUDA fine Q3 overdraws exchange bed")
    suspended = np.ascontiguousarray(scalar.final_suspended_mass_kg_m2)
    water_reference = np.sum(suspended*area[None, :], axis=1)
    bed_reference = np.sum(bed*area[None, :], axis=1)
    trial_h = hydraulic.wetdry_trial
    accepted_updates = parent.accepted_updates+1
    time_s = parent.time_s+trial_h.dt_s
    v78 = resolve_multifraction_suspended_state_v78(
        v78_execution, suspended_mass_kg_m2=suspended,
        bed_mass_kg_m2=bed,
        boundary_last_outflow_concentration_kg_m3=
            scalar.boundary_last_outflow_concentration_kg_m3,
        boundary_return_clock_s=scalar.boundary_return_clock_s,
        advective_boundary_ledger_kg=np.zeros((
            execution.fraction_count, v78_execution.base_execution.
            boundary_count)),
        diffusive_boundary_ledger_kg=np.zeros((
            execution.fraction_count, v78_execution.base_execution.
            boundary_count)),
        water_exchange_ledger_kg=np.zeros(execution.fraction_count),
        bed_exchange_ledger_kg=np.zeros(execution.fraction_count),
        wetdry_water_ledger_kg=np.zeros(execution.fraction_count),
        wetdry_bed_ledger_kg=np.zeros(execution.fraction_count),
        reference_water_mass_kg=water_reference,
        reference_bed_mass_kg=bed_reference,
        accepted_updates=accepted_updates, time_s=time_s)
    v91 = _wrap_v91_state(
        v91_execution, q=trial_h.endpoint_q, suspended_state=v78,
        receipt_chain=_hash({
            "scope_id": SCOPE_ID+":v91-private",
            "parent": parent.state_sha256,
            "hydraulic": hydraulic.trial_sha256,
            "scalar": scalar.result_sha256,
            "q3": q3.result_sha256,
            "local_windows_rebased": True,
        }))
    fluid = scalar.fluid_result
    scalar_reference = np.array((
        np.sum(fluid.h_temperature_c_m*area, dtype=np.float64),
        np.sum(fluid.h_salinity_ppt_m*area, dtype=np.float64)))
    v92 = _wrap_v92_state(
        v92_execution, v91,
        h_temperature_c_m=fluid.h_temperature_c_m,
        h_salinity_ppt_m=fluid.h_salinity_ppt_m,
        boundary_last_outflow_fluid=fluid.boundary_last_outflow_fluid,
        boundary_return_clock_s=fluid.boundary_return_clock_s,
        reference_scalar_integral=scalar_reference,
        advective_boundary_ledger=np.zeros(
            (2, v92_execution.boundary_count)),
        diffusive_boundary_ledger=np.zeros(
            (2, v92_execution.boundary_count)),
        wetdry_scalar_ledger=np.zeros(2),
        accepted_receipt_chain_sha256=_hash({
            "scope_id": SCOPE_ID+":v92-private",
            "parent": parent.state_sha256,
            "scalar": scalar.result_sha256,
            "local_windows_rebased": True,
        }),
        temperature_cell_c=fluid.temperature_cell_c,
        salinity_cell_ppt=fluid.salinity_cell_ppt,
        water_density_cell_kg_m3=scalar.endpoint_water_density_kg_m3,
        molecular_viscosity_cell_m2_s=
            scalar.endpoint_kinematic_viscosity_m2_s)

    source_q3 = source_provider.state
    q3_state = resolve_resident_q3_v84_mixed_state_v93(
        execution_fingerprint=
            execution.active_execution.factcr_execution.fingerprint,
        binding=source_provider.binding,
        authoritative_bed_mass_kg_m2=bed,
        bulk_density_kg_m3=source_q3.core.bulk_density_kg_m3,
        retained_fraction=source_q3.core.active_fraction,
        accepted_face_flux_kg_m_s=q3.face_flux_kg_m_s,
        accepted_cell_availability=q3.availability,
        named_boundary_positive_in_kg=(
            source_q3.core.named_boundary_positive_in_kg+
            q3.named_boundary_transfer_kg),
        named_boundary_absolute_kg=(
            source_q3.core.named_boundary_absolute_kg+
            q3.named_boundary_absolute_kg),
        depletion_roundoff_positive_in_kg=(
            source_q3.core.depletion_roundoff_positive_in_kg+
            q3.depletion_roundoff_positive_in_kg),
        accepted_updates=accepted_updates, time_s=time_s)
    private_provider = ResidentQ3V83ActiveWetDryExnerAdapterV97(
        outer_execution=execution.active_execution,
        binding=source_provider.binding, state=q3_state)
    v93_execution = execution.active_execution.base_execution
    v93 = _wrap_v93_state(
        v93_execution, v92,
        q3_provider_kind=private_provider.provider_kind,
        q3_provider_state_sha256=q3_state.state_sha256,
        bed_mass_kg_m2=bed,
        reference_bed_integral_kg=bed_reference,
        suspended_exchange_ledger_kg=np.zeros(execution.fraction_count),
        suspended_boundary_ledger_kg=np.zeros((
            execution.fraction_count, v92_execution.boundary_count)),
        excluded_water_exchange_kg=np.zeros(execution.fraction_count),
        q3_named_boundary_positive_in_kg=np.zeros((
            execution.fraction_count, len(v93_execution.q3_boundary_names))),
        q3_named_boundary_absolute_kg=np.zeros((
            execution.fraction_count, len(v93_execution.q3_boundary_names))),
        q3_depletion_roundoff_positive_in_kg=np.zeros(
            execution.fraction_count),
        wetdry_bed_ledger_kg=np.zeros(execution.fraction_count),
        accepted_exner_chain_sha256=_hash({
            "scope_id": SCOPE_ID+":v93-private",
            "q3": q3.result_sha256,
            "q3_state": q3_state.state_sha256,
            "local_windows_rebased": True,
        }))
    active = _wrap_v94_state(
        execution.active_execution, v93,
        suspended_diameter_m=scalar.candidate_suspended_diameter_m,
        accepted_iopsus_chain_sha256=_hash({
            "scope_id": SCOPE_ID+":v94-private",
            "candidate_dss": _digest(
                scalar.candidate_suspended_diameter_m),
            "dss_recompute_count": 1,
        }))

    validate_chkvic_receipt_v91(
        cap, v91_execution, parent.hles_state, handoff)
    hydraulic_receipt = make_hydraulic_turbulence_receipt_v91(
        v91_execution, _v92_state(parent.active_state).base_state,
        parent.hles_state, handoff, cap,
        midpoint_q=trial_h.midpoint_q,
        accepted_q=trial_h.endpoint_q, dt_s=trial_h.dt_s,
        residual_evaluation_count=trial_h.true_residual_evaluation_count,
        momentum_rate_evaluation_count=
            trial_h.true_residual_evaluation_count,
        backend="actual_cuda", actual_implicit_controller=True)
    hles_values = _hles_inputs(
        execution, trial_h.endpoint_q,
        trial_h.accepted_limited_face_water_flux_m2_s,
        closure_chezy_mhalf_s)
    hles_execution = v91_execution.hles_execution
    hles_input = make_hles_input_receipt_v90(
        hles_execution, parent.hles_state,
        accepted_step_receipt=hydraulic_receipt.base_step_receipt,
        current_handoff=handoff, closure_interval_s=trial_h.dt_s,
        accepted_face_normal_velocity_m_s=hles_values[0],
        active_face=hles_values[1], closure_depth_m=hles_values[2],
        closure_chezy_mhalf_s=hles_values[3], active_cell=hles_values[4])
    cp = hles_runtime.cp
    resident = tuple(cp.ascontiguousarray(cp.asarray(
        item, dtype=cp.int32 if index in (1, 4) else cp.float64))
        for index, item in enumerate(hles_values))
    if poll():
        raise InterruptedError("v98 CUDA fine cancelled before HLES")
    hles_trial = hles_runtime.begin_trial(
        parent.hles_state,
        accepted_step_receipt=hydraulic_receipt.base_step_receipt,
        hles_input_receipt=hles_input, current_handoff=handoff,
        accepted_face_normal_velocity_m_s=resident[0],
        active_face=resident[1], closure_depth_m=resident[2],
        closure_chezy_mhalf_s=resident[3], active_cell=resident[4],
        cancelled=poll())
    try:
        numerical = _wrap_v97_state(
            execution, active_state=active,
            hles_state=hles_trial.candidate,
            accepted_chain_sha256=_hash({
                "scope_id": SCOPE_ID+":private-parent",
                "parent": parent.state_sha256,
                "hydraulic": hydraulic.trial_sha256,
                "scalar": scalar.result_sha256,
                "q3": q3.result_sha256,
                "hles": hles_trial.trial_sha256,
                "outer_v98_ledgers_authoritative": True,
            }))
        validate_coupled_wetdry_gpu_state_v97(
            numerical, execution, private_provider)
    finally:
        hles_runtime.rollback(parent.hles_state, hles_trial)
    return numerical, private_provider, q3_state, hles_trial


def build_coupled_amr_fine_substep_state_gpu_v98(
    execution: CoupledWetDryGPUExecutionV97,
    parent: CoupledWetDryGPUStateV97,
    source_provider: ResidentQ3V83ActiveWetDryExnerAdapterV97, *,
    internal_trace: CoupledAMRInternalTraceGPUV98,
    scalar_trace: CoupledAMRScalarTraceGPUV98,
    substep_index: int,
    hydraulic_runtime: ResidentAMRInterfaceFwaveGPUV98,
    hydraulic_trial: WetDryAMRInterfaceGPUTrialV98,
    scalar_runtime: ResidentAMRScalarExchangeGPUV98,
    scalar_result: AMRScalarExchangeResultGPUV98,
    q3_runtime: ResidentAMRQ3FaceGPUV98,
    q3_result: AMRQ3FaceResultGPUV98,
    current_handoff: CurrentClosureHandoffV90,
    cap_receipt: ChkvicUsedTotalReceiptV91,
    hles_runtime: ResidentHlesProducerLifecycleV90,
    closure_chezy_mhalf_s: object,
    cancelled: bool | Callable[[], bool] = False,
) -> tuple[CoupledAMRFineSubstepStateGPUV98,
           ResidentQ3V83ActiveWetDryExnerAdapterV97]:
    poll = _poll(cancelled)
    validate_coupled_wetdry_gpu_state_v97(
        parent, execution, source_provider)
    if (type(source_provider) is not
            ResidentQ3V83ActiveWetDryExnerAdapterV97 or
            type(substep_index) is not int or substep_index not in (0, 1) or
            hydraulic_trial.substep_index != substep_index or
            scalar_result.substep_index != substep_index):
        raise CoupledAMRFineSubstepGPUV98Error(
            "v98 CUDA fine parent/provider/substep differs")
    validate_amr_scalar_exchange_result_gpu_v98(
        scalar_result, scalar_runtime)
    validate_amr_q3_face_result_gpu_v98(q3_result, q3_runtime)
    if scalar_result.fluid_result.result_sha256 != _fluid_result_token(
            scalar_result.fluid_result):
        raise CoupledAMRFineSubstepGPUV98Error(
            "v98 CUDA fine fluid receipt differs")
    if poll():
        raise InterruptedError("v98 CUDA fine cancelled before rebase")
    hydraulic_geometry = hydraulic_runtime.runtime_receipt.get(
        "geometry_binding_receipt_sha256")
    if (hydraulic_geometry !=
            scalar_result.diffusion_geometry_receipt_sha256 or
            ((hydraulic_geometry is None) !=
             (hydraulic_trial.auxiliary_final_evidence_sha256 is None))):
        raise CoupledAMRFineSubstepGPUV98Error(
            "v98 CUDA fine hydraulic/scalar geometry or auxiliary differs")
    numerical, private_provider, q3_state, hles_trial = \
        _build_private_parent(
            execution, parent, source_provider, hydraulic_trial,
            scalar_result, q3_result, current_handoff, cap_receipt,
            hles_runtime, closure_chezy_mhalf_s, poll)
    integrals = make_private_fine_face_time_integrals_gpu_v98(
        execution, parent, hydraulic_runtime, hydraulic_trial,
        scalar_runtime, scalar_result, q3_runtime, q3_result, numerical)
    scalar_internal = _frozen(
        scalar_result.advective_internal_interface_ledger_kg+
        scalar_result.diffusive_internal_interface_ledger_kg,
        (scalar_result.advective_internal_interface_ledger_kg.shape[0], 2),
        "scalar internal ledger")
    bed_internal = _frozen(
        q3_result.internal_interface_transfer_kg-
            scalar_result.suspended_bed_selection.selected_internal_interface_ledger_kg,
        (execution.fraction_count, 2), "bed internal ledger")
    physical_suspended = _frozen(
        scalar_result.suspended_advective_boundary_ledger_increment_kg+
        scalar_result.suspended_diffusive_boundary_ledger_increment_kg,
        (execution.fraction_count,
         _v92_execution(execution.active_execution).boundary_count),
        "suspended physical ledger")
    fluid = scalar_result.fluid_result
    physical_fluid = _frozen(
        fluid.advective_boundary_ledger+fluid.diffusive_boundary_ledger,
        (2, _v92_execution(execution.active_execution).boundary_count),
        "fluid physical ledger")
    counts = CoupledAMRFineSubstepCountsGPUV98()
    placeholder = CoupledAMRFineSubstepStateGPUV98(
        execution.fingerprint, parent.state_sha256,
        source_provider.state.state_sha256,
        internal_trace.receipt_sha256, scalar_trace.receipt_sha256,
        hydraulic_geometry, substep_index,
        hydraulic_trial.wetdry_trial.dt_s,
        hydraulic_trial.trial_sha256,
        hydraulic_trial.auxiliary_final_evidence_sha256,
        scalar_result, q3_result, integrals,
        scalar_result.transport_stage,
        scalar_result.candidate_suspended_diameter_m,
        q3_state, hles_trial, numerical, scalar_internal, bed_internal,
        physical_suspended, physical_fluid, counts, BACKEND, "")
    result = CoupledAMRFineSubstepStateGPUV98(
        **{**placeholder.__dict__,
           "state_sha256": _hash(_state_payload(placeholder))})
    validate_coupled_amr_fine_substep_state_gpu_v98(
        result, execution, parent, source_provider, private_provider)
    if poll():
        raise InterruptedError("v98 CUDA fine cancelled before private return")
    return result, private_provider


def validate_coupled_amr_fine_substep_state_gpu_v98(
    value: CoupledAMRFineSubstepStateGPUV98,
    execution: CoupledWetDryGPUExecutionV97,
    parent: CoupledWetDryGPUStateV97,
    source_provider: ResidentQ3V83ActiveWetDryExnerAdapterV97,
    private_provider: ResidentQ3V83ActiveWetDryExnerAdapterV97,
) -> None:
    validate_coupled_wetdry_gpu_state_v97(
        parent, execution, source_provider)
    validate_coupled_wetdry_gpu_state_v97(
        value.numerical_parent, execution, private_provider)
    validate_coupled_face_time_integrals_gpu_v98(
        value.face_time_integrals, execution)
    expected_bed = (
        value.scalar_exchange.final_bed_mass_kg_m2+
        value.q3_interface_result.cell_delta_kg_m2)
    if (type(value) is not CoupledAMRFineSubstepStateGPUV98 or
            type(private_provider) is not
                ResidentQ3V83ActiveWetDryExnerAdapterV97 or
            value.execution_fingerprint != execution.fingerprint or
            value.parent_state_sha256 != parent.state_sha256 or
            value.source_provider_state_sha256 !=
                source_provider.state.state_sha256 or
            value.geometry_binding_receipt_sha256 !=
                value.scalar_exchange.diffusion_geometry_receipt_sha256 or
            ((value.geometry_binding_receipt_sha256 is None) !=
             (value.hydraulic_auxiliary_evidence_sha256 is None)) or
            (value.hydraulic_auxiliary_evidence_sha256 is not None and
             (type(value.hydraulic_auxiliary_evidence_sha256) is not str or
              len(value.hydraulic_auxiliary_evidence_sha256) != 64 or
              any(character not in "0123456789abcdef" for character in
                  value.hydraulic_auxiliary_evidence_sha256))) or
            value.counts != CoupledAMRFineSubstepCountsGPUV98() or
            value.backend != BACKEND or
            value.state_sha256 != _hash(_state_payload(value)) or
            value.numerical_parent.accepted_updates !=
                parent.accepted_updates+1 or
            value.numerical_parent.time_s != parent.time_s+value.dt_s or
            value.q3_state.state_sha256 != private_provider.state.state_sha256 or
            value.face_time_integrals.parent_state_sha256 !=
                parent.state_sha256 or
            value.face_time_integrals.candidate_state_sha256 !=
                value.numerical_parent.state_sha256 or
            value.face_time_integrals.source_kind != SOURCE_PRIVATE_FINE or
            not np.array_equal(
                value.numerical_parent.active_state.base_state.
                bed_mass_kg_m2, expected_bed) or
            not np.array_equal(
                _v92_state(value.numerical_parent.active_state).
                suspended_state.suspended_mass_kg_m2,
                value.scalar_exchange.final_suspended_mass_kg_m2) or
            not np.array_equal(
                value.scalar_internal_interface_ledger_kg,
                value.scalar_exchange.
                    advective_internal_interface_ledger_kg+
                value.scalar_exchange.
                    diffusive_internal_interface_ledger_kg) or
            not np.array_equal(
                value.bed_internal_interface_ledger_kg,
                value.q3_interface_result.internal_interface_transfer_kg-
                    value.scalar_exchange.suspended_bed_selection.selected_internal_interface_ledger_kg)):
        raise CoupledAMRFineSubstepGPUV98Error(
            "v98 CUDA fine state identity differs")


__all__ = [
    "BACKEND", "CoupledAMRFineSubstepCountsGPUV98",
    "CoupledAMRFineSubstepGPUV98Error",
    "CoupledAMRFineSubstepStateGPUV98",
    "CoupledFaceTimeIntegralsGPUV98", "SCOPE_ID",
    "build_coupled_amr_fine_substep_state_gpu_v98",
    "make_private_fine_face_time_integrals_gpu_v98",
    "validate_coupled_amr_fine_substep_state_gpu_v98",
    "validate_coupled_face_time_integrals_gpu_v98",
]
