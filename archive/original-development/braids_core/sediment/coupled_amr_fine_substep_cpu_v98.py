"""Atomic private CPU v98 fine substep with real AMR interface operators.

The historical v97 accepted schemas have no internal-interface ledger.  This
owner therefore runs each physical producer exactly once, records its signed
coarse/fine transfer separately, then creates a *private numerical* v97 parent
whose local legacy ledger windows start at the corrected images.  Scientific
history remains in this v98 state; no internal transfer is relabelled as a
physical boundary flux and no public clock/provider is mutated.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Callable, Mapping

import numpy as np

from . import active_suspended_control_v94 as _v94
from . import coupled_amr_q3_trace_cpu_v98 as _q3_trace
from . import coupled_amr_register_cpu_v98 as _register
from . import transported_fluid_settling_v92 as _v92
from .active_suspended_control_v94 import (
    _candidate_diameter_v94,
    _wrap_state as _wrap_v94_state,
)
from .coupled_amr_fwave_native_cpu_v98 import (
    NativeAMRInterfaceFwaveRuntimeV98,
)
from .coupled_amr_q3_trace_cpu_v98 import (
    CoupledAMRQ3TraceCPUV98,
)
from .coupled_amr_register_cpu_v98 import (
    CoupledFaceTimeIntegralsCPUV98,
    validate_coupled_face_time_integrals_cpu_v98,
)
from .coupled_amr_scalar_trace_cpu_v98 import (
    CoupledAMRScalarTraceCPUV98,
    validate_coupled_amr_scalar_trace_cpu_v98,
)
from .coupled_amr_trace_cpu_v98 import (
    CoupledAMRInternalTraceCPUV98,
    validate_coupled_amr_internal_trace_structure_cpu_v98,
)
from .coupled_amr_turbulence_trace_cpu_v98 import (
    CoupledAMRTurbulenceTraceCPUV98,
    validate_coupled_amr_turbulence_trace_structure_cpu_v98,
)
from .coupled_amr_wetdry_cpu_v98 import (
    WetDryAMRInterfaceTrialV98,
    validate_wetdry_amr_interface_trial_v98,
)
from .coupled_wetdry_owner_cpu_v97 import (
    CoupledWetDryCPUExecutionV97,
    CoupledWetDryCPUStateV97,
    MAX_ACCEPTED_UPDATES,
    _accepted_density,
    _bed_from_mass,
    _closed_scalar_inputs,
    _hles_inputs,
    _v92_execution,
    _v92_state,
    _wrap_state as _wrap_v97_state,
    execution_document_coupled_wetdry_v97,
    validate_coupled_wetdry_cpu_state_v97,
)
from .direct_vector_wetdry_amr_live_v71 import (
    FACE_CLOSED,
    FACE_INTERNAL,
    FACE_OWNER_OUTLET,
)
from .recursive_surface_feedback_v114 import provider_surface_fraction_v114
from .factcr_native_cpu_v97 import (
    NativeFactCrEvidenceV97,
    NativeFactCrRuntimeV97,
)
from .hles_native_cpu_v97 import NativeHlesCandidateV97, NativeHlesRuntimeV97
from .hles_producer_v90 import (
    CurrentClosureHandoffV90,
    HlesProducerTrialV90,
    make_hles_input_receipt_v90,
    validate_current_closure_handoff_v90,
)
from .multifraction_suspended_lifecycle_v78 import (
    resolve_multifraction_suspended_state_v78,
)
from .q3_face_interface_native_cpu_v98 import (
    NativeQ3FaceInterfaceResultV98,
    NativeQ3FaceInterfaceRuntimeV98,
)
from .q3_mixed_route_exner_adapter_v93 import (
    resolve_cpu_q3_v84_mixed_state_v93,
)
from .q3_wetdry_adapter_cpu_v97 import (
    CPUQ3V83ActiveWetDryExnerAdapterV97,
)
from .scalar_boundary_interface_native_cpu_v98 import (
    NativeScalarBoundaryInterfaceRuntimeV98,
)
from .scalar_diffusion_distance_native_cpu_v98 import (
    NativeScalarDiffusionDistanceRuntimeV98,
)
from .scalar_fct_interface_native_cpu_v98 import (
    NativeScalarFCTInterfaceRuntimeV98,
)
from .settling_exchange_native_cpu_v97 import (
    NativeSettlingExchangeRuntimeV97,
)
from .shared_turbulence_consumers_v91 import (
    ChkvicUsedTotalReceiptV91,
    _wrap_state as _wrap_v91_state,
    make_hydraulic_turbulence_receipt_v91,
    validate_hydraulic_turbulence_receipt_v91,
    validate_chkvic_receipt_v91,
)
from .suspended_exner_owner_v93 import _wrap_state as _wrap_v93_state
from .suspended_support_closures_v82 import (
    resolve_suspended_support_execution_v82,
)
from .transported_fluid_amr_scalar_cpu_v98 import (
    AMRScalarStageResultV98,
    suspended_dynamic_trial_amr_scalar_cpu_v98,
    validate_amr_scalar_stage_result_v98,
)
from .wetdry_fwave_native_cpu_v97 import WetDryMidpointTrialV97


SCHEMA = 1
SCOPE_ID = "braids-q4-coupled-amr-fine-substep-cpu-v98-v1"
BACKEND = "native_cpu_private_amr"


class CoupledAMRFineSubstepCPUV98Error(ValueError):
    """The fine substep, internal trace, or private parent differs."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("ascii")


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _digest(*values: object) -> str:
    result = hashlib.sha256()
    for raw in values:
        value = np.ascontiguousarray(raw)
        result.update(value.dtype.str.encode("ascii"))
        result.update(_canonical(list(value.shape)))
        result.update(value.tobytes(order="C"))
    return result.hexdigest()


def _poll(value: bool | Callable[[], bool]) -> Callable[[], bool]:
    if isinstance(value, (bool, np.bool_)):
        return lambda: bool(value)
    if not callable(value):
        raise TypeError("cancelled must be Boolean or callable")
    return value


@dataclass(frozen=True)
class CoupledAMRFineSubstepCountsCPUV98:
    hydraulic_interface_owner_count: int = 1
    scalar_interface_owner_count: int = 1
    factcr_owner_count: int = 1
    q3_interface_owner_count: int = 1
    exchange_owner_count: int = 1
    hles_owner_count: int = 1
    internal_interface_ledger_owner_count: int = 1
    physical_boundary_surrogate_count: int = 0
    accepted_state_publication_count: int = 0
    source_provider_mutation_count: int = 0


@dataclass(frozen=True)
class CoupledAMRFineSubstepStateCPUV98:
    execution_fingerprint: str
    parent_state_sha256: str
    source_provider_state_sha256: str
    internal_trace_receipt_sha256: str
    scalar_trace_receipt_sha256: str
    q3_trace_receipt_sha256: str
    turbulence_trace_receipt_sha256: str
    substep_index: int
    dt_s: float
    hydraulic_trial_sha256: str
    scalar_stage: AMRScalarStageResultV98
    factcr_receipt: object
    native_factcr_evidence: NativeFactCrEvidenceV97
    q3_interface_result: NativeQ3FaceInterfaceResultV98
    face_time_integrals: CoupledFaceTimeIntegralsCPUV98
    transport_stage: np.ndarray
    suspended_diameter_m: np.ndarray
    q3_state: object
    hles_trial: HlesProducerTrialV90
    native_hles_evidence: NativeHlesCandidateV97
    numerical_parent: CoupledWetDryCPUStateV97
    scalar_internal_interface_ledger_kg: np.ndarray
    bed_internal_interface_ledger_kg: np.ndarray
    physical_suspended_boundary_increment_kg: np.ndarray
    physical_fluid_boundary_increment: np.ndarray
    counts: CoupledAMRFineSubstepCountsCPUV98
    backend: str
    state_sha256: str


def _state_payload(value: CoupledAMRFineSubstepStateCPUV98) -> dict:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID + ":state",
        "execution": value.execution_fingerprint,
        "parent": value.parent_state_sha256,
        "source_provider": value.source_provider_state_sha256,
        "traces": [
            value.internal_trace_receipt_sha256,
            value.scalar_trace_receipt_sha256,
            value.q3_trace_receipt_sha256,
            value.turbulence_trace_receipt_sha256,
        ],
        "substep_index": value.substep_index,
        "dt_s": value.dt_s,
        "hydraulic_trial": value.hydraulic_trial_sha256,
        "scalar_stage": value.scalar_stage.result_sha256,
        "factcr": value.factcr_receipt.receipt_sha256,
        "native_factcr": value.native_factcr_evidence.output_sha256,
        "q3_interface": value.q3_interface_result.output_sha256,
        "face_time_integrals": value.face_time_integrals.receipt_sha256,
        "q3_state": value.q3_state.state_sha256,
        "hles_trial": value.hles_trial.trial_sha256,
        "native_hles": value.native_hles_evidence.result_sha256,
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
    }


def _grid(execution: CoupledWetDryCPUExecutionV97):
    return (execution.active_execution.base_execution.base_execution.
            base_execution.base_execution.base_execution)


def _produce_factcr(
    execution: CoupledWetDryCPUExecutionV97,
    parent: CoupledWetDryCPUStateV97,
    provider: CPUQ3V83ActiveWetDryExnerAdapterV97,
    q3_inputs: Mapping[str, object],
    accepted_q: np.ndarray,
    settling_velocity_m_s: np.ndarray,
    runtime: NativeFactCrRuntimeV97,
    native_threads: int,
    cancelled: Callable[[], bool],
):
    control = execution.active_execution
    profile = _grid(execution).profile
    v92_parent = _v92_state(parent.active_state)
    frozen = control.frozen_uniform_threshold
    receipt, evidence = runtime.produce(
        control.factcr_execution,
        depth_after_m=accepted_q[0],
        endpoint_momentum_m2_s=accepted_q[1:],
        observation_height_m=q3_inputs["observation_height_m"],
        water_density_kg_m3=v92_parent.water_density_cell_kg_m3,
        gravity_m_s2=np.full(execution.cell_count, profile.gravity_m_s2),
        kinematic_viscosity_m2_s=np.full(
            execution.cell_count,
            frozen.initializer_kinematic_viscosity_m2_s),
        additional_shear_stress_pa=q3_inputs["additional_shear_stress_pa"],
        sediment_d50_m=profile.sediment_d50_m,
        sediment_d90_m=q3_inputs["sediment_d90_m"],
        sediment_density_kg_m3=profile.sediment_density_kg_m3,
        molecular_schmidt=q3_inputs["molecular_schmidt"],
        aks_factor=q3_inputs["aks_factor"],
        maximum_volume_concentration=
            q3_inputs["maximum_volume_concentration"],
        minimum_depth_m=q3_inputs["minimum_depth_m"],
        bed_factor=q3_inputs["bed_factor"],
        suspended_factor=q3_inputs["suspended_factor"],
        settling_velocity_m_s=settling_velocity_m_s,
        roughness_z0_m=q3_inputs["roughness_z0_m"],
        surface_fraction=provider_surface_fraction_v114(provider),
        frozen_uniform_threshold=(
            frozen.dstar, frozen.critical_shields,
            frozen.unfactored_critical_shear_pa,
            frozen.effective_critical_shear_pa),
        native_threads=native_threads, cancelled=cancelled)
    _v94.validate_factcr_producer_receipt_v83(
        receipt, control.factcr_execution)
    if (type(evidence) is not NativeFactCrEvidenceV97 or
            evidence.runtime_sha256 != runtime.receipt["receipt_sha256"] or
            evidence.v83_receipt_sha256 != receipt.receipt_sha256 or
            evidence.native_threads != native_threads):
        raise CoupledAMRFineSubstepCPUV98Error(
            "v98 fine FactCr evidence differs")
    return receipt, evidence


def _q3_interface(
    execution: CoupledWetDryCPUExecutionV97,
    provider: CPUQ3V83ActiveWetDryExnerAdapterV97,
    producer, q3_trace: CoupledAMRQ3TraceCPUV98,
    scalar_stage: AMRScalarStageResultV98,
    hydraulic: WetDryMidpointTrialV97,
    runtime: NativeQ3FaceInterfaceRuntimeV98,
    substep_index: int, native_threads: int,
    cancelled: Callable[[], bool],
) -> NativeQ3FaceInterfaceResultV98:
    topology = provider.binding.topology
    q = np.asarray(hydraulic.endpoint_q)
    wet = q[0] > _grid(execution).dry_tolerance_m
    internal = topology.neighbour >= 0
    safe = np.maximum(topology.neighbour, 0)
    face_active = wet[topology.owner] & (~internal | wet[safe])
    cell_vector = np.array(
        producer.bed_capacity_kg_m_s, dtype=np.float64, order="C", copy=True)
    cell_vector[:, :, ~wet] = 0.0
    mapping = np.asarray(provider.binding.family_boundary_faces)
    role, _ = provider._dynamic_roles(
        hydraulic_boundary=
            hydraulic.accepted_limited_face_water_flux_m2_s[mapping],
        cell_vector=cell_vector)
    role = np.array(role, dtype=np.int32, order="C", copy=True)
    role[~face_active] = FACE_CLOSED
    selected = np.asarray(q3_trace.interface_faces, dtype=np.int32)
    if np.any(~face_active[selected]):
        raise CoupledAMRFineSubstepCPUV98Error(
            "v98 fine Q3 trace face is dry/inactive")
    role[selected] = FACE_CLOSED
    ng, nc, nf = (execution.fraction_count, execution.cell_count,
                  execution.face_count)
    taurat = np.ascontiguousarray(producer.producer_diagnostics[2])
    stress = np.zeros((ng, nf), dtype=np.float64)
    mask = role == FACE_INTERNAL
    stress[:, mask] = 0.5*(
        taurat[:, topology.owner[mask]]+
        taurat[:, topology.neighbour[mask]])
    mask = role == FACE_OWNER_OUTLET
    stress[:, mask] = taurat[:, topology.owner[mask]]
    neutral = np.ones((ng, nf), dtype=np.float64)
    hiding = np.ones((ng, nc), dtype=np.float64)
    result = runtime.evaluate(
        _grid(execution), provider.binding.bed_slope_execution,
        cell_vector, provider.state.authoritative_bed_mass_kg_m2, role,
        downhill=np.ascontiguousarray(provider.binding.downhill),
        stress=stress, depth=neutral,
        d50=np.asarray(_grid(execution).profile.sediment_d50_m),
        mean_diameter=neutral, hiding=hiding,
        interface_faces=selected,
        # The first axis is the two horizontal vector components, not the
        # substep axis.  Delft's coarse v83 producer is held lagged over both
        # fine substeps and the typed trace already repeats it per fine face.
        interface_vector_kg_m_s=q3_trace.interface_vector_kg_m_s,
        interface_hiding=q3_trace.interface_hiding,
        interface_availability=q3_trace.interface_availability,
        dt_s=hydraulic.dt_s, native_threads=native_threads,
        cancelled=cancelled)
    if (type(result) is not NativeQ3FaceInterfaceResultV98 or
            result.internal_interface_owner_count != 1 or
            result.physical_boundary_surrogate_count != 0 or
            np.any(scalar_stage.final_bed_mass_kg_m2+
                   result.cell_delta_kg_m2 < 0.0)):
        raise CoupledAMRFineSubstepCPUV98Error(
            "v98 fine Q3 interface result differs or overdraws bed")
    return result


def _build_private_parent(
    execution: CoupledWetDryCPUExecutionV97,
    parent: CoupledWetDryCPUStateV97,
    source_provider: CPUQ3V83ActiveWetDryExnerAdapterV97,
    scalar: AMRScalarStageResultV98,
    q3: NativeQ3FaceInterfaceResultV98,
    candidate_dss: np.ndarray,
    hydraulic: WetDryMidpointTrialV97,
    hydraulic_receipt,
    handoff: CurrentClosureHandoffV90,
    hles_runtime: NativeHlesRuntimeV97,
    closure_chezy_mhalf_s: object,
    native_threads: int,
    cancelled: Callable[[], bool],
):
    """Create a private exact-v97 numerical image at corrected fine values."""
    v92_execution = _v92_execution(execution.active_execution)
    v91_execution = v92_execution.base_execution
    v78_execution = v91_execution.base_execution
    grid = _grid(execution)
    area = np.asarray(grid.cell_area_m2)
    bed = np.ascontiguousarray(
        scalar.final_bed_mass_kg_m2+q3.cell_delta_kg_m2)
    suspended = np.ascontiguousarray(scalar.final_suspended_mass_kg_m2)
    water_reference = np.sum(suspended*area[None, :], axis=1)
    bed_reference = np.sum(bed*area[None, :], axis=1)
    v78 = resolve_multifraction_suspended_state_v78(
        v78_execution, suspended_mass_kg_m2=suspended,
        bed_mass_kg_m2=bed,
        boundary_last_outflow_concentration_kg_m3=
            scalar.boundary_last_outflow_concentration_kg_m3,
        boundary_return_clock_s=scalar.boundary_return_clock_s,
        advective_boundary_ledger_kg=np.zeros(
            (execution.fraction_count, v78_execution.base_execution.
             boundary_count)),
        diffusive_boundary_ledger_kg=np.zeros(
            (execution.fraction_count, v78_execution.base_execution.
             boundary_count)),
        water_exchange_ledger_kg=np.zeros(execution.fraction_count),
        bed_exchange_ledger_kg=np.zeros(execution.fraction_count),
        wetdry_water_ledger_kg=np.zeros(execution.fraction_count),
        wetdry_bed_ledger_kg=np.zeros(execution.fraction_count),
        reference_water_mass_kg=water_reference,
        reference_bed_mass_kg=bed_reference,
        accepted_updates=parent.accepted_updates+1,
        time_s=parent.time_s+hydraulic.dt_s)
    v91 = _wrap_v91_state(
        v91_execution, q=hydraulic.endpoint_q, suspended_state=v78,
        receipt_chain=_hash({
            "scope_id": SCOPE_ID, "level": "v91",
            "parent": parent.state_sha256,
            "hydraulic": hydraulic.trial_sha256,
            "scalar": scalar.result_sha256,
            "q3": q3.output_sha256,
            "internal_ledger_window_rebase": True,
        }))
    fluid = scalar.fluid_result
    scalar_reference = np.array((
        np.sum(fluid.h_temperature_c_m*area, dtype=np.float64),
        np.sum(fluid.h_salinity_ppt_m*area, dtype=np.float64)))
    v92 = _v92._wrap_state(
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
            "scope_id": SCOPE_ID, "level": "v92",
            "parent": parent.state_sha256,
            "scalar": scalar.result_sha256,
            "internal_ledger_window_rebase": True,
        }),
        temperature_cell_c=fluid.temperature_cell_c,
        salinity_cell_ppt=fluid.salinity_cell_ppt)

    provider_parent = source_provider.state
    q3_state = resolve_cpu_q3_v84_mixed_state_v93(
        execution_fingerprint=
            execution.active_execution.factcr_execution.fingerprint,
        binding=source_provider.binding,
        authoritative_bed_mass_kg_m2=bed,
        bulk_density_kg_m3=provider_parent.bulk_density_kg_m3,
        retained_fraction=provider_parent.active_fraction,
        accepted_face_flux_kg_m_s=q3.face_flux_kg_m_s,
        accepted_cell_availability=q3.availability,
        named_boundary_positive_in_kg=(
            provider_parent.named_boundary_positive_in_kg+
            q3.named_boundary_transfer_kg),
        named_boundary_absolute_kg=(
            provider_parent.named_boundary_absolute_kg+
            q3.named_boundary_absolute_kg),
        depletion_roundoff_positive_in_kg=(
            provider_parent.depletion_roundoff_positive_in_kg+
            q3.depletion_roundoff_positive_in_kg),
        accepted_updates=parent.accepted_updates+1,
        time_s=parent.time_s+hydraulic.dt_s)
    private_provider = CPUQ3V83ActiveWetDryExnerAdapterV97(
        outer_execution=execution.active_execution,
        binding=source_provider.binding, state=q3_state,
        native_factcr_runtime=source_provider.native_factcr_runtime,
        native_q3_face_runtime=source_provider.native_q3_face_runtime,
        native_threads=native_threads)
    v93_execution = execution.active_execution.base_execution
    v93 = _wrap_v93_state(
        v93_execution, v92,
        q3_provider_kind=private_provider.provider_kind,
        q3_provider_state_sha256=q3_state.state_sha256,
        bed_mass_kg_m2=bed,
        reference_bed_integral_kg=bed_reference,
        suspended_exchange_ledger_kg=np.zeros(execution.fraction_count),
        suspended_boundary_ledger_kg=np.zeros(
            (execution.fraction_count, v92_execution.boundary_count)),
        excluded_water_exchange_kg=np.zeros(execution.fraction_count),
        q3_named_boundary_positive_in_kg=np.zeros(
            (execution.fraction_count,
             len(v93_execution.q3_boundary_names))),
        q3_named_boundary_absolute_kg=np.zeros(
            (execution.fraction_count,
             len(v93_execution.q3_boundary_names))),
        q3_depletion_roundoff_positive_in_kg=np.zeros(
            execution.fraction_count),
        wetdry_bed_ledger_kg=np.zeros(execution.fraction_count),
        accepted_exner_chain_sha256=_hash({
            "scope_id": SCOPE_ID, "level": "v93",
            "q3": q3.output_sha256, "q3_state": q3_state.state_sha256,
            "internal_ledger_window_rebase": True,
        }))
    active = _wrap_v94_state(
        execution.active_execution, v93,
        suspended_diameter_m=candidate_dss,
        accepted_iopsus_chain_sha256=_hash({
            "scope_id": SCOPE_ID, "level": "v94",
            "carried_candidate_dss": _digest(candidate_dss),
            # This is the ordinary source-ordered v94 candidate produced by
            # this fine update.  Reflux/rebasing must carry it unchanged; it
            # is not a second post-correction constitutive evaluation.
            "normal_v94_candidate_evaluation_count": 1,
            "post_reflux_dss_recompute_count": 0,
        }))

    hles_execution = v92_execution.base_execution.hles_execution
    hles_arrays = _hles_inputs(
        execution, hydraulic.endpoint_q,
        hydraulic.accepted_limited_face_water_flux_m2_s,
        closure_chezy_mhalf_s)
    step = hydraulic_receipt.base_step_receipt
    hles_input = make_hles_input_receipt_v90(
        hles_execution, parent.hles_state,
        accepted_step_receipt=step, current_handoff=handoff,
        closure_interval_s=hydraulic.dt_s,
        accepted_face_normal_velocity_m_s=hles_arrays[0],
        active_face=hles_arrays[1], closure_depth_m=hles_arrays[2],
        closure_chezy_mhalf_s=hles_arrays[3], active_cell=hles_arrays[4])
    hles_trial, hles_evidence = hles_runtime.begin_trial(
        hles_execution, parent.hles_state,
        accepted_step_receipt=step, hles_input_receipt=hles_input,
        current_handoff=handoff,
        accepted_face_normal_velocity_m_s=hles_arrays[0],
        active_face=hles_arrays[1], closure_depth_m=hles_arrays[2],
        closure_chezy_mhalf_s=hles_arrays[3], active_cell=hles_arrays[4],
        native_threads=native_threads, cancelled=cancelled)
    numerical = _wrap_v97_state(
        execution, active_state=active, hles_state=hles_trial.candidate,
        accepted_chain_sha256=_hash({
            "scope_id": SCOPE_ID, "level": "private_v97_parent",
            "parent": parent.state_sha256,
            "hydraulic": hydraulic.trial_sha256,
            "scalar": scalar.result_sha256,
            "q3": q3.output_sha256,
            "hles": hles_trial.trial_sha256,
            "outer_v98_ledgers_authoritative": True,
        }), q3_provider=private_provider)
    validate_coupled_wetdry_cpu_state_v97(
        numerical, execution, private_provider)
    return numerical, private_provider, q3_state, hles_trial, hles_evidence


def _readonly(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    raw = np.asarray(value)
    if (raw.shape != shape or raw.dtype.kind not in "fiu" or
            not np.isfinite(raw).all()):
        raise CoupledAMRFineSubstepCPUV98Error(
            f"v98 fine {name} differs")
    result = np.array(raw, dtype=np.float64, order="C", copy=True)
    result.setflags(write=False)
    return result


def _validate_q3_trace_structure(
    value: CoupledAMRQ3TraceCPUV98,
    internal_trace: CoupledAMRInternalTraceCPUV98,
    execution: CoupledWetDryCPUExecutionV97,
) -> None:
    """Validate the coarse-authenticated Q3 image without replaying it.

    The macro owner performs the full coarse-trial validation before this
    private fine consumer is entered.  This boundary still independently
    binds the immutable receipt, exact fine execution and every array/count.
    """
    ng = execution.fraction_count
    if (type(value) is not CoupledAMRQ3TraceCPUV98 or
            value.receipt_sha256 !=
                _q3_trace._hash(_q3_trace._payload(value)) or
            value.internal_trace_receipt_sha256 !=
                internal_trace.receipt_sha256 or
            value.fine_execution_fingerprint != execution.fingerprint or
            value.coarse_cell != internal_trace.coarse_cell or
            value.interface_faces != internal_trace.fine_faces or
            value.interface_vector_kg_m_s.shape != (2, ng, 2) or
            value.interface_taurat.shape != (ng, 2) or
            value.interface_hiding.shape != (ng, 2) or
            value.interface_availability.shape != (ng, 2) or
            any(array.flags.writeable or not np.isfinite(array).all()
                for array in (
                    value.interface_vector_kg_m_s,
                    value.interface_taurat,
                    value.interface_hiding,
                    value.interface_availability)) or
            np.any(value.interface_taurat < 0.0) or
            np.any(value.interface_hiding < 0.0) or
            np.any(value.interface_availability < 0.0) or
            np.any(value.interface_availability > 1.0) or
            value.constitutive_recompute_count != 0 or
            value.physical_boundary_owner_count != 0 or
            value.accepted_clock_increment_count != 0):
        raise CoupledAMRFineSubstepCPUV98Error(
            "v98 fine Q3 trace structure differs")


def _validate_runtime_identity(
    execution: CoupledWetDryCPUExecutionV97, *,
    hydraulic_runtime: NativeAMRInterfaceFwaveRuntimeV98,
    scalar_fct_runtime: NativeScalarFCTInterfaceRuntimeV98,
    scalar_diffusion_runtime: NativeScalarDiffusionDistanceRuntimeV98,
    scalar_boundary_runtime: NativeScalarBoundaryInterfaceRuntimeV98,
    settling_exchange_runtime: NativeSettlingExchangeRuntimeV97,
    factcr_runtime: NativeFactCrRuntimeV97,
    q3_interface_runtime: NativeQ3FaceInterfaceRuntimeV98,
    hles_runtime: NativeHlesRuntimeV97,
) -> None:
    if (type(hydraulic_runtime) is not NativeAMRInterfaceFwaveRuntimeV98 or
            hydraulic_runtime.execution is not
                execution.hydraulic_execution or
            type(scalar_fct_runtime) is not
                NativeScalarFCTInterfaceRuntimeV98 or
            type(scalar_diffusion_runtime) is not
                NativeScalarDiffusionDistanceRuntimeV98 or
            type(scalar_boundary_runtime) is not
                NativeScalarBoundaryInterfaceRuntimeV98 or
            type(settling_exchange_runtime) is not
                NativeSettlingExchangeRuntimeV97 or
            type(factcr_runtime) is not NativeFactCrRuntimeV97 or
            type(q3_interface_runtime) is not
                NativeQ3FaceInterfaceRuntimeV98 or
            type(hles_runtime) is not NativeHlesRuntimeV97 or
            settling_exchange_runtime.receipt["receipt_sha256"] !=
                execution.settling_exchange_runtime_sha256 or
            factcr_runtime.receipt["receipt_sha256"] !=
                execution.factcr_runtime_sha256 or
            hles_runtime.receipt["receipt_sha256"] !=
                execution.hles_runtime_sha256):
        raise CoupledAMRFineSubstepCPUV98Error(
            "v98 fine native runtime identity differs")


def _extract_fine_face_time_integrals(
    execution: CoupledWetDryCPUExecutionV97,
    parent: CoupledWetDryCPUStateV97,
    hydraulic_trial: WetDryAMRInterfaceTrialV98,
    scalar_stage: AMRScalarStageResultV98,
    q3_result: NativeQ3FaceInterfaceResultV98,
    candidate: CoupledWetDryCPUStateV97,
) -> CoupledFaceTimeIntegralsCPUV98:
    """Build the register image from the actual trace-replaced producers."""
    grid = _grid(execution)
    hydraulic = hydraulic_trial.wetdry_trial
    nf, ng = execution.face_count, execution.fraction_count
    dt = float(hydraulic.dt_s)
    length = np.asarray(grid.face_length_m)
    length_dt = length*dt
    native = hydraulic.native_rate_result
    water = np.asarray(hydraulic.accepted_limited_face_water_flux_m2_s)
    left = np.asarray(native.limited_left_face_flux)
    right = np.asarray(native.limited_right_face_flux)
    if (water.shape != (nf,) or left.shape != (3, nf) or
            right.shape != (3, nf) or
            not np.array_equal(left[0], water) or
            not np.array_equal(right[0], water)):
        raise CoupledAMRFineSubstepCPUV98Error(
            "v98 fine accepted augmented face flux differs")
    midpoint_depth = np.where(
        np.asarray(hydraulic.frozen_wet_mask, dtype=bool),
        np.asarray(hydraulic.midpoint_q[0]), 0.0)
    owner = np.asarray(grid.owner, dtype=np.int64)
    neighbour = np.asarray(grid.neighbour, dtype=np.int64)
    normal = np.asarray(grid.normal)
    pressure_owner = (
        0.5*grid.profile.gravity_m_s2*midpoint_depth[owner]**2
    )[None, :]*normal
    pressure_neighbour = np.zeros((2, nf), dtype=np.float64)
    internal = neighbour >= 0
    pressure_neighbour[:, internal] = (
        0.5*grid.profile.gravity_m_s2*
        midpoint_depth[neighbour[internal]]**2
    )[None, :]*normal[:, internal]
    internal_v91 = np.asarray(
        hydraulic_trial.accepted_internal_turbulence_result.
        face_flux_m4_s2)
    trace_v91 = np.asarray(
        hydraulic_trial.accepted_trace_turbulence_result[
            "face_flux_m4_s2"])
    viscous = internal_v91+trace_v91
    fluid = scalar_stage.fluid_result
    morphological_factor = float(
        execution.active_execution.base_execution.morphological_factor)
    left_integral = left*length_dt[None, :]
    right_integral = right*length_dt[None, :]
    arrays = (
        length, midpoint_depth, water*length_dt,
        left_integral, right_integral,
        pressure_owner*length_dt[None, :],
        pressure_neighbour*length_dt[None, :],
        right_integral[1:]-left_integral[1:], viscous*dt,
        scalar_stage.advective_face_flux_kg_m_s*length[None, :]*dt,
        scalar_stage.diffusive_face_mass_owner_out_kg,
        fluid.advective_face_flux*length[None, :]*dt,
        fluid.diffusive_face_amount,
        q3_result.face_flux_kg_m_s*length[None, :]*dt*
            morphological_factor,
    )
    shapes = ((nf,), (execution.cell_count,), (nf,), (3, nf), (3, nf),
              (2, nf), (2, nf), (2, nf), (2, nf), (ng, nf),
              (ng, nf), (2, nf), (2, nf), (ng, nf))
    names = ("face length", "midpoint depth", "water volume",
             "augmented left", "augmented right", "owner pressure",
             "neighbour pressure", "augmented jump", "viscous momentum",
             "suspended advection", "suspended diffusion",
             "fluid advection", "fluid diffusion", "bedload")
    frozen = tuple(
        _register._frozen(array, shape, name)
        for array, shape, name in zip(arrays, shapes, names))
    source_sha = _hash({
        "scope_id": SCOPE_ID+":fine-face-time-source",
        "hydraulic": hydraulic_trial.trial_sha256,
        "scalar": scalar_stage.result_sha256,
        "q3": q3_result.output_sha256,
        "candidate": candidate.state_sha256,
    })
    placeholder = CoupledFaceTimeIntegralsCPUV98(
        execution.fingerprint, parent.state_sha256, candidate.state_sha256,
        source_sha, native.result_sha256, q3_result.output_sha256,
        parent.accepted_updates, candidate.accepted_updates,
        parent.time_s, candidate.time_s, dt, morphological_factor,
        *frozen, _register.BACKEND, "", scalar_stage.suspended_bed_selection.mode == 1)
    result = CoupledFaceTimeIntegralsCPUV98(
        **{**placeholder.__dict__, "receipt_sha256": _register._hash(
            _register._integral_payload(placeholder))})
    validate_coupled_face_time_integrals_cpu_v98(result, execution)
    return result


def begin_coupled_amr_fine_substep_cpu_v98(
    execution: CoupledWetDryCPUExecutionV97,
    parent: CoupledWetDryCPUStateV97,
    source_provider: CPUQ3V83ActiveWetDryExnerAdapterV97, *,
    internal_trace: CoupledAMRInternalTraceCPUV98,
    scalar_trace: CoupledAMRScalarTraceCPUV98,
    q3_trace: CoupledAMRQ3TraceCPUV98,
    turbulence_trace: CoupledAMRTurbulenceTraceCPUV98,
    substep_index: int,
    hydraulic_trial: WetDryAMRInterfaceTrialV98,
    hydraulic_runtime: NativeAMRInterfaceFwaveRuntimeV98,
    current_handoff: CurrentClosureHandoffV90,
    scalar_fct_runtime: NativeScalarFCTInterfaceRuntimeV98,
    scalar_diffusion_runtime: NativeScalarDiffusionDistanceRuntimeV98,
    scalar_boundary_runtime: NativeScalarBoundaryInterfaceRuntimeV98,
    settling_exchange_runtime: NativeSettlingExchangeRuntimeV97,
    factcr_runtime: NativeFactCrRuntimeV97,
    q3_interface_runtime: NativeQ3FaceInterfaceRuntimeV98,
    hles_runtime: NativeHlesRuntimeV97,
    shear_velocity_m_s: object,
    closure_chezy_mhalf_s: object,
    q3_inputs: Mapping[str, object],
    native_threads: int = 4,
    cancelled: bool | Callable[[], bool] = False,
) -> tuple[CoupledAMRFineSubstepStateCPUV98,
           CPUQ3V83ActiveWetDryExnerAdapterV97]:
    """Compose one private fine substep without publishing a level state."""
    poll = _poll(cancelled)
    execution_document_coupled_wetdry_v97(execution)
    validate_coupled_wetdry_cpu_state_v97(
        parent, execution, source_provider)
    if (type(source_provider) is not
            CPUQ3V83ActiveWetDryExnerAdapterV97 or
            type(substep_index) is not int or substep_index not in (0, 1) or
            type(native_threads) is not int or not 1 <= native_threads <= 256):
        raise CoupledAMRFineSubstepCPUV98Error(
            "v98 fine provider, substep or thread identity differs")
    if parent.accepted_updates >= MAX_ACCEPTED_UPDATES:
        raise CoupledAMRFineSubstepCPUV98Error(
            "v98 fine accepted update limit was reached")
    if poll():
        raise InterruptedError("v98 fine cancelled before validation")
    _validate_runtime_identity(
        execution, hydraulic_runtime=hydraulic_runtime,
        scalar_fct_runtime=scalar_fct_runtime,
        scalar_diffusion_runtime=scalar_diffusion_runtime,
        scalar_boundary_runtime=scalar_boundary_runtime,
        settling_exchange_runtime=settling_exchange_runtime,
        factcr_runtime=factcr_runtime,
        q3_interface_runtime=q3_interface_runtime,
        hles_runtime=hles_runtime)
    validate_coupled_amr_internal_trace_structure_cpu_v98(
        internal_trace, execution)
    if (internal_trace.joined_geometry_receipt_sha256 is None or
            internal_trace.physical_boundary_distance_reuse_count != 0 or
            scalar_diffusion_runtime.internal_trace is not internal_trace):
        raise CoupledAMRFineSubstepCPUV98Error(
            "v98 fine substep requires authenticated donor-centre geometry")
    validate_coupled_amr_scalar_trace_cpu_v98(
        scalar_trace, internal_trace, execution)
    _validate_q3_trace_structure(q3_trace, internal_trace, execution)
    validate_coupled_amr_turbulence_trace_structure_cpu_v98(
        turbulence_trace, execution, internal_trace)
    if (hydraulic_runtime.trace is not internal_trace or
            hydraulic_runtime.turbulence_trace is not turbulence_trace or
            hydraulic_runtime.substep_index != substep_index or
            scalar_trace.interface_faces != internal_trace.fine_faces or
            q3_trace.interface_faces != internal_trace.fine_faces):
        raise CoupledAMRFineSubstepCPUV98Error(
            "v98 fine trace consumer binding differs")

    water, mixture = _accepted_density(execution, parent)
    hydraulic = hydraulic_trial.wetdry_trial
    validate_wetdry_amr_interface_trial_v98(
        hydraulic_trial, execution.hydraulic_execution,
        runtime=hydraulic_runtime, parent_q=parent.q,
        bed_elevation_m=parent.bed_elevation_m,
        accepted_water_density_kg_m3=water,
        accepted_mixture_density_kg_m3=mixture, dt_s=hydraulic.dt_s)
    expected_dt = float(
        internal_trace.substep_time_s[1]-
        internal_trace.substep_time_s[0])
    if (hydraulic.dt_s != expected_dt or
            parent.time_s+0.5*hydraulic.dt_s !=
                float(internal_trace.substep_time_s[substep_index])):
        raise CoupledAMRFineSubstepCPUV98Error(
            "v98 fine substep clock differs from coarse trace")

    v92_execution = _v92_execution(execution.active_execution)
    v92_parent = _v92_state(parent.active_state)
    v91_execution = v92_execution.base_execution
    validate_current_closure_handoff_v90(
        current_handoff, v91_execution.hles_execution,
        parent.hles_state)
    prepared = hydraulic_runtime.shared_turbulence_prepared
    cap = prepared.cap_receipt
    validate_chkvic_receipt_v91(
        cap, v91_execution, parent.hles_state, current_handoff)
    if (prepared.execution is not v91_execution or
            prepared.native_threads != native_threads or
            hydraulic_trial.accepted_internal_turbulence_result.
                cap_receipt is not cap):
        raise CoupledAMRFineSubstepCPUV98Error(
            "v98 fine hydraulic cap/preparation differs")
    hydraulic_receipt = make_hydraulic_turbulence_receipt_v91(
        v91_execution, v92_parent.base_state, parent.hles_state,
        current_handoff, cap,
        midpoint_q=hydraulic.midpoint_q,
        accepted_q=hydraulic.endpoint_q, dt_s=hydraulic.dt_s,
        residual_evaluation_count=hydraulic.true_residual_evaluation_count,
        momentum_rate_evaluation_count=
            hydraulic.true_residual_evaluation_count,
        backend="cpu_oracle", actual_implicit_controller=False)
    validate_hydraulic_turbulence_receipt_v91(
        hydraulic_receipt, v91_execution, v92_parent.base_state,
        parent.hles_state, current_handoff, cap)
    face_receipt = _v92.make_accepted_scalar_hydraulic_face_receipt_v92(
        v92_execution, v92_parent, hydraulic_receipt,
        accepted_q=hydraulic.endpoint_q,
        accepted_water_flux_m2_s=
            hydraulic.accepted_limited_face_water_flux_m2_s)
    scalar_execution = resolve_suspended_support_execution_v82(
        v92_execution.base_execution.base_execution,
        dicuv_cell_m2_s=cap.capped_total_dicuv_m2_s)

    if np.any(execution.active_execution.iopsus_by_fraction == 1):
        settling = _v92.make_settling_diameter_receipt_v92(
            v92_execution, v92_parent,
            suspended_diameter_m=parent.active_state.suspended_diameter_m,
            source_kind=_v92.SETTLING_DIAMETER_ACTIVE_IOPSUS_V94,
            source_state_sha256=parent.active_state.state_sha256)
    else:
        settling = _v92.make_settling_diameter_receipt_v92(
            v92_execution, v92_parent,
            suspended_diameter_m=parent.active_state.suspended_diameter_m)
    required_q3 = {
        "observation_height_m", "additional_shear_stress_pa",
        "sediment_d90_m", "molecular_schmidt", "aks_factor",
        "maximum_volume_concentration", "minimum_depth_m", "bed_factor",
        "suspended_factor", "roughness_z0_m",
    }
    if not isinstance(q3_inputs, Mapping) or set(q3_inputs) != required_q3:
        raise CoupledAMRFineSubstepCPUV98Error(
            "v98 fine Q3 producer input inventory differs")
    fluid_left, fluid_right, prescribed_fluid, scalar_inputs = \
        _closed_scalar_inputs(
            execution, parent, hydraulic.endpoint_q, shear_velocity_m_s)
    scalar_inputs["surface_fraction"] = provider_surface_fraction_v114(
        source_provider)
    produced: dict[str, object] = {}

    def capacity_resolver(**kwargs):
        if produced:
            raise CoupledAMRFineSubstepCPUV98Error(
                "v98 fine FactCr producer was invoked more than once")
        producer, evidence = _produce_factcr(
            execution, parent, source_provider, q3_inputs,
            np.asarray(kwargs["accepted_q"]),
            np.asarray(kwargs["settling_velocity_m_s"]), factcr_runtime,
            native_threads, poll)
        capacity = _v92.make_suspended_capacity_receipt_v92(
            v92_execution, v92_parent,
            accepted_q=kwargs["accepted_q"],
            settling_diameter_receipt=settling,
            suspended_capacity_kg_m_s=
                producer.lifecycle_capacity_kg_m_s,
            source_kind=_v92.SUSPENDED_CAPACITY_ACTIVE_V83_V94,
            source_receipt_sha256=producer.receipt_sha256)
        produced.update(producer=producer, evidence=evidence,
                        capacity=capacity)
        return capacity

    source_checkpoint = source_provider.checkpoint()
    scalar_stage = suspended_dynamic_trial_amr_scalar_cpu_v98(
        v92_execution, v92_parent, scalar_execution, hydraulic_receipt,
        hydraulic_face_receipt=face_receipt,
        settling_diameter_receipt=settling,
        depth_before_m=parent.q[0],
        depth_after_m=hydraulic.endpoint_q[0],
        endpoint_momentum_m2_s=hydraulic.endpoint_q[1:],
        reconstructed_left_fluid=fluid_left,
        reconstructed_right_fluid=fluid_right,
        prescribed_boundary_fluid=prescribed_fluid,
        dt_s=hydraulic.dt_s, scalar_inputs=scalar_inputs,
        fct_runtime=scalar_fct_runtime,
        diffusion_runtime=scalar_diffusion_runtime,
        native_threads=native_threads,
        settling_exchange_runtime=settling_exchange_runtime,
        scalar_boundary_runtime=scalar_boundary_runtime,
        scalar_trace=scalar_trace, internal_trace=internal_trace,
        turbulence_trace=turbulence_trace, fine_cap_receipt=cap,
        substep_index=substep_index,
        suspended_capacity_resolver=capacity_resolver,
        suspended_bed_execution=execution.active_execution.base_execution,
        cancelled=poll)
    validate_amr_scalar_stage_result_v98(
        scalar_stage, execution=v92_execution, scalar_trace=scalar_trace,
        fct_runtime=scalar_fct_runtime,
        diffusion_runtime=scalar_diffusion_runtime,
        substep_index=substep_index, native_threads=native_threads)
    if (set(produced) != {"producer", "evidence", "capacity"} or
            scalar_stage.suspended_capacity_receipt is not
                produced["capacity"]):
        raise CoupledAMRFineSubstepCPUV98Error(
            "v98 fine FactCr/capacity lifecycle differs")
    producer = produced["producer"]
    stage, candidate_dss = _candidate_diameter_v94(
        execution.active_execution, parent.active_state, producer)
    q3_result = _q3_interface(
        execution, source_provider, producer, q3_trace, scalar_stage,
        hydraulic, q3_interface_runtime, substep_index, native_threads,
        poll)
    numerical, private_provider, q3_state, hles_trial, hles_evidence = \
        _build_private_parent(
            execution, parent, source_provider, scalar_stage, q3_result,
            candidate_dss, hydraulic, hydraulic_receipt, current_handoff,
            hles_runtime, closure_chezy_mhalf_s, native_threads, poll)
    face_time_integrals = _extract_fine_face_time_integrals(
        execution, parent, hydraulic_trial, scalar_stage, q3_result,
        numerical)
    if source_provider.checkpoint() != source_checkpoint:
        raise CoupledAMRFineSubstepCPUV98Error(
            "v98 fine mutated the source provider")
    ng = execution.fraction_count
    nb = v92_execution.boundary_count
    lanes = scalar_stage.internal_interface_owner_ledger_kg.shape[0]
    scalar_internal = _readonly(
        scalar_stage.internal_interface_owner_ledger_kg,
        (lanes, len(internal_trace.fine_faces)),
        "scalar internal-interface ledger")
    bed_internal = _readonly(
        q3_result.internal_interface_transfer_kg-
            scalar_stage.suspended_bed_selection.selected_internal_interface_ledger_kg,
        (ng, len(internal_trace.fine_faces)),
        "bed internal-interface ledger")
    suspended_physical = _readonly(
        scalar_stage.advective_boundary_ledger_increment_kg+
        scalar_stage.diffusive_boundary_ledger_increment_kg,
        (ng, nb), "suspended physical-boundary increment")
    fluid_physical = _readonly(
        scalar_stage.fluid_result.advective_boundary_ledger+
        scalar_stage.fluid_result.diffusive_boundary_ledger,
        (2, nb), "fluid physical-boundary increment")
    counts = CoupledAMRFineSubstepCountsCPUV98()
    placeholder = CoupledAMRFineSubstepStateCPUV98(
        execution.fingerprint, parent.state_sha256,
        source_provider.state.state_sha256, internal_trace.receipt_sha256,
        scalar_trace.receipt_sha256, q3_trace.receipt_sha256,
        turbulence_trace.receipt_sha256, substep_index, hydraulic.dt_s,
        hydraulic_trial.trial_sha256, scalar_stage, producer,
        produced["evidence"], q3_result, face_time_integrals,
        stage, candidate_dss, q3_state,
        hles_trial, hles_evidence, numerical, scalar_internal, bed_internal,
        suspended_physical, fluid_physical, counts, BACKEND, "")
    result = CoupledAMRFineSubstepStateCPUV98(
        **{**placeholder.__dict__,
           "state_sha256": _hash(_state_payload(placeholder))})
    validate_coupled_amr_fine_substep_state_v98(
        result, execution, parent, source_provider, internal_trace,
        scalar_trace, q3_trace, turbulence_trace, private_provider)
    if poll():
        raise InterruptedError("v98 fine cancelled before private return")
    return result, private_provider


def validate_coupled_amr_fine_substep_state_v98(
    value: CoupledAMRFineSubstepStateCPUV98,
    execution: CoupledWetDryCPUExecutionV97,
    parent: CoupledWetDryCPUStateV97,
    source_provider: CPUQ3V83ActiveWetDryExnerAdapterV97,
    internal_trace: CoupledAMRInternalTraceCPUV98,
    scalar_trace: CoupledAMRScalarTraceCPUV98,
    q3_trace: CoupledAMRQ3TraceCPUV98,
    turbulence_trace: CoupledAMRTurbulenceTraceCPUV98,
    private_provider: CPUQ3V83ActiveWetDryExnerAdapterV97,
) -> None:
    validate_coupled_wetdry_cpu_state_v97(
        parent, execution, source_provider)
    validate_coupled_amr_internal_trace_structure_cpu_v98(
        internal_trace, execution)
    validate_coupled_amr_scalar_trace_cpu_v98(
        scalar_trace, internal_trace, execution)
    _validate_q3_trace_structure(q3_trace, internal_trace, execution)
    validate_coupled_amr_turbulence_trace_structure_cpu_v98(
        turbulence_trace, execution, internal_trace)
    if (type(value) is not CoupledAMRFineSubstepStateCPUV98 or
            type(private_provider) is not
                CPUQ3V83ActiveWetDryExnerAdapterV97 or
            value.execution_fingerprint != execution.fingerprint or
            value.parent_state_sha256 != parent.state_sha256 or
            value.source_provider_state_sha256 !=
                source_provider.state.state_sha256 or
            value.internal_trace_receipt_sha256 !=
                internal_trace.receipt_sha256 or
            value.scalar_trace_receipt_sha256 !=
                scalar_trace.receipt_sha256 or
            value.q3_trace_receipt_sha256 != q3_trace.receipt_sha256 or
            value.turbulence_trace_receipt_sha256 !=
                turbulence_trace.receipt_sha256 or
            value.substep_index not in (0, 1) or value.dt_s <= 0.0 or
            value.counts != CoupledAMRFineSubstepCountsCPUV98() or
            value.backend != BACKEND or
            value.state_sha256 != _hash(_state_payload(value)) or
            value.numerical_parent.accepted_updates !=
                parent.accepted_updates+1 or
            value.numerical_parent.time_s != parent.time_s+value.dt_s or
            value.q3_state.state_sha256 != private_provider.state.state_sha256 or
            private_provider.accepted_updates !=
                value.numerical_parent.accepted_updates or
            private_provider.time_s != value.numerical_parent.time_s or
            value.factcr_receipt.receipt_sha256 !=
                value.scalar_stage.suspended_capacity_receipt.
                source_receipt_sha256 or
            value.native_factcr_evidence.v83_receipt_sha256 !=
                value.factcr_receipt.receipt_sha256 or
            value.q3_interface_result.internal_interface_owner_count != 1 or
            value.q3_interface_result.physical_boundary_surrogate_count != 0 or
            value.face_time_integrals.candidate_state_sha256 !=
                value.numerical_parent.state_sha256 or
            value.face_time_integrals.parent_state_sha256 !=
                parent.state_sha256 or
            value.face_time_integrals.dt_s != value.dt_s or
            not np.array_equal(
                value.scalar_internal_interface_ledger_kg,
                value.scalar_stage.internal_interface_owner_ledger_kg) or
            not np.array_equal(
                value.bed_internal_interface_ledger_kg,
                value.q3_interface_result.internal_interface_transfer_kg-
                    value.scalar_stage.suspended_bed_selection.selected_internal_interface_ledger_kg) or
            not np.array_equal(
                value.numerical_parent.active_state.base_state.
                    bed_mass_kg_m2,
                value.scalar_stage.final_bed_mass_kg_m2+
                    value.q3_interface_result.cell_delta_kg_m2) or
            not np.array_equal(
                _v92_state(value.numerical_parent.active_state).
                    suspended_state.suspended_mass_kg_m2,
                value.scalar_stage.final_suspended_mass_kg_m2) or
            not np.array_equal(
                value.numerical_parent.active_state.suspended_diameter_m,
                value.suspended_diameter_m) or
            any(array.flags.writeable or not np.isfinite(array).all()
                for array in (
                    value.transport_stage, value.suspended_diameter_m,
                    value.scalar_internal_interface_ledger_kg,
                    value.bed_internal_interface_ledger_kg,
                    value.physical_suspended_boundary_increment_kg,
                    value.physical_fluid_boundary_increment))):
        raise CoupledAMRFineSubstepCPUV98Error(
            "v98 fine state identity, owner, or corrected image differs")
    validate_coupled_face_time_integrals_cpu_v98(
        value.face_time_integrals, execution)
    validate_coupled_wetdry_cpu_state_v97(
        value.numerical_parent, execution, private_provider)


__all__ = [
    "BACKEND", "SCHEMA", "SCOPE_ID",
    "CoupledAMRFineSubstepCPUV98Error",
    "CoupledAMRFineSubstepCountsCPUV98",
    "CoupledAMRFineSubstepStateCPUV98",
    "begin_coupled_amr_fine_substep_cpu_v98",
    "validate_coupled_amr_fine_substep_state_v98",
]
