"""Atomic actual-CUDA joined 2:1 leaf macro with one reflux publication.

The coarse and two fine updates are provisional source evidence.  One CUDA
corrected-image owner applies the internal register, derives bed/EOS images,
and constructs private numerical parents for the next macro.  The legacy v97
state is never asked to represent a nonzero internal-interface ledger.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import json
from typing import Callable, Mapping, Sequence

import numpy as np

from braids_core.sediment.active_suspended_control_v94 import (
    _wrap_state as _wrap_v94_state,
)
from braids_core.sediment.hles_producer_v90 import (
    make_hles_input_receipt_v90,
)
from braids_core.sediment.multifraction_suspended_lifecycle_v78 import (
    resolve_multifraction_suspended_state_v78,
)
from braids_core.sediment.shared_turbulence_consumers_v91 import (
    _wrap_state as _wrap_v91_state,
)
from braids_core.sediment.suspended_exner_owner_v93 import (
    _wrap_state as _wrap_v93_state,
)
from braids_core.sediment.transported_fluid_settling_v92 import (
    MODE_CONSTANT, MODE_TRANSPORTED, _wrap_state as _wrap_v92_state,
)
from .active_suspended_wetdry_v97 import (
    ResidentActiveSuspendedWetDryLifecycleV97,
    ResidentQ3V83ActiveWetDryExnerAdapterV97,
)
from .coupled_amr_corrected_coarse_v98 import (
    ResidentAMRCorrectedCoarseImagesV98,
)
from .coupled_amr_fine_substep_v98 import (
    CoupledAMRFineSubstepStateGPUV98,
    validate_coupled_amr_fine_substep_state_gpu_v98,
)
from .coupled_amr_register_v98 import (
    CoupledAMRCoarseCorrectionGPUV98,
    CoupledAMRInterfaceMapGPUV98,
    CoupledFaceTimeIntegralsGPUV98,
    validate_coupled_amr_coarse_correction_gpu_v98,
)
from .coupled_amr_geometry_v98 import (
    CoupledAMRGeometryBindingGPUV98,
)
from .coupled_wetdry_v97 import (
    CoupledWetDryGPUExecutionV97, CoupledWetDryGPUStateV97,
    CoupledWetDryGPUTrialV97, ResidentCoupledWetDryLifecycleV97,
    _grid, _hles_inputs, _v91_execution, _v92_execution, _v92_state,
    _wrap_state as _wrap_v97_state,
    validate_coupled_wetdry_gpu_state_v97,
)
from .hles_producer_v90 import ResidentHlesProducerLifecycleV90
from .q3_exner_adapter_v93 import (
    ResidentQ3V84MixedStateV93,
    resolve_resident_q3_v84_mixed_state_v93,
)
from .transported_fluid_lifecycle_v92 import (
    ResidentTransportedFluidSettlingLifecycleV92,
)


SCHEMA = 1
SCOPE_ID = "braids.q4.coupled_amr.joined_leaf.actual_cuda.v98"
BACKEND = "actual_cuda_atomic_joined_leaf"
MAX_MACRO_UPDATES = 2


class CoupledAMRJoinedLeafGPUV98Error(ValueError):
    """The joined CUDA macro state, receipt, or transaction differs."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          allow_nan=False).encode("ascii")
    except (TypeError, ValueError, OverflowError) as error:
        raise CoupledAMRJoinedLeafGPUV98Error(
            "v98 CUDA joined value is not canonical JSON") from error


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _digest(*values: object) -> str:
    result = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(value)
        result.update(array.dtype.str.encode("ascii"))
        result.update(_canonical(list(array.shape)))
        result.update(array.tobytes(order="C"))
    return result.hexdigest()


def _sha(value: object, name: str) -> str:
    if (not isinstance(value, str) or len(value) != 64 or
            any(character not in "0123456789abcdef" for character in value)):
        raise CoupledAMRJoinedLeafGPUV98Error(
            f"v98 CUDA joined {name} must be a lowercase SHA-256 digest")
    return value


def _poll(value: bool | Callable[[], bool]) -> Callable[[], bool]:
    if type(value) is bool:
        return lambda: value
    if callable(value):
        return lambda: bool(value())
    raise CoupledAMRJoinedLeafGPUV98Error(
        "v98 CUDA joined cancellation must be bool or callable")


def _frozen(value: object, shape: tuple[int, ...], name: str,
            dtype=np.float64) -> np.ndarray:
    raw = np.asarray(value)
    if (raw.shape != shape or raw.dtype.kind not in "fiu" or
            not np.isfinite(raw).all()):
        raise CoupledAMRJoinedLeafGPUV98Error(
            f"v98 CUDA joined {name} differs")
    result = np.array(raw, dtype=dtype, order="C", copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class CoupledAMRCorrectedCoarseCountsGPUV98:
    internal_interface_ledger_application_count: int = 1
    corrected_image_cuda_application_count: int = 1
    reflux_dry_cleanup_owner_count: int = 1
    bed_geometry_derivation_count: int = 1
    fluid_eos_derivation_count: int = 1
    mixture_eos_derivation_count: int = 1
    corrected_hles_owner_count: int = 1
    dss_recompute_count: int = 0
    hydraulic_recompute_count: int = 0
    scalar_recompute_count: int = 0
    exchange_recompute_count: int = 0
    q3_recompute_count: int = 0
    physical_boundary_ledger_application_count: int = 0
    provisional_hles_evaluation_count: int = 1
    discarded_provisional_hles_count: int = 1
    corrected_hles_evaluation_count: int = 1
    corrected_image_host_input_upload_count: int = 1
    corrected_image_host_readback_count: int = 1
    corrected_hles_input_host_to_device_count: int = 1


@dataclass(frozen=True)
class CoupledAMRCorrectedCoarseGPUStateV98:
    execution_fingerprint: str
    provisional_source_state_sha256: str
    provisional_source_trial_sha256: str
    correction_receipt_sha256: str
    internal_interface_ledger_sha256: str
    corrected_image_runtime_sha256: str
    corrected_image_receipt_sha256: str
    numerical_parent: CoupledWetDryGPUStateV97
    q3_state: ResidentQ3V84MixedStateV93
    corrected_face_water_flux_m2_s: np.ndarray
    corrected_face_bedload_flux_kg_m_s: np.ndarray
    reflux_dry_transfer_to_bed_kg_m2: np.ndarray
    reflux_dry_momentum_cleanup_m2_s: np.ndarray
    reflux_dry_fluid_cleanup: np.ndarray
    counts: CoupledAMRCorrectedCoarseCountsGPUV98
    backend: str
    state_sha256: str


def _corrected_payload(value: CoupledAMRCorrectedCoarseGPUStateV98) -> dict:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID+":corrected-coarse",
        "execution": value.execution_fingerprint,
        "provisional_source_state": value.provisional_source_state_sha256,
        "provisional_source_trial": value.provisional_source_trial_sha256,
        "correction": value.correction_receipt_sha256,
        "internal_interface_ledger":
            value.internal_interface_ledger_sha256,
        "corrected_image_runtime": value.corrected_image_runtime_sha256,
        "corrected_image_receipt": value.corrected_image_receipt_sha256,
        "numerical_parent": value.numerical_parent.state_sha256,
        "q3_state": value.q3_state.state_sha256,
        "arrays_sha256": _digest(
            value.corrected_face_water_flux_m2_s,
            value.corrected_face_bedload_flux_kg_m_s,
            value.reflux_dry_transfer_to_bed_kg_m2,
            value.reflux_dry_momentum_cleanup_m2_s,
            value.reflux_dry_fluid_cleanup),
        "counts": value.counts.__dict__,
        "backend": value.backend,
        "legacy_v97_internal_ledger_application_count": 0,
    }


@dataclass(frozen=True)
class CoupledAMRJoinedLeafCountsGPUV98:
    coarse_provisional_update_count: int = 1
    private_fine_substep_count: int = 2
    hydraulic_register_application_count: int = 1
    scalar_bed_register_application_count: int = 1
    corrected_coarse_image_owner_count: int = 1
    public_macro_state_publication_count: int = 1
    source_provider_mutation_count: int = 0
    physical_boundary_surrogate_count: int = 0
    covered_parent_accepted_row_count: int = 0


@dataclass(frozen=True)
class CoupledAMRJoinedLeafGPUStateV98:
    coarse_corrected: CoupledAMRCorrectedCoarseGPUStateV98
    fine_numerical_parent: CoupledWetDryGPUStateV97
    fine_q3_state: ResidentQ3V84MixedStateV93
    correction_receipt_sha256: str
    cumulative_water_volume_m3: float
    cumulative_momentum_m4_s: np.ndarray
    cumulative_suspended_mass_kg: np.ndarray
    cumulative_fluid_integral: np.ndarray
    cumulative_bed_mass_kg: np.ndarray
    accepted_macro_updates: int
    time_s: float
    previous_state_sha256: str | None
    counts: CoupledAMRJoinedLeafCountsGPUV98
    accepted_chain_sha256: str
    state_sha256: str

    @property
    def coarse_numerical_parent(self) -> CoupledWetDryGPUStateV97:
        return self.coarse_corrected.numerical_parent


def _joined_payload(value: CoupledAMRJoinedLeafGPUStateV98) -> dict:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID+":state",
        "coarse": value.coarse_corrected.state_sha256,
        "fine": value.fine_numerical_parent.state_sha256,
        "fine_q3": value.fine_q3_state.state_sha256,
        "correction": value.correction_receipt_sha256,
        "cumulative_sha256": _digest(
            np.asarray(value.cumulative_water_volume_m3),
            value.cumulative_momentum_m4_s,
            value.cumulative_suspended_mass_kg,
            value.cumulative_fluid_integral,
            value.cumulative_bed_mass_kg),
        "accepted_macro_updates": value.accepted_macro_updates,
        "time_s": value.time_s,
        "previous_state": value.previous_state_sha256,
        "counts": value.counts.__dict__,
        "accepted_chain": value.accepted_chain_sha256,
    }


@dataclass(frozen=True)
class CoupledAMRJoinedLeafGPUTrialV98:
    coarse_parent_state_sha256: str
    coarse_provisional_trial_sha256: str
    fine_parent_state_sha256: str
    fine1_state_sha256: str
    fine2_state_sha256: str
    correction: CoupledAMRCoarseCorrectionGPUV98
    corrected_coarse: CoupledAMRCorrectedCoarseGPUStateV98
    candidate: CoupledAMRJoinedLeafGPUStateV98
    corrected_coarse_provider: ResidentQ3V83ActiveWetDryExnerAdapterV97
    final_fine_provider: ResidentQ3V83ActiveWetDryExnerAdapterV97
    source_transaction_close_count: int
    candidate_publication_count: int
    trial_sha256: str


def _trial_payload(value: CoupledAMRJoinedLeafGPUTrialV98) -> dict:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID+":trial",
        "coarse_parent": value.coarse_parent_state_sha256,
        "coarse_provisional": value.coarse_provisional_trial_sha256,
        "fine_parent": value.fine_parent_state_sha256,
        "fine_states": [value.fine1_state_sha256,
                        value.fine2_state_sha256],
        "correction": value.correction.receipt_sha256,
        "corrected_coarse": value.corrected_coarse.state_sha256,
        "candidate": value.candidate.state_sha256,
        "corrected_coarse_q3":
            value.corrected_coarse_provider.state.state_sha256,
        "final_fine_q3": value.final_fine_provider.state.state_sha256,
        "source_transaction_close_count":
            value.source_transaction_close_count,
        "candidate_publication_count": value.candidate_publication_count,
        "actual_cuda": True,
        "automatic_cpu_fallback": False,
    }


def clone_wetdry_q3_provider_gpu_v98(
    execution: CoupledWetDryGPUExecutionV97,
    template: ResidentQ3V83ActiveWetDryExnerAdapterV97,
    state: ResidentQ3V84MixedStateV93,
) -> ResidentQ3V83ActiveWetDryExnerAdapterV97:
    if type(template) is not ResidentQ3V83ActiveWetDryExnerAdapterV97:
        raise CoupledAMRJoinedLeafGPUV98Error(
            "v98 CUDA provider template type differs")
    return ResidentQ3V83ActiveWetDryExnerAdapterV97(
        outer_execution=execution.active_execution,
        binding=template.binding, state=state)


def make_coupled_wetdry_runtime_gpu_v98(
    execution: CoupledWetDryGPUExecutionV97, hydraulic_runtime,
    hles_runtime: ResidentHlesProducerLifecycleV90,
    provider: ResidentQ3V83ActiveWetDryExnerAdapterV97,
) -> ResidentCoupledWetDryLifecycleV97:
    inner = ResidentTransportedFluidSettlingLifecycleV92(
        _v92_execution(execution.active_execution))
    active = ResidentActiveSuspendedWetDryLifecycleV97(
        execution.active_execution, inner, provider)
    return ResidentCoupledWetDryLifecycleV97(
        execution, hydraulic_runtime, active, hles_runtime)


def preflight_coupled_amr_joined_leaf_gpu_v98(
    coarse_execution: CoupledWetDryGPUExecutionV97,
    coarse_parent: CoupledWetDryGPUStateV97,
    coarse_provider: ResidentQ3V83ActiveWetDryExnerAdapterV97,
    fine_execution: CoupledWetDryGPUExecutionV97,
    fine_parent: CoupledWetDryGPUStateV97,
    fine_provider: ResidentQ3V83ActiveWetDryExnerAdapterV97, *,
    previous: CoupledAMRJoinedLeafGPUStateV98 | None = None,
) -> None:
    """Reject an inadmissible macro before any hydraulic child is launched."""
    validate_coupled_wetdry_gpu_state_v97(
        coarse_parent, coarse_execution, coarse_provider)
    validate_coupled_wetdry_gpu_state_v97(
        fine_parent, fine_execution, fine_provider)
    if previous is None:
        if (coarse_parent.accepted_updates != 0 or
                fine_parent.accepted_updates != 0 or
                coarse_parent.time_s != fine_parent.time_s):
            raise CoupledAMRJoinedLeafGPUV98Error(
                "v98 CUDA joined initial macro parents differ")
        return
    validate_coupled_amr_joined_leaf_state_gpu_v98(
        previous, coarse_execution, fine_execution,
        corrected_provider=coarse_provider, fine_provider=fine_provider)
    if (previous.accepted_macro_updates >= MAX_MACRO_UPDATES or
            previous.coarse_numerical_parent is not coarse_parent or
            previous.fine_numerical_parent is not fine_parent or
            previous.time_s != coarse_parent.time_s or
            previous.time_s != fine_parent.time_s):
        raise CoupledAMRJoinedLeafGPUV98Error(
            "v98 CUDA joined macro limit or continuation parent differs")


def _corrected_q3_state(
    execution: CoupledWetDryGPUExecutionV97,
    provider: ResidentQ3V83ActiveWetDryExnerAdapterV97,
    provisional: ResidentQ3V84MixedStateV93,
    bed: np.ndarray,
    face_flux: np.ndarray,
) -> ResidentQ3V84MixedStateV93:
    core = provisional.core
    return resolve_resident_q3_v84_mixed_state_v93(
        execution_fingerprint=provider.execution_fingerprint,
        binding=provider.binding,
        authoritative_bed_mass_kg_m2=bed,
        bulk_density_kg_m3=core.bulk_density_kg_m3,
        retained_fraction=provider.cp.asnumpy(provisional.active_fraction),
        accepted_face_flux_kg_m_s=face_flux,
        accepted_cell_availability=core.accepted_cell_availability,
        named_boundary_positive_in_kg=core.named_boundary_positive_in_kg,
        named_boundary_absolute_kg=core.named_boundary_absolute_kg,
        depletion_roundoff_positive_in_kg=
            core.depletion_roundoff_positive_in_kg,
        accepted_updates=core.accepted_updates, time_s=core.time_s)


def _corrected_numerical_parent(
    execution: CoupledWetDryGPUExecutionV97,
    parent: CoupledWetDryGPUStateV97,
    provisional_trial: CoupledWetDryGPUTrialV97,
    provider: ResidentQ3V83ActiveWetDryExnerAdapterV97,
    correction: CoupledAMRCoarseCorrectionGPUV98,
    interface: CoupledAMRInterfaceMapGPUV98,
    image_runtime: ResidentAMRCorrectedCoarseImagesV98,
    closure_chezy_mhalf_s: object,
    poll: Callable[[], bool],
) -> tuple[CoupledAMRCorrectedCoarseGPUStateV98,
           ResidentQ3V83ActiveWetDryExnerAdapterV97]:
    provisional = provisional_trial.candidate
    v92_execution = _v92_execution(execution.active_execution)
    v91_execution = _v91_execution(execution.active_execution)
    v78_execution = v91_execution.base_execution
    v92 = _v92_state(provisional.active_state)
    bed = provisional.active_state.base_state.bed_mass_kg_m2
    fluid = np.vstack((v92.h_temperature_c_m, v92.h_salinity_ppt_m))
    delta_q = np.concatenate((
        (correction.water_depth_delta_m,), correction.momentum_delta_m2_s))
    result = image_runtime.evaluate(
        v92.q, v92.suspended_state.suspended_mass_kg_m2, bed, fluid,
        delta_q, correction.suspended_delta_kg_m2,
        correction.bed_delta_kg_m2, correction.fluid_delta,
        correction_receipt_sha256=correction.receipt_sha256,
        source_state_sha256=provisional.state_sha256,
        cell=interface.coarse_leaf_cell,
        dry_tolerance_m=_grid(execution.active_execution).dry_tolerance_m,
        temperature_mode=v92_execution.temperature_mode,
        salinity_mode=v92_execution.salinity_mode,
        equilibrium_temperature_c=v92_execution.equilibrium_temperature_c,
        equilibrium_salinity_ppt=v92_execution.equilibrium_salinity_ppt,
        density_eos=v92_execution.density_eos,
        reference_water_density_kg_m3=(
            v91_execution.base_execution.base_execution.profile.
            water_density_kg_m3),
        cancelled=poll)
    images = result["images"]
    if poll():
        raise InterruptedError(
            "v98 CUDA joined cancelled after corrected images")
    provisional_prepared = provisional_trial.active_trial.base_v93_trial.\
        q3_prepared_commit
    q3_state = _corrected_q3_state(
        execution, provider, provisional_prepared.rebound_state,
        images["bed"], correction.corrected_face_bedload_flux_kg_m_s)
    corrected_provider = clone_wetdry_q3_provider_gpu_v98(
        execution, provider, q3_state)
    grid = _grid(execution.active_execution)
    area = np.asarray(grid.cell_area_m2)
    updates, time_s = provisional.accepted_updates, provisional.time_s
    suspended_reference = np.sum(
        images["suspended"]*area[None, :], axis=1)
    bed_reference = np.sum(images["bed"]*area[None, :], axis=1)
    suspended_parent = v92.suspended_state
    suspended_lifecycle = suspended_parent.lifecycle
    v78 = resolve_multifraction_suspended_state_v78(
        v78_execution,
        suspended_mass_kg_m2=images["suspended"],
        bed_mass_kg_m2=images["bed"],
        boundary_last_outflow_concentration_kg_m3=
            suspended_lifecycle.boundary_last_outflow_concentration_kg_m3,
        boundary_return_clock_s=suspended_lifecycle.boundary_return_clock_s,
        advective_boundary_ledger_kg=np.zeros_like(
            suspended_lifecycle.advective_boundary_ledger_kg),
        diffusive_boundary_ledger_kg=np.zeros_like(
            suspended_lifecycle.diffusive_boundary_ledger_kg),
        water_exchange_ledger_kg=np.zeros(execution.fraction_count),
        bed_exchange_ledger_kg=np.zeros(execution.fraction_count),
        wetdry_water_ledger_kg=np.zeros(execution.fraction_count),
        wetdry_bed_ledger_kg=np.zeros(execution.fraction_count),
        reference_water_mass_kg=suspended_reference,
        reference_bed_mass_kg=bed_reference,
        accepted_updates=updates, time_s=time_s)
    v91 = _wrap_v91_state(
        v91_execution, q=images["q"], suspended_state=v78,
        receipt_chain=_hash({"scope_id": SCOPE_ID+":v91-rebase",
                             "correction": correction.receipt_sha256}))
    scalar_reference = np.asarray((
        np.sum(images["fluid"][0]*area),
        np.sum(images["fluid"][1]*area)))
    v92_rebased = _wrap_v92_state(
        v92_execution, v91,
        h_temperature_c_m=images["fluid"][0],
        h_salinity_ppt_m=images["fluid"][1],
        boundary_last_outflow_fluid=v92.boundary_last_outflow_fluid,
        boundary_return_clock_s=v92.boundary_return_clock_s,
        reference_scalar_integral=scalar_reference,
        advective_boundary_ledger=np.zeros_like(
            v92.advective_boundary_ledger),
        diffusive_boundary_ledger=np.zeros_like(
            v92.diffusive_boundary_ledger),
        wetdry_scalar_ledger=np.zeros(2),
        accepted_receipt_chain_sha256=_hash({
            "scope_id": SCOPE_ID+":v92-rebase",
            "correction": correction.receipt_sha256}),
        temperature_cell_c=images["temperature"],
        salinity_cell_ppt=images["salinity"],
        water_density_cell_kg_m3=images["water_density"],
        molecular_viscosity_cell_m2_s=images["viscosity"])
    v93_execution = execution.active_execution.base_execution
    v93 = _wrap_v93_state(
        v93_execution, v92_rebased,
        q3_provider_kind=corrected_provider.provider_kind,
        q3_provider_state_sha256=q3_state.state_sha256,
        bed_mass_kg_m2=images["bed"],
        reference_bed_integral_kg=bed_reference,
        suspended_exchange_ledger_kg=np.zeros(execution.fraction_count),
        suspended_boundary_ledger_kg=np.zeros_like(
            provisional.active_state.base_state.
                suspended_boundary_ledger_kg),
        excluded_water_exchange_kg=np.zeros(execution.fraction_count),
        q3_named_boundary_positive_in_kg=np.zeros_like(
            provisional.active_state.base_state.
                q3_named_boundary_positive_in_kg),
        q3_named_boundary_absolute_kg=np.zeros_like(
            provisional.active_state.base_state.
                q3_named_boundary_absolute_kg),
        q3_depletion_roundoff_positive_in_kg=
            np.zeros(execution.fraction_count),
        wetdry_bed_ledger_kg=np.zeros(execution.fraction_count),
        accepted_exner_chain_sha256=_hash({
            "scope_id": SCOPE_ID+":v93-rebase",
            "correction": correction.receipt_sha256,
            "q3": q3_state.state_sha256}))
    active = _wrap_v94_state(
        execution.active_execution, v93,
        suspended_diameter_m=
            provisional.active_state.suspended_diameter_m,
        accepted_iopsus_chain_sha256=_hash({
            "scope_id": SCOPE_ID+":v94-rebase",
            "carried_dss": _digest(
                provisional.active_state.suspended_diameter_m),
            "dss_recompute_count": 0}))

    hles_execution = v91_execution.hles_execution
    hles_runtime = ResidentHlesProducerLifecycleV90(hles_execution)
    accepted_step = provisional_trial.active_trial.base_v93_trial.\
        base_v92_trial.hydraulic_receipt.base_step_receipt
    dt_s = provisional_trial.hydraulic_trial.dt_s
    hles_arrays = _hles_inputs(
        execution, images["q"], correction.corrected_face_water_flux_m2_s,
        closure_chezy_mhalf_s)
    hles_input = make_hles_input_receipt_v90(
        hles_execution, parent.hles_state,
        accepted_step_receipt=accepted_step,
        current_handoff=provisional_trial.current_handoff,
        closure_interval_s=dt_s,
        accepted_face_normal_velocity_m_s=hles_arrays[0],
        active_face=hles_arrays[1], closure_depth_m=hles_arrays[2],
        closure_chezy_mhalf_s=hles_arrays[3], active_cell=hles_arrays[4])
    cp = hles_runtime.cp
    resident = tuple(cp.ascontiguousarray(cp.asarray(
        item, dtype=cp.int32 if index in (1, 4) else cp.float64))
        for index, item in enumerate(hles_arrays))
    hles_trial = hles_runtime.begin_trial(
        parent.hles_state, accepted_step_receipt=accepted_step,
        hles_input_receipt=hles_input,
        current_handoff=provisional_trial.current_handoff,
        accepted_face_normal_velocity_m_s=resident[0],
        active_face=resident[1], closure_depth_m=resident[2],
        closure_chezy_mhalf_s=resident[3], active_cell=resident[4],
        cancelled=poll())
    try:
        numerical = _wrap_v97_state(
            execution, active_state=active, hles_state=hles_trial.candidate,
            accepted_chain_sha256=_hash({
                "scope_id": SCOPE_ID+":corrected-numerical-parent",
                "previous": parent.accepted_chain_sha256,
                "provisional": provisional_trial.trial_sha256,
                "correction": correction.receipt_sha256,
                "image": result["receipt_sha256"],
                "hles": hles_trial.trial_sha256}))
        validate_coupled_wetdry_gpu_state_v97(
            numerical, execution, corrected_provider)
    finally:
        hles_runtime.rollback(parent.hles_state, hles_trial)
    arrays = tuple(_frozen(value, shape, name) for value, shape, name in (
        (correction.corrected_face_water_flux_m2_s,
         (execution.face_count,), "corrected water flux"),
        (correction.corrected_face_bedload_flux_kg_m_s,
         (execution.fraction_count, execution.face_count),
         "corrected bed flux"),
        (images["dry_transfer"],
         (execution.fraction_count, execution.cell_count), "dry transfer"),
        (images["momentum_cleanup"],
         (2, execution.cell_count), "momentum cleanup"),
        (images["fluid_cleanup"],
         (2, execution.cell_count), "fluid cleanup")))
    counts = CoupledAMRCorrectedCoarseCountsGPUV98()
    placeholder = CoupledAMRCorrectedCoarseGPUStateV98(
        execution.fingerprint, provisional.state_sha256,
        provisional_trial.trial_sha256, correction.receipt_sha256,
        correction.internal_interface_ledger.receipt_sha256,
        image_runtime.receipt["receipt_sha256"], result["receipt_sha256"],
        numerical, q3_state, *arrays, counts, BACKEND, "")
    state = CoupledAMRCorrectedCoarseGPUStateV98(
        **{**placeholder.__dict__,
           "state_sha256": _hash(_corrected_payload(placeholder))})
    return state, corrected_provider


def prepare_coupled_amr_joined_leaf_gpu_v98(
    coarse_execution: CoupledWetDryGPUExecutionV97,
    coarse_parent: CoupledWetDryGPUStateV97,
    coarse_provider: ResidentQ3V83ActiveWetDryExnerAdapterV97,
    coarse_runtime: ResidentCoupledWetDryLifecycleV97,
    coarse_provisional_trial: CoupledWetDryGPUTrialV97,
    fine_execution: CoupledWetDryGPUExecutionV97,
    fine_parent: CoupledWetDryGPUStateV97,
    fine_source_provider: ResidentQ3V83ActiveWetDryExnerAdapterV97,
    fine1: CoupledAMRFineSubstepStateGPUV98,
    fine1_provider: ResidentQ3V83ActiveWetDryExnerAdapterV97,
    fine2: CoupledAMRFineSubstepStateGPUV98,
    fine2_provider: ResidentQ3V83ActiveWetDryExnerAdapterV97,
    interface: CoupledAMRInterfaceMapGPUV98,
    geometry_binding: CoupledAMRGeometryBindingGPUV98,
    coarse_integrals: CoupledFaceTimeIntegralsGPUV98,
    correction: CoupledAMRCoarseCorrectionGPUV98, *,
    corrected_image_runtime: ResidentAMRCorrectedCoarseImagesV98,
    closure_chezy_mhalf_s: object,
    previous: CoupledAMRJoinedLeafGPUStateV98 | None = None,
    cancelled: bool | Callable[[], bool] = False,
) -> CoupledAMRJoinedLeafGPUTrialV98:
    poll = _poll(cancelled)
    preflight_coupled_amr_joined_leaf_gpu_v98(
        coarse_execution, coarse_parent, coarse_provider,
        fine_execution, fine_parent, fine_source_provider,
        previous=previous)
    coarse_runtime._validate_trial(coarse_parent, coarse_provisional_trial)
    validate_coupled_amr_fine_substep_state_gpu_v98(
        fine1, fine_execution, fine_parent, fine_source_provider,
        fine1_provider)
    validate_coupled_amr_fine_substep_state_gpu_v98(
        fine2, fine_execution, fine1.numerical_parent, fine1_provider,
        fine2_provider)
    validate_coupled_amr_coarse_correction_gpu_v98(
        correction, interface, coarse_integrals,
        (fine1.face_time_integrals, fine2.face_time_integrals),
        coarse_execution=coarse_execution,
        fine_execution=fine_execution,
        geometry_binding=geometry_binding)
    if (coarse_runtime._parent is not coarse_parent or
            coarse_runtime._trial is not coarse_provisional_trial or
            fine1.hydraulic_auxiliary_evidence_sha256 is None or
            fine2.hydraulic_auxiliary_evidence_sha256 is None or
            poll()):
        raise CoupledAMRJoinedLeafGPUV98Error(
            "v98 CUDA joined source transaction or evidence differs")
    corrected, corrected_provider = _corrected_numerical_parent(
        coarse_execution, coarse_parent, coarse_provisional_trial,
        coarse_provider, correction, interface, corrected_image_runtime,
        closure_chezy_mhalf_s, poll)
    ledger = correction.internal_interface_ledger
    if previous is None:
        updates = 1
        prior = None
        cumulative = (
            float(ledger.water_volume_m3), ledger.momentum_m4_s,
            ledger.suspended_mass_kg, ledger.fluid_integral,
            ledger.bed_mass_kg)
        previous_chain = _hash({"scope_id": SCOPE_ID+":initial"})
    else:
        validate_coupled_amr_joined_leaf_state_gpu_v98(
            previous, coarse_execution, fine_execution,
            corrected_provider=None, fine_provider=None)
        if (previous.accepted_macro_updates >= MAX_MACRO_UPDATES or
                previous.coarse_numerical_parent is not coarse_parent or
                previous.fine_numerical_parent is not fine_parent):
            raise CoupledAMRJoinedLeafGPUV98Error(
                "v98 CUDA joined previous macro parent differs")
        updates = previous.accepted_macro_updates+1
        prior = previous.state_sha256
        cumulative = (
            previous.cumulative_water_volume_m3+ledger.water_volume_m3,
            previous.cumulative_momentum_m4_s+ledger.momentum_m4_s,
            previous.cumulative_suspended_mass_kg+ledger.suspended_mass_kg,
            previous.cumulative_fluid_integral+ledger.fluid_integral,
            previous.cumulative_bed_mass_kg+ledger.bed_mass_kg)
        previous_chain = previous.accepted_chain_sha256
    frozen = (
        float(cumulative[0]),
        _frozen(cumulative[1], (2,), "cumulative momentum"),
        _frozen(cumulative[2], (coarse_execution.fraction_count,),
                "cumulative suspended"),
        _frozen(cumulative[3], (2,), "cumulative fluid"),
        _frozen(cumulative[4], (coarse_execution.fraction_count,),
                "cumulative bed"))
    counts = CoupledAMRJoinedLeafCountsGPUV98()
    chain = _hash({
        "previous": previous_chain,
        "coarse": corrected.state_sha256,
        "fine": fine2.numerical_parent.state_sha256,
        "correction": correction.receipt_sha256,
        "macro_update": updates})
    placeholder_state = CoupledAMRJoinedLeafGPUStateV98(
        corrected, fine2.numerical_parent, fine2_provider.state,
        correction.receipt_sha256, *frozen, updates,
        corrected.numerical_parent.time_s, prior, counts, chain, "")
    candidate = CoupledAMRJoinedLeafGPUStateV98(
        **{**placeholder_state.__dict__,
           "state_sha256": _hash(_joined_payload(placeholder_state))})
    placeholder = CoupledAMRJoinedLeafGPUTrialV98(
        coarse_parent.state_sha256, coarse_provisional_trial.trial_sha256,
        fine_parent.state_sha256, fine1.state_sha256, fine2.state_sha256,
        correction, corrected, candidate, corrected_provider,
        fine2_provider, 0, 0, "")
    trial = CoupledAMRJoinedLeafGPUTrialV98(
        **{**placeholder.__dict__,
           "trial_sha256": _hash(_trial_payload(placeholder))})
    validate_coupled_amr_joined_leaf_trial_gpu_v98(
        trial, coarse_execution, coarse_parent, coarse_runtime,
        coarse_provisional_trial, fine_execution, fine_parent,
        fine2_provider, previous=previous)
    return trial


def validate_coupled_amr_joined_leaf_state_gpu_v98(
    value: CoupledAMRJoinedLeafGPUStateV98,
    coarse_execution: CoupledWetDryGPUExecutionV97,
    fine_execution: CoupledWetDryGPUExecutionV97, *,
    corrected_provider: ResidentQ3V83ActiveWetDryExnerAdapterV97 | None,
    fine_provider: ResidentQ3V83ActiveWetDryExnerAdapterV97 | None,
) -> None:
    ng, nc, nf = (coarse_execution.fraction_count,
                  coarse_execution.cell_count,
                  coarse_execution.face_count)
    corrected = value.coarse_corrected if isinstance(
        value, CoupledAMRJoinedLeafGPUStateV98) else None
    if (type(value) is not CoupledAMRJoinedLeafGPUStateV98 or
            value.accepted_macro_updates not in (1, 2) or
            value.coarse_numerical_parent.accepted_updates !=
                value.accepted_macro_updates or
            value.fine_numerical_parent.accepted_updates !=
                2*value.accepted_macro_updates or
            value.coarse_numerical_parent.time_s != value.time_s or
            value.fine_numerical_parent.time_s != value.time_s or
            value.counts != CoupledAMRJoinedLeafCountsGPUV98() or
            value.state_sha256 != _hash(_joined_payload(value)) or
            value.coarse_corrected.state_sha256 !=
                _hash(_corrected_payload(value.coarse_corrected))):
        raise CoupledAMRJoinedLeafGPUV98Error(
            "v98 CUDA joined state identity or clock differs")
    if (corrected.execution_fingerprint != coarse_execution.fingerprint or
            corrected.numerical_parent is not
                value.coarse_numerical_parent or
            corrected.q3_state is not value.coarse_corrected.q3_state or
            corrected.backend != BACKEND or
            corrected.counts != CoupledAMRCorrectedCoarseCountsGPUV98() or
            any(array.flags.writeable for array in (
                corrected.corrected_face_water_flux_m2_s,
                corrected.corrected_face_bedload_flux_kg_m_s,
                corrected.reflux_dry_transfer_to_bed_kg_m2,
                corrected.reflux_dry_momentum_cleanup_m2_s,
                corrected.reflux_dry_fluid_cleanup)) or
            corrected.corrected_face_water_flux_m2_s.shape != (nf,) or
            corrected.corrected_face_bedload_flux_kg_m_s.shape != (ng, nf) or
            corrected.reflux_dry_transfer_to_bed_kg_m2.shape != (ng, nc) or
            corrected.reflux_dry_momentum_cleanup_m2_s.shape != (2, nc) or
            corrected.reflux_dry_fluid_cleanup.shape != (2, nc) or
            any(not np.isfinite(array).all() for array in (
                corrected.corrected_face_water_flux_m2_s,
                corrected.corrected_face_bedload_flux_kg_m_s,
                corrected.reflux_dry_transfer_to_bed_kg_m2,
                corrected.reflux_dry_momentum_cleanup_m2_s,
                corrected.reflux_dry_fluid_cleanup)) or
            any(not isinstance(_sha(item, "corrected provenance"), str)
                for item in (
                    corrected.provisional_source_state_sha256,
                    corrected.provisional_source_trial_sha256,
                    corrected.correction_receipt_sha256,
                    corrected.internal_interface_ledger_sha256,
                    corrected.corrected_image_runtime_sha256,
                    corrected.corrected_image_receipt_sha256))):
        raise CoupledAMRJoinedLeafGPUV98Error(
            "v98 CUDA corrected coarse state differs")
    for array, shape, name in (
            (value.cumulative_momentum_m4_s, (2,), "cumulative momentum"),
            (value.cumulative_suspended_mass_kg, (ng,),
             "cumulative suspended"),
            (value.cumulative_fluid_integral, (2,), "cumulative fluid"),
            (value.cumulative_bed_mass_kg, (ng,), "cumulative bed")):
        if (array.shape != shape or array.flags.writeable or
                not np.isfinite(array).all()):
            raise CoupledAMRJoinedLeafGPUV98Error(
                f"v98 CUDA joined {name} differs")
    if (not np.isfinite(value.cumulative_water_volume_m3) or
            value.time_s < 0.0 or
            (value.previous_state_sha256 is not None and
             _sha(value.previous_state_sha256, "previous state") !=
                value.previous_state_sha256) or
            _sha(value.accepted_chain_sha256, "accepted chain") !=
                value.accepted_chain_sha256):
        raise CoupledAMRJoinedLeafGPUV98Error(
            "v98 CUDA joined cumulative identity differs")
    if corrected_provider is not None:
        validate_coupled_wetdry_gpu_state_v97(
            value.coarse_numerical_parent, coarse_execution,
            corrected_provider)
        if (corrected_provider.state.state_sha256 !=
                value.coarse_corrected.q3_state.state_sha256):
            raise CoupledAMRJoinedLeafGPUV98Error(
                "v98 CUDA corrected coarse provider differs")
    if fine_provider is not None:
        validate_coupled_wetdry_gpu_state_v97(
            value.fine_numerical_parent, fine_execution, fine_provider)
        if fine_provider.state.state_sha256 != value.fine_q3_state.state_sha256:
            raise CoupledAMRJoinedLeafGPUV98Error(
                "v98 CUDA final fine provider differs")


def validate_coupled_amr_joined_leaf_trial_gpu_v98(
    value: CoupledAMRJoinedLeafGPUTrialV98,
    coarse_execution: CoupledWetDryGPUExecutionV97,
    coarse_parent: CoupledWetDryGPUStateV97,
    coarse_runtime: ResidentCoupledWetDryLifecycleV97,
    coarse_provisional_trial: CoupledWetDryGPUTrialV97,
    fine_execution: CoupledWetDryGPUExecutionV97,
    fine_parent: CoupledWetDryGPUStateV97,
    fine_provider: ResidentQ3V83ActiveWetDryExnerAdapterV97, *,
    previous: CoupledAMRJoinedLeafGPUStateV98 | None,
) -> None:
    if (type(value) is not CoupledAMRJoinedLeafGPUTrialV98 or
            value.coarse_parent_state_sha256 != coarse_parent.state_sha256 or
            value.coarse_provisional_trial_sha256 !=
                coarse_provisional_trial.trial_sha256 or
            value.fine_parent_state_sha256 != fine_parent.state_sha256 or
            value.corrected_coarse is not value.candidate.coarse_corrected or
            value.corrected_coarse_provider.state.state_sha256 !=
                value.corrected_coarse.q3_state.state_sha256 or
            value.final_fine_provider is not fine_provider or
            value.source_transaction_close_count != 0 or
            value.candidate_publication_count != 0 or
            value.trial_sha256 != _hash(_trial_payload(value)) or
            coarse_runtime._parent is not coarse_parent or
            coarse_runtime._trial is not coarse_provisional_trial):
        raise CoupledAMRJoinedLeafGPUV98Error(
            "v98 CUDA joined trial identity differs")
    validate_coupled_amr_joined_leaf_state_gpu_v98(
        value.candidate, coarse_execution, fine_execution,
        corrected_provider=value.corrected_coarse_provider,
        fine_provider=fine_provider)
    expected_update = 1 if previous is None else \
        previous.accepted_macro_updates+1
    if value.candidate.accepted_macro_updates != expected_update:
        raise CoupledAMRJoinedLeafGPUV98Error(
            "v98 CUDA joined macro clock differs")


def commit_coupled_amr_joined_leaf_gpu_v98(
    value: CoupledAMRJoinedLeafGPUTrialV98,
    coarse_execution: CoupledWetDryGPUExecutionV97,
    coarse_parent: CoupledWetDryGPUStateV97,
    coarse_runtime: ResidentCoupledWetDryLifecycleV97,
    coarse_provisional_trial: CoupledWetDryGPUTrialV97,
    fine_execution: CoupledWetDryGPUExecutionV97,
    fine_parent: CoupledWetDryGPUStateV97, *,
    previous: CoupledAMRJoinedLeafGPUStateV98 | None,
) -> tuple[CoupledAMRJoinedLeafGPUStateV98,
           ResidentQ3V83ActiveWetDryExnerAdapterV97,
           ResidentQ3V83ActiveWetDryExnerAdapterV97]:
    validate_coupled_amr_joined_leaf_trial_gpu_v98(
        value, coarse_execution, coarse_parent, coarse_runtime,
        coarse_provisional_trial, fine_execution, fine_parent,
        value.final_fine_provider, previous=previous)
    restored = coarse_runtime.rollback(coarse_parent, coarse_provisional_trial)
    if restored is not coarse_parent:
        raise CoupledAMRJoinedLeafGPUV98Error(
            "v98 CUDA joined source rollback substituted its parent")
    return (value.candidate, value.corrected_coarse_provider,
            value.final_fine_provider)


def rollback_coupled_amr_joined_leaf_gpu_v98(
    value: CoupledAMRJoinedLeafGPUTrialV98,
    coarse_parent: CoupledWetDryGPUStateV97,
    coarse_runtime: ResidentCoupledWetDryLifecycleV97,
    coarse_provisional_trial: CoupledWetDryGPUTrialV97,
) -> CoupledWetDryGPUStateV97:
    if (type(value) is not CoupledAMRJoinedLeafGPUTrialV98 or
            coarse_runtime._parent is not coarse_parent or
            coarse_runtime._trial is not coarse_provisional_trial):
        raise CoupledAMRJoinedLeafGPUV98Error(
            "v98 CUDA joined rollback handle differs")
    return coarse_runtime.rollback(coarse_parent, coarse_provisional_trial)


def checkpoint_coupled_amr_joined_leaf_gpu_v98(
    state: CoupledAMRJoinedLeafGPUStateV98,
    coarse_execution: CoupledWetDryGPUExecutionV97,
    fine_execution: CoupledWetDryGPUExecutionV97,
    coarse_runtime: ResidentCoupledWetDryLifecycleV97,
    fine_runtime: ResidentCoupledWetDryLifecycleV97,
    corrected_provider: ResidentQ3V83ActiveWetDryExnerAdapterV97,
    fine_provider: ResidentQ3V83ActiveWetDryExnerAdapterV97,
) -> dict[str, object]:
    validate_coupled_amr_joined_leaf_state_gpu_v98(
        state, coarse_execution, fine_execution,
        corrected_provider=corrected_provider, fine_provider=fine_provider)
    body = {
        "scope_id": SCOPE_ID+":checkpoint", "schema": SCHEMA,
        "coarse_execution": coarse_execution.fingerprint,
        "fine_execution": fine_execution.fingerprint,
        "coarse_checkpoint":
            coarse_runtime.checkpoint(state.coarse_numerical_parent),
        "fine_checkpoint":
            fine_runtime.checkpoint(state.fine_numerical_parent),
        "corrected_coarse": _corrected_payload(state.coarse_corrected),
        "corrected_coarse_arrays": {
            "corrected_face_water_flux_m2_s":
                state.coarse_corrected.corrected_face_water_flux_m2_s.tolist(),
            "corrected_face_bedload_flux_kg_m_s":
                state.coarse_corrected.
                corrected_face_bedload_flux_kg_m_s.tolist(),
            "reflux_dry_transfer_to_bed_kg_m2":
                state.coarse_corrected.
                reflux_dry_transfer_to_bed_kg_m2.tolist(),
            "reflux_dry_momentum_cleanup_m2_s":
                state.coarse_corrected.
                reflux_dry_momentum_cleanup_m2_s.tolist(),
            "reflux_dry_fluid_cleanup":
                state.coarse_corrected.reflux_dry_fluid_cleanup.tolist(),
        },
        "correction_receipt_sha256": state.correction_receipt_sha256,
        "cumulative_water_volume_m3": state.cumulative_water_volume_m3,
        "cumulative_momentum_m4_s":
            state.cumulative_momentum_m4_s.tolist(),
        "cumulative_suspended_mass_kg":
            state.cumulative_suspended_mass_kg.tolist(),
        "cumulative_fluid_integral":
            state.cumulative_fluid_integral.tolist(),
        "cumulative_bed_mass_kg": state.cumulative_bed_mass_kg.tolist(),
        "accepted_macro_updates": state.accepted_macro_updates,
        "time_s": state.time_s,
        "previous_state_sha256": state.previous_state_sha256,
        "accepted_chain_sha256": state.accepted_chain_sha256,
        "state_sha256": state.state_sha256,
    }
    return {**body, "fingerprint": _hash(body)}


def load_coupled_amr_joined_leaf_checkpoint_gpu_v98(
    document: Mapping[str, object],
    coarse_execution: CoupledWetDryGPUExecutionV97,
    fine_execution: CoupledWetDryGPUExecutionV97,
    coarse_runtime: ResidentCoupledWetDryLifecycleV97,
    fine_runtime: ResidentCoupledWetDryLifecycleV97,
) -> tuple[CoupledAMRJoinedLeafGPUStateV98,
           ResidentQ3V83ActiveWetDryExnerAdapterV97,
           ResidentQ3V83ActiveWetDryExnerAdapterV97]:
    if not isinstance(document, Mapping):
        raise CoupledAMRJoinedLeafGPUV98Error(
            "v98 CUDA joined checkpoint type differs")
    body = {key: value for key, value in document.items()
            if key != "fingerprint"}
    expected_keys = {
        "scope_id", "schema", "coarse_execution", "fine_execution",
        "coarse_checkpoint", "fine_checkpoint", "corrected_coarse",
        "corrected_coarse_arrays", "correction_receipt_sha256",
        "cumulative_water_volume_m3", "cumulative_momentum_m4_s",
        "cumulative_suspended_mass_kg", "cumulative_fluid_integral",
        "cumulative_bed_mass_kg", "accepted_macro_updates", "time_s",
        "previous_state_sha256", "accepted_chain_sha256", "state_sha256",
        "fingerprint",
    }
    if (set(document) != expected_keys or
            document.get("fingerprint") != _hash(body) or
            document.get("scope_id") != SCOPE_ID+":checkpoint" or
            document.get("schema") != SCHEMA or
            document.get("coarse_execution") != coarse_execution.fingerprint or
            document.get("fine_execution") != fine_execution.fingerprint):
        raise CoupledAMRJoinedLeafGPUV98Error(
            "v98 CUDA joined checkpoint identity differs")
    coarse_provider = coarse_runtime.active.q3_provider
    fine_provider = fine_runtime.active.q3_provider
    coarse_provider_before = coarse_provider.checkpoint()
    fine_provider_before = fine_provider.checkpoint()
    runtimes = (coarse_runtime, fine_runtime)
    resident_before = tuple(tuple(value.copy() for value in (
        runtime.active.inner.combined.accepted_last,
        runtime.active.inner.combined.accepted_clock,
        runtime.active.inner.combined.trial_last,
        runtime.active.inner.combined.trial_clock)) for runtime in runtimes)
    receipt_before = tuple((
        copy.deepcopy(runtime.last_accepted_receipt),
        copy.deepcopy(runtime.active.last_accepted_receipt),
        copy.deepcopy(runtime.active.inner.last_accepted_receipt),
        copy.deepcopy(runtime.active.inner.base.last_accepted_receipt))
        for runtime in runtimes)
    try:
        coarse = coarse_runtime.restore_checkpoint(
            document["coarse_checkpoint"])
        fine = fine_runtime.restore_checkpoint(document["fine_checkpoint"])
        payload = document["corrected_coarse"]
        arrays = document["corrected_coarse_arrays"]
        if (not isinstance(payload, Mapping) or
                not isinstance(arrays, Mapping) or
                payload.get("execution") != coarse_execution.fingerprint or
                payload.get("numerical_parent") != coarse.state_sha256 or
                payload.get("q3_state") !=
                    coarse_provider.state.state_sha256 or
                payload.get("counts") !=
                    CoupledAMRCorrectedCoarseCountsGPUV98().__dict__ or
                payload.get("backend") != BACKEND or
                document["correction_receipt_sha256"] !=
                    payload.get("correction")):
            raise CoupledAMRJoinedLeafGPUV98Error(
                "v98 CUDA corrected checkpoint payload differs")
        ng, nc, nf = (coarse_execution.fraction_count,
                      coarse_execution.cell_count,
                      coarse_execution.face_count)
        frozen = (
            _frozen(arrays["corrected_face_water_flux_m2_s"], (nf,),
                    "checkpoint water flux"),
            _frozen(arrays["corrected_face_bedload_flux_kg_m_s"],
                    (ng, nf), "checkpoint bed flux"),
            _frozen(arrays["reflux_dry_transfer_to_bed_kg_m2"],
                    (ng, nc), "checkpoint dry transfer"),
            _frozen(arrays["reflux_dry_momentum_cleanup_m2_s"],
                    (2, nc), "checkpoint momentum cleanup"),
            _frozen(arrays["reflux_dry_fluid_cleanup"],
                    (2, nc), "checkpoint fluid cleanup"),
        )
        corrected0 = CoupledAMRCorrectedCoarseGPUStateV98(
            coarse_execution.fingerprint,
            _sha(payload["provisional_source_state"],
                 "provisional source state"),
            _sha(payload["provisional_source_trial"],
                 "provisional source trial"),
            _sha(payload["correction"], "correction"),
            _sha(payload["internal_interface_ledger"],
                 "internal interface ledger"),
            _sha(payload["corrected_image_runtime"],
                 "corrected image runtime"),
            _sha(payload["corrected_image_receipt"],
                 "corrected image receipt"),
            coarse, coarse_provider.state, *frozen,
            CoupledAMRCorrectedCoarseCountsGPUV98(), BACKEND, "")
        corrected = CoupledAMRCorrectedCoarseGPUStateV98(
            **{**corrected0.__dict__,
               "state_sha256": _hash(_corrected_payload(corrected0))})
        if (_corrected_payload(corrected) != dict(payload)):
            raise CoupledAMRJoinedLeafGPUV98Error(
                "v98 CUDA corrected checkpoint arrays differ")
        ng = coarse_execution.fraction_count
        state0 = CoupledAMRJoinedLeafGPUStateV98(
            corrected, fine, fine_provider.state,
            _sha(document["correction_receipt_sha256"], "correction"),
            float(document["cumulative_water_volume_m3"]),
            _frozen(document["cumulative_momentum_m4_s"], (2,),
                    "checkpoint cumulative momentum"),
            _frozen(document["cumulative_suspended_mass_kg"], (ng,),
                    "checkpoint cumulative suspended"),
            _frozen(document["cumulative_fluid_integral"], (2,),
                    "checkpoint cumulative fluid"),
            _frozen(document["cumulative_bed_mass_kg"], (ng,),
                    "checkpoint cumulative bed"),
            int(document["accepted_macro_updates"]),
            float(document["time_s"]), document["previous_state_sha256"],
            CoupledAMRJoinedLeafCountsGPUV98(),
            _sha(document["accepted_chain_sha256"], "accepted chain"), "")
        state = CoupledAMRJoinedLeafGPUStateV98(
            **{**state0.__dict__,
               "state_sha256": _hash(_joined_payload(state0))})
        validate_coupled_amr_joined_leaf_state_gpu_v98(
            state, coarse_execution, fine_execution,
            corrected_provider=coarse_provider, fine_provider=fine_provider)
        if state.state_sha256 != document["state_sha256"]:
            raise CoupledAMRJoinedLeafGPUV98Error(
                "v98 CUDA joined restored state differs")
        coarse_runtime.last_accepted_receipt = None
        fine_runtime.last_accepted_receipt = None
        return state, coarse_provider, fine_provider
    except Exception:
        for runtime, arrays, receipts in zip(
                runtimes, resident_before, receipt_before):
            for target, saved in zip((
                    runtime.active.inner.combined.accepted_last,
                    runtime.active.inner.combined.accepted_clock,
                    runtime.active.inner.combined.trial_last,
                    runtime.active.inner.combined.trial_clock), arrays):
                runtime.active.inner.cp.copyto(target, saved)
            (runtime.last_accepted_receipt,
             runtime.active.last_accepted_receipt,
             runtime.active.inner.last_accepted_receipt,
             runtime.active.inner.base.last_accepted_receipt) = receipts
        coarse_provider.restore_checkpoint(coarse_provider_before)
        fine_provider.restore_checkpoint(fine_provider_before)
        raise


__all__ = [
    "BACKEND", "MAX_MACRO_UPDATES", "SCOPE_ID",
    "CoupledAMRCorrectedCoarseCountsGPUV98",
    "CoupledAMRCorrectedCoarseGPUStateV98",
    "CoupledAMRJoinedLeafCountsGPUV98",
    "CoupledAMRJoinedLeafGPUStateV98",
    "CoupledAMRJoinedLeafGPUTrialV98",
    "CoupledAMRJoinedLeafGPUV98Error",
    "checkpoint_coupled_amr_joined_leaf_gpu_v98",
    "clone_wetdry_q3_provider_gpu_v98",
    "commit_coupled_amr_joined_leaf_gpu_v98",
    "load_coupled_amr_joined_leaf_checkpoint_gpu_v98",
    "make_coupled_wetdry_runtime_gpu_v98",
    "preflight_coupled_amr_joined_leaf_gpu_v98",
    "prepare_coupled_amr_joined_leaf_gpu_v98",
    "rollback_coupled_amr_joined_leaf_gpu_v98",
    "validate_coupled_amr_joined_leaf_state_gpu_v98",
    "validate_coupled_amr_joined_leaf_trial_gpu_v98",
]
