"""Pure bounded diagnostics for the first joined-leaf CPU v98 route.

The collector in this module performs no numerical work and publishes no
state.  It validates the already accepted one-interface/two-macro proof and
returns JSON-ready arrays and closure residuals for an evidence figure.  The
scope is deliberately one joined leaf interface; it is not regrid or general
hierarchy evidence.
"""
from __future__ import annotations

import json

import numpy as np

from .coupled_amr_corrected_coarse_cpu_v98 import (
    CoupledAMRCorrectedLeafStateCPUV98,
    validate_coupled_amr_corrected_coarse_state_v98,
)
from .coupled_amr_geometry_v98 import (
    CartesianAMRPatchGeometryV98,
    CoupledAMRJoinedGeometryV98,
    validate_cartesian_amr_patch_geometry_v98,
    validate_joined_amr_geometry_v98,
)
from .coupled_amr_joined_leaf_cpu_v98 import (
    CoupledAMRJoinedLeafStateCPUV98,
    CoupledAMRJoinedLeafTrialCPUV98,
    validate_coupled_amr_joined_leaf_state_v98,
)
from .coupled_amr_register_cpu_v98 import (
    CoupledAMRInterfaceMapCPUV98,
    validate_coupled_amr_coarse_correction_cpu_v98,
    validate_coupled_amr_interface_map_cpu_v98,
)
from .coupled_densin_cpu_v96 import _grid_execution, _v92_state
from .coupled_wetdry_owner_cpu_v97 import (
    CoupledWetDryCPUExecutionV97,
    CoupledWetDryCPUStateV97,
    validate_coupled_wetdry_cpu_state_v97,
)
from .q3_wetdry_adapter_cpu_v97 import (
    CPUQ3V83ActiveWetDryExnerAdapterV97,
)


SCHEMA = 1
SCOPE_ID = "braids.coupled_amr.joined_leaf.diagnostics.cpu.v98"


class CoupledAMRJoinedLeafDiagnosticsV98Error(ValueError):
    """The supplied bounded joined-leaf evidence is inconsistent."""


def _json_array(value: object) -> list:
    array = np.asarray(value)
    if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise CoupledAMRJoinedLeafDiagnosticsV98Error(
            "v98 diagnostic arrays must be finite numerical data")
    return array.tolist()


def _max_abs(value: object) -> float:
    array = np.asarray(value, dtype=np.float64)
    return float(np.max(np.abs(array))) if array.size else 0.0


def _class_inventory(
    value: CoupledWetDryCPUStateV97,
    execution: CoupledWetDryCPUExecutionV97,
) -> np.ndarray:
    v92 = _v92_state(value.active_state)
    area = np.asarray(_grid_execution(
        execution.active_execution).cell_area_m2)
    return np.sum((
        v92.suspended_state.suspended_mass_kg_m2+
        value.active_state.base_state.bed_mass_kg_m2)*area[None, :],
        axis=1, dtype=np.float64)


def _coarse_provisional_class_budget(
    parent: CoupledWetDryCPUStateV97,
    candidate: CoupledWetDryCPUStateV97,
    execution: CoupledWetDryCPUExecutionV97,
) -> dict[str, object]:
    """Reconstruct one ordinary v97 update from its accepted ledgers."""
    validate_coupled_wetdry_cpu_state_v97(parent, execution, None)
    validate_coupled_wetdry_cpu_state_v97(candidate, execution, None)
    p92, c92 = (_v92_state(parent.active_state),
                _v92_state(candidate.active_state))
    p76, c76 = (p92.suspended_state.lifecycle,
                c92.suspended_state.lifecycle)
    p93, c93 = (parent.active_state.base_state,
                candidate.active_state.base_state)
    advective = np.sum(
        c76.advective_boundary_ledger_kg-
        p76.advective_boundary_ledger_kg, axis=1)
    diffusive = np.sum(
        c76.diffusive_boundary_ledger_kg-
        p76.diffusive_boundary_ledger_kg, axis=1)
    water_exchange = (c76.water_exchange_ledger_kg-
                      p76.water_exchange_ledger_kg)
    wetdry_water = (c76.wetdry_water_ledger_kg-
                    p76.wetdry_water_ledger_kg)
    bed_exchange = (c93.suspended_exchange_ledger_kg-
                    p93.suspended_exchange_ledger_kg)
    excluded_exchange = (c93.excluded_water_exchange_kg-
                         p93.excluded_water_exchange_kg)
    suspended_boundary = np.sum(
        c93.suspended_boundary_ledger_kg-
        p93.suspended_boundary_ledger_kg, axis=1)
    q3_boundary = np.sum(
        c93.q3_named_boundary_positive_in_kg-
        p93.q3_named_boundary_positive_in_kg, axis=1)
    q3_depletion = (c93.q3_depletion_roundoff_positive_in_kg-
                    p93.q3_depletion_roundoff_positive_in_kg)
    wetdry_bed = (np.asarray(c93.wetdry_bed_ledger_kg)-
                  np.asarray(p93.wetdry_bed_ledger_kg))
    actual = (_class_inventory(candidate, execution)-
              _class_inventory(parent, execution))
    exchange_selector_net = water_exchange+bed_exchange
    exchange_exclusion_identity = exchange_selector_net+excluded_exchange
    wetdry_net = wetdry_water+wetdry_bed
    expected = (advective+diffusive+water_exchange+wetdry_water+
                bed_exchange+suspended_boundary+q3_boundary+
                q3_depletion+wetdry_bed)
    residual = actual-expected
    external = (advective+diffusive+suspended_boundary+q3_boundary+
                q3_depletion+exchange_selector_net+wetdry_net)
    exclusion_tolerance = 32768.0*np.finfo(np.float64).eps*np.maximum(
        1.0, np.maximum(np.abs(exchange_selector_net),
                        np.abs(excluded_exchange)))
    if np.any(np.abs(exchange_exclusion_identity) > exclusion_tolerance):
        raise CoupledAMRJoinedLeafDiagnosticsV98Error(
            "v98 coarse NeglectEntrainment excluded exchange differs")
    return {
        "actual_total_class_change_kg": _json_array(actual),
        "physical_suspended_advective_kg": _json_array(advective),
        "physical_suspended_diffusive_kg": _json_array(diffusive),
        "water_exchange_kg": _json_array(water_exchange),
        "bed_exchange_kg": _json_array(bed_exchange),
        "water_plus_selected_bed_exchange_net_kg":
            _json_array(exchange_selector_net),
        "excluded_water_exchange_bed_counterpart_kg":
            _json_array(excluded_exchange),
        "exchange_exclusion_identity_residual_kg":
            _json_array(exchange_exclusion_identity),
        "neglect_entrainment": bool(
            execution.active_execution.base_execution.
            neglect_entrainment),
        "wetdry_water_kg": _json_array(wetdry_water),
        "wetdry_bed_kg": _json_array(wetdry_bed),
        "selected_suspended_boundary_to_bed_kg":
            _json_array(suspended_boundary),
        "q3_physical_boundary_positive_in_kg": _json_array(q3_boundary),
        "q3_depletion_roundoff_positive_kg": _json_array(q3_depletion),
        "external_boundary_and_model_source_expected_change_kg":
            _json_array(external),
        "expected_total_class_change_kg": _json_array(expected),
        "ledger_residual_kg": _json_array(residual),
        "ledger_residual_max_abs_kg": _max_abs(residual),
    }


def _fine_substep_class_budget(
    parent: CoupledWetDryCPUStateV97,
    step: object,
    execution: CoupledWetDryCPUExecutionV97,
) -> dict[str, object]:
    """Decompose one private fine update, including its typed interface."""
    actual = (_class_inventory(step.numerical_parent, execution)-
              _class_inventory(parent, execution))
    scalar = step.scalar_stage
    q3 = step.q3_interface_result
    advective = np.sum(
        scalar.advective_boundary_ledger_increment_kg, axis=1)
    diffusive = np.sum(
        scalar.diffusive_boundary_ledger_increment_kg, axis=1)
    water_exchange = np.asarray(scalar.water_exchange_ledger_increment_kg)
    bed_exchange = np.asarray(scalar.bed_exchange_ledger_increment_kg)
    wetdry_water = np.asarray(scalar.wetdry_water_ledger_increment_kg)
    wetdry_bed = np.asarray(scalar.wetdry_bed_ledger_increment_kg)
    scalar_internal = np.sum(
        np.asarray(step.scalar_internal_interface_ledger_kg)[
            :execution.fraction_count], axis=1)
    # Native Q3 stores positive owner-out transfers.  Its classwise identity
    # is sum(cell delta)+named+internal-depletion=0, hence the signs below.
    q3_boundary = -np.sum(q3.named_boundary_transfer_kg, axis=1)
    q3_internal = -np.sum(q3.internal_interface_transfer_kg, axis=1)
    depletion = np.asarray(q3.depletion_roundoff_positive_in_kg)
    exchange_pair_net = water_exchange+bed_exchange
    selection = scalar.suspended_bed_selection
    selected_boundary = np.sum(selection.selected_boundary_ledger_kg, axis=1)
    selected_internal = np.sum(selection.selected_internal_interface_ledger_kg, axis=1)
    wetdry_pair_net = wetdry_water+wetdry_bed
    expected = (advective+diffusive+water_exchange+bed_exchange+
                wetdry_water+wetdry_bed+scalar_internal+q3_boundary+
                q3_internal+depletion+selected_boundary+selected_internal)
    residual = actual-expected
    external = (advective+diffusive+q3_boundary+depletion+
                exchange_pair_net+wetdry_pair_net+selected_boundary)
    internal = scalar_internal+q3_internal+selected_internal
    return {
        "substep_index": int(step.substep_index),
        "actual_total_class_change_kg": _json_array(actual),
        "physical_suspended_advective_kg": _json_array(advective),
        "physical_suspended_diffusive_kg": _json_array(diffusive),
        "water_exchange_kg": _json_array(water_exchange),
        "bed_exchange_kg": _json_array(bed_exchange),
        "selected_suspended_boundary_to_bed_kg": _json_array(selected_boundary),
        "selected_suspended_internal_to_bed_kg": _json_array(selected_internal),
        "excluded_water_exchange_bed_counterpart_kg": _json_array(
            selection.excluded_water_exchange_bed_counterpart_kg),
        "water_plus_bed_exchange_net_kg":
            _json_array(exchange_pair_net),
        "wetdry_water_kg": _json_array(wetdry_water),
        "wetdry_bed_kg": _json_array(wetdry_bed),
        "scalar_internal_interface_mass_in_kg":
            _json_array(scalar_internal),
        "q3_physical_boundary_mass_in_kg": _json_array(q3_boundary),
        "q3_internal_interface_mass_in_kg": _json_array(q3_internal),
        "q3_depletion_roundoff_positive_kg": _json_array(depletion),
        "external_boundary_and_model_source_expected_change_kg":
            _json_array(external),
        "internal_interface_expected_change_kg": _json_array(internal),
        "expected_total_class_change_kg": _json_array(expected),
        "ledger_residual_kg": _json_array(residual),
        "ledger_residual_max_abs_kg": _max_abs(residual),
    }


def _exact_state_images(
    value: CoupledWetDryCPUStateV97,
    execution: CoupledWetDryCPUExecutionV97,
) -> dict[str, object]:
    validate_coupled_wetdry_cpu_state_v97(value, execution, None)
    v92 = _v92_state(value.active_state)
    return {
        "state_sha256": value.state_sha256,
        "accepted_updates": value.accepted_updates,
        "time_s": value.time_s,
        "q": _json_array(value.q),
        "suspended_mass_kg_m2": _json_array(
            v92.suspended_state.suspended_mass_kg_m2),
        "bed_mass_kg_m2": _json_array(
            value.active_state.base_state.bed_mass_kg_m2),
        "h_temperature_c_m": _json_array(v92.h_temperature_c_m),
        "h_salinity_ppt_m": _json_array(v92.h_salinity_ppt_m),
    }


def _corrected_state_images(
    value: CoupledAMRCorrectedLeafStateCPUV98,
) -> dict[str, object]:
    arrays = (
        value.q, value.suspended_mass_kg_m2, value.bed_mass_kg_m2,
        value.h_temperature_c_m, value.h_salinity_ppt_m,
    )
    if any(not np.isfinite(np.asarray(array)).all() for array in arrays):
        raise CoupledAMRJoinedLeafDiagnosticsV98Error(
            "v98 corrected diagnostic state contains non-finite data")
    return {
        "state_sha256": value.state_sha256,
        "accepted_updates": value.accepted_updates,
        "time_s": value.time_s,
        "q": _json_array(value.q),
        "suspended_mass_kg_m2": _json_array(
            value.suspended_mass_kg_m2),
        "bed_mass_kg_m2": _json_array(value.bed_mass_kg_m2),
        "h_temperature_c_m": _json_array(value.h_temperature_c_m),
        "h_salinity_ppt_m": _json_array(value.h_salinity_ppt_m),
    }


def _corrected_class_inventory(
    value: CoupledAMRCorrectedLeafStateCPUV98,
    execution: CoupledWetDryCPUExecutionV97,
) -> np.ndarray:
    area = np.asarray(_grid_execution(
        execution.active_execution).cell_area_m2)
    return np.sum((value.suspended_mass_kg_m2+value.bed_mass_kg_m2)*
                  area[None, :], axis=1, dtype=np.float64)


def _provider_view(
    execution: CoupledWetDryCPUExecutionV97,
    template: CPUQ3V83ActiveWetDryExnerAdapterV97,
    state: object,
) -> CPUQ3V83ActiveWetDryExnerAdapterV97:
    if type(template) is not CPUQ3V83ActiveWetDryExnerAdapterV97:
        raise CoupledAMRJoinedLeafDiagnosticsV98Error(
            "v98 diagnostic provider template type differs")
    result = CPUQ3V83ActiveWetDryExnerAdapterV97(
        outer_execution=execution.active_execution,
        binding=template.binding, state=state,
        native_factcr_runtime=template.native_factcr_runtime,
        native_q3_face_runtime=template.native_q3_face_runtime,
        native_threads=template.native_factcr_threads)
    if (result.state is not state or result._open is not None or
            result._prepared is not None or result._closed is not None):
        raise CoupledAMRJoinedLeafDiagnosticsV98Error(
            "v98 diagnostic provider view differs")
    return result


def _phase_diagnostics(phase: object, ng: int) -> dict[str, object]:
    coarse = np.asarray(phase.coarse_integral, dtype=np.float64)
    accepted = np.sum(
        np.asarray(phase.fine_integral_slots, dtype=np.float64),
        axis=(0, 1), dtype=np.float64)
    correction = coarse-accepted
    residual = -coarse+correction+accepted
    if (not np.array_equal(accepted, phase.accepted_fine_integral) or
            not np.array_equal(correction,
                               phase.coarse_minus_fine_integral)):
        raise CoupledAMRJoinedLeafDiagnosticsV98Error(
            "v98 diagnostic register reconstruction differs")
    tolerance = 64.0*np.finfo(np.float64).eps*np.maximum(
        1.0, np.maximum(np.abs(-coarse+correction), np.abs(accepted)))
    if np.any(np.abs(residual) > tolerance):
        raise CoupledAMRJoinedLeafDiagnosticsV98Error(
            "v98 diagnostic register replacement does not close")
    result = {
        "phase": phase.phase,
        "lane_ids": list(phase.lane_ids),
        "coarse_integral": _json_array(coarse),
        "accepted_fine_integral": _json_array(accepted),
        "coarse_minus_fine_integral": _json_array(correction),
        "replacement_closure_residual": _json_array(residual),
        "replacement_closure_max_abs": _max_abs(residual),
        "fine_slot_consumption_count": _json_array(
            phase.fine_slot_consumption_count),
        "receipt_sha256": phase.receipt_sha256,
    }
    if phase.phase == "hydraulic":
        result["water_residual_abs"] = float(abs(residual[0]))
        result["momentum_residual_max_abs"] = _max_abs(residual[1:3])
    else:
        result["suspended_residual_max_abs"] = _max_abs(residual[:ng])
        result["fluid_residual_max_abs"] = _max_abs(
            residual[ng:ng+2])
        result["bed_residual_max_abs"] = _max_abs(residual[ng+2:])
    return result


def _macro_diagnostics(
    trial: CoupledAMRJoinedLeafTrialCPUV98,
    state: CoupledAMRJoinedLeafStateCPUV98,
    coarse_execution: CoupledWetDryCPUExecutionV97,
    fine_execution: CoupledWetDryCPUExecutionV97,
    interface: CoupledAMRInterfaceMapCPUV98,
    coarse_provider_template: CPUQ3V83ActiveWetDryExnerAdapterV97,
    coarse_parent: CoupledWetDryCPUStateV97,
    fine_parent: CoupledWetDryCPUStateV97,
) -> dict[str, object]:
    if (type(trial) is not CoupledAMRJoinedLeafTrialCPUV98 or
            type(state) is not CoupledAMRJoinedLeafStateCPUV98 or
            trial.candidate.state_sha256 != state.state_sha256 or
            trial.candidate.accepted_macro_updates !=
                state.accepted_macro_updates):
        raise CoupledAMRJoinedLeafDiagnosticsV98Error(
            "v98 diagnostic trial/state pair differs")
    validate_coupled_amr_joined_leaf_state_v98(
        state, coarse_execution, fine_execution, interface,
        coarse_provider=None, fine_provider=None)
    validate_coupled_amr_coarse_correction_cpu_v98(
        trial.correction, interface, trial.coarse_integrals,
        (trial.fine_step1.face_time_integrals,
         trial.fine_step2.face_time_integrals),
        coarse_execution=coarse_execution,
        fine_execution=fine_execution)
    provider_view = _provider_view(
        coarse_execution, coarse_provider_template,
        state.corrected_coarse.q3_state)
    validate_coupled_amr_corrected_coarse_state_v98(
        state.corrected_coarse, coarse_execution, interface,
        trial.correction, provider=provider_view)
    validate_coupled_wetdry_cpu_state_v97(
        state.coarse_numerical_parent, coarse_execution, None)
    validate_coupled_wetdry_cpu_state_v97(
        state.fine_numerical_parent, fine_execution, None)
    register = trial.correction.scheduled_register
    ledger = trial.correction.internal_interface_ledger
    coarse_budget = _coarse_provisional_class_budget(
        coarse_parent, state.corrected_coarse.provisional_source_state,
        coarse_execution)
    fine_budget1 = _fine_substep_class_budget(
        fine_parent, trial.fine_step1, fine_execution)
    fine_budget2 = _fine_substep_class_budget(
        trial.fine_step1.numerical_parent, trial.fine_step2,
        fine_execution)
    correction_total = (np.asarray(ledger.suspended_mass_kg)+
                        np.asarray(ledger.bed_mass_kg))
    actual_global = (
        _corrected_class_inventory(
            state.corrected_coarse, coarse_execution)-
        _class_inventory(coarse_parent, coarse_execution)+
        _class_inventory(state.fine_numerical_parent, fine_execution)-
        _class_inventory(fine_parent, fine_execution))
    expected_all = (
        np.asarray(coarse_budget["expected_total_class_change_kg"])+
        np.asarray(fine_budget1["expected_total_class_change_kg"])+
        np.asarray(fine_budget2["expected_total_class_change_kg"])+
        correction_total)
    external_expected = (
        np.asarray(coarse_budget[
            "external_boundary_and_model_source_expected_change_kg"])+
        np.asarray(fine_budget1[
            "external_boundary_and_model_source_expected_change_kg"])+
        np.asarray(fine_budget2[
            "external_boundary_and_model_source_expected_change_kg"]))
    internal_cancellation = (
        correction_total+
        np.asarray(fine_budget1[
            "internal_interface_expected_change_kg"])+
        np.asarray(fine_budget2[
            "internal_interface_expected_change_kg"]))
    residual = actual_global-expected_all
    external_residual = actual_global-external_expected
    tolerance = 262144.0*np.finfo(np.float64).eps*np.maximum(
        1.0, np.maximum(np.abs(actual_global), np.abs(expected_all)))
    component_max = max(
        coarse_budget["ledger_residual_max_abs_kg"],
        fine_budget1["ledger_residual_max_abs_kg"],
        fine_budget2["ledger_residual_max_abs_kg"])
    if (np.any(np.abs(residual) > tolerance) or
            np.any(np.abs(internal_cancellation) > tolerance) or
            np.any(np.abs(external_residual) > tolerance) or
            component_max > float(np.max(tolerance))):
        raise CoupledAMRJoinedLeafDiagnosticsV98Error(
            "v98 joined diagnostic class inventory ledgers do not close")
    return {
        "accepted_macro_update": state.accepted_macro_updates,
        "time_s": state.time_s,
        "trial_sha256": trial.trial_sha256,
        "state_sha256": state.state_sha256,
        "correction_receipt_sha256": trial.correction.receipt_sha256,
        "fine_substep_state_sha256": list(
            trial.fine_substep_state_sha256),
        "hydraulic_register": _phase_diagnostics(
            register.hydraulic, coarse_execution.fraction_count),
        "scalar_bed_register": _phase_diagnostics(
            register.scalar_bed, coarse_execution.fraction_count),
        "internal_interface_ledger": {
            "water_volume_m3": float(ledger.water_volume_m3),
            "momentum_m4_s": _json_array(ledger.momentum_m4_s),
            "suspended_mass_kg": _json_array(
                ledger.suspended_mass_kg),
            "fluid_integral": _json_array(ledger.fluid_integral),
            "bed_mass_kg": _json_array(ledger.bed_mass_kg),
            "physical_boundary_ledger_count":
                ledger.physical_boundary_ledger_count,
            "application_count": ledger.application_count,
            "receipt_sha256": ledger.receipt_sha256,
        },
        "class_inventory_budget": {
            "coarse_provisional": coarse_budget,
            "fine_substeps": [fine_budget1, fine_budget2],
            "coarse_register_correction_total_kg":
                _json_array(correction_total),
            "actual_global_total_class_change_kg":
                _json_array(actual_global),
            "expected_all_accepted_ledgers_kg":
                _json_array(expected_all),
            "accepted_ledger_residual_kg": _json_array(residual),
            "accepted_ledger_residual_max_abs_kg": _max_abs(residual),
            "external_boundary_and_model_source_expected_change_kg":
                _json_array(external_expected),
            "internal_interface_cancellation_residual_kg":
                _json_array(internal_cancellation),
            "external_budget_residual_kg":
                _json_array(external_residual),
            "external_budget_residual_max_abs_kg":
                _max_abs(external_residual),
            "component_ledger_residual_max_abs_kg": component_max,
            "morphological_factor": float(
                coarse_execution.active_execution.base_execution.
                morphological_factor),
        },
    }


def _array_differences(left: dict[str, object],
                       right: dict[str, object]) -> dict[str, float]:
    keys = ("q", "suspended_mass_kg_m2", "bed_mass_kg_m2",
            "h_temperature_c_m", "h_salinity_ppt_m")
    return {key+"_max_abs": _max_abs(
        np.asarray(left[key])-np.asarray(right[key])) for key in keys}


def collect_coupled_amr_joined_leaf_diagnostics_v98(
    coarse_execution: CoupledWetDryCPUExecutionV97,
    fine_execution: CoupledWetDryCPUExecutionV97,
    interface: CoupledAMRInterfaceMapCPUV98,
    joined_geometry: CoupledAMRJoinedGeometryV98,
    coarse_patch: CartesianAMRPatchGeometryV98,
    fine_patch: CartesianAMRPatchGeometryV98, *,
    initial_coarse: CoupledWetDryCPUStateV97,
    initial_fine: CoupledWetDryCPUStateV97,
    macro1_trial: CoupledAMRJoinedLeafTrialCPUV98,
    macro1_state: CoupledAMRJoinedLeafStateCPUV98,
    macro2_trial: CoupledAMRJoinedLeafTrialCPUV98,
    macro2_state: CoupledAMRJoinedLeafStateCPUV98,
    restarted_macro2_trial: CoupledAMRJoinedLeafTrialCPUV98,
    restarted_macro2_state: CoupledAMRJoinedLeafStateCPUV98,
    coarse_provider_template: CPUQ3V83ActiveWetDryExnerAdapterV97,
    restarted_coarse_provider_template:
        CPUQ3V83ActiveWetDryExnerAdapterV97,
) -> dict[str, object]:
    """Collect validated, JSON-ready numerical evidence for <=2 macros."""
    if (type(coarse_execution) is not CoupledWetDryCPUExecutionV97 or
            type(fine_execution) is not CoupledWetDryCPUExecutionV97):
        raise CoupledAMRJoinedLeafDiagnosticsV98Error(
            "v98 diagnostics require exact CPU v97 executions")
    validate_coupled_amr_interface_map_cpu_v98(
        interface, coarse_execution, fine_execution)
    coarse_grid = _grid_execution(coarse_execution.active_execution)
    fine_grid = _grid_execution(fine_execution.active_execution)
    validate_cartesian_amr_patch_geometry_v98(coarse_patch, coarse_grid)
    validate_cartesian_amr_patch_geometry_v98(fine_patch, fine_grid)
    validate_joined_amr_geometry_v98(
        joined_geometry, coarse_grid, fine_grid, coarse_patch, fine_patch)
    if (joined_geometry.interface_fingerprint != interface.fingerprint or
            joined_geometry.receipt_sha256 !=
                macro1_state.joined_geometry_receipt_sha256 or
            joined_geometry.receipt_sha256 !=
                macro2_state.joined_geometry_receipt_sha256):
        raise CoupledAMRJoinedLeafDiagnosticsV98Error(
            "v98 joined diagnostic geometry differs from accepted states")

    initial_coarse_image = _exact_state_images(
        initial_coarse, coarse_execution)
    initial_fine_image = _exact_state_images(initial_fine, fine_execution)
    macro1 = _macro_diagnostics(
        macro1_trial, macro1_state, coarse_execution, fine_execution,
        interface, coarse_provider_template, initial_coarse, initial_fine)
    macro2 = _macro_diagnostics(
        macro2_trial, macro2_state, coarse_execution, fine_execution,
        interface, coarse_provider_template,
        macro1_state.coarse_numerical_parent,
        macro1_state.fine_numerical_parent)
    restart2 = _macro_diagnostics(
        restarted_macro2_trial, restarted_macro2_state,
        coarse_execution, fine_execution, interface,
        restarted_coarse_provider_template,
        macro1_state.coarse_numerical_parent,
        macro1_state.fine_numerical_parent)
    if (macro1_state.accepted_macro_updates != 1 or
            macro2_state.accepted_macro_updates != 2 or
            restarted_macro2_state.accepted_macro_updates != 2 or
            macro2_state.previous_state_sha256 != macro1_state.state_sha256 or
            restarted_macro2_state.previous_state_sha256 !=
                macro1_state.state_sha256 or
            macro2_state.state_sha256 != restarted_macro2_state.state_sha256 or
            macro2_trial.correction.receipt_sha256 !=
                restarted_macro2_trial.correction.receipt_sha256):
        raise CoupledAMRJoinedLeafDiagnosticsV98Error(
            "v98 bounded direct/restart chain is not exact")

    coarse1 = _corrected_state_images(macro1_state.corrected_coarse)
    coarse2 = _corrected_state_images(macro2_state.corrected_coarse)
    coarse_restart2 = _corrected_state_images(
        restarted_macro2_state.corrected_coarse)
    fine1 = _exact_state_images(
        macro1_state.fine_numerical_parent, fine_execution)
    fine2 = _exact_state_images(
        macro2_state.fine_numerical_parent, fine_execution)
    fine_restart2 = _exact_state_images(
        restarted_macro2_state.fine_numerical_parent, fine_execution)
    coarse_diff = _array_differences(coarse2, coarse_restart2)
    fine_diff = _array_differences(fine2, fine_restart2)
    if any(value != 0.0 for value in (*coarse_diff.values(),
                                      *fine_diff.values())):
        raise CoupledAMRJoinedLeafDiagnosticsV98Error(
            "v98 direct/restart conserved images differ")

    result = {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "evidence_kind": "bounded_native_cpu_joined_leaf_observation",
        "scope_limit": "one_joined_leaf_interface_not_full_regrid",
        "accepted_macro_update_count": 2,
        "private_fine_substep_count": 4,
        "geometry": {
            "coarse_patch": {
                "receipt_sha256": coarse_patch.receipt_sha256,
                "cell_bounds_m": _json_array(coarse_patch.cell_bounds_m),
                "cell_global_ids": _json_array(
                    coarse_patch.cell_global_ids),
                "face_global_ids": _json_array(
                    coarse_patch.face_global_ids),
            },
            "fine_patch": {
                "receipt_sha256": fine_patch.receipt_sha256,
                "cell_bounds_m": _json_array(fine_patch.cell_bounds_m),
                "cell_global_ids": _json_array(fine_patch.cell_global_ids),
                "face_global_ids": _json_array(fine_patch.face_global_ids),
            },
            "joined": {
                "receipt_sha256": joined_geometry.receipt_sha256,
                "coarse_face": joined_geometry.coarse_face,
                "fine_faces": list(joined_geometry.fine_faces),
                "coarse_global_face_id": interface.coarse_global_face_id,
                "fine_global_face_ids": list(
                    interface.fine_global_face_ids),
                "coarse_owner_cell": joined_geometry.coarse_owner_cell,
                "fine_owner_cells": _json_array(
                    joined_geometry.fine_owner_cells),
                "normal_centre_distance_m": _json_array(
                    joined_geometry.normal_centre_distance_m),
                "tangential_centre_offset_m": _json_array(
                    joined_geometry.tangential_centre_offset_m),
                "covered_parent_global_id":
                    joined_geometry.covered_parent_global_id,
                "covered_parent_bounds_m": _json_array(
                    joined_geometry.covered_parent_bounds_m),
                "covered_child_cells": _json_array(
                    joined_geometry.covered_child_cells),
                "covered_child_global_ids": _json_array(
                    joined_geometry.covered_child_global_ids),
                "covered_parent_area_m2":
                    joined_geometry.covered_parent_area_m2,
                "covered_parent_accepted_row_count": 0,
            },
        },
        "evolution": {
            "labels": ["initial", "macro_1", "macro_2"],
            "time_s": [initial_coarse.time_s, macro1_state.time_s,
                       macro2_state.time_s],
            "coarse": [initial_coarse_image, coarse1, coarse2],
            "fine": [initial_fine_image, fine1, fine2],
        },
        "macro_registers": [macro1, macro2],
        "cumulative_class_inventory_budget": {
            "actual_total_class_change_kg": _json_array(
                np.asarray(macro1["class_inventory_budget"]
                           ["actual_global_total_class_change_kg"])+
                np.asarray(macro2["class_inventory_budget"]
                           ["actual_global_total_class_change_kg"])),
            "expected_boundary_and_model_source_change_kg": _json_array(
                np.asarray(macro1["class_inventory_budget"]
                           ["external_boundary_and_model_source_expected_change_kg"])+
                np.asarray(macro2["class_inventory_budget"]
                           ["external_boundary_and_model_source_expected_change_kg"])),
            "internal_interface_cancellation_residual_kg": _json_array(
                np.asarray(macro1["class_inventory_budget"]
                           ["internal_interface_cancellation_residual_kg"])+
                np.asarray(macro2["class_inventory_budget"]
                           ["internal_interface_cancellation_residual_kg"])),
            "accepted_ledger_residual_kg": _json_array(
                np.asarray(macro1["class_inventory_budget"]
                           ["accepted_ledger_residual_kg"])+
                np.asarray(macro2["class_inventory_budget"]
                           ["accepted_ledger_residual_kg"])),
            "budget_kind":
                "accepted_boundary_exchange_wetdry_q3_depletion_and_internal_register",
            "raw_change_is_not_expected_to_be_zero_when_external_or_stock_ledgers_are_nonzero":
                True,
        },
        "cumulative_internal_interface_ledger": {
            "water_volume_m3": macro2_state.cumulative_water_volume_m3,
            "momentum_m4_s": _json_array(
                macro2_state.cumulative_momentum_m4_s),
            "suspended_mass_kg": _json_array(
                macro2_state.cumulative_suspended_mass_kg),
            "fluid_integral": _json_array(
                macro2_state.cumulative_fluid_integral),
            "bed_mass_kg": _json_array(
                macro2_state.cumulative_bed_mass_kg),
        },
        "direct_vs_checkpoint_restart": {
            "state_sha256_equal": True,
            "correction_receipt_sha256_equal": True,
            "direct_state_sha256": macro2_state.state_sha256,
            "restart_state_sha256": restarted_macro2_state.state_sha256,
            "coarse_max_abs_differences": coarse_diff,
            "fine_max_abs_differences": fine_diff,
            "restart_macro": restart2,
        },
        "provenance": {
            "coarse_execution_fingerprint": coarse_execution.fingerprint,
            "fine_execution_fingerprint": fine_execution.fingerprint,
            "interface_fingerprint": interface.fingerprint,
            "macro1_trial_sha256": macro1_trial.trial_sha256,
            "macro2_trial_sha256": macro2_trial.trial_sha256,
            "restarted_macro2_trial_sha256":
                restarted_macro2_trial.trial_sha256,
        },
        "state_or_clock_publication_count": 0,
    }
    # This also guarantees that callers can write strict JSON without a
    # custom NumPy encoder or silently serializing NaN/Infinity.
    json.dumps(result, sort_keys=True, allow_nan=False)
    return result


__all__ = [
    "SCHEMA", "SCOPE_ID", "CoupledAMRJoinedLeafDiagnosticsV98Error",
    "collect_coupled_amr_joined_leaf_diagnostics_v98",
]
