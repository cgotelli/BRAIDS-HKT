"""M2-C CUDA ownership and identity for implicit law-11 suspension AMR.

The selected topology remains a synchronous fixed-layout implicit solve; this
module owns only the conservative phase transaction required when an accepted
state becomes hydraulically dry or is installed after a topology transfer.
It never selects a backend and has no CPU numerical fallback.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from braids_core.sediment.channel_composition import (
    TRA_COMBINE,
    channel_composition_profile_document,
    resolve_channel_composition_profile,
)
from braids_core.sediment.suspended_amr_contract import (
    SUSPENDED_AMR_METHOD_ID,
    SUSPENDED_AMR_SEMANTIC_ID,
    load_suspended_amr_contract,
)
from braids_core.sediment.suspension_profile import load_suspension_profile


HERE = Path(__file__).resolve().parent
CONTRACT_FIELD = "suspended_amr_contract"
SEMANTIC_FIELD = "suspended_amr_contract_semantic_id"
FINGERPRINT_FIELD = "suspended_amr_contract_fingerprint"
PREPARED_BINDING_FIELD = "implicit_suspension_amr_binding"
CUDA_OPTIONS = ("--std=c++11", "--fmad=false", "--prec-div=true",
                "--prec-sqrt=true", "--ftz=false")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def _authenticated_v49_header(deck_path, metadata, contract, profile):
    """Match the descriptive sidecar to the immutable physical v49 prefix."""
    _require(deck_path is not None,
             "M2-C metadata binding requires physical v49 deck authentication")
    path = Path(deck_path).resolve()
    _require(path.is_file() and not path.is_symlink(),
             "M2-C physical v49 deck is missing or linked")
    with path.open("r", encoding="ascii") as handle:
        lines = [handle.readline().strip() for _ in range(6)]
    _require(len(lines) == 6 and all(lines) and lines[0] == "49",
             "M2-C physical deck lacks the exact v49 prefix")
    values = lines[5].split()
    _require(len(values) == 8,
             "M2-C physical v49 numeric contract differs")
    try:
        numeric = (int(values[0]), float(values[1]), float(values[2]),
                   int(values[3]), int(values[4]), int(values[5]),
                   float(values[6]), float(values[7]))
    except ValueError as exc:
        raise ValueError("M2-C physical v49 numeric contract differs") from exc
    expected = (contract.fraction_count,
                contract.hydraulic_dry_tolerance_m,
                contract.diffusion_dry_threshold_m,
                int(contract.amr_enabled), contract.amr_levels,
                contract.amr_refinement_ratio,
                contract.amr_wet_depth_m,
                contract.amr_regrid_interval_s)
    _require(lines[1] == metadata.get("fingerprint") and
             lines[2] == contract.semantic_id and
             lines[3] == contract.fingerprint and
             lines[4] == profile.fingerprint and numeric == expected,
             "M2-C metadata differs from its physical v49 deck prefix")


def prepared_binding_from_case_metadata(metadata: Mapping[str, object],
                                        *, deck_path=None):
    """Extract the authenticated v49 profile used by selected AMR operators.

    Native selected-geometry records intentionally contain only numeric
    arrays.  The ordinary case metadata is already deck-fingerprint-bound and
    owns the typed capacity/suspension documents and route-level AMR
    instance.  This compact JSON binding makes those identities portable with
    every on-demand operator package instead of reconstructing them from
    topology-local values.
    """
    _require(isinstance(metadata, Mapping),
             "M2-C case metadata must be a mapping")
    raw_contract = metadata.get("suspended_amr_ownership")
    if raw_contract is None:
        return None
    contract = load_suspended_amr_contract(raw_contract)
    profile = load_suspension_profile(metadata.get("suspension_profile"))
    _authenticated_v49_header(deck_path, metadata, contract, profile)
    _require(metadata.get("format") == "xbraids_cutcell-v49" and
             contract.suspension_profile_fingerprint == profile.fingerprint and
             contract.amr_enabled and contract.fraction_count == 1,
             "M2-C case metadata/profile/AMR identity differs")
    amr = metadata.get("amr")
    _require(isinstance(amr, Mapping) and
             int(amr.get("levels", -1)) == contract.amr_levels and
             int(amr.get("refinement_ratio", -1)) ==
                contract.amr_refinement_ratio and
             float(amr.get("wet_depth_m", np.nan)) ==
                contract.amr_wet_depth_m and
             float(amr.get("regrid_interval_s", np.nan)) ==
                contract.amr_regrid_interval_s and
             float(metadata.get("dry_tolerance_m", np.nan)) ==
                contract.hydraulic_dry_tolerance_m,
             "M2-C case hierarchy differs from its ownership instance")
    constitutive = metadata.get("constitutive_contract")
    roughness = metadata.get("hydraulic_roughness")
    manning = metadata.get("manning")
    flow_boundary = metadata.get("flow2dh_open_boundary")
    ids = metadata.get("string_to_numeric_id")
    _require(isinstance(constitutive, Mapping) and
             isinstance(roughness, Mapping) and
             isinstance(manning, Mapping) and
             isinstance(flow_boundary, Mapping) and
             isinstance(ids, Mapping),
             "M2-C case metadata lacks typed constitutive/boundary controls")
    _require(roughness.get("mode") == "manning" and
             roughness.get("white_colebrook_chzk2d") is False,
             "M2-C implicit GPU route currently requires Manning roughness")
    ordered = sorted(ids, key=lambda name: int(ids[name]))
    _require([int(ids[name]) for name in ordered] ==
             list(range(1, len(ordered)+1)),
             "M2-C logical-boundary IDs are not contiguous")
    return_times = flow_boundary.get("return_time_s")
    _require(isinstance(return_times, Mapping) and
             set(return_times) == set(ordered),
             "M2-C boundary return-time inventory differs")
    clear = constitutive.get("settling_clear_water_velocity_m_s")
    _require(isinstance(clear, list) and len(clear) == 1,
             "M2-C clear-water settling inventory differs")
    friction_work_depth = float(
        manning.get("friction_chezy_minimum_depth_m", 0.0))
    typed_work_depth = float(roughness["minimum_work_depth_m"])
    _require(np.isfinite(friction_work_depth) and
             friction_work_depth >= 0.0 and
             np.isfinite(typed_work_depth) and typed_work_depth >= 0.0,
             "M2-C roughness/friction work-depth controls differ")
    body = dict(
        schema=1,
        capacity_runtime_profile=profile.capacity_profile.as_dict(),
        suspension_runtime_profile=profile.as_dict(),
        suspended_amr_contract=contract.as_dict(),
        capacity_kinematic_viscosity_m2_s=float(
            constitutive["kinematic_viscosity_m2_s"]),
        # Capacity consumes the effective work depth, while the native
        # receipt authenticates the typed roughness and legacy Manning
        # friction floors independently.
        capacity_roughness_work_depth_m=max(
            typed_work_depth, friction_work_depth),
        capacity_hydraulic_roughness_mode="manning",
        capacity_white_colebrook_chzk2d=False,
        settling_clear_water_velocity_m_s=[float(clear[0])],
        boundary_names=ordered,
        boundary_return_time_s=[float(return_times[name])
                                for name in ordered],
    )
    _require(all(np.isfinite(value) and value >= 0.0 for value in
                 (*body["settling_clear_water_velocity_m_s"],
                  *body["boundary_return_time_s"]) ) and
             np.isfinite(body["capacity_kinematic_viscosity_m2_s"]) and
             body["capacity_kinematic_viscosity_m2_s"] > 0.0 and
             np.isfinite(body["capacity_roughness_work_depth_m"]) and
             body["capacity_roughness_work_depth_m"] >= 0.0,
             "M2-C prepared numeric binding differs")
    return {**body, "fingerprint": hashlib.sha256(_canonical(body)).hexdigest()}


def apply_prepared_binding(data: dict, binding: Mapping[str, object] | None):
    """Bind a portable v49 route instance to one selected local topology."""
    if binding is None:
        return data
    _require(isinstance(data, dict) and isinstance(binding, Mapping),
             "M2-C prepared binding must target a mutable operator mapping")
    expected = {"schema", "capacity_runtime_profile",
        "suspension_runtime_profile", "suspended_amr_contract",
        "capacity_kinematic_viscosity_m2_s",
        "capacity_roughness_work_depth_m",
        "capacity_hydraulic_roughness_mode",
        "capacity_white_colebrook_chzk2d",
        "settling_clear_water_velocity_m_s", "boundary_names",
        "boundary_return_time_s", "fingerprint"}
    _require(set(binding) == expected and binding.get("schema") == 1,
             "M2-C prepared binding fields differ")
    body = {key: binding[key] for key in binding if key != "fingerprint"}
    _require(binding["fingerprint"] ==
             hashlib.sha256(_canonical(body)).hexdigest(),
             "M2-C prepared binding fingerprint differs")
    profile = load_suspension_profile(binding["suspension_runtime_profile"])
    contract = load_suspended_amr_contract(binding["suspended_amr_contract"])
    nc, ng, nb = (int(data[name]) for name in ("nc", "nfrac", "nb"))
    dry = float(data.get("dry", data.get("dry_tolerance", np.nan)))
    clear = np.asarray(binding["settling_clear_water_velocity_m_s"],
                       dtype=np.float64)
    return_time = np.asarray(binding["boundary_return_time_s"],
                             dtype=np.float64)
    native_return_time = np.asarray(
        data.get("native_suspended_boundary_return_time_s"))
    native_viscosity = np.asarray(
        data.get("native_constitutive_kinematic_viscosity_m2_s"))
    native_work_depth = np.asarray(
        data.get("native_roughness_minimum_work_depth_m"))
    native_roughness_mode = np.asarray(
        data.get("native_hydraulic_roughness_mode"))
    native_white_colebrook = np.asarray(
        data.get("native_white_colebrook_chzk2d"))
    _require(int(data.get("bedlaw", -1)) == 11 and
             int(data.get("exchange_mode", -1)) == 4 and
             int(data.get("secondary_flow_model", -1)) in (0, 1) and
             ng == 1 and nc > 0 and
             int(data.get("amr_enabled", -1)) == 0 and
             contract.amr_enabled and contract.fraction_count == ng and
             contract.suspension_profile_fingerprint == profile.fingerprint and
             dry == contract.hydraulic_dry_tolerance_m and
             clear.shape == (ng,) and return_time.shape == (nb,) and
             np.array_equal(np.asarray(data["d50_fraction"]),
                            np.asarray(profile.sediment_d50_m)) and
             np.array_equal(np.asarray(data["rhos_fraction"]),
                            np.asarray(profile.sediment_density_kg_m3)) and
             float(data["rho"]) == profile.water_density_kg_m3 and
             float(data["grav"]) == profile.gravity_m_s2 and
             np.array_equal(np.asarray(data["sedthr_fraction"]),
                            np.asarray([profile.sediment_threshold_m])) and
             np.array_equal(np.asarray(data["ws_fraction"]), clear) and
             native_return_time.dtype == np.float64 and
             native_return_time.shape == (nb,) and
             np.array_equal(native_return_time, return_time) and
             native_viscosity.dtype == np.float64 and
             native_viscosity.shape == () and
             native_viscosity.item() ==
                float(binding["capacity_kinematic_viscosity_m2_s"]) and
             native_work_depth.dtype == np.float64 and
             native_work_depth.shape == () and
             max(native_work_depth.item(), float(data[
                 "manning_friction_chezy_minimum_depth"])) ==
                float(binding["capacity_roughness_work_depth_m"]) and
             binding["capacity_hydraulic_roughness_mode"] == "manning" and
             binding["capacity_white_colebrook_chzk2d"] is False and
             native_roughness_mode.shape == () and
             native_roughness_mode.dtype.kind in "iu" and
             int(native_roughness_mode) == 2 and
             native_white_colebrook.shape == () and
             native_white_colebrook.dtype.kind in "iu" and
             int(native_white_colebrook) == 0,
             "M2-C selected operator differs from ordinary v49 preparation")
    fraction_ids = [value.fraction_id
                    for value in profile.capacity_profile.fractions]
    channel = resolve_channel_composition_profile(
        fraction_ids, [TRA_COMBINE] * len(fraction_ids),
        bed_factor=float(data["transport_calibration"]),
        suspended_factor=profile.suspended_calibration_factor,
        epsilon=1.0e-6)
    data.update(
        m1_profile_runtime_schema=np.int32(2),
        m1_profile_research_override="research_law11_vertical_slice_v1",
        capacity_suspended_route="general_capacity_galappatti_v1",
        capacity_roughness_method_id="roughness.manning",
        capacity_runtime_profile=profile.capacity_profile.as_dict(),
        capacity_profile_fingerprint=profile.capacity_profile.fingerprint,
        capacity_fraction_ids=fraction_ids,
        channel_composition_profile=
            channel_composition_profile_document(channel),
        channel_composition_fingerprint=channel.fingerprint,
        capacity_kinematic_viscosity_m2_s=np.float64(
            binding["capacity_kinematic_viscosity_m2_s"]),
        capacity_roughness_work_depth_m=np.float64(
            binding["capacity_roughness_work_depth_m"]),
        suspension_runtime_profile=profile.as_dict(),
        suspension_profile_fingerprint=profile.fingerprint,
        capacity_fraction_basis=profile.fraction_basis,
        ifractions_code=np.int32(profile.fraction_basis_code),
        factsd=np.float64(profile.fac_tsd),
        settlefrm_code=np.int32(profile.settling_form),
        settling_fac_dss=np.float64(profile.settling_fac_dss[0]),
        settling_temperature_mode_code=np.int32(profile.temperature_mode_code),
        settling_temperature_c=np.float64(profile.temperature_value_c),
        settling_reference_concentration_kg_m3=np.float64(
            profile.settling_reference_concentration_kg_m3),
        suspended_calibration_factor=np.float64(
            profile.suspended_calibration_factor),
        sediment_activity_contract_code=np.int32(1),
        wet_mask_provider_code=np.int32(1),
        settling_clear_water_velocity_m_s=np.ascontiguousarray(clear),
        boundary_return_time_s=np.ascontiguousarray(return_time),
        **{CONTRACT_FIELD: contract.as_dict(),
           SEMANTIC_FIELD: contract.semantic_id,
           FINGERPRINT_FIELD: contract.fingerprint},
    )
    return data


def contract_from_data(data: Mapping[str, object]):
    """Load the route-level M2-C instance from a selected local operator.

    The selected operator deliberately has local ``amr_enabled=0`` because it
    is advanced synchronously.  The nested M2-C contract describes the owning
    hierarchy and therefore must declare AMR enabled independently.
    """
    contract = load_suspended_amr_contract(data.get(CONTRACT_FIELD))
    _require(data.get(SEMANTIC_FIELD) == SUSPENDED_AMR_SEMANTIC_ID and
             data.get(FINGERPRINT_FIELD) == contract.fingerprint,
             "M2-C prepared semantic/fingerprint differs")
    _require(contract.suspension_profile_fingerprint ==
             data.get("suspension_profile_fingerprint"),
             "M2-C nested suspension-profile fingerprint differs")
    dry = float(data.get("dry", data.get("dry_tolerance", np.nan)))
    _require(int(data.get("bedlaw", -1)) == 11 and
             int(data.get("exchange_mode", -1)) == 4 and
             int(data.get("secondary_flow_model", -1)) in (0, 1) and
             int(data.get("nfrac", -1)) == contract.fraction_count == 1 and
             int(data.get("amr_enabled", -1)) == 0 and
             contract.amr_enabled and contract.amr_levels >= 2 and
             contract.amr_refinement_ratio == 2 and
             dry == contract.hydraulic_dry_tolerance_m,
             "M2-C implicit law-11 route/AMR numeric contract differs")
    profile = data.get("suspension_runtime_profile")
    _require(isinstance(profile, Mapping) and
             float(profile.get("diffusion_dry_threshold_m", np.nan)) ==
             contract.diffusion_dry_threshold_m,
             "M2-C diffusion DryFlc contract differs")
    return contract


def method_identity(data: Mapping[str, object]) -> dict:
    contract = contract_from_data(data)
    return dict(
        suspended_amr_contract_schema=contract.schema,
        suspended_amr_contract_method_id=SUSPENDED_AMR_METHOD_ID,
        suspended_amr_contract_semantic_id=contract.semantic_id,
        suspended_amr_contract_fingerprint=contract.fingerprint,
        suspended_amr_contract=contract.as_dict(),
        wet_dry_ownership=(
            "accepted_h_le_hydraulic_dry_tolerance_moves_each_suspended_"
            "class_to_same_class_active_bed_v1"),
        restriction_policy=(
            "area_weight_all_conserved_rows_then_wetdry_phase_owner_v1"),
        prolongation_policy=(
            "stage_fill_bounded_wet_child_concentration_exact_class_closure_v1"),
        amr_time_coupling="synchronous_selected_mesh_one_clock_v1",
        synchronous_shared_face_policy=(
            "one_accepted_integrated_advection_plus_horizontal_diffusion_v1"),
        recursive_subcycled_reflux_policy="not_qualified",
        open_boundary_memory_policy=(
            "stable_candidate_face_id_exact_one_to_one_no_average_v1"),
        regrid_transaction=(
            "state_topology_boundary_memory_ledgers_clocks_atomic_v1"),
        predictor_handoff="invalidate_topology_local_rate_guess_v1",
        automatic_CPU_fallback=False,
    )


class ResidentSuspendedWetDryOwnership:
    """Disjoint CUDA proposal for dry-cell suspended-to-active-bed transfer."""

    def __init__(self, area, fractions, dry_tolerance, *, source_path=None):
        import cupy as cp

        host = np.asarray(area, dtype=np.float64)
        _require(host.ndim == 1 and len(host) > 0 and
                 np.isfinite(host).all() and np.all(host > 0.0) and
                 isinstance(fractions, int) and fractions > 0 and
                 np.isfinite(dry_tolerance) and dry_tolerance > 0.0,
                 "Invalid M2-C wet/dry ownership geometry")
        self.cp, self.cells, self.fractions = cp, len(host), fractions
        self.area = cp.asarray(np.ascontiguousarray(host))
        self.dry = np.float64(dry_tolerance)
        self.next_suspended = cp.empty((fractions, self.cells), cp.float64)
        self.next_active = cp.empty_like(self.next_suspended)
        self.phase_change = cp.empty_like(self.next_suspended)
        self.status = cp.empty((fractions, self.cells), cp.int32)
        path = Path(source_path or HERE / "implicit_suspension_amr.cu")
        module = cp.RawModule(code=path.read_text(encoding="utf-8"),
            options=CUDA_OPTIONS,
            name_expressions=("m2c_wetdry_phase_owner",))
        self.kernel = module.get_function("m2c_wetdry_phase_owner")

    def propose(self, depth, suspended, active_bed):
        cp = self.cp
        _require(isinstance(depth, cp.ndarray) and
                 depth.shape == (self.cells,) and depth.dtype == cp.float64 and
                 depth.flags.c_contiguous and
                 isinstance(suspended, cp.ndarray) and
                 suspended.shape == (self.fractions, self.cells) and
                 suspended.dtype == cp.float64 and suspended.flags.c_contiguous and
                 isinstance(active_bed, cp.ndarray) and
                 active_bed.shape == suspended.shape and
                 active_bed.dtype == cp.float64 and active_bed.flags.c_contiguous,
                 "M2-C ownership requires contiguous resident FP64 state")
        count = self.cells*self.fractions
        self.kernel(((count+127)//128,), (128,), (depth, suspended, active_bed,
            self.area, np.int32(self.cells), np.int32(self.fractions), self.dry,
            self.next_suspended, self.next_active, self.phase_change,
            self.status))
        return dict(suspended=self.next_suspended, active_bed=self.next_active,
                    suspended_phase_change_kg=self.phase_change,
                    status=self.status)


__all__ = [
    "CONTRACT_FIELD", "SEMANTIC_FIELD", "FINGERPRINT_FIELD",
    "PREPARED_BINDING_FIELD", "apply_prepared_binding",
    "prepared_binding_from_case_metadata",
    "ResidentSuspendedWetDryOwnership", "contract_from_data",
    "method_identity",
]
