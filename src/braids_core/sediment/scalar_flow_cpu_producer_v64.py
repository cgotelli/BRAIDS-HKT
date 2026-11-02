"""Authenticated CPU producer foundation for the six v63 scalar FLOW laws.

Version 64 closes two point producers that the v63 capacity kernels deliberately
left upstream: FLOW's 2DH effective-velocity/RP_USTAR chain and its full
``compdiam``/``compsandfrac`` surface-mixture statistics.  It also fingerprints
the already accepted face-role, availability and Exner consumers without
claiming that a live solver calls them.

The product gate in this module is intentionally closed.  In particular, the
current project schema cannot serialize full within-class percentile tables,
and locked FLOW leaves ``taurat`` unassigned for these positive TraFrm values.
Neither a v63 diagnostic ratio nor RP_TAUB may be relabelled as that H4 input.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from types import MappingProxyType

from .scalar_flow_execution_v63 import (
    CONTRACT_BY_SELECTOR,
    load_scalar_flow_execution,
)


SCHEMA = 1
SCOPE_ID = "braids-q3-scalar-flow-cpu-producer-v64-component-v1"
PROFILE_SCOPE_ID = "braids-q3-scalar-flow-mixture-profile-v64-v1"
# iniphy assigns the unsuffixed default-real literal 0.41 to fp.  This is the
# exact binary32 value promoted to Python binary64, not the separate 0.4
# coefficient used by the classic secondary-flow relation.
FLOW_INIPHY_VON_KARMAN = 0.4099999964237213
_SHA = re.compile(r"[0-9a-f]{64}\Z")

LOCKED_SOURCES = MappingProxyType({
    "compdiam": (
        "resources/Delft3D-main/src/utils_gpl/morphology/packages/"
        "morphology_kernel/src/compdiam.f90",
        "1beb16c5bcc10c24704dbe443e2f5ddd709106adf9561f3a89dd1813327f517a",
    ),
    "compsandfrac": (
        "resources/Delft3D-main/src/utils_gpl/morphology/packages/"
        "morphology_kernel/src/compsandfrac.f90",
        "42ec4e73de1191242a7dc43ce205c1c865008c6bde54799be2576ac5db33bfdd",
    ),
    "dwnvel": (
        "resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/"
        "flow2d3d_kernel/src/compute_sediment/dwnvel.f90",
        "cb26d6c037804ce2be3bcc975dab81cce21b78571f75e33fa302bcc14bf17ccd",
    ),
    "erosed": (
        "resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/"
        "flow2d3d_kernel/src/compute_sediment/erosed.f90",
        "4571fd7912233db03f597161af1a18702ee974483676004cda8186ccf24308c0",
    ),
    "iniphy": (
        "resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/"
        "flow2d3d_kernel/src/inichk/iniphy.f90",
        "94dd55f636bb49e3c0a7d23deee06766dc15aeab4dd1152f60b09b98723f27c0",
    ),
    "eqtran": (
        "resources/Delft3D-main/src/utils_gpl/morphology/packages/"
        "morphology_kernel/src/eqtran.f90",
        "d2e8601cff68884f51ee3dcb0cd2fb52d9f0d676b1baaa3283c68913d4778388",
    ),
    "adjust_bedload": (
        "resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/"
        "flow2d3d_kernel/src/compute_sediment/adjust_bedload.f90",
        "a3dae8226082977ec1ee78748e0d439f0f7356d267a9f33cba58636a61afdcba",
    ),
})

IMPLEMENTATION_SOURCES = (
    "solver/wave/braids_delft_scalar_producer_v64_module.f90",
    "solver/wave/braids_delft_scalar_capacity_v63_module.f90",
    "solver/wave/braids_scalar_capacity_runtime_module.f90",
)

ROUGHNESS_METHODS = frozenset((
    "roughness.chezy", "roughness.manning", "roughness.white_colebrook",
    "roughness.white_colebrook_waqua",
))
SHEAR_METHODS = frozenset((
    "shear.total_roughness", "shear.soulsby_clarke_skin",
))
HIDING_METHODS = frozenset((
    "hiding.none", "hiding.egiazaroff", "hiding.ashida_michiue",
    "hiding.parker_klingeman", "hiding.wu_wang_jia",
))
SLOPE_METHODS = frozenset((
    "bed_slope.off", "bed_slope.delft_bagnold_ikeda",
    "bed_slope.delft_koch_flokstra", "bed_slope.delft_parker_andrews",
))

DEPENDENCY_STAGES = (
    "typed_forcing", "surface_composition_state", "settling_velocity",
    "capacity_hiding", "slope_hiding", "sediment_shear",
    "secondary_steering", "channel_composition", "face_reconstruction",
    "boundary_roles", "availability", "exner",
)

FACE_ROLE_POLICY = MappingProxyType({
    "internal": "shared_flux_final_donor_after_direction_correction",
    "closed_boundary": "exact_zero",
    "trace_inlet": "negative_trace_capacity_owner_inventory",
    "trace_outlet": "positive_trace_capacity_owner_inventory",
    "owner_outlet": "positive_owner_capacity_owner_inventory",
})

PRODUCT_BLOCKERS = (
    "project_schema_has_no_full_within_class_percentile_distribution",
    "live_deck_has_no_authenticated_two_depth_characteristic_velocity_z0rou_producer",
    "live_boundary_trace_producers_not_connected_to_v64",
    "spatial_D50_field_and_negative_sedd50_sentinel_not_in_product_identity",
    "positive_trafrm_taurat_is_unassigned_in_locked_eqtran",
    "v64_not_called_by_native_solver_or_normal_product_dispatch",
    "no_live_result_or_checkpoint_carries_the_v64_execution_fingerprint",
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _fingerprint(value: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise ValueError(f"Q3 scalar v64 {name} must be a lowercase SHA-256")
    return value


def _finite(value: object, name: str, *, positive: bool = False,
            nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Q3 scalar v64 {name} must be numeric")
    result = float(value)
    if (not math.isfinite(result) or (positive and result <= 0.0) or
            (nonnegative and result < 0.0)):
        raise ValueError(f"Q3 scalar v64 {name} is outside its finite domain")
    return 0.0 if result == 0.0 else result


def scalar_cpu_producer_semantic_document() -> dict:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "qualification": "cpu_producer_component_locked_source_parity_not_live",
        "selectors": [7, 13, 14, 16, 17, 18],
        "locked_sources": {
            name: {"path": path, "sha256": sha256}
            for name, (path, sha256) in sorted(LOCKED_SOURCES.items())
        },
        "effective_velocity": {
            "input": [
                "dwnvel_observation_depth", "erosed_physical_depth",
                "characteristic_u", "characteristic_v",
                "total_current_z0rou", "gravity", "von_karman",
                "direction_epsilon", "wave_false", "scour_false",
            ],
            "output": [
                "resolved_chezy", "RP_USTAR", "RP_USTAR_squared",
                "RP_EFUMN", "RP_EFVMN",
            ],
            "roughness_scope": (
                "required_for_every_live_chain_including_TraFrm_7_even_when_"
                "the_point_formula_does_not_receive_RP_CHEZY"
            ),
            "depth_roles": {
                "dwnvel_observation_depth": (
                    "owns_characteristic_velocity_normalization_and_0p368_"
                    "observation_height_may_include_MaximumWaterdepth_faces"
                ),
                "erosed_work_depth": (
                    "max_0p01_m_of_physical_cell_depth_for_Chezy_and_RP_DEPTH"
                ),
            },
            "admitted_regime": "current_only_wave_false_scour_false",
            "von_karman": {
                "value": FLOW_INIPHY_VON_KARMAN,
                "source": "iniphy_default_real_0p41_promoted_to_fp",
                "not_secondary_alpha": (
                    "classic_secondary_flow_uses_a_separate_0p4_literal"
                ),
            },
        },
        "mixture": {
            "input": (
                "surface_fractions_plus_full_within_class_percentile_tables_"
                "mean_diameter_median_diameter_and_sediment_types"
            ),
            "output": ["Dm", "RP_DG", "RP_D90MX", "RP_DGSD", "RP_SNDFR",
                       "RP_MUDFR"],
            "empty_or_mud_only": (
                "getfrac_composition_remains_normalized_mark_no_exposed_nonmud_"
                "and_do_not_call_noncohesive_capacity"
            ),
            "spatial_d50_dgsd": (
                "component_operator_accepts_explicit_field_and_returns_zero_"
                "but_product_identity_excludes_missing_field_fingerprint"
            ),
            "FLOW4_call_site_mean_diameter": (
                "erosed_and_initsedtra_pass_sedd50_as_both_seddm_and_sedd50"
            ),
        },
        "separate_stress_roles": {
            "capacity_RP_TAUB": (
                "typed_total_or_skin_shear_only_for_TraFrm_16_17_18"
            ),
            "slope_ust2": (
                "total_current_RP_USTAR_squared_then_secondary_vector_magnitude_"
                "for_ISlope_3_4"
            ),
            "capacity_diagnostic_rows": "never_slope_forcing",
            "ISlope_2_taurat": (
                "locked_positive_TraFrm_eqtran_INTENT_OUT_unassigned_while_"
                "caller_storage_is_initialized_zero_pending_direct_runtime_probe"
            ),
        },
        "separate_hiding_roles": {
            "capacity": "generic_HIDEXP_only_for_TraFrm_14",
            "slope": "ISlope_4_HIDEXP_for_every_selector",
        },
        "downstream_order": [
            "v63_scalar_capacity_and_density_conversion",
            "external_secondary_rotation_of_bed_only",
            "channel_composition_once",
            "unchanged_face_reconstruction",
            "raw_face_candidate_before_inventory",
            "classic_flow_slope_direction_when_qualified",
            "final_direction_donor_selection",
            "active_fraction_and_finite_inventory_availability_once",
            "role_signed_shared_face_flux",
            "conservative_exner_candidate",
        ],
        "face_role_policy": dict(FACE_ROLE_POLICY),
        "product_admitted": False,
        "product_blockers": list(PRODUCT_BLOCKERS),
    }


SEMANTIC_ID = hashlib.sha256(
    _canonical(scalar_cpu_producer_semantic_document())
).hexdigest()


def mixture_profile_document(*, fractions: Sequence[Mapping[str, object]],
        max_mud_sediment_type: int = 2, min_dxx_sediment_type: int = 3,
        spatial_d50: bool = False, spatial_log_sigma: float = 0.0) -> dict:
    """Bind the distribution inputs that the existing product model lacks."""
    if (not isinstance(fractions, Sequence) or isinstance(fractions, (str, bytes))
            or not fractions):
        raise ValueError("Q3 scalar v64 requires ordered sediment fractions")
    if (isinstance(max_mud_sediment_type, bool) or
            not isinstance(max_mud_sediment_type, int) or
            isinstance(min_dxx_sediment_type, bool) or
            not isinstance(min_dxx_sediment_type, int) or
            max_mud_sediment_type >= min_dxx_sediment_type):
        raise ValueError("Q3 scalar v64 sediment-type thresholds are invalid")
    if not isinstance(spatial_d50, bool):
        raise ValueError("Q3 scalar v64 spatial_d50 flag must be boolean")
    sigma = _finite(
        spatial_log_sigma, "spatial log sigma", nonnegative=True)
    if spatial_d50:
        raise ValueError(
            "Q3 scalar v64 product identity excludes spatial D50 until the "
            "field fingerprint and negative class sentinel are serialized")
    if sigma != 0.0:
        raise ValueError("Non-spatial Q3 scalar v64 profile does not consume sigma")

    rows = []
    ids: set[str] = set()
    for raw in fractions:
        if not isinstance(raw, Mapping):
            raise ValueError("Q3 scalar v64 fraction must be an object")
        expected = {
            "fraction_id", "sediment_type", "mean_diameter_m",
            "median_diameter_m", "percentile_table",
        }
        if set(raw) != expected:
            raise ValueError("Q3 scalar v64 fraction inventory differs")
        fraction_id = raw["fraction_id"]
        sediment_type = raw["sediment_type"]
        if (not isinstance(fraction_id, str) or not fraction_id or
                fraction_id in ids):
            raise ValueError("Q3 scalar v64 fraction IDs must be unique")
        ids.add(fraction_id)
        if isinstance(sediment_type, bool) or not isinstance(sediment_type, int):
            raise ValueError("Q3 scalar v64 sediment type must be an integer")
        mean = _finite(raw["mean_diameter_m"], "mean diameter",
                       positive=sediment_type > max_mud_sediment_type)
        median = _finite(raw["median_diameter_m"], "median diameter",
                         positive=sediment_type >= min_dxx_sediment_type)
        if mean != median:
            raise ValueError(
                "FLOW4 live identity requires mean diameter equal to D50")
        table = raw["percentile_table"]
        if (not isinstance(table, Sequence) or isinstance(table, (str, bytes))):
            raise ValueError("Q3 scalar v64 percentile table must be a list")
        points = []
        previous_probability = -1.0
        previous_diameter = -1.0
        for point in table:
            if not isinstance(point, Mapping) or set(point) != {
                    "probability", "diameter_m"}:
                raise ValueError("Q3 scalar v64 percentile point is malformed")
            probability = _finite(
                point["probability"], "percentile probability",
                nonnegative=True)
            diameter = _finite(
                point["diameter_m"], "percentile diameter", positive=True)
            if probability > 1.0 or probability < previous_probability:
                raise ValueError("Q3 scalar v64 probabilities are not ordered")
            if diameter <= previous_diameter:
                raise ValueError("Q3 scalar v64 percentile diameters are not ordered")
            previous_probability = probability
            previous_diameter = diameter
            points.append({"probability": probability, "diameter_m": diameter})
        if (sediment_type >= min_dxx_sediment_type and not spatial_d50 and
                not points):
            raise ValueError("Non-mud fraction requires its FLOW percentile table")
        rows.append({
            "fraction_id": fraction_id,
            "sediment_type": sediment_type,
            "mean_diameter_m": mean,
            "median_diameter_m": median,
            "percentile_table": points,
        })
    body = {
        "schema": SCHEMA,
        "scope_id": PROFILE_SCOPE_ID,
        "ordered_fractions": rows,
        "max_mud_sediment_type": max_mud_sediment_type,
        "min_dxx_sediment_type": min_dxx_sediment_type,
        "spatial_d50": spatial_d50,
        "spatial_log_sigma": sigma,
        "composition_basis": "FLOW_getfrac_surface_fraction",
        "broad_distribution_policy": "full_percentile_table_no_D50_substitution",
        "surface_fraction_sum_policy": "require_getfrac_normalized_sum_one",
        "depleted_inventory_policy": (
            "retain_normalized_surface_identity_and_apply_zero_availability_"
            "downstream"
        ),
    }
    return {**body, "fingerprint": _fingerprint(body)}


def load_mixture_profile(value: Mapping[str, object]) -> dict:
    if not isinstance(value, Mapping):
        raise ValueError("Q3 scalar v64 mixture profile must be an object")
    expected = {
        "schema", "scope_id", "ordered_fractions",
        "max_mud_sediment_type", "min_dxx_sediment_type", "spatial_d50",
        "spatial_log_sigma", "composition_basis", "broad_distribution_policy",
        "surface_fraction_sum_policy", "depleted_inventory_policy",
        "fingerprint",
    }
    if set(value) != expected:
        raise ValueError("Q3 scalar v64 mixture profile inventory differs")
    rebuilt = mixture_profile_document(
        fractions=value["ordered_fractions"],
        max_mud_sediment_type=value["max_mud_sediment_type"],
        min_dxx_sediment_type=value["min_dxx_sediment_type"],
        spatial_d50=value["spatial_d50"],
        spatial_log_sigma=value["spatial_log_sigma"],
    )
    if dict(value) != rebuilt or _canonical(value) != _canonical(rebuilt):
        raise ValueError("Q3 scalar v64 mixture profile is stale or forged")
    return rebuilt


def _dependency_rows(selector: int, bed_slope_method_id: str,
                     dependencies: Mapping[str, object]) -> list[dict]:
    if not isinstance(dependencies, Mapping) or set(dependencies) != set(
            DEPENDENCY_STAGES):
        raise ValueError("Q3 scalar v64 dependency inventory differs")
    consumed = {
        "typed_forcing", "surface_composition_state", "secondary_steering",
        "channel_composition", "face_reconstruction", "boundary_roles",
        "availability", "exner",
    }
    if selector == 7:
        consumed.add("settling_velocity")
    if selector == 14:
        consumed.add("capacity_hiding")
    if selector in (16, 17, 18):
        consumed.add("sediment_shear")
    if bed_slope_method_id == "bed_slope.delft_parker_andrews":
        consumed.add("slope_hiding")
    rows = []
    for stage in DEPENDENCY_STAGES:
        value = dependencies[stage]
        if stage in consumed:
            rows.append({"stage": stage, "status": "consumed",
                         "fingerprint": _sha(value, f"{stage} fingerprint")})
        else:
            if value is not None:
                raise ValueError(
                    f"Q3 scalar v64 {stage} is not consumed by this selection")
            rows.append({"stage": stage, "status": "not_consumed",
                         "fingerprint": None})
    return rows


def cpu_producer_execution_document(*, scalar_execution: Mapping[str, object],
        mixture_profile: Mapping[str, object],
        effective_velocity_roughness_method_id: str,
        sediment_shear_method_id: str, slope_hiding_method_id: str,
        bed_slope_method_id: str, secondary_method_id: str,
        dependencies: Mapping[str, object],
        implementation_source_hashes: Mapping[str, object]) -> dict:
    scalar = load_scalar_flow_execution(scalar_execution)
    mixture = load_mixture_profile(mixture_profile)
    selector = int(scalar["reference_selector"])
    if effective_velocity_roughness_method_id not in ROUGHNESS_METHODS:
        raise ValueError("Q3 scalar v64 requires typed total-current roughness")
    if selector != 7 and scalar["roughness_method_id"] != \
            effective_velocity_roughness_method_id:
        raise ValueError("Q3 scalar v64 point and velocity roughness differ")
    if selector in (16, 17, 18):
        if (sediment_shear_method_id not in SHEAR_METHODS or
                scalar["sediment_shear_method_id"] != sediment_shear_method_id):
            raise ValueError("Q3 scalar v64 RP_TAUB shear identity differs")
    elif sediment_shear_method_id != "shear.not_consumed":
        raise ValueError("Q3 scalar v64 selected formula does not consume RP_TAUB")
    if bed_slope_method_id not in SLOPE_METHODS:
        raise ValueError("Q3 scalar v64 bed-slope method is invalid")
    if bed_slope_method_id == "bed_slope.delft_parker_andrews":
        if slope_hiding_method_id not in HIDING_METHODS:
            raise ValueError("Q3 scalar v64 Parker slope requires slope HIDEXP")
    elif slope_hiding_method_id != "hiding.not_consumed_by_slope":
        raise ValueError("Q3 scalar v64 selected slope does not consume HIDEXP")
    if not isinstance(secondary_method_id, str) or not secondary_method_id:
        raise ValueError("Q3 scalar v64 secondary method identity is invalid")

    dependency_rows = _dependency_rows(
        selector, bed_slope_method_id, dependencies)
    by_stage = {row["stage"]: row for row in dependency_rows}
    if by_stage["typed_forcing"]["fingerprint"] != scalar["forcing_fingerprint"]:
        raise ValueError("Q3 scalar v64 forcing dependency differs from v63")
    required_sources = set(IMPLEMENTATION_SOURCES)
    if (not isinstance(implementation_source_hashes, Mapping) or
            set(implementation_source_hashes) != required_sources):
        raise ValueError("Q3 scalar v64 implementation source inventory differs")
    sources = {
        path: _sha(implementation_source_hashes[path], f"{path} source")
        for path in sorted(required_sources)
    }
    taurat = (
        "locked_positive_TraFrm_INTENT_OUT_unassigned_caller_initialized_zero_"
        "not_admitted"
        if bed_slope_method_id == "bed_slope.delft_bagnold_ikeda" else
        "not_consumed_by_selected_slope"
    )
    body = {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "qualification": "exact_cpu_producer_foundation_not_live",
        "backend": "cpu",
        "automatic_backend_fallback": False,
        "method_id": scalar["method_id"],
        "formula_code": scalar["formula_code"],
        "reference_selector": selector,
        "scalar_execution_fingerprint": scalar["fingerprint"],
        "mixture_profile_fingerprint": mixture["fingerprint"],
        "effective_velocity_roughness_method_id": (
            effective_velocity_roughness_method_id),
        "point_kernel_roughness_method_id": scalar["roughness_method_id"],
        "sediment_shear_method_id": sediment_shear_method_id,
        "capacity_hiding_method_id": scalar["hiding_method_id"],
        "slope_hiding_method_id": slope_hiding_method_id,
        "bed_slope_method_id": bed_slope_method_id,
        "secondary_method_id": secondary_method_id,
        "effective_velocity_regime": {
            "wave": False,
            "scour": False,
            "von_karman": FLOW_INIPHY_VON_KARMAN,
            "depth_contract": (
                "separate_dwnvel_observation_depth_and_erosed_physical_depth_"
                "with_internal_0p01_m_work_floor"
            ),
        },
        "dependencies": dependency_rows,
        "stress_ownership": {
            "capacity_RP_TAUB": (
                "typed_sediment_shear_only_never_hydraulic_ust2"),
            "slope_ust2": (
                "total_current_RP_USTAR_squared_after_secondary_rotation"),
            "ISlope_2_taurat": taurat,
            "v63_diagnostics_are_slope_forcing": False,
        },
        "hiding_ownership": {
            "capacity": "generic_only_for_TraFrm_14",
            "slope": "separate_ISlope_4_role_for_all_six",
        },
        "face_role_policy": dict(FACE_ROLE_POLICY),
        "active_fraction_application_count": 1,
        "availability_application_count": 1,
        "exner_policy": "candidate_only_conservative_commit_after_validation",
        "implementation_source_hashes": sources,
        "product_admitted": False,
        "live_dispatch_qualified": False,
        "restart_qualified": False,
        "product_blockers": list(PRODUCT_BLOCKERS),
    }
    return {**body, "fingerprint": _fingerprint(body)}


def load_cpu_producer_execution(value: Mapping[str, object], *,
        scalar_execution: Mapping[str, object],
        mixture_profile: Mapping[str, object]) -> dict:
    if not isinstance(value, Mapping):
        raise ValueError("Q3 scalar v64 execution must be an object")
    try:
        dependencies = {
            row["stage"]: row["fingerprint"]
            for row in value["dependencies"]
        }
        rebuilt = cpu_producer_execution_document(
            scalar_execution=scalar_execution,
            mixture_profile=mixture_profile,
            effective_velocity_roughness_method_id=value[
                "effective_velocity_roughness_method_id"],
            sediment_shear_method_id=value["sediment_shear_method_id"],
            slope_hiding_method_id=value["slope_hiding_method_id"],
            bed_slope_method_id=value["bed_slope_method_id"],
            secondary_method_id=value["secondary_method_id"],
            dependencies=dependencies,
            implementation_source_hashes=value["implementation_source_hashes"],
        )
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError("Invalid Q3 scalar v64 execution identity") from error
    if dict(value) != rebuilt or _canonical(value) != _canonical(rebuilt):
        raise ValueError("Q3 scalar v64 execution identity is stale or forged")
    return rebuilt


def lifecycle_identity_document(*, stage: str,
        execution_fingerprint: str, accepted_step: int, time_s: float,
        state_sha256: str, parent_result_fingerprint: str | None = None) -> dict:
    """Bind component evidence/replay state without claiming live restart."""
    if stage not in ("component_result", "component_restart"):
        raise ValueError("Q3 scalar v64 lifecycle stage is invalid")
    if (isinstance(accepted_step, bool) or not isinstance(accepted_step, int) or
            accepted_step < 0):
        raise ValueError("Q3 scalar v64 accepted step is invalid")
    parent = None
    if stage == "component_restart":
        parent = _sha(parent_result_fingerprint, "parent result fingerprint")
    elif parent_result_fingerprint is not None:
        raise ValueError("Q3 scalar v64 component result cannot name a parent")
    body = {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "stage": stage,
        "execution_fingerprint": _sha(
            execution_fingerprint, "execution fingerprint"),
        "accepted_step": accepted_step,
        "time_s": _finite(time_s, "accepted time", nonnegative=True),
        "state_sha256": _sha(state_sha256, "state fingerprint"),
        "parent_result_fingerprint": parent,
        "component_evidence_only": True,
        "live_result_admitted": False,
        "live_restart_admitted": False,
        "physical_state_extension": False,
    }
    return {**body, "fingerprint": _fingerprint(body)}


def load_lifecycle_identity(value: Mapping[str, object]) -> dict:
    if not isinstance(value, Mapping):
        raise ValueError("Q3 scalar v64 lifecycle identity must be an object")
    try:
        rebuilt = lifecycle_identity_document(
            stage=value["stage"],
            execution_fingerprint=value["execution_fingerprint"],
            accepted_step=value["accepted_step"], time_s=value["time_s"],
            state_sha256=value["state_sha256"],
            parent_result_fingerprint=value["parent_result_fingerprint"],
        )
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError("Invalid Q3 scalar v64 lifecycle identity") from error
    if dict(value) != rebuilt or _canonical(value) != _canonical(rebuilt):
        raise ValueError("Q3 scalar v64 lifecycle identity is stale or forged")
    return rebuilt


def require_product_admission(value: Mapping[str, object]) -> None:
    """Fail closed at the product seam until a real live v64 route exists."""
    if not isinstance(value, Mapping) or value.get("scope_id") != SCOPE_ID:
        raise ValueError("Q3 scalar v64 product gate requires its execution identity")
    raise ValueError(
        "Q3 scalar v64 is producer-qualified only; product admission is closed: "
        + ", ".join(PRODUCT_BLOCKERS)
    )


__all__ = [
    "DEPENDENCY_STAGES", "FACE_ROLE_POLICY", "IMPLEMENTATION_SOURCES",
    "FLOW_INIPHY_VON_KARMAN", "LOCKED_SOURCES", "PRODUCT_BLOCKERS",
    "PROFILE_SCOPE_ID", "SCHEMA",
    "SCOPE_ID", "SEMANTIC_ID", "cpu_producer_execution_document",
    "lifecycle_identity_document", "load_cpu_producer_execution",
    "load_lifecycle_identity", "load_mixture_profile",
    "mixture_profile_document", "require_product_admission",
    "scalar_cpu_producer_semantic_document",
]
