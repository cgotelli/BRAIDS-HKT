"""Canonical identity for the bounded v64 scalar-FLOW CUDA producers.

The document deliberately stops before live/public admission.  It binds the
resolved inputs needed by the v63 point laws to the selected
``implicit_fwave_midpoint`` GPU route, while recording the two source-order
questions that still block a Delft-equivalent H1--H4 promotion:

* positive scalar laws do not assign ``taurat`` before FLOW's ISlope=2 call;
* the BRAIDS face/boundary transaction has not yet consumed these producers.

Capacity hiding (TraFrm 14) and downstream Parker--Andrews slope hiding are
separate identities.  Likewise, the Wilcock-family ``RP_TAUB`` mobility input
is distinct from the total-current ``RP_USTAR**2`` consumed by ISlope=3/4.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Mapping, Sequence

from .scalar_flow_execution_v63 import CONTRACT_BY_SELECTOR


SCHEMA = 1
SCOPE_ID = "braids-q3-scalar-flow-gpu-producers-v64-bounded-v1"
INTEGRATOR = "implicit_fwave_midpoint"
BACKEND = "implicit_fwave_midpoint_gpu"
QUALIFICATION = "resident_cuda_producer_closure_not_live"
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_ROUGHNESS = frozenset((
    "roughness.chezy", "roughness.manning", "roughness.white_colebrook",
    "roughness.white_colebrook_waqua",
))
_SHEAR = frozenset((
    "shear.total_roughness", "shear.soulsby_clarke_skin",
))
_HIDING = frozenset((
    "hiding.none", "hiding.egiazaroff", "hiding.ashida_michiue",
    "hiding.parker_klingeman", "hiding.wu_wang_jia",
))

SOURCE_LOCKS = {
    "compdiam": {
        "path": ("resources/Delft3D-main/src/utils_gpl/morphology/packages/"
                 "morphology_kernel/src/compdiam.f90"),
        "sha256": "1beb16c5bcc10c24704dbe443e2f5ddd709106adf9561f3a89dd1813327f517a",
    },
    "compsandfrac": {
        "path": ("resources/Delft3D-main/src/utils_gpl/morphology/packages/"
                 "morphology_kernel/src/compsandfrac.f90"),
        "sha256": "42ec4e73de1191242a7dc43ce205c1c865008c6bde54799be2576ac5db33bfdd",
    },
    "dwnvel": {
        "path": ("resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/"
                 "flow2d3d_kernel/src/compute_sediment/dwnvel.f90"),
        "sha256": "cb26d6c037804ce2be3bcc975dab81cce21b78571f75e33fa302bcc14bf17ccd",
    },
    "erosed": {
        "path": ("resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/"
                 "flow2d3d_kernel/src/compute_sediment/erosed.f90"),
        "sha256": "4571fd7912233db03f597161af1a18702ee974483676004cda8186ccf24308c0",
    },
}


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def _fingerprint(name: str, value: object) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise ValueError(f"Scalar v64 {name} must be a lowercase SHA-256")
    return value


def scalar_flow_gpu_producer_document(*, reference_selector: int,
        scalar_execution_fingerprint: str, capacity_profile_fingerprint: str,
        forcing_fingerprint: str, distribution_fingerprint: str,
        fraction_ids: Sequence[str], roughness_method_id: str,
        sediment_shear_method_id: str,
        spatial_d50_mode: bool = False,
        maximum_waterdepth: bool = False,
        scour_enabled: bool = False,
        wave_enabled: bool = False,
        capacity_hiding_method_id: str = "hiding.none",
        slope_method_id: str = "bed_slope.off",
        slope_hiding_method_id: str = "hiding.none",
        secondary_method_id: str = "secondary.off") -> dict:
    """Build the exact non-live selected-GPU producer/route identity."""
    if isinstance(reference_selector, bool):
        raise ValueError("Scalar v64 reference selector must be an integer")
    try:
        law = CONTRACT_BY_SELECTOR[int(reference_selector)]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Scalar v64 selector is outside TraFrm 7/13/14/16/17/18") from error
    ids = tuple(fraction_ids)
    if not ids or any(not isinstance(value, str) or not value or
                      value != value.strip() for value in ids) or \
            len(set(ids)) != len(ids):
        raise ValueError("Scalar v64 fraction identities must be unique nonempty strings")
    if roughness_method_id not in _ROUGHNESS:
        raise ValueError("Scalar v64 effective velocity requires typed roughness")
    if law["reference_selector"] in (16, 17, 18):
        if sediment_shear_method_id not in _SHEAR:
            raise ValueError("Scalar v64 Wilcock-family capacity requires RP_TAUB identity")
    elif sediment_shear_method_id != "shear.not_consumed":
        raise ValueError("Selected scalar v64 capacity does not consume RP_TAUB")
    if law["reference_selector"] == 14:
        if capacity_hiding_method_id not in _HIDING:
            raise ValueError("TraFrm 14 requires an explicit capacity HIDEXP identity")
    elif capacity_hiding_method_id != "hiding.none":
        raise ValueError("Only TraFrm 14 consumes capacity HIDEXP")
    if not isinstance(slope_method_id, str) or not slope_method_id:
        raise ValueError("Scalar v64 slope identity must be nonempty")
    if slope_method_id == "bed_slope.delft_parker_andrews":
        if slope_hiding_method_id not in _HIDING - {"hiding.none"}:
            raise ValueError("Parker-Andrews requires a distinct downstream hiding identity")
    elif slope_hiding_method_id != "hiding.none":
        raise ValueError("Slope hiding is owned only by Parker-Andrews")
    if not isinstance(secondary_method_id, str) or not secondary_method_id:
        raise ValueError("Scalar v64 secondary identity must be nonempty")
    for name, value in (("spatial_d50_mode", spatial_d50_mode),
                        ("maximum_waterdepth", maximum_waterdepth),
                        ("scour_enabled", scour_enabled),
                        ("wave_enabled", wave_enabled)):
        if not isinstance(value, bool):
            raise ValueError(f"Scalar v64 {name} must be Boolean")
    if maximum_waterdepth:
        raise ValueError("Scalar v64 excludes FLOW MaximumWaterdepth observation depth")
    if scour_enabled:
        raise ValueError("Scalar v64 excludes scour/tauadd RP_USTAR modification")
    if wave_enabled:
        raise ValueError("Scalar v64 is current-only and excludes wave-enhanced forcing")

    dependencies = {
        "scalar_execution_fingerprint": _fingerprint(
            "scalar execution fingerprint", scalar_execution_fingerprint),
        "capacity_profile_fingerprint": _fingerprint(
            "capacity profile fingerprint", capacity_profile_fingerprint),
        "forcing_fingerprint": _fingerprint(
            "forcing fingerprint", forcing_fingerprint),
        "distribution_fingerprint": _fingerprint(
            "distribution fingerprint", distribution_fingerprint),
    }
    body = {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "qualification": QUALIFICATION,
        "backend": BACKEND,
        "integrator": INTEGRATOR,
        "automatic_cpu_fallback": False,
        "reference_selector": law["reference_selector"],
        "formula_code": law["formula_code"],
        "method_id": law["method_id"],
        "fraction_ids": list(ids),
        "fraction_basis": "active_surface_solid_volume",
        "active_fraction_normalization": (
            "resident_surface_mass_over_bulk_density_then_exact_cell_sum"),
        "malformed_prefraction_normalization": "rejected_not_renormalized",
        "depleted_active_surface_policy": (
            "explicit_normalized_retained_identity_zero_mass_and_zero_"
            "availability_no_synthetic_fraction_provisional_before_q5"),
        "spatial_d50_mode": spatial_d50_mode,
        "spatial_d50_contract": (
            "negative_single_class_sedd50_sentinel_plus_resident_positive_field"
            if spatial_d50_mode else
            "classic_erosed_seddm_equals_sedd50_broad_or_dirac_cdf"),
        "roughness_method_id": roughness_method_id,
        "sediment_shear_method_id": sediment_shear_method_id,
        "capacity_hiding_method_id": capacity_hiding_method_id,
        "slope_method_id": slope_method_id,
        "slope_hiding_method_id": slope_hiding_method_id,
        "secondary_method_id": secondary_method_id,
        "producer_order": [
            "active_surface_volume_fractions",
            "flow_compdiam_and_compsandfrac",
            "typed_hydraulic_roughness",
            "total_current_rp_ustar_and_effective_velocity",
            "selected_capacity_rp_taub",
            "v63_scalar_point_capacity",
            "active_fraction_exactly_once_pending_live_face_owner",
        ],
        "effective_velocity_source": (
            "flow2dh_h_over_e_observation_rp_ustar_chezy_v1"),
        "hydraulic_roughness_role": "accepted_typed_chezy_z0_drag",
        "sediment_chezy_role": (
            "recomputed_from_accepted_z0_and_max_0.01m_working_depth"),
        "observation_depth_source": (
            "accepted_midpoint_raw_depth_times_promoted_default_real_0.368_"
            "maximumwaterdepth_off"),
        "working_depth_source": "max_0.01m_accepted_midpoint_raw_depth",
        "working_depth_floor_m": 0.01,
        "direction_epsilon": "finite_positive_runtime_bound",
        "maximum_waterdepth": maximum_waterdepth,
        "scour_enabled": scour_enabled,
        "wave_enabled": wave_enabled,
        "forcing_scope": "current_only_2dh_no_orbital_forcing",
        "roughness_restart_binding": dependencies["forcing_fingerprint"],
        "mixture_source": (
            "flow_compdiam_compsandfrac_broad_distribution_v1"),
        "sbc_total": True,
        "external_secondary_bed_rotation": True,
        "suspended_secondary_rotation": False,
        "capacity_rp_taub_source": (
            sediment_shear_method_id if law["reference_selector"] in (16, 17, 18)
            else "not_consumed"),
        "slope_ust2_source": (
            "total_current_rp_ustar_squared_after_secondary_rotation"),
        "slope_taurat_source": (
            "positive_scalar_eqtran_unassigned_erosed_zero_pending_locked_probe"),
        "v63_diagnostic_rows_are_not_slope_forcing": True,
        "live_face_boundary_exner_connected": False,
        "checkpoint_admission": "identity_only_no_live_state",
        "source_locks": {name: dict(value)
                         for name, value in sorted(SOURCE_LOCKS.items())},
        "dependencies": dependencies,
    }
    return {**body, "fingerprint": hashlib.sha256(_canonical(body)).hexdigest()}


def load_scalar_flow_gpu_producer_document(value: Mapping[str, object]) -> dict:
    """Rebuild and byte-compare an untrusted v64 producer identity."""
    if not isinstance(value, Mapping):
        raise ValueError("Scalar v64 producer identity must be an object")
    expected_keys = set(scalar_flow_gpu_producer_document(
        reference_selector=7,
        scalar_execution_fingerprint="0"*64,
        capacity_profile_fingerprint="0"*64,
        forcing_fingerprint="0"*64,
        distribution_fingerprint="0"*64,
        fraction_ids=("probe",),
        roughness_method_id="roughness.manning",
        sediment_shear_method_id="shear.not_consumed",
    ))
    if set(value) != expected_keys:
        raise ValueError("Scalar v64 producer identity inventory differs")
    dependencies = value.get("dependencies")
    if not isinstance(dependencies, Mapping):
        raise ValueError("Scalar v64 producer dependencies are malformed")
    rebuilt = scalar_flow_gpu_producer_document(
        reference_selector=value.get("reference_selector"),
        scalar_execution_fingerprint=dependencies.get(
            "scalar_execution_fingerprint"),
        capacity_profile_fingerprint=dependencies.get(
            "capacity_profile_fingerprint"),
        forcing_fingerprint=dependencies.get("forcing_fingerprint"),
        distribution_fingerprint=dependencies.get("distribution_fingerprint"),
        fraction_ids=value.get("fraction_ids", ()),
        roughness_method_id=value.get("roughness_method_id"),
        sediment_shear_method_id=value.get("sediment_shear_method_id"),
        spatial_d50_mode=value.get("spatial_d50_mode"),
        maximum_waterdepth=value.get("maximum_waterdepth"),
        scour_enabled=value.get("scour_enabled"),
        wave_enabled=value.get("wave_enabled"),
        capacity_hiding_method_id=value.get("capacity_hiding_method_id"),
        slope_method_id=value.get("slope_method_id"),
        slope_hiding_method_id=value.get("slope_hiding_method_id"),
        secondary_method_id=value.get("secondary_method_id"),
    )
    if dict(value) != rebuilt:
        raise ValueError("Scalar v64 producer identity is stale or forged")
    return rebuilt


__all__ = [
    "SCHEMA", "SCOPE_ID", "INTEGRATOR", "BACKEND", "QUALIFICATION",
    "SOURCE_LOCKS", "scalar_flow_gpu_producer_document",
    "load_scalar_flow_gpu_producer_document",
]
