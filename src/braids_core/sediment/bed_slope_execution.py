"""Authenticated additive Q3-H4 FLOW bed-slope composition identity.

The locked Delft3D4 operator changes a bed-load vector *after* the allowed
secondary-flow/channel composition and *before* the final inventory donor is
chosen.  Older BRAIDS identities either described slope-off paths or historical
specialisations.  This module gives the complete ISlope=2/3/4 face operator an
additive identity without changing those frozen meanings.

This is a component identity, not public-route admission.  A live controller
must still authenticate the upstream identities named here and explicitly wire
the returned face flux into its accepted-state Exner transaction.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType


SCHEMA = 1
ENVELOPE_VERSION = 61
SCOPE_ID = "braids-q3h4-flow2dh-face-bed-slope-composition-v1"
SEMANTIC_ID = hashlib.sha256(SCOPE_ID.encode("ascii")).hexdigest()
EXECUTION_KEY = "bed_slope_execution_identity"
ORDERED_ROUTE_KEY = "bed_slope_ordered_route_identity"

METHODS = MappingProxyType({
    "bed_slope.delft_bagnold_ikeda": {
        "runtime_mode": 2,
        "delft_islope": 2,
        "persistence_code": 5,
        "stress_semantics": "flow_taurat_stress_ratio",
    },
    "bed_slope.delft_koch_flokstra": {
        "runtime_mode": 3,
        "delft_islope": 3,
        "persistence_code": 3,
        "stress_semantics": "absolute_shields",
    },
    "bed_slope.delft_parker_andrews": {
        "runtime_mode": 4,
        "delft_islope": 4,
        "persistence_code": 6,
        "stress_semantics": "absolute_shields",
    },
})

UPSTREAM_STAGES = (
    "capacity_execution",
    "secondary_steering_execution",
    "channel_composition_execution",
    "face_reconstruction_execution",
    "hiding_execution",
)

COMPOSITION_ORDER = (
    "raw_scalar_bed_capacity",
    "allowed_secondary_bed_steering_once",
    "channel_composition_once",
    "unchanged_face_reconstruction",
    "raw_face_candidate_selection_without_inventory_limiting",
    "classic_flow_bed_slope_direction_once",
    "corrected_direction_donor_selection",
    "finite_inventory_availability_once",
    "role_signed_shared_face_flux",
    "conservative_exner_candidate",
)

FACE_ROLE_POLICY = MappingProxyType({
    "internal": "shared_corrected_flux_owner_or_neighbour_final_donor",
    "closed_boundary": "exact_zero",
    "trace_inlet": "trace_capacity_negative_part_owner_inventory",
    "trace_outlet": "trace_capacity_positive_part_owner_inventory",
    "owner_outlet": "owner_capacity_positive_part_owner_inventory",
})

FIELD_OWNERSHIP = MappingProxyType({
    "transport": "post_secondary_post_channel_reconstructed_face_candidate",
    "downhill": "shared_face_gradient_positive_downhill",
    "stress": "authenticated_face_value_with_method_specific_semantics",
    "depth": "authenticated_face_hydraulic_depth",
    "d50": "ordered_fraction_constant_or_authenticated_material_field",
    "mean_diameter": "authenticated_face_surface_mixture_mean_diameter",
    "internal_hiding": "arithmetic_average_of_owner_and_neighbour_factors",
    "boundary_hiding": "owner_factor_replicated_on_both_sides",
})

LOCKED_FLOW_RELATIVE_PATH = (
    "resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/"
    "flow2d3d_kernel/src/compute_sediment/adjust_bedload.f90"
)
LOCKED_FLOW_SHA256 = (
    "a3dae8226082977ec1ee78748e0d439f0f7356d267a9f33cba58636a61afdcba"
)
IMPLEMENTATION_PATHS = (
    "braids_core/sediment/bed_slope_execution.py",
    "solver/wave/braids_flow_bed_slope_operator_module.f90",
    "solver/wave/braids_flow_bed_slope_face_composer_module.f90",
    "solver/wave/braids_scalar_capacity_runtime_module.f90",
    "braids_gpu/flow_bed_slope_runtime.py",
    "braids_gpu/flow_bed_slope_runtime.cu",
    "braids_gpu/scalar_capacity_runtime.py",
    "braids_gpu/scalar_capacity_runtime.cu",
    "braids_gpu/flow_bed_slope_composition.py",
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]*\Z")


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _fingerprint(value: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"Q3-H4 {name} must be a lowercase SHA-256 fingerprint")
    return value


def _finite(value: object, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Q3-H4 {name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        relation = "finite and nonnegative" if nonnegative else "finite"
        raise ValueError(f"Q3-H4 {name} must be {relation}")
    return 0.0 if result == 0.0 else result


def _fraction_ids(value: object) -> list[str]:
    if (not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or
            not value):
        raise ValueError("Q3-H4 ordered fraction identities must be a sequence")
    result = list(value)
    if (len(set(result)) != len(result) or
            any(not isinstance(item, str) or _IDENTIFIER.fullmatch(item) is None
                for item in result)):
        raise ValueError(
            "Q3-H4 fraction identities must be ordered unique identifiers"
        )
    return result


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    locked = root / LOCKED_FLOW_RELATIVE_PATH
    if not locked.is_file():
        raise ValueError("Q3-H4 locked Delft3D FLOW source is unavailable")
    locked_hash = hashlib.sha256(locked.read_bytes()).hexdigest()
    if locked_hash != LOCKED_FLOW_SHA256:
        raise ValueError("Q3-H4 locked Delft3D FLOW source identity changed")
    result = {LOCKED_FLOW_RELATIVE_PATH: locked_hash}
    for relative in IMPLEMENTATION_PATHS:
        path = root / relative
        if not path.is_file():
            raise ValueError(f"Q3-H4 implementation source is unavailable: {relative}")
        result[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def bed_slope_execution_document(
    *,
    method_id: str,
    fraction_ids: Sequence[str],
    upstream_fingerprints: Mapping[str, str],
    alfabs: float,
    alfabn: float,
    ashld: float,
    bshld: float,
    cshld: float,
    dshld: float,
    alfpa: float,
    thcrpa: float,
) -> dict[str, object]:
    """Build the canonical Q3-H4 face-slope component identity.

    Every FLOW coefficient is bound for every method, including coefficients
    not consumed by the selected branch.  This prevents a saved execution
    from being relabelled under a different parameter set later.
    """
    if not isinstance(method_id, str) or method_id not in METHODS:
        raise ValueError("Q3-H4 requires a classic FLOW bed-slope method")
    fractions = _fraction_ids(fraction_ids)
    if not isinstance(upstream_fingerprints, Mapping):
        raise ValueError("Q3-H4 upstream fingerprints must be an object")
    if set(upstream_fingerprints) != set(UPSTREAM_STAGES):
        raise ValueError(
            "Q3-H4 requires the complete ordered upstream identity set"
        )
    upstream = [
        {"stage": stage, "fingerprint": _sha(
            upstream_fingerprints[stage], f"{stage} fingerprint")}
        for stage in UPSTREAM_STAGES
    ]
    controls = {
        "alfabs": _finite(alfabs, "AlfaBs"),
        "alfabn": _finite(alfabn, "AlfaBn"),
        "ashld": _finite(ashld, "AShld"),
        "bshld": _finite(bshld, "BShld"),
        "cshld": _finite(cshld, "CShld"),
        "dshld": _finite(dshld, "DShld"),
        "alfpa": _finite(alfpa, "AlfPA"),
        "thcrpa": _finite(thcrpa, "ThCrPA", nonnegative=True),
    }
    method = dict(METHODS[method_id])
    body: dict[str, object] = {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "envelope_version": ENVELOPE_VERSION,
        "component_scope": "backend_neutral_face_and_open_boundary_operator",
        "method_id": method_id,
        **method,
        "ordered_fraction_ids": fractions,
        "parameters": controls,
        "upstream_identities": upstream,
        "composition_order": list(COMPOSITION_ORDER),
        "face_role_policy": dict(FACE_ROLE_POLICY),
        "field_ownership": dict(FIELD_OWNERSHIP),
        "transport_units": "solid_mass_per_time_per_width",
        "downhill_sign": "positive_downhill",
        "parker_hiding_average": "arithmetic",
        "invalid_policy": "reject_before_exner_no_partial_commit",
        "backend_selection": "explicit_user_choice",
        "automatic_cpu_fallback": False,
        "implementation_source_hashes": _source_hashes(),
        "runtime_admission_authority": "separate_exact_live_route_gate",
    }
    return {**body, "fingerprint": _fingerprint(body)}


def validate_bed_slope_execution_document(
    value: Mapping[str, object],
) -> dict[str, object]:
    """Rebuild and byte-compare an untrusted Q3-H4 identity."""
    if not isinstance(value, Mapping):
        raise ValueError("Q3-H4 bed-slope execution identity must be an object")
    try:
        supplied_upstream = value["upstream_identities"]
        if (not isinstance(supplied_upstream, list) or
                any(not isinstance(row, Mapping) for row in supplied_upstream)):
            raise ValueError("invalid upstream identity rows")
        upstream = {row["stage"]: row["fingerprint"] for row in supplied_upstream}
        controls = value["parameters"]
        rebuilt = bed_slope_execution_document(
            method_id=value["method_id"],
            fraction_ids=value["ordered_fraction_ids"],
            upstream_fingerprints=upstream,
            alfabs=controls["alfabs"],
            alfabn=controls["alfabn"],
            ashld=controls["ashld"],
            bshld=controls["bshld"],
            cshld=controls["cshld"],
            dshld=controls["dshld"],
            alfpa=controls["alfpa"],
            thcrpa=controls["thcrpa"],
        )
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError("Invalid Q3-H4 bed-slope execution identity") from error
    if dict(value) != rebuilt or _canonical(value) != _canonical(rebuilt):
        raise ValueError("Q3-H4 bed-slope execution identity is stale or forged")
    return rebuilt


__all__ = [
    "COMPOSITION_ORDER", "ENVELOPE_VERSION", "EXECUTION_KEY",
    "FACE_ROLE_POLICY", "FIELD_OWNERSHIP", "LOCKED_FLOW_SHA256",
    "METHODS", "ORDERED_ROUTE_KEY", "SCHEMA", "SCOPE_ID", "SEMANTIC_ID",
    "UPSTREAM_STAGES", "bed_slope_execution_document",
    "validate_bed_slope_execution_document",
]
