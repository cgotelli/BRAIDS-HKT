"""Additive Q3-H2 classic FLOW equilibrium-secondary identity.

The v56 envelope composes the classic Delft3D4 FLOW 2DH equilibrium
secondary operator around a frozen Q3-C/v54 capacity foundation and, when
present, a frozen Q3-H1/v55 hiding foundation.  A v55 document continues to
mean exactly what it says internally (including ``secondary.off``); this
module stores its fingerprint as an opaque foundation reference and describes
the new outer composition without rewriting either older identity.

This module authenticates semantics only.  It does not grant a CPU or GPU
route admission, and in particular does not turn a nonzero ``Betac`` identity
into evidence that the first live route supports nonzero ``Betac``.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from types import MappingProxyType


SCHEMA = 1
ENVELOPE_VERSION = 56
SCOPE_ID = "braids-q3h2-flow2dh-equilibrium-secondary-composition-v1"
SEMANTIC_ID = hashlib.sha256(SCOPE_ID.encode("ascii")).hexdigest()
EXECUTION_KEY = "secondary_foundation_execution_identity"
ORDERED_ROUTE_KEY = "secondary_foundation_ordered_route_identity"

METHOD_ID = "secondary.equilibrium"
MODEL = 1
FIXED_GRID_SCOPE = "fixed_grid"
BACKEND_SELECTION = "explicit_user_choice"
CAPACITY_METHOD_BY_LAW = MappingProxyType({
    2: "capacity.mpm",
    8: "capacity.engelund_hansen",
    11: "capacity.general",
})
ROUGHNESS_METHOD_IDS = frozenset((
    "roughness.chezy",
    "roughness.manning",
    "roughness.white_colebrook",
    "roughness.white_colebrook_waqua",
))
CORIOLIS_DISPOSITIONS = frozenset(("zero", "nonzero"))

CURVATURE_DISCRETIZATION = (
    "braids_fixed_grid_cell_lsq_streamline_curvature_v1"
)
CURVATURE_CAP_POLICY = "classic_flow2dh_rmincf_local_metric_cap_v1"
METRIC_POLICY = "fixed_cartesian_metric_sums_2dx_2dy_v1"
REFRESH_TIMING = "every_trial_stage_before_raw_bed_rotation"
BOUNDARY_POLICY = (
    "classic_flow2dh_adjacent_owner_open_boundary_equilibrium_v1"
)
ROUGHNESS_POLICY = "typed_raw_chezy_source_floor_rotation_unfloored_v1"
DEPTH_PAIR_POLICY = "same_transactional_trial_stage_v1"
H1_REFERENCE_POLICY = (
    "opaque_frozen_v55_reference_secondary.off_not_reinterpreted"
)
BASE_REFERENCE_POLICY = (
    "opaque_frozen_foundation_secondary.off_not_reinterpreted"
)

COMPOSITION_ORDER = (
    "transactional_trial_hydraulic_state",
    "classic_flow2dh_equilibrium_refresh",
    "transactional_trial_surface_active_volume",
    "optional_hiding_exposure_factors_once_per_trial_stage",
    "raw_scalar_cell_and_owner_indexed_boundary_capacity",
    "classic_flow2dh_raw_bed_rotation",
    "channel_composition_once",
    "face_reconstruction",
    "bed_slope_off",
    "donor_and_finite_availability_limiting",
    "conservative_flux_and_exner",
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SOURCE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]*\Z")


def _canonical(value: object) -> bytes:
    """Return the sole canonical ASCII JSON representation for this schema."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _fingerprint(body: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical(body)).hexdigest()


def _sha256(value: object, name: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"Q3-H2 {name} must be a lowercase SHA-256 fingerprint")
    return value


def _finite(value: object, name: str, *, positive: bool = False,
            nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Q3-H2 {name} must be a finite number")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"Q3-H2 {name} must be a finite number") from error
    if (not math.isfinite(result) or (positive and result <= 0.0) or
            (nonnegative and result < 0.0)):
        relation = "positive" if positive else "nonnegative" if nonnegative else "finite"
        raise ValueError(f"Q3-H2 {name} must be {relation}")
    # JSON distinguishes -0.0 from 0.0 although the operator does not.  One
    # canonical zero prevents two fingerprints for the same control value.
    return 0.0 if result == 0.0 else result


def secondary_foundation_execution_document(
    *,
    method_id: str,
    model: int,
    capacity_law: int,
    capacity_execution_fingerprint: str,
    capacity_profile_fingerprint: str,
    roughness_method_id: str,
    roughness_fingerprint: str,
    coriolis_disposition: str,
    coriolis_constant_s_inverse: float,
    coriolis_source: str,
    rmincf: float,
    chzmin: float,
    espir: float,
    betac: float,
    hiding_execution_fingerprint: str | None = None,
) -> dict[str, object]:
    """Build the canonical identity for one fixed-grid v56 composition.

    ``Betac`` is deliberately accepted at every finite value so the identity
    cannot collide when a later qualified route enables its stress branch.
    Runtime admission remains separate and may initially require ``betac=0``.
    """
    if method_id != METHOD_ID or type(model) is not int or model != MODEL:
        raise ValueError(
            "Q3-H2 requires classic FLOW secondary.equilibrium model 1"
        )
    if (type(capacity_law) is not int or
            capacity_law not in CAPACITY_METHOD_BY_LAW):
        raise ValueError("Q3-H2 capacity law must be 2, 8 or 11")
    if (not isinstance(roughness_method_id, str) or
            roughness_method_id not in ROUGHNESS_METHOD_IDS):
        raise ValueError("Q3-H2 requires a known typed C/n/ks roughness method")
    capacity_execution = _sha256(
        capacity_execution_fingerprint, "Q3-C execution fingerprint")
    capacity_profile = _sha256(
        capacity_profile_fingerprint, "Q3-C profile fingerprint")
    hiding_execution = _sha256(
        hiding_execution_fingerprint, "Q3-H1 execution fingerprint",
        optional=True)
    roughness = _sha256(
        roughness_fingerprint, "typed roughness fingerprint")
    if hiding_execution is not None and capacity_law not in (2, 11):
        raise ValueError(
            "Q3-H2 cannot compose Q3-H1 hiding with capacity law 8"
        )

    if (not isinstance(coriolis_disposition, str) or
            coriolis_disposition not in CORIOLIS_DISPOSITIONS):
        raise ValueError("Q3-H2 Coriolis disposition must be zero or nonzero")
    coriolis = _finite(
        coriolis_constant_s_inverse, "Coriolis constant")
    if ((coriolis_disposition == "zero" and coriolis != 0.0) or
            (coriolis_disposition == "nonzero" and coriolis == 0.0)):
        raise ValueError(
            "Q3-H2 Coriolis disposition and explicit value disagree"
        )
    if (not isinstance(coriolis_source, str) or
            _SOURCE_ID.fullmatch(coriolis_source) is None):
        raise ValueError(
            "Q3-H2 Coriolis source must be a nonempty canonical identifier"
        )
    if coriolis_source == "constant_zero_planar" and coriolis != 0.0:
        raise ValueError(
            "Q3-H2 constant_zero_planar Coriolis source requires zero"
        )

    resolved_rmincf = _finite(rmincf, "Rmincf", positive=True)
    resolved_chzmin = _finite(chzmin, "Chzmin", positive=True)
    resolved_espir = _finite(espir, "Espir", nonnegative=True)
    resolved_betac = _finite(betac, "Betac")
    has_hiding = hiding_execution is not None

    body: dict[str, object] = {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "envelope_version": ENVELOPE_VERSION,
        "grid_scope": FIXED_GRID_SCOPE,
        "method_id": method_id,
        "model": model,
        "selector_provenance": "Sub1=I;Equili=Y",
        "capacity_law": capacity_law,
        "capacity_method_id": CAPACITY_METHOD_BY_LAW[capacity_law],
        "base_envelope_version": 55 if has_hiding else 54,
        "base_reference_policy": BASE_REFERENCE_POLICY,
        "capacity_foundation_envelope_version": 54,
        "capacity_execution_fingerprint": capacity_execution,
        "capacity_profile_fingerprint": capacity_profile,
        "hiding_foundation_envelope_version": 55 if has_hiding else None,
        "hiding_execution_fingerprint": hiding_execution,
        "h1_reference_policy": H1_REFERENCE_POLICY if has_hiding else None,
        "roughness_method_id": roughness_method_id,
        "roughness_fingerprint": roughness,
        "roughness_policy": ROUGHNESS_POLICY,
        "coriolis_disposition": coriolis_disposition,
        "coriolis_constant_s_inverse": coriolis,
        "coriolis_source": coriolis_source,
        "rmincf": resolved_rmincf,
        "chzmin": resolved_chzmin,
        "espir": resolved_espir,
        "betac": resolved_betac,
        "curvature_discretization": CURVATURE_DISCRETIZATION,
        "curvature_cap_policy": CURVATURE_CAP_POLICY,
        "metric_policy": METRIC_POLICY,
        "depth_pair_policy": DEPTH_PAIR_POLICY,
        "refresh_timing": REFRESH_TIMING,
        "boundary_policy": BOUNDARY_POLICY,
        "composition_order": list(COMPOSITION_ORDER),
        "raw_bed_rotation_scope": "bedload_only_before_channel_composition",
        "suspended_channel_rotation": False,
        "downstream_numerics": (
            "existing_channel_composer_then_unchanged_face_reconstruction_"
            "availability_and_exner"
        ),
        "backend_selection": BACKEND_SELECTION,
        "automatic_cpu_fallback": False,
        "runtime_admission_authority": "separate_exact_route_gate",
    }
    return {**body, "fingerprint": _fingerprint(body)}


def validate_secondary_foundation_execution_document(
    value: Mapping[str, object],
) -> dict[str, object]:
    """Rebuild and byte-compare an untrusted Q3-H2 identity.

    Both Python equality and canonical JSON bytes are compared.  The first
    catches non-JSON substitutions such as tuple-for-list; the second catches
    Python-equal but byte-distinct substitutions such as integer-for-float.
    """
    if not isinstance(value, Mapping):
        raise ValueError("Q3-H2 secondary execution identity must be an object")
    try:
        rebuilt = secondary_foundation_execution_document(
            method_id=value["method_id"],
            model=value["model"],
            capacity_law=value["capacity_law"],
            capacity_execution_fingerprint=value[
                "capacity_execution_fingerprint"],
            capacity_profile_fingerprint=value[
                "capacity_profile_fingerprint"],
            hiding_execution_fingerprint=value[
                "hiding_execution_fingerprint"],
            roughness_method_id=value["roughness_method_id"],
            roughness_fingerprint=value["roughness_fingerprint"],
            coriolis_disposition=value["coriolis_disposition"],
            coriolis_constant_s_inverse=value[
                "coriolis_constant_s_inverse"],
            coriolis_source=value["coriolis_source"],
            rmincf=value["rmincf"],
            chzmin=value["chzmin"],
            espir=value["espir"],
            betac=value["betac"],
        )
        supplied_bytes = _canonical(value)
        rebuilt_bytes = _canonical(rebuilt)
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError("Invalid Q3-H2 secondary execution identity") from error
    if dict(value) != rebuilt or supplied_bytes != rebuilt_bytes:
        raise ValueError("Q3-H2 secondary execution identity is stale or forged")
    return rebuilt


__all__ = [
    "BACKEND_SELECTION",
    "BASE_REFERENCE_POLICY",
    "BOUNDARY_POLICY",
    "CAPACITY_METHOD_BY_LAW",
    "COMPOSITION_ORDER",
    "CORIOLIS_DISPOSITIONS",
    "CURVATURE_CAP_POLICY",
    "CURVATURE_DISCRETIZATION",
    "DEPTH_PAIR_POLICY",
    "ENVELOPE_VERSION",
    "EXECUTION_KEY",
    "FIXED_GRID_SCOPE",
    "H1_REFERENCE_POLICY",
    "METHOD_ID",
    "METRIC_POLICY",
    "MODEL",
    "ORDERED_ROUTE_KEY",
    "REFRESH_TIMING",
    "ROUGHNESS_METHOD_IDS",
    "ROUGHNESS_POLICY",
    "SCHEMA",
    "SCOPE_ID",
    "SEMANTIC_ID",
    "secondary_foundation_execution_document",
    "validate_secondary_foundation_execution_document",
]
