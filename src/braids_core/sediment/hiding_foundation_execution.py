"""Additive Q3-H1 hiding/exposure execution identity.

Q3-C's capacity/forcing identity is frozen and deliberately describes a
capacity path with generic hiding disabled.  Q3-H1 composes one additional
trial-stage operator around that path.  Factors are evaluated from the
transactional trial surface; only a whole accepted candidate is committed.
This document binds the exact operator, controls and ordered two-fraction
surface-bed ownership without reinterpreting any Q3-C fingerprint.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence


SCHEMA = 1
SCOPE_ID = "braids-q3h1-hiding-exposure-composition-v1"
SEMANTIC_ID = hashlib.sha256(SCOPE_ID.encode("ascii")).hexdigest()
EXECUTION_KEY = "hiding_foundation_execution_identity"
ORDERED_ROUTE_KEY = "hiding_foundation_ordered_route_identity"
ACTIVE_VOLUME_SOURCE = (
    "transactional_trial_surface_layer_mass_divided_by_bulk_density"
)
COMPOSITION_ORDER = (
    "transactional_trial_surface_active_volume",
    "hiding_exposure_factors_once_per_trial_stage",
    "cell_and_owner_indexed_boundary_capacity",
    "channel_composition",
    "face_reconstruction_and_exner",
)
METHOD_BY_MODE = {
    2: "hiding.egiazaroff",
    3: "hiding.ashida_michiue",
    4: "hiding.parker_klm",
    5: "hiding.wu_wang_jia",
}
_SHA = re.compile(r"[0-9a-f]{64}\Z")


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _fingerprint(body: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical(body)).hexdigest()


def hiding_foundation_execution_document(
    *, mode: int, method_id: str, capacity_law: int,
    fraction_ids: Sequence[str], parker_alpha: float, wu_exponent: float,
    capacity_execution_fingerprint: str,
    capacity_profile_fingerprint: str,
) -> dict[str, object]:
    """Return the canonical identity for the bounded Q3-H1 composition.

    Both calibration controls are bound for every mode.  This prevents a
    saved mode-2/3 run from later being relabelled as mode 4/5 under controls
    that were never part of its restart identity.
    """
    if isinstance(mode, bool) or mode not in METHOD_BY_MODE:
        raise ValueError("Q3-H1 hiding mode must be the integer 2, 3, 4 or 5")
    if METHOD_BY_MODE[mode] != method_id:
        raise ValueError("Q3-H1 hiding method does not match its Delft3D mode")
    if isinstance(capacity_law, bool) or capacity_law not in (2, 11):
        raise ValueError("Q3-H1 supports corrected MPM 2 or general law 11")
    ids = tuple(fraction_ids)
    if (len(ids) != 2 or len(set(ids)) != 2 or
            any(not isinstance(value, str) or not value or
                value != value.strip() for value in ids)):
        raise ValueError("Q3-H1 requires two unique ordered fraction identities")
    if (isinstance(parker_alpha, bool) or
            isinstance(wu_exponent, bool) or
            not math.isfinite(float(parker_alpha)) or
            not math.isfinite(float(wu_exponent))):
        raise ValueError("Q3-H1 Parker/Wu controls must be finite")
    if (not isinstance(capacity_execution_fingerprint, str) or
            _SHA.fullmatch(capacity_execution_fingerprint) is None or
            not isinstance(capacity_profile_fingerprint, str) or
            _SHA.fullmatch(capacity_profile_fingerprint) is None):
        raise ValueError("Q3-H1 base capacity fingerprints are invalid")
    body: dict[str, object] = {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "mode": mode,
        "method_id": method_id,
        "capacity_law": capacity_law,
        "fraction_ids": list(ids),
        "parker_alpha": float(parker_alpha),
        "wu_exponent": float(wu_exponent),
        "active_volume_source": ACTIVE_VOLUME_SOURCE,
        "commit_policy": "only_whole_accepted_candidate_commits",
        "composition_order": list(COMPOSITION_ORDER),
        "capacity_execution_fingerprint": capacity_execution_fingerprint,
        "capacity_profile_fingerprint": capacity_profile_fingerprint,
        "slope_operator": "bed_slope.off",
        "secondary_flow_operator": "secondary.off",
        "exchange_operator": "suspension.off",
        "backend_selection": "explicit_user_choice",
        "automatic_cpu_fallback": False,
    }
    return {**body, "fingerprint": _fingerprint(body)}


def validate_hiding_foundation_execution_document(
    value: Mapping[str, object],
) -> dict[str, object]:
    """Rebuild and byte-compare an untrusted Q3-H1 identity."""
    if not isinstance(value, Mapping):
        raise ValueError("Q3-H1 hiding execution identity must be an object")
    try:
        rebuilt = hiding_foundation_execution_document(
            mode=value["mode"], method_id=value["method_id"],
            capacity_law=value["capacity_law"],
            fraction_ids=value["fraction_ids"],
            parker_alpha=value["parker_alpha"],
            wu_exponent=value["wu_exponent"],
            capacity_execution_fingerprint=value[
                "capacity_execution_fingerprint"],
            capacity_profile_fingerprint=value[
                "capacity_profile_fingerprint"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Invalid Q3-H1 hiding execution identity") from error
    if _canonical(value) != _canonical(rebuilt):
        raise ValueError("Q3-H1 hiding execution identity is stale or forged")
    return rebuilt


__all__ = [
    "ACTIVE_VOLUME_SOURCE", "COMPOSITION_ORDER", "EXECUTION_KEY",
    "METHOD_BY_MODE", "ORDERED_ROUTE_KEY", "SCHEMA", "SCOPE_ID",
    "SEMANTIC_ID", "hiding_foundation_execution_document",
    "validate_hiding_foundation_execution_document",
]
