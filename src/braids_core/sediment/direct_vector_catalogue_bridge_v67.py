"""Fail-closed public-catalogue bridge for the Q3-H6 direct-vector family.

The live FLOW-2DH catalogue is the public configuration namespace.  Three
current-only v66/v67 producer profiles deliberately have more specific
internal names because their wave-enabled variants are not implemented.  A
QGIS or project document must never guess that relationship.  This module
records the mapping explicitly and binds it to the authenticated live
catalogue identity before a family execution receipt is built.

This bridge establishes identity only.  It does not grant exact-profile or
product admission; those remain properties of the live catalogue and the
later Q10 product envelope.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json

from .direct_vector_family_live_v67 import (
    METHOD_BIJKER_CURRENT,
    METHOD_SOULSBY_CURRENT,
    METHOD_SOULSBY_VAN_RIJN_CURRENT,
    METHOD_VAN_RIJN_1993_CURRENT,
    METHOD_VAN_THIEL_VAN_RIJN_2008_CURRENT,
    METHOD_XBEACH_SOULSBY_VAN_RIJN_CURRENT,
)
from .live_catalogue_v1 import load_live_catalogue


SCHEMA = 1
SCOPE_ID = "braids-q3-h6-direct-vector-catalogue-bridge-v67-v1"
FORCING_SCOPE = "current_only"

CATALOGUE_TO_PROFILE = {
    "capacity.soulsby": METHOD_SOULSBY_CURRENT,
    "capacity.soulsby_van_rijn": METHOD_SOULSBY_VAN_RIJN_CURRENT,
    "capacity.van_rijn_1993": METHOD_VAN_RIJN_1993_CURRENT,
    "capacity.bijker": METHOD_BIJKER_CURRENT,
    "capacity.van_thiel_van_rijn": METHOD_VAN_THIEL_VAN_RIJN_2008_CURRENT,
    "capacity.xbeach_soulsby_van_rijn": (
        METHOD_XBEACH_SOULSBY_VAN_RIJN_CURRENT
    ),
}
PROFILE_TO_CATALOGUE = {
    profile_id: catalogue_id
    for catalogue_id, profile_id in CATALOGUE_TO_PROFILE.items()
}


class DirectVectorCatalogueBridgeV67Error(ValueError):
    """A public method identity or bridge document is invalid."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise DirectVectorCatalogueBridgeV67Error(
            "Q3-H6/v67 catalogue bridge is not canonical JSON"
        ) from error


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _catalogue_identity() -> tuple[dict[str, object], dict[str, dict], str]:
    catalogue = load_live_catalogue()
    identity = catalogue.get("identity")
    if not isinstance(identity, Mapping):
        raise DirectVectorCatalogueBridgeV67Error(
            "Q3-H6/v67 live catalogue lacks an authenticated identity"
        )
    lifecycle = catalogue.get("entry_lifecycle")
    if not isinstance(lifecycle, list):
        raise DirectVectorCatalogueBridgeV67Error(
            "Q3-H6/v67 live catalogue lifecycle is malformed"
        )
    rows = {
        row["entry_id"]: row
        for row in lifecycle
        if isinstance(row, Mapping) and isinstance(row.get("entry_id"), str)
    }
    revision = catalogue.get("revision_id")
    if not isinstance(revision, str) or not revision:
        raise DirectVectorCatalogueBridgeV67Error(
            "Q3-H6/v67 live catalogue revision is malformed"
        )
    return dict(identity), rows, revision


def direct_vector_catalogue_bridge_document_v67(
    catalogue_entry_ids: Sequence[str], *, forcing_scope: str = FORCING_SCOPE,
) -> dict[str, object]:
    """Resolve public catalogue IDs to explicit current-only profile IDs."""
    if forcing_scope != FORCING_SCOPE:
        raise DirectVectorCatalogueBridgeV67Error(
            "Q3-H6/v67 direct-vector family supports current_only forcing"
        )
    if (not isinstance(catalogue_entry_ids, Sequence) or
            isinstance(catalogue_entry_ids, (str, bytes, bytearray)) or
            not catalogue_entry_ids):
        raise DirectVectorCatalogueBridgeV67Error(
            "Q3-H6/v67 requires one public catalogue entry per fraction"
        )

    identity, lifecycle, revision = _catalogue_identity()
    rows: list[dict[str, object]] = []
    for fraction_index, entry_id in enumerate(catalogue_entry_ids):
        if type(entry_id) is not str:
            raise DirectVectorCatalogueBridgeV67Error(
                "Q3-H6/v67 public method identities must be strings"
            )
        try:
            profile_id = CATALOGUE_TO_PROFILE[entry_id]
            entry = lifecycle[entry_id]
        except KeyError as error:
            raise DirectVectorCatalogueBridgeV67Error(
                f"Q3-H6/v67 unsupported public catalogue entry: {entry_id!r}"
            ) from error
        if entry.get("kind") != "method" or entry.get("family") != "capacity":
            raise DirectVectorCatalogueBridgeV67Error(
                f"Q3-H6/v67 catalogue entry has the wrong kind: {entry_id}"
            )
        rows.append({
            "fraction_index": fraction_index,
            "catalogue_entry_id": entry_id,
            "producer_profile_method_id": profile_id,
            "profile_variant": FORCING_SCOPE,
            "mapping_kind": (
                "identity" if entry_id == profile_id
                else "explicit_current_only_profile"
            ),
        })

    body: dict[str, object] = {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "forcing_scope": FORCING_SCOPE,
        "live_catalogue_revision": revision,
        "live_catalogue_manifest_id": identity["live_catalogue_manifest_id"],
        "fractions": rows,
        "public_namespace": "flow2dh_live_catalogue_entry_id",
        "internal_namespace": "direct_vector_family_v67_profile_method_id",
        "automatic_aliasing": False,
        "exact_profile_admission_granted": False,
        "product_admission_granted": False,
    }
    return {**body, "fingerprint": _fingerprint(body)}


def load_direct_vector_catalogue_bridge_v67(
    value: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise DirectVectorCatalogueBridgeV67Error(
            "Q3-H6/v67 catalogue bridge must be an object"
        )
    try:
        fractions = value["fractions"]
        if not isinstance(fractions, list):
            raise TypeError("fractions")
        entry_ids = [row["catalogue_entry_id"] for row in fractions]
        rebuilt = direct_vector_catalogue_bridge_document_v67(
            entry_ids, forcing_scope=value["forcing_scope"]
        )
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise DirectVectorCatalogueBridgeV67Error(
            "Invalid Q3-H6/v67 catalogue bridge"
        ) from error
    if dict(value) != rebuilt or _canonical(value) != _canonical(rebuilt):
        raise DirectVectorCatalogueBridgeV67Error(
            "Q3-H6/v67 catalogue bridge is stale or forged"
        )
    return rebuilt


def direct_vector_profile_method_ids_v67(
    bridge: Mapping[str, object],
) -> tuple[str, ...]:
    resolved = load_direct_vector_catalogue_bridge_v67(bridge)
    return tuple(
        row["producer_profile_method_id"]
        for row in resolved["fractions"]
    )


__all__ = [
    "CATALOGUE_TO_PROFILE", "FORCING_SCOPE", "PROFILE_TO_CATALOGUE",
    "SCHEMA", "SCOPE_ID", "DirectVectorCatalogueBridgeV67Error",
    "direct_vector_catalogue_bridge_document_v67",
    "direct_vector_profile_method_ids_v67",
    "load_direct_vector_catalogue_bridge_v67",
]
