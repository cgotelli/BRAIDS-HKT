"""Additive positive-``TraFrm`` ISlope=2 receipt for the v67 family.

The v67 six-method family remains slope-off by default.  This envelope binds
one exact v67 execution to the classic FLOW Bagnold/Ikeda face component and
to the audited positive-selector ``taurat`` contract.  Five family methods
are positive selectors and may use exact zero.  Van Rijn 1993 is a negative
``tram1`` selector and is rejected here until its producer-owned live stress
ratio is carried through the same accepted-state transaction.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path

from .bed_slope_execution import validate_bed_slope_execution_document
from .classic_flow_taurat_v69 import (
    DIRECT_VECTOR_METHOD_SELECTORS,
    TAURAT_POSITIVE_ZERO,
    direct_vector_taurat_semantics,
    source_contract_document_v69,
    verify_locked_sources_v69,
)
from .direct_vector_family_live_v67 import (
    load_direct_vector_family_execution_v67,
    verify_direct_vector_family_sources_v67,
)


SCHEMA = 1
ENVELOPE_VERSION = 69
SCOPE_ID = "braids-q3h7-direct-vector-positive-taurat-live-v69-v1"
MAX_ACCEPTED_STEPS = 4
METHOD_ID = "bed_slope.delft_bagnold_ikeda"

_COMMON_IMPLEMENTATION_PATHS = (
    "braids_core/sediment/classic_flow_taurat_v69.py",
    "braids_core/sediment/direct_vector_positive_taurat_v69.py",
)
_BACKEND_IMPLEMENTATION_PATHS = {
    "cpu": (
        "solver/wave/braids_direct_vector_family_native_v67_module.f90",
        "solver/wave/braids_scalar_flow_live_v65_module.f90",
    ),
    "implicit_fwave_midpoint_gpu": (
        "braids_gpu/direct_vector_positive_taurat_v69.py",
        "braids_gpu/direct_vector_face_exner_v67.py",
    ),
}


class DirectVectorPositiveTauratV69Error(ValueError):
    """The additive positive-selector slope receipt is invalid."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise DirectVectorPositiveTauratV69Error(
            "Q3-H7/v69 receipt is not canonical JSON"
        ) from error


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _implementation_sources(root: Path, backend: str) -> list[dict[str, str]]:
    try:
        paths = _COMMON_IMPLEMENTATION_PATHS + _BACKEND_IMPLEMENTATION_PATHS[
            backend
        ]
    except KeyError as error:
        raise DirectVectorPositiveTauratV69Error(
            "Q3-H7/v69 backend is outside the CPU/CUDA family"
        ) from error
    rows = []
    for relative in paths:
        path = root / relative
        if not path.is_file():
            raise DirectVectorPositiveTauratV69Error(
                f"Q3-H7/v69 implementation source is missing: {relative}"
            )
        rows.append({
            "path": relative,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    return rows


def direct_vector_positive_taurat_execution_document_v69(
    *,
    parent_execution: Mapping[str, object],
    bed_slope_execution: Mapping[str, object],
    repository_root: str | Path | None = None,
) -> dict[str, object]:
    """Bind a positive-only v67 family to one exact-zero ISlope=2 stage."""
    try:
        parent = load_direct_vector_family_execution_v67(parent_execution)
        slope = validate_bed_slope_execution_document(bed_slope_execution)
    except (TypeError, ValueError) as error:
        raise DirectVectorPositiveTauratV69Error(
            "Q3-H7/v69 parent or slope receipt is invalid"
        ) from error
    if slope["method_id"] != METHOD_ID or slope["runtime_mode"] != 2:
        raise DirectVectorPositiveTauratV69Error(
            "Q3-H7/v69 requires classic FLOW ISlope=2"
        )
    physical = parent["physical_profile"]
    methods = physical["fraction_methods"]
    fraction_ids = physical["fraction_ids"]
    if slope["ordered_fraction_ids"] != fraction_ids:
        raise DirectVectorPositiveTauratV69Error(
            "Q3-H7/v69 slope fractions differ from the family order"
        )
    upstream = {
        row["stage"]: row["fingerprint"]
        for row in slope["upstream_identities"]
    }
    if upstream.get("capacity_execution") != parent["fingerprint"]:
        raise DirectVectorPositiveTauratV69Error(
            "Q3-H7/v69 slope capacity parent differs from the v67 execution"
        )

    method_rows = []
    for row in methods:
        method_id = row["method_id"]
        selector = row["reference_selector"]
        expected = DIRECT_VECTOR_METHOD_SELECTORS.get(method_id)
        if expected != selector:
            raise DirectVectorPositiveTauratV69Error(
                "Q3-H7/v69 method/TraFrm identity differs from the audit"
            )
        semantics = direct_vector_taurat_semantics(method_id)
        if semantics != TAURAT_POSITIVE_ZERO:
            raise DirectVectorPositiveTauratV69Error(
                "Q3-H7/v69 rejects negative tram methods until their live "
                "producer-owned taurat is carried"
            )
        method_rows.append({
            "fraction_index": row["fraction_index"],
            "fraction_id": row["fraction_id"],
            "method_id": method_id,
            "reference_selector": selector,
            "taurat_semantics": semantics,
        })

    root = (Path(repository_root) if repository_root is not None else
            Path(__file__).resolve().parents[2])
    contract = verify_locked_sources_v69(root)
    sources = _implementation_sources(root, parent["backend"])
    body: dict[str, object] = {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "envelope_version": ENVELOPE_VERSION,
        "parent_execution": parent,
        "parent_execution_fingerprint": parent["fingerprint"],
        "bed_slope_execution": slope,
        "bed_slope_execution_fingerprint": slope["fingerprint"],
        "taurat_source_contract": contract,
        "taurat_source_contract_fingerprint": contract["fingerprint"],
        "backend": parent["backend"],
        "integrator": parent["integrator"],
        "method_id": METHOD_ID,
        "reference_islope": 2,
        "stress_semantics": TAURAT_POSITIVE_ZERO,
        "fraction_methods": method_rows,
        "positive_diagnostic_substitution_allowed": False,
        "composition_order": (
            "v67_fraction_free_direct_producers_then_raw_face_"
            "reconstruction_then_one_islope2_then_inventory_then_exner"
        ),
        "accepted_step_limit": MAX_ACCEPTED_STEPS,
        "automatic_backend_fallback": False,
        "normal_product_admission": False,
        "implementation_sources": sources,
        "implementation_source_fingerprint": _fingerprint(sources),
    }
    return {**body, "fingerprint": _fingerprint(body)}


def load_direct_vector_positive_taurat_execution_v69(
    value: Mapping[str, object], *,
    repository_root: str | Path | None = None,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise DirectVectorPositiveTauratV69Error(
            "Q3-H7/v69 execution must be an object"
        )
    try:
        rebuilt = direct_vector_positive_taurat_execution_document_v69(
            parent_execution=value["parent_execution"],
            bed_slope_execution=value["bed_slope_execution"],
            repository_root=repository_root,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise DirectVectorPositiveTauratV69Error(
            "Invalid Q3-H7/v69 execution identity"
        ) from error
    if dict(value) != rebuilt or _canonical(value) != _canonical(rebuilt):
        raise DirectVectorPositiveTauratV69Error(
            "Q3-H7/v69 execution identity is stale or forged"
        )
    return rebuilt


def verify_direct_vector_positive_taurat_sources_v69(
    execution: Mapping[str, object], *,
    repository_root: str | Path | None = None,
) -> dict[str, object]:
    resolved = load_direct_vector_positive_taurat_execution_v69(
        execution, repository_root=repository_root,
    )
    root = (Path(repository_root) if repository_root is not None else
            Path(__file__).resolve().parents[2])
    for row in resolved["implementation_sources"]:
        path = root / row["path"]
        if (not path.is_file() or
                hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]):
            raise DirectVectorPositiveTauratV69Error(
                f"Q3-H7/v69 implementation source differs: {row['path']}"
            )
    parent_receipt = verify_direct_vector_family_sources_v67(
        resolved["parent_execution"], repository_root=root,
    )
    return {
        "scope_id": SCOPE_ID,
        "execution_fingerprint": resolved["fingerprint"],
        "taurat_source_contract_fingerprint":
            resolved["taurat_source_contract_fingerprint"],
        "implementation_source_fingerprint":
            resolved["implementation_source_fingerprint"],
        "parent_source_receipt_fingerprint": parent_receipt["fingerprint"],
        "verified": True,
    }


__all__ = [
    "DirectVectorPositiveTauratV69Error", "ENVELOPE_VERSION",
    "MAX_ACCEPTED_STEPS", "METHOD_ID", "SCHEMA", "SCOPE_ID",
    "direct_vector_positive_taurat_execution_document_v69",
    "load_direct_vector_positive_taurat_execution_v69",
    "verify_direct_vector_positive_taurat_sources_v69",
]
