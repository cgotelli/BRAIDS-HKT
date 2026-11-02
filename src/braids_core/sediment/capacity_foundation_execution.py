"""Additive Q3-C forcing/execution identity.

This identity binds the exact typed C/n/ks forcing bytes before the canonical
capacity-profile fingerprint.  It is distinct from the historical Q2 v50
method documents; the v50 wire slot is reused only as an ordered envelope.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping


SCHEMA = 1
SCOPE_ID = "braids-q3c-capacity-forcing-execution-v1"
SEMANTIC_ID = hashlib.sha256(SCOPE_ID.encode("ascii")).hexdigest()
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_METHODS = {2: "capacity.mpm", 8: "capacity.engelund_hansen",
            11: "capacity.general"}
_ROUGHNESS = frozenset((
    "roughness.chezy", "roughness.manning", "roughness.white_colebrook",
    "roughness.white_colebrook_waqua"))


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def capacity_foundation_execution_document(*, method_id: str,
        formula_code: int, forcing_fingerprint: str,
        roughness_method_id: str, capacity_profile_fingerprint: str) -> dict:
    if (isinstance(formula_code, bool) or formula_code not in _METHODS or
            _METHODS[formula_code] != method_id or
            roughness_method_id not in _ROUGHNESS or
            not isinstance(forcing_fingerprint, str) or
            _SHA.fullmatch(forcing_fingerprint) is None or
            not isinstance(capacity_profile_fingerprint, str) or
            _SHA.fullmatch(capacity_profile_fingerprint) is None):
        raise ValueError("Invalid Q3-C capacity forcing/execution identity")
    body = {
        "schema": SCHEMA, "scope_id": SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "method_id": method_id, "formula_code": formula_code,
        "forcing_fingerprint": forcing_fingerprint,
        "roughness_method_id": roughness_method_id,
        "capacity_profile_fingerprint": capacity_profile_fingerprint,
    }
    return {**body, "fingerprint": hashlib.sha256(_canonical(body)).hexdigest()}


def load_capacity_foundation_execution(value: Mapping[str, object]) -> dict:
    if not isinstance(value, Mapping):
        raise ValueError("Q3-C capacity execution identity must be an object")
    expected = {"schema", "scope_id", "semantic_id", "method_id",
                "formula_code", "forcing_fingerprint", "roughness_method_id",
                "capacity_profile_fingerprint", "fingerprint"}
    if set(value) != expected:
        raise ValueError("Q3-C capacity execution identity inventory differs")
    rebuilt = capacity_foundation_execution_document(
        method_id=value["method_id"], formula_code=value["formula_code"],
        forcing_fingerprint=value["forcing_fingerprint"],
        roughness_method_id=value["roughness_method_id"],
        capacity_profile_fingerprint=value["capacity_profile_fingerprint"])
    if dict(value) != rebuilt:
        raise ValueError("Q3-C capacity execution fingerprint differs")
    return rebuilt


__all__ = ["SCHEMA", "SCOPE_ID", "SEMANTIC_ID",
           "capacity_foundation_execution_document",
           "load_capacity_foundation_execution"]
