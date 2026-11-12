"""Versioned modern-keyword defaults for the six scalar FLOW laws.

The shared capacity-parameter schema predates the v64 scalar-law work and is
embedded in frozen v50--v62 profile and restart identities.  Delft3D's legacy
positional reader requires both TraFrm 17/18 values, while its modern keyword
reader initializes them from ``pardef``.  This additive resolver implements
the latter behavior without silently changing the established schema-v1 hash.
Every returned execution document contains the resolved numbers explicitly.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from types import MappingProxyType

from .capacity_parameters import (
    canonical_parameter_items,
    resolve_capacity_parameters,
)


SCHEMA = 1
SCOPE_ID = "braids-q3-scalar-flow-modern-keyword-defaults-v64-v1"
METHODS = frozenset((
    "capacity.van_rijn_1984_modified",
    "capacity.wang_fredsoe",
    "capacity.ashida_michiue_generalized",
    "capacity.wilcock_crowe",
    "capacity.wilcock_crowe_modified",
    "capacity.gaeuman",
))
FLOW_KEYWORD_DEFAULTS = MappingProxyType({
    "capacity.wilcock_crowe_modified": MappingProxyType({
        "ThetaC0": 0.021,
        "Alpha0": 0.33,
    }),
    "capacity.gaeuman": MappingProxyType({
        "ThetaC0": 0.03,
        "Alpha0": 0.30,
    }),
})


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def scalar_flow_parameter_semantic_document_v64() -> dict:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "source_reader": "rdtrafrm_modern_keyword_path",
        "frozen_schema_v1_changed": False,
        "methods": sorted(METHODS),
        "defaults": {
            method_id: dict(values)
            for method_id, values in sorted(FLOW_KEYWORD_DEFAULTS.items())
        },
        "execution_identity": "resolved_values_explicit_not_default_marker",
    }


SEMANTIC_ID = hashlib.sha256(
    _canonical(scalar_flow_parameter_semantic_document_v64())
).hexdigest()


def resolve_scalar_flow_parameters_v64(
    method_id: str,
    supplied: Mapping[str, float | int],
) -> Mapping[str, float | int]:
    """Resolve modern FLOW defaults while retaining shared-v1 validation."""
    if method_id not in METHODS:
        raise ValueError(f"Unsupported v64 scalar FLOW method: {method_id!r}")
    if not isinstance(supplied, Mapping):
        raise ValueError("v64 scalar FLOW parameters must be a mapping")
    merged = dict(FLOW_KEYWORD_DEFAULTS.get(method_id, {}))
    merged.update(supplied)
    return resolve_capacity_parameters(method_id, merged)


def canonical_scalar_flow_parameter_items_v64(
    method_id: str,
    supplied: Mapping[str, float | int],
) -> tuple[tuple[str, float | int], ...]:
    resolved = resolve_scalar_flow_parameters_v64(method_id, supplied)
    return canonical_parameter_items(method_id, resolved)


def scalar_flow_parameter_document_v64(
    method_id: str,
    supplied: Mapping[str, float | int],
) -> dict:
    items = canonical_scalar_flow_parameter_items_v64(method_id, supplied)
    body = {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "method_id": method_id,
        "parameters": [
            {"keyword": keyword, "value": value}
            for keyword, value in items
        ],
    }
    return {**body, "fingerprint": hashlib.sha256(_canonical(body)).hexdigest()}


def load_scalar_flow_parameter_document_v64(
    value: Mapping[str, object],
) -> dict:
    if not isinstance(value, Mapping):
        raise ValueError("v64 scalar FLOW parameter identity must be an object")
    try:
        parameters = value["parameters"]
        if not isinstance(parameters, list) or any(
                not isinstance(row, Mapping) or
                set(row) != {"keyword", "value"}
                for row in parameters):
            raise ValueError("v64 scalar FLOW parameter rows are malformed")
        rebuilt = scalar_flow_parameter_document_v64(
            str(value["method_id"]),
            {str(row["keyword"]): row["value"] for row in parameters},
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Invalid v64 scalar FLOW parameter identity") from error
    if dict(value) != rebuilt or _canonical(value) != _canonical(rebuilt):
        raise ValueError("v64 scalar FLOW parameter identity is stale or forged")
    return rebuilt


__all__ = [
    "SCHEMA", "SCOPE_ID", "SEMANTIC_ID", "METHODS",
    "FLOW_KEYWORD_DEFAULTS", "scalar_flow_parameter_semantic_document_v64",
    "resolve_scalar_flow_parameters_v64",
    "canonical_scalar_flow_parameter_items_v64",
    "scalar_flow_parameter_document_v64",
    "load_scalar_flow_parameter_document_v64",
]
