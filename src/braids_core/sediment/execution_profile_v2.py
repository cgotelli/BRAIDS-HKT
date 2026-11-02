"""Canonical Delft3D FLOW 2DH sediment execution profile, schema v2.

This module is the additive identity envelope above the established capacity,
suspension, channel-composition and future morphology component profiles.  It
does not reinterpret schema-v1 projects or grant a numerical route merely
because a selector exists in the source ledger.  Runtime admission remains
explicit and fail closed for the selected backend.

The profile deliberately keeps three inventories separate:

* physical/constitutive method selections;
* scoped physical parameters and state inputs; and
* numerical/comparison controls.

That separation is part of the hashed semantic contract.  It prevents a new
parameter or a Delft comparison switch from inflating method coverage, and it
prevents fraction/boundary/layer fields from being silently broadcast as
model constants.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from braids_core.sediment_method_catalog import semantic_catalog_id


EXECUTION_PROFILE_SCHEMA = 2
EXECUTION_PROFILE_SCOPE_ID = "delft3d4-flow-2dh-execution-profile-v2"
SELECTOR_LEDGER_SCHEMA = 2
SELECTOR_LEDGER_SCOPE_ID = "delft3d4_flow_2dh_river_sediment_morphodynamics"
SELECTOR_LEDGER_FILE = Path(__file__).with_name("flow2dh_selector_ledger_v2.json")
ADMISSION_REGISTRY_SCHEMA = 1
FROZEN_Q3SP_LEDGER_SEMANTIC_ID = (
    "0d40b66e809d06cf4e08294ef239cde36d7280e548fdb90c8f5333e68dc694eb"
)
_FROZEN_Q3SP_METHOD_IDS = frozenset((
    "bed_slope.off", "capacity.general", "hiding.none", "secondary.off",
    "shear.total_roughness", "suspension.off", "transport_route.total",
))

BACKEND_CPU = "cpu"
BACKEND_GPU = "implicit_fwave_midpoint_gpu"
_BACKENDS = frozenset((BACKEND_CPU, BACKEND_GPU))
_ADMITTED = frozenset(("public", "public_limited"))
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_KINDS = frozenset(("method", "parameter", "control"))
_REPRESENTATIONS = frozenset(("inline", "artifact"))
_SCOPE_KEYS = frozenset((
    "model_id", "fraction_id", "boundary_id", "layer_id", "cell_id", "face_id",
))
_OWNER_KEY = {
    "model": "model_id",
    "fraction": "fraction_id",
    "boundary": "boundary_id",
    "layer": "layer_id",
    "cell": "cell_id",
    "face": "face_id",
}


class ExecutionProfileError(ValueError):
    """Fail-closed profile error with a stable machine-readable category."""

    def __init__(self, message: str, *, category: str):
        super().__init__(message)
        self.category = category


@dataclass(frozen=True)
class FractionLayout:
    fraction_id: str
    transport_class: int


@dataclass(frozen=True)
class MethodSelection:
    entry_id: str
    scope: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class ValueBinding:
    entry_id: str
    representation: str
    scope: tuple[tuple[str, str], ...]
    value: object


@dataclass(frozen=True)
class ComponentProfileBinding:
    component_id: str
    schema: int
    scope_id: str
    semantic_id: str
    fingerprint: str
    document_sha256: str


@dataclass(frozen=True)
class Flow2DHExecutionProfile:
    schema: int
    scope_id: str
    semantic_id: str
    ledger_semantic_id: str
    legacy_catalog_id: str
    backend: str | None
    fraction_layout: tuple[FractionLayout, ...]
    method_selections: tuple[MethodSelection, ...]
    parameter_bindings: tuple[ValueBinding, ...]
    control_bindings: tuple[ValueBinding, ...]
    component_profiles: tuple[ComponentProfileBinding, ...]
    fingerprint: str

    def as_dict(self) -> dict[str, object]:
        return execution_profile_document(self)


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ExecutionProfileError(
            "Execution profile is not canonical JSON",
            category="invalid_serialization",
        ) from error


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def selector_catalogue_semantic_payload(
    document: Mapping[str, object],
) -> dict[str, object]:
    """Return the stable physical-selector portion of a ledger.

    Backend implementation status and profile-level evidence are deliberately
    excluded. They may advance after validation without changing the physical
    method catalogue embedded in execution-profile fingerprints and native
    binaries.
    """
    allowed = document.get("allowed_values")
    allowed_values = copy.deepcopy(allowed) if isinstance(allowed, Mapping) else allowed
    if isinstance(allowed_values, dict):
        allowed_values.pop("backend_status", None)
    entries: list[object] = []
    raw_entries = document.get("entries")
    if isinstance(raw_entries, list):
        for raw in raw_entries:
            if isinstance(raw, Mapping):
                row = copy.deepcopy(dict(raw))
                row.pop("backend_status", None)
                entries.append(row)
            else:
                entries.append(copy.deepcopy(raw))
    else:
        entries = copy.deepcopy(raw_entries)  # type: ignore[assignment]
    return {
        "schema_version": copy.deepcopy(document.get("schema_version")),
        "scope_id": copy.deepcopy(document.get("scope_id")),
        "source": copy.deepcopy(document.get("source")),
        "allowed_values": allowed_values,
        "required_families": copy.deepcopy(document.get("required_families")),
        "entries": entries,
        "compatibility_rules": copy.deepcopy(document.get("compatibility_rules")),
    }


def admission_registry_payload(
    document: Mapping[str, object],
) -> dict[str, object]:
    """Return mutable backend admission/status evidence under its own ID."""
    rows: list[object] = []
    raw_entries = document.get("entries")
    if isinstance(raw_entries, list):
        for raw in raw_entries:
            if isinstance(raw, Mapping):
                rows.append({
                    "entry_id": copy.deepcopy(raw.get("entry_id")),
                    "backend_status": copy.deepcopy(raw.get("backend_status")),
                })
            else:
                rows.append(copy.deepcopy(raw))
    else:
        rows = copy.deepcopy(raw_entries)  # type: ignore[assignment]
    return {
        "schema": ADMISSION_REGISTRY_SCHEMA,
        "scope_id": copy.deepcopy(document.get("scope_id")),
        "selector_semantic_id": copy.deepcopy(document.get("semantic_id")),
        "entry_backend_status": rows,
        "backend_profile_admissions": copy.deepcopy(
            document.get("backend_profile_admissions")
        ),
    }


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ExecutionProfileError(
                f"Duplicate JSON key: {key!r}", category="invalid_serialization"
            )
        result[key] = value
    return result


def _read_json(value: str | bytes | Mapping[str, object], *, what: str) -> dict:
    if isinstance(value, Mapping):
        raw = json.loads(_canonical_json(value))
    elif isinstance(value, (str, bytes)):
        try:
            raw = json.loads(
                value,
                object_pairs_hook=_strict_object,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ExecutionProfileError(
                        f"Non-finite JSON number: {token}",
                        category="invalid_serialization",
                    )
                ),
            )
        except ExecutionProfileError:
            raise
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ExecutionProfileError(
                f"Invalid {what} JSON", category="invalid_serialization"
            ) from error
    else:
        raise ExecutionProfileError(
            f"{what} must be JSON text or an object",
            category="invalid_serialization",
        )
    if not isinstance(raw, dict):
        raise ExecutionProfileError(
            f"{what} must be a JSON object", category="invalid_serialization"
        )
    return raw


def execution_profile_semantic_document() -> dict[str, object]:
    """Static meaning of the schema, independent of a particular ledger."""
    return {
        "schema": EXECUTION_PROFILE_SCHEMA,
        "scope_id": EXECUTION_PROFILE_SCOPE_ID,
        "canonicalization": {
            "encoding": "ascii",
            "json": "sort_keys_compact_no_nan_ensure_ascii",
            "fingerprint": "sha256_of_payload_without_fingerprint",
        },
        "backend_values": [BACKEND_CPU, BACKEND_GPU, None],
        "ordered_fields": [
            "fraction_layout", "method_selections", "parameter_bindings",
            "control_bindings", "component_profiles",
        ],
        "scope_contract": {
            "dimensions": sorted(_OWNER_KEY),
            "binding_keys": sorted(_SCOPE_KEYS),
            "no_implicit_broadcast": True,
            "spatial_or_time_data": "inline_exact_or_sha256_artifact",
        },
        "coverage_contract": {
            "method_denominator": "ledger_entries_kind_method_and_counts_as_method_true",
            "parameters_and_controls_counted_separately": True,
        },
        "compatibility_contract": {
            "rules_are_executable": True,
            "unknown_or_unsupported_entries_fail_before_step": True,
            "backend_admission": sorted(_ADMITTED),
        },
        "component_binding": (
            "component id/schema/scope/semantic/fingerprint plus canonical document sha256"
        ),
        "legacy_contract": (
            "schema-v1 semantic catalogue remains separately identified and unchanged"
        ),
    }


def execution_profile_semantic_id() -> str:
    return _sha(execution_profile_semantic_document())


def load_selector_ledger(
    path: str | Path = SELECTOR_LEDGER_FILE,
) -> dict[str, object]:
    """Load the installed source ledger and verify its self identity.

    The full source/field validator lives with the generated evidence.  This
    runtime reader verifies the immutable wire properties it consumes and
    never follows a source path from the ledger.
    """
    ledger_path = Path(path)
    if not ledger_path.is_file():
        raise ExecutionProfileError(
            f"Installed selector ledger is missing: {ledger_path}",
            category="missing_ledger",
        )
    raw = _read_json(ledger_path.read_bytes(), what="selector ledger")
    legacy_required = {
        "schema_version", "scope_id", "semantic_id", "source", "allowed_values",
        "required_families", "entries", "compatibility_rules",
        "backend_profile_admissions", "counts", "inventory_notes",
    }
    current_required = legacy_required | {"admission_registry_id"}
    if set(raw) not in (legacy_required, current_required):
        raise ExecutionProfileError(
            "Selector ledger top-level inventory differs",
            category="ledger_integrity",
        )
    if (raw["schema_version"] != SELECTOR_LEDGER_SCHEMA or
            raw["scope_id"] != SELECTOR_LEDGER_SCOPE_ID or
            not isinstance(raw["semantic_id"], str) or
            _SHA256.fullmatch(raw["semantic_id"]) is None or
            not isinstance(raw["entries"], list) or
            not isinstance(raw["compatibility_rules"], list)):
        raise ExecutionProfileError(
            "Selector ledger schema or semantic identity differs",
            category="ledger_integrity",
        )
    if "admission_registry_id" in raw:
        registry_id = raw["admission_registry_id"]
        if _sha(selector_catalogue_semantic_payload(raw)) != raw["semantic_id"]:
            raise ExecutionProfileError(
                "Selector ledger semantic hash for the physical catalogue differs",
                category="ledger_integrity",
            )
        if (not isinstance(registry_id, str) or
                _SHA256.fullmatch(registry_id) is None or
                _sha(admission_registry_payload(raw)) != registry_id):
            raise ExecutionProfileError(
                "Selector ledger admission registry hash differs",
                category="ledger_integrity",
            )
    else:
        # Additive compatibility for persisted schema-v2 ledgers. Their one
        # identity covered both catalogue and admission state; new ledgers use
        # the split identities above. Frozen Q3-S/P projects retain this path.
        legacy_payload = {
            key: raw[key] for key in raw if key not in {"semantic_id", "counts"}
        }
        if (_sha(legacy_payload) != raw["semantic_id"] or
                raw["semantic_id"] != FROZEN_Q3SP_LEDGER_SEMANTIC_ID):
            raise ExecutionProfileError(
                "Selector ledger legacy semantic identity is not the frozen Q3-S/P ledger",
                category="ledger_integrity",
            )
    entries = raw["entries"]
    ids: set[str] = set()
    for entry in entries:
        if (not isinstance(entry, dict) or not isinstance(entry.get("entry_id"), str)
                or not entry["entry_id"] or entry["entry_id"] in ids
                or entry.get("kind") not in _KINDS):
            raise ExecutionProfileError(
                "Selector ledger contains a malformed or duplicate entry",
                category="ledger_integrity",
            )
        ids.add(entry["entry_id"])
        counts = entry.get("counts_as_method")
        if type(counts) is not bool or counts != (entry["kind"] == "method"):
            raise ExecutionProfileError(
                f"Selector ledger kind/count invariant differs for {entry['entry_id']}",
                category="ledger_integrity",
            )
    _validate_backend_profile_admissions(raw, ids)
    return raw


def _validate_backend_profile_admissions(
    ledger: Mapping[str, object], entry_ids: set[str]
) -> None:
    rows = ledger.get("backend_profile_admissions")
    if not isinstance(rows, list):
        raise ExecutionProfileError(
            "Selector ledger profile admissions are malformed",
            category="ledger_integrity",
        )
    expected = {
        "admission_id", "backend_status", "override_method_ids",
        "exact_method_ids", "method_selection_count", "fraction_layout",
        "evidence", "restrictions",
    }
    admission_ids: set[str] = set()
    signatures: set[tuple[str, tuple[str, ...], int, tuple[int, ...]]] = set()
    entries = _entry_map(ledger)
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != expected:
            raise ExecutionProfileError(
                "Selector ledger profile-admission inventory differs",
                category="ledger_integrity",
            )
        admission_id = row["admission_id"]
        backend = row["backend_status"]
        overrides = row["override_method_ids"]
        exact = row["exact_method_ids"]
        count = row["method_selection_count"]
        layout = row["fraction_layout"]
        evidence = row["evidence"]
        restrictions = row["restrictions"]
        if (not isinstance(admission_id, str) or not admission_id or
                admission_id != admission_id.strip() or
                admission_id in admission_ids):
            raise ExecutionProfileError(
                "Selector ledger profile-admission ID differs",
                category="ledger_integrity",
            )
        admission_ids.add(admission_id)
        if (not isinstance(backend, Mapping) or set(backend) != _BACKENDS or
                any(value not in {"public", "public_limited", "prototype",
                                  "missing", "not_applicable"}
                    for value in backend.values())):
            raise ExecutionProfileError(
                f"Malformed backend statuses for {admission_id}",
                category="ledger_integrity",
            )
        if (not isinstance(exact, list) or not exact or
                not all(isinstance(value, str) for value in exact) or
                exact != sorted(set(exact)) or not set(exact) <= entry_ids or
                any(entries[value].get("kind") != "method" for value in exact)):
            raise ExecutionProfileError(
                f"Malformed exact method inventory for {admission_id}",
                category="ledger_integrity",
            )
        if (not isinstance(overrides, list) or not overrides or
                not all(isinstance(value, str) for value in overrides) or
                overrides != sorted(set(overrides)) or
                not set(overrides) <= set(exact)):
            raise ExecutionProfileError(
                f"Malformed override inventory for {admission_id}",
                category="ledger_integrity",
            )
        if type(count) is not int or count < len(exact):
            raise ExecutionProfileError(
                f"Malformed selection count for {admission_id}",
                category="ledger_integrity",
            )
        if (not isinstance(layout, Mapping) or
                set(layout) != {"count", "transport_classes"} or
                type(layout["count"]) is not int or layout["count"] < 1 or
                not isinstance(layout["transport_classes"], list) or
                len(layout["transport_classes"]) != layout["count"] or
                any(type(value) is not int or value not in (1, 2, 3)
                    for value in layout["transport_classes"])):
            raise ExecutionProfileError(
                f"Malformed fraction layout for {admission_id}",
                category="ledger_integrity",
            )
        if (not isinstance(evidence, list) or not evidence or
                any(not isinstance(item, Mapping) or
                    set(item) != {"path", "sha256", "scope"} or
                    not isinstance(item["path"], str) or not item["path"] or
                    Path(item["path"]).is_absolute() or
                    ".." in Path(item["path"]).parts or
                    not isinstance(item["sha256"], str) or
                    _SHA256.fullmatch(item["sha256"]) is None or
                    not isinstance(item["scope"], str) or not item["scope"]
                    for item in evidence)):
            raise ExecutionProfileError(
                f"Malformed evidence inventory for {admission_id}",
                category="ledger_integrity",
            )
        if (not isinstance(restrictions, list) or not restrictions or
                not all(isinstance(value, str) and value for value in restrictions)):
            raise ExecutionProfileError(
                f"Malformed restrictions for {admission_id}",
                category="ledger_integrity",
            )
        for backend_id, route_status in backend.items():
            if route_status not in _ADMITTED:
                continue
            nonadmitted = {
                value for value in exact
                if entries[value].get("backend_status", {}).get(backend_id)
                not in _ADMITTED
            }
            if nonadmitted != set(overrides):
                raise ExecutionProfileError(
                    f"Profile overrides differ for {admission_id}/{backend_id}",
                    category="ledger_integrity",
                )
            signature = (
                str(backend_id), tuple(exact), int(layout["count"]),
                tuple(layout["transport_classes"]),
            )
            if signature in signatures:
                raise ExecutionProfileError(
                    "Selector ledger has ambiguous profile admissions",
                    category="ledger_integrity",
                )
            signatures.add(signature)


def selector_ledger_semantic_id(path: str | Path = SELECTOR_LEDGER_FILE) -> str:
    return str(load_selector_ledger(path)["semantic_id"])


def selector_ledger_admission_registry_id(
    path: str | Path = SELECTOR_LEDGER_FILE,
) -> str | None:
    value = load_selector_ledger(path).get("admission_registry_id")
    return str(value) if value is not None else None


def _entry_map(ledger: Mapping[str, object]) -> dict[str, dict]:
    return {str(row["entry_id"]): row for row in ledger["entries"]}  # type: ignore[index]


def _scope_tuple(value: Mapping[str, object] | Sequence[tuple[str, str]] | None) -> tuple[tuple[str, str], ...]:
    if value is None:
        return ()
    try:
        items = tuple(value.items()) if isinstance(value, Mapping) else tuple(value)
    except (TypeError, ValueError) as error:
        raise ExecutionProfileError(
            "Binding scope must be a string mapping",
            category="invalid_configuration",
        ) from error
    result: list[tuple[str, str]] = []
    for pair in items:
        if (not isinstance(pair, (tuple, list)) or len(pair) != 2 or
                pair[0] not in _SCOPE_KEYS or not isinstance(pair[1], str) or
                not pair[1] or pair[1] != pair[1].strip()):
            raise ExecutionProfileError(
                "Binding scope contains an unknown or invalid identity",
                category="scope_mismatch",
            )
        result.append((pair[0], pair[1]))
    if len({key for key, _ in result}) != len(result):
        raise ExecutionProfileError(
            "Binding scope repeats a dimension", category="scope_mismatch"
        )
    return tuple(sorted(result))


def _validate_scope(entry: Mapping[str, object], scope: tuple[tuple[str, str], ...]) -> None:
    definition = entry.get("scope")
    if not isinstance(definition, Mapping):
        raise ExecutionProfileError(
            f"Ledger scope is malformed for {entry.get('entry_id')}",
            category="ledger_integrity",
        )
    ownership = definition.get("ownership")
    if not isinstance(ownership, list) or not ownership:
        raise ExecutionProfileError(
            f"Ledger ownership is malformed for {entry.get('entry_id')}",
            category="ledger_integrity",
        )
    supplied = {key for key, _ in scope}
    cardinality = definition.get("cardinality")
    if not isinstance(cardinality, str) or not cardinality:
        raise ExecutionProfileError(
            f"Ledger cardinality is malformed for {entry.get('entry_id')}",
            category="ledger_integrity",
        )
    # Ownership lists every dimension a selector/value may vary over.  The
    # cardinality says which named identities belong in one profile row.  A
    # cell/face field is bound as one hashed artifact, never as millions of
    # individually scoped rows.
    required: set[str] = set()
    if "boundary_and_fraction" in cardinality or "boundary_and_suspended_fraction" in cardinality:
        required.update(("boundary_id", "fraction_id"))
    else:
        if "per_boundary" in cardinality or "named_boundary" in cardinality:
            required.add("boundary_id")
        if "per_fraction" in cardinality or "suspended_fraction" in cardinality:
            required.add("fraction_id")
        if "per_layer" in cardinality:
            required.add("layer_id")
    if not required and "model" in ownership:
        required.add("model_id")
    if supplied != required:
        raise ExecutionProfileError(
            f"Scoped binding for {entry.get('entry_id')} does not match ledger ownership",
            category="scope_mismatch",
        )


def _finite_json(value: object) -> object:
    # Round-trip through the canonical encoder to reject NaN, non-JSON values
    # and mutable custom objects.  Booleans remain distinct from integers.
    return json.loads(_canonical_json(value))


def _artifact(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ExecutionProfileError(
            "Artifact binding value must be an object", category="invalid_configuration"
        )
    required = {"path", "sha256", "shape", "dtype", "bytes"}
    if set(value) != required:
        raise ExecutionProfileError(
            "Artifact binding inventory differs", category="invalid_configuration"
        )
    shape = value["shape"]
    if (not isinstance(value["path"], str) or not value["path"] or
            Path(value["path"]).is_absolute() or ".." in Path(value["path"]).parts or
            not isinstance(value["sha256"], str) or
            _SHA256.fullmatch(value["sha256"]) is None or
            not isinstance(shape, (list, tuple)) or
            any(type(item) is not int or item < 0 for item in shape) or
            not isinstance(value["dtype"], str) or not value["dtype"] or
            type(value["bytes"]) is not int or value["bytes"] < 0):
        raise ExecutionProfileError(
            "Artifact binding metadata differs", category="invalid_configuration"
        )
    return {
        "path": value["path"], "sha256": value["sha256"],
        "shape": list(shape), "dtype": value["dtype"], "bytes": value["bytes"],
    }


def _normalize_fraction_layout(rows: Iterable[FractionLayout | Mapping[str, object]]) -> tuple[FractionLayout, ...]:
    result: list[FractionLayout] = []
    for raw in rows:
        if isinstance(raw, FractionLayout):
            row = raw
        elif isinstance(raw, Mapping) and set(raw) == {"fraction_id", "transport_class"}:
            row = FractionLayout(raw["fraction_id"], raw["transport_class"])  # type: ignore[arg-type]
        else:
            raise ExecutionProfileError(
                "Malformed fraction-layout row", category="invalid_configuration"
            )
        if (not isinstance(row.fraction_id, str) or not row.fraction_id or
                row.fraction_id != row.fraction_id.strip() or
                type(row.transport_class) is not int or
                row.transport_class not in (1, 2, 3)):
            raise ExecutionProfileError(
                "Fraction layout needs a unique ID and transport class 1, 2, or 3",
                category="invalid_configuration",
            )
        result.append(row)
    if not result or len({row.fraction_id for row in result}) != len(result):
        raise ExecutionProfileError(
            "Fraction layout must be nonempty with unique ordered IDs",
            category="invalid_configuration",
        )
    return tuple(result)


def _normalize_methods(
    rows: Iterable[MethodSelection | Mapping[str, object]],
    entries: Mapping[str, Mapping[str, object]],
) -> tuple[MethodSelection, ...]:
    result: list[MethodSelection] = []
    seen: set[tuple[str, tuple[tuple[str, str], ...]]] = set()
    for raw in rows:
        if isinstance(raw, MethodSelection):
            entry_id, scope = raw.entry_id, _scope_tuple(raw.scope)
        elif isinstance(raw, Mapping) and set(raw) == {"entry_id", "scope"}:
            entry_id, scope = raw["entry_id"], _scope_tuple(raw["scope"])  # type: ignore[arg-type]
        else:
            raise ExecutionProfileError(
                "Malformed method selection", category="invalid_configuration"
            )
        if not isinstance(entry_id, str) or entry_id not in entries:
            raise ExecutionProfileError(
                f"Unknown selector entry: {entry_id!r}", category="unknown_entry"
            )
        entry = entries[entry_id]
        if entry.get("kind") != "method" or entry.get("counts_as_method") is not True:
            raise ExecutionProfileError(
                f"{entry_id} is not a physical/constitutive method",
                category="kind_mismatch",
            )
        _validate_scope(entry, scope)
        key = (entry_id, scope)
        if key in seen:
            raise ExecutionProfileError(
                f"Duplicate method selection: {entry_id}",
                category="invalid_configuration",
            )
        seen.add(key)
        result.append(MethodSelection(entry_id, scope))
    if not result:
        raise ExecutionProfileError(
            "At least one method must be selected", category="invalid_configuration"
        )
    return tuple(result)


def _normalize_bindings(
    rows: Iterable[ValueBinding | Mapping[str, object]],
    entries: Mapping[str, Mapping[str, object]],
    kind: str,
) -> tuple[ValueBinding, ...]:
    result: list[ValueBinding] = []
    seen: set[tuple[str, tuple[tuple[str, str], ...]]] = set()
    for raw in rows:
        if isinstance(raw, ValueBinding):
            entry_id, representation = raw.entry_id, raw.representation
            scope, value = _scope_tuple(raw.scope), raw.value
        elif isinstance(raw, Mapping) and set(raw) == {
                "entry_id", "representation", "scope", "value"}:
            entry_id, representation = raw["entry_id"], raw["representation"]
            scope, value = _scope_tuple(raw["scope"]), raw["value"]  # type: ignore[arg-type]
        else:
            raise ExecutionProfileError(
                f"Malformed {kind} binding", category="invalid_configuration"
            )
        if not isinstance(entry_id, str) or entry_id not in entries:
            raise ExecutionProfileError(
                f"Unknown binding entry: {entry_id!r}", category="unknown_entry"
            )
        entry = entries[entry_id]
        if entry.get("kind") != kind or entry.get("counts_as_method") is not False:
            raise ExecutionProfileError(
                f"{entry_id} is not a {kind}", category="kind_mismatch"
            )
        if representation not in _REPRESENTATIONS:
            raise ExecutionProfileError(
                f"Unknown binding representation: {representation!r}",
                category="invalid_configuration",
            )
        _validate_scope(entry, scope)
        normalized = _artifact(value) if representation == "artifact" else _finite_json(value)
        key = (entry_id, scope)
        if key in seen:
            raise ExecutionProfileError(
                f"Duplicate scoped binding: {entry_id}",
                category="invalid_configuration",
            )
        seen.add(key)
        result.append(ValueBinding(entry_id, str(representation), scope, normalized))
    return tuple(result)


def bind_component_profile(
    component_id: str, document: Mapping[str, object]
) -> ComponentProfileBinding:
    if (not isinstance(component_id, str) or not component_id or
            component_id != component_id.strip() or not isinstance(document, Mapping)):
        raise ExecutionProfileError(
            "Component profile binding is malformed", category="invalid_configuration"
        )
    try:
        schema = document["schema"]
        scope_id = document["scope_id"]
        semantic_id = document["semantic_id"]
        fingerprint = document["fingerprint"]
    except KeyError as error:
        raise ExecutionProfileError(
            "Component profile lacks an identity field",
            category="invalid_configuration",
        ) from error
    if (type(schema) is not int or schema < 1 or not isinstance(scope_id, str) or
            not isinstance(semantic_id, str) or _SHA256.fullmatch(semantic_id) is None or
            not isinstance(fingerprint, str) or _SHA256.fullmatch(fingerprint) is None):
        raise ExecutionProfileError(
            "Component profile identity differs", category="invalid_configuration"
        )
    return ComponentProfileBinding(
        component_id, schema, scope_id, semantic_id, fingerprint, _sha(document)
    )


def _normalize_components(
    rows: Iterable[ComponentProfileBinding | Mapping[str, object]],
) -> tuple[ComponentProfileBinding, ...]:
    result: list[ComponentProfileBinding] = []
    for raw in rows:
        if isinstance(raw, ComponentProfileBinding):
            row = raw
        elif isinstance(raw, Mapping):
            expected = {
                "component_id", "schema", "scope_id", "semantic_id",
                "fingerprint", "document_sha256",
            }
            if set(raw) != expected:
                raise ExecutionProfileError(
                    "Component binding inventory differs",
                    category="invalid_configuration",
                )
            row = ComponentProfileBinding(**raw)  # type: ignore[arg-type]
        else:
            raise ExecutionProfileError(
                "Malformed component profile binding",
                category="invalid_configuration",
            )
        if (not isinstance(row.component_id, str) or not row.component_id or
                type(row.schema) is not int or row.schema < 1 or
                not isinstance(row.scope_id, str) or not row.scope_id or
                any(not isinstance(value, str) or _SHA256.fullmatch(value) is None
                    for value in (row.semantic_id, row.fingerprint,
                                  row.document_sha256))):
            raise ExecutionProfileError(
                "Component profile binding identity differs",
                category="invalid_configuration",
            )
        result.append(row)
    if len({row.component_id for row in result}) != len(result):
        raise ExecutionProfileError(
            "Component profile IDs must be unique", category="invalid_configuration"
        )
    return tuple(result)


def _scope_document(scope: tuple[tuple[str, str], ...]) -> dict[str, str]:
    return dict(scope)


def _selected_ids(methods: Sequence[MethodSelection]) -> set[str]:
    return {row.entry_id for row in methods}


def _enforce_backend_admission(
    ledger: Mapping[str, object],
    entries: Mapping[str, Mapping[str, object]],
    fractions: Sequence[FractionLayout],
    methods: Sequence[MethodSelection],
    backend: str | None,
) -> str | None:
    """Admit entry-public methods or one exact evidence-bounded profile.

    A profile rule may override only the explicitly listed non-admitted method
    rows.  It cannot make that method public in another physical combination.
    The returned ID is diagnostic; profile identity remains bound by the full
    ledger semantic ID and exact selected-method/fraction preimage.
    """
    if backend is None:
        return None
    nonadmitted = {
        row.entry_id for row in methods
        if not isinstance(entries[row.entry_id].get("backend_status"), Mapping)
        or entries[row.entry_id]["backend_status"].get(backend) not in _ADMITTED  # type: ignore[union-attr]
    }
    if not nonadmitted:
        return None
    selected = sorted(_selected_ids(methods))
    transport_classes = [row.transport_class for row in fractions]
    matches: list[str] = []
    rows = ledger.get("backend_profile_admissions")
    if not isinstance(rows, list):
        raise ExecutionProfileError(
            "Selector ledger profile admissions are malformed",
            category="ledger_integrity",
        )
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise ExecutionProfileError(
                "Selector ledger profile admission is malformed",
                category="ledger_integrity",
            )
        status = raw["backend_status"]
        layout = raw["fraction_layout"]
        if (not isinstance(status, Mapping) or status.get(backend) not in _ADMITTED or
                raw["exact_method_ids"] != selected or
                raw["method_selection_count"] != len(methods) or
                not isinstance(layout, Mapping) or
                layout["count"] != len(fractions) or
                layout["transport_classes"] != transport_classes or
                set(raw["override_method_ids"]) != nonadmitted):
            continue
        matches.append(str(raw["admission_id"]))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ExecutionProfileError(
            "Execution profile matches ambiguous backend admissions",
            category="ledger_integrity",
        )
    blocked = ", ".join(sorted(nonadmitted))
    raise ExecutionProfileError(
        f"{blocked} is not admitted on {backend} for this exact profile",
        category="implementation_restriction",
    )


def _groups(value: object) -> tuple[tuple[str, ...], ...]:
    if value in (None, []):
        return ()
    if not isinstance(value, list):
        raise ExecutionProfileError(
            "Compatibility group is malformed", category="ledger_integrity"
        )
    if value and all(isinstance(item, str) for item in value):
        return (tuple(value),)
    result = []
    for group in value:
        if not isinstance(group, list) or not group or not all(
                isinstance(item, str) for item in group):
            raise ExecutionProfileError(
                "Compatibility group is malformed", category="ledger_integrity"
            )
        result.append(tuple(group))
    return tuple(result)


def _enforce_rule(rule: Mapping[str, object], selected: set[str], *, label: str) -> None:
    when = rule.get("when_all", rule.get("if_all", []))
    if when is None:
        when = []
    if not isinstance(when, list) or not all(isinstance(item, str) for item in when):
        raise ExecutionProfileError(
            f"Malformed compatibility rule: {label}", category="ledger_integrity"
        )
    if when and not set(when).issubset(selected):
        return
    requires_all = rule.get("requires_all", [])
    if not isinstance(requires_all, list) or not all(
            isinstance(item, str) for item in requires_all):
        raise ExecutionProfileError(
            f"Malformed compatibility rule: {label}", category="ledger_integrity"
        )
    missing = set(requires_all) - selected
    if missing:
        raise ExecutionProfileError(
            f"{label} requires {', '.join(sorted(missing))}",
            category="physical_incompatibility",
        )
    for group in _groups(rule.get("requires_any", [])):
        if not selected.intersection(group):
            raise ExecutionProfileError(
                f"{label} requires one of {', '.join(group)}",
                category="physical_incompatibility",
            )
    excludes = rule.get("excludes", rule.get("forbids_any", []))
    if not isinstance(excludes, list) or not all(isinstance(item, str) for item in excludes):
        raise ExecutionProfileError(
            f"Malformed compatibility rule: {label}", category="ledger_integrity"
        )
    conflict = selected.intersection(excludes)
    if conflict:
        raise ExecutionProfileError(
            f"{label} excludes {', '.join(sorted(conflict))}",
            category="physical_incompatibility",
        )
    for group in _groups(rule.get("exactly_one_of", [])):
        found = selected.intersection(group)
        if len(found) != 1:
            raise ExecutionProfileError(
                f"{label} requires exactly one of {', '.join(group)}",
                category="physical_incompatibility",
            )


def _enforce_compatibility(
    ledger: Mapping[str, object], methods: Sequence[MethodSelection]
) -> None:
    selected = _selected_ids(methods)
    entries = _entry_map(ledger)
    for entry_id in sorted(selected):
        rule = entries[entry_id].get("compatibility", {})
        if not isinstance(rule, Mapping):
            raise ExecutionProfileError(
                f"Malformed compatibility for {entry_id}",
                category="ledger_integrity",
            )
        _enforce_rule(rule, selected, label=entry_id)
    rules = ledger.get("compatibility_rules")
    if not isinstance(rules, list):
        raise ExecutionProfileError(
            "Ledger compatibility rules are malformed", category="ledger_integrity"
        )
    for index, rule in enumerate(rules):
        if not isinstance(rule, Mapping):
            raise ExecutionProfileError(
                "Ledger compatibility rule is malformed", category="ledger_integrity"
            )
        label = str(rule.get("rule_id", f"ledger rule {index}"))
        _enforce_rule(rule, selected, label=label)


def _binding_dependency_active(
    entry: Mapping[str, object], selected: set[str], *, supplied: bool,
) -> bool:
    compatibility = entry.get("compatibility")
    if not isinstance(compatibility, Mapping):
        raise ExecutionProfileError(
            f"Malformed binding compatibility for {entry.get('entry_id')}",
            category="ledger_integrity",
        )
    requires_all = compatibility.get("requires_all", [])
    if not isinstance(requires_all, list) or not all(
            isinstance(item, str) for item in requires_all):
        raise ExecutionProfileError(
            "Malformed binding requires_all", category="ledger_integrity"
        )
    groups = _groups(compatibility.get("requires_any", []))
    excludes = compatibility.get("excludes", [])
    if not isinstance(excludes, list) or not all(isinstance(item, str) for item in excludes):
        raise ExecutionProfileError(
            "Malformed binding excludes", category="ledger_integrity"
        )
    active = (set(requires_all).issubset(selected) and
              all(selected.intersection(group) for group in groups) and
              not selected.intersection(excludes))
    # An unconditional parameter is a legal independent input.  Conditional
    # parameters are legal only while their declared method prerequisites are
    # active; this catches stale formula coefficients before a worker starts.
    has_dependencies = bool(requires_all or groups or excludes)
    if supplied and has_dependencies and not active:
        raise ExecutionProfileError(
            f"Binding {entry.get('entry_id')} is inactive for the selected methods",
            category="physical_incompatibility",
        )
    return active and bool(requires_all or groups)


def _expected_binding_scopes(
    entry: Mapping[str, object], methods: Sequence[MethodSelection],
    fractions: Sequence[FractionLayout],
) -> set[tuple[tuple[str, str], ...]]:
    definition = entry["scope"]
    cardinality = str(definition["cardinality"])
    compatibility = entry["compatibility"]
    required_methods = set(compatibility.get("requires_all", []))
    relevant = [row for row in methods if row.entry_id in required_methods]
    if "boundary_and_fraction" in cardinality or "boundary_and_suspended_fraction" in cardinality:
        scopes = {row.scope for row in relevant
                  if {key for key, _ in row.scope} == {"boundary_id", "fraction_id"}}
        return scopes
    if "per_boundary" in cardinality or "named_boundary" in cardinality:
        scopes = {tuple((key, value) for key, value in row.scope if key == "boundary_id")
                  for row in relevant if "boundary_id" in dict(row.scope)}
        return scopes
    if "per_fraction" in cardinality or "suspended_fraction" in cardinality:
        ids = {dict(row.scope)["fraction_id"] for row in relevant
               if "fraction_id" in dict(row.scope)}
        if not ids:
            ids = {row.fraction_id for row in fractions}
        return {(('fraction_id', fraction_id),) for fraction_id in ids}
    if "per_layer" in cardinality:
        return {tuple((key, value) for key, value in row.scope if key == "layer_id")
                for row in relevant if "layer_id" in dict(row.scope)}
    return {(('model_id', 'model'),)} if "model" in definition["ownership"] else set()


def _enforce_binding_completeness(
    ledger: Mapping[str, object], methods: Sequence[MethodSelection],
    fractions: Sequence[FractionLayout], parameters: Sequence[ValueBinding],
    controls: Sequence[ValueBinding],
) -> None:
    selected = _selected_ids(methods)
    supplied = {
        (row.entry_id, row.scope) for row in (*parameters, *controls)
    }
    entries = _entry_map(ledger)
    for row in (*parameters, *controls):
        _binding_dependency_active(entries[row.entry_id], selected, supplied=True)
    for entry in entries.values():
        kind = entry.get("kind")
        if kind not in ("parameter", "control"):
            continue
        dependency_required = _binding_dependency_active(
            entry, selected, supplied=False)
        globally_required = (
            kind == "control" and entry.get("requiredness") == "required"
        )
        if not (dependency_required or globally_required):
            continue
        expected = _expected_binding_scopes(entry, methods, fractions)
        if not expected:
            raise ExecutionProfileError(
                f"Cannot resolve required scope for {entry.get('entry_id')}",
                category="missing_scope",
            )
        missing = expected - {
            scope for entry_id, scope in supplied if entry_id == entry["entry_id"]
        }
        if missing:
            raise ExecutionProfileError(
                f"Missing required scoped binding for {entry.get('entry_id')}",
                category="missing_scope",
            )


def _payload(profile: Flow2DHExecutionProfile) -> dict[str, object]:
    return {
        "schema": profile.schema,
        "scope_id": profile.scope_id,
        "semantic_id": profile.semantic_id,
        "ledger_semantic_id": profile.ledger_semantic_id,
        "legacy_catalog_id": profile.legacy_catalog_id,
        "backend": profile.backend,
        "fraction_layout": [
            {"fraction_id": row.fraction_id, "transport_class": row.transport_class}
            for row in profile.fraction_layout
        ],
        "method_selections": [
            {"entry_id": row.entry_id, "scope": _scope_document(row.scope)}
            for row in profile.method_selections
        ],
        "parameter_bindings": [
            {"entry_id": row.entry_id, "representation": row.representation,
             "scope": _scope_document(row.scope), "value": row.value}
            for row in profile.parameter_bindings
        ],
        "control_bindings": [
            {"entry_id": row.entry_id, "representation": row.representation,
             "scope": _scope_document(row.scope), "value": row.value}
            for row in profile.control_bindings
        ],
        "component_profiles": [
            {
                "component_id": row.component_id, "schema": row.schema,
                "scope_id": row.scope_id, "semantic_id": row.semantic_id,
                "fingerprint": row.fingerprint,
                "document_sha256": row.document_sha256,
            }
            for row in profile.component_profiles
        ],
    }


def is_frozen_q3sp_execution_profile(
    value: Flow2DHExecutionProfile | Mapping[str, object],
) -> bool:
    """Recognize only the frozen seven-selector pre-Q3-C law-11 profile."""
    if isinstance(value, Flow2DHExecutionProfile):
        ledger_id = value.ledger_semantic_id
        backend = value.backend
        fractions: object = value.fraction_layout
        methods: object = value.method_selections
        components: object = value.component_profiles
    elif isinstance(value, Mapping):
        ledger_id = value.get("ledger_semantic_id")
        backend = value.get("backend")
        fractions = value.get("fraction_layout")
        methods = value.get("method_selections")
        components = value.get("component_profiles")
    else:
        return False
    if ledger_id != FROZEN_Q3SP_LEDGER_SEMANTIC_ID or backend not in (
            None, BACKEND_CPU, BACKEND_GPU):
        return False
    if not isinstance(fractions, (list, tuple)) or len(fractions) != 1:
        return False
    fraction = fractions[0]
    transport_class = (
        fraction.transport_class if isinstance(fraction, FractionLayout)
        else fraction.get("transport_class") if isinstance(fraction, Mapping)
        else None
    )
    if transport_class != 1:
        return False
    if not isinstance(methods, (list, tuple)) or len(methods) != 7:
        return False
    method_ids = {
        row.entry_id if isinstance(row, MethodSelection)
        else row.get("entry_id") if isinstance(row, Mapping)
        else None
        for row in methods
    }
    if method_ids != _FROZEN_Q3SP_METHOD_IDS:
        return False
    if not isinstance(components, (list, tuple)) or len(components) != 2:
        return False
    component_ids = {
        row.component_id if isinstance(row, ComponentProfileBinding)
        else row.get("component_id") if isinstance(row, Mapping)
        else None
        for row in components
    }
    return component_ids == {"capacity", "channel_composition"}


def _load_frozen_q3sp_execution_profile(
    raw: Mapping[str, object], *, ledger_path: str | Path,
) -> Flow2DHExecutionProfile:
    if not is_frozen_q3sp_execution_profile(raw):
        raise ExecutionProfileError(
            "Frozen Q3-S/P execution profile identity signature differs",
            category="identity_mismatch",
        )
    supplied_fingerprint = raw.get("fingerprint")
    payload = dict(raw)
    payload.pop("fingerprint", None)
    if (not isinstance(supplied_fingerprint, str) or
            _SHA256.fullmatch(supplied_fingerprint) is None or
            _sha(payload) != supplied_fingerprint):
        raise ExecutionProfileError(
            "Frozen Q3-S/P execution-profile fingerprint differs",
            category="identity_mismatch",
        )
    try:
        normalized = resolve_execution_profile(
            fraction_layout=raw["fraction_layout"],  # type: ignore[arg-type]
            method_selections=raw["method_selections"],  # type: ignore[arg-type]
            parameter_bindings=raw["parameter_bindings"],  # type: ignore[arg-type]
            control_bindings=raw["control_bindings"],  # type: ignore[arg-type]
            component_profiles=raw["component_profiles"],  # type: ignore[arg-type]
            backend=raw["backend"],  # type: ignore[arg-type]
            ledger_path=ledger_path,
        )
    except ExecutionProfileError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise ExecutionProfileError(
            "Frozen Q3-S/P execution profile is malformed",
            category="invalid_serialization",
        ) from error
    expected = execution_profile_document(normalized)
    for key, value in expected.items():
        if key not in {"ledger_semantic_id", "fingerprint"} and raw.get(key) != value:
            raise ExecutionProfileError(
                "Frozen Q3-S/P execution profile differs from its physical contract",
                category="identity_mismatch",
            )
    return Flow2DHExecutionProfile(
        schema=normalized.schema,
        scope_id=normalized.scope_id,
        semantic_id=normalized.semantic_id,
        ledger_semantic_id=FROZEN_Q3SP_LEDGER_SEMANTIC_ID,
        legacy_catalog_id=normalized.legacy_catalog_id,
        backend=normalized.backend,
        fraction_layout=normalized.fraction_layout,
        method_selections=normalized.method_selections,
        parameter_bindings=normalized.parameter_bindings,
        control_bindings=normalized.control_bindings,
        component_profiles=normalized.component_profiles,
        fingerprint=supplied_fingerprint,
    )


def execution_profile_payload_json(profile: Flow2DHExecutionProfile) -> str:
    """Exact no-newline ASCII preimage consumed by native schema v52."""
    checked = execution_profile_document(profile)
    checked.pop("fingerprint")
    return _canonical_json(checked)


def execution_profile_document(profile: Flow2DHExecutionProfile) -> dict[str, object]:
    if not isinstance(profile, Flow2DHExecutionProfile):
        raise ExecutionProfileError(
            "Expected a Flow2DHExecutionProfile", category="invalid_configuration"
        )
    if (profile.schema != EXECUTION_PROFILE_SCHEMA or
            profile.scope_id != EXECUTION_PROFILE_SCOPE_ID or
            profile.semantic_id != execution_profile_semantic_id() or
            profile.legacy_catalog_id != semantic_catalog_id()):
        raise ExecutionProfileError(
            "Execution-profile schema or semantic identity differs",
            category="identity_mismatch",
        )
    payload = _payload(profile)
    expected = _sha(payload)
    if profile.fingerprint != expected:
        raise ExecutionProfileError(
            "Execution-profile fingerprint differs", category="identity_mismatch"
        )
    document = dict(payload)
    document["fingerprint"] = expected
    return json.loads(_canonical_json(document))


def resolve_execution_profile(
    *,
    fraction_layout: Iterable[FractionLayout | Mapping[str, object]],
    method_selections: Iterable[MethodSelection | Mapping[str, object]],
    parameter_bindings: Iterable[ValueBinding | Mapping[str, object]] = (),
    control_bindings: Iterable[ValueBinding | Mapping[str, object]] = (),
    component_profiles: Iterable[ComponentProfileBinding | Mapping[str, object]] = (),
    backend: str | None = None,
    ledger_path: str | Path = SELECTOR_LEDGER_FILE,
) -> Flow2DHExecutionProfile:
    if backend is not None and backend not in _BACKENDS:
        raise ExecutionProfileError(
            f"Unknown backend: {backend!r}", category="invalid_configuration"
        )
    ledger = load_selector_ledger(ledger_path)
    entries = _entry_map(ledger)
    fractions = _normalize_fraction_layout(fraction_layout)
    methods = _normalize_methods(method_selections, entries)
    _enforce_backend_admission(ledger, entries, fractions, methods, backend)
    parameters = _normalize_bindings(parameter_bindings, entries, "parameter")
    controls = _normalize_bindings(control_bindings, entries, "control")
    components = _normalize_components(component_profiles)
    _enforce_compatibility(ledger, methods)
    _enforce_binding_completeness(
        ledger, methods, fractions, parameters, controls)

    # Scoped fraction references must resolve to the exact ordered layout.
    fraction_ids = {row.fraction_id for row in fractions}
    for row in (*methods, *parameters, *controls):
        scoped = dict(row.scope).get("fraction_id")
        if scoped is not None and scoped not in fraction_ids:
            raise ExecutionProfileError(
                f"Binding refers to unknown fraction {scoped!r}",
                category="scope_mismatch",
            )

    provisional = Flow2DHExecutionProfile(
        EXECUTION_PROFILE_SCHEMA,
        EXECUTION_PROFILE_SCOPE_ID,
        execution_profile_semantic_id(),
        str(ledger["semantic_id"]),
        semantic_catalog_id(),
        backend,
        fractions,
        methods,
        parameters,
        controls,
        components,
        "",
    )
    return Flow2DHExecutionProfile(
        **{**provisional.__dict__, "fingerprint": _sha(_payload(provisional))}
    )


def load_execution_profile(
    document: str | bytes | Mapping[str, object],
    *,
    ledger_path: str | Path = SELECTOR_LEDGER_FILE,
) -> Flow2DHExecutionProfile:
    raw = _read_json(document, what="execution profile")
    expected_keys = {
        "schema", "scope_id", "semantic_id", "ledger_semantic_id",
        "legacy_catalog_id", "backend", "fraction_layout", "method_selections",
        "parameter_bindings", "control_bindings", "component_profiles", "fingerprint",
    }
    if set(raw) != expected_keys:
        raise ExecutionProfileError(
            "Execution-profile inventory differs", category="invalid_serialization"
        )
    if raw.get("ledger_semantic_id") == FROZEN_Q3SP_LEDGER_SEMANTIC_ID:
        return _load_frozen_q3sp_execution_profile(
            raw, ledger_path=ledger_path
        )
    try:
        rebuilt = resolve_execution_profile(
            fraction_layout=raw["fraction_layout"],  # type: ignore[arg-type]
            method_selections=raw["method_selections"],  # type: ignore[arg-type]
            parameter_bindings=raw["parameter_bindings"],  # type: ignore[arg-type]
            control_bindings=raw["control_bindings"],  # type: ignore[arg-type]
            component_profiles=raw["component_profiles"],  # type: ignore[arg-type]
            backend=raw["backend"],  # type: ignore[arg-type]
            ledger_path=ledger_path,
        )
    except ExecutionProfileError:
        raise
    except (TypeError, ValueError) as error:
        raise ExecutionProfileError(
            "Malformed execution-profile document",
            category="invalid_serialization",
        ) from error
    expected = execution_profile_document(rebuilt)
    if _canonical_json(raw) != _canonical_json(expected):
        raise ExecutionProfileError(
            "Execution profile is stale, forged, or noncanonical",
            category="identity_mismatch",
        )
    return rebuilt


__all__ = [
    "EXECUTION_PROFILE_SCHEMA", "EXECUTION_PROFILE_SCOPE_ID",
    "SELECTOR_LEDGER_SCHEMA", "SELECTOR_LEDGER_SCOPE_ID", "SELECTOR_LEDGER_FILE",
    "ADMISSION_REGISTRY_SCHEMA", "FROZEN_Q3SP_LEDGER_SEMANTIC_ID",
    "BACKEND_CPU", "BACKEND_GPU", "ExecutionProfileError", "FractionLayout",
    "MethodSelection", "ValueBinding", "ComponentProfileBinding",
    "Flow2DHExecutionProfile", "execution_profile_semantic_document",
    "execution_profile_semantic_id", "load_selector_ledger",
    "selector_catalogue_semantic_payload", "admission_registry_payload",
    "selector_ledger_semantic_id", "selector_ledger_admission_registry_id",
    "is_frozen_q3sp_execution_profile", "bind_component_profile",
    "resolve_execution_profile", "execution_profile_document",
    "execution_profile_payload_json", "load_execution_profile",
]
