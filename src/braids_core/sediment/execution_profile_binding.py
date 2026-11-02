"""Project/preparation binding for the additive schema-v2 execution profile."""
from __future__ import annotations

import hashlib
import json
from typing import Mapping

from braids_core.sediment_method_catalog import semantic_catalog_id

from .execution_profile_v2 import (
    BACKEND_CPU,
    BACKEND_GPU,
    ComponentProfileBinding,
    ExecutionProfileError,
    Flow2DHExecutionProfile,
    bind_component_profile,
    execution_profile_document,
    is_frozen_q3sp_execution_profile,
    load_execution_profile,
    resolve_execution_profile,
)


EXECUTION_PROFILE_BINDING_SCHEMA = 2
EXECUTION_PROFILE_BINDING_KEY = "sediment_execution_profile_v2"


def _canonical(value: object) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ExecutionProfileError(
            "Execution-profile binding is not canonical JSON",
            category="invalid_serialization",
        ) from error


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("ascii")).hexdigest()


def _rebind_frozen_q3sp_profile(
    profile: Flow2DHExecutionProfile, *, backend: str | None,
    components: tuple[ComponentProfileBinding, ...],
) -> Flow2DHExecutionProfile:
    """Rebind backend/component documents without relabelling frozen Q3-S/P."""
    by_id = {row.component_id: row for row in components}
    if set(by_id) != {row.component_id for row in profile.component_profiles}:
        raise ExecutionProfileError(
            "Frozen Q3-S/P component inventory differs",
            category="identity_mismatch",
        )
    raw = execution_profile_document(profile)
    raw["backend"] = backend
    raw["component_profiles"] = [
        {
            "component_id": by_id[row.component_id].component_id,
            "schema": by_id[row.component_id].schema,
            "scope_id": by_id[row.component_id].scope_id,
            "semantic_id": by_id[row.component_id].semantic_id,
            "fingerprint": by_id[row.component_id].fingerprint,
            "document_sha256": by_id[row.component_id].document_sha256,
        }
        for row in profile.component_profiles
    ]
    raw.pop("fingerprint", None)
    raw["fingerprint"] = _sha(raw)
    return load_execution_profile(raw)


def legacy_execution_profile_binding(reason_code: str = "schema_v1_project") -> dict:
    if not isinstance(reason_code, str) or not reason_code:
        raise ExecutionProfileError(
            "Legacy profile reason must be text", category="invalid_configuration"
        )
    return {
        "schema": EXECUTION_PROFILE_BINDING_SCHEMA,
        "state": "legacy_schema_v1",
        "legacy_catalog_id": semantic_catalog_id(),
        "profile": None,
        "component_documents": {},
        "reason_code": reason_code,
    }


def canonical_execution_profile_binding(
    profile: Flow2DHExecutionProfile,
    component_documents: Mapping[str, Mapping[str, object]],
) -> dict:
    """Bind exact component documents to a backend-neutral project profile."""
    document = execution_profile_document(profile)
    if profile.backend is not None:
        raise ExecutionProfileError(
            "Project execution profile must be backend neutral",
            category="invalid_configuration",
        )
    if not isinstance(component_documents, Mapping):
        raise ExecutionProfileError(
            "Component documents must be a mapping",
            category="invalid_configuration",
        )
    expected = {row.component_id: row for row in profile.component_profiles}
    if set(component_documents) != set(expected):
        raise ExecutionProfileError(
            "Component document inventory differs from execution profile",
            category="identity_mismatch",
        )
    canonical_documents: dict[str, dict] = {}
    for component_id in sorted(component_documents):
        raw = component_documents[component_id]
        if not isinstance(raw, Mapping):
            raise ExecutionProfileError(
                f"Component document {component_id} is not an object",
                category="invalid_configuration",
            )
        normalized = json.loads(_canonical(raw))
        row = expected[component_id]
        if (_sha(normalized) != row.document_sha256 or
                normalized.get("schema") != row.schema or
                normalized.get("scope_id") != row.scope_id or
                normalized.get("semantic_id") != row.semantic_id or
                normalized.get("fingerprint") != row.fingerprint):
            raise ExecutionProfileError(
                f"Component document identity differs: {component_id}",
                category="identity_mismatch",
            )
        canonical_documents[component_id] = normalized
    return {
        "schema": EXECUTION_PROFILE_BINDING_SCHEMA,
        "state": "canonical",
        "legacy_catalog_id": semantic_catalog_id(),
        "profile": document,
        "component_documents": canonical_documents,
        "reason_code": None,
    }


def load_execution_profile_binding(
    value: Mapping[str, object], *, ledger_path=None,
) -> tuple[dict, Flow2DHExecutionProfile | None]:
    """Strictly validate a project binding without selecting a backend."""
    if not isinstance(value, Mapping):
        raise ExecutionProfileError(
            "Execution-profile binding must be an object",
            category="invalid_serialization",
        )
    required = {
        "schema", "state", "legacy_catalog_id", "profile",
        "component_documents", "reason_code",
    }
    if set(value) != required or value.get("schema") != EXECUTION_PROFILE_BINDING_SCHEMA:
        raise ExecutionProfileError(
            "Execution-profile binding inventory or schema differs",
            category="invalid_serialization",
        )
    if value.get("legacy_catalog_id") != semantic_catalog_id():
        raise ExecutionProfileError(
            "Execution-profile binding uses a different legacy catalogue",
            category="identity_mismatch",
        )
    state = value.get("state")
    if state == "legacy_schema_v1":
        expected = legacy_execution_profile_binding(value.get("reason_code"))  # type: ignore[arg-type]
        if _canonical(value) != _canonical(expected):
            raise ExecutionProfileError(
                "Legacy execution-profile binding differs",
                category="identity_mismatch",
            )
        return expected, None
    if state != "canonical" or value.get("reason_code") is not None:
        raise ExecutionProfileError(
            "Unknown execution-profile binding state",
            category="invalid_serialization",
        )
    kwargs = {} if ledger_path is None else {"ledger_path": ledger_path}
    profile = load_execution_profile(value.get("profile"), **kwargs)
    expected = canonical_execution_profile_binding(
        profile, value.get("component_documents")  # type: ignore[arg-type]
    )
    if _canonical(value) != _canonical(expected):
        raise ExecutionProfileError(
            "Execution-profile binding is stale or forged",
            category="identity_mismatch",
        )
    return expected, profile


def runtime_execution_profile(
    binding: Mapping[str, object], *, backend: str, ledger_path=None,
    component_documents: Mapping[str, Mapping[str, object]] | None = None,
) -> tuple[Flow2DHExecutionProfile, Mapping[str, Mapping[str, object]]]:
    """Re-resolve a canonical project profile for a selected backend."""
    if backend not in (BACKEND_CPU, BACKEND_GPU):
        raise ExecutionProfileError(
            "Runtime backend must be explicit", category="invalid_configuration"
        )
    kwargs = {} if ledger_path is None else {"ledger_path": ledger_path}
    checked, profile = load_execution_profile_binding(binding, **kwargs)
    if profile is None:
        raise ExecutionProfileError(
            "Schema-v1 project has no complete schema-v2 execution profile; "
            "edit and save explicit sediment methods before launch",
            category="migration_required",
        )
    documents = (checked["component_documents"] if component_documents is None
                 else component_documents)
    if not isinstance(documents, Mapping) or set(documents) != {
            row.component_id for row in profile.component_profiles}:
        raise ExecutionProfileError(
            "Runtime component document inventory differs",
            category="identity_mismatch",
        )
    components = tuple(
        bind_component_profile(component_id, documents[component_id])
        for component_id in sorted(documents)
    )
    if is_frozen_q3sp_execution_profile(profile):
        runtime_documents = {
            component_id: json.loads(_canonical(documents[component_id]))
            for component_id in sorted(documents)
        }
        neutral = _rebind_frozen_q3sp_profile(
            profile, backend=None, components=components
        )
        canonical_execution_profile_binding(neutral, runtime_documents)
        return (
            _rebind_frozen_q3sp_profile(
                neutral, backend=backend, components=components
            ),
            runtime_documents,
        )
    resolved = resolve_execution_profile(
        fraction_layout=profile.fraction_layout,
        method_selections=profile.method_selections,
        parameter_bindings=profile.parameter_bindings,
        control_bindings=profile.control_bindings,
        component_profiles=components,
        backend=backend,
        **kwargs,
    )
    # Revalidate the exact runtime component documents against the freshly
    # built bindings before returning them to a deck writer.
    runtime_documents = {
        component_id: json.loads(_canonical(documents[component_id]))
        for component_id in sorted(documents)
    }
    canonical_execution_profile_binding(
        resolve_execution_profile(
            fraction_layout=profile.fraction_layout,
            method_selections=profile.method_selections,
            parameter_bindings=profile.parameter_bindings,
            control_bindings=profile.control_bindings,
            component_profiles=components,
            backend=None,
            **kwargs,
        ),
        runtime_documents,
    )
    return resolved, runtime_documents


__all__ = [
    "EXECUTION_PROFILE_BINDING_SCHEMA", "EXECUTION_PROFILE_BINDING_KEY",
    "legacy_execution_profile_binding", "canonical_execution_profile_binding",
    "load_execution_profile_binding", "runtime_execution_profile",
]
