"""Authenticated fixed-grid CPU envelope for the Q3-H4 face-slope operator.

This module is deliberately additive.  Version 62 wraps an already prepared,
authenticated Q3-C scalar-capacity deck whose embedded method profile remains
``bed_slope.off``.  The public envelope owns the one subsequent FLOW bed-slope
composition and binds its compiled adapter, deck bytes and complete parameters.

The first CPU slice is intentionally narrow: fixed grid, fully wet internal
faces, capacity laws 2/8/11, exchange/adaptation/secondary/hiding off, and no
non-zero prescribed sediment feed.  Unsupported physics is rejected; there is
no legacy selector or CPU/GPU fallback hidden in this contract.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence

from .bed_slope_execution import (
    COMPOSITION_ORDER,
    FACE_ROLE_POLICY,
    FIELD_OWNERSHIP,
    IMPLEMENTATION_PATHS,
    LOCKED_FLOW_RELATIVE_PATH,
    LOCKED_FLOW_SHA256,
    METHODS,
    SCHEMA as COMPONENT_SCHEMA,
    SCOPE_ID as COMPONENT_SCOPE_ID,
    SEMANTIC_ID as COMPONENT_SEMANTIC_ID,
    UPSTREAM_STAGES,
)
from .capacity_foundation_execution import load_capacity_foundation_execution


SCHEMA = 1
ENVELOPE_VERSION = 62
SCOPE_ID = "braids-q3h4b-fixed-grid-cpu-bed-slope-live-v62"
SEMANTIC_ID = hashlib.sha256(SCOPE_ID.encode("ascii")).hexdigest()
PROJECT_BINDING_KEY = "bed_slope_cpu_v62_project_binding"
PREPARED_EXECUTION_KEY = "bed_slope_cpu_v62_prepared_execution"
RESULT_IDENTITY_KEY = "bed_slope_cpu_v62_result_identity"
RESTART_IDENTITY_KEY = "bed_slope_cpu_v62_restart_identity"

SUPPORTED_LAWS = frozenset((2, 8, 11))
_LAW_METHOD = {
    2: "capacity.mpm",
    8: "capacity.engelund_hansen",
    11: "capacity.general",
}
_SHA = re.compile(r"[0-9a-f]{64}\Z")

DEFAULT_CONTROLS = {
    "alfabs": 1.0,
    "alfabn": 1.5,
    "ashld": 0.85,
    "bshld": 0.5,
    "cshld": 0.0,
    "dshld": 0.0,
    "alfpa": 0.4,
    "thcrpa": 0.047,
}


def canonical_bytes(value: object) -> bytes:
    """Return the only JSON representation admitted by the v62 identity."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def fingerprint(value: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise ValueError(f"Q3-H4b {name} must be a lowercase SHA-256")
    return value


def _finite(value: object, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Q3-H4b {name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise ValueError(f"Q3-H4b {name} is outside its finite domain")
    return 0.0 if result == 0.0 else result


def _fraction_ids(value: object) -> list[str]:
    if (not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or
            not value or any(not isinstance(item, str) or not item
                             for item in value)):
        raise ValueError("Q3-H4b requires ordered non-empty fraction IDs")
    result = list(value)
    if len(set(result)) != len(result):
        raise ValueError("Q3-H4b fraction IDs must be unique")
    return result


def _controls(values: Mapping[str, object]) -> dict[str, float]:
    if not isinstance(values, Mapping) or set(values) != set(DEFAULT_CONTROLS):
        raise ValueError("Q3-H4b requires all eight FLOW slope controls")
    return {
        key: _finite(values[key], key, nonnegative=(key == "thcrpa"))
        for key in DEFAULT_CONTROLS
    }


def project_binding_document(
    *, method_id: str, formula_code: int, fraction_ids: Sequence[str],
    controls: Mapping[str, object],
) -> dict[str, object]:
    """Bind an explicit user choice before mesh/capacity preparation."""
    if method_id not in METHODS:
        raise ValueError("Q3-H4b requires a classic FLOW slope method")
    if (isinstance(formula_code, bool) or formula_code not in SUPPORTED_LAWS):
        raise ValueError("Q3-H4b supports capacity laws 2, 8 and 11 only")
    method = dict(METHODS[method_id])
    body: dict[str, object] = {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "envelope_version": ENVELOPE_VERSION,
        "stage": "project",
        "state": "deferred_authenticated_q3c_preparation",
        "backend": "cpu",
        "backend_selection": "explicit_user_choice",
        "automatic_backend_fallback": False,
        "method_id": method_id,
        **method,
        "formula_code": int(formula_code),
        "capacity_method_id": _LAW_METHOD[int(formula_code)],
        "ordered_fraction_ids": _fraction_ids(fraction_ids),
        "parameters": _controls(controls),
        "required_base_profile": "authenticated_q3c_bed_slope_off",
        "mesh_scope": "fixed_grid",
        "internal_face_scope": "fully_wet",
        "exchange_mode": 0,
        "bedload_adaptation": False,
        "secondary_steering": "off",
        "generic_hiding": "off",
        "prescribed_sediment_feed": "zero_only",
        "application_count": 1,
        "invalid_policy": "reject_before_launch_or_exner_no_fallback",
    }
    return {**body, "fingerprint": fingerprint(body)}


def validate_project_binding(value: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("Q3-H4b project binding must be an object")
    try:
        rebuilt = project_binding_document(
            method_id=value["method_id"], formula_code=value["formula_code"],
            fraction_ids=value["ordered_fraction_ids"],
            controls=value["parameters"],
        )
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError("Invalid Q3-H4b project binding") from error
    if dict(value) != rebuilt or canonical_bytes(value) != canonical_bytes(rebuilt):
        raise ValueError("Q3-H4b project binding is stale or forged")
    return rebuilt


def stage_fingerprint(stage: str, payload: Mapping[str, object]) -> str:
    """Typed fingerprint for an upstream operator with no separate document."""
    if stage not in UPSTREAM_STAGES:
        raise ValueError("Unknown Q3-H4b upstream stage")
    return fingerprint({"stage": stage, **dict(payload)})


def component_document(
    *, project_binding: Mapping[str, object],
    capacity_execution: Mapping[str, object],
    channel_composition_fingerprint: str,
    face_reconstruction_fingerprint: str,
    implementation_source_hashes: Mapping[str, str],
) -> dict[str, object]:
    """Materialise the backend-neutral H4 component after Q3-C preparation."""
    binding = validate_project_binding(project_binding)
    capacity = load_capacity_foundation_execution(capacity_execution)
    if (capacity["formula_code"] != binding["formula_code"] or
            capacity["method_id"] != binding["capacity_method_id"]):
        raise ValueError("Q3-H4b capacity and project method identities differ")
    capacity_fp = str(capacity["fingerprint"])
    upstream = {
        "capacity_execution": capacity_fp,
        "secondary_steering_execution": stage_fingerprint(
            "secondary_steering_execution", {
                "method_id": "secondary.off",
                "capacity_execution_fingerprint": capacity_fp,
            }),
        "channel_composition_execution": _sha(
            channel_composition_fingerprint,
            "channel-composition fingerprint"),
        "face_reconstruction_execution": _sha(
            face_reconstruction_fingerprint,
            "face-reconstruction fingerprint"),
        "hiding_execution": stage_fingerprint("hiding_execution", {
            "method_id": "hiding.none",
            "capacity_execution_fingerprint": capacity_fp,
        }),
    }
    required_sources = {LOCKED_FLOW_RELATIVE_PATH, *IMPLEMENTATION_PATHS}
    if (not isinstance(implementation_source_hashes, Mapping) or
            set(implementation_source_hashes) != required_sources):
        raise ValueError("Q3-H4b adapter receipt has an incomplete source inventory")
    sources = {
        name: _sha(implementation_source_hashes[name], f"{name} source hash")
        for name in sorted(required_sources)
    }
    if sources[LOCKED_FLOW_RELATIVE_PATH] != LOCKED_FLOW_SHA256:
        raise ValueError("Q3-H4b adapter does not bind the locked FLOW source")
    method_id = str(binding["method_id"])
    method = dict(METHODS[method_id])
    body: dict[str, object] = {
        "schema": COMPONENT_SCHEMA,
        "scope_id": COMPONENT_SCOPE_ID,
        "semantic_id": COMPONENT_SEMANTIC_ID,
        "envelope_version": 61,
        "component_scope": "backend_neutral_face_and_open_boundary_operator",
        "method_id": method_id,
        **method,
        "ordered_fraction_ids": list(binding["ordered_fraction_ids"]),
        "parameters": dict(binding["parameters"]),
        "upstream_identities": [
            {"stage": stage, "fingerprint": upstream[stage]}
            for stage in UPSTREAM_STAGES
        ],
        "composition_order": list(COMPOSITION_ORDER),
        "face_role_policy": dict(FACE_ROLE_POLICY),
        "field_ownership": dict(FIELD_OWNERSHIP),
        "transport_units": "solid_mass_per_time_per_width",
        "downhill_sign": "positive_downhill",
        "parker_hiding_average": "arithmetic",
        "invalid_policy": "reject_before_exner_no_partial_commit",
        "backend_selection": "explicit_user_choice",
        "automatic_cpu_fallback": False,
        "implementation_source_hashes": sources,
        "runtime_admission_authority": "separate_exact_live_route_gate",
    }
    return {**body, "fingerprint": fingerprint(body)}


def prepared_execution_document(
    *, project_binding: Mapping[str, object],
    component: Mapping[str, object],
    capacity_execution: Mapping[str, object],
    channel_composition_fingerprint: str,
    face_reconstruction_fingerprint: str,
    implementation_source_hashes: Mapping[str, str],
    base_deck_sha256: str, live_deck_sha256: str,
    base_metadata_sha256: str, live_metadata_sha256: str,
    capacity_sidecar_sha256: str, project_binding_sidecar_sha256: str,
    adapter_executable_sha256: str, adapter_receipt_sha256: str,
    adapter_derived_source_sha256: str,
) -> dict[str, object]:
    """Bind exact prepared bytes to the one compiled native adapter."""
    binding = validate_project_binding(project_binding)
    capacity = load_capacity_foundation_execution(capacity_execution)
    expected_component = component_document(
        project_binding=binding, capacity_execution=capacity,
        channel_composition_fingerprint=channel_composition_fingerprint,
        face_reconstruction_fingerprint=face_reconstruction_fingerprint,
        implementation_source_hashes=implementation_source_hashes,
    )
    if (not isinstance(component, Mapping) or dict(component) != expected_component or
            canonical_bytes(component) != canonical_bytes(expected_component)):
        raise ValueError("Q3-H4b component does not match prepared upstreams")
    actual_component = expected_component
    hashes = {
        "base_deck_sha256": _sha(base_deck_sha256, "base deck hash"),
        "live_deck_sha256": _sha(live_deck_sha256, "live deck hash"),
        "base_metadata_sha256": _sha(
            base_metadata_sha256, "base metadata hash"),
        "live_metadata_sha256": _sha(
            live_metadata_sha256, "live metadata hash"),
        "capacity_sidecar_sha256": _sha(
            capacity_sidecar_sha256, "capacity sidecar hash"),
        "project_binding_sidecar_sha256": _sha(
            project_binding_sidecar_sha256, "project binding sidecar hash"),
        "adapter_executable_sha256": _sha(
            adapter_executable_sha256, "adapter executable hash"),
        "adapter_receipt_sha256": _sha(
            adapter_receipt_sha256, "adapter receipt hash"),
        "adapter_derived_source_sha256": _sha(
            adapter_derived_source_sha256, "adapter source hash"),
    }
    body: dict[str, object] = {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "envelope_version": ENVELOPE_VERSION,
        "stage": "prepared",
        "backend": "cpu",
        "method_id": binding["method_id"],
        "persistence_code": binding["persistence_code"],
        "formula_code": binding["formula_code"],
        "ordered_fraction_ids": binding["ordered_fraction_ids"],
        "project_binding_fingerprint": binding["fingerprint"],
        "capacity_execution_fingerprint": capacity["fingerprint"],
        "channel_composition_fingerprint": _sha(
            channel_composition_fingerprint,
            "channel-composition fingerprint"),
        "face_reconstruction_fingerprint": _sha(
            face_reconstruction_fingerprint,
            "face-reconstruction fingerprint"),
        "bed_slope_execution_fingerprint": actual_component["fingerprint"],
        "bed_slope_execution_identity": actual_component,
        "artifact_hashes": hashes,
        "deck_transform": (
            "exactly_one_authenticated_five_field_slope_off_record_to_"
            "ten_field_H4_record"
        ),
        "legacy_face_selector_bypassed": True,
        "slope_application_count": 1,
        "availability_application_count": 1,
        "runtime_wetdry_guard": "reject_internal_face_at_or_below_dry_tolerance",
        "launch_policy": "verify_all_hashes_then_exact_adapter_no_fallback",
    }
    return {**body, "fingerprint": fingerprint(body)}


def validate_prepared_execution(value: Mapping[str, object], *,
                                project_binding: Mapping[str, object],
                                capacity_execution: Mapping[str, object],
                                channel_composition_fingerprint: str,
                                face_reconstruction_fingerprint: str,
                                implementation_source_hashes: Mapping[str, str],
                                ) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("Q3-H4b prepared execution must be an object")
    try:
        hashes = value["artifact_hashes"]
        rebuilt = prepared_execution_document(
            project_binding=project_binding,
            component=value["bed_slope_execution_identity"],
            capacity_execution=capacity_execution,
            channel_composition_fingerprint=channel_composition_fingerprint,
            face_reconstruction_fingerprint=face_reconstruction_fingerprint,
            implementation_source_hashes=implementation_source_hashes,
            **hashes,
        )
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError("Invalid Q3-H4b prepared execution") from error
    if dict(value) != rebuilt or canonical_bytes(value) != canonical_bytes(rebuilt):
        raise ValueError("Q3-H4b prepared execution is stale or forged")
    return rebuilt


def lifecycle_identity_document(
    *, stage: str, prepared_fingerprint: str,
    accepted_step: int, time_s: float,
    file_hashes: Mapping[str, str], parent_result_fingerprint: str | None = None,
) -> dict[str, object]:
    """Create result/restart identity without adding physical solver state."""
    if stage not in ("result", "restart"):
        raise ValueError("Q3-H4b lifecycle stage must be result or restart")
    if isinstance(accepted_step, bool) or not isinstance(accepted_step, int) or accepted_step < 0:
        raise ValueError("Q3-H4b accepted step must be a nonnegative integer")
    time_value = _finite(time_s, "accepted time", nonnegative=True)
    if not isinstance(file_hashes, Mapping) or not file_hashes:
        raise ValueError("Q3-H4b lifecycle identity requires output files")
    files = {str(name): _sha(value, f"{name} hash")
             for name, value in sorted(file_hashes.items())}
    if any(not name or name in (".", "..") or "/" in name or "\\" in name
           for name in files):
        raise ValueError("Q3-H4b lifecycle file names must be local basenames")
    parent = None
    if stage == "restart":
        parent = _sha(parent_result_fingerprint, "parent result fingerprint")
    elif parent_result_fingerprint is not None:
        raise ValueError("Result identity cannot declare a restart parent")
    body: dict[str, object] = {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "envelope_version": ENVELOPE_VERSION,
        "stage": stage,
        "prepared_execution_fingerprint": _sha(
            prepared_fingerprint, "prepared fingerprint"),
        "accepted_step": accepted_step,
        "time_s": time_value,
        "file_hashes": files,
        "parent_result_fingerprint": parent,
        "physical_state_extension": False,
        "rollback_state_extension": False,
        "cancel_state_extension": False,
        "restart_policy": "exact_prepared_and_parent_result_identity",
    }
    return {**body, "fingerprint": fingerprint(body)}


def validate_lifecycle_identity(value: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("Q3-H4b lifecycle identity must be an object")
    try:
        rebuilt = lifecycle_identity_document(
            stage=value["stage"],
            prepared_fingerprint=value["prepared_execution_fingerprint"],
            accepted_step=value["accepted_step"], time_s=value["time_s"],
            file_hashes=value["file_hashes"],
            parent_result_fingerprint=value["parent_result_fingerprint"],
        )
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError("Invalid Q3-H4b lifecycle identity") from error
    if dict(value) != rebuilt or canonical_bytes(value) != canonical_bytes(rebuilt):
        raise ValueError("Q3-H4b lifecycle identity is stale or forged")
    return rebuilt


__all__ = [
    "DEFAULT_CONTROLS", "ENVELOPE_VERSION", "PREPARED_EXECUTION_KEY",
    "PROJECT_BINDING_KEY", "RESTART_IDENTITY_KEY", "RESULT_IDENTITY_KEY",
    "SCHEMA", "SCOPE_ID", "SEMANTIC_ID", "SUPPORTED_LAWS",
    "canonical_bytes", "component_document", "fingerprint",
    "lifecycle_identity_document", "prepared_execution_document",
    "project_binding_document", "stage_fingerprint",
    "validate_lifecycle_identity", "validate_prepared_execution",
    "validate_project_binding",
]
