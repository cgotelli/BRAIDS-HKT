"""Strict runtime reader for the additive FLOW-2DH live catalogue v1.

The module deliberately does not merge or rewrite the frozen schema-v2 JSON.
It verifies the independent identity projections used by project/preflight
code and exposes deterministic compatibility and admission queries.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping


LIVE_CATALOGUE_FILE = Path(__file__).with_name("flow2dh_live_catalogue_v1.json")
PARENT_LEDGER_FILE = Path(__file__).with_name("flow2dh_selector_ledger_v2.json")
LIVE_CATALOGUE_SCHEMA = 1
LIVE_CATALOGUE_REVISION = "flow2dh-live-catalogue-v1"
BACKENDS = frozenset(("cpu", "implicit_fwave_midpoint_gpu"))
ADMITTED = frozenset(("public", "public_limited"))
_SHA = re.compile(r"[0-9a-f]{64}")
_PARENT_SHA256 = "389ddc2a10c2b1a04b8ee35d64a374ff31467d691ceeb2c1f314220fc90f4b76"
_PARENT_SEMANTIC_ID = "2d9346640e7235baf0be9ca99294328dca2b36f0f12d439e3734d6a6068a8611"
_PARENT_ADMISSION_ID = "71098b78d00838ec66abdca594a1c211cb3262a1d33440784d952d39d1c0535c"
_AMENDMENT_SHA256 = "c7e4976055ea91af4128994a228543a5a6bd359d1bfcc70b11a30da4af283fa1"
_FROZEN_LIVE_PHYSICAL_ID = "00f6f71f057c1834819c721373d1b6611ddc4f7744ec44435a167e1758a453da"
_SETTLING_HOST_EXTENSION_ID = "settling.user_dll"
SETTLING_HOST_EXTENSION_ID = _SETTLING_HOST_EXTENSION_ID
_CURRENT_ONLY = frozenset((
    "capacity.bijker", "capacity.santoss", "capacity.van_thiel_van_rijn",
    "capacity.xbeach_soulsby_van_rijn",
))
_EXTENSION_IDS = frozenset((
    "capacity.engelund_hansen_constant_chezy_legacy",
    "capacity.mpm_specified_cf_legacy", "capacity.gaia_research",
    "bed_slope.braids_legacy_bagnold", "bed_slope.gaia_talmon",
    "secondary.gaia_research", "redistribution.braids_toe_collapse",
    "bedload_adaptation.steady_streamline",
    "parameter.bedload_adaptation.length_diameters",
))
_IMPLEMENTATION_STATES = frozenset((
    "catalogued_only", "component_present", "legacy_live_route", "live_route",
    "missing", "not_applicable",
))
_CONNECTION_STATES = frozenset(("connected", "not_connected", "not_applicable"))
_EXPECTED_IDENTITY = {
    "live_physical_semantic_id": "00f6f71f057c1834819c721373d1b6611ddc4f7744ec44435a167e1758a453da",
    "catalogue_erratum_registry_id": "3a0a3b6a611fa254c14582414e556a41c20b2dbfcb64dea6df37a1a3e40f2cc1",
    "ownership_registry_id": "15cc9f42eecc68a8b1d26f38952605d10c765559cc126c09ff51ea65498ee410",
    "implementation_registry_id": "30a7d4af57109e08a94856e9510be7f9e51adcfe4b05d2c9d3a6e2f6339a942a",
    "accepted_state_registry_id": "35425ff32fb9c7adc71cbada725fc87e0e1e2a8f1c41ef8c0ff56080dd3698af",
    "exact_profile_registry_id": "8691b4611386532f246d5dbb49cb4eacbe409206866185556a0526621a8e15e4",
    "product_registry_id": "fa8885e15bb3b6837c9224f81e7d1b411cb071336d392e2a98ffa2584274de78",
    "live_catalogue_manifest_id": "2001ee9b2d887925372be29c1e0cd0433407cce51e46fc35d770e8216f0b4b6b",
}


class LiveCatalogueError(ValueError):
    """Fail-closed live-catalogue error with a stable category."""

    def __init__(self, message: str, *, category: str):
        super().__init__(message)
        self.category = category


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise LiveCatalogueError(
                f"Duplicate JSON key: {key!r}", category="invalid_serialization")
        result[key] = value
    return result


def _read_json(payload: bytes | str | Mapping[str, object]) -> dict[str, Any]:
    if isinstance(payload, Mapping):
        return copy.deepcopy(dict(payload))
    try:
        value = json.loads(payload, object_pairs_hook=_strict_object,
                           parse_constant=lambda value: (_ for _ in ()).throw(
                               ValueError(f"non-finite JSON value {value}")))
    except LiveCatalogueError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise LiveCatalogueError(
            "Live catalogue is not strict JSON", category="invalid_serialization") from exc
    if not isinstance(value, dict):
        raise LiveCatalogueError(
            "Live catalogue must be one JSON object", category="invalid_serialization")
    return value


def _semantic_bytes(value: object) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise LiveCatalogueError(
            "Live catalogue is not canonicalizable", category="invalid_serialization") from exc


def _hash(value: object) -> str:
    return hashlib.sha256(_semantic_bytes(value)).hexdigest()


def identity_payloads(document: Mapping[str, object]) -> dict[str, str]:
    """Recompute every independent live-catalogue identity projection."""
    scope = document.get("scope_contract")
    parent = document.get("parent_ledger")
    lifecycle = document.get("entry_lifecycle")
    if not isinstance(scope, Mapping) or not isinstance(parent, Mapping) or \
            not isinstance(lifecycle, list):
        raise LiveCatalogueError(
            "Live catalogue identity input is malformed", category="catalogue_integrity")
    physical = {
        "schema": 1,
        "scope_id": scope.get("effective_scope_id"),
        "parent_selector_semantic_id": parent.get("semantic_id"),
        "physical_revision": copy.deepcopy(document.get("physical_revision")),
    }
    physical_id = _hash(physical)
    erratum = {
        "schema": 1,
        "live_physical_semantic_id": physical_id,
        "catalogue_erratum": copy.deepcopy(document.get("catalogue_erratum")),
    }
    erratum_id = _hash(erratum)
    ownership = {
        "schema": 1, "live_physical_semantic_id": physical_id,
        "rows": [{"entry_id": row.get("entry_id"),
                  "namespace": row.get("namespace"),
                  "owner_edges": copy.deepcopy(row.get("owner_edges"))}
                 for row in lifecycle if isinstance(row, Mapping)],
    }
    ownership_id = _hash(ownership)
    implementation = {
        "schema": 1, "live_physical_semantic_id": physical_id,
        "rows": [{"entry_id": row.get("entry_id"),
                  "implementation": copy.deepcopy(row.get("implementation")),
                  "non_target_backend_history": copy.deepcopy(
                      row.get("non_target_backend_history"))}
                 for row in lifecycle if isinstance(row, Mapping)],
    }
    implementation_id = _hash(implementation)
    accepted = {
        "schema": 1, "implementation_registry_id": implementation_id,
        "rows": [{"entry_id": row.get("entry_id"),
                  "accepted_state_connection": copy.deepcopy(
                      row.get("accepted_state_connection"))}
                 for row in lifecycle if isinstance(row, Mapping)],
    }
    accepted_id = _hash(accepted)
    exact = {
        "schema": 1, "live_physical_semantic_id": physical_id,
        "accepted_state_registry_id": accepted_id,
        "profiles": copy.deepcopy(document.get("exact_profile_admissions")),
    }
    exact_id = _hash(exact)
    product = {
        "schema": 1, "exact_profile_registry_id": exact_id,
        "profiles": copy.deepcopy(document.get("product_profile_admissions")),
        "legacy_surfaces": [
            {"entry_id": row.get("entry_id"),
             "legacy_surface": copy.deepcopy(row.get("legacy_surface"))}
            for row in lifecycle if isinstance(row, Mapping)
            and isinstance(row.get("legacy_surface"), Mapping)
            and row["legacy_surface"].get("exposed") is True
        ],
    }
    product_id = _hash(product)
    manifest = {
        "schema": 1,
        "live_physical_semantic_id": physical_id,
        "catalogue_erratum_registry_id": erratum_id,
        "ownership_registry_id": ownership_id,
        "implementation_registry_id": implementation_id,
        "accepted_state_registry_id": accepted_id,
        "exact_profile_registry_id": exact_id,
        "product_registry_id": product_id,
        "counts": copy.deepcopy(document.get("counts")),
    }
    return {
        "live_physical_semantic_id": physical_id,
        "catalogue_erratum_registry_id": erratum_id,
        "ownership_registry_id": ownership_id,
        "implementation_registry_id": implementation_id,
        "accepted_state_registry_id": accepted_id,
        "exact_profile_registry_id": exact_id,
        "product_registry_id": product_id,
        "live_catalogue_manifest_id": _hash(manifest),
    }


def _fail(message: str) -> None:
    raise LiveCatalogueError(message, category="catalogue_integrity")


def _evidence_rows(value: object, label: str, *, nonempty: bool = False) -> list:
    if not isinstance(value, list) or (nonempty and not value):
        _fail(f"{label} evidence inventory differs")
    for row in value:
        if not isinstance(row, Mapping) or set(row) != {"path", "sha256", "scope"}:
            _fail(f"{label} has malformed evidence")
        path = row.get("path")
        digest = row.get("sha256")
        scope = row.get("scope")
        if (not isinstance(path, str) or not path or Path(path).is_absolute() or
                ".." in Path(path).parts or not isinstance(digest, str) or
                _SHA.fullmatch(digest) is None or not isinstance(scope, str) or
                not scope):
            _fail(f"{label} has malformed evidence")
    return value


def _source_anchor_rows(value: object, label: str) -> list:
    if not isinstance(value, list) or not value:
        _fail(f"{label} source-anchor inventory differs")
    for row in value:
        if not isinstance(row, Mapping) or set(row) != {
                "path", "lines", "symbol", "role", "sha256"}:
            _fail(f"{label} has malformed source anchor")
        path = row.get("path")
        if (not isinstance(path, str) or not path or Path(path).is_absolute() or
                ".." in Path(path).parts or
                not all(isinstance(row.get(key), str) and row[key]
                        for key in ("lines", "symbol", "role")) or
                not isinstance(row.get("sha256"), str) or
                _SHA.fullmatch(row["sha256"]) is None):
            _fail(f"{label} has malformed source anchor")
    return value


def _validate_catalogue_erratum(
    document: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    erratum = document.get("catalogue_erratum")
    if not isinstance(erratum, Mapping) or set(erratum) != {
            "erratum_id", "status", "scope", "identity_preservation",
            "host_extension_dispositions", "source_evidence_corrections",
            "ownership_supersessions", "claims"}:
        _fail("Q0a catalogue erratum inventory differs")
    if (erratum.get("erratum_id") != "q0a-external-settling-interface-v1" or
            erratum.get("status") != "applied_zero_step" or
            erratum.get("scope") !=
            "catalogue_evidence_ownership_and_product_disposition_only"):
        _fail("Q0a catalogue erratum identity differs")
    if erratum.get("identity_preservation") != {
            "parent_ledger_sha256": _PARENT_SHA256,
            "parent_selector_semantic_id": _PARENT_SEMANTIC_ID,
            "live_physical_semantic_id": _FROZEN_LIVE_PHYSICAL_ID,
            "parent_bytes_unchanged": True,
            "physical_revision_unchanged": True,
    }:
        _fail("Q0a frozen-identity preservation claim differs")
    host_rows = erratum.get("host_extension_dispositions")
    if not isinstance(host_rows, list) or len(host_rows) != 1 or not isinstance(
            host_rows[0], Mapping):
        _fail("Q0a host-extension disposition inventory differs")
    host = host_rows[0]
    if set(host) != {
            "entry_id", "namespace", "kind", "subkind", "family", "label",
            "counts_as_method", "source_status", "provenance_class",
            "built_in_law", "selector", "owner_edges", "source_anchors",
            "braids_product_disposition", "braids_failure_phase",
            "built_in_method_inference", "exact_profile_admission"}:
        _fail("Q0a host-extension disposition fields differ")
    if (host.get("entry_id") != _SETTLING_HOST_EXTENSION_ID or
            host.get("namespace") != "delft3d4_flow_2dh_host_extension" or
            host.get("kind") != "interface" or
            host.get("subkind") != "host_extension" or
            host.get("family") != "settling" or
            host.get("counts_as_method") is not False or
            host.get("source_status") != "classic_flow_active_host_extension" or
            host.get("provenance_class") !=
            "delft_host_extension_not_builtin_law" or
            host.get("built_in_law") is not False or
            host.get("braids_product_disposition") !=
            "product_excluded_fail_closed" or
            host.get("braids_failure_phase") != "before_library_load" or
            host.get("built_in_method_inference") != "forbidden" or
            host.get("exact_profile_admission") is not False):
        _fail("Q0a host-extension classification differs")
    if host.get("selector") != {
            "section": "Sediment", "activation_keyword": "SettleLib",
            "dependent_keywords": ["SettleFunction", "SettleInput"],
            "activation_condition": "nonblank SettleLib",
            "selected_internal_code": "WS_FORM_USER_ROUTINE",
            "built_in_selector_keyword": "SettleFrm",
            "built_in_selector_evaluated_when_active": False,
    }:
        _fail("Q0a external-settling selector mapping differs")
    if host.get("owner_edges") != [{
            "milestone": "Q0a",
            "responsibility": "external settling host-interface disposition"}]:
        _fail("Q0a host-extension owner differs")
    _source_anchor_rows(host.get("source_anchors"), "Q0a host extension")

    corrections = erratum.get("source_evidence_corrections")
    if not isinstance(corrections, list) or len(corrections) != 1 or not isinstance(
            corrections[0], Mapping):
        _fail("Q0a evidence-correction inventory differs")
    correction = corrections[0]
    if (set(correction) != {"entry_id", "disposition",
                            "removed_parent_evidence_symbol",
                            "effective_delft_selector_evidence",
                            "parent_entry_unchanged", "reason",
                            "source_anchors"} or
            correction.get("entry_id") != "suspension.settling_only" or
            correction.get("disposition") != "superseded_in_live_manifest" or
            correction.get("removed_parent_evidence_symbol") != "SettleFunction" or
            correction.get("effective_delft_selector_evidence") != [] or
            correction.get("parent_entry_unchanged") is not True or
            not isinstance(correction.get("reason"), str) or
            "not a Delft selector" not in correction["reason"]):
        _fail("Q0a settling-only evidence correction differs")
    _source_anchor_rows(correction.get("source_anchors"),
                        "Q0a settling-only correction")

    ownership = erratum.get("ownership_supersessions")
    if not isinstance(ownership, list) or len(ownership) != 1 or not isinstance(
            ownership[0], Mapping):
        _fail("Q0a ownership-supersession inventory differs")
    supersession = ownership[0]
    if (set(supersession) != {"entry_id", "previous_live_owner",
                              "effective_live_owner", "disposition", "reason",
                              "source_anchors"} or
            supersession.get("entry_id") != "settling.function_conditions" or
            supersession.get("previous_live_owner") != "Q4" or
            supersession.get("effective_live_owner") != "Q9" or
            supersession.get("disposition") != "superseded_in_q0a"):
        _fail("Q0a settling ownership supersession differs")
    _source_anchor_rows(supersession.get("source_anchors"),
                        "Q0a settling ownership")
    if erratum.get("claims") != {
            "new_builtin_settling_laws": 0,
            "simulation_steps": 0,
            "delft_source_modifications": 0,
    }:
        _fail("Q0a zero-step claims differ")
    return {_SETTLING_HOST_EXTENSION_ID: host}


def _installed_parent() -> dict[str, Any]:
    if not PARENT_LEDGER_FILE.is_file():
        _fail("Frozen parent ledger is missing from the installed runtime")
    payload = PARENT_LEDGER_FILE.read_bytes()
    if hashlib.sha256(payload).hexdigest() != _PARENT_SHA256:
        _fail("Frozen parent ledger hash differs")
    parent = _read_json(payload)
    if (parent.get("semantic_id") != _PARENT_SEMANTIC_ID or
            parent.get("admission_registry_id") != _PARENT_ADMISSION_ID):
        _fail("Frozen parent ledger identity differs")
    return parent


def _validate_runtime_document(document: Mapping[str, object]) -> None:
    """Validate trusted bindings and the full admission lattice.

    Identity hashes detect accidental mutation.  These independent structural
    checks also prevent a caller from changing the payload and recomputing its
    self-declared hashes to manufacture implementation or product admission.
    """
    required = {
        "artifact_schema", "artifact_schema_version", "revision_id", "status",
        "parent_ledger", "ownership_amendment", "scope_contract",
        "allowed_values", "catalogue_erratum", "physical_revision", "entry_lifecycle",
        "exact_profile_admissions", "product_profile_admissions", "counts",
        "identity", "notes",
    }
    if (set(document) != required or
            document.get("artifact_schema") != "braids.flow2dh_live_catalogue_revision" or
            document.get("artifact_schema_version") != LIVE_CATALOGUE_SCHEMA or
            document.get("revision_id") != LIVE_CATALOGUE_REVISION or
            document.get("status") != "additive_live_registry"):
        _fail("Live catalogue top-level schema differs")

    parent_binding = document.get("parent_ledger")
    expected_parent_binding = {
        "schema_version": 2,
        "scope_id": "delft3d4_flow_2dh_river_sediment_morphodynamics",
        "semantic_id": _PARENT_SEMANTIC_ID,
        "admission_registry_id": _PARENT_ADMISSION_ID,
        "sha256": _PARENT_SHA256,
        "canonical_path": "braids_core/sediment/flow2dh_selector_ledger_v2.json",
        "validation_copy_path": "validation/sediment/delft3d_current/schema_v2/flow2dh_selector_ledger_v2.json",
        "counts": {
            "entries_total": 536,
            "entry_kinds": {"control": 6, "method": 163, "parameter": 367},
            "methods_total": 163, "parameters": 367, "controls": 6,
            "backend_profile_admissions": 192,
        },
    }
    if parent_binding != expected_parent_binding:
        _fail("Live catalogue parent binding differs")
    amendment = document.get("ownership_amendment")
    if amendment != {
        "amendment_id": "flow2dh-selector-ledger-v2-ownership-amendment-v1",
        "sha256": _AMENDMENT_SHA256,
        "path": "validation/sediment/delft3d_current/schema_v2/flow2dh_selector_ledger_v2_ownership_amendment_v1.json",
        "unique_assigned_parent_entries": 84,
        "ownership_edges": 129,
    }:
        _fail("Live catalogue ownership-amendment binding differs")
    if document.get("scope_contract") != {
        "effective_scope_id": "delft3d4_flow_2dh_river_sediment_morphodynamics_live_v1",
        "engine_reference": "Delft3D4 FLOW 2DH",
        "forcing_scope": "current_only",
        "cpu_backend": "cpu",
        "gpu_backend": "implicit_fwave_midpoint_gpu",
        "gpu_integrator": "implicit_fwave_midpoint",
        "backend_is_user_selected": True,
        "silent_backend_fallback": False,
        "component_evidence_is_not_exact_profile_admission": True,
        "exact_profile_admission_is_not_product_admission": True,
    }:
        _fail("Live catalogue scope contract differs")
    if document.get("allowed_values") != {
        "implementation_state": ["catalogued_only", "component_present",
                                 "legacy_live_route", "live_route", "missing",
                                 "not_applicable"],
        "accepted_state_connection": ["connected", "not_connected", "not_applicable"],
        "backend": ["cpu", "implicit_fwave_midpoint_gpu"],
        "product_surface_state": ["admitted", "not_admitted"],
    }:
        _fail("Live catalogue allowed-value registry differs")

    parent = _installed_parent()
    host_extensions = _validate_catalogue_erratum(document)
    parent_entries = {
        row["entry_id"]: row for row in parent.get("entries", [])
        if isinstance(row, Mapping) and isinstance(row.get("entry_id"), str)}
    if len(parent_entries) != 536:
        _fail("Frozen parent entry inventory differs")
    physical = document.get("physical_revision")
    if not isinstance(physical, Mapping) or set(physical) != {
        "requiredness_overrides", "gate_resolutions", "control_resolutions",
        "compatibility_rule_additions", "parent_compatibility_rules_inherited",
        "parent_compatibility_rule_overrides", "extension_namespace",
    }:
        _fail("Live physical revision inventory differs")
    overrides = {row.get("entry_id"): row for row in
                 physical.get("requiredness_overrides", [])
                 if isinstance(row, Mapping)}
    if set(overrides) != _CURRENT_ONLY or any(
            row.get("effective_requiredness") != "required" or
            row.get("wave_enabled") != "excluded_fail_closed" or
            row.get("external_secondary_rotation") !=
            "bypassed_for_direct_bed_vector" or
            row.get("global_face_slope") != "source_applied_after_formula"
            for row in overrides.values()):
        _fail("Current-only requiredness overrides differ")
    gate_ids = {row.get("entry_id") for row in physical.get("gate_resolutions", [])
                if isinstance(row, Mapping)}
    if gate_ids != {
        "parameter.morphology_parameter.eulerisoglm",
        "parameter.morphology_parameter.glmisoeuler",
        "parameter.morphology_parameter.multi",
    }:
        _fail("Resolved gate inventory differs")
    control_ids = {row.get("entry_id") for row in
                   physical.get("control_resolutions", [])
                   if isinstance(row, Mapping)}
    if control_ids != {
        "control.bedform_cfl", "control.flux_limiter",
        "control.laterally_averaged_bedload", "control.maximum_waterdepth",
        "control.maximum_waterdepth_fraction", "control.upwind_bedload",
    }:
        _fail("Resolved control inventory differs")
    rule_ids = {row.get("rule_id") for row in
                physical.get("compatibility_rule_additions", [])
                if isinstance(row, Mapping)}
    if rule_ids != {
        "current_only_capacity_rejects_wave_forcing",
        "current_only_direct_vectors_bypass_external_secondary_rotation",
        "current_only_internal_and_external_slope_ordering",
        "santoss_defaults_static_van_rijn_2007_roughness",
        "multi_external_merge_not_domain_parallelism",
        "gaia_research_triplet_bedload_only",
        "bedload_adaptation_requires_length",
        "bedload_adaptation_parameter_requires_method",
    }:
        _fail("Additive compatibility-rule inventory differs")
    overrides = physical.get("parent_compatibility_rule_overrides")
    if not isinstance(overrides, list) or len(overrides) != 1:
        _fail("Parent compatibility-rule override inventory differs")
    override = overrides[0]
    if (not isinstance(override, Mapping) or
            override.get("parent_rule_id") != "santoss_static_roughness_bundle" or
            override.get("disposition") != "superseded_in_live_v1" or
            override.get("superseded_by") !=
            "santoss_defaults_static_van_rijn_2007_roughness" or
            override.get("effective_requires_any") != [
                ["bedform_static.van_rijn_2007",
                 "bedform_roughness.power",
                 "bedform_roughness.van_rijn_1984",
                 "bedform_roughness.van_rijn_2007"],
            ]):
        _fail("SANTOSS parent compatibility-rule supersession differs")
    inherited_rules = [row.get("rule_id") for row in
                       parent.get("compatibility_rules", [])
                       if row.get("rule_id") != "santoss_static_roughness_bundle"]
    if physical.get("parent_compatibility_rules_inherited") != inherited_rules:
        _fail("Inherited compatibility-rule binding differs")
    namespace = physical.get("extension_namespace")
    if (not isinstance(namespace, Mapping) or
            namespace.get("coverage_rule") !=
            "excluded_from_strict_delft_denominator"):
        _fail("BRAIDS extension namespace differs")
    extension_rows = {row.get("entry_id"): row for row in
                      namespace.get("entries", []) if isinstance(row, Mapping)}
    if set(extension_rows) != _EXTENSION_IDS:
        _fail("BRAIDS extension inventory differs")

    lifecycle = document.get("entry_lifecycle")
    if not isinstance(lifecycle, list) or len(lifecycle) != 546:
        _fail("Live catalogue lifecycle inventory differs")
    identifiers = [row.get("entry_id") for row in lifecycle
                   if isinstance(row, Mapping)]
    if identifiers != sorted(set(parent_entries) | _EXTENSION_IDS |
                             set(host_extensions)):
        _fail("Live catalogue lifecycle IDs differ")
    life_by_id = {row["entry_id"]: row for row in lifecycle}
    exposed = {
        "bedload_adaptation.steady_streamline",
        "parameter.bedload_adaptation.length_diameters",
        "suspension.settling_only",
    }
    for entry_id, row in life_by_id.items():
        if set(row) != {
            "entry_id", "namespace", "kind", "family", "owner_edges",
            "implementation", "non_target_backend_history",
            "accepted_state_connection", "exact_profile_refs",
            "product_profile_refs", "legacy_surface",
        }:
            _fail(f"{entry_id}: lifecycle field inventory differs")
        source = parent_entries.get(
            entry_id, extension_rows.get(entry_id, host_extensions.get(entry_id)))
        expected_namespace = ("delft3d4_flow_2dh_parent" if entry_id in
                              parent_entries else source.get(
                                  "coverage_namespace", source.get("namespace")))
        if (row.get("kind") != source.get("kind") or
                row.get("family") != source.get("family") or
                row.get("namespace") != expected_namespace):
            _fail(f"{entry_id}: lifecycle source identity differs")
        owners = row.get("owner_edges")
        if not isinstance(owners, list) or not owners:
            _fail(f"{entry_id}: owner inventory differs")
        legacy = row.get("legacy_surface")
        if (not isinstance(legacy, Mapping) or
                set(legacy) != {"exposed", "marker", "is_product_admission"} or
                legacy.get("is_product_admission") is not False or
                (entry_id in exposed) != (legacy.get("exposed") is True) or
                ((entry_id in exposed) != isinstance(legacy.get("marker"), str))):
            _fail(f"{entry_id}: legacy visibility differs")
        implementation = row.get("implementation")
        connection = row.get("accepted_state_connection")
        exact_refs = row.get("exact_profile_refs")
        product_refs = row.get("product_profile_refs")
        if (not isinstance(implementation, Mapping) or set(implementation) != BACKENDS or
                not isinstance(connection, Mapping) or set(connection) != BACKENDS or
                not isinstance(exact_refs, Mapping) or set(exact_refs) != BACKENDS or
                not isinstance(product_refs, Mapping) or
                set(product_refs) != {"qgis_prepare_run", "windows_package"}):
            _fail(f"{entry_id}: backend lifecycle inventory differs")
        for backend in BACKENDS:
            impl = implementation[backend]
            conn = connection[backend]
            if (not isinstance(impl, Mapping) or set(impl) != {"state", "evidence"} or
                    impl.get("state") not in _IMPLEMENTATION_STATES):
                _fail(f"{entry_id}/{backend}: implementation state differs")
            _evidence_rows(impl.get("evidence"), f"{entry_id}/{backend} implementation")
            if (not isinstance(conn, Mapping) or
                    set(conn) != {"state", "envelopes", "evidence"} or
                    conn.get("state") not in _CONNECTION_STATES or
                    not isinstance(conn.get("envelopes"), list)):
                _fail(f"{entry_id}/{backend}: accepted-state connection differs")
            _evidence_rows(conn.get("evidence"), f"{entry_id}/{backend} connection")
            if conn["state"] == "connected":
                if impl["state"] not in {"live_route", "legacy_live_route"}:
                    _fail(f"{entry_id}/{backend}: connected state lacks live implementation")
                if not impl["evidence"] or not conn["evidence"] or not conn["envelopes"]:
                    _fail(f"{entry_id}/{backend}: connected state lacks evidence")
            elif conn["envelopes"] or conn["evidence"]:
                _fail(f"{entry_id}/{backend}: unconnected state carries admission evidence")
            refs = exact_refs.get(backend)
            if not isinstance(refs, list) or refs != sorted(set(refs)):
                _fail(f"{entry_id}/{backend}: exact-profile refs differ")
            for surface, backend_rows in product_refs.items():
                if (not isinstance(backend_rows, Mapping) or
                        set(backend_rows) != BACKENDS or
                        not isinstance(backend_rows.get(backend), list) or
                        backend_rows[backend] != sorted(set(backend_rows[backend]))):
                    _fail(f"{entry_id}/{backend}: {surface} refs differ")
        if entry_id in _CURRENT_ONLY:
            for backend in BACKENDS:
                if (implementation[backend]["state"] != "missing" or
                        connection[backend]["state"] != "not_connected" or
                        exact_refs[backend] or any(
                            surface[backend] for surface in product_refs.values())):
                    _fail(f"{entry_id}/{backend}: current-only route was promoted")
        if entry_id == _SETTLING_HOST_EXTENSION_ID:
            if (row.get("owner_edges") != host_extensions[entry_id]["owner_edges"] or
                    any(implementation[backend]["state"] != "not_applicable" or
                        connection[backend]["state"] != "not_applicable" or
                        exact_refs[backend] or any(
                            surface[backend] for surface in product_refs.values())
                        for backend in BACKENDS)):
                _fail("Q0a host extension was promoted or mis-owned")

    if life_by_id["settling.function_conditions"]["owner_edges"] != [{
            "milestone": "Q9",
            "responsibility": "source-conditional clay/silt settling"}]:
        _fail("settling.function_conditions live owner differs from Q0a")

    exact_rows = document.get("exact_profile_admissions")
    if not isinstance(exact_rows, list):
        _fail("Exact-profile registry differs")
    exact = {row.get("admission_id"): row for row in exact_rows
             if isinstance(row, Mapping)}
    parent_exact = {row.get("admission_id"): row for row in
                    parent.get("backend_profile_admissions", [])
                    if isinstance(row, Mapping)}
    if len(exact_rows) != 192 or set(exact) != set(parent_exact):
        _fail("Exact-profile registry does not preserve the frozen parent")
    inverse_exact = {entry_id: {backend: [] for backend in BACKENDS}
                     for entry_id in life_by_id}
    for admission_id, row in exact.items():
        inherited = copy.deepcopy(dict(row))
        inherited.pop("origin", None)
        profile_id = inherited.pop("profile_semantic_id", None)
        if inherited != parent_exact[admission_id] or row.get("origin") != \
                "frozen_parent_registry":
            _fail(f"{admission_id}: exact-profile inheritance differs")
        preimage = {
            "parent_selector_semantic_id": _PARENT_SEMANTIC_ID,
            "admission_id": admission_id,
            "exact_method_ids": row.get("exact_method_ids"),
            "method_selection_count": row.get("method_selection_count"),
            "fraction_layout": row.get("fraction_layout"),
        }
        if profile_id != _hash(preimage):
            _fail(f"{admission_id}: exact-profile identity differs")
        for entry_id in row.get("exact_method_ids", []):
            if entry_id not in life_by_id or life_by_id[entry_id]["kind"] != "method":
                _fail(f"{admission_id}: exact profile contains an unknown method")
            for backend in BACKENDS:
                if row["backend_status"][backend] in ADMITTED:
                    inverse_exact[entry_id][backend].append(admission_id)
        _evidence_rows(row.get("evidence"), f"{admission_id} exact profile",
                       nonempty=True)
    for entry_id, row in life_by_id.items():
        for backend in BACKENDS:
            if row["exact_profile_refs"][backend] != sorted(
                    inverse_exact[entry_id][backend]):
                _fail(f"{entry_id}/{backend}: inverse exact-profile refs differ")
            if row["exact_profile_refs"][backend] and row[
                    "accepted_state_connection"][backend]["state"] != "connected":
                _fail(f"{entry_id}/{backend}: exact profile lacks accepted state")

    product_rows = document.get("product_profile_admissions")
    if not isinstance(product_rows, list) or len(product_rows) != 1:
        _fail("Product-profile registry differs")
    product = product_rows[0]
    product_id = "qgis.q3sp.law11_total_single_fraction_v1"
    exact_id = "q3sp.law11_total_single_fraction_v1"
    surfaces = {
        "qgis_project": "admitted", "qgis_prepare_run": "admitted",
        "qgis_results_restart": "admitted", "windows_package": "not_admitted",
    }
    if (not isinstance(product, Mapping) or
            product.get("product_profile_id") != product_id or
            product.get("exact_admission_id") != exact_id or
            product.get("backend_status") != {
                backend: "qgis_admitted" for backend in BACKENDS} or
            product.get("surfaces") != surfaces):
        _fail("Q3SP product admission differs")
    exact_product = exact.get(exact_id)
    if exact_product is None or any(exact_product["backend_status"][backend]
                                    not in ADMITTED for backend in BACKENDS):
        _fail("Q3SP product does not reference an admitted exact profile")
    _evidence_rows(product.get("evidence"), "Q3SP product", nonempty=True)
    surface_evidence = product.get("surface_evidence")
    if not isinstance(surface_evidence, Mapping) or set(surface_evidence) != set(surfaces):
        _fail("Q3SP surface-evidence inventory differs")
    for surface, state in surfaces.items():
        rows = _evidence_rows(surface_evidence.get(surface),
                              f"Q3SP {surface}")
        if (state == "admitted") != bool(rows):
            _fail(f"Q3SP {surface} evidence/admission differs")
    product_methods = set(exact_product["exact_method_ids"])
    for entry_id, row in life_by_id.items():
        for backend in BACKENDS:
            expected = [product_id] if entry_id in product_methods else []
            if row["product_profile_refs"]["qgis_prepare_run"][backend] != expected:
                _fail(f"{entry_id}/{backend}: inverse product refs differ")
            if row["product_profile_refs"]["windows_package"][backend]:
                _fail(f"{entry_id}/{backend}: packaged product is not admitted")
            if expected and exact_id not in row["exact_profile_refs"][backend]:
                _fail(f"{entry_id}/{backend}: product ref lacks exact-profile ref")

    counts = document.get("counts")
    if not isinstance(counts, Mapping) or counts.get("parent_lifecycle_rows") != 536 or \
            counts.get("extension_entries") != 9 or \
            counts.get("host_extension_dispositions") != 1 or \
            counts.get("catalogue_evidence_corrections") != 1 or \
            counts.get("catalogue_ownership_supersessions") != 1 or \
            counts.get("lifecycle_rows_total") != 546 or \
            counts.get("effective_strict_delft_methods") != 150 or \
            counts.get("parent_compatibility_rule_overrides") != 1 or \
            counts.get("exact_profile_rows") != 192 or \
            counts.get("product_profile_rows") != 1 or \
            counts.get("windows_package_admitted_rows") != 0:
        _fail("Live catalogue summary counts differ")
    for state_key, row_key in (("implementation_states", "implementation"),
                               ("accepted_state_states", "accepted_state_connection")):
        derived: dict[str, dict[str, int]] = {backend: {} for backend in BACKENDS}
        for row in lifecycle:
            for backend in BACKENDS:
                state = row[row_key][backend]["state"]
                derived[backend][state] = derived[backend].get(state, 0) + 1
        if counts.get(state_key) != derived:
            _fail(f"Live catalogue {state_key} counts differ")

    identity = document.get("identity")
    if (not isinstance(identity, Mapping) or set(identity) != {
            "live_physical_semantic_id", "catalogue_erratum_registry_id",
            "ownership_registry_id",
            "implementation_registry_id", "accepted_state_registry_id",
            "exact_profile_registry_id", "product_registry_id",
            "live_catalogue_manifest_id",
    } or any(not isinstance(value, str) or _SHA.fullmatch(value) is None
             for value in identity.values())):
        _fail("Live catalogue identity inventory differs")
    if dict(identity) != identity_payloads(document):
        _fail("Live catalogue identity projection differs")
    if identity.get("live_physical_semantic_id") != _FROZEN_LIVE_PHYSICAL_ID:
        _fail("Q0a changed the frozen live physical semantic ID")
    if dict(identity) != _EXPECTED_IDENTITY:
        _fail("Live catalogue identity is not the frozen v1 runtime identity")


def load_live_catalogue(
    path: str | Path = LIVE_CATALOGUE_FILE,
) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file():
        raise LiveCatalogueError(
            f"Installed live catalogue is missing: {target}", category="missing_catalogue")
    document = _read_json(target.read_bytes())
    _validate_runtime_document(document)
    return document


def _catalogue(document: Mapping[str, object] | None) -> dict[str, Any]:
    if document is None:
        return load_live_catalogue()
    if not isinstance(document, Mapping):
        raise LiveCatalogueError(
            "Injected live catalogue must be an object",
            category="catalogue_integrity")
    value = copy.deepcopy(dict(document))
    _validate_runtime_document(value)
    return value


def lifecycle_entry(entry_id: str, *, document: Mapping[str, object] | None = None) -> dict:
    catalogue = _catalogue(document)
    rows = catalogue.get("entry_lifecycle")
    if not isinstance(rows, list):
        raise LiveCatalogueError(
            "Live catalogue lifecycle is malformed", category="catalogue_integrity")
    for row in rows:
        if isinstance(row, Mapping) and row.get("entry_id") == entry_id:
            return copy.deepcopy(dict(row))
    raise LiveCatalogueError(
        f"Unknown live-catalogue entry: {entry_id}", category="unknown_entry")


def effective_requiredness(entry_id: str, *, document: Mapping[str, object] | None = None,
                           parent_requiredness: str | None = None) -> str | None:
    catalogue = _catalogue(document)
    known = {row["entry_id"] for row in catalogue["entry_lifecycle"]}
    if entry_id not in known:
        raise LiveCatalogueError(
            f"Unknown live-catalogue entry: {entry_id}", category="unknown_entry")
    physical = catalogue.get("physical_revision")
    if not isinstance(physical, Mapping):
        raise LiveCatalogueError(
            "Live physical revision is malformed", category="catalogue_integrity")
    rows = physical.get("requiredness_overrides")
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, Mapping) and row.get("entry_id") == entry_id:
                return str(row.get("effective_requiredness"))
    return parent_requiredness


def _composed_compatibility_rules(catalogue: Mapping[str, object]) -> list[dict]:
    physical = catalogue["physical_revision"]
    inherited_ids = set(physical["parent_compatibility_rules_inherited"])
    parent = _installed_parent()
    inherited = [copy.deepcopy(row) for row in parent["compatibility_rules"]
                 if row["rule_id"] in inherited_ids]
    additions = copy.deepcopy(physical["compatibility_rule_additions"])
    rules = inherited + additions
    identifiers = [row["rule_id"] for row in rules]
    if len(identifiers) != len(set(identifiers)):
        _fail("Composed compatibility-rule IDs are not unique")
    return rules


def effective_compatibility_rules(
    *, document: Mapping[str, object] | None = None,
) -> tuple[dict, ...]:
    """Return frozen parent rules after overlay supersessions, plus additions."""
    return tuple(_composed_compatibility_rules(_catalogue(document)))


def resolve_additive_compatibility(
    selected_method_ids: Iterable[str], *,
    parameter_values: Mapping[str, object] | None = None,
    wave_or_orbital_forcing: bool = False,
    document: Mapping[str, object] | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Apply Q0 additive rules and return canonical methods plus reasons.

    This is only the additive layer.  The existing schema-v2 resolver remains
    responsible for inherited rules and exact profile admission.
    """
    catalogue = _catalogue(document)
    if isinstance(selected_method_ids, (str, bytes)):
        raise LiveCatalogueError(
            "Selected method IDs must be an iterable, not a string",
            category="invalid_selection")
    try:
        selected_values = tuple(selected_method_ids)
    except TypeError as exc:
        raise LiveCatalogueError(
            "Selected method IDs must be iterable", category="invalid_selection") from exc
    if not all(isinstance(value, str) and value for value in selected_values):
        raise LiveCatalogueError(
            "Selected method IDs must be nonempty strings", category="invalid_selection")
    if len(set(selected_values)) != len(selected_values):
        raise LiveCatalogueError(
            "Selected method IDs must be unique", category="invalid_selection")
    selected = set(selected_values)
    if parameter_values is not None and not isinstance(parameter_values, Mapping):
        raise LiveCatalogueError(
            "Parameter values must be an object", category="invalid_parameter")
    parameters = dict(parameter_values or {})
    lifecycle = {row["entry_id"]: row for row in catalogue["entry_lifecycle"]}
    host_ids = {
        row["entry_id"] for row in catalogue["catalogue_erratum"][
            "host_extension_dispositions"]}
    requested_host_ids = selected & host_ids
    host_keywords = {"SettleLib", "SettleFunction", "SettleInput"}
    requested_host_keywords = {
        key for key in host_keywords if key in parameters and
        parameters[key] is not None and
        (not isinstance(parameters[key], str) or bool(parameters[key].strip()))}
    if requested_host_ids or requested_host_keywords:
        detail = sorted(requested_host_ids | requested_host_keywords)
        raise LiveCatalogueError(
            "External settling host interface is product-excluded and "
            "fails closed before library load: " + ", ".join(detail),
            category="product_restriction")
    known_methods = {entry_id for entry_id, row in lifecycle.items()
                     if row["kind"] == "method"}
    unknown = sorted(selected - known_methods)
    if unknown:
        raise LiveCatalogueError(
            f"Unknown or non-method selection: {', '.join(unknown)}",
            category="unknown_entry")
    rules = {row["rule_id"]: row for row in
             _composed_compatibility_rules(catalogue)}
    current = set(rules[
        "current_only_capacity_rejects_wave_forcing"]["when_any"])
    reasons: list[str] = []
    if selected & current and wave_or_orbital_forcing:
        raise LiveCatalogueError(
            "Current-only capacity cannot be combined with physical wave/orbital forcing",
            category="scope_incompatibility")
    if selected & current:
        secondary = selected & {
            "secondary.off", "secondary.equilibrium", "secondary.prognostic"}
        if not secondary:
            selected.add("secondary.off")
            reasons.append("secondary.off automatically selected as the neutral secondary-flow choice")
        else:
            reasons.append(
                "selected secondary-flow state remains active for its other consumers; direct bed-vector rotation is bypassed as in locked FLOW")
    santoss = rules["santoss_defaults_static_van_rijn_2007_roughness"]
    if set(santoss["when_all"]).issubset(selected):
        alternatives = selected & set(santoss["explicit_alternatives"])
        if not alternatives:
            selected.add(santoss["default_if_no_explicit_alternative"])
        reasons.append(santoss["reason"])
    gaia = rules["gaia_research_triplet_bedload_only"]
    if selected & set(gaia["when_any"]):
        selected.update(gaia["auto_select"])
        reasons.append(gaia["reason"])
    multi = rules["multi_external_merge_not_domain_parallelism"]
    multi_key, multi_value = next(iter(multi["when_parameter_equals"].items()))
    if parameters.get(multi_key) is multi_value:
        raise LiveCatalogueError(
            "Multi requests external Delft Mormerge and is not BRAIDS domain parallelism",
            category="scope_incompatibility")
    adaptation = rules["bedload_adaptation_requires_length"]
    adaptation_method = adaptation["when_all"][0]
    length_key = adaptation["required_parameter"]
    has_length = length_key in parameters
    if adaptation_method in selected:
        if not has_length or not _positive_parameter(parameters[length_key]):
            raise LiveCatalogueError(
                "Bedload adaptation requires a finite positive length in grain diameters",
                category="invalid_parameter")
        reasons.append(adaptation["reason"])
    orphan = rules["bedload_adaptation_parameter_requires_method"]
    if orphan["when_parameter_present"] in parameters and \
            orphan["requires_method"] not in selected:
        raise LiveCatalogueError(
            "Bedload adaptation length was supplied while its method is off",
            category="scope_incompatibility")
    exclusive_families = {"capacity", "bed_slope", "secondary_flow", "suspension",
                          "bedload_adaptation"}
    chosen_by_family: dict[str, list[str]] = {}
    for entry_id in selected:
        family = lifecycle[entry_id]["family"]
        if family in exclusive_families:
            chosen_by_family.setdefault(family, []).append(entry_id)
    conflicts = {family: sorted(values) for family, values in chosen_by_family.items()
                 if len(values) > 1}
    if conflicts:
        details = "; ".join(f"{family}={','.join(values)}"
                            for family, values in sorted(conflicts.items()))
        raise LiveCatalogueError(
            f"Mutually exclusive method selections: {details}",
            category="physical_incompatibility")
    return tuple(sorted(selected)), tuple(reasons)


def _positive_parameter(value: object) -> bool:
    def valid_scalar(item: object) -> bool:
        return (isinstance(item, (int, float)) and not isinstance(item, bool) and
                math.isfinite(float(item)) and float(item) > 0.0)
    if valid_scalar(value):
        return True
    if isinstance(value, (str, bytes)):
        return False
    if isinstance(value, (list, tuple)) and value:
        return all(valid_scalar(item) for item in value)
    return False


def exact_profile_admission(
    admission_id: str, backend: str, *, document: Mapping[str, object] | None = None,
) -> dict:
    if backend not in BACKENDS:
        raise LiveCatalogueError(
            f"Unknown backend: {backend}", category="invalid_backend")
    catalogue = _catalogue(document)
    for row in catalogue.get("exact_profile_admissions", []):
        if isinstance(row, Mapping) and row.get("admission_id") == admission_id:
            status = row.get("backend_status")
            if not isinstance(status, Mapping) or status.get(backend) not in ADMITTED:
                raise LiveCatalogueError(
                    f"Exact profile {admission_id} is not admitted on {backend}",
                    category="implementation_restriction")
            return copy.deepcopy(dict(row))
    raise LiveCatalogueError(
        f"Unknown exact profile admission: {admission_id}", category="unknown_profile")


def product_profile_admission(
    product_profile_id: str, backend: str, *, require_windows_package: bool = False,
    document: Mapping[str, object] | None = None,
) -> dict:
    if backend not in BACKENDS:
        raise LiveCatalogueError(
            f"Unknown backend: {backend}", category="invalid_backend")
    catalogue = _catalogue(document)
    for row in catalogue.get("product_profile_admissions", []):
        if not isinstance(row, Mapping) or row.get("product_profile_id") != product_profile_id:
            continue
        status = row.get("backend_status")
        surfaces = row.get("surfaces")
        if not isinstance(status, Mapping) or status.get(backend) != "qgis_admitted":
            raise LiveCatalogueError(
                f"Product profile {product_profile_id} is not admitted on {backend}",
                category="product_restriction")
        if require_windows_package and (not isinstance(surfaces, Mapping) or
                                        surfaces.get("windows_package") != "admitted"):
            raise LiveCatalogueError(
                f"Product profile {product_profile_id} is not admitted in a Windows package",
                category="product_restriction")
        return copy.deepcopy(dict(row))
    raise LiveCatalogueError(
        f"Unknown product profile admission: {product_profile_id}",
        category="unknown_product_profile")


__all__ = [
    "ADMITTED", "BACKENDS", "LIVE_CATALOGUE_FILE", "LIVE_CATALOGUE_REVISION",
    "LIVE_CATALOGUE_SCHEMA", "SETTLING_HOST_EXTENSION_ID",
    "LiveCatalogueError", "effective_requiredness",
    "effective_compatibility_rules", "exact_profile_admission",
    "identity_payloads", "lifecycle_entry",
    "load_live_catalogue", "product_profile_admission",
    "resolve_additive_compatibility",
]
