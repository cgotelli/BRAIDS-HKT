"""Automatic schema-v2 profiles for the first scalar-capacity foundation.

The execution-profile schema is intentionally generic: it validates a caller's
complete selection, but it does not decide which auxiliary operators are
physically consumed by a capacity formula.  This module owns that decision for
the first Q3-C slice.  It converts one canonical, backend-neutral capacity
profile into a complete bed-load-only execution profile for:

* corrected Delft3D MPM (initially with ``hiding.none``);
* corrected Delft3D Engelund--Hansen; and
* the parameterized Delft3D general relation.

All three consume the cell-local Chezy coefficient resolved from the selected
typed C/n/ks hydraulic field.  The roughness selector is therefore part of
every complete profile, including TraFrm=4; equal resolved Chezy values may
produce equal capacities without making those input-method identities
interchangeable.

The resolver never grants CPU or GPU admission.  Its output is backend neutral;
``runtime_execution_profile`` remains the single admission authority and will
reject a profile until the installed selector ledger contains an evidence-
bounded backend gate.  This separation lets QGIS persist every coefficient and
automatic dependency before a runtime implementation is promoted.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping

from .capacity_profile import (
    CapacityProfileError,
    capacity_profile_document,
    load_capacity_profile,
)
from .channel_composition import (
    TRA_BEDLOAD,
    channel_composition_profile_document,
    resolve_channel_composition_profile,
)
from .execution_profile_binding import canonical_execution_profile_binding
from .execution_profile_v2 import (
    Flow2DHExecutionProfile,
    bind_component_profile,
    execution_profile_document,
    load_selector_ledger,
    resolve_execution_profile,
)


CAPACITY_FOUNDATION_RESOLUTION_SCHEMA = 1
CAPACITY_FOUNDATION_RESOLUTION_SCOPE_ID = (
    "delft3d4-flow-2dh-q3c-capacity-foundation-resolution-v1"
)
CAPACITY_FOUNDATION_RESOLUTION_KEY = "capacity_foundation_resolution_v1"

CORRECTED_MPM_METHOD_ID = "capacity.mpm"
CORRECTED_EH_METHOD_ID = "capacity.engelund_hansen"
GENERAL_METHOD_ID = "capacity.general"
SUPPORTED_CAPACITY_METHOD_IDS = frozenset((
    CORRECTED_MPM_METHOD_ID,
    CORRECTED_EH_METHOD_ID,
    GENERAL_METHOD_ID,
))

_DIRECT_ROUGHNESS_CONSUMERS = frozenset((
    CORRECTED_MPM_METHOD_ID,
    CORRECTED_EH_METHOD_ID,
    GENERAL_METHOD_ID,
))
_ROUGHNESS_METHOD_IDS = frozenset((
    "roughness.chezy",
    "roughness.manning",
    "roughness.white_colebrook",
    "roughness.white_colebrook_waqua",
))
_FORMULA_PARAMETER_PREFIX = {
    CORRECTED_EH_METHOD_ID: "parameter.capacity.formula_1.",
    CORRECTED_MPM_METHOD_ID: "parameter.capacity.formula_2.",
    GENERAL_METHOD_ID: "parameter.capacity.formula_4.",
}
_MODEL_METHODS = (
    ("sediment_shear", "shear.total_roughness",
     "the selected capacities use total hydraulic forcing; no RP_TAUB skin replacement is consumed"),
    ("hiding", "hiding.none",
     "the first capacity-foundation slice applies no generic hiding/exposure correction"),
    ("bed_slope", "bed_slope.off",
     "bed-slope transport correction is isolated for the later Q3-H operator gate"),
    ("secondary_flow", "secondary.off",
     "secondary-flow direction steering is isolated for the later Q3-H operator gate"),
)
_CONTROL_VALUES = (
    ("control.upwind_bedload", True),
    ("control.laterally_averaged_bedload", False),
    ("control.maximum_waterdepth", False),
    ("control.maximum_waterdepth_fraction", 1.0),
    ("control.flux_limiter", "none"),
)


class CapacityFoundationProfileError(ValueError):
    """Fail-closed resolver error with a stable diagnostic category."""

    def __init__(self, message: str, *, category: str):
        super().__init__(message)
        self.category = category


@dataclass(frozen=True)
class CapacityFoundationBundle:
    """Backend-neutral profile, exact components, project binding and receipt."""

    execution_profile: Flow2DHExecutionProfile
    component_documents: Mapping[str, Mapping[str, object]]
    project_binding: Mapping[str, object]
    resolution_receipt: Mapping[str, object]


def _canonical(value: object) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise CapacityFoundationProfileError(
            "Capacity-foundation resolution is not canonical JSON",
            category="invalid_serialization",
        ) from error


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("ascii")).hexdigest()


def _canonical_mapping(value: Mapping[str, object]) -> dict[str, object]:
    return json.loads(_canonical(value))


def _roughness_method_from_project(project: Mapping[str, object]) -> str:
    roughness = project.get("roughness")
    if not isinstance(roughness, Mapping):
        raise CapacityFoundationProfileError(
            "Q3-C capacity requires an explicit hydraulic-roughness declaration",
            category="missing_dependency",
        )
    mode = roughness.get("mode", "manning")
    if mode == "chezy":
        return "roughness.chezy"
    if mode == "manning":
        return "roughness.manning"
    if mode == "white_colebrook":
        return (
            "roughness.white_colebrook_waqua"
            if roughness.get("white_colebrook_chzk2d", False) is True
            else "roughness.white_colebrook"
        )
    raise CapacityFoundationProfileError(
        f"Unknown hydraulic-roughness mode: {mode!r}",
        category="unknown_dependency",
    )


def _capacity_document(value: Mapping[str, object]) -> tuple[object, dict[str, object]]:
    try:
        profile = load_capacity_profile(value)
        document = capacity_profile_document(profile)
    except (CapacityProfileError, TypeError, ValueError) as error:
        raise CapacityFoundationProfileError(
            f"Capacity component is not canonical: {error}",
            category="invalid_component",
        ) from error
    if profile.backend is not None:
        raise CapacityFoundationProfileError(
            "Project capacity component must remain backend neutral",
            category="identity_mismatch",
        )
    return profile, document


def _uniform_method(profile: object) -> str:
    fractions = getattr(profile, "fractions", ())
    methods = {row.method_id for row in fractions}
    if len(methods) != 1:
        raise CapacityFoundationProfileError(
            "The first capacity-foundation profile requires one formula identity "
            "for every fraction",
            category="combination_not_implemented",
        )
    method_id = next(iter(methods))
    if method_id not in SUPPORTED_CAPACITY_METHOD_IDS:
        raise CapacityFoundationProfileError(
            f"Capacity method is outside the Q3-C scalar foundation: {method_id}",
            category="scope_restriction",
        )
    return method_id


def _require_initial_auxiliaries(profile: object) -> None:
    expected = {
        "sediment_shear": "shear.total_roughness",
        "hiding": "hiding.none",
        "bed_slope": "bed_slope.off",
        "secondary_flow": "secondary.off",
    }
    for field, method_id in expected.items():
        selected = getattr(getattr(profile, field), "method_id", None)
        if selected != method_id:
            raise CapacityFoundationProfileError(
                f"Initial capacity foundation resolves {field} automatically to "
                f"{method_id}; selected {selected!r}",
                category="combination_not_implemented",
            )


def _resolution_receipt(
    *, method_id: str, capacity_document: Mapping[str, object],
    roughness_method_id: str | None, profile: Flow2DHExecutionProfile,
    component_documents: Mapping[str, Mapping[str, object]],
    decisions: list[dict[str, object]],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": CAPACITY_FOUNDATION_RESOLUTION_SCHEMA,
        "scope_id": CAPACITY_FOUNDATION_RESOLUTION_SCOPE_ID,
        "capacity_method_id": method_id,
        "capacity_profile_fingerprint": capacity_document["fingerprint"],
        "direct_roughness_dependency": roughness_method_id,
        "resolved_methods": decisions,
        "parameter_binding_ids": [
            row.entry_id for row in profile.parameter_bindings
        ],
        "control_binding_ids": [
            row.entry_id for row in profile.control_bindings
        ],
        "component_fingerprints": {
            key: value["fingerprint"]
            for key, value in sorted(component_documents.items())
        },
        "execution_profile_fingerprint": profile.fingerprint,
        "backend": None,
        "admission_authority": (
            "installed selector-ledger exact-profile gate at runtime; this "
            "automatic resolver grants no CPU or GPU admission"
        ),
    }
    return {**payload, "fingerprint": _sha(payload)}


def resolve_capacity_foundation_profile(
    capacity_component: Mapping[str, object], *,
    roughness_method_id: str | None = None,
) -> CapacityFoundationBundle:
    """Resolve the complete neutral profile for one initial Q3-C combination.

    Every formula coefficient is copied from the canonical capacity component
    into a scoped schema-v2 parameter binding.  Optional FLOW coefficients are
    therefore already present with their source defaults; nothing is guessed by
    this resolver.  All fractions use the algebraic total-load/class-1 composer
    and no independently transported suspended reservoir.
    """
    capacity_profile, capacity_document = _capacity_document(capacity_component)
    method_id = _uniform_method(capacity_profile)
    _require_initial_auxiliaries(capacity_profile)

    if roughness_method_id not in _ROUGHNESS_METHOD_IDS:
        raise CapacityFoundationProfileError(
            "MPM, Engelund-Hansen, and the general relation require one typed "
            "C/n/ks hydraulic-roughness method",
            category="missing_dependency",
        )

    fractions = tuple(capacity_profile.fractions)
    fraction_ids = tuple(row.fraction_id for row in fractions)
    channel_profile = resolve_channel_composition_profile(
        fraction_ids, (TRA_BEDLOAD,) * len(fraction_ids),
        bed_factor=1.0, suspended_factor=1.0, epsilon=1.0e-6,
    )
    channel_document = channel_composition_profile_document(channel_profile)

    methods: list[dict[str, object]] = []
    decisions: list[dict[str, object]] = []
    for fraction_id in fraction_ids:
        scope = {"fraction_id": fraction_id}
        for family, selected, reason in (
            ("capacity", method_id,
             "explicit formula identity from the canonical capacity component"),
            ("transport_route", "transport_route.total",
             "class-1 algebraic total load owns bed plus formula-suspended capacity"),
            ("suspension", "suspension.off",
             "class-1 total load has no independent suspended reservoir"),
        ):
            methods.append({"entry_id": selected, "scope": scope})
            decisions.append({
                "family": family, "method_id": selected,
                "scope": scope, "reason": reason,
            })
    for family, selected, reason in _MODEL_METHODS:
        scope = {"model_id": "model"}
        methods.append({"entry_id": selected, "scope": scope})
        decisions.append({
            "family": family, "method_id": selected,
            "scope": scope, "reason": reason,
        })
    if roughness_method_id is not None:
        scope = {"model_id": "model"}
        methods.append({"entry_id": roughness_method_id, "scope": scope})
        decisions.append({
            "family": "hydraulic_roughness",
            "method_id": roughness_method_id,
            "scope": scope,
            "reason": (
                "corrected capacity consumes cell-local Chezy resolved from "
                "the selected typed C/n/ks hydraulic field"
            ),
        })

    prefix = _FORMULA_PARAMETER_PREFIX[method_id]
    parameters: list[dict[str, object]] = []
    ledger_entries = {
        row["entry_id"]: row for row in load_selector_ledger()["entries"]
    }
    for fraction in fractions:
        for coefficient in fraction.coefficients:
            entry_id = prefix + coefficient.keyword.lower()
            entry = ledger_entries.get(entry_id)
            if (not isinstance(entry, Mapping) or
                    entry.get("kind") != "parameter" or
                    method_id not in entry.get("compatibility", {}).get(
                        "requires_all", ())):
                raise CapacityFoundationProfileError(
                    f"Selector ledger lacks the scoped coefficient {entry_id}",
                    category="ledger_integrity",
                )
            parameters.append({
                "entry_id": entry_id,
                "representation": "inline",
                "scope": {"fraction_id": fraction.fraction_id},
                "value": coefficient.value,
            })

    controls = [{
        "entry_id": entry_id,
        "representation": "inline",
        "scope": {"model_id": "model"},
        "value": value,
    } for entry_id, value in _CONTROL_VALUES]
    components = {
        "capacity": capacity_document,
        "channel_composition": channel_document,
    }
    profile = resolve_execution_profile(
        backend=None,
        fraction_layout=[{
            "fraction_id": fraction_id,
            "transport_class": TRA_BEDLOAD,
        } for fraction_id in fraction_ids],
        method_selections=methods,
        parameter_bindings=parameters,
        control_bindings=controls,
        component_profiles=[
            bind_component_profile(component_id, components[component_id])
            for component_id in ("capacity", "channel_composition")
        ],
    )
    binding = canonical_execution_profile_binding(profile, components)
    receipt = _resolution_receipt(
        method_id=method_id,
        capacity_document=capacity_document,
        roughness_method_id=roughness_method_id,
        profile=profile,
        component_documents=components,
        decisions=decisions,
    )
    return CapacityFoundationBundle(
        profile,
        {
            key: _canonical_mapping(value) for key, value in components.items()
        },
        _canonical_mapping(binding),
        _canonical_mapping(receipt),
    )


def resolve_project_capacity_foundation_profile(
    project: Mapping[str, object],
) -> CapacityFoundationBundle | None:
    """Resolve an eligible QGIS project without selecting CPU or GPU.

    Unsupported formula families return ``None`` so established schema-v1
    projects retain their historical migration path.  Once one of the three
    Q3-C formula identities is present, missing or incompatible dependencies
    fail explicitly instead of falling back to a different formula.
    """
    binding = project.get("capacity_profile_binding")
    if not isinstance(binding, Mapping) or binding.get("state") != "canonical":
        return None
    component = binding.get("capacity_runtime_profile")
    if not isinstance(component, Mapping):
        return None
    capacity_profile, _ = _capacity_document(component)
    methods = {row.method_id for row in capacity_profile.fractions}
    if not methods.intersection(SUPPORTED_CAPACITY_METHOD_IDS):
        return None
    method_id = _uniform_method(capacity_profile)
    roughness = _roughness_method_from_project(project)
    return resolve_capacity_foundation_profile(
        component, roughness_method_id=roughness)


def validate_capacity_foundation_resolution(
    project: Mapping[str, object],
    receipt: Mapping[str, object],
) -> CapacityFoundationBundle:
    """Re-resolve a stored automatic choice and reject any changed byte."""
    if not isinstance(receipt, Mapping):
        raise CapacityFoundationProfileError(
            "Capacity-foundation resolution receipt must be an object",
            category="invalid_serialization",
        )
    bundle = resolve_project_capacity_foundation_profile(project)
    if bundle is None:
        raise CapacityFoundationProfileError(
            "Project no longer selects a Q3-C capacity-foundation formula",
            category="identity_mismatch",
        )
    if _canonical(receipt) != _canonical(bundle.resolution_receipt):
        raise CapacityFoundationProfileError(
            "Capacity-foundation automatic-resolution receipt is stale or forged",
            category="identity_mismatch",
        )
    return bundle


def capacity_foundation_profile_document(
    bundle: CapacityFoundationBundle,
) -> dict[str, object]:
    """Return the verified execution-profile document for consumers/tests."""
    if not isinstance(bundle, CapacityFoundationBundle):
        raise CapacityFoundationProfileError(
            "Expected a CapacityFoundationBundle",
            category="invalid_configuration",
        )
    return execution_profile_document(bundle.execution_profile)


__all__ = [
    "CAPACITY_FOUNDATION_RESOLUTION_SCHEMA",
    "CAPACITY_FOUNDATION_RESOLUTION_SCOPE_ID",
    "CAPACITY_FOUNDATION_RESOLUTION_KEY",
    "CORRECTED_MPM_METHOD_ID", "CORRECTED_EH_METHOD_ID",
    "GENERAL_METHOD_ID", "SUPPORTED_CAPACITY_METHOD_IDS",
    "CapacityFoundationProfileError", "CapacityFoundationBundle",
    "resolve_capacity_foundation_profile",
    "resolve_project_capacity_foundation_profile",
    "validate_capacity_foundation_resolution",
    "capacity_foundation_profile_document",
]
