"""Canonical runtime contract for Delft3D-current capacity profiles.

This module is deliberately independent of NumPy, QGIS, and either numerical
backend.  It resolves a project-facing selection into the immutable physical
contract that CPU and ``implicit_fwave_midpoint`` GPU dispatch must consume.
It does *not* make a prototype route public: specifying a backend admits only
catalogue routes whose status is already public or public-limited.

The identifiers in this file have three different meanings and must never be
substituted for one another:

* ``formula_code`` is BRAIDS' native formula-dispatch code.  For historical
  laws 2 and 8 it is not a complete persistence identity: the method ID also
  distinguishes the legacy closure from the corrected Delft3D closure;
* ``reference_selector`` is Delft3D FLOW's ``TraFrm`` value;
* ``method_id`` is the stable, human-readable semantic identity.

In particular, BRAIDS bed-slope persistence code 4 remains the historical
TELEMAC-GAIA Talmon option.  Delft3D ``ISlope=4`` (Parker-Andrews) is persisted
as BRAIDS code 6.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Iterable, Mapping

from braids_core.sediment.capacity_parameters import (
    FORMULA_BY_LEGACY_CODE,
    FORMULA_CONTRACTS,
    CapacityFormulaContract,
    CapacityParameterError,
    canonical_parameter_items,
    capacity_parameter_catalog_id,
)
from braids_core.sediment_method_catalog import (
    ADMITTED_ROUTE_STATUSES,
    METHOD_BY_ID,
    SCOPE_EXCLUDED,
    SCOPE_REFERENCE_DISABLED,
    MethodSpec,
    legacy_code_map,
    method,
    methods_for_component,
    semantic_catalog_id,
)


CAPACITY_PROFILE_SCHEMA = 1
CAPACITY_PROFILE_SCOPE_ID = "delft3d4-flow-2dh-current-capacity-runtime-v1"
PARKER_ANDREWS_BRAIDS_SLOPE_CODE = 6
GAIA_TALMON_BRAIDS_SLOPE_CODE = 4

BED_CHANNEL_SCALAR = "scalar_magnitude"
BED_CHANNEL_VECTOR = "direct_vector"
SUSPENDED_CHANNEL_SCALAR = "scalar_magnitude"
SUSPENDED_CHANNEL_VECTOR = "direct_vector"
SUSPENDED_CHANNEL_ZERO = "source_identically_zero"
DIRECTION_SCALAR = "external_scalar_direction"
DIRECTION_VECTOR = "formula_direct_vectors"
SHEAR_TOTAL_HYDRAULICS = "total_hydraulic_state"
SHEAR_SELECTED_RP_TAUB = "selected_rp_taub"
CAPACITY_UNITS = "solid_mass_per_time_per_width"

# The M1 profile identity above deliberately remains unchanged: v46 projects
# and restart files already bind it.  M2 adds a separate identity for the
# diagnostic row semantics so older v46 byte streams keep their exact meaning.
CAPACITY_DIAGNOSTIC_SCHEMA = 1
CAPACITY_DIAGNOSTIC_SCOPE_ID = (
    "delft3d4-flow-2dh-current-capacity-diagnostics-v1"
)
DIAGNOSTIC_ABSOLUTE_SHIELDS = 1
DIAGNOSTIC_CRITICAL_SHIELDS = 2
DIAGNOSTIC_STRESS_RATIO = 4
DIAGNOSTIC_TRANSPORT_STAGE = 8

_BACKENDS = frozenset(("cpu", "implicit_gpu"))
_SHEAR_METHOD_IDS = (
    "shear.total_roughness",
    "shear.soulsby_clarke_skin",
)
_HIDING_METHOD_IDS = (
    "hiding.none",
    "hiding.egiazaroff",
    "hiding.ashida_michiue",
    "hiding.parker_klm",
    "hiding.wu_wang_jia",
)
_HIDING_SLOPE_CONSUMER_METHOD_IDS = (
    "bed_slope.delft_parker_andrews",
)
_DELFT_SLOPE_METHOD_IDS = (
    "bed_slope.off",
    "bed_slope.delft_bagnold_ikeda",
    "bed_slope.delft_koch_flokstra",
    "bed_slope.delft_parker_andrews",
)
_SECONDARY_METHOD_IDS = (
    "secondary.off",
    "secondary.equilibrium",
    "secondary.prognostic",
)
_RP_TAUB_SELECTORS = frozenset((16, 17, 18))
# Locked-source assignment audit: these formulae initialise the suspended
# output and never write a nonzero value.  This is materially different from a
# computed channel that happens to evaluate to zero for one parameter/state.
_ZERO_SUSPENDED_SELECTORS = frozenset((2, 12, 14, 16, 17, 18))


class CapacityProfileError(ValueError):
    """Fail-closed profile error with a stable machine-readable category."""

    def __init__(self, message: str, *, category: str):
        super().__init__(message)
        self.category = category


@dataclass(frozen=True)
class CapacityFractionSelection:
    """Typed input for one sediment fraction.

    Formula identities are selected by ``method_id``.  Optional numeric
    identities can be supplied by readers of older decks; when present they
    are assertions and must match, never alternative ambiguous selectors.
    """

    fraction_id: str
    method_id: str
    parameters: Mapping[str, float | int]
    formula_code: int | None = None
    reference_selector: int | None = None


@dataclass(frozen=True)
class ResolvedCoefficient:
    keyword: str
    value: float | int


@dataclass(frozen=True)
class CapacityOutputChannels:
    """Common capacity ABI before face reconstruction and Exner update."""

    source_shape: str
    bed_channel: str
    suspended_channel: str
    bed_direction: str
    suspended_direction: str
    units: str = CAPACITY_UNITS


@dataclass(frozen=True)
class ResolvedCapacityFraction:
    fraction_id: str
    method_id: str
    formula_code: int
    reference_selector: int
    source_id: str
    coefficients: tuple[ResolvedCoefficient, ...]
    output: CapacityOutputChannels
    shear_consumer: str
    resolved_shear_source_method_id: str
    generic_hiding_supported: bool
    hiding_applied_to_capacity: bool
    hiding_applied_to_slope: bool
    secondary_steering_supported: bool
    secondary_steering_applied: bool
    compatible_hiding_method_ids: tuple[str, ...]
    compatible_bed_slope_method_ids: tuple[str, ...]
    compatible_secondary_method_ids: tuple[str, ...]


@dataclass(frozen=True)
class ResolvedMethod:
    method_id: str
    reference_selector: str | None
    persistence_code: int | str | None
    source_id: str


@dataclass(frozen=True)
class CapacityRuntimeProfile:
    schema: int
    scope_id: str
    semantic_id: str
    sediment_catalog_id: str
    parameter_catalog_id: str
    backend: str | None
    sediment_shear: ResolvedMethod
    hiding: ResolvedMethod
    bed_slope: ResolvedMethod
    secondary_flow: ResolvedMethod
    fractions: tuple[ResolvedCapacityFraction, ...]
    fingerprint: str

    def as_dict(self) -> dict:
        """Return a verified canonical JSON-compatible profile document."""
        return capacity_profile_document(self)


@dataclass(frozen=True)
class MethodChoiceView:
    """Small catalogue-backed view suitable for a QGIS combo-box model."""

    method_id: str
    label: str
    legacy_code: int | str | None
    reference_selector: str | None
    route_status: str
    admitted: bool


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise CapacityProfileError(
            "Capacity profile is not canonically JSON serializable",
            category="invalid_configuration",
        ) from error


def _backend_status(row: MethodSpec, backend: str) -> str:
    if backend == "cpu":
        return row.cpu_route
    if backend == "implicit_gpu":
        return row.implicit_gpu_route
    raise CapacityProfileError(
        "Backend must be 'cpu', 'implicit_gpu', or None",
        category="invalid_configuration",
    )


def _validate_backend(backend: str | None) -> None:
    if backend is not None and (isinstance(backend, bool) or backend not in _BACKENDS):
        raise CapacityProfileError(
            "Backend must be 'cpu', 'implicit_gpu', or None",
            category="invalid_configuration",
        )


def _require_route(row: MethodSpec, backend: str | None) -> None:
    if backend is None:
        return
    status = _backend_status(row, backend)
    if status not in ADMITTED_ROUTE_STATUSES:
        raise CapacityProfileError(
            f"{row.method_id} is not connected to the {backend} runtime "
            f"(catalogue route status: {status})",
            category="implementation_restriction",
        )


def _selected_method(method_id: str, component: str,
                     allowed: tuple[str, ...], backend: str | None) -> MethodSpec:
    if not isinstance(method_id, str) or not method_id:
        raise CapacityProfileError(
            f"{component} selection must be a non-empty method identity",
            category="invalid_configuration",
        )
    try:
        row = method(method_id)
    except ValueError as error:
        raise CapacityProfileError(str(error), category="unknown_method") from error
    if row.component != component:
        raise CapacityProfileError(
            f"{method_id} belongs to {row.component}, not {component}",
            category="invalid_configuration",
        )
    if method_id not in allowed:
        raise CapacityProfileError(
            f"{method_id} is outside the locked Delft3D-current capacity profile",
            category="scope_restriction",
        )
    if row.disposition in (SCOPE_EXCLUDED, SCOPE_REFERENCE_DISABLED):
        raise CapacityProfileError(
            f"{method_id} is outside the active river scope",
            category="scope_restriction",
        )
    _require_route(row, backend)
    return row


def _resolved_method(row: MethodSpec) -> ResolvedMethod:
    return ResolvedMethod(row.method_id, row.reference_selector,
                          row.legacy_code, row.source_id)


def _output_contract(source_shape: str, reference_selector: int) -> CapacityOutputChannels:
    suspended_zero = reference_selector in _ZERO_SUSPENDED_SELECTORS
    if source_shape == "split_vectors":
        return CapacityOutputChannels(
            source_shape, BED_CHANNEL_VECTOR,
            SUSPENDED_CHANNEL_ZERO if suspended_zero else SUSPENDED_CHANNEL_VECTOR,
            DIRECTION_VECTOR,
            "not_applicable" if suspended_zero else DIRECTION_VECTOR,
        )
    if source_shape in ("scalar_split", "intrinsic_mixture_scalar"):
        return CapacityOutputChannels(
            source_shape, BED_CHANNEL_SCALAR,
            SUSPENDED_CHANNEL_ZERO if suspended_zero else SUSPENDED_CHANNEL_SCALAR,
            DIRECTION_SCALAR,
            "not_applicable" if suspended_zero else "depth_averaged_flow_direction",
        )
    raise CapacityProfileError(
        f"Unknown capacity output shape: {source_shape!r}",
        category="catalogue_integrity",
    )


def _normal_coefficient(keyword: str, value: float | int) -> ResolvedCoefficient:
    # ``resolve_capacity_parameters`` has already enforced finite numeric
    # values and integer enums.  Canonicalize signed zero for stable identities.
    if isinstance(value, float) and value == 0.0:
        value = 0.0
    return ResolvedCoefficient(keyword, value)


def _method_semantics(row: MethodSpec) -> dict:
    return {
        "method_id": row.method_id,
        "reference_selector": row.reference_selector,
        "persistence_code": row.legacy_code,
        "source_id": row.source_id,
    }


def bed_slope_persistence_map() -> Mapping[int, str]:
    """Return all assigned BRAIDS slope codes without selector reinterpretation."""
    result: dict[int, str] = {}
    for code, row in legacy_code_map("bed_slope").items():
        if not isinstance(code, int) or isinstance(code, bool):
            raise CapacityProfileError(
                f"Bed-slope persistence code is not an integer: {code!r}",
                category="catalogue_integrity",
            )
        result[code] = row.method_id
    if result.get(GAIA_TALMON_BRAIDS_SLOPE_CODE) != "bed_slope.gaia_talmon":
        raise CapacityProfileError(
            "BRAIDS slope code 4 no longer identifies GAIA Talmon",
            category="catalogue_integrity",
        )
    if result.get(PARKER_ANDREWS_BRAIDS_SLOPE_CODE) != "bed_slope.delft_parker_andrews":
        raise CapacityProfileError(
            "Parker-Andrews must use unique BRAIDS slope code 6",
            category="catalogue_integrity",
        )
    return MappingProxyType(dict(sorted(result.items())))


def method_choice_views(component: str, *, backend: str,
                        include_out_of_scope: bool = False) -> tuple[MethodChoiceView, ...]:
    """Return catalogue choices without turning prototypes into admitted routes."""
    _validate_backend(backend)
    if backend is None:  # guarded above, retained for type checkers
        raise CapacityProfileError(
            "A backend is required for method choices",
            category="invalid_configuration",
        )
    if not isinstance(component, str) or not component:
        raise CapacityProfileError(
            "Component must be a non-empty string",
            category="invalid_configuration",
        )
    rows = methods_for_component(component)
    if not rows:
        raise CapacityProfileError(
            f"Unknown or empty sediment component: {component!r}",
            category="unknown_component",
        )
    result = []
    for row in rows:
        status = _backend_status(row, backend)
        if (not include_out_of_scope and
                row.disposition in (SCOPE_EXCLUDED, SCOPE_REFERENCE_DISABLED)):
            continue
        result.append(MethodChoiceView(
            row.method_id, row.label, row.legacy_code, row.reference_selector,
            status, status in ADMITTED_ROUTE_STATUSES,
        ))
    return tuple(result)


def capacity_profile_semantic_document() -> dict:
    """Return the static, restart-relevant meaning of this profile schema."""
    formulas = []
    for contract in sorted(FORMULA_CONTRACTS.values(), key=lambda row: row.method_id):
        catalogue_row = METHOD_BY_ID.get(contract.method_id)
        if catalogue_row is None:
            raise CapacityProfileError(
                f"Formula missing from sediment catalogue: {contract.method_id}",
                category="catalogue_integrity",
            )
        output = _output_contract(contract.output_shape, contract.reference_selector)
        formulas.append({
            "method": _method_semantics(catalogue_row),
            "formula_code": contract.legacy_code,
            "reference_selector": contract.reference_selector,
            "output": asdict(output),
            "shear_consumer": (
                SHEAR_SELECTED_RP_TAUB
                if contract.reference_selector in _RP_TAUB_SELECTORS
                else SHEAR_TOTAL_HYDRAULICS
            ),
            "generic_hiding": contract.generic_hiding,
            "secondary_steerable": contract.external_secondary_direction,
            "coefficient_keywords": [value.keyword for value in contract.parameters],
        })
    document = {
        "schema": CAPACITY_PROFILE_SCHEMA,
        "scope_id": CAPACITY_PROFILE_SCOPE_ID,
        "sediment_catalog_id": semantic_catalog_id(),
        "parameter_catalog_id": capacity_parameter_catalog_id(),
        "slope_persistence_ids": {
            str(code): method_id for code, method_id in bed_slope_persistence_map().items()
        },
        "allowed_methods": {
            "sediment_shear": list(_SHEAR_METHOD_IDS),
            "hiding": list(_HIDING_METHOD_IDS),
            "bed_slope": list(_DELFT_SLOPE_METHOD_IDS),
            "secondary_flow": list(_SECONDARY_METHOD_IDS),
        },
        "composition_rules": {
            "hiding": {
                "compatible_if_capacity_generic_hiding": True,
                "compatible_if_selected_bed_slope_method_ids": list(
                    _HIDING_SLOPE_CONSUMER_METHOD_IDS
                ),
                "capacity_application_requires_formula_generic_hiding": True,
                "slope_application_applies_to_every_fraction": True,
            },
        },
        "formulas": formulas,
    }
    return json.loads(_canonical_json(document))


def capacity_profile_semantic_id() -> str:
    return hashlib.sha256(
        _canonical_json(capacity_profile_semantic_document()).encode("ascii")
    ).hexdigest()


def capacity_diagnostic_contract_document() -> dict:
    """Return the v47 per-law diagnostic applicability contract.

    A numeric zero is not sufficient to distinguish a physically zero value
    from a quantity that a formula never defines.  V47 diagnostic rows carry
    an applicability mask; this document gives every bit a stable meaning and
    binds the formula-specific rules into deck and restart identity without
    changing the established v46 capacity-profile identity.
    """

    laws = []
    for formula_code in (11, 12, 13, 14):
        contract = FORMULA_BY_LEGACY_CODE[formula_code]
        laws.append({
            "formula_code": formula_code,
            "method_id": contract.method_id,
            "reference_selector": contract.reference_selector,
            "fields": {
                "absolute_shields": "defined",
                "critical_shields": "defined",
                "stress_ratio": "defined_when_critical_shields_positive",
                "transport_stage": (
                    "defined" if formula_code == 12 else "not_applicable_zero"
                ),
            },
        })
    document = {
        "schema": CAPACITY_DIAGNOSTIC_SCHEMA,
        "scope_id": CAPACITY_DIAGNOSTIC_SCOPE_ID,
        "row_mask_bits": {
            "absolute_shields": DIAGNOSTIC_ABSOLUTE_SHIELDS,
            "critical_shields": DIAGNOSTIC_CRITICAL_SHIELDS,
            "stress_ratio": DIAGNOSTIC_STRESS_RATIO,
            "transport_stage": DIAGNOSTIC_TRANSPORT_STAGE,
        },
        "not_applicable_numeric_value": 0.0,
        "laws": laws,
    }
    return json.loads(_canonical_json(document))


def capacity_diagnostic_contract_id() -> str:
    """Return the immutable SHA-256 identity of the v47 diagnostics ABI."""

    return hashlib.sha256(
        _canonical_json(capacity_diagnostic_contract_document()).encode("ascii")
    ).hexdigest()


def capacity_diagnostic_applicability_mask(
    formula_code: int, critical_shields: float
) -> int:
    """Return the v47 row mask for one evaluated capacity result."""

    if (isinstance(formula_code, bool) or formula_code not in (11, 12, 13, 14)):
        raise CapacityProfileError(
            f"No v47 diagnostic contract for formula code {formula_code!r}",
            category="scope_restriction",
        )
    if (isinstance(critical_shields, bool) or
            not isinstance(critical_shields, (int, float))):
        raise CapacityProfileError(
            "critical_shields must be a finite number",
            category="invalid_configuration",
        )
    value = float(critical_shields)
    if value != value or value in (float("inf"), float("-inf")):
        raise CapacityProfileError(
            "critical_shields must be a finite number",
            category="invalid_configuration",
        )
    mask = DIAGNOSTIC_ABSOLUTE_SHIELDS | DIAGNOSTIC_CRITICAL_SHIELDS
    if value > 0.0:
        mask |= DIAGNOSTIC_STRESS_RATIO
    if formula_code == 12:
        mask |= DIAGNOSTIC_TRANSPORT_STAGE
    return mask


def _profile_payload(profile: CapacityRuntimeProfile) -> dict:
    def method_payload(value: ResolvedMethod) -> dict:
        return asdict(value)

    fractions = []
    for value in profile.fractions:
        row = asdict(value)
        # JSON arrays are the canonical representation; do not expose tuple
        # implementation details to project/deck readers.
        fractions.append(row)
    return {
        "schema": profile.schema,
        "scope_id": profile.scope_id,
        "semantic_id": profile.semantic_id,
        "sediment_catalog_id": profile.sediment_catalog_id,
        "parameter_catalog_id": profile.parameter_catalog_id,
        "backend": profile.backend,
        "sediment_shear": method_payload(profile.sediment_shear),
        "hiding": method_payload(profile.hiding),
        "bed_slope": method_payload(profile.bed_slope),
        "secondary_flow": method_payload(profile.secondary_flow),
        "fractions": fractions,
    }


def _payload_fingerprint(payload: dict) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("ascii")).hexdigest()


def _rebuild_profile(profile: CapacityRuntimeProfile) -> CapacityRuntimeProfile:
    try:
        selections = []
        for fraction in profile.fractions:
            parameters = {}
            for coefficient in fraction.coefficients:
                if coefficient.keyword in parameters:
                    raise ValueError("duplicate coefficient")
                parameters[coefficient.keyword] = coefficient.value
            selections.append(CapacityFractionSelection(
                fraction.fraction_id, fraction.method_id, parameters,
                fraction.formula_code, fraction.reference_selector,
            ))
        return resolve_capacity_profile(
            selections,
            sediment_shear=profile.sediment_shear.method_id,
            hiding=profile.hiding.method_id,
            bed_slope=profile.bed_slope.method_id,
            secondary_flow=profile.secondary_flow.method_id,
            backend=profile.backend,
        )
    except CapacityProfileError:
        raise
    except (AttributeError, TypeError, ValueError) as error:
        raise CapacityProfileError(
            "Malformed in-memory capacity profile",
            category="invalid_configuration",
        ) from error


def capacity_profile_document(profile: CapacityRuntimeProfile) -> dict:
    """Return the canonical instance document, rejecting forged/stale objects."""
    if not isinstance(profile, CapacityRuntimeProfile):
        raise CapacityProfileError(
            "Expected a CapacityRuntimeProfile",
            category="invalid_configuration",
        )
    if (profile.schema != CAPACITY_PROFILE_SCHEMA or
            profile.scope_id != CAPACITY_PROFILE_SCOPE_ID or
            profile.semantic_id != capacity_profile_semantic_id() or
            profile.sediment_catalog_id != semantic_catalog_id() or
            profile.parameter_catalog_id != capacity_parameter_catalog_id()):
        raise CapacityProfileError(
            "Capacity profile schema or semantic identity is stale",
            category="identity_mismatch",
        )
    payload = _profile_payload(profile)
    rebuilt = _rebuild_profile(profile)
    if _canonical_json(payload) != _canonical_json(_profile_payload(rebuilt)):
        raise CapacityProfileError(
            "Capacity profile contents do not match the resolved contract",
            category="identity_mismatch",
        )
    expected = _payload_fingerprint(payload)
    if profile.fingerprint != expected:
        raise CapacityProfileError(
            "Capacity profile fingerprint does not match its contents",
            category="identity_mismatch",
        )
    document = dict(payload)
    document["fingerprint"] = expected
    return json.loads(_canonical_json(document))


def capacity_profile_json(profile: CapacityRuntimeProfile, *, indent: int | None = 2) -> str:
    document = capacity_profile_document(profile)
    return json.dumps(document, sort_keys=True, indent=indent,
                      ensure_ascii=True, allow_nan=False) + ("\n" if indent is not None else "")


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise CapacityProfileError(
                f"Duplicate JSON key in capacity profile: {key!r}",
                category="invalid_serialization",
            )
        result[key] = value
    return result


def load_capacity_profile(document: str | bytes | Mapping[str, object]) -> CapacityRuntimeProfile:
    """Rebuild and verify a serialized profile instead of trusting derived fields.

    Every derived contract flag, identity and coefficient order must match a
    fresh resolution under the current catalogues.  Extra keys, duplicate JSON
    keys and stale fingerprints are rejected rather than ignored.
    """
    if isinstance(document, (str, bytes)):
        try:
            raw = json.loads(document, object_pairs_hook=_strict_json_object,
                             parse_constant=lambda value: (_ for _ in ()).throw(
                                 CapacityProfileError(
                                     f"Non-finite JSON number: {value}",
                                     category="invalid_serialization")))
        except CapacityProfileError:
            raise
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise CapacityProfileError(
                "Invalid capacity-profile JSON",
                category="invalid_serialization",
            ) from error
    elif isinstance(document, Mapping):
        try:
            raw = json.loads(_canonical_json(document))
        except CapacityProfileError as error:
            raise CapacityProfileError(
                str(error), category="invalid_serialization") from error
    else:
        raise CapacityProfileError(
            "Serialized capacity profile must be JSON text or a mapping",
            category="invalid_serialization",
        )
    if not isinstance(raw, dict):
        raise CapacityProfileError(
            "Serialized capacity profile must be a JSON object",
            category="invalid_serialization",
        )

    try:
        fraction_rows = raw["fractions"]
        if not isinstance(fraction_rows, list):
            raise TypeError("fractions")
        selections = []
        for row in fraction_rows:
            if not isinstance(row, dict) or not isinstance(row["coefficients"], list):
                raise TypeError("fraction")
            parameters = {}
            for coefficient in row["coefficients"]:
                if not isinstance(coefficient, dict):
                    raise TypeError("coefficient")
                keyword = coefficient["keyword"]
                if not isinstance(keyword, str) or keyword in parameters:
                    raise ValueError("coefficient keyword")
                parameters[keyword] = coefficient["value"]
            selections.append(CapacityFractionSelection(
                fraction_id=row["fraction_id"],
                method_id=row["method_id"],
                parameters=parameters,
                formula_code=row["formula_code"],
                reference_selector=row["reference_selector"],
            ))
        rebuilt = resolve_capacity_profile(
            selections,
            sediment_shear=raw["sediment_shear"]["method_id"],
            hiding=raw["hiding"]["method_id"],
            bed_slope=raw["bed_slope"]["method_id"],
            secondary_flow=raw["secondary_flow"]["method_id"],
            backend=raw["backend"],
        )
    except CapacityProfileError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise CapacityProfileError(
            "Malformed capacity-profile document",
            category="invalid_serialization",
        ) from error

    expected = capacity_profile_document(rebuilt)
    if _canonical_json(raw) != _canonical_json(expected):
        category = (
            "identity_mismatch"
            if raw.get("fingerprint") != expected["fingerprint"] or
            raw.get("semantic_id") != expected["semantic_id"] or
            raw.get("sediment_catalog_id") != expected["sediment_catalog_id"] or
            raw.get("parameter_catalog_id") != expected["parameter_catalog_id"]
            else "invalid_serialization"
        )
        raise CapacityProfileError(
            "Serialized capacity profile is stale, forged, or non-canonical",
            category=category,
        )
    return rebuilt


def resolve_capacity_profile(
    selections: Iterable[CapacityFractionSelection],
    *,
    sediment_shear: str = "shear.total_roughness",
    hiding: str = "hiding.none",
    bed_slope: str = "bed_slope.off",
    secondary_flow: str = "secondary.off",
    backend: str | None = None,
) -> CapacityRuntimeProfile:
    """Resolve and fingerprint a multi-fraction current-capacity profile.

    ``backend=None`` resolves physical semantics for preparation/migration.
    A real runtime must provide ``cpu`` or ``implicit_gpu``; prototype and
    missing routes then fail closed until their catalogue admission changes.
    """
    _validate_backend(backend)
    if isinstance(selections, (str, bytes)):
        raise CapacityProfileError(
            "Capacity fractions must be an iterable of typed selections",
            category="invalid_configuration",
        )
    try:
        requested = tuple(selections)
    except TypeError as error:
        raise CapacityProfileError(
            "Capacity fractions must be iterable",
            category="invalid_configuration",
        ) from error
    if not requested:
        raise CapacityProfileError(
            "At least one active sediment fraction is required",
            category="invalid_configuration",
        )
    if any(not isinstance(value, CapacityFractionSelection) for value in requested):
        raise CapacityProfileError(
            "Every capacity fraction must be a CapacityFractionSelection",
            category="invalid_configuration",
        )

    shear_row = _selected_method(sediment_shear, "sediment_shear",
                                 _SHEAR_METHOD_IDS, backend)
    hiding_row = _selected_method(hiding, "hiding", _HIDING_METHOD_IDS, backend)
    slope_row = _selected_method(bed_slope, "bed_slope",
                                 _DELFT_SLOPE_METHOD_IDS, backend)
    secondary_row = _selected_method(secondary_flow, "secondary_flow",
                                     _SECONDARY_METHOD_IDS, backend)

    ids: set[str] = set()
    preliminary: list[tuple[CapacityFractionSelection, CapacityFormulaContract, MethodSpec,
                            tuple[ResolvedCoefficient, ...], CapacityOutputChannels]] = []
    for selection in requested:
        fraction_id = selection.fraction_id
        if (not isinstance(fraction_id, str) or not fraction_id or
                fraction_id != fraction_id.strip()):
            raise CapacityProfileError(
                "fraction_id must be a non-empty, already-trimmed string",
                category="invalid_configuration",
            )
        if fraction_id in ids:
            raise CapacityProfileError(
                f"Duplicate fraction identity: {fraction_id!r}",
                category="invalid_configuration",
            )
        ids.add(fraction_id)
        if not isinstance(selection.method_id, str):
            raise CapacityProfileError(
                f"Fraction {fraction_id!r} has no formula method identity",
                category="invalid_configuration",
            )
        try:
            contract = FORMULA_CONTRACTS[selection.method_id]
        except KeyError as error:
            raise CapacityProfileError(
                f"{selection.method_id!r} is not a registered locked "
                "Delft3D-current capacity formula",
                category="scope_restriction",
            ) from error
        catalogue_row = method(contract.method_id)
        if catalogue_row.component != "capacity":
            raise CapacityProfileError(
                f"{contract.method_id} is not catalogued as a capacity formula",
                category="catalogue_integrity",
            )
        if (catalogue_row.legacy_code not in (None, contract.legacy_code) or
                catalogue_row.reference_selector not in (
                    None, f"TraFrm={contract.reference_selector}")):
            raise CapacityProfileError(
                f"Formula identities disagree for {contract.method_id}",
                category="catalogue_integrity",
            )
        _require_route(catalogue_row, backend)
        if (selection.formula_code is not None and
                (isinstance(selection.formula_code, bool) or
                 not isinstance(selection.formula_code, int) or
                 selection.formula_code != contract.legacy_code)):
            raise CapacityProfileError(
                f"Fraction {fraction_id!r} formula code does not match "
                f"{contract.method_id} (expected {contract.legacy_code})",
                category="identity_mismatch",
            )
        if (selection.reference_selector is not None and
                (isinstance(selection.reference_selector, bool) or
                 not isinstance(selection.reference_selector, int) or
                 selection.reference_selector != contract.reference_selector)):
            raise CapacityProfileError(
                f"Fraction {fraction_id!r} Delft3D selector does not match "
                f"{contract.method_id} (expected {contract.reference_selector})",
                category="identity_mismatch",
            )
        try:
            items = canonical_parameter_items(contract.method_id, selection.parameters)
        except CapacityParameterError as error:
            raise CapacityProfileError(
                f"Fraction {fraction_id!r}: {error}",
                category="invalid_formula_parameters",
            ) from error
        coefficients = tuple(_normal_coefficient(key, value) for key, value in items)
        preliminary.append((selection, contract, catalogue_row, coefficients,
                            _output_contract(contract.output_shape,
                                             contract.reference_selector)))

    uses_rp_taub = any(
        contract.reference_selector in _RP_TAUB_SELECTORS
        for _, contract, _, _, _ in preliminary
    )
    uses_generic_hiding = any(
        contract.generic_hiding for _, contract, _, _, _ in preliminary
    )
    uses_scalar_direction = any(
        contract.external_secondary_direction
        for _, contract, _, _, _ in preliminary
    )
    parker = slope_row.method_id in _HIDING_SLOPE_CONSUMER_METHOD_IDS
    if shear_row.method_id == "shear.soulsby_clarke_skin" and not uses_rp_taub:
        raise CapacityProfileError(
            "Soulsby/Clarke skin shear was selected but no fraction consumes RP_TAUB",
            category="physical_incompatibility",
        )
    if hiding_row.method_id != "hiding.none" and not (uses_generic_hiding or parker):
        raise CapacityProfileError(
            "The selected hiding method has no capacity or Parker-Andrews consumer",
            category="physical_incompatibility",
        )
    if secondary_row.method_id != "secondary.off" and not uses_scalar_direction:
        raise CapacityProfileError(
            "Secondary-flow steering was selected but every formula returns direct vectors",
            category="physical_incompatibility",
        )

    resolved: list[ResolvedCapacityFraction] = []
    for selection, contract, catalogue_row, coefficients, output in preliminary:
        rp_taub = contract.reference_selector in _RP_TAUB_SELECTORS
        scalar = contract.external_secondary_direction
        hiding_to_capacity = (
            hiding_row.method_id != "hiding.none" and contract.generic_hiding
        )
        hiding_to_slope = hiding_row.method_id != "hiding.none" and parker
        resolved.append(ResolvedCapacityFraction(
            fraction_id=selection.fraction_id,
            method_id=contract.method_id,
            formula_code=contract.legacy_code,
            reference_selector=contract.reference_selector,
            source_id=catalogue_row.source_id,
            coefficients=coefficients,
            output=output,
            shear_consumer=(SHEAR_SELECTED_RP_TAUB if rp_taub
                            else SHEAR_TOTAL_HYDRAULICS),
            resolved_shear_source_method_id=(
                shear_row.method_id if rp_taub else "shear.total_roughness"
            ),
            generic_hiding_supported=contract.generic_hiding,
            hiding_applied_to_capacity=hiding_to_capacity,
            hiding_applied_to_slope=hiding_to_slope,
            secondary_steering_supported=scalar,
            secondary_steering_applied=(
                scalar and secondary_row.method_id != "secondary.off"
            ),
            compatible_hiding_method_ids=(
                _HIDING_METHOD_IDS if (contract.generic_hiding or parker)
                else ("hiding.none",)
            ),
            compatible_bed_slope_method_ids=_DELFT_SLOPE_METHOD_IDS,
            compatible_secondary_method_ids=(
                _SECONDARY_METHOD_IDS if scalar else ("secondary.off",)
            ),
        ))
    placeholder = CapacityRuntimeProfile(
        schema=CAPACITY_PROFILE_SCHEMA,
        scope_id=CAPACITY_PROFILE_SCOPE_ID,
        semantic_id=capacity_profile_semantic_id(),
        sediment_catalog_id=semantic_catalog_id(),
        parameter_catalog_id=capacity_parameter_catalog_id(),
        backend=backend,
        sediment_shear=_resolved_method(shear_row),
        hiding=_resolved_method(hiding_row),
        bed_slope=_resolved_method(slope_row),
        secondary_flow=_resolved_method(secondary_row),
        fractions=tuple(resolved),
        fingerprint="",
    )
    fingerprint = _payload_fingerprint(_profile_payload(placeholder))
    return CapacityRuntimeProfile(
        placeholder.schema, placeholder.scope_id, placeholder.semantic_id,
        placeholder.sediment_catalog_id, placeholder.parameter_catalog_id,
        placeholder.backend, placeholder.sediment_shear, placeholder.hiding,
        placeholder.bed_slope, placeholder.secondary_flow,
        placeholder.fractions, fingerprint,
    )


def resolve_capacity_profile_from_code(
    formula_code: int,
    fraction_ids: Iterable[str],
    parameters: (
        Mapping[str, float | int] |
        Mapping[str, Mapping[str, float | int]]
    ),
    *,
    sediment_shear: str = "shear.total_roughness",
    hiding: str = "hiding.none",
    bed_slope: str = "bed_slope.off",
    secondary_flow: str = "secondary.off",
    backend: str | None = None,
    method_id: str | None = None,
) -> CapacityRuntimeProfile:
    """Resolve a project/deck formula code over an ordered fraction layout.

    A single-fraction caller may pass the formula's source-keyword mapping
    directly or ``{fraction_id: mapping}``.  Multi-fraction callers must pass
    an exact mapping for every ID.  This keeps parameter completion, numeric
    validation and identity assertions in one place for project/deck readers.
    """
    if (isinstance(formula_code, bool) or
            not isinstance(formula_code, int)):
        raise CapacityProfileError(
            "BRAIDS capacity formula code must be an integer",
            category="invalid_configuration",
        )
    try:
        if method_id is None:
            # Integer-only laws 2 and 8 are historical decks. Their resolver
            # intentionally selects the legacy closure; a corrected Delft
            # route must carry its explicit method identity.
            contract = FORMULA_BY_LEGACY_CODE[formula_code]
        else:
            if not isinstance(method_id, str):
                raise KeyError(method_id)
            contract = FORMULA_CONTRACTS[method_id]
            if contract.legacy_code != formula_code:
                raise CapacityProfileError(
                    "Capacity method identity conflicts with the native formula code",
                    category="identity_mismatch",
                )
    except KeyError as error:
        raise CapacityProfileError(
            f"Unknown active BRAIDS capacity formula code or method: "
            f"{formula_code!r}, {method_id!r}",
            category="scope_restriction",
        ) from error
    if isinstance(fraction_ids, (str, bytes)):
        raise CapacityProfileError(
            "fraction_ids must be an ordered iterable of identities",
            category="invalid_configuration",
        )
    try:
        ordered_ids = tuple(fraction_ids)
    except TypeError as error:
        raise CapacityProfileError(
            "fraction_ids must be iterable",
            category="invalid_configuration",
        ) from error
    if not ordered_ids:
        raise CapacityProfileError(
            "At least one fraction identity is required",
            category="invalid_configuration",
        )
    if (any(not isinstance(value, str) or not value or value != value.strip()
            for value in ordered_ids) or len(set(ordered_ids)) != len(ordered_ids)):
        raise CapacityProfileError(
            "Fraction identities must be unique, non-empty, already-trimmed strings",
            category="invalid_configuration",
        )
    if not isinstance(parameters, Mapping):
        raise CapacityProfileError(
            "Capacity parameters must be a mapping",
            category="invalid_configuration",
        )

    expected = set(ordered_ids)
    nested = (
        len(ordered_ids) > 1 or
        any(isinstance(value, Mapping) for value in parameters.values())
    )
    parameter_maps: dict[str, Mapping[str, float | int]] = {}
    if nested:
        supplied = set(parameters)
        missing = tuple(sorted(expected - supplied))
        extra = tuple(sorted(supplied - expected, key=str))
        if missing or extra:
            details = []
            if missing:
                details.append("missing " + ", ".join(repr(value) for value in missing))
            if extra:
                details.append("extra " + ", ".join(repr(value) for value in extra))
            raise CapacityProfileError(
                "Fraction parameter mapping has " + " and ".join(details),
                category="invalid_configuration",
            )
        for fraction_id in ordered_ids:
            value = parameters[fraction_id]
            if not isinstance(value, Mapping):
                raise CapacityProfileError(
                    f"Parameters for fraction {fraction_id!r} must be a mapping",
                    category="invalid_configuration",
                )
            parameter_maps[fraction_id] = value
    else:
        # The only unambiguous non-nested form is a direct map for one fraction.
        parameter_maps[ordered_ids[0]] = parameters

    selections = tuple(
        CapacityFractionSelection(
            fraction_id=fraction_id,
            method_id=contract.method_id,
            parameters=parameter_maps[fraction_id],
            formula_code=contract.legacy_code,
            reference_selector=contract.reference_selector,
        )
        for fraction_id in ordered_ids
    )
    return resolve_capacity_profile(
        selections,
        sediment_shear=sediment_shear,
        hiding=hiding,
        bed_slope=bed_slope,
        secondary_flow=secondary_flow,
        backend=backend,
    )


__all__ = [
    "CAPACITY_PROFILE_SCHEMA",
    "CAPACITY_PROFILE_SCOPE_ID",
    "PARKER_ANDREWS_BRAIDS_SLOPE_CODE",
    "GAIA_TALMON_BRAIDS_SLOPE_CODE",
    "BED_CHANNEL_SCALAR",
    "BED_CHANNEL_VECTOR",
    "SUSPENDED_CHANNEL_SCALAR",
    "SUSPENDED_CHANNEL_VECTOR",
    "SUSPENDED_CHANNEL_ZERO",
    "DIRECTION_SCALAR",
    "DIRECTION_VECTOR",
    "SHEAR_TOTAL_HYDRAULICS",
    "SHEAR_SELECTED_RP_TAUB",
    "CAPACITY_UNITS",
    "CAPACITY_DIAGNOSTIC_SCHEMA",
    "CAPACITY_DIAGNOSTIC_SCOPE_ID",
    "DIAGNOSTIC_ABSOLUTE_SHIELDS",
    "DIAGNOSTIC_CRITICAL_SHIELDS",
    "DIAGNOSTIC_STRESS_RATIO",
    "DIAGNOSTIC_TRANSPORT_STAGE",
    "CapacityProfileError",
    "CapacityFractionSelection",
    "ResolvedCoefficient",
    "CapacityOutputChannels",
    "ResolvedCapacityFraction",
    "ResolvedMethod",
    "CapacityRuntimeProfile",
    "MethodChoiceView",
    "bed_slope_persistence_map",
    "method_choice_views",
    "capacity_profile_semantic_document",
    "capacity_profile_semantic_id",
    "capacity_diagnostic_contract_document",
    "capacity_diagnostic_contract_id",
    "capacity_diagnostic_applicability_mask",
    "capacity_profile_document",
    "capacity_profile_json",
    "load_capacity_profile",
    "resolve_capacity_profile",
    "resolve_capacity_profile_from_code",
]
