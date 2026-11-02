"""Strict execution identities for the two historically ambiguous capacities.

The formula selector alone cannot identify the physics of BRAIDS law 8 or
law 2.  Historical law 8 used an independent, constant Chezy coefficient,
whereas Delft3D FLOW ``TraFrm=1`` consumes ``RP_CHEZY`` resolved from the
selected hydraulic roughness.  Historical law 2 used a user-specified
friction coefficient, whereas the Delft3D MPM route consumes Chezy resolved
from the selected hydraulic C/M/W state plus generic hiding where selected.
MPM never consumes ``RP_TAUB`` or the Soulsby/Clarke skin-shear replacement.

This module is a zero-step project/deck/restart contract.  It deliberately
does not dispatch a numerical kernel or grant backend support.  Older readers
that have only the integer law and the historical controls are migrated to an
explicit legacy identity; they are never reinterpreted as a Delft3D route.
Corrected Delft identities require the authenticated typed-roughness binding,
including the canonical C/n/ks parameter name, units and array hash.  The
legacy native storage name ``manning_cell`` is recorded only as storage
provenance and is never used to infer a Manning physical type.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
from types import MappingProxyType
from typing import Mapping


CAPACITY_EXECUTION_SCHEMA = 1
CAPACITY_EXECUTION_SCOPE_ID = (
    "braids-delft3d4-capacity-execution-identity-v1"
)
FORCING_BINDING_SCHEMA = 2

EH_LEGACY_METHOD_ID = "capacity.engelund_hansen_constant_chezy_legacy"
EH_DELFT_METHOD_ID = "capacity.engelund_hansen"
MPM_LEGACY_METHOD_ID = "capacity.mpm_specified_cf_legacy"
MPM_DELFT_METHOD_ID = "capacity.mpm"

SELECTION_EXPLICIT = "explicit"
SELECTION_LEGACY_MIGRATION = "legacy_missing_method_identity"

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class CapacityExecutionParameter:
    """One canonical, unit-bearing route parameter."""

    name: str
    source_field: str
    units: str
    value: float


@dataclass(frozen=True)
class TypedRoughnessDependency:
    """Authenticated typed C/n/ks dependency consumed by a Delft route."""

    forcing_schema: int
    forcing_fingerprint: str
    roughness_method_id: str
    parameter_name: str
    parameter_units: str
    storage_field: str
    array_shape: tuple[int, ...]
    array_dtype: str
    array_sha256: str
    white_colebrook_chzk2d: bool
    sediment_shear_method_id: str


@dataclass(frozen=True)
class CapacityExecutionIdentity:
    """Canonical instance identity suitable for project/deck/restart binding."""

    schema: int
    scope_id: str
    semantic_id: str
    method_id: str
    formula_code: int
    reference_selector: int
    source_id: str
    closure: str
    selection_origin: str
    shear_consumer: str
    generic_hiding_consumer: bool
    parameters: tuple[CapacityExecutionParameter, ...]
    typed_roughness: TypedRoughnessDependency | None
    fingerprint: str

    def as_dict(self, *, forcing_binding: Mapping[str, object] | None = None) -> dict:
        return capacity_execution_document(self, forcing_binding=forcing_binding)


@dataclass(frozen=True)
class _MethodContract:
    method_id: str
    formula_code: int
    reference_selector: int
    source_id: str
    closure: str
    shear_consumer: str
    generic_hiding_consumer: bool
    parameter_specs: tuple[tuple[str, str, str, float | None, bool], ...]
    requires_typed_roughness: bool


def _c(method_id: str, formula_code: int, reference_selector: int,
       source_id: str, closure: str, shear_consumer: str,
       parameter_specs: tuple[tuple[str, str, str, float | None, bool], ...] = (),
       *, requires_typed_roughness: bool = False,
       generic_hiding_consumer: bool = False) -> _MethodContract:
    return _MethodContract(
        method_id, formula_code, reference_selector, source_id, closure,
        shear_consumer, generic_hiding_consumer, parameter_specs,
        requires_typed_roughness,
    )


_METHODS = (
    _c(
        EH_LEGACY_METHOD_ID, 8, 1,
        "braids_engelund_hansen_independent_constant_chezy",
        "independent_constant_chezy", "constant_chezy_not_hydraulic_state",
        (("constant_chezy_C", "engelund_hansen_chezy", "m^0.5/s", 28.8, True),),
    ),
    _c(
        EH_DELFT_METHOD_ID, 8, 1,
        "d3d4_engelund_hansen_resolved_hydraulic_chezy",
        "resolved_hydraulic_chezy", "resolved_hydraulic_chezy",
        requires_typed_roughness=True,
    ),
    _c(
        MPM_LEGACY_METHOD_ID, 2, 2,
        "braids_zhu_mpm_specified_friction",
        "specified_friction_coefficient", "specified_cf",
        (
            ("specified_friction_coefficient_cf", "mpm_cf", "1", 0.01, False),
            ("critical_shields", "mpm_critical_shields", "1", 0.0, False),
        ),
        generic_hiding_consumer=True,
    ),
    _c(
        MPM_DELFT_METHOD_ID, 2, 2,
        "d3d4_mpm_tranb2_resolved_hydraulic_roughness",
        "resolved_hydraulic_chezy", "total_hydraulic_state",
        (("critical_shields", "mpm_critical_shields", "1", 0.047, False),),
        requires_typed_roughness=True, generic_hiding_consumer=True,
    ),
)

METHOD_CONTRACTS = MappingProxyType({row.method_id: row for row in _METHODS})

_ROUGHNESS_PARAMETERS = MappingProxyType({
    "roughness.chezy": ("chezy_C", "m^0.5/s"),
    "roughness.manning": ("manning_n", "s/m^(1/3)"),
    "roughness.white_colebrook": ("nikuradse_ks", "m"),
    "roughness.white_colebrook_waqua": ("nikuradse_ks", "m"),
})
_ROUGHNESS_NATIVE = MappingProxyType({
    "roughness.chezy": ("chezy", 1, False),
    "roughness.manning": ("manning", 2, False),
    "roughness.white_colebrook": ("white_colebrook", 3, False),
    "roughness.white_colebrook_waqua": ("white_colebrook", 3, True),
})
_SHEAR_METHODS = frozenset((
    "shear.total_roughness",
    "shear.soulsby_clarke_skin",
))
_SHEAR_NATIVE = MappingProxyType({
    "shear.total_roughness": ("total", 0, False),
    "shear.soulsby_clarke_skin": ("soulsby_clarke_skin", 1, True),
})
_SKIN_FIELDS = frozenset((
    "kinematic_viscosity_m2_s", "sand_roughness_m", "silt_roughness_m",
    "critical_mud_fraction_low", "critical_mud_fraction_high",
    "fixed_fluff_fraction",
))

_FORCING_FIELDS = frozenset((
    "schema", "roughness_method_id", "hydraulic_roughness_mode",
    "native_hydraulic_roughness_mode", "white_colebrook_chzk2d",
    "roughness_parameter_name", "roughness_parameter_units",
    "roughness_storage_field", "roughness_default_parameter",
    "roughness_parameter_minimum", "roughness_parameter_maximum",
    "roughness_parameter_array", "roughness_minimum_work_depth_m",
    "friction_minimum_work_depth_m", "effective_minimum_work_depth_m",
    "sediment_shear_method_id", "sediment_shear_mode",
    "native_sediment_shear_mode", "skin_shear_controls",
    "skin_shear_controls_active", "hydraulic_friction_unchanged_by_skin",
    "fingerprint",
))


class CapacityExecutionIdentityError(ValueError):
    """Fail-closed execution-identity error with a stable category."""

    def __init__(self, message: str, *, category: str):
        super().__init__(message)
        self.category = category


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise CapacityExecutionIdentityError(
            "Capacity execution identity is not canonical JSON",
            category="invalid_serialization",
        ) from error


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def capacity_execution_semantic_document() -> dict:
    """Return the immutable meanings, parameters, units and defaults."""
    return {
        "schema": CAPACITY_EXECUTION_SCHEMA,
        "scope_id": CAPACITY_EXECUTION_SCOPE_ID,
        "forcing_binding_schema": FORCING_BINDING_SCHEMA,
        "methods": [
            {
                "method_id": row.method_id,
                "formula_code": row.formula_code,
                "reference_selector": row.reference_selector,
                "source_id": row.source_id,
                "closure": row.closure,
                "shear_consumer": row.shear_consumer,
                "generic_hiding_consumer": row.generic_hiding_consumer,
                "requires_typed_roughness": row.requires_typed_roughness,
                "parameters": [
                    {
                        "name": name,
                        "source_field": source,
                        "units": units,
                        "default": default,
                        "positive": positive,
                    }
                    for name, source, units, default, positive
                    in row.parameter_specs
                ],
            }
            for row in sorted(_METHODS, key=lambda item: item.method_id)
        ],
        "legacy_migration": {
            "law_8_without_method_id": EH_LEGACY_METHOD_ID,
            "law_2_mode_0_or_missing_without_method_id": MPM_LEGACY_METHOD_ID,
            "law_2_mode_1_without_method_id": MPM_DELFT_METHOD_ID,
        },
        "typed_roughness_parameters": {
            method_id: {"name": value[0], "units": value[1]}
            for method_id, value in sorted(_ROUGHNESS_PARAMETERS.items())
        },
        "storage_field_is_not_physical_type": True,
        "mpm_skin_shear_consumed": False,
        "mpm_hydraulic_input": (
            "Chezy resolved from selected hydraulic C/M/W roughness"
        ),
        "mpm_generic_hiding_consumed": True,
    }


def capacity_execution_semantic_id() -> str:
    return _hash(capacity_execution_semantic_document())


def _finite(value: object, name: str, *, positive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CapacityExecutionIdentityError(
            f"{name} must be a finite numeric value",
            category="invalid_parameters",
        )
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0) or (
            not positive and result < 0.0):
        relation = "positive" if positive else "nonnegative"
        raise CapacityExecutionIdentityError(
            f"{name} must be {relation} and finite",
            category="invalid_parameters",
        )
    return 0.0 if result == 0.0 else result


def _typed_roughness_dependency(
    forcing_binding: Mapping[str, object],
) -> TypedRoughnessDependency:
    if not isinstance(forcing_binding, Mapping):
        raise CapacityExecutionIdentityError(
            "A corrected Delft capacity requires a typed roughness binding",
            category="missing_dependency",
        )
    if set(forcing_binding) != _FORCING_FIELDS:
        raise CapacityExecutionIdentityError(
            "Typed roughness binding fields differ from schema 2",
            category="invalid_dependency",
        )
    if forcing_binding.get("schema") != FORCING_BINDING_SCHEMA:
        raise CapacityExecutionIdentityError(
            "Typed roughness binding schema differs",
            category="identity_mismatch",
        )
    body = {key: forcing_binding[key] for key in forcing_binding
            if key != "fingerprint"}
    fingerprint = forcing_binding.get("fingerprint")
    if not isinstance(fingerprint, str) or not _SHA256.fullmatch(fingerprint) or (
            fingerprint != _hash(body)):
        raise CapacityExecutionIdentityError(
            "Typed roughness binding fingerprint differs",
            category="identity_mismatch",
        )
    method_id = forcing_binding.get("roughness_method_id")
    try:
        parameter_name, parameter_units = _ROUGHNESS_PARAMETERS[method_id]
    except (KeyError, TypeError) as error:
        raise CapacityExecutionIdentityError(
            "Unknown typed hydraulic roughness method",
            category="invalid_dependency",
        ) from error
    if (forcing_binding.get("roughness_parameter_name") != parameter_name or
            forcing_binding.get("roughness_parameter_units") != parameter_units):
        raise CapacityExecutionIdentityError(
            "Typed roughness parameter name or units differ from its method",
            category="identity_mismatch",
        )
    expected_mode, expected_native, expected_chzk2d = _ROUGHNESS_NATIVE[method_id]
    chzk2d = forcing_binding.get("white_colebrook_chzk2d")
    if (type(chzk2d) is not bool or chzk2d is not expected_chzk2d or
            forcing_binding.get("hydraulic_roughness_mode") != expected_mode or
            forcing_binding.get("native_hydraulic_roughness_mode") != expected_native):
        raise CapacityExecutionIdentityError(
            "Hydraulic roughness mode or White-Colebrook branch identity differs",
            category="identity_mismatch",
        )
    shear = forcing_binding.get("sediment_shear_method_id")
    if shear not in _SHEAR_METHODS:
        raise CapacityExecutionIdentityError(
            "Unknown sediment-shear identity in forcing binding",
            category="invalid_dependency",
        )
    expected_shear_mode, expected_shear_native, expected_skin_active = (
        _SHEAR_NATIVE[shear]
    )
    if (forcing_binding.get("sediment_shear_mode") != expected_shear_mode or
            forcing_binding.get("native_sediment_shear_mode") != expected_shear_native or
            forcing_binding.get("skin_shear_controls_active") is not expected_skin_active or
            forcing_binding.get("hydraulic_friction_unchanged_by_skin") is not True):
        raise CapacityExecutionIdentityError(
            "Sediment-shear semantic tuple differs",
            category="identity_mismatch",
        )
    array = forcing_binding.get("roughness_parameter_array")
    if not isinstance(array, Mapping) or set(array) != {"shape", "dtype", "sha256"}:
        raise CapacityExecutionIdentityError(
            "Typed roughness array identity is malformed",
            category="invalid_dependency",
        )
    shape = array.get("shape")
    dtype = array.get("dtype")
    sha = array.get("sha256")
    if (not isinstance(shape, list) or not shape or
            any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in shape) or
            not isinstance(dtype, str) or not re.fullmatch(r"[<>=]f8", dtype) or
            not isinstance(sha, str) or not _SHA256.fullmatch(sha)):
        raise CapacityExecutionIdentityError(
            "Typed roughness array shape, dtype or SHA-256 is invalid",
            category="invalid_dependency",
        )
    storage = forcing_binding.get("roughness_storage_field")
    if storage != "manning_cell":
        raise CapacityExecutionIdentityError(
            "Typed roughness storage provenance is invalid",
            category="invalid_dependency",
        )
    numeric_fields = (
        "roughness_default_parameter", "roughness_parameter_minimum",
        "roughness_parameter_maximum", "roughness_minimum_work_depth_m",
        "friction_minimum_work_depth_m", "effective_minimum_work_depth_m",
    )
    numeric = {}
    for key in numeric_fields:
        value = forcing_binding.get(key)
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                not math.isfinite(float(value)) or float(value) < 0.0):
            raise CapacityExecutionIdentityError(
                f"Typed roughness numeric field is invalid: {key}",
                category="invalid_dependency",
            )
        numeric[key] = float(value)
    if (numeric["roughness_parameter_minimum"] >
            numeric["roughness_parameter_maximum"] or
            numeric["effective_minimum_work_depth_m"] != max(
                numeric["roughness_minimum_work_depth_m"],
                numeric["friction_minimum_work_depth_m"]) or
            (method_id != "roughness.manning" and
             (numeric["roughness_default_parameter"] <= 0.0 or
              numeric["roughness_parameter_minimum"] <= 0.0))):
        raise CapacityExecutionIdentityError(
            "Typed roughness numeric range or work-depth identity differs",
            category="identity_mismatch",
        )
    skin = forcing_binding.get("skin_shear_controls")
    if not isinstance(skin, Mapping) or set(skin) != _SKIN_FIELDS:
        raise CapacityExecutionIdentityError(
            "Skin-shear control inventory differs",
            category="invalid_dependency",
        )
    skin_values = {}
    for key in _SKIN_FIELDS:
        value = skin[key]
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                not math.isfinite(float(value))):
            raise CapacityExecutionIdentityError(
                f"Skin-shear control is invalid: {key}",
                category="invalid_dependency",
            )
        skin_values[key] = float(value)
    if (min(skin_values["kinematic_viscosity_m2_s"],
            skin_values["sand_roughness_m"],
            skin_values["silt_roughness_m"]) <= 0.0 or
            not (0.0 <= skin_values["critical_mud_fraction_low"] <
                 skin_values["critical_mud_fraction_high"] <= 1.0) or
            not (0.0 <= skin_values["fixed_fluff_fraction"] <= 1.0)):
        raise CapacityExecutionIdentityError(
            "Skin-shear control values are invalid",
            category="invalid_dependency",
        )
    return TypedRoughnessDependency(
        forcing_schema=FORCING_BINDING_SCHEMA,
        forcing_fingerprint=fingerprint,
        roughness_method_id=method_id,
        parameter_name=parameter_name,
        parameter_units=parameter_units,
        storage_field=storage,
        array_shape=tuple(shape),
        array_dtype=dtype,
        array_sha256=sha,
        white_colebrook_chzk2d=chzk2d,
        sediment_shear_method_id=shear,
    )


def validate_typed_roughness_binding(
    forcing_binding: Mapping[str, object],
) -> TypedRoughnessDependency:
    """Authenticate a schema-2 forcing binding without choosing a capacity law.

    The QGIS host must validate the exact same C/n/ks and sediment-shear
    identity as the numerical worker, while remaining independent of the
    separately installed ``braids_gpu`` package.  Keep the implementation in
    the canonical capacity-identity module so host and worker consumers do
    not invent a second interpretation of the forcing document.
    """
    return _typed_roughness_dependency(forcing_binding)


def _profile_payload(profile: CapacityExecutionIdentity) -> dict:
    return {
        "schema": profile.schema,
        "scope_id": profile.scope_id,
        "semantic_id": profile.semantic_id,
        "method_id": profile.method_id,
        "formula_code": profile.formula_code,
        "reference_selector": profile.reference_selector,
        "source_id": profile.source_id,
        "closure": profile.closure,
        "selection_origin": profile.selection_origin,
        "shear_consumer": profile.shear_consumer,
        "generic_hiding_consumer": profile.generic_hiding_consumer,
        "parameters": [asdict(value) for value in profile.parameters],
        "typed_roughness": (
            None if profile.typed_roughness is None
            else asdict(profile.typed_roughness)
        ),
    }


def resolve_capacity_execution_identity(
    method_id: str,
    *,
    parameters: Mapping[str, object] | None = None,
    forcing_binding: Mapping[str, object] | None = None,
    selection_origin: str = SELECTION_EXPLICIT,
) -> CapacityExecutionIdentity:
    """Resolve one explicit closure without executing a solver step."""
    try:
        contract = METHOD_CONTRACTS[method_id]
    except (KeyError, TypeError) as error:
        raise CapacityExecutionIdentityError(
            f"Unknown capacity execution method identity: {method_id!r}",
            category="unknown_method",
        ) from error
    if selection_origin not in (SELECTION_EXPLICIT, SELECTION_LEGACY_MIGRATION):
        raise CapacityExecutionIdentityError(
            "Unknown capacity identity selection origin",
            category="invalid_configuration",
        )
    supplied = {} if parameters is None else parameters
    if not isinstance(supplied, Mapping):
        raise CapacityExecutionIdentityError(
            "Capacity execution parameters must be a mapping",
            category="invalid_parameters",
        )
    expected = {row[1] for row in contract.parameter_specs}
    if set(supplied) != expected:
        missing = sorted(expected - set(supplied))
        extra = sorted(set(supplied) - expected, key=str)
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if extra:
            details.append("extra " + ", ".join(map(str, extra)))
        raise CapacityExecutionIdentityError(
            "Capacity execution parameter fields differ: " + " and ".join(details),
            category="invalid_parameters",
        )
    resolved_parameters = tuple(
        CapacityExecutionParameter(
            name=name,
            source_field=source,
            units=units,
            value=_finite(supplied[source], source, positive=positive),
        )
        for name, source, units, _default, positive in contract.parameter_specs
    )
    if method_id == MPM_DELFT_METHOD_ID:
        critical = next(value.value for value in resolved_parameters
                        if value.name == "critical_shields")
        if not math.isclose(critical, 0.047, rel_tol=0.0, abs_tol=1.0e-14):
            raise CapacityExecutionIdentityError(
                "Delft MPM requires the locked critical Shields value 0.047",
                category="invalid_parameters",
            )
    if contract.requires_typed_roughness:
        typed = _typed_roughness_dependency(forcing_binding)  # type: ignore[arg-type]
    else:
        if forcing_binding is not None:
            raise CapacityExecutionIdentityError(
                "Legacy capacity closure must not claim a typed roughness consumer",
                category="physical_incompatibility",
            )
        typed = None
    placeholder = CapacityExecutionIdentity(
        CAPACITY_EXECUTION_SCHEMA,
        CAPACITY_EXECUTION_SCOPE_ID,
        capacity_execution_semantic_id(),
        contract.method_id,
        contract.formula_code,
        contract.reference_selector,
        contract.source_id,
        contract.closure,
        selection_origin,
        contract.shear_consumer,
        contract.generic_hiding_consumer,
        resolved_parameters,
        typed,
        "",
    )
    fingerprint = _hash(_profile_payload(placeholder))
    return CapacityExecutionIdentity(
        placeholder.schema, placeholder.scope_id, placeholder.semantic_id,
        placeholder.method_id, placeholder.formula_code,
        placeholder.reference_selector, placeholder.source_id,
        placeholder.closure, placeholder.selection_origin,
        placeholder.shear_consumer, placeholder.generic_hiding_consumer,
        placeholder.parameters,
        placeholder.typed_roughness, fingerprint,
    )


def migrate_legacy_capacity_controls(
    numerics: Mapping[str, object],
    *,
    forcing_binding: Mapping[str, object] | None = None,
) -> CapacityExecutionIdentity:
    """Resolve old project controls while preserving their physical meaning.

    If ``capacity_method_id`` is absent, law 8 always remains the historical
    independent-Chezy closure.  Law 2 mode 0 (or absent mode) remains the
    specified-Cf closure; only the already-explicit old mode 1 maps to the
    Delft route.  New corrected law-8 projects must explicitly store their
    method identity and must remove the legacy independent-Chezy field.
    """
    if not isinstance(numerics, Mapping):
        raise CapacityExecutionIdentityError(
            "Legacy numerical controls must be a mapping",
            category="invalid_configuration",
        )
    law = numerics.get("bedload", numerics.get("bedload_law"))
    # Legacy selectors are an exact integer contract.  In particular, Python
    # would otherwise treat 2.0/8.0 and booleans as dictionary-compatible
    # aliases and silently assign a physical method identity.
    if type(law) is not int or law not in (2, 8):
        raise CapacityExecutionIdentityError(
            "Capacity identity migration covers only BRAIDS laws 2 and 8",
            category="scope_restriction",
        )
    explicit = numerics.get("capacity_method_id")
    if explicit is None:
        origin = SELECTION_LEGACY_MIGRATION
        if law == 8:
            method_id = EH_LEGACY_METHOD_ID
        else:
            mode = numerics.get("mpm_friction_mode", 0)
            if type(mode) is not int or mode not in (0, 1):
                raise CapacityExecutionIdentityError(
                    "MPM friction mode must be 0 or 1",
                    category="invalid_configuration",
                )
            method_id = MPM_DELFT_METHOD_ID if mode == 1 else MPM_LEGACY_METHOD_ID
    else:
        if not isinstance(explicit, str):
            raise CapacityExecutionIdentityError(
                "capacity_method_id must be text",
                category="invalid_configuration",
            )
        origin = SELECTION_EXPLICIT
        method_id = explicit
    try:
        contract = METHOD_CONTRACTS[method_id]
    except KeyError as error:
        raise CapacityExecutionIdentityError(
            f"Unknown capacity execution method identity: {method_id!r}",
            category="unknown_method",
        ) from error
    if contract.formula_code != law:
        raise CapacityExecutionIdentityError(
            "Capacity method identity conflicts with the integer formula code",
            category="identity_mismatch",
        )
    if method_id == EH_LEGACY_METHOD_ID:
        parameters = {
            "engelund_hansen_chezy": numerics.get("engelund_hansen_chezy", 28.8)
        }
        forcing = None
    elif method_id == MPM_LEGACY_METHOD_ID:
        mode = numerics.get("mpm_friction_mode", 0)
        if mode is not None and (type(mode) is not int or mode != 0):
            raise CapacityExecutionIdentityError(
                "Specified-Cf MPM conflicts with roughness-corrected mode",
                category="identity_mismatch",
            )
        parameters = {
            "mpm_cf": numerics.get("mpm_cf", 0.01),
            "mpm_critical_shields": numerics.get("mpm_critical_shields", 0.0),
        }
        forcing = None
    elif method_id == EH_DELFT_METHOD_ID:
        if "engelund_hansen_chezy" in numerics:
            raise CapacityExecutionIdentityError(
                "Corrected Delft Engelund-Hansen cannot retain an independent Chezy field",
                category="identity_mismatch",
            )
        parameters = {}
        forcing = forcing_binding
    else:
        mode = numerics.get("mpm_friction_mode", 1)
        if type(mode) is not int or mode != 1:
            raise CapacityExecutionIdentityError(
                "Delft MPM requires the resolved-roughness mode",
                category="identity_mismatch",
            )
        if origin == SELECTION_EXPLICIT and "mpm_cf" in numerics:
            raise CapacityExecutionIdentityError(
                "Delft MPM cannot retain a specified friction coefficient",
                category="identity_mismatch",
            )
        parameters = {
            "mpm_critical_shields": numerics.get("mpm_critical_shields", 0.047)
        }
        forcing = forcing_binding
    return resolve_capacity_execution_identity(
        method_id,
        parameters=parameters,
        forcing_binding=forcing,
        selection_origin=origin,
    )


def capacity_execution_document(
    profile: CapacityExecutionIdentity,
    *,
    forcing_binding: Mapping[str, object] | None = None,
) -> dict:
    if not isinstance(profile, CapacityExecutionIdentity):
        raise CapacityExecutionIdentityError(
            "Expected a CapacityExecutionIdentity",
            category="invalid_configuration",
        )
    rebuilt = resolve_capacity_execution_identity(
        profile.method_id,
        parameters={value.source_field: value.value for value in profile.parameters},
        forcing_binding=forcing_binding,
        selection_origin=profile.selection_origin,
    )
    if _canonical_json(_profile_payload(profile)) != _canonical_json(
            _profile_payload(rebuilt)):
        raise CapacityExecutionIdentityError(
            "Capacity execution object differs from its resolved contract",
            category="identity_mismatch",
        )
    if profile.fingerprint != _hash(_profile_payload(profile)):
        raise CapacityExecutionIdentityError(
            "Capacity execution fingerprint differs",
            category="identity_mismatch",
        )
    document = _profile_payload(profile)
    document["fingerprint"] = profile.fingerprint
    return json.loads(_canonical_json(document))


def _strict_object(pairs: list[tuple[str, object]]) -> dict:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise CapacityExecutionIdentityError(
                f"Duplicate capacity execution key: {key!r}",
                category="invalid_serialization",
            )
        result[key] = value
    return result


def load_capacity_execution_identity(
    document: str | bytes | Mapping[str, object],
    *,
    forcing_binding: Mapping[str, object] | None = None,
) -> CapacityExecutionIdentity:
    """Re-resolve a serialized identity and reject any unledgered change."""
    if isinstance(document, (str, bytes)):
        try:
            raw = json.loads(document, object_pairs_hook=_strict_object,
                             parse_constant=lambda value: (_ for _ in ()).throw(
                                 CapacityExecutionIdentityError(
                                     f"Non-finite JSON number: {value}",
                                     category="invalid_serialization")))
        except CapacityExecutionIdentityError:
            raise
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise CapacityExecutionIdentityError(
                "Invalid capacity execution JSON",
                category="invalid_serialization",
            ) from error
    elif isinstance(document, Mapping):
        try:
            raw = json.loads(_canonical_json(document))
        except CapacityExecutionIdentityError:
            raise
    else:
        raise CapacityExecutionIdentityError(
            "Capacity execution document must be JSON or a mapping",
            category="invalid_serialization",
        )
    if not isinstance(raw, dict):
        raise CapacityExecutionIdentityError(
            "Capacity execution document must be an object",
            category="invalid_serialization",
        )
    try:
        parameter_rows = raw["parameters"]
        if not isinstance(parameter_rows, list):
            raise TypeError("parameters")
        parameters = {}
        for row in parameter_rows:
            if not isinstance(row, dict) or set(row) != {
                    "name", "source_field", "units", "value"}:
                raise TypeError("parameter")
            if row["source_field"] in parameters:
                raise ValueError("duplicate parameter")
            parameters[row["source_field"]] = row["value"]
        rebuilt = resolve_capacity_execution_identity(
            raw["method_id"],
            parameters=parameters,
            forcing_binding=forcing_binding,
            selection_origin=raw["selection_origin"],
        )
    except CapacityExecutionIdentityError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise CapacityExecutionIdentityError(
            "Malformed capacity execution document",
            category="invalid_serialization",
        ) from error
    expected = _profile_payload(rebuilt)
    expected["fingerprint"] = rebuilt.fingerprint
    if _canonical_json(raw) != _canonical_json(expected):
        raise CapacityExecutionIdentityError(
            "Capacity execution document is stale, forged, or non-canonical",
            category="identity_mismatch",
        )
    return rebuilt


def capacity_execution_json(
    profile: CapacityExecutionIdentity,
    *,
    forcing_binding: Mapping[str, object] | None = None,
    indent: int | None = 2,
) -> str:
    """Serialize after a full re-resolution, including corrected dependencies."""
    if not isinstance(profile, CapacityExecutionIdentity):
        raise CapacityExecutionIdentityError(
            "Expected a CapacityExecutionIdentity",
            category="invalid_configuration",
        )
    document = capacity_execution_document(
        profile, forcing_binding=forcing_binding)
    return json.dumps(document, sort_keys=True, indent=indent,
                      ensure_ascii=True, allow_nan=False) + (
                          "\n" if indent is not None else "")


__all__ = [
    "CAPACITY_EXECUTION_SCHEMA", "CAPACITY_EXECUTION_SCOPE_ID",
    "FORCING_BINDING_SCHEMA", "EH_LEGACY_METHOD_ID", "EH_DELFT_METHOD_ID",
    "MPM_LEGACY_METHOD_ID", "MPM_DELFT_METHOD_ID", "SELECTION_EXPLICIT",
    "SELECTION_LEGACY_MIGRATION", "METHOD_CONTRACTS",
    "CapacityExecutionParameter", "TypedRoughnessDependency",
    "CapacityExecutionIdentity", "CapacityExecutionIdentityError",
    "capacity_execution_semantic_document", "capacity_execution_semantic_id",
    "resolve_capacity_execution_identity", "migrate_legacy_capacity_controls",
    "capacity_execution_document", "load_capacity_execution_identity",
    "capacity_execution_json", "validate_typed_roughness_binding",
]
