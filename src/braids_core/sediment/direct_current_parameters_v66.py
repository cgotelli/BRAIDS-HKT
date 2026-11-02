"""Additive typed identities for three current-only Delft direct formulae.

The frozen capacity-parameter schema predates Q3-H6 and already assigns
BRAIDS internal law ``5`` to Wilcock-Crowe.  This module therefore creates a
separate v66 wire namespace for Delft3D4 FLOW TraFrm 5, 19 and 20.  It does
not modify :mod:`capacity_parameters`, its catalogue digest, or any public
dispatch table.

The identity is deliberately stricter than ``rdtrafrm`` where the classic
FLOW call would otherwise consume undefined fallback state.  In particular,
Bijker requires explicit porosity and a positive ``TWave`` value, while
formula-internal slope/dilatancy options unsupported by the classic
current-only foundation are accepted only in their zero/off state.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Mapping

from .capacity_parameters import capacity_parameter_catalog_id


SCHEMA = 1
SCOPE_ID = "braids-q3-h6-direct-current-parameters-v66-v1"
LOCKED_REFERENCE_COMMIT = "c05ed672f0a7fe81c5b9867ef354cfdd53c25379"
FROZEN_V1_PARAMETER_CATALOG_ID = (
    "c8d95abb6f27c127b98f1ce058f415fc57df496608c83e7f79026cbb30ab23bd"
)

# This namespace is intentionally far outside the frozen one/two-digit M1
# wire range.  The final three digits retain the Delft reference selector.
WIRE_BIJKER_CURRENT = 66005
WIRE_VAN_THIEL_VAN_RIJN_2008_CURRENT = 66019
WIRE_XBEACH_SOULSBY_VAN_RIJN_CURRENT = 66020

METHOD_BIJKER_CURRENT = "capacity.bijker_current"
METHOD_VAN_THIEL_VAN_RIJN_2008_CURRENT = (
    "capacity.van_thiel_van_rijn_2008_current"
)
METHOD_XBEACH_SOULSBY_VAN_RIJN_CURRENT = (
    "capacity.xbeach_soulsby_van_rijn_current"
)

OUTPUT_EXPLICIT_VECTORS = "explicit_bed_and_suspended_mass_vectors"
OUTPUT_EQUILIBRIUM_CONCENTRATION = (
    "bed_mass_vector_and_equilibrium_mass_concentration"
)

_SOURCE_ROOT = "resources/Delft3D-main/"
LOCKED_SOURCES = MappingProxyType({
    "eqtran": (
        "src/utils_gpl/morphology/packages/morphology_kernel/src/eqtran.f90",
        "d2e8601cff68884f51ee3dcb0cd2fb52d9f0d676b1baaa3283c68913d4778388",
    ),
    "rdtrafrm": (
        "src/utils_gpl/morphology/packages/morphology_io/src/rdtrafrm.f90",
        "0a2542980bc9e9e03e46b2abb8b06eda2fea3e28a67c78c4e61b619572154ea7",
    ),
    "tranb5": (
        "src/utils_gpl/morphology/packages/morphology_kernel/src/tranb5.f90",
        "ca5933ccda6592174c627c81d2d10e977064dd47f96560d349b1a3fca50a2070",
    ),
    "fgyint": (
        "src/utils_gpl/morphology/packages/morphology_kernel/src/fgyint.f90",
        "135138690f0b56a0f89f18cd9f9bc85149f29458f76b167f0ce54ec695730e19",
    ),
    "termfy": (
        "src/utils_gpl/morphology/packages/morphology_kernel/src/termfy.f90",
        "55bb8621b7d09e7a64d4c4e1a8b4b38697801c3fcfb7f11792c56e9e9e0e11e5",
    ),
    "termgy": (
        "src/utils_gpl/morphology/packages/morphology_kernel/src/termgy.f90",
        "6d6d1b81af9140f299fbd8a3e04cfe10163f2a9db1d235651995f4a9232c4618",
    ),
    "trab19": (
        "src/utils_gpl/morphology/packages/morphology_kernel/src/trab19.f90",
        "bfc530b573b95c4467a9de4bd34e25757f3990d489426db995b28d4b7ad34c63",
    ),
    "trab20": (
        "src/utils_gpl/morphology/packages/morphology_kernel/src/trab20.f90",
        "a7cbf17f232fc6edae5a1cf5f7c9b2ca1e1e0626aee6dbb5270efc1c245e4d19",
    ),
    "sed_support_routines": (
        "src/utils_gpl/morphology/packages/morphology_kernel/src/"
        "sed_support_routines.f90",
        "9a3244290df46993421049b6813f79489c006974050a35485f7db30463b4699f",
    ),
})


@dataclass(frozen=True)
class DirectCurrentParameterV66:
    keyword: str
    kernel_field: str | None
    source_position: int
    source_required: bool
    source_default: float | int | None
    current_required: bool
    value_kind: str
    current_role: str
    allowed: tuple[int, ...] = ()
    minimum: float | int | None = None
    minimum_inclusive: bool = True
    maximum: float | int | None = None
    maximum_inclusive: bool = True
    source_normalization: str = "none"


@dataclass(frozen=True)
class DirectCurrentContractV66:
    method_id: str
    internal_wire_id: int
    reference_selector: int
    source_routine: str
    output_ownership: str
    parameters: tuple[DirectCurrentParameterV66, ...]
    source_names: tuple[str, ...]
    classic_zero_fields: tuple[str, ...]
    fail_closed_options: tuple[str, ...]


def _p(keyword: str, kernel_field: str | None, position: int, *,
       source_required: bool = False,
       source_default: float | int | None = None,
       current_required: bool = False,
       value_kind: str = "real", current_role: str = "active",
       allowed: tuple[int, ...] = (), minimum: float | int | None = None,
       minimum_inclusive: bool = True,
       maximum: float | int | None = None,
       maximum_inclusive: bool = True,
       source_normalization: str = "none") -> DirectCurrentParameterV66:
    return DirectCurrentParameterV66(
        keyword, kernel_field, position, source_required, source_default,
        current_required, value_kind, current_role, allowed, minimum,
        minimum_inclusive, maximum, maximum_inclusive, source_normalization,
    )


_BIJKER_PARAMETERS = (
    _p("CalBs", "cal_bs", 1, source_required=True),
    _p("CalBd", "cal_bd", 2, source_required=True),
    _p("CritCs", "crit_cs", 3, source_required=True),
    _p("CritCd", "crit_cd", 4, source_required=True),
    _p("[dummy]", None, 5, source_required=True, source_default=1.0,
       current_role="obsolete_retained_identity_not_forwarded"),
    _p("RouKs", "roughness_ks", 6, source_required=True,
       source_default=-999.0,
       current_role="active_negative_uses_chezy_roughness_fallback"),
    _p("WSettle", "formula_settling_velocity", 7, source_required=True,
       source_default=-999.0,
       current_role="active_negative_uses_fraction_settling_fallback"),
    _p("Porosity", "porosity", 8, source_required=True,
       source_default=-999.0, current_required=True,
       current_role="active_explicit_required_no_undefined_classic_fallback",
       minimum=0.0, maximum=1.0, maximum_inclusive=False),
    _p("TWave", "formula_wave_period", 9, source_required=True,
       source_default=-999.0, current_required=True,
       current_role="transport_inert_but_positive_required_for_source_wavenr",
       minimum=0.0, minimum_inclusive=False),
    _p("FacA", "fac_a", 10, source_default=0.0,
       current_role="inert_when_EpsSL_is_zero"),
    _p("FacU", "fac_u", 11, source_default=0.0,
       current_role="inert_when_EpsSL_is_zero"),
    _p("EpsSL", "eps_sl", 12, source_default=0.0,
       value_kind="integer", current_role="fail_closed_nonzero_FM_or_wave_slope_path",
       allowed=(0,)),
)


def _parameters_1920(*, xbeach: bool) -> tuple[DirectCurrentParameterV66, ...]:
    result = [
        _p("facua", "facua", 1, source_default=0.0,
           current_role="wave_only_inert"),
        _p("facAs", "facas", 2, source_default=0.2,
           current_role="wave_only_inert", source_normalization="clamp[0,1]"),
        _p("facSk", "facsk", 3, source_default=0.15,
           current_role="wave_only_inert", source_normalization="clamp[0,1]"),
        _p("waveform", "waveform", 4, source_default=2, value_kind="integer",
           current_role="wave_only_inert", allowed=(1, 2),
           source_normalization="invalid_to_2"),
        _p("sws", "sws", 5, source_default=1, value_kind="integer",
           current_role="wave_only_inert", allowed=(0, 1),
           source_normalization="invalid_to_1"),
        _p("lws", "lws", 6, source_default=1, value_kind="integer",
           current_role="source_unused_inert", allowed=(1,),
           source_normalization="invalid_to_1"),
        _p("dilatancy", "dilatancy", 7, source_default=0,
           value_kind="integer", current_role="fail_closed_FM_derived_nonzero",
           allowed=(0,)),
        _p("rheeA", "rheea", 8, source_default=0.75,
           current_role="inert_when_dilatancy_zero",
           source_normalization="clamp[0.75,2]"),
        _p("pormax", "pormax", 9, source_default=0.5,
           current_role="inert_when_dilatancy_zero",
           source_normalization="clamp[porosity,0.6]"),
        _p("bedslpini", "bedslpini", 10, source_default=0,
           value_kind="integer", current_role="fail_closed_FM_derived_nonzero",
           allowed=(0,)),
        _p("smax", "smax", 11, source_default=-1.0,
           current_role="active_velocity_cap_minus1_means_unbounded",
           source_normalization="clamp[-1,3]"),
        _p("reposeangle", "reposeangle", 12, source_default=30.0,
           current_role="inert_when_bedslpini_zero",
           source_normalization="clamp[30,45]"),
        _p("cmax", "cmax", 13, source_default=0.1,
           current_role="active_equilibrium_concentration_cap",
           source_normalization="clamp[0,1]"),
    ]
    if xbeach:
        result.append(_p(
            "z0", "z0", 14, source_default=0.006,
            current_role="wave_only_inert_but_drag_is_evaluated",
            source_normalization="clamp[0.0001,0.05]",
        ))
        alfa_position = 15
    else:
        alfa_position = 14
    result.append(_p(
        "alfad50", "alfad50", alfa_position, source_default=0.0,
        current_role="active_grain_size_multiplier",
        source_normalization="clamp[0,1.5]",
    ))
    return tuple(result)


_CONTRACT_ROWS = (
    DirectCurrentContractV66(
        METHOD_BIJKER_CURRENT, WIRE_BIJKER_CURRENT, 5, "tranb5",
        OUTPUT_EXPLICIT_VECTORS, _BIJKER_PARAMETERS,
        ("rdtrafrm", "eqtran", "tranb5", "fgyint", "termfy", "termgy"),
        ("wave_height", "wave_period", "wavelength", "wave_angle",
         "orbital_velocity", "breaker_turbulence", "bed_slope_x", "bed_slope_y"),
        ("EpsSL!=0",),
    ),
    DirectCurrentContractV66(
        METHOD_VAN_THIEL_VAN_RIJN_2008_CURRENT,
        WIRE_VAN_THIEL_VAN_RIJN_2008_CURRENT, 19, "trab19",
        OUTPUT_EQUILIBRIUM_CONCENTRATION, _parameters_1920(xbeach=False),
        ("rdtrafrm", "eqtran", "trab19", "sed_support_routines"),
        ("wave_height", "wave_period", "wavelength", "wave_angle",
         "orbital_velocity", "breaker_turbulence", "bed_change_rate",
         "bed_slope_x", "bed_slope_y"),
        ("dilatancy!=0", "bedslpini!=0"),
    ),
    DirectCurrentContractV66(
        METHOD_XBEACH_SOULSBY_VAN_RIJN_CURRENT,
        WIRE_XBEACH_SOULSBY_VAN_RIJN_CURRENT, 20, "trab20",
        OUTPUT_EQUILIBRIUM_CONCENTRATION, _parameters_1920(xbeach=True),
        ("rdtrafrm", "eqtran", "trab20", "sed_support_routines"),
        ("wave_height", "wave_period", "wavelength", "wave_angle",
         "orbital_velocity", "breaker_turbulence", "bed_change_rate",
         "bed_slope_x", "bed_slope_y"),
        ("dilatancy!=0", "bedslpini!=0"),
    ),
)

CONTRACTS = MappingProxyType({row.method_id: row for row in _CONTRACT_ROWS})
CONTRACT_BY_WIRE_ID = MappingProxyType({row.internal_wire_id: row
                                        for row in _CONTRACT_ROWS})
CONTRACT_BY_REFERENCE_SELECTOR = MappingProxyType({row.reference_selector: row
                                                   for row in _CONTRACT_ROWS})


class DirectCurrentParameterV66Error(ValueError):
    """Strict v66 identity error."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise DirectCurrentParameterV66Error(
            "Direct-current v66 identity is not canonical JSON"
        ) from error


def _assert_frozen_v1() -> None:
    if capacity_parameter_catalog_id() != FROZEN_V1_PARAMETER_CATALOG_ID:
        raise DirectCurrentParameterV66Error(
            "Frozen capacity-parameter schema-v1 identity changed"
        )


def direct_current_contract_v66(
    identifier: str | int,
) -> DirectCurrentContractV66:
    try:
        if isinstance(identifier, str):
            return CONTRACTS[identifier]
        if type(identifier) is not int:
            raise KeyError(identifier)
        # Integers in this API are v66 wire IDs only.  Reference selectors use
        # the explicitly named function below, preventing law-5 aliasing.
        return CONTRACT_BY_WIRE_ID[identifier]
    except (KeyError, TypeError) as error:
        raise DirectCurrentParameterV66Error(
            f"Unknown direct-current v66 method or wire ID: {identifier!r}"
        ) from error


def direct_current_contract_by_reference_selector_v66(
    selector: int,
) -> DirectCurrentContractV66:
    if type(selector) is not int:
        raise DirectCurrentParameterV66Error(
            f"Reference selector must be an exact integer: {selector!r}"
        )
    try:
        return CONTRACT_BY_REFERENCE_SELECTOR[selector]
    except KeyError as error:
        raise DirectCurrentParameterV66Error(
            f"Unknown direct-current Delft reference selector: {selector!r}"
        ) from error


def _validate_parameter(
    specification: DirectCurrentParameterV66, value: object,
) -> float | int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DirectCurrentParameterV66Error(
            f"{specification.keyword} must be a finite numeric value"
        )
    numeric = float(value)
    if not math.isfinite(numeric):
        raise DirectCurrentParameterV66Error(
            f"{specification.keyword} must be a finite numeric value"
        )
    if specification.value_kind == "integer":
        integer = int(numeric)
        if numeric != integer or (
                specification.allowed and integer not in specification.allowed):
            choices = f" in {specification.allowed}" if specification.allowed else ""
            raise DirectCurrentParameterV66Error(
                f"{specification.keyword} must be an integer{choices}"
            )
        result: float | int = integer
    else:
        result = 0.0 if numeric == 0.0 else numeric
    if specification.minimum is not None:
        below = numeric < specification.minimum
        excluded = (numeric == specification.minimum and
                    not specification.minimum_inclusive)
        if below or excluded:
            relation = ">=" if specification.minimum_inclusive else ">"
            raise DirectCurrentParameterV66Error(
                f"{specification.keyword} must be {relation} {specification.minimum}"
            )
    if specification.maximum is not None:
        above = numeric > specification.maximum
        excluded = (numeric == specification.maximum and
                    not specification.maximum_inclusive)
        if above or excluded:
            relation = "<=" if specification.maximum_inclusive else "<"
            raise DirectCurrentParameterV66Error(
                f"{specification.keyword} must be {relation} {specification.maximum}"
            )
    return result


def resolve_direct_current_parameters_v66(
    identifier: str | int,
    supplied: Mapping[str, float | int],
) -> Mapping[str, float | int]:
    _assert_frozen_v1()
    contract = direct_current_contract_v66(identifier)
    if not isinstance(supplied, Mapping):
        raise DirectCurrentParameterV66Error(
            "Direct-current v66 parameters must be a mapping"
        )
    expected = {row.keyword for row in contract.parameters}
    extra = tuple(sorted(set(supplied)-expected))
    if extra:
        raise DirectCurrentParameterV66Error(
            f"Unexpected {contract.method_id} parameter(s): {', '.join(extra)}"
        )
    resolved: dict[str, float | int] = {}
    for specification in contract.parameters:
        if specification.keyword in supplied:
            value = supplied[specification.keyword]
        elif specification.current_required:
            raise DirectCurrentParameterV66Error(
                f"Missing current-only required {contract.method_id} parameter: "
                f"{specification.keyword}"
            )
        elif specification.source_default is not None:
            value = specification.source_default
        elif specification.source_required:
            raise DirectCurrentParameterV66Error(
                f"Missing required {contract.method_id} parameter: "
                f"{specification.keyword}"
            )
        else:
            raise DirectCurrentParameterV66Error(
                f"No deterministic value for {specification.keyword}"
            )
        resolved[specification.keyword] = _validate_parameter(specification, value)
    return MappingProxyType(resolved)


def canonical_direct_current_parameter_items_v66(
    identifier: str | int,
    supplied: Mapping[str, float | int],
) -> tuple[tuple[str, float | int], ...]:
    contract = direct_current_contract_v66(identifier)
    resolved = resolve_direct_current_parameters_v66(contract.method_id, supplied)
    return tuple((row.keyword, resolved[row.keyword]) for row in contract.parameters)


def direct_current_parameter_semantic_document_v66() -> dict:
    _assert_frozen_v1()
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "locked_reference_commit": LOCKED_REFERENCE_COMMIT,
        "frozen_schema_v1_changed": False,
        "frozen_v1_parameter_catalog_id": FROZEN_V1_PARAMETER_CATALOG_ID,
        "integer_namespace": "v66_internal_wire_id_not_legacy_law_or_TraFrm",
        "locked_sources": {
            name: {"path": _SOURCE_ROOT+value[0], "sha256": value[1]}
            for name, value in sorted(LOCKED_SOURCES.items())
        },
        "methods": [
            {
                **{
                    key: (list(value) if isinstance(value, tuple) else value)
                    for key, value in asdict(contract).items()
                    if key != "parameters"
                },
                "parameters": [
                    {
                        **asdict(parameter),
                        "allowed": list(parameter.allowed),
                    }
                    for parameter in contract.parameters
                ],
            }
            for contract in sorted(_CONTRACT_ROWS,
                                   key=lambda row: row.internal_wire_id)
        ],
        "common_forcing": {
            "wave_fields": "explicit_exact_zero_fail_closed",
            "automatic_CPU_fallback": False,
            "external_secondary_rotation": False,
            "density_conversion": "positive_IFORM_volume_to_mass_exactly_once",
        },
    }


SEMANTIC_ID = hashlib.sha256(
    _canonical(direct_current_parameter_semantic_document_v66())
).hexdigest()


def direct_current_parameter_document_v66(
    identifier: str | int,
    supplied: Mapping[str, float | int],
) -> dict:
    contract = direct_current_contract_v66(identifier)
    items = canonical_direct_current_parameter_items_v66(
        contract.method_id, supplied,
    )
    source_rows = {
        name: {"path": _SOURCE_ROOT+LOCKED_SOURCES[name][0],
               "sha256": LOCKED_SOURCES[name][1]}
        for name in contract.source_names
    }
    body = {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "qualification": "bounded_constitutive_foundation_not_live_dispatch",
        "frozen_v1_parameter_catalog_id": FROZEN_V1_PARAMETER_CATALOG_ID,
        "method_id": contract.method_id,
        "internal_wire_id": contract.internal_wire_id,
        "reference_selector": contract.reference_selector,
        "source_routine": contract.source_routine,
        "output_ownership": contract.output_ownership,
        "parameters": [
            {"keyword": keyword, "value": value} for keyword, value in items
        ],
        "classic_zero_fields": list(contract.classic_zero_fields),
        "fail_closed_options": list(contract.fail_closed_options),
        "locked_sources": source_rows,
    }
    return {**body, "fingerprint": hashlib.sha256(_canonical(body)).hexdigest()}


def load_direct_current_parameter_document_v66(
    value: Mapping[str, object],
) -> dict:
    if not isinstance(value, Mapping):
        raise DirectCurrentParameterV66Error(
            "Direct-current v66 parameter identity must be an object"
        )
    expected = {
        "schema", "scope_id", "semantic_id", "qualification",
        "frozen_v1_parameter_catalog_id", "method_id", "internal_wire_id",
        "reference_selector", "source_routine", "output_ownership",
        "parameters", "classic_zero_fields", "fail_closed_options",
        "locked_sources", "fingerprint",
    }
    if set(value) != expected:
        raise DirectCurrentParameterV66Error(
            "Direct-current v66 parameter identity inventory differs"
        )
    parameters = value.get("parameters")
    if not isinstance(parameters, list) or any(
            not isinstance(row, Mapping) or set(row) != {"keyword", "value"}
            for row in parameters):
        raise DirectCurrentParameterV66Error(
            "Direct-current v66 parameter rows are malformed"
        )
    supplied = {str(row["keyword"]): row["value"] for row in parameters}
    rebuilt = direct_current_parameter_document_v66(
        value.get("method_id"), supplied,
    )
    if dict(value) != rebuilt or _canonical(value) != _canonical(rebuilt):
        raise DirectCurrentParameterV66Error(
            "Direct-current v66 parameter identity is stale or forged"
        )
    return rebuilt


__all__ = [
    "CONTRACTS", "CONTRACT_BY_REFERENCE_SELECTOR", "CONTRACT_BY_WIRE_ID",
    "DirectCurrentContractV66", "DirectCurrentParameterV66",
    "DirectCurrentParameterV66Error", "FROZEN_V1_PARAMETER_CATALOG_ID",
    "LOCKED_REFERENCE_COMMIT", "LOCKED_SOURCES", "METHOD_BIJKER_CURRENT",
    "METHOD_VAN_THIEL_VAN_RIJN_2008_CURRENT",
    "METHOD_XBEACH_SOULSBY_VAN_RIJN_CURRENT", "OUTPUT_EQUILIBRIUM_CONCENTRATION",
    "OUTPUT_EXPLICIT_VECTORS", "SCHEMA", "SCOPE_ID", "SEMANTIC_ID",
    "WIRE_BIJKER_CURRENT", "WIRE_VAN_THIEL_VAN_RIJN_2008_CURRENT",
    "WIRE_XBEACH_SOULSBY_VAN_RIJN_CURRENT",
    "canonical_direct_current_parameter_items_v66",
    "direct_current_contract_by_reference_selector_v66",
    "direct_current_contract_v66", "direct_current_parameter_document_v66",
    "direct_current_parameter_semantic_document_v66",
    "load_direct_current_parameter_document_v66",
    "resolve_direct_current_parameters_v66",
]
