"""Typed Delft3D FLOW current-capacity parameter contracts.

The method catalogue identifies a physical formula.  This module identifies
the coefficients that make one configured instance of that formula.  Keeping
the two concepts separate prevents route-support bookkeeping from changing a
restart identity, while preventing a solver from claiming formula equivalence
after silently substituting standard coefficients.

Names and defaults follow the locked ``rdtrafrm.f90`` reference.  Obsolete
compatibility fields are retained in the schema but can be marked inert by a
current-only implementation.  Numerical kernels still perform their own
state-dependent physical braids_support.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class CapacityParameter:
    keyword: str
    required: bool
    default: float | int | None = None
    value_kind: str = "real"
    current_role: str = "active"
    allowed: tuple[int, ...] = ()
    minimum: float | int | None = None
    minimum_inclusive: bool = True


@dataclass(frozen=True)
class CapacityFormulaContract:
    method_id: str
    legacy_code: int | None
    reference_selector: int | None
    output_shape: str
    external_secondary_direction: bool
    generic_hiding: bool
    parameters: tuple[CapacityParameter, ...]


def _p(keyword: str, *, required: bool = True,
       default: float | int | None = None, value_kind: str = "real",
       current_role: str = "active", allowed: tuple[int, ...] = (),
       minimum: float | int | None = None,
       minimum_inclusive: bool = True) -> CapacityParameter:
    return CapacityParameter(keyword, required, default, value_kind,
                             current_role, allowed, minimum,
                             minimum_inclusive)


def _f(method_id: str, code: int | None, selector: int | None, output: str,
       secondary: bool, hiding: bool, *parameters: CapacityParameter) -> CapacityFormulaContract:
    return CapacityFormulaContract(method_id, code, selector, output,
                                   secondary, hiding, tuple(parameters))


_EH_PARAMETERS = (
    _p("ACal"),
    _p("RouKs", required=False, default=1.0, current_role="obsolete"),
    _p("SusFac", required=False, default=0.0),
)
_MPM_PARAMETERS = (_p("ACal"),)


_CONTRACTS = (
    # A coupled research bundle, never an equilibrium Exner/native alias.
    _f("capacity.mpm_jc2026_p15_reconstruction_v1", None, None, "mobile_bedload_bundle", False, False,
       _p("VicMol", minimum=0., minimum_inclusive=False),
       _p("ReposeAngle_deg", minimum=0., minimum_inclusive=False),
       _p("ProtectInitialDryBank", value_kind="integer", allowed=(0, 1))),
    _f("capacity.wong_parker_2006", None, None, "scalar_split", True, False,
       _p("ACal", minimum=0)),
    _f("capacity.camenen_larson_2005", None, None, "scalar_split", True, False,
       _p("ACal", minimum=0.0), _p("VicMol", minimum=0.0, minimum_inclusive=False)),
    _f("capacity.recking_2010", None, None, "scalar_split", True, False,
       _p("ACal", minimum=0.0),_p("ReachSlope", minimum=0.0,minimum_inclusive=False)),
    _f("capacity.ancey_recking_2023", None, None, "scalar_split", True, False,
       _p("ACal", minimum=0.0)),
    # Independent published energy-slope law, not a FLOW selector or legacy alias.
    _f("capacity.smart_jaggi", None, None, "scalar_split", True, False,
       _p("ACal", minimum=0.0), _p("ThetaC", minimum=0.0)),
    _f("capacity.van_rijn_1993", 1, -1, "split_vectors", False, False,
       _p("IopSus", required=False, default=0, value_kind="integer", allowed=(0, 1)),
       _p("AksFac", required=False, default=1.0),
       _p("RWave", required=False, default=2.0, current_role="wave_only"),
       _p("RDC", required=False, default=0.01),
       _p("RDW", required=False, default=0.02, current_role="wave_only"),
       _p("IopKCW", required=False, default=1, value_kind="integer", allowed=(0, 1)),
       _p("EpsPar", required=False, default=0, value_kind="integer",
          current_role="wave_only", allowed=(0, 1)),
       _p("BetaM", required=False, default=3.0),
       _p("iTauCr", required=False, default=1, value_kind="integer", allowed=(1, 2))),
    # Formula codes 8 and 2 predate typed execution identities.  Both pairs
    # deliberately share the native formula code and FLOW selector while the
    # method ID preserves the different forcing closure. Integer-only readers
    # below resolve to the historical identities, never to the Delft routes.
    _f("capacity.engelund_hansen_constant_chezy_legacy", 8, 1,
       "scalar_split", True, False, *_EH_PARAMETERS),
    _f("capacity.engelund_hansen", 8, 1, "scalar_split", True, False,
       *_EH_PARAMETERS),
    _f("capacity.mpm_specified_cf_legacy", 2, 2, "scalar_split", True, True,
       *_MPM_PARAMETERS),
    _f("capacity.mpm", 2, 2, "scalar_split", True, True,
       *_MPM_PARAMETERS),
    _f("capacity.general", 11, 4, "scalar_split", True, True,
       _p("ACal", minimum=0.0), _p("PowerB"), _p("PowerC"),
       _p("RipFac", minimum=0.0), _p("ThetaC", minimum=0.0),
       _p("SusACal", required=False, default=0.0, minimum=0.0),
       _p("SusPowerB", required=False, default=3.0),
       _p("SusPowerC", required=False, default=0.0),
       _p("SusRipFac", required=False, default=1.0, minimum=0.0),
       _p("SusThetaC", required=False, default=0.0, minimum=0.0)),
    _f("capacity.van_rijn_1984_modified", 12, 7, "scalar_split", True, False,
       _p("ACal", minimum=0.0),
       _p("[dummy]", required=False, default=1.0, current_role="obsolete"),
       _p("Aks", minimum=0.0, minimum_inclusive=False),
       _p("WSettle", minimum=0.0),
       _p("BetaM", required=False, default=0.0)),
    _f("capacity.soulsby_van_rijn", 13, 11, "split_vectors", False, False,
       _p("ACal", minimum=0.0),
       _p("RatioD90D50", minimum=0.0, minimum_inclusive=False),
       _p("RouZ0", minimum=0.0, minimum_inclusive=False),
       _p("AlfaUrms", required=False, default=1.0, current_role="wave_only")),
    _f("capacity.soulsby", 4, 12, "split_vectors", False, False,
       _p("ACal"), _p("ModInd", value_kind="integer", allowed=tuple(range(1, 9)),
                       current_role="wave_only"),
       _p("RatioD50Z0")),
    _f("capacity.wang_fredsoe", 14, 13, "scalar_split", True, False,
       _p("VicMol", minimum=0.0, minimum_inclusive=False),
       _p("ACal", minimum=0.0)),
    _f("capacity.ashida_michiue_generalized", 3, 14, "scalar_split", True, True,
       _p("ACal"), _p("ThetaC"), _p("PowerM"), _p("PowerP"), _p("PowerQ")),
    _f("capacity.wilcock_crowe", 5, 16, "intrinsic_mixture_scalar", True, False,
       _p("ACal", required=False, default=1.0)),
    _f("capacity.wilcock_crowe_modified", 6, 17,
       "intrinsic_mixture_scalar", True, False,
       # Keep the established schema-v1 positional contract immutable.  The
       # modern keyword reader's defaults are completed by the additive v64
       # scalar-law resolver so old profile/restart identities remain valid.
       _p("ThetaC0"), _p("Alpha0")),
    _f("capacity.gaeuman", 7, 18, "intrinsic_mixture_scalar", True, False,
       _p("ThetaC0"), _p("Alpha0")),
)


FORMULA_CONTRACTS = MappingProxyType({row.method_id: row for row in _CONTRACTS})
_LEGACY_METHOD_BY_CODE = {
    2: "capacity.mpm_specified_cf_legacy",
    8: "capacity.engelund_hansen_constant_chezy_legacy",
}
FORMULA_BY_LEGACY_CODE = MappingProxyType({
    row.legacy_code: row for row in _CONTRACTS
    if row.legacy_code is not None and (row.legacy_code not in _LEGACY_METHOD_BY_CODE or
    row.method_id == _LEGACY_METHOD_BY_CODE[row.legacy_code])
})
_DELFT_METHOD_BY_SELECTOR = {
    1: "capacity.engelund_hansen",
    2: "capacity.mpm",
}
FORMULA_BY_REFERENCE_SELECTOR = MappingProxyType({
    row.reference_selector: row for row in _CONTRACTS
    if row.reference_selector is not None and (row.reference_selector not in _DELFT_METHOD_BY_SELECTOR or
    row.method_id == _DELFT_METHOD_BY_SELECTOR[row.reference_selector])
})


class CapacityParameterError(ValueError):
    pass


def capacity_parameter_document() -> dict:
    """Return the restart-relevant formula-parameter schema."""
    document = {
        "schema": 1,
        "contracts": [asdict(row) for row in sorted(
            _CONTRACTS, key=lambda value: value.method_id)],
    }
    return json.loads(json.dumps(document, sort_keys=True, separators=(",", ":"),
                                 ensure_ascii=True, allow_nan=False))


def capacity_parameter_catalog_id() -> str:
    encoded = json.dumps(capacity_parameter_document(), sort_keys=True,
                         separators=(",", ":"), ensure_ascii=True,
                         allow_nan=False).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def formula_contract(identifier: str | int) -> CapacityFormulaContract:
    try:
        if isinstance(identifier, str):
            return FORMULA_CONTRACTS[identifier]
        # Python considers ``2.0 == 2`` (and ``True == 1``) when indexing a
        # dictionary.  Formula selectors are an integer wire contract, so do
        # not let numerically equal floats or booleans silently select a law.
        if type(identifier) is not int:
            raise KeyError(identifier)
        return FORMULA_BY_LEGACY_CODE[identifier]
    except (KeyError, TypeError) as error:
        raise CapacityParameterError(
            f"Unknown current-capacity formula: {identifier!r}"
        ) from error


def resolve_capacity_parameters(
    identifier: str | int,
    supplied: Mapping[str, float | int],
) -> Mapping[str, float | int]:
    """Validate and complete one formula's source-keyword parameter map.

    Required values are never guessed.  Optional values are inserted from the
    locked FLOW defaults.  The returned mapping has reference-source order and
    is immutable so it can be hashed directly into project/run identities.
    """
    contract = formula_contract(identifier)
    if not isinstance(supplied, Mapping):
        raise CapacityParameterError("Capacity parameters must be a mapping")
    expected = {row.keyword for row in contract.parameters}
    extra = tuple(sorted(set(supplied) - expected))
    if extra:
        raise CapacityParameterError(
            f"Unexpected {contract.method_id} parameter(s): {', '.join(extra)}"
        )
    resolved: dict[str, float | int] = {}
    for specification in contract.parameters:
        if specification.keyword in supplied:
            value = supplied[specification.keyword]
        elif specification.required:
            raise CapacityParameterError(
                f"Missing required {contract.method_id} parameter: {specification.keyword}"
            )
        else:
            value = specification.default
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise CapacityParameterError(
                f"{specification.keyword} must be a finite numeric value"
            )
        numeric = float(value)
        if not math.isfinite(numeric):
            raise CapacityParameterError(
                f"{specification.keyword} must be a finite numeric value"
            )
        if specification.value_kind == "integer":
            integer = int(numeric)
            if numeric != integer or (specification.allowed and integer not in specification.allowed):
                choices = f" in {specification.allowed}" if specification.allowed else ""
                raise CapacityParameterError(
                    f"{specification.keyword} must be an integer{choices}"
                )
            resolved[specification.keyword] = integer
        else:
            resolved[specification.keyword] = numeric
        if specification.minimum is not None:
            below_minimum = numeric < specification.minimum
            at_excluded_minimum = (
                not specification.minimum_inclusive
                and numeric == specification.minimum
            )
            if below_minimum or at_excluded_minimum:
                relation = ">=" if specification.minimum_inclusive else ">"
                raise CapacityParameterError(
                    f"{specification.keyword} must be {relation} "
                    f"{specification.minimum}"
                )
    return MappingProxyType(resolved)


def canonical_parameter_items(
    identifier: str | int,
    supplied: Mapping[str, float | int],
) -> tuple[tuple[str, float | int], ...]:
    """Return stable source-ordered items suitable for identity hashing."""
    return tuple(resolve_capacity_parameters(identifier, supplied).items())
