"""Typed v67 adapter contract for current-only direct transport producers.

The four methods in this module do not share Delft3D's numeric selector
namespace and they do not share a single constitutive output shape.  The
adapter therefore gives them collision-free wire identities and normalizes
their *already solid-mass-valued* outputs to the generic direct-vector route
ABI.  It deliberately performs no density conversion and no secondary-flow
rotation.

This is an additive integration seam.  The established v66 constitutive
producers and route composer remain the numerical implementations and the
locked Delft3D checkout remains a read-only source oracle.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from types import MappingProxyType
from typing import Mapping

from .direct_vector_route_composer_v66 import (
    DirectVectorOutputKind,
    DirectVectorRouteResult,
    METHOD_BIJKER,
    METHOD_VAN_RIJN_1993,
    METHOD_VAN_THIEL_VAN_RIJN_2008,
    METHOD_XBEACH_SOULSBY_VAN_RIJN,
    compose_direct_vector_route_v66,
)


SCHEMA = 1
SCOPE_ID = "braids-q3-h6-direct-producer-adapter-v67-v1"
LOCKED_REFERENCE_COMMIT = "c05ed672f0a7fe81c5b9867ef354cfdd53c25379"

METHOD_VAN_RIJN_1993_CURRENT = "capacity.van_rijn_1993"
METHOD_BIJKER_CURRENT = "capacity.bijker_current"
METHOD_VAN_THIEL_VAN_RIJN_2008_CURRENT = (
    "capacity.van_thiel_van_rijn_2008_current"
)
METHOD_XBEACH_SOULSBY_VAN_RIJN_CURRENT = (
    "capacity.xbeach_soulsby_van_rijn_current"
)

# These are v67 wire identities, never legacy BRAIDS law IDs and never
# Delft3D TraFrm selectors.  Keeping the reference selector in the low digits
# is diagnostic only; the namespace itself prevents the law-5 collision with
# Wilcock-Crowe and the law-1 collision with the frozen M1 dispatcher.
WIRE_VAN_RIJN_1993_CURRENT = 67001
WIRE_BIJKER_CURRENT = 67005
WIRE_VAN_THIEL_VAN_RIJN_2008_CURRENT = 67019
WIRE_XBEACH_SOULSBY_VAN_RIJN_CURRENT = 67020

NORMALIZED_ABI = (
    "raw_bed_mass_x",
    "raw_bed_mass_y",
    "raw_suspended_mass_x",
    "raw_suspended_mass_y",
    "raw_equilibrium_mass_concentration",
)
FRACTION_OWNER_ROUTE_COMPOSER = "direct_route_composer_v66"
FRACTION_OWNER_DOWNSTREAM_FACE_EXNER = "downstream_face_exner_lifecycle"

_LOCKED_ROOT = "resources/Delft3D-main/src/utils_gpl/morphology/packages/"
LOCKED_SOURCES = MappingProxyType({
    "bedbc1993": (
        _LOCKED_ROOT + "morphology_kernel/src/bedbc1993.f90",
        "7fd7b619a111e3ee99ad6625097882a0bf3baa0a0f6f6d39f939671390fd73d5",
    ),
    "calseddf1993": (
        _LOCKED_ROOT + "morphology_kernel/src/calseddf1993.f90",
        "2c178c5d9e083e3b494f600cefd088c4d3d917f6d1b43f004d7bc98c064f10b5",
    ),
    "tram1": (
        _LOCKED_ROOT + "morphology_kernel/src/tram1.f90",
        "33f5f60439c8b98facc7d5d1431e8f4a06fc37fe7b14d531a8fa9d9be6f74246",
    ),
    "bedtr1993": (
        _LOCKED_ROOT + "morphology_kernel/src/bedtr1993.f90",
        "d330765e7b833b1402b66a524784157019d04737c979967fdc3b501e158c9e66",
    ),
    "eqtran": (
        _LOCKED_ROOT + "morphology_kernel/src/eqtran.f90",
        "d2e8601cff68884f51ee3dcb0cd2fb52d9f0d676b1baaa3283c68913d4778388",
    ),
    "tranb5": (
        _LOCKED_ROOT + "morphology_kernel/src/tranb5.f90",
        "ca5933ccda6592174c627c81d2d10e977064dd47f96560d349b1a3fca50a2070",
    ),
    "trab19": (
        _LOCKED_ROOT + "morphology_kernel/src/trab19.f90",
        "bfc530b573b95c4467a9de4bd34e25757f3990d489426db995b28d4b7ad34c63",
    ),
    "trab20": (
        _LOCKED_ROOT + "morphology_kernel/src/trab20.f90",
        "a7cbf17f232fc6edae5a1cf5f7c9b2ca1e1e0626aee6dbb5270efc1c245e4d19",
    ),
})


@dataclass(frozen=True)
class DirectProducerContractV67:
    method_id: str
    internal_wire_id: int
    reference_selector: int
    producer_id: str
    composer_method: str
    output_kind: DirectVectorOutputKind
    source_routines: tuple[str, ...]
    current_only: bool = True
    external_secondary_rotation: bool = False
    density_conversion_count: int = 0


_ROWS = (
    DirectProducerContractV67(
        METHOD_VAN_RIJN_1993_CURRENT,
        WIRE_VAN_RIJN_1993_CURRENT,
        -1,
        "braids_vr1993_current_producer_v66",
        METHOD_VAN_RIJN_1993,
        DirectVectorOutputKind.EQUILIBRIUM_CONCENTRATION,
        ("bedbc1993", "calseddf1993", "tram1", "bedtr1993", "eqtran"),
    ),
    DirectProducerContractV67(
        METHOD_BIJKER_CURRENT,
        WIRE_BIJKER_CURRENT,
        5,
        "braids_bijker_current_v66",
        METHOD_BIJKER,
        DirectVectorOutputKind.EXPLICIT_SUSPENDED_VECTOR,
        ("tranb5", "eqtran"),
    ),
    DirectProducerContractV67(
        METHOD_VAN_THIEL_VAN_RIJN_2008_CURRENT,
        WIRE_VAN_THIEL_VAN_RIJN_2008_CURRENT,
        19,
        "braids_van_thiel_van_rijn_current_v66",
        METHOD_VAN_THIEL_VAN_RIJN_2008,
        DirectVectorOutputKind.EQUILIBRIUM_CONCENTRATION,
        ("trab19", "eqtran"),
    ),
    DirectProducerContractV67(
        METHOD_XBEACH_SOULSBY_VAN_RIJN_CURRENT,
        WIRE_XBEACH_SOULSBY_VAN_RIJN_CURRENT,
        20,
        "braids_xbeach_soulsby_van_rijn_current_v66",
        METHOD_XBEACH_SOULSBY_VAN_RIJN,
        DirectVectorOutputKind.EQUILIBRIUM_CONCENTRATION,
        ("trab20", "eqtran"),
    ),
)

CONTRACTS = MappingProxyType({row.method_id: row for row in _ROWS})
CONTRACT_BY_WIRE_ID = MappingProxyType({row.internal_wire_id: row for row in _ROWS})
CONTRACT_BY_REFERENCE_SELECTOR = MappingProxyType({
    row.reference_selector: row for row in _ROWS
})


class DirectProducerAdapterV67Error(ValueError):
    """A method identity or normalized producer channel is invalid."""


def direct_producer_contract_v67(
    identifier: str | int,
) -> DirectProducerContractV67:
    """Resolve only a canonical method string or a v67 wire ID.

    Delft selectors are intentionally not accepted here.  Callers needing to
    inspect source provenance must use the explicitly named selector resolver.
    """
    try:
        if type(identifier) is str:
            return CONTRACTS[identifier]
        if type(identifier) is int:
            return CONTRACT_BY_WIRE_ID[identifier]
        raise KeyError(identifier)
    except KeyError as error:
        raise DirectProducerAdapterV67Error(
            f"unknown direct-producer v67 identity: {identifier!r}"
        ) from error


def direct_producer_contract_by_reference_selector_v67(
    selector: int,
) -> DirectProducerContractV67:
    if type(selector) is not int:
        raise DirectProducerAdapterV67Error(
            f"Delft reference selector must be an exact integer: {selector!r}"
        )
    try:
        return CONTRACT_BY_REFERENCE_SELECTOR[selector]
    except KeyError as error:
        raise DirectProducerAdapterV67Error(
            f"unknown Delft reference selector: {selector!r}"
        ) from error


def direct_producer_semantic_document_v67() -> dict:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "locked_reference_commit": LOCKED_REFERENCE_COMMIT,
        "integer_namespace": "v67_internal_wire_id_not_legacy_law_or_TraFrm",
        "normalized_abi": list(NORMALIZED_ABI),
        "methods": [
            {
                **asdict(row),
                "output_kind": int(row.output_kind),
                "source_routines": list(row.source_routines),
                "locked_sources": {
                    name: {"path": LOCKED_SOURCES[name][0],
                           "sha256": LOCKED_SOURCES[name][1]}
                    for name in row.source_routines
                },
            }
            for row in sorted(_ROWS, key=lambda item: item.internal_wire_id)
        ],
        "invariants": {
            "current_only": True,
            "wave_or_extra_physics": "fail_closed",
            "external_secondary_rotation": False,
            "producer_output_units": "solid_mass",
            "adapter_density_conversion_count": 0,
            "bed_sus_application": "route_composer_exactly_once",
            "active_fraction_application": "route_composer_exactly_once",
            "face_lifecycle_active_fraction": (
                "deferred_with_internal_exact_one_never_caller_emulated"
            ),
            "automatic_cpu_fallback": False,
        },
    }


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise DirectProducerAdapterV67Error(
            "direct-producer v67 identity is not canonical JSON"
        ) from error


SEMANTIC_ID = hashlib.sha256(
    _canonical(direct_producer_semantic_document_v67())
).hexdigest()


def direct_producer_identity_document_v67(identifier: str | int) -> dict:
    contract = direct_producer_contract_v67(identifier)
    body = {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "method_id": contract.method_id,
        "internal_wire_id": contract.internal_wire_id,
        "reference_selector": contract.reference_selector,
        "producer_id": contract.producer_id,
        "composer_method": contract.composer_method,
        "output_kind": int(contract.output_kind),
        "normalized_abi": list(NORMALIZED_ABI),
        "current_only": contract.current_only,
        "external_secondary_rotation": contract.external_secondary_rotation,
        "density_conversion_count": contract.density_conversion_count,
        "locked_sources": {
            name: {"path": LOCKED_SOURCES[name][0],
                   "sha256": LOCKED_SOURCES[name][1]}
            for name in contract.source_routines
        },
    }
    return {**body, "fingerprint": hashlib.sha256(_canonical(body)).hexdigest()}


def load_direct_producer_identity_v67(value: Mapping[str, object]) -> dict:
    if not isinstance(value, Mapping):
        raise DirectProducerAdapterV67Error(
            "direct-producer v67 identity must be an object"
        )
    method = value.get("method_id")
    rebuilt = direct_producer_identity_document_v67(method)  # type: ignore[arg-type]
    if dict(value) != rebuilt or _canonical(value) != _canonical(rebuilt):
        raise DirectProducerAdapterV67Error(
            "direct-producer v67 identity is stale or forged"
        )
    return rebuilt


@dataclass(frozen=True)
class NormalizedDirectProducerResultV67:
    """Scalar CPU-side representation of the shared producer/route ABI."""

    method_id: str
    internal_wire_id: int
    raw_bed_mass_x: float
    raw_bed_mass_y: float
    raw_suspended_mass_x: float
    raw_suspended_mass_y: float
    raw_equilibrium_mass_concentration: float
    route: DirectVectorRouteResult
    surface_fraction_application_count: int
    surface_fraction_owner: str


def normalize_direct_producer_v67(
    *,
    method: str | int,
    route: str,
    depth: float,
    velocity_x: float,
    velocity_y: float,
    raw_bed_mass_x: float,
    raw_bed_mass_y: float,
    raw_suspended_mass_x: float = 0.0,
    raw_suspended_mass_y: float = 0.0,
    raw_equilibrium_mass_concentration: float = 0.0,
    bed_factor: float = 1.0,
    suspended_factor: float = 1.0,
    active_fraction: float = 1.0,
    speed_epsilon: float = 1.0e-6,
) -> NormalizedDirectProducerResultV67:
    """Normalize an existing native producer output without changing physics."""
    contract = direct_producer_contract_v67(method)
    composed = compose_direct_vector_route_v66(
        method=contract.composer_method,
        route=route,
        depth=depth,
        velocity_x=velocity_x,
        velocity_y=velocity_y,
        raw_bed_mass_x=raw_bed_mass_x,
        raw_bed_mass_y=raw_bed_mass_y,
        raw_suspended_mass_x=raw_suspended_mass_x,
        raw_suspended_mass_y=raw_suspended_mass_y,
        raw_equilibrium_mass_concentration=(
            raw_equilibrium_mass_concentration
        ),
        bed_factor=bed_factor,
        suspended_factor=suspended_factor,
        active_fraction=active_fraction,
        speed_epsilon=speed_epsilon,
    )
    return NormalizedDirectProducerResultV67(
        contract.method_id,
        contract.internal_wire_id,
        float(raw_bed_mass_x),
        float(raw_bed_mass_y),
        float(raw_suspended_mass_x),
        float(raw_suspended_mass_y),
        float(raw_equilibrium_mass_concentration),
        composed,
        1,
        FRACTION_OWNER_ROUTE_COMPOSER,
    )


def normalize_direct_producer_for_face_v67(
    *,
    method: str | int,
    route: str,
    depth: float,
    velocity_x: float,
    velocity_y: float,
    raw_bed_mass_x: float,
    raw_bed_mass_y: float,
    raw_suspended_mass_x: float = 0.0,
    raw_suspended_mass_y: float = 0.0,
    raw_equilibrium_mass_concentration: float = 0.0,
    bed_factor: float = 1.0,
    suspended_factor: float = 1.0,
    speed_epsilon: float = 1.0e-6,
) -> NormalizedDirectProducerResultV67:
    """Compose unfractioned channels for the downstream face/Exner owner.

    There is intentionally no ``active_fraction`` argument.  An exact one is
    supplied internally, and the returned ownership metadata proves that the
    physical surface fraction has not yet been applied.
    """
    result = normalize_direct_producer_v67(
        method=method,
        route=route,
        depth=depth,
        velocity_x=velocity_x,
        velocity_y=velocity_y,
        raw_bed_mass_x=raw_bed_mass_x,
        raw_bed_mass_y=raw_bed_mass_y,
        raw_suspended_mass_x=raw_suspended_mass_x,
        raw_suspended_mass_y=raw_suspended_mass_y,
        raw_equilibrium_mass_concentration=(
            raw_equilibrium_mass_concentration
        ),
        bed_factor=bed_factor,
        suspended_factor=suspended_factor,
        active_fraction=1.0,
        speed_epsilon=speed_epsilon,
    )
    return replace(
        result,
        surface_fraction_application_count=0,
        surface_fraction_owner=FRACTION_OWNER_DOWNSTREAM_FACE_EXNER,
    )


__all__ = [
    "CONTRACTS", "CONTRACT_BY_REFERENCE_SELECTOR", "CONTRACT_BY_WIRE_ID",
    "DirectProducerAdapterV67Error", "DirectProducerContractV67",
    "FRACTION_OWNER_DOWNSTREAM_FACE_EXNER", "FRACTION_OWNER_ROUTE_COMPOSER",
    "LOCKED_REFERENCE_COMMIT", "LOCKED_SOURCES", "METHOD_VAN_RIJN_1993_CURRENT",
    "METHOD_BIJKER_CURRENT", "METHOD_VAN_THIEL_VAN_RIJN_2008_CURRENT",
    "METHOD_XBEACH_SOULSBY_VAN_RIJN_CURRENT", "NORMALIZED_ABI",
    "NormalizedDirectProducerResultV67", "SCHEMA", "SCOPE_ID", "SEMANTIC_ID",
    "WIRE_VAN_RIJN_1993_CURRENT", "WIRE_BIJKER_CURRENT",
    "WIRE_VAN_THIEL_VAN_RIJN_2008_CURRENT",
    "WIRE_XBEACH_SOULSBY_VAN_RIJN_CURRENT",
    "direct_producer_contract_by_reference_selector_v67",
    "direct_producer_contract_v67", "direct_producer_identity_document_v67",
    "direct_producer_semantic_document_v67", "load_direct_producer_identity_v67",
    "normalize_direct_producer_for_face_v67", "normalize_direct_producer_v67",
]
