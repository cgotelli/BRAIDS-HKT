"""Authenticated bounded profile for six scalar Delft3D4 FLOW capacities.

The profile is backend-neutral: native CPU and CUDA must consume the same
resolved FLOW point state and coefficient bytes.  It is intentionally an
additive qualification identity, not a live/public support declaration.
"""
from __future__ import annotations

import hashlib
import json
import re
from types import MappingProxyType
from typing import Mapping

from .capacity_parameters import (
    canonical_parameter_items,
    formula_contract,
)


SCHEMA = 1
SCOPE_ID = "braids-q3-scalar-flow-capacity-v63-bounded-point-v1"
_SHA = re.compile(r"[0-9a-f]{64}\Z")

_SOURCE_ROOT = "resources/Delft3D-main/"
_COMMON_SOURCES = MappingProxyType({
    "eqtran": (
        "src/utils_gpl/morphology/packages/morphology_kernel/src/eqtran.f90",
        "d2e8601cff68884f51ee3dcb0cd2fb52d9f0d676b1baaa3283c68913d4778388",
    ),
    "rdtrafrm": (
        "src/utils_gpl/morphology/packages/morphology_io/src/rdtrafrm.f90",
        "0a2542980bc9e9e03e46b2abb8b06eda2fea3e28a67c78c4e61b619572154ea7",
    ),
})


def _law(method_id: str, formula_code: int, selector: int, routine: str,
         source: str, source_sha256: str, point_inputs: tuple[str, ...],
         mixture_contract: str, shear_contract: str,
         roughness_required: bool, generic_hiding: bool) -> dict:
    return {
        "method_id": method_id,
        "formula_code": formula_code,
        "reference_selector": selector,
        "routine": routine,
        "source": source,
        "source_sha256": source_sha256,
        "point_inputs": point_inputs,
        "mixture_contract": mixture_contract,
        "shear_contract": shear_contract,
        "roughness_required": roughness_required,
        "generic_hiding": generic_hiding,
    }


_LAWS = (
    _law(
        "capacity.van_rijn_1984_modified", 12, 7, "tranb7",
        "src/utils_gpl/morphology/packages/morphology_kernel/src/tranb7.f90",
        "6d0d5ebaf9dfd508296bd81b89a61f64ca3952c1d9a151e0af5b3a2c777ab09a",
        ("effective_velocity", "depth", "fraction_d50", "RP_D90MX",
         "water_viscosity", "mud_fraction", "settling_velocity"),
        "RP_D90MX_and_RP_MUDFR_resolved_upstream",
        "internal_roughness_partition_no_RP_TAUB",
        False, False,
    ),
    _law(
        "capacity.wang_fredsoe", 14, 13, "tran9t",
        "src/utils_gpl/morphology/packages/morphology_kernel/src/tran9t.f90",
        "c9664a4e1d30befdc9fba28a554d84dae8c38dfc27ce9a427b9740e1b1a6a918",
        ("effective_velocity", "depth", "fraction_d50", "RP_D90MX",
         "resolved_chezy", "RP_USTAR", "VicMol"),
        "RP_D90MX_resolved_upstream",
        "RP_USTAR_total_current_chain_distinct_from_RP_TAUB",
        True, False,
    ),
    _law(
        "capacity.ashida_michiue_generalized", 3, 14, "trab14",
        "src/utils_gpl/morphology/packages/morphology_kernel/src/trab14.f90",
        "7d93f7b0e4d3f9d81c3d7c77c49cc3ae63bc5aec8f4149c6e0c67a7565be1b34",
        ("effective_velocity", "fraction_d50", "resolved_chezy", "HIDEXP"),
        "not_consumed",
        "resolved_chezy_no_RP_TAUB",
        True, True,
    ),
    _law(
        "capacity.wilcock_crowe", 5, 16, "trabwc",
        "src/utils_gpl/morphology/packages/morphology_kernel/src/trabwc.f90",
        "f30fcb04f3534586b748c92f5f29a74b9facb1bcc61b048b287c0605407e6b31",
        ("effective_velocity", "fraction_d50", "resolved_chezy", "RP_TAUB",
         "RP_DG", "RP_SNDFR"),
        "RP_DG_and_RP_SNDFR_resolved_upstream",
        "RP_TAUB_for_phi_total_chezy_for_transport_scale",
        True, False,
    ),
    _law(
        "capacity.wilcock_crowe_modified", 6, 17, "trabwc2",
        "src/utils_gpl/morphology/packages/morphology_kernel/src/trabwc2.f90",
        "5b6cbc7a5f12e15c00f02c00a6daed4d7fd6f35aa7ef56ed499ed30a858c1515",
        ("effective_velocity", "fraction_d50", "resolved_chezy", "RP_TAUB",
         "RP_DG", "RP_DGSD"),
        "RP_DG_and_true_geometric_RP_DGSD_resolved_upstream",
        "RP_TAUB_for_phi_total_chezy_for_transport_scale",
        True, False,
    ),
    _law(
        "capacity.gaeuman", 7, 18, "trabg",
        "src/utils_gpl/morphology/packages/morphology_kernel/src/trabg.f90",
        "51dc9db8a102fdaef997bd0d20329125c598bc4a5feb0a11019d995315939a11",
        ("effective_velocity", "fraction_d50", "resolved_chezy", "RP_TAUB",
         "RP_DG", "RP_DGSD"),
        "RP_DG_and_true_geometric_RP_DGSD_resolved_upstream",
        "RP_TAUB_for_phi_total_chezy_for_transport_scale",
        True, False,
    ),
)

CONTRACTS = MappingProxyType({row["method_id"]: MappingProxyType(row)
                              for row in _LAWS})
CONTRACT_BY_SELECTOR = MappingProxyType({row["reference_selector"]:
                                         MappingProxyType(row)
                                         for row in _LAWS})

_ROUGHNESS = frozenset((
    "roughness.chezy", "roughness.manning", "roughness.white_colebrook",
    "roughness.white_colebrook_waqua",
))
_SHEAR = frozenset((
    "shear.total_roughness", "shear.soulsby_clarke_skin",
))
_HIDING = frozenset((
    "hiding.none", "hiding.egiazaroff", "hiding.ashida_michiue",
    "hiding.parker_klingeman", "hiding.wu_wang_jia",
))


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def scalar_flow_semantic_document() -> dict:
    laws = []
    for row in sorted(_LAWS, key=lambda item: item["reference_selector"]):
        contract = formula_contract(row["method_id"])
        laws.append({
            **{key: (list(value) if isinstance(value, tuple) else value)
               for key, value in row.items()},
            "parameters": [parameter.keyword for parameter in contract.parameters],
        })
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "point_scope": "pre_active_fraction_capacity_only",
        "output_units": "solid_mass_flux_kg_per_m_s",
        "output_rows": [
            "bed_x", "bed_y", "suspended_x", "suspended_y",
            "absolute_shields", "reference_shields", "stress_ratio",
        ],
        "fraction_application": "downstream_exactly_once_not_in_point_operator",
        "upstream_preprocessing_qualified": False,
        "live_dispatch_qualified": False,
        "common_sources": {
            name: {"path": value[0], "sha256": value[1]}
            for name, value in sorted(_COMMON_SOURCES.items())
        },
        "laws": laws,
    }


SEMANTIC_ID = hashlib.sha256(_canonical(scalar_flow_semantic_document())).hexdigest()


def scalar_flow_execution_document(*, method_id: str,
        parameters: Mapping[str, float | int], roughness_method_id: str,
        sediment_shear_method_id: str, hiding_method_id: str,
        capacity_profile_fingerprint: str, forcing_fingerprint: str) -> dict:
    """Build one canonical CPU/GPU-common bounded execution identity."""
    try:
        row = CONTRACTS[method_id]
    except (KeyError, TypeError) as error:
        raise ValueError(f"Unsupported scalar FLOW method: {method_id!r}") from error
    if (not isinstance(capacity_profile_fingerprint, str) or
            _SHA.fullmatch(capacity_profile_fingerprint) is None or
            not isinstance(forcing_fingerprint, str) or
            _SHA.fullmatch(forcing_fingerprint) is None):
        raise ValueError("Scalar FLOW execution requires two SHA-256 dependencies")
    if row["roughness_required"]:
        if roughness_method_id not in _ROUGHNESS:
            raise ValueError("Scalar FLOW method requires a typed C/n/ks roughness")
    elif roughness_method_id != "roughness.not_consumed":
        raise ValueError("TraFrm 7 does not consume resolved hydraulic roughness")
    if row["reference_selector"] in (16, 17, 18):
        if sediment_shear_method_id not in _SHEAR:
            raise ValueError("Wilcock-family method requires an RP_TAUB shear identity")
    elif sediment_shear_method_id != "shear.not_consumed":
        raise ValueError("Selected scalar FLOW method does not consume RP_TAUB")
    if row["generic_hiding"]:
        if hiding_method_id not in _HIDING:
            raise ValueError("TraFrm 14 requires an explicit generic hiding identity")
    elif hiding_method_id != "hiding.intrinsic_or_not_consumed":
        raise ValueError("Selected method does not consume generic hiding")

    resolved = canonical_parameter_items(method_id, parameters)
    body = {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "qualification": "bounded_point_operator_not_live",
        "method_id": method_id,
        "formula_code": row["formula_code"],
        "reference_selector": row["reference_selector"],
        "source_routine": row["routine"],
        "source_path": _SOURCE_ROOT+row["source"],
        "source_sha256": row["source_sha256"],
        "parameters": [{"keyword": name, "value": value}
                       for name, value in resolved],
        "roughness_method_id": roughness_method_id,
        "sediment_shear_method_id": sediment_shear_method_id,
        "hiding_method_id": hiding_method_id,
        "mixture_contract": row["mixture_contract"],
        "shear_contract": row["shear_contract"],
        "capacity_profile_fingerprint": capacity_profile_fingerprint,
        "forcing_fingerprint": forcing_fingerprint,
    }
    return {**body, "fingerprint": hashlib.sha256(_canonical(body)).hexdigest()}


def load_scalar_flow_execution(value: Mapping[str, object]) -> dict:
    if not isinstance(value, Mapping):
        raise ValueError("Scalar FLOW execution identity must be an object")
    expected = {
        "schema", "scope_id", "semantic_id", "qualification", "method_id",
        "formula_code", "reference_selector", "source_routine", "source_path",
        "source_sha256", "parameters", "roughness_method_id",
        "sediment_shear_method_id", "hiding_method_id", "mixture_contract",
        "shear_contract", "capacity_profile_fingerprint", "forcing_fingerprint",
        "fingerprint",
    }
    if set(value) != expected:
        raise ValueError("Scalar FLOW execution identity inventory differs")
    parameters = value.get("parameters")
    if not isinstance(parameters, list) or any(
            not isinstance(item, Mapping) or set(item) != {"keyword", "value"}
            for item in parameters):
        raise ValueError("Scalar FLOW parameter identity is malformed")
    supplied = {item["keyword"]: item["value"] for item in parameters}
    rebuilt = scalar_flow_execution_document(
        method_id=value.get("method_id"), parameters=supplied,
        roughness_method_id=value.get("roughness_method_id"),
        sediment_shear_method_id=value.get("sediment_shear_method_id"),
        hiding_method_id=value.get("hiding_method_id"),
        capacity_profile_fingerprint=value.get("capacity_profile_fingerprint"),
        forcing_fingerprint=value.get("forcing_fingerprint"),
    )
    if dict(value) != rebuilt:
        raise ValueError("Scalar FLOW execution identity or fingerprint differs")
    return rebuilt


__all__ = [
    "SCHEMA", "SCOPE_ID", "SEMANTIC_ID", "CONTRACTS",
    "CONTRACT_BY_SELECTOR", "scalar_flow_semantic_document",
    "scalar_flow_execution_document", "load_scalar_flow_execution",
]
