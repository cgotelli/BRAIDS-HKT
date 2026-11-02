"""Authenticated v67 execution identity for the six direct-vector FLOW laws.

This module is an additive control-plane envelope.  It does not implement a
transport formula.  Instead it binds one canonical formula, parameter
identity and channel route to every sediment fraction and binds that physical
selection to the CPU or ``implicit_fwave_midpoint`` GPU implementation bytes.

The accepted-state lifecycle is intentionally narrower than the constitutive
inventory: ``TRA_BEDLOAD`` fractions can own an accepted Exner state, while
``TRA_COMBINE`` is recorded as a diagnostic-only Q4 hand-off until suspended
mass is part of the same transaction.  ``TRA_ADVDIFF`` never enters this
direct-capacity path.  No method alias, Delft selector, legacy BRAIDS law ID,
CPU fallback, external direction rotation, or post-producer density
conversion is accepted implicitly.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from types import MappingProxyType

import numpy as np

from .capacity_parameters import (
    canonical_parameter_items,
    capacity_parameter_catalog_id,
)
from .channel_composition import (
    TRA_ADVDIFF,
    TRA_BEDLOAD,
    TRA_COMBINE,
    ChannelCompositionProfile,
    channel_composition_profile_document,
    load_channel_composition_profile,
    resolve_channel_composition_profile,
)
from .direct_current_parameters_v66 import (
    METHOD_BIJKER_CURRENT,
    METHOD_VAN_THIEL_VAN_RIJN_2008_CURRENT,
    METHOD_XBEACH_SOULSBY_VAN_RIJN_CURRENT,
    direct_current_parameter_document_v66,
)
from .direct_producer_adapter_v67 import (
    METHOD_VAN_RIJN_1993_CURRENT,
    direct_producer_identity_document_v67,
)
from .direct_vector_route_composer_v66 import (
    DirectVectorOutputKind,
    METHOD_SOULSBY,
    METHOD_SOULSBY_VAN_RIJN,
    ROUTE_TRA_BEDLOAD,
    ROUTE_TRA_COMBINE,
)


SCHEMA = 1
ENVELOPE_VERSION = 67
SCOPE_ID = "braids-q3-h6-direct-vector-family-live-v67-v1"
MAX_ACCEPTED_STEPS = 20
BACKENDS = frozenset(("cpu", "implicit_fwave_midpoint_gpu"))
_SHA = re.compile(r"[0-9a-f]{64}\Z")

METHOD_SOULSBY_CURRENT = "capacity.soulsby"
METHOD_SOULSBY_VAN_RIJN_CURRENT = "capacity.soulsby_van_rijn"

# Dedicated family wires.  They are neither legacy BRAIDS law IDs nor Delft
# TraFrm selectors.  The remaining four wires are the existing v67 producer
# adapter identities, so all six occupy one non-colliding namespace.
WIRE_SOULSBY_CURRENT = 67012
WIRE_SOULSBY_VAN_RIJN_CURRENT = 67011


@dataclass(frozen=True)
class DirectVectorFamilyMethodV67:
    method_id: str
    internal_wire_id: int
    identity_tag: str
    reference_selector: int
    producer_id: str
    composer_method: str
    output_kind: int
    parameter_schema: str
    reference_source_names: tuple[str, ...]
    implementation_group: str


_METHOD_ROWS = (
    DirectVectorFamilyMethodV67(
        METHOD_SOULSBY_CURRENT, WIRE_SOULSBY_CURRENT,
        "direct_vector_family_v67:67012", 12,
        "braids_soulsby_current_producer_v66", METHOD_SOULSBY,
        int(DirectVectorOutputKind.ZERO_NONBED), "capacity_parameters_v1",
        ("trab12",), "soulsby",
    ),
    DirectVectorFamilyMethodV67(
        METHOD_SOULSBY_VAN_RIJN_CURRENT, WIRE_SOULSBY_VAN_RIJN_CURRENT,
        "direct_vector_family_v67:67011", 11,
        "braids_soulsby_van_rijn_current_producer_v66",
        METHOD_SOULSBY_VAN_RIJN,
        int(DirectVectorOutputKind.EXPLICIT_SUSPENDED_VECTOR),
        "capacity_parameters_v1", ("trab11",), "soulsby_van_rijn",
    ),
    DirectVectorFamilyMethodV67(
        METHOD_VAN_RIJN_1993_CURRENT, 67001,
        "direct_vector_family_v67:67001", -1,
        "braids_vr1993_current_producer_v66", "van_rijn_1993_current",
        int(DirectVectorOutputKind.EQUILIBRIUM_CONCENTRATION),
        "capacity_parameters_v1",
        ("bedbc1993", "calseddf1993", "tram1", "bedtr1993"),
        "van_rijn_1993",
    ),
    DirectVectorFamilyMethodV67(
        METHOD_BIJKER_CURRENT, 67005,
        "direct_vector_family_v67:67005", 5,
        "braids_bijker_current_v66", "bijker_current",
        int(DirectVectorOutputKind.EXPLICIT_SUSPENDED_VECTOR),
        "direct_current_parameters_v66", ("tranb5", "fgyint", "termfy", "termgy"),
        "delft_direct_current",
    ),
    DirectVectorFamilyMethodV67(
        METHOD_VAN_THIEL_VAN_RIJN_2008_CURRENT, 67019,
        "direct_vector_family_v67:67019", 19,
        "braids_van_thiel_van_rijn_current_v66",
        "van_thiel_van_rijn_2008_current",
        int(DirectVectorOutputKind.EQUILIBRIUM_CONCENTRATION),
        "direct_current_parameters_v66", ("trab19", "sed_support_routines"),
        "delft_direct_current",
    ),
    DirectVectorFamilyMethodV67(
        METHOD_XBEACH_SOULSBY_VAN_RIJN_CURRENT, 67020,
        "direct_vector_family_v67:67020", 20,
        "braids_xbeach_soulsby_van_rijn_current_v66",
        "xbeach_soulsby_van_rijn_current",
        int(DirectVectorOutputKind.EQUILIBRIUM_CONCENTRATION),
        "direct_current_parameters_v66", ("trab20", "sed_support_routines"),
        "delft_direct_current",
    ),
)

METHODS = MappingProxyType({row.method_id: row for row in _METHOD_ROWS})
METHODS_BY_WIRE = MappingProxyType({row.internal_wire_id: row for row in _METHOD_ROWS})
SUPPORTED_METHODS = frozenset(METHODS)


class DirectVectorFamilyLiveV67Error(ValueError):
    """A v67 method selection, receipt, or accepted state is invalid."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise DirectVectorFamilyLiveV67Error(
            "Q3-H6/v67 document is not canonical JSON"
        ) from error


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _finite(value: object, name: str, *, positive: bool = False,
            nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DirectVectorFamilyLiveV67Error(
            f"Q3-H6/v67 {name} must be finite numeric data"
        )
    result = float(value)
    if (not math.isfinite(result) or (positive and result <= 0.0) or
            (nonnegative and result < 0.0)):
        raise DirectVectorFamilyLiveV67Error(
            f"Q3-H6/v67 {name} is outside its finite domain"
        )
    return 0.0 if result == 0.0 else result


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise DirectVectorFamilyLiveV67Error(
            f"Q3-H6/v67 {name} must be a lowercase SHA-256 fingerprint"
        )
    return value


def _sequence(value: object, name: str, *, length: int | None = None,
              positive: bool = False) -> list[float]:
    if (not isinstance(value, Sequence) or
            isinstance(value, (str, bytes, bytearray))):
        raise DirectVectorFamilyLiveV67Error(
            f"Q3-H6/v67 {name} must be an ordered sequence"
        )
    result = [
        _finite(item, f"{name}[{index}]", positive=positive)
        for index, item in enumerate(value)
    ]
    if not result or (length is not None and len(result) != length):
        raise DirectVectorFamilyLiveV67Error(
            f"Q3-H6/v67 {name} has the wrong number of fractions"
        )
    return result


def _method(identifier: object) -> DirectVectorFamilyMethodV67:
    # Only canonical strings are accepted at configuration boundaries.  A
    # numeric value is never guessed to be a legacy law or Delft selector.
    if type(identifier) is not str:
        raise DirectVectorFamilyLiveV67Error(
            "Q3-H6/v67 method identities must be canonical strings"
        )
    try:
        return METHODS[identifier]
    except KeyError as error:
        raise DirectVectorFamilyLiveV67Error(
            f"Q3-H6/v67 unsupported direct-vector method: {identifier!r}"
        ) from error


def direct_vector_family_parameter_identity_v67(
    method_id: str, supplied: Mapping[str, object],
) -> dict[str, object]:
    """Return one strict, tagged formula-parameter identity."""
    method = _method(method_id)
    if not isinstance(supplied, Mapping):
        raise DirectVectorFamilyLiveV67Error(
            "Q3-H6/v67 formula parameters must be a mapping"
        )
    try:
        if method.parameter_schema == "capacity_parameters_v1":
            rows = [
                {"keyword": keyword, "value": value}
                for keyword, value in canonical_parameter_items(
                    method.method_id, supplied  # type: ignore[arg-type]
                )
            ]
            native = {
                "schema_id": "capacity_parameters_v1",
                "catalog_id": capacity_parameter_catalog_id(),
                "parameters": rows,
            }
        else:
            native = {
                "schema_id": "direct_current_parameters_v66",
                "document": direct_current_parameter_document_v66(
                    method.method_id, supplied  # type: ignore[arg-type]
                ),
            }
    except (KeyError, TypeError, ValueError) as error:
        raise DirectVectorFamilyLiveV67Error(
            f"Q3-H6/v67 invalid parameters for {method.method_id}"
        ) from error
    body: dict[str, object] = {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "kind": "method_parameter_identity",
        "method_id": method.method_id,
        "internal_wire_id": method.internal_wire_id,
        "identity_tag": method.identity_tag,
        "parameter_schema": method.parameter_schema,
        "native_identity": native,
    }
    return {**body, "fingerprint": _fingerprint(body)}


def load_direct_vector_family_parameter_identity_v67(
    value: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise DirectVectorFamilyLiveV67Error(
            "Q3-H6/v67 parameter identity must be an object"
        )
    try:
        method_id = value["method_id"]
        native = value["native_identity"]
        if not isinstance(native, Mapping):
            raise TypeError("native identity")
        if native.get("schema_id") == "capacity_parameters_v1":
            rows = native["parameters"]
        elif native.get("schema_id") == "direct_current_parameters_v66":
            document = native["document"]
            if not isinstance(document, Mapping):
                raise TypeError("parameter document")
            rows = document["parameters"]
        else:
            raise KeyError("schema_id")
        if (not isinstance(rows, list) or any(
                not isinstance(row, Mapping) or
                set(row) != {"keyword", "value"} for row in rows)):
            raise TypeError("parameter rows")
        supplied = {str(row["keyword"]): row["value"] for row in rows}
        if len(supplied) != len(rows):
            raise ValueError("duplicate parameter keyword")
        rebuilt = direct_vector_family_parameter_identity_v67(
            method_id, supplied  # type: ignore[arg-type]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise DirectVectorFamilyLiveV67Error(
            "Invalid Q3-H6/v67 parameter identity"
        ) from error
    if dict(value) != rebuilt or _canonical(value) != _canonical(rebuilt):
        raise DirectVectorFamilyLiveV67Error(
            "Q3-H6/v67 parameter identity is stale or forged"
        )
    return rebuilt


_REFERENCE_SOURCES = MappingProxyType({
    "eqtran": (
        "resources/Delft3D-main/src/utils_gpl/morphology/packages/"
        "morphology_kernel/src/eqtran.f90",
        "d2e8601cff68884f51ee3dcb0cd2fb52d9f0d676b1baaa3283c68913d4778388",
    ),
    "rdtrafrm": (
        "resources/Delft3D-main/src/utils_gpl/morphology/packages/"
        "morphology_io/src/rdtrafrm.f90",
        "0a2542980bc9e9e03e46b2abb8b06eda2fea3e28a67c78c4e61b619572154ea7",
    ),
    "upwbed": (
        "resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/"
        "flow2d3d_kernel/src/compute_sediment/upwbed.f90",
        "06be422d88821f3f983bc136d511fdaa80ab04c8529640474ab39b478d918158",
    ),
    "trab12": (
        "resources/Delft3D-main/src/utils_gpl/morphology/packages/"
        "morphology_kernel/src/trab12.f90",
        "887ed1a644afaee8764f4590cefee67affc1f5a44876ce8bc5892a4a013fd417",
    ),
    "trab11": (
        "resources/Delft3D-main/src/utils_gpl/morphology/packages/"
        "morphology_kernel/src/trab11.f90",
        "f1bfce663f2851bd8fc1ec96783394562d016e44a8d3d5cbc5138e326782b1ab",
    ),
    "bedbc1993": (
        "resources/Delft3D-main/src/utils_gpl/morphology/packages/"
        "morphology_kernel/src/bedbc1993.f90",
        "7fd7b619a111e3ee99ad6625097882a0bf3baa0a0f6f6d39f939671390fd73d5",
    ),
    "calseddf1993": (
        "resources/Delft3D-main/src/utils_gpl/morphology/packages/"
        "morphology_kernel/src/calseddf1993.f90",
        "2c178c5d9e083e3b494f600cefd088c4d3d917f6d1b43f004d7bc98c064f10b5",
    ),
    "tram1": (
        "resources/Delft3D-main/src/utils_gpl/morphology/packages/"
        "morphology_kernel/src/tram1.f90",
        "33f5f60439c8b98facc7d5d1431e8f4a06fc37fe7b14d531a8fa9d9be6f74246",
    ),
    "bedtr1993": (
        "resources/Delft3D-main/src/utils_gpl/morphology/packages/"
        "morphology_kernel/src/bedtr1993.f90",
        "d330765e7b833b1402b66a524784157019d04737c979967fdc3b501e158c9e66",
    ),
    "tranb5": (
        "resources/Delft3D-main/src/utils_gpl/morphology/packages/"
        "morphology_kernel/src/tranb5.f90",
        "ca5933ccda6592174c627c81d2d10e977064dd47f96560d349b1a3fca50a2070",
    ),
    "fgyint": (
        "resources/Delft3D-main/src/utils_gpl/morphology/packages/"
        "morphology_kernel/src/fgyint.f90",
        "135138690f0b56a0f89f18cd9f9bc85149f29458f76b167f0ce54ec695730e19",
    ),
    "termfy": (
        "resources/Delft3D-main/src/utils_gpl/morphology/packages/"
        "morphology_kernel/src/termfy.f90",
        "55bb8621b7d09e7a64d4c4e1a8b4b38697801c3fcfb7f11792c56e9e9e0e11e5",
    ),
    "termgy": (
        "resources/Delft3D-main/src/utils_gpl/morphology/packages/"
        "morphology_kernel/src/termgy.f90",
        "6d6d1b81af9140f299fbd8a3e04cfe10163f2a9db1d235651995f4a9232c4618",
    ),
    "trab19": (
        "resources/Delft3D-main/src/utils_gpl/morphology/packages/"
        "morphology_kernel/src/trab19.f90",
        "bfc530b573b95c4467a9de4bd34e25757f3990d489426db995b28d4b7ad34c63",
    ),
    "trab20": (
        "resources/Delft3D-main/src/utils_gpl/morphology/packages/"
        "morphology_kernel/src/trab20.f90",
        "a7cbf17f232fc6edae5a1cf5f7c9b2ca1e1e0626aee6dbb5270efc1c245e4d19",
    ),
    "sed_support_routines": (
        "resources/Delft3D-main/src/utils_gpl/morphology/packages/"
        "morphology_kernel/src/sed_support_routines.f90",
        "9a3244290df46993421049b6813f79489c006974050a35485f7db30463b4699f",
    ),
})


_IMPLEMENTATION_SOURCES = MappingProxyType({
    "cpu": {
        "common": (
            ("family_lifecycle_openmp", "solver/wave/braids_direct_vector_family_native_v67_module.f90",
             "4a170f529b0f7de67fa361c40cd1b859845e1c6a868cdbf950ad20d9fc0b1713"),
            ("route_composer", "solver/wave/braids_direct_vector_route_composer_v66_module.f90",
             "28284083e99f1929d272b25a3eb3f9f4b359365fbd7b8cb27fa5bcdc449ff52f"),
            ("face_exner", "solver/wave/braids_scalar_flow_live_v65_module.f90",
             "6956e5a88494d6b5dba5c4a2d46e671c61cbb6eee6bf7ca304fa5d8922002499"),
        ),
        "soulsby": (
            ("soulsby_producer", "solver/wave/braids_bedload_current_module.f90",
             "ca59bf18e07c48200ab8a5ec43326349f19dc97f044c71c846dad7ec21a8e402"),
        ),
        "soulsby_van_rijn": (
            ("soulsby_van_rijn_producer", "solver/wave/braids_soulsby_van_rijn_module.f90",
             "125e72f37a074df7a95e158651fd7365f69c4400457e388e8b5ceaa2ba1364b9"),
        ),
        "van_rijn_1993": (
            ("producer_adapter", "solver/wave/braids_direct_producer_adapter_v67_module.f90",
             "c940a8564abcb03f2fee8fda898893267b4f55150146b3f4613662590c1493d2"),
            ("van_rijn_1993_producer", "solver/wave/braids_vr1993_route_v66_module.f90",
             "28dfd4899cfc2fa1443b6202207cdff04de860fb787d4da9657f769a368e13d3"),
        ),
        "delft_direct_current": (
            ("producer_adapter", "solver/wave/braids_direct_producer_adapter_v67_module.f90",
             "c940a8564abcb03f2fee8fda898893267b4f55150146b3f4613662590c1493d2"),
            ("direct_current_producer", "solver/wave/braids_delft_direct_current_v66_module.f90",
             "5b48bc91ee3ecc70504800ad2f0dab1d76da944b5b30f9098c987301f4c63111"),
        ),
    },
    "implicit_fwave_midpoint_gpu": {
        "common": (
            ("family_lifecycle_python", "braids_gpu/direct_vector_family_live_v67.py",
             "b0d81dd557b63b1fa790d1bad83ff923b17b5ae68de239707ed9b9d4c375a6ba"),
            ("family_face_exner_python", "braids_gpu/direct_vector_face_exner_v67.py",
             "1ba0a45b21112f36d8d1816872d180cddd303e5e55c511d30fb0b957863fd02a"),
            ("route_composer_python", "braids_gpu/direct_vector_route_composer_v66.py",
             "c987ac8348a58baa53e5987f2c6d4a8824cf5774a7d3b0b5a7ec5e58334408d1"),
            ("route_composer_cuda", "braids_gpu/direct_vector_route_composer_v66.cu",
             "0452c4932ccc1bfdf7dc93d90ae7509061db1304610cc8fcf93f330b1de51911"),
            ("face_exner_python", "braids_gpu/scalar_capacity_runtime.py",
             "566c1baff6bdf9462ba7bdf006aab80033324e6012bd4fd839f30d4f6d3c65ab"),
            ("face_exner_cuda", "braids_gpu/scalar_capacity_runtime.cu",
             "7144971b9736f43b484f4edd50039b9dc3a2ab7389357a30c15fed4a078935fc"),
        ),
        "soulsby": (
            ("capacity_dispatch_python", "braids_gpu/m1_capacity_dispatch.py",
             "df6f4c0f4eceba8c398173a5d608c1493f76dfaad3e4bfc6639aa5ea0e6cf678"),
            ("capacity_dispatch_cuda", "braids_gpu/m1_capacity_dispatch.cu",
             "34938043207edb14e66cf05dc10ba84832f7c7641bdd851e35b83349436fee3b"),
        ),
        "soulsby_van_rijn": (
            ("capacity_dispatch_python", "braids_gpu/m1_capacity_dispatch.py",
             "df6f4c0f4eceba8c398173a5d608c1493f76dfaad3e4bfc6639aa5ea0e6cf678"),
            ("capacity_dispatch_cuda", "braids_gpu/m1_capacity_dispatch.cu",
             "34938043207edb14e66cf05dc10ba84832f7c7641bdd851e35b83349436fee3b"),
        ),
        "van_rijn_1993": (
            ("producer_adapter", "braids_gpu/direct_producer_adapter_v67.py",
             "ab1ff6902ccacdde027b6604fb24a3f1a5f2de796bf87a95279fa145eaf1cfc3"),
            ("van_rijn_1993_python", "braids_gpu/van_rijn_1993_route_v66.py",
             "2075bf882a74dfa0019f5834e389d91a67481330e0c83d9017e2df262a3f7c47"),
            ("van_rijn_1993_cuda", "braids_gpu/van_rijn_1993_route_v66.cu",
             "7108548796dceb88681cb7a7e08dade96d0c3e4488a98c10f99a5917c44029dd"),
        ),
        "delft_direct_current": (
            ("producer_adapter", "braids_gpu/direct_producer_adapter_v67.py",
             "ab1ff6902ccacdde027b6604fb24a3f1a5f2de796bf87a95279fa145eaf1cfc3"),
            ("direct_current_python", "braids_gpu/delft_direct_current_v66.py",
             "427871488970d105cf223247d1cda1290461724eac87eef090af2809a4208d98"),
            ("direct_current_cuda", "braids_gpu/delft_direct_current_v66.cu",
             "192595d7c7103c1445fe887c4d9451a0be5d201eb3c736386f94b949eec1e12d"),
        ),
    },
})


def direct_vector_family_producer_identity_v67(
    method_id: str,
) -> dict[str, object]:
    method = _method(method_id)
    native: dict[str, object]
    if method.method_id in {
            METHOD_SOULSBY_CURRENT, METHOD_SOULSBY_VAN_RIJN_CURRENT}:
        native = {
            "schema_id": "direct_vector_family_native_v67",
            "producer_id": method.producer_id,
            "composer_method": method.composer_method,
            "output_kind": method.output_kind,
        }
    else:
        native = {
            "schema_id": "direct_producer_adapter_v67",
            "document": direct_producer_identity_document_v67(method.method_id),
        }
    body: dict[str, object] = {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "kind": "method_producer_identity",
        "method_id": method.method_id,
        "internal_wire_id": method.internal_wire_id,
        "identity_tag": method.identity_tag,
        "reference_selector": method.reference_selector,
        "producer_id": method.producer_id,
        "composer_method": method.composer_method,
        "output_kind": method.output_kind,
        "native_identity": native,
        "constitutive_output_units": "solid_mass",
        "post_producer_density_conversion_count": 0,
        "external_secondary_rotation": False,
        "automatic_backend_fallback": False,
    }
    return {**body, "fingerprint": _fingerprint(body)}


def load_direct_vector_family_producer_identity_v67(
    value: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise DirectVectorFamilyLiveV67Error(
            "Q3-H6/v67 producer identity must be an object"
        )
    try:
        rebuilt = direct_vector_family_producer_identity_v67(
            value["method_id"]  # type: ignore[arg-type]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise DirectVectorFamilyLiveV67Error(
            "Invalid Q3-H6/v67 producer identity"
        ) from error
    if dict(value) != rebuilt or _canonical(value) != _canonical(rebuilt):
        raise DirectVectorFamilyLiveV67Error(
            "Q3-H6/v67 producer identity is stale or forged"
        )
    return rebuilt


def direct_vector_family_semantic_document_v67() -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "envelope_version": ENVELOPE_VERSION,
        "method_identity_namespace": (
            "direct_vector_family_v67_not_legacy_law_or_Delft_TraFrm"
        ),
        "methods": [
            {
                "method_id": row.method_id,
                "internal_wire_id": row.internal_wire_id,
                "identity_tag": row.identity_tag,
                "reference_selector": row.reference_selector,
                "output_kind": row.output_kind,
                "parameter_schema": row.parameter_schema,
            }
            for row in _METHOD_ROWS
        ],
        "one_canonical_method_per_fraction": True,
        "route_by_transport_class": {
            str(TRA_BEDLOAD): ROUTE_TRA_BEDLOAD,
            str(TRA_ADVDIFF): "rejected_bypasses_direct_capacity_owned_by_Q4",
            str(TRA_COMBINE): "diagnostic_only_Q4_concentration_no_state_commit",
        },
        "scope": {
            "forcing": "current_only",
            "mesh": "fixed_grid",
            "wet_dry": "fully_wet",
            "reconstruction": "first_order",
            "upwind": True,
            "lateral_averaging": False,
            "bed_slope": "off",
            "external_secondary_rotation": False,
        },
        "ownership": {
            "producer_output": "solid_mass",
            "post_producer_density_conversion_count": 0,
            "bed_and_suspended_factors": "route_composer_exactly_once",
            "active_fraction_owner": "face_exner_projection",
            "active_fraction_application_count": 1,
            "finite_inventory_fill_owner": "face_exner_projection",
            "finite_inventory_fill_application_count": 1,
        },
        "backend_choice": "explicit_user_choice_no_fallback",
        "gpu_integrator": "implicit_fwave_midpoint",
        "checkpoint": "portable_complete_accepted_state_no_trial_arrays",
        "accepted_step_limit": MAX_ACCEPTED_STEPS,
        "normal_product_admission": False,
    }


SEMANTIC_ID = _fingerprint(direct_vector_family_semantic_document_v67())


def _channel_document(
    value: ChannelCompositionProfile | Mapping[str, object],
) -> dict[str, object]:
    try:
        if isinstance(value, ChannelCompositionProfile):
            return channel_composition_profile_document(value)
        return channel_composition_profile_document(
            load_channel_composition_profile(value)
        )
    except (TypeError, ValueError) as error:
        raise DirectVectorFamilyLiveV67Error(
            "Q3-H6/v67 channel profile is invalid; class 2 is rejected"
        ) from error


def _source_manifest(
    methods: Sequence[DirectVectorFamilyMethodV67], backend: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    reference_names = {"eqtran", "rdtrafrm", "upwbed"}
    for method in methods:
        reference_names.update(method.reference_source_names)
    reference = [
        {"name": name, "path": _REFERENCE_SOURCES[name][0],
         "sha256": _REFERENCE_SOURCES[name][1]}
        for name in sorted(reference_names)
    ]

    rows = list(_IMPLEMENTATION_SOURCES[backend]["common"])
    for group in sorted({method.implementation_group for method in methods}):
        rows.extend(_IMPLEMENTATION_SOURCES[backend][group])
    by_path: dict[str, dict[str, object]] = {}
    for name, path, digest in rows:
        existing = by_path.get(path)
        if existing is not None and existing["sha256"] != digest:
            raise DirectVectorFamilyLiveV67Error(
                "Q3-H6/v67 implementation manifest hash collision"
            )
        if existing is None:
            by_path[path] = {"name": name, "path": path, "sha256": digest}
    implementation = [by_path[path] for path in sorted(by_path)]
    return reference, implementation


def direct_vector_family_execution_document_v67(
    *, backend: str,
    channel_profile: ChannelCompositionProfile | Mapping[str, object],
    method_ids: Sequence[str], parameters: Sequence[Mapping[str, object]],
    d50_m: Sequence[float], d90_m: Sequence[float],
    solid_density_kg_m3: Sequence[float],
    bulk_density_kg_m3: Sequence[float], roughness_method_id: str,
    forcing_fingerprint: str, roughness_fingerprint: str,
    settling_fingerprint: str, vertical_fingerprint: str,
    support_fingerprint: str, topology_fingerprint: str,
    boundary_fingerprint: str, face_fingerprint: str,
    exner_fingerprint: str, water_density_kg_m3: float = 1000.0,
    gravity_m_s2: float = 9.81, minimum_work_depth_m: float = 0.01,
) -> dict[str, object]:
    """Build a strict per-fraction six-method CPU/GPU execution receipt."""
    if backend not in BACKENDS:
        raise DirectVectorFamilyLiveV67Error(
            "Q3-H6/v67 backend must be explicitly CPU or implicit GPU"
        )
    channel = _channel_document(channel_profile)
    fraction_rows = channel["fractions"]
    assert isinstance(fraction_rows, list)  # canonical channel document
    nfrac = len(fraction_rows)
    if (not isinstance(method_ids, Sequence) or
            isinstance(method_ids, (str, bytes, bytearray)) or
            len(method_ids) != nfrac):
        raise DirectVectorFamilyLiveV67Error(
            "Q3-H6/v67 requires one canonical method per fraction"
        )
    if (not isinstance(parameters, Sequence) or
            isinstance(parameters, (str, bytes, bytearray)) or
            len(parameters) != nfrac):
        raise DirectVectorFamilyLiveV67Error(
            "Q3-H6/v67 requires one parameter mapping per fraction"
        )
    methods = [_method(value) for value in method_ids]
    d50 = _sequence(d50_m, "D50", length=nfrac, positive=True)
    d90 = _sequence(d90_m, "D90", length=nfrac, positive=True)
    solid_density = _sequence(
        solid_density_kg_m3, "solid density", length=nfrac, positive=True
    )
    bulk_density = _sequence(
        bulk_density_kg_m3, "bulk density", length=nfrac, positive=True
    )
    if any(d90[index] < d50[index] for index in range(nfrac)):
        raise DirectVectorFamilyLiveV67Error(
            "Q3-H6/v67 D90 may not be below D50"
        )
    if (not isinstance(roughness_method_id, str) or roughness_method_id not in {
            "roughness.chezy", "roughness.manning",
            "roughness.white_colebrook", "roughness.white_colebrook_waqua"}):
        raise DirectVectorFamilyLiveV67Error(
            "Q3-H6/v67 requires a typed hydraulic roughness method"
        )

    classes = [row["transport_class"] for row in fraction_rows]
    if any(value == TRA_ADVDIFF for value in classes):
        raise DirectVectorFamilyLiveV67Error(
            "Q3-H6/v67 class 2 bypasses direct capacity and is rejected"
        )
    if any(value not in (TRA_BEDLOAD, TRA_COMBINE) for value in classes):
        raise DirectVectorFamilyLiveV67Error(
            "Q3-H6/v67 unsupported transport class"
        )
    routes = [
        ROUTE_TRA_BEDLOAD if value == TRA_BEDLOAD else ROUTE_TRA_COMBINE
        for value in classes
    ]
    fraction_methods = []
    for index, (fraction, method, supplied, route) in enumerate(zip(
            fraction_rows, methods, parameters, routes, strict=True)):
        parameter_identity = direct_vector_family_parameter_identity_v67(
            method.method_id, supplied
        )
        producer_identity = direct_vector_family_producer_identity_v67(
            method.method_id
        )
        fraction_methods.append({
            "fraction_index": index,
            "fraction_id": fraction["fraction_id"],
            "transport_class": classes[index],
            "route": route,
            "method_id": method.method_id,
            "internal_wire_id": method.internal_wire_id,
            "identity_tag": method.identity_tag,
            "reference_selector": method.reference_selector,
            "parameter_identity": parameter_identity,
            "producer_identity": producer_identity,
            "state_admission": (
                "accepted_exner" if classes[index] == TRA_BEDLOAD
                else "diagnostic_q4_only_no_commit"
            ),
        })

    dependencies = {
        "forcing": _sha(forcing_fingerprint, "forcing fingerprint"),
        "roughness": _sha(roughness_fingerprint, "roughness fingerprint"),
        "settling": _sha(settling_fingerprint, "settling fingerprint"),
        "vertical": _sha(vertical_fingerprint, "vertical fingerprint"),
        "support": _sha(support_fingerprint, "support fingerprint"),
        "topology": _sha(topology_fingerprint, "topology fingerprint"),
        "boundary": _sha(boundary_fingerprint, "boundary fingerprint"),
        "face": _sha(face_fingerprint, "face fingerprint"),
        "exner": _sha(exner_fingerprint, "Exner fingerprint"),
        "channel": channel["fingerprint"],
    }
    reference_sources, implementation_sources = _source_manifest(methods, backend)
    source_body = {
        "reference_sources": reference_sources,
        "implementation_sources": implementation_sources,
    }
    physical = {
        "fraction_ids": [row["fraction_id"] for row in fraction_rows],
        "transport_classes": classes,
        "routes": routes,
        "fraction_methods": fraction_methods,
        "d50_m": d50,
        "d90_m": d90,
        "solid_density_kg_m3": solid_density,
        "bulk_density_kg_m3": bulk_density,
        "water_density_kg_m3": _finite(
            water_density_kg_m3, "water density", positive=True
        ),
        "gravity_m_s2": _finite(gravity_m_s2, "gravity", positive=True),
        "minimum_work_depth_m": _finite(
            minimum_work_depth_m, "minimum work depth", nonnegative=True
        ),
        "roughness_method_id": roughness_method_id,
        "bed_factor": channel["bed_factor"],
        "suspended_factor": channel["suspended_factor"],
        "speed_epsilon": channel["epsilon"],
        "forcing_scope": "current_only",
        "external_secondary_rotation": False,
        "post_producer_density_conversion_count": 0,
        "active_fraction_owner": "face_exner_projection",
        "upwind_bedload": True,
        "laterally_averaged_bedload": False,
        "bed_slope_method_id": "bed_slope.off",
    }
    body: dict[str, object] = {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "envelope_version": ENVELOPE_VERSION,
        "qualification": "q3h6_bounded_family_receipt_not_product_dispatch",
        "backend": backend,
        "integrator": (
            "native_openmp_reference" if backend == "cpu"
            else "implicit_fwave_midpoint"
        ),
        "automatic_backend_fallback": False,
        "physical_profile": physical,
        "physical_profile_fingerprint": _fingerprint(physical),
        "dependencies": dependencies,
        "reference_sources": reference_sources,
        "implementation_sources": implementation_sources,
        "source_manifest_fingerprint": _fingerprint(source_body),
        "transport_class_scope": (
            "class1_accepted_class3_diagnostic_only_class2_rejected"
        ),
        "accepted_state_admitted": all(
            value == TRA_BEDLOAD for value in classes
        ),
        "active_fraction_owner": "face_exner_projection",
        "active_fraction_application_count": 1,
        "finite_inventory_fill_application_count": 1,
        "accepted_step_limit": MAX_ACCEPTED_STEPS,
        "normal_product_admission": False,
    }
    return {**body, "fingerprint": _fingerprint(body)}


def load_direct_vector_family_execution_v67(
    value: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise DirectVectorFamilyLiveV67Error(
            "Q3-H6/v67 execution identity must be an object"
        )
    try:
        physical = value["physical_profile"]
        dependencies = value["dependencies"]
        fraction_methods = physical["fraction_methods"]
        if not isinstance(fraction_methods, list):
            raise TypeError("fraction methods")
        method_ids = []
        parameters = []
        for row in fraction_methods:
            if not isinstance(row, Mapping):
                raise TypeError("fraction method row")
            method_ids.append(row["method_id"])
            parameter_identity = row["parameter_identity"]
            loaded = load_direct_vector_family_parameter_identity_v67(
                parameter_identity
            )
            native = loaded["native_identity"]
            if native["schema_id"] == "capacity_parameters_v1":
                parameter_rows = native["parameters"]
            else:
                parameter_rows = native["document"]["parameters"]
            parameters.append({
                item["keyword"]: item["value"] for item in parameter_rows
            })
            load_direct_vector_family_producer_identity_v67(
                row["producer_identity"]
            )
        channel = resolve_channel_composition_profile(
            physical["fraction_ids"], physical["transport_classes"],
            bed_factor=physical["bed_factor"],
            suspended_factor=physical["suspended_factor"],
            epsilon=physical["speed_epsilon"],
        )
        rebuilt = direct_vector_family_execution_document_v67(
            backend=value["backend"], channel_profile=channel,
            method_ids=method_ids, parameters=parameters,
            d50_m=physical["d50_m"], d90_m=physical["d90_m"],
            solid_density_kg_m3=physical["solid_density_kg_m3"],
            bulk_density_kg_m3=physical["bulk_density_kg_m3"],
            roughness_method_id=physical["roughness_method_id"],
            forcing_fingerprint=dependencies["forcing"],
            roughness_fingerprint=dependencies["roughness"],
            settling_fingerprint=dependencies["settling"],
            vertical_fingerprint=dependencies["vertical"],
            support_fingerprint=dependencies["support"],
            topology_fingerprint=dependencies["topology"],
            boundary_fingerprint=dependencies["boundary"],
            face_fingerprint=dependencies["face"],
            exner_fingerprint=dependencies["exner"],
            water_density_kg_m3=physical["water_density_kg_m3"],
            gravity_m_s2=physical["gravity_m_s2"],
            minimum_work_depth_m=physical["minimum_work_depth_m"],
        )
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise DirectVectorFamilyLiveV67Error(
            "Invalid Q3-H6/v67 execution identity"
        ) from error
    if dict(value) != rebuilt or _canonical(value) != _canonical(rebuilt):
        raise DirectVectorFamilyLiveV67Error(
            "Q3-H6/v67 execution identity is stale or forged"
        )
    return rebuilt


def verify_direct_vector_family_sources_v67(
    execution: Mapping[str, object], *, repository_root: str | Path | None = None,
) -> dict[str, object]:
    """Verify exact BRAIDS bytes and read-only locked Delft3D provenance."""
    resolved = load_direct_vector_family_execution_v67(execution)
    root = (Path(repository_root) if repository_root is not None else
            Path(__file__).resolve().parents[2])
    rows = []
    for category in ("reference_sources", "implementation_sources"):
        for source in resolved[category]:
            path = root/source["path"]
            if not path.is_file():
                raise DirectVectorFamilyLiveV67Error(
                    f"Q3-H6/v67 authenticated source is missing: {source['path']}"
                )
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != source["sha256"]:
                raise DirectVectorFamilyLiveV67Error(
                    f"Q3-H6/v67 authenticated source differs: {source['path']}"
                )
            rows.append({**source, "category": category, "verified": True})
    body = {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "execution_fingerprint": resolved["fingerprint"],
        "source_manifest_fingerprint": resolved["source_manifest_fingerprint"],
        "sources": rows,
        "delft3d_source_access": "read_only_hash_verification",
    }
    return {**body, "fingerprint": _fingerprint(body)}


def _immutable_array(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise DirectVectorFamilyLiveV67Error(
            f"Q3-H6/v67 {name} must be finite with shape {shape}"
        )
    result = np.array(array, dtype=np.float64, copy=True, order="C")
    result.setflags(write=False)
    return result


def _array_digest(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        data = np.ascontiguousarray(value)
        digest.update(str(data.dtype).encode("ascii"))
        digest.update(_canonical(list(data.shape)))
        digest.update(data.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class DirectVectorFamilyStateV67:
    execution_fingerprint: str
    bed_mass_kg_m2: np.ndarray
    bulk_density_kg_m3: np.ndarray
    active_fraction: np.ndarray
    boundary_signed_kg: np.ndarray
    boundary_absolute_kg: np.ndarray
    accepted_step: int
    time_s: float
    state_sha256: str


def _state_token(state: DirectVectorFamilyStateV67) -> str:
    return _fingerprint({
        "execution_fingerprint": state.execution_fingerprint,
        "accepted_step": state.accepted_step,
        "time_s": state.time_s,
        "arrays_sha256": _array_digest(
            state.bed_mass_kg_m2, state.bulk_density_kg_m3,
            state.active_fraction, state.boundary_signed_kg,
            state.boundary_absolute_kg,
        ),
    })


def resolve_direct_vector_family_state_v67(
    *, execution: Mapping[str, object], bed_mass_kg_m2: object,
    active_fraction: object | None = None,
    boundary_signed_kg: object | None = None,
    boundary_absolute_kg: object | None = None,
    accepted_step: int = 0, time_s: float = 0.0,
) -> DirectVectorFamilyStateV67:
    resolved = load_direct_vector_family_execution_v67(execution)
    if not resolved["accepted_state_admitted"]:
        raise DirectVectorFamilyLiveV67Error(
            "Q3-H6/v67 class 3 is diagnostic only and cannot own accepted state"
        )
    raw = np.asarray(bed_mass_kg_m2, dtype=np.float64)
    nfrac = len(resolved["physical_profile"]["fraction_ids"])
    if raw.ndim != 2 or raw.shape[0] != nfrac or raw.shape[1] <= 0:
        raise DirectVectorFamilyLiveV67Error(
            "Q3-H6/v67 bed mass must have shape (fraction, cell)"
        )
    mass = _immutable_array(raw, raw.shape, "bed mass")
    if np.any(mass < 0.0):
        raise DirectVectorFamilyLiveV67Error(
            "Q3-H6/v67 bed mass may not be negative"
        )
    density = _immutable_array(
        resolved["physical_profile"]["bulk_density_kg_m3"], (nfrac,),
        "bulk density",
    )
    volume = mass/density[:, None]
    total = np.sum(volume, axis=0, keepdims=True)
    positive_stock = total[0] > 0.0
    derived = np.zeros_like(volume)
    derived[:, positive_stock] = (
        volume[:, positive_stock]/total[:, positive_stock])
    if active_fraction is None:
        if np.any(~positive_stock):
            raise DirectVectorFamilyLiveV67Error(
                "Q3-H6/v67 depleted cells require an explicit retained "
                "surface identity"
            )
        fraction = _immutable_array(derived, raw.shape, "active fraction")
    else:
        fraction = _immutable_array(active_fraction, raw.shape, "active fraction")
        tolerance = 512.0*np.finfo(np.float64).eps
        if (np.any(fraction < 0.0) or
                not np.allclose(np.sum(fraction, axis=0), 1.0,
                                rtol=0.0, atol=tolerance*nfrac) or
                not np.allclose(fraction[:, positive_stock],
                                derived[:, positive_stock],
                                rtol=tolerance, atol=tolerance)):
            raise DirectVectorFamilyLiveV67Error(
                "Q3-H6/v67 active fraction differs from positive accepted "
                "stock or is not a normalized retained identity"
            )
    signed = _immutable_array(
        np.zeros(nfrac) if boundary_signed_kg is None else boundary_signed_kg,
        (nfrac,), "signed boundary ledger",
    )
    absolute = _immutable_array(
        np.zeros(nfrac) if boundary_absolute_kg is None else boundary_absolute_kg,
        (nfrac,), "absolute boundary ledger",
    )
    if np.any(absolute < np.abs(signed)):
        raise DirectVectorFamilyLiveV67Error(
            "Q3-H6/v67 absolute boundary ledger must bound signed transport"
        )
    if (isinstance(accepted_step, bool) or type(accepted_step) is not int or
            accepted_step < 0 or accepted_step > MAX_ACCEPTED_STEPS):
        raise DirectVectorFamilyLiveV67Error(
            "Q3-H6/v67 accepted step is invalid"
        )
    accepted_time = _finite(time_s, "accepted time", nonnegative=True)
    placeholder = DirectVectorFamilyStateV67(
        resolved["fingerprint"], mass, density, fraction, signed, absolute,
        accepted_step, accepted_time, "",
    )
    return DirectVectorFamilyStateV67(
        placeholder.execution_fingerprint, placeholder.bed_mass_kg_m2,
        placeholder.bulk_density_kg_m3, placeholder.active_fraction,
        placeholder.boundary_signed_kg, placeholder.boundary_absolute_kg,
        placeholder.accepted_step, placeholder.time_s, _state_token(placeholder),
    )


def validate_direct_vector_family_state_v67(
    state: DirectVectorFamilyStateV67, *, execution: Mapping[str, object],
) -> DirectVectorFamilyStateV67:
    resolved = load_direct_vector_family_execution_v67(execution)
    if not isinstance(state, DirectVectorFamilyStateV67):
        raise DirectVectorFamilyLiveV67Error(
            "Q3-H6/v67 accepted state is missing"
        )
    if (state.execution_fingerprint != resolved["fingerprint"] or
            state.state_sha256 != _state_token(state)):
        raise DirectVectorFamilyLiveV67Error(
            "Q3-H6/v67 accepted state is stale, mutated, or execution-mismatched"
        )
    rebuilt = resolve_direct_vector_family_state_v67(
        execution=resolved, bed_mass_kg_m2=state.bed_mass_kg_m2,
        active_fraction=state.active_fraction,
        boundary_signed_kg=state.boundary_signed_kg,
        boundary_absolute_kg=state.boundary_absolute_kg,
        accepted_step=state.accepted_step, time_s=state.time_s,
    )
    if rebuilt.state_sha256 != state.state_sha256:
        raise DirectVectorFamilyLiveV67Error(
            "Q3-H6/v67 accepted state cannot be reconstructed"
        )
    return state


def advance_direct_vector_family_state_v67(
    *, execution: Mapping[str, object], parent: DirectVectorFamilyStateV67,
    bed_mass_kg_m2: object, dt_s: float,
    boundary_signed_delta_kg: object | None = None,
    boundary_absolute_delta_kg: object | None = None,
) -> DirectVectorFamilyStateV67:
    """Publish one already-validated candidate as the next accepted state.

    Numerical flux evaluation remains owned by the native CPU/GPU path.  This
    function supplies the portable, mutation-detecting acceptance boundary
    used by bounded restart tests.
    """
    resolved = load_direct_vector_family_execution_v67(execution)
    validate_direct_vector_family_state_v67(parent, execution=resolved)
    if parent.accepted_step >= MAX_ACCEPTED_STEPS:
        raise DirectVectorFamilyLiveV67Error(
            "Q3-H6/v67 bounded accepted-step limit was reached"
        )
    dt = _finite(dt_s, "accepted dt", positive=True)
    nfrac = len(resolved["physical_profile"]["fraction_ids"])
    signed_delta = _immutable_array(
        np.zeros(nfrac) if boundary_signed_delta_kg is None
        else boundary_signed_delta_kg,
        (nfrac,), "signed boundary increment",
    )
    absolute_delta = _immutable_array(
        np.zeros(nfrac) if boundary_absolute_delta_kg is None
        else boundary_absolute_delta_kg,
        (nfrac,), "absolute boundary increment",
    )
    tolerance = 64.0*np.finfo(np.float64).eps*np.maximum(
        1.0, np.abs(signed_delta)
    )
    if np.any(absolute_delta < 0.0) or np.any(
            absolute_delta+tolerance < np.abs(signed_delta)):
        raise DirectVectorFamilyLiveV67Error(
            "Q3-H6/v67 absolute boundary increment must bound signed increment"
        )
    return resolve_direct_vector_family_state_v67(
        execution=resolved, bed_mass_kg_m2=bed_mass_kg_m2,
        boundary_signed_kg=parent.boundary_signed_kg+signed_delta,
        boundary_absolute_kg=parent.boundary_absolute_kg+absolute_delta,
        accepted_step=parent.accepted_step+1, time_s=parent.time_s+dt,
    )


def direct_vector_family_state_checkpoint_v67(
    state: DirectVectorFamilyStateV67,
) -> dict[str, object]:
    if not isinstance(state, DirectVectorFamilyStateV67):
        raise DirectVectorFamilyLiveV67Error(
            "Q3-H6/v67 checkpoint state is missing"
        )
    body: dict[str, object] = {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "envelope_version": ENVELOPE_VERSION,
        "stage": "checkpoint",
        "execution_fingerprint": state.execution_fingerprint,
        "accepted_step": state.accepted_step,
        "time_s": state.time_s,
        "bed_mass_kg_m2": state.bed_mass_kg_m2.tolist(),
        "bulk_density_kg_m3": state.bulk_density_kg_m3.tolist(),
        "active_fraction": state.active_fraction.tolist(),
        "boundary_signed_kg": state.boundary_signed_kg.tolist(),
        "boundary_absolute_kg": state.boundary_absolute_kg.tolist(),
        "state_sha256": state.state_sha256,
        "transaction_schema": "complete_accepted_state_no_trial_arrays",
    }
    return {**body, "fingerprint": _fingerprint(body)}


def load_direct_vector_family_state_checkpoint_v67(
    value: Mapping[str, object], *, execution: Mapping[str, object],
) -> DirectVectorFamilyStateV67:
    resolved = load_direct_vector_family_execution_v67(execution)
    expected = {
        "schema", "scope_id", "semantic_id", "envelope_version", "stage",
        "execution_fingerprint", "accepted_step", "time_s",
        "bed_mass_kg_m2", "bulk_density_kg_m3", "active_fraction",
        "boundary_signed_kg", "boundary_absolute_kg", "state_sha256",
        "transaction_schema", "fingerprint",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise DirectVectorFamilyLiveV67Error(
            "Q3-H6/v67 checkpoint inventory differs"
        )
    body = {key: item for key, item in value.items() if key != "fingerprint"}
    if (value["fingerprint"] != _fingerprint(body) or
            value["schema"] != SCHEMA or value["scope_id"] != SCOPE_ID or
            value["semantic_id"] != SEMANTIC_ID or
            value["envelope_version"] != ENVELOPE_VERSION or
            value["stage"] != "checkpoint" or
            value["transaction_schema"] !=
            "complete_accepted_state_no_trial_arrays" or
            value["execution_fingerprint"] != resolved["fingerprint"] or
            value["bulk_density_kg_m3"] !=
            resolved["physical_profile"]["bulk_density_kg_m3"]):
        raise DirectVectorFamilyLiveV67Error(
            "Q3-H6/v67 checkpoint header or execution differs"
        )
    state = resolve_direct_vector_family_state_v67(
        execution=resolved, bed_mass_kg_m2=value["bed_mass_kg_m2"],
        active_fraction=value["active_fraction"],
        boundary_signed_kg=value["boundary_signed_kg"],
        boundary_absolute_kg=value["boundary_absolute_kg"],
        accepted_step=value["accepted_step"], time_s=value["time_s"],
    )
    if state.state_sha256 != value["state_sha256"]:
        raise DirectVectorFamilyLiveV67Error(
            "Q3-H6/v67 checkpoint state digest differs"
        )
    return state


__all__ = [
    "BACKENDS", "ENVELOPE_VERSION", "MAX_ACCEPTED_STEPS", "METHODS",
    "METHODS_BY_WIRE", "METHOD_BIJKER_CURRENT", "METHOD_SOULSBY_CURRENT",
    "METHOD_SOULSBY_VAN_RIJN_CURRENT", "METHOD_VAN_RIJN_1993_CURRENT",
    "METHOD_VAN_THIEL_VAN_RIJN_2008_CURRENT",
    "METHOD_XBEACH_SOULSBY_VAN_RIJN_CURRENT", "SCHEMA", "SCOPE_ID",
    "SEMANTIC_ID", "SUPPORTED_METHODS", "WIRE_SOULSBY_CURRENT",
    "WIRE_SOULSBY_VAN_RIJN_CURRENT", "DirectVectorFamilyLiveV67Error",
    "DirectVectorFamilyMethodV67", "DirectVectorFamilyStateV67",
    "advance_direct_vector_family_state_v67",
    "direct_vector_family_execution_document_v67",
    "direct_vector_family_parameter_identity_v67",
    "direct_vector_family_producer_identity_v67",
    "direct_vector_family_semantic_document_v67",
    "direct_vector_family_state_checkpoint_v67",
    "load_direct_vector_family_execution_v67",
    "load_direct_vector_family_parameter_identity_v67",
    "load_direct_vector_family_producer_identity_v67",
    "load_direct_vector_family_state_checkpoint_v67",
    "resolve_direct_vector_family_state_v67",
    "validate_direct_vector_family_state_v67",
    "verify_direct_vector_family_sources_v67",
]
