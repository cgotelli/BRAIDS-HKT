"""Source-ordered ``MaximumWaterdepth`` bridge for the six FLOW 2DH laws.

Classic FLOW does *not* pass ``q/h`` directly to all transport formulae.  Its
``erosed`` chain first calls ``dwnvel`` once, then constructs the effective
current vector from the resulting characteristic direction and the
shear-derived speed.  Only Van Rijn 1993 consumes the characteristic velocity
and observation height themselves.  This additive v68 layer preserves that
ordering while leaving the accepted v66/v67 implementations immutable.

The module is both the independent NumPy oracle and the authenticated control
plane for the native OpenMP and resident-CUDA bridges.  Delft3D sources are
only read and hash-verified.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from types import MappingProxyType

import numpy as np

from .direct_vector_family_live_v67 import (
    METHOD_BIJKER_CURRENT,
    METHOD_SOULSBY_CURRENT,
    METHOD_SOULSBY_VAN_RIJN_CURRENT,
    METHOD_VAN_RIJN_1993_CURRENT,
    METHOD_VAN_THIEL_VAN_RIJN_2008_CURRENT,
    METHOD_XBEACH_SOULSBY_VAN_RIJN_CURRENT,
    load_direct_vector_family_execution_v67,
    verify_direct_vector_family_sources_v67,
)
from .direct_vector_catalogue_bridge_v67 import (
    PROFILE_TO_CATALOGUE,
    direct_vector_catalogue_bridge_document_v67,
)
from .maximum_waterdepth_live_v67 import (
    load_support_descriptor_v67,
    support_descriptor_document_v67,
    verify_support_descriptor_v67,
)
from .maximum_waterdepth_v66 import (
    maximum_waterdepth_characteristic_velocity_v66,
)


SCHEMA = 1
ENVELOPE_VERSION = 68
SCOPE_ID = "braids-q3-h6c-direct-vector-maximum-waterdepth-v68-v1"
DEFAULT_ENABLED = False
FLOW_EPS = float(np.float32(1.0e-6))
_SHA = re.compile(r"[0-9a-f]{64}\Z")

METHODS = (
    METHOD_SOULSBY_CURRENT,
    METHOD_SOULSBY_VAN_RIJN_CURRENT,
    METHOD_VAN_RIJN_1993_CURRENT,
    METHOD_BIJKER_CURRENT,
    METHOD_VAN_THIEL_VAN_RIJN_2008_CURRENT,
    METHOD_XBEACH_SOULSBY_VAN_RIJN_CURRENT,
)

# The exact data dependency proven by the locked call chain.  The switch is a
# valid global FLOW control for every method.  It changes Van Rijn 1993's
# constitutive speed/height directly; the other five observe it only through
# the source-ordered direction normalization (including ``+eps`` and dwnvel's
# component zeroing), while continuing to consume the physical water depth.
METHOD_EFFECTS = MappingProxyType({
    METHOD_SOULSBY_CURRENT: "effective_current_from_characteristic_direction",
    METHOD_SOULSBY_VAN_RIJN_CURRENT:
        "effective_current_from_characteristic_direction",
    METHOD_VAN_RIJN_1993_CURRENT:
        "characteristic_velocity_and_observation_height",
    METHOD_BIJKER_CURRENT: "effective_current_from_characteristic_direction",
    METHOD_VAN_THIEL_VAN_RIJN_2008_CURRENT:
        "effective_current_from_characteristic_direction",
    METHOD_XBEACH_SOULSBY_VAN_RIJN_CURRENT:
        "effective_current_from_characteristic_direction",
})

# The three coastal formula entries retain their public Delft3D catalogue
# names while the v67 producer namespace uses explicit ``*_current`` IDs.
# Keep both identities in the receipt; never treat one as an unrecorded alias.
CATALOGUE_ENTRY_IDS = MappingProxyType(dict(PROFILE_TO_CATALOGUE))

LOCKED_SOURCES = MappingProxyType({
    "dwnvel": {
        "path": (
            "resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/"
            "flow2d3d_kernel/src/compute_sediment/dwnvel.f90"
        ),
        "sha256": "cb26d6c037804ce2be3bcc975dab81cce21b78571f75e33fa302bcc14bf17ccd",
        "lines": "143-264",
    },
    "erosed": {
        "path": (
            "resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/"
            "flow2d3d_kernel/src/compute_sediment/erosed.f90"
        ),
        "sha256": "4571fd7912233db03f597161af1a18702ee974483676004cda8186ccf24308c0",
        "lines": "660-667,856-875,929-933,974-982,1256-1269,1345-1358",
    },
    "eqtran": {
        "path": (
            "resources/Delft3D-main/src/utils_gpl/morphology/packages/"
            "morphology_kernel/src/eqtran.f90"
        ),
        "sha256": "d2e8601cff68884f51ee3dcb0cd2fb52d9f0d676b1baaa3283c68913d4778388",
        "lines": "239-247,326-350,414-425,495-524,561-590",
    },
    "tram1": {
        "path": (
            "resources/Delft3D-main/src/utils_gpl/morphology/packages/"
            "morphology_kernel/src/tram1.f90"
        ),
        "sha256": "33f5f60439c8b98facc7d5d1431e8f4a06fc37fe7b14d531a8fa9d9be6f74246",
        "lines": "175-179",
    },
    "sysini": {
        "path": (
            "resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/"
            "flow2d3d_kernel/src/main/sysini.f90"
        ),
        "sha256": "70b407afb044727356d78685c662f736fe5c7c1e5b519c31df054ebc368ae7db",
        "lines": "96-116",
    },
})

# Filled only with immutable implementation files.  This control-plane module
# deliberately does not attempt an impossible self hash.
_CPU_IMPLEMENTATION_SOURCES = MappingProxyType({
    "maximum_waterdepth_openmp": (
        "solver/wave/braids_maximum_waterdepth_v66_module.f90",
        "4b5d756b21d1f35178a1eb71fd506e6899ae2f5ccbaecdcd9b6fd5b105980e8d",
    ),
    "family_bridge_openmp": (
        "solver/wave/braids_direct_vector_maximum_waterdepth_v68_module.f90",
        "cd5b02501050d823c56ad222d18dd9b4a93560c07d08ebe8634ae8e31e6a4dcd",
    ),
})

_GPU_IMPLEMENTATION_SOURCES = MappingProxyType({
    "family_bridge_cuda_wrapper": (
        "braids_gpu/direct_vector_maximum_waterdepth_bridge_v68.py",
        "f01363fede0b1d76bdc0862d8e0ef02ed774b8ff1fb81ea4a4f7a06196cb8aab",
    ),
    "family_bridge_cuda_kernel": (
        "braids_gpu/direct_vector_maximum_waterdepth_bridge_v68.cu",
        "4ce6896eb6c89757a065686bc4d9936cad9f061a7d76daa73101c2b2122265af",
    ),
})


class DirectVectorMaximumWaterdepthV68Error(ValueError):
    """A v68 source, execution, topology, or hydraulic input is invalid."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise DirectVectorMaximumWaterdepthV68Error(
            "Q3-H6c/v68 document is not canonical JSON"
        ) from error


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _strict_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise DirectVectorMaximumWaterdepthV68Error(
            f"Q3-H6c/v68 {name} must be Boolean"
        )
    return value


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise DirectVectorMaximumWaterdepthV68Error(
            f"Q3-H6c/v68 {name} must be a lowercase SHA-256 fingerprint"
        )
    return value


def _method(value: object) -> str:
    if type(value) is not str or value not in METHOD_EFFECTS:
        raise DirectVectorMaximumWaterdepthV68Error(
            f"Q3-H6c/v68 unsupported direct-vector method: {value!r}"
        )
    return value


def direct_vector_maximum_waterdepth_semantic_document_v68() -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "envelope_version": ENVELOPE_VERSION,
        "locked_call_order": [
            "erosed:dwnvel_once_before_fraction_loop",
            "erosed:utot_from_shear_and_roughness",
            "erosed:u_v_from_utot_times_uchar_vchar_over_velch_plus_eps",
            "eqtran:selected_constitutive_producer",
        ],
        "method_effects": dict(METHOD_EFFECTS),
        "catalogue_binding": [
            {
                "producer_method_id": method,
                "catalogue_entry_id": CATALOGUE_ENTRY_IDS[method],
                "forcing_variant": "current_only",
            }
            for method in METHODS
        ],
        "physical_depth": "never_replaced_by_observation_depth",
        "flow_eps": {
            "value": FLOW_EPS,
            "source_semantics": "default_real_binary32_promoted_to_fp",
            "used_by": ["dwnvel_component_zeroing", "erosed_direction_denominator"],
        },
        "maximum_waterdepth_fraction": "parsed_but_unused_by_classic_FLOW",
        "current_only": True,
        "external_secondary_rotation": False,
        "preprocessor_application_count_per_hydraulic_point": 1,
        "backend_choice": "inherited_explicit_choice_no_fallback",
        "gpu_integrator": "implicit_fwave_midpoint",
        "accepted_step_limit": 20,
        "normal_product_admission": False,
    }


SEMANTIC_ID = _fingerprint(
    direct_vector_maximum_waterdepth_semantic_document_v68()
)


def _implementation_rows(backend: str) -> list[dict[str, str]]:
    source = (_CPU_IMPLEMENTATION_SOURCES if backend == "cpu"
              else _GPU_IMPLEMENTATION_SOURCES)
    rows = []
    for name, (path, digest) in sorted(source.items()):
        if not _SHA.fullmatch(digest):
            raise DirectVectorMaximumWaterdepthV68Error(
                "Q3-H6c/v68 implementation manifest is not finalized"
            )
        rows.append({"name": name, "path": path, "sha256": digest})
    return rows


def direct_vector_maximum_waterdepth_execution_document_v68(
    *, parent_execution: Mapping[str, object],
    enabled: bool = DEFAULT_ENABLED,
    topology_fingerprint: str,
    cell_support: Mapping[str, object] | None = None,
    trace_support: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Bind the v67 six-method family to exact dwnvel support topology."""
    parent = load_direct_vector_family_execution_v67(parent_execution)
    switch = _strict_bool(enabled, "MaximumWaterdepth switch")
    topology = _sha(topology_fingerprint, "mesh topology fingerprint")
    if topology != parent["dependencies"]["topology"]:
        raise DirectVectorMaximumWaterdepthV68Error(
            "Q3-H6c/v68 topology differs from the parent family receipt"
        )
    if switch:
        if cell_support is None or trace_support is None:
            raise DirectVectorMaximumWaterdepthV68Error(
                "Q3-H6c/v68 enabled execution requires cell and trace support"
            )
        cell = load_support_descriptor_v67(cell_support)
        trace = load_support_descriptor_v67(trace_support)
    else:
        if cell_support is not None or trace_support is not None:
            raise DirectVectorMaximumWaterdepthV68Error(
                "Q3-H6c/v68 disabled execution rejects support descriptors"
            )
        cell = None
        trace = None

    selected = [
        row["method_id"]
        for row in parent["physical_profile"]["fraction_methods"]
    ]
    catalogue_bridge = direct_vector_catalogue_bridge_document_v67(
        [CATALOGUE_ENTRY_IDS[method] for method in selected],
        forcing_scope="current_only",
    )
    applicability = [
        {
            "method_id": _method(method),
            "catalogue_entry_id": CATALOGUE_ENTRY_IDS[method],
            "forcing_variant": "current_only",
            "maximum_waterdepth_effect": METHOD_EFFECTS[method],
            "physical_depth_preserved": True,
            "preprocessor_application_count": 1,
        }
        for method in selected
    ]
    reference = [
        {"name": name, **dict(row)} for name, row in sorted(LOCKED_SOURCES.items())
    ]
    implementation = _implementation_rows(parent["backend"])
    source_body = {
        "reference_sources": reference,
        "implementation_sources": implementation,
    }
    physical = {
        **dict(parent["physical_profile"]),
        "maximum_waterdepth": switch,
        "maximum_waterdepth_default": DEFAULT_ENABLED,
        "maximum_waterdepth_fraction": "unused_in_classic_FLOW",
        "topology_fingerprint": topology,
        "cell_support_fingerprint": (
            None if cell is None else cell["fingerprint"]
        ),
        "trace_support_fingerprint": (
            None if trace is None else trace["fingerprint"]
        ),
        "method_applicability": applicability,
    }
    body: dict[str, object] = {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "envelope_version": ENVELOPE_VERSION,
        "parent_execution": parent,
        "parent_execution_fingerprint": parent["fingerprint"],
        "catalogue_bridge": catalogue_bridge,
        "catalogue_bridge_fingerprint": catalogue_bridge["fingerprint"],
        "backend": parent["backend"],
        "integrator": parent["integrator"],
        "automatic_backend_fallback": False,
        "maximum_waterdepth": switch,
        "topology_fingerprint": topology,
        "cell_support": cell,
        "trace_support": trace,
        "physical_profile": physical,
        "physical_profile_fingerprint": _fingerprint(physical),
        "reference_sources": reference,
        "implementation_sources": implementation,
        "source_manifest_fingerprint": _fingerprint(source_body),
        "normal_product_admission": False,
    }
    return {**body, "fingerprint": _fingerprint(body)}


def load_direct_vector_maximum_waterdepth_execution_v68(
    value: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise DirectVectorMaximumWaterdepthV68Error(
            "Q3-H6c/v68 execution identity must be an object"
        )
    try:
        rebuilt = direct_vector_maximum_waterdepth_execution_document_v68(
            parent_execution=value["parent_execution"],
            enabled=value["maximum_waterdepth"],
            topology_fingerprint=value["topology_fingerprint"],
            cell_support=value["cell_support"],
            trace_support=value["trace_support"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise DirectVectorMaximumWaterdepthV68Error(
            "Invalid Q3-H6c/v68 execution identity"
        ) from error
    if dict(value) != rebuilt or _canonical(value) != _canonical(rebuilt):
        raise DirectVectorMaximumWaterdepthV68Error(
            "Q3-H6c/v68 execution identity is stale or forged"
        )
    return rebuilt


def verify_direct_vector_maximum_waterdepth_sources_v68(
    execution: Mapping[str, object], *, repository_root: str | Path | None = None,
) -> dict[str, object]:
    resolved = load_direct_vector_maximum_waterdepth_execution_v68(execution)
    root = (Path(repository_root) if repository_root is not None else
            Path(__file__).resolve().parents[2])
    parent_receipt = verify_direct_vector_family_sources_v67(
        resolved["parent_execution"], repository_root=root
    )
    rows = []
    for category in ("reference_sources", "implementation_sources"):
        for source in resolved[category]:
            path = root/source["path"]
            if not path.is_file():
                raise DirectVectorMaximumWaterdepthV68Error(
                    f"Q3-H6c/v68 authenticated source is missing: {source['path']}"
                )
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != source["sha256"]:
                raise DirectVectorMaximumWaterdepthV68Error(
                    f"Q3-H6c/v68 authenticated source differs: {source['path']}"
                )
            rows.append({**source, "category": category, "verified": True})
    body = {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "execution_fingerprint": resolved["fingerprint"],
        "source_manifest_fingerprint": resolved["source_manifest_fingerprint"],
        "parent_source_receipt_fingerprint": parent_receipt["fingerprint"],
        "sources": rows,
        "delft3d_source_access": "read_only_hash_verification",
    }
    return {**body, "fingerprint": _fingerprint(body)}


def _fp64(value: object, name: str, size: int | None = None) -> np.ndarray:
    result = np.asarray(value)
    if (result.dtype != np.float64 or result.ndim != 1 or
            not result.flags.c_contiguous or
            (size is not None and result.size != size)):
        suffix = "" if size is None else f" containing {size} values"
        raise DirectVectorMaximumWaterdepthV68Error(
            f"Q3-H6c/v68 {name} must be contiguous one-dimensional FP64{suffix}"
        )
    return result


def _i32(value: object, name: str, size: int | None = None) -> np.ndarray:
    result = np.asarray(value)
    if (result.dtype != np.int32 or result.ndim != 1 or
            not result.flags.c_contiguous or
            (size is not None and result.size != size)):
        raise DirectVectorMaximumWaterdepthV68Error(
            f"Q3-H6c/v68 {name} must be contiguous one-dimensional int32"
        )
    return result


def _i64(value: object, name: str, size: int | None = None) -> np.ndarray:
    result = np.asarray(value)
    if (result.dtype != np.int64 or result.ndim != 1 or
            not result.flags.c_contiguous or
            (size is not None and result.size != size)):
        raise DirectVectorMaximumWaterdepthV68Error(
            f"Q3-H6c/v68 {name} must be contiguous one-dimensional int64"
        )
    return result


def _immutable(value: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(value).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class DirectVectorHydraulicInputsV68:
    physical_depth: np.ndarray
    observation_depth: np.ndarray
    characteristic_u: np.ndarray
    characteristic_v: np.ndarray
    characteristic_speed: np.ndarray
    observation_height: np.ndarray
    effective_u: np.ndarray
    effective_v: np.ndarray
    effective_speed: np.ndarray
    active: np.ndarray
    status: np.ndarray
    maximum_waterdepth: bool
    preprocessor_application_count: int = 1

    def producer_velocity(self, method_id: str) -> np.ndarray:
        method = _method(method_id)
        if method == METHOD_VAN_RIJN_1993_CURRENT:
            return np.ascontiguousarray(np.stack(
                (self.characteristic_u, self.characteristic_v), axis=0
            ))
        return np.ascontiguousarray(np.stack(
            (self.effective_u, self.effective_v), axis=0
        ))


def prepare_direct_vector_hydraulics_v68(
    *, cell_depth: object, transport_u: object, transport_v: object,
    effective_speed: object, active: object, support_offset: object,
    support_depth: object, support_active: object,
    maximum_waterdepth: bool, direction_epsilon: float = FLOW_EPS,
    effective_direction_epsilon: float = FLOW_EPS,
) -> DirectVectorHydraulicInputsV68:
    """Reproduce ``dwnvel`` then EROSED's effective-current reconstruction."""
    switch = _strict_bool(maximum_waterdepth, "MaximumWaterdepth switch")
    depth = _fp64(cell_depth, "cell depth")
    count = int(depth.size)
    numerator_u = _fp64(transport_u, "transport-u numerator", count)
    numerator_v = _fp64(transport_v, "transport-v numerator", count)
    speed = _fp64(effective_speed, "effective speed", count)
    flags = _i32(active, "active flags", count)
    offset = _i64(support_offset, "support offset", count+1)
    links = _fp64(support_depth, "support depth")
    link_flags = _i32(support_active, "support activity", int(links.size))
    try:
        eps_effective = float(effective_direction_epsilon)
    except (TypeError, ValueError) as error:
        raise DirectVectorMaximumWaterdepthV68Error(
            "Q3-H6c/v68 effective direction epsilon must be numeric"
        ) from error
    if not math.isfinite(eps_effective) or eps_effective <= 0.0:
        raise DirectVectorMaximumWaterdepthV68Error(
            "Q3-H6c/v68 effective direction epsilon must be finite and positive"
        )

    # Exactly one source-order characteristic preprocessing operation.
    characteristic = maximum_waterdepth_characteristic_velocity_v66(
        cell_depth=depth, transport_u=numerator_u, transport_v=numerator_v,
        active=flags, support_offset=offset, support_depth=links,
        support_active=link_flags, enabled=switch,
        direction_epsilon=direction_epsilon,
    )
    status = np.array(characteristic.status, dtype=np.int32, copy=True)
    invalid_speed = (~np.isfinite(speed)) | (speed < 0.0)
    status[invalid_speed] = 1
    valid = (flags == 1) & (status == 0)
    denominator = characteristic.characteristic_speed + eps_effective
    effective_u = np.zeros(count, dtype=np.float64)
    effective_v = np.zeros(count, dtype=np.float64)
    effective_u[valid] = (
        speed[valid]*characteristic.characteristic_u[valid]/denominator[valid]
    )
    effective_v[valid] = (
        speed[valid]*characteristic.characteristic_v[valid]/denominator[valid]
    )
    resolved_speed = np.hypot(effective_u, effective_v)
    malformed = (~np.isfinite(effective_u)) | (~np.isfinite(effective_v))
    if np.any(malformed):
        status[malformed] = 1
        effective_u[malformed] = 0.0
        effective_v[malformed] = 0.0
        resolved_speed[malformed] = 0.0

    return DirectVectorHydraulicInputsV68(
        _immutable(depth), _immutable(characteristic.observation_depth),
        _immutable(characteristic.characteristic_u),
        _immutable(characteristic.characteristic_v),
        _immutable(characteristic.characteristic_speed),
        _immutable(characteristic.observation_height),
        _immutable(effective_u), _immutable(effective_v),
        _immutable(resolved_speed), _immutable(flags), _immutable(status),
        switch, 1,
    )


def producer_fields_v68(
    method_id: str, hydraulics: DirectVectorHydraulicInputsV68,
    fields: Mapping[str, object],
) -> dict[str, object]:
    """Inject only the source-owned hydraulic fields for one producer."""
    method = _method(method_id)
    if not isinstance(hydraulics, DirectVectorHydraulicInputsV68):
        raise DirectVectorMaximumWaterdepthV68Error(
            "Q3-H6c/v68 hydraulic preprocessing receipt is missing"
        )
    if hydraulics.preprocessor_application_count != 1:
        raise DirectVectorMaximumWaterdepthV68Error(
            "Q3-H6c/v68 characteristic preprocessing was not applied exactly once"
        )
    if not isinstance(fields, Mapping) or any(type(key) is not str for key in fields):
        raise DirectVectorMaximumWaterdepthV68Error(
            "Q3-H6c/v68 producer fields must be a string-keyed mapping"
        )
    result = dict(fields)
    result["depth"] = hydraulics.physical_depth
    if method == METHOD_VAN_RIJN_1993_CURRENT:
        result["velocity"] = hydraulics.producer_velocity(method)
        result["observation_height"] = hydraulics.observation_height
        result.pop("u", None)
        result.pop("v", None)
    else:
        result["u"] = hydraulics.effective_u
        result["v"] = hydraulics.effective_v
        result.pop("velocity", None)
        result.pop("observation_height", None)
    return result


class DirectVectorMaximumWaterdepthCPUBridgeV68:
    """Authenticated NumPy/OpenMP-side setup and hydraulic bridge."""

    backend = "cpu"

    def __init__(
        self, *, execution: Mapping[str, object], topology_fingerprint: str,
        cell_support_offset: object | None = None,
        cell_support_ids: object | None = None,
        trace_support_offset: object | None = None,
        trace_support_ids: object | None = None,
        verify_sources: bool = True,
    ):
        resolved = load_direct_vector_maximum_waterdepth_execution_v68(execution)
        if resolved["backend"] != "cpu":
            raise DirectVectorMaximumWaterdepthV68Error(
                "Q3-H6c/v68 CPU bridge requires an explicit CPU receipt"
            )
        topology = _sha(topology_fingerprint, "runtime topology fingerprint")
        if topology != resolved["topology_fingerprint"]:
            raise DirectVectorMaximumWaterdepthV68Error(
                "Q3-H6c/v68 runtime mesh topology differs"
            )
        enabled = bool(resolved["maximum_waterdepth"])
        arrays = (cell_support_offset, cell_support_ids,
                  trace_support_offset, trace_support_ids)
        if enabled and any(value is None for value in arrays):
            raise DirectVectorMaximumWaterdepthV68Error(
                "Q3-H6c/v68 enabled bridge requires complete support arrays"
            )
        if not enabled and any(value is not None for value in arrays):
            raise DirectVectorMaximumWaterdepthV68Error(
                "Q3-H6c/v68 MaximumWaterdepth-off rejects support arrays"
            )
        if enabled:
            cell_offset = _i64(cell_support_offset, "cell support offset")
            cell_ids = _i64(cell_support_ids, "cell support IDs")
            trace_offset = _i64(trace_support_offset, "trace support offset")
            trace_ids = _i64(trace_support_ids, "trace support IDs")
            verify_support_descriptor_v67(
                resolved["cell_support"], support_offset=cell_offset,
                support_ids=cell_ids,
            )
            verify_support_descriptor_v67(
                resolved["trace_support"], support_offset=trace_offset,
                support_ids=trace_ids,
            )
            self._offset = {"cell": cell_offset, "trace": trace_offset}
            self._link_count = {
                "cell": int(cell_ids.size), "trace": int(trace_ids.size),
            }
        else:
            self._offset = {"cell": None, "trace": None}
            self._link_count = {"cell": 0, "trace": 0}
        self.execution = resolved
        self.enabled = enabled
        self.source_receipt = (
            verify_direct_vector_maximum_waterdepth_sources_v68(resolved)
            if verify_sources else None
        )

    def prepare(
        self, kind: str, *, cell_depth: object, transport_u: object,
        transport_v: object, effective_speed: object, active: object,
        support_depth: object | None = None,
        support_active: object | None = None,
        direction_epsilon: float = FLOW_EPS,
        effective_direction_epsilon: float = FLOW_EPS,
    ) -> DirectVectorHydraulicInputsV68:
        if kind not in {"cell", "trace"}:
            raise DirectVectorMaximumWaterdepthV68Error(
                "Q3-H6c/v68 support kind must be cell or trace"
            )
        count = int(_fp64(cell_depth, "cell depth").size)
        if self.enabled:
            if support_depth is None or support_active is None:
                raise DirectVectorMaximumWaterdepthV68Error(
                    f"Q3-H6c/v68 enabled {kind} evaluation lacks support data"
                )
            depth_links = _fp64(
                support_depth, f"{kind} support depth", self._link_count[kind]
            )
            activity = _i32(
                support_active, f"{kind} support activity", self._link_count[kind]
            )
            offset = self._offset[kind]
            assert offset is not None
            if int(offset.size) != count+1:
                raise DirectVectorMaximumWaterdepthV68Error(
                    f"Q3-H6c/v68 {kind} point count differs from support receipt"
                )
        else:
            if support_depth is not None or support_active is not None:
                raise DirectVectorMaximumWaterdepthV68Error(
                    "Q3-H6c/v68 MaximumWaterdepth-off rejects support data"
                )
            offset = np.zeros(count+1, dtype=np.int64)
            depth_links = np.empty(0, dtype=np.float64)
            activity = np.empty(0, dtype=np.int32)
        return prepare_direct_vector_hydraulics_v68(
            cell_depth=cell_depth, transport_u=transport_u,
            transport_v=transport_v, effective_speed=effective_speed,
            active=active, support_offset=np.ascontiguousarray(offset),
            support_depth=np.ascontiguousarray(depth_links),
            support_active=np.ascontiguousarray(activity),
            maximum_waterdepth=self.enabled,
            direction_epsilon=direction_epsilon,
            effective_direction_epsilon=effective_direction_epsilon,
        )


__all__ = [
    "CATALOGUE_ENTRY_IDS", "DEFAULT_ENABLED", "ENVELOPE_VERSION", "FLOW_EPS",
    "LOCKED_SOURCES", "METHODS",
    "METHOD_EFFECTS", "SCHEMA", "SCOPE_ID", "SEMANTIC_ID",
    "DirectVectorHydraulicInputsV68",
    "DirectVectorMaximumWaterdepthCPUBridgeV68",
    "DirectVectorMaximumWaterdepthV68Error",
    "direct_vector_maximum_waterdepth_execution_document_v68",
    "direct_vector_maximum_waterdepth_semantic_document_v68",
    "load_direct_vector_maximum_waterdepth_execution_v68",
    "prepare_direct_vector_hydraulics_v68", "producer_fields_v68",
    "support_descriptor_document_v67",
    "verify_direct_vector_maximum_waterdepth_sources_v68",
]
