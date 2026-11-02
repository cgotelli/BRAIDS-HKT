"""Authenticated Q3-H6b admission for FLOW ``MaximumWaterdepth``.

The v66 operator reproduces the characteristic-velocity divisor in classic
FLOW.  This additive v67 envelope makes that operator admissible in the
bounded direct-vector lifecycle without changing the v66 receipt or runtime.
It binds the parent execution, the prepared mesh identity, and the exact CSR
support ordering.  Dynamic support depths and wet flags are intentionally not
part of the execution identity; their *layout* is.

The switch is off by default.  An off execution rejects support arrays rather
than accepting inert data.  An on execution requires both cell and boundary
trace support descriptors and verifies their bytes before evaluation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from collections.abc import Mapping

import numpy as np

from .direct_vector_live_v66 import (
    DirectVectorV66State,
    load_direct_vector_live_execution_v66,
    load_state_checkpoint_v66,
    state_checkpoint_document_v66,
    verify_direct_vector_sources_v66,
)
from .maximum_waterdepth_v66 import (
    LOCKED_SOURCES,
    SEMANTIC_ID as OPERATOR_SEMANTIC_ID,
    MaximumWaterdepthResultV66,
    maximum_waterdepth_characteristic_velocity_v66,
)


SCHEMA = 1
ENVELOPE_VERSION = 67
SCOPE_ID = "braids-q3h6b-maximum-waterdepth-live-v67-v1"
DEFAULT_ENABLED = False
_SHA = re.compile(r"[0-9a-f]{64}\Z")

_CPU_IMPLEMENTATION_SOURCES = {
    "maximum_waterdepth_host_oracle": (
        "braids_core/sediment/maximum_waterdepth_v66.py",
        "a6edc8384fe5701ad821116dc9318267bf0cb76fd6abfd45903412bb03b58e05",
    ),
    "maximum_waterdepth_openmp": (
        "solver/wave/braids_maximum_waterdepth_v66_module.f90",
        "4b5d756b21d1f35178a1eb71fd506e6899ae2f5ccbaecdcd9b6fd5b105980e8d",
    ),
}

_GPU_IMPLEMENTATION_SOURCES = {
    "maximum_waterdepth_host_oracle": (
        "braids_core/sediment/maximum_waterdepth_v66.py",
        "a6edc8384fe5701ad821116dc9318267bf0cb76fd6abfd45903412bb03b58e05",
    ),
    "maximum_waterdepth_cuda_wrapper": (
        "braids_gpu/maximum_waterdepth_v66.py",
        "32d99e7dbcd05aca4d7a673cfd2ea0414022b28ac0b18161d39b28a728e84054",
    ),
    "maximum_waterdepth_cuda_kernel": (
        "braids_gpu/maximum_waterdepth_v66.cu",
        "f3dcf13d1bb99f81f8b330d852bc96a146be3b1154c23edc50e33cd1384d1d35",
    ),
    "maximum_waterdepth_capacity_integration": (
        "braids_gpu/scalar_capacity_runtime.py",
        "566c1baff6bdf9462ba7bdf006aab80033324e6012bd4fd839f30d4f6d3c65ab",
    ),
    "maximum_waterdepth_capacity_kernels": (
        "braids_gpu/scalar_capacity_runtime.cu",
        "7144971b9736f43b484f4edd50039b9dc3a2ab7389357a30c15fed4a078935fc",
    ),
    "maximum_waterdepth_v67_admission": (
        "braids_gpu/maximum_waterdepth_live_v67.py",
        # Filled from the immutable wrapper bytes.  This is deliberately in
        # the core manifest so the wrapper does not attempt a self hash.
        "5e003a5f1ccbe3f0dd006aa8018ad34e8ef95b184b2b7d92bad1b18e7b8ed363",
    ),
}


class MaximumWaterdepthAdmissionV67Error(ValueError):
    """Fail-closed execution, support, source, or restart error."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _fingerprint(value: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise MaximumWaterdepthAdmissionV67Error(
            f"Q3-H6b/v67 {name} must be a lowercase SHA-256 fingerprint")
    return value


def _strict_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise MaximumWaterdepthAdmissionV67Error(
            f"Q3-H6b/v67 {name} must be Boolean")
    return value


def _int64_vector(value: object, name: str) -> np.ndarray:
    array = np.asarray(value)
    if (array.dtype != np.int64 or array.ndim != 1 or
            not array.flags.c_contiguous):
        raise MaximumWaterdepthAdmissionV67Error(
            f"Q3-H6b/v67 {name} must be contiguous one-dimensional int64")
    return array


def _array_sha256(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(_canonical(list(array.shape)))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def support_descriptor_document_v67(
    support_offset: object, support_ids: object,
) -> dict[str, object]:
    """Content-address one canonical CSR support ordering.

    ``support_ids`` are stable velocity-point identifiers prepared with the
    mesh.  Repeated identifiers across different cells are valid; the ordered
    CSR bytes, not set membership, define the dwnvel neighbourhood.
    """
    offset = _int64_vector(support_offset, "support offset")
    ids = _int64_vector(support_ids, "support IDs")
    if offset.size < 1:
        raise MaximumWaterdepthAdmissionV67Error(
            "Q3-H6b/v67 support offset may not be empty")
    if (int(offset[0]) != 0 or int(offset[-1]) != int(ids.size) or
            np.any(offset[1:] < offset[:-1])):
        raise MaximumWaterdepthAdmissionV67Error(
            "Q3-H6b/v67 support offset is not canonical CSR")
    if np.any(ids < 0):
        raise MaximumWaterdepthAdmissionV67Error(
            "Q3-H6b/v67 support IDs must be nonnegative")
    body: dict[str, object] = {
        "schema": SCHEMA,
        "point_count": int(offset.size - 1),
        "link_count": int(ids.size),
        "offset_sha256": _array_sha256(offset),
        "support_ids_sha256": _array_sha256(ids),
        "ordering": "ordered_csr_velocity_point_ids",
    }
    return {**body, "fingerprint": _fingerprint(body)}


def load_support_descriptor_v67(value: Mapping[str, object]) -> dict[str, object]:
    expected = {
        "schema", "point_count", "link_count", "offset_sha256",
        "support_ids_sha256", "ordering", "fingerprint",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise MaximumWaterdepthAdmissionV67Error(
            "Q3-H6b/v67 support descriptor inventory differs")
    try:
        point_count = value["point_count"]
        link_count = value["link_count"]
        if (type(point_count) is not int or point_count < 0 or
                type(link_count) is not int or link_count < 0):
            raise TypeError
        body: dict[str, object] = {
            "schema": SCHEMA,
            "point_count": point_count,
            "link_count": link_count,
            "offset_sha256": _sha(value["offset_sha256"], "offset hash"),
            "support_ids_sha256": _sha(
                value["support_ids_sha256"], "support-ID hash"),
            "ordering": "ordered_csr_velocity_point_ids",
        }
    except (KeyError, TypeError, ValueError) as error:
        raise MaximumWaterdepthAdmissionV67Error(
            "Invalid Q3-H6b/v67 support descriptor") from error
    rebuilt = {**body, "fingerprint": _fingerprint(body)}
    if dict(value) != rebuilt or _canonical(value) != _canonical(rebuilt):
        raise MaximumWaterdepthAdmissionV67Error(
            "Q3-H6b/v67 support descriptor is stale or forged")
    return rebuilt


def verify_support_descriptor_v67(
    expected: Mapping[str, object], *, support_offset: object,
    support_ids: object,
) -> dict[str, object]:
    resolved = load_support_descriptor_v67(expected)
    actual = support_descriptor_document_v67(support_offset, support_ids)
    if actual != resolved or _canonical(actual) != _canonical(resolved):
        raise MaximumWaterdepthAdmissionV67Error(
            "Q3-H6b/v67 prepared support topology differs")
    return resolved


def maximum_waterdepth_live_semantic_document_v67() -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "envelope_version": ENVELOPE_VERSION,
        "operator_semantic_id": OPERATOR_SEMANTIC_ID,
        "default_enabled": DEFAULT_ENABLED,
        "enabled_contract": (
            "mesh_fingerprint_and_ordered_cell_and_trace_CSR_support_required"
        ),
        "disabled_contract": "all_support_arrays_and_descriptors_rejected",
        "dynamic_inputs": "support_depth_and_support_activity_not_identity",
        "checkpoint": (
            "v67_execution_bound_envelope_around_complete_v66_accepted_state"
        ),
        "backend_choice": "inherited_explicit_parent_choice_no_fallback",
        "accepted_step_limit": 20,
        "normal_product_admission": False,
    }


SEMANTIC_ID = _fingerprint(maximum_waterdepth_live_semantic_document_v67())


def _source_rows(backend: str) -> tuple[list[dict], list[dict]]:
    reference = [
        {
            "name": name,
            "path": row["path"],
            "sha256": row["sha256"],
            "lines": row["lines"],
        }
        for name, row in sorted(LOCKED_SOURCES.items())
    ]
    sources = (_CPU_IMPLEMENTATION_SOURCES if backend == "cpu"
               else _GPU_IMPLEMENTATION_SOURCES)
    implementation = [
        {"name": name, "path": path, "sha256": digest}
        for name, (path, digest) in sorted(sources.items())
    ]
    return reference, implementation


def maximum_waterdepth_live_execution_document_v67(
    *, parent_execution: Mapping[str, object], enabled: bool = DEFAULT_ENABLED,
    topology_fingerprint: str,
    cell_support: Mapping[str, object] | None = None,
    trace_support: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Derive a v67 identity from one authenticated direct-vector receipt."""
    parent = load_direct_vector_live_execution_v66(parent_execution)
    switch = _strict_bool(enabled, "MaximumWaterdepth switch")
    topology = _sha(topology_fingerprint, "mesh topology fingerprint")
    if parent["physical_profile"]["maximum_waterdepth"] is not False:
        raise MaximumWaterdepthAdmissionV67Error(
            "Q3-H6b/v67 parent must be the unmodified v66 off receipt")
    if switch:
        if cell_support is None or trace_support is None:
            raise MaximumWaterdepthAdmissionV67Error(
                "Q3-H6b/v67 enabled execution requires cell and trace support")
        cell = load_support_descriptor_v67(cell_support)
        trace = load_support_descriptor_v67(trace_support)
    else:
        if cell_support is not None or trace_support is not None:
            raise MaximumWaterdepthAdmissionV67Error(
                "Q3-H6b/v67 disabled execution rejects support descriptors")
        cell = None
        trace = None
    reference, implementation = _source_rows(parent["backend"])
    physical = dict(parent["physical_profile"])
    physical.update({
        "maximum_waterdepth": switch,
        "maximum_waterdepth_default": DEFAULT_ENABLED,
        "maximum_waterdepth_operator_semantic_id": OPERATOR_SEMANTIC_ID,
        "topology_fingerprint": topology,
        "cell_support_fingerprint": None if cell is None else cell["fingerprint"],
        "trace_support_fingerprint": None if trace is None else trace["fingerprint"],
    })
    body: dict[str, object] = {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "envelope_version": ENVELOPE_VERSION,
        "parent_execution": parent,
        "parent_execution_fingerprint": parent["fingerprint"],
        "backend": parent["backend"],
        "integrator": parent["integrator"],
        "automatic_backend_fallback": False,
        "maximum_waterdepth": switch,
        "maximum_waterdepth_default": DEFAULT_ENABLED,
        "topology_fingerprint": topology,
        "cell_support": cell,
        "trace_support": trace,
        "physical_profile": physical,
        "physical_profile_fingerprint": _fingerprint(physical),
        "reference_sources": reference,
        "implementation_sources": implementation,
        "accepted_step_limit": parent["accepted_step_limit"],
        "normal_product_admission": False,
    }
    return {**body, "fingerprint": _fingerprint(body)}


def load_maximum_waterdepth_live_execution_v67(
    value: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise MaximumWaterdepthAdmissionV67Error(
            "Q3-H6b/v67 execution identity must be an object")
    try:
        rebuilt = maximum_waterdepth_live_execution_document_v67(
            parent_execution=value["parent_execution"],
            enabled=value["maximum_waterdepth"],
            topology_fingerprint=value["topology_fingerprint"],
            cell_support=value["cell_support"],
            trace_support=value["trace_support"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise MaximumWaterdepthAdmissionV67Error(
            "Invalid Q3-H6b/v67 execution identity") from error
    if dict(value) != rebuilt or _canonical(value) != _canonical(rebuilt):
        raise MaximumWaterdepthAdmissionV67Error(
            "Q3-H6b/v67 execution identity is stale or forged")
    return rebuilt


def verify_maximum_waterdepth_sources_v67(
    execution: Mapping[str, object], *, repository_root: str | Path | None = None,
) -> dict[str, object]:
    """Verify the parent receipt plus read-only reference/operator bytes."""
    resolved = load_maximum_waterdepth_live_execution_v67(execution)
    root = (Path(repository_root) if repository_root is not None else
            Path(__file__).resolve().parents[2])
    parent_receipt = verify_direct_vector_sources_v66(
        resolved["parent_execution"], repository_root=root)
    rows = []
    for category in ("reference_sources", "implementation_sources"):
        for source in resolved[category]:
            path = root/source["path"]
            if not path.is_file():
                raise MaximumWaterdepthAdmissionV67Error(
                    f"Q3-H6b/v67 authenticated source is missing: {source['path']}")
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != source["sha256"]:
                raise MaximumWaterdepthAdmissionV67Error(
                    f"Q3-H6b/v67 authenticated source differs: {source['path']}")
            rows.append({**source, "category": category, "verified": True})
    body: dict[str, object] = {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "execution_fingerprint": resolved["fingerprint"],
        "parent_source_receipt_fingerprint": parent_receipt["fingerprint"],
        "sources": rows,
        "delft3d_source_access": "read_only_hash_verification",
    }
    return {**body, "fingerprint": _fingerprint(body)}


def _validate_runtime_binding(
    execution: Mapping[str, object], *, topology_fingerprint: str,
    cell_support_offset: object | None, cell_support_ids: object | None,
    trace_support_offset: object | None, trace_support_ids: object | None,
) -> dict[str, object]:
    resolved = load_maximum_waterdepth_live_execution_v67(execution)
    if _sha(topology_fingerprint, "runtime topology fingerprint") != \
            resolved["topology_fingerprint"]:
        raise MaximumWaterdepthAdmissionV67Error(
            "Q3-H6b/v67 runtime mesh topology differs")
    arrays = (cell_support_offset, cell_support_ids,
              trace_support_offset, trace_support_ids)
    if not resolved["maximum_waterdepth"]:
        if any(item is not None for item in arrays):
            raise MaximumWaterdepthAdmissionV67Error(
                "Q3-H6b/v67 MaximumWaterdepth-off rejects support arrays")
        return resolved
    if any(item is None for item in arrays):
        raise MaximumWaterdepthAdmissionV67Error(
            "Q3-H6b/v67 MaximumWaterdepth-on requires complete support arrays")
    verify_support_descriptor_v67(
        resolved["cell_support"], support_offset=cell_support_offset,
        support_ids=cell_support_ids)
    verify_support_descriptor_v67(
        resolved["trace_support"], support_offset=trace_support_offset,
        support_ids=trace_support_ids)
    return resolved


class MaximumWaterdepthCPUAdmissionV67:
    """Authenticated host/OpenMP-oracle admission for bounded CPU use."""

    def __init__(
        self, *, execution: Mapping[str, object], topology_fingerprint: str,
        cell_support_offset: object | None = None,
        cell_support_ids: object | None = None,
        trace_support_offset: object | None = None,
        trace_support_ids: object | None = None,
        verify_sources: bool = True,
    ):
        self.execution = _validate_runtime_binding(
            execution, topology_fingerprint=topology_fingerprint,
            cell_support_offset=cell_support_offset,
            cell_support_ids=cell_support_ids,
            trace_support_offset=trace_support_offset,
            trace_support_ids=trace_support_ids)
        if self.execution["backend"] != "cpu":
            raise MaximumWaterdepthAdmissionV67Error(
                "Q3-H6b/v67 CPU admission requires a CPU parent")
        self.source_receipt = (verify_maximum_waterdepth_sources_v67(
            self.execution) if verify_sources else None)
        self.enabled = bool(self.execution["maximum_waterdepth"])
        self._offset = {
            "cell": cell_support_offset,
            "trace": trace_support_offset,
        }

    def _apply(
        self, kind: str, *, cell_depth: object, transport_u: object,
        transport_v: object, active: object,
        support_depth: object | None = None,
        support_active: object | None = None,
        direction_epsilon: float = 1.0e-12,
    ) -> MaximumWaterdepthResultV66:
        if kind not in {"cell", "trace"}:
            raise AssertionError(kind)
        depth = np.asarray(cell_depth)
        if depth.dtype != np.float64 or depth.ndim != 1 or \
                not depth.flags.c_contiguous:
            raise MaximumWaterdepthAdmissionV67Error(
                f"Q3-H6b/v67 {kind} depth must be contiguous FP64")
        if self.enabled:
            if support_depth is None or support_active is None:
                raise MaximumWaterdepthAdmissionV67Error(
                    f"Q3-H6b/v67 enabled {kind} evaluation lacks support data")
            offset = self._offset[kind]
        else:
            if support_depth is not None or support_active is not None:
                raise MaximumWaterdepthAdmissionV67Error(
                    "Q3-H6b/v67 MaximumWaterdepth-off rejects support arrays")
            offset = np.zeros(depth.size + 1, dtype=np.int64)
            support_depth = np.empty(0, dtype=np.float64)
            support_active = np.empty(0, dtype=np.int32)
        return maximum_waterdepth_characteristic_velocity_v66(
            cell_depth=depth, transport_u=transport_u,
            transport_v=transport_v, active=active,
            support_offset=offset, support_depth=support_depth,
            support_active=support_active, enabled=self.enabled,
            direction_epsilon=direction_epsilon)

    def cells(self, **values) -> MaximumWaterdepthResultV66:
        return self._apply("cell", **values)

    def traces(self, **values) -> MaximumWaterdepthResultV66:
        return self._apply("trace", **values)


def state_checkpoint_document_v67(
    state: DirectVectorV66State, *, execution: Mapping[str, object],
) -> dict[str, object]:
    resolved = load_maximum_waterdepth_live_execution_v67(execution)
    if (not isinstance(state, DirectVectorV66State) or
            state.execution_fingerprint != resolved["parent_execution_fingerprint"]):
        raise MaximumWaterdepthAdmissionV67Error(
            "Q3-H6b/v67 checkpoint state belongs to another parent execution")
    base = state_checkpoint_document_v66(state)
    body: dict[str, object] = {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "envelope_version": ENVELOPE_VERSION,
        "stage": "checkpoint",
        "execution_fingerprint": resolved["fingerprint"],
        "parent_execution_fingerprint": resolved["parent_execution_fingerprint"],
        "topology_fingerprint": resolved["topology_fingerprint"],
        "cell_support_fingerprint": (
            None if resolved["cell_support"] is None
            else resolved["cell_support"]["fingerprint"]),
        "trace_support_fingerprint": (
            None if resolved["trace_support"] is None
            else resolved["trace_support"]["fingerprint"]),
        "maximum_waterdepth": resolved["maximum_waterdepth"],
        "parent_checkpoint": base,
        "transaction_schema": "complete_accepted_state_no_trial_arrays",
    }
    return {**body, "fingerprint": _fingerprint(body)}


def unwrap_state_checkpoint_v67(
    value: Mapping[str, object], *, execution: Mapping[str, object],
) -> dict[str, object]:
    resolved = load_maximum_waterdepth_live_execution_v67(execution)
    expected = {
        "schema", "scope_id", "semantic_id", "envelope_version", "stage",
        "execution_fingerprint", "parent_execution_fingerprint",
        "topology_fingerprint", "cell_support_fingerprint",
        "trace_support_fingerprint", "maximum_waterdepth",
        "parent_checkpoint", "transaction_schema", "fingerprint",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise MaximumWaterdepthAdmissionV67Error(
            "Q3-H6b/v67 checkpoint inventory differs")
    body = {key: item for key, item in value.items() if key != "fingerprint"}
    expected_cell = (None if resolved["cell_support"] is None
                     else resolved["cell_support"]["fingerprint"])
    expected_trace = (None if resolved["trace_support"] is None
                      else resolved["trace_support"]["fingerprint"])
    if (value["fingerprint"] != _fingerprint(body) or
            value["schema"] != SCHEMA or value["scope_id"] != SCOPE_ID or
            value["semantic_id"] != SEMANTIC_ID or
            value["envelope_version"] != ENVELOPE_VERSION or
            value["stage"] != "checkpoint" or
            value["execution_fingerprint"] != resolved["fingerprint"] or
            value["parent_execution_fingerprint"] !=
            resolved["parent_execution_fingerprint"] or
            value["topology_fingerprint"] != resolved["topology_fingerprint"] or
            value["cell_support_fingerprint"] != expected_cell or
            value["trace_support_fingerprint"] != expected_trace or
            value["maximum_waterdepth"] is not
            resolved["maximum_waterdepth"] or
            value["transaction_schema"] !=
            "complete_accepted_state_no_trial_arrays"):
        raise MaximumWaterdepthAdmissionV67Error(
            "Q3-H6b/v67 checkpoint header or execution differs")
    state = load_state_checkpoint_v66(
        value["parent_checkpoint"], execution=resolved["parent_execution"])
    canonical = state_checkpoint_document_v66(state)
    if value["parent_checkpoint"] != canonical or \
            _canonical(value["parent_checkpoint"]) != _canonical(canonical):
        raise MaximumWaterdepthAdmissionV67Error(
            "Q3-H6b/v67 parent checkpoint is stale or forged")
    return canonical


def load_state_checkpoint_v67(
    value: Mapping[str, object], *, execution: Mapping[str, object],
) -> DirectVectorV66State:
    resolved = load_maximum_waterdepth_live_execution_v67(execution)
    base = unwrap_state_checkpoint_v67(value, execution=resolved)
    return load_state_checkpoint_v66(base, execution=resolved["parent_execution"])


__all__ = [
    "DEFAULT_ENABLED", "ENVELOPE_VERSION", "SCHEMA", "SCOPE_ID",
    "SEMANTIC_ID", "MaximumWaterdepthAdmissionV67Error",
    "MaximumWaterdepthCPUAdmissionV67",
    "load_maximum_waterdepth_live_execution_v67",
    "load_state_checkpoint_v67", "load_support_descriptor_v67",
    "maximum_waterdepth_live_execution_document_v67",
    "maximum_waterdepth_live_semantic_document_v67",
    "state_checkpoint_document_v67", "support_descriptor_document_v67",
    "unwrap_state_checkpoint_v67", "verify_maximum_waterdepth_sources_v67",
    "verify_support_descriptor_v67",
]
