"""Q3-H3/v59 accepted-transaction and schema-10 restart contracts.

This module is deliberately backend neutral.  CPU and CUDA owners export
committed host arrays into the same strict envelope; the envelope neither
advances physics nor permits a backend fallback.  It closes the reusable part
of T1/R1 while the selected-mesh routes retain ownership of device rollback
and atomic filesystem publication.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from .prognostic_secondary_amr_execution import (
    CHECKPOINT_SCHEMA,
    TIME_COUPLING,
    canonical_amr_instance_fingerprints,
    validate_prognostic_secondary_amr_execution_document,
)


REQUIRED_LEDGER_FIELDS = frozenset({
    "water_boundary_m3",
    "sediment_boundary_kg",
    "secondary_advective_boundary_m4_s",
    "secondary_diffusive_boundary_m4_s",
    "betac_momentum_boundary_m4_s",
    "coriolis_momentum_m4_s",
})
REGISTER_COMPONENTS = (
    "advective_h_i_m4_s",
    "diffusive_h_i_m4_s",
    "betac_velocity_m3_s",
)


class PrognosticSecondaryAMRLifecycleError(ValueError):
    """Fail-closed lifecycle/checkpoint error."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha(value: object, name: str) -> str:
    if (not isinstance(value, str) or len(value) != 64 or
            any(character not in "0123456789abcdef" for character in value)):
        raise PrognosticSecondaryAMRLifecycleError(
            f"Q3-H3/schema-10 {name} must be a lowercase SHA-256"
        )
    return value


def _json_copy(value: object, name: str):
    try:
        return json.loads(_canonical(value).decode("ascii"))
    except (TypeError, ValueError, OverflowError) as error:
        raise PrognosticSecondaryAMRLifecycleError(
            f"Q3-H3/schema-10 {name} is not finite canonical JSON"
        ) from error


def _array(value: object, name: str, dtype, ndim: int) -> np.ndarray:
    item = np.asarray(value)
    expected = np.dtype(dtype)
    if (item.dtype != expected or item.ndim != ndim or
            not item.flags.c_contiguous or
            (item.dtype.kind == "f" and not np.isfinite(item).all())):
        raise PrognosticSecondaryAMRLifecycleError(
            f"Q3-H3/schema-10 array differs: {name}"
        )
    result = np.array(item, copy=True, order="C")
    result.setflags(write=False)
    return result


def _array_sha(name: str, value: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(_canonical(list(value.shape)))
    digest.update(b"\0")
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class RegisterAudit:
    component: str
    topology_fingerprint: str
    mesh_epoch: int
    face_source_ids: np.ndarray
    values: np.ndarray
    pending: bool
    digest: str


def register_audit(
    component: str,
    *,
    topology_fingerprint: str,
    mesh_epoch: int,
    face_source_ids: object,
    values: object,
    pending: bool = False,
) -> RegisterAudit:
    """Freeze one topology-tagged completed physical-face register."""
    if component not in REGISTER_COMPONENTS:
        raise PrognosticSecondaryAMRLifecycleError(
            "Q3-H3/schema-10 register component differs"
        )
    topology = _sha(topology_fingerprint, "register topology fingerprint")
    if isinstance(mesh_epoch, bool) or not isinstance(mesh_epoch, int) or mesh_epoch < 0:
        raise PrognosticSecondaryAMRLifecycleError(
            "Q3-H3/schema-10 register mesh epoch differs"
        )
    if not isinstance(pending, bool):
        raise PrognosticSecondaryAMRLifecycleError(
            "Q3-H3/schema-10 register pending flag differs"
        )
    ids = _array(face_source_ids, "register face source IDs", np.int64, 1)
    if ids.size == 0 or np.any(ids <= 0) or np.any(np.diff(ids) <= 0):
        raise PrognosticSecondaryAMRLifecycleError(
            "Q3-H3/schema-10 register face source IDs differ"
        )
    value = np.asarray(values)
    expected_shape = ((ids.size, 2) if component == "betac_velocity_m3_s"
                      else (ids.size,))
    if value.shape != expected_shape:
        raise PrognosticSecondaryAMRLifecycleError(
            "Q3-H3/schema-10 register value shape differs"
        )
    frozen = _array(value, "register values", np.float64, len(expected_shape))
    document = {
        "component": component,
        "topology_fingerprint": topology,
        "mesh_epoch": mesh_epoch,
        "pending": pending,
        "face_source_ids_sha256": _array_sha("face_source_ids", ids),
        "values_sha256": _array_sha("values", frozen),
    }
    return RegisterAudit(
        component, topology, mesh_epoch, ids, frozen, pending,
        hashlib.sha256(_canonical(document)).hexdigest(),
    )


@dataclass(frozen=True)
class AcceptedSnapshot:
    metadata: Mapping[str, object]
    arrays: Mapping[str, np.ndarray]
    digest: str


def accepted_snapshot(
    arrays: Mapping[str, object], metadata: Mapping[str, object]
) -> AcceptedSnapshot:
    """Freeze host-visible accepted bytes for a T1 rollback assertion."""
    if not isinstance(arrays, Mapping) or not arrays:
        raise PrognosticSecondaryAMRLifecycleError(
            "Q3-H3 accepted snapshot arrays are absent"
        )
    frozen: dict[str, np.ndarray] = {}
    for name, value in arrays.items():
        if not isinstance(name, str) or not name:
            raise PrognosticSecondaryAMRLifecycleError(
                "Q3-H3 accepted snapshot array name differs"
            )
        item = np.asarray(value)
        if (item.dtype.kind not in "iufb" or not item.flags.c_contiguous or
                (item.dtype.kind == "f" and not np.isfinite(item).all())):
            raise PrognosticSecondaryAMRLifecycleError(
                "Q3-H3 accepted snapshot array differs: " + name
            )
        copy = np.array(item, copy=True, order="C")
        copy.setflags(write=False)
        frozen[name] = copy
    meta = _json_copy(metadata, "accepted snapshot metadata")
    document = {
        "metadata": meta,
        "arrays": {
            name: {
                "dtype": value.dtype.str,
                "shape": list(value.shape),
                "sha256": _array_sha(name, value),
            }
            for name, value in sorted(frozen.items())
        },
    }
    return AcceptedSnapshot(
        MappingProxyType(meta), MappingProxyType(frozen),
        hashlib.sha256(_canonical(document)).hexdigest(),
    )


def assert_exact_rollback(
    snapshot: AcceptedSnapshot,
    arrays: Mapping[str, object],
    metadata: Mapping[str, object],
) -> None:
    """Reject any byte or metadata change after cancellation/rejection."""
    candidate = accepted_snapshot(arrays, metadata)
    if candidate.digest != snapshot.digest:
        raise PrognosticSecondaryAMRLifecycleError(
            "Q3-H3 accepted rollback changed state, topology, ledger or clock bytes"
        )


@dataclass(frozen=True)
class Schema10Checkpoint:
    document: Mapping[str, object]
    arrays: Mapping[str, np.ndarray]
    completed_registers: tuple[RegisterAudit, ...]
    payload_digest: str


def schema10_checkpoint(
    *,
    execution_identity: Mapping[str, object],
    topology_document: object,
    face_catalog_document: object,
    boundary_catalog_document: object,
    tree_parent: object,
    tree_level: object,
    tree_area_m2: object,
    active_node_ids: object,
    active_state: object,
    tree_state: object,
    selector_delay: object,
    secondary_state_index: int,
    mesh_epoch: int,
    ledgers: Mapping[str, object],
    controller_state: Mapping[str, object],
    predictor_arrays: Mapping[str, object],
    regrid_events: Sequence[object],
    completed_registers: Sequence[RegisterAudit],
    time_s: float,
    accepted_steps: int,
    maximum_cfl: float,
    trial_open: bool = False,
    pending_register: bool = False,
) -> Schema10Checkpoint:
    """Build one committed schema-10 payload shared by CPU and GPU routes."""
    identity = validate_prognostic_secondary_amr_execution_document(
        execution_identity)
    fingerprints = canonical_amr_instance_fingerprints(
        topology_document=topology_document,
        face_catalog_document=face_catalog_document,
        boundary_catalog_document=boundary_catalog_document,
    )
    for name, value in fingerprints.items():
        if identity[name] != value:
            raise PrognosticSecondaryAMRLifecycleError(
                "Q3-H3/schema-10 catalog fingerprint differs: " + name
            )
    if trial_open is not False or pending_register is not False:
        raise PrognosticSecondaryAMRLifecycleError(
            "Q3-H3/schema-10 requires a committed point with no pending register"
        )
    if (isinstance(mesh_epoch, bool) or not isinstance(mesh_epoch, int) or
            mesh_epoch < 0 or isinstance(accepted_steps, bool) or
            not isinstance(accepted_steps, int) or accepted_steps < 0 or
            isinstance(secondary_state_index, bool) or
            not isinstance(secondary_state_index, int)):
        raise PrognosticSecondaryAMRLifecycleError(
            "Q3-H3/schema-10 integer clock/index differs"
        )
    if (isinstance(time_s, bool) or not isinstance(time_s, (int, float)) or
            not np.isfinite(time_s) or time_s < 0.0 or
            isinstance(maximum_cfl, bool) or
            not isinstance(maximum_cfl, (int, float)) or
            not np.isfinite(maximum_cfl) or maximum_cfl < 0.0 or
            ((accepted_steps == 0) != (float(time_s) == 0.0))):
        raise PrognosticSecondaryAMRLifecycleError(
            "Q3-H3/schema-10 accepted clock differs"
        )

    parent = _array(tree_parent, "tree parent", np.int64, 1)
    level = _array(tree_level, "tree level", np.int64, 1)
    area = _array(tree_area_m2, "tree area", np.float64, 1)
    active = _array(active_node_ids, "active node IDs", np.int64, 1)
    state = _array(active_state, "active state", np.float64, 2)
    tree = _array(tree_state, "tree state", np.float64, 2)
    delay = _array(selector_delay, "selector delay", np.int64, 1)
    nn = parent.size
    if (nn == 0 or level.shape != (nn,) or area.shape != (nn,) or
            tree.shape[0] != nn or delay.shape != (nn,) or
            np.any(parent >= np.arange(nn)) or np.any(parent < -1) or
            np.any(level < 0) or np.any(area <= 0.0) or np.any(delay < 0) or
            active.size == 0 or np.any(np.diff(active) <= 0) or
            np.any((active < 0) | (active >= nn)) or
            state.shape != (active.size, tree.shape[1]) or
            not 0 <= secondary_state_index < state.shape[1] or
            not np.array_equal(tree[active], state)):
        raise PrognosticSecondaryAMRLifecycleError(
            "Q3-H3/schema-10 active/tree duplicate or hierarchy differs"
        )
    child = parent >= 0
    if np.any(level[child] != level[parent[child]] + 1):
        raise PrognosticSecondaryAMRLifecycleError(
            "Q3-H3/schema-10 tree levels differ"
        )

    if not isinstance(ledgers, Mapping) or not REQUIRED_LEDGER_FIELDS.issubset(ledgers):
        raise PrognosticSecondaryAMRLifecycleError(
            "Q3-H3/schema-10 cumulative ledger inventory differs"
        )
    arrays: dict[str, np.ndarray] = {
        "tree_parent": parent, "tree_level": level, "tree_area_m2": area,
        "active_node_ids": active, "active_state": state,
        "tree_state": tree, "selector_delay": delay,
    }
    for name, value in sorted(ledgers.items()):
        if not isinstance(name, str) or not name:
            raise PrognosticSecondaryAMRLifecycleError(
                "Q3-H3/schema-10 ledger name differs"
            )
        item = np.asarray(value)
        arrays["ledger/" + name] = _array(
            item, "ledger " + name, np.float64, item.ndim)
    if not isinstance(predictor_arrays, Mapping):
        raise PrognosticSecondaryAMRLifecycleError(
            "Q3-H3/schema-10 predictor inventory differs"
        )
    for name, value in sorted(predictor_arrays.items()):
        item = np.asarray(value)
        if item.dtype not in (np.dtype("float64"), np.dtype("int64")):
            raise PrognosticSecondaryAMRLifecycleError(
                "Q3-H3/schema-10 predictor dtype differs: " + name
            )
        arrays["predictor/" + name] = _array(
            item, "predictor " + name, item.dtype, item.ndim)

    registers = tuple(completed_registers)
    if ((accepted_steps == 0 and registers) or
            (accepted_steps > 0 and
             tuple(item.component for item in registers) != REGISTER_COMPONENTS) or
            any(not isinstance(item, RegisterAudit) or item.pending
                for item in registers)):
        raise PrognosticSecondaryAMRLifecycleError(
            "Q3-H3/schema-10 completed register inventory differs"
        )
    events = _json_copy(list(regrid_events), "regrid events")
    controller = _json_copy(controller_state, "controller state")
    catalog = {
        "topology": _json_copy(topology_document, "topology document"),
        "faces": _json_copy(face_catalog_document, "face catalog document"),
        "boundaries": _json_copy(boundary_catalog_document,
                                  "boundary catalog document"),
    }
    document = {
        "schema": CHECKPOINT_SCHEMA,
        # Keep the complete validated execution document in the additive
        # schema.  A fingerprint alone detects accidental corruption but is
        # insufficient for an offline restart to authenticate the parent,
        # source/profile hashes, numerical controls, backend and integrator
        # without consulting mutable run-directory state.
        "execution_identity": identity,
        "scope_id": identity["scope_id"],
        "execution_fingerprint": identity["fingerprint"],
        "backend": identity["backend"],
        "integrator": identity["integrator"],
        "cuda_device_id": identity["cuda_device_id"],
        "time_coupling": TIME_COUPLING,
        "mesh_epoch": mesh_epoch,
        "secondary_state_index": secondary_state_index,
        "time_s": float(time_s),
        "accepted_steps": accepted_steps,
        "maximum_cfl": float(maximum_cfl),
        "current_register_pending": False,
        "catalog_fingerprints": fingerprints,
        "catalog": catalog,
        "controller_state": controller,
        "regrid_events": events,
        "completed_registers": [
            {
                "component": item.component,
                "topology_fingerprint": item.topology_fingerprint,
                "mesh_epoch": item.mesh_epoch,
                "pending": item.pending,
                "digest": item.digest,
            }
            for item in registers
        ],
        "arrays": {
            name: {
                "dtype": value.dtype.str,
                "shape": list(value.shape),
                "sha256": _array_sha(name, value),
            }
            for name, value in sorted(arrays.items())
        },
    }
    digest = hashlib.sha256(_canonical(document)).hexdigest()
    return Schema10Checkpoint(
        MappingProxyType(document), MappingProxyType(arrays), registers, digest)


def validate_schema10_checkpoint(checkpoint: Schema10Checkpoint) -> None:
    """Recompute all bound array/register hashes before route allocation."""
    if not isinstance(checkpoint, Schema10Checkpoint):
        raise PrognosticSecondaryAMRLifecycleError(
            "Q3-H3/schema-10 checkpoint object differs"
        )
    document = dict(checkpoint.document)
    if (document.get("schema") != CHECKPOINT_SCHEMA or
            document.get("time_coupling") != TIME_COUPLING or
            document.get("current_register_pending") is not False):
        raise PrognosticSecondaryAMRLifecycleError(
            "Q3-H3/schema-10 checkpoint header differs"
        )
    try:
        identity = validate_prognostic_secondary_amr_execution_document(
            document.get("execution_identity")
        )
    except (TypeError, ValueError) as error:
        raise PrognosticSecondaryAMRLifecycleError(
            "Q3-H3/schema-10 execution identity differs"
        ) from error
    if (
        document.get("scope_id") != identity["scope_id"]
        or document.get("execution_fingerprint") != identity["fingerprint"]
        or document.get("backend") != identity["backend"]
        or document.get("integrator") != identity["integrator"]
        or document.get("cuda_device_id") != identity["cuda_device_id"]
    ):
        raise PrognosticSecondaryAMRLifecycleError(
            "Q3-H3/schema-10 execution header/identity differs"
        )
    catalog = document.get("catalog")
    if not isinstance(catalog, Mapping) or set(catalog) != {
        "topology", "faces", "boundaries"
    }:
        raise PrognosticSecondaryAMRLifecycleError(
            "Q3-H3/schema-10 catalog inventory differs"
        )
    fingerprints = canonical_amr_instance_fingerprints(
        topology_document=catalog["topology"],
        face_catalog_document=catalog["faces"],
        boundary_catalog_document=catalog["boundaries"],
    )
    if (
        document.get("catalog_fingerprints") != fingerprints
        or any(identity[name] != value for name, value in fingerprints.items())
    ):
        raise PrognosticSecondaryAMRLifecycleError(
            "Q3-H3/schema-10 catalog identity differs"
        )
    expected_arrays = document.get("arrays")
    actual_arrays = {
        name: {
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "sha256": _array_sha(name, value),
        }
        for name, value in sorted(checkpoint.arrays.items())
    }
    if expected_arrays != actual_arrays:
        raise PrognosticSecondaryAMRLifecycleError(
            "Q3-H3/schema-10 checkpoint array digest differs"
        )
    expected_registers = document.get("completed_registers")
    actual_registers = []
    for item in checkpoint.completed_registers:
        if not isinstance(item, RegisterAudit):
            raise PrognosticSecondaryAMRLifecycleError(
                "Q3-H3/schema-10 register object differs"
            )
        rebuilt = register_audit(
            item.component,
            topology_fingerprint=item.topology_fingerprint,
            mesh_epoch=item.mesh_epoch,
            face_source_ids=item.face_source_ids,
            values=item.values,
            pending=item.pending,
        )
        if rebuilt.digest != item.digest:
            raise PrognosticSecondaryAMRLifecycleError(
                "Q3-H3/schema-10 register payload digest differs"
            )
        actual_registers.append({
            "component": item.component,
            "topology_fingerprint": item.topology_fingerprint,
            "mesh_epoch": item.mesh_epoch,
            "pending": item.pending,
            "digest": item.digest,
        })
    if expected_registers != actual_registers:
        raise PrognosticSecondaryAMRLifecycleError(
            "Q3-H3/schema-10 checkpoint register digest differs"
        )
    if hashlib.sha256(_canonical(document)).hexdigest() != checkpoint.payload_digest:
        raise PrognosticSecondaryAMRLifecycleError(
            "Q3-H3/schema-10 payload digest differs"
        )


__all__ = [
    "AcceptedSnapshot", "PrognosticSecondaryAMRLifecycleError",
    "REGISTER_COMPONENTS", "REQUIRED_LEDGER_FIELDS", "RegisterAudit",
    "Schema10Checkpoint", "accepted_snapshot", "assert_exact_rollback",
    "register_audit", "schema10_checkpoint", "validate_schema10_checkpoint",
]
