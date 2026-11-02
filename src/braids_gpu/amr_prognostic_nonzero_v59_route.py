"""Fail-closed composition seam for a live Q3-H3/v59 selected AMR route.

This module authenticates the actual schema-3 selected-mesh topology advanced
by the accepted v58 controller and admits the bounded fully-wet, closed-wall
live slice supplied by :mod:`implicit_midpoint_fwave_amr_v59`.  Unsupported
open-boundary numerics, recursive subcycling and restart still reject before
CUDA construction; there is never a v58-only fallback.

The intended ownership is explicit:

* v58 owns the one-clock controller and unchanged hydraulic/sediment/secondary
  chain on the selected physical-face graph, but its legacy nonzero ``Betac``
  formula is not admitted by this seam;
* v59 owns AMR identity, signed-hI transfer and topology-handoff registers at
  an accepted macro boundary and must replace (not add to) the legacy v58
  ``Betac`` residual field with the corrected depth-weighted operator; and
* the existing selected-mesh owner transfers every other conserved sediment
  row unchanged.

The seam now authenticates the stable finest-level open-boundary catalog and
can prove an exact catalog handoff across a regrid.  The numerical open-face
operator/ledger, wet/dry, recursive subcycling and restart are still outside
this first seam and remain fail-closed.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from braids_core.sediment.prognostic_secondary_amr_execution import (
    canonical_amr_instance_fingerprints,
)


SCOPE_ID = "q3h3-v59-live-synchronous-selected-mesh-composition-seam-v1"
INTEGRATOR = "implicit_fwave_midpoint"
TIME_COUPLING = "synchronous_selected_mesh_one_clock_q3h_v1"
TOPOLOGY_KIND = "general_selected_active_leaf_mesh_v1"
FACE_KIND = "ordered_selected_physical_face_segments_v2"
BOUNDARY_KIND = "stable_finest_open_boundary_segments_v1"
LIVE_INTERFACE_SCHEMA = 1
REQUIRED_LIVE_CAPABILITIES = (
    "general_selected_topology",
    "full_tree_active_leaf_projection",
    "signed_hi_accepted_boundary_handoff",
    "transactional_refine_coarsen",
    "external_v58_macrostep_ownership",
    "corrected_depth_weighted_betac_override",
    "stable_finest_open_boundary_catalog_handoff",
)
EXPLICITLY_UNSUPPORTED = (
    "open_boundary_numeric_operator_and_ledgers",
    "wet_dry",
    "recursive_implicit_subcycling",
    "restart",
)


class LiveAMRCompositionUnavailable(ValueError):
    """The additive live route cannot be entered without a named interface."""

    def __init__(self, report: Mapping[str, object]):
        self.report = dict(report)
        blockers = self.report.get("blockers", ())
        message = "; ".join(str(item) for item in blockers)
        super().__init__(
            "Q3-H3/v59 live selected-mesh composition is unavailable: "
            + (message or "unspecified interface blocker")
        )


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _scalar(data: Mapping[str, object], name: str):
    if name not in data:
        raise ValueError("Q3-H3/v59 selected topology lacks " + name)
    value = np.asarray(data[name])
    if value.shape != ():
        raise ValueError("Q3-H3/v59 selected control is not scalar: " + name)
    return value.item()


def _integer(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value)
    if result.shape != shape or result.dtype.kind not in "iu":
        raise ValueError(
            "Q3-H3/v59 selected integer array differs: " + name
        )
    return np.ascontiguousarray(result, dtype=np.int64)


def _real(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value)
    if (result.shape != shape or result.dtype != np.dtype("float64") or
            not np.isfinite(result).all()):
        raise ValueError(
            "Q3-H3/v59 selected FP64 array differs: " + name
        )
    return np.ascontiguousarray(result)


def _complete_active_cover(parent: np.ndarray, selected: np.ndarray) -> None:
    """Require one non-overlapping active representation of every root."""
    selected_set = {int(value) for value in selected}
    children = tuple(np.flatnonzero(parent == node) for node in range(len(parent)))
    if any(len(row) not in (0, 4) for row in children):
        raise ValueError(
            "Q3-H3/v59 selected hierarchy needs complete four-child families"
        )
    for node in selected_set:
        ancestor = int(parent[node])
        while ancestor >= 0:
            if ancestor in selected_set:
                raise ValueError(
                    "Q3-H3/v59 selected leaves overlap an active ancestor"
                )
            ancestor = int(parent[ancestor])

    def covered(node: int) -> bool:
        if node in selected_set:
            return True
        row = children[node]
        return len(row) == 4 and all(covered(int(child)) for child in row)

    roots = np.flatnonzero(parent < 0)
    if not len(roots) or not all(covered(int(root)) for root in roots):
        raise ValueError(
            "Q3-H3/v59 selected leaves do not cover the complete hierarchy"
        )


@dataclass(frozen=True)
class SelectedMeshContract:
    """Authenticated topology-local projection for the future v59 handoff."""

    node_ids: np.ndarray
    levels: np.ndarray
    internal_face_indices: np.ndarray
    internal_source_ids: np.ndarray
    physical_face_indices: np.ndarray
    physical_source_ids: np.ndarray
    open_face_indices: np.ndarray
    open_source_ids: np.ndarray
    open_logical_boundary_ids: np.ndarray
    open_owner_node_ids: np.ndarray
    owner_levels: np.ndarray
    neighbour_levels: np.ndarray
    topology_document: dict[str, object]
    face_catalog_document: dict[str, object]
    boundary_catalog_document: list[object]
    fingerprints: dict[str, str]
    mixed_levels: bool
    coarse_fine_segment_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "scope_id": SCOPE_ID,
            "integrator": INTEGRATOR,
            "time_coupling": TIME_COUPLING,
            "node_ids": self.node_ids.tolist(),
            "levels": self.levels.tolist(),
            "internal_face_indices": self.internal_face_indices.tolist(),
            "internal_source_ids": self.internal_source_ids.tolist(),
            "physical_face_indices": self.physical_face_indices.tolist(),
            "physical_source_ids": self.physical_source_ids.tolist(),
            "open_face_indices": self.open_face_indices.tolist(),
            "open_source_ids": self.open_source_ids.tolist(),
            "open_logical_boundary_ids": self.open_logical_boundary_ids.tolist(),
            "open_owner_node_ids": self.open_owner_node_ids.tolist(),
            "owner_levels": self.owner_levels.tolist(),
            "neighbour_levels": self.neighbour_levels.tolist(),
            "mixed_levels": self.mixed_levels,
            "coarse_fine_segment_count": self.coarse_fine_segment_count,
            **self.fingerprints,
        }


def selected_mesh_contract(prepared, topology, *, require_mixed: bool = True
                           ) -> SelectedMeshContract:
    """Authenticate one real schema-3 selected topology without CUDA work.

    Every selected physical segment is identity-bound.  Open faces are
    admitted only when their owner is on the globally finest tree level; this
    function authenticates their stable identity but does not claim that the
    numerical O1 operator is composed yet.
    """
    if (getattr(prepared, "schema", None) != 3 or
            not any(topology is item for item in
                    getattr(prepared, "topologies", ()) or ())):
        raise ValueError(
            "Q3-H3/v59 requires an authenticated schema-3 selected topology"
        )
    tree = getattr(prepared, "tree", None)
    data = getattr(topology, "data", None)
    if not isinstance(tree, Mapping) or not isinstance(data, Mapping):
        raise ValueError("Q3-H3/v59 selected topology payload is absent")

    parent_raw = np.asarray(tree.get("parent"))
    count = len(parent_raw) if parent_raw.ndim == 1 else -1
    parent = _integer(parent_raw, (count,), "tree parent")
    level = _integer(tree.get("node_level"), (count,), "tree level")
    tree_area = _real(tree.get("area"), (count,), "tree area")
    tree_xy = _real(tree.get("xy"), (count, 2), "tree centre")
    if (count < 5 or np.any(parent >= np.arange(count)) or
            np.any(parent < -1) or np.any(level < 0) or
            np.any(tree_area <= 0.0)):
        raise ValueError("Q3-H3/v59 selected hierarchy is invalid")
    child = parent >= 0
    if np.any(level[child] != level[parent[child]] + 1):
        raise ValueError("Q3-H3/v59 selected hierarchy levels differ")

    node_ids_raw = np.asarray(getattr(topology, "node_ids", None))
    nc = int(_scalar(data, "nc"))
    node_ids = _integer(node_ids_raw, (nc,), "selected node IDs")
    if (nc <= 0 or np.any((node_ids < 0) | (node_ids >= count)) or
            np.any(np.diff(node_ids) <= 0)):
        raise ValueError(
            "Q3-H3/v59 selected node IDs must be unique and strictly ordered"
        )
    _complete_active_cover(parent, node_ids)
    levels = np.ascontiguousarray(level[node_ids])
    mixed = len(np.unique(levels)) > 1
    if require_mixed and not mixed:
        raise ValueError("Q3-H3/v59 live gate requires a mixed-level topology")

    area = _real(data.get("area"), (nc,), "selected area")
    centre = _real(data.get("xy"), (2, nc), "selected centre")
    if (not np.array_equal(area, tree_area[node_ids]) or
            not np.array_equal(centre.T, tree_xy[node_ids])):
        raise ValueError(
            "Q3-H3/v59 selected cells differ from the hierarchy projection"
        )

    nf = int(_scalar(data, "nf"))
    if nf <= 0:
        raise ValueError("Q3-H3/v59 selected topology has no physical faces")
    owner = _integer(data.get("owner"), (nf,), "face owner")
    neighbour = _integer(data.get("neighbour"), (nf,), "face neighbour")
    source = _integer(
        data.get("face_source_ids"), (nf,), "face source IDs")
    length = _real(data.get("length"), (nf,), "face length")
    normal = _real(data.get("normal"), (2, nf), "face normal")
    midpoint = _real(data.get("mid"), (2, nf), "face midpoint")
    bid = _integer(data.get("bid"), (nf,), "face boundary ID")
    if (np.any((owner < 0) | (owner >= nc)) or
            np.any((neighbour < -1) | (neighbour >= nc)) or
            np.any(neighbour == owner) or np.any(source <= 0) or
            np.any(np.diff(source) <= 0) or np.any(length <= 0.0) or
            not np.allclose(np.linalg.norm(normal, axis=0), 1.0,
                            rtol=0.0, atol=1.0e-12)):
        raise ValueError("Q3-H3/v59 selected physical-face catalog is invalid")

    exterior = neighbour < 0
    declared_open = exterior & (bid >= 0)
    open_flag = np.zeros(nf, dtype=bool)
    if "secondary_open_face" in data:
        flag = np.asarray(data["secondary_open_face"])
        if flag.shape not in ((0,), (nf,)):
            raise ValueError(
                "Q3-H3/v59 selected secondary open-face map differs"
            )
        if flag.shape == (nf,):
            if flag.dtype.kind not in "iub" or np.any((flag != 0) & (flag != 1)):
                raise ValueError(
                    "Q3-H3/v59 selected secondary open-face map is invalid"
                )
            open_flag = flag.astype(bool)
            declared_open |= open_flag
    if np.any(open_flag & (~exterior | (bid < 0))):
        raise ValueError(
            "Q3-H3/v59 selected secondary open-face map lacks a logical "
            "exterior boundary"
        )
    if np.any(bid[~exterior] >= 0):
        raise ValueError(
            "Q3-H3/v59 internal face owns a logical boundary ID"
        )

    internal = np.flatnonzero(~exterior).astype(np.int64)
    physical = np.arange(nf, dtype=np.int64)
    open_faces = np.flatnonzero(declared_open).astype(np.int64)
    if not len(internal):
        raise ValueError("Q3-H3/v59 selected topology has no internal segments")
    owner_level = levels[owner[internal]]
    neighbour_level = levels[neighbour[internal]]
    if np.any(np.abs(owner_level - neighbour_level) > 1):
        raise ValueError("Q3-H3/v59 selected topology is not 2:1 balanced")
    projected = np.sum(
        (centre[:, neighbour[internal]] - centre[:, owner[internal]]) *
        normal[:, internal], axis=0)
    if np.any(~np.isfinite(projected) | (projected <= 0.0)):
        raise ValueError(
            "Q3-H3/v59 internal normal-projected distance is nonpositive"
        )
    exterior_projected = np.sum(
        (midpoint[:, exterior] - centre[:, owner[exterior]]) *
        normal[:, exterior], axis=0)
    if np.any(~np.isfinite(exterior_projected) | (exterior_projected <= 0.0)):
        raise ValueError(
            "Q3-H3/v59 exterior normal-projected distance is nonpositive"
        )
    global_finest_level = int(np.max(level))
    if np.any(levels[owner[open_faces]] != global_finest_level):
        raise ValueError(
            "Q3-H3/v59 open boundary owner is not on the globally finest level"
        )
    internal_source = source[internal]
    if np.any(np.diff(internal_source) <= 0):
        raise ValueError(
            "Q3-H3/v59 internal source IDs must remain strictly ordered"
        )

    topology_document = {
        "schema": 1,
        "kind": TOPOLOGY_KIND,
        "maximum_level": int(np.max(level)),
        "active_cell_id": node_ids.tolist(),
        "active_cell_level": levels.tolist(),
        "active_cell_area_m2": area.tolist(),
        "active_cell_centre_m": centre.T.tolist(),
    }
    neighbour_level_all = np.full(nf, -1, dtype=np.int64)
    neighbour_level_all[internal] = levels[neighbour[internal]]
    segment_level = levels[owner].copy()
    segment_level[internal] = np.maximum(
        levels[owner[internal]], neighbour_level_all[internal])
    projected_all = np.empty(nf, dtype=np.float64)
    projected_all[internal] = projected
    projected_all[exterior] = exterior_projected
    face_kind = np.full(nf, "wall", dtype=object)
    face_kind[internal] = "internal"
    face_kind[open_faces] = "open"
    face_catalog_document = {
        "schema": 1,
        "kind": FACE_KIND,
        "source_id": source.tolist(),
        "owner_local_index": owner.tolist(),
        "neighbour_local_index": neighbour.tolist(),
        "owner_cell_id": node_ids[owner].tolist(),
        "neighbour_cell_id": [
            None if item < 0 else int(node_ids[item]) for item in neighbour
        ],
        "owner_level": levels[owner].tolist(),
        "neighbour_level": [
            None if item < 0 else int(levels[item]) for item in neighbour
        ],
        "level": segment_level.tolist(),
        "owner_to_neighbour_or_exterior_normal": normal.T.tolist(),
        "midpoint_m": midpoint.T.tolist(),
        "length_m": length.tolist(),
        "normal_projected_distance_m": projected_all.tolist(),
        "boundary_kind": face_kind.tolist(),
        "logical_boundary_id": [
            int(bid[index]) if declared_open[index] else None
            for index in range(nf)
        ],
    }
    boundary_document: list[object] = [
        {
            "source_id": int(source[index]),
            "logical_boundary_id": int(bid[index]),
            "owner_cell_id": int(node_ids[owner[index]]),
            "owner_level": int(levels[owner[index]]),
            "normal": normal[:, index].tolist(),
            "midpoint_m": midpoint[:, index].tolist(),
            "length_m": float(length[index]),
            "normal_projected_distance_m": float(projected_all[index]),
        }
        for index in open_faces
    ]
    fingerprints = canonical_amr_instance_fingerprints(
        topology_document=topology_document,
        face_catalog_document=face_catalog_document,
        boundary_catalog_document=boundary_document,
    )
    arrays = (
        node_ids, levels, internal, internal_source, physical, source,
        open_faces, source[open_faces], bid[open_faces],
        node_ids[owner[open_faces]],
        np.ascontiguousarray(owner_level),
        np.ascontiguousarray(neighbour_level),
    )
    for array in arrays:
        array.setflags(write=False)
    return SelectedMeshContract(
        node_ids=arrays[0], levels=arrays[1],
        internal_face_indices=arrays[2], internal_source_ids=arrays[3],
        physical_face_indices=arrays[4], physical_source_ids=arrays[5],
        open_face_indices=arrays[6], open_source_ids=arrays[7],
        open_logical_boundary_ids=arrays[8], open_owner_node_ids=arrays[9],
        owner_levels=arrays[10], neighbour_levels=arrays[11],
        topology_document=topology_document,
        face_catalog_document=face_catalog_document,
        boundary_catalog_document=boundary_document,
        fingerprints=fingerprints,
        mixed_levels=mixed,
        coarse_fine_segment_count=int(np.count_nonzero(
            owner_level != neighbour_level)),
    )


def validate_stable_open_boundary_handoff(
    before: SelectedMeshContract, after: SelectedMeshContract
) -> dict[str, object]:
    """Require bitwise-stable finest physical open faces across a regrid."""
    if not isinstance(before, SelectedMeshContract) or not isinstance(
            after, SelectedMeshContract):
        raise ValueError("Q3-H3/v59 boundary handoff requires selected contracts")
    if (before.boundary_catalog_document != after.boundary_catalog_document or
            before.fingerprints["boundary_catalog_fingerprint"] !=
            after.fingerprints["boundary_catalog_fingerprint"] or
            not np.array_equal(before.open_source_ids, after.open_source_ids) or
            not np.array_equal(before.open_logical_boundary_ids,
                               after.open_logical_boundary_ids) or
            not np.array_equal(before.open_owner_node_ids,
                               after.open_owner_node_ids)):
        raise ValueError(
            "Q3-H3/v59 regrid changes the stable finest open-boundary catalog"
        )
    return {
        "schema": 1,
        "kind": BOUNDARY_KIND,
        "boundary_catalog_fingerprint": before.fingerprints[
            "boundary_catalog_fingerprint"],
        "open_segment_count": int(before.open_face_indices.size),
        "handoff": "ordered_exact_no_averaging_v1",
    }


def required_interface_patch() -> dict[str, object]:
    """Record the implemented bounded interfaces and remaining O1 work."""
    patches = [
        {
            "owner": "selected_initializer",
            "interface": "current_q3c_selected_export_v1",
            "state": "implemented_closed_synchronous_slice",
            "requirement": (
                "Create schema-3 selected operators from current v54+ Q3-C/"
                "v57/v58 decks while preserving native capacity, forcing, "
                "secondary identities and stable face_source_ids. The frozen "
                "v49 general initializer must remain unchanged."
            ),
        },
        {
            "owner": "v59_amr_component",
            "interface": "general_selected_handoff_v1",
            "state": "implemented_closed_synchronous_slice",
            "requirement": (
                "Accept a complete hierarchy plus arbitrary authenticated "
                "active leaves and ordered physical segments; map inactive "
                "parent/child cache rows separately from topology-local v58 "
                "solver rows. Remove the canonical 3-cell/5-row case split."
            ),
        },
        {
            "owner": "v59_amr_component",
            "interface": "external_v58_macrostep_owner_v1",
            "state": "implemented_closed_synchronous_slice",
            "requirement": (
                "Expose signed-hI transfer/register handoff without executing "
                "v59 advection, diffusion or Coriolis a second time. The v58 "
                "controller owns the clock and unchanged chain, not an "
                "independent duplicate AMR operator pass."
            ),
        },
        {
            "owner": "v59_amr_component",
            "interface": "corrected_depth_weighted_betac_override_v1",
            "state": "implemented_closed_synchronous_slice",
            "requirement": (
                "Replace the unqualified legacy v58 Betac residual field with "
                "the corrected Delft h*div(T) segment operator at every "
                "implicit residual. The old and corrected fields may never "
                "both be active; identity and rollback must record the "
                "replacement explicitly."
            ),
        },
        {
            "owner": "selected_mesh_transaction",
            "interface": "atomic_topology_handoff_v1",
            "state": "implemented_closed_synchronous_slice",
            "requirement": (
                "Construct the candidate selected solver, transfer unchanged "
                "sediment rows with the existing owner, transfer signed hI "
                "with v59, then publish topology/state/registers atomically; "
                "cancellation or rejection restores exact accepted bytes."
            ),
        },
        {
            "owner": "v59_open_boundary_operator",
            "interface": "finest_open_boundary_operator_v1",
            "state": "blocked_numeric_operator_and_ledgers",
            "requirement": (
                "Use the authenticated adjacent-interior equilibrium owner "
                "for both flow directions, evaluate signed advection, "
                "Dirichlet diffusion and corrected Betac traction once per "
                "physical segment, and publish separate inward-positive "
                "root ledgers without any cached parent-face contribution."
            ),
        },
        {
            "owner": "v59_amr_component",
            "interface": "live_selected_capability_manifest_v1",
            "state": "implemented_closed_synchronous_slice",
            "requirement": (
                "Publish a schema-1 manifest containing every required live "
                "capability and bind it to finalized v58/v59 source hashes."
            ),
        },
    ]
    body = {
        "schema": 1,
        "scope_id": SCOPE_ID,
        "status": "bounded_live_closed_selected_admission",
        "required_live_capabilities": list(REQUIRED_LIVE_CAPABILITIES),
        "explicitly_unsupported": list(EXPLICITLY_UNSUPPORTED),
        "operator_ownership": {
            "accepted_macrostep_controller": "v58_implicit_fwave_midpoint",
            "legacy_v58_betac": "replaced_in_bounded_live_route",
            "corrected_depth_weighted_betac": "v59_exactly_once_residual_override",
            "amr_identity_signed_hi_handoff_register": "v59",
            "other_sediment_rows": "existing_selected_mesh_transfer_owner",
            "duplicate_v59_within_step_physics": False,
        },
        "patches": patches,
        "Delft3D_source_modified": False,
    }
    return {**body, "fingerprint": _digest(body)}


def _live_manifest() -> tuple[dict[str, object] | None, list[str]]:
    """Inspect the additive live component without constructing CUDA state."""
    from . import implicit_midpoint_fwave_amr_v59 as component

    value = getattr(component, "LIVE_SELECTED_INTERFACE", None)
    blockers: list[str] = []
    if not isinstance(value, dict):
        blockers.append(
            "v59 lacks LIVE_SELECTED_INTERFACE for a general selected mesh"
        )
        return None, blockers
    expected = {
        "schema", "scope_id", "capabilities", "automatic_cpu_fallback",
        "external_v58_macrostep_owner", "fingerprint",
    }
    body = {key: item for key, item in value.items() if key != "fingerprint"}
    if (set(value) != expected or value.get("schema") != LIVE_INTERFACE_SCHEMA or
            value.get("scope_id") != SCOPE_ID or
            value.get("capabilities") != list(REQUIRED_LIVE_CAPABILITIES) or
            value.get("automatic_cpu_fallback") is not False or
            value.get("external_v58_macrostep_owner") is not True or
            value.get("fingerprint") != _digest(body)):
        blockers.append("v59 live selected capability manifest is invalid")
        return None, blockers
    return dict(value), blockers


def composition_readiness(prepared, topology) -> dict[str, object]:
    """Return a non-mutating readiness report for the real selected topology."""
    blockers: list[str] = []
    contract = None
    try:
        contract = selected_mesh_contract(
            prepared, topology, require_mixed=False)
    except ValueError as error:
        blockers.append(str(error))
    if contract is not None and contract.open_face_indices.size:
        blockers.append(
            "stable open-boundary topology is authenticated but the O1 "
            "numeric operator and root ledgers are not implemented"
        )

    data = getattr(topology, "data", {})
    if isinstance(data, Mapping):
        if bool(np.asarray(data.get("amr_enabled", 0)).item()):
            blockers.append(
                "topology-local v58 view must retain amr_enabled=0"
            )
        state = np.asarray(data.get("state", ()))
        nc = int(np.asarray(data.get("nc", -1)).item())
        dry = float(np.asarray(data.get(
            "dry", data.get("dry_tolerance", np.nan))).item())
        if (state.ndim != 2 or state.shape[1] != nc or not np.isfinite(dry) or
                dry < 0.0 or not np.isfinite(state).all() or
                np.any(state[0] <= dry)):
            blockers.append("selected v58 accepted state is not fully wet")
        for key in ("recursive_subcycling", "restart_enabled"):
            if bool(np.asarray(data.get(key, 0)).item()):
                blockers.append("unsupported selected control is enabled: " + key)
        try:
            from .flow2dh_prognostic_nonzero_v58 import validate_scope
            validate_scope(dict(data))
        except (KeyError, TypeError, ValueError) as error:
            blockers.append(
                "selected operator is not an authenticated v58 parent: " +
                str(error)
            )
    else:
        blockers.append("selected topology has no operator mapping")

    manifest, interface_blockers = _live_manifest()
    blockers.extend(interface_blockers)
    patch = required_interface_patch()
    report = {
        "schema": 1,
        "scope_id": SCOPE_ID,
        "status": "ready" if not blockers else "blocked",
        "live_route_admitted": not blockers,
        "automatic_cpu_fallback": False,
        "contract": None if contract is None else contract.as_dict(),
        "live_interface": manifest,
        "blockers": blockers,
        "required_interface_patch_fingerprint": patch["fingerprint"],
        "Delft3D_source_modified": False,
    }
    body = dict(report)
    report["fingerprint"] = _digest(body)
    return report


@contextmanager
def constructors(folder: str | Path, prepared, topology):
    """Yield the additive live constructor after all host gates pass."""
    report = composition_readiness(prepared, topology)
    if not report["live_route_admitted"]:
        raise LiveAMRCompositionUnavailable(report)
    from .implicit_midpoint_fwave_amr_v59 import constructors as live
    with live(folder, prepared, topology) as Solver:
        yield Solver


__all__ = [
    "BOUNDARY_KIND", "EXPLICITLY_UNSUPPORTED", "FACE_KIND", "INTEGRATOR",
    "LIVE_INTERFACE_SCHEMA", "LiveAMRCompositionUnavailable",
    "REQUIRED_LIVE_CAPABILITIES", "SCOPE_ID", "SelectedMeshContract",
    "TIME_COUPLING", "TOPOLOGY_KIND", "composition_readiness",
    "constructors", "required_interface_patch", "selected_mesh_contract",
    "validate_stable_open_boundary_handoff",
]
