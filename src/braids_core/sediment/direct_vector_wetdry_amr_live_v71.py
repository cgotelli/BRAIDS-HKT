"""Q3-H7/v71 wet/dry and AMR ownership contract for direct bed load.

This module is the backend-neutral control plane for one deliberately small
composition seam.  The v67 scalar/direct-vector family continues to own the
constitutive formula, accepted surface fraction, inventory projection and
Exner update.  v71 owns only:

* the mapping of an *accepted* bed inventory between two selected AMR covers;
* isolation of sediment transport from cells that are hydraulically dry; and
* invalidation of topology-local face registers when ownership or wetness
  changes.

Bed mass is parent-owned: refinement distributes an old cell-average density
over its child overlap areas and coarsening takes the corresponding
area-weighted average.  A face joining two cells is a transport face only when
both cells are wet.  Dry cells retain their bed inventory; only ephemeral
producer/face data are cleared.  This is consistent with the locked Delft3D4
``erosed``/``dwnvel``/``upwbed`` chain (``kfsed=0`` gives zero velocity and a
bed-load velocity point requires two active sediment cells).  AMR itself is a
BRAIDS implementation concern and is not claimed to be a Delft3D operator.

The contract is intentionally bounded to noncohesive ``TRA_BEDLOAD`` state.
Dry-bank redistribution, suspended load, stratigraphy and recursive time
subcycling have separate owners and are not silently enabled here.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np


SCHEMA = 1
ENVELOPE_VERSION = 71
SCOPE_ID = "braids-q3-h7-direct-vector-wetdry-amr-live-v71-v1"
INTEGRATOR = "implicit_fwave_midpoint"
BACKENDS = frozenset(("cpu", "implicit_fwave_midpoint_gpu"))
MAX_ACCEPTED_STEPS = 4

FACE_INTERNAL = 0
FACE_TRACE_INLET = 1
FACE_TRACE_OUTLET = 2
FACE_OWNER_OUTLET = 3
FACE_CLOSED = 4
FACE_ROLES = frozenset(range(5))


class DirectVectorWetDryAMRV71Error(ValueError):
    """A v71 topology, transfer, execution or checkpoint is invalid."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, OverflowError) as error:
        raise DirectVectorWetDryAMRV71Error(
            "Q3-H7/v71 document is not finite canonical JSON"
        ) from error


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _array(value: object, name: str, dtype, ndim: int) -> np.ndarray:
    result = np.asarray(value)
    if result.dtype != np.dtype(dtype) or result.ndim != ndim:
        raise DirectVectorWetDryAMRV71Error(
            f"Q3-H7/v71 {name} must be contiguous {np.dtype(dtype)} {ndim}-D data"
        )
    result = np.ascontiguousarray(result)
    if result.dtype.kind == "f" and not np.isfinite(result).all():
        raise DirectVectorWetDryAMRV71Error(
            f"Q3-H7/v71 {name} contains nonfinite values"
        )
    return result


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


REFERENCE_SOURCES = MappingProxyType({
    "erosed": (
        "resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/"
        "flow2d3d_kernel/src/compute_sediment/erosed.f90",
        "4571fd7912233db03f597161af1a18702ee974483676004cda8186ccf24308c0",
    ),
    "dwnvel": (
        "resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/"
        "flow2d3d_kernel/src/compute_sediment/dwnvel.f90",
        "cb26d6c037804ce2be3bcc975dab81cce21b78571f75e33fa302bcc14bf17ccd",
    ),
    "upwbed": (
        "resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/"
        "flow2d3d_kernel/src/compute_sediment/upwbed.f90",
        "06be422d88821f3f983bc136d511fdaa80ab04c8529640474ab39b478d918158",
    ),
    "bott3d": (
        "resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/"
        "flow2d3d_kernel/src/compute_sediment/bott3d.f90",
        "5abfee8230f66ba42b3eb87ff89948f65a0bfccc915a015c25f8c1412f4e47b1",
    ),
})


def authenticate_locked_reference_sources(
    root: str | Path | None = None,
) -> dict[str, str]:
    """Authenticate, but never write, the four locked reference owners."""
    repository = Path(root or Path(__file__).resolve().parents[2]).resolve()
    actual: dict[str, str] = {}
    for name, (relative, expected) in REFERENCE_SOURCES.items():
        path = repository / relative
        if not path.is_file() or path.is_symlink():
            raise DirectVectorWetDryAMRV71Error(
                f"Q3-H7/v71 locked Delft3D source is absent or linked: {name}"
            )
        actual[name] = _sha(path)
        if actual[name] != expected:
            raise DirectVectorWetDryAMRV71Error(
                f"Q3-H7/v71 locked Delft3D source identity differs: {name}"
            )
    return actual


@dataclass(frozen=True)
class WetDryTopologyV71:
    topology_id: str
    node_ids: np.ndarray
    cell_area_m2: np.ndarray
    wet: np.ndarray
    owner: np.ndarray
    neighbour: np.ndarray
    role: np.ndarray
    document: Mapping[str, object]

    @property
    def ncell(self) -> int:
        return int(self.node_ids.size)

    @property
    def nface(self) -> int:
        return int(self.owner.size)


def wetdry_topology_v71(
    *, node_ids: object, cell_area_m2: object, wet: object,
    owner: object, neighbour: object, role: object,
) -> WetDryTopologyV71:
    """Validate a selected cover and its dry-isolated transport graph.

    ``owner``/``neighbour`` use zero-based cell indices and ``-1`` for an
    exterior face.  Dry cells may remain in the bed inventory, but every face
    they own must be closed and no internal face may touch them.
    """
    nodes = _array(node_ids, "node IDs", np.int64, 1)
    area = _array(cell_area_m2, "cell area", np.float64, 1)
    wet_array = _array(wet, "wet mask", np.bool_, 1)
    own = _array(owner, "face owner", np.int32, 1)
    other = _array(neighbour, "face neighbour", np.int32, 1)
    face_role = _array(role, "face role", np.int32, 1)
    ncell = nodes.size
    nface = own.size
    if (
        ncell < 1 or nface < 1 or area.shape != (ncell,)
        or wet_array.shape != (ncell,) or other.shape != (nface,)
        or face_role.shape != (nface,) or np.any(nodes < 0)
        or np.any(np.diff(nodes) <= 0) or np.any(area <= 0.0)
        or np.any((own < 0) | (own >= ncell))
        or np.any(~np.isin(face_role, tuple(FACE_ROLES)))
    ):
        raise DirectVectorWetDryAMRV71Error(
            "Q3-H7/v71 selected topology shape/order/domain differs"
        )
    internal = face_role == FACE_INTERNAL
    if (
        np.any(internal & ((other < 0) | (other >= ncell) | (other == own)))
        or np.any((~internal) & (other != -1))
    ):
        raise DirectVectorWetDryAMRV71Error(
            "Q3-H7/v71 face role/neighbour semantics differ"
        )
    if np.any(internal & (~wet_array[own] | ~wet_array[other])):
        raise DirectVectorWetDryAMRV71Error(
            "Q3-H7/v71 a transport face touches a dry cell"
        )
    dry_owner = ~wet_array[own]
    if np.any(dry_owner & (face_role != FACE_CLOSED)):
        raise DirectVectorWetDryAMRV71Error(
            "Q3-H7/v71 a dry owner has a nonclosed face"
        )
    # Every inventory cell must occur in the selected transport graph, even
    # when its only edge is a closed dry placeholder.
    seen = np.bincount(own, minlength=ncell)
    if np.any(internal):
        seen += np.bincount(other[internal], minlength=ncell)
    if np.any(seen == 0):
        raise DirectVectorWetDryAMRV71Error(
            "Q3-H7/v71 selected topology omits an inventory cell"
        )
    body = {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "kind": "wetdry_selected_topology",
        "node_ids": nodes.tolist(),
        "cell_area_m2": area.tolist(),
        "wet": wet_array.tolist(),
        "owner": own.tolist(),
        "neighbour": other.tolist(),
        "role": face_role.tolist(),
        "dry_inventory_policy": "retain_bed_mass_clear_ephemeral_flux",
        "internal_face_policy": "both_cells_wet",
    }
    topology_id = _fingerprint(body)
    document = MappingProxyType({**body, "fingerprint": topology_id})
    for item in (nodes, area, wet_array, own, other, face_role):
        item.setflags(write=False)
    return WetDryTopologyV71(
        topology_id, nodes, area, wet_array, own, other, face_role, document,
    )


@dataclass(frozen=True)
class ConservativeTransferPlanV71:
    old_topology_id: str
    new_topology_id: str
    old_index: np.ndarray
    new_index: np.ndarray
    overlap_area_m2: np.ndarray
    new_offsets: np.ndarray
    plan_id: str
    classification: str


def conservative_transfer_plan_v71(
    old: WetDryTopologyV71,
    new: WetDryTopologyV71,
    *, old_index: object, new_index: object, overlap_area_m2: object,
) -> ConservativeTransferPlanV71:
    """Seal a conservative old-to-new overlap map.

    Links are canonicalized by ``(new_index, old_index)``.  Both old and new
    area partitions must close, which prevents either duplicated child mass
    or a missing parent contribution.
    """
    if not isinstance(old, WetDryTopologyV71) or not isinstance(
        new, WetDryTopologyV71
    ):
        raise DirectVectorWetDryAMRV71Error(
            "Q3-H7/v71 transfer endpoints are not selected topologies"
        )
    oi = _array(old_index, "transfer old index", np.int32, 1)
    ni = _array(new_index, "transfer new index", np.int32, 1)
    overlap = _array(overlap_area_m2, "transfer overlap", np.float64, 1)
    if (
        oi.size < 1 or ni.shape != oi.shape or overlap.shape != oi.shape
        or np.any((oi < 0) | (oi >= old.ncell))
        or np.any((ni < 0) | (ni >= new.ncell)) or np.any(overlap <= 0.0)
    ):
        raise DirectVectorWetDryAMRV71Error(
            "Q3-H7/v71 overlap link domain differs"
        )
    order = np.lexsort((oi, ni))
    oi = np.ascontiguousarray(oi[order])
    ni = np.ascontiguousarray(ni[order])
    overlap = np.ascontiguousarray(overlap[order])
    pairs = np.column_stack((ni, oi))
    if pairs.shape[0] > 1 and np.any(np.all(pairs[1:] == pairs[:-1], axis=1)):
        raise DirectVectorWetDryAMRV71Error(
            "Q3-H7/v71 overlap map contains a duplicate ownership link"
        )
    old_cover = np.bincount(oi, weights=overlap, minlength=old.ncell)
    new_cover = np.bincount(ni, weights=overlap, minlength=new.ncell)
    tolerance = 8192.0*np.finfo(np.float64).eps
    if (
        not np.allclose(old_cover, old.cell_area_m2, rtol=tolerance, atol=0.0)
        or not np.allclose(new_cover, new.cell_area_m2,
                          rtol=tolerance, atol=0.0)
    ):
        raise DirectVectorWetDryAMRV71Error(
            "Q3-H7/v71 overlap map does not cover both selected meshes"
        )
    offsets = np.zeros(new.ncell+1, dtype=np.int32)
    np.add.at(offsets, ni+1, 1)
    offsets = np.ascontiguousarray(np.cumsum(offsets, dtype=np.int32))
    old_degree = np.bincount(oi, minlength=old.ncell)
    new_degree = np.bincount(ni, minlength=new.ncell)
    if np.any(old_degree > 1) and np.any(new_degree > 1):
        classification = "mixed_overlap"
    elif np.any(old_degree > 1):
        classification = "refine"
    elif np.any(new_degree > 1):
        classification = "coarsen"
    else:
        classification = "identity_or_relabel"
    body = {
        "schema": SCHEMA, "scope_id": SCOPE_ID,
        "old_topology_id": old.topology_id,
        "new_topology_id": new.topology_id,
        "old_index": oi.tolist(), "new_index": ni.tolist(),
        "overlap_area_m2": overlap.tolist(),
        "new_offsets": offsets.tolist(),
        "classification": classification,
        "transfer_owner": "accepted_parent_bed_mass",
    }
    plan_id = _fingerprint(body)
    for item in (oi, ni, overlap, offsets):
        item.setflags(write=False)
    return ConservativeTransferPlanV71(
        old.topology_id, new.topology_id, oi, ni, overlap, offsets,
        plan_id, classification,
    )


def transfer_bed_mass_reference_v71(
    old_mass_kg_m2: object,
    old: WetDryTopologyV71,
    new: WetDryTopologyV71,
    plan: ConservativeTransferPlanV71,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic CPU oracle for the native/OpenMP and CUDA transfers."""
    mass = _array(old_mass_kg_m2, "old bed mass", np.float64, 2)
    if (
        mass.shape[1] != old.ncell or np.any(mass < 0.0)
        or plan.old_topology_id != old.topology_id
        or plan.new_topology_id != new.topology_id
    ):
        raise DirectVectorWetDryAMRV71Error(
            "Q3-H7/v71 bed inventory/transfer identity differs"
        )
    result = np.zeros((mass.shape[0], new.ncell), dtype=np.float64)
    for target in range(new.ncell):
        start, finish = plan.new_offsets[target:target+2]
        for link in range(int(start), int(finish)):
            result[:, target] += (
                mass[:, plan.old_index[link]]*plan.overlap_area_m2[link]
            )
        result[:, target] /= new.cell_area_m2[target]
    residual = (
        np.sum(result*new.cell_area_m2[None, :], axis=1)
        - np.sum(mass*old.cell_area_m2[None, :], axis=1)
    )
    scale = np.maximum(
        1.0,
        np.maximum(
            np.sum(np.abs(result)*new.cell_area_m2[None, :], axis=1),
            np.sum(np.abs(mass)*old.cell_area_m2[None, :], axis=1),
        ),
    )
    if np.any(np.abs(residual) > 16384.0*np.finfo(np.float64).eps*scale):
        raise DirectVectorWetDryAMRV71Error(
            "Q3-H7/v71 conservative transfer residual is nonzero"
        )
    return np.ascontiguousarray(result), np.ascontiguousarray(residual)


def mask_face_flux_reference_v71(
    flux: object, topology: WetDryTopologyV71,
) -> np.ndarray:
    """Clear closed/dry faces without changing any live internal face."""
    value = _array(flux, "face flux", np.float64, 2)
    if value.shape[1] != topology.nface:
        raise DirectVectorWetDryAMRV71Error(
            "Q3-H7/v71 face flux shape differs"
        )
    result = value.copy()
    result[:, topology.role == FACE_CLOSED] = 0.0
    # Internal dry contact has already been rejected by the topology receipt.
    return np.ascontiguousarray(result)


def execution_document_v71(
    *, backend: str, physics_fingerprint: str,
    topology_ids: Sequence[str], maximum_accepted_steps: int = 4,
) -> dict[str, object]:
    """Bind user-selected backend and the finite topology set for this gate."""
    if backend not in BACKENDS:
        raise DirectVectorWetDryAMRV71Error(
            "Q3-H7/v71 backend must be explicitly CPU or implicit GPU"
        )
    if (
        not isinstance(physics_fingerprint, str)
        or len(physics_fingerprint) != 64
        or any(character not in "0123456789abcdef"
               for character in physics_fingerprint)
    ):
        raise DirectVectorWetDryAMRV71Error(
            "Q3-H7/v71 physics fingerprint differs"
        )
    ids = tuple(topology_ids)
    if (
        not ids or len(set(ids)) != len(ids)
        or any(not isinstance(value, str) or len(value) != 64
               or any(character not in "0123456789abcdef" for character in value)
               for value in ids)
        or isinstance(maximum_accepted_steps, bool)
        or not isinstance(maximum_accepted_steps, int)
        or not 1 <= maximum_accepted_steps <= MAX_ACCEPTED_STEPS
    ):
        raise DirectVectorWetDryAMRV71Error(
            "Q3-H7/v71 topology inventory or bounded step count differs"
        )
    authenticate_locked_reference_sources()
    body = {
        "schema": SCHEMA, "envelope_version": ENVELOPE_VERSION,
        "scope_id": SCOPE_ID, "backend": backend,
        "integrator": INTEGRATOR if backend.endswith("gpu") else "native_openmp",
        "physics_fingerprint": physics_fingerprint,
        "topology_ids": list(ids),
        "maximum_accepted_steps": maximum_accepted_steps,
        "automatic_backend_fallback": False,
        "constitutive_owner": "q3_h6_v67_direct_vector_family",
        "transfer_owner": "accepted_parent_bed_mass_overlap_v71",
        "dry_policy": (
            "retain_inventory_isolate_faces_clear_flux_and_availability"
        ),
        "recursive_subcycling": False,
        "claims": [
            "one_refine_dry_rewet_coarsen_cycle",
            "noncohesive_bedload_only",
            "synchronous_selected_mesh",
        ],
        "nonclaims": [
            "delft3d_amr_equivalence", "dry_bank_redistribution",
            "suspended_load", "stratigraphy", "recursive_subcycling",
            "production_qgis_admission",
        ],
        "reference_source_sha256": {
            name: expected for name, (_path, expected) in REFERENCE_SOURCES.items()
        },
    }
    return {**body, "fingerprint": _fingerprint(body)}


__all__ = [
    "BACKENDS", "ConservativeTransferPlanV71",
    "DirectVectorWetDryAMRV71Error", "ENVELOPE_VERSION", "FACE_CLOSED",
    "FACE_INTERNAL", "FACE_OWNER_OUTLET", "FACE_TRACE_INLET",
    "FACE_TRACE_OUTLET", "INTEGRATOR", "MAX_ACCEPTED_STEPS",
    "REFERENCE_SOURCES", "SCHEMA", "SCOPE_ID", "WetDryTopologyV71",
    "authenticate_locked_reference_sources", "conservative_transfer_plan_v71",
    "execution_document_v71", "mask_face_flux_reference_v71",
    "transfer_bed_mass_reference_v71", "wetdry_topology_v71",
]
