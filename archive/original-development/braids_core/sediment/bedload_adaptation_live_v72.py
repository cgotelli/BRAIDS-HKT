"""Law-neutral BRAIDS bed-load adaptation-length execution contract.

This module gives the existing BRAIDS steady streamline relaxation closure a
versioned identity that can be composed after any authenticated scalar or
direct-vector bed-capacity producer.  It is deliberately *not* a Delft3D
method: the locked classic-FLOW reference does not own this operator and the
contract therefore uses analytic and conservation oracles only.

The closure is derived at every stage,

    L_g dq_g/ds = q_eq,g - q_g,       L_g = C_L D50_g,

so it owns no accepted sediment state.  The downstream capacity lifecycle
continues to own active fractions, finite inventory, face fluxes, Exner,
boundary ledgers, rollback, and restart.  Regridding merely changes the
selected-leaf graph on which this derived solve is repeated.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Mapping, Sequence

import numpy as np


SCHEMA = 1
ENVELOPE_VERSION = 72
SCOPE_ID = "braids-q3-bedload-adaptation-live-v72-v1"
METHOD_ID = "braids.bedload_adaptation.steady_streamline"
MAX_ACCEPTED_STEPS = 4

BACKEND_CPU = "cpu_openmp"
BACKEND_GPU = "implicit_fwave_midpoint_gpu"
CAPACITY_SCALAR = "scalar_flow_v65"
CAPACITY_DIRECT = "direct_vector_family_v67"
CAPACITY_FAMILIES = frozenset((CAPACITY_SCALAR, CAPACITY_DIRECT))

BOUNDARY_FEED = 0
BOUNDARY_EQUILIBRIUM = 1
INLET_HYDRAULIC_MODES = frozenset((1, 6, 7, 8, 9))
_SHA = re.compile(r"[0-9a-f]{64}\Z")


class BedloadAdaptationV72Error(ValueError):
    """The v72 extension identity, graph, or derived solve is invalid."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise BedloadAdaptationV72Error(
            "Q3/v72 data are not canonical JSON") from error


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise BedloadAdaptationV72Error(
            f"Q3/v72 {name} must be a lowercase SHA-256 identity")
    return value


def _finite(value: object, name: str, *, positive: bool = False,
            nonnegative: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)):
        raise BedloadAdaptationV72Error(f"Q3/v72 {name} must be numeric")
    result = float(value)
    if (not math.isfinite(result) or (positive and result <= 0.0) or
            (nonnegative and result < 0.0)):
        raise BedloadAdaptationV72Error(
            f"Q3/v72 {name} is outside its finite domain")
    return 0.0 if result == 0.0 else result


def _strings(values: Sequence[object], name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise BedloadAdaptationV72Error(f"Q3/v72 {name} must be a sequence")
    result = tuple(values)
    if (not result or any(not isinstance(item, str) or not item.strip()
                          for item in result) or len(set(result)) != len(result)):
        raise BedloadAdaptationV72Error(
            f"Q3/v72 {name} must contain unique nonempty strings")
    return result


def bedload_adaptation_semantic_document_v72() -> dict[str, object]:
    """Return the immutable physical/ownership statement for this extension."""
    return {
        "schema": SCHEMA,
        "envelope_version": ENVELOPE_VERSION,
        "scope_id": SCOPE_ID,
        "method_id": METHOD_ID,
        "reference_model": "BRAIDS_extension_not_Delft3D_coverage",
        "equation": "L_g*dq_g/ds=q_equilibrium_g-q_g;L_g=C_L*D50_g",
        "source_order": [
            "authenticated_capacity_and_permitted_direction_steering",
            "derived_stage_local_streamline_adaptation",
            "face_reconstruction_and_optional_global_slope",
            "single_active_fraction_and_inventory_projection",
            "single_conservative_face_flux_and_Exner_owner",
        ],
        "accepted_state_ownership": {
            "adaptation": "none_derived_every_stage",
            "capacity": "parent_authenticated_producer",
            "surface_identity_and_availability": "parent_state_owner",
            "face_flux_and_Exner": "exactly_one_parent_owner",
            "checkpoint": "configuration_identity_only_no_hidden_history",
        },
        "boundaries": {
            "equilibrium_inlet": "equilibrium_capacity_source",
            "prescribed_feed": "mass_per_width_source_in_adaptation_graph",
            "free_export": "owner_adapted_capacity",
            "closed": "zero_downstream_face_flux",
            "live_composition_limit": (
                "prescribed external feed requires the Q6/v70 single-owner "
                "face merger and otherwise fails closed"
            ),
        },
        "wet_dry": (
            "dry_or_zero_speed_cells_return_local_equilibrium_exactly; "
            "the parent wet/dry owner decides face activity"
        ),
        "amr": (
            "recompute_on_selected_leaf_graph; no parent_or_child "
            "adaptation_state_is_transferred"
        ),
        "parallelism": {
            "cpu": "OpenMP independent class_cell Jacobi rows",
            "gpu": "actual_CUDA independent class_cell Jacobi rows",
            "nested_teams": False,
        },
        "bounded_accepted_steps": MAX_ACCEPTED_STEPS,
        "explicit_nonclaims": [
            "Delft3D_equivalence_or_feature_coverage",
            "suspended_load_adaptation",
            "persistent_transport_memory",
            "Q5_authoritative_layers_or_stratigraphy",
            "Q6_prescribed_boundary_face_merger",
            "QGIS_or_Windows_package_admission",
        ],
    }


SEMANTIC_ID = _fingerprint(bedload_adaptation_semantic_document_v72())


def execution_document_v72(
    *, backend: str, capacity_family: str,
    parent_execution_fingerprint: str, topology_fingerprint: str,
    fraction_ids: Sequence[object], d50_m: Sequence[object],
    length_diameters: object,
    wetdry_amr_execution_fingerprint: str | None = None,
) -> dict[str, object]:
    """Resolve a strict backend-specific v72 execution identity."""
    if backend not in (BACKEND_CPU, BACKEND_GPU):
        raise BedloadAdaptationV72Error(
            "Q3/v72 backend must be cpu_openmp or implicit_fwave_midpoint_gpu")
    if capacity_family not in CAPACITY_FAMILIES:
        raise BedloadAdaptationV72Error(
            "Q3/v72 capacity family must be scalar_v65 or direct_vector_v67")
    fractions = _strings(fraction_ids, "fraction IDs")
    if len(d50_m) != len(fractions):
        raise BedloadAdaptationV72Error(
            "Q3/v72 D50 must contain one value per fraction")
    diameters = [_finite(value, "D50", positive=True) for value in d50_m]
    parent = _sha(parent_execution_fingerprint, "parent execution")
    topology = _sha(topology_fingerprint, "topology dependency")
    wetdry = (None if wetdry_amr_execution_fingerprint is None else
              _sha(wetdry_amr_execution_fingerprint,
                   "wet/dry AMR execution"))
    length = _finite(length_diameters, "adaptation length", positive=True)
    body: dict[str, object] = {
        "schema": SCHEMA,
        "envelope_version": ENVELOPE_VERSION,
        "scope_id": SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "method_id": METHOD_ID,
        "backend": backend,
        "integrator": ("implicit_fwave_midpoint" if backend == BACKEND_GPU
                       else "parent_native_cpu_integrator"),
        "automatic_backend_fallback": False,
        "capacity_family": capacity_family,
        "parent_execution_fingerprint": parent,
        "topology_fingerprint": topology,
        "wetdry_amr_execution_fingerprint": wetdry,
        "fraction_ids": list(fractions),
        "d50_m": diameters,
        "length_diameters": length,
        "adaptation_length_m": [length * value for value in diameters],
        "state_semantics": "derived_stage_local_no_accepted_state",
        "accepted_step_limit": MAX_ACCEPTED_STEPS,
        "delft3d_coverage": False,
    }
    return {**body, "fingerprint": _fingerprint(body)}


def load_execution_v72(value: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise BedloadAdaptationV72Error("Q3/v72 execution must be an object")
    try:
        rebuilt = execution_document_v72(
            backend=value["backend"],
            capacity_family=value["capacity_family"],
            parent_execution_fingerprint=value["parent_execution_fingerprint"],
            topology_fingerprint=value["topology_fingerprint"],
            fraction_ids=value["fraction_ids"], d50_m=value["d50_m"],
            length_diameters=value["length_diameters"],
            wetdry_amr_execution_fingerprint=value.get(
                "wetdry_amr_execution_fingerprint"),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise BedloadAdaptationV72Error(
            "Q3/v72 execution is incomplete") from error
    if dict(value) != rebuilt or _canonical(value) != _canonical(rebuilt):
        raise BedloadAdaptationV72Error(
            "Q3/v72 execution is stale or forged")
    return rebuilt


@dataclass(frozen=True)
class BedloadAdaptationTopologyV72:
    execution_fingerprint: str
    ncell: int
    nface: int
    nboundary: int
    xy: np.ndarray
    mid: np.ndarray
    normal: np.ndarray
    face_length_m: np.ndarray
    cell_area_m2: np.ndarray
    owner: np.ndarray
    neighbour: np.ndarray
    boundary_index: np.ndarray
    cell_face_offset: np.ndarray
    cell_faces: np.ndarray
    signs: np.ndarray
    hydraulic_mode: np.ndarray
    bed_boundary_mode: np.ndarray
    topology_sha256: str


@dataclass(frozen=True)
class BedloadAdaptationResultV72:
    execution_fingerprint: str
    topology_sha256: str
    parent_capacity_sha256: str
    adapted_capacity: np.ndarray
    iterations: int
    residual: float
    tolerance: float
    dry_identity_count: int
    result_sha256: str


def _array(value: object, shape: tuple[int, ...], name: str, *,
           integer: bool = False) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != shape or (integer and raw.dtype.kind not in "iu"):
        raise BedloadAdaptationV72Error(
            f"Q3/v72 {name} must have shape {shape}")
    result = np.ascontiguousarray(
        raw, dtype=np.int32 if integer else np.float64)
    if not integer and not np.all(np.isfinite(result)):
        raise BedloadAdaptationV72Error(f"Q3/v72 {name} must be finite")
    return result


def _array_digest(*values: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in values:
        data = np.ascontiguousarray(value)
        digest.update(str(data.dtype).encode("ascii"))
        digest.update(_canonical(list(data.shape)))
        digest.update(data.tobytes(order="C"))
    return digest.hexdigest()


def resolve_topology_v72(
    execution: Mapping[str, object], *, xy: object, mid: object,
    normal: object, face_length_m: object, cell_area_m2: object,
    owner: object, neighbour: object, boundary_index: object,
    cell_face_offset: object, cell_faces: object, signs: object,
    hydraulic_mode: object, bed_boundary_mode: object,
) -> BedloadAdaptationTopologyV72:
    resolved = load_execution_v72(execution)
    owner_raw = np.asarray(owner)
    if owner_raw.ndim != 1 or owner_raw.size < 1:
        raise BedloadAdaptationV72Error("Q3/v72 owner must be a nonempty vector")
    nface = int(owner_raw.size)
    area_raw = np.asarray(cell_area_m2)
    if area_raw.ndim != 1 or area_raw.size < 1:
        raise BedloadAdaptationV72Error(
            "Q3/v72 cell area must be a nonempty vector")
    ncell = int(area_raw.size)
    mode_raw = np.asarray(hydraulic_mode)
    if mode_raw.ndim != 1:
        raise BedloadAdaptationV72Error(
            "Q3/v72 hydraulic modes must be a vector")
    nboundary = int(mode_raw.size)
    offset_raw = np.asarray(cell_face_offset)
    if offset_raw.shape != (ncell + 1,):
        raise BedloadAdaptationV72Error(
            "Q3/v72 adjacency offset has the wrong shape")
    nadjacency = int(np.asarray(cell_faces).size)
    arrays = {
        "xy": _array(xy, (2, ncell), "cell coordinates"),
        "mid": _array(mid, (2, nface), "face coordinates"),
        "normal": _array(normal, (2, nface), "face normals"),
        "face_length_m": _array(face_length_m, (nface,), "face length"),
        "cell_area_m2": _array(cell_area_m2, (ncell,), "cell area"),
        "owner": _array(owner, (nface,), "owner", integer=True),
        "neighbour": _array(neighbour, (nface,), "neighbour", integer=True),
        "boundary_index": _array(
            boundary_index, (nface,), "boundary index", integer=True),
        "cell_face_offset": _array(
            cell_face_offset, (ncell + 1,), "cell-face offset", integer=True),
        "cell_faces": _array(
            cell_faces, (nadjacency,), "cell faces", integer=True),
        "signs": _array(signs, (nadjacency,), "cell-face signs", integer=True),
        "hydraulic_mode": _array(
            hydraulic_mode, (nboundary,), "hydraulic mode", integer=True),
        "bed_boundary_mode": _array(
            bed_boundary_mode, (nboundary,), "bed boundary mode", integer=True),
    }
    if (np.any(arrays["face_length_m"] <= 0.0) or
            np.any(arrays["cell_area_m2"] <= 0.0) or
            np.any((arrays["owner"] < 0) | (arrays["owner"] >= ncell)) or
            np.any((arrays["neighbour"] < -1) |
                   (arrays["neighbour"] >= ncell)) or
            np.any((arrays["boundary_index"] < -1) |
                   (arrays["boundary_index"] >= nboundary)) or
            np.any((arrays["neighbour"] >= 0) &
                   (arrays["boundary_index"] >= 0)) or
            arrays["cell_face_offset"][0] != 0 or
            arrays["cell_face_offset"][-1] != nadjacency or
            np.any(np.diff(arrays["cell_face_offset"]) < 0) or
            np.any((arrays["cell_faces"] < 0) |
                   (arrays["cell_faces"] >= nface)) or
            not np.isin(arrays["signs"], (-1, 1)).all() or
            not np.isin(arrays["bed_boundary_mode"],
                        (BOUNDARY_FEED, BOUNDARY_EQUILIBRIUM)).all()):
        raise BedloadAdaptationV72Error("Q3/v72 topology is invalid")
    for cell in range(ncell):
        start, finish = arrays["cell_face_offset"][cell:cell + 2]
        faces = arrays["cell_faces"][start:finish]
        signs_row = arrays["signs"][start:finish]
        cells = np.where(signs_row > 0, arrays["owner"][faces],
                         arrays["neighbour"][faces])
        if np.any(cells != cell):
            raise BedloadAdaptationV72Error(
                "Q3/v72 adjacency belongs to another cell")
    digest = _array_digest(*arrays.values())
    token = _fingerprint({
        "scope_id": SCOPE_ID,
        "execution_fingerprint": resolved["fingerprint"],
        "dependency_fingerprint": resolved["topology_fingerprint"],
        "ncell": ncell, "nface": nface, "nboundary": nboundary,
        "arrays_sha256": digest,
    })
    for value in arrays.values():
        value.setflags(write=False)
    return BedloadAdaptationTopologyV72(
        resolved["fingerprint"], ncell, nface, nboundary,
        arrays["xy"], arrays["mid"], arrays["normal"],
        arrays["face_length_m"], arrays["cell_area_m2"], arrays["owner"],
        arrays["neighbour"], arrays["boundary_index"],
        arrays["cell_face_offset"], arrays["cell_faces"], arrays["signs"],
        arrays["hydraulic_mode"], arrays["bed_boundary_mode"], token,
    )


def validate_topology_v72(
    execution: Mapping[str, object], topology: BedloadAdaptationTopologyV72,
) -> BedloadAdaptationTopologyV72:
    if not isinstance(topology, BedloadAdaptationTopologyV72):
        raise BedloadAdaptationV72Error("Q3/v72 topology receipt is missing")
    rebuilt = resolve_topology_v72(
        execution, xy=topology.xy, mid=topology.mid, normal=topology.normal,
        face_length_m=topology.face_length_m,
        cell_area_m2=topology.cell_area_m2, owner=topology.owner,
        neighbour=topology.neighbour, boundary_index=topology.boundary_index,
        cell_face_offset=topology.cell_face_offset,
        cell_faces=topology.cell_faces, signs=topology.signs,
        hydraulic_mode=topology.hydraulic_mode,
        bed_boundary_mode=topology.bed_boundary_mode,
    )
    if rebuilt.topology_sha256 != topology.topology_sha256:
        raise BedloadAdaptationV72Error(
            "Q3/v72 topology receipt is stale or mutated")
    return topology


def relax_capacity_v72(
    execution: Mapping[str, object], topology: BedloadAdaptationTopologyV72,
    *, q: object, equilibrium_capacity: object, boundary_feed_kg_m_s: object,
    dry_tolerance: object, dx_m: object, dy_m: object,
    max_iterations: int = 4096, cancelled=lambda: False,
) -> BedloadAdaptationResultV72:
    """Reference Jacobi solve matching the native and actual-CUDA kernels."""
    resolved = load_execution_v72(execution)
    validate_topology_v72(resolved, topology)
    if not callable(cancelled):
        raise TypeError("cancelled must be callable")
    if cancelled():
        raise InterruptedError("Q3/v72 adaptation cancelled before solve")
    if (isinstance(max_iterations, bool) or type(max_iterations) is not int or
            max_iterations < 1):
        raise BedloadAdaptationV72Error(
            "Q3/v72 max_iterations must be a positive integer")
    nclass = len(resolved["fraction_ids"])
    state = _array(q, (3, topology.ncell), "hydraulic state")
    equilibrium = _array(
        equilibrium_capacity, (2, nclass, topology.ncell),
        "equilibrium capacity")
    feed = _array(
        boundary_feed_kg_m_s, (nclass, topology.nboundary), "boundary feed")
    if np.any(feed < 0.0):
        raise BedloadAdaptationV72Error(
            "Q3/v72 boundary feed must be nonnegative")
    dry = _finite(dry_tolerance, "dry tolerance", nonnegative=True)
    dx = _finite(dx_m, "dx", positive=True)
    dy = _finite(dy_m, "dy", positive=True)
    d50 = np.asarray(resolved["d50_m"], dtype=np.float64)
    factor = float(resolved["length_diameters"])
    boundary_span = np.bincount(
        topology.boundary_index[topology.boundary_index >= 0],
        weights=topology.face_length_m[topology.boundary_index >= 0],
        minlength=topology.nboundary,
    ).astype(np.float64)
    velocity = np.zeros((2, topology.ncell), dtype=np.float64)
    wet = state[0] > dry
    velocity[:, wet] = state[1:, wet] / state[0, wet]
    speed = np.sqrt(np.sum(velocity * velocity, axis=0))
    weights = np.zeros(topology.cell_faces.size, dtype=np.float64)
    distance = np.zeros(topology.ncell, dtype=np.float64)
    total = np.zeros(topology.ncell, dtype=np.float64)
    for cell in range(topology.ncell):
        dsum = 0.0
        for item in range(topology.cell_face_offset[cell],
                          topology.cell_face_offset[cell + 1]):
            face = int(topology.cell_faces[item])
            sign = int(topology.signs[item])
            other = (int(topology.neighbour[face]) if sign > 0 else
                     int(topology.owner[face]))
            outward = sign * topology.normal[:, face]
            segment = 0.0
            weight = 0.0
            if wet[cell] and speed[cell] > 0.0:
                if other >= 0:
                    face_velocity = 0.5 * (velocity[:, cell] +
                                           velocity[:, other])
                    inflow = -float(np.dot(face_velocity, outward))
                    if inflow > 0.0:
                        weight = inflow * topology.face_length_m[face]
                        face_speed = float(np.linalg.norm(face_velocity))
                        delta = topology.xy[:, cell] - topology.xy[:, other]
                        if face_speed > 0.0:
                            segment = abs(float(np.dot(
                                delta, face_velocity / face_speed)))
                        if segment <= 64.0 * np.finfo(float).eps * max(dx, dy):
                            segment = float(np.linalg.norm(delta))
                else:
                    boundary = int(topology.boundary_index[face])
                    if (boundary >= 0 and
                            int(topology.hydraulic_mode[boundary]) in
                            INLET_HYDRAULIC_MODES):
                        inflow = -float(np.dot(velocity[:, cell], outward))
                        if inflow > 0.0:
                            weight = inflow * topology.face_length_m[face]
                            delta = (topology.xy[:, cell] -
                                     topology.mid[:, face])
                            segment = abs(float(np.dot(
                                delta, velocity[:, cell] / speed[cell])))
                            if segment <= (64.0 * np.finfo(float).eps *
                                           max(dx, dy)):
                                segment = float(np.linalg.norm(delta))
            weights[item] = weight
            total[cell] += weight
            dsum += weight * segment
        if total[cell] > 0.0:
            distance[cell] = dsum / total[cell]
            if distance[cell] <= 64.0 * np.finfo(float).eps * max(dx, dy):
                distance[cell] = (topology.cell_area_m2[cell] * speed[cell] /
                                  total[cell])
    previous = equilibrium.copy()
    following = np.empty_like(previous)
    scale = max(float(np.max(np.abs(equilibrium))), 1.0e-20)
    tolerance = 1.0e-11 * scale
    residual = math.inf
    converged = False
    for iteration in range(1, max_iterations + 1):
        if cancelled():
            raise InterruptedError("Q3/v72 adaptation cancelled during solve")
        following[...] = equilibrium
        for cell in range(topology.ncell):
            if total[cell] <= 0.0:
                continue
            for grain in range(nclass):
                upstream = np.zeros(2, dtype=np.float64)
                for item in range(topology.cell_face_offset[cell],
                                  topology.cell_face_offset[cell + 1]):
                    weight = weights[item]
                    if weight <= 0.0:
                        continue
                    face = int(topology.cell_faces[item])
                    sign = int(topology.signs[item])
                    other = (int(topology.neighbour[face]) if sign > 0 else
                             int(topology.owner[face]))
                    if other >= 0:
                        source = previous[:, grain, other]
                    else:
                        boundary = int(topology.boundary_index[face])
                        if topology.bed_boundary_mode[boundary] == BOUNDARY_EQUILIBRIUM:
                            source = equilibrium[:, grain, cell]
                        elif boundary_span[boundary] > 0.0:
                            source = (-sign * topology.normal[:, face] *
                                      feed[grain, boundary] /
                                      boundary_span[boundary])
                        else:
                            source = np.zeros(2, dtype=np.float64)
                    upstream += weight * source
                attenuation = math.exp(
                    -distance[cell] / (factor * d50[grain]))
                following[:, grain, cell] = (
                    (1.0 - attenuation) * equilibrium[:, grain, cell] +
                    attenuation * upstream / total[cell]
                )
        residual = float(np.max(np.abs(following - previous)))
        if not math.isfinite(residual):
            raise BedloadAdaptationV72Error(
                "Q3/v72 adaptation produced a nonfinite capacity")
        if residual <= tolerance:
            converged = True
            break
        previous, following = following, previous
    if not converged:
        raise BedloadAdaptationV72Error(
            "Q3/v72 adaptation solve did not converge")
    adapted = np.ascontiguousarray(following)
    parent_digest = _array_digest(equilibrium)
    token = _fingerprint({
        "scope_id": SCOPE_ID,
        "execution_fingerprint": resolved["fingerprint"],
        "topology_sha256": topology.topology_sha256,
        "parent_capacity_sha256": parent_digest,
        "adapted_capacity_sha256": _array_digest(adapted),
        "iterations": iteration, "residual": residual,
        "tolerance": tolerance,
        "dry_identity_count": int(np.count_nonzero(~wet | (speed <= 0.0))),
    })
    adapted.setflags(write=False)
    return BedloadAdaptationResultV72(
        resolved["fingerprint"], topology.topology_sha256, parent_digest,
        adapted, iteration, residual, tolerance,
        int(np.count_nonzero(~wet | (speed <= 0.0))), token,
    )


def restart_document_v72(execution: Mapping[str, object]) -> dict[str, object]:
    """Checkpoint identity proving that adaptation owns no hidden state."""
    resolved = load_execution_v72(execution)
    body: dict[str, object] = {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "execution_fingerprint": resolved["fingerprint"],
        "accepted_state_fields": [],
        "trial_fields": [],
        "restart_action": "recompute_from_restored_parent_capacity",
    }
    return {**body, "fingerprint": _fingerprint(body)}


def verify_sources_v72(root: str | Path = ".") -> dict[str, str]:
    """Hash the reused kernels and additive native/GPU composition sources."""
    base = Path(root)
    paths = (
        "validation/gpu_fwave/bedload_adaptation.cu",
        "validation/gpu_fwave/bedload_adaptation.py",
        "solver/wave/braids_bedload_adaptation_live_v72_module.f90",
        "braids_gpu/bedload_adaptation_live_v72.py",
    )
    result: dict[str, str] = {}
    for relative in paths:
        path = base / relative
        if not path.is_file():
            raise BedloadAdaptationV72Error(
                f"Q3/v72 implementation source is missing: {relative}")
        result[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


__all__ = [
    "BACKEND_CPU", "BACKEND_GPU", "BOUNDARY_EQUILIBRIUM", "BOUNDARY_FEED",
    "CAPACITY_DIRECT", "CAPACITY_SCALAR", "ENVELOPE_VERSION", "METHOD_ID",
    "MAX_ACCEPTED_STEPS", "SCHEMA", "SCOPE_ID", "SEMANTIC_ID",
    "BedloadAdaptationResultV72", "BedloadAdaptationTopologyV72",
    "BedloadAdaptationV72Error", "bedload_adaptation_semantic_document_v72",
    "execution_document_v72", "load_execution_v72", "relax_capacity_v72",
    "resolve_topology_v72", "restart_document_v72", "validate_topology_v72",
    "verify_sources_v72",
]
