"""Delft3D4 FLOW 2DH ``DensIn`` feedback for the BRAIDS f-wave path.

The locked FLOW sources first form

``rho = rhowat + sum_l C_l * (1 - rhowat/rhosol_l)``

and, after constituent transport, retain that density for the *next*
momentum solve.  For ``KMAX=1`` (``thick=1``, ``sig=-1/2``), FLOW's
``cucnp`` baroclinic acceleration at an internal face reduces exactly to

``a_n = -g (rho_R-rho_L) (h_R+h_L) / (4 rho_ref distance)``.

V85 is an additive, bounded operator transaction.  It binds an accepted
suspended state to one converged augmented-f-wave midpoint/base-rate receipt,
adds the density source once, and publishes the next hydraulic candidate.
The BRAIDS face-to-cell projection is a weighted local least-squares
projection (a spatial-discretisation choice, not a claim of staggered-grid
identity).  The admitted scope is fixed-grid, fully wet, KMAX=1, default
baroclinic boundaries, ``ICREEP=0`` and at most four accepted updates.
AMR, cut cells, drying during the hydraulic step and unsupported boundary
density modes fail closed.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Mapping

import numpy as np


SCHEMA = 1
SCOPE_ID = "braids-q4-densin-fwave-v85-v1"
CONCENTRATION_RECEIPT_SCOPE_ID = SCOPE_ID + ":accepted-concentration"
HYDRAULIC_RECEIPT_SCOPE_ID = SCOPE_ID + ":implicit-midpoint-trial"
CHECKPOINT_SCOPE_ID = SCOPE_ID + ":checkpoint"
INTEGRATOR = "implicit_fwave_midpoint"
MAX_ACCEPTED_UPDATES = 4
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")

LOCKED_UPSTREAM_COMMIT = "c05ed672f0a7fe81c5b9867ef354cfdd53c25379"
LOCKED_SOURCES = {
    "dens": (
        "resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/"
        "flow2d3d_kernel/src/compute/dens.f90",
        "a51ef91d3e4a4bf06cce7fed836408a99370323c64f6de6c3b4b7d5366dd8ca4",
    ),
    "cucnp": (
        "resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/"
        "flow2d3d_kernel/src/compute/cucnp.f90",
        "c5ff08747da0176717aaa849ae85ba8bfc7bbc548d4ca756dea0e464a7febab2",
    ),
    "chkgeo": (
        "resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/"
        "flow2d3d_kernel/src/inichk/chkgeo.f90",
        "70c0e5b32dd4d374ad3737406dc015936aff8b33ee6ae5ac45e46d66d70dda9f",
    ),
    "trisol": (
        "resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/"
        "flow2d3d_kernel/src/main/trisol.f90",
        "4b3e7ce63aed6c0cf95fe20ea3b9aa80fee9b3cd6fdf8101f0fbf7af5ca689e0",
    ),
}


class DensInV85Error(ValueError):
    """The bounded v85 identity, receipt or transaction is invalid."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError, OverflowError) as error:
        raise DensInV85Error(
            "v85 value is not finite canonical JSON") from error


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _digest(*values: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(value)
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(_canonical(list(array.shape)))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise DensInV85Error(f"v85 {name} must be a lowercase SHA-256")
    return value


def _finite(value: object, name: str, *, positive: bool = False) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(float(value))):
        raise DensInV85Error(f"v85 {name} must be finite numeric data")
    result = float(value)
    if positive and result <= 0.0:
        raise DensInV85Error(f"v85 {name} must be positive")
    return result


def _array(value: object, shape: tuple[int, ...], name: str, *,
           positive: bool = False, nonnegative: bool = False,
           integer: bool = False) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != shape or raw.dtype.kind not in "fiu":
        raise DensInV85Error(f"v85 {name} must be numeric with shape {shape}")
    dtype = np.int32 if integer else np.float64
    result = np.ascontiguousarray(raw, dtype=dtype)
    if not np.isfinite(result).all():
        raise DensInV85Error(f"v85 {name} must be finite")
    if positive and np.any(result <= 0):
        raise DensInV85Error(f"v85 {name} must be positive")
    if nonnegative and np.any(result < 0):
        raise DensInV85Error(f"v85 {name} must be nonnegative")
    result.setflags(write=False)
    return result


def verify_locked_sources_v85(root: str | Path) -> dict[str, str]:
    """Verify, without modifying, the exact locked source files used here."""
    root = Path(root)
    result: dict[str, str] = {}
    for symbol, (relative, expected) in LOCKED_SOURCES.items():
        path = root / relative
        if not path.is_file():
            raise DensInV85Error("v85 locked source is absent: " + relative)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise DensInV85Error("v85 locked source hash differs: " + relative)
        result[symbol] = actual
    return result


def mixture_density_kg_m3_v85(
    concentration_kg_m3: object, water_density_kg_m3: object,
    solid_density_kg_m3: object,
) -> np.ndarray:
    """Exact ``dens.f90`` sediment contribution, in mass-concentration units."""
    concentration = np.asarray(concentration_kg_m3)
    if concentration.ndim != 2:
        raise DensInV85Error("v85 concentration must be fraction by cell")
    ng, nc = concentration.shape
    c = _array(concentration, (ng, nc), "concentration", nonnegative=True)
    water = _array(water_density_kg_m3, (nc,), "water density", positive=True)
    solid = _array(solid_density_kg_m3, (ng,), "solid density", positive=True)
    if np.any(solid[:, None] <= water[None, :]):
        raise DensInV85Error("v85 solid density must exceed water density")
    result = water + np.sum(
        c * (1.0-water[None, :]/solid[:, None]), axis=0)
    if not np.isfinite(result).all() or np.any(result <= 0.0):
        raise DensInV85Error("v85 mixture density is invalid")
    return np.ascontiguousarray(result)


def delft_kmax1_face_acceleration_v85(
    depth_m: object, mixture_density_kg_m3: object, owner: object,
    neighbour: object, distance_m: object, *, gravity_m_s2: float,
    hydraulic_reference_density_kg_m3: float,
) -> np.ndarray:
    """Exact KMAX=1 FLOW face-normal baroclinic acceleration.

    Physical boundary faces use the bounded v85 zero-normal-density-gradient
    policy and therefore return zero.  Internal faces use ``ICREEP=0`` and
    the default active baroclinic flag.
    """
    h = np.asarray(depth_m)
    if h.ndim != 1:
        raise DensInV85Error("v85 depth must be one-dimensional")
    nc = h.size
    rho = _array(mixture_density_kg_m3, (nc,), "mixture density", positive=True)
    h = _array(h, (nc,), "midpoint depth", positive=True)
    own_raw, nei_raw, dist_raw = map(np.asarray, (owner, neighbour, distance_m))
    if own_raw.ndim != 1 or nei_raw.shape != own_raw.shape or dist_raw.shape != own_raw.shape:
        raise DensInV85Error("v85 face arrays must have one identical dimension")
    nf = own_raw.size
    own = _array(own_raw, (nf,), "face owner", integer=True)
    nei = _array(nei_raw, (nf,), "face neighbour", integer=True)
    dist = _array(dist_raw, (nf,), "face distance", positive=True)
    if np.any(own < 0) or np.any(own >= nc) or np.any(nei >= nc) or np.any(nei < -1):
        raise DensInV85Error("v85 face connectivity is invalid")
    gravity = _finite(gravity_m_s2, "gravity", positive=True)
    reference = _finite(
        hydraulic_reference_density_kg_m3, "hydraulic reference density",
        positive=True)
    result = np.zeros(nf, dtype=np.float64)
    internal = nei >= 0
    left, right = own[internal], nei[internal]
    result[internal] = (
        -gravity*(rho[right]-rho[left])*(h[right]+h[left]) /
        (4.0*reference*dist[internal]))
    return result


def project_face_acceleration_v85(
    face_normal_acceleration_m_s2: object, normal: object,
    face_length_m: object, cell_face_offset: object, cell_faces: object,
    cell_face_sign: object,
) -> np.ndarray:
    """Project staggered face-normal acceleration to BRAIDS cell momenta."""
    accel_raw = np.asarray(face_normal_acceleration_m_s2)
    if accel_raw.ndim != 1:
        raise DensInV85Error("v85 face acceleration must be one-dimensional")
    nf = accel_raw.size
    accel = _array(accel_raw, (nf,), "face acceleration")
    normal = _array(normal, (2, nf), "face normal")
    length = _array(face_length_m, (nf,), "face length", positive=True)
    offset_raw = np.asarray(cell_face_offset)
    if offset_raw.ndim != 1 or offset_raw.size < 2:
        raise DensInV85Error("v85 cell-face offsets are invalid")
    nc = offset_raw.size-1
    offset = _array(offset_raw, (nc+1,), "cell-face offset", integer=True)
    faces_raw, signs_raw = np.asarray(cell_faces), np.asarray(cell_face_sign)
    if faces_raw.shape != signs_raw.shape or faces_raw.ndim != 1:
        raise DensInV85Error("v85 cell-face adjacency arrays differ")
    nadj = faces_raw.size
    faces = _array(faces_raw, (nadj,), "cell faces", integer=True)
    signs = _array(signs_raw, (nadj,), "cell-face signs", integer=True)
    if (offset[0] != 0 or offset[-1] != nadj or np.any(np.diff(offset) < 0) or
            np.any(faces < 0) or np.any(faces >= nf) or
            np.any((signs != -1) & (signs != 1))):
        raise DensInV85Error("v85 cell-face adjacency is invalid")
    result = np.zeros((2, nc), dtype=np.float64)
    for cell in range(nc):
        matrix = np.zeros((2, 2), dtype=np.float64)
        rhs = np.zeros(2, dtype=np.float64)
        for slot in range(offset[cell], offset[cell+1]):
            face = faces[slot]
            outward = signs[slot]*normal[:, face]
            directional = signs[slot]*accel[face]
            weight = length[face]
            matrix += weight*np.outer(outward, outward)
            rhs += weight*outward*directional
        determinant = matrix[0, 0]*matrix[1, 1]-matrix[0, 1]*matrix[1, 0]
        scale = max(1.0, float(np.max(np.abs(matrix))))
        if determinant <= 256.0*np.finfo(float).eps*scale*scale:
            raise DensInV85Error(
                "v85 face normals do not span the two momentum directions")
        result[:, cell] = np.linalg.solve(matrix, rhs)
    return result


@dataclass(frozen=True)
class DensInExecutionV85:
    role: str
    solid_density_kg_m3: np.ndarray
    water_density_kg_m3: np.ndarray
    hydraulic_reference_density_kg_m3: float
    gravity_m_s2: float
    dry_tolerance_m: float
    owner: np.ndarray
    neighbour: np.ndarray
    normal: np.ndarray
    face_length_m: np.ndarray
    face_distance_m: np.ndarray
    cell_area_m2: np.ndarray
    cell_face_offset: np.ndarray
    cell_faces: np.ndarray
    cell_face_sign: np.ndarray
    hydraulic_execution_sha256: str
    fingerprint: str

    @property
    def fraction_count(self) -> int:
        return int(self.solid_density_kg_m3.size)

    @property
    def cell_count(self) -> int:
        return int(self.water_density_kg_m3.size)

    @property
    def face_count(self) -> int:
        return int(self.owner.size)


def _execution_payload(value: DensInExecutionV85) -> dict:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "role": value.role,
        "integrator": INTEGRATOR,
        "kmax": 1,
        "fixed_grid": True,
        "fully_wet_hydraulic_scope": True,
        "amr": False,
        "cut_cells": False,
        "icreep": 0,
        "open_boundary_baroclinic_flag": 1,
        "boundary_density_policy": "zero_normal_gradient",
        "automatic_cpu_fallback": False,
        "accepted_update_limit": MAX_ACCEPTED_UPDATES,
        "solid_density_kg_m3": value.solid_density_kg_m3.tolist(),
        "water_density_kg_m3": value.water_density_kg_m3.tolist(),
        "hydraulic_reference_density_kg_m3":
            value.hydraulic_reference_density_kg_m3,
        "reference_density_ownership": (
            "upstream_water_EOS_owns_rhowat; hydraulic_configuration_owns_"
            "scalar_rhow_denominator; DensIn_only_adds_sediment_increment"),
        "gravity_m_s2": value.gravity_m_s2,
        "dry_tolerance_m": value.dry_tolerance_m,
        "geometry_sha256": _digest(
            value.owner, value.neighbour, value.normal, value.face_length_m,
            value.face_distance_m, value.cell_area_m2,
            value.cell_face_offset, value.cell_faces, value.cell_face_sign),
        "hydraulic_execution_sha256": value.hydraulic_execution_sha256,
        "mixture_density_equation":
            "rhowat+sum(C_l*(1-rhowat/rhosol_l))",
        "kmax1_face_equation":
            "-g*(rho_R-rho_L)*(h_R+h_L)/(4*rhow_ref*distance)",
        "braids_projection": "weighted_face_normal_least_squares",
        "source_call_order":
            "accepted_constituent_state_to_density_to_next_hydraulic_midpoint",
    }


def resolve_densin_execution_v85(
    *, role: str, solid_density_kg_m3: object,
    water_density_kg_m3: object,
    hydraulic_reference_density_kg_m3: float, gravity_m_s2: float,
    dry_tolerance_m: float, owner: object, neighbour: object, normal: object,
    face_length_m: object, face_distance_m: object, cell_area_m2: object,
    cell_face_offset: object, cell_faces: object, cell_face_sign: object,
    hydraulic_execution_sha256: str,
) -> DensInExecutionV85:
    if role not in ("native_cpu", "implicit_fwave_midpoint_gpu"):
        raise DensInV85Error("v85 backend must be explicitly selected")
    solid_raw, water_raw = np.asarray(solid_density_kg_m3), np.asarray(water_density_kg_m3)
    if solid_raw.ndim != 1 or water_raw.ndim != 1:
        raise DensInV85Error("v85 density fields must be one-dimensional")
    ng, nc = solid_raw.size, water_raw.size
    if not 1 <= ng <= 32 or nc < 1:
        raise DensInV85Error("v85 fraction/cell count is outside scope")
    own_raw = np.asarray(owner)
    if own_raw.ndim != 1 or own_raw.size < 1:
        raise DensInV85Error("v85 requires at least one face")
    nf = own_raw.size
    offset_raw, faces_raw = np.asarray(cell_face_offset), np.asarray(cell_faces)
    solid = _array(solid_raw, (ng,), "solid density", positive=True)
    water = _array(water_raw, (nc,), "water density", positive=True)
    if np.any(solid[:, None] <= water[None, :]):
        raise DensInV85Error("v85 solid density must exceed water density")
    arrays = (
        solid, water,
        _array(own_raw, (nf,), "owner", integer=True),
        _array(neighbour, (nf,), "neighbour", integer=True),
        _array(normal, (2, nf), "normal"),
        _array(face_length_m, (nf,), "face length", positive=True),
        _array(face_distance_m, (nf,), "face distance", positive=True),
        _array(cell_area_m2, (nc,), "cell area", positive=True),
        _array(offset_raw, (nc+1,), "cell-face offset", integer=True),
        _array(faces_raw, (faces_raw.size,), "cell faces", integer=True),
        _array(cell_face_sign, (faces_raw.size,), "cell-face signs", integer=True),
    )
    _, _, own, nei, normals, length, distance, area, offset, faces, signs = arrays
    if (np.any(own < 0) or np.any(own >= nc) or np.any(nei < -1) or
            np.any(nei >= nc) or np.any(np.abs(np.linalg.norm(normals, axis=0)-1.0) > 2e-13) or
            offset[0] != 0 or offset[-1] != faces.size or
            np.any(np.diff(offset) < 0) or np.any(faces < 0) or
            np.any(faces >= nf) or np.any((signs != -1) & (signs != 1))):
        raise DensInV85Error("v85 fixed-grid topology is invalid")
    # Exercise the projection geometry at admission time.
    project_face_acceleration_v85(
        np.zeros(nf), normals, length, offset, faces, signs)
    reference = _finite(hydraulic_reference_density_kg_m3,
                        "hydraulic reference density", positive=True)
    gravity = _finite(gravity_m_s2, "gravity", positive=True)
    dry = _finite(dry_tolerance_m, "dry tolerance")
    if dry < 0.0:
        raise DensInV85Error("v85 dry tolerance must be nonnegative")
    placeholder = DensInExecutionV85(
        role, solid, water, reference, gravity, dry, own, nei, normals,
        length, distance, area, offset, faces, signs,
        _sha(hydraulic_execution_sha256, "hydraulic execution digest"), "")
    return DensInExecutionV85(
        **{**placeholder.__dict__,
           "fingerprint": _fingerprint(_execution_payload(placeholder))})


def execution_document_v85(value: DensInExecutionV85) -> dict:
    if not isinstance(value, DensInExecutionV85):
        raise DensInV85Error("v85 execution object differs")
    payload = _execution_payload(value)
    if value.fingerprint != _fingerprint(payload):
        raise DensInV85Error("v85 execution fingerprint differs")
    return {**payload, "fingerprint": value.fingerprint}


@dataclass(frozen=True)
class AcceptedConcentrationReceiptV85:
    execution_fingerprint: str
    source_kind: str
    suspended_state_sha256: str
    accepted_update: int
    time_s: float
    depth_m: np.ndarray
    suspended_mass_kg_m2: np.ndarray
    concentration_kg_m3: np.ndarray
    mixture_density_kg_m3: np.ndarray
    receipt_sha256: str


def _concentration_token(value: AcceptedConcentrationReceiptV85) -> str:
    return _fingerprint({
        "scope_id": CONCENTRATION_RECEIPT_SCOPE_ID,
        "execution_fingerprint": value.execution_fingerprint,
        "source_kind": value.source_kind,
        "suspended_state_sha256": value.suspended_state_sha256,
        "accepted_update": value.accepted_update,
        "time_s": value.time_s,
        "arrays_sha256": _digest(
            value.depth_m, value.suspended_mass_kg_m2,
            value.concentration_kg_m3, value.mixture_density_kg_m3),
        "accepted": True,
        "call_order": "accepted_suspension_before_next_hydraulics",
    })


def make_accepted_concentration_receipt_v85(
    execution: DensInExecutionV85, *, suspended_state_sha256: str,
    accepted_update: int, time_s: float, depth_m: object,
    suspended_mass_kg_m2: object,
    _source_kind: str = "standalone_equation_oracle",
) -> AcceptedConcentrationReceiptV85:
    execution_document_v85(execution)
    if (type(accepted_update) is not int or not
            0 <= accepted_update <= MAX_ACCEPTED_UPDATES):
        raise DensInV85Error("v85 accepted concentration update is outside scope")
    time = _finite(time_s, "accepted concentration time")
    if time < 0.0:
        raise DensInV85Error("v85 accepted concentration time is negative")
    nc, ng = execution.cell_count, execution.fraction_count
    depth = _array(depth_m, (nc,), "accepted concentration depth", nonnegative=True)
    mass = _array(suspended_mass_kg_m2, (ng, nc),
                  "accepted suspended mass", nonnegative=True)
    wet = depth > execution.dry_tolerance_m
    if np.any(mass[:, ~wet] != 0.0):
        raise DensInV85Error(
            "v85 dry cells must be cleaned before density feedback")
    concentration = np.zeros_like(mass)
    concentration[:, wet] = mass[:, wet]/depth[None, wet]
    mixture = mixture_density_kg_m3_v85(
        concentration, execution.water_density_kg_m3,
        execution.solid_density_kg_m3)
    for value in (concentration, mixture):
        value.setflags(write=False)
    placeholder = AcceptedConcentrationReceiptV85(
        execution.fingerprint, _source_kind,
        _sha(suspended_state_sha256, "suspended state digest"),
        accepted_update, time, depth, mass, concentration, mixture, "")
    return AcceptedConcentrationReceiptV85(
        **{**placeholder.__dict__,
           "receipt_sha256": _concentration_token(placeholder)})


def validate_accepted_concentration_receipt_v85(
    value: AcceptedConcentrationReceiptV85, execution: DensInExecutionV85, *,
    accepted_update: int, time_s: float, require_typed_v78: bool = False,
) -> None:
    if (not isinstance(value, AcceptedConcentrationReceiptV85) or
            value.execution_fingerprint != execution.fingerprint or
            value.accepted_update != accepted_update or
            value.time_s != float(time_s) or
            value.receipt_sha256 != _concentration_token(value)):
        raise DensInV85Error(
            "v85 accepted concentration receipt is stale or mutated")
    expected = mixture_density_kg_m3_v85(
        value.concentration_kg_m3, execution.water_density_kg_m3,
        execution.solid_density_kg_m3)
    reconstructed_mass = value.concentration_kg_m3*value.depth_m[None, :]
    mass_scale = max(1.0, float(np.max(value.suspended_mass_kg_m2)))
    if (not np.array_equal(expected, value.mixture_density_kg_m3) or
            np.max(np.abs(value.suspended_mass_kg_m2-reconstructed_mass)) >
            32.0*np.finfo(np.float64).eps*mass_scale):
        raise DensInV85Error("v85 accepted concentration receipt is inconsistent")
    if require_typed_v78 and value.source_kind != "typed_v78_accepted_state":
        raise DensInV85Error(
            "v85 coupled transaction requires a typed accepted v78 state")


def make_accepted_concentration_from_v78_v85(
    execution: DensInExecutionV85, *, suspended_execution,
    suspended_state, depth_m: object,
) -> AcceptedConcentrationReceiptV85:
    """Bind density feedback to a validated accepted Q4/v78 state.

    This is the admitted transaction handoff.  The generic receipt maker above
    remains useful only as an equation/source oracle and is rejected by the
    coupled CPU/CUDA transactions.
    """
    from .multifraction_suspended_lifecycle_v78 import (
        ROLE_CPU_ORACLE, ROLE_CUDA,
        MultifractionSuspendedExecutionV78,
        MultifractionSuspendedStateV78,
        validate_multifraction_suspended_state_v78,
    )
    if (not isinstance(suspended_execution, MultifractionSuspendedExecutionV78)
            or not isinstance(suspended_state, MultifractionSuspendedStateV78)):
        raise DensInV85Error("v85 requires typed v78 execution/state objects")
    try:
        validate_multifraction_suspended_state_v78(
            suspended_state, suspended_execution)
    except (TypeError, ValueError) as error:
        raise DensInV85Error("v85 accepted v78 state is invalid") from error
    expected_role = (ROLE_CPU_ORACLE if execution.role == "native_cpu"
                     else ROLE_CUDA)
    profile = suspended_execution.base_execution.profile
    if (suspended_execution.role != expected_role or
            suspended_execution.fraction_count != execution.fraction_count or
            suspended_execution.cell_count != execution.cell_count or
            not np.array_equal(
                np.asarray(profile.sediment_density_kg_m3, dtype=np.float64),
                execution.solid_density_kg_m3) or
            profile.water_density_kg_m3 !=
                execution.hydraulic_reference_density_kg_m3):
        raise DensInV85Error(
            "v85 v78 fractions, backend or reference density differ")
    return make_accepted_concentration_receipt_v85(
        execution, suspended_state_sha256=suspended_state.state_sha256,
        accepted_update=suspended_state.accepted_updates,
        time_s=suspended_state.time_s, depth_m=depth_m,
        suspended_mass_kg_m2=suspended_state.suspended_mass_kg_m2,
        _source_kind="typed_v78_accepted_state")


@dataclass(frozen=True)
class DensInStateV85:
    execution_fingerprint: str
    q: np.ndarray
    accepted_updates: int
    time_s: float
    accepted_chain_sha256: str
    state_sha256: str


def _state_token(value: DensInStateV85) -> str:
    return _fingerprint({
        "execution_fingerprint": value.execution_fingerprint,
        "q_sha256": _digest(value.q),
        "accepted_updates": value.accepted_updates,
        "time_s": value.time_s,
        "accepted_chain_sha256": value.accepted_chain_sha256,
    })


def _wrap_state(execution: DensInExecutionV85, *, q: object,
                accepted_updates: int, time_s: float,
                accepted_chain_sha256: str) -> DensInStateV85:
    execution_document_v85(execution)
    if (type(accepted_updates) is not int or not
            0 <= accepted_updates <= MAX_ACCEPTED_UPDATES):
        raise DensInV85Error("v85 state update count is outside scope")
    time = _finite(time_s, "state time")
    if time < 0.0:
        raise DensInV85Error("v85 state time is negative")
    state_q = _array(q, (3, execution.cell_count), "hydraulic state")
    if np.any(state_q[0] <= execution.dry_tolerance_m):
        raise DensInV85Error("v85 hydraulic coupling requires a fully wet state")
    chain = _sha(accepted_chain_sha256, "accepted chain")
    placeholder = DensInStateV85(
        execution.fingerprint, state_q, accepted_updates, time, chain, "")
    return DensInStateV85(
        **{**placeholder.__dict__, "state_sha256": _state_token(placeholder)})


def resolve_densin_state_v85(execution: DensInExecutionV85, *,
                             q: object) -> DensInStateV85:
    chain = _fingerprint({
        "scope_id": SCOPE_ID,
        "execution_fingerprint": execution.fingerprint,
        "initial_q_sha256": _digest(np.asarray(q, dtype=np.float64)),
    })
    return _wrap_state(execution, q=q, accepted_updates=0, time_s=0.0,
                       accepted_chain_sha256=chain)


def validate_densin_state_v85(state: DensInStateV85,
                              execution: DensInExecutionV85) -> None:
    if (not isinstance(state, DensInStateV85) or
            state.execution_fingerprint != execution.fingerprint or
            state.state_sha256 != _state_token(state)):
        raise DensInV85Error("v85 state is stale, mutated or mismatched")
    _wrap_state(execution, q=state.q,
                accepted_updates=state.accepted_updates, time_s=state.time_s,
                accepted_chain_sha256=state.accepted_chain_sha256)


def make_midpoint_fwave_trial_receipt_v85(
    execution: DensInExecutionV85, parent: DensInStateV85, *, dt_s: float,
    midpoint_q: object, augmented_fwave_base_rate: object,
    nonlinear_converged: bool = True,
) -> dict:
    """Seal the converged pre-DensIn augmented-f-wave midpoint operator."""
    validate_densin_state_v85(parent, execution)
    dt = _finite(dt_s, "hydraulic timestep", positive=True)
    midpoint = _array(midpoint_q, (3, execution.cell_count), "midpoint state")
    base_rate = _array(augmented_fwave_base_rate,
                       (3, execution.cell_count), "augmented f-wave base rate")
    if np.any(midpoint[0] <= execution.dry_tolerance_m):
        raise DensInV85Error("v85 midpoint must stay fully wet")
    if nonlinear_converged is not True:
        raise DensInV85Error("v85 requires a converged implicit midpoint")
    body = {
        "schema": SCHEMA,
        "scope_id": HYDRAULIC_RECEIPT_SCOPE_ID,
        "execution_fingerprint": execution.fingerprint,
        "hydraulic_execution_sha256": execution.hydraulic_execution_sha256,
        "integrator": INTEGRATOR,
        "fixed_grid": True,
        "fully_wet": True,
        "automatic_cpu_fallback": False,
        "parent_state_sha256": parent.state_sha256,
        "accepted_update": parent.accepted_updates+1,
        "time_before_s": parent.time_s,
        "dt_s": dt,
        "time_after_s": parent.time_s+dt,
        "nonlinear_converged": True,
        "midpoint_q_sha256": _digest(midpoint),
        "augmented_fwave_base_rate_sha256": _digest(base_rate),
        "density_source_owner_count": 0,
    }
    return {**body, "fingerprint": _fingerprint(body)}


def _validate_hydraulic_receipt(
    value: Mapping[str, object], execution: DensInExecutionV85,
    parent: DensInStateV85, dt: float, midpoint: np.ndarray,
    base_rate: np.ndarray,
) -> None:
    if not isinstance(value, Mapping) or "fingerprint" not in value:
        raise DensInV85Error("v85 hydraulic receipt is absent")
    body = {key: value[key] for key in value if key != "fingerprint"}
    expected = make_midpoint_fwave_trial_receipt_v85(
        execution, parent, dt_s=dt, midpoint_q=midpoint,
        augmented_fwave_base_rate=base_rate)
    if dict(value) != expected or value["fingerprint"] != _fingerprint(body):
        raise DensInV85Error("v85 hydraulic receipt is stale or mutated")


@dataclass(frozen=True)
class DensInTrialV85:
    execution_fingerprint: str
    parent_state_sha256: str
    concentration_receipt_sha256: str
    hydraulic_receipt_sha256: str
    candidate: DensInStateV85
    mixture_density_kg_m3: np.ndarray
    face_normal_acceleration_m_s2: np.ndarray
    cell_acceleration_m_s2: np.ndarray
    density_momentum_rate_m2_s2: np.ndarray
    density_application_count: int
    depth_mass_residual_m3: float
    suspended_inventory_residual_kg: float
    backend: str
    trial_sha256: str


def _trial_token(value: DensInTrialV85) -> str:
    return _fingerprint({
        "execution_fingerprint": value.execution_fingerprint,
        "parent_state_sha256": value.parent_state_sha256,
        "concentration_receipt_sha256": value.concentration_receipt_sha256,
        "hydraulic_receipt_sha256": value.hydraulic_receipt_sha256,
        "candidate_state_sha256": value.candidate.state_sha256,
        "arrays_sha256": _digest(
            value.mixture_density_kg_m3,
            value.face_normal_acceleration_m_s2,
            value.cell_acceleration_m_s2,
            value.density_momentum_rate_m2_s2),
        "density_application_count": value.density_application_count,
        "depth_mass_residual_m3": value.depth_mass_residual_m3,
        "suspended_inventory_residual_kg":
            value.suspended_inventory_residual_kg,
        "backend": value.backend,
    })


def _build_trial(
    execution: DensInExecutionV85, parent: DensInStateV85,
    concentration: AcceptedConcentrationReceiptV85,
    hydraulic_receipt: Mapping[str, object], *, candidate_q: object,
    mixture_density: object, face_acceleration: object,
    cell_acceleration: object, density_rate: object,
    baseline_candidate_depth_m: object, backend: str,
) -> DensInTrialV85:
    nc, nf = execution.cell_count, execution.face_count
    q = _array(candidate_q, (3, nc), "candidate hydraulic state")
    mixture = _array(mixture_density, (nc,), "trial mixture density", positive=True)
    face = _array(face_acceleration, (nf,), "trial face acceleration")
    cell = _array(cell_acceleration, (2, nc), "trial cell acceleration")
    rate = _array(density_rate, (2, nc), "trial density momentum rate")
    baseline_depth = _array(
        baseline_candidate_depth_m, (nc,), "baseline candidate depth")
    # The CUDA reduction evaluates the same per-fraction density equation in
    # a different floating-point operation order.  Authenticate the exact
    # readback through the trial digest below, but test this derived invariant
    # with a small ULP-scaled bound rather than impossible bit identity.
    mixture_scale = max(
        1.0, float(np.max(np.abs(concentration.mixture_density_kg_m3))))
    mixture_error = float(np.max(np.abs(
        mixture-concentration.mixture_density_kg_m3)))
    if mixture_error > 32.0*np.finfo(np.float64).eps*mixture_scale:
        raise DensInV85Error("v85 trial density differs from accepted concentration")
    if np.any(q[0] <= execution.dry_tolerance_m):
        raise DensInV85Error("v85 candidate left the fully wet scope")
    count = 1
    depth_residual = float(np.sum(
        execution.cell_area_m2*(q[0]-baseline_depth)))
    if not np.array_equal(q[0], baseline_depth):
        raise DensInV85Error("v85 DensIn changed the hydraulic mass equation")
    before_inventory = float(np.sum(
        concentration.suspended_mass_kg_m2*execution.cell_area_m2[None, :]))
    after_inventory = before_inventory
    inventory_residual = after_inventory-before_inventory
    chain = _fingerprint({
        "previous": parent.accepted_chain_sha256,
        "concentration_receipt": concentration.receipt_sha256,
        "hydraulic_receipt": hydraulic_receipt["fingerprint"],
        "density_source_sha256": _digest(rate),
        "density_application_count": count,
    })
    candidate = _wrap_state(
        execution, q=q, accepted_updates=parent.accepted_updates+1,
        time_s=float(hydraulic_receipt["time_after_s"]),
        accepted_chain_sha256=chain)
    placeholder = DensInTrialV85(
        execution.fingerprint, parent.state_sha256,
        concentration.receipt_sha256,
        str(hydraulic_receipt["fingerprint"]), candidate, mixture, face,
        cell, rate, count, depth_residual, inventory_residual, backend, "")
    return DensInTrialV85(
        **{**placeholder.__dict__, "trial_sha256": _trial_token(placeholder)})


def begin_densin_cpu_trial_v85(
    execution: DensInExecutionV85, parent: DensInStateV85, *,
    accepted_concentration: AcceptedConcentrationReceiptV85,
    hydraulic_trial_receipt: Mapping[str, object], midpoint_q: object,
    augmented_fwave_base_rate: object, dt_s: float,
    cancelled: bool = False,
) -> DensInTrialV85:
    """Compose the CPU augmented-f-wave rate and DensIn source once."""
    if execution.role != "native_cpu":
        raise DensInV85Error("v85 CPU transaction requires native_cpu role")
    validate_densin_state_v85(parent, execution)
    if cancelled:
        raise DensInV85Error("v85 CPU trial cancelled before publication")
    if parent.accepted_updates >= MAX_ACCEPTED_UPDATES:
        raise DensInV85Error("v85 accepted update limit was reached")
    validate_accepted_concentration_receipt_v85(
        accepted_concentration, execution,
        accepted_update=parent.accepted_updates, time_s=parent.time_s,
        require_typed_v78=True)
    nc = execution.cell_count
    midpoint = _array(midpoint_q, (3, nc), "midpoint state")
    base_rate = _array(augmented_fwave_base_rate, (3, nc), "base f-wave rate")
    dt = _finite(dt_s, "timestep", positive=True)
    _validate_hydraulic_receipt(
        hydraulic_trial_receipt, execution, parent, dt, midpoint, base_rate)
    if (not np.array_equal(parent.q[0], accepted_concentration.depth_m) or
            np.any(midpoint[0] <= execution.dry_tolerance_m)):
        raise DensInV85Error(
            "v85 accepted concentration depth differs from the parent state")
    face = delft_kmax1_face_acceleration_v85(
        midpoint[0], accepted_concentration.mixture_density_kg_m3,
        execution.owner, execution.neighbour, execution.face_distance_m,
        gravity_m_s2=execution.gravity_m_s2,
        hydraulic_reference_density_kg_m3=
            execution.hydraulic_reference_density_kg_m3)
    cell = project_face_acceleration_v85(
        face, execution.normal, execution.face_length_m,
        execution.cell_face_offset, execution.cell_faces,
        execution.cell_face_sign)
    rate = midpoint[0][None, :]*cell
    full_rate = np.array(base_rate, copy=True)
    full_rate[1:] += rate
    candidate_q = parent.q+dt*full_rate
    return _build_trial(
        execution, parent, accepted_concentration,
        hydraulic_trial_receipt, candidate_q=candidate_q,
        mixture_density=accepted_concentration.mixture_density_kg_m3,
        face_acceleration=face, cell_acceleration=cell,
        density_rate=rate,
        baseline_candidate_depth_m=parent.q[0]+dt*base_rate[0],
        backend="native_cpu")


def _validate_trial(execution: DensInExecutionV85, parent: DensInStateV85,
                    trial: DensInTrialV85) -> None:
    validate_densin_state_v85(parent, execution)
    if (not isinstance(trial, DensInTrialV85) or
            trial.execution_fingerprint != execution.fingerprint or
            trial.parent_state_sha256 != parent.state_sha256 or
            trial.density_application_count != 1 or
            trial.trial_sha256 != _trial_token(trial)):
        raise DensInV85Error("v85 trial is stale, mutated or mismatched")
    validate_densin_state_v85(trial.candidate, execution)
    if (trial.candidate.accepted_updates != parent.accepted_updates+1 or
            trial.depth_mass_residual_m3 != 0.0 or
            trial.suspended_inventory_residual_kg != 0.0):
        raise DensInV85Error("v85 trial ledger or update ownership differs")


def commit_densin_trial_v85(execution: DensInExecutionV85,
                            parent: DensInStateV85,
                            trial: DensInTrialV85) -> DensInStateV85:
    _validate_trial(execution, parent, trial)
    return trial.candidate


def rollback_densin_trial_v85(execution: DensInExecutionV85,
                              parent: DensInStateV85,
                              trial: DensInTrialV85) -> DensInStateV85:
    _validate_trial(execution, parent, trial)
    return parent


def cancel_densin_trial_v85(execution: DensInExecutionV85,
                            parent: DensInStateV85,
                            trial: DensInTrialV85) -> DensInStateV85:
    return rollback_densin_trial_v85(execution, parent, trial)


def densin_checkpoint_v85(execution: DensInExecutionV85,
                          state: DensInStateV85) -> dict:
    validate_densin_state_v85(state, execution)
    body = {
        "schema": SCHEMA,
        "scope_id": CHECKPOINT_SCOPE_ID,
        "execution_fingerprint": execution.fingerprint,
        "q": state.q.tolist(),
        "accepted_updates": state.accepted_updates,
        "time_s": state.time_s,
        "accepted_chain_sha256": state.accepted_chain_sha256,
        "state_sha256": state.state_sha256,
    }
    return {**body, "fingerprint": _fingerprint(body)}


def load_densin_checkpoint_v85(execution: DensInExecutionV85,
                               document: Mapping[str, object]) -> DensInStateV85:
    expected = {
        "schema", "scope_id", "execution_fingerprint", "q",
        "accepted_updates", "time_s", "accepted_chain_sha256",
        "state_sha256", "fingerprint",
    }
    if not isinstance(document, Mapping) or set(document) != expected:
        raise DensInV85Error("v85 checkpoint inventory differs")
    body = {key: document[key] for key in document if key != "fingerprint"}
    if (document["fingerprint"] != _fingerprint(body) or
            document["schema"] != SCHEMA or
            document["scope_id"] != CHECKPOINT_SCOPE_ID or
            document["execution_fingerprint"] != execution.fingerprint):
        raise DensInV85Error("v85 checkpoint identity differs")
    state = _wrap_state(
        execution, q=document["q"],
        accepted_updates=document["accepted_updates"],
        time_s=document["time_s"],
        accepted_chain_sha256=document["accepted_chain_sha256"])
    if state.state_sha256 != document["state_sha256"]:
        raise DensInV85Error("v85 checkpoint state digest differs")
    return state


def method_identity_v85() -> dict:
    return {
        "scope_id": SCOPE_ID,
        "delft3d_upstream_commit": LOCKED_UPSTREAM_COMMIT,
        "integrator": INTEGRATOR,
        "mixture_density": "locked_dens_f90_exact_mass_concentration_v1",
        "baroclinic_face_acceleration": "locked_cucnp_kmax1_reduction_v1",
        "braids_face_to_cell_projection":
            "weighted_face_normal_least_squares_v1",
        "momentum_coupling":
            "bounded_density_source_operator_composed_once_after_caller_"
            "authenticated_converged_midpoint_base_rate",
        "accepted_concentration_timing": "next_hydraulic_step_only",
        "fixed_grid": True,
        "fully_wet": True,
        "actual_cpu_operator": True,
        "actual_cuda_operator": True,
        "automatic_cpu_fallback": False,
        "not_claimed": [
            "AMR", "cut_cells", "drying_hydraulic_step", "KMAX_gt_1",
            "ICREEP_nonzero", "nondefault_open_boundary_baroclinic_mode",
            "salinity_or_temperature_EOS_ownership", "Delft_staggered_grid_identity",
            "production_implicit_midpoint_nonlinear_residual_integration",
            "full_coupled_solver_or_product_admission",
        ],
    }


__all__ = [
    "AcceptedConcentrationReceiptV85", "DensInExecutionV85",
    "DensInStateV85", "DensInTrialV85", "DensInV85Error",
    "INTEGRATOR", "LOCKED_SOURCES", "MAX_ACCEPTED_UPDATES",
    "begin_densin_cpu_trial_v85", "cancel_densin_trial_v85",
    "commit_densin_trial_v85", "delft_kmax1_face_acceleration_v85",
    "densin_checkpoint_v85", "execution_document_v85",
    "load_densin_checkpoint_v85", "make_accepted_concentration_receipt_v85",
    "make_accepted_concentration_from_v78_v85",
    "make_midpoint_fwave_trial_receipt_v85", "method_identity_v85",
    "mixture_density_kg_m3_v85", "project_face_acceleration_v85",
    "resolve_densin_execution_v85", "resolve_densin_state_v85",
    "rollback_densin_trial_v85", "validate_accepted_concentration_receipt_v85",
    "validate_densin_state_v85", "verify_locked_sources_v85",
]
