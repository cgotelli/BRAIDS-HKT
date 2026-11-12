"""Bounded Q4 fixed-grid suspended noncohesive lifecycle (v76).

This additive module composes the already authenticated FLOW 2DH physical
operators with BRAIDS' fixed-grid accepted-water-flux FCT discretisation.  It
does not alter, wrap, or copy Delft3D source.  The constitutive pieces follow
locked Delft3D4 FLOW 2DH ``SettleFrm=2``, Galappatti exchange, open-boundary
reversal/Thatcher--Harleman memory, and ``dicuv/0.7`` horizontal diffusion.
The scalar time/space discretisation is deliberately different: BRAIDS uses
explicit conservative FCT plus donor-limited diffusion, whereas FLOW uses its
ADI scalar solve.

Only accepted states own mass, boundary memory, clocks, and cumulative
ledgers.  A trial is immutable, is limited to four accepted updates, and can
be committed, rolled back, or cancelled without repairing accepted state.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np

from .flow2dh_boundary_diffusion import (
    FLOW2DH_HORIZONTAL_DIFFUSION_ID,
    FLOW2DH_HORIZONTAL_DIFFUSION_SEMANTIC_ID,
    FLOW2DH_OPEN_BOUNDARY_ID,
    FLOW2DH_OPEN_BOUNDARY_SEMANTIC_ID,
    Flow2DHBoundaryMemory,
    flow2dh_horizontal_diffusive_face_mass,
    resolve_flow2dh_open_boundary,
)
from .flow2dh_exchange import (
    FLOW2DH_EXCHANGE_ID,
    FLOW2DH_EXCHANGE_SEMANTIC_ID,
    advance_flow2dh_equilibrium_exchange,
)
from .flow2dh_settling import (
    FLOW2DH_SETTLING_FORM,
    FLOW2DH_SETTLING_ID,
    FLOW2DH_SETTLING_SEMANTIC_ID,
    settlefrm2_noncohesive,
)
from .suspension_profile import (
    ADVECTION_METHOD_ID,
    Flow2DHSuspensionProfile,
    load_suspension_profile,
    suspension_profile_document,
)


SCHEMA = 1
SCOPE_ID = "braids-q4-fixed-grid-suspended-lifecycle-v76-v1"
SEMANTIC_ID = hashlib.sha256(SCOPE_ID.encode("ascii")).hexdigest()
BACKEND_CPU = "cpu"
BACKEND_CUDA = "cuda"
BACKENDS = (BACKEND_CPU, BACKEND_CUDA)
MAX_ACCEPTED_UPDATES = 4
NUMERICAL_METHOD = "braids_accepted_water_flux_fct_explicit_diffusion_v1"
REFERENCE_NUMERICAL_METHOD = "delft3d4_flow2dh_adi_scalar_transport"
WET_DRY_METHOD = "deposit_entire_dry_water_inventory_to_same_class_bed_v1"
CHECKPOINT_SCOPE_ID = "braids-q4-fixed-grid-suspended-checkpoint-v76-v1"

LOCKED_SOURCE_ANCHORS = (
    "resources/Delft3D-main/src/utils_gpl/morphology/packages/morphology_kernel/src/eqtran.f90",
    "resources/Delft3D-main/src/utils_gpl/morphology/packages/morphology_kernel/src/soursin_2d.f90",
    "resources/Delft3D-main/src/utils_gpl/morphology/packages/morphology_kernel/src/eqsettle.f90",
    "resources/Delft3D-main/src/utils_gpl/morphology/packages/morphology_io/src/rdsed.f90",
    "resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/flow2d3d_kernel/src/compute/fallve.f90",
    "resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/flow2d3d_kernel/src/compute/difu.f90",
    "resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/flow2d3d_kernel/src/compute/difuflux.f90",
    "resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/flow2d3d_kernel/src/compute/thahbc.f90",
    "resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/flow2d3d_kernel/src/timedep/incbcc.f90",
)


class SuspendedLifecycleV76Error(ValueError):
    """Fail-closed v76 configuration, transaction, or ledger error."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, OverflowError) as error:
        raise SuspendedLifecycleV76Error(
            "v76 value is not finite canonical JSON") from error


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _array_digest(*values: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(_canonical(list(value.shape)))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _finite(value: object, name: str, *, positive: bool = False,
            nonnegative: bool = False) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(float(value))):
        raise SuspendedLifecycleV76Error(f"v76 {name} must be finite")
    result = float(value)
    if positive and result <= 0.0:
        raise SuspendedLifecycleV76Error(f"v76 {name} must be positive")
    if nonnegative and result < 0.0:
        raise SuspendedLifecycleV76Error(f"v76 {name} must be nonnegative")
    return result


def _array(value: object, shape: tuple[int, ...], name: str, *,
           dtype=np.float64, nonnegative: bool = False,
           positive: bool = False) -> np.ndarray:
    raw = np.asarray(value)
    expected = np.dtype(dtype)
    if raw.shape != shape:
        raise SuspendedLifecycleV76Error(
            f"v76 {name} must have shape {shape}, got {raw.shape}")
    if expected.kind in "iu":
        if raw.dtype.kind not in "iu":
            raise SuspendedLifecycleV76Error(f"v76 {name} must be integer")
    elif expected.kind == "b":
        if raw.dtype.kind != "b":
            raise SuspendedLifecycleV76Error(f"v76 {name} must be Boolean")
    elif raw.dtype.kind not in "fiu" or not np.isfinite(raw).all():
        raise SuspendedLifecycleV76Error(f"v76 {name} must be finite numeric")
    result = np.ascontiguousarray(raw, dtype=expected)
    if nonnegative and np.any(result < 0):
        raise SuspendedLifecycleV76Error(f"v76 {name} must be nonnegative")
    if positive and np.any(result <= 0):
        raise SuspendedLifecycleV76Error(f"v76 {name} must be positive")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class SuspendedLifecycleExecutionV76:
    backend: str
    profile: Flow2DHSuspensionProfile
    cell_area_m2: np.ndarray
    face_length_m: np.ndarray
    centre_distance_m: np.ndarray
    owner: np.ndarray
    neighbour: np.ndarray
    boundary_index: np.ndarray
    cell_face_offset: np.ndarray
    cell_faces: np.ndarray
    cell_face_sign: np.ndarray
    normal: np.ndarray
    rectangle_width_m: np.ndarray
    weno_cells: np.ndarray
    boundary_return_time_s: np.ndarray
    dry_tolerance_m: float
    semantic_id: str
    fingerprint: str

    @property
    def fraction_count(self) -> int:
        return len(self.profile.sediment_d50_m)

    @property
    def cell_count(self) -> int:
        return int(self.cell_area_m2.size)

    @property
    def face_count(self) -> int:
        return int(self.face_length_m.size)

    @property
    def boundary_count(self) -> int:
        return int(self.boundary_return_time_s.shape[1])


def _execution_payload(value: SuspendedLifecycleExecutionV76) -> dict:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "semantic_id": value.semantic_id,
        "backend": value.backend,
        "profile": suspension_profile_document(value.profile),
        "dry_tolerance_m": value.dry_tolerance_m,
        "cell_area_m2": value.cell_area_m2.tolist(),
        "face_length_m": value.face_length_m.tolist(),
        "centre_distance_m": value.centre_distance_m.tolist(),
        "owner": value.owner.tolist(),
        "neighbour": value.neighbour.tolist(),
        "boundary_index": value.boundary_index.tolist(),
        "cell_face_offset": value.cell_face_offset.tolist(),
        "cell_faces": value.cell_faces.tolist(),
        "cell_face_sign": value.cell_face_sign.tolist(),
        "normal": value.normal.tolist(),
        "rectangle_width_m": value.rectangle_width_m.tolist(),
        "weno_cells": value.weno_cells.tolist(),
        "boundary_return_time_s": value.boundary_return_time_s.tolist(),
        "accepted_update_limit": MAX_ACCEPTED_UPDATES,
        "numerical_method": NUMERICAL_METHOD,
        "reference_numerical_method": REFERENCE_NUMERICAL_METHOD,
        "numerical_method_difference": (
            "BRAIDS advances class mass with accepted-water-flux FCT and an "
            "explicit donor-limited diffusion pass; Delft3D4 FLOW advances "
            "the scalar with its ADI discretisation. Physical constitutive "
            "operators are shared identities, numerical solutions are not "
            "asserted bitwise or discretely equivalent."
        ),
        "wet_dry_cleanup": WET_DRY_METHOD,
        "locked_sources_read_only": True,
    }


def resolve_suspended_lifecycle_execution_v76(
    *, backend: str, suspension_profile: Flow2DHSuspensionProfile | Mapping[str, object],
    cell_area_m2: object, face_length_m: object, centre_distance_m: object,
    owner: object, neighbour: object, boundary_index: object,
    cell_face_offset: object, cell_faces: object, cell_face_sign: object,
    normal: object, rectangle_width_m: object, weno_cells: object,
    boundary_return_time_s: object, dry_tolerance_m: float,
) -> SuspendedLifecycleExecutionV76:
    """Authenticate one fixed-grid topology and its suspension profile."""
    if backend not in BACKENDS:
        raise SuspendedLifecycleV76Error(
            "v76 backend must be explicitly 'cpu' or 'cuda'")
    try:
        profile = (suspension_profile if isinstance(
            suspension_profile, Flow2DHSuspensionProfile) else
            load_suspension_profile(suspension_profile))
        suspension_profile_document(profile)
    except (TypeError, ValueError) as error:
        raise SuspendedLifecycleV76Error(
            "v76 suspension profile authentication failed") from error
    if (profile.settling_form != FLOW2DH_SETTLING_FORM or
            profile.advection_method_id != ADVECTION_METHOD_ID or
            profile.exchange_method_id != FLOW2DH_EXCHANGE_ID or
            profile.settling_operator_id != FLOW2DH_SETTLING_ID or
            profile.open_boundary_operator_id != FLOW2DH_OPEN_BOUNDARY_ID or
            profile.horizontal_diffusion_operator_id !=
            FLOW2DH_HORIZONTAL_DIFFUSION_ID or
            any(kind != "sand" for kind in profile.sediment_fraction_classes)):
        raise SuspendedLifecycleV76Error(
            "v76 requires the authenticated noncohesive FLOW 2DH profile")

    area_raw = np.asarray(cell_area_m2)
    length_raw = np.asarray(face_length_m)
    if area_raw.ndim != 1 or area_raw.size < 1 or length_raw.ndim != 1 or length_raw.size < 1:
        raise SuspendedLifecycleV76Error("v76 topology dimensions are empty")
    nc, nf, ng = int(area_raw.size), int(length_raw.size), len(profile.sediment_d50_m)
    area = _array(area_raw, (nc,), "cell area", positive=True)
    length = _array(length_raw, (nf,), "face length", positive=True)
    distance = _array(centre_distance_m, (nf,), "centre distance", positive=True)
    owner_a = _array(owner, (nf,), "owner", dtype=np.int32)
    neighbour_a = _array(neighbour, (nf,), "neighbour", dtype=np.int32)
    bid = _array(boundary_index, (nf,), "boundary index", dtype=np.int32)
    if (np.any((owner_a < 0) | (owner_a >= nc)) or
            np.any((neighbour_a < -1) | (neighbour_a >= nc)) or
            np.any(neighbour_a == owner_a) or np.any(bid < -1) or
            np.any((neighbour_a >= 0) & (bid >= 0))):
        raise SuspendedLifecycleV76Error("v76 face ownership is invalid")
    nb = int(np.max(bid) + 1) if np.any(bid >= 0) else 0
    if nb and set(bid[bid >= 0].tolist()) != set(range(nb)):
        raise SuspendedLifecycleV76Error(
            "v76 logical boundary IDs must be contiguous")
    offset = _array(cell_face_offset, (nc + 1,), "cell-face offset", dtype=np.int32)
    faces_raw = np.asarray(cell_faces)
    signs_raw = np.asarray(cell_face_sign)
    if faces_raw.ndim != 1 or signs_raw.shape != faces_raw.shape:
        raise SuspendedLifecycleV76Error("v76 cell-face CSR shapes differ")
    faces = _array(faces_raw, faces_raw.shape, "cell faces", dtype=np.int32)
    signs = _array(signs_raw, signs_raw.shape, "cell-face sign", dtype=np.int32)
    if (offset[0] != 0 or offset[-1] != faces.size or
            np.any(np.diff(offset) < 0) or np.any((faces < 0) | (faces >= nf)) or
            np.any(~np.isin(signs, (-1, 1)))):
        raise SuspendedLifecycleV76Error("v76 cell-face CSR is invalid")
    expected: list[tuple[int, int, int]] = []
    actual: list[tuple[int, int, int]] = []
    for face in range(nf):
        expected.append((int(owner_a[face]), face, 1))
        if neighbour_a[face] >= 0:
            expected.append((int(neighbour_a[face]), face, -1))
    for cell in range(nc):
        for slot in range(int(offset[cell]), int(offset[cell + 1])):
            actual.append((cell, int(faces[slot]), int(signs[slot])))
    if sorted(expected) != sorted(actual):
        raise SuspendedLifecycleV76Error(
            "v76 CSR must contain each physical face with owner +1 and neighbour -1")
    normal_a = _array(normal, (2, nf), "face normal")
    rectangle = _array(
        rectangle_width_m, (2, nc), "rectangle width", nonnegative=True)
    weno = _array(weno_cells, (5, 2, nf), "WENO cells", dtype=np.int32)
    if np.any((weno < -1) | (weno >= nc)):
        raise SuspendedLifecycleV76Error("v76 WENO cell indices are invalid")
    return_raw = np.asarray(boundary_return_time_s)
    if return_raw.shape == (nb,):
        return_raw = np.broadcast_to(return_raw[None, :], (ng, nb))
    return_time = _array(
        return_raw, (ng, nb), "boundary return time")
    dry = _finite(dry_tolerance_m, "dry tolerance", nonnegative=True)

    placeholder = SuspendedLifecycleExecutionV76(
        backend, profile, area, length, distance, owner_a, neighbour_a, bid,
        offset, faces, signs, normal_a, rectangle, weno, return_time, dry,
        SEMANTIC_ID, "",
    )
    return SuspendedLifecycleExecutionV76(
        **{**placeholder.__dict__, "fingerprint": _fingerprint(
            _execution_payload(placeholder))}
    )


def execution_document_v76(value: SuspendedLifecycleExecutionV76) -> dict:
    if not isinstance(value, SuspendedLifecycleExecutionV76):
        raise SuspendedLifecycleV76Error("v76 execution object differs")
    payload = _execution_payload(value)
    if value.fingerprint != _fingerprint(payload):
        raise SuspendedLifecycleV76Error("v76 execution fingerprint differs")
    return {**payload, "fingerprint": value.fingerprint}


@dataclass(frozen=True)
class SuspendedLifecycleStateV76:
    execution_fingerprint: str
    suspended_mass_kg_m2: np.ndarray
    bed_mass_kg_m2: np.ndarray
    boundary_last_outflow_concentration_kg_m3: np.ndarray
    boundary_return_clock_s: np.ndarray
    advective_boundary_ledger_kg: np.ndarray
    diffusive_boundary_ledger_kg: np.ndarray
    water_exchange_ledger_kg: np.ndarray
    bed_exchange_ledger_kg: np.ndarray
    wetdry_water_ledger_kg: np.ndarray
    wetdry_bed_ledger_kg: np.ndarray
    reference_water_mass_kg: np.ndarray
    reference_bed_mass_kg: np.ndarray
    accepted_updates: int
    time_s: float
    state_sha256: str


def _state_token(value: SuspendedLifecycleStateV76) -> str:
    return _fingerprint({
        "execution_fingerprint": value.execution_fingerprint,
        "accepted_updates": value.accepted_updates,
        "time_s": value.time_s,
        "arrays_sha256": _array_digest(
            value.suspended_mass_kg_m2, value.bed_mass_kg_m2,
            value.boundary_last_outflow_concentration_kg_m3,
            value.boundary_return_clock_s,
            value.advective_boundary_ledger_kg,
            value.diffusive_boundary_ledger_kg,
            value.water_exchange_ledger_kg, value.bed_exchange_ledger_kg,
            value.wetdry_water_ledger_kg, value.wetdry_bed_ledger_kg,
            value.reference_water_mass_kg, value.reference_bed_mass_kg,
        ),
    })


def _ledger_residuals(
    execution: SuspendedLifecycleExecutionV76,
    state: SuspendedLifecycleStateV76,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    area = execution.cell_area_m2[None, :]
    water = np.sum(state.suspended_mass_kg_m2 * area, axis=1)
    bed = np.sum(state.bed_mass_kg_m2 * area, axis=1)
    boundary = np.sum(
        state.advective_boundary_ledger_kg +
        state.diffusive_boundary_ledger_kg, axis=1)
    water_expected = (
        state.reference_water_mass_kg + boundary +
        state.water_exchange_ledger_kg + state.wetdry_water_ledger_kg)
    bed_expected = (
        state.reference_bed_mass_kg + state.bed_exchange_ledger_kg +
        state.wetdry_bed_ledger_kg)
    return water - water_expected, bed - bed_expected, (
        water + bed - state.reference_water_mass_kg -
        state.reference_bed_mass_kg - boundary)


def _assert_ledger(execution: SuspendedLifecycleExecutionV76,
                   state: SuspendedLifecycleStateV76) -> None:
    residuals = _ledger_residuals(execution, state)
    scale = np.maximum(1.0, state.reference_water_mass_kg +
                       state.reference_bed_mass_kg +
                       np.sum(np.abs(state.advective_boundary_ledger_kg) +
                              np.abs(state.diffusive_boundary_ledger_kg), axis=1))
    tolerance = 32768.0 * np.finfo(np.float64).eps * scale
    if any(np.any(np.abs(value) > tolerance) for value in residuals):
        raise SuspendedLifecycleV76Error(
            "v76 water/bed/reference mass ledger did not close")
    if (np.any(np.abs(state.water_exchange_ledger_kg +
                      state.bed_exchange_ledger_kg) > tolerance) or
            np.any(np.abs(state.wetdry_water_ledger_kg +
                          state.wetdry_bed_ledger_kg) > tolerance)):
        raise SuspendedLifecycleV76Error(
            "v76 internal phase-transfer ledger did not close")


def resolve_suspended_lifecycle_state_v76(
    execution: SuspendedLifecycleExecutionV76, *,
    suspended_mass_kg_m2: object, bed_mass_kg_m2: object,
    boundary_last_outflow_concentration_kg_m3: object | None = None,
    boundary_return_clock_s: object | None = None,
    advective_boundary_ledger_kg: object | None = None,
    diffusive_boundary_ledger_kg: object | None = None,
    water_exchange_ledger_kg: object | None = None,
    bed_exchange_ledger_kg: object | None = None,
    wetdry_water_ledger_kg: object | None = None,
    wetdry_bed_ledger_kg: object | None = None,
    reference_water_mass_kg: object | None = None,
    reference_bed_mass_kg: object | None = None,
    accepted_updates: int = 0, time_s: float = 0.0,
) -> SuspendedLifecycleStateV76:
    execution_document_v76(execution)
    ng, nc, nf, nb = (execution.fraction_count, execution.cell_count,
                      execution.face_count, execution.boundary_count)
    shape = (ng, nc)
    suspended = _array(suspended_mass_kg_m2, shape, "suspended mass",
                       nonnegative=True)
    bed = _array(bed_mass_kg_m2, shape, "bed mass", nonnegative=True)

    def optional(value: object | None, shape_: tuple[int, ...], name: str,
                 *, nonnegative: bool = False) -> np.ndarray:
        return _array(np.zeros(shape_, dtype=np.float64) if value is None else value,
                      shape_, name, nonnegative=nonnegative)

    last = optional(boundary_last_outflow_concentration_kg_m3, (ng, nf),
                    "boundary last outflow", nonnegative=True)
    clock = optional(boundary_return_clock_s, (ng, nf),
                     "boundary return clock", nonnegative=True)
    adv = optional(advective_boundary_ledger_kg, (ng, nb),
                   "advective boundary ledger")
    diff = optional(diffusive_boundary_ledger_kg, (ng, nb),
                    "diffusive boundary ledger")
    wx = optional(water_exchange_ledger_kg, (ng,), "water exchange ledger")
    bx = optional(bed_exchange_ledger_kg, (ng,), "bed exchange ledger")
    wd_w = optional(wetdry_water_ledger_kg, (ng,), "wet/dry water ledger")
    wd_b = optional(wetdry_bed_ledger_kg, (ng,), "wet/dry bed ledger")
    area = execution.cell_area_m2[None, :]
    reference_water = _array(
        np.sum(suspended * area, axis=1) if reference_water_mass_kg is None
        else reference_water_mass_kg, (ng,), "reference water mass",
        nonnegative=True)
    reference_bed = _array(
        np.sum(bed * area, axis=1) if reference_bed_mass_kg is None
        else reference_bed_mass_kg, (ng,), "reference bed mass",
        nonnegative=True)
    if (isinstance(accepted_updates, bool) or type(accepted_updates) is not int or
            not 0 <= accepted_updates <= MAX_ACCEPTED_UPDATES):
        raise SuspendedLifecycleV76Error(
            "v76 accepted update count is outside its bounded scope")
    time = _finite(time_s, "accepted time", nonnegative=True)
    if (accepted_updates == 0) != (time == 0.0):
        raise SuspendedLifecycleV76Error(
            "v76 accepted clock and update count differ")
    # Memory exists only on open faces, and positive clocks only where the
    # authenticated return-time mode is positive.
    open_face = execution.boundary_index >= 0
    if (np.any(last[:, ~open_face] != 0.0) or
            np.any(clock[:, ~open_face] != 0.0)):
        raise SuspendedLifecycleV76Error(
            "v76 boundary memory exists on a non-open face")
    expanded_return = np.zeros((ng, nf), dtype=np.float64)
    if nb:
        expanded_return[:, open_face] = execution.boundary_return_time_s[
            :, execution.boundary_index[open_face]]
    tolerance = 256.0*np.finfo(float).eps*np.maximum(1.0, expanded_return)
    if (np.any(clock[expanded_return <= 0.0] != 0.0) or
            np.any(clock > expanded_return + tolerance)):
        raise SuspendedLifecycleV76Error(
            "v76 boundary return clock is outside its authenticated range")
    placeholder = SuspendedLifecycleStateV76(
        execution.fingerprint, suspended, bed, last, clock, adv, diff, wx, bx,
        wd_w, wd_b, reference_water, reference_bed, accepted_updates, time, "")
    result = SuspendedLifecycleStateV76(
        **{**placeholder.__dict__, "state_sha256": _state_token(placeholder)})
    _assert_ledger(execution, result)
    return result


def validate_suspended_lifecycle_state_v76(
    state: SuspendedLifecycleStateV76,
    execution: SuspendedLifecycleExecutionV76,
) -> SuspendedLifecycleStateV76:
    execution_document_v76(execution)
    if (not isinstance(state, SuspendedLifecycleStateV76) or
            state.execution_fingerprint != execution.fingerprint or
            state.state_sha256 != _state_token(state)):
        raise SuspendedLifecycleV76Error(
            "v76 state is stale, mutated, or execution-mismatched")
    _assert_ledger(execution, state)
    return state


@dataclass(frozen=True)
class SuspendedLifecycleTrialV76:
    execution_fingerprint: str
    parent_state_sha256: str
    candidate: SuspendedLifecycleStateV76
    boundary_concentration_kg_m3: np.ndarray
    advective_face_flux_kg_m_s: np.ndarray
    diffusive_face_mass_owner_out_kg: np.ndarray
    settling_velocity_m_s: np.ndarray
    equilibrium_concentration_kg_m3: np.ndarray
    transferred_to_suspension_kg_m2: np.ndarray
    wetdry_transfer_to_bed_kg_m2: np.ndarray
    water_ledger_residual_kg: np.ndarray
    bed_ledger_residual_kg: np.ndarray
    combined_ledger_residual_kg: np.ndarray
    observed_parallel_workers: int
    backend: str
    trial_sha256: str


def _trial_token(value: SuspendedLifecycleTrialV76) -> str:
    return _fingerprint({
        "execution_fingerprint": value.execution_fingerprint,
        "parent_state_sha256": value.parent_state_sha256,
        "candidate_state_sha256": value.candidate.state_sha256,
        "backend": value.backend,
        "observed_parallel_workers": value.observed_parallel_workers,
        "arrays_sha256": _array_digest(
            value.boundary_concentration_kg_m3,
            value.advective_face_flux_kg_m_s,
            value.diffusive_face_mass_owner_out_kg,
            value.settling_velocity_m_s,
            value.equilibrium_concentration_kg_m3,
            value.transferred_to_suspension_kg_m2,
            value.wetdry_transfer_to_bed_kg_m2,
            value.water_ledger_residual_kg,
            value.bed_ledger_residual_kg,
            value.combined_ledger_residual_kg,
        ),
    })


def _smooth_bound(
    execution: SuspendedLifecycleExecutionV76, concentration: np.ndarray,
    hydro_constant: np.ndarray, grain: int, cell: int, lower: float,
    upper: float,
) -> tuple[float, float]:
    """CPU transcription of the existing fixed-grid CUDA/native extension."""
    rectangle = execution.rectangle_width_m
    if hydro_constant[cell] or rectangle[0, cell] <= 0.0 or rectangle[1, cell] <= 0.0:
        return lower, upper
    begin, end = execution.cell_face_offset[cell:cell + 2]
    incident = execution.cell_faces[begin:end]
    if np.any(execution.boundary_index[incident] >= 0):
        return lower, upper
    lo, hi = lower, upper
    scale = max(abs(lower), abs(upper), np.finfo(float).tiny)
    tolerance = 4096.0*np.finfo(float).eps*scale
    extremum_kind = 1
    for axis in range(2):
        ids = None
        for face in incident:
            if abs(execution.normal[axis, face]) != 1.0:
                continue
            side = 0 if cell == execution.owner[face] else 1
            if execution.weno_cells[0, side, face] >= 0:
                ids = execution.weno_cells[:, side, face]
        if ids is None:
            for face in incident:
                if abs(execution.normal[axis, face]) != 1.0:
                    continue
                neighbour = (execution.neighbour[face] if
                             cell == execution.owner[face] else
                             execution.owner[face])
                if neighbour < 0:
                    continue
                if (hydro_constant[neighbour] or
                        abs(concentration[grain, neighbour] -
                            concentration[grain, cell]) > tolerance):
                    return lower, upper
            continue
        if np.any(ids < 0) or np.any(hydro_constant[ids]):
            return lower, upper
        values = concentration[grain, ids]
        value_scale = np.max(np.abs(values))
        if value_scale == 0.0:
            continue
        z = values/value_scale
        if np.max(z)-np.min(z) <= 4096.0*np.finfo(float).eps:
            continue
        if (z[2]-z[1])*(z[3]-z[2]) > 0.0:
            return lower, upper
        d20, d21, d22 = (z[0]-2*z[1]+z[2], z[1]-2*z[2]+z[3],
                         z[2]-2*z[3]+z[4])
        amin, amax = min(abs(d20), abs(d21), abs(d22)), max(
            abs(d20), abs(d21), abs(d22))
        same_sign = ((d20 > 0 and d21 > 0 and d22 > 0) or
                     (d20 < 0 and d21 < 0 and d22 < 0))
        if (amin <= 4096.0*np.finfo(float).eps or not same_sign or
                amin < 0.5*amax):
            return lower, upper
        slope = 0.5*(z[3]-z[1])
        vertex = max(-0.5, min(0.5, -slope/d21))
        point = z[2]+slope*vertex+0.5*d21*vertex*vertex-d21/24.0
        kind = 2 if d21 < 0.0 else 3
        candidate = (value_scale*(z[2]+2*max(point-z[2], 0.0)) if kind == 2
                     else max(0.0, value_scale*(z[2]-2*max(z[2]-point, 0.0))))
        if extremum_kind != 1 and kind != extremum_kind:
            return lower, upper
        extremum_kind = kind
        if kind == 2:
            hi = max(hi, candidate)
        else:
            lo = min(lo, candidate)
    return max(0.0, lo), hi


def _cpu_fct(
    execution: SuspendedLifecycleExecutionV76, mass: np.ndarray,
    depth: np.ndarray, next_depth: np.ndarray, water_flux: np.ndarray,
    left: np.ndarray, right: np.ndarray, boundary: np.ndarray,
    hydro_constant: np.ndarray, dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ng, nc, nf, nb = (execution.fraction_count, execution.cell_count,
                      execution.face_count, execution.boundary_count)
    concentration = np.where(
        depth[None, :] > execution.dry_tolerance_m,
        mass/np.where(depth[None, :] > execution.dry_tolerance_m,
                      depth[None, :], 1.0), 0.0)
    low_flux = np.zeros((ng, nf), dtype=np.float64)
    anti = np.zeros_like(low_flux)
    for face in range(nf):
        c, n, b, w = (execution.owner[face], execution.neighbour[face],
                      execution.boundary_index[face], water_flux[face])
        if n < 0:
            if b < 0:
                if w != 0.0:
                    raise SuspendedLifecycleV76Error(
                        "v76 closed face has nonzero accepted water flux")
                continue
            low_flux[:, face] = (w*boundary[:, face] if w < 0.0 else
                                 w*concentration[:, c])
        elif w >= 0.0:
            low_flux[:, face] = w*concentration[:, c]
            anti[:, face] = w*left[:, face]-low_flux[:, face]
        else:
            low_flux[:, face] = w*concentration[:, n]
            anti[:, face] = w*right[:, face]-low_flux[:, face]
    low_state = np.array(mass, copy=True)
    rplus = np.ones_like(mass)
    rminus = np.ones_like(mass)
    for grain in range(ng):
        for cell in range(nc):
            low, lower, upper, pp, pm = (mass[grain, cell],
                                         concentration[grain, cell],
                                         concentration[grain, cell], 0.0, 0.0)
            for slot in range(execution.cell_face_offset[cell],
                              execution.cell_face_offset[cell + 1]):
                face = execution.cell_faces[slot]
                sign = execution.cell_face_sign[slot]
                adjacent = (execution.neighbour[face] if sign > 0 else
                            execution.owner[face])
                if adjacent >= 0:
                    value = concentration[grain, adjacent]
                    lower, upper = min(lower, value), max(upper, value)
                elif execution.boundary_index[face] >= 0 and water_flux[face] < 0.0:
                    value = boundary[grain, face]
                    lower, upper = min(lower, value), max(upper, value)
                z = -dt*sign*execution.face_length_m[face]/execution.cell_area_m2[cell]
                correction = z*anti[grain, face]
                low += z*low_flux[grain, face]
                pp += max(correction, 0.0)
                pm += min(correction, 0.0)
            scale = max(1.0, abs(mass[grain, cell]))
            tolerance = 4096.0*np.finfo(float).eps*scale
            if low < -tolerance or not math.isfinite(low):
                raise SuspendedLifecycleV76Error(
                    "v76 accepted water flux overdraws low-order suspended mass")
            low = max(0.0, low)
            lower, upper = _smooth_bound(
                execution, concentration, hydro_constant, grain, cell,
                lower, upper)
            low_state[grain, cell] = low
            if pp > 0.0:
                rplus[grain, cell] = min(
                    1.0, max(0.0, upper*next_depth[cell]-low-
                             64.0*np.finfo(float).tiny)/pp)
                rplus[grain, cell] *= 1.0-32.0*np.finfo(float).eps
            if pm < 0.0:
                rminus[grain, cell] = min(
                    1.0, max(0.0, low-lower*next_depth[cell]-
                             64.0*np.finfo(float).tiny)/(-pm))
                rminus[grain, cell] *= 1.0-32.0*np.finfo(float).eps
    scaled = np.zeros_like(anti)
    for grain in range(ng):
        for face in range(nf):
            c, n, correction = (execution.owner[face],
                                execution.neighbour[face], anti[grain, face])
            if correction > 0.0:
                factor = rminus[grain, c]
                if n >= 0:
                    factor = min(factor, rplus[grain, n])
            elif correction < 0.0:
                factor = rplus[grain, c]
                if n >= 0:
                    factor = min(factor, rminus[grain, n])
            else:
                factor = 1.0
            scaled[grain, face] = factor*correction
    flux = low_flux+scaled
    result = np.array(low_state, copy=True)
    for cell in range(nc):
        for slot in range(execution.cell_face_offset[cell],
                          execution.cell_face_offset[cell + 1]):
            face = execution.cell_faces[slot]
            result[:, cell] -= (
                dt*execution.cell_face_sign[slot]*execution.face_length_m[face]*
                scaled[:, face]/execution.cell_area_m2[cell])
    tolerance = 4096.0*np.finfo(float).eps*np.maximum(1.0, np.abs(mass))
    if np.any(result < -tolerance) or not np.isfinite(result).all():
        raise SuspendedLifecycleV76Error("v76 FCT result is invalid")
    result = np.maximum(0.0, result)
    ledger = np.zeros((ng, nb), dtype=np.float64)
    for face in range(nf):
        b = execution.boundary_index[face]
        if b >= 0:
            ledger[:, b] -= dt*execution.face_length_m[face]*flux[:, face]
    return result, flux, ledger


def _cpu_diffusion(
    execution: SuspendedLifecycleExecutionV76, mass: np.ndarray,
    depth: np.ndarray, boundary: np.ndarray, exchange_area: np.ndarray,
    face_active: np.ndarray, dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ng, nc, nf, nb = (execution.fraction_count, execution.cell_count,
                      execution.face_count, execution.boundary_count)
    face_mass = np.zeros((ng, nf), dtype=np.float64)
    dicuv = execution.profile.horizontal_diffusivity_m2_s
    for face in range(nf):
        c, n, b = (execution.owner[face], execution.neighbour[face],
                   execution.boundary_index[face])
        if n < 0 and b < 0:
            if face_active[face]:
                raise SuspendedLifecycleV76Error(
                    "v76 closed face cannot be diffusion-active")
            continue
        hl, hr = depth[c], depth[n] if n >= 0 else depth[c]
        cl = mass[:, c]/hl if hl > 0.0 else np.zeros(ng)
        cr = (mass[:, n]/hr if n >= 0 and hr > 0.0 else
              (np.zeros(ng) if n >= 0 else boundary[:, face]))
        oracle = flow2dh_horizontal_diffusive_face_mass(
            cl, cr, dicuv, dicuv, exchange_area[face],
            execution.centre_distance_m[face], hl, hr, dt_s=dt,
            dry_threshold_m=execution.profile.diffusion_dry_threshold_m,
            transport_face_active=bool(face_active[face]))
        face_mass[:, face] = oracle.mass_left_to_right_kg
    theta = np.ones((ng, nc), dtype=np.float64)
    margin = 1.0-32.0*np.finfo(float).eps
    for grain in range(ng):
        for cell in range(nc):
            outgoing = 0.0
            for slot in range(execution.cell_face_offset[cell],
                              execution.cell_face_offset[cell + 1]):
                face = execution.cell_faces[slot]
                outgoing += max(
                    execution.cell_face_sign[slot]*face_mass[grain, face], 0.0)
            inventory = mass[grain, cell]*execution.cell_area_m2[cell]
            if outgoing > 0.0 and outgoing >= inventory:
                theta[grain, cell] = min(1.0, inventory/outgoing)*margin
    for grain in range(ng):
        for face in range(nf):
            value = face_mass[grain, face]
            donor = (execution.owner[face] if value > 0.0 else
                     execution.neighbour[face] if value < 0.0 else -1)
            if donor >= 0:
                face_mass[grain, face] *= theta[grain, donor]
    result = np.array(mass, copy=True)
    for cell in range(nc):
        for slot in range(execution.cell_face_offset[cell],
                          execution.cell_face_offset[cell + 1]):
            face = execution.cell_faces[slot]
            result[:, cell] -= (
                execution.cell_face_sign[slot]*face_mass[:, face]/
                execution.cell_area_m2[cell])
    tolerance = 4096.0*np.finfo(float).eps*np.maximum(1.0, np.abs(mass))
    if np.any(result < -tolerance) or not np.isfinite(result).all():
        raise SuspendedLifecycleV76Error("v76 diffusion result is invalid")
    result = np.maximum(0.0, result)
    ledger = np.zeros((ng, nb), dtype=np.float64)
    for face in range(nf):
        b = execution.boundary_index[face]
        if b >= 0:
            ledger[:, b] -= face_mass[:, face]
    return result, face_mass, ledger


def _freeze_trial_array(value: object) -> np.ndarray:
    result = np.ascontiguousarray(value)
    result.setflags(write=False)
    return result


def begin_suspended_lifecycle_trial_v76(
    execution: SuspendedLifecycleExecutionV76,
    parent: SuspendedLifecycleStateV76, *,
    depth_before_m: object, depth_after_m: object,
    accepted_water_flux_m2_s: object,
    reconstructed_left_concentration_kg_m3: object,
    reconstructed_right_concentration_kg_m3: object,
    prescribed_boundary_concentration_kg_m3: object,
    accepted_face_exchange_area_m2: object,
    transport_face_active: object,
    endpoint_momentum_m2_s: object,
    shear_velocity_m_s: object,
    suspended_transport_capacity_kg_m_s: object,
    surface_fraction: object,
    hydro_constant: object,
    dt_s: float,
    cancelled: bool = False,
) -> SuspendedLifecycleTrialV76:
    """Build one complete CPU candidate without mutating accepted state."""
    if execution.backend != BACKEND_CPU:
        raise SuspendedLifecycleV76Error(
            "v76 CPU trial requires an explicitly selected CPU execution")
    validate_suspended_lifecycle_state_v76(parent, execution)
    if cancelled:
        raise SuspendedLifecycleV76Error(
            "v76 update cancelled before candidate publication")
    if parent.accepted_updates >= MAX_ACCEPTED_UPDATES:
        raise SuspendedLifecycleV76Error("v76 accepted update limit was reached")
    dt = _finite(dt_s, "timestep", positive=True)
    ng, nc, nf, nb = (execution.fraction_count, execution.cell_count,
                      execution.face_count, execution.boundary_count)
    h0 = _array(depth_before_m, (nc,), "depth before", nonnegative=True)
    h1 = _array(depth_after_m, (nc,), "depth after", nonnegative=True)
    water = _array(accepted_water_flux_m2_s, (nf,), "accepted water flux")
    left = _array(reconstructed_left_concentration_kg_m3, (ng, nf),
                  "left reconstructed concentration")
    right = _array(reconstructed_right_concentration_kg_m3, (ng, nf),
                   "right reconstructed concentration")
    prescribed = _array(prescribed_boundary_concentration_kg_m3, (ng, nb),
                        "prescribed boundary concentration", nonnegative=True)
    face_area = _array(accepted_face_exchange_area_m2, (nf,),
                       "accepted face exchange area", nonnegative=True)
    face_active = _array(transport_face_active, (nf,),
                         "transport face active", dtype=np.bool_)
    momentum = _array(endpoint_momentum_m2_s, (2, nc), "endpoint momentum")
    shear = _array(shear_velocity_m_s, (nc,), "shear velocity", nonnegative=True)
    capacity = _array(suspended_transport_capacity_kg_m_s, (ng, nc),
                      "suspended transport capacity", nonnegative=True)
    fraction = _array(surface_fraction, (ng, nc), "surface fraction",
                      nonnegative=True)
    if np.any(fraction > 1.0):
        raise SuspendedLifecycleV76Error("v76 surface fraction exceeds one")
    constant = _array(hydro_constant, (nc,), "hydro constant", dtype=np.bool_)
    wet = h1 > execution.dry_tolerance_m
    speed = np.zeros(nc, dtype=np.float64)
    speed[wet] = np.hypot(momentum[0, wet], momentum[1, wet])/h1[wet]

    interior = np.where(
        h0[None, :] > execution.dry_tolerance_m,
        parent.suspended_mass_kg_m2/np.where(
            h0[None, :] > execution.dry_tolerance_m, h0[None, :], 1.0), 0.0)
    boundary = np.zeros((ng, nf), dtype=np.float64)
    next_last = np.array(parent.boundary_last_outflow_concentration_kg_m3,
                         copy=True)
    next_clock = np.array(parent.boundary_return_clock_s, copy=True)
    expected_advective = np.zeros((ng, nb), dtype=np.float64)
    for face in range(nf):
        b = execution.boundary_index[face]
        if b < 0:
            continue
        c = execution.owner[face]
        q_in = -water[face]*execution.face_length_m[face]
        result = resolve_flow2dh_open_boundary(
            prescribed[:, b], interior[:, c], np.full(ng, q_in), dt_s=dt,
            neumann=np.asarray(execution.profile.neumann_boundary_by_fraction,
                               dtype=bool),
            return_time_s=execution.boundary_return_time_s[:, b],
            previous_memory=Flow2DHBoundaryMemory(
                parent.boundary_last_outflow_concentration_kg_m3[:, face],
                parent.boundary_return_clock_s[:, face]))
        boundary[:, face] = result.boundary_concentration_kg_m3
        next_last[:, face] = result.memory.last_outflow_concentration_kg_m3
        next_clock[:, face] = result.memory.return_clock_s
        expected_advective[:, b] += result.advective_mass_in_kg

    advected, advective_flux, advective_ledger = _cpu_fct(
        execution, parent.suspended_mass_kg_m2, h0, h1, water, left, right,
        boundary, constant, dt)
    adv_scale = np.maximum(1.0, np.maximum(
        np.abs(expected_advective), np.abs(advective_ledger)))
    if np.any(np.abs(expected_advective-advective_ledger) >
              8192.0*np.finfo(float).eps*adv_scale):
        raise SuspendedLifecycleV76Error(
            "v76 FCT boundary ledger differs from resolved per-face boundary")
    diffused, diffusive_face_mass, diffusive_ledger = _cpu_diffusion(
        execution, advected, h1, boundary, face_area, face_active, dt)

    concentration = np.zeros((nc, ng), dtype=np.float64)
    concentration[wet] = (diffused[:, wet]/h1[wet]).T
    settling = settlefrm2_noncohesive(
        concentration, execution.profile.sediment_d50_m,
        execution.profile.settling_fac_dss,
        execution.profile.sediment_density_kg_m3,
        execution.profile.water_density_kg_m3,
        execution.profile.temperature_value_c,
        settling_reference_concentration_kg_m3=
            execution.profile.settling_reference_concentration_kg_m3,
        gravity_m_s2=execution.profile.gravity_m_s2,
        wet_mask=wet,
    )
    exchange = advance_flow2dh_equilibrium_exchange(
        diffused, parent.bed_mass_kg_m2, capacity, fraction, speed, shear,
        h0, h1, settling.settling_velocity_m_s.T,
        execution.cell_area_m2, dt_s=dt,
        fac_tsd=execution.profile.fac_tsd,
        sediment_threshold_m=execution.profile.sediment_threshold_m,
        minimum_reference_depth_m=execution.profile.reference_depth_floor_m,
        flow_wet_mask=wet,
    )
    cleanup = np.where(wet[None, :], 0.0,
                       exchange.suspended_mass_kg_m2)
    final_suspended = exchange.suspended_mass_kg_m2-cleanup
    final_bed = exchange.bed_mass_kg_m2+cleanup
    cleanup_water = -np.sum(cleanup*execution.cell_area_m2[None, :], axis=1)
    cleanup_bed = -cleanup_water

    candidate = resolve_suspended_lifecycle_state_v76(
        execution, suspended_mass_kg_m2=final_suspended,
        bed_mass_kg_m2=final_bed,
        boundary_last_outflow_concentration_kg_m3=next_last,
        boundary_return_clock_s=next_clock,
        advective_boundary_ledger_kg=(
            parent.advective_boundary_ledger_kg+advective_ledger),
        diffusive_boundary_ledger_kg=(
            parent.diffusive_boundary_ledger_kg+diffusive_ledger),
        water_exchange_ledger_kg=(
            parent.water_exchange_ledger_kg+
            exchange.suspended_exchange_ledger_kg),
        bed_exchange_ledger_kg=(
            parent.bed_exchange_ledger_kg+exchange.bed_exchange_ledger_kg),
        wetdry_water_ledger_kg=parent.wetdry_water_ledger_kg+cleanup_water,
        wetdry_bed_ledger_kg=parent.wetdry_bed_ledger_kg+cleanup_bed,
        reference_water_mass_kg=parent.reference_water_mass_kg,
        reference_bed_mass_kg=parent.reference_bed_mass_kg,
        accepted_updates=parent.accepted_updates+1,
        time_s=parent.time_s+dt,
    )
    residuals = _ledger_residuals(execution, candidate)
    arrays = tuple(_freeze_trial_array(value) for value in (
        boundary, advective_flux, diffusive_face_mass,
        settling.settling_velocity_m_s.T,
        exchange.equilibrium_concentration_kg_m3,
        exchange.transferred_to_suspension_kg_m2, cleanup,
        *residuals,
    ))
    placeholder = SuspendedLifecycleTrialV76(
        execution.fingerprint, parent.state_sha256, candidate, *arrays,
        1, BACKEND_CPU, "")
    return SuspendedLifecycleTrialV76(
        **{**placeholder.__dict__, "trial_sha256": _trial_token(placeholder)})


def _validate_trial(
    execution: SuspendedLifecycleExecutionV76,
    parent: SuspendedLifecycleStateV76,
    trial: SuspendedLifecycleTrialV76,
) -> None:
    validate_suspended_lifecycle_state_v76(parent, execution)
    if (not isinstance(trial, SuspendedLifecycleTrialV76) or
            trial.execution_fingerprint != execution.fingerprint or
            trial.parent_state_sha256 != parent.state_sha256 or
            trial.trial_sha256 != _trial_token(trial)):
        raise SuspendedLifecycleV76Error(
            "v76 trial is stale, mutated, or has a different accepted parent")
    validate_suspended_lifecycle_state_v76(trial.candidate, execution)


def commit_suspended_lifecycle_trial_v76(
    execution: SuspendedLifecycleExecutionV76,
    parent: SuspendedLifecycleStateV76,
    trial: SuspendedLifecycleTrialV76,
) -> SuspendedLifecycleStateV76:
    _validate_trial(execution, parent, trial)
    return trial.candidate


def rollback_suspended_lifecycle_trial_v76(
    execution: SuspendedLifecycleExecutionV76,
    parent: SuspendedLifecycleStateV76,
    trial: SuspendedLifecycleTrialV76,
) -> SuspendedLifecycleStateV76:
    _validate_trial(execution, parent, trial)
    return parent


def cancel_suspended_lifecycle_trial_v76(
    execution: SuspendedLifecycleExecutionV76,
    parent: SuspendedLifecycleStateV76,
    trial: SuspendedLifecycleTrialV76,
) -> SuspendedLifecycleStateV76:
    return rollback_suspended_lifecycle_trial_v76(execution, parent, trial)


def suspended_lifecycle_checkpoint_v76(
    execution: SuspendedLifecycleExecutionV76,
    state: SuspendedLifecycleStateV76,
) -> dict:
    validate_suspended_lifecycle_state_v76(state, execution)
    body = {
        "schema": SCHEMA,
        "scope_id": CHECKPOINT_SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "execution_fingerprint": execution.fingerprint,
        "accepted_updates": state.accepted_updates,
        "time_s": state.time_s,
        "transaction_schema": "accepted_state_no_trial_arrays_v1",
        "state": {
            name: getattr(state, name).tolist() for name in (
                "suspended_mass_kg_m2", "bed_mass_kg_m2",
                "boundary_last_outflow_concentration_kg_m3",
                "boundary_return_clock_s", "advective_boundary_ledger_kg",
                "diffusive_boundary_ledger_kg", "water_exchange_ledger_kg",
                "bed_exchange_ledger_kg", "wetdry_water_ledger_kg",
                "wetdry_bed_ledger_kg", "reference_water_mass_kg",
                "reference_bed_mass_kg",
            )
        },
        "state_sha256": state.state_sha256,
    }
    return {**body, "fingerprint": _fingerprint(body)}


def load_suspended_lifecycle_checkpoint_v76(
    execution: SuspendedLifecycleExecutionV76,
    document: Mapping[str, object],
) -> SuspendedLifecycleStateV76:
    expected = {
        "schema", "scope_id", "semantic_id", "execution_fingerprint",
        "accepted_updates", "time_s", "transaction_schema", "state",
        "state_sha256", "fingerprint",
    }
    if not isinstance(document, Mapping) or set(document) != expected:
        raise SuspendedLifecycleV76Error("v76 checkpoint inventory differs")
    body = {key: value for key, value in document.items()
            if key != "fingerprint"}
    if (document["fingerprint"] != _fingerprint(body) or
            document["schema"] != SCHEMA or
            document["scope_id"] != CHECKPOINT_SCOPE_ID or
            document["semantic_id"] != SEMANTIC_ID or
            document["execution_fingerprint"] != execution.fingerprint or
            document["transaction_schema"] !=
            "accepted_state_no_trial_arrays_v1"):
        raise SuspendedLifecycleV76Error(
            "v76 checkpoint identity or transaction schema differs")
    fields = document["state"]
    names = {
        "suspended_mass_kg_m2", "bed_mass_kg_m2",
        "boundary_last_outflow_concentration_kg_m3",
        "boundary_return_clock_s", "advective_boundary_ledger_kg",
        "diffusive_boundary_ledger_kg", "water_exchange_ledger_kg",
        "bed_exchange_ledger_kg", "wetdry_water_ledger_kg",
        "wetdry_bed_ledger_kg", "reference_water_mass_kg",
        "reference_bed_mass_kg",
    }
    if not isinstance(fields, Mapping) or set(fields) != names:
        raise SuspendedLifecycleV76Error("v76 checkpoint state inventory differs")
    state = resolve_suspended_lifecycle_state_v76(
        execution, **fields, accepted_updates=document["accepted_updates"],
        time_s=document["time_s"])
    if state.state_sha256 != document["state_sha256"]:
        raise SuspendedLifecycleV76Error("v76 checkpoint state digest differs")
    return state


def method_identity_v76() -> dict:
    """Return the explicit physical and numerical provenance split."""
    return {
        "scope_id": SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "accepted_update_limit": MAX_ACCEPTED_UPDATES,
        "advection_method": ADVECTION_METHOD_ID,
        "settling_method": FLOW2DH_SETTLING_ID,
        "settling_semantic_id": FLOW2DH_SETTLING_SEMANTIC_ID,
        "settling_form": FLOW2DH_SETTLING_FORM,
        "exchange_method": FLOW2DH_EXCHANGE_ID,
        "exchange_semantic_id": FLOW2DH_EXCHANGE_SEMANTIC_ID,
        "boundary_method": FLOW2DH_OPEN_BOUNDARY_ID,
        "boundary_semantic_id": FLOW2DH_OPEN_BOUNDARY_SEMANTIC_ID,
        "diffusion_method": FLOW2DH_HORIZONTAL_DIFFUSION_ID,
        "diffusion_semantic_id": FLOW2DH_HORIZONTAL_DIFFUSION_SEMANTIC_ID,
        "braids_numerical_method": NUMERICAL_METHOD,
        "delft3d_reference_numerical_method": REFERENCE_NUMERICAL_METHOD,
        "numerical_equivalence_claimed": False,
        "wet_dry_cleanup": WET_DRY_METHOD,
        "transaction": "immutable_candidate_commit_rollback_cancel_restart_v1",
        "mass_ledger": "water_bed_boundary_per_class_reference_mass_v1",
        "locked_delft_sources_read_only": list(LOCKED_SOURCE_ANCHORS),
    }


def verify_sources_v76(root: str | Path) -> dict[str, str]:
    """Hash additive implementation dependencies without writing them."""
    root = Path(root)
    paths = (
        "braids_core/sediment/suspended_lifecycle_live_v76.py",
        "braids_gpu/suspended_lifecycle_live_v76.py",
        "braids_core/sediment/flow2dh_settling.py",
        "braids_core/sediment/flow2dh_exchange.py",
        "braids_core/sediment/flow2dh_boundary_diffusion.py",
        "braids_core/sediment/suspension_profile.py",
        "braids_gpu/law11_soursin_exchange.cu",
        "braids_gpu/flow2dh_boundary_diffusion.cu",
        "validation/gpu_fwave/suspended_advection.cu",
    )
    result = {}
    for relative in paths:
        path = root/relative
        if not path.is_file():
            raise SuspendedLifecycleV76Error(
                "v76 source dependency is absent: " + relative)
        result[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def verify_locked_delft_sources_v76(root: str | Path) -> dict[str, object]:
    """Verify every read-only Delft3D anchor against ``source.lock.json``."""
    root = Path(root)
    lock_path = root/"validation/sediment/delft3d_current/source.lock.json"
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise SuspendedLifecycleV76Error(
            "v76 Delft3D source lock is unavailable") from error
    if (lock.get("schema_version") != 1 or
            lock.get("source_root") != "resources/Delft3D-main" or
            not isinstance(lock.get("files"), Mapping)):
        raise SuspendedLifecycleV76Error(
            "v76 Delft3D source lock schema differs")
    hashes = {}
    prefix = "resources/Delft3D-main/"
    for anchored in LOCKED_SOURCE_ANCHORS:
        if not anchored.startswith(prefix):
            raise SuspendedLifecycleV76Error(
                "v76 Delft3D anchor is outside the locked source root")
        relative = anchored[len(prefix):]
        expected = lock["files"].get(relative)
        path = root/anchored
        if not isinstance(expected, str) or not path.is_file():
            raise SuspendedLifecycleV76Error(
                "v76 locked source anchor is absent: " + relative)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise SuspendedLifecycleV76Error(
                "v76 locked source hash differs: " + relative)
        hashes[relative] = actual
    upstream = lock.get("upstream", {})
    return {
        "lock": str(lock_path.relative_to(root)).replace("\\", "/"),
        "commit": upstream.get("commit"),
        "flow_version_in_source": upstream.get("flow_version_in_source"),
        "read_only": True,
        "files": hashes,
    }


__all__ = [
    "BACKENDS", "BACKEND_CPU", "BACKEND_CUDA", "CHECKPOINT_SCOPE_ID",
    "LOCKED_SOURCE_ANCHORS", "MAX_ACCEPTED_UPDATES", "NUMERICAL_METHOD",
    "REFERENCE_NUMERICAL_METHOD", "SCHEMA", "SCOPE_ID", "SEMANTIC_ID",
    "SuspendedLifecycleExecutionV76", "SuspendedLifecycleStateV76",
    "SuspendedLifecycleTrialV76", "SuspendedLifecycleV76Error",
    "begin_suspended_lifecycle_trial_v76",
    "cancel_suspended_lifecycle_trial_v76",
    "commit_suspended_lifecycle_trial_v76", "execution_document_v76",
    "load_suspended_lifecycle_checkpoint_v76", "method_identity_v76",
    "resolve_suspended_lifecycle_execution_v76",
    "resolve_suspended_lifecycle_state_v76",
    "rollback_suspended_lifecycle_trial_v76",
    "suspended_lifecycle_checkpoint_v76",
    "validate_suspended_lifecycle_state_v76", "verify_sources_v76",
    "verify_locked_delft_sources_v76",
]
