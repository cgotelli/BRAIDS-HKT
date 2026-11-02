"""Bounded Q4/v90 FLOW-2DH HLES closure producer.

This module closes the missing ``lpfluc -> protke -> detvic`` producer without
owning suspended transport, Exner, or hydraulic momentum.  The accepted
low-pass filter and raw HLES slots are immutable transaction state.  A trial
uses background plus the *parent* raw HLES slot, evaluates one accepted
endpoint, and publishes a candidate slot for the following BRAIDS update.

The target reference is classic Delft3D4 FLOW, sigma, KMAX=1.  ``Hturrt`` is
in minutes although the older v87 compatibility field is named ``reltim_s``;
the source exponent uses the documented half integration step
``hdt = 0.5*dt``.  Classic FLOW's staggered ``protke`` stencil is available as
an exact oracle below.  The bounded BRAIDS mapping is explicitly different:
face-normal fluctuations are reconstructed by weighted least squares and the
symmetric-strain invariant is evaluated with a finite-volume Green--Gauss
gradient on a uniform Cartesian fixed grid.  Boundary cells receive zero
production, matching the documented FLOW boundary assumption.  No stretched,
cut-cell, nonorthogonal, or arbitrary-AMR equivalence is claimed.
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

from .horizontal_turbulence_lifecycle_v87 import (
    HorizontalTurbulenceExecutionV87,
    evaluate_detvic_field_v87,
    execution_document_v87,
)
from .multifraction_suspended_lifecycle_v78 import (
    ROLE_CPU_ORACLE,
    ROLE_CUDA,
    validate_implicit_fwave_midpoint_step_receipt_v78,
)
from .single_owner_q4_v89 import (
    BACKEND_CPU,
    BACKEND_GPU,
    ClosureReceiptV89,
    SingleOwnerExecutionV89,
    SingleOwnerStateV89,
    make_closure_receipt_v89,
    validate_single_owner_state_v89,
)


SCHEMA = 1
SCOPE_ID = "braids-q4-flow2dh-hles-producer-v90-v1"
CHECKPOINT_SCOPE_ID = SCOPE_ID + ":checkpoint"
INPUT_RECEIPT_SCOPE_ID = SCOPE_ID + ":accepted-endpoint-input"
OUTPUT_RECEIPT_SCOPE_ID = SCOPE_ID + ":closure-output"
PRIME_RECEIPT_SCOPE_ID = SCOPE_ID + ":cold-prime-input"
CURRENT_HANDOFF_SCOPE_ID = SCOPE_ID + ":pre-hydraulic-current-handoff"
SEMANTIC_ID = hashlib.sha256(SCOPE_ID.encode("ascii")).hexdigest()
INTEGRATOR = "implicit_fwave_midpoint"
MAX_ACCEPTED_UPDATES = 4
MAPPING_ID = "braids_uniform_cartesian_fixed_grid_face_wls_green_gauss_v1"
STRUCTURED_ORACLE_ID = "locked_delft3d4_staggered_protke_v1"
FILTER_INTERVAL = "explicit_closure_interval_seconds"
PRODUCER_CADENCE = "one_post_accepted_endpoint_invocation_per_braids_update"
TARGET_FLOW_PATH = "classic_flow_sigma_kmax1"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class HlesProducerV90Error(ValueError):
    """A v90 execution, receipt, state, or transaction is invalid."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False).encode("ascii")
    except (TypeError, ValueError, OverflowError) as error:
        raise HlesProducerV90Error(
            "v90 value is not finite canonical JSON") from error


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _array_digest(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(_canonical(list(value.shape)))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise HlesProducerV90Error(f"v90 {name} must be a SHA-256 digest")
    return value


def _finite(value: object, name: str, *, positive: bool = False) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(float(value))):
        raise HlesProducerV90Error(f"v90 {name} must be finite")
    result = float(value)
    if positive and result <= 0.0:
        raise HlesProducerV90Error(f"v90 {name} must be positive")
    return result


def _array(value: object, shape: tuple[int, ...], name: str, *,
           dtype=np.float64, nonnegative: bool = False,
           binary: bool = False) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != shape or raw.dtype.kind not in "fiu b":
        raise HlesProducerV90Error(
            f"v90 {name} must be numeric with shape {shape}")
    if np.dtype(dtype).kind in "iu" and raw.dtype.kind not in "iub":
        raise HlesProducerV90Error(
            f"v90 {name} must contain exact integer values")
    if binary and np.any((raw != 0) & (raw != 1)):
        raise HlesProducerV90Error(f"v90 {name} must contain only 0/1")
    result = np.ascontiguousarray(raw, dtype=dtype)
    if not np.isfinite(result).all():
        raise HlesProducerV90Error(f"v90 {name} must be finite")
    if nonnegative and np.any(result < 0.0):
        raise HlesProducerV90Error(f"v90 {name} must be nonnegative")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class HlesProducerExecutionV90:
    closure_execution: HorizontalTurbulenceExecutionV87
    backend: str
    hturrt_min: float
    mapping_id: str
    target_flow_path: str
    semantic_id: str
    fingerprint: str

    @property
    def cell_count(self) -> int:
        return self.closure_execution.cell_count

    @property
    def face_count(self) -> int:
        return self.closure_execution.base_execution.face_count

    @property
    def relaxation_time_min(self) -> float:
        return self.hturrt_min


def _execution_payload(value: HlesProducerExecutionV90) -> dict:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "semantic_id": value.semantic_id,
        "closure_v87_execution": execution_document_v87(
            value.closure_execution),
        "backend": value.backend,
        "hturrt_min": value.hturrt_min,
        "integrator": INTEGRATOR,
        "mapping_id": value.mapping_id,
        "structured_oracle_id": STRUCTURED_ORACLE_ID,
        "target_flow_path": value.target_flow_path,
        "kmax": 1,
        "Hturrt_unit": "minutes",
        "v87_reltim_s_field_disposition":
            "compatibility_name_only_value_interpreted_as_Hturrt_minutes",
        "filter_interval": FILTER_INTERVAL,
        "producer_cadence": PRODUCER_CADENCE,
        "classic_flow_cadence": "two_invocations_per_ADI_full_step",
        "full_timestep_filter_trajectory_identity_claimed": False,
        "raw_dynamic_field_lag_accepted_updates": 1,
        "candidate_dicuv_same_update_scalar_consumer": False,
        "used_total_cap_owner": "v91_not_this_producer",
        "momentum_consumer": "v91_not_this_producer",
        "momentum_consumer_count": 0,
        "v91_required_hydraulic_echo": True,
        "scalar_consumer": "v89_v96_not_this_producer",
        "accepted_update_limit": MAX_ACCEPTED_UPDATES,
        "automatic_cpu_fallback": False,
        "product_admission": False,
        "general_amr_claimed": False,
        "mapping_admission": (
            "uniform_axis_aligned_cartesian_fixed_grid_only; cut cells, "
            "stretched/rotated grids and AMR fail closed until an "
            "authenticated geometry operator exists"),
    }


def resolve_hles_producer_execution_v90(
    closure_execution: HorizontalTurbulenceExecutionV87, *, backend: str,
    hturrt_min: float,
    mapping_id: str = MAPPING_ID,
    target_flow_path: str = TARGET_FLOW_PATH,
) -> HlesProducerExecutionV90:
    """Resolve the explicit CPU or GPU closure producer."""
    execution_document_v87(closure_execution)
    if backend not in (BACKEND_CPU, BACKEND_GPU):
        raise HlesProducerV90Error("v90 backend must be explicit")
    expected = ROLE_CPU_ORACLE if backend == BACKEND_CPU else ROLE_CUDA
    if closure_execution.role != expected:
        raise HlesProducerV90Error("v90 backend and v87 role differ")
    if mapping_id != MAPPING_ID:
        raise HlesProducerV90Error("v90 general-mesh mapping is unsupported")
    if target_flow_path != TARGET_FLOW_PATH:
        raise HlesProducerV90Error(
            "v90 supports only classic FLOW sigma KMAX=1; the z-model "
            "KMAX=1 disposition is a separate unsupported target")
    if not closure_execution.hles_enabled:
        raise HlesProducerV90Error(
            "v90 is the active HLES producer; disabled HLES belongs upstream")
    hturrt = _finite(hturrt_min, "Hturrt minutes")
    if not (hturrt > 0.0 or hturrt == -1.0):
        raise HlesProducerV90Error(
            "v90 Hturrt must be positive minutes or exactly -1")
    if hturrt != closure_execution.reltim_s:
        raise HlesProducerV90Error(
            "v90 explicit Hturrt minutes differs from the v87 compatibility "
            "value; rebuild the v87 identity rather than guessing units")
    base = closure_execution.base_execution.base_execution
    norm = np.hypot(base.normal[0], base.normal[1])
    if not np.allclose(norm, 1.0, rtol=0.0, atol=2.0e-12):
        raise HlesProducerV90Error("v90 face normals must be unit vectors")
    tol = 2.0e-12
    axis_aligned = ((np.abs(base.normal[0]) <= tol) |
                    (np.abs(base.normal[1]) <= tol))
    width = base.rectangle_width_m
    uniform_width = (
        np.allclose(width[0], width[0, 0], rtol=0.0, atol=tol) and
        np.allclose(width[1], width[1, 0], rtol=0.0, atol=tol))
    vertical = np.abs(base.normal[0]) > 1.0-tol
    expected_length = np.where(vertical, width[1, 0], width[0, 0])
    expected_distance = np.where(vertical, width[0, 0], width[1, 0])
    expected_distance = np.where(
        base.neighbour < 0, 0.5*expected_distance, expected_distance)
    local_cartesian = True
    cardinal = {(-1, 0), (1, 0), (0, -1), (0, 1)}
    for cell in range(base.cell_count):
        slots = range(base.cell_face_offset[cell],
                      base.cell_face_offset[cell+1])
        directions = {
            (int(round(base.cell_face_sign[slot]*base.normal[0, face])),
             int(round(base.cell_face_sign[slot]*base.normal[1, face])))
            for slot in slots for face in (int(base.cell_faces[slot]),)
        }
        if directions != cardinal or len(tuple(slots)) != 4:
            local_cartesian = False
            break
    if (not np.all(axis_aligned) or np.any(width <= 0.0) or
            not uniform_width or not local_cartesian or
            not np.allclose(width[0]*width[1], base.cell_area_m2,
                            rtol=0.0, atol=2.0e-12) or
            not np.allclose(base.face_length_m, expected_length,
                            rtol=0.0, atol=2.0e-12) or
            not np.allclose(base.centre_distance_m, expected_distance,
                            rtol=0.0, atol=2.0e-12)):
        raise HlesProducerV90Error(
            "v90 BRAIDS mapping admits authenticated uniform Cartesian "
            "fixed grids only")
    placeholder = HlesProducerExecutionV90(
        closure_execution, backend, hturrt, mapping_id, TARGET_FLOW_PATH,
        SEMANTIC_ID, "")
    return HlesProducerExecutionV90(
        closure_execution, backend, hturrt, mapping_id, TARGET_FLOW_PATH,
        SEMANTIC_ID, _fingerprint(_execution_payload(placeholder)))


def execution_document_v90(value: HlesProducerExecutionV90) -> dict:
    if not isinstance(value, HlesProducerExecutionV90):
        raise HlesProducerV90Error("v90 execution object differs")
    payload = _execution_payload(value)
    if value.fingerprint != _fingerprint(payload):
        raise HlesProducerV90Error("v90 execution fingerprint differs")
    return {**payload, "fingerprint": value.fingerprint}


@dataclass(frozen=True)
class HlesProducerStateV90:
    execution_fingerprint: str
    accepted_filtered_face_normal_velocity_m_s: np.ndarray
    accepted_dynamic_vicuv_m2_s: np.ndarray
    accepted_dynamic_dicuv_m2_s: np.ndarray
    primed: bool
    prime_receipt_sha256: str | None
    accepted_updates: int
    time_s: float
    accepted_receipt_chain_sha256: str
    state_sha256: str


def _state_token(value: HlesProducerStateV90) -> str:
    return _fingerprint({
        "execution_fingerprint": value.execution_fingerprint,
        "arrays_sha256": [_array_digest(array) for array in (
            value.accepted_filtered_face_normal_velocity_m_s,
            value.accepted_dynamic_vicuv_m2_s,
            value.accepted_dynamic_dicuv_m2_s)],
        "primed": value.primed,
        "prime_receipt_sha256": value.prime_receipt_sha256,
        "accepted_updates": value.accepted_updates,
        "time_s": value.time_s,
        "accepted_receipt_chain_sha256":
            value.accepted_receipt_chain_sha256,
    })


def _wrap_state(
    execution: HlesProducerExecutionV90, *, filtered: object,
    dynamic_vicuv: object, dynamic_dicuv: object, accepted_updates: int,
    time_s: float, receipt_chain: str, primed: bool,
    prime_receipt_sha256: str | None,
) -> HlesProducerStateV90:
    execution_document_v90(execution)
    if (isinstance(accepted_updates, bool) or type(accepted_updates) is not int
            or not 0 <= accepted_updates <= MAX_ACCEPTED_UPDATES):
        raise HlesProducerV90Error("v90 accepted update count differs")
    time = _finite(time_s, "time")
    if time < 0.0:
        raise HlesProducerV90Error("v90 time must be nonnegative")
    chain = _sha(receipt_chain, "accepted receipt chain")
    if type(primed) is not bool:
        raise HlesProducerV90Error("v90 primed flag must be Boolean")
    if primed:
        prime_sha = _sha(prime_receipt_sha256, "prime receipt")
    elif prime_receipt_sha256 is not None:
        raise HlesProducerV90Error(
            "v90 unprimed state cannot carry a prime receipt")
    else:
        prime_sha = None
    arrays = (
        _array(filtered, (execution.face_count,), "accepted filter"),
        _array(dynamic_vicuv, (execution.cell_count,),
               "accepted dynamic vicuv", nonnegative=True),
        _array(dynamic_dicuv, (execution.cell_count,),
               "accepted dynamic dicuv", nonnegative=True),
    )
    vic, dic = arrays[1], arrays[2]
    dynamic_present = (vic != 0.0) | (dic != 0.0)
    molecular_difference = (
        execution.closure_execution.vicmol_m2_s-
        execution.closure_execution.dicmol_m2_s)
    tolerance = 16.0*np.finfo(np.float64).eps*max(
        1.0, float(np.max(np.abs(vic), initial=0.0)),
        float(np.max(np.abs(dic), initial=0.0)))
    if np.any(dynamic_present) and not np.allclose(
            (vic-dic)[dynamic_present], molecular_difference,
            rtol=16.0*np.finfo(float).eps, atol=tolerance):
        raise HlesProducerV90Error(
            "v90 accepted dynamic vicuv/dicuv relation differs")
    if (not primed and (accepted_updates != 0 or time != 0.0 or
                        any(np.any(array != 0.0) for array in arrays))):
        raise HlesProducerV90Error(
            "v90 unprimed state must be the exact zero-time zero-memory seed")
    if primed and ((accepted_updates == 0 and time != 0.0) or
                   (accepted_updates > 0 and time <= 0.0)):
        raise HlesProducerV90Error(
            "v90 primed state update/time relationship differs")
    placeholder = HlesProducerStateV90(
        execution.fingerprint, *arrays, primed, prime_sha,
        accepted_updates, time, chain, "")
    return HlesProducerStateV90(
        execution.fingerprint, *arrays, primed, prime_sha,
        accepted_updates, time, chain,
        _state_token(placeholder))


def resolve_hles_producer_state_v90(
    execution: HlesProducerExecutionV90, *, accepted_updates: int = 0,
    time_s: float = 0.0,
    filtered_face_normal_velocity_m_s: object | None = None,
    dynamic_vicuv_m2_s: object | None = None,
    dynamic_dicuv_m2_s: object | None = None,
) -> HlesProducerStateV90:
    zeros_f = np.zeros(execution.face_count)
    zeros_c = np.zeros(execution.cell_count)
    chain = _fingerprint({
        "scope_id": SCOPE_ID,
        "execution_fingerprint": execution.fingerprint,
        "accepted_updates": accepted_updates,
        "time_s": float(time_s),
        "cold": True,
    })
    return _wrap_state(
        execution,
        filtered=(zeros_f if filtered_face_normal_velocity_m_s is None else
                  filtered_face_normal_velocity_m_s),
        dynamic_vicuv=(zeros_c if dynamic_vicuv_m2_s is None else
                       dynamic_vicuv_m2_s),
        dynamic_dicuv=(zeros_c if dynamic_dicuv_m2_s is None else
                       dynamic_dicuv_m2_s),
        accepted_updates=accepted_updates, time_s=time_s,
        receipt_chain=chain, primed=False, prime_receipt_sha256=None)


def validate_hles_producer_state_v90(
    state: HlesProducerStateV90, execution: HlesProducerExecutionV90,
) -> None:
    if (not isinstance(state, HlesProducerStateV90) or
            state.execution_fingerprint != execution.fingerprint or
            state.state_sha256 != _state_token(state)):
        raise HlesProducerV90Error(
            "v90 state is stale, mutated or execution-mismatched")
    _wrap_state(
        execution,
        filtered=state.accepted_filtered_face_normal_velocity_m_s,
        dynamic_vicuv=state.accepted_dynamic_vicuv_m2_s,
        dynamic_dicuv=state.accepted_dynamic_dicuv_m2_s,
        accepted_updates=state.accepted_updates, time_s=state.time_s,
        receipt_chain=state.accepted_receipt_chain_sha256,
        primed=state.primed,
        prime_receipt_sha256=state.prime_receipt_sha256)


def evaluate_lpfluc_v90(
    previous_filtered: object, mean_velocity: object, active_face: object, *,
    closure_interval_s: float, relaxation_time_min: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Evaluate one locked-form filter call for explicit ``hdt`` seconds."""
    previous_raw = np.asarray(previous_filtered)
    if previous_raw.ndim != 1:
        raise HlesProducerV90Error("v90 filter state must be one-dimensional")
    n = previous_raw.size
    previous = _array(previous_raw, (n,), "previous filter")
    mean = _array(mean_velocity, (n,), "mean face velocity")
    active = _array(active_face, (n,), "active face", dtype=np.int32,
                    binary=True)
    interval = _finite(
        closure_interval_s, "closure interval", positive=True)
    reltim = _finite(relaxation_time_min, "Hturrt")
    if not (reltim > 0.0 or reltim == -1.0):
        raise HlesProducerV90Error(
            "v90 Hturrt must be positive minutes or exactly -1")
    if reltim > 0.0:
        a = math.exp(-interval/(60.0*reltim))
        filtered = np.where(active == 1,
                            a*previous+(1.0-a)*mean, 0.0)
    else:
        a = 0.0
        filtered = np.zeros(n, dtype=np.float64)
    fluctuation = mean-filtered
    filtered = np.ascontiguousarray(filtered)
    fluctuation = np.ascontiguousarray(fluctuation)
    filtered.setflags(write=False)
    fluctuation.setflags(write=False)
    return filtered, fluctuation, a


def evaluate_structured_protke_oracle_v90(
    *, u_fluctuation: object, v_fluctuation: object,
    active_scalar: object, active_u: object, active_v: object,
    guu_m: object, gvv_m: object, west: object, south: object,
    east: object, north: object, southwest: object,
) -> np.ndarray:
    """Exact flat-index translation of locked ``protke.f90``.

    Arrays include every ghost/corner slot referenced by the supplied index
    maps.  This oracle is intentionally separate from the BRAIDS FV mapping.
    """
    raw = np.asarray(u_fluctuation)
    if raw.ndim != 1 or raw.size < 1:
        raise HlesProducerV90Error("v90 structured oracle is empty")
    n = raw.size
    u = _array(raw, (n,), "oracle u fluctuation")
    v = _array(v_fluctuation, (n,), "oracle v fluctuation")
    kfs = _array(active_scalar, (n,), "oracle kfs", dtype=np.int32,
                 binary=True)
    kfu = _array(active_u, (n,), "oracle kfu", dtype=np.int32, binary=True)
    kfv = _array(active_v, (n,), "oracle kfv", dtype=np.int32, binary=True)
    guu = _array(guu_m, (n,), "oracle guu", nonnegative=True)
    gvv = _array(gvv_m, (n,), "oracle gvv", nonnegative=True)
    maps = tuple(_array(value, (n,), name, dtype=np.int64)
                 for value, name in zip(
                     (west, south, east, north, southwest),
                     ("west", "south", "east", "north", "southwest")))
    if any(np.any((index < 0) | (index >= n)) for index in maps):
        raise HlesProducerV90Error("v90 structured oracle index is invalid")
    west_a, south_a, east_a, north_a, southwest_a = maps
    wrk1 = np.zeros(n)
    wrk2 = np.zeros(n)
    for cell in range(n):
        if kfs[cell] == 1:
            dx = 0.5*(gvv[cell]+gvv[south_a[cell]])
            dy = 0.5*(guu[cell]+guu[west_a[cell]])
            if dx <= 0.0 or dy <= 0.0:
                raise HlesProducerV90Error(
                    "v90 structured oracle spacing is nonpositive")
            wrk1[cell] = (
                ((u[cell]-u[west_a[cell]])/dx)**2+
                ((v[cell]-v[south_a[cell]])/dy)**2)
        if (kfv[cell]*kfv[east_a[cell]]*kfu[cell]*
                kfu[north_a[cell]] == 1):
            dx = 0.5*(gvv[cell]+gvv[east_a[cell]])
            dy = 0.5*(guu[cell]+guu[north_a[cell]])
            if dx <= 0.0 or dy <= 0.0:
                raise HlesProducerV90Error(
                    "v90 structured oracle corner spacing is nonpositive")
            wrk2[cell] = (
                (u[north_a[cell]]-u[cell])/dy+
                (v[east_a[cell]]-v[cell])/dx)**2
    result = np.zeros(n)
    wet = kfs == 1
    result[wet] = (
        2.0*wrk1[wet]+0.25*(
            wrk2[wet]+wrk2[west_a[wet]]+wrk2[south_a[wet]]+
            wrk2[southwest_a[wet]]))
    result.setflags(write=False)
    return result


def _reconstruct_cell_vector(
    execution: HlesProducerExecutionV90, face_normal_value: np.ndarray,
    face_active: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    base = execution.closure_execution.base_execution.base_execution
    nc = execution.cell_count
    vector = np.zeros((2, nc), dtype=np.float64)
    complete = np.zeros(nc, dtype=np.int32)
    for cell in range(nc):
        axx = axy = ayy = bx = by = 0.0
        count = 0
        for slot in range(base.cell_face_offset[cell],
                          base.cell_face_offset[cell+1]):
            face = int(base.cell_faces[slot])
            if face_active[face] == 0:
                continue
            nx, ny = base.normal[:, face]
            weight = base.face_length_m[face]/base.centre_distance_m[face]
            q = face_normal_value[face]
            axx += weight*nx*nx
            axy += weight*nx*ny
            ayy += weight*ny*ny
            bx += weight*nx*q
            by += weight*ny*q
            count += 1
        determinant = axx*ayy-axy*axy
        scale = max(1.0, (axx+ayy)*(axx+ayy))
        if count >= 2 and determinant > 256.0*np.finfo(float).eps*scale:
            vector[0, cell] = (ayy*bx-axy*by)/determinant
            vector[1, cell] = (axx*by-axy*bx)/determinant
            complete[cell] = 1
    vector.setflags(write=False)
    complete.setflags(write=False)
    return vector, complete


def _general_protke(
    execution: HlesProducerExecutionV90, fluctuation_cell: np.ndarray,
    reconstruction_complete: np.ndarray, active_cell: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    base = execution.closure_execution.base_execution.base_execution
    nc = execution.cell_count
    result = np.zeros(nc, dtype=np.float64)
    boundary = np.zeros(nc, dtype=np.int32)
    for cell in range(nc):
        dudx = dudy = dvdx = dvdy = 0.0
        valid = active_cell[cell] == 1 and reconstruction_complete[cell] == 1
        for slot in range(base.cell_face_offset[cell],
                          base.cell_face_offset[cell+1]):
            face = int(base.cell_faces[slot])
            other = (int(base.neighbour[face]) if
                     int(base.owner[face]) == cell else int(base.owner[face]))
            if other < 0:
                boundary[cell] = 1
                valid = False
                continue
            if (reconstruction_complete[other] == 0 or
                    active_cell[other] == 0):
                valid = False
                continue
            sign = float(base.cell_face_sign[slot])
            nx, ny = sign*base.normal[:, face]
            length = base.face_length_m[face]
            uf = 0.5*(fluctuation_cell[0, cell]+fluctuation_cell[0, other])
            vf = 0.5*(fluctuation_cell[1, cell]+fluctuation_cell[1, other])
            dudx += uf*nx*length
            dudy += uf*ny*length
            dvdx += vf*nx*length
            dvdy += vf*ny*length
        if valid:
            inverse_area = 1.0/base.cell_area_m2[cell]
            dudx *= inverse_area
            dudy *= inverse_area
            dvdx *= inverse_area
            dvdy *= inverse_area
            result[cell] = (
                2.0*(dudx*dudx+dvdy*dvdy)+(dudy+dvdx)**2)
    if not np.isfinite(result).all() or np.any(result < 0.0):
        raise HlesProducerV90Error("v90 protke mapping produced invalid data")
    result.setflags(write=False)
    boundary.setflags(write=False)
    return result, boundary


def _input_arrays(
    execution: HlesProducerExecutionV90, *,
    accepted_face_normal_velocity_m_s: object, active_face: object,
    closure_depth_m: object, closure_chezy_mhalf_s: object,
    active_cell: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        _array(accepted_face_normal_velocity_m_s,
               (execution.face_count,), "accepted face-normal velocity"),
        _array(active_face, (execution.face_count,), "active face",
               dtype=np.int32, binary=True),
        _array(closure_depth_m, (execution.cell_count,), "closure depth",
               nonnegative=True),
        _array(closure_chezy_mhalf_s, (execution.cell_count,),
               "closure Chezy", nonnegative=True),
        _array(active_cell, (execution.cell_count,), "active cell",
               dtype=np.int32, binary=True),
    )


def make_hles_prime_receipt_v90(
    execution: HlesProducerExecutionV90, parent: HlesProducerStateV90, *,
    initial_hydraulic_state_sha256: str, closure_interval_s: float,
    accepted_face_normal_velocity_m_s: object, active_face: object,
    closure_depth_m: object, closure_chezy_mhalf_s: object,
    active_cell: object,
) -> dict:
    """Seal the zero-time initialization call used before the first step."""
    validate_hles_producer_state_v90(parent, execution)
    if parent.primed or parent.accepted_updates != 0 or parent.time_s != 0.0:
        raise HlesProducerV90Error(
            "v90 cold prime requires the unprimed zero-time state")
    arrays = _input_arrays(
        execution,
        accepted_face_normal_velocity_m_s=accepted_face_normal_velocity_m_s,
        active_face=active_face, closure_depth_m=closure_depth_m,
        closure_chezy_mhalf_s=closure_chezy_mhalf_s,
        active_cell=active_cell)
    interval = _finite(
        closure_interval_s, "prime closure interval", positive=True)
    body = {
        "schema": SCHEMA,
        "scope_id": PRIME_RECEIPT_SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "execution_fingerprint": execution.fingerprint,
        "parent_state_sha256": parent.state_sha256,
        "initial_hydraulic_state_sha256": _sha(
            initial_hydraulic_state_sha256, "initial hydraulic state"),
        "arrays_sha256": [_array_digest(array) for array in arrays],
        "closure_interval_s": interval,
        "Hturrt_unit": "minutes",
        "accepted_update_increment": 0,
        "time_increment_s": 0.0,
        "source_call_site": "inchkr_lpfluc_protke_detvic_before_trisol",
    }
    return {**body, "fingerprint": _fingerprint(body)}


def _validate_prime_receipt(
    execution: HlesProducerExecutionV90, parent: HlesProducerStateV90,
    receipt: Mapping[str, object], arrays: tuple[np.ndarray, ...],
    closure_interval_s: float,
) -> None:
    keys = {
        "schema", "scope_id", "semantic_id", "execution_fingerprint",
        "parent_state_sha256", "initial_hydraulic_state_sha256",
        "arrays_sha256", "closure_interval_s", "Hturrt_unit",
        "accepted_update_increment", "time_increment_s",
        "source_call_site", "fingerprint",
    }
    if not isinstance(receipt, Mapping) or set(receipt) != keys:
        raise HlesProducerV90Error("v90 prime receipt inventory differs")
    body = {key: receipt[key] for key in receipt if key != "fingerprint"}
    if (receipt["fingerprint"] != _fingerprint(body) or
            receipt["schema"] != SCHEMA or
            receipt["scope_id"] != PRIME_RECEIPT_SCOPE_ID or
            receipt["semantic_id"] != SEMANTIC_ID or
            receipt["execution_fingerprint"] != execution.fingerprint or
            receipt["parent_state_sha256"] != parent.state_sha256 or
            not _SHA256.fullmatch(str(
                receipt["initial_hydraulic_state_sha256"])) or
            receipt["arrays_sha256"] !=
                [_array_digest(array) for array in arrays] or
            receipt["closure_interval_s"] != float(closure_interval_s) or
            receipt["Hturrt_unit"] != "minutes" or
            receipt["accepted_update_increment"] != 0 or
            receipt["time_increment_s"] != 0.0 or
            receipt["source_call_site"] !=
                "inchkr_lpfluc_protke_detvic_before_trisol"):
        raise HlesProducerV90Error(
            "v90 prime receipt is stale, mutated or input-mismatched")


@dataclass(frozen=True)
class HlesPrimeTrialV90:
    execution_fingerprint: str
    parent_state_sha256: str
    prime_receipt_fingerprint: str
    candidate: HlesProducerStateV90
    filter_coefficient: float
    filtered_face_normal_velocity_m_s: np.ndarray
    fluctuating_face_normal_velocity_m_s: np.ndarray
    reconstructed_mean_velocity_m_s: np.ndarray
    reconstructed_fluctuating_velocity_m_s: np.ndarray
    reconstruction_complete: np.ndarray
    boundary_cell: np.ndarray
    detvic_active_cell: np.ndarray
    ptke_s2: np.ndarray
    hles_vicuv_m2_s: np.ndarray
    elder_vicuv_m2_s: np.ndarray
    backend: str
    trial_sha256: str


def _prime_trial_token(value: HlesPrimeTrialV90) -> str:
    return _fingerprint({
        "execution_fingerprint": value.execution_fingerprint,
        "parent_state_sha256": value.parent_state_sha256,
        "prime_receipt_fingerprint": value.prime_receipt_fingerprint,
        "candidate_state_sha256": value.candidate.state_sha256,
        "filter_coefficient": value.filter_coefficient,
        "arrays_sha256": [_array_digest(array) for array in (
            value.filtered_face_normal_velocity_m_s,
            value.fluctuating_face_normal_velocity_m_s,
            value.reconstructed_mean_velocity_m_s,
            value.reconstructed_fluctuating_velocity_m_s,
            value.reconstruction_complete, value.boundary_cell,
            value.detvic_active_cell, value.ptke_s2,
            value.hles_vicuv_m2_s,
            value.elder_vicuv_m2_s)],
        "backend": value.backend,
    })


def _wrap_prime_trial(
    execution: HlesProducerExecutionV90, parent: HlesProducerStateV90,
    prime_receipt: Mapping[str, object], values: tuple, backend: str,
) -> HlesPrimeTrialV90:
    coefficient, filtered, fluctuation, mean_cell, fluct_cell, complete, \
        boundary, detvic_active, ptke, hles, elder, dynamic_vicuv, \
        dynamic_dicuv = values
    receipt_sha = _sha(
        prime_receipt.get("fingerprint"), "prime receipt fingerprint")
    chain = _fingerprint({
        "previous": parent.accepted_receipt_chain_sha256,
        "cold_prime": receipt_sha,
        "candidate_dynamic_vicuv": _array_digest(dynamic_vicuv),
        "candidate_dynamic_dicuv": _array_digest(dynamic_dicuv),
    })
    candidate = _wrap_state(
        execution, filtered=filtered, dynamic_vicuv=dynamic_vicuv,
        dynamic_dicuv=dynamic_dicuv, accepted_updates=0, time_s=0.0,
        receipt_chain=chain, primed=True,
        prime_receipt_sha256=receipt_sha)
    arrays = (
        _array(filtered, (execution.face_count,), "prime filter"),
        _array(fluctuation, (execution.face_count,), "prime fluctuation"),
        _array(mean_cell, (2, execution.cell_count), "prime mean cell"),
        _array(fluct_cell, (2, execution.cell_count), "prime fluct cell"),
        _array(complete, (execution.cell_count,), "prime complete",
               dtype=np.int32, binary=True),
        _array(boundary, (execution.cell_count,), "prime boundary",
               dtype=np.int32, binary=True),
        _array(detvic_active, (execution.cell_count,), "prime detvic active",
               dtype=np.int32, binary=True),
        _array(ptke, (execution.cell_count,), "prime ptke",
               nonnegative=True),
        _array(hles, (execution.cell_count,), "prime HLES",
               nonnegative=True),
        _array(elder, (execution.cell_count,), "prime Elder",
               nonnegative=True),
    )
    placeholder = HlesPrimeTrialV90(
        execution.fingerprint, parent.state_sha256, receipt_sha, candidate,
        float(coefficient), *arrays, backend, "")
    return HlesPrimeTrialV90(
        **{**placeholder.__dict__,
           "trial_sha256": _prime_trial_token(placeholder)})


def make_hles_input_receipt_v90(
    execution: HlesProducerExecutionV90, parent: HlesProducerStateV90, *,
    accepted_step_receipt: Mapping[str, object],
    current_handoff: CurrentClosureHandoffV90,
    accepted_face_normal_velocity_m_s: object, active_face: object,
    closure_depth_m: object, closure_chezy_mhalf_s: object,
    active_cell: object, closure_interval_s: float,
) -> dict:
    validate_hles_producer_state_v90(parent, execution)
    validate_current_closure_handoff_v90(
        current_handoff, execution, parent)
    arrays = _input_arrays(
        execution,
        accepted_face_normal_velocity_m_s=accepted_face_normal_velocity_m_s,
        active_face=active_face, closure_depth_m=closure_depth_m,
        closure_chezy_mhalf_s=closure_chezy_mhalf_s,
        active_cell=active_cell)
    if not isinstance(accepted_step_receipt, Mapping):
        raise HlesProducerV90Error("v90 accepted hydraulic receipt is absent")
    dt = _finite(accepted_step_receipt.get("dt_s"), "accepted dt",
                 positive=True)
    interval = _finite(
        closure_interval_s, "closure interval", positive=True)
    if interval > dt:
        raise HlesProducerV90Error(
            "v90 one endpoint closure interval cannot exceed accepted dt")
    validate_implicit_fwave_midpoint_step_receipt_v78(
        accepted_step_receipt,
        hydraulic_execution_sha256=
            execution.closure_execution.base_execution.hydraulic_execution_sha256,
        parent_accepted_updates=parent.accepted_updates,
        parent_time_s=parent.time_s, dt_s=dt)
    body = {
        "schema": SCHEMA,
        "scope_id": INPUT_RECEIPT_SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "execution_fingerprint": execution.fingerprint,
        "parent_state_sha256": parent.state_sha256,
        "accepted_step_receipt_fingerprint":
            _sha(accepted_step_receipt.get("fingerprint"),
                 "hydraulic receipt fingerprint"),
        "arrays_sha256": [_array_digest(array) for array in arrays],
        "mapping_id": MAPPING_ID,
        "filter_interval": FILTER_INTERVAL,
        "closure_interval_s": interval,
        "producer_cadence": PRODUCER_CADENCE,
        "current_handoff_receipt_sha256":
            current_handoff.receipt_sha256,
        "v91_hydraulic_echo_status": (
            "required_not_implemented_v90_zero_momentum_consumers"),
    }
    return {**body, "fingerprint": _fingerprint(body)}


def _validate_input_receipt(
    execution: HlesProducerExecutionV90, parent: HlesProducerStateV90,
    step: Mapping[str, object], receipt: Mapping[str, object],
    arrays: tuple[np.ndarray, ...],
    current_handoff: CurrentClosureHandoffV90,
) -> tuple[float, float]:
    validate_current_closure_handoff_v90(
        current_handoff, execution, parent)
    keys = {
        "schema", "scope_id", "semantic_id", "execution_fingerprint",
        "parent_state_sha256", "accepted_step_receipt_fingerprint",
        "arrays_sha256", "mapping_id", "filter_interval",
        "closure_interval_s", "producer_cadence",
        "current_handoff_receipt_sha256",
        "v91_hydraulic_echo_status", "fingerprint",
    }
    if not isinstance(receipt, Mapping) or set(receipt) != keys:
        raise HlesProducerV90Error("v90 input receipt inventory differs")
    body = {key: receipt[key] for key in receipt if key != "fingerprint"}
    if (receipt["fingerprint"] != _fingerprint(body) or
            receipt["schema"] != SCHEMA or
            receipt["scope_id"] != INPUT_RECEIPT_SCOPE_ID or
            receipt["semantic_id"] != SEMANTIC_ID or
            receipt["execution_fingerprint"] != execution.fingerprint or
            receipt["parent_state_sha256"] != parent.state_sha256 or
            receipt["accepted_step_receipt_fingerprint"] !=
                step.get("fingerprint") or
            receipt["arrays_sha256"] !=
                [_array_digest(array) for array in arrays] or
            receipt["mapping_id"] != MAPPING_ID or
            receipt["filter_interval"] != FILTER_INTERVAL or
            receipt["producer_cadence"] != PRODUCER_CADENCE or
            receipt["current_handoff_receipt_sha256"] !=
                current_handoff.receipt_sha256 or
            receipt["v91_hydraulic_echo_status"] !=
                "required_not_implemented_v90_zero_momentum_consumers"):
        raise HlesProducerV90Error(
            "v90 input receipt is stale, mutated or input-mismatched")
    dt = _finite(step.get("dt_s"), "accepted dt", positive=True)
    validate_implicit_fwave_midpoint_step_receipt_v78(
        step,
        hydraulic_execution_sha256=
            execution.closure_execution.base_execution.hydraulic_execution_sha256,
        parent_accepted_updates=parent.accepted_updates,
        parent_time_s=parent.time_s, dt_s=dt)
    interval = _finite(
        receipt["closure_interval_s"], "closure interval", positive=True)
    if interval > dt:
        raise HlesProducerV90Error(
            "v90 one endpoint closure interval exceeds accepted dt")
    return dt, interval


@dataclass(frozen=True)
class HlesClosureOutputReceiptV90:
    execution_fingerprint: str
    parent_state_sha256: str
    accepted_step_receipt_fingerprint: str
    backend: str
    current_handoff_receipt_sha256: str
    background_vicuv_m2_s: np.ndarray
    background_dicuv_m2_s: np.ndarray
    lagged_dynamic_vicuv_m2_s: np.ndarray
    lagged_dynamic_dicuv_m2_s: np.ndarray
    candidate_dynamic_vicuv_m2_s: np.ndarray
    candidate_dynamic_dicuv_m2_s: np.ndarray
    receipt_sha256: str

    @property
    def used_total_vicuv_m2_s(self) -> np.ndarray:
        value = np.ascontiguousarray(
            self.background_vicuv_m2_s+self.lagged_dynamic_vicuv_m2_s)
        value.setflags(write=False)
        return value

    @property
    def used_total_dicuv_m2_s(self) -> np.ndarray:
        value = np.ascontiguousarray(
            self.background_dicuv_m2_s+self.lagged_dynamic_dicuv_m2_s)
        value.setflags(write=False)
        return value


def _output_token(value: HlesClosureOutputReceiptV90) -> str:
    return _fingerprint({
        "scope_id": OUTPUT_RECEIPT_SCOPE_ID,
        "execution_fingerprint": value.execution_fingerprint,
        "parent_state_sha256": value.parent_state_sha256,
        "accepted_step_receipt_fingerprint":
            value.accepted_step_receipt_fingerprint,
        "backend": value.backend,
        "current_handoff_receipt_sha256":
            value.current_handoff_receipt_sha256,
        "arrays_sha256": [_array_digest(array) for array in (
            value.background_vicuv_m2_s, value.background_dicuv_m2_s,
            value.lagged_dynamic_vicuv_m2_s,
            value.lagged_dynamic_dicuv_m2_s,
            value.candidate_dynamic_vicuv_m2_s,
            value.candidate_dynamic_dicuv_m2_s)],
        "raw_field_lag": "one_accepted_update_both_vicuv_and_dicuv",
        "candidate_dicuv_same_update_scalar_consumer": False,
        "momentum_consumer_count": 0,
        "v91_required_hydraulic_echo": True,
        "owner_counts": {"filter": 1, "protke": 1, "detvic": 1,
                         "v82_v78_lifecycle": 0},
    })


@dataclass(frozen=True)
class CurrentClosureHandoffV90:
    execution_fingerprint: str
    parent_state_sha256: str
    parent_accepted_updates: int
    parent_time_s: float
    backend: str
    hydraulic_trial_request_sha256: str
    background_vicuv_m2_s: np.ndarray
    background_dicuv_m2_s: np.ndarray
    accepted_dynamic_vicuv_m2_s: np.ndarray
    accepted_dynamic_dicuv_m2_s: np.ndarray
    receipt_sha256: str

    @property
    def used_total_vicuv_m2_s(self) -> np.ndarray:
        result = np.ascontiguousarray(
            self.background_vicuv_m2_s+self.accepted_dynamic_vicuv_m2_s)
        result.setflags(write=False)
        return result

    @property
    def used_total_dicuv_m2_s(self) -> np.ndarray:
        result = np.ascontiguousarray(
            self.background_dicuv_m2_s+self.accepted_dynamic_dicuv_m2_s)
        result.setflags(write=False)
        return result


def _handoff_token(value: CurrentClosureHandoffV90) -> str:
    return _fingerprint({
        "scope_id": CURRENT_HANDOFF_SCOPE_ID,
        "execution_fingerprint": value.execution_fingerprint,
        "parent_state_sha256": value.parent_state_sha256,
        "parent_accepted_updates": value.parent_accepted_updates,
        "parent_time_s": value.parent_time_s,
        "backend": value.backend,
        "hydraulic_trial_request_sha256":
            value.hydraulic_trial_request_sha256,
        "arrays_sha256": [_array_digest(array) for array in (
            value.background_vicuv_m2_s, value.background_dicuv_m2_s,
            value.accepted_dynamic_vicuv_m2_s,
            value.accepted_dynamic_dicuv_m2_s)],
        "momentum_disposition": "v91_authenticated_consumer_not_v90",
        "scalar_disposition": "v89_v96_authenticated_consumer_not_v90",
        "candidate_disposition": "absent_pre_hydraulic",
    })


def make_current_closure_handoff_v90(
    execution: HlesProducerExecutionV90, parent: HlesProducerStateV90, *,
    hydraulic_trial_request_sha256: str,
) -> CurrentClosureHandoffV90:
    """Seal the lagged fields before the hydraulic trial starts."""
    validate_hles_producer_state_v90(parent, execution)
    if not parent.primed:
        raise HlesProducerV90Error(
            "v90 current closure handoff requires a primed parent")
    background = execution.closure_execution.background
    placeholder = CurrentClosureHandoffV90(
        execution.fingerprint, parent.state_sha256,
        parent.accepted_updates, parent.time_s, execution.backend,
        _sha(hydraulic_trial_request_sha256, "hydraulic trial request"),
        background.vicuv_m2_s, background.dicuv_m2_s,
        parent.accepted_dynamic_vicuv_m2_s,
        parent.accepted_dynamic_dicuv_m2_s, "")
    return CurrentClosureHandoffV90(
        **{**placeholder.__dict__,
           "receipt_sha256": _handoff_token(placeholder)})


def validate_current_closure_handoff_v90(
    value: CurrentClosureHandoffV90, execution: HlesProducerExecutionV90,
    parent: HlesProducerStateV90,
) -> None:
    validate_hles_producer_state_v90(parent, execution)
    background = execution.closure_execution.background
    if (not isinstance(value, CurrentClosureHandoffV90) or
            value.execution_fingerprint != execution.fingerprint or
            value.parent_state_sha256 != parent.state_sha256 or
            value.parent_accepted_updates != parent.accepted_updates or
            value.parent_time_s != parent.time_s or
            value.backend != execution.backend or
            not np.array_equal(
                value.background_vicuv_m2_s, background.vicuv_m2_s) or
            not np.array_equal(
                value.background_dicuv_m2_s, background.dicuv_m2_s) or
            not np.array_equal(
                value.accepted_dynamic_vicuv_m2_s,
                parent.accepted_dynamic_vicuv_m2_s) or
            not np.array_equal(
                value.accepted_dynamic_dicuv_m2_s,
                parent.accepted_dynamic_dicuv_m2_s) or
            value.receipt_sha256 != _handoff_token(value)):
        raise HlesProducerV90Error(
            "v90 current closure handoff is stale, mutated or mismatched")


@dataclass(frozen=True)
class HlesProducerTrialV90:
    execution_fingerprint: str
    parent_state_sha256: str
    input_receipt_fingerprint: str
    output_receipt: HlesClosureOutputReceiptV90
    candidate: HlesProducerStateV90
    accepted_dt_s: float
    filter_coefficient: float
    filtered_face_normal_velocity_m_s: np.ndarray
    fluctuating_face_normal_velocity_m_s: np.ndarray
    reconstructed_mean_velocity_m_s: np.ndarray
    reconstructed_fluctuating_velocity_m_s: np.ndarray
    reconstruction_complete: np.ndarray
    boundary_cell: np.ndarray
    detvic_active_cell: np.ndarray
    ptke_s2: np.ndarray
    hles_vicuv_m2_s: np.ndarray
    elder_vicuv_m2_s: np.ndarray
    backend: str
    trial_sha256: str


def _trial_token(value: HlesProducerTrialV90) -> str:
    return _fingerprint({
        "execution_fingerprint": value.execution_fingerprint,
        "parent_state_sha256": value.parent_state_sha256,
        "input_receipt_fingerprint": value.input_receipt_fingerprint,
        "output_receipt_sha256": value.output_receipt.receipt_sha256,
        "candidate_state_sha256": value.candidate.state_sha256,
        "accepted_dt_s": value.accepted_dt_s,
        "filter_coefficient": value.filter_coefficient,
        "arrays_sha256": [_array_digest(array) for array in (
            value.filtered_face_normal_velocity_m_s,
            value.fluctuating_face_normal_velocity_m_s,
            value.reconstructed_mean_velocity_m_s,
            value.reconstructed_fluctuating_velocity_m_s,
            value.reconstruction_complete, value.boundary_cell,
            value.detvic_active_cell, value.ptke_s2,
            value.hles_vicuv_m2_s,
            value.elder_vicuv_m2_s)],
        "backend": value.backend,
    })


def _wrap_trial(
    execution: HlesProducerExecutionV90, parent: HlesProducerStateV90,
    step: Mapping[str, object], input_receipt: Mapping[str, object], *,
    current_handoff: CurrentClosureHandoffV90,
    dt_s: float, coefficient: float, filtered: object, fluctuation: object,
    mean_cell: object, fluct_cell: object, complete: object, boundary: object,
    detvic_active: object, ptke: object, hles: object, elder: object,
    dynamic_vicuv: object, dynamic_dicuv: object, backend: str,
) -> HlesProducerTrialV90:
    arrays = (
        _array(filtered, (execution.face_count,), "candidate filter"),
        _array(fluctuation, (execution.face_count,), "face fluctuation"),
        _array(mean_cell, (2, execution.cell_count), "cell mean velocity"),
        _array(fluct_cell, (2, execution.cell_count),
               "cell fluctuating velocity"),
        _array(complete, (execution.cell_count,), "reconstruction complete",
               dtype=np.int32, binary=True),
        _array(boundary, (execution.cell_count,), "boundary cell",
               dtype=np.int32, binary=True),
        _array(detvic_active, (execution.cell_count,), "detvic active cell",
               dtype=np.int32, binary=True),
        _array(ptke, (execution.cell_count,), "ptke", nonnegative=True),
        _array(hles, (execution.cell_count,), "HLES viscosity",
               nonnegative=True),
        _array(elder, (execution.cell_count,), "Elder viscosity",
               nonnegative=True),
        _array(dynamic_vicuv, (execution.cell_count,), "dynamic vicuv",
               nonnegative=True),
        _array(dynamic_dicuv, (execution.cell_count,), "dynamic dicuv",
               nonnegative=True),
    )
    background = execution.closure_execution.background
    validate_current_closure_handoff_v90(
        current_handoff, execution, parent)
    output_placeholder = HlesClosureOutputReceiptV90(
        execution.fingerprint, parent.state_sha256,
        _sha(step.get("fingerprint"), "hydraulic receipt fingerprint"),
        backend, current_handoff.receipt_sha256,
        background.vicuv_m2_s, background.dicuv_m2_s,
        parent.accepted_dynamic_vicuv_m2_s,
        parent.accepted_dynamic_dicuv_m2_s, arrays[10], arrays[11], "")
    output = HlesClosureOutputReceiptV90(
        **{**output_placeholder.__dict__,
           "receipt_sha256": _output_token(output_placeholder)})
    chain = _fingerprint({
        "previous": parent.accepted_receipt_chain_sha256,
        "accepted_step": step["fingerprint"],
        "input": input_receipt["fingerprint"],
        "output": output.receipt_sha256,
    })
    candidate = _wrap_state(
        execution, filtered=arrays[0], dynamic_vicuv=arrays[10],
        dynamic_dicuv=arrays[11],
        accepted_updates=parent.accepted_updates+1,
        time_s=parent.time_s+dt_s, receipt_chain=chain,
        primed=True, prime_receipt_sha256=parent.prime_receipt_sha256)
    placeholder = HlesProducerTrialV90(
        execution.fingerprint, parent.state_sha256,
        str(input_receipt["fingerprint"]), output, candidate,
        float(dt_s), float(coefficient), *arrays[:10], backend, "")
    return HlesProducerTrialV90(
        **{**placeholder.__dict__, "trial_sha256": _trial_token(placeholder)})


def _cpu_candidate(
    execution: HlesProducerExecutionV90, parent: HlesProducerStateV90,
    arrays: tuple[np.ndarray, ...], closure_interval_s: float,
) -> tuple:
    mean_face, active_face, depth, chezy, active_cell = arrays
    filtered, fluctuation, coefficient = evaluate_lpfluc_v90(
        parent.accepted_filtered_face_normal_velocity_m_s, mean_face,
        active_face, closure_interval_s=closure_interval_s,
        relaxation_time_min=execution.relaxation_time_min)
    mean_cell, mean_complete = _reconstruct_cell_vector(
        execution, mean_face, active_face)
    fluct_cell, fluct_complete = _reconstruct_cell_vector(
        execution, fluctuation, active_face)
    complete = np.ascontiguousarray(
        ((mean_complete == 1) & (fluct_complete == 1)).astype(np.int32))
    complete.setflags(write=False)
    ptke, boundary = _general_protke(
        execution, fluct_cell, complete, active_cell)
    detvic_active = np.ascontiguousarray(
        ((active_cell == 1) & (mean_complete == 1)).astype(np.int32))
    field = evaluate_detvic_field_v87(
        execution.closure_execution, depth_m=depth,
        mean_velocity_m_s=mean_cell, chezy_mhalf_s=chezy, ptke_s2=ptke,
        active_cell=detvic_active)
    return (coefficient, filtered, fluctuation, mean_cell, fluct_cell,
            complete, boundary, detvic_active, ptke,
            field.hles_vicuv_m2_s, field.elder_vicuv_m2_s,
            field.dynamic_vicuv_m2_s, field.dynamic_dicuv_m2_s)


def begin_hles_cpu_prime_v90(
    execution: HlesProducerExecutionV90, parent: HlesProducerStateV90, *,
    prime_receipt: Mapping[str, object], closure_interval_s: float,
    accepted_face_normal_velocity_m_s: object, active_face: object,
    closure_depth_m: object, closure_chezy_mhalf_s: object,
    active_cell: object, cancelled: bool = False,
) -> HlesPrimeTrialV90:
    if execution.backend != BACKEND_CPU:
        raise HlesProducerV90Error(
            "v90 CPU prime requires the explicit CPU execution")
    validate_hles_producer_state_v90(parent, execution)
    if cancelled:
        raise HlesProducerV90Error(
            "v90 cold prime cancelled before any closure operator ran")
    if parent.primed or parent.accepted_updates != 0 or parent.time_s != 0.0:
        raise HlesProducerV90Error(
            "v90 cold prime requires the unprimed zero-time state")
    arrays = _input_arrays(
        execution,
        accepted_face_normal_velocity_m_s=accepted_face_normal_velocity_m_s,
        active_face=active_face, closure_depth_m=closure_depth_m,
        closure_chezy_mhalf_s=closure_chezy_mhalf_s,
        active_cell=active_cell)
    interval = _finite(
        closure_interval_s, "prime closure interval", positive=True)
    _validate_prime_receipt(
        execution, parent, prime_receipt, arrays, interval)
    return _wrap_prime_trial(
        execution, parent, prime_receipt,
        _cpu_candidate(execution, parent, arrays, interval), BACKEND_CPU)


def _validate_detvic_cross_fields(
    execution: HlesProducerExecutionV90, trial: object,
) -> None:
    active = np.asarray(trial.detvic_active_cell, dtype=np.float64)
    hles = np.asarray(trial.hles_vicuv_m2_s)
    elder = np.asarray(trial.elder_vicuv_m2_s)
    expected_vic = active*(
        hles+elder+execution.closure_execution.vicmol_m2_s)
    expected_dic = active*(
        hles+elder+execution.closure_execution.dicmol_m2_s)
    candidate = trial.candidate
    scale = max(
        1.0, float(np.max(np.abs(expected_vic), initial=0.0)),
        float(np.max(np.abs(expected_dic), initial=0.0)))
    tolerance = 16.0*np.finfo(np.float64).eps*scale
    inactive = active == 0.0
    if (np.any(hles[inactive] != 0.0) or
            np.any(elder[inactive] != 0.0) or
            (not execution.closure_execution.elder_enabled and
             np.any(elder != 0.0)) or
            not np.array_equal(
            candidate.accepted_filtered_face_normal_velocity_m_s,
            trial.filtered_face_normal_velocity_m_s) or
            not np.allclose(candidate.accepted_dynamic_vicuv_m2_s,
                            expected_vic, rtol=16.0*np.finfo(float).eps,
                            atol=tolerance) or
            not np.allclose(candidate.accepted_dynamic_dicuv_m2_s,
                            expected_dic, rtol=16.0*np.finfo(float).eps,
                            atol=tolerance)):
        raise HlesProducerV90Error(
            "v90 trial filter/detvic candidate cross-fields differ")


def _validate_prime_trial(
    execution: HlesProducerExecutionV90, parent: HlesProducerStateV90,
    trial: HlesPrimeTrialV90,
) -> None:
    validate_hles_producer_state_v90(parent, execution)
    if (not isinstance(trial, HlesPrimeTrialV90) or
            trial.execution_fingerprint != execution.fingerprint or
            trial.parent_state_sha256 != parent.state_sha256 or
            trial.backend != execution.backend or
            trial.trial_sha256 != _prime_trial_token(trial)):
        raise HlesProducerV90Error(
            "v90 prime trial is stale, mutated or parent-mismatched")
    validate_hles_producer_state_v90(trial.candidate, execution)
    expected_chain = _fingerprint({
        "previous": parent.accepted_receipt_chain_sha256,
        "cold_prime": trial.prime_receipt_fingerprint,
        "candidate_dynamic_vicuv": _array_digest(
            trial.candidate.accepted_dynamic_vicuv_m2_s),
        "candidate_dynamic_dicuv": _array_digest(
            trial.candidate.accepted_dynamic_dicuv_m2_s),
    })
    if (not trial.candidate.primed or
            trial.candidate.accepted_updates != parent.accepted_updates or
            trial.candidate.time_s != parent.time_s or
            trial.candidate.prime_receipt_sha256 !=
                trial.prime_receipt_fingerprint or
            trial.candidate.accepted_receipt_chain_sha256 != expected_chain):
        raise HlesProducerV90Error("v90 prime candidate timing differs")
    _validate_detvic_cross_fields(execution, trial)


def commit_hles_prime_v90(
    execution: HlesProducerExecutionV90, parent: HlesProducerStateV90,
    trial: HlesPrimeTrialV90,
) -> HlesProducerStateV90:
    _validate_prime_trial(execution, parent, trial)
    return trial.candidate


def rollback_hles_prime_v90(
    execution: HlesProducerExecutionV90, parent: HlesProducerStateV90,
    trial: HlesPrimeTrialV90,
) -> HlesProducerStateV90:
    _validate_prime_trial(execution, parent, trial)
    return parent


def begin_hles_cpu_trial_v90(
    execution: HlesProducerExecutionV90, parent: HlesProducerStateV90, *,
    accepted_step_receipt: Mapping[str, object],
    hles_input_receipt: Mapping[str, object],
    current_handoff: CurrentClosureHandoffV90,
    accepted_face_normal_velocity_m_s: object, active_face: object,
    closure_depth_m: object, closure_chezy_mhalf_s: object,
    active_cell: object, cancelled: bool = False,
) -> HlesProducerTrialV90:
    if execution.backend != BACKEND_CPU:
        raise HlesProducerV90Error(
            "v90 CPU trial requires the explicit CPU execution")
    validate_hles_producer_state_v90(parent, execution)
    if cancelled:
        raise HlesProducerV90Error(
            "v90 cancelled before any closure operator ran")
    if parent.accepted_updates >= MAX_ACCEPTED_UPDATES:
        raise HlesProducerV90Error("v90 accepted update limit was reached")
    arrays = _input_arrays(
        execution,
        accepted_face_normal_velocity_m_s=accepted_face_normal_velocity_m_s,
        active_face=active_face, closure_depth_m=closure_depth_m,
        closure_chezy_mhalf_s=closure_chezy_mhalf_s,
        active_cell=active_cell)
    dt, interval = _validate_input_receipt(
        execution, parent, accepted_step_receipt, hles_input_receipt, arrays,
        current_handoff)
    if not parent.primed:
        raise HlesProducerV90Error(
            "v90 accepted update requires an authenticated cold prime")
    values = _cpu_candidate(execution, parent, arrays, interval)
    return _wrap_trial(
        execution, parent, accepted_step_receipt, hles_input_receipt,
        current_handoff=current_handoff,
        dt_s=dt, coefficient=values[0], filtered=values[1],
        fluctuation=values[2], mean_cell=values[3], fluct_cell=values[4],
        complete=values[5], boundary=values[6], detvic_active=values[7],
        ptke=values[8], hles=values[9], elder=values[10],
        dynamic_vicuv=values[11], dynamic_dicuv=values[12],
        backend=BACKEND_CPU)


def _validate_trial(
    execution: HlesProducerExecutionV90, parent: HlesProducerStateV90,
    trial: HlesProducerTrialV90,
) -> None:
    validate_hles_producer_state_v90(parent, execution)
    output = trial.output_receipt if isinstance(
        trial, HlesProducerTrialV90) else None
    background = execution.closure_execution.background
    if (not isinstance(trial, HlesProducerTrialV90) or
            not isinstance(output, HlesClosureOutputReceiptV90) or
            trial.execution_fingerprint != execution.fingerprint or
            trial.parent_state_sha256 != parent.state_sha256 or
            trial.backend != execution.backend or
            output.execution_fingerprint != execution.fingerprint or
            output.parent_state_sha256 != parent.state_sha256 or
            output.backend != execution.backend or
            _SHA256.fullmatch(output.current_handoff_receipt_sha256) is None or
            output.receipt_sha256 != _output_token(output) or
            not np.array_equal(
                output.background_vicuv_m2_s, background.vicuv_m2_s) or
            not np.array_equal(
                output.background_dicuv_m2_s, background.dicuv_m2_s) or
            not np.array_equal(
                output.lagged_dynamic_vicuv_m2_s,
                parent.accepted_dynamic_vicuv_m2_s) or
            not np.array_equal(
                output.lagged_dynamic_dicuv_m2_s,
                parent.accepted_dynamic_dicuv_m2_s) or
            not np.array_equal(
                output.candidate_dynamic_vicuv_m2_s,
                trial.candidate.accepted_dynamic_vicuv_m2_s) or
            not np.array_equal(
                output.candidate_dynamic_dicuv_m2_s,
                trial.candidate.accepted_dynamic_dicuv_m2_s) or
            trial.trial_sha256 != _trial_token(trial)):
        raise HlesProducerV90Error(
            "v90 trial is stale, mutated or belongs to another parent")
    validate_hles_producer_state_v90(trial.candidate, execution)
    expected_chain = _fingerprint({
        "previous": parent.accepted_receipt_chain_sha256,
        "accepted_step": output.accepted_step_receipt_fingerprint,
        "input": trial.input_receipt_fingerprint,
        "output": output.receipt_sha256,
    })
    if (trial.accepted_dt_s <= 0.0 or
            trial.candidate.accepted_updates != parent.accepted_updates+1 or
            trial.candidate.time_s != parent.time_s+trial.accepted_dt_s or
            trial.candidate.prime_receipt_sha256 !=
                parent.prime_receipt_sha256 or
            trial.candidate.accepted_receipt_chain_sha256 != expected_chain):
        raise HlesProducerV90Error(
            "v90 trial candidate timing or prime provenance differs")
    _validate_detvic_cross_fields(execution, trial)


def commit_hles_trial_v90(
    execution: HlesProducerExecutionV90, parent: HlesProducerStateV90,
    trial: HlesProducerTrialV90,
) -> HlesProducerStateV90:
    _validate_trial(execution, parent, trial)
    return trial.candidate


def rollback_hles_trial_v90(
    execution: HlesProducerExecutionV90, parent: HlesProducerStateV90,
    trial: HlesProducerTrialV90,
) -> HlesProducerStateV90:
    _validate_trial(execution, parent, trial)
    return parent


def cancel_hles_trial_v90(
    execution: HlesProducerExecutionV90, parent: HlesProducerStateV90,
    trial: HlesProducerTrialV90,
) -> HlesProducerStateV90:
    return rollback_hles_trial_v90(execution, parent, trial)


class HlesProducerLifecycleV90:
    """CPU transaction shell; it owns closure memory and no v82/v78 work."""

    def __init__(self, execution: HlesProducerExecutionV90):
        if execution.backend != BACKEND_CPU:
            raise HlesProducerV90Error(
                "v90 CPU owner requires the explicit CPU execution")
        self.execution = execution
        self._open_parent: str | None = None
        self._open_trial: str | None = None

    def begin_trial(self, parent: HlesProducerStateV90, **kwargs
                    ) -> HlesProducerTrialV90:
        if self._open_trial is not None:
            raise HlesProducerV90Error("v90 previous trial is unfinished")
        trial = begin_hles_cpu_trial_v90(self.execution, parent, **kwargs)
        self._open_parent = parent.state_sha256
        self._open_trial = trial.trial_sha256
        return trial

    def begin_prime(self, parent: HlesProducerStateV90, **kwargs
                    ) -> HlesPrimeTrialV90:
        if self._open_trial is not None:
            raise HlesProducerV90Error("v90 previous transaction is unfinished")
        trial = begin_hles_cpu_prime_v90(self.execution, parent, **kwargs)
        self._open_parent = parent.state_sha256
        self._open_trial = trial.trial_sha256
        return trial

    def _finish(self, parent, trial, *, commit: bool):
        _validate_trial(self.execution, parent, trial)
        if (self._open_parent != parent.state_sha256 or
                self._open_trial != trial.trial_sha256):
            raise HlesProducerV90Error(
                "v90 trial is not the open CPU transaction")
        self._open_parent = self._open_trial = None
        return trial.candidate if commit else parent

    def commit(self, parent, trial):
        return self._finish(parent, trial, commit=True)

    def rollback(self, parent, trial):
        return self._finish(parent, trial, commit=False)

    def cancel(self, parent, trial):
        return self.rollback(parent, trial)

    def _finish_prime(self, parent, trial, *, commit: bool):
        _validate_prime_trial(self.execution, parent, trial)
        if (self._open_parent != parent.state_sha256 or
                self._open_trial != trial.trial_sha256):
            raise HlesProducerV90Error(
                "v90 prime trial is not the open CPU transaction")
        self._open_parent = self._open_trial = None
        return trial.candidate if commit else parent

    def commit_prime(self, parent, trial):
        return self._finish_prime(parent, trial, commit=True)

    def rollback_prime(self, parent, trial):
        return self._finish_prime(parent, trial, commit=False)

    def cancel_prime(self, parent, trial):
        return self.rollback_prime(parent, trial)

    def checkpoint(self, state):
        if self._open_trial is not None:
            raise HlesProducerV90Error(
                "v90 checkpoint requires a committed point")
        return hles_checkpoint_v90(self.execution, state)

    def restore_checkpoint(self, document):
        if self._open_trial is not None:
            raise HlesProducerV90Error(
                "v90 restore requires no open trial")
        return load_hles_checkpoint_v90(self.execution, document)


def make_v89_closure_receipt_v90(
    execution: HlesProducerExecutionV90, parent: HlesProducerStateV90,
    trial: HlesProducerTrialV90, *,
    single_owner_execution: SingleOwnerExecutionV89,
    single_owner_parent: SingleOwnerStateV89,
    accepted_step_receipt: Mapping[str, object],
) -> ClosureReceiptV89:
    """Adapt a v90 output to the v89 ABI without invoking either lifecycle."""
    _validate_trial(execution, parent, trial)
    validate_single_owner_state_v89(single_owner_parent,
                                    single_owner_execution)
    if (single_owner_execution.backend != execution.backend or
            single_owner_execution.base_execution.fingerprint !=
                execution.closure_execution.base_execution.fingerprint or
            single_owner_parent.accepted_updates != parent.accepted_updates or
            single_owner_parent.time_s != parent.time_s or
            accepted_step_receipt.get("fingerprint") !=
                trial.output_receipt.accepted_step_receipt_fingerprint):
        raise HlesProducerV90Error(
            "v90 and v89 execution, parent, or accepted step differ")
    output = trial.output_receipt
    return make_closure_receipt_v89(
        single_owner_execution, single_owner_parent,
        accepted_step_receipt=accepted_step_receipt,
        background_vicuv_m2_s=output.background_vicuv_m2_s,
        background_dicuv_m2_s=output.background_dicuv_m2_s,
        lagged_dynamic_vicuv_m2_s=output.lagged_dynamic_vicuv_m2_s,
        lagged_dynamic_dicuv_m2_s=output.lagged_dynamic_dicuv_m2_s,
        candidate_dynamic_vicuv_m2_s=output.candidate_dynamic_vicuv_m2_s,
        candidate_dynamic_dicuv_m2_s=output.candidate_dynamic_dicuv_m2_s,
        activate_candidate=False)


def hles_checkpoint_v90(
    execution: HlesProducerExecutionV90, state: HlesProducerStateV90,
) -> dict:
    validate_hles_producer_state_v90(state, execution)
    body = {
        "schema": SCHEMA,
        "scope_id": CHECKPOINT_SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "execution_fingerprint": execution.fingerprint,
        "integrator": INTEGRATOR,
        "backend": execution.backend,
        "filtered_face_normal_velocity_m_s":
            state.accepted_filtered_face_normal_velocity_m_s.tolist(),
        "dynamic_vicuv_m2_s": state.accepted_dynamic_vicuv_m2_s.tolist(),
        "dynamic_dicuv_m2_s": state.accepted_dynamic_dicuv_m2_s.tolist(),
        "primed": state.primed,
        "prime_receipt_sha256": state.prime_receipt_sha256,
        "accepted_updates": state.accepted_updates,
        "time_s": state.time_s,
        "accepted_receipt_chain_sha256":
            state.accepted_receipt_chain_sha256,
        "state_sha256": state.state_sha256,
    }
    return {**body, "fingerprint": _fingerprint(body)}


def load_hles_checkpoint_v90(
    execution: HlesProducerExecutionV90, document: Mapping[str, object],
) -> HlesProducerStateV90:
    expected = {
        "schema", "scope_id", "semantic_id", "execution_fingerprint",
        "integrator", "backend", "filtered_face_normal_velocity_m_s",
        "dynamic_vicuv_m2_s", "dynamic_dicuv_m2_s", "primed",
        "prime_receipt_sha256", "accepted_updates", "time_s",
        "accepted_receipt_chain_sha256", "state_sha256", "fingerprint",
    }
    if not isinstance(document, Mapping) or set(document) != expected:
        raise HlesProducerV90Error("v90 checkpoint inventory differs")
    body = {key: document[key] for key in document if key != "fingerprint"}
    if (document["fingerprint"] != _fingerprint(body) or
            document["schema"] != SCHEMA or
            document["scope_id"] != CHECKPOINT_SCOPE_ID or
            document["semantic_id"] != SEMANTIC_ID or
            document["execution_fingerprint"] != execution.fingerprint or
            document["integrator"] != INTEGRATOR or
            document["backend"] != execution.backend):
        raise HlesProducerV90Error("v90 checkpoint identity differs")
    state = _wrap_state(
        execution,
        filtered=document["filtered_face_normal_velocity_m_s"],
        dynamic_vicuv=document["dynamic_vicuv_m2_s"],
        dynamic_dicuv=document["dynamic_dicuv_m2_s"],
        accepted_updates=document["accepted_updates"],
        time_s=document["time_s"],
        receipt_chain=document["accepted_receipt_chain_sha256"],
        primed=document["primed"],
        prime_receipt_sha256=document["prime_receipt_sha256"])
    if state.state_sha256 != document["state_sha256"]:
        raise HlesProducerV90Error("v90 checkpoint state differs")
    return state


LOCKED_SOURCE_SHA256 = {
    "packages/flow2d3d_kernel/src/compute/lpfluc.f90":
        "f5c5bcb1bc8ef3cc292dc953b4313a2486cf3398b144cadf1badacf9ba0b597e",
    "packages/flow2d3d_kernel/src/compute/protke.f90":
        "f139db5be3261375dae141de3d77d9b82c63957a55a7bc774118c3f2fb3095e5",
    "packages/flow2d3d_kernel/src/compute/detvic.f90":
        "10de82c1be34cfe671ffd3765f35f31db97bb9657c06bf7c84fae8914bbc90dd",
    "packages/flow2d3d_kernel/src/compute/turclo.f90":
        "f91385a371000ae1d5c550dcb18fa617563379c1050ceaa31b37217d7b75270f",
    "packages/flow2d3d_kernel/src/compute/difu.f90":
        "e8e11d438a6e3aa8e912ce1133efdec093b5bad60605aefa6c4b037ea6aa5c17",
    "packages/flow2d3d_kernel/src/main/trisol.f90":
        "4b3e7ce63aed6c0cf95fe20ea3b9aa80fee9b3cd6fdf8101f0fbf7af5ca689e0",
    "packages/flow2d3d_kernel/src/inichk/inchkr.f90":
        "c1126a29ca032f3183aeb9281a49df00096b231c9516aac8b35f1eb67989ba0f",
    "packages/flow2d3d_io/src/input/rdhyvd.f90":
        "d6df2cb7a31fbb493930d445ca52b9bafb0149b49adb39e2f58cae01d1c7e11f",
    "packages/flow2d3d_data/src/allocation/esm_alloc_real.f90":
        "9304cc59a9f229373c74788cf05a004556b203ff4e1f50b12d7713c488fdffc4",
    "packages/flow2d3d_data/src/allocation/mkfpnt.f90":
        "69d2c25986d623ae8747fac45379000a886d6b34e108ad36a22d1e21cea06ac3",
    "packages/flow2d3d_data/src/allocation/mkdpnt.f90":
        "2dbab71f41f3bd0d047cd545f0af6d6de55637b82791b013934f526389281679",
    "packages/flow2d3d_data/src/allocation/mkrpnt.f90":
        "5906835ff5cfdf66a49eb4401a3617fac9bf68246ee13225bf9941507b535251",
    "packages/flow2d3d_data/src/allocation/dnull.f90":
        "992d03a3c5dd41bf858ca412597c8ff7715499f4d46660387f2957ca6da99bd2",
    "packages/flow2d3d_data/src/allocation/rnull.f90":
        "dbdfbb62b1b72e190d20454875cfdc0f60278e42edbf02985ad3342178393b45",
    "packages/flow2d3d_manager/src/tricom_init.F90":
        "18ce87c3622a3f0a5ea594049d688d893765957dd99c8d02d2697682bd2db8ad",
    "packages/flow2d3d_kernel/src/compute/z_turclo.f90":
        "b3e27e904a56f71b6562ab5163356a93f1bd8d61a18abd6fcbec5741e29e2119",
    "packages/flow2d3d_kernel/src/compute/z_difu.f90":
        "0ca9a85c727967309e7c48a125178a4332b92caeaafb9e070cbccc33b52c3c47",
    "packages/flow2d3d_io/src/input/restart_trim_flow.f90":
        "0c7437457ff15a743f1f05c269b00c71eb84cab16e521ecb67845ed64a27cbaa",
    "packages/flow2d3d_io/src/output/wrirst.f90":
        "e302f708aa62a345054b60ded6b8cb50a1feeb24419b69cd4fc96be88b995cd3",
    "packages/flow2d3d_kernel/src/inichk/chkvic.f90":
        "17cc4d2d6287769edb5b44fee9d88749811da9c740cfe04a16c9efb6bc222200",
    "packages/flow2d3d_kernel/src/inichk/iniphy.f90":
        "94dd55f636bb49e3c0a7d23deee06766dc15aeab4dd1152f60b09b98723f27c0",
    "packages/flow2d3d_data/src/gdp/initwrirst.f90":
        "8a9d1318047dc84590d0314cfc4c8f3214d3033ee223e92a959ba335cb3ea3e5",
}


def verify_locked_sources_v90(root: str | Path) -> dict[str, str]:
    root = Path(root)
    source = root/"resources/Delft3D-main/src/engines_gpl/flow2d3d"
    result = {}
    for relative, expected in LOCKED_SOURCE_SHA256.items():
        actual = hashlib.sha256((source/relative).read_bytes()).hexdigest()
        if actual != expected:
            raise HlesProducerV90Error(
                f"v90 locked source hash differs: {relative}")
        result[relative] = actual
    return result


def method_identity_v90() -> dict:
    return {
        "scope_id": SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "integrator": INTEGRATOR,
        "target_flow_path": TARGET_FLOW_PATH,
        "formula_chain": "lpfluc->protke->existing_v87_detvic",
        "Hturrt_unit": "minutes",
        "filter_interval": FILTER_INTERVAL,
        "producer_cadence": PRODUCER_CADENCE,
        "cold_start": "mandatory_authenticated_zero_time_prime",
        "structured_oracle": STRUCTURED_ORACLE_ID,
        "braids_mapping": MAPPING_ID,
        "general_mesh_source_equivalence_claimed": False,
        "mapping_scope": "uniform_axis_aligned_cartesian_fixed_grid",
        "cpu_implementation_scope": (
            "bounded_serial_python_component_oracle; native/OpenMP "
            "production promotion remains v96/Q10"),
        "cuda_receipt_scope": (
            "actual device physical kernels with host digest/readback "
            "overhead; resident validation promotion remains v96/Q10"),
        "molecular_viscosity": (
            "explicit_v87_configuration; dynamic v88 consumer remains v92"),
        "raw_dynamic_field_lag_accepted_updates": 1,
        "candidate_dicuv_same_update_scalar_consumer": False,
        "pre_hydraulic_handoff": (
            "sealed_uncapped_parent_dynamic_plus_background"),
        "momentum_consumer_count": 0,
        "v91_required_hydraulic_echo": True,
        "owner_counts": {"filter": 1, "protke": 1, "detvic": 1,
                         "v82_v78_lifecycle": 0},
        "backend_roles": [BACKEND_CPU, BACKEND_GPU],
        "accepted_update_limit": MAX_ACCEPTED_UPDATES,
        "automatic_cpu_fallback": False,
        "not_claimed": [
            "momentum consumption or chkvic cap (v91)",
            "accepted hydraulic receipt echo of the handoff (v91)",
            "moving wet/dry fronts or partially active reconstruction "
            "accuracy beyond inactive-cell zeroing",
            "arbitrary AMR mapping", "z-model KMAX=1 semantics",
            "classic two-ADI-stage full-step filter trajectory",
            "full Q4 coupled transaction", "long-run equivalence",
        ],
        "delft3d_source_modified": False,
    }


__all__ = [
    "BACKEND_CPU", "BACKEND_GPU", "CHECKPOINT_SCOPE_ID",
    "CURRENT_HANDOFF_SCOPE_ID", "CurrentClosureHandoffV90",
    "FILTER_INTERVAL", "PRODUCER_CADENCE", "HlesClosureOutputReceiptV90",
    "HlesProducerExecutionV90", "HlesProducerLifecycleV90",
    "HlesPrimeTrialV90", "HlesProducerStateV90", "HlesProducerTrialV90",
    "HlesProducerV90Error", "INTEGRATOR", "LOCKED_SOURCE_SHA256",
    "MAPPING_ID", "MAX_ACCEPTED_UPDATES", "SCHEMA", "SCOPE_ID",
    "SEMANTIC_ID", "STRUCTURED_ORACLE_ID", "TARGET_FLOW_PATH",
    "begin_hles_cpu_prime_v90", "begin_hles_cpu_trial_v90",
    "cancel_hles_trial_v90", "commit_hles_prime_v90",
    "commit_hles_trial_v90", "evaluate_lpfluc_v90",
    "evaluate_structured_protke_oracle_v90", "execution_document_v90",
    "hles_checkpoint_v90", "load_hles_checkpoint_v90",
    "make_current_closure_handoff_v90",
    "make_hles_input_receipt_v90", "make_hles_prime_receipt_v90",
    "make_v89_closure_receipt_v90",
    "method_identity_v90", "resolve_hles_producer_execution_v90",
    "resolve_hles_producer_state_v90", "rollback_hles_prime_v90",
    "rollback_hles_trial_v90",
    "validate_current_closure_handoff_v90",
    "validate_hles_producer_state_v90", "verify_locked_sources_v90",
]
