"""Bounded Q4/v91 shared consumers of the lagged FLOW-2DH closure.

V90 seals the background plus previously accepted HLES/Elder fields before
hydraulics.  This module is the single bounded consumer of that handoff.  It
applies the classic-FLOW ``chkvic`` stability ceiling to the *used total*
fields once, supplies the capped ``vicuv`` to a face-viscous momentum rate,
and supplies the same capped ``dicuv`` to the v82 scalar owner.  V82 alone
applies the literal KMAX=1 horizontal Schmidt division by 0.7.

The CPU path is an implicit-midpoint numerical oracle with a frozen caller
base rate.  The selected production CUDA controller hook lives in
``braids_gpu.shared_turbulence_consumers_v91`` and is invoked from every real
``ImplicitMidpointController._full_rate`` evaluation.  This milestone is
limited to the v90 uniform Cartesian, fully wet, fixed-grid scope and at most
four accepted updates; it is not the eventual v96 full Q4 owner.
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

from .hles_producer_v90 import (
    BACKEND_CPU,
    BACKEND_GPU,
    CurrentClosureHandoffV90,
    HlesProducerExecutionV90,
    HlesProducerStateV90,
    execution_document_v90,
    validate_current_closure_handoff_v90,
    validate_hles_producer_state_v90,
)
from .multifraction_suspended_lifecycle_v78 import (
    MAX_ACCEPTED_UPDATES,
    MultifractionSuspendedStateV78,
    ROLE_CPU_ORACLE,
    ROLE_CUDA,
    load_multifraction_checkpoint_v78,
    make_implicit_fwave_midpoint_step_receipt_v78,
    multifraction_checkpoint_v78,
    validate_implicit_fwave_midpoint_step_receipt_v78,
    validate_multifraction_suspended_state_v78,
)
from .suspended_support_closures_v82 import (
    SuspendedSupportExecutionV82,
    SuspendedSupportTrialV82,
    _validate_trial as _validate_v82_trial,
    _wrap_state as _wrap_v82_state,
    begin_suspended_support_cpu_trial_v82,
    commit_suspended_support_trial_v82,
    resolve_suspended_support_execution_v82,
    rollback_suspended_support_trial_v82,
)


SCHEMA = 1
SCOPE_ID = "braids-q4-shared-turbulence-consumers-v91-v1"
CAP_SCOPE_ID = SCOPE_ID + ":used-total-chkvic"
HYDRAULIC_RECEIPT_SCOPE_ID = SCOPE_ID + ":accepted-hydraulic"
CHECKPOINT_SCOPE_ID = SCOPE_ID + ":checkpoint"
SEMANTIC_ID = hashlib.sha256(SCOPE_ID.encode("ascii")).hexdigest()
INTEGRATOR = "implicit_fwave_midpoint"
MAX_UPDATES = MAX_ACCEPTED_UPDATES
HORIZONTAL_SCHMIDT = 0.7
CHKVIC_SAFETY = 0.9
CHKVIC_HDT_FRACTION = 0.5
MOMENTUM_OPERATOR_ID = (
    "uniform_cartesian_cell_vicuv_arithmetic_face_velocity_laplacian_v1")
CPU_ORACLE_ID = "frozen_base_rate_implicit_midpoint_v1"
STATE_DOMAIN_FULLY_WET_V91 = "fully_wet_v91"
STATE_DOMAIN_MOVING_WETDRY_V97 = "moving_wetdry_v97"
STATE_DOMAIN_MODES = (
    STATE_DOMAIN_FULLY_WET_V91,
    STATE_DOMAIN_MOVING_WETDRY_V97,
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class SharedTurbulenceV91Error(ValueError):
    """A v91 execution, receipt, state, or transaction is invalid."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError, OverflowError) as error:
        raise SharedTurbulenceV91Error(
            "v91 value is not finite canonical JSON") from error


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(_canonical(list(array.shape)))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise SharedTurbulenceV91Error(f"v91 {name} must be a SHA-256 digest")
    return value


def _finite(value: object, name: str, *, positive: bool = False) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(float(value))):
        raise SharedTurbulenceV91Error(f"v91 {name} must be finite")
    result = float(value)
    if positive and result <= 0.0:
        raise SharedTurbulenceV91Error(f"v91 {name} must be positive")
    return result


def _array(value: object, shape: tuple[int, ...], name: str, *,
           nonnegative: bool = False) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != shape or raw.dtype.kind not in "fiu":
        raise SharedTurbulenceV91Error(
            f"v91 {name} must be numeric with shape {shape}")
    # Validation must not change a caller-owned ndarray's writeability.  A
    # private copy is frozen below and used in the authenticated object.
    result = np.array(raw, dtype=np.float64, order="C", copy=True)
    if not np.isfinite(result).all():
        raise SharedTurbulenceV91Error(f"v91 {name} must be finite")
    if nonnegative and np.any(result < 0.0):
        raise SharedTurbulenceV91Error(f"v91 {name} must be nonnegative")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class SharedTurbulenceExecutionV91:
    hles_execution: HlesProducerExecutionV90
    backend: str
    semantic_id: str
    fingerprint: str
    # Additive v97 state-domain admission.  The default is deliberately
    # omitted from the serialized execution document below, preserving every
    # historical v91/v92/v93/v94 fingerprint byte-for-byte.
    state_domain_mode: str = STATE_DOMAIN_FULLY_WET_V91

    @property
    def base_execution(self):
        return self.hles_execution.closure_execution.base_execution

    @property
    def cell_count(self) -> int:
        return self.hles_execution.cell_count

    @property
    def face_count(self) -> int:
        return self.hles_execution.face_count


def _execution_payload(value: SharedTurbulenceExecutionV91) -> dict:
    if value.state_domain_mode not in STATE_DOMAIN_MODES:
        raise SharedTurbulenceV91Error(
            "v91 state-domain mode is not admitted")
    payload = {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "semantic_id": value.semantic_id,
        "hles_v90_execution": execution_document_v90(value.hles_execution),
        "backend": value.backend,
        "integrator": INTEGRATOR,
        "mapping_scope": "uniform_axis_aligned_cartesian_fixed_grid",
        "fully_wet": True,
        "all_cells_active": True,
        "amr": False,
        "chkvic_target": "classic_flow_sigma_kmax1_used_total_once",
        "chkvic_hdt_fraction_of_dt": CHKVIC_HDT_FRACTION,
        "chkvic_safety": CHKVIC_SAFETY,
        "momentum_operator": MOMENTUM_OPERATOR_ID,
        "scalar_operator": "v82_once_with_literal_dicuv_over_0p7",
        "duplicate_post_source_count": 0,
        "accepted_update_limit": MAX_UPDATES,
        "automatic_cpu_fallback": False,
        "product_admission": False,
    }
    if value.state_domain_mode == STATE_DOMAIN_MOVING_WETDRY_V97:
        payload.update({
            "state_domain_mode": STATE_DOMAIN_MOVING_WETDRY_V97,
            "fully_wet": False,
            "all_cells_active": True,
            "dry_state_policy": (
                "nonnegative_depth_exact_zero_dry_momentum_and_suspended"),
            "momentum_operator":
                "v97_native_wet_active_same_v91_equation",
            "scalar_operator":
                "v97_one_combined_sediment_fluid_pass_with_dry_transfer",
        })
    return payload


def resolve_shared_turbulence_execution_v91(
    hles_execution: HlesProducerExecutionV90,
) -> SharedTurbulenceExecutionV91:
    """Resolve the explicit CPU or selected actual-CUDA consumer route."""
    execution_document_v90(hles_execution)
    if hles_execution.backend not in (BACKEND_CPU, BACKEND_GPU):
        raise SharedTurbulenceV91Error("v91 backend must be explicit")
    expected = (ROLE_CPU_ORACLE if hles_execution.backend == BACKEND_CPU
                else ROLE_CUDA)
    if hles_execution.closure_execution.role != expected:
        raise SharedTurbulenceV91Error("v91 v90 role/backend differs")
    placeholder = SharedTurbulenceExecutionV91(
        hles_execution, hles_execution.backend, SEMANTIC_ID, "")
    return SharedTurbulenceExecutionV91(
        hles_execution, hles_execution.backend, SEMANTIC_ID,
        _fingerprint(_execution_payload(placeholder)))


def resolve_shared_turbulence_wetdry_execution_v97(
    hles_execution: HlesProducerExecutionV90,
) -> SharedTurbulenceExecutionV91:
    """Resolve the additive moving-wet/dry state carrier for v97.

    This does not admit the historical v91 CPU midpoint oracle on dry cells.
    It only lets the existing accepted-state/checkpoint hierarchy carry the
    one physical wet/dry hydraulic state; the v97 native masked owner remains
    the sole momentum producer.
    """
    historical = resolve_shared_turbulence_execution_v91(hles_execution)
    placeholder = SharedTurbulenceExecutionV91(
        historical.hles_execution, historical.backend,
        historical.semantic_id, "", STATE_DOMAIN_MOVING_WETDRY_V97)
    return SharedTurbulenceExecutionV91(
        historical.hles_execution, historical.backend,
        historical.semantic_id,
        _fingerprint(_execution_payload(placeholder)),
        STATE_DOMAIN_MOVING_WETDRY_V97)


def execution_document_v91(value: SharedTurbulenceExecutionV91) -> dict:
    if not isinstance(value, SharedTurbulenceExecutionV91):
        raise SharedTurbulenceV91Error("v91 execution object differs")
    payload = _execution_payload(value)
    if value.fingerprint != _fingerprint(payload):
        raise SharedTurbulenceV91Error("v91 execution fingerprint differs")
    return {**payload, "fingerprint": value.fingerprint}


@dataclass(frozen=True)
class ChkvicUsedTotalReceiptV91:
    execution_fingerprint: str
    parent_hles_state_sha256: str
    current_handoff_receipt_sha256: str
    dt_s: float
    vicmax_m2_s: np.ndarray
    dicmax_m2_s: np.ndarray
    raw_total_vicuv_m2_s: np.ndarray
    raw_total_dicuv_m2_s: np.ndarray
    capped_total_vicuv_m2_s: np.ndarray
    capped_total_dicuv_m2_s: np.ndarray
    capped_vicuv_cells: int
    capped_dicuv_cells: int
    receipt_sha256: str


def _cap_token(value: ChkvicUsedTotalReceiptV91) -> str:
    return _fingerprint({
        "scope_id": CAP_SCOPE_ID,
        "execution_fingerprint": value.execution_fingerprint,
        "parent_hles_state_sha256": value.parent_hles_state_sha256,
        "current_handoff_receipt_sha256":
            value.current_handoff_receipt_sha256,
        "dt_s": value.dt_s,
        "arrays_sha256": [_array_digest(array) for array in (
            value.vicmax_m2_s, value.dicmax_m2_s,
            value.raw_total_vicuv_m2_s, value.raw_total_dicuv_m2_s,
            value.capped_total_vicuv_m2_s,
            value.capped_total_dicuv_m2_s)],
        "capped_vicuv_cells": value.capped_vicuv_cells,
        "capped_dicuv_cells": value.capped_dicuv_cells,
        "strict_comparison": ">",
        "total_before_cap": True,
        "cap_application_count": 1,
        "dicuv_cap_relation": "dicmax=0.7*vicmax",
    })


def _chkvic_limits(
    execution: SharedTurbulenceExecutionV91, dt_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    dt = _finite(dt_s, "chkvic timestep", positive=True)
    widths = execution.base_execution.base_execution.rectangle_width_m
    dx, dy = widths[0], widths[1]
    inverse_metric = 1.0/(dx*dx)+1.0/(dy*dy)
    # Locked source: vicmax=.9/(4*hdt*metric), with hdt=.5*dt.
    vicmax = CHKVIC_SAFETY/(4.0*CHKVIC_HDT_FRACTION*dt*inverse_metric)
    dicmax = HORIZONTAL_SCHMIDT*vicmax
    return np.ascontiguousarray(vicmax), np.ascontiguousarray(dicmax)


def apply_chkvic_used_total_v91(
    execution: SharedTurbulenceExecutionV91,
    parent_hles: HlesProducerStateV90,
    handoff: CurrentClosureHandoffV90, *, dt_s: float,
) -> ChkvicUsedTotalReceiptV91:
    """Apply the locked strict-``>`` ceiling once to each used total field."""
    execution_document_v91(execution)
    validate_hles_producer_state_v90(parent_hles, execution.hles_execution)
    validate_current_closure_handoff_v90(
        handoff, execution.hles_execution, parent_hles)
    vicmax, dicmax = _chkvic_limits(execution, dt_s)
    raw_vic = _array(handoff.used_total_vicuv_m2_s,
                     (execution.cell_count,), "raw total vicuv",
                     nonnegative=True)
    raw_dic = _array(handoff.used_total_dicuv_m2_s,
                     (execution.cell_count,), "raw total dicuv",
                     nonnegative=True)
    # Preserve exact equality.  Classic dynamic chkvic uses a strict > test.
    capped_vic = np.ascontiguousarray(np.where(raw_vic > vicmax,
                                               vicmax, raw_vic))
    capped_dic = np.ascontiguousarray(np.where(raw_dic > dicmax,
                                               dicmax, raw_dic))
    for value in (vicmax, dicmax, capped_vic, capped_dic):
        value.setflags(write=False)
    placeholder = ChkvicUsedTotalReceiptV91(
        execution.fingerprint, parent_hles.state_sha256,
        handoff.receipt_sha256, float(dt_s), vicmax, dicmax, raw_vic,
        raw_dic, capped_vic, capped_dic,
        int(np.count_nonzero(raw_vic > vicmax)),
        int(np.count_nonzero(raw_dic > dicmax)), "")
    return ChkvicUsedTotalReceiptV91(
        **{**placeholder.__dict__, "receipt_sha256": _cap_token(placeholder)})


def validate_chkvic_receipt_v91(
    value: ChkvicUsedTotalReceiptV91,
    execution: SharedTurbulenceExecutionV91,
    parent_hles: HlesProducerStateV90,
    handoff: CurrentClosureHandoffV90,
) -> None:
    if not isinstance(value, ChkvicUsedTotalReceiptV91):
        raise SharedTurbulenceV91Error(
            "v91 chkvic receipt object differs")
    expected = apply_chkvic_used_total_v91(
        execution, parent_hles, handoff, dt_s=value.dt_s)
    if (value.receipt_sha256 != _cap_token(value) or
            value.receipt_sha256 != expected.receipt_sha256 or
            any(not np.array_equal(left, right) for left, right in zip(
                (value.vicmax_m2_s, value.dicmax_m2_s,
                 value.raw_total_vicuv_m2_s, value.raw_total_dicuv_m2_s,
                 value.capped_total_vicuv_m2_s,
                 value.capped_total_dicuv_m2_s),
                (expected.vicmax_m2_s, expected.dicmax_m2_s,
                 expected.raw_total_vicuv_m2_s,
                 expected.raw_total_dicuv_m2_s,
                 expected.capped_total_vicuv_m2_s,
                 expected.capped_total_dicuv_m2_s)))):
        raise SharedTurbulenceV91Error(
            "v91 chkvic receipt is stale, mutated or input-mismatched")


def momentum_diffusion_rate_v91(
    execution: SharedTurbulenceExecutionV91, q: object,
    capped_vicuv_m2_s: object,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one variable-viscosity velocity-Laplacian momentum rate.

    Physical boundary faces carry a zero normal viscous flux.  Internal face
    viscosity and depth are arithmetic means of the two authenticated cell
    values.  The resulting face momentum flux is scattered antisymmetrically.
    The mapping is conservative in momentum and exactly preserves a
    constant velocity even when depth varies; it does not claim staggered
    Delft momentum-discretisation identity.
    """
    execution_document_v91(execution)
    if execution.state_domain_mode != STATE_DOMAIN_FULLY_WET_V91:
        raise SharedTurbulenceV91Error(
            "v91 historical momentum oracle cannot consume the v97 "
            "moving-wet/dry mode")
    nc, nf = execution.cell_count, execution.face_count
    state = _array(q, (3, nc), "hydraulic midpoint")
    vic = _array(capped_vicuv_m2_s, (nc,), "capped vicuv",
                 nonnegative=True)
    base = execution.base_execution.base_execution
    if np.any(state[0] <= base.dry_tolerance_m):
        raise SharedTurbulenceV91Error(
            "v91 momentum consumer requires the fully wet v90 scope")
    velocity = state[1:]/state[0]
    flux = np.zeros((2, nf), dtype=np.float64)
    for face in range(nf):
        left = int(base.owner[face])
        right = int(base.neighbour[face])
        if right < 0:
            continue
        distance = float(base.centre_distance_m[face])
        if distance <= 0.0:
            raise SharedTurbulenceV91Error(
                "v91 internal face distance must be positive")
        face_depth = 0.5*(state[0, left]+state[0, right])
        coefficient = (face_depth*0.5*(vic[left]+vic[right]) *
                       base.face_length_m[face]/distance)
        flux[:, face] = coefficient*(velocity[:, right]-velocity[:, left])
    rate = np.zeros((3, nc), dtype=np.float64)
    for cell in range(nc):
        divergence = np.zeros(2, dtype=np.float64)
        for slot in range(base.cell_face_offset[cell],
                          base.cell_face_offset[cell+1]):
            face = int(base.cell_faces[slot])
            divergence += base.cell_face_sign[slot]*flux[:, face]
        rate[1:, cell] = divergence/base.cell_area_m2[cell]
    if not np.isfinite(rate).all() or not np.isfinite(flux).all():
        raise SharedTurbulenceV91Error("v91 momentum rate is nonfinite")
    return np.ascontiguousarray(rate), np.ascontiguousarray(flux)


@dataclass(frozen=True)
class SharedTurbulenceStateV91:
    execution_fingerprint: str
    q: np.ndarray
    suspended_state: MultifractionSuspendedStateV78
    accepted_receipt_chain_sha256: str
    state_sha256: str

    @property
    def accepted_updates(self) -> int:
        return self.suspended_state.accepted_updates

    @property
    def time_s(self) -> float:
        return self.suspended_state.time_s


def _state_token(value: SharedTurbulenceStateV91) -> str:
    return _fingerprint({
        "execution_fingerprint": value.execution_fingerprint,
        "q_sha256": _array_digest(value.q),
        "suspended_state_sha256": value.suspended_state.state_sha256,
        "accepted_receipt_chain_sha256":
            value.accepted_receipt_chain_sha256,
    })


def _wrap_state(
    execution: SharedTurbulenceExecutionV91, *, q: object,
    suspended_state: MultifractionSuspendedStateV78, receipt_chain: str,
) -> SharedTurbulenceStateV91:
    execution_document_v91(execution)
    validate_multifraction_suspended_state_v78(
        suspended_state, execution.base_execution)
    q_a = _array(q, (3, execution.cell_count), "accepted hydraulic state")
    dry = execution.base_execution.base_execution.dry_tolerance_m
    if execution.state_domain_mode == STATE_DOMAIN_FULLY_WET_V91:
        if np.any(q_a[0] <= dry):
            raise SharedTurbulenceV91Error(
                "v91 state must remain inside the fully wet scope")
    else:
        dry_cell = q_a[0] <= dry
        suspended = np.asarray(
            suspended_state.suspended_mass_kg_m2)
        if (np.any(q_a[0] < 0.0) or
                np.any(q_a[1:, dry_cell] != 0.0) or
                np.any(suspended[:, dry_cell] != 0.0)):
            raise SharedTurbulenceV91Error(
                "v97 dry state requires nonnegative depth and exact-zero "
                "momentum/suspended inventory")
    chain = _sha(receipt_chain, "accepted receipt chain")
    placeholder = SharedTurbulenceStateV91(
        execution.fingerprint, q_a, suspended_state, chain, "")
    return SharedTurbulenceStateV91(
        execution.fingerprint, q_a, suspended_state, chain,
        _state_token(placeholder))


def resolve_shared_turbulence_state_v91(
    execution: SharedTurbulenceExecutionV91, *,
    hles_state: HlesProducerStateV90,
    suspended_state: MultifractionSuspendedStateV78, q: object,
) -> SharedTurbulenceStateV91:
    validate_hles_producer_state_v90(hles_state, execution.hles_execution)
    if not hles_state.primed:
        raise SharedTurbulenceV91Error("v91 requires the v90 cold prime")
    if (hles_state.accepted_updates != suspended_state.accepted_updates or
            hles_state.time_s != suspended_state.time_s):
        raise SharedTurbulenceV91Error(
            "v91 HLES and scalar accepted clocks differ")
    chain = _fingerprint({
        "scope_id": SCOPE_ID,
        "execution_fingerprint": execution.fingerprint,
        "hles_state_sha256": hles_state.state_sha256,
        "suspended_state_sha256": suspended_state.state_sha256,
        "cold": True,
    })
    return _wrap_state(execution, q=q, suspended_state=suspended_state,
                       receipt_chain=chain)


def validate_shared_turbulence_state_v91(
    state: SharedTurbulenceStateV91,
    execution: SharedTurbulenceExecutionV91,
) -> None:
    if (not isinstance(state, SharedTurbulenceStateV91) or
            state.execution_fingerprint != execution.fingerprint or
            state.state_sha256 != _state_token(state)):
        raise SharedTurbulenceV91Error(
            "v91 state is stale, mutated or execution-mismatched")
    _wrap_state(execution, q=state.q,
                suspended_state=state.suspended_state,
                receipt_chain=state.accepted_receipt_chain_sha256)


@dataclass(frozen=True)
class HydraulicTurbulenceReceiptV91:
    execution_fingerprint: str
    parent_state_sha256: str
    parent_hles_state_sha256: str
    current_handoff_receipt_sha256: str
    chkvic_receipt_sha256: str
    backend: str
    actual_implicit_controller: bool
    dt_s: float
    midpoint_q_sha256: str
    accepted_q_sha256: str
    residual_evaluation_count: int
    momentum_rate_evaluation_count: int
    momentum_consumer_count: int
    duplicate_post_source_count: int
    base_step_receipt: dict
    receipt_sha256: str


def _hydraulic_token(value: HydraulicTurbulenceReceiptV91) -> str:
    return _fingerprint({
        "scope_id": HYDRAULIC_RECEIPT_SCOPE_ID,
        "execution_fingerprint": value.execution_fingerprint,
        "parent_state_sha256": value.parent_state_sha256,
        "parent_hles_state_sha256": value.parent_hles_state_sha256,
        "current_handoff_receipt_sha256":
            value.current_handoff_receipt_sha256,
        "chkvic_receipt_sha256": value.chkvic_receipt_sha256,
        "backend": value.backend,
        "actual_implicit_controller": value.actual_implicit_controller,
        "dt_s": value.dt_s,
        "midpoint_q_sha256": value.midpoint_q_sha256,
        "accepted_q_sha256": value.accepted_q_sha256,
        "residual_evaluation_count": value.residual_evaluation_count,
        "momentum_rate_evaluation_count":
            value.momentum_rate_evaluation_count,
        "momentum_consumer_count": value.momentum_consumer_count,
        "duplicate_post_source_count": value.duplicate_post_source_count,
        "base_step_receipt_fingerprint":
            value.base_step_receipt.get("fingerprint"),
    })


def make_hydraulic_turbulence_receipt_v91(
    execution: SharedTurbulenceExecutionV91,
    parent: SharedTurbulenceStateV91,
    parent_hles: HlesProducerStateV90,
    handoff: CurrentClosureHandoffV90,
    cap: ChkvicUsedTotalReceiptV91, *, midpoint_q: object,
    accepted_q: object, dt_s: float, residual_evaluation_count: int,
    momentum_rate_evaluation_count: int, backend: str,
    actual_implicit_controller: bool,
) -> HydraulicTurbulenceReceiptV91:
    validate_shared_turbulence_state_v91(parent, execution)
    validate_hles_producer_state_v90(parent_hles, execution.hles_execution)
    validate_current_closure_handoff_v90(
        handoff, execution.hles_execution, parent_hles)
    validate_chkvic_receipt_v91(
        cap, execution, parent_hles, handoff)
    if (parent.accepted_updates != parent_hles.accepted_updates or
            parent.time_s != parent_hles.time_s):
        raise SharedTurbulenceV91Error(
            "v91 hydraulic/HLES parent clocks differ")
    dt = _finite(dt_s, "hydraulic timestep", positive=True)
    mid = _array(midpoint_q, (3, execution.cell_count), "accepted midpoint")
    accepted = _array(
        accepted_q, (3, execution.cell_count), "accepted endpoint")
    if (type(residual_evaluation_count) is not int or
            type(momentum_rate_evaluation_count) is not int or
            residual_evaluation_count < 1 or
            momentum_rate_evaluation_count != residual_evaluation_count):
        raise SharedTurbulenceV91Error(
            "v91 every nonlinear residual must consume momentum once")
    expected_backend = ("cpu_oracle" if execution.backend == BACKEND_CPU
                        else "actual_cuda")
    if backend != expected_backend:
        raise SharedTurbulenceV91Error("v91 receipt backend differs")
    if bool(actual_implicit_controller) != (backend == "actual_cuda"):
        raise SharedTurbulenceV91Error(
            "v91 actual-controller evidence/backend differs")
    step_sha = _fingerprint({
        "scope_id": HYDRAULIC_RECEIPT_SCOPE_ID,
        "parent": parent.state_sha256,
        "handoff": handoff.receipt_sha256,
        "cap": cap.receipt_sha256,
        "midpoint": _array_digest(mid),
        "accepted": _array_digest(accepted),
        "residual_evaluations": residual_evaluation_count,
        "backend": backend,
    })
    step = make_implicit_fwave_midpoint_step_receipt_v78(
        hydraulic_execution_sha256=
            execution.base_execution.hydraulic_execution_sha256,
        accepted_step_sha256=step_sha,
        accepted_update=parent.accepted_updates+1,
        time_before_s=parent.time_s, dt_s=dt)
    placeholder = HydraulicTurbulenceReceiptV91(
        execution.fingerprint, parent.state_sha256,
        parent_hles.state_sha256, handoff.receipt_sha256,
        cap.receipt_sha256, backend, bool(actual_implicit_controller), dt,
        _array_digest(mid), _array_digest(accepted),
        residual_evaluation_count, momentum_rate_evaluation_count,
        1, 0, step, "")
    return HydraulicTurbulenceReceiptV91(
        **{**placeholder.__dict__,
           "receipt_sha256": _hydraulic_token(placeholder)})


def validate_hydraulic_turbulence_receipt_v91(
    value: HydraulicTurbulenceReceiptV91,
    execution: SharedTurbulenceExecutionV91,
    parent: SharedTurbulenceStateV91,
    parent_hles: HlesProducerStateV90,
    handoff: CurrentClosureHandoffV90,
    cap: ChkvicUsedTotalReceiptV91,
) -> None:
    if (not isinstance(value, HydraulicTurbulenceReceiptV91) or
            value.execution_fingerprint != execution.fingerprint or
            value.parent_state_sha256 != parent.state_sha256 or
            value.parent_hles_state_sha256 != parent_hles.state_sha256 or
            value.current_handoff_receipt_sha256 != handoff.receipt_sha256 or
            value.chkvic_receipt_sha256 != cap.receipt_sha256 or
            value.momentum_consumer_count != 1 or
            value.duplicate_post_source_count != 0 or
            value.momentum_rate_evaluation_count !=
                value.residual_evaluation_count or
            value.residual_evaluation_count < 1 or
            value.receipt_sha256 != _hydraulic_token(value)):
        raise SharedTurbulenceV91Error(
            "v91 hydraulic receipt is stale, mutated or mismatched")
    validate_implicit_fwave_midpoint_step_receipt_v78(
        value.base_step_receipt,
        hydraulic_execution_sha256=
            execution.base_execution.hydraulic_execution_sha256,
        parent_accepted_updates=parent.accepted_updates,
        parent_time_s=parent.time_s, dt_s=value.dt_s)


@dataclass(frozen=True)
class SharedTurbulenceTrialV91:
    execution_fingerprint: str
    parent_state_sha256: str
    cap_receipt: ChkvicUsedTotalReceiptV91
    hydraulic_receipt: HydraulicTurbulenceReceiptV91
    scalar_execution: SuspendedSupportExecutionV82
    scalar_trial: SuspendedSupportTrialV82
    midpoint_q: np.ndarray
    momentum_rate_m2_s2: np.ndarray
    momentum_face_flux_m4_s2: np.ndarray
    candidate: SharedTurbulenceStateV91
    scalar_consumer_count: int
    backend: str
    trial_sha256: str


def _trial_token(value: SharedTurbulenceTrialV91) -> str:
    return _fingerprint({
        "execution_fingerprint": value.execution_fingerprint,
        "parent_state_sha256": value.parent_state_sha256,
        "cap_receipt_sha256": value.cap_receipt.receipt_sha256,
        "hydraulic_receipt_sha256": value.hydraulic_receipt.receipt_sha256,
        "scalar_execution_fingerprint": value.scalar_execution.fingerprint,
        "scalar_trial_sha256": value.scalar_trial.trial_sha256,
        "arrays_sha256": [_array_digest(value.midpoint_q),
                           _array_digest(value.momentum_rate_m2_s2),
                           _array_digest(value.momentum_face_flux_m4_s2)],
        "candidate_state_sha256": value.candidate.state_sha256,
        "scalar_consumer_count": value.scalar_consumer_count,
        "backend": value.backend,
    })


def _wrap_trial(
    execution: SharedTurbulenceExecutionV91,
    parent: SharedTurbulenceStateV91,
    cap: ChkvicUsedTotalReceiptV91,
    hydraulic: HydraulicTurbulenceReceiptV91,
    scalar_execution: SuspendedSupportExecutionV82,
    scalar_trial: SuspendedSupportTrialV82, *, midpoint_q: object,
    momentum_rate: object, momentum_flux: object, accepted_q: object,
    backend: str,
) -> SharedTurbulenceTrialV91:
    mid = _array(midpoint_q, (3, execution.cell_count), "trial midpoint")
    rate = _array(momentum_rate, (3, execution.cell_count),
                  "trial momentum rate")
    flux = _array(momentum_flux, (2, execution.face_count),
                  "trial momentum face flux")
    candidate_chain = _fingerprint({
        "previous": parent.accepted_receipt_chain_sha256,
        "hydraulic": hydraulic.receipt_sha256,
        "scalar": scalar_trial.trial_sha256,
    })
    candidate = _wrap_state(
        execution, q=accepted_q,
        suspended_state=scalar_trial.candidate.base_state,
        receipt_chain=candidate_chain)
    placeholder = SharedTurbulenceTrialV91(
        execution.fingerprint, parent.state_sha256, cap, hydraulic,
        scalar_execution, scalar_trial, mid, rate, flux, candidate, 1,
        backend, "")
    return SharedTurbulenceTrialV91(
        **{**placeholder.__dict__, "trial_sha256": _trial_token(placeholder)})


def begin_shared_turbulence_cpu_trial_v91(
    execution: SharedTurbulenceExecutionV91,
    parent: SharedTurbulenceStateV91,
    parent_hles: HlesProducerStateV90,
    handoff: CurrentClosureHandoffV90, *,
    augmented_fwave_base_rate_m2_s2: object,
    dt_s: float, cancelled: bool = False,
    nonlinear_rtol: float = 1.0e-13,
    nonlinear_atol: float = 1.0e-14,
    nonlinear_maxiter: int = 256,
    **scalar_inputs,
) -> SharedTurbulenceTrialV91:
    """Run the CPU implicit-midpoint oracle and one complete v82 lifecycle."""
    if execution.backend != BACKEND_CPU:
        raise SharedTurbulenceV91Error(
            "v91 CPU trial requires the explicit CPU execution")
    validate_shared_turbulence_state_v91(parent, execution)
    validate_hles_producer_state_v90(parent_hles, execution.hles_execution)
    validate_current_closure_handoff_v90(
        handoff, execution.hles_execution, parent_hles)
    if cancelled:
        raise SharedTurbulenceV91Error(
            "v91 CPU trial cancelled before any consumer")
    if parent.accepted_updates >= MAX_UPDATES:
        raise SharedTurbulenceV91Error("v91 accepted update limit was reached")
    if (parent.accepted_updates != parent_hles.accepted_updates or
            parent.time_s != parent_hles.time_s):
        raise SharedTurbulenceV91Error("v91 parent clocks differ")
    dt = _finite(dt_s, "CPU timestep", positive=True)
    base_rate = _array(
        augmented_fwave_base_rate_m2_s2, (3, execution.cell_count),
        "frozen augmented-fwave base rate")
    rtol = _finite(nonlinear_rtol, "CPU nonlinear rtol", positive=True)
    atol = _finite(nonlinear_atol, "CPU nonlinear atol", positive=True)
    if (type(nonlinear_maxiter) is not int or nonlinear_maxiter < 1 or
            rtol >= 1.0 or atol >= 1.0):
        raise SharedTurbulenceV91Error("v91 CPU nonlinear controls differ")
    cap = apply_chkvic_used_total_v91(
        execution, parent_hles, handoff, dt_s=dt)
    midpoint = np.array(parent.q+0.5*dt*base_rate, copy=True)
    residual_evaluations = 0
    converged = False
    rate = flux = None
    for _ in range(nonlinear_maxiter):
        rate, flux = momentum_diffusion_rate_v91(
            execution, midpoint, cap.capped_total_vicuv_m2_s)
        residual_evaluations += 1
        candidate_midpoint = parent.q+0.5*dt*(base_rate+rate)
        defect = float(np.linalg.norm(candidate_midpoint-midpoint))
        scale = max(1.0, float(np.linalg.norm(candidate_midpoint)))
        midpoint = candidate_midpoint
        if defect <= max(atol, rtol*scale):
            converged = True
            break
    if not converged or rate is None or flux is None:
        raise SharedTurbulenceV91Error(
            "v91 CPU implicit midpoint did not converge")
    # Reevaluate at the accepted midpoint; this is the true residual rate.
    rate, flux = momentum_diffusion_rate_v91(
        execution, midpoint, cap.capped_total_vicuv_m2_s)
    residual_evaluations += 1
    accepted_q = np.ascontiguousarray(parent.q+dt*(base_rate+rate))
    hydraulic = make_hydraulic_turbulence_receipt_v91(
        execution, parent, parent_hles, handoff, cap,
        midpoint_q=midpoint, accepted_q=accepted_q, dt_s=dt,
        residual_evaluation_count=residual_evaluations,
        momentum_rate_evaluation_count=residual_evaluations,
        backend="cpu_oracle", actual_implicit_controller=False)
    # These scalar cross-fields belong to the hydraulic transaction, not to
    # the caller.  The actual-CUDA owner applies the same override after its
    # accepted endpoint readback.  Keeping the CPU oracle authoritative here
    # prevents a valid hydraulic receipt from being paired with pre-step or
    # otherwise unrelated depths/momentum in the v82 consumer.
    scalar_values = dict(scalar_inputs)
    scalar_values.update(
        depth_before_m=np.array(parent.q[0], copy=True),
        depth_after_m=np.array(accepted_q[0], copy=True),
        endpoint_momentum_m2_s=np.array(accepted_q[1:], copy=True))
    scalar_execution = resolve_suspended_support_execution_v82(
        execution.base_execution,
        dicuv_cell_m2_s=cap.capped_total_dicuv_m2_s)
    scalar_parent = _wrap_v82_state(
        scalar_execution, parent.suspended_state)
    try:
        scalar_trial = begin_suspended_support_cpu_trial_v82(
            scalar_execution, scalar_parent,
            accepted_step_receipt=hydraulic.base_step_receipt,
            dt_s=dt, **scalar_values)
    except (TypeError, ValueError) as error:
        raise SharedTurbulenceV91Error(
            "v91 CPU scalar consumer rejected the trial") from error
    return _wrap_trial(
        execution, parent, cap, hydraulic, scalar_execution, scalar_trial,
        midpoint_q=midpoint, momentum_rate=rate, momentum_flux=flux,
        accepted_q=accepted_q, backend="cpu_oracle")


def _validate_trial(
    execution: SharedTurbulenceExecutionV91,
    parent: SharedTurbulenceStateV91,
    parent_hles: HlesProducerStateV90,
    handoff: CurrentClosureHandoffV90,
    trial: SharedTurbulenceTrialV91,
) -> None:
    validate_shared_turbulence_state_v91(parent, execution)
    if (not isinstance(trial, SharedTurbulenceTrialV91) or
            trial.execution_fingerprint != execution.fingerprint or
            trial.parent_state_sha256 != parent.state_sha256 or
            trial.scalar_consumer_count != 1 or
            trial.trial_sha256 != _trial_token(trial)):
        raise SharedTurbulenceV91Error(
            "v91 trial is stale, mutated or belongs to another parent")
    validate_chkvic_receipt_v91(
        trial.cap_receipt, execution, parent_hles, handoff)
    validate_hydraulic_turbulence_receipt_v91(
        trial.hydraulic_receipt, execution, parent, parent_hles, handoff,
        trial.cap_receipt)
    _validate_v82_trial(
        trial.scalar_execution,
        _wrap_v82_state(trial.scalar_execution, parent.suspended_state),
        trial.scalar_trial)
    expected_rate, expected_flux = momentum_diffusion_rate_v91(
        execution, trial.midpoint_q,
        trial.cap_receipt.capped_total_vicuv_m2_s)
    if trial.backend == "cpu_oracle":
        momentum_equal = (
            np.array_equal(expected_rate, trial.momentum_rate_m2_s2) and
            np.array_equal(expected_flux, trial.momentum_face_flux_m4_s2))
    elif trial.backend == "actual_cuda":
        momentum_equal = (
            np.allclose(expected_rate, trial.momentum_rate_m2_s2,
                        rtol=2.e-15, atol=2.e-16) and
            np.allclose(expected_flux, trial.momentum_face_flux_m4_s2,
                        rtol=2.e-15, atol=2.e-16))
    else:
        momentum_equal = False
    # Reconstruct the live-controller receipt from the trial's authoritative
    # arrays.  Merely re-fingerprinting mutually inconsistent receipt fields
    # must not authenticate a cross-field forgery.
    expected_hydraulic = make_hydraulic_turbulence_receipt_v91(
        execution, parent, parent_hles, handoff, trial.cap_receipt,
        midpoint_q=trial.midpoint_q, accepted_q=trial.candidate.q,
        dt_s=trial.hydraulic_receipt.dt_s,
        residual_evaluation_count=
            trial.hydraulic_receipt.residual_evaluation_count,
        momentum_rate_evaluation_count=
            trial.hydraulic_receipt.momentum_rate_evaluation_count,
        backend=trial.backend,
        actual_implicit_controller=(trial.backend == "actual_cuda"))
    if (not momentum_equal or
            expected_hydraulic.receipt_sha256 !=
                trial.hydraulic_receipt.receipt_sha256):
        raise SharedTurbulenceV91Error(
            "v91 trial momentum cross-fields differ")
    validate_shared_turbulence_state_v91(trial.candidate, execution)


def commit_shared_turbulence_trial_v91(
    execution: SharedTurbulenceExecutionV91,
    parent: SharedTurbulenceStateV91,
    parent_hles: HlesProducerStateV90,
    handoff: CurrentClosureHandoffV90,
    trial: SharedTurbulenceTrialV91,
) -> SharedTurbulenceStateV91:
    _validate_trial(execution, parent, parent_hles, handoff, trial)
    scalar_parent = _wrap_v82_state(
        trial.scalar_execution, parent.suspended_state)
    scalar = commit_suspended_support_trial_v82(
        trial.scalar_execution, scalar_parent, trial.scalar_trial)
    if scalar.base_state.state_sha256 != \
            trial.candidate.suspended_state.state_sha256:
        raise SharedTurbulenceV91Error("v91 committed scalar state differs")
    return trial.candidate


def rollback_shared_turbulence_trial_v91(
    execution: SharedTurbulenceExecutionV91,
    parent: SharedTurbulenceStateV91,
    parent_hles: HlesProducerStateV90,
    handoff: CurrentClosureHandoffV90,
    trial: SharedTurbulenceTrialV91,
) -> SharedTurbulenceStateV91:
    _validate_trial(execution, parent, parent_hles, handoff, trial)
    scalar_parent = _wrap_v82_state(
        trial.scalar_execution, parent.suspended_state)
    rolled = rollback_suspended_support_trial_v82(
        trial.scalar_execution, scalar_parent, trial.scalar_trial)
    if rolled is not scalar_parent:
        raise SharedTurbulenceV91Error("v91 scalar rollback differs")
    return parent


def cancel_shared_turbulence_trial_v91(*args, **kwargs):
    return rollback_shared_turbulence_trial_v91(*args, **kwargs)


def shared_turbulence_checkpoint_v91(
    execution: SharedTurbulenceExecutionV91,
    state: SharedTurbulenceStateV91,
) -> dict:
    validate_shared_turbulence_state_v91(state, execution)
    body = {
        "schema": SCHEMA,
        "scope_id": CHECKPOINT_SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "execution_fingerprint": execution.fingerprint,
        "integrator": INTEGRATOR,
        "q": state.q.tolist(),
        "base_v78_checkpoint": multifraction_checkpoint_v78(
            execution.base_execution, state.suspended_state),
        "accepted_receipt_chain_sha256":
            state.accepted_receipt_chain_sha256,
        "state_sha256": state.state_sha256,
    }
    return {**body, "fingerprint": _fingerprint(body)}


def load_shared_turbulence_checkpoint_v91(
    execution: SharedTurbulenceExecutionV91,
    document: Mapping[str, object],
) -> SharedTurbulenceStateV91:
    expected = {
        "schema", "scope_id", "semantic_id", "execution_fingerprint",
        "integrator", "q", "base_v78_checkpoint",
        "accepted_receipt_chain_sha256", "state_sha256", "fingerprint",
    }
    if not isinstance(document, Mapping) or set(document) != expected:
        raise SharedTurbulenceV91Error("v91 checkpoint inventory differs")
    body = {key: document[key] for key in document if key != "fingerprint"}
    if (document["fingerprint"] != _fingerprint(body) or
            document["schema"] != SCHEMA or
            document["scope_id"] != CHECKPOINT_SCOPE_ID or
            document["semantic_id"] != SEMANTIC_ID or
            document["execution_fingerprint"] != execution.fingerprint or
            document["integrator"] != INTEGRATOR):
        raise SharedTurbulenceV91Error("v91 checkpoint identity differs")
    try:
        suspended = load_multifraction_checkpoint_v78(
            execution.base_execution, document["base_v78_checkpoint"])
        state = _wrap_state(
            execution, q=document["q"], suspended_state=suspended,
            receipt_chain=document["accepted_receipt_chain_sha256"])
    except (TypeError, ValueError) as error:
        raise SharedTurbulenceV91Error("v91 checkpoint state is invalid") \
            from error
    if state.state_sha256 != document["state_sha256"]:
        raise SharedTurbulenceV91Error("v91 checkpoint state digest differs")
    return state


LOCKED_SOURCES = {
    "resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/"
    "flow2d3d_manager/src/tricom_init.F90":
        "18ce87c3622a3f0a5ea594049d688d893765957dd99c8d02d2697682bd2db8ad",
    "resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/"
    "flow2d3d_kernel/src/inichk/chkvic.f90":
        "17cc4d2d6287769edb5b44fee9d88749811da9c740cfe04a16c9efb6bc222200",
    "resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/"
    "flow2d3d_kernel/src/main/trisol.f90":
        "4b3e7ce63aed6c0cf95fe20ea3b9aa80fee9b3cd6fdf8101f0fbf7af5ca689e0",
    "resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/"
    "flow2d3d_kernel/src/compute/turclo.f90":
        "f91385a371000ae1d5c550dcb18fa617563379c1050ceaa31b37217d7b75270f",
    "resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/"
    "flow2d3d_kernel/src/compute/uzd.f90":
        "8c1a2eb0f2d7208d3d3659368b8ff2902bd82b493f2cf76fb82fce1fae0932c2",
    "resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/"
    "flow2d3d_kernel/src/compute/vihrov.f90":
        "faaf5868ca814f58d9e116d3c64679a38903b0887c23cd14e66a65a93ed73fd5",
    "resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/"
    "flow2d3d_kernel/src/compute/difu.f90":
        "e8e11d438a6e3aa8e912ce1133efdec093b5bad60605aefa6c4b037ea6aa5c17",
    "resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/"
    "flow2d3d_kernel/src/compute/difuflux.f90":
        "a37d207110427a211c6b02de5a1dca94902a4e0299bbc724bea1b228861491c9",
}


def verify_locked_sources_v91(root: str | Path) -> dict[str, str]:
    root = Path(root)
    result = {}
    for relative, expected in LOCKED_SOURCES.items():
        actual = hashlib.sha256((root/relative).read_bytes()).hexdigest()
        if actual != expected:
            raise SharedTurbulenceV91Error(
                "v91 locked source hash differs: " + relative)
        result[relative] = actual
    return result


def method_identity_v91() -> dict:
    return {
        "scope_id": SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "integrator": INTEGRATOR,
        "backend_roles": ["cpu_oracle", "actual_cuda"],
        "mapping_scope": "uniform_axis_aligned_cartesian_fixed_grid",
        "chkvic": "used_total_once_strict_greater_than",
        "chkvic_hdt_fraction_of_dt": CHKVIC_HDT_FRACTION,
        "momentum_operator": MOMENTUM_OPERATOR_ID,
        "cpu_midpoint_oracle": CPU_ORACLE_ID,
        "gpu_true_residual_hook": True,
        "scalar_consumer": "v82_dicuv_divided_by_0p7_once",
        "owner_counts": {
            "chkvic": 1,
            "momentum_consumer": 1,
            "scalar_consumer": 1,
            "duplicate_post_source": 0,
        },
        "accepted_update_limit": MAX_UPDATES,
        "automatic_cpu_fallback": False,
        "product_admission": False,
        "not_claimed": [
            "Delft_staggered_momentum_discretisation_identity",
            "moving_wet_dry", "cut_cells", "general_AMR",
            "full_v96_atomic_solver_transaction", "long_run_equivalence",
        ],
        "delft3d_source_modified": False,
    }


__all__ = [
    "CAP_SCOPE_ID", "CHECKPOINT_SCOPE_ID", "CHKVIC_HDT_FRACTION",
    "CHKVIC_SAFETY", "CPU_ORACLE_ID", "HORIZONTAL_SCHMIDT",
    "HYDRAULIC_RECEIPT_SCOPE_ID", "HydraulicTurbulenceReceiptV91",
    "INTEGRATOR", "MAX_UPDATES", "MOMENTUM_OPERATOR_ID", "SCHEMA",
    "SCOPE_ID", "SEMANTIC_ID", "STATE_DOMAIN_FULLY_WET_V91",
    "STATE_DOMAIN_MOVING_WETDRY_V97", "STATE_DOMAIN_MODES",
    "ChkvicUsedTotalReceiptV91",
    "SharedTurbulenceExecutionV91", "SharedTurbulenceStateV91",
    "SharedTurbulenceTrialV91", "SharedTurbulenceV91Error",
    "apply_chkvic_used_total_v91", "begin_shared_turbulence_cpu_trial_v91",
    "cancel_shared_turbulence_trial_v91",
    "commit_shared_turbulence_trial_v91", "execution_document_v91",
    "load_shared_turbulence_checkpoint_v91",
    "make_hydraulic_turbulence_receipt_v91", "method_identity_v91",
    "momentum_diffusion_rate_v91", "resolve_shared_turbulence_execution_v91",
    "resolve_shared_turbulence_wetdry_execution_v97",
    "resolve_shared_turbulence_state_v91",
    "rollback_shared_turbulence_trial_v91",
    "shared_turbulence_checkpoint_v91", "validate_chkvic_receipt_v91",
    "validate_hydraulic_turbulence_receipt_v91",
    "validate_shared_turbulence_state_v91", "verify_locked_sources_v91",
]
