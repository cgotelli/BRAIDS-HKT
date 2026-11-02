"""Atomic actual-CUDA composition of v95 hydraulics and v94b physics.

This is the deliberately bounded v96 transaction.  The accepted v94/v94b
state is the sole hydraulic/scalar/bed parent.  A typed receipt derives the
accepted mixture density directly from that state (never through a reminted
v78 handoff), one composite residual hook adds the accepted lagged v91
closure and the v85 density force to every true augmented-f-wave residual,
and the converged endpoint/face flux is consumed unchanged by one v94b
scalar/Q3/Exner transaction.

The admitted fixture is fixed-grid, fully wet and closed-boundary.  Accepted
bed mass drives the next-step hydraulic bed, accepted transported scalars drive
the spatial water EOS, and one lagged v91 closure is applied in every true
implicit residual.  Open boundaries, wet/dry and AMR remain explicit later
v96/v97 work.  Nothing here changes Delft3D.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Callable, Mapping

import numpy as np

from braids_core.sediment.accepted_boundary_scalar_state_cpu_v96 import (
    AcceptedBoundaryScalarStateV96,
    accepted_boundary_scalar_checkpoint_v96,
    load_accepted_boundary_scalar_checkpoint_v96,
    make_accepted_boundary_scalar_state_from_v92_trial_v96,
    make_boundary_density_from_accepted_scalar_state_v96,
    validate_accepted_boundary_scalar_state_v96,
)
from braids_core.sediment.coupled_open_boundary_native_cpu_v96 import (
    AcceptedBoundaryDensityReceiptV96,
    validate_accepted_boundary_density_receipt_v96,
)
from braids_core.sediment.active_suspended_control_v94 import (
    ActiveSuspendedControlExecutionV94,
    ActiveSuspendedControlStateV94,
    ActiveSuspendedControlTrialV94,
    SEMANTIC_ID as SEMANTIC_ID_V94,
)
from braids_core.sediment.active_suspended_control_v94b import (
    SEMANTIC_ID_V94B,
    execution_document_v94b,
    validate_active_suspended_control_state_v94b,
)
from braids_core.sediment.densin_fwave_v85 import (
    delft_kmax1_face_acceleration_v85,
    mixture_density_kg_m3_v85,
    project_face_acceleration_v85,
)
from braids_core.sediment.hles_producer_v90 import (
    CurrentClosureHandoffV90,
    HlesProducerStateV90,
    HlesProducerTrialV90,
    make_current_closure_handoff_v90,
    make_hles_input_receipt_v90,
    validate_hles_producer_state_v90,
)
from braids_core.sediment.hydraulic_boundary_trace_cpu_v96 import (
    HydraulicBoundaryTraceExecutionV96,
    HydraulicBoundaryTraceProviderV96,
    HydraulicBoundaryTraceReceiptV96,
    execution_document_hydraulic_boundary_trace_v96,
    validate_hydraulic_boundary_trace_receipt_v96,
)
from braids_core.sediment.suspended_exner_owner_v93 import (
    SUSPENDED_VECTOR_CONSTRUCTED_V93,
    SUSPENDED_VECTOR_EXACT_ZERO_V93,
)
from braids_gpu.active_suspended_control_v94 import (
    ResidentActiveSuspendedControlLifecycleV94,
    ResidentQ3V83ActiveExnerAdapterV94,
)
from braids_gpu.active_suspended_control_v94b import (
    ResidentActiveSuspendedControlLifecycleV94b,
)
from braids_gpu.densin_residual_v95 import (
    BACKEND as V95_BACKEND,
    DensInResidualGPUExecutionV95,
    DensInResidualGPUV95Error,
    ResidentDensInResidualV95,
    _digest as _v95_digest,
    _manning_half_step_oracle,
    execution_document_v95,
)
from braids_gpu.hles_producer_v90 import ResidentHlesProducerLifecycleV90
from braids_gpu.coupled_open_boundary_v96 import (
    CoupledOpenBoundaryGPUExecutionV96,
    ResidentOpenBoundaryDensInGPUV96,
    execution_document_coupled_open_boundary_gpu_v96,
)


SCHEMA = 1
SCOPE_ID = "braids-q4-coupled-densin-gpu-v96-v1"
SEMANTIC_ID = SCOPE_ID + ":atomic-live-v95-v94b-owner"
CHECKPOINT_SCOPE_ID = SCOPE_ID + ":checkpoint"
BACKEND = "actual_cuda"
MAX_ACCEPTED_UPDATES = 4
GEOMETRY_POLICY = "preserve_h_shift_eta_after_exner"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ROOT = Path(__file__).resolve().parents[1]


class CoupledDensInGPUV96Error(ValueError):
    """The bounded actual-CUDA v96 execution or transaction is invalid."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError, OverflowError) as error:
        raise CoupledDensInGPUV96Error(
            "v96 CUDA value is not finite canonical JSON") from error


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
        raise CoupledDensInGPUV96Error(
            f"v96 CUDA {name} must be a lowercase SHA-256")
    return value


def _finite(value: object, name: str, *, positive: bool = False) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(float(value))):
        raise CoupledDensInGPUV96Error(
            f"v96 CUDA {name} must be finite numeric data")
    result = float(value)
    if positive and result <= 0.0:
        raise CoupledDensInGPUV96Error(
            f"v96 CUDA {name} must be positive")
    return result


def _array(value: object, shape: tuple[int, ...], name: str, *,
           positive: bool = False, nonnegative: bool = False,
           integer: bool = False) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != shape or raw.dtype.kind not in "fiub":
        raise CoupledDensInGPUV96Error(
            f"v96 CUDA {name} must be numeric with shape {shape}")
    result = np.ascontiguousarray(
        raw, dtype=np.int32 if integer else np.float64)
    if not np.isfinite(result).all():
        raise CoupledDensInGPUV96Error(f"v96 CUDA {name} must be finite")
    if positive and np.any(result <= 0.0):
        raise CoupledDensInGPUV96Error(f"v96 CUDA {name} must be positive")
    if nonnegative and np.any(result < 0.0):
        raise CoupledDensInGPUV96Error(
            f"v96 CUDA {name} must be nonnegative")
    result.setflags(write=False)
    return result


def _is_zero_datum_flat_bed(value: np.ndarray) -> bool:
    """Admit a zero flat bed without pretending FP layer sums are exact.

    The live bed is reconstructed as ``rock + sum(layer thickness)``.  Even
    when both inputs describe an exact zero datum, that sum may retain a few
    roundoff bits.  Keep those exact live bytes in the execution fingerprint,
    while limiting this first v96 stage to roundoff-flat, zero-datum beds.
    """
    bed = np.asarray(value, dtype=np.float64)
    tolerance = 64.0 * np.finfo(np.float64).eps
    return bool(
        np.max(np.abs(bed), initial=0.0) <= tolerance and
        np.ptp(bed) <= tolerance
    )


def _cancelled(value: bool | Callable[[], bool]) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if not callable(value):
        raise CoupledDensInGPUV96Error(
            "v96 CUDA cancellation must be Boolean or callable")
    result = value()
    if not isinstance(result, (bool, np.bool_)):
        raise CoupledDensInGPUV96Error(
            "v96 CUDA cancellation callback must return Boolean data")
    return bool(result)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CoupledDensInGPUV96Error(message)


def _v92_execution(execution: ActiveSuspendedControlExecutionV94):
    return execution.base_execution.base_execution


def _v92_state(state: ActiveSuspendedControlStateV94):
    return state.base_state.base_state


def _grid_execution(execution: ActiveSuspendedControlExecutionV94):
    return _v92_execution(execution).base_execution.base_execution.base_execution


def _provider_is_closed(provider: object) -> bool:
    binding = getattr(provider, "binding", None)
    document = getattr(binding, "boundary_execution", None)
    if document is None:
        document = getattr(getattr(binding, "core", None),
                           "boundary_execution", None)
    if not isinstance(document, Mapping):
        return False
    rows = document.get("boundaries")
    return bool(isinstance(rows, list) and rows and all(
        isinstance(row, Mapping) and
        row.get("public_mode_id") == "bed_transport.closed" and
        row.get("internal_mode") == 0 and
        all(float(item) == 0.0 for item in row.get("source_rates", (1.0,)))
        for row in rows))


def _portable_v95_runtime_identity(runtime: ResidentDensInResidualV95) -> str:
    """Bind v95 code bytes without binding an extraction-directory name.

    The authenticated controller can be copied into an isolated runtime
    directory; v95 correctly records that resolved path for live provenance.
    A restart on a fresh owner must instead match the compiled source bytes,
    device/runtime inventory, and all nested receipts.  The path string is not
    numerical state and is intentionally omitted only from this outer token.
    """
    runtime._validate_runtime_identity(require_controller=True)
    return _fingerprint({
        key: value for key, value in runtime.runtime_receipt.items()
        if key not in {"controller_cuda_source", "receipt_sha256"}
    })


@dataclass(frozen=True)
class V94bAcceptedDensityReceiptGPUV96:
    execution_fingerprint: str
    active_execution_fingerprint: str
    parent_state_sha256: str
    base_v92_state_sha256: str
    suspended_state_sha256: str
    accepted_update: int
    time_s: float
    depth_m: np.ndarray
    suspended_mass_kg_m2: np.ndarray
    concentration_kg_m3: np.ndarray
    water_density_kg_m3: np.ndarray
    mixture_density_kg_m3: np.ndarray
    receipt_sha256: str


def _density_receipt_token(value: V94bAcceptedDensityReceiptGPUV96) -> str:
    return _fingerprint({
        "scope_id": SCOPE_ID + ":accepted-v94b-density",
        "source_kind": "typed_v94b_accepted_state",
        "backend": BACKEND,
        "execution_fingerprint": value.execution_fingerprint,
        "active_execution_fingerprint": value.active_execution_fingerprint,
        "parent_state_sha256": value.parent_state_sha256,
        "base_v92_state_sha256": value.base_v92_state_sha256,
        "suspended_state_sha256": value.suspended_state_sha256,
        "accepted_update": value.accepted_update,
        "time_s": value.time_s,
        "arrays_sha256": _digest(
            value.depth_m, value.suspended_mass_kg_m2,
            value.concentration_kg_m3, value.water_density_kg_m3,
            value.mixture_density_kg_m3),
        "call_order": "accepted_v94b_state_to_next_live_cuda_hydraulics",
    })


@dataclass(frozen=True)
class CoupledDensInGPUExecutionV96:
    active_execution: ActiveSuspendedControlExecutionV94
    hydraulic_execution: DensInResidualGPUExecutionV95
    initial_bed_elevation_m: np.ndarray
    reference_bed_mass_kg_m2: np.ndarray
    dry_bulk_density_kg_m3: np.ndarray
    bed_porosity_by_fraction: np.ndarray
    geometry_policy: str
    include_sediment_density_increment: bool
    open_boundary_execution: CoupledOpenBoundaryGPUExecutionV96 | None
    semantic_id: str
    fingerprint: str

    @property
    def flat_bed_elevation_m(self) -> np.ndarray:
        """Compatibility alias retained for the bounded v96 evidence."""
        return self.initial_bed_elevation_m

    @property
    def fraction_count(self) -> int:
        return self.active_execution.fraction_count

    @property
    def cell_count(self) -> int:
        return self.active_execution.cell_count

    @property
    def face_count(self) -> int:
        return _v92_execution(self.active_execution).face_count


def _execution_payload(value: CoupledDensInGPUExecutionV96) -> dict:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "semantic_id": value.semantic_id,
        "backend": BACKEND,
        "active_v94b_execution": execution_document_v94b(
            value.active_execution),
        "live_v95_execution": execution_document_v95(
            value.hydraulic_execution),
        "initial_bed_sha256": _digest(value.initial_bed_elevation_m),
        "reference_bed_mass_sha256": _digest(
            value.reference_bed_mass_kg_m2),
        "dry_bulk_density_sha256": _digest(value.dry_bulk_density_kg_m3),
        "bed_porosity_sha256": _digest(value.bed_porosity_by_fraction),
        "dry_bulk_density_origin": "rhosol*(1-porosity)",
        "bed_mapping": "z0+sum((M-M0)/rho_dry_by_fraction)",
        "geometry_policy": value.geometry_policy,
        "density_mode": (
            "water_plus_sediment" if
            value.include_sediment_density_increment else "water_eos"),
        "open_boundary_execution": (
            None if value.open_boundary_execution is None else
            execution_document_coupled_open_boundary_gpu_v96(
                value.open_boundary_execution)),
        "geometry_water_volume_ledger": "exact_zero",
        "hydraulic_bed_timing": "accepted_parent_bed_used_current_step",
        "exner_bed_timing": "candidate_bed_visible_next_step",
        "accepted_density_source": "typed_v94b_accepted_state",
        "joint_state": "v94b_q_plus_hles_plus_cuda_predictor_no_shadow_q",
        "residual_order": "augmented_fwave_then_lagged_v91_then_v85_density",
        "hydraulic_endpoint_owner_count": 1,
        "water_face_flux_owner_count": 1,
        "scalar_lifecycle_owner_count": 1,
        "authoritative_exner_owner_count": 1,
        "hles_clock_owner_count": 1,
        "boundary_mode": (
            "closed_reflective" if value.open_boundary_execution is None
            else "sole_owner_physical_open_trace"),
        "grid_scope": "evolving_bed_fully_wet_cartesian_fixed_grid",
        "water_density_scope": "spatial_accepted_v92_field",
        "turbulence_scope": "lagged_v91_each_true_residual",
        "integrator": "implicit_fwave_midpoint",
        "accepted_update_limit": MAX_ACCEPTED_UPDATES,
        "automatic_cpu_fallback": False,
        "product_admission": False,
    }


def _topology_matches(value: CoupledDensInGPUExecutionV96) -> bool:
    grid = _grid_execution(value.active_execution)
    base = value.hydraulic_execution.base_execution
    return all(np.array_equal(left, right) for left, right in (
        (grid.owner, base.owner),
        (grid.neighbour, base.neighbour),
        (grid.normal, base.normal),
        (grid.face_length_m, base.face_length_m),
        (grid.centre_distance_m, base.face_distance_m),
        (grid.cell_area_m2, base.cell_area_m2),
        (grid.cell_face_offset, base.cell_face_offset),
        (grid.cell_faces, base.cell_faces),
        (grid.cell_face_sign, base.cell_face_sign),
    ))


def resolve_coupled_densin_gpu_execution_v96(
    active_execution: ActiveSuspendedControlExecutionV94,
    hydraulic_execution: DensInResidualGPUExecutionV95, *,
    initial_bed_elevation_m: object,
    reference_bed_mass_kg_m2: object,
    dry_bulk_density_kg_m3: object,
    bed_porosity_by_fraction: object,
    include_sediment_density_increment: bool = True,
    open_boundary_execution: CoupledOpenBoundaryGPUExecutionV96 | None = None,
) -> CoupledDensInGPUExecutionV96:
    execution_document_v94b(active_execution)
    execution_document_v95(hydraulic_execution)
    _require(active_execution.semantic_id in {
                 SEMANTIC_ID_V94, SEMANTIC_ID_V94B} and
             active_execution.backend == "implicit_fwave_midpoint_gpu" and
             active_execution.base_execution.suspended_vector_mode in {
                 SUSPENDED_VECTOR_CONSTRUCTED_V93,
                 SUSPENDED_VECTOR_EXACT_ZERO_V93},
             "v96 CUDA requires a reviewed actual-CUDA v94/v94b route")
    v92 = _v92_execution(active_execution)
    v78 = v92.base_execution.base_execution
    profile = v78.base_execution.profile
    base = hydraulic_execution.base_execution
    nc, ng = active_execution.cell_count, active_execution.fraction_count
    bed = _array(initial_bed_elevation_m, (nc,), "initial bed elevation")
    reference = _array(
        reference_bed_mass_kg_m2, (ng, nc), "reference bed mass",
        nonnegative=True)
    bulk = _array(
        dry_bulk_density_kg_m3, (ng,), "dry bulk density", positive=True)
    porosity = _array(
        bed_porosity_by_fraction, (ng,), "bed porosity", nonnegative=True)
    _require(type(include_sediment_density_increment) is bool,
             "v96 CUDA sediment-density selection must be Boolean")
    if open_boundary_execution is not None:
        execution_document_coupled_open_boundary_gpu_v96(
            open_boundary_execution)
    _require(np.all(porosity < 1.0) and np.allclose(
                 bulk,
                 np.asarray(profile.sediment_density_kg_m3)*(1.0-porosity),
                 rtol=8*np.finfo(float).eps, atol=0.0),
             "v96 CUDA dry bulk density must equal rhosol*(1-porosity)")
    _require(hydraulic_execution.cell_count == active_execution.cell_count and
             hydraulic_execution.face_count == v92.face_count and
             base.fraction_count == active_execution.fraction_count and
             base.hydraulic_execution_sha256 ==
                v78.hydraulic_execution_sha256 and
             np.array_equal(base.solid_density_kg_m3,
                            profile.sediment_density_kg_m3) and
             base.hydraulic_reference_density_kg_m3 ==
                profile.water_density_kg_m3 and
             base.gravity_m_s2 == profile.gravity_m_s2 and
             base.dry_tolerance_m == v78.base_execution.dry_tolerance_m,
             "v96 CUDA hydraulic fractions/constants differ")
    placeholder = CoupledDensInGPUExecutionV96(
        active_execution, hydraulic_execution, bed, reference, bulk,
        porosity, GEOMETRY_POLICY, include_sediment_density_increment,
        open_boundary_execution, SEMANTIC_ID, "")
    _require(_topology_matches(placeholder),
             "v96 CUDA hydraulic and v94b topologies differ")
    if open_boundary_execution is not None:
        trace = open_boundary_execution.boundary_trace_execution
        _require(trace.cell_count == nc and trace.face_count == v92.face_count and
                 np.array_equal(trace.owner, base.owner) and
                 np.array_equal(trace.neighbour, base.neighbour) and
                 np.array_equal(trace.face_distance_m,
                                base.face_distance_m),
                 "v96 CUDA open-boundary and hydraulic topologies differ")
    return CoupledDensInGPUExecutionV96(
        **{**placeholder.__dict__,
           "fingerprint": _fingerprint(_execution_payload(placeholder))})


def execution_document_v96(value: CoupledDensInGPUExecutionV96) -> dict:
    _require(type(value) is CoupledDensInGPUExecutionV96,
             "v96 CUDA execution type differs")
    payload = _execution_payload(value)
    _require(value.semantic_id == SEMANTIC_ID and
             value.active_execution.semantic_id in {
                 SEMANTIC_ID_V94, SEMANTIC_ID_V94B} and
             value.geometry_policy == GEOMETRY_POLICY and
             type(value.include_sediment_density_increment) is bool and
             (value.open_boundary_execution is None or
              type(value.open_boundary_execution) is
                CoupledOpenBoundaryGPUExecutionV96) and
             _topology_matches(value) and
             value.fingerprint == _fingerprint(payload),
             "v96 CUDA execution fingerprint differs")
    return {**payload, "fingerprint": value.fingerprint}


@dataclass(frozen=True)
class CoupledDensInGPUStateV96:
    execution_fingerprint: str
    active_state: ActiveSuspendedControlStateV94
    hles_state: HlesProducerStateV90
    predictor_rate_m2_s2: np.ndarray
    have_predictor: bool
    hydraulic_boundary_ledger_m3: np.ndarray
    bed_elevation_m: np.ndarray
    free_surface_elevation_m: np.ndarray
    geometry_water_volume_ledger_m3: float
    accepted_boundary_scalar_state: AcceptedBoundaryScalarStateV96 | None
    accepted_chain_sha256: str
    state_sha256: str

    @property
    def q(self) -> np.ndarray:
        return _v92_state(self.active_state).q

    @property
    def accepted_updates(self) -> int:
        return self.active_state.accepted_updates

    @property
    def time_s(self) -> float:
        return self.active_state.time_s


def _state_token(value: CoupledDensInGPUStateV96) -> str:
    return _fingerprint({
        "scope_id": SCOPE_ID + ":state",
        "execution_fingerprint": value.execution_fingerprint,
        "active_v94b_state_sha256": value.active_state.state_sha256,
        "hles_state_sha256": value.hles_state.state_sha256,
        "controller_arrays_sha256": _digest(
            value.predictor_rate_m2_s2,
            value.hydraulic_boundary_ledger_m3),
        "geometry_arrays_sha256": _digest(
            value.bed_elevation_m, value.free_surface_elevation_m),
        "geometry_water_volume_ledger_m3":
            value.geometry_water_volume_ledger_m3,
        "accepted_boundary_scalar_state_sha256": (
            None if value.accepted_boundary_scalar_state is None else
            value.accepted_boundary_scalar_state.state_sha256),
        "have_predictor": value.have_predictor,
        "accepted_chain_sha256": value.accepted_chain_sha256,
        "shared_update": value.accepted_updates,
        "shared_time_s": value.time_s,
})


def _bed_from_mass(
    execution: CoupledDensInGPUExecutionV96,
    bed_mass_kg_m2: object,
) -> np.ndarray:
    mass = _array(
        bed_mass_kg_m2,
        (execution.fraction_count, execution.cell_count),
        "accepted bed mass", nonnegative=True)
    bed = np.ascontiguousarray(
        execution.initial_bed_elevation_m+
        np.sum((mass-execution.reference_bed_mass_kg_m2)/
               execution.dry_bulk_density_kg_m3[:, None], axis=0))
    _require(np.isfinite(bed).all(),
             "v96 CUDA derived bed elevation is non-finite")
    bed.setflags(write=False)
    return bed


def _wrap_state(
    execution: CoupledDensInGPUExecutionV96, *,
    active_state: ActiveSuspendedControlStateV94,
    hles_state: HlesProducerStateV90,
    predictor_rate_m2_s2: object,
    have_predictor: bool,
    hydraulic_boundary_ledger_m3: object,
    accepted_chain_sha256: str,
    q3_provider: object | None = None,
    accepted_boundary_scalar_state: AcceptedBoundaryScalarStateV96 | None = None,
) -> CoupledDensInGPUStateV96:
    execution_document_v96(execution)
    validate_active_suspended_control_state_v94b(
        active_state, execution.active_execution, q3_provider)
    hles_execution = _v92_execution(
        execution.active_execution).base_execution.hles_execution
    validate_hles_producer_state_v90(hles_state, hles_execution)
    v92 = _v92_state(active_state)
    predictor = _array(
        predictor_rate_m2_s2, (3, execution.cell_count), "predictor rate")
    boundary = _array(
        hydraulic_boundary_ledger_m3,
        (_v92_execution(execution.active_execution).boundary_count,),
        "hydraulic boundary ledger")
    _require(type(have_predictor) is bool,
             "v96 CUDA predictor flag must be Boolean")
    open_execution = execution.open_boundary_execution
    if open_execution is None:
        _require(accepted_boundary_scalar_state is None,
                 "v96 CUDA closed state cannot carry an open scalar image")
    else:
        _require(type(accepted_boundary_scalar_state) is
                 AcceptedBoundaryScalarStateV96,
                 "v96 CUDA open state requires an accepted scalar image")
        validate_accepted_boundary_scalar_state_v96(
            accepted_boundary_scalar_state,
            _v92_execution(execution.active_execution),
            open_execution.boundary_trace_execution,
            accepted_update=active_state.accepted_updates,
            time_s=active_state.time_s)
    bed = _bed_from_mass(execution, active_state.base_state.bed_mass_kg_m2)
    eta = np.ascontiguousarray(v92.q[0]+bed)
    eta.setflags(write=False)
    _require(active_state.accepted_updates == hles_state.accepted_updates and
             active_state.time_s == hles_state.time_s and
             active_state.accepted_updates <= MAX_ACCEPTED_UPDATES and
             np.all(v92.q[0] >
                    execution.hydraulic_execution.base_execution.
                    dry_tolerance_m) and
             (q3_provider is None or _provider_is_closed(q3_provider)),
             "v96 CUDA parent clocks, wetness, or boundary scope differ")
    chain = _sha(accepted_chain_sha256, "accepted chain")
    placeholder = CoupledDensInGPUStateV96(
        execution.fingerprint, active_state, hles_state, predictor,
        have_predictor, boundary, bed, eta, 0.0,
        accepted_boundary_scalar_state, chain, "")
    return CoupledDensInGPUStateV96(
        **{**placeholder.__dict__, "state_sha256": _state_token(placeholder)})


def resolve_coupled_densin_gpu_state_v96(
    execution: CoupledDensInGPUExecutionV96, *,
    active_state: ActiveSuspendedControlStateV94,
    hles_state: HlesProducerStateV90,
    q3_provider: object,
    boundary_count: int,
    initial_boundary_scalar_state: AcceptedBoundaryScalarStateV96 | None = None,
) -> CoupledDensInGPUStateV96:
    _require(active_state.accepted_updates == 0 and
             hles_state.accepted_updates == 0,
             "v96 CUDA fresh joint state must start at update zero")
    _require(np.array_equal(
                 active_state.base_state.bed_mass_kg_m2,
                 execution.reference_bed_mass_kg_m2),
             "v96 CUDA reference bed mass differs from initial state")
    _require(type(boundary_count) is int and boundary_count ==
             _v92_execution(execution.active_execution).boundary_count,
             "v96 CUDA hydraulic boundary count differs")
    chain = _fingerprint({
        "scope_id": SCOPE_ID,
        "execution_fingerprint": execution.fingerprint,
        "initial_active_state_sha256": active_state.state_sha256,
        "initial_hles_state_sha256": hles_state.state_sha256,
        "initial_bed_sha256": _digest(execution.initial_bed_elevation_m),
        "initial_boundary_scalar_state_sha256": (
            None if initial_boundary_scalar_state is None else
            initial_boundary_scalar_state.state_sha256),
    })
    return _wrap_state(
        execution, active_state=active_state, hles_state=hles_state,
        predictor_rate_m2_s2=np.zeros((3, execution.cell_count)),
        have_predictor=False,
        hydraulic_boundary_ledger_m3=np.zeros(boundary_count),
        accepted_chain_sha256=chain, q3_provider=q3_provider,
        accepted_boundary_scalar_state=initial_boundary_scalar_state)


def validate_coupled_densin_gpu_state_v96(
    state: CoupledDensInGPUStateV96,
    execution: CoupledDensInGPUExecutionV96,
    q3_provider: object | None = None,
) -> None:
    _require(type(state) is CoupledDensInGPUStateV96 and
             state.execution_fingerprint == execution.fingerprint and
             state.geometry_water_volume_ledger_m3 == 0.0 and
             state.state_sha256 == _state_token(state),
             "v96 CUDA state is stale, mutated, or cross-execution")
    rebuilt = _wrap_state(
        execution, active_state=state.active_state,
        hles_state=state.hles_state,
        predictor_rate_m2_s2=state.predictor_rate_m2_s2,
        have_predictor=state.have_predictor,
        hydraulic_boundary_ledger_m3=
            state.hydraulic_boundary_ledger_m3,
        accepted_chain_sha256=state.accepted_chain_sha256,
        q3_provider=q3_provider,
        accepted_boundary_scalar_state=state.accepted_boundary_scalar_state)
    _require(rebuilt.state_sha256 == state.state_sha256,
             "v96 CUDA rebuilt state differs")


def make_v94b_accepted_density_receipt_gpu_v96(
    execution: CoupledDensInGPUExecutionV96,
    parent: CoupledDensInGPUStateV96,
    q3_provider: object,
) -> V94bAcceptedDensityReceiptGPUV96:
    validate_coupled_densin_gpu_state_v96(parent, execution, q3_provider)
    v92 = _v92_state(parent.active_state)
    suspended = v92.suspended_state
    depth = _array(v92.q[0], (execution.cell_count,), "accepted depth",
                   positive=True)
    mass = _array(
        suspended.suspended_mass_kg_m2,
        (execution.fraction_count, execution.cell_count),
        "accepted suspended mass", nonnegative=True)
    concentration = np.ascontiguousarray(mass/depth[None, :])
    water = _array(
        v92.water_density_cell_kg_m3, (execution.cell_count,),
        "accepted water density", positive=True)
    mixture = mixture_density_kg_m3_v85(
        concentration, water,
        execution.hydraulic_execution.base_execution.solid_density_kg_m3)
    concentration.setflags(write=False)
    mixture.setflags(write=False)
    placeholder = V94bAcceptedDensityReceiptGPUV96(
        execution.fingerprint, execution.active_execution.fingerprint,
        parent.state_sha256, v92.state_sha256, suspended.state_sha256,
        parent.accepted_updates, parent.time_s, depth, mass, concentration,
        water, mixture, "")
    return V94bAcceptedDensityReceiptGPUV96(
        **{**placeholder.__dict__,
           "receipt_sha256": _density_receipt_token(placeholder)})


def validate_v94b_accepted_density_receipt_gpu_v96(
    value: V94bAcceptedDensityReceiptGPUV96,
    execution: CoupledDensInGPUExecutionV96,
    parent: CoupledDensInGPUStateV96,
    q3_provider: object,
) -> None:
    _require(type(value) is V94bAcceptedDensityReceiptGPUV96,
             "v96 CUDA density receipt type differs")
    expected = make_v94b_accepted_density_receipt_gpu_v96(
        execution, parent, q3_provider)
    _require(value.receipt_sha256 == _density_receipt_token(value) and
             value.receipt_sha256 == expected.receipt_sha256 and
             value.parent_state_sha256 == parent.state_sha256,
             "v96 CUDA accepted v94b density receipt is stale or mismatched")


@dataclass(frozen=True)
class HydraulicDensInTrialGPUV96:
    execution_fingerprint: str
    parent_state_sha256: str
    accepted_density: V94bAcceptedDensityReceiptGPUV96
    accepted_bed_sha256: str
    dt_s: float
    residual_base_q: np.ndarray
    residual_endpoint_q: np.ndarray
    candidate_q: np.ndarray
    midpoint_q: np.ndarray
    mixture_density_kg_m3: np.ndarray
    base_fwave_rate: np.ndarray
    density_momentum_rate: np.ndarray
    full_rate: np.ndarray
    final_residual: np.ndarray
    face_normal_acceleration_m_s2: np.ndarray
    cell_acceleration_m_s2: np.ndarray
    face_water_flux_m2_s: np.ndarray
    water_boundary_ledger_m3: np.ndarray
    midpoint_q_by_evaluation: np.ndarray
    accepted_boundary_density: AcceptedBoundaryDensityReceiptV96 | None
    boundary_trace_receipts: tuple[HydraulicBoundaryTraceReceiptV96, ...]
    open_density_kernel_count: int
    midpoint_depth_sha256_by_evaluation: tuple[str, ...]
    residual_evaluation_count: int
    density_source_evaluation_count: int
    augmented_riemann_call_count: int
    approximate_jacobian_evaluation_count: int
    density_jacobian_derivative_evaluation_count: int
    post_convergence_density_application_count: int
    accepted_parent_density_mutation_count: int
    nonlinear_iterations: int
    nonlinear_residual_linf: float
    water_volume_residual_m3: float
    backend: str
    trial_sha256: str


def _hydraulic_trial_token(value: HydraulicDensInTrialGPUV96) -> str:
    return _fingerprint({
        "scope_id": SCOPE_ID + ":hydraulic-trial",
        "execution_fingerprint": value.execution_fingerprint,
        "parent_state_sha256": value.parent_state_sha256,
        "accepted_density_receipt_sha256":
            value.accepted_density.receipt_sha256,
        "accepted_bed_sha256": value.accepted_bed_sha256,
        "dt_s": value.dt_s,
        "arrays_sha256": _digest(
            value.residual_base_q, value.residual_endpoint_q,
            value.candidate_q, value.midpoint_q,
            value.mixture_density_kg_m3, value.base_fwave_rate,
            value.density_momentum_rate, value.full_rate,
            value.final_residual, value.face_normal_acceleration_m_s2,
            value.cell_acceleration_m_s2, value.face_water_flux_m2_s,
            value.water_boundary_ledger_m3,
            value.midpoint_q_by_evaluation),
        "accepted_boundary_density_receipt_sha256": (
            None if value.accepted_boundary_density is None else
            value.accepted_boundary_density.receipt_sha256),
        "boundary_trace_receipt_sha256": [
            item.receipt_sha256 for item in value.boundary_trace_receipts],
        "open_density_kernel_count": value.open_density_kernel_count,
        "midpoint_depth_sha256_by_evaluation":
            list(value.midpoint_depth_sha256_by_evaluation),
        "counts": [
            value.residual_evaluation_count,
            value.density_source_evaluation_count,
            value.augmented_riemann_call_count,
            value.approximate_jacobian_evaluation_count,
            value.density_jacobian_derivative_evaluation_count,
            value.post_convergence_density_application_count,
            value.accepted_parent_density_mutation_count,
            value.nonlinear_iterations,
        ],
        "nonlinear_residual_linf": value.nonlinear_residual_linf,
        "water_volume_residual_m3": value.water_volume_residual_m3,
        "backend": value.backend,
    })


def _validate_hydraulic_trial(
    execution: CoupledDensInGPUExecutionV96,
    parent: CoupledDensInGPUStateV96,
    provider: object,
    trial: HydraulicDensInTrialGPUV96,
) -> None:
    validate_coupled_densin_gpu_state_v96(parent, execution, provider)
    validate_v94b_accepted_density_receipt_gpu_v96(
        trial.accepted_density, execution, parent, provider)
    _require(type(trial) is HydraulicDensInTrialGPUV96 and
             trial.execution_fingerprint == execution.fingerprint and
             trial.parent_state_sha256 == parent.state_sha256 and
             trial.backend == BACKEND and
             trial.trial_sha256 == _hydraulic_trial_token(trial),
             "v96 CUDA hydraulic trial is stale or mismatched")
    _require(trial.accepted_bed_sha256 ==
             _digest(parent.bed_elevation_m),
             "v96 CUDA hydraulic trial used a different accepted bed")
    hydraulic = execution.hydraulic_execution
    nc, nf = execution.cell_count, execution.face_count
    for array, shape, name in (
        (trial.residual_base_q, (3, nc), "residual base"),
        (trial.residual_endpoint_q, (3, nc), "residual endpoint"),
        (trial.candidate_q, (3, nc), "hydraulic candidate"),
        (trial.midpoint_q, (3, nc), "midpoint"),
        (trial.mixture_density_kg_m3, (nc,), "mixture"),
        (trial.base_fwave_rate, (3, nc), "base rate"),
        (trial.density_momentum_rate, (2, nc), "density rate"),
        (trial.full_rate, (3, nc), "full rate"),
        (trial.final_residual, (3, nc), "residual"),
        (trial.face_normal_acceleration_m_s2, (nf,), "face acceleration"),
        (trial.cell_acceleration_m_s2, (2, nc), "cell acceleration"),
        (trial.face_water_flux_m2_s, (nf,), "face water flux"),
        (trial.water_boundary_ledger_m3,
         (_v92_execution(execution.active_execution).boundary_count,),
         "water boundary ledger"),
        (trial.midpoint_q_by_evaluation, (trial.residual_evaluation_count,
                                         3, nc),
         "midpoint history"),
    ):
        _array(array, shape, name)
    count = trial.residual_evaluation_count
    dt = _finite(trial.dt_s, "trial timestep", positive=True)
    _require(dt <= hydraulic.maximum_dt_s and
             1 <= count <= hydraulic.maximum_residual_evaluations and
             trial.density_source_evaluation_count == count and
             trial.augmented_riemann_call_count == nf*count and
             trial.approximate_jacobian_evaluation_count >= 0 and
             trial.density_jacobian_derivative_evaluation_count == 0 and
             trial.post_convergence_density_application_count == 0 and
             trial.accepted_parent_density_mutation_count == 0 and
             trial.nonlinear_iterations == count-1 and
             np.array_equal(trial.midpoint_q_by_evaluation[-1],
                            trial.midpoint_q) and
             len(trial.midpoint_depth_sha256_by_evaluation) == count and
             all(_SHA256.fullmatch(item) is not None for item in
                 trial.midpoint_depth_sha256_by_evaluation) and
             trial.midpoint_depth_sha256_by_evaluation[-1] ==
                _digest(trial.midpoint_q[0]),
             "v96 CUDA hydraulic counts or depth history differ")
    open_execution = execution.open_boundary_execution
    if open_execution is None:
        _require(trial.accepted_boundary_density is None and
                 trial.boundary_trace_receipts == () and
                 trial.open_density_kernel_count == 0,
                 "v96 CUDA closed trial contains open-boundary evidence")
    else:
        trace_execution = open_execution.boundary_trace_execution
        _require(type(trial.accepted_boundary_density) is
                 AcceptedBoundaryDensityReceiptV96 and
                 len(trial.boundary_trace_receipts) == count and
                 trial.open_density_kernel_count == count,
                 "v96 CUDA open-boundary density ownership differs")
        validate_accepted_boundary_density_receipt_v96(
            trial.accepted_boundary_density, trace_execution)
        _require(trial.accepted_boundary_density.parent_state_sha256 ==
                 parent.state_sha256 and
                 trial.accepted_boundary_density.accepted_update ==
                 parent.accepted_updates and
                 trial.accepted_boundary_density.time_s == parent.time_s,
                 "v96 CUDA accepted boundary density is stale")
        for index, receipt in enumerate(trial.boundary_trace_receipts, 1):
            validate_hydraulic_boundary_trace_receipt_v96(
                receipt, trace_execution,
                midpoint_q=trial.midpoint_q_by_evaluation[index-1],
                bed_elevation_m=parent.bed_elevation_m,
                hydraulic_request_sha256=
                    receipt.hydraulic_request_sha256,
                parent_state_sha256=parent.state_sha256,
                evaluation_index=index,
                evaluation_time_s=parent.time_s+.5*dt)
    selected_density = (
        trial.accepted_density.mixture_density_kg_m3
        if execution.include_sediment_density_increment else
        trial.accepted_density.water_density_kg_m3)
    _require(np.array_equal(trial.accepted_density.depth_m, parent.q[0]) and
             np.allclose(
                 trial.mixture_density_kg_m3, selected_density,
                 rtol=2*np.finfo(float).eps, atol=0.0),
             "v96 CUDA accepted v94b density mutated")
    expected_full = np.array(trial.base_fwave_rate, copy=True)
    expected_full[1:] += trial.density_momentum_rate
    _require(np.array_equal(expected_full, trial.full_rate) and
             np.array_equal(trial.full_rate[0], trial.base_fwave_rate[0]),
             "v96 CUDA density/full-rate composition differs")
    expected_residual = (
        trial.midpoint_q-trial.residual_base_q-
        .5*dt*trial.full_rate)
    residual_scale = max(1.0, float(np.max(np.abs(expected_residual))))
    _require(np.allclose(
                 expected_residual, trial.final_residual,
                 rtol=16*np.finfo(float).eps,
                 atol=16*np.finfo(float).eps*residual_scale) and
             trial.nonlinear_residual_linf ==
                float(np.max(np.abs(trial.final_residual))),
             "v96 CUDA nonlinear residual evidence differs")
    residual_norm = float(np.linalg.norm(trial.final_residual))
    update_norm = float(np.linalg.norm(
        trial.midpoint_q-trial.residual_base_q))
    forcing_norm = float(np.linalg.norm(.5*dt*trial.full_rate))
    tolerance = max(
        hydraulic.nonlinear_atol*math.sqrt(3.0*nc),
        hydraulic.nonlinear_rtol*max(update_norm, forcing_norm))
    _require(math.isfinite(residual_norm) and residual_norm <= tolerance,
             "v96 CUDA nonlinear residual did not converge")
    endpoint = (2.0*trial.midpoint_q-trial.residual_base_q-
                2.0*trial.final_residual)
    scale = max(1.0, float(np.max(np.abs(endpoint))))
    _require(np.allclose(
                 endpoint, trial.residual_endpoint_q,
                 rtol=128*np.finfo(float).eps,
                 atol=128*np.finfo(float).eps*scale) and
             np.array_equal(trial.residual_base_q[0], parent.q[0]) and
             np.array_equal(trial.candidate_q[0],
                            trial.residual_endpoint_q[0]),
             "v96 CUDA implicit endpoint or split depth differs")
    expected_base = _manning_half_step_oracle(
        parent.q, hydraulic.manning_roughness, .5*dt,
        gravity_m_s2=hydraulic.base_execution.gravity_m_s2,
        dry_tolerance_m=hydraulic.base_execution.dry_tolerance_m,
        chezy_minimum_depth_m=hydraulic.manning_chezy_minimum_depth_m)
    expected_candidate = _manning_half_step_oracle(
        trial.residual_endpoint_q, hydraulic.manning_roughness, .5*dt,
        gravity_m_s2=hydraulic.base_execution.gravity_m_s2,
        dry_tolerance_m=hydraulic.base_execution.dry_tolerance_m,
        chezy_minimum_depth_m=hydraulic.manning_chezy_minimum_depth_m)
    friction_scale = max(
        1.0, float(np.max(np.abs(expected_base))),
        float(np.max(np.abs(expected_candidate))))
    _require(np.allclose(
                 trial.residual_base_q, expected_base,
                 rtol=64*np.finfo(float).eps,
                 atol=64*np.finfo(float).eps*friction_scale) and
             np.allclose(
                 trial.candidate_q, expected_candidate,
                 rtol=64*np.finfo(float).eps,
                 atol=64*np.finfo(float).eps*friction_scale),
             "v96 CUDA authenticated Manning split differs")
    face = delft_kmax1_face_acceleration_v85(
        trial.midpoint_q[0], trial.mixture_density_kg_m3,
        hydraulic.base_execution.owner,
        hydraulic.base_execution.neighbour,
        hydraulic.base_execution.face_distance_m,
        gravity_m_s2=hydraulic.base_execution.gravity_m_s2,
        hydraulic_reference_density_kg_m3=
            hydraulic.base_execution.hydraulic_reference_density_kg_m3)
    if open_execution is not None:
        receipt = trial.boundary_trace_receipts[-1]
        active = receipt.exterior_densin_active_mask.astype(bool)
        if open_execution.ibaroc_open_boundary:
            owner = hydraulic.base_execution.owner[active]
            face[active] = (
                -hydraulic.base_execution.gravity_m_s2*
                (trial.accepted_boundary_density.
                 mixture_density_kg_m3[active]-
                 trial.mixture_density_kg_m3[owner])*
                (receipt.trace_q[0, active]+trial.midpoint_q[0, owner])/
                (4.0*hydraulic.base_execution.
                 hydraulic_reference_density_kg_m3*
                 hydraulic.base_execution.face_distance_m[active]))
        else:
            face[active] = 0.0
    cell = project_face_acceleration_v85(
        face, hydraulic.base_execution.normal,
        hydraulic.base_execution.face_length_m,
        hydraulic.base_execution.cell_face_offset,
        hydraulic.base_execution.cell_faces,
        hydraulic.base_execution.cell_face_sign)
    density = trial.midpoint_q[0][None, :]*cell
    source_scale = max(1.0, float(np.max(np.abs(density))))
    _require(np.allclose(
                 face, trial.face_normal_acceleration_m_s2,
                 rtol=8*np.finfo(float).eps, atol=1e-18) and
             np.allclose(
                 cell, trial.cell_acceleration_m_s2,
                 rtol=16*np.finfo(float).eps, atol=1e-18) and
             np.allclose(
                 density, trial.density_momentum_rate,
                 rtol=16*np.finfo(float).eps,
                 atol=16*np.finfo(float).eps*source_scale),
             "v96 CUDA density source differs from v85 oracle")
    area = hydraulic.base_execution.cell_area_m2
    water_scale = max(1.0, float(np.sum(area*parent.q[0])))
    water_residual = float(np.sum(
        area*(trial.candidate_q[0]-parent.q[0]))-
        np.sum(trial.water_boundary_ledger_m3))
    _require(abs(trial.water_volume_residual_m3-water_residual) <=
                128*np.finfo(float).eps*water_scale and
             abs(trial.water_volume_residual_m3) <=
                max(1.e-12, 1.e-11*water_scale),
             "v96 CUDA authenticated water budget differs")


class ResidentV94bDensInHookV96(ResidentDensInResidualV95):
    """V95 numerical kernels with an exact typed-v94b parent seam."""

    is_v95_densin_residual_consumer = True
    is_v96_typed_v94b_density_consumer = True

    def __init__(self, outer_execution: CoupledDensInGPUExecutionV96,
                 provider: ResidentQ3V83ActiveExnerAdapterV94):
        super().__init__(outer_execution.hydraulic_execution)
        self.outer_execution = outer_execution
        self.provider = provider
        self.accepted_bed = self.cp.empty(
            outer_execution.cell_count, dtype=self.cp.float64)
        self._density_concentration_input = self.cp.empty(
            (outer_execution.fraction_count, outer_execution.cell_count),
            dtype=self.cp.float64)
        rows = outer_execution.hydraulic_execution.maximum_residual_evaluations
        self.midpoint_q_history = self.cp.empty(
            (rows, 3, self.nc), dtype=self.cp.float64)
        self.open_density = (
            None if outer_execution.open_boundary_execution is None else
            ResidentOpenBoundaryDensInGPUV96(
                outer_execution.open_boundary_execution,
                cupy_module=self.cp))
        self.boundary_trace_history = self.cp.zeros(
            (rows, 3, self.nf), dtype=self.cp.float64)
        self.boundary_flux_history = self.cp.zeros(
            (rows, 3, self.nf), dtype=self.cp.float64)
        self._accepted_boundary_density = None
        self._hydraulic_request_sha256 = None
        self._initial_open_density_kernel_count = 0

    def accepted_bed_for_controller(self):
        _require(self._open and self._parent is not None,
                 "v96 CUDA controller requested bed outside a trial")
        return self.accepted_bed

    def _after_face_density(self, midpoint_q) -> None:
        row = self._residual_evaluations
        self.cp.copyto(self.midpoint_q_history[row], midpoint_q)
        if self.open_density is None:
            return
        trace_execution = (
            self.outer_execution.open_boundary_execution.
            boundary_trace_execution)
        physical = self.cp.asarray(
            trace_execution.open_mask.astype(np.int32))
        active = physical
        self.cp.copyto(
            self.boundary_trace_history[row],
            self._controller.midpoint_boundary_trace)
        self.cp.copyto(
            self.boundary_flux_history[row],
            self._controller.midpoint_physical_flux)
        inactive = physical == 0
        self.boundary_trace_history[row, :, inactive] = 0.0
        self.boundary_flux_history[row, :, inactive] = 0.0
        self.open_density.overwrite_active_open_faces(
            cell_depth=midpoint_q[0],
            cell_mixture_density=self.operator.mixture,
            hydraulic_trace_q=self.boundary_trace_history[row],
            active_open_face=active,
            face_acceleration=self.operator.face_acceleration,
            face_status=self.operator.face_status,
            gravity_m_s2=self.execution.base_execution.gravity_m_s2,
            reference_density_kg_m3=self.execution.base_execution.
                hydraulic_reference_density_kg_m3)

    def _make_boundary_trace_receipts(
        self, midpoint_history: np.ndarray,
        trace_history: np.ndarray,
        flux_history: np.ndarray,
    ) -> tuple[HydraulicBoundaryTraceReceiptV96, ...]:
        if self.open_density is None:
            return ()
        trace_execution = (
            self.outer_execution.open_boundary_execution.
            boundary_trace_execution)
        physical = trace_execution.open_mask.astype(np.int32)
        # This first fully-wet open route admits sampled-active physical open
        # faces.  The two-mask receipt remains explicit, so later wet/dry or
        # sampled-closed work cannot silently conflate density with flux.
        densin = np.array(physical, copy=True)
        owner_state = {
            "controller_source_sha256": self._controller.source_sha256,
            "parent_state_sha256": self._parent.state_sha256,
            "hydraulic_request_sha256": self._hydraulic_request_sha256,
        }
        current = {"row": 0}

        def resolver(**_):
            row = current["row"]
            trace = trace_history[row]
            flux = flux_history[row]
            reconstruction = _fingerprint({
                "scope_id": SCOPE_ID + ":live-reconstruction-receipt",
                "controller_source_sha256": self._controller.source_sha256,
                "midpoint_q_sha256": _digest(midpoint_history[row]),
                "trace_sha256": _digest(trace),
            })
            branch = _fingerprint({
                "scope_id": SCOPE_ID + ":live-boundary-branch-receipt",
                "boundary_owner_runtime_sha256":
                    trace_execution.hydraulic_owner_runtime_sha256,
                "physical_mask_sha256": _digest(physical),
                "densin_mask_sha256": _digest(densin),
                "physical_flux_sha256": _digest(flux),
            })
            return {
                "trace_q": trace,
                "physical_open_face_flux": flux,
                "physical_boundary_flux_owned_mask": physical,
                "exterior_densin_active_mask": densin,
                "reconstruction_receipt_sha256": reconstruction,
                "boundary_branch_receipt_sha256": branch,
                "owner_evaluation_sha256": _fingerprint({
                    "reconstruction": reconstruction,
                    "branch": branch,
                    "evaluation": row+1,
                }),
            }

        provider = HydraulicBoundaryTraceProviderV96(
            trace_execution, resolver=resolver,
            state_probe=lambda: owner_state)
        values = []
        for row in range(midpoint_history.shape[0]):
            current["row"] = row
            values.append(provider.resolve(
                midpoint_history[row], self._parent.bed_elevation_m,
                hydraulic_request_sha256=self._hydraulic_request_sha256,
                parent_state_sha256=self._parent.state_sha256,
                evaluation_index=row+1,
                evaluation_time_s=self._parent.time_s+.5*self._dt))
        return tuple(values)

    def begin_v94b_hydraulic_trial(
        self, parent: CoupledDensInGPUStateV96, *,
        accepted_density: V94bAcceptedDensityReceiptGPUV96,
        dt_s: float, time_s: float,
        hydraulic_request_sha256: str,
        cancelled: bool | Callable[[], bool] = False,
    ) -> None:
        _require(self._controller is not None,
                 "v96 CUDA density hook must be bound to controller")
        _require(not self._open and self._pending is None,
                 "v96 CUDA previous density trial is unfinished")
        validate_coupled_densin_gpu_state_v96(
            parent, self.outer_execution, self.provider)
        validate_v94b_accepted_density_receipt_gpu_v96(
            accepted_density, self.outer_execution, parent, self.provider)
        self._validate_runtime_identity(require_controller=True)
        self._validate_bound_controller_state()
        _require(not _cancelled(cancelled),
                 "v96 CUDA cancelled before density launch")
        _require(parent.accepted_updates < MAX_ACCEPTED_UPDATES,
                 "v96 CUDA accepted update limit was reached")
        dt = _finite(dt_s, "timestep", positive=True)
        _require(dt <= self.execution.maximum_dt_s and
                 float(time_s) == parent.time_s and
                 np.array_equal(parent.q[0], accepted_density.depth_m),
                 "v96 CUDA density handoff clock/depth differs")
        self._sync_parent(parent)
        self.accepted_bed.set(parent.bed_elevation_m)
        self._hydraulic_request_sha256 = _sha(
            hydraulic_request_sha256, "hydraulic request")
        self._cancelled_callback = cancelled
        self._initial_kernel_counts = dict(self.kernel_launch_counts)
        try:
            if self.open_density is not None:
                boundary_state = parent.accepted_boundary_scalar_state
                _require(type(boundary_state) is AcceptedBoundaryScalarStateV96,
                         "v96 CUDA open density lacks accepted scalar state")
                self._accepted_boundary_density = \
                    make_boundary_density_from_accepted_scalar_state_v96(
                        boundary_state,
                        _v92_execution(self.outer_execution.active_execution),
                        self.outer_execution.open_boundary_execution.
                            boundary_trace_execution,
                        parent_state_sha256=parent.state_sha256)
                self.open_density.set_parent_boundary_density(
                    self._accepted_boundary_density)
                self._initial_open_density_kernel_count = \
                    self.open_density.kernel_launch_count
            else:
                self._accepted_boundary_density = None
                self._initial_open_density_kernel_count = 0
            self.operator.concentration.set(accepted_density.concentration_kg_m3)
            # The v85 execution carries the hydraulic reference density, but
            # FLOW evaluates the current residual with the accepted-parent
            # scalar-derived water density.  V96 owns that live seam; do not
            # leave the operator on its initializer water-density vector.
            self.operator.water.set(accepted_density.water_density_kg_m3)
            if self.outer_execution.include_sediment_density_increment:
                self.cp.copyto(
                    self._density_concentration_input,
                    self.operator.concentration)
                expected_density = accepted_density.mixture_density_kg_m3
            else:
                # FLOW's DensIn-off route retains the accepted spatial water
                # EOS; only the suspended-sediment increment is omitted.
                self._density_concentration_input.fill(0.0)
                expected_density = accepted_density.water_density_kg_m3
            self.operator.cell_status.fill(0)
            self.kernel_launch_counts["v85_mixture_production"] += 1
            self.operator._launch("densin_mixture_v85", self.nc, (
                self._density_concentration_input, self.operator.water,
                self.operator.solid, np.int64(self.operator.ng),
                np.int64(self.nc), self.operator.mixture,
                self.operator.cell_status))
            _require(not bool(self.cp.any(
                         self.operator.cell_status != 0).item()),
                     "v96 CUDA mixture-density kernel failed")
            mixture = np.ascontiguousarray(
                self._readback(self.operator.mixture))
            _require(np.allclose(
                         mixture, expected_density,
                         rtol=2*np.finfo(float).eps, atol=0.0),
                     "v96 CUDA accepted mixture density differs")
            _require(not _cancelled(cancelled),
                     "v96 CUDA cancelled after mixture launch")
        except Exception:
            self.abort_hydraulic_trial()
            raise
        self._parent = parent
        self._accepted_concentration = accepted_density
        self._frozen_mixture_host = mixture
        self._dt, self._time = dt, float(time_s)
        self._initial_fwave_evaluations = int(
            self._controller.fwave_evaluations)
        self._initial_linear_matvecs = int(self._controller.linear_matvecs)
        self._residual_evaluations = 0
        self._open = True

    def finish_hydraulic_trial(
        self, midpoint_q, accepted_q, accepted_water_flux, *,
        dt_s: float, time_s: float,
    ) -> dict[str, object]:
        if _cancelled(self._cancelled_callback):
            self.abort_hydraulic_trial()
            raise InterruptedError(
                "v96 CUDA cancelled before hydraulic finish")
        _require(self._open and self._pending is None and
                 self._residual_endpoint_captured,
                 "v96 CUDA density finish has no complete open trial")
        _require(float(dt_s) == self._dt and float(time_s) == self._time,
                 "v96 CUDA hydraulic finish clock differs")
        midpoint = self._device(midpoint_q, (3, self.nc), "accepted midpoint")
        endpoint = self._device(accepted_q, (3, self.nc), "accepted endpoint")
        water = self._device(
            accepted_water_flux, (self.nf,), "accepted water flux")
        try:
            _require(bool(self.cp.array_equal(
                         water, self._controller.midpoint_water).item()),
                     "v96 CUDA accepted water flux differs from the "
                     "authoritative controller midpoint flux")
            count = self._residual_evaluations
            _require(count >= 1 and
                     int(self._controller.fwave_evaluations)-
                     self._initial_fwave_evaluations == count,
                     "v96 CUDA density/f-wave residual counts differ")
            launch_delta = {
                name: self.kernel_launch_counts[name]-
                    self._initial_kernel_counts[name]
                for name in self.kernel_launch_counts
            }
            _require(launch_delta == {
                         "v85_mixture_production": 1,
                         "v85_face_residual": count,
                         "v85_projection_residual": count,
                         "resident_rate_add": count,
                         "host_numerical_fallback": 0,
                     }, "v96 CUDA density kernel ownership differs")
            arrays = [
                self._readback(value) for value in (
                    self.residual_base, self.residual_endpoint, endpoint,
                    midpoint, self.operator.mixture, self.base_rate_last,
                    self.operator.density_rate, self.full_rate_last,
                    self._controller.residual,
                    self.operator.face_acceleration,
                    self.operator.cell_acceleration, water)
            ]
            (base_h, residual_endpoint_h, candidate_h, midpoint_h,
             mixture_h, base_rate_h, density_rate_h, full_rate_h,
             residual_h, face_h, cell_h, water_h) = (
                np.ascontiguousarray(value) for value in arrays)
            history = np.ascontiguousarray(
                self._readback(self.depth_history[:count]))
            midpoint_history = np.ascontiguousarray(
                self._readback(self.midpoint_q_history[:count]))
            trace_receipts = self._make_boundary_trace_receipts(
                midpoint_history,
                np.ascontiguousarray(self._readback(
                    self.boundary_trace_history[:count])),
                np.ascontiguousarray(self._readback(
                    self.boundary_flux_history[:count])))
            open_kernel_count = (0 if self.open_density is None else
                self.open_density.kernel_launch_count-
                self._initial_open_density_kernel_count)
            concentration = np.ascontiguousarray(
                self._readback(self.operator.concentration))
            boundary_ledger = np.ascontiguousarray(
                self._readback(self._controller.water_ledger))
            _require(np.array_equal(midpoint_h,
                                    self._readback(self.midpoint_last)) and
                     np.array_equal(mixture_h,
                                    self._frozen_mixture_host) and
                     np.array_equal(
                         concentration,
                         self._accepted_concentration.concentration_kg_m3),
                     "v96 CUDA accepted parent density/midpoint mutated")
            _require(not _cancelled(self._cancelled_callback),
                     "v96 CUDA cancelled after hydraulic readback")
            depth_digests = tuple(_digest(row) for row in history)
            mass_delta = float(np.sum(
                self.execution.base_execution.cell_area_m2*
                (candidate_h[0]-self._parent.q[0])))
            water_residual = mass_delta-float(np.sum(boundary_ledger))
            approximate = int(self._controller.linear_matvecs)-\
                self._initial_linear_matvecs
            placeholder = HydraulicDensInTrialGPUV96(
                self.outer_execution.fingerprint,
                self._parent.state_sha256,
                self._accepted_concentration,
                _digest(self._parent.bed_elevation_m), self._dt,
                _array(base_h, (3, self.nc), "residual base"),
                _array(residual_endpoint_h, (3, self.nc),
                       "residual endpoint"),
                _array(candidate_h, (3, self.nc), "candidate"),
                _array(midpoint_h, (3, self.nc), "midpoint"),
                _array(mixture_h, (self.nc,), "mixture", positive=True),
                _array(base_rate_h, (3, self.nc), "base rate"),
                _array(density_rate_h, (2, self.nc), "density rate"),
                _array(full_rate_h, (3, self.nc), "full rate"),
                _array(residual_h, (3, self.nc), "residual"),
                _array(face_h, (self.nf,), "face acceleration"),
                _array(cell_h, (2, self.nc), "cell acceleration"),
                _array(water_h, (self.nf,), "water flux"),
                _array(boundary_ledger, (boundary_ledger.size,),
                       "water boundary ledger"),
                _array(midpoint_history, (count, 3, self.nc),
                       "midpoint history"),
                self._accepted_boundary_density, trace_receipts,
                open_kernel_count,
                depth_digests, count, count, self.nf*count,
                approximate, 0, 0, 0, count-1,
                float(np.max(np.abs(residual_h))), water_residual,
                BACKEND, "")
            trial = HydraulicDensInTrialGPUV96(
                **{**placeholder.__dict__,
                   "trial_sha256": _hydraulic_trial_token(placeholder)})
            _validate_hydraulic_trial(
                self.outer_execution, self._parent, self.provider, trial)
        except Exception:
            self.abort_hydraulic_trial()
            raise
        self._pending = trial
        self._open = False
        return {
            "scope_id": SCOPE_ID + ":hydraulic-trial-receipt",
            "trial_sha256": trial.trial_sha256,
            "accepted_v94b_density_receipt_sha256":
                trial.accepted_density.receipt_sha256,
            "residual_evaluation_count": count,
            "density_source_evaluation_count": count,
            "augmented_riemann_call_count": self.nf*count,
            "post_convergence_density_application_count": 0,
            "backend": BACKEND,
            "outer_transaction_status": "provisional",
        }

    def accept_outer(self, trial: HydraulicDensInTrialGPUV96,
                     accepted_state_sha256: str) -> None:
        _require(self._pending is trial,
                 "v96 CUDA density trial is not pending")
        self.accepted_q.set(trial.candidate_q)
        self._resident_parent_sha256 = accepted_state_sha256
        self.last_accepted_receipt = {
            "scope_id": SCOPE_ID + ":hydraulic-trial-receipt",
            "trial_sha256": trial.trial_sha256,
            "accepted_state_sha256": accepted_state_sha256,
            "outer_transaction_status": "accepted",
            "actual_cuda_kernels": True,
            "automatic_cpu_fallback": False,
        }
        self._clear()

    def discard_outer(self) -> None:
        self._clear()

    def _clear(self) -> None:
        super()._clear()
        self._accepted_boundary_density = None
        self._hydraulic_request_sha256 = None
        self._initial_open_density_kernel_count = 0

    def commit(self, *args, **kwargs):  # pragma: no cover - fail-closed API
        raise CoupledDensInGPUV96Error(
            "v96 CUDA density hook is provisional; outer owner commits")


class _CompositeResidualHookV96:
    """One controller hook composing v91 then v85 on every residual."""

    is_v95_densin_residual_consumer = True
    is_v96_coupled_residual_consumer = True

    def __init__(self, active_runtime, density_hook):
        self.active_runtime = active_runtime
        self.density_hook = density_hook

    def bind_controller(self, controller) -> None:
        self.active_runtime.bind_controller(controller)
        self.density_hook.bind_controller(controller)

    def require_open_for_controller(self, dt_s, time_s, q) -> None:
        self.active_runtime.require_open_for_controller(dt_s, time_s, q)
        self.density_hook.require_open_for_controller(dt_s, time_s, q)

    def add_momentum_rate(self, q, full_rate) -> None:
        self.active_runtime.add_momentum_rate(q, full_rate)
        self.density_hook.add_momentum_rate(q, full_rate)

    def accepted_bed_for_controller(self):
        return self.density_hook.accepted_bed_for_controller()

    def capture_converged_residual_endpoint(self, base, endpoint) -> None:
        self.density_hook.capture_converged_residual_endpoint(base, endpoint)

    def finish_hydraulic_trial(self, midpoint, endpoint, water, **clock):
        try:
            v91 = self.active_runtime.finish_hydraulic_trial(
                midpoint, endpoint, water, **clock)
            density = self.density_hook.finish_hydraulic_trial(
                midpoint, endpoint, water, **clock)
            return {"v91": v91, "density": density}
        except Exception:
            self.abort_hydraulic_trial()
            raise

    def abort_hydraulic_trial(self) -> None:
        errors = []
        for owner in (self.active_runtime, self.density_hook):
            try:
                owner.abort_hydraulic_trial()
            except Exception as error:  # pragma: no cover - injected gate
                errors.append(error)
        if errors:
            raise RuntimeError(
                "v96 CUDA composite residual abort failed") from errors[0]


@dataclass(frozen=True)
class CoupledDensInOwnerCountsGPUV96:
    native_hydraulic_owner_count: int
    water_face_flux_owner_count: int
    lagged_v91_residual_owner_count: int
    density_residual_owner_count: int
    scalar_lifecycle_owner_count: int
    authoritative_exner_owner_count: int
    hles_clock_owner_count: int
    duplicate_hydraulic_owner_count: int
    duplicate_scalar_owner_count: int
    duplicate_exner_owner_count: int


@dataclass(frozen=True)
class CoupledDensInGPUTrialV96:
    execution_fingerprint: str
    parent_state_sha256: str
    accepted_density: V94bAcceptedDensityReceiptGPUV96
    current_handoff: CurrentClosureHandoffV90
    dt_s: float
    hydraulic_trial: HydraulicDensInTrialGPUV96
    active_trial: ActiveSuspendedControlTrialV94
    hles_trial: HlesProducerTrialV90
    owner_counts: CoupledDensInOwnerCountsGPUV96
    candidate: CoupledDensInGPUStateV96
    backend: str
    trial_sha256: str


def _owner_counts_payload(value: CoupledDensInOwnerCountsGPUV96):
    return tuple(value.__dict__.values())


def _trial_token(value: CoupledDensInGPUTrialV96) -> str:
    return _fingerprint({
        "scope_id": SCOPE_ID + ":trial",
        "execution_fingerprint": value.execution_fingerprint,
        "parent_state_sha256": value.parent_state_sha256,
        "accepted_density_receipt_sha256":
            value.accepted_density.receipt_sha256,
        "current_handoff_receipt_sha256":
            value.current_handoff.receipt_sha256,
        "dt_s": value.dt_s,
        "hydraulic_trial_sha256": value.hydraulic_trial.trial_sha256,
        "active_trial_sha256": value.active_trial.trial_sha256,
        "hles_trial_sha256": value.hles_trial.trial_sha256,
        "owner_counts": list(_owner_counts_payload(value.owner_counts)),
        "candidate_state_sha256": value.candidate.state_sha256,
        "backend": value.backend,
    })


def _closed_scalar_inputs(
    runtime, parent, endpoint, shear_velocity_m_s, *,
    prescribed_boundary_concentration_kg_m3=None,
    prescribed_boundary_fluid_transport_scalar=None,
):
    cp = runtime.cp
    execution = runtime.execution
    v92_execution = _v92_execution(execution.active_execution)
    v92 = _v92_state(parent.active_state)
    grid = _grid_execution(execution.active_execution)
    ng, nc, nf, nb = (execution.fraction_count, execution.cell_count,
                      execution.face_count, v92_execution.boundary_count)
    shear = _array(
        shear_velocity_m_s, (nc,), "shear velocity", nonnegative=True)
    concentration = np.ascontiguousarray(
        v92.suspended_state.suspended_mass_kg_m2/v92.q[0][None, :])
    left = np.ascontiguousarray(concentration[:, grid.owner])
    right = np.array(left, copy=True, order="C")
    internal = grid.neighbour >= 0
    right[:, internal] = concentration[:, grid.neighbour[internal]]
    fluid = np.vstack((v92.temperature_cell_c, v92.salinity_cell_ppt))
    fluid_left = np.ascontiguousarray(fluid[:, grid.owner])
    fluid_right = np.array(fluid_left, copy=True, order="C")
    fluid_right[:, internal] = fluid[:, grid.neighbour[internal]]
    if execution.open_boundary_execution is None:
        _require(prescribed_boundary_concentration_kg_m3 is None and
                 prescribed_boundary_fluid_transport_scalar is None,
                 "v96 CUDA closed route received open scalar forcing")
        prescribed = np.zeros((ng, nb), dtype=np.float64)
        prescribed_fluid = np.zeros((2, nb), dtype=np.float64)
        seen: set[int] = set()
        for face in np.flatnonzero(grid.boundary_index >= 0):
            boundary = int(grid.boundary_index[face])
            if boundary not in seen:
                owner = int(grid.owner[face])
                prescribed[:, boundary] = concentration[:, owner]
                prescribed_fluid[:, boundary] = fluid[:, owner]
                seen.add(boundary)
    else:
        # These are the current prescribed/time-series inputs.  They are
        # deliberately distinct from ``accepted_boundary_scalar_state``:
        # that face image is lagged evidence for the *next hydraulic DensIn*
        # solve and is never substituted for current scalar forcing.
        _require(prescribed_boundary_concentration_kg_m3 is not None and
                 prescribed_boundary_fluid_transport_scalar is not None,
                 "v96 CUDA open scalar forcing must be supplied explicitly")
        prescribed = _array(
            prescribed_boundary_concentration_kg_m3, (ng, nb),
            "prescribed boundary suspended concentration", nonnegative=True)
        prescribed_fluid = _array(
            prescribed_boundary_fluid_transport_scalar, (2, nb),
            "prescribed boundary transported fluid")
    face_depth = np.asarray(endpoint[0, grid.owner], dtype=np.float64).copy()
    face_depth[internal] = .5*(
        endpoint[0, grid.owner[internal]]+
        endpoint[0, grid.neighbour[internal]])
    transport_active = np.array(internal, copy=True)
    if execution.open_boundary_execution is not None:
        open_mask = (
            execution.open_boundary_execution.boundary_trace_execution.
            open_mask)
        trace_receipts = runtime.density.pending_trial.boundary_trace_receipts
        _require(trace_receipts,
                 "v96 CUDA open scalar path lacks a physical trace receipt")
        final_trace = trace_receipts[-1]
        face_depth[open_mask] = final_trace.trace_q[0, open_mask]
        transport_active |= open_mask
    def device(value, *, integer=False):
        return cp.ascontiguousarray(cp.asarray(
            value, dtype=cp.int32 if integer else cp.float64))
    return {
        "reconstructed_left_fluid": device(fluid_left),
        "reconstructed_right_fluid": device(fluid_right),
        "prescribed_boundary_fluid": device(prescribed_fluid),
        "reconstructed_left_concentration_kg_m3": device(left),
        "reconstructed_right_concentration_kg_m3": device(right),
        "prescribed_boundary_concentration_kg_m3": device(prescribed),
        "accepted_face_exchange_area_m2": device(
            np.where(transport_active,
                     face_depth*grid.face_length_m, 0.0)),
        "transport_face_active": device(transport_active, integer=True),
        "shear_velocity_m_s": device(shear),
        "hydro_constant": device(np.ones(nc, dtype=np.int32), integer=True),
    }


def _face_velocity(execution, endpoint, face_flux, boundary_trace_receipts=()):
    grid = _grid_execution(execution.active_execution)
    internal = grid.neighbour >= 0
    depth = np.asarray(endpoint[0, grid.owner], dtype=np.float64).copy()
    depth[internal] = .5*(
        endpoint[0, grid.owner[internal]]+
        endpoint[0, grid.neighbour[internal]])
    if execution.open_boundary_execution is not None:
        _require(boundary_trace_receipts,
                 "v96 CUDA open HLES path lacks boundary trace evidence")
        open_mask = (
            execution.open_boundary_execution.boundary_trace_execution.
            open_mask)
        depth[open_mask] = boundary_trace_receipts[-1].trace_q[0, open_mask]
    _require(np.all(depth > 0.0),
             "v96 CUDA HLES face depth left fully wet scope")
    return np.ascontiguousarray(face_flux/depth)


class ResidentCoupledDensInLifecycleV96:
    """One atomic resident owner for the bounded live-CUDA v96 route."""

    is_v96_coupled_densin_owner = True

    def __init__(
        self, execution: CoupledDensInGPUExecutionV96,
        active_runtime: ResidentActiveSuspendedControlLifecycleV94,
        hles_runtime: ResidentHlesProducerLifecycleV90,
        model: object,
        initial_state: CoupledDensInGPUStateV96,
    ):
        execution_document_v96(execution)
        zero_mode = (execution.active_execution.base_execution.
                     suspended_vector_mode ==
                     SUSPENDED_VECTOR_EXACT_ZERO_V93)
        expected_active_type = (
            ResidentActiveSuspendedControlLifecycleV94b if zero_mode else
            ResidentActiveSuspendedControlLifecycleV94)
        _require(type(active_runtime) is expected_active_type and
                 active_runtime.control_execution is
                    execution.active_execution and
                 type(active_runtime.q3_provider) is
                    ResidentQ3V83ActiveExnerAdapterV94,
                 "v96 CUDA requires its exact reviewed v94/v94b owner")
        hles_execution = _v92_execution(
            execution.active_execution).base_execution.hles_execution
        _require(type(hles_runtime) is ResidentHlesProducerLifecycleV90 and
                 hles_runtime.execution is hles_execution,
                 "v96 CUDA requires the exact reviewed HLES resident owner")
        controller = getattr(model, "implicit_midpoint", None)
        _require(controller is not None and
                 getattr(controller, "solver", None) is model and
                 not controller.component_composition and
                 not controller.prognostic_secondary_composition and
                 not controller.suspended_composition and
                 controller.shared_turbulence_v91 is active_runtime.inner and
                 controller.densin_residual_v95 is None,
                 "v96 CUDA requires the exact v94b hydraulic owner binding")
        cp = active_runtime.cp
        _require(getattr(controller, "cp", None) is cp and
                 getattr(hles_runtime, "cp", None) is cp,
                 "v96 CUDA owners must share one CuPy backend")
        boundary_face = np.asarray(
            execution.hydraulic_execution.base_execution.neighbour) < 0
        controller_mode = cp.asnumpy(controller.boundary_mode)
        if execution.open_boundary_execution is None:
            _require(np.all(controller_mode[boundary_face] == 0),
                     "v96 CUDA closed stage requires reflective faces")
        else:
            trace_execution = (execution.open_boundary_execution.
                               boundary_trace_execution)
            _require(np.array_equal(
                         controller_mode, trace_execution.boundary_mode) and
                     np.all(trace_execution.open_mask ==
                            ((trace_execution.neighbour < 0) &
                             (trace_execution.boundary_index >= 0))),
                     "v96 CUDA controller and open-boundary modes differ")
        _require(_provider_is_closed(active_runtime.q3_provider),
                 "v96 CUDA bounded stage requires closed Q3 boundaries")
        self.execution = execution
        self.active = active_runtime
        self.hles = hles_runtime
        self.model = model
        self.controller = controller
        self.cp = cp
        self.provider = active_runtime.q3_provider
        self.density = ResidentV94bDensInHookV96(
            execution, self.provider)
        self.composite = _CompositeResidualHookV96(
            active_runtime, self.density)
        # v94b initially occupies the historical v91-only hook.  V96 is the
        # first owner allowed to replace that one binding with a composite
        # residual owner; the same v94b object remains the v91 child and the
        # v85 density child is added exactly once.  Restore the old binding if
        # construction cannot complete.
        controller.shared_turbulence_v91 = None
        try:
            controller.bind_densin_residual_v95(self.composite)
        except Exception:
            controller.shared_turbulence_v91 = active_runtime.inner
            raise
        self._parent = None
        self._trial = None
        self._snapshot = None
        self._accepted_state = None
        self.last_accepted_receipt = None
        self.kernel_launch_counts = {
            "joint_owner_publication": 0,
            "duplicate_hydraulic_owner": 0,
            "duplicate_scalar_owner": 0,
            "duplicate_exner_owner": 0,
            "host_numerical_fallback": 0,
        }
        sources = (
            Path(__file__).resolve(),
            _ROOT/"braids_gpu/densin_residual_v95.py",
            _ROOT/"braids_gpu/active_suspended_control_v94.py",
            _ROOT/"braids_gpu/active_suspended_control_v94b.py",
            _ROOT/"braids_gpu/implicit_midpoint_fwave.py",
            _ROOT/"braids_gpu/implicit_midpoint_fwave.cu",
            _ROOT/"braids_gpu/coupled_open_boundary_v96.py",
            _ROOT/"braids_gpu/coupled_open_boundary_v96.cu",
            _ROOT/"braids_core/sediment/"
                  "hydraulic_boundary_trace_cpu_v96.py",
            _ROOT/"braids_core/sediment/"
                  "accepted_boundary_scalar_state_cpu_v96.py",
        )
        self.runtime_receipt = {
            "scope_id": SCOPE_ID + ":runtime",
            "execution_fingerprint": execution.fingerprint,
            "backend": BACKEND,
            "integrator": "implicit_fwave_midpoint",
            "actual_cuda_residual": True,
            "actual_cuda_scalar_exner": True,
            "typed_v94_or_v94b_parent_no_v78_shadow": True,
            "lagged_v91_then_v85_each_true_residual": True,
            "post_convergence_density_application_count": 0,
            "bed_elevation_feedback": True,
            "spatial_accepted_water_eos": True,
            "density_mode": (
                "water_plus_sediment" if
                execution.include_sediment_density_increment else
                "water_eos"),
            "nonzero_lagged_v91_supported": True,
            "open_boundary_supported":
                execution.open_boundary_execution is not None,
            "automatic_cpu_fallback": False,
            "nested_v95_runtime_receipt_sha256":
                _portable_v95_runtime_identity(self.density),
            "nested_active_execution_fingerprint":
                execution.active_execution.fingerprint,
            "nested_hles_runtime_receipt_sha256":
                _fingerprint(hles_runtime.runtime_receipt),
            "source_sha256": {
                str(path.relative_to(_ROOT)).replace("\\", "/"):
                    hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sources
            },
        }
        self.runtime_receipt["receipt_sha256"] = _fingerprint(
            self.runtime_receipt)
        validate_coupled_densin_gpu_state_v96(
            initial_state, execution, self.provider)
        self._publish_model_state(initial_state, initial=True)
        self._accepted_state = initial_state

    def _validate_runtime_identity(self) -> None:
        expected = {
            key: hashlib.sha256((_ROOT/key).read_bytes()).hexdigest()
            for key in self.runtime_receipt["source_sha256"]
        }
        _require(self.runtime_receipt["source_sha256"] == expected and
                 self.runtime_receipt["nested_v95_runtime_receipt_sha256"] ==
                    _portable_v95_runtime_identity(self.density) and
                 self.runtime_receipt["nested_hles_runtime_receipt_sha256"] ==
                    _fingerprint(self.hles.runtime_receipt) and
                 self.runtime_receipt["receipt_sha256"] == _fingerprint({
                     key: value for key, value in self.runtime_receipt.items()
                     if key != "receipt_sha256"
                 }),
                 "v96 CUDA runtime/source identity differs")

    def _model_arrays(self) -> dict[str, object]:
        result: dict[str, object] = {}
        for prefix, mapping in (("state", self.model.state),
                                ("ledger", self.model.ledger)):
            if isinstance(mapping, Mapping):
                for key, value in mapping.items():
                    if isinstance(value, self.cp.ndarray):
                        result[f"{prefix}.{key}"] = value
        boundary = getattr(self.model, "boundary", None)
        arrays = getattr(boundary, "arrays", None)
        if isinstance(arrays, Mapping):
            for key, value in arrays.items():
                if isinstance(value, self.cp.ndarray):
                    result[f"boundary.arrays.{key}"] = value
        for name in ("ledger", "last_outflow", "return_clock"):
            value = getattr(boundary, name, None)
            if isinstance(value, self.cp.ndarray):
                result[f"boundary.{name}"] = value
        # Boundary mode bridges own per-residual latches, trace records and
        # status arrays outside the public arrays mapping.  They are mutable
        # even for commit=False nonlinear evaluations and therefore belong to
        # the outer atomic snapshot as much as q and the reversal memories.
        for name, value in getattr(boundary, "__dict__", {}).items():
            if isinstance(value, self.cp.ndarray):
                result.setdefault(f"boundary.private.{name}", value)
            elif isinstance(value, Mapping):
                for child, array in value.items():
                    if isinstance(array, self.cp.ndarray):
                        result.setdefault(
                            f"boundary.private.{name}.{child}", array)
            elif hasattr(value, "__dict__"):
                for child, array in value.__dict__.items():
                    if isinstance(array, self.cp.ndarray):
                        result.setdefault(
                            f"boundary.private.{name}.{child}", array)
                    elif isinstance(array, Mapping):
                        for grandchild, nested in array.items():
                            if isinstance(nested, self.cp.ndarray):
                                result.setdefault(
                                    f"boundary.private.{name}.{child}."
                                    f"{grandchild}", nested)
        result["controller.predictor_rate"] = self.controller.predictor_rate
        result["controller.midpoint_boundary_trace"] = \
            self.controller.midpoint_boundary_trace
        result["controller.midpoint_physical_flux"] = \
            self.controller.midpoint_physical_flux
        return result

    def _take_snapshot(self, parent: CoupledDensInGPUStateV96) -> dict:
        arrays = self._model_arrays()
        return {
            "arrays": {key: value.copy() for key, value in arrays.items()},
            "pointers": {key: int(value.data.ptr)
                         for key, value in arrays.items()},
            "have_predictor": bool(self.controller.have_predictor),
            "accepted_steps": int(self.controller.accepted_steps),
            "rejected_steps": int(self.controller.rejected_steps),
            "fwave_evaluations": int(self.controller.fwave_evaluations),
            "linear_matvecs": int(self.controller.linear_matvecs),
            "last_diagnostics": copy.deepcopy(
                self.controller.last_diagnostics),
            "active_checkpoint": self.active.checkpoint(parent.active_state),
            "hles_checkpoint": self.hles.checkpoint(parent.hles_state),
            "provider_checkpoint": self.provider.checkpoint(),
            "active_receipt": copy.deepcopy(self.active.last_accepted_receipt),
            "inner_receipt": copy.deepcopy(
                self.active.inner.last_accepted_receipt),
            "base_receipt": copy.deepcopy(
                self.active.inner.base.last_accepted_receipt),
            "density_receipt": copy.deepcopy(
                self.density.last_accepted_receipt),
            "outer_receipt": copy.deepcopy(self.last_accepted_receipt),
        }

    def _restore_model_snapshot(self, snapshot: Mapping[str, object]) -> None:
        arrays = self._model_arrays()
        _require(set(arrays) == set(snapshot["arrays"]),
                 "v96 CUDA model array inventory changed during transaction")
        for key, value in arrays.items():
            _require(int(value.data.ptr) == snapshot["pointers"][key],
                     "v96 CUDA model array pointer changed: " + key)
            self.cp.copyto(value, snapshot["arrays"][key])
        self.controller.have_predictor = snapshot["have_predictor"]
        self.controller.accepted_steps = snapshot["accepted_steps"]
        self.controller.rejected_steps = snapshot["rejected_steps"]
        self.controller.fwave_evaluations = snapshot["fwave_evaluations"]
        self.controller.linear_matvecs = snapshot["linear_matvecs"]
        self.controller.last_diagnostics = copy.deepcopy(
            snapshot["last_diagnostics"])

    def _publish_model_state(
        self, state: CoupledDensInGPUStateV96, *, initial: bool = False,
    ) -> None:
        validate_coupled_densin_gpu_state_v96(
            state, self.execution, self.provider)
        self.cp.copyto(self.model.state["q"], self.cp.asarray(state.q))
        self.cp.copyto(
            self.controller.predictor_rate,
            self.cp.asarray(state.predictor_rate_m2_s2))
        self.controller.have_predictor = state.have_predictor
        water = self.model.ledger.get("water")
        _require(isinstance(water, self.cp.ndarray) and
                 water.shape == state.hydraulic_boundary_ledger_m3.shape,
                 "v96 CUDA model water ledger shape differs")
        self.cp.copyto(
            water, self.cp.asarray(state.hydraulic_boundary_ledger_m3))
        self.controller.accepted_steps = state.accepted_updates
        if not initial:
            self.kernel_launch_counts["joint_owner_publication"] += 1

    def _close_children_for_restore(self) -> None:
        try:
            if self.active._control_parent is not None:
                self.active.abort_hydraulic_trial()
        finally:
            self.density.discard_outer()
        if self.hles._open_trial is not None:
            # A trial object is retained on the outer trial when available;
            # the core rollback closes the exact resident transaction.
            if self._trial is not None:
                try:
                    self.hles.rollback(
                        self._parent.hles_state, self._trial.hles_trial)
                except Exception:
                    self.hles._open_parent = self.hles._open_trial = None
                    self.hles._open_kind = None
            else:
                self.hles._open_parent = self.hles._open_trial = None
                self.hles._open_kind = None

    def _restore_transaction(self) -> None:
        snapshot = self._snapshot
        parent = self._parent
        if snapshot is None or parent is None:
            return
        errors = []
        try:
            self._close_children_for_restore()
        except Exception as error:
            errors.append(error)
        try:
            self.active.restore_checkpoint(snapshot["active_checkpoint"])
        except Exception as error:
            errors.append(error)
        try:
            self.hles.restore_checkpoint(snapshot["hles_checkpoint"])
        except Exception as error:
            errors.append(error)
        try:
            self._restore_model_snapshot(snapshot)
        except Exception as error:
            errors.append(error)
        self.active.last_accepted_receipt = copy.deepcopy(
            snapshot["active_receipt"])
        self.active.inner.last_accepted_receipt = copy.deepcopy(
            snapshot["inner_receipt"])
        self.active.inner.base.last_accepted_receipt = copy.deepcopy(
            snapshot["base_receipt"])
        self.density.last_accepted_receipt = copy.deepcopy(
            snapshot["density_receipt"])
        self.last_accepted_receipt = copy.deepcopy(snapshot["outer_receipt"])
        if self.provider.checkpoint() != snapshot["provider_checkpoint"]:
            try:
                self.provider.restore_checkpoint(
                    snapshot["provider_checkpoint"])
            except Exception as error:
                errors.append(error)
        if errors:
            raise RuntimeError(
                "v96 CUDA failed to restore the atomic parent") from errors[0]

    def _clear(self) -> None:
        self._parent = self._trial = self._snapshot = None

    def begin_trial(
        self, parent: CoupledDensInGPUStateV96, *,
        shear_velocity_m_s: object,
        closure_chezy_mhalf_s: object,
        q3_inputs: Mapping[str, object],
        dt_s: float,
        prescribed_boundary_concentration_kg_m3: object | None = None,
        prescribed_boundary_fluid_transport_scalar: object | None = None,
        cancelled: bool | Callable[[], bool] = False,
    ) -> CoupledDensInGPUTrialV96:
        _require(self._parent is None and self._trial is None,
                 "v96 CUDA previous outer transaction is unfinished")
        validate_coupled_densin_gpu_state_v96(
            parent, self.execution, self.provider)
        self._validate_runtime_identity()
        _require(parent is self._accepted_state or
                 parent.state_sha256 == self._accepted_state.state_sha256,
                 "v96 CUDA parent is not the resident accepted state")
        _require(parent.accepted_updates < MAX_ACCEPTED_UPDATES,
                 "v96 CUDA accepted update limit was reached")
        _require(isinstance(q3_inputs, Mapping),
                 "v96 CUDA Q3 inputs must be a mapping")
        if _cancelled(cancelled):
            raise InterruptedError("v96 CUDA cancelled before child work")
        dt = _finite(dt_s, "timestep", positive=True)
        _require(dt <= self.execution.hydraulic_execution.maximum_dt_s,
                 "v96 CUDA timestep exceeds v95 bound")
        self._parent = parent
        self._snapshot = self._take_snapshot(parent)
        active_trial = None
        hles_trial = None
        try:
            _require(np.array_equal(
                         self.cp.asnumpy(self.model.state["q"]), parent.q) and
                     bool(self.controller.have_predictor) ==
                        parent.have_predictor and
                     np.array_equal(
                         self.cp.asnumpy(self.controller.predictor_rate),
                         parent.predictor_rate_m2_s2),
                     "v96 CUDA controller is not synchronized to parent")
            density = make_v94b_accepted_density_receipt_gpu_v96(
                self.execution, parent, self.provider)
            request = _fingerprint({
                "scope_id": SCOPE_ID + ":hydraulic-request",
                "execution_fingerprint": self.execution.fingerprint,
                "parent_state_sha256": parent.state_sha256,
                "accepted_density_receipt_sha256": density.receipt_sha256,
                "dt_s": dt,
            })
            hles_execution = _v92_execution(
                self.execution.active_execution).base_execution.hles_execution
            handoff = make_current_closure_handoff_v90(
                hles_execution, parent.hles_state,
                hydraulic_trial_request_sha256=request)
            self.active.begin_hydraulic_trial(
                parent.active_state, parent.hles_state, handoff,
                dt_s=dt, time_s=parent.time_s, cancelled=cancelled)
            self.density.begin_v94b_hydraulic_trial(
                parent, accepted_density=density, dt_s=dt,
                time_s=parent.time_s,
                hydraulic_request_sha256=request,
                cancelled=cancelled)
            if _cancelled(cancelled):
                raise InterruptedError(
                    "v96 CUDA cancelled before live hydraulic solve")
            result = self.model.step(dt, parent.time_s, commit=False)
            if int(result["status"].item()) != 0:
                raise CoupledDensInGPUV96Error(
                    "v96 CUDA live implicit hydraulic trial rejected: " +
                    str(result.get("diagnostics", {}).get("error", "unknown")))
            hydraulic = self.density.pending_trial
            _require(type(hydraulic) is HydraulicDensInTrialGPUV96,
                     "v96 CUDA live controller produced no hydraulic trial")
            _validate_hydraulic_trial(
                self.execution, parent, self.provider, hydraulic)
            if _cancelled(cancelled):
                raise InterruptedError(
                    "v96 CUDA cancelled before scalar/Q3 work")
            scalar = _closed_scalar_inputs(
                self, parent, hydraulic.candidate_q, shear_velocity_m_s,
                prescribed_boundary_concentration_kg_m3=
                    prescribed_boundary_concentration_kg_m3,
                prescribed_boundary_fluid_transport_scalar=
                    prescribed_boundary_fluid_transport_scalar)
            active_trial = self.active.begin_scalar_trial(
                q3_inputs=dict(q3_inputs), cancelled=cancelled, **scalar)
            v92_trial = active_trial.base_v93_trial.base_v92_trial
            next_boundary_scalar_state = (
                None if self.execution.open_boundary_execution is None else
                make_accepted_boundary_scalar_state_from_v92_trial_v96(
                    _v92_execution(self.execution.active_execution),
                    self.execution.open_boundary_execution.
                        boundary_trace_execution,
                    v92_trial))
            _require(np.array_equal(
                         v92_trial.candidate.q,
                         hydraulic.candidate_q) and
                     np.array_equal(
                         v92_trial.hydraulic_face_receipt.
                         accepted_water_flux_m2_s,
                         hydraulic.face_water_flux_m2_s) and
                     v92_trial.cap_receipt.
                         current_handoff_receipt_sha256 ==
                         handoff.receipt_sha256 and
                     v92_trial.hydraulic_receipt.
                         current_handoff_receipt_sha256 ==
                         handoff.receipt_sha256,
                     "v96 CUDA v92 did not consume sole native endpoint")
            face_velocity = _face_velocity(
                self.execution, hydraulic.candidate_q,
                hydraulic.face_water_flux_m2_s,
                hydraulic.boundary_trace_receipts)
            nc, nf = self.execution.cell_count, self.execution.face_count
            active_face_h = np.ones(nf, dtype=np.int32)
            active_cell_h = np.ones(nc, dtype=np.int32)
            chezy_h = _array(
                closure_chezy_mhalf_s, (nc,), "closure Chezy",
                positive=True)
            closure_inputs_h = {
                "accepted_face_normal_velocity_m_s": face_velocity,
                "active_face": active_face_h,
                "closure_depth_m": hydraulic.candidate_q[0],
                "closure_chezy_mhalf_s": chezy_h,
                "active_cell": active_cell_h,
            }
            step = v92_trial.hydraulic_receipt.base_step_receipt
            hles_input = make_hles_input_receipt_v90(
                hles_execution, parent.hles_state,
                accepted_step_receipt=step, current_handoff=handoff,
                closure_interval_s=dt, **closure_inputs_h)
            hles_trial = self.hles.begin_trial(
                parent.hles_state, accepted_step_receipt=step,
                hles_input_receipt=hles_input, current_handoff=handoff,
                accepted_face_normal_velocity_m_s=self.cp.asarray(
                    face_velocity),
                active_face=self.cp.asarray(active_face_h),
                closure_depth_m=self.cp.asarray(
                    hydraulic.candidate_q[0]),
                closure_chezy_mhalf_s=self.cp.asarray(chezy_h),
                active_cell=self.cp.asarray(active_cell_h),
                cancelled=False)
            if _cancelled(cancelled):
                raise InterruptedError(
                    "v96 CUDA cancelled after scalar/HLES work")
            chain = _fingerprint({
                "previous": parent.accepted_chain_sha256,
                "accepted_density": density.receipt_sha256,
                "hydraulic_trial": hydraulic.trial_sha256,
                "active_trial": active_trial.trial_sha256,
                "hles_trial": hles_trial.trial_sha256,
                "accepted_boundary_scalar_state": (
                    None if next_boundary_scalar_state is None else
                    next_boundary_scalar_state.state_sha256),
                "dt_s": dt,
            })
            candidate = _wrap_state(
                self.execution, active_state=active_trial.candidate,
                hles_state=hles_trial.candidate,
                predictor_rate_m2_s2=hydraulic.full_rate,
                have_predictor=True,
                hydraulic_boundary_ledger_m3=(
                    parent.hydraulic_boundary_ledger_m3+
                    hydraulic.water_boundary_ledger_m3),
                accepted_chain_sha256=chain,
                accepted_boundary_scalar_state=next_boundary_scalar_state)
            counts = CoupledDensInOwnerCountsGPUV96(
                1, 1, 1, 1, 1, 1, 1, 0, 0, 0)
            placeholder = CoupledDensInGPUTrialV96(
                self.execution.fingerprint, parent.state_sha256,
                density, handoff, dt, hydraulic, active_trial, hles_trial,
                counts, candidate, BACKEND, "")
            trial = CoupledDensInGPUTrialV96(
                **{**placeholder.__dict__,
                   "trial_sha256": _trial_token(placeholder)})
            self._trial = trial
            self._validate_trial(parent, trial)
            return trial
        except Exception:
            try:
                self._restore_transaction()
            finally:
                self._clear()
            raise

    def _validate_trial(
        self, parent: CoupledDensInGPUStateV96,
        trial: CoupledDensInGPUTrialV96,
    ) -> None:
        validate_coupled_densin_gpu_state_v96(
            parent, self.execution, self.provider)
        _require(type(trial) is CoupledDensInGPUTrialV96 and
                 trial.execution_fingerprint == self.execution.fingerprint and
                 trial.parent_state_sha256 == parent.state_sha256 and
                 trial.backend == BACKEND and
                 trial.trial_sha256 == _trial_token(trial),
                 "v96 CUDA joint trial is stale or mismatched")
        validate_v94b_accepted_density_receipt_gpu_v96(
            trial.accepted_density, self.execution, parent, self.provider)
        _validate_hydraulic_trial(
            self.execution, parent, self.provider, trial.hydraulic_trial)
        expected_counts = CoupledDensInOwnerCountsGPUV96(
            1, 1, 1, 1, 1, 1, 1, 0, 0, 0)
        _require(trial.owner_counts == expected_counts and
                 trial.dt_s == trial.hydraulic_trial.dt_s and
                 trial.candidate.accepted_updates ==
                    parent.accepted_updates+1 and
                 trial.candidate.time_s == parent.time_s+trial.dt_s,
                 "v96 CUDA owner counts or clocks differ")
        self.active._validate_resident_control_trial(
            parent.active_state, trial.active_trial)
        v92_trial = trial.active_trial.base_v93_trial.base_v92_trial
        _require(np.array_equal(
                     v92_trial.candidate.q,
                     trial.hydraulic_trial.candidate_q) and
                 np.array_equal(
                     v92_trial.hydraulic_face_receipt.
                     accepted_water_flux_m2_s,
                     trial.hydraulic_trial.face_water_flux_m2_s),
                 "v96 CUDA nested v92 hydraulic provenance differs")
        if self.execution.open_boundary_execution is None:
            _require(trial.candidate.accepted_boundary_scalar_state is None,
                     "v96 CUDA closed candidate carries boundary scalars")
        else:
            expected_boundary = \
                make_accepted_boundary_scalar_state_from_v92_trial_v96(
                    _v92_execution(self.execution.active_execution),
                    self.execution.open_boundary_execution.
                        boundary_trace_execution,
                    v92_trial)
            _require(type(trial.candidate.accepted_boundary_scalar_state) is
                     AcceptedBoundaryScalarStateV96 and
                     trial.candidate.accepted_boundary_scalar_state.
                        state_sha256 == expected_boundary.state_sha256,
                     "v96 CUDA accepted boundary scalar image differs")
        validate_hles_producer_state_v90(
            trial.hles_trial.candidate, self.hles.execution)
        _require(trial.hles_trial.parent_state_sha256 ==
                 parent.hles_state.state_sha256 and
                 self.hles._open_trial == trial.hles_trial.trial_sha256,
                 "v96 CUDA HLES child is not the open transaction")
        expected_chain = _fingerprint({
            "previous": parent.accepted_chain_sha256,
            "accepted_density": trial.accepted_density.receipt_sha256,
            "hydraulic_trial": trial.hydraulic_trial.trial_sha256,
            "active_trial": trial.active_trial.trial_sha256,
            "hles_trial": trial.hles_trial.trial_sha256,
            "accepted_boundary_scalar_state": (
                None if trial.candidate.accepted_boundary_scalar_state is None
                else trial.candidate.accepted_boundary_scalar_state.
                    state_sha256),
            "dt_s": trial.dt_s,
        })
        rebuilt = _wrap_state(
            self.execution, active_state=trial.active_trial.candidate,
            hles_state=trial.hles_trial.candidate,
            predictor_rate_m2_s2=trial.hydraulic_trial.full_rate,
            have_predictor=True,
            hydraulic_boundary_ledger_m3=(
                parent.hydraulic_boundary_ledger_m3+
                trial.hydraulic_trial.water_boundary_ledger_m3),
            accepted_chain_sha256=expected_chain,
            accepted_boundary_scalar_state=
                trial.candidate.accepted_boundary_scalar_state)
        _require(trial.candidate.accepted_chain_sha256 == expected_chain and
                 trial.candidate.state_sha256 == rebuilt.state_sha256,
                 "v96 CUDA joint candidate chain differs")

    def commit(
        self, parent: CoupledDensInGPUStateV96,
        trial: CoupledDensInGPUTrialV96,
    ) -> CoupledDensInGPUStateV96:
        _require(parent is self._parent and trial is self._trial,
                 "v96 CUDA trial is not the resident open transaction")
        self._validate_trial(parent, trial)
        try:
            hles = self.hles.commit(parent.hles_state, trial.hles_trial)
            _require(hles is trial.hles_trial.candidate,
                     "v96 CUDA HLES commit substituted candidate")
            active = self.active.commit(parent.active_state,
                                        trial.active_trial)
            _require(active is trial.active_trial.candidate,
                     "v96 CUDA v94b commit substituted candidate")
            self._publish_model_state(trial.candidate)
            self.density.accept_outer(
                trial.hydraulic_trial, trial.candidate.state_sha256)
            validate_coupled_densin_gpu_state_v96(
                trial.candidate, self.execution, self.provider)
        except Exception:
            try:
                self._restore_transaction()
            finally:
                self._clear()
            raise
        self._accepted_state = trial.candidate
        self.last_accepted_receipt = {
            "scope_id": SCOPE_ID + ":accepted-transaction",
            "execution_fingerprint": self.execution.fingerprint,
            "trial_sha256": trial.trial_sha256,
            "accepted_state_sha256": trial.candidate.state_sha256,
            "actual_cuda_kernels": True,
            "automatic_cpu_fallback": False,
            "one_joint_owner": True,
        }
        self._clear()
        return trial.candidate

    def rollback(
        self, parent: CoupledDensInGPUStateV96,
        trial: CoupledDensInGPUTrialV96,
    ) -> CoupledDensInGPUStateV96:
        _require(parent is self._parent and trial is self._trial,
                 "v96 CUDA trial is not the resident open transaction")
        self._validate_trial(parent, trial)
        try:
            self.hles.rollback(parent.hles_state, trial.hles_trial)
            self.active.rollback(parent.active_state, trial.active_trial)
            self.density.discard_outer()
            self._restore_model_snapshot(self._snapshot)
            validate_coupled_densin_gpu_state_v96(
                parent, self.execution, self.provider)
        except Exception:
            try:
                self._restore_transaction()
            finally:
                self._clear()
            raise
        self._clear()
        return parent

    def cancel(self, parent, trial):
        return self.rollback(parent, trial)

    def checkpoint(
        self, state: CoupledDensInGPUStateV96,
    ) -> dict[str, object]:
        _require(self._parent is None and self._trial is None and
                 self.density.pending_trial is None,
                 "v96 CUDA checkpoint requires a committed point")
        validate_coupled_densin_gpu_state_v96(
            state, self.execution, self.provider)
        self._validate_runtime_identity()
        _require(self._accepted_state is not None and
                 state.state_sha256 == self._accepted_state.state_sha256 and
                 np.array_equal(self.cp.asnumpy(self.model.state["q"]),
                                state.q) and
                 np.array_equal(
                     self.cp.asnumpy(self.controller.predictor_rate),
                     state.predictor_rate_m2_s2) and
                 bool(self.controller.have_predictor) ==
                    state.have_predictor and
                 np.array_equal(
                     self.cp.asnumpy(self.model.ledger["water"]),
                     state.hydraulic_boundary_ledger_m3),
                 "v96 CUDA resident model differs from checkpoint state")
        body = {
            "schema": SCHEMA,
            "scope_id": CHECKPOINT_SCOPE_ID,
            "semantic_id": SEMANTIC_ID,
            "execution_fingerprint": self.execution.fingerprint,
            "runtime_receipt_sha256":
                self.runtime_receipt["receipt_sha256"],
            "active_v94b_checkpoint":
                self.active.checkpoint(state.active_state),
            "hles_checkpoint": self.hles.checkpoint(state.hles_state),
            "predictor_rate_m2_s2": state.predictor_rate_m2_s2.tolist(),
            "have_predictor": state.have_predictor,
            "hydraulic_boundary_ledger_m3":
                state.hydraulic_boundary_ledger_m3.tolist(),
            "bed_elevation_m": state.bed_elevation_m.tolist(),
            "free_surface_elevation_m":
                state.free_surface_elevation_m.tolist(),
            "geometry_water_volume_ledger_m3":
                state.geometry_water_volume_ledger_m3,
            "accepted_boundary_scalar_checkpoint": (
                None if state.accepted_boundary_scalar_state is None else
                accepted_boundary_scalar_checkpoint_v96(
                    state.accepted_boundary_scalar_state,
                    _v92_execution(self.execution.active_execution),
                    self.execution.open_boundary_execution.
                        boundary_trace_execution)),
            "accepted_chain_sha256": state.accepted_chain_sha256,
            "state_sha256": state.state_sha256,
        }
        return {**body, "fingerprint": _fingerprint(body)}

    def restore_checkpoint(
        self, document: Mapping[str, object],
    ) -> CoupledDensInGPUStateV96:
        _require(self._parent is None and self._trial is None and
                 self._accepted_state is not None,
                 "v96 CUDA restart cannot replace an open transaction")
        self._validate_runtime_identity()
        expected = {
            "schema", "scope_id", "semantic_id", "execution_fingerprint",
            "runtime_receipt_sha256", "active_v94b_checkpoint",
            "hles_checkpoint", "predictor_rate_m2_s2", "have_predictor",
            "hydraulic_boundary_ledger_m3", "bed_elevation_m",
            "free_surface_elevation_m",
            "geometry_water_volume_ledger_m3", "accepted_chain_sha256",
            "accepted_boundary_scalar_checkpoint",
            "state_sha256", "fingerprint",
        }
        _require(isinstance(document, Mapping) and set(document) == expected,
                 "v96 CUDA checkpoint inventory differs")
        body = {key: document[key] for key in document
                if key != "fingerprint"}
        _require(document["fingerprint"] == _fingerprint(body) and
                 document["schema"] == SCHEMA and
                 document["scope_id"] == CHECKPOINT_SCOPE_ID and
                 document["semantic_id"] == SEMANTIC_ID and
                 document["execution_fingerprint"] ==
                    self.execution.fingerprint and
                 document["runtime_receipt_sha256"] ==
                    self.runtime_receipt["receipt_sha256"],
                 "v96 CUDA checkpoint identity differs")
        old = self._accepted_state
        snapshot = self._take_snapshot(old)
        try:
            active = self.active.restore_checkpoint(
                document["active_v94b_checkpoint"])
            hles = self.hles.restore_checkpoint(document["hles_checkpoint"])
            if self.execution.open_boundary_execution is None:
                _require(document["accepted_boundary_scalar_checkpoint"] is
                         None,
                         "v96 CUDA closed checkpoint has boundary scalars")
                boundary_scalar_state = None
            else:
                boundary_scalar_state = \
                    load_accepted_boundary_scalar_checkpoint_v96(
                        document["accepted_boundary_scalar_checkpoint"],
                        _v92_execution(self.execution.active_execution),
                        self.execution.open_boundary_execution.
                            boundary_trace_execution)
            state = _wrap_state(
                self.execution, active_state=active, hles_state=hles,
                predictor_rate_m2_s2=document["predictor_rate_m2_s2"],
                have_predictor=document["have_predictor"],
                hydraulic_boundary_ledger_m3=
                    document["hydraulic_boundary_ledger_m3"],
                accepted_chain_sha256=document["accepted_chain_sha256"],
                q3_provider=self.provider,
                accepted_boundary_scalar_state=boundary_scalar_state)
            _require(np.array_equal(
                         state.bed_elevation_m,
                         _array(document["bed_elevation_m"],
                                (self.execution.cell_count,),
                                "checkpoint bed elevation")) and
                     np.array_equal(
                         state.free_surface_elevation_m,
                         _array(document["free_surface_elevation_m"],
                                (self.execution.cell_count,),
                                "checkpoint free surface")) and
                     document["geometry_water_volume_ledger_m3"] == 0.0 and
                     state.state_sha256 == document["state_sha256"],
                     "v96 CUDA checkpoint state digest differs")
            self._publish_model_state(state, initial=True)
            self.density.accepted_q.set(state.q)
            self.density._resident_parent_sha256 = state.state_sha256
            self.density.last_accepted_receipt = None
            self.active.last_accepted_receipt = None
            self.active.inner.last_accepted_receipt = None
            self.active.inner.base.last_accepted_receipt = None
            self.last_accepted_receipt = None
            self._accepted_state = state
            return state
        except Exception:
            errors = []
            try:
                self.active.restore_checkpoint(snapshot["active_checkpoint"])
                self.hles.restore_checkpoint(snapshot["hles_checkpoint"])
            except Exception as error:
                errors.append(error)
            try:
                self._restore_model_snapshot(snapshot)
            except Exception as error:
                errors.append(error)
            self.active.last_accepted_receipt = copy.deepcopy(
                snapshot["active_receipt"])
            self.active.inner.last_accepted_receipt = copy.deepcopy(
                snapshot["inner_receipt"])
            self.active.inner.base.last_accepted_receipt = copy.deepcopy(
                snapshot["base_receipt"])
            self.density.last_accepted_receipt = copy.deepcopy(
                snapshot["density_receipt"])
            self.last_accepted_receipt = copy.deepcopy(
                snapshot["outer_receipt"])
            self._accepted_state = old
            if errors:
                raise RuntimeError(
                    "v96 CUDA checkpoint failure restore failed") \
                    from errors[0]
            raise


def method_identity_v96_cuda() -> dict[str, object]:
    return {
        "scope_id": SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "backend": BACKEND,
        "integrator": "implicit_fwave_midpoint",
        "density_source": "genuine_accepted_v94b_state_receipt",
        "residual_order": "augmented_fwave_lagged_v91_v85_density",
        "joint_state": "v94b_q_hles_predictor_no_shadow_q",
        "actual_cuda_kernels": True,
        "automatic_cpu_fallback": False,
        "maximum_accepted_updates": MAX_ACCEPTED_UPDATES,
        "product_admission": False,
        "bed_geometry_policy": GEOMETRY_POLICY,
        "spatial_accepted_water_eos": True,
        "lagged_nonzero_v91_residual": True,
        "density_modes": ["water_eos", "water_plus_sediment"],
        "not_claimed": [
            "open_hydraulic_or_sediment_boundaries", "moving_wet_dry",
            "AMR", "cut_cells", "plugin_support",
            "long_trajectory_qualification",
        ],
    }


__all__ = [
    "BACKEND", "MAX_ACCEPTED_UPDATES", "SEMANTIC_ID",
    "CoupledDensInGPUExecutionV96", "CoupledDensInGPUStateV96",
    "CoupledDensInGPUTrialV96", "CoupledDensInGPUV96Error",
    "CoupledDensInOwnerCountsGPUV96", "HydraulicDensInTrialGPUV96",
    "ResidentCoupledDensInLifecycleV96", "ResidentV94bDensInHookV96",
    "V94bAcceptedDensityReceiptGPUV96", "execution_document_v96",
    "make_v94b_accepted_density_receipt_gpu_v96",
    "method_identity_v96_cuda", "resolve_coupled_densin_gpu_execution_v96",
    "resolve_coupled_densin_gpu_state_v96",
    "validate_coupled_densin_gpu_state_v96",
    "validate_v94b_accepted_density_receipt_gpu_v96",
]
