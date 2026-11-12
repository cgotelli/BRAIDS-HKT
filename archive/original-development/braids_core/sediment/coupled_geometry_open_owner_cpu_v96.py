"""Additive atomic v96 CPU owner for delegated physical open boundaries.

This owner composes the already-qualified accepted-bed/spatial-EOS v96 state,
the exact physical trace/flux adapter, native OpenMP hydraulics and v91, and
one native scalar/Q3/Exner/HLES transaction.  It leaves the closed v96 owner
and all of its identities unchanged.  The current manufactured boundary
provider qualifies transaction algebra only; production CPU mode support still
requires the cut-cell-owned pure trace/flux core described by the source audit.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Callable, Mapping

import numpy as np

from . import active_suspended_control_v94 as _v94
from .accepted_boundary_scalar_state_cpu_v96 import (
    AcceptedBoundaryScalarStateV96,
    accepted_boundary_scalar_checkpoint_v96,
    load_accepted_boundary_scalar_checkpoint_v96,
    make_accepted_boundary_scalar_state_from_v92_trial_v96,
    make_boundary_density_from_accepted_scalar_state_v96,
    validate_accepted_boundary_scalar_state_v96,
)
from .active_suspended_preconverged_cpu_v96 import (
    begin_active_suspended_preconverged_cpu_trial_v96,
)
from .coupled_densin_cpu_v96 import (
    _cancelled, _closed_scalar_inputs, _face_normal_velocity,
    _grid_execution, _residual_norm, _restore_provider_after_failure,
    _v92_execution,
)
from . import coupled_geometry_owner_cpu_v96 as _base
from .coupled_geometry_owner_cpu_v96 import (
    CoupledGeometryCPUExecutionV96,
    CoupledGeometryCPUStateV96,
    V94bAcceptedDensityReceiptV96,
    AcceptedBedGeometryReceiptV96,
    execution_document_coupled_geometry_owner_v96,
    make_accepted_bed_geometry_receipt_v96,
    make_v94b_accepted_geometry_density_receipt_v96,
    validate_coupled_geometry_cpu_state_v96,
)
from .coupled_open_boundary_native_cpu_v96 import (
    AcceptedBoundaryDensityReceiptV96,
    CoupledOpenBoundaryNativeCPUExecutionV96,
    NativeCoupledOpenBoundaryRateResultV96,
    NativeCoupledOpenBoundaryRuntimeV96,
    execution_document_coupled_open_boundary_native_v96,
    validate_accepted_boundary_density_receipt_v96,
)
from .densin_fwave_v85 import (
    delft_kmax1_face_acceleration_v85,
    project_face_acceleration_v85,
)
from .hles_producer_v90 import (
    CurrentClosureHandoffV90, HlesProducerTrialV90,
    begin_hles_cpu_trial_v90, commit_hles_trial_v90,
    make_current_closure_handoff_v90, make_hles_input_receipt_v90,
    rollback_hles_trial_v90,
)
from .hydraulic_boundary_trace_cpu_v96 import (
    HydraulicBoundaryTraceProviderV96,
    HydraulicBoundaryTraceReceiptV96,
    execution_document_hydraulic_boundary_trace_v96,
    validate_hydraulic_boundary_trace_receipt_v96,
)
from .scalar_diffusion_native_cpu_v96 import NativeScalarDiffusionRuntimeV96
from .scalar_fct_native_cpu_v96 import NativeScalarFCTRuntimeV96
from .shared_turbulence_native_cpu_v96 import (
    NativeMomentumRateV96, NativeSharedTurbulenceRuntimeV96,
)
from .transported_fluid_native_scalar_cpu_v96 import (
    NativeCombinedScalarEvidenceV96,
    _evidence_token as _native_scalar_evidence_token,
)


SCHEMA = 1
SCOPE_ID = "braids-q4-coupled-geometry-open-owner-cpu-v96-v1"
SEMANTIC_ID = SCOPE_ID + ":delegated-open-flux-atomic-owner"
CHECKPOINT_SCOPE_ID = SCOPE_ID + ":checkpoint"
BACKEND = "native_cpu_openmp"
MAX_ACCEPTED_UPDATES = 4
_SHA256 = __import__("re").compile(r"[0-9a-f]{64}\Z")


class CoupledGeometryOpenOwnerV96Error(ValueError):
    """The additive open v96 execution, state, or trial is invalid."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError, OverflowError) as error:
        raise CoupledGeometryOpenOwnerV96Error(
            "v96 open-owner value is not finite canonical JSON") from error


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _digest(*values: np.ndarray) -> str:
    result = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(value)
        result.update(array.dtype.str.encode("ascii"))
        result.update(_canonical(list(array.shape)))
        result.update(array.tobytes(order="C"))
    return result.hexdigest()


def _array(value: object, shape: tuple[int, ...], name: str, *,
           positive: bool = False, nonnegative: bool = False) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != shape or raw.dtype.kind not in "fiub":
        raise CoupledGeometryOpenOwnerV96Error(
            f"v96 open-owner {name} must have shape {shape}")
    result = np.ascontiguousarray(raw, dtype=np.float64)
    if (not np.isfinite(result).all() or
            (positive and np.any(result <= 0.0)) or
            (nonnegative and np.any(result < 0.0))):
        raise CoupledGeometryOpenOwnerV96Error(
            f"v96 open-owner {name} contains invalid values")
    result.setflags(write=False)
    return result


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise CoupledGeometryOpenOwnerV96Error(
            f"v96 open-owner {name} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True)
class CoupledGeometryOpenCPUExecutionV96:
    base_execution: CoupledGeometryCPUExecutionV96
    open_hydraulic_execution: CoupledOpenBoundaryNativeCPUExecutionV96
    semantic_id: str
    fingerprint: str

    @property
    def active_execution(self):
        return self.base_execution.active_execution

    @property
    def cell_count(self) -> int:
        return self.base_execution.cell_count

    @property
    def face_count(self) -> int:
        return self.base_execution.face_count

    @property
    def fraction_count(self) -> int:
        return self.base_execution.fraction_count


def _execution_payload(value: CoupledGeometryOpenCPUExecutionV96) -> dict:
    return {
        "schema": SCHEMA, "scope_id": SCOPE_ID,
        "semantic_id": value.semantic_id, "backend": BACKEND,
        "accepted_bed_scalar_execution":
            execution_document_coupled_geometry_owner_v96(
                value.base_execution),
        "authoritative_open_hydraulic_execution":
            execution_document_coupled_open_boundary_native_v96(
                value.open_hydraulic_execution),
        "nested_closed_boundary_identity_role":
            "topology-and-constitutive-compatibility-only",
        "boundary_trace_timing": "current-picard-evaluation",
        "boundary_scalar_density_timing":
            "accepted-parent-frozen-before-picard",
        "open_flux_owner_count": 1,
        "scalar_lifecycle_owner_count": 1,
        "exner_owner_count": 1,
        "accepted_update_limit": MAX_ACCEPTED_UPDATES,
        "production_admission": False,
    }


def resolve_coupled_geometry_open_cpu_execution_v96(
    base_execution: CoupledGeometryCPUExecutionV96,
    open_hydraulic_execution: CoupledOpenBoundaryNativeCPUExecutionV96,
) -> CoupledGeometryOpenCPUExecutionV96:
    execution_document_coupled_geometry_owner_v96(base_execution)
    execution_document_coupled_open_boundary_native_v96(
        open_hydraulic_execution)
    boundary = open_hydraulic_execution.boundary_execution
    grid = _grid_execution(base_execution.active_execution)
    if (open_hydraulic_execution.base_execution.fingerprint !=
            base_execution.hydraulic_execution.fingerprint or
            not np.array_equal(boundary.owner, grid.owner) or
            not np.array_equal(boundary.neighbour, grid.neighbour) or
            not np.array_equal(boundary.boundary_index,
                               grid.boundary_index) or
            not np.array_equal(boundary.face_distance_m,
                               grid.centre_distance_m)):
        raise CoupledGeometryOpenOwnerV96Error(
            "v96 open-owner topology or native base differs")
    placeholder = CoupledGeometryOpenCPUExecutionV96(
        base_execution, open_hydraulic_execution, SEMANTIC_ID, "")
    return CoupledGeometryOpenCPUExecutionV96(
        **{**placeholder.__dict__,
           "fingerprint": _fingerprint(_execution_payload(placeholder))})


def execution_document_coupled_geometry_open_owner_v96(
    value: CoupledGeometryOpenCPUExecutionV96,
) -> dict:
    if type(value) is not CoupledGeometryOpenCPUExecutionV96:
        raise CoupledGeometryOpenOwnerV96Error(
            "v96 open-owner execution type differs")
    payload = _execution_payload(value)
    if (value.semantic_id != SEMANTIC_ID or
            value.fingerprint != _fingerprint(payload)):
        raise CoupledGeometryOpenOwnerV96Error(
            "v96 open-owner execution fingerprint differs")
    return {**payload, "fingerprint": value.fingerprint}


@dataclass(frozen=True)
class CoupledGeometryOpenCPUStateV96:
    execution_fingerprint: str
    base_state: CoupledGeometryCPUStateV96
    accepted_boundary_scalar: AcceptedBoundaryScalarStateV96
    accepted_chain_sha256: str
    state_sha256: str

    @property
    def q(self) -> np.ndarray:
        return self.base_state.q

    @property
    def accepted_updates(self) -> int:
        return self.base_state.accepted_updates

    @property
    def time_s(self) -> float:
        return self.base_state.time_s


def _state_token(value: CoupledGeometryOpenCPUStateV96) -> str:
    return _fingerprint({
        "scope_id": SCOPE_ID + ":state",
        "execution_fingerprint": value.execution_fingerprint,
        "base_state_sha256": value.base_state.state_sha256,
        "accepted_boundary_scalar_state_sha256":
            value.accepted_boundary_scalar.state_sha256,
        "accepted_chain_sha256": value.accepted_chain_sha256,
        "shared_update": value.accepted_updates,
        "shared_time_s": value.time_s,
    })


def _wrap_state(
    execution: CoupledGeometryOpenCPUExecutionV96, *,
    base_state: CoupledGeometryCPUStateV96,
    accepted_boundary_scalar: AcceptedBoundaryScalarStateV96,
    accepted_chain_sha256: str,
    q3_provider: object | None = None,
) -> CoupledGeometryOpenCPUStateV96:
    execution_document_coupled_geometry_open_owner_v96(execution)
    validate_coupled_geometry_cpu_state_v96(
        base_state, execution.base_execution, q3_provider)
    v92 = _v92_execution(execution.active_execution)
    validate_accepted_boundary_scalar_state_v96(
        accepted_boundary_scalar, v92,
        execution.open_hydraulic_execution.boundary_execution,
        accepted_update=base_state.accepted_updates,
        time_s=base_state.time_s)
    placeholder = CoupledGeometryOpenCPUStateV96(
        execution.fingerprint, base_state, accepted_boundary_scalar,
        _sha(accepted_chain_sha256, "accepted chain"), "")
    return CoupledGeometryOpenCPUStateV96(
        **{**placeholder.__dict__,
           "state_sha256": _state_token(placeholder)})


def resolve_coupled_geometry_open_cpu_state_v96(
    execution: CoupledGeometryOpenCPUExecutionV96, *,
    base_state: CoupledGeometryCPUStateV96,
    accepted_boundary_scalar: AcceptedBoundaryScalarStateV96,
    q3_provider: object,
) -> CoupledGeometryOpenCPUStateV96:
    if base_state.accepted_updates != 0:
        raise CoupledGeometryOpenOwnerV96Error(
            "v96 open-owner initial state must start at update zero")
    chain = _fingerprint({
        "scope_id": SCOPE_ID,
        "execution_fingerprint": execution.fingerprint,
        "base_state_sha256": base_state.state_sha256,
        "initial_boundary_scalar_state_sha256":
            accepted_boundary_scalar.state_sha256,
    })
    return _wrap_state(
        execution, base_state=base_state,
        accepted_boundary_scalar=accepted_boundary_scalar,
        accepted_chain_sha256=chain, q3_provider=q3_provider)


def validate_coupled_geometry_open_cpu_state_v96(
    value: CoupledGeometryOpenCPUStateV96,
    execution: CoupledGeometryOpenCPUExecutionV96,
    q3_provider: object | None = None,
) -> None:
    if (type(value) is not CoupledGeometryOpenCPUStateV96 or
            value.execution_fingerprint != execution.fingerprint or
            value.state_sha256 != _state_token(value)):
        raise CoupledGeometryOpenOwnerV96Error(
            "v96 open-owner state is stale or cross-execution")
    rebuilt = _wrap_state(
        execution, base_state=value.base_state,
        accepted_boundary_scalar=value.accepted_boundary_scalar,
        accepted_chain_sha256=value.accepted_chain_sha256,
        q3_provider=q3_provider)
    if rebuilt.state_sha256 != value.state_sha256:
        raise CoupledGeometryOpenOwnerV96Error(
            "v96 open-owner rebuilt state differs")


def _open_scalar_inputs(
    execution: CoupledGeometryOpenCPUExecutionV96,
    parent: CoupledGeometryOpenCPUStateV96,
    endpoint: np.ndarray,
    final_trace: HydraulicBoundaryTraceReceiptV96,
    shear_velocity_m_s: object,
    prescribed_boundary_concentration_kg_m3: object,
    prescribed_boundary_fluid: object,
) -> dict[str, object]:
    inputs = _closed_scalar_inputs(
        execution.base_execution, parent.base_state, endpoint,
        shear_velocity_m_s)
    v92 = _v92_execution(execution.active_execution)
    grid = _grid_execution(execution.active_execution)
    ng, nb = execution.fraction_count, v92.boundary_count
    inputs["prescribed_boundary_concentration_kg_m3"] = _array(
        prescribed_boundary_concentration_kg_m3, (ng, nb),
        "prescribed sediment boundary", nonnegative=True)
    inputs["prescribed_boundary_fluid"] = _array(
        prescribed_boundary_fluid, (2, nb),
        "prescribed fluid boundary", nonnegative=True)
    physical = np.asarray(
        final_trace.physical_boundary_flux_owned_mask, dtype=bool)
    internal = grid.neighbour >= 0
    active = np.ascontiguousarray(internal | physical)
    area = np.array(inputs["accepted_face_exchange_area_m2"], copy=True)
    area[physical] = (
        final_trace.trace_q[0, physical]*grid.face_length_m[physical])
    inputs["accepted_face_exchange_area_m2"] = np.ascontiguousarray(area)
    inputs["transport_face_active"] = active
    return inputs


@dataclass(frozen=True)
class CoupledGeometryOpenOwnerCountsV96:
    native_hydraulic_owner_count: int
    physical_open_flux_owner_count: int
    boundary_trace_owner_call_count: int
    boundary_scalar_density_owner_count: int
    turbulence_cap_owner_count: int
    turbulence_momentum_owner_count: int
    scalar_lifecycle_owner_count: int
    authoritative_exner_owner_count: int
    hles_clock_owner_count: int
    duplicate_owner_count: int


@dataclass(frozen=True)
class CoupledGeometryOpenCPUTrialV96:
    execution_fingerprint: str
    parent_state_sha256: str
    accepted_density: V94bAcceptedDensityReceiptV96
    accepted_bed_geometry: AcceptedBedGeometryReceiptV96
    accepted_boundary_density: AcceptedBoundaryDensityReceiptV96
    current_handoff: CurrentClosureHandoffV90
    turbulence_preparation_sha256: str
    turbulence_runtime_sha256: str
    native_scalar_evidence: NativeCombinedScalarEvidenceV96
    dt_s: float
    midpoint_q: np.ndarray
    midpoint_q_by_evaluation: tuple[np.ndarray, ...]
    boundary_trace_by_evaluation: tuple[HydraulicBoundaryTraceReceiptV96, ...]
    base_fwave_rate: np.ndarray
    density_momentum_rate: np.ndarray
    turbulence_momentum_rate_m2_s2: np.ndarray
    turbulence_face_flux_m4_s2: np.ndarray
    full_rate: np.ndarray
    final_residual: np.ndarray
    face_normal_acceleration_m_s2: np.ndarray
    cell_acceleration_m_s2: np.ndarray
    face_water_flux_m2_s: np.ndarray
    nonlinear_iterations: int
    nonlinear_residual_linf: float
    water_volume_residual_m3: float
    active_trial: object
    hles_trial: HlesProducerTrialV90
    owner_counts: CoupledGeometryOpenOwnerCountsV96
    candidate: CoupledGeometryOpenCPUStateV96
    trial_sha256: str


def _trial_token(value: CoupledGeometryOpenCPUTrialV96) -> str:
    return _fingerprint({
        "scope_id": SCOPE_ID + ":trial",
        "execution_fingerprint": value.execution_fingerprint,
        "parent_state_sha256": value.parent_state_sha256,
        "accepted_density": value.accepted_density.receipt_sha256,
        "accepted_bed": value.accepted_bed_geometry.receipt_sha256,
        "accepted_boundary_density":
            value.accepted_boundary_density.receipt_sha256,
        "handoff": value.current_handoff.receipt_sha256,
        "turbulence_preparation": value.turbulence_preparation_sha256,
        "turbulence_runtime": value.turbulence_runtime_sha256,
        "native_scalar": value.native_scalar_evidence.evidence_sha256,
        "dt_s": value.dt_s,
        "midpoints_sha256": [_digest(item)
                              for item in value.midpoint_q_by_evaluation],
        "boundary_trace_receipts": [item.receipt_sha256 for item in
                                    value.boundary_trace_by_evaluation],
        "arrays_sha256": _digest(
            value.midpoint_q, value.base_fwave_rate,
            value.density_momentum_rate,
            value.turbulence_momentum_rate_m2_s2,
            value.turbulence_face_flux_m4_s2, value.full_rate,
            value.final_residual, value.face_normal_acceleration_m_s2,
            value.cell_acceleration_m_s2, value.face_water_flux_m2_s),
        "nonlinear_iterations": value.nonlinear_iterations,
        "nonlinear_residual_linf": value.nonlinear_residual_linf,
        "water_volume_residual_m3": value.water_volume_residual_m3,
        "active_trial": value.active_trial.trial_sha256,
        "hles_trial": value.hles_trial.trial_sha256,
        "owner_counts": list(value.owner_counts.__dict__.values()),
        "candidate": value.candidate.state_sha256,
    })


def begin_coupled_geometry_open_cpu_trial_v96(
    execution: CoupledGeometryOpenCPUExecutionV96,
    parent: CoupledGeometryOpenCPUStateV96,
    q3_provider: object, *,
    open_native_runtime: NativeCoupledOpenBoundaryRuntimeV96,
    boundary_trace_provider: HydraulicBoundaryTraceProviderV96,
    turbulence_runtime: NativeSharedTurbulenceRuntimeV96,
    scalar_fct_runtime: NativeScalarFCTRuntimeV96,
    scalar_diffusion_runtime: NativeScalarDiffusionRuntimeV96,
    shear_velocity_m_s: object,
    closure_chezy_mhalf_s: object,
    prescribed_boundary_concentration_kg_m3: object,
    prescribed_boundary_fluid: object,
    q3_inputs: Mapping[str, object],
    dt_s: float,
    cancelled: bool | Callable[[], bool] = False,
) -> CoupledGeometryOpenCPUTrialV96:
    execution_document_coupled_geometry_open_owner_v96(execution)
    validate_coupled_geometry_open_cpu_state_v96(
        parent, execution, q3_provider)
    hydraulic = execution.open_hydraulic_execution
    base_execution = execution.base_execution
    if (type(open_native_runtime) is not
            NativeCoupledOpenBoundaryRuntimeV96 or
            open_native_runtime.receipt != hydraulic.native_build_receipt or
            type(boundary_trace_provider) is not
            HydraulicBoundaryTraceProviderV96 or
            boundary_trace_provider.execution.fingerprint !=
                hydraulic.boundary_execution.fingerprint):
        raise CoupledGeometryOpenOwnerV96Error(
            "v96 open-owner hydraulic runtime/provider identity differs")
    if (type(turbulence_runtime) is not NativeSharedTurbulenceRuntimeV96 or
            turbulence_runtime.receipt.get("receipt_sha256") !=
                base_execution.turbulence_native_receipt_sha256 or
            type(scalar_fct_runtime) is not NativeScalarFCTRuntimeV96 or
            scalar_fct_runtime.runtime_sha256 !=
                base_execution.scalar_fct_native_receipt_sha256 or
            type(scalar_diffusion_runtime) is not
                NativeScalarDiffusionRuntimeV96 or
            scalar_diffusion_runtime.receipt.get("receipt_sha256") !=
                base_execution.scalar_diffusion_native_receipt_sha256):
        raise CoupledGeometryOpenOwnerV96Error(
            "v96 open-owner turbulence/scalar runtime identity differs")
    if parent.accepted_updates >= MAX_ACCEPTED_UPDATES:
        raise CoupledGeometryOpenOwnerV96Error(
            "v96 open-owner accepted update limit was reached")
    if _cancelled(cancelled):
        raise InterruptedError("v96 open-owner cancelled before hydraulics")
    dt = float(dt_s)
    if (not math.isfinite(dt) or dt <= 0.0 or
            dt > hydraulic.base_execution.maximum_dt_s):
        raise CoupledGeometryOpenOwnerV96Error(
            "v96 open-owner timestep is invalid")
    density = make_v94b_accepted_geometry_density_receipt_v96(
        base_execution, parent.base_state, q3_provider)
    bed = make_accepted_bed_geometry_receipt_v96(
        base_execution, parent.base_state, q3_provider)
    boundary_density = make_boundary_density_from_accepted_scalar_state_v96(
        parent.accepted_boundary_scalar,
        _v92_execution(execution.active_execution),
        hydraulic.boundary_execution,
        parent_state_sha256=parent.state_sha256)
    request = _fingerprint({
        "scope_id": SCOPE_ID + ":hydraulic-request",
        "execution_fingerprint": execution.fingerprint,
        "parent_state_sha256": parent.state_sha256,
        "accepted_density": density.receipt_sha256,
        "accepted_bed": bed.receipt_sha256,
        "accepted_boundary_density": boundary_density.receipt_sha256,
        "dt_s": dt,
    })
    hles_execution = _v92_execution(
        execution.active_execution).base_execution.hles_execution
    handoff = make_current_closure_handoff_v90(
        hles_execution, parent.base_state.hles_state,
        hydraulic_trial_request_sha256=request)
    v91_execution = _v92_execution(execution.active_execution).base_execution
    prepared = turbulence_runtime.prepare(
        v91_execution, parent.base_state.hles_state, handoff, dt_s=dt,
        native_threads=hydraulic.base_execution.native_threads)
    midpoint = np.array(parent.q, copy=True, order="C")
    midpoint_history: list[np.ndarray] = []
    trace_history: list[HydraulicBoundaryTraceReceiptV96] = []
    result: NativeCoupledOpenBoundaryRateResultV96 | None = None
    turbulence: NativeMomentumRateV96 | None = None
    full_rate = residual = None
    residual_norm = math.inf
    nonlinear_iterations = 0
    for evaluation in range(
            1, hydraulic.base_execution.max_nonlinear_iterations+2):
        if _cancelled(cancelled):
            raise InterruptedError(
                "v96 open-owner cancelled during midpoint")
        trace = boundary_trace_provider.resolve(
            midpoint, bed.bed_elevation_m,
            hydraulic_request_sha256=request,
            parent_state_sha256=parent.state_sha256,
            evaluation_index=evaluation,
            evaluation_time_s=parent.time_s+0.5*dt,
            evaluation_dt_s=dt)
        result = open_native_runtime.evaluate(
            hydraulic, midpoint, bed_elevation_m=bed.bed_elevation_m,
            accepted_water_density_kg_m3=density.water_density_kg_m3,
            accepted_mixture_density_kg_m3=density.mixture_density_kg_m3,
            boundary_density=boundary_density, boundary_trace=trace)
        turbulence = turbulence_runtime.evaluate(
            prepared, midpoint,
            cancelled=(cancelled if callable(cancelled)
                       else lambda: bool(cancelled)))
        full_rate = np.array(result.full_rate, copy=True, order="C")
        full_rate += turbulence.momentum_rate_m2_s2
        full_rate = np.ascontiguousarray(full_rate)
        residual = np.ascontiguousarray(
            midpoint-parent.q-0.5*dt*full_rate)
        residual_norm = _residual_norm(
            residual, midpoint,
            hydraulic.base_execution.base_execution.gravity_m_s2)
        frozen_midpoint = np.array(midpoint, copy=True, order="C")
        frozen_midpoint.setflags(write=False)
        midpoint_history.append(frozen_midpoint)
        trace_history.append(trace)
        if residual_norm <= hydraulic.base_execution.nonlinear_tolerance:
            break
        if evaluation > hydraulic.base_execution.max_nonlinear_iterations:
            raise CoupledGeometryOpenOwnerV96Error(
                "v96 open-owner midpoint failed to converge")
        midpoint = np.ascontiguousarray(midpoint-residual)
        if (not np.isfinite(midpoint).all() or
                np.any(midpoint[0] <=
                       hydraulic.base_execution.base_execution.
                       dry_tolerance_m)):
            raise CoupledGeometryOpenOwnerV96Error(
                "v96 open-owner midpoint left fully-wet scope")
        nonlinear_iterations += 1
    if result is None or turbulence is None or full_rate is None or residual is None:
        raise CoupledGeometryOpenOwnerV96Error(
            "v96 open-owner hydraulic solve produced no result")
    endpoint = np.ascontiguousarray(parent.q+dt*full_rate)
    if np.any(endpoint[0] <=
              hydraulic.base_execution.base_execution.dry_tolerance_m):
        raise CoupledGeometryOpenOwnerV96Error(
            "v96 open-owner endpoint left fully-wet scope")
    topology = hydraulic.base_execution.base_execution
    boundary = topology.neighbour < 0
    water_residual = float(np.sum(
        topology.cell_area_m2*(endpoint[0]-parent.q[0]))+
        dt*np.sum(topology.face_length_m[boundary]*
                  result.face_water_flux_m2_s[boundary]))
    water_scale = max(1.0, float(np.sum(topology.cell_area_m2*parent.q[0])))
    if abs(water_residual) > 256*np.finfo(float).eps*water_scale:
        raise CoupledGeometryOpenOwnerV96Error(
            "v96 open-owner water budget differs")
    scalar = _open_scalar_inputs(
        execution, parent, endpoint, trace_history[-1],
        shear_velocity_m_s, prescribed_boundary_concentration_kg_m3,
        prescribed_boundary_fluid)
    provider_checkpoint = q3_provider.checkpoint()
    active_trial = None
    try:
        active_trial, native_scalar = \
            begin_active_suspended_preconverged_cpu_trial_v96(
                execution.active_execution,
                parent.base_state.active_state,
                parent.base_state.hles_state, handoff, q3_provider,
                cap_receipt=prepared.cap_receipt,
                midpoint_q=midpoint, accepted_q=endpoint,
                native_momentum_rate_m2_s2=
                    turbulence.momentum_rate_m2_s2,
                native_momentum_face_flux_m4_s2=
                    turbulence.face_flux_m4_s2,
                residual_evaluation_count=len(midpoint_history),
                accepted_water_flux_m2_s=result.face_water_flux_m2_s,
                dt_s=dt, v92_scalar_inputs=scalar,
                q3_inputs=dict(q3_inputs),
                scalar_fct_runtime=scalar_fct_runtime,
                scalar_diffusion_runtime=scalar_diffusion_runtime,
                native_scalar_threads=
                    hydraulic.base_execution.native_threads,
                return_native_scalar_evidence=True,
                cancelled=(cancelled if callable(cancelled)
                           else lambda: bool(cancelled)))
        if native_scalar is None:
            raise CoupledGeometryOpenOwnerV96Error(
                "v96 open-owner native scalar evidence is absent")
        v92_trial = active_trial.base_v93_trial.base_v92_trial
        if (not np.array_equal(v92_trial.candidate.q, endpoint) or
                not np.array_equal(
                    v92_trial.hydraulic_face_receipt.
                    accepted_water_flux_m2_s,
                    result.face_water_flux_m2_s)):
            raise CoupledGeometryOpenOwnerV96Error(
                "v96 open-owner scalar phase changed hydraulics")
        face_velocity = _face_normal_velocity(
            base_execution, endpoint, result.face_water_flux_m2_s)
        chezy = _array(
            closure_chezy_mhalf_s, (execution.cell_count,),
            "closure Chezy", positive=True)
        closure_inputs = {
            "accepted_face_normal_velocity_m_s": face_velocity,
            "active_face": np.ones(execution.face_count, dtype=np.int32),
            "closure_depth_m": endpoint[0],
            "closure_chezy_mhalf_s": chezy,
            "active_cell": np.ones(execution.cell_count, dtype=np.int32),
        }
        step = v92_trial.hydraulic_receipt.base_step_receipt
        hles_input = make_hles_input_receipt_v90(
            hles_execution, parent.base_state.hles_state,
            accepted_step_receipt=step, current_handoff=handoff,
            closure_interval_s=dt, **closure_inputs)
        hles_trial = begin_hles_cpu_trial_v90(
            hles_execution, parent.base_state.hles_state,
            accepted_step_receipt=step,
            hles_input_receipt=hles_input, current_handoff=handoff,
            cancelled=False, **closure_inputs)
        boundary_candidate = \
            make_accepted_boundary_scalar_state_from_v92_trial_v96(
                _v92_execution(execution.active_execution),
                hydraulic.boundary_execution, v92_trial)
        base_chain = _fingerprint({
            "previous": parent.base_state.accepted_chain_sha256,
            "outer_open_request": request,
            "active_trial": active_trial.trial_sha256,
            "hles_trial": hles_trial.trial_sha256,
            "dt_s": dt,
        })
        base_candidate = _base._wrap_state(
            base_execution, active_state=active_trial.candidate,
            hles_state=hles_trial.candidate,
            accepted_chain_sha256=base_chain)
        outer_chain = _fingerprint({
            "previous": parent.accepted_chain_sha256,
            "request": request,
            "midpoints": [_digest(item) for item in midpoint_history],
            "boundary_traces": [item.receipt_sha256
                                for item in trace_history],
            "native_scalar": native_scalar.evidence_sha256,
            "boundary_candidate": boundary_candidate.state_sha256,
            "base_candidate": base_candidate.state_sha256,
        })
        candidate = _wrap_state(
            execution, base_state=base_candidate,
            accepted_boundary_scalar=boundary_candidate,
            accepted_chain_sha256=outer_chain)
        counts = CoupledGeometryOpenOwnerCountsV96(
            1, 1, len(trace_history), 1, 1, len(trace_history),
            1, 1, 1, 0)
        placeholder = CoupledGeometryOpenCPUTrialV96(
            execution.fingerprint, parent.state_sha256, density, bed,
            boundary_density, handoff, prepared.preparation_sha256,
            prepared.native_runtime_sha256, native_scalar, dt,
            _array(midpoint, (3, execution.cell_count), "midpoint"),
            tuple(midpoint_history), tuple(trace_history),
            result.base_fwave_rate, result.density_momentum_rate,
            turbulence.momentum_rate_m2_s2,
            turbulence.face_flux_m4_s2, full_rate,
            _array(residual, (3, execution.cell_count), "residual"),
            result.face_normal_acceleration_m_s2,
            result.cell_acceleration_m_s2,
            result.face_water_flux_m2_s, nonlinear_iterations,
            residual_norm, water_residual, active_trial, hles_trial,
            counts, candidate, "")
        trial = CoupledGeometryOpenCPUTrialV96(
            **{**placeholder.__dict__,
               "trial_sha256": _trial_token(placeholder)})
        _validate_trial(execution, parent, q3_provider, trial)
        return trial
    except Exception as original_error:
        try:
            _restore_provider_after_failure(
                active_trial, q3_provider, provider_checkpoint)
        except Exception as cleanup_error:
            raise RuntimeError(
                "v96 open-owner begin and provider restore both failed") \
                from cleanup_error
        raise original_error


def _validate_trial(
    execution: CoupledGeometryOpenCPUExecutionV96,
    parent: CoupledGeometryOpenCPUStateV96,
    q3_provider: object,
    trial: CoupledGeometryOpenCPUTrialV96,
) -> None:
    validate_coupled_geometry_open_cpu_state_v96(
        parent, execution, q3_provider)
    if (type(trial) is not CoupledGeometryOpenCPUTrialV96 or
            trial.execution_fingerprint != execution.fingerprint or
            trial.parent_state_sha256 != parent.state_sha256 or
            trial.trial_sha256 != _trial_token(trial)):
        raise CoupledGeometryOpenOwnerV96Error(
            "v96 open-owner trial is stale or cross-parent")
    hydraulic = execution.open_hydraulic_execution
    boundary_execution = hydraulic.boundary_execution
    _base.validate_v94b_accepted_geometry_density_receipt_v96(
        trial.accepted_density, execution.base_execution,
        parent.base_state, q3_provider)
    _base.validate_accepted_bed_geometry_receipt_v96(
        trial.accepted_bed_geometry, execution.base_execution,
        parent.base_state, q3_provider)
    validate_accepted_boundary_density_receipt_v96(
        trial.accepted_boundary_density, boundary_execution)
    expected_boundary_density = \
        make_boundary_density_from_accepted_scalar_state_v96(
            parent.accepted_boundary_scalar,
            _v92_execution(execution.active_execution),
            boundary_execution, parent_state_sha256=parent.state_sha256)
    if (trial.accepted_boundary_density.receipt_sha256 !=
            expected_boundary_density.receipt_sha256):
        raise CoupledGeometryOpenOwnerV96Error(
            "v96 open-owner accepted boundary density differs")
    expected_request = _fingerprint({
        "scope_id": SCOPE_ID + ":hydraulic-request",
        "execution_fingerprint": execution.fingerprint,
        "parent_state_sha256": parent.state_sha256,
        "accepted_density": trial.accepted_density.receipt_sha256,
        "accepted_bed": trial.accepted_bed_geometry.receipt_sha256,
        "accepted_boundary_density":
            trial.accepted_boundary_density.receipt_sha256,
        "dt_s": trial.dt_s,
    })
    if (len(trial.midpoint_q_by_evaluation) < 1 or
            len(trial.midpoint_q_by_evaluation) !=
                len(trial.boundary_trace_by_evaluation) or
            trial.nonlinear_iterations !=
                len(trial.midpoint_q_by_evaluation)-1):
        raise CoupledGeometryOpenOwnerV96Error(
            "v96 open-owner evaluation inventory differs")
    for index, (midpoint, trace) in enumerate(zip(
            trial.midpoint_q_by_evaluation,
            trial.boundary_trace_by_evaluation), start=1):
        validate_hydraulic_boundary_trace_receipt_v96(
            trace, boundary_execution, midpoint_q=midpoint,
            bed_elevation_m=trial.accepted_bed_geometry.bed_elevation_m,
            hydraulic_request_sha256=expected_request,
            parent_state_sha256=parent.state_sha256,
            evaluation_index=index,
            evaluation_time_s=parent.time_s+0.5*trial.dt_s)
    if not np.array_equal(
            trial.midpoint_q, trial.midpoint_q_by_evaluation[-1]):
        raise CoupledGeometryOpenOwnerV96Error(
            "v96 open-owner final midpoint differs")
    midpoint = trial.midpoint_q
    expected_full = np.array(trial.base_fwave_rate, copy=True)
    expected_full[1:] += trial.density_momentum_rate
    expected_full += trial.turbulence_momentum_rate_m2_s2
    residual = midpoint-parent.q-0.5*trial.dt_s*trial.full_rate
    endpoint = parent.q+trial.dt_s*trial.full_rate
    norm = _residual_norm(
        residual, midpoint,
        hydraulic.base_execution.base_execution.gravity_m_s2)
    topology = hydraulic.base_execution.base_execution
    boundary = topology.neighbour < 0
    water_residual = float(np.sum(
        topology.cell_area_m2*(endpoint[0]-parent.q[0]))+
        trial.dt_s*np.sum(topology.face_length_m[boundary]*
                          trial.face_water_flux_m2_s[boundary]))
    effective_density = (
        trial.accepted_density.mixture_density_kg_m3 if
        hydraulic.base_execution.density_mode ==
        _base.DENSITY_WATER_PLUS_SEDIMENT else
        trial.accepted_density.water_density_kg_m3)
    expected_face = delft_kmax1_face_acceleration_v85(
        midpoint[0], effective_density, topology.owner,
        topology.neighbour, topology.face_distance_m,
        gravity_m_s2=topology.gravity_m_s2,
        hydraulic_reference_density_kg_m3=
            topology.hydraulic_reference_density_kg_m3)
    final_trace = trial.boundary_trace_by_evaluation[-1]
    exterior = (trial.accepted_boundary_density.mixture_density_kg_m3 if
                hydraulic.base_execution.density_mode ==
                _base.DENSITY_WATER_PLUS_SEDIMENT else
                trial.accepted_boundary_density.water_density_kg_m3)
    densin = final_trace.exterior_densin_active_mask.astype(bool)
    expected_face[boundary] = 0.0
    owners = topology.owner[densin]
    expected_face[densin] = (
        -topology.gravity_m_s2*
        (exterior[densin]-effective_density[owners])*
        (final_trace.trace_q[0, densin]+midpoint[0, owners])/
        (4.0*topology.hydraulic_reference_density_kg_m3*
         topology.face_distance_m[densin]))
    if not hydraulic.ibaroc_open_boundary:
        expected_face[densin] = 0.0
    expected_cell = project_face_acceleration_v85(
        expected_face, topology.normal, topology.face_length_m,
        topology.cell_face_offset, topology.cell_faces,
        topology.cell_face_sign)
    expected_density = midpoint[0][None, :]*expected_cell
    v92_trial = trial.active_trial.base_v93_trial.base_v92_trial
    expected_boundary = make_accepted_boundary_scalar_state_from_v92_trial_v96(
        _v92_execution(execution.active_execution), boundary_execution,
        v92_trial)
    expected_counts = CoupledGeometryOpenOwnerCountsV96(
        1, 1, len(trial.midpoint_q_by_evaluation), 1, 1,
        len(trial.midpoint_q_by_evaluation), 1, 1, 1, 0)
    physical = final_trace.physical_boundary_flux_owned_mask.astype(bool)
    base_chain = _fingerprint({
        "previous": parent.base_state.accepted_chain_sha256,
        "outer_open_request": expected_request,
        "active_trial": trial.active_trial.trial_sha256,
        "hles_trial": trial.hles_trial.trial_sha256,
        "dt_s": trial.dt_s,
    })
    expected_base_candidate = _base._wrap_state(
        execution.base_execution,
        active_state=trial.active_trial.candidate,
        hles_state=trial.hles_trial.candidate,
        accepted_chain_sha256=base_chain)
    expected_outer_chain = _fingerprint({
        "previous": parent.accepted_chain_sha256,
        "request": expected_request,
        "midpoints": [_digest(item)
                      for item in trial.midpoint_q_by_evaluation],
        "boundary_traces": [item.receipt_sha256
                            for item in trial.boundary_trace_by_evaluation],
        "native_scalar": trial.native_scalar_evidence.evidence_sha256,
        "boundary_candidate": expected_boundary.state_sha256,
        "base_candidate": expected_base_candidate.state_sha256,
    })
    rebuilt_candidate = _wrap_state(
        execution, base_state=expected_base_candidate,
        accepted_boundary_scalar=expected_boundary,
        accepted_chain_sha256=expected_outer_chain)
    scalar_evidence = trial.native_scalar_evidence
    if (type(scalar_evidence) is not NativeCombinedScalarEvidenceV96 or
            scalar_evidence.fct_runtime_sha256 !=
                execution.base_execution.scalar_fct_native_receipt_sha256 or
            scalar_evidence.diffusion_runtime_sha256 !=
                execution.base_execution.
                scalar_diffusion_native_receipt_sha256 or
            scalar_evidence.suspended_trial_sha256 !=
                v92_trial.suspended_trial.trial_sha256 or
            scalar_evidence.fluid_result_sha256 !=
                v92_trial.fluid_transport.result_sha256 or
            scalar_evidence.evidence_sha256 !=
                _native_scalar_evidence_token(scalar_evidence)):
        raise CoupledGeometryOpenOwnerV96Error(
            "v96 open-owner native scalar evidence differs")
    if (trial.owner_counts != expected_counts or
            trial.dt_s <= 0.0 or
            trial.dt_s > hydraulic.base_execution.maximum_dt_s or
            trial.candidate.accepted_updates != parent.accepted_updates+1 or
            trial.candidate.time_s != parent.time_s+trial.dt_s or
            not np.array_equal(expected_full, trial.full_rate) or
            not np.array_equal(residual, trial.final_residual) or
            not np.array_equal(endpoint, trial.candidate.q) or
            trial.nonlinear_residual_linf != norm or
            norm > hydraulic.base_execution.nonlinear_tolerance or
            trial.water_volume_residual_m3 != water_residual or
            abs(water_residual) > 256*np.finfo(float).eps*max(
                1.0, float(np.sum(topology.cell_area_m2*parent.q[0]))) or
            not np.allclose(expected_face,
                            trial.face_normal_acceleration_m_s2,
                            rtol=8*np.finfo(float).eps, atol=1e-18) or
            not np.allclose(expected_density,
                            trial.density_momentum_rate,
                            rtol=24*np.finfo(float).eps, atol=1e-16) or
            expected_boundary.state_sha256 !=
                trial.candidate.accepted_boundary_scalar.state_sha256 or
            not np.array_equal(
                final_trace.physical_open_face_flux[0, physical],
                trial.face_water_flux_m2_s[physical]) or
            not np.array_equal(v92_trial.candidate.q, endpoint) or
            not np.array_equal(
                v92_trial.hydraulic_face_receipt.accepted_water_flux_m2_s,
                trial.face_water_flux_m2_s) or
            trial.candidate.accepted_chain_sha256 != expected_outer_chain or
            trial.candidate.base_state.state_sha256 !=
                expected_base_candidate.state_sha256 or
            trial.candidate.state_sha256 != rebuilt_candidate.state_sha256):
        raise CoupledGeometryOpenOwnerV96Error(
            "v96 open-owner residual, density, boundary, or ledger differs")
    _v94._validate_trial(
        execution.active_execution, parent.base_state.active_state,
        parent.base_state.hles_state, trial.current_handoff,
        trial.active_trial)
    if rollback_hles_trial_v90(
            _v92_execution(execution.active_execution).base_execution.
            hles_execution, parent.base_state.hles_state,
            trial.hles_trial) is not parent.base_state.hles_state:
        raise CoupledGeometryOpenOwnerV96Error(
            "v96 open-owner HLES rollback differs")
    validate_coupled_geometry_open_cpu_state_v96(
        trial.candidate, execution)


def commit_coupled_geometry_open_cpu_trial_v96(
    execution: CoupledGeometryOpenCPUExecutionV96,
    parent: CoupledGeometryOpenCPUStateV96,
    q3_provider: object,
    trial: CoupledGeometryOpenCPUTrialV96,
) -> CoupledGeometryOpenCPUStateV96:
    checkpoint = trial.active_trial.base_v93_trial.q3_parent_checkpoint
    try:
        _validate_trial(execution, parent, q3_provider, trial)
        commit_hles_trial_v90(
            _v92_execution(execution.active_execution).base_execution.
            hles_execution, parent.base_state.hles_state,
            trial.hles_trial)
        _base.commit_active_suspended_control_trial_v94b(
            execution.active_execution, parent.base_state.active_state,
            parent.base_state.hles_state, trial.current_handoff,
            trial.active_trial)
        return trial.candidate
    except Exception as original_error:
        try:
            _restore_provider_after_failure(
                getattr(trial, "active_trial", None), q3_provider, checkpoint)
        except Exception as cleanup_error:
            raise RuntimeError(
                "v96 open-owner commit and restore both failed") \
                from cleanup_error
        raise original_error


def rollback_coupled_geometry_open_cpu_trial_v96(
    execution: CoupledGeometryOpenCPUExecutionV96,
    parent: CoupledGeometryOpenCPUStateV96,
    q3_provider: object,
    trial: CoupledGeometryOpenCPUTrialV96,
) -> CoupledGeometryOpenCPUStateV96:
    _validate_trial(execution, parent, q3_provider, trial)
    active = _base.rollback_active_suspended_control_trial_v94b(
        execution.active_execution, parent.base_state.active_state,
        parent.base_state.hles_state, trial.current_handoff,
        trial.active_trial)
    hles = rollback_hles_trial_v90(
        _v92_execution(execution.active_execution).base_execution.
        hles_execution, parent.base_state.hles_state, trial.hles_trial)
    if (active is not parent.base_state.active_state or
            hles is not parent.base_state.hles_state):
        raise CoupledGeometryOpenOwnerV96Error(
            "v96 open-owner rollback changed parent")
    return parent


cancel_coupled_geometry_open_cpu_trial_v96 = \
    rollback_coupled_geometry_open_cpu_trial_v96


def coupled_geometry_open_cpu_checkpoint_v96(
    execution: CoupledGeometryOpenCPUExecutionV96,
    state: CoupledGeometryOpenCPUStateV96,
    q3_provider: object,
) -> dict[str, object]:
    validate_coupled_geometry_open_cpu_state_v96(
        state, execution, q3_provider)
    v92 = _v92_execution(execution.active_execution)
    body = {
        "schema": SCHEMA, "scope_id": CHECKPOINT_SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "execution_fingerprint": execution.fingerprint,
        "base_checkpoint": _base.coupled_geometry_cpu_checkpoint_v96(
            execution.base_execution, state.base_state, q3_provider),
        "accepted_boundary_scalar_checkpoint":
            accepted_boundary_scalar_checkpoint_v96(
                state.accepted_boundary_scalar, v92,
                execution.open_hydraulic_execution.boundary_execution),
        "accepted_chain_sha256": state.accepted_chain_sha256,
        "state_sha256": state.state_sha256,
    }
    return {**body, "fingerprint": _fingerprint(body)}


def load_coupled_geometry_open_cpu_checkpoint_v96(
    execution: CoupledGeometryOpenCPUExecutionV96,
    document: Mapping[str, object],
    q3_provider: object,
) -> CoupledGeometryOpenCPUStateV96:
    keys = {
        "schema", "scope_id", "semantic_id", "execution_fingerprint",
        "base_checkpoint", "accepted_boundary_scalar_checkpoint",
        "accepted_chain_sha256", "state_sha256", "fingerprint",
    }
    if not isinstance(document, Mapping) or set(document) != keys:
        raise CoupledGeometryOpenOwnerV96Error(
            "v96 open-owner checkpoint inventory differs")
    body = {key: document[key] for key in document if key != "fingerprint"}
    if (document["schema"] != SCHEMA or
            document["scope_id"] != CHECKPOINT_SCOPE_ID or
            document["semantic_id"] != SEMANTIC_ID or
            document["execution_fingerprint"] != execution.fingerprint or
            document["fingerprint"] != _fingerprint(body)):
        raise CoupledGeometryOpenOwnerV96Error(
            "v96 open-owner checkpoint identity differs")
    checkpoint = q3_provider.checkpoint()
    try:
        base_state = _base.load_coupled_geometry_cpu_checkpoint_v96(
            execution.base_execution, document["base_checkpoint"],
            q3_provider)
        boundary_state = load_accepted_boundary_scalar_checkpoint_v96(
            document["accepted_boundary_scalar_checkpoint"],
            _v92_execution(execution.active_execution),
            execution.open_hydraulic_execution.boundary_execution)
        state = _wrap_state(
            execution, base_state=base_state,
            accepted_boundary_scalar=boundary_state,
            accepted_chain_sha256=document["accepted_chain_sha256"],
            q3_provider=q3_provider)
        if state.state_sha256 != document["state_sha256"]:
            raise CoupledGeometryOpenOwnerV96Error(
                "v96 open-owner checkpoint state differs")
        return state
    except Exception as original_error:
        try:
            q3_provider.restore_checkpoint(checkpoint)
        except Exception as cleanup_error:
            raise RuntimeError(
                "v96 open-owner checkpoint and restore both failed") \
                from cleanup_error
        raise original_error


def method_identity_coupled_geometry_open_owner_v96() -> dict[str, object]:
    return {
        "scope_id": SCOPE_ID, "semantic_id": SEMANTIC_ID,
        "backend": BACKEND,
        "closed_owner_changed": False,
        "open_flux": "delegated-same-owner-corrected-physical-flux",
        "open_trace_is_riemann_right_state": False,
        "open_density": "parent-accepted-scalars-frozen-once",
        "exterior_densin_mask_separate_from_physical_flux_mask": True,
        "bed_geometry": "accepted-parent-current-candidate-next",
        "scalar_q3_exner": "single-preconverged-owner",
        "accepted_update_limit": MAX_ACCEPTED_UPDATES,
        "product_admission": False,
        "not_claimed": [
            "manufactured-provider-physical-mode-qualification",
            "cutcell-pure-boundary-core", "wet-dry", "AMR",
            "plugin-support", "long-trajectory-qualification"],
    }


__all__ = [
    "BACKEND", "MAX_ACCEPTED_UPDATES", "SEMANTIC_ID",
    "CoupledGeometryOpenCPUExecutionV96",
    "CoupledGeometryOpenCPUStateV96",
    "CoupledGeometryOpenCPUTrialV96",
    "CoupledGeometryOpenOwnerCountsV96",
    "CoupledGeometryOpenOwnerV96Error",
    "begin_coupled_geometry_open_cpu_trial_v96",
    "cancel_coupled_geometry_open_cpu_trial_v96",
    "commit_coupled_geometry_open_cpu_trial_v96",
    "coupled_geometry_open_cpu_checkpoint_v96",
    "execution_document_coupled_geometry_open_owner_v96",
    "load_coupled_geometry_open_cpu_checkpoint_v96",
    "method_identity_coupled_geometry_open_owner_v96",
    "resolve_coupled_geometry_open_cpu_execution_v96",
    "resolve_coupled_geometry_open_cpu_state_v96",
    "rollback_coupled_geometry_open_cpu_trial_v96",
    "validate_coupled_geometry_open_cpu_state_v96",
]
