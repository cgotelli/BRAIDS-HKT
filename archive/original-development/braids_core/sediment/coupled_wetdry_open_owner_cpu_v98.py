"""Atomic CPU v98 wet/dry owner with delegated physical boundaries.

This module is additive.  The frozen v97 closed-grid transaction remains the
authoritative scalar/Q3/Exner/HLES owner; v98 supplies a physical-boundary
hydraulic trial and persists the accepted boundary scalar image used by the
next hydraulic solve.  The cut-cell trace/flux owner is never replaced by a
ghost-state Riemann surrogate.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Callable, Mapping

import numpy as np

from . import coupled_wetdry_owner_cpu_v97 as _base
from .accepted_boundary_scalar_state_cpu_v96 import (
    AcceptedBoundaryScalarStateV96,
    accepted_boundary_scalar_checkpoint_v96,
    load_accepted_boundary_scalar_checkpoint_v96,
    make_accepted_boundary_scalar_state_from_v92_trial_v96,
    make_boundary_density_from_accepted_scalar_state_v96,
    validate_accepted_boundary_scalar_state_v96,
)
from .active_suspended_wetdry_cpu_v97 import (
    begin_active_suspended_wetdry_cpu_trial_v97,
)
from .coupled_densin_cpu_v96 import (
    _grid_execution,
    _restore_provider_after_failure,
    _v92_execution,
    _v92_state,
)
from .coupled_open_boundary_native_cpu_v96 import (
    AcceptedBoundaryDensityReceiptV96,
)
from .cutcell_boundary_trace_runtime_v98 import FROZEN_PARENT_SHA256
from .density_active_projection_v97 import (
    NativeDensityActiveProjectionRuntimeV97,
)
from .factcr_native_cpu_v97 import NativeFactCrRuntimeV97
from .hles_native_cpu_v97 import NativeHlesRuntimeV97
from .hles_producer_v90 import (
    commit_hles_trial_v90,
    make_current_closure_handoff_v90,
    make_hles_input_receipt_v90,
    rollback_hles_trial_v90,
)
from .hydraulic_boundary_trace_cpu_v96 import (
    OWNER_CUTCELL,
    HydraulicBoundaryTraceExecutionV96,
    HydraulicBoundaryTraceProviderV96,
    execution_document_hydraulic_boundary_trace_v96,
)
from .q3_face_inventory_native_cpu_v97 import (
    NativeQ3FaceInventoryRuntimeV97,
)
from .physical_flux_gather_native_cpu_v98 import (
    NativePhysicalFluxGatherRuntimeV98,
)
from .q3_wetdry_adapter_cpu_v97 import (
    CPUQ3V83ActiveWetDryExnerAdapterV97,
)
from .scalar_boundary_native_cpu_v97 import NativeScalarBoundaryRuntimeV97
from .scalar_diffusion_native_cpu_v96 import NativeScalarDiffusionRuntimeV96
from .scalar_fct_native_cpu_v96 import NativeScalarFCTRuntimeV96
from .settling_exchange_native_cpu_v97 import (
    NativeSettlingExchangeRuntimeV97,
)
from .shared_turbulence_native_cpu_v97 import (
    NativeSharedTurbulenceRuntimeV97,
)
from .transported_fluid_wetdry_cpu_v97 import (
    make_wetdry_hydraulic_evidence_v97,
)
from .wetdry_fwave_native_cpu_v97 import NativeWetDryFwaveRuntimeV97
from .wetdry_open_boundary_cpu_v98 import (
    WetDryOpenBoundaryRateRuntimeV98,
    WetDryOpenBoundaryTrialV98,
    solve_wetdry_midpoint_with_physical_boundary_v98,
    validate_wetdry_open_boundary_trial_v98,
)


SCHEMA = 1
SCOPE_ID = "braids-q4-coupled-moving-wetdry-open-owner-cpu-v98-v1"
SEMANTIC_ID = SCOPE_ID + ":cutcell-physical-boundary-atomic-owner"
CHECKPOINT_SCOPE_ID = SCOPE_ID + ":checkpoint"
BACKEND = "native_cpu_openmp_plus_cutcell_physical_boundary"


class CoupledWetDryOpenOwnerV98Error(ValueError):
    """The v98 connected open-boundary transaction is invalid."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"),
            allow_nan=False).encode("ascii")
    except (TypeError, ValueError, OverflowError) as error:
        raise CoupledWetDryOpenOwnerV98Error(
            "v98 open owner value is not canonical finite JSON") from error


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _digest(*values: object) -> str:
    result = hashlib.sha256()
    for raw in values:
        value = np.ascontiguousarray(raw)
        result.update(value.dtype.str.encode("ascii"))
        result.update(_canonical(list(value.shape)))
        result.update(value.tobytes(order="C"))
    return result.hexdigest()


def _array(value: object, shape: tuple[int, ...], name: str, *,
           nonnegative: bool = False) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != shape or raw.dtype.kind not in "fiub":
        raise CoupledWetDryOpenOwnerV98Error(
            f"v98 open owner {name} must have shape {shape}")
    result = np.array(raw, dtype=np.float64, order="C", copy=True)
    if (not np.isfinite(result).all() or
            (nonnegative and np.any(result < 0.0))):
        raise CoupledWetDryOpenOwnerV98Error(
            f"v98 open owner {name} is outside its finite domain")
    result.setflags(write=False)
    return result


def _cancelled(value: bool | Callable[[], bool]) -> bool:
    if type(value) is bool:
        return value
    if not callable(value):
        raise TypeError("cancelled must be Boolean or callable")
    return bool(value())


@dataclass(frozen=True)
class CoupledWetDryOpenCPUExecutionV98:
    base_execution: _base.CoupledWetDryCPUExecutionV97
    boundary_execution: HydraulicBoundaryTraceExecutionV96
    physical_gather_runtime_sha256: str
    admitted_boundary_modes: tuple[int, ...]
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


def _execution_payload(value: CoupledWetDryOpenCPUExecutionV98) -> dict:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "semantic_id": value.semantic_id,
        "backend": BACKEND,
        "base_v97_execution":
            _base.execution_document_coupled_wetdry_v97(
                value.base_execution),
        "boundary_execution":
            execution_document_hydraulic_boundary_trace_v96(
                value.boundary_execution),
        "physical_gather_runtime_sha256":
            value.physical_gather_runtime_sha256,
        "admitted_boundary_modes": list(value.admitted_boundary_modes),
        "hydraulic_order": (
            "cutcell-physical-flux-replaces-reflected-face-before-one-"
            "donor-limiter-and-one-density-projection"),
        "boundary_scalar_timing":
            "accepted-parent-image-frozen-before-picard-next-image-on-accept",
        "scalar_q3_exner_hles_owner": "unchanged-v97-single-transaction",
        "physical_boundary_surrogate_count": 0,
        "product_admission": False,
    }


def resolve_coupled_wetdry_open_cpu_execution_v98(
    base_execution: _base.CoupledWetDryCPUExecutionV97,
    boundary_execution: HydraulicBoundaryTraceExecutionV96,
    *, physical_gather_runtime: NativePhysicalFluxGatherRuntimeV98,
) -> CoupledWetDryOpenCPUExecutionV98:
    _base.execution_document_coupled_wetdry_v97(base_execution)
    document = execution_document_hydraulic_boundary_trace_v96(
        boundary_execution)
    grid = _grid_execution(base_execution.active_execution)
    topology = (
        boundary_execution.cell_count == base_execution.cell_count and
        boundary_execution.face_count == base_execution.face_count and
        np.array_equal(boundary_execution.owner, grid.owner) and
        np.array_equal(boundary_execution.neighbour, grid.neighbour) and
        np.array_equal(
            boundary_execution.face_distance_m, grid.centre_distance_m))
    physical = boundary_execution.open_mask
    modes = tuple(sorted(int(item) for item in np.unique(
        boundary_execution.boundary_mode[physical])))
    if (not topology or not modes or
            any(item not in range(1, 10) for item in modes) or
            boundary_execution.hydraulic_owner_kind != OWNER_CUTCELL or
            boundary_execution.hydraulic_owner_source_sha256 !=
                FROZEN_PARENT_SHA256 or
            document["physical_boundary_qualification"] is not True or
            type(physical_gather_runtime) is not
                NativePhysicalFluxGatherRuntimeV98):
        raise CoupledWetDryOpenOwnerV98Error(
            "v98 open owner topology or physical cutcell identity differs")
    placeholder = CoupledWetDryOpenCPUExecutionV98(
        base_execution, boundary_execution,
        physical_gather_runtime.receipt["receipt_sha256"],
        modes, SEMANTIC_ID, "")
    return CoupledWetDryOpenCPUExecutionV98(
        **{**placeholder.__dict__,
           "fingerprint": _hash(_execution_payload(placeholder))})


def execution_document_coupled_wetdry_open_v98(
    value: CoupledWetDryOpenCPUExecutionV98,
) -> dict[str, object]:
    if type(value) is not CoupledWetDryOpenCPUExecutionV98:
        raise CoupledWetDryOpenOwnerV98Error(
            "v98 open owner execution type differs")
    payload = _execution_payload(value)
    if (value.semantic_id != SEMANTIC_ID or
            value.fingerprint != _hash(payload)):
        raise CoupledWetDryOpenOwnerV98Error(
            "v98 open owner execution fingerprint differs")
    return {**payload, "fingerprint": value.fingerprint}


@dataclass(frozen=True)
class CoupledWetDryOpenCPUStateV98:
    execution_fingerprint: str
    base_state: _base.CoupledWetDryCPUStateV97
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


def _state_token(value: CoupledWetDryOpenCPUStateV98) -> str:
    return _hash({
        "scope_id": SCOPE_ID + ":state",
        "execution": value.execution_fingerprint,
        "base_state": value.base_state.state_sha256,
        "boundary_scalar": value.accepted_boundary_scalar.state_sha256,
        "accepted_chain": value.accepted_chain_sha256,
        "updates": value.accepted_updates,
        "time_s": value.time_s,
    })


def _wrap_state(
    execution: CoupledWetDryOpenCPUExecutionV98, *,
    base_state: _base.CoupledWetDryCPUStateV97,
    accepted_boundary_scalar: AcceptedBoundaryScalarStateV96,
    accepted_chain_sha256: str,
    q3_provider: object | None = None,
) -> CoupledWetDryOpenCPUStateV98:
    execution_document_coupled_wetdry_open_v98(execution)
    _base.validate_coupled_wetdry_cpu_state_v97(
        base_state, execution.base_execution, q3_provider)
    validate_accepted_boundary_scalar_state_v96(
        accepted_boundary_scalar,
        _v92_execution(execution.active_execution),
        execution.boundary_execution,
        accepted_update=base_state.accepted_updates,
        time_s=base_state.time_s)
    placeholder = CoupledWetDryOpenCPUStateV98(
        execution.fingerprint, base_state, accepted_boundary_scalar,
        accepted_chain_sha256, "")
    return CoupledWetDryOpenCPUStateV98(
        **{**placeholder.__dict__,
           "state_sha256": _state_token(placeholder)})


def resolve_coupled_wetdry_open_cpu_state_v98(
    execution: CoupledWetDryOpenCPUExecutionV98, *,
    base_state: _base.CoupledWetDryCPUStateV97,
    accepted_boundary_scalar: AcceptedBoundaryScalarStateV96,
    q3_provider: object,
) -> CoupledWetDryOpenCPUStateV98:
    if base_state.accepted_updates != 0:
        raise CoupledWetDryOpenOwnerV98Error(
            "v98 open owner fresh state must start at zero")
    chain = _hash({
        "scope_id": SCOPE_ID,
        "execution": execution.fingerprint,
        "base": base_state.state_sha256,
        "boundary": accepted_boundary_scalar.state_sha256,
    })
    return _wrap_state(
        execution, base_state=base_state,
        accepted_boundary_scalar=accepted_boundary_scalar,
        accepted_chain_sha256=chain, q3_provider=q3_provider)


def validate_coupled_wetdry_open_cpu_state_v98(
    value: CoupledWetDryOpenCPUStateV98,
    execution: CoupledWetDryOpenCPUExecutionV98,
    q3_provider: object | None = None,
) -> None:
    if (type(value) is not CoupledWetDryOpenCPUStateV98 or
            value.execution_fingerprint != execution.fingerprint or
            value.state_sha256 != _state_token(value)):
        raise CoupledWetDryOpenOwnerV98Error(
            "v98 open owner state is stale or cross-execution")
    rebuilt = _wrap_state(
        execution, base_state=value.base_state,
        accepted_boundary_scalar=value.accepted_boundary_scalar,
        accepted_chain_sha256=value.accepted_chain_sha256,
        q3_provider=q3_provider)
    if rebuilt.state_sha256 != value.state_sha256:
        raise CoupledWetDryOpenOwnerV98Error(
            "v98 open owner rebuilt state differs")


def _open_scalar_inputs(
    execution: CoupledWetDryOpenCPUExecutionV98,
    parent: CoupledWetDryOpenCPUStateV98,
    endpoint_q: np.ndarray,
    hydraulic: WetDryOpenBoundaryTrialV98,
    shear_velocity_m_s: object,
    prescribed_boundary_concentration_kg_m3: object,
    prescribed_boundary_fluid: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    fluid_left, fluid_right, _, scalar = _base._closed_scalar_inputs(
        execution.base_execution, parent.base_state,
        endpoint_q, shear_velocity_m_s)
    v92 = _v92_execution(execution.active_execution)
    ng, nb = execution.fraction_count, v92.boundary_count
    prescribed_c = _array(
        prescribed_boundary_concentration_kg_m3, (ng, nb),
        "prescribed sediment boundary", nonnegative=True)
    prescribed_f = _array(
        prescribed_boundary_fluid, (2, nb),
        "prescribed fluid boundary")
    trace = hydraulic.accepted_boundary_trace
    physical = trace.physical_boundary_flux_owned_mask.astype(bool)
    grid = _grid_execution(execution.active_execution)
    internal = grid.neighbour >= 0
    wet = endpoint_q[0] > grid.dry_tolerance_m
    active = internal & wet[grid.owner] & wet[np.maximum(grid.neighbour, 0)]
    active |= physical & wet[grid.owner]
    area = np.array(scalar["accepted_face_exchange_area_m2"], copy=True)
    area[physical] = (
        trace.trace_q[0, physical] * grid.face_length_m[physical])
    scalar["prescribed_boundary_concentration_kg_m3"] = prescribed_c
    scalar["accepted_face_exchange_area_m2"] = np.ascontiguousarray(area)
    scalar["transport_face_active"] = np.ascontiguousarray(active)
    return fluid_left, fluid_right, prescribed_f, scalar


@dataclass(frozen=True)
class CoupledWetDryOpenOwnerCountsV98:
    connected_wetdry_hydraulic_owner_count: int = 1
    physical_cutcell_flux_owner_count: int = 1
    physical_boundary_scalar_memory_owner_count: int = 1
    boundary_density_owner_count: int = 1
    donor_limiter_owner_count: int = 1
    density_projection_owner_count: int = 1
    internal_v91_owner_count: int = 1
    scalar_lifecycle_owner_count: int = 1
    q3_exner_owner_count: int = 1
    bed_geometry_owner_count: int = 1
    hles_clock_owner_count: int = 1
    physical_boundary_surrogate_count: int = 0
    duplicate_owner_count: int = 0


@dataclass(frozen=True)
class CoupledWetDryOpenCPUTrialV98:
    execution_fingerprint: str
    parent_state_sha256: str
    hydraulic_request_sha256: str
    accepted_boundary_density: AcceptedBoundaryDensityReceiptV96
    open_hydraulic_trial: WetDryOpenBoundaryTrialV98
    base_trial: _base.CoupledWetDryCPUTrialV97
    boundary_scalar_candidate: AcceptedBoundaryScalarStateV96
    owner_counts: CoupledWetDryOpenOwnerCountsV98
    candidate: CoupledWetDryOpenCPUStateV98
    trial_sha256: str


def _trial_token(value: CoupledWetDryOpenCPUTrialV98) -> str:
    return _hash({
        "scope_id": SCOPE_ID + ":trial",
        "execution": value.execution_fingerprint,
        "parent": value.parent_state_sha256,
        "request": value.hydraulic_request_sha256,
        "boundary_density": value.accepted_boundary_density.receipt_sha256,
        "hydraulic": value.open_hydraulic_trial.trial_sha256,
        "base_trial": value.base_trial.trial_sha256,
        "boundary_candidate": value.boundary_scalar_candidate.state_sha256,
        "owners": list(value.owner_counts.__dict__.values()),
        "candidate": value.candidate.state_sha256,
    })


def begin_coupled_wetdry_open_cpu_trial_v98(
    execution: CoupledWetDryOpenCPUExecutionV98,
    parent: CoupledWetDryOpenCPUStateV98,
    q3_provider: object, *,
    boundary_trace_provider: HydraulicBoundaryTraceProviderV96,
    physical_gather_runtime: NativePhysicalFluxGatherRuntimeV98,
    native_runtime: NativeWetDryFwaveRuntimeV97,
    density_projection_runtime: NativeDensityActiveProjectionRuntimeV97,
    turbulence_runtime: NativeSharedTurbulenceRuntimeV97,
    scalar_fct_runtime: NativeScalarFCTRuntimeV96,
    scalar_diffusion_runtime: NativeScalarDiffusionRuntimeV96,
    scalar_boundary_runtime: NativeScalarBoundaryRuntimeV97,
    settling_exchange_runtime: NativeSettlingExchangeRuntimeV97,
    hles_runtime: NativeHlesRuntimeV97,
    shear_velocity_m_s: object,
    closure_chezy_mhalf_s: object,
    prescribed_boundary_concentration_kg_m3: object,
    prescribed_boundary_fluid: object,
    q3_inputs: Mapping[str, object],
    dt_s: float,
    cancelled: bool | Callable[[], bool] = False,
) -> CoupledWetDryOpenCPUTrialV98:
    execution_document_coupled_wetdry_open_v98(execution)
    validate_coupled_wetdry_open_cpu_state_v98(
        parent, execution, q3_provider)
    if (type(boundary_trace_provider) is not
            HydraulicBoundaryTraceProviderV96 or
            boundary_trace_provider.execution.fingerprint !=
                execution.boundary_execution.fingerprint):
        raise CoupledWetDryOpenOwnerV98Error(
            "v98 open owner boundary provider differs")
    if (type(physical_gather_runtime) is not
            NativePhysicalFluxGatherRuntimeV98 or
            physical_gather_runtime.receipt["receipt_sha256"] !=
                execution.physical_gather_runtime_sha256):
        raise CoupledWetDryOpenOwnerV98Error(
            "v98 open owner physical gather runtime differs")
    if (type(q3_provider) is not
            CPUQ3V83ActiveWetDryExnerAdapterV97 or
            q3_provider.native_factcr_runtime_sha256 !=
                execution.base_execution.factcr_runtime_sha256 or
            q3_provider.native_q3_face_runtime_sha256 !=
                execution.base_execution.q3_face_runtime_sha256):
        raise CoupledWetDryOpenOwnerV98Error(
            "v98 open owner Q3 provider differs")
    _base._validate_runtimes(
        execution.base_execution,
        native_runtime=native_runtime,
        density_projection_runtime=density_projection_runtime,
        turbulence_runtime=turbulence_runtime,
        scalar_fct_runtime=scalar_fct_runtime,
        scalar_diffusion_runtime=scalar_diffusion_runtime,
        scalar_boundary_runtime=scalar_boundary_runtime,
        settling_exchange_runtime=settling_exchange_runtime,
        hles_runtime=hles_runtime)
    if parent.accepted_updates >= _base.MAX_ACCEPTED_UPDATES:
        raise CoupledWetDryOpenOwnerV98Error(
            "v98 open owner accepted update limit was reached")
    if not isinstance(q3_inputs, Mapping):
        raise CoupledWetDryOpenOwnerV98Error(
            "v98 open owner Q3 inputs must be a mapping")
    if _cancelled(cancelled):
        raise InterruptedError("v98 open owner cancelled before hydraulics")
    dt = float(dt_s)
    if (not math.isfinite(dt) or dt <= 0.0 or
            dt > execution.base_execution.hydraulic_execution.
                base_execution.maximum_dt_s):
        raise CoupledWetDryOpenOwnerV98Error(
            "v98 open owner timestep differs")

    water, mixture = _base._accepted_density(
        execution.base_execution, parent.base_state)
    boundary_density = make_boundary_density_from_accepted_scalar_state_v96(
        parent.accepted_boundary_scalar,
        _v92_execution(execution.active_execution),
        execution.boundary_execution,
        parent_state_sha256=parent.state_sha256)
    request = _hash({
        "scope_id": SCOPE_ID + ":hydraulic-request",
        "execution": execution.fingerprint,
        "parent": parent.state_sha256,
        "dt_s": dt,
        "bed": _digest(parent.base_state.bed_elevation_m),
        "density": _digest(water, mixture),
        "boundary_density": boundary_density.receipt_sha256,
    })
    hles_execution = _v92_execution(
        execution.active_execution).base_execution.hles_execution
    handoff = make_current_closure_handoff_v90(
        hles_execution, parent.base_state.hles_state,
        hydraulic_trial_request_sha256=request)
    v91_execution = _v92_execution(execution.active_execution).base_execution
    prepared = turbulence_runtime.prepare(
        v91_execution, parent.base_state.hles_state, handoff,
        dt_s=dt,
        native_threads=execution.base_execution.hydraulic_execution.
            native_threads)
    rate_runtime = WetDryOpenBoundaryRateRuntimeV98(
        execution.base_execution.hydraulic_execution,
        native_runtime,
        boundary_trace_provider=boundary_trace_provider,
        boundary_density=boundary_density,
        hydraulic_request_sha256=request,
        parent_state_sha256=parent.state_sha256,
        evaluation_time_s=parent.time_s + 0.5 * dt,
        dt_s=dt,
        turbulence_runtime=turbulence_runtime,
        turbulence_prepared=prepared,
        physical_gather_runtime=physical_gather_runtime)
    hydraulic_open = solve_wetdry_midpoint_with_physical_boundary_v98(
        execution.base_execution.hydraulic_execution,
        parent.q,
        bed_elevation_m=parent.base_state.bed_elevation_m,
        accepted_water_density_kg_m3=water,
        accepted_mixture_density_kg_m3=mixture,
        dt_s=dt,
        rate_runtime=rate_runtime,
        density_projection_runtime=density_projection_runtime,
        cancelled=cancelled)
    hydraulic = hydraulic_open.wetdry_trial
    evidence = make_wetdry_hydraulic_evidence_v97(
        v91_execution, _v92_state(parent.base_state.active_state),
        prepared.cap_receipt,
        wetdry_execution=execution.base_execution.hydraulic_execution,
        wetdry_trial=hydraulic,
        accepted_bed_elevation_m=parent.base_state.bed_elevation_m,
        accepted_water_density_kg_m3=water,
        accepted_mixture_density_kg_m3=mixture)
    fluid_left, fluid_right, prescribed_fluid, scalar = _open_scalar_inputs(
        execution, parent, hydraulic.endpoint_q, hydraulic_open,
        shear_velocity_m_s, prescribed_boundary_concentration_kg_m3,
        prescribed_boundary_fluid)

    provider_checkpoint = q3_provider.checkpoint()
    active_trial = None
    try:
        if _cancelled(cancelled):
            raise InterruptedError(
                "v98 open owner cancelled before scalar work")
        active_trial, scalar_evidence, factcr_evidence = \
            begin_active_suspended_wetdry_cpu_trial_v97(
                execution.active_execution,
                parent.base_state.active_state,
                parent.base_state.hles_state,
                handoff, q3_provider,
                hydraulic_evidence=evidence,
                reconstructed_left_fluid=fluid_left,
                reconstructed_right_fluid=fluid_right,
                prescribed_boundary_fluid=prescribed_fluid,
                v92_scalar_inputs=scalar,
                q3_inputs=dict(q3_inputs),
                scalar_fct_runtime=scalar_fct_runtime,
                scalar_diffusion_runtime=scalar_diffusion_runtime,
                scalar_boundary_runtime=scalar_boundary_runtime,
                settling_exchange_runtime=settling_exchange_runtime,
                native_scalar_threads=
                    execution.base_execution.hydraulic_execution.
                    native_threads,
                return_native_scalar_evidence=True,
                cancelled=(cancelled if callable(cancelled)
                           else lambda: bool(cancelled)))
        q3_face_evidence = q3_provider.native_q3_face_evidence_v97
        v92_trial = active_trial.base_v93_trial.base_v92_trial
        if (not np.array_equal(v92_trial.candidate.q,
                               hydraulic.endpoint_q) or
                not np.array_equal(
                    v92_trial.hydraulic_face_receipt.
                    accepted_water_flux_m2_s,
                    hydraulic.accepted_limited_face_water_flux_m2_s)):
            raise CoupledWetDryOpenOwnerV98Error(
                "v98 open owner scalar child changed hydraulics")
        hles_arrays = _base._hles_inputs(
            execution.base_execution, hydraulic.endpoint_q,
            hydraulic.accepted_limited_face_water_flux_m2_s,
            closure_chezy_mhalf_s)
        step = v92_trial.hydraulic_receipt.base_step_receipt
        hles_input = make_hles_input_receipt_v90(
            hles_execution, parent.base_state.hles_state,
            accepted_step_receipt=step, current_handoff=handoff,
            closure_interval_s=dt,
            accepted_face_normal_velocity_m_s=hles_arrays[0],
            active_face=hles_arrays[1], closure_depth_m=hles_arrays[2],
            closure_chezy_mhalf_s=hles_arrays[3],
            active_cell=hles_arrays[4])
        hles_trial, hles_evidence = hles_runtime.begin_trial(
            hles_execution, parent.base_state.hles_state,
            accepted_step_receipt=step,
            hles_input_receipt=hles_input,
            current_handoff=handoff,
            accepted_face_normal_velocity_m_s=hles_arrays[0],
            active_face=hles_arrays[1], closure_depth_m=hles_arrays[2],
            closure_chezy_mhalf_s=hles_arrays[3],
            active_cell=hles_arrays[4],
            native_threads=execution.base_execution.hydraulic_execution.
                native_threads,
            cancelled=cancelled)
        base_chain = _base._hash({
            "previous": parent.base_state.accepted_chain_sha256,
            "hydraulic": evidence.evidence_sha256,
            "scalar": scalar_evidence.evidence_sha256,
            "factcr": _base._hash(factcr_evidence.__dict__),
            "q3_face": q3_face_evidence.output_sha256,
            "active": active_trial.trial_sha256,
            "native_hles": hles_evidence.result_sha256,
            "hles": hles_trial.trial_sha256})
        base_candidate = _base._wrap_state(
            execution.base_execution,
            active_state=active_trial.candidate,
            hles_state=hles_trial.candidate,
            accepted_chain_sha256=base_chain)
        base_placeholder = _base.CoupledWetDryCPUTrialV97(
            execution.base_execution.fingerprint,
            parent.base_state.state_sha256,
            handoff, evidence, scalar_evidence, factcr_evidence,
            q3_face_evidence, active_trial, hles_trial, hles_evidence,
            hles_arrays, _base.CoupledWetDryOwnerCountsV97(),
            base_candidate, _base.BACKEND, "")
        base_trial = _base.CoupledWetDryCPUTrialV97(
            **{**base_placeholder.__dict__,
               "trial_sha256": _base._trial_token(base_placeholder)})
        _base._validate_trial(
            execution.base_execution, parent.base_state,
            q3_provider, base_trial)
        boundary_candidate = \
            make_accepted_boundary_scalar_state_from_v92_trial_v96(
                _v92_execution(execution.active_execution),
                execution.boundary_execution, v92_trial)
        chain = _hash({
            "previous": parent.accepted_chain_sha256,
            "request": request,
            "hydraulic": hydraulic_open.trial_sha256,
            "base_trial": base_trial.trial_sha256,
            "boundary_candidate": boundary_candidate.state_sha256,
        })
        candidate = _wrap_state(
            execution, base_state=base_candidate,
            accepted_boundary_scalar=boundary_candidate,
            accepted_chain_sha256=chain)
        placeholder = CoupledWetDryOpenCPUTrialV98(
            execution.fingerprint, parent.state_sha256, request,
            boundary_density, hydraulic_open, base_trial,
            boundary_candidate, CoupledWetDryOpenOwnerCountsV98(),
            candidate, "")
        trial = CoupledWetDryOpenCPUTrialV98(
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
                "v98 open owner begin and provider restore both failed") \
                from cleanup_error
        raise original_error


def _validate_trial(
    execution: CoupledWetDryOpenCPUExecutionV98,
    parent: CoupledWetDryOpenCPUStateV98,
    q3_provider: object,
    trial: CoupledWetDryOpenCPUTrialV98,
) -> None:
    validate_coupled_wetdry_open_cpu_state_v98(
        parent, execution, q3_provider)
    if (type(trial) is not CoupledWetDryOpenCPUTrialV98 or
            trial.execution_fingerprint != execution.fingerprint or
            trial.parent_state_sha256 != parent.state_sha256 or
            trial.owner_counts != CoupledWetDryOpenOwnerCountsV98() or
            trial.trial_sha256 != _trial_token(trial)):
        raise CoupledWetDryOpenOwnerV98Error(
            "v98 open owner trial is stale or cross-parent")
    hydraulic = trial.base_trial.hydraulic_evidence.wetdry_trial
    validate_wetdry_open_boundary_trial_v98(
        trial.open_hydraulic_trial,
        execution.base_execution.hydraulic_execution,
        boundary_execution=execution.boundary_execution,
        parent_q=parent.q,
        bed_elevation_m=parent.base_state.bed_elevation_m,
        accepted_water_density_kg_m3=
            trial.base_trial.hydraulic_evidence.
            accepted_water_density_kg_m3,
        accepted_mixture_density_kg_m3=
            trial.base_trial.hydraulic_evidence.
            accepted_mixture_density_kg_m3,
        boundary_density=trial.accepted_boundary_density,
        hydraulic_request_sha256=trial.hydraulic_request_sha256,
        parent_state_sha256=parent.state_sha256,
        evaluation_time_s=parent.time_s + 0.5 * hydraulic.dt_s,
        dt_s=hydraulic.dt_s)
    _base._validate_trial(
        execution.base_execution, parent.base_state,
        q3_provider, trial.base_trial)
    if any(
            receipt.physical_gather_runtime_sha256 !=
                execution.physical_gather_runtime_sha256
            for receipt in
            trial.open_hydraulic_trial.residual_audit.rate_receipts):
        raise CoupledWetDryOpenOwnerV98Error(
            "v98 open owner physical gather receipt differs")
    v92_trial = trial.base_trial.active_trial.base_v93_trial.base_v92_trial
    rebuilt_boundary = make_accepted_boundary_scalar_state_from_v92_trial_v96(
        _v92_execution(execution.active_execution),
        execution.boundary_execution, v92_trial)
    expected_chain = _hash({
        "previous": parent.accepted_chain_sha256,
        "request": trial.hydraulic_request_sha256,
        "hydraulic": trial.open_hydraulic_trial.trial_sha256,
        "base_trial": trial.base_trial.trial_sha256,
        "boundary_candidate": rebuilt_boundary.state_sha256,
    })
    rebuilt = _wrap_state(
        execution, base_state=trial.base_trial.candidate,
        accepted_boundary_scalar=rebuilt_boundary,
        accepted_chain_sha256=expected_chain)
    if (trial.boundary_scalar_candidate.state_sha256 !=
            rebuilt_boundary.state_sha256 or
            trial.candidate.base_state is not trial.base_trial.candidate or
            trial.candidate.accepted_boundary_scalar.state_sha256 !=
                rebuilt_boundary.state_sha256 or
            trial.candidate.accepted_chain_sha256 != expected_chain or
            trial.candidate.state_sha256 != rebuilt.state_sha256):
        raise CoupledWetDryOpenOwnerV98Error(
            "v98 open owner boundary memory or candidate differs")


def commit_coupled_wetdry_open_cpu_trial_v98(
    execution: CoupledWetDryOpenCPUExecutionV98,
    parent: CoupledWetDryOpenCPUStateV98,
    q3_provider: object,
    trial: CoupledWetDryOpenCPUTrialV98,
) -> CoupledWetDryOpenCPUStateV98:
    _validate_trial(execution, parent, q3_provider, trial)
    accepted = _base.commit_coupled_wetdry_cpu_trial_v97(
        execution.base_execution, parent.base_state,
        q3_provider, trial.base_trial)
    if accepted is not trial.base_trial.candidate:
        raise CoupledWetDryOpenOwnerV98Error(
            "v98 open owner base commit substituted candidate")
    return trial.candidate


def rollback_coupled_wetdry_open_cpu_trial_v98(
    execution: CoupledWetDryOpenCPUExecutionV98,
    parent: CoupledWetDryOpenCPUStateV98,
    q3_provider: object,
    trial: CoupledWetDryOpenCPUTrialV98,
) -> CoupledWetDryOpenCPUStateV98:
    _validate_trial(execution, parent, q3_provider, trial)
    restored = _base.rollback_coupled_wetdry_cpu_trial_v97(
        execution.base_execution, parent.base_state,
        q3_provider, trial.base_trial)
    if restored is not parent.base_state:
        raise CoupledWetDryOpenOwnerV98Error(
            "v98 open owner rollback changed parent")
    return parent


cancel_coupled_wetdry_open_cpu_trial_v98 = \
    rollback_coupled_wetdry_open_cpu_trial_v98


def coupled_wetdry_open_cpu_checkpoint_v98(
    execution: CoupledWetDryOpenCPUExecutionV98,
    state: CoupledWetDryOpenCPUStateV98,
    q3_provider: object,
) -> dict[str, object]:
    validate_coupled_wetdry_open_cpu_state_v98(
        state, execution, q3_provider)
    body = {
        "schema": SCHEMA,
        "scope_id": CHECKPOINT_SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "execution_fingerprint": execution.fingerprint,
        "base_checkpoint": _base.coupled_wetdry_cpu_checkpoint_v97(
            execution.base_execution, state.base_state, q3_provider),
        "boundary_checkpoint": accepted_boundary_scalar_checkpoint_v96(
            state.accepted_boundary_scalar,
            _v92_execution(execution.active_execution),
            execution.boundary_execution),
        "accepted_chain_sha256": state.accepted_chain_sha256,
        "state_sha256": state.state_sha256,
    }
    return {**body, "fingerprint": _hash(body)}


def load_coupled_wetdry_open_cpu_checkpoint_v98(
    execution: CoupledWetDryOpenCPUExecutionV98,
    document: Mapping[str, object],
    q3_provider: object,
) -> CoupledWetDryOpenCPUStateV98:
    keys = {
        "schema", "scope_id", "semantic_id", "execution_fingerprint",
        "base_checkpoint", "boundary_checkpoint",
        "accepted_chain_sha256", "state_sha256", "fingerprint",
    }
    if not isinstance(document, Mapping) or set(document) != keys:
        raise CoupledWetDryOpenOwnerV98Error(
            "v98 open owner checkpoint inventory differs")
    body = {key: value for key, value in document.items()
            if key != "fingerprint"}
    if (document["fingerprint"] != _hash(body) or
            document["schema"] != SCHEMA or
            document["scope_id"] != CHECKPOINT_SCOPE_ID or
            document["semantic_id"] != SEMANTIC_ID or
            document["execution_fingerprint"] != execution.fingerprint):
        raise CoupledWetDryOpenOwnerV98Error(
            "v98 open owner checkpoint identity differs")
    provider_checkpoint = q3_provider.checkpoint()
    try:
        base_state = _base.load_coupled_wetdry_cpu_checkpoint_v97(
            execution.base_execution,
            document["base_checkpoint"], q3_provider)
        boundary = load_accepted_boundary_scalar_checkpoint_v96(
            document["boundary_checkpoint"],
            _v92_execution(execution.active_execution),
            execution.boundary_execution)
        state = _wrap_state(
            execution, base_state=base_state,
            accepted_boundary_scalar=boundary,
            accepted_chain_sha256=document["accepted_chain_sha256"],
            q3_provider=q3_provider)
        if state.state_sha256 != document["state_sha256"]:
            raise CoupledWetDryOpenOwnerV98Error(
                "v98 open owner checkpoint state differs")
        return state
    except Exception as original_error:
        try:
            q3_provider.restore_checkpoint(provider_checkpoint)
        except Exception as cleanup_error:
            raise RuntimeError(
                "v98 open owner checkpoint and restore both failed") \
                from cleanup_error
        raise original_error


__all__ = [
    "BACKEND", "CHECKPOINT_SCOPE_ID", "SCHEMA", "SEMANTIC_ID", "SCOPE_ID",
    "CoupledWetDryOpenCPUExecutionV98",
    "CoupledWetDryOpenCPUStateV98",
    "CoupledWetDryOpenCPUTrialV98",
    "CoupledWetDryOpenOwnerCountsV98",
    "CoupledWetDryOpenOwnerV98Error",
    "begin_coupled_wetdry_open_cpu_trial_v98",
    "cancel_coupled_wetdry_open_cpu_trial_v98",
    "commit_coupled_wetdry_open_cpu_trial_v98",
    "coupled_wetdry_open_cpu_checkpoint_v98",
    "execution_document_coupled_wetdry_open_v98",
    "load_coupled_wetdry_open_cpu_checkpoint_v98",
    "resolve_coupled_wetdry_open_cpu_execution_v98",
    "resolve_coupled_wetdry_open_cpu_state_v98",
    "rollback_coupled_wetdry_open_cpu_trial_v98",
    "validate_coupled_wetdry_open_cpu_state_v98",
]
