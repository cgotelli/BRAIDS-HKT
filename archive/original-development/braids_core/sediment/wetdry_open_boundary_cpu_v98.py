"""Additive physical-boundary adapter for the CPU v97 wet/dry midpoint.

The historical v97 residual is intentionally left unchanged.  This adapter
delegates every physical boundary evaluation to the exact cut-cell boundary
owner, replaces the discarded reflected face flux by that owner's corrected
physical flux, then rebuilds the v97 donor limiter and conservative gather.
The same accepted-parent boundary density is frozen across the Picard solve.

This connected implementation keeps the augmented f-wave, density projection
and v91 work in their qualified native runtimes.  Physical-face replacement,
donor limiting, pressure/gather and the water ledger are one additive native
OpenMP batch.  NumPy rebuilds the same identities only in independent focused
tests; the supported numerical path records zero Python cell/face loops.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
import threading
from typing import Callable

import numpy as np

from .coupled_geometry_native_cpu_v96 import DENSITY_WATER_PLUS_SEDIMENT
from .coupled_open_boundary_native_cpu_v96 import (
    AcceptedBoundaryDensityReceiptV96,
    validate_accepted_boundary_density_receipt_v96,
)
from .density_active_projection_v97 import (
    NativeDensityActiveProjectionRuntimeV97,
)
from .hydraulic_boundary_trace_cpu_v96 import (
    HydraulicBoundaryTraceProviderV96,
    HydraulicBoundaryTraceReceiptV96,
    execution_document_hydraulic_boundary_trace_v96,
    validate_hydraulic_boundary_trace_receipt_v96,
)
from .physical_flux_gather_native_cpu_v98 import (
    NativePhysicalFluxGatherRuntimeV98,
)
from .shared_turbulence_native_cpu_v97 import (
    NativeMomentumRateV97,
    NativeSharedTurbulenceRuntimeV97,
    PreparedNativeTurbulenceV97,
    _digest as _v91_digest,
    _result_token as _v91_result_token,
)
from .wetdry_fwave_native_cpu_v97 import (
    NativeWetDryFwaveRateResultV97,
    NativeWetDryFwaveRuntimeV97,
    WetDryFwaveCPUExecutionV97,
    WetDryMidpointTrialV97,
    _fingerprint as _v97_fingerprint,
    _midpoint_trial_payload,
    _result_payload as _v97_result_payload,
    execution_document_wetdry_fwave_v97,
    solve_wetdry_midpoint_trial_v97,
    validate_native_wetdry_rate_result_v97,
    validate_wetdry_midpoint_trial_v97,
)


SCHEMA = 1
SCOPE_ID = "braids.q4.wetdry.physical_boundary.cpu.v98"
BACKEND = "native_cpu_openmp_physical_boundary_gather"


class WetDryOpenBoundaryCPUV98Error(ValueError):
    """The delegated boundary residual, audit, or midpoint differs."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("ascii")


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


def _poll(value: bool | Callable[[], bool]) -> Callable[[], bool]:
    if type(value) is bool:
        return lambda: value
    if callable(value):
        return value
    raise TypeError("cancelled must be Boolean or callable")


def _readonly(value: object, dtype=np.float64) -> np.ndarray:
    result = np.array(value, dtype=dtype, order="C", copy=True)
    if not np.isfinite(result).all():
        raise WetDryOpenBoundaryCPUV98Error(
            "v98 open wet/dry result is non-finite")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class WetDryOpenBoundaryRateReceiptV98:
    execution_fingerprint: str
    boundary_execution_fingerprint: str
    boundary_density_receipt_sha256: str
    boundary_trace_receipt_sha256: str
    evaluation_index: int
    midpoint_sha256: str
    reflected_base_result_sha256: str
    embedded_result_sha256: str
    internal_turbulence_result_sha256: str
    physical_gather_runtime_sha256: str
    physical_gather_result_sha256: str
    physical_open_face_count: int
    exterior_densin_face_count: int
    discarded_reflective_riemann_call_count: int
    physical_flux_consumption_count: int
    draining_limiter_application_count: int
    native_augmented_riemann_call_count: int
    vectorized_boundary_rebuild_count: int
    native_physical_gather_launch_count: int
    numerical_python_cell_face_loop_count: int
    physical_boundary_viscosity_face_count: int
    state_or_clock_publication_count: int
    receipt_sha256: str


def _rate_payload(value: WetDryOpenBoundaryRateReceiptV98) -> dict:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID+":rate",
        **{key: item for key, item in value.__dict__.items()
           if key != "receipt_sha256"},
    }


@dataclass(frozen=True)
class WetDryOpenBoundaryResidualAuditV98:
    execution_fingerprint: str
    boundary_execution_fingerprint: str
    hydraulic_request_sha256: str
    parent_state_sha256: str
    evaluation_time_s: float
    dt_s: float
    rate_receipts: tuple[WetDryOpenBoundaryRateReceiptV98, ...]
    evaluation_count: int
    physical_flux_consumption_count: int
    boundary_trace_owner_call_count: int
    exterior_densin_face_evaluation_count: int
    internal_turbulence_application_count: int
    physical_boundary_viscosity_face_count: int
    accepted_clock_increment_count: int
    audit_sha256: str


def _audit_payload(value: WetDryOpenBoundaryResidualAuditV98) -> dict:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID+":audit",
        "execution": value.execution_fingerprint,
        "boundary_execution": value.boundary_execution_fingerprint,
        "request": value.hydraulic_request_sha256,
        "parent": value.parent_state_sha256,
        "evaluation_time_s": value.evaluation_time_s,
        "dt_s": value.dt_s,
        "rate_receipts": [item.receipt_sha256
                          for item in value.rate_receipts],
        "counts": [
            value.evaluation_count,
            value.physical_flux_consumption_count,
            value.boundary_trace_owner_call_count,
            value.exterior_densin_face_evaluation_count,
            value.internal_turbulence_application_count,
            value.physical_boundary_viscosity_face_count,
            value.accepted_clock_increment_count,
        ],
    }


@dataclass(frozen=True)
class WetDryOpenBoundaryTrialV98:
    execution_fingerprint: str
    boundary_execution_fingerprint: str
    wetdry_trial: WetDryMidpointTrialV97
    residual_audit: WetDryOpenBoundaryResidualAuditV98
    accepted_boundary_trace: HydraulicBoundaryTraceReceiptV96
    accepted_internal_turbulence_result: NativeMomentumRateV97
    physical_boundary_viscosity_policy: str
    physical_boundary_surrogate_count: int
    state_or_clock_publication_count: int
    trial_sha256: str


def _trial_payload(value: WetDryOpenBoundaryTrialV98) -> dict:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID+":trial",
        "execution": value.execution_fingerprint,
        "boundary_execution": value.boundary_execution_fingerprint,
        "wetdry_trial": value.wetdry_trial.trial_sha256,
        "audit": value.residual_audit.audit_sha256,
        "accepted_trace": value.accepted_boundary_trace.receipt_sha256,
        "accepted_internal_turbulence":
            value.accepted_internal_turbulence_result.result_sha256,
        "physical_boundary_viscosity_policy":
            value.physical_boundary_viscosity_policy,
        "physical_boundary_surrogate_count":
            value.physical_boundary_surrogate_count,
        "state_or_clock_publication_count":
            value.state_or_clock_publication_count,
    }


class WetDryOpenBoundaryRateRuntimeV98:
    """Exact-provider residual adapter consumed by the historical midpoint."""

    def __init__(
        self, execution: WetDryFwaveCPUExecutionV97,
        base_runtime: NativeWetDryFwaveRuntimeV97, *,
        boundary_trace_provider: HydraulicBoundaryTraceProviderV96,
        boundary_density: AcceptedBoundaryDensityReceiptV96,
        hydraulic_request_sha256: str,
        parent_state_sha256: str,
        evaluation_time_s: float,
        dt_s: float,
        turbulence_runtime: NativeSharedTurbulenceRuntimeV97,
        turbulence_prepared: PreparedNativeTurbulenceV97,
        physical_gather_runtime: NativePhysicalFluxGatherRuntimeV98,
    ) -> None:
        execution_document_wetdry_fwave_v97(execution)
        boundary_execution = boundary_trace_provider.execution
        execution_document_hydraulic_boundary_trace_v96(boundary_execution)
        validate_accepted_boundary_density_receipt_v96(
            boundary_density, boundary_execution)
        base = execution.base_execution.base_execution
        if (type(base_runtime) is not NativeWetDryFwaveRuntimeV97 or
                base_runtime.receipt != execution.native_build_receipt or
                type(boundary_trace_provider) is not
                    HydraulicBoundaryTraceProviderV96 or
                type(turbulence_runtime) is not
                    NativeSharedTurbulenceRuntimeV97 or
                type(turbulence_prepared) is not
                    PreparedNativeTurbulenceV97 or
                type(physical_gather_runtime) is not
                    NativePhysicalFluxGatherRuntimeV98 or
                boundary_execution.cell_count != execution.cell_count or
                boundary_execution.face_count != execution.face_count or
                not np.array_equal(boundary_execution.owner, base.owner) or
                not np.array_equal(
                    boundary_execution.neighbour, base.neighbour) or
                not np.array_equal(
                    boundary_execution.face_distance_m,
                    base.face_distance_m) or
                not math.isfinite(float(evaluation_time_s)) or
                float(evaluation_time_s) < 0.0 or
                not math.isfinite(float(dt_s)) or float(dt_s) <= 0.0 or
                turbulence_prepared.cap_receipt.dt_s != float(dt_s)):
            raise WetDryOpenBoundaryCPUV98Error(
                "v98 open wet/dry runtime binding differs")
        self.execution = execution
        self.base_runtime = base_runtime
        self.boundary_trace_provider = boundary_trace_provider
        self.boundary_density = boundary_density
        self.hydraulic_request_sha256 = hydraulic_request_sha256
        self.parent_state_sha256 = parent_state_sha256
        self.evaluation_time_s = float(evaluation_time_s)
        self.dt_s = float(dt_s)
        self.turbulence_runtime = turbulence_runtime
        self.turbulence_prepared = turbulence_prepared
        self.physical_gather_runtime = physical_gather_runtime
        self._lock = threading.Lock()
        self._receipts: list[WetDryOpenBoundaryRateReceiptV98] = []
        self._traces: list[HydraulicBoundaryTraceReceiptV96] = []
        self._turbulence: list[NativeMomentumRateV97] = []

    def reset_residual_audit(self) -> None:
        with self._lock:
            self._receipts.clear()
            self._traces.clear()
            self._turbulence.clear()

    def residual_audit(self) -> WetDryOpenBoundaryResidualAuditV98:
        with self._lock:
            receipts = tuple(self._receipts)
        placeholder = WetDryOpenBoundaryResidualAuditV98(
            self.execution.fingerprint,
            self.boundary_trace_provider.execution.fingerprint,
            self.hydraulic_request_sha256, self.parent_state_sha256,
            self.evaluation_time_s, self.dt_s, receipts, len(receipts),
            sum(item.physical_flux_consumption_count for item in receipts),
            len(receipts),
            sum(item.exterior_densin_face_count for item in receipts),
            len(receipts), 0, 0, "")
        return WetDryOpenBoundaryResidualAuditV98(
            **{**placeholder.__dict__, "audit_sha256": _hash(
                _audit_payload(placeholder))})

    def traces(self) -> tuple[HydraulicBoundaryTraceReceiptV96, ...]:
        with self._lock:
            return tuple(self._traces)

    def turbulence_results(self) -> tuple[NativeMomentumRateV97, ...]:
        with self._lock:
            return tuple(self._turbulence)

    def evaluate(
        self, execution: WetDryFwaveCPUExecutionV97,
        midpoint_q: object, *, bed_elevation_m: object,
        accepted_water_density_kg_m3: object,
        accepted_mixture_density_kg_m3: object,
        frozen_wet_mask: object, dt_s: float,
        cancelled: bool | Callable[[], bool] = False,
    ) -> NativeWetDryFwaveRateResultV97:
        poll = _poll(cancelled)
        if poll():
            raise InterruptedError(
                "v98 open wet/dry cancelled before residual")
        if execution is not self.execution or float(dt_s) != self.dt_s:
            raise WetDryOpenBoundaryCPUV98Error(
                "v98 open wet/dry residual execution or dt differs")
        reflected = self.base_runtime.evaluate(
            execution, midpoint_q, bed_elevation_m=bed_elevation_m,
            accepted_water_density_kg_m3=accepted_water_density_kg_m3,
            accepted_mixture_density_kg_m3=accepted_mixture_density_kg_m3,
            frozen_wet_mask=frozen_wet_mask, dt_s=dt_s,
            cancelled=poll)
        base = execution.base_execution.base_execution
        # The generic boundary receipt validators deliberately freeze their
        # accepted input arrays.  Keep transaction-local copies here so an
        # audit cannot make the midpoint solver's mutable iterate read-only.
        q = np.array(midpoint_q, dtype=np.float64, order="C", copy=True)
        bed = np.array(
            bed_elevation_m, dtype=np.float64, order="C", copy=True)
        water = np.array(
            accepted_water_density_kg_m3,
            dtype=np.float64, order="C", copy=True)
        mixture = np.array(
            accepted_mixture_density_kg_m3,
            dtype=np.float64, order="C", copy=True)
        wet = np.array(
            frozen_wet_mask, dtype=np.int32, order="C", copy=True)
        with self._lock:
            evaluation_index = len(self._receipts)+1
        trace = self.boundary_trace_provider.resolve(
            q, bed,
            hydraulic_request_sha256=self.hydraulic_request_sha256,
            parent_state_sha256=self.parent_state_sha256,
            evaluation_index=evaluation_index,
            evaluation_time_s=self.evaluation_time_s,
            evaluation_dt_s=self.dt_s)
        validate_hydraulic_boundary_trace_receipt_v96(
            trace, self.boundary_trace_provider.execution,
            midpoint_q=q, bed_elevation_m=bed,
            hydraulic_request_sha256=self.hydraulic_request_sha256,
            parent_state_sha256=self.parent_state_sha256,
            evaluation_index=evaluation_index,
            evaluation_time_s=self.evaluation_time_s)
        physical = np.asarray(
            trace.physical_boundary_flux_owned_mask, dtype=bool)
        exterior = np.asarray(
            trace.exterior_densin_active_mask, dtype=bool)
        open_faces = np.flatnonzero(physical)
        if open_faces.size < 1:
            raise WetDryOpenBoundaryCPUV98Error(
                "v98 open wet/dry requires a physical open face")

        gathered = self.physical_gather_runtime.evaluate(
            q, wet,
            reflected.raw_left_face_flux,
            reflected.raw_right_face_flux,
            trace.physical_open_face_flux,
            trace.physical_boundary_flux_owned_mask,
            owner=base.owner, neighbour=base.neighbour,
            cell_face_offset=base.cell_face_offset,
            cell_faces=base.cell_faces,
            cell_face_sign=base.cell_face_sign,
            face_length_m=base.face_length_m,
            cell_area_m2=base.cell_area_m2,
            normal=base.normal,
            gravity_m_s2=base.gravity_m_s2,
            dry_tolerance_m=base.dry_tolerance_m,
            dt_s=dt_s,
            physical_trace_receipt_sha256=trace.receipt_sha256,
            reflected_result_sha256=reflected.result_sha256,
            native_threads=execution.native_threads,
            cancelled=poll)
        raw_left = gathered["raw_left_face_flux"]
        raw_right = gathered["raw_right_face_flux"]
        limited_left = gathered["limited_left_face_flux"]
        limited_right = gathered["limited_right_face_flux"]
        theta = gathered["draining_factor"]
        rate = np.array(gathered["base_rate"], copy=True)
        water_flux = gathered[
            "accepted_limited_face_water_flux_m2_s"]

        # Internal v91 is evaluated once on the actual internal wet/wet mask.
        # Physical boundary viscous transport is already part of the cut-cell
        # corrected physical flux and is not applied again here.
        turbulence = self.turbulence_runtime.evaluate(
            self.turbulence_prepared, q, wet,
            reflected.wet_wet_face_mask,
            native_threads=execution.native_threads, cancelled=poll)
        rate += turbulence.momentum_rate_m2_s2

        selected_density = (
            mixture if execution.base_execution.density_mode ==
            DENSITY_WATER_PLUS_SEDIMENT else water)
        boundary_density = (
            self.boundary_density.mixture_density_kg_m3
            if execution.base_execution.density_mode ==
            DENSITY_WATER_PLUS_SEDIMENT else
            self.boundary_density.water_density_kg_m3)
        acceleration = np.array(
            reflected.face_normal_acceleration_m_s2, copy=True)
        density_mask = np.array(
            reflected.wet_wet_face_mask, dtype=np.int32, copy=True)
        acceleration[physical] = 0.0
        density_mask[physical] = 0
        owner = np.asarray(base.owner, dtype=np.int64)
        active_exterior = (
            exterior & (wet[owner] != 0) &
            (trace.trace_q[0] > 0.0))
        f = np.flatnonzero(active_exterior)
        if f.size:
            o = owner[f]
            acceleration[f] = (
                -base.gravity_m_s2*
                (boundary_density[f]-selected_density[o])*
                (trace.trace_q[0, f]+q[0, o])/
                (4.0*base.hydraulic_reference_density_kg_m3*
                 np.asarray(base.face_distance_m)[f]))
            density_mask[f] = 1
        if poll():
            raise InterruptedError(
                "v98 open wet/dry cancelled after residual")

        outputs = (
            raw_left, raw_right, limited_left, limited_right, theta, rate,
            water_flux, acceleration, density_mask, wet)
        frozen = tuple(_readonly(item, np.int32 if index in (8, 9)
                                 else np.float64)
                       for index, item in enumerate(outputs))
        placeholder = NativeWetDryFwaveRateResultV97(
            *frozen, _digest(q), _digest(bed), _digest(selected_density),
            float(dt_s), reflected.augmented_riemann_call_count,
            int(gathered["metadata"]["limited_face_count"]), 0, 0,
            execution.native_threads, "")
        result = NativeWetDryFwaveRateResultV97(
            **{**placeholder.__dict__, "result_sha256": _v97_fingerprint(
                _v97_result_payload(placeholder, execution))})
        validate_native_wetdry_rate_result_v97(result, execution)
        receipt0 = WetDryOpenBoundaryRateReceiptV98(
            execution.fingerprint,
            self.boundary_trace_provider.execution.fingerprint,
            self.boundary_density.receipt_sha256,
            trace.receipt_sha256, evaluation_index, _digest(q),
            reflected.result_sha256, result.result_sha256,
            turbulence.result_sha256,
            self.physical_gather_runtime.receipt["receipt_sha256"],
            gathered["receipt_sha256"],
            int(open_faces.size), int(f.size),
            int(open_faces.size), int(open_faces.size),
            int(gathered["metadata"]["limited_face_count"]),
            reflected.augmented_riemann_call_count, 0, 1, 0, 0, 0, "")
        receipt = WetDryOpenBoundaryRateReceiptV98(
            **{**receipt0.__dict__, "receipt_sha256": _hash(
                _rate_payload(receipt0))})
        with self._lock:
            if len(self._receipts)+1 != evaluation_index:
                raise WetDryOpenBoundaryCPUV98Error(
                    "v98 open wet/dry residual audit order differs")
            self._receipts.append(receipt)
            self._traces.append(trace)
            self._turbulence.append(turbulence)
        return result


def _publish_turbulence_split(
    execution: WetDryFwaveCPUExecutionV97,
    trial: WetDryMidpointTrialV97,
    turbulence: NativeMomentumRateV97,
) -> WetDryMidpointTrialV97:
    """Move embedded internal v91 into the historical typed evidence lane."""
    raw = trial.native_rate_result
    # The additive residual runtime evaluates the physical-boundary f-wave and
    # v91 momentum in one native rate.  The historical v97 receipt, however,
    # carries those two owners in separate lanes and validates their sum with
    # exact array equality.  A plain ``combined - v91`` is not sufficient:
    # adding v91 back can differ from the accepted combined rate by one ULP
    # once the lagged HLES field becomes non-zero (normally update two).
    #
    # Preserve the already-converged physical rate exactly and compensate only
    # the compatibility base lane.  This is receipt publication, not another
    # physical evaluation; the authoritative v91 array remains byte-identical
    # to the native result recorded by the v98 outer audit.
    density_rate = np.asarray(trial.accepted_density_momentum_rate)
    target_physical_rate = (
        np.asarray(trial.accepted_base_fwave_rate)+density_rate)
    v91_rate = np.asarray(turbulence.momentum_rate_m2_s2)
    base_rate = np.asarray(raw.base_fwave_rate)-v91_rate
    for _ in range(4):
        rebuilt = (base_rate+density_rate)+v91_rate
        correction = target_physical_rate-rebuilt
        if np.all(correction == 0.0):
            break
        base_rate = base_rate+correction
    if not np.array_equal(
            (base_rate+density_rate)+v91_rate, target_physical_rate):
        raise WetDryOpenBoundaryCPUV98Error(
            "v98 open wet/dry compatibility split is not exact")
    base_rate = _readonly(base_rate)
    raw0 = replace(raw, base_fwave_rate=base_rate, result_sha256="")
    raw_split = replace(
        raw0, result_sha256=_v97_fingerprint(
            _v97_result_payload(raw0, execution)))
    compat0 = replace(
        turbulence,
        active_wet_wet_face_mask_sha256=
            _v91_digest(raw_split.wet_wet_face_mask),
        result_sha256="")
    compat = replace(compat0, result_sha256=_v91_result_token(compat0))
    trial0 = replace(
        trial, native_rate_result=raw_split,
        accepted_base_fwave_rate=base_rate,
        accepted_turbulence_momentum_rate=compat.momentum_rate_m2_s2,
        accepted_turbulence_face_flux=compat.face_flux_m4_s2,
        shared_turbulence_result=compat,
        shared_turbulence_application_count=
            trial.true_residual_evaluation_count,
        trial_sha256="")
    return replace(
        trial0, trial_sha256=_v97_fingerprint(
            _midpoint_trial_payload(trial0)))


def solve_wetdry_midpoint_with_physical_boundary_v98(
    execution: WetDryFwaveCPUExecutionV97,
    parent_q: object, *, bed_elevation_m: object,
    accepted_water_density_kg_m3: object,
    accepted_mixture_density_kg_m3: object,
    dt_s: float,
    rate_runtime: WetDryOpenBoundaryRateRuntimeV98,
    density_projection_runtime: NativeDensityActiveProjectionRuntimeV97,
    cancelled: bool | Callable[[], bool] = False,
) -> WetDryOpenBoundaryTrialV98:
    if (type(rate_runtime) is not WetDryOpenBoundaryRateRuntimeV98 or
            rate_runtime.execution is not execution or
            float(dt_s) != rate_runtime.dt_s):
        raise WetDryOpenBoundaryCPUV98Error(
            "v98 open wet/dry solve/runtime binding differs")
    rate_runtime.reset_residual_audit()
    try:
        embedded = solve_wetdry_midpoint_trial_v97(
            execution, parent_q, bed_elevation_m=bed_elevation_m,
            accepted_water_density_kg_m3=accepted_water_density_kg_m3,
            accepted_mixture_density_kg_m3=
                accepted_mixture_density_kg_m3,
            dt_s=dt_s, native_runtime=rate_runtime,
            density_projection_runtime=density_projection_runtime,
            shared_turbulence_runtime=None,
            shared_turbulence_prepared=None, cancelled=cancelled)
        turbulence = rate_runtime.turbulence_results()
        traces = rate_runtime.traces()
        audit = rate_runtime.residual_audit()
        if (len(turbulence) != embedded.true_residual_evaluation_count or
                len(traces) != embedded.true_residual_evaluation_count or
                audit.evaluation_count !=
                    embedded.true_residual_evaluation_count):
            raise WetDryOpenBoundaryCPUV98Error(
                "v98 open wet/dry residual evidence count differs")
        trial = _publish_turbulence_split(
            execution, embedded, turbulence[-1])
        validate_wetdry_midpoint_trial_v97(
            trial, execution, parent_q=parent_q,
            bed_elevation_m=bed_elevation_m,
            accepted_water_density_kg_m3=accepted_water_density_kg_m3,
            accepted_mixture_density_kg_m3=
                accepted_mixture_density_kg_m3,
            dt_s=dt_s)
        placeholder = WetDryOpenBoundaryTrialV98(
            execution.fingerprint,
            rate_runtime.boundary_trace_provider.execution.fingerprint,
            trial, audit, traces[-1], turbulence[-1],
            "cutcell_corrected_physical_flux_owns_boundary_viscosity_"
            "internal_v91_not_duplicated", 0, 0, "")
        result = WetDryOpenBoundaryTrialV98(
            **{**placeholder.__dict__, "trial_sha256": _hash(
                _trial_payload(placeholder))})
        validate_wetdry_open_boundary_trial_v98(
            result, execution,
            boundary_execution=
                rate_runtime.boundary_trace_provider.execution,
            parent_q=parent_q, bed_elevation_m=bed_elevation_m,
            accepted_water_density_kg_m3=
                accepted_water_density_kg_m3,
            accepted_mixture_density_kg_m3=
                accepted_mixture_density_kg_m3,
            boundary_density=rate_runtime.boundary_density,
            hydraulic_request_sha256=
                rate_runtime.hydraulic_request_sha256,
            parent_state_sha256=rate_runtime.parent_state_sha256,
            evaluation_time_s=rate_runtime.evaluation_time_s,
            dt_s=dt_s)
        return result
    except Exception:
        rate_runtime.reset_residual_audit()
        raise


def validate_wetdry_open_boundary_trial_v98(
    value: WetDryOpenBoundaryTrialV98,
    execution: WetDryFwaveCPUExecutionV97, *,
    boundary_execution,
    parent_q: object,
    bed_elevation_m: object,
    accepted_water_density_kg_m3: object,
    accepted_mixture_density_kg_m3: object,
    boundary_density: AcceptedBoundaryDensityReceiptV96,
    hydraulic_request_sha256: str,
    parent_state_sha256: str,
    evaluation_time_s: float,
    dt_s: float,
) -> None:
    """Rebuild the connected hydraulic evidence without rerunning an owner."""
    if type(value) is not WetDryOpenBoundaryTrialV98:
        raise WetDryOpenBoundaryCPUV98Error(
            "v98 open wet/dry trial type differs")
    execution_document_wetdry_fwave_v97(execution)
    execution_document_hydraulic_boundary_trace_v96(boundary_execution)
    validate_accepted_boundary_density_receipt_v96(
        boundary_density, boundary_execution)
    validate_wetdry_midpoint_trial_v97(
        value.wetdry_trial, execution, parent_q=parent_q,
        bed_elevation_m=bed_elevation_m,
        accepted_water_density_kg_m3=accepted_water_density_kg_m3,
        accepted_mixture_density_kg_m3=accepted_mixture_density_kg_m3,
        dt_s=dt_s)
    audit = value.residual_audit
    receipts = audit.rate_receipts
    if (type(audit) is not WetDryOpenBoundaryResidualAuditV98 or
            audit.audit_sha256 != _hash(_audit_payload(audit)) or
            audit.execution_fingerprint != execution.fingerprint or
            audit.boundary_execution_fingerprint !=
                boundary_execution.fingerprint or
            audit.hydraulic_request_sha256 != hydraulic_request_sha256 or
            audit.parent_state_sha256 != parent_state_sha256 or
            audit.evaluation_time_s != float(evaluation_time_s) or
            audit.dt_s != float(dt_s) or
            audit.evaluation_count != len(receipts) or
            audit.evaluation_count !=
                value.wetdry_trial.true_residual_evaluation_count or
            audit.physical_flux_consumption_count != sum(
                item.physical_flux_consumption_count for item in receipts) or
            audit.boundary_trace_owner_call_count != len(receipts) or
            audit.exterior_densin_face_evaluation_count != sum(
                item.exterior_densin_face_count for item in receipts) or
            audit.internal_turbulence_application_count != len(receipts) or
            audit.physical_boundary_viscosity_face_count != 0 or
            audit.accepted_clock_increment_count != 0):
        raise WetDryOpenBoundaryCPUV98Error(
            "v98 open wet/dry residual audit differs")
    for index, receipt in enumerate(receipts, 1):
        if (type(receipt) is not WetDryOpenBoundaryRateReceiptV98 or
                receipt.receipt_sha256 != _hash(_rate_payload(receipt)) or
                receipt.execution_fingerprint != execution.fingerprint or
                receipt.boundary_execution_fingerprint !=
                    boundary_execution.fingerprint or
                receipt.boundary_density_receipt_sha256 !=
                    boundary_density.receipt_sha256 or
                receipt.evaluation_index != index or
                receipt.physical_open_face_count !=
                    int(np.count_nonzero(boundary_execution.open_mask)) or
                receipt.discarded_reflective_riemann_call_count !=
                    receipt.physical_open_face_count or
                receipt.physical_flux_consumption_count !=
                    receipt.physical_open_face_count or
                receipt.native_augmented_riemann_call_count !=
                    execution.face_count or
                not isinstance(receipt.physical_gather_runtime_sha256, str) or
                len(receipt.physical_gather_runtime_sha256) != 64 or
                not isinstance(receipt.physical_gather_result_sha256, str) or
                len(receipt.physical_gather_result_sha256) != 64 or
                receipt.vectorized_boundary_rebuild_count != 0 or
                receipt.native_physical_gather_launch_count != 1 or
                receipt.numerical_python_cell_face_loop_count != 0 or
                receipt.physical_boundary_viscosity_face_count != 0 or
                receipt.state_or_clock_publication_count != 0):
            raise WetDryOpenBoundaryCPUV98Error(
                "v98 open wet/dry rate receipt differs")
    accepted = value.accepted_boundary_trace
    validate_hydraulic_boundary_trace_receipt_v96(
        accepted, boundary_execution,
        midpoint_q=value.wetdry_trial.midpoint_q,
        bed_elevation_m=bed_elevation_m,
        hydraulic_request_sha256=hydraulic_request_sha256,
        parent_state_sha256=parent_state_sha256,
        evaluation_index=len(receipts),
        evaluation_time_s=evaluation_time_s)
    turbulence = value.accepted_internal_turbulence_result
    physical = accepted.physical_boundary_flux_owned_mask.astype(bool)
    raw = value.wetdry_trial.native_rate_result
    if (not receipts or
            accepted.receipt_sha256 !=
                receipts[-1].boundary_trace_receipt_sha256 or
            turbulence.result_sha256 !=
                receipts[-1].internal_turbulence_result_sha256 or
            turbulence.midpoint_q_sha256 !=
                _v91_digest(value.wetdry_trial.midpoint_q) or
            not np.array_equal(
                turbulence.momentum_rate_m2_s2,
                value.wetdry_trial.accepted_turbulence_momentum_rate) or
            not np.array_equal(
                turbulence.face_flux_m4_s2,
                value.wetdry_trial.accepted_turbulence_face_flux) or
            not np.array_equal(
                raw.raw_left_face_flux[:, physical],
                accepted.physical_open_face_flux[:, physical]) or
            not np.array_equal(
                raw.raw_right_face_flux[:, physical],
                accepted.physical_open_face_flux[:, physical]) or
            value.execution_fingerprint != execution.fingerprint or
            value.boundary_execution_fingerprint !=
                boundary_execution.fingerprint or
            value.physical_boundary_viscosity_policy !=
                "cutcell_corrected_physical_flux_owns_boundary_viscosity_"
                "internal_v91_not_duplicated" or
            value.physical_boundary_surrogate_count != 0 or
            value.state_or_clock_publication_count != 0 or
            value.trial_sha256 != _hash(_trial_payload(value))):
        raise WetDryOpenBoundaryCPUV98Error(
            "v98 open wet/dry trial evidence differs")


__all__ = [
    "BACKEND", "SCHEMA", "SCOPE_ID",
    "WetDryOpenBoundaryCPUV98Error",
    "WetDryOpenBoundaryRateReceiptV98",
    "WetDryOpenBoundaryResidualAuditV98",
    "WetDryOpenBoundaryTrialV98",
    "WetDryOpenBoundaryRateRuntimeV98",
    "solve_wetdry_midpoint_with_physical_boundary_v98",
    "validate_wetdry_open_boundary_trial_v98",
]
