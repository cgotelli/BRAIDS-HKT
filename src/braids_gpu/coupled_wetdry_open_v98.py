"""Atomic physical-open extension of the connected GPU wet/dry owner.

The historical v97 owner remains untouched.  This additive v98 wrapper binds
the already loaded physical boundary owner, augments its live face-density
image before the one rank-aware projection, feeds the same accepted physical
water flux to v92, and persists the accepted physical boundary scalar image
for the next hydraulic update.  It never reruns the boundary solver or a
Riemann problem and never treats a sampled density-inactive physical face as a
wall.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from types import MethodType
from typing import Callable, Mapping

import numpy as np

from braids_core.sediment.accepted_boundary_scalar_state_cpu_v96 import (
    AcceptedBoundaryScalarStateV96,
    make_accepted_boundary_scalar_state_from_v92_trial_v96,
    make_boundary_density_from_accepted_scalar_state_v96,
    validate_accepted_boundary_scalar_state_v96,
)
from braids_core.sediment.coupled_open_boundary_native_cpu_v96 import (
    AcceptedBoundaryDensityReceiptV96,
    validate_accepted_boundary_density_receipt_v96,
)
from braids_core.sediment.hydraulic_boundary_trace_cpu_v96 import (
    HydraulicBoundaryTraceProviderV96,
    HydraulicBoundaryTraceReceiptV96,
    execution_document_hydraulic_boundary_trace_v96,
    validate_hydraulic_boundary_trace_receipt_v96,
)
from braids_gpu.coupled_open_boundary_v96 import (
    CoupledOpenBoundaryGPUExecutionV96,
    execution_document_coupled_open_boundary_gpu_v96,
)
from braids_gpu.coupled_wetdry_v97 import (
    CoupledWetDryGPUExecutionV97,
    CoupledWetDryGPUStateV97,
    CoupledWetDryGPUTrialV97,
    ResidentCoupledWetDryLifecycleV97,
    _grid,
    _v92_execution,
    _v92_state,
    execution_document_coupled_wetdry_gpu_v97,
    validate_coupled_wetdry_gpu_state_v97,
)
from braids_gpu.wetdry_open_densin_v98 import (
    ResidentWetDryOpenDensInGPUV98,
)


SCHEMA = 1
SCOPE_ID = "braids.q4.coupled_wetdry.physical_open.actual_cuda.v98"
SEMANTIC_ID = "one_live_boundary_owner_wet_active_densin_scalar_memory"
BACKEND = "actual_cuda_physical_open_moving_wetdry"
MAX_ACCEPTED_UPDATES = 4
_SHA = re.compile(r"[0-9a-f]{64}\Z")


class CoupledWetDryOpenGPUV98Error(ValueError):
    """The physical-open moving-wet/dry composition differs."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError, OverflowError) as error:
        raise CoupledWetDryOpenGPUV98Error(
            "v98 open wet/dry value is not finite canonical JSON") from error


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


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise CoupledWetDryOpenGPUV98Error(
            f"v98 open wet/dry {name} must be a SHA-256 digest")
    return value


def _poll(value: bool | Callable[[], bool]) -> Callable[[], bool]:
    if isinstance(value, (bool, np.bool_)):
        return lambda: bool(value)
    if not callable(value):
        raise TypeError("cancelled must be Boolean or callable")

    def checked() -> bool:
        result = value()
        if not isinstance(result, (bool, np.bool_)):
            raise TypeError("cancelled callback must return Boolean data")
        return bool(result)
    return checked


def _frozen(value, shape, name, *, integer=False, nonnegative=False):
    raw = np.asarray(value)
    if (raw.shape != shape or raw.dtype.kind not in "fiub" or
            not np.isfinite(raw).all() or
            (integer and np.any(raw != np.floor(raw))) or
            (nonnegative and np.any(raw < 0.0))):
        raise CoupledWetDryOpenGPUV98Error(
            f"v98 open wet/dry {name} differs")
    result = np.array(
        raw, dtype=np.int64 if integer else np.float64,
        order="C", copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class CoupledWetDryOpenGPUExecutionV98:
    base_execution: CoupledWetDryGPUExecutionV97
    open_boundary_execution: CoupledOpenBoundaryGPUExecutionV96
    semantic_id: str
    fingerprint: str

    @property
    def active_execution(self):
        return self.base_execution.active_execution

    @property
    def fraction_count(self):
        return self.base_execution.fraction_count

    @property
    def cell_count(self):
        return self.base_execution.cell_count

    @property
    def face_count(self):
        return self.base_execution.face_count


def _execution_payload(value: CoupledWetDryOpenGPUExecutionV98) -> dict:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "semantic_id": value.semantic_id,
        "base_execution": value.base_execution.fingerprint,
        "open_boundary_execution": value.open_boundary_execution.fingerprint,
        "physical_flux_owner": "existing_live_boundary_apply_then_pressure",
        "density_owner": "wet_active_open_augment_then_one_projection",
        "scalar_boundary_timing":
            "current_forcing_to_scalar_then_accepted_image_next_hydraulics",
        "moving_wetdry": True,
        "automatic_cpu_fallback": False,
        "production_admission": False,
        "backend": BACKEND,
    }


def resolve_coupled_wetdry_open_gpu_execution_v98(
    base_execution: CoupledWetDryGPUExecutionV97,
    open_boundary_execution: CoupledOpenBoundaryGPUExecutionV96,
) -> CoupledWetDryOpenGPUExecutionV98:
    execution_document_coupled_wetdry_gpu_v97(base_execution)
    execution_document_coupled_open_boundary_gpu_v96(
        open_boundary_execution)
    trace = open_boundary_execution.boundary_trace_execution
    hydraulic = base_execution.hydraulic_execution.density_execution
    if (trace.cell_count != base_execution.cell_count or
            trace.face_count != base_execution.face_count or
            not np.array_equal(trace.owner, hydraulic.owner) or
            not np.array_equal(trace.neighbour, hydraulic.neighbour) or
            not np.array_equal(trace.face_distance_m,
                               hydraulic.face_distance_m)):
        raise CoupledWetDryOpenGPUV98Error(
            "v98 open wet/dry boundary/hydraulic geometry differs")
    placeholder = CoupledWetDryOpenGPUExecutionV98(
        base_execution, open_boundary_execution, SEMANTIC_ID, "")
    return CoupledWetDryOpenGPUExecutionV98(
        **{**placeholder.__dict__,
           "fingerprint": _hash(_execution_payload(placeholder))})


def execution_document_coupled_wetdry_open_gpu_v98(value) -> dict:
    if type(value) is not CoupledWetDryOpenGPUExecutionV98:
        raise CoupledWetDryOpenGPUV98Error(
            "v98 open wet/dry execution type differs")
    execution_document_coupled_wetdry_gpu_v97(value.base_execution)
    execution_document_coupled_open_boundary_gpu_v96(
        value.open_boundary_execution)
    payload = _execution_payload(value)
    if value.semantic_id != SEMANTIC_ID or value.fingerprint != _hash(payload):
        raise CoupledWetDryOpenGPUV98Error(
            "v98 open wet/dry execution fingerprint differs")
    return {**payload, "fingerprint": value.fingerprint}


@dataclass(frozen=True)
class CoupledWetDryOpenGPUStateV98:
    execution_fingerprint: str
    base_state: CoupledWetDryGPUStateV97
    accepted_boundary_scalar_state: AcceptedBoundaryScalarStateV96
    accepted_chain_sha256: str
    state_sha256: str

    @property
    def q(self):
        return self.base_state.q

    @property
    def accepted_updates(self):
        return self.base_state.accepted_updates

    @property
    def time_s(self):
        return self.base_state.time_s


def _state_payload(value: CoupledWetDryOpenGPUStateV98) -> dict:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID+":state",
        "execution": value.execution_fingerprint,
        "base_state": value.base_state.state_sha256,
        "accepted_boundary_scalar_state":
            value.accepted_boundary_scalar_state.state_sha256,
        "accepted_chain": value.accepted_chain_sha256,
    }


def _wrap_state(execution, base_state, boundary_state, chain):
    placeholder = CoupledWetDryOpenGPUStateV98(
        execution.fingerprint, base_state, boundary_state, chain, "")
    return CoupledWetDryOpenGPUStateV98(
        **{**placeholder.__dict__,
           "state_sha256": _hash(_state_payload(placeholder))})


def resolve_initial_coupled_wetdry_open_gpu_state_v98(
    execution: CoupledWetDryOpenGPUExecutionV98, *,
    base_state: CoupledWetDryGPUStateV97,
    accepted_boundary_scalar_state: AcceptedBoundaryScalarStateV96,
    q3_provider,
) -> CoupledWetDryOpenGPUStateV98:
    execution_document_coupled_wetdry_open_gpu_v98(execution)
    validate_coupled_wetdry_gpu_state_v97(
        base_state, execution.base_execution, q3_provider)
    validate_accepted_boundary_scalar_state_v96(
        accepted_boundary_scalar_state,
        _v92_execution(execution.active_execution),
        execution.open_boundary_execution.boundary_trace_execution,
        accepted_update=base_state.accepted_updates,
        time_s=base_state.time_s)
    chain = _hash({
        "scope_id": SCOPE_ID+":initial",
        "base_state": base_state.state_sha256,
        "boundary_state": accepted_boundary_scalar_state.state_sha256,
    })
    return _wrap_state(
        execution, base_state, accepted_boundary_scalar_state, chain)


def validate_coupled_wetdry_open_gpu_state_v98(
    value, execution, q3_provider,
) -> None:
    execution_document_coupled_wetdry_open_gpu_v98(execution)
    if (type(value) is not CoupledWetDryOpenGPUStateV98 or
            value.execution_fingerprint != execution.fingerprint or
            value.state_sha256 != _hash(_state_payload(value))):
        raise CoupledWetDryOpenGPUV98Error(
            "v98 open wet/dry state identity differs")
    validate_coupled_wetdry_gpu_state_v97(
        value.base_state, execution.base_execution, q3_provider)
    validate_accepted_boundary_scalar_state_v96(
        value.accepted_boundary_scalar_state,
        _v92_execution(execution.active_execution),
        execution.open_boundary_execution.boundary_trace_execution,
        accepted_update=value.accepted_updates, time_s=value.time_s)


@dataclass(frozen=True)
class CoupledWetDryOpenHydraulicEvidenceGPUV98:
    accepted_boundary_density: AcceptedBoundaryDensityReceiptV96
    boundary_trace_receipts: tuple[HydraulicBoundaryTraceReceiptV96, ...]
    midpoint_q_history: np.ndarray
    open_density_runtime_sha256: str
    open_density_kernel_count: int
    physical_boundary_owner_call_count: int
    second_boundary_solver_call_count: int
    second_riemann_call_count: int
    density_projection_count: int
    boundary_memory_publication_count: int
    evidence_sha256: str


def _evidence_payload(value: CoupledWetDryOpenHydraulicEvidenceGPUV98):
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID+":hydraulic-evidence",
        "accepted_boundary_density":
            value.accepted_boundary_density.receipt_sha256,
        "trace_receipts": [
            receipt.receipt_sha256
            for receipt in value.boundary_trace_receipts],
        "midpoint_q_history": _digest(value.midpoint_q_history),
        "open_density_runtime": value.open_density_runtime_sha256,
        "counts": [value.open_density_kernel_count,
                   value.physical_boundary_owner_call_count,
                   value.second_boundary_solver_call_count,
                   value.second_riemann_call_count,
                   value.density_projection_count,
                   value.boundary_memory_publication_count],
        "call_order": (
            "physical_trace_flux_then_open_densin_before_one_projection;"
            "accepted_scalar_image_next_hydraulic_update"),
        "backend": BACKEND,
    }


@dataclass(frozen=True)
class CoupledWetDryOpenGPUTrialV98:
    execution_fingerprint: str
    parent_state_sha256: str
    base_trial: CoupledWetDryGPUTrialV97
    hydraulic_evidence: CoupledWetDryOpenHydraulicEvidenceGPUV98
    candidate_boundary_scalar_state: AcceptedBoundaryScalarStateV96
    candidate: CoupledWetDryOpenGPUStateV98
    candidate_publication_count: int
    backend: str
    trial_sha256: str


def _trial_payload(value: CoupledWetDryOpenGPUTrialV98):
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID+":trial",
        "execution": value.execution_fingerprint,
        "parent": value.parent_state_sha256,
        "base_trial": value.base_trial.trial_sha256,
        "hydraulic_evidence": value.hydraulic_evidence.evidence_sha256,
        "candidate_boundary_scalar_state":
            value.candidate_boundary_scalar_state.state_sha256,
        "candidate": value.candidate.state_sha256,
        "candidate_publication_count": value.candidate_publication_count,
        "backend": value.backend,
    }


class _LiveOpenResidualAdapterV98:
    """Instance-only adapters restored after each provisional solve."""

    def __init__(self, hydraulic, open_density, *, parent,
                 accepted_boundary_density, hydraulic_request_sha256,
                 cancelled):
        self.hydraulic = hydraulic
        self.cp = hydraulic.cp
        self.controller = hydraulic.controller
        self.boundary = self.controller.solver.boundary
        self.open_density = open_density
        self.parent = parent
        self.accepted_boundary_density = accepted_boundary_density
        self.request = _sha(hydraulic_request_sha256, "hydraulic request")
        self.poll = _poll(cancelled)
        self.physical_host = np.ascontiguousarray(
            open_density.execution.open_mask.astype(np.int32))
        self.physical = self.cp.ascontiguousarray(
            self.cp.asarray(self.physical_host))
        self.requested = self.physical.copy()
        self._saved = None
        self._current_midpoint = None
        self._current_trace = None
        self._base_density_mask = self.cp.empty(
            hydraulic.nf, dtype=self.cp.int32)
        self.midpoints = []
        self.traces = []
        self.fluxes = []
        self.densin_masks = []
        self.owner_shas = []

    def install(self):
        if self._saved is not None:
            raise CoupledWetDryOpenGPUV98Error(
                "v98 open residual adapter is already installed")
        self.open_density.set_parent_boundary_density(
            self.accepted_boundary_density)
        self.open_density.validate_runtime()
        original_apply = self.boundary.apply
        original_face = self.hydraulic._launch_face_density
        original_projection = self.hydraulic._launch_projection
        original_rate = self.hydraulic._rate
        self._saved = (original_apply, original_face,
                       original_projection, original_rate)
        adapter = self

        def apply(boundary_self, q, trace):
            result = original_apply(q, trace)
            adapter._current_midpoint = q.copy()
            adapter._current_trace = trace[:3].copy()
            return result

        def face(runtime_self, depth, selected_density):
            original_face(depth, selected_density)
            adapter.cp.copyto(
                adapter._base_density_mask, runtime_self.face_active)
            index = len(adapter.midpoints)+1
            owner_sha = _hash({
                "scope_id": SCOPE_ID+":live-trace-owner",
                "request": adapter.request,
                "parent": adapter.parent.state_sha256,
                "evaluation": index,
            })
            result = adapter.open_density.launch_resident(
                depth, selected_density, runtime_self.wet,
                adapter._current_trace, adapter.physical,
                adapter.requested, runtime_self.face_acceleration,
                runtime_self.face_active,
                current_trace_owner_sha256=owner_sha,
                cancelled=adapter.poll)
            if bool(adapter.cp.any(result["status"] != 0).item()):
                raise CoupledWetDryOpenGPUV98Error(
                    "v98 wet-active exterior density kernel failed")
            adapter.cp.copyto(
                runtime_self.face_acceleration,
                result["face_acceleration"])
            adapter.cp.copyto(runtime_self.face_active,
                              result["density_mask"])
            adapter.owner_shas.append(owner_sha)
            adapter.densin_masks.append(result["exterior_active"].copy())

        def projection(runtime_self):
            original_projection()
            # Ordinary v91 consumes the original internal wet-wet mask; the
            # open density mask belongs only to the single projection above.
            adapter.cp.copyto(
                runtime_self.face_active, adapter._base_density_mask)

        def rate(runtime_self, midpoint, wet, selected_density,
                 dt, time, poll):
            result = original_rate(midpoint, wet, selected_density,
                                   dt, time, poll)
            if (adapter._current_midpoint is None or
                    adapter._current_trace is None):
                raise CoupledWetDryOpenGPUV98Error(
                    "v98 live boundary trace was not captured")
            adapter.midpoints.append(adapter._current_midpoint.copy())
            adapter.traces.append(adapter._current_trace.copy())
            adapter.fluxes.append(
                runtime_self.v98_final_augmented_left_flux.copy())
            adapter._current_midpoint = None
            adapter._current_trace = None
            return result

        self.boundary.apply = MethodType(apply, self.boundary)
        self.hydraulic._launch_face_density = MethodType(
            face, self.hydraulic)
        self.hydraulic._launch_projection = MethodType(
            projection, self.hydraulic)
        self.hydraulic._rate = MethodType(rate, self.hydraulic)

    def remove(self, *, clear: bool):
        if self._saved is not None:
            (self.boundary.apply, self.hydraulic._launch_face_density,
             self.hydraulic._launch_projection,
             self.hydraulic._rate) = self._saved
            self._saved = None
        self._current_midpoint = self._current_trace = None
        self.open_density.last_resident_result = None
        if clear:
            self.midpoints.clear(); self.traces.clear(); self.fluxes.clear()
            self.densin_masks.clear(); self.owner_shas.clear()

    def host_histories(self):
        count = len(self.midpoints)
        if (count < 1 or len(self.traces) != count or
                len(self.fluxes) != count or len(self.densin_masks) != count or
                len(self.owner_shas) != count):
            raise CoupledWetDryOpenGPUV98Error(
                "v98 open residual history counts differ")
        physical = self.physical_host.astype(bool)
        midpoint = np.ascontiguousarray(self.cp.asnumpy(
            self.cp.stack(self.midpoints)))
        trace = np.ascontiguousarray(self.cp.asnumpy(
            self.cp.stack(self.traces)))
        flux = np.ascontiguousarray(self.cp.asnumpy(
            self.cp.stack(self.fluxes)))
        densin = np.ascontiguousarray(self.cp.asnumpy(
            self.cp.stack(self.densin_masks)), dtype=np.int32)
        trace[:, :, ~physical] = 0.0
        flux[:, :, ~physical] = 0.0
        return midpoint, trace, flux, densin


def _make_trace_receipts(execution, parent, request, dt,
                         midpoint, trace, flux, densin, owner_shas):
    trace_execution = (
        execution.open_boundary_execution.boundary_trace_execution)
    physical = trace_execution.open_mask.astype(np.int32)
    state_probe = {
        "parent": parent.state_sha256,
        "request": request,
        "boundary_state": parent.accepted_boundary_scalar_state.state_sha256,
    }
    current = {"row": 0}

    def resolver(**_):
        row = current["row"]
        reconstruction = _hash({
            "scope_id": SCOPE_ID+":live-reconstruction",
            "midpoint": _digest(midpoint[row]),
            "trace": _digest(trace[row]),
        })
        branch = _hash({
            "scope_id": SCOPE_ID+":live-boundary-branch",
            "owner_runtime": trace_execution.hydraulic_owner_runtime_sha256,
            "physical_mask": _digest(physical),
            "densin_mask": _digest(densin[row]),
            "flux": _digest(flux[row]),
        })
        return {
            "trace_q": trace[row],
            "physical_open_face_flux": flux[row],
            "physical_boundary_flux_owned_mask": physical,
            "exterior_densin_active_mask": densin[row],
            "reconstruction_receipt_sha256": reconstruction,
            "boundary_branch_receipt_sha256": branch,
            "owner_evaluation_sha256": owner_shas[row],
        }

    provider = HydraulicBoundaryTraceProviderV96(
        trace_execution, resolver=resolver,
        state_probe=lambda: state_probe)
    values = []
    for row in range(midpoint.shape[0]):
        current["row"] = row
        receipt = provider.resolve(
            midpoint[row], parent.base_state.bed_elevation_m,
            hydraulic_request_sha256=request,
            parent_state_sha256=parent.state_sha256,
            evaluation_index=row+1,
            evaluation_time_s=parent.time_s+.5*dt,
            evaluation_dt_s=dt)
        values.append(receipt)
    return tuple(values)


class ResidentCoupledWetDryOpenLifecycleV98:
    """One atomic physical-open v98 transaction around the v97 owner."""

    def __init__(self, execution, base_runtime,
                 open_density_runtime: ResidentWetDryOpenDensInGPUV98):
        execution_document_coupled_wetdry_open_gpu_v98(execution)
        if (type(base_runtime) is not ResidentCoupledWetDryLifecycleV97 or
                type(open_density_runtime) is not
                    ResidentWetDryOpenDensInGPUV98 or
                base_runtime.execution.fingerprint !=
                    execution.base_execution.fingerprint or
                open_density_runtime.execution.fingerprint !=
                    execution.open_boundary_execution.
                    boundary_trace_execution.fingerprint):
            raise CoupledWetDryOpenGPUV98Error(
                "v98 open wet/dry resident owner binding differs")
        self.execution = execution
        self.base = base_runtime
        self.open_density = open_density_runtime
        self.cp = base_runtime.cp
        self._parent = None
        self._trial = None
        self.last_accepted_receipt = None

    def begin(self, parent, *, q3_inputs: Mapping[str, object],
              shear_velocity_m_s, closure_chezy_mhalf_s, dt_s: float,
              prescribed_boundary_concentration_kg_m3,
              prescribed_boundary_fluid_transport_scalar,
              cancelled=False):
        validate_coupled_wetdry_open_gpu_state_v98(
            parent, self.execution, self.base.active.q3_provider)
        if self._parent is not None or self._trial is not None:
            raise CoupledWetDryOpenGPUV98Error(
                "v98 open wet/dry transaction is unfinished")
        if parent.accepted_updates >= MAX_ACCEPTED_UPDATES:
            raise CoupledWetDryOpenGPUV98Error(
                "v98 open wet/dry update limit was reached")
        poll = _poll(cancelled)
        if poll():
            raise InterruptedError("v98 open wet/dry cancelled before work")
        dt = float(dt_s)
        if not math.isfinite(dt) or dt <= 0.0:
            raise CoupledWetDryOpenGPUV98Error(
                "v98 open wet/dry timestep differs")
        v92_execution = _v92_execution(self.execution.active_execution)
        ng, nb = self.execution.fraction_count, v92_execution.boundary_count
        prescribed_sediment = _frozen(
            prescribed_boundary_concentration_kg_m3, (ng, nb),
            "prescribed boundary concentration", nonnegative=True)
        prescribed_fluid = _frozen(
            prescribed_boundary_fluid_transport_scalar, (2, nb),
            "prescribed boundary fluid")
        request = _hash({
            "scope_id": SCOPE_ID+":request",
            "execution": self.execution.fingerprint,
            "parent": parent.state_sha256,
            "dt_s": dt,
        })
        trace_execution = (
            self.execution.open_boundary_execution.
            boundary_trace_execution)
        accepted_density = make_boundary_density_from_accepted_scalar_state_v96(
            parent.accepted_boundary_scalar_state, v92_execution,
            trace_execution, parent_state_sha256=parent.state_sha256)
        validate_accepted_boundary_density_receipt_v96(
            accepted_density, trace_execution)
        adapter = _LiveOpenResidualAdapterV98(
            self.base.hydraulic, self.open_density, parent=parent,
            accepted_boundary_density=accepted_density,
            hydraulic_request_sha256=request, cancelled=poll)
        original_active_begin = self.base.active.begin_wetdry_trial
        open_mask = trace_execution.open_mask
        owner = np.asarray(trace_execution.owner)
        neighbour = np.asarray(trace_execution.neighbour)
        internal = neighbour >= 0
        grid = _grid(self.execution.active_execution)
        outer = self

        def active_begin(active_self, active_parent, *args, **kwargs):
            if not adapter.traces:
                raise CoupledWetDryOpenGPUV98Error(
                    "v98 scalar stage lacks accepted physical trace")
            hydraulic = kwargs["hydraulic_evidence"].wetdry_trial
            endpoint = np.asarray(hydraulic.endpoint_q)
            trace = outer.cp.asnumpy(adapter.traces[-1])
            densin = outer.cp.asnumpy(adapter.densin_masks[-1]).astype(bool)
            wet = endpoint[0] > grid.dry_tolerance_m
            face_depth = np.zeros(outer.execution.face_count)
            active = np.zeros(outer.execution.face_count, dtype=np.int32)
            active[internal] = (
                wet[owner[internal]] & wet[neighbour[internal]])
            face_depth[internal] = .5*(
                endpoint[0, owner[internal]]+
                endpoint[0, neighbour[internal]])
            active[open_mask] = densin[open_mask]
            face_depth[open_mask] = trace[0, open_mask]
            scalar = kwargs["v92_scalar_inputs"]
            scalar["prescribed_boundary_concentration_kg_m3"] = \
                outer.cp.ascontiguousarray(outer.cp.asarray(
                    prescribed_sediment))
            scalar["transport_face_active"] = outer.cp.ascontiguousarray(
                outer.cp.asarray(active, dtype=outer.cp.int32))
            scalar["accepted_face_exchange_area_m2"] = \
                outer.cp.ascontiguousarray(outer.cp.asarray(
                    face_depth*np.asarray(grid.face_length_m)))
            kwargs["prescribed_boundary_fluid"] = \
                outer.cp.ascontiguousarray(outer.cp.asarray(
                    prescribed_fluid))
            return original_active_begin(active_parent, *args, **kwargs)

        base_trial = None
        try:
            adapter.install()
            self.base.active.begin_wetdry_trial = MethodType(
                active_begin, self.base.active)
            base_trial = self.base.begin(
                parent.base_state, q3_inputs=q3_inputs,
                shear_velocity_m_s=shear_velocity_m_s,
                closure_chezy_mhalf_s=closure_chezy_mhalf_s,
                dt_s=dt, cancelled=poll)
            midpoint, trace, flux, densin = adapter.host_histories()
            if (len(adapter.owner_shas) !=
                    base_trial.hydraulic_trial.true_residual_evaluation_count):
                raise CoupledWetDryOpenGPUV98Error(
                    "v98 open density/residual counts differ")
            receipts = _make_trace_receipts(
                self.execution, parent, request, dt, midpoint, trace, flux,
                densin, adapter.owner_shas)
            v92_trial = base_trial.active_trial.base_v93_trial.base_v92_trial
            boundary_candidate = \
                make_accepted_boundary_scalar_state_from_v92_trial_v96(
                    v92_execution, trace_execution, v92_trial)
            chain = _hash({
                "previous": parent.accepted_chain_sha256,
                "base": base_trial.candidate.state_sha256,
                "boundary": boundary_candidate.state_sha256,
                "traces": [receipt.receipt_sha256 for receipt in receipts],
            })
            candidate = _wrap_state(
                self.execution, base_trial.candidate,
                boundary_candidate, chain)
            frozen_midpoint = np.array(midpoint, order="C", copy=True)
            frozen_midpoint.setflags(write=False)
            evidence_placeholder = CoupledWetDryOpenHydraulicEvidenceGPUV98(
                accepted_density, receipts, frozen_midpoint,
                self.open_density.receipt["receipt_sha256"],
                len(receipts), len(receipts), 0, 0, len(receipts), 0, "")
            evidence = CoupledWetDryOpenHydraulicEvidenceGPUV98(
                **{**evidence_placeholder.__dict__,
                   "evidence_sha256":
                       _hash(_evidence_payload(evidence_placeholder))})
            placeholder = CoupledWetDryOpenGPUTrialV98(
                self.execution.fingerprint, parent.state_sha256,
                base_trial, evidence, boundary_candidate, candidate,
                0, BACKEND, "")
            trial = CoupledWetDryOpenGPUTrialV98(
                **{**placeholder.__dict__,
                   "trial_sha256": _hash(_trial_payload(placeholder))})
            self._validate_trial(parent, trial)
            if poll():
                raise InterruptedError(
                    "v98 open wet/dry cancelled before publication")
        except Exception:
            if (base_trial is not None and
                    self.base._parent is parent.base_state):
                self.base.rollback(parent.base_state, base_trial)
            adapter.remove(clear=True)
            self.open_density.parent_density = None
            self.open_density.parent_density_device = None
            raise
        finally:
            self.base.active.begin_wetdry_trial = original_active_begin
            adapter.remove(clear=False)
        self.open_density.parent_density = None
        self.open_density.parent_density_device = None
        self._parent, self._trial = parent, trial
        return trial

    def _validate_trial(self, parent, trial):
        validate_coupled_wetdry_open_gpu_state_v98(
            parent, self.execution, self.base.active.q3_provider)
        if (type(trial) is not CoupledWetDryOpenGPUTrialV98 or
                trial.execution_fingerprint != self.execution.fingerprint or
                trial.parent_state_sha256 != parent.state_sha256 or
                trial.backend != BACKEND or
                trial.candidate_publication_count != 0 or
                trial.trial_sha256 != _hash(_trial_payload(trial))):
            raise CoupledWetDryOpenGPUV98Error(
                "v98 open wet/dry trial identity differs")
        self.base._validate_trial(parent.base_state, trial.base_trial)
        evidence = trial.hydraulic_evidence
        trace_execution = (
            self.execution.open_boundary_execution.
            boundary_trace_execution)
        validate_accepted_boundary_density_receipt_v96(
            evidence.accepted_boundary_density, trace_execution)
        count = (
            trial.base_trial.hydraulic_trial.
            true_residual_evaluation_count)
        accepted_masked_midpoint = np.where(
            trial.base_trial.hydraulic_trial.
            frozen_wet_mask[None, :] != 0,
            trial.base_trial.hydraulic_trial.midpoint_q, 0.0)
        midpoint_history = np.asarray(evidence.midpoint_q_history)
        count_checks = {
            "type": type(evidence) is
                CoupledWetDryOpenHydraulicEvidenceGPUV98,
            "token": evidence.evidence_sha256 ==
                _hash(_evidence_payload(evidence)),
            "midpoint_shape": midpoint_history.shape ==
                (count, 3, self.execution.cell_count),
            "midpoint_dtype": midpoint_history.dtype == np.float64,
            "midpoint_finite": np.isfinite(midpoint_history).all(),
            "accepted_midpoint": np.array_equal(
                midpoint_history[-1],
                accepted_masked_midpoint),
            "trace_count": len(evidence.boundary_trace_receipts) == count,
            "density_count": evidence.open_density_kernel_count == count,
            "boundary_count":
                evidence.physical_boundary_owner_call_count == count,
            "second_boundary": evidence.second_boundary_solver_call_count == 0,
            "second_riemann": evidence.second_riemann_call_count == 0,
            "projection_count": evidence.density_projection_count == count,
            "memory_publication":
                evidence.boundary_memory_publication_count == 0,
        }
        if not all(count_checks.values()):
            midpoint_delta = (float(np.max(np.abs(
                midpoint_history[-1]-
                accepted_masked_midpoint)))
                if midpoint_history.shape ==
                    (count, 3, self.execution.cell_count) else math.inf)
            raise CoupledWetDryOpenGPUV98Error(
                "v98 open wet/dry evidence counts differ: "+
                ",".join(name for name, passed in count_checks.items()
                         if not passed)+
                f"; midpoint_linf={midpoint_delta:.17g}")
        for index, receipt in enumerate(
                evidence.boundary_trace_receipts, 1):
            validate_hydraulic_boundary_trace_receipt_v96(
                receipt, trace_execution,
                midpoint_q=midpoint_history[index-1],
                bed_elevation_m=parent.base_state.bed_elevation_m,
                hydraulic_request_sha256=receipt.hydraulic_request_sha256,
                parent_state_sha256=parent.state_sha256,
                evaluation_index=index,
                evaluation_time_s=(
                    parent.time_s+.5*
                    trial.base_trial.hydraulic_trial.dt_s))
        v92_trial = trial.base_trial.active_trial.base_v93_trial.base_v92_trial
        validate_accepted_boundary_scalar_state_v96(
            trial.candidate_boundary_scalar_state,
            _v92_execution(self.execution.active_execution),
            trace_execution,
            accepted_update=trial.base_trial.candidate.accepted_updates,
            time_s=trial.base_trial.candidate.time_s)
        expected_boundary = \
            make_accepted_boundary_scalar_state_from_v92_trial_v96(
                _v92_execution(self.execution.active_execution),
                trace_execution, v92_trial)
        if (expected_boundary.state_sha256 !=
                trial.candidate_boundary_scalar_state.state_sha256 or
                trial.candidate.base_state is not trial.base_trial.candidate or
                trial.candidate.accepted_boundary_scalar_state is not
                    trial.candidate_boundary_scalar_state or
                trial.candidate.state_sha256 !=
                    _hash(_state_payload(trial.candidate))):
            raise CoupledWetDryOpenGPUV98Error(
                "v98 open wet/dry candidate differs")

    def commit(self, parent, trial):
        if parent is not self._parent or trial is not self._trial:
            raise CoupledWetDryOpenGPUV98Error(
                "v98 open wet/dry trial is not open")
        try:
            self._validate_trial(parent, trial)
            base = self.base.commit(parent.base_state, trial.base_trial)
            if base is not trial.base_trial.candidate:
                raise CoupledWetDryOpenGPUV98Error(
                    "v98 open wet/dry base commit substituted candidate")
        except Exception:
            self.base.restore_failed_publication(parent.base_state, trial.base_trial)
            self._parent = self._trial = None
            self.open_density.last_resident_result = None
            raise
        self._parent = self._trial = None
        self.last_accepted_receipt = {
            "scope_id": SCOPE_ID+":accepted",
            "trial_sha256": trial.trial_sha256,
            "accepted_state_sha256": trial.candidate.state_sha256,
            "actual_cuda_kernels": True,
            "automatic_cpu_fallback": False,
        }
        return trial.candidate

    def rollback(self, parent, trial):
        if parent is not self._parent or trial is not self._trial:
            raise CoupledWetDryOpenGPUV98Error(
                "v98 open wet/dry trial is not open")
        self._validate_trial(parent, trial)
        base = self.base.rollback(parent.base_state, trial.base_trial)
        if base is not parent.base_state:
            raise CoupledWetDryOpenGPUV98Error(
                "v98 open wet/dry rollback substituted parent")
        self._parent = self._trial = None
        return parent


__all__ = [
    "BACKEND", "SCHEMA", "SCOPE_ID", "SEMANTIC_ID",
    "CoupledWetDryOpenGPUV98Error",
    "CoupledWetDryOpenGPUExecutionV98",
    "CoupledWetDryOpenGPUStateV98",
    "CoupledWetDryOpenHydraulicEvidenceGPUV98",
    "CoupledWetDryOpenGPUTrialV98",
    "ResidentCoupledWetDryOpenLifecycleV98",
    "resolve_coupled_wetdry_open_gpu_execution_v98",
    "execution_document_coupled_wetdry_open_gpu_v98",
    "resolve_initial_coupled_wetdry_open_gpu_state_v98",
    "validate_coupled_wetdry_open_gpu_state_v98",
]
