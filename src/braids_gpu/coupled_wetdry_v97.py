"""Connected actual-CUDA v97 wet/dry morphodynamic transaction.

This bounded owner advances one hydraulic active-set solve, one transported
scalar/settling exchange, one Q3/authoritative Exner update, and one lagged
HLES candidate under a single accepted clock.  Hydraulics are provisional;
the v94b/v92 candidate is the sole accepted ``q`` image.  Bed elevation is
derived from accepted bed mass and becomes visible to the next hydraulic
trial without mutating the solver's layer arrays during a provisional step.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np

from braids_core.sediment.active_suspended_control_v94b import (
    execution_document_v94b,
    validate_active_suspended_control_state_v94b,
)
from braids_core.sediment.hles_producer_v90 import (
    CurrentClosureHandoffV90,
    HlesProducerStateV90,
    HlesProducerTrialV90,
    execution_document_v90,
    make_current_closure_handoff_v90,
    make_hles_input_receipt_v90,
    validate_hles_producer_state_v90,
)
from braids_core.sediment.shared_turbulence_consumers_v91 import (
    apply_chkvic_used_total_v91,
)
from braids_gpu.active_suspended_wetdry_v97 import (
    ResidentActiveSuspendedWetDryLifecycleV97,
)
from braids_gpu.hles_producer_v90 import ResidentHlesProducerLifecycleV90
from braids_gpu.transported_fluid_wetdry_v97 import (
    WetDryHydraulicEvidenceGPUV97,
    make_wetdry_hydraulic_evidence_gpu_v97,
    validate_wetdry_hydraulic_evidence_gpu_v97,
)
from braids_gpu.wetdry_fwave_v97 import (
    ResidentWetDryFwaveV97,
    WetDryFwaveGPUExecutionV97,
    WetDryFwaveGPUTrialV97,
    execution_document_wetdry_fwave_gpu_v97,
    validate_wetdry_fwave_gpu_trial_v97,
)


SCOPE_ID = "braids-q4-coupled-wetdry-owner-actual-cuda-v97-v1"
CHECKPOINT_SCOPE_ID = SCOPE_ID + ":checkpoint"
BACKEND = "actual_cuda"
MAX_ACCEPTED_UPDATES = 4


class CoupledWetDryGPUV97Error(ValueError):
    """The connected resident v97 transaction differs or was tampered."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          allow_nan=False).encode("ascii")
    except (TypeError, ValueError, OverflowError) as error:
        raise CoupledWetDryGPUV97Error(
            "v97 CUDA outer identity is not canonical JSON") from error


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _digest(*values: object) -> str:
    result = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(value)
        result.update(array.dtype.str.encode("ascii"))
        result.update(_canonical(list(array.shape)))
        result.update(array.tobytes(order="C"))
    return result.hexdigest()


def _readonly(value: object, shape: tuple[int, ...], name: str,
              *, positive=False) -> np.ndarray:
    raw = np.asarray(value)
    if (raw.shape != shape or raw.dtype.kind not in "fiu" or
            not np.isfinite(raw).all() or
            (positive and np.any(raw <= 0.0))):
        raise CoupledWetDryGPUV97Error(
            f"v97 CUDA outer {name} differs")
    result = np.array(raw, dtype=np.float64, order="C", copy=True)
    result.setflags(write=False)
    return result


def _v92_execution(active_execution):
    return active_execution.base_execution.base_execution


def _v91_execution(active_execution):
    return _v92_execution(active_execution).base_execution


def _grid(active_execution):
    return _v91_execution(active_execution).base_execution.base_execution


def _v92_state(active_state):
    return active_state.base_state.base_state


@dataclass(frozen=True)
class CoupledWetDryGPUExecutionV97:
    active_execution: object
    hydraulic_execution: WetDryFwaveGPUExecutionV97
    initial_bed_elevation_m: np.ndarray
    reference_bed_mass_kg_m2: np.ndarray
    dry_bulk_density_kg_m3: np.ndarray
    fingerprint: str

    @property
    def fraction_count(self) -> int:
        return self.active_execution.fraction_count

    @property
    def cell_count(self) -> int:
        return self.active_execution.cell_count

    @property
    def face_count(self) -> int:
        return _v92_execution(self.active_execution).face_count


def _execution_payload(value: CoupledWetDryGPUExecutionV97) -> dict:
    return {
        "scope_id": SCOPE_ID,
        "active_execution_fingerprint": value.active_execution.fingerprint,
        "hydraulic_execution_fingerprint":
            value.hydraulic_execution.fingerprint,
        "geometry_arrays_sha256": _digest(
            value.initial_bed_elevation_m,
            value.reference_bed_mass_kg_m2,
            value.dry_bulk_density_kg_m3),
        "bed_geometry_policy": "preserve_h_shift_eta_after_exner",
        "hydraulic_state_owner": "v94b_v92_candidate_only",
        "hles_lag": "accepted_parent_current_candidate_next_update",
        "maximum_accepted_updates": MAX_ACCEPTED_UPDATES,
        "backend": BACKEND,
        "automatic_cpu_fallback": False,
        "product_admission": False,
    }


def resolve_coupled_wetdry_gpu_execution_v97(
    active_execution,
    hydraulic_execution: WetDryFwaveGPUExecutionV97, *,
    initial_bed_elevation_m: object,
    reference_bed_mass_kg_m2: object,
    dry_bulk_density_kg_m3: object,
) -> CoupledWetDryGPUExecutionV97:
    execution_document_v94b(active_execution)
    execution_document_wetdry_fwave_gpu_v97(hydraulic_execution)
    v91 = _v91_execution(active_execution)
    ng, nc = active_execution.fraction_count, active_execution.cell_count
    if (active_execution.backend != "implicit_fwave_midpoint_gpu" or
            hydraulic_execution.shared_turbulence_execution_fingerprint !=
                v91.fingerprint or
            hydraulic_execution.cell_count != nc or
            hydraulic_execution.face_count !=
                _v92_execution(active_execution).face_count):
        raise CoupledWetDryGPUV97Error(
            "v97 CUDA outer child executions differ")
    bed = _readonly(initial_bed_elevation_m, (nc,), "initial bed")
    reference = _readonly(
        reference_bed_mass_kg_m2, (ng, nc), "reference bed mass")
    bulk = _readonly(
        dry_bulk_density_kg_m3, (ng,), "dry bulk density", positive=True)
    placeholder = CoupledWetDryGPUExecutionV97(
        active_execution, hydraulic_execution, bed, reference, bulk, "")
    return CoupledWetDryGPUExecutionV97(
        **{**placeholder.__dict__,
           "fingerprint": _hash(_execution_payload(placeholder))})


def execution_document_coupled_wetdry_gpu_v97(
    value: CoupledWetDryGPUExecutionV97,
) -> dict[str, object]:
    if type(value) is not CoupledWetDryGPUExecutionV97:
        raise CoupledWetDryGPUV97Error(
            "v97 CUDA outer execution type differs")
    execution_document_v94b(value.active_execution)
    execution_document_wetdry_fwave_gpu_v97(value.hydraulic_execution)
    payload = _execution_payload(value)
    if value.fingerprint != _hash(payload):
        raise CoupledWetDryGPUV97Error(
            "v97 CUDA outer execution fingerprint differs")
    return {**payload, "fingerprint": value.fingerprint}


def _bed_from_mass(execution, active_state) -> np.ndarray:
    mass = np.asarray(active_state.base_state.bed_mass_kg_m2)
    result = np.ascontiguousarray(
        execution.initial_bed_elevation_m+
        np.sum((mass-execution.reference_bed_mass_kg_m2)/
               execution.dry_bulk_density_kg_m3[:, None], axis=0))
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class CoupledWetDryGPUStateV97:
    execution_fingerprint: str
    active_state: object
    hles_state: HlesProducerStateV90
    bed_elevation_m: np.ndarray
    free_surface_elevation_m: np.ndarray
    accepted_chain_sha256: str
    state_sha256: str

    @property
    def q(self):
        return _v92_state(self.active_state).q

    @property
    def accepted_updates(self) -> int:
        return self.active_state.accepted_updates

    @property
    def time_s(self) -> float:
        return self.active_state.time_s


def _state_token(value: CoupledWetDryGPUStateV97) -> str:
    return _hash({
        "scope_id": SCOPE_ID+":state",
        "execution": value.execution_fingerprint,
        "active": value.active_state.state_sha256,
        "hles": value.hles_state.state_sha256,
        "geometry": _digest(value.bed_elevation_m,
                            value.free_surface_elevation_m),
        "chain": value.accepted_chain_sha256,
    })


def _wrap_state(execution, *, active_state, hles_state,
                accepted_chain_sha256) -> CoupledWetDryGPUStateV97:
    bed = _bed_from_mass(execution, active_state)
    eta = np.ascontiguousarray(_v92_state(active_state).q[0]+bed)
    eta.setflags(write=False)
    placeholder = CoupledWetDryGPUStateV97(
        execution.fingerprint, active_state, hles_state, bed, eta,
        accepted_chain_sha256, "")
    return CoupledWetDryGPUStateV97(
        **{**placeholder.__dict__, "state_sha256": _state_token(placeholder)})


def resolve_initial_coupled_wetdry_gpu_state_v97(
    execution: CoupledWetDryGPUExecutionV97, *, active_state,
    hles_state: HlesProducerStateV90, q3_provider,
) -> CoupledWetDryGPUStateV97:
    execution_document_coupled_wetdry_gpu_v97(execution)
    validate_active_suspended_control_state_v94b(
        active_state, execution.active_execution, q3_provider)
    hles_execution = _v91_execution(
        execution.active_execution).hles_execution
    validate_hles_producer_state_v90(hles_state, hles_execution)
    if (active_state.accepted_updates != hles_state.accepted_updates or
            active_state.time_s != hles_state.time_s):
        raise CoupledWetDryGPUV97Error(
            "v97 CUDA initial active/HLES clocks differ")
    chain = _hash({"scope_id": SCOPE_ID+":initial",
                   "active": active_state.state_sha256,
                   "hles": hles_state.state_sha256})
    return _wrap_state(
        execution, active_state=active_state, hles_state=hles_state,
        accepted_chain_sha256=chain)


def validate_coupled_wetdry_gpu_state_v97(
    value: CoupledWetDryGPUStateV97,
    execution: CoupledWetDryGPUExecutionV97, q3_provider,
) -> None:
    execution_document_coupled_wetdry_gpu_v97(execution)
    if type(value) is not CoupledWetDryGPUStateV97:
        raise CoupledWetDryGPUV97Error("v97 CUDA outer state type differs")
    validate_active_suspended_control_state_v94b(
        value.active_state, execution.active_execution, q3_provider)
    hles_execution = _v91_execution(
        execution.active_execution).hles_execution
    validate_hles_producer_state_v90(value.hles_state, hles_execution)
    expected_bed = _bed_from_mass(execution, value.active_state)
    if (value.execution_fingerprint != execution.fingerprint or
            value.active_state.accepted_updates !=
                value.hles_state.accepted_updates or
            value.active_state.time_s != value.hles_state.time_s or
            not np.array_equal(value.bed_elevation_m, expected_bed) or
            not np.array_equal(value.free_surface_elevation_m,
                               value.q[0]+expected_bed) or
            value.state_sha256 != _state_token(value)):
        raise CoupledWetDryGPUV97Error(
            "v97 CUDA outer state is stale, mutated, or cross-clock")


@dataclass(frozen=True)
class CoupledWetDryGPUTrialV97:
    execution_fingerprint: str
    parent_state_sha256: str
    current_handoff: CurrentClosureHandoffV90
    hydraulic_trial: WetDryFwaveGPUTrialV97
    hydraulic_evidence: WetDryHydraulicEvidenceGPUV97
    active_trial: object
    hles_trial: HlesProducerTrialV90
    hles_inputs: tuple[np.ndarray, ...]
    candidate: CoupledWetDryGPUStateV97
    backend: str
    trial_sha256: str


def _trial_token(value: CoupledWetDryGPUTrialV97) -> str:
    return _hash({
        "scope_id": SCOPE_ID+":trial",
        "execution": value.execution_fingerprint,
        "parent": value.parent_state_sha256,
        "handoff": value.current_handoff.receipt_sha256,
        "hydraulic": value.hydraulic_trial.trial_sha256,
        "evidence": value.hydraulic_evidence.evidence_sha256,
        "active": value.active_trial.trial_sha256,
        "hles": value.hles_trial.trial_sha256,
        "hles_inputs": _digest(*value.hles_inputs),
        "candidate": value.candidate.state_sha256,
        "backend": value.backend,
    })


def _accepted_density(execution, state):
    v92 = _v92_state(state.active_state)
    water = np.ascontiguousarray(v92.water_density_cell_kg_m3)
    depth = np.asarray(v92.q[0])
    dry = _grid(execution.active_execution).dry_tolerance_m
    concentration = np.zeros_like(
        v92.suspended_state.suspended_mass_kg_m2)
    wet = depth > dry
    concentration[:, wet] = (
        v92.suspended_state.suspended_mass_kg_m2[:, wet]/depth[wet])
    profile = _grid(execution.active_execution).profile
    mixture = water.copy()
    for grain, solid in enumerate(profile.sediment_density_kg_m3):
        mixture += concentration[grain]*(1.0-water/solid)
    return water, np.ascontiguousarray(mixture)


def _scalar_inputs(runtime, state, endpoint, shear_velocity):
    cp = runtime.cp
    execution = runtime.control_execution
    v92_execution = _v92_execution(execution)
    v92 = _v92_state(state.active_state)
    grid = _grid(execution)
    ng, nc, nf, nb = (execution.fraction_count, execution.cell_count,
                      v92_execution.face_count, v92_execution.boundary_count)
    shear = _readonly(shear_velocity, (nc,), "shear velocity")
    if np.any(shear < 0.0):
        raise CoupledWetDryGPUV97Error(
            "v97 CUDA shear velocity must be nonnegative")
    h = np.asarray(v92.q[0])
    wet0 = h > grid.dry_tolerance_m
    concentration = np.zeros((ng, nc), dtype=np.float64)
    concentration[:, wet0] = (
        v92.suspended_state.suspended_mass_kg_m2[:, wet0]/h[wet0])
    left = np.ascontiguousarray(concentration[:, grid.owner])
    right = np.array(left, copy=True, order="C")
    internal = grid.neighbour >= 0
    right[:, internal] = concentration[:, grid.neighbour[internal]]
    fluid = np.vstack((v92.temperature_cell_c, v92.salinity_cell_ppt))
    fluid_left = np.ascontiguousarray(fluid[:, grid.owner])
    fluid_right = np.array(fluid_left, copy=True, order="C")
    fluid_right[:, internal] = fluid[:, grid.neighbour[internal]]
    prescribed = np.zeros((ng, nb), dtype=np.float64)
    prescribed_fluid = np.broadcast_to(
        np.asarray((v92_execution.equilibrium_temperature_c,
                    v92_execution.equilibrium_salinity_ppt))[:, None],
        (2, nb)).copy()
    wet1 = np.asarray(endpoint[0]) > grid.dry_tolerance_m
    active = (internal & wet1[grid.owner] &
              wet1[np.maximum(grid.neighbour, 0)])
    face_depth = np.zeros(nf)
    face_depth[active] = .5*(
        endpoint[0, grid.owner[active]]+
        endpoint[0, grid.neighbour[active]])
    host = {
        "reconstructed_left_concentration_kg_m3": left,
        "reconstructed_right_concentration_kg_m3": right,
        "prescribed_boundary_concentration_kg_m3": prescribed,
        "accepted_face_exchange_area_m2":
            face_depth*grid.face_length_m,
        "transport_face_active": active.astype(np.int32),
        "shear_velocity_m_s": shear,
        "hydro_constant": np.ones(nc, dtype=np.int32),
    }
    device = {
        key: cp.ascontiguousarray(cp.asarray(
            value, dtype=cp.int32 if key in (
                "transport_face_active", "hydro_constant") else cp.float64))
        for key, value in host.items()
    }
    fluids = tuple(cp.ascontiguousarray(cp.asarray(value, dtype=cp.float64))
                   for value in (fluid_left, fluid_right,
                                 prescribed_fluid))
    return device, fluids


def _hles_inputs(execution, endpoint, face_flux, chezy):
    grid = _grid(execution.active_execution)
    wet = endpoint[0] > grid.dry_tolerance_m
    internal = grid.neighbour >= 0
    active = (internal & wet[grid.owner] &
              wet[np.maximum(grid.neighbour, 0)])
    depth = np.zeros(execution.face_count)
    depth[active] = .5*(endpoint[0, grid.owner[active]]+
                         endpoint[0, grid.neighbour[active]])
    velocity = np.zeros(execution.face_count)
    velocity[active] = face_flux[active]/depth[active]
    closure = _readonly(chezy, (execution.cell_count,),
                        "closure Chezy", positive=True)
    result = tuple(np.ascontiguousarray(value) for value in (
        velocity, active.astype(np.int32), endpoint[0], closure,
        wet.astype(np.int32)))
    for value in result:
        value.setflags(write=False)
    return result


class ResidentCoupledWetDryLifecycleV97:
    """Atomic connected actual-CUDA v97 owner."""

    def __init__(self, execution, hydraulic_runtime, active_runtime,
                 hles_runtime):
        execution_document_coupled_wetdry_gpu_v97(execution)
        if (type(hydraulic_runtime) is not ResidentWetDryFwaveV97 or
                type(active_runtime) is not
                    ResidentActiveSuspendedWetDryLifecycleV97 or
                type(hles_runtime) is not ResidentHlesProducerLifecycleV90 or
                hydraulic_runtime.execution.fingerprint !=
                    execution.hydraulic_execution.fingerprint or
                active_runtime.control_execution.fingerprint !=
                    execution.active_execution.fingerprint or
                hles_runtime.execution.fingerprint !=
                    _v91_execution(execution.active_execution).
                    hles_execution.fingerprint):
            raise CoupledWetDryGPUV97Error(
                "v97 CUDA outer resident owners differ")
        self.execution = execution
        self.hydraulic = hydraulic_runtime
        self.active = active_runtime
        self.hles = hles_runtime
        self.cp = hydraulic_runtime.cp
        self._parent = None
        self._trial = None
        self.last_accepted_receipt = None
        self._parent_checkpoint = None

    def begin(self, parent, *, q3_inputs: Mapping[str, object],
              shear_velocity_m_s, closure_chezy_mhalf_s,
              dt_s: float, cancelled=False):
        validate_coupled_wetdry_gpu_state_v97(
            parent, self.execution, self.active.q3_provider)
        if self._parent is not None or self._trial is not None:
            raise CoupledWetDryGPUV97Error(
                "v97 CUDA outer transaction is unfinished")
        if parent.accepted_updates >= MAX_ACCEPTED_UPDATES:
            raise CoupledWetDryGPUV97Error(
                "v97 CUDA outer accepted update limit was reached")
        poll = cancelled if callable(cancelled) else lambda: bool(cancelled)
        if poll():
            raise InterruptedError("v97 CUDA outer cancelled before work")
        dt = float(dt_s)
        if not math.isfinite(dt) or dt <= 0.0:
            raise CoupledWetDryGPUV97Error(
                "v97 CUDA outer timestep differs")
        # A child may fail after publication. Keep the full committed
        # resident/scalar/provider checkpoint, not just an open-trial handle.
        self._parent_checkpoint = self.checkpoint(parent)
        request = _hash({"scope_id": SCOPE_ID+":request",
                         "execution": self.execution.fingerprint,
                         "parent": parent.state_sha256, "dt_s": dt})
        hles_execution = _v91_execution(
            self.execution.active_execution).hles_execution
        handoff = make_current_closure_handoff_v90(
            hles_execution, parent.hles_state,
            hydraulic_trial_request_sha256=request)
        cap = apply_chkvic_used_total_v91(
            _v91_execution(self.execution.active_execution),
            parent.hles_state, handoff, dt_s=dt)
        self.hydraulic.bind_shared_turbulence_cap(cap)
        water, mixture = _accepted_density(self.execution, parent)
        active_trial = None
        hles_trial = None
        try:
            hydraulic = self.hydraulic.solve(
                self.cp.ascontiguousarray(self.cp.asarray(parent.q)),
                accepted_water_density_kg_m3=water,
                accepted_mixture_density_kg_m3=mixture,
                bed_elevation_m=parent.bed_elevation_m,
                dt_s=dt, evaluation_time_s=parent.time_s+.5*dt,
                cancelled=poll)
            evidence = make_wetdry_hydraulic_evidence_gpu_v97(
                _v91_execution(self.execution.active_execution),
                _v92_state(parent.active_state), cap,
                wetdry_execution=self.execution.hydraulic_execution,
                wetdry_trial=hydraulic,
                accepted_bed_elevation_m=parent.bed_elevation_m,
                accepted_water_density_kg_m3=water,
                accepted_mixture_density_kg_m3=mixture)
            scalar, fluids = _scalar_inputs(
                self.active, parent, hydraulic.endpoint_q,
                shear_velocity_m_s)
            active_trial = self.active.begin_wetdry_trial(
                parent.active_state, parent.hles_state, handoff,
                cap_receipt=cap, hydraulic_evidence=evidence,
                reconstructed_left_fluid=fluids[0],
                reconstructed_right_fluid=fluids[1],
                prescribed_boundary_fluid=fluids[2],
                v92_scalar_inputs=scalar, q3_inputs=q3_inputs,
                cancelled=poll)
            v92_trial = active_trial.base_v93_trial.base_v92_trial
            hles_inputs = _hles_inputs(
                self.execution, hydraulic.endpoint_q,
                hydraulic.accepted_limited_face_water_flux_m2_s,
                closure_chezy_mhalf_s)
            step = v92_trial.hydraulic_receipt.base_step_receipt
            hles_input = make_hles_input_receipt_v90(
                hles_execution, parent.hles_state,
                accepted_step_receipt=step, current_handoff=handoff,
                closure_interval_s=dt,
                accepted_face_normal_velocity_m_s=hles_inputs[0],
                active_face=hles_inputs[1], closure_depth_m=hles_inputs[2],
                closure_chezy_mhalf_s=hles_inputs[3],
                active_cell=hles_inputs[4])
            resident = tuple(self.cp.ascontiguousarray(self.cp.asarray(
                value, dtype=self.cp.int32 if index in (1, 4)
                else self.cp.float64))
                for index, value in enumerate(hles_inputs))
            hles_trial = self.hles.begin_trial(
                parent.hles_state, accepted_step_receipt=step,
                hles_input_receipt=hles_input, current_handoff=handoff,
                accepted_face_normal_velocity_m_s=resident[0],
                active_face=resident[1], closure_depth_m=resident[2],
                closure_chezy_mhalf_s=resident[3], active_cell=resident[4],
                cancelled=poll())
            chain = _hash({
                "previous": parent.accepted_chain_sha256,
                "hydraulic": evidence.evidence_sha256,
                "active": active_trial.trial_sha256,
                "hles": hles_trial.trial_sha256})
            candidate = _wrap_state(
                self.execution, active_state=active_trial.candidate,
                hles_state=hles_trial.candidate,
                accepted_chain_sha256=chain)
            placeholder = CoupledWetDryGPUTrialV97(
                self.execution.fingerprint, parent.state_sha256, handoff,
                hydraulic, evidence, active_trial, hles_trial,
                hles_inputs, candidate, BACKEND, "")
            trial = CoupledWetDryGPUTrialV97(
                **{**placeholder.__dict__,
                   "trial_sha256": _trial_token(placeholder)})
            self._validate_trial(parent, trial)
            if poll():
                raise InterruptedError(
                    "v97 CUDA outer cancelled before publication")
        except Exception:
            if hles_trial is not None and self.hles._open_trial is not None:
                self.hles.rollback(parent.hles_state, hles_trial)
            if active_trial is not None and self.active._control_parent is not None:
                self.active.rollback(parent.active_state, active_trial)
            self.hydraulic.invalidate_v98_hydraulic_side_evidence()
            raise
        self._parent, self._trial = parent, trial
        return trial

    def _validate_trial(self, parent, trial):
        validate_coupled_wetdry_gpu_state_v97(
            parent, self.execution, self.active.q3_provider)
        if (type(trial) is not CoupledWetDryGPUTrialV97 or
                trial.execution_fingerprint != self.execution.fingerprint or
                trial.parent_state_sha256 != parent.state_sha256 or
                trial.backend != BACKEND or
                trial.trial_sha256 != _trial_token(trial)):
            raise CoupledWetDryGPUV97Error(
                "v97 CUDA outer trial is stale, mutated, or cross-parent")
        validate_wetdry_fwave_gpu_trial_v97(
            trial.hydraulic_trial, self.execution.hydraulic_execution)
        v92_trial = trial.active_trial.base_v93_trial.base_v92_trial
        validate_wetdry_hydraulic_evidence_gpu_v97(
            trial.hydraulic_evidence,
            _v91_execution(self.execution.active_execution),
            _v92_state(parent.active_state), v92_trial.cap_receipt,
            midpoint_q=v92_trial.midpoint_q,
            accepted_q=v92_trial.candidate.q,
            momentum_rate_m2_s2=v92_trial.momentum_rate_m2_s2,
            momentum_face_flux_m4_s2=v92_trial.momentum_face_flux_m4_s2)
        self.active._validate_resident_control_trial(
            parent.active_state, trial.active_trial)
        if (not np.array_equal(v92_trial.candidate.q,
                               trial.hydraulic_trial.endpoint_q) or
                not np.array_equal(
                    v92_trial.hydraulic_face_receipt.
                        accepted_water_flux_m2_s,
                    trial.hydraulic_trial.
                        accepted_limited_face_water_flux_m2_s) or
                trial.candidate.active_state is not
                    trial.active_trial.candidate or
                trial.candidate.hles_state is not
                    trial.hles_trial.candidate or
                trial.candidate.accepted_updates !=
                    parent.accepted_updates+1 or
                trial.candidate.time_s !=
                    parent.time_s+trial.hydraulic_trial.dt_s):
            raise CoupledWetDryGPUV97Error(
                "v97 CUDA outer accepted child image differs")
        expected = _wrap_state(
            self.execution, active_state=trial.active_trial.candidate,
            hles_state=trial.hles_trial.candidate,
            accepted_chain_sha256=_hash({
                "previous": parent.accepted_chain_sha256,
                "hydraulic": trial.hydraulic_evidence.evidence_sha256,
                "active": trial.active_trial.trial_sha256,
                "hles": trial.hles_trial.trial_sha256}))
        if expected.state_sha256 != trial.candidate.state_sha256:
            raise CoupledWetDryGPUV97Error(
                "v97 CUDA outer candidate chain differs")

    def commit(self, parent, trial):
        if parent is not self._parent or trial is not self._trial:
            raise CoupledWetDryGPUV97Error(
                "v97 CUDA outer trial is not open")
        try:
            self._validate_trial(parent, trial)
            hles = self.hles.commit(parent.hles_state, trial.hles_trial)
            if hles is not trial.hles_trial.candidate:
                raise CoupledWetDryGPUV97Error(
                    "v97 CUDA HLES commit substituted candidate")
            active = self.active.commit(parent.active_state, trial.active_trial)
            if active is not trial.active_trial.candidate:
                raise CoupledWetDryGPUV97Error(
                    "v97 CUDA active commit substituted candidate")
            accepted = trial.candidate
            self.last_accepted_receipt = {
                "scope_id": SCOPE_ID+":accepted",
                "trial_sha256": trial.trial_sha256,
                "accepted_state_sha256": accepted.state_sha256,
                "actual_cuda_kernels": True,
                "automatic_cpu_fallback": False,
            }
            self._parent = self._trial = None
            return accepted
        except Exception:
            self.restore_failed_publication(parent, trial)
            raise
        finally:
            self.hydraulic.invalidate_v98_hydraulic_side_evidence()

    def restore_failed_publication(self, parent, trial):
        """Restore the exact accepted point after any partial child commit."""
        checkpoint = self._parent_checkpoint
        if checkpoint is None or checkpoint['state_sha256'] != parent.state_sha256:
            raise CoupledWetDryGPUV97Error('v100 atomic recovery parent checkpoint differs')
        if self.hles._open_trial is not None:
            self.hles.rollback(parent.hles_state, trial.hles_trial)
        self.active.abort_hydraulic_trial()
        self.active._clear_control()
        self._parent = self._trial = None
        restored = self.restore_checkpoint(checkpoint)
        if restored.state_sha256 != parent.state_sha256:
            raise CoupledWetDryGPUV97Error('v100 atomic recovery changed the accepted point')
        return restored

    def rollback(self, parent, trial):
        if parent is not self._parent or trial is not self._trial:
            raise CoupledWetDryGPUV97Error(
                "v97 CUDA outer trial is not open")
        try:
            self._validate_trial(parent, trial)
            self.hles.rollback(parent.hles_state, trial.hles_trial)
            self.active.rollback(parent.active_state, trial.active_trial)
            self._parent = self._trial = None
            return parent
        finally:
            self.hydraulic.invalidate_v98_hydraulic_side_evidence()

    def cancel(self, parent, trial):
        return self.rollback(parent, trial)

    def checkpoint(self, state):
        if self._parent is not None:
            raise CoupledWetDryGPUV97Error(
                "v97 CUDA outer checkpoint requires committed point")
        validate_coupled_wetdry_gpu_state_v97(
            state, self.execution, self.active.q3_provider)
        body = {
            "scope_id": CHECKPOINT_SCOPE_ID,
            "execution_fingerprint": self.execution.fingerprint,
            "active_checkpoint": self.active.checkpoint(state.active_state),
            "hles_checkpoint": self.hles.checkpoint(state.hles_state),
            "bed_elevation_m": state.bed_elevation_m.tolist(),
            "free_surface_elevation_m":
                state.free_surface_elevation_m.tolist(),
            "accepted_chain_sha256": state.accepted_chain_sha256,
            "state_sha256": state.state_sha256,
        }
        return {**body, "fingerprint": _hash(body)}

    def restore_checkpoint(self, document):
        self.hydraulic.invalidate_v98_hydraulic_side_evidence()
        if self._parent is not None or not isinstance(document, Mapping):
            raise CoupledWetDryGPUV97Error(
                "v97 CUDA outer restart differs")
        body = {key: value for key, value in document.items()
                if key != "fingerprint"}
        if (set(document) != set(body) | {"fingerprint"} or
                document.get("fingerprint") != _hash(body) or
                document.get("scope_id") != CHECKPOINT_SCOPE_ID or
                document.get("execution_fingerprint") !=
                    self.execution.fingerprint):
            raise CoupledWetDryGPUV97Error(
                "v97 CUDA outer checkpoint identity differs")
        provider_checkpoint = self.active.q3_provider.checkpoint()
        active_before = None
        try:
            active = self.active.restore_checkpoint(
                document["active_checkpoint"])
            active_before = active
            hles = self.hles.restore_checkpoint(document["hles_checkpoint"])
            state = _wrap_state(
                self.execution, active_state=active, hles_state=hles,
                accepted_chain_sha256=document["accepted_chain_sha256"])
            if (state.state_sha256 != document["state_sha256"] or
                    not np.array_equal(state.bed_elevation_m,
                                       document["bed_elevation_m"]) or
                    not np.array_equal(state.free_surface_elevation_m,
                                       document["free_surface_elevation_m"])):
                raise CoupledWetDryGPUV97Error(
                    "v97 CUDA restored outer state differs")
            self.last_accepted_receipt = None
            return state
        except Exception:
            # Active restore is already failure-atomic internally.  If HLES
            # or outer validation fails after it succeeds, restore the exact
            # provider checkpoint; callers retry using the original owner.
            self.active.q3_provider.restore_checkpoint(provider_checkpoint)
            raise


__all__ = [
    "BACKEND", "SCOPE_ID", "CoupledWetDryGPUExecutionV97",
    "CoupledWetDryGPUStateV97", "CoupledWetDryGPUTrialV97",
    "CoupledWetDryGPUV97Error", "ResidentCoupledWetDryLifecycleV97",
    "execution_document_coupled_wetdry_gpu_v97",
    "resolve_coupled_wetdry_gpu_execution_v97",
    "resolve_initial_coupled_wetdry_gpu_state_v97",
    "validate_coupled_wetdry_gpu_state_v97",
]
