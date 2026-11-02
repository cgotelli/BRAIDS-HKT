"""Actual-CUDA Q3/v70 noncohesive bed-vector boundary owner."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Mapping

import numpy as np

from braids_core.sediment.bedload_boundary_live_v70 import (
    MAX_ACCEPTED_STEPS,
    SCOPE_ID,
    BedloadBoundaryCandidateV70,
    BedloadBoundaryStateV70,
    BedloadBoundaryTopologyV70,
    BedloadBoundaryV70Error,
    _array_digest,
    _canonical,
    _fingerprint,
    _finite,
    bedload_boundary_checkpoint_v70,
    load_bedload_boundary_checkpoint_v70,
    load_bedload_boundary_execution_v70,
    resolve_bedload_boundary_state_v70,
    resolve_bedload_boundary_topology_v70,
)


CUDA_OPTIONS = (
    "--std=c++11", "--fmad=false", "--prec-div=true",
    "--prec-sqrt=true", "--ftz=false",
)
KERNEL_NAMES = (
    "bed_boundary_v70_candidate",
    "bed_boundary_v70_demand_fill",
    "bed_boundary_v70_apply",
    "bed_boundary_v70_cell_delta",
    "bed_boundary_v70_ledger",
)


def _host_digest(cp, *values) -> str:
    return _array_digest(*(cp.asnumpy(value) for value in values))


@dataclass(frozen=True)
class ResidentBedloadBoundaryTopologyV70:
    core: BedloadBoundaryTopologyV70
    owner: object
    boundary_index: object
    mode: object
    normal: object
    face_length_m: object
    cell_area_m2: object
    prescribed_mass_kg_m_s: object
    device_id: int

    @property
    def ncell(self) -> int:
        return self.core.ncell

    @property
    def nface(self) -> int:
        return self.core.nface

    @property
    def topology_sha256(self) -> str:
        return self.core.topology_sha256


@dataclass(frozen=True)
class ResidentBedloadBoundaryCandidateV70:
    execution_fingerprint: str
    topology_sha256: str
    potential_kg_m_s: object
    requires_owner_availability: object
    source: object
    candidate_sha256: str
    device_id: int


@dataclass(frozen=True)
class ResidentBedloadBoundaryStateV70:
    execution_fingerprint: str
    bed_mass_kg_m2: object
    active_fraction: object
    boundary_signed_kg: object
    boundary_absolute_kg: object
    accepted_step: int
    time_s: float
    state_sha256: str
    device_id: int


@dataclass(frozen=True)
class ResidentBedloadBoundaryTrialV70:
    execution_fingerprint: str
    parent_state_sha256: str
    topology_sha256: str
    dt_s: float
    morphological_factor: float
    candidate_flux: ResidentBedloadBoundaryCandidateV70
    provisional_flux_kg_m_s: object
    outgoing_demand_kg_m2: object
    inventory_fill: object
    accepted_face_availability: object
    face_flux_kg_m_s: object
    cell_delta_kg_m2: object
    boundary_transfer_kg: object
    boundary_absolute_transfer_kg: object
    conservation_residual_kg: object
    candidate: ResidentBedloadBoundaryStateV70
    active_fraction_application_count: int
    inventory_fill_application_count: int
    trial_sha256: str
    device_id: int


class ResidentBedloadBoundaryV70:
    """Run the authenticated boundary transaction on the selected CUDA GPU."""

    backend = "implicit_fwave_midpoint_gpu"
    integrator = "implicit_fwave_midpoint"

    def __init__(self, execution: Mapping[str, object]):
        import cupy as cp

        self.cp = cp
        self.device_id = int(cp.cuda.Device().id)
        self.execution = load_bedload_boundary_execution_v70(execution)
        if (self.execution["backend"] != self.backend or
                self.execution["integrator"] != self.integrator or
                self.execution["automatic_backend_fallback"] is not False):
            raise BedloadBoundaryV70Error(
                "Q3/v70 CUDA owner requires explicit implicit_fwave_midpoint_gpu")
        self.execution_fingerprint = self.execution["fingerprint"]
        self.nclass = len(self.execution["fraction_ids"])
        self.nboundary = len(self.execution["boundaries"])
        source_path = Path(__file__).with_suffix(".cu")
        if not source_path.is_file():
            raise BedloadBoundaryV70Error("Q3/v70 CUDA source is missing")
        self.cuda_source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
        with cp.cuda.Device(self.device_id):
            module = cp.RawModule(
                code=source_path.read_text(encoding="utf-8"),
                options=CUDA_OPTIONS,
                name_expressions=KERNEL_NAMES,
            )
            self.kernels = {name: module.get_function(name)
                            for name in KERNEL_NAMES}

    def _array(self, value, shape, name, *, dtype=None):
        cp = self.cp
        result = cp.ascontiguousarray(cp.asarray(
            value, dtype=cp.float64 if dtype is None else dtype))
        if result.shape != shape:
            raise BedloadBoundaryV70Error(
                f"Q3/v70 CUDA {name} must have shape {shape}")
        return result

    def _all(self, value) -> bool:
        return bool(self.cp.asnumpy(value).item())

    def prepare_topology(self, **values) -> ResidentBedloadBoundaryTopologyV70:
        cp = self.cp
        core = resolve_bedload_boundary_topology_v70(self.execution, **values)
        with cp.cuda.Device(self.device_id):
            return ResidentBedloadBoundaryTopologyV70(
                core,
                cp.ascontiguousarray(cp.asarray(core.owner, dtype=cp.int32)),
                cp.ascontiguousarray(cp.asarray(core.boundary_index, dtype=cp.int32)),
                cp.ascontiguousarray(cp.asarray(core.mode, dtype=cp.int32)),
                cp.ascontiguousarray(cp.asarray(core.normal, dtype=cp.float64)),
                cp.ascontiguousarray(cp.asarray(core.face_length_m, dtype=cp.float64)),
                cp.ascontiguousarray(cp.asarray(core.cell_area_m2, dtype=cp.float64)),
                cp.ascontiguousarray(cp.asarray(
                    core.prescribed_mass_kg_m_s, dtype=cp.float64)),
                self.device_id,
            )

    def _validate_topology(self, topology: ResidentBedloadBoundaryTopologyV70):
        if (not isinstance(topology, ResidentBedloadBoundaryTopologyV70) or
                topology.core.execution_fingerprint != self.execution_fingerprint or
                topology.core.dependency_fingerprint !=
                    self.execution["topology_fingerprint"] or
                topology.device_id != self.device_id):
            raise BedloadBoundaryV70Error("Q3/v70 CUDA topology receipt differs")
        expected = resolve_bedload_boundary_topology_v70(
            self.execution, owner=self.cp.asnumpy(topology.owner),
            boundary_index=self.cp.asnumpy(topology.boundary_index),
            normal=self.cp.asnumpy(topology.normal),
            face_length_m=self.cp.asnumpy(topology.face_length_m),
            cell_area_m2=self.cp.asnumpy(topology.cell_area_m2))
        if (expected.topology_sha256 != topology.topology_sha256 or
                not self._all(self.cp.all(
                    topology.mode == self.cp.asarray(expected.mode))) or
                not self._all(self.cp.all(
                    topology.prescribed_mass_kg_m_s == self.cp.asarray(
                        expected.prescribed_mass_kg_m_s)))):
            raise BedloadBoundaryV70Error("Q3/v70 CUDA topology arrays are stale")

    def resolve_state(self, *, bed_mass_kg_m2, active_fraction,
                      boundary_signed_kg=None, boundary_absolute_kg=None,
                      accepted_step=0, time_s=0.0) -> ResidentBedloadBoundaryStateV70:
        cp = self.cp
        mass_raw = cp.asarray(bed_mass_kg_m2)
        if mass_raw.ndim != 2 or mass_raw.shape[0] != self.nclass:
            raise BedloadBoundaryV70Error(
                "Q3/v70 CUDA bed mass must have shape (nfraction,ncell)")
        ncell = int(mass_raw.shape[1])
        mass = self._array(mass_raw, (self.nclass, ncell), "bed mass")
        fraction = self._array(active_fraction, (self.nclass, ncell),
                               "active fraction")
        signed = self._array(
            cp.zeros((self.nclass, self.nboundary), dtype=cp.float64)
            if boundary_signed_kg is None else boundary_signed_kg,
            (self.nclass, self.nboundary), "signed ledger")
        absolute = self._array(
            cp.zeros((self.nclass, self.nboundary), dtype=cp.float64)
            if boundary_absolute_kg is None else boundary_absolute_kg,
            (self.nclass, self.nboundary), "absolute ledger")
        if (not self._all(cp.all(cp.isfinite(mass) & (mass >= 0.0))) or
                not self._all(cp.all(cp.isfinite(fraction) &
                                     (fraction >= 0.0) & (fraction <= 1.0))) or
                not self._all(cp.all(cp.isfinite(signed))) or
                not self._all(cp.all(cp.isfinite(absolute) & (absolute >= 0.0)))):
            raise BedloadBoundaryV70Error("Q3/v70 CUDA state values differ")
        if (isinstance(accepted_step, bool) or not isinstance(accepted_step, int)
                or accepted_step < 0 or accepted_step > MAX_ACCEPTED_STEPS):
            raise BedloadBoundaryV70Error("Q3/v70 CUDA accepted step differs")
        time = _finite(time_s, "CUDA state time", nonnegative=True)
        bulk = cp.asarray(self.execution["bulk_density_kg_m3"],
                          dtype=cp.float64)[:, None]
        volume = mass / bulk
        stocked = cp.sum(volume, axis=0) > 0.0
        if self._all(cp.any(stocked)) and not self._all(cp.all(
                cp.abs(cp.sum(fraction[:, stocked], axis=0) - 1.0) <= 1e-12)):
            raise BedloadBoundaryV70Error(
                "Q3/v70 CUDA active fractions do not sum to one")
        token = _fingerprint({
            "scope_id": SCOPE_ID,
            "execution_fingerprint": self.execution_fingerprint,
            "accepted_step": accepted_step,
            "time_s": time,
            "arrays_sha256": _host_digest(
                cp, mass, fraction, signed, absolute),
        })
        return ResidentBedloadBoundaryStateV70(
            self.execution_fingerprint, mass, fraction, signed, absolute,
            accepted_step, time, token, self.device_id)

    def _validate_state(self, state: ResidentBedloadBoundaryStateV70):
        if (not isinstance(state, ResidentBedloadBoundaryStateV70) or
                state.execution_fingerprint != self.execution_fingerprint or
                state.device_id != self.device_id):
            raise BedloadBoundaryV70Error("Q3/v70 CUDA state receipt differs")
        expected = self.resolve_state(
            bed_mass_kg_m2=state.bed_mass_kg_m2,
            active_fraction=state.active_fraction,
            boundary_signed_kg=state.boundary_signed_kg,
            boundary_absolute_kg=state.boundary_absolute_kg,
            accepted_step=state.accepted_step, time_s=state.time_s)
        if expected.state_sha256 != state.state_sha256:
            raise BedloadBoundaryV70Error("Q3/v70 CUDA state arrays are stale")

    def candidate(self, topology: ResidentBedloadBoundaryTopologyV70, *,
                  hydraulic_outward_flux, owner_bed_mass_vector,
                  trace_bed_mass_vector) -> ResidentBedloadBoundaryCandidateV70:
        cp = self.cp
        self._validate_topology(topology)
        water = self._array(hydraulic_outward_flux, (topology.nface,),
                            "hydraulic outward flux")
        owner_vector = self._array(
            owner_bed_mass_vector, (2, self.nclass, topology.ncell),
            "owner vector")
        trace_vector = self._array(
            trace_bed_mass_vector, (2, self.nclass, topology.nface),
            "trace vector")
        potential = cp.zeros((self.nclass, topology.nface), dtype=cp.float64)
        requires = cp.zeros((self.nclass, topology.nface), dtype=cp.int32)
        source = cp.zeros((self.nclass, topology.nface), dtype=cp.int32)
        status = cp.zeros(topology.nface, dtype=cp.int32)
        count = self.nclass * topology.nface
        if count:
            self.kernels["bed_boundary_v70_candidate"](
                ((count + 255) // 256,), (256,), (
                    np.int32(topology.ncell), np.int32(topology.nface),
                    np.int32(self.nclass), topology.owner, topology.mode,
                    topology.normal, water, owner_vector, trace_vector,
                    topology.prescribed_mass_kg_m_s, potential, requires,
                    source, status))
        if not self._all(cp.all(status == 0)):
            raise BedloadBoundaryV70Error("Q3/v70 CUDA candidate rejected")
        potential = cp.ascontiguousarray(potential)
        requires = cp.ascontiguousarray(requires)
        source = cp.ascontiguousarray(source)
        token = _fingerprint({
            "scope_id": SCOPE_ID,
            "execution_fingerprint": self.execution_fingerprint,
            "topology_sha256": topology.topology_sha256,
            "arrays_sha256": _host_digest(cp, potential, requires, source),
        })
        return ResidentBedloadBoundaryCandidateV70(
            self.execution_fingerprint, topology.topology_sha256, potential,
            requires, source, token, self.device_id)

    def begin_trial(self, *, state: ResidentBedloadBoundaryStateV70,
                    topology: ResidentBedloadBoundaryTopologyV70,
                    hydraulic_outward_flux, owner_bed_mass_vector,
                    trace_bed_mass_vector, dt_s: float,
                    morphological_factor: float = 1.0,
                    cancelled=lambda: False) -> ResidentBedloadBoundaryTrialV70:
        cp = self.cp
        if not callable(cancelled):
            raise TypeError("cancelled must be callable")
        if cancelled():
            raise InterruptedError("Q3/v70 CUDA transaction cancelled before launch")
        self._validate_state(state)
        self._validate_topology(topology)
        if state.accepted_step >= MAX_ACCEPTED_STEPS:
            raise BedloadBoundaryV70Error("Q3/v70 CUDA accepted-step limit reached")
        if state.bed_mass_kg_m2.shape[1] != topology.ncell:
            raise BedloadBoundaryV70Error("Q3/v70 CUDA state/topology differs")
        dt = _finite(dt_s, "CUDA trial dt", positive=True)
        morfac = _finite(morphological_factor, "CUDA morphological factor",
                         nonnegative=True)
        candidate_flux = self.candidate(
            topology, hydraulic_outward_flux=hydraulic_outward_flux,
            owner_bed_mass_vector=owner_bed_mass_vector,
            trace_bed_mass_vector=trace_bed_mass_vector)
        owner_fraction = state.active_fraction[:, topology.owner]
        provisional = cp.ascontiguousarray(cp.where(
            candidate_flux.requires_owner_availability != 0,
            candidate_flux.potential_kg_m_s * owner_fraction,
            candidate_flux.potential_kg_m_s))
        demand = cp.zeros_like(state.bed_mass_kg_m2)
        fill = cp.zeros_like(state.bed_mass_kg_m2)
        availability = cp.zeros_like(state.bed_mass_kg_m2)
        cell_status = cp.zeros(topology.ncell, dtype=cp.int32)
        cell_count = self.nclass * topology.ncell
        if cell_count:
            self.kernels["bed_boundary_v70_demand_fill"](
                ((cell_count + 255) // 256,), (256,), (
                    np.int32(topology.ncell), np.int32(topology.nface),
                    np.int32(self.nclass), topology.owner,
                    candidate_flux.requires_owner_availability,
                    candidate_flux.potential_kg_m_s, topology.face_length_m,
                    topology.cell_area_m2, state.bed_mass_kg_m2,
                    state.active_fraction, np.float64(dt), np.float64(morfac),
                    demand, fill, availability, cell_status))
        if not self._all(cp.all(cell_status == 0)):
            raise BedloadBoundaryV70Error("Q3/v70 CUDA inventory projection rejected")
        flux = cp.zeros_like(candidate_flux.potential_kg_m_s)
        face_status = cp.zeros(topology.nface, dtype=cp.int32)
        face_count = self.nclass * topology.nface
        if face_count:
            self.kernels["bed_boundary_v70_apply"](
                ((face_count + 255) // 256,), (256,), (
                    np.int32(topology.ncell), np.int32(topology.nface),
                    np.int32(self.nclass), topology.owner,
                    candidate_flux.requires_owner_availability,
                    candidate_flux.potential_kg_m_s, state.active_fraction,
                    fill, flux, face_status))
        if not self._all(cp.all(face_status == 0)):
            raise BedloadBoundaryV70Error("Q3/v70 CUDA face application rejected")
        new_mass = cp.empty_like(state.bed_mass_kg_m2)
        delta = cp.empty_like(state.bed_mass_kg_m2)
        exner_status = cp.zeros(topology.ncell, dtype=cp.int32)
        if cell_count:
            self.kernels["bed_boundary_v70_cell_delta"](
                ((cell_count + 255) // 256,), (256,), (
                    np.int32(topology.ncell), np.int32(topology.nface),
                    np.int32(self.nclass), topology.owner,
                    topology.face_length_m, topology.cell_area_m2, flux,
                    state.bed_mass_kg_m2, np.float64(dt), np.float64(morfac),
                    new_mass, delta, exner_status))
        if not self._all(cp.all(exner_status == 0)):
            raise BedloadBoundaryV70Error("Q3/v70 CUDA Exner candidate rejected")
        transfer = cp.zeros((self.nclass, self.nboundary), dtype=cp.float64)
        absolute = cp.zeros_like(transfer)
        ledger_status = cp.zeros(self.nboundary, dtype=cp.int32)
        ledger_count = self.nclass * self.nboundary
        if ledger_count:
            self.kernels["bed_boundary_v70_ledger"](
                ((ledger_count + 255) // 256,), (256,), (
                    np.int32(topology.nface), np.int32(self.nclass),
                    np.int32(self.nboundary), topology.boundary_index,
                    topology.face_length_m, flux, np.float64(dt),
                    np.float64(morfac), transfer, absolute, ledger_status))
        if not self._all(cp.all(ledger_status == 0)):
            raise BedloadBoundaryV70Error("Q3/v70 CUDA ledger rejected")
        residual = cp.ascontiguousarray(
            cp.sum(delta * topology.cell_area_m2[None, :], axis=1) +
            cp.sum(transfer, axis=1))
        tolerance = 4096.0 * np.finfo(np.float64).eps * cp.maximum(
            1.0,
            cp.sum(cp.abs(delta * topology.cell_area_m2[None, :]), axis=1) +
            cp.sum(absolute, axis=1))
        if not self._all(cp.all(cp.abs(residual) <= tolerance)):
            raise BedloadBoundaryV70Error("Q3/v70 CUDA conservation ledger differs")
        bulk = cp.asarray(self.execution["bulk_density_kg_m3"],
                          dtype=cp.float64)[:, None]
        volume = new_mass / bulk
        total = cp.sum(volume, axis=0, keepdims=True)
        fraction = cp.ascontiguousarray(cp.where(
            total > 0.0, volume / cp.where(total > 0.0, total, 1.0),
            state.active_fraction))
        next_state = self.resolve_state(
            bed_mass_kg_m2=cp.ascontiguousarray(new_mass),
            active_fraction=fraction,
            boundary_signed_kg=cp.ascontiguousarray(
                state.boundary_signed_kg + transfer),
            boundary_absolute_kg=cp.ascontiguousarray(
                state.boundary_absolute_kg + absolute),
            accepted_step=state.accepted_step + 1, time_s=state.time_s + dt)
        arrays = tuple(cp.ascontiguousarray(value) for value in (
            provisional, demand, fill, availability, flux, delta, transfer,
            absolute, residual))
        token = _fingerprint({
            "scope_id": SCOPE_ID,
            "execution_fingerprint": self.execution_fingerprint,
            "parent_state_sha256": state.state_sha256,
            "topology_sha256": topology.topology_sha256,
            "candidate_flux_sha256": candidate_flux.candidate_sha256,
            "candidate_state_sha256": next_state.state_sha256,
            "dt_s": dt,
            "morphological_factor": morfac,
            "active_fraction_application_count": 1,
            "inventory_fill_application_count": 1,
            "arrays_sha256": _host_digest(cp, *arrays),
        })
        trial = ResidentBedloadBoundaryTrialV70(
            self.execution_fingerprint, state.state_sha256,
            topology.topology_sha256, dt, morfac, candidate_flux,
            *arrays, next_state, 1, 1, token, self.device_id)
        if cancelled():
            raise InterruptedError(
                "Q3/v70 CUDA transaction cancelled before publication")
        return trial

    def _validate_trial(self, state, trial):
        self._validate_state(state)
        if (not isinstance(trial, ResidentBedloadBoundaryTrialV70) or
                trial.execution_fingerprint != self.execution_fingerprint or
                trial.parent_state_sha256 != state.state_sha256 or
                trial.device_id != self.device_id or
                trial.active_fraction_application_count != 1 or
                trial.inventory_fill_application_count != 1):
            raise BedloadBoundaryV70Error("Q3/v70 CUDA trial receipt differs")
        candidate_token = _fingerprint({
            "scope_id": SCOPE_ID,
            "execution_fingerprint": self.execution_fingerprint,
            "topology_sha256": trial.topology_sha256,
            "arrays_sha256": _host_digest(
                self.cp, trial.candidate_flux.potential_kg_m_s,
                trial.candidate_flux.requires_owner_availability,
                trial.candidate_flux.source),
        })
        if candidate_token != trial.candidate_flux.candidate_sha256:
            raise BedloadBoundaryV70Error(
                "Q3/v70 CUDA candidate arrays are stale")
        self._validate_state(trial.candidate)
        arrays = (
            trial.provisional_flux_kg_m_s, trial.outgoing_demand_kg_m2,
            trial.inventory_fill, trial.accepted_face_availability,
            trial.face_flux_kg_m_s, trial.cell_delta_kg_m2,
            trial.boundary_transfer_kg,
            trial.boundary_absolute_transfer_kg,
            trial.conservation_residual_kg)
        token = _fingerprint({
            "scope_id": SCOPE_ID,
            "execution_fingerprint": self.execution_fingerprint,
            "parent_state_sha256": state.state_sha256,
            "topology_sha256": trial.topology_sha256,
            "candidate_flux_sha256": trial.candidate_flux.candidate_sha256,
            "candidate_state_sha256": trial.candidate.state_sha256,
            "dt_s": trial.dt_s,
            "morphological_factor": trial.morphological_factor,
            "active_fraction_application_count": 1,
            "inventory_fill_application_count": 1,
            "arrays_sha256": _host_digest(self.cp, *arrays),
        })
        if token != trial.trial_sha256:
            raise BedloadBoundaryV70Error("Q3/v70 CUDA trial arrays are stale")

    def commit(self, state, trial) -> ResidentBedloadBoundaryStateV70:
        self._validate_trial(state, trial)
        return trial.candidate

    def rollback(self, state, trial) -> ResidentBedloadBoundaryStateV70:
        self._validate_trial(state, trial)
        return state

    def cancel(self, state, trial) -> ResidentBedloadBoundaryStateV70:
        return self.rollback(state, trial)

    def checkpoint(self, topology, state) -> dict[str, object]:
        self._validate_topology(topology)
        self._validate_state(state)
        core_state = resolve_bedload_boundary_state_v70(
            self.execution, bed_mass_kg_m2=self.cp.asnumpy(state.bed_mass_kg_m2),
            active_fraction=self.cp.asnumpy(state.active_fraction),
            boundary_signed_kg=self.cp.asnumpy(state.boundary_signed_kg),
            boundary_absolute_kg=self.cp.asnumpy(state.boundary_absolute_kg),
            accepted_step=state.accepted_step, time_s=state.time_s)
        return bedload_boundary_checkpoint_v70(
            self.execution, topology.core, core_state)

    def restart(self, topology, checkpoint) -> ResidentBedloadBoundaryStateV70:
        self._validate_topology(topology)
        core = load_bedload_boundary_checkpoint_v70(
            self.execution, topology.core, checkpoint)
        return self.resolve_state(
            bed_mass_kg_m2=core.bed_mass_kg_m2,
            active_fraction=core.active_fraction,
            boundary_signed_kg=core.boundary_signed_kg,
            boundary_absolute_kg=core.boundary_absolute_kg,
            accepted_step=core.accepted_step, time_s=core.time_s)


__all__ = [
    "ResidentBedloadBoundaryCandidateV70", "ResidentBedloadBoundaryStateV70",
    "ResidentBedloadBoundaryTopologyV70", "ResidentBedloadBoundaryTrialV70",
    "ResidentBedloadBoundaryV70",
]
