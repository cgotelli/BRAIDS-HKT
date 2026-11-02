"""Resident Q3-H6a lifecycle for FLOW direct-vector laws 4 and 13.

The backend composes the exact constitutive bed/non-bed vectors before the
existing common face/Exner consumer.  It never accepts a direction override or
falls back to CPU.  Accepted bed state, candidates, finite-inventory factors
and ledgers remain on one CUDA device; only explicit checkpoints are portable.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np

from braids_core.sediment.capacity_parameters import formula_contract
from braids_core.sediment.channel_composition import TRA_BEDLOAD, TRA_COMBINE
from braids_core.sediment.direct_vector_live_v66 import (
    DirectVectorLiveV66Error,
    load_direct_vector_live_execution_v66,
    load_state_checkpoint_v66 as load_portable_checkpoint_v66,
    resolve_state_v66 as resolve_portable_state_v66,
    state_checkpoint_document_v66 as portable_checkpoint_document_v66,
    verify_direct_vector_sources_v66,
)
from .scalar_capacity_runtime import (
    FACE_CLOSED,
    FACE_INTERNAL,
    FACE_OWNER_OUTLET,
    FACE_TRACE_INLET,
    FACE_TRACE_OUTLET,
    ResidentScalarCapacityRuntime,
    ResidentScalarFaces,
    ResidentScalarPoints,
)


BACKEND = "implicit_fwave_midpoint_gpu"
INTEGRATOR = "implicit_fwave_midpoint"
CUDA_OPTIONS = (
    "--std=c++11", "--fmad=false", "--prec-div=true",
    "--prec-sqrt=true", "--ftz=false",
)
KERNEL_NAMES = ("direct_v66_digest_fp64", "direct_v66_inventory_fill")


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def _fingerprint(value: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _finite(value: object, name: str, *, positive: bool = False,
            nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DirectVectorLiveV66Error(f"Q3-H6a/v66 GPU {name} must be numeric")
    result = float(value)
    if (not math.isfinite(result) or (positive and result <= 0.0) or
            (nonnegative and result < 0.0)):
        raise DirectVectorLiveV66Error(
            f"Q3-H6a/v66 GPU {name} is outside its finite domain")
    return 0.0 if result == 0.0 else result


@dataclass(frozen=True)
class ResidentDirectVectorChannelsV66:
    execution_fingerprint: str
    raw: ResidentScalarPoints
    composed: ResidentScalarPoints
    raw_bed_mass: object
    raw_suspended_mass: object
    retained_suspended_mass: object
    digest: str
    device_id: int


@dataclass(frozen=True)
class ResidentDirectVectorStateV66:
    execution_fingerprint: str
    bed_mass_kg_m2: object
    bulk_density_kg_m3: object
    active_fraction: object
    boundary_signed_kg: object
    boundary_absolute_kg: object
    accepted_step: int
    time_s: float
    state_sha256: str
    device_id: int


@dataclass(frozen=True)
class ResidentDirectVectorTrialV66:
    execution_fingerprint: str
    parent_state_sha256: str
    dt_s: float
    morphological_factor: float
    provisional_faces: ResidentScalarFaces
    faces: ResidentScalarFaces
    inventory_fill: object
    availability: object
    cell_delta_kg_m2: object
    boundary_transfer_kg: object
    boundary_absolute_transfer_kg: object
    conservation_residual_kg: object
    candidate: ResidentDirectVectorStateV66
    active_fraction_application_count: int
    inventory_fill_application_count: int
    trial_sha256: str
    device_id: int


class ResidentDirectVectorLiveV66:
    """One-device authenticated live transaction for laws 4 and 13."""

    backend = BACKEND
    integrator = INTEGRATOR

    def __init__(self, *, execution: Mapping[str, object],
                 source_path: str | Path | None = None,
                 verify_sources: bool = True):
        import cupy as cp

        self.cp = cp
        self.device_id = int(cp.cuda.Device().id)
        self.execution = load_direct_vector_live_execution_v66(execution)
        if (self.execution["backend"] != BACKEND or
                self.execution["integrator"] != INTEGRATOR or
                self.execution["automatic_backend_fallback"] is not False):
            raise DirectVectorLiveV66Error(
                "Q3-H6a/v66 GPU execution backend/integrator differs")
        if verify_sources:
            self.source_receipt = verify_direct_vector_sources_v66(self.execution)
        else:
            self.source_receipt = None
        physical = self.execution["physical_profile"]
        self.formula_code = int(self.execution["formula_code"])
        self.nclass = len(physical["fraction_ids"])
        self.transport_class = cp.ascontiguousarray(cp.asarray(
            physical["transport_classes"], dtype=cp.int32))
        self.bulk_density = cp.ascontiguousarray(cp.asarray(
            physical["bulk_density_kg_m3"], dtype=cp.float64))
        contract = formula_contract(self.execution["method_id"])
        parameter_rows = physical["parameters"]
        parameters = {}
        for specification in contract.parameters:
            values = [
                next(item["value"] for item in row["values"]
                     if item["keyword"] == specification.keyword)
                for row in parameter_rows
            ]
            dtype = cp.int32 if specification.value_kind == "integer" else cp.float64
            parameters[specification.keyword] = cp.ascontiguousarray(
                cp.asarray(values, dtype=dtype))
        roughness = physical["roughness_method_id"]
        roughness_mode = {
            "roughness.chezy": "chezy",
            "roughness.manning": "manning",
            "roughness.white_colebrook": "white_colebrook",
            "roughness.white_colebrook_waqua": "white_colebrook",
        }[roughness]
        self.capacity = ResidentScalarCapacityRuntime(
            self.formula_code,
            d50=cp.ascontiguousarray(cp.asarray(physical["d50_m"], dtype=cp.float64)),
            d90=cp.ascontiguousarray(cp.asarray(physical["d90_m"], dtype=cp.float64)),
            solid_density=cp.ascontiguousarray(cp.asarray(
                physical["solid_density_kg_m3"], dtype=cp.float64)),
            parameters=parameters,
            water_density=physical["water_density_kg_m3"],
            gravity=physical["gravity_m_s2"],
            roughness_mode=roughness_mode,
            minimum_work_depth=physical["minimum_work_depth_m"],
            waqua_white_colebrook=(roughness == "roughness.white_colebrook_waqua"),
        )
        if self.capacity.device_id != self.device_id:
            raise DirectVectorLiveV66Error(
                "Q3-H6a/v66 capacity runtime selected a different CUDA device")
        path = Path(source_path or Path(__file__).with_suffix(".cu"))
        source = path.read_text(encoding="utf-8")
        with cp.cuda.Device(self.device_id):
            module = cp.RawModule(
                code=source, options=CUDA_OPTIONS, name_expressions=KERNEL_NAMES)
            self.kernels = {name: module.get_function(name) for name in KERNEL_NAMES}

    def _array(self, name: str, value: object, ndim: int,
               shape: tuple[int, ...] | None = None, *, integer: bool = False,
               copy: bool = False):
        cp = self.cp
        dtype = cp.int32 if integer else cp.float64
        if (not isinstance(value, cp.ndarray) or value.dtype != dtype or
                value.ndim != ndim or not value.flags.c_contiguous):
            label = "int32" if integer else "FP64"
            raise TypeError(
                f"{name} must be a resident contiguous {ndim}-D {label} array")
        if int(value.device.id) != self.device_id:
            raise DirectVectorLiveV66Error(
                f"{name} is not resident on CUDA device {self.device_id}")
        if shape is not None and tuple(value.shape) != tuple(shape):
            raise DirectVectorLiveV66Error(
                f"{name} must have shape {shape}, got {tuple(value.shape)}")
        return value.copy(order="C") if copy else value

    @staticmethod
    def _flag(value: object) -> bool:
        return bool(value.item() if hasattr(value, "item") else value)

    def _all(self, value: object) -> bool:
        return self._flag(self.cp.all(value))

    def _digest(self, *arrays: object, label: str) -> str:
        cp = self.cp
        with cp.cuda.Device(self.device_id):
            digest = cp.zeros(2, dtype=cp.uint64)
            inventory = []
            for ordinal, raw in enumerate(arrays):
                if not isinstance(raw, cp.ndarray) or int(raw.device.id) != self.device_id:
                    raise DirectVectorLiveV66Error(
                        "Q3-H6a/v66 digest received a nonresident array")
                value = (raw if raw.dtype == cp.float64 else
                         cp.ascontiguousarray(raw.astype(cp.float64)))
                value = cp.ascontiguousarray(value.reshape(-1))
                if value.size:
                    self.kernels["direct_v66_digest_fp64"](
                        ((int(value.size)+255)//256,), (256,), (
                            np.int64(value.size),
                            np.uint64(0x9E3779B97F4A7C15 ^ (ordinal+1)),
                            value, digest))
                inventory.append({
                    "dtype": str(raw.dtype), "shape": list(raw.shape),
                    "ordinal": ordinal,
                })
            words = [int(digest[0].item()), int(digest[1].item())]
        return _fingerprint({
            "backend": BACKEND, "label": label, "inventory": inventory,
            "resident_digest_u64": words,
        })

    def _state_token(self, mass, density, fraction, signed, absolute,
                     step: int, time_s: float) -> str:
        return _fingerprint({
            "execution_fingerprint": self.execution["fingerprint"],
            "accepted_step": step, "time_s": time_s,
            "resident_arrays_sha256": self._digest(
                mass, density, fraction, signed, absolute,
                label="accepted_state"),
        })

    def _validate_state(self, state: ResidentDirectVectorStateV66) -> None:
        if (not isinstance(state, ResidentDirectVectorStateV66) or
                state.device_id != self.device_id or
                state.execution_fingerprint != self.execution["fingerprint"]):
            raise DirectVectorLiveV66Error(
                "Q3-H6a/v66 GPU state differs from execution/device")
        expected = self._state_token(
            state.bed_mass_kg_m2, state.bulk_density_kg_m3,
            state.active_fraction, state.boundary_signed_kg,
            state.boundary_absolute_kg, state.accepted_step, state.time_s)
        if expected != state.state_sha256:
            raise DirectVectorLiveV66Error(
                "Q3-H6a/v66 GPU state is stale or mutated")

    def resolve_state(self, *, bed_mass_kg_m2: object,
                      active_fraction: object | None = None,
                      boundary_signed_kg: object | None = None,
                      boundary_absolute_kg: object | None = None,
                      accepted_step: int = 0,
                      time_s: float = 0.0) -> ResidentDirectVectorStateV66:
        cp = self.cp
        mass = self._array("bed_mass_kg_m2", bed_mass_kg_m2, 2, copy=True)
        nfrac, ncell = map(int, mass.shape)
        if nfrac != self.nclass or ncell <= 0:
            raise DirectVectorLiveV66Error(
                "Q3-H6a/v66 GPU bed mass layout differs")
        if not self._all(cp.isfinite(mass) & (mass >= 0.0)):
            raise DirectVectorLiveV66Error(
                "Q3-H6a/v66 GPU bed mass must be finite nonnegative")
        density = self.bulk_density.copy()
        volume = mass/density[:, None]
        total = cp.sum(volume, axis=0, keepdims=True)
        if not self._all(cp.isfinite(total) & (total > 0.0)):
            raise DirectVectorLiveV66Error(
                "Q3-H6a/v66 GPU all-class depleted cell is outside H6a")
        derived = cp.ascontiguousarray(volume/total)
        if active_fraction is None:
            fraction = derived
        else:
            fraction = self._array(
                "active_fraction", active_fraction, 2, (nfrac, ncell), copy=True)
            tolerance = 512.0*np.finfo(np.float64).eps
            if (not self._all(cp.isfinite(fraction) & (fraction >= 0.0)) or
                    not self._flag(cp.allclose(
                        fraction, derived, rtol=tolerance, atol=tolerance))):
                raise DirectVectorLiveV66Error(
                    "Q3-H6a/v66 GPU active fraction differs from stock")
        signed = (cp.zeros(nfrac, dtype=cp.float64) if boundary_signed_kg is None
                  else self._array("boundary_signed_kg", boundary_signed_kg, 1,
                                   (nfrac,), copy=True))
        absolute = (cp.zeros(nfrac, dtype=cp.float64)
                    if boundary_absolute_kg is None else self._array(
                        "boundary_absolute_kg", boundary_absolute_kg, 1,
                        (nfrac,), copy=True))
        if (not self._all(cp.isfinite(signed)) or
                not self._all(cp.isfinite(absolute) & (absolute >= 0.0))):
            raise DirectVectorLiveV66Error(
                "Q3-H6a/v66 GPU boundary ledger is invalid")
        limit = int(self.execution["accepted_step_limit"])
        if (isinstance(accepted_step, bool) or type(accepted_step) is not int or
                accepted_step < 0 or accepted_step > limit):
            raise DirectVectorLiveV66Error(
                "Q3-H6a/v66 GPU accepted step is invalid")
        accepted_time = _finite(time_s, "accepted time", nonnegative=True)
        token = self._state_token(
            mass, density, fraction, signed, absolute,
            accepted_step, accepted_time)
        return ResidentDirectVectorStateV66(
            self.execution["fingerprint"], mass, density, fraction, signed,
            absolute, accepted_step, accepted_time, token, self.device_id)

    def _compose(self, points: ResidentScalarPoints) -> ResidentDirectVectorChannelsV66:
        cp = self.cp
        if points.law != self.formula_code:
            raise DirectVectorLiveV66Error(
                "Q3-H6a/v66 producer formula differs from execution")
        if self.formula_code == 4 and self._flag(cp.any(points.suspended_mass != 0.0)):
            raise DirectVectorLiveV66Error(
                "Q3-H6a/v66 Soulsby source-zero suspended channel changed")
        physical = self.execution["physical_profile"]
        raw_bed = points.bed_mass
        raw_suspended = points.suspended_mass
        bed = np.float64(physical["bed_factor"])*raw_bed
        nonbed = np.float64(physical["suspended_factor"])*raw_suspended
        is_bedload = self.transport_class == np.int32(TRA_BEDLOAD)
        is_combine = self.transport_class == np.int32(TRA_COMBINE)
        composed_bed = cp.ascontiguousarray(
            bed+cp.where(is_bedload[None, :, None], nonbed, 0.0))
        retained = cp.ascontiguousarray(
            cp.where(is_combine[None, :, None], nonbed, 0.0))
        composed = ResidentScalarPoints(
            points.law, points.result, composed_bed, retained,
            points.status, points.active)
        digest = self._digest(
            raw_bed, raw_suspended, composed_bed, retained, points.status,
            label="direct_vector_channels")
        return ResidentDirectVectorChannelsV66(
            self.execution["fingerprint"], points, composed, raw_bed,
            raw_suspended, retained, digest, self.device_id)

    def _validate_channels(self, value: ResidentDirectVectorChannelsV66,
                           npoint: int) -> None:
        if (not isinstance(value, ResidentDirectVectorChannelsV66) or
                value.device_id != self.device_id or
                value.execution_fingerprint != self.execution["fingerprint"] or
                value.composed.bed_mass.shape != (2, self.nclass, npoint)):
            raise DirectVectorLiveV66Error(
                "Q3-H6a/v66 GPU channel receipt differs")
        expected = self._digest(
            value.raw_bed_mass, value.raw_suspended_mass,
            value.composed.bed_mass, value.retained_suspended_mass,
            value.composed.status, label="direct_vector_channels")
        if expected != value.digest:
            raise DirectVectorLiveV66Error(
                "Q3-H6a/v66 GPU channel arrays are stale or mutated")

    def cells(self, **values) -> ResidentDirectVectorChannelsV66:
        """Produce and compose authenticated cell vectors; no direction input."""
        return self._compose(self.capacity.cells(**values))

    def traces(self, **values) -> ResidentDirectVectorChannelsV66:
        """Produce and compose explicit inlet/outlet trace vectors."""
        return self._compose(self.capacity.traces(**values))

    def begin_trial(
        self, *, state: ResidentDirectVectorStateV66,
        cells: ResidentDirectVectorChannelsV66,
        traces: ResidentDirectVectorChannelsV66,
        owner: object, neighbour: object, role: object, normal: object,
        face_length: object, cell_area: object, offset: object,
        adjacency_faces: object, signs: object, dt_s: float,
        morphological_factor: float = 1.0, cancelled=lambda: False,
    ) -> ResidentDirectVectorTrialV66:
        """Build a complete resident candidate without mutating ``state``."""
        cp = self.cp
        if not callable(cancelled):
            raise TypeError("cancelled must be callable")
        if cancelled():
            raise InterruptedError("Q3-H6a/v66 GPU transaction cancelled before launch")
        self._validate_state(state)
        if state.accepted_step >= int(self.execution["accepted_step_limit"]):
            raise DirectVectorLiveV66Error(
                "Q3-H6a/v66 bounded accepted-step limit reached")
        classes = self.execution["physical_profile"]["transport_classes"]
        if any(value == TRA_COMBINE for value in classes):
            raise DirectVectorLiveV66Error(
                "Q3-H6a/v66 TRA_COMBINE cannot commit before Q4 owns "
                "water-column mass and exchange")
        if any(value != TRA_BEDLOAD for value in classes):
            raise DirectVectorLiveV66Error(
                "Q3-H6a/v66 live bed transaction accepts class 1 only")
        nfrac, ncell = map(int, state.bed_mass_kg_m2.shape)
        self._validate_channels(cells, ncell)
        owner = self._array("owner", owner, 1, integer=True)
        nface = int(owner.size)
        self._validate_channels(traces, nface)
        neighbour = self._array("neighbour", neighbour, 1, (nface,), integer=True)
        role = self._array("role", role, 1, (nface,), integer=True)
        normal = self._array("normal", normal, 2, (2, nface))
        length = self._array("face_length", face_length, 1, (nface,))
        area = self._array("cell_area", cell_area, 1, (ncell,))
        offset = self._array("offset", offset, 1, (ncell+1,), integer=True)
        adjacency = self._array(
            "adjacency_faces", adjacency_faces, 1, integer=True)
        signs = self._array("signs", signs, 1, (int(adjacency.size),), integer=True)
        if (not self._all(cp.isfinite(length) & (length >= 0.0)) or
                not self._all(cp.isfinite(area) & (area > 0.0))):
            raise DirectVectorLiveV66Error(
                "Q3-H6a/v66 GPU face length/cell area is invalid")
        known_role = ((role == FACE_INTERNAL) | (role == FACE_TRACE_INLET) |
                      (role == FACE_TRACE_OUTLET) |
                      (role == FACE_OWNER_OUTLET) | (role == FACE_CLOSED))
        if (not self._all(known_role) or
                not self._all((role == FACE_INTERNAL) == (neighbour >= 0)) or
                not self._all((owner >= 0) & (owner < ncell)) or
                not self._all((neighbour >= -1) & (neighbour < ncell)) or
                not self._all(cp.isfinite(normal))):
            raise DirectVectorLiveV66Error(
                "Q3-H6a/v66 GPU face topology/role contract differs")
        if (int(offset[0].item()) != 0 or
                int(offset[-1].item()) != int(adjacency.size) or
                not self._all(offset[1:] >= offset[:-1]) or
                not self._all((adjacency >= 0) & (adjacency < nface)) or
                not self._all((signs == 1) | (signs == -1))):
            raise DirectVectorLiveV66Error(
                "Q3-H6a/v66 GPU adjacency inventory differs")
        cell_ids = cp.repeat(
            cp.arange(ncell, dtype=cp.int32), cp.diff(offset))
        adjacent_owner = owner[adjacency]
        adjacent_neighbour = neighbour[adjacency]
        incidence_ok = (((cell_ids == adjacent_owner) & (signs == 1)) |
                        ((cell_ids == adjacent_neighbour) & (signs == -1)))
        incidence_count = cp.bincount(adjacency, minlength=nface)
        expected_count = cp.where(role == FACE_INTERNAL, 2, 1)
        if (not self._all(incidence_ok) or
                not self._all(incidence_count == expected_count)):
            raise DirectVectorLiveV66Error(
                "Q3-H6a/v66 GPU adjacency ownership/sign contract differs")
        dt = _finite(dt_s, "trial dt", positive=True)
        morfac = _finite(
            morphological_factor, "morphological factor", nonnegative=True)

        provisional = self.capacity.faces(
            cells=cells.composed, traces=traces.composed,
            owner=owner, neighbour=neighbour, role=role, normal=normal,
            availability=state.active_fraction)
        if self._flag(cp.any(provisional.status != 0)):
            raise DirectVectorLiveV66Error(
                "Q3-H6a/v66 provisional face reconstruction rejected")
        fill = cp.empty_like(state.active_fraction)
        availability = cp.empty_like(state.active_fraction)
        inventory_status = cp.zeros(ncell, dtype=cp.int32)
        count = nfrac*ncell
        if count:
            self.kernels["direct_v66_inventory_fill"](
                ((count+255)//256,), (256,), (
                    np.int32(ncell), np.int32(nface), np.int32(adjacency.size),
                    np.int32(nfrac), offset, adjacency, owner, neighbour, role,
                    provisional.donor, provisional.flux, length, area,
                    state.bed_mass_kg_m2, state.active_fraction,
                    np.float64(dt), np.float64(morfac), fill, availability,
                    inventory_status))
        if self._flag(cp.any(inventory_status != 0)):
            raise DirectVectorLiveV66Error(
                "Q3-H6a/v66 finite-inventory projection rejected")
        faces = self.capacity.faces(
            cells=cells.composed, traces=traces.composed,
            owner=owner, neighbour=neighbour, role=role, normal=normal,
            availability=cp.ascontiguousarray(availability))
        if self._flag(cp.any(faces.status != 0)):
            raise DirectVectorLiveV66Error(
                "Q3-H6a/v66 final face reconstruction rejected")
        exner = self.capacity.exner_candidate(
            old_mass=state.bed_mass_kg_m2, faces=faces, offset=offset,
            adjacency_faces=adjacency, signs=signs, face_length=length,
            area=area, dt=dt, morphological_factor=morfac)
        if self._flag(cp.any(exner.status != 0)):
            raise DirectVectorLiveV66Error(
                "Q3-H6a/v66 conservative Exner candidate rejected")
        tolerance = 4096.0*np.finfo(np.float64).eps*cp.maximum(
            1.0, cp.sum(cp.abs(exner.delta*area[None, :]), axis=1)+
            cp.abs(exner.boundary_transfer))
        if self._flag(cp.any(cp.abs(exner.conservation_residual) > tolerance)):
            raise DirectVectorLiveV66Error(
                "Q3-H6a/v66 conservative Exner ledger does not close")
        boundary_mask = ((role == FACE_TRACE_INLET) |
                         (role == FACE_TRACE_OUTLET) |
                         (role == FACE_OWNER_OUTLET))
        boundary_absolute = cp.ascontiguousarray(
            np.float64(dt)*np.float64(morfac)*cp.sum(
                cp.abs(faces.flux)*length[None, :]*boundary_mask[None, :],
                axis=1))
        volume = exner.mass/state.bulk_density_kg_m3[:, None]
        if not self._all(cp.isfinite(exner.mass) & (exner.mass >= 0.0)) or \
                not self._all(cp.sum(volume, axis=0) > 0.0):
            raise DirectVectorLiveV66Error(
                "Q3-H6a/v66 candidate stock is invalid")
        candidate = self.resolve_state(
            bed_mass_kg_m2=cp.ascontiguousarray(exner.mass),
            boundary_signed_kg=cp.ascontiguousarray(
                state.boundary_signed_kg+exner.boundary_transfer),
            boundary_absolute_kg=cp.ascontiguousarray(
                state.boundary_absolute_kg+boundary_absolute),
            accepted_step=state.accepted_step+1, time_s=state.time_s+dt)
        arrays = (
            provisional.flux, provisional.donor, faces.flux, faces.donor,
            fill, availability, exner.delta, exner.boundary_transfer,
            boundary_absolute, exner.conservation_residual)
        token = _fingerprint({
            "execution_fingerprint": self.execution["fingerprint"],
            "parent_state_sha256": state.state_sha256,
            "candidate_state_sha256": candidate.state_sha256,
            "dt_s": dt, "morphological_factor": morfac,
            "resident_arrays_sha256": self._digest(
                *arrays, label="direct_vector_trial"),
            "active_fraction_application_count": 1,
            "inventory_fill_application_count": 1,
        })
        trial = ResidentDirectVectorTrialV66(
            self.execution["fingerprint"], state.state_sha256, dt, morfac,
            provisional, faces, cp.ascontiguousarray(fill),
            cp.ascontiguousarray(availability), exner.delta,
            exner.boundary_transfer, boundary_absolute,
            exner.conservation_residual, candidate, 1, 1, token,
            self.device_id)
        if cancelled():
            raise InterruptedError(
                "Q3-H6a/v66 GPU transaction cancelled before publication")
        return trial

    def _validate_trial(self, state: ResidentDirectVectorStateV66,
                        trial: ResidentDirectVectorTrialV66) -> None:
        self._validate_state(state)
        if (not isinstance(trial, ResidentDirectVectorTrialV66) or
                trial.device_id != self.device_id or
                trial.execution_fingerprint != state.execution_fingerprint or
                trial.parent_state_sha256 != state.state_sha256 or
                trial.active_fraction_application_count != 1 or
                trial.inventory_fill_application_count != 1):
            raise DirectVectorLiveV66Error(
                "Q3-H6a/v66 GPU trial parent or once-only counters differ")
        arrays = (
            trial.provisional_faces.flux, trial.provisional_faces.donor,
            trial.faces.flux, trial.faces.donor, trial.inventory_fill,
            trial.availability, trial.cell_delta_kg_m2,
            trial.boundary_transfer_kg, trial.boundary_absolute_transfer_kg,
            trial.conservation_residual_kg)
        token = _fingerprint({
            "execution_fingerprint": trial.execution_fingerprint,
            "parent_state_sha256": trial.parent_state_sha256,
            "candidate_state_sha256": trial.candidate.state_sha256,
            "dt_s": trial.dt_s,
            "morphological_factor": trial.morphological_factor,
            "resident_arrays_sha256": self._digest(
                *arrays, label="direct_vector_trial"),
            "active_fraction_application_count": 1,
            "inventory_fill_application_count": 1,
        })
        if token != trial.trial_sha256:
            raise DirectVectorLiveV66Error(
                "Q3-H6a/v66 GPU trial is stale or mutated")
        self._validate_state(trial.candidate)

    def commit_trial(self, state: ResidentDirectVectorStateV66,
                     trial: ResidentDirectVectorTrialV66
                     ) -> ResidentDirectVectorStateV66:
        self._validate_trial(state, trial)
        return trial.candidate

    def rollback_trial(self, state: ResidentDirectVectorStateV66,
                       trial: ResidentDirectVectorTrialV66
                       ) -> ResidentDirectVectorStateV66:
        self._validate_trial(state, trial)
        return state

    def cancel_trial(self, state: ResidentDirectVectorStateV66,
                     trial: ResidentDirectVectorTrialV66
                     ) -> ResidentDirectVectorStateV66:
        self._validate_trial(state, trial)
        return state

    def state_checkpoint_document(
        self, state: ResidentDirectVectorStateV66,
    ) -> dict[str, object]:
        """Cross the host boundary only for a complete accepted checkpoint."""
        self._validate_state(state)
        cp = self.cp
        portable = resolve_portable_state_v66(
            execution=self.execution,
            bed_mass_kg_m2=cp.asnumpy(state.bed_mass_kg_m2),
            active_fraction=cp.asnumpy(state.active_fraction),
            boundary_signed_kg=cp.asnumpy(state.boundary_signed_kg),
            boundary_absolute_kg=cp.asnumpy(state.boundary_absolute_kg),
            accepted_step=state.accepted_step, time_s=state.time_s)
        return portable_checkpoint_document_v66(portable)

    def load_state_checkpoint(
        self, value: Mapping[str, object],
    ) -> ResidentDirectVectorStateV66:
        portable = load_portable_checkpoint_v66(value, execution=self.execution)
        cp = self.cp
        with cp.cuda.Device(self.device_id):
            return self.resolve_state(
                bed_mass_kg_m2=cp.ascontiguousarray(cp.asarray(
                    portable.bed_mass_kg_m2, dtype=cp.float64)),
                active_fraction=cp.ascontiguousarray(cp.asarray(
                    portable.active_fraction, dtype=cp.float64)),
                boundary_signed_kg=cp.ascontiguousarray(cp.asarray(
                    portable.boundary_signed_kg, dtype=cp.float64)),
                boundary_absolute_kg=cp.ascontiguousarray(cp.asarray(
                    portable.boundary_absolute_kg, dtype=cp.float64)),
                accepted_step=portable.accepted_step,
                time_s=portable.time_s)


__all__ = [
    "BACKEND", "INTEGRATOR", "ResidentDirectVectorChannelsV66",
    "ResidentDirectVectorLiveV66", "ResidentDirectVectorStateV66",
    "ResidentDirectVectorTrialV66",
]
