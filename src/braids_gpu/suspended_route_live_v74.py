"""Actual-CUDA bounded owner for the Q4/v74 suspended route split.

The CUDA kernel composes the already mass-valued Q3 constitutive handoff and
performs the combine-only ``soursin_2d`` phase exchange.  Arrays remain on the
selected CUDA device throughout a trial.  Checkpoint serialization is the
only host-transfer path and there is deliberately no automatic CPU fallback.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Mapping

import numpy as np

from braids_core.sediment.suspended_route_live_v74 import (
    MAX_ACCEPTED_UPDATES,
    ROUTE_COMBINE,
    ROUTE_OFF,
    ROUTE_TOTAL,
    SuspendedRouteV74Error,
    load_route_checkpoint_v74,
    load_route_execution_v74,
    resolve_route_state_v74,
    route_checkpoint_document_v74,
)


SCOPE_ID = "braids-q4-split-suspended-route-owner-gpu-v74-v1"


def _canonical_array_digest(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(repr(tuple(array.shape)).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class ResidentSuspendedRouteStateV74:
    execution_fingerprint: str
    bed_mass_kg_m2: object
    suspended_mass_kg_m2: object
    suspended_exchange_ledger_kg: object
    bed_exchange_ledger_kg: object
    accepted_updates: int
    time_s: float
    state_sha256: str
    device_id: int


@dataclass(frozen=True)
class ResidentSuspendedRouteTrialV74:
    execution_fingerprint: str
    parent_state_sha256: str
    candidate: ResidentSuspendedRouteStateV74
    bed_capacity_x_kg_m_s: object
    bed_capacity_y_kg_m_s: object
    equilibrium_concentration_kg_m3: object
    suspended_equivalent_x_kg_m_s: object
    suspended_equivalent_y_kg_m_s: object
    dimensionless_timescale: object
    adaptation_velocity_m_s: object
    transferred_to_suspension_kg_m2: object
    limiter_correction_kg: object
    mass_residual_kg: object
    bed_suspended_owner_count: object
    water_suspended_owner_count: object
    suspended_factor_application_count: object
    active_fraction_application_count: object
    exchange_application_count: object
    duplicate_suspended_owner_count: object
    post_q3_density_conversion_count: int
    threads_per_block: int
    blocks: int
    trial_sha256: str
    device_id: int


class ResidentSuspendedRouteV74:
    """One explicit-device v74 route owner with transactional state."""

    backend = "implicit_fwave_midpoint_gpu"
    integrator = "implicit_fwave_midpoint"

    def __init__(self, execution: Mapping[str, object], *,
                 threads_per_block: int = 128):
        try:
            import cupy as cp
        except Exception as exc:  # pragma: no cover - environment dependent
            raise SuspendedRouteV74Error(
                "Q4/v74 actual CUDA requires CuPy and a working CUDA device"
            ) from exc
        try:
            if cp.cuda.runtime.getDeviceCount() < 1:
                raise SuspendedRouteV74Error(
                    "Q4/v74 actual CUDA requires a selected CUDA device")
        except Exception as exc:
            if isinstance(exc, SuspendedRouteV74Error):
                raise
            raise SuspendedRouteV74Error(
                "Q4/v74 actual CUDA device discovery failed") from exc
        self.cp = cp
        self.device_id = int(cp.cuda.Device().id)
        self.execution = load_route_execution_v74(execution)
        if (self.execution["backend"] != self.backend or
                self.execution["gpu_integrator"] != self.integrator or
                self.execution["automatic_backend_fallback"] is not False):
            raise SuspendedRouteV74Error(
                "Q4/v74 CUDA requires the explicitly selected "
                "implicit_fwave_midpoint_gpu backend without fallback")
        if (isinstance(threads_per_block, bool) or
                type(threads_per_block) is not int or
                not 32 <= threads_per_block <= 1024):
            raise SuspendedRouteV74Error(
                "Q4/v74 CUDA threads_per_block must be in [32,1024]")
        self.threads_per_block = threads_per_block
        self.nlane = len(self.execution["method_ids"])
        self.routes = cp.ascontiguousarray(cp.asarray(
            self.execution["routes"], dtype=cp.int32))
        self.output_kinds = cp.ascontiguousarray(cp.asarray(
            self.execution["output_kinds"], dtype=cp.int32))
        root = Path(__file__).resolve().parents[1]
        source_path = root / "braids_gpu/suspended_route_live_v74.cu"
        if not source_path.is_file():
            raise SuspendedRouteV74Error(
                "Q4/v74 CUDA implementation source is missing")
        source_bytes = source_path.read_bytes()
        self.cuda_source_sha256 = hashlib.sha256(source_bytes).hexdigest()
        try:
            module = cp.RawModule(
                code=source_bytes.decode("utf-8"),
                options=("--std=c++11",),
                name_expressions=("suspended_route_update_v74",),
            )
            self._kernel = module.get_function("suspended_route_update_v74")
        except Exception as exc:
            raise SuspendedRouteV74Error(
                "Q4/v74 CUDA kernel compilation failed") from exc
        properties = cp.cuda.runtime.getDeviceProperties(self.device_id)
        raw_name = properties.get("name", b"unknown")
        self.device_name = (
            raw_name.decode("utf-8", errors="replace")
            if isinstance(raw_name, bytes) else str(raw_name)
        )
        self.runtime_receipt = {
            "scope_id": SCOPE_ID,
            "execution_fingerprint": self.execution["fingerprint"],
            "backend": self.backend,
            "integrator": self.integrator,
            "automatic_backend_fallback": False,
            "actual_cuda_kernel": True,
            "device_id": self.device_id,
            "device_name": self.device_name,
            "cuda_source": "braids_gpu/suspended_route_live_v74.cu",
            "cuda_source_sha256": self.cuda_source_sha256,
            "threads_per_block": self.threads_per_block,
        }

    def _array(self, value, shape: tuple[int, ...], name: str,
               *, nonnegative: bool = False):
        cp = self.cp
        if (not isinstance(value, cp.ndarray) or value.dtype != cp.float64 or
                value.shape != shape or not value.flags.c_contiguous or
                int(value.device.id) != self.device_id):
            raise SuspendedRouteV74Error(
                f"Q4/v74 CUDA {name} must be resident contiguous FP64 "
                f"with shape {shape}")
        if not bool(cp.all(cp.isfinite(value)).item()):
            raise SuspendedRouteV74Error(
                f"Q4/v74 CUDA {name} contains non-finite data")
        if nonnegative and bool(cp.any(value < 0.0).item()):
            raise SuspendedRouteV74Error(
                f"Q4/v74 CUDA {name} must be nonnegative")
        return value

    def _digest(self, *values) -> str:
        return _canonical_array_digest(
            *(self.cp.asnumpy(value) for value in values))

    def _state_token(self, state: ResidentSuspendedRouteStateV74) -> str:
        digest = hashlib.sha256()
        digest.update(SCOPE_ID.encode("ascii"))
        digest.update(state.execution_fingerprint.encode("ascii"))
        digest.update(str(state.accepted_updates).encode("ascii"))
        digest.update(float(state.time_s).hex().encode("ascii"))
        digest.update(self._digest(
            state.bed_mass_kg_m2, state.suspended_mass_kg_m2,
            state.suspended_exchange_ledger_kg,
            state.bed_exchange_ledger_kg,
        ).encode("ascii"))
        return digest.hexdigest()

    def resolve_state(self, *, bed_mass_kg_m2, suspended_mass_kg_m2,
                      suspended_exchange_ledger_kg=None,
                      bed_exchange_ledger_kg=None, accepted_updates: int = 0,
                      time_s: float = 0.0) -> ResidentSuspendedRouteStateV74:
        cp = self.cp
        if (not isinstance(bed_mass_kg_m2, cp.ndarray) or
                bed_mass_kg_m2.ndim != 2 or
                bed_mass_kg_m2.shape[0] != self.nlane or
                bed_mass_kg_m2.shape[1] < 1):
            raise SuspendedRouteV74Error(
                "Q4/v74 CUDA phase inventories must have shape "
                "(route_lane,cell)")
        shape = tuple(map(int, bed_mass_kg_m2.shape))
        bed = self._array(bed_mass_kg_m2, shape, "bed mass", nonnegative=True)
        suspended = self._array(
            suspended_mass_kg_m2, shape, "suspended mass", nonnegative=True)
        if suspended_exchange_ledger_kg is None:
            suspended_exchange_ledger_kg = cp.zeros(
                self.nlane, dtype=cp.float64)
        if bed_exchange_ledger_kg is None:
            bed_exchange_ledger_kg = cp.zeros(self.nlane, dtype=cp.float64)
        water_ledger = self._array(
            suspended_exchange_ledger_kg, (self.nlane,),
            "suspended exchange ledger")
        bed_ledger = self._array(
            bed_exchange_ledger_kg, (self.nlane,), "bed exchange ledger")
        if (isinstance(accepted_updates, bool) or
                type(accepted_updates) is not int or
                not 0 <= accepted_updates <= MAX_ACCEPTED_UPDATES or
                not np.isfinite(float(time_s)) or float(time_s) < 0.0):
            raise SuspendedRouteV74Error(
                "Q4/v74 CUDA accepted update or time is invalid")
        # Own contiguous snapshots; external mutation cannot alter acceptance.
        placeholder = ResidentSuspendedRouteStateV74(
            self.execution["fingerprint"], bed.copy(order="C"),
            suspended.copy(order="C"), water_ledger.copy(order="C"),
            bed_ledger.copy(order="C"), accepted_updates, float(time_s), "",
            self.device_id,
        )
        return ResidentSuspendedRouteStateV74(
            placeholder.execution_fingerprint, placeholder.bed_mass_kg_m2,
            placeholder.suspended_mass_kg_m2,
            placeholder.suspended_exchange_ledger_kg,
            placeholder.bed_exchange_ledger_kg,
            placeholder.accepted_updates, placeholder.time_s,
            self._state_token(placeholder), placeholder.device_id,
        )

    def _validate_state(self, value: ResidentSuspendedRouteStateV74) -> None:
        if (not isinstance(value, ResidentSuspendedRouteStateV74) or
                value.execution_fingerprint != self.execution["fingerprint"] or
                value.device_id != self.device_id or
                value.bed_mass_kg_m2.shape[0] != self.nlane or
                value.state_sha256 != self._state_token(value)):
            raise SuspendedRouteV74Error(
                "Q4/v74 CUDA accepted state is stale, mutated, or mismatched")

    def begin_trial(
        self, *, parent: ResidentSuspendedRouteStateV74, depth_m,
        velocity_x_m_s, velocity_y_m_s, raw_bed_x_kg_m_s,
        raw_bed_y_kg_m_s, raw_suspended_x_kg_m_s,
        raw_suspended_y_kg_m_s, raw_equilibrium_concentration_kg_m3,
        active_fraction, depth_before_m, depth_after_m, speed_m_s,
        shear_velocity_m_s, settling_velocity_m_s, cell_area_m2,
        dt_s: float, cancelled=lambda: False,
    ) -> ResidentSuspendedRouteTrialV74:
        cp = self.cp
        self._validate_state(parent)
        if not callable(cancelled):
            raise TypeError("cancelled must be callable")
        if cancelled():
            raise InterruptedError("Q4/v74 CUDA update cancelled before launch")
        if parent.accepted_updates >= MAX_ACCEPTED_UPDATES:
            raise SuspendedRouteV74Error(
                "Q4/v74 CUDA accepted update limit was reached")
        if not np.isfinite(float(dt_s)) or float(dt_s) <= 0.0:
            raise SuspendedRouteV74Error("Q4/v74 CUDA dt_s must be positive")
        nlane, ncell = map(int, parent.bed_mass_kg_m2.shape)
        shape = (nlane, ncell)
        depth = self._array(depth_m, (ncell,), "depth", nonnegative=True)
        velocity_x = self._array(
            velocity_x_m_s, (ncell,), "velocity x")
        velocity_y = self._array(
            velocity_y_m_s, (ncell,), "velocity y")
        raw_bed_x = self._array(raw_bed_x_kg_m_s, shape, "raw bed x")
        raw_bed_y = self._array(raw_bed_y_kg_m_s, shape, "raw bed y")
        raw_suspended_x = self._array(
            raw_suspended_x_kg_m_s, shape, "raw suspended x")
        raw_suspended_y = self._array(
            raw_suspended_y_kg_m_s, shape, "raw suspended y")
        raw_equilibrium = self._array(
            raw_equilibrium_concentration_kg_m3, shape,
            "raw equilibrium concentration", nonnegative=True)
        fraction = self._array(
            active_fraction, shape, "active fraction", nonnegative=True)
        if bool(cp.any(fraction > 1.0).item()):
            raise SuspendedRouteV74Error(
                "Q4/v74 CUDA active fraction must be in [0,1]")
        depth_before = self._array(
            depth_before_m, (ncell,), "depth before", nonnegative=True)
        depth_after = self._array(
            depth_after_m, (ncell,), "depth after", nonnegative=True)
        speed = self._array(speed_m_s, (ncell,), "speed", nonnegative=True)
        shear = self._array(
            shear_velocity_m_s, (ncell,), "shear velocity", nonnegative=True)
        settling = self._array(
            settling_velocity_m_s, shape, "settling velocity",
            nonnegative=True)
        area = self._array(
            cell_area_m2, (ncell,), "cell area", nonnegative=True)
        if (bool(cp.any(depth <= 0.0).item()) or
                bool(cp.any(depth_before <= 0.0).item()) or
                bool(cp.any(depth_after <= 0.0).item()) or
                bool(cp.any(area <= 0.0).item())):
            raise SuspendedRouteV74Error(
                "Q4/v74 CUDA gate requires positive wet depths and cell area")

        outputs = [cp.empty(shape, dtype=cp.float64) for _ in range(12)]
        status = cp.empty(shape, dtype=cp.int32)
        count = nlane*ncell
        blocks = (count+self.threads_per_block-1)//self.threads_per_block
        self._kernel(
            (blocks,), (self.threads_per_block,),
            (
                np.int32(nlane), np.int32(ncell), self.output_kinds,
                self.routes, depth, velocity_x, velocity_y, raw_bed_x,
                raw_bed_y, raw_suspended_x, raw_suspended_y,
                raw_equilibrium, fraction, parent.bed_mass_kg_m2,
                parent.suspended_mass_kg_m2, depth_before, depth_after,
                speed, shear, settling, area,
                np.float64(self.execution["bed_factor"]),
                np.float64(self.execution["suspended_factor"]),
                np.float64(self.execution["speed_epsilon"]),
                np.float64(self.execution["fac_tsd"]),
                np.float64(self.execution["sediment_threshold_m"]),
                np.float64(self.execution["minimum_reference_depth_m"]),
                np.float64(dt_s), *outputs, status,
            ),
        )
        cp.cuda.get_current_stream().synchronize()
        if cancelled():
            raise InterruptedError(
                "Q4/v74 CUDA update cancelled before publication")
        if bool(cp.any(status != 0).item()):
            first = cp.asnumpy(cp.argwhere(status != 0)[0]).tolist()
            raise SuspendedRouteV74Error(
                f"Q4/v74 CUDA kernel rejected lane/cell {first}")
        (bed_capacity_x, bed_capacity_y, equilibrium, equivalent_x,
         equivalent_y, next_bed, next_suspended, transfer, timescale,
         adaptation_velocity, limiter_area, local_residual) = outputs

        water_delta = cp.ascontiguousarray(cp.sum(
            transfer*area[None, :], axis=1))
        bed_delta = cp.ascontiguousarray(-water_delta)
        limiter = cp.ascontiguousarray(cp.sum(limiter_area, axis=1))
        residual = cp.ascontiguousarray(cp.sum(local_residual, axis=1))
        route_host = np.asarray(self.execution["routes"], dtype=np.int32)
        bed_owner = cp.ascontiguousarray(cp.asarray(
            (route_host == ROUTE_TOTAL).astype(np.int32)))
        water_owner = cp.ascontiguousarray(cp.asarray(
            (route_host == ROUTE_COMBINE).astype(np.int32)))
        active_owner = cp.ascontiguousarray(cp.asarray(
            (route_host != ROUTE_OFF).astype(np.int32)))
        exchange_count = water_owner.copy(order="C")
        duplicate = cp.ascontiguousarray(bed_owner*water_owner)
        if bool(cp.any(duplicate != 0).item()):
            raise SuspendedRouteV74Error(
                "Q4/v74 CUDA formula suspended half has duplicate owners")
        candidate = self.resolve_state(
            bed_mass_kg_m2=next_bed,
            suspended_mass_kg_m2=next_suspended,
            suspended_exchange_ledger_kg=cp.ascontiguousarray(
                parent.suspended_exchange_ledger_kg+water_delta),
            bed_exchange_ledger_kg=cp.ascontiguousarray(
                parent.bed_exchange_ledger_kg+bed_delta),
            accepted_updates=parent.accepted_updates+1,
            time_s=parent.time_s+float(dt_s),
        )
        trial_digest = hashlib.sha256()
        trial_digest.update(SCOPE_ID.encode("ascii"))
        trial_digest.update(parent.state_sha256.encode("ascii"))
        trial_digest.update(candidate.state_sha256.encode("ascii"))
        trial_digest.update(self._digest(
            bed_capacity_x, bed_capacity_y, equilibrium, equivalent_x,
            equivalent_y, timescale, adaptation_velocity, transfer,
            limiter, residual, bed_owner, water_owner, active_owner,
            active_owner, exchange_count, duplicate,
        ).encode("ascii"))
        trial_digest.update(b"post_q3_density_conversion_count=0")
        trial_digest.update(str(self.threads_per_block).encode("ascii"))
        trial_digest.update(str(blocks).encode("ascii"))
        token = trial_digest.hexdigest()
        return ResidentSuspendedRouteTrialV74(
            self.execution["fingerprint"], parent.state_sha256, candidate,
            bed_capacity_x, bed_capacity_y, equilibrium, equivalent_x,
            equivalent_y, timescale, adaptation_velocity, transfer,
            limiter, residual, bed_owner, water_owner, active_owner.copy(),
            active_owner, exchange_count, duplicate, 0,
            self.threads_per_block, blocks, token, self.device_id,
        )

    def _validate_trial(self, parent: ResidentSuspendedRouteStateV74,
                        trial: ResidentSuspendedRouteTrialV74) -> None:
        self._validate_state(parent)
        if (not isinstance(trial, ResidentSuspendedRouteTrialV74) or
                trial.execution_fingerprint != self.execution["fingerprint"] or
                trial.parent_state_sha256 != parent.state_sha256 or
                trial.device_id != self.device_id or
                trial.post_q3_density_conversion_count != 0 or
                trial.threads_per_block != self.threads_per_block):
            raise SuspendedRouteV74Error(
                "Q4/v74 CUDA trial has a different accepted parent")
        self._validate_state(trial.candidate)
        digest = hashlib.sha256()
        digest.update(SCOPE_ID.encode("ascii"))
        digest.update(parent.state_sha256.encode("ascii"))
        digest.update(trial.candidate.state_sha256.encode("ascii"))
        digest.update(self._digest(
            trial.bed_capacity_x_kg_m_s, trial.bed_capacity_y_kg_m_s,
            trial.equilibrium_concentration_kg_m3,
            trial.suspended_equivalent_x_kg_m_s,
            trial.suspended_equivalent_y_kg_m_s,
            trial.dimensionless_timescale, trial.adaptation_velocity_m_s,
            trial.transferred_to_suspension_kg_m2,
            trial.limiter_correction_kg, trial.mass_residual_kg,
            trial.bed_suspended_owner_count,
            trial.water_suspended_owner_count,
            trial.suspended_factor_application_count,
            trial.active_fraction_application_count,
            trial.exchange_application_count,
            trial.duplicate_suspended_owner_count,
        ).encode("ascii"))
        digest.update(b"post_q3_density_conversion_count=0")
        digest.update(str(trial.threads_per_block).encode("ascii"))
        digest.update(str(trial.blocks).encode("ascii"))
        if digest.hexdigest() != trial.trial_sha256:
            raise SuspendedRouteV74Error(
                "Q4/v74 CUDA trial receipt is stale or mutated")

    def commit_trial(self, parent, trial) -> ResidentSuspendedRouteStateV74:
        self._validate_trial(parent, trial)
        return trial.candidate

    def rollback_trial(self, parent, trial) -> ResidentSuspendedRouteStateV74:
        self._validate_trial(parent, trial)
        return parent

    def cancel_trial(self, parent, trial) -> ResidentSuspendedRouteStateV74:
        return self.rollback_trial(parent, trial)

    def checkpoint_document(self, state) -> dict[str, object]:
        """Serialize one accepted resident state using the portable schema."""
        self._validate_state(state)
        portable = resolve_route_state_v74(
            execution=self.execution,
            bed_mass_kg_m2=self.cp.asnumpy(state.bed_mass_kg_m2),
            suspended_mass_kg_m2=self.cp.asnumpy(state.suspended_mass_kg_m2),
            suspended_exchange_ledger_kg=self.cp.asnumpy(
                state.suspended_exchange_ledger_kg),
            bed_exchange_ledger_kg=self.cp.asnumpy(
                state.bed_exchange_ledger_kg),
            accepted_updates=state.accepted_updates, time_s=state.time_s,
        )
        return route_checkpoint_document_v74(
            portable, execution=self.execution)

    def load_checkpoint(self, value: Mapping[str, object]):
        """Restore a portable checkpoint to the selected CUDA device."""
        portable = load_route_checkpoint_v74(value, execution=self.execution)
        cp = self.cp
        return self.resolve_state(
            bed_mass_kg_m2=cp.ascontiguousarray(cp.asarray(
                portable.bed_mass_kg_m2, dtype=cp.float64)),
            suspended_mass_kg_m2=cp.ascontiguousarray(cp.asarray(
                portable.suspended_mass_kg_m2, dtype=cp.float64)),
            suspended_exchange_ledger_kg=cp.ascontiguousarray(cp.asarray(
                portable.suspended_exchange_ledger_kg, dtype=cp.float64)),
            bed_exchange_ledger_kg=cp.ascontiguousarray(cp.asarray(
                portable.bed_exchange_ledger_kg, dtype=cp.float64)),
            accepted_updates=portable.accepted_updates,
            time_s=portable.time_s,
        )


__all__ = [
    "ResidentSuspendedRouteStateV74", "ResidentSuspendedRouteTrialV74",
    "ResidentSuspendedRouteV74", "SCOPE_ID",
]
