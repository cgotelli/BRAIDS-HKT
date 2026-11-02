"""Actual-CUDA owner for the bounded Q4/v85 DensIn transaction."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from braids_core.sediment.densin_fwave_v85 import (
    MAX_ACCEPTED_UPDATES,
    AcceptedConcentrationReceiptV85,
    DensInExecutionV85,
    DensInStateV85,
    DensInTrialV85,
    DensInV85Error,
    _build_trial,
    _validate_hydraulic_receipt,
    _validate_trial,
    densin_checkpoint_v85,
    load_densin_checkpoint_v85,
    validate_accepted_concentration_receipt_v85,
    validate_densin_state_v85,
)


ROOT = Path(__file__).resolve().parents[1]
CUDA_SOURCE = Path(__file__).with_suffix(".cu")
CUDA_OPTIONS = ("--std=c++11", "--fmad=false", "--prec-div=true",
                "--prec-sqrt=true", "--ftz=false")
KERNELS = (
    "densin_mixture_v85", "densin_face_v85",
    "densin_project_stride_v85", "densin_compose_v85",
)
THREADS = 128
CUDA_COMPOSITION_ID = "resident-densin-implicit-fwave-midpoint-v85-v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DensInV85Error(message)


class ResidentDensInFwaveV85:
    """Keep geometry and trial work resident; read back only acceptance evidence."""

    def __init__(self, execution: DensInExecutionV85):
        if (not isinstance(execution, DensInExecutionV85) or
                execution.role != "implicit_fwave_midpoint_gpu"):
            raise DensInV85Error(
                "v85 CUDA owner requires the selected implicit GPU role")
        import cupy as cp

        self.cp = cp
        self.execution = execution
        self.device_id = int(cp.cuda.Device().id)
        source = CUDA_SOURCE.read_text(encoding="utf-8")
        self.module = cp.RawModule(
            code=source, options=CUDA_OPTIONS, name_expressions=KERNELS)
        self.kernels = {name: self.module.get_function(name)
                        for name in KERNELS}
        self.nc, self.nf, self.ng = (
            execution.cell_count, execution.face_count,
            execution.fraction_count)
        self.owner = cp.asarray(execution.owner)
        self.neighbour = cp.asarray(execution.neighbour)
        self.normal = cp.asarray(execution.normal)
        self.length = cp.asarray(execution.face_length_m)
        self.distance = cp.asarray(execution.face_distance_m)
        self.offset = cp.asarray(execution.cell_face_offset)
        self.faces = cp.asarray(execution.cell_faces)
        self.signs = cp.asarray(execution.cell_face_sign)
        self.water = cp.asarray(execution.water_density_kg_m3)
        self.solid = cp.asarray(execution.solid_density_kg_m3)
        self.concentration = cp.empty((self.ng, self.nc), dtype=cp.float64)
        self.mixture = cp.empty(self.nc, dtype=cp.float64)
        self.face_acceleration = cp.empty(self.nf, dtype=cp.float64)
        self.cell_acceleration = cp.empty((2, self.nc), dtype=cp.float64)
        self.density_rate = cp.empty((2, self.nc), dtype=cp.float64)
        self.candidate_q = cp.empty((3, self.nc), dtype=cp.float64)
        self.accepted_q = cp.empty((3, self.nc), dtype=cp.float64)
        self.cell_status = cp.empty(self.nc, dtype=cp.int32)
        self.face_status = cp.empty(self.nf, dtype=cp.int32)
        self._resident_parent = None
        self._open_parent = None
        self._open_trial = None
        properties = cp.cuda.runtime.getDeviceProperties(self.device_id)
        name = properties["name"]
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="replace")
        sources = (
            ROOT/"braids_core/sediment/densin_fwave_v85.py",
            ROOT/"braids_gpu/densin_fwave_v85.py", CUDA_SOURCE,
        )
        self.runtime_receipt = {
            "cuda_composition_id": CUDA_COMPOSITION_ID,
            "execution_fingerprint": execution.fingerprint,
            "backend": "actual_cuda",
            "integrator": "implicit_fwave_midpoint",
            "actual_cuda_kernels": True,
            "geometry_resident": True,
            "candidate_resident_until_acceptance_readback": True,
            "accepted_q_resident": True,
            "automatic_cpu_fallback": False,
            "device_id": self.device_id,
            "device_name": str(name),
            "threads_per_block": THREADS,
            "source_sha256": {
                str(path.relative_to(ROOT)).replace("\\", "/"):
                    hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sources
            },
        }

    def _launch(self, name: str, size: int, args: tuple) -> None:
        if size:
            self.kernels[name](((size+THREADS-1)//THREADS,),
                               (THREADS,), args)

    def _device(self, value, shape, name):
        cp = self.cp
        if (not isinstance(value, cp.ndarray) or value.shape != shape or
                value.dtype != cp.float64 or not value.flags.c_contiguous or
                int(value.device.id) != self.device_id):
            raise DensInV85Error(
                f"v85 CUDA {name} must be resident contiguous float64 {shape}")
        return value

    def _sync_parent(self, parent: DensInStateV85) -> None:
        if self._resident_parent != parent.state_sha256:
            self.accepted_q.set(parent.q)
            self._resident_parent = parent.state_sha256

    def begin_trial(
        self, parent: DensInStateV85, *,
        accepted_concentration: AcceptedConcentrationReceiptV85,
        hydraulic_trial_receipt, midpoint_q,
        augmented_fwave_base_rate, dt_s: float,
        cancelled: bool = False,
    ) -> DensInTrialV85:
        cp = self.cp
        validate_densin_state_v85(parent, self.execution)
        _require(self._open_trial is None,
                 "v85 CUDA previous transaction is unfinished")
        if cancelled:
            raise DensInV85Error(
                "v85 CUDA trial cancelled before kernel launch")
        if parent.accepted_updates >= MAX_ACCEPTED_UPDATES:
            raise DensInV85Error("v85 accepted update limit was reached")
        validate_accepted_concentration_receipt_v85(
            accepted_concentration, self.execution,
            accepted_update=parent.accepted_updates, time_s=parent.time_s,
            require_typed_v78=True)
        midpoint = self._device(midpoint_q, (3, self.nc), "midpoint state")
        base_rate = self._device(
            augmented_fwave_base_rate, (3, self.nc), "base f-wave rate")
        if (isinstance(dt_s, bool) or not isinstance(dt_s, (int, float)) or
                not np.isfinite(float(dt_s)) or float(dt_s) <= 0.0):
            raise DensInV85Error("v85 CUDA timestep must be finite and positive")
        dt = float(dt_s)
        host_midpoint = np.ascontiguousarray(cp.asnumpy(midpoint))
        host_base_rate = np.ascontiguousarray(cp.asnumpy(base_rate))
        _validate_hydraulic_receipt(
            hydraulic_trial_receipt, self.execution, parent, dt,
            host_midpoint, host_base_rate)
        if (not np.array_equal(parent.q[0],
                               accepted_concentration.depth_m) or
                np.any(host_midpoint[0] <= self.execution.dry_tolerance_m)):
            raise DensInV85Error(
                "v85 CUDA concentration depth differs from the parent state")
        self._sync_parent(parent)
        self.concentration.set(accepted_concentration.concentration_kg_m3)
        self.cell_status.fill(0)
        self._launch("densin_mixture_v85", self.nc, (
            self.concentration, self.water, self.solid,
            np.int64(self.ng), np.int64(self.nc), self.mixture,
            self.cell_status))
        if bool(cp.any(self.cell_status != 0).item()):
            raise DensInV85Error("v85 CUDA mixture-density kernel failed")
        self.face_status.fill(0)
        self._launch("densin_face_v85", self.nf, (
            midpoint[0], self.mixture, self.owner, self.neighbour,
            self.distance, np.int64(self.nf),
            np.float64(self.execution.gravity_m_s2),
            np.float64(self.execution.hydraulic_reference_density_kg_m3),
            self.face_acceleration, self.face_status))
        if bool(cp.any(self.face_status != 0).item()):
            raise DensInV85Error("v85 CUDA face-density kernel failed")
        self.cell_status.fill(0)
        self._launch("densin_project_stride_v85", self.nc, (
            self.face_acceleration, self.normal, self.length,
            self.offset, self.faces, self.signs, midpoint[0],
            np.int64(self.nc), np.int64(self.nf),
            self.cell_acceleration, self.density_rate, self.cell_status))
        if bool(cp.any(self.cell_status != 0).item()):
            raise DensInV85Error("v85 CUDA face-to-cell projection failed")
        self.cell_status.fill(0)
        self._launch("densin_compose_v85", self.nc, (
            self.accepted_q, base_rate, self.density_rate,
            np.int64(self.nc), np.float64(dt),
            np.float64(self.execution.dry_tolerance_m),
            self.candidate_q, self.cell_status))
        if bool(cp.any(self.cell_status != 0).item()):
            raise DensInV85Error("v85 CUDA coupled midpoint candidate failed")
        mixture = np.ascontiguousarray(cp.asnumpy(self.mixture))
        face = np.ascontiguousarray(cp.asnumpy(self.face_acceleration))
        cell = np.ascontiguousarray(cp.asnumpy(self.cell_acceleration))
        rate = np.ascontiguousarray(cp.asnumpy(self.density_rate))
        candidate = np.ascontiguousarray(cp.asnumpy(self.candidate_q))
        trial = _build_trial(
            self.execution, parent, accepted_concentration,
            hydraulic_trial_receipt, candidate_q=candidate,
            mixture_density=mixture, face_acceleration=face,
            cell_acceleration=cell, density_rate=rate,
            baseline_candidate_depth_m=parent.q[0]+dt*host_base_rate[0],
            backend="actual_cuda")
        self._open_parent = parent.state_sha256
        self._open_trial = trial.trial_sha256
        return trial

    def _finish(self, parent: DensInStateV85, trial: DensInTrialV85, *,
                commit: bool) -> DensInStateV85:
        _require(self._open_parent == parent.state_sha256 and
                 self._open_trial == trial.trial_sha256,
                 "v85 CUDA transaction is not the resident open trial")
        _validate_trial(self.execution, parent, trial)
        if commit:
            self.accepted_q.set(trial.candidate.q)
            self._resident_parent = trial.candidate.state_sha256
            result = trial.candidate
        else:
            result = parent
        self._open_parent = None
        self._open_trial = None
        return result

    def commit(self, parent: DensInStateV85,
               trial: DensInTrialV85) -> DensInStateV85:
        return self._finish(parent, trial, commit=True)

    def rollback(self, parent: DensInStateV85,
                 trial: DensInTrialV85) -> DensInStateV85:
        return self._finish(parent, trial, commit=False)

    def cancel(self, parent: DensInStateV85,
               trial: DensInTrialV85) -> DensInStateV85:
        return self.rollback(parent, trial)

    def checkpoint(self, state: DensInStateV85) -> dict:
        _require(self._open_trial is None,
                 "v85 CUDA checkpoint requires a committed point")
        validate_densin_state_v85(state, self.execution)
        self._sync_parent(state)
        resident = np.ascontiguousarray(self.cp.asnumpy(self.accepted_q))
        _require(np.array_equal(resident, state.q),
                 "v85 CUDA resident state differs from checkpoint state")
        return densin_checkpoint_v85(self.execution, state)

    def restore_checkpoint(self, document) -> DensInStateV85:
        _require(self._open_trial is None,
                 "v85 CUDA restart cannot replace an open transaction")
        state = load_densin_checkpoint_v85(self.execution, document)
        self.accepted_q.set(state.q)
        self._resident_parent = state.state_sha256
        return state


def method_identity_v85_cuda() -> dict:
    return {
        "cuda_composition_id": CUDA_COMPOSITION_ID,
        "integrator": "implicit_fwave_midpoint",
        "actual_cuda_kernels": True,
        "resident_geometry": True,
        "accepted_q_resident": True,
        "automatic_cpu_fallback": False,
        "scope": "fixed_grid_fully_wet_kmax1_operator_transaction",
        "nonlinear_residual_integration": False,
        "product_admission": False,
    }


__all__ = [
    "CUDA_COMPOSITION_ID", "ResidentDensInFwaveV85",
    "method_identity_v85_cuda",
]
