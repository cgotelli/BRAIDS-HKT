"""Actual-CUDA transaction owner for the bounded Q4/v90 HLES producer."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np

from braids_core.sediment.hles_producer_v90 import (
    BACKEND_GPU,
    FILTER_INTERVAL,
    INTEGRATOR,
    MAPPING_ID,
    MAX_ACCEPTED_UPDATES,
    PRODUCER_CADENCE,
    HlesPrimeTrialV90,
    HlesProducerExecutionV90,
    HlesProducerStateV90,
    HlesProducerTrialV90,
    HlesProducerV90Error,
    _input_arrays,
    _finite,
    _validate_input_receipt,
    _validate_prime_receipt,
    _validate_prime_trial,
    _validate_trial,
    _wrap_prime_trial,
    _wrap_trial,
    hles_checkpoint_v90,
    load_hles_checkpoint_v90,
    validate_hles_producer_state_v90,
)
from braids_gpu.multifraction_suspended_lifecycle_v78 import (
    CUDA_OPTIONS,
    CUDA_THREADS_PER_BLOCK,
)


ROOT = Path(__file__).resolve().parents[1]
CUDA_SOURCE = ROOT/"braids_gpu/hles_producer_v90.cu"
DETVIC_CUDA_SOURCE = ROOT/"braids_gpu/horizontal_turbulence_lifecycle_v87.cu"
CUDA_COMPOSITION_ID = "actual_cuda_flow2dh_hles_pure_producer_v90_v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HlesProducerV90Error(message)


class ResidentHlesProducerLifecycleV90:
    """Selected actual-CUDA closure owner with no CPU numerical fallback."""

    def __init__(self, execution: HlesProducerExecutionV90):
        _require(isinstance(execution, HlesProducerExecutionV90) and
                 execution.backend == BACKEND_GPU,
                 "v90 CUDA owner requires explicit GPU execution")
        import cupy as cp

        self.execution = execution
        self.cp = cp
        self.device_id = int(cp.cuda.Device().id)
        try:
            code = (CUDA_SOURCE.read_text(encoding="utf-8") + "\n" +
                    DETVIC_CUDA_SOURCE.read_text(encoding="utf-8"))
            names = (
                "v90_lpfluc_face", "v90_reconstruct_cell_vector_nface",
                "v90_protke_green_gauss", "v87_detvic_current_only",
            )
            module = cp.RawModule(
                code=code, options=CUDA_OPTIONS, name_expressions=names)
            self.filter_kernel = module.get_function(names[0])
            self.reconstruct_kernel = module.get_function(names[1])
            self.protke_kernel = module.get_function(names[2])
            self.detvic_kernel = module.get_function(names[3])
        except Exception as error:
            raise HlesProducerV90Error(
                "v90 CUDA closure kernels failed to compile") from error

        base = execution.closure_execution.base_execution.base_execution
        self.owner = cp.ascontiguousarray(cp.asarray(base.owner, dtype=cp.int32))
        self.neighbour = cp.ascontiguousarray(
            cp.asarray(base.neighbour, dtype=cp.int32))
        self.offset = cp.ascontiguousarray(
            cp.asarray(base.cell_face_offset, dtype=cp.int32))
        self.faces = cp.ascontiguousarray(
            cp.asarray(base.cell_faces, dtype=cp.int32))
        self.signs = cp.ascontiguousarray(
            cp.asarray(base.cell_face_sign, dtype=cp.int32))
        self.normal = cp.ascontiguousarray(
            cp.asarray(base.normal, dtype=cp.float64))
        self.length = cp.ascontiguousarray(
            cp.asarray(base.face_length_m, dtype=cp.float64))
        self.distance = cp.ascontiguousarray(
            cp.asarray(base.centre_distance_m, dtype=cp.float64))
        self.area = cp.ascontiguousarray(
            cp.asarray(base.cell_area_m2, dtype=cp.float64))

        props = cp.cuda.runtime.getDeviceProperties(self.device_id)
        name = props["name"]
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="replace")
        sources = (
            ROOT/"braids_core/sediment/hles_producer_v90.py",
            ROOT/"braids_gpu/hles_producer_v90.py", CUDA_SOURCE,
            DETVIC_CUDA_SOURCE,
        )
        self.runtime_receipt = {
            "cuda_composition_id": CUDA_COMPOSITION_ID,
            "execution_fingerprint": execution.fingerprint,
            "integrator": INTEGRATOR,
            "backend": BACKEND_GPU,
            "device_id": self.device_id,
            "device_name": str(name),
            "threads_per_block": CUDA_THREADS_PER_BLOCK,
            "actual_cuda_filter": True,
            "actual_cuda_reconstruction": True,
            "actual_cuda_protke": True,
            "actual_cuda_detvic_v87": True,
            "automatic_cpu_fallback": False,
            "mapping_id": MAPPING_ID,
            "filter_interval": FILTER_INTERVAL,
            "producer_cadence": PRODUCER_CADENCE,
            "nested_v82_v78_lifecycle": False,
            "source_sha256": {
                str(path.relative_to(ROOT)).replace("\\", "/"):
                    hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sources
            },
        }
        self._open_parent: str | None = None
        self._open_trial: str | None = None
        self._open_kind: str | None = None
        self._kernel_launch_counts = {
            "lpfluc": 0, "reconstruct": 0, "protke": 0, "detvic": 0}

    @property
    def kernel_launch_counts(self) -> dict[str, int]:
        """Return actual physical launches made by this bounded owner."""
        return dict(self._kernel_launch_counts)

    def _device(self, value, shape, dtype, name):
        cp = self.cp
        _require(isinstance(value, cp.ndarray) and value.shape == shape and
                 value.dtype == dtype and value.flags.c_contiguous and
                 int(value.device.id) == self.device_id,
                 f"v90 CUDA {name} must be contiguous resident {dtype} {shape}")
        return value

    def _resident_inputs(
        self, accepted_face_normal_velocity_m_s, active_face,
        closure_depth_m, closure_chezy_mhalf_s, active_cell,
    ):
        cp = self.cp
        nf, nc = self.execution.face_count, self.execution.cell_count
        return (
            self._device(accepted_face_normal_velocity_m_s, (nf,),
                         cp.float64, "accepted face-normal velocity"),
            self._device(active_face, (nf,), cp.int32, "active face"),
            self._device(closure_depth_m, (nc,), cp.float64,
                         "closure depth"),
            self._device(closure_chezy_mhalf_s, (nc,), cp.float64,
                         "closure Chezy"),
            self._device(active_cell, (nc,), cp.int32, "active cell"),
        )

    def _host_inputs(self, resident):
        cp = self.cp
        try:
            values = tuple(cp.asnumpy(array) for array in resident)
            return _input_arrays(
                self.execution,
                accepted_face_normal_velocity_m_s=values[0],
                active_face=values[1], closure_depth_m=values[2],
                closure_chezy_mhalf_s=values[3], active_cell=values[4])
        except Exception as error:
            raise HlesProducerV90Error(
                "v90 CUDA closure input readback failed") from error

    def _compute(self, parent, resident, closure_interval_s: float):
        cp = self.cp
        mean_face, active_face, depth, chezy, active_cell = resident
        nf, nc = self.execution.face_count, self.execution.cell_count
        face_blocks = ((nf+CUDA_THREADS_PER_BLOCK-1)//CUDA_THREADS_PER_BLOCK,)
        cell_blocks = ((nc+CUDA_THREADS_PER_BLOCK-1)//CUDA_THREADS_PER_BLOCK,)
        threads = (CUDA_THREADS_PER_BLOCK,)
        previous = cp.ascontiguousarray(cp.asarray(
            parent.accepted_filtered_face_normal_velocity_m_s,
            dtype=cp.float64))
        filtered = cp.empty(nf, dtype=cp.float64)
        fluctuation = cp.empty(nf, dtype=cp.float64)
        face_status = cp.empty(nf, dtype=cp.int32)
        mean_cell = cp.empty((2, nc), dtype=cp.float64)
        fluct_cell = cp.empty((2, nc), dtype=cp.float64)
        mean_complete = cp.empty(nc, dtype=cp.int32)
        fluct_complete = cp.empty(nc, dtype=cp.int32)
        mean_status = cp.empty(nc, dtype=cp.int32)
        fluct_status = cp.empty(nc, dtype=cp.int32)
        complete = cp.empty(nc, dtype=cp.int32)
        ptke = cp.empty(nc, dtype=cp.float64)
        boundary = cp.empty(nc, dtype=cp.int32)
        protke_status = cp.empty(nc, dtype=cp.int32)
        hles = cp.empty(nc, dtype=cp.float64)
        elder = cp.empty(nc, dtype=cp.float64)
        dynamic_vic = cp.empty(nc, dtype=cp.float64)
        dynamic_dic = cp.empty(nc, dtype=cp.float64)
        detvic_status = cp.empty(nc, dtype=cp.int32)
        try:
            self._kernel_launch_counts["lpfluc"] += 1
            self.filter_kernel(face_blocks, threads, (
                np.int32(nf), np.float64(closure_interval_s),
                np.float64(self.execution.relaxation_time_min), previous,
                mean_face, active_face, filtered, fluctuation, face_status))
            for face_value, vector, valid, status in (
                (mean_face, mean_cell, mean_complete, mean_status),
                (fluctuation, fluct_cell, fluct_complete, fluct_status),
            ):
                self._kernel_launch_counts["reconstruct"] += 1
                self.reconstruct_kernel(cell_blocks, threads, (
                    np.int32(nc), np.int32(nf), self.offset, self.faces,
                    self.normal, self.length, self.distance, active_face,
                    face_value, vector, valid, status))
            complete[...] = ((mean_complete == 1) &
                             (fluct_complete == 1)).astype(cp.int32)
            self._kernel_launch_counts["protke"] += 1
            self.protke_kernel(cell_blocks, threads, (
                np.int32(nc), np.int32(nf), self.offset, self.faces,
                self.signs, self.owner, self.neighbour, self.normal,
                self.length, self.area, fluct_cell, complete, active_cell,
                ptke, boundary, protke_status))
            detvic_active = ((active_cell == 1) &
                             (mean_complete == 1)).astype(cp.int32)
            closure = self.execution.closure_execution
            self._kernel_launch_counts["detvic"] += 1
            self.detvic_kernel(cell_blocks, threads, (
                np.int32(nc), np.int32(closure.hles_enabled),
                np.int32(closure.elder_enabled), np.float64(closure.gamma),
                np.float64(closure.sigmat), np.float64(closure.flp),
                np.float64(closure.vicmol_m2_s),
                np.float64(closure.dicmol_m2_s),
                np.float64(closure.von_karman),
                np.float64(closure.gravity_m_s2),
                np.float64(closure.base_execution.base_execution.dry_tolerance_m),
                self.area, depth, mean_cell, chezy, ptke, detvic_active,
                hles, elder, dynamic_vic, dynamic_dic, detvic_status))
            if bool(cp.any(face_status != 0).item() or
                    cp.any(mean_status != 0).item() or
                    cp.any(fluct_status != 0).item() or
                    cp.any(protke_status != 0).item() or
                    cp.any(detvic_status != 0).item()):
                raise HlesProducerV90Error(
                    "v90 CUDA closure kernel status rejected the trial")
            host = tuple(cp.asnumpy(array) for array in (
                filtered, fluctuation, mean_cell, fluct_cell, complete,
                boundary, detvic_active, ptke, hles, elder,
                dynamic_vic, dynamic_dic))
        except HlesProducerV90Error:
            raise
        except Exception as error:
            raise HlesProducerV90Error(
                "v90 CUDA closure output readback failed") from error
        coefficient = (math.exp(-closure_interval_s/(
            60.0*self.execution.relaxation_time_min))
            if self.execution.relaxation_time_min > 0.0 else 0.0)
        return (coefficient, *host)

    def begin_prime(
        self, parent: HlesProducerStateV90, *, prime_receipt,
        closure_interval_s, accepted_face_normal_velocity_m_s, active_face,
        closure_depth_m, closure_chezy_mhalf_s, active_cell,
        cancelled: bool = False,
    ) -> HlesPrimeTrialV90:
        validate_hles_producer_state_v90(parent, self.execution)
        _require(self._open_trial is None,
                 "v90 CUDA previous transaction is unfinished")
        if cancelled:
            raise HlesProducerV90Error(
                "v90 CUDA cold prime cancelled before closure launch")
        if parent.primed or parent.accepted_updates != 0 or parent.time_s != 0.0:
            raise HlesProducerV90Error(
                "v90 CUDA cold prime requires unprimed zero-time state")
        interval = _finite(
            closure_interval_s, "prime closure interval", positive=True)
        resident = self._resident_inputs(
            accepted_face_normal_velocity_m_s, active_face,
            closure_depth_m, closure_chezy_mhalf_s, active_cell)
        host = self._host_inputs(resident)
        _validate_prime_receipt(
            self.execution, parent, prime_receipt, host, interval)
        values = self._compute(parent, resident, interval)
        trial = _wrap_prime_trial(
            self.execution, parent, prime_receipt, values, BACKEND_GPU)
        self._open_parent = parent.state_sha256
        self._open_trial = trial.trial_sha256
        self._open_kind = "prime"
        return trial

    def begin_trial(
        self, parent: HlesProducerStateV90, *, accepted_step_receipt,
        hles_input_receipt, current_handoff,
        accepted_face_normal_velocity_m_s, active_face,
        closure_depth_m, closure_chezy_mhalf_s, active_cell,
        cancelled: bool = False,
    ) -> HlesProducerTrialV90:
        validate_hles_producer_state_v90(parent, self.execution)
        _require(self._open_trial is None,
                 "v90 CUDA previous transaction is unfinished")
        if cancelled:
            raise HlesProducerV90Error(
                "v90 CUDA update cancelled before closure launch")
        if not parent.primed:
            raise HlesProducerV90Error(
                "v90 CUDA accepted update requires cold prime")
        if parent.accepted_updates >= MAX_ACCEPTED_UPDATES:
            raise HlesProducerV90Error(
                "v90 CUDA accepted update limit was reached")
        resident = self._resident_inputs(
            accepted_face_normal_velocity_m_s, active_face,
            closure_depth_m, closure_chezy_mhalf_s, active_cell)
        host = self._host_inputs(resident)
        dt, interval = _validate_input_receipt(
            self.execution, parent, accepted_step_receipt,
            hles_input_receipt, host, current_handoff)
        values = self._compute(parent, resident, interval)
        trial = _wrap_trial(
            self.execution, parent, accepted_step_receipt,
            hles_input_receipt, current_handoff=current_handoff,
            dt_s=dt, coefficient=values[0],
            filtered=values[1], fluctuation=values[2],
            mean_cell=values[3], fluct_cell=values[4], complete=values[5],
            boundary=values[6], detvic_active=values[7], ptke=values[8],
            hles=values[9], elder=values[10], dynamic_vicuv=values[11],
            dynamic_dicuv=values[12], backend=BACKEND_GPU)
        self._open_parent = parent.state_sha256
        self._open_trial = trial.trial_sha256
        self._open_kind = "update"
        return trial

    def _finish(self, parent, trial, *, commit: bool, kind: str):
        if kind == "prime":
            _validate_prime_trial(self.execution, parent, trial)
        else:
            _validate_trial(self.execution, parent, trial)
        _require(self._open_parent == parent.state_sha256 and
                 self._open_trial == trial.trial_sha256 and
                 self._open_kind == kind,
                 "v90 CUDA trial is not the open transaction")
        self._open_parent = self._open_trial = self._open_kind = None
        return trial.candidate if commit else parent

    def commit_prime(self, parent, trial):
        return self._finish(parent, trial, commit=True, kind="prime")

    def rollback_prime(self, parent, trial):
        return self._finish(parent, trial, commit=False, kind="prime")

    def cancel_prime(self, parent, trial):
        return self.rollback_prime(parent, trial)

    def commit(self, parent, trial):
        return self._finish(parent, trial, commit=True, kind="update")

    def rollback(self, parent, trial):
        return self._finish(parent, trial, commit=False, kind="update")

    def cancel(self, parent, trial):
        return self.rollback(parent, trial)

    def checkpoint(self, state):
        _require(self._open_trial is None,
                 "v90 CUDA checkpoint requires a committed point")
        return hles_checkpoint_v90(self.execution, state)

    def restore_checkpoint(self, document):
        _require(self._open_trial is None,
                 "v90 CUDA restore requires no open trial")
        return load_hles_checkpoint_v90(self.execution, document)


def method_identity_v90_cuda() -> dict:
    return {
        "cuda_composition_id": CUDA_COMPOSITION_ID,
        "integrator": INTEGRATOR,
        "actual_cuda_filter_reconstruction_protke_detvic": True,
        "mapping_id": MAPPING_ID,
        "nested_v82_v78_lifecycle": False,
        "automatic_cpu_fallback": False,
    }


__all__ = [
    "CUDA_COMPOSITION_ID", "ResidentHlesProducerLifecycleV90",
    "method_identity_v90_cuda",
]
