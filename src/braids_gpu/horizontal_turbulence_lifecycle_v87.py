"""Actual-CUDA owner for the bounded Q4/v87 turbulence lifecycle."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from braids_core.sediment.horizontal_turbulence_lifecycle_v87 import (
    CALL_ORDER,
    INTEGRATOR,
    DetvicFieldV87,
    HorizontalTurbulenceExecutionV87,
    HorizontalTurbulenceStateV87,
    HorizontalTurbulenceTrialV87,
    HorizontalTurbulenceV87Error,
    _closure_inputs,
    _validate_detvic_receipt,
    _validate_trial,
    _v82_context,
    _wrap_trial,
    horizontal_turbulence_checkpoint_v87,
    load_horizontal_turbulence_checkpoint_v87,
    validate_horizontal_turbulence_state_v87,
)
from braids_core.sediment.multifraction_suspended_lifecycle_v78 import (
    ROLE_CUDA,
)
from braids_gpu.multifraction_suspended_lifecycle_v78 import (
    CUDA_OPTIONS,
    CUDA_THREADS_PER_BLOCK,
)
from braids_gpu.suspended_support_closures_v82 import (
    ResidentSuspendedSupportLifecycleV82,
)


ROOT = Path(__file__).resolve().parents[1]
CUDA_SOURCE = ROOT/"braids_gpu/horizontal_turbulence_lifecycle_v87.cu"
CUDA_COMPOSITION_ID = (
    "resident_flow2dh_detvic_previous_accepted_dicuv_lifecycle_v87_v1"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HorizontalTurbulenceV87Error(message)


class ResidentHorizontalTurbulenceLifecycleV87:
    """One explicit-device, no-fallback v87 transaction owner."""

    def __init__(self, execution: HorizontalTurbulenceExecutionV87):
        _require(isinstance(execution, HorizontalTurbulenceExecutionV87) and
                 execution.role == ROLE_CUDA,
                 "v87 CUDA owner requires an explicit actual_cuda execution")
        import cupy as cp

        self.execution = execution
        self.cp = cp
        self.device_id = int(cp.cuda.Device().id)
        try:
            module = cp.RawModule(
                code=CUDA_SOURCE.read_text(encoding="utf-8"),
                options=CUDA_OPTIONS,
                name_expressions=("v87_detvic_current_only",))
            self.kernel = module.get_function("v87_detvic_current_only")
        except Exception as error:
            raise HorizontalTurbulenceV87Error(
                "v87 CUDA detvic kernel compilation failed") from error
        props = cp.cuda.runtime.getDeviceProperties(self.device_id)
        name = props["name"]
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="replace")
        sources = (
            ROOT/"braids_core/sediment/horizontal_turbulence_lifecycle_v87.py",
            ROOT/"braids_gpu/horizontal_turbulence_lifecycle_v87.py",
            CUDA_SOURCE,
            ROOT/"braids_core/sediment/suspended_support_closures_v82.py",
            ROOT/"braids_gpu/suspended_support_closures_v82.py",
        )
        self.runtime_receipt = {
            "cuda_composition_id": CUDA_COMPOSITION_ID,
            "execution_fingerprint": execution.fingerprint,
            "integrator": INTEGRATOR,
            "backend": "actual_cuda",
            "device_id": self.device_id,
            "device_name": str(name),
            "threads_per_block": CUDA_THREADS_PER_BLOCK,
            "actual_cuda_detvic_kernel": True,
            "actual_cuda_v82_lifecycle": True,
            "automatic_cpu_fallback": False,
            "kmax": 1,
            "call_order": CALL_ORDER,
            "dynamic_field_lag_accepted_updates": 1,
            "horizontal_schmidt": 0.7,
            "source_sha256": {
                str(path.relative_to(ROOT)).replace("\\", "/"):
                    hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sources
            },
            "host_readback_scope": (
                "strict detvic input/output receipt authentication and "
                "candidate/checkpoint evidence only"),
        }
        self._open_parent_sha256: str | None = None
        self._open_trial_sha256: str | None = None
        self._open_v82_runtime = None
        self._open_v82_parent = None

    def _device(self, value, shape, dtype, name):
        cp = self.cp
        _require(isinstance(value, cp.ndarray) and value.shape == shape and
                 value.dtype == dtype and value.flags.c_contiguous and
                 int(value.device.id) == self.device_id,
                 f"v87 CUDA {name} must be contiguous resident {dtype} {shape}")
        return value

    def begin_trial(
        self, parent: HorizontalTurbulenceStateV87, *,
        accepted_step_receipt, detvic_input_receipt,
        closure_depth_m, closure_mean_velocity_m_s,
        closure_chezy_mhalf_s, closure_ptke_s2, closure_active_cell,
        cancelled: bool = False, **inputs,
    ) -> HorizontalTurbulenceTrialV87:
        cp = self.cp
        validate_horizontal_turbulence_state_v87(parent, self.execution)
        _require(self._open_trial_sha256 is None,
                 "v87 CUDA previous trial is unfinished")
        if cancelled:
            raise HorizontalTurbulenceV87Error(
                "v87 CUDA update cancelled before candidate publication")
        nc = self.execution.cell_count
        depth = self._device(
            closure_depth_m, (nc,), cp.float64, "closure depth")
        velocity = self._device(
            closure_mean_velocity_m_s, (2, nc), cp.float64,
            "closure mean velocity")
        chezy = self._device(
            closure_chezy_mhalf_s, (nc,), cp.float64, "closure Chezy")
        ptke = self._device(
            closure_ptke_s2, (nc,), cp.float64, "closure ptke")
        active = self._device(
            closure_active_cell, (nc,), cp.int32, "closure active cell")

        # Strictly bind the resident arrays to the CPU-created hydraulic
        # handoff receipt before opening the nested v82 transaction.
        try:
            host_arrays = _closure_inputs(
                self.execution, depth_m=cp.asnumpy(depth),
                mean_velocity_m_s=cp.asnumpy(velocity),
                chezy_mhalf_s=cp.asnumpy(chezy), ptke_s2=cp.asnumpy(ptke),
                active_cell=cp.asnumpy(active))
        except Exception as error:
            raise HorizontalTurbulenceV87Error(
                "v87 CUDA detvic input readback failed") from error
        _validate_detvic_receipt(
            self.execution, parent, detvic_input_receipt,
            accepted_step_receipt, host_arrays)

        v82_execution, v82_parent, used_vic, used_dic = _v82_context(
            self.execution, parent)
        nested = ResidentSuspendedSupportLifecycleV82(v82_execution)
        try:
            base_trial = nested.begin_trial(
                v82_parent, accepted_step_receipt=accepted_step_receipt,
                **inputs)
        except (TypeError, ValueError) as error:
            raise HorizontalTurbulenceV87Error(
                "v87 accepted actual-CUDA v82 lifecycle rejected the trial") \
                from error

        hles = cp.empty(nc, dtype=cp.float64)
        elder = cp.empty(nc, dtype=cp.float64)
        dynamic_vic = cp.empty(nc, dtype=cp.float64)
        dynamic_dic = cp.empty(nc, dtype=cp.float64)
        status = cp.empty(nc, dtype=cp.int32)
        area = cp.ascontiguousarray(cp.asarray(
            self.execution.base_execution.base_execution.cell_area_m2,
            dtype=cp.float64))
        try:
            blocks = ((nc+CUDA_THREADS_PER_BLOCK-1)//CUDA_THREADS_PER_BLOCK,)
            self.kernel(blocks, (CUDA_THREADS_PER_BLOCK,), (
                np.int32(nc), np.int32(self.execution.hles_enabled),
                np.int32(self.execution.elder_enabled),
                np.float64(self.execution.gamma),
                np.float64(self.execution.sigmat),
                np.float64(self.execution.flp),
                np.float64(self.execution.vicmol_m2_s),
                np.float64(self.execution.dicmol_m2_s),
                np.float64(self.execution.von_karman),
                np.float64(self.execution.gravity_m_s2),
                np.float64(
                    self.execution.base_execution.base_execution.dry_tolerance_m),
                area, depth, velocity, chezy, ptke, active, hles, elder,
                dynamic_vic, dynamic_dic, status))
            if bool(cp.any(status != 0).item()):
                raise HorizontalTurbulenceV87Error(
                    "v87 CUDA detvic kernel status rejected the trial")
            field = DetvicFieldV87(
                cp.asnumpy(hles), cp.asnumpy(elder),
                cp.asnumpy(dynamic_vic), cp.asnumpy(dynamic_dic))
            trial = _wrap_trial(
                self.execution, parent, v82_execution, base_trial,
                used_vic, used_dic, field,
                str(detvic_input_receipt["fingerprint"]))
        except Exception as error:
            try:
                nested.rollback(v82_parent, base_trial)
            except Exception:
                pass
            if isinstance(error, HorizontalTurbulenceV87Error):
                raise
            raise HorizontalTurbulenceV87Error(
                "v87 CUDA detvic output readback failed") from error

        self._open_parent_sha256 = parent.state_sha256
        self._open_trial_sha256 = trial.trial_sha256
        self._open_v82_runtime = nested
        self._open_v82_parent = v82_parent
        return trial

    def _finish(self, parent: HorizontalTurbulenceStateV87,
                trial: HorizontalTurbulenceTrialV87, *, commit: bool
                ) -> HorizontalTurbulenceStateV87:
        _require(self._open_parent_sha256 == parent.state_sha256 and
                 self._open_trial_sha256 == trial.trial_sha256 and
                 self._open_v82_runtime is not None and
                 self._open_v82_parent is not None,
                 "v87 CUDA trial is not the open transaction")
        # Authenticate all outer arrays/digests before nested boundary memory
        # may be committed or rolled back.
        v82_execution, _ = _validate_trial(self.execution, parent, trial)
        _require(v82_execution.fingerprint ==
                 self._open_v82_runtime.execution.fingerprint,
                 "v87 CUDA nested execution differs")
        nested = self._open_v82_runtime
        v82_parent = self._open_v82_parent
        try:
            result = (nested.commit(v82_parent, trial.base_trial)
                      if commit else
                      nested.rollback(v82_parent, trial.base_trial))
            if commit:
                _require(result.base_state.state_sha256 ==
                         trial.candidate.base_state.state_sha256,
                         "v87 CUDA committed candidate differs")
                return trial.candidate
            _require(result is v82_parent,
                     "v87 CUDA rollback changed the nested parent")
            return parent
        finally:
            self._open_parent_sha256 = None
            self._open_trial_sha256 = None
            self._open_v82_runtime = None
            self._open_v82_parent = None

    def commit(self, parent: HorizontalTurbulenceStateV87,
               trial: HorizontalTurbulenceTrialV87
               ) -> HorizontalTurbulenceStateV87:
        return self._finish(parent, trial, commit=True)

    def rollback(self, parent: HorizontalTurbulenceStateV87,
                 trial: HorizontalTurbulenceTrialV87
                 ) -> HorizontalTurbulenceStateV87:
        return self._finish(parent, trial, commit=False)

    def cancel(self, parent: HorizontalTurbulenceStateV87,
               trial: HorizontalTurbulenceTrialV87
               ) -> HorizontalTurbulenceStateV87:
        return self.rollback(parent, trial)

    def checkpoint(self, state: HorizontalTurbulenceStateV87) -> dict:
        _require(self._open_trial_sha256 is None,
                 "v87 CUDA checkpoint requires a committed point")
        return horizontal_turbulence_checkpoint_v87(self.execution, state)

    def restore_checkpoint(self, document) -> HorizontalTurbulenceStateV87:
        _require(self._open_trial_sha256 is None,
                 "v87 CUDA restore requires no open trial")
        return load_horizontal_turbulence_checkpoint_v87(
            self.execution, document)


def method_identity_v87_cuda() -> dict:
    return {
        "cuda_composition_id": CUDA_COMPOSITION_ID,
        "integrator": INTEGRATOR,
        "actual_cuda_detvic_kernel": True,
        "actual_cuda_v82_lifecycle": True,
        "automatic_cpu_fallback": False,
        "call_order": CALL_ORDER,
        "dynamic_field_lag_accepted_updates": 1,
        "kmax": 1,
        "horizontal_schmidt": 0.7,
    }


__all__ = [
    "CUDA_COMPOSITION_ID", "ResidentHorizontalTurbulenceLifecycleV87",
    "method_identity_v87_cuda",
]
