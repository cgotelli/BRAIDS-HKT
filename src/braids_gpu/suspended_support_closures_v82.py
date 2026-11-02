"""Actual-CUDA owner for the Q4/v82 FLOW 2DH support closure.

The accepted generic v78 kernels already receive a resident cell ``dicuv``
array.  V78 filled it with one profile scalar; this additive owner replaces
that resident allocation with the authenticated v82 cell field and leaves the
frozen kernel and lifecycle sources unchanged.  No CPU fallback exists.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from braids_core.sediment.multifraction_suspended_lifecycle_v78 import (
    ROLE_CUDA,
)
from braids_core.sediment.suspended_support_closures_v82 import (
    SuspendedSupportExecutionV82,
    SuspendedSupportStateV82,
    SuspendedSupportTrialV82,
    SuspendedSupportV82Error,
    _validate_trial,
    _wrap_state,
    _wrap_trial,
    load_suspended_support_checkpoint_v82,
    suspended_support_checkpoint_v82,
    validate_suspended_support_state_v82,
)
from braids_gpu.multifraction_suspended_lifecycle_v78 import (
    CUDA_THREADS_PER_BLOCK,
    ResidentMultifractionSuspendedLifecycleV78,
)


ROOT = Path(__file__).resolve().parents[1]
CUDA_COMPOSITION_ID = (
    "resident_multifraction_spatial_dicuv_flow2dh_kmax1_v82_v1"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SuspendedSupportV82Error(message)


class ResidentSuspendedSupportLifecycleV82:
    """One explicit-device, no-fallback v82 transaction owner."""

    def __init__(self, execution: SuspendedSupportExecutionV82):
        _require(isinstance(execution, SuspendedSupportExecutionV82) and
                 execution.role == ROLE_CUDA,
                 "v82 CUDA owner requires an explicit actual_cuda execution")
        self.execution = execution
        self.base = ResidentMultifractionSuspendedLifecycleV78(
            execution.base_execution)
        self.cp = self.base.cp
        field = self.cp.ascontiguousarray(
            self.cp.asarray(execution.dicuv_cell_m2_s,
                            dtype=self.cp.float64))
        _require(field.shape == (execution.cell_count,) and
                 int(field.device.id) == self.base.device_id,
                 "v82 CUDA dicuv field is not resident on the selected device")
        # The v78 adapter owns this array and every diffusion launch reads it.
        self.base.boundary.dicuv = field
        device = self.base.runtime_receipt
        source_paths = (
            ROOT/"braids_core/sediment/suspended_support_closures_v82.py",
            ROOT/"braids_gpu/suspended_support_closures_v82.py",
            ROOT/"braids_gpu/flow2dh_boundary_diffusion.cu",
        )
        self.runtime_receipt = {
            "cuda_composition_id": CUDA_COMPOSITION_ID,
            "execution_fingerprint": execution.fingerprint,
            "integrator": "implicit_fwave_midpoint",
            "backend": "actual_cuda",
            "fixed_grid": True,
            "fraction_count": execution.fraction_count,
            "automatic_cpu_fallback": False,
            "actual_cuda_kernels": True,
            "device_id": device["device_id"],
            "device_name": device["device_name"],
            "threads_per_block": CUDA_THREADS_PER_BLOCK,
            "dicuv_resident_cell_field": True,
            "horizontal_schmidt_value": 0.7,
            "per_fraction_tpsnumber_horizontal_effect": False,
            "source_sha256": {
                str(path.relative_to(ROOT)).replace("\\", "/"):
                    hashlib.sha256(path.read_bytes()).hexdigest()
                for path in source_paths
            },
            "host_readback_scope":
                "candidate authentication and checkpoint evidence only",
        }

    def begin_trial(
        self, parent: SuspendedSupportStateV82, *,
        accepted_step_receipt, cancelled: bool = False, **inputs,
    ) -> SuspendedSupportTrialV82:
        validate_suspended_support_state_v82(parent, self.execution)
        try:
            base_trial = self.base.begin_trial(
                parent.base_state,
                accepted_step_receipt=accepted_step_receipt,
                cancelled=cancelled, **inputs)
        except (TypeError, ValueError) as error:
            raise SuspendedSupportV82Error(
                "v82 CUDA lifecycle rejected the trial") from error
        effective = self.cp.asnumpy(
            self.base.boundary.effective_diffusivity)
        _require(effective.shape == (
            self.execution.fraction_count, self.execution.face_count),
            "v82 CUDA effective-diffusivity evidence shape differs")
        reference = effective[0]
        _require(np.array_equal(
            effective,
            np.broadcast_to(reference[None, :], effective.shape)),
            "v82 KMAX=1 horizontal diffusivity became constituent-dependent")
        return _wrap_trial(
            self.execution, parent, base_trial, reference)

    def _finish(self, parent: SuspendedSupportStateV82,
                trial: SuspendedSupportTrialV82, *, commit: bool
                ) -> SuspendedSupportStateV82:
        # Authenticate the outer transaction before allowing the resident
        # predecessor to commit its boundary memory.
        _validate_trial(self.execution, parent, trial)
        base_result = (self.base.commit(parent.base_state, trial.base_trial)
                       if commit else
                       self.base.rollback(parent.base_state, trial.base_trial))
        if commit:
            result = _wrap_state(self.execution, base_result)
            _require(result.state_sha256 == trial.candidate.state_sha256,
                     "v82 CUDA committed candidate differs")
            return result
        _require(base_result is parent.base_state,
                 "v82 CUDA rollback changed the parent")
        return parent

    def commit(self, parent: SuspendedSupportStateV82,
               trial: SuspendedSupportTrialV82) -> SuspendedSupportStateV82:
        return self._finish(parent, trial, commit=True)

    def rollback(self, parent: SuspendedSupportStateV82,
                 trial: SuspendedSupportTrialV82) -> SuspendedSupportStateV82:
        return self._finish(parent, trial, commit=False)

    def cancel(self, parent: SuspendedSupportStateV82,
               trial: SuspendedSupportTrialV82) -> SuspendedSupportStateV82:
        return self.rollback(parent, trial)

    def checkpoint(self, state: SuspendedSupportStateV82) -> dict:
        # The resident call enforces that no trial is open.
        base_document = self.base.checkpoint(state.base_state)
        document = suspended_support_checkpoint_v82(self.execution, state)
        _require(document["base_v78_checkpoint"] == base_document,
                 "v82 CUDA/base checkpoint differs")
        return document

    def restore_checkpoint(self, document) -> SuspendedSupportStateV82:
        state = load_suspended_support_checkpoint_v82(
            self.execution, document)
        resident = self.base.restore_checkpoint(
            document["base_v78_checkpoint"])
        _require(resident.state_sha256 == state.base_state.state_sha256,
                 "v82 CUDA/base restored state differs")
        return state


def method_identity_v82_cuda() -> dict:
    return {
        "cuda_composition_id": CUDA_COMPOSITION_ID,
        "integrator": "implicit_fwave_midpoint",
        "threads_per_block": CUDA_THREADS_PER_BLOCK,
        "automatic_cpu_fallback": False,
        "actual_cuda_kernels": True,
        "fraction_scope": "2..3 authenticated noncohesive classes",
        "dicuv_storage": "resident authenticated cell field",
        "horizontal_schmidt": 0.7,
        "tpsnumber_horizontal_effect": False,
    }


__all__ = [
    "CUDA_COMPOSITION_ID", "ResidentSuspendedSupportLifecycleV82",
    "method_identity_v82_cuda",
]
