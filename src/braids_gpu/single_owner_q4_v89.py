"""Actual-CUDA owner for the Q4/v89 single-owner ABI.

The accepted resident v78 implementation owns boundary memory, FCT,
per-trial spatial diffusion, SettleFrm=2, exchange, wet/dry cleanup and the
one canonical candidate.  V89 changes only the orchestration: capacity and
lagged ``dicuv`` come from authenticated provider receipts, and no v84/v86/
v87/v88 lifecycle wrapper is nested around the resident owner.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

import numpy as np

from braids_core.sediment.single_owner_q4_v89 import (
    BACKEND_GPU,
    MAX_ACCEPTED_UPDATES,
    ProviderBundleV89,
    SingleOwnerExecutionV89,
    SingleOwnerQ4V89Error,
    SingleOwnerStateV89,
    SingleOwnerTrialV89,
    _build_trial,
    _validate_trial,
    load_single_owner_checkpoint_v89,
    single_owner_checkpoint_v89,
    validate_provider_bundle_v89,
    validate_single_owner_state_v89,
)
from braids_core.sediment.multifraction_suspended_lifecycle_v78 import (
    ROLE_CUDA,
)
from braids_gpu.multifraction_suspended_lifecycle_v78 import (
    CUDA_THREADS_PER_BLOCK,
    ResidentMultifractionSuspendedLifecycleV78,
)


ROOT = Path(__file__).resolve().parents[1]
CUDA_COMPOSITION_ID = "resident_q4_single_owner_provider_abi_v89_v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SingleOwnerQ4V89Error(message)


class ResidentSingleOwnerQ4LifecycleV89:
    """One explicit-device, no-fallback, one-update v89 owner."""

    def __init__(self, execution: SingleOwnerExecutionV89):
        _require(isinstance(execution, SingleOwnerExecutionV89) and
                 execution.backend == BACKEND_GPU,
                 "v89 CUDA owner requires explicit GPU execution")
        self.execution = execution
        try:
            self.base = ResidentMultifractionSuspendedLifecycleV78(
                execution.base_execution)
        except Exception as error:
            raise SingleOwnerQ4V89Error(
                "v89 CUDA resident v78 construction failed") from error
        self.cp = self.base.cp
        self._readback = self.cp.asnumpy
        self._open_parent: str | None = None
        self._open_trial: str | None = None
        paths = (
            ROOT/"braids_core/sediment/single_owner_q4_v89.py",
            ROOT/"braids_gpu/single_owner_q4_v89.py",
            ROOT/"braids_gpu/multifraction_suspended_lifecycle_v78.py",
            ROOT/"braids_gpu/flow2dh_boundary_diffusion.cu",
            ROOT/"braids_gpu/law11_soursin_exchange.cu",
        )
        base_receipt = self.base.runtime_receipt
        self.runtime_receipt = {
            "cuda_composition_id": CUDA_COMPOSITION_ID,
            "execution_fingerprint": execution.fingerprint,
            "base_v78_execution_fingerprint":
                execution.base_execution.fingerprint,
            "integrator": "implicit_fwave_midpoint",
            "backend": "actual_cuda",
            "actual_cuda_kernels": True,
            "device_id": base_receipt["device_id"],
            "device_name": base_receipt["device_name"],
            "threads_per_block": CUDA_THREADS_PER_BLOCK,
            "automatic_cpu_fallback": False,
            "canonical_v78_runtime_count": 1,
            "accepted_update_limit": MAX_ACCEPTED_UPDATES,
            "per_trial_dicuv_receipt": True,
            "source_sha256": {
                str(path.relative_to(ROOT)).replace("\\", "/"):
                    hashlib.sha256(path.read_bytes()).hexdigest()
                for path in paths
            },
            "host_readback_scope":
                "actual device publication guard and candidate authentication",
        }

    def begin_trial(self, parent: SingleOwnerStateV89, *,
                    accepted_step_receipt: Mapping[str, object],
                    providers: ProviderBundleV89,
                    cancelled: bool = False, **inputs) -> SingleOwnerTrialV89:
        validate_single_owner_state_v89(parent, self.execution)
        _require(self._open_trial is None,
                 "v89 CUDA previous trial is unfinished")
        if cancelled:
            raise SingleOwnerQ4V89Error(
                "v89 CUDA cancelled before any kernel launch")
        if parent.accepted_updates >= MAX_ACCEPTED_UPDATES:
            raise SingleOwnerQ4V89Error(
                "v89 CUDA accepted update limit was reached")
        validate_provider_bundle_v89(
            providers, self.execution, parent, accepted_step_receipt)
        if "suspended_transport_capacity_kg_m_s" in inputs:
            raise SingleOwnerQ4V89Error(
                "v89 CUDA capacity must come only from the Q3 receipt")
        h0 = inputs.get("depth_before_m")
        h1 = inputs.get("depth_after_m")
        _require(isinstance(h0, self.cp.ndarray) and
                 isinstance(h1, self.cp.ndarray),
                 "v89 CUDA depth fields must be resident")
        dry = self.execution.base_execution.base_execution.dry_tolerance_m
        _require(not bool(self.cp.any((h0 <= dry) | (h1 <= dry)).item()),
                 "v89 moving wet/dry composition is staged for v97")
        # These are the only per-update fields not owned by the hydraulic
        # input mapping.  Both transfers are explicit and no CPU path exists.
        dicuv = self.cp.ascontiguousarray(self.cp.asarray(
            providers.closure.used_total_dicuv_m2_s,
            dtype=self.cp.float64))
        capacity = self.cp.ascontiguousarray(self.cp.asarray(
            providers.q3.suspended_capacity_kg_m_s,
            dtype=self.cp.float64))
        _require(dicuv.shape == (self.execution.cell_count,) and
                 capacity.shape == (self.execution.fraction_count,
                                    self.execution.cell_count),
                 "v89 CUDA provider device shapes differ")
        self.base.boundary.dicuv = dicuv
        base_trial = None
        try:
            base_trial = self.base.begin_trial(
                parent.base_state,
                accepted_step_receipt=accepted_step_receipt,
                suspended_transport_capacity_kg_m_s=capacity,
                cancelled=False, **inputs)
            # This is an actual device-to-host publication readback.  The
            # injectable hook proves that a failure rolls resident boundary
            # memory back before any outer trial is published.
            clock = self._readback(self.base.boundary.trial_clock)
            _require(clock.shape == (
                self.execution.fraction_count, self.execution.face_count) and
                np.isfinite(clock).all(),
                "v89 CUDA publication readback differs")
            trial = _build_trial(
                self.execution, parent, accepted_step_receipt, providers,
                base_trial, ROLE_CUDA)
        except Exception as error:
            if base_trial is not None and self.base._open_trial is not None:
                try:
                    self.base.rollback(parent.base_state, base_trial)
                except Exception as rollback_error:
                    raise SingleOwnerQ4V89Error(
                        "v89 CUDA publication and rollback both failed") \
                        from rollback_error
            if isinstance(error, SingleOwnerQ4V89Error):
                raise
            raise SingleOwnerQ4V89Error(
                "v89 CUDA provider publication failed") from error
        self._open_parent = parent.state_sha256
        self._open_trial = trial.trial_sha256
        return trial

    def _finish(self, parent: SingleOwnerStateV89,
                trial: SingleOwnerTrialV89, *, commit: bool
                ) -> SingleOwnerStateV89:
        _validate_trial(self.execution, parent, trial)
        _require(self._open_parent == parent.state_sha256 and
                 self._open_trial == trial.trial_sha256,
                 "v89 CUDA trial is not the open transaction")
        base = (self.base.commit(parent.base_state, trial.base_trial)
                if commit else
                self.base.rollback(parent.base_state, trial.base_trial))
        result = trial.candidate if commit else parent
        if commit:
            _require(base.state_sha256 == result.base_state.state_sha256,
                     "v89 CUDA committed state differs")
        else:
            _require(base is parent.base_state,
                     "v89 CUDA rollback changed parent")
        self._open_parent = self._open_trial = None
        return result

    def commit(self, parent, trial):
        return self._finish(parent, trial, commit=True)

    def rollback(self, parent, trial):
        return self._finish(parent, trial, commit=False)

    def cancel(self, parent, trial):
        return self.rollback(parent, trial)

    def checkpoint(self, state: SingleOwnerStateV89) -> dict:
        _require(self._open_trial is None,
                 "v89 CUDA checkpoint requires a committed point")
        base = self.base.checkpoint(state.base_state)
        document = single_owner_checkpoint_v89(self.execution, state)
        _require(document["base_v78_checkpoint"] == base,
                 "v89 CUDA/base checkpoint differs")
        return document

    def restore_checkpoint(self, document) -> SingleOwnerStateV89:
        _require(self._open_trial is None,
                 "v89 CUDA restart cannot replace an open trial")
        state = load_single_owner_checkpoint_v89(self.execution, document)
        base = self.base.restore_checkpoint(document["base_v78_checkpoint"])
        _require(base.state_sha256 == state.base_state.state_sha256,
                 "v89 CUDA/base restored state differs")
        return state


def method_identity_v89_cuda() -> dict:
    return {
        "cuda_composition_id": CUDA_COMPOSITION_ID,
        "integrator": "implicit_fwave_midpoint",
        "actual_cuda_kernels": True,
        "threads_per_block": CUDA_THREADS_PER_BLOCK,
        "canonical_v78_runtime_count": 1,
        "per_trial_dicuv_receipt": True,
        "automatic_cpu_fallback": False,
        "accepted_update_limit": MAX_ACCEPTED_UPDATES,
        "staged_consumers": [
            "vicuv_momentum", "new_closure_commit", "dynamic_fluid",
            "q3_bed_transport",
        ],
    }


__all__ = [
    "CUDA_COMPOSITION_ID", "ResidentSingleOwnerQ4LifecycleV89",
    "method_identity_v89_cuda",
]
