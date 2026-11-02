"""Actual-CUDA owner for the bounded Q4/v80 two-fine-step composition."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from braids_core.sediment.multifraction_suspended_lifecycle_v78 import ROLE_CUDA
from braids_core.sediment.multifraction_suspended_amr_v79 import (
    TOPOLOGY_REFINED,
)
from braids_core.sediment.multifraction_suspended_amr_subcycle_v80 import (
    MultifractionSuspendedAMRSubcycleV80Error,
    _build_trial,
    _compose_candidate,
    _derive_fine_integrals,
    _expanded_integrals,
    _extract_fine_initial,
    _validate_receipt,
    _validate_trial,
    load_multifraction_suspended_amr_subcycle_checkpoint_v80,
    method_identity_v80,
    multifraction_suspended_amr_subcycle_checkpoint_v80,
    validate_multifraction_suspended_amr_subcycle_state_v80,
)
from braids_gpu.multifraction_suspended_lifecycle_v78 import (
    ResidentMultifractionSuspendedLifecycleV78,
)
from braids_support.gpu_fwave.amr_reflux import ResidentAMRReflux


ROOT = Path(__file__).resolve().parents[1]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MultifractionSuspendedAMRSubcycleV80Error(message)


class ResidentMultifractionSuspendedAMRSubcycleV80:
    """One atomic outer proposal containing two pending fine CUDA trials."""

    def __init__(self, execution):
        _require(execution.role == ROLE_CUDA,
                 "v80 CUDA owner requires the explicit actual_cuda execution")
        try:
            import cupy as cp
            _require(cp.cuda.runtime.getDeviceCount() >= 1,
                     "v80 actual CUDA requires a CUDA device")
        except MultifractionSuspendedAMRSubcycleV80Error:
            raise
        except Exception as error:  # pragma: no cover - environment dependent
            raise MultifractionSuspendedAMRSubcycleV80Error(
                "v80 CUDA discovery failed; CPU fallback is forbidden") from error
        self.cp = cp
        self.execution = execution
        self.device_id = int(cp.cuda.Device().id)
        self._pending_parent = None
        self._pending_trial = None
        self._pending_reflux = None
        self._pending_fine_runtimes = None
        self._pending_fine_parents = None
        self._pending_fine_trials = None
        properties = cp.cuda.runtime.getDeviceProperties(self.device_id)
        raw = properties.get("name", b"unknown")
        device_name = raw.decode(errors="replace") if isinstance(raw, bytes) \
            else str(raw)
        dependencies = (
            "braids_gpu/multifraction_suspended_lifecycle_v78.py",
            "braids_support/gpu_fwave/amr_reflux.py",
            "braids_support/gpu_fwave/amr_reflux.cu",
        )
        self.runtime_receipt = {
            "scope_id":
                "braids-q4-multifraction-suspended-amr-subcycle-v80-cuda-v1",
            "execution_fingerprint": execution.fingerprint,
            "device_id": self.device_id,
            "device_name": device_name,
            "cupy_version": cp.__version__,
            "cuda_driver_version": int(cp.cuda.runtime.driverGetVersion()),
            "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
            "integrator": "implicit_fwave_midpoint",
            "fraction_count": execution.fraction_count,
            "reflux_levels": 1,
            "fine_substeps": 2,
            "authenticated_fine_substeps": 2,
            "actual_cuda_v78_lifecycle_owners": 2,
            "actual_cuda_reflux": True,
            "automatic_cpu_fallback": False,
            "arbitrary_depth": False,
            "outer_validation_before_all_owner_commits": True,
            "host_readback_scope":
                "immutable candidate authentication, ledgers, and checkpoint evidence",
            "source_sha256": {
                name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                for name in dependencies
            },
        }

    @property
    def pending(self):
        return self._pending_trial is not None

    def _resident(self, value, shape, dtype, name):
        cp = self.cp
        _require(isinstance(value, cp.ndarray) and value.shape == shape and
                 value.dtype == dtype and value.flags.c_contiguous and
                 int(value.device.id) == self.device_id,
                 f"v80 CUDA {name} must be contiguous resident {dtype} {shape}")
        if dtype == cp.float64:
            _require(bool(cp.all(cp.isfinite(value)).item()),
                     f"v80 CUDA {name} must be finite")
        return value

    def _validate_device_chain(self, parent, bundles, dt_s):
        cp = self.cp
        _require(isinstance(bundles, (tuple, list)) and len(bundles) == 2,
                 "v80 CUDA requires exactly two fine input bundles")
        cells = np.asarray(self.execution.amr_execution.child_cells)
        expected = cp.asarray(parent.base_state.depth_m[cells])
        final_depth = None
        final_momentum = None
        half = 0.5*float(dt_s)
        for step, bundle in enumerate(bundles):
            _require(isinstance(bundle, dict) and
                     float(bundle.get("dt_s", -1.0)) == half,
                     "v80 CUDA fine timestep differs from the receipt")
            before = self._resident(
                bundle.get("depth_before_m"), (4,), cp.float64,
                f"fine step {step} depth before")
            after = self._resident(
                bundle.get("depth_after_m"), (4,), cp.float64,
                f"fine step {step} depth after")
            momentum = self._resident(
                bundle.get("endpoint_momentum_m2_s"), (2, 4), cp.float64,
                f"fine step {step} endpoint momentum")
            _require(bool(cp.array_equal(before, expected)),
                     "v80 CUDA fine hydraulic states are not temporally contiguous")
            expected = after
            final_depth, final_momentum = after, momentum
        return cp.asnumpy(final_depth), cp.asnumpy(final_momentum)

    def begin_subcycle(
        self, parent, *, accepted_subcycle_receipt,
        coarse_advective_integral_kg,
        coarse_diffusive_integral_kg,
        fine_step_inputs,
    ):
        _require(not self.pending,
                 "resolve the previous v80 CUDA transaction first")
        validate_multifraction_suspended_amr_subcycle_state_v80(
            parent, self.execution)
        _require(parent.base_state.topology == TOPOLOGY_REFINED,
                 "v80 CUDA subcycle requires the refined accepted topology")
        cp, execution = self.cp, self.execution
        ng, ns = execution.fraction_count, 3+2*execution.fraction_count
        ca_device = self._resident(
            coarse_advective_integral_kg, (2, ng), cp.float64,
            "coarse advective integral")
        cd_device = self._resident(
            coarse_diffusive_integral_kg, (2, ng), cp.float64,
            "coarse diffusive integral")
        ca, cd = cp.asnumpy(ca_device), cp.asnumpy(cd_device)
        dt_s = float(accepted_subcycle_receipt.get("dt_s", float("nan")))
        receipt = _validate_receipt(
            execution, parent, accepted_subcycle_receipt, dt_s, ca, cd)
        final_depth, final_momentum = self._validate_device_chain(
            parent, fine_step_inputs, dt_s)
        initial = _extract_fine_initial(execution, parent)
        runtime0 = ResidentMultifractionSuspendedLifecycleV78(
            execution.fine_lifecycle_execution)
        runtime1 = ResidentMultifractionSuspendedLifecycleV78(
            execution.fine_lifecycle_execution)
        trial0 = trial1 = reflux = None
        try:
            trial0 = runtime0.begin_trial(
                initial,
                accepted_step_receipt=receipt["fine_step_receipts"][0],
                **fine_step_inputs[0])
            # A second private owner consumes the immutable first candidate.
            # Neither owner publishes before the outer v80 authentication.
            trial1 = runtime1.begin_trial(
                trial0.candidate,
                accepted_step_receipt=receipt["fine_step_receipts"][1],
                **fine_step_inputs[1])
            trials = (trial0, trial1)
            fa, fd = _derive_fine_integrals(execution, trials, dt_s)
            candidate, before, post, expected_after, expected_register = \
                _compose_candidate(
                    execution, parent, receipt, initial, trials, ca, cd,
                    fa, fd, final_depth, final_momentum)
            coarse, fine = _expanded_integrals(execution, ca, cd, fa, fd)
            area = np.asarray([
                execution.amr_execution.lifecycle_execution.base_execution
                .cell_area_m2[index]
                for index in execution.amr_execution.reflux_cells],
                dtype=np.float64)
            reflux = ResidentAMRReflux(
                cp.ascontiguousarray(cp.asarray(post)), area, [0], [0, 2],
                [0, 1], [0, 1], 2, grains=ng)
            after_device, register_device, status = reflux.propose(
                cp.ascontiguousarray(cp.asarray(coarse)),
                cp.ascontiguousarray(cp.asarray(fine)))
            if bool(cp.any(status != 0).item()):
                raise MultifractionSuspendedAMRSubcycleV80Error(
                    "v80 CUDA reflux requires unsupported repair")
            after = cp.asnumpy(after_device)
            register = cp.asnumpy(register_device)
            _require(np.array_equal(after, expected_after) and
                     np.array_equal(register, expected_register),
                     "v80 CUDA reflux differs from the isolated CPU oracle")
            trial = _build_trial(
                execution, parent, candidate, receipt, initial, trials,
                ca, cd, fa, fd, register, before, post, after, ROLE_CUDA)
        except Exception:
            if reflux is not None and reflux.pending:
                reflux.rollback()
            if trial1 is not None and runtime1._open_trial is not None:
                runtime1.rollback(trial0.candidate, trial1)
            if trial0 is not None and runtime0._open_trial is not None:
                runtime0.rollback(initial, trial0)
            raise
        self._pending_parent = parent.state_sha256
        self._pending_trial = trial.trial_sha256
        self._pending_reflux = reflux
        self._pending_fine_runtimes = (runtime0, runtime1)
        self._pending_fine_parents = (initial, trial0.candidate)
        self._pending_fine_trials = trials
        return trial

    def _finish(self, parent, trial, *, commit):
        _require(self.pending and
                 self._pending_parent == parent.state_sha256 and
                 self._pending_trial == trial.trial_sha256,
                 "v80 CUDA transaction parent or trial differs")
        # This validation authenticates the full outer object, both nested
        # v78 trials/receipts, their exact flux integrals, the candidate, and
        # the register before any resident owner is allowed to publish.
        _validate_trial(self.execution, parent, trial)
        runtime0, runtime1 = self._pending_fine_runtimes
        parent0, parent1 = self._pending_fine_parents
        trial0, trial1 = self._pending_fine_trials
        try:
            if commit:
                self._pending_reflux.commit()
                runtime0.commit(parent0, trial0)
                runtime1.commit(parent1, trial1)
                result = trial.candidate
            else:
                self._pending_reflux.rollback()
                runtime0.rollback(parent0, trial0)
                runtime1.rollback(parent1, trial1)
                result = parent
        finally:
            self._pending_parent = None
            self._pending_trial = None
            self._pending_reflux = None
            self._pending_fine_runtimes = None
            self._pending_fine_parents = None
            self._pending_fine_trials = None
        return result

    def commit(self, parent, trial):
        return self._finish(parent, trial, commit=True)

    def rollback(self, parent, trial):
        return self._finish(parent, trial, commit=False)

    def cancel(self, parent, trial):
        return self.rollback(parent, trial)

    def checkpoint(self, state):
        _require(not self.pending,
                 "v80 CUDA checkpoint requires an accepted transaction boundary")
        return multifraction_suspended_amr_subcycle_checkpoint_v80(
            self.execution, state)

    def restore_checkpoint(self, document):
        _require(not self.pending,
                 "v80 CUDA restore requires an accepted transaction boundary")
        return load_multifraction_suspended_amr_subcycle_checkpoint_v80(
            self.execution, document)


def method_identity_v80_cuda() -> dict:
    return {
        **method_identity_v80(),
        "cuda_owner": "resident_multifraction_suspended_amr_subcycle_v80",
        "fine_lifecycle_owners": 2,
        "reflux_kernel": "amr_fine_register/amr_reflux_coarse",
        "outer_validation_before_all_owner_commits": True,
        "actual_cuda_required": True,
        "automatic_cpu_fallback": False,
    }


__all__ = [
    "ResidentMultifractionSuspendedAMRSubcycleV80",
    "method_identity_v80_cuda",
]
