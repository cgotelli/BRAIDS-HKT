"""Actual-CUDA Q4/v91 consumer bound to the real midpoint residual."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from braids_core.sediment.hles_producer_v90 import (
    BACKEND_GPU,
    CurrentClosureHandoffV90,
    HlesProducerStateV90,
    validate_current_closure_handoff_v90,
    validate_hles_producer_state_v90,
)
from braids_core.sediment.shared_turbulence_consumers_v91 import (
    HYDRAULIC_RECEIPT_SCOPE_ID,
    INTEGRATOR,
    SharedTurbulenceExecutionV91,
    SharedTurbulenceStateV91,
    SharedTurbulenceTrialV91,
    SharedTurbulenceV91Error,
    _validate_trial,
    _wrap_trial,
    apply_chkvic_used_total_v91,
    execution_document_v91,
    load_shared_turbulence_checkpoint_v91,
    make_hydraulic_turbulence_receipt_v91,
    momentum_diffusion_rate_v91,
    shared_turbulence_checkpoint_v91,
    validate_shared_turbulence_state_v91,
)
from braids_core.sediment.suspended_support_closures_v82 import (
    _wrap_state as _wrap_v82_state,
    resolve_suspended_support_execution_v82,
)
from braids_gpu.multifraction_suspended_lifecycle_v78 import (
    CUDA_OPTIONS,
    CUDA_THREADS_PER_BLOCK,
)
from braids_gpu.suspended_support_closures_v82 import (
    ResidentSuspendedSupportLifecycleV82,
)


ROOT = Path(__file__).resolve().parents[1]
CUDA_SOURCE = ROOT/"braids_gpu/shared_turbulence_consumers_v91.cu"
CUDA_COMPOSITION_ID = (
    "actual_cuda_implicit_midpoint_shared_turbulence_consumers_v91_v1")
KERNEL_NAMES = (
    "v91_chkvic_used_total", "v91_momentum_face_flux",
    "v91_momentum_gather_add",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SharedTurbulenceV91Error(message)


def _cancelled(value) -> bool:
    if callable(value):
        return bool(value())
    if type(value) is bool:
        return value
    raise TypeError("cancelled must be Boolean or callable")


class ResidentSharedTurbulenceConsumersV91:
    """One selected CUDA owner; no automatic CPU or post-source fallback."""

    is_v91_shared_turbulence_consumer = True

    def __init__(self, execution: SharedTurbulenceExecutionV91):
        _require(isinstance(execution, SharedTurbulenceExecutionV91) and
                 execution.backend == BACKEND_GPU,
                 "v91 CUDA owner requires the explicit GPU execution")
        execution_document_v91(execution)
        import cupy as cp

        self.execution, self.cp = execution, cp
        self.device_id = int(cp.cuda.Device().id)
        try:
            module = cp.RawModule(
                code=CUDA_SOURCE.read_text(encoding="utf-8"),
                options=CUDA_OPTIONS, name_expressions=KERNEL_NAMES)
            self.kernels = {name: module.get_function(name)
                            for name in KERNEL_NAMES}
        except Exception as error:
            raise SharedTurbulenceV91Error(
                "v91 CUDA kernels failed to compile") from error

        base = execution.base_execution.base_execution
        self.nc, self.nf = execution.cell_count, execution.face_count
        self.owner = cp.asarray(base.owner, dtype=cp.int32)
        self.neighbour = cp.asarray(base.neighbour, dtype=cp.int32)
        self.length = cp.asarray(base.face_length_m, dtype=cp.float64)
        self.distance = cp.asarray(base.centre_distance_m, dtype=cp.float64)
        self.area = cp.asarray(base.cell_area_m2, dtype=cp.float64)
        self.offset = cp.asarray(base.cell_face_offset, dtype=cp.int32)
        self.faces = cp.asarray(base.cell_faces, dtype=cp.int32)
        self.signs = cp.asarray(base.cell_face_sign, dtype=cp.int32)
        self.width = cp.asarray(base.rectangle_width_m, dtype=cp.float64)
        self.background_vic = cp.asarray(
            execution.hles_execution.closure_execution.background.vicuv_m2_s,
            dtype=cp.float64)
        self.background_dic = cp.asarray(
            execution.hles_execution.closure_execution.background.dicuv_m2_s,
            dtype=cp.float64)
        self.dry = np.float64(base.dry_tolerance_m)
        self.vicmax = cp.empty(self.nc, dtype=cp.float64)
        self.dicmax = cp.empty(self.nc, dtype=cp.float64)
        self.capped_vic = cp.empty(self.nc, dtype=cp.float64)
        self.capped_dic = cp.empty(self.nc, dtype=cp.float64)
        self.cap_vic_flag = cp.empty(self.nc, dtype=cp.int32)
        self.cap_dic_flag = cp.empty(self.nc, dtype=cp.int32)
        self.cap_status = cp.empty(self.nc, dtype=cp.int32)
        self.face_flux = cp.empty((2, self.nf), dtype=cp.float64)
        self.face_status = cp.empty(self.nf, dtype=cp.int32)
        self.closure_rate = cp.empty((3, self.nc), dtype=cp.float64)
        self.cell_status = cp.empty(self.nc, dtype=cp.int32)
        props = cp.cuda.runtime.getDeviceProperties(self.device_id)
        name = props.get("name", b"unknown")
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="replace")
        paths = (
            ROOT/"braids_core/sediment/shared_turbulence_consumers_v91.py",
            Path(__file__).resolve(), CUDA_SOURCE,
            ROOT/"braids_gpu/implicit_midpoint_fwave.py",
        )
        self.runtime_receipt = {
            "cuda_composition_id": CUDA_COMPOSITION_ID,
            "execution_fingerprint": execution.fingerprint,
            "integrator": INTEGRATOR,
            "backend": "actual_cuda",
            "device_id": self.device_id,
            "device_name": str(name),
            "threads_per_block": CUDA_THREADS_PER_BLOCK,
            "actual_cuda_chkvic": True,
            "actual_cuda_momentum_face_operator": True,
            "true_full_rate_hook": True,
            "automatic_cpu_fallback": False,
            "duplicate_post_source_count": 0,
            "source_sha256": {
                str(path.relative_to(ROOT)).replace("\\", "/"):
                    hashlib.sha256(path.read_bytes()).hexdigest()
                for path in paths
            },
        }
        self.kernel_launch_counts = {"chkvic": 0, "momentum_face": 0,
                                     "momentum_gather": 0, "v82": 0}
        self._controller = None
        self._parent = self._parent_hles = self._handoff = self._cap = None
        self._dt = self._time = None
        self._residual_evaluations = 0
        self._pending = None
        self._scalar_runtime = None
        self._open = False
        self._cancelled_callback = False
        self.last_accepted_receipt = None

    def _launch(self, name, size, args):
        if size:
            self.kernels[name](
                ((size+CUDA_THREADS_PER_BLOCK-1)//CUDA_THREADS_PER_BLOCK,),
                (CUDA_THREADS_PER_BLOCK,), args)

    def _device(self, value, shape, dtype, name):
        cp = self.cp
        _require(isinstance(value, cp.ndarray) and value.shape == shape and
                 value.dtype == dtype and value.flags.c_contiguous and
                 int(value.device.id) == self.device_id,
                 f"v91 CUDA {name} must be contiguous resident {dtype} {shape}")
        return value

    def bind_controller(self, controller) -> None:
        """Bind once to the actual controller after checking exact geometry."""
        _require(self._controller is None or self._controller is controller,
                 "v91 CUDA owner is already bound to another controller")
        cp = self.cp
        _require(getattr(controller, "nc", None) == self.nc and
                 getattr(controller, "nf", None) == self.nf and
                 getattr(controller, "cp", None) is cp,
                 "v91 controller dimensions/backend differ")
        _require(bool(cp.all(controller.active == 1).item()),
                 "v91 CUDA owner requires every fixed-grid cell active")
        _require(getattr(controller.solver, "diffusion", None) is None and
                 float(controller.solver.data.get(
                     "horizontal_viscosity", 0.0)) == 0.0,
                 "v91 replaces, rather than duplicates, post-source viscosity")
        checks = (
            (controller.owner, self.owner, "owner"),
            (controller.neighbour, self.neighbour, "neighbour"),
            (controller.length, self.length, "length"),
            (controller.area, self.area, "area"),
            (controller.offset, self.offset, "offset"),
            (controller.faces, self.faces, "faces"),
            (controller.signs, self.signs, "signs"),
        )
        for actual, expected, name in checks:
            _require(bool(cp.array_equal(actual, expected).item()),
                     "v91 controller geometry differs: " + name)
        self._controller = controller

    def begin_hydraulic_trial(
        self, parent: SharedTurbulenceStateV91,
        parent_hles: HlesProducerStateV90,
        handoff: CurrentClosureHandoffV90, *, dt_s: float, time_s: float,
        cancelled=False,
    ) -> None:
        _require(self._controller is not None,
                 "v91 CUDA owner must be bound to the live controller")
        _require(not self._open and self._pending is None,
                 "v91 CUDA previous trial is unfinished")
        validate_shared_turbulence_state_v91(parent, self.execution)
        validate_hles_producer_state_v90(
            parent_hles, self.execution.hles_execution)
        validate_current_closure_handoff_v90(
            handoff, self.execution.hles_execution, parent_hles)
        _require(not _cancelled(cancelled),
                 "v91 CUDA trial cancelled before any launch")
        _require(parent.accepted_updates < 4 and
                 parent.accepted_updates == parent_hles.accepted_updates and
                 parent.time_s == parent_hles.time_s == float(time_s),
                 "v91 CUDA parent clocks or update bound differ")
        _require(np.isfinite(dt_s) and dt_s > 0.0,
                 "v91 CUDA timestep must be finite and positive")
        lagged_vic = self.cp.asarray(
            handoff.accepted_dynamic_vicuv_m2_s, dtype=self.cp.float64)
        lagged_dic = self.cp.asarray(
            handoff.accepted_dynamic_dicuv_m2_s, dtype=self.cp.float64)
        self._cancelled_callback = cancelled
        try:
            self.kernel_launch_counts["chkvic"] += 1
            self._launch("v91_chkvic_used_total", self.nc, (
                np.int64(self.nc), np.float64(dt_s), self.width,
                self.background_vic, self.background_dic, lagged_vic,
                lagged_dic, self.vicmax, self.dicmax, self.capped_vic,
                self.capped_dic, self.cap_vic_flag, self.cap_dic_flag,
                self.cap_status))
            _require(not bool(self.cp.any(self.cap_status != 0).item()),
                     "v91 CUDA chkvic kernel rejected the handoff")
            _require(not _cancelled(self._cancelled_callback),
                     "v91 CUDA trial cancelled after chkvic launch")
            cap = apply_chkvic_used_total_v91(
                self.execution, parent_hles, handoff, dt_s=float(dt_s))
            for resident, expected, name in (
                (self.vicmax, cap.vicmax_m2_s, "vicmax"),
                (self.dicmax, cap.dicmax_m2_s, "dicmax"),
                (self.capped_vic, cap.capped_total_vicuv_m2_s, "vicuv"),
                (self.capped_dic, cap.capped_total_dicuv_m2_s, "dicuv"),
            ):
                actual = self._readback(resident)
                _require(np.array_equal(actual, expected),
                         "v91 CUDA/CPU chkvic differs: " + name)
        except Exception:
            self.abort_hydraulic_trial()
            raise
        self._parent, self._parent_hles = parent, parent_hles
        self._handoff, self._cap = handoff, cap
        self._dt, self._time = float(dt_s), float(time_s)
        self._residual_evaluations = 0
        self._open = True

    def _readback(self, value):
        """Single injection point for post-kernel failure tests."""
        return self.cp.asnumpy(value)

    def require_open_for_controller(self, dt_s: float, time_s: float,
                                    q) -> None:
        if _cancelled(self._cancelled_callback):
            self.abort_hydraulic_trial()
            raise SharedTurbulenceV91Error(
                "v91 CUDA trial cancelled before controller launch")
        _require(self._open and self._pending is None,
                 "v91 CUDA controller entered without an open trial")
        _require(float(dt_s) == self._dt and float(time_s) == self._time,
                 "v91 CUDA controller clock differs from its handoff")
        state = self._device(q, (3, self.nc), self.cp.float64,
                             "controller accepted state")
        _require(np.array_equal(self._readback(state), self._parent.q),
                 "v91 CUDA controller accepted state differs from parent")

    def add_momentum_rate(self, q, full_rate) -> None:
        """Add the closure in every authoritative ``_full_rate`` call."""
        if _cancelled(self._cancelled_callback):
            self.abort_hydraulic_trial()
            raise SharedTurbulenceV91Error(
                "v91 CUDA trial cancelled before momentum launch")
        _require(self._open and self._pending is None,
                 "v91 CUDA momentum consumer has no open trial")
        q = self._device(q, (3, self.nc), self.cp.float64,
                         "midpoint state")
        full_rate = self._device(full_rate, (3, self.nc), self.cp.float64,
                                 "full residual rate")
        self.kernel_launch_counts["momentum_face"] += 1
        self._launch("v91_momentum_face_flux", self.nf, (
            np.int64(self.nc), np.int64(self.nf), self.dry, q,
            self.capped_vic, self.owner, self.neighbour, self.length,
            self.distance, self.face_flux, self.face_status))
        self.kernel_launch_counts["momentum_gather"] += 1
        self._launch("v91_momentum_gather_add", self.nc, (
            np.int64(self.nc), np.int64(self.nf), self.dry, q, self.area,
            self.offset, self.faces, self.signs, self.face_flux,
            self.face_status, self.closure_rate, full_rate,
            self.cell_status))
        # A bad residual must reject immediately.  Deferring this check until
        # the final midpoint would let a later launch overwrite the status and
        # falsely authenticate an earlier invalid evaluation.
        _require(not bool(self.cp.any(self.face_status != 0).item()) and
                 not bool(self.cp.any(self.cell_status != 0).item()),
                 "v91 CUDA momentum residual status differs")
        if _cancelled(self._cancelled_callback):
            self.abort_hydraulic_trial()
            raise SharedTurbulenceV91Error(
                "v91 CUDA trial cancelled after momentum launch")
        self._residual_evaluations += 1

    def finish_hydraulic_trial(self, midpoint_q, accepted_q,
                               accepted_water_flux, *, dt_s: float,
                               time_s: float) -> dict:
        if _cancelled(self._cancelled_callback):
            self.abort_hydraulic_trial()
            raise SharedTurbulenceV91Error(
                "v91 CUDA trial cancelled before hydraulic finish")
        _require(self._open and self._pending is None,
                 "v91 CUDA hydraulic finish has no open trial")
        _require(float(dt_s) == self._dt and float(time_s) == self._time,
                 "v91 CUDA hydraulic finish clock differs")
        midpoint = self._device(
            midpoint_q, (3, self.nc), self.cp.float64, "accepted midpoint")
        endpoint = self._device(
            accepted_q, (3, self.nc), self.cp.float64, "accepted endpoint")
        water = self._device(
            accepted_water_flux, (self.nf,), self.cp.float64,
            "accepted midpoint water flux")
        try:
            _require(self._residual_evaluations >= 1 and
                     not bool(self.cp.any(self.face_status != 0).item()) and
                     not bool(self.cp.any(self.cell_status != 0).item()),
                     "v91 CUDA momentum operator status differs")
            midpoint_h = self._readback(midpoint)
            endpoint_h = self._readback(endpoint)
            water_h = self._readback(water)
            rate_h = self._readback(self.closure_rate)
            flux_h = self._readback(self.face_flux)
            expected_rate, expected_flux = momentum_diffusion_rate_v91(
                self.execution, midpoint_h,
                self._cap.capped_total_vicuv_m2_s)
            _require(np.allclose(rate_h, expected_rate, rtol=2e-15,
                                 atol=2e-16) and
                     np.allclose(flux_h, expected_flux, rtol=2e-15,
                                 atol=2e-16),
                     "v91 CUDA final residual differs from CPU face oracle")
            receipt = make_hydraulic_turbulence_receipt_v91(
                self.execution, self._parent, self._parent_hles,
                self._handoff, self._cap, midpoint_q=midpoint_h,
                accepted_q=endpoint_h, dt_s=self._dt,
                residual_evaluation_count=self._residual_evaluations,
                momentum_rate_evaluation_count=self._residual_evaluations,
                backend="actual_cuda", actual_implicit_controller=True)
            _require(not _cancelled(self._cancelled_callback),
                     "v91 CUDA trial cancelled after hydraulic readback")
        except Exception:
            self.abort_hydraulic_trial()
            raise
        self._pending = {
            "receipt": receipt,
            "midpoint": midpoint_h,
            "accepted": endpoint_h,
            "water": np.ascontiguousarray(water_h),
            "rate": np.ascontiguousarray(rate_h),
            "flux": np.ascontiguousarray(flux_h),
        }
        self._open = False
        return self.hydraulic_receipt_document(receipt)

    @staticmethod
    def hydraulic_receipt_document(receipt) -> dict:
        return {
            "scope_id": HYDRAULIC_RECEIPT_SCOPE_ID,
            "receipt_sha256": receipt.receipt_sha256,
            "current_handoff_receipt_sha256":
                receipt.current_handoff_receipt_sha256,
            "chkvic_receipt_sha256": receipt.chkvic_receipt_sha256,
            "base_step_receipt": dict(receipt.base_step_receipt),
            "residual_evaluation_count": receipt.residual_evaluation_count,
            "momentum_rate_evaluation_count":
                receipt.momentum_rate_evaluation_count,
            "momentum_consumer_count": receipt.momentum_consumer_count,
            "duplicate_post_source_count":
                receipt.duplicate_post_source_count,
            "actual_implicit_controller": receipt.actual_implicit_controller,
            "backend": receipt.backend,
            "outer_transaction_status": "provisional",
        }

    def accepted_receipt_document(self) -> dict:
        _require(self.last_accepted_receipt is not None,
                 "v91 CUDA has no committed outer receipt")
        return dict(self.last_accepted_receipt)

    def begin_scalar_trial(self, *, cancelled=False,
                           **scalar_inputs) -> SharedTurbulenceTrialV91:
        _require(self._pending is not None and self._scalar_runtime is None,
                 "v91 CUDA scalar consumer requires one accepted hydraulic trial")
        _require(not _cancelled(cancelled),
                 "v91 CUDA scalar trial cancelled before its one launch path")
        scalar_execution = resolve_suspended_support_execution_v82(
            self.execution.base_execution,
            dicuv_cell_m2_s=self._cap.capped_total_dicuv_m2_s)
        scalar_parent = _wrap_v82_state(
            scalar_execution, self._parent.suspended_state)
        runtime = ResidentSuspendedSupportLifecycleV82(scalar_execution)
        cp = self.cp
        values = dict(scalar_inputs)
        values.update(
            depth_before_m=cp.asarray(self._parent.q[0]),
            depth_after_m=cp.asarray(self._pending["accepted"][0]),
            accepted_water_flux_m2_s=cp.asarray(self._pending["water"]),
            endpoint_momentum_m2_s=cp.asarray(
                self._pending["accepted"][1:]),
            dt_s=self._dt,
        )
        try:
            scalar_trial = runtime.begin_trial(
                scalar_parent,
                accepted_step_receipt=
                    self._pending["receipt"].base_step_receipt,
                **values)
        except (TypeError, ValueError) as error:
            self.abort_hydraulic_trial()
            raise SharedTurbulenceV91Error(
                "v91 CUDA v82 scalar consumer rejected the trial") from error
        self.kernel_launch_counts["v82"] += 1
        trial = _wrap_trial(
            self.execution, self._parent, self._cap,
            self._pending["receipt"], scalar_execution, scalar_trial,
            midpoint_q=self._pending["midpoint"],
            momentum_rate=self._pending["rate"],
            momentum_flux=self._pending["flux"],
            accepted_q=self._pending["accepted"], backend="actual_cuda")
        self._scalar_runtime = runtime
        self._pending["trial"] = trial
        return trial

    def commit(self, parent: SharedTurbulenceStateV91,
               trial: SharedTurbulenceTrialV91) -> SharedTurbulenceStateV91:
        _require(self._pending is not None and
                 self._pending.get("trial") is trial and
                 self._scalar_runtime is not None,
                 "v91 CUDA trial is not the resident open transaction")
        _validate_trial(self.execution, parent, self._parent_hles,
                        self._handoff, trial)
        scalar_parent = _wrap_v82_state(
            trial.scalar_execution, parent.suspended_state)
        scalar = self._scalar_runtime.commit(
            scalar_parent, trial.scalar_trial)
        _require(scalar.base_state.state_sha256 ==
                 trial.candidate.suspended_state.state_sha256,
                 "v91 CUDA committed scalar state differs")
        result = trial.candidate
        accepted = self.hydraulic_receipt_document(trial.hydraulic_receipt)
        accepted.update(
            outer_transaction_status="accepted",
            scalar_trial_sha256=trial.scalar_trial.trial_sha256,
            accepted_state_sha256=result.state_sha256)
        self.last_accepted_receipt = accepted
        self._clear()
        return result

    def rollback(self, parent: SharedTurbulenceStateV91,
                 trial: SharedTurbulenceTrialV91) -> SharedTurbulenceStateV91:
        _require(self._pending is not None and
                 self._pending.get("trial") is trial and
                 self._scalar_runtime is not None,
                 "v91 CUDA trial is not the resident open transaction")
        _validate_trial(self.execution, parent, self._parent_hles,
                        self._handoff, trial)
        scalar_parent = _wrap_v82_state(
            trial.scalar_execution, parent.suspended_state)
        rolled = self._scalar_runtime.rollback(
            scalar_parent, trial.scalar_trial)
        _require(rolled is scalar_parent,
                 "v91 CUDA scalar rollback changed the parent")
        self._clear()
        return parent

    def cancel(self, parent, trial):
        return self.rollback(parent, trial)

    def abort_hydraulic_trial(self) -> None:
        if self._scalar_runtime is not None and self._pending is not None and \
                "trial" in self._pending:
            trial = self._pending["trial"]
            scalar_parent = _wrap_v82_state(
                trial.scalar_execution, self._parent.suspended_state)
            try:
                self._scalar_runtime.rollback(scalar_parent,
                                              trial.scalar_trial)
            except Exception:
                pass
        self._clear()

    def _clear(self):
        self._parent = self._parent_hles = self._handoff = self._cap = None
        self._dt = self._time = None
        self._residual_evaluations = 0
        self._pending = None
        self._scalar_runtime = None
        self._open = False
        self._cancelled_callback = False

    def checkpoint(self, state: SharedTurbulenceStateV91) -> dict:
        _require(not self._open and self._pending is None,
                 "v91 CUDA checkpoint requires a committed point")
        return shared_turbulence_checkpoint_v91(self.execution, state)

    def restore_checkpoint(self, document) -> SharedTurbulenceStateV91:
        _require(not self._open and self._pending is None,
                 "v91 CUDA restart cannot replace an open trial")
        # A successful restart begins a new publication history.  A rejected
        # document must leave the previously accepted diagnostic receipt
        # untouched so failure remains atomic.
        state = load_shared_turbulence_checkpoint_v91(
            self.execution, document)
        self.last_accepted_receipt = None
        return state


def method_identity_v91_cuda() -> dict:
    return {
        "cuda_composition_id": CUDA_COMPOSITION_ID,
        "integrator": INTEGRATOR,
        "actual_cuda_chkvic": True,
        "actual_cuda_momentum_face_operator": True,
        "true_full_rate_hook": True,
        "commit_false_receipt_status": "provisional_until_outer_commit",
        "scalar_owner": "actual_cuda_v82_once",
        "automatic_cpu_fallback": False,
        "product_admission": False,
    }


__all__ = [
    "CUDA_COMPOSITION_ID", "ResidentSharedTurbulenceConsumersV91",
    "method_identity_v91_cuda",
]
