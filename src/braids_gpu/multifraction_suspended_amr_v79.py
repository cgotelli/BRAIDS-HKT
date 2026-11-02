"""Actual-CUDA owner for the bounded Q4/v79 suspended AMR composition."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from braids_core.sediment.multifraction_suspended_lifecycle_v78 import ROLE_CUDA
from braids_core.sediment.multifraction_suspended_amr_v79 import (
    TOPOLOGY_COARSE,
    TOPOLOGY_REFINED,
    MultifractionSuspendedAMRV79Error,
    _build_trial,
    _expanded_integrals,
    _lifecycle_with_rows,
    _restrict_children,
    _validate_receipt,
    _validate_trial,
    _wrap_state,
    commit_multifraction_suspended_amr_trial_v79,
    load_multifraction_suspended_amr_checkpoint_v79,
    method_identity_v79,
    multifraction_suspended_amr_checkpoint_v79,
    native_rows_v79,
    validate_multifraction_suspended_amr_state_v79,
)
from braids_gpu.multifraction_suspended_lifecycle_v78 import (
    ResidentMultifractionSuspendedLifecycleV78,
)
from braids_support.gpu_fwave.amr_reflux import ResidentAMRReflux
from braids_support.gpu_fwave.amr_transfer import ResidentAMRTransfer


ROOT = Path(__file__).resolve().parents[1]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MultifractionSuspendedAMRV79Error(message)


class ResidentMultifractionSuspendedAMRV79:
    """One private v79 proposal at a time; all AMR arithmetic stays on CUDA."""

    def __init__(self, execution):
        _require(execution.role == ROLE_CUDA,
                 "v79 resident owner requires the actual_cuda execution")
        import cupy as cp
        self.cp = cp
        self.execution = execution
        self.lifecycle = ResidentMultifractionSuspendedLifecycleV78(
            execution.lifecycle_execution)
        self._pending_kind = None
        self._pending_transfer = None
        self._pending_reflux = None
        self._pending_parent = None
        self._pending_trial = None
        self.device_id = int(cp.cuda.Device().id)
        properties = cp.cuda.runtime.getDeviceProperties(self.device_id)
        raw = properties.get("name", b"unknown")
        device_name = raw.decode(errors="replace") if isinstance(raw, bytes) \
            else str(raw)
        dependencies = (
            "braids_support/gpu_fwave/amr_transfer.py",
            "braids_support/gpu_fwave/amr_transfer.cu",
            "braids_support/gpu_fwave/amr_reflux.py",
            "braids_support/gpu_fwave/amr_reflux.cu",
            "braids_gpu/multifraction_suspended_lifecycle_v78.py",
        )
        self.runtime_receipt = {
            "scope_id": "braids-q4-multifraction-suspended-amr-v79-cuda-v1",
            "device_id": self.device_id,
            "device_name": device_name,
            "cupy_version": cp.__version__,
            "cuda_driver_version": int(cp.cuda.runtime.driverGetVersion()),
            "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
            "integrator": "implicit_fwave_midpoint",
            "fraction_count": execution.fraction_count,
            "reflux_levels": 1,
            "fine_integral_slots": 2,
            "authenticated_fine_substeps": 0,
            "recursive_subcycling_admitted": False,
            "actual_cuda_transfer": True,
            "actual_cuda_reflux": True,
            "actual_cuda_v78_lifecycle": True,
            "automatic_cpu_fallback": False,
            "host_readback_scope":
                "candidate authentication, ledgers, and checkpoint evidence only",
            "source_sha256": {
                name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                for name in dependencies
            },
        }

    @property
    def pending(self):
        return self._pending_kind is not None

    def _open(self, kind, parent, trial, *, transfer=None, reflux=None):
        _require(not self.pending, "resolve the previous v79 CUDA transaction first")
        self._pending_kind = kind
        self._pending_transfer = transfer
        self._pending_reflux = reflux
        self._pending_parent = parent.state_sha256
        self._pending_trial = trial.trial_sha256
        return trial

    def _transfer(self, cached_children):
        e = self.execution
        return ResidentAMRTransfer(
            np.ascontiguousarray(cached_children, dtype=np.float64),
            np.asarray([e.child_area_m2], dtype=np.float64),
            np.asarray([e.parent_area_m2], dtype=np.float64),
            np.asarray([e.child_bedrock_m], dtype=np.float64),
            np.asarray([e.child_position_m], dtype=np.float64),
            np.asarray(e.bulk_density_kg_m3, dtype=np.float64),
            dry=e.amr_contract.hydraulic_dry_tolerance_m)

    def begin_refine(self, parent, *, gradient=None):
        _require(not self.pending, "resolve the previous v79 CUDA transaction first")
        validate_multifraction_suspended_amr_state_v79(parent, self.execution)
        _require(parent.topology == TOPOLOGY_COARSE,
                 "v79 CUDA refine requires the coarse accepted topology")
        cp, e = self.cp, self.execution
        rows = native_rows_v79(e, parent.lifecycle,
                               parent.depth_m, parent.momentum_m2_s)
        child_rows = rows[np.asarray(e.child_cells)]
        _require(np.array_equal(child_rows, np.broadcast_to(
                     parent.coarse_parent_row, child_rows.shape)),
                 "v79 CUDA zero-gradient refinement requires a uniform child cache")
        children = np.ascontiguousarray(child_rows[None, :, :])
        transfer = self._transfer(children)
        primitives = 4+e.fraction_count
        if gradient is None:
            gradient = cp.zeros((1, primitives, 2), dtype=cp.float64)
        _require(isinstance(gradient, cp.ndarray) and
                 gradient.shape == (1, primitives, 2) and
                 gradient.dtype == cp.float64 and gradient.flags.c_contiguous and
                 not bool(cp.any(gradient != 0.0)),
                 "v79 CUDA bounded split requires a resident zero gradient")
        evolved = cp.ascontiguousarray(cp.asarray(
            parent.coarse_parent_row[None, :]))
        proposed, status = transfer.propose(evolved, gradient)
        if bool(cp.any(status != 0)):
            transfer.rollback()
            raise MultifractionSuspendedAMRV79Error(
                "v79 CUDA prolongation requires unsupported repair")
        next_rows = rows.copy()
        next_rows[np.asarray(e.child_cells)] = cp.asnumpy(proposed[0])
        lifecycle = _lifecycle_with_rows(e, parent.lifecycle, next_rows)
        candidate = _wrap_state(
            e, topology=TOPOLOGY_REFINED, lifecycle=lifecycle,
            depth_m=next_rows[:, 0], momentum_m2_s=next_rows[:, 1:3].T,
            coarse_parent_row=parent.coarse_parent_row,
            accepted_fine_flux_register_kg=parent.accepted_fine_flux_register_kg,
            topology_events=parent.topology_events+("refine",))
        trial = _build_trial(e, parent, "refine", candidate,
                             backend="actual_cuda")
        return self._open("refine", parent, trial, transfer=transfer)

    @staticmethod
    def _resident(cp, value, shape, name):
        _require(isinstance(value, cp.ndarray) and value.shape == shape and
                 value.dtype == cp.float64 and value.flags.c_contiguous and
                 bool(cp.all(cp.isfinite(value))),
                 f"v79 CUDA {name} must be resident contiguous FP64")
        return value

    def begin_transport(self, parent, *, accepted_step_receipt,
                        coarse_advective_integral_kg,
                        coarse_diffusive_integral_kg,
                        fine_advective_slots_kg,
                        fine_diffusive_slots_kg,
                        **lifecycle_inputs):
        _require(not self.pending, "resolve the previous v79 CUDA transaction first")
        validate_multifraction_suspended_amr_state_v79(parent, self.execution)
        _require(parent.topology == TOPOLOGY_REFINED,
                 "v79 CUDA transport requires the refined accepted topology")
        cp, e = self.cp, self.execution
        dt_s = float(lifecycle_inputs.get("dt_s", float("nan")))
        receipt = _validate_receipt(e, parent, accepted_step_receipt, dt_s)
        nf, ng, ns = (e.lifecycle_execution.face_count, e.fraction_count,
                      3+2*e.fraction_count)
        water = self._resident(cp, lifecycle_inputs.get("accepted_water_flux_m2_s"),
                               (nf,), "water flux")
        active = lifecycle_inputs.get("transport_face_active")
        _require(isinstance(active, cp.ndarray) and active.shape == (nf,) and
                 active.dtype == cp.int32 and active.flags.c_contiguous,
                 "v79 CUDA face-active mask must be resident contiguous int32")
        interface = cp.asarray(e.interface_faces)
        _require(not bool(cp.any(water[interface] != 0.0)) and
                 not bool(cp.any(active[interface])),
                 "v79 level-local lifecycle must leave registered faces single-owned")
        ca = self._resident(cp, coarse_advective_integral_kg,
                            (2, ng), "coarse advective integral")
        cd = self._resident(cp, coarse_diffusive_integral_kg,
                            (2, ng), "coarse diffusive integral")
        fa = self._resident(cp, fine_advective_slots_kg,
                            (2, 2, ng), "fine advective integral slots")
        fd = self._resident(cp, fine_diffusive_slots_kg,
                            (2, 2, ng), "fine diffusive integral slots")
        base_trial = None
        reflux = None
        try:
            base_trial = self.lifecycle.begin_trial(
                parent.lifecycle,
                accepted_step_receipt=receipt["level_local_v78_receipt"],
                **lifecycle_inputs)
            base_state = base_trial.candidate
            depth = cp.asnumpy(lifecycle_inputs["depth_after_m"])
            momentum = cp.asnumpy(lifecycle_inputs["endpoint_momentum_m2_s"])
            host_rows = native_rows_v79(e, base_state, depth, momentum)
            rows = cp.ascontiguousarray(cp.asarray(host_rows))
            coarse = cp.zeros((2, 2, ns), dtype=cp.float64)
            fine = cp.zeros((2, 2, 2, ns), dtype=cp.float64)
            coarse[:, :, 3:3+ng] = (ca+cd)[:, None, :]
            fine[:, :, :, 3:3+ng] = (fa+fd)[:, :, None, :]
            cells = cp.asarray(e.reflux_cells)
            before = cp.ascontiguousarray(rows[cells].copy())
            pre = before.copy()
            area = np.asarray([
                e.lifecycle_execution.base_execution.cell_area_m2[index]
                for index in e.reflux_cells], dtype=np.float64)
            pre[0] -= coarse[0, 0]/area[0]
            pre[0] += coarse[1, 1]/area[0]
            for step in range(2):
                pre[1] += fine[step, 0, 1]/area[1]
                pre[2] -= fine[step, 1, 0]/area[2]
            reflux = ResidentAMRReflux(
                pre, area, [0], [0, 2], [0, 1], [0, 1], 2, grains=ng)
            after, register, status = reflux.propose(coarse, fine)
            if bool(cp.any(status != 0)):
                reflux.rollback()
                self.lifecycle.rollback(parent.lifecycle, base_trial)
                raise MultifractionSuspendedAMRV79Error(
                    "v79 CUDA reflux requires unsupported conservative repair")
            rows[cells] = after
            host_after = cp.asnumpy(rows)
            lifecycle = _lifecycle_with_rows(e, base_state, host_after)
            parent_row = _restrict_children(e, host_after)
            candidate = _wrap_state(
                e, topology=TOPOLOGY_REFINED, lifecycle=lifecycle,
                depth_m=host_after[:, 0], momentum_m2_s=host_after[:, 1:3].T,
                coarse_parent_row=parent_row,
                accepted_fine_flux_register_kg=cp.asnumpy(register),
                topology_events=parent.topology_events+
                    ("transport_exchange_reflux",))
            arrays = (
                cp.asnumpy(ca), cp.asnumpy(cd), cp.asnumpy(fa), cp.asnumpy(fd),
                cp.asnumpy(register), cp.asnumpy(before), cp.asnumpy(after))
            trial = _build_trial(
                e, parent, "transport", candidate,
                lifecycle_trial=base_trial, arrays=arrays,
                receipt_fingerprint=receipt["fingerprint"],
                backend="actual_cuda")
        except Exception:
            if reflux is not None and reflux.pending:
                reflux.rollback()
            if base_trial is not None and self.lifecycle._open_trial is not None:
                self.lifecycle.rollback(parent.lifecycle, base_trial)
            raise
        return self._open("transport", parent, trial, reflux=reflux)

    def begin_coarsen(self, parent):
        _require(not self.pending, "resolve the previous v79 CUDA transaction first")
        validate_multifraction_suspended_amr_state_v79(parent, self.execution)
        _require(parent.topology == TOPOLOGY_REFINED,
                 "v79 CUDA coarsen requires the refined accepted topology")
        cp, e = self.cp, self.execution
        rows = native_rows_v79(e, parent.lifecycle,
                               parent.depth_m, parent.momentum_m2_s)
        children = np.ascontiguousarray(rows[np.asarray(e.child_cells)][None, :, :])
        transfer = self._transfer(children)
        restricted = cp.asnumpy(transfer.restrict()[0])
        candidate = _wrap_state(
            e, topology=TOPOLOGY_COARSE, lifecycle=parent.lifecycle,
            depth_m=parent.depth_m, momentum_m2_s=parent.momentum_m2_s,
            coarse_parent_row=restricted,
            accepted_fine_flux_register_kg=parent.accepted_fine_flux_register_kg,
            topology_events=parent.topology_events+("coarsen",))
        trial = _build_trial(e, parent, "coarsen", candidate,
                             backend="actual_cuda")
        return self._open("coarsen", parent, trial, transfer=transfer)

    def _finish(self, parent, trial, *, commit):
        _require(self.pending and self._pending_parent == parent.state_sha256 and
                 self._pending_trial == trial.trial_sha256,
                 "v79 CUDA transaction parent or trial differs")
        # The outer immutable transaction authenticates topology, lifecycle,
        # register, boundary memory, ledgers, and every evidence array.  It
        # must fail before either resident child owner can publish.
        _validate_trial(self.execution, parent, trial)
        try:
            if self._pending_kind == "refine":
                if commit:
                    self._pending_transfer.commit()
                else:
                    self._pending_transfer.rollback()
            elif self._pending_kind == "transport":
                if commit:
                    self._pending_reflux.commit()
                    self.lifecycle.commit(parent.lifecycle, trial.lifecycle_trial)
                else:
                    self._pending_reflux.rollback()
                    self.lifecycle.rollback(parent.lifecycle, trial.lifecycle_trial)
            result = (commit_multifraction_suspended_amr_trial_v79(
                self.execution, parent, trial) if commit else parent)
        finally:
            self._pending_kind = None
            self._pending_transfer = None
            self._pending_reflux = None
            self._pending_parent = None
            self._pending_trial = None
        return result

    def commit(self, parent, trial):
        return self._finish(parent, trial, commit=True)

    def rollback(self, parent, trial):
        return self._finish(parent, trial, commit=False)

    def cancel(self, parent, trial):
        return self.rollback(parent, trial)

    def checkpoint(self, state):
        _require(not self.pending,
                 "v79 CUDA checkpoint requires an accepted transaction boundary")
        return multifraction_suspended_amr_checkpoint_v79(
            self.execution, state)

    def restore_checkpoint(self, document):
        _require(not self.pending,
                 "v79 CUDA restore requires an accepted transaction boundary")
        state = load_multifraction_suspended_amr_checkpoint_v79(
            self.execution, document)
        self.lifecycle.restore_checkpoint(document["v78_checkpoint"])
        return state


def method_identity_v79_cuda() -> dict:
    return {
        **method_identity_v79(),
        "cuda_owner": "resident_multifraction_suspended_amr_v79",
        "transfer_kernel": "amr_restrict_native_rows/amr_prolong_native_dry_bed",
        "reflux_kernel": "amr_fine_register/amr_reflux_coarse",
        "actual_cuda_required": True,
        "automatic_cpu_fallback": False,
    }


__all__ = ["ResidentMultifractionSuspendedAMRV79",
           "method_identity_v79_cuda"]
