"""Actual-CUDA composition for the Q4/v86 suspended controls.

The accepted v84 producer/lifecycle transaction remains the sole owner of
capacity, settling, exchange, advection, and diffusion.  This module runs the
new morphological bed-balance selection on the selected CUDA device and then
authenticates the bounded host evidence.  There is no CPU fallback.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np

from braids_core.sediment.route_lifecycle_composition_v81 import (
    ROUTE_COMBINE,
    ROUTE_OFF,
    ROUTE_TOTAL,
)
from braids_core.sediment.suspended_controls_v86 import (
    MAX_ACCEPTED_UPDATES,
    SuspendedControlsExecutionV86,
    SuspendedControlsStateV86,
    SuspendedControlsTrialV86,
    SuspendedControlsV86Error,
    _trial_token,
    build_suspended_controls_trial_v86,
    validate_suspended_controls_state_v86,
)


CUDA_SCOPE_ID = "braids-q4-flow2dh-suspended-controls-cuda-v86-v1"
CUDA_THREADS_PER_BLOCK = 128
ROOT = Path(__file__).resolve().parents[1]
CUDA_SOURCE = ROOT/"braids_gpu/suspended_controls_v86.cu"
CUDA_OPTIONS = (
    "--std=c++11",
    "--fmad=false",
    "--prec-div=true",
    "--prec-sqrt=true",
    "--ftz=false",
)
KERNELS = (
    "iopsus_diameter_v86",
    "suspended_bed_balance_v86",
    "suspended_boundary_ledger_v86",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SuspendedControlsV86Error(message)


class ResidentSuspendedControlOperatorsV86:
    """Compile and execute only the additive v86 CUDA operators."""

    def __init__(self, execution: SuspendedControlsExecutionV86 | None = None):
        try:
            import cupy as cp
        except Exception as error:  # pragma: no cover - host dependent
            raise SuspendedControlsV86Error(
                "v86 actual CUDA requires CuPy") from error
        try:
            _require(cp.cuda.runtime.getDeviceCount() >= 1,
                     "v86 actual CUDA requires a working device")
        except SuspendedControlsV86Error:
            raise
        except Exception as error:  # pragma: no cover - host dependent
            raise SuspendedControlsV86Error(
                "v86 CUDA device discovery failed") from error
        _require(CUDA_SOURCE.is_file(), "v86 CUDA source is absent")
        source = CUDA_SOURCE.read_bytes()
        try:
            module = cp.RawModule(
                code=source.decode("utf-8"), options=CUDA_OPTIONS,
                name_expressions=KERNELS)
            self.kernels = {name: module.get_function(name)
                            for name in KERNELS}
        except Exception as error:
            raise SuspendedControlsV86Error(
                "v86 CUDA operator compilation failed") from error
        self.cp = cp
        self.device_id = int(cp.cuda.Device().id)
        self.execution = execution
        self.source_sha256 = hashlib.sha256(source).hexdigest()
        if execution is not None:
            _require(execution.backend == "implicit_fwave_midpoint_gpu",
                     "v86 CUDA operators require the GPU execution")
            base = execution.lifecycle_execution.lifecycle_execution.base_execution
            self.routes = cp.ascontiguousarray(cp.asarray(
                execution.lifecycle_execution.route_execution["routes"],
                dtype=cp.int32))
            self.offset = cp.ascontiguousarray(cp.asarray(
                base.cell_face_offset, dtype=cp.int32))
            self.faces = cp.ascontiguousarray(cp.asarray(
                base.cell_faces, dtype=cp.int32))
            self.signs = cp.ascontiguousarray(cp.asarray(
                base.cell_face_sign, dtype=cp.int32))
            self.area = cp.ascontiguousarray(cp.asarray(
                base.cell_area_m2, dtype=cp.float64))
            self.length = cp.ascontiguousarray(cp.asarray(
                base.face_length_m, dtype=cp.float64))
            self.boundary_index = cp.ascontiguousarray(cp.asarray(
                base.boundary_index, dtype=cp.int32))

    def _launch(self, name: str, count: int, arguments: tuple) -> None:
        if count:
            blocks = (count+CUDA_THREADS_PER_BLOCK-1)//CUDA_THREADS_PER_BLOCK
            self.kernels[name]((blocks,), (CUDA_THREADS_PER_BLOCK,), arguments)

    def _device(self, value, shape, dtype, name):
        cp = self.cp
        _require(isinstance(value, cp.ndarray) and value.shape == shape and
                 value.dtype == dtype and value.flags.c_contiguous and
                 int(value.device.id) == self.device_id,
                 f"v86 CUDA {name} must be resident contiguous {dtype} {shape}")
        return value

    def iopsus_diameter(
        self, *, formula_code: int, iopsus: int, fixed_dss_m, d50_m,
        transport_stage=None, d10_m=None, mobility_psi=None,
        silt_diameter_m: float = 32.0e-6,
    ):
        """Evaluate the exact locked -1/-2/-4 IopSus branch on CUDA."""
        cp = self.cp
        _require(type(formula_code) is int and formula_code in (-4, -2, -1),
                 "v86 CUDA IopSus is defined only for TraFrm -1/-2/-4")
        _require(type(iopsus) is int and iopsus in (0, 1),
                 "v86 CUDA IopSus must be integer 0 or 1")
        _require(isinstance(d50_m, cp.ndarray) and d50_m.ndim >= 1,
                 "v86 CUDA D50 must be a resident array")
        shape = d50_m.shape
        fixed = self._device(fixed_dss_m, shape, cp.float64, "fixed DSS")
        d50 = self._device(d50_m, shape, cp.float64, "D50")
        dummy = cp.ones(shape, dtype=cp.float64)
        if formula_code == -1 and iopsus == 1:
            _require(transport_stage is not None,
                     "v86 CUDA Van Rijn 1993 requires transport stage")
            stage = self._device(
                transport_stage, shape, cp.float64, "transport stage")
        else:
            stage = dummy
        if formula_code in (-4, -2) and iopsus == 1:
            _require(d10_m is not None and mobility_psi is not None,
                     "v86 CUDA TRANSPOR2004 requires D10 and mobility psi")
            d10 = self._device(d10_m, shape, cp.float64, "D10")
            psi = self._device(
                mobility_psi, shape, cp.float64, "mobility psi")
        else:
            d10 = psi = dummy
        _require(not isinstance(silt_diameter_m, bool) and
                 isinstance(silt_diameter_m, (int, float)) and
                 math.isfinite(float(silt_diameter_m)) and
                 float(silt_diameter_m) > 0.0,
                 "v86 CUDA silt diameter must be positive")
        output = cp.empty(shape, dtype=cp.float64)
        status = cp.empty(shape, dtype=cp.int32)
        count = int(d50.size)
        self._launch("iopsus_diameter_v86", count, (
            np.int32(count), np.int32(formula_code), np.int32(iopsus),
            np.float64(silt_diameter_m), fixed, d50, stage, d10, psi,
            output, status))
        cp.cuda.get_current_stream().synchronize()
        if bool(cp.any(status != 0).item()):
            first = int(cp.flatnonzero(status.ravel() != 0)[0].item())
            code = int(status.ravel()[first].item())
            raise SuspendedControlsV86Error(
                f"v86 CUDA IopSus rejected element {first} (status {code})")
        return cp.ascontiguousarray(output)

    def bed_balance(self, inner_trial, dt_s: float):
        """Run the morphological owner selection on authenticated v84 data."""
        _require(self.execution is not None,
                 "v86 CUDA bed balance requires an execution")
        execution = self.execution
        evidence = inner_trial.lifecycle_trial.lifecycle_trial
        return self.bed_balance_topology(
            base_execution=(execution.lifecycle_execution.
                            lifecycle_execution.base_execution),
            routes=execution.lifecycle_execution.route_execution["routes"],
            neglect_entrainment=execution.neglect_entrainment,
            transferred_to_suspension_kg_m2=
                evidence.transferred_to_suspension_kg_m2,
            advective_face_flux_kg_m_s=
                evidence.advective_face_flux_kg_m_s,
            diffusive_face_mass_owner_out_kg=
                evidence.diffusive_face_mass_owner_out_kg,
            dt_s=dt_s)

    def bed_balance_topology(
        self, *, base_execution, routes,
        neglect_entrainment: bool,
        transferred_to_suspension_kg_m2,
        advective_face_flux_kg_m_s,
        diffusive_face_mass_owner_out_kg,
        dt_s: float,
    ):
        """Canonical CUDA peer of the exported v86 topology-selector ABI.

        Composition owners such as v93 may bind an exact accepted scalar
        trial to this method.  It intentionally uses the same kernels as the
        accepted v86 lifecycle rather than maintaining another transcription.
        """
        cp = self.cp
        required = (
            "fraction_count", "cell_count", "face_count", "boundary_count",
            "cell_face_offset", "cell_faces", "cell_face_sign",
            "cell_area_m2", "face_length_m", "boundary_index",
        )
        _require(all(hasattr(base_execution, name) for name in required),
                 "v86 CUDA topology selector requires a v76 execution")
        _require(type(neglect_entrainment) is bool,
                 "v86 CUDA NeglectEntrainment must be Boolean")
        _require(not isinstance(dt_s, bool) and
                 isinstance(dt_s, (int, float)) and
                 math.isfinite(float(dt_s)) and float(dt_s) > 0.0,
                 "v86 CUDA timestep must be positive")
        ng, nc, nf, nb = (
            int(base_execution.fraction_count),
            int(base_execution.cell_count), int(base_execution.face_count),
            int(base_execution.boundary_count))
        route_host = np.asarray(routes)
        _require(route_host.shape == (ng,) and route_host.dtype.kind in "iu" and
                 np.all(np.isin(
                     route_host, (ROUTE_OFF, ROUTE_TOTAL, ROUTE_COMBINE))),
                 "v86 CUDA topology selector route differs")
        route = cp.ascontiguousarray(cp.asarray(route_host, dtype=cp.int32))
        offset = cp.ascontiguousarray(cp.asarray(
            base_execution.cell_face_offset, dtype=cp.int32))
        faces = cp.ascontiguousarray(cp.asarray(
            base_execution.cell_faces, dtype=cp.int32))
        signs = cp.ascontiguousarray(cp.asarray(
            base_execution.cell_face_sign, dtype=cp.int32))
        area = cp.ascontiguousarray(cp.asarray(
            base_execution.cell_area_m2, dtype=cp.float64))
        length = cp.ascontiguousarray(cp.asarray(
            base_execution.face_length_m, dtype=cp.float64))
        boundary_index = cp.ascontiguousarray(cp.asarray(
            base_execution.boundary_index, dtype=cp.int32))
        adv = cp.ascontiguousarray(cp.asarray(
            advective_face_flux_kg_m_s, dtype=cp.float64))
        diff = cp.ascontiguousarray(cp.asarray(
            diffusive_face_mass_owner_out_kg, dtype=cp.float64))
        transfer = cp.ascontiguousarray(cp.asarray(
            transferred_to_suspension_kg_m2, dtype=cp.float64))
        _require(adv.shape == (ng, nf) and diff.shape == (ng, nf) and
                 transfer.shape == (ng, nc),
                 "v86 CUDA topology selector evidence shape differs")
        outputs = [cp.empty((ng, nc), dtype=cp.float64) for _ in range(3)]
        cell_status = cp.empty((ng, nc), dtype=cp.int32)
        self._launch("suspended_bed_balance_v86", ng*nc, (
            np.int32(ng), np.int32(nc), np.int32(nf),
            np.int32(int(neglect_entrainment)),
            np.int32(ROUTE_COMBINE), np.float64(dt_s), route,
            offset, faces, signs, area, length,
            adv, diff, transfer, *outputs, cell_status))
        boundary = cp.empty((ng, nb), dtype=cp.float64)
        boundary_status = cp.empty((ng, nb), dtype=cp.int32)
        self._launch("suspended_boundary_ledger_v86", ng*nb, (
            np.int32(ng), np.int32(nb), np.int32(nf), np.float64(dt_s),
            boundary_index, length, adv, diff, boundary,
            boundary_status))
        if nb:
            boundary = cp.where(
                route[:, None] == ROUTE_COMBINE, boundary, 0.0)
        cp.cuda.get_current_stream().synchronize()
        if bool(cp.any(cell_status != 0).item()) or \
                bool(cp.any(boundary_status != 0).item()):
            raise SuspendedControlsV86Error(
                "v86 CUDA suspended bed-balance kernel rejected evidence")
        return (*outputs, cp.ascontiguousarray(boundary))


class ResidentSuspendedControlsLifecycleV86:
    """One no-fallback CUDA transaction around exactly one v84 transaction."""

    def __init__(self, execution: SuspendedControlsExecutionV86, inner_runtime):
        _require(execution.backend == "implicit_fwave_midpoint_gpu",
                 "v86 CUDA owner requires the selected GPU execution")
        _require(hasattr(inner_runtime, "execution") and
                 inner_runtime.execution["fingerprint"] ==
                 execution.binding_execution["fingerprint"],
                 "v86 CUDA inner v84 runtime differs")
        self.execution = execution
        self.inner = inner_runtime
        self.operators = ResidentSuspendedControlOperatorsV86(execution)
        self.cp = self.operators.cp
        self.device_id = self.operators.device_id
        _require(getattr(inner_runtime, "device_id", None) == self.device_id,
                 "v86 controls and v84 lifecycle selected different devices")
        properties = self.cp.cuda.runtime.getDeviceProperties(self.device_id)
        raw_name = properties.get("name", b"unknown")
        device_name = (raw_name.decode("utf-8", errors="replace")
                       if isinstance(raw_name, bytes) else str(raw_name))
        self.runtime_receipt = {
            "scope_id": CUDA_SCOPE_ID,
            "execution_fingerprint": execution.fingerprint,
            "backend": "actual_cuda",
            "selected_integrator": "implicit_fwave_midpoint",
            "actual_cuda_bed_balance": True,
            "authenticated_inner_v84_transaction_count_per_trial": 1,
            "host_readback_scope":
                "bounded candidate authentication and immutable evidence",
            "automatic_cpu_fallback": False,
            "device_id": self.device_id,
            "device_name": device_name,
            "threads_per_block": CUDA_THREADS_PER_BLOCK,
            "source_sha256": self.operators.source_sha256,
            "suscor_2dh_application_count": 0,
            "product_admission": False,
        }
        self._open_parent: str | None = None
        self._open_trial: str | None = None

    def _host_readback(self, value):
        return np.ascontiguousarray(self.cp.asnumpy(value))

    def begin_trial(self, parent: SuspendedControlsStateV86, *,
                    cancelled: bool = False, **inputs) -> SuspendedControlsTrialV86:
        validate_suspended_controls_state_v86(parent, self.execution)
        _require(self._open_trial is None,
                 "v86 CUDA previous trial is unfinished")
        if cancelled:
            raise SuspendedControlsV86Error(
                "v86 CUDA cancelled before the inner transaction")
        _require(parent.accepted_updates < MAX_ACCEPTED_UPDATES,
                 "v86 CUDA accepted update limit was reached")
        inner_trial = None
        try:
            inner_trial = self.inner.begin_trial(
                parent.inner, cancelled=False, **inputs)
            gpu_exchange, gpu_flux, gpu_selected, gpu_boundary = \
                self.operators.bed_balance(inner_trial, inputs["dt_s"])
            exchange = self._host_readback(gpu_exchange)
            flux = self._host_readback(gpu_flux)
            selected = self._host_readback(gpu_selected)
            boundary = self._host_readback(gpu_boundary)
            trial = build_suspended_controls_trial_v86(
                self.execution, parent, inner_trial, flux_delta=flux,
                step_boundary_ledger=boundary, backend="actual_cuda")
            np.testing.assert_allclose(
                exchange, trial.exchange_bed_delta_kg_m2,
                rtol=2.0e-13, atol=2.0e-15)
            np.testing.assert_allclose(
                selected, trial.selected_bed_delta_kg_m2,
                rtol=2.0e-13, atol=2.0e-15)
        except Exception as error:
            if inner_trial is not None:
                try:
                    self.inner.rollback(parent.inner, inner_trial)
                except Exception as rollback_error:
                    raise SuspendedControlsV86Error(
                        "v86 CUDA candidate and inner rollback both failed") \
                        from rollback_error
            raise SuspendedControlsV86Error(
                "v86 CUDA control transaction failed") from error
        self._open_parent = parent.state_sha256
        self._open_trial = trial.trial_sha256
        return trial

    def _validate(self, parent, trial):
        validate_suspended_controls_state_v86(parent, self.execution)
        _require(isinstance(trial, SuspendedControlsTrialV86) and
                 trial.execution_fingerprint == self.execution.fingerprint and
                 trial.parent_state_sha256 == parent.state_sha256 and
                 trial.trial_sha256 == _trial_token(trial) and
                 self._open_parent == parent.state_sha256 and
                 self._open_trial == trial.trial_sha256,
                 "v86 CUDA trial is not the open transaction")
        validate_suspended_controls_state_v86(trial.candidate, self.execution)

    def _finish(self, parent, trial, *, commit):
        self._validate(parent, trial)
        inner = (self.inner.commit(parent.inner, trial.inner_trial) if commit
                 else self.inner.rollback(parent.inner, trial.inner_trial))
        result = trial.candidate if commit else parent
        if commit:
            _require(inner.state_sha256 == result.inner.state_sha256,
                     "v86 CUDA committed inner state differs")
        else:
            _require(inner is parent.inner, "v86 CUDA rollback changed state")
        self._open_parent = self._open_trial = None
        return result

    def commit(self, parent, trial):
        return self._finish(parent, trial, commit=True)

    def rollback(self, parent, trial):
        return self._finish(parent, trial, commit=False)

    def cancel(self, parent, trial):
        return self.rollback(parent, trial)

    def checkpoint(self, state):
        from braids_core.sediment.suspended_controls_v86 import (
            SuspendedControlsLifecycleV86,
        )
        validate_suspended_controls_state_v86(state, self.execution)
        _require(self._open_trial is None,
                 "v86 CUDA checkpoint requires a committed point")
        # The format is common to CPU and CUDA; call the unbound serializer.
        proxy = object.__new__(SuspendedControlsLifecycleV86)
        proxy.execution = self.execution
        proxy.inner = self.inner
        proxy._open_parent = None
        proxy._open_trial = None
        return SuspendedControlsLifecycleV86.checkpoint(proxy, state)

    def restore_checkpoint(self, document):
        from braids_core.sediment.suspended_controls_v86 import (
            SuspendedControlsLifecycleV86,
        )
        _require(self._open_trial is None,
                 "v86 CUDA restart cannot replace an open transaction")
        proxy = object.__new__(SuspendedControlsLifecycleV86)
        proxy.execution = self.execution
        proxy.inner = self.inner
        proxy._open_parent = None
        proxy._open_trial = None
        return SuspendedControlsLifecycleV86.restore_checkpoint(proxy, document)


__all__ = [
    "CUDA_SCOPE_ID", "CUDA_SOURCE", "CUDA_THREADS_PER_BLOCK", "KERNELS",
    "ResidentSuspendedControlOperatorsV86",
    "ResidentSuspendedControlsLifecycleV86",
]
