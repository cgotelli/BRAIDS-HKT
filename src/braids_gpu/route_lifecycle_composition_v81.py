"""Actual-CUDA owner for the additive Q4/v81 route/lifecycle bridge."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path

import numpy as np

from braids_core.sediment.multifraction_suspended_lifecycle_v78 import (
    MAX_ACCEPTED_UPDATES,
    ROLE_CUDA,
)
from braids_core.sediment.route_lifecycle_composition_v81 import (
    ROUTE_COMBINE,
    ROUTE_OFF,
    ROUTE_TOTAL,
    RouteLifecycleExecutionV81,
    RouteLifecycleStateV81,
    RouteLifecycleTrialV81,
    RouteLifecycleV81Error,
    _build_trial,
    _digest,
    _fingerprint,
    _validate_trial,
    load_route_lifecycle_checkpoint_v81,
    route_lifecycle_checkpoint_v81,
    validate_route_lifecycle_state_v81,
)
from braids_core.sediment.suspended_route_live_v74 import (
    RouteHandoffV74,
    _handoff_token as _v74_handoff_token,
    validate_route_handoff_v74,
)
from braids_gpu.multifraction_suspended_lifecycle_v78 import (
    ResidentMultifractionSuspendedLifecycleV78,
)


CUDA_THREADS_PER_BLOCK = 128
CUDA_SCOPE_ID = "braids-q4-route-lifecycle-composition-cuda-v81-v1"
ROOT = Path(__file__).resolve().parents[1]
CUDA_SOURCE = ROOT/"braids_gpu/route_lifecycle_composition_v81.cu"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RouteLifecycleV81Error(message)


@dataclass(frozen=True)
class ResidentRouteHandoffV81:
    """Pure resident v74 bridge output with no v78 lifecycle side effect."""

    handoff: RouteHandoffV74
    bed_capacity_x: object
    bed_capacity_y: object
    equilibrium_concentration: object
    suspended_equivalent_x: object
    suspended_equivalent_y: object
    lifecycle_capacity: object
    status: object
    resident_sha256: str
    device_id: int


class ResidentRouteLifecycleV81:
    """One no-fallback CUDA transaction spanning route and lifecycle."""

    def __init__(self, execution: RouteLifecycleExecutionV81):
        try:
            import cupy as cp
        except Exception as error:  # pragma: no cover - environment dependent
            raise RouteLifecycleV81Error(
                "v81 actual CUDA requires CuPy and a working CUDA device") \
                from error
        try:
            _require(cp.cuda.runtime.getDeviceCount() >= 1,
                     "v81 actual CUDA requires a CUDA device")
        except RouteLifecycleV81Error:
            raise
        except Exception as error:  # pragma: no cover - environment dependent
            raise RouteLifecycleV81Error(
                "v81 CUDA device discovery failed") from error
        _require(isinstance(execution, RouteLifecycleExecutionV81) and
                 execution.role == ROLE_CUDA,
                 "v81 CUDA owner requires an actual_cuda execution")
        if not CUDA_SOURCE.is_file():
            raise RouteLifecycleV81Error("v81 CUDA bridge source is absent")
        self.cp = cp
        self.execution = execution
        self.device_id = int(cp.cuda.Device().id)
        source = CUDA_SOURCE.read_bytes()
        try:
            module = cp.RawModule(
                code=source.decode("utf-8"), options=("--std=c++11",),
                name_expressions=("route_lifecycle_bridge_v81",))
            self._bridge = module.get_function("route_lifecycle_bridge_v81")
        except Exception as error:
            raise RouteLifecycleV81Error(
                "v81 CUDA bridge compilation failed") from error
        self.lifecycle = ResidentMultifractionSuspendedLifecycleV78(
            execution.lifecycle_execution)
        self.routes_host = np.asarray(
            execution.route_execution["routes"], dtype=np.int32)
        self.routes = cp.ascontiguousarray(cp.asarray(
            self.routes_host, dtype=cp.int32))
        self.kinds = cp.ascontiguousarray(cp.asarray(
            execution.route_execution["output_kinds"], dtype=cp.int32))
        properties = cp.cuda.runtime.getDeviceProperties(self.device_id)
        raw_name = properties.get("name", b"unknown")
        device_name = (raw_name.decode("utf-8", errors="replace")
                       if isinstance(raw_name, bytes) else str(raw_name))
        self.runtime_receipt = {
            "scope_id": CUDA_SCOPE_ID,
            "execution_fingerprint": execution.fingerprint,
            "backend": "actual_cuda",
            "integrator": "implicit_fwave_midpoint",
            "fixed_grid": True,
            "wave_active": False,
            "automatic_cpu_fallback": False,
            "actual_cuda_route_bridge": True,
            "actual_cuda_lifecycle_kernels": True,
            "device_id": self.device_id,
            "device_name": device_name,
            "threads_per_block": CUDA_THREADS_PER_BLOCK,
            "route_bridge_source_sha256": hashlib.sha256(source).hexdigest(),
            "lifecycle_runtime_receipt": self.lifecycle.runtime_receipt,
            "host_readback_scope":
                "candidate authentication and checkpoint evidence only",
        }
        self._open_parent: str | None = None
        self._open_trial: str | None = None

    def _device(self, value, shape, dtype, name):
        cp = self.cp
        _require(isinstance(value, cp.ndarray) and value.shape == shape and
                 value.dtype == dtype and value.flags.c_contiguous and
                 int(value.device.id) == self.device_id,
                 f"v81 CUDA {name} must be resident contiguous {dtype} {shape}")
        return value

    def compose_route_handoff(
        self, *, depth_after_m, endpoint_momentum_m2_s,
        raw_bed_x_kg_m_s, raw_bed_y_kg_m_s,
        raw_suspended_x_kg_m_s, raw_suspended_y_kg_m_s,
        raw_equilibrium_concentration_kg_m3, active_fraction,
    ) -> ResidentRouteHandoffV81:
        """Execute only the actual-CUDA v74 route bridge.

        No v78 state or transaction is opened.  The returned host handoff is
        immutable acceptance evidence; every numerical output used by the
        downstream face owner remains resident on this selected device.
        """
        cp = self.cp
        ng = self.execution.fraction_count
        nc = self.execution.cell_count
        shape = (ng, nc)
        h1 = self._device(
            depth_after_m, (nc,), cp.float64, "pure-route depth")
        momentum = self._device(
            endpoint_momentum_m2_s, (2, nc), cp.float64,
            "pure-route endpoint momentum")
        raw_bed_x = self._device(
            raw_bed_x_kg_m_s, shape, cp.float64, "pure-route raw bed x")
        raw_bed_y = self._device(
            raw_bed_y_kg_m_s, shape, cp.float64, "pure-route raw bed y")
        raw_sus_x = self._device(
            raw_suspended_x_kg_m_s, shape, cp.float64,
            "pure-route raw suspended x")
        raw_sus_y = self._device(
            raw_suspended_y_kg_m_s, shape, cp.float64,
            "pure-route raw suspended y")
        raw_eq = self._device(
            raw_equilibrium_concentration_kg_m3, shape, cp.float64,
            "pure-route raw equilibrium")
        fraction = self._device(
            active_fraction, shape, cp.float64,
            "pure-route active fraction")
        for value, name in (
            (h1, "depth"), (momentum, "momentum"),
            (raw_bed_x, "raw bed x"), (raw_bed_y, "raw bed y"),
            (raw_sus_x, "raw suspended x"),
            (raw_sus_y, "raw suspended y"),
            (raw_eq, "raw equilibrium"), (fraction, "active fraction"),
        ):
            _require(bool(cp.all(cp.isfinite(value)).item()),
                     "v81 CUDA pure-route input is not finite: "+name)
        _require(bool(cp.all((fraction >= 0.0) &
                             (fraction <= 1.0)).item()),
                 "v81 CUDA pure-route active fraction is outside [0,1]")
        outputs = [cp.empty(shape, dtype=cp.float64) for _ in range(6)]
        status = cp.empty(shape, dtype=cp.int32)
        count = ng*nc
        blocks = (count+CUDA_THREADS_PER_BLOCK-1)//CUDA_THREADS_PER_BLOCK
        self._bridge(
            (blocks,), (CUDA_THREADS_PER_BLOCK,),
            (np.int32(ng), np.int32(nc), self.kinds, self.routes, h1,
             momentum, raw_bed_x, raw_bed_y, raw_sus_x, raw_sus_y, raw_eq,
             fraction,
             np.float64(self.execution.route_execution["bed_factor"]),
             np.float64(self.execution.route_execution["suspended_factor"]),
             np.float64(self.execution.route_execution["speed_epsilon"]),
             np.float64(self.execution.lifecycle_execution.
                        base_execution.dry_tolerance_m),
             np.float64(self.execution.lifecycle_execution.base_execution.
                        profile.sediment_threshold_m),
             np.float64(self.execution.lifecycle_execution.base_execution.
                        profile.reference_depth_floor_m),
             *outputs, status))
        cp.cuda.get_current_stream().synchronize()
        if bool(cp.any(status != 0).item()):
            first = cp.asnumpy(cp.argwhere(status != 0)[0]).tolist()
            raise RouteLifecycleV81Error(
                f"v81 CUDA pure route bridge rejected lane/cell {first}")
        outputs = [cp.ascontiguousarray(value) for value in outputs]
        host = [np.ascontiguousarray(cp.asnumpy(value)) for value in outputs]
        bed_owner = (self.routes_host == ROUTE_TOTAL).astype(np.int32)
        water_owner = (self.routes_host == ROUTE_COMBINE).astype(np.int32)
        active = (self.routes_host != ROUTE_OFF).astype(np.int32)
        placeholder = RouteHandoffV74(
            self.execution.route_execution["fingerprint"],
            self.execution.route_execution["q3_handoff_fingerprint"],
            host[0], host[1], host[2], host[3], host[4],
            bed_owner, water_owner, active, active, 0, "")
        handoff = RouteHandoffV74(
            **{**placeholder.__dict__,
               "handoff_sha256": _v74_handoff_token(placeholder)})
        validate_route_handoff_v74(
            handoff, execution=self.execution.route_execution)
        token = _fingerprint({
            "scope_id": CUDA_SCOPE_ID+":pure-route",
            "execution_fingerprint": self.execution.fingerprint,
            "handoff_sha256": handoff.handoff_sha256,
            "resident_arrays_sha256": _digest(
                *host, np.ascontiguousarray(cp.asnumpy(status))),
            "device_id": self.device_id,
            "scalar_lifecycle_application_count": 0,
        })
        return ResidentRouteHandoffV81(
            handoff, *outputs, cp.ascontiguousarray(status), token,
            self.device_id)

    def validate_route_handoff(
        self, value: ResidentRouteHandoffV81,
    ) -> ResidentRouteHandoffV81:
        cp = self.cp
        if (not isinstance(value, ResidentRouteHandoffV81) or
                value.device_id != self.device_id):
            raise RouteLifecycleV81Error(
                "v81 CUDA pure route handoff device differs")
        validate_route_handoff_v74(
            value.handoff, execution=self.execution.route_execution)
        arrays = (
            value.bed_capacity_x, value.bed_capacity_y,
            value.equilibrium_concentration,
            value.suspended_equivalent_x, value.suspended_equivalent_y,
            value.lifecycle_capacity,
        )
        shape = (self.execution.fraction_count, self.execution.cell_count)
        for array in arrays:
            self._device(array, shape, cp.float64, "pure-route output")
        self._device(value.status, shape, cp.int32, "pure-route status")
        host = [np.ascontiguousarray(cp.asnumpy(item)) for item in arrays]
        expected = _fingerprint({
            "scope_id": CUDA_SCOPE_ID+":pure-route",
            "execution_fingerprint": self.execution.fingerprint,
            "handoff_sha256": value.handoff.handoff_sha256,
            "resident_arrays_sha256": _digest(
                *host, np.ascontiguousarray(cp.asnumpy(value.status))),
            "device_id": self.device_id,
            "scalar_lifecycle_application_count": 0,
        })
        if value.resident_sha256 != expected:
            raise RouteLifecycleV81Error(
                "v81 CUDA pure route resident arrays are stale")
        return value

    def begin_trial(
        self, parent: RouteLifecycleStateV81, *, accepted_step_receipt,
        depth_before_m, depth_after_m, accepted_water_flux_m2_s,
        reconstructed_left_concentration_kg_m3,
        reconstructed_right_concentration_kg_m3,
        prescribed_boundary_concentration_kg_m3,
        accepted_face_exchange_area_m2, transport_face_active,
        endpoint_momentum_m2_s, shear_velocity_m_s, hydro_constant,
        raw_bed_x_kg_m_s, raw_bed_y_kg_m_s,
        raw_suspended_x_kg_m_s, raw_suspended_y_kg_m_s,
        raw_equilibrium_concentration_kg_m3, active_fraction,
        dt_s: float, cancelled: bool = False,
    ) -> RouteLifecycleTrialV81:
        cp = self.cp
        validate_route_lifecycle_state_v81(parent, self.execution)
        _require(self._open_trial is None,
                 "v81 CUDA previous trial is unfinished")
        if cancelled:
            raise RouteLifecycleV81Error(
                "v81 CUDA update cancelled before candidate publication")
        if parent.accepted_updates >= MAX_ACCEPTED_UPDATES:
            raise RouteLifecycleV81Error(
                "v81 CUDA accepted update limit was reached")
        _require(not isinstance(dt_s, bool) and
                 isinstance(dt_s, (int, float)) and
                 math.isfinite(float(dt_s)) and float(dt_s) > 0.0,
                 "v81 CUDA timestep must be finite and positive")
        ng = self.execution.fraction_count
        base = self.execution.lifecycle_execution.base_execution
        nc, nf, nb = base.cell_count, base.face_count, base.boundary_count
        h0 = self._device(depth_before_m, (nc,), cp.float64, "depth before")
        h1 = self._device(depth_after_m, (nc,), cp.float64, "depth after")
        water = self._device(
            accepted_water_flux_m2_s, (nf,), cp.float64, "water flux")
        left = self._device(
            reconstructed_left_concentration_kg_m3, (ng, nf), cp.float64,
            "left reconstruction")
        right = self._device(
            reconstructed_right_concentration_kg_m3, (ng, nf), cp.float64,
            "right reconstruction")
        prescribed = self._device(
            prescribed_boundary_concentration_kg_m3, (ng, nb), cp.float64,
            "prescribed boundary concentration")
        face_area = self._device(
            accepted_face_exchange_area_m2, (nf,), cp.float64,
            "face exchange area")
        face_active = self._device(
            transport_face_active, (nf,), cp.int32, "face activity")
        momentum = self._device(
            endpoint_momentum_m2_s, (2, nc), cp.float64,
            "endpoint momentum")
        shear = self._device(
            shear_velocity_m_s, (nc,), cp.float64, "shear velocity")
        constant = self._device(
            hydro_constant, (nc,), cp.int32, "hydro constant")
        shape = (ng, nc)
        raw_bed_x = self._device(
            raw_bed_x_kg_m_s, shape, cp.float64, "raw bed x")
        raw_bed_y = self._device(
            raw_bed_y_kg_m_s, shape, cp.float64, "raw bed y")
        raw_sus_x = self._device(
            raw_suspended_x_kg_m_s, shape, cp.float64, "raw suspended x")
        raw_sus_y = self._device(
            raw_suspended_y_kg_m_s, shape, cp.float64, "raw suspended y")
        raw_eq = self._device(
            raw_equilibrium_concentration_kg_m3, shape, cp.float64,
            "raw equilibrium concentration")
        fraction = self._device(
            active_fraction, shape, cp.float64, "active fraction")
        for value, name in (
            (h0, "depth before"), (h1, "depth after"),
            (water, "water flux"), (left, "left reconstruction"),
            (right, "right reconstruction"), (prescribed, "boundary"),
            (face_area, "face area"), (momentum, "momentum"),
            (shear, "shear"), (raw_bed_x, "raw bed x"),
            (raw_bed_y, "raw bed y"), (raw_sus_x, "raw suspended x"),
            (raw_sus_y, "raw suspended y"), (raw_eq, "raw equilibrium"),
            (fraction, "active fraction"),
        ):
            _require(bool(cp.all(cp.isfinite(value)).item()),
                     "v81 CUDA input must be finite: "+name)
        inactive = cp.asarray(self.routes_host != ROUTE_COMBINE)
        if bool(cp.any(left[inactive] != 0.0).item()) or \
                bool(cp.any(right[inactive] != 0.0).item()) or \
                bool(cp.any(prescribed[inactive] != 0.0).item()):
            raise RouteLifecycleV81Error(
                "v81 CUDA off/total routes cannot receive boundary data")

        outputs = [cp.empty(shape, dtype=cp.float64) for _ in range(6)]
        status = cp.empty(shape, dtype=cp.int32)
        count = ng*nc
        blocks = (count+CUDA_THREADS_PER_BLOCK-1)//CUDA_THREADS_PER_BLOCK
        self._bridge(
            (blocks,), (CUDA_THREADS_PER_BLOCK,),
            (np.int32(ng), np.int32(nc), self.kinds, self.routes, h1,
             momentum, raw_bed_x, raw_bed_y, raw_sus_x, raw_sus_y, raw_eq,
             fraction, np.float64(self.execution.route_execution["bed_factor"]),
             np.float64(self.execution.route_execution["suspended_factor"]),
             np.float64(self.execution.route_execution["speed_epsilon"]),
             np.float64(base.dry_tolerance_m),
             np.float64(base.profile.sediment_threshold_m),
             np.float64(base.profile.reference_depth_floor_m),
             *outputs, status))
        cp.cuda.get_current_stream().synchronize()
        if bool(cp.any(status != 0).item()):
            first = cp.asnumpy(cp.argwhere(status != 0)[0]).tolist()
            raise RouteLifecycleV81Error(
                f"v81 CUDA route bridge rejected lane/cell {first}")
        (bed_x, bed_y, target, equivalent_x, equivalent_y, capacity) = outputs
        try:
            lifecycle_trial = self.lifecycle.begin_trial(
                parent.lifecycle, accepted_step_receipt=accepted_step_receipt,
                depth_before_m=h0, depth_after_m=h1,
                accepted_water_flux_m2_s=water,
                reconstructed_left_concentration_kg_m3=left,
                reconstructed_right_concentration_kg_m3=right,
                prescribed_boundary_concentration_kg_m3=prescribed,
                accepted_face_exchange_area_m2=face_area,
                transport_face_active=face_active,
                endpoint_momentum_m2_s=momentum,
                shear_velocity_m_s=shear,
                suspended_transport_capacity_kg_m_s=
                    cp.ascontiguousarray(capacity),
                surface_fraction=fraction, hydro_constant=constant,
                dt_s=float(dt_s))
        except Exception:
            raise

        wet = h1 > base.dry_tolerance_m
        velocity_x = cp.where(wet, momentum[0]/cp.where(wet, h1, 1.0), 0.0)
        velocity_y = cp.where(wet, momentum[1]/cp.where(wet, h1, 1.0), 0.0)
        speed = cp.hypot(velocity_x, velocity_y)
        sediment_active = wet & (h1 > base.profile.sediment_threshold_m)
        exchange_depth = cp.where(
            sediment_active,
            cp.maximum(base.profile.reference_depth_floor_m, h1), 1.0)
        host_inputs = [cp.asnumpy(value) for value in (
            h1, momentum, raw_bed_x, raw_bed_y, raw_sus_x, raw_sus_y,
            raw_eq, fraction, exchange_depth, speed)]
        control = _fingerprint({
            "q3_receipt_fingerprint": self.execution.q3_receipt["fingerprint"],
            "route_execution_fingerprint":
                self.execution.route_execution["fingerprint"],
            "arrays_sha256": _digest(*host_inputs),
        })
        host_outputs = [cp.asnumpy(value) for value in outputs]
        bed_owner = (self.routes_host == ROUTE_TOTAL).astype(np.int32)
        water_owner = (self.routes_host == ROUTE_COMBINE).astype(np.int32)
        factor_count = (self.routes_host != ROUTE_OFF).astype(np.int32)
        route_evidence = _fingerprint({
            "scope_id": CUDA_SCOPE_ID,
            "execution_fingerprint": self.execution.fingerprint,
            "q3_control_sha256": control,
            "arrays_sha256": _digest(*host_outputs, bed_owner, water_owner,
                                      factor_count, factor_count),
            "post_q3_density_conversion_count": 0,
        })
        try:
            trial = _build_trial(
                self.execution, parent, lifecycle_trial,
                accepted_step_receipt, q3_control_sha256=control,
                route_handoff_sha256=route_evidence,
                bed_capacity_x=host_outputs[0],
                bed_capacity_y=host_outputs[1],
                equilibrium_target=host_outputs[2],
                equivalent_x=host_outputs[3], equivalent_y=host_outputs[4],
                lifecycle_capacity=host_outputs[5], bed_owner=bed_owner,
                water_owner=water_owner,
                suspended_factor_count=factor_count,
                active_fraction_count=factor_count,
                raw_suspended_x=host_inputs[4],
                raw_suspended_y=host_inputs[5],
                active_fraction=host_inputs[7], backend=ROLE_CUDA)
        except Exception:
            self.lifecycle.rollback(parent.lifecycle, lifecycle_trial)
            raise
        self._open_parent = parent.state_sha256
        self._open_trial = trial.trial_sha256
        return trial

    def _finish(self, parent, trial, *, commit):
        _validate_trial(self.execution, parent, trial)
        _require(self._open_parent == parent.state_sha256 and
                 self._open_trial == trial.trial_sha256,
                 "v81 CUDA trial is not the resident open transaction")
        lifecycle = (self.lifecycle.commit(
            parent.lifecycle, trial.lifecycle_trial) if commit else
            self.lifecycle.rollback(parent.lifecycle, trial.lifecycle_trial))
        result = trial.candidate if commit else parent
        if commit:
            _require(lifecycle.state_sha256 == result.lifecycle.state_sha256,
                     "v81 CUDA committed lifecycle differs from candidate")
        self._open_parent = None
        self._open_trial = None
        return result

    def commit(self, parent, trial) -> RouteLifecycleStateV81:
        return self._finish(parent, trial, commit=True)

    def rollback(self, parent, trial) -> RouteLifecycleStateV81:
        return self._finish(parent, trial, commit=False)

    def cancel(self, parent, trial) -> RouteLifecycleStateV81:
        return self.rollback(parent, trial)

    def checkpoint(self, state: RouteLifecycleStateV81) -> dict:
        _require(self._open_trial is None,
                 "v81 CUDA checkpoint requires a committed point")
        validate_route_lifecycle_state_v81(state, self.execution)
        return route_lifecycle_checkpoint_v81(self.execution, state)

    def restore_checkpoint(self, document) -> RouteLifecycleStateV81:
        _require(self._open_trial is None,
                 "v81 CUDA restart cannot replace an open trial")
        state = load_route_lifecycle_checkpoint_v81(
            self.execution, document)
        restored = self.lifecycle.restore_checkpoint(
            document["lifecycle_checkpoint_v78"])
        _require(restored.state_sha256 == state.lifecycle.state_sha256,
                 "v81 CUDA resident restart differs from authenticated state")
        return state


def method_identity_v81_cuda() -> dict[str, object]:
    return {
        "scope_id": CUDA_SCOPE_ID,
        "integrator": "implicit_fwave_midpoint",
        "threads_per_block": CUDA_THREADS_PER_BLOCK,
        "fixed_grid_only": True,
        "actual_cuda_route_bridge": True,
        "actual_cuda_lifecycle_kernels": True,
        "automatic_cpu_fallback": False,
        "host_readback_scope":
            "candidate authentication and checkpoint evidence only",
    }


__all__ = [
    "CUDA_SCOPE_ID", "CUDA_THREADS_PER_BLOCK", "ResidentRouteHandoffV81",
    "ResidentRouteLifecycleV81",
    "method_identity_v81_cuda",
]
