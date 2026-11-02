"""Actual-CUDA peer for the bounded v92 parent-fluid/Q3 handoff.

The CUDA kernels are the already accepted v66 direct-current kernels.  This
adapter supplies the authenticated cell-wise parent density, viscosity and
settling fields instead of allowing the v67 family wrapper to manufacture a
constant profile density.  There is no CPU numerical fallback.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

import numpy as np

from braids_core.sediment.direct_vector_family_live_v67 import (
    load_direct_vector_family_execution_v67,
)
from braids_core.sediment.q3_dynamic_fluid_handoff_v92 import (
    Q3DynamicFluidAcceptedRowsV92,
    Q3DynamicFluidHandoffV92,
    Q3DynamicFluidV92Error,
    _accepted_result,
    _fingerprint,
    validate_q3_dynamic_fluid_handoff_v92,
    validate_q3_dynamic_fluid_result_v92,
)
from braids_core.sediment.q3_producer_lifecycle_binding_v84 import (
    Q3AcceptedProducerRowsV84,
    _producer_token,
    validate_binding_executions_v84,
)
from braids_core.sediment.route_lifecycle_composition_v81 import (
    RouteLifecycleExecutionV81,
)
from braids_core.sediment.shared_turbulence_consumers_v91 import (
    HydraulicTurbulenceReceiptV91,
)
from braids_core.sediment.transported_fluid_settling_v92 import (
    TransportedFluidExecutionV92,
    TransportedFluidStateV92,
)
from .delft_direct_current_v66 import DelftDirectCurrentV66CUDA


ROOT = Path(__file__).resolve().parents[1]


class ResidentQ3DynamicFluidProducerV92:
    """Run Bijker/19/20 on CUDA with exact transported-parent fluid fields."""

    def __init__(self, *, producer_execution: Mapping[str, object],
                 lifecycle_execution: RouteLifecycleExecutionV81,
                 fluid_execution: TransportedFluidExecutionV92,
                 root: str | Path | None = None):
        self.root = Path(root or ROOT)
        self.fluid_execution = fluid_execution
        self.execution = validate_binding_executions_v84(
            producer_execution, lifecycle_execution,
            backend="implicit_fwave_midpoint_gpu", root=self.root)
        self.producer_execution = load_direct_vector_family_execution_v67(
            producer_execution)
        if fluid_execution.fraction_count != 3:
            raise Q3DynamicFluidV92Error(
                "v92 Q3 CUDA producer requires three transported fractions")
        try:
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount() < 1:
                raise Q3DynamicFluidV92Error(
                    "v92 Q3 dynamic CUDA requires an actual device")
        except Q3DynamicFluidV92Error:
            raise
        except Exception as error:
            raise Q3DynamicFluidV92Error(
                "v92 Q3 dynamic CUDA runtime is unavailable") from error
        self.cp = cp
        self.device_id = int(cp.cuda.Device().id)
        self.direct = DelftDirectCurrentV66CUDA()
        if self.direct.device_id != self.device_id:
            raise Q3DynamicFluidV92Error(
                "v92 Q3 dynamic CUDA producer device differs")
        self.sediment_threshold_m = (
            fluid_execution.base_execution.base_execution.base_execution
            .profile.sediment_threshold_m)
        paths = (
            Path(__file__).resolve(),
            ROOT/"braids_gpu/delft_direct_current_v66.py",
            ROOT/"braids_gpu/delft_direct_current_v66.cu",
        )
        self.runtime_receipt = {
            "backend": "actual_cuda",
            "automatic_cpu_fallback": False,
            "device_id": self.device_id,
            "v84_execution_fingerprint": self.execution["fingerprint"],
            "cell_water_density": True,
            "source_sha256": {
                str(path.relative_to(ROOT)).replace("\\", "/"):
                    hashlib.sha256(path.read_bytes()).hexdigest()
                for path in paths
            },
        }

    def _full(self, value, count: int, *, integer: bool = False):
        dtype = self.cp.int32 if integer else self.cp.float64
        return self.cp.full(count, value, dtype=dtype)

    def _common_1920(self, *, depth, u, v, rho, nu, chezy,
                     d50: float, d90: float, solid_density: float):
        nc = int(depth.size)
        zeros = self.cp.zeros(nc, dtype=self.cp.float64)
        return {
            "depth": depth,
            "u": u,
            "v": v,
            "d15": zeros.copy(),
            "d50": self._full(d50, nc),
            "d90": self._full(d90, nc),
            "solid_density": self._full(solid_density, nc),
            "water_density": rho,
            "gravity": self._full(
                self.producer_execution["physical_profile"]["gravity_m_s2"],
                nc),
            "kinematic_viscosity": nu,
            "chezy": chezy,
            "facua": zeros.copy(),
            "facas": self._full(0.2, nc),
            "facsk": self._full(0.15, nc),
            "waveform": self._full(2, nc, integer=True),
            "sws": self._full(1, nc, integer=True),
            "lws": self._full(1, nc, integer=True),
            "dilatancy": self._full(0, nc, integer=True),
            "rheea": self._full(0.75, nc),
            "pormax": self._full(0.5, nc),
            "bedslpini": self._full(0, nc, integer=True),
            "smax": self._full(-1.0, nc),
            "reposeangle": self._full(30.0, nc),
            "cmax": self._full(0.1, nc),
            "alfad50": zeros.copy(),
            "wave_height": zeros.copy(),
            "wave_period": zeros.copy(),
            "wavelength": zeros.copy(),
            "wave_angle": zeros.copy(),
            "orbital_velocity": zeros.copy(),
            "breaker_turbulence": zeros.copy(),
            "bed_change_rate": zeros.copy(),
            "bed_slope_x": zeros.copy(),
            "bed_slope_y": zeros.copy(),
        }

    def produce(
        self, parent: TransportedFluidStateV92,
        hydraulic_receipt: HydraulicTurbulenceReceiptV91,
        handoff: Q3DynamicFluidHandoffV92,
    ) -> Q3DynamicFluidAcceptedRowsV92:
        handoff = validate_q3_dynamic_fluid_handoff_v92(
            handoff, self.fluid_execution, parent, hydraulic_receipt)
        cp = self.cp
        q = cp.ascontiguousarray(cp.asarray(
            handoff.accepted_q, dtype=cp.float64))
        depth = q[0]
        dry = (self.fluid_execution.base_execution.base_execution
               .base_execution.dry_tolerance_m)
        active = (depth > dry) & (depth > self.sediment_threshold_m)
        safe_depth = cp.where(active, depth, 1.0)
        u = cp.where(active, q[1]/safe_depth, 0.0)
        v = cp.where(active, q[2]/safe_depth, 0.0)
        u = cp.where(cp.abs(u) < 1.0e-12, 0.0, u)
        v = cp.where(cp.abs(v) < 1.0e-12, 0.0, v)
        speed = cp.hypot(u, v)
        effective_u = cp.ascontiguousarray(speed*u/(speed+1.0e-6))
        effective_v = cp.ascontiguousarray(speed*v/(speed+1.0e-6))
        physical = self.producer_execution["physical_profile"]
        work_depth = cp.maximum(
            safe_depth, float(physical["minimum_work_depth_m"]))
        roughness = cp.ascontiguousarray(cp.asarray(
            handoff.roughness_parameter, dtype=cp.float64))
        chezy = cp.ascontiguousarray(work_depth**(1.0/6.0)/roughness)
        rho = cp.ascontiguousarray(cp.asarray(
            handoff.water_density_cell_kg_m3, dtype=cp.float64))
        nu = cp.ascontiguousarray(cp.asarray(
            handoff.molecular_viscosity_cell_m2_s, dtype=cp.float64))
        ws = cp.ascontiguousarray(cp.asarray(
            handoff.settling_velocity_m_s, dtype=cp.float64))
        nc = self.fluid_execution.cell_count
        zeros = cp.zeros(nc, dtype=cp.float64)
        gravity = self._full(physical["gravity_m_s2"], nc)

        bijker = {
            "depth": safe_depth, "u": effective_u, "v": effective_v,
            "d50": self._full(physical["d50_m"][0], nc),
            "d90": self._full(physical["d90_m"][0], nc),
            "solid_density": self._full(
                physical["solid_density_kg_m3"][0], nc),
            "water_density": rho, "gravity": gravity, "chezy": chezy,
            "settling_velocity": cp.ascontiguousarray(ws[0]),
            "cal_bs": self._full(5.0, nc),
            "cal_bd": self._full(4.0, nc),
            "crit_cs": self._full(0.4, nc),
            "crit_cd": self._full(0.05, nc),
            "roughness_ks": self._full(-999.0, nc),
            "formula_settling_velocity": self._full(-999.0, nc),
            "porosity": self._full(0.4, nc),
            "formula_wave_period": self._full(5.0, nc),
            "fac_a": zeros.copy(), "fac_u": zeros.copy(),
            "eps_sl": zeros.copy(), "wave_height": zeros.copy(),
            "wave_period": zeros.copy(), "wavelength": zeros.copy(),
            "wave_angle": zeros.copy(), "orbital_velocity": zeros.copy(),
            "breaker_turbulence": zeros.copy(),
            "bed_slope_x": zeros.copy(), "bed_slope_y": zeros.copy(),
        }
        result5 = self.direct.evaluate(5, **bijker)
        common19 = self._common_1920(
            depth=safe_depth, u=effective_u, v=effective_v, rho=rho, nu=nu,
            chezy=chezy, d50=physical["d50_m"][1],
            d90=physical["d90_m"][1],
            solid_density=physical["solid_density_kg_m3"][1])
        result19 = self.direct.evaluate(19, **common19)
        common20 = self._common_1920(
            depth=safe_depth, u=effective_u, v=effective_v, rho=rho, nu=nu,
            chezy=chezy, d50=physical["d50_m"][2],
            d90=physical["d90_m"][2],
            solid_density=physical["solid_density_kg_m3"][2])
        common20["z0"] = self._full(0.006, nc)
        result20 = self.direct.evaluate(20, **common20)
        # The fifth Bijker kernel row is a producer diagnostic, not the
        # equilibrium-concentration channel owned by the Q3/Q4 lifecycle.
        # Match the accepted v67 adapter contract: explicit Bijker bed and
        # suspended vectors, with a zero equilibrium-concentration row.
        bijker_rows = cp.ascontiguousarray(result5.values[:5].copy())
        bijker_rows[4] = 0.0
        raw_device = cp.stack(
            (bijker_rows, result19.values[:5], result20.values[:5]), axis=1)
        raw_device = cp.where(active[None, None, :], raw_device, 0.0)
        raw = np.ascontiguousarray(cp.asnumpy(raw_device))
        wet_i = np.ascontiguousarray(cp.asnumpy(active).astype(np.int32))
        raw.setflags(write=False)
        wet_i.setflags(write=False)
        producer_receipt = _fingerprint({
            "scope_id": "braids-q4-q3-dynamic-fluid-handoff-v92-v1:cuda",
            "v84_execution": self.execution["fingerprint"],
            "handoff_sha256": handoff.handoff_sha256,
            "runtime_receipt": self.runtime_receipt,
        })
        placeholder = Q3AcceptedProducerRowsV84(
            self.execution["fingerprint"], "actual_cuda", raw, wet_i,
            producer_receipt,
            self.execution["producer_source_receipt_fingerprint"], 0, 0, "")
        producer = Q3AcceptedProducerRowsV84(
            **{**placeholder.__dict__,
               "producer_sha256": _producer_token(placeholder)})
        result = _accepted_result(
            handoff, self.execution["fingerprint"], "actual_cuda", producer)
        return validate_q3_dynamic_fluid_result_v92(
            result, handoff, self.execution["fingerprint"])


__all__ = ["ResidentQ3DynamicFluidProducerV92"]
