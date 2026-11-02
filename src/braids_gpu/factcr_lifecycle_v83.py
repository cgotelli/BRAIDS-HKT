"""Actual-CUDA owner for the Q4/v83 route-aware ``FactCr`` lifecycle."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np

from braids_core.sediment.factcr_lifecycle_v83 import (
    FORMATION_UNIFORM_D50,
    SPEED_EPSILON,
    FactCrExecutionV83,
    FactCrStateV83,
    FactCrTrialV83,
    FactCrV83Error,
    _build_trial,
    _validate_trial,
    factcr_checkpoint_v83,
    load_factcr_checkpoint_v83,
    validate_factcr_state_v83,
)
from braids_core.sediment.multifraction_suspended_lifecycle_v78 import (
    MAX_ACCEPTED_UPDATES,
    ROLE_CUDA,
)
from braids_gpu.multifraction_suspended_lifecycle_v78 import (
    ResidentMultifractionSuspendedLifecycleV78,
)
from braids_gpu.van_rijn_1993_route_v66 import (
    ROUTE_COMBINE,
    ResidentVanRijn1993RouteV66,
)


ROOT = Path(__file__).resolve().parents[1]
CUDA_SOURCE = ROOT/"braids_gpu/factcr_lifecycle_v83.cu"
CUDA_SCOPE_ID = "braids-q4-flow2dh-factcr-lifecycle-cuda-v83-v1"
CUDA_THREADS_PER_BLOCK = 128
CUDA_OPTIONS = (
    "--std=c++11", "--fmad=false", "--prec-div=true",
    "--prec-sqrt=true", "--ftz=false",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FactCrV83Error(message)


class ResidentFactCrLifecycleV83:
    """One explicit-device transaction spanning FactCr, v66 and v78."""

    def __init__(self, execution: FactCrExecutionV83):
        try:
            import cupy as cp
        except Exception as error:  # pragma: no cover - host dependent
            raise FactCrV83Error(
                "v83 actual CUDA requires CuPy and a working device") from error
        try:
            _require(cp.cuda.runtime.getDeviceCount() >= 1,
                     "v83 actual CUDA requires a CUDA device")
        except FactCrV83Error:
            raise
        except Exception as error:  # pragma: no cover - host dependent
            raise FactCrV83Error("v83 CUDA discovery failed") from error
        _require(isinstance(execution, FactCrExecutionV83) and
                 execution.role == ROLE_CUDA,
                 "v83 CUDA owner requires an actual_cuda execution")
        _require(CUDA_SOURCE.is_file(), "v83 CUDA threshold source is absent")
        self.cp = cp
        self.execution = execution
        self.device_id = int(cp.cuda.Device().id)
        source = CUDA_SOURCE.read_bytes()
        try:
            module = cp.RawModule(
                code=source.decode("utf-8"), options=CUDA_OPTIONS,
                name_expressions=("factcr_threshold_v83",))
            self.threshold_kernel = module.get_function(
                "factcr_threshold_v83")
        except Exception as error:
            raise FactCrV83Error(
                "v83 CUDA threshold kernel compilation failed") from error
        self.producer = ResidentVanRijn1993RouteV66(ROUTE_COMBINE)
        self.lifecycle = ResidentMultifractionSuspendedLifecycleV78(
            execution.lifecycle_execution)
        _require(self.producer.device_id == self.device_id and
                 self.lifecycle.device_id == self.device_id,
                 "v83 CUDA components selected different devices")
        properties = cp.cuda.runtime.getDeviceProperties(self.device_id)
        raw_name = properties.get("name", b"unknown")
        device_name = (raw_name.decode("utf-8", errors="replace")
                       if isinstance(raw_name, bytes) else str(raw_name))
        source_paths = (
            ROOT/"braids_core/sediment/factcr_lifecycle_v83.py",
            ROOT/"braids_gpu/factcr_lifecycle_v83.py",
            CUDA_SOURCE,
            ROOT/"braids_gpu/van_rijn_1993_route_v66.py",
            ROOT/"braids_gpu/van_rijn_1993_route_v66.cu",
        )
        self.runtime_receipt = {
            "scope_id": CUDA_SCOPE_ID,
            "execution_fingerprint": execution.fingerprint,
            "backend": "actual_cuda",
            "integrator": "implicit_fwave_midpoint",
            "formula_code": -1,
            "itaucr": 1,
            "factcr": execution.factcr,
            "critical_shear_formation_mode": FORMATION_UNIFORM_D50,
            "factcr_application_count": 1,
            "fixed_grid": True,
            "actual_cuda_threshold_kernel": True,
            "actual_cuda_v66_producer": True,
            "actual_cuda_v78_lifecycle": True,
            "automatic_cpu_fallback": False,
            "device_id": self.device_id,
            "device_name": device_name,
            "threads_per_block": CUDA_THREADS_PER_BLOCK,
            "source_sha256": {
                str(path.relative_to(ROOT)).replace("\\", "/"):
                    hashlib.sha256(path.read_bytes()).hexdigest()
                for path in source_paths
            },
            "lifecycle_runtime_receipt": self.lifecycle.runtime_receipt,
            "host_readback_scope":
                "immutable candidate authentication/checkpoint evidence only",
            "host_readback_failure_rolls_back_inner_trial": True,
        }
        self._open_parent: str | None = None
        self._open_trial: str | None = None

    def _device(self, value, shape, dtype, name):
        cp = self.cp
        _require(isinstance(value, cp.ndarray) and value.shape == shape and
                 value.dtype == dtype and value.flags.c_contiguous and
                 int(value.device.id) == self.device_id,
                 f"v83 CUDA {name} must be resident contiguous {dtype} {shape}")
        return value

    def _host_readback(self, values):
        """Read immutable trial evidence after the resident candidate exists."""
        return [self.cp.asnumpy(value) for value in values]

    def begin_trial(
        self, parent: FactCrStateV83, *, accepted_step_receipt,
        depth_before_m, depth_after_m, accepted_water_flux_m2_s,
        reconstructed_left_concentration_kg_m3,
        reconstructed_right_concentration_kg_m3,
        prescribed_boundary_concentration_kg_m3,
        accepted_face_exchange_area_m2, transport_face_active,
        endpoint_momentum_m2_s, shear_velocity_m_s, hydro_constant,
        observation_height_m,
        water_density_kg_m3, gravity_m_s2, kinematic_viscosity_m2_s,
        additional_shear_stress_pa, sediment_d50_m, sediment_d90_m,
        sediment_density_kg_m3, molecular_schmidt, aks_factor,
        maximum_volume_concentration, minimum_depth_m, bed_factor,
        suspended_factor, settling_velocity_m_s, roughness_z0_m,
        surface_fraction, dt_s: float, cancelled: bool = False,
    ) -> FactCrTrialV83:
        cp = self.cp
        validate_factcr_state_v83(parent, self.execution)
        _require(self._open_trial is None,
                 "v83 CUDA previous trial is unfinished")
        if cancelled:
            raise FactCrV83Error(
                "v83 CUDA update cancelled before candidate publication")
        if parent.accepted_updates >= MAX_ACCEPTED_UPDATES:
            raise FactCrV83Error("v83 CUDA accepted update limit was reached")
        _require(not isinstance(dt_s, bool) and
                 isinstance(dt_s, (int, float)) and
                 math.isfinite(float(dt_s)) and float(dt_s) > 0.0,
                 "v83 CUDA timestep must be finite and positive")
        ng, nc = self.execution.fraction_count, self.execution.cell_count
        base = self.execution.lifecycle_execution.base_execution
        nf, nb = base.face_count, base.boundary_count
        h0 = self._device(depth_before_m, (nc,), cp.float64, "depth before")
        h1 = self._device(depth_after_m, (nc,), cp.float64, "depth after")
        water = self._device(accepted_water_flux_m2_s, (nf,), cp.float64,
                             "accepted water flux")
        left = self._device(reconstructed_left_concentration_kg_m3,
                            (ng, nf), cp.float64, "left reconstruction")
        right = self._device(reconstructed_right_concentration_kg_m3,
                             (ng, nf), cp.float64, "right reconstruction")
        prescribed = self._device(prescribed_boundary_concentration_kg_m3,
                                  (ng, nb), cp.float64, "boundary concentration")
        face_area = self._device(accepted_face_exchange_area_m2, (nf,),
                                 cp.float64, "face exchange area")
        face_active = self._device(transport_face_active, (nf,), cp.int32,
                                   "face activity")
        momentum = self._device(endpoint_momentum_m2_s, (2, nc), cp.float64,
                                "endpoint momentum")
        cell_shear = self._device(shear_velocity_m_s, (nc,), cp.float64,
                                  "accepted hydraulic cell shear velocity")
        constant = self._device(hydro_constant, (nc,), cp.int32,
                                "hydro constant")
        zvel = self._device(observation_height_m, (nc,), cp.float64,
                            "observation height")
        rho = self._device(water_density_kg_m3, (nc,), cp.float64,
                           "water density")
        gravity = self._device(gravity_m_s2, (nc,), cp.float64, "gravity")
        viscosity = self._device(kinematic_viscosity_m2_s, (nc,), cp.float64,
                                 "kinematic viscosity")
        tauadd = self._device(additional_shear_stress_pa, (nc,), cp.float64,
                              "additional shear")
        d50 = self._device(sediment_d50_m, (ng,), cp.float64, "D50")
        d90 = self._device(sediment_d90_m, (ng,), cp.float64, "D90")
        rho_s = self._device(sediment_density_kg_m3, (ng,), cp.float64,
                             "solid density")
        schmidt = self._device(molecular_schmidt, (ng,), cp.float64,
                               "molecular Schmidt")
        aks = self._device(aks_factor, (ng,), cp.float64, "Aks factor")
        camax = self._device(maximum_volume_concentration, (ng,), cp.float64,
                             "maximum volume concentration")
        hmin = self._device(minimum_depth_m, (ng,), cp.float64,
                            "minimum depth")
        bedfac = self._device(bed_factor, (ng,), cp.float64, "bed factor")
        susfac = self._device(suspended_factor, (ng,), cp.float64,
                              "suspended factor")
        ws = self._device(settling_velocity_m_s, (ng, nc), cp.float64,
                          "settling velocity")
        z0 = self._device(roughness_z0_m, (ng, nc), cp.float64,
                          "roughness z0")
        fraction = self._device(surface_fraction, (ng, nc), cp.float64,
                                "surface fraction")
        floating = (h0, h1, water, left, right, prescribed, face_area,
                    momentum, cell_shear, zvel, rho, gravity, viscosity,
                    tauadd, d50,
                    d90, rho_s, schmidt, aks, camax, hmin, bedfac, susfac,
                    ws, z0, fraction)
        _require(all(bool(cp.all(cp.isfinite(value)).item())
                     for value in floating),
                 "v83 CUDA inputs must be finite")
        _require(bool(cp.all(h1 >= 0.0).item()) and
                 bool(cp.all(d50 > 0.0).item()) and
                 bool(cp.all(d90 >= d50).item()) and
                 bool(cp.all(rho_s[:, None] > rho[None, :]).item()) and
                 bool(cp.all((fraction >= 0.0) & (fraction <= 1.0)).item()) and
                 bool(cp.all(cell_shear >= 0.0).item()),
                 "v83 CUDA physical input bounds differ")
        _require(bool(cp.all(cp.abs(cp.sum(fraction, axis=0)-1.0) <=
                                32.0*np.finfo(np.float64).eps).item()),
                 "v83 CUDA surface fractions must sum to one")
        profile = base.profile
        _require(bool(cp.array_equal(
            d50, cp.asarray(profile.sediment_d50_m, dtype=cp.float64)).item()) and
            bool(cp.array_equal(
                rho_s, cp.asarray(profile.sediment_density_kg_m3,
                                  dtype=cp.float64)).item()) and
            bool(cp.all(rho == profile.water_density_kg_m3).item()) and
            bool(cp.all(gravity == profile.gravity_m_s2).item()),
            "v83 CUDA producer properties differ from the lifecycle")

        shape = (ng, nc)
        threshold = [cp.empty(shape, dtype=cp.float64) for _ in range(4)]
        status = cp.empty(shape, dtype=cp.int32)
        count = ng*nc
        blocks = (count+CUDA_THREADS_PER_BLOCK-1)//CUDA_THREADS_PER_BLOCK
        self.threshold_kernel(
            (blocks,), (CUDA_THREADS_PER_BLOCK,),
            (np.int64(nc), np.int32(ng), np.float64(self.execution.factcr),
             d50, rho_s, rho, gravity, viscosity, *threshold, status))
        cp.cuda.get_current_stream().synchronize()
        if bool(cp.any(status != 0).item()):
            raise FactCrV83Error("v83 CUDA threshold kernel rejected a point")
        dstar, theta, base_taucr, effective_taucr = threshold
        wet = h1 > base.dry_tolerance_m
        velocity = cp.ascontiguousarray(cp.where(
            wet[None, :], momentum/cp.where(wet, h1, 1.0)[None, :], 0.0))
        produced = self.producer.evaluate(
            depth=h1, velocity=velocity, observation_height=zvel,
            water_density=rho, gravity=gravity,
            kinematic_viscosity=viscosity,
            additional_shear_stress=tauadd, d50=d50, d90=d90,
            solid_density=rho_s, molecular_schmidt=schmidt,
            aks_factor=aks, maximum_volume_concentration=camax,
            minimum_depth=hmin, bed_factor=bedfac,
            suspended_factor=susfac, settling_velocity=ws,
            roughness_z0=z0, dstar=cp.ascontiguousarray(dstar),
            critical_shear_stress=cp.ascontiguousarray(effective_taucr),
            surface_fraction=fraction)
        sediment_active = wet & (h1 > profile.sediment_threshold_m)
        exchange_depth = cp.where(
            sediment_active,
            cp.maximum(profile.reference_depth_floor_m, h1), 1.0)
        speed = cp.hypot(velocity[0], velocity[1])
        capacity = (susfac[:, None]*produced.raw_conc2d*
                    (speed[None, :]+SPEED_EPSILON)*exchange_depth[None, :])
        capacity[:, ~sediment_active] = 0.0
        target = cp.ascontiguousarray(produced.equilibrium_concentration)
        target[:, ~sediment_active] = 0.0
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
            shear_velocity_m_s=cell_shear,
            suspended_transport_capacity_kg_m_s=
                cp.ascontiguousarray(capacity),
            surface_fraction=fraction, hydro_constant=constant,
            dt_s=float(dt_s))
        try:
            host = self._host_readback((
                dstar, theta, base_taucr, effective_taucr,
                produced.raw_bed_mass, produced.raw_conc2d,
                produced.bed_mass, capacity, target, produced.diagnostics))
            trial = _build_trial(
                self.execution, parent, lifecycle_trial,
                accepted_step_receipt, dstar=host[0],
                critical_shields=host[1], unfactored_taucr=host[2],
                effective_taucr=host[3], raw_bed=host[4], raw_conc=host[5],
                bed_capacity=host[6], lifecycle_capacity=host[7],
                target=host[8], diagnostics=host[9], backend=ROLE_CUDA)
        except Exception as error:
            try:
                self.lifecycle.rollback(parent.lifecycle, lifecycle_trial)
            except Exception as rollback_error:
                raise FactCrV83Error(
                    "v83 CUDA candidate authentication failed and resident "
                    "rollback also failed") from rollback_error
            if self._open_trial is not None or self._open_parent is not None:
                raise FactCrV83Error(
                    "v83 CUDA outer transaction opened before authentication") \
                    from error
            raise
        self._open_parent = parent.state_sha256
        self._open_trial = trial.trial_sha256
        return trial

    def _finish(self, parent: FactCrStateV83, trial: FactCrTrialV83, *,
                commit: bool) -> FactCrStateV83:
        _validate_trial(self.execution, parent, trial)
        _require(self._open_parent == parent.state_sha256 and
                 self._open_trial == trial.trial_sha256,
                 "v83 CUDA trial is not the resident open transaction")
        base = (self.lifecycle.commit(parent.lifecycle, trial.lifecycle_trial)
                if commit else
                self.lifecycle.rollback(parent.lifecycle,
                                        trial.lifecycle_trial))
        result = trial.candidate if commit else parent
        if commit:
            _require(base.state_sha256 == result.lifecycle.state_sha256,
                     "v83 CUDA committed lifecycle differs from candidate")
        else:
            _require(base is parent.lifecycle,
                     "v83 CUDA rollback changed the accepted lifecycle")
        self._open_parent = None
        self._open_trial = None
        return result

    def commit(self, parent: FactCrStateV83,
               trial: FactCrTrialV83) -> FactCrStateV83:
        return self._finish(parent, trial, commit=True)

    def rollback(self, parent: FactCrStateV83,
                 trial: FactCrTrialV83) -> FactCrStateV83:
        return self._finish(parent, trial, commit=False)

    def cancel(self, parent: FactCrStateV83,
               trial: FactCrTrialV83) -> FactCrStateV83:
        return self.rollback(parent, trial)

    def checkpoint(self, state: FactCrStateV83) -> dict[str, object]:
        _require(self._open_trial is None,
                 "v83 CUDA checkpoint requires a committed point")
        return factcr_checkpoint_v83(self.execution, state)

    def restore_checkpoint(self, document) -> FactCrStateV83:
        _require(self._open_trial is None,
                 "v83 CUDA restart cannot replace an open trial")
        state = load_factcr_checkpoint_v83(self.execution, document)
        restored = self.lifecycle.restore_checkpoint(
            document["lifecycle_checkpoint_v78"])
        _require(restored.state_sha256 == state.lifecycle.state_sha256,
                 "v83 CUDA resident restart differs")
        return state


def method_identity_v83_cuda() -> dict[str, object]:
    return {
        "scope_id": CUDA_SCOPE_ID,
        "integrator": "implicit_fwave_midpoint",
        "formula_code": -1,
        "itaucr": 1,
        "factcr_exactly_once": True,
        "critical_shear_formation_mode": FORMATION_UNIFORM_D50,
        "actual_cuda_threshold_kernel": True,
        "actual_cuda_v66_producer": True,
        "actual_cuda_v78_lifecycle": True,
        "host_readback_failure_rolls_back_inner_trial": True,
        "automatic_cpu_fallback": False,
        "fixed_grid_only": True,
        "accepted_update_limit": MAX_ACCEPTED_UPDATES,
    }


__all__ = [
    "CUDA_SCOPE_ID", "CUDA_THREADS_PER_BLOCK", "ResidentFactCrLifecycleV83",
    "method_identity_v83_cuda",
]
