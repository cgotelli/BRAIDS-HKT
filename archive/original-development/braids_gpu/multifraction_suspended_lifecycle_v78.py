"""Actual-CUDA multi-fraction suspended lifecycle owner for Q4/v78.

The adapter exercises the already generic class dimension of the accepted
v76 CUDA FCT, boundary/diffusion, and Galappatti kernels while leaving all v74
and v76 files byte-for-byte unchanged.  The new admission layer requires at
least two authenticated noncohesive fractions and an accepted
``implicit_fwave_midpoint`` step receipt.  There is no CPU fallback.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np

from braids_core.sediment.multifraction_suspended_lifecycle_v78 import (
    MAX_ACCEPTED_UPDATES,
    ROLE_CUDA,
    MultifractionSuspendedExecutionV78,
    MultifractionSuspendedStateV78,
    MultifractionSuspendedTrialV78,
    MultifractionSuspendedV78Error,
    build_multifraction_trial_v78,
    commit_multifraction_trial_v78,
    load_multifraction_checkpoint_v78,
    multifraction_checkpoint_v78,
    rollback_multifraction_trial_v78,
    validate_implicit_fwave_midpoint_step_receipt_v78,
    validate_multifraction_suspended_state_v78,
)
from braids_core.sediment.suspended_lifecycle_live_v76 import (
    BACKEND_CUDA,
    resolve_suspended_lifecycle_state_v76,
)
from braids_gpu.flow2dh_boundary_diffusion import KERNELS as BOUNDARY_KERNELS
from braids_gpu.flow2dh_boundary_diffusion import OPTIONS as CUDA_OPTIONS
from braids_gpu.law11_soursin_exchange import KERNEL as EXCHANGE_KERNEL
from validation.gpu_fwave.suspended_advection import (
    ResidentSuspendedAdvection,
)


CUDA_THREADS_PER_BLOCK = 128
CUDA_COMPOSITION_ID = (
    "resident_multifraction_fct_boundary_diffusion_settlefrm2_"
    "galappatti_v78_v1"
)
ROOT = Path(__file__).resolve().parents[1]
BOUNDARY_SOURCE = ROOT/"braids_gpu/flow2dh_boundary_diffusion.cu"
EXCHANGE_SOURCE = ROOT/"braids_gpu/law11_soursin_exchange.cu"
ADVECTION_SOURCE = ROOT/"validation/gpu_fwave/suspended_advection.cu"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MultifractionSuspendedV78Error(message)


class _ResidentMultifractionBoundaryDiffusionV78:
    """v78 admission and storage for the generic accepted v76 kernels."""

    def __init__(self, execution, advection):
        import cupy as cp

        self.cp = cp
        self.advection = advection
        self.nc, self.nf = advection.nc, advection.nf
        self.ng, self.nb = advection.ng, advection.nb
        profile = execution.profile
        _require(self.ng == execution.fraction_count and self.ng >= 2,
                 "v78 boundary adapter requires multiple fractions")
        _require(profile.sediment_fraction_classes == ("sand",)*self.ng,
                 "v78 boundary adapter admits noncohesive classes only")
        self.profile = profile
        self.advection_dry = float(execution.dry_tolerance_m)
        self.dry = float(profile.diffusion_dry_threshold_m)
        self.prescribed_neumann = np.ascontiguousarray(
            profile.neumann_boundary_by_fraction, dtype=np.int32)
        self.neumann = cp.asarray(self.prescribed_neumann)
        self.dicuv = cp.full(
            self.nc, profile.horizontal_diffusivity_m2_s, dtype=cp.float64)

        geometry = advection.geometry
        self.owner = geometry["owner"]
        self.neighbour = geometry["neighbour"]
        self.bid = geometry["bid"]
        self.offset = geometry["offset"]
        self.csr_faces = geometry["faces"]
        self.signs = geometry["signs"]
        self.length = geometry["length"]
        self.area = geometry["area"]
        self.centre_distance = cp.asarray(np.ascontiguousarray(
            execution.centre_distance_m, dtype=np.float64))

        bids = np.asarray(execution.boundary_index)
        lists = [np.flatnonzero(bids == boundary).astype(np.int32)
                 for boundary in range(self.nb)]
        boundary_faces = (np.concatenate(lists) if lists else
                          np.empty(0, dtype=np.int32))
        boundary_offset = np.r_[
            0, np.cumsum([len(value) for value in lists])].astype(np.int32)
        self.boundary_faces = cp.asarray(boundary_faces)
        self.boundary_offset = cp.asarray(boundary_offset)

        return_time = np.zeros((self.ng, self.nf), dtype=np.float64)
        open_face = bids >= 0
        if self.nb:
            return_time[:, open_face] = execution.boundary_return_time_s[
                :, bids[open_face]]
        self.return_time_host = return_time
        self.return_time = cp.asarray(return_time)

        try:
            module = cp.RawModule(
                code=BOUNDARY_SOURCE.read_text(encoding="utf-8"),
                options=CUDA_OPTIONS, name_expressions=BOUNDARY_KERNELS)
            self.k = {name: module.get_function(name)
                      for name in BOUNDARY_KERNELS}
        except Exception as error:
            raise MultifractionSuspendedV78Error(
                "v78 CUDA boundary/diffusion kernel compilation failed") \
                from error

        ng, nf, nc, nb = self.ng, self.nf, self.nc, self.nb
        self.accepted_last = cp.zeros((ng, nf), dtype=cp.float64)
        self.accepted_clock = cp.zeros((ng, nf), dtype=cp.float64)
        self.trial_last = cp.zeros_like(self.accepted_last)
        self.trial_clock = cp.zeros_like(self.accepted_clock)
        self.face_concentration = cp.zeros((ng, nf), dtype=cp.float64)
        self.expected_advective_mass_in = cp.zeros(
            (ng, nf), dtype=cp.float64)
        self.expected_advective_ledger = cp.zeros(
            (ng, nb), dtype=cp.float64)
        self.expected_advective_ledger_status = cp.zeros(
            (ng, nb), dtype=cp.int32)
        self.boundary_status = cp.zeros((ng, nf), dtype=cp.int32)
        self.effective_diffusivity = cp.zeros((ng, nf), dtype=cp.float64)
        self.diffusive_face_mass_owner_out = cp.zeros(
            (ng, nf), dtype=cp.float64)
        self.diffusive_face_active = cp.zeros((ng, nf), dtype=cp.int32)
        self.diffusive_face_status = cp.zeros((ng, nf), dtype=cp.int32)
        self.diffusive_theta = cp.ones((ng, nc), dtype=cp.float64)
        self.diffusive_theta_status = cp.zeros((ng, nc), dtype=cp.int32)
        self.next_mass = cp.zeros((ng, nc), dtype=cp.float64)
        self.diffusive_cell_status = cp.zeros((ng, nc), dtype=cp.int32)
        self.diffusive_ledger = cp.zeros((ng, nb), dtype=cp.float64)
        self.diffusive_ledger_status = cp.zeros(
            (ng, nb), dtype=cp.int32)

    def _launch(self, name, count, args):
        if count:
            self.k[name](((count+CUDA_THREADS_PER_BLOCK-1)//
                          CUDA_THREADS_PER_BLOCK,),
                         (CUDA_THREADS_PER_BLOCK,), args)

    def _array(self, value, shape, dtype, name):
        _require(isinstance(value, self.cp.ndarray) and
                 value.shape == shape and value.dtype == dtype and
                 value.flags.c_contiguous,
                 f"v78 CUDA {name} must be contiguous device {dtype} {shape}")

    def restore_checkpoint(self, state) -> None:
        last = np.asarray(
            state.boundary_last_outflow_concentration_kg_m3)
        clock = np.asarray(state.boundary_return_clock_s)
        shape = (self.ng, self.nf)
        _require(last.shape == clock.shape == shape and
                 last.dtype == clock.dtype == np.float64 and
                 np.isfinite(last).all() and np.isfinite(clock).all() and
                 np.all(last >= 0.0) and np.all(clock >= 0.0),
                 "v78 CUDA boundary checkpoint memory differs")
        positive = self.return_time_host > 0.0
        tolerance = (256.0*np.finfo(np.float64).eps*
                     np.maximum(1.0, self.return_time_host))
        _require(np.all(clock[positive] <=
                        self.return_time_host[positive]+tolerance[positive]) and
                 np.all(clock[~positive] == 0.0),
                 "v78 CUDA boundary return clock is outside its contract")
        self.cp.copyto(self.accepted_last, self.cp.asarray(last))
        self.cp.copyto(self.accepted_clock, self.cp.asarray(clock))
        self.cp.copyto(self.trial_last, self.accepted_last)
        self.cp.copyto(self.trial_clock, self.accepted_clock)

    def resolve_boundary(self, depth, suspended_mass, prescribed_by_id,
                         accepted_water_flux, dt):
        cp = self.cp
        self._array(depth, (self.nc,), cp.float64, "depth")
        self._array(suspended_mass, (self.ng, self.nc), cp.float64,
                    "suspended mass")
        self._array(prescribed_by_id, (self.ng, self.nb), cp.float64,
                    "prescribed boundary concentration")
        self._array(accepted_water_flux, (self.nf,), cp.float64,
                    "accepted water flux")
        self._launch("m2b_resolve_boundary", self.ng*self.nf, (
            np.int32(self.nc), np.int32(self.nf), np.int32(self.ng),
            np.int32(self.nb), np.float64(self.advection_dry), np.float64(dt),
            depth, suspended_mass, prescribed_by_id, accepted_water_flux,
            self.length, self.owner, self.neighbour, self.bid, self.neumann,
            self.return_time, self.accepted_last, self.accepted_clock,
            self.face_concentration, self.expected_advective_mass_in,
            self.trial_last, self.trial_clock, self.boundary_status))
        self._launch("m2b_expected_advective_ledger", self.ng*self.nb, (
            np.int32(self.nf), np.int32(self.ng), np.int32(self.nb),
            self.boundary_offset, self.boundary_faces,
            self.expected_advective_mass_in, self.boundary_status,
            self.expected_advective_ledger,
            self.expected_advective_ledger_status))
        return {
            "concentration": self.face_concentration,
            "expected_advective_ledger": self.expected_advective_ledger,
            "status": (self.boundary_status,
                       self.expected_advective_ledger_status),
        }

    def advect(self, depth, mass, next_depth, accepted_water_flux, left,
               right, hydro_constant, dt):
        cp, adv = self.cp, self.advection
        a, b = adv.geometry, adv.buffers
        nc, nf, ng, nb = map(np.int32,
                             (self.nc, self.nf, self.ng, self.nb))
        dt64 = np.float64(dt)
        adv._launch("sa_concentration", self.nc*self.ng, (
            nc, ng, depth, mass, np.float64(adv.dry), b["concentration"],
            b["concentration_status"]))
        self._launch("m2b_sa_face_low", self.nf*self.ng, (
            nc, nf, ng, accepted_water_flux, b["concentration"], left, right,
            self.face_concentration, a["owner"], a["neighbour"], a["bid"],
            b["concentration_status"], b["low_flux"], b["anti"],
            b["face_status"]))
        self._launch("m2b_sa_cell_bounds", self.nc*self.ng, (
            nc, nf, ng, mass, next_depth, b["concentration"],
            self.face_concentration, accepted_water_flux, b["low_flux"],
            b["anti"], a["area"], a["length"], a["normal"],
            a["rectangle_width"], hydro_constant, a["weno_cells"],
            a["owner"], a["neighbour"], a["bid"], a["offset"], a["faces"],
            a["signs"], b["concentration_status"], b["face_status"], dt64,
            b["low_state"], b["rplus"], b["rminus"], b["cell_status"]))
        adv._launch("sa_limit_faces", self.nf*self.ng, (
            nc, nf, ng, b["low_flux"], b["anti"], b["rplus"], b["rminus"],
            a["owner"], a["neighbour"], b["cell_status"],
            b["scaled_anti"], b["flux"], b["flux_status"]))
        adv._launch("sa_update", self.nc*self.ng, (
            nc, nf, ng, b["low_state"], b["scaled_anti"], next_depth,
            a["length"], a["area"], a["offset"], a["faces"], a["signs"],
            b["cell_status"], b["flux_status"], dt64, np.float64(adv.dry),
            b["next_mass"], b["next_concentration"], b["update_status"]))
        adv._launch("sa_boundary_ledger", self.nb*self.ng, (
            nf, ng, nb, a["boundary_offset"], a["boundary_faces"], b["flux"],
            a["length"], b["flux_status"], dt64, b["ledger"],
            b["ledger_status"]))
        return {
            "next_mass": b["next_mass"], "flux": b["flux"],
            "ledger": b["ledger"],
            "status": (b["concentration_status"], b["face_status"],
                       b["cell_status"], b["flux_status"],
                       b["update_status"], b["ledger_status"]),
        }

    def diffuse(self, endpoint_depth, mass_after_advection,
                accepted_face_exchange_area, transport_face_active, dt):
        cp = self.cp
        self._array(endpoint_depth, (self.nc,), cp.float64, "endpoint depth")
        self._array(mass_after_advection, (self.ng, self.nc), cp.float64,
                    "post-advection mass")
        self._array(accepted_face_exchange_area, (self.nf,), cp.float64,
                    "accepted face exchange area")
        self._array(transport_face_active, (self.nf,), cp.int32,
                    "transport face activity")
        self._launch("m2b_diffusion_faces", self.ng*self.nf, (
            np.int32(self.nc), np.int32(self.nf), np.int32(self.ng),
            np.float64(self.dry), np.float64(dt), endpoint_depth,
            mass_after_advection, self.face_concentration, self.dicuv,
            accepted_face_exchange_area, self.centre_distance,
            transport_face_active, self.owner, self.neighbour, self.bid,
            self.effective_diffusivity, self.diffusive_face_mass_owner_out,
            self.diffusive_face_active, self.diffusive_face_status))
        self._launch("m2b_diffusion_theta", self.ng*self.nc, (
            np.int32(self.nc), np.int32(self.nf), np.int32(self.ng),
            mass_after_advection, self.diffusive_face_mass_owner_out,
            self.area, self.offset, self.csr_faces, self.signs,
            self.diffusive_face_status, self.diffusive_theta,
            self.diffusive_theta_status))
        self._launch("m2b_diffusion_limit", self.ng*self.nf, (
            np.int32(self.nc), np.int32(self.nf), np.int32(self.ng),
            self.owner, self.neighbour, self.diffusive_theta,
            self.diffusive_theta_status, self.diffusive_face_mass_owner_out,
            self.diffusive_face_status))
        self._launch("m2b_diffusion_update", self.ng*self.nc, (
            np.int32(self.nc), np.int32(self.nf), np.int32(self.ng),
            mass_after_advection, self.diffusive_face_mass_owner_out,
            self.area, self.offset, self.csr_faces, self.signs,
            self.diffusive_face_status, self.next_mass,
            self.diffusive_cell_status))
        self._launch("m2b_diffusion_ledger", self.ng*self.nb, (
            np.int32(self.nf), np.int32(self.ng), np.int32(self.nb),
            self.boundary_offset, self.boundary_faces,
            self.diffusive_face_mass_owner_out, self.diffusive_face_status,
            self.diffusive_ledger, self.diffusive_ledger_status))
        return {
            "next_mass": self.next_mass,
            "face_mass_owner_out": self.diffusive_face_mass_owner_out,
            "ledger": self.diffusive_ledger,
            "status": (self.diffusive_face_status,
                       self.diffusive_theta_status,
                       self.diffusive_cell_status,
                       self.diffusive_ledger_status),
        }

    def commit_boundary_memory(self) -> None:
        self.cp.copyto(self.accepted_last, self.trial_last)
        self.cp.copyto(self.accepted_clock, self.trial_clock)


class _ResidentMultifractionExchangeV78:
    """Multi-class admission for the class-general accepted CUDA kernel."""

    def __init__(self, execution):
        import cupy as cp

        self.cp = cp
        self.nc = execution.cell_count
        self.ng = execution.fraction_count
        profile = execution.profile
        _require(self.ng >= 2 and
                 len(profile.sediment_d50_m) == self.ng,
                 "v78 exchange adapter requires multiple fractions")
        self.rho = np.float64(profile.water_density_kg_m3)
        self.dry = np.float64(execution.dry_tolerance_m)
        self.factsd = np.float64(profile.fac_tsd)
        self.temperature = np.float64(profile.temperature_value_c)
        self.reference_concentration = np.float64(
            profile.settling_reference_concentration_kg_m3)
        self.capacity_speed_epsilon = np.float64(1.0e-6)
        self.gravity = np.float64(profile.gravity_m_s2)
        self.reference_depth_floor = np.float64(
            profile.reference_depth_floor_m)
        self.sediment_d50 = cp.asarray(np.ascontiguousarray(
            profile.sediment_d50_m, dtype=np.float64))
        self.settling_fac_dss = cp.asarray(np.ascontiguousarray(
            profile.settling_fac_dss, dtype=np.float64))
        self.sediment_density = cp.asarray(np.ascontiguousarray(
            profile.sediment_density_kg_m3, dtype=np.float64))
        self.sediment_threshold = cp.asarray(np.array(
            [profile.sediment_threshold_m], dtype=np.float64))
        try:
            module = cp.RawModule(
                code=EXCHANGE_SOURCE.read_text(encoding="utf-8"),
                options=CUDA_OPTIONS, name_expressions=(EXCHANGE_KERNEL,))
            self.kernel = module.get_function(EXCHANGE_KERNEL)
        except Exception as error:
            raise MultifractionSuspendedV78Error(
                "v78 CUDA multi-fraction exchange kernel compilation failed") \
                from error
        shape = (self.ng, self.nc)
        self.next_suspended = cp.empty(shape, dtype=cp.float64)
        self.next_active = cp.empty(shape, dtype=cp.float64)
        self.equilibrium_concentration = cp.empty(shape, dtype=cp.float64)
        self.adaptation_velocity = cp.empty(shape, dtype=cp.float64)
        self.dimensionless_timescale = cp.empty(shape, dtype=cp.float64)
        self.clear_water_settling_velocity = cp.empty(shape, dtype=cp.float64)
        self.settling_velocity = cp.empty(shape, dtype=cp.float64)
        self.hindered_factor = cp.empty(self.nc, dtype=cp.float64)
        self.delta = cp.empty(shape, dtype=cp.float64)
        self.limiter_correction = cp.empty(shape, dtype=cp.float64)
        self.sediment_active = cp.empty(self.nc, dtype=cp.int32)
        self.status = cp.empty(self.nc, dtype=cp.int32)

    def _array(self, value, shape, name):
        _require(isinstance(value, self.cp.ndarray) and
                 value.shape == shape and value.dtype == self.cp.float64 and
                 value.flags.c_contiguous,
                 f"v78 CUDA {name} must be contiguous device float64 {shape}")

    def step(self, depth_before, depth_after, endpoint_momentum, suspended,
             active_bed, equilibrium_capacity, surface_fraction, total_shear,
             wet_mask, dt):
        cp = self.cp
        for value, shape, name in (
            (depth_before, (self.nc,), "depth before"),
            (depth_after, (self.nc,), "depth after"),
            (endpoint_momentum, (2, self.nc), "endpoint momentum"),
            (suspended, (self.ng, self.nc), "suspended mass"),
            (active_bed, (self.ng, self.nc), "active bed"),
            (equilibrium_capacity, (self.ng, self.nc),
             "equilibrium capacity"),
            (surface_fraction, (self.ng, self.nc), "surface fraction"),
            (total_shear, (self.nc,), "total shear"),
        ):
            self._array(value, shape, name)
        _require(isinstance(wet_mask, cp.ndarray) and
                 wet_mask.shape == (self.nc,) and
                 wet_mask.dtype == cp.int32 and wet_mask.flags.c_contiguous,
                 "v78 CUDA wet mask must be contiguous device int32")
        self.kernel(((self.nc+CUDA_THREADS_PER_BLOCK-1)//
                     CUDA_THREADS_PER_BLOCK,),
                    (CUDA_THREADS_PER_BLOCK,), (
            np.int32(self.nc), np.int32(self.ng), depth_before, depth_after,
            endpoint_momentum, suspended, active_bed, equilibrium_capacity,
            surface_fraction, total_shear, wet_mask, self.sediment_d50,
            self.settling_fac_dss, self.sediment_density,
            self.sediment_threshold, self.rho, self.temperature,
            self.reference_concentration, self.capacity_speed_epsilon,
            self.gravity, self.dry, self.factsd,
            self.reference_depth_floor, np.float64(dt),
            self.next_suspended, self.next_active,
            self.equilibrium_concentration, self.adaptation_velocity,
            self.dimensionless_timescale,
            self.clear_water_settling_velocity, self.settling_velocity,
            self.hindered_factor, self.delta, self.limiter_correction,
            self.sediment_active, self.status))
        return {
            "suspended": self.next_suspended,
            "active": self.next_active,
            "equilibrium_concentration": self.equilibrium_concentration,
            "settling_velocity": self.settling_velocity,
            "delta": self.delta,
            "status": self.status,
        }


class ResidentMultifractionSuspendedLifecycleV78:
    """One explicit-device, no-fallback v78 transaction owner."""

    def __init__(self, execution: MultifractionSuspendedExecutionV78):
        try:
            import cupy as cp
        except Exception as error:  # pragma: no cover - environment dependent
            raise MultifractionSuspendedV78Error(
                "v78 actual CUDA requires CuPy and a working CUDA device") \
                from error
        try:
            _require(cp.cuda.runtime.getDeviceCount() >= 1,
                     "v78 actual CUDA requires a CUDA device")
        except MultifractionSuspendedV78Error:
            raise
        except Exception as error:  # pragma: no cover - environment dependent
            raise MultifractionSuspendedV78Error(
                "v78 CUDA device discovery failed") from error
        _require(isinstance(execution, MultifractionSuspendedExecutionV78) and
                 execution.role == ROLE_CUDA and
                 execution.base_execution.backend == BACKEND_CUDA,
                 "v78 CUDA owner requires an explicit actual_cuda execution")
        self.cp = cp
        self.execution = execution
        self.base = execution.base_execution
        self.device_id = int(cp.cuda.Device().id)
        geometry = {
            "scheme": np.int32(6), "amr_enabled": np.int32(0),
            "nb": np.int32(self.base.boundary_count),
            "dry_tolerance": np.float64(self.base.dry_tolerance_m),
            "active": np.ones(self.base.cell_count, dtype=np.int32),
            "area": np.array(self.base.cell_area_m2, copy=True),
            "length": np.array(self.base.face_length_m, copy=True),
            "normal": np.array(self.base.normal, copy=True),
            "rectangle_width": np.array(
                self.base.rectangle_width_m, copy=True),
            "weno_cells": np.array(self.base.weno_cells, copy=True),
            "owner": np.array(self.base.owner, copy=True),
            "neighbour": np.array(self.base.neighbour, copy=True),
            "bid": np.array(self.base.boundary_index, copy=True),
            "offset": np.array(self.base.cell_face_offset, copy=True),
            "faces": np.array(self.base.cell_faces, copy=True),
            "signs": np.array(self.base.cell_face_sign, copy=True),
        }
        try:
            advection = ResidentSuspendedAdvection(
                geometry, self.base.fraction_count)
        except Exception as error:
            raise MultifractionSuspendedV78Error(
                "v78 CUDA multi-fraction FCT construction failed") from error
        self.boundary = _ResidentMultifractionBoundaryDiffusionV78(
            self.base, advection)
        self.exchange = _ResidentMultifractionExchangeV78(self.base)
        properties = cp.cuda.runtime.getDeviceProperties(self.device_id)
        raw_name = properties.get("name", b"unknown")
        device_name = (raw_name.decode("utf-8", errors="replace")
                       if isinstance(raw_name, bytes) else str(raw_name))
        source_hashes = {
            str(path.relative_to(ROOT)).replace("\\", "/"):
                hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (ADVECTION_SOURCE, BOUNDARY_SOURCE, EXCHANGE_SOURCE)
        }
        self.runtime_receipt = {
            "cuda_composition_id": CUDA_COMPOSITION_ID,
            "execution_fingerprint": execution.fingerprint,
            "hydraulic_execution_sha256":
                execution.hydraulic_execution_sha256,
            "integrator": "implicit_fwave_midpoint",
            "backend": "actual_cuda",
            "fixed_grid": True,
            "fraction_count": self.base.fraction_count,
            "automatic_cpu_fallback": False,
            "actual_cuda_kernels": True,
            "device_id": self.device_id,
            "device_name": device_name,
            "threads_per_block": CUDA_THREADS_PER_BLOCK,
            "source_sha256": source_hashes,
            "host_readback_scope":
                "candidate authentication and checkpoint evidence only",
        }
        self._open_parent = None
        self._open_trial = None

    def _device(self, value, shape, dtype, name):
        cp = self.cp
        _require(isinstance(value, cp.ndarray) and value.shape == shape and
                 value.dtype == dtype and value.flags.c_contiguous and
                 int(value.device.id) == self.device_id,
                 f"v78 CUDA {name} must be contiguous resident {dtype} {shape}")
        return value

    def begin_trial(
        self, parent: MultifractionSuspendedStateV78, *,
        accepted_step_receipt,
        depth_before_m, depth_after_m, accepted_water_flux_m2_s,
        reconstructed_left_concentration_kg_m3,
        reconstructed_right_concentration_kg_m3,
        prescribed_boundary_concentration_kg_m3,
        accepted_face_exchange_area_m2, transport_face_active,
        endpoint_momentum_m2_s, shear_velocity_m_s,
        suspended_transport_capacity_kg_m_s, surface_fraction,
        hydro_constant, dt_s: float, cancelled: bool = False,
    ) -> MultifractionSuspendedTrialV78:
        cp = self.cp
        validate_multifraction_suspended_state_v78(parent, self.execution)
        _require(self._open_trial is None,
                 "v78 CUDA previous trial is unfinished")
        if cancelled:
            raise MultifractionSuspendedV78Error(
                "v78 CUDA update cancelled before candidate publication")
        if parent.accepted_updates >= MAX_ACCEPTED_UPDATES:
            raise MultifractionSuspendedV78Error(
                "v78 CUDA accepted update limit was reached")
        _require(not isinstance(dt_s, bool) and
                 isinstance(dt_s, (int, float)) and
                 math.isfinite(float(dt_s)) and float(dt_s) > 0.0,
                 "v78 CUDA timestep must be finite and positive")
        dt = float(dt_s)
        receipt = validate_implicit_fwave_midpoint_step_receipt_v78(
            accepted_step_receipt,
            hydraulic_execution_sha256=
                self.execution.hydraulic_execution_sha256,
            parent_accepted_updates=parent.accepted_updates,
            parent_time_s=parent.time_s, dt_s=dt)
        ng, nc, nf, nb = (self.base.fraction_count, self.base.cell_count,
                          self.base.face_count, self.base.boundary_count)
        h0 = self._device(depth_before_m, (nc,), cp.float64, "depth before")
        h1 = self._device(depth_after_m, (nc,), cp.float64, "depth after")
        water = self._device(
            accepted_water_flux_m2_s, (nf,), cp.float64,
            "accepted water flux")
        left = self._device(
            reconstructed_left_concentration_kg_m3, (ng, nf), cp.float64,
            "left reconstructed concentration")
        right = self._device(
            reconstructed_right_concentration_kg_m3, (ng, nf), cp.float64,
            "right reconstructed concentration")
        prescribed = self._device(
            prescribed_boundary_concentration_kg_m3, (ng, nb), cp.float64,
            "prescribed boundary concentration")
        face_area = self._device(
            accepted_face_exchange_area_m2, (nf,), cp.float64,
            "accepted face exchange area")
        active_face = self._device(
            transport_face_active, (nf,), cp.int32,
            "transport face activity")
        momentum = self._device(
            endpoint_momentum_m2_s, (2, nc), cp.float64,
            "endpoint momentum")
        shear = self._device(
            shear_velocity_m_s, (nc,), cp.float64, "shear velocity")
        capacity = self._device(
            suspended_transport_capacity_kg_m_s, (ng, nc), cp.float64,
            "suspended transport capacity")
        fraction = self._device(
            surface_fraction, (ng, nc), cp.float64, "surface fraction")
        constant = self._device(
            hydro_constant, (nc,), cp.int32, "hydro constant")
        for value, name in (
            (h0, "depth before"), (h1, "depth after"),
            (water, "accepted water flux"), (left, "left reconstruction"),
            (right, "right reconstruction"),
            (prescribed, "prescribed boundary concentration"),
            (face_area, "accepted face exchange area"),
            (momentum, "endpoint momentum"), (shear, "shear velocity"),
            (capacity, "transport capacity"),
            (fraction, "surface fraction"),
        ):
            _require(bool(cp.all(cp.isfinite(value)).item()),
                     "v78 CUDA input must be finite: " + name)
        _require(bool(cp.all((h0 >= 0.0) & (h1 >= 0.0)).item()),
                 "v78 CUDA depths must be nonnegative")
        _require(bool(cp.all(prescribed >= 0.0).item()) and
                 bool(cp.all(face_area >= 0.0).item()) and
                 bool(cp.all(shear >= 0.0).item()) and
                 bool(cp.all(capacity >= 0.0).item()),
                 "v78 CUDA physical inputs must be nonnegative")
        _require(bool(cp.all((fraction >= 0.0) & (fraction <= 1.0)).item()),
                 "v78 CUDA surface fractions must lie in [0,1]")
        _require(bool(cp.all((active_face == 0) | (active_face == 1)).item()) and
                 bool(cp.all((constant == 0) | (constant == 1)).item()),
                 "v78 CUDA activity masks must contain only 0/1")
        closed = ((self.base.neighbour < 0) &
                  (self.base.boundary_index < 0))
        if np.any(closed):
            _require(not bool(cp.any(
                active_face[cp.asarray(closed)] != 0).item()),
                "v78 CUDA closed face cannot be diffusion-active")

        self.boundary.restore_checkpoint(parent.lifecycle)
        accepted_suspended = cp.asarray(parent.suspended_mass_kg_m2)
        accepted_bed = cp.asarray(parent.bed_mass_kg_m2)
        boundary = self.boundary.resolve_boundary(
            h0, accepted_suspended, prescribed, water, dt)
        advected = self.boundary.advect(
            h0, accepted_suspended, h1, water, left, right, constant, dt)
        if any(bool(cp.any(value != 0).item()) for value in
               (*boundary["status"], *advected["status"])):
            raise MultifractionSuspendedV78Error(
                "v78 CUDA boundary/FCT kernel status rejected the trial")
        expected = boundary["expected_advective_ledger"]
        actual = advected["ledger"]
        scale = cp.maximum(1.0, cp.maximum(cp.abs(expected), cp.abs(actual)))
        if bool(cp.any(cp.abs(expected-actual) >
                       8192.0*np.finfo(float).eps*scale).item()):
            raise MultifractionSuspendedV78Error(
                "v78 CUDA FCT ledger differs from boundary resolution")
        diffused = self.boundary.diffuse(
            h1, advected["next_mass"], face_area, active_face, dt)
        if any(bool(cp.any(value != 0).item())
               for value in diffused["status"]):
            raise MultifractionSuspendedV78Error(
                "v78 CUDA diffusion kernel status rejected the trial")

        wet = (h1 > self.base.dry_tolerance_m).astype(cp.int32)
        speed = cp.where(
            wet != 0, cp.hypot(momentum[0], momentum[1])/
            cp.where(wet != 0, h1, 1.0), 0.0)
        sediment_active = ((wet != 0) &
                           (h1 > self.base.profile.sediment_threshold_m))
        safe_h1 = cp.where(
            sediment_active,
            cp.maximum(self.base.profile.reference_depth_floor_m, h1), 1.0)
        equilibrium_capacity = (
            capacity/(speed[None, :]+1.0e-6)/safe_h1[None, :])
        total_shear = self.base.profile.water_density_kg_m3*shear*shear
        exchanged = self.exchange.step(
            h0, h1, momentum, diffused["next_mass"], accepted_bed,
            cp.ascontiguousarray(equilibrium_capacity), fraction,
            cp.ascontiguousarray(total_shear), cp.ascontiguousarray(wet), dt)
        if bool(cp.any(exchanged["status"] != 0).item()):
            raise MultifractionSuspendedV78Error(
                "v78 CUDA multi-fraction exchange kernel rejected the trial")

        cleanup = cp.where(wet[None, :] != 0, 0.0,
                           exchanged["suspended"])
        final_suspended = exchanged["suspended"]-cleanup
        final_bed = exchanged["active"]+cleanup
        area = self.boundary.area[None, :]
        exchange_water = cp.sum(
            (exchanged["suspended"]-diffused["next_mass"])*area, axis=1)
        exchange_bed = cp.sum(
            (exchanged["active"]-accepted_bed)*area, axis=1)
        cleanup_water = -cp.sum(cleanup*area, axis=1)
        cleanup_bed = -cleanup_water
        base_candidate = resolve_suspended_lifecycle_state_v76(
            self.base,
            suspended_mass_kg_m2=cp.asnumpy(final_suspended),
            bed_mass_kg_m2=cp.asnumpy(final_bed),
            boundary_last_outflow_concentration_kg_m3=
                cp.asnumpy(self.boundary.trial_last),
            boundary_return_clock_s=cp.asnumpy(self.boundary.trial_clock),
            advective_boundary_ledger_kg=(
                parent.lifecycle.advective_boundary_ledger_kg+
                cp.asnumpy(actual)),
            diffusive_boundary_ledger_kg=(
                parent.lifecycle.diffusive_boundary_ledger_kg+
                cp.asnumpy(diffused["ledger"])),
            water_exchange_ledger_kg=(
                parent.lifecycle.water_exchange_ledger_kg+
                cp.asnumpy(exchange_water)),
            bed_exchange_ledger_kg=(
                parent.lifecycle.bed_exchange_ledger_kg+
                cp.asnumpy(exchange_bed)),
            wetdry_water_ledger_kg=(
                parent.lifecycle.wetdry_water_ledger_kg+
                cp.asnumpy(cleanup_water)),
            wetdry_bed_ledger_kg=(
                parent.lifecycle.wetdry_bed_ledger_kg+
                cp.asnumpy(cleanup_bed)),
            reference_water_mass_kg=
                parent.lifecycle.reference_water_mass_kg,
            reference_bed_mass_kg=parent.lifecycle.reference_bed_mass_kg,
            accepted_updates=parent.accepted_updates+1,
            time_s=parent.time_s+dt,
        )
        trial = build_multifraction_trial_v78(
            self.execution, parent, receipt, base_candidate,
            boundary_concentration_kg_m3=
                cp.asnumpy(boundary["concentration"]),
            advective_face_flux_kg_m_s=cp.asnumpy(advected["flux"]),
            diffusive_face_mass_owner_out_kg=
                cp.asnumpy(diffused["face_mass_owner_out"]),
            settling_velocity_m_s=
                cp.asnumpy(exchanged["settling_velocity"]),
            equilibrium_concentration_kg_m3=
                cp.asnumpy(exchanged["equilibrium_concentration"]),
            transferred_to_suspension_kg_m2=cp.asnumpy(exchanged["delta"]),
            wetdry_transfer_to_bed_kg_m2=cp.asnumpy(cleanup),
            observed_parallel_workers=CUDA_THREADS_PER_BLOCK,
            backend=ROLE_CUDA)
        self._open_parent = parent.state_sha256
        self._open_trial = trial.trial_sha256
        return trial

    def _finish(self, parent, trial, *, commit):
        _require(self._open_parent == parent.state_sha256 and
                 self._open_trial == trial.trial_sha256,
                 "v78 CUDA trial is not the resident open transaction")
        result = (commit_multifraction_trial_v78(
            self.execution, parent, trial) if commit else
            rollback_multifraction_trial_v78(
                self.execution, parent, trial))
        if commit:
            self.boundary.commit_boundary_memory()
        self._open_parent = None
        self._open_trial = None
        return result

    def commit(self, parent, trial) -> MultifractionSuspendedStateV78:
        return self._finish(parent, trial, commit=True)

    def rollback(self, parent, trial) -> MultifractionSuspendedStateV78:
        return self._finish(parent, trial, commit=False)

    def cancel(self, parent, trial) -> MultifractionSuspendedStateV78:
        return self.rollback(parent, trial)

    def checkpoint(self, state: MultifractionSuspendedStateV78) -> dict:
        _require(self._open_trial is None,
                 "v78 CUDA checkpoint requires a committed point")
        return multifraction_checkpoint_v78(self.execution, state)

    def restore_checkpoint(self, document) -> MultifractionSuspendedStateV78:
        _require(self._open_trial is None,
                 "v78 CUDA restart cannot replace an open trial")
        state = load_multifraction_checkpoint_v78(self.execution, document)
        self.boundary.restore_checkpoint(state.lifecycle)
        return state


def method_identity_v78_cuda() -> dict:
    return {
        "cuda_composition_id": CUDA_COMPOSITION_ID,
        "integrator": "implicit_fwave_midpoint",
        "threads_per_block": CUDA_THREADS_PER_BLOCK,
        "automatic_cpu_fallback": False,
        "actual_cuda_kernels": True,
        "fraction_scope": "2..3 authenticated noncohesive classes",
        "reused_frozen_kernels": [
            "validation/gpu_fwave/suspended_advection.cu",
            "braids_gpu/flow2dh_boundary_diffusion.cu",
            "braids_gpu/law11_soursin_exchange.cu",
        ],
        "host_readback_scope":
            "candidate authentication and checkpoint evidence only",
    }


__all__ = [
    "CUDA_COMPOSITION_ID", "CUDA_THREADS_PER_BLOCK",
    "ResidentMultifractionSuspendedLifecycleV78",
    "method_identity_v78_cuda",
]
