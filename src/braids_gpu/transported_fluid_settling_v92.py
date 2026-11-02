"""Actual-CUDA Q4/v92 transported fluid scalars and dynamic settling.

The selected GPU path transports two conserved cell quantities: ``hT`` and
``hS``.  The accepted nonnegative FCT is reused without changing its limiter
by transporting ``h*(T-T_floor)`` for temperature, where ``T_floor=-20 C``.
Cellwise accepted-water continuity is authenticated on the device before the
inverse affine map is applied.  The resulting signed ``hT`` equation is
therefore algebraically identical to direct transport, including its boundary
ledger.  Salinity is transported directly.

This module contains the low-level CUDA components.  The selected atomic v92
transaction lives in :mod:`braids_gpu.transported_fluid_lifecycle_v92`, where
sediment fractions and only the active T/S lanes share one FCT/diffusion
owner.  Neither layer contains a CPU numerical fallback.  Host readback is
limited to transaction authentication and publication of the candidate
accepted state.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from braids_gpu.flow2dh_boundary_diffusion import (
    KERNELS as BOUNDARY_KERNELS,
    OPTIONS as CUDA_OPTIONS,
)
from braids_support.gpu_fwave.suspended_advection import (
    ResidentSuspendedAdvection,
)
from braids_gpu.multifraction_suspended_lifecycle_v78 import (
    _ResidentMultifractionExchangeV78,
)


ROOT = Path(__file__).resolve().parents[1]
CUDA_SOURCE = ROOT/"braids_gpu/transported_fluid_settling_v92.cu"
BOUNDARY_SOURCE = ROOT/"braids_gpu/flow2dh_boundary_diffusion.cu"
ADVECTION_SOURCE = ROOT/"braids_support/gpu_fwave/suspended_advection.cu"
CUDA_THREADS_PER_BLOCK = 128
CUDA_COMPOSITION_ID = (
    "resident_affine_signed_hT_hS_fct_flow2dh_diffusion_parent_"
    "settling_v92_v1"
)
TEMPERATURE_FLOOR_C = -20.0
SCALAR_COUNT = 2
V92_KERNELS = (
    "v92_pack_affine_fluid", "v92_shift_fluid_values",
    "v92_unshift_boundary_memory", "v92_water_continuity",
    "v92_unpack_affine_fluid", "v92_fluid_properties",
    "v92_boundary_water_ledger", "v92_parent_settling",
    "v92_parent_settling_diameter",
    "v92_parent_timed_dynamic_exchange",
)


def _error_type():
    # Keep import-time package traversal independent while the paired core
    # module is being built; at runtime every error has the public v92 type.
    try:
        from braids_core.sediment.transported_fluid_settling_v92 import (
            TransportedFluidSettlingV92Error,
        )
        return TransportedFluidSettlingV92Error
    except ImportError:  # pragma: no cover - only during partial installation
        return ValueError


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise _error_type()(message)


def _topology_execution(execution):
    """Return the v76 topology below a v92->v91->v78 execution."""
    try:
        return execution.base_execution.base_execution.base_execution
    except AttributeError as error:
        raise _error_type()("v92 CUDA execution topology differs") from error


class ResidentFluidScalarTransportV92:
    """Resident two-scalar FCT/diffusion/boundary transaction.

    The first lane is affine temperature and the second is salinity.  Classic
    FLOW Thatcher--Harleman memory is maintained per face and per scalar; the
    public state stores actual T/S while this resident owner stores shifted
    theta/S.  The shift is private to the CUDA transaction: restart and
    Thatcher--Harleman memory are always published in physical T/S units.
    NeuBcSand and NeuBcMud are deliberately absent because they do not apply
    to temperature or salinity in locked FLOW.
    """

    def __init__(self, execution):
        try:
            import cupy as cp
        except Exception as error:  # pragma: no cover - environment dependent
            raise _error_type()(
                "v92 actual CUDA requires CuPy and a working CUDA device") \
                from error
        try:
            _require(cp.cuda.runtime.getDeviceCount() >= 1,
                     "v92 actual CUDA requires a CUDA device")
        except Exception as error:
            if isinstance(error, _error_type()):
                raise
            raise _error_type()("v92 CUDA device discovery failed") from error
        self.cp = cp
        self.execution = execution
        self.base = _topology_execution(execution)
        self.nc = int(self.base.cell_count)
        self.nf = int(self.base.face_count)
        self.nb = int(self.base.boundary_count)
        self.ng = SCALAR_COUNT
        self.device_id = int(cp.cuda.Device().id)
        backend = getattr(execution, "backend", None)
        _require(backend in ("implicit_fwave_midpoint_gpu", "actual_cuda"),
                 "v92 CUDA owner requires the explicitly selected GPU route")
        _require(getattr(execution, "integrator", "implicit_fwave_midpoint") ==
                 "implicit_fwave_midpoint",
                 "v92 CUDA owner requires implicit_fwave_midpoint")

        geometry = {
            "scheme": np.int32(6), "amr_enabled": np.int32(0),
            "nb": np.int32(self.nb),
            "dry_tolerance": np.float64(self.base.dry_tolerance_m),
            "active": np.ones(self.nc, dtype=np.int32),
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
            self.advection = ResidentSuspendedAdvection(
                geometry, SCALAR_COUNT)
            boundary_module = cp.RawModule(
                code=BOUNDARY_SOURCE.read_text(encoding="utf-8"),
                options=CUDA_OPTIONS, name_expressions=BOUNDARY_KERNELS)
            self.boundary_kernels = {
                name: boundary_module.get_function(name)
                for name in BOUNDARY_KERNELS
            }
            v92_module = cp.RawModule(
                code=CUDA_SOURCE.read_text(encoding="utf-8"),
                options=CUDA_OPTIONS, name_expressions=V92_KERNELS)
            self.kernels = {
                name: v92_module.get_function(name) for name in V92_KERNELS
            }
        except Exception as error:
            raise _error_type()(
                "v92 CUDA scalar/settling kernels failed to compile") from error

        a = self.advection.geometry
        self.owner, self.neighbour, self.bid = (
            a["owner"], a["neighbour"], a["bid"])
        self.offset, self.faces, self.signs = (
            a["offset"], a["faces"], a["signs"])
        self.length, self.area = a["length"], a["area"]
        self.normal, self.rectangle, self.weno = (
            a["normal"], a["rectangle_width"], a["weno_cells"])
        centre_distance = np.ascontiguousarray(
            self.base.centre_distance_m, dtype=np.float64)
        _require(centre_distance.shape == (self.nf,) and
                 np.all(np.isfinite(centre_distance)) and
                 np.all(centre_distance > 0.0),
                 "v92 CUDA centre distances differ")
        self.centre_distance = cp.asarray(centre_distance)
        bids = np.asarray(self.base.boundary_index)
        lists = [np.flatnonzero(bids == boundary).astype(np.int32)
                 for boundary in range(self.nb)]
        boundary_faces = (np.concatenate(lists) if lists else
                          np.empty(0, dtype=np.int32))
        boundary_offset = np.r_[
            0, np.cumsum([len(value) for value in lists])].astype(np.int32)
        self.boundary_faces = cp.asarray(boundary_faces)
        self.boundary_offset = cp.asarray(boundary_offset)

        return_by_id = np.asarray(
            getattr(execution, "fluid_boundary_return_time_s",
                    np.zeros((SCALAR_COUNT, self.nb))))
        _require(return_by_id.shape == (SCALAR_COUNT, self.nb) and
                 return_by_id.dtype.kind in "fiu" and
                 np.isfinite(return_by_id).all(),
                 "v92 CUDA fluid boundary return times differ")
        expanded = np.zeros((SCALAR_COUNT, self.nf), dtype=np.float64)
        open_face = bids >= 0
        if self.nb:
            expanded[:, open_face] = return_by_id[:, bids[open_face]]
        self.return_time_host = expanded
        self.return_time = cp.asarray(expanded)
        self.neumann = cp.zeros(SCALAR_COUNT, dtype=cp.int32)

        shape_face = (SCALAR_COUNT, self.nf)
        shape_cell = (SCALAR_COUNT, self.nc)
        shape_boundary = (SCALAR_COUNT, self.nb)
        self.accepted_last = cp.zeros(shape_face, dtype=cp.float64)
        self.accepted_clock = cp.zeros(shape_face, dtype=cp.float64)
        self.trial_last = cp.zeros(shape_face, dtype=cp.float64)
        self.trial_clock = cp.zeros(shape_face, dtype=cp.float64)
        self.face_boundary = cp.zeros(shape_face, dtype=cp.float64)
        self.expected_face_mass_in = cp.zeros(shape_face, dtype=cp.float64)
        self.expected_ledger = cp.zeros(shape_boundary, dtype=cp.float64)
        self.boundary_status = cp.zeros(shape_face, dtype=cp.int32)
        self.boundary_ledger_status = cp.zeros(
            shape_boundary, dtype=cp.int32)
        self.shifted_mass = cp.zeros(shape_cell, dtype=cp.float64)
        self.parent_actual = cp.zeros(shape_cell, dtype=cp.float64)
        self.pack_status = cp.zeros(self.nc, dtype=cp.int32)
        self.left_shifted = cp.zeros(shape_face, dtype=cp.float64)
        self.right_shifted = cp.zeros(shape_face, dtype=cp.float64)
        self.prescribed_shifted = cp.zeros(shape_boundary, dtype=cp.float64)
        self.left_status = cp.zeros(shape_face, dtype=cp.int32)
        self.right_status = cp.zeros(shape_face, dtype=cp.int32)
        self.prescribed_status = cp.zeros(shape_boundary, dtype=cp.int32)
        self.actual_trial_last = cp.zeros(shape_face, dtype=cp.float64)
        self.memory_status = cp.zeros(shape_face, dtype=cp.int32)
        self.water_residual = cp.zeros(self.nc, dtype=cp.float64)
        self.water_tolerance = cp.zeros(self.nc, dtype=cp.float64)
        self.water_status = cp.zeros(self.nc, dtype=cp.int32)
        self.effective_diffusivity = cp.zeros(shape_face, dtype=cp.float64)
        self.diffusive_face_mass = cp.zeros(shape_face, dtype=cp.float64)
        self.diffusive_face_active = cp.zeros(shape_face, dtype=cp.int32)
        self.diffusive_face_status = cp.zeros(shape_face, dtype=cp.int32)
        self.diffusive_theta = cp.ones(shape_cell, dtype=cp.float64)
        self.diffusive_theta_status = cp.zeros(shape_cell, dtype=cp.int32)
        self.diffused_mass = cp.zeros(shape_cell, dtype=cp.float64)
        self.diffusive_cell_status = cp.zeros(shape_cell, dtype=cp.int32)
        self.diffusive_ledger = cp.zeros(shape_boundary, dtype=cp.float64)
        self.diffusive_ledger_status = cp.zeros(
            shape_boundary, dtype=cp.int32)
        self.h_temperature = cp.zeros(self.nc, dtype=cp.float64)
        self.h_salinity = cp.zeros(self.nc, dtype=cp.float64)
        self.temperature = cp.zeros(self.nc, dtype=cp.float64)
        self.salinity = cp.zeros(self.nc, dtype=cp.float64)
        self.wetdry_removed = cp.zeros(shape_cell, dtype=cp.float64)
        self.unpack_status = cp.zeros(self.nc, dtype=cp.int32)
        self.water_ledger = cp.zeros(self.nb, dtype=cp.float64)
        self.water_ledger_status = cp.zeros(self.nb, dtype=cp.int32)
        self.actual_advective_flux = cp.zeros(shape_face, dtype=cp.float64)
        self.actual_advective_ledger = cp.zeros(
            shape_boundary, dtype=cp.float64)
        self.endpoint_density = cp.zeros(self.nc, dtype=cp.float64)
        self.endpoint_viscosity = cp.zeros(self.nc, dtype=cp.float64)
        self.endpoint_property_status = cp.zeros(self.nc, dtype=cp.int32)
        self._open = False
        self.kernel_launch_counts = {
            "pack": 0, "shift": 0, "water_continuity": 0,
            "boundary": 0, "fct": 0, "diffusion": 0,
            "unpack": 0, "water_ledger": 0,
            "properties": 0,
        }
        props = cp.cuda.runtime.getDeviceProperties(self.device_id)
        name = props.get("name", b"unknown")
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="replace")
        paths = (CUDA_SOURCE, BOUNDARY_SOURCE, ADVECTION_SOURCE,
                 Path(__file__).resolve())
        self.runtime_receipt = {
            "cuda_composition_id": CUDA_COMPOSITION_ID,
            "backend": "actual_cuda",
            "integrator": "implicit_fwave_midpoint",
            "device_id": self.device_id,
            "device_name": str(name),
            "actual_cuda_transport": True,
            "temperature_variable": "h*(T-(-20C))_affine",
            "published_temperature_variable": "signed_hT",
            "cellwise_water_continuity_authenticated": True,
            "salinity_variable": "hS",
            "horizontal_schmidt": 0.7,
            "fluid_boundary_memory": "physical_temperature_and_salinity_per_face",
            "constant_lane_execution": (
                "shared_two_lane_cuda_operator_then_exact_constant_lane_"
                "discard; no_public_memory_or_ledger"),
            "sediment_neumann_flags_applied": False,
            "automatic_cpu_fallback": False,
            "source_sha256": {
                str(path.relative_to(ROOT)).replace("\\", "/"):
                    hashlib.sha256(path.read_bytes()).hexdigest()
                for path in paths
            },
        }

    def _launch(self, table, name, count, args):
        if count:
            table[name](((count+CUDA_THREADS_PER_BLOCK-1)//
                         CUDA_THREADS_PER_BLOCK,),
                        (CUDA_THREADS_PER_BLOCK,), args)

    def _device(self, value, shape, dtype, name):
        cp = self.cp
        _require(isinstance(value, cp.ndarray) and value.shape == shape and
                 value.dtype == dtype and value.flags.c_contiguous and
                 int(value.device.id) == self.device_id,
                 f"v92 CUDA {name} must be contiguous resident {dtype} {shape}")
        return value

    def restore_boundary_memory(self, last_fluid, clock) -> None:
        _require(not self._open,
                 "v92 CUDA cannot restore fluid memory during a trial")
        last = np.asarray(last_fluid)
        clocks = np.asarray(clock)
        shape = (SCALAR_COUNT, self.nf)
        _require(last.shape == clocks.shape == shape and
                 last.dtype.kind in "fiu" and clocks.dtype.kind in "fiu" and
                 np.isfinite(last).all() and np.isfinite(clocks).all() and
                 np.all(last[0] > TEMPERATURE_FLOOR_C) and
                 np.all(last[1] >= 0.0) and np.all(clocks >= 0.0),
                 "v92 CUDA fluid boundary checkpoint memory differs")
        bids = np.asarray(self.base.boundary_index)
        closed = bids < 0
        active = np.asarray(getattr(
            self.execution, "transported_mask", (True, True)), dtype=bool)
        _require(np.all(last[:, closed] == 0.0) and
                 np.all(clocks[:, closed] == 0.0) and
                 np.all(last[~active] == 0.0) and
                 np.all(clocks[~active] == 0.0),
                 "v92 CUDA fluid boundary memory exists outside its mode")
        tolerance = (256.0*np.finfo(np.float64).eps*
                     np.maximum(1.0, self.return_time_host))
        positive = self.return_time_host > 0.0
        _require(np.all(clocks[positive] <=
                        self.return_time_host[positive]+tolerance[positive]) and
                 np.all(clocks[~positive] == 0.0),
                 "v92 CUDA fluid boundary clocks differ")
        # Public/checkpoint memory is physical.  Only the resident operator
        # sees theta=T-T_floor.  Non-open and equilibrium lanes stay exactly
        # zero in both representations and therefore own no hidden memory.
        shifted = np.ascontiguousarray(last, dtype=np.float64)
        shifted = np.array(shifted, copy=True)
        shifted[0, ~closed] -= TEMPERATURE_FLOOR_C
        shifted[:, closed] = 0.0
        shifted[~active] = 0.0
        self.cp.copyto(self.accepted_last, self.cp.asarray(shifted))
        self.cp.copyto(self.accepted_clock,
                       self.cp.asarray(clocks, dtype=self.cp.float64))
        self.cp.copyto(self.trial_last, self.accepted_last)
        self.cp.copyto(self.trial_clock, self.accepted_clock)

    def begin_trial(
        self, *, depth_before_m, depth_after_m,
        h_temperature_c_m, h_salinity_ppt_m,
        accepted_water_flux_m2_s,
        reconstructed_left_fluid,
        reconstructed_right_fluid,
        prescribed_boundary_fluid,
        accepted_face_exchange_area_m2, transport_face_active,
        hydro_constant, dicuv_cell_m2_s, dt_s: float,
        equilibrium_temperature_c: float,
        equilibrium_salinity_ppt: float,
        boundary_last_outflow_fluid, boundary_return_clock_s,
        cancelled: bool = False,
    ) -> dict:
        """Begin one resident scalar trial and return host evidence arrays."""
        cp = self.cp
        _require(not self._open,
                 "v92 CUDA previous fluid scalar trial is unfinished")
        _require(not cancelled,
                 "v92 CUDA fluid trial cancelled before any launch")
        _require(not isinstance(dt_s, bool) and isinstance(dt_s, (int, float))
                 and np.isfinite(dt_s) and dt_s > 0.0,
                 "v92 CUDA timestep must be finite and positive")
        dt = float(dt_s)
        h0 = self._device(depth_before_m, (self.nc,), cp.float64,
                          "depth before")
        h1 = self._device(depth_after_m, (self.nc,), cp.float64,
                          "depth after")
        ht = self._device(h_temperature_c_m, (self.nc,), cp.float64,
                          "parent hT")
        hs = self._device(h_salinity_ppt_m, (self.nc,), cp.float64,
                          "parent hS")
        water = self._device(accepted_water_flux_m2_s, (self.nf,), cp.float64,
                             "accepted water flux")
        left = self._device(reconstructed_left_fluid,
                            (SCALAR_COUNT, self.nf), cp.float64,
                            "left fluid reconstruction")
        right = self._device(reconstructed_right_fluid,
                             (SCALAR_COUNT, self.nf), cp.float64,
                             "right fluid reconstruction")
        prescribed = self._device(prescribed_boundary_fluid,
                                  (SCALAR_COUNT, self.nb), cp.float64,
                                  "prescribed boundary fluid")
        face_area = self._device(accepted_face_exchange_area_m2,
                                 (self.nf,), cp.float64,
                                 "accepted face exchange area")
        active_face = self._device(transport_face_active, (self.nf,), cp.int32,
                                   "transport face activity")
        constant = self._device(hydro_constant, (self.nc,), cp.int32,
                                "hydro constant")
        dicuv = self._device(dicuv_cell_m2_s, (self.nc,), cp.float64,
                             "capped dicuv")
        for value, name in ((h0, "depth before"), (h1, "depth after"),
                            (ht, "hT"), (hs, "hS"),
                            (water, "water flux"), (left, "left fluid"),
                            (right, "right fluid"),
                            (prescribed, "boundary fluid"),
                            (face_area, "face area"), (dicuv, "dicuv")):
            _require(bool(cp.all(cp.isfinite(value)).item()),
                     "v92 CUDA input must be finite: "+name)
        _require(bool(cp.all((h0 >= 0.0) & (h1 >= 0.0)).item()) and
                 bool(cp.all(face_area >= 0.0).item()) and
                 bool(cp.all(dicuv >= 0.0).item()),
                 "v92 CUDA nonnegative hydraulic/diffusion inputs differ")
        _require(bool(cp.all((active_face == 0) | (active_face == 1)).item())
                 and bool(cp.all((constant == 0) | (constant == 1)).item()),
                 "v92 CUDA masks must contain only zero or one")
        closed = ((np.asarray(self.base.neighbour) < 0) &
                  (np.asarray(self.base.boundary_index) < 0))
        if np.any(closed):
            _require(not bool(cp.any(
                active_face[cp.asarray(closed)] != 0).item()),
                "v92 CUDA closed face cannot be diffusion-active")
        self.restore_boundary_memory(
            boundary_last_outflow_fluid,
            boundary_return_clock_s)

        try:
            self.kernel_launch_counts["pack"] += 1
            self._launch(self.kernels, "v92_pack_affine_fluid", self.nc, (
                np.int32(self.nc), np.float64(self.base.dry_tolerance_m),
                np.float64(TEMPERATURE_FLOOR_C), h0, ht, hs,
                self.shifted_mass, self.parent_actual, self.pack_status))
            self.kernel_launch_counts["shift"] += 3
            for actual, shifted, status, count in (
                (left, self.left_shifted, self.left_status, self.nf),
                (right, self.right_shifted, self.right_status, self.nf),
                (prescribed, self.prescribed_shifted,
                 self.prescribed_status, self.nb),
            ):
                self._launch(self.kernels, "v92_shift_fluid_values",
                             SCALAR_COUNT*count, (
                    np.int32(count), np.float64(TEMPERATURE_FLOOR_C),
                    actual, shifted, status))
            _require(not bool(cp.any(self.pack_status != 0).item()) and
                     not bool(cp.any(self.left_status != 0).item()) and
                     not bool(cp.any(self.right_status != 0).item()) and
                     not bool(cp.any(self.prescribed_status != 0).item()),
                     "v92 CUDA affine field preparation rejected the trial")

            self.kernel_launch_counts["water_continuity"] += 1
            transported = tuple(bool(value) for value in getattr(
                self.execution, "transported_mask", (True, True)))
            if transported[0]:
                self._launch(self.kernels, "v92_water_continuity", self.nc, (
                    np.int32(self.nc), np.float64(dt), h0, h1, water,
                    self.length, self.area, self.offset, self.faces, self.signs,
                    self.water_residual, self.water_tolerance,
                    self.water_status))
                _require(not bool(cp.any(self.water_status != 0).item()),
                         "v92 CUDA affine hT requires accepted cellwise water "
                         "continuity")
            else:
                self.water_residual.fill(0.0)
                self.water_tolerance.fill(0.0)
                self.water_status.fill(0)

            self.kernel_launch_counts["boundary"] += 2
            self._launch(self.boundary_kernels, "m2b_resolve_boundary",
                         SCALAR_COUNT*self.nf, (
                np.int32(self.nc), np.int32(self.nf),
                np.int32(SCALAR_COUNT), np.int32(self.nb),
                np.float64(self.base.dry_tolerance_m), np.float64(dt), h0,
                self.shifted_mass, self.prescribed_shifted, water,
                self.length, self.owner, self.neighbour, self.bid,
                self.neumann, self.return_time, self.accepted_last,
                self.accepted_clock, self.face_boundary,
                self.expected_face_mass_in, self.trial_last,
                self.trial_clock, self.boundary_status))
            self._launch(self.boundary_kernels,
                         "m2b_expected_advective_ledger",
                         SCALAR_COUNT*self.nb, (
                np.int32(self.nf), np.int32(SCALAR_COUNT), np.int32(self.nb),
                self.boundary_offset, self.boundary_faces,
                self.expected_face_mass_in, self.boundary_status,
                self.expected_ledger, self.boundary_ledger_status))

            adv = self.advection
            a, b = adv.geometry, adv.buffers
            self.kernel_launch_counts["fct"] += 6
            adv._launch("sa_concentration", self.nc*SCALAR_COUNT, (
                np.int32(self.nc), np.int32(SCALAR_COUNT), h0,
                self.shifted_mass, np.float64(adv.dry), b["concentration"],
                b["concentration_status"]))
            self._launch(self.boundary_kernels, "m2b_sa_face_low",
                         self.nf*SCALAR_COUNT, (
                np.int32(self.nc), np.int32(self.nf),
                np.int32(SCALAR_COUNT), water, b["concentration"],
                self.left_shifted, self.right_shifted, self.face_boundary,
                a["owner"], a["neighbour"], a["bid"],
                b["concentration_status"], b["low_flux"], b["anti"],
                b["face_status"]))
            self._launch(self.boundary_kernels, "m2b_sa_cell_bounds",
                         self.nc*SCALAR_COUNT, (
                np.int32(self.nc), np.int32(self.nf),
                np.int32(SCALAR_COUNT), self.shifted_mass, h1,
                b["concentration"], self.face_boundary, water,
                b["low_flux"], b["anti"], a["area"], a["length"],
                a["normal"], a["rectangle_width"], constant,
                a["weno_cells"], a["owner"], a["neighbour"], a["bid"],
                a["offset"], a["faces"], a["signs"],
                b["concentration_status"], b["face_status"],
                np.float64(dt), b["low_state"], b["rplus"], b["rminus"],
                b["cell_status"]))
            adv._launch("sa_limit_faces", self.nf*SCALAR_COUNT, (
                np.int32(self.nc), np.int32(self.nf),
                np.int32(SCALAR_COUNT), b["low_flux"], b["anti"],
                b["rplus"], b["rminus"], a["owner"], a["neighbour"],
                b["cell_status"], b["scaled_anti"], b["flux"],
                b["flux_status"]))
            adv._launch("sa_update", self.nc*SCALAR_COUNT, (
                np.int32(self.nc), np.int32(self.nf),
                np.int32(SCALAR_COUNT), b["low_state"], b["scaled_anti"],
                h1, a["length"], a["area"], a["offset"], a["faces"],
                a["signs"], b["cell_status"], b["flux_status"],
                np.float64(dt), np.float64(adv.dry), b["next_mass"],
                b["next_concentration"], b["update_status"]))
            adv._launch("sa_boundary_ledger", self.nb*SCALAR_COUNT, (
                np.int32(self.nf), np.int32(SCALAR_COUNT), np.int32(self.nb),
                a["boundary_offset"], a["boundary_faces"], b["flux"],
                a["length"], b["flux_status"], np.float64(dt), b["ledger"],
                b["ledger_status"]))
            statuses = (self.boundary_status, self.boundary_ledger_status,
                        b["concentration_status"], b["face_status"],
                        b["cell_status"], b["flux_status"],
                        b["update_status"], b["ledger_status"])
            _require(not any(bool(cp.any(value != 0).item())
                             for value in statuses),
                     "v92 CUDA boundary/FCT kernel rejected the trial")
            scale = cp.maximum(1.0, cp.maximum(
                cp.abs(self.expected_ledger), cp.abs(b["ledger"])))
            _require(not bool(cp.any(cp.abs(
                self.expected_ledger-b["ledger"]) >
                8192.0*np.finfo(float).eps*scale).item()),
                "v92 CUDA FCT ledger differs from boundary resolution")

            self.kernel_launch_counts["diffusion"] += 5
            self._launch(self.boundary_kernels, "m2b_diffusion_faces",
                         SCALAR_COUNT*self.nf, (
                np.int32(self.nc), np.int32(self.nf),
                np.int32(SCALAR_COUNT),
                np.float64(self.base.profile.diffusion_dry_threshold_m),
                np.float64(dt), h1, b["next_mass"], self.face_boundary,
                dicuv, face_area, self.centre_distance, active_face,
                self.owner, self.neighbour, self.bid,
                self.effective_diffusivity, self.diffusive_face_mass,
                self.diffusive_face_active, self.diffusive_face_status))
            self._launch(self.boundary_kernels, "m2b_diffusion_theta",
                         SCALAR_COUNT*self.nc, (
                np.int32(self.nc), np.int32(self.nf),
                np.int32(SCALAR_COUNT), b["next_mass"],
                self.diffusive_face_mass, self.area, self.offset, self.faces,
                self.signs, self.diffusive_face_status,
                self.diffusive_theta, self.diffusive_theta_status))
            self._launch(self.boundary_kernels, "m2b_diffusion_limit",
                         SCALAR_COUNT*self.nf, (
                np.int32(self.nc), np.int32(self.nf),
                np.int32(SCALAR_COUNT), self.owner, self.neighbour,
                self.diffusive_theta, self.diffusive_theta_status,
                self.diffusive_face_mass, self.diffusive_face_status))
            self._launch(self.boundary_kernels, "m2b_diffusion_update",
                         SCALAR_COUNT*self.nc, (
                np.int32(self.nc), np.int32(self.nf),
                np.int32(SCALAR_COUNT), b["next_mass"],
                self.diffusive_face_mass, self.area, self.offset, self.faces,
                self.signs, self.diffusive_face_status, self.diffused_mass,
                self.diffusive_cell_status))
            self._launch(self.boundary_kernels, "m2b_diffusion_ledger",
                         SCALAR_COUNT*self.nb, (
                np.int32(self.nf), np.int32(SCALAR_COUNT), np.int32(self.nb),
                self.boundary_offset, self.boundary_faces,
                self.diffusive_face_mass, self.diffusive_face_status,
                self.diffusive_ledger, self.diffusive_ledger_status))
            _require(not any(bool(cp.any(value != 0).item()) for value in (
                self.diffusive_face_status, self.diffusive_theta_status,
                self.diffusive_cell_status, self.diffusive_ledger_status)),
                "v92 CUDA fluid diffusion kernel rejected the trial")

            self.kernel_launch_counts["unpack"] += 1
            self._launch(self.kernels, "v92_unpack_affine_fluid", self.nc, (
                np.int32(self.nc), np.float64(self.base.dry_tolerance_m),
                np.float64(TEMPERATURE_FLOOR_C),
                np.float64(equilibrium_temperature_c),
                np.float64(equilibrium_salinity_ppt), h1,
                self.diffused_mass, self.h_temperature, self.h_salinity,
                self.temperature, self.salinity, self.wetdry_removed,
                self.unpack_status))
            _require(not bool(cp.any(self.unpack_status != 0).item()),
                     "v92 CUDA affine inverse rejected the trial")

            self.kernel_launch_counts["water_ledger"] += 1
            self._launch(self.kernels, "v92_boundary_water_ledger", self.nb, (
                np.int32(self.nb), np.float64(dt), self.boundary_offset,
                self.boundary_faces, water, self.length, self.water_ledger,
                self.water_ledger_status))
            _require(not bool(cp.any(self.water_ledger_status != 0).item()),
                     "v92 CUDA boundary water ledger rejected the trial")
            cp.copyto(self.actual_advective_flux, b["flux"])
            self.actual_advective_flux[0] += TEMPERATURE_FLOOR_C*water
            cp.copyto(self.actual_advective_ledger, b["ledger"])
            self.actual_advective_ledger[0] += (
                TEMPERATURE_FLOOR_C*self.water_ledger)
            constants = (float(self.execution.equilibrium_temperature_c),
                         float(self.execution.equilibrium_salinity_ppt))
            shifted_constants = (constants[0]-TEMPERATURE_FLOOR_C,
                                 constants[1])
            for lane in range(SCALAR_COUNT):
                if transported[lane]:
                    continue
                if lane == 0:
                    cp.copyto(self.h_temperature, h1*constants[0])
                    self.temperature.fill(constants[0])
                else:
                    cp.copyto(self.h_salinity, h1*constants[1])
                    self.salinity.fill(constants[1])
                self.face_boundary[lane].fill(shifted_constants[lane])
                self.trial_last[lane].fill(0.0)
                self.trial_clock[lane].fill(0.0)
                self.actual_advective_flux[lane].fill(0.0)
                self.diffusive_face_mass[lane].fill(0.0)
                self.actual_advective_ledger[lane].fill(0.0)
                self.diffusive_ledger[lane].fill(0.0)
                self.wetdry_removed[lane].fill(0.0)
            wetdry_ledger = -cp.sum(
                self.wetdry_removed*self.area[None, :], axis=1)
            # Convert resident affine boundary traces/memory back to the
            # physical public contract.  Do this on device and zero locations
            # that do not own memory before the single evidence readback.
            boundary_physical = cp.array(self.face_boundary, copy=True)
            boundary_physical[0] += TEMPERATURE_FLOOR_C
            last_physical = cp.array(self.trial_last, copy=True)
            last_physical[0] += TEMPERATURE_FLOOR_C
            open_face = cp.asarray(
                np.asarray(self.base.boundary_index) >= 0)
            last_physical[:, ~open_face] = 0.0
            for lane in range(SCALAR_COUNT):
                if not transported[lane]:
                    last_physical[lane].fill(0.0)
            self._open = True
            return {
                "h_temperature_c_m": cp.asnumpy(self.h_temperature),
                "h_salinity_ppt_m": cp.asnumpy(self.h_salinity),
                "temperature_cell_c": cp.asnumpy(self.temperature),
                "salinity_cell_ppt": cp.asnumpy(self.salinity),
                "boundary_transport_scalar": cp.asnumpy(
                    boundary_physical),
                "boundary_last_outflow_fluid": cp.asnumpy(
                    last_physical),
                "boundary_return_clock_s": cp.asnumpy(self.trial_clock),
                "advective_face_flux": cp.asnumpy(
                    self.actual_advective_flux),
                "diffusive_face_mass": cp.asnumpy(
                    self.diffusive_face_mass),
                "advective_boundary_ledger": cp.asnumpy(
                    self.actual_advective_ledger),
                "diffusive_boundary_ledger": cp.asnumpy(
                    self.diffusive_ledger),
                "wetdry_scalar_ledger": cp.asnumpy(wetdry_ledger),
                "water_continuity_residual": cp.asnumpy(
                    self.water_residual),
                "water_continuity_tolerance": cp.asnumpy(
                    self.water_tolerance),
                "effective_horizontal_diffusivity_m2_s": cp.asnumpy(
                    self.effective_diffusivity),
                "resident_shifted_mass": self.diffused_mass,
            }
        except Exception:
            self._open = False
            raise

    def commit(self) -> None:
        _require(self._open, "v92 CUDA fluid trial is not open")
        self.cp.copyto(self.accepted_last, self.trial_last)
        self.cp.copyto(self.accepted_clock, self.trial_clock)
        self._open = False

    def derive_endpoint_properties(self) -> tuple[np.ndarray, np.ndarray]:
        """Derive, never transport, endpoint EOS and molecular viscosity."""
        _require(self._open,
                 "v92 CUDA endpoint properties require an open fluid trial")
        eos_code = {"eckart": 0, "unesco": 1, "nacl": 2}.get(
            getattr(self.execution, "density_eos", None))
        _require(eos_code is not None,
                 "v92 CUDA endpoint EOS identity differs")
        constituents_active = int(
            getattr(self.execution, "temperature_mode", "transported") !=
            "equilibrium_constant" or
            getattr(self.execution, "salinity_mode", "transported") !=
            "equilibrium_constant")
        reference = float(self.base.profile.water_density_kg_m3)
        self.kernel_launch_counts["properties"] += 1
        self._launch(self.kernels, "v92_fluid_properties", self.nc, (
            np.int32(self.nc), self.temperature, self.salinity,
            np.int32(eos_code), np.int32(constituents_active),
            np.float64(reference), self.endpoint_density,
            self.endpoint_viscosity, self.endpoint_property_status))
        _require(not bool(self.cp.any(
            self.endpoint_property_status != 0).item()),
            "v92 CUDA endpoint EOS/viscosity rejected the trial")
        return (self.cp.asnumpy(self.endpoint_density),
                self.cp.asnumpy(self.endpoint_viscosity))

    def rollback(self) -> None:
        _require(self._open, "v92 CUDA fluid trial is not open")
        self.cp.copyto(self.trial_last, self.accepted_last)
        self.cp.copyto(self.trial_clock, self.accepted_clock)
        self._open = False


class ResidentParentTimedDynamicExchangeV92(
        _ResidentMultifractionExchangeV78):
    """One CUDA SettleFrm=2 preparation followed by one exchange.

    ``prepare_parent`` must be called before the suspended FCT owner.  It
    computes current settling velocities from parent ``ctot``, T and rho.
    ``step`` then consumes those immutable arrays after transport.  Thus the
    output cannot accidentally change settling timing to post-FCT ctot.
    """

    def __init__(self, base_execution, execution):
        super().__init__(base_execution)
        self.execution_v92 = execution
        cp = self.cp
        try:
            module = cp.RawModule(
                code=CUDA_SOURCE.read_text(encoding="utf-8"),
                options=CUDA_OPTIONS, name_expressions=V92_KERNELS)
            self.parent_kernel = module.get_function("v92_parent_settling")
            self.parent_diameter_kernel = module.get_function(
                "v92_parent_settling_diameter")
            self.kernel = module.get_function(
                "v92_parent_timed_dynamic_exchange")
        except Exception as error:
            raise _error_type()(
                "v92 CUDA parent-timed settling kernels failed to compile") \
                from error
        self.parent_temperature = None
        self.parent_density = None
        self.parent_suspended = None
        self.parent_depth = None
        self.parent_status = cp.zeros(self.nc, dtype=cp.int32)
        self.parent_settling_diameter = None
        self._prepared = False
        self.parent_settling_launches = 0
        self.exchange_launches = 0

    def _cell_field(self, value, name):
        cp = self.cp
        _require(isinstance(value, cp.ndarray) and
                 value.shape == (self.nc,) and value.dtype == cp.float64 and
                 value.flags.c_contiguous,
                 f"v92 CUDA {name} must be contiguous device float64 "
                 f"({self.nc},)")
        return value

    def prepare_parent(self, *, depth_before, suspended_mass,
                       temperature, water_density,
                       settling_diameter_m=None) -> None:
        cp = self.cp
        _require(not self._prepared,
                 "v92 CUDA parent settling was already prepared")
        self.parent_depth = self._cell_field(depth_before, "parent depth")
        self.parent_temperature = self._cell_field(
            temperature, "parent temperature")
        self.parent_density = self._cell_field(
            water_density, "parent water density")
        self._array(suspended_mass, (self.ng, self.nc),
                    "parent suspended mass")
        self.parent_suspended = suspended_mass
        if settling_diameter_m is None:
            self.parent_settling_diameter = None
            kernel = self.parent_kernel
            diameter_arguments = (
                self.sediment_d50, self.settling_fac_dss)
        else:
            self._array(
                settling_diameter_m, (self.ng, self.nc),
                "parent settling diameter")
            _require(bool(cp.all(cp.isfinite(settling_diameter_m) &
                                (settling_diameter_m > 0.0)).item()),
                     "v92 CUDA parent settling diameter must be positive")
            self.parent_settling_diameter = settling_diameter_m
            kernel = self.parent_diameter_kernel
            diameter_arguments = (settling_diameter_m,)
        self.parent_settling_launches += 1
        kernel(
            ((self.nc+CUDA_THREADS_PER_BLOCK-1)//CUDA_THREADS_PER_BLOCK,),
            (CUDA_THREADS_PER_BLOCK,),
            (np.int32(self.nc), np.int32(self.ng), self.dry, self.gravity,
             self.reference_concentration, self.parent_depth,
             self.parent_suspended, self.parent_temperature,
             self.parent_density, *diameter_arguments, self.sediment_density,
             self.clear_water_settling_velocity, self.settling_velocity,
             self.hindered_factor, self.parent_status))
        _require(not bool(cp.any(self.parent_status != 0).item()),
                 "v92 CUDA parent-state settling rejected the trial")
        self._prepared = True

    def step(self, depth_before, depth_after, endpoint_momentum, suspended,
             active_bed, equilibrium_capacity, surface_fraction, total_shear,
             wet_mask, dt):
        cp = self.cp
        _require(self._prepared,
                 "v92 CUDA settling must be prepared before scalar transport")
        for value, shape, name in (
            (depth_before, (self.nc,), "depth before"),
            (depth_after, (self.nc,), "depth after"),
            (endpoint_momentum, (2, self.nc), "endpoint momentum"),
            (suspended, (self.ng, self.nc), "transported suspended mass"),
            (active_bed, (self.ng, self.nc), "active bed"),
            (equilibrium_capacity, (self.ng, self.nc),
             "equilibrium capacity"),
            (surface_fraction, (self.ng, self.nc), "surface fraction"),
            (total_shear, (self.nc,), "reference total shear"),
        ):
            self._array(value, shape, name)
        _require(isinstance(wet_mask, cp.ndarray) and
                 wet_mask.shape == (self.nc,) and
                 wet_mask.dtype == cp.int32 and wet_mask.flags.c_contiguous,
                 "v92 CUDA wet mask must be contiguous device int32")
        # Require the exact parent depth array used during preparation.
        _require(depth_before is self.parent_depth or
                 bool(cp.array_equal(depth_before, self.parent_depth).item()),
                 "v92 CUDA exchange parent depth differs from fallve input")
        self.exchange_launches += 1
        self.kernel(
            ((self.nc+CUDA_THREADS_PER_BLOCK-1)//CUDA_THREADS_PER_BLOCK,),
            (CUDA_THREADS_PER_BLOCK,),
            (np.int32(self.nc), np.int32(self.ng), depth_before, depth_after,
             endpoint_momentum, suspended, active_bed,
             equilibrium_capacity, surface_fraction, total_shear, wet_mask,
             self.sediment_threshold,
             self.clear_water_settling_velocity, self.settling_velocity,
             self.hindered_factor, self.rho, self.capacity_speed_epsilon,
             self.gravity, self.dry, self.factsd,
             self.reference_depth_floor, np.float64(dt),
             self.next_suspended, self.next_active,
             self.equilibrium_concentration, self.adaptation_velocity,
             self.dimensionless_timescale,
             self.clear_water_settling_velocity, self.settling_velocity,
             self.hindered_factor, self.delta, self.limiter_correction,
             self.sediment_active, self.status))
        self._prepared = False
        self.parent_settling_diameter = None
        return {
            "suspended": self.next_suspended,
            "active": self.next_active,
            "equilibrium_concentration": self.equilibrium_concentration,
            "settling_velocity": self.settling_velocity,
            "delta": self.delta,
            "status": self.status,
        }


def method_identity_v92_cuda() -> dict:
    return {
        "cuda_composition_id": CUDA_COMPOSITION_ID,
        "integrator": "implicit_fwave_midpoint",
        "actual_cuda_transport": True,
        "temperature_variable": "affine_h_times_T_minus_minus20C",
        "published_temperature_variable": "signed_hT",
        "water_continuity_required": True,
        "salinity_variable": "hS",
        "flow2dh_horizontal_schmidt": 0.7,
        "flow2dh_thatcher_harleman": True,
        "sediment_neumann_flags_applied_to_fluid": False,
        "parent_pretransport_ctot_settling": True,
        "automatic_cpu_fallback": False,
        "product_admission": False,
    }


__all__ = [
    "CUDA_COMPOSITION_ID", "TEMPERATURE_FLOOR_C",
    "ResidentFluidScalarTransportV92",
    "ResidentParentTimedDynamicExchangeV92",
    "method_identity_v92_cuda",
]
