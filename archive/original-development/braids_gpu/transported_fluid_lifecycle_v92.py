"""Atomic actual-CUDA Q4/v92 transported-fluid lifecycle.

This is the selected GPU transaction, not a component demonstration.  One
generic CUDA scalar owner advances all suspended fractions and only the active
temperature/salinity lanes in the same FCT and diffusion launches.  The result
is split once, parent-timed dynamic settling is applied once, and the combined
hydraulic/sediment/fluid candidate is then committed or rolled back atomically.

Temperature is private-affine on device (theta=T+20 C) but every public state,
checkpoint and Thatcher--Harleman memory value is in physical Celsius.  There
is no CPU numerical fallback.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np

from braids_core.sediment.multifraction_suspended_lifecycle_v78 import (
    ROLE_CUDA,
    build_multifraction_trial_v78,
    validate_implicit_fwave_midpoint_step_receipt_v78,
)
from braids_core.sediment.shared_turbulence_consumers_v91 import (
    STATE_DOMAIN_MOVING_WETDRY_V97,
    _wrap_state as _wrap_v91_state,
)
from braids_core.sediment.suspended_lifecycle_live_v76 import (
    resolve_suspended_lifecycle_state_v76,
)
from braids_core.sediment.suspended_support_closures_v82 import (
    _wrap_state as _wrap_v82_state,
    _wrap_trial as _wrap_v82_trial,
    resolve_suspended_support_execution_v82,
)
from braids_core.sediment.transported_fluid_settling_v92 import (
    FLUID_SALINITY,
    FLUID_SCALAR_COUNT,
    FLUID_TEMPERATURE,
    MAX_UPDATES,
    MODE_TRANSPORTED,
    SETTLING_DIAMETER_STATIC_PROFILE,
    SUSPENDED_CAPACITY_LEGACY_INPUT,
    TEMPERATURE_FLOOR_C,
    FluidScalarTransportResultV92,
    TransportedFluidExecutionV92,
    TransportedFluidSettlingV92Error,
    TransportedFluidStateV92,
    SuspendedCapacityReceiptV92,
    _array_digest,
    _fingerprint,
    _fluid_result_token,
    _validate_state,
    _wrap_state as _wrap_v92_state,
    _wrap_trial as _wrap_v92_trial,
    commit_transported_fluid_trial_v92,
    load_transported_fluid_checkpoint_v92,
    make_accepted_scalar_hydraulic_face_receipt_v92,
    make_settling_diameter_receipt_v92,
    make_suspended_capacity_receipt_v92,
    rollback_transported_fluid_trial_v92,
    transported_fluid_checkpoint_v92,
    validate_settling_diameter_receipt_v92,
    validate_suspended_capacity_receipt_v92,
)
from braids_gpu.flow2dh_boundary_diffusion import (
    KERNELS as BOUNDARY_KERNELS,
    OPTIONS as CUDA_OPTIONS,
)
from braids_gpu.multifraction_suspended_lifecycle_v78 import (
    CUDA_THREADS_PER_BLOCK,
)
from braids_gpu.shared_turbulence_consumers_v91 import (
    ResidentSharedTurbulenceConsumersV91,
)
from braids_gpu.transported_fluid_settling_v92 import (
    CUDA_SOURCE as FLUID_CUDA_SOURCE,
    V92_KERNELS,
    ResidentParentTimedDynamicExchangeV92,
)
from validation.gpu_fwave.suspended_advection import (
    ResidentSuspendedAdvection,
)


ROOT = Path(__file__).resolve().parents[1]
BOUNDARY_SOURCE = ROOT/"braids_gpu/flow2dh_boundary_diffusion.cu"
ADVECTION_SOURCE = ROOT/"validation/gpu_fwave/suspended_advection.cu"
CUDA_COMPOSITION_ID = (
    "atomic_combined_sediment_active_fluid_implicit_midpoint_v92_v1")


@dataclass(frozen=True)
class ResidentSuspendedCapacityResolutionV94:
    """Authenticated host evidence plus its actual-CUDA numerical value.

    V92 deliberately knows nothing about the v94 constitutive provider.  This
    narrow carrier lets that provider publish the ordinary core receipt while
    retaining the capacity array on the selected CUDA device.  The host copy
    is provenance/checkpoint evidence only; it is never copied back into the
    numerical path.
    """

    core_receipt: SuspendedCapacityReceiptV92
    resident_capacity: object


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TransportedFluidSettlingV92Error(message)


def _cancelled(value) -> bool:
    if callable(value):
        return bool(value())
    if type(value) is bool:
        return value
    raise TypeError("cancelled must be Boolean or callable")


def _frozen(value) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=np.float64)
    result.setflags(write=False)
    return result


def _geometry(base) -> dict:
    return {
        "scheme": np.int32(6), "amr_enabled": np.int32(0),
        "nb": np.int32(base.boundary_count),
        "dry_tolerance": np.float64(base.dry_tolerance_m),
        "active": np.ones(base.cell_count, dtype=np.int32),
        "area": np.array(base.cell_area_m2, copy=True),
        "length": np.array(base.face_length_m, copy=True),
        "normal": np.array(base.normal, copy=True),
        "rectangle_width": np.array(base.rectangle_width_m, copy=True),
        "weno_cells": np.array(base.weno_cells, copy=True),
        "owner": np.array(base.owner, copy=True),
        "neighbour": np.array(base.neighbour, copy=True),
        "bid": np.array(base.boundary_index, copy=True),
        "offset": np.array(base.cell_face_offset, copy=True),
        "faces": np.array(base.cell_faces, copy=True),
        "signs": np.array(base.cell_face_sign, copy=True),
    }


class _ResidentCombinedScalarOwnerV92:
    """One combined resident scalar operator and boundary-memory owner."""

    def __init__(self, execution: TransportedFluidExecutionV92):
        _require(isinstance(execution, TransportedFluidExecutionV92),
                 "v92 CUDA execution object differs")
        _require(execution.backend == "implicit_fwave_midpoint_gpu",
                 "v92 CUDA lifecycle requires the explicitly selected GPU")
        try:
            import cupy as cp
        except Exception as error:  # pragma: no cover - host dependent
            raise TransportedFluidSettlingV92Error(
                "v92 actual CUDA requires CuPy") from error
        try:
            _require(cp.cuda.runtime.getDeviceCount() >= 1,
                     "v92 actual CUDA requires a CUDA device")
        except TransportedFluidSettlingV92Error:
            raise
        except Exception as error:  # pragma: no cover - host dependent
            raise TransportedFluidSettlingV92Error(
                "v92 CUDA device discovery failed") from error

        self.execution, self.cp = execution, cp
        self.sediment_execution = execution.base_execution.base_execution
        self.base = self.sediment_execution.base_execution
        self.nsed = int(execution.fraction_count)
        self.active_fluid = tuple(
            index for index, active in enumerate(execution.transported_mask)
            if active)
        self.nfluid = len(self.active_fluid)
        self.ntotal = self.nsed+self.nfluid
        self.nc, self.nf, self.nb = (
            execution.cell_count, execution.face_count,
            execution.boundary_count)
        self.device_id = int(cp.cuda.Device().id)
        try:
            self.advection = ResidentSuspendedAdvection(
                _geometry(self.base), self.ntotal)
            boundary_module = cp.RawModule(
                code=BOUNDARY_SOURCE.read_text(encoding="utf-8"),
                options=CUDA_OPTIONS, name_expressions=BOUNDARY_KERNELS)
            self.boundary_kernels = {
                name: boundary_module.get_function(name)
                for name in BOUNDARY_KERNELS
            }
            fluid_module = cp.RawModule(
                code=FLUID_CUDA_SOURCE.read_text(encoding="utf-8"),
                options=CUDA_OPTIONS, name_expressions=V92_KERNELS)
            self.fluid_kernels = {
                name: fluid_module.get_function(name)
                for name in V92_KERNELS
            }
        except Exception as error:
            raise TransportedFluidSettlingV92Error(
                "v92 combined CUDA kernels failed to compile") from error

        a = self.advection.geometry
        self.owner, self.neighbour, self.bid = (
            a["owner"], a["neighbour"], a["bid"])
        self.offset, self.faces, self.signs = (
            a["offset"], a["faces"], a["signs"])
        self.length, self.area = a["length"], a["area"]
        self.normal, self.rectangle, self.weno = (
            a["normal"], a["rectangle_width"], a["weno_cells"])
        self.centre_distance = cp.asarray(np.ascontiguousarray(
            self.base.centre_distance_m, dtype=np.float64))
        bids = np.asarray(self.base.boundary_index)
        groups = [np.flatnonzero(bids == boundary).astype(np.int32)
                  for boundary in range(self.nb)]
        self.boundary_faces = cp.asarray(
            np.concatenate(groups) if groups else np.empty(0, np.int32))
        self.boundary_offset = cp.asarray(np.r_[
            0, np.cumsum([len(group) for group in groups])].astype(np.int32))

        sediment_return = np.zeros((self.nsed, self.nf))
        fluid_return = np.zeros((self.nfluid, self.nf))
        open_face = bids >= 0
        if self.nb:
            sediment_return[:, open_face] = \
                self.base.boundary_return_time_s[
                    :, bids[open_face]]
            for local, lane in enumerate(self.active_fluid):
                fluid_return[local, open_face] = \
                    execution.fluid_boundary_return_time_s[
                        lane, bids[open_face]]
        self.return_time_host = np.vstack((sediment_return, fluid_return))
        self.return_time = cp.asarray(self.return_time_host)
        sediment_neumann = np.asarray(
            self.base.profile.neumann_boundary_by_fraction,
            dtype=np.int32)
        self.neumann = cp.asarray(np.r_[
            sediment_neumann, np.zeros(self.nfluid, dtype=np.int32)])

        ft, fc, fb = ((self.ntotal, self.nf),
                      (self.ntotal, self.nc),
                      (self.ntotal, self.nb))
        self.accepted_last = cp.zeros(ft, dtype=cp.float64)
        self.accepted_clock = cp.zeros(ft, dtype=cp.float64)
        self.trial_last = cp.zeros(ft, dtype=cp.float64)
        self.trial_clock = cp.zeros(ft, dtype=cp.float64)
        self.face_boundary = cp.zeros(ft, dtype=cp.float64)
        self.expected_mass_in = cp.zeros(ft, dtype=cp.float64)
        self.expected_ledger = cp.zeros(fb, dtype=cp.float64)
        self.boundary_status = cp.zeros(ft, dtype=cp.int32)
        self.expected_status = cp.zeros(fb, dtype=cp.int32)
        self.effective = cp.zeros(ft, dtype=cp.float64)
        self.diffusive_face_mass = cp.zeros(ft, dtype=cp.float64)
        self.diffusive_face_active = cp.zeros(ft, dtype=cp.int32)
        self.diffusive_face_status = cp.zeros(ft, dtype=cp.int32)
        self.theta = cp.ones(fc, dtype=cp.float64)
        self.theta_status = cp.zeros(fc, dtype=cp.int32)
        self.diffused_mass = cp.zeros(fc, dtype=cp.float64)
        self.diffused_status = cp.zeros(fc, dtype=cp.int32)
        self.diffusive_ledger = cp.zeros(fb, dtype=cp.float64)
        self.diffusive_ledger_status = cp.zeros(fb, dtype=cp.int32)
        self.dicuv = cp.zeros(self.nc, dtype=cp.float64)

        # Private two-lane staging for affine T/S, regardless of how many
        # lanes are active.  Only active rows enter the combined owner above.
        self.fluid_mass0 = cp.zeros((2, self.nc), dtype=cp.float64)
        self.fluid_parent = cp.zeros((2, self.nc), dtype=cp.float64)
        self.fluid_pack_status = cp.zeros(self.nc, dtype=cp.int32)
        self.fluid_left = cp.zeros((2, self.nf), dtype=cp.float64)
        self.fluid_right = cp.zeros((2, self.nf), dtype=cp.float64)
        self.fluid_prescribed = cp.zeros((2, self.nb), dtype=cp.float64)
        self.fluid_left_status = cp.zeros((2, self.nf), dtype=cp.int32)
        self.fluid_right_status = cp.zeros((2, self.nf), dtype=cp.int32)
        self.fluid_prescribed_status = cp.zeros(
            (2, self.nb), dtype=cp.int32)
        self.full_final_fluid = cp.zeros((2, self.nc), dtype=cp.float64)
        self.h_temperature = cp.zeros(self.nc, dtype=cp.float64)
        self.h_salinity = cp.zeros(self.nc, dtype=cp.float64)
        self.temperature = cp.zeros(self.nc, dtype=cp.float64)
        self.salinity = cp.zeros(self.nc, dtype=cp.float64)
        self.wetdry_fluid_removed = cp.zeros((2, self.nc), dtype=cp.float64)
        self.fluid_unpack_status = cp.zeros(self.nc, dtype=cp.int32)
        self.water_residual = cp.zeros(self.nc, dtype=cp.float64)
        self.water_tolerance = cp.zeros(self.nc, dtype=cp.float64)
        self.water_status = cp.zeros(self.nc, dtype=cp.int32)
        self.water_ledger = cp.zeros(self.nb, dtype=cp.float64)
        self.water_ledger_status = cp.zeros(self.nb, dtype=cp.int32)
        self.endpoint_density = cp.zeros(self.nc, dtype=cp.float64)
        self.endpoint_viscosity = cp.zeros(self.nc, dtype=cp.float64)
        self.endpoint_property_status = cp.zeros(self.nc, dtype=cp.int32)

        self.exchange = ResidentParentTimedDynamicExchangeV92(
            self.base, execution)
        self._open = False
        self._parent_sha = None
        self.kernel_launch_counts = {
            "parent_settling": 0, "endpoint_settling_wet_gate": 0,
            "fluid_pack": 0,
            "water_continuity": 0, "combined_boundary": 0,
            "combined_fct": 0, "combined_diffusion": 0,
            "exchange": 0, "fluid_unpack": 0, "properties": 0,
        }

    def _launch(self, table, name, count, args) -> None:
        if count:
            table[name](((count+CUDA_THREADS_PER_BLOCK-1)//
                         CUDA_THREADS_PER_BLOCK,),
                        (CUDA_THREADS_PER_BLOCK,), args)

    def _device(self, value, shape, dtype, name):
        cp = self.cp
        _require(isinstance(value, cp.ndarray) and value.shape == shape and
                 value.dtype == dtype and value.flags.c_contiguous and
                 int(value.device.id) == self.device_id,
                 f"v92 CUDA {name} must be contiguous resident {dtype} "
                 f"{shape}")
        return value

    def _readback(self, value):
        """Single post-kernel publication seam for failure injection."""
        return self.cp.asnumpy(value)

    def _restore_memory(self, parent: TransportedFluidStateV92) -> None:
        sediment_last = np.asarray(
            parent.suspended_state.lifecycle.
            boundary_last_outflow_concentration_kg_m3)
        sediment_clock = np.asarray(
            parent.suspended_state.lifecycle.boundary_return_clock_s)
        fluid_last = np.asarray(parent.boundary_last_outflow_fluid)
        fluid_clock = np.asarray(parent.boundary_return_clock_s)
        last = np.zeros((self.ntotal, self.nf))
        clock = np.zeros((self.ntotal, self.nf))
        last[:self.nsed] = sediment_last
        clock[:self.nsed] = sediment_clock
        for local, lane in enumerate(self.active_fluid):
            value = np.array(fluid_last[lane], copy=True)
            if lane == FLUID_TEMPERATURE:
                value -= TEMPERATURE_FLOOR_C
            last[self.nsed+local] = value
            clock[self.nsed+local] = fluid_clock[lane]
        closed = np.asarray(self.base.boundary_index) < 0
        last[:, closed] = 0.0
        clock[:, closed] = 0.0
        _require(np.isfinite(last).all() and np.isfinite(clock).all() and
                 np.all(last >= 0.0) and np.all(clock >= 0.0),
                 "v92 CUDA combined boundary memory differs")
        tolerance = 256.0*np.finfo(float).eps*np.maximum(
            1.0, self.return_time_host)
        positive = self.return_time_host > 0.0
        _require(np.all(clock[positive] <=
                        self.return_time_host[positive]+tolerance[positive])
                 and np.all(clock[~positive] == 0.0),
                 "v92 CUDA combined boundary clocks differ")
        self.cp.copyto(self.accepted_last, self.cp.asarray(last))
        self.cp.copyto(self.accepted_clock, self.cp.asarray(clock))
        self.cp.copyto(self.trial_last, self.accepted_last)
        self.cp.copyto(self.trial_clock, self.accepted_clock)

    def abort(self) -> None:
        if self._open:
            self.cp.copyto(self.trial_last, self.accepted_last)
            self.cp.copyto(self.trial_clock, self.accepted_clock)
        self.exchange._prepared = False
        self.exchange.parent_settling_diameter = None
        self._open = False
        self._parent_sha = None

    def commit(self, parent_sha: str) -> None:
        _require(self._open and self._parent_sha == parent_sha,
                 "v92 CUDA combined transaction is not open for commit")
        self.cp.copyto(self.accepted_last, self.trial_last)
        self.cp.copyto(self.accepted_clock, self.trial_clock)
        self._open = False
        self._parent_sha = None

    def rollback(self, parent_sha: str) -> None:
        _require(self._open and self._parent_sha == parent_sha,
                 "v92 CUDA combined transaction is not open for rollback")
        self.cp.copyto(self.trial_last, self.accepted_last)
        self.cp.copyto(self.trial_clock, self.accepted_clock)
        self._open = False
        self._parent_sha = None

    def begin(self, parent: TransportedFluidStateV92, *,
              dicuv_cell_m2_s, depth_before_m, depth_after_m,
              accepted_water_flux_m2_s, endpoint_momentum_m2_s,
              reconstructed_left_concentration_kg_m3,
              reconstructed_right_concentration_kg_m3,
              prescribed_boundary_concentration_kg_m3,
              reconstructed_left_fluid, reconstructed_right_fluid,
              prescribed_boundary_fluid,
              accepted_face_exchange_area_m2, transport_face_active,
              shear_velocity_m_s, suspended_transport_capacity_kg_m_s,
              surface_fraction, hydro_constant, settling_diameter_m=None,
              settling_diameter_receipt=None,
              suspended_capacity_resolver=None,
              dt_s: float,
              cancelled=False) -> dict:
        """Run one combined scalar owner and publish host evidence once."""
        cp = self.cp
        _validate_state(parent, self.execution)
        if settling_diameter_receipt is None:
            settling_diameter_receipt = make_settling_diameter_receipt_v92(
                self.execution, parent)
        validate_settling_diameter_receipt_v92(
            settling_diameter_receipt, self.execution, parent)
        _require(not self._open,
                 "v92 CUDA previous combined scalar trial is unfinished")
        _require(not _cancelled(cancelled),
                 "v92 CUDA combined scalar trial cancelled before launch")
        _require(not isinstance(dt_s, bool) and
                 isinstance(dt_s, (int, float)) and
                 np.isfinite(dt_s) and dt_s > 0.0,
                 "v92 CUDA timestep must be finite and positive")
        dt = float(dt_s)
        h0 = self._device(depth_before_m, (self.nc,), cp.float64,
                          "depth before")
        h1 = self._device(depth_after_m, (self.nc,), cp.float64,
                          "depth after")
        water = self._device(accepted_water_flux_m2_s, (self.nf,),
                             cp.float64, "accepted water flux")
        momentum = self._device(endpoint_momentum_m2_s, (2, self.nc),
                                cp.float64, "endpoint momentum")
        sediment_left = self._device(
            reconstructed_left_concentration_kg_m3,
            (self.nsed, self.nf), cp.float64,
            "sediment left reconstruction")
        sediment_right = self._device(
            reconstructed_right_concentration_kg_m3,
            (self.nsed, self.nf), cp.float64,
            "sediment right reconstruction")
        sediment_prescribed = self._device(
            prescribed_boundary_concentration_kg_m3,
            (self.nsed, self.nb), cp.float64,
            "prescribed sediment boundary")
        fluid_left_actual = self._device(
            reconstructed_left_fluid, (2, self.nf), cp.float64,
            "fluid left reconstruction")
        fluid_right_actual = self._device(
            reconstructed_right_fluid, (2, self.nf), cp.float64,
            "fluid right reconstruction")
        fluid_prescribed_actual = self._device(
            prescribed_boundary_fluid, (2, self.nb), cp.float64,
            "prescribed fluid boundary")
        face_area = self._device(
            accepted_face_exchange_area_m2, (self.nf,), cp.float64,
            "face exchange area")
        face_active = self._device(
            transport_face_active, (self.nf,), cp.int32,
            "transport face activity")
        shear = self._device(shear_velocity_m_s, (self.nc,), cp.float64,
                             "shear velocity")
        capacity = None
        if suspended_capacity_resolver is None:
            capacity = self._device(
                suspended_transport_capacity_kg_m_s,
                (self.nsed, self.nc), cp.float64,
                "suspended transport capacity")
        else:
            _require(callable(suspended_capacity_resolver),
                     "v92 CUDA suspended-capacity resolver must be callable")
        fraction = self._device(surface_fraction,
                                (self.nsed, self.nc), cp.float64,
                                "surface fraction")
        constant = self._device(hydro_constant, (self.nc,), cp.int32,
                                "hydro constant")
        dicuv = self._device(dicuv_cell_m2_s, (self.nc,), cp.float64,
                             "capped dicuv")
        settling_diameter = None
        if settling_diameter_m is not None:
            settling_diameter = self._device(
                settling_diameter_m, (self.nsed, self.nc), cp.float64,
                "parent settling diameter")
        for value, name in (
            (h0, "depth before"), (h1, "depth after"),
            (water, "water flux"), (momentum, "endpoint momentum"),
            (sediment_left, "sediment left"),
            (sediment_right, "sediment right"),
            (sediment_prescribed, "sediment boundary"),
            (fluid_left_actual, "fluid left"),
            (fluid_right_actual, "fluid right"),
            (fluid_prescribed_actual, "fluid boundary"),
            (face_area, "face area"), (shear, "shear"),
            (fraction, "surface fraction"),
            (dicuv, "dicuv"),
        ):
            _require(bool(cp.all(cp.isfinite(value)).item()),
                     "v92 CUDA input must be finite: "+name)
        _require(bool(cp.all((h0 >= 0.0) & (h1 >= 0.0)).item()) and
                 bool(cp.all(sediment_left >= 0.0).item()) and
                 bool(cp.all(sediment_right >= 0.0).item()) and
                 bool(cp.all(sediment_prescribed >= 0.0).item()) and
                 bool(cp.all(face_area >= 0.0).item()) and
                 bool(cp.all(shear >= 0.0).item()) and
                 bool(cp.all(dicuv >= 0.0).item()) and
                 bool(cp.all((fraction >= 0.0) & (fraction <= 1.0)).item()),
                 "v92 CUDA nonnegative scalar inputs differ")
        if settling_diameter is not None:
            _require(bool(cp.all(cp.isfinite(settling_diameter) &
                                (settling_diameter > 0.0)).item()),
                     "v92 CUDA parent settling diameter must be positive")
        if capacity is not None:
            _require(bool(cp.all(cp.isfinite(capacity) &
                                (capacity >= 0.0)).item()),
                     "v92 CUDA suspended capacity must be nonnegative")
        _require(bool(cp.all((face_active == 0) | (face_active == 1)).item())
                 and bool(cp.all((constant == 0) | (constant == 1)).item()),
                 "v92 CUDA activity masks must contain only zero or one")
        closed = ((np.asarray(self.base.neighbour) < 0) &
                  (np.asarray(self.base.boundary_index) < 0))
        if np.any(closed):
            _require(not bool(cp.any(
                face_active[cp.asarray(closed)] != 0).item()),
                "v92 CUDA closed face cannot be diffusion-active")

        self._restore_memory(parent)
        parent_suspended = cp.ascontiguousarray(cp.asarray(
            parent.suspended_state.suspended_mass_kg_m2,
            dtype=cp.float64))
        parent_bed = cp.ascontiguousarray(cp.asarray(
            parent.suspended_state.bed_mass_kg_m2, dtype=cp.float64))
        parent_ht = cp.ascontiguousarray(cp.asarray(
            parent.h_temperature_c_m, dtype=cp.float64))
        parent_hs = cp.ascontiguousarray(cp.asarray(
            parent.h_salinity_ppt_m, dtype=cp.float64))
        parent_temperature = cp.ascontiguousarray(cp.asarray(
            parent.temperature_cell_c, dtype=cp.float64))
        parent_density = cp.ascontiguousarray(cp.asarray(
            parent.water_density_cell_kg_m3, dtype=cp.float64))
        self.cp.copyto(self.dicuv, dicuv)

        try:
            # FLOW fallve timing: prepare ws from parent ctot/T/rho before the
            # single combined advection/diffusion owner is launched.
            self.exchange.prepare_parent(
                depth_before=h0, suspended_mass=parent_suspended,
                temperature=parent_temperature, water_density=parent_density,
                settling_diameter_m=settling_diameter)
            self.kernel_launch_counts["parent_settling"] += 1
            # FLOW advances hydraulics before the sediment stage.  Parent
            # ctot/T/rho/DSS remain lagged inputs to SettleFrm=2, but a cell
            # that is dry at the accepted hydraulic endpoint has no current
            # settling column.  Apply that endpoint wet gate before the v83
            # capacity callback so its authenticated incoming-ws digest is
            # identical to the value later consumed/published by exchange.
            if (self.execution.base_execution.state_domain_mode ==
                    STATE_DOMAIN_MOVING_WETDRY_V97):
                endpoint_dry = h1 <= self.base.dry_tolerance_m
                self.exchange.settling_velocity[:, endpoint_dry] = 0.0
                self.kernel_launch_counts[
                    "endpoint_settling_wet_gate"] += 1
            _require(not _cancelled(cancelled),
                     "v92 CUDA scalar trial cancelled after settling launch")

            # FLOW order is fallve(old accepted DSS), then eqtran/tram1 at
            # the accepted hydraulic endpoint.  The v94 provider therefore
            # resolves capacity here, exactly once, while its numerical array
            # stays resident.  The host receipt is authentication evidence,
            # not a CPU numerical path.
            accepted_q = cp.ascontiguousarray(cp.vstack((h1, momentum)))
            accepted_q_host = self._readback(accepted_q)
            if suspended_capacity_resolver is not None:
                try:
                    resolved = suspended_capacity_resolver(
                        execution=self.execution, parent=parent,
                        accepted_q=accepted_q,
                        settling_velocity_m_s=
                            self.exchange.settling_velocity,
                        settling_diameter_receipt=
                            settling_diameter_receipt,
                        surface_fraction=fraction,
                        shear_velocity_m_s=shear)
                except InterruptedError:
                    raise
                except Exception as error:
                    raise TransportedFluidSettlingV92Error(
                        "v92 CUDA suspended-capacity resolver failed") \
                        from error
                _require(isinstance(
                    resolved, ResidentSuspendedCapacityResolutionV94),
                    "v92 CUDA suspended-capacity resolver did not return "
                    "resident authenticated evidence")
                capacity = self._device(
                    resolved.resident_capacity,
                    (self.nsed, self.nc), cp.float64,
                    "resolved suspended transport capacity")
                capacity_receipt = resolved.core_receipt
                validate_suspended_capacity_receipt_v92(
                    capacity_receipt, self.execution, parent,
                    accepted_q=accepted_q_host,
                    settling_diameter_receipt=settling_diameter_receipt)
                _require(bool(cp.array_equal(
                    capacity, cp.asarray(
                        capacity_receipt.suspended_capacity_kg_m_s,
                        dtype=cp.float64)).item()),
                    "v92 CUDA resident capacity differs from its receipt")
            else:
                assert capacity is not None
                capacity_host = self._readback(capacity)
                capacity_receipt = make_suspended_capacity_receipt_v92(
                    self.execution, parent,
                    accepted_q=accepted_q_host,
                    settling_diameter_receipt=settling_diameter_receipt,
                    suspended_capacity_kg_m_s=capacity_host,
                    source_kind=SUSPENDED_CAPACITY_LEGACY_INPUT,
                    source_receipt_sha256=_array_digest(capacity_host))
            _require(bool(cp.all(cp.isfinite(capacity) &
                                (capacity >= 0.0)).item()),
                     "v92 CUDA resolved suspended capacity must be "
                     "nonnegative")
            _require(not _cancelled(cancelled),
                     "v92 CUDA scalar trial cancelled after capacity launch")

            self.kernel_launch_counts["fluid_pack"] += 4
            self._launch(self.fluid_kernels, "v92_pack_affine_fluid",
                         self.nc, (
                np.int32(self.nc), np.float64(self.base.dry_tolerance_m),
                np.float64(TEMPERATURE_FLOOR_C), h0, parent_ht, parent_hs,
                self.fluid_mass0, self.fluid_parent,
                self.fluid_pack_status))
            for actual, shifted, status, count in (
                (fluid_left_actual, self.fluid_left,
                 self.fluid_left_status, self.nf),
                (fluid_right_actual, self.fluid_right,
                 self.fluid_right_status, self.nf),
                (fluid_prescribed_actual, self.fluid_prescribed,
                 self.fluid_prescribed_status, self.nb),
            ):
                self._launch(self.fluid_kernels, "v92_shift_fluid_values",
                             2*count, (
                    np.int32(count), np.float64(TEMPERATURE_FLOOR_C),
                    actual, shifted, status))
            _require(not bool(cp.any(self.fluid_pack_status != 0).item()) and
                     not bool(cp.any(self.fluid_left_status != 0).item()) and
                     not bool(cp.any(self.fluid_right_status != 0).item()) and
                     not bool(cp.any(
                         self.fluid_prescribed_status != 0).item()),
                     "v92 CUDA fluid affine preparation rejected the trial")

            if FLUID_TEMPERATURE in self.active_fluid:
                self._launch(self.fluid_kernels, "v92_water_continuity",
                             self.nc, (
                    np.int32(self.nc), np.float64(dt), h0, h1, water,
                    self.length, self.area, self.offset, self.faces,
                    self.signs, self.water_residual,
                    self.water_tolerance, self.water_status))
                self.kernel_launch_counts["water_continuity"] += 1
                _require(not bool(cp.any(self.water_status != 0).item()),
                         "v92 CUDA affine hT requires accepted cellwise "
                         "water continuity")
            else:
                self.water_residual.fill(0.0)
                self.water_tolerance.fill(0.0)
                self.water_status.fill(0)

            active_index = cp.asarray(self.active_fluid, dtype=cp.int32)
            mass = (cp.concatenate((parent_suspended,
                                    self.fluid_mass0[active_index]), axis=0)
                    if self.nfluid else parent_suspended)
            left = (cp.concatenate((sediment_left,
                                    self.fluid_left[active_index]), axis=0)
                    if self.nfluid else sediment_left)
            right = (cp.concatenate((sediment_right,
                                     self.fluid_right[active_index]), axis=0)
                     if self.nfluid else sediment_right)
            prescribed = (cp.concatenate((
                sediment_prescribed,
                self.fluid_prescribed[active_index]), axis=0)
                if self.nfluid else sediment_prescribed)
            mass = cp.ascontiguousarray(mass)
            left, right, prescribed = map(cp.ascontiguousarray,
                                           (left, right, prescribed))

            self._launch(self.boundary_kernels, "m2b_resolve_boundary",
                         self.ntotal*self.nf, (
                np.int32(self.nc), np.int32(self.nf),
                np.int32(self.ntotal), np.int32(self.nb),
                np.float64(self.base.dry_tolerance_m), np.float64(dt), h0,
                mass, prescribed, water, self.length, self.owner,
                self.neighbour, self.bid, self.neumann, self.return_time,
                self.accepted_last, self.accepted_clock,
                self.face_boundary, self.expected_mass_in,
                self.trial_last, self.trial_clock, self.boundary_status))
            self._launch(
                self.boundary_kernels, "m2b_expected_advective_ledger",
                self.ntotal*self.nb, (
                    np.int32(self.nf), np.int32(self.ntotal),
                    np.int32(self.nb), self.boundary_offset,
                    self.boundary_faces, self.expected_mass_in,
                    self.boundary_status, self.expected_ledger,
                    self.expected_status))
            self.kernel_launch_counts["combined_boundary"] += 2

            adv, a, b = (self.advection, self.advection.geometry,
                         self.advection.buffers)
            adv._launch("sa_concentration", self.nc*self.ntotal, (
                np.int32(self.nc), np.int32(self.ntotal), h0, mass,
                np.float64(adv.dry), b["concentration"],
                b["concentration_status"]))
            self._launch(self.boundary_kernels, "m2b_sa_face_low",
                         self.nf*self.ntotal, (
                np.int32(self.nc), np.int32(self.nf),
                np.int32(self.ntotal), water, b["concentration"], left,
                right, self.face_boundary, a["owner"], a["neighbour"],
                a["bid"], b["concentration_status"], b["low_flux"],
                b["anti"], b["face_status"]))
            self._launch(self.boundary_kernels, "m2b_sa_cell_bounds",
                         self.nc*self.ntotal, (
                np.int32(self.nc), np.int32(self.nf),
                np.int32(self.ntotal), mass, h1, b["concentration"],
                self.face_boundary, water, b["low_flux"], b["anti"],
                a["area"], a["length"], a["normal"],
                a["rectangle_width"], constant, a["weno_cells"],
                a["owner"], a["neighbour"], a["bid"], a["offset"],
                a["faces"], a["signs"], b["concentration_status"],
                b["face_status"], np.float64(dt), b["low_state"],
                b["rplus"], b["rminus"], b["cell_status"]))
            adv._launch("sa_limit_faces", self.nf*self.ntotal, (
                np.int32(self.nc), np.int32(self.nf),
                np.int32(self.ntotal), b["low_flux"], b["anti"],
                b["rplus"], b["rminus"], a["owner"], a["neighbour"],
                b["cell_status"], b["scaled_anti"], b["flux"],
                b["flux_status"]))
            adv._launch("sa_update", self.nc*self.ntotal, (
                np.int32(self.nc), np.int32(self.nf),
                np.int32(self.ntotal), b["low_state"], b["scaled_anti"],
                h1, a["length"], a["area"], a["offset"], a["faces"],
                a["signs"], b["cell_status"], b["flux_status"],
                np.float64(dt), np.float64(adv.dry), b["next_mass"],
                b["next_concentration"], b["update_status"]))
            adv._launch("sa_boundary_ledger", self.nb*self.ntotal, (
                np.int32(self.nf), np.int32(self.ntotal),
                np.int32(self.nb), a["boundary_offset"],
                a["boundary_faces"], b["flux"], a["length"],
                b["flux_status"], np.float64(dt), b["ledger"],
                b["ledger_status"]))
            self.kernel_launch_counts["combined_fct"] += 1
            statuses = (self.boundary_status, self.expected_status,
                        b["concentration_status"], b["face_status"],
                        b["cell_status"], b["flux_status"],
                        b["update_status"], b["ledger_status"])
            _require(not any(bool(cp.any(value != 0).item())
                             for value in statuses),
                     "v92 CUDA combined boundary/FCT rejected the trial")
            scale = cp.maximum(1.0, cp.maximum(
                cp.abs(self.expected_ledger), cp.abs(b["ledger"])))
            _require(not bool(cp.any(cp.abs(
                self.expected_ledger-b["ledger"]) >
                8192.0*np.finfo(float).eps*scale).item()),
                "v92 CUDA combined FCT ledger differs")

            self._launch(self.boundary_kernels, "m2b_diffusion_faces",
                         self.ntotal*self.nf, (
                np.int32(self.nc), np.int32(self.nf),
                np.int32(self.ntotal),
                np.float64(self.base.profile.diffusion_dry_threshold_m),
                np.float64(dt), h1, b["next_mass"], self.face_boundary,
                self.dicuv, face_area, self.centre_distance, face_active,
                self.owner, self.neighbour, self.bid, self.effective,
                self.diffusive_face_mass, self.diffusive_face_active,
                self.diffusive_face_status))
            self._launch(self.boundary_kernels, "m2b_diffusion_theta",
                         self.ntotal*self.nc, (
                np.int32(self.nc), np.int32(self.nf),
                np.int32(self.ntotal), b["next_mass"],
                self.diffusive_face_mass, self.area, self.offset,
                self.faces, self.signs, self.diffusive_face_status,
                self.theta, self.theta_status))
            self._launch(self.boundary_kernels, "m2b_diffusion_limit",
                         self.ntotal*self.nf, (
                np.int32(self.nc), np.int32(self.nf),
                np.int32(self.ntotal), self.owner, self.neighbour,
                self.theta, self.theta_status, self.diffusive_face_mass,
                self.diffusive_face_status))
            self._launch(self.boundary_kernels, "m2b_diffusion_update",
                         self.ntotal*self.nc, (
                np.int32(self.nc), np.int32(self.nf),
                np.int32(self.ntotal), b["next_mass"],
                self.diffusive_face_mass, self.area, self.offset,
                self.faces, self.signs, self.diffusive_face_status,
                self.diffused_mass, self.diffused_status))
            self._launch(self.boundary_kernels, "m2b_diffusion_ledger",
                         self.ntotal*self.nb, (
                np.int32(self.nf), np.int32(self.ntotal),
                np.int32(self.nb), self.boundary_offset,
                self.boundary_faces, self.diffusive_face_mass,
                self.diffusive_face_status, self.diffusive_ledger,
                self.diffusive_ledger_status))
            self.kernel_launch_counts["combined_diffusion"] += 1
            _require(not any(bool(cp.any(value != 0).item()) for value in (
                self.diffusive_face_status, self.theta_status,
                self.diffused_status, self.diffusive_ledger_status)),
                "v92 CUDA combined diffusion rejected the trial")

            wet = (h1 > self.base.dry_tolerance_m).astype(cp.int32)
            speed = cp.where(wet != 0, cp.hypot(momentum[0], momentum[1])/
                             cp.where(wet != 0, h1, 1.0), 0.0)
            sediment_active = ((wet != 0) &
                               (h1 > self.base.profile.sediment_threshold_m))
            safe_h1 = cp.where(
                sediment_active,
                cp.maximum(self.base.profile.reference_depth_floor_m, h1),
                1.0)
            equilibrium_capacity = cp.ascontiguousarray(
                capacity/(speed[None, :]+1.0e-6)/safe_h1[None, :])
            total_shear = cp.ascontiguousarray(
                self.base.profile.water_density_kg_m3*shear*shear)
            exchanged = self.exchange.step(
                h0, h1, momentum, self.diffused_mass[:self.nsed],
                parent_bed, equilibrium_capacity, fraction, total_shear,
                cp.ascontiguousarray(wet), dt)
            self.kernel_launch_counts["exchange"] += 1
            _require(not bool(cp.any(exchanged["status"] != 0).item()),
                     "v92 CUDA parent-timed exchange rejected the trial")
            cleanup_sediment = cp.where(
                wet[None, :] != 0, 0.0, exchanged["suspended"])
            final_suspended = exchanged["suspended"]-cleanup_sediment
            final_bed = exchanged["active"]+cleanup_sediment

            # Expand only active fluid rows back into fixed physical T/S order.
            constants = (self.execution.equilibrium_temperature_c-
                         TEMPERATURE_FLOOR_C,
                         self.execution.equilibrium_salinity_ppt)
            self.full_final_fluid[0] = h1*constants[0]
            self.full_final_fluid[1] = h1*constants[1]
            for local, lane in enumerate(self.active_fluid):
                self.full_final_fluid[lane] = \
                    self.diffused_mass[self.nsed+local]
            self._launch(self.fluid_kernels, "v92_unpack_affine_fluid",
                         self.nc, (
                np.int32(self.nc), np.float64(self.base.dry_tolerance_m),
                np.float64(TEMPERATURE_FLOOR_C),
                np.float64(self.execution.equilibrium_temperature_c),
                np.float64(self.execution.equilibrium_salinity_ppt), h1,
                self.full_final_fluid, self.h_temperature, self.h_salinity,
                self.temperature, self.salinity,
                self.wetdry_fluid_removed, self.fluid_unpack_status))
            self.kernel_launch_counts["fluid_unpack"] += 1
            _require(not bool(cp.any(self.fluid_unpack_status != 0).item()),
                     "v92 CUDA fluid affine inverse rejected the trial")
            # Inactive lanes never entered the combined operator.  Publish
            # their exact conserved equilibrium definition, avoiding an
            # unnecessary affine add/subtract round trip for constant T.
            if FLUID_TEMPERATURE not in self.active_fluid:
                cp.multiply(
                    h1, np.float64(self.execution.equilibrium_temperature_c),
                    out=self.h_temperature)
                self.temperature.fill(
                    self.execution.equilibrium_temperature_c)
                self.wetdry_fluid_removed[FLUID_TEMPERATURE].fill(0.0)
            if FLUID_SALINITY not in self.active_fluid:
                cp.multiply(
                    h1, np.float64(self.execution.equilibrium_salinity_ppt),
                    out=self.h_salinity)
                self.salinity.fill(self.execution.equilibrium_salinity_ppt)
                self.wetdry_fluid_removed[FLUID_SALINITY].fill(0.0)
            self._launch(self.fluid_kernels, "v92_boundary_water_ledger",
                         self.nb, (
                np.int32(self.nb), np.float64(dt), self.boundary_offset,
                self.boundary_faces, water, self.length,
                self.water_ledger, self.water_ledger_status))
            _require(not bool(cp.any(
                self.water_ledger_status != 0).item()),
                "v92 CUDA water ledger rejected the trial")
            eos_code = {"eckart": 0, "unesco": 1, "nacl": 2}[
                self.execution.density_eos]
            self._launch(self.fluid_kernels, "v92_fluid_properties",
                         self.nc, (
                np.int32(self.nc), self.temperature, self.salinity,
                np.int32(eos_code), np.int32(bool(self.nfluid)),
                np.float64(self.base.profile.water_density_kg_m3),
                self.endpoint_density, self.endpoint_viscosity,
                self.endpoint_property_status))
            self.kernel_launch_counts["properties"] += 1
            _require(not bool(cp.any(
                self.endpoint_property_status != 0).item()),
                "v92 CUDA endpoint fluid properties rejected the trial")
            _require(not _cancelled(cancelled),
                     "v92 CUDA scalar trial cancelled after resident launches")

            result = self._publish(
                parent, h1, water, wet, final_suspended, final_bed,
                cleanup_sediment, exchanged, b, dt)
            result["suspended_capacity_receipt"] = capacity_receipt
            _require(not _cancelled(cancelled),
                     "v92 CUDA scalar trial cancelled before publication")
            self._open = True
            self._parent_sha = parent.state_sha256
            return result
        except Exception:
            self.abort()
            raise

    def _publish(self, parent, h1, water, wet, final_suspended, final_bed,
                 cleanup_sediment, exchanged, advection_buffers,
                 dt: float) -> dict:
        """Split the one combined owner and construct immutable evidence."""
        cp = self.cp
        b = advection_buffers
        area = self.area[None, :]
        parent_bed = cp.asarray(parent.suspended_state.bed_mass_kg_m2)
        exchange_water = cp.sum(
            (exchanged["suspended"]-self.diffused_mass[:self.nsed])*area,
            axis=1)
        exchange_bed = cp.sum(
            (exchanged["active"]-parent_bed)*area, axis=1)
        cleanup_water = -cp.sum(cleanup_sediment*area, axis=1)
        cleanup_bed = -cleanup_water
        lifecycle = parent.suspended_state.lifecycle

        sediment_last = self._readback(self.trial_last[:self.nsed])
        sediment_clock = self._readback(self.trial_clock[:self.nsed])
        sediment_advective = self._readback(b["ledger"][:self.nsed])
        sediment_diffusive = self._readback(
            self.diffusive_ledger[:self.nsed])
        candidate_lifecycle = resolve_suspended_lifecycle_state_v76(
            self.base,
            suspended_mass_kg_m2=self._readback(final_suspended),
            bed_mass_kg_m2=self._readback(final_bed),
            boundary_last_outflow_concentration_kg_m3=sediment_last,
            boundary_return_clock_s=sediment_clock,
            advective_boundary_ledger_kg=(
                lifecycle.advective_boundary_ledger_kg+
                sediment_advective),
            diffusive_boundary_ledger_kg=(
                lifecycle.diffusive_boundary_ledger_kg+
                sediment_diffusive),
            water_exchange_ledger_kg=(
                lifecycle.water_exchange_ledger_kg+
                self._readback(exchange_water)),
            bed_exchange_ledger_kg=(
                lifecycle.bed_exchange_ledger_kg+
                self._readback(exchange_bed)),
            wetdry_water_ledger_kg=(
                lifecycle.wetdry_water_ledger_kg+
                self._readback(cleanup_water)),
            wetdry_bed_ledger_kg=(
                lifecycle.wetdry_bed_ledger_kg+
                self._readback(cleanup_bed)),
            reference_water_mass_kg=lifecycle.reference_water_mass_kg,
            reference_bed_mass_kg=lifecycle.reference_bed_mass_kg,
            accepted_updates=parent.accepted_updates+1,
            time_s=parent.time_s+dt)

        boundary_fluid = cp.empty(
            (FLUID_SCALAR_COUNT, self.nf), dtype=cp.float64)
        boundary_fluid[0].fill(self.execution.equilibrium_temperature_c)
        boundary_fluid[1].fill(self.execution.equilibrium_salinity_ppt)
        last_fluid = cp.zeros_like(boundary_fluid)
        clock_fluid = cp.zeros_like(boundary_fluid)
        advective_flux = cp.zeros_like(boundary_fluid)
        diffusive_amount = cp.zeros_like(boundary_fluid)
        advective_ledger = cp.zeros(
            (FLUID_SCALAR_COUNT, self.nb), dtype=cp.float64)
        diffusive_ledger = cp.zeros_like(advective_ledger)
        # KMAX=1 uses one constituent-independent dicuv/0.7 face field.
        # Publish it once, exactly like the CPU combined owner.
        effective = self.effective[0]
        lane_count = np.zeros(FLUID_SCALAR_COUNT, dtype=np.int32)
        open_face = cp.asarray(np.asarray(self.base.boundary_index) >= 0)
        for local, lane in enumerate(self.active_fluid):
            row = self.nsed+local
            boundary_fluid[lane] = self.face_boundary[row]
            last_fluid[lane] = self.trial_last[row]
            clock_fluid[lane] = self.trial_clock[row]
            advective_flux[lane] = b["flux"][row]
            diffusive_amount[lane] = self.diffusive_face_mass[row]
            advective_ledger[lane] = b["ledger"][row]
            diffusive_ledger[lane] = self.diffusive_ledger[row]
            lane_count[lane] = 1
            if lane == FLUID_TEMPERATURE:
                boundary_fluid[lane] += TEMPERATURE_FLOOR_C
                last_fluid[lane, open_face] += TEMPERATURE_FLOOR_C
                advective_flux[lane] += TEMPERATURE_FLOOR_C*water
                advective_ledger[lane] += (
                    TEMPERATURE_FLOOR_C*self.water_ledger)
        last_fluid[:, ~open_face] = 0.0
        wetdry = -cp.sum(
            self.wetdry_fluid_removed*self.area[None, :], axis=1)
        for lane in range(FLUID_SCALAR_COUNT):
            if lane not in self.active_fluid:
                wetdry[lane] = 0.0

        arrays = [
            _frozen(self._readback(value)) for value in (
                self.h_temperature, self.h_salinity,
                self.temperature, self.salinity, boundary_fluid,
                last_fluid, clock_fluid, advective_flux,
                diffusive_amount, advective_ledger, diffusive_ledger,
                wetdry, effective, self.water_residual)
        ]
        lane_frozen = np.ascontiguousarray(lane_count)
        lane_frozen.setflags(write=False)
        fluid_placeholder = FluidScalarTransportResultV92(
            *arrays, lane_frozen, "")
        fluid_result = FluidScalarTransportResultV92(
            **{**fluid_placeholder.__dict__,
               "result_sha256": _fluid_result_token(fluid_placeholder)})

        parent_ctot = cp.sum(
            cp.asarray(parent.suspended_state.suspended_mass_kg_m2)/
            cp.asarray(parent.q[0])[None, :], axis=0)
        return {
            "candidate_lifecycle": candidate_lifecycle,
            "sediment_boundary": self._readback(
                self.face_boundary[:self.nsed]),
            "sediment_advective_flux": self._readback(
                b["flux"][:self.nsed]),
            "sediment_diffusive_face_mass": self._readback(
                self.diffusive_face_mass[:self.nsed]),
            "sediment_effective_diffusivity": self._readback(
                self.effective[0]),
            "settling_velocity": self._readback(
                exchanged["settling_velocity"]),
            "equilibrium_concentration": self._readback(
                exchanged["equilibrium_concentration"]),
            "exchange_delta": self._readback(exchanged["delta"]),
            "wetdry_sediment_cleanup": self._readback(cleanup_sediment),
            "fluid_result": fluid_result,
            "endpoint_density": self._readback(self.endpoint_density),
            "endpoint_viscosity": self._readback(self.endpoint_viscosity),
            "parent_ctot": self._readback(parent_ctot),
        }


class ResidentTransportedFluidSettlingLifecycleV92:
    """Canonical hydraulic + combined-scalar actual-CUDA transaction."""

    is_v91_shared_turbulence_consumer = True
    is_v92_transported_fluid_consumer = True

    def __init__(self, execution: TransportedFluidExecutionV92):
        _require(isinstance(execution, TransportedFluidExecutionV92) and
                 execution.backend == "implicit_fwave_midpoint_gpu",
                 "v92 CUDA lifecycle requires explicit GPU execution")
        self.execution = execution
        self.base = ResidentSharedTurbulenceConsumersV91(
            execution.base_execution)
        self.combined = _ResidentCombinedScalarOwnerV92(execution)
        self.cp = self.base.cp
        self.device_id = self.base.device_id
        self._parent = None
        self._parent_hles = None
        self._handoff = None
        self._trial = None
        self._scalar_execution = None
        self.last_accepted_receipt = None
        properties = self.cp.cuda.runtime.getDeviceProperties(self.device_id)
        raw_name = properties.get("name", b"unknown")
        device_name = (raw_name.decode("utf-8", errors="replace")
                       if isinstance(raw_name, bytes) else str(raw_name))
        paths = (Path(__file__).resolve(), FLUID_CUDA_SOURCE,
                 BOUNDARY_SOURCE, ADVECTION_SOURCE)
        self.runtime_receipt = {
            "cuda_composition_id": CUDA_COMPOSITION_ID,
            "execution_fingerprint": execution.fingerprint,
            "integrator": "implicit_fwave_midpoint",
            "backend": "actual_cuda",
            "device_id": self.device_id,
            "device_name": device_name,
            "actual_cuda_kernels": True,
            "automatic_cpu_fallback": False,
            "one_combined_scalar_owner": True,
            "sediment_fraction_count": execution.fraction_count,
            "active_fluid_lanes": list(self.combined.active_fluid),
            "combined_lane_count": self.combined.ntotal,
            "constant_fluid_lanes_enter_fct": False,
            "parent_settling_before_combined_transport": True,
            "settling_diameter_source": (
                "typed_static_profile_or_accepted_parent_cellwise_DSS"),
            "resident_suspended_capacity_resolver": True,
            "resolved_capacity_cpu_numerical_fallback": False,
            "fluid_boundary_memory_units": "physical_T_C_and_S_ppt",
            "horizontal_diffusion": "shared_dicuv_over_0p7_once",
            "source_sha256": {
                str(path.relative_to(ROOT)).replace("\\", "/"):
                    hashlib.sha256(path.read_bytes()).hexdigest()
                for path in paths
            },
        }

    @property
    def kernel_launch_counts(self):
        return {
            "hydraulic": dict(self.base.kernel_launch_counts),
            "combined_scalar": dict(self.combined.kernel_launch_counts),
        }

    def bind_controller(self, controller) -> None:
        self.base.bind_controller(controller)

    def begin_hydraulic_trial(self, parent, parent_hles, handoff, *,
                              dt_s: float, time_s: float,
                              cancelled=False) -> None:
        _validate_state(parent, self.execution)
        _require(self._parent is None and self._trial is None,
                 "v92 CUDA previous outer transaction is unfinished")
        _require(not _cancelled(cancelled),
                 "v92 CUDA hydraulic trial cancelled before any launch")
        self.base.begin_hydraulic_trial(
            parent.base_state, parent_hles, handoff, dt_s=dt_s,
            time_s=time_s, cancelled=cancelled)
        if _cancelled(cancelled):
            self.base.abort_hydraulic_trial()
            raise TransportedFluidSettlingV92Error(
                "v92 CUDA hydraulic trial cancelled after child launch")
        self._parent = parent
        self._parent_hles = parent_hles
        self._handoff = handoff

    def require_open_for_controller(self, dt_s, time_s, q) -> None:
        self.base.require_open_for_controller(dt_s, time_s, q)

    def add_momentum_rate(self, q, full_rate) -> None:
        self.base.add_momentum_rate(q, full_rate)

    def finish_hydraulic_trial(self, midpoint_q, accepted_q,
                               accepted_water_flux, *, dt_s: float,
                               time_s: float) -> dict:
        return self.base.finish_hydraulic_trial(
            midpoint_q, accepted_q, accepted_water_flux,
            dt_s=dt_s, time_s=time_s)

    def abort_hydraulic_trial(self) -> None:
        self.combined.abort()
        self.base.abort_hydraulic_trial()
        self._parent = self._parent_hles = self._handoff = None
        self._trial = self._scalar_execution = None

    def begin_scalar_trial(self, *, reconstructed_left_fluid,
                           reconstructed_right_fluid,
                           prescribed_boundary_fluid,
                           settling_diameter_receipt=None,
                           suspended_capacity_resolver=None,
                           wetdry_hydraulic_evidence=None,
                           cancelled=False,
                           **scalar_inputs):
        _require(self._parent is not None and
                 self.base._pending is not None and self._trial is None,
                 "v92 CUDA scalar owner requires one hydraulic candidate")
        if _cancelled(cancelled):
            self.abort_hydraulic_trial()
            raise TransportedFluidSettlingV92Error(
                "v92 CUDA scalar trial cancelled before any launch")
        if self._parent.accepted_updates >= MAX_UPDATES:
            self.abort_hydraulic_trial()
            raise TransportedFluidSettlingV92Error(
                "v92 accepted update limit was reached")
        pending = self.base._pending
        moving_wetdry = (
            self.execution.base_execution.state_domain_mode ==
            STATE_DOMAIN_MOVING_WETDRY_V97)
        _require(moving_wetdry == (wetdry_hydraulic_evidence is not None),
                 "v92 CUDA moving wet/dry evidence selection differs")
        scalar_execution = resolve_suspended_support_execution_v82(
            self.execution.base_execution.base_execution,
            dicuv_cell_m2_s=self.base._cap.capped_total_dicuv_m2_s)
        scalar_parent = _wrap_v82_state(
            scalar_execution, self._parent.suspended_state)
        values = dict(scalar_inputs)
        hydraulic_face_receipt = \
            make_accepted_scalar_hydraulic_face_receipt_v92(
                self.execution, self._parent, pending["receipt"],
                accepted_q=pending["accepted"],
                accepted_water_flux_m2_s=pending["water"])
        if settling_diameter_receipt is None:
            settling_diameter_receipt = make_settling_diameter_receipt_v92(
                self.execution, self._parent)
        validate_settling_diameter_receipt_v92(
            settling_diameter_receipt, self.execution, self._parent)
        values.update(
            dicuv_cell_m2_s=self.cp.ascontiguousarray(self.base.capped_dic),
            depth_before_m=self.cp.ascontiguousarray(
                self.cp.asarray(self._parent.q[0])),
            depth_after_m=self.cp.ascontiguousarray(
                self.cp.asarray(pending["accepted"][0])),
            accepted_water_flux_m2_s=self.cp.ascontiguousarray(
                self.cp.asarray(
                    hydraulic_face_receipt.accepted_water_flux_m2_s)),
            endpoint_momentum_m2_s=self.cp.ascontiguousarray(
                self.cp.asarray(pending["accepted"][1:])),
            settling_diameter_m=(
                None if settling_diameter_receipt.source_kind ==
                SETTLING_DIAMETER_STATIC_PROFILE else
                self.cp.ascontiguousarray(self.cp.asarray(
                    settling_diameter_receipt.suspended_diameter_m,
                    dtype=self.cp.float64))),
            settling_diameter_receipt=settling_diameter_receipt,
            suspended_capacity_resolver=suspended_capacity_resolver,
            reconstructed_left_fluid=reconstructed_left_fluid,
            reconstructed_right_fluid=reconstructed_right_fluid,
            prescribed_boundary_fluid=prescribed_boundary_fluid,
            cancelled=cancelled,
            dt_s=self.base._dt,
        )
        try:
            receipt = validate_implicit_fwave_midpoint_step_receipt_v78(
                pending["receipt"].base_step_receipt,
                hydraulic_execution_sha256=
                    self.execution.base_execution.base_execution.
                    hydraulic_execution_sha256,
                parent_accepted_updates=self._parent.accepted_updates,
                parent_time_s=self._parent.time_s,
                dt_s=self.base._dt)
            output = self.combined.begin(self._parent, **values)
            base_trial = build_multifraction_trial_v78(
                self.execution.base_execution.base_execution,
                self._parent.suspended_state, receipt,
                output["candidate_lifecycle"],
                boundary_concentration_kg_m3=
                    output["sediment_boundary"],
                advective_face_flux_kg_m_s=
                    output["sediment_advective_flux"],
                diffusive_face_mass_owner_out_kg=
                    output["sediment_diffusive_face_mass"],
                settling_velocity_m_s=output["settling_velocity"],
                equilibrium_concentration_kg_m3=
                    output["equilibrium_concentration"],
                transferred_to_suspension_kg_m2=
                    output["exchange_delta"],
                wetdry_transfer_to_bed_kg_m2=
                    output["wetdry_sediment_cleanup"],
                observed_parallel_workers=CUDA_THREADS_PER_BLOCK,
                backend=ROLE_CUDA)
            suspended_trial = _wrap_v82_trial(
                scalar_execution, scalar_parent, base_trial,
                output["sediment_effective_diffusivity"])
            chain_v91 = _fingerprint({
                "previous":
                    self._parent.base_state.accepted_receipt_chain_sha256,
                "hydraulic": pending["receipt"].receipt_sha256,
                "scalar": suspended_trial.trial_sha256,
            })
            candidate_base = _wrap_v91_state(
                self.execution.base_execution, q=pending["accepted"],
                suspended_state=suspended_trial.candidate.base_state,
                receipt_chain=chain_v91)
            fluid = output["fluid_result"]
            chain = _fingerprint({
                "previous": self._parent.accepted_receipt_chain_sha256,
                "hydraulic": pending["receipt"].receipt_sha256,
                "suspended": suspended_trial.trial_sha256,
                "fluid": fluid.result_sha256,
            })
            candidate = _wrap_v92_state(
                self.execution, candidate_base,
                h_temperature_c_m=fluid.h_temperature_c_m,
                h_salinity_ppt_m=fluid.h_salinity_ppt_m,
                boundary_last_outflow_fluid=
                    fluid.boundary_last_outflow_fluid,
                boundary_return_clock_s=fluid.boundary_return_clock_s,
                reference_scalar_integral=
                    self._parent.reference_scalar_integral,
                advective_boundary_ledger=(
                    self._parent.advective_boundary_ledger+
                    fluid.advective_boundary_ledger),
                diffusive_boundary_ledger=(
                    self._parent.diffusive_boundary_ledger+
                    fluid.diffusive_boundary_ledger),
                wetdry_scalar_ledger=(
                    self._parent.wetdry_scalar_ledger+
                    fluid.wetdry_scalar_ledger),
                accepted_receipt_chain_sha256=chain)
            _require(np.allclose(
                output["endpoint_density"],
                candidate.water_density_cell_kg_m3,
                rtol=5.e-14, atol=5.e-13),
                "v92 CUDA endpoint density differs from CPU EOS oracle")
            _require(np.allclose(
                output["endpoint_viscosity"],
                candidate.molecular_viscosity_cell_m2_s,
                rtol=5.e-14, atol=5.e-16),
                "v92 CUDA endpoint viscosity differs from CPU EOS oracle")
            trial = _wrap_v92_trial(
                self.execution, self._parent,
                cap_receipt=self.base._cap,
                hydraulic_receipt=pending["receipt"],
                hydraulic_face_receipt=hydraulic_face_receipt,
                settling_diameter_receipt=settling_diameter_receipt,
                suspended_capacity_receipt=
                    output["suspended_capacity_receipt"],
                scalar_execution=scalar_execution,
                suspended_trial=suspended_trial,
                fluid_transport=fluid,
                midpoint_q=pending["midpoint"],
                momentum_rate_m2_s2=pending["rate"],
                momentum_face_flux_m4_s2=pending["flux"],
                parent_total_suspended_concentration_kg_m3=
                    output["parent_ctot"],
                settling_water_density_kg_m3=
                    self._parent.water_density_cell_kg_m3,
                settling_molecular_viscosity_m2_s=
                    self._parent.molecular_viscosity_cell_m2_s,
                candidate=candidate, backend="actual_cuda",
                wetdry_hydraulic_evidence=wetdry_hydraulic_evidence)
        except Exception:
            self.abort_hydraulic_trial()
            raise
        self._trial, self._scalar_execution = trial, scalar_execution
        return trial

    def commit(self, parent, trial):
        _require(parent is self._parent and trial is self._trial and
                 self.combined._open,
                 "v92 CUDA trial is not the resident open transaction")
        result = commit_transported_fluid_trial_v92(
            self.execution, parent, self._parent_hles,
            self._handoff, trial)
        self.combined.commit(parent.state_sha256)
        accepted = self.base.hydraulic_receipt_document(
            trial.hydraulic_receipt)
        accepted.update(
            outer_transaction_status="accepted",
            scalar_trial_sha256=trial.suspended_trial.trial_sha256,
            fluid_trial_sha256=trial.fluid_transport.result_sha256,
            accepted_state_sha256=result.state_sha256)
        self.last_accepted_receipt = accepted
        self.base.last_accepted_receipt = accepted
        self.base._clear()
        self._parent = self._parent_hles = self._handoff = None
        self._trial = self._scalar_execution = None
        return result

    def rollback(self, parent, trial):
        _require(parent is self._parent and trial is self._trial and
                 self.combined._open,
                 "v92 CUDA trial is not the resident open transaction")
        result = rollback_transported_fluid_trial_v92(
            self.execution, parent, self._parent_hles,
            self._handoff, trial)
        self.combined.rollback(parent.state_sha256)
        self.base._clear()
        self._parent = self._parent_hles = self._handoff = None
        self._trial = self._scalar_execution = None
        return result

    def cancel(self, parent, trial):
        return self.rollback(parent, trial)

    def accepted_receipt_document(self) -> dict:
        _require(self.last_accepted_receipt is not None,
                 "v92 CUDA has no committed outer receipt")
        return dict(self.last_accepted_receipt)

    def checkpoint(self, state) -> dict:
        _require(self._parent is None and not self.combined._open and
                 self.base._pending is None and not self.base._open,
                 "v92 CUDA checkpoint requires a committed point")
        return transported_fluid_checkpoint_v92(self.execution, state)

    def restore_checkpoint(self, document):
        _require(self._parent is None and not self.combined._open and
                 self.base._pending is None and not self.base._open,
                 "v92 CUDA restart cannot replace an open trial")
        cp = self.cp
        old_accepted_last = self.combined.accepted_last.copy()
        old_accepted_clock = self.combined.accepted_clock.copy()
        old_trial_last = self.combined.trial_last.copy()
        old_trial_clock = self.combined.trial_clock.copy()
        old_receipt = copy.deepcopy(self.last_accepted_receipt)
        old_base_receipt = copy.deepcopy(self.base.last_accepted_receipt)
        try:
            state = load_transported_fluid_checkpoint_v92(
                self.execution, document)
            self.combined._restore_memory(state)
            self.last_accepted_receipt = None
            self.base.last_accepted_receipt = None
            return state
        except Exception:
            cp.copyto(self.combined.accepted_last, old_accepted_last)
            cp.copyto(self.combined.accepted_clock, old_accepted_clock)
            cp.copyto(self.combined.trial_last, old_trial_last)
            cp.copyto(self.combined.trial_clock, old_trial_clock)
            self.last_accepted_receipt = old_receipt
            self.base.last_accepted_receipt = old_base_receipt
            raise


def method_identity_v92_cuda_lifecycle() -> dict:
    return {
        "cuda_composition_id": CUDA_COMPOSITION_ID,
        "integrator": "implicit_fwave_midpoint",
        "actual_cuda_kernels": True,
        "automatic_cpu_fallback": False,
        "scalar_owner": "one_concatenated_sediment_plus_active_fluid_owner",
        "inactive_fluid_lane_application_count": 0,
        "parent_pretransport_ctot_temperature_density_settling": True,
        "accepted_parent_cellwise_settling_diameter_receipt": True,
        "resident_capacity_resolver": True,
        "resolved_capacity_cpu_numerical_fallback": False,
        "physical_boundary_memory": True,
        "atomic_commit_rollback_checkpoint": True,
        "product_admission": False,
    }


__all__ = [
    "CUDA_COMPOSITION_ID",
    "ResidentSuspendedCapacityResolutionV94",
    "ResidentTransportedFluidSettlingLifecycleV92",
    "method_identity_v92_cuda_lifecycle",
]
