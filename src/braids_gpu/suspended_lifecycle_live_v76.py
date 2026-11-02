"""Thin resident-CUDA transaction owner for Q4 suspension lifecycle v76.

No numerical kernel is duplicated here.  The adapter calls the qualified
resident accepted-flux FCT, FLOW boundary/diffusion, and law-11 Galappatti
operators, then publishes one host-authenticated immutable candidate.  CUDA
is an explicit backend choice and there is no CPU fallback.
"""
from __future__ import annotations

import math

import numpy as np

from braids_core.sediment.suspended_lifecycle_live_v76 import (
    BACKEND_CUDA,
    MAX_ACCEPTED_UPDATES,
    SuspendedLifecycleExecutionV76,
    SuspendedLifecycleStateV76,
    SuspendedLifecycleTrialV76,
    SuspendedLifecycleV76Error,
    _freeze_trial_array,
    _ledger_residuals,
    _trial_token,
    _validate_trial,
    load_suspended_lifecycle_checkpoint_v76,
    resolve_suspended_lifecycle_state_v76,
    suspended_lifecycle_checkpoint_v76,
    validate_suspended_lifecycle_state_v76,
)
from braids_gpu.flow2dh_boundary_diffusion import (
    ResidentFlow2DHBoundaryDiffusion,
)
from braids_gpu.law11_soursin_exchange import (
    ResidentLaw11SoursinExchange,
)


CUDA_THREADS_PER_BLOCK = 128
CUDA_COMPOSITION_ID = (
    "resident_existing_fct_boundary_diffusion_settlefrm2_galappatti_v76_v1"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SuspendedLifecycleV76Error(message)


class ResidentSuspendedLifecycleV76:
    """One explicitly selected CUDA lifecycle, initially one sand class.

    The authenticated resident law-11 exchange currently admits one fraction;
    the CPU v76 lifecycle remains class-general.  This limitation is rejected
    at construction rather than silently moving any physics to the CPU.
    """

    def __init__(
        self,
        execution: SuspendedLifecycleExecutionV76,
        boundary_diffusion: ResidentFlow2DHBoundaryDiffusion,
        exchange: ResidentLaw11SoursinExchange,
    ):
        import cupy as cp

        _require(isinstance(execution, SuspendedLifecycleExecutionV76) and
                 execution.backend == BACKEND_CUDA,
                 "v76 resident CUDA owner requires an explicit CUDA execution")
        _require(isinstance(boundary_diffusion,
                            ResidentFlow2DHBoundaryDiffusion),
                 "v76 resident CUDA boundary/diffusion owner differs")
        _require(isinstance(exchange, ResidentLaw11SoursinExchange),
                 "v76 resident CUDA Galappatti owner differs")
        _require(execution.fraction_count == boundary_diffusion.ng ==
                 exchange.ng == 1,
                 "v76 CUDA currently requires one authenticated sand class")
        _require(execution.cell_count == boundary_diffusion.nc == exchange.nc and
                 execution.face_count == boundary_diffusion.nf and
                 execution.boundary_count == boundary_diffusion.nb,
                 "v76 CUDA component topology dimensions differ")
        _require(boundary_diffusion.profile.fingerprint ==
                 execution.profile.fingerprint,
                 "v76 CUDA component/profile fingerprint differs")
        geometry = boundary_diffusion.advection.geometry
        pairs = (
            ("area", execution.cell_area_m2),
            ("length", execution.face_length_m),
            ("normal", execution.normal),
            ("rectangle_width", execution.rectangle_width_m),
            ("weno_cells", execution.weno_cells),
            ("owner", execution.owner),
            ("neighbour", execution.neighbour),
            ("bid", execution.boundary_index),
            ("offset", execution.cell_face_offset),
            ("faces", execution.cell_faces),
            ("signs", execution.cell_face_sign),
        )
        for key, expected in pairs:
            actual = cp.asnumpy(geometry[key])
            _require(np.array_equal(actual, expected),
                     "v76 CUDA component topology differs: " + key)
        boundary_aliases = (
            (boundary_diffusion.owner, execution.owner, "owner"),
            (boundary_diffusion.neighbour, execution.neighbour, "neighbour"),
            (boundary_diffusion.bid, execution.boundary_index, "boundary ID"),
            (boundary_diffusion.offset, execution.cell_face_offset, "offset"),
            (boundary_diffusion.csr_faces, execution.cell_faces, "CSR faces"),
            (boundary_diffusion.signs, execution.cell_face_sign, "CSR signs"),
            (boundary_diffusion.length, execution.face_length_m, "face length"),
            (boundary_diffusion.area, execution.cell_area_m2, "cell area"),
        )
        for actual, expected, name in boundary_aliases:
            _require(np.array_equal(cp.asnumpy(actual), expected),
                     "v76 CUDA boundary/diffusion topology differs: " + name)
        lists = [np.flatnonzero(execution.boundary_index == boundary_id).astype(
            np.int32) for boundary_id in range(execution.boundary_count)]
        expected_boundary_faces = (np.concatenate(lists) if lists else
                                   np.empty(0, dtype=np.int32))
        expected_boundary_offset = np.r_[
            0, np.cumsum([len(value) for value in lists])].astype(np.int32)
        for source, name in (
            (geometry, "advection"),
            ({"boundary_faces": boundary_diffusion.boundary_faces,
              "boundary_offset": boundary_diffusion.boundary_offset},
             "boundary/diffusion"),
        ):
            _require(np.array_equal(cp.asnumpy(source["boundary_faces"]),
                                    expected_boundary_faces) and
                     np.array_equal(cp.asnumpy(source["boundary_offset"]),
                                    expected_boundary_offset),
                     "v76 CUDA " + name + " boundary-face CSR differs")
        _require(np.array_equal(
            cp.asnumpy(boundary_diffusion.centre_distance),
            execution.centre_distance_m),
            "v76 CUDA diffusion centre distance differs")
        _require(advection_dry_matches := (
            boundary_diffusion.advection.dry == execution.dry_tolerance_m and
            boundary_diffusion.advection_dry == execution.dry_tolerance_m),
            "v76 CUDA advection dry tolerance differs")
        _require(advection_dry_matches and boundary_diffusion.dry ==
                 execution.profile.diffusion_dry_threshold_m,
                 "v76 CUDA diffusion dry threshold differs")
        expected_neumann = np.asarray(
            execution.profile.neumann_boundary_by_fraction, dtype=np.int32)
        _require(np.array_equal(boundary_diffusion.prescribed_neumann,
                                expected_neumann) and
                 np.array_equal(cp.asnumpy(boundary_diffusion.neumann),
                                expected_neumann),
                 "v76 CUDA boundary Neumann controls differ")
        _require(np.array_equal(
            cp.asnumpy(boundary_diffusion.dicuv),
            np.full(execution.cell_count,
                    execution.profile.horizontal_diffusivity_m2_s)),
            "v76 CUDA diffusion coefficient differs")
        expected_return = np.zeros(
            (execution.fraction_count, execution.face_count), dtype=np.float64)
        open_face = execution.boundary_index >= 0
        if execution.boundary_count:
            expected_return[:, open_face] = execution.boundary_return_time_s[
                :, execution.boundary_index[open_face]]
        _require(np.array_equal(
            boundary_diffusion.return_time_host, expected_return) and
            np.array_equal(cp.asnumpy(boundary_diffusion.return_time),
                           expected_return),
            "v76 CUDA boundary return-time configuration differs")
        profile = execution.profile
        scalar_controls = (
            (exchange.rho, profile.water_density_kg_m3, "water density"),
            (exchange.dry, execution.dry_tolerance_m, "exchange dry tolerance"),
            (exchange.factsd, profile.fac_tsd, "FacTsd"),
            (exchange.temperature, profile.temperature_value_c,
             "settling temperature"),
            (exchange.reference_concentration,
             profile.settling_reference_concentration_kg_m3,
             "settling reference concentration"),
            (exchange.suspended_calibration,
             profile.suspended_calibration_factor,
             "suspended calibration"),
            (exchange.capacity_speed_epsilon, 1.0e-6,
             "capacity speed epsilon"),
            (exchange.gravity, profile.gravity_m_s2, "gravity"),
            (exchange.reference_depth_floor, profile.reference_depth_floor_m,
             "reference depth floor"),
        )
        for actual, expected, name in scalar_controls:
            _require(float(actual) == float(expected),
                     "v76 CUDA exchange control differs: " + name)
        vector_controls = (
            (exchange.sediment_d50, profile.sediment_d50_m, "SedD50"),
            (exchange.settling_fac_dss, profile.settling_fac_dss,
             "settling FacDss"),
            (exchange.sediment_density, profile.sediment_density_kg_m3,
             "sediment density"),
            (exchange.sediment_threshold,
             [profile.sediment_threshold_m], "SedThr"),
        )
        for actual, expected, name in vector_controls:
            _require(np.array_equal(cp.asnumpy(actual),
                                    np.asarray(expected, dtype=np.float64)),
                     "v76 CUDA exchange control differs: " + name)
        self.cp = cp
        self.execution = execution
        self.boundary = boundary_diffusion
        self.exchange = exchange
        self._open_parent = None
        self._open_trial = None

    def _device(self, value, shape, dtype, name):
        cp = self.cp
        _require(isinstance(value, cp.ndarray) and value.shape == shape and
                 value.dtype == dtype and value.flags.c_contiguous,
                 f"v76 CUDA {name} must be contiguous device {dtype} {shape}")
        return value

    def _restore_parent_memory(self, parent: SuspendedLifecycleStateV76) -> None:
        self.boundary.restore_checkpoint({
            "flow2dh_boundary_last_outflow_concentration":
                np.array(parent.boundary_last_outflow_concentration_kg_m3,
                         copy=True),
            "flow2dh_boundary_return_clock":
                np.array(parent.boundary_return_clock_s, copy=True),
        })

    def begin_trial(
        self,
        parent: SuspendedLifecycleStateV76,
        *,
        depth_before_m,
        depth_after_m,
        accepted_water_flux_m2_s,
        reconstructed_left_concentration_kg_m3,
        reconstructed_right_concentration_kg_m3,
        prescribed_boundary_concentration_kg_m3,
        accepted_face_exchange_area_m2,
        transport_face_active,
        endpoint_momentum_m2_s,
        shear_velocity_m_s,
        suspended_transport_capacity_kg_m_s,
        surface_fraction,
        hydro_constant,
        dt_s: float,
        cancelled: bool = False,
    ) -> SuspendedLifecycleTrialV76:
        """Execute one complete private CUDA trial and publish host evidence."""
        cp = self.cp
        execution = self.execution
        validate_suspended_lifecycle_state_v76(parent, execution)
        _require(self._open_trial is None,
                 "v76 CUDA previous trial is unfinished")
        if cancelled:
            raise SuspendedLifecycleV76Error(
                "v76 CUDA update cancelled before candidate publication")
        if parent.accepted_updates >= MAX_ACCEPTED_UPDATES:
            raise SuspendedLifecycleV76Error(
                "v76 CUDA accepted update limit was reached")
        _require(not isinstance(dt_s, bool) and isinstance(dt_s, (int, float)) and
                 math.isfinite(float(dt_s)) and float(dt_s) > 0.0,
                 "v76 CUDA timestep must be finite and positive")
        dt = float(dt_s)
        ng, nc, nf, nb = (execution.fraction_count, execution.cell_count,
                          execution.face_count, execution.boundary_count)
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
        finite_inputs = (
            (h0, "depth before"), (h1, "depth after"),
            (water, "accepted water flux"),
            (left, "left reconstructed concentration"),
            (right, "right reconstructed concentration"),
            (prescribed, "prescribed boundary concentration"),
            (face_area, "accepted face exchange area"),
            (momentum, "endpoint momentum"), (shear, "shear velocity"),
            (capacity, "suspended transport capacity"),
            (fraction, "surface fraction"),
        )
        for value, name in finite_inputs:
            _require(bool(cp.all(cp.isfinite(value)).item()),
                     "v76 CUDA input must be finite: " + name)
        _require(bool(cp.all((h0 >= 0.0) & (h1 >= 0.0)).item()),
                 "v76 CUDA depths must be nonnegative")
        _require(bool(cp.all(prescribed >= 0.0).item()),
                 "v76 CUDA prescribed boundary concentration must be nonnegative")
        _require(bool(cp.all(face_area >= 0.0).item()),
                 "v76 CUDA accepted face exchange area must be nonnegative")
        _require(bool(cp.all(shear >= 0.0).item()),
                 "v76 CUDA shear velocity must be nonnegative")
        _require(bool(cp.all(capacity >= 0.0).item()),
                 "v76 CUDA suspended transport capacity must be nonnegative")
        _require(bool(cp.all((fraction >= 0.0) & (fraction <= 1.0)).item()),
                 "v76 CUDA surface fraction must lie in [0, 1]")
        _require(bool(cp.all((active_face == 0) | (active_face == 1)).item()),
                 "v76 CUDA transport face activity must be a 0/1 mask")
        _require(bool(cp.all((constant == 0) | (constant == 1)).item()),
                 "v76 CUDA hydro constant must be a 0/1 mask")
        closed = ((execution.neighbour < 0) &
                  (execution.boundary_index < 0))
        if np.any(closed):
            _require(not bool(cp.any(active_face[cp.asarray(closed)] != 0).item()),
                     "v76 CUDA closed face cannot be diffusion-active")
        self._restore_parent_memory(parent)
        accepted_suspended = cp.asarray(parent.suspended_mass_kg_m2)
        accepted_bed = cp.asarray(parent.bed_mass_kg_m2)

        boundary = self.boundary.resolve_boundary(
            h0, accepted_suspended, prescribed, water, dt)
        advected = self.boundary.advect(
            h0, accepted_suspended, h1, water, left, right, constant, dt)
        if (any(bool(cp.any(value != 0).item()) for value in
                (*boundary["status"], *advected["status"]))):
            raise SuspendedLifecycleV76Error(
                "v76 CUDA boundary/FCT kernel status rejected the trial")
        expected = boundary["expected_advective_ledger"]
        actual = advected["ledger"]
        scale = cp.maximum(1.0, cp.maximum(cp.abs(expected), cp.abs(actual)))
        if bool(cp.any(cp.abs(expected-actual) >
                       8192.0*np.finfo(float).eps*scale).item()):
            raise SuspendedLifecycleV76Error(
                "v76 CUDA FCT boundary ledger differs from resolved boundary")
        diffused = self.boundary.diffuse(
            h1, advected["next_mass"], face_area, active_face, dt)
        if any(bool(cp.any(value != 0).item()) for value in diffused["status"]):
            raise SuspendedLifecycleV76Error(
                "v76 CUDA diffusion kernel status rejected the trial")

        wet = (h1 > execution.dry_tolerance_m).astype(cp.int32)
        speed = cp.where(wet != 0, cp.hypot(momentum[0], momentum[1]) /
                         cp.where(wet != 0, h1, 1.0), 0.0)
        sediment_active = ((wet != 0) &
                           (h1 > execution.profile.sediment_threshold_m))
        safe_h1 = cp.where(
            sediment_active,
            cp.maximum(execution.profile.reference_depth_floor_m, h1), 1.0)
        equilibrium_capacity = capacity/(speed[None, :]+1.0e-6)/safe_h1[None, :]
        total_shear = execution.profile.water_density_kg_m3*shear*shear
        exchanged = self.exchange.step(
            h0, h1, momentum, diffused["next_mass"], accepted_bed,
            cp.ascontiguousarray(equilibrium_capacity), fraction,
            cp.ascontiguousarray(total_shear), cp.ascontiguousarray(wet), dt)
        if bool(cp.any(exchanged["status"] != 0).item()):
            raise SuspendedLifecycleV76Error(
                "v76 CUDA SettleFrm=2/Galappatti kernel rejected the trial")

        cleanup = cp.where(wet[None, :] != 0, 0.0, exchanged["suspended"])
        final_suspended = exchanged["suspended"]-cleanup
        final_bed = exchanged["active"]+cleanup
        area = self.boundary.area[None, :]
        exchange_water = cp.sum(
            (exchanged["suspended"]-diffused["next_mass"])*area, axis=1)
        exchange_bed = cp.sum(
            (exchanged["active"]-accepted_bed)*area, axis=1)
        cleanup_water = -cp.sum(cleanup*area, axis=1)
        cleanup_bed = -cleanup_water
        candidate = resolve_suspended_lifecycle_state_v76(
            execution,
            suspended_mass_kg_m2=cp.asnumpy(final_suspended),
            bed_mass_kg_m2=cp.asnumpy(final_bed),
            boundary_last_outflow_concentration_kg_m3=
                cp.asnumpy(self.boundary.trial_last),
            boundary_return_clock_s=cp.asnumpy(self.boundary.trial_clock),
            advective_boundary_ledger_kg=(
                parent.advective_boundary_ledger_kg+cp.asnumpy(actual)),
            diffusive_boundary_ledger_kg=(
                parent.diffusive_boundary_ledger_kg+
                cp.asnumpy(diffused["ledger"])),
            water_exchange_ledger_kg=(
                parent.water_exchange_ledger_kg+cp.asnumpy(exchange_water)),
            bed_exchange_ledger_kg=(
                parent.bed_exchange_ledger_kg+cp.asnumpy(exchange_bed)),
            wetdry_water_ledger_kg=(
                parent.wetdry_water_ledger_kg+cp.asnumpy(cleanup_water)),
            wetdry_bed_ledger_kg=(
                parent.wetdry_bed_ledger_kg+cp.asnumpy(cleanup_bed)),
            reference_water_mass_kg=parent.reference_water_mass_kg,
            reference_bed_mass_kg=parent.reference_bed_mass_kg,
            accepted_updates=parent.accepted_updates+1,
            time_s=parent.time_s+dt,
        )
        residuals = _ledger_residuals(execution, candidate)
        arrays = tuple(_freeze_trial_array(value) for value in (
            cp.asnumpy(boundary["concentration"]),
            cp.asnumpy(advected["flux"]),
            cp.asnumpy(diffused["face_mass_owner_out"]),
            cp.asnumpy(exchanged["settling_velocity"]),
            cp.asnumpy(exchanged["equilibrium_concentration"]),
            cp.asnumpy(exchanged["delta"]), cp.asnumpy(cleanup), *residuals,
        ))
        placeholder = SuspendedLifecycleTrialV76(
            execution.fingerprint, parent.state_sha256, candidate, *arrays,
            CUDA_THREADS_PER_BLOCK, BACKEND_CUDA, "")
        trial = SuspendedLifecycleTrialV76(
            **{**placeholder.__dict__,
               "trial_sha256": _trial_token(placeholder)})
        self._open_parent = parent.state_sha256
        self._open_trial = trial.trial_sha256
        return trial

    def _finish(self, parent: SuspendedLifecycleStateV76,
                trial: SuspendedLifecycleTrialV76, *, commit: bool):
        _validate_trial(self.execution, parent, trial)
        _require(self._open_parent == parent.state_sha256 and
                 self._open_trial == trial.trial_sha256,
                 "v76 CUDA trial is not the resident open transaction")
        if commit:
            self.boundary.commit_boundary_memory()
        self._open_parent = None
        self._open_trial = None
        return trial.candidate if commit else parent

    def commit(self, parent: SuspendedLifecycleStateV76,
               trial: SuspendedLifecycleTrialV76) -> SuspendedLifecycleStateV76:
        return self._finish(parent, trial, commit=True)

    def rollback(self, parent: SuspendedLifecycleStateV76,
                 trial: SuspendedLifecycleTrialV76) -> SuspendedLifecycleStateV76:
        return self._finish(parent, trial, commit=False)

    def cancel(self, parent: SuspendedLifecycleStateV76,
               trial: SuspendedLifecycleTrialV76) -> SuspendedLifecycleStateV76:
        return self.rollback(parent, trial)

    def checkpoint(self, state: SuspendedLifecycleStateV76) -> dict:
        _require(self._open_trial is None,
                 "v76 CUDA checkpoint requires a committed point")
        return suspended_lifecycle_checkpoint_v76(self.execution, state)

    def restore_checkpoint(self, document) -> SuspendedLifecycleStateV76:
        _require(self._open_trial is None,
                 "v76 CUDA restart cannot replace an open trial")
        state = load_suspended_lifecycle_checkpoint_v76(
            self.execution, document)
        self._restore_parent_memory(state)
        return state


def method_identity_v76_cuda() -> dict:
    return {
        "cuda_composition_id": CUDA_COMPOSITION_ID,
        "threads_per_block": CUDA_THREADS_PER_BLOCK,
        "automatic_cpu_fallback": False,
        "resident_kernels": [
            "existing suspended_advection.cu FCT kernels",
            "existing flow2dh_boundary_diffusion.cu kernels",
            "existing law11_soursin_exchange.cu kernel",
        ],
        "current_fraction_scope": "one authenticated noncohesive sand class",
        "host_readback_scope": "candidate authentication/checkpoint evidence",
    }


__all__ = [
    "CUDA_COMPOSITION_ID", "CUDA_THREADS_PER_BLOCK",
    "ResidentSuspendedLifecycleV76", "method_identity_v76_cuda",
]
