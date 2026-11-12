"""Authenticated resident-CUDA foundation for Q3-H3 secondary flow.

This owner is intentionally distinct from the historical GPU model-2 route.
It reuses only the accepted reconstructed conservative transport machinery and
the Q3-H2 classic-FLOW curvature/typed-roughness primitives.  Source reaction,
open-boundary ownership, dry cleanup and bed rotation are v57 kernels.  No
exact-exponential relaxation, curvature filtering, Manning-only conversion or
CPU fallback is reachable through this class.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from braids_core.sediment.prognostic_secondary_execution import (
    EXECUTION_KEY,
    GPU_CHECKPOINT_SCHEMA,
    ORDERED_ROUTE_KEY,
    validate_prognostic_secondary_execution_document,
)
from .flow2dh_secondary_operator import (
    CUDA_OPTIONS,
    ResidentClassicFlow2DHSecondary,
)


KERNEL_NAMES = (
    "flow2dh_prognostic_reaction",
    "flow2dh_prognostic_primitive",
    "flow2dh_prognostic_low_flux",
    "flow2dh_prognostic_fct_bounds",
    "flow2dh_prognostic_rotate_boundary_bed",
)


def validate_scope(data: dict) -> dict[str, object]:
    """Authenticate the bounded first Q3-H3 GPU admission before CUDA."""
    identity = validate_prognostic_secondary_execution_document(
        data.get(EXECUTION_KEY))
    route = data.get(ORDERED_ROUTE_KEY)
    if not isinstance(route, dict):
        raise ValueError("Q3-H3 ordered route identity is missing")
    expected = identity["fingerprint"]
    if (set(route) != {
            "capacity_execution_fingerprint",
            "capacity_profile_fingerprint",
            "hiding_execution_fingerprint",
            "prognostic_secondary_execution_fingerprint",
            } or route["prognostic_secondary_execution_fingerprint"] != expected or
            route["capacity_execution_fingerprint"] !=
                identity["capacity_execution_fingerprint"] or
            route["capacity_profile_fingerprint"] !=
                identity["capacity_profile_fingerprint"] or
            route["hiding_execution_fingerprint"] !=
                identity["hiding_execution_fingerprint"]):
        raise ValueError("Q3-H3 ordered route/execution identity differs")

    def scalar(name, default=None):
        if name not in data:
            if default is None:
                raise ValueError("Q3-H3 prepared input lacks " + name)
            return default
        value = np.asarray(data[name])
        if value.shape != ():
            raise ValueError("Q3-H3 control must be scalar: " + name)
        return value.item()

    if int(scalar("secondary_flow_model")) != 2:
        raise ValueError("Q3-H3 requires prognostic secondary model 2")
    if not bool(scalar("secondary_flow_transport_direction")):
        raise ValueError("Q3-H3 requires transported-state bed rotation")
    if int(scalar("secondary_flow_inflow_mode", 1)) != 1:
        raise ValueError("Q3-H3 requires classic equilibrium boundary mode 1")
    beta = float(scalar("secondary_flow_momentum_beta", 0.0))
    if not np.isfinite(beta) or beta != 0.0 or float(identity["betac"]) != 0.0:
        raise ValueError("Q3-H3 first GPU admission excludes nonzero Betac")
    if float(identity["scalar_diffusivity_m2_s"]) != 0.0:
        raise ValueError(
            "Q3-H3 first GPU admission excludes spiral-intensity diffusivity")
    if (identity["coriolis_disposition"] != "zero" or
            float(identity["coriolis_constant_s_inverse"]) != 0.0):
        raise ValueError(
            "Q3-H3 first GPU admission excludes nonzero Coriolis")
    if identity["gpu_checkpoint_schema"] != GPU_CHECKPOINT_SCHEMA:
        raise ValueError("Q3-H3 GPU checkpoint schema differs")
    if int(scalar("nfrac")) != identity["runtime_layout"]["fraction_count"]:
        raise ValueError("Q3-H3 fraction layout differs")
    if int(scalar("nbed_layers")) != identity["runtime_layout"]["bed_layer_count"]:
        raise ValueError("Q3-H3 layer layout differs")
    for key, layout_key in (
        ("secondary_index", "secondary_state_index"),
        ("secondary_primitive_index", "secondary_primitive_index"),
        ("secondary_ledger_index", "secondary_ledger_index"),
    ):
        if key in data and int(scalar(key)) != identity["runtime_layout"][layout_key]:
            raise ValueError("Q3-H3 runtime layout differs: " + key)
    return identity


class ResidentClassicFlow2DHPrognosticSecondary:
    """Transactional signed-hI owner for the first fixed-grid v57 route."""

    def __init__(self, data, reconstruction, geometry, *, source_folder,
                 transport_source_folder=None, prognostic_source_path=None,
                 transport=None):
        # Authenticate every semantic/control gate before importing CuPy,
        # creating a device context, compiling a module or allocating state.
        self.identity = validate_scope(data)
        import cupy as cp

        self.cp = cp
        self.device_id = int(cp.cuda.Device().id)
        self.nc = int(data["nc"])
        self.nf = int(data["nf"])
        self.ng = int(data["nfrac"])
        self.nb = int(len(np.asarray(data.get("boundary_offset", ()))) - 1)
        if self.nb < 0:
            self.nb = int(len(np.asarray(data.get("mode", ()))))
        self.dry = float(data["dry"])
        self.gravity = float(data["grav"])
        self.espir = float(self.identity["espir"])
        self.geometry = geometry
        self.reconstruction = reconstruction

        folder = Path(source_folder).resolve()
        transport_folder = Path(
            transport_source_folder or source_folder).resolve()
        source_path = Path(
            prognostic_source_path or folder /
            "flow2dh_prognostic_secondary.cu").resolve()
        if not source_path.is_file() or source_path.is_symlink():
            raise ValueError("Archived Q3-H3 CUDA source is missing or linked")
        self.classic = ResidentClassicFlow2DHSecondary(
            data, geometry, source_folder=folder,
            execution_identity=self.identity)
        # Reuse only its already-qualified reconstructed scalar transport
        # kernels.  Every source/boundary/rotation entry point below bypasses
        # the legacy model-2 methods and their filter/relaxation semantics.
        if transport is None:
            from validation.gpu_fwave.secondary_transport import \
                SecondaryTransport
            self.transport = SecondaryTransport(
                data, reconstruction,
                source_path=transport_folder / "secondary_transport.cu",
                curvature_source_path=
                    transport_folder / "secondary_curvature.cu")
        else:
            self.transport = transport
        if self.transport.model != 2:
            raise ValueError("Q3-H3 scalar transport owner differs")
        with cp.cuda.Device(self.device_id):
            self.module = cp.RawModule(
                code=source_path.read_text(encoding="utf-8"),
                options=CUDA_OPTIONS, name_expressions=KERNEL_NAMES)
            self.kernels = {
                name: self.module.get_function(name) for name in KERNEL_NAMES
            }
            self.trial = cp.zeros(self.nc, dtype=cp.float64)
            self.new_hydraulic = cp.zeros((3, self.nc), dtype=cp.float64)
            self.intensity = cp.zeros(self.nc, dtype=cp.float64)
            self.source_work = cp.zeros(self.nc, dtype=cp.float64)
            self.explicit_source_work = cp.zeros(self.nc, dtype=cp.float64)
            self.implicit_sink_work = cp.zeros(self.nc, dtype=cp.float64)
            self.source_ledger_cells = cp.zeros(self.nc, dtype=cp.float64)
            self.boundary_ledger = cp.zeros(self.nb, dtype=cp.float64)
            self.reaction_status = cp.zeros(self.nc, dtype=cp.int32)
            self.primitive_status = cp.zeros(self.nc, dtype=cp.int32)
            self.boundary_rotation_status = cp.zeros(self.nf, dtype=cp.int32)
        self.refresh_count = 0
        self._trial_open = False
        self._last_status = None
        self._phase = "idle"

    def _launch(self, name, count, args):
        if count:
            self.kernels[name](((count + 255) // 256,), (256,), args)

    def _fp64(self, name, value, shape):
        cp = self.cp
        if (not isinstance(value, cp.ndarray) or value.dtype != cp.float64 or
                value.shape != shape or not value.flags.c_contiguous or
                int(value.device.id) != self.device_id):
            raise ValueError(
                f"Q3-H3 {name} must be resident contiguous FP64 {shape}")
        return value

    def _active(self, value):
        cp = self.cp
        if (not isinstance(value, cp.ndarray) or value.dtype != cp.int32 or
                value.shape != (self.nc,) or not value.flags.c_contiguous or
                int(value.device.id) != self.device_id):
            raise ValueError("Q3-H3 active mask must be resident int32")
        return value

    def begin_trial(self, accepted):
        """Snapshot accepted signed hI; subsequent operations are private."""
        self._fp64("accepted signed hI", accepted, (self.nc,))
        self.cp.copyto(self.trial, accepted)
        self.source_ledger_cells.fill(0.0)
        self.boundary_ledger.fill(0.0)
        self._trial_open = True
        self._last_status = None
        self._phase = "begun"
        return self.trial

    def _require_trial(self):
        if not self._trial_open:
            raise RuntimeError("Q3-H3 transaction has not begun")

    def refresh(self, q_old, q_new, active):
        """Refresh FLOW coefficients from new velocities and old/new depths."""
        self._fp64("old hydraulic state", q_old, (3, self.nc))
        self._fp64("new hydraulic state", q_new, (3, self.nc))
        self._active(active)
        # The H2 live primitive computes raw curvature, typed raw Chezy and
        # active masks from the current (new) hydraulic state.  Re-evaluate
        # only its point formula with the distinct FLOW s0/s1 depth pair.
        status = self.classic.refresh(q_new, active)
        classic = self.classic
        classic.point.evaluate(
            classic.operator_active, q_old[0], q_new[0], classic.u,
            classic.v, classic.raw_curvature, classic.metric_ksi_sum,
            classic.metric_eta_sum, classic.coriolis, classic.raw_chezy,
            classic.zero, classic.zero, classic.zero, dryflc=self.dry,
            rmincf=classic.rmincf, chzmin=classic.chzmin,
            gravity=self.gravity, espir=self.espir, betac=0.0,
            reject_invalid=False, out=classic.result)
        self.cp.maximum(classic.status, classic.result.status,
                        out=classic.status)
        self.cp.copyto(self.new_hydraulic, q_new)
        self.refresh_count += 1
        return classic.status

    def _primitive(self, depth):
        self._launch("flow2dh_prognostic_primitive", self.nc, (
            np.int32(self.nc), depth, self.trial,
            self.classic.operator_active, np.float64(self.dry),
            self.intensity, self.primitive_status))
        return self.primitive_status

    def _reaction(self, q_old, q_new, dt, active, *, refresh_coefficients):
        self._require_trial()
        if not np.isfinite(dt) or dt < 0.0:
            raise ValueError("Q3-H3 reaction timestep must be nonnegative")
        foundation = (self.refresh(q_old, q_new, active)
                      if refresh_coefficients else self.classic.status)
        result = self.classic.result
        self._launch("flow2dh_prognostic_reaction", self.nc, (
            np.int32(self.nc), q_old[0], q_new[0],
            self.classic.operator_active,
            result.source_coefficient, result.sink_coefficient,
            self.transport.inputs["area"], np.float64(self.dry),
            np.float64(dt), self.trial, self.intensity, self.source_work,
            self.explicit_source_work, self.implicit_sink_work,
            self.reaction_status))
        self.source_ledger_cells += self.source_work
        status = self.cp.maximum(self.cp.max(foundation),
                                 self.cp.max(self.reaction_status))
        self._last_status = status
        self._phase = "reacted"
        return {
            "state": self.trial,
            "intensity": self.intensity,
            "source_integral": self.cp.sum(self.source_work),
            "explicit_source_integral": self.cp.sum(
                self.explicit_source_work),
            "implicit_sink_integral": self.cp.sum(
                self.implicit_sink_work),
            "status": status,
            "refresh_generation": self.refresh_count,
        }

    def reaction(self, q_old, q_new, dt, active):
        """Apply classic explicit-source/implicit-sink reaction in place."""
        return self._reaction(
            q_old, q_new, dt, active, refresh_coefficients=True)

    def reconstruct(self, q, bed, active, *, hydraulic_buffers=None):
        """Reconstruct transported I without invoking the legacy filter."""
        self._require_trial()
        self._fp64("hydraulic state", q, (3, self.nc))
        self._fp64("bed", bed, (self.nc,))
        self._active(active)
        b, a = self.transport.buffers, self.transport.inputs
        hydro = self.transport._hydro(hydraulic_buffers)
        primitive = self._primitive(q[0])
        args = (
            np.int32(self.nc), np.int32(self.nf),
            np.int32(self.transport.nadj), q, bed, self.intensity,
            a["xy"], a["mid"], a["normal"], a["owner"], a["neighbour"],
            a["bid"], a["boundary_mode"], a["offset"], a["faces"],
            a["signs"], hydro["active"], a["hydraulic_ls_matrix"],
            a["hydraulic_ls_weight"], a["bed_ls_delta"],
            np.float64(self.dry), np.float64(self.gravity),
        )
        if self.transport.outlet_dispatch:
            args += (np.int32(self.transport.outlet_extension),)
        self.transport.launch(
            "reconstruct_cells", self.nc, args + (b["gradient"],))
        self.transport.launch("reconstruct_faces", self.nf, (
            np.int32(self.nc), np.int32(self.nf), self.intensity,
            b["gradient"], a["owner"], a["neighbour"], a["weno_cells"],
            a["face_displacement"], a["wall_stencil_ids"], a["wall_slot"],
            hydro["wall_ready"], b["face_intensity"],
            b["reconstruction_status"]))
        status = self.cp.maximum(self.cp.max(primitive),
                                 self.cp.max(b["reconstruction_status"]))
        self._last_status = status
        return {"intensity": self.intensity,
                "face_intensity": b["face_intensity"], "status": status}

    def advect(self, q, next_q, bed, water_flux, dt, active, *,
               hydraulic_buffers=None):
        """Advance signed hI with equilibrium on every open face."""
        self._require_trial()
        self._fp64("hydraulic state", q, (3, self.nc))
        self._fp64("next hydraulic state", next_q, (3, self.nc))
        self._fp64("bed", bed, (self.nc,))
        self._active(active)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("Q3-H3 advection timestep must be positive")
        if getattr(water_flux, "shape", None) == (3, self.nf):
            water_flux = water_flux[0]
        self._fp64("water flux", water_flux, (self.nf,))
        foundation = self.refresh(q, next_q, active)
        reconstructed = self.reconstruct(
            q, bed, active, hydraulic_buffers=hydraulic_buffers)
        a, b = self.transport.inputs, self.transport.buffers
        hydro = self.transport._hydro(hydraulic_buffers)
        self._launch("flow2dh_prognostic_low_flux", self.nf, (
            np.int32(self.nc), np.int32(self.nf), self.intensity,
            b["face_intensity"], self.classic.result.boundary_equilibrium,
            self.classic.result.boundary_defined, a["owner"], a["neighbour"],
            water_flux, b["low_flux"], b["anti_flux"],
            b["boundary_intensity"], b["flux_status"]))
        self._launch("flow2dh_prognostic_fct_bounds", self.nc, (
            np.int32(self.nc), np.int32(self.nf), self.intensity, self.trial,
            next_q[0], a["area"], a["length"], a["owner"], a["neighbour"],
            a["bid"], a["offset"], a["faces"], a["signs"], hydro["active"],
            b["low_flux"], b["anti_flux"], b["boundary_intensity"],
            np.float64(dt), b["lowstate"], b["plus"], b["minus"],
            b["fct_status"]))
        self.transport.launch("fct_faces", self.nf, (
            np.int32(self.nf), a["owner"], a["neighbour"], b["low_flux"],
            b["plus"], b["minus"], b["anti_flux"], b["flux"]))
        self.transport.launch("fct_update", self.nc, (
            np.int32(self.nc), next_q, self.trial, b["lowstate"], a["area"],
            a["length"], a["offset"], a["faces"], a["signs"],
            hydro["active"], b["anti_flux"], np.float64(dt), b["next"],
            b["fct_status"]))
        if self.nb:
            self.transport.kernels["ledger"]((self.nb,), (128,), (
                np.int32(self.nb), a["boundary_offset"], a["boundary_faces"],
                a["length"], b["flux"], np.float64(dt), b["ledger"]))
        self.cp.copyto(self.trial, b["next"])
        self.boundary_ledger += b["ledger"]
        status = self.cp.maximum(
            self.cp.max(foundation),
            self.cp.maximum(reconstructed["status"],
                            self.cp.maximum(self.cp.max(b["fct_status"]),
                                            self.cp.max(b["flux_status"]))))
        self._last_status = status
        self._phase = "advected"
        return {"state": self.trial, "flux": b["flux"],
                "ledger": b["ledger"], "status": status,
                "refresh_generation": self.refresh_count}

    def advance(self, q_old, q_new, bed, water_flux, dt, active, *,
                hydraulic_buffers=None):
        """Execute the sole admitted v57 hI update order.

        The accepted inventory is first transported with the converged
        hydraulic water flux.  FLOW's one full-step explicit-source/implicit-
        sink reaction then consumes the accepted old depth and post-hydraulic
        new depth/velocity/curvature/Chezy.  This API intentionally has no
        half-source option.
        """
        self._require_trial()
        if self._phase != "begun":
            raise RuntimeError("Q3-H3 advance requires a fresh transaction")
        advected = self.advect(
            q_old, q_new, bed, water_flux, dt, active,
            hydraulic_buffers=hydraulic_buffers)
        advective_status = advected["status"]
        # Advection has just refreshed the exact same old/new FLOW pair for
        # its sign-independent secbou trace.  Reuse those coefficients for the
        # immediately following reaction rather than repeating curvature and
        # typed-roughness work on every accepted trial.
        reacted = self._reaction(
            q_old, q_new, dt, active, refresh_coefficients=False)
        status = self.cp.maximum(advective_status, reacted["status"])
        self._last_status = status
        self._phase = "reacted"
        return {
            "state": self.trial,
            "flux": advected["flux"],
            "boundary_ledger": advected["ledger"],
            "source_integral": reacted["source_integral"],
            "explicit_source_integral":
                reacted["explicit_source_integral"],
            "implicit_sink_integral": reacted["implicit_sink_integral"],
            "status": status,
            "refresh_generation": self.refresh_count,
        }

    def rotate_cells(self, q_capacity, capacity_velocity, capacity_chezy,
                     raw_bed, active):
        """Rotate raw bed capacity once at its weak-midpoint forcing state."""
        self._require_trial()
        if self._phase != "reacted":
            raise RuntimeError(
                "Q3-H3 bed rotation requires the completed v57 update")
        self._fp64("capacity hydraulic state", q_capacity, (3, self.nc))
        self._fp64(
            "capacity velocity", capacity_velocity, (2, self.nc))
        self._fp64("capacity Chezy", capacity_chezy, (self.nc,))
        self._fp64("raw bed capacity", raw_bed, (2, self.ng, self.nc))
        self._active(active)
        # ``advance``/``reaction`` owns endpoint coefficients and open-face
        # equilibrium.  Capacity remains a weak-midpoint operator, so its raw
        # magnitude and direction must consume the exact same midpoint u/v,
        # typed Chezy and depth.  Never refresh or overwrite the endpoint FLOW
        # target/boundary pair here.
        foundation = self.classic.status
        primitive = self._primitive(q_capacity[0])
        self.classic._launch("flow2dh_rotate_cell_bed", self.nc, (
            np.int32(self.nc), np.int32(self.ng), capacity_velocity[0],
            capacity_velocity[1], capacity_chezy, self.intensity,
            self.classic.operator_active, np.float64(self.espir),
            np.float64(self.gravity), raw_bed,
            self.classic.cell_rotation_status))
        status = self.cp.maximum(
            self.cp.max(foundation),
            self.cp.maximum(self.cp.max(primitive),
                            self.cp.max(self.classic.cell_rotation_status)))
        self._last_status = status
        self._phase = "rotated"
        return status

    def rotate_boundaries(self, trace, owner, role, raw_boundary_bed,
                          rotated_cell_bed):
        """Rotate trace roles 1/2; copy once-rotated owner capacity for role 3."""
        self._require_trial()
        cp = self.cp
        self._fp64("boundary trace", trace, (3, self.nf))
        self._fp64(
            "raw boundary bed capacity", raw_boundary_bed,
            (2, self.ng, self.nf))
        self._fp64(
            "rotated cell bed capacity", rotated_cell_bed,
            (2, self.ng, self.nc))
        for name, value in (("owner", owner), ("role", role)):
            if (not isinstance(value, cp.ndarray) or value.dtype != cp.int32 or
                    value.shape != (self.nf,) or not value.flags.c_contiguous or
                    int(value.device.id) != self.device_id):
                raise ValueError("Q3-H3 boundary " + name + " differs")
        classic = self.classic
        if self.nf:
            classic.roughness.kernels["hydraulic_roughness_state"](
                ((self.nf + 255) // 256,), (256,), (
                    np.int64(self.nf), np.int32(classic.roughness_mode),
                    trace[0], classic.boundary_roughness_parameter,
                    np.float64(self.gravity),
                    np.float64(classic.roughness_work_depth),
                    np.int32(classic.roughness_waqua),
                    classic.boundary_raw_chezy,
                    classic.boundary_roughness_z0,
                    classic.boundary_roughness_drag,
                    classic.boundary_roughness_status))
        self._launch("flow2dh_prognostic_rotate_boundary_bed", self.nf, (
            np.int32(self.nc), np.int32(self.nf), np.int32(self.ng), trace,
            owner, role, classic.boundary_raw_chezy,
            classic.boundary_roughness_status,
            classic.result.boundary_equilibrium, np.float64(self.dry),
            np.float64(self.espir), np.float64(self.gravity),
            rotated_cell_bed, raw_boundary_bed,
            self.boundary_rotation_status))
        status = cp.maximum(cp.max(classic.boundary_roughness_status),
                            cp.max(self.boundary_rotation_status))
        self._last_status = status
        return status

    def finish_trial(self, accepted, *, commit):
        """Commit only an explicitly accepted transaction; otherwise rollback."""
        self._require_trial()
        self._fp64("accepted signed hI", accepted, (self.nc,))
        if not isinstance(commit, (bool, np.bool_)):
            raise TypeError("Q3-H3 commit flag must be boolean")
        if bool(commit):
            if self._last_status is None or int(self._last_status.item()) != 0:
                raise RuntimeError("Cannot commit an invalid Q3-H3 transaction")
            self.cp.copyto(accepted, self.trial)
        self._trial_open = False
        self._phase = "idle"
        return accepted

    def abort_trial(self):
        """Close a private trial without touching the accepted signed hI."""
        if self._trial_open:
            self._trial_open = False
            self._phase = "idle"
        return None

    def storage_identity(self):
        values = (
            self.trial, self.new_hydraulic, self.intensity, self.source_work,
            self.explicit_source_work, self.implicit_sink_work,
            self.source_ledger_cells, self.boundary_ledger,
            self.reaction_status, self.primitive_status,
            self.boundary_rotation_status,
        )
        return tuple((id(value), int(value.data.ptr), value.shape,
                      value.dtype.str, value.strides, int(value.device.id))
                     for value in values)

    def diagnostic(self):
        return {
            "execution_fingerprint": self.identity["fingerprint"],
            "checkpoint_schema": GPU_CHECKPOINT_SCHEMA,
            "state_semantics": self.identity["state_semantics"],
            "integrated_ledger_units":
                self.identity["integrated_ledger_units"],
            "refresh_count": self.refresh_count,
            "source_ledger_cells": self.source_ledger_cells,
            "boundary_ledger": self.boundary_ledger,
            "classic": self.classic.diagnostic(),
        }


__all__ = [
    "KERNEL_NAMES", "ResidentClassicFlow2DHPrognosticSecondary",
    "validate_scope",
]
