"""Additive resident-CUDA Q3-H3/v58 nonzero secondary-flow component.

This module does not replace or edit the accepted v57 implementation.  It is
the bounded fixed-Cartesian follow-on for three nonzero operators documented in
``validation/sediment/delft3d_current/q3h_integration/h3_nonzero``:

* one signed Coriolis constant shared by hydraulics and ``secrhs``;
* accepted-start-lagged Cartesian ``vihsec`` face fluxes; and
* conservative backward-Euler diffusion of signed spiral intensity.

The backend and integrator are explicit user choices.  Every array remains on
the selected CUDA device until the final status/diagnostic reduction, and a
failed or cancelled transaction never mutates the accepted arrays.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from braids_core.sediment.prognostic_secondary_execution import (
    EXECUTION_KEY,
    ORDERED_ROUTE_KEY,
    validate_prognostic_secondary_execution_document,
)
from braids_core.sediment.prognostic_secondary_nonzero_execution import (
    EXECUTION_KEY as NONZERO_EXECUTION_KEY,
    ORDERED_ROUTE_KEY as NONZERO_ORDERED_ROUTE_KEY,
    validate_prognostic_secondary_nonzero_execution_document,
    validate_prognostic_secondary_nonzero_ordered_route,
)
from .flow2dh_secondary_operator import CUDA_OPTIONS, Flow2DHSecondaryCUDA


ENVELOPE_VERSION = 58
CHECKPOINT_SCHEMA = 9
SCOPE_ID = "q3h3-classic-flow2dh-prognostic-nonzero-gpu-v1"
INTEGRATOR = "implicit_fwave_midpoint"
SCHMIDT_NUMBER = 0.7
DIFFUSION_RTOL = 1.0e-13
DIFFUSION_ATOL = 1.0e-14
DIFFUSION_MAX_ITERATIONS = 256
STATUS_OK = 0
STATUS_INVALID_INPUT = 1
STATUS_INVALID_GEOMETRY = 2
STATUS_INVALID_LINEAR_SYSTEM = 3
STATUS_NOT_CONVERGED = 4
STATUS_CANCELLED = 5
STATUS_INVALID_OUTPUT = 6

KERNEL_NAMES = (
    "v58_constant_coriolis",
    "v58_betac_cells",
    "v58_betac_faces",
    "v58_add_frozen_stress",
    "v58_stress_boundary_rate",
    "v58_diffusion_conductance",
    "v58_diffusion_setup",
    "v58_diffusion_matvec",
    "v58_pcg_initialize_vectors",
    "v58_pcg_initialize_control",
    "v58_pcg_alpha",
    "v58_pcg_update_xr",
    "v58_pcg_precondition",
    "v58_pcg_convergence_beta",
    "v58_pcg_update_p",
    "v58_diffusion_finish",
    "v58_diffusion_boundary_ledger",
)


def _scalar(data: dict, name: str, default=None):
    if name not in data:
        if default is None:
            raise ValueError("Q3-H3/v58 prepared input lacks " + name)
        return default
    value = np.asarray(data[name])
    if value.shape != ():
        raise ValueError("Q3-H3/v58 control must be scalar: " + name)
    return value.item()


def validate_scope(data: dict) -> dict[str, object]:
    """Authenticate the isolated v58 route before importing/initialising CUDA."""
    parent = validate_prognostic_secondary_execution_document(
        data.get(EXECUTION_KEY))
    route = data.get(ORDERED_ROUTE_KEY)
    expected_parent_route = {
        "capacity_execution_fingerprint":
            parent["capacity_execution_fingerprint"],
        "capacity_profile_fingerprint":
            parent["capacity_profile_fingerprint"],
        "hiding_execution_fingerprint":
            parent["hiding_execution_fingerprint"],
        "prognostic_secondary_execution_fingerprint":
            parent["fingerprint"],
    }
    if not isinstance(route, dict) or route != expected_parent_route:
        raise ValueError("Q3-H3/v58 ordered route/execution identity differs")
    identity = validate_prognostic_secondary_nonzero_execution_document(
        data.get(NONZERO_EXECUTION_KEY), parent=parent)
    validate_prognostic_secondary_nonzero_ordered_route(
        data.get(NONZERO_ORDERED_ROUTE_KEY), identity)
    if (identity["scalar_diffusion_linear_solver"] !=
            "jacobi_preconditioned_conjugate_gradient" or
            identity["scalar_diffusion_relative_tolerance"] !=
                DIFFUSION_RTOL or
            identity["scalar_diffusion_absolute_tolerance"] !=
                DIFFUSION_ATOL or
            identity["scalar_diffusion_max_iterations"] !=
                DIFFUSION_MAX_ITERATIONS):
        raise ValueError("Q3-H3/v58 scalar diffusion solver identity differs")
    if str(_scalar(data, "gpu_integrator")) != INTEGRATOR:
        raise ValueError(
            "Q3-H3/v58 requires the user-selected implicit_fwave_midpoint")
    if str(_scalar(data, "execution_backend")) != "gpu":
        raise ValueError("Q3-H3/v58 backend must be explicitly GPU")
    if bool(_scalar(data, "amr_enabled", False)):
        raise ValueError("Q3-H3/v58 first slice is fixed-grid only")
    if int(_scalar(data, "secondary_flow_model")) != 2:
        raise ValueError("Q3-H3/v58 requires prognostic secondary model 2")
    if not bool(_scalar(data, "secondary_flow_transport_direction")):
        raise ValueError("Q3-H3/v58 requires transported-state bed rotation")
    if int(_scalar(data, "secondary_flow_inflow_mode", 1)) != 1:
        raise ValueError("Q3-H3/v58 requires classic equilibrium boundary mode")
    if int(_scalar(data, "bedlaw")) not in (2, 8, 11):
        raise ValueError("Q3-H3/v58 capacity law is outside the first slice")
    for key in (
            "bed_slope", "exchange_mode", "bank_failure",
            "bank_front_tracking", "bedform_roughness",
            "general_avalanche"):
        if bool(_scalar(data, key)):
            raise ValueError(
                "Q3-H3/v58 first-slice process must be disabled: " + key)
    controls = np.asarray((
        identity["coriolis_constant_s_inverse"], identity["betac"],
        identity["scalar_diffusivity_m2_s"],
        identity["von_karman_constant"], identity["chzmin"],
    ), dtype=np.float64)
    if (not np.isfinite(controls).all() or controls[2] < 0.0 or
            controls[3] <= 0.0 or controls[4] <= 0.0):
        raise ValueError("Q3-H3/v58 nonzero controls are invalid")
    f = float(identity["coriolis_constant_s_inverse"])
    disposition = identity["coriolis_disposition"]
    source = identity["coriolis_source"]
    if ((f == 0.0 and (disposition != "zero" or
                       source != "constant_zero_planar")) or
            (f != 0.0 and (disposition != "nonzero" or
                           source != "constant_planar"))):
        raise ValueError("Q3-H3/v58 hydraulic/secondary Coriolis identity differs")
    beta = float(_scalar(data, "secondary_flow_momentum_beta", identity["betac"]))
    if not np.isfinite(beta) or beta != float(identity["betac"]):
        raise ValueError("Q3-H3/v58 prepared Betac differs from its identity")
    return identity


class ResidentClassicFlow2DHPrognosticNonzeroV58:
    """Transactional GPU owner for the bounded nonzero v58 operators.

    ``frozen_stress_flux`` is the hook consumed by every nonlinear residual of
    the selected implicit midpoint step.  It is assembled exactly once from
    the accepted-start state and is never recomputed from Newton iterates.
    """

    def __init__(self, data: dict, *, source_path: str | Path | None = None):
        self.identity = validate_scope(data)
        import cupy as cp

        self.cp = cp
        self.device_id = int(cp.cuda.Device().id)
        self.nc = int(_scalar(data, "nc"))
        self.nf = int(_scalar(data, "nf"))
        self.dry = float(_scalar(data, "dry"))
        self.gravity = float(_scalar(data, "grav"))
        self.coriolis = float(self.identity["coriolis_constant_s_inverse"])
        self.betac = float(self.identity["betac"])
        self.diffusivity = float(self.identity["scalar_diffusivity_m2_s"])
        self.von_karman = float(self.identity["von_karman_constant"])
        self.chzmin = float(self.identity["chzmin"])
        if (self.nc <= 0 or self.nf < 0 or not np.isfinite((
                self.dry, self.gravity, self.coriolis, self.betac,
                self.diffusivity, self.von_karman, self.chzmin)).all() or
                self.dry < 0.0 or self.gravity <= 0.0 or
                self.diffusivity < 0.0):
            raise ValueError("Q3-H3/v58 dimensions or controls are invalid")

        def host_integer(name, shape):
            value = np.ascontiguousarray(data[name], dtype=np.int32)
            if value.shape != shape:
                raise ValueError("Q3-H3/v58 geometry differs: " + name)
            return value

        def host_real(name, shape):
            value = np.ascontiguousarray(data[name], dtype=np.float64)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError("Q3-H3/v58 geometry differs: " + name)
            return value

        owner = host_integer("owner", (self.nf,))
        neighbour = host_integer("neighbour", (self.nf,))
        offset = host_integer("offset", (self.nc + 1,))
        faces = np.ascontiguousarray(data["faces"], dtype=np.int32)
        if (faces.ndim != 1 or offset[0] != 0 or
                offset[-1] != faces.size or np.any(np.diff(offset) < 0)):
            raise ValueError("Q3-H3/v58 adjacency layout differs")
        area = host_real("area", (self.nc,))
        length = host_real("length", (self.nf,))
        normal = host_real("normal", (2, self.nf))
        if (np.any(area <= 0.0) or np.any(length <= 0.0) or
                np.any(owner < 0) or np.any(owner >= self.nc) or
                np.any(neighbour < -1) or np.any(neighbour >= self.nc) or
                np.any(faces < 0) or np.any(faces >= self.nf)):
            raise ValueError("Q3-H3/v58 geometry is invalid")
        # CUDA graph kernels dereference ``faces`` directly.  Authenticate the
        # full CSR incidence on the host before any device launch so malformed
        # prepared geometry fails closed instead of becoming an OOB access.
        incidence = np.zeros(self.nf, dtype=np.int32)
        for cell in range(self.nc):
            row = faces[offset[cell]:offset[cell + 1]]
            if row.size != np.unique(row).size:
                raise ValueError("Q3-H3/v58 adjacency contains duplicates")
            if np.any((owner[row] != cell) & (neighbour[row] != cell)):
                raise ValueError("Q3-H3/v58 adjacency incidence differs")
            incidence[row] += 1
        expected_incidence = np.where(neighbour >= 0, 2, 1)
        if not np.array_equal(incidence, expected_incidence):
            raise ValueError("Q3-H3/v58 adjacency multiplicity differs")
        bid = np.ascontiguousarray(
            data.get("bid", np.full(self.nf, -1)), dtype=np.int32)
        if bid.shape != (self.nf,):
            raise ValueError("Q3-H3/v58 boundary ids differ")
        open_host = np.ascontiguousarray(data.get(
            "secondary_open_face", (neighbour < 0) & (bid >= 0)),
            dtype=np.int32)
        if (open_host.shape != (self.nf,) or
                np.any((open_host != 0) & (open_host != 1)) or
                np.any((open_host != 0) & (neighbour >= 0))):
            raise ValueError("Q3-H3/v58 open-face ownership differs")
        open_bids = bid[open_host != 0]
        if open_bids.size and np.any(open_bids < 0):
            raise ValueError("Q3-H3/v58 open faces require boundary ids")
        inferred_nb = int(open_bids.max() + 1) if open_bids.size else 0
        if "boundary_offset" in data:
            declared_offset = np.asarray(data["boundary_offset"])
            if (declared_offset.ndim != 1 or declared_offset.size < 1 or
                    int(declared_offset[0]) != 0):
                raise ValueError("Q3-H3/v58 boundary inventory differs")
            declared_nb = int(declared_offset.size - 1)
        elif "mode" in data:
            declared_mode = np.asarray(data["mode"])
            if declared_mode.ndim != 1:
                raise ValueError("Q3-H3/v58 boundary inventory differs")
            declared_nb = int(declared_mode.size)
        else:
            declared_nb = inferred_nb
        self.nb = max(declared_nb, inferred_nb)
        self.has_open_boundary = bool(open_bids.size)
        if np.any((bid >= self.nb) & (bid >= 0)):
            raise ValueError("Q3-H3/v58 boundary ids exceed the inventory")

        distance_value = data.get(
            "secondary_diffusion_distance", data.get("distance"))
        if distance_value is None:
            raise ValueError("Q3-H3/v58 diffusion centre distance is missing")
        distance = np.asarray(distance_value, dtype=np.float64)
        if distance.shape == (2, self.nf):
            distance = distance[0] + distance[1]
        distance = np.ascontiguousarray(distance, dtype=np.float64)
        if distance.shape != (self.nf,) or not np.isfinite(distance).all():
            raise ValueError("Q3-H3/v58 diffusion distance differs")
        participating = (neighbour >= 0) | (open_host != 0)
        if self.diffusivity > 0.0 and np.any(distance[participating] <= 0.0):
            raise ValueError("Q3-H3/v58 diffusion distance must be positive")

        mesh_digest = hashlib.sha256()
        mesh_digest.update(b"q3h3-v58-fixed-cartesian-mesh-v1\0")
        mesh_digest.update(np.asarray(
            (self.nc, self.nf, self.nb), dtype="<i8").tobytes())
        for name, value in (
                ("owner", owner), ("neighbour", neighbour),
                ("offset", offset), ("faces", faces), ("area", area),
                ("length", length), ("normal", normal), ("bid", bid),
                ("open", open_host), ("distance", distance)):
            canonical = np.ascontiguousarray(value)
            mesh_digest.update(name.encode("ascii") + b"\0")
            mesh_digest.update(str(canonical.dtype).encode("ascii") + b"\0")
            mesh_digest.update(np.asarray(
                canonical.shape, dtype="<i8").tobytes())
            mesh_digest.update(canonical.tobytes(order="C"))
        self.mesh_fingerprint = mesh_digest.hexdigest()

        source = Path(source_path or Path(__file__).with_suffix(".cu")).resolve()
        if not source.is_file() or source.is_symlink():
            raise ValueError("Q3-H3/v58 CUDA source is missing or linked")
        self.source_path = source
        self.source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
        with cp.cuda.Device(self.device_id):
            self.module = cp.RawModule(
                code=source.read_text(encoding="utf-8"), options=CUDA_OPTIONS,
                name_expressions=KERNEL_NAMES)
            self.kernels = {
                name: self.module.get_function(name) for name in KERNEL_NAMES
            }
            self.owner = cp.asarray(owner)
            self.neighbour = cp.asarray(neighbour)
            self.offset = cp.asarray(offset)
            self.faces = cp.asarray(faces)
            self.area = cp.asarray(area)
            self.length = cp.asarray(length)
            self.normal = cp.asarray(normal)
            self.bid = cp.asarray(bid)
            self.open_face = cp.asarray(open_host)
            self.distance = cp.asarray(distance)
            self.accepted_q_snapshot = cp.zeros((3, self.nc), dtype=cp.float64)
            self.accepted_hi_snapshot = cp.zeros(self.nc, dtype=cp.float64)
            self.q_trial = cp.zeros((3, self.nc), dtype=cp.float64)
            self.hi_trial = cp.zeros(self.nc, dtype=cp.float64)
            self.cell_tensor = cp.zeros((3, self.nc), dtype=cp.float64)
            self.frozen_stress_flux = cp.zeros((2, self.nf), dtype=cp.float64)
            self.stress_boundary_rate = cp.zeros((2, self.nb), dtype=cp.float64)
            self.stress_boundary_increment = cp.zeros(
                (2, self.nb), dtype=cp.float64)
            self.accepted_stress_boundary_cumulative = cp.zeros(
                (2, self.nb), dtype=cp.float64)
            self.trial_stress_boundary_cumulative = cp.zeros(
                (2, self.nb), dtype=cp.float64)
            self.coriolis_increment_cells = cp.zeros(
                (2, self.nc), dtype=cp.float64)
            self.coriolis_integrated_increment = cp.zeros(2, dtype=cp.float64)
            self.accepted_coriolis_cumulative = cp.zeros(2, dtype=cp.float64)
            self.trial_coriolis_cumulative = cp.zeros(2, dtype=cp.float64)
            self.diffusion_conductance = cp.zeros(self.nf, dtype=cp.float64)
            self.diffusion_trace = cp.zeros(self.nf, dtype=cp.float64)
            self.diffusion_candidate = cp.zeros(self.nc, dtype=cp.float64)
            self.diffusion_boundary_work = cp.zeros(self.nb, dtype=cp.float64)
            self.accepted_diffusion_boundary_cumulative = cp.zeros(
                self.nb, dtype=cp.float64)
            self.trial_diffusion_boundary_cumulative = cp.zeros(
                self.nb, dtype=cp.float64)
            self.x = cp.zeros(self.nc, dtype=cp.float64)
            self.rhs = cp.zeros(self.nc, dtype=cp.float64)
            self.diagonal = cp.ones(self.nc, dtype=cp.float64)
            self.r = cp.zeros(self.nc, dtype=cp.float64)
            self.z = cp.zeros(self.nc, dtype=cp.float64)
            self.p = cp.zeros(self.nc, dtype=cp.float64)
            self.ap = cp.zeros(self.nc, dtype=cp.float64)
            self.product = cp.zeros(self.nc, dtype=cp.float64)
            self.pcg_scalars = cp.zeros(6, dtype=cp.float64)
            self.pcg_control = cp.zeros(4, dtype=cp.int32)
            self.cancel_flag = cp.zeros(1, dtype=cp.int32)
            self.rhs2 = cp.zeros((), dtype=cp.float64)
            self.rr = cp.zeros((), dtype=cp.float64)
            self.rz = cp.zeros((), dtype=cp.float64)
            self.denominator = cp.zeros((), dtype=cp.float64)
            self.rz_new = cp.zeros((), dtype=cp.float64)
            self.cell_status = cp.zeros(self.nc, dtype=cp.int32)
            self.face_status = cp.zeros(self.nf, dtype=cp.int32)
            self.diffusion_cell_status = cp.zeros(self.nc, dtype=cp.int32)
            self.diffusion_face_status = cp.zeros(self.nf, dtype=cp.int32)
            self.zero = cp.zeros(self.nc, dtype=cp.float64)
            self.coriolis_field = cp.full(
                self.nc, self.coriolis, dtype=cp.float64)
        self.point = Flow2DHSecondaryCUDA()
        self.point_result = self.point.allocate_result(self.nc)
        self._trial_open = False
        self._stress_frozen = False
        self._accepted_q_external_ptr = None
        self._accepted_hi_external_ptr = None
        self._last_status = STATUS_OK
        self._last_diffusion = {
            "iterations": 0, "residual": 0.0, "target": 0.0,
            "converged": True, "cancelled": False,
            "conservation_residual_m4_s": 0.0,
        }
        self.freeze_generation = 0
        self._stress_ledger_recorded = False
        self._stress_freeze_attempted = False
        self._coriolis_applied = False
        self._diffusion_applied = False
        self._have_accepted_state = False

    def _launch(self, name, count, args):
        if count:
            with self.cp.cuda.Device(self.device_id):
                self.kernels[name](((count + 255) // 256,), (256,), args)

    def _fp64(self, name, value, shape):
        cp = self.cp
        if (not isinstance(value, cp.ndarray) or value.dtype != cp.float64 or
                value.shape != shape or not value.flags.c_contiguous or
                int(value.device.id) != self.device_id):
            raise ValueError(
                f"Q3-H3/v58 {name} must be resident contiguous FP64 {shape}")
        return value

    def _int32(self, name, value, shape):
        cp = self.cp
        if (not isinstance(value, cp.ndarray) or value.dtype != cp.int32 or
                value.shape != shape or not value.flags.c_contiguous or
                int(value.device.id) != self.device_id):
            raise ValueError(
                f"Q3-H3/v58 {name} must be resident contiguous int32 {shape}")
        return value

    def begin_trial(self, accepted_q, accepted_hi):
        self._fp64("accepted hydraulics", accepted_q, (3, self.nc))
        self._fp64("accepted signed hI", accepted_hi, (self.nc,))
        if self._trial_open:
            raise RuntimeError("Q3-H3/v58 transaction is already open")
        cp = self.cp
        # The first admitted v58 slice is deliberately fully wet.  Reject the
        # accepted state before any CUDA operator launch; wet/dry ownership is
        # a later scope and must never be silently approximated here.
        if (not bool(cp.isfinite(accepted_q).all().item()) or
                not bool(cp.isfinite(accepted_hi).all().item()) or
                float(cp.min(accepted_q[0]).item()) <= self.dry):
            raise ValueError(
                "Q3-H3/v58 first slice requires finite fully wet state")
        cp.copyto(self.accepted_q_snapshot, accepted_q)
        cp.copyto(self.accepted_hi_snapshot, accepted_hi)
        cp.copyto(self.q_trial, accepted_q)
        cp.copyto(self.hi_trial, accepted_hi)
        self.stress_boundary_rate.fill(0.0)
        self.stress_boundary_increment.fill(0.0)
        cp.copyto(self.trial_stress_boundary_cumulative,
                  self.accepted_stress_boundary_cumulative)
        self.coriolis_increment_cells.fill(0.0)
        self.coriolis_integrated_increment.fill(0.0)
        cp.copyto(self.trial_coriolis_cumulative,
                  self.accepted_coriolis_cumulative)
        cp.copyto(self.trial_diffusion_boundary_cumulative,
                  self.accepted_diffusion_boundary_cumulative)
        self.cancel_flag.fill(0)
        self._accepted_q_external_ptr = int(accepted_q.data.ptr)
        self._accepted_hi_external_ptr = int(accepted_hi.data.ptr)
        self._trial_open = True
        self._stress_frozen = False
        self._last_status = STATUS_OK
        self._stress_ledger_recorded = False
        self._stress_freeze_attempted = False
        self._coriolis_applied = False
        self._diffusion_applied = False
        self._have_accepted_state = True
        return self.q_trial, self.hi_trial

    def _require_trial(self):
        if not self._trial_open:
            raise RuntimeError("Q3-H3/v58 transaction has not begun")

    def evaluate_source_sink(self, active, h_old, h_new, u, v, curvature,
                             metric_x, metric_y, raw_chezy):
        """Evaluate locked ``secrhs`` with the shared signed Coriolis field."""
        cp = self.cp
        if (not isinstance(active, cp.ndarray) or active.dtype != cp.uint8 or
                active.shape != (self.nc,) or not active.flags.c_contiguous or
                int(active.device.id) != self.device_id):
            raise ValueError("Q3-H3/v58 point active mask must be resident uint8")
        arrays = (h_old, h_new, u, v, curvature, metric_x, metric_y, raw_chezy)
        for name, value in zip((
                "old depth", "new depth", "u", "v", "curvature",
                "metric x", "metric y", "raw Chezy"), arrays):
            self._fp64(name, value, (self.nc,))
        return self.point.evaluate(
            active, h_old, h_new, u, v, curvature, metric_x, metric_y,
            self.coriolis_field, raw_chezy, self.zero, self.zero, self.zero,
            dryflc=self.dry, rmincf=float(self.identity["rmincf"]),
            chzmin=self.chzmin, gravity=self.gravity,
            von_karman=self.von_karman, espir=float(self.identity["espir"]),
            betac=0.0, reject_invalid=False, out=self.point_result)

    def freeze_betac(self, cell_chezy, boundary_trace,
                     boundary_equilibrium, boundary_chezy, active):
        """Freeze one accepted-start ``-T.n`` field for all midpoint trials."""
        self._require_trial()
        if self._stress_freeze_attempted:
            raise RuntimeError(
                "Q3-H3/v58 Betac feedback may be frozen only once per trial")
        self._stress_freeze_attempted = True
        self._fp64("cell Chezy", cell_chezy, (self.nc,))
        self._fp64("boundary trace", boundary_trace, (3, self.nf))
        self._fp64(
            "boundary equilibrium", boundary_equilibrium, (self.nc,))
        self._fp64("boundary Chezy", boundary_chezy, (self.nf,))
        self._int32("active mask", active, (self.nc,))
        self.cell_status.fill(0);self.face_status.fill(0)
        self.cell_tensor.fill(0.0);self.frozen_stress_flux.fill(0.0)
        self.stress_boundary_rate.fill(0.0)
        self._launch("v58_betac_cells", self.nc, (
            np.int32(self.nc), self.accepted_q_snapshot,
            self.accepted_hi_snapshot, cell_chezy, active,
            np.float64(self.dry), np.float64(self.betac),
            np.float64(self.gravity), np.float64(self.von_karman),
            np.float64(self.chzmin), self.cell_tensor, self.cell_status))
        self._launch("v58_betac_faces", self.nf, (
            np.int32(self.nc), np.int32(self.nf), boundary_trace, self.owner,
            self.neighbour, self.open_face, self.normal, self.cell_tensor,
            boundary_equilibrium, boundary_chezy, np.float64(self.dry),
            np.float64(self.betac), np.float64(self.gravity),
            np.float64(self.von_karman), np.float64(self.chzmin),
            self.frozen_stress_flux, self.face_status))
        self._launch("v58_stress_boundary_rate", self.nf, (
            np.int32(self.nf), np.int32(self.nb), self.neighbour,
            self.open_face, self.bid, self.length, self.frozen_stress_flux,
            self.stress_boundary_rate, self.face_status))
        status = max(int(self.cp.max(self.cell_status).item()),
                     int(self.cp.max(self.face_status).item()))
        self._last_status = max(self._last_status, status)
        self._stress_frozen = self._last_status == STATUS_OK
        if self._stress_frozen:
            self.freeze_generation += 1
        return {"flux": self.frozen_stress_flux,
                "boundary_rate": self.stress_boundary_rate,
                "status": status,
                "generation": self.freeze_generation}

    def add_frozen_stress(self, flux_left, flux_right):
        """Add the immutable accepted-start flux to one implicit residual."""
        self._require_trial()
        if not self._stress_frozen:
            raise RuntimeError("Q3-H3/v58 Betac flux has not been frozen")
        self._fp64("left f-wave flux", flux_left, (3, self.nf))
        self._fp64("right f-wave flux", flux_right, (3, self.nf))
        self._launch("v58_add_frozen_stress", self.nf, (
            np.int32(self.nf), self.frozen_stress_flux,
            flux_left, flux_right))
        return self.frozen_stress_flux

    def accept_stress_ledger(self, dt):
        self._require_trial()
        if not np.isfinite(dt) or dt < 0.0 or not self._stress_frozen:
            raise ValueError("Q3-H3/v58 stress-ledger timestep differs")
        if self._stress_ledger_recorded:
            raise RuntimeError("Q3-H3/v58 stress ledger is already recorded")
        self.cp.multiply(
            self.stress_boundary_rate, np.float64(dt),
            out=self.stress_boundary_increment)
        self.trial_stress_boundary_cumulative += \
            self.stress_boundary_increment
        self._stress_ledger_recorded = True
        return self.stress_boundary_increment

    def load_hydraulic_endpoint(self, q_endpoint):
        self._require_trial()
        self._fp64("hydraulic endpoint", q_endpoint, (3, self.nc))
        self.cp.copyto(self.q_trial, q_endpoint)
        return self.q_trial

    def apply_coriolis(self, dt):
        """Apply the exact split rotation to the private hydraulic endpoint."""
        self._require_trial()
        if not np.isfinite(dt) or dt < 0.0:
            raise ValueError("Q3-H3/v58 Coriolis timestep must be nonnegative")
        if self._coriolis_applied:
            raise RuntimeError("Q3-H3/v58 Coriolis split is already applied")
        self.cell_status.fill(0)
        self.coriolis_increment_cells.fill(0.0)
        self._launch("v58_constant_coriolis", self.nc, (
            np.int32(self.nc), np.float64(self.coriolis), np.float64(dt),
            self.area, self.q_trial, self.coriolis_increment_cells,
            self.cell_status))
        status = int(self.cp.max(self.cell_status).item())
        if status == STATUS_OK:
            self.cp.sum(
                self.coriolis_increment_cells, axis=1,
                out=self.coriolis_integrated_increment)
            self.trial_coriolis_cumulative += \
                self.coriolis_integrated_increment
            self._coriolis_applied = True
        self._last_status = max(self._last_status, status)
        return {"state": self.q_trial,
                "integrated_increment": self.coriolis_integrated_increment,
                "status": status}

    def request_cancel(self):
        """Set a resident cooperative cancellation flag."""
        self.cancel_flag.fill(1)

    def diffuse(self, depth_new, boundary_equilibrium, dt, *,
                boundary_depth=None,
                relative_tolerance=DIFFUSION_RTOL,
                absolute_tolerance=DIFFUSION_ATOL,
                max_iterations=DIFFUSION_MAX_ITERATIONS):
        """Solve the conservative signed graph BE system wholly on CUDA."""
        self._require_trial()
        if self._diffusion_applied:
            raise RuntimeError("Q3-H3/v58 diffusion is already applied")
        cp = self.cp
        self._fp64("new depth", depth_new, (self.nc,))
        self._fp64(
            "diffusion boundary equilibrium", boundary_equilibrium,
            (self.nc,))
        if boundary_depth is None:
            if (self.has_open_boundary and self.diffusivity > 0.0 and
                    dt > 0.0):
                raise ValueError(
                    "Q3-H3/v58 open diffusion requires hydraulic trace depth")
            boundary_depth = depth_new[self.owner]
        self._fp64(
            "diffusion boundary depth", boundary_depth, (self.nf,))
        controls = np.asarray((dt, relative_tolerance, absolute_tolerance),
                              dtype=np.float64)
        if (not np.isfinite(controls).all() or dt < 0.0 or
                relative_tolerance < 0.0 or absolute_tolerance < 0.0 or
                isinstance(max_iterations, bool) or
                not isinstance(max_iterations, int) or max_iterations < 1):
            raise ValueError("Q3-H3/v58 diffusion controls are invalid")
        if int(self.cancel_flag.item()) != 0:
            self._last_status = STATUS_CANCELLED
            self._last_diffusion = {
                "iterations": 0, "residual": np.inf, "target": 0.0,
                "converged": False, "cancelled": True,
                "conservation_residual_m4_s": 0.0,
            }
            self._diffusion_applied = True
            return {"state": self.hi_trial,
                    "boundary_ledger": self.diffusion_boundary_work,
                    "status": STATUS_CANCELLED, **self._last_diffusion}
        # This branch is deliberately exact: v58 with D=0 is byte-identical
        # to the accepted v57 state update and launches no numerical solver.
        if self.diffusivity == 0.0 or dt == 0.0:
            self.diffusion_boundary_work.fill(0.0)
            self._last_diffusion = {
                "iterations": 0, "residual": 0.0, "target": 0.0,
                "converged": True, "cancelled": False,
                "conservation_residual_m4_s": 0.0,
            }
            self._diffusion_applied = True
            return {"state": self.hi_trial,
                    "boundary_ledger": self.diffusion_boundary_work,
                    "status": STATUS_OK, **self._last_diffusion}

        cp.take(boundary_equilibrium, self.owner, out=self.diffusion_trace)
        self.diffusion_face_status.fill(0)
        self.diffusion_cell_status.fill(0)
        self.diffusion_boundary_work.fill(0.0)
        self._launch("v58_diffusion_conductance", self.nf, (
            np.int32(self.nc), np.int32(self.nf), depth_new, boundary_depth,
            np.float64(self.diffusivity), self.length, self.distance,
            self.owner, self.neighbour, self.open_face,
            self.diffusion_conductance, self.diffusion_face_status))
        self._launch("v58_diffusion_setup", self.nc, (
            np.int32(self.nc), np.int32(self.nf), self.offset, self.faces,
            self.owner, self.neighbour, self.open_face, self.area, depth_new,
            self.diffusion_conductance, self.diffusion_trace, self.hi_trial,
            np.float64(dt), self.x, self.rhs, self.diagonal,
            self.diffusion_cell_status))
        self.pcg_control.fill(1)
        self._launch("v58_diffusion_matvec", self.nc, (
            np.int32(self.nc), np.int32(self.nf), self.offset, self.faces,
            self.owner, self.neighbour, self.open_face, self.area, depth_new,
            self.diffusion_conductance, np.float64(dt), self.x, self.ap,
            self.pcg_control, np.int32(1)))
        self._launch("v58_pcg_initialize_vectors", self.nc, (
            np.int32(self.nc), self.rhs, self.ap, self.diagonal,
            self.r, self.z, self.p))
        cp.multiply(self.rhs, self.rhs, out=self.product)
        cp.sum(self.product, out=self.rhs2)
        cp.multiply(self.r, self.r, out=self.product)
        cp.sum(self.product, out=self.rr)
        cp.multiply(self.r, self.z, out=self.product)
        cp.sum(self.product, out=self.rz)
        self._launch("v58_pcg_initialize_control", 1, (
            np.float64(relative_tolerance), np.float64(absolute_tolerance),
            self.rhs2, self.rr, self.rz, self.cancel_flag,
            self.pcg_scalars, self.pcg_control))
        for iteration in range(1, max_iterations + 1):
            self._launch("v58_diffusion_matvec", self.nc, (
                np.int32(self.nc), np.int32(self.nf), self.offset, self.faces,
                self.owner, self.neighbour, self.open_face, self.area,
                depth_new, self.diffusion_conductance, np.float64(dt),
                self.p, self.ap, self.pcg_control, np.int32(0)))
            cp.multiply(self.p, self.ap, out=self.product)
            cp.sum(self.product, out=self.denominator)
            self._launch("v58_pcg_alpha", 1, (
                self.denominator, self.cancel_flag, self.pcg_scalars,
                self.pcg_control))
            self._launch("v58_pcg_update_xr", self.nc, (
                np.int32(self.nc), self.p, self.ap, self.pcg_scalars,
                self.pcg_control, self.x, self.r))
            self._launch("v58_pcg_precondition", self.nc, (
                np.int32(self.nc), self.r, self.diagonal, self.pcg_control,
                self.z))
            cp.multiply(self.r, self.r, out=self.product)
            cp.sum(self.product, out=self.rr)
            cp.multiply(self.r, self.z, out=self.product)
            cp.sum(self.product, out=self.rz_new)
            self._launch("v58_pcg_convergence_beta", 1, (
                np.int32(iteration), self.rr, self.rz_new,
                self.cancel_flag, self.pcg_scalars, self.pcg_control))
            self._launch("v58_pcg_update_p", self.nc, (
                np.int32(self.nc), self.z, self.pcg_scalars,
                self.pcg_control, self.p))

        cp.copyto(self.diffusion_candidate, self.hi_trial)
        self._launch("v58_diffusion_finish", self.nc, (
            np.int32(self.nc), depth_new, self.x, self.pcg_control,
            self.diffusion_candidate, self.diffusion_cell_status))
        self._launch("v58_diffusion_boundary_ledger", self.nf, (
            np.int32(self.nf), np.int32(self.nb), self.owner,
            self.neighbour, self.open_face, self.bid,
            self.diffusion_conductance, self.diffusion_trace, self.x,
            np.float64(dt), self.diffusion_boundary_work,
            self.diffusion_face_status))

        # One final diagnostic transfer is allowed by the v58 residency
        # contract.  No iteration performs a host convergence round-trip.
        control = cp.asnumpy(self.pcg_control)
        scalars = cp.asnumpy(self.pcg_scalars)
        cell_status = int(cp.max(self.diffusion_cell_status).item())
        face_status = int(cp.max(self.diffusion_face_status).item())
        status = max(cell_status, face_status, int(control[1]))
        if status == STATUS_OK and not bool(control[3]):
            status = STATUS_NOT_CONVERGED
        before = cp.sum(self.area * self.hi_trial)
        after = cp.sum(self.area * self.diffusion_candidate)
        boundary = cp.sum(self.diffusion_boundary_work)
        conservation = float((after - before - boundary).item())
        scale = max(1.0, abs(float(before.item())), abs(float(after.item())),
                    abs(float(boundary.item())))
        if (status == STATUS_OK and
                (not np.isfinite(conservation) or
                 abs(conservation) > max(1.0e-12, 2.0e-12 * scale))):
            status = STATUS_INVALID_OUTPUT
        if status == STATUS_OK:
            cp.copyto(self.hi_trial, self.diffusion_candidate)
            self.trial_diffusion_boundary_cumulative += \
                self.diffusion_boundary_work
            self._diffusion_applied = True
        self._last_status = max(self._last_status, status)
        self._last_diffusion = {
            "iterations": int(control[2]),
            "residual": float(scalars[2]),
            "target": float(scalars[3]),
            "converged": status == STATUS_OK,
            "cancelled": status == STATUS_CANCELLED,
            "conservation_residual_m4_s": conservation,
        }
        return {"state": self.hi_trial,
                "boundary_ledger": self.diffusion_boundary_work,
                "status": status, **self._last_diffusion}

    def finish_trial(self, accepted_q, accepted_hi, *, commit):
        """Atomically publish both private states or publish neither."""
        self._require_trial()
        self._fp64("accepted hydraulics", accepted_q, (3, self.nc))
        self._fp64("accepted signed hI", accepted_hi, (self.nc,))
        if not isinstance(commit, (bool, np.bool_)):
            raise TypeError("Q3-H3/v58 commit flag must be boolean")
        if (int(accepted_q.data.ptr) != self._accepted_q_external_ptr or
                int(accepted_hi.data.ptr) != self._accepted_hi_external_ptr):
            raise RuntimeError(
                "Q3-H3/v58 accepted state ownership changed during trial")
        if bool(commit):
            if self._last_status != STATUS_OK:
                raise RuntimeError("Cannot commit an invalid Q3-H3/v58 trial")
            self.cp.copyto(accepted_q, self.q_trial)
            self.cp.copyto(accepted_hi, self.hi_trial)
            self.cp.copyto(self.accepted_q_snapshot, self.q_trial)
            self.cp.copyto(self.accepted_hi_snapshot, self.hi_trial)
            self.cp.copyto(self.accepted_stress_boundary_cumulative,
                           self.trial_stress_boundary_cumulative)
            self.cp.copyto(self.accepted_coriolis_cumulative,
                           self.trial_coriolis_cumulative)
            self.cp.copyto(self.accepted_diffusion_boundary_cumulative,
                           self.trial_diffusion_boundary_cumulative)
        self._trial_open = False
        self._stress_frozen = False
        self._accepted_q_external_ptr = None
        self._accepted_hi_external_ptr = None
        return accepted_q, accepted_hi

    def abort_trial(self):
        if self._trial_open:
            self._trial_open = False
            self._stress_frozen = False
            self._accepted_q_external_ptr = None
            self._accepted_hi_external_ptr = None

    def checkpoint_fields(self):
        """Export only committed state and ledgers for restart schema 9."""
        if self._trial_open:
            raise RuntimeError("Q3-H3/v58 cannot checkpoint an open trial")
        if not self._have_accepted_state:
            raise RuntimeError("Q3-H3/v58 has no accepted state to checkpoint")
        cp = self.cp
        return {
            "schema": np.int32(CHECKPOINT_SCHEMA),
            "scope_id": SCOPE_ID,
            "execution_fingerprint": self.identity["fingerprint"],
            "cuda_source_sha256": self.source_sha256,
            "mesh_fingerprint": self.mesh_fingerprint,
            "q": cp.asnumpy(self.accepted_q_snapshot),
            "h_i": cp.asnumpy(self.accepted_hi_snapshot),
            "diffusion_boundary_cumulative_m4_s": cp.asnumpy(
                self.accepted_diffusion_boundary_cumulative),
            "betac_momentum_boundary_cumulative_m4_s": cp.asnumpy(
                self.accepted_stress_boundary_cumulative),
            "coriolis_increment_cumulative_m4_s": cp.asnumpy(
                self.accepted_coriolis_cumulative),
        }

    def validate_checkpoint_fields(self, fields):
        """Validate a serialized checkpoint without mutating resident state."""
        if self._trial_open:
            raise RuntimeError("Q3-H3/v58 cannot validate during a trial")
        required = {
            "schema", "scope_id", "execution_fingerprint",
            "cuda_source_sha256", "mesh_fingerprint", "q", "h_i",
            "diffusion_boundary_cumulative_m4_s",
            "betac_momentum_boundary_cumulative_m4_s",
            "coriolis_increment_cumulative_m4_s",
        }
        if not isinstance(fields, dict) or set(fields) != required:
            raise ValueError("Q3-H3/v58 checkpoint inventory differs")
        if (type(fields["schema"]) not in (int, np.int32, np.int64) or
                int(fields["schema"]) != CHECKPOINT_SCHEMA or
                fields["scope_id"] != SCOPE_ID or
                fields["execution_fingerprint"] != self.identity["fingerprint"] or
                fields["cuda_source_sha256"] != self.source_sha256 or
                fields["mesh_fingerprint"] != self.mesh_fingerprint):
            raise ValueError("Q3-H3/v58 checkpoint identity differs")

        arrays = {
            "q": ((3, self.nc), fields["q"]),
            "h_i": ((self.nc,), fields["h_i"]),
            "diffusion": ((self.nb,),
                fields["diffusion_boundary_cumulative_m4_s"]),
            "stress": ((2, self.nb),
                fields["betac_momentum_boundary_cumulative_m4_s"]),
            "coriolis": ((2,), fields["coriolis_increment_cumulative_m4_s"]),
        }
        host = {}
        for name, (shape, value) in arrays.items():
            item = np.asarray(value)
            if (item.shape != shape or item.dtype != np.float64 or
                    not item.flags.c_contiguous or not np.isfinite(item).all()):
                raise ValueError("Q3-H3/v58 checkpoint field differs: " + name)
            host[name] = item
        return host

    def restore_checkpoint(self, fields, accepted_q, accepted_hi):
        """Restore one authenticated committed v58 state without fallback."""
        self._fp64("restart hydraulics", accepted_q, (3, self.nc))
        self._fp64("restart signed hI", accepted_hi, (self.nc,))
        host = self.validate_checkpoint_fields(fields)
        accepted_q.set(host["q"]);accepted_hi.set(host["h_i"])
        self.accepted_q_snapshot.set(host["q"])
        self.accepted_hi_snapshot.set(host["h_i"])
        self.accepted_diffusion_boundary_cumulative.set(host["diffusion"])
        self.accepted_stress_boundary_cumulative.set(host["stress"])
        self.accepted_coriolis_cumulative.set(host["coriolis"])
        self._have_accepted_state = True
        return accepted_q, accepted_hi

    def storage_identity(self):
        values = (
            self.accepted_q_snapshot, self.accepted_hi_snapshot,
            self.q_trial, self.hi_trial, self.cell_tensor,
            self.frozen_stress_flux, self.stress_boundary_rate,
            self.stress_boundary_increment,
            self.accepted_stress_boundary_cumulative,
            self.trial_stress_boundary_cumulative,
            self.coriolis_increment_cells,
            self.coriolis_integrated_increment,
            self.accepted_coriolis_cumulative,
            self.trial_coriolis_cumulative,
            self.diffusion_conductance, self.diffusion_trace,
            self.diffusion_candidate, self.diffusion_boundary_work,
            self.accepted_diffusion_boundary_cumulative,
            self.trial_diffusion_boundary_cumulative, self.x, self.rhs,
            self.diagonal, self.r, self.z, self.p, self.ap, self.product,
            self.pcg_scalars, self.pcg_control, self.cancel_flag,
        )
        return tuple((id(value), int(value.data.ptr), value.shape,
                      value.dtype.str, value.strides, int(value.device.id))
                     for value in values)

    def diagnostic(self):
        return {
            "scope_id": SCOPE_ID,
            "envelope_version": ENVELOPE_VERSION,
            "checkpoint_schema": CHECKPOINT_SCHEMA,
            "execution_fingerprint": self.identity["fingerprint"],
            "cuda_source_sha256": self.source_sha256,
            "mesh_fingerprint": self.mesh_fingerprint,
            "device_id": self.device_id,
            "automatic_cpu_fallback": False,
            "integrator": INTEGRATOR,
            "stress_lag": "accepted_start_once_per_macro_step",
            "freeze_generation": self.freeze_generation,
            "coriolis_integrated_increment_m4_s":
                self.coriolis_integrated_increment,
            "stress_boundary_rate_m4_s2": self.stress_boundary_rate,
            "stress_boundary_increment_m4_s":
                self.stress_boundary_increment,
            "accepted_betac_momentum_boundary_cumulative_m4_s":
                self.accepted_stress_boundary_cumulative,
            "accepted_coriolis_increment_cumulative_m4_s":
                self.accepted_coriolis_cumulative,
            "accepted_diffusion_boundary_cumulative_m4_s":
                self.accepted_diffusion_boundary_cumulative,
            "diffusion": dict(self._last_diffusion),
        }


__all__ = [
    "ENVELOPE_VERSION", "INTEGRATOR", "KERNEL_NAMES", "SCOPE_ID",
    "SCHMIDT_NUMBER", "STATUS_CANCELLED", "STATUS_NOT_CONVERGED",
    "STATUS_OK", "ResidentClassicFlow2DHPrognosticNonzeroV58",
    "validate_scope",
]
