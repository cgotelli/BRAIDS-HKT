"""Isolated resident-CUDA classic Delft3D4 FLOW 2DH secondary operator.

This is a constitutive foundation, not a live/public solver route.  It accepts
already differentiated signed inverse radius and explicit FLOW metric sums; it
does not reconstruct geometry, choose a backend, transfer host arrays, or fall
back to CPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


CUDA_OPTIONS = (
    "--std=c++11",
    "--fmad=false",
    "--prec-div=true",
    "--prec-sqrt=true",
    "--ftz=false",
)
KERNEL_NAME = "flow2dh_secondary_operator"

STATUS_OK = 0
STATUS_INVALID_INPUT = 1
STATUS_INVALID_GEOMETRY = 2
STATUS_INVALID_RELAXATION = 3
STATUS_INVALID_OUTPUT = 4

SOURCE_ANCHORS = {
    "source_sink": "secrhs.f90:172-200",
    "open_boundary": "secbou.f90:132-154",
    "bed_rotation": "eqtran.f90:713-747",
    "momentum_stress": "vihsec.f90:137-207",
}


@dataclass(frozen=True)
class ResidentFlow2DHSecondaryResult:
    capped_inverse_radius: object
    cap_limit: object
    equilibrium_target: object
    source_coefficient: object
    sink_coefficient: object
    boundary_equilibrium: object
    rotated_x: object
    rotated_y: object
    spiral_alpha: object
    effective_chezy: object
    relaxation_alpha: object
    betast: object
    beta: object
    stress_xx_basis: object
    stress_xy_basis: object
    was_capped: object
    boundary_defined: object
    betac_enabled: object
    beta_was_clamped: object
    status: object


class Flow2DHSecondaryCUDA:
    """Compile and evaluate the isolated FP64 operator on resident arrays."""

    _ARRAY_NAMES = (
        "depth_old",
        "depth_new",
        "u",
        "v",
        "raw_inverse_radius",
        "metric_ksi_sum",
        "metric_eta_sum",
        "coriolis",
        "raw_chezy",
        "spiral_intensity",
        "bed_x",
        "bed_y",
    )
    _OUTPUT_NAMES = (
        "capped_inverse_radius",
        "cap_limit",
        "equilibrium_target",
        "source_coefficient",
        "sink_coefficient",
        "boundary_equilibrium",
        "rotated_x",
        "rotated_y",
        "spiral_alpha",
        "effective_chezy",
        "relaxation_alpha",
        "betast",
        "beta",
        "stress_xx_basis",
        "stress_xy_basis",
    )
    _FLAG_NAMES = (
        "was_capped",
        "boundary_defined",
        "betac_enabled",
        "beta_was_clamped",
    )

    def __init__(self, source_path: str | Path | None = None):
        import cupy as cp

        self.cp = cp
        self.device_id = int(cp.cuda.Device().id)
        path = Path(source_path or Path(__file__).with_suffix(".cu"))
        with cp.cuda.Device(self.device_id):
            self.module = cp.RawModule(
                code=path.read_text(encoding="utf-8"),
                options=CUDA_OPTIONS,
                name_expressions=(KERNEL_NAME,),
            )
            self.kernel = self.module.get_function(KERNEL_NAME)

    @staticmethod
    def _finite_scalar(name: str, value: float) -> float:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must be a finite scalar")
        try:
            result = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be a finite scalar") from exc
        if not np.isfinite(result):
            raise ValueError(f"{name} must be a finite scalar")
        return result

    def _fp64_array(self, name: str, value: object, shape=None):
        cp = self.cp
        if (
            not isinstance(value, cp.ndarray)
            or value.dtype != cp.float64
            or value.ndim != 1
            or not value.flags.c_contiguous
        ):
            raise TypeError(
                f"{name} must be a resident contiguous 1-D FP64 CuPy array"
            )
        if int(value.device.id) != self.device_id:
            raise ValueError(
                f"{name} is not resident on CUDA device {self.device_id}"
            )
        if shape is not None and tuple(value.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
        return value

    def _active_array(self, value: object, shape=None):
        cp = self.cp
        if (
            not isinstance(value, cp.ndarray)
            or value.dtype != cp.uint8
            or value.ndim != 1
            or not value.flags.c_contiguous
        ):
            raise TypeError(
                "active must be a resident contiguous 1-D uint8 CuPy array"
            )
        if int(value.device.id) != self.device_id:
            raise ValueError(
                f"active is not resident on CUDA device {self.device_id}"
            )
        if shape is not None and tuple(value.shape) != shape:
            raise ValueError(f"active must have shape {shape}, got {value.shape}")
        return value

    def allocate_result(self, count: int) -> ResidentFlow2DHSecondaryResult:
        """Allocate one reusable resident result inventory.

        The isolated convenience API still allocates by default.  Live solver
        composition calls this once and then supplies the result to every
        trial-stage evaluation, avoiding allocator traffic and host copies.
        """
        cp = self.cp
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("FLOW 2DH result count must be a nonnegative integer")
        shape = (count,)
        with cp.cuda.Device(self.device_id):
            outputs = {
                name: cp.empty(shape, dtype=cp.float64)
                for name in self._OUTPUT_NAMES
            }
            flags = {
                name: cp.empty(shape, dtype=cp.uint8)
                for name in self._FLAG_NAMES
            }
            status = cp.empty(shape, dtype=cp.int32)
        return ResidentFlow2DHSecondaryResult(
            **outputs, **flags, status=status)

    def _result_arrays(self, value, shape):
        if not isinstance(value, ResidentFlow2DHSecondaryResult):
            raise TypeError("out must be a ResidentFlow2DHSecondaryResult")
        outputs = {
            name: self._fp64_array(name, getattr(value, name), shape)
            for name in self._OUTPUT_NAMES
        }
        flags = {
            name: self._active_array(getattr(value, name), shape)
            for name in self._FLAG_NAMES
        }
        status = getattr(value, "status")
        cp = self.cp
        if (
            not isinstance(status, cp.ndarray)
            or status.dtype != cp.int32
            or status.ndim != 1
            or tuple(status.shape) != shape
            or not status.flags.c_contiguous
            or int(status.device.id) != self.device_id
        ):
            raise TypeError(
                "status must be a resident contiguous 1-D int32 CuPy array"
            )
        return outputs, flags, status

    def evaluate(
        self,
        active,
        depth_old,
        depth_new,
        u,
        v,
        raw_inverse_radius,
        metric_ksi_sum,
        metric_eta_sum,
        coriolis,
        raw_chezy,
        spiral_intensity,
        bed_x,
        bed_y,
        *,
        dryflc: float = 1.0e-3,
        rmincf: float = 2.5,
        chzmin: float = 20.0,
        gravity: float = 9.81,
        von_karman: float = 0.41,
        espir: float = 1.0,
        betac: float = 0.0,
        epsilon_value: float = np.finfo(np.float64).eps,
        reject_invalid: bool = True,
        out: ResidentFlow2DHSecondaryResult | None = None,
    ) -> ResidentFlow2DHSecondaryResult:
        """Evaluate one local FLOW operator state per resident array entry.

        ``out`` is optional for the isolated API and mandatory by convention
        in the live route.  Supplying it performs the launch wholly in place.
        """
        cp = self.cp
        arrays = {
            name: self._fp64_array(name, value)
            for name, value in zip(
                self._ARRAY_NAMES,
                (
                    depth_old,
                    depth_new,
                    u,
                    v,
                    raw_inverse_radius,
                    metric_ksi_sum,
                    metric_eta_sum,
                    coriolis,
                    raw_chezy,
                    spiral_intensity,
                    bed_x,
                    bed_y,
                ),
            )
        }
        shape = tuple(arrays["depth_old"].shape)
        for name in self._ARRAY_NAMES[1:]:
            self._fp64_array(name, arrays[name], shape)
        active = self._active_array(active, shape)
        controls = {
            name: self._finite_scalar(name, value)
            for name, value in (
                ("dryflc", dryflc),
                ("rmincf", rmincf),
                ("chzmin", chzmin),
                ("gravity", gravity),
                ("von_karman", von_karman),
                ("espir", espir),
                ("betac", betac),
                ("epsilon_value", epsilon_value),
            )
        }
        if controls["dryflc"] < 0.0:
            raise ValueError("dryflc must be nonnegative")
        if controls["rmincf"] <= 0.0:
            raise ValueError("rmincf must be positive")
        if controls["chzmin"] <= 0.0:
            raise ValueError("chzmin must be positive")
        if controls["gravity"] <= 0.0:
            raise ValueError("gravity must be positive")
        if controls["von_karman"] <= 0.0:
            raise ValueError("von_karman must be positive")
        if controls["espir"] < 0.0:
            raise ValueError("espir must be nonnegative")
        if controls["epsilon_value"] < 0.0:
            raise ValueError("epsilon_value must be nonnegative")
        if not isinstance(reject_invalid, (bool, np.bool_)):
            raise TypeError("reject_invalid must be boolean")

        count = int(shape[0])
        with cp.cuda.Device(self.device_id):
            result = self.allocate_result(count) if out is None else out
            outputs, flags, status = self._result_arrays(result, shape)
            if count:
                self.kernel(
                    ((count + 255) // 256,),
                    (256,),
                    (
                        np.int64(count),
                        active,
                        *[arrays[name] for name in self._ARRAY_NAMES],
                        *[
                            np.float64(controls[name])
                            for name in (
                                "dryflc",
                                "rmincf",
                                "chzmin",
                                "gravity",
                                "von_karman",
                                "espir",
                                "betac",
                                "epsilon_value",
                            )
                        ],
                        *[outputs[name] for name in self._OUTPUT_NAMES],
                        *[flags[name] for name in self._FLAG_NAMES],
                        status,
                    ),
                )
            if bool(reject_invalid) and count:
                invalid_count = int(cp.count_nonzero(status).item())
                if invalid_count:
                    values, counts = cp.unique(status, return_counts=True)
                    summary = {
                        int(value): int(number)
                        for value, number in zip(values.get().tolist(), counts.get().tolist())
                        if int(value) != STATUS_OK
                    }
                    raise ValueError(
                        "CUDA FLOW 2DH secondary operator rejected "
                        f"{invalid_count} state(s): {summary}"
                    )
        return result


class ResidentClassicFlow2DHSecondary:
    """Persistent fixed-grid live owner for classic equilibrium secondary flow.

    The caller authenticates the v56 envelope before construction.  All stage
    arrays are allocated here once, remain on one CUDA device, and are updated
    in place.  This class neither selects a backend nor substitutes a CPU path.
    """

    LIVE_KERNELS = (
        "flow2dh_lsq_curvature",
        "flow2dh_rotate_cell_bed",
        "flow2dh_rotate_boundary_bed",
    )

    def __init__(self, data, geometry, *, source_folder,
                 execution_identity):
        import cupy as cp

        from .forcing_profile import forcing_identity
        from .hydraulic_roughness import HydraulicRoughnessCUDA

        self.cp = cp
        self.device_id = int(cp.cuda.Device().id)
        self.identity = dict(execution_identity)
        self.forcing = forcing_identity(data)
        self.nc = int(data["nc"])
        self.nf = int(data["nf"])
        self.ng = int(data["nfrac"])
        self.na = int(np.asarray(data["faces"]).size)
        self.dry = float(data["dry"])
        self.gravity = float(data["grav"])
        self.dx = float(data["dx"])
        self.dy = float(data["dy"])
        self.rmincf = float(self.identity["rmincf"])
        self.chzmin = float(self.identity["chzmin"])
        self.espir = float(self.identity["espir"])
        self.betac = float(self.identity["betac"])
        self.coriolis_value = float(
            self.identity["coriolis_constant_s_inverse"])
        if (self.nc <= 0 or self.nf <= 0 or self.ng <= 0 or self.na <= 0 or
                not np.isfinite((self.dry, self.gravity, self.dx, self.dy,
                                 self.rmincf, self.chzmin, self.espir,
                                 self.betac, self.coriolis_value)).all() or
                self.dry < 0.0 or self.gravity <= 0.0 or
                self.dx <= 0.0 or self.dy <= 0.0 or self.rmincf <= 0.0 or
                self.chzmin <= 0.0 or self.espir < 0.0 or
                self.betac != 0.0 or self.coriolis_value != 0.0):
            raise ValueError("Invalid live classic FLOW secondary controls")

        folder = Path(source_folder).resolve()
        paths = {
            "point": folder / "flow2dh_secondary_operator.cu",
            "live": folder / "flow2dh_secondary_live.cu",
            "roughness": folder / "hydraulic_roughness.cu",
        }
        if any(not path.is_file() or path.is_symlink()
               for path in paths.values()):
            raise ValueError(
                "Archived classic FLOW secondary CUDA inventory differs")
        self.point = Flow2DHSecondaryCUDA(paths["point"])
        self.roughness = HydraulicRoughnessCUDA(paths["roughness"])
        with cp.cuda.Device(self.device_id):
            self.module = cp.RawModule(
                code=paths["live"].read_text(encoding="utf-8"),
                options=CUDA_OPTIONS, name_expressions=self.LIVE_KERNELS)
            self.kernels = {
                name: self.module.get_function(name)
                for name in self.LIVE_KERNELS
            }
        binding = data.get("delft_forcing_binding")
        if not isinstance(binding, dict):
            raise ValueError("Classic FLOW secondary lacks typed roughness binding")
        self.roughness_mode = self.roughness._mode(
            binding["hydraulic_roughness_mode"])
        self.roughness_waqua = int(bool(
            self.forcing["white_colebrook_chzk2d"]))
        self.roughness_work_depth = float(
            self.forcing["roughness_minimum_work_depth_m"])
        self.geometry = geometry
        required = {
            "bed_ls_matrix": (4, self.nc),
            "bed_ls_delta": (2, self.na),
            "bed_ls_weight": (self.na,),
            "owner": (self.nf,),
            "neighbour": (self.nf,),
            "offset": (self.nc + 1,),
            "faces": (self.na,),
            "signs": (self.na,),
            "manning_cell": (self.nc,),
        }
        for name, shape in required.items():
            value = geometry.get(name)
            if (not isinstance(value, cp.ndarray) or value.shape != shape or
                    not value.flags.c_contiguous or
                    int(value.device.id) != self.device_id):
                raise ValueError(
                    "Classic FLOW secondary geometry differs: " + name)

        self.result = self.point.allocate_result(self.nc)
        self.u = cp.zeros(self.nc, dtype=cp.float64)
        self.v = cp.zeros(self.nc, dtype=cp.float64)
        self.raw_curvature = cp.zeros(self.nc, dtype=cp.float64)
        self.raw_chezy = cp.zeros(self.nc, dtype=cp.float64)
        self.roughness_z0 = cp.zeros(self.nc, dtype=cp.float64)
        self.roughness_drag = cp.zeros(self.nc, dtype=cp.float64)
        # Open-boundary equilibrium/stage roles use the hydraulic trace state,
        # not the adjacent cell depth.  Keep a resident owner-parameter view
        # and typed roughness workspace so Manning and White-Colebrook Chezy
        # are refreshed at the trace depth without allocating in each stage.
        self.boundary_roughness_parameter = cp.ascontiguousarray(
            geometry["manning_cell"][geometry["owner"]])
        self.boundary_raw_chezy = cp.zeros(self.nf, dtype=cp.float64)
        self.boundary_roughness_z0 = cp.zeros(self.nf, dtype=cp.float64)
        self.boundary_roughness_drag = cp.zeros(self.nf, dtype=cp.float64)
        self.boundary_roughness_status = cp.zeros(self.nf, dtype=cp.int32)
        self.metric_ksi_sum = cp.full(
            self.nc, 2.0 * self.dx, dtype=cp.float64)
        self.metric_eta_sum = cp.full(
            self.nc, 2.0 * self.dy, dtype=cp.float64)
        self.coriolis = cp.full(
            self.nc, self.coriolis_value, dtype=cp.float64)
        self.zero = cp.zeros(self.nc, dtype=cp.float64)
        self.operator_active = cp.zeros(self.nc, dtype=cp.uint8)
        self.curvature_status = cp.zeros(self.nc, dtype=cp.int32)
        self.roughness_status = cp.zeros(self.nc, dtype=cp.int32)
        self.cell_rotation_status = cp.zeros(self.nc, dtype=cp.int32)
        self.boundary_rotation_status = cp.zeros(self.nf, dtype=cp.int32)
        self.status = cp.zeros(self.nc, dtype=cp.int32)

    def _launch(self, name, count, args):
        if count:
            self.kernels[name](((count + 255) // 256,), (256,), args)

    def refresh(self, q, active):
        """Refresh curvature, typed Chezy and FLOW equilibrium for one stage."""
        cp = self.cp
        if (not isinstance(q, cp.ndarray) or q.dtype != cp.float64 or
                q.shape != (3, self.nc) or not q.flags.c_contiguous or
                not isinstance(active, cp.ndarray) or active.dtype != cp.int32 or
                active.shape != (self.nc,) or not active.flags.c_contiguous):
            raise ValueError("Classic FLOW secondary stage arrays differ")
        a = self.geometry
        self._launch("flow2dh_lsq_curvature", self.nc, (
            np.int32(self.nc), np.int32(self.nf), np.int32(self.na), q,
            active, a["bed_ls_matrix"], a["bed_ls_delta"],
            a["bed_ls_weight"], a["owner"], a["neighbour"], a["offset"],
            a["faces"], a["signs"], np.float64(self.dry), self.u, self.v,
            self.raw_curvature, self.operator_active,
            self.curvature_status))
        if self.nc:
            self.roughness.kernels["hydraulic_roughness_state"](
                ((self.nc + 255) // 256,), (256,), (
                    np.int64(self.nc), np.int32(self.roughness_mode), q[0],
                    a["manning_cell"], np.float64(self.gravity),
                    np.float64(self.roughness_work_depth),
                    np.int32(self.roughness_waqua), self.raw_chezy,
                    self.roughness_z0, self.roughness_drag,
                    self.roughness_status))
        self.point.evaluate(
            self.operator_active, q[0], q[0], self.u, self.v,
            self.raw_curvature, self.metric_ksi_sum, self.metric_eta_sum,
            self.coriolis, self.raw_chezy, self.zero, self.zero, self.zero,
            dryflc=self.dry, rmincf=self.rmincf, chzmin=self.chzmin,
            gravity=self.gravity, espir=self.espir, betac=self.betac,
            reject_invalid=False, out=self.result)
        cp.maximum(self.curvature_status, self.roughness_status,
                   out=self.status)
        cp.maximum(self.status, self.result.status, out=self.status)
        return self.status

    def rotate_cells(self, raw_bed):
        cp = self.cp
        if (not isinstance(raw_bed, cp.ndarray) or
                raw_bed.dtype != cp.float64 or
                raw_bed.shape != (2, self.ng, self.nc) or
                not raw_bed.flags.c_contiguous):
            raise ValueError("Classic FLOW raw cell bed channel differs")
        self._launch("flow2dh_rotate_cell_bed", self.nc, (
            np.int32(self.nc), np.int32(self.ng), self.u, self.v,
            self.raw_chezy, self.result.equilibrium_target,
            self.operator_active, np.float64(self.espir),
            np.float64(self.gravity), raw_bed,
            self.cell_rotation_status))
        return self.cell_rotation_status

    def rotate_boundaries(self, trace, owner, role, raw_boundary_bed,
                          rotated_cell_bed):
        cp = self.cp
        if (not isinstance(trace, cp.ndarray) or trace.dtype != cp.float64 or
                trace.ndim != 2 or trace.shape[0] < 3 or
                trace.shape[1] != self.nf or not trace.flags.c_contiguous or
                not isinstance(owner, cp.ndarray) or owner.dtype != cp.int32 or
                owner.shape != (self.nf,) or not owner.flags.c_contiguous or
                int(owner.device.id) != self.device_id or
                int(owner.data.ptr) != int(self.geometry["owner"].data.ptr) or
                not isinstance(role, cp.ndarray) or role.dtype != cp.int32 or
                role.shape != (self.nf,) or not role.flags.c_contiguous or
                int(role.device.id) != self.device_id or
                not isinstance(raw_boundary_bed, cp.ndarray) or
                raw_boundary_bed.shape != (2, self.ng, self.nf) or
                raw_boundary_bed.dtype != cp.float64 or
                not raw_boundary_bed.flags.c_contiguous or
                not isinstance(rotated_cell_bed, cp.ndarray) or
                rotated_cell_bed.shape != (2, self.ng, self.nc) or
                rotated_cell_bed.dtype != cp.float64 or
                not rotated_cell_bed.flags.c_contiguous):
            raise ValueError("Classic FLOW raw boundary bed channel differs")
        if self.nf:
            self.roughness.kernels["hydraulic_roughness_state"](
                ((self.nf + 255) // 256,), (256,), (
                    np.int64(self.nf), np.int32(self.roughness_mode), trace[0],
                    self.boundary_roughness_parameter,
                    np.float64(self.gravity),
                    np.float64(self.roughness_work_depth),
                    np.int32(self.roughness_waqua), self.boundary_raw_chezy,
                    self.boundary_roughness_z0, self.boundary_roughness_drag,
                    self.boundary_roughness_status))
        self._launch("flow2dh_rotate_boundary_bed", self.nf, (
            np.int32(self.nc), np.int32(self.nf), np.int32(self.ng), trace,
            owner, role, self.boundary_raw_chezy,
            self.boundary_roughness_status,
            self.result.boundary_equilibrium,
            rotated_cell_bed, np.float64(self.dry),
            np.float64(self.espir), np.float64(self.gravity),
            raw_boundary_bed, self.boundary_rotation_status))
        return self.boundary_rotation_status

    def storage_identity(self):
        """Expose allocation identity for bounded no-churn regression gates."""
        values = [self.u, self.v, self.raw_curvature, self.raw_chezy,
                  self.roughness_z0, self.roughness_drag,
                  self.boundary_roughness_parameter,
                  self.boundary_raw_chezy, self.boundary_roughness_z0,
                  self.boundary_roughness_drag,
                  self.boundary_roughness_status,
                  self.metric_ksi_sum, self.metric_eta_sum, self.coriolis,
                  self.zero, self.operator_active, self.curvature_status,
                  self.roughness_status, self.cell_rotation_status,
                  self.boundary_rotation_status, self.status]
        values.extend(getattr(self.result, name)
                      for name in (*self.point._OUTPUT_NAMES,
                                   *self.point._FLAG_NAMES, "status"))
        return tuple((id(value), int(value.data.ptr), value.shape,
                      value.dtype.str, value.strides, int(value.device.id))
                     for value in values)

    def diagnostic(self):
        return {
            "execution_fingerprint": self.identity["fingerprint"],
            "raw_curvature": self.raw_curvature,
            "capped_inverse_radius": self.result.capped_inverse_radius,
            "equilibrium_intensity": self.result.equilibrium_target,
            "boundary_equilibrium_intensity":
                self.result.boundary_equilibrium,
            "raw_chezy": self.raw_chezy,
            "effective_source_chezy": self.result.effective_chezy,
            "cell_rotation_status": self.cell_rotation_status,
            "boundary_rotation_status": self.boundary_rotation_status,
            "status": self.status,
            "metric_ksi_sum": self.metric_ksi_sum,
            "metric_eta_sum": self.metric_eta_sum,
        }

__all__ = [
    "CUDA_OPTIONS",
    "Flow2DHSecondaryCUDA",
    "ResidentClassicFlow2DHSecondary",
    "KERNEL_NAME",
    "ResidentFlow2DHSecondaryResult",
    "SOURCE_ANCHORS",
    "STATUS_INVALID_GEOMETRY",
    "STATUS_INVALID_INPUT",
    "STATUS_INVALID_OUTPUT",
    "STATUS_INVALID_RELAXATION",
    "STATUS_OK",
]
