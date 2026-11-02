"""Resident CUDA typed-roughness viscosity-equilibrated inlet operator.

The section solve remains on the selected GPU during implicit residual
evaluations.  One small serial tridiagonal solve runs per inlet while separate
inlets execute as independent CUDA blocks; there is no CPU numerical fallback.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np

from .hydraulic_roughness import CUDA_OPTIONS, ROUGHNESS_MODES


KERNEL_NAME = "viscosity_equilibrated_profiles"
TRACE_KERNEL_NAME = "viscosity_equilibrated_traces"
SUCCESS = 0
INVALID = 1
NO_WET_SECTION = 2
NO_DRAG = 3
NONCONVERGED = 4


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _vector(value, dtype, name):
    array = np.asarray(value)
    _require(array.ndim == 1 and array.dtype == dtype and
             array.flags.c_contiguous, f"{name} must be a contiguous {dtype} vector")
    return array


def _array_identity(value):
    array = np.ascontiguousarray(value)
    return dict(shape=[int(item) for item in array.shape], dtype=array.dtype.str,
                sha256=hashlib.sha256(array.tobytes(order="C")).hexdigest())


class ViscosityEquilibratedInletCUDA:
    """Compile and retain one authenticated static section topology."""

    def __init__(self, *, width, center, owner, face, offsets, nc, nf,
                 source_path: str | Path | None = None,
                 roughness_source_path: str | Path | None = None):
        import cupy as cp

        width = _vector(width, np.dtype(np.float64), "Section width")
        center = _vector(center, np.dtype(np.float64), "Section centre")
        owner = _vector(owner, np.dtype(np.int32), "Section owner")
        face = _vector(face, np.dtype(np.int32), "Section face")
        offsets = _vector(offsets, np.dtype(np.int32), "Section offsets")
        count = len(width)
        _require(center.shape == owner.shape == face.shape == (count,),
                 "Section vectors must have identical lengths")
        _require(isinstance(nc, int) and not isinstance(nc, bool) and nc > 0 and
                 isinstance(nf, int) and not isinstance(nf, bool) and nf > 0,
                 "Positive integer cell/face counts are required")
        nb = len(offsets) - 1
        _require(nb > 0 and offsets[0] == 0 and offsets[-1] == count and
                 np.all(np.diff(offsets.astype(np.int64)) > 0),
                 "Each viscosity-equilibrated boundary needs a nonempty CSR interval")
        _require(np.isfinite(width).all() and np.all(width > 0.0) and
                 np.isfinite(center).all(), "Section geometry must be finite and positive")
        _require(np.all((owner >= 0) & (owner < nc)) and
                 np.all((face >= 0) & (face < nf)) and len(np.unique(face)) == count,
                 "Section owner/face indices are invalid or repeated")

        sorted_width = np.empty_like(width)
        sorted_center = np.empty_like(center)
        sorted_owner = np.empty_like(owner)
        sorted_face = np.empty_like(face)
        touching = np.zeros(count, dtype=np.int32)
        for boundary in range(nb):
            first, last = int(offsets[boundary]), int(offsets[boundary + 1])
            order = np.argsort(center[first:last], kind="stable") + first
            sorted_width[first:last] = width[order]
            sorted_center[first:last] = center[order]
            sorted_owner[first:last] = owner[order]
            sorted_face[first:last] = face[order]
            scale = max(1.0, float(np.max(np.abs(sorted_center[first:last]))),
                        float(np.max(sorted_width[first:last])))
            tolerance = 512.0 * np.finfo(np.float64).eps * scale
            if last - first > 1:
                distance = np.diff(sorted_center[first:last])
                gap = distance - .5 * (
                    sorted_width[first:last - 1] + sorted_width[first + 1:last]
                )
                _require(np.all(distance > 0.0) and np.all(gap >= -tolerance),
                         "Section fragments overlap or have repeated centres")
                touching[first:last - 1] = gap <= tolerance

        source = Path(source_path or Path(__file__).with_suffix(".cu"))
        roughness_source = Path(
            roughness_source_path or Path(__file__).with_name("hydraulic_roughness.cu")
        )
        _require(source.is_file() and roughness_source.is_file() and
                 not source.is_symlink() and not roughness_source.is_symlink(),
                 "Viscous-inlet CUDA sources are absent or indirect")
        code = (roughness_source.read_text(encoding="utf-8") + "\n" +
                source.read_text(encoding="utf-8"))
        self.module = cp.RawModule(code=code, options=CUDA_OPTIONS,
                                   name_expressions=(KERNEL_NAME, TRACE_KERNEL_NAME))
        self.kernel = self.module.get_function(KERNEL_NAME)
        self.trace_kernel = self.module.get_function(TRACE_KERNEL_NAME)
        self.cp, self.nc, self.nf, self.nb, self.count = cp, nc, nf, nb, count
        self.static = {
            "width": cp.asarray(sorted_width), "center": cp.asarray(sorted_center),
            "owner": cp.asarray(sorted_owner), "face": cp.asarray(sorted_face),
            "offsets": cp.asarray(offsets), "touching": cp.asarray(touching),
        }
        self.face_speed = cp.zeros(nf, dtype=cp.float64)
        self.driving = cp.zeros(nb, dtype=cp.float64)
        self.discharge_residual = cp.zeros(nb, dtype=cp.float64)
        self.acceleration_residual = cp.zeros(nb, dtype=cp.float64)
        self.iterations = cp.zeros(nb, dtype=cp.int32)
        self.status = cp.zeros(nb, dtype=cp.int32)
        self.face_component = cp.full(nf, -1, dtype=cp.int32)
        self.wet_slot = cp.empty(count, dtype=cp.int32)
        names = ("width", "depth", "beta", "weight", "edge", "u", "rhs",
                 "diagonal", "lower", "upper", "cprime", "dprime", "primary",
                 "sensitivity", "delta")
        self.work = {name: cp.empty(count, dtype=cp.float64) for name in names}
        document = dict(
            schema=1, backend="cuda", numerical_fallback="none",
            algorithm="resident per-boundary bordered-Newton/tridiagonal",
            width=_array_identity(sorted_width), center=_array_identity(sorted_center),
            owner=_array_identity(sorted_owner), face=_array_identity(sorted_face),
            offsets=_array_identity(offsets), touching=_array_identity(touching),
            cuda_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            roughness_source_sha256=hashlib.sha256(roughness_source.read_bytes()).hexdigest(),
        )
        preimage = json.dumps(document, sort_keys=True, separators=(",", ":"),
                              ensure_ascii=True).encode("ascii")
        self.identity = MappingProxyType({
            **document, "fingerprint": hashlib.sha256(preimage).hexdigest()
        })

    @classmethod
    def from_native(cls, data, **kwargs):
        """Build static sorted sections from actual native exporter arrays."""
        nc, nf = int(data["nc"]), int(data["nf"])
        offsets = np.ascontiguousarray(data["boundary_offset"], dtype=np.int32)
        faces = np.ascontiguousarray(data["boundary_faces"], dtype=np.int32)
        owner_all = np.asarray(data["owner"])
        normal = np.asarray(data["normal"])
        midpoint = np.asarray(data["mid"])
        length = np.asarray(data["length"])
        _require(normal.shape == midpoint.shape == (2, nf) and
                 owner_all.shape == length.shape == (nf,),
                 "Native section geometry dimensions differ")
        centers = np.empty(len(faces), dtype=np.float64)
        widths = np.empty(len(faces), dtype=np.float64)
        owners = np.empty(len(faces), dtype=np.int32)
        for boundary in range(len(offsets) - 1):
            first, last = int(offsets[boundary]), int(offsets[boundary + 1])
            section_faces = faces[first:last]
            tangent = np.array([-normal[1, section_faces[0]],
                                normal[0, section_faces[0]]], dtype=np.float64)
            if tangent[0] < 0.0 or (tangent[0] == 0.0 and tangent[1] < 0.0):
                tangent = -tangent
            centers[first:last] = midpoint[:, section_faces].T @ tangent
            widths[first:last] = length[section_faces]
            owners[first:last] = owner_all[section_faces]
        return cls(width=np.ascontiguousarray(widths),
                   center=np.ascontiguousarray(centers),
                   owner=np.ascontiguousarray(owners), face=faces,
                   offsets=offsets, nc=nc, nf=nf, **kwargs)

    @staticmethod
    def _mode(mode):
        value = ROUGHNESS_MODES.get(mode, mode)
        if isinstance(value, bool) or value not in (1, 2, 3):
            raise ValueError("Roughness mode must be chezy, manning, or white_colebrook")
        return int(value)

    def solve(self, q, parameter, active, discharge, *, mode,
              viscosity, gravity=9.81, dry_tolerance=1.0e-4,
              minimum_work_depth=0.0, waqua_white_colebrook=False):
        """Solve using resident state/forcing arrays and return resident outputs."""
        cp = self.cp
        if (not isinstance(q, cp.ndarray) or q.dtype != cp.float64 or
                q.ndim != 2 or q.shape[0] < 3 or q.shape[1] != self.nc or
                not q.flags.c_contiguous):
            raise ValueError("Inlet state must be resident contiguous FP64 (state, cell)")
        if (not isinstance(parameter, cp.ndarray) or parameter.dtype != cp.float64 or
                parameter.shape != (self.nc,) or not parameter.flags.c_contiguous):
            raise ValueError("Raw roughness parameter must be a resident contiguous FP64 cell vector")
        if (not isinstance(active, cp.ndarray) or active.dtype != cp.int32 or
                active.shape != (self.nb,) or not active.flags.c_contiguous or
                not isinstance(discharge, cp.ndarray) or discharge.dtype != cp.float64 or
                discharge.shape != (self.nb,) or not discharge.flags.c_contiguous):
            raise ValueError("Active flags and discharge must be resident canonical boundary vectors")
        controls = np.asarray((viscosity, gravity, dry_tolerance,
                               minimum_work_depth), dtype=np.float64)
        _require(np.isfinite(controls).all() and viscosity >= 0.0 and
                 gravity > 0.0 and dry_tolerance > 0.0 and
                 minimum_work_depth >= 0.0,
                 "Viscous-inlet controls are invalid")
        a, w = self.static, self.work
        self.kernel((self.nb,), (1,), (
            q, parameter, a["owner"], a["face"], a["width"], a["center"],
            a["touching"], a["offsets"], active, discharge,
            np.int64(self.nc), np.int64(self.nf), np.int32(self.nb),
            np.int32(self._mode(mode)), np.int32(bool(waqua_white_colebrook)),
            np.float64(viscosity), np.float64(gravity),
            np.float64(dry_tolerance), np.float64(minimum_work_depth),
            self.face_speed, self.driving, self.discharge_residual,
            self.acceleration_residual, self.iterations, self.status,
            self.face_component, self.wet_slot, w["width"], w["depth"],
            w["beta"], w["weight"], w["edge"], w["u"], w["rhs"],
            w["diagonal"], w["lower"], w["upper"], w["cprime"],
            w["dprime"], w["primary"], w["sensitivity"], w["delta"],
        ))
        if bool(cp.any(self.status != SUCCESS).item()):
            codes = cp.asnumpy(self.status)
            raise ValueError(f"CUDA viscosity-equilibrated inlet failed with status {codes.tolist()}")
        return MappingProxyType(dict(
            face_speed=self.face_speed, driving=self.driving,
            discharge_residual=self.discharge_residual,
            acceleration_residual=self.acceleration_residual,
            iterations=self.iterations, status=self.status,
            face_component=self.face_component,
            identity=self.identity,
            automatic_CPU_fallback=False,
        ))

    def apply_traces(self, q, trace, *, normal, owner, bid, active,
                     speed_limit, face_status, boundary_status=None,
                     dry_tolerance=1.0e-4):
        """Overwrite only selected mode-6 faces with the resident profile."""
        cp = self.cp
        boundary_status = self.status if boundary_status is None else boundary_status
        arrays = ((normal, cp.float64, (2, self.nf), "normal"),
                  (owner, cp.int32, (self.nf,), "owner"),
                  (bid, cp.int32, (self.nf,), "boundary ID"),
                  (active, cp.int32, (self.nb,), "active flag"),
                  (face_status, cp.int32, (self.nf,), "face status"),
                  (boundary_status, cp.int32, (self.nb,), "boundary status"))
        if (not isinstance(q, cp.ndarray) or q.dtype != cp.float64 or
                q.shape[0] < 3 or q.shape[1] != self.nc or not q.flags.c_contiguous):
            raise ValueError("Trace composition requires the resident canonical state")
        # Coupled reconstruction appends sediment trace rows after the three
        # hydraulic rows.  The inlet owns only rows 0:3 and must preserve all
        # appended state exactly.
        if (not isinstance(trace, cp.ndarray) or trace.dtype != cp.float64 or
                trace.ndim != 2 or trace.shape[0] < 3 or
                trace.shape[1] != self.nf or not trace.flags.c_contiguous):
            raise ValueError("Trace composition trace array differs")
        for value, dtype, shape, name in arrays:
            if (not isinstance(value, cp.ndarray) or value.dtype != dtype or
                    value.shape != shape or not value.flags.c_contiguous):
                raise ValueError(f"Trace composition {name} array differs")
        controls = np.asarray((speed_limit, dry_tolerance), dtype=np.float64)
        _require(np.isfinite(controls).all() and speed_limit > 0.0 and
                 dry_tolerance > 0.0, "Trace composition controls are invalid")
        self.trace_kernel(((self.nf + 127) // 128,), (128,), (
            q, self.face_speed, owner, bid, active, normal,
            np.int64(self.nc), np.int64(self.nf), np.int32(self.nb),
            np.float64(dry_tolerance), np.float64(speed_limit), trace,
            boundary_status, face_status,
        ))
        return trace


__all__ = [
    "ViscosityEquilibratedInletCUDA", "SUCCESS", "INVALID",
    "NO_WET_SECTION", "NO_DRAG", "NONCONVERGED",
]
