"""Authenticated reflective closure for the bounded v98 joined-leaf gate.

The canonical component record carries real mode-9/mode-4 physical boundaries.
Changing the implicit controller's acoustic ``boundary_mode`` array does not
disable those owners.  This additive validation binding replaces every exact
source kind-3 face with the ordinary kind-2 reflective branch *before* the sole
augmented f-wave call and prevents the physical trace/pressure owners from
running.  Frozen v96/v97 implementations and source inputs remain unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import json
from pathlib import Path
from types import MethodType

import numpy as np


HERE = Path(__file__).resolve().parent
CUDA_SOURCE = HERE / "coupled_amr_closed_boundary_v98.cu"
CUDA_OPTIONS = ("--std=c++11", "--fmad=false", "--prec-div=true",
                "--prec-sqrt=true", "--ftz=false")
SCOPE_ID = "braids-q4-coupled-amr-closed-boundary-gpu-v98-v1"
SEMANTIC_ID = (SCOPE_ID +
               ":authenticated-kind3-to-kind2-before-sole-fwave")


class CoupledAMRClosedBoundaryGPUV98Error(ValueError):
    pass


def _require(condition, message):
    if not condition:
        raise CoupledAMRClosedBoundaryGPUV98Error(message)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _digest_array(value):
    array = np.ascontiguousarray(np.asarray(value))
    body = (array.dtype.str.encode("ascii") + b"|" +
            json.dumps(array.shape, separators=(",", ":")).encode("ascii") +
            b"|" + array.tobytes(order="C"))
    return hashlib.sha256(body).hexdigest()


def _token(value):
    payload = {
        "scope": SCOPE_ID,
        "semantic_id": value.semantic_id,
        "hydraulic_execution_fingerprint":
            value.hydraulic_execution_fingerprint,
        "source_boundary_owner_mode": value.source_boundary_owner_mode.tolist(),
        "source_controller_boundary_mode_sha256":
            _digest_array(value.source_controller_boundary_mode),
        "source_bid_sha256": _digest_array(value.source_bid),
        "physical_faces": value.physical_faces.tolist(),
        "source_forcing_sha256": value.source_forcing_sha256,
        "source_bed_offset_sha256": value.source_bed_offset_sha256,
        "python_source_sha256": value.python_source_sha256,
        "cuda_source_sha256": value.cuda_source_sha256,
        "source_kind3_face_count": value.source_kind3_face_count,
        "source_physical_owner_count": value.source_physical_owner_count,
        "target_reflective_kind2_face_count":
            value.target_reflective_kind2_face_count,
        "target_physical_owner_count": value.target_physical_owner_count,
        "target_trace_owner_count": value.target_trace_owner_count,
        "target_pressure_owner_count": value.target_pressure_owner_count,
        "target_boundary_ledger_owner_count":
            value.target_boundary_ledger_owner_count,
        "controller_zero_is_closure_evidence":
            value.controller_zero_is_closure_evidence,
        "explicit_transformation_count": value.explicit_transformation_count,
        "backend": value.backend,
    }
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest()


def _frozen(value, dtype=None):
    result = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ReflectiveBoundaryClosureReceiptGPUV98:
    hydraulic_execution_fingerprint: str
    source_boundary_owner_mode: np.ndarray
    source_controller_boundary_mode: np.ndarray
    source_bid: np.ndarray
    physical_faces: np.ndarray
    source_forcing_sha256: str
    source_bed_offset_sha256: str
    python_source_sha256: str
    cuda_source_sha256: str
    source_kind3_face_count: int
    source_physical_owner_count: int
    target_reflective_kind2_face_count: int
    target_physical_owner_count: int
    target_trace_owner_count: int
    target_pressure_owner_count: int
    target_boundary_ledger_owner_count: int
    controller_zero_is_closure_evidence: bool
    explicit_transformation_count: int
    backend: str
    semantic_id: str
    receipt_sha256: str


class ResidentReflectiveBoundaryClosureGPUV98:
    """Instance-only adapter around the exact live physical boundary owner."""

    def __init__(self, model, data, *, hydraulic_execution_fingerprint):
        import cupy as cp
        _require(type(hydraulic_execution_fingerprint) is str and
                 len(hydraulic_execution_fingerprint) == 64,
                 "v98 closed binding hydraulic execution differs")
        boundary = getattr(model, "boundary", None)
        controller = getattr(model, "implicit_midpoint", None)
        reconstruction = getattr(model, "reconstruction", None)
        _require(boundary is not None and controller is not None and
                 reconstruction is not None and
                 controller.solver is model,
                 "v98 closed binding live owner differs")
        _require(not hasattr(boundary, "_v98_reflective_closure_binding"),
                 "v98 closed binding is already installed")
        neighbour = np.ascontiguousarray(data["neighbour"], dtype=np.int32)
        bid = np.ascontiguousarray(data["bid"], dtype=np.int32)
        modes = np.ascontiguousarray(data["mode"], dtype=np.int32)
        controller_mode = np.ascontiguousarray(
            data["boundary_mode"], dtype=np.int32)
        faces = np.flatnonzero((neighbour < 0) & (bid >= 0)).astype(np.int32)
        offsets = np.asarray(data["boundary_offset"], dtype=np.int32)
        listed = np.asarray(data["boundary_faces"], dtype=np.int32)
        _require(faces.size > 0 and offsets.shape == (modes.size+1,) and
                 offsets[0] == 0 and offsets[-1] == listed.size and
                 np.array_equal(np.sort(listed), faces) and
                 np.all(controller_mode[faces] == modes[bid[faces]]) and
                 np.all(neighbour[faces] < 0),
                 "v98 closed binding source physical topology differs")
        _require(np.array_equal(cp.asnumpy(boundary.arrays["mode"]), modes) and
                 np.array_equal(cp.asnumpy(boundary.arrays["bid"]), bid) and
                 np.array_equal(cp.asnumpy(
                     reconstruction.geometry["bid"]), bid),
                 "v98 closed binding resident physical topology differs")
        # The old fixture wrote zeros here.  Record explicitly that these are
        # only target acoustic modes and never evidence that the physical
        # owner was disabled.
        _require(np.all(cp.asnumpy(controller.boundary_mode)[faces] == 0),
                 "v98 closed binding target acoustic modes differ")
        source_forcing = np.asarray(data["forcing"], dtype=np.float64)
        _require(source_forcing.ndim == 2 and source_forcing.shape[0] >= 2,
                 "v98 closed binding source forcing differs")
        # The hydraulic boundary owner receives the first two rows of the
        # broader coupled forcing table (Q/stage); scalar and sediment rows
        # remain independently owned by their later operators.
        forcing = np.ascontiguousarray(source_forcing[:2], dtype=np.float64)
        _require(np.array_equal(cp.asnumpy(boundary.arrays["forcing"]), forcing),
                 "v98 closed binding resident forcing differs")
        bed_offset = np.ascontiguousarray(
            data.get("bedoffset", data.get("boundary_bed_offset")),
            dtype=np.float64)
        _require(bed_offset.shape == neighbour.shape,
                 "v98 closed binding source bed offsets differ")

        self.cp = cp
        self.model = model
        self.boundary = boundary
        self.controller = controller
        self.reconstruction = reconstruction
        self.faces_host = faces
        self.faces = cp.asarray(faces)
        self.count = int(faces.size)
        self.apply_count = 0
        self.pressure_count = 0
        self.begin_count = 0
        self._installed = False
        self._original_begin = boundary.begin_trial
        self._original_apply = boundary.apply
        self._original_pressure = boundary.apply_pressure
        code = CUDA_SOURCE.read_text(encoding="utf-8")
        names = ("v98_close_physical_faces",
                 "v98_validate_reflective_faces")
        module = cp.RawModule(code=code, options=CUDA_OPTIONS,
                             name_expressions=names)
        self._module = module
        self._close_kernel = module.get_function(names[0])
        self._validate_kernel = module.get_function(names[1])
        receipt = ReflectiveBoundaryClosureReceiptGPUV98(
            hydraulic_execution_fingerprint=hydraulic_execution_fingerprint,
            source_boundary_owner_mode=_frozen(modes),
            source_controller_boundary_mode=_frozen(controller_mode),
            source_bid=_frozen(bid), physical_faces=_frozen(faces),
            source_forcing_sha256=_digest_array(forcing),
            source_bed_offset_sha256=_digest_array(bed_offset),
            python_source_sha256=_sha(__file__),
            cuda_source_sha256=_sha(CUDA_SOURCE),
            source_kind3_face_count=self.count,
            source_physical_owner_count=1,
            target_reflective_kind2_face_count=self.count,
            target_physical_owner_count=0,
            target_trace_owner_count=0,
            target_pressure_owner_count=0,
            target_boundary_ledger_owner_count=0,
            controller_zero_is_closure_evidence=False,
            explicit_transformation_count=1,
            backend="actual_cuda", semantic_id=SEMANTIC_ID,
            receipt_sha256="")
        object.__setattr__(receipt, "receipt_sha256", _token(receipt))
        self.receipt = receipt

    def _validate_static(self):
        cp = self.cp
        r = self.receipt
        _require(r.receipt_sha256 == _token(r) and
                 r.python_source_sha256 == _sha(__file__) and
                 r.cuda_source_sha256 == _sha(CUDA_SOURCE) and
                 np.array_equal(cp.asnumpy(
                     self.boundary.arrays["mode"]),
                     r.source_boundary_owner_mode) and
                 np.array_equal(cp.asnumpy(
                     self.boundary.arrays["bid"]), r.source_bid) and
                 np.array_equal(cp.asnumpy(
                     self.reconstruction.geometry["bid"]), r.source_bid) and
                 _digest_array(cp.asnumpy(
                     self.boundary.arrays["forcing"])) ==
                     r.source_forcing_sha256,
                 "v98 closed binding source/runtime identity differs")
        return True

    def _begin_trial(self):
        self._validate_static()
        self.boundary.status.fill(0)
        self.boundary.face_status.fill(0)
        self.begin_count += 1

    def _apply(self, q, trace):
        self._validate_static()
        _require(q is self.controller.midpoint and
                 trace is self.reconstruction.buffers["trace"],
                 "v98 closed binding canonical residual arrays differ")
        self._close_kernel(((self.count+127)//128,), (128,), (
            self.faces, np.int64(self.count),
            np.int64(self.reconstruction.nf),
            self.reconstruction.buffers["kind"], trace,
            self.boundary.face_status))
        self.apply_count += 1

    def _apply_pressure(self, q, bed, trace, fl, fr, face_status):
        _require(q is self.controller.midpoint and
                 trace is self.reconstruction.buffers["trace"] and
                 face_status is not None,
                 "v98 closed binding pressure arrays differ")
        self._validate_kernel(((self.count+127)//128,), (128,), (
            self.faces, np.int64(self.count),
            np.int64(self.reconstruction.nf),
            self.reconstruction.buffers["kind"],
            self.boundary.face_status))
        self.pressure_count += 1

    def install(self):
        _require(not self._installed,
                 "v98 closed binding is already installed")
        binding = self

        def begin_trial(boundary_self):
            return binding._begin_trial()

        def apply(boundary_self, q, trace):
            return binding._apply(q, trace)

        def apply_pressure(boundary_self, q, bed, trace, fl, fr,
                           face_status):
            return binding._apply_pressure(
                q, bed, trace, fl, fr, face_status)

        self._bound_begin = MethodType(begin_trial, self.boundary)
        self._bound_apply = MethodType(apply, self.boundary)
        self._bound_pressure = MethodType(apply_pressure, self.boundary)
        self.boundary.begin_trial = self._bound_begin
        self.boundary.apply = self._bound_apply
        self.boundary.apply_pressure = self._bound_pressure
        self.boundary._v98_reflective_closure_binding = self
        self._installed = True
        return self

    def validate_installed(self):
        _require(self._installed and
                 getattr(self.boundary,
                         "_v98_reflective_closure_binding", None) is self and
                 self.boundary.begin_trial is self._bound_begin and
                 self.boundary.apply is self._bound_apply and
                 self.boundary.apply_pressure is self._bound_pressure,
                 "v98 closed binding installation differs")
        return self._validate_static()

    def restore(self):
        if not self._installed:
            return
        self.boundary.begin_trial = self._original_begin
        self.boundary.apply = self._original_apply
        self.boundary.apply_pressure = self._original_pressure
        if getattr(self.boundary,
                   "_v98_reflective_closure_binding", None) is self:
            delattr(self.boundary, "_v98_reflective_closure_binding")
        self._installed = False

    def audit_trial(self, accepted_water_flux_m2_s):
        self.validate_installed()
        flux = np.asarray(accepted_water_flux_m2_s, dtype=np.float64)
        _require(flux.shape == (self.reconstruction.nf,) and
                 np.array_equal(flux[self.faces_host],
                                np.zeros(self.count, dtype=np.float64)) and
                 self.begin_count > 0 and
                 self.apply_count == self.pressure_count == self.begin_count,
                 "v98 closed binding accepted physical water flux differs")
        return True


def bind_reflective_boundary_closure_gpu_v98(
        model, data, *, hydraulic_execution_fingerprint):
    return ResidentReflectiveBoundaryClosureGPUV98(
        model, data,
        hydraulic_execution_fingerprint=
            hydraulic_execution_fingerprint).install()


__all__ = [
    "CoupledAMRClosedBoundaryGPUV98Error",
    "ReflectiveBoundaryClosureReceiptGPUV98",
    "ResidentReflectiveBoundaryClosureGPUV98",
    "bind_reflective_boundary_closure_gpu_v98",
]
