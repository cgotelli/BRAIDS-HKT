"""Resident CUDA conservative 2:1 transfer, not a hierarchy/state owner.

One numerical work item per independent parent fills four children by a single
dry-aware stage, preserves water/transported inventories, then area-restricts.
The caller authenticates accepted parents, geometry and root boundary memory,
rejects the candidate status before publication, and derives bed/EOS/HLES
outside this batch. Neither a timestep nor a constitutive law advances here.
"""
from __future__ import annotations

from pathlib import Path
import re

import numpy as np

from braids_core.sediment.density_active_projection_v97 import _digest, _hash, _sha


SOURCE = Path(__file__).with_suffix(".cu")
OPTIONS = ("--std=c++11", "--fmad=false", "--prec-div=true",
           "--prec-sqrt=true", "--ftz=false")
SHA_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
KEYS = ("child_q", "child_suspended", "child_bed", "child_fluid",
        "restricted_q", "restricted_suspended", "restricted_bed",
        "restricted_fluid")


def _check_sha(value, name):
    if not isinstance(value, str) or SHA_PATTERN.fullmatch(value) is None:
        raise ValueError(f"v98 zero-time CUDA {name} must be a SHA-256 digest")
    return value


def _poll(cancelled):
    if isinstance(cancelled, (bool, np.bool_)):
        return bool(cancelled)
    if not callable(cancelled):
        raise TypeError("cancelled must be a Boolean or callback")
    result = cancelled()
    if not isinstance(result, (bool, np.bool_)):
        raise TypeError("cancelled callback must return Boolean data")
    return bool(result)


class ResidentAMRZeroTimeTransferGPUV98:
    """Parallel native transfer on an explicitly selected CUDA device."""

    def __init__(self, *, backend, parent_area_m2, child_area_m2,
                 child_bed_elevation_m, fraction_count, dry_tolerance_m):
        if backend != "implicit_fwave_midpoint_gpu":
            raise ValueError("v98 zero-time transfer requires explicit implicit GPU")
        if type(fraction_count) is not int or fraction_count < 1:
            raise ValueError("v98 zero-time fraction count must be positive")
        a0 = np.asarray(parent_area_m2)
        if a0.ndim != 1 or a0.size < 1:
            raise ValueError("v98 zero-time parent areas must be a nonempty vector")
        self.np, self.ng = int(a0.size), fraction_count
        geometry = tuple(np.array(a, dtype=np.float64, order="C", copy=True)
                         for a in (a0, child_area_m2, child_bed_elevation_m))
        if any(a.shape != shape or not np.isfinite(a).all()
               for a, shape in zip(geometry, ((self.np,), (self.np, 4), (self.np, 4)))):
            raise ValueError("v98 zero-time geometry shape or finite domain differs")
        if np.any(geometry[0] <= 0.) or np.any(geometry[1] <= 0.):
            raise ValueError("v98 zero-time areas must be positive")
        tol = 2048.*np.finfo(np.float64).eps*np.maximum(1., geometry[0])
        if np.any(np.abs(np.sum(geometry[1], axis=1)-geometry[0]) > tol):
            raise ValueError("v98 zero-time child areas do not partition the parent")
        if (isinstance(dry_tolerance_m, (bool, np.bool_)) or
                not np.isscalar(dry_tolerance_m) or
                not np.isfinite(dry_tolerance_m) or dry_tolerance_m <= 0.):
            raise ValueError("v98 zero-time dry tolerance must be finite and positive")
        self.dry = float(dry_tolerance_m)
        for a in geometry:
            a.setflags(write=False)
        self.geometry_host = geometry
        import cupy as cp
        self.cp, self.device_id = cp, int(cp.cuda.runtime.getDevice())
        self.geometry_device = tuple(cp.asarray(a) for a in geometry)
        self.module = cp.RawModule(code=SOURCE.read_text(encoding="utf-8"),
                                   options=OPTIONS,
                                   name_expressions=("amr_zero_time_transfer_v98",))
        self.kernel = self.module.get_function("amr_zero_time_transfer_v98")
        self.receipt = dict(schema=1, backend=backend, actual_cuda=True,
            source_sha256=_sha(SOURCE), wrapper_sha256=_sha(Path(__file__)),
            geometry_sha256=_digest(*geometry), device_id=self.device_id,
            parent_count=self.np, fraction_count=self.ng, dry_tolerance_m=self.dry,
            cuda_options=list(OPTIONS), automatic_cpu_fallback=False,
            conserved_transfer_only=True, constitutive_evaluation_count=0,
            state_or_clock_publication_count=0, production_admission=False)
        self.receipt["receipt_sha256"] = _hash(self.receipt)
        self.last_resident_result = None
        self.numerical_kernel_launch_count = 0

    def validate_runtime(self, *, check_resident_geometry=True):
        """Sequence-entry authentication; never called from the numerical batch."""
        receipt = {k: v for k, v in self.receipt.items() if k != "receipt_sha256"}
        if (self.receipt["receipt_sha256"] != _hash(receipt) or
                _sha(SOURCE) != receipt["source_sha256"] or
                _sha(Path(__file__)) != receipt["wrapper_sha256"] or
                _digest(*self.geometry_host) != receipt["geometry_sha256"] or
                self.np != receipt["parent_count"] or self.ng != receipt["fraction_count"] or
                self.dry != receipt["dry_tolerance_m"] or
                int(self.cp.cuda.runtime.getDevice()) != self.device_id):
            raise ValueError("v98 zero-time runtime/source/geometry identity differs")
        if check_resident_geometry and any(
                not np.array_equal(self.cp.asnumpy(a), b)
                for a, b in zip(self.geometry_device, self.geometry_host)):
            raise ValueError("v98 zero-time resident geometry differs")

    def launch_resident(self, parent_q, parent_suspended, parent_bed, parent_fluid,
                        *, hierarchy_fingerprint, parent_state_sha256,
                        boundary_memory_sha256, cancelled=False):
        """Return device candidates/status; caller rejects status before commit.

        This path performs no numerical host readback, nor accepted-parent
        construction. Digests are supplied by the authoritative outer owner.
        """
        self.last_resident_result = None
        for value, name in ((hierarchy_fingerprint, "hierarchy"),
                            (parent_state_sha256, "accepted parent"),
                            (boundary_memory_sha256, "root boundary memory")):
            _check_sha(value, name)
        if _poll(cancelled):
            raise RuntimeError("v98 zero-time transfer cancelled before launch")
        cp, n, g = self.cp, self.np, self.ng
        inputs = (parent_q, parent_suspended, parent_bed, parent_fluid)
        shapes = ((3, n), (g, n), (g, n), (2, n))
        if any(not isinstance(a, cp.ndarray) or a.device.id != self.device_id or
               a.dtype != cp.float64 or not a.flags.c_contiguous or a.shape != shape
               for a, shape in zip(inputs, shapes)):
            raise ValueError("v98 zero-time inputs must be contiguous resident doubles")
        out = {name: cp.empty(shape, dtype=cp.float64) for name, shape in zip(KEYS,
            ((3, n, 4), (g, n, 4), (g, n, 4), (2, n, 4),
             (3, n), (g, n), (g, n), (2, n)))}
        out["parent_status"] = cp.empty(n, dtype=cp.int32)
        self.kernel(((n+127)//128,), (128,),
            (np.int32(n), np.int32(g), np.float64(self.dry),
             *self.geometry_device, *inputs, *(out[name] for name in KEYS),
             out["parent_status"]))
        self.numerical_kernel_launch_count += 1
        if _poll(cancelled):
            raise RuntimeError("v98 zero-time transfer cancelled after launch")
        out["audit"] = dict(runtime_receipt_sha256=self.receipt["receipt_sha256"],
            hierarchy_fingerprint=hierarchy_fingerprint,
            parent_state_sha256=parent_state_sha256,
            boundary_memory_sha256=boundary_memory_sha256,
            actual_cuda=True, numerical_kernel_launch_count=1,
            python_numerical_parent_child_loop_count=0,
            numerical_host_readback_count=0, constitutive_evaluation_count=0,
            physical_boundary_evaluation_count=0, state_or_clock_publication_count=0)
        self.last_resident_result = out
        return out

    def evaluate(self, parent_q, parent_suspended, parent_bed, parent_fluid, **kwargs):
        """Validation adapter: upload inputs and reject status on the host."""
        self.last_resident_result = None
        self.validate_runtime()
        inputs = tuple(np.ascontiguousarray(a, dtype=np.float64) for a in
                       (parent_q, parent_suspended, parent_bed, parent_fluid))
        result = self.launch_resident(*(self.cp.asarray(a) for a in inputs), **kwargs)
        status = self.cp.asnumpy(result["parent_status"])
        if np.any(status):
            self.last_resident_result = None
            raise ValueError(f"v98 zero-time CUDA candidate rejected: status {status.tolist()}")
        observed = {name: self.cp.asnumpy(result[name]) for name in KEYS}
        if any(not np.isfinite(a).all() for a in observed.values()):
            self.last_resident_result = None
            raise ValueError("v98 zero-time CUDA candidate is not finite")
        for a in observed.values():
            a.setflags(write=False)
        observed["parent_status"] = status
        observed["audit"] = dict(result["audit"], validation_input_upload_count=4,
                                 validation_output_readback_count=len(KEYS)+1)
        return observed


__all__ = ["ResidentAMRZeroTimeTransferGPUV98", "KEYS"]
