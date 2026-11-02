"""Resident physical flux replacement/draining/gather, not a boundary owner.

An authenticated physical trace supplies both the flux and its ownership mask.
The live path consumes device arrays without numerical host readback. The host
``evaluate`` API is only for independent same-input validation observations.
"""
from pathlib import Path

import numpy as np

from braids_core.sediment.density_active_projection_v97 import _array, _digest, _hash, _sha
from braids_core.sediment.density_trace_projection_v98 import _hex
from braids_core.sediment.densin_fwave_v85 import verify_locked_sources_v85
from braids_core.sediment.physical_flux_gather_native_cpu_v98 import KEYS

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(__file__).with_suffix(".cu")
OPTIONS = ("--std=c++11", "--fmad=false", "--prec-div=true", "--prec-sqrt=true", "--ftz=false")


def _controls(gravity_m_s2, dry_tolerance_m, dt_s):
    values = (gravity_m_s2, dry_tolerance_m, dt_s)
    if (any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v)
            for v in values) or gravity_m_s2 <= 0 or dry_tolerance_m < 0 or dt_s <= 0):
        raise ValueError("v98 CUDA physical gather controls differ")
    return dict(gravity_m_s2=float(gravity_m_s2), dry_tolerance_m=float(dry_tolerance_m), dt_s=float(dt_s))


def _cancel(cancelled):
    if not callable(cancelled) and type(cancelled) is not bool:
        raise ValueError("v98 CUDA physical gather cancellation must be bool/callable")
    return cancelled if callable(cancelled) else lambda: cancelled


class ResidentPhysicalFluxGatherGPUV98:
    def __init__(self, *, backend, owner, neighbour, cell_face_offset, cell_faces,
                 cell_face_sign, face_length_m, cell_area_m2, normal):
        if backend != "implicit_fwave_midpoint_gpu":
            raise ValueError("v98 physical gather requires explicitly selected implicit GPU")
        own_raw, area_raw = np.asarray(owner), np.asarray(cell_area_m2)
        if own_raw.ndim != 1 or area_raw.ndim != 1 or min(own_raw.size, area_raw.size) < 1:
            raise ValueError("v98 CUDA physical gather geometry shape differs")
        self.nf, self.nc = int(own_raw.size), int(area_raw.size)
        nf, nc = self.nf, self.nc
        own = _array(owner, (nf,), "owner", integer=True)
        nei = _array(neighbour, (nf,), "neighbour", integer=True)
        off = _array(cell_face_offset, (nc+1,), "offset", integer=True)
        nadj = int(off[-1])
        if nadj < 1:
            raise ValueError("v98 CUDA physical gather CSR incidence differs")
        faces = _array(cell_faces, (nadj,), "faces", integer=True)
        signs = _array(cell_face_sign, (nadj,), "signs", integer=True)
        length = _array(face_length_m, (nf,), "length")
        area = _array(cell_area_m2, (nc,), "area")
        normals = _array(normal, (2, nf), "normal")
        if (np.any(own < 0) or np.any(own >= nc) or np.any(nei < -1) or np.any(nei >= nc)
                or off[0] != 0 or np.any(np.diff(off) < 0) or np.any(faces < 0)
                or np.any(faces >= nf) or np.any((signs != -1) & (signs != 1))
                or np.any(length <= 0) or np.any(area <= 0)):
            raise ValueError("v98 CUDA physical gather topology/domain differs")
        cells = np.repeat(np.arange(nc), np.diff(off))
        pairs = np.stack((cells, faces, signs), axis=1)
        internal = np.flatnonzero(nei >= 0)
        expected = np.concatenate((np.column_stack((own, np.arange(nf), np.ones(nf, dtype=int))),
            np.column_stack((nei[internal], internal, -np.ones(internal.size, dtype=int)))), axis=0)
        order = lambda a: a[np.lexsort((a[:, 2], a[:, 1], a[:, 0]))]
        if not np.array_equal(order(pairs), order(expected)):
            raise ValueError("v98 CUDA physical gather CSR incidence differs")
        self.geometry_host = tuple(np.array(a, copy=True, order="C")
            for a in (own, nei, off, faces, signs, length, area, normals))
        for value in self.geometry_host:
            value.setflags(write=False)
        import cupy as cp
        self.cp = cp
        self.device_id = int(cp.cuda.runtime.getDevice())
        self.geometry_device = tuple(cp.asarray(a) for a in self.geometry_host)
        names = ("physical_replace_v98", "physical_draining_v98", "physical_limit_v98",
                 "physical_pressure_gather_v98")
        self.module = cp.RawModule(code=SOURCE.read_text(encoding="utf-8"), options=OPTIONS,
                                  name_expressions=names)
        self.kernels = tuple(self.module.get_function(name) for name in names)
        self.receipt = dict(backend=backend, actual_cuda=True, automatic_cpu_fallback=False,
            source_sha256=_sha(SOURCE), wrapper_sha256=_sha(Path(__file__)),
            geometry_sha256=_digest(*self.geometry_host), device_id=self.device_id,
            cell_count=nc, face_count=nf, cuda_options=list(OPTIONS),
            locked_source_sha256=verify_locked_sources_v85(ROOT), production_admission=False)
        self.receipt["receipt_sha256"] = _hash(self.receipt)
        self.last_resident_result = None

    def validate_runtime(self, *, check_resident_geometry=True):
        """Authenticate once at sequence entry, outside the numerical residual."""
        if (self.receipt["receipt_sha256"] != _hash({k: v for k, v in self.receipt.items()
                if k != "receipt_sha256"}) or _sha(SOURCE) != self.receipt["source_sha256"]
                or _sha(Path(__file__)) != self.receipt["wrapper_sha256"]
                or _digest(*self.geometry_host) != self.receipt["geometry_sha256"]):
            raise ValueError("v98 CUDA physical gather runtime/source/geometry identity differs")
        if check_resident_geometry and any(not np.array_equal(self.cp.asnumpy(a), b)
                for a, b in zip(self.geometry_device, self.geometry_host)):
            raise ValueError("v98 CUDA physical gather resident geometry differs")

    def launch_resident(self, q, wet_mask, raw_left, raw_right, physical_flux, physical_mask, *,
                        gravity_m_s2, dry_tolerance_m, dt_s, physical_trace_receipt_sha256,
                        reflected_result_sha256, cancelled=False):
        """Four batched CUDA phases; no numerical host arrays or owner publication."""
        self.last_resident_result = None
        cancel = _cancel(cancelled)
        if cancel():
            raise InterruptedError("v98 CUDA physical gather cancelled before launch")
        controls = _controls(gravity_m_s2, dry_tolerance_m, dt_s)
        provenance = dict(physical_trace_receipt_sha256=_hex(physical_trace_receipt_sha256, "physical trace"),
                          reflected_result_sha256=_hex(reflected_result_sha256, "reflected result"))
        cp, nc, nf = self.cp, self.nc, self.nf
        inputs = (q, wet_mask, raw_left, raw_right, physical_flux, physical_mask)
        shapes = ((3, nc), (nc,), (3, nf), (3, nf), (3, nf), (nf,))
        if any(not isinstance(a, cp.ndarray) or a.device.id != self.device_id
                or not a.flags.c_contiguous or a.shape != shape or a.dtype != dtype
                for a, shape, dtype in zip(inputs, shapes,
                    (cp.float64, cp.int32, cp.float64, cp.float64, cp.float64, cp.int32))):
            raise ValueError("v98 CUDA physical gather requires contiguous selected-device arrays")
        out = {key: cp.empty(shape, dtype=cp.float64) for key, shape in zip(KEYS,
            ((3, nf),)*4+((nc,), (3, nc), (nf,)))}
        out.update(face_status=cp.empty(nf, cp.int32), cell_status=cp.empty(nc, cp.int32),
                   physical_consumed=cp.empty(nf, cp.int32), limited_faces=cp.empty(nf, cp.int32))
        own, nei, off, faces, signs, length, area, normal = self.geometry_device
        faces_grid, cells_grid, block = ((nf+127)//128,), ((nc+127)//128,), (128,)
        with cp.cuda.Device(self.device_id):
            self.kernels[0](faces_grid, block, (np.int32(nf), raw_left, raw_right, physical_flux,
                physical_mask, nei, out[KEYS[0]], out[KEYS[1]], out["physical_consumed"], out["face_status"]))
            self.kernels[1](cells_grid, block, (np.int32(nc), np.float64(controls["dt_s"]),
                q, wet_mask, off, faces, signs, length, area, out[KEYS[0]], out[KEYS[4]], out["cell_status"]))
            self.kernels[2](faces_grid, block, (np.int32(nc), np.int32(nf),
                np.float64(controls["dry_tolerance_m"]), q, wet_mask, own, nei, out[KEYS[0]], out[KEYS[1]],
                out[KEYS[4]], out[KEYS[2]], out[KEYS[3]], out[KEYS[6]], out["limited_faces"]))
            self.kernels[3](cells_grid, block, (np.int32(nc), np.int32(nf),
                np.float64(controls["gravity_m_s2"]), q, wet_mask, off, faces, signs, length, area,
                normal, out[KEYS[2]], out[KEYS[3]], out[KEYS[5]], out["cell_status"]))
        if cancel():
            raise InterruptedError("v98 CUDA physical gather cancelled after launch")
        out["audit"] = dict(runtime_sha256=self.receipt["receipt_sha256"], controls_sha256=_hash(controls),
            **provenance, cuda_kernel_launch_count=4, physical_flux_application_count=1,
            draining_limiter_owner_count=1, pressure_gather_owner_count=1,
            riemann_evaluation_count=0, turbulence_evaluation_count=0, density_evaluation_count=0,
            numerical_host_readback_count=0, numerical_python_cell_face_loops=0,
            boundary_memory_publication_count=0, state_or_clock_publication_count=0,
            automatic_cpu_fallback=False, production_admission=False)
        self.last_resident_result = out
        return out

    def evaluate(self, q, wet_mask, raw_left, raw_right, physical_flux, physical_mask, **kwargs):
        """Validation adapter: retain the real resident result, then read observations."""
        self.last_resident_result = None
        cancel = _cancel(kwargs.get("cancelled", False))
        if cancel():
            raise InterruptedError("v98 CUDA physical gather cancelled before launch")
        shapes = ((3, self.nc), (self.nc,), (3, self.nf), (3, self.nf), (3, self.nf), (self.nf,))
        arrays = tuple(_array(a, shape, "physical gather input", integer=i in (1, 5))
            for i, (a, shape) in enumerate(zip((q, wet_mask, raw_left, raw_right,
                                              physical_flux, physical_mask), shapes)))
        with self.cp.cuda.Device(self.device_id):
            self.validate_runtime()
            # launch_resident polls before and after once; this earlier poll is
            # deliberately not forwarded as another user cancellation event.
            out = self.launch_resident(*(self.cp.asarray(a) for a in arrays),
                **{**kwargs, "cancelled": False})
            if self.cp.any(out["face_status"]).item() or self.cp.any(out["cell_status"]).item():
                self.last_resident_result = None
                raise ValueError("v98 CUDA physical gather candidate is outside finite/shared-water domain")
            values = {key: self.cp.asnumpy(out[key]) for key in KEYS}
            counts = (int(self.cp.sum(out["physical_consumed"]).item()),
                      int(self.cp.sum(out["limited_faces"]).item()))
        if cancel():
            self.last_resident_result = None
            raise InterruptedError("v98 CUDA physical gather cancelled after launch")
        for a in values.values():
            a.setflags(write=False)
        metadata = dict(**out["audit"], input_sha256=_digest(*arrays),
            output_sha256=_digest(*(values[key] for key in KEYS)),
            physical_flux_consumption_count=counts[0], limited_face_count=counts[1],
            validation_observation_host_readback=True)
        return dict(**values, metadata=metadata, receipt_sha256=_hash(metadata))
