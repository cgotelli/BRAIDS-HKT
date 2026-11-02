"""Parallel pure diffusion consumer for one combined v96 scalar owner.

The numerical update matches the frozen v92 face law and inventory limiter.
It owns no timestep, boundary memory, settling, exchange, or accepted state.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np

from .flow2dh_boundary_diffusion import FLOW2DH_HORIZONTAL_DIFFUSION_SCHEMA
from .shared_turbulence_consumers_v91 import verify_locked_sources_v91

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/"solver/wave/braids_scalar_diffusion_v96_module.f90"
ABI_ID = "braids-v96-native-combined-scalar-diffusion-v1"


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _array(value, shape, name, *, integer=False, nonnegative=False):
    raw = np.asarray(value)
    if raw.shape != shape or raw.dtype.kind not in "fiub":
        raise ValueError(f"native v96 diffusion {name} shape/type differs")
    if not np.isfinite(raw).all():
        raise ValueError(f"native v96 diffusion {name} is nonfinite")
    if integer and (np.any(raw != np.floor(raw)) or np.any(np.abs(raw) > 2147483647)):
        raise ValueError(f"native v96 diffusion {name} is not int32")
    result = np.ascontiguousarray(raw, dtype=np.int32 if integer else np.float64)
    if nonnegative and np.any(result < 0):
        raise ValueError(f"native v96 diffusion {name} is negative")
    return result


def _cancel(value):
    if type(value) is bool:
        return lambda: value
    if callable(value):
        return value
    raise ValueError("native v96 diffusion cancellation must be bool/callable")


class NativeScalarDiffusionRuntimeV96:
    def __init__(self, receipt, library, dependency, library_path):
        self.receipt = receipt
        self.library = library
        self.dependency = dependency
        self.library_path = Path(library_path).resolve()
        self._function = library.braids_scalar_diffusion_v96
        fa = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        ia = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
        self._function.argtypes = [ctypes.c_int]*6 + [ia]*8 + [fa]*7 + [ia] + [
            ctypes.c_double, ctypes.c_double, ctypes.c_int, fa, fa, fa, fa,
            ctypes.POINTER(ctypes.c_int)]
        self._function.restype = None

    def evaluate(self, topology, dicuv_cell_m2_s, mass, depth, boundary,
                 exchange_area, face_active, *, dt_s, native_threads=4,
                 cancelled=False):
        cancel = _cancel(cancelled)
        if cancel():
            raise InterruptedError("native v96 diffusion cancelled before launch")
        if type(native_threads) is not int or not 1 <= native_threads <= 256:
            raise ValueError("native v96 diffusion thread count differs")
        if (isinstance(dt_s, bool) or not isinstance(dt_s, (int, float)) or
                not np.isfinite(dt_s) or dt_s <= 0):
            raise ValueError("native v96 diffusion dt must be finite and positive")
        if self.receipt.get("receipt_sha256") != _hash({
                k: v for k, v in self.receipt.items() if k != "receipt_sha256"}):
            raise ValueError("native v96 diffusion runtime receipt was changed")
        if (_sha(SOURCE) != self.receipt["source_sha256"] or
                _sha(self.library_path) != self.receipt["library_sha256"]):
            raise ValueError("native v96 diffusion source/library identity differs")
        nc, nf, nb = topology.cell_count, topology.face_count, topology.boundary_count
        raw = np.asarray(mass)
        if raw.ndim != 2 or raw.shape[0] < 1:
            raise ValueError("native v96 diffusion requires nonempty combined lanes")
        nl = raw.shape[0]
        mass = _array(mass, (nl, nc), "mass", nonnegative=True)
        depth = _array(depth, (nc,), "depth", nonnegative=True)
        dic = _array(dicuv_cell_m2_s, (nc,), "dicuv", nonnegative=True)
        boundary = _array(boundary, (nl, nf), "boundary", nonnegative=True)
        exchange = _array(exchange_area, (nf,), "exchange area", nonnegative=True)
        active = _array(face_active, (nf,), "active mask", integer=True)
        if np.any((active != 0) & (active != 1)):
            raise ValueError("native v96 diffusion active mask is not boolean")
        owner = _array(topology.owner, (nf,), "owner", integer=True)
        neighbour = _array(topology.neighbour, (nf,), "neighbour", integer=True)
        bid = _array(topology.boundary_index, (nf,), "boundary indices", integer=True)
        offsets = _array(topology.cell_face_offset, (nc+1,), "cell offsets", integer=True)
        nadj = int(offsets[-1])
        if nadj < 1:
            raise ValueError("native v96 diffusion cell adjacency is empty")
        faces = _array(topology.cell_faces, (nadj,), "cell faces", integer=True)
        signs = _array(topology.cell_face_sign, (nadj,), "cell signs", integer=True)
        area = _array(topology.cell_area_m2, (nc,), "cell areas")
        distance = _array(topology.centre_distance_m, (nf,), "face distances")
        if (np.any(owner < 0) or np.any(owner >= nc) or np.any(neighbour < -1) or
                np.any(neighbour >= nc) or np.any(bid < -1) or np.any(bid >= nb) or
                offsets[0] != 0 or np.any(np.diff(offsets) < 0) or
                np.any(faces < 0) or np.any(faces >= nf) or
                np.any((signs != -1) & (signs != 1)) or
                np.any(area <= 0) or np.any(distance <= 0)):
            raise ValueError("native v96 diffusion topology is invalid")
        dry = float(topology.profile.diffusion_dry_threshold_m)
        if not np.isfinite(dry) or dry < 0:
            raise ValueError("native v96 diffusion dry threshold is invalid")
        # Boundary-owned CSR keeps each named ledger deterministic without
        # atomics or an O(nb*nf) scan; no Python numerical cell/face loops.
        exterior = np.flatnonzero(bid >= 0)
        bfaces = np.ascontiguousarray(exterior[np.argsort(bid[exterior], kind="stable")], dtype=np.int32)
        boffset = np.ascontiguousarray(np.concatenate(([0], np.cumsum(
            np.bincount(bid[exterior], minlength=nb)))), dtype=np.int32)
        result = np.empty((nl, nc)); amount = np.empty((nl, nf))
        ledger = np.empty((nl, nb)); effective = np.empty(nf)
        status = ctypes.c_int()
        self._function(nc, nf, nl, nb, nadj, len(bfaces), owner, neighbour,
            bid, offsets, faces, signs, boffset, bfaces, area, distance,
            dic, mass, depth, boundary, exchange, active, float(dt_s), dry,
            native_threads, result, amount, ledger, effective, ctypes.byref(status))
        if status.value or not all(np.isfinite(a).all() for a in (result, amount, ledger, effective)):
            raise ValueError(f"native v96 diffusion kernel rejected input: status {status.value}")
        if cancel():
            raise InterruptedError("native v96 diffusion cancelled after launch")
        for a in (result, amount, ledger, effective):
            a.setflags(write=False)
        return dict(mass_kg_m2=result, diffusive_face_mass_owner_out_kg=amount,
            diffusive_boundary_ledger_kg=ledger, effective_diffusivity_m2_s=effective,
            native_runtime_sha256=self.receipt["receipt_sha256"],
            native_threads=native_threads, scalar_diffusion_owner_count=1,
            python_numerical_cell_face_loop_count=0,
            state_or_clock_publication_count=0, native_diffusion_launch_count=1)

    def evaluate_arrays(self, *args, **kwargs):
        """One evaluation, returning the existing v92 four-array tuple."""
        result = self.evaluate(*args, **kwargs)
        return tuple(result[k] for k in (
            "mass_kg_m2", "diffusive_face_mass_owner_out_kg",
            "diffusive_boundary_ledger_kg", "effective_diffusivity_m2_s"))


def build_native_scalar_diffusion_runtime_v96(build_directory, *, compiler=None):
    locked = verify_locked_sources_v91(ROOT)
    selected = compiler or shutil.which("gfortran")
    if selected is None and os.name == "nt":
        candidate = Path(r"C:\Strawberry\c\bin\gfortran.exe")
        selected = candidate if candidate.is_file() else None
    if selected is None:
        raise ValueError("native v96 scalar diffusion requires gfortran")
    selected = Path(selected).resolve()
    flags = ["-O3", "-fopenmp", "-shared", "-ffree-line-length-none",
             "-static-libgfortran", "-static-libgcc"]
    build_root = Path(build_directory).resolve()
    build_root.mkdir(parents=True, exist_ok=True)
    source_sha = _sha(SOURCE)
    key = _hash(dict(source=source_sha, compiler=_sha(selected), flags=flags))[:16]
    build = Path(tempfile.mkdtemp(prefix=f"v96_diffusion_{key}_", dir=build_root))
    name = "braids_scalar_diffusion_v96.dll" if os.name == "nt" else "libbraids_scalar_diffusion_v96.so"
    library_path = build/name
    partial = build/(name+".partial")
    process = subprocess.run([str(selected), *flags, str(SOURCE), "-o", str(partial)],
        cwd=build, capture_output=True, text=True, timeout=120)
    if process.returncode or not partial.is_file():
        raise ValueError("native v96 scalar diffusion build failed: "+process.stdout+process.stderr)
    if source_sha != _sha(SOURCE):
        raise ValueError("native v96 scalar diffusion source changed during build")
    os.replace(partial, library_path)
    version = subprocess.run([str(selected), "--version"], capture_output=True,
                             text=True, check=True, timeout=20).stdout.splitlines()[0]
    receipt = dict(abi=ABI_ID, source_sha256=source_sha, library_sha256=_sha(library_path),
        compiler_sha256=_sha(selected), compiler_identity=version, compile_flags=flags,
        locked_source_sha256=locked, source_equation=FLOW2DH_HORIZONTAL_DIFFUSION_SCHEMA,
        openmp_face_cell_boundary_loops=True, production_admission=False)
    receipt["receipt_sha256"] = _hash(receipt)
    dependency = None
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        dependency = os.add_dll_directory(str(selected.parent))
    return NativeScalarDiffusionRuntimeV96(receipt, ctypes.CDLL(str(library_path)), dependency, library_path)


__all__ = ["NativeScalarDiffusionRuntimeV96", "build_native_scalar_diffusion_runtime_v96"]
