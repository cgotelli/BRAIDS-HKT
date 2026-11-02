"""Pure OpenMP combined sediment/affine-fluid boundary and water-ledger batch.

The caller owns affine temperature conversion, hydraulic fluxes and accepted
state. This batch retains FLOW concentration/return-memory ordering; it neither
reruns hydraulics nor publishes a second boundary or time owner.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np

from .flow2dh_boundary_diffusion import FLOW2DH_OPEN_BOUNDARY_SCHEMA
from .scalar_diffusion_native_cpu_v96 import _array, _cancel

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/"solver/wave/braids_scalar_boundary_v97_module.f90"
WRAPPER = Path(__file__).resolve()
ABI_ID = "braids-v97-native-combined-scalar-boundary-ledger-v1"
LOCKED_SOURCES = {
    "resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/flow2d3d_kernel/src/compute/thahbc.f90":
        "e2c53f54fa0bf18e68d21b59e2d83658443ab1994bd833ea5d35181c8ce502a8",
    "resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/flow2d3d_kernel/src/timedep/incbcc.f90":
        "c7c4f5b1e285b46b6c87d29e94314cdf0071ef2581330e59edba8d66788279ba",
    "resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/flow2d3d_kernel/src/compute/difu.f90":
        "e8e11d438a6e3aa8e912ce1133efdec093b5bad60605aefa6c4b037ea6aa5c17",
}


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _digest(*values):
    result = hashlib.sha256()
    for raw in values:
        value = np.ascontiguousarray(raw)
        result.update(value.dtype.str.encode())
        result.update(json.dumps(value.shape).encode())
        result.update(value.tobytes())
    return result.hexdigest()


@dataclass(frozen=True)
class NativeScalarBoundaryResultV97:
    boundary_concentration: np.ndarray
    advective_face_amount_in: np.ndarray
    next_last_outflow: np.ndarray
    next_return_clock_s: np.ndarray
    expected_advective_boundary_ledger: np.ndarray
    water_divergence_m_s: np.ndarray
    water_boundary_ledger_m3: np.ndarray
    continuity_residual_m: np.ndarray
    runtime_sha256: str
    input_sha256: str
    output_sha256: str
    native_threads: int
    scalar_boundary_batch_owner_count: int = 1
    python_numerical_cell_face_loop_count: int = 0
    state_or_clock_publication_count: int = 0


class NativeScalarBoundaryRuntimeV97:
    def __init__(self, receipt, library, dependency, library_path):
        self.receipt = receipt
        self.library = library
        self.dependency = dependency
        self.library_path = Path(library_path).resolve()
        self._function = library.braids_scalar_boundary_v97
        fa = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        ia = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
        self._function.argtypes = ([ctypes.c_int]*6 + [ia]*8 + [fa]*10 + [ia] +
            [ctypes.c_double]*2 + [ctypes.c_int] + [fa]*8 +
            [ctypes.POINTER(ctypes.c_int)])
        self._function.restype = None

    def _validate_runtime(self):
        if self.receipt.get("receipt_sha256") != _hash({
                k: v for k, v in self.receipt.items() if k != "receipt_sha256"}):
            raise ValueError("native v97 scalar boundary receipt was changed")
        if (_sha(SOURCE) != self.receipt["source_sha256"] or
                _sha(WRAPPER) != self.receipt["wrapper_sha256"] or
                _sha(self.library_path) != self.receipt["library_sha256"]):
            raise ValueError("native v97 scalar boundary source/library identity differs")

    def evaluate(self, topology, mass, depth_before_m, depth_after_m, water,
                 prescribed, return_time_s, last_outflow, return_clock_s,
                 neumann, *, dt_s, native_threads=4, cancelled=False):
        cancel = _cancel(cancelled)
        if cancel():
            raise InterruptedError("native v97 scalar boundary cancelled before launch")
        self._validate_runtime()
        if type(native_threads) is not int or not 1 <= native_threads <= 256:
            raise ValueError("native v97 scalar boundary thread count differs")
        if (isinstance(dt_s, bool) or not isinstance(dt_s, (int, float)) or
                not np.isfinite(dt_s) or dt_s <= 0):
            raise ValueError("native v97 scalar boundary dt must be finite and positive")
        nc, nf, nb = topology.cell_count, topology.face_count, topology.boundary_count
        if any(type(n) is not int or n < lower for n, lower in ((nc, 1), (nf, 1), (nb, 0))):
            raise ValueError("native v97 scalar boundary topology counts differ")
        raw = np.asarray(mass)
        if raw.ndim != 2 or raw.shape[0] < 1:
            raise ValueError("native v97 scalar boundary requires nonempty combined lanes")
        nl = raw.shape[0]
        mass = _array(mass, (nl, nc), "combined mass", nonnegative=True)
        h0 = _array(depth_before_m, (nc,), "parent depth", nonnegative=True)
        h1 = _array(depth_after_m, (nc,), "endpoint depth", nonnegative=True)
        water = _array(water, (nf,), "accepted water flux")
        prescribed = _array(prescribed, (nl, nb), "prescribed values", nonnegative=True)
        times = _array(return_time_s, (nl, nb), "return times")
        last = _array(last_outflow, (nl, nf), "last outflow", nonnegative=True)
        clocks = _array(return_clock_s, (nl, nf), "return clocks", nonnegative=True)
        neu = _array(neumann, (nl,), "Neumann masks", integer=True)
        if np.any((neu != 0) & (neu != 1)):
            raise ValueError("native v97 scalar boundary Neumann mask is not boolean")
        owner = _array(topology.owner, (nf,), "owner", integer=True)
        neighbour = _array(topology.neighbour, (nf,), "neighbour", integer=True)
        bid = _array(topology.boundary_index, (nf,), "boundary index", integer=True)
        offsets = _array(topology.cell_face_offset, (nc+1,), "CSR offsets", integer=True)
        nadj = int(offsets[-1])
        if nadj < 1:
            raise ValueError("native v97 scalar boundary CSR is empty")
        faces = _array(topology.cell_faces, (nadj,), "CSR faces", integer=True)
        signs = _array(topology.cell_face_sign, (nadj,), "CSR signs", integer=True)
        area = _array(topology.cell_area_m2, (nc,), "cell areas")
        length = _array(topology.face_length_m, (nf,), "face lengths")
        dry = float(topology.dry_tolerance_m)
        if (np.any(owner < 0) or np.any(owner >= nc) or np.any(neighbour < -1) or
                np.any(neighbour >= nc) or np.any(bid < -1) or np.any(bid >= nb) or
                np.any((bid >= 0) & (neighbour >= 0)) or offsets[0] != 0 or
                np.any(np.diff(offsets) < 0) or np.any(faces < 0) or np.any(faces >= nf) or
                np.any((signs != -1) & (signs != 1)) or np.any(area <= 0) or
                np.any(length <= 0) or not np.isfinite(dry) or dry < 0):
            raise ValueError("native v97 scalar boundary topology is invalid")
        # Verify every owner/neighbor incidence exactly once, not just bounds.
        cells = np.repeat(np.arange(nc), np.diff(offsets))
        expected_cell = np.where(signs > 0, owner[faces], neighbour[faces])
        if (not np.array_equal(cells, expected_cell) or
                not np.array_equal(np.bincount(faces, minlength=nf),
                    1+(neighbour >= 0).astype(np.int32)) or
                len(np.unique(faces.astype(np.int64)*2+(signs < 0))) != nadj):
            raise ValueError("native v97 scalar boundary CSR incidence differs")
        if np.any(water[(neighbour < 0) & (bid < 0)] != 0):
            raise ValueError("native v97 scalar boundary closed wall has nonzero water flux")
        exterior = np.flatnonzero(bid >= 0)
        bfaces = np.ascontiguousarray(exterior[np.argsort(bid[exterior], kind="stable")], dtype=np.int32)
        boffsets = np.ascontiguousarray(np.r_[0, np.cumsum(
            np.bincount(bid[exterior], minlength=nb))], dtype=np.int32)
        boundary = np.empty((nl, nf)); amount = np.empty((nl, nf))
        next_last = np.empty((nl, nf)); next_clock = np.empty((nl, nf))
        ledger = np.empty((nl, nb)); divergence = np.empty(nc)
        water_ledger = np.empty(nb); continuity = np.empty(nc)
        outputs = (boundary, amount, next_last, next_clock, ledger,
                   divergence, water_ledger, continuity)
        inputs = (owner, neighbour, bid, offsets, faces, signs, area, length,
                  h0, h1, mass, prescribed, water, times, last, clocks, neu)
        input_sha = _hash({"arrays": _digest(*inputs), "dt_s": float(dt_s), "dry": dry})
        status = ctypes.c_int()
        self._function(nc, nf, nl, nb, nadj, len(bfaces), owner, neighbour,
            bid, offsets, faces, signs, boffsets, bfaces, area, length, h0,
            h1, mass, prescribed, water, times, last, clocks, neu, float(dt_s), dry,
            native_threads, *outputs, ctypes.byref(status))
        if status.value or not all(np.isfinite(a).all() for a in outputs):
            raise ValueError(f"native v97 scalar boundary kernel rejected input: status {status.value}")
        self._validate_runtime()
        if cancel():
            raise InterruptedError("native v97 scalar boundary cancelled after launch")
        for value in outputs:
            value.setflags(write=False)
        return NativeScalarBoundaryResultV97(*outputs, self.receipt["receipt_sha256"],
            input_sha, _digest(*outputs), native_threads)


def build_native_scalar_boundary_runtime_v97(build_directory, *, compiler=None):
    for relative, expected in LOCKED_SOURCES.items():
        if _sha(ROOT/relative) != expected:
            raise ValueError("native v97 scalar boundary locked source differs: "+relative)
    selected = compiler or shutil.which("gfortran")
    if selected is None and os.name == "nt":
        candidate = Path(r"C:\Strawberry\c\bin\gfortran.exe")
        selected = candidate if candidate.is_file() else None
    if selected is None:
        raise ValueError("native v97 scalar boundary requires gfortran")
    selected = Path(selected).resolve()
    flags = ["-O3", "-fopenmp", "-shared", "-ffree-line-length-none",
             "-static-libgfortran", "-static-libgcc"]
    directory = Path(build_directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    source_sha, wrapper_sha = _sha(SOURCE), _sha(WRAPPER)
    key = _hash(dict(source=source_sha, compiler=_sha(selected), flags=flags))[:16]
    build = Path(tempfile.mkdtemp(prefix=f"v97_boundary_{key}_", dir=directory))
    name = "braids_scalar_boundary_v97.dll" if os.name == "nt" else "libbraids_scalar_boundary_v97.so"
    library_path = build/name
    partial = build/(name+".partial")
    process = subprocess.run([str(selected), *flags, str(SOURCE), "-o", str(partial)],
        cwd=build, capture_output=True, text=True, timeout=120)
    if process.returncode or not partial.is_file():
        raise ValueError("native v97 scalar boundary build failed: "+process.stdout+process.stderr)
    if (source_sha, wrapper_sha) != (_sha(SOURCE), _sha(WRAPPER)):
        raise ValueError("native v97 scalar boundary source changed during build")
    os.replace(partial, library_path)
    version = subprocess.run([str(selected), "--version"], capture_output=True,
        text=True, check=True, timeout=20).stdout.splitlines()[0]
    receipt = dict(abi=ABI_ID, source_sha256=source_sha, wrapper_sha256=wrapper_sha,
        library_sha256=_sha(library_path), compiler_sha256=_sha(selected),
        compiler_identity=version, compile_flags=flags, locked_source_sha256=LOCKED_SOURCES,
        source_equation=FLOW2DH_OPEN_BOUNDARY_SCHEMA, combined_lane_openmp=True,
        accepted_state_publication_count=0, production_admission=False)
    receipt["receipt_sha256"] = _hash(receipt)
    dependency = None
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        dependency = os.add_dll_directory(str(selected.parent))
    return NativeScalarBoundaryRuntimeV97(receipt, ctypes.CDLL(str(library_path)),
        dependency, library_path)


__all__ = ["NativeScalarBoundaryResultV97", "NativeScalarBoundaryRuntimeV97",
           "build_native_scalar_boundary_runtime_v97"]
