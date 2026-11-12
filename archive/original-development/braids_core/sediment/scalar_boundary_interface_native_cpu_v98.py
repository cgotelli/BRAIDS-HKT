"""Native scalar-boundary batch with typed AMR internal trace faces.

This additive wrapper compiles and calls the unchanged v97 numerical batch.
It only admits an authenticated subset of ``neighbour=-1, boundary_index=-1``
faces into the CSR water-divergence gather.  Those faces remain absent from
physical boundary concentration, return-memory and named-ledger ownership.
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
from .scalar_boundary_native_cpu_v97 import LOCKED_SOURCES


ROOT = Path(__file__).resolve().parents[2]
BASE_SOURCE = ROOT/"solver/wave/braids_scalar_boundary_v97_module.f90"
SOURCE = ROOT/"solver/wave/braids_scalar_boundary_interface_v98_module.f90"
WRAPPER = Path(__file__).resolve()
ABI_ID = "braids-v98-native-scalar-boundary-internal-interface-v1"


class NativeScalarBoundaryInterfaceV98Error(ValueError):
    """The v98 boundary/interface batch or its provenance differs."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("ascii")


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _digest(*values: object) -> str:
    result = hashlib.sha256()
    for raw in values:
        value = np.ascontiguousarray(raw)
        result.update(value.dtype.str.encode("ascii"))
        result.update(_canonical(list(value.shape)))
        result.update(value.tobytes(order="C"))
    return result.hexdigest()


def _hex(value: object, name: str) -> str:
    if (not isinstance(value, str) or len(value) != 64 or
            any(ch not in "0123456789abcdef" for ch in value)):
        raise NativeScalarBoundaryInterfaceV98Error(
            f"v98 scalar boundary {name} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True)
class NativeScalarBoundaryInterfaceResultV98:
    boundary_concentration: np.ndarray
    advective_face_amount_in: np.ndarray
    next_last_outflow: np.ndarray
    next_return_clock_s: np.ndarray
    expected_advective_boundary_ledger: np.ndarray
    water_divergence_m_s: np.ndarray
    water_boundary_ledger_m3: np.ndarray
    continuity_residual_m: np.ndarray
    interface_faces: np.ndarray
    runtime_sha256: str
    input_sha256: str
    output_sha256: str
    trace_receipt_sha256: str
    trace_source_sha256: str
    trace_runtime_sha256: str
    substep_index: int
    native_threads: int
    scalar_boundary_batch_owner_count: int = 1
    internal_interface_water_divergence_owner_count: int = 1
    physical_boundary_surrogate_count: int = 0
    boundary_memory_consumer_count: int = 1
    python_numerical_cell_face_loop_count: int = 0
    state_or_clock_publication_count: int = 0


class NativeScalarBoundaryInterfaceRuntimeV98:
    def __init__(self, receipt, library, dependency, library_path):
        self.receipt = dict(receipt)
        self.library = library
        self.dependency = dependency
        self.library_path = Path(library_path).resolve()
        self._function = library.braids_scalar_boundary_interface_v98
        fa = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        ia = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
        self._function.argtypes = (
            [ctypes.c_int]*6 + [ia]*9 + [fa]*10 + [ia] +
            [ctypes.c_double]*2 + [ctypes.c_int] + [fa]*8 +
            [ctypes.POINTER(ctypes.c_int)])
        self._function.restype = None

    def _validate_runtime(self) -> None:
        body = {key: value for key, value in self.receipt.items()
                if key != "receipt_sha256"}
        if (self.receipt.get("receipt_sha256") != _hash(body) or
                _sha(BASE_SOURCE) !=
                self.receipt.get("base_source_sha256") or
                _sha(SOURCE) != self.receipt.get("source_sha256") or
                _sha(WRAPPER) != self.receipt.get("wrapper_sha256") or
                not self.library_path.is_file() or
                _sha(self.library_path) !=
                self.receipt.get("library_sha256")):
            raise NativeScalarBoundaryInterfaceV98Error(
                "v98 scalar boundary source/library identity differs")

    def evaluate(
        self, topology, mass, depth_before_m, depth_after_m, water,
        prescribed, return_time_s, last_outflow, return_clock_s, neumann, *,
        trace_faces, trace_receipt_sha256, trace_source_sha256,
        trace_runtime_sha256, substep_index, dt_s, native_threads=4,
        cancelled=False,
    ) -> NativeScalarBoundaryInterfaceResultV98:
        cancel = _cancel(cancelled)
        if cancel():
            raise InterruptedError(
                "native v98 scalar boundary cancelled before launch")
        self._validate_runtime()
        if type(native_threads) is not int or not 1 <= native_threads <= 256:
            raise NativeScalarBoundaryInterfaceV98Error(
                "v98 scalar boundary thread count differs")
        if (type(substep_index) is not int or substep_index not in (0, 1) or
                isinstance(dt_s, bool) or
                not isinstance(dt_s, (int, float)) or
                not np.isfinite(dt_s) or dt_s <= 0.0):
            raise NativeScalarBoundaryInterfaceV98Error(
                "v98 scalar boundary substep or dt differs")
        trace_receipt = _hex(trace_receipt_sha256, "trace receipt")
        trace_source = _hex(trace_source_sha256, "trace source")
        trace_runtime = _hex(trace_runtime_sha256, "trace runtime")

        nc, nf, nb = (topology.cell_count, topology.face_count,
                      topology.boundary_count)
        raw_mass = np.asarray(mass)
        if raw_mass.ndim != 2 or raw_mass.shape[0] < 1:
            raise NativeScalarBoundaryInterfaceV98Error(
                "v98 scalar boundary requires nonempty combined lanes")
        nl = raw_mass.shape[0]
        mass = _array(mass, (nl, nc), "combined mass", nonnegative=True)
        h0 = _array(depth_before_m, (nc,), "parent depth", nonnegative=True)
        h1 = _array(depth_after_m, (nc,), "endpoint depth", nonnegative=True)
        water = _array(water, (nf,), "accepted water flux")
        prescribed = _array(
            prescribed, (nl, nb), "prescribed values", nonnegative=True)
        times = _array(return_time_s, (nl, nb), "return times")
        last = _array(last_outflow, (nl, nf), "last outflow", nonnegative=True)
        clocks = _array(
            return_clock_s, (nl, nf), "return clocks", nonnegative=True)
        neu = _array(neumann, (nl,), "Neumann masks", integer=True)
        if np.any((neu != 0) & (neu != 1)):
            raise NativeScalarBoundaryInterfaceV98Error(
                "v98 scalar boundary Neumann mask is not Boolean")
        owner = _array(topology.owner, (nf,), "owner", integer=True)
        neighbour = _array(
            topology.neighbour, (nf,), "neighbour", integer=True)
        bid = _array(
            topology.boundary_index, (nf,), "boundary index", integer=True)
        offsets = _array(
            topology.cell_face_offset, (nc+1,), "CSR offsets", integer=True)
        nadj = int(offsets[-1])
        if nadj < 1:
            raise NativeScalarBoundaryInterfaceV98Error(
                "v98 scalar boundary CSR is empty")
        faces = _array(topology.cell_faces, (nadj,), "CSR faces", integer=True)
        signs = _array(
            topology.cell_face_sign, (nadj,), "CSR signs", integer=True)
        area = _array(topology.cell_area_m2, (nc,), "cell areas")
        length = _array(topology.face_length_m, (nf,), "face lengths")
        dry = float(topology.dry_tolerance_m)
        selected_raw = np.asarray(trace_faces)
        if (selected_raw.ndim != 1 or selected_raw.size < 1 or
                selected_raw.dtype.kind not in "iu"):
            raise NativeScalarBoundaryInterfaceV98Error(
                "v98 scalar boundary trace faces differ")
        selected = np.ascontiguousarray(selected_raw, dtype=np.int32)
        if (np.any(selected < 0) or np.any(selected >= nf) or
                len(np.unique(selected)) != len(selected) or
                np.any(neighbour[selected] != -1) or
                np.any(bid[selected] != -1)):
            raise NativeScalarBoundaryInterfaceV98Error(
                "v98 scalar boundary trace topology differs")
        trace_mask = np.zeros(nf, dtype=np.int32)
        trace_mask[selected] = 1
        unowned = (neighbour < 0) & (bid < 0) & (trace_mask == 0)
        if np.any(water[unowned] != 0.0):
            raise NativeScalarBoundaryInterfaceV98Error(
                "v98 non-trace closed wall has nonzero water flux")
        if (np.any(owner < 0) or np.any(owner >= nc) or
                np.any(neighbour < -1) or np.any(neighbour >= nc) or
                np.any(bid < -1) or np.any(bid >= nb) or
                np.any((bid >= 0) & (neighbour >= 0)) or offsets[0] != 0 or
                np.any(np.diff(offsets) < 0) or np.any(faces < 0) or
                np.any(faces >= nf) or
                np.any((signs != -1) & (signs != 1)) or
                np.any(area <= 0.0) or np.any(length <= 0.0) or
                not np.isfinite(dry) or dry < 0.0):
            raise NativeScalarBoundaryInterfaceV98Error(
                "v98 scalar boundary topology is invalid")
        cells = np.repeat(np.arange(nc), np.diff(offsets))
        expected_cells = np.where(signs > 0, owner[faces], neighbour[faces])
        if (not np.array_equal(cells, expected_cells) or
                not np.array_equal(
                    np.bincount(faces, minlength=nf),
                    1+(neighbour >= 0).astype(np.int32)) or
                len(np.unique(faces.astype(np.int64)*2+(signs < 0))) !=
                nadj):
            raise NativeScalarBoundaryInterfaceV98Error(
                "v98 scalar boundary CSR incidence differs")

        exterior = np.flatnonzero(bid >= 0)
        bfaces = np.ascontiguousarray(
            exterior[np.argsort(bid[exterior], kind="stable")],
            dtype=np.int32)
        boffsets = np.ascontiguousarray(
            np.r_[0, np.cumsum(np.bincount(bid[exterior], minlength=nb))],
            dtype=np.int32)
        boundary = np.empty((nl, nf))
        amount = np.empty((nl, nf))
        next_last = np.empty((nl, nf))
        next_clock = np.empty((nl, nf))
        ledger = np.empty((nl, nb))
        divergence = np.empty(nc)
        water_ledger = np.empty(nb)
        continuity = np.empty(nc)
        outputs = (boundary, amount, next_last, next_clock, ledger,
                   divergence, water_ledger, continuity)
        inputs = (owner, neighbour, bid, offsets, faces, signs, boffsets,
                  bfaces, trace_mask, area, length, h0, h1, mass, prescribed,
                  water, times, last, clocks, neu)
        provenance = {
            "trace_receipt_sha256": trace_receipt,
            "trace_source_sha256": trace_source,
            "trace_runtime_sha256": trace_runtime,
            "substep_index": substep_index,
        }
        input_sha = _hash({
            "arrays_sha256": _digest(*inputs),
            "dt_s": float(dt_s), "dry_tolerance_m": dry,
            **provenance,
        })
        status = ctypes.c_int()
        self._function(
            nc, nf, nl, nb, nadj, len(bfaces), owner, neighbour, bid,
            offsets, faces, signs, boffsets, bfaces, trace_mask, area,
            length, h0, h1, mass, prescribed, water, times, last, clocks,
            neu, float(dt_s), dry, native_threads, *outputs,
            ctypes.byref(status))
        if status.value or not all(np.isfinite(value).all()
                                   for value in outputs):
            raise NativeScalarBoundaryInterfaceV98Error(
                "v98 scalar boundary kernel rejected input: status " +
                str(status.value))
        self._validate_runtime()
        if cancel():
            raise InterruptedError(
                "native v98 scalar boundary cancelled after launch")

        # Independently reconstruct the CSR water equation.  Trace faces are
        # real internal transfers for this level-local cell; they are still
        # absent from every physical boundary output.
        expected_divergence = np.zeros(nc)
        for cell in range(nc):
            for slot in range(offsets[cell], offsets[cell+1]):
                face = int(faces[slot])
                expected_divergence[cell] += (
                    signs[slot]*length[face]*water[face]/area[cell])
        expected_continuity = h1-h0+float(dt_s)*expected_divergence
        if (not np.array_equal(divergence, expected_divergence) or
                not np.array_equal(continuity, expected_continuity) or
                np.any(boundary[:, selected] != 0.0) or
                np.any(amount[:, selected] != 0.0) or
                not np.array_equal(next_last[:, selected], last[:, selected]) or
                not np.array_equal(
                    next_clock[:, selected], clocks[:, selected])):
            raise NativeScalarBoundaryInterfaceV98Error(
                "v98 scalar boundary trace ownership or continuity differs")
        for value in (*outputs, selected):
            value.setflags(write=False)
        output_sha = _digest(*outputs, selected)
        return NativeScalarBoundaryInterfaceResultV98(
            *outputs, selected, self.receipt["receipt_sha256"], input_sha,
            output_sha, trace_receipt, trace_source, trace_runtime,
            substep_index, native_threads)


def build_native_scalar_boundary_interface_runtime_v98(
    build_directory: str | Path, *, compiler: str | Path | None = None,
) -> NativeScalarBoundaryInterfaceRuntimeV98:
    for relative, expected in LOCKED_SOURCES.items():
        if _sha(ROOT/relative) != expected:
            raise NativeScalarBoundaryInterfaceV98Error(
                "v98 scalar boundary locked Delft source differs: " +
                relative)
    chosen = Path(compiler).resolve() if compiler is not None else None
    if chosen is None:
        located = shutil.which("gfortran")
        chosen = Path(located).resolve() if located else None
    if chosen is None and os.name == "nt":
        candidate = Path(r"C:\Strawberry\c\bin\gfortran.exe")
        chosen = candidate.resolve() if candidate.is_file() else None
    if chosen is None or not chosen.is_file():
        raise NativeScalarBoundaryInterfaceV98Error(
            "v98 scalar boundary requires gfortran")
    flags = ["-O3", "-fopenmp", "-shared", "-ffree-line-length-none",
             "-static-libgfortran", "-static-libgcc"]
    directory = Path(build_directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    source_sha = _sha(SOURCE)
    base_sha = _sha(BASE_SOURCE)
    wrapper_sha = _sha(WRAPPER)
    key = _hash({"source": source_sha, "base": base_sha,
                 "compiler": _sha(chosen), "flags": flags})[:16]
    build = Path(tempfile.mkdtemp(
        prefix=f"v98_scalar_boundary_{key}_", dir=directory))
    name = ("braids_scalar_boundary_interface_v98.dll" if os.name == "nt"
            else "libbraids_scalar_boundary_interface_v98.so")
    library_path = build/name
    partial = build/(name+".partial")
    process = subprocess.run(
        [str(chosen), *flags, str(BASE_SOURCE), str(SOURCE), "-o",
         str(partial)], cwd=build, capture_output=True, text=True,
        timeout=120)
    if process.returncode or not partial.is_file():
        raise NativeScalarBoundaryInterfaceV98Error(
            "v98 scalar boundary build failed: " +
            process.stdout+process.stderr)
    if ((base_sha, source_sha, wrapper_sha) !=
            (_sha(BASE_SOURCE), _sha(SOURCE), _sha(WRAPPER))):
        raise NativeScalarBoundaryInterfaceV98Error(
            "v98 scalar boundary source changed during build")
    os.replace(partial, library_path)
    version = subprocess.run(
        [str(chosen), "--version"], capture_output=True, text=True,
        check=True, timeout=20).stdout.splitlines()[0]
    receipt = {
        "abi": ABI_ID,
        "base_source_sha256": base_sha,
        "source_sha256": source_sha,
        "wrapper_sha256": wrapper_sha,
        "library_sha256": _sha(library_path),
        "compiler_sha256": _sha(chosen),
        "compiler_identity": version,
        "compile_flags": flags,
        "locked_source_sha256": LOCKED_SOURCES,
        "source_equation": FLOW2DH_OPEN_BOUNDARY_SCHEMA,
        "combined_lane_openmp": True,
        "internal_trace_water_divergence": True,
        "internal_trace_boundary_memory_count": 0,
        "internal_trace_physical_ledger_count": 0,
        "accepted_state_publication_count": 0,
        "production_admission": False,
    }
    receipt["receipt_sha256"] = _hash(receipt)
    dependency = None
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        dependency = os.add_dll_directory(str(chosen.parent))
    return NativeScalarBoundaryInterfaceRuntimeV98(
        receipt, ctypes.CDLL(str(library_path)), dependency, library_path)


__all__ = [
    "ABI_ID", "NativeScalarBoundaryInterfaceResultV98",
    "NativeScalarBoundaryInterfaceRuntimeV98",
    "NativeScalarBoundaryInterfaceV98Error",
    "build_native_scalar_boundary_interface_runtime_v98",
]
