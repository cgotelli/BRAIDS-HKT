"""Native Q3 cell-vector face selection, inventory limit and divergence.

This matches the v94 cell-vector reconstruction without changing it. Source-
owned inlet/trace capacity remains outside this batch's scope and fails closed.
Wet/dry internal masks retain the real topology while closing sediment faces.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np

from .bed_slope_execution import validate_bed_slope_execution_document
from .scalar_boundary_native_cpu_v97 import _hash, _sha, _digest
from .scalar_diffusion_native_cpu_v96 import _array, _cancel

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/"solver/wave/braids_q3_face_inventory_v97_module.f90"
WRAPPER = Path(__file__).resolve()
SOURCES = (ROOT/"solver/wave/braids_flow_bed_slope_operator_module.f90",
    ROOT/"solver/wave/braids_flow_bed_slope_face_composer_module.f90", SOURCE)
ABI_ID = "braids-v97-native-q3-cell-vector-inventory-face-v1"


@dataclass(frozen=True)
class NativeQ3FaceInventoryResultV97:
    face_flux_kg_m_s: np.ndarray
    cell_delta_kg_m2: np.ndarray
    availability: np.ndarray
    named_boundary_transfer_kg: np.ndarray
    named_boundary_absolute_kg: np.ndarray
    depletion_roundoff_positive_in_kg: np.ndarray
    classwise_residual_kg: np.ndarray
    runtime_sha256: str
    method_fingerprint: str
    input_sha256: str
    output_sha256: str
    native_threads: int
    native_q3_face_batch_owner_count: int = 1
    python_numerical_cell_face_loop_count: int = 0
    accepted_bed_or_clock_publication_count: int = 0


class NativeQ3FaceInventoryRuntimeV97:
    def __init__(self, receipt, library, dependency, library_path):
        self.receipt = receipt
        self.library = library
        self.dependency = dependency
        self.library_path = Path(library_path).resolve()
        fa = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        ia = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
        self._function = library.braids_q3_face_inventory_v97
        self._function.argtypes = ([ctypes.c_int]*8 + [ia]*9 + [fa]*12 +
            [ctypes.c_double] + [fa]*7 + [ctypes.POINTER(ctypes.c_int)])
        self._function.restype = None

    def _validate_runtime(self):
        if self.receipt.get("receipt_sha256") != _hash({
                k: v for k, v in self.receipt.items() if k != "receipt_sha256"}):
            raise ValueError("native v97 Q3 face receipt was changed")
        current = {str(p.relative_to(ROOT)).replace("\\", "/"): _sha(p) for p in SOURCES}
        if (current != self.receipt["source_sha256"] or
                _sha(WRAPPER) != self.receipt["wrapper_sha256"] or
                _sha(self.library_path) != self.receipt["library_sha256"]):
            raise ValueError("native v97 Q3 face source/library identity differs")

    def evaluate(self, topology, method_identity, cell_vector_kg_m_s,
                 bed_mass_kg_m2, face_role, *, downhill, stress, depth,
                 d50, mean_diameter, hiding, dt_s, native_threads=4,
                 cancelled=False):
        cancel = _cancel(cancelled)
        if cancel():
            raise InterruptedError("native v97 Q3 face cancelled before launch")
        self._validate_runtime()
        identity = validate_bed_slope_execution_document(method_identity)
        if type(native_threads) is not int or not 1 <= native_threads <= 256:
            raise ValueError("native v97 Q3 face thread count differs")
        if (isinstance(dt_s, bool) or not isinstance(dt_s, (int, float)) or
                not np.isfinite(dt_s) or dt_s <= 0):
            raise ValueError("native v97 Q3 face dt must be finite and positive")
        nc, nf, nb = topology.cell_count, topology.face_count, topology.boundary_count
        ng = len(identity["ordered_fraction_ids"])
        owner = _array(topology.owner, (nf,), "owner", integer=True)
        neighbour = _array(topology.neighbour, (nf,), "neighbour", integer=True)
        role = _array(face_role, (nf,), "face role", integer=True)
        bid = _array(topology.boundary_index, (nf,), "boundary index", integer=True)
        offsets = _array(topology.cell_face_offset, (nc+1,), "CSR offsets", integer=True)
        nadj = int(offsets[-1])
        if nadj < 1:
            raise ValueError("native v97 Q3 face CSR is empty")
        faces = _array(topology.cell_faces, (nadj,), "CSR faces", integer=True)
        signs = _array(topology.cell_face_sign, (nadj,), "CSR signs", integer=True)
        area = _array(topology.cell_area_m2, (nc,), "areas")
        length = _array(topology.face_length_m, (nf,), "lengths")
        normal = _array(topology.normal, (2, nf), "normal")
        if (np.any(owner < 0) or np.any(owner >= nc) or np.any(neighbour < -1) or
                np.any(neighbour >= nc) or np.any(bid < -1) or np.any(bid >= nb) or
                np.any((bid >= 0) & (neighbour >= 0)) or np.any(~np.isin(role, (0, 3, 4))) or
                np.any((role == 3) & ((neighbour >= 0) | (bid < 0))) or offsets[0] != 0 or
                np.any(np.diff(offsets) < 0) or np.any(faces < 0) or np.any(faces >= nf) or
                np.any((signs != -1) & (signs != 1)) or np.any(area <= 0) or np.any(length <= 0) or
                not np.allclose(np.sum(normal*normal, axis=0), 1., rtol=0., atol=128*np.finfo(float).eps)):
            raise ValueError("native v97 Q3 face topology/roles invalid; inlet traces unsupported")
        cells = np.repeat(np.arange(nc), np.diff(offsets))
        if (not np.array_equal(cells, np.where(signs > 0, owner[faces], neighbour[faces])) or
                not np.array_equal(np.bincount(faces, minlength=nf), 1+(neighbour >= 0).astype(np.int32)) or
                len(np.unique(faces.astype(np.int64)*2+(signs < 0))) != nadj):
            raise ValueError("native v97 Q3 face CSR incidence differs")
        down = _array(downhill, (2, nf), "downhill")
        vector = _array(cell_vector_kg_m_s, (2, ng, nc), "cell vector")
        stress = _array(stress, (ng, nf), "stress", nonnegative=True)
        depth = _array(depth, (ng, nf), "depth", nonnegative=True)
        d50 = _array(d50, (ng,), "D50")
        mean = _array(mean_diameter, (ng, nf), "mean diameter", nonnegative=True)
        hiding = _array(hiding, (ng, nc), "hiding", nonnegative=True)
        bed = _array(bed_mass_kg_m2, (ng, nc), "bed inventory", nonnegative=True)
        if np.any(d50 <= 0):
            raise ValueError("native v97 Q3 face D50 must be positive")
        controls = np.ascontiguousarray([identity["parameters"][key] for key in
            ("alfabs", "alfabn", "ashld", "bshld", "cshld", "dshld", "alfpa", "thcrpa")])
        exterior = np.flatnonzero(bid >= 0)
        bfaces = np.ascontiguousarray(exterior[np.argsort(bid[exterior], kind="stable")], dtype=np.int32)
        boffsets = np.ascontiguousarray(np.r_[0, np.cumsum(np.bincount(bid[exterior], minlength=nb))], dtype=np.int32)
        flux = np.empty((ng, nf)); delta = np.empty((ng, nc)); fill = np.empty((ng, nc))
        named = np.empty((ng, nb)); absolute = np.empty((ng, nb))
        depletion = np.empty(ng); residual = np.empty(ng)
        outputs = (flux, delta, fill, named, absolute, depletion, residual)
        inputs = (owner, neighbour, role, bid, offsets, faces, signs, area, length,
            normal, down, vector, stress, depth, d50, mean, hiding, bed, controls)
        input_sha = _hash(dict(arrays=_digest(*inputs), dt_s=float(dt_s), method=identity["fingerprint"]))
        status = ctypes.c_int()
        self._function(nc, nf, ng, nb, nadj, len(bfaces), int(identity["runtime_mode"]),
            native_threads, owner, neighbour, role, bid, offsets, faces, signs, boffsets, bfaces,
            area, length, normal, down, vector, stress, depth, d50, mean, hiding, bed, controls,
            float(dt_s), *outputs, ctypes.byref(status))
        if status.value or not all(np.isfinite(a).all() for a in outputs) or np.any(bed+delta < 0.):
            raise ValueError(f"native v97 Q3 face kernel rejected input: status {status.value}")
        tolerance = 8192*np.finfo(float).eps*np.maximum(1.,
            np.sum(np.abs(delta)*area, axis=1)+np.sum(absolute, axis=1)+depletion)
        if np.any(np.abs(residual) > tolerance):
            raise ValueError("native v97 Q3 face classwise ledger did not close")
        self._validate_runtime()
        if cancel():
            raise InterruptedError("native v97 Q3 face cancelled after launch")
        for a in outputs:
            a.setflags(write=False)
        return NativeQ3FaceInventoryResultV97(*outputs, self.receipt["receipt_sha256"],
            identity["fingerprint"], input_sha, _digest(*outputs), native_threads)


def build_native_q3_face_inventory_runtime_v97(build_directory, *, compiler=None):
    selected = compiler or shutil.which("gfortran")
    if selected is None and os.name == "nt":
        candidate = Path(r"C:\Strawberry\c\bin\gfortran.exe")
        selected = candidate if candidate.is_file() else None
    if selected is None:
        raise ValueError("native v97 Q3 face requires gfortran")
    selected = Path(selected).resolve()
    flags = ["-O3", "-fopenmp", "-shared", "-ffree-line-length-none", "-static-libgfortran", "-static-libgcc"]
    directory = Path(build_directory).resolve(); directory.mkdir(parents=True, exist_ok=True)
    sources = {str(p.relative_to(ROOT)).replace("\\", "/"): _sha(p) for p in SOURCES}
    wrapper_sha = _sha(WRAPPER)
    build = Path(tempfile.mkdtemp(prefix="v97_q3_face_", dir=directory))
    name = "braids_q3_face_inventory_v97.dll" if os.name == "nt" else "libbraids_q3_face_inventory_v97.so"
    library_path = build/name; partial = build/(name+".partial")
    process = subprocess.run([str(selected), *flags, *(str(p) for p in SOURCES), "-o", str(partial)],
        cwd=build, capture_output=True, text=True, timeout=120)
    if process.returncode or not partial.is_file():
        raise ValueError("native v97 Q3 face build failed: "+process.stdout+process.stderr)
    if (sources != {str(p.relative_to(ROOT)).replace("\\", "/"): _sha(p) for p in SOURCES} or
            wrapper_sha != _sha(WRAPPER)):
        raise ValueError("native v97 Q3 face source changed during build")
    os.replace(partial, library_path)
    version = subprocess.run([str(selected), "--version"], capture_output=True, text=True,
        check=True, timeout=20).stdout.splitlines()[0]
    receipt = dict(abi=ABI_ID, source_sha256=sources, wrapper_sha256=wrapper_sha,
        library_sha256=_sha(library_path), compiler_sha256=_sha(selected), compiler_identity=version,
        compile_flags=flags, unchanged_braids_flow_slope_primitives=True,
        closed_internal_preserves_real_neighbour=True, inlet_trace_admitted=False,
        cell_face_grain_openmp=True, fixed_1024_cell_ledger_blocks=True, production_admission=False)
    receipt["receipt_sha256"] = _hash(receipt)
    dependency = None
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        dependency = os.add_dll_directory(str(selected.parent))
    return NativeQ3FaceInventoryRuntimeV97(receipt, ctypes.CDLL(str(library_path)), dependency, library_path)


__all__ = ["NativeQ3FaceInventoryRuntimeV97", "NativeQ3FaceInventoryResultV97",
           "build_native_q3_face_inventory_runtime_v97"]
