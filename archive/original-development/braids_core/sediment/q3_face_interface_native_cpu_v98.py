"""Native Q3 face/inventory operator for a real AMR internal trace."""
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
SOURCE = ROOT/"solver/wave/braids_q3_face_interface_v98_module.f90"
WRAPPER = Path(__file__).resolve()
SOURCES = (
    ROOT/"solver/wave/braids_flow_bed_slope_operator_module.f90",
    ROOT/"solver/wave/braids_flow_bed_slope_face_composer_module.f90",
    SOURCE,
)
ABI_ID = "braids-v98-native-q3-internal-interface-v1"


@dataclass(frozen=True)
class NativeQ3FaceInterfaceResultV98:
    face_flux_kg_m_s: np.ndarray
    cell_delta_kg_m2: np.ndarray
    availability: np.ndarray
    named_boundary_transfer_kg: np.ndarray
    named_boundary_absolute_kg: np.ndarray
    internal_interface_transfer_kg: np.ndarray
    depletion_roundoff_positive_in_kg: np.ndarray
    classwise_residual_kg: np.ndarray
    interface_faces: np.ndarray
    runtime_sha256: str
    method_fingerprint: str
    input_sha256: str
    output_sha256: str
    native_threads: int
    native_q3_face_batch_owner_count: int = 1
    internal_interface_owner_count: int = 1
    physical_boundary_surrogate_count: int = 0
    python_numerical_cell_face_loop_count: int = 0
    accepted_bed_or_clock_publication_count: int = 0


class NativeQ3FaceInterfaceRuntimeV98:
    def __init__(self, receipt, library, dependency, library_path):
        self.receipt = dict(receipt)
        self.library = library
        self.dependency = dependency
        self.library_path = Path(library_path).resolve()
        fa = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        ia = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
        self._function = library.braids_q3_face_interface_v98
        self._function.argtypes = (
            [ctypes.c_int]*9 + [ia]*11 + [fa]*15 +
            [ctypes.c_double] + [fa]*8 +
            [ctypes.POINTER(ctypes.c_int)])
        self._function.restype = None

    @property
    def runtime_sha256(self) -> str:
        return str(self.receipt["receipt_sha256"])

    def _validate_runtime(self) -> None:
        body = {key: value for key, value in self.receipt.items()
                if key != "receipt_sha256"}
        current = {
            str(path.relative_to(ROOT)).replace("\\", "/"): _sha(path)
            for path in SOURCES}
        if (self.receipt.get("receipt_sha256") != _hash(body) or
                current != self.receipt.get("source_sha256") or
                _sha(WRAPPER) != self.receipt.get("wrapper_sha256") or
                not self.library_path.is_file() or
                _sha(self.library_path) != self.receipt.get("library_sha256")):
            raise ValueError(
                "native v98 Q3 interface source/library identity differs")

    def evaluate(
        self, topology, method_identity, cell_vector_kg_m_s,
        bed_mass_kg_m2, face_role, *, downhill, stress, depth, d50,
        mean_diameter, hiding, interface_faces, interface_vector_kg_m_s,
        interface_hiding, interface_availability, dt_s,
        native_threads=4, cancelled=False,
    ) -> NativeQ3FaceInterfaceResultV98:
        cancel = _cancel(cancelled)
        if cancel():
            raise InterruptedError(
                "native v98 Q3 interface cancelled before launch")
        self._validate_runtime()
        identity = validate_bed_slope_execution_document(method_identity)
        if type(native_threads) is not int or not 1 <= native_threads <= 256:
            raise ValueError("native v98 Q3 interface thread count differs")
        if (isinstance(dt_s, bool) or
                not isinstance(dt_s, (int, float)) or
                not np.isfinite(dt_s) or dt_s <= 0):
            raise ValueError(
                "native v98 Q3 interface dt must be finite and positive")
        nc, nf, nb = (topology.cell_count, topology.face_count,
                      topology.boundary_count)
        ng = len(identity["ordered_fraction_ids"])
        owner = _array(topology.owner, (nf,), "owner", integer=True)
        neighbour = _array(
            topology.neighbour, (nf,), "neighbour", integer=True)
        role = _array(face_role, (nf,), "face role", integer=True)
        bid = _array(
            topology.boundary_index, (nf,), "boundary index", integer=True)
        offsets = _array(
            topology.cell_face_offset, (nc+1,), "CSR offsets", integer=True)
        nadj = int(offsets[-1])
        faces = _array(
            topology.cell_faces, (nadj,), "CSR faces", integer=True)
        signs = _array(
            topology.cell_face_sign, (nadj,), "CSR signs", integer=True)
        area = _array(topology.cell_area_m2, (nc,), "areas")
        length = _array(topology.face_length_m, (nf,), "lengths")
        normal = _array(topology.normal, (2, nf), "normal")
        raw_selected = np.asarray(interface_faces)
        if raw_selected.ndim != 1 or raw_selected.size < 1:
            raise ValueError(
                "native v98 Q3 interface face inventory differs")
        ni = int(raw_selected.size)
        selected = _array(
            raw_selected, (ni,), "interface faces", integer=True)
        if (len(set(map(int, selected))) != ni or
                np.any(selected < 0) or np.any(selected >= nf) or
                np.any(neighbour[selected] != -1) or
                np.any(bid[selected] != -1) or
                np.any(role[selected] != 4)):
            raise ValueError(
                "native v98 Q3 interface is not closed internal topology")
        active = np.zeros(nf, dtype=np.int32)
        active[selected] = 1
        if (np.any(owner < 0) or np.any(owner >= nc) or
                np.any(neighbour < -1) or np.any(neighbour >= nc) or
                np.any(bid < -1) or np.any(bid >= nb) or
                np.any((bid >= 0) & (neighbour >= 0)) or
                np.any(~np.isin(role, (0, 3, 4))) or
                np.any((role == 3) & ((neighbour >= 0) | (bid < 0))) or
                offsets[0] != 0 or np.any(np.diff(offsets) < 0) or
                np.any(faces < 0) or np.any(faces >= nf) or
                np.any((signs != -1) & (signs != 1)) or
                np.any(area <= 0) or np.any(length <= 0) or
                not np.allclose(np.sum(normal*normal, axis=0), 1.0,
                                rtol=0.0,
                                atol=128*np.finfo(float).eps)):
            raise ValueError(
                "native v98 Q3 interface topology/roles differ")
        cells = np.repeat(np.arange(nc), np.diff(offsets))
        if (not np.array_equal(
                cells, np.where(signs > 0, owner[faces], neighbour[faces])) or
                not np.array_equal(
                    np.bincount(faces, minlength=nf),
                    1+(neighbour >= 0).astype(np.int32)) or
                len(np.unique(faces.astype(np.int64)*2+(signs < 0))) !=
                nadj):
            raise ValueError("native v98 Q3 interface CSR incidence differs")
        down = _array(downhill, (2, nf), "downhill")
        vector = _array(
            cell_vector_kg_m_s, (2, ng, nc), "cell vector")
        trace_vector = _array(
            interface_vector_kg_m_s, (2, ng, ni),
            "interface vector")
        stress = _array(stress, (ng, nf), "stress", nonnegative=True)
        depth = _array(depth, (ng, nf), "depth", nonnegative=True)
        d50 = _array(d50, (ng,), "D50")
        mean = _array(
            mean_diameter, (ng, nf), "mean diameter", nonnegative=True)
        hiding = _array(hiding, (ng, nc), "hiding", nonnegative=True)
        trace_hiding = _array(
            interface_hiding, (ng, ni), "interface hiding",
            nonnegative=True)
        trace_availability = _array(
            interface_availability, (ng, ni),
            "interface availability", nonnegative=True)
        bed = _array(
            bed_mass_kg_m2, (ng, nc), "bed inventory", nonnegative=True)
        if np.any(d50 <= 0) or np.any(trace_availability > 1.0):
            raise ValueError(
                "native v98 Q3 interface material controls differ")
        controls = np.ascontiguousarray([
            identity["parameters"][key] for key in (
                "alfabs", "alfabn", "ashld", "bshld", "cshld", "dshld",
                "alfpa", "thcrpa")])
        exterior = np.flatnonzero(bid >= 0)
        bfaces = np.ascontiguousarray(
            exterior[np.argsort(bid[exterior], kind="stable")],
            dtype=np.int32)
        boffsets = np.ascontiguousarray(np.r_[
            0, np.cumsum(np.bincount(bid[exterior], minlength=nb))],
            dtype=np.int32)
        flux = np.empty((ng, nf)); delta = np.empty((ng, nc))
        fill = np.empty((ng, nc)); named = np.empty((ng, nb))
        absolute = np.empty((ng, nb)); interface = np.empty((ng, ni))
        depletion = np.empty(ng); residual = np.empty(ng)
        outputs = (flux, delta, fill, named, absolute, interface,
                   depletion, residual)
        inputs = (owner, neighbour, role, bid, active, selected, offsets,
                  faces, signs, area, length, normal, down, vector,
                  trace_vector, stress, depth, d50, mean, hiding,
                  trace_hiding, trace_availability, bed, controls)
        input_sha = _hash({
            "arrays": _digest(*inputs), "dt_s": float(dt_s),
            "method": identity["fingerprint"]})
        status = ctypes.c_int(-1)
        self._function(
            nc, nf, ng, nb, nadj, len(bfaces), ni,
            int(identity["runtime_mode"]), native_threads,
            owner, neighbour, role, bid, active, selected, offsets, faces,
            signs, boffsets, bfaces, area, length, normal, down, vector,
            trace_vector, stress, depth, d50, mean, hiding, trace_hiding,
            trace_availability, bed, controls, float(dt_s), *outputs,
            ctypes.byref(status))
        if (status.value or not all(np.isfinite(value).all()
                                    for value in outputs) or
                np.any(bed+delta < 0.0)):
            raise ValueError(
                "native v98 Q3 interface kernel rejected input: "
                f"status {status.value}")
        expected = flux[:, selected]*length[selected][None, :]*float(dt_s)
        if not np.array_equal(interface, expected):
            raise ValueError(
                "native v98 Q3 interface transfer differs")
        tolerance = 8192*np.finfo(float).eps*np.maximum(
            1.0, np.sum(np.abs(delta)*area, axis=1)+
            np.sum(absolute, axis=1)+np.sum(np.abs(interface), axis=1)+
            depletion)
        if np.any(np.abs(residual) > tolerance):
            raise ValueError(
                "native v98 Q3 interface classwise ledger did not close")
        self._validate_runtime()
        if cancel():
            raise InterruptedError(
                "native v98 Q3 interface cancelled after launch")
        for value in (*outputs, selected):
            value.setflags(write=False)
        return NativeQ3FaceInterfaceResultV98(
            *outputs, selected, self.runtime_sha256,
            identity["fingerprint"], input_sha,
            _digest(*outputs, selected), native_threads)


def build_native_q3_face_interface_runtime_v98(
    build_directory, *, compiler=None,
) -> NativeQ3FaceInterfaceRuntimeV98:
    selected = compiler or shutil.which("gfortran")
    if selected is None and os.name == "nt":
        candidate = Path(r"C:\Strawberry\c\bin\gfortran.exe")
        selected = candidate if candidate.is_file() else None
    if selected is None:
        raise ValueError("native v98 Q3 interface requires gfortran")
    selected = Path(selected).resolve()
    flags = ["-O3", "-fopenmp", "-shared", "-ffree-line-length-none",
             "-fno-fast-math", "-ffp-contract=off",
             "-static-libgfortran", "-static-libgcc"]
    directory = Path(build_directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    sources = {
        str(path.relative_to(ROOT)).replace("\\", "/"): _sha(path)
        for path in SOURCES}
    wrapper_sha = _sha(WRAPPER)
    build = Path(tempfile.mkdtemp(prefix="v98_q3_interface_", dir=directory))
    name = ("braids_q3_face_interface_v98.dll" if os.name == "nt" else
            "libbraids_q3_face_interface_v98.so")
    library_path, partial = build/name, build/(name+".partial")
    process = subprocess.run(
        [str(selected), *flags, *(str(path) for path in SOURCES),
         "-o", str(partial)], cwd=build, capture_output=True, text=True,
        timeout=120)
    if process.returncode or not partial.is_file():
        raise ValueError(
            "native v98 Q3 interface build failed: "+
            process.stdout+process.stderr)
    if (sources != {
            str(path.relative_to(ROOT)).replace("\\", "/"): _sha(path)
            for path in SOURCES} or wrapper_sha != _sha(WRAPPER)):
        raise ValueError(
            "native v98 Q3 interface source changed during build")
    os.replace(partial, library_path)
    version = subprocess.run(
        [str(selected), "--version"], capture_output=True, text=True,
        check=True, timeout=20).stdout.splitlines()[0]
    receipt = {
        "abi": ABI_ID, "source_sha256": sources,
        "wrapper_sha256": wrapper_sha,
        "library_sha256": _sha(library_path),
        "compiler_sha256": _sha(selected),
        "compiler_identity": version, "compile_flags": flags,
        "unchanged_braids_flow_slope_primitives": True,
        "internal_trace_admitted": True,
        "physical_boundary_surrogate_count": 0,
        "cell_face_grain_openmp": True,
        "fixed_1024_cell_ledger_blocks": True,
        "state_or_clock_owner_count": 0,
        "production_admission": False,
    }
    receipt["receipt_sha256"] = _hash(receipt)
    dependency = None
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        dependency = os.add_dll_directory(str(selected.parent))
    return NativeQ3FaceInterfaceRuntimeV98(
        receipt, ctypes.CDLL(str(library_path)), dependency, library_path)


__all__ = [
    "NativeQ3FaceInterfaceResultV98",
    "NativeQ3FaceInterfaceRuntimeV98",
    "build_native_q3_face_interface_runtime_v98",
]
