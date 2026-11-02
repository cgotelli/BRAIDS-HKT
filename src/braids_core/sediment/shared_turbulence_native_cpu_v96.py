"""Parallel native v91 cap and momentum consumer for the new v96 owner.

Pure auxiliary operators, not a hydraulic integrator or product-admission API.
The accepted HLES handoff supplies one cap; current midpoint velocity supplies
every momentum evaluation. Scalar transport consumes the same cap separately.
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

from .shared_turbulence_consumers_v91 import (
    ChkvicUsedTotalReceiptV91, SharedTurbulenceExecutionV91,
    apply_chkvic_used_total_v91, execution_document_v91, verify_locked_sources_v91,
)
from .hles_producer_v90 import BACKEND_CPU

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "solver/wave/braids_shared_turbulence_v96_module.f90"
ABI_ID = "braids-v96-native-v91-shared-turbulence-v1"


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _readonly(value):
    array = np.ascontiguousarray(value)
    array.setflags(write=False)
    return array


def _threads(value):
    if type(value) is not int or not 1 <= value <= 256:
        raise ValueError("native v96 thread count must be an integer in 1..256")
    return value


def _array(value, shape, name, dtype=np.float64):
    raw = np.asarray(value)
    if raw.shape != shape or raw.dtype.kind not in "fiu":
        raise ValueError(f"native v96 {name} has incompatible shape/type")
    array = np.ascontiguousarray(raw, dtype=dtype)
    if not np.isfinite(array).all():
        raise ValueError(f"native v96 {name} is not finite")
    return array


@dataclass(frozen=True)
class PreparedNativeTurbulenceV96:
    execution: SharedTurbulenceExecutionV91
    cap_receipt: ChkvicUsedTotalReceiptV91
    native_runtime_sha256: str
    native_threads: int
    preparation_sha256: str


def _preparation_token(value):
    return _hash({
        "abi": ABI_ID, "execution": value.execution.fingerprint,
        "cap_receipt": value.cap_receipt.receipt_sha256,
        "parent_hles": value.cap_receipt.parent_hles_state_sha256,
        "handoff": value.cap_receipt.current_handoff_receipt_sha256,
        "runtime": value.native_runtime_sha256, "threads": value.native_threads,
    })


@dataclass(frozen=True)
class NativeMomentumRateV96:
    momentum_rate_m2_s2: np.ndarray
    face_flux_m4_s2: np.ndarray
    cap_receipt: ChkvicUsedTotalReceiptV91
    parent_hles_state_sha256: str
    native_runtime_sha256: str
    native_threads: int
    momentum_rate_evaluation_count: int = 1
    depth_source_evaluation_count: int = 0
    numerical_python_cell_face_loops: int = 0


class NativeSharedTurbulenceRuntimeV96:
    def __init__(self, receipt, library, dependency, library_path):
        self.receipt = receipt
        self.library = library
        self.dependency = dependency
        self.library_path = Path(library_path).resolve()
        self._cap = library.braids_v96_cap
        self._momentum = library.braids_v96_momentum
        fa = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        ia = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
        ip = ctypes.POINTER(ctypes.c_int)
        self._cap.argtypes = [ctypes.c_int, fa, fa, fa, ctypes.c_double,
                             ctypes.c_int, fa, fa, fa, fa, ia, ip]
        self._momentum.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int, fa, fa, ia, ia, fa, fa,
            fa, ia, ia, ia, ctypes.c_double, ctypes.c_int, fa, fa, ip]
        self._cap.restype = self._momentum.restype = None

    def _validate_runtime(self):
        if self.receipt.get("receipt_sha256") != _hash({
                k: v for k, v in self.receipt.items() if k != "receipt_sha256"}):
            raise ValueError("native v96 turbulence runtime receipt was changed")
        if (_sha(SOURCE) != self.receipt["native_source_sha256"] or
                _sha(self.library_path) != self.receipt["library_sha256"]):
            raise ValueError("native v96 turbulence source/library identity differs")

    def prepare(self, execution, parent_hles, handoff, *, dt_s, native_threads=4):
        """Compute one native cap and authenticate the existing v91 receipt ABI."""
        self._validate_runtime()
        execution_document_v91(execution)
        if execution.backend != BACKEND_CPU:
            raise ValueError("native v96 turbulence requires explicit CPU execution")
        threads = _threads(native_threads)
        # Existing vector equation is an independent readback/authentication
        # oracle only; it does not advance state or own another physical cap.
        oracle = apply_chkvic_used_total_v91(
            execution, parent_hles, handoff, dt_s=dt_s)
        nc = execution.cell_count
        width = _array(execution.base_execution.base_execution.rectangle_width_m,
                       (2, nc), "cell widths")
        raw_vic = _array(handoff.used_total_vicuv_m2_s, (nc,), "raw vicuv")
        raw_dic = _array(handoff.used_total_dicuv_m2_s, (nc,), "raw dicuv")
        outputs = [np.empty(nc, dtype=np.float64) for _ in range(4)]
        flags = np.empty((2, nc), dtype=np.int32)
        status = ctypes.c_int()
        self._cap(nc, width, raw_vic, raw_dic, float(dt_s), threads,
                  *outputs, flags, ctypes.byref(status))
        if status.value != 0:
            raise ValueError("native v96 cap status failed")
        expected = [oracle.vicmax_m2_s, oracle.dicmax_m2_s,
                    oracle.capped_total_vicuv_m2_s, oracle.capped_total_dicuv_m2_s]
        for actual, reference in zip(outputs, expected):
            if not np.array_equal(actual, reference):
                raise ValueError("native v96 cap differs from strict v91 equation")
        if (int(flags[0].sum()) != oracle.capped_vicuv_cells or
                int(flags[1].sum()) != oracle.capped_dicuv_cells):
            raise ValueError("native v96 cap counts differ")
        value = PreparedNativeTurbulenceV96(
            execution, oracle, self.receipt["receipt_sha256"], threads, "")
        return PreparedNativeTurbulenceV96(
            **{**value.__dict__, "preparation_sha256": _preparation_token(value)})

    def evaluate(self, prepared: PreparedNativeTurbulenceV96, q, *, cancelled=False):
        """Current-iterate velocity rate; native face and cell loops only."""
        if callable(cancelled):
            cancel = cancelled
        elif type(cancelled) is bool:
            cancel = lambda: cancelled
        else:
            raise ValueError("native v96 cancellation is not bool/callable")
        if cancel():
            raise InterruptedError("native v96 momentum cancelled before launch")
        self._validate_runtime()
        if (type(prepared) is not PreparedNativeTurbulenceV96 or
                prepared.preparation_sha256 != _preparation_token(prepared) or
                prepared.native_runtime_sha256 != self.receipt["receipt_sha256"]):
            raise ValueError("native v96 turbulence preparation differs")
        execution_document_v91(prepared.execution)
        from .shared_turbulence_consumers_v91 import _cap_token
        if prepared.cap_receipt.receipt_sha256 != _cap_token(prepared.cap_receipt):
            raise ValueError("native v96 cap receipt was changed")
        ex = prepared.execution
        nc, nf = ex.cell_count, ex.face_count
        base = ex.base_execution.base_execution
        state = _array(q, (3, nc), "midpoint q")
        if np.any(state[0] <= base.dry_tolerance_m):
            raise ValueError("native v96 momentum requires fully wet cells")
        vic = _array(prepared.cap_receipt.capped_total_vicuv_m2_s, (nc,), "capped vicuv")
        rate = np.empty((3, nc), dtype=np.float64)
        flux = np.empty((2, nf), dtype=np.float64)
        status = ctypes.c_int()
        ints = [np.ascontiguousarray(a, dtype=np.int32) for a in
                (base.owner, base.neighbour, base.cell_face_offset,
                 base.cell_faces, base.cell_face_sign)]
        self._momentum(nc, nf, len(ints[3]), state, vic,
                       ints[0], ints[1], base.face_length_m,
                       base.centre_distance_m, base.cell_area_m2,
                       ints[2], ints[3], ints[4], float(base.dry_tolerance_m),
                       prepared.native_threads, flux, rate, ctypes.byref(status))
        if status.value != 0 or not np.isfinite(rate).all() or not np.isfinite(flux).all():
            raise ValueError("native v96 momentum status failed")
        if cancel():
            raise InterruptedError("native v96 momentum cancelled after launch")
        return NativeMomentumRateV96(
            _readonly(rate), _readonly(flux), prepared.cap_receipt,
            prepared.cap_receipt.parent_hles_state_sha256,
            prepared.native_runtime_sha256, prepared.native_threads)


def build_native_shared_turbulence_runtime_v96(build_directory, *, compiler=None):
    """Build a uniquely named isolated DLL; never replace a loaded library."""
    locked = verify_locked_sources_v91(ROOT)
    chosen = compiler or shutil.which("gfortran")
    if chosen is None and os.name == "nt":
        fallback = Path(r"C:\Strawberry\c\bin\gfortran.exe")
        chosen = fallback if fallback.is_file() else None
    if chosen is None:
        raise ValueError("native v96 shared turbulence requires gfortran")
    compiler_path = Path(chosen).resolve()
    flags = ["-O3", "-fopenmp", "-shared", "-ffree-line-length-none",
             "-static-libgfortran", "-static-libgcc"]
    build_root = Path(build_directory).resolve()
    build_root.mkdir(parents=True, exist_ok=True)
    source_sha = _sha(SOURCE)
    key = _hash({"source": source_sha, "compiler": _sha(compiler_path), "flags": flags})[:16]
    build = Path(tempfile.mkdtemp(prefix=f"v96_vicuv_{key}_", dir=build_root))
    name = "braids_v96_vicuv.dll" if os.name == "nt" else "libbraids_v96_vicuv.so"
    library_path = build / name
    partial = build / (name + ".partial")
    result = subprocess.run([str(compiler_path), *flags, str(SOURCE), "-o", str(partial)],
                            cwd=build, capture_output=True, text=True, timeout=120)
    if result.returncode != 0 or not partial.is_file():
        raise ValueError("native v96 build failed: " + result.stdout + result.stderr)
    if source_sha != _sha(SOURCE):
        raise ValueError("native v96 source changed during compilation")
    os.replace(partial, library_path)
    version = subprocess.run([str(compiler_path), "--version"], capture_output=True,
                             text=True, check=True, timeout=20).stdout.splitlines()[0]
    receipt = {
        "abi": ABI_ID, "native_source_sha256": source_sha,
        "library_sha256": _sha(library_path), "compiler_identity": version,
        "compiler_sha256": _sha(compiler_path), "compile_flags": flags,
        "locked_source_sha256": locked,
        "openmp_face_cell_cap_loops": True,
        "thread_policy": "per_region_num_threads_no_nested_team",
        "reentrant_pure_evaluation": True,
        "python_numerical_cell_face_loop_count": 0,
        "production_admission": False,
    }
    receipt["receipt_sha256"] = _hash(receipt)
    dependency = None
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        dependency = os.add_dll_directory(str(compiler_path.parent))
    return NativeSharedTurbulenceRuntimeV96(
        receipt, ctypes.CDLL(str(library_path)), dependency, library_path)
