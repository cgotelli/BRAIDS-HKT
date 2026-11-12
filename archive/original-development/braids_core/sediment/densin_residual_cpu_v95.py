"""Bounded native-CPU DensIn feedback inside an augmented-f-wave residual.

V95 is a validation micro-integrator, not production CPU admission.  It keeps
the frozen GeoClaw augmented Riemann routine and evaluates a first-order,
fully-wet, closed Cartesian finite-volume rate in native OpenMP Fortran.  The
locked Delft3D KMAX=1 sediment-density increment is recomputed from the current
midpoint depth on every full-rate evaluation.  Accepted concentration and
water density stay frozen for the whole hydraulic trial.

The older v85 NumPy projection remains an equation oracle only.  No NumPy
cell/face loop participates in the supported v95 update.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import threading
from typing import Callable, Mapping

import numpy as np

from .densin_fwave_v85 import (
    AcceptedConcentrationReceiptV85,
    DensInExecutionV85,
    LOCKED_UPSTREAM_COMMIT,
    delft_kmax1_face_acceleration_v85,
    execution_document_v85,
    project_face_acceleration_v85,
    validate_accepted_concentration_receipt_v85,
    verify_locked_sources_v85,
)


SCHEMA = 1
SCOPE_ID = "braids-q4-densin-residual-cpu-v95-v1"
SEMANTIC_ID = SCOPE_ID + ":native-first-order-midpoint"
CHECKPOINT_SCOPE_ID = SCOPE_ID + ":checkpoint"
NATIVE_ABI_ID = SCOPE_ID + ":openmp-full-rate-abi-v1"
MAX_ACCEPTED_UPDATES = 4
DENSIN_OFF = "off"
DENSIN_SEDIMENT_INCREMENT = "sediment_increment"
_MODES = frozenset((DENSIN_OFF, DENSIN_SEDIMENT_INCREMENT))
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ROOT = Path(__file__).resolve().parents[2]

_NATIVE_RELATIVE = "solver/wave/braids_densin_residual_v95_module.f90"
_PARAMETER_RELATIVE = "solver/cut_cells/geoclaw_parameters.f90"
_RPN2_RELATIVE = "solver/clawpack-v5.14.0/riemann/src/rpn2_geoclaw.f"
_UTILS_RELATIVE = "solver/clawpack-v5.14.0/riemann/src/geoclaw_riemann_utils.f"
_EXPECTED_NATIVE_SOURCE_SHA256 = (
    "e701c096e400402c4386cc2015459d5900ce073094b86f97c0ad94c2516a61e3")
_FROZEN_SOURCE_HASHES = {
    _PARAMETER_RELATIVE:
        "b6e5a70595e93279eac113401dfc2d857978f7a6e885fe730c783c17cc5b3c1d",
    _RPN2_RELATIVE:
        "c52649def884ce374dbb1d3dc1e91f7a9c33aa45a0aff691badc8811b9157d1f",
    _UTILS_RELATIVE:
        "df73207446ac730f9cda9a8ca4dbbc625151ba95d13aa514ce87ca23cb598761",
}


class DensInResidualV95Error(ValueError):
    """The bounded v95 execution, native receipt or transaction is invalid."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False).encode("ascii")
    except (TypeError, ValueError, OverflowError) as error:
        raise DensInResidualV95Error(
            "v95 value is not finite canonical JSON") from error


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _digest(*values: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(value)
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(_canonical(list(array.shape)))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise DensInResidualV95Error(f"v95 {name} must be a lowercase SHA-256")
    return value


def _finite(value: object, name: str, *, positive: bool = False) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(float(value))):
        raise DensInResidualV95Error(f"v95 {name} must be finite numeric data")
    result = float(value)
    if positive and result <= 0.0:
        raise DensInResidualV95Error(f"v95 {name} must be positive")
    return result


def _array(value: object, shape: tuple[int, ...], name: str, *,
           positive: bool = False) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != shape or raw.dtype.kind not in "fiu":
        raise DensInResidualV95Error(
            f"v95 {name} must be numeric with shape {shape}")
    result = np.ascontiguousarray(raw, dtype=np.float64)
    if not np.isfinite(result).all():
        raise DensInResidualV95Error(f"v95 {name} must be finite")
    if positive and np.any(result <= 0.0):
        raise DensInResidualV95Error(f"v95 {name} must be positive")
    result.setflags(write=False)
    return result


def _cancelled(value: bool | Callable[[], bool]) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if not callable(value):
        raise DensInResidualV95Error(
            "v95 cancellation must be boolean or a zero-argument callable")
    result = value()
    if not isinstance(result, (bool, np.bool_)):
        raise DensInResidualV95Error(
            "v95 cancellation callback must return boolean data")
    return bool(result)


@dataclass(frozen=True)
class NativeDensInBuildReceiptV95:
    abi_id: str
    native_module_sha256: str
    geoclaw_parameters_sha256: str
    augmented_rpn2_sha256: str
    riemann_utils_sha256: str
    library_sha256: str
    compiler_identity: str
    compile_flags: tuple[str, ...]
    build_utc: str
    openmp_enabled: bool
    locked_delft_commit: str
    locked_delft_source_sha256: tuple[tuple[str, str], ...]
    receipt_sha256: str


def _build_receipt_payload(value: NativeDensInBuildReceiptV95) -> dict:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID + ":native-build",
        "abi_id": value.abi_id,
        "native_module_sha256": value.native_module_sha256,
        "geoclaw_parameters_sha256": value.geoclaw_parameters_sha256,
        "augmented_rpn2_sha256": value.augmented_rpn2_sha256,
        "riemann_utils_sha256": value.riemann_utils_sha256,
        "library_sha256": value.library_sha256,
        "compiler_identity": value.compiler_identity,
        "compile_flags": list(value.compile_flags),
        "openmp_enabled": value.openmp_enabled,
        "locked_delft_commit": value.locked_delft_commit,
        "locked_delft_source_sha256":
            dict(value.locked_delft_source_sha256),
        "frozen_augmented_riemann_sources_unchanged": True,
    }


def _validate_build_receipt(value: NativeDensInBuildReceiptV95) -> None:
    if type(value) is not NativeDensInBuildReceiptV95:
        raise DensInResidualV95Error("v95 native build receipt type differs")
    for name in (
        "native_module_sha256", "geoclaw_parameters_sha256",
        "augmented_rpn2_sha256", "riemann_utils_sha256", "library_sha256",
    ):
        _sha(getattr(value, name), name)
    locked = tuple(sorted(verify_locked_sources_v85(_ROOT).items()))
    if (value.abi_id != NATIVE_ABI_ID or value.openmp_enabled is not True or
            not value.compiler_identity or not value.compile_flags or
            not value.build_utc.endswith("Z") or
            value.locked_delft_commit != LOCKED_UPSTREAM_COMMIT or
            value.locked_delft_source_sha256 != locked or
            value.native_module_sha256 != _EXPECTED_NATIVE_SOURCE_SHA256 or
            value.geoclaw_parameters_sha256 !=
                _FROZEN_SOURCE_HASHES[_PARAMETER_RELATIVE] or
            value.augmented_rpn2_sha256 !=
                _FROZEN_SOURCE_HASHES[_RPN2_RELATIVE] or
            value.riemann_utils_sha256 !=
                _FROZEN_SOURCE_HASHES[_UTILS_RELATIVE] or
            value.receipt_sha256 !=
                _fingerprint(_build_receipt_payload(value))):
        raise DensInResidualV95Error(
            "v95 native build receipt is stale or mismatched")


@dataclass(frozen=True)
class NativeRateResultV95:
    base_fwave_rate: np.ndarray
    density_momentum_rate: np.ndarray
    full_rate: np.ndarray
    face_normal_acceleration_m_s2: np.ndarray
    cell_acceleration_m_s2: np.ndarray
    face_water_flux_m2_s: np.ndarray
    augmented_riemann_call_count: int
    density_source_evaluation_count: int


class NativeDensInRuntimeV95:
    """Exact loaded native build corresponding to a typed build receipt."""

    __slots__ = ("receipt", "library_path", "_function", "_library",
                 "_dependency", "_call_lock")

    def __init__(self, receipt: NativeDensInBuildReceiptV95,
                 library_path: Path, function, library, dependency) -> None:
        _validate_build_receipt(receipt)
        self.receipt = receipt
        self.library_path = Path(library_path).resolve()
        self._function = function
        self._library = library
        self._dependency = dependency
        # GeoClaw's gravity/dry parameters are module globals.  The native
        # OpenMP team only reads them, but two host calls must not rewrite them
        # concurrently with different executions.
        self._call_lock = threading.Lock()

    def evaluate(
        self, execution: "DensInResidualCPUExecutionV95", midpoint_q: object,
        mixture_density_kg_m3: object,
    ) -> NativeRateResultV95:
        if type(execution) is not DensInResidualCPUExecutionV95:
            raise DensInResidualV95Error("v95 native execution type differs")
        if self.receipt != execution.native_build_receipt:
            raise DensInResidualV95Error("v95 native runtime receipt differs")
        if (not self.library_path.is_file() or
                _file_sha256(self.library_path) != self.receipt.library_sha256):
            raise DensInResidualV95Error("v95 native library bytes differ")
        base = execution.base_execution
        nc, nf = base.cell_count, base.face_count
        q = np.ascontiguousarray(
            _array(midpoint_q, (3, nc), "native midpoint"), dtype=np.float64)
        mixture = np.ascontiguousarray(
            _array(mixture_density_kg_m3, (nc,), "native mixture density",
                   positive=True), dtype=np.float64)
        bed = np.ascontiguousarray(execution.bed_elevation_m)
        owner = np.ascontiguousarray(base.owner, dtype=np.int32)
        neighbour = np.ascontiguousarray(base.neighbour, dtype=np.int32)
        normal = np.ascontiguousarray(base.normal)
        length = np.ascontiguousarray(base.face_length_m)
        distance = np.ascontiguousarray(base.face_distance_m)
        area = np.ascontiguousarray(base.cell_area_m2)
        offset = np.ascontiguousarray(base.cell_face_offset, dtype=np.int32)
        faces = np.ascontiguousarray(base.cell_faces, dtype=np.int32)
        signs = np.ascontiguousarray(base.cell_face_sign, dtype=np.int32)
        base_rate = np.empty((3, nc), dtype=np.float64)
        density_rate = np.empty((2, nc), dtype=np.float64)
        full_rate = np.empty((3, nc), dtype=np.float64)
        face_acceleration = np.empty(nf, dtype=np.float64)
        cell_acceleration = np.empty((2, nc), dtype=np.float64)
        face_water_flux = np.empty(nf, dtype=np.float64)
        riemann_calls = ctypes.c_int(-1)
        density_calls = ctypes.c_int(-1)
        status = ctypes.c_int(-1)
        with self._call_lock:
            self._function(
                nc, nf, faces.size, q, bed, owner, neighbour, normal, length,
                distance, area, offset, faces, signs, mixture,
                base.hydraulic_reference_density_kg_m3, base.gravity_m_s2,
                base.dry_tolerance_m,
                int(execution.densin_mode == DENSIN_SEDIMENT_INCREMENT),
                execution.native_threads, base_rate, density_rate, full_rate,
                face_acceleration, cell_acceleration, face_water_flux,
                ctypes.byref(riemann_calls), ctypes.byref(density_calls),
                ctypes.byref(status))
        if status.value != 0:
            raise DensInResidualV95Error(
                f"v95 native full-rate evaluation failed with status {status.value}")
        if (riemann_calls.value != nf or density_calls.value !=
                int(execution.densin_mode == DENSIN_SEDIMENT_INCREMENT)):
            raise DensInResidualV95Error("v95 native owner counts differ")
        arrays = (base_rate, density_rate, full_rate, face_acceleration,
                  cell_acceleration, face_water_flux)
        if not all(np.isfinite(array).all() for array in arrays):
            raise DensInResidualV95Error("v95 native readback is non-finite")
        for array in arrays:
            array.setflags(write=False)
        return NativeRateResultV95(
            base_rate, density_rate, full_rate, face_acceleration,
            cell_acceleration, face_water_flux, riemann_calls.value,
            density_calls.value)


def build_native_densin_runtime_v95(
    repository_root: str | Path, build_directory: str | Path, *,
    compiler: str | Path | None = None,
) -> NativeDensInRuntimeV95:
    """Build and load the isolated OpenMP v95 native micro-integrator."""
    root = Path(repository_root).resolve()
    build_root = Path(build_directory).resolve()
    build_root.mkdir(parents=True, exist_ok=True)
    sources = [root/_PARAMETER_RELATIVE, root/_RPN2_RELATIVE,
               root/_UTILS_RELATIVE, root/_NATIVE_RELATIVE]
    if not all(path.is_file() for path in sources):
        raise DensInResidualV95Error("v95 native source inventory is incomplete")
    hashes = {_PARAMETER_RELATIVE: _file_sha256(sources[0]),
              _RPN2_RELATIVE: _file_sha256(sources[1]),
              _UTILS_RELATIVE: _file_sha256(sources[2]),
              _NATIVE_RELATIVE: _file_sha256(sources[3])}
    locked = tuple(sorted(verify_locked_sources_v85(root).items()))
    for relative, expected in _FROZEN_SOURCE_HASHES.items():
        if hashes[relative] != expected:
            raise DensInResidualV95Error(
                "v95 frozen augmented-fwave source differs: " + relative)
    selected = Path(compiler) if compiler is not None else None
    if selected is None:
        discovered = shutil.which("gfortran")
        if discovered is None and os.name == "nt":
            fallback = Path(r"C:\Strawberry\c\bin\gfortran.exe")
            discovered = str(fallback) if fallback.is_file() else None
        if discovered is None:
            raise DensInResidualV95Error("v95 requires gfortran")
        selected = Path(discovered)
    selected = selected.resolve()
    if not selected.is_file():
        raise DensInResidualV95Error("v95 compiler is unavailable")
    flags = ("-O3", "-flto", "-fopenmp", "-shared",
             "-ffree-line-length-none", "-ffixed-line-length-none",
             "-static-libgfortran", "-static-libgcc")
    build_key = _fingerprint({
        "source_sha256": hashes,
        "compiler_sha256": _file_sha256(selected),
        "flags": list(flags),
        "abi_id": NATIVE_ABI_ID,
    })[:16]
    # A loaded Windows DLL cannot be replaced.  Every build is compiled in a
    # unique hash-labelled directory and atomically published there, so
    # concurrent pytest workers and a later rebuild never target loaded bytes.
    build = Path(tempfile.mkdtemp(prefix=f"v95_{build_key}_", dir=build_root))
    library_name = ("braids_densin_residual_v95.dll" if os.name == "nt"
                    else "libbraids_densin_residual_v95.so")
    partial_library = build/(library_name + ".partial")
    library_path = build/library_name
    command = [str(selected), *flags, *(str(path) for path in sources),
               "-o", str(partial_library)]
    completed = subprocess.run(
        command, cwd=build, capture_output=True, text=True, timeout=180,
        check=False)
    if completed.returncode != 0 or not partial_library.is_file():
        raise DensInResidualV95Error(
            "v95 native build failed:\n" + completed.stdout + completed.stderr)
    os.replace(partial_library, library_path)
    version = subprocess.run(
        [str(selected), "--version"], capture_output=True, text=True,
        timeout=30, check=True).stdout.splitlines()[0]
    placeholder = NativeDensInBuildReceiptV95(
        NATIVE_ABI_ID, hashes[_NATIVE_RELATIVE],
        hashes[_PARAMETER_RELATIVE], hashes[_RPN2_RELATIVE],
        hashes[_UTILS_RELATIVE], _file_sha256(library_path), version, flags,
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
            "+00:00", "Z"), True, LOCKED_UPSTREAM_COMMIT, locked, "")
    receipt = NativeDensInBuildReceiptV95(
        **{**placeholder.__dict__,
           "receipt_sha256": _fingerprint(_build_receipt_payload(placeholder))})
    dependency = None
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        dependency = os.add_dll_directory(str(selected.parent))
    library = ctypes.CDLL(str(library_path))
    function = library.braids_densin_full_rate_v95
    float_array = np.ctypeslib.ndpointer(
        dtype=np.float64, flags="C_CONTIGUOUS")
    int_array = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
    function.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        float_array, float_array, int_array, int_array, float_array,
        float_array, float_array, float_array, int_array, int_array,
        int_array, float_array, ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_int, ctypes.c_int, float_array, float_array,
        float_array, float_array, float_array, float_array,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]
    function.restype = None
    return NativeDensInRuntimeV95(
        receipt, library_path, function, library, dependency)


@dataclass(frozen=True)
class DensInResidualCPUExecutionV95:
    base_execution: DensInExecutionV85
    native_build_receipt: NativeDensInBuildReceiptV95
    densin_mode: str
    bed_elevation_m: np.ndarray
    native_threads: int
    nonlinear_tolerance: float
    max_nonlinear_iterations: int
    maximum_dt_s: float
    locked_delft_commit: str
    locked_delft_source_sha256: tuple[tuple[str, str], ...]
    fingerprint: str

    @property
    def cell_count(self) -> int:
        return self.base_execution.cell_count

    @property
    def face_count(self) -> int:
        return self.base_execution.face_count


def _execution_payload(value: DensInResidualCPUExecutionV95) -> dict:
    base = value.base_execution
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "backend": "native_cpu_openmp",
        "base_v85_execution_fingerprint": base.fingerprint,
        "native_build_receipt_sha256":
            value.native_build_receipt.receipt_sha256,
        "densin_mode": value.densin_mode,
        "accepted_concentration_source": "typed_v78_via_v85_only",
        "water_eos_mode": "uniform_accepted_water_density",
        "spatial_water_density": "rejected_deferred",
        "integrator": "implicit_midpoint_picard",
        "spatial_flux": "unchanged_augmented_geoclaw_rpn2",
        "reconstruction": "piecewise_constant_first_order",
        "boundary_mode": "closed_reflective_hydraulic_zero_normal_density",
        "grid_scope": "fully_wet_flat_cartesian_fixed_grid",
        "viscous_operator": "authenticated_zero",
        "bed_sha256": _digest(value.bed_elevation_m),
        "native_threads": value.native_threads,
        "nonlinear_tolerance": value.nonlinear_tolerance,
        "max_nonlinear_iterations": value.max_nonlinear_iterations,
        "maximum_dt_s": value.maximum_dt_s,
        "locked_delft_commit": value.locked_delft_commit,
        "locked_delft_source_sha256": dict(value.locked_delft_source_sha256),
        "maximum_accepted_updates": MAX_ACCEPTED_UPDATES,
        "density_timing": (
            "accepted_parent_mixture_frozen;current_midpoint_depth_"
            "reevaluated_each_full_rate;no_post_convergence_apply"),
        "production_admission": False,
    }


def resolve_densin_residual_cpu_execution_v95(
    base_execution: DensInExecutionV85,
    native_build_receipt: NativeDensInBuildReceiptV95, *,
    densin_enabled: bool, bed_elevation_m: object, native_threads: int,
    nonlinear_tolerance: float = 2.0e-11,
    max_nonlinear_iterations: int = 40,
    maximum_dt_s: float = 0.02,
) -> DensInResidualCPUExecutionV95:
    if type(base_execution) is not DensInExecutionV85:
        raise DensInResidualV95Error("v95 requires an exact v85 execution")
    execution_document_v85(base_execution)
    locked = tuple(sorted(verify_locked_sources_v85(_ROOT).items()))
    if base_execution.role != "native_cpu":
        raise DensInResidualV95Error("v95 CPU requires native_cpu v85 role")
    _validate_build_receipt(native_build_receipt)
    if type(densin_enabled) is not bool:
        raise DensInResidualV95Error("v95 DensIn selection must be boolean")
    if (np.ptp(base_execution.water_density_kg_m3) != 0.0 or
            not np.all((np.abs(base_execution.normal[0]) <= 2e-13) |
                       (np.abs(base_execution.normal[1]) <= 2e-13))):
        raise DensInResidualV95Error(
            "v95 rejects spatial water density and non-Cartesian faces")
    bed = _array(
        bed_elevation_m, (base_execution.cell_count,), "bed elevation")
    if np.ptp(bed) != 0.0:
        raise DensInResidualV95Error("v95 bounded proof requires a flat bed")
    if type(native_threads) is not int or not 1 <= native_threads <= 256:
        raise DensInResidualV95Error("v95 native thread count is invalid")
    if (type(max_nonlinear_iterations) is not int or
            not 1 <= max_nonlinear_iterations <= 80):
        raise DensInResidualV95Error(
            "v95 nonlinear iteration bound is invalid")
    tolerance = _finite(
        nonlinear_tolerance, "nonlinear tolerance", positive=True)
    if tolerance < 32*np.finfo(np.float64).eps or tolerance > 1e-6:
        raise DensInResidualV95Error(
            "v95 nonlinear tolerance is outside bounded scope")
    maximum_dt = _finite(maximum_dt_s, "maximum timestep", positive=True)
    # Admission-only topology audit.  Per-update face/cell work remains native.
    for cell in range(base_execution.cell_count):
        for slot in range(base_execution.cell_face_offset[cell],
                          base_execution.cell_face_offset[cell+1]):
            face = int(base_execution.cell_faces[slot])
            sign = int(base_execution.cell_face_sign[slot])
            if not ((sign == 1 and base_execution.owner[face] == cell) or
                    (sign == -1 and base_execution.neighbour[face] == cell)):
                raise DensInResidualV95Error(
                    "v95 cell-face ownership is inconsistent")
    mode = DENSIN_SEDIMENT_INCREMENT if densin_enabled else DENSIN_OFF
    placeholder = DensInResidualCPUExecutionV95(
        base_execution, native_build_receipt, mode, bed, native_threads,
        tolerance, max_nonlinear_iterations, maximum_dt,
        LOCKED_UPSTREAM_COMMIT, locked, "")
    return DensInResidualCPUExecutionV95(
        **{**placeholder.__dict__,
           "fingerprint": _fingerprint(_execution_payload(placeholder))})


def execution_document_v95(value: DensInResidualCPUExecutionV95) -> dict:
    if type(value) is not DensInResidualCPUExecutionV95:
        raise DensInResidualV95Error("v95 execution type differs")
    execution_document_v85(value.base_execution)
    _validate_build_receipt(value.native_build_receipt)
    locked = tuple(sorted(verify_locked_sources_v85(_ROOT).items()))
    payload = _execution_payload(value)
    base = value.base_execution
    topology_ok = True
    for cell in range(base.cell_count):
        for slot in range(base.cell_face_offset[cell],
                          base.cell_face_offset[cell+1]):
            face = int(base.cell_faces[slot])
            sign = int(base.cell_face_sign[slot])
            topology_ok = topology_ok and (
                (sign == 1 and base.owner[face] == cell) or
                (sign == -1 and base.neighbour[face] == cell))
    if (value.densin_mode not in _MODES or
            np.ptp(base.water_density_kg_m3) != 0.0 or
            value.bed_elevation_m.shape != (base.cell_count,) or
            not np.isfinite(value.bed_elevation_m).all() or
            np.ptp(value.bed_elevation_m) != 0.0 or
            not np.all((np.abs(base.normal[0]) <= 2e-13) |
                       (np.abs(base.normal[1]) <= 2e-13)) or
            type(value.native_threads) is not int or
            not 1 <= value.native_threads <= 256 or
            type(value.max_nonlinear_iterations) is not int or
            not 1 <= value.max_nonlinear_iterations <= 80 or
            not math.isfinite(value.nonlinear_tolerance) or
            not (32*np.finfo(np.float64).eps <=
                 value.nonlinear_tolerance <= 1e-6) or
            not math.isfinite(value.maximum_dt_s) or
            value.maximum_dt_s <= 0.0 or not topology_ok or
            value.locked_delft_commit != LOCKED_UPSTREAM_COMMIT or
            value.locked_delft_source_sha256 != locked or
            value.fingerprint != _fingerprint(payload)):
        raise DensInResidualV95Error("v95 execution fingerprint differs")
    return {**payload, "fingerprint": value.fingerprint}


@dataclass(frozen=True)
class DensInResidualCPUStateV95:
    execution_fingerprint: str
    q: np.ndarray
    accepted_updates: int
    time_s: float
    accepted_chain_sha256: str
    state_sha256: str


def _state_token(value: DensInResidualCPUStateV95) -> str:
    return _fingerprint({
        "execution_fingerprint": value.execution_fingerprint,
        "q_sha256": _digest(value.q),
        "accepted_updates": value.accepted_updates,
        "time_s": value.time_s,
        "accepted_chain_sha256": value.accepted_chain_sha256,
    })


def _wrap_state(execution: DensInResidualCPUExecutionV95, *, q: object,
                accepted_updates: int, time_s: float,
                accepted_chain_sha256: str) -> DensInResidualCPUStateV95:
    execution_document_v95(execution)
    if (type(accepted_updates) is not int or
            not 0 <= accepted_updates <= MAX_ACCEPTED_UPDATES):
        raise DensInResidualV95Error("v95 state update count is outside scope")
    time = _finite(time_s, "state time")
    if time < 0.0:
        raise DensInResidualV95Error("v95 state time is negative")
    state_q = _array(q, (3, execution.cell_count), "hydraulic state")
    if np.any(state_q[0] <= execution.base_execution.dry_tolerance_m):
        raise DensInResidualV95Error("v95 state left the fully wet scope")
    placeholder = DensInResidualCPUStateV95(
        execution.fingerprint, state_q, accepted_updates, time,
        _sha(accepted_chain_sha256, "accepted chain"), "")
    return DensInResidualCPUStateV95(
        **{**placeholder.__dict__, "state_sha256": _state_token(placeholder)})


def resolve_densin_residual_cpu_state_v95(
    execution: DensInResidualCPUExecutionV95, *, q: object,
) -> DensInResidualCPUStateV95:
    execution_document_v95(execution)
    chain = _fingerprint({
        "scope_id": SCOPE_ID,
        "execution_fingerprint": execution.fingerprint,
        "initial_q_sha256": _digest(np.asarray(q, dtype=np.float64)),
    })
    return _wrap_state(
        execution, q=q, accepted_updates=0, time_s=0.0,
        accepted_chain_sha256=chain)


def validate_densin_residual_cpu_state_v95(
    state: DensInResidualCPUStateV95,
    execution: DensInResidualCPUExecutionV95,
) -> None:
    if (type(state) is not DensInResidualCPUStateV95 or
            state.execution_fingerprint != execution.fingerprint or
            state.state_sha256 != _state_token(state)):
        raise DensInResidualV95Error("v95 state is stale or mismatched")
    _wrap_state(
        execution, q=state.q, accepted_updates=state.accepted_updates,
        time_s=state.time_s,
        accepted_chain_sha256=state.accepted_chain_sha256)


@dataclass(frozen=True)
class DensInResidualCPUTrialV95:
    execution_fingerprint: str
    parent_state_sha256: str
    accepted_concentration: AcceptedConcentrationReceiptV85
    dt_s: float
    candidate: DensInResidualCPUStateV95
    midpoint_q: np.ndarray
    mixture_density_kg_m3: np.ndarray
    base_fwave_rate: np.ndarray
    density_momentum_rate: np.ndarray
    full_rate: np.ndarray
    final_residual: np.ndarray
    face_normal_acceleration_m_s2: np.ndarray
    cell_acceleration_m_s2: np.ndarray
    face_water_flux_m2_s: np.ndarray
    midpoint_depth_sha256_by_evaluation: tuple[str, ...]
    residual_evaluation_count: int
    density_source_evaluation_count: int
    augmented_riemann_call_count: int
    approximate_jacobian_evaluation_count: int
    density_jacobian_derivative_evaluation_count: int
    post_convergence_density_application_count: int
    accepted_parent_density_mutation_count: int
    nonlinear_iterations: int
    nonlinear_residual_linf: float
    water_volume_residual_m3: float
    frozen_suspended_inventory_residual_kg: float
    backend: str
    trial_sha256: str


def _trial_token(value: DensInResidualCPUTrialV95) -> str:
    return _fingerprint({
        "execution_fingerprint": value.execution_fingerprint,
        "parent_state_sha256": value.parent_state_sha256,
        "accepted_concentration_receipt_sha256":
            value.accepted_concentration.receipt_sha256,
        "dt_s": value.dt_s,
        "candidate_state_sha256": value.candidate.state_sha256,
        "arrays_sha256": _digest(
            value.midpoint_q, value.mixture_density_kg_m3,
            value.base_fwave_rate, value.density_momentum_rate,
            value.full_rate, value.final_residual,
            value.face_normal_acceleration_m_s2,
            value.cell_acceleration_m_s2, value.face_water_flux_m2_s),
        "midpoint_depth_sha256_by_evaluation":
            list(value.midpoint_depth_sha256_by_evaluation),
        "residual_evaluation_count": value.residual_evaluation_count,
        "density_source_evaluation_count":
            value.density_source_evaluation_count,
        "augmented_riemann_call_count":
            value.augmented_riemann_call_count,
        "approximate_jacobian_evaluation_count":
            value.approximate_jacobian_evaluation_count,
        "density_jacobian_derivative_evaluation_count":
            value.density_jacobian_derivative_evaluation_count,
        "post_convergence_density_application_count":
            value.post_convergence_density_application_count,
        "accepted_parent_density_mutation_count":
            value.accepted_parent_density_mutation_count,
        "nonlinear_iterations": value.nonlinear_iterations,
        "nonlinear_residual_linf": value.nonlinear_residual_linf,
        "water_volume_residual_m3": value.water_volume_residual_m3,
        "frozen_suspended_inventory_residual_kg":
            value.frozen_suspended_inventory_residual_kg,
        "backend": value.backend,
    })


def _residual_norm(residual: np.ndarray, q: np.ndarray,
                   gravity: float) -> float:
    depth_scale = max(1.0, float(np.max(np.abs(q[0]))))
    momentum_scale = max(
        1.0, float(np.max(np.abs(q[1:]))),
        depth_scale*math.sqrt(gravity*depth_scale))
    scale = np.asarray([depth_scale, momentum_scale, momentum_scale])[:, None]
    return float(np.max(np.abs(residual)/scale))


def begin_densin_residual_cpu_trial_v95(
    execution: DensInResidualCPUExecutionV95,
    parent: DensInResidualCPUStateV95, *,
    accepted_concentration: AcceptedConcentrationReceiptV85,
    native_runtime: NativeDensInRuntimeV95, dt_s: float,
    cancelled: bool | Callable[[], bool] = False,
) -> DensInResidualCPUTrialV95:
    """Solve one bounded native augmented-f-wave implicit-midpoint trial."""
    execution_document_v95(execution)
    validate_densin_residual_cpu_state_v95(parent, execution)
    if type(native_runtime) is not NativeDensInRuntimeV95:
        raise DensInResidualV95Error("v95 native runtime type differs")
    if native_runtime.receipt != execution.native_build_receipt:
        raise DensInResidualV95Error("v95 native runtime identity differs")
    if _cancelled(cancelled):
        raise DensInResidualV95Error("v95 trial cancelled before native work")
    if parent.accepted_updates >= MAX_ACCEPTED_UPDATES:
        raise DensInResidualV95Error("v95 accepted update limit was reached")
    dt = _finite(dt_s, "timestep", positive=True)
    if dt > execution.maximum_dt_s:
        raise DensInResidualV95Error("v95 timestep exceeds the bounded limit")
    try:
        validate_accepted_concentration_receipt_v85(
            accepted_concentration, execution.base_execution,
            accepted_update=parent.accepted_updates, time_s=parent.time_s,
            require_typed_v78=True)
    except (TypeError, ValueError) as error:
        raise DensInResidualV95Error(
            "v95 requires the current typed v85 concentration handoff") from error
    if not np.array_equal(parent.q[0], accepted_concentration.depth_m):
        raise DensInResidualV95Error(
            "v95 accepted concentration depth differs from parent hydraulics")
    mixture = np.ascontiguousarray(
        accepted_concentration.mixture_density_kg_m3, dtype=np.float64)
    frozen_mixture_sha256 = _digest(mixture)
    midpoint = np.array(parent.q, copy=True, order="C")
    depth_digests: list[str] = []
    riemann_calls = 0
    density_calls = 0
    result: NativeRateResultV95 | None = None
    final_residual: np.ndarray | None = None
    nonlinear_residual = math.inf
    nonlinear_iterations = 0
    for evaluation in range(1, execution.max_nonlinear_iterations+2):
        if _cancelled(cancelled):
            raise DensInResidualV95Error(
                "v95 trial cancelled during midpoint iteration")
        result = native_runtime.evaluate(execution, midpoint, mixture)
        if _digest(mixture) != frozen_mixture_sha256:
            raise DensInResidualV95Error(
                "v95 native evaluation mutated accepted parent density")
        depth_digests.append(_digest(midpoint[0]))
        riemann_calls += result.augmented_riemann_call_count
        density_calls += result.density_source_evaluation_count
        final_residual = np.ascontiguousarray(
            midpoint-parent.q-0.5*dt*result.full_rate)
        nonlinear_residual = _residual_norm(
            final_residual, midpoint, execution.base_execution.gravity_m_s2)
        if nonlinear_residual <= execution.nonlinear_tolerance:
            break
        if evaluation > execution.max_nonlinear_iterations:
            raise DensInResidualV95Error(
                "v95 implicit midpoint failed to converge")
        midpoint = np.ascontiguousarray(
            midpoint-final_residual, dtype=np.float64)
        if (not np.isfinite(midpoint).all() or
                np.any(midpoint[0] <=
                       execution.base_execution.dry_tolerance_m)):
            raise DensInResidualV95Error(
                "v95 midpoint iteration left the fully wet scope")
        nonlinear_iterations += 1
    if result is None or final_residual is None:
        raise DensInResidualV95Error("v95 produced no native residual")
    if _cancelled(cancelled):
        raise DensInResidualV95Error(
            "v95 trial cancelled before publication")
    midpoint = _array(midpoint, (3, execution.cell_count), "midpoint")
    final_residual = _array(
        final_residual, (3, execution.cell_count), "final residual")
    candidate_q = np.ascontiguousarray(parent.q+dt*result.full_rate)
    if np.any(candidate_q[0] <= execution.base_execution.dry_tolerance_m):
        raise DensInResidualV95Error("v95 candidate left the fully wet scope")
    boundary = execution.base_execution.neighbour < 0
    boundary_outflow = float(np.sum(
        execution.base_execution.face_length_m[boundary]*
        result.face_water_flux_m2_s[boundary]))
    water_residual = float(np.sum(
        execution.base_execution.cell_area_m2*(candidate_q[0]-parent.q[0])) +
        dt*boundary_outflow)
    inventory = float(np.sum(
        accepted_concentration.suspended_mass_kg_m2*
        execution.base_execution.cell_area_m2[None, :]))
    inventory_residual = inventory-inventory
    expected_density_calls = (
        len(depth_digests) if execution.densin_mode ==
        DENSIN_SEDIMENT_INCREMENT else 0)
    if (riemann_calls != execution.face_count*len(depth_digests) or
            density_calls != expected_density_calls):
        raise DensInResidualV95Error("v95 accumulated native counts differ")
    chain = _fingerprint({
        "previous": parent.accepted_chain_sha256,
        "accepted_concentration_receipt":
            accepted_concentration.receipt_sha256,
        "native_build_receipt":
            execution.native_build_receipt.receipt_sha256,
        "dt_s": dt,
        "midpoint_sha256": _digest(midpoint),
        "full_rate_sha256": _digest(result.full_rate),
        "residual_evaluation_count": len(depth_digests),
        "density_source_evaluation_count": density_calls,
        "post_convergence_density_application_count": 0,
    })
    candidate = _wrap_state(
        execution, q=candidate_q,
        accepted_updates=parent.accepted_updates+1,
        time_s=parent.time_s+dt, accepted_chain_sha256=chain)
    placeholder = DensInResidualCPUTrialV95(
        execution.fingerprint, parent.state_sha256, accepted_concentration,
        dt, candidate, midpoint, mixture, result.base_fwave_rate,
        result.density_momentum_rate, result.full_rate, final_residual,
        result.face_normal_acceleration_m_s2,
        result.cell_acceleration_m_s2, result.face_water_flux_m2_s,
        tuple(depth_digests), len(depth_digests), density_calls,
        riemann_calls, 0, 0, 0, 0, nonlinear_iterations, nonlinear_residual,
        water_residual, inventory_residual, "native_cpu_openmp", "")
    return DensInResidualCPUTrialV95(
        **{**placeholder.__dict__, "trial_sha256": _trial_token(placeholder)})


def _validate_trial(
    execution: DensInResidualCPUExecutionV95,
    parent: DensInResidualCPUStateV95,
    trial: DensInResidualCPUTrialV95,
) -> None:
    validate_densin_residual_cpu_state_v95(parent, execution)
    if (type(trial) is not DensInResidualCPUTrialV95 or
            trial.execution_fingerprint != execution.fingerprint or
            trial.parent_state_sha256 != parent.state_sha256 or
            trial.backend != "native_cpu_openmp" or
            trial.trial_sha256 != _trial_token(trial)):
        raise DensInResidualV95Error("v95 trial is stale or mismatched")
    try:
        validate_accepted_concentration_receipt_v85(
            trial.accepted_concentration, execution.base_execution,
            accepted_update=parent.accepted_updates, time_s=parent.time_s,
            require_typed_v78=True)
    except (TypeError, ValueError) as error:
        raise DensInResidualV95Error(
            "v95 trial concentration receipt differs") from error
    validate_densin_residual_cpu_state_v95(trial.candidate, execution)
    nc, nf = execution.cell_count, execution.face_count
    midpoint = _array(trial.midpoint_q, (3, nc), "trial midpoint")
    mixture = _array(
        trial.mixture_density_kg_m3, (nc,), "trial mixture", positive=True)
    base_rate = _array(
        trial.base_fwave_rate, (3, nc), "trial base rate")
    density_rate = _array(
        trial.density_momentum_rate, (2, nc), "trial density rate")
    full_rate = _array(trial.full_rate, (3, nc), "trial full rate")
    stored_residual = _array(
        trial.final_residual, (3, nc), "trial final residual")
    _array(trial.face_normal_acceleration_m_s2, (nf,),
           "trial face acceleration")
    _array(trial.cell_acceleration_m_s2, (2, nc),
           "trial cell acceleration")
    face_water_flux = _array(
        trial.face_water_flux_m2_s, (nf,), "trial face water flux")
    dt = _finite(trial.dt_s, "trial timestep", positive=True)
    expected_calls = (trial.residual_evaluation_count if
                      execution.densin_mode == DENSIN_SEDIMENT_INCREMENT else 0)
    if (trial.candidate.accepted_updates != parent.accepted_updates+1 or
            trial.candidate.time_s != parent.time_s+dt or
            dt > execution.maximum_dt_s or
            not np.array_equal(
                trial.accepted_concentration.depth_m, parent.q[0]) or
            trial.residual_evaluation_count < 1 or
            trial.residual_evaluation_count >
                execution.max_nonlinear_iterations+1 or
            trial.nonlinear_iterations != trial.residual_evaluation_count-1 or
            trial.nonlinear_iterations > execution.max_nonlinear_iterations or
            len(trial.midpoint_depth_sha256_by_evaluation) !=
                trial.residual_evaluation_count or
            any(_SHA256.fullmatch(value) is None for value in
                trial.midpoint_depth_sha256_by_evaluation) or
            trial.midpoint_depth_sha256_by_evaluation[-1] !=
                _digest(midpoint[0]) or
            trial.density_source_evaluation_count != expected_calls or
            trial.augmented_riemann_call_count !=
                execution.face_count*trial.residual_evaluation_count or
            trial.approximate_jacobian_evaluation_count != 0 or
            trial.density_jacobian_derivative_evaluation_count != 0 or
            trial.post_convergence_density_application_count != 0 or
            trial.accepted_parent_density_mutation_count != 0 or
            trial.frozen_suspended_inventory_residual_kg != 0.0):
        raise DensInResidualV95Error("v95 trial counts or ledgers differ")
    if not np.array_equal(
            trial.mixture_density_kg_m3,
            trial.accepted_concentration.mixture_density_kg_m3):
        raise DensInResidualV95Error("v95 trial mixture density differs")
    expected_full = np.array(base_rate, copy=True)
    expected_full[1:] += density_rate
    residual = midpoint-parent.q-0.5*dt*full_rate
    candidate_q = parent.q+dt*full_rate
    recomputed_norm = _residual_norm(
        residual, midpoint, execution.base_execution.gravity_m_s2)
    boundary = execution.base_execution.neighbour < 0
    boundary_outflow = float(np.sum(
        execution.base_execution.face_length_m[boundary]*
        face_water_flux[boundary]))
    recomputed_water_residual = float(np.sum(
        execution.base_execution.cell_area_m2*
        (candidate_q[0]-parent.q[0]))+dt*boundary_outflow)
    if (not np.array_equal(expected_full, full_rate) or
            not np.array_equal(residual, stored_residual) or
            not np.array_equal(candidate_q, trial.candidate.q)):
        raise DensInResidualV95Error(
            "v95 residual or endpoint composition differs")
    if (not math.isfinite(trial.nonlinear_residual_linf) or
            trial.nonlinear_residual_linf != recomputed_norm or
            recomputed_norm > execution.nonlinear_tolerance or
            not math.isfinite(trial.water_volume_residual_m3) or
            trial.water_volume_residual_m3 != recomputed_water_residual):
        raise DensInResidualV95Error(
            "v95 recomputed residual or water budget differs")
    if execution.densin_mode == DENSIN_SEDIMENT_INCREMENT:
        face = delft_kmax1_face_acceleration_v85(
            midpoint[0], mixture,
            execution.base_execution.owner, execution.base_execution.neighbour,
            execution.base_execution.face_distance_m,
            gravity_m_s2=execution.base_execution.gravity_m_s2,
            hydraulic_reference_density_kg_m3=
                execution.base_execution.hydraulic_reference_density_kg_m3)
        cell = project_face_acceleration_v85(
            face, execution.base_execution.normal,
            execution.base_execution.face_length_m,
            execution.base_execution.cell_face_offset,
            execution.base_execution.cell_faces,
            execution.base_execution.cell_face_sign)
        density = midpoint[0][None, :]*cell
        scale = max(1.0, float(np.max(np.abs(density))))
        if (not np.allclose(face, trial.face_normal_acceleration_m_s2,
                            rtol=8*np.finfo(float).eps, atol=1e-18) or
                not np.allclose(cell, trial.cell_acceleration_m_s2,
                                rtol=16*np.finfo(float).eps, atol=1e-18) or
                not np.allclose(density, trial.density_momentum_rate,
                                rtol=16*np.finfo(float).eps,
                                atol=16*np.finfo(float).eps*scale)):
            raise DensInResidualV95Error(
                "v95 native density source differs from the v85 oracle")
    elif (np.any(trial.face_normal_acceleration_m_s2 != 0.0) or
          np.any(trial.cell_acceleration_m_s2 != 0.0) or
          np.any(trial.density_momentum_rate != 0.0)):
        raise DensInResidualV95Error("v95 DensIn-off source is not exact zero")
    water_scale = max(
        1.0, float(np.sum(execution.base_execution.cell_area_m2*parent.q[0])))
    if abs(trial.water_volume_residual_m3) > \
            128*np.finfo(float).eps*water_scale:
        raise DensInResidualV95Error("v95 closed water budget differs")


def commit_densin_residual_cpu_trial_v95(
    execution: DensInResidualCPUExecutionV95,
    parent: DensInResidualCPUStateV95,
    trial: DensInResidualCPUTrialV95,
) -> DensInResidualCPUStateV95:
    _validate_trial(execution, parent, trial)
    return trial.candidate


def rollback_densin_residual_cpu_trial_v95(
    execution: DensInResidualCPUExecutionV95,
    parent: DensInResidualCPUStateV95,
    trial: DensInResidualCPUTrialV95,
) -> DensInResidualCPUStateV95:
    _validate_trial(execution, parent, trial)
    return parent


def cancel_densin_residual_cpu_trial_v95(
    execution: DensInResidualCPUExecutionV95,
    parent: DensInResidualCPUStateV95,
    trial: DensInResidualCPUTrialV95,
) -> DensInResidualCPUStateV95:
    return rollback_densin_residual_cpu_trial_v95(execution, parent, trial)


def densin_residual_cpu_checkpoint_v95(
    execution: DensInResidualCPUExecutionV95,
    state: DensInResidualCPUStateV95,
) -> dict:
    validate_densin_residual_cpu_state_v95(state, execution)
    body = {
        "schema": SCHEMA,
        "scope_id": CHECKPOINT_SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "execution_fingerprint": execution.fingerprint,
        "q": state.q.tolist(),
        "accepted_updates": state.accepted_updates,
        "time_s": state.time_s,
        "accepted_chain_sha256": state.accepted_chain_sha256,
        "state_sha256": state.state_sha256,
    }
    return {**body, "fingerprint": _fingerprint(body)}


def load_densin_residual_cpu_checkpoint_v95(
    execution: DensInResidualCPUExecutionV95,
    document: Mapping[str, object],
) -> DensInResidualCPUStateV95:
    expected = {
        "schema", "scope_id", "semantic_id", "execution_fingerprint", "q",
        "accepted_updates", "time_s", "accepted_chain_sha256",
        "state_sha256", "fingerprint",
    }
    if not isinstance(document, Mapping) or set(document) != expected:
        raise DensInResidualV95Error("v95 checkpoint inventory differs")
    body = {key: document[key] for key in document if key != "fingerprint"}
    if (document["fingerprint"] != _fingerprint(body) or
            document["schema"] != SCHEMA or
            document["scope_id"] != CHECKPOINT_SCOPE_ID or
            document["semantic_id"] != SEMANTIC_ID or
            document["execution_fingerprint"] != execution.fingerprint):
        raise DensInResidualV95Error("v95 checkpoint identity differs")
    state = _wrap_state(
        execution, q=document["q"],
        accepted_updates=document["accepted_updates"],
        time_s=document["time_s"],
        accepted_chain_sha256=document["accepted_chain_sha256"])
    if state.state_sha256 != document["state_sha256"]:
        raise DensInResidualV95Error("v95 checkpoint state digest differs")
    return state


def method_identity_v95() -> dict:
    return {
        "scope_id": SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "native_operator": "openmp_augmented_fwave_full_rate_v1",
        "density_equation": "v85_locked_delft3d_kmax1_sediment_increment",
        "density_in_nonlinear_residual": True,
        "nonlinear_solver": "picard_fixed_point_no_jacobian",
        "approximate_jacobian_evaluation_count": 0,
        "density_jacobian_derivative_evaluation_count": 0,
        "post_convergence_density_application_count": 0,
        "accepted_parent_density_frozen": True,
        "midpoint_depth_recomputed": True,
        "reconstruction": "piecewise_constant_first_order",
        "boundaries": "closed_reflective",
        "bed": "flat_fixed",
        "viscous_operator": "authenticated_zero",
        "water_eos_spatial_gradient": "rejected_deferred",
        "openmp_face_cell_loops": True,
        "production_admission": False,
        "not_claimed": [
            "production_cutcell_or_WENO_controller", "AMR", "wet_dry",
            "open_boundaries", "nonflat_bed", "spatial_water_EOS",
            "nonzero_vicuv", "Delft_staggered_grid_identity",
            "plugin_support", "long_trajectory_qualification",
        ],
    }


__all__ = [
    "DENSIN_OFF", "DENSIN_SEDIMENT_INCREMENT",
    "DensInResidualCPUExecutionV95", "DensInResidualCPUStateV95",
    "DensInResidualCPUTrialV95", "DensInResidualV95Error",
    "MAX_ACCEPTED_UPDATES", "NativeDensInBuildReceiptV95",
    "NativeDensInRuntimeV95", "NativeRateResultV95",
    "begin_densin_residual_cpu_trial_v95", "build_native_densin_runtime_v95",
    "cancel_densin_residual_cpu_trial_v95",
    "commit_densin_residual_cpu_trial_v95",
    "densin_residual_cpu_checkpoint_v95", "execution_document_v95",
    "load_densin_residual_cpu_checkpoint_v95", "method_identity_v95",
    "resolve_densin_residual_cpu_execution_v95",
    "resolve_densin_residual_cpu_state_v95",
    "rollback_densin_residual_cpu_trial_v95",
    "validate_densin_residual_cpu_state_v95",
]
