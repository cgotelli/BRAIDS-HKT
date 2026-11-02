"""Native CPU augmented-f-wave evaluator with runtime bed/EOS geometry.

This is the additive Q4/v96 native ABI.  It leaves the frozen v95 execution
and GeoClaw Riemann sources untouched, but admits the *accepted parent* bed
elevation and spatial water/mixture density on every residual evaluation.
The outer atomic owner decides whether the density field is water EOS alone or
water plus the authenticated suspended-sediment increment.
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

import numpy as np

from .densin_fwave_v85 import (
    DensInExecutionV85,
    LOCKED_UPSTREAM_COMMIT,
    execution_document_v85,
    verify_locked_sources_v85,
)


SCHEMA = 1
SCOPE_ID = "braids-q4-coupled-geometry-native-cpu-v96-v1"
SEMANTIC_ID = SCOPE_ID + ":runtime-bed-spatial-eos"
NATIVE_ABI_ID = SCOPE_ID + ":openmp-full-rate-abi-v1"
DENSITY_WATER_EOS = "water_eos"
DENSITY_WATER_PLUS_SEDIMENT = "water_plus_sediment"
_DENSITY_MODES = frozenset((DENSITY_WATER_EOS,
                            DENSITY_WATER_PLUS_SEDIMENT))
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ROOT = Path(__file__).resolve().parents[2]

_WRAPPER_RELATIVE = "solver/wave/braids_coupled_geometry_v96_module.f90"
_V95_RELATIVE = "solver/wave/braids_densin_residual_v95_module.f90"
_PARAMETER_RELATIVE = "solver/cut_cells/geoclaw_parameters.f90"
_RPN2_RELATIVE = "solver/clawpack-v5.14.0/riemann/src/rpn2_geoclaw.f"
_UTILS_RELATIVE = "solver/clawpack-v5.14.0/riemann/src/geoclaw_riemann_utils.f"
_EXPECTED_SOURCE_HASHES = {
    _WRAPPER_RELATIVE:
        "dfb8c468d40559ef64c2af091fa2c3e74c2e9fabbb76848510b00a6dd039eda6",
    _V95_RELATIVE:
        "e701c096e400402c4386cc2015459d5900ce073094b86f97c0ad94c2516a61e3",
    _PARAMETER_RELATIVE:
        "b6e5a70595e93279eac113401dfc2d857978f7a6e885fe730c783c17cc5b3c1d",
    _RPN2_RELATIVE:
        "c52649def884ce374dbb1d3dc1e91f7a9c33aa45a0aff691badc8811b9157d1f",
    _UTILS_RELATIVE:
        "df73207446ac730f9cda9a8ca4dbbc625151ba95d13aa514ce87ca23cb598761",
}


class CoupledGeometryNativeV96Error(ValueError):
    """The v96 native build, execution, or evaluation is invalid."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError, OverflowError) as error:
        raise CoupledGeometryNativeV96Error(
            "v96 native value is not finite canonical JSON") from error


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(_canonical(list(array.shape)))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _array(value: object, shape: tuple[int, ...], name: str, *,
           positive: bool = False) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != shape or raw.dtype.kind not in "fiu":
        raise CoupledGeometryNativeV96Error(
            f"v96 native {name} must be numeric with shape {shape}")
    result = np.ascontiguousarray(raw, dtype=np.float64)
    if not np.isfinite(result).all():
        raise CoupledGeometryNativeV96Error(
            f"v96 native {name} must be finite")
    if positive and np.any(result <= 0.0):
        raise CoupledGeometryNativeV96Error(
            f"v96 native {name} must be positive")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class NativeCoupledGeometryBuildReceiptV96:
    abi_id: str
    wrapper_source_sha256: str
    v95_kernel_source_sha256: str
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


def _build_payload(value: NativeCoupledGeometryBuildReceiptV96) -> dict:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID + ":native-build",
        "abi_id": value.abi_id,
        "wrapper_source_sha256": value.wrapper_source_sha256,
        "v95_kernel_source_sha256": value.v95_kernel_source_sha256,
        "geoclaw_parameters_sha256": value.geoclaw_parameters_sha256,
        "augmented_rpn2_sha256": value.augmented_rpn2_sha256,
        "riemann_utils_sha256": value.riemann_utils_sha256,
        "library_sha256": value.library_sha256,
        "compiler_identity": value.compiler_identity,
        "compile_flags": list(value.compile_flags),
        "openmp_enabled": value.openmp_enabled,
        "locked_delft_commit": value.locked_delft_commit,
        "locked_delft_source_sha256": dict(
            value.locked_delft_source_sha256),
        "frozen_v95_and_augmented_riemann_sources_unchanged": True,
    }


def validate_native_coupled_geometry_build_receipt_v96(
    value: NativeCoupledGeometryBuildReceiptV96,
) -> None:
    if type(value) is not NativeCoupledGeometryBuildReceiptV96:
        raise CoupledGeometryNativeV96Error(
            "v96 native build receipt type differs")
    locked = tuple(sorted(verify_locked_sources_v85(_ROOT).items()))
    hashes = (
        value.wrapper_source_sha256, value.v95_kernel_source_sha256,
        value.geoclaw_parameters_sha256, value.augmented_rpn2_sha256,
        value.riemann_utils_sha256, value.library_sha256,
    )
    if any(_SHA256.fullmatch(item) is None for item in hashes):
        raise CoupledGeometryNativeV96Error(
            "v96 native receipt contains a malformed source hash")
    expected = tuple(_EXPECTED_SOURCE_HASHES[key] for key in (
        _WRAPPER_RELATIVE, _V95_RELATIVE, _PARAMETER_RELATIVE,
        _RPN2_RELATIVE, _UTILS_RELATIVE))
    if (value.abi_id != NATIVE_ABI_ID or
            tuple(hashes[:5]) != expected or
            value.openmp_enabled is not True or
            not value.compiler_identity or not value.compile_flags or
            not value.build_utc.endswith("Z") or
            value.locked_delft_commit != LOCKED_UPSTREAM_COMMIT or
            value.locked_delft_source_sha256 != locked or
            value.receipt_sha256 != _fingerprint(_build_payload(value))):
        raise CoupledGeometryNativeV96Error(
            "v96 native build receipt is stale or mismatched")


@dataclass(frozen=True)
class CoupledGeometryNativeCPUExecutionV96:
    base_execution: DensInExecutionV85
    native_build_receipt: NativeCoupledGeometryBuildReceiptV96
    density_mode: str
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


def _execution_payload(value: CoupledGeometryNativeCPUExecutionV96) -> dict:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "backend": "native_cpu_openmp",
        "base_v85_execution_fingerprint": value.base_execution.fingerprint,
        "native_build_receipt_sha256":
            value.native_build_receipt.receipt_sha256,
        "density_mode": value.density_mode,
        "water_eos": "accepted_spatial_v92_field_always_retained",
        "sediment_increment": (
            "accepted_v94b_suspended_concentration" if value.density_mode ==
            DENSITY_WATER_PLUS_SEDIMENT else "disabled"),
        "bed_geometry": "accepted_parent_runtime_cell_field",
        "integrator": "outer_implicit_midpoint_picard",
        "spatial_flux": "unchanged_augmented_geoclaw_rpn2",
        "reconstruction": "piecewise_constant_first_order",
        "boundary_mode": "closed_reflective",
        "wet_dry_scope": "fully_wet",
        "native_threads": value.native_threads,
        "nonlinear_tolerance": value.nonlinear_tolerance,
        "max_nonlinear_iterations": value.max_nonlinear_iterations,
        "maximum_dt_s": value.maximum_dt_s,
        "locked_delft_commit": value.locked_delft_commit,
        "locked_delft_source_sha256": dict(
            value.locked_delft_source_sha256),
        "production_admission": False,
    }


def resolve_coupled_geometry_native_cpu_execution_v96(
    base_execution: DensInExecutionV85,
    native_build_receipt: NativeCoupledGeometryBuildReceiptV96, *,
    include_sediment_density_increment: bool,
    native_threads: int,
    nonlinear_tolerance: float = 2.0e-11,
    max_nonlinear_iterations: int = 40,
    maximum_dt_s: float = 0.02,
) -> CoupledGeometryNativeCPUExecutionV96:
    if type(base_execution) is not DensInExecutionV85:
        raise CoupledGeometryNativeV96Error(
            "v96 native execution requires exact v85 topology/constants")
    execution_document_v85(base_execution)
    validate_native_coupled_geometry_build_receipt_v96(native_build_receipt)
    locked = tuple(sorted(verify_locked_sources_v85(_ROOT).items()))
    if base_execution.role != "native_cpu":
        raise CoupledGeometryNativeV96Error(
            "v96 native CPU execution requires native_cpu role")
    if type(include_sediment_density_increment) is not bool:
        raise CoupledGeometryNativeV96Error(
            "v96 sediment-density selection must be boolean")
    if type(native_threads) is not int or not 1 <= native_threads <= 256:
        raise CoupledGeometryNativeV96Error(
            "v96 native thread count is invalid")
    if (type(max_nonlinear_iterations) is not int or
            not 1 <= max_nonlinear_iterations <= 80):
        raise CoupledGeometryNativeV96Error(
            "v96 native nonlinear iteration bound is invalid")
    tolerance = float(nonlinear_tolerance)
    maximum_dt = float(maximum_dt_s)
    if (not math.isfinite(tolerance) or
            not 32*np.finfo(float).eps <= tolerance <= 1e-6 or
            not math.isfinite(maximum_dt) or maximum_dt <= 0.0):
        raise CoupledGeometryNativeV96Error(
            "v96 native nonlinear controls are invalid")
    mode = (DENSITY_WATER_PLUS_SEDIMENT if
            include_sediment_density_increment else DENSITY_WATER_EOS)
    placeholder = CoupledGeometryNativeCPUExecutionV96(
        base_execution, native_build_receipt, mode, native_threads,
        tolerance, max_nonlinear_iterations, maximum_dt,
        LOCKED_UPSTREAM_COMMIT, locked, "")
    return CoupledGeometryNativeCPUExecutionV96(
        **{**placeholder.__dict__,
           "fingerprint": _fingerprint(_execution_payload(placeholder))})


def execution_document_coupled_geometry_native_v96(
    value: CoupledGeometryNativeCPUExecutionV96,
) -> dict:
    if type(value) is not CoupledGeometryNativeCPUExecutionV96:
        raise CoupledGeometryNativeV96Error(
            "v96 native execution object differs")
    execution_document_v85(value.base_execution)
    validate_native_coupled_geometry_build_receipt_v96(
        value.native_build_receipt)
    locked = tuple(sorted(verify_locked_sources_v85(_ROOT).items()))
    payload = _execution_payload(value)
    if (value.density_mode not in _DENSITY_MODES or
            type(value.native_threads) is not int or
            not 1 <= value.native_threads <= 256 or
            type(value.max_nonlinear_iterations) is not int or
            not 1 <= value.max_nonlinear_iterations <= 80 or
            not math.isfinite(value.nonlinear_tolerance) or
            not 32*np.finfo(float).eps <= value.nonlinear_tolerance <= 1e-6 or
            not math.isfinite(value.maximum_dt_s) or
            value.maximum_dt_s <= 0.0 or
            value.locked_delft_commit != LOCKED_UPSTREAM_COMMIT or
            value.locked_delft_source_sha256 != locked or
            value.fingerprint != _fingerprint(payload)):
        raise CoupledGeometryNativeV96Error(
            "v96 native execution fingerprint differs")
    return {**payload, "fingerprint": value.fingerprint}


@dataclass(frozen=True)
class NativeCoupledGeometryRateResultV96:
    base_fwave_rate: np.ndarray
    density_momentum_rate: np.ndarray
    full_rate: np.ndarray
    face_normal_acceleration_m_s2: np.ndarray
    cell_acceleration_m_s2: np.ndarray
    face_water_flux_m2_s: np.ndarray
    accepted_bed_sha256: str
    accepted_density_sha256: str
    augmented_riemann_call_count: int
    density_source_evaluation_count: int


class NativeCoupledGeometryRuntimeV96:
    __slots__ = ("receipt", "library_path", "_function", "_library",
                 "_dependency", "_call_lock")

    def __init__(self, receipt: NativeCoupledGeometryBuildReceiptV96,
                 library_path: Path, function, library, dependency) -> None:
        validate_native_coupled_geometry_build_receipt_v96(receipt)
        self.receipt = receipt
        self.library_path = Path(library_path).resolve()
        self._function = function
        self._library = library
        self._dependency = dependency
        self._call_lock = threading.Lock()

    def evaluate(
        self, execution: CoupledGeometryNativeCPUExecutionV96,
        midpoint_q: object, *, bed_elevation_m: object,
        accepted_water_density_kg_m3: object,
        accepted_mixture_density_kg_m3: object,
    ) -> NativeCoupledGeometryRateResultV96:
        execution_document_coupled_geometry_native_v96(execution)
        if self.receipt != execution.native_build_receipt:
            raise CoupledGeometryNativeV96Error(
                "v96 native runtime receipt differs")
        if (not self.library_path.is_file() or
                _file_sha256(self.library_path) != self.receipt.library_sha256):
            raise CoupledGeometryNativeV96Error(
                "v96 native loaded library bytes differ")
        base = execution.base_execution
        nc, nf = base.cell_count, base.face_count
        q = np.ascontiguousarray(_array(
            midpoint_q, (3, nc), "midpoint state"))
        bed = np.ascontiguousarray(_array(
            bed_elevation_m, (nc,), "accepted bed elevation"))
        water = np.ascontiguousarray(_array(
            accepted_water_density_kg_m3, (nc,), "accepted water density",
            positive=True))
        mixture = np.ascontiguousarray(_array(
            accepted_mixture_density_kg_m3, (nc,),
            "accepted mixture density", positive=True))
        if np.any(mixture < water):
            raise CoupledGeometryNativeV96Error(
                "v96 mixture density is below accepted water density")
        density = np.ascontiguousarray(
            mixture if execution.density_mode ==
            DENSITY_WATER_PLUS_SEDIMENT else water)
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
                distance, area, offset, faces, signs, density,
                base.hydraulic_reference_density_kg_m3, base.gravity_m_s2,
                base.dry_tolerance_m, 1, execution.native_threads,
                base_rate, density_rate, full_rate, face_acceleration,
                cell_acceleration, face_water_flux,
                ctypes.byref(riemann_calls), ctypes.byref(density_calls),
                ctypes.byref(status))
        if status.value != 0:
            raise CoupledGeometryNativeV96Error(
                f"v96 native full-rate failed with status {status.value}")
        if riemann_calls.value != nf or density_calls.value != 1:
            raise CoupledGeometryNativeV96Error(
                "v96 native evaluator owner counts differ")
        arrays = (base_rate, density_rate, full_rate, face_acceleration,
                  cell_acceleration, face_water_flux)
        if not all(np.isfinite(array).all() for array in arrays):
            raise CoupledGeometryNativeV96Error(
                "v96 native evaluator returned non-finite data")
        for array in arrays:
            array.setflags(write=False)
        return NativeCoupledGeometryRateResultV96(
            base_rate, density_rate, full_rate, face_acceleration,
            cell_acceleration, face_water_flux, _digest(bed),
            _digest(density), riemann_calls.value, density_calls.value)


def build_native_coupled_geometry_runtime_v96(
    repository_root: str | Path, build_directory: str | Path, *,
    compiler: str | Path | None = None,
) -> NativeCoupledGeometryRuntimeV96:
    root = Path(repository_root).resolve()
    build_root = Path(build_directory).resolve()
    build_root.mkdir(parents=True, exist_ok=True)
    relatives = (_PARAMETER_RELATIVE, _RPN2_RELATIVE, _UTILS_RELATIVE,
                 _V95_RELATIVE, _WRAPPER_RELATIVE)
    sources = [root/item for item in relatives]
    if not all(path.is_file() for path in sources):
        raise CoupledGeometryNativeV96Error(
            "v96 native source inventory is incomplete")
    hashes = {item: _file_sha256(path)
              for item, path in zip(relatives, sources)}
    if hashes != {item: _EXPECTED_SOURCE_HASHES[item] for item in relatives}:
        raise CoupledGeometryNativeV96Error(
            "v96 native or frozen augmented-fwave source differs")
    locked = tuple(sorted(verify_locked_sources_v85(root).items()))
    selected = Path(compiler) if compiler is not None else None
    if selected is None:
        discovered = shutil.which("gfortran")
        if discovered is None and os.name == "nt":
            fallback = Path(r"C:\Strawberry\c\bin\gfortran.exe")
            discovered = str(fallback) if fallback.is_file() else None
        if discovered is None:
            raise CoupledGeometryNativeV96Error("v96 native requires gfortran")
        selected = Path(discovered)
    selected = selected.resolve()
    if not selected.is_file():
        raise CoupledGeometryNativeV96Error("v96 native compiler is unavailable")
    flags = ("-O3", "-flto", "-fopenmp", "-shared",
             "-ffree-line-length-none", "-ffixed-line-length-none",
             "-static-libgfortran", "-static-libgcc")
    build_key = _fingerprint({
        "source_sha256": hashes,
        "compiler_sha256": _file_sha256(selected),
        "flags": list(flags), "abi_id": NATIVE_ABI_ID,
    })[:16]
    build = Path(tempfile.mkdtemp(
        prefix=f"v96_geometry_{build_key}_", dir=build_root))
    library_name = ("braids_coupled_geometry_v96.dll" if os.name == "nt"
                    else "libbraids_coupled_geometry_v96.so")
    partial = build/(library_name + ".partial")
    library_path = build/library_name
    completed = subprocess.run(
        [str(selected), *flags, *(str(path) for path in sources),
         "-o", str(partial)], cwd=build, capture_output=True, text=True,
        timeout=180, check=False)
    if completed.returncode != 0 or not partial.is_file():
        raise CoupledGeometryNativeV96Error(
            "v96 native build failed:\n" + completed.stdout + completed.stderr)
    os.replace(partial, library_path)
    version = subprocess.run(
        [str(selected), "--version"], capture_output=True, text=True,
        timeout=30, check=True).stdout.splitlines()[0]
    placeholder = NativeCoupledGeometryBuildReceiptV96(
        NATIVE_ABI_ID, hashes[_WRAPPER_RELATIVE], hashes[_V95_RELATIVE],
        hashes[_PARAMETER_RELATIVE], hashes[_RPN2_RELATIVE],
        hashes[_UTILS_RELATIVE], _file_sha256(library_path), version, flags,
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
            "+00:00", "Z"), True, LOCKED_UPSTREAM_COMMIT, locked, "")
    receipt = NativeCoupledGeometryBuildReceiptV96(
        **{**placeholder.__dict__,
           "receipt_sha256": _fingerprint(_build_payload(placeholder))})
    dependency = None
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        dependency = os.add_dll_directory(str(selected.parent))
    library = ctypes.CDLL(str(library_path))
    function = library.braids_coupled_full_rate_v96
    float_array = np.ctypeslib.ndpointer(
        dtype=np.float64, flags="C_CONTIGUOUS")
    int_array = np.ctypeslib.ndpointer(
        dtype=np.int32, flags="C_CONTIGUOUS")
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
    return NativeCoupledGeometryRuntimeV96(
        receipt, library_path, function, library, dependency)


def method_identity_coupled_geometry_native_v96() -> dict[str, object]:
    return {
        "scope_id": SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "backend": "native_cpu_openmp",
        "augmented_fwave": "frozen_geoclaw_rpn2_unchanged",
        "runtime_inputs": ["accepted_parent_bed_elevation",
                           "accepted_spatial_water_density",
                           "accepted_spatial_mixture_density"],
        "water_eos_when_sediment_densin_off": "retained",
        "production_admission": False,
    }


__all__ = [
    "DENSITY_WATER_EOS", "DENSITY_WATER_PLUS_SEDIMENT",
    "CoupledGeometryNativeCPUExecutionV96",
    "CoupledGeometryNativeV96Error", "NativeCoupledGeometryBuildReceiptV96",
    "NativeCoupledGeometryRateResultV96", "NativeCoupledGeometryRuntimeV96",
    "build_native_coupled_geometry_runtime_v96",
    "execution_document_coupled_geometry_native_v96",
    "method_identity_coupled_geometry_native_v96",
    "resolve_coupled_geometry_native_cpu_execution_v96",
    "validate_native_coupled_geometry_build_receipt_v96",
]
