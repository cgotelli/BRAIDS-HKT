"""Native OpenMP four-child zero-time refine/restrict numerical batch.

The batch is a pure candidate producer.  It advances no clock, evaluates no
constitutive law, and owns no hierarchy publication.  Its equations mirror
``coupled_amr_zero_time_transfer_v98``: stable dry-aware stage fill, two
roundoff-closure corrections, endpoint dry transfer, then area restriction.
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
from typing import Callable, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "solver" / "wave" / \
    "braids_amr_zero_time_transfer_v98_module.f90"
SCHEMA = 1
SCOPE_ID = "braids.coupled_amr.zero_time_transfer.native_cpu.v98"
CHILD_COUNT = 4


class NativeAMRZeroTimeTransferCPUV98Error(ValueError):
    """Native zero-time input, build, result, or identity differs."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False).encode("ascii")
    except (TypeError, ValueError, OverflowError) as error:
        raise NativeAMRZeroTimeTransferCPUV98Error(
            "v98 native zero-time value is not canonical JSON") from error


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(*values: object) -> str:
    digest = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(value)
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(_canonical(list(array.shape)))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _array(value: object, shape: tuple[int, ...], name: str, *,
           nonnegative: bool = False) -> np.ndarray:
    raw = np.asarray(value)
    if (raw.shape != shape or raw.dtype.kind not in "fiu" or
            not np.isfinite(raw).all() or
            (nonnegative and np.any(raw < 0.0))):
        raise NativeAMRZeroTimeTransferCPUV98Error(
            f"v98 native zero-time {name} differs")
    return np.ascontiguousarray(raw, dtype=np.float64)


def _poll(cancelled: bool | Callable[[], bool]) -> Callable[[], bool]:
    if isinstance(cancelled, (bool, np.bool_)):
        return lambda: bool(cancelled)
    if not callable(cancelled):
        raise TypeError("cancelled must be Boolean or callable")

    def checked() -> bool:
        result = cancelled()
        if not isinstance(result, (bool, np.bool_)):
            raise TypeError("cancelled callback must return Boolean data")
        return bool(result)
    return checked


@dataclass(frozen=True)
class NativeAMRZeroTimeTransferResultCPUV98:
    child_q: np.ndarray
    child_suspended: np.ndarray
    child_bed: np.ndarray
    child_fluid: np.ndarray
    restricted_q: np.ndarray
    restricted_suspended: np.ndarray
    restricted_bed: np.ndarray
    restricted_fluid: np.ndarray
    parent_status: np.ndarray
    runtime_sha256: str
    backend: str
    native_threads: int
    parent_count: int
    fraction_count: int
    input_sha256: str
    output_sha256: str
    stage_fill_evaluation_count: int
    refinement_operator_application_count: int
    restriction_operator_application_count: int
    dry_cleanup_owner_count: int
    accepted_clock_increment_count: int
    physical_time_increment_count: int
    constitutive_evaluation_count: int
    state_publication_count: int
    numerical_python_parent_child_loop_count: int
    receipt_sha256: str


def _result_payload(
    value: NativeAMRZeroTimeTransferResultCPUV98,
) -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID + ":result",
        "runtime_sha256": value.runtime_sha256,
        "backend": value.backend,
        "native_threads": value.native_threads,
        "parent_count": value.parent_count,
        "fraction_count": value.fraction_count,
        "input_sha256": value.input_sha256,
        "output_sha256": value.output_sha256,
        "stage_fill_evaluation_count": value.stage_fill_evaluation_count,
        "refinement_operator_application_count":
            value.refinement_operator_application_count,
        "restriction_operator_application_count":
            value.restriction_operator_application_count,
        "dry_cleanup_owner_count": value.dry_cleanup_owner_count,
        "accepted_clock_increment_count":
            value.accepted_clock_increment_count,
        "physical_time_increment_count": value.physical_time_increment_count,
        "constitutive_evaluation_count": value.constitutive_evaluation_count,
        "state_publication_count": value.state_publication_count,
        "numerical_python_parent_child_loop_count":
            value.numerical_python_parent_child_loop_count,
        "transfer": "stable_stage_fill_two_corrections_then_area_restrict",
    }


def validate_native_amr_zero_time_transfer_result_cpu_v98(
    value: NativeAMRZeroTimeTransferResultCPUV98,
) -> None:
    if type(value) is not NativeAMRZeroTimeTransferResultCPUV98:
        raise NativeAMRZeroTimeTransferCPUV98Error(
            "v98 native zero-time result type differs")
    np_, ng = value.parent_count, value.fraction_count
    arrays = (
        (value.child_q, (3, np_, 4), "child q", False),
        (value.child_suspended, (ng, np_, 4), "child suspension", True),
        (value.child_bed, (ng, np_, 4), "child bed", True),
        (value.child_fluid, (2, np_, 4), "child fluid", False),
        (value.restricted_q, (3, np_), "restricted q", False),
        (value.restricted_suspended, (ng, np_),
         "restricted suspension", True),
        (value.restricted_bed, (ng, np_), "restricted bed", True),
        (value.restricted_fluid, (2, np_), "restricted fluid", False),
    )
    for array, shape, name, nonnegative in arrays:
        checked = _array(array, shape, name, nonnegative=nonnegative)
        if (not np.array_equal(checked, array) or array.flags.writeable or
                not array.flags.c_contiguous):
            raise NativeAMRZeroTimeTransferCPUV98Error(
                f"v98 native zero-time {name} identity differs")
    status = np.asarray(value.parent_status)
    if (status.shape != (np_,) or status.dtype != np.int32 or
            status.flags.writeable or not status.flags.c_contiguous or
            np.any(status != 0)):
        raise NativeAMRZeroTimeTransferCPUV98Error(
            "v98 native zero-time parent status differs")
    if (value.backend != "native_cpu_openmp" or
            type(value.native_threads) is not int or
            not 1 <= value.native_threads <= 256 or
            min(np_, ng) < 1 or
            value.stage_fill_evaluation_count != np_ or
            value.refinement_operator_application_count != 1 or
            value.restriction_operator_application_count != 1 or
            value.dry_cleanup_owner_count != 1 or
            value.accepted_clock_increment_count != 0 or
            value.physical_time_increment_count != 0 or
            value.constitutive_evaluation_count != 0 or
            value.state_publication_count != 0 or
            value.numerical_python_parent_child_loop_count != 0 or
            value.output_sha256 != _digest(
                value.child_q, value.child_suspended, value.child_bed,
                value.child_fluid, value.restricted_q,
                value.restricted_suspended, value.restricted_bed,
                value.restricted_fluid, value.parent_status) or
            value.receipt_sha256 != _hash(_result_payload(value))):
        raise NativeAMRZeroTimeTransferCPUV98Error(
            "v98 native zero-time result identity or owner counts differ")


class NativeAMRZeroTimeTransferCPUV98:
    """Loaded native batch; one OpenMP work item per independent parent."""

    def __init__(self, receipt: Mapping[str, object], library: object,
                 dependency: object, path: Path):
        self.receipt = dict(receipt)
        self.library = library
        self.dependency = dependency
        self.path = Path(path)
        self.function = library.braids_amr_zero_time_transfer_v98
        fa = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        ia = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
        self.function.argtypes = [ctypes.c_int]*3 + [ctypes.c_double] + \
            [fa]*15 + [ia]
        self.function.restype = None

    @property
    def runtime_sha256(self) -> str:
        return str(self.receipt["receipt_sha256"])

    def _validate_identity(self) -> None:
        payload = {key: value for key, value in self.receipt.items()
                   if key != "receipt_sha256"}
        if (self.receipt.get("receipt_sha256") != _hash(payload) or
                self.receipt.get("source_sha256") != _sha(SOURCE) or
                self.receipt.get("wrapper_sha256") !=
                _sha(Path(__file__)) or
                self.receipt.get("library_sha256") != _sha(self.path)):
            raise NativeAMRZeroTimeTransferCPUV98Error(
                "v98 native zero-time runtime/source identity differs")

    def evaluate(
        self, parent_area_m2: object, child_area_m2: object,
        child_bed_elevation_m: object, parent_q: object,
        parent_suspended_kg_m2: object, parent_bed_kg_m2: object,
        parent_fluid_integral: object, *, dry_tolerance_m: float,
        native_threads: int = 4,
        cancelled: bool | Callable[[], bool] = False,
    ) -> NativeAMRZeroTimeTransferResultCPUV98:
        poll = _poll(cancelled)
        if poll():
            raise InterruptedError(
                "v98 native zero-time transfer cancelled before launch")
        self._validate_identity()
        if (type(native_threads) is not int or
                not 1 <= native_threads <= 256 or
                isinstance(dry_tolerance_m, bool) or
                not isinstance(dry_tolerance_m, (int, float)) or
                not np.isfinite(dry_tolerance_m) or
                float(dry_tolerance_m) <= 0.0):
            raise NativeAMRZeroTimeTransferCPUV98Error(
                "v98 native zero-time controls differ")
        parent_area_raw = np.asarray(parent_area_m2)
        if parent_area_raw.ndim != 1 or parent_area_raw.size < 1:
            raise NativeAMRZeroTimeTransferCPUV98Error(
                "v98 native zero-time parent area shape differs")
        np_ = int(parent_area_raw.size)
        suspended_raw = np.asarray(parent_suspended_kg_m2)
        if suspended_raw.ndim != 2 or suspended_raw.shape[1] != np_ or \
                suspended_raw.shape[0] < 1:
            raise NativeAMRZeroTimeTransferCPUV98Error(
                "v98 native zero-time suspended shape differs")
        ng = int(suspended_raw.shape[0])
        inputs = (
            _array(parent_area_m2, (np_,), "parent area", nonnegative=True),
            _array(child_area_m2, (np_, 4), "child area", nonnegative=True),
            _array(child_bed_elevation_m, (np_, 4), "child bed elevation"),
            _array(parent_q, (3, np_), "parent q"),
            _array(parent_suspended_kg_m2, (ng, np_),
                   "parent suspension", nonnegative=True),
            _array(parent_bed_kg_m2, (ng, np_),
                   "parent bed", nonnegative=True),
            _array(parent_fluid_integral, (2, np_), "parent fluid"),
        )
        child_q = np.empty((3, np_, 4), dtype=np.float64)
        child_suspended = np.empty((ng, np_, 4), dtype=np.float64)
        child_bed = np.empty((ng, np_, 4), dtype=np.float64)
        child_fluid = np.empty((2, np_, 4), dtype=np.float64)
        restricted_q = np.empty((3, np_), dtype=np.float64)
        restricted_suspended = np.empty((ng, np_), dtype=np.float64)
        restricted_bed = np.empty((ng, np_), dtype=np.float64)
        restricted_fluid = np.empty((2, np_), dtype=np.float64)
        status = np.empty(np_, dtype=np.int32)
        outputs = (child_q, child_suspended, child_bed, child_fluid,
                   restricted_q, restricted_suspended, restricted_bed,
                   restricted_fluid)
        self.function(
            np_, ng, native_threads, float(dry_tolerance_m), *inputs,
            *outputs, status)
        if np.any(status != 0):
            failing = np.flatnonzero(status != 0).tolist()
            raise NativeAMRZeroTimeTransferCPUV98Error(
                f"v98 native zero-time parent status differs at {failing}")
        if poll():
            raise InterruptedError(
                "v98 native zero-time transfer cancelled after launch")
        for array in (*outputs, status):
            array.setflags(write=False)
        placeholder = NativeAMRZeroTimeTransferResultCPUV98(
            *outputs, status, self.runtime_sha256, "native_cpu_openmp",
            native_threads, np_, ng, _digest(*inputs),
            _digest(*outputs, status), np_, 1, 1, 1, 0, 0, 0, 0, 0, "")
        result = NativeAMRZeroTimeTransferResultCPUV98(
            **{**placeholder.__dict__,
               "receipt_sha256": _hash(_result_payload(placeholder))})
        validate_native_amr_zero_time_transfer_result_cpu_v98(result)
        return result


def build_native_amr_zero_time_transfer_cpu_v98(
    build_directory: str | Path, *, compiler: str | Path | None = None,
) -> NativeAMRZeroTimeTransferCPUV98:
    """Compile/load one immutable native transfer runtime."""
    chosen = Path(compiler).resolve() if compiler is not None else None
    if chosen is None:
        located = shutil.which("gfortran")
        if located is None and os.name == "nt":
            fallback = Path(r"C:\Strawberry\c\bin\gfortran.exe")
            located = str(fallback) if fallback.is_file() else None
        chosen = None if located is None else Path(located).resolve()
    if chosen is None or not chosen.is_file():
        raise NativeAMRZeroTimeTransferCPUV98Error(
            "v98 native zero-time transfer requires gfortran")
    root = Path(build_directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    build = Path(tempfile.mkdtemp(prefix="amr_zero_time_v98_", dir=root))
    path = build / ("amr_zero_time_v98.dll" if os.name == "nt"
                    else "libamr_zero_time_v98.so")
    partial = path.with_suffix(path.suffix + ".partial")
    flags = ["-O3", "-fopenmp", "-shared", "-ffree-line-length-none",
             "-static-libgfortran", "-static-libgcc"]
    source_sha = _sha(SOURCE)
    process = subprocess.run(
        [str(chosen), *flags, str(SOURCE), "-o", str(partial)],
        cwd=build, capture_output=True, text=True, timeout=120)
    if process.returncode or not partial.is_file():
        raise NativeAMRZeroTimeTransferCPUV98Error(
            "v98 native zero-time build failed: " +
            process.stdout + process.stderr)
    if _sha(SOURCE) != source_sha:
        raise NativeAMRZeroTimeTransferCPUV98Error(
            "v98 native zero-time source changed during build")
    os.replace(partial, path)
    receipt = {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID + ":runtime",
        "source_sha256": source_sha,
        "wrapper_sha256": _sha(Path(__file__)),
        "library_sha256": _sha(path),
        "compiler_sha256": _sha(chosen),
        "compile_flags": flags,
        "backend": "native_cpu_openmp",
        "thread_policy": "one_static_OpenMP_parent_loop_no_nested_team",
        "child_count": CHILD_COUNT,
        "numerical_python_parent_child_loop_count": 0,
        "state_or_clock_publication_count": 0,
        "production_admission": False,
    }
    receipt["receipt_sha256"] = _hash(receipt)
    dependency = (os.add_dll_directory(str(chosen.parent))
                  if os.name == "nt" and hasattr(os, "add_dll_directory")
                  else None)
    return NativeAMRZeroTimeTransferCPUV98(
        receipt, ctypes.CDLL(str(path)), dependency, path)


__all__ = [
    "CHILD_COUNT",
    "NativeAMRZeroTimeTransferCPUV98",
    "NativeAMRZeroTimeTransferCPUV98Error",
    "NativeAMRZeroTimeTransferResultCPUV98",
    "build_native_amr_zero_time_transfer_cpu_v98",
    "validate_native_amr_zero_time_transfer_result_cpu_v98",
]
