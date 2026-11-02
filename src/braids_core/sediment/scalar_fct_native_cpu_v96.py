"""Native OpenMP auxiliary for the accepted v92 multi-lane scalar FCT.

The runtime owns no state, clock, boundary memory, or scalar lifecycle.  One
call consumes all sediment and active fluid lanes and returns the same three
numerical arrays as ``_fluid_fct_active_lanes_v92``.  The outer v96 owner is
the sole lifecycle/commit owner.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Callable, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "solver" / "wave" / "braids_scalar_fct_v96_module.f90"
ABI_ID = "braids-v96-native-openmp-scalar-fct-v1"
RESULT_KEYS = frozenset({
    "mass_kg_m2", "advective_face_flux_kg_m_s",
    "advective_boundary_ledger_kg", "evaluation_count", "native_threads",
    "native_runtime_sha256", "result_sha256",
})


class NativeScalarFCTV96Error(ValueError):
    """The native scalar-FCT build, call, or result is invalid."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False).encode("ascii")
    except (TypeError, ValueError, OverflowError) as error:
        raise NativeScalarFCTV96Error(
            "native v96 scalar FCT identity is not canonical") from error


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _digest(*values: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(value)
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(_canonical(list(array.shape)))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _finite(value: object, name: str, *, positive: bool = False) -> float:
    if (isinstance(value, (bool, np.bool_)) or
            not isinstance(value, (int, float, np.integer, np.floating)) or
            not math.isfinite(float(value))):
        raise NativeScalarFCTV96Error(
            f"native v96 scalar FCT {name} must be finite numeric data")
    result = float(value)
    if positive and result <= 0.0:
        raise NativeScalarFCTV96Error(
            f"native v96 scalar FCT {name} must be positive")
    return result


def _threads(value: object) -> int:
    if type(value) is not int or not 1 <= value <= 256:
        raise NativeScalarFCTV96Error(
            "native v96 scalar FCT threads must be an integer in 1..256")
    return value


def _array(value: object, shape: tuple[int, ...], name: str, *,
           dtype=np.float64, nonnegative: bool = False) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != shape or raw.dtype.kind not in "fiub":
        raise NativeScalarFCTV96Error(
            f"native v96 scalar FCT {name} must have shape {shape}")
    if not np.isfinite(raw).all():
        raise NativeScalarFCTV96Error(
            f"native v96 scalar FCT {name} is not finite")
    target = np.dtype(dtype)
    if np.issubdtype(target, np.integer):
        limits = np.iinfo(target)
        if (np.any(raw != np.floor(raw)) or np.any(raw < limits.min) or
                np.any(raw > limits.max)):
            raise NativeScalarFCTV96Error(
                f"native v96 scalar FCT {name} is not exact integer data")
    result = np.ascontiguousarray(raw, dtype=target)
    if nonnegative and np.any(result < 0):
        raise NativeScalarFCTV96Error(
            f"native v96 scalar FCT {name} must be nonnegative")
    return result


def _cancel_check(value: bool | Callable[[], bool]) -> Callable[[], bool]:
    if isinstance(value, (bool, np.bool_)):
        return lambda: bool(value)
    if not callable(value):
        raise NativeScalarFCTV96Error(
            "native v96 scalar FCT cancellation must be Boolean or callable")

    def checked() -> bool:
        result = value()
        if not isinstance(result, (bool, np.bool_)):
            raise NativeScalarFCTV96Error(
                "native v96 scalar FCT cancellation returned non-Boolean data")
        return bool(result)

    return checked


def _topology_data(topology: object, nc: int, nf: int) -> dict[str, object]:
    try:
        declared_nc = int(topology.cell_count)
        declared_nf = int(topology.face_count)
        nb = int(topology.boundary_count)
        dry = _finite(topology.dry_tolerance_m, "dry tolerance")
    except (AttributeError, TypeError, ValueError) as error:
        raise NativeScalarFCTV96Error(
            "native v96 scalar FCT topology metadata differs") from error
    if declared_nc != nc or declared_nf != nf or nb < 0 or dry < 0.0:
        raise NativeScalarFCTV96Error(
            "native v96 scalar FCT topology dimensions differ")
    owner = _array(topology.owner, (nf,), "owner", dtype=np.int32)
    neighbour = _array(
        topology.neighbour, (nf,), "neighbour", dtype=np.int32)
    boundary_index = _array(
        topology.boundary_index, (nf,), "boundary index", dtype=np.int32)
    length = _array(
        topology.face_length_m, (nf,), "face length", nonnegative=True)
    area = _array(topology.cell_area_m2, (nc,), "cell area", nonnegative=True)
    offset = _array(
        topology.cell_face_offset, (nc + 1,), "cell-face offset",
        dtype=np.int32)
    faces_raw = np.asarray(topology.cell_faces)
    if faces_raw.ndim != 1:
        raise NativeScalarFCTV96Error(
            "native v96 scalar FCT cell faces must be one-dimensional")
    nadj = int(faces_raw.size)
    faces = _array(faces_raw, (nadj,), "cell faces", dtype=np.int32)
    signs = _array(
        topology.cell_face_sign, (nadj,), "cell-face sign", dtype=np.int32)
    width = _array(
        topology.rectangle_width_m, (2, nc), "rectangle width")
    normal = _array(topology.normal, (2, nf), "face normal")
    weno = _array(
        topology.weno_cells, (5, 2, nf), "WENO cells", dtype=np.int32)
    if (np.any(owner < 0) or np.any(owner >= nc) or
            np.any(neighbour < -1) or np.any(neighbour >= nc) or
            np.any(boundary_index < -1) or
            (nb == 0 and np.any(boundary_index >= 0)) or
            (nb > 0 and np.any(boundary_index >= nb)) or
            np.any(length <= 0.0) or np.any(area <= 0.0) or
            offset[0] != 0 or offset[-1] != nadj or
            np.any(np.diff(offset) < 0) or np.any(faces < 0) or
            np.any(faces >= nf) or
            np.any((signs != -1) & (signs != 1)) or
            np.any(weno < -1) or np.any(weno >= nc)):
        raise NativeScalarFCTV96Error(
            "native v96 scalar FCT topology connectivity differs")
    # Vectorized metadata validation keeps the Python bridge out of the
    # per-cell/per-face numerical hot path on large reaches.
    adjacency_cell = np.repeat(
        np.arange(nc, dtype=np.int32), np.diff(offset).astype(np.int64))
    if np.any((adjacency_cell != owner[faces]) &
              (adjacency_cell != neighbour[faces])):
        raise NativeScalarFCTV96Error(
            "native v96 scalar FCT adjacency owner differs")
    exterior = np.flatnonzero(boundary_index >= 0)
    boundary_faces = np.ascontiguousarray(
        exterior[np.argsort(boundary_index[exterior], kind="stable")],
        dtype=np.int32)
    boundary_offset = np.ascontiguousarray(np.concatenate((
        np.array([0], dtype=np.int64),
        np.cumsum(np.bincount(
            boundary_index[exterior], minlength=nb), dtype=np.int64),
    )), dtype=np.int32)
    return {
        "nb": nb, "nadj": nadj, "nbf": int(boundary_faces.size),
        "dry": dry, "owner": owner,
        "neighbour": neighbour, "boundary_index": boundary_index,
        "length": length, "area": area, "offset": offset, "faces": faces,
        "signs": signs, "width": width, "normal": normal, "weno": weno,
        "boundary_offset": boundary_offset,
        "boundary_faces": boundary_faces,
    }


def _prepare(topology: object, mass: object, depth: object,
             next_depth: object, water_flux: object, left: object,
             right: object, boundary: object, hydro_constant: object,
             *, dt_s: float, native_threads: int) -> dict[str, object]:
    raw_mass = np.asarray(mass)
    if raw_mass.ndim != 2 or raw_mass.shape[0] < 1 or raw_mass.shape[1] < 1:
        raise NativeScalarFCTV96Error(
            "native v96 scalar FCT mass must contain lanes and cells")
    nl, nc = map(int, raw_mass.shape)
    try:
        nf = int(topology.face_count)
    except (AttributeError, TypeError, ValueError) as error:
        raise NativeScalarFCTV96Error(
            "native v96 scalar FCT face count differs") from error
    values = {
        "mass": _array(raw_mass, (nl, nc), "mass", nonnegative=True),
        "depth": _array(depth, (nc,), "depth", nonnegative=True),
        "next_depth": _array(
            next_depth, (nc,), "next depth", nonnegative=True),
        "water": _array(water_flux, (nf,), "water flux"),
        "left": _array(left, (nl, nf), "left reconstruction",
                       nonnegative=True),
        "right": _array(right, (nl, nf), "right reconstruction",
                        nonnegative=True),
        "boundary": _array(boundary, (nl, nf), "boundary concentration",
                           nonnegative=True),
        "hydro": _array(hydro_constant, (nc,), "hydro constant",
                        dtype=np.int32),
    }
    if np.any((values["hydro"] != 0) & (values["hydro"] != 1)):
        raise NativeScalarFCTV96Error(
            "native v96 scalar FCT hydro flags must be Boolean")
    values.update(_topology_data(topology, nc, nf))
    values.update({"nl": nl, "nc": nc, "nf": nf,
                   "dt": _finite(dt_s, "timestep", positive=True),
                   "threads": _threads(native_threads)})
    values["input_sha256"] = _hash({
        "abi": ABI_ID, "dt_s": values["dt"], "dry": values["dry"],
        "native_threads": values["threads"],
        "arrays_sha256": _digest(*(
            values[key] for key in (
                "mass", "depth", "next_depth", "water", "left", "right",
                "boundary", "hydro", "owner", "neighbour",
                "boundary_index", "length", "area", "offset", "faces",
                "signs", "width", "normal", "weno", "boundary_offset",
                "boundary_faces"))),
    })
    return values


def _result_token(runtime_sha256: str, prepared: Mapping[str, object],
                  mass: np.ndarray, flux: np.ndarray,
                  ledger: np.ndarray) -> str:
    return _hash({
        "abi": ABI_ID, "native_runtime_sha256": runtime_sha256,
        "native_threads": prepared["threads"],
        "evaluation_count": 1,
        "input_sha256": prepared["input_sha256"],
        "output_sha256": _digest(mass, flux, ledger),
    })


class NativeScalarFCTRuntimeV96:
    """Re-entrant pure native scalar-FCT evaluator."""

    def __init__(self, receipt: Mapping[str, object], library_path: Path,
                 library, dependency):
        self.receipt = dict(receipt)
        self.library_path = Path(library_path)
        self._library = library
        self._dependency = dependency
        self._fct = library.braids_scalar_fct_v96
        fa = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        ia = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
        self._fct.argtypes = (
            [ctypes.c_int]*6 + [ctypes.c_double]*2 + [ctypes.c_int] +
            [fa]*7 + [ia]*4 + [fa]*2 + [ia]*3 + [fa]*2 + [ia]*3 +
            [fa]*3 + [ctypes.POINTER(ctypes.c_int)])
        self._fct.restype = None

    @property
    def runtime_sha256(self) -> str:
        return str(self.receipt["receipt_sha256"])

    def _validate_runtime(self) -> None:
        body = {key: value for key, value in self.receipt.items()
                if key != "receipt_sha256"}
        if (self.receipt.get("abi") != ABI_ID or
                self.receipt.get("receipt_sha256") != _hash(body) or
                self.receipt.get("native_source_sha256") != _sha(SOURCE) or
                not self.library_path.is_file() or
                self.receipt.get("library_sha256") != _sha(self.library_path)):
            raise NativeScalarFCTV96Error(
                "native v96 scalar FCT runtime identity differs")

    def evaluate(
        self, topology: object, mass: object, depth: object,
        next_depth: object, water_flux: object, left: object, right: object,
        boundary: object, hydro_constant: object, *, dt_s: float,
        native_threads: int = 4,
        cancelled: bool | Callable[[], bool] = False,
    ) -> dict[str, object]:
        """Evaluate all lanes once without advancing any lifecycle clock."""
        self._validate_runtime()
        cancel = _cancel_check(cancelled)
        if cancel():
            raise InterruptedError(
                "native v96 scalar FCT cancelled before launch")
        p = _prepare(
            topology, mass, depth, next_depth, water_flux, left, right,
            boundary, hydro_constant, dt_s=dt_s,
            native_threads=native_threads)
        result = np.empty((p["nl"], p["nc"]), dtype=np.float64)
        flux = np.empty((p["nl"], p["nf"]), dtype=np.float64)
        ledger = np.empty((p["nl"], p["nb"]), dtype=np.float64)
        status = ctypes.c_int()
        self._fct(
            p["nl"], p["nc"], p["nf"], p["nb"], p["nadj"], p["nbf"],
            p["dt"], p["dry"], p["threads"], p["mass"], p["depth"],
            p["next_depth"], p["water"], p["left"], p["right"],
            p["boundary"], p["hydro"], p["owner"], p["neighbour"],
            p["boundary_index"], p["length"], p["area"], p["offset"],
            p["faces"], p["signs"], p["width"], p["normal"], p["weno"],
            p["boundary_offset"], p["boundary_faces"],
            result, flux, ledger, ctypes.byref(status))
        if status.value != 0:
            messages = {
                10: "closed face has nonzero accepted water flux",
                20: "accepted water flux overdraws low-order scalar mass",
                30: "scalar FCT result is invalid",
            }
            raise NativeScalarFCTV96Error(
                "native v96 scalar FCT " +
                messages.get(status.value, f"kernel status {status.value}"))
        if cancel():
            raise InterruptedError(
                "native v96 scalar FCT cancelled after launch")
        if (not np.isfinite(result).all() or not np.isfinite(flux).all() or
                not np.isfinite(ledger).all()):
            raise NativeScalarFCTV96Error(
                "native v96 scalar FCT returned nonfinite arrays")
        for array in (result, flux, ledger):
            array.setflags(write=False)
        output = {
            "mass_kg_m2": result,
            "advective_face_flux_kg_m_s": flux,
            "advective_boundary_ledger_kg": ledger,
            "evaluation_count": 1,
            "native_threads": p["threads"],
            "native_runtime_sha256": self.runtime_sha256,
            "result_sha256": _result_token(
                self.runtime_sha256, p, result, flux, ledger),
        }
        return output

    def evaluate_arrays(self, *args, **kwargs) -> tuple[np.ndarray, ...]:
        """Return the existing v92 tuple from the same single native launch."""
        result = self.evaluate(*args, **kwargs)
        return (result["mass_kg_m2"],
                result["advective_face_flux_kg_m_s"],
                result["advective_boundary_ledger_kg"])

    def validate_result(
        self, result: Mapping[str, object], topology: object, mass: object,
        depth: object, next_depth: object, water_flux: object, left: object,
        right: object, boundary: object, hydro_constant: object, *,
        dt_s: float, native_threads: int = 4,
    ) -> None:
        """Recompute the call binding and authenticate an existing result."""
        self._validate_runtime()
        if not isinstance(result, Mapping) or set(result) != RESULT_KEYS:
            raise NativeScalarFCTV96Error(
                "native v96 scalar FCT result inventory differs")
        p = _prepare(
            topology, mass, depth, next_depth, water_flux, left, right,
            boundary, hydro_constant, dt_s=dt_s,
            native_threads=native_threads)
        arrays = (
            _array(result["mass_kg_m2"], (p["nl"], p["nc"]), "result mass",
                   nonnegative=True),
            _array(result["advective_face_flux_kg_m_s"],
                   (p["nl"], p["nf"]), "result flux"),
            _array(result["advective_boundary_ledger_kg"],
                   (p["nl"], p["nb"]), "result ledger"),
        )
        if (result["evaluation_count"] != 1 or
                result["native_threads"] != p["threads"] or
                result["native_runtime_sha256"] != self.runtime_sha256 or
                result["result_sha256"] !=
                _result_token(self.runtime_sha256, p, *arrays)):
            raise NativeScalarFCTV96Error(
                "native v96 scalar FCT result is stale or input-mismatched")


def build_native_scalar_fct_runtime_v96(
    build_directory: str | Path, *, compiler: str | Path | None = None,
) -> NativeScalarFCTRuntimeV96:
    """Build an isolated, source-authenticated native OpenMP library."""
    if not SOURCE.is_file():
        raise NativeScalarFCTV96Error(
            "native v96 scalar FCT source is missing")
    chosen = Path(compiler).resolve() if compiler is not None else None
    if chosen is None:
        located = shutil.which("gfortran")
        chosen = Path(located).resolve() if located else None
    if chosen is None and os.name == "nt":
        fallback = Path(r"C:\Strawberry\c\bin\gfortran.exe")
        chosen = fallback.resolve() if fallback.is_file() else None
    if chosen is None or not chosen.is_file():
        raise NativeScalarFCTV96Error(
            "native v96 scalar FCT requires gfortran")
    flags = [
        "-O3", "-fopenmp", "-shared", "-ffree-line-length-none",
        "-fno-fast-math", "-ffp-contract=off", "-static-libgfortran",
        "-static-libgcc",
    ]
    source_sha = _sha(SOURCE)
    compiler_sha = _sha(chosen)
    key = _hash({"source": source_sha, "compiler": compiler_sha,
                 "flags": flags})[:16]
    build_root = Path(build_directory).resolve()
    build_root.mkdir(parents=True, exist_ok=True)
    build = Path(tempfile.mkdtemp(prefix=f"v96_scalar_fct_{key}_",
                                  dir=build_root))
    name = ("braids_scalar_fct_v96.dll" if os.name == "nt" else
            "libbraids_scalar_fct_v96.so")
    partial = build/(name + ".partial")
    library_path = build/name
    completed = subprocess.run(
        [str(chosen), *flags, str(SOURCE), "-o", str(partial)], cwd=build,
        capture_output=True, text=True, timeout=120)
    if completed.returncode != 0 or not partial.is_file():
        raise NativeScalarFCTV96Error(
            "native v96 scalar FCT build failed:\n" +
            completed.stdout + completed.stderr)
    if source_sha != _sha(SOURCE):
        raise NativeScalarFCTV96Error(
            "native v96 scalar FCT source changed during compilation")
    os.replace(partial, library_path)
    version = subprocess.run(
        [str(chosen), "--version"], capture_output=True, text=True,
        check=True, timeout=20).stdout.splitlines()[0]
    body = {
        "abi": ABI_ID,
        "native_source": str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
        "native_source_sha256": source_sha,
        "library_sha256": _sha(library_path),
        "compiler_identity": version,
        "compiler_sha256": compiler_sha,
        "compile_flags": flags,
        "openmp_parallel_face_cell_lane_loops": True,
        "cell_owned_deterministic_gathers": True,
        "boundary_owned_csr_deterministic_gathers": True,
        "nested_openmp_team_creation": False,
        "numerical_python_cell_face_loop_count": 0,
        "state_or_clock_owner_count": 0,
        "production_admission": False,
    }
    receipt = {**body, "receipt_sha256": _hash(body)}
    dependency = None
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        dependency = os.add_dll_directory(str(chosen.parent))
    return NativeScalarFCTRuntimeV96(
        receipt, library_path, ctypes.CDLL(str(library_path)), dependency)


__all__ = [
    "ABI_ID", "NativeScalarFCTRuntimeV96", "NativeScalarFCTV96Error",
    "RESULT_KEYS", "build_native_scalar_fct_runtime_v96",
]
