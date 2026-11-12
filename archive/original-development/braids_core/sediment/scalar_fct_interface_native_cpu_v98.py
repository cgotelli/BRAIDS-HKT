"""Native OpenMP scalar FCT with an authenticated AMR internal trace.

This is an additive v98 auxiliary.  It preserves the v96 FCT algebra on all
ordinary and physical-boundary faces, while selected ``neighbour == -1`` and
``boundary_index == -1`` faces consume an exterior coarse/fine trace.  Those
faces never enter the named physical-boundary ledger; their owner-oriented
mass integral is returned separately for the AMR register.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Callable, Mapping

import numpy as np

from .scalar_fct_native_cpu_v96 import (
    ROOT,
    SOURCE as BASE_SOURCE,
    _array,
    _cancel_check,
    _digest,
    _hash,
    _prepare,
    _sha,
)


SOURCE = ROOT / "solver" / "wave" / "braids_scalar_fct_interface_v98_module.f90"
ABI_ID = "braids-v98-native-openmp-scalar-fct-internal-interface-v1"
RESULT_KEYS = frozenset({
    "mass_kg_m2", "advective_face_flux_kg_m_s",
    "advective_boundary_ledger_kg", "internal_interface_owner_ledger_kg",
    "internal_interface_faces", "evaluation_count",
    "internal_interface_owner_count", "physical_boundary_surrogate_count",
    "native_threads", "native_runtime_sha256", "result_sha256",
})


class NativeScalarFCTInterfaceV98Error(ValueError):
    """The native v98 internal-interface scalar call is invalid."""


def _interface_inputs(
    prepared: Mapping[str, object], interface_faces: object,
    interface_concentration: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_faces = np.asarray(interface_faces)
    if raw_faces.ndim != 1 or raw_faces.size < 1:
        raise NativeScalarFCTInterfaceV98Error(
            "v98 scalar interface requires at least one selected face")
    ni = int(raw_faces.size)
    selected = _array(
        raw_faces, (ni,), "interface faces", dtype=np.int32)
    if (len(set(map(int, selected))) != ni or np.any(selected < 0) or
            np.any(selected >= prepared["nf"])):
        raise NativeScalarFCTInterfaceV98Error(
            "v98 scalar interface faces must be distinct topology indices")
    neighbour = prepared["neighbour"]
    boundary = prepared["boundary_index"]
    if (np.any(neighbour[selected] != -1) or
            np.any(boundary[selected] != -1)):
        raise NativeScalarFCTInterfaceV98Error(
            "v98 scalar interface is not an internal trace topology")
    concentration = _array(
        interface_concentration, (prepared["nl"], ni),
        "interface concentration", nonnegative=True)
    active = np.zeros(prepared["nf"], dtype=np.int32)
    active[selected] = 1
    expanded = np.zeros(
        (prepared["nl"], prepared["nf"]), dtype=np.float64)
    expanded[:, selected] = concentration
    return selected, active, expanded


def _input_token(prepared: Mapping[str, object], selected: np.ndarray,
                 active: np.ndarray, expanded: np.ndarray) -> str:
    return _hash({
        "abi": ABI_ID,
        "base_input_sha256": prepared["input_sha256"],
        "interface_arrays_sha256": _digest(selected, active, expanded),
    })


def _result_token(runtime_sha256: str, input_sha256: str,
                  prepared: Mapping[str, object], mass: np.ndarray,
                  flux: np.ndarray, ledger: np.ndarray,
                  interface_ledger: np.ndarray,
                  selected: np.ndarray) -> str:
    return _hash({
        "abi": ABI_ID,
        "native_runtime_sha256": runtime_sha256,
        "native_threads": prepared["threads"],
        "evaluation_count": 1,
        "internal_interface_owner_count": 1,
        "physical_boundary_surrogate_count": 0,
        "input_sha256": input_sha256,
        "output_sha256": _digest(
            mass, flux, ledger, interface_ledger, selected),
    })


class NativeScalarFCTInterfaceRuntimeV98:
    """Re-entrant scalar-FCT evaluator; owns no lifecycle or clock."""

    def __init__(self, receipt: Mapping[str, object], library_path: Path,
                 library, dependency):
        self.receipt = dict(receipt)
        self.library_path = Path(library_path)
        self._library = library
        self._dependency = dependency
        self._fct = library.braids_scalar_fct_interface_v98
        fa = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        ia = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
        self._fct.argtypes = (
            [ctypes.c_int]*7 + [ctypes.c_double]*2 + [ctypes.c_int] +
            [fa]*8 + [ia]*6 + [fa]*2 + [ia]*3 + [fa]*2 + [ia]*3 +
            [fa]*4 + [ctypes.POINTER(ctypes.c_int)])
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
                self.receipt.get("base_source_sha256") !=
                _sha(BASE_SOURCE) or not self.library_path.is_file() or
                self.receipt.get("library_sha256") !=
                _sha(self.library_path)):
            raise NativeScalarFCTInterfaceV98Error(
                "v98 scalar interface runtime identity differs")

    def _prepared(
        self, topology: object, mass: object, depth: object,
        next_depth: object, water_flux: object, left: object, right: object,
        boundary: object, hydro_constant: object, *,
        interface_faces: object, interface_concentration: object,
        dt_s: float, native_threads: int,
    ) -> tuple[dict[str, object], np.ndarray, np.ndarray, np.ndarray, str]:
        try:
            p = _prepare(
                topology, mass, depth, next_depth, water_flux, left, right,
                boundary, hydro_constant, dt_s=dt_s,
                native_threads=native_threads)
        except Exception as error:
            raise NativeScalarFCTInterfaceV98Error(
                "v98 scalar interface base inputs differ") from error
        selected, active, expanded = _interface_inputs(
            p, interface_faces, interface_concentration)
        return p, selected, active, expanded, _input_token(
            p, selected, active, expanded)

    def evaluate(
        self, topology: object, mass: object, depth: object,
        next_depth: object, water_flux: object, left: object, right: object,
        boundary: object, hydro_constant: object, *,
        interface_faces: object, interface_concentration: object,
        dt_s: float, native_threads: int = 4,
        cancelled: bool | Callable[[], bool] = False,
    ) -> dict[str, object]:
        self._validate_runtime()
        cancel = _cancel_check(cancelled)
        if cancel():
            raise InterruptedError(
                "v98 scalar interface cancelled before launch")
        p, selected, active, expanded, input_sha = self._prepared(
            topology, mass, depth, next_depth, water_flux, left, right,
            boundary, hydro_constant, interface_faces=interface_faces,
            interface_concentration=interface_concentration, dt_s=dt_s,
            native_threads=native_threads)
        result = np.empty((p["nl"], p["nc"]), dtype=np.float64)
        flux = np.empty((p["nl"], p["nf"]), dtype=np.float64)
        ledger = np.empty((p["nl"], p["nb"]), dtype=np.float64)
        interface_ledger = np.empty(
            (p["nl"], selected.size), dtype=np.float64)
        status = ctypes.c_int(-1)
        self._fct(
            p["nl"], p["nc"], p["nf"], p["nb"], p["nadj"], p["nbf"],
            selected.size, p["dt"], p["dry"], p["threads"], p["mass"],
            p["depth"], p["next_depth"], p["water"], p["left"],
            p["right"], p["boundary"], expanded, p["hydro"], p["owner"],
            p["neighbour"], p["boundary_index"], active, selected,
            p["length"], p["area"], p["offset"], p["faces"], p["signs"],
            p["width"], p["normal"], p["weno"], p["boundary_offset"],
            p["boundary_faces"], result, flux, ledger, interface_ledger,
            ctypes.byref(status))
        if status.value != 0:
            messages = {
                10: "closed face has nonzero accepted water flux",
                11: "internal-interface topology inventory differs",
                12: "internal-interface trace is invalid",
                20: "accepted water flux overdraws low-order scalar mass",
                30: "scalar FCT result is invalid",
            }
            raise NativeScalarFCTInterfaceV98Error(
                "v98 scalar interface " +
                messages.get(status.value, f"kernel status {status.value}"))
        if cancel():
            raise InterruptedError(
                "v98 scalar interface cancelled after launch")
        if any(not value.all() for value in (
                np.isfinite(result), np.isfinite(flux),
                np.isfinite(ledger), np.isfinite(interface_ledger))):
            raise NativeScalarFCTInterfaceV98Error(
                "v98 scalar interface returned nonfinite arrays")
        expected_interface = (
            -p["dt"]*p["length"][selected][None, :]*flux[:, selected])
        if not np.array_equal(interface_ledger, expected_interface):
            raise NativeScalarFCTInterfaceV98Error(
                "v98 scalar interface owner ledger differs")
        for value in (result, flux, ledger, interface_ledger, selected):
            value.setflags(write=False)
        output = {
            "mass_kg_m2": result,
            "advective_face_flux_kg_m_s": flux,
            "advective_boundary_ledger_kg": ledger,
            "internal_interface_owner_ledger_kg": interface_ledger,
            "internal_interface_faces": selected,
            "evaluation_count": 1,
            "internal_interface_owner_count": 1,
            "physical_boundary_surrogate_count": 0,
            "native_threads": p["threads"],
            "native_runtime_sha256": self.runtime_sha256,
            "result_sha256": _result_token(
                self.runtime_sha256, input_sha, p, result, flux, ledger,
                interface_ledger, selected),
        }
        return output

    def evaluate_arrays(self, *args, **kwargs) -> tuple[np.ndarray, ...]:
        result = self.evaluate(*args, **kwargs)
        return (result["mass_kg_m2"],
                result["advective_face_flux_kg_m_s"],
                result["advective_boundary_ledger_kg"],
                result["internal_interface_owner_ledger_kg"])

    def validate_result(self, result: Mapping[str, object], *args,
                        **kwargs) -> None:
        self._validate_runtime()
        if not isinstance(result, Mapping) or set(result) != RESULT_KEYS:
            raise NativeScalarFCTInterfaceV98Error(
                "v98 scalar interface result inventory differs")
        # The bounded auxiliary is deterministic.  Replay is deliberately used
        # here so reminted metadata cannot bypass numerical output validation.
        expected = self.evaluate(*args, **kwargs)
        for key in RESULT_KEYS:
            left, right = result[key], expected[key]
            if isinstance(right, np.ndarray):
                if not np.array_equal(np.asarray(left), right):
                    raise NativeScalarFCTInterfaceV98Error(
                        "v98 scalar interface result differs")
            elif left != right:
                raise NativeScalarFCTInterfaceV98Error(
                    "v98 scalar interface result differs")


def build_native_scalar_fct_interface_runtime_v98(
    build_directory: str | Path, *, compiler: str | Path | None = None,
) -> NativeScalarFCTInterfaceRuntimeV98:
    """Build the isolated v98 interface FCT and its frozen v96 dependency."""
    if not SOURCE.is_file() or not BASE_SOURCE.is_file():
        raise NativeScalarFCTInterfaceV98Error(
            "v98 scalar interface native source is missing")
    chosen = Path(compiler).resolve() if compiler is not None else None
    if chosen is None:
        located = shutil.which("gfortran")
        chosen = Path(located).resolve() if located else None
    if chosen is None and os.name == "nt":
        fallback = Path(r"C:\Strawberry\c\bin\gfortran.exe")
        chosen = fallback.resolve() if fallback.is_file() else None
    if chosen is None or not chosen.is_file():
        raise NativeScalarFCTInterfaceV98Error(
            "v98 scalar interface requires gfortran")
    flags = [
        "-O3", "-fopenmp", "-shared", "-ffree-line-length-none",
        "-fno-fast-math", "-ffp-contract=off", "-static-libgfortran",
        "-static-libgcc",
    ]
    source_sha, base_sha, compiler_sha = (
        _sha(SOURCE), _sha(BASE_SOURCE), _sha(chosen))
    key = _hash({"source": source_sha, "base": base_sha,
                 "compiler": compiler_sha, "flags": flags})[:16]
    build_root = Path(build_directory).resolve()
    build_root.mkdir(parents=True, exist_ok=True)
    build = Path(tempfile.mkdtemp(
        prefix=f"v98_scalar_interface_{key}_", dir=build_root))
    name = ("braids_scalar_fct_interface_v98.dll" if os.name == "nt" else
            "libbraids_scalar_fct_interface_v98.so")
    partial, library_path = build/(name+".partial"), build/name
    completed = subprocess.run(
        [str(chosen), *flags, str(BASE_SOURCE), str(SOURCE), "-o",
         str(partial)], cwd=build, capture_output=True, text=True,
        timeout=120)
    if completed.returncode != 0 or not partial.is_file():
        raise NativeScalarFCTInterfaceV98Error(
            "v98 scalar interface build failed:\n"+
            completed.stdout+completed.stderr)
    if source_sha != _sha(SOURCE) or base_sha != _sha(BASE_SOURCE):
        raise NativeScalarFCTInterfaceV98Error(
            "v98 scalar interface source changed during compilation")
    os.replace(partial, library_path)
    version = subprocess.run(
        [str(chosen), "--version"], capture_output=True, text=True,
        check=True, timeout=20).stdout.splitlines()[0]
    body = {
        "abi": ABI_ID,
        "native_source": str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
        "native_source_sha256": source_sha,
        "base_source": str(BASE_SOURCE.relative_to(ROOT)).replace("\\", "/"),
        "base_source_sha256": base_sha,
        "library_sha256": _sha(library_path),
        "compiler_identity": version,
        "compiler_sha256": compiler_sha,
        "compile_flags": flags,
        "openmp_parallel_face_cell_lane_loops": True,
        "cell_owned_deterministic_gathers": True,
        "internal_interface_owner_count": 1,
        "physical_boundary_surrogate_count": 0,
        "state_or_clock_owner_count": 0,
        "production_admission": False,
    }
    receipt = {**body, "receipt_sha256": _hash(body)}
    dependency = None
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        dependency = os.add_dll_directory(str(chosen.parent))
    return NativeScalarFCTInterfaceRuntimeV98(
        receipt, library_path, ctypes.CDLL(str(library_path)), dependency)


__all__ = [
    "ABI_ID", "RESULT_KEYS", "NativeScalarFCTInterfaceRuntimeV98",
    "NativeScalarFCTInterfaceV98Error",
    "build_native_scalar_fct_interface_runtime_v98",
]
