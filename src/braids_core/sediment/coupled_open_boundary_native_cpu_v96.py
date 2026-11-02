"""Native v96 augmented-f-wave residual with delegated open boundaries.

Interior and wall faces retain the frozen augmented GeoClaw f-wave call.  An
open face consumes the physical trace and already-corrected physical flux from
the sole hydraulic boundary owner.  The trace is deliberately never used as
a fabricated Riemann right state.  The optional exterior density force is the
locked FLOW KMAX=1, ICREEP=0 expression and uses a parent-frozen scalar-density
receipt plus the current nonlinear trace depth.

The manufactured trace-owner kind is useful for algebra tests only.  It is not
evidence that a production inlet/outlet mode has been qualified.
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

from .coupled_geometry_native_cpu_v96 import (
    DENSITY_WATER_PLUS_SEDIMENT,
    CoupledGeometryNativeCPUExecutionV96,
    execution_document_coupled_geometry_native_v96,
)
from .densin_fwave_v85 import LOCKED_UPSTREAM_COMMIT, verify_locked_sources_v85
from .hydraulic_boundary_trace_cpu_v96 import (
    KIND_CLOSED,
    KIND_INTERNAL,
    KIND_OPEN,
    HydraulicBoundaryTraceExecutionV96,
    HydraulicBoundaryTraceReceiptV96,
    execution_document_hydraulic_boundary_trace_v96,
    validate_hydraulic_boundary_trace_receipt_v96,
)


SCHEMA = 1
SCOPE_ID = "braids-q4-coupled-open-boundary-native-cpu-v96-v1"
SEMANTIC_ID = SCOPE_ID + ":delegated-physical-flux-exterior-densin"
NATIVE_ABI_ID = SCOPE_ID + ":openmp-full-rate-abi-v1"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ROOT = Path(__file__).resolve().parents[2]

_WRAPPER_RELATIVE = "solver/wave/braids_coupled_open_boundary_v96_module.f90"
_V95_RELATIVE = "solver/wave/braids_densin_residual_v95_module.f90"
_PARAMETER_RELATIVE = "solver/cut_cells/geoclaw_parameters.f90"
_RPN2_RELATIVE = "solver/clawpack-v5.14.0/riemann/src/rpn2_geoclaw.f"
_UTILS_RELATIVE = "solver/clawpack-v5.14.0/riemann/src/geoclaw_riemann_utils.f"
_EXPECTED_SOURCE_HASHES = {
    _WRAPPER_RELATIVE:
        "2341a9899eb6da52455cc54304aa95f41ecb9cf7bd0931b1b0d7dcd9d6c98d35",
    _V95_RELATIVE:
        "e701c096e400402c4386cc2015459d5900ce073094b86f97c0ad94c2516a61e3",
    _PARAMETER_RELATIVE:
        "b6e5a70595e93279eac113401dfc2d857978f7a6e885fe730c783c17cc5b3c1d",
    _RPN2_RELATIVE:
        "c52649def884ce374dbb1d3dc1e91f7a9c33aa45a0aff691badc8811b9157d1f",
    _UTILS_RELATIVE:
        "df73207446ac730f9cda9a8ca4dbbc625151ba95d13aa514ce87ca23cb598761",
}
_GLOBAL_GEOCLAW_CALL_LOCK = threading.Lock()


class CoupledOpenBoundaryNativeV96Error(ValueError):
    """The open-boundary native execution, receipt, or result is invalid."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError, OverflowError) as error:
        raise CoupledOpenBoundaryNativeV96Error(
            "v96 open-boundary value is not finite canonical JSON") from error


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
        raise CoupledOpenBoundaryNativeV96Error(
            f"v96 open-boundary {name} must be a lowercase SHA-256")
    return value


def _array(value: object, shape: tuple[int, ...], name: str, *,
           positive: bool = False) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != shape or raw.dtype.kind not in "fiub":
        raise CoupledOpenBoundaryNativeV96Error(
            f"v96 open-boundary {name} must be numeric with shape {shape}")
    result = np.ascontiguousarray(raw, dtype=np.float64)
    if (not np.isfinite(result).all() or
            (positive and np.any(result <= 0.0))):
        raise CoupledOpenBoundaryNativeV96Error(
            f"v96 open-boundary {name} contains invalid values")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class NativeCoupledOpenBoundaryBuildReceiptV96:
    abi_id: str
    wrapper_source_sha256: str
    v95_kernel_source_sha256: str
    geoclaw_parameters_sha256: str
    augmented_rpn2_sha256: str
    riemann_utils_sha256: str
    library_sha256: str
    compiler_identity: str
    compiler_sha256: str
    compile_flags: tuple[str, ...]
    build_utc: str
    locked_delft_commit: str
    locked_delft_source_sha256: tuple[tuple[str, str], ...]
    receipt_sha256: str


def _build_payload(value: NativeCoupledOpenBoundaryBuildReceiptV96) -> dict:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID + ":native-build",
        "abi_id": value.abi_id,
        "source_sha256": {
            _WRAPPER_RELATIVE: value.wrapper_source_sha256,
            _V95_RELATIVE: value.v95_kernel_source_sha256,
            _PARAMETER_RELATIVE: value.geoclaw_parameters_sha256,
            _RPN2_RELATIVE: value.augmented_rpn2_sha256,
            _UTILS_RELATIVE: value.riemann_utils_sha256,
        },
        "library_sha256": value.library_sha256,
        "compiler_identity": value.compiler_identity,
        "compiler_sha256": value.compiler_sha256,
        "compile_flags": list(value.compile_flags),
        "openmp": "per-region-num_threads-no-nested-team",
        "geoclaw_global_call_lock": True,
        "locked_delft_commit": value.locked_delft_commit,
        "locked_delft_source_sha256": dict(
            value.locked_delft_source_sha256),
        "frozen_v95_and_augmented_riemann_sources_unchanged": True,
    }


def validate_native_coupled_open_boundary_build_receipt_v96(
    value: NativeCoupledOpenBoundaryBuildReceiptV96,
) -> None:
    if type(value) is not NativeCoupledOpenBoundaryBuildReceiptV96:
        raise CoupledOpenBoundaryNativeV96Error(
            "v96 open-boundary build receipt type differs")
    locked = tuple(sorted(verify_locked_sources_v85(_ROOT).items()))
    actual = {
        _WRAPPER_RELATIVE: value.wrapper_source_sha256,
        _V95_RELATIVE: value.v95_kernel_source_sha256,
        _PARAMETER_RELATIVE: value.geoclaw_parameters_sha256,
        _RPN2_RELATIVE: value.augmented_rpn2_sha256,
        _UTILS_RELATIVE: value.riemann_utils_sha256,
    }
    hashes = tuple(actual.values())+(value.library_sha256,
                                     value.compiler_sha256)
    if (any(_SHA256.fullmatch(item) is None for item in hashes) or
            value.abi_id != NATIVE_ABI_ID or
            actual != _EXPECTED_SOURCE_HASHES or
            not value.compiler_identity or not value.compile_flags or
            not value.build_utc.endswith("Z") or
            value.locked_delft_commit != LOCKED_UPSTREAM_COMMIT or
            value.locked_delft_source_sha256 != locked or
            value.receipt_sha256 != _fingerprint(_build_payload(value))):
        raise CoupledOpenBoundaryNativeV96Error(
            "v96 open-boundary build receipt is stale or mismatched")


@dataclass(frozen=True)
class AcceptedBoundaryDensityReceiptV96:
    parent_state_sha256: str
    accepted_boundary_scalar_state_sha256: str
    source_trial_sha256: str
    accepted_update: int
    time_s: float
    water_density_kg_m3: np.ndarray
    mixture_density_kg_m3: np.ndarray
    open_mask_sha256: str
    receipt_sha256: str


def _boundary_density_token(value: AcceptedBoundaryDensityReceiptV96) -> str:
    return _fingerprint({
        "scope_id": SCOPE_ID + ":accepted-boundary-density",
        "parent_state_sha256": value.parent_state_sha256,
        "accepted_boundary_scalar_state_sha256":
            value.accepted_boundary_scalar_state_sha256,
        "source_trial_sha256": value.source_trial_sha256,
        "accepted_update": value.accepted_update,
        "time_s": value.time_s,
        "density_sha256": _digest(value.water_density_kg_m3,
                                  value.mixture_density_kg_m3),
        "open_mask_sha256": value.open_mask_sha256,
        "timing": "accepted-parent-scalars-frozen-before-picard",
    })


def make_accepted_boundary_density_receipt_v96(
    execution: HydraulicBoundaryTraceExecutionV96, *,
    parent_state_sha256: str,
    accepted_boundary_scalar_state_sha256: str,
    source_trial_sha256: str,
    accepted_update: int,
    time_s: float,
    water_density_kg_m3: object,
    mixture_density_kg_m3: object,
) -> AcceptedBoundaryDensityReceiptV96:
    execution_document_hydraulic_boundary_trace_v96(execution)
    if type(accepted_update) is not int or accepted_update < 0:
        raise CoupledOpenBoundaryNativeV96Error(
            "v96 boundary-density update is invalid")
    time = float(time_s)
    if not math.isfinite(time) or time < 0.0:
        raise CoupledOpenBoundaryNativeV96Error(
            "v96 boundary-density time is invalid")
    nf = execution.face_count
    water = _array(water_density_kg_m3, (nf,), "boundary water density")
    mixture = _array(
        mixture_density_kg_m3, (nf,), "boundary mixture density")
    open_mask = execution.open_mask
    if (np.any(water[open_mask] <= 0.0) or
            np.any(mixture[open_mask] < water[open_mask]) or
            np.any(water[~open_mask] != 0.0) or
            np.any(mixture[~open_mask] != 0.0)):
        raise CoupledOpenBoundaryNativeV96Error(
            "v96 accepted boundary densities differ from the open mask")
    placeholder = AcceptedBoundaryDensityReceiptV96(
        _sha(parent_state_sha256, "parent state"),
        _sha(accepted_boundary_scalar_state_sha256,
             "accepted boundary scalar state"),
        _sha(source_trial_sha256, "source trial"), accepted_update, time,
        water, mixture, _digest(open_mask.astype(np.int32)), "")
    return AcceptedBoundaryDensityReceiptV96(
        **{**placeholder.__dict__,
           "receipt_sha256": _boundary_density_token(placeholder)})


def validate_accepted_boundary_density_receipt_v96(
    value: AcceptedBoundaryDensityReceiptV96,
    execution: HydraulicBoundaryTraceExecutionV96,
) -> None:
    execution_document_hydraulic_boundary_trace_v96(execution)
    if type(value) is not AcceptedBoundaryDensityReceiptV96:
        raise CoupledOpenBoundaryNativeV96Error(
            "v96 boundary-density receipt type differs")
    nf = execution.face_count
    water = _array(value.water_density_kg_m3, (nf,), "receipt water density")
    mixture = _array(
        value.mixture_density_kg_m3, (nf,), "receipt mixture density")
    open_mask = execution.open_mask
    if (type(value.accepted_update) is not int or value.accepted_update < 0 or
            not math.isfinite(value.time_s) or value.time_s < 0.0 or
            value.open_mask_sha256 != _digest(open_mask.astype(np.int32)) or
            np.any(water[open_mask] <= 0.0) or
            np.any(mixture[open_mask] < water[open_mask]) or
            np.any(water[~open_mask] != 0.0) or
            np.any(mixture[~open_mask] != 0.0) or
            value.receipt_sha256 != _boundary_density_token(value)):
        raise CoupledOpenBoundaryNativeV96Error(
            "v96 accepted boundary-density receipt is stale or mismatched")


@dataclass(frozen=True)
class CoupledOpenBoundaryNativeCPUExecutionV96:
    base_execution: CoupledGeometryNativeCPUExecutionV96
    boundary_execution: HydraulicBoundaryTraceExecutionV96
    native_build_receipt: NativeCoupledOpenBoundaryBuildReceiptV96
    ibaroc_open_boundary: bool
    semantic_id: str
    fingerprint: str

    @property
    def cell_count(self) -> int:
        return self.base_execution.cell_count

    @property
    def face_count(self) -> int:
        return self.base_execution.face_count


def _execution_payload(value: CoupledOpenBoundaryNativeCPUExecutionV96) -> dict:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "semantic_id": value.semantic_id,
        "backend": "native_cpu_openmp",
        "base_geometry_execution":
            execution_document_coupled_geometry_native_v96(
                value.base_execution),
        "boundary_trace_execution":
            execution_document_hydraulic_boundary_trace_v96(
                value.boundary_execution),
        "native_build_receipt_sha256":
            value.native_build_receipt.receipt_sha256,
        "ibaroc_open_boundary": value.ibaroc_open_boundary,
        "open_mask": "(neighbour<0)&(boundary_index>=0)",
        "open_hydraulic_flux": "delegated-physical-flux-no-riemann-rerun",
        "open_density": "FLOW-KMAX1-ICREEP0-parent-frozen-scalars",
        "interior_density": "always-active-when-base-densin-enabled",
        "manufactured_delegate": "algebra-only-not-mode-qualification",
        "production_admission": False,
    }


def resolve_coupled_open_boundary_native_cpu_execution_v96(
    base_execution: CoupledGeometryNativeCPUExecutionV96,
    boundary_execution: HydraulicBoundaryTraceExecutionV96,
    native_build_receipt: NativeCoupledOpenBoundaryBuildReceiptV96, *,
    ibaroc_open_boundary: bool = True,
) -> CoupledOpenBoundaryNativeCPUExecutionV96:
    execution_document_coupled_geometry_native_v96(base_execution)
    execution_document_hydraulic_boundary_trace_v96(boundary_execution)
    validate_native_coupled_open_boundary_build_receipt_v96(
        native_build_receipt)
    if type(ibaroc_open_boundary) is not bool:
        raise CoupledOpenBoundaryNativeV96Error(
            "v96 IBaroc selection must be Boolean")
    topology = base_execution.base_execution
    if (boundary_execution.cell_count != base_execution.cell_count or
            boundary_execution.face_count != base_execution.face_count or
            not np.array_equal(boundary_execution.owner, topology.owner) or
            not np.array_equal(boundary_execution.neighbour,
                               topology.neighbour) or
            not np.array_equal(boundary_execution.face_distance_m,
                               topology.face_distance_m)):
        raise CoupledOpenBoundaryNativeV96Error(
            "v96 open-boundary and hydraulic topologies differ")
    placeholder = CoupledOpenBoundaryNativeCPUExecutionV96(
        base_execution, boundary_execution, native_build_receipt,
        ibaroc_open_boundary, SEMANTIC_ID, "")
    return CoupledOpenBoundaryNativeCPUExecutionV96(
        **{**placeholder.__dict__,
           "fingerprint": _fingerprint(_execution_payload(placeholder))})


def execution_document_coupled_open_boundary_native_v96(
    value: CoupledOpenBoundaryNativeCPUExecutionV96,
) -> dict:
    if type(value) is not CoupledOpenBoundaryNativeCPUExecutionV96:
        raise CoupledOpenBoundaryNativeV96Error(
            "v96 open-boundary execution type differs")
    validate_native_coupled_open_boundary_build_receipt_v96(
        value.native_build_receipt)
    payload = _execution_payload(value)
    if (value.semantic_id != SEMANTIC_ID or
            type(value.ibaroc_open_boundary) is not bool or
            value.fingerprint != _fingerprint(payload)):
        raise CoupledOpenBoundaryNativeV96Error(
            "v96 open-boundary execution fingerprint differs")
    return {**payload, "fingerprint": value.fingerprint}


@dataclass(frozen=True)
class NativeCoupledOpenBoundaryRateResultV96:
    base_fwave_rate: np.ndarray
    density_momentum_rate: np.ndarray
    full_rate: np.ndarray
    face_normal_acceleration_m_s2: np.ndarray
    cell_acceleration_m_s2: np.ndarray
    face_water_flux_m2_s: np.ndarray
    accepted_bed_sha256: str
    accepted_density_sha256: str
    boundary_density_receipt_sha256: str
    boundary_trace_receipt_sha256: str
    augmented_riemann_call_count: int
    density_source_evaluation_count: int
    delegated_open_flux_consumption_count: int
    open_density_face_evaluation_count: int


class NativeCoupledOpenBoundaryRuntimeV96:
    __slots__ = ("receipt", "library_path", "_function", "_library",
                 "_dependency")

    def __init__(self, receipt: NativeCoupledOpenBoundaryBuildReceiptV96,
                 library_path: Path, function, library, dependency) -> None:
        validate_native_coupled_open_boundary_build_receipt_v96(receipt)
        self.receipt = receipt
        self.library_path = Path(library_path).resolve()
        self._function = function
        self._library = library
        self._dependency = dependency

    def _validate_runtime(self) -> None:
        validate_native_coupled_open_boundary_build_receipt_v96(self.receipt)
        if (not self.library_path.is_file() or
                _file_sha256(self.library_path) !=
                    self.receipt.library_sha256 or
                _file_sha256(_ROOT/_WRAPPER_RELATIVE) !=
                    self.receipt.wrapper_source_sha256):
            raise CoupledOpenBoundaryNativeV96Error(
                "v96 open-boundary loaded source/library identity differs")

    def evaluate(
        self, execution: CoupledOpenBoundaryNativeCPUExecutionV96,
        midpoint_q: object, *, bed_elevation_m: object,
        accepted_water_density_kg_m3: object,
        accepted_mixture_density_kg_m3: object,
        boundary_density: AcceptedBoundaryDensityReceiptV96,
        boundary_trace: HydraulicBoundaryTraceReceiptV96,
    ) -> NativeCoupledOpenBoundaryRateResultV96:
        execution_document_coupled_open_boundary_native_v96(execution)
        self._validate_runtime()
        if self.receipt != execution.native_build_receipt:
            raise CoupledOpenBoundaryNativeV96Error(
                "v96 open-boundary runtime receipt differs")
        boundary_execution = execution.boundary_execution
        validate_accepted_boundary_density_receipt_v96(
            boundary_density, boundary_execution)
        base = execution.base_execution.base_execution
        nc, nf = execution.cell_count, execution.face_count
        q = np.ascontiguousarray(_array(
            midpoint_q, (3, nc), "midpoint state"))
        bed = np.ascontiguousarray(_array(
            bed_elevation_m, (nc,), "accepted bed"))
        water = np.ascontiguousarray(_array(
            accepted_water_density_kg_m3, (nc,), "accepted water density",
            positive=True))
        mixture = np.ascontiguousarray(_array(
            accepted_mixture_density_kg_m3, (nc,),
            "accepted mixture density", positive=True))
        if np.any(mixture < water):
            raise CoupledOpenBoundaryNativeV96Error(
                "v96 open-boundary mixture density is below water density")
        validate_hydraulic_boundary_trace_receipt_v96(
            boundary_trace, boundary_execution, midpoint_q=q,
            bed_elevation_m=bed,
            hydraulic_request_sha256=
                boundary_trace.hydraulic_request_sha256,
            parent_state_sha256=boundary_trace.parent_state_sha256,
            evaluation_index=boundary_trace.evaluation_index,
            evaluation_time_s=boundary_trace.evaluation_time_s)
        density = np.ascontiguousarray(
            mixture if execution.base_execution.density_mode ==
            DENSITY_WATER_PLUS_SEDIMENT else water)
        exterior = np.ascontiguousarray(
            boundary_density.mixture_density_kg_m3 if
            execution.base_execution.density_mode ==
            DENSITY_WATER_PLUS_SEDIMENT else
            boundary_density.water_density_kg_m3)
        physical_open = np.asarray(
            boundary_trace.physical_boundary_flux_owned_mask, dtype=bool)
        exterior_densin = np.ascontiguousarray(
            boundary_trace.exterior_densin_active_mask, dtype=np.int32)
        kind = np.where(
            boundary_execution.neighbour >= 0, KIND_INTERNAL,
            np.where(physical_open, KIND_OPEN, KIND_CLOSED)).astype(np.int32)
        integers = [np.ascontiguousarray(item, dtype=np.int32) for item in (
            base.owner, base.neighbour, kind, exterior_densin,
            base.cell_face_offset,
            base.cell_faces, base.cell_face_sign)]
        outputs = (
            np.empty((3, nc), dtype=np.float64),
            np.empty((2, nc), dtype=np.float64),
            np.empty((3, nc), dtype=np.float64),
            np.empty(nf, dtype=np.float64),
            np.empty((2, nc), dtype=np.float64),
            np.empty(nf, dtype=np.float64),
        )
        counts = [ctypes.c_int(-1) for _ in range(5)]
        with _GLOBAL_GEOCLAW_CALL_LOCK:
            self._function(
                nc, nf, integers[5].size, q, bed,
                integers[0], integers[1], integers[2], integers[3],
                np.ascontiguousarray(base.normal),
                np.ascontiguousarray(base.face_length_m),
                np.ascontiguousarray(base.face_distance_m),
                np.ascontiguousarray(base.cell_area_m2),
                integers[4], integers[5], integers[6], density, exterior,
                np.ascontiguousarray(boundary_trace.trace_q),
                np.ascontiguousarray(
                    boundary_trace.physical_open_face_flux),
                base.hydraulic_reference_density_kg_m3,
                base.gravity_m_s2, base.dry_tolerance_m, 1,
                int(execution.ibaroc_open_boundary),
                execution.base_execution.native_threads,
                *outputs, *(ctypes.byref(item) for item in counts))
        status = counts[-1].value
        if status != 0:
            raise CoupledOpenBoundaryNativeV96Error(
                f"v96 open-boundary native full-rate failed with status {status}")
        open_count = int(np.count_nonzero(physical_open))
        densin_count = int(np.count_nonzero(exterior_densin))
        expected = (nf-open_count, 1, open_count,
                    densin_count if execution.ibaroc_open_boundary else 0)
        actual = tuple(item.value for item in counts[:-1])
        if actual != expected or not all(np.isfinite(item).all()
                                         for item in outputs):
            raise CoupledOpenBoundaryNativeV96Error(
                "v96 open-boundary native counts or arrays differ")
        for item in outputs:
            item.setflags(write=False)
        return NativeCoupledOpenBoundaryRateResultV96(
            *outputs, _digest(bed), _digest(density),
            boundary_density.receipt_sha256,
            boundary_trace.receipt_sha256, *actual)


def build_native_coupled_open_boundary_runtime_v96(
    repository_root: str | Path, build_directory: str | Path, *,
    compiler: str | Path | None = None,
) -> NativeCoupledOpenBoundaryRuntimeV96:
    root = Path(repository_root).resolve()
    build_root = Path(build_directory).resolve()
    build_root.mkdir(parents=True, exist_ok=True)
    relatives = (_PARAMETER_RELATIVE, _RPN2_RELATIVE, _UTILS_RELATIVE,
                 _V95_RELATIVE, _WRAPPER_RELATIVE)
    sources = [root/item for item in relatives]
    if not all(path.is_file() for path in sources):
        raise CoupledOpenBoundaryNativeV96Error(
            "v96 open-boundary source inventory is incomplete")
    hashes = {item: _file_sha256(path)
              for item, path in zip(relatives, sources)}
    if hashes != {item: _EXPECTED_SOURCE_HASHES[item] for item in relatives}:
        raise CoupledOpenBoundaryNativeV96Error(
            "v96 open-boundary native or frozen f-wave source differs")
    locked = tuple(sorted(verify_locked_sources_v85(root).items()))
    selected = Path(compiler) if compiler is not None else None
    if selected is None:
        discovered = shutil.which("gfortran")
        if discovered is None and os.name == "nt":
            fallback = Path(r"C:\Strawberry\c\bin\gfortran.exe")
            discovered = str(fallback) if fallback.is_file() else None
        if discovered is None:
            raise CoupledOpenBoundaryNativeV96Error(
                "v96 open-boundary native runtime requires gfortran")
        selected = Path(discovered)
    selected = selected.resolve()
    if not selected.is_file():
        raise CoupledOpenBoundaryNativeV96Error(
            "v96 open-boundary compiler is unavailable")
    flags = ("-O3", "-flto", "-fopenmp", "-shared",
             "-ffree-line-length-none", "-ffixed-line-length-none",
             "-static-libgfortran", "-static-libgcc")
    key = _fingerprint({
        "sources": hashes, "compiler": _file_sha256(selected),
        "flags": list(flags), "abi": NATIVE_ABI_ID,
    })[:16]
    build = Path(tempfile.mkdtemp(
        prefix=f"v96_open_{key}_", dir=build_root))
    name = ("braids_coupled_open_boundary_v96.dll" if os.name == "nt"
            else "libbraids_coupled_open_boundary_v96.so")
    partial, library_path = build/(name+".partial"), build/name
    completed = subprocess.run(
        [str(selected), *flags, *(str(path) for path in sources),
         "-o", str(partial)], cwd=build, capture_output=True, text=True,
        timeout=180, check=False)
    if completed.returncode != 0 or not partial.is_file():
        raise CoupledOpenBoundaryNativeV96Error(
            "v96 open-boundary native build failed:\n"+
            completed.stdout+completed.stderr)
    if hashes != {item: _file_sha256(root/item) for item in relatives}:
        raise CoupledOpenBoundaryNativeV96Error(
            "v96 open-boundary source changed during compilation")
    os.replace(partial, library_path)
    version = subprocess.run(
        [str(selected), "--version"], capture_output=True, text=True,
        check=True, timeout=30).stdout.splitlines()[0]
    placeholder = NativeCoupledOpenBoundaryBuildReceiptV96(
        NATIVE_ABI_ID, hashes[_WRAPPER_RELATIVE], hashes[_V95_RELATIVE],
        hashes[_PARAMETER_RELATIVE], hashes[_RPN2_RELATIVE],
        hashes[_UTILS_RELATIVE], _file_sha256(library_path), version,
        _file_sha256(selected), flags,
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
            "+00:00", "Z"), LOCKED_UPSTREAM_COMMIT, locked, "")
    receipt = NativeCoupledOpenBoundaryBuildReceiptV96(
        **{**placeholder.__dict__,
           "receipt_sha256": _fingerprint(_build_payload(placeholder))})
    dependency = None
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        dependency = os.add_dll_directory(str(selected.parent))
    library = ctypes.CDLL(str(library_path))
    function = library.braids_coupled_open_full_rate_v96
    fa = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    ia = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
    ip = ctypes.POINTER(ctypes.c_int)
    function.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        fa, fa, ia, ia, ia, ia, fa, fa, fa, fa, ia, ia, ia,
        fa, fa, fa, fa, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        fa, fa, fa, fa, fa, fa, ip, ip, ip, ip, ip,
    ]
    function.restype = None
    return NativeCoupledOpenBoundaryRuntimeV96(
        receipt, library_path, function, library, dependency)


def method_identity_coupled_open_boundary_native_v96() -> dict[str, object]:
    return {
        "scope_id": SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "backend": "native_cpu_openmp",
        "interior_closed_faces": "frozen_augmented_geoclaw_fwave",
        "open_faces": "same-owner-physical-trace-and-corrected-flux",
        "open_trace_is_riemann_right_state": False,
        "open_density": "FLOW-KMAX1-ICREEP0",
        "ibaroc": "global-default-true-open-density-only",
        "boundary_scalar_timing": "accepted-parent-frozen-before-picard",
        "manufactured_provider_qualification": "algebra-only",
        "production_admission": False,
    }


__all__ = [
    "AcceptedBoundaryDensityReceiptV96",
    "CoupledOpenBoundaryNativeCPUExecutionV96",
    "CoupledOpenBoundaryNativeV96Error",
    "NativeCoupledOpenBoundaryBuildReceiptV96",
    "NativeCoupledOpenBoundaryRateResultV96",
    "NativeCoupledOpenBoundaryRuntimeV96",
    "build_native_coupled_open_boundary_runtime_v96",
    "execution_document_coupled_open_boundary_native_v96",
    "make_accepted_boundary_density_receipt_v96",
    "method_identity_coupled_open_boundary_native_v96",
    "resolve_coupled_open_boundary_native_cpu_execution_v96",
    "validate_accepted_boundary_density_receipt_v96",
    "validate_native_coupled_open_boundary_build_receipt_v96",
]
