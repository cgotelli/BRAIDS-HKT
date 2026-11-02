"""Additive native-CPU moving-wet/dry augmented-f-wave owner for Q4/v97.

The historical fully-wet v95/v96 identities are not changed.  This module
wraps the same frozen GeoClaw augmented face routine with an explicit outer
wet active set, the established donor-draining limiter, positive-depth
fraction-to-boundary damping, and true-residual reevaluation.  It is a
bounded closed/fixed-grid CPU route; open boundaries, scalar dry transfer and
the full v94b atomic transaction are composed by later v97 owners.
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

from .coupled_geometry_native_cpu_v96 import (
    CoupledGeometryNativeCPUExecutionV96,
    DENSITY_WATER_PLUS_SEDIMENT,
    execution_document_coupled_geometry_native_v96,
)
from .density_active_projection_v97 import (
    POLICY as DENSITY_PROJECTION_POLICY_V97,
    NativeDensityActiveProjectionRuntimeV97,
    resolve_density_projection_geometry_v97,
    validate_projection_result_v97,
)
from .densin_fwave_v85 import LOCKED_UPSTREAM_COMMIT, verify_locked_sources_v85


SCHEMA = 1
SCOPE_ID = "braids-q4-moving-wetdry-native-cpu-v97-v1"
SEMANTIC_ID = SCOPE_ID + ":active-set-limited-augmented-fwave"
NATIVE_ABI_ID = SCOPE_ID + ":openmp-face-cell-abi-v1"
MAX_ACTIVE_SET_RESTARTS = 6
MAX_INNER_ITERATIONS = 80
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_RELATIVE = "solver/wave/braids_wetdry_fwave_v97_module.f90"
_V95_RELATIVE = "solver/wave/braids_densin_residual_v95_module.f90"
_PARAMETER_RELATIVE = "solver/cut_cells/geoclaw_parameters.f90"
_RPN2_RELATIVE = "solver/clawpack-v5.14.0/riemann/src/rpn2_geoclaw.f"
_UTILS_RELATIVE = "solver/clawpack-v5.14.0/riemann/src/geoclaw_riemann_utils.f"
_EXPECTED_FROZEN = {
    _V95_RELATIVE:
        "e701c096e400402c4386cc2015459d5900ce073094b86f97c0ad94c2516a61e3",
    _PARAMETER_RELATIVE:
        "b6e5a70595e93279eac113401dfc2d857978f7a6e885fe730c783c17cc5b3c1d",
    _RPN2_RELATIVE:
        "c52649def884ce374dbb1d3dc1e91f7a9c33aa45a0aff691badc8811b9157d1f",
    _UTILS_RELATIVE:
        "df73207446ac730f9cda9a8ca4dbbc625151ba95d13aa514ce87ca23cb598761",
}
_GE0CLAW_CALL_LOCK = threading.Lock()


class WetDryFwaveNativeV97Error(ValueError):
    """The bounded v97 execution, runtime or active-set solve is invalid."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError, OverflowError) as error:
        raise WetDryFwaveNativeV97Error(
            "v97 wet/dry identity is not finite canonical JSON") from error


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _digest(*values: np.ndarray) -> str:
    result = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(value)
        result.update(array.dtype.str.encode("ascii"))
        result.update(_canonical(list(array.shape)))
        result.update(array.tobytes(order="C"))
    return result.hexdigest()


def _array(value: object, shape: tuple[int, ...], name: str, *,
           integer: bool = False) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != shape or raw.dtype.kind not in "fiub" or not np.isfinite(raw).all():
        raise WetDryFwaveNativeV97Error(
            f"v97 wet/dry {name} shape/type/finite identity differs")
    if integer and (np.any(raw != np.floor(raw)) or
                    np.any(np.abs(raw) > np.iinfo(np.int32).max)):
        raise WetDryFwaveNativeV97Error(
            f"v97 wet/dry {name} must be int32-valued")
    result = np.array(raw, dtype=np.int32 if integer else np.float64,
                      order="C", copy=True)
    result.setflags(write=False)
    return result


def _cancel(cancelled: bool | Callable[[], bool]) -> Callable[[], bool]:
    if type(cancelled) is bool:
        return lambda: cancelled
    if callable(cancelled):
        return cancelled
    raise WetDryFwaveNativeV97Error(
        "v97 wet/dry cancellation must be bool or callable")


@dataclass(frozen=True)
class NativeWetDryFwaveBuildReceiptV97:
    abi_id: str
    source_sha256: str
    wrapper_sha256: str
    frozen_v95_sha256: str
    geoclaw_parameters_sha256: str
    augmented_rpn2_sha256: str
    riemann_utils_sha256: str
    library_sha256: str
    compiler_identity: str
    compile_flags: tuple[str, ...]
    build_utc: str
    locked_delft_commit: str
    locked_delft_source_sha256: tuple[tuple[str, str], ...]
    receipt_sha256: str


def _build_payload(value: NativeWetDryFwaveBuildReceiptV97) -> dict:
    return {
        "schema": SCHEMA, "scope_id": SCOPE_ID + ":native-build",
        "abi_id": value.abi_id, "source_sha256": value.source_sha256,
        "wrapper_sha256": value.wrapper_sha256,
        "frozen_v95_sha256": value.frozen_v95_sha256,
        "geoclaw_parameters_sha256": value.geoclaw_parameters_sha256,
        "augmented_rpn2_sha256": value.augmented_rpn2_sha256,
        "riemann_utils_sha256": value.riemann_utils_sha256,
        "library_sha256": value.library_sha256,
        "compiler_identity": value.compiler_identity,
        "compile_flags": list(value.compile_flags),
        "build_utc": value.build_utc,
        "locked_delft_commit": value.locked_delft_commit,
        "locked_delft_source_sha256": dict(value.locked_delft_source_sha256),
        "openmp_face_cell_loops": True,
        "frozen_augmented_fwave_unchanged": True,
    }


def validate_native_wetdry_fwave_build_receipt_v97(
    value: NativeWetDryFwaveBuildReceiptV97,
) -> None:
    if type(value) is not NativeWetDryFwaveBuildReceiptV97:
        raise WetDryFwaveNativeV97Error("v97 native build receipt type differs")
    current = {
        _SOURCE_RELATIVE: _sha(_ROOT/_SOURCE_RELATIVE),
        _V95_RELATIVE: _sha(_ROOT/_V95_RELATIVE),
        _PARAMETER_RELATIVE: _sha(_ROOT/_PARAMETER_RELATIVE),
        _RPN2_RELATIVE: _sha(_ROOT/_RPN2_RELATIVE),
        _UTILS_RELATIVE: _sha(_ROOT/_UTILS_RELATIVE),
    }
    locked = tuple(sorted(verify_locked_sources_v85(_ROOT).items()))
    expected = (current[_SOURCE_RELATIVE], _sha(Path(__file__)),
                _EXPECTED_FROZEN[_V95_RELATIVE],
                _EXPECTED_FROZEN[_PARAMETER_RELATIVE],
                _EXPECTED_FROZEN[_RPN2_RELATIVE],
                _EXPECTED_FROZEN[_UTILS_RELATIVE])
    actual = (value.source_sha256, value.wrapper_sha256,
              value.frozen_v95_sha256, value.geoclaw_parameters_sha256,
              value.augmented_rpn2_sha256, value.riemann_utils_sha256)
    if (current[_V95_RELATIVE] != _EXPECTED_FROZEN[_V95_RELATIVE] or
            current[_PARAMETER_RELATIVE] != _EXPECTED_FROZEN[_PARAMETER_RELATIVE] or
            current[_RPN2_RELATIVE] != _EXPECTED_FROZEN[_RPN2_RELATIVE] or
            current[_UTILS_RELATIVE] != _EXPECTED_FROZEN[_UTILS_RELATIVE] or
            actual != expected or value.abi_id != NATIVE_ABI_ID or
            _SHA.fullmatch(value.library_sha256) is None or
            not value.compiler_identity or not value.compile_flags or
            not value.build_utc.endswith("Z") or
            value.locked_delft_commit != LOCKED_UPSTREAM_COMMIT or
            value.locked_delft_source_sha256 != locked or
            value.receipt_sha256 != _fingerprint(_build_payload(value))):
        raise WetDryFwaveNativeV97Error(
            "v97 native build/source receipt is stale or mismatched")


@dataclass(frozen=True)
class WetDryFwaveCPUExecutionV97:
    base_execution: CoupledGeometryNativeCPUExecutionV96
    native_build_receipt: NativeWetDryFwaveBuildReceiptV97
    domain_active_mask: np.ndarray
    native_threads: int
    max_active_set_restarts: int
    max_inner_iterations: int
    nonlinear_tolerance: float
    positivity_fraction: float
    fingerprint: str

    @property
    def cell_count(self) -> int:
        return self.base_execution.cell_count

    @property
    def face_count(self) -> int:
        return self.base_execution.face_count


def _execution_payload(value: WetDryFwaveCPUExecutionV97) -> dict:
    return {
        "schema": SCHEMA, "scope_id": SCOPE_ID,
        "semantic_id": SEMANTIC_ID, "backend": "native_cpu_openmp",
        "base_v96_execution_fingerprint": value.base_execution.fingerprint,
        "native_build_receipt_sha256": value.native_build_receipt.receipt_sha256,
        "domain_active_sha256": _digest(value.domain_active_mask),
        "native_threads": value.native_threads,
        "max_active_set_restarts": value.max_active_set_restarts,
        "max_inner_iterations": value.max_inner_iterations,
        "nonlinear_tolerance": value.nonlinear_tolerance,
        "positivity_fraction": value.positivity_fraction,
        "wet_mask": "domain_active_and_depth_strictly_above_dry_tolerance",
        "inner_mask_policy":
            "monotone_wetting_restart_drying_accepts_physical_ledger",
        "endpoint_dry_momentum_policy":
            "authenticated_zero_projection_all_accepted_dry_cells",
        "implicit_residual_audit": "preprojection_true_residual",
        "projected_endpoint_identity_audit": "separate_exact_identity",
        "spatial_flux": "unchanged_augmented_geoclaw_rpn2",
        "draining": "donor_limited_full_face_flux",
        "nonlinear_solver": "damped_picard_true_residual",
        "approximate_jacobian": "identity_dry_rows_picard_wet_rows",
        "limiter_derivative_evaluation_count": 0,
        "mask_derivative_evaluation_count": 0,
        "explicit_fallback": False,
        "boundary_scope": "closed_reflective",
        "grid_scope": "fixed_grid",
        "production_admission": False,
    }


def resolve_wetdry_fwave_cpu_execution_v97(
    base_execution: CoupledGeometryNativeCPUExecutionV96,
    native_build_receipt: NativeWetDryFwaveBuildReceiptV97, *,
    domain_active_mask: object,
    native_threads: int = 4,
    max_active_set_restarts: int = MAX_ACTIVE_SET_RESTARTS,
    max_inner_iterations: int = MAX_INNER_ITERATIONS,
    nonlinear_tolerance: float = 2e-10,
    positivity_fraction: float = 0.9,
) -> WetDryFwaveCPUExecutionV97:
    if type(base_execution) is not CoupledGeometryNativeCPUExecutionV96:
        raise WetDryFwaveNativeV97Error("v97 requires exact v96 CPU execution")
    execution_document_coupled_geometry_native_v96(base_execution)
    validate_native_wetdry_fwave_build_receipt_v97(native_build_receipt)
    nc = base_execution.cell_count
    active = _array(domain_active_mask, (nc,), "domain-active mask", integer=True)
    if np.any((active != 0) & (active != 1)) or not np.any(active):
        raise WetDryFwaveNativeV97Error("v97 domain-active mask is invalid")
    # Bounded fixed-grid topology has no inactive holes.  The separate field is
    # still authenticated so later cut-cell/AMR routes cannot conflate it with
    # the evolving wet mask.
    if np.any(active == 0):
        raise WetDryFwaveNativeV97Error(
            "v97 bounded fixed-grid topology does not yet admit inactive holes")
    if type(native_threads) is not int or not 1 <= native_threads <= 256:
        raise WetDryFwaveNativeV97Error("v97 native thread count differs")
    if (type(max_active_set_restarts) is not int or
            not 0 <= max_active_set_restarts <= 16 or
            type(max_inner_iterations) is not int or
            not 1 <= max_inner_iterations <= 200):
        raise WetDryFwaveNativeV97Error("v97 iteration bounds differ")
    if (isinstance(nonlinear_tolerance, bool) or
            not math.isfinite(float(nonlinear_tolerance)) or
            not 64*np.finfo(float).eps <= float(nonlinear_tolerance) <= 1e-5 or
            isinstance(positivity_fraction, bool) or
            not math.isfinite(float(positivity_fraction)) or
            not 0.5 <= float(positivity_fraction) < 1.0):
        raise WetDryFwaveNativeV97Error("v97 tolerance/damping differs")
    placeholder = WetDryFwaveCPUExecutionV97(
        base_execution, native_build_receipt, active, native_threads,
        max_active_set_restarts, max_inner_iterations,
        float(nonlinear_tolerance), float(positivity_fraction), "")
    return WetDryFwaveCPUExecutionV97(
        **{**placeholder.__dict__,
           "fingerprint": _fingerprint(_execution_payload(placeholder))})


def execution_document_wetdry_fwave_v97(
    value: WetDryFwaveCPUExecutionV97,
) -> dict[str, object]:
    if type(value) is not WetDryFwaveCPUExecutionV97:
        raise WetDryFwaveNativeV97Error("v97 execution type differs")
    execution_document_coupled_geometry_native_v96(value.base_execution)
    validate_native_wetdry_fwave_build_receipt_v97(value.native_build_receipt)
    payload = _execution_payload(value)
    if (value.domain_active_mask.shape != (value.cell_count,) or
            value.domain_active_mask.dtype != np.int32 or
            value.domain_active_mask.flags.writeable or
            np.any(value.domain_active_mask != 1) or
            type(value.native_threads) is not int or
            not 1 <= value.native_threads <= 256 or
            type(value.max_active_set_restarts) is not int or
            not 0 <= value.max_active_set_restarts <= 16 or
            type(value.max_inner_iterations) is not int or
            not 1 <= value.max_inner_iterations <= 200 or
            not 64*np.finfo(float).eps <= value.nonlinear_tolerance <= 1e-5 or
            not 0.5 <= value.positivity_fraction < 1.0 or
            value.fingerprint != _fingerprint(payload)):
        raise WetDryFwaveNativeV97Error("v97 execution fingerprint differs")
    return {**payload, "fingerprint": value.fingerprint}


@dataclass(frozen=True)
class NativeWetDryFwaveRateResultV97:
    raw_left_face_flux: np.ndarray
    raw_right_face_flux: np.ndarray
    limited_left_face_flux: np.ndarray
    limited_right_face_flux: np.ndarray
    draining_theta: np.ndarray
    base_fwave_rate: np.ndarray
    face_water_flux_m2_s: np.ndarray
    face_normal_acceleration_m_s2: np.ndarray
    wet_wet_face_mask: np.ndarray
    frozen_wet_mask: np.ndarray
    midpoint_sha256: str
    accepted_bed_sha256: str
    accepted_density_sha256: str
    dt_s: float
    augmented_riemann_call_count: int
    draining_limiter_application_count: int
    limiter_derivative_evaluation_count: int
    mask_derivative_evaluation_count: int
    native_threads: int
    result_sha256: str


def _result_payload(value: NativeWetDryFwaveRateResultV97,
                    execution: WetDryFwaveCPUExecutionV97) -> dict:
    return {
        "schema": SCHEMA, "scope_id": SCOPE_ID + ":rate-result",
        "execution_fingerprint": execution.fingerprint,
        "array_sha256": _digest(
            value.raw_left_face_flux, value.raw_right_face_flux,
            value.limited_left_face_flux, value.limited_right_face_flux,
            value.draining_theta, value.base_fwave_rate,
            value.face_water_flux_m2_s,
            value.face_normal_acceleration_m_s2, value.wet_wet_face_mask,
            value.frozen_wet_mask),
        "midpoint_sha256": value.midpoint_sha256,
        "accepted_bed_sha256": value.accepted_bed_sha256,
        "accepted_density_sha256": value.accepted_density_sha256,
        "dt_s": value.dt_s,
        "augmented_riemann_call_count": value.augmented_riemann_call_count,
        "draining_limiter_application_count":
            value.draining_limiter_application_count,
        "limiter_derivative_evaluation_count":
            value.limiter_derivative_evaluation_count,
        "mask_derivative_evaluation_count":
            value.mask_derivative_evaluation_count,
        "native_threads": value.native_threads,
    }


class NativeWetDryFwaveRuntimeV97:
    __slots__ = ("receipt", "library_path", "_library", "_function",
                 "_dependency")

    def __init__(self, receipt: NativeWetDryFwaveBuildReceiptV97,
                 library_path: Path, library, function, dependency) -> None:
        validate_native_wetdry_fwave_build_receipt_v97(receipt)
        self.receipt = receipt
        self.library_path = Path(library_path).resolve()
        self._library = library
        self._function = function
        self._dependency = dependency

    def evaluate(
        self, execution: WetDryFwaveCPUExecutionV97,
        midpoint_q: object, *, bed_elevation_m: object,
        accepted_water_density_kg_m3: object,
        accepted_mixture_density_kg_m3: object,
        frozen_wet_mask: object, dt_s: float,
        cancelled: bool | Callable[[], bool] = False,
    ) -> NativeWetDryFwaveRateResultV97:
        execution_document_wetdry_fwave_v97(execution)
        if self.receipt != execution.native_build_receipt:
            raise WetDryFwaveNativeV97Error("v97 runtime receipt differs")
        if not self.library_path.is_file() or _sha(self.library_path) != self.receipt.library_sha256:
            raise WetDryFwaveNativeV97Error("v97 loaded native library differs")
        poll = _cancel(cancelled)
        if poll():
            raise InterruptedError("v97 wet/dry rate cancelled before launch")
        base = execution.base_execution.base_execution
        nc, nf = execution.cell_count, execution.face_count
        q = _array(midpoint_q, (3, nc), "midpoint q")
        bed = _array(bed_elevation_m, (nc,), "accepted bed")
        water = _array(accepted_water_density_kg_m3, (nc,), "water density")
        mixture = _array(accepted_mixture_density_kg_m3, (nc,), "mixture density")
        wet = _array(frozen_wet_mask, (nc,), "frozen wet mask", integer=True)
        if (np.any(q[0] < 0) or np.any(water <= 0) or np.any(mixture < water) or
                np.any((wet != 0) & (wet != 1)) or
                np.any((wet == 1) & (q[0] <= 0)) or
                isinstance(dt_s, bool) or not math.isfinite(float(dt_s)) or
                not 0 < float(dt_s) <= execution.base_execution.maximum_dt_s):
            raise WetDryFwaveNativeV97Error("v97 rate physical inputs differ")
        density = (mixture if execution.base_execution.density_mode ==
                   DENSITY_WATER_PLUS_SEDIMENT else water)
        arrays = [np.empty((3, nf)) for _ in range(4)]
        theta = np.empty(nc); rate = np.empty((3, nc))
        water_flux = np.empty(nf); acceleration = np.empty(nf)
        wet_wet = np.empty(nf, np.int32)
        riemann = ctypes.c_int(-1); limiter = ctypes.c_int(-1)
        status = ctypes.c_int(-1)
        with _GE0CLAW_CALL_LOCK:
            self._function(
                nc, nf, base.cell_faces.size,
                np.ascontiguousarray(q), np.ascontiguousarray(bed),
                np.ascontiguousarray(base.owner, dtype=np.int32),
                np.ascontiguousarray(base.neighbour, dtype=np.int32),
                np.ascontiguousarray(base.normal),
                np.ascontiguousarray(base.face_length_m),
                np.ascontiguousarray(base.face_distance_m),
                np.ascontiguousarray(base.cell_area_m2),
                np.ascontiguousarray(base.cell_face_offset, dtype=np.int32),
                np.ascontiguousarray(base.cell_faces, dtype=np.int32),
                np.ascontiguousarray(base.cell_face_sign, dtype=np.int32),
                np.ascontiguousarray(execution.domain_active_mask, dtype=np.int32),
                np.ascontiguousarray(wet, dtype=np.int32),
                np.ascontiguousarray(density),
                base.hydraulic_reference_density_kg_m3, base.gravity_m_s2,
                base.dry_tolerance_m, float(dt_s), execution.native_threads,
                *arrays, theta, rate, water_flux, acceleration, wet_wet,
                ctypes.byref(riemann), ctypes.byref(limiter),
                ctypes.byref(status))
        if status.value != 0:
            raise WetDryFwaveNativeV97Error(
                f"v97 native wet/dry rate failed with status {status.value}")
        if poll():
            raise InterruptedError("v97 wet/dry rate cancelled after launch")
        if (riemann.value != nf or limiter.value != int(np.count_nonzero(theta < 1.0))
                and limiter.value < 0):
            # A donor can limit several faces; only malformed negative counts
            # are rejected here. Exact face count is independently rebuilt below.
            raise WetDryFwaveNativeV97Error("v97 native owner counts differ")
        all_arrays = (*arrays, theta, rate, water_flux, acceleration, wet_wet)
        if not all(np.isfinite(value).all() for value in all_arrays):
            raise WetDryFwaveNativeV97Error("v97 native result is non-finite")
        expected_wet_wet = ((base.neighbour >= 0) &
            (wet[base.owner] == 1) &
            (wet[np.maximum(base.neighbour, 0)] == 1)).astype(np.int32)
        if not np.array_equal(wet_wet, expected_wet_wet):
            raise WetDryFwaveNativeV97Error("v97 wet-wet face mask differs")
        slots_per_cell = np.diff(base.cell_face_offset)
        cell = np.repeat(np.arange(nc), slots_per_cell)
        face = base.cell_faces
        signed = base.cell_face_sign*water_flux[face]*base.face_length_m[face]
        rebuilt = -np.bincount(cell, weights=signed, minlength=nc)/base.cell_area_m2
        if not np.allclose(rate[0], rebuilt, rtol=64*np.finfo(float).eps,
                           atol=64*np.finfo(float).eps):
            raise WetDryFwaveNativeV97Error("v97 limited water ledger differs")
        for value in all_arrays:
            value.setflags(write=False)
        placeholder = NativeWetDryFwaveRateResultV97(
            arrays[0], arrays[1], arrays[2], arrays[3], theta, rate,
            water_flux, acceleration, wet_wet, wet, _digest(q), _digest(bed),
            _digest(density), float(dt_s), riemann.value, limiter.value, 0, 0,
            execution.native_threads, "")
        return NativeWetDryFwaveRateResultV97(
            **{**placeholder.__dict__,
               "result_sha256": _fingerprint(_result_payload(
                   placeholder, execution))})


def validate_native_wetdry_rate_result_v97(
    value: NativeWetDryFwaveRateResultV97,
    execution: WetDryFwaveCPUExecutionV97,
) -> None:
    if type(value) is not NativeWetDryFwaveRateResultV97:
        raise WetDryFwaveNativeV97Error("v97 rate result type differs")
    execution_document_wetdry_fwave_v97(execution)
    nc, nf = execution.cell_count, execution.face_count
    shapes = ((3, nf), (3, nf), (3, nf), (3, nf), (nc,), (3, nc),
              (nf,), (nf,), (nf,), (nc,))
    arrays = (value.raw_left_face_flux, value.raw_right_face_flux,
              value.limited_left_face_flux, value.limited_right_face_flux,
              value.draining_theta, value.base_fwave_rate,
              value.face_water_flux_m2_s,
              value.face_normal_acceleration_m_s2,
              value.wet_wet_face_mask, value.frozen_wet_mask)
    if (any(array.shape != shape or array.flags.writeable or
            not np.isfinite(array).all() for array, shape in zip(arrays, shapes)) or
            np.any(value.draining_theta < 0) or
            np.any(value.draining_theta > 1) or
            value.augmented_riemann_call_count != nf or
            value.draining_limiter_application_count < 0 or
            value.limiter_derivative_evaluation_count != 0 or
            value.mask_derivative_evaluation_count != 0 or
            value.native_threads != execution.native_threads or
            any(_SHA.fullmatch(item) is None for item in (
                value.midpoint_sha256, value.accepted_bed_sha256,
                value.accepted_density_sha256)) or
            value.result_sha256 != _fingerprint(_result_payload(value, execution))):
        raise WetDryFwaveNativeV97Error("v97 rate result identity differs")


@dataclass(frozen=True)
class WetDryMidpointTrialV97:
    midpoint_q: np.ndarray
    endpoint_q: np.ndarray
    frozen_wet_mask: np.ndarray
    accepted_wet_mask: np.ndarray
    accepted_limited_face_water_flux_m2_s: np.ndarray
    accepted_draining_theta: np.ndarray
    accepted_base_fwave_rate: np.ndarray
    accepted_density_momentum_rate: np.ndarray
    accepted_turbulence_momentum_rate: np.ndarray
    accepted_turbulence_face_flux: np.ndarray
    accepted_total_rate: np.ndarray
    dry_momentum_cleanup_rate_m2_s2: np.ndarray
    density_projection_receipt_sha256: str
    active_set_restart_count: int
    inner_iteration_count: int
    true_residual_evaluation_count: int
    damping_application_count: int
    draining_limiter_application_count: int
    augmented_riemann_call_count: int
    density_projection_application_count: int
    shared_turbulence_application_count: int
    limiter_derivative_evaluation_count: int
    mask_derivative_evaluation_count: int
    dry_momentum_cleanup_cell_count: int
    dry_momentum_cleanup_linf_m2_s2: float
    projected_endpoint_identity_linf: float
    water_volume_residual_m3: float
    nonlinear_residual_linf: float
    execution_fingerprint: str
    parent_q_sha256: str
    accepted_bed_sha256: str
    accepted_water_density_sha256: str
    accepted_mixture_density_sha256: str
    dt_s: float
    native_rate_result: NativeWetDryFwaveRateResultV97
    density_projection_result: Mapping[str, object]
    shared_turbulence_result: object | None
    trial_sha256: str


def _midpoint_trial_payload(value: WetDryMidpointTrialV97) -> dict:
    turbulence_sha = (None if value.shared_turbulence_result is None else
                      getattr(value.shared_turbulence_result,
                              "result_sha256", None))
    return {
        "schema": SCHEMA, "scope_id": SCOPE_ID + ":trial",
        "execution_fingerprint": value.execution_fingerprint,
        "parent_q_sha256": value.parent_q_sha256,
        "accepted_bed_sha256": value.accepted_bed_sha256,
        "accepted_water_density_sha256":
            value.accepted_water_density_sha256,
        "accepted_mixture_density_sha256":
            value.accepted_mixture_density_sha256,
        "arrays_sha256": _digest(
            value.midpoint_q, value.endpoint_q, value.frozen_wet_mask,
            value.accepted_wet_mask,
            value.accepted_limited_face_water_flux_m2_s,
            value.accepted_draining_theta,
            value.accepted_base_fwave_rate,
            value.accepted_density_momentum_rate,
            value.accepted_turbulence_momentum_rate,
            value.accepted_turbulence_face_flux,
            value.accepted_total_rate,
            value.dry_momentum_cleanup_rate_m2_s2),
        "dt_s": value.dt_s,
        "native_rate_result_sha256":
            value.native_rate_result.result_sha256,
        "density_projection_receipt_sha256":
            value.density_projection_receipt_sha256,
        "shared_turbulence_result_sha256": turbulence_sha,
        "active_set_restart_count": value.active_set_restart_count,
        "inner_iteration_count": value.inner_iteration_count,
        "true_residual_evaluation_count":
            value.true_residual_evaluation_count,
        "damping_application_count": value.damping_application_count,
        "draining_limiter_application_count":
            value.draining_limiter_application_count,
        "augmented_riemann_call_count":
            value.augmented_riemann_call_count,
        "density_projection_application_count":
            value.density_projection_application_count,
        "shared_turbulence_application_count":
            value.shared_turbulence_application_count,
        "limiter_derivative_evaluation_count":
            value.limiter_derivative_evaluation_count,
        "mask_derivative_evaluation_count":
            value.mask_derivative_evaluation_count,
        "dry_momentum_cleanup_cell_count":
            value.dry_momentum_cleanup_cell_count,
        "dry_momentum_cleanup_linf_m2_s2":
            value.dry_momentum_cleanup_linf_m2_s2,
        "projected_endpoint_identity_linf":
            value.projected_endpoint_identity_linf,
        "water_volume_residual_m3": value.water_volume_residual_m3,
        "nonlinear_residual_linf": value.nonlinear_residual_linf,
    }


def validate_wetdry_midpoint_trial_v97(
    value: WetDryMidpointTrialV97,
    execution: WetDryFwaveCPUExecutionV97, *, parent_q: object,
    bed_elevation_m: object, accepted_water_density_kg_m3: object,
    accepted_mixture_density_kg_m3: object, dt_s: float,
) -> None:
    """Rebuild the accepted active-set trial and all public cross-fields."""
    if type(value) is not WetDryMidpointTrialV97:
        raise WetDryFwaveNativeV97Error("v97 midpoint trial type differs")
    execution_document_wetdry_fwave_v97(execution)
    nc, nf = execution.cell_count, execution.face_count
    q0 = _array(parent_q, (3, nc), "trial parent q")
    bed = _array(bed_elevation_m, (nc,), "trial accepted bed")
    water = _array(
        accepted_water_density_kg_m3, (nc,), "trial water density")
    mixture = _array(
        accepted_mixture_density_kg_m3, (nc,), "trial mixture density")
    dt = float(dt_s)
    arrays = (
        (value.midpoint_q, (3, nc)), (value.endpoint_q, (3, nc)),
        (value.frozen_wet_mask, (nc,)),
        (value.accepted_wet_mask, (nc,)),
        (value.accepted_limited_face_water_flux_m2_s, (nf,)),
        (value.accepted_draining_theta, (nc,)),
        (value.accepted_base_fwave_rate, (3, nc)),
        (value.accepted_density_momentum_rate, (3, nc)),
        (value.accepted_turbulence_momentum_rate, (3, nc)),
        (value.accepted_turbulence_face_flux, (2, nf)),
        (value.accepted_total_rate, (3, nc)),
        (value.dry_momentum_cleanup_rate_m2_s2, (3, nc)),
    )
    if any(np.asarray(array).shape != shape or
           np.asarray(array).flags.writeable or
           not np.isfinite(array).all() for array, shape in arrays):
        raise WetDryFwaveNativeV97Error(
            "v97 midpoint trial array inventory differs")
    validate_native_wetdry_rate_result_v97(
        value.native_rate_result, execution)
    raw = value.native_rate_result
    selected_density = (mixture if execution.base_execution.density_mode ==
                        DENSITY_WATER_PLUS_SEDIMENT else water)
    mask = np.asarray(value.frozen_wet_mask, dtype=np.int32)
    accepted_mask = np.asarray(value.accepted_wet_mask, dtype=np.int32)
    base = execution.base_execution.base_execution
    expected_accepted_mask = (
        execution.domain_active_mask.astype(bool) &
        (value.endpoint_q[0] > base.dry_tolerance_m))
    initial_mask = (execution.domain_active_mask.astype(bool) &
                    (q0[0] > base.dry_tolerance_m))
    expected_evaluation_mask = initial_mask | expected_accepted_mask
    geometry = resolve_density_projection_geometry_v97(
        execution.base_execution.base_execution.normal,
        execution.base_execution.base_execution.face_length_m,
        execution.base_execution.base_execution.cell_face_offset,
        execution.base_execution.base_execution.cell_faces,
        execution.base_execution.base_execution.cell_face_sign)
    validate_projection_result_v97(
        value.density_projection_result, geometry,
        value.midpoint_q[0], raw.face_normal_acceleration_m_s2,
        mask, raw.wet_wet_face_mask,
        runtime_sha=value.density_projection_result[
            "native_runtime_sha256"])
    projection_rate = np.zeros((3, nc))
    projection_rate[1:] = value.density_projection_result[
        "momentum_rate_m2_s2"]
    if value.shared_turbulence_result is None:
        turbulence_ok = (
            value.shared_turbulence_application_count == 0 and
            np.all(value.accepted_turbulence_momentum_rate == 0.0) and
            np.all(value.accepted_turbulence_face_flux == 0.0))
    else:
        from .shared_turbulence_native_cpu_v97 import (
            NativeMomentumRateV97, _digest as _v91_digest,
            _result_token as _v91_result_token,
        )
        turbulence = value.shared_turbulence_result
        turbulence_ok = (
            type(turbulence) is NativeMomentumRateV97 and
            turbulence.result_sha256 == _v91_result_token(turbulence) and
            turbulence.midpoint_q_sha256 == _v91_digest(value.midpoint_q) and
            turbulence.wet_cell_mask_sha256 == _v91_digest(mask) and
            turbulence.active_wet_wet_face_mask_sha256 ==
                _v91_digest(raw.wet_wet_face_mask) and
            value.shared_turbulence_application_count ==
                value.true_residual_evaluation_count and
            np.array_equal(value.accepted_turbulence_momentum_rate,
                           turbulence.momentum_rate_m2_s2) and
            np.array_equal(value.accepted_turbulence_face_flux,
                           turbulence.face_flux_m4_s2))
    physical_total = (value.accepted_base_fwave_rate+
                      value.accepted_density_momentum_rate+
                      value.accepted_turbulence_momentum_rate)
    physical_endpoint = q0+dt*physical_total
    accepted_dry = ~expected_accepted_mask
    expected_cleanup = np.zeros((3, nc), dtype=np.float64)
    expected_cleanup[1:, accepted_dry] = \
        -physical_endpoint[1:, accepted_dry]/dt
    projected_cells = np.any(
        physical_endpoint[1:, accepted_dry] != 0.0, axis=0)
    total = physical_total+expected_cleanup
    midpoint_residual = value.midpoint_q-q0-.5*dt*physical_total
    residual = float(np.max(np.abs(midpoint_residual)))
    cleanup_linf = float(np.max(np.abs(expected_cleanup)))
    endpoint_identity = value.endpoint_q-(q0+dt*total)
    projected_identity_linf = float(np.max(np.abs(endpoint_identity)))
    volume_change = float(np.sum(
        base.cell_area_m2*(value.endpoint_q[0]-q0[0])))
    boundary = base.neighbour < 0
    boundary_ledger = -dt*float(np.sum(
        value.accepted_limited_face_water_flux_m2_s[boundary]*
        base.face_length_m[boundary]))
    water_residual = volume_change-boundary_ledger
    if (value.execution_fingerprint != execution.fingerprint or
            value.parent_q_sha256 != _digest(q0) or
            value.accepted_bed_sha256 != _digest(bed) or
            value.accepted_water_density_sha256 != _digest(water) or
            value.accepted_mixture_density_sha256 != _digest(mixture) or
            value.dt_s != dt or raw.midpoint_sha256 != _digest(
                value.midpoint_q) or
            raw.accepted_bed_sha256 != _digest(bed) or
            raw.accepted_density_sha256 != _digest(selected_density) or
            raw.dt_s != dt or
            not np.array_equal(mask, expected_evaluation_mask.astype(
                np.int32)) or
            not np.array_equal(accepted_mask, expected_accepted_mask.astype(
                np.int32)) or
            not np.array_equal(raw.frozen_wet_mask, mask) or
            not np.array_equal(value.accepted_limited_face_water_flux_m2_s,
                               raw.face_water_flux_m2_s) or
            not np.array_equal(value.accepted_draining_theta,
                               raw.draining_theta) or
            not np.array_equal(value.accepted_base_fwave_rate,
                               raw.base_fwave_rate) or
            not np.array_equal(value.accepted_density_momentum_rate,
                               projection_rate) or
            not np.array_equal(
                value.dry_momentum_cleanup_rate_m2_s2,
                expected_cleanup) or
            not turbulence_ok or not np.array_equal(
                value.accepted_total_rate, total) or
            not np.array_equal(value.endpoint_q,
                               q0+dt*value.accepted_total_rate) or
            value.density_projection_receipt_sha256 !=
                value.density_projection_result["receipt_sha256"] or
            value.density_projection_application_count !=
                value.true_residual_evaluation_count or
            value.limiter_derivative_evaluation_count != 0 or
            value.mask_derivative_evaluation_count != 0 or
            value.dry_momentum_cleanup_cell_count !=
                int(np.count_nonzero(projected_cells)) or
            value.dry_momentum_cleanup_linf_m2_s2 != cleanup_linf or
            value.projected_endpoint_identity_linf !=
                projected_identity_linf or
            np.any(value.endpoint_q[1:, ~expected_accepted_mask] != 0.0) or
            abs(value.nonlinear_residual_linf-residual) >
                16*np.finfo(float).eps*max(1.0, residual) or
            abs(value.water_volume_residual_m3-water_residual) >
                16*np.finfo(float).eps*max(1.0, abs(water_residual)) or
            value.trial_sha256 != _fingerprint(
                _midpoint_trial_payload(value))):
        raise WetDryFwaveNativeV97Error(
            "v97 midpoint trial is stale, tampered or cross-input")


def solve_wetdry_midpoint_trial_v97(
    execution: WetDryFwaveCPUExecutionV97,
    parent_q: object, *, bed_elevation_m: object,
    accepted_water_density_kg_m3: object,
    accepted_mixture_density_kg_m3: object,
    dt_s: float,
    native_runtime: NativeWetDryFwaveRuntimeV97,
    density_projection_runtime: NativeDensityActiveProjectionRuntimeV97,
    shared_turbulence_runtime=None,
    shared_turbulence_prepared=None,
    cancelled: bool | Callable[[], bool] = False,
) -> WetDryMidpointTrialV97:
    """Solve one bounded closed-grid midpoint trial with a stable wet set.

    This owns no accepted clock/state mutation.  Cancellation and every
    failure therefore leave all inputs byte-identical and permit exact retry.
    """
    execution_document_wetdry_fwave_v97(execution)
    poll = _cancel(cancelled)
    nc = execution.cell_count
    q0 = _array(parent_q, (3, nc), "parent q")
    bed = _array(bed_elevation_m, (nc,), "accepted bed")
    water = _array(accepted_water_density_kg_m3, (nc,), "water density")
    mixture = _array(accepted_mixture_density_kg_m3, (nc,), "mixture density")
    base = execution.base_execution.base_execution
    if (np.any(q0[0] < 0) or np.any((q0[0] <= base.dry_tolerance_m) &
                                    (q0[1:] != 0))):
        raise WetDryFwaveNativeV97Error(
            "v97 parent dry cells must have exact zero momentum")
    if (shared_turbulence_runtime is None) != (shared_turbulence_prepared is None):
        raise WetDryFwaveNativeV97Error(
            "v97 masked-v91 runtime and preparation must be supplied together")
    if shared_turbulence_runtime is not None:
        from .shared_turbulence_native_cpu_v97 import (
            NativeSharedTurbulenceRuntimeV97,
            PreparedNativeTurbulenceV97,
            _digest as _v91_digest,
            _result_token as _v91_result_token,
        )
        if (type(shared_turbulence_runtime) is not NativeSharedTurbulenceRuntimeV97 or
                type(shared_turbulence_prepared) is not PreparedNativeTurbulenceV97):
            raise WetDryFwaveNativeV97Error(
                "v97 masked-v91 provider type differs")
        viscous_base = shared_turbulence_prepared.execution.base_execution.base_execution
        topology_equal = (
            viscous_base.cell_count == nc and viscous_base.face_count == execution.face_count and
            np.array_equal(viscous_base.owner, base.owner) and
            np.array_equal(viscous_base.neighbour, base.neighbour) and
            np.array_equal(viscous_base.cell_face_offset, base.cell_face_offset) and
            np.array_equal(viscous_base.cell_faces, base.cell_faces) and
            np.array_equal(viscous_base.cell_face_sign, base.cell_face_sign) and
            np.array_equal(viscous_base.face_length_m, base.face_length_m) and
            np.array_equal(viscous_base.cell_area_m2, base.cell_area_m2))
        if (not topology_equal or
                shared_turbulence_prepared.cap_receipt.dt_s != float(dt_s)):
            raise WetDryFwaveNativeV97Error(
                "v97 masked-v91 topology/timestep differs from hydraulics")
    geometry = resolve_density_projection_geometry_v97(
        base.normal, base.face_length_m, base.cell_face_offset,
        base.cell_faces, base.cell_face_sign)
    tau = .5*float(dt_s)
    mask = (execution.domain_active_mask.astype(bool) &
            (q0[0] > base.dry_tolerance_m))
    seen: set[str] = set()
    midpoint = np.array(q0, copy=True)
    final_rate = None; final_raw = None; final_projection = None
    final_turbulence = None
    accepted_mask = np.array(mask, copy=True)
    dry_momentum_cleanup = np.zeros((3, nc), dtype=np.float64)
    accepted_total_rate = None
    dry_momentum_cleanup_cell_count = 0
    total_iterations = total_residuals = damping_count = 0
    total_limiter = total_riemann = total_turbulence = 0
    residual_norm = math.inf
    for restart in range(execution.max_active_set_restarts + 1):
        mask_key = _digest(mask.astype(np.int32))
        if mask_key in seen:
            raise WetDryFwaveNativeV97Error(
                "v97 wet active set oscillated before acceptance")
        seen.add(mask_key)
        midpoint = np.array(q0 if restart == 0 else midpoint, copy=True)
        midpoint[1:, ~mask] = 0.0
        converged = False
        for _ in range(execution.max_inner_iterations + 1):
            if poll():
                raise InterruptedError("v97 wet/dry midpoint solve cancelled")
            raw = native_runtime.evaluate(
                execution, midpoint, bed_elevation_m=bed,
                accepted_water_density_kg_m3=water,
                accepted_mixture_density_kg_m3=mixture,
                frozen_wet_mask=mask.astype(np.int32), dt_s=dt_s,
                cancelled=poll)
            validate_native_wetdry_rate_result_v97(raw, execution)
            projection = density_projection_runtime.evaluate(
                geometry, midpoint[0], raw.face_normal_acceleration_m_s2,
                mask.astype(np.int32), raw.wet_wet_face_mask,
                policy=DENSITY_PROJECTION_POLICY_V97,
                native_threads=execution.native_threads, cancelled=poll)
            validate_projection_result_v97(
                projection, geometry, midpoint[0],
                raw.face_normal_acceleration_m_s2, mask.astype(np.int32),
                raw.wet_wet_face_mask,
                runtime_sha=density_projection_runtime.receipt["receipt_sha256"])
            rate = np.array(raw.base_fwave_rate, copy=True)
            rate[1:] += projection["momentum_rate_m2_s2"]
            if shared_turbulence_runtime is not None:
                turbulent = shared_turbulence_runtime.evaluate(
                    shared_turbulence_prepared, midpoint,
                    mask.astype(np.int32), raw.wet_wet_face_mask,
                    native_threads=execution.native_threads, cancelled=poll)
                if (turbulent.momentum_rate_m2_s2.shape != (3, nc) or
                        turbulent.result_sha256 != _v91_result_token(turbulent) or
                        turbulent.midpoint_q_sha256 != _v91_digest(midpoint) or
                        turbulent.wet_cell_mask_sha256 != _v91_digest(mask.astype(np.int32)) or
                        turbulent.active_wet_wet_face_mask_sha256 !=
                        _v91_digest(raw.wet_wet_face_mask) or
                        turbulent.momentum_rate_evaluation_count != 1 or
                        turbulent.depth_source_evaluation_count != 0 or
                        turbulent.state_or_clock_publication_count != 0):
                    raise WetDryFwaveNativeV97Error(
                        "v97 masked-v91 result identity differs")
                rate += turbulent.momentum_rate_m2_s2
                total_turbulence += 1
                final_turbulence = turbulent
            residual = midpoint-q0-tau*rate
            residual_norm = float(np.max(np.abs(residual)))
            scale = max(1.0, float(np.max(np.abs(midpoint))),
                        float(np.max(np.abs(q0))),
                        float(np.max(np.abs(tau*rate))))
            total_residuals += 1; total_riemann += raw.augmented_riemann_call_count
            total_limiter += raw.draining_limiter_application_count
            final_rate, final_raw, final_projection = rate, raw, projection
            if residual_norm <= execution.nonlinear_tolerance*scale:
                converged = True
                break
            if total_iterations >= ((restart+1)*execution.max_inner_iterations):
                break
            delta = -residual
            alpha = 1.0
            falling = delta[0] < 0.0
            if np.any(falling):
                floor = np.where(mask, np.finfo(float).tiny, 0.0)
                available = midpoint[0, falling]-floor[falling]
                if np.any(available <= 0.0):
                    raise WetDryFwaveNativeV97Error(
                        "v97 damped correction has no positive-depth fraction")
                alpha = min(alpha, execution.positivity_fraction*float(
                    np.min(available/(-delta[0, falling]))))
            if not math.isfinite(alpha) or alpha <= 0.0:
                raise WetDryFwaveNativeV97Error(
                    "v97 positive-depth damping factor is invalid")
            if alpha < 1.0:
                damping_count += 1
            midpoint += alpha*delta
            if np.any(midpoint[0] < 0.0) or not np.isfinite(midpoint).all():
                raise WetDryFwaveNativeV97Error(
                    "v97 damped midpoint left the nonnegative envelope")
            total_iterations += 1
        if not converged or final_rate is None or final_raw is None:
            raise WetDryFwaveNativeV97Error(
                "v97 damped midpoint did not converge")
        physical_endpoint = q0+float(dt_s)*final_rate
        if (not np.isfinite(physical_endpoint).all() or
                np.any(physical_endpoint[0] < 0.0)):
            raise WetDryFwaveNativeV97Error(
                "v97 limited endpoint is negative or non-finite")
        new_mask = (execution.domain_active_mask.astype(bool) &
                    (physical_endpoint[0] > base.dry_tolerance_m))
        if np.array_equal(new_mask, mask):
            accepted_mask = np.array(new_mask, copy=True)
            dry_momentum_cleanup = np.zeros((3, nc), dtype=np.float64)
            accepted_dry = ~accepted_mask
            dry_momentum_cleanup[1:, accepted_dry] = (
                -physical_endpoint[1:, accepted_dry]/float(dt_s))
            accepted_total_rate = final_rate+dry_momentum_cleanup
            endpoint = q0+float(dt_s)*accepted_total_rate
            dry_momentum_cleanup_cell_count = int(np.count_nonzero(
                np.any(physical_endpoint[1:, accepted_dry] != 0.0,
                       axis=0)))
            break
        newly_dry = mask & ~new_mask
        newly_wet = ~mask & new_mask
        if np.any(newly_dry) and np.any(newly_wet):
            raise WetDryFwaveNativeV97Error(
                "v97 simultaneous wetting and drying is outside the "
                "bounded active-set route")
        if np.any(newly_dry):
            # The converged residual was evaluated on ``mask``.  Removing a
            # drying cell and its draining faces before acceptance destroys
            # the conservative depth update and can make the active set
            # oscillate.  Keep that physical update and face ledger, and own
            # only the endpoint dry-momentum projection as an explicit,
            # authenticated cleanup source.  Scalars and Q3 consume the
            # accepted endpoint mask below.
            initial_mask = (execution.domain_active_mask.astype(bool) &
                            (q0[0] > base.dry_tolerance_m))
            if not np.array_equal(mask, initial_mask | new_mask):
                raise WetDryFwaveNativeV97Error(
                    "v97 transient wet/dry active-set history is ambiguous")
            accepted_mask = np.array(new_mask, copy=True)
            dry_momentum_cleanup = np.zeros((3, nc), dtype=np.float64)
            accepted_dry = ~accepted_mask
            dry_momentum_cleanup[1:, accepted_dry] = (
                -physical_endpoint[1:, accepted_dry]/float(dt_s))
            accepted_total_rate = final_rate+dry_momentum_cleanup
            endpoint = q0+float(dt_s)*accepted_total_rate
            dry_momentum_cleanup_cell_count = int(np.count_nonzero(
                np.any(physical_endpoint[1:, accepted_dry] != 0.0,
                       axis=0)))
            break
        # Wetting is the only active-set change that requires a restart: the
        # newly wet cell and its faces must participate in the residual.
        midpoint = .5*(q0+physical_endpoint)
        midpoint[1:, ~new_mask] = 0.0
        mask = new_mask
    else:
        raise WetDryFwaveNativeV97Error(
            "v97 active-set restart bound was exhausted")
    if accepted_total_rate is None:
        raise WetDryFwaveNativeV97Error(
            "v97 accepted endpoint rate was not constructed")
    expected_evaluation_mask = (
        (execution.domain_active_mask.astype(bool) &
         (q0[0] > base.dry_tolerance_m)) | accepted_mask)
    if not np.array_equal(mask, expected_evaluation_mask):
        raise WetDryFwaveNativeV97Error(
            "v97 accepted residual and endpoint wet masks differ")
    physical_endpoint_identity = (
        physical_endpoint-(2.0*midpoint-q0) +
        2.0*(midpoint-q0-tau*final_rate))
    identity_scale = max(1.0, float(np.max(np.abs(physical_endpoint))),
                         float(np.max(np.abs(midpoint))),
                         float(np.max(np.abs(q0))))
    if (np.max(np.abs(physical_endpoint_identity)) >
            256*np.finfo(float).eps*identity_scale):
        raise WetDryFwaveNativeV97Error(
            "v97 pre-projection endpoint identity differs")
    projected_endpoint_identity = endpoint-(q0+float(dt_s)*accepted_total_rate)
    projected_identity_linf = float(np.max(np.abs(
        projected_endpoint_identity)))
    if projected_identity_linf != 0.0:
        raise WetDryFwaveNativeV97Error(
            "v97 projected endpoint identity differs")
    volume_change = float(np.sum(base.cell_area_m2*(endpoint[0]-q0[0])))
    # Closed physical boundaries have exact-zero limited mass flux; internal
    # faces cancel through the cell gather.  This is the accepted face ledger,
    # not a post-hoc clipping correction.
    boundary = base.neighbour < 0
    boundary_ledger = -float(dt_s)*float(np.sum(
        final_raw.face_water_flux_m2_s[boundary]*
        base.face_length_m[boundary]))
    water_residual = volume_change-boundary_ledger
    volume_scale = max(1.0, abs(volume_change),
                       float(np.sum(base.cell_area_m2*q0[0])))
    if abs(water_residual) > 2e-11*volume_scale:
        raise WetDryFwaveNativeV97Error("v97 accepted water ledger differs")
    density_rate = np.zeros((3, nc))
    density_rate[1:] = final_projection["momentum_rate_m2_s2"]
    turbulence_rate = (np.zeros((3, nc)) if final_turbulence is None else
                       np.array(final_turbulence.momentum_rate_m2_s2, copy=True))
    turbulence_flux = (np.zeros((2, execution.face_count))
                       if final_turbulence is None else
                       np.array(final_turbulence.face_flux_m4_s2, copy=True))
    cleanup_linf = float(np.max(np.abs(dry_momentum_cleanup)))
    outputs = (midpoint, endpoint, mask.astype(np.int32),
               accepted_mask.astype(np.int32),
               final_raw.face_water_flux_m2_s, final_raw.draining_theta,
               final_raw.base_fwave_rate, density_rate, turbulence_rate,
               turbulence_flux, accepted_total_rate,
               dry_momentum_cleanup)
    for value in outputs:
        value.setflags(write=False)
    placeholder = WetDryMidpointTrialV97(
        *outputs, final_projection["receipt_sha256"], restart,
        total_iterations, total_residuals, damping_count, total_limiter,
        total_riemann, total_residuals, total_turbulence, 0, 0,
        dry_momentum_cleanup_cell_count, cleanup_linf,
        projected_identity_linf, water_residual, residual_norm,
        execution.fingerprint, _digest(q0),
        _digest(bed), _digest(water), _digest(mixture), float(dt_s),
        final_raw, final_projection, final_turbulence, "")
    result = WetDryMidpointTrialV97(
        **{**placeholder.__dict__,
           "trial_sha256": _fingerprint(
               _midpoint_trial_payload(placeholder))})
    validate_wetdry_midpoint_trial_v97(
        result, execution, parent_q=q0, bed_elevation_m=bed,
        accepted_water_density_kg_m3=water,
        accepted_mixture_density_kg_m3=mixture, dt_s=dt_s)
    return result


def build_native_wetdry_fwave_runtime_v97(
    build_directory: str | Path, *, compiler: str | Path | None = None,
) -> NativeWetDryFwaveRuntimeV97:
    locked = tuple(sorted(verify_locked_sources_v85(_ROOT).items()))
    relatives = (_PARAMETER_RELATIVE, _RPN2_RELATIVE, _UTILS_RELATIVE,
                 _V95_RELATIVE, _SOURCE_RELATIVE)
    sources = [_ROOT/item for item in relatives]
    if not all(path.is_file() for path in sources):
        raise WetDryFwaveNativeV97Error("v97 native source inventory differs")
    hashes = {item: _sha(path) for item, path in zip(relatives, sources)}
    if any(hashes[key] != expected for key, expected in _EXPECTED_FROZEN.items()):
        raise WetDryFwaveNativeV97Error("v97 frozen augmented-fwave source differs")
    chosen = Path(compiler).resolve() if compiler is not None else None
    if chosen is None:
        found = shutil.which("gfortran")
        if found is None and os.name == "nt":
            fallback = Path(r"C:\Strawberry\c\bin\gfortran.exe")
            found = str(fallback) if fallback.is_file() else None
        if found is None:
            raise WetDryFwaveNativeV97Error("v97 native build requires gfortran")
        chosen = Path(found).resolve()
    if not chosen.is_file():
        raise WetDryFwaveNativeV97Error("v97 native compiler is unavailable")
    flags = ("-O3", "-flto", "-fopenmp", "-shared",
             "-ffree-line-length-none", "-ffixed-line-length-none",
             "-static-libgfortran", "-static-libgcc")
    root = Path(build_directory).resolve(); root.mkdir(parents=True, exist_ok=True)
    key = _fingerprint({"source": hashes, "compiler": _sha(chosen),
                        "flags": list(flags), "abi": NATIVE_ABI_ID})[:16]
    build = Path(tempfile.mkdtemp(prefix=f"v97_wetdry_{key}_", dir=root))
    name = "braids_wetdry_fwave_v97.dll" if os.name == "nt" else "libbraids_wetdry_fwave_v97.so"
    partial = build/(name+".partial"); library_path = build/name
    completed = subprocess.run(
        [str(chosen), *flags, *(str(path) for path in sources),
         "-o", str(partial)], cwd=build, capture_output=True, text=True,
        timeout=180, check=False)
    if completed.returncode != 0 or not partial.is_file():
        raise WetDryFwaveNativeV97Error(
            "v97 native build failed:\n"+completed.stdout+completed.stderr)
    if any(_sha(path) != hashes[item] for item, path in zip(relatives, sources)):
        raise WetDryFwaveNativeV97Error("v97 source changed during compilation")
    os.replace(partial, library_path)
    version = subprocess.run(
        [str(chosen), "--version"], capture_output=True, text=True,
        timeout=30, check=True).stdout.splitlines()[0]
    placeholder = NativeWetDryFwaveBuildReceiptV97(
        NATIVE_ABI_ID, hashes[_SOURCE_RELATIVE], _sha(Path(__file__)),
        hashes[_V95_RELATIVE], hashes[_PARAMETER_RELATIVE],
        hashes[_RPN2_RELATIVE], hashes[_UTILS_RELATIVE],
        _sha(library_path), version, flags,
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
            "+00:00", "Z"), LOCKED_UPSTREAM_COMMIT, locked, "")
    receipt = NativeWetDryFwaveBuildReceiptV97(
        **{**placeholder.__dict__,
           "receipt_sha256": _fingerprint(_build_payload(placeholder))})
    dependency = (os.add_dll_directory(str(chosen.parent))
                  if os.name == "nt" and hasattr(os, "add_dll_directory")
                  else None)
    library = ctypes.CDLL(str(library_path))
    function = library.braids_wetdry_fwave_rate_v97
    fa = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    ia = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
    function.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        fa, fa, ia, ia, fa, fa, fa, fa, ia, ia, ia, ia, ia, fa,
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_int, fa, fa, fa, fa, fa, fa, fa, fa, ia,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]
    function.restype = None
    runtime = NativeWetDryFwaveRuntimeV97(
        receipt, library_path, library, function, dependency)
    validate_native_wetdry_fwave_build_receipt_v97(receipt)
    return runtime


def method_identity_wetdry_fwave_v97() -> dict[str, object]:
    return {
        "scope_id": SCOPE_ID, "semantic_id": SEMANTIC_ID,
        "backend": "native_cpu_openmp",
        "augmented_fwave": "frozen_geoclaw_rpn2_unchanged",
        "active_set": "wetting_restart_drying_no_restart_monotone_policy",
        "draining": "authenticated_donor_limited_face_ledger",
        "endpoint_dry_momentum":
            "separate_authenticated_zero_projection_all_dry_cells",
        "residual_audit": "preprojection_implicit_and_projected_endpoint",
        "nonlinear_solver": "damped_picard_true_residual",
        "omitted_derivatives": ["draining_limiter", "wet_mask"],
        "explicit_fallback": False,
        "scalar_dry_transfer": "not_owned_by_this_hydraulic_primitive",
        "production_admission": False,
    }


__all__ = [
    "MAX_ACTIVE_SET_RESTARTS", "MAX_INNER_ITERATIONS",
    "NativeWetDryFwaveBuildReceiptV97", "NativeWetDryFwaveRateResultV97",
    "NativeWetDryFwaveRuntimeV97", "WetDryFwaveCPUExecutionV97",
    "WetDryFwaveNativeV97Error", "WetDryMidpointTrialV97",
    "build_native_wetdry_fwave_runtime_v97",
    "execution_document_wetdry_fwave_v97",
    "method_identity_wetdry_fwave_v97",
    "resolve_wetdry_fwave_cpu_execution_v97",
    "solve_wetdry_midpoint_trial_v97",
    "validate_wetdry_midpoint_trial_v97",
    "validate_native_wetdry_fwave_build_receipt_v97",
    "validate_native_wetdry_rate_result_v97",
]
