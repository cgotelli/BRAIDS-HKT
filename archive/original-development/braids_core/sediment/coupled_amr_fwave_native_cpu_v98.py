"""Native CPU augmented-f-wave residual with a typed AMR interface trace.

This is an additive v98 research adapter.  It reuses the unchanged v97
wet/dry residual and replaces only two authenticated negative-neighbour faces
with the coarse trace selected for one fine substep.  It returns the exact v97
rate-result type so the existing active-set iteration remains the sole
hydraulic time integrator, while retaining a separate ordered audit of every
external-trace evaluation.  An AMR trace is never a physical boundary owner.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass, replace
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
from typing import Callable

import numpy as np

from .coupled_amr_trace_cpu_v98 import (
    CoupledAMRInternalTraceCPUV98,
    validate_coupled_amr_internal_trace_structure_cpu_v98,
)
from .coupled_amr_turbulence_trace_cpu_v98 import (
    CoupledAMRTurbulenceTraceCPUV98,
    validate_coupled_amr_turbulence_trace_structure_cpu_v98,
)
from .coupled_geometry_native_cpu_v96 import DENSITY_WATER_PLUS_SEDIMENT
from .density_active_projection_v97 import (
    resolve_density_projection_geometry_v97,
)
from .density_trace_faces_v98 import (
    NativeDensityTraceFacesRuntimeV98,
    validate_density_trace_faces_result_v98,
)
from .density_trace_projection_v98 import (
    resolve_density_trace_geometry_v98,
)
from .densin_fwave_v85 import LOCKED_UPSTREAM_COMMIT, verify_locked_sources_v85
from .momentum_trace_v98 import (
    NativeMomentumTraceRuntimeV98,
)
from .shared_turbulence_native_cpu_v97 import (
    NativeMomentumRateV97,
    NativeSharedTurbulenceRuntimeV97,
    PreparedNativeTurbulenceV97,
)
from .wetdry_fwave_native_cpu_v97 import (
    NativeWetDryFwaveRateResultV97,
    WetDryFwaveCPUExecutionV97,
    _GE0CLAW_CALL_LOCK,
    _result_payload as _v97_result_payload,
    execution_document_wetdry_fwave_v97,
    validate_native_wetdry_rate_result_v97,
)


SCHEMA = 1
SCOPE_ID = "braids.q4.coupled_amr.interface_fwave.native_cpu.v98"
NATIVE_ABI_ID = SCOPE_ID + ":openmp-trace-face-density-viscosity-v4"
INTERFACE_FACE_COUNT = 2
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_RELATIVE = "solver/wave/braids_amr_interface_fwave_v98_module.f90"
_V97_RELATIVE = "solver/wave/braids_wetdry_fwave_v97_module.f90"
_V95_RELATIVE = "solver/wave/braids_densin_residual_v95_module.f90"
_PARAMETER_RELATIVE = "solver/cut_cells/geoclaw_parameters.f90"
_RPN2_RELATIVE = "solver/clawpack-v5.14.0/riemann/src/rpn2_geoclaw.f"
_UTILS_RELATIVE = "solver/clawpack-v5.14.0/riemann/src/geoclaw_riemann_utils.f"
_EXPECTED_FROZEN = {
    _V97_RELATIVE:
        "f6fd6aedfda5e745464f3b98ef87f817481b52ee63fe8ff1bc424b39d4d913d1",
    _V95_RELATIVE:
        "e701c096e400402c4386cc2015459d5900ce073094b86f97c0ad94c2516a61e3",
    _PARAMETER_RELATIVE:
        "b6e5a70595e93279eac113401dfc2d857978f7a6e885fe730c783c17cc5b3c1d",
    _RPN2_RELATIVE:
        "c52649def884ce374dbb1d3dc1e91f7a9c33aa45a0aff691badc8811b9157d1f",
    _UTILS_RELATIVE:
        "df73207446ac730f9cda9a8ca4dbbc625151ba95d13aa514ce87ca23cb598761",
}


class CoupledAMRFwaveNativeCPUV98Error(ValueError):
    """The v98 native interface residual or its audit is invalid."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError, OverflowError) as error:
        raise CoupledAMRFwaveNativeCPUV98Error(
            "v98 interface residual identity is not canonical JSON") from error


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _digest(*values: object) -> str:
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
    if (raw.shape != shape or raw.dtype.kind not in "fiub" or
            not np.isfinite(raw).all() or
            (integer and np.any(raw != np.floor(raw)))):
        raise CoupledAMRFwaveNativeCPUV98Error(
            f"v98 interface residual {name} differs")
    return np.array(
        raw, dtype=np.int32 if integer else np.float64,
        order="C", copy=True)


def _cancel(cancelled: bool | Callable[[], bool]) -> Callable[[], bool]:
    if type(cancelled) is bool:
        return lambda: cancelled
    if callable(cancelled):
        return cancelled
    raise CoupledAMRFwaveNativeCPUV98Error(
        "v98 interface residual cancellation must be bool or callable")


@dataclass(frozen=True)
class NativeAMRInterfaceFwaveBuildReceiptV98:
    abi_id: str
    source_sha256: str
    wrapper_sha256: str
    frozen_v97_sha256: str
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
    density_trace_runtime_sha256: str
    turbulence_trace_receipt_sha256: str
    momentum_trace_runtime_sha256: str
    shared_turbulence_runtime_sha256: str
    shared_turbulence_preparation_sha256: str
    receipt_sha256: str


def _build_payload(value: NativeAMRInterfaceFwaveBuildReceiptV98) -> dict:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID + ":build",
        "abi_id": value.abi_id,
        "source_sha256": value.source_sha256,
        "wrapper_sha256": value.wrapper_sha256,
        "frozen_v97_sha256": value.frozen_v97_sha256,
        "frozen_v95_sha256": value.frozen_v95_sha256,
        "geoclaw_parameters_sha256": value.geoclaw_parameters_sha256,
        "augmented_rpn2_sha256": value.augmented_rpn2_sha256,
        "riemann_utils_sha256": value.riemann_utils_sha256,
        "library_sha256": value.library_sha256,
        "compiler_identity": value.compiler_identity,
        "compile_flags": list(value.compile_flags),
        "build_utc": value.build_utc,
        "locked_delft_commit": value.locked_delft_commit,
        "locked_delft_source_sha256": dict(
            value.locked_delft_source_sha256),
        "density_trace_runtime_sha256":
            value.density_trace_runtime_sha256,
        "turbulence_trace_receipt_sha256":
            value.turbulence_trace_receipt_sha256,
        "momentum_trace_runtime_sha256":
            value.momentum_trace_runtime_sha256,
        "shared_turbulence_runtime_sha256":
            value.shared_turbulence_runtime_sha256,
        "shared_turbulence_preparation_sha256":
            value.shared_turbulence_preparation_sha256,
        "frozen_v97_residual_reused": True,
        "physical_boundary_owner_count": 0,
    }


def validate_native_amr_interface_build_receipt_v98(
    value: NativeAMRInterfaceFwaveBuildReceiptV98,
) -> None:
    if type(value) is not NativeAMRInterfaceFwaveBuildReceiptV98:
        raise CoupledAMRFwaveNativeCPUV98Error(
            "v98 interface build receipt type differs")
    relatives = (_SOURCE_RELATIVE, _V97_RELATIVE, _V95_RELATIVE,
                 _PARAMETER_RELATIVE, _RPN2_RELATIVE, _UTILS_RELATIVE)
    current = {item: _sha(_ROOT/item) for item in relatives}
    locked = tuple(sorted(verify_locked_sources_v85(_ROOT).items()))
    actual = (value.frozen_v97_sha256, value.frozen_v95_sha256,
              value.geoclaw_parameters_sha256,
              value.augmented_rpn2_sha256, value.riemann_utils_sha256)
    expected = tuple(_EXPECTED_FROZEN[item] for item in relatives[1:])
    if (any(current[item] != expected_hash
            for item, expected_hash in _EXPECTED_FROZEN.items()) or
            actual != expected or
            value.source_sha256 != current[_SOURCE_RELATIVE] or
            value.wrapper_sha256 != _sha(Path(__file__)) or
            value.abi_id != NATIVE_ABI_ID or
            _SHA.fullmatch(value.library_sha256) is None or
            not value.compiler_identity or not value.compile_flags or
            not value.build_utc.endswith("Z") or
            value.locked_delft_commit != LOCKED_UPSTREAM_COMMIT or
            value.locked_delft_source_sha256 != locked or
            _SHA.fullmatch(value.density_trace_runtime_sha256) is None or
            _SHA.fullmatch(value.turbulence_trace_receipt_sha256) is None or
            _SHA.fullmatch(value.momentum_trace_runtime_sha256) is None or
            _SHA.fullmatch(value.shared_turbulence_runtime_sha256) is None or
            _SHA.fullmatch(value.shared_turbulence_preparation_sha256) is None or
            value.receipt_sha256 != _fingerprint(_build_payload(value))):
        raise CoupledAMRFwaveNativeCPUV98Error(
            "v98 interface build/source receipt is stale or mismatched")


@dataclass(frozen=True)
class NativeAMRInterfaceRateReceiptV98:
    execution_fingerprint: str
    trace_receipt_sha256: str
    substep_index: int
    evaluation_index: int
    midpoint_sha256: str
    trace_slice_sha256: str
    v97_result_sha256: str
    interface_faces: tuple[int, int]
    interface_riemann_call_count: int
    discarded_reflective_interface_riemann_call_count: int
    total_native_riemann_call_count: int
    density_trace_result_sha256: str
    density_trace_augmentation_application_count: int
    density_trace_projection_application_count: int
    density_trace_face_evaluation_count: int
    internal_turbulence_result_sha256: str
    trace_turbulence_result_sha256: str
    internal_turbulence_application_count: int
    trace_turbulence_application_count: int
    trace_viscosity_face_evaluation_count: int
    turbulence_cap_application_count: int
    physical_boundary_owner_count: int
    receipt_sha256: str


def _rate_receipt_payload(value: NativeAMRInterfaceRateReceiptV98) -> dict:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID + ":rate-receipt",
        "execution_fingerprint": value.execution_fingerprint,
        "trace_receipt_sha256": value.trace_receipt_sha256,
        "substep_index": value.substep_index,
        "evaluation_index": value.evaluation_index,
        "midpoint_sha256": value.midpoint_sha256,
        "trace_slice_sha256": value.trace_slice_sha256,
        "v97_result_sha256": value.v97_result_sha256,
        "interface_faces": list(value.interface_faces),
        "interface_riemann_call_count":
            value.interface_riemann_call_count,
        "discarded_reflective_interface_riemann_call_count":
            value.discarded_reflective_interface_riemann_call_count,
        "total_native_riemann_call_count":
            value.total_native_riemann_call_count,
        "density_trace_result_sha256":
            value.density_trace_result_sha256,
        "density_trace_augmentation_application_count":
            value.density_trace_augmentation_application_count,
        "density_trace_projection_application_count":
            value.density_trace_projection_application_count,
        "density_trace_face_evaluation_count":
            value.density_trace_face_evaluation_count,
        "internal_turbulence_result_sha256":
            value.internal_turbulence_result_sha256,
        "trace_turbulence_result_sha256":
            value.trace_turbulence_result_sha256,
        "internal_turbulence_application_count":
            value.internal_turbulence_application_count,
        "trace_turbulence_application_count":
            value.trace_turbulence_application_count,
        "trace_viscosity_face_evaluation_count":
            value.trace_viscosity_face_evaluation_count,
        "turbulence_cap_application_count":
            value.turbulence_cap_application_count,
        "physical_boundary_owner_count":
            value.physical_boundary_owner_count,
    }


@dataclass(frozen=True)
class AMRInterfaceResidualAuditV98:
    execution_fingerprint: str
    trace_receipt_sha256: str
    substep_index: int
    interface_faces: tuple[int, int]
    rate_receipts: tuple[NativeAMRInterfaceRateReceiptV98, ...]
    evaluation_count: int
    interface_riemann_call_count: int
    discarded_reflective_interface_riemann_call_count: int
    total_native_riemann_call_count: int
    density_trace_augmentation_application_count: int
    density_trace_projection_application_count: int
    density_trace_face_evaluation_count: int
    internal_turbulence_application_count: int
    trace_turbulence_application_count: int
    trace_viscosity_face_evaluation_count: int
    turbulence_cap_application_count: int
    physical_boundary_owner_count: int
    boundary_memory_consumer_count: int
    accepted_clock_increment_count: int
    audit_sha256: str


def _audit_payload(value: AMRInterfaceResidualAuditV98) -> dict:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID + ":residual-audit",
        "execution_fingerprint": value.execution_fingerprint,
        "trace_receipt_sha256": value.trace_receipt_sha256,
        "substep_index": value.substep_index,
        "interface_faces": list(value.interface_faces),
        "rate_receipts": [item.receipt_sha256
                          for item in value.rate_receipts],
        "evaluation_count": value.evaluation_count,
        "interface_riemann_call_count":
            value.interface_riemann_call_count,
        "discarded_reflective_interface_riemann_call_count":
            value.discarded_reflective_interface_riemann_call_count,
        "total_native_riemann_call_count":
            value.total_native_riemann_call_count,
        "density_trace_augmentation_application_count":
            value.density_trace_augmentation_application_count,
        "density_trace_projection_application_count":
            value.density_trace_projection_application_count,
        "density_trace_face_evaluation_count":
            value.density_trace_face_evaluation_count,
        "internal_turbulence_application_count":
            value.internal_turbulence_application_count,
        "trace_turbulence_application_count":
            value.trace_turbulence_application_count,
        "trace_viscosity_face_evaluation_count":
            value.trace_viscosity_face_evaluation_count,
        "turbulence_cap_application_count":
            value.turbulence_cap_application_count,
        "physical_boundary_owner_count":
            value.physical_boundary_owner_count,
        "boundary_memory_consumer_count":
            value.boundary_memory_consumer_count,
        "accepted_clock_increment_count":
            value.accepted_clock_increment_count,
    }


def validate_amr_interface_residual_audit_v98(
    value: AMRInterfaceResidualAuditV98,
) -> None:
    if (type(value) is not AMRInterfaceResidualAuditV98 or
            value.evaluation_count != len(value.rate_receipts) or
            value.interface_riemann_call_count !=
                INTERFACE_FACE_COUNT*value.evaluation_count or
            value.discarded_reflective_interface_riemann_call_count !=
                INTERFACE_FACE_COUNT*value.evaluation_count or
            value.total_native_riemann_call_count !=
                sum(item.total_native_riemann_call_count
                    for item in value.rate_receipts) or
            value.density_trace_augmentation_application_count !=
                value.evaluation_count or
            value.density_trace_projection_application_count != 0 or
            value.density_trace_face_evaluation_count !=
                INTERFACE_FACE_COUNT*value.evaluation_count or
            value.internal_turbulence_application_count !=
                value.evaluation_count or
            value.trace_turbulence_application_count !=
                value.evaluation_count or
            value.trace_viscosity_face_evaluation_count !=
                INTERFACE_FACE_COUNT*value.evaluation_count or
            value.turbulence_cap_application_count != 0 or
            value.physical_boundary_owner_count != 0 or
            value.boundary_memory_consumer_count != 0 or
            value.accepted_clock_increment_count != 0 or
            value.audit_sha256 != _fingerprint(_audit_payload(value)) or
            any(type(item) is not NativeAMRInterfaceRateReceiptV98 or
                item.evaluation_index != index or
                item.execution_fingerprint != value.execution_fingerprint or
                item.trace_receipt_sha256 != value.trace_receipt_sha256 or
                item.substep_index != value.substep_index or
                item.interface_faces != value.interface_faces or
                item.interface_riemann_call_count != INTERFACE_FACE_COUNT or
                item.discarded_reflective_interface_riemann_call_count !=
                    INTERFACE_FACE_COUNT or
                item.total_native_riemann_call_count <=
                    item.interface_riemann_call_count or
                _SHA.fullmatch(item.density_trace_result_sha256) is None or
                item.density_trace_augmentation_application_count != 1 or
                item.density_trace_projection_application_count != 0 or
                item.density_trace_face_evaluation_count !=
                    INTERFACE_FACE_COUNT or
                _SHA.fullmatch(item.internal_turbulence_result_sha256) is None or
                _SHA.fullmatch(item.trace_turbulence_result_sha256) is None or
                item.internal_turbulence_application_count != 1 or
                item.trace_turbulence_application_count != 1 or
                item.trace_viscosity_face_evaluation_count !=
                    INTERFACE_FACE_COUNT or
                item.turbulence_cap_application_count != 0 or
                item.physical_boundary_owner_count != 0 or
                item.receipt_sha256 !=
                    _fingerprint(_rate_receipt_payload(item))
                for index, item in enumerate(value.rate_receipts))):
        raise CoupledAMRFwaveNativeCPUV98Error(
            "v98 interface residual audit identity differs")


class NativeAMRInterfaceFwaveRuntimeV98:
    """Exact v97-rate producer with one typed external coarse trace."""

    __slots__ = (
        "receipt", "library_path", "_library", "_function", "_dependency",
        "execution", "trace", "substep_index", "_rate_receipts",
        "_audit_lock", "_coupled_execution", "density_trace_runtime",
        "density_trace_geometry", "turbulence_trace",
        "momentum_trace_runtime", "shared_turbulence_runtime",
        "shared_turbulence_prepared", "_turbulence_results")

    def __init__(
        self, receipt: NativeAMRInterfaceFwaveBuildReceiptV98,
        library_path: Path, library, function, dependency, *,
        execution: WetDryFwaveCPUExecutionV97,
        coupled_execution: object,
        trace: CoupledAMRInternalTraceCPUV98,
        substep_index: int,
        density_trace_runtime: NativeDensityTraceFacesRuntimeV98,
        turbulence_trace: CoupledAMRTurbulenceTraceCPUV98,
        momentum_trace_runtime: NativeMomentumTraceRuntimeV98,
        shared_turbulence_runtime: NativeSharedTurbulenceRuntimeV97,
        shared_turbulence_prepared: PreparedNativeTurbulenceV97,
    ) -> None:
        from .coupled_wetdry_owner_cpu_v97 import (
            CoupledWetDryCPUExecutionV97,
            _grid_execution,
            _v92_execution,
        )

        validate_native_amr_interface_build_receipt_v98(receipt)
        execution_document_wetdry_fwave_v97(execution)
        if (type(coupled_execution) is not CoupledWetDryCPUExecutionV97 or
                coupled_execution.hydraulic_execution is not execution):
            raise CoupledAMRFwaveNativeCPUV98Error(
                "v98 interface coupled/hydraulic execution binding differs")
        validate_coupled_amr_internal_trace_structure_cpu_v98(
            trace, coupled_execution)
        if type(substep_index) is not int or substep_index not in (0, 1):
            raise CoupledAMRFwaveNativeCPUV98Error(
                "v98 interface substep must be zero or one")
        if (type(density_trace_runtime) is not
                NativeDensityTraceFacesRuntimeV98 or
                receipt.density_trace_runtime_sha256 !=
                density_trace_runtime.receipt["receipt_sha256"]):
            raise CoupledAMRFwaveNativeCPUV98Error(
                "v98 interface density-trace runtime binding differs")
        validate_coupled_amr_turbulence_trace_structure_cpu_v98(
            turbulence_trace, coupled_execution, trace)
        expected_v91 = _v92_execution(
            coupled_execution.active_execution).base_execution
        if (type(momentum_trace_runtime) is not
                NativeMomentumTraceRuntimeV98 or
                type(shared_turbulence_runtime) is not
                NativeSharedTurbulenceRuntimeV97 or
                type(shared_turbulence_prepared) is not
                PreparedNativeTurbulenceV97 or
                receipt.turbulence_trace_receipt_sha256 !=
                    turbulence_trace.receipt_sha256 or
                receipt.momentum_trace_runtime_sha256 !=
                    momentum_trace_runtime.receipt["receipt_sha256"] or
                receipt.shared_turbulence_runtime_sha256 !=
                    shared_turbulence_runtime.receipt["receipt_sha256"] or
                receipt.shared_turbulence_preparation_sha256 !=
                    shared_turbulence_prepared.preparation_sha256 or
                shared_turbulence_prepared.native_runtime_sha256 !=
                    shared_turbulence_runtime.receipt["receipt_sha256"] or
                shared_turbulence_prepared.execution.fingerprint !=
                    expected_v91.fingerprint):
            raise CoupledAMRFwaveNativeCPUV98Error(
                "v98 interface viscosity trace/runtime binding differs")
        base = execution.base_execution.base_execution
        grid = _grid_execution(coupled_execution.active_execution)
        if (not np.array_equal(base.owner, grid.owner) or
                not np.array_equal(base.neighbour, grid.neighbour)):
            raise CoupledAMRFwaveNativeCPUV98Error(
                "v98 interface hydraulic/scalar topology differs")
        projection_geometry = resolve_density_projection_geometry_v97(
            base.normal, base.face_length_m, base.cell_face_offset,
            base.cell_faces, base.cell_face_sign)
        density_geometry = resolve_density_trace_geometry_v98(
            projection_geometry, base.owner, base.neighbour,
            grid.boundary_index)
        self.receipt = receipt
        self.library_path = Path(library_path).resolve()
        self._library = library
        self._function = function
        self._dependency = dependency
        self.execution = execution
        self._coupled_execution = coupled_execution
        self.trace = trace
        self.substep_index = substep_index
        self.density_trace_runtime = density_trace_runtime
        self.density_trace_geometry = density_geometry
        self.turbulence_trace = turbulence_trace
        self.momentum_trace_runtime = momentum_trace_runtime
        self.shared_turbulence_runtime = shared_turbulence_runtime
        self.shared_turbulence_prepared = shared_turbulence_prepared
        self._rate_receipts: list[NativeAMRInterfaceRateReceiptV98] = []
        self._turbulence_results: list[
            tuple[NativeMomentumRateV97, dict[str, object]]] = []
        self._audit_lock = threading.Lock()

    def reset_residual_audit(self) -> None:
        with self._audit_lock:
            self._rate_receipts.clear()
            self._turbulence_results.clear()

    def turbulence_results(
        self,
    ) -> tuple[tuple[NativeMomentumRateV97, dict[str, object]], ...]:
        with self._audit_lock:
            return tuple(self._turbulence_results)

    def residual_audit(self) -> AMRInterfaceResidualAuditV98:
        with self._audit_lock:
            receipts = tuple(self._rate_receipts)
        placeholder = AMRInterfaceResidualAuditV98(
            self.execution.fingerprint, self.trace.receipt_sha256,
            self.substep_index, tuple(self.trace.fine_faces), receipts,
            len(receipts), sum(item.interface_riemann_call_count
                               for item in receipts),
            sum(item.discarded_reflective_interface_riemann_call_count
                for item in receipts),
            sum(item.total_native_riemann_call_count
                for item in receipts),
            sum(item.density_trace_augmentation_application_count
                for item in receipts),
            sum(item.density_trace_projection_application_count
                for item in receipts),
            sum(item.density_trace_face_evaluation_count
                for item in receipts),
            sum(item.internal_turbulence_application_count
                for item in receipts),
            sum(item.trace_turbulence_application_count
                for item in receipts),
            sum(item.trace_viscosity_face_evaluation_count
                for item in receipts),
            sum(item.turbulence_cap_application_count
                for item in receipts), 0, 0, 0, "")
        result = AMRInterfaceResidualAuditV98(
            **{**placeholder.__dict__,
               "audit_sha256": _fingerprint(_audit_payload(placeholder))})
        validate_amr_interface_residual_audit_v98(result)
        return result

    def evaluate(
        self, execution: WetDryFwaveCPUExecutionV97,
        midpoint_q: object, *, bed_elevation_m: object,
        accepted_water_density_kg_m3: object,
        accepted_mixture_density_kg_m3: object,
        frozen_wet_mask: object, dt_s: float,
        cancelled: bool | Callable[[], bool] = False,
    ) -> NativeWetDryFwaveRateResultV97:
        execution_document_wetdry_fwave_v97(execution)
        if (execution is not self.execution or
                execution.fingerprint != self.execution.fingerprint):
            raise CoupledAMRFwaveNativeCPUV98Error(
                "v98 interface runtime execution differs")
        validate_native_amr_interface_build_receipt_v98(self.receipt)
        validate_coupled_amr_internal_trace_structure_cpu_v98(
            self.trace, self._coupled_execution)
        validate_coupled_amr_turbulence_trace_structure_cpu_v98(
            self.turbulence_trace, self._coupled_execution, self.trace)
        if (not self.library_path.is_file() or
                _sha(self.library_path) != self.receipt.library_sha256):
            raise CoupledAMRFwaveNativeCPUV98Error(
                "v98 interface native library differs")
        poll = _cancel(cancelled)
        if poll():
            raise InterruptedError(
                "v98 interface residual cancelled before launch")
        base = execution.base_execution.base_execution
        nc, nf = execution.cell_count, execution.face_count
        q = _array(midpoint_q, (3, nc), "midpoint q")
        bed = _array(bed_elevation_m, (nc,), "accepted bed")
        water = _array(
            accepted_water_density_kg_m3, (nc,), "water density")
        mixture = _array(
            accepted_mixture_density_kg_m3, (nc,), "mixture density")
        wet = _array(
            frozen_wet_mask, (nc,), "frozen wet mask", integer=True)
        if (np.any(q[0] < 0.0) or np.any(water <= 0.0) or
                np.any(mixture < water) or
                np.any((wet != 0) & (wet != 1)) or
                np.any((wet == 1) & (q[0] <= 0.0)) or
                isinstance(dt_s, bool) or not math.isfinite(float(dt_s)) or
                not 0.0 < float(dt_s) <=
                    execution.base_execution.maximum_dt_s):
            raise CoupledAMRFwaveNativeCPUV98Error(
                "v98 interface rate physical inputs differ")
        density = (mixture if execution.base_execution.density_mode ==
                   DENSITY_WATER_PLUS_SEDIMENT else water)
        active, trace_q, trace_bed, trace_density, trace_wet = \
            self._expanded_trace(density_reference=float(np.min(density)))
        trace_faces = np.asarray(self.trace.fine_faces, dtype=np.int64)
        # The trace carries coarse cell-centre donor values.  Therefore its
        # normal distance is coarse-half plus fine-half, not the fine patch's
        # negative-neighbour physical-boundary half distance.  Replace only
        # the two authenticated trace entries; all ordinary faces retain the
        # exact v97 geometry image.
        coupled_face_distance = np.array(
            base.face_distance_m, dtype=np.float64, order="C", copy=True)
        coupled_face_distance[trace_faces] = \
            self.trace.interface_centre_distance_m
        with self._audit_lock:
            evaluation_index = len(self._rate_receipts)+1
        arrays = [np.empty((3, nf)) for _ in range(4)]
        theta = np.empty(nc)
        rate = np.empty((3, nc))
        water_flux = np.empty(nf)
        acceleration = np.empty(nf)
        wet_wet = np.empty(nf, dtype=np.int32)
        riemann = ctypes.c_int(-1)
        limiter = ctypes.c_int(-1)
        interface_calls = ctypes.c_int(-1)
        status = ctypes.c_int(-1)
        with _GE0CLAW_CALL_LOCK:
            self._function(
                nc, nf, base.cell_faces.size,
                np.ascontiguousarray(q), np.ascontiguousarray(bed),
                np.ascontiguousarray(base.owner, dtype=np.int32),
                np.ascontiguousarray(base.neighbour, dtype=np.int32),
                np.ascontiguousarray(base.normal),
                np.ascontiguousarray(base.face_length_m),
                np.ascontiguousarray(coupled_face_distance),
                np.ascontiguousarray(base.cell_area_m2),
                np.ascontiguousarray(base.cell_face_offset, dtype=np.int32),
                np.ascontiguousarray(base.cell_faces, dtype=np.int32),
                np.ascontiguousarray(base.cell_face_sign, dtype=np.int32),
                np.ascontiguousarray(
                    execution.domain_active_mask, dtype=np.int32),
                np.ascontiguousarray(wet, dtype=np.int32),
                np.ascontiguousarray(density),
                base.hydraulic_reference_density_kg_m3,
                base.gravity_m_s2, base.dry_tolerance_m, float(dt_s),
                execution.native_threads, active, trace_q, trace_bed,
                trace_density, trace_wet, *arrays, theta, rate,
                water_flux, acceleration, wet_wet,
                ctypes.byref(riemann), ctypes.byref(limiter),
                ctypes.byref(interface_calls), ctypes.byref(status))
        if status.value != 0:
            raise CoupledAMRFwaveNativeCPUV98Error(
                "v98 native interface rate failed with status "
                f"{status.value}")
        if poll():
            raise InterruptedError(
                "v98 interface residual cancelled after launch")
        if (riemann.value != nf or interface_calls.value !=
                INTERFACE_FACE_COUNT or limiter.value < 0):
            raise CoupledAMRFwaveNativeCPUV98Error(
                "v98 native interface owner counts differ")
        base_arrays = (*arrays, theta, rate, water_flux, acceleration,
                       wet_wet)
        if not all(np.isfinite(item).all() for item in base_arrays):
            raise CoupledAMRFwaveNativeCPUV98Error(
                "v98 native interface result is non-finite")
        if self.shared_turbulence_prepared.cap_receipt.dt_s != float(dt_s):
            raise CoupledAMRFwaveNativeCPUV98Error(
                "v98 fine-cell turbulence cap timestep differs")
        internal_turbulence = self.shared_turbulence_runtime.evaluate(
            self.shared_turbulence_prepared, q, wet, wet_wet,
            native_threads=execution.native_threads, cancelled=poll)
        trace_turbulence = self.momentum_trace_runtime.evaluate(
            self.density_trace_geometry, q,
            self.shared_turbulence_prepared.cap_receipt.
                capped_total_vicuv_m2_s,
            cell_area=base.cell_area_m2,
            face_distance=coupled_face_distance,
            wet_cell_mask=wet,
            trace_faces=trace_faces,
            trace_q=trace_q[:, trace_faces],
            trace_capped_vicuv=self.turbulence_trace.
                trace_capped_vicuv_m2_s[self.substep_index],
            trace_wet_mask=trace_wet[trace_faces],
            trace_receipt_sha256=self.turbulence_trace.receipt_sha256,
            trace_source_sha256=
                self.turbulence_trace.coarse_provisional_trial_sha256,
            trace_runtime_sha256=self.receipt.receipt_sha256,
            cap_receipt_sha256=self.shared_turbulence_prepared.
                cap_receipt.receipt_sha256,
            trace_cap_receipt_sha256=self.turbulence_trace.
                coarse_cap_receipt_sha256,
            evaluation_index=evaluation_index,
            native_threads=execution.native_threads,
            cancelled=poll)
        rate += internal_turbulence.momentum_rate_m2_s2
        rate[1:] += trace_turbulence["momentum_rate_m2_s2"]
        density_trace = self.density_trace_runtime.evaluate(
            self.density_trace_geometry, q[0], density,
            reference_density=base.hydraulic_reference_density_kg_m3,
            gravity=base.gravity_m_s2,
            face_distance=coupled_face_distance,
            base_face_acceleration=acceleration,
            wet_cell_mask=wet,
            base_density_face_mask=wet_wet,
            trace_faces=trace_faces,
            trace_depth=trace_q[0, trace_faces],
            trace_density=trace_density[trace_faces],
            trace_wet_mask=trace_wet[trace_faces],
            trace_receipt_sha256=self.trace.receipt_sha256,
            trace_source_sha256=
                self.trace.coarse_provisional_state_sha256,
            trace_runtime_sha256=self.receipt.receipt_sha256,
            evaluation_index=evaluation_index,
            native_threads=execution.native_threads, cancelled=poll)
        validate_density_trace_faces_result_v98(
            density_trace, self.density_trace_geometry, q[0], density,
            runtime_sha=
                self.density_trace_runtime.receipt["receipt_sha256"],
            reference_density=base.hydraulic_reference_density_kg_m3,
            gravity=base.gravity_m_s2,
            face_distance=coupled_face_distance,
            base_face_acceleration=acceleration,
            wet_cell_mask=wet,
            base_density_face_mask=wet_wet,
            trace_faces=trace_faces,
            trace_depth=trace_q[0, trace_faces],
            trace_density=trace_density[trace_faces],
            trace_wet_mask=trace_wet[trace_faces],
            trace_receipt_sha256=self.trace.receipt_sha256,
            trace_source_sha256=
                self.trace.coarse_provisional_state_sha256,
            trace_runtime_sha256=self.receipt.receipt_sha256,
            evaluation_index=evaluation_index,
            native_threads=execution.native_threads, cancelled=False)
        # The historical v97 midpoint owner performs exactly one projection
        # after this rate producer returns.  Supply the combined internal plus
        # trace face equation/mask to that single solve; projecting two masks
        # independently and adding their cell rates is not algebraically
        # equivalent for rank-aware least-squares cells.
        acceleration = np.array(
            density_trace["face_acceleration_m_s2"], dtype=np.float64,
            order="C", copy=True)
        wet_wet = np.array(
            density_trace["density_face_mask"], dtype=np.int32,
            order="C", copy=True)
        all_arrays = (*arrays, theta, rate, water_flux, acceleration,
                      wet_wet)
        if not all(np.isfinite(item).all() for item in all_arrays):
            raise CoupledAMRFwaveNativeCPUV98Error(
                "v98 density-trace augmented face equation is non-finite")
        slots_per_cell = np.diff(base.cell_face_offset)
        cell = np.repeat(np.arange(nc), slots_per_cell)
        face = base.cell_faces
        signed = (base.cell_face_sign*water_flux[face]*
                  base.face_length_m[face])
        rebuilt = -np.bincount(
            cell, weights=signed, minlength=nc)/base.cell_area_m2
        if not np.allclose(
                rate[0], rebuilt, rtol=64*np.finfo(float).eps,
                atol=64*np.finfo(float).eps):
            raise CoupledAMRFwaveNativeCPUV98Error(
                "v98 interface limited water ledger differs")
        for item in all_arrays:
            item.setflags(write=False)
        # ``wet`` is carried in the exact v97 result identity even though it is
        # an input rather than a native output.  Freeze this copied image just
        # like the historical v97 runtime does before authenticating it.
        wet.setflags(write=False)
        placeholder = NativeWetDryFwaveRateResultV97(
            arrays[0], arrays[1], arrays[2], arrays[3], theta, rate,
            water_flux, acceleration, wet_wet, wet, _digest(q),
            _digest(bed), _digest(density), float(dt_s), riemann.value,
            limiter.value, 0, 0, execution.native_threads, "")
        result = NativeWetDryFwaveRateResultV97(
            **{**placeholder.__dict__,
               "result_sha256": _fingerprint(
                   _v97_result_payload(placeholder, execution))})
        validate_native_wetdry_rate_result_v97(result, execution)
        trace_slice = _digest(
            active, trace_q, trace_bed, trace_density, trace_wet)
        with self._audit_lock:
            index = len(self._rate_receipts)
            if evaluation_index != index+1:
                raise CoupledAMRFwaveNativeCPUV98Error(
                    "v98 concurrent residual audit order differs")
            receipt0 = NativeAMRInterfaceRateReceiptV98(
                execution.fingerprint, self.trace.receipt_sha256,
                self.substep_index, index, result.midpoint_sha256,
                trace_slice, result.result_sha256,
                tuple(self.trace.fine_faces), interface_calls.value,
                interface_calls.value, riemann.value+interface_calls.value,
                density_trace["receipt_sha256"], 1, 0,
                INTERFACE_FACE_COUNT,
                internal_turbulence.result_sha256,
                trace_turbulence["receipt_sha256"], 1, 1,
                INTERFACE_FACE_COUNT, 0, 0, "")
            receipt = NativeAMRInterfaceRateReceiptV98(
                **{**receipt0.__dict__, "receipt_sha256": _fingerprint(
                    _rate_receipt_payload(receipt0))})
            self._rate_receipts.append(receipt)
            self._turbulence_results.append(
                (internal_turbulence, trace_turbulence))
        return result

    def _expanded_trace(
        self, *, density_reference: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        nf = self.execution.face_count
        active = np.zeros(nf, dtype=np.int32)
        q = np.zeros((3, nf), dtype=np.float64)
        bed = np.zeros(nf, dtype=np.float64)
        density = np.full(nf, density_reference, dtype=np.float64)
        wet = np.zeros(nf, dtype=np.int32)
        source_density = (
            self.trace.trace_mixture_density_kg_m3
            if self.execution.base_execution.density_mode ==
                DENSITY_WATER_PLUS_SEDIMENT
            else self.trace.trace_water_density_kg_m3)
        for slot, face in enumerate(self.trace.fine_faces):
            active[face] = 1
            q[:, face] = self.trace.trace_q[self.substep_index, :, slot]
            bed[face] = self.trace.trace_bed_elevation_m[
                self.substep_index, slot]
            density[face] = source_density[self.substep_index, slot]
            wet[face] = self.trace.trace_wet_mask[self.substep_index, slot]
        return tuple(np.ascontiguousarray(item)
                     for item in (active, q, bed, density, wet))


def build_native_amr_interface_fwave_runtime_v98(
    build_directory: str | Path, *,
    execution: WetDryFwaveCPUExecutionV97,
    coupled_execution: object,
    trace: CoupledAMRInternalTraceCPUV98,
    substep_index: int,
    density_trace_runtime: NativeDensityTraceFacesRuntimeV98,
    turbulence_trace: CoupledAMRTurbulenceTraceCPUV98,
    momentum_trace_runtime: NativeMomentumTraceRuntimeV98,
    shared_turbulence_runtime: NativeSharedTurbulenceRuntimeV97,
    shared_turbulence_prepared: PreparedNativeTurbulenceV97,
    compiler: str | Path | None = None,
) -> NativeAMRInterfaceFwaveRuntimeV98:
    """Build and bind one exact substep adapter.

    ``coupled_execution`` is exact-type checked lazily to avoid a circular
    import at module import time.  It must own ``execution`` by object identity.
    """
    from .coupled_wetdry_owner_cpu_v97 import CoupledWetDryCPUExecutionV97

    if (type(coupled_execution) is not CoupledWetDryCPUExecutionV97 or
            coupled_execution.hydraulic_execution is not execution):
        raise CoupledAMRFwaveNativeCPUV98Error(
            "v98 interface coupled/hydraulic execution binding differs")
    if type(density_trace_runtime) is not NativeDensityTraceFacesRuntimeV98:
        raise CoupledAMRFwaveNativeCPUV98Error(
            "v98 interface requires the exact native density-trace runtime")
    validate_coupled_amr_turbulence_trace_structure_cpu_v98(
        turbulence_trace, coupled_execution, trace)
    if (type(momentum_trace_runtime) is not NativeMomentumTraceRuntimeV98 or
            type(shared_turbulence_runtime) is not
                NativeSharedTurbulenceRuntimeV97 or
            type(shared_turbulence_prepared) is not
                PreparedNativeTurbulenceV97):
        raise CoupledAMRFwaveNativeCPUV98Error(
            "v98 interface requires exact native viscosity runtimes")
    validate_coupled_amr_internal_trace_structure_cpu_v98(
        trace, coupled_execution)
    locked = tuple(sorted(verify_locked_sources_v85(_ROOT).items()))
    relatives = (_PARAMETER_RELATIVE, _RPN2_RELATIVE, _UTILS_RELATIVE,
                 _V95_RELATIVE, _V97_RELATIVE, _SOURCE_RELATIVE)
    sources = [_ROOT/item for item in relatives]
    if not all(path.is_file() for path in sources):
        raise CoupledAMRFwaveNativeCPUV98Error(
            "v98 interface native source inventory differs")
    hashes = {item: _sha(path) for item, path in zip(relatives, sources)}
    if any(hashes[item] != expected
           for item, expected in _EXPECTED_FROZEN.items()):
        raise CoupledAMRFwaveNativeCPUV98Error(
            "v98 interface frozen augmented-fwave source differs")
    chosen = Path(compiler).resolve() if compiler is not None else None
    if chosen is None:
        found = shutil.which("gfortran")
        if found is None and os.name == "nt":
            fallback = Path(r"C:\Strawberry\c\bin\gfortran.exe")
            found = str(fallback) if fallback.is_file() else None
        if found is None:
            raise CoupledAMRFwaveNativeCPUV98Error(
                "v98 interface native build requires gfortran")
        chosen = Path(found).resolve()
    if not chosen.is_file():
        raise CoupledAMRFwaveNativeCPUV98Error(
            "v98 interface native compiler is unavailable")
    flags = ("-O3", "-flto", "-fopenmp", "-shared",
             "-ffree-line-length-none", "-ffixed-line-length-none",
             "-static-libgfortran", "-static-libgcc")
    root = Path(build_directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    key = _fingerprint({
        "source": hashes, "compiler": _sha(chosen),
        "flags": list(flags), "abi": NATIVE_ABI_ID})[:16]
    build = Path(tempfile.mkdtemp(prefix=f"v98_amr_{key}_", dir=root))
    name = ("braids_amr_interface_fwave_v98.dll" if os.name == "nt" else
            "libbraids_amr_interface_fwave_v98.so")
    partial = build/(name+".partial")
    library_path = build/name
    completed = subprocess.run(
        [str(chosen), *flags, *(str(path) for path in sources),
         "-o", str(partial)], cwd=build, capture_output=True, text=True,
        timeout=180, check=False)
    if completed.returncode != 0 or not partial.is_file():
        raise CoupledAMRFwaveNativeCPUV98Error(
            "v98 interface native build failed:\n"+
            completed.stdout+completed.stderr)
    if any(_sha(path) != hashes[item]
           for item, path in zip(relatives, sources)):
        raise CoupledAMRFwaveNativeCPUV98Error(
            "v98 interface source changed during compilation")
    os.replace(partial, library_path)
    version = subprocess.run(
        [str(chosen), "--version"], capture_output=True, text=True,
        timeout=30, check=True).stdout.splitlines()[0]
    receipt0 = NativeAMRInterfaceFwaveBuildReceiptV98(
        NATIVE_ABI_ID, hashes[_SOURCE_RELATIVE], _sha(Path(__file__)),
        hashes[_V97_RELATIVE], hashes[_V95_RELATIVE],
        hashes[_PARAMETER_RELATIVE], hashes[_RPN2_RELATIVE],
        hashes[_UTILS_RELATIVE], _sha(library_path), version, flags,
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
            "+00:00", "Z"), LOCKED_UPSTREAM_COMMIT, locked,
        density_trace_runtime.receipt["receipt_sha256"],
        turbulence_trace.receipt_sha256,
        momentum_trace_runtime.receipt["receipt_sha256"],
        shared_turbulence_runtime.receipt["receipt_sha256"],
        shared_turbulence_prepared.preparation_sha256, "")
    receipt = NativeAMRInterfaceFwaveBuildReceiptV98(
        **{**receipt0.__dict__, "receipt_sha256": _fingerprint(
            _build_payload(receipt0))})
    dependency = (os.add_dll_directory(str(chosen.parent))
                  if os.name == "nt" and hasattr(os, "add_dll_directory")
                  else None)
    library = ctypes.CDLL(str(library_path))
    function = library.braids_amr_interface_fwave_rate_v98
    fa = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    ia = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
    function.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        fa, fa, ia, ia, fa, fa, fa, fa, ia, ia, ia, ia, ia, fa,
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_int, ia, fa, fa, fa, ia,
        fa, fa, fa, fa, fa, fa, fa, fa, ia,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
    ]
    function.restype = None
    runtime = NativeAMRInterfaceFwaveRuntimeV98(
        receipt, library_path, library, function, dependency,
        execution=execution, coupled_execution=coupled_execution,
        trace=trace, substep_index=substep_index,
        density_trace_runtime=density_trace_runtime,
        turbulence_trace=turbulence_trace,
        momentum_trace_runtime=momentum_trace_runtime,
        shared_turbulence_runtime=shared_turbulence_runtime,
        shared_turbulence_prepared=shared_turbulence_prepared)
    validate_native_amr_interface_build_receipt_v98(receipt)
    return runtime


def bind_native_amr_interface_fwave_runtime_v98(
    prototype: NativeAMRInterfaceFwaveRuntimeV98, *,
    execution: WetDryFwaveCPUExecutionV97,
    coupled_execution: object,
    trace: CoupledAMRInternalTraceCPUV98,
    substep_index: int,
    density_trace_runtime: NativeDensityTraceFacesRuntimeV98,
    turbulence_trace: CoupledAMRTurbulenceTraceCPUV98,
    momentum_trace_runtime: NativeMomentumTraceRuntimeV98,
    shared_turbulence_runtime: NativeSharedTurbulenceRuntimeV97,
    shared_turbulence_prepared: PreparedNativeTurbulenceV97,
) -> NativeAMRInterfaceFwaveRuntimeV98:
    """Bind another trace/substep to one already authenticated native image.

    The compiled operator contains no trace- or state-specific data.  A new
    immutable build/binding receipt authenticates the supplied trace and v91
    preparation while retaining the exact same source and library hashes.
    This avoids recompiling an identical native image for fine substep two or
    the next bounded macro update.
    """
    if type(prototype) is not NativeAMRInterfaceFwaveRuntimeV98:
        raise CoupledAMRFwaveNativeCPUV98Error(
            "v98 interface prototype type differs")
    validate_native_amr_interface_build_receipt_v98(prototype.receipt)
    if (not prototype.library_path.is_file() or
            _sha(prototype.library_path) !=
                prototype.receipt.library_sha256):
        raise CoupledAMRFwaveNativeCPUV98Error(
            "v98 interface prototype library identity differs")
    receipt0 = replace(
        prototype.receipt,
        density_trace_runtime_sha256=
            density_trace_runtime.receipt["receipt_sha256"],
        turbulence_trace_receipt_sha256=turbulence_trace.receipt_sha256,
        momentum_trace_runtime_sha256=
            momentum_trace_runtime.receipt["receipt_sha256"],
        shared_turbulence_runtime_sha256=
            shared_turbulence_runtime.receipt["receipt_sha256"],
        shared_turbulence_preparation_sha256=
            shared_turbulence_prepared.preparation_sha256,
        receipt_sha256="")
    receipt = replace(
        receipt0,
        receipt_sha256=_fingerprint(_build_payload(receipt0)))
    runtime = NativeAMRInterfaceFwaveRuntimeV98(
        receipt, prototype.library_path, prototype._library,
        prototype._function, prototype._dependency,
        execution=execution, coupled_execution=coupled_execution,
        trace=trace, substep_index=substep_index,
        density_trace_runtime=density_trace_runtime,
        turbulence_trace=turbulence_trace,
        momentum_trace_runtime=momentum_trace_runtime,
        shared_turbulence_runtime=shared_turbulence_runtime,
        shared_turbulence_prepared=shared_turbulence_prepared)
    validate_native_amr_interface_build_receipt_v98(receipt)
    return runtime


__all__ = [
    "AMRInterfaceResidualAuditV98",
    "CoupledAMRFwaveNativeCPUV98Error",
    "NativeAMRInterfaceFwaveBuildReceiptV98",
    "NativeAMRInterfaceFwaveRuntimeV98",
    "NativeAMRInterfaceRateReceiptV98",
    "bind_native_amr_interface_fwave_runtime_v98",
    "build_native_amr_interface_fwave_runtime_v98",
    "validate_amr_interface_residual_audit_v98",
    "validate_native_amr_interface_build_receipt_v98",
]
