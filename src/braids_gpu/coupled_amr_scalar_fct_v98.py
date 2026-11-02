"""Actual-CUDA combined scalar FCT with a typed v98 internal trace.

The frozen v92 CUDA concentration, limiter, update, and physical-boundary
ledger kernels are reused verbatim.  Two additive kernels give selected
``neighbour == boundary_index == -1`` faces an authenticated coarse/fine
concentration and record their transfer outside the physical-boundary ledger.
This object is a provisional numerical owner: it publishes no scalar state,
boundary memory, clock, or accepted update.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np

from braids_support.gpu_fwave.suspended_advection import (
    HERE as ADVECTION_FOLDER,
    OPTIONS as CUDA_OPTIONS,
    ResidentSuspendedAdvection,
)
from .coupled_amr_scalar_trace_v98 import (
    CoupledAMRScalarTraceGPUV98,
    validate_coupled_amr_scalar_trace_gpu_v98,
)
from .coupled_amr_trace_v98 import CoupledAMRInternalTraceGPUV98
from .coupled_wetdry_v97 import (
    CoupledWetDryGPUExecutionV97,
    _grid,
    _v92_execution,
)
from .transported_fluid_lifecycle_v92 import _geometry


ROOT = Path(__file__).resolve().parents[1]
CUDA_SOURCE = Path(__file__).with_suffix(".cu")
BASE_CUDA_SOURCE = ADVECTION_FOLDER/"suspended_advection.cu"
SCHEMA = 1
SCOPE_ID = "braids.q4.coupled_amr.scalar_fct.actual_cuda.v98"
BACKEND = "actual_cuda"
KERNELS = (
    "v98_sa_face_low", "v98_sa_cell_bounds",
    "v98_sa_interface_ledger",
)
THREADS = 128


class CoupledAMRScalarFCTGPUV98Error(ValueError):
    """The resident scalar-interface operator or evidence differs."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError, OverflowError) as error:
        raise CoupledAMRScalarFCTGPUV98Error(
            "v98 CUDA scalar FCT identity is not canonical JSON") from error


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _digest(*values: object) -> str:
    result = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(value)
        result.update(array.dtype.str.encode("ascii"))
        result.update(_canonical(list(array.shape)))
        result.update(array.tobytes(order="C"))
    return result.hexdigest()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _poll(value: bool | Callable[[], bool]) -> Callable[[], bool]:
    if type(value) is bool:
        return lambda: value
    if callable(value):
        return lambda: bool(value())
    raise CoupledAMRScalarFCTGPUV98Error(
        "v98 CUDA scalar cancellation must be bool or callable")


def _frozen(value: object, dtype=np.float64) -> np.ndarray:
    result = np.array(value, dtype=dtype, order="C", copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class AMRScalarFCTResultGPUV98:
    execution_fingerprint: str
    runtime_receipt_sha256: str
    scalar_trace_receipt_sha256: str
    substep_index: int
    dt_s: float
    input_sha256: str
    mass_kg_m2: np.ndarray
    advective_face_flux_kg_m_s: np.ndarray
    advective_boundary_ledger_kg: np.ndarray
    internal_interface_owner_ledger_kg: np.ndarray
    internal_interface_faces: np.ndarray
    inventory_residual_kg: np.ndarray
    concentration_kernel_launch_count: int
    interface_face_kernel_launch_count: int
    cell_bound_kernel_launch_count: int
    limiter_kernel_launch_count: int
    update_kernel_launch_count: int
    physical_boundary_ledger_kernel_launch_count: int
    internal_interface_ledger_kernel_launch_count: int
    internal_interface_owner_count: int
    physical_boundary_surrogate_count: int
    boundary_memory_consumer_count: int
    state_or_clock_publication_count: int
    backend: str
    result_sha256: str


def _result_payload(value: AMRScalarFCTResultGPUV98) -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID+":result",
        "execution": value.execution_fingerprint,
        "runtime": value.runtime_receipt_sha256,
        "scalar_trace": value.scalar_trace_receipt_sha256,
        "substep_index": value.substep_index,
        "dt_s": value.dt_s,
        "input_sha256": value.input_sha256,
        "output_sha256": _digest(
            value.mass_kg_m2, value.advective_face_flux_kg_m_s,
            value.advective_boundary_ledger_kg,
            value.internal_interface_owner_ledger_kg,
            value.internal_interface_faces,
            value.inventory_residual_kg),
        "launch_counts": [
            value.concentration_kernel_launch_count,
            value.interface_face_kernel_launch_count,
            value.cell_bound_kernel_launch_count,
            value.limiter_kernel_launch_count,
            value.update_kernel_launch_count,
            value.physical_boundary_ledger_kernel_launch_count,
            value.internal_interface_ledger_kernel_launch_count,
        ],
        "internal_interface_owner_count":
            value.internal_interface_owner_count,
        "physical_boundary_surrogate_count":
            value.physical_boundary_surrogate_count,
        "boundary_memory_consumer_count":
            value.boundary_memory_consumer_count,
        "state_or_clock_publication_count":
            value.state_or_clock_publication_count,
        "backend": value.backend,
        "actual_cuda_kernels": True,
        "automatic_cpu_fallback": False,
    }


class ResidentAMRScalarFCTGPUV98:
    """One provisional combined scalar update on the selected CUDA device."""

    def __init__(
        self,
        execution: CoupledWetDryGPUExecutionV97,
        internal_trace: CoupledAMRInternalTraceGPUV98,
        scalar_trace: CoupledAMRScalarTraceGPUV98,
    ) -> None:
        if type(execution) is not CoupledWetDryGPUExecutionV97:
            raise CoupledAMRScalarFCTGPUV98Error(
                "v98 CUDA scalar execution type differs")
        validate_coupled_amr_scalar_trace_gpu_v98(
            scalar_trace, internal_trace, execution)
        try:
            import cupy as cp
        except Exception as error:  # pragma: no cover - host dependent
            raise CoupledAMRScalarFCTGPUV98Error(
                "v98 CUDA scalar FCT requires CuPy") from error
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise CoupledAMRScalarFCTGPUV98Error(
                "v98 CUDA scalar FCT requires a CUDA device")
        self.cp = cp
        self.execution = execution
        self.internal_trace = internal_trace
        self.scalar_trace = scalar_trace
        self.v92_execution = _v92_execution(execution.active_execution)
        self.grid = _grid(execution.active_execution)
        self.ng = int(scalar_trace.fraction_count)
        self.active_fluid = scalar_trace.active_fluid_lanes
        self.nl = self.ng+len(self.active_fluid)
        self.nc, self.nf, self.nb = (
            execution.cell_count, execution.face_count,
            self.v92_execution.boundary_count)
        self.advection = ResidentSuspendedAdvection(
            _geometry(self.grid), self.nl)
        if not CUDA_SOURCE.is_file() or not BASE_CUDA_SOURCE.is_file():
            raise CoupledAMRScalarFCTGPUV98Error(
                "v98 CUDA scalar source is missing")
        try:
            module = cp.RawModule(
                code=CUDA_SOURCE.read_text(encoding="utf-8"),
                options=CUDA_OPTIONS, name_expressions=KERNELS)
            self.kernels = {
                name: module.get_function(name) for name in KERNELS}
        except Exception as error:
            raise CoupledAMRScalarFCTGPUV98Error(
                "v98 CUDA scalar interface kernels failed to compile") from error
        faces = np.asarray(scalar_trace.interface_faces, dtype=np.int32)
        if (faces.shape != (2,) or len(set(map(int, faces))) != 2 or
                np.any(np.asarray(self.grid.neighbour)[faces] != -1) or
                np.any(np.asarray(self.grid.boundary_index)[faces] != -1)):
            raise CoupledAMRScalarFCTGPUV98Error(
                "v98 CUDA scalar faces are not internal trace topology")
        self.interface_faces_host = np.ascontiguousarray(faces)
        self.interface_faces = cp.asarray(self.interface_faces_host)
        self.interface_active = cp.zeros(self.nf, dtype=cp.int32)
        self.interface_active[self.interface_faces] = 1
        self.interface_state = cp.zeros(
            (self.nl, self.nf), dtype=cp.float64)
        self.interface_ledger = cp.zeros(
            (self.nl, 2), dtype=cp.float64)
        self.interface_ledger_status = cp.zeros(
            (self.nl, 2), dtype=cp.int32)
        self.runtime_receipt = {
            "schema": SCHEMA,
            "scope_id": SCOPE_ID+":runtime",
            "execution_fingerprint": execution.fingerprint,
            "internal_trace_receipt_sha256": internal_trace.receipt_sha256,
            "scalar_trace_receipt_sha256": scalar_trace.receipt_sha256,
            "cuda_source_sha256": _sha(CUDA_SOURCE),
            "frozen_v92_advection_source_sha256": _sha(BASE_CUDA_SOURCE),
            "cuda_options": list(CUDA_OPTIONS),
            "device_id": int(cp.cuda.Device().id),
            "interface_faces": faces.tolist(),
            "combined_lane_count": self.nl,
            "actual_cuda_kernels": True,
            "automatic_cpu_fallback": False,
            "state_or_clock_owner_count": 0,
            "production_admission": False,
        }
        self.runtime_receipt["receipt_sha256"] = _hash(
            self.runtime_receipt)
        self._busy = False
        self.last_result = None
        self.resident_mass = None
        self.resident_flux = None
        self.resident_interface_ledger = None

    def _validate_runtime(self) -> None:
        receipt = self.runtime_receipt
        body = {key: value for key, value in receipt.items()
                if key != "receipt_sha256"}
        if (receipt.get("receipt_sha256") != _hash(body) or
                receipt.get("cuda_source_sha256") != _sha(CUDA_SOURCE) or
                receipt.get("frozen_v92_advection_source_sha256") !=
                    _sha(BASE_CUDA_SOURCE) or
                receipt.get("device_id") != int(self.cp.cuda.Device().id)):
            raise CoupledAMRScalarFCTGPUV98Error(
                "v98 CUDA scalar runtime/source identity differs")
        validate_coupled_amr_scalar_trace_gpu_v98(
            self.scalar_trace, self.internal_trace, self.execution)

    def _device(self, value, shape: tuple[int, ...], dtype, name: str):
        if (not isinstance(value, self.cp.ndarray) or value.shape != shape or
                value.dtype != dtype or not value.flags.c_contiguous or
                int(value.device.id) != int(self.cp.cuda.Device().id)):
            raise CoupledAMRScalarFCTGPUV98Error(
                f"v98 CUDA scalar {name} must be resident {dtype} {shape}")
        return value

    def _launch(self, table, name: str, count: int,
                args: tuple[object, ...]) -> None:
        if count:
            table[name](((count+THREADS-1)//THREADS,), (THREADS,), args)

    def abort(self) -> None:
        self._busy = False
        self.last_result = None
        self.resident_mass = None
        self.resident_flux = None
        self.resident_interface_ledger = None
        self.interface_state.fill(0.0)
        self.interface_ledger.fill(0.0)
        self.interface_ledger_status.fill(0)

    def evaluate(
        self, *, substep_index: int,
        mass_kg_m2, depth_before_m, depth_after_m,
        accepted_water_flux_m2_s,
        reconstructed_left_concentration,
        reconstructed_right_concentration,
        resolved_physical_boundary_concentration,
        hydro_constant, dt_s: float,
        cancelled: bool | Callable[[], bool] = False,
    ) -> AMRScalarFCTResultGPUV98:
        self._validate_runtime()
        if self._busy or type(substep_index) is not int or \
                substep_index not in (0, 1):
            raise CoupledAMRScalarFCTGPUV98Error(
                "v98 CUDA scalar transaction/substep differs")
        poll = _poll(cancelled)
        if poll():
            raise InterruptedError(
                "v98 CUDA scalar FCT cancelled before launch")
        if (isinstance(dt_s, bool) or not isinstance(dt_s, (int, float)) or
                not math.isfinite(dt_s) or dt_s <= 0.0):
            raise CoupledAMRScalarFCTGPUV98Error(
                "v98 CUDA scalar timestep differs")
        cp, adv = self.cp, self.advection
        a, b = adv.geometry, adv.buffers
        mass = self._device(
            mass_kg_m2, (self.nl, self.nc), cp.float64, "mass")
        h0 = self._device(
            depth_before_m, (self.nc,), cp.float64, "depth before")
        h1 = self._device(
            depth_after_m, (self.nc,), cp.float64, "depth after")
        water = self._device(
            accepted_water_flux_m2_s, (self.nf,), cp.float64, "water flux")
        left = self._device(
            reconstructed_left_concentration, (self.nl, self.nf),
            cp.float64, "left reconstruction")
        right = self._device(
            reconstructed_right_concentration, (self.nl, self.nf),
            cp.float64, "right reconstruction")
        boundary = self._device(
            resolved_physical_boundary_concentration, (self.nl, self.nb),
            cp.float64, "physical boundary concentration")
        hydro = self._device(
            hydro_constant, (self.nc,), cp.int32, "hydro constant")
        self._busy = True
        self.last_result = None
        self.resident_mass = None
        self.resident_flux = None
        self.resident_interface_ledger = None
        try:
            self.interface_state.fill(0.0)
            self.interface_state[:, self.interface_faces] = cp.asarray(
                self.scalar_trace.substep_concentration[substep_index])
            # The high-order inflow candidate is the same authenticated trace.
            if not bool(cp.array_equal(
                    right[:, self.interface_faces],
                    self.interface_state[:, self.interface_faces]).item()):
                raise CoupledAMRScalarFCTGPUV98Error(
                    "v98 CUDA scalar right trace differs")
            nc, nf, nl, nb = map(
                np.int32, (self.nc, self.nf, self.nl, self.nb))
            dt = np.float64(dt_s)
            adv._launch("sa_concentration", self.nc*self.nl, (
                nc, nl, h0, mass, np.float64(adv.dry),
                b["concentration"], b["concentration_status"]))
            self._launch(self.kernels, "v98_sa_face_low", self.nf*self.nl, (
                nc, nf, nl, nb, water, b["concentration"], left, right,
                boundary, self.interface_state, self.interface_active,
                a["owner"], a["neighbour"], a["bid"],
                b["concentration_status"], b["low_flux"], b["anti"],
                b["face_status"]))
            self._launch(
                self.kernels, "v98_sa_cell_bounds", self.nc*self.nl, (
                    nc, nf, nl, nb, mass, h1, b["concentration"], boundary,
                    self.interface_state, self.interface_active, water,
                    b["low_flux"], b["anti"], a["area"], a["length"],
                    a["normal"], a["rectangle_width"], hydro,
                    a["weno_cells"], a["owner"], a["neighbour"], a["bid"],
                    a["offset"], a["faces"], a["signs"],
                    b["concentration_status"], b["face_status"], dt,
                    b["low_state"], b["rplus"], b["rminus"],
                    b["cell_status"]))
            adv._launch("sa_limit_faces", self.nf*self.nl, (
                nc, nf, nl, b["low_flux"], b["anti"], b["rplus"],
                b["rminus"], a["owner"], a["neighbour"],
                b["cell_status"], b["scaled_anti"], b["flux"],
                b["flux_status"]))
            adv._launch("sa_update", self.nc*self.nl, (
                nc, nf, nl, b["low_state"], b["scaled_anti"], h1,
                a["length"], a["area"], a["offset"], a["faces"],
                a["signs"], b["cell_status"], b["flux_status"], dt,
                np.float64(adv.dry), b["next_mass"],
                b["next_concentration"], b["update_status"]))
            adv._launch("sa_boundary_ledger", self.nb*self.nl, (
                nf, nl, nb, a["boundary_offset"], a["boundary_faces"],
                b["flux"], a["length"], b["flux_status"], dt,
                b["ledger"], b["ledger_status"]))
            self._launch(
                self.kernels, "v98_sa_interface_ledger", 2*self.nl, (
                    nf, nl, np.int32(2), self.interface_faces, b["flux"],
                    a["length"], b["flux_status"], dt,
                    self.interface_ledger, self.interface_ledger_status))
            statuses = (
                b["concentration_status"], b["face_status"],
                b["cell_status"], b["flux_status"], b["update_status"],
                b["ledger_status"], self.interface_ledger_status)
            if any(bool(cp.any(value != 0).item()) for value in statuses):
                raise CoupledAMRScalarFCTGPUV98Error(
                    "v98 CUDA scalar FCT kernel rejected the trace")
            if poll():
                raise InterruptedError(
                    "v98 CUDA scalar FCT cancelled after launch")

            host_inputs = tuple(cp.asnumpy(value) for value in (
                mass, h0, h1, water, left, right, boundary, hydro))
            input_sha = _digest(
                *host_inputs,
                self.scalar_trace.substep_concentration[substep_index],
                self.interface_faces_host, np.asarray((dt_s,)))
            host_mass = _frozen(cp.asnumpy(b["next_mass"]))
            host_flux = _frozen(cp.asnumpy(b["flux"]))
            host_boundary = _frozen(cp.asnumpy(b["ledger"]))
            host_interface = _frozen(cp.asnumpy(self.interface_ledger))
            host_faces = _frozen(self.interface_faces_host, np.int32)
            area = np.asarray(self.grid.cell_area_m2)
            inventory_change = np.sum(
                (host_mass-host_inputs[0])*area[None, :], axis=1)
            accounted = np.sum(host_boundary, axis=1)+np.sum(
                host_interface, axis=1)
            residual = _frozen(inventory_change-accounted)
            scale = np.maximum(
                1.0, np.maximum(np.abs(inventory_change),
                                np.abs(accounted)))
            if np.any(np.abs(residual) >
                      65536.0*np.finfo(float).eps*scale):
                raise CoupledAMRScalarFCTGPUV98Error(
                    "v98 CUDA scalar inventory ledger differs")
            placeholder = AMRScalarFCTResultGPUV98(
                self.execution.fingerprint,
                self.runtime_receipt["receipt_sha256"],
                self.scalar_trace.receipt_sha256, substep_index,
                float(dt_s), input_sha, host_mass, host_flux,
                host_boundary, host_interface, host_faces, residual,
                1, 1, 1, 1, 1, int(self.nb > 0), 1,
                1, 0, 0, 0, BACKEND, "")
            result = AMRScalarFCTResultGPUV98(
                **{**placeholder.__dict__,
                   "result_sha256": _hash(_result_payload(placeholder))})
            validate_amr_scalar_fct_result_gpu_v98(result, self)
            self.resident_mass = b["next_mass"].copy()
            self.resident_flux = b["flux"].copy()
            self.resident_interface_ledger = self.interface_ledger.copy()
            self.last_result = result
            return result
        except Exception:
            self.abort()
            raise
        finally:
            self._busy = False


def validate_amr_scalar_fct_result_gpu_v98(
    value: AMRScalarFCTResultGPUV98,
    runtime: ResidentAMRScalarFCTGPUV98,
) -> None:
    if (type(value) is not AMRScalarFCTResultGPUV98 or
            type(runtime) is not ResidentAMRScalarFCTGPUV98):
        raise CoupledAMRScalarFCTGPUV98Error(
            "v98 CUDA scalar result/runtime type differs")
    runtime._validate_runtime()
    arrays_shapes = (
        (value.mass_kg_m2, (runtime.nl, runtime.nc)),
        (value.advective_face_flux_kg_m_s, (runtime.nl, runtime.nf)),
        (value.advective_boundary_ledger_kg, (runtime.nl, runtime.nb)),
        (value.internal_interface_owner_ledger_kg, (runtime.nl, 2)),
        (value.internal_interface_faces, (2,)),
        (value.inventory_residual_kg, (runtime.nl,)),
    )
    expected_interface = (
        -float(value.dt_s)*
        np.asarray(runtime.grid.face_length_m)[
            runtime.interface_faces_host][None, :]*
        np.asarray(value.advective_face_flux_kg_m_s)[
            :, runtime.interface_faces_host])
    if (value.execution_fingerprint != runtime.execution.fingerprint or
            value.runtime_receipt_sha256 !=
                runtime.runtime_receipt["receipt_sha256"] or
            value.scalar_trace_receipt_sha256 !=
                runtime.scalar_trace.receipt_sha256 or
            value.substep_index not in (0, 1) or
            not math.isfinite(value.dt_s) or value.dt_s <= 0.0 or
            len(value.input_sha256) != 64 or
            any(np.asarray(array).shape != shape or
                np.asarray(array).flags.writeable or
                not np.isfinite(array).all()
                for array, shape in arrays_shapes) or
            not np.array_equal(value.internal_interface_faces,
                               runtime.interface_faces_host) or
            not np.array_equal(
                value.internal_interface_owner_ledger_kg,
                expected_interface) or
            any(count != 1 for count in (
                value.concentration_kernel_launch_count,
                value.interface_face_kernel_launch_count,
                value.cell_bound_kernel_launch_count,
                value.limiter_kernel_launch_count,
                value.update_kernel_launch_count,
                value.internal_interface_ledger_kernel_launch_count,
                value.internal_interface_owner_count)) or
            value.physical_boundary_ledger_kernel_launch_count !=
                int(runtime.nb > 0) or
            value.physical_boundary_surrogate_count != 0 or
            value.boundary_memory_consumer_count != 0 or
            value.state_or_clock_publication_count != 0 or
            value.backend != BACKEND or
            value.result_sha256 != _hash(_result_payload(value))):
        raise CoupledAMRScalarFCTGPUV98Error(
            "v98 CUDA scalar result identity differs")


def method_identity_coupled_amr_scalar_fct_gpu_v98() -> dict[str, object]:
    return {
        "scope_id": SCOPE_ID,
        "backend": BACKEND,
        "operator": "v92_combined_nonnegative_fct_with_internal_trace",
        "frozen_v92_kernels_reused": [
            "sa_concentration", "sa_limit_faces", "sa_update",
            "sa_boundary_ledger"],
        "additive_interface_kernels": list(KERNELS),
        "physical_boundary_surrogate_count": 0,
        "state_or_clock_publication_count": 0,
        "actual_cuda_kernels": True,
        "automatic_cpu_fallback": False,
        "production_admission": False,
    }


__all__ = [
    "AMRScalarFCTResultGPUV98", "BACKEND",
    "CoupledAMRScalarFCTGPUV98Error", "ResidentAMRScalarFCTGPUV98",
    "SCOPE_ID", "method_identity_coupled_amr_scalar_fct_gpu_v98",
    "validate_amr_scalar_fct_result_gpu_v98",
]
