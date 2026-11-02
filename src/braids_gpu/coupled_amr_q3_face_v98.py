"""Actual-CUDA Q3 face/Exner operator for a v98 internal trace.

The operator reuses the selected resident FLOW bed-slope composer and the
deterministic v93 inventory kernels.  Two selected closed-topology faces are
temporarily represented by two private trace cells, so the unchanged shared
face law sees one genuine internal pair.  Only real fine cells enter Exner;
the exterior transfer is returned in a separate AMR ledger.  No provider
state, bed state, boundary owner, or clock is published here.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np

from braids_core.sediment.direct_vector_wetdry_amr_live_v71 import (
    FACE_CLOSED,
    FACE_INTERNAL,
    FACE_OWNER_OUTLET,
)
from braids_core.sediment.q3_exner_adapter_v93 import _base_topology
from .active_suspended_control_v94 import ResidentQ3V83ActiveFaceTrialV94
from .active_suspended_wetdry_v97 import (
    ResidentQ3V83ActiveWetDryExnerAdapterV97,
)
from .coupled_amr_q3_trace_v98 import (
    CoupledAMRQ3TraceGPUV98,
    validate_coupled_amr_q3_trace_gpu_v98,
)
from .coupled_amr_trace_v98 import CoupledAMRInternalTraceGPUV98
from .coupled_wetdry_v97 import (
    CoupledWetDryGPUExecutionV97,
    CoupledWetDryGPUTrialV97,
    _grid,
)
from .flow_bed_slope_composition import (
    DONOR_NEIGHBOUR,
    DONOR_OWNER,
    DONOR_TRACE,
    ResidentFlowBedSlopeFaceComposer,
)
from .flow_bed_slope_runtime import FlowBedSlopeCUDA
from .q3_exner_adapter_v93 import (
    CUDA_OPTIONS,
    DETERMINISTIC_KERNEL_NAMES,
)


ROOT = Path(__file__).resolve().parents[1]
DETERMINISTIC_SOURCE = Path(__file__).with_name("q3_exner_adapter_v93.cu")
SLOPE_SOURCE = Path(__file__).with_name("flow_bed_slope_runtime.cu")
COMPOSER_SOURCE = Path(__file__).with_name("flow_bed_slope_composition.py")
SCHEMA = 1
SCOPE_ID = "braids.q4.coupled_amr.q3_face.actual_cuda.v98"
BACKEND = "actual_cuda"
THREADS = 128


class CoupledAMRQ3FaceGPUV98Error(ValueError):
    """The actual-CUDA Q3 interface operator or evidence differs."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError, OverflowError) as error:
        raise CoupledAMRQ3FaceGPUV98Error(
            "v98 CUDA Q3 face identity is not canonical JSON") from error


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
    raise CoupledAMRQ3FaceGPUV98Error(
        "v98 CUDA Q3 cancellation must be bool or callable")


def _frozen(value: object, dtype=np.float64) -> np.ndarray:
    result = np.array(value, dtype=dtype, order="C", copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class AMRQ3FaceResultGPUV98:
    execution_fingerprint: str
    runtime_receipt_sha256: str
    q3_trace_receipt_sha256: str
    dt_s: float
    input_sha256: str
    method_fingerprint: str
    face_flux_kg_m_s: np.ndarray
    cell_delta_kg_m2: np.ndarray
    availability: np.ndarray
    named_boundary_transfer_kg: np.ndarray
    named_boundary_absolute_kg: np.ndarray
    internal_interface_transfer_kg: np.ndarray
    depletion_roundoff_positive_in_kg: np.ndarray
    classwise_residual_kg: np.ndarray
    interface_faces: np.ndarray
    provisional_face_batch_count: int
    accepted_face_batch_count: int
    outgoing_inventory_kernel_launch_count: int
    delta_kernel_launch_count: int
    named_boundary_kernel_launch_count: int
    interface_transfer_materialization_count: int
    internal_interface_owner_count: int
    physical_boundary_surrogate_count: int
    python_numerical_cell_face_loop_count: int
    accepted_bed_or_clock_publication_count: int
    backend: str
    result_sha256: str


def _result_payload(value: AMRQ3FaceResultGPUV98) -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID+":result",
        "execution": value.execution_fingerprint,
        "runtime": value.runtime_receipt_sha256,
        "q3_trace": value.q3_trace_receipt_sha256,
        "dt_s": value.dt_s,
        "input_sha256": value.input_sha256,
        "method_fingerprint": value.method_fingerprint,
        "output_sha256": _digest(
            value.face_flux_kg_m_s, value.cell_delta_kg_m2,
            value.availability, value.named_boundary_transfer_kg,
            value.named_boundary_absolute_kg,
            value.internal_interface_transfer_kg,
            value.depletion_roundoff_positive_in_kg,
            value.classwise_residual_kg, value.interface_faces),
        "counts": [
            value.provisional_face_batch_count,
            value.accepted_face_batch_count,
            value.outgoing_inventory_kernel_launch_count,
            value.delta_kernel_launch_count,
            value.named_boundary_kernel_launch_count,
            value.interface_transfer_materialization_count,
            value.internal_interface_owner_count,
            value.physical_boundary_surrogate_count,
            value.python_numerical_cell_face_loop_count,
            value.accepted_bed_or_clock_publication_count,
        ],
        "backend": value.backend,
        "actual_cuda_kernels": True,
        "automatic_cpu_fallback": False,
    }


class ResidentAMRQ3FaceGPUV98:
    """One provisional resident Q3 face/inventory calculation."""

    def __init__(
        self, execution: CoupledWetDryGPUExecutionV97,
        provider: ResidentQ3V83ActiveWetDryExnerAdapterV97,
        internal_trace: CoupledAMRInternalTraceGPUV98,
        q3_trace: CoupledAMRQ3TraceGPUV98,
        coarse_trial: CoupledWetDryGPUTrialV97,
    ) -> None:
        if (type(execution) is not CoupledWetDryGPUExecutionV97 or
                type(provider) is not
                    ResidentQ3V83ActiveWetDryExnerAdapterV97):
            raise CoupledAMRQ3FaceGPUV98Error(
                "v98 CUDA Q3 execution/provider type differs")
        validate_coupled_amr_q3_trace_gpu_v98(
            q3_trace, internal_trace, coarse_trial, execution)
        if provider.control_execution is not execution.active_execution:
            raise CoupledAMRQ3FaceGPUV98Error(
                "v98 CUDA Q3 provider is not the fine execution owner")
        self.execution = execution
        self.provider = provider
        self.internal_trace = internal_trace
        self.q3_trace = q3_trace
        self.coarse_trial = coarse_trial
        self.cp = provider.cp
        self.device_id = provider.device_id
        self.binding = provider.binding
        self.ng = execution.fraction_count
        self.nc, self.nf = execution.cell_count, execution.face_count
        self.nb = len(provider.boundary_names)
        self.grid = _grid(execution.active_execution)
        self.faces_host = np.ascontiguousarray(
            np.asarray(q3_trace.interface_faces, dtype=np.int32))
        self.faces = self.cp.asarray(self.faces_host)
        method = self.binding.core.bed_slope_execution
        self.composer = ResidentFlowBedSlopeFaceComposer(
            method, operator=FlowBedSlopeCUDA())
        source = DETERMINISTIC_SOURCE.read_text(encoding="utf-8")
        try:
            module = self.cp.RawModule(
                code=source, options=CUDA_OPTIONS,
                name_expressions=DETERMINISTIC_KERNEL_NAMES)
            self.kernels = {
                name: module.get_function(name)
                for name in DETERMINISTIC_KERNEL_NAMES}
        except Exception as error:
            raise CoupledAMRQ3FaceGPUV98Error(
                "v98 CUDA deterministic Q3 kernels failed to compile") \
                from error
        self.runtime_receipt = {
            "schema": SCHEMA,
            "scope_id": SCOPE_ID+":runtime",
            "execution_fingerprint": execution.fingerprint,
            "active_execution_fingerprint":
                execution.active_execution.fingerprint,
            "binding_sha256": self.binding.binding_sha256,
            "q3_trace_receipt_sha256": q3_trace.receipt_sha256,
            "method_fingerprint": method["fingerprint"],
            "deterministic_source_sha256": _sha(DETERMINISTIC_SOURCE),
            "slope_source_sha256": _sha(SLOPE_SOURCE),
            "composer_source_sha256": _sha(COMPOSER_SOURCE),
            "wrapper_source_sha256": _sha(Path(__file__)),
            "device_id": self.device_id,
            "actual_cuda_kernels": True,
            "automatic_cpu_fallback": False,
            "state_or_clock_owner_count": 0,
            "production_admission": False,
        }
        self.runtime_receipt["receipt_sha256"] = _hash(
            self.runtime_receipt)
        self._busy = False
        self.last_result = None
        self.resident_face_flux = None
        self.resident_cell_delta = None
        self.resident_interface_transfer = None

    def _validate_runtime(self) -> None:
        receipt = self.runtime_receipt
        body = {key: value for key, value in receipt.items()
                if key != "receipt_sha256"}
        if (receipt.get("receipt_sha256") != _hash(body) or
                receipt.get("deterministic_source_sha256") !=
                    _sha(DETERMINISTIC_SOURCE) or
                receipt.get("slope_source_sha256") != _sha(SLOPE_SOURCE) or
                receipt.get("composer_source_sha256") !=
                    _sha(COMPOSER_SOURCE) or
                receipt.get("wrapper_source_sha256") !=
                    _sha(Path(__file__)) or
                receipt.get("device_id") != int(self.cp.cuda.Device().id) or
                receipt.get("binding_sha256") !=
                    self.provider.binding.binding_sha256):
            raise CoupledAMRQ3FaceGPUV98Error(
                "v98 CUDA Q3 runtime/source identity differs")
        validate_coupled_amr_q3_trace_gpu_v98(
            self.q3_trace, self.internal_trace, self.coarse_trial,
            self.execution)

    def _device(self, value, shape: tuple[int, ...], dtype, name: str):
        if (not isinstance(value, self.cp.ndarray) or value.shape != shape or
                value.dtype != dtype or not value.flags.c_contiguous or
                int(value.device.id) != self.device_id):
            raise CoupledAMRQ3FaceGPUV98Error(
                f"v98 CUDA Q3 {name} must be resident {dtype} {shape}")
        return value

    def _launch(self, name: str, count: int,
                args: tuple[object, ...]) -> None:
        if count:
            self.kernels[name](
                ((count+THREADS-1)//THREADS,), (THREADS,), args)

    def abort(self) -> None:
        self._busy = False
        self.last_result = None
        self.resident_face_flux = None
        self.resident_cell_delta = None
        self.resident_interface_transfer = None

    def evaluate(
        self, *, cell_vector_kg_m_s, bed_mass_kg_m2, face_role,
        downhill, stress, depth, d50, mean_diameter, hiding,
        dt_s: float, cancelled: bool | Callable[[], bool] = False,
    ) -> AMRQ3FaceResultGPUV98:
        self._validate_runtime()
        if self._busy:
            raise CoupledAMRQ3FaceGPUV98Error(
                "v98 CUDA Q3 transaction is already open")
        poll = _poll(cancelled)
        if poll():
            raise InterruptedError("v98 CUDA Q3 cancelled before launch")
        if (isinstance(dt_s, bool) or not isinstance(dt_s, (int, float)) or
                not math.isfinite(dt_s) or dt_s <= 0.0):
            raise CoupledAMRQ3FaceGPUV98Error(
                "v98 CUDA Q3 timestep differs")
        cp = self.cp
        vector = self._device(
            cell_vector_kg_m_s, (2, self.ng, self.nc), cp.float64,
            "cell vector")
        bed = self._device(
            bed_mass_kg_m2, (self.ng, self.nc), cp.float64, "bed mass")
        role_input = self._device(
            face_role, (self.nf,), cp.int32, "face role")
        down = self._device(
            downhill, (2, self.nf), cp.float64, "downhill")
        stress = self._device(
            stress, (self.ng, self.nf), cp.float64, "stress")
        depth = self._device(
            depth, (self.ng, self.nf), cp.float64, "depth")
        d50 = self._device(d50, (self.ng,), cp.float64, "D50")
        mean = self._device(
            mean_diameter, (self.ng, self.nf), cp.float64,
            "mean diameter")
        hiding = self._device(
            hiding, (self.ng, self.nc), cp.float64, "hiding")
        if (not bool(cp.all(role_input[self.faces] == FACE_CLOSED).item()) or
                not bool(cp.all(
                    self.binding.neighbour[self.faces] == -1).item())):
            raise CoupledAMRQ3FaceGPUV98Error(
                "v98 CUDA Q3 selected faces are not closed internal traces")
        self._busy = True
        self.abort()
        self._busy = True
        try:
            # Two private trace cells make the selected faces ordinary
            # internal pairs for the unchanged resident face composer.
            trace_vector = cp.asarray(
                self.q3_trace.interface_vector_kg_m_s, dtype=cp.float64)
            trace_hiding = cp.asarray(
                self.q3_trace.interface_hiding, dtype=cp.float64)
            trace_availability = cp.asarray(
                self.q3_trace.interface_availability, dtype=cp.float64)
            extended_vector = cp.concatenate((vector, trace_vector), axis=2)
            extended_hiding = cp.concatenate((hiding, trace_hiding), axis=1)
            neighbour = self.binding.neighbour.copy()
            neighbour[self.faces] = cp.arange(
                self.nc, self.nc+2, dtype=cp.int32)
            role = role_input.copy()
            role[self.faces] = FACE_INTERNAL
            owner = self.binding.owner
            safe_other = cp.clip(neighbour, 0, self.nc+1)
            left = cp.ascontiguousarray(extended_vector[:, :, owner])
            right = cp.ascontiguousarray(extended_vector[:, :, safe_other])
            zero_trace = cp.zeros_like(left)
            availability_one = cp.ones(
                (self.ng, self.nc+2), dtype=cp.float64)
            common = dict(
                left_bed=left, right_bed=right, owner_bed=left,
                trace_bed=zero_trace, owner=owner, neighbour=neighbour,
                role=role, normal=self.binding.normal, downhill=down,
                stress=stress, depth=depth, d50=d50,
                mean_diameter=mean, hiding=extended_hiding)
            provisional = self.composer.compose(
                availability=availability_one, **common)
            outgoing = cp.zeros((self.ng, self.nc), dtype=cp.float64)
            self._launch("q3_v93_outgoing_cell_gather", self.ng*self.nc, (
                np.int32(self.ng), np.int32(self.nc), np.int32(self.nf),
                self.binding.cell_face_offset, self.binding.cell_faces,
                provisional.donor, owner, neighbour, role,
                provisional.flux, self.binding.face_length_m,
                np.float64(dt_s), self.binding.cell_area_m2,
                np.int32(DONOR_OWNER), np.int32(DONOR_NEIGHBOUR),
                np.int32(DONOR_TRACE), np.int32(FACE_INTERNAL),
                np.int32(FACE_OWNER_OUTLET), outgoing))
            fill = cp.where(bed > 0.0, 1.0, 0.0)
            fill = cp.where(
                outgoing > 0.0,
                cp.minimum(1.0, bed/cp.where(outgoing > 0.0, outgoing, 1.0)),
                fill)
            fill = cp.ascontiguousarray(fill)
            accepted_availability = cp.concatenate(
                (fill, trace_availability), axis=1)
            accepted = self.composer.compose(
                availability=accepted_availability, **common)
            flux = cp.ascontiguousarray(accepted.flux)
            amount = cp.ascontiguousarray(
                flux*self.binding.face_length_m[None, :]*float(dt_s))
            delta = cp.zeros((self.ng, self.nc), dtype=cp.float64)
            self._launch("q3_v93_delta_cell_gather", self.ng*self.nc, (
                np.int32(self.ng), np.int32(self.nc), np.int32(self.nf),
                self.binding.cell_face_offset, self.binding.cell_faces,
                self.binding.cell_face_sign, amount,
                self.binding.cell_area_m2, delta))
            mapping = self.binding.family_boundary_faces
            named = cp.zeros((self.ng, self.nb), dtype=cp.float64)
            absolute = cp.zeros_like(named)
            if self.ng*self.nb:
                self._launch(
                    "q3_v93_named_boundary_gather", self.ng*self.nb, (
                        np.int32(self.ng), np.int32(self.nb),
                        np.int32(self.binding.boundary_topology.nface),
                        np.int32(self.nf), mapping,
                        self.binding.boundary_topology.boundary_index,
                        amount, named, absolute))
            candidate = bed+delta
            tolerance = (4096.0*np.finfo(float).eps *
                         cp.maximum(1.0, cp.abs(bed)))
            if bool(cp.any(candidate < -tolerance).item()):
                raise CoupledAMRQ3FaceGPUV98Error(
                    "v98 CUDA Q3 inventory limiter produced negative bed")
            before = delta.copy()
            delta = cp.ascontiguousarray(cp.where(candidate < 0.0, -bed, delta))
            depletion = cp.ascontiguousarray(cp.sum(
                (delta-before)*self.binding.cell_area_m2[None, :], axis=1))
            interface = cp.ascontiguousarray(amount[:, self.faces])
            residual = cp.ascontiguousarray(
                cp.sum(delta*self.binding.cell_area_m2[None, :], axis=1)+
                cp.sum(named, axis=1)+cp.sum(interface, axis=1)-depletion)
            scale = cp.maximum(
                1.0,
                cp.sum(cp.abs(delta)*self.binding.cell_area_m2[None, :],
                       axis=1)+cp.sum(absolute, axis=1)+
                cp.sum(cp.abs(interface), axis=1)+depletion)
            if bool(cp.any(
                    cp.abs(residual) >
                    8192.0*np.finfo(float).eps*scale).item()):
                raise CoupledAMRQ3FaceGPUV98Error(
                    "v98 CUDA Q3 interface ledger did not close")
            if poll():
                raise InterruptedError("v98 CUDA Q3 cancelled after launch")
            host_inputs = tuple(cp.asnumpy(value) for value in (
                vector, bed, role_input, down, stress, depth, d50, mean,
                hiding))
            input_sha = _digest(
                *host_inputs, self.q3_trace.interface_vector_kg_m_s,
                self.q3_trace.interface_hiding,
                self.q3_trace.interface_availability,
                self.faces_host, np.asarray((dt_s,)))
            outputs = tuple(_frozen(cp.asnumpy(value)) for value in (
                flux, delta, fill, named, absolute, interface, depletion,
                residual))
            faces_host = _frozen(self.faces_host, np.int32)
            placeholder = AMRQ3FaceResultGPUV98(
                self.execution.fingerprint,
                self.runtime_receipt["receipt_sha256"],
                self.q3_trace.receipt_sha256, float(dt_s), input_sha,
                self.binding.core.bed_slope_execution["fingerprint"],
                *outputs, faces_host,
                1, 1, 1, 1, int(self.nb > 0), 1,
                1, 0, 0, 0, BACKEND, "")
            result = AMRQ3FaceResultGPUV98(
                **{**placeholder.__dict__,
                   "result_sha256": _hash(_result_payload(placeholder))})
            validate_amr_q3_face_result_gpu_v98(result, self)
            self.resident_face_flux = flux.copy()
            self.resident_cell_delta = delta.copy()
            self.resident_interface_transfer = interface.copy()
            self.last_result = result
            return result
        except Exception:
            self.abort()
            raise
        finally:
            self._busy = False


def validate_amr_q3_face_result_gpu_v98(
    value: AMRQ3FaceResultGPUV98,
    runtime: ResidentAMRQ3FaceGPUV98,
) -> None:
    if (type(value) is not AMRQ3FaceResultGPUV98 or
            type(runtime) is not ResidentAMRQ3FaceGPUV98):
        raise CoupledAMRQ3FaceGPUV98Error(
            "v98 CUDA Q3 result/runtime type differs")
    runtime._validate_runtime()
    shapes = (
        (value.face_flux_kg_m_s, (runtime.ng, runtime.nf)),
        (value.cell_delta_kg_m2, (runtime.ng, runtime.nc)),
        (value.availability, (runtime.ng, runtime.nc)),
        (value.named_boundary_transfer_kg, (runtime.ng, runtime.nb)),
        (value.named_boundary_absolute_kg, (runtime.ng, runtime.nb)),
        (value.internal_interface_transfer_kg, (runtime.ng, 2)),
        (value.depletion_roundoff_positive_in_kg, (runtime.ng,)),
        (value.classwise_residual_kg, (runtime.ng,)),
        (value.interface_faces, (2,)),
    )
    expected_interface = (
        np.asarray(value.face_flux_kg_m_s)[:, runtime.faces_host]*
        np.asarray(runtime.grid.face_length_m)[runtime.faces_host][None, :]*
        float(value.dt_s))
    expected_residual = (
        np.sum(
            np.asarray(value.cell_delta_kg_m2)*
            np.asarray(runtime.grid.cell_area_m2)[None, :], axis=1)+
        np.sum(value.named_boundary_transfer_kg, axis=1)+
        np.sum(value.internal_interface_transfer_kg, axis=1)-
        np.asarray(value.depletion_roundoff_positive_in_kg))
    residual_scale = np.maximum(
        1.0,
        np.sum(
            np.abs(value.cell_delta_kg_m2)*
            np.asarray(runtime.grid.cell_area_m2)[None, :], axis=1)+
        np.sum(value.named_boundary_absolute_kg, axis=1)+
        np.sum(np.abs(value.internal_interface_transfer_kg), axis=1)+
        np.asarray(value.depletion_roundoff_positive_in_kg))
    if (value.execution_fingerprint != runtime.execution.fingerprint or
            value.runtime_receipt_sha256 !=
                runtime.runtime_receipt["receipt_sha256"] or
            value.q3_trace_receipt_sha256 !=
                runtime.q3_trace.receipt_sha256 or
            not math.isfinite(value.dt_s) or value.dt_s <= 0.0 or
            len(value.input_sha256) != 64 or
            value.method_fingerprint !=
                runtime.binding.core.bed_slope_execution["fingerprint"] or
            any(np.asarray(array).shape != shape or
                np.asarray(array).flags.writeable or
                not np.isfinite(array).all()
                for array, shape in shapes) or
            not np.array_equal(value.interface_faces, runtime.faces_host) or
            not np.array_equal(
                value.internal_interface_transfer_kg,
                expected_interface) or
            np.any(np.abs(value.classwise_residual_kg-expected_residual) >
                   8192.0*np.finfo(float).eps*residual_scale) or
            any(count != 1 for count in (
                value.provisional_face_batch_count,
                value.accepted_face_batch_count,
                value.outgoing_inventory_kernel_launch_count,
                value.delta_kernel_launch_count,
                value.interface_transfer_materialization_count,
                value.internal_interface_owner_count)) or
            value.named_boundary_kernel_launch_count != int(runtime.nb > 0) or
            value.physical_boundary_surrogate_count != 0 or
            value.python_numerical_cell_face_loop_count != 0 or
            value.accepted_bed_or_clock_publication_count != 0 or
            value.backend != BACKEND or
            value.result_sha256 != _hash(_result_payload(value))):
        raise CoupledAMRQ3FaceGPUV98Error(
            "v98 CUDA Q3 result identity differs")


def method_identity_coupled_amr_q3_face_gpu_v98() -> dict[str, object]:
    return {
        "scope_id": SCOPE_ID,
        "backend": BACKEND,
        "operator": "resident_selected_q3_with_two_private_trace_cells",
        "constitutive_recompute_count": 0,
        "provisional_face_batch_count": 1,
        "accepted_face_batch_count": 1,
        "internal_interface_owner_count": 1,
        "physical_boundary_surrogate_count": 0,
        "accepted_bed_or_clock_publication_count": 0,
        "actual_cuda_kernels": True,
        "automatic_cpu_fallback": False,
        "production_admission": False,
    }


__all__ = [
    "AMRQ3FaceResultGPUV98", "BACKEND",
    "CoupledAMRQ3FaceGPUV98Error", "ResidentAMRQ3FaceGPUV98", "SCOPE_ID",
    "method_identity_coupled_amr_q3_face_gpu_v98",
    "validate_amr_q3_face_result_gpu_v98",
]
