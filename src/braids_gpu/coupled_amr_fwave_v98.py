"""Actual-CUDA augmented-f-wave residual with a typed v98 AMR trace.

The historical v97 face pipeline first evaluates every level-local face.  Two
authenticated negative-neighbour faces are provisional reflective calls.  In
this additive route those two images are replaced, inside the same true
nonlinear residual and before cell gathering, by two calls to the *same*
resident augmented-f-wave and assembly kernels using the coarse trace.  The
discarded reflective calls and accepted replacements are counted separately.

The first bounded interface is deliberately narrow: density is exactly
continuous and shared-viscosity flux is exactly zero on the two interface
faces.  Scalar and Q3 interface ownership are later v98 phases; this module
publishes no state or clock and performs no CPU numerical fallback.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np

from .coupled_amr_trace_v98 import (
    CoupledAMRInternalTraceGPUV98,
    INTERFACE_FACE_COUNT,
    validate_coupled_amr_internal_trace_structure_gpu_v98,
)
from .coupled_amr_geometry_v98 import (
    CoupledAMRGeometryBindingGPUV98,
    validate_coupled_amr_geometry_binding_gpu_v98,
)
from .coupled_wetdry_v97 import (
    CoupledWetDryGPUExecutionV97,
    _grid,
)
from .implicit_midpoint_fwave import ImplicitStepCancelled, ImplicitStepRejected
from .wetdry_fwave_v97 import (
    MODE_WATER_PLUS_SEDIMENT,
    ResidentWetDryFwaveV97,
    WetDryFwaveGPUTrialV97,
    validate_wetdry_fwave_gpu_trial_v97,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 1
SCOPE_ID = "braids.q4.coupled_amr.interface_fwave.actual_cuda.v98"
BACKEND = "actual_cuda"


class CoupledAMRFwaveGPUV98Error(ValueError):
    """The resident v98 trace replacement or its audit differs."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError, OverflowError) as error:
        raise CoupledAMRFwaveGPUV98Error(
            "v98 CUDA interface identity is not canonical JSON") from error


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _digest(*values: object) -> str:
    digest = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(value)
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(_canonical(list(array.shape)))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _file_sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source_name(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(resolved)


def _poll(value: bool | Callable[[], bool]) -> Callable[[], bool]:
    if type(value) is bool:
        return lambda: value
    if callable(value):
        return lambda: bool(value())
    raise CoupledAMRFwaveGPUV98Error(
        "v98 CUDA interface cancellation must be bool or callable")


@dataclass(frozen=True)
class AMRInterfaceRateReceiptGPUV98:
    execution_fingerprint: str
    runtime_receipt_sha256: str
    trace_receipt_sha256: str
    substep_index: int
    evaluation_index: int
    trace_slice_sha256: str
    interface_faces: tuple[int, int]
    interface_riemann_call_count: int
    discarded_reflective_interface_riemann_call_count: int
    total_cuda_riemann_call_count: int
    physical_boundary_owner_count: int
    receipt_sha256: str


def _rate_payload(value: AMRInterfaceRateReceiptGPUV98) -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID+":rate",
        "execution": value.execution_fingerprint,
        "runtime": value.runtime_receipt_sha256,
        "trace": value.trace_receipt_sha256,
        "substep_index": value.substep_index,
        "evaluation_index": value.evaluation_index,
        "trace_slice_sha256": value.trace_slice_sha256,
        "interface_faces": list(value.interface_faces),
        "interface_riemann_call_count":
            value.interface_riemann_call_count,
        "discarded_reflective_interface_riemann_call_count":
            value.discarded_reflective_interface_riemann_call_count,
        "total_cuda_riemann_call_count":
            value.total_cuda_riemann_call_count,
        "physical_boundary_owner_count":
            value.physical_boundary_owner_count,
        "actual_cuda_kernel": True,
        "automatic_cpu_fallback": False,
    }


@dataclass(frozen=True)
class AMRInterfaceResidualAuditGPUV98:
    execution_fingerprint: str
    runtime_receipt_sha256: str
    trace_receipt_sha256: str
    substep_index: int
    interface_faces: tuple[int, int]
    rate_receipts: tuple[AMRInterfaceRateReceiptGPUV98, ...]
    evaluation_count: int
    interface_riemann_call_count: int
    discarded_reflective_interface_riemann_call_count: int
    total_cuda_riemann_call_count: int
    final_wetdry_trial_sha256: str
    final_limited_left_flux: np.ndarray
    final_limited_right_flux: np.ndarray
    final_full_limited_left_flux: np.ndarray
    final_full_limited_right_flux: np.ndarray
    physical_boundary_owner_count: int
    boundary_memory_consumer_count: int
    accepted_clock_increment_count: int
    audit_sha256: str


def _audit_payload(
    value: AMRInterfaceResidualAuditGPUV98,
) -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID+":residual-audit",
        "execution": value.execution_fingerprint,
        "runtime": value.runtime_receipt_sha256,
        "trace": value.trace_receipt_sha256,
        "substep_index": value.substep_index,
        "interface_faces": list(value.interface_faces),
        "rate_receipts": [item.receipt_sha256
                          for item in value.rate_receipts],
        "evaluation_count": value.evaluation_count,
        "interface_riemann_call_count":
            value.interface_riemann_call_count,
        "discarded_reflective_interface_riemann_call_count":
            value.discarded_reflective_interface_riemann_call_count,
        "total_cuda_riemann_call_count":
            value.total_cuda_riemann_call_count,
        "final_wetdry_trial_sha256":
            value.final_wetdry_trial_sha256,
        "final_flux_sha256": _digest(
            value.final_limited_left_flux,
            value.final_limited_right_flux,
            value.final_full_limited_left_flux,
            value.final_full_limited_right_flux),
        "physical_boundary_owner_count":
            value.physical_boundary_owner_count,
        "boundary_memory_consumer_count":
            value.boundary_memory_consumer_count,
        "accepted_clock_increment_count":
            value.accepted_clock_increment_count,
        "resident_final_only": True,
        "per_residual_host_readback_count": 0,
    }


@dataclass(frozen=True)
class WetDryAMRInterfaceGPUTrialV98:
    execution_fingerprint: str
    trace_receipt_sha256: str
    substep_index: int
    wetdry_trial: WetDryFwaveGPUTrialV97
    residual_audit: AMRInterfaceResidualAuditGPUV98
    density_continuous_interface_face_count: int
    zero_interface_viscosity_face_count: int
    interface_turbulence_flux_linf_m4_s2: float
    scalar_interface_owner_count: int
    q3_interface_owner_count: int
    state_or_clock_publication_count: int
    backend: str
    trial_sha256: str
    auxiliary_runtime_receipt_sha256: str | None = None
    auxiliary_density_face_augmentation_count: int = 0
    auxiliary_trace_v91_face_count: int = 0
    auxiliary_trace_v91_gather_count: int = 0
    auxiliary_final_density_face_acceleration_m_s2: np.ndarray | None = None
    auxiliary_final_density_mask: np.ndarray | None = None
    auxiliary_final_trace_v91_face_flux_m4_s2: np.ndarray | None = None
    auxiliary_final_trace_v91_rate_m2_s2: np.ndarray | None = None
    auxiliary_final_evidence_sha256: str | None = None


def _auxiliary_evidence_sha256(value: dict[str, object]) -> str:
    """Bind the final true-residual inputs and resident CUDA outputs."""
    density_result = value["density_result"]
    momentum_result = value["momentum_result"]
    arrays = (
        *value["density_inputs"], value["density_lookup"],
        *(density_result[key] for key in sorted(density_result)),
        *value["momentum_inputs"], value["momentum_lookup"],
        *(momentum_result[key] for key in sorted(momentum_result)),
    )
    return _hash({
        "arrays_sha256": _digest(*arrays),
        "runtime_receipt_sha256": value["runtime_receipt_sha256"],
        "launch_counts": value["launch_counts"],
        "density_evaluation_index": value["density_evaluation_index"],
        "momentum_evaluation_index": value["momentum_evaluation_index"],
        "numerical_operator_replay_count":
            value["numerical_operator_replay_count"],
    })


def _trial_payload(value: WetDryAMRInterfaceGPUTrialV98) -> dict[str, object]:
    result = {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID+":trial",
        "execution": value.execution_fingerprint,
        "trace": value.trace_receipt_sha256,
        "substep_index": value.substep_index,
        "wetdry_trial": value.wetdry_trial.trial_sha256,
        "residual_audit": value.residual_audit.audit_sha256,
        "density_continuous_interface_face_count":
            value.density_continuous_interface_face_count,
        "zero_interface_viscosity_face_count":
            value.zero_interface_viscosity_face_count,
        "interface_turbulence_flux_linf_m4_s2":
            value.interface_turbulence_flux_linf_m4_s2,
        "scalar_interface_owner_count": value.scalar_interface_owner_count,
        "q3_interface_owner_count": value.q3_interface_owner_count,
        "state_or_clock_publication_count":
            value.state_or_clock_publication_count,
        "backend": value.backend,
        "scope":
            "density_continuous_zero_interface_viscosity_hydraulics_only",
        "scalar_q3_interface_qualification": False,
    }
    if value.auxiliary_runtime_receipt_sha256 is not None:
        result.update(
            auxiliary_residual={
                "runtime_receipt_sha256":
                    value.auxiliary_runtime_receipt_sha256,
                "density_face_augmentation_count":
                    value.auxiliary_density_face_augmentation_count,
                "trace_v91_face_count":
                    value.auxiliary_trace_v91_face_count,
                "trace_v91_gather_count":
                    value.auxiliary_trace_v91_gather_count,
                "final_arrays_sha256": _digest(
                    value.auxiliary_final_density_face_acceleration_m_s2,
                    value.auxiliary_final_density_mask,
                    value.auxiliary_final_trace_v91_face_flux_m4_s2,
                    value.auxiliary_final_trace_v91_rate_m2_s2),
                "final_evidence_sha256":
                    value.auxiliary_final_evidence_sha256,
                "numerical_operator_replay_count": 0,
            },
            scope="joined_density_trace_and_precapped_v91_true_residual",
        )
    return result


class ResidentAMRInterfaceFwaveGPUV98:
    """Two-face resident replacement using the already compiled f-wave."""

    def __init__(
        self,
        hydraulic_runtime: ResidentWetDryFwaveV97,
        coupled_execution: CoupledWetDryGPUExecutionV97,
        trace: CoupledAMRInternalTraceGPUV98, *,
        substep_index: int,
        geometry_binding: CoupledAMRGeometryBindingGPUV98 | None = None,
        auxiliary_residual=None,
    ) -> None:
        if (type(hydraulic_runtime) is not ResidentWetDryFwaveV97 or
                type(coupled_execution) is not
                    CoupledWetDryGPUExecutionV97 or
                hydraulic_runtime.execution is not
                    coupled_execution.hydraulic_execution or
                type(substep_index) is not int or
                substep_index not in (0, 1)):
            raise CoupledAMRFwaveGPUV98Error(
                "v98 CUDA interface runtime binding differs")
        validate_coupled_amr_internal_trace_structure_gpu_v98(
            trace, coupled_execution)
        cp = hydraulic_runtime.cp
        controller = hydraulic_runtime.controller
        solver = controller.solver
        faces = np.asarray(trace.fine_faces, dtype=np.int32)
        grid = _grid(coupled_execution.active_execution)
        if (not np.all(grid.neighbour[faces] < 0) or
                not np.all(grid.boundary_index[faces] < 0) or
                not np.array_equal(
                    cp.asnumpy(controller.neighbour)[faces],
                    grid.neighbour[faces])):
            raise CoupledAMRFwaveGPUV98Error(
                "v98 CUDA selected faces are not internal trace faces")
        self.cp = cp
        self.hydraulic_runtime = hydraulic_runtime
        self.coupled_execution = coupled_execution
        self.execution = hydraulic_runtime.execution
        self.trace = trace
        self.substep_index = substep_index
        self.geometry_binding = geometry_binding
        self.auxiliary_residual = auxiliary_residual
        self.faces = tuple(int(face) for face in faces)
        self.face_index = cp.asarray(faces)
        self.fine_owner = controller.owner[self.face_index]
        self.trace_q = cp.ascontiguousarray(cp.asarray(
            trace.trace_q[substep_index], dtype=cp.float64))
        self.trace_bed = cp.ascontiguousarray(cp.asarray(
            trace.trace_bed_elevation_m[substep_index],
            dtype=cp.float64))
        source_density = (
            trace.trace_mixture_density_kg_m3
            if self.execution.density_mode == MODE_WATER_PLUS_SEDIMENT
            else trace.trace_water_density_kg_m3)
        self.trace_density = cp.ascontiguousarray(cp.asarray(
            source_density[substep_index], dtype=cp.float64))
        self.trace_wet = cp.ascontiguousarray(cp.asarray(
            trace.trace_wet_mask[substep_index], dtype=cp.int32))

        # Synthetic two-face internal topology.  Left cells are the two fine
        # owners; right cells are the two immutable coarse trace images.
        self.local_owner = cp.asarray((0, 1), dtype=cp.int32)
        self.local_neighbour = cp.asarray((2, 3), dtype=cp.int32)
        self.local_kind = cp.ones(INTERFACE_FACE_COUNT, dtype=cp.int32)
        self.local_boundary_mode = cp.zeros(
            INTERFACE_FACE_COUNT, dtype=cp.int32)
        self.local_sampled = cp.zeros(
            INTERFACE_FACE_COUNT, dtype=cp.int32)
        self.local_bedoffset = cp.zeros(
            INTERFACE_FACE_COUNT, dtype=cp.float64)
        self.local_q = cp.empty((3, 4), dtype=cp.float64)
        self.local_bed = cp.empty(4, dtype=cp.float64)
        self.local_trace = cp.empty(
            (10, INTERFACE_FACE_COUNT), dtype=cp.float64)
        self.local_normal = cp.ascontiguousarray(
            controller.normal[:, self.face_index])
        actual_distance = solver.flux.inputs["distance"]
        if geometry_binding is None:
            # Only the explicitly structural same-grid test uses this route.
            # A publishable v98 hierarchy must supply a joined 2:1 geometry.
            self.local_distance = cp.ascontiguousarray(
                actual_distance[:, self.face_index])
            self.geometry_qualified = False
            geometry_receipt = None
        else:
            validate_coupled_amr_geometry_binding_gpu_v98(
                geometry_binding)
            joined = geometry_binding.joined_geometry
            if (geometry_binding.coarse_execution_fingerprint !=
                    trace.coarse_execution_fingerprint or
                    geometry_binding.fine_execution_fingerprint !=
                    trace.fine_execution_fingerprint or
                    geometry_binding.fine_grid.fingerprint !=
                    grid.fingerprint or
                    joined.fine_faces != tuple(trace.fine_faces) or
                    not np.array_equal(
                        joined.fine_owner_cells,
                        np.asarray(grid.owner)[faces])):
                raise CoupledAMRFwaveGPUV98Error(
                    "v98 CUDA joined geometry and trace differ")
            owner_half = cp.asnumpy(
                actual_distance[0, self.face_index])
            total = np.asarray(joined.normal_centre_distance_m)
            donor_half = total-owner_half
            if (not np.isfinite(owner_half).all() or
                    not np.isfinite(donor_half).all() or
                    np.any(owner_half <= 0.0) or
                    np.any(donor_half <= 0.0)):
                raise CoupledAMRFwaveGPUV98Error(
                    "v98 CUDA joined centre-distance partition differs")
            self.local_distance = cp.ascontiguousarray(cp.asarray(
                np.vstack((owner_half, donor_half)), dtype=cp.float64))
            self.geometry_qualified = True
            geometry_receipt = geometry_binding.receipt_sha256
        if auxiliary_residual is not None:
            from .coupled_amr_auxiliary_residual_v98 import (
                ResidentAMRAuxiliaryResidualGPUV98,
            )
            if (type(auxiliary_residual) is not
                    ResidentAMRAuxiliaryResidualGPUV98 or
                    geometry_binding is None or
                    auxiliary_residual.hydraulic_runtime is not
                        hydraulic_runtime or
                    auxiliary_residual.execution is not coupled_execution or
                    auxiliary_residual.trace is not trace or
                    auxiliary_residual.geometry_binding is not
                        geometry_binding or
                    auxiliary_residual.substep_index != substep_index):
                raise CoupledAMRFwaveGPUV98Error(
                    "v98 CUDA auxiliary residual binding differs")
            auxiliary_residual.validate_runtime()
        self.local_hydro_constant = cp.ones(4, dtype=cp.int32)
        self.local = cp.empty((8, INTERFACE_FACE_COUNT), cp.float64)
        self.riemann = cp.empty((18, INTERFACE_FACE_COUNT), cp.float64)
        self.closed = cp.empty(INTERFACE_FACE_COUNT, cp.int32)
        self.augmented_status = cp.empty(
            INTERFACE_FACE_COUNT, cp.int32)
        self.face_status = cp.empty(INTERFACE_FACE_COUNT, cp.int32)
        self.left_flux = cp.empty((3, INTERFACE_FACE_COUNT), cp.float64)
        self.right_flux = cp.empty((3, INTERFACE_FACE_COUNT), cp.float64)
        self.speed = cp.empty(INTERFACE_FACE_COUNT, cp.float64)
        self.final_left = cp.zeros((3, INTERFACE_FACE_COUNT), cp.float64)
        self.final_right = cp.zeros((3, INTERFACE_FACE_COUNT), cp.float64)

        folder = Path(solver.folder).resolve()
        canonical = ROOT/"validation"/"gpu_fwave"
        relatives = ("augmented_fwave.cu", "resident_update.cu",
                     "reconstruction.cu")
        sources = {str((folder/name).resolve()): _file_sha(folder/name)
                   for name in relatives}
        canonical_hashes = {name: _file_sha(canonical/name)
                            for name in relatives}
        if any(sources[str((folder/name).resolve())] !=
               canonical_hashes[name] for name in relatives):
            raise CoupledAMRFwaveGPUV98Error(
                "v98 CUDA interface resident source differs")
        self.runtime_receipt = {
            "schema": SCHEMA,
            "scope_id": SCOPE_ID+":runtime",
            "execution_fingerprint": self.execution.fingerprint,
            "coupled_execution_fingerprint":
                coupled_execution.fingerprint,
            "trace_receipt_sha256": trace.receipt_sha256,
            "substep_index": substep_index,
            "faces": list(self.faces),
            "geometry_binding_receipt_sha256": geometry_receipt,
            "joined_cell_centre_distance": self.geometry_qualified,
            "backend": BACKEND,
            "actual_cuda_kernels": True,
            "reused_resident_augmented_fwave_module": True,
            "source_sha256": sources,
            "canonical_source_sha256": canonical_hashes,
            "wrapper_source_sha256": _file_sha(Path(__file__)),
            "v97_hook_source_sha256": _file_sha(
                Path(__file__).with_name("wetdry_fwave_v97.py")),
            "device_id": int(cp.cuda.Device().id),
            "per_residual_host_readback_count": 0,
            "automatic_cpu_fallback": False,
            "production_admission": False,
        }
        if auxiliary_residual is not None:
            self.runtime_receipt[
                "auxiliary_residual_runtime_receipt_sha256"] = \
                auxiliary_residual.receipt["receipt_sha256"]
        self.runtime_receipt["receipt_sha256"] = _hash(
            self.runtime_receipt)
        self._rate_receipts: list[AMRInterfaceRateReceiptGPUV98] = []
        self._capture_count = 0
        self._open = False
        self._density_checked = False
        self._final_trial_sha256 = None

    def _validate_runtime(self) -> None:
        solver = self.hydraulic_runtime.controller.solver
        folder = Path(solver.folder).resolve()
        relatives = ("augmented_fwave.cu", "resident_update.cu",
                     "reconstruction.cu")
        sources = {str((folder/name).resolve()): _file_sha(folder/name)
                   for name in relatives}
        receipt = self.runtime_receipt
        if (receipt.get("receipt_sha256") != _hash({
                key: value for key, value in receipt.items()
                if key != "receipt_sha256"}) or
                receipt.get("source_sha256") != sources or
                receipt.get("wrapper_source_sha256") !=
                    _file_sha(Path(__file__)) or
                receipt.get("v97_hook_source_sha256") != _file_sha(
                    Path(__file__).with_name("wetdry_fwave_v97.py")) or
                int(self.cp.cuda.Device().id) != receipt.get("device_id")):
            raise CoupledAMRFwaveGPUV98Error(
                "v98 CUDA interface runtime/source identity differs")
        validate_coupled_amr_internal_trace_structure_gpu_v98(
            self.trace, self.coupled_execution)
        actual_distance = solver.flux.inputs["distance"]
        if self.geometry_binding is None:
            expected_distance = self.cp.asnumpy(
                actual_distance[:, self.face_index])
        else:
            validate_coupled_amr_geometry_binding_gpu_v98(
                self.geometry_binding)
            owner_half = self.cp.asnumpy(
                actual_distance[0, self.face_index])
            expected_distance = np.vstack((
                owner_half,
                np.asarray(self.geometry_binding.joined_geometry.
                           normal_centre_distance_m)-owner_half))
        if not np.array_equal(
                self.cp.asnumpy(self.local_distance), expected_distance):
            raise CoupledAMRFwaveGPUV98Error(
                "v98 CUDA resident joined distance differs")
        expected_auxiliary = (None if self.auxiliary_residual is None else
            self.auxiliary_residual.receipt["receipt_sha256"])
        if receipt.get(
                "auxiliary_residual_runtime_receipt_sha256") != \
                expected_auxiliary:
            raise CoupledAMRFwaveGPUV98Error(
                "v98 CUDA auxiliary runtime receipt differs")
        if self.auxiliary_residual is not None:
            self.auxiliary_residual.validate_runtime()

    def begin_residual_sequence(self) -> None:
        self._validate_runtime()
        if (self._open or
                "_launch_face_density" in
                    self.hydraulic_runtime.__dict__ or
                "_launch_projection" in
                    self.hydraulic_runtime.__dict__ or
                (self.hydraulic_runtime.turbulence is not None and
                 "evaluate" in
                    self.hydraulic_runtime.turbulence.__dict__)):
            raise CoupledAMRFwaveGPUV98Error(
                "v98 CUDA interface residual sequence or adapter is stale")
        if self.auxiliary_residual is not None:
            self.auxiliary_residual.reset_residual_sequence()
        self._rate_receipts.clear()
        self._capture_count = 0
        self._density_checked = False
        self._final_trial_sha256 = None
        self.final_left.fill(0.0)
        self.final_right.fill(0.0)
        self._open = True

    def abort_residual_sequence(self) -> None:
        self._rate_receipts.clear()
        self._capture_count = 0
        self._density_checked = False
        self._final_trial_sha256 = None
        self.final_left.fill(0.0)
        self.final_right.fill(0.0)
        if self.auxiliary_residual is not None:
            self.auxiliary_residual.reset_residual_sequence()
        self._open = False

    def _launch(self, name: str, args: tuple[object, ...]) -> None:
        kernel = self.hydraulic_runtime.controller.solver.flux.kernels[name]
        kernel((1,), (128,), args)

    def replace_reflective_faces(
        self, *, q_eval, bed_elevation_m, reconstructed_trace,
        selected_density_kg_m3, flux_buffers,
        cancelled: bool | Callable[[], bool],
    ) -> None:
        poll = _poll(cancelled)
        if poll():
            raise ImplicitStepCancelled(
                "v98 CUDA interface residual cancelled before replacement")
        if not self._open:
            raise CoupledAMRFwaveGPUV98Error(
                "v98 CUDA interface residual sequence is not open")
        self._validate_runtime()
        cp = self.cp
        solver = self.hydraulic_runtime.controller.solver
        if not self._density_checked:
            density_continuous = bool(cp.array_equal(
                selected_density_kg_m3[self.fine_owner],
                self.trace_density).item())
            if ((not density_continuous and
                 self.auxiliary_residual is None) or
                    not bool(cp.all(
                        solver.flux.inputs["kind"][self.face_index] == 2
                    ).item())):
                raise ImplicitStepRejected(
                    "v98 CUDA interface density route or provisional "
                    "reflective faces differ")
            self._density_checked = True

        self.local_q[:, :2] = q_eval[:, self.fine_owner]
        self.local_q[:, 2:] = self.trace_q
        self.local_bed[:2] = bed_elevation_m[self.fine_owner]
        self.local_bed[2:] = self.trace_bed
        self.local_trace[:4] = reconstructed_trace[:4, self.face_index]
        self.local_trace[4:7] = self.trace_q
        self.local_trace[7] = self.trace_bed
        self.local_trace[8] = reconstructed_trace[8, self.face_index]
        self.local_trace[9] = self.trace_q[0]
        self.local_hydro_constant[:2] = (
            solver.flux.inputs["hydro_constant"][self.fine_owner])
        self.local_hydro_constant[2:] = 1
        nf, nc = np.int64(INTERFACE_FACE_COUNT), np.int64(4)
        grav = np.float64(
            self.execution.density_execution.gravity_m_s2)
        dry = np.float64(
            self.execution.density_execution.dry_tolerance_m)
        self._launch("resident_prepare_faces", (
            self.local_trace, self.local_q, self.local_bed,
            self.local_normal, self.local_distance, self.local_owner,
            self.local_neighbour, self.local_kind, nc, nf, grav, dry,
            self.local, self.closed))
        self._launch("augmented_fwave_local", (
            self.local, nf, grav, dry, self.riemann,
            self.augmented_status))
        self._launch("resident_assemble_flux", (
            self.local_trace, self.local, self.riemann, self.local_q,
            self.local_bed, self.local_normal, self.local_owner,
            self.local_neighbour, self.local_kind,
            self.local_boundary_mode, self.local_sampled,
            self.local_bedoffset, self.local_hydro_constant, self.closed,
            nc, nf, grav, dry, self.left_flux, self.right_flux,
            self.speed, self.face_status))
        if (bool(cp.any(self.augmented_status != 0).item()) or
                bool(cp.any(self.face_status != 0).item()) or
                bool(cp.any(self.closed != 0).item())):
            raise ImplicitStepRejected(
                "v98 CUDA interface augmented-f-wave replacement failed")
        flux_buffers["fl"][:, self.face_index] = self.left_flux
        flux_buffers["fr"][:, self.face_index] = self.right_flux
        flux_buffers["speed"][self.face_index] = self.speed
        flux_buffers["augmented_status"][self.face_index] = (
            self.augmented_status)
        flux_buffers["face_status"][self.face_index] = self.face_status
        flux_buffers["closed"][self.face_index] = self.closed
        index = len(self._rate_receipts)
        trace_slice = _digest(
            self.trace.trace_q[self.substep_index],
            self.trace.trace_bed_elevation_m[self.substep_index],
            (self.trace.trace_mixture_density_kg_m3
             if self.execution.density_mode == MODE_WATER_PLUS_SEDIMENT
             else self.trace.trace_water_density_kg_m3)[
                 self.substep_index],
            self.trace.trace_wet_mask[self.substep_index])
        placeholder = AMRInterfaceRateReceiptGPUV98(
            self.execution.fingerprint,
            self.runtime_receipt["receipt_sha256"],
            self.trace.receipt_sha256, self.substep_index, index,
            trace_slice, self.faces, INTERFACE_FACE_COUNT,
            INTERFACE_FACE_COUNT,
            self.execution.face_count+INTERFACE_FACE_COUNT, 0, "")
        self._rate_receipts.append(AMRInterfaceRateReceiptGPUV98(
            **{**placeholder.__dict__,
               "receipt_sha256": _hash(_rate_payload(placeholder))}))
        if poll():
            raise ImplicitStepCancelled(
                "v98 CUDA interface residual cancelled after replacement")

    def capture_limited_faces(
        self, *, flux_buffers,
        cancelled: bool | Callable[[], bool],
    ) -> None:
        poll = _poll(cancelled)
        if poll():
            raise ImplicitStepCancelled(
                "v98 CUDA interface residual cancelled before capture")
        if (not self._open or
                len(self._rate_receipts) != self._capture_count+1):
            raise CoupledAMRFwaveGPUV98Error(
                "v98 CUDA interface replacement/capture order differs")
        self.cp.copyto(
            self.final_left, flux_buffers["fl"][:, self.face_index])
        self.cp.copyto(
            self.final_right, flux_buffers["fr"][:, self.face_index])
        self._capture_count += 1

    def finish_residual_sequence(
        self, trial: WetDryFwaveGPUTrialV97,
    ) -> AMRInterfaceResidualAuditGPUV98:
        validate_wetdry_fwave_gpu_trial_v97(trial, self.execution)
        if (not self._open or not self._rate_receipts or
                len(self._rate_receipts) !=
                    trial.true_residual_evaluation_count or
                self._capture_count != trial.true_residual_evaluation_count):
            raise CoupledAMRFwaveGPUV98Error(
                "v98 CUDA interface residual count differs")
        cp = self.cp
        left = np.ascontiguousarray(cp.asnumpy(self.final_left))
        right = np.ascontiguousarray(cp.asnumpy(self.final_right))
        resident_left = cp.asnumpy(
            self.hydraulic_runtime.v98_final_augmented_left_flux[
                :, self.face_index])
        resident_right = cp.asnumpy(
            self.hydraulic_runtime.v98_final_augmented_right_flux[
                :, self.face_index])
        full_left = np.ascontiguousarray(cp.asnumpy(
            self.hydraulic_runtime.v98_final_augmented_left_flux))
        full_right = np.ascontiguousarray(cp.asnumpy(
            self.hydraulic_runtime.v98_final_augmented_right_flux))
        if (not np.array_equal(left, resident_left) or
                not np.array_equal(right, resident_right) or
                not np.array_equal(
                    left[0],
                    trial.accepted_limited_face_water_flux_m2_s[
                        np.asarray(self.faces)])):
            raise CoupledAMRFwaveGPUV98Error(
                "v98 CUDA final resident interface flux differs")
        for array in (left, right, full_left, full_right):
            array.setflags(write=False)
        receipts = tuple(self._rate_receipts)
        count = len(receipts)
        placeholder = AMRInterfaceResidualAuditGPUV98(
            self.execution.fingerprint,
            self.runtime_receipt["receipt_sha256"],
            self.trace.receipt_sha256, self.substep_index, self.faces,
            receipts, count, INTERFACE_FACE_COUNT*count,
            INTERFACE_FACE_COUNT*count,
            (self.execution.face_count+INTERFACE_FACE_COUNT)*count,
            trial.trial_sha256, left, right, full_left, full_right,
            0, 0, 0, "")
        result = AMRInterfaceResidualAuditGPUV98(
            **{**placeholder.__dict__,
               "audit_sha256": _hash(_audit_payload(placeholder))})
        self._final_trial_sha256 = trial.trial_sha256
        self._open = False
        validate_amr_interface_residual_audit_gpu_v98(
            result, self, trial)
        return result


def validate_amr_interface_residual_audit_gpu_v98(
    value: AMRInterfaceResidualAuditGPUV98,
    runtime: ResidentAMRInterfaceFwaveGPUV98,
    trial: WetDryFwaveGPUTrialV97,
) -> None:
    if (type(value) is not AMRInterfaceResidualAuditGPUV98 or
            type(runtime) is not ResidentAMRInterfaceFwaveGPUV98 or
            type(trial) is not WetDryFwaveGPUTrialV97):
        raise CoupledAMRFwaveGPUV98Error(
            "v98 CUDA interface audit type differs")
    runtime._validate_runtime()
    validate_wetdry_fwave_gpu_trial_v97(trial, runtime.execution)
    count = trial.true_residual_evaluation_count
    arrays = (value.final_limited_left_flux,
              value.final_limited_right_flux)
    full_arrays = (value.final_full_limited_left_flux,
                   value.final_full_limited_right_flux)
    if (value.execution_fingerprint != runtime.execution.fingerprint or
            value.runtime_receipt_sha256 !=
                runtime.runtime_receipt["receipt_sha256"] or
            value.trace_receipt_sha256 != runtime.trace.receipt_sha256 or
            value.substep_index != runtime.substep_index or
            value.interface_faces != runtime.faces or
            value.evaluation_count != count or
            len(value.rate_receipts) != count or
            value.interface_riemann_call_count !=
                INTERFACE_FACE_COUNT*count or
            value.discarded_reflective_interface_riemann_call_count !=
                INTERFACE_FACE_COUNT*count or
            value.total_cuda_riemann_call_count !=
                (runtime.execution.face_count+INTERFACE_FACE_COUNT)*count or
            value.final_wetdry_trial_sha256 != trial.trial_sha256 or
            any(array.shape != (3, INTERFACE_FACE_COUNT) or
                array.flags.writeable or not np.isfinite(array).all()
                for array in arrays) or
            any(array.shape != (3, runtime.execution.face_count) or
                array.flags.writeable or not np.isfinite(array).all()
                for array in full_arrays) or
            not np.array_equal(
                value.final_full_limited_left_flux[:,
                    np.asarray(runtime.faces)],
                value.final_limited_left_flux) or
            not np.array_equal(
                value.final_full_limited_right_flux[:,
                    np.asarray(runtime.faces)],
                value.final_limited_right_flux) or
            not np.array_equal(
                value.final_full_limited_left_flux[0],
                trial.accepted_limited_face_water_flux_m2_s) or
            not np.array_equal(
                value.final_full_limited_right_flux[0],
                trial.accepted_limited_face_water_flux_m2_s) or
            not np.array_equal(
                value.final_limited_left_flux[0],
                trial.accepted_limited_face_water_flux_m2_s[
                    np.asarray(runtime.faces)]) or
            value.physical_boundary_owner_count != 0 or
            value.boundary_memory_consumer_count != 0 or
            value.accepted_clock_increment_count != 0 or
            value.audit_sha256 != _hash(_audit_payload(value)) or
            any(type(item) is not AMRInterfaceRateReceiptGPUV98 or
                item.evaluation_index != index or
                item.execution_fingerprint != runtime.execution.fingerprint or
                item.runtime_receipt_sha256 !=
                    runtime.runtime_receipt["receipt_sha256"] or
                item.trace_receipt_sha256 != runtime.trace.receipt_sha256 or
                item.substep_index != runtime.substep_index or
                item.interface_faces != runtime.faces or
                item.interface_riemann_call_count != INTERFACE_FACE_COUNT or
                item.discarded_reflective_interface_riemann_call_count !=
                    INTERFACE_FACE_COUNT or
                item.total_cuda_riemann_call_count !=
                    runtime.execution.face_count+INTERFACE_FACE_COUNT or
                item.physical_boundary_owner_count != 0 or
                item.receipt_sha256 != _hash(_rate_payload(item))
                for index, item in enumerate(value.rate_receipts))):
        raise CoupledAMRFwaveGPUV98Error(
            "v98 CUDA interface residual audit differs")


@contextmanager
def _installed_auxiliary_residual_adapters_gpu_v98(
    runtime: ResidentWetDryFwaveV97,
    interface_runtime: ResidentAMRInterfaceFwaveGPUV98,
    poll: Callable[[], bool],
):
    """Temporarily compose trace density/v91 into the unchanged v97 route.

    The density face image is augmented before the one historical projection.
    Its combined mask is restored to the historical internal mask before the
    ordinary v91 operator runs.  The trace v91 contribution is then added to
    that operator's resident result without a second cap or clock owner.
    """
    auxiliary = interface_runtime.auxiliary_residual
    if auxiliary is None:
        yield
        return
    from .shared_turbulence_wet_active_v97 import (
        ResidentWetActiveMomentumOperatorV97,
    )
    turbulence = runtime.turbulence
    if (type(turbulence) is not ResidentWetActiveMomentumOperatorV97 or
            not runtime.execution.shared_turbulence_enabled or
            any(name in runtime.__dict__ for name in (
                "_launch_face_density", "_launch_projection")) or
            "evaluate" in turbulence.__dict__):
        raise CoupledAMRFwaveGPUV98Error(
            "v98 CUDA auxiliary residual adapter binding differs")
    cp = runtime.cp
    original_density = runtime._launch_face_density
    original_projection = runtime._launch_projection
    original_turbulence = turbulence.evaluate
    base_density_mask = None

    def launch_face_density(depth, selected_density) -> None:
        nonlocal base_density_mask
        if base_density_mask is not None:
            raise CoupledAMRFwaveGPUV98Error(
                "v98 CUDA density adapter has an unprojected image")
        original_density(depth, selected_density)
        base_density_mask = runtime.face_active.copy()
        result = auxiliary.augment_density_resident(
            depth, selected_density, runtime.face_acceleration,
            base_density_mask, runtime.wet, cancelled=poll)
        cp.copyto(runtime.face_acceleration, result["face_acceleration"])
        cp.copyto(runtime.face_active, result["density_mask"])

    def launch_projection() -> None:
        nonlocal base_density_mask
        if base_density_mask is None:
            raise CoupledAMRFwaveGPUV98Error(
                "v98 CUDA density projection lacks its base mask")
        try:
            original_projection()
        finally:
            cp.copyto(runtime.face_active, base_density_mask)
            base_density_mask = None

    def evaluate_turbulence(
        q, capped_vicuv, wet_cell_mask, active_wet_wet_face_mask, *,
        cancelled=False,
    ):
        if base_density_mask is not None:
            raise CoupledAMRFwaveGPUV98Error(
                "v98 CUDA ordinary v91 saw the density-augmented mask")
        ordinary = original_turbulence(
            q, capped_vicuv, wet_cell_mask,
            active_wet_wet_face_mask, cancelled=cancelled)
        trace = auxiliary.evaluate_momentum_resident(
            q, capped_vicuv, wet_cell_mask, cancelled=cancelled)
        combined_rate = ordinary["momentum_rate_m2_s2"].copy()
        combined_flux = ordinary["face_flux_m4_s2"].copy()
        combined_rate[1:] += trace["momentum_rate"]
        combined_flux += trace["face_flux"]
        result = dict(ordinary)
        result["momentum_rate_m2_s2"] = combined_rate
        result["face_flux_m4_s2"] = combined_flux
        return result

    runtime._launch_face_density = launch_face_density
    runtime._launch_projection = launch_projection
    turbulence.evaluate = evaluate_turbulence
    try:
        yield
    finally:
        if base_density_mask is not None:
            cp.copyto(runtime.face_active, base_density_mask)
        runtime.__dict__.pop("_launch_face_density", None)
        runtime.__dict__.pop("_launch_projection", None)
        turbulence.__dict__.pop("evaluate", None)


def _validate_density_continuity(
    runtime: ResidentAMRInterfaceFwaveGPUV98,
    water: object,
    mixture: object,
) -> int:
    nc = runtime.execution.cell_count
    water_host = np.asarray(water, dtype=np.float64)
    mixture_host = np.asarray(mixture, dtype=np.float64)
    if (water_host.shape != (nc,) or mixture_host.shape != (nc,) or
            not np.isfinite(water_host).all() or
            not np.isfinite(mixture_host).all() or
            np.any(water_host <= 0.0) or
            np.any(mixture_host < water_host)):
        raise CoupledAMRFwaveGPUV98Error(
            "v98 CUDA interface accepted density differs")
    selected = (mixture_host
                if runtime.execution.density_mode ==
                MODE_WATER_PLUS_SEDIMENT else water_host)
    source = (runtime.trace.trace_mixture_density_kg_m3
              if runtime.execution.density_mode ==
              MODE_WATER_PLUS_SEDIMENT else
              runtime.trace.trace_water_density_kg_m3)
    owner = runtime.execution.density_execution.owner[
        np.asarray(runtime.faces)]
    equal = selected[owner] == source[runtime.substep_index]
    if (not np.all(equal) and runtime.auxiliary_residual is None):
        raise CoupledAMRFwaveGPUV98Error(
            "v98 CUDA first interface seam requires exact density continuity")
    return int(np.count_nonzero(equal))


def solve_wetdry_midpoint_with_external_trace_gpu_v98(
    runtime: ResidentWetDryFwaveV97,
    parent_q, *,
    accepted_water_density_kg_m3,
    accepted_mixture_density_kg_m3,
    bed_elevation_m,
    dt_s: float,
    evaluation_time_s: float,
    interface_runtime: ResidentAMRInterfaceFwaveGPUV98,
    cancelled: bool | Callable[[], bool] = False,
) -> WetDryAMRInterfaceGPUTrialV98:
    if (type(runtime) is not ResidentWetDryFwaveV97 or
            type(interface_runtime) is not
                ResidentAMRInterfaceFwaveGPUV98 or
            interface_runtime.hydraulic_runtime is not runtime):
        raise CoupledAMRFwaveGPUV98Error(
            "v98 CUDA interface solve/runtime binding differs")
    density_continuous_count = _validate_density_continuity(
        interface_runtime, accepted_water_density_kg_m3,
        accepted_mixture_density_kg_m3)
    poll = _poll(cancelled)
    if poll():
        raise InterruptedError(
            "v98 CUDA interface solve cancelled before child work")
    interface_runtime.begin_residual_sequence()
    runtime._bind_amr_interface_v98(interface_runtime)
    completed = False
    try:
        with _installed_auxiliary_residual_adapters_gpu_v98(
                runtime, interface_runtime, poll):
            trial = runtime.solve(
                parent_q,
                accepted_water_density_kg_m3=
                    accepted_water_density_kg_m3,
                accepted_mixture_density_kg_m3=
                    accepted_mixture_density_kg_m3,
                bed_elevation_m=bed_elevation_m, dt_s=dt_s,
                evaluation_time_s=evaluation_time_s, cancelled=poll)
        audit = interface_runtime.finish_residual_sequence(trial)
        faces = np.asarray(interface_runtime.faces)
        interface_flux_linf = float(np.max(np.abs(
            trial.shared_turbulence_face_flux[:, faces])))
        auxiliary = interface_runtime.auxiliary_residual
        auxiliary_kwargs: dict[str, object] = {}
        if auxiliary is None:
            zero_viscosity_count = INTERFACE_FACE_COUNT
        else:
            evidence = auxiliary.final_evidence_host()
            count = trial.true_residual_evaluation_count
            launches = evidence["launch_counts"]
            if (evidence["runtime_receipt_sha256"] !=
                    auxiliary.receipt["receipt_sha256"] or
                    evidence["density_evaluation_index"] != count or
                    evidence["momentum_evaluation_index"] != count or
                    evidence["numerical_operator_replay_count"] != 0 or
                    launches != {
                        "density_face_augmentation": count,
                        "trace_v91_momentum_faces": count,
                        "trace_v91_momentum_gathers": count,
                        "density_projection": 0,
                        "turbulence_cap": 0,
                        "hydraulic_riemann": 0,
                        "state_or_clock_publication": 0,
                    }):
                raise CoupledAMRFwaveGPUV98Error(
                    "v98 CUDA auxiliary residual launch evidence differs")
            density_acceleration = np.array(
                evidence["density_result"][
                    "selected_face_acceleration"],
                dtype=np.float64, copy=True, order="C")
            density_mask = np.array(
                evidence["density_result"]["selected_density_mask"],
                dtype=np.int32, copy=True, order="C")
            trace_flux = np.array(
                evidence["momentum_result"]["selected_face_flux"],
                dtype=np.float64, copy=True, order="C")
            trace_rate = np.array(
                evidence["momentum_result"]["momentum_rate"],
                dtype=np.float64, copy=True, order="C")
            for array in (density_acceleration, density_mask,
                          trace_flux, trace_rate):
                array.setflags(write=False)
            momentum_inputs = evidence["momentum_inputs"]
            owner = interface_runtime.execution.density_execution.owner[faces]
            mean_viscosity = .5*(
                momentum_inputs[1][owner]+momentum_inputs[7])
            zero_viscosity_count = int(np.count_nonzero(
                mean_viscosity == 0.0))
            auxiliary_kwargs = {
                "auxiliary_runtime_receipt_sha256":
                    auxiliary.receipt["receipt_sha256"],
                "auxiliary_density_face_augmentation_count": count,
                "auxiliary_trace_v91_face_count": count,
                "auxiliary_trace_v91_gather_count": count,
                "auxiliary_final_density_face_acceleration_m_s2":
                    density_acceleration,
                "auxiliary_final_density_mask": density_mask,
                "auxiliary_final_trace_v91_face_flux_m4_s2": trace_flux,
                "auxiliary_final_trace_v91_rate_m2_s2": trace_rate,
                "auxiliary_final_evidence_sha256":
                    _auxiliary_evidence_sha256(evidence),
            }
        placeholder = WetDryAMRInterfaceGPUTrialV98(
            runtime.execution.fingerprint,
            interface_runtime.trace.receipt_sha256,
            interface_runtime.substep_index, trial, audit,
            density_continuous_count, zero_viscosity_count,
            interface_flux_linf, 0, 0, 0, BACKEND, "",
            **auxiliary_kwargs)
        result = WetDryAMRInterfaceGPUTrialV98(
            **{**placeholder.__dict__,
               "trial_sha256": _hash(_trial_payload(placeholder))})
        validate_wetdry_amr_interface_trial_gpu_v98(
            result, runtime=interface_runtime,
            accepted_water_density_kg_m3=
                accepted_water_density_kg_m3,
            accepted_mixture_density_kg_m3=
                accepted_mixture_density_kg_m3)
        completed = True
        return result
    except ImplicitStepCancelled as error:
        raise InterruptedError(str(error)) from error
    finally:
        runtime._unbind_amr_interface_v98(interface_runtime)
        if not completed:
            interface_runtime.abort_residual_sequence()


def validate_wetdry_amr_interface_trial_gpu_v98(
    value: WetDryAMRInterfaceGPUTrialV98, *,
    runtime: ResidentAMRInterfaceFwaveGPUV98,
    accepted_water_density_kg_m3,
    accepted_mixture_density_kg_m3,
) -> None:
    if (type(value) is not WetDryAMRInterfaceGPUTrialV98 or
            type(runtime) is not ResidentAMRInterfaceFwaveGPUV98):
        raise CoupledAMRFwaveGPUV98Error(
            "v98 CUDA interface trial/runtime type differs")
    density_continuous_count = _validate_density_continuity(
        runtime, accepted_water_density_kg_m3,
        accepted_mixture_density_kg_m3)
    validate_wetdry_fwave_gpu_trial_v97(
        value.wetdry_trial, runtime.execution)
    validate_amr_interface_residual_audit_gpu_v98(
        value.residual_audit, runtime, value.wetdry_trial)
    faces = np.asarray(runtime.faces)
    interface_linf = float(np.max(np.abs(
        value.wetdry_trial.shared_turbulence_face_flux[:, faces])))
    auxiliary = runtime.auxiliary_residual
    auxiliary_invalid = False
    expected_zero_viscosity_count = INTERFACE_FACE_COUNT
    if auxiliary is None:
        auxiliary_invalid = (
            value.auxiliary_runtime_receipt_sha256 is not None or
            value.auxiliary_density_face_augmentation_count != 0 or
            value.auxiliary_trace_v91_face_count != 0 or
            value.auxiliary_trace_v91_gather_count != 0 or
            value.auxiliary_final_density_face_acceleration_m_s2 is not None or
            value.auxiliary_final_density_mask is not None or
            value.auxiliary_final_trace_v91_face_flux_m4_s2 is not None or
            value.auxiliary_final_trace_v91_rate_m2_s2 is not None or
            value.auxiliary_final_evidence_sha256 is not None or
            interface_linf != 0.0)
    else:
        evidence = auxiliary.final_evidence_host()
        count = value.wetdry_trial.true_residual_evaluation_count
        launches = evidence["launch_counts"]
        dinputs = evidence["density_inputs"]
        minputs = evidence["momentum_inputs"]
        dresult = evidence["density_result"]
        mresult = evidence["momentum_result"]
        selected_density = np.asarray(
            accepted_mixture_density_kg_m3
            if runtime.execution.density_mode == MODE_WATER_PLUS_SEDIMENT
            else accepted_water_density_kg_m3, dtype=np.float64)
        owner = runtime.execution.density_execution.owner[faces]
        trace_depth = runtime.trace.trace_q[runtime.substep_index, 0]
        trace_density = (
            runtime.trace.trace_mixture_density_kg_m3
            if runtime.execution.density_mode == MODE_WATER_PLUS_SEDIMENT
            else runtime.trace.trace_water_density_kg_m3
        )[runtime.substep_index]
        trace_wet = runtime.trace.trace_wet_mask[runtime.substep_index]
        chosen = ((value.wetdry_trial.frozen_wet_mask[owner] != 0) &
                  (trace_wet != 0))
        expected_density_acceleration = np.zeros(INTERFACE_FACE_COUNT)
        distance = dinputs[2][faces]
        expected_density_acceleration[chosen] = (
            -runtime.execution.density_execution.gravity_m_s2*
            (trace_density[chosen]-selected_density[owner[chosen]])*
            (trace_depth[chosen]+value.wetdry_trial.midpoint_q[0,
                                                             owner[chosen]])/
            (4*runtime.execution.density_execution.
             hydraulic_reference_density_kg_m3*distance[chosen]))
        density_scale = np.maximum(
            np.finfo(float).tiny,
            np.abs(expected_density_acceleration))
        q = minputs[0]
        capped = minputs[1]
        donor_capped = minputs[7]
        coefficient = (
            .5*(q[0, owner]+minputs[6][0])*
            .5*(capped[owner]+donor_capped)*
            runtime.execution.density_execution.face_length_m[faces]/
            minputs[3][faces])
        expected_trace_flux = np.zeros((2, INTERFACE_FACE_COUNT))
        expected_trace_flux[:, chosen] = (
            coefficient[chosen]*(
                minputs[6][1:, chosen]/minputs[6][0, chosen]-
                q[1:, owner[chosen]]/q[0, owner[chosen]]))
        flux_scale = np.maximum(
            np.finfo(float).tiny, np.abs(expected_trace_flux))
        projection = auxiliary.geometry.projection
        cell = np.repeat(
            np.arange(runtime.execution.cell_count),
            np.diff(projection.offset))
        full_flux = mresult["face_flux"]
        expected_trace_rate = np.vstack([
            np.bincount(
                cell,
                weights=full_flux[k, projection.faces]*projection.signs,
                minlength=runtime.execution.cell_count)
            for k in (0, 1)
        ])/minputs[2]
        expected_trace_rate[:, minputs[4] == 0] = 0.0
        rate_scale = np.maximum(
            np.finfo(float).tiny, np.abs(expected_trace_rate))
        expected_zero_viscosity_count = int(np.count_nonzero(
            .5*(capped[owner]+donor_capped) == 0.0))
        stored_arrays = (
            (value.auxiliary_final_density_face_acceleration_m_s2,
             (INTERFACE_FACE_COUNT,)),
            (value.auxiliary_final_density_mask,
             (INTERFACE_FACE_COUNT,)),
            (value.auxiliary_final_trace_v91_face_flux_m4_s2,
             (2, INTERFACE_FACE_COUNT)),
            (value.auxiliary_final_trace_v91_rate_m2_s2,
             (2, runtime.execution.cell_count)),
        )
        auxiliary_invalid = (
            value.auxiliary_runtime_receipt_sha256 !=
                auxiliary.receipt["receipt_sha256"] or
            value.auxiliary_density_face_augmentation_count != count or
            value.auxiliary_trace_v91_face_count != count or
            value.auxiliary_trace_v91_gather_count != count or
            value.auxiliary_final_evidence_sha256 !=
                _auxiliary_evidence_sha256(evidence) or
            evidence["runtime_receipt_sha256"] !=
                auxiliary.receipt["receipt_sha256"] or
            evidence["density_evaluation_index"] != count or
            evidence["momentum_evaluation_index"] != count or
            evidence["numerical_operator_replay_count"] != 0 or
            launches != {
                "density_face_augmentation": count,
                "trace_v91_momentum_faces": count,
                "trace_v91_momentum_gathers": count,
                "density_projection": 0,
                "turbulence_cap": 0,
                "hydraulic_riemann": 0,
                "state_or_clock_publication": 0,
            } or
            any(array is None or array.shape != shape or
                array.flags.writeable or not np.isfinite(array).all()
                for array, shape in stored_arrays) or
            not np.array_equal(
                dinputs[0], value.wetdry_trial.midpoint_q[0]) or
            not np.array_equal(dinputs[1], selected_density) or
            not np.array_equal(
                dinputs[4], value.wetdry_trial.frozen_wet_mask) or
            not np.array_equal(
                dinputs[5], value.wetdry_trial.active_wet_wet_face_mask) or
            not np.array_equal(dinputs[6], faces) or
            not np.array_equal(dinputs[7], trace_depth) or
            not np.array_equal(dinputs[8], trace_density) or
            not np.array_equal(dinputs[9], trace_wet) or
            not np.array_equal(minputs[0], value.wetdry_trial.midpoint_q) or
            not np.array_equal(
                minputs[4], value.wetdry_trial.frozen_wet_mask) or
            not np.array_equal(minputs[5], faces) or
            not np.array_equal(
                minputs[6], runtime.trace.trace_q[runtime.substep_index]) or
            not np.array_equal(minputs[8], trace_wet) or
            not np.array_equal(
                value.auxiliary_final_density_face_acceleration_m_s2,
                dresult["selected_face_acceleration"]) or
            not np.array_equal(
                value.auxiliary_final_density_mask,
                dresult["selected_density_mask"]) or
            not np.array_equal(
                value.auxiliary_final_trace_v91_face_flux_m4_s2,
                mresult["selected_face_flux"]) or
            not np.array_equal(
                value.auxiliary_final_trace_v91_rate_m2_s2,
                mresult["momentum_rate"]) or
            not np.array_equal(
                value.auxiliary_final_density_mask,
                chosen.astype(np.int32)) or
            np.any(np.abs(
                value.auxiliary_final_density_face_acceleration_m_s2-
                expected_density_acceleration) >
                32*np.finfo(float).eps*density_scale) or
            np.any(np.abs(
                value.auxiliary_final_trace_v91_face_flux_m4_s2-
                expected_trace_flux) >
                64*np.finfo(float).eps*flux_scale) or
            np.any(np.abs(
                value.auxiliary_final_trace_v91_rate_m2_s2-
                expected_trace_rate) >
                64*np.finfo(float).eps*rate_scale) or
            not np.array_equal(
                value.wetdry_trial.shared_turbulence_face_flux[:, faces],
                value.auxiliary_final_trace_v91_face_flux_m4_s2) or
            any(name in runtime.hydraulic_runtime.__dict__ for name in (
                "_launch_face_density", "_launch_projection")) or
            "evaluate" in runtime.hydraulic_runtime.turbulence.__dict__)
    if (value.execution_fingerprint != runtime.execution.fingerprint or
            value.trace_receipt_sha256 != runtime.trace.receipt_sha256 or
            value.substep_index != runtime.substep_index or
            value.density_continuous_interface_face_count !=
                density_continuous_count or
            value.zero_interface_viscosity_face_count !=
                expected_zero_viscosity_count or
            value.interface_turbulence_flux_linf_m4_s2 != interface_linf or
            auxiliary_invalid or
            value.scalar_interface_owner_count != 0 or
            value.q3_interface_owner_count != 0 or
            value.state_or_clock_publication_count != 0 or
            value.backend != BACKEND or
            value.trial_sha256 != _hash(_trial_payload(value))):
        raise CoupledAMRFwaveGPUV98Error(
            "v98 CUDA external-trace trial differs")


def method_identity_coupled_amr_fwave_gpu_v98() -> dict[str, object]:
    return {
        "scope_id": SCOPE_ID,
        "backend": BACKEND,
        "spatial_solver": "resident_augmented_fwave_external_coarse_trace",
        "provisional_reflective_calls": "evaluated_then_replaced",
        "accepted_interface_calls_per_residual": INTERFACE_FACE_COUNT,
        "density_interface": "exactly_continuous_only",
        "interface_shared_viscosity": "exact_zero_only",
        "scalar_interface_owner_count": 0,
        "q3_interface_owner_count": 0,
        "actual_cuda_kernels": True,
        "automatic_cpu_fallback": False,
        "production_admission": False,
    }


__all__ = [
    "AMRInterfaceRateReceiptGPUV98",
    "AMRInterfaceResidualAuditGPUV98", "BACKEND",
    "CoupledAMRFwaveGPUV98Error", "ResidentAMRInterfaceFwaveGPUV98",
    "SCOPE_ID", "WetDryAMRInterfaceGPUTrialV98",
    "method_identity_coupled_amr_fwave_gpu_v98",
    "solve_wetdry_midpoint_with_external_trace_gpu_v98",
    "validate_amr_interface_residual_audit_gpu_v98",
    "validate_wetdry_amr_interface_trial_gpu_v98",
]
