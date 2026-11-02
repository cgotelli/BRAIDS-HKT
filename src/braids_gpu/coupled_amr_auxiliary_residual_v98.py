"""Resident v98 interface density/viscosity terms for the true residual.

This bridge augments the ordinary density FACE equation before the hydraulic
owner's single projection, and supplies only the missing internal-trace v91
momentum flux.  It owns no Riemann solve, cap, time, source replay or accepted
state.  The frozen v97 residual/default route is not modified here.
"""
from pathlib import Path

import numpy as np

from braids_core.sediment.density_active_projection_v97 import _digest, _hash, _sha
from braids_core.sediment.density_trace_projection_v98 import (
    _hex, resolve_density_trace_geometry_v98,
)
from .coupled_amr_geometry_v98 import validate_coupled_amr_geometry_binding_gpu_v98
from .coupled_amr_trace_v98 import validate_coupled_amr_internal_trace_structure_gpu_v98
from .coupled_wetdry_v97 import _grid
from .density_trace_faces_v98 import ResidentDensityTraceFacesV98
from .momentum_trace_v98 import ResidentMomentumTraceV98
from .wetdry_fwave_v97 import MODE_WATER_PLUS_SEDIMENT


SCOPE_ID = "braids.q4.coupled_amr.resident_auxiliary_residual.v98"


def _poll(cancelled):
    if not callable(cancelled) and type(cancelled) is not bool:
        raise ValueError("v98 auxiliary cancellation must be bool/callable")
    return cancelled if callable(cancelled) else lambda: cancelled


class ResidentAMRAuxiliaryResidualGPUV98:
    def __init__(self, hydraulic_runtime, coupled_execution, trace, *,
                 geometry_binding, substep_index,
                 donor_capped_vicuv_m2_s, donor_cap_receipt_sha256):
        validate_coupled_amr_geometry_binding_gpu_v98(geometry_binding)
        validate_coupled_amr_internal_trace_structure_gpu_v98(trace, coupled_execution)
        grid = _grid(coupled_execution.active_execution)
        joined = geometry_binding.joined_geometry
        if (hydraulic_runtime.execution is not coupled_execution.hydraulic_execution or
                type(substep_index) is not int or substep_index not in (0, 1) or
                geometry_binding.coarse_execution_fingerprint != trace.coarse_execution_fingerprint or
                geometry_binding.fine_execution_fingerprint != trace.fine_execution_fingerprint or
                geometry_binding.fine_grid.fingerprint != grid.fingerprint or
                joined.fine_faces != tuple(trace.fine_faces)):
            raise ValueError("v98 auxiliary trace/execution/geometry binding differs")
        donor = np.array(donor_capped_vicuv_m2_s, dtype=np.float64, copy=True, order="C")
        if donor.shape != (2,) or not np.isfinite(donor).all() or np.any(donor < 0.):
            raise ValueError("v98 auxiliary donor pre-capped viscosity differs")
        donor.setflags(write=False)
        _hex(donor_cap_receipt_sha256, "donor cap receipt")
        self.hydraulic_runtime, self.execution, self.trace = hydraulic_runtime, coupled_execution, trace
        self.geometry_binding, self.substep_index = geometry_binding, substep_index
        self.cp = cp = hydraulic_runtime.cp
        self.nc, self.nf = grid.cell_count, grid.face_count
        self.grid = grid
        self.device_id = int(cp.cuda.Device().id)
        self.donor_vicuv_host = donor
        self.faces_host = np.asarray(trace.fine_faces, dtype=np.int32)
        self.geometry = resolve_density_trace_geometry_v98(
            hydraulic_runtime.projection.geometry, grid.owner, grid.neighbour, grid.boundary_index)
        self.density_operator = ResidentDensityTraceFacesV98(
            self.geometry, backend="implicit_fwave_midpoint_gpu")
        self.momentum_operator = ResidentMomentumTraceV98(
            self.geometry, backend="implicit_fwave_midpoint_gpu")
        distance = np.array(grid.centre_distance_m, copy=True)
        distance[self.faces_host] = joined.normal_centre_distance_m
        lookup = np.full(self.nf, -1, dtype=np.int32)
        lookup[self.faces_host] = np.arange(2, dtype=np.int32)
        selected_trace_density = (trace.trace_mixture_density_kg_m3
            if hydraulic_runtime.execution.density_mode == MODE_WATER_PLUS_SEDIMENT
            else trace.trace_water_density_kg_m3)
        self.distance = cp.asarray(distance)
        self.lookup = cp.asarray(lookup)
        self.faces = cp.asarray(self.faces_host)
        self.area = cp.asarray(grid.cell_area_m2)
        self.trace_q = cp.asarray(np.ascontiguousarray(trace.trace_q[substep_index]))
        self.trace_depth = cp.ascontiguousarray(self.trace_q[0])
        self.trace_density = cp.asarray(selected_trace_density[substep_index])
        self.trace_wet = cp.asarray(trace.trace_wet_mask[substep_index], dtype=cp.int32)
        self.donor_vicuv = cp.asarray(donor)
        self.geometry_inputs_sha256 = _digest(distance, lookup, self.faces_host,
            grid.cell_area_m2, trace.trace_q[substep_index],
            selected_trace_density[substep_index], trace.trace_wet_mask[substep_index], donor)
        self.receipt = dict(scope_id=SCOPE_ID, backend="actual_cuda", actual_cuda_kernels=True,
            automatic_cpu_fallback=False, wrapper_sha256=_sha(Path(__file__)),
            fine_execution_sha256=coupled_execution.fingerprint,
            trace_receipt_sha256=trace.receipt_sha256,
            geometry_binding_sha256=geometry_binding.receipt_sha256,
            joined_geometry_sha256=joined.receipt_sha256,
            donor_cap_receipt_sha256=donor_cap_receipt_sha256,
            donor_capped_vicuv_sha256=_digest(donor),
            geometry_inputs_sha256=self.geometry_inputs_sha256,
            substep_index=substep_index, device_id=self.device_id,
            density_operator_sha256=self.density_operator.receipt["receipt_sha256"],
            momentum_operator_sha256=self.momentum_operator.receipt["receipt_sha256"],
            density_projection_evaluation_count=0, turbulence_cap_evaluation_count=0,
            ordinary_v91_face_mask_modification_count=0,
            hydraulic_riemann_evaluation_count=0, state_or_clock_publication_count=0,
            production_admission=False)
        self.receipt["receipt_sha256"] = _hash(self.receipt)
        self.reset_residual_sequence()

    def validate_runtime(self):
        if (self.receipt["receipt_sha256"] != _hash({k:v for k,v in self.receipt.items() if k != "receipt_sha256"}) or
                self.receipt["wrapper_sha256"] != _sha(Path(__file__)) or
                int(self.cp.cuda.Device().id) != self.device_id or
                self.receipt["donor_capped_vicuv_sha256"] != _digest(self.donor_vicuv_host)):
            raise ValueError("v98 auxiliary resident runtime/source identity differs")
        validate_coupled_amr_geometry_binding_gpu_v98(self.geometry_binding)
        validate_coupled_amr_internal_trace_structure_gpu_v98(self.trace, self.execution)
        self.density_operator._validate_runtime()
        self.momentum_operator._validate_runtime()
        # Boundary/trace controls are immutable numerical inputs.  Authenticate
        # once at sequence entry; never read back the fine iterate in a residual.
        actual = _digest(*(self.cp.asnumpy(a) for a in (self.distance, self.lookup, self.faces,
            self.area, self.trace_q, self.trace_density, self.trace_wet, self.donor_vicuv)))
        if actual != self.geometry_inputs_sha256:
            raise ValueError("v98 auxiliary resident trace/metric controls differ")

    def reset_residual_sequence(self):
        self.density_augmentation_launch_count = 0
        self.trace_momentum_face_launch_count = 0
        self.trace_momentum_gather_launch_count = 0
        self.last_density_result = None
        self.last_momentum_result = None
        self.last_density_input_images = None
        self.last_momentum_input_images = None

    def augment_density_resident(self, depth, selected_density,
            base_face_acceleration, base_density_mask, wet_cell_mask, *, cancelled=False):
        cancel = _poll(cancelled)
        self.last_density_result = None
        self.last_density_input_images = None
        if cancel():
            raise InterruptedError("v98 density trace cancelled before resident augmentation")
        result = self.density_operator.launch_resident((
            depth, selected_density, self.distance, base_face_acceleration,
            wet_cell_mask, base_density_mask, self.faces, self.trace_depth,
            self.trace_density, self.trace_wet), self.lookup,
            reference_density=self.hydraulic_runtime.execution.density_execution.hydraulic_reference_density_kg_m3,
            gravity=self.hydraulic_runtime.execution.density_execution.gravity_m_s2)
        self.density_augmentation_launch_count += 1
        if bool(self.cp.any(result["status"] != 0).item()):
            raise ValueError("v98 density trace resident kernel rejected controls")
        if cancel():
            raise InterruptedError("v98 density trace cancelled after resident augmentation")
        result["selected_face_acceleration"] = self.cp.ascontiguousarray(result["face_acceleration"][self.faces])
        result["selected_density_mask"] = self.cp.ascontiguousarray(result["density_mask"][self.faces])
        # The hydraulic owner restores its scratch when solve returns.  Keep
        # device-to-device copies of the LAST true-residual dynamic inputs,
        # rather than references that would later point at restored scratch.
        self.last_density_input_images = (depth.copy(), selected_density.copy(),
            self.distance, base_face_acceleration.copy(), wet_cell_mask.copy(),
            base_density_mask.copy(), self.faces, self.trace_depth, self.trace_density, self.trace_wet)
        self.last_density_result = result
        return result

    def evaluate_momentum_resident(self, q, capped_vicuv, wet_cell_mask, *, cancelled=False):
        cancel = _poll(cancelled)
        self.last_momentum_result = None
        self.last_momentum_input_images = None
        if cancel():
            raise InterruptedError("v98 momentum trace cancelled before resident launch")
        result = self.momentum_operator.launch_resident((q, capped_vicuv,
            self.area, self.distance, wet_cell_mask, self.faces, self.trace_q,
            self.donor_vicuv, self.trace_wet), self.lookup)
        self.trace_momentum_face_launch_count += 1
        self.trace_momentum_gather_launch_count += 1
        if bool(self.cp.any(result["face_status"] != 0).item()) or \
                bool(self.cp.any(result["cell_status"] != 0).item()):
            raise ValueError("v98 momentum trace resident kernel rejected controls")
        if cancel():
            raise InterruptedError("v98 momentum trace cancelled after resident launch")
        result["selected_face_flux"] = self.cp.ascontiguousarray(result["face_flux"][:, self.faces])
        self.last_momentum_input_images = (q.copy(), capped_vicuv.copy(), self.area,
            self.distance, wet_cell_mask.copy(), self.faces, self.trace_q, self.donor_vicuv, self.trace_wet)
        self.last_momentum_result = result
        return result

    def final_evidence_host(self):
        """Read only the retained final evidence, outside the residual loop.

        Input ordering matches validate_density_trace_inputs_v98 and
        validate_momentum_trace_inputs_v98.  The outer v98 owner authenticates
        its trace/cap/evaluation provenance and rebuilds the public receipts.
        This function neither re-runs an operator nor constructs an endpoint.
        """
        self.validate_runtime()
        if (self.last_density_input_images is None or self.last_momentum_input_images is None or
                self.last_density_result is None or self.last_momentum_result is None):
            raise ValueError("v98 auxiliary final residual evidence is incomplete")
        host = lambda value: np.ascontiguousarray(self.cp.asnumpy(value))
        return dict(density_inputs=tuple(host(a) for a in self.last_density_input_images),
            density_lookup=host(self.lookup),
            density_result={key:host(value) for key,value in self.last_density_result.items()},
            momentum_inputs=tuple(host(a) for a in self.last_momentum_input_images),
            momentum_lookup=host(self.lookup),
            momentum_result={key:host(value) for key,value in self.last_momentum_result.items()},
            runtime_receipt_sha256=self.receipt["receipt_sha256"], launch_counts=self.launch_counts(),
            density_evaluation_index=self.density_augmentation_launch_count,
            momentum_evaluation_index=self.trace_momentum_face_launch_count,
            numerical_operator_replay_count=0)

    def launch_counts(self):
        return dict(density_face_augmentation=self.density_augmentation_launch_count,
            trace_v91_momentum_faces=self.trace_momentum_face_launch_count,
            trace_v91_momentum_gathers=self.trace_momentum_gather_launch_count,
            density_projection=0, turbulence_cap=0, hydraulic_riemann=0,
            state_or_clock_publication=0)


__all__ = ["SCOPE_ID", "ResidentAMRAuxiliaryResidualGPUV98"]
