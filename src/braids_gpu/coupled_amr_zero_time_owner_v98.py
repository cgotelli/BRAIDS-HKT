"""Transactional zero-time refine/coarsen owner for one GPU joined leaf.

This additive v98 owner consumes an *accepted evolved* ``CoupledWetDryGPU``
state.  It area-restricts the four authenticated fine children to a virtual
parent, launches the reviewed resident CUDA conserved transfer exactly once,
and derives immutable refined/coarsened hierarchy images.  It advances no
physical clock and invokes no hydraulic, scalar, Q3, settling, exchange, or
constitutive operator.

The CUDA transfer is the numerical owner.  The Python transfer is an
independent validation oracle only.  Bed geometry, wet state, EOS, viscosity
and mixture density reuse the existing CUDA derived-image kernels with an
exactly zero register correction. DSS/HLES mapping, topology and immutable
checkpoint images retain explicit host work; there is no end-to-end-device-
resident claim.  The output is an
additive hierarchy image, not a forged v97 accepted state and not general
regridding admission.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Callable
from types import MappingProxyType

import numpy as np

from braids_core.sediment.coupled_amr_geometry_v98 import (
    CoupledAMRJoinedGeometryV98,
)
from braids_core.sediment.coupled_amr_zero_time_transfer_v98 import (
    CoupledAMRZeroTimeTransferV98,
    resolve_zero_time_refine_coarsen_v98,
    validate_zero_time_refine_coarsen_v98,
)
from braids_gpu.coupled_amr_corrected_coarse_v98 import (
    ResidentAMRCorrectedCoarseImagesV98,
)
from braids_gpu.coupled_amr_geometry_v98 import (
    CoupledAMRGeometryBindingGPUV98,
    validate_coupled_amr_geometry_binding_gpu_v98,
)
from braids_gpu.coupled_amr_register_v98 import (
    CoupledAMRInterfaceMapGPUV98,
    validate_coupled_amr_interface_map_gpu_v98,
)
from braids_gpu.coupled_amr_zero_time_transfer_v98 import (
    KEYS as TRANSFER_KEYS,
    ResidentAMRZeroTimeTransferGPUV98,
)
from braids_gpu.coupled_wetdry_v97 import (
    CoupledWetDryGPUExecutionV97,
    CoupledWetDryGPUStateV97,
    _grid,
    _v92_execution,
    _v92_state,
    execution_document_coupled_wetdry_gpu_v97,
    validate_coupled_wetdry_gpu_state_v97,
)


SCHEMA = 1
SCOPE_ID = "braids.q4.coupled_amr.zero_time_owner.actual_cuda.v98"
SEMANTIC_ID = "one_evolved_parent_four_children_refine_then_coarsen"
BACKEND = "actual_cuda_zero_time_hierarchy_image"
CHILD_COUNT = 4
SIDE_NAMES = ("west", "east", "south", "north")
_SHA = re.compile(r"[0-9a-f]{64}\Z")


class CoupledAMRZeroTimeOwnerGPUV98Error(ValueError):
    """The bounded GPU zero-time owner identity or algebra differs."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError, OverflowError) as error:
        raise CoupledAMRZeroTimeOwnerGPUV98Error(
            "v98 CUDA zero-time value is not finite canonical JSON") from error


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _digest(*values: object) -> str:
    result = hashlib.sha256()
    for raw in values:
        value = np.ascontiguousarray(raw)
        result.update(value.dtype.str.encode("ascii"))
        result.update(_canonical(list(value.shape)))
        result.update(value.tobytes(order="C"))
    return result.hexdigest()


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise CoupledAMRZeroTimeOwnerGPUV98Error(
            f"v98 CUDA zero-time {name} must be a lowercase SHA-256 digest")
    return value


def _readonly(value: object, shape: tuple[int, ...], name: str, *,
              nonnegative: bool = False, positive: bool = False,
              integer: bool = False) -> np.ndarray:
    raw = np.asarray(value)
    if (raw.shape != shape or raw.dtype.kind not in "fiub" or
            not np.isfinite(raw).all() or
            (nonnegative and np.any(raw < 0.0)) or
            (positive and np.any(raw <= 0.0)) or
            (integer and np.any(raw != np.floor(raw)))):
        raise CoupledAMRZeroTimeOwnerGPUV98Error(
            f"v98 CUDA zero-time {name} differs")
    result = np.array(
        raw, dtype=np.int64 if integer else np.float64,
        order="C", copy=True)
    result.setflags(write=False)
    return result


def _poll(value: bool | Callable[[], bool]) -> Callable[[], bool]:
    if isinstance(value, (bool, np.bool_)):
        return lambda: bool(value)
    if not callable(value):
        raise TypeError("cancelled must be Boolean or callable")

    def checked() -> bool:
        result = value()
        if not isinstance(result, (bool, np.bool_)):
            raise TypeError("cancelled callback must return Boolean data")
        return bool(result)
    return checked


def _tolerance(*values: object) -> float:
    scale = 1.0
    for value in values:
        array = np.asarray(value, dtype=np.float64)
        if array.size:
            scale = max(scale, float(np.max(np.abs(array))))
    return 32768.0*np.finfo(np.float64).eps*scale


def _weighted(value: np.ndarray, area: np.ndarray,
              total_area: float) -> np.ndarray:
    return np.sum(value*area, axis=-1, dtype=np.float64)/total_area


@dataclass(frozen=True)
class CoupledAMRRootBoundaryMemoryGPUV98:
    source_face_indices: np.ndarray
    source_face_global_ids: np.ndarray
    boundary_source_ids: np.ndarray
    sediment_last_outflow_concentration_kg_m3: np.ndarray
    sediment_return_clock_s: np.ndarray
    fluid_last_outflow: np.ndarray
    fluid_return_clock_s: np.ndarray
    mapping_source_face_global_ids: np.ndarray
    finest_segment_storage_count: int
    remap_evaluation_count: int
    physical_boundary_ledger_application_count: int
    receipt_sha256: str


def _boundary_payload(value: CoupledAMRRootBoundaryMemoryGPUV98) -> dict:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID+":root-boundary-memory",
        "arrays_sha256": _digest(
            value.source_face_indices, value.source_face_global_ids,
            value.boundary_source_ids,
            value.sediment_last_outflow_concentration_kg_m3,
            value.sediment_return_clock_s,
            value.fluid_last_outflow, value.fluid_return_clock_s,
            value.mapping_source_face_global_ids),
        "finest_segment_storage_count": value.finest_segment_storage_count,
        "remap_evaluation_count": value.remap_evaluation_count,
        "physical_boundary_ledger_application_count":
            value.physical_boundary_ledger_application_count,
        "mapping": "identity_on_stable_finest_physical_face_global_id",
    }


@dataclass(frozen=True)
class CoupledAMRZeroTimeHlesMapGPUV98:
    source_hles_state_sha256: str
    incident_fine_face_indices: np.ndarray
    incident_fine_face_global_ids: np.ndarray
    incident_face_role: np.ndarray
    incident_outer_side_index: np.ndarray
    incident_orientation_to_parent_outward: np.ndarray
    parent_side_filtered_outward_velocity_m_s: np.ndarray
    refined_face_filtered_normal_velocity_m_s: np.ndarray
    source_child_dynamic_vicuv_m2_s: np.ndarray
    source_child_dynamic_dicuv_m2_s: np.ndarray
    parent_dynamic_vicuv_m2_s: float
    parent_dynamic_dicuv_m2_s: float
    refined_child_dynamic_vicuv_m2_s: np.ndarray
    refined_child_dynamic_dicuv_m2_s: np.ndarray
    parent_outer_restriction_residual_m_s: np.ndarray
    outer_segment_mapping_count: int
    new_internal_face_initialization_count: int
    hles_physical_time_evaluation_count: int
    receipt_sha256: str


def _hles_payload(value: CoupledAMRZeroTimeHlesMapGPUV98) -> dict:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID+":hles-map",
        "source_hles": value.source_hles_state_sha256,
        "arrays_sha256": _digest(
            value.incident_fine_face_indices,
            value.incident_fine_face_global_ids,
            value.incident_face_role,
            value.incident_outer_side_index,
            value.incident_orientation_to_parent_outward,
            value.parent_side_filtered_outward_velocity_m_s,
            value.refined_face_filtered_normal_velocity_m_s,
            value.source_child_dynamic_vicuv_m2_s,
            value.source_child_dynamic_dicuv_m2_s,
            np.asarray(value.parent_dynamic_vicuv_m2_s),
            np.asarray(value.parent_dynamic_dicuv_m2_s),
            value.refined_child_dynamic_vicuv_m2_s,
            value.refined_child_dynamic_dicuv_m2_s,
            value.parent_outer_restriction_residual_m_s),
        "outer_segment_mapping_count": value.outer_segment_mapping_count,
        "new_internal_face_initialization_count":
            value.new_internal_face_initialization_count,
        "hles_physical_time_evaluation_count":
            value.hles_physical_time_evaluation_count,
        "mapping": (
            "outer_length_weighted_parent_then_piecewise_constant_refine;"
            "new_internal_faces_equal_transferred_endpoint_mean_velocity"),
    }


@dataclass(frozen=True)
class CoupledAMRZeroTimeActiveImageGPUV98:
    layout: str
    global_cell_ids: np.ndarray
    q: np.ndarray
    suspended_mass_kg_m2: np.ndarray
    bed_mass_kg_m2: np.ndarray
    h_temperature_c_m: np.ndarray
    h_salinity_ppt_m: np.ndarray
    temperature_cell_c: np.ndarray
    salinity_cell_ppt: np.ndarray
    water_density_cell_kg_m3: np.ndarray
    mixture_density_cell_kg_m3: np.ndarray
    molecular_viscosity_cell_m2_s: np.ndarray
    bed_elevation_m: np.ndarray
    free_surface_elevation_m: np.ndarray
    wet_mask: np.ndarray
    suspended_diameter_m: np.ndarray
    hles_face_keys: tuple[str, ...]
    hles_filtered_face_normal_velocity_m_s: np.ndarray
    hles_dynamic_vicuv_m2_s: np.ndarray
    hles_dynamic_dicuv_m2_s: np.ndarray
    accepted_updates: int
    time_s: float
    derived_cuda_runtime: ResidentAMRCorrectedCoarseImagesV98
    derived_cuda_evidence: MappingProxyType
    receipt_sha256: str


def _image_payload(value: CoupledAMRZeroTimeActiveImageGPUV98) -> dict:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID+":active-image",
        "layout": value.layout,
        "global_cell_ids": value.global_cell_ids.tolist(),
        "arrays_sha256": _digest(
            value.q, value.suspended_mass_kg_m2,
            value.bed_mass_kg_m2, value.h_temperature_c_m,
            value.h_salinity_ppt_m, value.temperature_cell_c,
            value.salinity_cell_ppt, value.water_density_cell_kg_m3,
            value.mixture_density_cell_kg_m3,
            value.molecular_viscosity_cell_m2_s,
            value.bed_elevation_m, value.free_surface_elevation_m,
            value.wet_mask, value.suspended_diameter_m,
            value.hles_filtered_face_normal_velocity_m_s,
            value.hles_dynamic_vicuv_m2_s,
            value.hles_dynamic_dicuv_m2_s),
        "hles_face_keys": list(value.hles_face_keys),
        "accepted_updates": value.accepted_updates,
        "time_s": value.time_s,
        "derived_field_evaluation_count": 1,
        "physical_time_operator_evaluation_count": 0,
        "derived_cuda_evidence": dict(value.derived_cuda_evidence),
        "backend": BACKEND,
    }


@dataclass(frozen=True)
class CoupledAMRZeroTimeCUDAEvidenceGPUV98:
    runtime_receipt_sha256: str
    runtime_source_sha256: str
    runtime_wrapper_sha256: str
    hierarchy_fingerprint: str
    restricted_parent_state_sha256: str
    root_boundary_memory_sha256: str
    input_sha256: str
    output_sha256: str
    parent_status: np.ndarray
    host_oracle_max_abs_error: np.ndarray
    numerical_kernel_launch_count: int
    validation_input_upload_count: int
    validation_output_readback_count: int
    numerical_python_parent_child_loop_count: int
    constitutive_evaluation_count: int
    physical_boundary_evaluation_count: int
    state_or_clock_publication_count: int
    automatic_cpu_fallback: bool
    backend: str
    receipt_sha256: str


def _cuda_payload(value: CoupledAMRZeroTimeCUDAEvidenceGPUV98) -> dict:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID+":cuda-evidence",
        "runtime": [value.runtime_receipt_sha256,
                    value.runtime_source_sha256,
                    value.runtime_wrapper_sha256],
        "hierarchy": value.hierarchy_fingerprint,
        "restricted_parent": value.restricted_parent_state_sha256,
        "boundary_memory": value.root_boundary_memory_sha256,
        "input_sha256": value.input_sha256,
        "output_sha256": value.output_sha256,
        "arrays_sha256": _digest(
            value.parent_status, value.host_oracle_max_abs_error),
        "counts": [
            value.numerical_kernel_launch_count,
            value.validation_input_upload_count,
            value.validation_output_readback_count,
            value.numerical_python_parent_child_loop_count,
            value.constitutive_evaluation_count,
            value.physical_boundary_evaluation_count,
            value.state_or_clock_publication_count,
        ],
        "automatic_cpu_fallback": value.automatic_cpu_fallback,
        "backend": value.backend,
        "host_oracle_tolerance": {"rtol": 3.0e-14, "atol": 3.0e-13},
    }


@dataclass(frozen=True)
class CoupledAMRZeroTimePlanGPUV98:
    coarse_execution_fingerprint: str
    fine_execution_fingerprint: str
    source_fine_state: CoupledWetDryGPUStateV97
    interface_fingerprint: str
    geometry_binding_receipt_sha256: str
    joined_geometry_receipt_sha256: str
    coarse_patch_receipt_sha256: str
    fine_patch_receipt_sha256: str
    covered_child_cells: np.ndarray
    covered_child_global_ids: np.ndarray
    covered_parent_global_id: int
    transfer_runtime: ResidentAMRZeroTimeTransferGPUV98
    cuda_evidence: CoupledAMRZeroTimeCUDAEvidenceGPUV98
    transfer_oracle: CoupledAMRZeroTimeTransferV98
    root_boundary_memory: CoupledAMRRootBoundaryMemoryGPUV98
    hles_map: CoupledAMRZeroTimeHlesMapGPUV98
    refined_image: CoupledAMRZeroTimeActiveImageGPUV98
    coarsened_image: CoupledAMRZeroTimeActiveImageGPUV98
    virtual_parent_restriction_owner_count: int
    accepted_covered_parent_row_count: int
    resident_transfer_owner_count: int
    host_oracle_evaluation_count: int
    host_oracle_application_count: int
    dss_transfer_count: int
    dss_constitutive_evaluation_count: int
    eos_derivation_count: int
    derived_host_image_count: int
    next_macro_solver_state_construction_count: int
    plan_sha256: str


def _plan_payload(value: CoupledAMRZeroTimePlanGPUV98) -> dict:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID+":plan",
        "semantic_id": SEMANTIC_ID,
        "executions": [value.coarse_execution_fingerprint,
                       value.fine_execution_fingerprint],
        "source_fine_state": value.source_fine_state.state_sha256,
        "interface": value.interface_fingerprint,
        "geometry": [value.geometry_binding_receipt_sha256,
                     value.joined_geometry_receipt_sha256,
                     value.coarse_patch_receipt_sha256,
                     value.fine_patch_receipt_sha256],
        "covered_children_sha256": _digest(
            value.covered_child_cells, value.covered_child_global_ids),
        "covered_parent_global_id": value.covered_parent_global_id,
        "transfer": {
            "authoritative_actual_cuda": value.cuda_evidence.receipt_sha256,
            "validation_only_host_oracle":
                value.transfer_oracle.receipt_sha256,
        },
        "boundary_memory": value.root_boundary_memory.receipt_sha256,
        "hles_map": value.hles_map.receipt_sha256,
        "images": [value.refined_image.receipt_sha256,
                   value.coarsened_image.receipt_sha256],
        "owner_counts": {
            "virtual_parent_restriction":
                value.virtual_parent_restriction_owner_count,
            "accepted_covered_parent_row":
                value.accepted_covered_parent_row_count,
            "resident_transfer": value.resident_transfer_owner_count,
            "host_oracle_evaluation": value.host_oracle_evaluation_count,
            "host_oracle_application": value.host_oracle_application_count,
            "dss_transfer": value.dss_transfer_count,
            "dss_constitutive_evaluation":
                value.dss_constitutive_evaluation_count,
            "eos_derivation": value.eos_derivation_count,
            "derived_host_image": value.derived_host_image_count,
            "next_macro_solver_state_construction":
                value.next_macro_solver_state_construction_count,
        },
        "scope_limit": "one_evolved_joined_leaf_not_general_regrid",
        "end_to_end_device_resident": False,
        "backend": BACKEND,
    }


@dataclass(frozen=True)
class CoupledAMRZeroTimeHierarchyStateGPUV98:
    plan: CoupledAMRZeroTimePlanGPUV98
    active_layout: str
    active_image: CoupledAMRZeroTimeActiveImageGPUV98
    accepted_updates: int
    time_s: float
    refine_event_count: int
    coarsen_event_count: int
    accepted_update_increment_count: int
    physical_time_increment_count: int
    hydraulic_operator_evaluation_count: int
    scalar_operator_evaluation_count: int
    q3_operator_evaluation_count: int
    constitutive_evaluation_count: int
    active_parent_owner_count: int
    active_child_owner_count: int
    state_sha256: str


def _state_payload(value: CoupledAMRZeroTimeHierarchyStateGPUV98) -> dict:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID+":state",
        "plan": value.plan.plan_sha256,
        "active_layout": value.active_layout,
        "active_image": value.active_image.receipt_sha256,
        "accepted_updates": value.accepted_updates,
        "time_s": value.time_s,
        "event_counts": [value.refine_event_count,
                         value.coarsen_event_count],
        "clock_counts": [value.accepted_update_increment_count,
                         value.physical_time_increment_count],
        "operator_counts": [value.hydraulic_operator_evaluation_count,
                            value.scalar_operator_evaluation_count,
                            value.q3_operator_evaluation_count,
                            value.constitutive_evaluation_count],
        "active_owner_counts": [value.active_parent_owner_count,
                                value.active_child_owner_count],
        "boundary_memory": value.plan.root_boundary_memory.receipt_sha256,
        "next_macro_solver_state_construction_count":
            value.plan.next_macro_solver_state_construction_count,
    }


@dataclass(frozen=True)
class CoupledAMRZeroTimeEventTrialGPUV98:
    event_kind: str
    parent_state_sha256: str
    plan: CoupledAMRZeroTimePlanGPUV98
    candidate: CoupledAMRZeroTimeHierarchyStateGPUV98
    transfer_owner_count: int
    boundary_memory_remap_owner_count: int
    hles_mapping_owner_count: int
    candidate_publication_count: int
    trial_sha256: str


def _trial_payload(value: CoupledAMRZeroTimeEventTrialGPUV98) -> dict:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID+":event-trial",
        "event_kind": value.event_kind,
        "parent": value.parent_state_sha256,
        "plan": value.plan.plan_sha256,
        "candidate": value.candidate.state_sha256,
        "owner_counts": [value.transfer_owner_count,
                         value.boundary_memory_remap_owner_count,
                         value.hles_mapping_owner_count],
        "candidate_publication_count": value.candidate_publication_count,
    }


def _build_boundary_memory(source, fine_execution, fine_patch):
    grid = _grid(fine_execution.active_execution)
    v92 = _v92_state(source.active_state)
    lifecycle = v92.suspended_state.lifecycle
    physical = np.flatnonzero(np.asarray(grid.boundary_index) >= 0)
    if physical.size == 0:
        raise CoupledAMRZeroTimeOwnerGPUV98Error(
            "v98 CUDA zero-time fixture has no physical root segments")
    nonphysical = np.flatnonzero(np.asarray(grid.boundary_index) < 0)
    memory_arrays = (
        lifecycle.boundary_last_outflow_concentration_kg_m3,
        lifecycle.boundary_return_clock_s,
        v92.boundary_last_outflow_fluid,
        v92.boundary_return_clock_s,
    )
    if any(np.any(np.asarray(array)[:, nonphysical] != 0.0)
           for array in memory_arrays):
        raise CoupledAMRZeroTimeOwnerGPUV98Error(
            "v98 CUDA nonphysical faces carry boundary return memory")
    count = int(physical.size)
    indices = _readonly(
        physical, (count,), "physical root face indices",
        nonnegative=True, integer=True)
    global_ids = _readonly(
        fine_patch.face_global_ids[physical], (count,),
        "physical root face global IDs", nonnegative=True, integer=True)
    source_ids = _readonly(
        np.asarray(grid.boundary_index)[physical], (count,),
        "physical root boundary source IDs", nonnegative=True, integer=True)
    ng = fine_execution.fraction_count
    placeholder = CoupledAMRRootBoundaryMemoryGPUV98(
        indices, global_ids, source_ids,
        _readonly(
            lifecycle.boundary_last_outflow_concentration_kg_m3[:, physical],
            (ng, count), "sediment boundary memory", nonnegative=True),
        _readonly(
            lifecycle.boundary_return_clock_s[:, physical],
            (ng, count), "sediment boundary clocks", nonnegative=True),
        _readonly(
            v92.boundary_last_outflow_fluid[:, physical],
            (2, count), "fluid boundary memory"),
        _readonly(
            v92.boundary_return_clock_s[:, physical],
            (2, count), "fluid boundary clocks", nonnegative=True),
        global_ids, count, 1, 0, "")
    return CoupledAMRRootBoundaryMemoryGPUV98(
        **{**placeholder.__dict__,
           "receipt_sha256": _hash(_boundary_payload(placeholder))})


def _side_index(endpoints: np.ndarray, bounds: np.ndarray) -> int:
    tolerance = 512.0*np.finfo(np.float64).eps*max(
        1.0, float(np.max(np.abs(bounds))))
    if np.all(np.abs(endpoints[:, 0]-bounds[0]) <= tolerance):
        return 0
    if np.all(np.abs(endpoints[:, 0]-bounds[2]) <= tolerance):
        return 1
    if np.all(np.abs(endpoints[:, 1]-bounds[1]) <= tolerance):
        return 2
    if np.all(np.abs(endpoints[:, 1]-bounds[3]) <= tolerance):
        return 3
    raise CoupledAMRZeroTimeOwnerGPUV98Error(
        "v98 CUDA incident outer face is not on the parent boundary")


def _build_hles_map(source, fine_execution, fine_patch, joined,
                    transferred_q):
    grid = _grid(fine_execution.active_execution)
    cells = np.asarray(joined.covered_child_cells, dtype=np.int64)
    selected = np.zeros(fine_execution.cell_count, dtype=bool)
    selected[cells] = True
    owner = np.asarray(grid.owner, dtype=np.int64)
    neighbour = np.asarray(grid.neighbour, dtype=np.int64)
    owner_selected = selected[owner]
    neighbour_selected = np.zeros(fine_execution.face_count, dtype=bool)
    valid_neighbour = neighbour >= 0
    neighbour_selected[valid_neighbour] = selected[neighbour[valid_neighbour]]
    incident = np.flatnonzero(owner_selected | neighbour_selected)
    role = np.zeros(incident.size, dtype=np.int64)
    side = np.full(incident.size, -1, dtype=np.int64)
    orientation = np.zeros(incident.size, dtype=np.float64)
    parent_filtered = np.zeros(4, dtype=np.float64)
    side_length = np.zeros(4, dtype=np.float64)
    source_filtered = np.asarray(
        source.hles_state.accepted_filtered_face_normal_velocity_m_s)
    length = np.asarray(grid.face_length_m)
    for local, face in enumerate(incident):
        o_selected = owner_selected[face]
        n_selected = neighbour_selected[face]
        if o_selected and n_selected:
            role[local] = 2
            continue
        role[local] = 1
        side[local] = _side_index(
            np.asarray(fine_patch.face_endpoints_m[face]),
            np.asarray(joined.covered_parent_bounds_m))
        orientation[local] = 1.0 if o_selected else -1.0
        parent_filtered[side[local]] += (
            orientation[local]*source_filtered[face]*length[face])
        side_length[side[local]] += length[face]
    if np.any(role == 0) or np.any(side_length <= 0.0):
        raise CoupledAMRZeroTimeOwnerGPUV98Error(
            "v98 CUDA HLES incident-face classification is incomplete")
    bounds = np.asarray(joined.covered_parent_bounds_m)
    expected_length = np.array((
        bounds[3]-bounds[1], bounds[3]-bounds[1],
        bounds[2]-bounds[0], bounds[2]-bounds[0]))
    if np.any(np.abs(side_length-expected_length) >
              _tolerance(side_length, expected_length)):
        raise CoupledAMRZeroTimeOwnerGPUV98Error(
            "v98 CUDA HLES outer segments do not partition parent sides")
    parent_filtered /= side_length

    refined_filtered = np.zeros(incident.size, dtype=np.float64)
    cell_to_local = {int(cell): index for index, cell in enumerate(cells)}
    for local, face in enumerate(incident):
        if role[local] == 1:
            refined_filtered[local] = (
                orientation[local]*parent_filtered[side[local]])
            continue
        o = cell_to_local[int(owner[face])]
        n = cell_to_local[int(neighbour[face])]
        velocities = []
        for child in (o, n):
            depth = transferred_q[0, child]
            velocities.append(
                transferred_q[1:, child]/depth
                if depth > grid.dry_tolerance_m else np.zeros(2))
        refined_filtered[local] = 0.5*float(
            np.dot(velocities[0]+velocities[1], grid.normal[:, face]))

    residual = np.zeros(4, dtype=np.float64)
    for side_index in range(4):
        chosen = np.flatnonzero((role == 1) & (side == side_index))
        restricted = np.sum(
            orientation[chosen]*refined_filtered[chosen]*
            length[incident[chosen]], dtype=np.float64)/side_length[side_index]
        residual[side_index] = restricted-parent_filtered[side_index]
    if np.any(np.abs(residual) > _tolerance(parent_filtered, residual)):
        raise CoupledAMRZeroTimeOwnerGPUV98Error(
            "v98 CUDA HLES refine/coarsen outer memory does not close")

    area = np.asarray(grid.cell_area_m2)[cells]
    total_area = float(np.sum(area, dtype=np.float64))
    source_vic = np.asarray(
        source.hles_state.accepted_dynamic_vicuv_m2_s)[cells]
    source_dic = np.asarray(
        source.hles_state.accepted_dynamic_dicuv_m2_s)[cells]
    parent_vic = float(np.sum(source_vic*area, dtype=np.float64)/total_area)
    parent_dic = float(np.sum(source_dic*area, dtype=np.float64)/total_area)
    refined_vic = np.full(CHILD_COUNT, parent_vic)
    refined_dic = np.full(CHILD_COUNT, parent_dic)
    placeholder = CoupledAMRZeroTimeHlesMapGPUV98(
        source.hles_state.state_sha256,
        _readonly(incident, (incident.size,), "incident fine faces",
                  nonnegative=True, integer=True),
        _readonly(fine_patch.face_global_ids[incident], (incident.size,),
                  "incident global face IDs", nonnegative=True,
                  integer=True),
        _readonly(role, (incident.size,), "incident face roles",
                  nonnegative=True, integer=True),
        _readonly(side, (incident.size,), "incident side indices",
                  integer=True),
        _readonly(orientation, (incident.size,), "incident orientations"),
        _readonly(parent_filtered, (4,), "parent HLES filter"),
        _readonly(refined_filtered, (incident.size,), "refined HLES filter"),
        _readonly(source_vic, (4,), "source child HLES vicuv",
                  nonnegative=True),
        _readonly(source_dic, (4,), "source child HLES dicuv",
                  nonnegative=True),
        parent_vic, parent_dic,
        _readonly(refined_vic, (4,), "refined child HLES vicuv",
                  nonnegative=True),
        _readonly(refined_dic, (4,), "refined child HLES dicuv",
                  nonnegative=True),
        _readonly(residual, (4,), "parent HLES restriction residual"),
        int(np.count_nonzero(role == 1)),
        int(np.count_nonzero(role == 2)), 0, "")
    return CoupledAMRZeroTimeHlesMapGPUV98(
        **{**placeholder.__dict__,
           "receipt_sha256": _hash(_hles_payload(placeholder))})


def _derive_image(execution, *, layout, global_cell_ids, q, suspended, bed,
                  fluid, dss, initial_bed, reference_bed, hles_face_keys,
                  hles_filter, hles_vic, hles_dic, accepted_updates, time_s):
    ng = execution.fraction_count
    nc = int(np.asarray(global_cell_ids).size)
    qv = _readonly(q, (3, nc), "active conserved hydraulic image")
    suspended_v = _readonly(
        suspended, (ng, nc), "active suspension image", nonnegative=True)
    bed_v = _readonly(
        bed, (ng, nc), "active bed image", nonnegative=True)
    fluid_v = _readonly(fluid, (2, nc), "active fluid image")
    dss_v = _readonly(dss, (ng, nc), "active DSS image", positive=True)
    z0 = _readonly(initial_bed, (nc,), "active initial bed")
    m0 = _readonly(
        reference_bed, (ng, nc), "active reference bed", nonnegative=True)
    if np.any(qv[0] < 0.0) or np.any(fluid_v[1] < 0.0):
        raise CoupledAMRZeroTimeOwnerGPUV98Error(
            "v98 CUDA active depth or salinity inventory is negative")
    grid = _grid(execution.active_execution)
    wet = qv[0] > grid.dry_tolerance_m
    if (np.any(qv[1:, ~wet] != 0.0) or
            np.any(suspended_v[:, ~wet] != 0.0) or
            np.any(fluid_v[:, ~wet] != 0.0)):
        raise CoupledAMRZeroTimeOwnerGPUV98Error(
            "v98 CUDA active dry cells retain transported inventory")
    v92_execution = _v92_execution(execution.active_execution)
    runtime = ResidentAMRCorrectedCoarseImagesV98(
        backend="implicit_fwave_midpoint_gpu",
        initial_bed_elevation_m=z0, reference_bed_mass_kg_m2=m0,
        dry_bulk_density_kg_m3=execution.dry_bulk_density_kg_m3,
        solid_density_kg_m3=grid.profile.sediment_density_kg_m3)
    cp = runtime.cp
    controls = dict(cell=0, dry_tolerance_m=grid.dry_tolerance_m,
        temperature_mode=v92_execution.temperature_mode,
        salinity_mode=v92_execution.salinity_mode,
        equilibrium_temperature_c=v92_execution.equilibrium_temperature_c,
        equilibrium_salinity_ppt=v92_execution.equilibrium_salinity_ppt,
        density_eos=v92_execution.density_eos,
        reference_water_density_kg_m3=
            grid.profile.water_density_kg_m3)
    # Launch the existing numerical kernels directly: its validation evaluate()
    # names an applied reflux and is deliberately not used for this zero-time
    # derivation. Conserved fields and all cleanup vectors must stay unchanged.
    with cp.cuda.Device(runtime.device_id):
        runtime._validate_runtime()
        resident = runtime.launch_resident(
            *(cp.asarray(value) for value in (qv, suspended_v, bed_v, fluid_v)),
            cp.zeros(3, dtype=cp.float64), cp.zeros(ng, dtype=cp.float64),
            cp.zeros(ng, dtype=cp.float64), cp.zeros(2, dtype=cp.float64),
            **controls)
        if any(np.any(cp.asnumpy(resident[key])) for key in
               ("image_status", "eos_status", "mixture_status")):
            raise CoupledAMRZeroTimeOwnerGPUV98Error(
                "v98 zero-time CUDA derived image is outside its EOS domain")
        for key, original in (("q", qv), ("suspended", suspended_v),
                              ("bed", bed_v), ("fluid", fluid_v)):
            if not np.array_equal(cp.asnumpy(resident[key]), original):
                raise CoupledAMRZeroTimeOwnerGPUV98Error(
                    "v98 zero-time derivation changed a conserved field")
        if any(np.any(cp.asnumpy(resident[key]) != 0.) for key in
               ("dry_transfer", "momentum_cleanup", "fluid_cleanup")):
            raise CoupledAMRZeroTimeOwnerGPUV98Error(
                "v98 zero-time derivation reapplied dry cleanup")
        derived = [cp.asnumpy(resident[key]) for key in (
            "bed_elevation", "eta", "temperature", "salinity", "water_density",
            "mixture_density", "viscosity", "wet_mask")]
    derived[-1] = derived[-1].astype(np.int64)
    frozen = []
    for array in derived:
        item = np.ascontiguousarray(array)
        item.setflags(write=False)
        frozen.append(item)
    metadata = dict(schema=1, backend="actual_cuda_zero_time_derived_fields",
        runtime_receipt_sha256=runtime.receipt["receipt_sha256"],
        controls_sha256=_hash(controls),
        conserved_input_sha256=_digest(qv, suspended_v, bed_v, fluid_v),
        derived_output_sha256=_digest(*frozen),
        numerical_kernel_launch_count=3, derived_field_owner_count=1,
        reflux_application_count=0, dry_cleanup_application_count=0,
        settling_exchange_q3_replay_count=0, dss_recompute_count=0,
        hles_physical_evaluation_count=0, state_or_clock_publication_count=0,
        automatic_cpu_fallback=False)
    evidence = MappingProxyType({**metadata, "receipt_sha256": _hash(metadata)})
    if (type(accepted_updates) is not int or not 0 <= accepted_updates <= 4 or
            not isinstance(time_s, (int, float)) or not math.isfinite(time_s)
            or time_s < 0.0 or len(set(hles_face_keys)) != len(hles_face_keys)):
        raise CoupledAMRZeroTimeOwnerGPUV98Error(
            "v98 CUDA active image clocks or HLES keys differ")
    placeholder = CoupledAMRZeroTimeActiveImageGPUV98(
        layout,
        _readonly(global_cell_ids, (nc,), "active global cell IDs",
                  nonnegative=True, integer=True),
        qv, suspended_v, bed_v, fluid_v[0], fluid_v[1],
        frozen[2], frozen[3], frozen[4], frozen[5], frozen[6],
        frozen[0], frozen[1], frozen[7], dss_v,
        tuple(hles_face_keys),
        _readonly(hles_filter, (len(hles_face_keys),), "active HLES filter"),
        _readonly(hles_vic, (nc,), "active HLES vicuv", nonnegative=True),
        _readonly(hles_dic, (nc,), "active HLES dicuv", nonnegative=True),
        accepted_updates, float(time_s), runtime, evidence, "")
    return CoupledAMRZeroTimeActiveImageGPUV98(
        **{**placeholder.__dict__,
           "receipt_sha256": _hash(_image_payload(placeholder))})


def _validate_image(value, execution):
    if (type(value) is not CoupledAMRZeroTimeActiveImageGPUV98 or
            value.receipt_sha256 != _hash(_image_payload(value)) or
            value.accepted_updates < 0 or value.accepted_updates > 4 or
            value.time_s < 0.0):
        raise CoupledAMRZeroTimeOwnerGPUV98Error(
            "v98 CUDA active image identity differs")
    if (type(value.derived_cuda_runtime) is not
            ResidentAMRCorrectedCoarseImagesV98 or
            type(value.derived_cuda_evidence) is not MappingProxyType):
        raise CoupledAMRZeroTimeOwnerGPUV98Error(
            "v98 zero-time derived CUDA runtime/evidence differs")
    value.derived_cuda_runtime._validate_runtime()
    evidence = value.derived_cuda_evidence
    if (evidence["receipt_sha256"] != _hash({k: v for k, v in
            evidence.items() if k != "receipt_sha256"}) or
            evidence["runtime_receipt_sha256"] !=
                value.derived_cuda_runtime.receipt["receipt_sha256"] or
            evidence["conserved_input_sha256"] != _digest(
                value.q, value.suspended_mass_kg_m2, value.bed_mass_kg_m2,
                np.vstack((value.h_temperature_c_m, value.h_salinity_ppt_m))) or
            evidence["derived_output_sha256"] != _digest(
                value.bed_elevation_m, value.free_surface_elevation_m,
                value.temperature_cell_c, value.salinity_cell_ppt,
                value.water_density_cell_kg_m3, value.mixture_density_cell_kg_m3,
                value.molecular_viscosity_cell_m2_s, value.wet_mask) or
            evidence["numerical_kernel_launch_count"] != 3 or
            evidence["derived_field_owner_count"] != 1 or
            any(evidence[key] != 0 for key in (
                "reflux_application_count", "dry_cleanup_application_count",
                "settling_exchange_q3_replay_count", "dss_recompute_count",
                "hles_physical_evaluation_count", "state_or_clock_publication_count")) or
            evidence["automatic_cpu_fallback"] is not False):
        raise CoupledAMRZeroTimeOwnerGPUV98Error(
            "v98 zero-time derived CUDA receipt or owner counts differ")
    arrays = (
        value.global_cell_ids, value.q, value.suspended_mass_kg_m2,
        value.bed_mass_kg_m2, value.h_temperature_c_m,
        value.h_salinity_ppt_m, value.temperature_cell_c,
        value.salinity_cell_ppt, value.water_density_cell_kg_m3,
        value.mixture_density_cell_kg_m3,
        value.molecular_viscosity_cell_m2_s, value.bed_elevation_m,
        value.free_surface_elevation_m, value.wet_mask,
        value.suspended_diameter_m,
        value.hles_filtered_face_normal_velocity_m_s,
        value.hles_dynamic_vicuv_m2_s, value.hles_dynamic_dicuv_m2_s)
    if any(array.flags.writeable for array in arrays):
        raise CoupledAMRZeroTimeOwnerGPUV98Error(
            "v98 CUDA active image arrays are mutable")
    dry = _grid(execution.active_execution).dry_tolerance_m
    if (not np.array_equal(
            value.free_surface_elevation_m,
            value.q[0]+value.bed_elevation_m) or
            not np.array_equal(value.wet_mask,
                               (value.q[0] > dry).astype(np.int64))):
        raise CoupledAMRZeroTimeOwnerGPUV98Error(
            "v98 CUDA active image stage or wet mask differs")


def _validate_cuda(value, runtime):
    runtime.validate_runtime()
    if (type(value) is not CoupledAMRZeroTimeCUDAEvidenceGPUV98 or
            value.receipt_sha256 != _hash(_cuda_payload(value)) or
            value.runtime_receipt_sha256 !=
                runtime.receipt["receipt_sha256"] or
            value.runtime_source_sha256 != runtime.receipt["source_sha256"] or
            value.runtime_wrapper_sha256 !=
                runtime.receipt["wrapper_sha256"] or
            value.parent_status.shape != (1,) or
            value.parent_status.dtype != np.int32 or
            np.any(value.parent_status != 0) or
            value.parent_status.flags.writeable or
            value.host_oracle_max_abs_error.shape != (10,) or
            value.host_oracle_max_abs_error.flags.writeable or
            np.any(value.host_oracle_max_abs_error < 0.0) or
            value.numerical_kernel_launch_count != 1 or
            value.validation_input_upload_count != 4 or
            value.validation_output_readback_count != len(TRANSFER_KEYS)+1 or
            value.numerical_python_parent_child_loop_count != 0 or
            value.constitutive_evaluation_count != 0 or
            value.physical_boundary_evaluation_count != 0 or
            value.state_or_clock_publication_count != 0 or
            value.automatic_cpu_fallback or value.backend != BACKEND):
        raise CoupledAMRZeroTimeOwnerGPUV98Error(
            "v98 CUDA transfer evidence identity or counts differ")


def _make_plan(coarse_execution, fine_execution, source, fine_provider,
               interface, geometry_binding, transfer_runtime, *, cancelled):
    poll = _poll(cancelled)
    if type(transfer_runtime) is not ResidentAMRZeroTimeTransferGPUV98:
        raise CoupledAMRZeroTimeOwnerGPUV98Error(
            "v98 CUDA transfer runtime type differs")
    execution_document_coupled_wetdry_gpu_v97(coarse_execution)
    execution_document_coupled_wetdry_gpu_v97(fine_execution)
    validate_coupled_wetdry_gpu_state_v97(
        source, fine_execution, fine_provider)
    validate_coupled_amr_geometry_binding_gpu_v98(geometry_binding)
    validate_coupled_amr_interface_map_gpu_v98(
        interface, coarse_execution, fine_execution, geometry_binding)
    if (geometry_binding.coarse_execution_fingerprint !=
            coarse_execution.fingerprint or
            geometry_binding.fine_execution_fingerprint !=
            fine_execution.fingerprint or
            interface.joined_interface_fingerprint !=
            geometry_binding.joined_geometry.interface_fingerprint):
        raise CoupledAMRZeroTimeOwnerGPUV98Error(
            "v98 CUDA zero-time geometry/execution binding differs")
    joined = geometry_binding.joined_geometry
    fine_patch = geometry_binding.fine_patch
    cells = np.asarray(joined.covered_child_cells, dtype=np.int64)
    if cells.shape != (4,) or len(set(map(int, cells))) != 4:
        raise CoupledAMRZeroTimeOwnerGPUV98Error(
            "v98 CUDA covered-child selection differs")
    fine_grid = _grid(fine_execution.active_execution)
    area = np.asarray(fine_grid.cell_area_m2)[cells]
    total_area = float(joined.covered_parent_area_m2)
    if abs(float(np.sum(area))-total_area) > _tolerance(area, total_area):
        raise CoupledAMRZeroTimeOwnerGPUV98Error(
            "v98 CUDA child areas do not cover virtual parent")
    v92 = _v92_state(source.active_state)
    q_source = np.asarray(v92.q)[:, cells]
    suspended_source = np.asarray(
        v92.suspended_state.suspended_mass_kg_m2)[:, cells]
    bed_source = np.asarray(
        source.active_state.base_state.bed_mass_kg_m2)[:, cells]
    fluid_source = np.vstack((
        np.asarray(v92.h_temperature_c_m)[cells],
        np.asarray(v92.h_salinity_ppt_m)[cells]))
    dss_source = np.asarray(
        source.active_state.suspended_diameter_m)[:, cells]
    parent_q = _weighted(q_source, area, total_area)
    parent_suspended = _weighted(suspended_source, area, total_area)
    parent_bed = _weighted(bed_source, area, total_area)
    parent_fluid = _weighted(fluid_source, area, total_area)
    parent_dss = _weighted(dss_source, area, total_area)
    initial_child = np.asarray(fine_execution.initial_bed_elevation_m)[cells]
    reference_child = np.asarray(
        fine_execution.reference_bed_mass_kg_m2)[:, cells]
    initial_parent = np.asarray((
        float(np.sum(initial_child*area, dtype=np.float64)/total_area),))
    reference_parent = _weighted(reference_child, area, total_area)[:, None]
    seed_child_bed = np.ascontiguousarray(
        initial_child+np.sum(
            (parent_bed[:, None]-reference_child)/
            fine_execution.dry_bulk_density_kg_m3[:, None], axis=0))
    boundary = _build_boundary_memory(source, fine_execution, fine_patch)
    source_ids = tuple(sorted(set(map(int, boundary.boundary_source_ids))))
    restriction_sha = _hash({
        "source": source.state_sha256,
        "children": cells.tolist(),
        "conserved": _digest(
            parent_q, parent_suspended, parent_bed, parent_fluid),
    })
    oracle = resolve_zero_time_refine_coarsen_v98(
        hierarchy_fingerprint=joined.receipt_sha256,
        parent_state_sha256=restriction_sha,
        parent_area_m2=total_area,
        child_area_m2=area,
        child_bed_elevation_m=seed_child_bed,
        dry_tolerance_m=fine_grid.dry_tolerance_m,
        parent_depth_m=float(parent_q[0]),
        parent_momentum_m2_s=parent_q[1:],
        parent_suspended_kg_m2=parent_suspended,
        parent_bed_kg_m2=parent_bed,
        parent_fluid_integral=parent_fluid,
        boundary_source_ids_before=source_ids,
        boundary_source_ids_after=source_ids,
        boundary_memory_before_sha256=boundary.receipt_sha256,
        boundary_memory_after_sha256=boundary.receipt_sha256,
        accepted_updates=source.accepted_updates,
        time_s=source.time_s)
    transfer_runtime.validate_runtime()
    if (transfer_runtime.np != 1 or
            transfer_runtime.ng != fine_execution.fraction_count or
            transfer_runtime.dry != fine_grid.dry_tolerance_m or
            not np.array_equal(
                transfer_runtime.geometry_host[0],
                np.asarray((total_area,), dtype=np.float64)) or
            not np.array_equal(
                transfer_runtime.geometry_host[1], area[None, :]) or
            not np.array_equal(
                transfer_runtime.geometry_host[2], seed_child_bed[None, :])):
        raise CoupledAMRZeroTimeOwnerGPUV98Error(
            "v98 CUDA transfer runtime geometry differs")
    if poll():
        raise InterruptedError("v98 CUDA zero-time refine cancelled before launch")
    try:
        observed = transfer_runtime.evaluate(
            np.ascontiguousarray(parent_q[:, None]),
            np.ascontiguousarray(parent_suspended[:, None]),
            np.ascontiguousarray(parent_bed[:, None]),
            np.ascontiguousarray(parent_fluid[:, None]),
            hierarchy_fingerprint=joined.receipt_sha256,
            parent_state_sha256=restriction_sha,
            boundary_memory_sha256=boundary.receipt_sha256,
            cancelled=poll)
    except RuntimeError as error:
        transfer_runtime.last_resident_result = None
        if "cancelled" in str(error).lower():
            raise InterruptedError(str(error)) from error
        raise
    if poll():
        transfer_runtime.last_resident_result = None
        raise InterruptedError("v98 CUDA zero-time refine cancelled after readback")
    arrays = [observed[name] for name in TRANSFER_KEYS]
    oracle_pairs = (
        (observed["child_q"][0, 0], oracle.child_depth_m),
        (observed["child_q"][1:, 0], oracle.child_momentum_m2_s),
        (observed["child_suspended"][:, 0], oracle.child_suspended_kg_m2),
        (observed["child_bed"][:, 0], oracle.child_bed_kg_m2),
        (observed["child_fluid"][:, 0], oracle.child_fluid_integral),
        (observed["restricted_q"][0, 0], oracle.restricted_depth_m),
        (observed["restricted_q"][1:, 0], oracle.restricted_momentum_m2_s),
        (observed["restricted_suspended"][:, 0],
         oracle.restricted_suspended_kg_m2),
        (observed["restricted_bed"][:, 0], oracle.restricted_bed_kg_m2),
        (observed["restricted_fluid"][:, 0], oracle.restricted_fluid_integral),
    )
    errors = np.asarray([
        float(np.max(np.abs(authoritative-reference), initial=0.0))
        for authoritative, reference in oracle_pairs], dtype=np.float64)
    for authoritative, reference in oracle_pairs:
        if not np.allclose(
                authoritative, reference, rtol=3.0e-14, atol=3.0e-13,
                equal_nan=False):
            transfer_runtime.last_resident_result = None
            raise CoupledAMRZeroTimeOwnerGPUV98Error(
                "v98 CUDA transfer differs from independent host oracle")
    errors.setflags(write=False)
    status = np.array(observed["parent_status"], dtype=np.int32,
                      order="C", copy=True)
    status.setflags(write=False)
    audit = observed["audit"]
    evidence_placeholder = CoupledAMRZeroTimeCUDAEvidenceGPUV98(
        transfer_runtime.receipt["receipt_sha256"],
        transfer_runtime.receipt["source_sha256"],
        transfer_runtime.receipt["wrapper_sha256"],
        joined.receipt_sha256, restriction_sha, boundary.receipt_sha256,
        _digest(parent_q, parent_suspended, parent_bed, parent_fluid),
        _digest(*arrays, status), status, errors,
        audit["numerical_kernel_launch_count"],
        audit["validation_input_upload_count"],
        audit["validation_output_readback_count"],
        audit["python_numerical_parent_child_loop_count"],
        audit["constitutive_evaluation_count"],
        audit["physical_boundary_evaluation_count"],
        audit["state_or_clock_publication_count"], False, BACKEND, "")
    evidence = CoupledAMRZeroTimeCUDAEvidenceGPUV98(
        **{**evidence_placeholder.__dict__,
           "receipt_sha256": _hash(_cuda_payload(evidence_placeholder))})
    _validate_cuda(evidence, transfer_runtime)

    child_q = np.asarray(observed["child_q"][:, 0, :])
    hles = _build_hles_map(
        source, fine_execution, fine_patch, joined, child_q)
    child_dss = np.broadcast_to(
        parent_dss[:, None], (fine_execution.fraction_count, 4))
    refined = _derive_image(
        fine_execution, layout="refined_children",
        global_cell_ids=joined.covered_child_global_ids,
        q=child_q, suspended=observed["child_suspended"][:, 0, :],
        bed=observed["child_bed"][:, 0, :],
        fluid=observed["child_fluid"][:, 0, :], dss=child_dss,
        initial_bed=initial_child, reference_bed=reference_child,
        hles_face_keys=tuple(
            f"fine:{int(value)}" for value in
            hles.incident_fine_face_global_ids),
        hles_filter=hles.refined_face_filtered_normal_velocity_m_s,
        hles_vic=hles.refined_child_dynamic_vicuv_m2_s,
        hles_dic=hles.refined_child_dynamic_dicuv_m2_s,
        accepted_updates=source.accepted_updates, time_s=source.time_s)
    coarsened = _derive_image(
        fine_execution, layout="coarsened_virtual_parent",
        global_cell_ids=np.asarray((joined.covered_parent_global_id,)),
        q=np.asarray(observed["restricted_q"][:, 0])[:, None],
        suspended=observed["restricted_suspended"][:, 0, None],
        bed=observed["restricted_bed"][:, 0, None],
        fluid=observed["restricted_fluid"][:, 0, None],
        dss=parent_dss[:, None], initial_bed=initial_parent,
        reference_bed=reference_parent,
        hles_face_keys=tuple(
            f"parent:{joined.covered_parent_global_id}:{side}"
            for side in SIDE_NAMES),
        hles_filter=hles.parent_side_filtered_outward_velocity_m_s,
        hles_vic=np.asarray((hles.parent_dynamic_vicuv_m2_s,)),
        hles_dic=np.asarray((hles.parent_dynamic_dicuv_m2_s,)),
        accepted_updates=source.accepted_updates, time_s=source.time_s)
    placeholder = CoupledAMRZeroTimePlanGPUV98(
        coarse_execution.fingerprint, fine_execution.fingerprint, source,
        interface.fingerprint, geometry_binding.receipt_sha256,
        joined.receipt_sha256,
        geometry_binding.coarse_patch.receipt_sha256,
        fine_patch.receipt_sha256,
        _readonly(cells, (4,), "covered child cells", nonnegative=True,
                  integer=True),
        _readonly(joined.covered_child_global_ids, (4,),
                  "covered child global IDs", nonnegative=True,
                  integer=True),
        joined.covered_parent_global_id, transfer_runtime, evidence, oracle,
        boundary, hles, refined, coarsened, 1, 0, 1, 1, 0, 1, 0, 2, 2, 0,
        "")
    result = CoupledAMRZeroTimePlanGPUV98(
        **{**placeholder.__dict__,
           "plan_sha256": _hash(_plan_payload(placeholder))})
    transfer_runtime.last_resident_result = None
    return result


def validate_coupled_amr_zero_time_plan_gpu_v98(
    value, coarse_execution, fine_execution, fine_provider, interface,
    geometry_binding,
) -> None:
    if (type(value) is not CoupledAMRZeroTimePlanGPUV98 or
            value.plan_sha256 != _hash(_plan_payload(value)) or
            type(value.transfer_runtime) is not
                ResidentAMRZeroTimeTransferGPUV98 or
            value.coarse_execution_fingerprint !=
                coarse_execution.fingerprint or
            value.fine_execution_fingerprint != fine_execution.fingerprint or
            value.geometry_binding_receipt_sha256 !=
                geometry_binding.receipt_sha256 or
            value.interface_fingerprint != interface.fingerprint or
            value.virtual_parent_restriction_owner_count != 1 or
            value.accepted_covered_parent_row_count != 0 or
            value.resident_transfer_owner_count != 1 or
            value.host_oracle_evaluation_count != 1 or
            value.host_oracle_application_count != 0 or
            value.dss_transfer_count != 1 or
            value.dss_constitutive_evaluation_count != 0 or
            value.eos_derivation_count != 2 or
            value.derived_host_image_count != 2 or
            value.next_macro_solver_state_construction_count != 0):
        raise CoupledAMRZeroTimeOwnerGPUV98Error(
            "v98 CUDA zero-time plan identity or owner counts differ")
    validate_coupled_wetdry_gpu_state_v97(
        value.source_fine_state, fine_execution, fine_provider)
    validate_coupled_amr_geometry_binding_gpu_v98(geometry_binding)
    validate_coupled_amr_interface_map_gpu_v98(
        interface, coarse_execution, fine_execution, geometry_binding)
    validate_zero_time_refine_coarsen_v98(value.transfer_oracle)
    _validate_cuda(value.cuda_evidence, value.transfer_runtime)
    _validate_image(value.refined_image, fine_execution)
    _validate_image(value.coarsened_image, fine_execution)
    if (value.root_boundary_memory.receipt_sha256 !=
            _hash(_boundary_payload(value.root_boundary_memory)) or
            value.hles_map.receipt_sha256 != _hash(_hles_payload(value.hles_map))
            or value.covered_child_cells.flags.writeable or
            value.covered_child_global_ids.flags.writeable):
        raise CoupledAMRZeroTimeOwnerGPUV98Error(
            "v98 CUDA zero-time auxiliary evidence differs")
    expected_boundary = _build_boundary_memory(
        value.source_fine_state, fine_execution,
        geometry_binding.fine_patch)
    if (expected_boundary.receipt_sha256 !=
            value.root_boundary_memory.receipt_sha256):
        raise CoupledAMRZeroTimeOwnerGPUV98Error(
            "v98 CUDA root boundary memory differs from accepted source")


def _state(plan, *, layout, refine_events, coarsen_events):
    image = (plan.refined_image if layout == "refined_children"
             else plan.coarsened_image)
    parent_count, child_count = ((0, 4) if layout == "refined_children"
                                 else (1, 0))
    placeholder = CoupledAMRZeroTimeHierarchyStateGPUV98(
        plan, layout, image, image.accepted_updates, image.time_s,
        refine_events, coarsen_events, 0, 0, 0, 0, 0, 0,
        parent_count, child_count, "")
    return CoupledAMRZeroTimeHierarchyStateGPUV98(
        **{**placeholder.__dict__,
           "state_sha256": _hash(_state_payload(placeholder))})


def validate_coupled_amr_zero_time_state_gpu_v98(
    value, coarse_execution, fine_execution, fine_provider, interface,
    geometry_binding,
) -> None:
    if (type(value) is not CoupledAMRZeroTimeHierarchyStateGPUV98 or
            value.state_sha256 != _hash(_state_payload(value)) or
            value.accepted_updates !=
                value.plan.source_fine_state.accepted_updates or
            value.time_s != value.plan.source_fine_state.time_s or
            any(count != 0 for count in (
                value.accepted_update_increment_count,
                value.physical_time_increment_count,
                value.hydraulic_operator_evaluation_count,
                value.scalar_operator_evaluation_count,
                value.q3_operator_evaluation_count,
                value.constitutive_evaluation_count))):
        raise CoupledAMRZeroTimeOwnerGPUV98Error(
            "v98 CUDA zero-time state identity or clocks differ")
    validate_coupled_amr_zero_time_plan_gpu_v98(
        value.plan, coarse_execution, fine_execution, fine_provider,
        interface, geometry_binding)
    if value.active_layout == "refined_children":
        expected = (value.plan.refined_image, 1, 0, 0, 4)
    elif value.active_layout == "coarsened_virtual_parent":
        expected = (value.plan.coarsened_image, 1, 1, 1, 0)
    else:
        raise CoupledAMRZeroTimeOwnerGPUV98Error(
            "v98 CUDA zero-time active layout differs")
    if (value.active_image.receipt_sha256 != expected[0].receipt_sha256 or
            (value.refine_event_count, value.coarsen_event_count,
             value.active_parent_owner_count, value.active_child_owner_count)
            != expected[1:]):
        raise CoupledAMRZeroTimeOwnerGPUV98Error(
            "v98 CUDA zero-time event or active-owner counts differ")


def _trial(event, parent, plan, candidate, *, boundary_owner):
    placeholder = CoupledAMRZeroTimeEventTrialGPUV98(
        event, parent, plan, candidate, 1, boundary_owner, 1, 0, "")
    return CoupledAMRZeroTimeEventTrialGPUV98(
        **{**placeholder.__dict__,
           "trial_sha256": _hash(_trial_payload(placeholder))})


def validate_coupled_amr_zero_time_trial_gpu_v98(
    value, coarse_execution, fine_execution, fine_provider, interface,
    geometry_binding,
) -> None:
    if (type(value) is not CoupledAMRZeroTimeEventTrialGPUV98 or
            value.event_kind not in ("refine", "coarsen") or
            value.trial_sha256 != _hash(_trial_payload(value)) or
            value.transfer_owner_count != 1 or
            value.hles_mapping_owner_count != 1 or
            value.candidate_publication_count != 0 or
            value.boundary_memory_remap_owner_count not in (0, 1)):
        raise CoupledAMRZeroTimeOwnerGPUV98Error(
            "v98 CUDA zero-time trial identity differs")
    _sha(value.parent_state_sha256, "event parent state")
    validate_coupled_amr_zero_time_state_gpu_v98(
        value.candidate, coarse_execution, fine_execution, fine_provider,
        interface, geometry_binding)
    if value.plan.plan_sha256 != value.candidate.plan.plan_sha256:
        raise CoupledAMRZeroTimeOwnerGPUV98Error(
            "v98 CUDA zero-time trial/candidate plan differs")


def prepare_coupled_amr_zero_time_refine_gpu_v98(
    coarse_execution: CoupledWetDryGPUExecutionV97,
    fine_execution: CoupledWetDryGPUExecutionV97,
    evolved_fine_state: CoupledWetDryGPUStateV97,
    fine_provider,
    interface: CoupledAMRInterfaceMapGPUV98,
    geometry_binding: CoupledAMRGeometryBindingGPUV98, *,
    transfer_runtime: ResidentAMRZeroTimeTransferGPUV98,
    cancelled: bool | Callable[[], bool] = False,
) -> CoupledAMRZeroTimeEventTrialGPUV98:
    poll = _poll(cancelled)
    if poll():
        raise InterruptedError("v98 CUDA zero-time refine cancelled before transfer")
    try:
        plan = _make_plan(
            coarse_execution, fine_execution, evolved_fine_state,
            fine_provider, interface, geometry_binding, transfer_runtime,
            cancelled=poll)
        if poll():
            raise InterruptedError(
                "v98 CUDA zero-time refine cancelled after transfer")
        candidate = _state(
            plan, layout="refined_children", refine_events=1,
            coarsen_events=0)
        trial = _trial(
            "refine", evolved_fine_state.state_sha256, plan, candidate,
            boundary_owner=1)
        validate_coupled_amr_zero_time_trial_gpu_v98(
            trial, coarse_execution, fine_execution, fine_provider,
            interface, geometry_binding)
        return trial
    except Exception:
        transfer_runtime.last_resident_result = None
        raise


def prepare_coupled_amr_zero_time_coarsen_gpu_v98(
    refined: CoupledAMRZeroTimeHierarchyStateGPUV98,
    coarse_execution: CoupledWetDryGPUExecutionV97,
    fine_execution: CoupledWetDryGPUExecutionV97,
    fine_provider,
    interface: CoupledAMRInterfaceMapGPUV98,
    geometry_binding: CoupledAMRGeometryBindingGPUV98, *,
    cancelled: bool | Callable[[], bool] = False,
) -> CoupledAMRZeroTimeEventTrialGPUV98:
    poll = _poll(cancelled)
    if poll():
        raise InterruptedError("v98 CUDA zero-time coarsen cancelled before transfer")
    validate_coupled_amr_zero_time_state_gpu_v98(
        refined, coarse_execution, fine_execution, fine_provider,
        interface, geometry_binding)
    if refined.active_layout != "refined_children":
        raise CoupledAMRZeroTimeOwnerGPUV98Error(
            "v98 CUDA coarsen requires committed refined layout")
    candidate = _state(
        refined.plan, layout="coarsened_virtual_parent", refine_events=1,
        coarsen_events=1)
    if poll():
        raise InterruptedError("v98 CUDA zero-time coarsen cancelled after transfer")
    trial = _trial(
        "coarsen", refined.state_sha256, refined.plan, candidate,
        boundary_owner=0)
    validate_coupled_amr_zero_time_trial_gpu_v98(
        trial, coarse_execution, fine_execution, fine_provider,
        interface, geometry_binding)
    return trial


def commit_coupled_amr_zero_time_event_gpu_v98(
    trial: CoupledAMRZeroTimeEventTrialGPUV98,
    coarse_execution: CoupledWetDryGPUExecutionV97,
    fine_execution: CoupledWetDryGPUExecutionV97,
    fine_provider,
    interface: CoupledAMRInterfaceMapGPUV98,
    geometry_binding: CoupledAMRGeometryBindingGPUV98, *,
    cancelled: bool | Callable[[], bool] = False,
) -> CoupledAMRZeroTimeHierarchyStateGPUV98:
    poll = _poll(cancelled)
    if poll():
        raise InterruptedError("v98 CUDA zero-time event cancelled before publication")
    validate_coupled_amr_zero_time_trial_gpu_v98(
        trial, coarse_execution, fine_execution, fine_provider,
        interface, geometry_binding)
    if poll():
        raise InterruptedError("v98 CUDA zero-time event cancelled after validation")
    return trial.candidate


__all__ = [
    "BACKEND", "SCHEMA", "SCOPE_ID", "SEMANTIC_ID",
    "CoupledAMRZeroTimeOwnerGPUV98Error",
    "CoupledAMRRootBoundaryMemoryGPUV98",
    "CoupledAMRZeroTimeHlesMapGPUV98",
    "CoupledAMRZeroTimeActiveImageGPUV98",
    "CoupledAMRZeroTimeCUDAEvidenceGPUV98",
    "CoupledAMRZeroTimePlanGPUV98",
    "CoupledAMRZeroTimeHierarchyStateGPUV98",
    "CoupledAMRZeroTimeEventTrialGPUV98",
    "prepare_coupled_amr_zero_time_refine_gpu_v98",
    "prepare_coupled_amr_zero_time_coarsen_gpu_v98",
    "commit_coupled_amr_zero_time_event_gpu_v98",
    "validate_coupled_amr_zero_time_plan_gpu_v98",
    "validate_coupled_amr_zero_time_state_gpu_v98",
    "validate_coupled_amr_zero_time_trial_gpu_v98",
]
