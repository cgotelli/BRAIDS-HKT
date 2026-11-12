"""Transactional zero-time owner for the bounded joined-leaf CPU v98 gate.

The owner in this module starts from the *actual evolved* fine numerical
parent produced by :mod:`coupled_amr_joined_leaf_cpu_v98`.  It restricts the
four authenticated children to one non-owning virtual parent, invokes the
existing conservative zero-time transfer, derives bed/EOS fields from the
transferred conserved rows, and maps HLES and physical-boundary memories
without advancing a solver clock.

The resulting state is an additive v98 hierarchy image, not a forged v97
state.  In particular, this module does not use private v97 constructors and
does not claim that the coarsened image can yet drive another coupled macro
update.  Physical-boundary memories stay keyed to their finest root segments;
there is no face-memory averaging or physical-boundary ledger surrogate.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Callable

import numpy as np

from .coupled_amr_geometry_v98 import (
    CartesianAMRPatchGeometryV98,
    CoupledAMRJoinedGeometryV98,
    validate_cartesian_amr_patch_geometry_v98,
    validate_joined_amr_geometry_v98,
)
from .coupled_amr_register_cpu_v98 import (
    CoupledAMRInterfaceMapCPUV98,
    validate_coupled_amr_interface_map_cpu_v98,
)
from .coupled_amr_zero_time_transfer_v98 import (
    CoupledAMRZeroTimeTransferV98,
    resolve_zero_time_refine_coarsen_v98,
    validate_zero_time_refine_coarsen_v98,
)
from .coupled_amr_zero_time_native_cpu_v98 import (
    NativeAMRZeroTimeTransferCPUV98,
    NativeAMRZeroTimeTransferResultCPUV98,
    validate_native_amr_zero_time_transfer_result_cpu_v98,
)
from .coupled_densin_cpu_v96 import _grid_execution, _v92_execution, _v92_state
from .coupled_wetdry_owner_cpu_v97 import (
    CoupledWetDryCPUExecutionV97,
    CoupledWetDryCPUStateV97,
    validate_coupled_wetdry_cpu_state_v97,
)
from .densin_fwave_v85 import mixture_density_kg_m3_v85
from .transported_fluid_settling_v92 import (
    MODE_TRANSPORTED,
    _properties as _fluid_properties,
)


SCHEMA = 1
SCOPE_ID = "braids.coupled_amr.zero_time_owner.cpu.v98"
SEMANTIC_ID = "one_evolved_parent_four_children_refine_then_coarsen"
CHILD_COUNT = 4
SIDE_NAMES = ("west", "east", "south", "north")
_SHA = re.compile(r"[0-9a-f]{64}\Z")


class CoupledAMRZeroTimeOwnerCPUV98Error(ValueError):
    """The bounded zero-time owner identity or algebra differs."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError, OverflowError) as error:
        raise CoupledAMRZeroTimeOwnerCPUV98Error(
            "v98 zero-time value is not finite canonical JSON") from error


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
        raise CoupledAMRZeroTimeOwnerCPUV98Error(
            f"v98 zero-time {name} must be a lowercase SHA-256 digest")
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
        raise CoupledAMRZeroTimeOwnerCPUV98Error(
            f"v98 zero-time {name} differs")
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
class CoupledAMRRootBoundaryMemoryCPUV98:
    """Boundary history retained on immutable finest root segments."""

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


def _boundary_payload(value: CoupledAMRRootBoundaryMemoryCPUV98) -> dict:
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
        "finest_segment_storage_count":
            value.finest_segment_storage_count,
        "remap_evaluation_count": value.remap_evaluation_count,
        "physical_boundary_ledger_application_count":
            value.physical_boundary_ledger_application_count,
        "mapping": "identity_on_stable_finest_physical_face_global_id",
    }


@dataclass(frozen=True)
class CoupledAMRZeroTimeHlesMapCPUV98:
    """Local HLES memory restriction/prolongation for the four-child patch."""

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


def _hles_payload(value: CoupledAMRZeroTimeHlesMapCPUV98) -> dict:
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
        "outer_segment_mapping_count":
            value.outer_segment_mapping_count,
        "new_internal_face_initialization_count":
            value.new_internal_face_initialization_count,
        "hles_physical_time_evaluation_count":
            value.hles_physical_time_evaluation_count,
        "mapping": (
            "outer_length_weighted_parent_then_piecewise_constant_refine;"
            "new_internal_faces_equal_transferred_endpoint_mean_velocity"),
    }


@dataclass(frozen=True)
class CoupledAMRZeroTimeActiveImageCPUV98:
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
    receipt_sha256: str


def _image_payload(value: CoupledAMRZeroTimeActiveImageCPUV98) -> dict:
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
    }


@dataclass(frozen=True)
class CoupledAMRZeroTimePlanCPUV98:
    coarse_execution_fingerprint: str
    fine_execution_fingerprint: str
    source_fine_state: CoupledWetDryCPUStateV97
    interface_fingerprint: str
    joined_geometry_receipt_sha256: str
    coarse_patch_receipt_sha256: str
    fine_patch_receipt_sha256: str
    covered_child_cells: np.ndarray
    covered_child_global_ids: np.ndarray
    covered_parent_global_id: int
    native_transfer_runtime: NativeAMRZeroTimeTransferCPUV98
    native_transfer_result: NativeAMRZeroTimeTransferResultCPUV98
    transfer: CoupledAMRZeroTimeTransferV98
    root_boundary_memory: CoupledAMRRootBoundaryMemoryCPUV98
    hles_map: CoupledAMRZeroTimeHlesMapCPUV98
    refined_image: CoupledAMRZeroTimeActiveImageCPUV98
    coarsened_image: CoupledAMRZeroTimeActiveImageCPUV98
    virtual_parent_restriction_owner_count: int
    accepted_covered_parent_row_count: int
    native_transfer_owner_count: int
    host_oracle_evaluation_count: int
    host_oracle_application_count: int
    dss_transfer_count: int
    dss_constitutive_evaluation_count: int
    eos_derivation_count: int
    next_macro_solver_state_construction_count: int
    plan_sha256: str


def _plan_payload(value: CoupledAMRZeroTimePlanCPUV98) -> dict:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID+":plan",
        "semantic_id": SEMANTIC_ID,
        "executions": [value.coarse_execution_fingerprint,
                       value.fine_execution_fingerprint],
        "source_fine_state": value.source_fine_state.state_sha256,
        "interface": value.interface_fingerprint,
        "geometry": [value.joined_geometry_receipt_sha256,
                     value.coarse_patch_receipt_sha256,
                     value.fine_patch_receipt_sha256],
        "covered_children_sha256": _digest(
            value.covered_child_cells, value.covered_child_global_ids),
        "covered_parent_global_id": value.covered_parent_global_id,
        "transfer": {
            "authoritative_native_runtime":
                value.native_transfer_runtime.runtime_sha256,
            "authoritative_native_result":
                value.native_transfer_result.receipt_sha256,
            "validation_only_host_oracle": value.transfer.receipt_sha256,
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
            "native_transfer": value.native_transfer_owner_count,
            "host_oracle_evaluation": value.host_oracle_evaluation_count,
            "host_oracle_application": value.host_oracle_application_count,
            "dss_transfer": value.dss_transfer_count,
            "dss_constitutive_evaluation":
                value.dss_constitutive_evaluation_count,
            "eos_derivation": value.eos_derivation_count,
            "next_macro_solver_state_construction":
                value.next_macro_solver_state_construction_count,
        },
        "scope_limit": "one_evolved_joined_leaf_not_general_regrid",
    }


@dataclass(frozen=True)
class CoupledAMRZeroTimeHierarchyStateCPUV98:
    plan: CoupledAMRZeroTimePlanCPUV98
    active_layout: str
    active_image: CoupledAMRZeroTimeActiveImageCPUV98
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


def _state_payload(value: CoupledAMRZeroTimeHierarchyStateCPUV98) -> dict:
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
class CoupledAMRZeroTimeEventTrialCPUV98:
    event_kind: str
    parent_state_sha256: str
    plan: CoupledAMRZeroTimePlanCPUV98
    candidate: CoupledAMRZeroTimeHierarchyStateCPUV98
    transfer_owner_count: int
    boundary_memory_remap_owner_count: int
    hles_mapping_owner_count: int
    candidate_publication_count: int
    trial_sha256: str


def _trial_payload(value: CoupledAMRZeroTimeEventTrialCPUV98) -> dict:
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


def _build_boundary_memory(
    source: CoupledWetDryCPUStateV97,
    fine_execution: CoupledWetDryCPUExecutionV97,
    fine_patch: CartesianAMRPatchGeometryV98,
) -> CoupledAMRRootBoundaryMemoryCPUV98:
    grid = _grid_execution(fine_execution.active_execution)
    v92 = _v92_state(source.active_state)
    lifecycle = v92.suspended_state.lifecycle
    physical = np.flatnonzero(np.asarray(grid.boundary_index) >= 0)
    if physical.size == 0:
        raise CoupledAMRZeroTimeOwnerCPUV98Error(
            "v98 zero-time fixture has no physical root boundary segments")
    nonphysical = np.flatnonzero(np.asarray(grid.boundary_index) < 0)
    memory_arrays = (
        lifecycle.boundary_last_outflow_concentration_kg_m3,
        lifecycle.boundary_return_clock_s,
        v92.boundary_last_outflow_fluid,
        v92.boundary_return_clock_s,
    )
    if any(np.any(np.asarray(array)[:, nonphysical] != 0.0)
           for array in memory_arrays):
        raise CoupledAMRZeroTimeOwnerCPUV98Error(
            "v98 nonphysical internal faces carry boundary return memory")
    indices = _readonly(
        physical, (physical.size,), "physical root face indices",
        nonnegative=True, integer=True)
    global_ids = _readonly(
        fine_patch.face_global_ids[physical], (physical.size,),
        "physical root face global IDs", nonnegative=True, integer=True)
    source_ids = _readonly(
        np.asarray(grid.boundary_index)[physical], (physical.size,),
        "physical root boundary source IDs", nonnegative=True, integer=True)
    ng = fine_execution.fraction_count
    sediment_last = _readonly(
        lifecycle.boundary_last_outflow_concentration_kg_m3[:, physical],
        (ng, physical.size), "sediment boundary memory", nonnegative=True)
    sediment_clock = _readonly(
        lifecycle.boundary_return_clock_s[:, physical],
        (ng, physical.size), "sediment boundary clocks", nonnegative=True)
    fluid_last = _readonly(
        v92.boundary_last_outflow_fluid[:, physical],
        (2, physical.size), "fluid boundary memory")
    fluid_clock = _readonly(
        v92.boundary_return_clock_s[:, physical],
        (2, physical.size), "fluid boundary clocks", nonnegative=True)
    placeholder = CoupledAMRRootBoundaryMemoryCPUV98(
        indices, global_ids, source_ids, sediment_last, sediment_clock,
        fluid_last, fluid_clock, global_ids, int(physical.size), 1, 0, "")
    return CoupledAMRRootBoundaryMemoryCPUV98(
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
    raise CoupledAMRZeroTimeOwnerCPUV98Error(
        "v98 incident outer face is not on the covered-parent boundary")


def _build_hles_map(
    source: CoupledWetDryCPUStateV97,
    fine_execution: CoupledWetDryCPUExecutionV97,
    fine_patch: CartesianAMRPatchGeometryV98,
    joined: CoupledAMRJoinedGeometryV98,
    transferred_q: np.ndarray,
) -> CoupledAMRZeroTimeHlesMapCPUV98:
    grid = _grid_execution(fine_execution.active_execution)
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
        raise CoupledAMRZeroTimeOwnerCPUV98Error(
            "v98 HLES incident-face classification is incomplete")
    parent_bounds = np.asarray(joined.covered_parent_bounds_m)
    expected_length = np.array((
        parent_bounds[3]-parent_bounds[1],
        parent_bounds[3]-parent_bounds[1],
        parent_bounds[2]-parent_bounds[0],
        parent_bounds[2]-parent_bounds[0]))
    if np.any(np.abs(side_length-expected_length) >
              _tolerance(side_length, expected_length)):
        raise CoupledAMRZeroTimeOwnerCPUV98Error(
            "v98 HLES outer segments do not partition parent sides")
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
            if depth > grid.dry_tolerance_m:
                velocities.append(transferred_q[1:, child]/depth)
            else:
                velocities.append(np.zeros(2))
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
        raise CoupledAMRZeroTimeOwnerCPUV98Error(
            "v98 HLES refine/coarsen outer memory does not close")

    area = np.asarray(grid.cell_area_m2)[cells]
    total_area = float(np.sum(area, dtype=np.float64))
    source_vic = np.asarray(source.hles_state.accepted_dynamic_vicuv_m2_s)[cells]
    source_dic = np.asarray(source.hles_state.accepted_dynamic_dicuv_m2_s)[cells]
    parent_vic = float(np.sum(source_vic*area, dtype=np.float64)/total_area)
    parent_dic = float(np.sum(source_dic*area, dtype=np.float64)/total_area)
    refined_vic = np.full(CHILD_COUNT, parent_vic)
    refined_dic = np.full(CHILD_COUNT, parent_dic)
    placeholder = CoupledAMRZeroTimeHlesMapCPUV98(
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
    return CoupledAMRZeroTimeHlesMapCPUV98(
        **{**placeholder.__dict__,
           "receipt_sha256": _hash(_hles_payload(placeholder))})


def _derive_image(
    execution: CoupledWetDryCPUExecutionV97, *, layout: str,
    global_cell_ids: np.ndarray, q: np.ndarray, suspended: np.ndarray,
    bed: np.ndarray, fluid: np.ndarray, dss: np.ndarray,
    initial_bed: np.ndarray, reference_bed: np.ndarray,
    hles_face_keys: tuple[str, ...], hles_filter: np.ndarray,
    hles_vic: np.ndarray, hles_dic: np.ndarray,
    accepted_updates: int, time_s: float,
) -> CoupledAMRZeroTimeActiveImageCPUV98:
    ng = execution.fraction_count
    nc = int(np.asarray(global_cell_ids).size)
    qv = _readonly(q, (3, nc), "active conserved hydraulic image",
                   nonnegative=False)
    suspended_v = _readonly(
        suspended, (ng, nc), "active suspension image", nonnegative=True)
    bed_v = _readonly(bed, (ng, nc), "active bed image", nonnegative=True)
    fluid_v = _readonly(fluid, (2, nc), "active fluid image")
    dss_v = _readonly(dss, (ng, nc), "active DSS image", positive=True)
    z0 = _readonly(initial_bed, (nc,), "active initial bed")
    m0 = _readonly(reference_bed, (ng, nc), "active reference bed",
                   nonnegative=True)
    if np.any(qv[0] < 0.0) or np.any(fluid_v[1] < 0.0):
        raise CoupledAMRZeroTimeOwnerCPUV98Error(
            "v98 active water depth or salinity inventory is negative")
    grid = _grid_execution(execution.active_execution)
    dry_tolerance = grid.dry_tolerance_m
    wet = qv[0] > dry_tolerance
    if (np.any(qv[1:, ~wet] != 0.0) or
            np.any(suspended_v[:, ~wet] != 0.0) or
            np.any(fluid_v[:, ~wet] != 0.0)):
        raise CoupledAMRZeroTimeOwnerCPUV98Error(
            "v98 active dry cells retain transported inventory")
    bed_z = np.ascontiguousarray(
        z0+np.sum((bed_v-m0)/execution.dry_bulk_density_kg_m3[:, None],
                  axis=0))
    eta = np.ascontiguousarray(qv[0]+bed_z)
    v92_execution = _v92_execution(execution.active_execution)
    temperature = np.full(nc, v92_execution.equilibrium_temperature_c)
    salinity = np.full(nc, v92_execution.equilibrium_salinity_ppt)
    if v92_execution.temperature_mode == MODE_TRANSPORTED:
        temperature[wet] = fluid_v[0, wet]/qv[0, wet]
    if v92_execution.salinity_mode == MODE_TRANSPORTED:
        salinity[wet] = fluid_v[1, wet]/qv[0, wet]
    if np.any(temperature <= -20.0) or np.any(salinity < -_tolerance(salinity)):
        raise CoupledAMRZeroTimeOwnerCPUV98Error(
            "v98 transferred temperature or salinity is outside EOS scope")
    salinity = np.maximum(salinity, 0.0)
    water_density, viscosity = _fluid_properties(
        v92_execution, np.ascontiguousarray(temperature),
        np.ascontiguousarray(salinity))
    concentration = np.zeros((ng, nc), dtype=np.float64)
    concentration[:, wet] = suspended_v[:, wet]/qv[0, wet]
    mixture_density = mixture_density_kg_m3_v85(
        concentration, water_density,
        np.asarray(grid.profile.sediment_density_kg_m3))
    arrays = [bed_z, eta, temperature, salinity, water_density,
              mixture_density, viscosity, wet.astype(np.int64)]
    frozen = []
    for array in arrays:
        item = np.ascontiguousarray(array)
        item.setflags(write=False)
        frozen.append(item)
    ids = _readonly(global_cell_ids, (nc,), "active global cell IDs",
                    nonnegative=True, integer=True)
    filter_v = _readonly(
        hles_filter, (len(hles_face_keys),), "active HLES filter")
    vic_v = _readonly(hles_vic, (nc,), "active HLES vicuv",
                     nonnegative=True)
    dic_v = _readonly(hles_dic, (nc,), "active HLES dicuv",
                     nonnegative=True)
    if (type(accepted_updates) is not int or not 0 <= accepted_updates <= 4 or
            not isinstance(time_s, (int, float)) or not math.isfinite(time_s)
            or time_s < 0.0 or len(set(hles_face_keys)) != len(hles_face_keys)):
        raise CoupledAMRZeroTimeOwnerCPUV98Error(
            "v98 active image clocks or HLES keys differ")
    placeholder = CoupledAMRZeroTimeActiveImageCPUV98(
        layout, ids, qv, suspended_v, bed_v, fluid_v[0], fluid_v[1],
        frozen[2], frozen[3], frozen[4], frozen[5], frozen[6],
        frozen[0], frozen[1], frozen[7], dss_v,
        tuple(hles_face_keys), filter_v, vic_v, dic_v,
        accepted_updates, float(time_s), "")
    return CoupledAMRZeroTimeActiveImageCPUV98(
        **{**placeholder.__dict__,
           "receipt_sha256": _hash(_image_payload(placeholder))})


def _make_plan(
    coarse_execution: CoupledWetDryCPUExecutionV97,
    fine_execution: CoupledWetDryCPUExecutionV97,
    source: CoupledWetDryCPUStateV97,
    interface: CoupledAMRInterfaceMapCPUV98,
    joined: CoupledAMRJoinedGeometryV98,
    coarse_patch: CartesianAMRPatchGeometryV98,
    fine_patch: CartesianAMRPatchGeometryV98,
    native_transfer_runtime: NativeAMRZeroTimeTransferCPUV98, *,
    native_threads: int = 4,
    cancelled: bool | Callable[[], bool] = False,
    validated_native_result: NativeAMRZeroTimeTransferResultCPUV98 | None = None,
    validated_host_oracle: CoupledAMRZeroTimeTransferV98 | None = None,
) -> CoupledAMRZeroTimePlanCPUV98:
    if type(native_transfer_runtime) is not NativeAMRZeroTimeTransferCPUV98:
        raise CoupledAMRZeroTimeOwnerCPUV98Error(
            "v98 zero-time native transfer runtime type differs")
    validate_coupled_wetdry_cpu_state_v97(source, fine_execution, None)
    validate_coupled_amr_interface_map_cpu_v98(
        interface, coarse_execution, fine_execution)
    coarse_grid = _grid_execution(coarse_execution.active_execution)
    fine_grid = _grid_execution(fine_execution.active_execution)
    validate_cartesian_amr_patch_geometry_v98(coarse_patch, coarse_grid)
    validate_cartesian_amr_patch_geometry_v98(fine_patch, fine_grid)
    validate_joined_amr_geometry_v98(
        joined, coarse_grid, fine_grid, coarse_patch, fine_patch)
    if joined.interface_fingerprint != interface.fingerprint:
        raise CoupledAMRZeroTimeOwnerCPUV98Error(
            "v98 zero-time joined geometry and interface differ")
    cells = np.asarray(joined.covered_child_cells, dtype=np.int64)
    if cells.shape != (4,) or len(set(map(int, cells))) != 4:
        raise CoupledAMRZeroTimeOwnerCPUV98Error(
            "v98 zero-time covered-child selection differs")
    area = np.asarray(fine_grid.cell_area_m2)[cells]
    total_area = float(joined.covered_parent_area_m2)
    if abs(float(np.sum(area))-total_area) > _tolerance(area, total_area):
        raise CoupledAMRZeroTimeOwnerCPUV98Error(
            "v98 zero-time child areas do not cover the virtual parent")
    v92 = _v92_state(source.active_state)
    q_source = np.asarray(v92.q)[:, cells]
    suspended_source = np.asarray(
        v92.suspended_state.suspended_mass_kg_m2)[:, cells]
    bed_source = np.asarray(source.active_state.base_state.bed_mass_kg_m2)[:, cells]
    fluid_source = np.vstack((
        np.asarray(v92.h_temperature_c_m)[cells],
        np.asarray(v92.h_salinity_ppt_m)[cells]))
    dss_source = np.asarray(source.active_state.suspended_diameter_m)[:, cells]
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
    # The transfer's water stage must use the bed implied by the bed mass that
    # will actually be prolonged, not the pre-event source bed.  Dry transfer
    # can subsequently change only dry-child bed rows.
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
    if validated_host_oracle is None:
        transfer = resolve_zero_time_refine_coarsen_v98(
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
    else:
        transfer = validated_host_oracle
        validate_zero_time_refine_coarsen_v98(transfer)
        if (transfer.hierarchy_fingerprint != joined.receipt_sha256 or
                transfer.parent_state_sha256 != restriction_sha or
                transfer.parent_area_m2 != total_area or
                transfer.dry_tolerance_m != fine_grid.dry_tolerance_m or
                transfer.parent_depth_m != float(parent_q[0]) or
                transfer.boundary_source_ids != source_ids or
                transfer.boundary_memory_sha256 != boundary.receipt_sha256 or
                transfer.accepted_updates != source.accepted_updates or
                transfer.time_s != source.time_s or
                not np.array_equal(transfer.child_area_m2, area) or
                not np.array_equal(
                    transfer.child_bed_elevation_m, seed_child_bed) or
                not np.array_equal(
                    transfer.parent_momentum_m2_s, parent_q[1:]) or
                not np.array_equal(
                    transfer.parent_suspended_kg_m2, parent_suspended) or
                not np.array_equal(transfer.parent_bed_kg_m2, parent_bed) or
                not np.array_equal(
                    transfer.parent_fluid_integral, parent_fluid)):
            raise CoupledAMRZeroTimeOwnerCPUV98Error(
                "v98 zero-time stored host oracle inputs differ")
    native_inputs = (
        np.asarray((total_area,), dtype=np.float64),
        np.ascontiguousarray(area[None, :], dtype=np.float64),
        np.ascontiguousarray(seed_child_bed[None, :], dtype=np.float64),
        np.ascontiguousarray(parent_q[:, None], dtype=np.float64),
        np.ascontiguousarray(parent_suspended[:, None], dtype=np.float64),
        np.ascontiguousarray(parent_bed[:, None], dtype=np.float64),
        np.ascontiguousarray(parent_fluid[:, None], dtype=np.float64),
    )
    if validated_native_result is None:
        native = native_transfer_runtime.evaluate(
            *native_inputs, dry_tolerance_m=fine_grid.dry_tolerance_m,
            native_threads=native_threads, cancelled=cancelled)
    else:
        native = validated_native_result
        native_transfer_runtime._validate_identity()
        validate_native_amr_zero_time_transfer_result_cpu_v98(native)
        if (native.runtime_sha256 != native_transfer_runtime.runtime_sha256 or
                native.native_threads != native_threads or
                native.input_sha256 != _digest(*native_inputs)):
            raise CoupledAMRZeroTimeOwnerCPUV98Error(
                "v98 zero-time stored native result inputs differ")
    validate_native_amr_zero_time_transfer_result_cpu_v98(native)
    oracle_arrays = (
        (native.child_q[0, 0], transfer.child_depth_m, "child depth"),
        (native.child_q[1:, 0], transfer.child_momentum_m2_s,
         "child momentum"),
        (native.child_suspended[:, 0], transfer.child_suspended_kg_m2,
         "child suspended mass"),
        (native.child_bed[:, 0], transfer.child_bed_kg_m2,
         "child bed mass"),
        (native.child_fluid[:, 0], transfer.child_fluid_integral,
         "child transported fluid"),
        (native.restricted_q[0, 0], transfer.restricted_depth_m,
         "restricted depth"),
        (native.restricted_q[1:, 0], transfer.restricted_momentum_m2_s,
         "restricted momentum"),
        (native.restricted_suspended[:, 0],
         transfer.restricted_suspended_kg_m2,
         "restricted suspended mass"),
        (native.restricted_bed[:, 0], transfer.restricted_bed_kg_m2,
         "restricted bed mass"),
        (native.restricted_fluid[:, 0], transfer.restricted_fluid_integral,
         "restricted transported fluid"),
    )
    for authoritative, oracle, name in oracle_arrays:
        if not np.allclose(
                authoritative, oracle, rtol=3.0e-14, atol=3.0e-13,
                equal_nan=False):
            raise CoupledAMRZeroTimeOwnerCPUV98Error(
                f"v98 native zero-time {name} differs from host oracle")
    # Only the native result is consumed below.  The Python transfer remains
    # an independent validation oracle and never owns a numerical application.
    child_q = np.asarray(native.child_q[:, 0, :])
    hles = _build_hles_map(
        source, fine_execution, fine_patch, joined, child_q)
    child_dss = np.broadcast_to(
        parent_dss[:, None], (fine_execution.fraction_count, 4))
    refined = _derive_image(
        fine_execution, layout="refined_children",
        global_cell_ids=joined.covered_child_global_ids,
        q=child_q, suspended=native.child_suspended[:, 0, :],
        bed=native.child_bed[:, 0, :],
        fluid=native.child_fluid[:, 0, :], dss=child_dss,
        initial_bed=initial_child, reference_bed=reference_child,
        hles_face_keys=tuple(
            f"fine:{int(value)}" for value in
            hles.incident_fine_face_global_ids),
        hles_filter=hles.refined_face_filtered_normal_velocity_m_s,
        hles_vic=hles.refined_child_dynamic_vicuv_m2_s,
        hles_dic=hles.refined_child_dynamic_dicuv_m2_s,
        accepted_updates=source.accepted_updates, time_s=source.time_s)
    parent_q_after = np.asarray(native.restricted_q[:, 0])[:, None]
    coarsened = _derive_image(
        fine_execution, layout="coarsened_virtual_parent",
        global_cell_ids=np.asarray((joined.covered_parent_global_id,)),
        q=parent_q_after,
        suspended=native.restricted_suspended[:, 0, None],
        bed=native.restricted_bed[:, 0, None],
        fluid=native.restricted_fluid[:, 0, None],
        dss=parent_dss[:, None], initial_bed=initial_parent,
        reference_bed=reference_parent,
        hles_face_keys=tuple(
            f"parent:{joined.covered_parent_global_id}:{side}"
            for side in SIDE_NAMES),
        hles_filter=hles.parent_side_filtered_outward_velocity_m_s,
        hles_vic=np.asarray((hles.parent_dynamic_vicuv_m2_s,)),
        hles_dic=np.asarray((hles.parent_dynamic_dicuv_m2_s,)),
        accepted_updates=source.accepted_updates, time_s=source.time_s)
    placeholder = CoupledAMRZeroTimePlanCPUV98(
        coarse_execution_fingerprint=coarse_execution.fingerprint,
        fine_execution_fingerprint=fine_execution.fingerprint,
        source_fine_state=source,
        interface_fingerprint=interface.fingerprint,
        joined_geometry_receipt_sha256=joined.receipt_sha256,
        coarse_patch_receipt_sha256=coarse_patch.receipt_sha256,
        fine_patch_receipt_sha256=fine_patch.receipt_sha256,
        covered_child_cells=_readonly(
            cells, (4,), "covered child cells", nonnegative=True,
            integer=True),
        covered_child_global_ids=_readonly(
            joined.covered_child_global_ids, (4,),
            "covered child global IDs", nonnegative=True, integer=True),
        covered_parent_global_id=joined.covered_parent_global_id,
        native_transfer_runtime=native_transfer_runtime,
        native_transfer_result=native,
        transfer=transfer,
        root_boundary_memory=boundary,
        hles_map=hles,
        refined_image=refined,
        coarsened_image=coarsened,
        virtual_parent_restriction_owner_count=1,
        accepted_covered_parent_row_count=0,
        native_transfer_owner_count=1,
        host_oracle_evaluation_count=1,
        host_oracle_application_count=0,
        dss_transfer_count=1,
        dss_constitutive_evaluation_count=0,
        eos_derivation_count=2,
        next_macro_solver_state_construction_count=0,
        plan_sha256="")
    return CoupledAMRZeroTimePlanCPUV98(
        **{**placeholder.__dict__,
           "plan_sha256": _hash(_plan_payload(placeholder))})


def validate_coupled_amr_zero_time_plan_cpu_v98(
    value: CoupledAMRZeroTimePlanCPUV98,
    coarse_execution: CoupledWetDryCPUExecutionV97,
    fine_execution: CoupledWetDryCPUExecutionV97,
    interface: CoupledAMRInterfaceMapCPUV98,
    joined: CoupledAMRJoinedGeometryV98,
    coarse_patch: CartesianAMRPatchGeometryV98,
    fine_patch: CartesianAMRPatchGeometryV98,
) -> None:
    if (type(value) is not CoupledAMRZeroTimePlanCPUV98 or
            type(value.native_transfer_runtime) is not
                NativeAMRZeroTimeTransferCPUV98 or
            type(value.native_transfer_result) is not
                NativeAMRZeroTimeTransferResultCPUV98 or
            value.plan_sha256 != _hash(_plan_payload(value)) or
            value.coarse_execution_fingerprint !=
                coarse_execution.fingerprint or
            value.fine_execution_fingerprint != fine_execution.fingerprint or
            value.interface_fingerprint != interface.fingerprint or
            value.joined_geometry_receipt_sha256 != joined.receipt_sha256 or
            value.coarse_patch_receipt_sha256 !=
                coarse_patch.receipt_sha256 or
            value.fine_patch_receipt_sha256 != fine_patch.receipt_sha256 or
            value.virtual_parent_restriction_owner_count != 1 or
            value.accepted_covered_parent_row_count != 0 or
            value.native_transfer_owner_count != 1 or
            value.host_oracle_evaluation_count != 1 or
            value.host_oracle_application_count != 0 or
            value.dss_transfer_count != 1 or
            value.dss_constitutive_evaluation_count != 0 or
            value.eos_derivation_count != 2 or
            value.next_macro_solver_state_construction_count != 0):
        raise CoupledAMRZeroTimeOwnerCPUV98Error(
            "v98 zero-time plan identity or owner counts differ")
    validate_native_amr_zero_time_transfer_result_cpu_v98(
        value.native_transfer_result)
    if (value.native_transfer_result.runtime_sha256 !=
            value.native_transfer_runtime.runtime_sha256 or
            value.native_transfer_result.refinement_operator_application_count
                != 1 or
            value.native_transfer_result.restriction_operator_application_count
                != 1 or
            value.native_transfer_result.dry_cleanup_owner_count != 1 or
            any(count != 0 for count in (
                value.native_transfer_result.accepted_clock_increment_count,
                value.native_transfer_result.physical_time_increment_count,
                value.native_transfer_result.constitutive_evaluation_count,
                value.native_transfer_result.state_publication_count,
                value.native_transfer_result.
                    numerical_python_parent_child_loop_count))):
        raise CoupledAMRZeroTimeOwnerCPUV98Error(
            "v98 zero-time native result provenance or counts differ")
    validate_zero_time_refine_coarsen_v98(value.transfer)
    validate_coupled_amr_zero_time_active_image_cpu_v98(
        value.refined_image, fine_execution)
    validate_coupled_amr_zero_time_active_image_cpu_v98(
        value.coarsened_image, fine_execution)
    rebuilt = _make_plan(
        coarse_execution, fine_execution, value.source_fine_state,
        interface, joined, coarse_patch, fine_patch,
        value.native_transfer_runtime,
        native_threads=value.native_transfer_result.native_threads,
        validated_native_result=value.native_transfer_result,
        validated_host_oracle=value.transfer)
    if rebuilt.plan_sha256 != value.plan_sha256:
        raise CoupledAMRZeroTimeOwnerCPUV98Error(
            "v98 zero-time plan does not rebuild from the evolved source")


def validate_coupled_amr_zero_time_active_image_cpu_v98(
    value: CoupledAMRZeroTimeActiveImageCPUV98,
    execution: CoupledWetDryCPUExecutionV97,
) -> None:
    if (type(value) is not CoupledAMRZeroTimeActiveImageCPUV98 or
            value.layout not in
                ("refined_children", "coarsened_virtual_parent") or
            value.receipt_sha256 != _hash(_image_payload(value)) or
            value.accepted_updates > 4 or value.accepted_updates < 0 or
            value.time_s < 0.0 or not math.isfinite(value.time_s)):
        raise CoupledAMRZeroTimeOwnerCPUV98Error(
            "v98 zero-time active image identity differs")
    nc = value.global_cell_ids.size
    ng = execution.fraction_count
    arrays = (
        (value.global_cell_ids, (nc,), True, True),
        (value.q, (3, nc), False, False),
        (value.suspended_mass_kg_m2, (ng, nc), True, False),
        (value.bed_mass_kg_m2, (ng, nc), True, False),
        (value.h_temperature_c_m, (nc,), False, False),
        (value.h_salinity_ppt_m, (nc,), True, False),
        (value.temperature_cell_c, (nc,), False, False),
        (value.salinity_cell_ppt, (nc,), True, False),
        (value.water_density_cell_kg_m3, (nc,), True, False),
        (value.mixture_density_cell_kg_m3, (nc,), True, False),
        (value.molecular_viscosity_cell_m2_s, (nc,), True, False),
        (value.bed_elevation_m, (nc,), False, False),
        (value.free_surface_elevation_m, (nc,), False, False),
        (value.wet_mask, (nc,), True, True),
        (value.suspended_diameter_m, (ng, nc), True, False),
        (value.hles_filtered_face_normal_velocity_m_s,
         (len(value.hles_face_keys),), False, False),
        (value.hles_dynamic_vicuv_m2_s, (nc,), True, False),
        (value.hles_dynamic_dicuv_m2_s, (nc,), True, False),
    )
    for array, shape, nonnegative, integer in arrays:
        checked = _readonly(
            array, shape, "validated active image", nonnegative=nonnegative,
            positive=(array is value.suspended_diameter_m), integer=integer)
        if (array.flags.writeable or not array.flags.c_contiguous or
                not np.array_equal(array, checked)):
            raise CoupledAMRZeroTimeOwnerCPUV98Error(
                "v98 zero-time active image arrays are mutable or stale")
    if (not np.array_equal(
            value.free_surface_elevation_m,
            value.q[0]+value.bed_elevation_m) or
            not np.array_equal(value.wet_mask,
                (value.q[0] > _grid_execution(
                    execution.active_execution).dry_tolerance_m).astype(
                        np.int64))):
        raise CoupledAMRZeroTimeOwnerCPUV98Error(
            "v98 zero-time derived stage or wet mask differs")


def _state(
    plan: CoupledAMRZeroTimePlanCPUV98, *, layout: str,
    refine_events: int, coarsen_events: int,
) -> CoupledAMRZeroTimeHierarchyStateCPUV98:
    image = (plan.refined_image if layout == "refined_children"
             else plan.coarsened_image)
    parent_count, child_count = ((0, 4) if layout == "refined_children"
                                 else (1, 0))
    placeholder = CoupledAMRZeroTimeHierarchyStateCPUV98(
        plan, layout, image, image.accepted_updates, image.time_s,
        refine_events, coarsen_events, 0, 0, 0, 0, 0, 0,
        parent_count, child_count, "")
    return CoupledAMRZeroTimeHierarchyStateCPUV98(
        **{**placeholder.__dict__,
           "state_sha256": _hash(_state_payload(placeholder))})


def validate_coupled_amr_zero_time_state_cpu_v98(
    value: CoupledAMRZeroTimeHierarchyStateCPUV98,
    coarse_execution: CoupledWetDryCPUExecutionV97,
    fine_execution: CoupledWetDryCPUExecutionV97,
    interface: CoupledAMRInterfaceMapCPUV98,
    joined: CoupledAMRJoinedGeometryV98,
    coarse_patch: CartesianAMRPatchGeometryV98,
    fine_patch: CartesianAMRPatchGeometryV98,
) -> None:
    if (type(value) is not CoupledAMRZeroTimeHierarchyStateCPUV98 or
            value.state_sha256 != _hash(_state_payload(value)) or
            value.accepted_updates != value.plan.source_fine_state.accepted_updates
            or value.time_s != value.plan.source_fine_state.time_s or
            any(count != 0 for count in (
                value.accepted_update_increment_count,
                value.physical_time_increment_count,
                value.hydraulic_operator_evaluation_count,
                value.scalar_operator_evaluation_count,
                value.q3_operator_evaluation_count,
                value.constitutive_evaluation_count))):
        raise CoupledAMRZeroTimeOwnerCPUV98Error(
            "v98 zero-time hierarchy state identity or clocks differ")
    validate_coupled_amr_zero_time_plan_cpu_v98(
        value.plan, coarse_execution, fine_execution, interface, joined,
        coarse_patch, fine_patch)
    if value.active_layout == "refined_children":
        expected = (value.plan.refined_image, 1, 0, 0, 4)
    elif value.active_layout == "coarsened_virtual_parent":
        expected = (value.plan.coarsened_image, 1, 1, 1, 0)
    else:
        raise CoupledAMRZeroTimeOwnerCPUV98Error(
            "v98 zero-time active layout differs")
    if (value.active_image.receipt_sha256 != expected[0].receipt_sha256 or
            (value.refine_event_count, value.coarsen_event_count,
             value.active_parent_owner_count, value.active_child_owner_count)
            != expected[1:]):
        raise CoupledAMRZeroTimeOwnerCPUV98Error(
            "v98 zero-time event or active owner counts differ")


def _trial(event: str, parent: str, plan: CoupledAMRZeroTimePlanCPUV98,
           candidate: CoupledAMRZeroTimeHierarchyStateCPUV98,
           *, boundary_owner: int) -> CoupledAMRZeroTimeEventTrialCPUV98:
    placeholder = CoupledAMRZeroTimeEventTrialCPUV98(
        event, parent, plan, candidate, 1, boundary_owner, 1, 0, "")
    return CoupledAMRZeroTimeEventTrialCPUV98(
        **{**placeholder.__dict__,
           "trial_sha256": _hash(_trial_payload(placeholder))})


def validate_coupled_amr_zero_time_trial_cpu_v98(
    value: CoupledAMRZeroTimeEventTrialCPUV98,
    coarse_execution: CoupledWetDryCPUExecutionV97,
    fine_execution: CoupledWetDryCPUExecutionV97,
    interface: CoupledAMRInterfaceMapCPUV98,
    joined: CoupledAMRJoinedGeometryV98,
    coarse_patch: CartesianAMRPatchGeometryV98,
    fine_patch: CartesianAMRPatchGeometryV98,
) -> None:
    if (type(value) is not CoupledAMRZeroTimeEventTrialCPUV98 or
            value.event_kind not in ("refine", "coarsen") or
            value.trial_sha256 != _hash(_trial_payload(value)) or
            value.transfer_owner_count != 1 or
            value.hles_mapping_owner_count != 1 or
            value.candidate_publication_count != 0 or
            value.boundary_memory_remap_owner_count not in (0, 1)):
        raise CoupledAMRZeroTimeOwnerCPUV98Error(
            "v98 zero-time event trial identity differs")
    _sha(value.parent_state_sha256, "event parent state")
    validate_coupled_amr_zero_time_state_cpu_v98(
        value.candidate, coarse_execution, fine_execution, interface, joined,
        coarse_patch, fine_patch)
    if value.plan.plan_sha256 != value.candidate.plan.plan_sha256:
        raise CoupledAMRZeroTimeOwnerCPUV98Error(
            "v98 zero-time trial/candidate plan differs")


def prepare_coupled_amr_zero_time_refine_cpu_v98(
    coarse_execution: CoupledWetDryCPUExecutionV97,
    fine_execution: CoupledWetDryCPUExecutionV97,
    evolved_fine_state: CoupledWetDryCPUStateV97,
    interface: CoupledAMRInterfaceMapCPUV98,
    joined: CoupledAMRJoinedGeometryV98,
    coarse_patch: CartesianAMRPatchGeometryV98,
    fine_patch: CartesianAMRPatchGeometryV98, *,
    native_transfer_runtime: NativeAMRZeroTimeTransferCPUV98,
    native_threads: int = 4,
    cancelled: bool | Callable[[], bool] = False,
) -> CoupledAMRZeroTimeEventTrialCPUV98:
    poll = _poll(cancelled)
    if poll():
        raise InterruptedError("v98 zero-time refine cancelled before transfer")
    plan = _make_plan(
        coarse_execution, fine_execution, evolved_fine_state, interface,
        joined, coarse_patch, fine_patch, native_transfer_runtime,
        native_threads=native_threads, cancelled=poll)
    if poll():
        raise InterruptedError("v98 zero-time refine cancelled after transfer")
    candidate = _state(
        plan, layout="refined_children", refine_events=1,
        coarsen_events=0)
    trial = _trial(
        "refine", evolved_fine_state.state_sha256, plan, candidate,
        boundary_owner=1)
    validate_coupled_amr_zero_time_trial_cpu_v98(
        trial, coarse_execution, fine_execution, interface, joined,
        coarse_patch, fine_patch)
    return trial


def prepare_coupled_amr_zero_time_coarsen_cpu_v98(
    refined: CoupledAMRZeroTimeHierarchyStateCPUV98,
    coarse_execution: CoupledWetDryCPUExecutionV97,
    fine_execution: CoupledWetDryCPUExecutionV97,
    interface: CoupledAMRInterfaceMapCPUV98,
    joined: CoupledAMRJoinedGeometryV98,
    coarse_patch: CartesianAMRPatchGeometryV98,
    fine_patch: CartesianAMRPatchGeometryV98, *,
    cancelled: bool | Callable[[], bool] = False,
) -> CoupledAMRZeroTimeEventTrialCPUV98:
    poll = _poll(cancelled)
    if poll():
        raise InterruptedError("v98 zero-time coarsen cancelled before transfer")
    validate_coupled_amr_zero_time_state_cpu_v98(
        refined, coarse_execution, fine_execution, interface, joined,
        coarse_patch, fine_patch)
    if refined.active_layout != "refined_children":
        raise CoupledAMRZeroTimeOwnerCPUV98Error(
            "v98 zero-time coarsen requires the committed refined layout")
    candidate = _state(
        refined.plan, layout="coarsened_virtual_parent", refine_events=1,
        coarsen_events=1)
    if poll():
        raise InterruptedError("v98 zero-time coarsen cancelled after transfer")
    trial = _trial(
        "coarsen", refined.state_sha256, refined.plan, candidate,
        boundary_owner=0)
    validate_coupled_amr_zero_time_trial_cpu_v98(
        trial, coarse_execution, fine_execution, interface, joined,
        coarse_patch, fine_patch)
    return trial


def commit_coupled_amr_zero_time_event_cpu_v98(
    trial: CoupledAMRZeroTimeEventTrialCPUV98,
    coarse_execution: CoupledWetDryCPUExecutionV97,
    fine_execution: CoupledWetDryCPUExecutionV97,
    interface: CoupledAMRInterfaceMapCPUV98,
    joined: CoupledAMRJoinedGeometryV98,
    coarse_patch: CartesianAMRPatchGeometryV98,
    fine_patch: CartesianAMRPatchGeometryV98, *,
    cancelled: bool | Callable[[], bool] = False,
) -> CoupledAMRZeroTimeHierarchyStateCPUV98:
    poll = _poll(cancelled)
    if poll():
        raise InterruptedError("v98 zero-time event cancelled before publication")
    validate_coupled_amr_zero_time_trial_cpu_v98(
        trial, coarse_execution, fine_execution, interface, joined,
        coarse_patch, fine_patch)
    if poll():
        raise InterruptedError("v98 zero-time event cancelled after validation")
    # Immutable candidate publication has no external provider or clock side
    # effect.  A failed/cancelled call therefore leaves all inputs byte-exact.
    return trial.candidate


__all__ = [
    "SCHEMA", "SCOPE_ID", "SEMANTIC_ID",
    "CoupledAMRZeroTimeOwnerCPUV98Error",
    "CoupledAMRRootBoundaryMemoryCPUV98",
    "CoupledAMRZeroTimeHlesMapCPUV98",
    "CoupledAMRZeroTimeActiveImageCPUV98",
    "CoupledAMRZeroTimePlanCPUV98",
    "CoupledAMRZeroTimeHierarchyStateCPUV98",
    "CoupledAMRZeroTimeEventTrialCPUV98",
    "prepare_coupled_amr_zero_time_refine_cpu_v98",
    "prepare_coupled_amr_zero_time_coarsen_cpu_v98",
    "commit_coupled_amr_zero_time_event_cpu_v98",
    "validate_coupled_amr_zero_time_active_image_cpu_v98",
    "validate_coupled_amr_zero_time_plan_cpu_v98",
    "validate_coupled_amr_zero_time_state_cpu_v98",
    "validate_coupled_amr_zero_time_trial_cpu_v98",
]
