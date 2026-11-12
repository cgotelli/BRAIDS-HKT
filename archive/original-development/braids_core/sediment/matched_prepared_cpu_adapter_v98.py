"""Exact prepared-record adapter for the bounded matched CPU/GPU replay.

This module does not invent a second case generator.  It copies and seals the
arrays exported by the canonical 128-cell prepared record, then constructs the
existing CPU v76--v97 owners on that exact topology and accepted state.  The
first admitted execution is deliberately closed-boundary and limited to short
validation updates.  The hydraulic reconstruction/least-squares arrays are
authenticated even though the current native CPU v97 midpoint is the bounded
piecewise-constant reference and therefore does not consume them numerically.

The adapter is validation infrastructure, not product/plugin admission.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from .active_suspended_control_v94 import (
    CPUQ3V83ActiveExnerAdapterV94,
)
from .active_suspended_control_v94b import (
    resolve_active_suspended_control_execution_v94b,
    resolve_initial_active_suspended_control_state_v94b,
)
from .bed_slope_execution import (
    UPSTREAM_STAGES,
    bed_slope_execution_document,
)
from .bedload_boundary_live_v70 import (
    MODE_CLOSED,
    build_bedload_boundary_execution_v70,
    resolve_bedload_boundary_topology_v70,
)
from .channel_composition import (
    TRA_COMBINE,
    resolve_channel_composition_profile,
)
from .coupled_geometry_native_cpu_v96 import (
    resolve_coupled_geometry_native_cpu_execution_v96,
)
from .coupled_wetdry_owner_cpu_v97 import (
    resolve_coupled_wetdry_cpu_execution_v97,
    resolve_coupled_wetdry_cpu_state_v97,
)
from .densin_fwave_v85 import (
    mixture_density_kg_m3_v85,
    resolve_densin_execution_v85,
)
from .direct_vector_family_live_v67 import (
    METHOD_VAN_RIJN_1993_CURRENT,
    direct_vector_family_execution_document_v67,
)
from .direct_vector_live_taurat_v73 import (
    direct_vector_live_taurat_execution_document_v73,
)
from .direct_vector_wetdry_amr_live_v71 import (
    FACE_CLOSED,
    FACE_INTERNAL,
    FACE_TRACE_INLET,
    wetdry_topology_v71,
)
from .factcr_lifecycle_v83 import resolve_factcr_execution_v83
from .hles_producer_v90 import (
    BACKEND_CPU as HLES_BACKEND_CPU,
    _wrap_state as _wrap_hles_state,
    make_current_closure_handoff_v90,
    resolve_hles_producer_execution_v90,
)
from .horizontal_turbulence_lifecycle_v87 import (
    resolve_horizontal_turbulence_execution_v87,
    spatial_turbulence_background_v87,
)
from .multifraction_suspended_lifecycle_v78 import (
    ROLE_CPU_ORACLE,
    resolve_multifraction_suspended_execution_v78,
    resolve_multifraction_suspended_state_v78,
)
from .q3_mixed_route_exner_adapter_v93 import (
    resolve_cpu_q3_v84_mixed_binding_v93,
    resolve_cpu_q3_v84_mixed_state_v93,
)
from .q3_wetdry_adapter_cpu_v97 import (
    CPUQ3V83ActiveWetDryExnerAdapterV97,
)
from .route_lifecycle_composition_v81 import ROUTE_COMBINE
from .shared_turbulence_consumers_v91 import (
    resolve_shared_turbulence_state_v91,
    resolve_shared_turbulence_wetdry_execution_v97,
)
from .suspended_exner_owner_v93 import (
    resolve_suspended_exner_execution_v93,
    resolve_suspended_exner_zero_vector_execution_v93,
)
from .suspended_lifecycle_live_v76 import (
    BACKEND_CPU as V76_BACKEND_CPU,
    resolve_suspended_lifecycle_execution_v76,
)
from .suspension_profile import Flow2DHSuspensionProfile
from .transported_fluid_settling_v92 import (
    MODE_TRANSPORTED,
    resolve_transported_fluid_execution_v92,
    resolve_transported_fluid_state_v92,
)
from .wetdry_fwave_native_cpu_v97 import (
    resolve_wetdry_fwave_cpu_execution_v97,
)


SCHEMA = 2
SCOPE_ID = "braids-q4-matched-prepared-cpu-adapter-v98-v2"
SEMANTIC_ID = SCOPE_ID + ":canonical-record-to-closed-coupled-v97"
DT_S = 5.0e-4
IOPSUS = (1, 0, 1)
ROUTES = (ROUTE_COMBINE, ROUTE_COMBINE, ROUTE_COMBINE)
FACTCR = 1.2
CONTROL_ET = "ET"
CONTROL_EF = "EF"
CONTROL_DT = "DT"
CONTROL_DF = "DF"
CONTROL_IDS = (CONTROL_ET, CONTROL_EF, CONTROL_DT, CONTROL_DF)
LIFECYCLE_HISTORICAL_V94 = "historical_v94"
LIFECYCLE_EXACT_ZERO_V94B = "exact_zero_v94b"
BED_OWNER_EXCHANGE = "exchange"
BED_OWNER_CONSTRUCTED_DIVERGENCE = \
    "constructed_suspended_vector_divergence_plus_q3_once"
BED_OWNER_EXACT_ZERO = "exact_zero_suspended_vector_plus_q3_once"
CONTROLS = dict(
    alfabs=0.74, alfabn=1.43, ashld=0.91, bshld=0.63,
    cshld=0.27, dshld=-0.34, alfpa=0.37, thcrpa=0.052,
)


class MatchedPreparedCPUAdapterV98Error(ValueError):
    """The canonical input, receipt, or constructed CPU case is invalid."""


@dataclass(frozen=True)
class MatchedPreparedControlV99:
    """Authenticated source-legal IopSus/SusCor/Neglect selection.

    This is deliberately separate from the canonical prepared numerical
    record.  The same accepted q/T/S/topology image is replayed for every
    control row; only this typed receipt and the resolved v94/v94b execution
    select the suspended-bed owner.
    """

    iopsus_by_fraction: np.ndarray
    suscor: bool
    neglect_entrainment: bool
    control_id: str
    selected_bed_owner: str
    lifecycle_route: str
    receipt_sha256: str


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError, OverflowError) as error:
        raise MatchedPreparedCPUAdapterV98Error(
            "v98 matched prepared value is not canonical JSON") from error


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _digest(*values: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(value)
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(_canonical(list(array.shape)))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _control_payload(value: MatchedPreparedControlV99) -> dict[str, object]:
    return {
        "scope_id": SCOPE_ID + ":legal-control-v99",
        "iopsus_by_fraction": value.iopsus_by_fraction.tolist(),
        "suscor": value.suscor,
        "neglect_entrainment": value.neglect_entrainment,
        "control_id": value.control_id,
        "selected_bed_owner": value.selected_bed_owner,
        "lifecycle_route": value.lifecycle_route,
        "source_legal": True,
        "accepted_update_limit": 4,
    }


def resolve_matched_prepared_control_v99(
    *, iopsus_by_fraction: object = IOPSUS,
    suscor: bool = False,
    neglect_entrainment: bool = True,
) -> MatchedPreparedControlV99:
    raw = np.asarray(iopsus_by_fraction)
    if (raw.shape != (3,) or raw.dtype.kind not in "iub" or
            np.any(~np.isin(raw, (0, 1)))):
        raise MatchedPreparedCPUAdapterV98Error(
            "v99 matched IopSus profile must contain three values in {0,1}")
    if type(suscor) is not bool or type(neglect_entrainment) is not bool:
        raise MatchedPreparedCPUAdapterV98Error(
            "v99 matched SusCor and NeglectEntrainment must be Boolean")
    profile = np.ascontiguousarray(raw, dtype=np.int8)
    profile.setflags(write=False)
    if not neglect_entrainment:
        control_id = CONTROL_ET if suscor else CONTROL_EF
        owner = BED_OWNER_EXCHANGE
        route = LIFECYCLE_HISTORICAL_V94
    elif suscor:
        control_id = CONTROL_DT
        owner = BED_OWNER_CONSTRUCTED_DIVERGENCE
        route = LIFECYCLE_HISTORICAL_V94
    else:
        control_id = CONTROL_DF
        owner = BED_OWNER_EXACT_ZERO
        route = LIFECYCLE_EXACT_ZERO_V94B
    placeholder = MatchedPreparedControlV99(
        profile, suscor, neglect_entrainment, control_id, owner, route, "")
    return MatchedPreparedControlV99(
        **{**placeholder.__dict__,
           "receipt_sha256": _hash(_control_payload(placeholder))})


def validate_matched_prepared_control_v99(
    value: MatchedPreparedControlV99,
) -> None:
    if type(value) is not MatchedPreparedControlV99:
        raise MatchedPreparedCPUAdapterV98Error(
            "v99 matched control receipt type differs")
    if (type(value.iopsus_by_fraction) is not np.ndarray or
            value.iopsus_by_fraction.dtype != np.dtype(np.int8) or
            value.iopsus_by_fraction.shape != (3,) or
            value.iopsus_by_fraction.flags.writeable):
        raise MatchedPreparedCPUAdapterV98Error(
            "v99 matched control profile storage differs")
    rebuilt = resolve_matched_prepared_control_v99(
        iopsus_by_fraction=value.iopsus_by_fraction,
        suscor=value.suscor,
        neglect_entrainment=value.neglect_entrainment)
    if (value.receipt_sha256 != _hash(_control_payload(value)) or
            not np.array_equal(value.iopsus_by_fraction,
                               rebuilt.iopsus_by_fraction) or
            value.suscor != rebuilt.suscor or
            value.neglect_entrainment != rebuilt.neglect_entrainment or
            value.control_id != rebuilt.control_id or
            value.selected_bed_owner != rebuilt.selected_bed_owner or
            value.lifecycle_route != rebuilt.lifecycle_route or
            value.receipt_sha256 != rebuilt.receipt_sha256):
        raise MatchedPreparedCPUAdapterV98Error(
            "v99 matched control receipt differs")


def _array(value: object, shape: tuple[int, ...], name: str, *,
           dtype=np.float64, nonnegative: bool = False,
           positive: bool = False) -> np.ndarray:
    raw = np.asarray(value)
    expected = np.dtype(dtype)
    if raw.shape != shape:
        raise MatchedPreparedCPUAdapterV98Error(
            f"v98 matched {name} shape differs: {raw.shape} != {shape}")
    if expected.kind in "iu":
        if raw.dtype.kind not in "iu":
            raise MatchedPreparedCPUAdapterV98Error(
                f"v98 matched {name} must be integer")
    elif raw.dtype.kind not in "fiu" or not np.isfinite(raw).all():
        raise MatchedPreparedCPUAdapterV98Error(
            f"v98 matched {name} must be finite numeric data")
    result = np.ascontiguousarray(raw, dtype=expected)
    if nonnegative and np.any(result < 0):
        raise MatchedPreparedCPUAdapterV98Error(
            f"v98 matched {name} must be nonnegative")
    if positive and np.any(result <= 0):
        raise MatchedPreparedCPUAdapterV98Error(
            f"v98 matched {name} must be positive")
    result.setflags(write=False)
    return result


def _record_array(record: Mapping[str, object], key: str,
                  shape: tuple[int, ...], *, dtype=np.float64,
                  nonnegative: bool = False,
                  positive: bool = False) -> np.ndarray:
    if key not in record:
        raise MatchedPreparedCPUAdapterV98Error(
            f"v98 matched prepared record lacks {key}")
    return _array(record[key], shape, key, dtype=dtype,
                  nonnegative=nonnegative, positive=positive)


@dataclass(frozen=True)
class MatchedPreparedBoundaryClosureReceiptV98:
    """Authenticated translation from the live source BC to a closed replay.

    ``source_controller_boundary_mode`` is the face-valued mode image consumed
    by reconstruction/flux code.  ``source_boundary_owner_mode`` remains the
    boundary-valued mode image consumed by the physical trace owner.  Requiring
    those two images to agree on every ``bid >= 0`` face prevents a zeroed
    controller array from masquerading as a closed hydraulic boundary while a
    mode-9/mode-4 trace owner is still live.
    """

    source_controller_boundary_mode: np.ndarray
    source_topological_open_mask: np.ndarray
    source_reflective_wall_mask: np.ndarray
    target_reflective_boundary_mask: np.ndarray
    source_physical_face_count: int
    source_reflective_face_count: int
    target_reflective_face_count: int
    source_physical_boundary_owner_count: int
    source_physical_trace_owner_count: int
    source_boundary_pressure_owner_count: int
    source_boundary_ledger_owner_count: int
    target_physical_boundary_owner_count: int
    target_physical_trace_owner_count: int
    target_boundary_pressure_owner_count: int
    target_boundary_ledger_owner_count: int
    target_reflective_boundary_owner_count: int
    topological_open_to_reflective_face_count: int
    boundary_closure_transformation_count: int
    controller_mode_zero_closure_evidence_count: int
    receipt_sha256: str


def _boundary_closure_payload(
    value: MatchedPreparedBoundaryClosureReceiptV98,
) -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID + ":boundary-closure",
        "semantic_id": SEMANTIC_ID + ":explicit-source-to-reflective",
        "source_controller_boundary_mode_sha256":
            _digest(value.source_controller_boundary_mode),
        "source_topological_open_mask_sha256":
            _digest(value.source_topological_open_mask),
        "source_reflective_wall_mask_sha256":
            _digest(value.source_reflective_wall_mask),
        "target_reflective_boundary_mask_sha256":
            _digest(value.target_reflective_boundary_mask),
        "source_physical_face_count": value.source_physical_face_count,
        "source_reflective_face_count": value.source_reflective_face_count,
        "target_reflective_face_count": value.target_reflective_face_count,
        "source_physical_boundary_owner_count":
            value.source_physical_boundary_owner_count,
        "source_physical_trace_owner_count":
            value.source_physical_trace_owner_count,
        "source_boundary_pressure_owner_count":
            value.source_boundary_pressure_owner_count,
        "source_boundary_ledger_owner_count":
            value.source_boundary_ledger_owner_count,
        "target_physical_boundary_owner_count":
            value.target_physical_boundary_owner_count,
        "target_physical_trace_owner_count":
            value.target_physical_trace_owner_count,
        "target_boundary_pressure_owner_count":
            value.target_boundary_pressure_owner_count,
        "target_boundary_ledger_owner_count":
            value.target_boundary_ledger_owner_count,
        "target_reflective_boundary_owner_count":
            value.target_reflective_boundary_owner_count,
        "topological_open_to_reflective_face_count":
            value.topological_open_to_reflective_face_count,
        "boundary_closure_transformation_count":
            value.boundary_closure_transformation_count,
        "controller_mode_zero_closure_evidence_count":
            value.controller_mode_zero_closure_evidence_count,
        "source_physical_kind": 3,
        "target_reflective_kind": 2,
        "physical_boundary_flux_consumer_in_target": False,
    }


def _resolve_boundary_closure_receipt(
    *, neighbour: np.ndarray, boundary_index: np.ndarray,
    source_controller_boundary_mode: np.ndarray,
    source_boundary_owner_mode: np.ndarray,
    boundary_offset: np.ndarray, boundary_faces: np.ndarray,
) -> MatchedPreparedBoundaryClosureReceiptV98:
    exterior = neighbour < 0
    physical = exterior & (boundary_index >= 0)
    wall = exterior & (boundary_index < 0)
    nb = int(source_boundary_owner_mode.size)
    if (boundary_offset[0] != 0 or boundary_offset[-1] != boundary_faces.size or
            np.any(np.diff(boundary_offset) < 0) or
            np.any(boundary_faces < 0) or
            np.any(boundary_faces >= neighbour.size) or
            np.unique(boundary_faces).size != boundary_faces.size):
        raise MatchedPreparedCPUAdapterV98Error(
            "v98 matched physical-boundary CSR is invalid")
    if (np.any(boundary_index[physical] >= nb) or
            not np.array_equal(np.sort(boundary_faces),
                               np.flatnonzero(physical))):
        raise MatchedPreparedCPUAdapterV98Error(
            "v98 matched physical-boundary CSR and topology differ")
    for boundary in range(nb):
        first, last = (int(boundary_offset[boundary]),
                       int(boundary_offset[boundary+1]))
        selected = boundary_faces[first:last]
        if (selected.size == 0 or
                np.any(boundary_index[selected] != boundary)):
            raise MatchedPreparedCPUAdapterV98Error(
                "v98 matched physical-boundary CSR ownership differs")
    expected_controller = source_boundary_owner_mode[boundary_index[physical]]
    if (not np.array_equal(source_controller_boundary_mode[physical],
                           expected_controller) or
            np.any(source_controller_boundary_mode[wall] != 0)):
        raise MatchedPreparedCPUAdapterV98Error(
            "v98 matched controller boundary modes differ from the live "
            "physical-boundary owner")
    if not np.any(physical):
        raise MatchedPreparedCPUAdapterV98Error(
            "v98 matched canonical source has no physical boundary faces")
    arrays = []
    for source in (source_controller_boundary_mode, physical, wall, exterior):
        item = np.ascontiguousarray(source)
        item.setflags(write=False)
        arrays.append(item)
    placeholder = MatchedPreparedBoundaryClosureReceiptV98(
        arrays[0], arrays[1], arrays[2], arrays[3],
        int(np.count_nonzero(physical)), int(np.count_nonzero(wall)),
        int(np.count_nonzero(exterior)),
        1, 1, 1, 1,
        0, 0, 0, 0, 1,
        int(np.count_nonzero(physical)), 1, 0, "")
    return MatchedPreparedBoundaryClosureReceiptV98(
        **{**placeholder.__dict__,
           "receipt_sha256": _hash(_boundary_closure_payload(placeholder))})


def _validate_boundary_closure_receipt(
    value: MatchedPreparedBoundaryClosureReceiptV98, *,
    neighbour: np.ndarray, boundary_index: np.ndarray,
    source_boundary_owner_mode: np.ndarray,
    boundary_offset: np.ndarray, boundary_faces: np.ndarray,
) -> None:
    if type(value) is not MatchedPreparedBoundaryClosureReceiptV98:
        raise MatchedPreparedCPUAdapterV98Error(
            "v98 matched boundary closure receipt type differs")
    expected = _resolve_boundary_closure_receipt(
        neighbour=neighbour, boundary_index=boundary_index,
        source_controller_boundary_mode=
            value.source_controller_boundary_mode,
        source_boundary_owner_mode=source_boundary_owner_mode,
        boundary_offset=boundary_offset, boundary_faces=boundary_faces)
    if (value.receipt_sha256 != _hash(_boundary_closure_payload(value)) or
            _boundary_closure_payload(value) !=
            _boundary_closure_payload(expected) or
            not np.array_equal(value.source_controller_boundary_mode,
                               expected.source_controller_boundary_mode) or
            not np.array_equal(value.source_topological_open_mask,
                               expected.source_topological_open_mask) or
            not np.array_equal(value.source_reflective_wall_mask,
                               expected.source_reflective_wall_mask) or
            not np.array_equal(value.target_reflective_boundary_mask,
                               expected.target_reflective_boundary_mask)):
        raise MatchedPreparedCPUAdapterV98Error(
            "v98 matched boundary closure receipt differs")


@dataclass(frozen=True)
class MatchedPreparedCPUInputV98:
    profile_fingerprint: str
    dry_tolerance_m: float
    q: np.ndarray
    temperature_cell_c: np.ndarray
    salinity_cell_ppt: np.ndarray
    h_temperature_c_m: np.ndarray
    h_salinity_ppt_m: np.ndarray
    suspended_mass_kg_m2: np.ndarray
    bed_mass_kg_m2: np.ndarray
    bed_elevation_m: np.ndarray
    accepted_water_density_kg_m3: np.ndarray
    accepted_mixture_density_kg_m3: np.ndarray
    hles_filtered_face_normal_velocity_m_s: np.ndarray
    hles_dynamic_vicuv_m2_s: np.ndarray
    hles_dynamic_dicuv_m2_s: np.ndarray
    owner: np.ndarray
    neighbour: np.ndarray
    boundary_index: np.ndarray
    cell_face_offset: np.ndarray
    cell_faces: np.ndarray
    cell_face_sign: np.ndarray
    normal: np.ndarray
    face_length_m: np.ndarray
    raw_face_distance_m: np.ndarray
    centre_distance_m: np.ndarray
    cell_area_m2: np.ndarray
    rectangle_width_m: np.ndarray
    weno_cells: np.ndarray
    hydraulic_ls_matrix: np.ndarray
    hydraulic_ls_weight: np.ndarray
    bed_ls_matrix: np.ndarray
    bed_ls_delta: np.ndarray
    bed_ls_weight: np.ndarray
    boundary_mode: np.ndarray
    boundary_offset: np.ndarray
    boundary_faces: np.ndarray
    boundary_forcing: np.ndarray
    boundary_bed_offset_m: np.ndarray
    boundary_closure: MatchedPreparedBoundaryClosureReceiptV98
    source_topology_sha256: str
    source_reconstruction_sha256: str
    source_state_sha256: str
    source_boundary_sha256: str
    receipt_sha256: str

    @property
    def cell_count(self) -> int:
        return int(self.q.shape[1])

    @property
    def face_count(self) -> int:
        return int(self.owner.size)

    @property
    def boundary_count(self) -> int:
        return int(self.boundary_mode.size)

    @property
    def fraction_count(self) -> int:
        return int(self.suspended_mass_kg_m2.shape[0])


def _input_payload(value: MatchedPreparedCPUInputV98) -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID + ":input",
        "semantic_id": SEMANTIC_ID,
        "profile_fingerprint": value.profile_fingerprint,
        "dry_tolerance_m": value.dry_tolerance_m,
        "counts": [value.cell_count, value.face_count,
                   value.boundary_count, value.fraction_count],
        "source_topology_sha256": value.source_topology_sha256,
        "source_reconstruction_sha256":
            value.source_reconstruction_sha256,
        "source_state_sha256": value.source_state_sha256,
        "source_boundary_sha256": value.source_boundary_sha256,
        "boundary_closure_receipt_sha256":
            value.boundary_closure.receipt_sha256,
        "q_ts_unmodified": True,
        "topology_weno_csr_ls_unmodified": True,
        "cpu_closed_boundary_for_first_replay": True,
        "cpu_hydraulic_reconstruction_consumer_count": 0,
        "cpu_hydraulic_reconstruction_scope":
            "bounded_piecewise_constant_native_v97_reference",
        "dt_s": DT_S,
        "control_neutral_prepared_record": True,
        "supported_iopsus_values_per_fraction": [0, 1],
        "supported_controls": list(CONTROL_IDS),
        "routes": list(ROUTES),
        "factcr": FACTCR,
        "arrays_sha256": _digest(
            value.q, value.temperature_cell_c, value.salinity_cell_ppt,
            value.h_temperature_c_m, value.h_salinity_ppt_m,
            value.suspended_mass_kg_m2, value.bed_mass_kg_m2,
            value.bed_elevation_m, value.accepted_water_density_kg_m3,
            value.accepted_mixture_density_kg_m3,
            value.hles_filtered_face_normal_velocity_m_s,
            value.hles_dynamic_vicuv_m2_s,
            value.hles_dynamic_dicuv_m2_s, value.owner, value.neighbour,
            value.boundary_index, value.cell_face_offset, value.cell_faces,
            value.cell_face_sign, value.normal, value.face_length_m,
            value.raw_face_distance_m, value.centre_distance_m,
            value.cell_area_m2, value.rectangle_width_m, value.weno_cells,
            value.hydraulic_ls_matrix, value.hydraulic_ls_weight,
            value.bed_ls_matrix, value.bed_ls_delta, value.bed_ls_weight,
            value.boundary_mode, value.boundary_offset,
            value.boundary_faces, value.boundary_forcing,
            value.boundary_bed_offset_m,
            value.boundary_closure.source_controller_boundary_mode,
            value.boundary_closure.source_topological_open_mask,
            value.boundary_closure.target_reflective_boundary_mask),
    }


def resolve_matched_prepared_cpu_input_v98(
    record: Mapping[str, object], *,
    profile: Flow2DHSuspensionProfile,
    q: object,
    temperature_cell_c: object,
    salinity_cell_ppt: object,
    h_temperature_c_m: object,
    h_salinity_ppt_m: object,
    suspended_mass_kg_m2: object,
    bed_mass_kg_m2: object,
    bed_elevation_m: object,
    accepted_water_density_kg_m3: object,
    accepted_mixture_density_kg_m3: object,
    hles_filtered_face_normal_velocity_m_s: object,
    hles_dynamic_vicuv_m2_s: object,
    hles_dynamic_dicuv_m2_s: object,
) -> MatchedPreparedCPUInputV98:
    """Seal a canonical prepared record without regenerating q or T/S."""
    if not isinstance(record, Mapping):
        raise MatchedPreparedCPUAdapterV98Error(
            "v98 matched prepared record must be a mapping")
    nc, nf, nb, ng = (
        int(record.get("nc", -1)), int(record.get("nf", -1)),
        int(record.get("nb", -1)), len(profile.sediment_d50_m))
    # In addition to the frozen 128-cell matrix, admit the single virtual
    # parent used by the bounded coarsen/next-solver gate. This is not an
    # arbitrary-grid or product admission.
    if (nc not in (1, 128) or nf <= 0 or nb <= 0 or ng != 3 or
            profile.fingerprint == ""):
        raise MatchedPreparedCPUAdapterV98Error(
            "v98 matched canonical counts/profile differ")
    dry_tolerance_m = float(record.get("dry", np.nan))
    if (not np.isfinite(dry_tolerance_m) or dry_tolerance_m <= 0.0):
        raise MatchedPreparedCPUAdapterV98Error(
            "v98 matched prepared dry tolerance is invalid")
    q_a = _array(q, (3, nc), "accepted q")
    temperature = _array(
        temperature_cell_c, (nc,), "accepted temperature")
    salinity = _array(
        salinity_cell_ppt, (nc,), "accepted salinity", nonnegative=True)
    ht = _array(h_temperature_c_m, (nc,), "accepted hT")
    hs = _array(h_salinity_ppt_m, (nc,), "accepted hS",
                nonnegative=True)
    if (not np.array_equal(ht, q_a[0]*temperature) or
            not np.array_equal(hs, q_a[0]*salinity)):
        raise MatchedPreparedCPUAdapterV98Error(
            "v98 matched transported T/S integrals differ from q and T/S")
    suspended = _array(
        suspended_mass_kg_m2, (ng, nc), "suspended mass",
        nonnegative=True)
    bed_mass = _array(
        bed_mass_kg_m2, (ng, nc), "bed mass", nonnegative=True)
    bed = _array(bed_elevation_m, (nc,), "bed elevation")
    water = _array(
        accepted_water_density_kg_m3, (nc,), "accepted water density",
        positive=True)
    mixture = _array(
        accepted_mixture_density_kg_m3, (nc,),
        "accepted mixture density", positive=True)
    if np.any(mixture < water):
        raise MatchedPreparedCPUAdapterV98Error(
            "v98 matched mixture density is below water density")
    owner = _record_array(record, "owner", (nf,), dtype=np.int32)
    neighbour = _record_array(
        record, "neighbour", (nf,), dtype=np.int32)
    boundary_index = _record_array(
        record, "bid", (nf,), dtype=np.int32)
    offset = _record_array(
        record, "offset", (nc+1,), dtype=np.int32)
    faces = _record_array(
        record, "faces", (int(offset[-1]),), dtype=np.int32)
    signs = _record_array(
        record, "signs", (int(offset[-1]),), dtype=np.int32)
    normal = _record_array(record, "normal", (2, nf))
    length = _record_array(
        record, "length", (nf,), positive=True)
    raw_distance = _record_array(
        record, "distance", (2, nf), nonnegative=True)
    distance = np.ascontiguousarray(np.where(
        neighbour >= 0, raw_distance[0]+raw_distance[1],
        raw_distance[0]))
    if np.any(distance <= 0.0):
        raise MatchedPreparedCPUAdapterV98Error(
            "v98 matched centre distance must be positive")
    distance.setflags(write=False)
    area = _record_array(record, "area", (nc,), positive=True)
    rectangle = _record_array(record, "rectangle_width", (2, nc))
    weno = _record_array(record, "weno_cells", (5, 2, nf),
                         dtype=np.int32)
    hls_matrix = _record_array(
        record, "hydraulic_ls_matrix", (4, nc))
    hls_weight = _record_array(
        record, "hydraulic_ls_weight",
        (int(np.asarray(record["hydraulic_ls_weight"]).size),))
    bls_matrix = _record_array(record, "bed_ls_matrix", (4, nc))
    bls_delta = _record_array(
        record, "bed_ls_delta",
        tuple(np.asarray(record["bed_ls_delta"]).shape))
    bls_weight = _record_array(
        record, "bed_ls_weight",
        (int(np.asarray(record["bed_ls_weight"]).size),))
    boundary_mode = _record_array(
        record, "mode", (nb,), dtype=np.int32)
    controller_boundary_mode = _record_array(
        record, "boundary_mode", (nf,), dtype=np.int32)
    boundary_offset = _record_array(
        record, "boundary_offset", (nb+1,), dtype=np.int32)
    boundary_faces = _record_array(
        record, "boundary_faces", (int(boundary_offset[-1]),),
        dtype=np.int32)
    forcing = _record_array(record, "forcing", (6, nb))
    bed_offset = _record_array(
        record, "boundary_bed_offset", (nf,))
    boundary_closure = _resolve_boundary_closure_receipt(
        neighbour=neighbour, boundary_index=boundary_index,
        source_controller_boundary_mode=controller_boundary_mode,
        source_boundary_owner_mode=boundary_mode,
        boundary_offset=boundary_offset, boundary_faces=boundary_faces)
    filtered = _array(
        hles_filtered_face_normal_velocity_m_s, (nf,),
        "accepted HLES filtered velocity")
    dynamic_v = _array(
        hles_dynamic_vicuv_m2_s, (nc,), "accepted HLES vicuv",
        nonnegative=True)
    dynamic_d = _array(
        hles_dynamic_dicuv_m2_s, (nc,), "accepted HLES dicuv",
        nonnegative=True)
    topology_sha = _digest(
        owner, neighbour, boundary_index, offset, faces, signs,
        normal, length, raw_distance, distance, area)
    reconstruction_sha = _digest(
        rectangle, weno, hls_matrix, hls_weight,
        bls_matrix, bls_delta, bls_weight)
    state_sha = _digest(
        q_a, temperature, salinity, ht, hs, suspended, bed_mass, bed,
        water, mixture, filtered, dynamic_v, dynamic_d)
    boundary_sha = _digest(
        boundary_mode, controller_boundary_mode, boundary_offset,
        boundary_faces, forcing, bed_offset)
    placeholder = MatchedPreparedCPUInputV98(
        profile_fingerprint=profile.fingerprint,
        dry_tolerance_m=dry_tolerance_m,
        q=q_a, temperature_cell_c=temperature,
        salinity_cell_ppt=salinity, h_temperature_c_m=ht,
        h_salinity_ppt_m=hs, suspended_mass_kg_m2=suspended,
        bed_mass_kg_m2=bed_mass, bed_elevation_m=bed,
        accepted_water_density_kg_m3=water,
        accepted_mixture_density_kg_m3=mixture,
        hles_filtered_face_normal_velocity_m_s=filtered,
        hles_dynamic_vicuv_m2_s=dynamic_v,
        hles_dynamic_dicuv_m2_s=dynamic_d,
        owner=owner, neighbour=neighbour, boundary_index=boundary_index,
        cell_face_offset=offset, cell_faces=faces, cell_face_sign=signs,
        normal=normal, face_length_m=length,
        raw_face_distance_m=raw_distance, centre_distance_m=distance,
        cell_area_m2=area, rectangle_width_m=rectangle,
        weno_cells=weno, hydraulic_ls_matrix=hls_matrix,
        hydraulic_ls_weight=hls_weight, bed_ls_matrix=bls_matrix,
        bed_ls_delta=bls_delta, bed_ls_weight=bls_weight,
        boundary_mode=boundary_mode, boundary_offset=boundary_offset,
        boundary_faces=boundary_faces, boundary_forcing=forcing,
        boundary_bed_offset_m=bed_offset,
        boundary_closure=boundary_closure,
        source_topology_sha256=topology_sha,
        source_reconstruction_sha256=reconstruction_sha,
        source_state_sha256=state_sha,
        source_boundary_sha256=boundary_sha, receipt_sha256="")
    return MatchedPreparedCPUInputV98(
        **{**placeholder.__dict__,
           "receipt_sha256": _hash(_input_payload(placeholder))})


def validate_matched_prepared_cpu_input_v98(
    value: MatchedPreparedCPUInputV98,
) -> None:
    if type(value) is not MatchedPreparedCPUInputV98:
        raise MatchedPreparedCPUAdapterV98Error(
            "v98 matched prepared input type differs")
    _validate_boundary_closure_receipt(
        value.boundary_closure, neighbour=value.neighbour,
        boundary_index=value.boundary_index,
        source_boundary_owner_mode=value.boundary_mode,
        boundary_offset=value.boundary_offset,
        boundary_faces=value.boundary_faces)
    if value.receipt_sha256 != _hash(_input_payload(value)):
        raise MatchedPreparedCPUAdapterV98Error(
            "v98 matched prepared input receipt differs")


@dataclass(frozen=True)
class MatchedPreparedClosedCPUCaseV98:
    prepared_input: MatchedPreparedCPUInputV98
    control: MatchedPreparedControlV99
    execution: object
    state: object
    q3_provider: object
    q3_inputs: Mapping[str, np.ndarray]
    runtimes: Mapping[str, object]
    construction_sha256: str


def _case_token(value: MatchedPreparedClosedCPUCaseV98) -> str:
    def runtime_id(runtime: object) -> str | None:
        direct = getattr(runtime, "runtime_sha256", None)
        if isinstance(direct, str):
            return direct
        receipt = getattr(runtime, "receipt", None)
        if isinstance(receipt, Mapping):
            candidate = receipt.get("receipt_sha256")
            return candidate if isinstance(candidate, str) else None
        for name in ("receipt_sha256", "build_sha256", "library_sha256"):
            candidate = getattr(receipt, name, None)
            if isinstance(candidate, str):
                return candidate
        return None
    runtime_ids = {
        name: runtime_id(runtime)
        for name, runtime in value.runtimes.items()
    }
    return _hash({
        "scope_id": SCOPE_ID + ":closed-case",
        "input": value.prepared_input.receipt_sha256,
        "control": value.control.receipt_sha256,
        "execution": value.execution.fingerprint,
        "state": value.state.state_sha256,
        "provider": value.q3_provider.checkpoint(),
        "q3_inputs": {name: _digest(np.asarray(array))
                      for name, array in value.q3_inputs.items()},
        "runtimes": runtime_ids,
        "dt_s": DT_S,
        "boundary_closure":
            value.prepared_input.boundary_closure.receipt_sha256,
        "source_physical_boundary_owner_count": 1,
        "boundary_closure_transformation_count": 1,
        "physical_boundary_owner_count": 0,
        "physical_boundary_trace_owner_count": 0,
        "boundary_pressure_owner_count": 0,
        "boundary_ledger_owner_count": 0,
        "reflective_boundary_owner_count": 1,
        "accepted_update_limit": 4,
    })


def build_matched_prepared_closed_cpu_case_v98(
    prepared: MatchedPreparedCPUInputV98, *,
    profile: Flow2DHSuspensionProfile,
    repository_root: Path,
    geometry_runtime: object,
    wetdry_runtime: object,
    density_projection_runtime: object,
    turbulence_runtime: object,
    scalar_fct_runtime: object,
    scalar_diffusion_runtime: object,
    scalar_boundary_runtime: object,
    hles_runtime: object,
    settling_exchange_runtime: object,
    factcr_runtime: object,
    q3_face_runtime: object,
    native_threads: int = 4,
    iopsus_by_fraction: object = IOPSUS,
    suscor: bool = False,
    neglect_entrainment: bool = True,
) -> MatchedPreparedClosedCPUCaseV98:
    """Construct one genuine closed CPU v97 owner from sealed input arrays."""
    validate_matched_prepared_cpu_input_v98(prepared)
    control = resolve_matched_prepared_control_v99(
        iopsus_by_fraction=iopsus_by_fraction, suscor=suscor,
        neglect_entrainment=neglect_entrainment)
    closure = prepared.boundary_closure
    if (closure.boundary_closure_transformation_count != 1 or
            closure.source_physical_boundary_owner_count != 1 or
            closure.target_physical_boundary_owner_count != 0 or
            closure.target_reflective_boundary_owner_count != 1):
        raise MatchedPreparedCPUAdapterV98Error(
            "v98 matched explicit boundary closure ownership differs")
    if profile.fingerprint != prepared.profile_fingerprint:
        raise MatchedPreparedCPUAdapterV98Error(
            "v98 matched profile differs from the prepared receipt")
    ng, nc, nf, nb = (prepared.fraction_count, prepared.cell_count,
                      prepared.face_count, prepared.boundary_count)
    v76 = resolve_suspended_lifecycle_execution_v76(
        backend=V76_BACKEND_CPU, suspension_profile=profile,
        cell_area_m2=prepared.cell_area_m2,
        face_length_m=prepared.face_length_m,
        centre_distance_m=prepared.centre_distance_m,
        owner=prepared.owner, neighbour=prepared.neighbour,
        boundary_index=prepared.boundary_index,
        cell_face_offset=prepared.cell_face_offset,
        cell_faces=prepared.cell_faces,
        cell_face_sign=prepared.cell_face_sign,
        normal=prepared.normal,
        rectangle_width_m=prepared.rectangle_width_m,
        weno_cells=prepared.weno_cells,
        boundary_return_time_s=np.zeros((ng, nb)),
        dry_tolerance_m=prepared.dry_tolerance_m)
    v78 = resolve_multifraction_suspended_execution_v78(
        v76, role=ROLE_CPU_ORACLE,
        hydraulic_execution_sha256=prepared.source_topology_sha256)
    background = spatial_turbulence_background_v87(
        vicuv_m2_s=np.zeros(nc), dicuv_m2_s=np.zeros(nc),
        source_name="matched-canonical-closed-v98")
    closure = resolve_horizontal_turbulence_execution_v87(
        v78, background=background,
        tpsnumber_by_fraction=np.linspace(.8, 1.2, ng),
        hles_enabled=True, elder_enabled=True, alpha=5.0/3.0, nd=2,
        sigmat=.7, flp=1.0/3.0, reltim_s=2.0,
        vicmol_m2_s=0.0, dicmol_m2_s=0.0,
        von_karman=.41, gravity_m_s2=profile.gravity_m_s2)
    hles_execution = resolve_hles_producer_execution_v90(
        closure, backend=HLES_BACKEND_CPU, hturrt_min=2.0)
    hles = _wrap_hles_state(
        hles_execution,
        filtered=prepared.hles_filtered_face_normal_velocity_m_s,
        dynamic_vicuv=prepared.hles_dynamic_vicuv_m2_s,
        dynamic_dicuv=prepared.hles_dynamic_dicuv_m2_s,
        accepted_updates=0, time_s=0.0,
        receipt_chain=_hash({"input": prepared.receipt_sha256,
                             "kind": "accepted-hles"}),
        primed=True,
        prime_receipt_sha256=_hash(
            {"input": prepared.receipt_sha256, "kind": "hles-prime"}))
    historical_v91 = resolve_shared_turbulence_wetdry_execution_v97(
        hles_execution)
    suspended_state = resolve_multifraction_suspended_state_v78(
        historical_v91.base_execution,
        suspended_mass_kg_m2=prepared.suspended_mass_kg_m2,
        bed_mass_kg_m2=prepared.bed_mass_kg_m2)
    v91_state = resolve_shared_turbulence_state_v91(
        historical_v91, hles_state=hles,
        suspended_state=suspended_state, q=prepared.q)
    v92 = resolve_transported_fluid_execution_v92(
        historical_v91, temperature_mode=MODE_TRANSPORTED,
        salinity_mode=MODE_TRANSPORTED,
        equilibrium_temperature_c=11.0,
        equilibrium_salinity_ppt=3.0, density_eos="unesco")
    v92_state = resolve_transported_fluid_state_v92(
        v92, base_state=v91_state,
        temperature_cell_c=prepared.temperature_cell_c,
        salinity_cell_ppt=prepared.salinity_cell_ppt)
    if (not np.array_equal(v92_state.h_temperature_c_m,
                           prepared.h_temperature_c_m) or
            not np.array_equal(v92_state.h_salinity_ppt_m,
                               prepared.h_salinity_ppt_m) or
            not np.array_equal(v92_state.water_density_cell_kg_m3,
                               prepared.accepted_water_density_kg_m3)):
        raise MatchedPreparedCPUAdapterV98Error(
            "v98 matched v92 accepted q/T/S image differs")
    concentration = np.divide(
        prepared.suspended_mass_kg_m2, prepared.q[0][None, :],
        out=np.zeros_like(prepared.suspended_mass_kg_m2),
        where=prepared.q[0][None, :] > prepared.dry_tolerance_m)
    accepted_mixture = mixture_density_kg_m3_v85(
        concentration, v92_state.water_density_cell_kg_m3,
        profile.sediment_density_kg_m3)
    if not np.array_equal(
            accepted_mixture, prepared.accepted_mixture_density_kg_m3):
        raise MatchedPreparedCPUAdapterV98Error(
            "v98 matched accepted mixture-density image differs")

    ids = tuple(item.fraction_id
                for item in profile.capacity_profile.fractions)
    topology = wetdry_topology_v71(
        node_ids=np.arange(nc, dtype=np.int64),
        cell_area_m2=prepared.cell_area_m2,
        wet=np.ones(nc, dtype=np.bool_), owner=prepared.owner,
        neighbour=prepared.neighbour,
        role=np.where(
            prepared.neighbour >= 0, FACE_INTERNAL,
            np.where(prepared.boundary_index >= 0,
                     FACE_TRACE_INLET, FACE_CLOSED)).astype(np.int32))
    tag = lambda name: _hash({"input": prepared.receipt_sha256,
                              "stage": name})
    family = direct_vector_family_execution_document_v67(
        backend="cpu",
        channel_profile=resolve_channel_composition_profile(
            ids, (TRA_COMBINE,)*ng, bed_factor=1.0,
            suspended_factor=1.0, epsilon=1.0e-6),
        method_ids=(METHOD_VAN_RIJN_1993_CURRENT,)*ng,
        parameters=tuple({} for _ in range(ng)),
        d50_m=profile.sediment_d50_m,
        d90_m=tuple(1.6*np.asarray(profile.sediment_d50_m)),
        solid_density_kg_m3=profile.sediment_density_kg_m3,
        bulk_density_kg_m3=(1590.0,)*ng,
        roughness_method_id="roughness.manning",
        forcing_fingerprint=tag("forcing"),
        roughness_fingerprint=tag("roughness"),
        settling_fingerprint=tag("settling"),
        vertical_fingerprint=tag("vertical"),
        support_fingerprint=tag("support"),
        topology_fingerprint=topology.topology_id,
        boundary_fingerprint=tag("boundary"),
        face_fingerprint=tag("face"), exner_fingerprint=tag("exner"),
        minimum_work_depth_m=.01)
    upstream = {
        stage: (family["fingerprint"] if stage == "capacity_execution"
                else tag(stage)) for stage in UPSTREAM_STAGES}
    slope = bed_slope_execution_document(
        method_id="bed_slope.delft_bagnold_ikeda",
        fraction_ids=ids, upstream_fingerprints=upstream, **CONTROLS)
    live = direct_vector_live_taurat_execution_document_v73(
        parent_execution=family, bed_slope_execution=slope,
        repository_root=repository_root)
    family_faces = np.flatnonzero(
        prepared.boundary_index >= 0).astype(np.int32)
    boundary = build_bedload_boundary_execution_v70(
        backend="cpu_openmp",
        parent_execution_fingerprint=family["fingerprint"],
        topology_fingerprint=topology.topology_id,
        fraction_ids=ids, bulk_density_kg_m3=(1590.0,)*ng,
        solid_density_kg_m3=profile.sediment_density_kg_m3,
        boundaries=[{"name": f"boundary-{index}", "mode": MODE_CLOSED,
                     "rates": [0.0]*ng} for index in range(nb)],
        root=repository_root)
    boundary_topology = resolve_bedload_boundary_topology_v70(
        boundary, owner=prepared.owner[family_faces],
        boundary_index=prepared.boundary_index[family_faces],
        normal=prepared.normal[:, family_faces],
        face_length_m=prepared.face_length_m[family_faces],
        cell_area_m2=prepared.cell_area_m2)
    binding = resolve_cpu_q3_v84_mixed_binding_v93(
        topology=topology, normal=prepared.normal,
        face_length_m=prepared.face_length_m,
        cell_face_offset=prepared.cell_face_offset,
        cell_faces=prepared.cell_faces, signs=prepared.cell_face_sign,
        downhill=np.zeros((2, nf)), bed_slope_execution=slope,
        boundary_execution=boundary,
        boundary_topology=boundary_topology,
        family_boundary_faces=family_faces)
    factcr_execution = resolve_factcr_execution_v83(v78, factcr=FACTCR)
    selector_kwargs = dict(
        routes=ROUTES, ordered_fraction_ids=ids,
        q3_execution_fingerprint=factcr_execution.fingerprint,
        q3_topology_id=binding.topology_id,
        q3_binding_sha256=binding.binding_sha256,
        q3_boundary_names=binding.boundary_names)
    if control.lifecycle_route == LIFECYCLE_EXACT_ZERO_V94B:
        v93 = resolve_suspended_exner_zero_vector_execution_v93(
            v92, **selector_kwargs)
    else:
        v93 = resolve_suspended_exner_execution_v93(
            v92, neglect_entrainment=control.neglect_entrainment,
            **selector_kwargs)
    active_execution = resolve_active_suspended_control_execution_v94b(
        v93, factcr_execution=factcr_execution,
        live_taurat_execution=live,
        iopsus_by_fraction=control.iopsus_by_fraction,
        suscor=control.suscor,
        initializer_water_density_kg_m3=profile.water_density_kg_m3,
        initializer_kinematic_viscosity_m2_s=1.0e-6)
    provider_state = resolve_cpu_q3_v84_mixed_state_v93(
        execution_fingerprint=factcr_execution.fingerprint,
        binding=binding,
        authoritative_bed_mass_kg_m2=prepared.bed_mass_kg_m2,
        bulk_density_kg_m3=(1590.0,)*ng)
    provider = CPUQ3V83ActiveWetDryExnerAdapterV97(
        outer_execution=active_execution, binding=binding,
        state=provider_state, native_factcr_runtime=factcr_runtime,
        native_q3_face_runtime=q3_face_runtime,
        native_threads=native_threads)
    active_state = resolve_initial_active_suspended_control_state_v94b(
        active_execution, base_v92_state=v92_state,
        q3_provider=provider)

    density_execution = resolve_densin_execution_v85(
        role="native_cpu",
        solid_density_kg_m3=profile.sediment_density_kg_m3,
        water_density_kg_m3=prepared.accepted_water_density_kg_m3,
        hydraulic_reference_density_kg_m3=profile.water_density_kg_m3,
        gravity_m_s2=profile.gravity_m_s2,
        dry_tolerance_m=v76.dry_tolerance_m,
        owner=prepared.owner, neighbour=prepared.neighbour,
        normal=prepared.normal, face_length_m=prepared.face_length_m,
        face_distance_m=prepared.centre_distance_m,
        cell_area_m2=prepared.cell_area_m2,
        cell_face_offset=prepared.cell_face_offset,
        cell_faces=prepared.cell_faces,
        cell_face_sign=prepared.cell_face_sign,
        hydraulic_execution_sha256=v78.hydraulic_execution_sha256)
    geometry_execution = resolve_coupled_geometry_native_cpu_execution_v96(
        density_execution, geometry_runtime.receipt,
        include_sediment_density_increment=True,
        native_threads=native_threads, maximum_dt_s=.01)
    hydraulic_execution = resolve_wetdry_fwave_cpu_execution_v97(
        geometry_execution, wetdry_runtime.receipt,
        domain_active_mask=np.ones(nc, np.int32),
        native_threads=native_threads,
        nonlinear_tolerance=2.0e-10, max_inner_iterations=160)
    solid = np.asarray(profile.sediment_density_kg_m3)
    porosity = np.full(ng, .4)
    bulk = solid*(1.0-porosity)
    execution = resolve_coupled_wetdry_cpu_execution_v97(
        active_execution, hydraulic_execution,
        density_projection_runtime=density_projection_runtime,
        turbulence_runtime=turbulence_runtime,
        scalar_fct_runtime=scalar_fct_runtime,
        scalar_diffusion_runtime=scalar_diffusion_runtime,
        scalar_boundary_runtime=scalar_boundary_runtime,
        settling_exchange_runtime=settling_exchange_runtime,
        factcr_runtime=factcr_runtime,
        q3_face_runtime=q3_face_runtime,
        hles_runtime=hles_runtime,
        initial_bed_elevation_m=prepared.bed_elevation_m,
        reference_bed_mass_kg_m2=prepared.bed_mass_kg_m2,
        dry_bulk_density_kg_m3=bulk,
        bed_porosity_by_fraction=porosity)
    state = resolve_coupled_wetdry_cpu_state_v97(
        execution, active_state=active_state, hles_state=hles,
        q3_provider=provider)
    depth = np.asarray(state.q[0])
    q3_inputs = {
        "observation_height_m": np.maximum(depth, .001)/np.e,
        "additional_shear_stress_pa": np.zeros(nc),
        "sediment_d90_m": 1.6*np.asarray(profile.sediment_d50_m),
        "molecular_schmidt": np.asarray((1.0, .9, .8)),
        "aks_factor": np.asarray((1.0, 1.1, 1.2)),
        "maximum_volume_concentration": np.asarray((.65, .62, .60)),
        "minimum_depth_m": np.asarray((.05, .05, .05)),
        "bed_factor": np.asarray((1.0, .9, 1.1)),
        "suspended_factor": np.asarray((.8, 1.0, 1.2)),
        "roughness_z0_m": np.ascontiguousarray(np.broadcast_to(
            np.asarray((1.2e-4, 1.5e-4, 1.8e-4))[:, None],
            (ng, nc))),
    }
    runtimes = {
        "native_runtime": wetdry_runtime,
        "density_projection_runtime": density_projection_runtime,
        "turbulence_runtime": turbulence_runtime,
        "scalar_fct_runtime": scalar_fct_runtime,
        "scalar_diffusion_runtime": scalar_diffusion_runtime,
        "scalar_boundary_runtime": scalar_boundary_runtime,
        "settling_exchange_runtime": settling_exchange_runtime,
        "hles_runtime": hles_runtime,
    }
    placeholder = MatchedPreparedClosedCPUCaseV98(
        prepared, control, execution, state, provider, q3_inputs, runtimes,
        "")
    return MatchedPreparedClosedCPUCaseV98(
        **{**placeholder.__dict__,
           "construction_sha256": _case_token(placeholder)})


def validate_matched_prepared_closed_cpu_case_v98(
    value: MatchedPreparedClosedCPUCaseV98,
) -> None:
    validate_matched_prepared_cpu_input_v98(value.prepared_input)
    validate_matched_prepared_control_v99(value.control)
    if (type(value) is not MatchedPreparedClosedCPUCaseV98 or
            value.construction_sha256 != _case_token(value) or
            not np.array_equal(value.state.q, value.prepared_input.q) or
            not np.array_equal(
                value.execution.active_execution.iopsus_by_fraction,
                value.control.iopsus_by_fraction) or
            value.execution.active_execution.suscor != value.control.suscor or
            value.execution.active_execution.base_execution.neglect_entrainment
                != value.control.neglect_entrainment):
        raise MatchedPreparedCPUAdapterV98Error(
            "v98 matched closed CPU case differs")


__all__ = [
    "DT_S",
    "IOPSUS",
    "CONTROL_ET", "CONTROL_EF", "CONTROL_DT", "CONTROL_DF",
    "CONTROL_IDS",
    "LIFECYCLE_HISTORICAL_V94", "LIFECYCLE_EXACT_ZERO_V94B",
    "BED_OWNER_EXCHANGE", "BED_OWNER_CONSTRUCTED_DIVERGENCE",
    "BED_OWNER_EXACT_ZERO",
    "MatchedPreparedCPUAdapterV98Error",
    "MatchedPreparedBoundaryClosureReceiptV98",
    "MatchedPreparedCPUInputV98",
    "MatchedPreparedControlV99",
    "MatchedPreparedClosedCPUCaseV98",
    "build_matched_prepared_closed_cpu_case_v98",
    "resolve_matched_prepared_control_v99",
    "resolve_matched_prepared_cpu_input_v98",
    "validate_matched_prepared_control_v99",
    "validate_matched_prepared_closed_cpu_case_v98",
    "validate_matched_prepared_cpu_input_v98",
]
