"""Atomic closed-grid CPU owner for the v97 moving wet/dry route.

One accepted state is shared by native augmented-f-wave hydraulics, the
rank-aware density source, masked lagged-v91 momentum, combined scalar
transport, v94b/Q3/Exner, bed feedback, and the lagged HLES producer.  Dry
cells are represented in that accepted state; no fully-wet shadow state is
constructed.  This bounded owner is fixed-grid and closed-boundary only.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Callable, Mapping

import numpy as np

from . import active_suspended_control_v94 as _v94
from .active_suspended_control_v94 import (
    ActiveSuspendedControlExecutionV94,
    ActiveSuspendedControlStateV94,
    ActiveSuspendedControlTrialV94,
)
from .active_suspended_control_v94b import (
    checkpoint_active_suspended_control_state_v94b,
    commit_active_suspended_control_trial_v94b,
    execution_document_v94b,
    load_active_suspended_control_checkpoint_v94b,
    rollback_active_suspended_control_trial_v94b,
    validate_active_suspended_control_state_v94b,
)
from .active_suspended_wetdry_cpu_v97 import (
    begin_active_suspended_wetdry_cpu_trial_v97,
)
from .coupled_densin_cpu_v96 import (
    _grid_execution,
    _provider_is_closed,
    _restore_provider_after_failure,
    _v92_execution,
    _v92_state,
)
from .density_active_projection_v97 import (
    NativeDensityActiveProjectionRuntimeV97,
)
from .factcr_native_cpu_v97 import (
    NativeFactCrEvidenceV97,
    NativeFactCrRuntimeV97,
)
from .densin_fwave_v85 import mixture_density_kg_m3_v85
from .hles_native_cpu_v97 import NativeHlesCandidateV97, NativeHlesRuntimeV97
from .hles_producer_v90 import (
    CurrentClosureHandoffV90,
    HlesProducerStateV90,
    HlesProducerTrialV90,
    commit_hles_trial_v90,
    hles_checkpoint_v90,
    load_hles_checkpoint_v90,
    make_current_closure_handoff_v90,
    make_hles_input_receipt_v90,
    rollback_hles_trial_v90,
    validate_hles_producer_state_v90,
)
from .scalar_diffusion_native_cpu_v96 import NativeScalarDiffusionRuntimeV96
from .scalar_fct_native_cpu_v96 import NativeScalarFCTRuntimeV96
from .scalar_boundary_native_cpu_v97 import NativeScalarBoundaryRuntimeV97
from .settling_exchange_native_cpu_v97 import (
    NativeSettlingExchangeRuntimeV97,
)
from .q3_wetdry_adapter_cpu_v97 import (
    CPUQ3V83ActiveWetDryExnerAdapterV97,
)
from .q3_face_inventory_native_cpu_v97 import (
    NativeQ3FaceInventoryResultV97,
    NativeQ3FaceInventoryRuntimeV97,
    _digest as _native_q3_digest,
)
from .shared_turbulence_consumers_v91 import (
    STATE_DOMAIN_MOVING_WETDRY_V97,
)
from .shared_turbulence_native_cpu_v97 import (
    NativeSharedTurbulenceRuntimeV97,
)
from .transported_fluid_native_scalar_cpu_v96 import (
    NativeCombinedScalarEvidenceV96,
    _evidence_token as _scalar_evidence_token,
)
from .transported_fluid_wetdry_cpu_v97 import (
    WetDryHydraulicEvidenceV97,
    make_wetdry_hydraulic_evidence_v97,
    validate_wetdry_hydraulic_evidence_v97,
)
from .wetdry_fwave_native_cpu_v97 import (
    NativeWetDryFwaveRuntimeV97,
    WetDryFwaveCPUExecutionV97,
    execution_document_wetdry_fwave_v97,
    solve_wetdry_midpoint_trial_v97,
)
from . import hles_native_cpu_v97 as _native_hles


SCHEMA = 1
SCOPE_ID = "braids-q4-coupled-moving-wetdry-owner-cpu-v97-v1"
SEMANTIC_ID = SCOPE_ID + ":closed-fixed-grid-atomic-owner"
CHECKPOINT_SCOPE_ID = SCOPE_ID + ":checkpoint"
BACKEND = "native_cpu_openmp"
MAX_ACCEPTED_UPDATES = 4
GEOMETRY_POLICY = "preserve_h_shift_eta_after_exner"


class CoupledWetDryOwnerV97Error(ValueError):
    """The v97 outer execution, state, receipt, or trial is invalid."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          allow_nan=False).encode("ascii")
    except (TypeError, ValueError, OverflowError) as error:
        raise CoupledWetDryOwnerV97Error(
            "v97 outer value is not finite canonical JSON") from error


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


def _array(value: object, shape: tuple[int, ...], name: str, *,
           nonnegative: bool = False, positive: bool = False) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != shape or raw.dtype.kind not in "fiub":
        raise CoupledWetDryOwnerV97Error(
            f"v97 outer {name} must be numeric with shape {shape}")
    result = np.array(raw, dtype=np.float64, order="C", copy=True)
    if not np.isfinite(result).all() or \
            (nonnegative and np.any(result < 0.0)) or \
            (positive and np.any(result <= 0.0)):
        raise CoupledWetDryOwnerV97Error(
            f"v97 outer {name} is outside its finite domain")
    result.setflags(write=False)
    return result


def _cancelled(value: bool | Callable[[], bool]) -> bool:
    if type(value) is bool:
        return value
    if not callable(value):
        raise TypeError("cancelled must be Boolean or callable")
    return bool(value())


@dataclass(frozen=True)
class CoupledWetDryCPUExecutionV97:
    active_execution: ActiveSuspendedControlExecutionV94
    hydraulic_execution: WetDryFwaveCPUExecutionV97
    density_projection_runtime_sha256: str
    turbulence_runtime_sha256: str
    scalar_fct_runtime_sha256: str
    scalar_diffusion_runtime_sha256: str
    scalar_boundary_runtime_sha256: str
    settling_exchange_runtime_sha256: str
    factcr_runtime_sha256: str
    q3_face_runtime_sha256: str
    hles_runtime_sha256: str
    initial_bed_elevation_m: np.ndarray
    reference_bed_mass_kg_m2: np.ndarray
    dry_bulk_density_kg_m3: np.ndarray
    bed_porosity_by_fraction: np.ndarray
    semantic_id: str
    fingerprint: str

    @property
    def fraction_count(self) -> int:
        return self.active_execution.fraction_count

    @property
    def cell_count(self) -> int:
        return self.active_execution.cell_count

    @property
    def face_count(self) -> int:
        return _v92_execution(self.active_execution).face_count


def _execution_payload(value: CoupledWetDryCPUExecutionV97) -> dict:
    return {
        "schema": SCHEMA, "scope_id": SCOPE_ID,
        "semantic_id": value.semantic_id, "backend": BACKEND,
        "active_execution": execution_document_v94b(value.active_execution),
        "wetdry_execution": execution_document_wetdry_fwave_v97(
            value.hydraulic_execution),
        "runtime_sha256": {
            "density_projection": value.density_projection_runtime_sha256,
            "masked_v91": value.turbulence_runtime_sha256,
            "scalar_fct": value.scalar_fct_runtime_sha256,
            "scalar_diffusion": value.scalar_diffusion_runtime_sha256,
            "scalar_boundary": value.scalar_boundary_runtime_sha256,
            "settling_exchange": value.settling_exchange_runtime_sha256,
            "factcr": value.factcr_runtime_sha256,
            "q3_face_inventory": value.q3_face_runtime_sha256,
            "hles": value.hles_runtime_sha256,
        },
        "initial_bed_sha256": _digest(value.initial_bed_elevation_m),
        "reference_bed_mass_sha256": _digest(
            value.reference_bed_mass_kg_m2),
        "dry_bulk_density_sha256": _digest(value.dry_bulk_density_kg_m3),
        "bed_porosity_sha256": _digest(value.bed_porosity_by_fraction),
        "dry_bulk_density_origin": "rhosol*(1-porosity)",
        "geometry_policy": GEOMETRY_POLICY,
        "hydraulic_owner": "native_augmented_fwave_active_set",
        "scalar_owner": "one_combined_native_fct_diffusion",
        "dry_transfer_owner": "one_v92_cleanup_consumed_by_v93",
        "hles_owner": "one_native_candidate_one_v90_clock",
        "closed_boundary_only": True,
        "accepted_update_limit": MAX_ACCEPTED_UPDATES,
        "product_admission": False,
    }


def resolve_coupled_wetdry_cpu_execution_v97(
    active_execution: ActiveSuspendedControlExecutionV94,
    hydraulic_execution: WetDryFwaveCPUExecutionV97, *,
    density_projection_runtime: NativeDensityActiveProjectionRuntimeV97,
    turbulence_runtime: NativeSharedTurbulenceRuntimeV97,
    scalar_fct_runtime: NativeScalarFCTRuntimeV96,
    scalar_diffusion_runtime: NativeScalarDiffusionRuntimeV96,
    scalar_boundary_runtime: NativeScalarBoundaryRuntimeV97,
    settling_exchange_runtime: NativeSettlingExchangeRuntimeV97,
    factcr_runtime: NativeFactCrRuntimeV97,
    q3_face_runtime: NativeQ3FaceInventoryRuntimeV97,
    hles_runtime: NativeHlesRuntimeV97,
    initial_bed_elevation_m: object,
    reference_bed_mass_kg_m2: object,
    dry_bulk_density_kg_m3: object,
    bed_porosity_by_fraction: object,
) -> CoupledWetDryCPUExecutionV97:
    execution_document_v94b(active_execution)
    execution_document_wetdry_fwave_v97(hydraulic_execution)
    v92 = _v92_execution(active_execution)
    grid = _grid_execution(active_execution)
    if (active_execution.backend != "cpu" or
            v92.base_execution.state_domain_mode !=
                STATE_DOMAIN_MOVING_WETDRY_V97):
        raise CoupledWetDryOwnerV97Error(
            "v97 outer route requires CPU moving wet/dry state semantics")
    hydro = hydraulic_execution.base_execution.base_execution
    topology = all(np.array_equal(a, b) for a, b in (
        (grid.owner, hydro.owner), (grid.neighbour, hydro.neighbour),
        (grid.normal, hydro.normal),
        (grid.face_length_m, hydro.face_length_m),
        (grid.centre_distance_m, hydro.face_distance_m),
        (grid.cell_area_m2, hydro.cell_area_m2),
        (grid.cell_face_offset, hydro.cell_face_offset),
        (grid.cell_faces, hydro.cell_faces),
        (grid.cell_face_sign, hydro.cell_face_sign)))
    if not topology or grid.dry_tolerance_m != hydro.dry_tolerance_m:
        raise CoupledWetDryOwnerV97Error(
            "v97 outer scalar and hydraulic topologies differ")
    ng, nc = active_execution.fraction_count, active_execution.cell_count
    z0 = _array(initial_bed_elevation_m, (nc,), "initial bed")
    m0 = _array(reference_bed_mass_kg_m2, (ng, nc),
                "reference bed mass", nonnegative=True)
    bulk = _array(dry_bulk_density_kg_m3, (ng,), "dry bulk density",
                  positive=True)
    porosity = _array(bed_porosity_by_fraction, (ng,), "bed porosity",
                      nonnegative=True)
    solid = np.asarray(grid.profile.sediment_density_kg_m3)
    if np.any(porosity >= 1.0) or not np.allclose(
            bulk, solid*(1.0-porosity), rtol=0.0, atol=1e-11):
        raise CoupledWetDryOwnerV97Error(
            "v97 outer dry bulk density origin differs")
    if (type(scalar_boundary_runtime) is not
            NativeScalarBoundaryRuntimeV97 or
            type(factcr_runtime) is not NativeFactCrRuntimeV97 or
            type(q3_face_runtime) is not NativeQ3FaceInventoryRuntimeV97):
        raise CoupledWetDryOwnerV97Error(
            "v97 outer native boundary/FactCr/Q3 face runtime type differs")
    runtime_sha = (
        density_projection_runtime.receipt["receipt_sha256"],
        turbulence_runtime.receipt["receipt_sha256"],
        scalar_fct_runtime.runtime_sha256,
        scalar_diffusion_runtime.receipt["receipt_sha256"],
        scalar_boundary_runtime.receipt["receipt_sha256"],
        settling_exchange_runtime.receipt["receipt_sha256"],
        factcr_runtime.receipt["receipt_sha256"],
        q3_face_runtime.receipt["receipt_sha256"],
        hles_runtime.receipt["receipt_sha256"],
    )
    placeholder = CoupledWetDryCPUExecutionV97(
        active_execution, hydraulic_execution, *runtime_sha,
        z0, m0, bulk, porosity, SEMANTIC_ID, "")
    return CoupledWetDryCPUExecutionV97(
        **{**placeholder.__dict__,
           "fingerprint": _hash(_execution_payload(placeholder))})


def execution_document_coupled_wetdry_v97(
    value: CoupledWetDryCPUExecutionV97,
) -> dict:
    if type(value) is not CoupledWetDryCPUExecutionV97:
        raise CoupledWetDryOwnerV97Error("v97 outer execution type differs")
    payload = _execution_payload(value)
    if (value.semantic_id != SEMANTIC_ID or
            value.fingerprint != _hash(payload)):
        raise CoupledWetDryOwnerV97Error(
            "v97 outer execution identity differs")
    return {**payload, "fingerprint": value.fingerprint}


def _bed_from_mass(execution: CoupledWetDryCPUExecutionV97,
                   mass: object) -> np.ndarray:
    bed_mass = _array(
        mass, (execution.fraction_count, execution.cell_count),
        "accepted bed mass", nonnegative=True)
    bed = np.ascontiguousarray(
        execution.initial_bed_elevation_m+
        np.sum((bed_mass-execution.reference_bed_mass_kg_m2)/
               execution.dry_bulk_density_kg_m3[:, None], axis=0))
    if not np.isfinite(bed).all():
        raise CoupledWetDryOwnerV97Error("v97 outer bed is non-finite")
    bed.setflags(write=False)
    return bed


@dataclass(frozen=True)
class CoupledWetDryCPUStateV97:
    execution_fingerprint: str
    active_state: ActiveSuspendedControlStateV94
    hles_state: HlesProducerStateV90
    bed_elevation_m: np.ndarray
    free_surface_elevation_m: np.ndarray
    accepted_chain_sha256: str
    state_sha256: str

    @property
    def q(self) -> np.ndarray:
        return _v92_state(self.active_state).q

    @property
    def accepted_updates(self) -> int:
        return self.active_state.accepted_updates

    @property
    def time_s(self) -> float:
        return self.active_state.time_s


def _state_token(value: CoupledWetDryCPUStateV97) -> str:
    return _hash({
        "scope_id": SCOPE_ID+":state",
        "execution_fingerprint": value.execution_fingerprint,
        "active_state": value.active_state.state_sha256,
        "hles_state": value.hles_state.state_sha256,
        "geometry": _digest(value.bed_elevation_m,
                            value.free_surface_elevation_m),
        "accepted_chain": value.accepted_chain_sha256,
        "updates": value.accepted_updates, "time_s": value.time_s,
    })


def _wrap_state(execution: CoupledWetDryCPUExecutionV97, *,
                active_state: ActiveSuspendedControlStateV94,
                hles_state: HlesProducerStateV90,
                accepted_chain_sha256: str,
                q3_provider: object | None = None) -> CoupledWetDryCPUStateV97:
    execution_document_coupled_wetdry_v97(execution)
    validate_active_suspended_control_state_v94b(
        active_state, execution.active_execution, q3_provider)
    hles_execution = _v92_execution(
        execution.active_execution).base_execution.hles_execution
    validate_hles_producer_state_v90(hles_state, hles_execution)
    v92 = _v92_state(active_state)
    dry = (v92.q[0] <=
           execution.hydraulic_execution.base_execution.base_execution.
           dry_tolerance_m)
    if (active_state.accepted_updates != hles_state.accepted_updates or
            active_state.time_s != hles_state.time_s or
            active_state.accepted_updates > MAX_ACCEPTED_UPDATES or
            np.any(v92.q[0] < 0.0) or np.any(v92.q[1:, dry] != 0.0) or
            np.any(v92.suspended_state.suspended_mass_kg_m2[:, dry] != 0.0) or
            (q3_provider is not None and not _provider_is_closed(q3_provider))):
        raise CoupledWetDryOwnerV97Error(
            "v97 outer clocks, dry state, or provider scope differs")
    bed = _bed_from_mass(execution, active_state.base_state.bed_mass_kg_m2)
    eta = np.ascontiguousarray(v92.q[0]+bed)
    eta.setflags(write=False)
    placeholder = CoupledWetDryCPUStateV97(
        execution.fingerprint, active_state, hles_state, bed, eta,
        accepted_chain_sha256, "")
    return CoupledWetDryCPUStateV97(
        **{**placeholder.__dict__, "state_sha256": _state_token(placeholder)})


def resolve_coupled_wetdry_cpu_state_v97(
    execution: CoupledWetDryCPUExecutionV97, *,
    active_state: ActiveSuspendedControlStateV94,
    hles_state: HlesProducerStateV90,
    q3_provider: object,
) -> CoupledWetDryCPUStateV97:
    if active_state.accepted_updates != 0 or hles_state.accepted_updates != 0:
        raise CoupledWetDryOwnerV97Error(
            "v97 outer fresh state must start at update zero")
    if not np.array_equal(active_state.base_state.bed_mass_kg_m2,
                          execution.reference_bed_mass_kg_m2):
        raise CoupledWetDryOwnerV97Error(
            "v97 outer reference bed mass differs")
    chain = _hash({"scope_id": SCOPE_ID,
                   "execution": execution.fingerprint,
                   "active": active_state.state_sha256,
                   "hles": hles_state.state_sha256,
                   "bed": _digest(execution.initial_bed_elevation_m)})
    return _wrap_state(execution, active_state=active_state,
                       hles_state=hles_state, accepted_chain_sha256=chain,
                       q3_provider=q3_provider)


def validate_coupled_wetdry_cpu_state_v97(
    value: CoupledWetDryCPUStateV97,
    execution: CoupledWetDryCPUExecutionV97,
    q3_provider: object | None = None,
) -> None:
    if (type(value) is not CoupledWetDryCPUStateV97 or
            value.execution_fingerprint != execution.fingerprint or
            value.state_sha256 != _state_token(value)):
        raise CoupledWetDryOwnerV97Error(
            "v97 outer state is stale or cross-execution")
    rebuilt = _wrap_state(
        execution, active_state=value.active_state,
        hles_state=value.hles_state,
        accepted_chain_sha256=value.accepted_chain_sha256,
        q3_provider=q3_provider)
    if rebuilt.state_sha256 != value.state_sha256:
        raise CoupledWetDryOwnerV97Error("v97 outer rebuilt state differs")


def _accepted_density(
    execution: CoupledWetDryCPUExecutionV97,
    parent: CoupledWetDryCPUStateV97,
) -> tuple[np.ndarray, np.ndarray]:
    v92 = _v92_state(parent.active_state)
    grid = _grid_execution(execution.active_execution)
    h = np.asarray(v92.q[0])
    wet = h > grid.dry_tolerance_m
    concentration = np.zeros((execution.fraction_count,
                              execution.cell_count), dtype=np.float64)
    concentration[:, wet] = (
        v92.suspended_state.suspended_mass_kg_m2[:, wet]/h[wet])
    water = np.ascontiguousarray(v92.water_density_cell_kg_m3)
    mixture = mixture_density_kg_m3_v85(
        concentration, water,
        execution.hydraulic_execution.base_execution.base_execution.
        solid_density_kg_m3)
    water.setflags(write=False)
    mixture.setflags(write=False)
    return water, mixture


def _closed_scalar_inputs(
    execution: CoupledWetDryCPUExecutionV97,
    parent: CoupledWetDryCPUStateV97,
    endpoint_q: np.ndarray,
    shear_velocity_m_s: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    v92_execution = _v92_execution(execution.active_execution)
    v92 = _v92_state(parent.active_state)
    grid = _grid_execution(execution.active_execution)
    ng, nc, nf, nb = (execution.fraction_count, execution.cell_count,
                      execution.face_count, v92_execution.boundary_count)
    shear = _array(
        shear_velocity_m_s, (nc,), "shear velocity", nonnegative=True)
    h0 = np.asarray(v92.q[0])
    wet0 = h0 > grid.dry_tolerance_m
    concentration = np.zeros((ng, nc), dtype=np.float64)
    concentration[:, wet0] = (
        v92.suspended_state.suspended_mass_kg_m2[:, wet0]/h0[wet0])
    left = np.ascontiguousarray(concentration[:, grid.owner])
    right = np.array(left, copy=True, order="C")
    internal = grid.neighbour >= 0
    right[:, internal] = concentration[:, grid.neighbour[internal]]
    fluid = np.vstack((v92.temperature_cell_c, v92.salinity_cell_ppt))
    fluid_left = np.ascontiguousarray(fluid[:, grid.owner])
    fluid_right = np.array(fluid_left, copy=True, order="C")
    fluid_right[:, internal] = fluid[:, grid.neighbour[internal]]
    prescribed = np.zeros((ng, nb), dtype=np.float64)
    prescribed_fluid = np.broadcast_to(
        np.asarray((v92_execution.equilibrium_temperature_c,
                    v92_execution.equilibrium_salinity_ppt))[:, None],
        (2, nb)).copy()
    wet1 = endpoint_q[0] > grid.dry_tolerance_m
    face_active = (internal & wet1[grid.owner] &
                   wet1[np.maximum(grid.neighbour, 0)])
    face_depth = np.zeros(nf, dtype=np.float64)
    face_depth[face_active] = 0.5*(
        endpoint_q[0, grid.owner[face_active]]+
        endpoint_q[0, grid.neighbour[face_active]])
    scalar = {
        "reconstructed_left_concentration_kg_m3": left,
        "reconstructed_right_concentration_kg_m3": right,
        "prescribed_boundary_concentration_kg_m3": prescribed,
        "accepted_face_exchange_area_m2": np.ascontiguousarray(
            face_depth*grid.face_length_m),
        "transport_face_active": np.ascontiguousarray(face_active),
        "shear_velocity_m_s": shear,
        "hydro_constant": np.ones(nc, dtype=np.bool_),
    }
    return fluid_left, fluid_right, prescribed_fluid, scalar


def _hles_inputs(
    execution: CoupledWetDryCPUExecutionV97,
    endpoint_q: np.ndarray,
    face_flux: np.ndarray,
    closure_chezy_mhalf_s: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grid = _grid_execution(execution.active_execution)
    dry = grid.dry_tolerance_m
    wet = endpoint_q[0] > dry
    internal = grid.neighbour >= 0
    active_face = (internal & wet[grid.owner] &
                   wet[np.maximum(grid.neighbour, 0)])
    depth = np.zeros(execution.face_count, dtype=np.float64)
    depth[active_face] = 0.5*(
        endpoint_q[0, grid.owner[active_face]]+
        endpoint_q[0, grid.neighbour[active_face]])
    velocity = np.zeros(execution.face_count, dtype=np.float64)
    velocity[active_face] = face_flux[active_face]/depth[active_face]
    chezy = _array(closure_chezy_mhalf_s, (execution.cell_count,),
                   "closure Chezy", positive=True)
    result = (np.ascontiguousarray(velocity),
              np.ascontiguousarray(active_face, dtype=np.int32),
              np.ascontiguousarray(endpoint_q[0]), chezy,
              np.ascontiguousarray(wet, dtype=np.int32))
    for value in result:
        value.setflags(write=False)
    return result


@dataclass(frozen=True)
class CoupledWetDryOwnerCountsV97:
    native_hydraulic_owner_count: int = 1
    water_flux_limiter_owner_count: int = 1
    density_projection_owner_count: int = 1
    masked_v91_owner_count: int = 1
    scalar_fct_owner_count: int = 1
    scalar_diffusion_owner_count: int = 1
    scalar_boundary_owner_count: int = 1
    scalar_lifecycle_owner_count: int = 1
    wetdry_bed_transfer_owner_count: int = 1
    authoritative_exner_owner_count: int = 1
    bed_geometry_owner_count: int = 1
    native_hles_candidate_owner_count: int = 1
    native_factcr_producer_owner_count: int = 1
    native_q3_face_inventory_owner_count: int = 1
    hles_clock_owner_count: int = 1
    duplicate_owner_count: int = 0


@dataclass(frozen=True)
class CoupledWetDryCPUTrialV97:
    execution_fingerprint: str
    parent_state_sha256: str
    current_handoff: CurrentClosureHandoffV90
    hydraulic_evidence: WetDryHydraulicEvidenceV97
    native_scalar_evidence: NativeCombinedScalarEvidenceV96
    native_factcr_evidence: NativeFactCrEvidenceV97
    native_q3_face_evidence: NativeQ3FaceInventoryResultV97
    active_trial: ActiveSuspendedControlTrialV94
    hles_trial: HlesProducerTrialV90
    native_hles_evidence: NativeHlesCandidateV97
    hles_inputs: tuple[np.ndarray, np.ndarray, np.ndarray,
                       np.ndarray, np.ndarray]
    owner_counts: CoupledWetDryOwnerCountsV97
    candidate: CoupledWetDryCPUStateV97
    backend: str
    trial_sha256: str


def _trial_token(value: CoupledWetDryCPUTrialV97) -> str:
    return _hash({
        "scope_id": SCOPE_ID+":trial",
        "execution": value.execution_fingerprint,
        "parent": value.parent_state_sha256,
        "handoff": value.current_handoff.receipt_sha256,
        "hydraulic": value.hydraulic_evidence.evidence_sha256,
        "native_scalar": value.native_scalar_evidence.evidence_sha256,
        "native_factcr": _hash(value.native_factcr_evidence.__dict__),
        "native_q3_face": {
            "runtime": value.native_q3_face_evidence.runtime_sha256,
            "method": value.native_q3_face_evidence.method_fingerprint,
            "input": value.native_q3_face_evidence.input_sha256,
            "output": value.native_q3_face_evidence.output_sha256,
            "threads": value.native_q3_face_evidence.native_threads,
        },
        "active": value.active_trial.trial_sha256,
        "hles": value.hles_trial.trial_sha256,
        "native_hles": value.native_hles_evidence.result_sha256,
        "hles_inputs": _digest(*value.hles_inputs),
        "owners": list(value.owner_counts.__dict__.values()),
        "candidate": value.candidate.state_sha256,
        "backend": value.backend,
    })


def _validate_runtimes(
    execution: CoupledWetDryCPUExecutionV97, *,
    native_runtime: NativeWetDryFwaveRuntimeV97,
    density_projection_runtime: NativeDensityActiveProjectionRuntimeV97,
    turbulence_runtime: NativeSharedTurbulenceRuntimeV97,
    scalar_fct_runtime: NativeScalarFCTRuntimeV96,
    scalar_diffusion_runtime: NativeScalarDiffusionRuntimeV96,
    scalar_boundary_runtime: NativeScalarBoundaryRuntimeV97,
    settling_exchange_runtime: NativeSettlingExchangeRuntimeV97,
    hles_runtime: NativeHlesRuntimeV97,
) -> None:
    if (type(native_runtime) is not NativeWetDryFwaveRuntimeV97 or
            native_runtime.receipt !=
                execution.hydraulic_execution.native_build_receipt or
            type(density_projection_runtime) is not
                NativeDensityActiveProjectionRuntimeV97 or
            density_projection_runtime.receipt["receipt_sha256"] !=
                execution.density_projection_runtime_sha256 or
            type(turbulence_runtime) is not
                NativeSharedTurbulenceRuntimeV97 or
            turbulence_runtime.receipt["receipt_sha256"] !=
                execution.turbulence_runtime_sha256 or
            type(scalar_fct_runtime) is not NativeScalarFCTRuntimeV96 or
            scalar_fct_runtime.runtime_sha256 !=
                execution.scalar_fct_runtime_sha256 or
            type(scalar_diffusion_runtime) is not
                NativeScalarDiffusionRuntimeV96 or
            scalar_diffusion_runtime.receipt["receipt_sha256"] !=
                execution.scalar_diffusion_runtime_sha256 or
            type(scalar_boundary_runtime) is not
                NativeScalarBoundaryRuntimeV97 or
            scalar_boundary_runtime.receipt["receipt_sha256"] !=
                execution.scalar_boundary_runtime_sha256 or
            type(settling_exchange_runtime) is not
                NativeSettlingExchangeRuntimeV97 or
            settling_exchange_runtime.receipt["receipt_sha256"] !=
                execution.settling_exchange_runtime_sha256 or
            type(hles_runtime) is not NativeHlesRuntimeV97 or
            hles_runtime.receipt["receipt_sha256"] !=
                execution.hles_runtime_sha256):
        raise CoupledWetDryOwnerV97Error(
            "v97 outer native runtime identity differs")


def begin_coupled_wetdry_cpu_trial_v97(
    execution: CoupledWetDryCPUExecutionV97,
    parent: CoupledWetDryCPUStateV97,
    q3_provider: object, *,
    native_runtime: NativeWetDryFwaveRuntimeV97,
    density_projection_runtime: NativeDensityActiveProjectionRuntimeV97,
    turbulence_runtime: NativeSharedTurbulenceRuntimeV97,
    scalar_fct_runtime: NativeScalarFCTRuntimeV96,
    scalar_diffusion_runtime: NativeScalarDiffusionRuntimeV96,
    scalar_boundary_runtime: NativeScalarBoundaryRuntimeV97,
    settling_exchange_runtime: NativeSettlingExchangeRuntimeV97,
    hles_runtime: NativeHlesRuntimeV97,
    shear_velocity_m_s: object,
    closure_chezy_mhalf_s: object,
    q3_inputs: Mapping[str, object],
    dt_s: float,
    cancelled: bool | Callable[[], bool] = False,
) -> CoupledWetDryCPUTrialV97:
    execution_document_coupled_wetdry_v97(execution)
    validate_coupled_wetdry_cpu_state_v97(parent, execution, q3_provider)
    if (type(q3_provider) is not
            CPUQ3V83ActiveWetDryExnerAdapterV97 or
            q3_provider.native_factcr_runtime_sha256 !=
                execution.factcr_runtime_sha256 or
            q3_provider.native_q3_face_runtime_sha256 !=
                execution.q3_face_runtime_sha256 or
            q3_provider.native_factcr_threads !=
                execution.hydraulic_execution.native_threads):
        raise CoupledWetDryOwnerV97Error(
            "v97 outer Q3/native FactCr/face provider identity differs")
    _validate_runtimes(
        execution, native_runtime=native_runtime,
        density_projection_runtime=density_projection_runtime,
        turbulence_runtime=turbulence_runtime,
        scalar_fct_runtime=scalar_fct_runtime,
        scalar_diffusion_runtime=scalar_diffusion_runtime,
        scalar_boundary_runtime=scalar_boundary_runtime,
        settling_exchange_runtime=settling_exchange_runtime,
        hles_runtime=hles_runtime)
    if parent.accepted_updates >= MAX_ACCEPTED_UPDATES:
        raise CoupledWetDryOwnerV97Error(
            "v97 outer accepted update limit was reached")
    if not isinstance(q3_inputs, Mapping):
        raise CoupledWetDryOwnerV97Error(
            "v97 outer Q3 inputs must be a mapping")
    if _cancelled(cancelled):
        raise InterruptedError("v97 outer cancelled before hydraulic work")
    dt = float(dt_s)
    if (not np.isfinite(dt) or dt <= 0.0 or
            dt > execution.hydraulic_execution.base_execution.maximum_dt_s):
        raise CoupledWetDryOwnerV97Error(
            "v97 outer timestep is outside the hydraulic bound")
    water, mixture = _accepted_density(execution, parent)
    request = _hash({"scope_id": SCOPE_ID+":request",
                     "execution": execution.fingerprint,
                     "parent": parent.state_sha256, "dt_s": dt,
                     "bed": _digest(parent.bed_elevation_m),
                     "density": _digest(water, mixture)})
    hles_execution = _v92_execution(
        execution.active_execution).base_execution.hles_execution
    handoff = make_current_closure_handoff_v90(
        hles_execution, parent.hles_state,
        hydraulic_trial_request_sha256=request)
    v91_execution = _v92_execution(execution.active_execution).base_execution
    turbulence_prepared = turbulence_runtime.prepare(
        v91_execution, parent.hles_state, handoff, dt_s=dt,
        native_threads=execution.hydraulic_execution.native_threads)
    hydraulic = solve_wetdry_midpoint_trial_v97(
        execution.hydraulic_execution, parent.q,
        bed_elevation_m=parent.bed_elevation_m,
        accepted_water_density_kg_m3=water,
        accepted_mixture_density_kg_m3=mixture, dt_s=dt,
        native_runtime=native_runtime,
        density_projection_runtime=density_projection_runtime,
        shared_turbulence_runtime=turbulence_runtime,
        shared_turbulence_prepared=turbulence_prepared,
        cancelled=cancelled)
    evidence = make_wetdry_hydraulic_evidence_v97(
        v91_execution, _v92_state(parent.active_state),
        turbulence_prepared.cap_receipt,
        wetdry_execution=execution.hydraulic_execution,
        wetdry_trial=hydraulic,
        accepted_bed_elevation_m=parent.bed_elevation_m,
        accepted_water_density_kg_m3=water,
        accepted_mixture_density_kg_m3=mixture)
    fluid_left, fluid_right, prescribed_fluid, scalar = \
        _closed_scalar_inputs(
            execution, parent, hydraulic.endpoint_q, shear_velocity_m_s)
    provider_checkpoint = q3_provider.checkpoint()
    active_trial: ActiveSuspendedControlTrialV94 | None = None
    try:
        if _cancelled(cancelled):
            raise InterruptedError("v97 outer cancelled before scalar work")
        active_trial, scalar_evidence, factcr_evidence = \
            begin_active_suspended_wetdry_cpu_trial_v97(
                execution.active_execution, parent.active_state,
                parent.hles_state, handoff, q3_provider,
                hydraulic_evidence=evidence,
                reconstructed_left_fluid=fluid_left,
                reconstructed_right_fluid=fluid_right,
                prescribed_boundary_fluid=prescribed_fluid,
                v92_scalar_inputs=scalar, q3_inputs=dict(q3_inputs),
                scalar_fct_runtime=scalar_fct_runtime,
                scalar_diffusion_runtime=scalar_diffusion_runtime,
                scalar_boundary_runtime=scalar_boundary_runtime,
                settling_exchange_runtime=settling_exchange_runtime,
                native_scalar_threads=
                    execution.hydraulic_execution.native_threads,
                return_native_scalar_evidence=True,
                cancelled=(cancelled if callable(cancelled)
                           else lambda: bool(cancelled)))
        q3_face_evidence = q3_provider.native_q3_face_evidence_v97
        v92_trial = active_trial.base_v93_trial.base_v92_trial
        if (not np.array_equal(v92_trial.candidate.q,
                               hydraulic.endpoint_q) or
                not np.array_equal(
                    v92_trial.hydraulic_face_receipt.
                    accepted_water_flux_m2_s,
                    hydraulic.accepted_limited_face_water_flux_m2_s)):
            raise CoupledWetDryOwnerV97Error(
                "v97 outer scalar child changed accepted hydraulics")
        hles_arrays = _hles_inputs(
            execution, hydraulic.endpoint_q,
            hydraulic.accepted_limited_face_water_flux_m2_s,
            closure_chezy_mhalf_s)
        step = v92_trial.hydraulic_receipt.base_step_receipt
        hles_input = make_hles_input_receipt_v90(
            hles_execution, parent.hles_state,
            accepted_step_receipt=step, current_handoff=handoff,
            closure_interval_s=dt,
            accepted_face_normal_velocity_m_s=hles_arrays[0],
            active_face=hles_arrays[1], closure_depth_m=hles_arrays[2],
            closure_chezy_mhalf_s=hles_arrays[3],
            active_cell=hles_arrays[4])
        hles_trial, hles_evidence = hles_runtime.begin_trial(
            hles_execution, parent.hles_state,
            accepted_step_receipt=step, hles_input_receipt=hles_input,
            current_handoff=handoff,
            accepted_face_normal_velocity_m_s=hles_arrays[0],
            active_face=hles_arrays[1], closure_depth_m=hles_arrays[2],
            closure_chezy_mhalf_s=hles_arrays[3],
            active_cell=hles_arrays[4],
            native_threads=execution.hydraulic_execution.native_threads,
            cancelled=cancelled)
        chain = _hash({
            "previous": parent.accepted_chain_sha256,
            "hydraulic": evidence.evidence_sha256,
            "scalar": scalar_evidence.evidence_sha256,
            "factcr": _hash(factcr_evidence.__dict__),
            "q3_face": q3_face_evidence.output_sha256,
            "active": active_trial.trial_sha256,
            "native_hles": hles_evidence.result_sha256,
            "hles": hles_trial.trial_sha256})
        candidate = _wrap_state(
            execution, active_state=active_trial.candidate,
            hles_state=hles_trial.candidate,
            accepted_chain_sha256=chain)
        placeholder = CoupledWetDryCPUTrialV97(
            execution.fingerprint, parent.state_sha256, handoff, evidence,
            scalar_evidence, factcr_evidence, q3_face_evidence, active_trial,
            hles_trial, hles_evidence,
            hles_arrays, CoupledWetDryOwnerCountsV97(), candidate,
            BACKEND, "")
        trial = CoupledWetDryCPUTrialV97(
            **{**placeholder.__dict__,
               "trial_sha256": _trial_token(placeholder)})
        _validate_trial(execution, parent, q3_provider, trial)
        return trial
    except Exception as original_error:
        try:
            _restore_provider_after_failure(
                active_trial, q3_provider, provider_checkpoint)
        except Exception as cleanup_error:
            raise RuntimeError(
                "v97 outer begin failed and provider restore also failed") \
                from cleanup_error
        raise original_error


def _validate_native_hles(
    execution: CoupledWetDryCPUExecutionV97,
    parent: CoupledWetDryCPUStateV97,
    trial: CoupledWetDryCPUTrialV97,
) -> None:
    evidence = trial.native_hles_evidence
    hles_execution = _v92_execution(
        execution.active_execution).base_execution.hles_execution
    if (type(evidence) is not NativeHlesCandidateV97 or
            evidence.native_runtime_sha256 != execution.hles_runtime_sha256 or
            evidence.parent_state_sha256 != parent.hles_state.state_sha256 or
            evidence.execution_fingerprint != hles_execution.fingerprint or
            evidence.native_threads !=
                execution.hydraulic_execution.native_threads or
            evidence.filter_owner_count != 1 or
            evidence.reconstruction_owner_count != 1 or
            evidence.protke_owner_count != 1 or
            evidence.detvic_owner_count != 1 or
            evidence.python_numerical_cell_face_loop_count != 0 or
            evidence.state_or_clock_publication_count != 0 or
            evidence.native_candidate_launch_count != 1):
        raise CoupledWetDryOwnerV97Error(
            "v97 outer native HLES evidence identity differs")
    inputs = trial.hles_inputs
    if not isinstance(inputs, tuple) or len(inputs) != 5:
        raise CoupledWetDryOwnerV97Error(
            "v97 outer HLES input inventory differs")
    input_sha = _native_hles._hash(dict(
        execution=hles_execution.fingerprint,
        parent=parent.hles_state.state_sha256,
        fields=_native_hles._digest(*inputs),
        interval=trial.hydraulic_evidence.wetdry_trial.dt_s))
    values = evidence.values
    if not isinstance(values, tuple) or len(values) != 13:
        raise CoupledWetDryOwnerV97Error(
            "v97 outer native HLES output inventory differs")
    result_sha = _native_hles._hash(dict(
        runtime=evidence.native_runtime_sha256, inputs=input_sha,
        coefficient=float(values[0]), outputs=_native_hles._digest(*values[1:]),
        native_threads=evidence.native_threads))
    hles_trial = trial.hles_trial
    comparable = (
        hles_trial.filtered_face_normal_velocity_m_s,
        hles_trial.fluctuating_face_normal_velocity_m_s,
        hles_trial.reconstructed_mean_velocity_m_s,
        hles_trial.reconstructed_fluctuating_velocity_m_s,
        hles_trial.reconstruction_complete,
        hles_trial.boundary_cell, hles_trial.detvic_active_cell,
        hles_trial.ptke_s2, hles_trial.hles_vicuv_m2_s,
        hles_trial.elder_vicuv_m2_s,
        hles_trial.output_receipt.candidate_dynamic_vicuv_m2_s,
        hles_trial.output_receipt.candidate_dynamic_dicuv_m2_s,
    )
    if (evidence.inputs_sha256 != input_sha or
            evidence.result_sha256 != result_sha or
            hles_trial.filter_coefficient != float(values[0]) or
            any(not np.array_equal(a, b)
                for a, b in zip(comparable, values[1:]))):
        raise CoupledWetDryOwnerV97Error(
            "v97 outer native HLES result differs from v90 candidate")


def _validate_trial(
    execution: CoupledWetDryCPUExecutionV97,
    parent: CoupledWetDryCPUStateV97,
    q3_provider: object,
    trial: CoupledWetDryCPUTrialV97,
) -> None:
    validate_coupled_wetdry_cpu_state_v97(parent, execution, q3_provider)
    if (type(trial) is not CoupledWetDryCPUTrialV97 or
            trial.execution_fingerprint != execution.fingerprint or
            trial.parent_state_sha256 != parent.state_sha256 or
            trial.backend != BACKEND or
            trial.trial_sha256 != _trial_token(trial) or
            trial.owner_counts != CoupledWetDryOwnerCountsV97()):
        raise CoupledWetDryOwnerV97Error(
            "v97 outer trial is stale, mutated, or cross-parent")
    v92_parent = _v92_state(parent.active_state)
    v92_trial = trial.active_trial.base_v93_trial.base_v92_trial
    cap = v92_trial.cap_receipt
    validate_wetdry_hydraulic_evidence_v97(
        trial.hydraulic_evidence,
        _v92_execution(execution.active_execution).base_execution,
        v92_parent, cap,
        midpoint_q=v92_trial.midpoint_q,
        accepted_q=v92_trial.candidate.q,
        momentum_rate_m2_s2=v92_trial.momentum_rate_m2_s2,
        momentum_face_flux_m4_s2=v92_trial.momentum_face_flux_m4_s2)
    hydraulic = trial.hydraulic_evidence.wetdry_trial
    if (not np.array_equal(v92_trial.candidate.q, hydraulic.endpoint_q) or
            not np.array_equal(
                v92_trial.hydraulic_face_receipt.
                accepted_water_flux_m2_s,
                hydraulic.accepted_limited_face_water_flux_m2_s)):
        raise CoupledWetDryOwnerV97Error(
            "v97 outer accepted hydraulic image differs")
    scalar = trial.native_scalar_evidence
    if (type(scalar) is not NativeCombinedScalarEvidenceV96 or
            scalar.fct_runtime_sha256 !=
                execution.scalar_fct_runtime_sha256 or
            scalar.diffusion_runtime_sha256 !=
                execution.scalar_diffusion_runtime_sha256 or
            scalar.scalar_boundary_runtime_sha256 !=
                execution.scalar_boundary_runtime_sha256 or
            not isinstance(scalar.scalar_boundary_result_sha256, str) or
            len(scalar.scalar_boundary_result_sha256) != 64 or
            scalar.scalar_boundary_launch_count != 1 or
            scalar.settling_exchange_runtime_sha256 !=
                execution.settling_exchange_runtime_sha256 or
            not isinstance(scalar.settling_exchange_result_sha256, str) or
            len(scalar.settling_exchange_result_sha256) != 64 or
            scalar.settling_launch_count != 1 or
            scalar.exchange_dry_transfer_launch_count != 1 or
            scalar.suspended_trial_sha256 !=
                v92_trial.suspended_trial.trial_sha256 or
            scalar.fluid_result_sha256 !=
                v92_trial.fluid_transport.result_sha256 or
            scalar.fct_launch_count != 1 or
            scalar.diffusion_launch_count != 1 or
            scalar.python_numerical_cell_face_loop_count != 0 or
            scalar.state_or_clock_publication_count != 0 or
            scalar.evidence_sha256 != _scalar_evidence_token(scalar)):
        raise CoupledWetDryOwnerV97Error(
            "v97 outer native scalar evidence differs")
    factcr = trial.native_factcr_evidence
    if (type(factcr) is not NativeFactCrEvidenceV97 or
            factcr.runtime_sha256 != execution.factcr_runtime_sha256 or
            factcr.execution_fingerprint !=
                execution.active_execution.factcr_execution.fingerprint or
            factcr.v83_receipt_sha256 !=
                trial.active_trial.producer_receipt.receipt_sha256 or
            factcr.native_threads !=
                execution.hydraulic_execution.native_threads or
            factcr.native_producer_batch_owner_count != 1 or
            factcr.python_numerical_cell_face_loop_count != 0 or
            factcr.scalar_or_clock_publication_count != 0 or
            not isinstance(factcr.input_sha256, str) or
            len(factcr.input_sha256) != 64 or
            not isinstance(factcr.output_sha256, str) or
            len(factcr.output_sha256) != 64):
        raise CoupledWetDryOwnerV97Error(
            "v97 outer native FactCr evidence differs")
    q3_native = trial.native_q3_face_evidence
    q3_face = trial.active_trial.base_v93_trial.q3_open_trial.face_trial
    try:
        provider_q3_native = q3_provider.native_q3_face_evidence_v97
    except (AttributeError, ValueError) as error:
        raise CoupledWetDryOwnerV97Error(
            "v97 outer native Q3 provider evidence is unavailable") from error
    if (type(q3_native) is not NativeQ3FaceInventoryResultV97 or
            q3_native is not provider_q3_native or
            q3_native.runtime_sha256 != execution.q3_face_runtime_sha256 or
            q3_native.method_fingerprint !=
                q3_provider.binding.bed_slope_execution["fingerprint"] or
            q3_native.native_threads !=
                execution.hydraulic_execution.native_threads or
            q3_native.native_q3_face_batch_owner_count != 1 or
            q3_native.python_numerical_cell_face_loop_count != 0 or
            q3_native.accepted_bed_or_clock_publication_count != 0 or
            q3_native.output_sha256 != _native_q3_digest(
                q3_native.face_flux_kg_m_s,
                q3_native.cell_delta_kg_m2, q3_native.availability,
                q3_native.named_boundary_transfer_kg,
                q3_native.named_boundary_absolute_kg,
                q3_native.depletion_roundoff_positive_in_kg,
                q3_native.classwise_residual_kg) or
            not np.array_equal(q3_face.face_flux_kg_m_s,
                               q3_native.face_flux_kg_m_s) or
            not np.array_equal(q3_face.cell_delta_kg_m2,
                               q3_native.cell_delta_kg_m2) or
            not np.array_equal(q3_face.inventory_fill,
                               q3_native.availability) or
            not np.array_equal(q3_face.named_boundary_transfer_kg,
                               q3_native.named_boundary_transfer_kg) or
            not np.array_equal(q3_face.named_boundary_absolute_kg,
                               q3_native.named_boundary_absolute_kg) or
            not np.array_equal(
                q3_face.depletion_roundoff_positive_in_kg,
                q3_native.depletion_roundoff_positive_in_kg) or
            not np.array_equal(q3_face.conservation_residual_kg,
                               q3_native.classwise_residual_kg)):
        raise CoupledWetDryOwnerV97Error(
            "v97 outer native Q3 face/inventory evidence differs")
    _v94._validate_trial(
        execution.active_execution, parent.active_state,
        parent.hles_state, trial.current_handoff, trial.active_trial)
    hles_execution = _v92_execution(
        execution.active_execution).base_execution.hles_execution
    if rollback_hles_trial_v90(
            hles_execution, parent.hles_state,
            trial.hles_trial) is not parent.hles_state:
        raise CoupledWetDryOwnerV97Error(
            "v97 outer HLES rollback validation differs")
    _validate_native_hles(execution, parent, trial)
    expected_bed = _bed_from_mass(
        execution, trial.active_trial.candidate.base_state.bed_mass_kg_m2)
    if (trial.candidate.active_state is not trial.active_trial.candidate or
            trial.candidate.hles_state is not trial.hles_trial.candidate or
            trial.candidate.accepted_updates != parent.accepted_updates+1 or
            trial.candidate.time_s !=
                parent.time_s+hydraulic.dt_s or
            not np.array_equal(trial.candidate.bed_elevation_m,
                               expected_bed) or
            not np.array_equal(trial.candidate.free_surface_elevation_m,
                               trial.candidate.q[0]+expected_bed)):
        raise CoupledWetDryOwnerV97Error(
            "v97 outer candidate clock or bed feedback differs")
    expected_chain = _hash({
        "previous": parent.accepted_chain_sha256,
        "hydraulic": trial.hydraulic_evidence.evidence_sha256,
        "scalar": trial.native_scalar_evidence.evidence_sha256,
        "factcr": _hash(trial.native_factcr_evidence.__dict__),
        "q3_face": trial.native_q3_face_evidence.output_sha256,
        "active": trial.active_trial.trial_sha256,
        "native_hles": trial.native_hles_evidence.result_sha256,
        "hles": trial.hles_trial.trial_sha256})
    rebuilt = _wrap_state(
        execution, active_state=trial.active_trial.candidate,
        hles_state=trial.hles_trial.candidate,
        accepted_chain_sha256=expected_chain)
    if (trial.candidate.accepted_chain_sha256 != expected_chain or
            trial.candidate.state_sha256 != rebuilt.state_sha256):
        raise CoupledWetDryOwnerV97Error(
            "v97 outer candidate chain differs")


def commit_coupled_wetdry_cpu_trial_v97(
    execution: CoupledWetDryCPUExecutionV97,
    parent: CoupledWetDryCPUStateV97,
    q3_provider: object,
    trial: CoupledWetDryCPUTrialV97,
) -> CoupledWetDryCPUStateV97:
    checkpoint = trial.active_trial.base_v93_trial.q3_parent_checkpoint
    try:
        _validate_trial(execution, parent, q3_provider, trial)
        hles = commit_hles_trial_v90(
            _v92_execution(execution.active_execution).base_execution.
            hles_execution, parent.hles_state, trial.hles_trial)
        if hles is not trial.hles_trial.candidate:
            raise CoupledWetDryOwnerV97Error(
                "v97 outer HLES commit substituted its candidate")
    except Exception as original_error:
        try:
            _restore_provider_after_failure(
                getattr(trial, "active_trial", None),
                q3_provider, checkpoint)
        except Exception as cleanup_error:
            raise RuntimeError(
                "v97 outer commit failed and restore also failed") \
                from cleanup_error
        raise original_error
    commit_active_suspended_control_trial_v94b(
        execution.active_execution, parent.active_state,
        parent.hles_state, trial.current_handoff, trial.active_trial)
    return trial.candidate


def rollback_coupled_wetdry_cpu_trial_v97(
    execution: CoupledWetDryCPUExecutionV97,
    parent: CoupledWetDryCPUStateV97,
    q3_provider: object,
    trial: CoupledWetDryCPUTrialV97,
) -> CoupledWetDryCPUStateV97:
    _validate_trial(execution, parent, q3_provider, trial)
    if rollback_hles_trial_v90(
            _v92_execution(execution.active_execution).base_execution.
            hles_execution, parent.hles_state,
            trial.hles_trial) is not parent.hles_state:
        raise CoupledWetDryOwnerV97Error(
            "v97 outer rollback changed HLES parent")
    active = rollback_active_suspended_control_trial_v94b(
        execution.active_execution, parent.active_state,
        parent.hles_state, trial.current_handoff, trial.active_trial)
    if active is not parent.active_state:
        raise CoupledWetDryOwnerV97Error(
            "v97 outer rollback changed active parent")
    return parent


def cancel_coupled_wetdry_cpu_trial_v97(*args, **kwargs):
    return rollback_coupled_wetdry_cpu_trial_v97(*args, **kwargs)


def coupled_wetdry_cpu_checkpoint_v97(
    execution: CoupledWetDryCPUExecutionV97,
    state: CoupledWetDryCPUStateV97,
    q3_provider: object,
) -> dict[str, object]:
    validate_coupled_wetdry_cpu_state_v97(state, execution, q3_provider)
    hles_execution = _v92_execution(
        execution.active_execution).base_execution.hles_execution
    body = {
        "schema": SCHEMA, "scope_id": CHECKPOINT_SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "execution_fingerprint": execution.fingerprint,
        "active_checkpoint": checkpoint_active_suspended_control_state_v94b(
            execution.active_execution, state.active_state, q3_provider),
        "hles_checkpoint": hles_checkpoint_v90(
            hles_execution, state.hles_state),
        "bed_elevation_m": state.bed_elevation_m.tolist(),
        "free_surface_elevation_m":
            state.free_surface_elevation_m.tolist(),
        "accepted_chain_sha256": state.accepted_chain_sha256,
        "state_sha256": state.state_sha256,
    }
    return {**body, "fingerprint": _hash(body)}


def load_coupled_wetdry_cpu_checkpoint_v97(
    execution: CoupledWetDryCPUExecutionV97,
    document: Mapping[str, object],
    q3_provider: object,
) -> CoupledWetDryCPUStateV97:
    keys = {"schema", "scope_id", "semantic_id",
            "execution_fingerprint", "active_checkpoint",
            "hles_checkpoint", "bed_elevation_m",
            "free_surface_elevation_m", "accepted_chain_sha256",
            "state_sha256", "fingerprint"}
    if not isinstance(document, Mapping) or set(document) != keys:
        raise CoupledWetDryOwnerV97Error(
            "v97 outer checkpoint inventory differs")
    body = {key: value for key, value in document.items()
            if key != "fingerprint"}
    if (document["fingerprint"] != _hash(body) or
            document["schema"] != SCHEMA or
            document["scope_id"] != CHECKPOINT_SCOPE_ID or
            document["semantic_id"] != SEMANTIC_ID or
            document["execution_fingerprint"] != execution.fingerprint):
        raise CoupledWetDryOwnerV97Error(
            "v97 outer checkpoint identity differs")
    provider_checkpoint = q3_provider.checkpoint()
    try:
        hles = load_hles_checkpoint_v90(
            _v92_execution(execution.active_execution).base_execution.
            hles_execution, document["hles_checkpoint"])
        active = load_active_suspended_control_checkpoint_v94b(
            execution.active_execution, document["active_checkpoint"],
            q3_provider)
        state = _wrap_state(
            execution, active_state=active, hles_state=hles,
            accepted_chain_sha256=document["accepted_chain_sha256"],
            q3_provider=q3_provider)
        saved_bed = _array(document["bed_elevation_m"],
                           (execution.cell_count,), "checkpoint bed")
        saved_eta = _array(document["free_surface_elevation_m"],
                           (execution.cell_count,), "checkpoint surface")
        if (not np.array_equal(saved_bed, state.bed_elevation_m) or
                not np.array_equal(saved_eta,
                                   state.free_surface_elevation_m) or
                document["state_sha256"] != state.state_sha256):
            raise CoupledWetDryOwnerV97Error(
                "v97 outer checkpoint state differs")
        return state
    except Exception as original_error:
        try:
            q3_provider.restore_checkpoint(provider_checkpoint)
        except Exception as cleanup_error:
            raise RuntimeError(
                "v97 outer checkpoint failed and restore also failed") \
                from cleanup_error
        raise original_error


__all__ = [
    "BACKEND", "CHECKPOINT_SCOPE_ID", "GEOMETRY_POLICY",
    "MAX_ACCEPTED_UPDATES", "SEMANTIC_ID", "SCOPE_ID",
    "CoupledWetDryCPUExecutionV97", "CoupledWetDryCPUStateV97",
    "CoupledWetDryCPUTrialV97", "CoupledWetDryOwnerCountsV97",
    "CoupledWetDryOwnerV97Error", "begin_coupled_wetdry_cpu_trial_v97",
    "cancel_coupled_wetdry_cpu_trial_v97",
    "commit_coupled_wetdry_cpu_trial_v97",
    "coupled_wetdry_cpu_checkpoint_v97",
    "execution_document_coupled_wetdry_v97",
    "load_coupled_wetdry_cpu_checkpoint_v97",
    "resolve_coupled_wetdry_cpu_execution_v97",
    "resolve_coupled_wetdry_cpu_state_v97",
    "rollback_coupled_wetdry_cpu_trial_v97",
    "validate_coupled_wetdry_cpu_state_v97",
]
