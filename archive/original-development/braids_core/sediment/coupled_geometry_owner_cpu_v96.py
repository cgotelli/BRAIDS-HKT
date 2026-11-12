"""Atomic v96 CPU owner with accepted bed feedback and spatial water EOS.

Unlike the bounded flat-bed v96 proof, this additive owner derives bed
elevation from the accepted v94b Exner inventory and supplies that bed to the
next native augmented-f-wave midpoint.  The water state remains conservative:
Exner changes bed elevation after the hydraulic endpoint without changing
``h``; therefore ``eta=h+z`` shifts by the bed change and the geometry water
volume ledger is exactly zero.  This deliberate split policy is authenticated
and can be compared with Delft3D's wet-cell stage policy in later validation.

The owner supports all four source-legal v94/v94b SusCor/NeglectEntrainment
routes on closed, fully-wet fixed grids.  Spatial v92 water density is active even when the
sediment contribution to DensIn is disabled.  One prepared native v91 cap is
shared by every midpoint evaluation and the scalar phase, so lagged momentum
diffusion and scalar diffusion are both active without a second hydraulic
iterate or a duplicate source application.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Callable, Mapping

import numpy as np

from . import active_suspended_control_v94 as _v94
from .active_suspended_control_v94 import (
    ActiveSuspendedControlExecutionV94,
    ActiveSuspendedControlStateV94,
    ActiveSuspendedControlTrialV94,
)
from .active_suspended_control_v94b import (
    SEMANTIC_ID_V94B,
    begin_active_suspended_control_cpu_trial_v94b,
    checkpoint_active_suspended_control_state_v94b,
    commit_active_suspended_control_trial_v94b,
    execution_document_v94b,
    load_active_suspended_control_checkpoint_v94b,
    rollback_active_suspended_control_trial_v94b,
    validate_active_suspended_control_state_v94b,
)
from .active_suspended_preconverged_cpu_v96 import (
    begin_active_suspended_preconverged_cpu_trial_v96,
)
from .coupled_densin_cpu_v96 import (
    V94bAcceptedDensityReceiptV96,
    _cancelled,
    _closed_scalar_inputs,
    _face_normal_velocity,
    _grid_execution,
    _provider_is_closed,
    _residual_norm,
    _restore_provider_after_failure,
    _v92_execution,
    _v92_state,
)
from .coupled_geometry_native_cpu_v96 import (
    DENSITY_WATER_PLUS_SEDIMENT,
    CoupledGeometryNativeCPUExecutionV96,
    NativeCoupledGeometryRateResultV96,
    NativeCoupledGeometryRuntimeV96,
    execution_document_coupled_geometry_native_v96,
)
from .densin_fwave_v85 import (
    LOCKED_UPSTREAM_COMMIT,
    delft_kmax1_face_acceleration_v85,
    mixture_density_kg_m3_v85,
    project_face_acceleration_v85,
)
from .hles_producer_v90 import (
    CurrentClosureHandoffV90,
    HlesProducerStateV90,
    HlesProducerTrialV90,
    begin_hles_cpu_trial_v90,
    commit_hles_trial_v90,
    hles_checkpoint_v90,
    load_hles_checkpoint_v90,
    make_current_closure_handoff_v90,
    make_hles_input_receipt_v90,
    rollback_hles_trial_v90,
    validate_hles_producer_state_v90,
)
from .shared_turbulence_native_cpu_v96 import (
    NativeMomentumRateV96,
    NativeSharedTurbulenceRuntimeV96,
    PreparedNativeTurbulenceV96,
)
from . import shared_turbulence_native_cpu_v96 as _native_v91
from .scalar_fct_native_cpu_v96 import NativeScalarFCTRuntimeV96
from .scalar_diffusion_native_cpu_v96 import (
    NativeScalarDiffusionRuntimeV96,
)
from . import scalar_diffusion_native_cpu_v96 as _native_scalar_diffusion
from .transported_fluid_native_scalar_cpu_v96 import (
    NativeCombinedScalarEvidenceV96,
    _evidence_token as _native_scalar_evidence_token,
)


SCHEMA = 1
SCOPE_ID = "braids-q4-coupled-geometry-owner-cpu-v96-v1"
SEMANTIC_ID = SCOPE_ID + ":accepted-bed-spatial-eos-atomic-owner"
CHECKPOINT_SCOPE_ID = SCOPE_ID + ":checkpoint"
BACKEND = "native_cpu_openmp"
MAX_ACCEPTED_UPDATES = 4
GEOMETRY_POLICY = "preserve_h_shift_eta_after_exner"
_SHA256 = __import__("re").compile(r"[0-9a-f]{64}\Z")


class CoupledGeometryOwnerV96Error(ValueError):
    """The v96 geometry execution, receipt, state, or trial is invalid."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError, OverflowError) as error:
        raise CoupledGeometryOwnerV96Error(
            "v96 geometry value is not finite canonical JSON") from error


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _digest(*values: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(value)
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(_canonical(list(array.shape)))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _array(value: object, shape: tuple[int, ...], name: str, *,
           positive: bool = False, nonnegative: bool = False) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != shape or raw.dtype.kind not in "fiub":
        raise CoupledGeometryOwnerV96Error(
            f"v96 geometry {name} must be numeric with shape {shape}")
    result = np.ascontiguousarray(raw, dtype=np.float64)
    if not np.isfinite(result).all():
        raise CoupledGeometryOwnerV96Error(
            f"v96 geometry {name} must be finite")
    if positive and np.any(result <= 0.0):
        raise CoupledGeometryOwnerV96Error(
            f"v96 geometry {name} must be positive")
    if nonnegative and np.any(result < 0.0):
        raise CoupledGeometryOwnerV96Error(
            f"v96 geometry {name} must be nonnegative")
    result.setflags(write=False)
    return result


def _finite(value: object, name: str, *, positive: bool = False) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(float(value))):
        raise CoupledGeometryOwnerV96Error(
            f"v96 geometry {name} must be finite numeric data")
    result = float(value)
    if positive and result <= 0.0:
        raise CoupledGeometryOwnerV96Error(
            f"v96 geometry {name} must be positive")
    return result


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise CoupledGeometryOwnerV96Error(
            f"v96 geometry {name} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True)
class CoupledGeometryCPUExecutionV96:
    active_execution: ActiveSuspendedControlExecutionV94
    hydraulic_execution: CoupledGeometryNativeCPUExecutionV96
    turbulence_native_receipt_sha256: str
    turbulence_native_source_sha256: str
    scalar_fct_native_receipt_sha256: str
    scalar_fct_native_source_sha256: str
    scalar_diffusion_native_receipt_sha256: str
    scalar_diffusion_native_source_sha256: str
    initial_bed_elevation_m: np.ndarray
    reference_bed_mass_kg_m2: np.ndarray
    dry_bulk_density_kg_m3: np.ndarray
    bed_porosity_by_fraction: np.ndarray
    geometry_policy: str
    locked_delft_commit: str
    locked_delft_source_sha256: tuple[tuple[str, str], ...]
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


def _topology_matches(value: CoupledGeometryCPUExecutionV96) -> bool:
    grid = _grid_execution(value.active_execution)
    base = value.hydraulic_execution.base_execution
    return all(np.array_equal(left, right) for left, right in (
        (grid.owner, base.owner), (grid.neighbour, base.neighbour),
        (grid.normal, base.normal), (grid.face_length_m, base.face_length_m),
        (grid.centre_distance_m, base.face_distance_m),
        (grid.cell_area_m2, base.cell_area_m2),
        (grid.cell_face_offset, base.cell_face_offset),
        (grid.cell_faces, base.cell_faces),
        (grid.cell_face_sign, base.cell_face_sign),
    ))


def _execution_payload(value: CoupledGeometryCPUExecutionV96) -> dict:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "semantic_id": value.semantic_id,
        "backend": BACKEND,
        "active_v94b_execution": execution_document_v94b(
            value.active_execution),
        "native_geometry_execution":
            execution_document_coupled_geometry_native_v96(
                value.hydraulic_execution),
        "native_turbulence_receipt_sha256":
            value.turbulence_native_receipt_sha256,
        "native_turbulence_source_sha256":
            value.turbulence_native_source_sha256,
        "native_scalar_fct_receipt_sha256":
            value.scalar_fct_native_receipt_sha256,
        "native_scalar_fct_source_sha256":
            value.scalar_fct_native_source_sha256,
        "native_scalar_diffusion_receipt_sha256":
            value.scalar_diffusion_native_receipt_sha256,
        "native_scalar_diffusion_source_sha256":
            value.scalar_diffusion_native_source_sha256,
        "initial_bed_sha256": _digest(value.initial_bed_elevation_m),
        "reference_bed_mass_sha256": _digest(
            value.reference_bed_mass_kg_m2),
        "dry_bulk_density_sha256": _digest(value.dry_bulk_density_kg_m3),
        "bed_porosity_sha256": _digest(value.bed_porosity_by_fraction),
        "dry_bulk_density_origin": "rhosol*(1-porosity)",
        "bed_mapping": "z0+sum((M-M0)/rho_dry_by_fraction)",
        "geometry_policy": value.geometry_policy,
        "geometry_water_volume_ledger": "exact_zero",
        "hydraulic_bed_timing": "accepted_parent_bed_used_current_step",
        "exner_bed_timing": "candidate_bed_visible_next_step",
        "spatial_water_eos": "accepted_v92_field",
        "lagged_v91_momentum": "native_openmp_each_midpoint_residual",
        "lagged_v91_scalar": "same_prepared_cap_consumed_once",
        "execution_partition": (
            "native_openmp_hydraulics_v91_momentum_scalar_fct_diffusion;"
            "host_python_boundaries_settling_exchange_q3_exner"),
        "nested_v91_v92_receipt_role": (
            "compatibility_oracle_only;outer_native_chain_authoritative"),
        "native_hydraulic_owner_count": 1,
        "bed_geometry_owner_count": 1,
        "scalar_lifecycle_owner_count": 1,
        "authoritative_exner_owner_count": 1,
        "hles_clock_owner_count": 1,
        "locked_delft_commit": value.locked_delft_commit,
        "locked_delft_source_sha256": dict(
            value.locked_delft_source_sha256),
        "accepted_update_limit": MAX_ACCEPTED_UPDATES,
        "product_admission": False,
    }


def resolve_coupled_geometry_cpu_execution_v96(
    active_execution: ActiveSuspendedControlExecutionV94,
    hydraulic_execution: CoupledGeometryNativeCPUExecutionV96, *,
    turbulence_runtime: NativeSharedTurbulenceRuntimeV96,
    scalar_fct_runtime: NativeScalarFCTRuntimeV96,
    scalar_diffusion_runtime: NativeScalarDiffusionRuntimeV96,
    initial_bed_elevation_m: object,
    reference_bed_mass_kg_m2: object,
    dry_bulk_density_kg_m3: object,
    bed_porosity_by_fraction: object,
) -> CoupledGeometryCPUExecutionV96:
    execution_document_v94b(active_execution)
    native_document = execution_document_coupled_geometry_native_v96(
        hydraulic_execution)
    if (active_execution.semantic_id not in
            (_v94.SEMANTIC_ID, SEMANTIC_ID_V94B) or
            active_execution.backend != "cpu"):
        raise CoupledGeometryOwnerV96Error(
            "v96 geometry requires an authenticated CPU v94/v94b route")
    if type(turbulence_runtime) is not NativeSharedTurbulenceRuntimeV96:
        raise CoupledGeometryOwnerV96Error(
            "v96 geometry requires the native v91 turbulence runtime")
    turbulence_receipt = dict(turbulence_runtime.receipt)
    turbulence_sha = turbulence_receipt.pop("receipt_sha256", None)
    if (not isinstance(turbulence_sha, str) or
            _SHA256.fullmatch(turbulence_sha) is None or
            turbulence_sha != _native_v91._hash(turbulence_receipt)):
        raise CoupledGeometryOwnerV96Error(
            "v96 geometry native turbulence receipt differs")
    turbulence_source_sha = turbulence_runtime.receipt.get(
        "native_source_sha256")
    if (not isinstance(turbulence_source_sha, str) or
            _SHA256.fullmatch(turbulence_source_sha) is None):
        raise CoupledGeometryOwnerV96Error(
            "v96 geometry native turbulence source identity differs")
    if type(scalar_fct_runtime) is not NativeScalarFCTRuntimeV96:
        raise CoupledGeometryOwnerV96Error(
            "v96 geometry requires the native scalar FCT runtime")
    scalar_fct_runtime._validate_runtime()
    scalar_fct_sha = scalar_fct_runtime.runtime_sha256
    scalar_fct_source_sha = scalar_fct_runtime.receipt.get(
        "native_source_sha256")
    if (_SHA256.fullmatch(scalar_fct_sha) is None or
            not isinstance(scalar_fct_source_sha, str) or
            _SHA256.fullmatch(scalar_fct_source_sha) is None):
        raise CoupledGeometryOwnerV96Error(
            "v96 geometry native scalar FCT identity differs")
    if type(scalar_diffusion_runtime) is not NativeScalarDiffusionRuntimeV96:
        raise CoupledGeometryOwnerV96Error(
            "v96 geometry requires the native scalar diffusion runtime")
    diffusion_receipt = dict(scalar_diffusion_runtime.receipt)
    scalar_diffusion_sha = diffusion_receipt.pop("receipt_sha256", None)
    scalar_diffusion_source_sha = scalar_diffusion_runtime.receipt.get(
        "source_sha256")
    if (not isinstance(scalar_diffusion_sha, str) or
            scalar_diffusion_sha !=
                _native_scalar_diffusion._hash(diffusion_receipt) or
            not isinstance(scalar_diffusion_source_sha, str) or
            _SHA256.fullmatch(scalar_diffusion_source_sha) is None or
            scalar_diffusion_source_sha !=
                _native_scalar_diffusion._sha(
                    _native_scalar_diffusion.SOURCE)):
        raise CoupledGeometryOwnerV96Error(
            "v96 geometry native scalar diffusion identity differs")
    v92 = _v92_execution(active_execution)
    v78 = v92.base_execution.base_execution
    profile = v78.base_execution.profile
    base = hydraulic_execution.base_execution
    if (hydraulic_execution.cell_count != active_execution.cell_count or
            hydraulic_execution.face_count != v92.face_count or
            base.fraction_count != active_execution.fraction_count or
            base.hydraulic_execution_sha256 != v78.hydraulic_execution_sha256 or
            not np.array_equal(base.solid_density_kg_m3,
                               profile.sediment_density_kg_m3) or
            base.hydraulic_reference_density_kg_m3 !=
                profile.water_density_kg_m3 or
            base.gravity_m_s2 != profile.gravity_m_s2 or
            base.dry_tolerance_m != v78.base_execution.dry_tolerance_m):
        raise CoupledGeometryOwnerV96Error(
            "v96 geometry hydraulic fractions or constants differ")
    nc, ng = active_execution.cell_count, active_execution.fraction_count
    bed = _array(initial_bed_elevation_m, (nc,), "initial bed")
    reference = _array(reference_bed_mass_kg_m2, (ng, nc),
                       "reference bed mass", nonnegative=True)
    bulk = _array(dry_bulk_density_kg_m3, (ng,), "dry bulk density",
                  positive=True)
    porosity = _array(bed_porosity_by_fraction, (ng,), "bed porosity",
                      nonnegative=True)
    if (np.any(porosity >= 1.0) or
            not np.allclose(
                bulk,
                np.asarray(profile.sediment_density_kg_m3, dtype=np.float64)*
                (1.0-porosity), rtol=8*np.finfo(float).eps, atol=0.0)):
        raise CoupledGeometryOwnerV96Error(
            "v96 geometry dry bulk density must equal rhosol*(1-porosity)")
    placeholder = CoupledGeometryCPUExecutionV96(
        active_execution, hydraulic_execution, turbulence_sha,
        turbulence_source_sha, scalar_fct_sha, scalar_fct_source_sha,
        scalar_diffusion_sha, scalar_diffusion_source_sha,
        bed, reference, bulk, porosity,
        GEOMETRY_POLICY, LOCKED_UPSTREAM_COMMIT,
        tuple(hydraulic_execution.locked_delft_source_sha256),
        SEMANTIC_ID, "")
    if not _topology_matches(placeholder):
        raise CoupledGeometryOwnerV96Error(
            "v96 geometry native and v94b topologies differ")
    # Keep the local variable intentional: constructing the outer execution
    # revalidates the native locked-source identity before hashing it.
    if native_document["locked_delft_commit"] != LOCKED_UPSTREAM_COMMIT:
        raise CoupledGeometryOwnerV96Error(
            "v96 geometry locked Delft identity differs")
    return CoupledGeometryCPUExecutionV96(
        **{**placeholder.__dict__,
           "fingerprint": _fingerprint(_execution_payload(placeholder))})


def execution_document_coupled_geometry_owner_v96(
    value: CoupledGeometryCPUExecutionV96,
) -> dict:
    if type(value) is not CoupledGeometryCPUExecutionV96:
        raise CoupledGeometryOwnerV96Error(
            "v96 geometry execution object differs")
    payload = _execution_payload(value)
    if (value.semantic_id != SEMANTIC_ID or
            value.active_execution.semantic_id not in
                (_v94.SEMANTIC_ID, SEMANTIC_ID_V94B) or
            value.geometry_policy != GEOMETRY_POLICY or
            value.locked_delft_commit != LOCKED_UPSTREAM_COMMIT or
            value.locked_delft_source_sha256 !=
                value.hydraulic_execution.locked_delft_source_sha256 or
            _SHA256.fullmatch(value.turbulence_native_receipt_sha256) is None or
            _SHA256.fullmatch(value.turbulence_native_source_sha256) is None or
            _SHA256.fullmatch(value.scalar_fct_native_receipt_sha256) is None or
            _SHA256.fullmatch(value.scalar_fct_native_source_sha256) is None or
            _SHA256.fullmatch(
                value.scalar_diffusion_native_receipt_sha256) is None or
            _SHA256.fullmatch(
                value.scalar_diffusion_native_source_sha256) is None or
            not _topology_matches(value) or
            value.fingerprint != _fingerprint(payload)):
        raise CoupledGeometryOwnerV96Error(
            "v96 geometry execution fingerprint differs")
    return {**payload, "fingerprint": value.fingerprint}


def _bed_from_mass(execution: CoupledGeometryCPUExecutionV96,
                   bed_mass_kg_m2: object) -> np.ndarray:
    mass = _array(bed_mass_kg_m2,
                  (execution.fraction_count, execution.cell_count),
                  "accepted bed mass", nonnegative=True)
    result = np.ascontiguousarray(
        execution.initial_bed_elevation_m+
        np.sum((mass-execution.reference_bed_mass_kg_m2)/
               execution.dry_bulk_density_kg_m3[:, None], axis=0))
    if not np.isfinite(result).all():
        raise CoupledGeometryOwnerV96Error(
            "v96 geometry derived bed is non-finite")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class CoupledGeometryCPUStateV96:
    execution_fingerprint: str
    active_state: ActiveSuspendedControlStateV94
    hles_state: HlesProducerStateV90
    bed_elevation_m: np.ndarray
    free_surface_elevation_m: np.ndarray
    geometry_water_volume_ledger_m3: float
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


def _state_token(value: CoupledGeometryCPUStateV96) -> str:
    return _fingerprint({
        "scope_id": SCOPE_ID + ":state",
        "execution_fingerprint": value.execution_fingerprint,
        "active_v94b_state_sha256": value.active_state.state_sha256,
        "hles_state_sha256": value.hles_state.state_sha256,
        "geometry_sha256": _digest(value.bed_elevation_m,
                                   value.free_surface_elevation_m),
        "geometry_water_volume_ledger_m3":
            value.geometry_water_volume_ledger_m3,
        "accepted_chain_sha256": value.accepted_chain_sha256,
        "shared_update": value.accepted_updates,
        "shared_time_s": value.time_s,
    })


def _wrap_state(
    execution: CoupledGeometryCPUExecutionV96, *,
    active_state: ActiveSuspendedControlStateV94,
    hles_state: HlesProducerStateV90,
    accepted_chain_sha256: str,
    q3_provider: object | None = None,
) -> CoupledGeometryCPUStateV96:
    execution_document_coupled_geometry_owner_v96(execution)
    validate_active_suspended_control_state_v94b(
        active_state, execution.active_execution, q3_provider)
    hles_execution = _v92_execution(
        execution.active_execution).base_execution.hles_execution
    validate_hles_producer_state_v90(hles_state, hles_execution)
    v92 = _v92_state(active_state)
    if (active_state.accepted_updates != hles_state.accepted_updates or
            active_state.time_s != hles_state.time_s or
            active_state.accepted_updates > MAX_ACCEPTED_UPDATES or
            np.any(v92.q[0] <=
                   execution.hydraulic_execution.base_execution.
                   dry_tolerance_m) or
            (q3_provider is not None and not _provider_is_closed(q3_provider))):
        raise CoupledGeometryOwnerV96Error(
            "v96 geometry clocks, wetness, boundary, or turbulence scope differs")
    bed_mass = active_state.base_state.bed_mass_kg_m2
    bed = _bed_from_mass(execution, bed_mass)
    eta = np.ascontiguousarray(v92.q[0]+bed)
    eta.setflags(write=False)
    placeholder = CoupledGeometryCPUStateV96(
        execution.fingerprint, active_state, hles_state, bed, eta, 0.0,
        _sha(accepted_chain_sha256, "accepted chain"), "")
    return CoupledGeometryCPUStateV96(
        **{**placeholder.__dict__, "state_sha256": _state_token(placeholder)})


def resolve_coupled_geometry_cpu_state_v96(
    execution: CoupledGeometryCPUExecutionV96, *,
    active_state: ActiveSuspendedControlStateV94,
    hles_state: HlesProducerStateV90,
    q3_provider: object,
) -> CoupledGeometryCPUStateV96:
    if active_state.accepted_updates != 0 or hles_state.accepted_updates != 0:
        raise CoupledGeometryOwnerV96Error(
            "v96 geometry fresh state must start at update zero")
    if not np.array_equal(active_state.base_state.bed_mass_kg_m2,
                          execution.reference_bed_mass_kg_m2):
        raise CoupledGeometryOwnerV96Error(
            "v96 geometry reference bed mass differs from initial v94b state")
    chain = _fingerprint({
        "scope_id": SCOPE_ID,
        "execution_fingerprint": execution.fingerprint,
        "initial_active_state_sha256": active_state.state_sha256,
        "initial_hles_state_sha256": hles_state.state_sha256,
        "initial_bed_sha256": _digest(execution.initial_bed_elevation_m),
    })
    return _wrap_state(execution, active_state=active_state,
                       hles_state=hles_state,
                       accepted_chain_sha256=chain,
                       q3_provider=q3_provider)


def validate_coupled_geometry_cpu_state_v96(
    state: CoupledGeometryCPUStateV96,
    execution: CoupledGeometryCPUExecutionV96,
    q3_provider: object | None = None,
) -> None:
    if (type(state) is not CoupledGeometryCPUStateV96 or
            state.execution_fingerprint != execution.fingerprint or
            state.geometry_water_volume_ledger_m3 != 0.0 or
            state.state_sha256 != _state_token(state)):
        raise CoupledGeometryOwnerV96Error(
            "v96 geometry state is stale, mutated, or cross-execution")
    rebuilt = _wrap_state(
        execution, active_state=state.active_state,
        hles_state=state.hles_state,
        accepted_chain_sha256=state.accepted_chain_sha256,
        q3_provider=q3_provider)
    if (not np.array_equal(state.bed_elevation_m,
                           rebuilt.bed_elevation_m) or
            not np.array_equal(state.free_surface_elevation_m,
                               rebuilt.free_surface_elevation_m) or
            state.state_sha256 != rebuilt.state_sha256):
        raise CoupledGeometryOwnerV96Error(
            "v96 geometry rebuilt state differs")


def _density_receipt_token(value: V94bAcceptedDensityReceiptV96) -> str:
    return _fingerprint({
        "scope_id": SCOPE_ID + ":accepted-v94b-density",
        "source_kind": "typed_v94b_accepted_state",
        "execution_fingerprint": value.execution_fingerprint,
        "active_execution_fingerprint": value.active_execution_fingerprint,
        "parent_state_sha256": value.parent_state_sha256,
        "base_v92_state_sha256": value.base_v92_state_sha256,
        "suspended_state_sha256": value.suspended_state_sha256,
        "accepted_update": value.accepted_update,
        "time_s": value.time_s,
        "arrays_sha256": _digest(
            value.depth_m, value.suspended_mass_kg_m2,
            value.concentration_kg_m3, value.water_density_kg_m3,
            value.mixture_density_kg_m3),
        "call_order": "accepted_v94b_state_to_next_native_geometry_hydraulics",
    })


def make_v94b_accepted_geometry_density_receipt_v96(
    execution: CoupledGeometryCPUExecutionV96,
    parent: CoupledGeometryCPUStateV96,
    q3_provider: object,
) -> V94bAcceptedDensityReceiptV96:
    validate_coupled_geometry_cpu_state_v96(parent, execution, q3_provider)
    v92 = _v92_state(parent.active_state)
    suspended = v92.suspended_state
    depth = _array(v92.q[0], (execution.cell_count,), "accepted depth",
                   positive=True)
    mass = _array(suspended.suspended_mass_kg_m2,
                  (execution.fraction_count, execution.cell_count),
                  "accepted suspended mass", nonnegative=True)
    concentration = np.ascontiguousarray(mass/depth[None, :])
    water = _array(v92.water_density_cell_kg_m3,
                   (execution.cell_count,), "accepted water density",
                   positive=True)
    mixture = mixture_density_kg_m3_v85(
        concentration, water,
        execution.hydraulic_execution.base_execution.solid_density_kg_m3)
    concentration.setflags(write=False)
    mixture.setflags(write=False)
    placeholder = V94bAcceptedDensityReceiptV96(
        execution.fingerprint, execution.active_execution.fingerprint,
        parent.state_sha256, v92.state_sha256, suspended.state_sha256,
        parent.accepted_updates, parent.time_s, depth, mass, concentration,
        water, mixture, "")
    return V94bAcceptedDensityReceiptV96(
        **{**placeholder.__dict__,
           "receipt_sha256": _density_receipt_token(placeholder)})


def validate_v94b_accepted_geometry_density_receipt_v96(
    value: V94bAcceptedDensityReceiptV96,
    execution: CoupledGeometryCPUExecutionV96,
    parent: CoupledGeometryCPUStateV96,
    q3_provider: object,
) -> None:
    if type(value) is not V94bAcceptedDensityReceiptV96:
        raise CoupledGeometryOwnerV96Error(
            "v96 geometry density receipt type differs")
    expected = make_v94b_accepted_geometry_density_receipt_v96(
        execution, parent, q3_provider)
    if (value.receipt_sha256 != _density_receipt_token(value) or
            value.receipt_sha256 != expected.receipt_sha256):
        raise CoupledGeometryOwnerV96Error(
            "v96 geometry density receipt is stale or mismatched")


@dataclass(frozen=True)
class AcceptedBedGeometryReceiptV96:
    execution_fingerprint: str
    parent_state_sha256: str
    active_state_sha256: str
    bed_mass_sha256: str
    bed_elevation_m: np.ndarray
    free_surface_elevation_m: np.ndarray
    geometry_policy: str
    geometry_water_volume_ledger_m3: float
    receipt_sha256: str


def _bed_receipt_token(value: AcceptedBedGeometryReceiptV96) -> str:
    return _fingerprint({
        "scope_id": SCOPE_ID + ":accepted-bed-geometry",
        "execution_fingerprint": value.execution_fingerprint,
        "parent_state_sha256": value.parent_state_sha256,
        "active_state_sha256": value.active_state_sha256,
        "bed_mass_sha256": value.bed_mass_sha256,
        "geometry_sha256": _digest(value.bed_elevation_m,
                                   value.free_surface_elevation_m),
        "geometry_policy": value.geometry_policy,
        "geometry_water_volume_ledger_m3":
            value.geometry_water_volume_ledger_m3,
        "call_order": "accepted_parent_bed_to_current_hydraulics",
    })


def make_accepted_bed_geometry_receipt_v96(
    execution: CoupledGeometryCPUExecutionV96,
    parent: CoupledGeometryCPUStateV96,
    q3_provider: object,
) -> AcceptedBedGeometryReceiptV96:
    validate_coupled_geometry_cpu_state_v96(parent, execution, q3_provider)
    bed_mass = parent.active_state.base_state.bed_mass_kg_m2
    placeholder = AcceptedBedGeometryReceiptV96(
        execution.fingerprint, parent.state_sha256,
        parent.active_state.state_sha256, _digest(bed_mass),
        parent.bed_elevation_m, parent.free_surface_elevation_m,
        GEOMETRY_POLICY, 0.0, "")
    return AcceptedBedGeometryReceiptV96(
        **{**placeholder.__dict__,
           "receipt_sha256": _bed_receipt_token(placeholder)})


def validate_accepted_bed_geometry_receipt_v96(
    value: AcceptedBedGeometryReceiptV96,
    execution: CoupledGeometryCPUExecutionV96,
    parent: CoupledGeometryCPUStateV96,
    q3_provider: object,
) -> None:
    if type(value) is not AcceptedBedGeometryReceiptV96:
        raise CoupledGeometryOwnerV96Error(
            "v96 geometry bed receipt type differs")
    expected = make_accepted_bed_geometry_receipt_v96(
        execution, parent, q3_provider)
    if (value.receipt_sha256 != _bed_receipt_token(value) or
            value.receipt_sha256 != expected.receipt_sha256):
        raise CoupledGeometryOwnerV96Error(
            "v96 geometry bed receipt is stale or mismatched")


@dataclass(frozen=True)
class CoupledGeometryOwnerCountsV96:
    native_hydraulic_owner_count: int
    water_face_flux_owner_count: int
    bed_geometry_owner_count: int
    turbulence_cap_owner_count: int
    turbulence_momentum_owner_count: int
    turbulence_scalar_consumer_count: int
    scalar_fct_owner_count: int
    scalar_diffusion_owner_count: int
    scalar_lifecycle_owner_count: int
    authoritative_exner_owner_count: int
    hles_clock_owner_count: int
    duplicate_hydraulic_owner_count: int
    duplicate_bed_geometry_owner_count: int
    duplicate_turbulence_owner_count: int
    duplicate_scalar_owner_count: int
    duplicate_exner_owner_count: int


@dataclass(frozen=True)
class CoupledGeometryCPUTrialV96:
    execution_fingerprint: str
    parent_state_sha256: str
    accepted_density: V94bAcceptedDensityReceiptV96
    accepted_bed_geometry: AcceptedBedGeometryReceiptV96
    current_handoff: CurrentClosureHandoffV90
    turbulence_preparation_sha256: str
    turbulence_runtime_sha256: str
    native_scalar_evidence: NativeCombinedScalarEvidenceV96
    dt_s: float
    midpoint_q: np.ndarray
    base_fwave_rate: np.ndarray
    density_momentum_rate: np.ndarray
    turbulence_momentum_rate_m2_s2: np.ndarray
    turbulence_face_flux_m4_s2: np.ndarray
    full_rate: np.ndarray
    final_residual: np.ndarray
    face_normal_acceleration_m_s2: np.ndarray
    cell_acceleration_m_s2: np.ndarray
    face_water_flux_m2_s: np.ndarray
    midpoint_depth_sha256_by_evaluation: tuple[str, ...]
    residual_evaluation_count: int
    density_source_evaluation_count: int
    turbulence_momentum_evaluation_count: int
    augmented_riemann_call_count: int
    nonlinear_iterations: int
    nonlinear_residual_linf: float
    water_volume_residual_m3: float
    active_trial: ActiveSuspendedControlTrialV94
    hles_trial: HlesProducerTrialV90
    owner_counts: CoupledGeometryOwnerCountsV96
    candidate: CoupledGeometryCPUStateV96
    backend: str
    trial_sha256: str


def _trial_token(value: CoupledGeometryCPUTrialV96) -> str:
    return _fingerprint({
        "scope_id": SCOPE_ID + ":trial",
        "execution_fingerprint": value.execution_fingerprint,
        "parent_state_sha256": value.parent_state_sha256,
        "density_receipt": value.accepted_density.receipt_sha256,
        "bed_receipt": value.accepted_bed_geometry.receipt_sha256,
        "handoff": value.current_handoff.receipt_sha256,
        "turbulence_preparation": value.turbulence_preparation_sha256,
        "turbulence_runtime": value.turbulence_runtime_sha256,
        "native_scalar_evidence":
            value.native_scalar_evidence.evidence_sha256,
        "dt_s": value.dt_s,
        "arrays_sha256": _digest(
            value.midpoint_q, value.base_fwave_rate,
            value.density_momentum_rate,
            value.turbulence_momentum_rate_m2_s2,
            value.turbulence_face_flux_m4_s2, value.full_rate,
            value.final_residual, value.face_normal_acceleration_m_s2,
            value.cell_acceleration_m_s2, value.face_water_flux_m2_s),
        "midpoint_depth_sha256_by_evaluation":
            list(value.midpoint_depth_sha256_by_evaluation),
        "counts": {
            "residual": value.residual_evaluation_count,
            "density": value.density_source_evaluation_count,
            "turbulence": value.turbulence_momentum_evaluation_count,
            "riemann": value.augmented_riemann_call_count,
            "nonlinear_iterations": value.nonlinear_iterations,
            "owners": list(value.owner_counts.__dict__.values()),
        },
        "nonlinear_residual_linf": value.nonlinear_residual_linf,
        "water_volume_residual_m3": value.water_volume_residual_m3,
        "active_trial": value.active_trial.trial_sha256,
        "hles_trial": value.hles_trial.trial_sha256,
        "candidate": value.candidate.state_sha256,
        "backend": value.backend,
    })


def begin_coupled_geometry_cpu_trial_v96(
    execution: CoupledGeometryCPUExecutionV96,
    parent: CoupledGeometryCPUStateV96,
    q3_provider: object, *,
    native_runtime: NativeCoupledGeometryRuntimeV96,
    turbulence_runtime: NativeSharedTurbulenceRuntimeV96,
    scalar_fct_runtime: NativeScalarFCTRuntimeV96,
    scalar_diffusion_runtime: NativeScalarDiffusionRuntimeV96,
    shear_velocity_m_s: object,
    closure_chezy_mhalf_s: object,
    q3_inputs: Mapping[str, object],
    dt_s: float,
    cancelled: bool | Callable[[], bool] = False,
) -> CoupledGeometryCPUTrialV96:
    execution_document_coupled_geometry_owner_v96(execution)
    validate_coupled_geometry_cpu_state_v96(parent, execution, q3_provider)
    if (type(native_runtime) is not NativeCoupledGeometryRuntimeV96 or
            native_runtime.receipt !=
                execution.hydraulic_execution.native_build_receipt):
        raise CoupledGeometryOwnerV96Error(
            "v96 geometry native runtime identity differs")
    if (type(turbulence_runtime) is not NativeSharedTurbulenceRuntimeV96 or
            turbulence_runtime.receipt.get("receipt_sha256") !=
                execution.turbulence_native_receipt_sha256 or
            turbulence_runtime.receipt.get("native_source_sha256") !=
                execution.turbulence_native_source_sha256):
        raise CoupledGeometryOwnerV96Error(
            "v96 geometry native turbulence runtime identity differs")
    if (type(scalar_fct_runtime) is not NativeScalarFCTRuntimeV96 or
            scalar_fct_runtime.runtime_sha256 !=
                execution.scalar_fct_native_receipt_sha256 or
            scalar_fct_runtime.receipt.get("native_source_sha256") !=
                execution.scalar_fct_native_source_sha256 or
            type(scalar_diffusion_runtime) is not
                NativeScalarDiffusionRuntimeV96 or
            scalar_diffusion_runtime.receipt.get("receipt_sha256") !=
                execution.scalar_diffusion_native_receipt_sha256 or
            scalar_diffusion_runtime.receipt.get("source_sha256") !=
                execution.scalar_diffusion_native_source_sha256):
        raise CoupledGeometryOwnerV96Error(
            "v96 geometry native scalar runtime identity differs")
    if parent.accepted_updates >= MAX_ACCEPTED_UPDATES:
        raise CoupledGeometryOwnerV96Error(
            "v96 geometry accepted update limit was reached")
    if not isinstance(q3_inputs, Mapping):
        raise CoupledGeometryOwnerV96Error(
            "v96 geometry Q3 inputs must be a mapping")
    if _cancelled(cancelled):
        raise InterruptedError(
            "v96 geometry cancelled before native hydraulic work")
    dt = _finite(dt_s, "timestep", positive=True)
    hydraulic = execution.hydraulic_execution
    if dt > hydraulic.maximum_dt_s:
        raise CoupledGeometryOwnerV96Error(
            "v96 geometry timestep exceeds the native bound")
    density = make_v94b_accepted_geometry_density_receipt_v96(
        execution, parent, q3_provider)
    bed = make_accepted_bed_geometry_receipt_v96(
        execution, parent, q3_provider)
    request = _fingerprint({
        "scope_id": SCOPE_ID + ":hydraulic-request",
        "execution_fingerprint": execution.fingerprint,
        "parent_state_sha256": parent.state_sha256,
        "density_receipt": density.receipt_sha256,
        "bed_receipt": bed.receipt_sha256,
        "dt_s": dt,
    })
    hles_execution = _v92_execution(
        execution.active_execution).base_execution.hles_execution
    handoff = make_current_closure_handoff_v90(
        hles_execution, parent.hles_state,
        hydraulic_trial_request_sha256=request)
    v91_execution = _v92_execution(execution.active_execution).base_execution
    prepared_turbulence = turbulence_runtime.prepare(
        v91_execution, parent.hles_state, handoff, dt_s=dt,
        native_threads=hydraulic.native_threads)
    density_inputs_sha = _digest(
        density.water_density_kg_m3, density.mixture_density_kg_m3)
    bed_sha = _digest(bed.bed_elevation_m)
    midpoint = np.array(parent.q, copy=True, order="C")
    depth_digests: list[str] = []
    density_calls = turbulence_calls = riemann_calls = nonlinear_iterations = 0
    result: NativeCoupledGeometryRateResultV96 | None = None
    turbulence: NativeMomentumRateV96 | None = None
    full_rate: np.ndarray | None = None
    residual: np.ndarray | None = None
    residual_norm = math.inf
    for evaluation in range(1, hydraulic.max_nonlinear_iterations+2):
        if _cancelled(cancelled):
            raise InterruptedError(
                "v96 geometry cancelled during native midpoint")
        result = native_runtime.evaluate(
            hydraulic, midpoint, bed_elevation_m=bed.bed_elevation_m,
            accepted_water_density_kg_m3=density.water_density_kg_m3,
            accepted_mixture_density_kg_m3=density.mixture_density_kg_m3)
        if (_digest(density.water_density_kg_m3,
                    density.mixture_density_kg_m3) != density_inputs_sha or
                _digest(bed.bed_elevation_m) != bed_sha or
                result.accepted_bed_sha256 != bed_sha):
            raise CoupledGeometryOwnerV96Error(
                "v96 geometry native evaluation mutated accepted inputs")
        turbulence = turbulence_runtime.evaluate(
            prepared_turbulence, midpoint,
            cancelled=(cancelled if callable(cancelled)
                       else lambda: bool(cancelled)))
        full_rate = np.array(result.full_rate, copy=True, order="C")
        full_rate += turbulence.momentum_rate_m2_s2
        full_rate = np.ascontiguousarray(full_rate)
        depth_digests.append(_digest(midpoint[0]))
        density_calls += result.density_source_evaluation_count
        turbulence_calls += turbulence.momentum_rate_evaluation_count
        riemann_calls += result.augmented_riemann_call_count
        residual = np.ascontiguousarray(
            midpoint-parent.q-0.5*dt*full_rate)
        residual_norm = _residual_norm(
            residual, midpoint, hydraulic.base_execution.gravity_m_s2)
        if residual_norm <= hydraulic.nonlinear_tolerance:
            break
        if evaluation > hydraulic.max_nonlinear_iterations:
            raise CoupledGeometryOwnerV96Error(
                "v96 geometry native midpoint failed to converge")
        midpoint = np.ascontiguousarray(midpoint-residual)
        if (not np.isfinite(midpoint).all() or
                np.any(midpoint[0] <=
                       hydraulic.base_execution.dry_tolerance_m)):
            raise CoupledGeometryOwnerV96Error(
                "v96 geometry midpoint left the fully-wet scope")
        nonlinear_iterations += 1
    if (result is None or turbulence is None or full_rate is None or
            residual is None):
        raise CoupledGeometryOwnerV96Error(
            "v96 geometry native solve produced no result")
    endpoint = np.ascontiguousarray(parent.q+dt*full_rate)
    if np.any(endpoint[0] <= hydraulic.base_execution.dry_tolerance_m):
        raise CoupledGeometryOwnerV96Error(
            "v96 geometry endpoint left the fully-wet scope")
    boundary = hydraulic.base_execution.neighbour < 0
    boundary_outflow = float(np.sum(
        hydraulic.base_execution.face_length_m[boundary]*
        result.face_water_flux_m2_s[boundary]))
    water_residual = float(np.sum(
        hydraulic.base_execution.cell_area_m2*(endpoint[0]-parent.q[0]))+
        dt*boundary_outflow)
    water_scale = max(1.0, float(np.sum(
        hydraulic.base_execution.cell_area_m2*parent.q[0])))
    if abs(water_residual) > 128*np.finfo(float).eps*water_scale:
        raise CoupledGeometryOwnerV96Error(
            "v96 geometry native closed water budget differs")
    scalar = _closed_scalar_inputs(
        execution, parent, endpoint, shear_velocity_m_s)
    provider_checkpoint = q3_provider.checkpoint()
    active_trial: ActiveSuspendedControlTrialV94 | None = None
    try:
        if _cancelled(cancelled):
            raise InterruptedError(
                "v96 geometry cancelled before v94b child work")
        active_result = begin_active_suspended_preconverged_cpu_trial_v96(
            execution.active_execution, parent.active_state,
            parent.hles_state, handoff, q3_provider,
            cap_receipt=prepared_turbulence.cap_receipt,
            midpoint_q=midpoint, accepted_q=endpoint,
            native_momentum_rate_m2_s2=
                turbulence.momentum_rate_m2_s2,
            native_momentum_face_flux_m4_s2=
                turbulence.face_flux_m4_s2,
            residual_evaluation_count=len(depth_digests),
            accepted_water_flux_m2_s=result.face_water_flux_m2_s,
            dt_s=dt, v92_scalar_inputs=scalar,
            q3_inputs=dict(q3_inputs),
            scalar_fct_runtime=scalar_fct_runtime,
            scalar_diffusion_runtime=scalar_diffusion_runtime,
            native_scalar_threads=hydraulic.native_threads,
            return_native_scalar_evidence=True,
            cancelled=(cancelled if callable(cancelled)
                       else lambda: bool(cancelled)))
        active_trial, native_scalar_evidence = active_result
        if native_scalar_evidence is None:
            raise CoupledGeometryOwnerV96Error(
                "v96 geometry native scalar evidence is absent")
        base_trial = active_trial.base_v93_trial.base_v92_trial
        if (not np.array_equal(base_trial.candidate.q, endpoint) or
                not np.array_equal(
                    base_trial.hydraulic_face_receipt.
                    accepted_water_flux_m2_s,
                    result.face_water_flux_m2_s) or
                base_trial.cap_receipt.receipt_sha256 !=
                    prepared_turbulence.cap_receipt.receipt_sha256 or
                not np.allclose(
                    base_trial.momentum_rate_m2_s2,
                    turbulence.momentum_rate_m2_s2,
                    rtol=2e-14, atol=2e-16) or
                not np.allclose(
                    base_trial.momentum_face_flux_m4_s2,
                    turbulence.face_flux_m4_s2,
                    rtol=2e-14, atol=2e-16)):
            raise CoupledGeometryOwnerV96Error(
                "v96 geometry nested v92 duplicated or changed hydraulics")
        if _cancelled(cancelled):
            raise InterruptedError(
                "v96 geometry cancelled after v94b child work")
        face_velocity = _face_normal_velocity(
            execution, endpoint, result.face_water_flux_m2_s)
        chezy = _array(
            closure_chezy_mhalf_s, (execution.cell_count,),
            "closure Chezy", positive=True)
        closure_inputs = {
            "accepted_face_normal_velocity_m_s": face_velocity,
            "active_face": np.ones(execution.face_count, dtype=np.int32),
            "closure_depth_m": endpoint[0],
            "closure_chezy_mhalf_s": chezy,
            "active_cell": np.ones(execution.cell_count, dtype=np.int32),
        }
        step = base_trial.hydraulic_receipt.base_step_receipt
        hles_input = make_hles_input_receipt_v90(
            hles_execution, parent.hles_state,
            accepted_step_receipt=step, current_handoff=handoff,
            closure_interval_s=dt, **closure_inputs)
        hles_trial = begin_hles_cpu_trial_v90(
            hles_execution, parent.hles_state,
            accepted_step_receipt=step,
            hles_input_receipt=hles_input, current_handoff=handoff,
            # v90's frozen public ABI accepts only a Boolean.  The callable
            # is polled immediately above after all v94b child work, so no
            # closure operator starts after cancellation is requested.
            cancelled=False, **closure_inputs)
        chain = _fingerprint({
            "previous": parent.accepted_chain_sha256,
            "accepted_density": density.receipt_sha256,
            "accepted_bed_geometry": bed.receipt_sha256,
            "native_build": hydraulic.native_build_receipt.receipt_sha256,
            "native_midpoint": _digest(midpoint),
            "native_geometry_rate": _digest(result.full_rate),
            "native_turbulence_preparation":
                prepared_turbulence.preparation_sha256,
            "native_turbulence_rate":
                _digest(turbulence.momentum_rate_m2_s2,
                        turbulence.face_flux_m4_s2),
            "composed_full_rate": _digest(full_rate),
            "native_face_water_flux": _digest(result.face_water_flux_m2_s),
            "native_scalar": native_scalar_evidence.evidence_sha256,
            "active_trial": active_trial.trial_sha256,
            "hles_trial": hles_trial.trial_sha256,
            "dt_s": dt,
        })
        candidate = _wrap_state(
            execution, active_state=active_trial.candidate,
            hles_state=hles_trial.candidate,
            accepted_chain_sha256=chain)
        counts = CoupledGeometryOwnerCountsV96(
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            0, 0, 0, 0, 0)
        placeholder = CoupledGeometryCPUTrialV96(
            execution.fingerprint, parent.state_sha256, density, bed,
            handoff, prepared_turbulence.preparation_sha256,
            prepared_turbulence.native_runtime_sha256,
            native_scalar_evidence,
            dt, _array(midpoint, (3, execution.cell_count),
                                "midpoint"),
            result.base_fwave_rate, result.density_momentum_rate,
            turbulence.momentum_rate_m2_s2,
            turbulence.face_flux_m4_s2, full_rate,
            _array(residual, (3, execution.cell_count), "final residual"),
            result.face_normal_acceleration_m_s2,
            result.cell_acceleration_m_s2, result.face_water_flux_m2_s,
            tuple(depth_digests), len(depth_digests), density_calls,
            turbulence_calls, riemann_calls, nonlinear_iterations, residual_norm,
            water_residual, active_trial, hles_trial, counts, candidate,
            BACKEND, "")
        trial = CoupledGeometryCPUTrialV96(
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
                "v96 geometry begin failed and provider restore also failed") \
                from cleanup_error
        raise original_error


def _validate_trial(
    execution: CoupledGeometryCPUExecutionV96,
    parent: CoupledGeometryCPUStateV96,
    q3_provider: object,
    trial: CoupledGeometryCPUTrialV96,
) -> None:
    validate_coupled_geometry_cpu_state_v96(parent, execution, q3_provider)
    if (type(trial) is not CoupledGeometryCPUTrialV96 or
            trial.execution_fingerprint != execution.fingerprint or
            trial.parent_state_sha256 != parent.state_sha256 or
            trial.backend != BACKEND or
            trial.trial_sha256 != _trial_token(trial)):
        raise CoupledGeometryOwnerV96Error(
            "v96 geometry trial is stale, mutated, or cross-parent")
    validate_v94b_accepted_geometry_density_receipt_v96(
        trial.accepted_density, execution, parent, q3_provider)
    validate_accepted_bed_geometry_receipt_v96(
        trial.accepted_bed_geometry, execution, parent, q3_provider)
    hydraulic = execution.hydraulic_execution
    nc, nf = execution.cell_count, execution.face_count
    midpoint = _array(trial.midpoint_q, (3, nc), "trial midpoint")
    base_rate = _array(trial.base_fwave_rate, (3, nc), "base rate")
    density_rate = _array(trial.density_momentum_rate, (2, nc),
                          "density rate")
    turbulence_rate = _array(
        trial.turbulence_momentum_rate_m2_s2, (3, nc),
        "turbulence momentum rate")
    turbulence_flux = _array(
        trial.turbulence_face_flux_m4_s2, (2, nf),
        "turbulence face flux")
    full_rate = _array(trial.full_rate, (3, nc), "full rate")
    stored_residual = _array(trial.final_residual, (3, nc),
                             "stored residual")
    face_acceleration = _array(
        trial.face_normal_acceleration_m_s2, (nf,), "face acceleration")
    cell_acceleration = _array(
        trial.cell_acceleration_m_s2, (2, nc), "cell acceleration")
    face_flux = _array(trial.face_water_flux_m2_s, (nf,), "face flux")
    dt = _finite(trial.dt_s, "trial timestep", positive=True)
    expected_full = np.array(base_rate, copy=True)
    expected_full[1:] += density_rate
    expected_full += turbulence_rate
    residual = midpoint-parent.q-0.5*dt*full_rate
    endpoint = parent.q+dt*full_rate
    norm = _residual_norm(
        residual, midpoint, hydraulic.base_execution.gravity_m_s2)
    boundary = hydraulic.base_execution.neighbour < 0
    water_residual = float(np.sum(
        hydraulic.base_execution.cell_area_m2*(endpoint[0]-parent.q[0]))+
        dt*np.sum(hydraulic.base_execution.face_length_m[boundary]*
                  face_flux[boundary]))
    effective_density = (
        trial.accepted_density.mixture_density_kg_m3
        if hydraulic.density_mode == DENSITY_WATER_PLUS_SEDIMENT else
        trial.accepted_density.water_density_kg_m3)
    expected_face = delft_kmax1_face_acceleration_v85(
        midpoint[0], effective_density,
        hydraulic.base_execution.owner, hydraulic.base_execution.neighbour,
        hydraulic.base_execution.face_distance_m,
        gravity_m_s2=hydraulic.base_execution.gravity_m_s2,
        hydraulic_reference_density_kg_m3=
            hydraulic.base_execution.hydraulic_reference_density_kg_m3)
    expected_cell = project_face_acceleration_v85(
        expected_face, hydraulic.base_execution.normal,
        hydraulic.base_execution.face_length_m,
        hydraulic.base_execution.cell_face_offset,
        hydraulic.base_execution.cell_faces,
        hydraulic.base_execution.cell_face_sign)
    expected_density = midpoint[0][None, :]*expected_cell
    v92_trial = trial.active_trial.base_v93_trial.base_v92_trial
    prepared = PreparedNativeTurbulenceV96(
        _v92_execution(execution.active_execution).base_execution,
        v92_trial.cap_receipt, trial.turbulence_runtime_sha256,
        hydraulic.native_threads, trial.turbulence_preparation_sha256)
    expected_counts = CoupledGeometryOwnerCountsV96(
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0)
    if (dt > hydraulic.maximum_dt_s or
            trial.owner_counts != expected_counts or
            trial.candidate.accepted_updates != parent.accepted_updates+1 or
            trial.candidate.time_s != parent.time_s+dt or
            trial.residual_evaluation_count < 1 or
            trial.residual_evaluation_count >
                hydraulic.max_nonlinear_iterations+1 or
            trial.nonlinear_iterations != trial.residual_evaluation_count-1 or
            trial.density_source_evaluation_count !=
                trial.residual_evaluation_count or
            trial.turbulence_momentum_evaluation_count !=
                trial.residual_evaluation_count or
            trial.augmented_riemann_call_count !=
                nf*trial.residual_evaluation_count or
            len(trial.midpoint_depth_sha256_by_evaluation) !=
                trial.residual_evaluation_count or
            trial.midpoint_depth_sha256_by_evaluation[-1] !=
                _digest(midpoint[0]) or
            not np.array_equal(expected_full, full_rate) or
            not np.array_equal(residual, stored_residual) or
            not np.array_equal(endpoint, trial.candidate.q) or
            trial.nonlinear_residual_linf != norm or
            norm > hydraulic.nonlinear_tolerance or
            trial.water_volume_residual_m3 != water_residual or
            trial.candidate.geometry_water_volume_ledger_m3 != 0.0):
        raise CoupledGeometryOwnerV96Error(
            "v96 geometry residual, endpoint, counts, or ledgers differ")
    scale = max(1.0, float(np.max(np.abs(expected_density))))
    if (not np.allclose(expected_face, face_acceleration,
                        rtol=8*np.finfo(float).eps, atol=1e-18) or
            not np.allclose(expected_cell, cell_acceleration,
                            rtol=16*np.finfo(float).eps, atol=1e-18) or
            not np.allclose(expected_density, density_rate,
                            rtol=16*np.finfo(float).eps,
                            atol=16*np.finfo(float).eps*scale) or
            trial.turbulence_runtime_sha256 !=
                execution.turbulence_native_receipt_sha256 or
            trial.turbulence_preparation_sha256 !=
                _native_v91._preparation_token(prepared)):
        raise CoupledGeometryOwnerV96Error(
            "v96 geometry native density source differs from v85 oracle")
    if (not np.array_equal(v92_trial.candidate.q, endpoint) or
            not np.array_equal(
                v92_trial.hydraulic_face_receipt.accepted_water_flux_m2_s,
                face_flux) or
            v92_trial.cap_receipt.receipt_sha256 !=
                prepared.cap_receipt.receipt_sha256 or
            not np.allclose(v92_trial.momentum_rate_m2_s2,
                            turbulence_rate, rtol=2e-14, atol=2e-16) or
            not np.allclose(v92_trial.momentum_face_flux_m4_s2,
                            turbulence_flux, rtol=2e-14, atol=2e-16)):
        raise CoupledGeometryOwnerV96Error(
            "v96 geometry scalar owner did not consume native hydraulics")
    scalar_evidence = trial.native_scalar_evidence
    if (type(scalar_evidence) is not NativeCombinedScalarEvidenceV96 or
            scalar_evidence.fct_runtime_sha256 !=
                execution.scalar_fct_native_receipt_sha256 or
            scalar_evidence.diffusion_runtime_sha256 !=
                execution.scalar_diffusion_native_receipt_sha256 or
            scalar_evidence.suspended_trial_sha256 !=
                v92_trial.suspended_trial.trial_sha256 or
            scalar_evidence.fluid_result_sha256 !=
                v92_trial.fluid_transport.result_sha256 or
            scalar_evidence.combined_lane_count !=
                execution.fraction_count+int(np.count_nonzero(
                    _v92_execution(execution.active_execution).
                    transported_mask)) or
            scalar_evidence.native_threads != hydraulic.native_threads or
            scalar_evidence.fct_launch_count != 1 or
            scalar_evidence.diffusion_launch_count != 1 or
            scalar_evidence.python_numerical_cell_face_loop_count != 0 or
            scalar_evidence.state_or_clock_publication_count != 0 or
            scalar_evidence.evidence_sha256 !=
                _native_scalar_evidence_token(scalar_evidence)):
        raise CoupledGeometryOwnerV96Error(
            "v96 geometry native scalar evidence differs")
    _v94._validate_trial(
        execution.active_execution, parent.active_state, parent.hles_state,
        trial.current_handoff, trial.active_trial)
    if rollback_hles_trial_v90(
            _v92_execution(execution.active_execution).base_execution.
            hles_execution, parent.hles_state,
            trial.hles_trial) is not parent.hles_state:
        raise CoupledGeometryOwnerV96Error(
            "v96 geometry HLES rollback validation differs")
    expected_bed = _bed_from_mass(
        execution, trial.active_trial.candidate.base_state.bed_mass_kg_m2)
    if (not np.array_equal(trial.candidate.bed_elevation_m, expected_bed) or
            not np.array_equal(
                trial.candidate.free_surface_elevation_m,
                trial.candidate.q[0]+expected_bed)):
        raise CoupledGeometryOwnerV96Error(
            "v96 geometry candidate bed feedback differs")
    expected_chain = _fingerprint({
        "previous": parent.accepted_chain_sha256,
        "accepted_density": trial.accepted_density.receipt_sha256,
        "accepted_bed_geometry":
            trial.accepted_bed_geometry.receipt_sha256,
        "native_build": hydraulic.native_build_receipt.receipt_sha256,
        "native_midpoint": _digest(midpoint),
        "native_geometry_rate": _digest(
            np.vstack((base_rate[0], base_rate[1:]+density_rate))),
        "native_turbulence_preparation":
            trial.turbulence_preparation_sha256,
        "native_turbulence_rate":
            _digest(turbulence_rate, turbulence_flux),
        "composed_full_rate": _digest(full_rate),
        "native_face_water_flux": _digest(face_flux),
        "native_scalar": trial.native_scalar_evidence.evidence_sha256,
        "active_trial": trial.active_trial.trial_sha256,
        "hles_trial": trial.hles_trial.trial_sha256,
        "dt_s": dt,
    })
    rebuilt = _wrap_state(
        execution, active_state=trial.active_trial.candidate,
        hles_state=trial.hles_trial.candidate,
        accepted_chain_sha256=expected_chain)
    if (trial.candidate.accepted_chain_sha256 != expected_chain or
            trial.candidate.state_sha256 != rebuilt.state_sha256):
        raise CoupledGeometryOwnerV96Error(
            "v96 geometry candidate chain differs")


def commit_coupled_geometry_cpu_trial_v96(
    execution: CoupledGeometryCPUExecutionV96,
    parent: CoupledGeometryCPUStateV96,
    q3_provider: object,
    trial: CoupledGeometryCPUTrialV96,
) -> CoupledGeometryCPUStateV96:
    checkpoint = trial.active_trial.base_v93_trial.q3_parent_checkpoint
    try:
        _validate_trial(execution, parent, q3_provider, trial)
        hles = commit_hles_trial_v90(
            _v92_execution(execution.active_execution).base_execution.
            hles_execution, parent.hles_state, trial.hles_trial)
        if hles is not trial.hles_trial.candidate:
            raise CoupledGeometryOwnerV96Error(
                "v96 geometry HLES commit substituted its candidate")
    except Exception as original_error:
        try:
            _restore_provider_after_failure(
                getattr(trial, "active_trial", None), q3_provider, checkpoint)
        except Exception as cleanup_error:
            raise RuntimeError(
                "v96 geometry commit failed and restore also failed") \
                from cleanup_error
        raise original_error
    commit_active_suspended_control_trial_v94b(
        execution.active_execution, parent.active_state, parent.hles_state,
        trial.current_handoff, trial.active_trial)
    return trial.candidate


def rollback_coupled_geometry_cpu_trial_v96(
    execution: CoupledGeometryCPUExecutionV96,
    parent: CoupledGeometryCPUStateV96,
    q3_provider: object,
    trial: CoupledGeometryCPUTrialV96,
) -> CoupledGeometryCPUStateV96:
    _validate_trial(execution, parent, q3_provider, trial)
    if rollback_hles_trial_v90(
            _v92_execution(execution.active_execution).base_execution.
            hles_execution, parent.hles_state,
            trial.hles_trial) is not parent.hles_state:
        raise CoupledGeometryOwnerV96Error(
            "v96 geometry rollback changed HLES parent")
    active = rollback_active_suspended_control_trial_v94b(
        execution.active_execution, parent.active_state, parent.hles_state,
        trial.current_handoff, trial.active_trial)
    if active is not parent.active_state:
        raise CoupledGeometryOwnerV96Error(
            "v96 geometry rollback changed active parent")
    return parent


def cancel_coupled_geometry_cpu_trial_v96(*args, **kwargs):
    return rollback_coupled_geometry_cpu_trial_v96(*args, **kwargs)


def coupled_geometry_cpu_checkpoint_v96(
    execution: CoupledGeometryCPUExecutionV96,
    state: CoupledGeometryCPUStateV96,
    q3_provider: object,
) -> dict[str, object]:
    validate_coupled_geometry_cpu_state_v96(state, execution, q3_provider)
    hles_execution = _v92_execution(
        execution.active_execution).base_execution.hles_execution
    body = {
        "schema": SCHEMA, "scope_id": CHECKPOINT_SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "execution_fingerprint": execution.fingerprint,
        "active_v94b_checkpoint":
            checkpoint_active_suspended_control_state_v94b(
                execution.active_execution, state.active_state, q3_provider),
        "hles_checkpoint": hles_checkpoint_v90(
            hles_execution, state.hles_state),
        "bed_elevation_m": state.bed_elevation_m.tolist(),
        "free_surface_elevation_m":
            state.free_surface_elevation_m.tolist(),
        "geometry_water_volume_ledger_m3":
            state.geometry_water_volume_ledger_m3,
        "accepted_chain_sha256": state.accepted_chain_sha256,
        "state_sha256": state.state_sha256,
    }
    return {**body, "fingerprint": _fingerprint(body)}


def load_coupled_geometry_cpu_checkpoint_v96(
    execution: CoupledGeometryCPUExecutionV96,
    document: Mapping[str, object],
    q3_provider: object,
) -> CoupledGeometryCPUStateV96:
    expected = {
        "schema", "scope_id", "semantic_id", "execution_fingerprint",
        "active_v94b_checkpoint", "hles_checkpoint", "bed_elevation_m",
        "free_surface_elevation_m", "geometry_water_volume_ledger_m3",
        "accepted_chain_sha256", "state_sha256", "fingerprint",
    }
    if not isinstance(document, Mapping) or set(document) != expected:
        raise CoupledGeometryOwnerV96Error(
            "v96 geometry checkpoint inventory differs")
    body = {key: document[key] for key in document if key != "fingerprint"}
    if (document["fingerprint"] != _fingerprint(body) or
            document["schema"] != SCHEMA or
            document["scope_id"] != CHECKPOINT_SCOPE_ID or
            document["semantic_id"] != SEMANTIC_ID or
            document["execution_fingerprint"] != execution.fingerprint):
        raise CoupledGeometryOwnerV96Error(
            "v96 geometry checkpoint identity differs")
    hles_execution = _v92_execution(
        execution.active_execution).base_execution.hles_execution
    hles = load_hles_checkpoint_v90(
        hles_execution, document["hles_checkpoint"])
    provider_checkpoint = q3_provider.checkpoint()
    try:
        active = load_active_suspended_control_checkpoint_v94b(
            execution.active_execution,
            document["active_v94b_checkpoint"], q3_provider)
        state = _wrap_state(
            execution, active_state=active, hles_state=hles,
            accepted_chain_sha256=document["accepted_chain_sha256"],
            q3_provider=q3_provider)
        saved_bed = _array(document["bed_elevation_m"],
                           (execution.cell_count,), "checkpoint bed")
        saved_eta = _array(document["free_surface_elevation_m"],
                           (execution.cell_count,), "checkpoint free surface")
        if (not np.array_equal(saved_bed, state.bed_elevation_m) or
                not np.array_equal(saved_eta,
                                   state.free_surface_elevation_m) or
                document["geometry_water_volume_ledger_m3"] != 0.0 or
                document["state_sha256"] != state.state_sha256):
            raise CoupledGeometryOwnerV96Error(
                "v96 geometry checkpoint state differs")
        return state
    except Exception as original_error:
        try:
            q3_provider.restore_checkpoint(provider_checkpoint)
        except Exception as cleanup_error:
            raise RuntimeError(
                "v96 geometry checkpoint failed and restore also failed") \
                from cleanup_error
        raise original_error


def method_identity_coupled_geometry_owner_v96() -> dict[str, object]:
    return {
        "scope_id": SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "backend": BACKEND,
        "accepted_state_source": "genuine_v94b_state_no_shadow_q",
        "bed_mapping": "z0+sum((M-M0)/rho_dry_by_fraction)",
        "dry_bulk_density_origin": "rhosol*(1-porosity)",
        "bed_timing": "accepted_parent_current;candidate_next",
        "geometry_policy": GEOMETRY_POLICY,
        "geometry_water_volume_ledger": "exact_zero",
        "water_eos": "accepted_spatial_v92_field_retained_with_densin_off",
        "hydraulics": "native_openmp_augmented_fwave_midpoint",
        "lagged_v91_turbulence": (
            "one_native_prepared_cap;momentum_each_residual;"
            "scalar_diffusion_once"),
        "execution_partition": (
            "native_openmp_hydraulics_v91_momentum_scalar_fct_diffusion;"
            "host_python_boundaries_settling_exchange_q3_exner"),
        "nested_v91_v92_receipt_role": (
            "compatibility_oracle_only;outer_native_chain_authoritative"),
        "accepted_update_limit": MAX_ACCEPTED_UPDATES,
        "product_admission": False,
        "not_claimed": [
            "open_hydraulic_or_sediment_boundaries",
            "wet_dry_stage_adjustment", "AMR", "cut_cells",
            "plugin_support", "long_trajectory_qualification",
        ],
    }


__all__ = [
    "BACKEND", "GEOMETRY_POLICY", "MAX_ACCEPTED_UPDATES", "SEMANTIC_ID",
    "AcceptedBedGeometryReceiptV96", "CoupledGeometryCPUExecutionV96",
    "CoupledGeometryCPUStateV96", "CoupledGeometryCPUTrialV96",
    "CoupledGeometryOwnerCountsV96", "CoupledGeometryOwnerV96Error",
    "begin_coupled_geometry_cpu_trial_v96",
    "cancel_coupled_geometry_cpu_trial_v96",
    "commit_coupled_geometry_cpu_trial_v96",
    "coupled_geometry_cpu_checkpoint_v96",
    "execution_document_coupled_geometry_owner_v96",
    "load_coupled_geometry_cpu_checkpoint_v96",
    "make_accepted_bed_geometry_receipt_v96",
    "make_v94b_accepted_geometry_density_receipt_v96",
    "method_identity_coupled_geometry_owner_v96",
    "resolve_coupled_geometry_cpu_execution_v96",
    "resolve_coupled_geometry_cpu_state_v96",
    "rollback_coupled_geometry_cpu_trial_v96",
    "validate_accepted_bed_geometry_receipt_v96",
    "validate_coupled_geometry_cpu_state_v96",
    "validate_v94b_accepted_geometry_density_receipt_v96",
]
