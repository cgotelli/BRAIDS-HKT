"""Bounded Q4/v94 active ``IopSus`` composition for current-only sand.

Locked Delft3D4 FLOW evaluates ``fallve`` with the accepted parent ``Dss``
before ``tram1`` produces a new diameter.  This module preserves that lag:
v92 owns one scalar/exchange pass using parent ``Dss``; one pure v83 producer
then supplies both its suspended capacity and the Q3 bed vector, while its
live ``taurat`` diagnostic produces the candidate ``Dss`` for the next
accepted update.  No second v78 lifecycle or Exner owner is admitted.

The evidence envelope is deliberately small: three pure-sand COMBINE
fractions, TraFrm=-1, SettleFrm=2, iTauCr=1, KMAX=1, exact-zero downhill,
fully-wet fixed grid, MorFac=1, closed/free-export boundaries, and at most
four accepted updates.  This is not product admission.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
from typing import Any

import numpy as np

from .direct_vector_family_live_v67 import METHOD_VAN_RIJN_1993_CURRENT
from .direct_vector_live_taurat_v73 import (
    load_direct_vector_live_taurat_execution_v73,
)
from .direct_vector_wetdry_amr_live_v71 import (
    FACE_CLOSED,
    FACE_INTERNAL,
    FACE_OWNER_OUTLET,
)
from .factcr_lifecycle_v83 import (
    FactCrExecutionV83,
    FactCrProducerReceiptV83,
    ROLE_CPU_ORACLE,
    ROLE_CUDA,
    _digest as _v83_digest,
    critical_shear_fields_v83,
    execution_document_v83,
    produce_factcr_rows_cpu_v83,
    validate_factcr_producer_receipt_v83,
)
from .hles_producer_v90 import CurrentClosureHandoffV90, HlesProducerStateV90
from .q3_exner_adapter_v93 import (
    BACKEND_CPU,
    MORPHOLOGICAL_FACTOR,
    _array_digest,
    _base_topology,
    _finite,
    _host_array,
)
from .q3_mixed_route_exner_adapter_v93 import (
    CPUQ3V84MixedBindingV93,
    CPUQ3V84MixedExnerAdapterV93,
    CPUQ3V84MixedOpenTrialV93,
    CPUQ3V84MixedStateV93,
    validate_cpu_q3_v84_mixed_state_v93,
)
from .q3_producer_lifecycle_binding_v84 import _digest as _v84_digest
from .route_lifecycle_composition_v81 import ROUTE_COMBINE
from .recursive_surface_feedback_v114 import (
    provider_surface_fraction_v114,
    provider_surface_slope_fields_v114,
)
from .suspended_controls_v86 import iopsus_diameter_v86
from .suspended_exner_owner_v93 import (
    SuspendedExnerExecutionV93,
    SuspendedExnerStateV93,
    SuspendedExnerTrialV93,
    begin_suspended_exner_cpu_trial_v93,
    cancel_suspended_exner_trial_v93,
    commit_suspended_exner_trial_v93,
    execution_document_v93,
    load_suspended_exner_checkpoint_v93,
    resolve_suspended_exner_state_v93,
    rollback_suspended_exner_trial_v93,
    suspended_exner_checkpoint_v93,
    validate_suspended_exner_state_v93,
)
from .transported_fluid_settling_v92 import (
    SETTLING_DIAMETER_ACTIVE_IOPSUS_V94,
    SETTLING_DIAMETER_STATIC_PROFILE,
    SUSPENDED_CAPACITY_ACTIVE_V83_V94,
    SettlingDiameterReceiptV92,
    SuspendedCapacityReceiptV92,
    TransportedFluidTrialV92,
    _validate_trial as _validate_v92_trial,
    make_settling_diameter_receipt_v92,
    make_suspended_capacity_receipt_v92,
    validate_settling_diameter_receipt_v92,
    validate_suspended_capacity_receipt_v92,
)


SCHEMA = 1
SCOPE_ID = "braids-q4-flow2dh-iopsus-active-v94-v1"
CHECKPOINT_SCOPE_ID = SCOPE_ID + ":checkpoint"
SEMANTIC_ID = hashlib.sha256(SCOPE_ID.encode("ascii")).hexdigest()
INTEGRATOR = "implicit_fwave_midpoint"
FORMULA_CODE = -1
SETTLING_FORM = 2
KMAX = 1
FRACTION_COUNT = 3
MAX_ACCEPTED_UPDATES = 4
CPU_PROVIDER_KIND = "q4_owned_v83_trafrm_minus1_active_iopsus_face_no_scalar"
GPU_PROVIDER_KIND = CPU_PROVIDER_KIND + "_actual_cuda"
_SHA256 = __import__("re").compile(r"[0-9a-f]{64}\Z")


class ActiveSuspendedControlV94Error(ValueError):
    """A v94 identity, lagged control, provider, or transaction is invalid."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError, OverflowError) as error:
        raise ActiveSuspendedControlV94Error(
            "v94 value is not finite canonical JSON") from error


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _readonly(value: object, shape: tuple[int, ...], name: str, *,
              integer: bool = False, nonnegative: bool = False) -> np.ndarray:
    try:
        result = _host_array(
            value, shape, "v94 "+name, integer=integer,
            nonnegative=nonnegative)
    except Exception as error:
        raise ActiveSuspendedControlV94Error(str(error)) from error
    return result


def _iopsus_profile(value: object, count: int) -> np.ndarray:
    if type(value) is int:
        if value not in (0, 1):
            raise ActiveSuspendedControlV94Error(
                "v94 scalar IopSus must be exact integer 0 or 1")
        raw = np.full(count, value, dtype=np.int32)
    else:
        candidate = np.asarray(value)
        if (candidate.shape != (count,) or candidate.dtype.kind not in "iu" or
                candidate.dtype.kind == "b" or
                np.any((candidate != 0) & (candidate != 1))):
            raise ActiveSuspendedControlV94Error(
                "v94 per-fraction IopSus must be length-ng exact integers 0/1")
        raw = np.ascontiguousarray(candidate, dtype=np.int32)
    raw.setflags(write=False)
    return raw


@dataclass(frozen=True)
class FrozenUniformThresholdReceiptV94:
    """Initializer-owned uniform-D50 threshold reused by every update."""

    factcr_execution_fingerprint: str
    initializer_water_density_kg_m3: float
    initializer_kinematic_viscosity_m2_s: float
    dstar: np.ndarray
    critical_shields: np.ndarray
    unfactored_critical_shear_pa: np.ndarray
    effective_critical_shear_pa: np.ndarray
    receipt_sha256: str


def _frozen_threshold_token_v94(
    value: FrozenUniformThresholdReceiptV94,
) -> str:
    return _fingerprint({
        "scope_id": SCOPE_ID + ":frozen-uniform-threshold",
        "factcr_execution_fingerprint":
            value.factcr_execution_fingerprint,
        "initializer_water_density_kg_m3":
            value.initializer_water_density_kg_m3,
        "initializer_kinematic_viscosity_m2_s":
            value.initializer_kinematic_viscosity_m2_s,
        "arrays_sha256": _array_digest(
            value.dstar, value.critical_shields,
            value.unfactored_critical_shear_pa,
            value.effective_critical_shear_pa),
        "formation_mode": "uniform_D50_initializer_once",
    })


def _make_frozen_threshold_receipt_v94(
    factcr_execution: FactCrExecutionV83, *,
    initializer_water_density_kg_m3: object,
    initializer_kinematic_viscosity_m2_s: object,
) -> FrozenUniformThresholdReceiptV94:
    execution_document_v83(factcr_execution)
    rho = _finite(
        initializer_water_density_kg_m3,
        "initializer water density", positive=True)
    viscosity = _finite(
        initializer_kinematic_viscosity_m2_s,
        "initializer kinematic viscosity", positive=True)
    profile = factcr_execution.lifecycle_execution.base_execution.profile
    if np.any(np.asarray(profile.sediment_density_kg_m3) <= rho):
        raise ActiveSuspendedControlV94Error(
            "v94 initializer water density must remain below sediment "
            "density")
    values = critical_shear_fields_v83(
        d50_m=profile.sediment_d50_m,
        solid_density_kg_m3=profile.sediment_density_kg_m3,
        water_density_kg_m3=np.asarray((rho,)),
        gravity_m_s2=np.asarray((profile.gravity_m_s2,)),
        kinematic_viscosity_m2_s=np.asarray((viscosity,)),
        factcr=factcr_execution.factcr,
        formula_code=FORMULA_CODE)
    arrays = []
    for value in values:
        array = np.ascontiguousarray(value[:, 0])
        array.setflags(write=False)
        arrays.append(array)
    placeholder = FrozenUniformThresholdReceiptV94(
        factcr_execution.fingerprint, rho, viscosity, *arrays, "")
    return FrozenUniformThresholdReceiptV94(
        **{**placeholder.__dict__,
           "receipt_sha256": _frozen_threshold_token_v94(placeholder)})


def validate_frozen_threshold_receipt_v94(
    value: FrozenUniformThresholdReceiptV94,
    factcr_execution: FactCrExecutionV83,
) -> None:
    if (not isinstance(value, FrozenUniformThresholdReceiptV94) or
            value.factcr_execution_fingerprint !=
                factcr_execution.fingerprint or
            value.receipt_sha256 != _frozen_threshold_token_v94(value)):
        raise ActiveSuspendedControlV94Error(
            "v94 frozen uniform threshold receipt differs")
    rebuilt = _make_frozen_threshold_receipt_v94(
        factcr_execution,
        initializer_water_density_kg_m3=
            value.initializer_water_density_kg_m3,
        initializer_kinematic_viscosity_m2_s=
            value.initializer_kinematic_viscosity_m2_s)
    if (rebuilt.receipt_sha256 != value.receipt_sha256 or
            not all(np.array_equal(left, right) for left, right in zip(
                (rebuilt.dstar, rebuilt.critical_shields,
                 rebuilt.unfactored_critical_shear_pa,
                 rebuilt.effective_critical_shear_pa),
                (value.dstar, value.critical_shields,
                 value.unfactored_critical_shear_pa,
                 value.effective_critical_shear_pa)))):
        raise ActiveSuspendedControlV94Error(
            "v94 frozen uniform threshold does not rebuild exactly")


def _frozen_threshold_document_v94(
    value: FrozenUniformThresholdReceiptV94,
) -> dict[str, object]:
    return {
        "factcr_execution_fingerprint":
            value.factcr_execution_fingerprint,
        "initializer_water_density_kg_m3":
            value.initializer_water_density_kg_m3,
        "initializer_kinematic_viscosity_m2_s":
            value.initializer_kinematic_viscosity_m2_s,
        "dstar": value.dstar.tolist(),
        "critical_shields": value.critical_shields.tolist(),
        "unfactored_critical_shear_pa":
            value.unfactored_critical_shear_pa.tolist(),
        "effective_critical_shear_pa":
            value.effective_critical_shear_pa.tolist(),
        "formation_mode": "uniform_D50_initializer_once",
        "receipt_sha256": value.receipt_sha256,
    }


def _v78_execution(value: SuspendedExnerExecutionV93):
    return value.base_execution.base_execution.base_execution


@dataclass(frozen=True)
class ActiveSuspendedControlExecutionV94:
    base_execution: SuspendedExnerExecutionV93
    factcr_execution: FactCrExecutionV83
    live_taurat_execution: Mapping[str, object]
    iopsus_by_fraction: np.ndarray
    suscor: bool
    frozen_uniform_threshold: FrozenUniformThresholdReceiptV94
    semantic_id: str
    fingerprint: str

    @property
    def backend(self) -> str:
        return self.base_execution.backend

    @property
    def fraction_count(self) -> int:
        return self.base_execution.fraction_count

    @property
    def cell_count(self) -> int:
        return self.base_execution.cell_count


def _execution_payload(value: ActiveSuspendedControlExecutionV94) -> dict:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "semantic_id": value.semantic_id,
        "base_v93_execution": execution_document_v93(value.base_execution),
        "factcr_v83_execution": execution_document_v83(value.factcr_execution),
        "frozen_uniform_threshold": _frozen_threshold_document_v94(
            value.frozen_uniform_threshold),
        "live_taurat_v73_execution": dict(value.live_taurat_execution),
        "backend": value.backend,
        "integrator": INTEGRATOR,
        "formula_codes": [FORMULA_CODE]*value.fraction_count,
        "settling_forms": [SETTLING_FORM]*value.fraction_count,
        "iopsus_by_fraction": value.iopsus_by_fraction.tolist(),
        "iopsus_mixed_profiles_supported": True,
        "current_settling_diameter": "accepted_parent_DSS",
        "current_transport_stage": "max(0,taurat-1)",
        "candidate_diameter_visibility": "next_accepted_update_only",
        "inactive_fraction_policy": "retain_accepted_parent_DSS_exactly",
        "factcr_application_count": 1,
        "scalar_lifecycle_owner_count": 1,
        "additional_scalar_lifecycle_count": 0,
        "authoritative_exner_owner_count": 1,
        "suscor": value.suscor,
        "suscor_kmax1_correction_application_count": 0,
        "suspended_vector_construction_owner_count":
            1 if value.suscor else 0,
        "neglect_entrainment": value.base_execution.neglect_entrainment,
        "kmax": KMAX,
        "morphological_factor": 1.0,
        "downhill": "exact_zero_live_taurat_authenticated_but_inert",
        "fraction_count": FRACTION_COUNT,
        "routes": ["TRA_COMBINE"]*FRACTION_COUNT,
        "accepted_update_limit": MAX_ACCEPTED_UPDATES,
        "automatic_cpu_fallback": False,
        "fixed_grid": True,
        "moving_wet_dry": False,
        "product_admission": False,
    }


def resolve_active_suspended_control_execution_v94(
    base_execution: SuspendedExnerExecutionV93, *,
    factcr_execution: FactCrExecutionV83,
    live_taurat_execution: Mapping[str, object],
    iopsus_by_fraction: object = 1,
    suscor: bool = True,
    formula_code: int = FORMULA_CODE,
    initializer_water_density_kg_m3: object | None = None,
    initializer_kinematic_viscosity_m2_s: object = 1.0e-6,
) -> ActiveSuspendedControlExecutionV94:
    """Resolve the exact current-only v94 combination before any child work."""
    execution_document_v93(base_execution)
    execution_document_v83(factcr_execution)
    live = load_direct_vector_live_taurat_execution_v73(
        live_taurat_execution)
    if type(formula_code) is not int or formula_code != FORMULA_CODE:
        raise ActiveSuspendedControlV94Error(
            "v94 admits only current-only TraFrm=-1; -2/-4 remain fail-closed")
    if base_execution.fraction_count != FRACTION_COUNT:
        raise ActiveSuspendedControlV94Error(
            "v94 coupled evidence requires exactly three fractions")
    if not np.all(np.asarray(base_execution.routes) == ROUTE_COMBINE):
        raise ActiveSuspendedControlV94Error(
            "v94 active IopSus requires all fractions on TRA_COMBINE")
    if type(suscor) is not bool:
        raise ActiveSuspendedControlV94Error("v94 SusCor must be Boolean")
    if base_execution.neglect_entrainment and not suscor:
        raise ActiveSuspendedControlV94Error(
            "v94 SusCor=false plus NeglectEntrainment=true is a source-legal "
            "zero-suspended-vector route queued for v94b, not a physical "
            "incompatibility")
    selector = _iopsus_profile(iopsus_by_fraction, FRACTION_COUNT)
    v78 = _v78_execution(base_execution)
    if (factcr_execution.lifecycle_execution.fingerprint != v78.fingerprint or
            base_execution.q3_execution_fingerprint !=
                factcr_execution.fingerprint):
        raise ActiveSuspendedControlV94Error(
            "v94 v83/v92 lifecycle or Q3 execution identity differs")
    expected_role = ROLE_CPU_ORACLE if base_execution.backend == "cpu" else ROLE_CUDA
    if factcr_execution.role != expected_role:
        raise ActiveSuspendedControlV94Error(
            "v94 v83 role differs from the selected backend")
    profile = v78.base_execution.profile
    initial_rho = (profile.water_density_kg_m3
                   if initializer_water_density_kg_m3 is None else
                   initializer_water_density_kg_m3)
    frozen = _make_frozen_threshold_receipt_v94(
        factcr_execution,
        initializer_water_density_kg_m3=initial_rho,
        initializer_kinematic_viscosity_m2_s=
            initializer_kinematic_viscosity_m2_s)
    if profile.settling_form != SETTLING_FORM:
        raise ActiveSuspendedControlV94Error(
            "v94 admits only SettleFrm=2")
    physical = live["parent_execution"]["physical_profile"]
    rows = physical["fraction_methods"]
    if (live["backend"] != base_execution.backend or
            tuple(physical["fraction_ids"]) !=
                tuple(base_execution.ordered_fraction_ids) or
            any(row["method_id"] != METHOD_VAN_RIJN_1993_CURRENT or
                int(row["reference_selector"]) != FORMULA_CODE
                for row in rows) or
            not np.array_equal(np.asarray(physical["d50_m"]),
                               np.asarray(profile.sediment_d50_m)) or
            not np.array_equal(np.asarray(physical["solid_density_kg_m3"]),
                               np.asarray(profile.sediment_density_kg_m3))):
        raise ActiveSuspendedControlV94Error(
            "v94 live v73 family is not the same all-current v83 profile")
    placeholder = ActiveSuspendedControlExecutionV94(
        base_execution, factcr_execution, live, selector, suscor, frozen,
        SEMANTIC_ID, "")
    return ActiveSuspendedControlExecutionV94(
        **{**placeholder.__dict__,
           "fingerprint": _fingerprint(_execution_payload(placeholder))})


def execution_document_v94(value: ActiveSuspendedControlExecutionV94) -> dict:
    if not isinstance(value, ActiveSuspendedControlExecutionV94):
        raise ActiveSuspendedControlV94Error("v94 execution object differs")
    validate_frozen_threshold_receipt_v94(
        value.frozen_uniform_threshold, value.factcr_execution)
    payload = _execution_payload(value)
    if value.fingerprint != _fingerprint(payload):
        raise ActiveSuspendedControlV94Error(
            "v94 execution fingerprint differs")
    return {**payload, "fingerprint": value.fingerprint}


@dataclass(frozen=True)
class ActiveSuspendedControlStateV94:
    execution_fingerprint: str
    base_state: SuspendedExnerStateV93
    suspended_diameter_m: np.ndarray
    frozen_threshold_receipt_sha256: str
    accepted_iopsus_chain_sha256: str
    state_sha256: str

    @property
    def accepted_updates(self) -> int:
        return self.base_state.accepted_updates

    @property
    def time_s(self) -> float:
        return self.base_state.time_s


def _state_token(value: ActiveSuspendedControlStateV94) -> str:
    return _fingerprint({
        "scope_id": SCOPE_ID + ":state",
        "execution_fingerprint": value.execution_fingerprint,
        "base_v93_state_sha256": value.base_state.state_sha256,
        "suspended_diameter_sha256": _array_digest(
            value.suspended_diameter_m),
        "frozen_threshold_receipt_sha256":
            value.frozen_threshold_receipt_sha256,
        "accepted_iopsus_chain_sha256": value.accepted_iopsus_chain_sha256,
    })


def _wrap_state(
    execution: ActiveSuspendedControlExecutionV94,
    base_state: SuspendedExnerStateV93, *,
    suspended_diameter_m: object,
    accepted_iopsus_chain_sha256: str,
) -> ActiveSuspendedControlStateV94:
    execution_document_v94(execution)
    validate_suspended_exner_state_v93(base_state, execution.base_execution)
    diameter = _readonly(
        suspended_diameter_m,
        (execution.fraction_count, execution.cell_count),
        "suspended diameter", nonnegative=True)
    if np.any(diameter <= 0.0):
        raise ActiveSuspendedControlV94Error(
            "v94 suspended diameter must be positive")
    if (_SHA256.fullmatch(accepted_iopsus_chain_sha256) is None or
            base_state.accepted_updates > MAX_ACCEPTED_UPDATES):
        raise ActiveSuspendedControlV94Error(
            "v94 accepted IopSus state identity differs")
    placeholder = ActiveSuspendedControlStateV94(
        execution.fingerprint, base_state, diameter,
        execution.frozen_uniform_threshold.receipt_sha256,
        accepted_iopsus_chain_sha256, "")
    return ActiveSuspendedControlStateV94(
        **{**placeholder.__dict__, "state_sha256": _state_token(placeholder)})


def validate_active_suspended_control_state_v94(
    value: ActiveSuspendedControlStateV94,
    execution: ActiveSuspendedControlExecutionV94,
    provider: object | None = None,
) -> None:
    if (not isinstance(value, ActiveSuspendedControlStateV94) or
            value.execution_fingerprint != execution.fingerprint or
            value.frozen_threshold_receipt_sha256 !=
                execution.frozen_uniform_threshold.receipt_sha256 or
            value.state_sha256 != _state_token(value)):
        raise ActiveSuspendedControlV94Error(
            "v94 state is stale, mutated, or cross-execution")
    rebuilt = _wrap_state(
        execution, value.base_state,
        suspended_diameter_m=value.suspended_diameter_m,
        accepted_iopsus_chain_sha256=value.accepted_iopsus_chain_sha256)
    if rebuilt.state_sha256 != value.state_sha256:
        raise ActiveSuspendedControlV94Error("v94 rebuilt state differs")
    if provider is not None:
        if getattr(provider, "control_execution", None) is not execution:
            raise ActiveSuspendedControlV94Error(
                "v94 provider is bound to a different outer execution")
        validate_suspended_exner_state_v93(
            value.base_state, execution.base_execution, provider)


def resolve_initial_active_suspended_control_state_v94(
    execution: ActiveSuspendedControlExecutionV94, *,
    base_state: SuspendedExnerStateV93 | None = None,
    base_v92_state: object | None = None,
    q3_provider: object,
) -> ActiveSuspendedControlStateV94:
    """Create update-zero state; initial DSS is exactly D50*FacDSS."""
    execution_document_v94(execution)
    if base_state is None:
        if base_v92_state is None:
            raise ActiveSuspendedControlV94Error(
                "v94 initial state requires v93 or v92 parent state")
        base_state = resolve_suspended_exner_state_v93(
            execution.base_execution, base_state=base_v92_state,
            q3_provider=q3_provider)
    elif base_v92_state is not None:
        raise ActiveSuspendedControlV94Error(
            "v94 initial state accepts one parent-state form")
    validate_suspended_exner_state_v93(
        base_state, execution.base_execution, q3_provider)
    if base_state.accepted_updates != 0:
        raise ActiveSuspendedControlV94Error(
            "v94 fresh state must start at update zero")
    profile = _v78_execution(execution.base_execution).base_execution.profile
    diameter = np.broadcast_to(
        np.asarray(profile.sediment_d50_m)[:, None]*
        np.asarray(profile.settling_fac_dss)[:, None],
        (execution.fraction_count, execution.cell_count))
    chain = _fingerprint({
        "scope_id": SCOPE_ID,
        "execution_fingerprint": execution.fingerprint,
        "initial_v93_state_sha256": base_state.state_sha256,
        "initial_DSS_sha256": _array_digest(diameter),
        "iopsus_by_fraction": execution.iopsus_by_fraction.tolist(),
    })
    return _wrap_state(
        execution, base_state, suspended_diameter_m=diameter,
        accepted_iopsus_chain_sha256=chain)


@dataclass(frozen=True)
class CPUQ3V83ActiveFaceTrialV94:
    parent_state_sha256: str
    producer_receipt: FactCrProducerReceiptV83
    accepted_endpoint_q_sha256: str
    accepted_hydraulic_face_receipt_sha256: str
    dt_s: float
    morphological_factor: float
    face_flux_kg_m_s: np.ndarray
    cell_delta_kg_m2: np.ndarray
    inventory_fill: np.ndarray
    accepted_cell_availability: np.ndarray
    named_boundary_transfer_kg: np.ndarray
    named_boundary_absolute_kg: np.ndarray
    depletion_roundoff_positive_in_kg: np.ndarray
    conservation_residual_kg: np.ndarray
    constitutive_producer_count: np.ndarray
    route_handoff_count: np.ndarray
    face_reconstruction_count: np.ndarray
    bed_slope_count: np.ndarray
    boundary_owner_count: np.ndarray
    inventory_fill_count: np.ndarray
    face_exner_count: np.ndarray
    extra_scalar_lifecycle_count: np.ndarray
    live_taurat: np.ndarray
    transport_stage: np.ndarray
    candidate_suspended_diameter_m: np.ndarray
    iopsus_application_count: np.ndarray
    factcr_application_count: np.ndarray
    suscor_correction_application_count: np.ndarray
    suspended_vector_construction_owner_count: np.ndarray
    trial_sha256: str


def _face_token(value: CPUQ3V83ActiveFaceTrialV94) -> str:
    return _fingerprint({
        "scope_id": SCOPE_ID + ":face-trial",
        "parent_state_sha256": value.parent_state_sha256,
        "producer_receipt_sha256": value.producer_receipt.receipt_sha256,
        "accepted_endpoint_q_sha256": value.accepted_endpoint_q_sha256,
        "accepted_hydraulic_face_receipt_sha256":
            value.accepted_hydraulic_face_receipt_sha256,
        "dt_s": value.dt_s,
        "morphological_factor": value.morphological_factor,
        "arrays_sha256": _array_digest(
            value.face_flux_kg_m_s, value.cell_delta_kg_m2,
            value.inventory_fill, value.accepted_cell_availability,
            value.named_boundary_transfer_kg,
            value.named_boundary_absolute_kg,
            value.depletion_roundoff_positive_in_kg,
            value.conservation_residual_kg,
            value.constitutive_producer_count, value.route_handoff_count,
            value.face_reconstruction_count, value.bed_slope_count,
            value.boundary_owner_count, value.inventory_fill_count,
            value.face_exner_count, value.extra_scalar_lifecycle_count,
            value.live_taurat, value.transport_stage,
            value.candidate_suspended_diameter_m,
            value.iopsus_application_count,
            value.factcr_application_count,
            value.suscor_correction_application_count,
            value.suspended_vector_construction_owner_count),
        "single_v83_producer": True,
        "single_outer_face_owner": True,
        "child_scalar_lifecycle": False,
    })


class CPUQ3V83ActiveExnerAdapterV94(CPUQ3V84MixedExnerAdapterV93):
    """Exact CPU provider sharing one pure v83 producer with v92 and v93."""

    backend = BACKEND_CPU
    provider_kind = CPU_PROVIDER_KIND

    def __init__(
        self, *, outer_execution: ActiveSuspendedControlExecutionV94,
        binding: CPUQ3V84MixedBindingV93,
        state: CPUQ3V84MixedStateV93,
    ):
        execution_document_v94(outer_execution)
        if (outer_execution.backend != "cpu" or
                type(binding) is not CPUQ3V84MixedBindingV93):
            raise ActiveSuspendedControlV94Error(
                "v94 CPU provider requires exact CPU v94/binding objects")
        base = outer_execution.base_execution
        if (binding.topology_id != base.q3_topology_id or
                binding.binding_sha256 != base.q3_binding_sha256 or
                binding.boundary_names != tuple(base.q3_boundary_names)):
            raise ActiveSuspendedControlV94Error(
                "v94 CPU provider topology, binding, or boundary differs")
        if (outer_execution.live_taurat_execution[
                "bed_slope_execution_fingerprint"] !=
                binding.bed_slope_execution["fingerprint"]):
            raise ActiveSuspendedControlV94Error(
                "v94 live-taurat and bound slope executions differ")
        if not np.array_equal(
                state.bulk_density_kg_m3,
                np.asarray(binding.boundary_execution[
                    "bulk_density_kg_m3"], dtype=np.float64)):
            raise ActiveSuspendedControlV94Error(
                "v94 CPU bulk density differs from bound v70 physics")
        self.control_execution = outer_execution
        self.outer_execution = base
        self.binding = binding
        self.boundary_names = binding.boundary_names
        self._validate_topology_and_profile()
        validate_cpu_q3_v84_mixed_state_v93(
            state, execution_fingerprint=self.execution_fingerprint,
            binding=binding)
        self._state = state
        self._open = None
        self._prepared = None
        self._closed = None
        self._armed: dict[str, Any] | None = None
        self._producer_receipt: FactCrProducerReceiptV83 | None = None
        self._capacity_receipt: SuspendedCapacityReceiptV92 | None = None

    @property
    def execution_fingerprint(self) -> str:
        return self.control_execution.factcr_execution.fingerprint

    def _validate_topology_and_profile(self) -> None:
        base = _base_topology(self.outer_execution)
        topology = self.binding.topology
        mapping = np.asarray(self.binding.family_boundary_faces)
        local_boundary = np.asarray(self.binding.boundary_topology.boundary_index)
        expected_faces = np.flatnonzero(base.boundary_index >= 0)
        if not all((
            topology.ncell == base.cell_count,
            topology.nface == base.face_count,
            np.array_equal(topology.owner, base.owner),
            np.array_equal(topology.neighbour, base.neighbour),
            np.array_equal(topology.cell_area_m2, base.cell_area_m2),
            np.array_equal(self.binding.normal, base.normal),
            np.array_equal(self.binding.face_length_m, base.face_length_m),
            np.array_equal(self.binding.cell_face_offset, base.cell_face_offset),
            np.array_equal(self.binding.cell_faces, base.cell_faces),
            np.array_equal(self.binding.signs, base.cell_face_sign),
            base.boundary_count == len(self.boundary_names),
            np.array_equal(mapping, expected_faces),
            np.array_equal(local_boundary, base.boundary_index[mapping]),
            bool(np.all(topology.wet)),
            np.all(self.binding.downhill == 0.0),
        )):
            raise ActiveSuspendedControlV94Error(
                "v94 CPU topology/boundary/exact-zero slope differs from v92")
        profile = base.profile
        physical = self.control_execution.live_taurat_execution[
            "parent_execution"]["physical_profile"]
        if (tuple(physical["fraction_ids"]) != self.ordered_fraction_ids or
                tuple(self.binding.bed_slope_execution[
                    "ordered_fraction_ids"]) != self.ordered_fraction_ids or
                not np.array_equal(np.asarray(physical["d50_m"]),
                                   np.asarray(profile.sediment_d50_m)) or
                not np.array_equal(
                    np.asarray(physical["solid_density_kg_m3"]),
                    np.asarray(profile.sediment_density_kg_m3))):
            raise ActiveSuspendedControlV94Error(
                "v94 CPU physical profile differs from v92")

    def arm_capacity(
        self, *, control_parent: ActiveSuspendedControlStateV94,
        q3_inputs: Mapping[str, object],
        cancelled: Callable[[], bool] = lambda: False,
    ) -> None:
        validate_active_suspended_control_state_v94(
            control_parent, self.control_execution, self)
        if (self._armed is not None or self._producer_receipt is not None or
                self._capacity_receipt is not None or self._open is not None or
                self._prepared is not None or self._closed is not None):
            raise RuntimeError("v94 CPU provider already has a transaction")
        if not isinstance(q3_inputs, Mapping) or not callable(cancelled):
            raise ActiveSuspendedControlV94Error(
                "v94 CPU producer inputs or cancellation callback differ")
        required = {
            "observation_height_m", "additional_shear_stress_pa",
            "sediment_d90_m", "molecular_schmidt", "aks_factor",
            "maximum_volume_concentration", "minimum_depth_m",
            "bed_factor", "suspended_factor", "roughness_z0_m",
        }
        if set(q3_inputs) != required:
            raise ActiveSuspendedControlV94Error(
                "v94 CPU producer input inventory differs")
        self._armed = {
            "control_parent": control_parent,
            "q3_inputs": dict(q3_inputs),
            "cancelled": cancelled,
        }

    def disarm_capacity(self) -> None:
        if self._open is not None or self._prepared is not None or \
                self._closed is not None:
            raise RuntimeError("v94 CPU cannot disarm an open transaction")
        self._armed = None
        self._producer_receipt = None
        self._capacity_receipt = None

    def resolve_armed_capacity(
        self, *, execution, parent, accepted_q,
        settling_velocity_m_s, settling_diameter_receipt,
        surface_fraction, shear_velocity_m_s,
    ) -> SuspendedCapacityReceiptV92:
        if self._armed is None or self._producer_receipt is not None:
            raise ActiveSuspendedControlV94Error(
                "v94 CPU capacity provider is not singly armed")
        armed = self._armed
        control_parent = armed["control_parent"]
        if (execution is not self.outer_execution.base_execution or
                parent is not control_parent.base_state.base_state):
            raise ActiveSuspendedControlV94Error(
                "v94 CPU capacity parent or execution differs")
        if armed["cancelled"]():
            raise InterruptedError("v94 CPU cancelled before v83 production")
        active_iopsus = bool(np.any(
            self.control_execution.iopsus_by_fraction == 1))
        expected_source = (SETTLING_DIAMETER_ACTIVE_IOPSUS_V94
                           if active_iopsus else
                           SETTLING_DIAMETER_STATIC_PROFILE)
        expected_source_state = (control_parent.state_sha256
                                 if active_iopsus else
                                 execution.fingerprint)
        if (not isinstance(settling_diameter_receipt,
                           SettlingDiameterReceiptV92) or
                settling_diameter_receipt.source_kind != expected_source or
                settling_diameter_receipt.source_state_sha256 !=
                    expected_source_state):
            raise ActiveSuspendedControlV94Error(
                "v94 CPU parent DSS receipt differs")
        validate_settling_diameter_receipt_v92(
            settling_diameter_receipt, execution, parent)
        fraction = _readonly(
            surface_fraction,
            (self.control_execution.fraction_count,
             self.control_execution.cell_count),
            "surface fraction", nonnegative=True)
        if not np.array_equal(
                fraction, provider_surface_fraction_v114(self)):
            raise ActiveSuspendedControlV94Error(
                "v94 CPU v92 and Q3 surface owners differ")
        _readonly(
            shear_velocity_m_s, (self.control_execution.cell_count,),
            "accepted shear velocity", nonnegative=True)
        q = _readonly(
            accepted_q, (3, self.control_execution.cell_count),
            "accepted hydraulic endpoint")
        v92_parent = control_parent.base_state.base_state
        profile = _v78_execution(
            self.control_execution.base_execution).base_execution.profile
        inputs = armed["q3_inputs"]
        receipt = produce_factcr_rows_cpu_v83(
            self.control_execution.factcr_execution,
            depth_after_m=q[0], endpoint_momentum_m2_s=q[1:],
            observation_height_m=inputs["observation_height_m"],
            water_density_kg_m3=v92_parent.water_density_cell_kg_m3,
            gravity_m_s2=np.full(
                self.control_execution.cell_count, profile.gravity_m_s2),
            # Locked FLOW uses global initializer vicmol in eqtran/tram1's
            # background molecular diffusion.  Dynamic T/S viscosity remains
            # owned by v92 SettleFrm=2 and never enters this v83 role.
            kinematic_viscosity_m2_s=np.full(
                self.control_execution.cell_count,
                self.control_execution.frozen_uniform_threshold.
                initializer_kinematic_viscosity_m2_s),
            additional_shear_stress_pa=
                inputs["additional_shear_stress_pa"],
            sediment_d50_m=profile.sediment_d50_m,
            sediment_d90_m=inputs["sediment_d90_m"],
            sediment_density_kg_m3=profile.sediment_density_kg_m3,
            molecular_schmidt=inputs["molecular_schmidt"],
            aks_factor=inputs["aks_factor"],
            maximum_volume_concentration=
                inputs["maximum_volume_concentration"],
            minimum_depth_m=inputs["minimum_depth_m"],
            bed_factor=inputs["bed_factor"],
            suspended_factor=inputs["suspended_factor"],
            settling_velocity_m_s=settling_velocity_m_s,
            roughness_z0_m=inputs["roughness_z0_m"],
            surface_fraction=fraction,
            frozen_uniform_threshold=(
                self.control_execution.frozen_uniform_threshold.dstar,
                self.control_execution.frozen_uniform_threshold.
                    critical_shields,
                self.control_execution.frozen_uniform_threshold.
                    unfactored_critical_shear_pa,
                self.control_execution.frozen_uniform_threshold.
                    effective_critical_shear_pa))
        validate_factcr_producer_receipt_v83(
            receipt, self.control_execution.factcr_execution)
        frozen = self.control_execution.frozen_uniform_threshold
        frozen_arrays = (
            frozen.dstar, frozen.critical_shields,
            frozen.unfactored_critical_shear_pa,
            frozen.effective_critical_shear_pa)
        produced_arrays = (
            receipt.dstar, receipt.critical_shields,
            receipt.unfactored_critical_shear_pa,
            receipt.effective_critical_shear_pa)
        if not all(np.array_equal(
                produced,
                np.broadcast_to(expected[:, None], produced.shape))
                for produced, expected in zip(
                    produced_arrays, frozen_arrays)):
            raise ActiveSuspendedControlV94Error(
                "v94 CPU producer did not consume the frozen initializer "
                "threshold")
        expected_background_viscosity = np.full(
            self.control_execution.cell_count,
            frozen.initializer_kinematic_viscosity_m2_s)
        if receipt.background_kinematic_viscosity_sha256 != \
                _v83_digest(expected_background_viscosity):
            raise ActiveSuspendedControlV94Error(
                "v94 CPU equilibrium diffusion did not consume initializer "
                "vicmol")
        capacity = make_suspended_capacity_receipt_v92(
            execution, parent, accepted_q=q,
            settling_diameter_receipt=settling_diameter_receipt,
            suspended_capacity_kg_m_s=receipt.lifecycle_capacity_kg_m_s,
            source_kind=SUSPENDED_CAPACITY_ACTIVE_V83_V94,
            source_receipt_sha256=receipt.receipt_sha256)
        validate_suspended_capacity_receipt_v92(
            capacity, execution, parent, accepted_q=q,
            settling_diameter_receipt=settling_diameter_receipt)
        if armed["cancelled"]():
            raise InterruptedError("v94 CPU cancelled after v83 production")
        self._producer_receipt = receipt
        self._capacity_receipt = capacity
        return capacity

    def _candidate_diameter(
        self, parent: ActiveSuspendedControlStateV94,
        producer: FactCrProducerReceiptV83,
    ) -> tuple[np.ndarray, np.ndarray]:
        stage = np.ascontiguousarray(
            np.maximum(0.0, producer.producer_diagnostics[2]-1.0))
        candidate = np.array(parent.suspended_diameter_m, copy=True, order="C")
        profile = _v78_execution(
            self.control_execution.base_execution).base_execution.profile
        d50 = np.broadcast_to(
            np.asarray(profile.sediment_d50_m)[:, None], candidate.shape)
        for grain, active in enumerate(
                self.control_execution.iopsus_by_fraction):
            if int(active) == 1:
                candidate[grain] = iopsus_diameter_v86(
                    formula_code=FORMULA_CODE, iopsus=1,
                    fixed_dss_m=parent.suspended_diameter_m[grain],
                    d50_m=d50[grain], transport_stage=stage[grain])
        stage.setflags(write=False)
        candidate = np.ascontiguousarray(candidate)
        candidate.setflags(write=False)
        return stage, candidate

    def _face_trial(
        self, *, producer: FactCrProducerReceiptV83,
        v92_trial: TransportedFluidTrialV92,
        stage: np.ndarray, candidate_dss: np.ndarray,
    ) -> CPUQ3V83ActiveFaceTrialV94:
        from braids_gpu.flow_bed_slope_composition import (
            DONOR_NEIGHBOUR, DONOR_NONE, DONOR_OWNER, DONOR_TRACE,
            PortableFlowBedSlopeFaceComposer,
        )
        ng = self.control_execution.fraction_count
        nc, nf = self.binding.topology.ncell, self.binding.topology.nface
        cell_vector = np.ascontiguousarray(producer.bed_capacity_kg_m_s)
        mapping = np.asarray(self.binding.family_boundary_faces)
        hydraulic_boundary = np.ascontiguousarray(
            v92_trial.hydraulic_face_receipt.
            accepted_water_flux_m2_s[mapping])
        role, trace = self._dynamic_roles(
            hydraulic_boundary=hydraulic_boundary,
            cell_vector=cell_vector)
        topology = self.binding.topology
        safe_other = np.clip(topology.neighbour, 0, nc-1)
        left = np.ascontiguousarray(cell_vector[:, :, topology.owner])
        right = np.ascontiguousarray(cell_vector[:, :, safe_other])
        trace_bed = np.ascontiguousarray(trace)

        # V73's exact projection is carried even though the admitted downhill
        # vector is exact zero and therefore produces no numerical rotation.
        taurat = np.ascontiguousarray(producer.producer_diagnostics[2])
        stress = np.zeros((ng, nf), dtype=np.float64)
        internal = role == FACE_INTERNAL
        stress[:, internal] = 0.5*(
            taurat[:, topology.owner[internal]]+
            taurat[:, topology.neighbour[internal]])
        outlet = role == FACE_OWNER_OUTLET
        stress[:, outlet] = taurat[:, topology.owner[outlet]]
        stress[:, role == FACE_CLOSED] = 0.0
        neutral = np.ones((ng, nf), dtype=np.float64)
        hiding, surface_mean = provider_surface_slope_fields_v114(
            self, topology, np)
        d50 = np.ascontiguousarray(np.asarray(
            _base_topology(self.outer_execution).profile.sediment_d50_m,
            dtype=np.float64))
        composer = PortableFlowBedSlopeFaceComposer(
            self.binding.bed_slope_execution)
        common = dict(
            left_bed=left, right_bed=right, owner_bed=left,
            trace_bed=trace_bed,
            owner=np.ascontiguousarray(topology.owner, dtype=np.int32),
            neighbour=np.ascontiguousarray(topology.neighbour,
                                            dtype=np.int32),
            role=np.ascontiguousarray(role, dtype=np.int32),
            normal=np.ascontiguousarray(self.binding.normal),
            downhill=np.ascontiguousarray(self.binding.downhill),
            stress=stress, depth=neutral, d50=d50,
            mean_diameter=surface_mean, hiding=hiding)
        provisional = composer.compose(
            availability=np.ones((ng, nc), dtype=np.float64), **common)
        dt = _finite(v92_trial.hydraulic_receipt.dt_s, "accepted dt",
                     positive=True)
        donor_cell = np.full((ng, nf), -1, dtype=np.int32)
        donor_cell[provisional.donor == DONOR_OWNER] = np.broadcast_to(
            topology.owner, donor_cell.shape)[
                provisional.donor == DONOR_OWNER]
        donor_cell[provisional.donor == DONOR_NEIGHBOUR] = np.broadcast_to(
            safe_other, donor_cell.shape)[
                provisional.donor == DONOR_NEIGHBOUR]
        donor_cell[provisional.donor == DONOR_TRACE] = np.broadcast_to(
            topology.owner, donor_cell.shape)[
                provisional.donor == DONOR_TRACE]
        donor_cell[provisional.donor == DONOR_NONE] = -1
        outgoing = np.zeros((ng, nc), dtype=np.float64)
        for face in range(nf):
            for grain in range(ng):
                cell = int(donor_cell[grain, face])
                value = float(provisional.flux[grain, face])
                is_outgoing = (
                    cell >= 0 and
                    ((int(role[face]) == FACE_INTERNAL and value != 0.0) or
                     (int(role[face]) == FACE_OWNER_OUTLET and value > 0.0)))
                if is_outgoing:
                    outgoing[grain, cell] += (
                        abs(value)*self.binding.face_length_m[face]*dt /
                        topology.cell_area_m2[cell])
        fill = np.ones((ng, nc), dtype=np.float64)
        fill[self._state.authoritative_bed_mass_kg_m2 == 0.0] = 0.0
        demand = outgoing > 0.0
        fill[demand] = np.minimum(
            1.0, self._state.authoritative_bed_mass_kg_m2[demand]/
            outgoing[demand])
        accepted = composer.compose(
            availability=np.ascontiguousarray(fill), **common)
        flux = np.ascontiguousarray(accepted.flux)
        delta = np.zeros((ng, nc), dtype=np.float64)
        named = np.zeros((ng, len(self.boundary_names)), dtype=np.float64)
        absolute = np.zeros_like(named)
        inverse = {int(face): local for local, face in enumerate(mapping)}
        for face in range(nf):
            amount = flux[:, face]*self.binding.face_length_m[face]*dt
            owner = int(topology.owner[face])
            delta[:, owner] -= amount/topology.cell_area_m2[owner]
            if int(role[face]) == FACE_INTERNAL:
                other = int(topology.neighbour[face])
                delta[:, other] += amount/topology.cell_area_m2[other]
            else:
                local = inverse.get(face)
                if local is None:
                    raise ActiveSuspendedControlV94Error(
                        "v94 boundary face lacks a logical owner")
                boundary = int(
                    self.binding.boundary_topology.boundary_index[local])
                named[:, boundary] += amount
                absolute[:, boundary] += np.abs(amount)
        candidate = self._state.authoritative_bed_mass_kg_m2+delta
        tolerance = 4096.0*np.finfo(np.float64).eps*np.maximum(
            1.0, np.abs(self._state.authoritative_bed_mass_kg_m2))
        if np.any(candidate < -tolerance):
            raise ActiveSuspendedControlV94Error(
                "v94 inventory limiter produced a negative bed")
        before = np.array(delta, copy=True)
        negative = candidate < 0.0
        delta[negative] = -self._state.authoritative_bed_mass_kg_m2[negative]
        correction = delta-before
        depletion = np.sum(
            correction*topology.cell_area_m2[None, :], axis=1)
        if np.any(self._state.authoritative_bed_mass_kg_m2+delta < 0.0):
            raise ActiveSuspendedControlV94Error(
                "v94 depletion correction did not close exactly")
        residual = np.sum(
            delta*topology.cell_area_m2[None, :], axis=1)+\
            np.sum(named, axis=1)-depletion
        scale = np.maximum(
            1.0, np.sum(np.abs(delta*topology.cell_area_m2[None, :]),
                        axis=1)+np.sum(absolute, axis=1)+depletion)
        if np.any(np.abs(residual) >
                  8192.0*np.finfo(np.float64).eps*scale):
            raise ActiveSuspendedControlV94Error(
                "v94 face/Exner ledger does not close")

        ones = np.ones(ng, dtype=np.int32)
        zeros = np.zeros(ng, dtype=np.int32)
        iopsus_count = np.broadcast_to(
            self.control_execution.iopsus_by_fraction[:, None],
            (ng, nc)).copy()
        vector_count = np.full(
            (ng, nc), int(self.control_execution.suscor), dtype=np.int32)
        correction_count = np.zeros((ng, nc), dtype=np.int32)
        values = (
            flux, delta, fill, fill, named, absolute, depletion, residual,
            ones, ones, ones, zeros, ones, ones, ones, zeros,
            taurat, stage, candidate_dss, iopsus_count,
            producer.factcr_application_count, correction_count,
            vector_count,
        )
        arrays = []
        for value in values:
            array = np.ascontiguousarray(value)
            array.setflags(write=False)
            arrays.append(array)
        placeholder = CPUQ3V83ActiveFaceTrialV94(
            self._state.state_sha256, producer,
            _v84_digest(
                v92_trial.candidate.q[0], v92_trial.candidate.q[1:]),
            v92_trial.hydraulic_face_receipt.receipt_sha256,
            dt, MORPHOLOGICAL_FACTOR, *arrays, "")
        return CPUQ3V83ActiveFaceTrialV94(
            **{**placeholder.__dict__,
               "trial_sha256": _face_token(placeholder)})

    def _validate_face_trial(self, value: CPUQ3V83ActiveFaceTrialV94) -> None:
        if (not isinstance(value, CPUQ3V83ActiveFaceTrialV94) or
                value.parent_state_sha256 != self._state.state_sha256 or
                value.dt_s <= 0.0 or
                value.morphological_factor != MORPHOLOGICAL_FACTOR or
                value.trial_sha256 != _face_token(value) or
                value.producer_receipt is not self._producer_receipt or
                np.any(value.extra_scalar_lifecycle_count != 0) or
                np.any(value.suscor_correction_application_count != 0)):
            raise ActiveSuspendedControlV94Error(
                "v94 CPU face trial is stale or duplicates owned work")
        validate_factcr_producer_receipt_v83(
            value.producer_receipt,
            self.control_execution.factcr_execution)
        if (value.producer_receipt.active_fraction_sha256 !=
                _v83_digest(provider_surface_fraction_v114(self)) or
                not np.array_equal(
                    value.factcr_application_count,
                    value.producer_receipt.factcr_application_count) or
                not np.array_equal(
                    value.iopsus_application_count,
                    np.broadcast_to(
                        self.control_execution.iopsus_by_fraction[:, None],
                        value.iopsus_application_count.shape)) or
                np.any(value.suspended_vector_construction_owner_count !=
                       int(self.control_execution.suscor))):
            raise ActiveSuspendedControlV94Error(
                "v94 CPU producer/control owner counts differ")
        armed = self._armed
        if armed is None:
            raise ActiveSuspendedControlV94Error(
                "v94 CPU face trial lost its parent control owner")
        expected_stage, expected_dss = self._candidate_diameter(
            armed["control_parent"], value.producer_receipt)
        if (not np.array_equal(value.transport_stage, expected_stage) or
                not np.array_equal(
                    value.candidate_suspended_diameter_m, expected_dss) or
                not np.array_equal(
                    value.live_taurat,
                    value.producer_receipt.producer_diagnostics[2])):
            raise ActiveSuspendedControlV94Error(
                "v94 CPU transport stage or candidate DSS differs")

    def begin(
        self, *, outer_parent, parent_hles, handoff,
        v92_trial: TransportedFluidTrialV92,
        cancelled: Callable[[], bool] = lambda: False,
    ) -> CPUQ3V84MixedOpenTrialV93:
        if not callable(cancelled):
            raise TypeError("cancelled must be callable")
        if (self._armed is None or self._producer_receipt is None or
                self._capacity_receipt is None):
            raise ActiveSuspendedControlV94Error(
                "v94 CPU provider capacity was not resolved once")
        control_parent = self._armed["control_parent"]
        if (outer_parent is not control_parent.base_state or
                self._open is not None or self._prepared is not None or
                self._closed is not None):
            raise ActiveSuspendedControlV94Error(
                "v94 CPU provider parent or transaction differs")
        _validate_v92_trial(
            self.outer_execution.base_execution, outer_parent.base_state,
            parent_hles, handoff, v92_trial)
        self.validate_current_bed_clock(outer_parent)
        if (v92_trial.suspended_capacity_receipt is not
                self._capacity_receipt or
                v92_trial.suspended_capacity_receipt.source_receipt_sha256 !=
                    self._producer_receipt.receipt_sha256 or
                v92_trial.settling_diameter_receipt.source_kind !=
                    (SETTLING_DIAMETER_ACTIVE_IOPSUS_V94
                     if np.any(self.control_execution.iopsus_by_fraction)
                     else SETTLING_DIAMETER_STATIC_PROFILE) or
                self._producer_receipt.accepted_endpoint_sha256 !=
                    _v83_digest(v92_trial.candidate.q) or
                self._producer_receipt.incoming_settling_velocity_sha256 !=
                    _v83_digest(v92_trial.suspended_trial.base_trial.
                                settling_velocity_m_s)):
            raise ActiveSuspendedControlV94Error(
                "v94 CPU v92 did not consume the armed producer/parent DSS")
        if cancelled():
            raise InterruptedError("v94 CPU cancelled before Q3 face work")
        try:
            stage, candidate_dss = self._candidate_diameter(
                control_parent, self._producer_receipt)
            face = self._face_trial(
                producer=self._producer_receipt, v92_trial=v92_trial,
                stage=stage, candidate_dss=candidate_dss)
            self._validate_face_trial(face)
            receipt = self._make_receipt(
                outer_parent=outer_parent, v92_trial=v92_trial,
                face_trial=face)
            handle = CPUQ3V84MixedOpenTrialV93(
                self._state, face, receipt, "")
            from .q3_mixed_route_exner_adapter_v93 import _open_token
            handle = CPUQ3V84MixedOpenTrialV93(
                handle.parent_state, handle.face_trial, handle.receipt,
                _open_token(handle))
            if cancelled():
                raise InterruptedError(
                    "v94 CPU cancelled before Q3 trial publication")
            self._open = handle
            return handle
        except Exception:
            self._open = None
            self._prepared = None
            self._armed = None
            self._producer_receipt = None
            self._capacity_receipt = None
            raise

    def publish_commit(self, value):
        result = super().publish_commit(value)
        self._armed = None
        self._producer_receipt = None
        self._capacity_receipt = None
        return result

    def rollback(self, value):
        result = super().rollback(value)
        self._armed = None
        self._producer_receipt = None
        self._capacity_receipt = None
        return result

    def cancel(self, value):
        return self.rollback(value)

    def checkpoint(self) -> dict[str, object]:
        # V93 snapshots the committed provider after v92 has resolved the
        # armed capacity but before this provider opens face/Exner work.  The
        # transient arm is deliberately absent from the committed checkpoint.
        return super().checkpoint()

    def restore_checkpoint(self, value: Mapping[str, object]):
        self._armed = None
        self._producer_receipt = None
        self._capacity_receipt = None
        return super().restore_checkpoint(value)


@dataclass(frozen=True)
class ActiveSuspendedControlOwnerCountsV94:
    settling_diameter_consumer_count: np.ndarray
    capacity_producer_count: np.ndarray
    factcr_application_count: np.ndarray
    scalar_lifecycle_count: np.ndarray
    additional_scalar_lifecycle_count: np.ndarray
    authoritative_exner_count: np.ndarray
    iopsus_application_count: np.ndarray
    suscor_correction_application_count: np.ndarray
    suspended_vector_construction_owner_count: np.ndarray


@dataclass(frozen=True)
class ActiveSuspendedControlTrialV94:
    execution_fingerprint: str
    parent_state_sha256: str
    base_v93_trial: SuspendedExnerTrialV93
    settling_diameter_receipt: SettlingDiameterReceiptV92
    suspended_capacity_receipt: SuspendedCapacityReceiptV92
    producer_receipt: FactCrProducerReceiptV83
    live_taurat: np.ndarray
    transport_stage: np.ndarray
    candidate_suspended_diameter_m: np.ndarray
    owner_counts: ActiveSuspendedControlOwnerCountsV94
    candidate: ActiveSuspendedControlStateV94
    backend: str
    trial_sha256: str


def _trial_token(value: ActiveSuspendedControlTrialV94) -> str:
    counts = value.owner_counts
    return _fingerprint({
        "scope_id": SCOPE_ID + ":trial",
        "execution_fingerprint": value.execution_fingerprint,
        "parent_state_sha256": value.parent_state_sha256,
        "base_v93_trial_sha256": value.base_v93_trial.trial_sha256,
        "settling_diameter_receipt_sha256":
            value.settling_diameter_receipt.receipt_sha256,
        "suspended_capacity_receipt_sha256":
            value.suspended_capacity_receipt.receipt_sha256,
        "producer_receipt_sha256": value.producer_receipt.receipt_sha256,
        "candidate_state_sha256": value.candidate.state_sha256,
        "arrays_sha256": _array_digest(
            value.live_taurat, value.transport_stage,
            value.candidate_suspended_diameter_m,
            counts.settling_diameter_consumer_count,
            counts.capacity_producer_count,
            counts.factcr_application_count,
            counts.scalar_lifecycle_count,
            counts.additional_scalar_lifecycle_count,
            counts.authoritative_exner_count,
            counts.iopsus_application_count,
            counts.suscor_correction_application_count,
            counts.suspended_vector_construction_owner_count),
        "backend": value.backend,
    })


def _candidate_diameter_v94(
    execution: ActiveSuspendedControlExecutionV94,
    parent: ActiveSuspendedControlStateV94,
    producer: FactCrProducerReceiptV83,
) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild the lagged IopSus result from authenticated parent/v83 data."""
    validate_factcr_producer_receipt_v83(
        producer, execution.factcr_execution)
    stage = np.ascontiguousarray(
        np.maximum(0.0, producer.producer_diagnostics[2]-1.0))
    candidate = np.array(parent.suspended_diameter_m, copy=True, order="C")
    profile = _v78_execution(
        execution.base_execution).base_execution.profile
    d50 = np.broadcast_to(
        np.asarray(profile.sediment_d50_m)[:, None], candidate.shape)
    for grain, active in enumerate(execution.iopsus_by_fraction):
        if int(active) == 1:
            candidate[grain] = iopsus_diameter_v86(
                formula_code=FORMULA_CODE, iopsus=1,
                fixed_dss_m=parent.suspended_diameter_m[grain],
                d50_m=d50[grain], transport_stage=stage[grain])
    stage.setflags(write=False)
    candidate.setflags(write=False)
    return stage, candidate


def _validate_trial(
    execution: ActiveSuspendedControlExecutionV94,
    parent: ActiveSuspendedControlStateV94,
    parent_hles: HlesProducerStateV90,
    handoff: CurrentClosureHandoffV90,
    trial: ActiveSuspendedControlTrialV94,
) -> None:
    if not isinstance(trial, ActiveSuspendedControlTrialV94):
        raise ActiveSuspendedControlV94Error("v94 trial type differs")
    validate_active_suspended_control_state_v94(
        parent, execution, trial.base_v93_trial.q3_provider)
    expected_backend = ("cpu_oracle" if execution.backend == "cpu" else
                        "actual_cuda")
    if (trial.execution_fingerprint != execution.fingerprint or
            trial.parent_state_sha256 != parent.state_sha256 or
            trial.trial_sha256 != _trial_token(trial) or
            trial.backend != expected_backend):
        raise ActiveSuspendedControlV94Error(
            "v94 trial is stale, mutated, or cross-execution")
    v92_trial = trial.base_v93_trial.base_v92_trial
    validate_settling_diameter_receipt_v92(
        trial.settling_diameter_receipt,
        execution.base_execution.base_execution,
        parent.base_state.base_state)
    active_iopsus = bool(np.any(execution.iopsus_by_fraction == 1))
    expected_source = (SETTLING_DIAMETER_ACTIVE_IOPSUS_V94
                       if active_iopsus else SETTLING_DIAMETER_STATIC_PROFILE)
    expected_source_state = (parent.state_sha256 if active_iopsus else
                             execution.base_execution.base_execution.
                             fingerprint)
    face = getattr(trial.base_v93_trial.q3_open_trial, "face_trial", None)
    if (face is None or
            getattr(face, "producer_receipt", None) is not
                trial.producer_receipt or
            trial.settling_diameter_receipt is not
                v92_trial.settling_diameter_receipt or
            trial.suspended_capacity_receipt is not
                v92_trial.suspended_capacity_receipt or
            trial.settling_diameter_receipt.source_kind != expected_source or
            trial.settling_diameter_receipt.source_state_sha256 !=
                expected_source_state or
            not np.array_equal(
                trial.settling_diameter_receipt.suspended_diameter_m,
                parent.suspended_diameter_m) or
            trial.suspended_capacity_receipt.source_kind !=
                SUSPENDED_CAPACITY_ACTIVE_V83_V94 or
            trial.suspended_capacity_receipt.source_receipt_sha256 !=
                trial.producer_receipt.receipt_sha256 or
            not np.array_equal(
                trial.candidate_suspended_diameter_m,
                trial.candidate.suspended_diameter_m)):
        raise ActiveSuspendedControlV94Error(
            "v94 lagged DSS or shared v83 receipt differs")
    validate_factcr_producer_receipt_v83(
        trial.producer_receipt, execution.factcr_execution)
    frozen = execution.frozen_uniform_threshold
    expected_background_viscosity = np.full(
        execution.cell_count,
        frozen.initializer_kinematic_viscosity_m2_s)
    if trial.producer_receipt.background_kinematic_viscosity_sha256 != \
            _v83_digest(expected_background_viscosity):
        raise ActiveSuspendedControlV94Error(
            "v94 producer background diffusion viscosity is not the "
            "initializer vicmol")
    if not all(np.array_equal(
            produced,
            np.broadcast_to(expected[:, None], produced.shape))
            for produced, expected in zip(
                (trial.producer_receipt.dstar,
                 trial.producer_receipt.critical_shields,
                 trial.producer_receipt.unfactored_critical_shear_pa,
                 trial.producer_receipt.effective_critical_shear_pa),
                (frozen.dstar, frozen.critical_shields,
                 frozen.unfactored_critical_shear_pa,
                 frozen.effective_critical_shear_pa))):
        raise ActiveSuspendedControlV94Error(
            "v94 producer threshold differs from its frozen initializer")
    validate_suspended_capacity_receipt_v92(
        trial.suspended_capacity_receipt,
        execution.base_execution.base_execution,
        parent.base_state.base_state,
        accepted_q=v92_trial.candidate.q,
        settling_diameter_receipt=trial.settling_diameter_receipt)
    if not np.array_equal(
            trial.suspended_capacity_receipt.suspended_capacity_kg_m_s,
            trial.producer_receipt.lifecycle_capacity_kg_m_s):
        raise ActiveSuspendedControlV94Error(
            "v94 shared v83 suspended capacity differs")

    expected_stage, expected_dss = _candidate_diameter_v94(
        execution, parent, trial.producer_receipt)
    ng, nc = execution.fraction_count, execution.cell_count
    shape = (ng, nc)
    expected_iopsus = np.broadcast_to(
        execution.iopsus_by_fraction[:, None], shape)
    expected_factcr = np.broadcast_to(
        trial.producer_receipt.factcr_application_count, shape)
    counts = trial.owner_counts
    if (not np.array_equal(
            trial.live_taurat,
            trial.producer_receipt.producer_diagnostics[2]) or
            not np.array_equal(trial.transport_stage, expected_stage) or
            not np.array_equal(
                trial.candidate_suspended_diameter_m, expected_dss) or
            not np.array_equal(
                counts.settling_diameter_consumer_count,
                np.ones(shape, dtype=np.int32)) or
            not np.array_equal(
                counts.capacity_producer_count,
                np.ones(shape, dtype=np.int32)) or
            not np.array_equal(
                counts.factcr_application_count, expected_factcr) or
            not np.array_equal(
                counts.scalar_lifecycle_count,
                np.ones(shape, dtype=np.int32)) or
            not np.array_equal(
                counts.additional_scalar_lifecycle_count,
                np.zeros(shape, dtype=np.int32)) or
            not np.array_equal(
                counts.authoritative_exner_count,
                np.ones(shape, dtype=np.int32)) or
            not np.array_equal(
                counts.iopsus_application_count, expected_iopsus) or
            not np.array_equal(
                counts.suscor_correction_application_count,
                np.zeros(shape, dtype=np.int32)) or
            not np.array_equal(
                counts.suspended_vector_construction_owner_count,
                np.full(shape, int(execution.suscor), dtype=np.int32))):
        raise ActiveSuspendedControlV94Error(
            "v94 stage, candidate DSS, or owner counts differ")

    expected_chain = _fingerprint({
        "previous": parent.accepted_iopsus_chain_sha256,
        "base_v93_trial": trial.base_v93_trial.trial_sha256,
        "parent_DSS_receipt": trial.settling_diameter_receipt.receipt_sha256,
        "v83_producer": trial.producer_receipt.receipt_sha256,
        "candidate_DSS_sha256": _array_digest(expected_dss),
        "iopsus_by_fraction": execution.iopsus_by_fraction.tolist(),
    })
    if (trial.candidate.base_state is not trial.base_v93_trial.candidate or
            trial.candidate.accepted_iopsus_chain_sha256 != expected_chain or
            not np.array_equal(
                trial.candidate.suspended_diameter_m, expected_dss)):
        raise ActiveSuspendedControlV94Error(
            "v94 candidate or accepted IopSus chain differs")
    validate_active_suspended_control_state_v94(
        trial.candidate, execution)


def begin_active_suspended_control_cpu_trial_v94(
    execution: ActiveSuspendedControlExecutionV94,
    parent: ActiveSuspendedControlStateV94,
    parent_hles: HlesProducerStateV90,
    handoff: CurrentClosureHandoffV90,
    q3_provider: CPUQ3V83ActiveExnerAdapterV94, *,
    v92_inputs: Mapping[str, object],
    q3_inputs: Mapping[str, object],
    cancelled: Callable[[], bool] = lambda: False,
) -> ActiveSuspendedControlTrialV94:
    if execution.backend != "cpu" or type(q3_provider) is not \
            CPUQ3V83ActiveExnerAdapterV94:
        raise ActiveSuspendedControlV94Error(
            "v94 CPU route requires its exact reviewed CPU provider")
    if not callable(cancelled):
        raise TypeError("cancelled must be callable")
    validate_active_suspended_control_state_v94(
        parent, execution, q3_provider)
    # Reject terminal/malformed work before arming either child owner.
    if parent.accepted_updates >= MAX_ACCEPTED_UPDATES:
        raise ActiveSuspendedControlV94Error(
            "v94 accepted update limit was reached")
    if cancelled():
        raise InterruptedError("v94 CPU cancelled before child work")
    inputs = dict(v92_inputs)
    if ("settling_diameter_receipt" in inputs or
            "suspended_capacity_resolver" in inputs):
        raise ActiveSuspendedControlV94Error(
            "v94 owns the v92 settling/capacity seams")
    inputs.pop("suspended_transport_capacity_kg_m_s", None)
    surface = inputs.get("surface_fraction")
    if surface is not None and not np.array_equal(
            np.asarray(surface), provider_surface_fraction_v114(q3_provider)):
        raise ActiveSuspendedControlV94Error(
            "v94 caller surface fraction differs from the bed owner")
    inputs["surface_fraction"] = provider_surface_fraction_v114(q3_provider)
    if np.any(execution.iopsus_by_fraction == 1):
        settling = make_settling_diameter_receipt_v92(
            execution.base_execution.base_execution,
            parent.base_state.base_state,
            suspended_diameter_m=parent.suspended_diameter_m,
            source_kind=SETTLING_DIAMETER_ACTIVE_IOPSUS_V94,
            source_state_sha256=parent.state_sha256)
    else:
        settling = make_settling_diameter_receipt_v92(
            execution.base_execution.base_execution,
            parent.base_state.base_state,
            suspended_diameter_m=parent.suspended_diameter_m)
    provider_checkpoint = q3_provider.checkpoint()
    base_trial = None
    try:
        q3_provider.arm_capacity(
            control_parent=parent, q3_inputs=q3_inputs,
            cancelled=cancelled)
        base_trial = begin_suspended_exner_cpu_trial_v93(
            execution.base_execution, parent.base_state,
            parent_hles, handoff, q3_provider,
            v92_inputs={
                **inputs,
                "settling_diameter_receipt": settling,
                "suspended_capacity_resolver":
                    q3_provider.resolve_armed_capacity,
            },
            q3_inputs={}, cancelled=cancelled)
        face = base_trial.q3_open_trial.face_trial
        if not isinstance(face, CPUQ3V83ActiveFaceTrialV94):
            raise ActiveSuspendedControlV94Error(
                "v94 CPU provider returned a different face trial")
        chain = _fingerprint({
            "previous": parent.accepted_iopsus_chain_sha256,
            "base_v93_trial": base_trial.trial_sha256,
            "parent_DSS_receipt": settling.receipt_sha256,
            "v83_producer": face.producer_receipt.receipt_sha256,
            "candidate_DSS_sha256": _array_digest(
                face.candidate_suspended_diameter_m),
            "iopsus_by_fraction": execution.iopsus_by_fraction.tolist(),
        })
        candidate = _wrap_state(
            execution, base_trial.candidate,
            suspended_diameter_m=face.candidate_suspended_diameter_m,
            accepted_iopsus_chain_sha256=chain)
        ng, nc = execution.fraction_count, execution.cell_count
        ones = _readonly(np.ones((ng, nc), np.int32), (ng, nc),
                         "owner ones", integer=True)
        zeros = _readonly(np.zeros((ng, nc), np.int32), (ng, nc),
                          "owner zeros", integer=True)
        counts = ActiveSuspendedControlOwnerCountsV94(
            ones, ones, face.factcr_application_count, ones, zeros, ones,
            face.iopsus_application_count,
            face.suscor_correction_application_count,
            face.suspended_vector_construction_owner_count)
        placeholder = ActiveSuspendedControlTrialV94(
            execution.fingerprint, parent.state_sha256, base_trial,
            settling, base_trial.base_v92_trial.suspended_capacity_receipt,
            face.producer_receipt, face.live_taurat,
            face.transport_stage, face.candidate_suspended_diameter_m,
            counts, candidate, "cpu_oracle", "")
        trial = ActiveSuspendedControlTrialV94(
            **{**placeholder.__dict__,
               "trial_sha256": _trial_token(placeholder)})
        _validate_trial(execution, parent, parent_hles, handoff, trial)
        return trial
    except Exception as original_error:
        try:
            if (base_trial is not None and
                    q3_provider._closed is
                        base_trial.q3_prepared_commit):
                q3_provider.abort_closed_commit(
                    base_trial.q3_prepared_commit)
            elif (base_trial is not None and
                  q3_provider._open is base_trial.q3_open_trial):
                q3_provider.rollback(base_trial.q3_open_trial)
            if (q3_provider._open is not None or
                    q3_provider._prepared is not None or
                    q3_provider._closed is not None):
                raise RuntimeError(
                    "v94 CPU begin cleanup left an open Q3 transaction")
            q3_provider.restore_checkpoint(provider_checkpoint)
        except Exception as cleanup_error:
            raise RuntimeError(
                "v94 CPU begin failed and provider restore also failed") \
                from cleanup_error
        raise original_error


def commit_active_suspended_control_trial_v94(
    execution: ActiveSuspendedControlExecutionV94,
    parent: ActiveSuspendedControlStateV94,
    parent_hles: HlesProducerStateV90,
    handoff: CurrentClosureHandoffV90,
    trial: ActiveSuspendedControlTrialV94,
) -> ActiveSuspendedControlStateV94:
    _validate_trial(execution, parent, parent_hles, handoff, trial)
    try:
        commit_suspended_exner_trial_v93(
            execution.base_execution, parent.base_state, parent_hles,
            handoff, trial.base_v93_trial)
    except Exception as original_error:
        provider = trial.base_v93_trial.q3_provider
        try:
            if (getattr(provider, "_open", None) is not None or
                    getattr(provider, "_prepared", None) is not None or
                    getattr(provider, "_closed", None) is not None):
                raise RuntimeError(
                    "v94 failed commit left an open Q3 transaction")
            provider.restore_checkpoint(
                trial.base_v93_trial.q3_parent_checkpoint)
        except Exception as cleanup_error:
            raise RuntimeError(
                "v94 commit failed and provider restore also failed") \
                from cleanup_error
        raise original_error
    # The v93 commit publishes the exact candidate already authenticated by
    # this trial; no fallible operation follows provider publication.
    return trial.candidate


def rollback_active_suspended_control_trial_v94(
    execution: ActiveSuspendedControlExecutionV94,
    parent: ActiveSuspendedControlStateV94,
    parent_hles: HlesProducerStateV90,
    handoff: CurrentClosureHandoffV90,
    trial: ActiveSuspendedControlTrialV94,
) -> ActiveSuspendedControlStateV94:
    _validate_trial(execution, parent, parent_hles, handoff, trial)
    base = rollback_suspended_exner_trial_v93(
        execution.base_execution, parent.base_state, parent_hles, handoff,
        trial.base_v93_trial)
    if base is not parent.base_state:
        raise ActiveSuspendedControlV94Error(
            "v94 rollback changed the parent v93 state")
    return parent


def cancel_active_suspended_control_trial_v94(*args, **kwargs):
    return rollback_active_suspended_control_trial_v94(*args, **kwargs)


def checkpoint_active_suspended_control_state_v94(
    execution: ActiveSuspendedControlExecutionV94,
    state: ActiveSuspendedControlStateV94,
    q3_provider: object,
) -> dict[str, object]:
    validate_active_suspended_control_state_v94(
        state, execution, q3_provider)
    body = {
        "schema": SCHEMA,
        "scope_id": CHECKPOINT_SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "execution_fingerprint": execution.fingerprint,
        "iopsus_by_fraction": execution.iopsus_by_fraction.tolist(),
        "base_v93_checkpoint": suspended_exner_checkpoint_v93(
            execution.base_execution, state.base_state, q3_provider),
        "suspended_diameter_m": state.suspended_diameter_m.tolist(),
        "frozen_threshold_receipt_sha256":
            state.frozen_threshold_receipt_sha256,
        "accepted_iopsus_chain_sha256":
            state.accepted_iopsus_chain_sha256,
        "state_sha256": state.state_sha256,
    }
    return {**body, "fingerprint": _fingerprint(body)}


def load_active_suspended_control_checkpoint_v94(
    execution: ActiveSuspendedControlExecutionV94,
    document: Mapping[str, object],
    q3_provider: object,
) -> ActiveSuspendedControlStateV94:
    expected = {
        "schema", "scope_id", "semantic_id", "execution_fingerprint",
        "iopsus_by_fraction", "base_v93_checkpoint",
        "suspended_diameter_m", "frozen_threshold_receipt_sha256",
        "accepted_iopsus_chain_sha256",
        "state_sha256", "fingerprint",
    }
    if not isinstance(document, Mapping) or set(document) != expected:
        raise ActiveSuspendedControlV94Error(
            "v94 checkpoint inventory differs")
    body = {key: value for key, value in document.items()
            if key != "fingerprint"}
    if (document["fingerprint"] != _fingerprint(body) or
            document["schema"] != SCHEMA or
            document["scope_id"] != CHECKPOINT_SCOPE_ID or
            document["semantic_id"] != SEMANTIC_ID or
            document["execution_fingerprint"] != execution.fingerprint or
            document["iopsus_by_fraction"] !=
                execution.iopsus_by_fraction.tolist() or
            document["frozen_threshold_receipt_sha256"] !=
                execution.frozen_uniform_threshold.receipt_sha256):
        raise ActiveSuspendedControlV94Error(
            "v94 checkpoint execution or IopSus identity differs")
    provider_checkpoint = q3_provider.checkpoint()
    try:
        base = load_suspended_exner_checkpoint_v93(
            execution.base_execution, document["base_v93_checkpoint"],
            q3_provider)
        state = _wrap_state(
            execution, base,
            suspended_diameter_m=document["suspended_diameter_m"],
            accepted_iopsus_chain_sha256=
                document["accepted_iopsus_chain_sha256"])
        if state.state_sha256 != document["state_sha256"]:
            raise ActiveSuspendedControlV94Error(
                "v94 checkpoint state digest differs")
        validate_active_suspended_control_state_v94(
            state, execution, q3_provider)
        return state
    except Exception as original_error:
        try:
            q3_provider.restore_checkpoint(provider_checkpoint)
        except Exception as cleanup_error:
            raise RuntimeError(
                "v94 checkpoint load failed and provider restore also "
                "failed") from cleanup_error
        raise original_error


def method_identity_v94() -> dict[str, object]:
    return {
        "scope_id": SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "integrator": INTEGRATOR,
        "formula_code": FORMULA_CODE,
        "settling_form": SETTLING_FORM,
        "iopsus": "per_fraction_exact_0_or_1_all_eight_profiles",
        "lag_order": (
            "parent_DSS_to_current_settling_then_current_taurat_to_"
            "next_accepted_DSS"),
        "transport_stage": "max(0,taurat-1)",
        "dss_branches": {"T<=1": "0.64*D50",
                         "1<T<25": "D50*(1+0.015*(T-25))",
                         "T>=25": "D50"},
        "factcr_same_step_settling": False,
        "critical_threshold": (
            "uniform_D50_initializer_rhow_vicmol_frozen_all_updates"),
        "current_fluid_roles": (
            "cell_rhowat_to_shear_and_current_vicmol_to_settling_diffusion"),
        "suscor_kmax1_correction_application_count": 0,
        "suscor_vector_owner": "one_when_true_zero_when_false",
        "suscor_false_neglect_true":
            "source_legal_zero_vector_fail_closed_queued_v94b",
        "one_scalar_owner": "v92",
        "one_bed_owner": "v93",
        "pure_v83_producer": True,
        "live_v73_taurat": True,
        "maximum_accepted_updates": MAX_ACCEPTED_UPDATES,
        "morphological_factor": 1.0,
        "fixed_grid": True,
        "fully_wet": True,
        "downhill": "exact_zero_only",
        "automatic_cpu_fallback": False,
        "long_run_claimed": False,
        "product_admission": False,
        "delft3d_source_modified": False,
    }


__all__ = [
    "CHECKPOINT_SCOPE_ID", "CPU_PROVIDER_KIND", "GPU_PROVIDER_KIND",
    "FORMULA_CODE", "FRACTION_COUNT", "INTEGRATOR", "KMAX",
    "MAX_ACCEPTED_UPDATES", "SCHEMA", "SCOPE_ID", "SEMANTIC_ID",
    "SETTLING_FORM", "ActiveSuspendedControlExecutionV94",
    "ActiveSuspendedControlOwnerCountsV94",
    "ActiveSuspendedControlStateV94", "ActiveSuspendedControlTrialV94",
    "ActiveSuspendedControlV94Error", "CPUQ3V83ActiveExnerAdapterV94",
    "CPUQ3V83ActiveFaceTrialV94",
    "FrozenUniformThresholdReceiptV94",
    "begin_active_suspended_control_cpu_trial_v94",
    "cancel_active_suspended_control_trial_v94",
    "checkpoint_active_suspended_control_state_v94",
    "commit_active_suspended_control_trial_v94", "execution_document_v94",
    "load_active_suspended_control_checkpoint_v94", "method_identity_v94",
    "resolve_active_suspended_control_execution_v94",
    "resolve_initial_active_suspended_control_state_v94",
    "rollback_active_suspended_control_trial_v94",
    "validate_frozen_threshold_receipt_v94",
    "validate_active_suspended_control_state_v94",
]
