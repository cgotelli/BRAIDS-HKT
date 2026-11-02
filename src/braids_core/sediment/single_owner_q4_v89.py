"""Q4/v89 single-owner ABI and bounded scalar transaction.

This module is an ownership refactor, not a new sediment formulation.  It
extracts immutable provider receipts from the v84/v86/v87/v88 component
contracts and gives the accepted v78 state exactly one numerical owner.  The
CPU path calls the already accepted boundary, FCT, spatial-diffusion,
SettleFrm=2 and Galappatti operators once, then publishes exactly one v78
trial.  The CUDA peer uses the accepted resident v78 owner once.

Only the identity-compatible slice is admitted here: all fractions use the
``combine`` suspended route, Q3 bed/total-load vectors are exact zero,
``IopSus=0``, ``NeglectEntrainment=false``, static FLOW fluid properties and
fully wet/fixed-grid v78 semantics.  Lagged ``dicuv`` is consumed; ``vicuv``,
the newly produced closure candidate and dynamic T/S are explicitly staged
for v90--v92.  No staged artifact is presented as numerically consumed.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Mapping

import numpy as np

from .flow2dh_boundary_diffusion import (
    Flow2DHBoundaryMemory,
    resolve_flow2dh_open_boundary,
)
from .flow2dh_exchange import advance_flow2dh_equilibrium_exchange
from .flow2dh_settling import settlefrm2_noncohesive
from .multifraction_suspended_lifecycle_v78 import (
    ROLE_CPU_ORACLE,
    ROLE_CUDA,
    MultifractionSuspendedExecutionV78,
    MultifractionSuspendedStateV78,
    MultifractionSuspendedTrialV78,
    build_multifraction_trial_v78,
    commit_multifraction_trial_v78,
    execution_document_v78,
    load_multifraction_checkpoint_v78,
    multifraction_checkpoint_v78,
    resolve_multifraction_suspended_state_v78,
    rollback_multifraction_trial_v78,
    _validate_trial as _validate_multifraction_trial_v78,
    validate_implicit_fwave_midpoint_step_receipt_v78,
    validate_multifraction_suspended_state_v78,
)
from .route_lifecycle_composition_v81 import ROUTE_COMBINE
from .suspended_lifecycle_live_v76 import (
    _cpu_fct,
    resolve_suspended_lifecycle_state_v76,
)
from .suspended_support_closures_v82 import (
    _cpu_spatial_diffusion,
    resolve_suspended_support_execution_v82,
)


SCHEMA = 1
SCOPE_ID = "braids-q4-single-owner-abi-v89-v1"
CHECKPOINT_SCOPE_ID = SCOPE_ID + ":checkpoint"
SEMANTIC_ID = hashlib.sha256(SCOPE_ID.encode("ascii")).hexdigest()
INTEGRATOR = "implicit_fwave_midpoint"
MAX_ACCEPTED_UPDATES = 1
BACKEND_CPU = "cpu"
BACKEND_GPU = "implicit_fwave_midpoint_gpu"
PROVIDER_Q3 = "q3_capacity_route_v84_v81_abi"
PROVIDER_CONTROLS = "suspended_controls_v86_selector"
PROVIDER_CLOSURE = "horizontal_turbulence_v87_receipt"
PROVIDER_FLUID = "dynamic_fluid_v88_static_identity"
_SHA256 = re.compile(r"[0-9a-f]{64}")


class SingleOwnerQ4V89Error(ValueError):
    """A v89 execution, receipt, transaction or checkpoint is invalid."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, OverflowError) as error:
        raise SingleOwnerQ4V89Error(
            "v89 value is not finite canonical JSON") from error


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(_canonical(list(array.shape)))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise SingleOwnerQ4V89Error(f"v89 {name} must be a SHA-256 digest")
    return value


def _frozen(value: object, shape: tuple[int, ...], name: str, *,
            nonnegative: bool = False, integer: bool = False) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != shape or raw.dtype.kind not in "fiub":
        raise SingleOwnerQ4V89Error(
            f"v89 {name} must be numeric with shape {shape}")
    dtype = np.int32 if integer else np.float64
    result = np.ascontiguousarray(raw, dtype=dtype)
    if not np.isfinite(result).all():
        raise SingleOwnerQ4V89Error(f"v89 {name} must be finite")
    if nonnegative and np.any(result < 0):
        raise SingleOwnerQ4V89Error(f"v89 {name} must be nonnegative")
    result.setflags(write=False)
    return result


def _input_array(value: object, shape: tuple[int, ...], name: str, *,
                 nonnegative: bool = False, boolean: bool = False) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != shape or raw.dtype.kind not in "fiub":
        raise SingleOwnerQ4V89Error(
            f"v89 {name} must be numeric with shape {shape}")
    dtype = np.bool_ if boolean else np.float64
    result = np.ascontiguousarray(raw, dtype=dtype)
    if not np.isfinite(result).all():
        raise SingleOwnerQ4V89Error(f"v89 {name} must be finite")
    if nonnegative and np.any(result < 0):
        raise SingleOwnerQ4V89Error(f"v89 {name} must be nonnegative")
    return result


@dataclass(frozen=True)
class SingleOwnerExecutionV89:
    base_execution: MultifractionSuspendedExecutionV78
    backend: str
    semantic_id: str
    fingerprint: str

    @property
    def fraction_count(self) -> int:
        return self.base_execution.fraction_count

    @property
    def cell_count(self) -> int:
        return self.base_execution.cell_count

    @property
    def face_count(self) -> int:
        return self.base_execution.face_count

    @property
    def boundary_count(self) -> int:
        return self.base_execution.boundary_count


def _execution_payload(value: SingleOwnerExecutionV89) -> dict:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "semantic_id": value.semantic_id,
        "base_v78_execution": execution_document_v78(value.base_execution),
        "backend": value.backend,
        "integrator": INTEGRATOR,
        "canonical_state_owner": "one_v78_state",
        "per_trial_dicuv_receipt": True,
        "accepted_update_limit": MAX_ACCEPTED_UPDATES,
        "automatic_cpu_fallback": False,
        "active_consumers": [
            "q3_suspended_capacity", "combine_route", "lagged_dicuv",
            "default_exchange_bed_selector", "static_fluid_identity_guard",
        ],
        "staged_not_consumed": [
            "q3_bed_or_total_load", "lagged_vicuv_momentum",
            "new_closure_candidate", "dynamic_temperature_salinity",
        ],
        "product_admission": False,
        "provider_receipt_provenance":
            "caller_sealed_typed_ABI_not_real_v84_or_v87_invocation",
        "owner_count_semantics":
            "authenticated_expected_stage_counts_not_kernel_launch_counters",
    }


def resolve_single_owner_execution_v89(
    base_execution: MultifractionSuspendedExecutionV78, *, backend: str,
) -> SingleOwnerExecutionV89:
    execution_document_v78(base_execution)
    if backend not in (BACKEND_CPU, BACKEND_GPU):
        raise SingleOwnerQ4V89Error("v89 backend must be explicit")
    expected = ROLE_CPU_ORACLE if backend == BACKEND_CPU else ROLE_CUDA
    if base_execution.role != expected:
        raise SingleOwnerQ4V89Error("v89 backend and v78 role differ")
    if base_execution.base_execution.profile.horizontal_diffusivity_m2_s != 0.0:
        raise SingleOwnerQ4V89Error(
            "v89 canonical v78 execution must have zero built-in diffusion")
    placeholder = SingleOwnerExecutionV89(
        base_execution, backend, SEMANTIC_ID, "")
    return SingleOwnerExecutionV89(
        base_execution, backend, SEMANTIC_ID,
        _fingerprint(_execution_payload(placeholder)))


def execution_document_v89(value: SingleOwnerExecutionV89) -> dict:
    if not isinstance(value, SingleOwnerExecutionV89):
        raise SingleOwnerQ4V89Error("v89 execution object differs")
    payload = _execution_payload(value)
    if value.fingerprint != _fingerprint(payload):
        raise SingleOwnerQ4V89Error("v89 execution fingerprint differs")
    return {**payload, "fingerprint": value.fingerprint}


@dataclass(frozen=True)
class SingleOwnerStateV89:
    execution_fingerprint: str
    base_state: MultifractionSuspendedStateV78
    accepted_receipt_chain_sha256: str
    state_sha256: str

    @property
    def accepted_updates(self) -> int:
        return self.base_state.accepted_updates

    @property
    def time_s(self) -> float:
        return self.base_state.time_s

    @property
    def suspended_mass_kg_m2(self) -> np.ndarray:
        return self.base_state.suspended_mass_kg_m2

    @property
    def bed_mass_kg_m2(self) -> np.ndarray:
        return self.base_state.bed_mass_kg_m2


def _state_token(value: SingleOwnerStateV89) -> str:
    return _fingerprint({
        "execution_fingerprint": value.execution_fingerprint,
        "base_state_sha256": value.base_state.state_sha256,
        "accepted_receipt_chain_sha256":
            value.accepted_receipt_chain_sha256,
    })


def _wrap_state(execution: SingleOwnerExecutionV89,
                base_state: MultifractionSuspendedStateV78,
                chain: str) -> SingleOwnerStateV89:
    execution_document_v89(execution)
    validate_multifraction_suspended_state_v78(
        base_state, execution.base_execution)
    _sha(chain, "accepted receipt chain")
    if base_state.accepted_updates > MAX_ACCEPTED_UPDATES:
        raise SingleOwnerQ4V89Error("v89 state exceeds its one-update bound")
    placeholder = SingleOwnerStateV89(
        execution.fingerprint, base_state, chain, "")
    return SingleOwnerStateV89(
        execution.fingerprint, base_state, chain, _state_token(placeholder))


def resolve_single_owner_state_v89(
    execution: SingleOwnerExecutionV89, **state_fields,
) -> SingleOwnerStateV89:
    base = resolve_multifraction_suspended_state_v78(
        execution.base_execution, **state_fields)
    chain = _fingerprint({
        "scope_id": SCOPE_ID,
        "execution_fingerprint": execution.fingerprint,
        "initial_base_state_sha256": base.state_sha256,
    })
    return _wrap_state(execution, base, chain)


def validate_single_owner_state_v89(
    state: SingleOwnerStateV89, execution: SingleOwnerExecutionV89,
) -> None:
    if (not isinstance(state, SingleOwnerStateV89) or
            state.execution_fingerprint != execution.fingerprint or
            state.state_sha256 != _state_token(state)):
        raise SingleOwnerQ4V89Error(
            "v89 state is stale, mutated or execution-mismatched")
    _wrap_state(execution, state.base_state,
                state.accepted_receipt_chain_sha256)


def _receipt_common(execution: SingleOwnerExecutionV89,
                    parent: SingleOwnerStateV89,
                    step: Mapping[str, object], provider: str) -> dict:
    validate_single_owner_state_v89(parent, execution)
    if not isinstance(step, Mapping):
        raise SingleOwnerQ4V89Error("v89 hydraulic receipt is absent")
    step_sha = _sha(step.get("fingerprint"), "hydraulic receipt fingerprint")
    return {
        "execution_fingerprint": execution.fingerprint,
        "base_v78_execution_fingerprint": execution.base_execution.fingerprint,
        "parent_base_state_sha256": parent.base_state.state_sha256,
        "accepted_step_receipt_fingerprint": step_sha,
        "backend": execution.backend,
        "provider_id": provider,
        "automatic_cpu_fallback": False,
    }


@dataclass(frozen=True)
class Q3CapacityRouteReceiptV89:
    execution_fingerprint: str
    base_v78_execution_fingerprint: str
    parent_base_state_sha256: str
    accepted_step_receipt_fingerprint: str
    backend: str
    provider_id: str
    provider_execution_sha256: str
    provider_output_sha256: str
    routes: np.ndarray
    suspended_capacity_kg_m_s: np.ndarray
    bed_capacity_x_kg_m_s: np.ndarray
    bed_capacity_y_kg_m_s: np.ndarray
    receipt_sha256: str


def _q3_token(value: Q3CapacityRouteReceiptV89) -> str:
    return _fingerprint({
        "execution_fingerprint": value.execution_fingerprint,
        "base_v78_execution_fingerprint":
            value.base_v78_execution_fingerprint,
        "parent_base_state_sha256": value.parent_base_state_sha256,
        "accepted_step_receipt_fingerprint":
            value.accepted_step_receipt_fingerprint,
        "backend": value.backend,
        "provider_id": value.provider_id,
        "provider_execution_sha256": value.provider_execution_sha256,
        "provider_output_sha256": value.provider_output_sha256,
        "arrays_sha256": [_array_digest(array) for array in (
            value.routes, value.suspended_capacity_kg_m_s,
            value.bed_capacity_x_kg_m_s, value.bed_capacity_y_kg_m_s)],
        "bed_or_total_load_disposition":
            "exact_zero_staged_until_authoritative_exner_owner",
    })


def make_q3_capacity_route_receipt_v89(
    execution: SingleOwnerExecutionV89, parent: SingleOwnerStateV89, *,
    accepted_step_receipt: Mapping[str, object],
    suspended_capacity_kg_m_s: object,
    provider_execution_sha256: str,
    provider_output_sha256: str | None = None,
    routes: object | None = None,
    bed_capacity_x_kg_m_s: object | None = None,
    bed_capacity_y_kg_m_s: object | None = None,
) -> Q3CapacityRouteReceiptV89:
    common = _receipt_common(
        execution, parent, accepted_step_receipt, PROVIDER_Q3)
    ng, nc = execution.fraction_count, execution.cell_count
    route = _frozen(
        np.full(ng, ROUTE_COMBINE, dtype=np.int32) if routes is None else routes,
        (ng,), "Q3 routes", integer=True)
    if np.any(route != ROUTE_COMBINE):
        raise SingleOwnerQ4V89Error(
            "v89 only admits the all-combine Q3 route slice")
    capacity = _frozen(
        suspended_capacity_kg_m_s, (ng, nc), "Q3 suspended capacity",
        nonnegative=True)
    bx = _frozen(
        np.zeros((ng, nc)) if bed_capacity_x_kg_m_s is None else
        bed_capacity_x_kg_m_s, (ng, nc), "Q3 bed capacity x")
    by = _frozen(
        np.zeros((ng, nc)) if bed_capacity_y_kg_m_s is None else
        bed_capacity_y_kg_m_s, (ng, nc), "Q3 bed capacity y")
    if np.any(bx != 0.0) or np.any(by != 0.0):
        raise SingleOwnerQ4V89Error(
            "v89 Q3 bed/total-load output has no authoritative Exner consumer")
    output_sha = _fingerprint({
        "routes": _array_digest(route),
        "suspended_capacity": _array_digest(capacity),
        "bed_capacity_x": _array_digest(bx),
        "bed_capacity_y": _array_digest(by),
    })
    if provider_output_sha256 is not None and \
            _sha(provider_output_sha256, "Q3 provider output") != output_sha:
        raise SingleOwnerQ4V89Error(
            "v89 Q3 provider output digest differs from the receipt arrays")
    placeholder = Q3CapacityRouteReceiptV89(
        common["execution_fingerprint"],
        common["base_v78_execution_fingerprint"],
        common["parent_base_state_sha256"],
        common["accepted_step_receipt_fingerprint"], common["backend"],
        common["provider_id"], _sha(provider_execution_sha256,
                                     "Q3 provider execution"),
        output_sha,
        route, capacity, bx, by, "")
    return Q3CapacityRouteReceiptV89(
        **{**placeholder.__dict__, "receipt_sha256": _q3_token(placeholder)})


@dataclass(frozen=True)
class SuspendedControlReceiptV89:
    execution_fingerprint: str
    base_v78_execution_fingerprint: str
    parent_base_state_sha256: str
    accepted_step_receipt_fingerprint: str
    backend: str
    provider_id: str
    iopsus: int
    neglect_entrainment: bool
    suscor: bool
    receipt_sha256: str


def _control_token(value: SuspendedControlReceiptV89) -> str:
    return _fingerprint({
        **{key: getattr(value, key) for key in (
            "execution_fingerprint", "base_v78_execution_fingerprint",
            "parent_base_state_sha256", "accepted_step_receipt_fingerprint",
            "backend", "provider_id", "iopsus", "neglect_entrainment",
            "suscor")},
        "selected_bed_owner": "v78_exchange",
        "suscor_2dh_application_count": 0,
    })


def make_suspended_control_receipt_v89(
    execution: SingleOwnerExecutionV89, parent: SingleOwnerStateV89, *,
    accepted_step_receipt: Mapping[str, object], iopsus: int = 0,
    neglect_entrainment: bool = False, suscor: bool = True,
) -> SuspendedControlReceiptV89:
    common = _receipt_common(
        execution, parent, accepted_step_receipt, PROVIDER_CONTROLS)
    if type(iopsus) is not int or iopsus != 0:
        raise SingleOwnerQ4V89Error(
            "v89 active IopSus is staged for v94 and fails closed")
    if type(neglect_entrainment) is not bool or neglect_entrainment:
        raise SingleOwnerQ4V89Error(
            "v89 flux-divergence bed ownership is staged for v93")
    if type(suscor) is not bool:
        raise SingleOwnerQ4V89Error("v89 SusCor must be boolean")
    placeholder = SuspendedControlReceiptV89(
        common["execution_fingerprint"],
        common["base_v78_execution_fingerprint"],
        common["parent_base_state_sha256"],
        common["accepted_step_receipt_fingerprint"], common["backend"],
        common["provider_id"], iopsus, neglect_entrainment, suscor, "")
    return SuspendedControlReceiptV89(
        **{**placeholder.__dict__,
           "receipt_sha256": _control_token(placeholder)})


@dataclass(frozen=True)
class ClosureReceiptV89:
    execution_fingerprint: str
    base_v78_execution_fingerprint: str
    parent_base_state_sha256: str
    accepted_step_receipt_fingerprint: str
    backend: str
    provider_id: str
    background_vicuv_m2_s: np.ndarray
    background_dicuv_m2_s: np.ndarray
    lagged_dynamic_vicuv_m2_s: np.ndarray
    lagged_dynamic_dicuv_m2_s: np.ndarray
    candidate_dynamic_vicuv_m2_s: np.ndarray
    candidate_dynamic_dicuv_m2_s: np.ndarray
    receipt_sha256: str

    @property
    def used_total_dicuv_m2_s(self) -> np.ndarray:
        result = np.ascontiguousarray(
            self.background_dicuv_m2_s+self.lagged_dynamic_dicuv_m2_s)
        result.setflags(write=False)
        return result

    @property
    def used_total_vicuv_m2_s(self) -> np.ndarray:
        result = np.ascontiguousarray(
            self.background_vicuv_m2_s+self.lagged_dynamic_vicuv_m2_s)
        result.setflags(write=False)
        return result


def _closure_token(value: ClosureReceiptV89) -> str:
    return _fingerprint({
        **{key: getattr(value, key) for key in (
            "execution_fingerprint", "base_v78_execution_fingerprint",
            "parent_base_state_sha256", "accepted_step_receipt_fingerprint",
            "backend", "provider_id")},
        "arrays_sha256": [_array_digest(array) for array in (
            value.background_vicuv_m2_s, value.background_dicuv_m2_s,
            value.lagged_dynamic_vicuv_m2_s,
            value.lagged_dynamic_dicuv_m2_s,
            value.candidate_dynamic_vicuv_m2_s,
            value.candidate_dynamic_dicuv_m2_s)],
        "dicuv_disposition": "lagged_total_consumed_once",
        "vicuv_disposition": "staged_no_momentum_consumer",
        "candidate_disposition": "staged_not_committed_until_v90",
    })


def make_closure_receipt_v89(
    execution: SingleOwnerExecutionV89, parent: SingleOwnerStateV89, *,
    accepted_step_receipt: Mapping[str, object],
    background_vicuv_m2_s: object, background_dicuv_m2_s: object,
    lagged_dynamic_vicuv_m2_s: object | None = None,
    lagged_dynamic_dicuv_m2_s: object | None = None,
    candidate_dynamic_vicuv_m2_s: object | None = None,
    candidate_dynamic_dicuv_m2_s: object | None = None,
    activate_candidate: bool = False,
) -> ClosureReceiptV89:
    common = _receipt_common(
        execution, parent, accepted_step_receipt, PROVIDER_CLOSURE)
    if type(activate_candidate) is not bool or activate_candidate:
        raise SingleOwnerQ4V89Error(
            "v89 closure candidate is staged and cannot be activated")
    nc = execution.cell_count
    zero = np.zeros(nc)
    arrays = tuple(_frozen(value, (nc,), name, nonnegative=True) for value, name in (
        (background_vicuv_m2_s, "background vicuv"),
        (background_dicuv_m2_s, "background dicuv"),
        (zero if lagged_dynamic_vicuv_m2_s is None else
         lagged_dynamic_vicuv_m2_s, "lagged dynamic vicuv"),
        (zero if lagged_dynamic_dicuv_m2_s is None else
         lagged_dynamic_dicuv_m2_s, "lagged dynamic dicuv"),
        (zero if candidate_dynamic_vicuv_m2_s is None else
         candidate_dynamic_vicuv_m2_s, "candidate dynamic vicuv"),
        (zero if candidate_dynamic_dicuv_m2_s is None else
         candidate_dynamic_dicuv_m2_s, "candidate dynamic dicuv"),
    ))
    placeholder = ClosureReceiptV89(
        common["execution_fingerprint"],
        common["base_v78_execution_fingerprint"],
        common["parent_base_state_sha256"],
        common["accepted_step_receipt_fingerprint"], common["backend"],
        common["provider_id"], *arrays, "")
    return ClosureReceiptV89(
        **{**placeholder.__dict__,
           "receipt_sha256": _closure_token(placeholder)})


@dataclass(frozen=True)
class FluidPropertyReceiptV89:
    execution_fingerprint: str
    base_v78_execution_fingerprint: str
    parent_base_state_sha256: str
    accepted_step_receipt_fingerprint: str
    backend: str
    provider_id: str
    temperature_cell_c: np.ndarray
    salinity_cell_ppt: np.ndarray
    water_density_cell_kg_m3: np.ndarray
    molecular_viscosity_cell_m2_s: np.ndarray
    density_eos: str
    receipt_sha256: str


def _fluid_token(value: FluidPropertyReceiptV89) -> str:
    return _fingerprint({
        **{key: getattr(value, key) for key in (
            "execution_fingerprint", "base_v78_execution_fingerprint",
            "parent_base_state_sha256", "accepted_step_receipt_fingerprint",
            "backend", "provider_id", "density_eos")},
        "arrays_sha256": [_array_digest(array) for array in (
            value.temperature_cell_c, value.salinity_cell_ppt,
            value.water_density_cell_kg_m3,
            value.molecular_viscosity_cell_m2_s)],
        "numerical_disposition":
            "static_profile_identity_guard_dynamic_transport_staged_v92",
    })


def make_fluid_property_receipt_v89(
    execution: SingleOwnerExecutionV89, parent: SingleOwnerStateV89, *,
    accepted_step_receipt: Mapping[str, object],
    temperature_cell_c: object | None = None,
    salinity_cell_ppt: object | None = None,
    water_density_cell_kg_m3: object | None = None,
    molecular_viscosity_cell_m2_s: object | None = None,
    density_eos: str = "constant_rhow_no_constituents",
) -> FluidPropertyReceiptV89:
    common = _receipt_common(
        execution, parent, accepted_step_receipt, PROVIDER_FLUID)
    profile = execution.base_execution.base_execution.profile
    nc = execution.cell_count
    temperature = np.full(nc, profile.temperature_value_c)
    salinity = np.zeros(nc)
    density = np.full(nc, profile.water_density_kg_m3)
    viscosity = 4.0e-5/(20.0+temperature)
    supplied = (
        temperature if temperature_cell_c is None else temperature_cell_c,
        salinity if salinity_cell_ppt is None else salinity_cell_ppt,
        density if water_density_cell_kg_m3 is None else
        water_density_cell_kg_m3,
        viscosity if molecular_viscosity_cell_m2_s is None else
        molecular_viscosity_cell_m2_s,
    )
    arrays = tuple(_frozen(value, (nc,), name, nonnegative=True)
                   for value, name in zip(supplied, (
                       "temperature", "salinity", "water density",
                       "molecular viscosity")))
    expected = (temperature, salinity, density, viscosity)
    if any(not np.array_equal(actual, wanted)
           for actual, wanted in zip(arrays, expected)):
        raise SingleOwnerQ4V89Error(
            "v89 dynamic fluid properties are staged for v92")
    if density_eos != "constant_rhow_no_constituents":
        raise SingleOwnerQ4V89Error(
            "v89 active EOS/constituent transport is staged for v92")
    placeholder = FluidPropertyReceiptV89(
        common["execution_fingerprint"],
        common["base_v78_execution_fingerprint"],
        common["parent_base_state_sha256"],
        common["accepted_step_receipt_fingerprint"], common["backend"],
        common["provider_id"], *arrays, density_eos, "")
    return FluidPropertyReceiptV89(
        **{**placeholder.__dict__,
           "receipt_sha256": _fluid_token(placeholder)})


@dataclass(frozen=True)
class ProviderBundleV89:
    q3: Q3CapacityRouteReceiptV89
    controls: SuspendedControlReceiptV89
    closure: ClosureReceiptV89
    fluid: FluidPropertyReceiptV89
    bundle_sha256: str


def _bundle_token(value: ProviderBundleV89) -> str:
    return _fingerprint({
        "q3": value.q3.receipt_sha256,
        "controls": value.controls.receipt_sha256,
        "closure": value.closure.receipt_sha256,
        "fluid": value.fluid.receipt_sha256,
    })


def make_provider_bundle_v89(
    q3: Q3CapacityRouteReceiptV89,
    controls: SuspendedControlReceiptV89,
    closure: ClosureReceiptV89,
    fluid: FluidPropertyReceiptV89,
) -> ProviderBundleV89:
    placeholder = ProviderBundleV89(q3, controls, closure, fluid, "")
    return ProviderBundleV89(
        q3, controls, closure, fluid, _bundle_token(placeholder))


def validate_provider_bundle_v89(
    bundle: ProviderBundleV89, execution: SingleOwnerExecutionV89,
    parent: SingleOwnerStateV89, step: Mapping[str, object],
) -> None:
    if not isinstance(bundle, ProviderBundleV89) or \
            bundle.bundle_sha256 != _bundle_token(bundle):
        raise SingleOwnerQ4V89Error("v89 provider bundle is stale or mutated")
    step_sha = _sha(step.get("fingerprint") if isinstance(step, Mapping)
                    else None, "hydraulic receipt fingerprint")
    common = (execution.fingerprint, execution.base_execution.fingerprint,
              parent.base_state.state_sha256, step_sha, execution.backend)
    rows = (
        (bundle.q3, _q3_token, PROVIDER_Q3),
        (bundle.controls, _control_token, PROVIDER_CONTROLS),
        (bundle.closure, _closure_token, PROVIDER_CLOSURE),
        (bundle.fluid, _fluid_token, PROVIDER_FLUID),
    )
    for value, token, provider in rows:
        actual = (value.execution_fingerprint,
                  value.base_v78_execution_fingerprint,
                  value.parent_base_state_sha256,
                  value.accepted_step_receipt_fingerprint,
                  value.backend)
        if (actual != common or value.provider_id != provider or
                value.receipt_sha256 != token(value)):
            raise SingleOwnerQ4V89Error(
                f"v89 {provider} receipt is stale, mutated or mismatched")
    if (np.any(bundle.q3.routes != ROUTE_COMBINE) or
            np.any(bundle.q3.bed_capacity_x_kg_m_s != 0.0) or
            np.any(bundle.q3.bed_capacity_y_kg_m_s != 0.0) or
            bundle.controls.iopsus != 0 or
            bundle.controls.neglect_entrainment):
        raise SingleOwnerQ4V89Error(
            "v89 provider combination is outside the safe ownership slice")


@dataclass(frozen=True)
class SelectedBedReceiptV89:
    control_receipt_sha256: str
    base_trial_sha256: str
    exchange_bed_delta_kg_m2: np.ndarray
    wetdry_bed_delta_kg_m2: np.ndarray
    selected_morphological_bed_delta_kg_m2: np.ndarray
    exchange_owner_count: np.ndarray
    flux_divergence_owner_count: np.ndarray
    selector_application_count: np.ndarray
    receipt_sha256: str


def _bed_token(value: SelectedBedReceiptV89) -> str:
    return _fingerprint({
        "control_receipt_sha256": value.control_receipt_sha256,
        "base_trial_sha256": value.base_trial_sha256,
        "arrays_sha256": [_array_digest(array) for array in (
            value.exchange_bed_delta_kg_m2,
            value.wetdry_bed_delta_kg_m2,
            value.selected_morphological_bed_delta_kg_m2,
            value.exchange_owner_count,
            value.flux_divergence_owner_count,
            value.selector_application_count)],
    })


def select_bed_receipt_v89(
    control: SuspendedControlReceiptV89,
    parent: MultifractionSuspendedStateV78,
    trial: MultifractionSuspendedTrialV78,
) -> SelectedBedReceiptV89:
    if control.receipt_sha256 != _control_token(control):
        raise SingleOwnerQ4V89Error("v89 control receipt differs")
    exchange = _frozen(
        -trial.transferred_to_suspension_kg_m2,
        parent.suspended_mass_kg_m2.shape, "exchange bed delta")
    wetdry = _frozen(
        trial.wetdry_transfer_to_bed_kg_m2,
        exchange.shape, "wet/dry bed delta", nonnegative=True)
    observed = trial.candidate.bed_mass_kg_m2-parent.bed_mass_kg_m2
    scale = np.maximum(1.0, np.maximum(np.abs(observed),
                                      np.abs(exchange+wetdry)))
    if np.any(np.abs(observed-(exchange+wetdry)) >
              16384.0*np.finfo(float).eps*scale):
        raise SingleOwnerQ4V89Error(
            "v89 inherited v78 bed mutation differs from selected exchange")
    ng = exchange.shape[0]
    one = _frozen(np.ones(ng, dtype=np.int32), (ng,),
                  "exchange owner count", integer=True)
    zero = _frozen(np.zeros(ng, dtype=np.int32), (ng,),
                   "flux owner count", integer=True)
    placeholder = SelectedBedReceiptV89(
        control.receipt_sha256, trial.trial_sha256, exchange, wetdry,
        exchange, one, zero, one, "")
    return SelectedBedReceiptV89(
        **{**placeholder.__dict__, "receipt_sha256": _bed_token(placeholder)})


@dataclass(frozen=True)
class OwnerCountsV89:
    q3_capacity_receipt: int = 1
    route_selector: int = 1
    boundary: int = 1
    fct: int = 1
    horizontal_diffusion: int = 1
    settling: int = 1
    exchange: int = 1
    wetdry_cleanup: int = 1
    bed_selector: int = 1
    base_trial_publication: int = 1
    dynamic_vicuv_momentum_consumer: int = 0
    closure_candidate_commit: int = 0
    dynamic_fluid_consumer: int = 0
    q3_bed_transport_consumer: int = 0
    duplicate_owner: int = 0


@dataclass(frozen=True)
class SingleOwnerTrialV89:
    execution_fingerprint: str
    parent_state_sha256: str
    accepted_step_receipt_sha256: str
    provider_bundle_sha256: str
    base_trial: MultifractionSuspendedTrialV78
    selected_bed: SelectedBedReceiptV89
    candidate: SingleOwnerStateV89
    owner_counts: OwnerCountsV89
    backend: str
    trial_sha256: str


def _trial_token(value: SingleOwnerTrialV89) -> str:
    return _fingerprint({
        "execution_fingerprint": value.execution_fingerprint,
        "parent_state_sha256": value.parent_state_sha256,
        "accepted_step_receipt_sha256":
            value.accepted_step_receipt_sha256,
        "provider_bundle_sha256": value.provider_bundle_sha256,
        "base_trial_sha256": value.base_trial.trial_sha256,
        "selected_bed_receipt_sha256": value.selected_bed.receipt_sha256,
        "candidate_state_sha256": value.candidate.state_sha256,
        "owner_counts": value.owner_counts.__dict__,
        "backend": value.backend,
    })


def _build_trial(execution: SingleOwnerExecutionV89,
                 parent: SingleOwnerStateV89,
                 step: Mapping[str, object], bundle: ProviderBundleV89,
                 base_trial: MultifractionSuspendedTrialV78,
                 backend: str) -> SingleOwnerTrialV89:
    try:
        _validate_multifraction_trial_v78(
            execution.base_execution, parent.base_state, base_trial)
    except (TypeError, ValueError) as error:
        raise SingleOwnerQ4V89Error(
            "v89 base trial is stale or cross-parent") from error
    expected_backend = (ROLE_CPU_ORACLE if execution.backend == BACKEND_CPU
                        else ROLE_CUDA)
    expected_time = parent.time_s+float(step["dt_s"])
    if (base_trial.execution_fingerprint !=
            execution.base_execution.fingerprint or
            base_trial.parent_state_sha256 != parent.base_state.state_sha256 or
            base_trial.accepted_step_receipt_sha256 != step["fingerprint"] or
            base_trial.backend != backend or backend != expected_backend or
            base_trial.candidate.accepted_updates != parent.accepted_updates+1 or
            base_trial.candidate.time_s != expected_time):
        raise SingleOwnerQ4V89Error(
            "v89 base trial identity, backend or accepted endpoint differs")
    selected = select_bed_receipt_v89(
        bundle.controls, parent.base_state, base_trial)
    chain = _fingerprint({
        "previous": parent.accepted_receipt_chain_sha256,
        "accepted_step_receipt": step["fingerprint"],
        "provider_bundle": bundle.bundle_sha256,
        "base_trial": base_trial.trial_sha256,
        "selected_bed": selected.receipt_sha256,
    })
    candidate = _wrap_state(execution, base_trial.candidate, chain)
    counts = OwnerCountsV89()
    if any(value not in (0, 1) for value in counts.__dict__.values()) or \
            counts.duplicate_owner != 0:
        raise SingleOwnerQ4V89Error("v89 owner counts differ")
    placeholder = SingleOwnerTrialV89(
        execution.fingerprint, parent.state_sha256, str(step["fingerprint"]),
        bundle.bundle_sha256, base_trial, selected, candidate, counts,
        backend, "")
    return SingleOwnerTrialV89(
        **{**placeholder.__dict__, "trial_sha256": _trial_token(placeholder)})


def _validate_trial(execution: SingleOwnerExecutionV89,
                    parent: SingleOwnerStateV89,
                    trial: SingleOwnerTrialV89) -> None:
    validate_single_owner_state_v89(parent, execution)
    if (not isinstance(trial, SingleOwnerTrialV89) or
            trial.execution_fingerprint != execution.fingerprint or
            trial.parent_state_sha256 != parent.state_sha256 or
            trial.trial_sha256 != _trial_token(trial) or
            trial.selected_bed.receipt_sha256 !=
                _bed_token(trial.selected_bed)):
        raise SingleOwnerQ4V89Error(
            "v89 trial is stale, mutated or belongs to another parent")
    validate_single_owner_state_v89(trial.candidate, execution)


def _begin_cpu_numerics(
    execution: SingleOwnerExecutionV89, parent: SingleOwnerStateV89,
    step: Mapping[str, object], bundle: ProviderBundleV89, inputs: dict,
) -> MultifractionSuspendedTrialV78:
    """Compose accepted operators once and publish one canonical v78 trial."""
    base78 = execution.base_execution
    base = base78.base_execution
    ng, nc, nf, nb = (execution.fraction_count, execution.cell_count,
                      execution.face_count, execution.boundary_count)
    if "suspended_transport_capacity_kg_m_s" in inputs:
        raise SingleOwnerQ4V89Error(
            "v89 capacity must come only from the Q3 receipt")
    required = {
        "depth_before_m", "depth_after_m", "accepted_water_flux_m2_s",
        "reconstructed_left_concentration_kg_m3",
        "reconstructed_right_concentration_kg_m3",
        "prescribed_boundary_concentration_kg_m3",
        "accepted_face_exchange_area_m2", "transport_face_active",
        "endpoint_momentum_m2_s", "shear_velocity_m_s",
        "surface_fraction", "hydro_constant", "dt_s",
    }
    if set(inputs) != required:
        raise SingleOwnerQ4V89Error("v89 CPU input inventory differs")
    dt_raw = inputs["dt_s"]
    if (isinstance(dt_raw, bool) or not isinstance(dt_raw, (int, float)) or
            not math.isfinite(float(dt_raw)) or float(dt_raw) <= 0.0):
        raise SingleOwnerQ4V89Error("v89 timestep must be positive")
    dt = float(dt_raw)
    receipt = validate_implicit_fwave_midpoint_step_receipt_v78(
        step, hydraulic_execution_sha256=base78.hydraulic_execution_sha256,
        parent_accepted_updates=parent.accepted_updates,
        parent_time_s=parent.time_s, dt_s=dt)
    h0 = _input_array(inputs["depth_before_m"], (nc,), "depth before",
                      nonnegative=True)
    h1 = _input_array(inputs["depth_after_m"], (nc,), "depth after",
                      nonnegative=True)
    if (np.any(h0 <= base.dry_tolerance_m) or
            np.any(h1 <= base.dry_tolerance_m)):
        raise SingleOwnerQ4V89Error(
            "v89 moving wet/dry composition is staged for v97")
    water = _input_array(inputs["accepted_water_flux_m2_s"], (nf,),
                         "accepted water flux")
    left = _input_array(inputs["reconstructed_left_concentration_kg_m3"],
                        (ng, nf), "left concentration")
    right = _input_array(inputs["reconstructed_right_concentration_kg_m3"],
                         (ng, nf), "right concentration")
    prescribed = _input_array(
        inputs["prescribed_boundary_concentration_kg_m3"], (ng, nb),
        "prescribed boundary concentration", nonnegative=True)
    face_area = _input_array(inputs["accepted_face_exchange_area_m2"],
                             (nf,), "face exchange area", nonnegative=True)
    face_active = _input_array(inputs["transport_face_active"], (nf,),
                               "transport face active", boolean=True)
    momentum = _input_array(inputs["endpoint_momentum_m2_s"], (2, nc),
                            "endpoint momentum")
    shear = _input_array(inputs["shear_velocity_m_s"], (nc,),
                         "shear velocity", nonnegative=True)
    fraction = _input_array(inputs["surface_fraction"], (ng, nc),
                            "surface fraction", nonnegative=True)
    constant = _input_array(inputs["hydro_constant"], (nc,),
                            "hydro constant", boolean=True)
    if np.any(fraction > 1.0):
        raise SingleOwnerQ4V89Error("v89 surface fraction exceeds one")

    accepted = parent.base_state.lifecycle
    interior = np.where(
        h0[None, :] > base.dry_tolerance_m,
        accepted.suspended_mass_kg_m2/np.where(
            h0[None, :] > base.dry_tolerance_m, h0[None, :], 1.0), 0.0)
    boundary = np.zeros((ng, nf), dtype=np.float64)
    next_last = np.array(
        accepted.boundary_last_outflow_concentration_kg_m3, copy=True)
    next_clock = np.array(accepted.boundary_return_clock_s, copy=True)
    expected_advective = np.zeros((ng, nb), dtype=np.float64)
    for face in range(nf):
        boundary_id = int(base.boundary_index[face])
        if boundary_id < 0:
            continue
        cell = int(base.owner[face])
        q_in = -water[face]*base.face_length_m[face]
        result = resolve_flow2dh_open_boundary(
            prescribed[:, boundary_id], interior[:, cell],
            np.full(ng, q_in), dt_s=dt,
            neumann=np.asarray(
                base.profile.neumann_boundary_by_fraction, dtype=bool),
            return_time_s=base.boundary_return_time_s[:, boundary_id],
            previous_memory=Flow2DHBoundaryMemory(
                accepted.boundary_last_outflow_concentration_kg_m3[:, face],
                accepted.boundary_return_clock_s[:, face]))
        boundary[:, face] = result.boundary_concentration_kg_m3
        next_last[:, face] = result.memory.last_outflow_concentration_kg_m3
        next_clock[:, face] = result.memory.return_clock_s
        expected_advective[:, boundary_id] += result.advective_mass_in_kg

    advected, advective_flux, advective_ledger = _cpu_fct(
        base, accepted.suspended_mass_kg_m2, h0, h1, water, left, right,
        boundary, constant, dt)
    scale = np.maximum(1.0, np.maximum(
        np.abs(expected_advective), np.abs(advective_ledger)))
    if np.any(np.abs(expected_advective-advective_ledger) >
              8192.0*np.finfo(float).eps*scale):
        raise SingleOwnerQ4V89Error(
            "v89 FCT boundary ledger differs from the boundary owner")
    v82_execution = resolve_suspended_support_execution_v82(
        base78, dicuv_cell_m2_s=bundle.closure.used_total_dicuv_m2_s,
        tpsnumber_by_fraction=np.ones(ng))
    diffused, diffusive_face_mass, diffusive_ledger, _ = \
        _cpu_spatial_diffusion(
            v82_execution, advected, h1, boundary, face_area,
            face_active, dt)

    wet = h1 > base.dry_tolerance_m
    speed = np.zeros(nc)
    speed[wet] = np.hypot(momentum[0, wet], momentum[1, wet])/h1[wet]
    concentration = np.zeros((nc, ng))
    concentration[wet] = (diffused[:, wet]/h1[wet]).T
    settling = settlefrm2_noncohesive(
        concentration, base.profile.sediment_d50_m,
        base.profile.settling_fac_dss,
        base.profile.sediment_density_kg_m3,
        base.profile.water_density_kg_m3,
        base.profile.temperature_value_c,
        settling_reference_concentration_kg_m3=
            base.profile.settling_reference_concentration_kg_m3,
        gravity_m_s2=base.profile.gravity_m_s2, wet_mask=wet)
    exchange = advance_flow2dh_equilibrium_exchange(
        diffused, accepted.bed_mass_kg_m2,
        bundle.q3.suspended_capacity_kg_m_s, fraction, speed, shear,
        h0, h1, settling.settling_velocity_m_s.T,
        base.cell_area_m2, dt_s=dt, fac_tsd=base.profile.fac_tsd,
        sediment_threshold_m=base.profile.sediment_threshold_m,
        minimum_reference_depth_m=base.profile.reference_depth_floor_m,
        flow_wet_mask=wet)
    cleanup = np.where(wet[None, :], 0.0,
                       exchange.suspended_mass_kg_m2)
    final_suspended = exchange.suspended_mass_kg_m2-cleanup
    final_bed = exchange.bed_mass_kg_m2+cleanup
    cleanup_water = -np.sum(cleanup*base.cell_area_m2[None, :], axis=1)
    cleanup_bed = -cleanup_water
    candidate_lifecycle = resolve_suspended_lifecycle_state_v76(
        base, suspended_mass_kg_m2=final_suspended,
        bed_mass_kg_m2=final_bed,
        boundary_last_outflow_concentration_kg_m3=next_last,
        boundary_return_clock_s=next_clock,
        advective_boundary_ledger_kg=(
            accepted.advective_boundary_ledger_kg+advective_ledger),
        diffusive_boundary_ledger_kg=(
            accepted.diffusive_boundary_ledger_kg+diffusive_ledger),
        water_exchange_ledger_kg=(
            accepted.water_exchange_ledger_kg+
            exchange.suspended_exchange_ledger_kg),
        bed_exchange_ledger_kg=(
            accepted.bed_exchange_ledger_kg+
            exchange.bed_exchange_ledger_kg),
        wetdry_water_ledger_kg=accepted.wetdry_water_ledger_kg+cleanup_water,
        wetdry_bed_ledger_kg=accepted.wetdry_bed_ledger_kg+cleanup_bed,
        reference_water_mass_kg=accepted.reference_water_mass_kg,
        reference_bed_mass_kg=accepted.reference_bed_mass_kg,
        accepted_updates=parent.accepted_updates+1,
        time_s=parent.time_s+dt)
    return build_multifraction_trial_v78(
        base78, parent.base_state, receipt, candidate_lifecycle,
        boundary_concentration_kg_m3=boundary,
        advective_face_flux_kg_m_s=advective_flux,
        diffusive_face_mass_owner_out_kg=diffusive_face_mass,
        settling_velocity_m_s=settling.settling_velocity_m_s.T,
        equilibrium_concentration_kg_m3=
            exchange.equilibrium_concentration_kg_m3,
        transferred_to_suspension_kg_m2=
            exchange.transferred_to_suspension_kg_m2,
        wetdry_transfer_to_bed_kg_m2=cleanup,
        observed_parallel_workers=1, backend=ROLE_CPU_ORACLE)


class SingleOwnerQ4LifecycleV89:
    """Explicit CPU owner for one bounded Q4/v89 accepted update."""

    def __init__(self, execution: SingleOwnerExecutionV89):
        if execution.backend != BACKEND_CPU:
            raise SingleOwnerQ4V89Error(
                "v89 CPU owner requires the explicit CPU execution")
        self.execution = execution
        self._open_parent: str | None = None
        self._open_trial: str | None = None

    def begin_trial(self, parent: SingleOwnerStateV89, *,
                    accepted_step_receipt: Mapping[str, object],
                    providers: ProviderBundleV89,
                    cancelled: bool = False, **inputs) -> SingleOwnerTrialV89:
        validate_single_owner_state_v89(parent, self.execution)
        if self._open_trial is not None:
            raise SingleOwnerQ4V89Error("v89 previous trial is unfinished")
        if cancelled:
            raise SingleOwnerQ4V89Error(
                "v89 cancelled before any numerical owner ran")
        if parent.accepted_updates >= MAX_ACCEPTED_UPDATES:
            raise SingleOwnerQ4V89Error(
                "v89 accepted update limit was reached")
        validate_provider_bundle_v89(
            providers, self.execution, parent, accepted_step_receipt)
        try:
            base_trial = _begin_cpu_numerics(
                self.execution, parent, accepted_step_receipt, providers,
                dict(inputs))
            trial = _build_trial(
                self.execution, parent, accepted_step_receipt, providers,
                base_trial, ROLE_CPU_ORACLE)
        except Exception as error:
            if isinstance(error, SingleOwnerQ4V89Error):
                raise
            raise SingleOwnerQ4V89Error(
                "v89 CPU single-owner transaction failed") from error
        self._open_parent = parent.state_sha256
        self._open_trial = trial.trial_sha256
        return trial

    def _finish(self, parent: SingleOwnerStateV89,
                trial: SingleOwnerTrialV89, *, commit: bool
                ) -> SingleOwnerStateV89:
        _validate_trial(self.execution, parent, trial)
        if (self._open_parent != parent.state_sha256 or
                self._open_trial != trial.trial_sha256):
            raise SingleOwnerQ4V89Error(
                "v89 trial is not the open CPU transaction")
        base = (commit_multifraction_trial_v78(
                    self.execution.base_execution, parent.base_state,
                    trial.base_trial) if commit else
                rollback_multifraction_trial_v78(
                    self.execution.base_execution, parent.base_state,
                    trial.base_trial))
        result = trial.candidate if commit else parent
        if commit and base.state_sha256 != result.base_state.state_sha256:
            raise SingleOwnerQ4V89Error("v89 CPU committed state differs")
        if not commit and base is not parent.base_state:
            raise SingleOwnerQ4V89Error("v89 CPU rollback changed parent")
        self._open_parent = self._open_trial = None
        return result

    def commit(self, parent, trial):
        return self._finish(parent, trial, commit=True)

    def rollback(self, parent, trial):
        return self._finish(parent, trial, commit=False)

    def cancel(self, parent, trial):
        return self.rollback(parent, trial)

    def checkpoint(self, state: SingleOwnerStateV89) -> dict:
        if self._open_trial is not None:
            raise SingleOwnerQ4V89Error(
                "v89 checkpoint requires a committed point")
        return single_owner_checkpoint_v89(self.execution, state)

    def restore_checkpoint(self, document: Mapping[str, object]
                           ) -> SingleOwnerStateV89:
        if self._open_trial is not None:
            raise SingleOwnerQ4V89Error(
                "v89 restart cannot replace an open transaction")
        return load_single_owner_checkpoint_v89(self.execution, document)


def single_owner_checkpoint_v89(execution: SingleOwnerExecutionV89,
                                state: SingleOwnerStateV89) -> dict:
    validate_single_owner_state_v89(state, execution)
    body = {
        "schema": SCHEMA,
        "scope_id": CHECKPOINT_SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "execution_fingerprint": execution.fingerprint,
        "integrator": INTEGRATOR,
        "accepted_receipt_chain_sha256":
            state.accepted_receipt_chain_sha256,
        "base_v78_checkpoint": multifraction_checkpoint_v78(
            execution.base_execution, state.base_state),
        "state_sha256": state.state_sha256,
    }
    return {**body, "fingerprint": _fingerprint(body)}


def load_single_owner_checkpoint_v89(
    execution: SingleOwnerExecutionV89, document: Mapping[str, object],
) -> SingleOwnerStateV89:
    expected = {
        "schema", "scope_id", "semantic_id", "execution_fingerprint",
        "integrator", "accepted_receipt_chain_sha256",
        "base_v78_checkpoint", "state_sha256", "fingerprint",
    }
    if not isinstance(document, Mapping) or set(document) != expected:
        raise SingleOwnerQ4V89Error("v89 checkpoint inventory differs")
    body = {key: document[key] for key in document if key != "fingerprint"}
    if (document["fingerprint"] != _fingerprint(body) or
            document["schema"] != SCHEMA or
            document["scope_id"] != CHECKPOINT_SCOPE_ID or
            document["semantic_id"] != SEMANTIC_ID or
            document["execution_fingerprint"] != execution.fingerprint or
            document["integrator"] != INTEGRATOR):
        raise SingleOwnerQ4V89Error("v89 checkpoint identity differs")
    base = load_multifraction_checkpoint_v78(
        execution.base_execution, document["base_v78_checkpoint"])
    state = _wrap_state(
        execution, base, str(document["accepted_receipt_chain_sha256"]))
    if state.state_sha256 != document["state_sha256"]:
        raise SingleOwnerQ4V89Error("v89 checkpoint state differs")
    return state


def method_identity_v89() -> dict:
    return {
        "scope_id": SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "integrator": INTEGRATOR,
        "backend_roles": [BACKEND_CPU, BACKEND_GPU],
        "canonical_state_owner": "one_v78_state",
        "one_update_limit": MAX_ACCEPTED_UPDATES,
        "q3": "suspended_capacity_and_all_combine_route_consumed_once",
        "controls": "exchange_bed_selector_authenticated_once",
        "closure": {
            "lagged_dicuv": "consumed_once",
            "vicuv": "staged_no_momentum_consumer",
            "candidate": "staged_not_committed",
        },
        "fluid": "static_profile_identity_guard_dynamic_TS_staged",
        "provider_receipt_provenance":
            "caller_sealed_typed_ABI_not_real_v84_or_v87_invocation",
        "owner_count_semantics":
            "authenticated_expected_stage_counts_not_kernel_launch_counters",
        "automatic_cpu_fallback": False,
        "long_run_claimed": False,
        "product_admission": False,
        "delft3d_source_modified": False,
    }


def verify_sources_v89(root: str | Path) -> dict[str, str]:
    root = Path(root)
    paths = (
        "braids_core/sediment/single_owner_q4_v89.py",
        "braids_gpu/single_owner_q4_v89.py",
        "tests/test_q4_single_owner_v89.py",
        "tests/test_q4_single_owner_cuda_v89.py",
    )
    result = {}
    for relative in paths:
        path = root/relative
        if not path.is_file():
            raise SingleOwnerQ4V89Error(
                "v89 additive source is absent: " + relative)
        result[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


__all__ = [
    "BACKEND_CPU", "BACKEND_GPU", "CHECKPOINT_SCOPE_ID", "INTEGRATOR",
    "MAX_ACCEPTED_UPDATES", "SCHEMA", "SCOPE_ID", "SEMANTIC_ID",
    "ClosureReceiptV89", "FluidPropertyReceiptV89", "OwnerCountsV89",
    "ProviderBundleV89", "Q3CapacityRouteReceiptV89",
    "SelectedBedReceiptV89", "SingleOwnerExecutionV89",
    "SingleOwnerQ4LifecycleV89", "SingleOwnerQ4V89Error",
    "SingleOwnerStateV89", "SingleOwnerTrialV89",
    "SuspendedControlReceiptV89", "execution_document_v89",
    "load_single_owner_checkpoint_v89", "make_closure_receipt_v89",
    "make_fluid_property_receipt_v89", "make_provider_bundle_v89",
    "make_q3_capacity_route_receipt_v89",
    "make_suspended_control_receipt_v89", "method_identity_v89",
    "resolve_single_owner_execution_v89", "resolve_single_owner_state_v89",
    "select_bed_receipt_v89", "single_owner_checkpoint_v89",
    "validate_provider_bundle_v89", "validate_single_owner_state_v89",
    "verify_sources_v89",
]
