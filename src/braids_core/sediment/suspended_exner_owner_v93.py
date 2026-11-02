"""Authoritative Q4/v93 suspended/bed-load Exner composition.

The locked FLOW ``bott3d`` owner forms one bed change from two independent
terms: bed/total-load flux divergence and, for a combined suspended fraction,
either water-column exchange or suspended-flux divergence selected by
``NeglectEntrainment``.  This module applies that ownership rule to the exact
accepted v92 scalar trial.  It deliberately leaves the v92 water-column
exchange candidate untouched and advances a separate authoritative bed state.

The Q3 input is already face reconstructed, availability limited and integrated
over the accepted step.  Its typed receipt is therefore a *consumer ABI*, not
a second constitutive or face-flux owner.  The scope is fully wet, fixed grid,
at most four accepted updates, and no product admission.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Callable, Mapping, Protocol

import numpy as np

from .hles_producer_v90 import CurrentClosureHandoffV90, HlesProducerStateV90
from .shared_turbulence_consumers_v91 import (
    STATE_DOMAIN_MOVING_WETDRY_V97,
)
from .route_lifecycle_composition_v81 import (
    ROUTE_COMBINE,
    ROUTE_OFF,
    ROUTE_TOTAL,
)
from .suspended_controls_v86 import (
    TOPOLOGY_SELECTOR_ABI_ID,
    suspended_flux_divergence_topology_v86,
)
from .v92_bed_projection_v93 import (
    V92BedProjectionReceiptV93,
    project_v92_authoritative_bed_v93,
    validate_v92_bed_projection_receipt_v93,
)
from .transported_fluid_settling_v92 import (
    TransportedFluidExecutionV92,
    TransportedFluidStateV92,
    TransportedFluidTrialV92,
    _validate_state as _validate_v92_state,
    _validate_trial as _validate_v92_trial,
    begin_transported_fluid_cpu_trial_v92,
    commit_transported_fluid_trial_v92,
    execution_document_v92,
    load_transported_fluid_checkpoint_v92,
    rollback_transported_fluid_trial_v92,
    transported_fluid_checkpoint_v92,
)


SCHEMA = 1
SCOPE_ID = "braids-q4-flow2dh-authoritative-exner-v93-v1"
CHECKPOINT_SCOPE_ID = SCOPE_ID + ":checkpoint"
SELECTOR_SCOPE_ID = SCOPE_ID + ":v86-selector"
Q3_STEP_SCOPE_ID = SCOPE_ID + ":q3-bed-step"
SEMANTIC_ID = hashlib.sha256(SCOPE_ID.encode("ascii")).hexdigest()
ZERO_VECTOR_SCOPE_ID_V93 = SCOPE_ID + ":suscor-false-zero-vector-v94b"
ZERO_VECTOR_SEMANTIC_ID_V93 = hashlib.sha256(
    ZERO_VECTOR_SCOPE_ID_V93.encode("ascii")).hexdigest()
INTEGRATOR = "implicit_fwave_midpoint"
MAX_ACCEPTED_UPDATES = 4
BED_BALANCE_EXCHANGE = "erosion_deposition_exchange"
BED_BALANCE_FLUX_DIVERGENCE = "suspended_flux_divergence"
SUSPENDED_VECTOR_CONSTRUCTED_V93 = "constructed_suscor_vector"
SUSPENDED_VECTOR_EXACT_ZERO_V93 = "suscor_false_exact_zero_vector"
ZERO_VECTOR_SELECTOR_ABI_ID_V93 = hashlib.sha256(
    (ZERO_VECTOR_SCOPE_ID_V93 + ":selector").encode("ascii")).hexdigest()
ACCEPTED_Q3_PROVIDER_KIND = {
    "cpu": "q4_owned_v84_mixed_face_no_scalar",
    "implicit_fwave_midpoint_gpu":
        "q4_owned_v84_mixed_face_no_scalar_actual_cuda",
}
ACTIVE_Q3_PROVIDER_KIND_V94 = {
    "cpu": "q4_owned_v83_trafrm_minus1_active_iopsus_face_no_scalar",
    "implicit_fwave_midpoint_gpu":
        "q4_owned_v83_trafrm_minus1_active_iopsus_face_no_scalar_actual_cuda",
}
_SHA256 = __import__("re").compile(r"[0-9a-f]{64}\Z")


class SuspendedExnerV93Error(ValueError):
    """A v93 identity, receipt, state, ledger, or transaction is invalid."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError, OverflowError) as error:
        raise SuspendedExnerV93Error(
            "v93 value is not finite canonical JSON") from error


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise SuspendedExnerV93Error(f"v93 {name} must be a SHA-256 digest")
    return value


def _array_digest(*values: np.ndarray) -> str:
    digest = hashlib.sha256()
    for raw in values:
        value = np.ascontiguousarray(raw)
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(_canonical(list(value.shape)))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _array(value: object, shape: tuple[int, ...], name: str, *,
           integer: bool = False, nonnegative: bool = False) -> np.ndarray:
    raw = np.asarray(value)
    admitted_kinds = "iu" if integer else "fiu"
    if raw.shape != shape or raw.dtype.kind not in admitted_kinds:
        raise SuspendedExnerV93Error(
            f"v93 {name} must be numeric with shape {shape}")
    if integer and raw.size:
        limits = np.iinfo(np.int32)
        if np.any(raw < limits.min) or np.any(raw > limits.max):
            raise SuspendedExnerV93Error(
                f"v93 {name} is outside the authenticated int32 domain")
    result = np.array(raw, dtype=np.int32 if integer else np.float64,
                      order="C", copy=True)
    if not np.isfinite(result).all():
        raise SuspendedExnerV93Error(f"v93 {name} must be finite")
    if nonnegative and np.any(result < 0.0):
        raise SuspendedExnerV93Error(f"v93 {name} must be nonnegative")
    result.setflags(write=False)
    return result


def _ledger_tolerance(*values: np.ndarray) -> np.ndarray:
    scale = np.ones(np.asarray(values[0]).shape, dtype=np.float64)
    for value in values:
        scale = np.maximum(scale, np.abs(value))
    return 131072.0*np.finfo(np.float64).eps*scale


def _moving_wetdry(execution: "SuspendedExnerExecutionV93") -> bool:
    return (execution.base_execution.base_execution.state_domain_mode ==
            STATE_DOMAIN_MOVING_WETDRY_V97)


@dataclass(frozen=True)
class SuspendedExnerExecutionV93:
    base_execution: TransportedFluidExecutionV92
    routes: np.ndarray
    ordered_fraction_ids: tuple[str, ...]
    neglect_entrainment: bool
    morphological_factor: float
    q3_execution_fingerprint: str
    q3_topology_id: str
    q3_binding_sha256: str
    q3_boundary_names: tuple[str, ...]
    semantic_id: str
    fingerprint: str
    suspended_vector_mode: str = SUSPENDED_VECTOR_CONSTRUCTED_V93

    @property
    def backend(self) -> str:
        return self.base_execution.backend

    @property
    def fraction_count(self) -> int:
        return self.base_execution.fraction_count

    @property
    def cell_count(self) -> int:
        return self.base_execution.cell_count

    @property
    def face_count(self) -> int:
        return self.base_execution.face_count


def _execution_payload(value: SuspendedExnerExecutionV93) -> dict:
    if value.suspended_vector_mode == SUSPENDED_VECTOR_CONSTRUCTED_V93:
        if value.semantic_id != SEMANTIC_ID:
            raise SuspendedExnerV93Error(
                "v93 constructed-vector semantic identity differs")
    elif value.suspended_vector_mode == SUSPENDED_VECTOR_EXACT_ZERO_V93:
        if (value.semantic_id != ZERO_VECTOR_SEMANTIC_ID_V93 or
                not value.neglect_entrainment):
            raise SuspendedExnerV93Error(
                "v93 exact-zero vector mode requires its additive semantic "
                "identity and NeglectEntrainment=true")
    else:
        raise SuspendedExnerV93Error(
            "v93 suspended-vector selection mode is not admitted")
    payload = {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "semantic_id": value.semantic_id,
        "base_v92_execution": execution_document_v92(value.base_execution),
        "backend": value.backend,
        "integrator": INTEGRATOR,
        "routes": value.routes.tolist(),
        "ordered_fraction_ids": list(value.ordered_fraction_ids),
        "neglect_entrainment": value.neglect_entrainment,
        "morphological_factor": value.morphological_factor,
        "bed_balance_method": (BED_BALANCE_FLUX_DIVERGENCE
                               if value.neglect_entrainment else
                               BED_BALANCE_EXCHANGE),
        "q3_execution_fingerprint": value.q3_execution_fingerprint,
        "q3_topology_id": value.q3_topology_id,
        "q3_binding_sha256": value.q3_binding_sha256,
        "q3_boundary_names": list(value.q3_boundary_names),
        "route_semantics": {
            "off": "zero_suspended_zero_q3_zero_exner",
            "total": "zero_suspended_plus_q3_total_load_once",
            "combine": "selected_suspended_plus_q3_bed_load_once",
        },
        "authoritative_bed_owner_count": 1,
        "water_column_exchange_changed_by_selector": False,
        "accepted_update_limit": MAX_ACCEPTED_UPDATES,
        "fixed_grid": True,
        "moving_wet_dry": False,
        "automatic_cpu_fallback": False,
        "product_admission": False,
    }
    # Omit every additive v94b key on the historical path.  This preserves
    # old execution documents and fingerprints byte-for-byte.
    if value.suspended_vector_mode == SUSPENDED_VECTOR_EXACT_ZERO_V93:
        payload.update({
            "suspended_vector_mode": value.suspended_vector_mode,
            "topology_divergence_kernel_count_per_combine_fraction": 0,
            "zero_vector_materialization_count_per_combine_fraction": 1,
        })
    return payload


def resolve_suspended_exner_execution_v93(
    base_execution: TransportedFluidExecutionV92, *, routes: object,
    ordered_fraction_ids: tuple[str, ...] | list[str],
    neglect_entrainment: bool,
    q3_execution_fingerprint: str,
    q3_topology_id: str, q3_binding_sha256: str,
    q3_boundary_names: tuple[str, ...] | list[str],
    morphological_factor: float = 1.0,
) -> SuspendedExnerExecutionV93:
    execution_document_v92(base_execution)
    route = _array(routes, (base_execution.fraction_count,), "routes",
                   integer=True)
    if np.any(~np.isin(route, (ROUTE_OFF, ROUTE_TOTAL, ROUTE_COMBINE))):
        raise SuspendedExnerV93Error("v93 route is not OFF, TOTAL, or COMBINE")
    if type(neglect_entrainment) is not bool:
        raise SuspendedExnerV93Error(
            "v93 NeglectEntrainment must be Boolean")
    ids = tuple(ordered_fraction_ids)
    if (len(ids) != base_execution.fraction_count or
            any(not isinstance(value, str) or not value for value in ids) or
            len(set(ids)) != len(ids)):
        raise SuspendedExnerV93Error(
            "v93 ordered fraction identities are invalid")
    if (isinstance(morphological_factor, bool) or
            not isinstance(morphological_factor, (int, float)) or
            float(morphological_factor) != 1.0):
        raise SuspendedExnerV93Error(
            "v93 admits only authenticated morphological factor 1")
    q3 = _sha(q3_execution_fingerprint, "Q3 execution fingerprint")
    topology = _sha(q3_topology_id, "Q3 topology")
    binding = _sha(q3_binding_sha256, "Q3 binding")
    names = tuple(q3_boundary_names)
    boundary_count = (base_execution.base_execution.base_execution.
                      base_execution.boundary_count)
    if (len(names) != boundary_count or
            any(not isinstance(value, str) or not value for value in names) or
            len(set(names)) != len(names)):
        raise SuspendedExnerV93Error("v93 Q3 boundary names are invalid")
    placeholder = SuspendedExnerExecutionV93(
        base_execution, route, ids, neglect_entrainment, 1.0, q3,
        topology, binding, names, SEMANTIC_ID, "")
    return SuspendedExnerExecutionV93(
        **{**placeholder.__dict__,
           "fingerprint": _fingerprint(_execution_payload(placeholder))})


def resolve_suspended_exner_zero_vector_execution_v93(
    base_execution: TransportedFluidExecutionV92, *, routes: object,
    ordered_fraction_ids: tuple[str, ...] | list[str],
    q3_execution_fingerprint: str,
    q3_topology_id: str, q3_binding_sha256: str,
    q3_boundary_names: tuple[str, ...] | list[str],
    morphological_factor: float = 1.0,
) -> SuspendedExnerExecutionV93:
    """Resolve the source-legal SusCor=false/Neglect=true v94b selector.

    The historical resolver cannot construct this mode.  Reuse it only for
    its established topology/profile validation, then replace the two
    additive identities and recompute the execution fingerprint.
    """
    historical = resolve_suspended_exner_execution_v93(
        base_execution, routes=routes,
        ordered_fraction_ids=ordered_fraction_ids,
        neglect_entrainment=True,
        q3_execution_fingerprint=q3_execution_fingerprint,
        q3_topology_id=q3_topology_id,
        q3_binding_sha256=q3_binding_sha256,
        q3_boundary_names=q3_boundary_names,
        morphological_factor=morphological_factor)
    placeholder = SuspendedExnerExecutionV93(
        **{**historical.__dict__,
           "semantic_id": ZERO_VECTOR_SEMANTIC_ID_V93,
           "fingerprint": "",
           "suspended_vector_mode": SUSPENDED_VECTOR_EXACT_ZERO_V93})
    return SuspendedExnerExecutionV93(
        **{**placeholder.__dict__,
           "fingerprint": _fingerprint(_execution_payload(placeholder))})


def execution_document_v93(value: SuspendedExnerExecutionV93) -> dict:
    if not isinstance(value, SuspendedExnerExecutionV93):
        raise SuspendedExnerV93Error("v93 execution object differs")
    payload = _execution_payload(value)
    if value.fingerprint != _fingerprint(payload):
        raise SuspendedExnerV93Error("v93 execution fingerprint differs")
    return {**payload, "fingerprint": value.fingerprint}


@dataclass(frozen=True)
class SuspendedBedSelectionReceiptV93:
    canonical_selector_abi_id: str
    execution_fingerprint: str
    parent_state_sha256: str
    accepted_v92_trial_sha256: str
    accepted_hydraulic_receipt_sha256: str
    accepted_hydraulic_face_receipt_sha256: str
    exchange_bed_delta_kg_m2: np.ndarray
    flux_divergence_bed_delta_kg_m2: np.ndarray
    selected_bed_delta_kg_m2: np.ndarray
    selected_exchange_ledger_kg: np.ndarray
    selected_boundary_ledger_kg: np.ndarray
    excluded_water_exchange_kg: np.ndarray
    exchange_owner_count: np.ndarray
    flux_divergence_owner_count: np.ndarray
    selector_application_count: np.ndarray
    backend: str
    receipt_sha256: str
    topology_divergence_kernel_count: np.ndarray | None = None
    zero_vector_materialization_count: np.ndarray | None = None


def _selector_token(value: SuspendedBedSelectionReceiptV93) -> str:
    payload = {
        "scope_id": SELECTOR_SCOPE_ID,
        "canonical_selector_abi_id": value.canonical_selector_abi_id,
        "execution_fingerprint": value.execution_fingerprint,
        "parent_state_sha256": value.parent_state_sha256,
        "accepted_v92_trial_sha256": value.accepted_v92_trial_sha256,
        "accepted_hydraulic_receipt_sha256":
            value.accepted_hydraulic_receipt_sha256,
        "accepted_hydraulic_face_receipt_sha256":
            value.accepted_hydraulic_face_receipt_sha256,
        "arrays_sha256": _array_digest(
            value.exchange_bed_delta_kg_m2,
            value.flux_divergence_bed_delta_kg_m2,
            value.selected_bed_delta_kg_m2,
            value.selected_exchange_ledger_kg,
            value.selected_boundary_ledger_kg,
            value.excluded_water_exchange_kg,
            value.exchange_owner_count,
            value.flux_divergence_owner_count,
            value.selector_application_count),
        "backend": value.backend,
    }
    topology = value.topology_divergence_kernel_count
    materialization = value.zero_vector_materialization_count
    if (topology is None) != (materialization is None):
        raise SuspendedExnerV93Error(
            "v93 zero-vector receipt count inventory differs")
    if topology is not None:
        payload["zero_vector_owner_count_sha256"] = _array_digest(
            topology, materialization)
    return _fingerprint(payload)


def _suspended_components_cpu(
    execution: SuspendedExnerExecutionV93,
    parent: "SuspendedExnerStateV93",
    v92_trial: TransportedFluidTrialV92,
) -> tuple[np.ndarray, ...]:
    base_trial = v92_trial.suspended_trial.base_trial
    base = (execution.base_execution.base_execution.base_execution.
            base_execution)
    ng, nc = execution.fraction_count, execution.cell_count
    nb = (execution.base_execution.base_execution.base_execution.
          base_execution.boundary_count)
    exchange = -np.asarray(base_trial.transferred_to_suspension_kg_m2)
    cleanup = np.asarray(base_trial.wetdry_transfer_to_bed_kg_m2)
    if np.any(cleanup != 0.0) and not _moving_wetdry(execution):
        raise SuspendedExnerV93Error(
            "v93 moving wet/dry transfer is deferred to v97")
    adv = np.asarray(base_trial.advective_face_flux_kg_m_s)
    diff = np.asarray(base_trial.diffusive_face_mass_owner_out_kg)
    combine = np.asarray(execution.routes) == ROUTE_COMBINE
    inactive = ~combine
    parent_water = np.asarray(
        parent.base_state.suspended_state.suspended_mass_kg_m2)
    candidate_water = np.asarray(
        v92_trial.candidate.suspended_state.suspended_mass_kg_m2)
    if any(np.any(value[inactive] != 0.0) for value in (
            exchange, adv, diff, parent_water, candidate_water)):
        raise SuspendedExnerV93Error(
            "v93 OFF/TOTAL route acquired water-column suspended work")
    exchange[inactive] = 0.0
    if execution.suspended_vector_mode == SUSPENDED_VECTOR_EXACT_ZERO_V93:
        # Locked FLOW leaves ssuu/ssvv at their exact initialization zeros
        # when SusCor=false.  NeglectEntrainment then selects the divergence
        # of those zeros.  Raw v92 scalar fluxes/exchange above remain live.
        flux = np.zeros((ng, nc), dtype=np.float64)
        boundary = np.zeros((ng, nb), dtype=np.float64)
    else:
        flux, boundary = suspended_flux_divergence_topology_v86(
            base_execution=base, routes=execution.routes,
            advective_face_flux_kg_m_s=adv,
            diffusive_face_mass_owner_out_kg=diff,
            dt_s=float(v92_trial.hydraulic_receipt.dt_s))
    # The canonical v86 ABI, or the exact-zero v94b branch, has already
    # zeroed inactive routes.
    selected = flux if execution.neglect_entrainment else exchange
    exchange_ledger = np.sum(
        exchange*base.cell_area_m2[None, :], axis=1)
    exchange_ledger[inactive] = 0.0
    if execution.neglect_entrainment:
        selected_exchange = np.zeros(ng)
        selected_boundary = boundary
    else:
        selected_exchange = exchange_ledger
        selected_boundary = np.zeros(
            (ng, base.boundary_count), dtype=np.float64)
    return tuple(np.ascontiguousarray(value) for value in (
        exchange, flux, selected, selected_exchange, selected_boundary))


def _wrap_suspended_selection_v93(
    execution: SuspendedExnerExecutionV93,
    parent: "SuspendedExnerStateV93",
    v92_trial: TransportedFluidTrialV92, *,
    exchange_delta: object, flux_delta: object, selected_delta: object,
    selected_exchange_ledger: object, selected_boundary_ledger: object,
    backend: str,
) -> SuspendedBedSelectionReceiptV93:
    ng, nc = execution.fraction_count, execution.cell_count
    nb = (execution.base_execution.base_execution.base_execution.
          base_execution.boundary_count)
    arrays = [
        _array(exchange_delta, (ng, nc), "exchange bed delta"),
        _array(flux_delta, (ng, nc), "flux-divergence bed delta"),
        _array(selected_delta, (ng, nc), "selected bed delta"),
        _array(selected_exchange_ledger, (ng,), "selected exchange ledger"),
        _array(selected_boundary_ledger, (ng, nb),
               "selected named boundary ledger"),
    ]
    combine = np.asarray(execution.routes) == ROUTE_COMBINE
    area = (execution.base_execution.base_execution.base_execution.
            base_execution.cell_area_m2)
    excluded = _array(
        (np.sum(arrays[0]*area[None, :], axis=1)
         if execution.neglect_entrainment else np.zeros(ng)),
        (ng,), "excluded water exchange")
    exchange_owner = np.ascontiguousarray(
        (combine & (not execution.neglect_entrainment)).astype(np.int32))
    flux_owner = np.ascontiguousarray(
        (combine & execution.neglect_entrainment).astype(np.int32))
    selector = np.ascontiguousarray(combine.astype(np.int32))
    owners = [
        _array(exchange_owner, (ng,), "exchange owner", integer=True),
        _array(flux_owner, (ng,), "flux owner", integer=True),
        _array(selector, (ng,), "selector count", integer=True),
    ]
    expected = arrays[1] if execution.neglect_entrainment else arrays[0]
    if (not np.array_equal(arrays[2], expected) or
            np.any(arrays[2][~combine] != 0.0) or
            np.any(arrays[3][~combine] != 0.0) or
            np.any(arrays[4][~combine] != 0.0) or
            np.any(owners[0]*owners[1] != 0) or
            not np.array_equal(owners[0]+owners[1], owners[2])):
        raise SuspendedExnerV93Error(
            "v93 suspended selector branch or owner counts differ")
    integrated = np.sum(arrays[2]*area[None, :], axis=1)
    ledger = arrays[3]+np.sum(arrays[4], axis=1)
    if np.any(np.abs(integrated-ledger) >
              _ledger_tolerance(integrated, ledger)):
        raise SuspendedExnerV93Error(
            "v93 selected suspended-bed ledger did not close")
    for owner in owners:
        owner.setflags(write=False)
    zero_mode = (
        execution.suspended_vector_mode == SUSPENDED_VECTOR_EXACT_ZERO_V93)
    topology_count = None
    materialization_count = None
    selector_abi = TOPOLOGY_SELECTOR_ABI_ID
    if zero_mode:
        if (np.any(arrays[1] != 0.0) or np.any(arrays[2] != 0.0) or
                np.any(arrays[4] != 0.0)):
            raise SuspendedExnerV93Error(
                "v93 exact-zero vector mode acquired suspended bed work")
        topology_count = _array(
            np.zeros(ng, dtype=np.int32), (ng,),
            "topology divergence kernel count", integer=True)
        materialization_count = _array(
            selector, (ng,), "zero-vector materialization count",
            integer=True)
        selector_abi = ZERO_VECTOR_SELECTOR_ABI_ID_V93
    placeholder = SuspendedBedSelectionReceiptV93(
        canonical_selector_abi_id=selector_abi,
        execution_fingerprint=execution.fingerprint,
        parent_state_sha256=parent.state_sha256,
        accepted_v92_trial_sha256=v92_trial.trial_sha256,
        accepted_hydraulic_receipt_sha256=
            v92_trial.hydraulic_receipt.receipt_sha256,
        accepted_hydraulic_face_receipt_sha256=
            v92_trial.hydraulic_face_receipt.receipt_sha256,
        exchange_bed_delta_kg_m2=arrays[0],
        flux_divergence_bed_delta_kg_m2=arrays[1],
        selected_bed_delta_kg_m2=arrays[2],
        selected_exchange_ledger_kg=arrays[3],
        selected_boundary_ledger_kg=arrays[4],
        excluded_water_exchange_kg=excluded,
        exchange_owner_count=owners[0],
        flux_divergence_owner_count=owners[1],
        selector_application_count=owners[2],
        backend=backend, receipt_sha256="",
        topology_divergence_kernel_count=topology_count,
        zero_vector_materialization_count=materialization_count)
    return SuspendedBedSelectionReceiptV93(
        **{**placeholder.__dict__,
           "receipt_sha256": _selector_token(placeholder)})


def select_suspended_bed_cpu_v93(
    execution: SuspendedExnerExecutionV93,
    parent: "SuspendedExnerStateV93",
    parent_hles: HlesProducerStateV90,
    handoff: CurrentClosureHandoffV90,
    v92_trial: TransportedFluidTrialV92,
) -> SuspendedBedSelectionReceiptV93:
    validate_suspended_exner_state_v93(parent, execution)
    _validate_v92_trial(execution.base_execution, parent.base_state,
                        parent_hles, handoff, v92_trial)
    if v92_trial.backend != "cpu_oracle":
        raise SuspendedExnerV93Error(
            "v93 CPU selector requires the exact CPU v92 trial")
    values = _suspended_components_cpu(execution, parent, v92_trial)
    return _wrap_suspended_selection_v93(
        execution, parent, v92_trial,
        exchange_delta=values[0], flux_delta=values[1],
        selected_delta=values[2], selected_exchange_ledger=values[3],
        selected_boundary_ledger=values[4], backend="cpu_oracle")


class Q3ExnerProviderV93(Protocol):
    """Identity-held Q3 provider protocol; receipts alone are never inputs."""

    provider_kind: str
    backend: str
    execution_fingerprint: str
    binding_sha256: str
    topology_id: str
    boundary_names: tuple[str, ...]
    ordered_fraction_ids: tuple[str, ...]
    state_sha256: str
    accepted_updates: int
    time_s: float
    authoritative_bed_mass_kg_m2: np.ndarray


def _provider_value(provider: object, name: str) -> Any:
    if not hasattr(provider, name):
        raise SuspendedExnerV93Error(
            f"v93 Q3 provider does not expose authenticated {name}")
    return getattr(provider, name)


def _validate_provider_identity(
    execution: SuspendedExnerExecutionV93,
    parent: "SuspendedExnerStateV93 | None",
    provider: Q3ExnerProviderV93,
) -> None:
    # A protocol shape and a matching public string are not provenance.  Limit
    # admission to the two concrete, reviewed adapters so a caller cannot mint
    # its own receipt-producing object and borrow an accepted provider_kind.
    # The CUDA import stays lazy: importing the CPU coordinator must not load a
    # CUDA runtime or turn unavailable CUDA into a CPU fallback.
    if execution.backend == "cpu":
        from .q3_mixed_route_exner_adapter_v93 import (
            CPUQ3V84MixedExnerAdapterV93,
        )
        from .active_suspended_control_v94 import (
            CPUQ3V83ActiveExnerAdapterV94,
        )
        from .q3_wetdry_adapter_cpu_v97 import (
            CPUQ3V83ActiveWetDryExnerAdapterV97,
        )
        accepted_types: dict[type[object], str] = {
            CPUQ3V84MixedExnerAdapterV93:
                ACCEPTED_Q3_PROVIDER_KIND["cpu"],
            CPUQ3V83ActiveExnerAdapterV94:
                ACTIVE_Q3_PROVIDER_KIND_V94["cpu"],
            CPUQ3V83ActiveWetDryExnerAdapterV97:
                ACTIVE_Q3_PROVIDER_KIND_V94["cpu"],
        }
    elif execution.backend == "implicit_fwave_midpoint_gpu":
        try:
            from braids_gpu.q3_exner_adapter_v93 import (
                ResidentQ3V84MixedExnerAdapterV93,
            )
            from braids_gpu.active_suspended_control_v94 import (
                ResidentQ3V83ActiveExnerAdapterV94,
            )
            from braids_gpu.active_suspended_wetdry_v97 import (
                ResidentQ3V83ActiveWetDryExnerAdapterV97,
            )
        except Exception as error:
            raise SuspendedExnerV93Error(
                "v93 actual-CUDA Q3 provider capability is unavailable") from error
        accepted_types = {
            ResidentQ3V84MixedExnerAdapterV93:
                ACCEPTED_Q3_PROVIDER_KIND["implicit_fwave_midpoint_gpu"],
            ResidentQ3V83ActiveExnerAdapterV94:
                ACTIVE_Q3_PROVIDER_KIND_V94[
                    "implicit_fwave_midpoint_gpu"],
            ResidentQ3V83ActiveWetDryExnerAdapterV97:
                ACTIVE_Q3_PROVIDER_KIND_V94[
                    "implicit_fwave_midpoint_gpu"],
        }
    else:  # execution construction is already fail-closed; keep this local too.
        raise SuspendedExnerV93Error("v93 Q3 provider backend is unsupported")
    if type(provider) not in accepted_types:
        raise SuspendedExnerV93Error(
            "v93 Q3 provider lacks the exact admitted capability identity")
    expected_backend = ("cpu_oracle" if execution.backend == "cpu" else
                        "actual_cuda")
    provider_kind = _provider_value(provider, "provider_kind")
    if not isinstance(provider_kind, str) or not provider_kind:
        raise SuspendedExnerV93Error(
            "v93 Q3 provider kind must be a nonempty string")
    if provider_kind != accepted_types[type(provider)]:
        raise SuspendedExnerV93Error(
            "v93 Q3 provider kind is not admitted for this backend")
    actual = (
        _provider_value(provider, "backend"),
        _provider_value(provider, "execution_fingerprint"),
        _provider_value(provider, "topology_id"),
        _provider_value(provider, "binding_sha256"),
        tuple(_provider_value(provider, "boundary_names")),
        tuple(_provider_value(provider, "ordered_fraction_ids")),
    )
    expected = (
        expected_backend, execution.q3_execution_fingerprint,
        execution.q3_topology_id, execution.q3_binding_sha256,
        execution.q3_boundary_names, execution.ordered_fraction_ids,
    )
    if actual != expected:
        raise SuspendedExnerV93Error(
            "v93 Q3 provider backend, mesh, boundary, or fraction identity differs")
    _sha(_provider_value(provider, "state_sha256"), "Q3 provider state")
    if parent is not None:
        if (provider_kind != parent.q3_provider_kind or
                _provider_value(provider, "state_sha256") !=
                parent.q3_provider_state_sha256 or
                int(_provider_value(provider, "accepted_updates")) !=
                parent.accepted_updates or
                float(_provider_value(provider, "time_s")) != parent.time_s or
                not np.array_equal(
                    np.asarray(_provider_value(
                        provider, "authoritative_bed_mass_kg_m2")),
                    parent.bed_mass_kg_m2)):
            raise SuspendedExnerV93Error(
                "v93 Q3 provider is not on the authoritative parent bed/clock")


@dataclass(frozen=True)
class SuspendedExnerStateV93:
    execution_fingerprint: str
    base_state: TransportedFluidStateV92
    q3_provider_kind: str
    q3_provider_state_sha256: str
    bed_mass_kg_m2: np.ndarray
    reference_bed_integral_kg: np.ndarray
    suspended_exchange_ledger_kg: np.ndarray
    suspended_boundary_ledger_kg: np.ndarray
    excluded_water_exchange_kg: np.ndarray
    q3_named_boundary_positive_in_kg: np.ndarray
    q3_named_boundary_absolute_kg: np.ndarray
    q3_depletion_roundoff_positive_in_kg: np.ndarray
    accepted_exner_chain_sha256: str
    state_sha256: str
    wetdry_bed_ledger_kg: np.ndarray | None = None

    @property
    def accepted_updates(self) -> int:
        return self.base_state.accepted_updates

    @property
    def time_s(self) -> float:
        return self.base_state.time_s


def _state_token(value: SuspendedExnerStateV93) -> str:
    payload = {
        "execution_fingerprint": value.execution_fingerprint,
        "base_v92_state_sha256": value.base_state.state_sha256,
        "q3_provider_kind": value.q3_provider_kind,
        "q3_provider_state_sha256": value.q3_provider_state_sha256,
        "accepted_exner_chain_sha256": value.accepted_exner_chain_sha256,
        "arrays_sha256": _array_digest(
            value.bed_mass_kg_m2, value.reference_bed_integral_kg,
            value.suspended_exchange_ledger_kg,
            value.suspended_boundary_ledger_kg,
            value.excluded_water_exchange_kg,
            value.q3_named_boundary_positive_in_kg,
            value.q3_named_boundary_absolute_kg,
            value.q3_depletion_roundoff_positive_in_kg),
    }
    if value.wetdry_bed_ledger_kg is not None:
        payload["wetdry_bed_ledger_sha256"] = _array_digest(
            value.wetdry_bed_ledger_kg)
    return _fingerprint(payload)


def _v92_bed(state: TransportedFluidStateV92) -> np.ndarray:
    return np.asarray(state.suspended_state.bed_mass_kg_m2)


def _wrap_state(
    execution: SuspendedExnerExecutionV93,
    base_state: TransportedFluidStateV92, *,
    q3_provider_kind: str, q3_provider_state_sha256: str,
    bed_mass_kg_m2: object,
    reference_bed_integral_kg: object,
    suspended_exchange_ledger_kg: object,
    suspended_boundary_ledger_kg: object,
    excluded_water_exchange_kg: object,
    q3_named_boundary_positive_in_kg: object,
    q3_named_boundary_absolute_kg: object,
    q3_depletion_roundoff_positive_in_kg: object,
    accepted_exner_chain_sha256: str,
    wetdry_bed_ledger_kg: object | None = None,
) -> SuspendedExnerStateV93:
    execution_document_v93(execution)
    _validate_v92_state(base_state, execution.base_execution)
    ng, nc = execution.fraction_count, execution.cell_count
    nsb = (execution.base_execution.base_execution.base_execution.
           base_execution.boundary_count)
    nqb = len(execution.q3_boundary_names)
    if not isinstance(q3_provider_kind, str) or not q3_provider_kind:
        raise SuspendedExnerV93Error("v93 Q3 provider kind is missing")
    provider_state = _sha(q3_provider_state_sha256, "Q3 provider state")
    bed = _array(bed_mass_kg_m2, (ng, nc), "authoritative bed mass",
                 nonnegative=True)
    if not np.array_equal(_v92_bed(base_state), bed):
        raise SuspendedExnerV93Error(
            "v93 nested v92 bed differs from the authoritative bed")
    reference = _array(reference_bed_integral_kg, (ng,),
                       "reference bed integral")
    exchange = _array(suspended_exchange_ledger_kg, (ng,),
                      "suspended exchange ledger")
    suspended_boundary = _array(
        suspended_boundary_ledger_kg, (ng, nsb),
        "suspended named-boundary ledger")
    excluded = _array(excluded_water_exchange_kg, (ng,),
                      "excluded water exchange")
    q3_named = _array(q3_named_boundary_positive_in_kg, (ng, nqb),
                      "Q3 named-boundary positive-in ledger")
    q3_absolute = _array(q3_named_boundary_absolute_kg, (ng, nqb),
                         "Q3 named-boundary absolute ledger",
                         nonnegative=True)
    depletion = _array(q3_depletion_roundoff_positive_in_kg, (ng,),
                       "Q3 depletion-roundoff ledger")
    moving = _moving_wetdry(execution)
    if moving:
        wetdry = _array(wetdry_bed_ledger_kg, (ng,),
                        "wet/dry bed-transfer ledger", nonnegative=True)
    elif wetdry_bed_ledger_kg is not None:
        raise SuspendedExnerV93Error(
            "v93 historical state acquired a v97 wet/dry bed ledger")
    else:
        wetdry = None
    if np.any(np.abs(q3_named) > q3_absolute +
              _ledger_tolerance(q3_named, q3_absolute)):
        raise SuspendedExnerV93Error(
            "v93 Q3 signed named ledger exceeds its absolute diagnostic")
    area = (execution.base_execution.base_execution.base_execution.
            base_execution.cell_area_m2)
    current = np.sum(bed*area[None, :], axis=1)
    expected = (reference+exchange+np.sum(suspended_boundary, axis=1)+
                np.sum(q3_named, axis=1)+depletion+
                (wetdry if wetdry is not None else 0.0))
    if np.any(np.abs(current-expected) >
              _ledger_tolerance(current, expected)):
        raise SuspendedExnerV93Error(
            "v93 authoritative bed ledger did not close")
    combine = np.asarray(execution.routes) == ROUTE_COMBINE
    if (np.any(exchange[~combine] != 0.0) or
            np.any(suspended_boundary[~combine] != 0.0) or
            np.any(excluded[~combine] != 0.0) or
            (execution.neglect_entrainment and np.any(exchange != 0.0)) or
            (not execution.neglect_entrainment and
             (np.any(suspended_boundary != 0.0) or
              np.any(excluded != 0.0)))):
        raise SuspendedExnerV93Error(
            "v93 suspended selector ledgers differ from its branch")
    inactive = np.asarray(execution.routes) == ROUTE_OFF
    if (np.any(q3_named[inactive] != 0.0) or
            np.any(q3_absolute[inactive] != 0.0) or
            np.any(depletion[inactive] != 0.0)):
        raise SuspendedExnerV93Error("v93 OFF route acquired Q3 ledger work")
    chain = _sha(accepted_exner_chain_sha256, "accepted Exner chain")
    placeholder = SuspendedExnerStateV93(
        execution.fingerprint, base_state, q3_provider_kind, provider_state,
        bed, reference, exchange, suspended_boundary, excluded, q3_named,
        q3_absolute, depletion, chain, "", wetdry)
    return SuspendedExnerStateV93(
        **{**placeholder.__dict__, "state_sha256": _state_token(placeholder)})


def resolve_suspended_exner_state_v93(
    execution: SuspendedExnerExecutionV93, *,
    base_state: TransportedFluidStateV92,
    q3_provider: Q3ExnerProviderV93,
    bed_mass_kg_m2: object | None = None,
) -> SuspendedExnerStateV93:
    _validate_v92_state(base_state, execution.base_execution)
    if base_state.accepted_updates != 0:
        raise SuspendedExnerV93Error("v93 fresh state must start at update zero")
    _validate_provider_identity(execution, None, q3_provider)
    bed = _array(
        _v92_bed(base_state) if bed_mass_kg_m2 is None else bed_mass_kg_m2,
        (execution.fraction_count, execution.cell_count),
        "initial authoritative bed", nonnegative=True)
    if (not np.array_equal(_v92_bed(base_state), bed) or
            not np.array_equal(np.asarray(_provider_value(
                q3_provider, "authoritative_bed_mass_kg_m2")), bed) or
            int(_provider_value(q3_provider, "accepted_updates")) != 0 or
            float(_provider_value(q3_provider, "time_s")) !=
                base_state.time_s):
        raise SuspendedExnerV93Error(
            "v93 initial v92/Q3 beds or clocks differ")
    area = (execution.base_execution.base_execution.base_execution.
            base_execution.cell_area_m2)
    reference = np.sum(bed*area[None, :], axis=1)
    chain = _fingerprint({
        "scope_id": SCOPE_ID,
        "execution_fingerprint": execution.fingerprint,
        "initial_v92_state_sha256": base_state.state_sha256,
        "initial_q3_state_sha256": _provider_value(
            q3_provider, "state_sha256"),
        "initial_bed_sha256": _array_digest(bed),
    })
    ng = execution.fraction_count
    return _wrap_state(
        execution, base_state,
        q3_provider_kind=_provider_value(q3_provider, "provider_kind"),
        q3_provider_state_sha256=_provider_value(q3_provider, "state_sha256"),
        bed_mass_kg_m2=bed, reference_bed_integral_kg=reference,
        suspended_exchange_ledger_kg=np.zeros(ng),
        suspended_boundary_ledger_kg=np.zeros((ng, (
            execution.base_execution.base_execution.base_execution.
            base_execution.boundary_count))),
        excluded_water_exchange_kg=np.zeros(ng),
        q3_named_boundary_positive_in_kg=np.zeros(
            (ng, len(execution.q3_boundary_names))),
        q3_named_boundary_absolute_kg=np.zeros(
            (ng, len(execution.q3_boundary_names))),
        q3_depletion_roundoff_positive_in_kg=np.zeros(ng),
        accepted_exner_chain_sha256=chain,
        wetdry_bed_ledger_kg=(np.zeros(ng) if _moving_wetdry(execution)
                              else None))


def validate_suspended_exner_state_v93(
    state: SuspendedExnerStateV93,
    execution: SuspendedExnerExecutionV93,
    q3_provider: Q3ExnerProviderV93 | None = None,
) -> None:
    if (not isinstance(state, SuspendedExnerStateV93) or
            state.execution_fingerprint != execution.fingerprint or
            state.state_sha256 != _state_token(state)):
        raise SuspendedExnerV93Error(
            "v93 state is stale, mutated, or execution-mismatched")
    rebuilt = _wrap_state(
        execution, state.base_state,
        q3_provider_kind=state.q3_provider_kind,
        q3_provider_state_sha256=state.q3_provider_state_sha256,
        bed_mass_kg_m2=state.bed_mass_kg_m2,
        reference_bed_integral_kg=state.reference_bed_integral_kg,
        suspended_exchange_ledger_kg=state.suspended_exchange_ledger_kg,
        suspended_boundary_ledger_kg=state.suspended_boundary_ledger_kg,
        excluded_water_exchange_kg=state.excluded_water_exchange_kg,
        q3_named_boundary_positive_in_kg=
            state.q3_named_boundary_positive_in_kg,
        q3_named_boundary_absolute_kg=state.q3_named_boundary_absolute_kg,
        q3_depletion_roundoff_positive_in_kg=
            state.q3_depletion_roundoff_positive_in_kg,
        accepted_exner_chain_sha256=state.accepted_exner_chain_sha256,
        wetdry_bed_ledger_kg=state.wetdry_bed_ledger_kg)
    if rebuilt.state_sha256 != state.state_sha256:
        raise SuspendedExnerV93Error("v93 rebuilt state differs")
    if q3_provider is not None:
        _validate_provider_identity(execution, state, q3_provider)


@dataclass(frozen=True)
class ExnerOwnerCountsV93:
    suspended_selector_count: np.ndarray
    q3_constitutive_producer_count: np.ndarray
    q3_face_exner_count: np.ndarray
    additional_scalar_lifecycle_count: np.ndarray
    authoritative_exner_count: np.ndarray
    duplicate_exner_owner_count: np.ndarray
    wetdry_transfer_owner_count: np.ndarray | None = None


@dataclass(frozen=True)
class SuspendedExnerTrialV93:
    execution_fingerprint: str
    parent_state_sha256: str
    base_v92_trial: TransportedFluidTrialV92
    suspended_selection: SuspendedBedSelectionReceiptV93
    q3_provider: Q3ExnerProviderV93
    q3_parent_checkpoint: Mapping[str, object]
    q3_open_trial: object
    q3_prepared_commit: object
    q3_receipt: object
    v92_bed_projection: V92BedProjectionReceiptV93
    authoritative_bed_delta_kg_m2: np.ndarray
    candidate: SuspendedExnerStateV93
    owner_counts: ExnerOwnerCountsV93
    backend: str
    trial_sha256: str


def _prepared_token(value: object) -> str:
    for name in ("preparation_sha256", "prepared_sha256"):
        if hasattr(value, name):
            return _sha(getattr(value, name), "Q3 prepared commit")
    raise SuspendedExnerV93Error(
        "v93 Q3 provider prepared commit lacks an authenticated token")


def _trial_token(value: SuspendedExnerTrialV93) -> str:
    payload = {
        "execution_fingerprint": value.execution_fingerprint,
        "parent_state_sha256": value.parent_state_sha256,
        "base_v92_trial_sha256": value.base_v92_trial.trial_sha256,
        "suspended_selection_sha256":
            value.suspended_selection.receipt_sha256,
        "q3_parent_checkpoint_sha256":
            _fingerprint(value.q3_parent_checkpoint),
        "q3_receipt_sha256": value.q3_receipt.receipt_sha256,
        "q3_prepared_sha256": _prepared_token(value.q3_prepared_commit),
        "v92_bed_projection_sha256":
            value.v92_bed_projection.receipt_sha256,
        "authoritative_bed_delta_sha256":
            _array_digest(value.authoritative_bed_delta_kg_m2),
        "candidate_state_sha256": value.candidate.state_sha256,
        "owner_count_sha256": _array_digest(
            value.owner_counts.suspended_selector_count,
            value.owner_counts.q3_constitutive_producer_count,
            value.owner_counts.q3_face_exner_count,
            value.owner_counts.additional_scalar_lifecycle_count,
            value.owner_counts.authoritative_exner_count,
            value.owner_counts.duplicate_exner_owner_count),
        "backend": value.backend,
    }
    if value.owner_counts.wetdry_transfer_owner_count is not None:
        payload["wetdry_transfer_owner_count_sha256"] = _array_digest(
            value.owner_counts.wetdry_transfer_owner_count)
    return _fingerprint(payload)


def _validate_selector_receipt(
    execution: SuspendedExnerExecutionV93,
    parent: SuspendedExnerStateV93,
    v92_trial: TransportedFluidTrialV92,
    receipt: SuspendedBedSelectionReceiptV93,
) -> None:
    zero_mode = (
        execution.suspended_vector_mode == SUSPENDED_VECTOR_EXACT_ZERO_V93)
    expected_abi = (ZERO_VECTOR_SELECTOR_ABI_ID_V93 if zero_mode else
                    TOPOLOGY_SELECTOR_ABI_ID)
    common = (
        expected_abi, execution.fingerprint,
        parent.state_sha256, v92_trial.trial_sha256,
        v92_trial.hydraulic_receipt.receipt_sha256,
        v92_trial.hydraulic_face_receipt.receipt_sha256,
    )
    actual = (
        receipt.canonical_selector_abi_id, receipt.execution_fingerprint,
        receipt.parent_state_sha256, receipt.accepted_v92_trial_sha256,
        receipt.accepted_hydraulic_receipt_sha256,
        receipt.accepted_hydraulic_face_receipt_sha256,
    )
    if actual != common or receipt.receipt_sha256 != _selector_token(receipt):
        raise SuspendedExnerV93Error(
            "v93 canonical v86 selector receipt is stale or cross-trial")
    if zero_mode:
        ng = execution.fraction_count
        combine = np.asarray(execution.routes) == ROUTE_COMBINE
        topology = _array(
            receipt.topology_divergence_kernel_count, (ng,),
            "topology divergence kernel count", integer=True)
        materialization = _array(
            receipt.zero_vector_materialization_count, (ng,),
            "zero-vector materialization count", integer=True)
        if (np.any(receipt.flux_divergence_bed_delta_kg_m2 != 0.0) or
                np.any(receipt.selected_bed_delta_kg_m2 != 0.0) or
                np.any(receipt.selected_boundary_ledger_kg != 0.0) or
                np.any(topology != 0) or
                not np.array_equal(materialization,
                                   combine.astype(np.int32)) or
                not np.array_equal(receipt.flux_divergence_owner_count,
                                   combine.astype(np.int32))):
            raise SuspendedExnerV93Error(
                "v93 exact-zero vector receipt or owner counts differ")
    elif (receipt.topology_divergence_kernel_count is not None or
          receipt.zero_vector_materialization_count is not None):
        raise SuspendedExnerV93Error(
            "v93 historical selector acquired additive v94b counts")


def _validate_q3_receipt(
    execution: SuspendedExnerExecutionV93,
    parent: SuspendedExnerStateV93,
    v92_trial: TransportedFluidTrialV92,
    provider: Q3ExnerProviderV93,
    open_trial: object,
) -> object:
    try:
        receipt = provider.validate_open(
            open_trial, outer_parent=parent, v92_trial=v92_trial)
    except Exception as error:
        raise SuspendedExnerV93Error(
            "v93 live Q3 provider validation failed") from error
    ng, nc, nb = (execution.fraction_count, execution.cell_count,
                  len(execution.q3_boundary_names))
    expected_backend = ("cpu_oracle" if execution.backend == "cpu" else
                        "actual_cuda")
    required = (
        "receipt_sha256", "execution_fingerprint", "parent_state_sha256",
        "accepted_v92_trial_sha256", "accepted_hydraulic_receipt_sha256",
        "accepted_hydraulic_face_receipt_sha256",
        "q3_execution_fingerprint", "q3_binding_sha256", "q3_topology_id",
        "dt_s", "morphological_factor", "boundary_names", "routes",
        "bed_delta_kg_m2", "named_boundary_positive_in_kg",
        "named_boundary_absolute_kg", "depletion_roundoff_positive_in_kg",
        "boundary_ledger_kg", "q3_owner_count", "backend",
        "provider_kind", "ordered_fraction_ids",
        "constitutive_producer_count", "face_exner_count",
        "additional_scalar_lifecycle_count",
    )
    if any(not hasattr(receipt, name) for name in required):
        raise SuspendedExnerV93Error(
            "v93 live Q3 receipt inventory is incomplete")
    common = (
        receipt.execution_fingerprint, receipt.parent_state_sha256,
        receipt.accepted_v92_trial_sha256,
        receipt.accepted_hydraulic_receipt_sha256,
        receipt.accepted_hydraulic_face_receipt_sha256,
        receipt.q3_execution_fingerprint, receipt.q3_topology_id,
        receipt.q3_binding_sha256, tuple(receipt.boundary_names),
        tuple(receipt.ordered_fraction_ids), receipt.backend,
        receipt.provider_kind, float(receipt.dt_s),
        float(receipt.morphological_factor),
    )
    expected = (
        execution.fingerprint, parent.state_sha256, v92_trial.trial_sha256,
        v92_trial.hydraulic_receipt.receipt_sha256,
        v92_trial.hydraulic_face_receipt.receipt_sha256,
        execution.q3_execution_fingerprint, execution.q3_topology_id,
        execution.q3_binding_sha256, execution.q3_boundary_names,
        execution.ordered_fraction_ids, expected_backend,
        _provider_value(provider, "provider_kind"),
        float(v92_trial.hydraulic_receipt.dt_s), 1.0,
    )
    if common != expected:
        raise SuspendedExnerV93Error(
            "v93 live Q3 receipt mesh, fraction, timing, or parent differs")
    delta = _array(receipt.bed_delta_kg_m2, (ng, nc), "Q3 bed delta")
    named = _array(receipt.named_boundary_positive_in_kg, (ng, nb),
                   "Q3 named-boundary positive-in ledger")
    absolute = _array(receipt.named_boundary_absolute_kg, (ng, nb),
                      "Q3 named-boundary absolute ledger", nonnegative=True)
    depletion = _array(receipt.depletion_roundoff_positive_in_kg, (ng,),
                       "Q3 depletion-roundoff ledger")
    boundary = _array(receipt.boundary_ledger_kg, (ng,),
                      "Q3 global boundary ledger")
    routes = _array(receipt.routes, (ng,), "Q3 routes", integer=True)
    active = np.asarray(execution.routes) != ROUTE_OFF
    expected_count = active.astype(np.int32)
    owner = _array(receipt.q3_owner_count, (ng,), "Q3 owner count",
                   integer=True)
    producer = _array(receipt.constitutive_producer_count, (ng,),
                      "Q3 constitutive producer count", integer=True)
    face = _array(receipt.face_exner_count, (ng,),
                  "Q3 face Exner count", integer=True)
    extra_scalar = _array(receipt.additional_scalar_lifecycle_count, (ng,),
                          "additional scalar lifecycle count", integer=True)
    area = (execution.base_execution.base_execution.base_execution.
            base_execution.cell_area_m2)
    integrated = np.sum(delta*area[None, :], axis=1)
    expected_boundary = np.sum(named, axis=1)+depletion
    if (not np.array_equal(routes, execution.routes) or
            not np.array_equal(owner, expected_count) or
            not np.array_equal(producer, expected_count) or
            not np.array_equal(face, expected_count) or
            np.any(extra_scalar != 0) or
            not np.array_equal(boundary, expected_boundary) or
            np.any(np.abs(integrated-boundary) >
                   _ledger_tolerance(integrated, boundary)) or
            np.any(delta[~active] != 0.0) or
            np.any(named[~active] != 0.0) or
            np.any(absolute[~active] != 0.0) or
            np.any(depletion[~active] != 0.0) or
            np.any(np.abs(named) > absolute+
                   _ledger_tolerance(named, absolute))):
        raise SuspendedExnerV93Error(
            "v93 live Q3 receipt ownership or ledger differs")
    _sha(receipt.receipt_sha256, "Q3 receipt")
    return receipt


def _prepared_state(value: object) -> object:
    if not hasattr(value, "rebound_state"):
        raise SuspendedExnerV93Error(
            "v93 Q3 prepared commit lacks a rebound state")
    return value.rebound_state


def _rebound_digest(value: object) -> str:
    state = _prepared_state(value)
    if not hasattr(state, "state_sha256"):
        raise SuspendedExnerV93Error(
            "v93 Q3 rebound state lacks an authenticated digest")
    return _sha(state.state_sha256, "Q3 rebound state")


def _authoritative_candidate_bed_v93(
    parent_bed_mass_kg_m2: object,
    authoritative_bed_delta_kg_m2: object, *,
    candidate_bed_mass_kg_m2: object | None = None,
) -> np.ndarray:
    """Form the sole Exner candidate without an unledgered positivity repair.

    Even a floating-point-size negative value is rejected.  Silently clipping
    such a value would create bed mass outside both the suspended and Q3
    ledgers, so positivity limiting belongs upstream in the authenticated Q3
    availability owner rather than in this composition step.
    """
    parent = np.asarray(parent_bed_mass_kg_m2, dtype=np.float64)
    delta = np.asarray(authoritative_bed_delta_kg_m2, dtype=np.float64)
    if parent.shape != delta.shape or parent.ndim != 2:
        raise SuspendedExnerV93Error(
            "v93 parent bed and authoritative delta shapes differ")
    expected = parent+delta
    candidate = np.array(
        expected if candidate_bed_mass_kg_m2 is None else
        candidate_bed_mass_kg_m2, dtype=np.float64, order="C", copy=True)
    if candidate.shape != expected.shape or not np.array_equal(candidate, expected):
        raise SuspendedExnerV93Error(
            "v93 authoritative candidate differs from parent plus delta")
    if not np.isfinite(candidate).all():
        raise SuspendedExnerV93Error(
            "v93 authoritative candidate must be finite")
    if np.any(candidate < 0.0):
        raise SuspendedExnerV93Error(
            "v93 Exner candidate exceeds available authoritative bed mass")
    candidate.setflags(write=False)
    return candidate


def _build_trial_from_live_provider(
    execution: SuspendedExnerExecutionV93,
    parent: SuspendedExnerStateV93,
    v92_trial: TransportedFluidTrialV92,
    suspended: SuspendedBedSelectionReceiptV93,
    provider: Q3ExnerProviderV93,
    parent_q3_checkpoint: Mapping[str, object],
    open_trial: object, *, backend: str,
    authoritative_bed_delta_kg_m2: object | None = None,
    candidate_bed_mass_kg_m2: object | None = None,
) -> SuspendedExnerTrialV93:
    validate_suspended_exner_state_v93(parent, execution, provider)
    if parent.accepted_updates >= MAX_ACCEPTED_UPDATES:
        raise SuspendedExnerV93Error("v93 accepted update limit was reached")
    _validate_selector_receipt(execution, parent, v92_trial, suspended)
    q3 = _validate_q3_receipt(
        execution, parent, v92_trial, provider, open_trial)
    cleanup = np.asarray(
        v92_trial.suspended_trial.base_trial.wetdry_transfer_to_bed_kg_m2)
    if _moving_wetdry(execution):
        cleanup = _array(
            cleanup, (execution.fraction_count, execution.cell_count),
            "wet/dry transfer to bed", nonnegative=True)
        combine = np.asarray(execution.routes) == ROUTE_COMBINE
        if np.any(cleanup[~combine] != 0.0):
            raise SuspendedExnerV93Error(
                "v93 OFF/TOTAL route acquired wet/dry suspended cleanup")
    else:
        if np.any(cleanup != 0.0):
            raise SuspendedExnerV93Error(
                "v93 historical route acquired wet/dry suspended cleanup")
        cleanup = np.zeros_like(q3.bed_delta_kg_m2)
    expected = (suspended.selected_bed_delta_kg_m2+
                q3.bed_delta_kg_m2+cleanup)
    delta = _array(expected if authoritative_bed_delta_kg_m2 is None else
                   authoritative_bed_delta_kg_m2,
                   (execution.fraction_count, execution.cell_count),
                   "authoritative bed delta")
    if not np.array_equal(delta, expected):
        raise SuspendedExnerV93Error(
            "v93 authoritative delta differs from its two source terms")
    candidate_bed = _authoritative_candidate_bed_v93(
        parent.bed_mass_kg_m2, delta,
        candidate_bed_mass_kg_m2=candidate_bed_mass_kg_m2)
    projected, projection_receipt = project_v92_authoritative_bed_v93(
        execution.base_execution, v92_trial.candidate,
        authoritative_bed_mass_kg_m2=candidate_bed)
    validate_v92_bed_projection_receipt_v93(
        projection_receipt, execution=execution.base_execution,
        source=v92_trial.candidate, projected=projected,
        authoritative_bed_mass_kg_m2=candidate_bed)
    try:
        prepared = provider.prepare_rebound(
            open_trial, authoritative_bed_mass_kg_m2=candidate_bed)
        provider.validate_prepared(
            prepared, authoritative_bed_mass_kg_m2=candidate_bed)
    except Exception as error:
        raise SuspendedExnerV93Error(
            "v93 Q3 authoritative-bed rebound failed") from error
    rebound = _prepared_state(prepared)
    rebound_bed = np.asarray(getattr(
        rebound, "authoritative_bed_mass_kg_m2",
        getattr(getattr(rebound, "direct_state", None),
                "bed_mass_kg_m2", np.empty(0))))
    if not np.array_equal(rebound_bed, candidate_bed):
        raise SuspendedExnerV93Error(
            "v93 Q3 rebound bed differs from the authoritative bed")
    chain = _fingerprint({
        "previous": parent.accepted_exner_chain_sha256,
        "v92_trial": v92_trial.trial_sha256,
        "selector": suspended.receipt_sha256,
        "q3": q3.receipt_sha256,
        "projection": projection_receipt.receipt_sha256,
        "delta": _array_digest(delta),
        **({"wetdry_cleanup": _array_digest(cleanup)}
           if _moving_wetdry(execution) else {}),
    })
    area = (execution.base_execution.base_execution.base_execution.
            base_execution.cell_area_m2)
    wetdry_step_ledger = np.sum(cleanup*area[None, :], axis=1)
    candidate = _wrap_state(
        execution, projected,
        q3_provider_kind=_provider_value(provider, "provider_kind"),
        q3_provider_state_sha256=_rebound_digest(prepared),
        bed_mass_kg_m2=candidate_bed,
        reference_bed_integral_kg=parent.reference_bed_integral_kg,
        suspended_exchange_ledger_kg=(
            parent.suspended_exchange_ledger_kg+
            suspended.selected_exchange_ledger_kg),
        suspended_boundary_ledger_kg=(
            parent.suspended_boundary_ledger_kg+
            suspended.selected_boundary_ledger_kg),
        excluded_water_exchange_kg=(
            parent.excluded_water_exchange_kg+
            suspended.excluded_water_exchange_kg),
        q3_named_boundary_positive_in_kg=(
            parent.q3_named_boundary_positive_in_kg+
            q3.named_boundary_positive_in_kg),
        q3_named_boundary_absolute_kg=(
            parent.q3_named_boundary_absolute_kg+
            q3.named_boundary_absolute_kg),
        q3_depletion_roundoff_positive_in_kg=(
            parent.q3_depletion_roundoff_positive_in_kg+
            q3.depletion_roundoff_positive_in_kg),
        accepted_exner_chain_sha256=chain,
        wetdry_bed_ledger_kg=(
            parent.wetdry_bed_ledger_kg+wetdry_step_ledger
            if _moving_wetdry(execution) else None))
    combine = np.asarray(execution.routes) == ROUTE_COMBINE
    active = np.asarray(execution.routes) != ROUTE_OFF
    zero = np.zeros(execution.fraction_count, dtype=np.int32)
    wetdry_owner = (_array(
        combine.astype(np.int32), (execution.fraction_count,),
        "wet/dry transfer owner count", integer=True)
        if _moving_wetdry(execution) else None)
    counts = ExnerOwnerCountsV93(
        _array(combine.astype(np.int32), (execution.fraction_count,),
               "suspended selector count", integer=True),
        _array(q3.constitutive_producer_count, (execution.fraction_count,),
               "Q3 constitutive producer count", integer=True),
        _array(q3.face_exner_count, (execution.fraction_count,),
               "Q3 face Exner count", integer=True),
        _array(q3.additional_scalar_lifecycle_count,
               (execution.fraction_count,),
               "additional scalar lifecycle count", integer=True),
        _array(active.astype(np.int32), (execution.fraction_count,),
               "authoritative Exner count", integer=True),
        _array(zero, (execution.fraction_count,),
               "duplicate Exner owner count", integer=True),
        wetdry_owner)
    placeholder = SuspendedExnerTrialV93(
        execution.fingerprint, parent.state_sha256, v92_trial, suspended,
        provider, parent_q3_checkpoint, open_trial, prepared, q3,
        projection_receipt, delta,
        candidate, counts, backend, "")
    return SuspendedExnerTrialV93(
        **{**placeholder.__dict__, "trial_sha256": _trial_token(placeholder)})


def begin_suspended_exner_cpu_trial_v93(
    execution: SuspendedExnerExecutionV93,
    parent: SuspendedExnerStateV93,
    parent_hles: HlesProducerStateV90,
    handoff: CurrentClosureHandoffV90,
    q3_provider: Q3ExnerProviderV93, *,
    v92_inputs: Mapping[str, object],
    q3_inputs: Mapping[str, object],
    cancelled: Callable[[], bool] = lambda: False,
) -> SuspendedExnerTrialV93:
    if execution.backend != "cpu":
        raise SuspendedExnerV93Error(
            "v93 CPU owner requires the explicit CPU execution")
    if not callable(cancelled):
        raise TypeError("cancelled must be callable")
    if cancelled():
        raise SuspendedExnerV93Error(
            "v93 CPU transaction cancelled before Exner work")
    validate_suspended_exner_state_v93(parent, execution, q3_provider)
    if parent.accepted_updates >= MAX_ACCEPTED_UPDATES:
        raise SuspendedExnerV93Error("v93 accepted update limit was reached")
    v92_trial = begin_transported_fluid_cpu_trial_v92(
        execution.base_execution, parent.base_state, parent_hles, handoff,
        cancelled=cancelled, **dict(v92_inputs))
    suspended = select_suspended_bed_cpu_v93(
        execution, parent, parent_hles, handoff, v92_trial)
    open_trial = None
    parent_q3_checkpoint = q3_provider.checkpoint()
    try:
        open_trial = q3_provider.begin(
            outer_parent=parent, parent_hles=parent_hles, handoff=handoff,
            v92_trial=v92_trial, cancelled=cancelled, **dict(q3_inputs))
        return _build_trial_from_live_provider(
            execution, parent, v92_trial, suspended, q3_provider,
            parent_q3_checkpoint, open_trial, backend="cpu_oracle")
    except Exception:
        if open_trial is not None:
            try:
                q3_provider.rollback(open_trial)
            except Exception as cleanup_error:
                raise RuntimeError(
                    "v93 CPU begin failed and Q3 rollback also failed") \
                    from cleanup_error
        raise


def _validate_trial(
    execution: SuspendedExnerExecutionV93,
    parent: SuspendedExnerStateV93,
    parent_hles: HlesProducerStateV90,
    handoff: CurrentClosureHandoffV90,
    trial: SuspendedExnerTrialV93,
) -> None:
    if not isinstance(trial, SuspendedExnerTrialV93):
        raise SuspendedExnerV93Error("v93 trial type differs")
    validate_suspended_exner_state_v93(
        parent, execution, trial.q3_provider)
    expected_backend = ("cpu_oracle" if execution.backend == "cpu" else
                        "actual_cuda")
    if (trial.execution_fingerprint != execution.fingerprint or
            trial.parent_state_sha256 != parent.state_sha256 or
            trial.trial_sha256 != _trial_token(trial) or
            trial.backend != expected_backend):
        raise SuspendedExnerV93Error(
            "v93 trial is stale, mutated, or execution-mismatched")
    _validate_v92_trial(execution.base_execution, parent.base_state,
                        parent_hles, handoff, trial.base_v92_trial)
    _validate_selector_receipt(
        execution, parent, trial.base_v92_trial, trial.suspended_selection)
    q3 = _validate_q3_receipt(
        execution, parent, trial.base_v92_trial, trial.q3_provider,
        trial.q3_open_trial)
    if q3 is not trial.q3_receipt:
        raise SuspendedExnerV93Error(
            "v93 provider returned a substituted Q3 receipt")
    cleanup = np.asarray(
        trial.base_v92_trial.suspended_trial.base_trial.
        wetdry_transfer_to_bed_kg_m2)
    if not _moving_wetdry(execution):
        cleanup = np.zeros_like(cleanup)
    expected_delta = (trial.suspended_selection.selected_bed_delta_kg_m2+
                      q3.bed_delta_kg_m2+cleanup)
    if (not np.array_equal(trial.authoritative_bed_delta_kg_m2,
                           expected_delta) or
            not np.array_equal(trial.candidate.bed_mass_kg_m2,
                               parent.bed_mass_kg_m2+expected_delta)):
        raise SuspendedExnerV93Error(
            "v93 authoritative wet/dry/Q3 Exner composition differs")
    combine = np.asarray(execution.routes) == ROUTE_COMBINE
    expected_wetdry_owner = (combine.astype(np.int32)
                             if _moving_wetdry(execution) else None)
    if expected_wetdry_owner is None:
        if trial.owner_counts.wetdry_transfer_owner_count is not None:
            raise SuspendedExnerV93Error(
                "v93 historical trial acquired a wet/dry transfer owner")
    elif not np.array_equal(
            trial.owner_counts.wetdry_transfer_owner_count,
            expected_wetdry_owner):
        raise SuspendedExnerV93Error(
            "v93 wet/dry transfer owner count differs")
    trial.q3_provider.validate_prepared(
        trial.q3_prepared_commit,
        authoritative_bed_mass_kg_m2=trial.candidate.bed_mass_kg_m2)
    validate_v92_bed_projection_receipt_v93(
        trial.v92_bed_projection, execution=execution.base_execution,
        source=trial.base_v92_trial.candidate,
        projected=trial.candidate.base_state,
        authoritative_bed_mass_kg_m2=trial.candidate.bed_mass_kg_m2)
    validate_suspended_exner_state_v93(trial.candidate, execution)


def commit_suspended_exner_trial_v93(
    execution: SuspendedExnerExecutionV93,
    parent: SuspendedExnerStateV93,
    parent_hles: HlesProducerStateV90,
    handoff: CurrentClosureHandoffV90,
    trial: SuspendedExnerTrialV93,
) -> SuspendedExnerStateV93:
    _validate_trial(execution, parent, parent_hles, handoff, trial)
    provider = trial.q3_provider
    prepared = trial.q3_prepared_commit
    try:
        rebound = provider.close_commit(prepared)
        if getattr(rebound, "state_sha256", None) != \
                trial.candidate.q3_provider_state_sha256:
            raise SuspendedExnerV93Error(
                "v93 closed Q3 child differs from rebound candidate")
        base = commit_transported_fluid_trial_v92(
            execution.base_execution, parent.base_state, parent_hles,
            handoff, trial.base_v92_trial)
        if base.state_sha256 != trial.v92_bed_projection.source_state_sha256:
            raise SuspendedExnerV93Error(
                "v93 committed v92 source candidate differs")
        # All fallible validation is complete.  Provider publication is the
        # final state assignment; no post-publication operation may throw.
        provider.publish_commit(prepared)
        return trial.candidate
    except Exception as original_error:
        # Failure injection may raise *after* close_commit or publish_commit
        # has mutated its private transaction markers.  Inspect the exact
        # admitted provider and restore the saved committed checkpoint rather
        # than relying on a local flag set only after a successful return.
        try:
            if getattr(provider, "_closed", None) is prepared:
                provider.abort_closed_commit(prepared)
            elif getattr(provider, "_open", None) is trial.q3_open_trial:
                provider.rollback(trial.q3_open_trial)
            if _provider_value(provider, "state_sha256") != \
                    parent.q3_provider_state_sha256:
                provider.restore_checkpoint(trial.q3_parent_checkpoint)
        except Exception as cleanup_error:
            raise RuntimeError(
                "v93 commit failed and Q3 restore also failed") \
                from cleanup_error
        raise original_error


def rollback_suspended_exner_trial_v93(
    execution: SuspendedExnerExecutionV93,
    parent: SuspendedExnerStateV93,
    parent_hles: HlesProducerStateV90,
    handoff: CurrentClosureHandoffV90,
    trial: SuspendedExnerTrialV93,
) -> SuspendedExnerStateV93:
    _validate_trial(execution, parent, parent_hles, handoff, trial)
    q3 = trial.q3_provider.rollback(trial.q3_open_trial)
    if getattr(q3, "state_sha256", None) != parent.q3_provider_state_sha256:
        raise SuspendedExnerV93Error("v93 Q3 rollback changed parent state")
    base = rollback_transported_fluid_trial_v92(
        execution.base_execution, parent.base_state, parent_hles, handoff,
        trial.base_v92_trial)
    if base is not parent.base_state:
        raise SuspendedExnerV93Error("v93 v92 rollback changed parent state")
    validate_suspended_exner_state_v93(parent, execution, trial.q3_provider)
    return parent


def cancel_suspended_exner_trial_v93(*args, **kwargs):
    return rollback_suspended_exner_trial_v93(*args, **kwargs)


def suspended_exner_checkpoint_v93(
    execution: SuspendedExnerExecutionV93,
    state: SuspendedExnerStateV93,
    q3_provider: Q3ExnerProviderV93,
) -> dict:
    validate_suspended_exner_state_v93(state, execution, q3_provider)
    authoritative_bed = {
        "bed_mass_kg_m2": state.bed_mass_kg_m2.tolist(),
        "reference_bed_integral_kg":
            state.reference_bed_integral_kg.tolist(),
        "suspended_exchange_ledger_kg":
            state.suspended_exchange_ledger_kg.tolist(),
        "suspended_boundary_ledger_kg":
            state.suspended_boundary_ledger_kg.tolist(),
        "excluded_water_exchange_kg":
            state.excluded_water_exchange_kg.tolist(),
        "q3_named_boundary_positive_in_kg":
            state.q3_named_boundary_positive_in_kg.tolist(),
        "q3_named_boundary_absolute_kg":
            state.q3_named_boundary_absolute_kg.tolist(),
        "q3_depletion_roundoff_positive_in_kg":
            state.q3_depletion_roundoff_positive_in_kg.tolist(),
        "accepted_exner_chain_sha256":
            state.accepted_exner_chain_sha256,
    }
    if state.wetdry_bed_ledger_kg is not None:
        authoritative_bed["wetdry_bed_ledger_kg"] = \
            state.wetdry_bed_ledger_kg.tolist()
    body = {
        "schema": SCHEMA,
        "scope_id": CHECKPOINT_SCOPE_ID,
        "semantic_id": execution.semantic_id,
        "execution_fingerprint": execution.fingerprint,
        "base_v92_checkpoint": transported_fluid_checkpoint_v92(
            execution.base_execution, state.base_state),
        "q3_provider_kind": state.q3_provider_kind,
        "q3_provider_checkpoint": q3_provider.checkpoint(),
        "authoritative_bed": authoritative_bed,
        "state_sha256": state.state_sha256,
    }
    return {**body, "fingerprint": _fingerprint(body)}


def load_suspended_exner_checkpoint_v93(
    execution: SuspendedExnerExecutionV93,
    document: Mapping[str, object],
    q3_provider: Q3ExnerProviderV93,
) -> SuspendedExnerStateV93:
    expected = {"schema", "scope_id", "semantic_id",
                "execution_fingerprint", "base_v92_checkpoint",
                "q3_provider_kind", "q3_provider_checkpoint",
                "authoritative_bed", "state_sha256", "fingerprint"}
    if not isinstance(document, Mapping) or set(document) != expected:
        raise SuspendedExnerV93Error("v93 checkpoint inventory differs")
    body = {key: value for key, value in document.items()
            if key != "fingerprint"}
    if (document["fingerprint"] != _fingerprint(body) or
            document["schema"] != SCHEMA or
            document["scope_id"] != CHECKPOINT_SCOPE_ID or
            document["semantic_id"] != execution.semantic_id or
            document["execution_fingerprint"] != execution.fingerprint or
            document["q3_provider_kind"] !=
                _provider_value(q3_provider, "provider_kind")):
        raise SuspendedExnerV93Error("v93 checkpoint identity differs")
    fields = document["authoritative_bed"]
    keys = {"bed_mass_kg_m2", "reference_bed_integral_kg",
            "suspended_exchange_ledger_kg",
            "suspended_boundary_ledger_kg", "excluded_water_exchange_kg",
            "q3_named_boundary_positive_in_kg",
            "q3_named_boundary_absolute_kg",
            "q3_depletion_roundoff_positive_in_kg",
            "accepted_exner_chain_sha256"}
    if _moving_wetdry(execution):
        keys.add("wetdry_bed_ledger_kg")
    if not isinstance(fields, Mapping) or set(fields) != keys:
        raise SuspendedExnerV93Error("v93 checkpoint bed inventory differs")
    previous_q3 = q3_provider.checkpoint()
    try:
        base = load_transported_fluid_checkpoint_v92(
            execution.base_execution, document["base_v92_checkpoint"])
        q3_provider.restore_checkpoint(document["q3_provider_checkpoint"])
        state = _wrap_state(
            execution, base,
            q3_provider_kind=document["q3_provider_kind"],
            q3_provider_state_sha256=_provider_value(
                q3_provider, "state_sha256"),
            bed_mass_kg_m2=fields["bed_mass_kg_m2"],
            reference_bed_integral_kg=fields["reference_bed_integral_kg"],
            suspended_exchange_ledger_kg=
                fields["suspended_exchange_ledger_kg"],
            suspended_boundary_ledger_kg=
                fields["suspended_boundary_ledger_kg"],
            excluded_water_exchange_kg=
                fields["excluded_water_exchange_kg"],
            q3_named_boundary_positive_in_kg=
                fields["q3_named_boundary_positive_in_kg"],
            q3_named_boundary_absolute_kg=
                fields["q3_named_boundary_absolute_kg"],
            q3_depletion_roundoff_positive_in_kg=
                fields["q3_depletion_roundoff_positive_in_kg"],
            accepted_exner_chain_sha256=
                fields["accepted_exner_chain_sha256"],
            wetdry_bed_ledger_kg=fields.get("wetdry_bed_ledger_kg"))
        if state.state_sha256 != document["state_sha256"]:
            raise SuspendedExnerV93Error(
                "v93 checkpoint state digest differs")
        validate_suspended_exner_state_v93(state, execution, q3_provider)
        return state
    except Exception:
        q3_provider.restore_checkpoint(previous_q3)
        raise


def method_identity_v93() -> dict:
    return {
        "scope_id": SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "integrator": INTEGRATOR,
        "accepted_update_limit": MAX_ACCEPTED_UPDATES,
        "morphological_factor": 1.0,
        "neglect_entrainment": {
            False: BED_BALANCE_EXCHANGE,
            True: BED_BALANCE_FLUX_DIVERGENCE,
        },
        "routes": {"off": ROUTE_OFF, "total": ROUTE_TOTAL,
                   "combine": ROUTE_COMBINE},
        "one_authoritative_exner_owner": True,
        "canonical_v86_selector_abi": TOPOLOGY_SELECTOR_ABI_ID,
        "water_column_exchange_unchanged_by_selector": True,
        "excluded_water_exchange_retained": True,
        "q3_input": "identity_held_live_provider_only",
        "q3_provider_families": {
            "portable_cpu_oracle_v75_all_total": "non_admission",
            "real_v84_producer_q4_face_cpu": "acceptance_provider",
            "real_v84_producer_q4_face_actual_cuda":
                "bounded_acceptance_provider_not_product",
            "actual_cuda_v75_live": "non_admission",
            "real_v83_active_iopsus_q4_face_cpu_v94":
                "bounded_acceptance_provider_not_product",
            "real_v83_active_iopsus_q4_face_actual_cuda_v94":
                "bounded_acceptance_provider_not_product",
        },
        "q3_scope": {
            "downhill": "exact_zero_only",
            "wet_dry": "fully_wet_fixed_grid_only",
            "boundaries": "closed_or_owner_free_export_only",
            "fraction_count": 3,
            "morphological_factor": 1.0,
            "maximum_accepted_updates": MAX_ACCEPTED_UPDATES,
        },
        "fixed_grid": True,
        "moving_wet_dry": False,
        "automatic_cpu_fallback": False,
        "long_run_claimed": False,
        "product_admission": False,
        "delft3d_source_modified": False,
    }


def method_identity_zero_vector_v93() -> dict:
    """Return the additive, source-legal SusCor=false selector identity."""
    return {
        "scope_id": ZERO_VECTOR_SCOPE_ID_V93,
        "semantic_id": ZERO_VECTOR_SEMANTIC_ID_V93,
        "integrator": INTEGRATOR,
        "suspended_vector_mode": SUSPENDED_VECTOR_EXACT_ZERO_V93,
        "neglect_entrainment": True,
        "raw_v92_scalar_lifecycle": "unchanged_and_may_be_nonzero",
        "effective_suspended_vector": "exact_zero",
        "topology_divergence_kernel_count_per_combine_fraction": 0,
        "zero_vector_materialization_count_per_combine_fraction": 1,
        "q3_bed_delta": "already_timestep_integrated_applied_once",
        "maximum_accepted_updates": MAX_ACCEPTED_UPDATES,
        "morphological_factor": 1.0,
        "product_admission": False,
        "delft3d_source_modified": False,
    }


__all__ = [
    "ACCEPTED_Q3_PROVIDER_KIND", "ACTIVE_Q3_PROVIDER_KIND_V94",
    "BED_BALANCE_EXCHANGE",
    "BED_BALANCE_FLUX_DIVERGENCE",
    "CHECKPOINT_SCOPE_ID", "ExnerOwnerCountsV93", "INTEGRATOR",
    "MAX_ACCEPTED_UPDATES", "Q3ExnerProviderV93", "SCHEMA", "SCOPE_ID",
    "SEMANTIC_ID", "SUSPENDED_VECTOR_CONSTRUCTED_V93",
    "SUSPENDED_VECTOR_EXACT_ZERO_V93", "ZERO_VECTOR_SCOPE_ID_V93",
    "ZERO_VECTOR_SELECTOR_ABI_ID_V93", "ZERO_VECTOR_SEMANTIC_ID_V93",
    "SuspendedBedSelectionReceiptV93",
    "SuspendedExnerExecutionV93", "SuspendedExnerStateV93",
    "SuspendedExnerTrialV93", "SuspendedExnerV93Error",
    "begin_suspended_exner_cpu_trial_v93",
    "cancel_suspended_exner_trial_v93",
    "commit_suspended_exner_trial_v93", "execution_document_v93",
    "load_suspended_exner_checkpoint_v93", "method_identity_v93",
    "method_identity_zero_vector_v93",
    "resolve_suspended_exner_execution_v93",
    "resolve_suspended_exner_zero_vector_execution_v93",
    "resolve_suspended_exner_state_v93",
    "rollback_suspended_exner_trial_v93", "select_suspended_bed_cpu_v93",
    "suspended_exner_checkpoint_v93", "validate_suspended_exner_state_v93",
]
