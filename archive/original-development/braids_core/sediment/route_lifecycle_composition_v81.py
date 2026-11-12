"""Authenticated Q4 bridge from the v74 split route to the v78 lifecycle.

The frozen v74 and v78 APIs deliberately use different quantities at their
boundary.  V74 exposes the final equilibrium concentration, after ``Sus`` and
active-fraction scaling, while v78 accepts an equilibrium *transport
capacity* and performs its own speed/depth and active-fraction conversion.
Passing the former as the latter would silently apply the conversion twice.

This additive bridge therefore consumes the authenticated raw v74 producer
rows and constructs the pre-fraction capacity expected by v78.  It also keeps
the v74 handoff as independent ownership evidence and checks that v78
reconstructs the same equilibrium target.  The scope is intentionally small:
three current-only noncohesive lanes (Bijker, TraFrm 19 and TraFrm 20), a
fixed grid, no waves or external secondary-flow rotation, and no more than
four accepted ``implicit_fwave_midpoint`` updates.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Mapping, Sequence

import numpy as np

from .multifraction_suspended_lifecycle_v78 import (
    MAX_ACCEPTED_UPDATES,
    ROLE_CPU_ORACLE,
    ROLE_CUDA,
    MultifractionSuspendedExecutionV78,
    MultifractionSuspendedStateV78,
    MultifractionSuspendedTrialV78,
    begin_multifraction_cpu_oracle_trial_v78,
    cancel_multifraction_trial_v78,
    commit_multifraction_trial_v78,
    execution_document_v78,
    load_multifraction_checkpoint_v78,
    multifraction_checkpoint_v78,
    rollback_multifraction_trial_v78,
    validate_multifraction_suspended_state_v78,
)
from .suspended_route_live_v74 import (
    METHOD_BIJKER_CURRENT,
    METHOD_VAN_THIEL_VAN_RIJN_2008_CURRENT,
    METHOD_XBEACH_SOULSBY_VAN_RIJN_CURRENT,
    ROUTE_COMBINE,
    ROUTE_OFF,
    ROUTE_TOTAL,
    RouteHandoffV74,
    compose_route_handoff_v74,
    load_route_execution_v74,
    validate_route_handoff_v74,
)


SCHEMA = 1
SCOPE_ID = "braids-q4-route-lifecycle-composition-v81-v1"
CHECKPOINT_SCOPE_ID = SCOPE_ID + ":checkpoint"
Q3_RECEIPT_SCOPE_ID = SCOPE_ID + ":q3-raw-split-receipt"
SEMANTIC_ID = hashlib.sha256(SCOPE_ID.encode("ascii")).hexdigest()
METHODS = (
    METHOD_BIJKER_CURRENT,
    METHOD_VAN_THIEL_VAN_RIJN_2008_CURRENT,
    METHOD_XBEACH_SOULSBY_VAN_RIJN_CURRENT,
)
RAW_ROWS = (
    "bed_x_kg_m_s", "bed_y_kg_m_s", "suspended_x_kg_m_s",
    "suspended_y_kg_m_s", "equilibrium_concentration_kg_m3",
)
SPEED_EPSILON = 1.0e-6
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class RouteLifecycleV81Error(ValueError):
    """Fail-closed v81 identity, ownership, or transaction error."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, OverflowError) as error:
        raise RouteLifecycleV81Error(
            "v81 value is not finite canonical JSON") from error


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise RouteLifecycleV81Error(
            f"v81 {name} must be a lowercase SHA-256 digest")
    return value


def _array(value: object, shape: tuple[int, ...], name: str, *,
           nonnegative: bool = False) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise RouteLifecycleV81Error(
            f"v81 {name} must be finite with shape {shape}")
    if nonnegative and np.any(result < 0.0):
        raise RouteLifecycleV81Error(f"v81 {name} must be nonnegative")
    result = np.ascontiguousarray(result)
    result.setflags(write=False)
    return result


def _digest(*values: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(value)
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(_canonical(list(array.shape)))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def make_q3_raw_split_receipt_v81(
    *, producer_execution_sha256: str, q3_handoff_fingerprint: str,
    method_ids: Sequence[str] = METHODS,
) -> dict[str, object]:
    """Authenticate the declared ABI for five raw, mass-valued v74 rows.

    The receipt closes the otherwise untyped SHA slot in the frozen v74 API;
    each trial separately binds the caller-supplied row bytes.  Neither step
    claims to re-authenticate or rerun a Q3/v67 constitutive formula.
    """
    methods = tuple(method_ids)
    if methods != METHODS:
        raise RouteLifecycleV81Error(
            "v81 Q3 receipt requires ordered Bijker/TraFrm19/TraFrm20 lanes")
    body = {
        "schema": SCHEMA,
        "scope_id": Q3_RECEIPT_SCOPE_ID,
        "producer_execution_sha256": _require_sha256(
            producer_execution_sha256, "Q3 producer execution digest"),
        "q3_handoff_fingerprint": _require_sha256(
            q3_handoff_fingerprint, "Q3 handoff digest"),
        "method_ids": list(methods),
        "raw_rows": list(RAW_ROWS),
        "mass_valued": True,
        "forcing": "current_only_no_wave",
        "wave_active": False,
        "external_secondary_flow_rotation": False,
        "post_q3_density_conversion_count": 0,
        "row_provenance_scope": (
            "caller_supplied_rows_bound_per_trial_not_end_to_end_v67_execution"),
    }
    return {**body, "fingerprint": _fingerprint(body)}


def validate_q3_raw_split_receipt_v81(value: Mapping[str, object]) -> dict:
    expected = {
        "schema", "scope_id", "producer_execution_sha256",
        "q3_handoff_fingerprint", "method_ids", "raw_rows", "mass_valued",
        "forcing", "wave_active", "external_secondary_flow_rotation",
        "post_q3_density_conversion_count", "row_provenance_scope",
        "fingerprint",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise RouteLifecycleV81Error("v81 Q3 receipt inventory differs")
    body = {key: value[key] for key in value if key != "fingerprint"}
    if (value["fingerprint"] != _fingerprint(body) or
            value["schema"] != SCHEMA or
            value["scope_id"] != Q3_RECEIPT_SCOPE_ID or
            tuple(value["method_ids"]) != METHODS or
            tuple(value["raw_rows"]) != RAW_ROWS or
            value["mass_valued"] is not True or
            value["forcing"] != "current_only_no_wave" or
            value["wave_active"] is not False or
            value["external_secondary_flow_rotation"] is not False or
            value["post_q3_density_conversion_count"] != 0 or
            value["row_provenance_scope"] !=
            "caller_supplied_rows_bound_per_trial_not_end_to_end_v67_execution"):
        raise RouteLifecycleV81Error(
            "v81 Q3 receipt does not authenticate the current-only raw split")
    _require_sha256(value["producer_execution_sha256"],
                    "Q3 producer execution digest")
    _require_sha256(value["q3_handoff_fingerprint"], "Q3 handoff digest")
    return dict(value)


@dataclass(frozen=True)
class RouteLifecycleExecutionV81:
    route_execution: dict[str, object]
    lifecycle_execution: MultifractionSuspendedExecutionV78
    q3_receipt: dict[str, object]
    semantic_id: str
    fingerprint: str

    @property
    def role(self) -> str:
        return self.lifecycle_execution.role

    @property
    def fraction_count(self) -> int:
        return self.lifecycle_execution.fraction_count

    @property
    def cell_count(self) -> int:
        return self.lifecycle_execution.cell_count


def _execution_payload(value: RouteLifecycleExecutionV81) -> dict:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "semantic_id": value.semantic_id,
        "route_execution_v74": value.route_execution,
        "lifecycle_execution_v78": execution_document_v78(
            value.lifecycle_execution),
        "q3_raw_split_receipt": value.q3_receipt,
        "q3_row_provenance": (
            "caller supplied; every trial hashes all five raw rows, but v81 "
            "does not claim an end-to-end v67 formula invocation"),
        "route_to_lifecycle_bridge": {
            "Bijker": "Sus*hypot(raw_sscu,raw_sscv)",
            "TraFrm19_20":
                "Sus*raw_Ceq*(endpoint_speed+1e-6)*exchange_depth",
            "surface_fraction_owner": "v78_capacity_to_concentration_once",
            "off_and_total_capacity": "exact_zero",
            "combine_equilibrium_owner_count": 1,
        },
        "integrator": "implicit_fwave_midpoint",
        "fixed_grid": True,
        "wave_active": False,
        "external_secondary_flow_rotation": False,
        "automatic_backend_fallback": False,
        "accepted_update_limit": MAX_ACCEPTED_UPDATES,
        "amr": False,
        "product_admission": False,
    }


def resolve_route_lifecycle_execution_v81(
    route_execution: Mapping[str, object],
    lifecycle_execution: MultifractionSuspendedExecutionV78, *,
    q3_receipt: Mapping[str, object], root: str | Path | None = None,
) -> RouteLifecycleExecutionV81:
    try:
        route = load_route_execution_v74(route_execution, root=root)
        receipt = validate_q3_raw_split_receipt_v81(q3_receipt)
        lifecycle_document = execution_document_v78(lifecycle_execution)
    except (TypeError, ValueError) as error:
        raise RouteLifecycleV81Error(
            "v81 predecessor execution or receipt authentication failed") \
            from error
    if tuple(route["method_ids"]) != METHODS or lifecycle_execution.fraction_count != 3:
        raise RouteLifecycleV81Error(
            "v81 requires exactly three ordered Bijker/TraFrm19/TraFrm20 lanes")
    if (receipt["q3_handoff_fingerprint"] !=
            route["q3_handoff_fingerprint"]):
        raise RouteLifecycleV81Error(
            "v81 typed Q3 receipt and v74 handoff digest differ")
    expected_backend = (
        "cpu" if lifecycle_execution.role == ROLE_CPU_ORACLE
        else "implicit_fwave_midpoint_gpu"
    )
    if (lifecycle_execution.role not in (ROLE_CPU_ORACLE, ROLE_CUDA) or
            route["backend"] != expected_backend or
            route["automatic_backend_fallback"] is not False or
            lifecycle_document["automatic_cpu_fallback"] is not False):
        raise RouteLifecycleV81Error(
            "v81 route/lifecycle backend roles differ or permit fallback")
    profile = lifecycle_execution.base_execution.profile
    checks = (
        (float(route["speed_epsilon"]), SPEED_EPSILON, "speed epsilon"),
        (float(route["fac_tsd"]), float(profile.fac_tsd), "fac_tsd"),
        (float(route["sediment_threshold_m"]),
         float(profile.sediment_threshold_m), "sediment threshold"),
        (float(route["minimum_reference_depth_m"]),
         float(profile.reference_depth_floor_m), "reference depth"),
    )
    for left, right, name in checks:
        if left != right:
            raise RouteLifecycleV81Error(
                f"v81 v74/v78 {name} identities differ")
    placeholder = RouteLifecycleExecutionV81(
        route, lifecycle_execution, receipt, SEMANTIC_ID, "")
    return RouteLifecycleExecutionV81(
        route, lifecycle_execution, receipt, SEMANTIC_ID,
        _fingerprint(_execution_payload(placeholder)))


def execution_document_v81(value: RouteLifecycleExecutionV81) -> dict:
    if not isinstance(value, RouteLifecycleExecutionV81):
        raise RouteLifecycleV81Error("v81 execution object differs")
    payload = _execution_payload(value)
    if value.fingerprint != _fingerprint(payload):
        raise RouteLifecycleV81Error("v81 execution fingerprint differs")
    return {**payload, "fingerprint": value.fingerprint}


@dataclass(frozen=True)
class RouteLifecycleStateV81:
    execution_fingerprint: str
    lifecycle: MultifractionSuspendedStateV78
    accepted_route_chain_sha256: str
    state_sha256: str

    @property
    def accepted_updates(self) -> int:
        return self.lifecycle.accepted_updates

    @property
    def time_s(self) -> float:
        return self.lifecycle.time_s

    @property
    def suspended_mass_kg_m2(self) -> np.ndarray:
        return self.lifecycle.suspended_mass_kg_m2

    @property
    def bed_mass_kg_m2(self) -> np.ndarray:
        return self.lifecycle.bed_mass_kg_m2


def _state_token(value: RouteLifecycleStateV81) -> str:
    return _fingerprint({
        "execution_fingerprint": value.execution_fingerprint,
        "lifecycle_state_sha256": value.lifecycle.state_sha256,
        "accepted_route_chain_sha256": value.accepted_route_chain_sha256,
    })


def _assert_nonwater_routes(
    execution: RouteLifecycleExecutionV81,
    lifecycle: MultifractionSuspendedStateV78,
) -> None:
    routes = np.asarray(execution.route_execution["routes"], dtype=np.int32)
    inactive = routes != ROUTE_COMBINE
    base = lifecycle.lifecycle
    arrays = (
        base.suspended_mass_kg_m2[inactive],
        base.boundary_last_outflow_concentration_kg_m3[inactive],
        base.boundary_return_clock_s[inactive],
        base.advective_boundary_ledger_kg[inactive],
        base.diffusive_boundary_ledger_kg[inactive],
        base.water_exchange_ledger_kg[inactive],
        base.bed_exchange_ledger_kg[inactive],
        base.wetdry_water_ledger_kg[inactive],
        base.wetdry_bed_ledger_kg[inactive],
        base.reference_water_mass_kg[inactive],
    )
    if any(np.any(value != 0.0) for value in arrays):
        raise RouteLifecycleV81Error(
            "v81 off/total routes cannot own water-column mass or ledgers")


def _wrap_state(
    execution: RouteLifecycleExecutionV81,
    lifecycle: MultifractionSuspendedStateV78,
    route_chain_sha256: str,
) -> RouteLifecycleStateV81:
    execution_document_v81(execution)
    validate_multifraction_suspended_state_v78(
        lifecycle, execution.lifecycle_execution)
    _require_sha256(route_chain_sha256, "accepted route-chain digest")
    _assert_nonwater_routes(execution, lifecycle)
    placeholder = RouteLifecycleStateV81(
        execution.fingerprint, lifecycle, route_chain_sha256, "")
    return RouteLifecycleStateV81(
        placeholder.execution_fingerprint, placeholder.lifecycle,
        placeholder.accepted_route_chain_sha256, _state_token(placeholder))


def resolve_route_lifecycle_state_v81(
    execution: RouteLifecycleExecutionV81, **state_fields,
) -> RouteLifecycleStateV81:
    from .multifraction_suspended_lifecycle_v78 import (
        resolve_multifraction_suspended_state_v78,
    )
    try:
        lifecycle = resolve_multifraction_suspended_state_v78(
            execution.lifecycle_execution, **state_fields)
    except (TypeError, ValueError) as error:
        raise RouteLifecycleV81Error(
            "v81 initial lifecycle state is invalid") from error
    if lifecycle.accepted_updates != 0:
        raise RouteLifecycleV81Error(
            "v81 fresh state must start before the first accepted update")
    chain = _fingerprint({
        "scope_id": SCOPE_ID,
        "execution_fingerprint": execution.fingerprint,
        "initial_lifecycle_state_sha256": lifecycle.state_sha256,
    })
    return _wrap_state(execution, lifecycle, chain)


def validate_route_lifecycle_state_v81(
    state: RouteLifecycleStateV81,
    execution: RouteLifecycleExecutionV81,
) -> None:
    if (not isinstance(state, RouteLifecycleStateV81) or
            state.execution_fingerprint != execution.fingerprint or
            state.state_sha256 != _state_token(state)):
        raise RouteLifecycleV81Error(
            "v81 state is stale, mutated, or execution-mismatched")
    _wrap_state(execution, state.lifecycle, state.accepted_route_chain_sha256)


@dataclass(frozen=True)
class RouteLifecycleTrialV81:
    execution_fingerprint: str
    parent_state_sha256: str
    accepted_step_receipt_sha256: str
    q3_control_sha256: str
    route_handoff_sha256: str
    candidate: RouteLifecycleStateV81
    lifecycle_trial: MultifractionSuspendedTrialV78
    bed_capacity_x_kg_m_s: np.ndarray
    bed_capacity_y_kg_m_s: np.ndarray
    equilibrium_target_kg_m3: np.ndarray
    suspended_equivalent_x_kg_m_s: np.ndarray
    suspended_equivalent_y_kg_m_s: np.ndarray
    lifecycle_capacity_kg_m_s: np.ndarray
    base_bed_capacity_owner_count: np.ndarray
    bed_suspended_owner_count: np.ndarray
    water_suspended_owner_count: np.ndarray
    equilibrium_injection_owner_count: np.ndarray
    suspended_factor_application_count: np.ndarray
    active_fraction_application_count: np.ndarray
    duplicate_owner_count: np.ndarray
    post_q3_density_conversion_count: int
    equilibrium_reconstruction_max_abs_kg_m3: float
    bijker_direction_cross_max: float
    backend: str
    trial_sha256: str


def _trial_token(value: RouteLifecycleTrialV81) -> str:
    return _fingerprint({
        "execution_fingerprint": value.execution_fingerprint,
        "parent_state_sha256": value.parent_state_sha256,
        "accepted_step_receipt_sha256":
            value.accepted_step_receipt_sha256,
        "q3_control_sha256": value.q3_control_sha256,
        "route_handoff_sha256": value.route_handoff_sha256,
        "candidate_state_sha256": value.candidate.state_sha256,
        "lifecycle_trial_sha256": value.lifecycle_trial.trial_sha256,
        "arrays_sha256": _digest(
            value.bed_capacity_x_kg_m_s,
            value.bed_capacity_y_kg_m_s,
            value.equilibrium_target_kg_m3,
            value.suspended_equivalent_x_kg_m_s,
            value.suspended_equivalent_y_kg_m_s,
            value.lifecycle_capacity_kg_m_s,
            value.base_bed_capacity_owner_count,
            value.bed_suspended_owner_count,
            value.water_suspended_owner_count,
            value.equilibrium_injection_owner_count,
            value.suspended_factor_application_count,
            value.active_fraction_application_count,
            value.duplicate_owner_count,
        ),
        "post_q3_density_conversion_count":
            value.post_q3_density_conversion_count,
        "equilibrium_reconstruction_max_abs_kg_m3":
            value.equilibrium_reconstruction_max_abs_kg_m3,
        "bijker_direction_cross_max": value.bijker_direction_cross_max,
        "backend": value.backend,
    })


def _prepare_route_cpu(
    execution: RouteLifecycleExecutionV81, *, depth_after_m: object,
    endpoint_momentum_m2_s: object, raw_bed_x_kg_m_s: object,
    raw_bed_y_kg_m_s: object, raw_suspended_x_kg_m_s: object,
    raw_suspended_y_kg_m_s: object,
    raw_equilibrium_concentration_kg_m3: object,
    active_fraction: object,
) -> tuple[RouteHandoffV74, np.ndarray, np.ndarray, np.ndarray, str]:
    nc, ng = execution.cell_count, execution.fraction_count
    base = execution.lifecycle_execution.base_execution
    h1 = _array(depth_after_m, (nc,), "depth after", nonnegative=True)
    momentum = _array(endpoint_momentum_m2_s, (2, nc), "endpoint momentum")
    shape = (ng, nc)
    raw_bed_x = _array(raw_bed_x_kg_m_s, shape, "raw bed x")
    raw_bed_y = _array(raw_bed_y_kg_m_s, shape, "raw bed y")
    raw_sus_x = _array(raw_suspended_x_kg_m_s, shape, "raw suspended x")
    raw_sus_y = _array(raw_suspended_y_kg_m_s, shape, "raw suspended y")
    raw_eq = _array(raw_equilibrium_concentration_kg_m3, shape,
                    "raw equilibrium concentration", nonnegative=True)
    fraction = _array(active_fraction, shape, "active fraction",
                      nonnegative=True)
    if np.any(fraction > 1.0):
        raise RouteLifecycleV81Error("v81 active fraction exceeds one")
    wet = h1 > base.dry_tolerance_m
    sediment_active = wet & (h1 > base.profile.sediment_threshold_m)
    raw_rows = (raw_bed_x, raw_bed_y, raw_sus_x, raw_sus_y, raw_eq)
    if any(np.any(value[:, ~sediment_active] != 0.0) for value in raw_rows):
        raise RouteLifecycleV81Error(
            "v81 raw formula rows must be exact zero outside active wet cells")
    speed = np.zeros(nc, dtype=np.float64)
    velocity_x = np.zeros(nc, dtype=np.float64)
    velocity_y = np.zeros(nc, dtype=np.float64)
    velocity_x[wet] = momentum[0, wet]/h1[wet]
    velocity_y[wet] = momentum[1, wet]/h1[wet]
    speed[wet] = np.hypot(velocity_x[wet], velocity_y[wet])
    exchange_depth = np.where(
        sediment_active,
        np.maximum(base.profile.reference_depth_floor_m, h1), 1.0)
    try:
        handoff = compose_route_handoff_v74(
            execution=execution.route_execution, depth_m=exchange_depth,
            velocity_x_m_s=velocity_x, velocity_y_m_s=velocity_y,
            raw_bed_x_kg_m_s=raw_bed_x,
            raw_bed_y_kg_m_s=raw_bed_y,
            raw_suspended_x_kg_m_s=raw_sus_x,
            raw_suspended_y_kg_m_s=raw_sus_y,
            raw_equilibrium_concentration_kg_m3=raw_eq,
            active_fraction=fraction)
        validate_route_handoff_v74(
            handoff, execution=execution.route_execution)
    except (TypeError, ValueError) as error:
        raise RouteLifecycleV81Error("v81 v74 route handoff failed") from error
    routes = np.asarray(execution.route_execution["routes"], dtype=np.int32)
    sus = float(execution.route_execution["suspended_factor"])
    capacity = np.zeros(shape, dtype=np.float64)
    combine = routes == ROUTE_COMBINE
    capacity[0] = sus*np.hypot(raw_sus_x[0], raw_sus_y[0])
    capacity[1:] = (sus*raw_eq[1:]*(speed[None, :]+SPEED_EPSILON)*
                    exchange_depth[None, :])
    capacity[~combine] = 0.0
    capacity = np.ascontiguousarray(capacity)
    capacity.setflags(write=False)
    control = _fingerprint({
        "q3_receipt_fingerprint": execution.q3_receipt["fingerprint"],
        "route_execution_fingerprint":
            execution.route_execution["fingerprint"],
        "arrays_sha256": _digest(
            h1, momentum, *raw_rows, fraction, exchange_depth, speed),
    })
    return handoff, capacity, fraction, speed, control


def _assert_boundary_ownership(
    execution: RouteLifecycleExecutionV81, *, left: np.ndarray,
    right: np.ndarray, prescribed: np.ndarray,
) -> None:
    routes = np.asarray(execution.route_execution["routes"], dtype=np.int32)
    inactive = routes != ROUTE_COMBINE
    if (np.any(left[inactive] != 0.0) or np.any(right[inactive] != 0.0) or
            np.any(prescribed[inactive] != 0.0)):
        raise RouteLifecycleV81Error(
            "v81 off/total routes cannot receive water-column boundary data")


def _build_trial(
    execution: RouteLifecycleExecutionV81,
    parent: RouteLifecycleStateV81,
    lifecycle_trial: MultifractionSuspendedTrialV78,
    accepted_step_receipt: Mapping[str, object], *,
    q3_control_sha256: str, route_handoff_sha256: str,
    bed_capacity_x: object, bed_capacity_y: object,
    equilibrium_target: object, equivalent_x: object, equivalent_y: object,
    lifecycle_capacity: object, bed_owner: object, water_owner: object,
    suspended_factor_count: object, active_fraction_count: object,
    raw_suspended_x: object, raw_suspended_y: object,
    active_fraction: object, backend: str,
) -> RouteLifecycleTrialV81:
    ng, nc = execution.fraction_count, execution.cell_count
    shape = (ng, nc)
    arrays = [
        _array(bed_capacity_x, shape, "bed capacity x"),
        _array(bed_capacity_y, shape, "bed capacity y"),
        _array(equilibrium_target, shape, "equilibrium target",
               nonnegative=True),
        _array(equivalent_x, shape, "suspended equivalent x"),
        _array(equivalent_y, shape, "suspended equivalent y"),
        _array(lifecycle_capacity, shape, "lifecycle capacity",
               nonnegative=True),
    ]
    owners = []
    for raw, name in (
        (bed_owner, "bed owner"), (water_owner, "water owner"),
        (suspended_factor_count, "suspended-factor count"),
        (active_fraction_count, "active-fraction count"),
    ):
        value = np.asarray(raw, dtype=np.int32)
        if value.shape != (ng,):
            raise RouteLifecycleV81Error(f"v81 {name} shape differs")
        value = np.ascontiguousarray(value)
        value.setflags(write=False)
        owners.append(value)
    bed_count, water_count, sus_count, fraction_count = owners
    injection = np.ascontiguousarray(water_count.copy())
    duplicate = np.ascontiguousarray(
        bed_count*water_count + np.maximum(injection-1, 0))
    injection.setflags(write=False); duplicate.setflags(write=False)
    routes = np.asarray(execution.route_execution["routes"], dtype=np.int32)
    expected_bed = (routes == ROUTE_TOTAL).astype(np.int32)
    expected_water = (routes == ROUTE_COMBINE).astype(np.int32)
    expected_factor = (routes != ROUTE_OFF).astype(np.int32)
    base_bed_owner = np.ascontiguousarray(expected_factor.copy())
    base_bed_owner.setflags(write=False)
    if (not np.array_equal(bed_count, expected_bed) or
            not np.array_equal(water_count, expected_water) or
            not np.array_equal(injection, expected_water) or
            not np.array_equal(sus_count, expected_factor) or
            not np.array_equal(fraction_count, expected_factor) or
            np.any(duplicate != 0)):
        raise RouteLifecycleV81Error(
            "v81 route owner or exactly-once factor count differs")
    target_error = float(np.max(np.abs(
        lifecycle_trial.equilibrium_concentration_kg_m3-arrays[2])))
    target_scale = max(1.0, float(np.max(np.abs(arrays[2]))))
    if target_error > 64.0*np.finfo(np.float64).eps*target_scale:
        raise RouteLifecycleV81Error(
            "v81 v78 equilibrium reconstruction differs from v74 target")
    raw_x = _array(raw_suspended_x, shape, "raw suspended x")
    raw_y = _array(raw_suspended_y, shape, "raw suspended y")
    fraction = _array(active_fraction, shape, "active fraction")
    suspended_factor = float(execution.route_execution["suspended_factor"])
    expected_x = fraction[0]*(suspended_factor*raw_x[0])
    expected_y = fraction[0]*(suspended_factor*raw_y[0])
    if routes[0] == ROUTE_OFF:
        expected_x = np.zeros_like(expected_x)
        expected_y = np.zeros_like(expected_y)
    if (not np.array_equal(arrays[3][0], expected_x) or
            not np.array_equal(arrays[4][0], expected_y)):
        raise RouteLifecycleV81Error(
            "v81 Bijker suspended-vector direction was altered")
    direction_cross = float(np.max(np.abs(
        arrays[3][0]*expected_y-arrays[4][0]*expected_x)))
    if any(np.any(value[routes != ROUTE_COMBINE] != 0.0) for value in (
            lifecycle_trial.candidate.suspended_mass_kg_m2,
            lifecycle_trial.transferred_to_suspension_kg_m2,
            lifecycle_trial.advective_face_flux_kg_m_s,
            lifecycle_trial.diffusive_face_mass_owner_out_kg,
            lifecycle_trial.wetdry_transfer_to_bed_kg_m2)):
        raise RouteLifecycleV81Error(
            "v81 off/total route created water-column mass")
    chain = _fingerprint({
        "previous": parent.accepted_route_chain_sha256,
        "accepted_step_receipt": accepted_step_receipt["fingerprint"],
        "q3_control_sha256": q3_control_sha256,
        "route_handoff_sha256": route_handoff_sha256,
        "lifecycle_trial_sha256": lifecycle_trial.trial_sha256,
    })
    candidate = _wrap_state(
        execution, lifecycle_trial.candidate, chain)
    placeholder = RouteLifecycleTrialV81(
        execution.fingerprint, parent.state_sha256,
        str(accepted_step_receipt["fingerprint"]), q3_control_sha256,
        route_handoff_sha256, candidate, lifecycle_trial, *arrays,
        base_bed_owner, bed_count, water_count, injection, sus_count,
        fraction_count,
        duplicate, 0, target_error, direction_cross, backend, "")
    return RouteLifecycleTrialV81(
        **{**placeholder.__dict__, "trial_sha256": _trial_token(placeholder)})


def begin_route_lifecycle_cpu_trial_v81(
    execution: RouteLifecycleExecutionV81,
    parent: RouteLifecycleStateV81, *,
    accepted_step_receipt: Mapping[str, object],
    depth_before_m: object, depth_after_m: object,
    accepted_water_flux_m2_s: object,
    reconstructed_left_concentration_kg_m3: object,
    reconstructed_right_concentration_kg_m3: object,
    prescribed_boundary_concentration_kg_m3: object,
    accepted_face_exchange_area_m2: object,
    transport_face_active: object, endpoint_momentum_m2_s: object,
    shear_velocity_m_s: object, hydro_constant: object,
    raw_bed_x_kg_m_s: object, raw_bed_y_kg_m_s: object,
    raw_suspended_x_kg_m_s: object, raw_suspended_y_kg_m_s: object,
    raw_equilibrium_concentration_kg_m3: object,
    active_fraction: object, dt_s: float, cancelled: bool = False,
) -> RouteLifecycleTrialV81:
    if execution.role != ROLE_CPU_ORACLE:
        raise RouteLifecycleV81Error(
            "v81 CPU oracle requires the isolated_cpu_oracle role")
    validate_route_lifecycle_state_v81(parent, execution)
    if cancelled:
        raise RouteLifecycleV81Error(
            "v81 update cancelled before candidate publication")
    if parent.accepted_updates >= MAX_ACCEPTED_UPDATES:
        raise RouteLifecycleV81Error("v81 accepted update limit was reached")
    ng, nf = execution.fraction_count, execution.lifecycle_execution.face_count
    nb = execution.lifecycle_execution.boundary_count
    left = _array(reconstructed_left_concentration_kg_m3, (ng, nf),
                  "left reconstructed concentration")
    right = _array(reconstructed_right_concentration_kg_m3, (ng, nf),
                   "right reconstructed concentration")
    prescribed = _array(prescribed_boundary_concentration_kg_m3, (ng, nb),
                        "prescribed boundary concentration", nonnegative=True)
    _assert_boundary_ownership(
        execution, left=left, right=right, prescribed=prescribed)
    handoff, capacity, fraction, _, control = _prepare_route_cpu(
        execution, depth_after_m=depth_after_m,
        endpoint_momentum_m2_s=endpoint_momentum_m2_s,
        raw_bed_x_kg_m_s=raw_bed_x_kg_m_s,
        raw_bed_y_kg_m_s=raw_bed_y_kg_m_s,
        raw_suspended_x_kg_m_s=raw_suspended_x_kg_m_s,
        raw_suspended_y_kg_m_s=raw_suspended_y_kg_m_s,
        raw_equilibrium_concentration_kg_m3=
            raw_equilibrium_concentration_kg_m3,
        active_fraction=active_fraction)
    try:
        lifecycle_trial = begin_multifraction_cpu_oracle_trial_v78(
            execution.lifecycle_execution, parent.lifecycle,
            accepted_step_receipt=accepted_step_receipt,
            depth_before_m=depth_before_m, depth_after_m=depth_after_m,
            accepted_water_flux_m2_s=accepted_water_flux_m2_s,
            reconstructed_left_concentration_kg_m3=left,
            reconstructed_right_concentration_kg_m3=right,
            prescribed_boundary_concentration_kg_m3=prescribed,
            accepted_face_exchange_area_m2=accepted_face_exchange_area_m2,
            transport_face_active=transport_face_active,
            endpoint_momentum_m2_s=endpoint_momentum_m2_s,
            shear_velocity_m_s=shear_velocity_m_s,
            suspended_transport_capacity_kg_m_s=capacity,
            surface_fraction=fraction, hydro_constant=hydro_constant,
            dt_s=dt_s)
    except (TypeError, ValueError) as error:
        raise RouteLifecycleV81Error(
            "v81 CPU lifecycle rejected the bridged trial") from error
    return _build_trial(
        execution, parent, lifecycle_trial, accepted_step_receipt,
        q3_control_sha256=control,
        route_handoff_sha256=handoff.handoff_sha256,
        bed_capacity_x=handoff.bed_capacity_x,
        bed_capacity_y=handoff.bed_capacity_y,
        equilibrium_target=handoff.equilibrium_concentration,
        equivalent_x=handoff.suspended_equivalent_x,
        equivalent_y=handoff.suspended_equivalent_y,
        lifecycle_capacity=capacity,
        bed_owner=handoff.bed_suspended_owner_count,
        water_owner=handoff.water_suspended_owner_count,
        suspended_factor_count=handoff.suspended_factor_application_count,
        active_fraction_count=handoff.active_fraction_application_count,
        raw_suspended_x=raw_suspended_x_kg_m_s,
        raw_suspended_y=raw_suspended_y_kg_m_s,
        active_fraction=active_fraction, backend=ROLE_CPU_ORACLE)


def _validate_trial(
    execution: RouteLifecycleExecutionV81,
    parent: RouteLifecycleStateV81,
    trial: RouteLifecycleTrialV81,
) -> None:
    validate_route_lifecycle_state_v81(parent, execution)
    if (not isinstance(trial, RouteLifecycleTrialV81) or
            trial.execution_fingerprint != execution.fingerprint or
            trial.parent_state_sha256 != parent.state_sha256 or
            trial.trial_sha256 != _trial_token(trial)):
        raise RouteLifecycleV81Error(
            "v81 trial is stale, mutated, or belongs to another parent")
    validate_route_lifecycle_state_v81(trial.candidate, execution)


def commit_route_lifecycle_trial_v81(
    execution: RouteLifecycleExecutionV81,
    parent: RouteLifecycleStateV81,
    trial: RouteLifecycleTrialV81,
) -> RouteLifecycleStateV81:
    _validate_trial(execution, parent, trial)
    commit_multifraction_trial_v78(
        execution.lifecycle_execution, parent.lifecycle,
        trial.lifecycle_trial)
    return trial.candidate


def rollback_route_lifecycle_trial_v81(
    execution: RouteLifecycleExecutionV81,
    parent: RouteLifecycleStateV81,
    trial: RouteLifecycleTrialV81,
) -> RouteLifecycleStateV81:
    _validate_trial(execution, parent, trial)
    rollback_multifraction_trial_v78(
        execution.lifecycle_execution, parent.lifecycle,
        trial.lifecycle_trial)
    return parent


def cancel_route_lifecycle_trial_v81(
    execution: RouteLifecycleExecutionV81,
    parent: RouteLifecycleStateV81,
    trial: RouteLifecycleTrialV81,
) -> RouteLifecycleStateV81:
    _validate_trial(execution, parent, trial)
    cancel_multifraction_trial_v78(
        execution.lifecycle_execution, parent.lifecycle,
        trial.lifecycle_trial)
    return parent


def route_lifecycle_checkpoint_v81(
    execution: RouteLifecycleExecutionV81,
    state: RouteLifecycleStateV81,
) -> dict:
    validate_route_lifecycle_state_v81(state, execution)
    body = {
        "schema": SCHEMA,
        "scope_id": CHECKPOINT_SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "execution_fingerprint": execution.fingerprint,
        "transaction_schema": "accepted_route_and_lifecycle_state_v1",
        "accepted_route_chain_sha256": state.accepted_route_chain_sha256,
        "lifecycle_checkpoint_v78": multifraction_checkpoint_v78(
            execution.lifecycle_execution, state.lifecycle),
        "state_sha256": state.state_sha256,
    }
    return {**body, "fingerprint": _fingerprint(body)}


def load_route_lifecycle_checkpoint_v81(
    execution: RouteLifecycleExecutionV81,
    document: Mapping[str, object],
) -> RouteLifecycleStateV81:
    expected = {
        "schema", "scope_id", "semantic_id", "execution_fingerprint",
        "transaction_schema", "accepted_route_chain_sha256",
        "lifecycle_checkpoint_v78", "state_sha256", "fingerprint",
    }
    if not isinstance(document, Mapping) or set(document) != expected:
        raise RouteLifecycleV81Error("v81 checkpoint inventory differs")
    body = {key: document[key] for key in document if key != "fingerprint"}
    if (document["fingerprint"] != _fingerprint(body) or
            document["schema"] != SCHEMA or
            document["scope_id"] != CHECKPOINT_SCOPE_ID or
            document["semantic_id"] != SEMANTIC_ID or
            document["execution_fingerprint"] != execution.fingerprint or
            document["transaction_schema"] !=
            "accepted_route_and_lifecycle_state_v1"):
        raise RouteLifecycleV81Error("v81 checkpoint identity differs")
    try:
        lifecycle = load_multifraction_checkpoint_v78(
            execution.lifecycle_execution,
            document["lifecycle_checkpoint_v78"])
    except (TypeError, ValueError) as error:
        raise RouteLifecycleV81Error(
            "v81 embedded v78 checkpoint is invalid") from error
    state = _wrap_state(
        execution, lifecycle, document["accepted_route_chain_sha256"])
    if state.state_sha256 != document["state_sha256"]:
        raise RouteLifecycleV81Error("v81 checkpoint state digest differs")
    return state


def method_identity_v81() -> dict[str, object]:
    return {
        "scope_id": SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "builds_additively_on": [
            "braids-q4-split-suspended-route-owner-v74-v1",
            "braids-q4-multifraction-suspended-lifecycle-v78-v1",
        ],
        "methods": list(METHODS),
        "routes": {"off": ROUTE_OFF, "total": ROUTE_TOTAL,
                   "combine": ROUTE_COMBINE},
        "bridge": (
            "raw pre-fraction capacity reconstructs the authenticated v74 "
            "equilibrium target inside v78"),
        "q3_row_provenance": (
            "caller-supplied raw rows bound into each trial; no end-to-end "
            "v67 formula execution is claimed"),
        "base_bed_capacity_owner":
            "one algebraic bed-vector owner for total and combine routes",
        "bed_suspended_owner":
            "formula suspended half belongs to algebraic bed only for total",
        "factor_ownership":
            "Bed/Sus once in route algebra; active fraction once per owner",
        "bijker_direction": "raw vector preserved; no external rotation",
        "trafrm19_20": "equilibrium-concentration semantics preserved",
        "integrator": "implicit_fwave_midpoint",
        "strict_accepted_receipts": True,
        "fixed_grid_only": True,
        "wave_active": False,
        "automatic_backend_fallback": False,
        "accepted_update_limit": MAX_ACCEPTED_UPDATES,
        "transaction": "immutable_commit_rollback_cancel_exact_restart_v1",
        "amr_claimed": False,
        "product_admission_claimed": False,
        "delft_trajectory_equivalence_claimed": False,
        "delft3d_source_modified": False,
    }


FROZEN_PREDECESSOR_SHA256 = {
    "braids_core/sediment/suspended_route_live_v74.py":
        "938dffc627dc6ec1102aff13d8999f2a5410de9853026e5c7cbfcfa65c29f8dd",
    "braids_gpu/suspended_route_live_v74.py":
        "c13c9875b246dff65371810654cd5610bedd088cd7095a81bead942f2e0412cb",
    "braids_gpu/suspended_route_live_v74.cu":
        "b905edce1829fe65af1e3c2857c9d56b8e6f7c0b9940a7520a1f5889bf657a45",
    "braids_core/sediment/suspended_lifecycle_live_v76.py":
        "20ba39ba7228ea6c50e2ea310777d59516c9c157c05d71c6bf3298fed07d4ee6",
    "braids_gpu/suspended_lifecycle_live_v76.py":
        "182355c4ef1184fb64b877ca71ef5ada344bd40eb27bc6c5c0968a0541661ff8",
    "braids_core/sediment/multifraction_suspended_lifecycle_v78.py":
        "eaba4861d38369eaf740803c43a722e10a0ee70bb9a2e1dcf56bddba2a7e10dd",
    "braids_gpu/multifraction_suspended_lifecycle_v78.py":
        "e4fdd4aa5244d0ceff7f7b5479cdfbe4bc2cd9c5344f731cecf5351180facc6b",
}


def verify_frozen_predecessors_v81(root: str | Path) -> dict[str, str]:
    root = Path(root)
    result = {}
    for relative, expected in FROZEN_PREDECESSOR_SHA256.items():
        path = root/relative
        if not path.is_file():
            raise RouteLifecycleV81Error(
                "v81 frozen predecessor is absent: "+relative)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise RouteLifecycleV81Error(
                "v81 frozen predecessor hash differs: "+relative)
        result[relative] = actual
    return result


def verify_sources_v81(root: str | Path) -> dict[str, str]:
    root = Path(root)
    paths = (
        "braids_core/sediment/route_lifecycle_composition_v81.py",
        "braids_gpu/route_lifecycle_composition_v81.py",
        "braids_gpu/route_lifecycle_composition_v81.cu",
        "tests/test_q4_route_lifecycle_composition_v81.py",
        "tests/test_q4_route_lifecycle_composition_cuda_v81.py",
        "validation/sediment/delft3d_current/"
        "q4_route_lifecycle_composition_v81/build_report.py",
    )
    result = {}
    for relative in paths:
        path = root/relative
        if not path.is_file():
            raise RouteLifecycleV81Error(
                "v81 additive source is absent: "+relative)
        result[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


__all__ = [
    "CHECKPOINT_SCOPE_ID", "MAX_ACCEPTED_UPDATES", "METHODS", "RAW_ROWS",
    "RouteLifecycleExecutionV81", "RouteLifecycleStateV81",
    "RouteLifecycleTrialV81", "RouteLifecycleV81Error", "SCHEMA",
    "SCOPE_ID", "SEMANTIC_ID", "SPEED_EPSILON",
    "begin_route_lifecycle_cpu_trial_v81",
    "cancel_route_lifecycle_trial_v81", "commit_route_lifecycle_trial_v81",
    "execution_document_v81", "load_route_lifecycle_checkpoint_v81",
    "make_q3_raw_split_receipt_v81", "method_identity_v81",
    "resolve_route_lifecycle_execution_v81",
    "resolve_route_lifecycle_state_v81",
    "rollback_route_lifecycle_trial_v81", "route_lifecycle_checkpoint_v81",
    "validate_q3_raw_split_receipt_v81",
    "validate_route_lifecycle_state_v81",
    "verify_frozen_predecessors_v81", "verify_sources_v81",
]
