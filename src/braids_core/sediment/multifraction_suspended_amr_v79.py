"""Bounded Q4 multi-fraction suspended wet/dry-AMR composition (v79).

This additive validation layer composes the accepted two/three-class v78
lifecycle with the authenticated suspended AMR ownership contract.  Its
deliberately narrow topology is one interior 2:1 parent/four-child group,
two persistent root-owned open-boundary leaves, and one coarse/fine interface
covered by two faces.  Two caller-supplied fine-integral slots exercise the
resident register replacement, but v79 does not authenticate or execute fine
hydraulic/lifecycle substeps and therefore does not admit recursive subcycling.

The CPU implementation in this module is an isolated oracle.  The actual-CUDA
owner is :mod:`braids_gpu.multifraction_suspended_amr_v79` and reuses the
accepted resident AMR transfer and reflux kernels without a CPU fallback.
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

from .multifraction_suspended_lifecycle_v78 import (
    INTEGRATOR,
    MAX_ACCEPTED_UPDATES,
    ROLE_CPU_ORACLE,
    ROLE_CUDA,
    MultifractionSuspendedExecutionV78,
    MultifractionSuspendedStateV78,
    MultifractionSuspendedTrialV78,
    _validate_trial as _validate_trial_v78,
    begin_multifraction_cpu_oracle_trial_v78,
    cancel_multifraction_trial_v78,
    commit_multifraction_trial_v78,
    execution_document_v78,
    load_multifraction_checkpoint_v78,
    make_implicit_fwave_midpoint_step_receipt_v78,
    multifraction_checkpoint_v78,
    resolve_multifraction_suspended_state_v78,
    rollback_multifraction_trial_v78,
    validate_multifraction_suspended_state_v78,
)
from .suspended_amr_contract import (
    SUSPENDED_AMR_SEMANTIC_ID,
    SuspendedAMRContract,
)


SCHEMA = 1
SCOPE_ID = "braids-q4-multifraction-suspended-amr-v79-v1"
CHECKPOINT_SCOPE_ID = SCOPE_ID + ":checkpoint"
STEP_RECEIPT_SCOPE_ID = SCOPE_ID + ":accepted-hydraulic-step"
SEMANTIC_ID = hashlib.sha256(SCOPE_ID.encode("ascii")).hexdigest()
TOPOLOGY_COARSE = "interior_parent"
TOPOLOGY_REFINED = "four_children"
TOPOLOGIES = (TOPOLOGY_COARSE, TOPOLOGY_REFINED)
FINE_INTEGRAL_SLOTS = 2
REFLUX_LEVELS = 1
AUTHENTICATED_FINE_SUBSTEPS = 0
NATIVE_STATE_PREFIX = 3
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class MultifractionSuspendedAMRV79Error(ValueError):
    """Fail-closed v79 identity, topology, receipt, or transaction error."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError, OverflowError) as error:
        raise MultifractionSuspendedAMRV79Error(
            "v79 value is not finite canonical JSON") from error


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(_canonical(list(array.shape)))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _frozen(value: object, shape: tuple[int, ...], name: str,
            *, nonnegative: bool = False) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise MultifractionSuspendedAMRV79Error(
            f"v79 {name} shape or finite values differ")
    if nonnegative and np.any(array < 0.0):
        raise MultifractionSuspendedAMRV79Error(
            f"v79 {name} must be nonnegative")
    result = np.ascontiguousarray(array)
    result.setflags(write=False)
    return result


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise MultifractionSuspendedAMRV79Error(
            f"v79 {name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True)
class MultifractionSuspendedAMRExecutionV79:
    role: str
    lifecycle_execution: MultifractionSuspendedExecutionV78
    amr_contract: SuspendedAMRContract
    child_cells: tuple[int, int, int, int]
    persistent_boundary_cells: tuple[int, int]
    reflux_cells: tuple[int, int, int, int]
    interface_faces: tuple[int, int]
    interface_coarse_sides: tuple[int, int]
    boundary_face_source_ids: tuple[int, ...]
    child_area_m2: tuple[float, float, float, float]
    parent_area_m2: float
    child_position_m: tuple[tuple[float, float], ...]
    child_bedrock_m: tuple[float, float, float, float]
    bulk_density_kg_m3: tuple[float, ...]
    fingerprint: str

    @property
    def fraction_count(self) -> int:
        return self.lifecycle_execution.fraction_count

    @property
    def cell_count(self) -> int:
        return self.lifecycle_execution.cell_count


def _execution_payload(value: MultifractionSuspendedAMRExecutionV79) -> dict:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "suspended_amr_semantic_id": SUSPENDED_AMR_SEMANTIC_ID,
        "role": value.role,
        "v78_execution": execution_document_v78(value.lifecycle_execution),
        "amr_contract": value.amr_contract.as_dict(),
        "child_cells": list(value.child_cells),
        "persistent_boundary_cells": list(value.persistent_boundary_cells),
        "reflux_cells": list(value.reflux_cells),
        "interface_faces": list(value.interface_faces),
        "interface_coarse_sides": list(value.interface_coarse_sides),
        "boundary_face_source_ids": list(value.boundary_face_source_ids),
        "child_area_m2": list(value.child_area_m2),
        "parent_area_m2": value.parent_area_m2,
        "child_position_m": [list(row) for row in value.child_position_m],
        "child_bedrock_m": list(value.child_bedrock_m),
        "bulk_density_kg_m3": list(value.bulk_density_kg_m3),
        "reflux_levels": REFLUX_LEVELS,
        "fine_integral_slots": FINE_INTEGRAL_SLOTS,
        "authenticated_fine_substeps": AUTHENTICATED_FINE_SUBSTEPS,
        "recursive_subcycling_admitted": False,
        "automatic_cpu_fallback": False,
        "accepted_update_limit": MAX_ACCEPTED_UPDATES,
    }


def resolve_multifraction_suspended_amr_execution_v79(
    lifecycle_execution: MultifractionSuspendedExecutionV78,
    amr_contract: SuspendedAMRContract, *,
    child_cells: tuple[int, int, int, int],
    persistent_boundary_cells: tuple[int, int],
    reflux_cells: tuple[int, int, int, int],
    interface_faces: tuple[int, int],
    interface_coarse_sides: tuple[int, int],
    boundary_face_source_ids: tuple[int, ...],
    child_area_m2: tuple[float, float, float, float],
    parent_area_m2: float,
    child_position_m: tuple[tuple[float, float], ...],
    child_bedrock_m: tuple[float, float, float, float],
    bulk_density_kg_m3: tuple[float, ...],
) -> MultifractionSuspendedAMRExecutionV79:
    """Resolve the only admitted bounded v79 topology."""
    execution_document_v78(lifecycle_execution)
    if lifecycle_execution.role not in (ROLE_CPU_ORACLE, ROLE_CUDA):
        raise MultifractionSuspendedAMRV79Error("v79 v78 role differs")
    if (not isinstance(amr_contract, SuspendedAMRContract) or
            not amr_contract.amr_enabled or amr_contract.amr_levels != 2 or
            amr_contract.amr_refinement_ratio != 2 or
            amr_contract.fraction_count != lifecycle_execution.fraction_count):
        raise MultifractionSuspendedAMRV79Error(
            "v79 requires the authenticated two-level 2:1 suspended AMR contract")
    nc = lifecycle_execution.cell_count
    nf = lifecycle_execution.face_count
    if (len(child_cells) != 4 or len(set(child_cells)) != 4 or
            any(type(i) is not int or not 0 <= i < nc for i in child_cells)):
        raise MultifractionSuspendedAMRV79Error(
            "v79 requires four distinct interior child cells")
    if (len(persistent_boundary_cells) != 2 or
            len(set(persistent_boundary_cells)) != 2 or
            set(persistent_boundary_cells) & set(child_cells) or
            any(type(i) is not int or not 0 <= i < nc
                for i in persistent_boundary_cells)):
        raise MultifractionSuspendedAMRV79Error(
            "v79 requires two persistent boundary cells outside the regrid group")
    if (len(reflux_cells) != 4 or reflux_cells[0] in child_cells or
            reflux_cells[1] not in child_cells or
            reflux_cells[2] not in child_cells or
            len(set(reflux_cells)) != 4 or
            any(type(i) is not int or not 0 <= i < nc for i in reflux_cells)):
        raise MultifractionSuspendedAMRV79Error(
            "v79 reflux cell order must be coarse, fine-neighbour, fine-owner, spare")
    if (len(interface_faces) != 2 or len(set(interface_faces)) != 2 or
            any(type(i) is not int or not 0 <= i < nf for i in interface_faces) or
            tuple(interface_coarse_sides) != (0, 1)):
        raise MultifractionSuspendedAMRV79Error(
            "v79 requires two single-owned interface faces in owner/neighbour order")
    boundary = lifecycle_execution.base_execution.boundary_index
    owner = lifecycle_execution.base_execution.owner
    neighbour = lifecycle_execution.base_execution.neighbour
    if any(boundary[i] >= 0 for i in interface_faces):
        raise MultifractionSuspendedAMRV79Error(
            "v79 coarse/fine interface faces must be interior")
    coarse, fine_neighbour, fine_owner, _ = reflux_cells
    face0, face1 = interface_faces
    if not (owner[face0] == coarse and neighbour[face0] == fine_neighbour and
            owner[face1] == fine_owner and neighbour[face1] == coarse):
        raise MultifractionSuspendedAMRV79Error(
            "v79 interface ownership does not match the authenticated reflux order")
    ids = tuple(boundary_face_source_ids)
    if len(ids) != nf:
        raise MultifractionSuspendedAMRV79Error(
            "v79 boundary source-ID inventory differs")
    open_faces = np.flatnonzero(boundary >= 0)
    open_ids = [ids[index] for index in open_faces]
    if (any(type(value) is not int or value < 0 for value in open_ids) or
            len(set(open_ids)) != len(open_ids) or
            any(ids[index] != -1 for index in range(nf)
                if boundary[index] < 0)):
        raise MultifractionSuspendedAMRV79Error(
            "v79 stable source IDs must map one-to-one to open faces only")
    if any(owner[index] not in persistent_boundary_cells for index in open_faces):
        raise MultifractionSuspendedAMRV79Error(
            "v79 open-boundary owners must remain outside the regrid group")
    areas = np.asarray(child_area_m2, dtype=np.float64)
    positions = np.asarray(child_position_m, dtype=np.float64)
    rock = np.asarray(child_bedrock_m, dtype=np.float64)
    bulk = np.asarray(bulk_density_kg_m3, dtype=np.float64)
    if (areas.shape != (4,) or positions.shape != (4, 2) or rock.shape != (4,) or
            bulk.shape != (lifecycle_execution.fraction_count,) or
            not all(np.isfinite(value).all()
                    for value in (areas, positions, rock, bulk)) or
            np.any(areas <= 0.0) or np.any(bulk <= 0.0) or
            not math.isfinite(parent_area_m2) or parent_area_m2 <= 0.0 or
            not np.isclose(np.sum(areas), parent_area_m2,
                           rtol=1e-12, atol=1e-13) or
            not np.allclose(np.sum(areas[:, None]*positions, axis=0), 0.0,
                            rtol=0.0, atol=1e-12)):
        raise MultifractionSuspendedAMRV79Error(
            "v79 four-child geometry is not a conservative centred partition")
    if not np.array_equal(rock, np.full(4, rock[0])):
        raise MultifractionSuspendedAMRV79Error(
            "v79 zero-gradient split requires uniform child bedrock")
    actual_child_area = lifecycle_execution.base_execution.cell_area_m2[
        np.asarray(child_cells)]
    if not np.array_equal(actual_child_area, areas):
        raise MultifractionSuspendedAMRV79Error(
            "v79 lifecycle child areas and transfer geometry differ")
    placeholder = MultifractionSuspendedAMRExecutionV79(
        lifecycle_execution.role, lifecycle_execution, amr_contract,
        tuple(child_cells), tuple(persistent_boundary_cells), tuple(reflux_cells),
        tuple(interface_faces), tuple(interface_coarse_sides), ids,
        tuple(map(float, areas)), float(parent_area_m2),
        tuple(tuple(map(float, row)) for row in positions),
        tuple(map(float, rock)), tuple(map(float, bulk)), "")
    return MultifractionSuspendedAMRExecutionV79(
        **{**placeholder.__dict__,
           "fingerprint": _fingerprint(_execution_payload(placeholder))})


def execution_document_v79(value: MultifractionSuspendedAMRExecutionV79) -> dict:
    if not isinstance(value, MultifractionSuspendedAMRExecutionV79):
        raise MultifractionSuspendedAMRV79Error("v79 execution object differs")
    body = _execution_payload(value)
    if value.fingerprint != _fingerprint(body):
        raise MultifractionSuspendedAMRV79Error(
            "v79 execution fingerprint differs")
    return {**body, "fingerprint": value.fingerprint}


def native_rows_v79(execution: MultifractionSuspendedAMRExecutionV79,
                     lifecycle: MultifractionSuspendedStateV78,
                     depth_m: np.ndarray,
                     momentum_m2_s: np.ndarray) -> np.ndarray:
    rows = np.zeros((execution.cell_count, 3+2*execution.fraction_count),
                    dtype=np.float64)
    rows[:, 0] = depth_m
    rows[:, 1:3] = momentum_m2_s.T
    ng = execution.fraction_count
    rows[:, 3:3+ng] = lifecycle.suspended_mass_kg_m2.T
    rows[:, 3+ng:3+2*ng] = lifecycle.bed_mass_kg_m2.T
    return rows


def _restrict_children(execution: MultifractionSuspendedAMRExecutionV79,
                       rows: np.ndarray) -> np.ndarray:
    cells = np.asarray(execution.child_cells)
    weights = np.asarray(execution.child_area_m2)
    return np.sum(weights[:, None]*rows[cells], axis=0,
                  dtype=np.float64)/execution.parent_area_m2


def _lifecycle_with_rows(execution: MultifractionSuspendedAMRExecutionV79,
                         source: MultifractionSuspendedStateV78,
                         rows: np.ndarray) -> MultifractionSuspendedStateV78:
    lifecycle = source.lifecycle
    fields = {
        name: getattr(lifecycle, name) for name in lifecycle.__dataclass_fields__
        if name not in ("execution_fingerprint", "state_sha256")
    }
    ng = execution.fraction_count
    fields["suspended_mass_kg_m2"] = rows[:, 3:3+ng].T
    fields["bed_mass_kg_m2"] = rows[:, 3+ng:3+2*ng].T
    return resolve_multifraction_suspended_state_v78(
        execution.lifecycle_execution, **fields)


@dataclass(frozen=True)
class MultifractionSuspendedAMRStateV79:
    execution_fingerprint: str
    topology: str
    lifecycle: MultifractionSuspendedStateV78
    depth_m: np.ndarray
    momentum_m2_s: np.ndarray
    coarse_parent_row: np.ndarray
    accepted_fine_flux_register_kg: np.ndarray
    topology_events: tuple[str, ...]
    state_sha256: str

    @property
    def accepted_updates(self) -> int:
        return self.lifecycle.accepted_updates

    @property
    def time_s(self) -> float:
        return self.lifecycle.time_s


def _state_token(value: MultifractionSuspendedAMRStateV79) -> str:
    return _fingerprint({
        "execution_fingerprint": value.execution_fingerprint,
        "topology": value.topology,
        "lifecycle_state_sha256": value.lifecycle.state_sha256,
        "arrays_sha256": [_array_digest(array) for array in (
            value.depth_m, value.momentum_m2_s, value.coarse_parent_row,
            value.accepted_fine_flux_register_kg)],
        "topology_events": list(value.topology_events),
    })


def _wrap_state(execution: MultifractionSuspendedAMRExecutionV79, *,
                topology: str, lifecycle: MultifractionSuspendedStateV78,
                depth_m: object, momentum_m2_s: object,
                coarse_parent_row: object,
                accepted_fine_flux_register_kg: object,
                topology_events: tuple[str, ...]) -> MultifractionSuspendedAMRStateV79:
    execution_document_v79(execution)
    validate_multifraction_suspended_state_v78(
        lifecycle, execution.lifecycle_execution)
    if topology not in TOPOLOGIES:
        raise MultifractionSuspendedAMRV79Error("v79 topology differs")
    nc, ng = execution.cell_count, execution.fraction_count
    ns = 3+2*ng
    depth = _frozen(depth_m, (nc,), "depth", nonnegative=True)
    momentum = _frozen(momentum_m2_s, (2, nc), "momentum")
    parent = _frozen(coarse_parent_row, (ns,), "coarse parent row",
                     nonnegative=False)
    register = _frozen(accepted_fine_flux_register_kg,
                       (2, 2, ns), "fine flux register")
    if (not isinstance(topology_events, tuple) or not topology_events or
            any(not isinstance(item, str) or not item for item in topology_events)):
        raise MultifractionSuspendedAMRV79Error(
            "v79 topology event history differs")
    rows = native_rows_v79(execution, lifecycle, depth, momentum)
    if parent[0] < 0.0 or np.any(parent[3:] < 0.0):
        raise MultifractionSuspendedAMRV79Error(
            "v79 coarse parent water and sediment rows must be nonnegative")
    dry = depth <= execution.amr_contract.hydraulic_dry_tolerance_m
    if (np.any(momentum[:, dry] != 0.0) or
            np.any(lifecycle.suspended_mass_kg_m2[:, dry] != 0.0)):
        raise MultifractionSuspendedAMRV79Error(
            "v79 dry leaves retain momentum or suspended mass")
    if topology == TOPOLOGY_COARSE:
        expected = _restrict_children(execution, rows)
        if not np.allclose(parent, expected, rtol=2e-13, atol=2e-14):
            raise MultifractionSuspendedAMRV79Error(
                "v79 coarse parent differs from conservative child restriction")
    placeholder = MultifractionSuspendedAMRStateV79(
        execution.fingerprint, topology, lifecycle, depth, momentum, parent,
        register, topology_events, "")
    return MultifractionSuspendedAMRStateV79(
        **{**placeholder.__dict__, "state_sha256": _state_token(placeholder)})


def resolve_multifraction_suspended_amr_state_v79(
    execution: MultifractionSuspendedAMRExecutionV79, *,
    topology: str, lifecycle: MultifractionSuspendedStateV78,
    depth_m: object, momentum_m2_s: object,
    coarse_parent_row: object | None = None,
    accepted_fine_flux_register_kg: object | None = None,
    topology_events: tuple[str, ...] = ("initial_coarse",),
) -> MultifractionSuspendedAMRStateV79:
    nc, ng = execution.cell_count, execution.fraction_count
    depth = np.asarray(depth_m, dtype=np.float64)
    momentum = np.asarray(momentum_m2_s, dtype=np.float64)
    rows = native_rows_v79(execution, lifecycle, depth, momentum)
    parent = (_restrict_children(execution, rows) if coarse_parent_row is None
              else coarse_parent_row)
    register = (np.zeros((2, 2, 3+2*ng), dtype=np.float64)
                if accepted_fine_flux_register_kg is None
                else accepted_fine_flux_register_kg)
    return _wrap_state(
        execution, topology=topology, lifecycle=lifecycle, depth_m=depth,
        momentum_m2_s=momentum, coarse_parent_row=parent,
        accepted_fine_flux_register_kg=register,
        topology_events=topology_events)


def validate_multifraction_suspended_amr_state_v79(
    state: MultifractionSuspendedAMRStateV79,
    execution: MultifractionSuspendedAMRExecutionV79,
) -> None:
    if (not isinstance(state, MultifractionSuspendedAMRStateV79) or
            state.execution_fingerprint != execution.fingerprint or
            state.state_sha256 != _state_token(state)):
        raise MultifractionSuspendedAMRV79Error(
            "v79 state is stale, mutated, or belongs to another execution")
    _wrap_state(
        execution, topology=state.topology, lifecycle=state.lifecycle,
        depth_m=state.depth_m, momentum_m2_s=state.momentum_m2_s,
        coarse_parent_row=state.coarse_parent_row,
        accepted_fine_flux_register_kg=state.accepted_fine_flux_register_kg,
        topology_events=state.topology_events)


def make_implicit_fwave_midpoint_amr_receipt_v79(
    execution: MultifractionSuspendedAMRExecutionV79,
    parent: MultifractionSuspendedAMRStateV79, *,
    accepted_step_sha256: str, dt_s: float,
) -> dict:
    validate_multifraction_suspended_amr_state_v79(parent, execution)
    step_sha = _require_sha256(accepted_step_sha256,
                               "accepted hydraulic step digest")
    if (not math.isfinite(dt_s) or dt_s <= 0.0 or
            parent.topology != TOPOLOGY_REFINED or
            parent.accepted_updates >= MAX_ACCEPTED_UPDATES):
        raise MultifractionSuspendedAMRV79Error(
            "v79 receipt requires a refined bounded positive update")
    nested = make_implicit_fwave_midpoint_step_receipt_v78(
        hydraulic_execution_sha256=
            execution.lifecycle_execution.hydraulic_execution_sha256,
        accepted_step_sha256=step_sha,
        accepted_update=parent.accepted_updates+1,
        time_before_s=parent.time_s, dt_s=float(dt_s))
    body = {
        "schema": SCHEMA,
        "scope_id": STEP_RECEIPT_SCOPE_ID,
        "execution_fingerprint": execution.fingerprint,
        "integrator": INTEGRATOR,
        "hydraulic_backend": "implicit_fwave_midpoint_gpu",
        "amr": True,
        "accepted": True,
        "automatic_cpu_fallback": False,
        "topology": parent.topology,
        "reflux_levels": REFLUX_LEVELS,
        "fine_integral_slots": FINE_INTEGRAL_SLOTS,
        "authenticated_fine_substeps": AUTHENTICATED_FINE_SUBSTEPS,
        "recursive_subcycling_admitted": False,
        "accepted_update": parent.accepted_updates+1,
        "time_before_s": parent.time_s,
        "dt_s": float(dt_s),
        "time_after_s": parent.time_s+float(dt_s),
        "accepted_step_sha256": step_sha,
        "level_local_v78_receipt": nested,
    }
    return {**body, "fingerprint": _fingerprint(body)}


def _validate_receipt(execution, parent, receipt, dt_s) -> dict:
    expected = {
        "schema", "scope_id", "execution_fingerprint", "integrator",
        "hydraulic_backend", "amr", "accepted", "automatic_cpu_fallback",
        "topology", "reflux_levels", "fine_integral_slots",
        "authenticated_fine_substeps", "recursive_subcycling_admitted",
        "accepted_update",
        "time_before_s", "dt_s", "time_after_s", "accepted_step_sha256",
        "level_local_v78_receipt", "fingerprint",
    }
    if not isinstance(receipt, Mapping) or set(receipt) != expected:
        raise MultifractionSuspendedAMRV79Error(
            "v79 hydraulic receipt inventory differs")
    body = {key: receipt[key] for key in receipt if key != "fingerprint"}
    if (receipt["fingerprint"] != _fingerprint(body) or
            receipt["schema"] != SCHEMA or
            receipt["scope_id"] != STEP_RECEIPT_SCOPE_ID or
            receipt["execution_fingerprint"] != execution.fingerprint or
            receipt["integrator"] != INTEGRATOR or
            receipt["hydraulic_backend"] != "implicit_fwave_midpoint_gpu" or
            receipt["amr"] is not True or receipt["accepted"] is not True or
            receipt["automatic_cpu_fallback"] is not False or
            receipt["topology"] != TOPOLOGY_REFINED or
            receipt["reflux_levels"] != REFLUX_LEVELS or
            receipt["fine_integral_slots"] != FINE_INTEGRAL_SLOTS or
            receipt["authenticated_fine_substeps"] !=
            AUTHENTICATED_FINE_SUBSTEPS or
            receipt["recursive_subcycling_admitted"] is not False or
            receipt["accepted_update"] != parent.accepted_updates+1 or
            receipt["time_before_s"] != parent.time_s or
            receipt["dt_s"] != float(dt_s) or
            receipt["time_after_s"] != parent.time_s+float(dt_s)):
        raise MultifractionSuspendedAMRV79Error(
            "v79 requires the current accepted implicit_fwave_midpoint AMR receipt")
    expected_nested = make_implicit_fwave_midpoint_step_receipt_v78(
        hydraulic_execution_sha256=
            execution.lifecycle_execution.hydraulic_execution_sha256,
        accepted_step_sha256=receipt["accepted_step_sha256"],
        accepted_update=parent.accepted_updates+1,
        time_before_s=parent.time_s, dt_s=float(dt_s))
    if receipt["level_local_v78_receipt"] != expected_nested:
        raise MultifractionSuspendedAMRV79Error(
            "v79 level-local v78 receipt differs")
    return dict(receipt)


@dataclass(frozen=True)
class MultifractionSuspendedAMRTrialV79:
    execution_fingerprint: str
    parent_state_sha256: str
    operation: str
    candidate: MultifractionSuspendedAMRStateV79
    lifecycle_trial: MultifractionSuspendedTrialV78 | None
    coarse_advective_integral_kg: np.ndarray
    coarse_diffusive_integral_kg: np.ndarray
    fine_advective_slots_kg: np.ndarray
    fine_diffusive_slots_kg: np.ndarray
    fine_register_kg: np.ndarray
    reflux_before_rows: np.ndarray
    reflux_after_rows: np.ndarray
    receipt_fingerprint: str | None
    backend: str
    trial_sha256: str


def _trial_token(value: MultifractionSuspendedAMRTrialV79) -> str:
    arrays = (
        value.coarse_advective_integral_kg,
        value.coarse_diffusive_integral_kg,
        value.fine_advective_slots_kg,
        value.fine_diffusive_slots_kg,
        value.fine_register_kg,
        value.reflux_before_rows,
        value.reflux_after_rows,
    )
    return _fingerprint({
        "execution_fingerprint": value.execution_fingerprint,
        "parent_state_sha256": value.parent_state_sha256,
        "operation": value.operation,
        "candidate_state_sha256": value.candidate.state_sha256,
        "lifecycle_trial_sha256": (None if value.lifecycle_trial is None else
                                   value.lifecycle_trial.trial_sha256),
        "arrays_sha256": [_array_digest(array) for array in arrays],
        "receipt_fingerprint": value.receipt_fingerprint,
        "backend": value.backend,
    })


def _empty_trial_arrays(execution):
    ng, ns = execution.fraction_count, 3+2*execution.fraction_count
    return (
        np.zeros((2, ng)), np.zeros((2, ng)),
        np.zeros((2, 2, ng)), np.zeros((2, 2, ng)),
        np.zeros((2, 2, ns)), np.zeros((4, ns)), np.zeros((4, ns)),
    )


def _build_trial(execution, parent, operation, candidate, *,
                 lifecycle_trial=None, arrays=None, receipt_fingerprint=None,
                 backend="cpu_oracle"):
    values = _empty_trial_arrays(execution) if arrays is None else arrays
    shapes = ((2, execution.fraction_count), (2, execution.fraction_count),
              (2, 2, execution.fraction_count),
              (2, 2, execution.fraction_count),
              (2, 2, 3+2*execution.fraction_count),
              (4, 3+2*execution.fraction_count),
              (4, 3+2*execution.fraction_count))
    frozen = tuple(_frozen(value, shape, "trial evidence")
                   for value, shape in zip(values, shapes))
    placeholder = MultifractionSuspendedAMRTrialV79(
        execution.fingerprint, parent.state_sha256, operation, candidate,
        lifecycle_trial, *frozen, receipt_fingerprint, backend, "")
    return MultifractionSuspendedAMRTrialV79(
        **{**placeholder.__dict__, "trial_sha256": _trial_token(placeholder)})


def begin_refine_cpu_oracle_v79(execution, parent, *,
                                gradient: object | None = None):
    validate_multifraction_suspended_amr_state_v79(parent, execution)
    if execution.role != ROLE_CPU_ORACLE or parent.topology != TOPOLOGY_COARSE:
        raise MultifractionSuspendedAMRV79Error(
            "v79 CPU refine requires the coarse CPU-oracle state")
    primitives = 4+execution.fraction_count
    grad = np.zeros((1, primitives, 2), dtype=np.float64) if gradient is None \
        else np.asarray(gradient, dtype=np.float64)
    if grad.shape != (1, primitives, 2) or np.any(grad != 0.0):
        raise MultifractionSuspendedAMRV79Error(
            "v79 bounded CPU oracle admits only the authenticated zero-gradient split")
    rows = native_rows_v79(execution, parent.lifecycle,
                           parent.depth_m, parent.momentum_m2_s)
    children = rows[np.asarray(execution.child_cells)]
    if not np.array_equal(children, np.broadcast_to(
            parent.coarse_parent_row, children.shape)):
        raise MultifractionSuspendedAMRV79Error(
            "v79 bounded zero-gradient refinement requires a uniform child cache")
    rows[np.asarray(execution.child_cells)] = parent.coarse_parent_row
    lifecycle = _lifecycle_with_rows(execution, parent.lifecycle, rows)
    candidate = _wrap_state(
        execution, topology=TOPOLOGY_REFINED, lifecycle=lifecycle,
        depth_m=rows[:, 0], momentum_m2_s=rows[:, 1:3].T,
        coarse_parent_row=parent.coarse_parent_row,
        accepted_fine_flux_register_kg=parent.accepted_fine_flux_register_kg,
        topology_events=parent.topology_events+("refine",))
    return _build_trial(execution, parent, "refine", candidate)


def _expanded_integrals(execution, coarse_adv, coarse_diff,
                        fine_adv, fine_diff):
    ng, ns = execution.fraction_count, 3+2*execution.fraction_count
    ca = _frozen(coarse_adv, (2, ng), "coarse advective integral")
    cd = _frozen(coarse_diff, (2, ng), "coarse diffusive integral")
    fa = _frozen(fine_adv, (2, 2, ng), "fine advective integral slots")
    fd = _frozen(fine_diff, (2, 2, ng), "fine diffusive integral slots")
    coarse = np.zeros((2, 2, ns), dtype=np.float64)
    fine = np.zeros((2, 2, 2, ns), dtype=np.float64)
    coarse[:, :, 3:3+ng] = (ca+cd)[:, None, :]
    fine[:, :, :, 3:3+ng] = (fa+fd)[:, :, None, :]
    return ca, cd, fa, fd, coarse, fine


def _cpu_reflux(execution, rows, coarse, fine):
    area = np.asarray([execution.lifecycle_execution.base_execution.cell_area_m2[i]
                       for i in execution.reflux_cells])
    before = np.ascontiguousarray(rows[np.asarray(execution.reflux_cells)].copy())
    pre = before.copy()
    owner = (0, 2)
    neighbour = (1, 0)
    for face in range(2):
        if owner[face] == 0:
            pre[0] -= coarse[face, 0]/area[0]
        if neighbour[face] == 0:
            pre[0] += coarse[face, 1]/area[0]
    for step in range(2):
        for face in range(2):
            if owner[face] != 0:
                pre[owner[face]] -= fine[step, face, 0]/area[owner[face]]
            if neighbour[face] != 0:
                pre[neighbour[face]] += fine[step, face, 1]/area[neighbour[face]]
    register = np.sum(fine, axis=0, dtype=np.float64)
    after = pre.copy()
    for face, side in enumerate((0, 1)):
        delta = coarse[face, side]-register[face, side]
        after[0] += (delta if side == 0 else -delta)/area[0]
    residual = np.sum(area[:, None]*(after-before), axis=0, dtype=np.float64)
    scale = np.maximum(1.0, np.sum(np.abs(area[:, None]*before), axis=0))
    if np.any(np.abs(residual) > 4096*np.finfo(float).eps*scale):
        raise MultifractionSuspendedAMRV79Error(
            "v79 CPU reflux did not conserve every native row")
    if np.any(after[:, 0] < 0.0) or np.any(after[:, 3:] < 0.0):
        raise MultifractionSuspendedAMRV79Error(
            "v79 CPU reflux would require positivity repair")
    result = rows.copy()
    result[np.asarray(execution.reflux_cells)] = after
    return result, before, after, register


def begin_transport_cpu_oracle_v79(
    execution, parent, *, accepted_step_receipt: Mapping[str, object],
    coarse_advective_integral_kg: object,
    coarse_diffusive_integral_kg: object,
    fine_advective_slots_kg: object,
    fine_diffusive_slots_kg: object,
    **lifecycle_inputs,
):
    validate_multifraction_suspended_amr_state_v79(parent, execution)
    if execution.role != ROLE_CPU_ORACLE or parent.topology != TOPOLOGY_REFINED:
        raise MultifractionSuspendedAMRV79Error(
            "v79 CPU transport requires the refined CPU-oracle state")
    dt_s = float(lifecycle_inputs.get("dt_s", float("nan")))
    receipt = _validate_receipt(execution, parent,
                                accepted_step_receipt, dt_s)
    water = np.asarray(lifecycle_inputs.get("accepted_water_flux_m2_s"))
    active = np.asarray(lifecycle_inputs.get("transport_face_active"))
    if (water.shape != (execution.lifecycle_execution.face_count,) or
            active.shape != water.shape or
            np.any(water[np.asarray(execution.interface_faces)] != 0.0) or
            np.any(active[np.asarray(execution.interface_faces)])):
        raise MultifractionSuspendedAMRV79Error(
            "v79 level-local lifecycle must leave registered interface faces single-owned")
    base_trial = begin_multifraction_cpu_oracle_trial_v78(
        execution.lifecycle_execution, parent.lifecycle,
        accepted_step_receipt=receipt["level_local_v78_receipt"],
        **lifecycle_inputs)
    base_state = commit_multifraction_trial_v78(
        execution.lifecycle_execution, parent.lifecycle, base_trial)
    depth = np.asarray(lifecycle_inputs["depth_after_m"], dtype=np.float64)
    momentum = np.asarray(lifecycle_inputs["endpoint_momentum_m2_s"],
                          dtype=np.float64)
    rows = native_rows_v79(execution, base_state, depth, momentum)
    ca, cd, fa, fd, coarse, fine = _expanded_integrals(
        execution, coarse_advective_integral_kg,
        coarse_diffusive_integral_kg, fine_advective_slots_kg,
        fine_diffusive_slots_kg)
    rows, before, after, register = _cpu_reflux(
        execution, rows, coarse, fine)
    lifecycle = _lifecycle_with_rows(execution, base_state, rows)
    parent_row = _restrict_children(execution, rows)
    candidate = _wrap_state(
        execution, topology=TOPOLOGY_REFINED, lifecycle=lifecycle,
        depth_m=rows[:, 0], momentum_m2_s=rows[:, 1:3].T,
        coarse_parent_row=parent_row,
        accepted_fine_flux_register_kg=register,
        topology_events=parent.topology_events+("transport_exchange_reflux",))
    arrays = (ca, cd, fa, fd, register, before, after)
    return _build_trial(
        execution, parent, "transport", candidate,
        lifecycle_trial=base_trial, arrays=arrays,
        receipt_fingerprint=receipt["fingerprint"])


def begin_coarsen_cpu_oracle_v79(execution, parent):
    validate_multifraction_suspended_amr_state_v79(parent, execution)
    if execution.role != ROLE_CPU_ORACLE or parent.topology != TOPOLOGY_REFINED:
        raise MultifractionSuspendedAMRV79Error(
            "v79 CPU coarsen requires the refined CPU-oracle state")
    rows = native_rows_v79(execution, parent.lifecycle,
                           parent.depth_m, parent.momentum_m2_s)
    restricted = _restrict_children(execution, rows)
    candidate = _wrap_state(
        execution, topology=TOPOLOGY_COARSE, lifecycle=parent.lifecycle,
        depth_m=parent.depth_m, momentum_m2_s=parent.momentum_m2_s,
        coarse_parent_row=restricted,
        accepted_fine_flux_register_kg=parent.accepted_fine_flux_register_kg,
        topology_events=parent.topology_events+("coarsen",))
    return _build_trial(execution, parent, "coarsen", candidate)


def _validate_trial(execution, parent, trial):
    validate_multifraction_suspended_amr_state_v79(parent, execution)
    if (not isinstance(trial, MultifractionSuspendedAMRTrialV79) or
            trial.execution_fingerprint != execution.fingerprint or
            trial.parent_state_sha256 != parent.state_sha256 or
            trial.trial_sha256 != _trial_token(trial)):
        raise MultifractionSuspendedAMRV79Error(
            "v79 trial is stale, mutated, or belongs to another parent")
    validate_multifraction_suspended_amr_state_v79(trial.candidate, execution)
    if trial.operation == "transport":
        if trial.lifecycle_trial is None:
            raise MultifractionSuspendedAMRV79Error(
                "v79 transport trial is missing its nested v78 owner")
        # Authenticate the complete nested object, not only its stored token,
        # before a resident reflux or lifecycle owner can publish.
        _validate_trial_v78(execution.lifecycle_execution, parent.lifecycle,
                            trial.lifecycle_trial)
    elif trial.operation in ("refine", "coarsen"):
        if trial.lifecycle_trial is not None:
            raise MultifractionSuspendedAMRV79Error(
                "v79 topology trial unexpectedly contains a lifecycle owner")
    else:
        raise MultifractionSuspendedAMRV79Error(
            "v79 trial operation is outside the bounded sequence")


def commit_multifraction_suspended_amr_trial_v79(execution, parent, trial):
    _validate_trial(execution, parent, trial)
    return trial.candidate


def rollback_multifraction_suspended_amr_trial_v79(execution, parent, trial):
    _validate_trial(execution, parent, trial)
    return parent


def cancel_multifraction_suspended_amr_trial_v79(execution, parent, trial):
    return rollback_multifraction_suspended_amr_trial_v79(
        execution, parent, trial)


def multifraction_suspended_amr_checkpoint_v79(execution, state) -> dict:
    validate_multifraction_suspended_amr_state_v79(state, execution)
    body = {
        "schema": SCHEMA,
        "scope_id": CHECKPOINT_SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "execution_fingerprint": execution.fingerprint,
        "integrator": INTEGRATOR,
        "transaction_schema": "accepted_topology_state_register_memory_v1",
        "topology": state.topology,
        "v78_checkpoint": multifraction_checkpoint_v78(
            execution.lifecycle_execution, state.lifecycle),
        "depth_m": state.depth_m.tolist(),
        "momentum_m2_s": state.momentum_m2_s.tolist(),
        "coarse_parent_row": state.coarse_parent_row.tolist(),
        "accepted_fine_flux_register_kg":
            state.accepted_fine_flux_register_kg.tolist(),
        "boundary_face_source_ids": list(execution.boundary_face_source_ids),
        "topology_events": list(state.topology_events),
        "state_sha256": state.state_sha256,
    }
    return {**body, "fingerprint": _fingerprint(body)}


def load_multifraction_suspended_amr_checkpoint_v79(execution, document):
    expected = {
        "schema", "scope_id", "semantic_id", "execution_fingerprint",
        "integrator", "transaction_schema", "topology", "v78_checkpoint",
        "depth_m", "momentum_m2_s", "coarse_parent_row",
        "accepted_fine_flux_register_kg", "boundary_face_source_ids",
        "topology_events", "state_sha256", "fingerprint",
    }
    if not isinstance(document, Mapping) or set(document) != expected:
        raise MultifractionSuspendedAMRV79Error(
            "v79 checkpoint inventory differs")
    body = {key: document[key] for key in document if key != "fingerprint"}
    if (document["fingerprint"] != _fingerprint(body) or
            document["schema"] != SCHEMA or
            document["scope_id"] != CHECKPOINT_SCOPE_ID or
            document["semantic_id"] != SEMANTIC_ID or
            document["execution_fingerprint"] != execution.fingerprint or
            document["integrator"] != INTEGRATOR or
            document["transaction_schema"] !=
            "accepted_topology_state_register_memory_v1" or
            tuple(document["boundary_face_source_ids"]) !=
            execution.boundary_face_source_ids):
        raise MultifractionSuspendedAMRV79Error(
            "v79 checkpoint identity or stable boundary face map differs")
    lifecycle = load_multifraction_checkpoint_v78(
        execution.lifecycle_execution, document["v78_checkpoint"])
    state = _wrap_state(
        execution, topology=document["topology"], lifecycle=lifecycle,
        depth_m=document["depth_m"],
        momentum_m2_s=document["momentum_m2_s"],
        coarse_parent_row=document["coarse_parent_row"],
        accepted_fine_flux_register_kg=
            document["accepted_fine_flux_register_kg"],
        topology_events=tuple(document["topology_events"]))
    if document["state_sha256"] != state.state_sha256:
        raise MultifractionSuspendedAMRV79Error(
            "v79 checkpoint accepted state digest differs")
    return state


def classwise_combined_ledger_v79(execution, state):
    validate_multifraction_suspended_amr_state_v79(state, execution)
    base = state.lifecycle.lifecycle
    area = execution.lifecycle_execution.base_execution.cell_area_m2[None, :]
    current = np.sum(
        (base.suspended_mass_kg_m2+base.bed_mass_kg_m2)*area, axis=1)
    boundary = np.sum(base.advective_boundary_ledger_kg+
                      base.diffusive_boundary_ledger_kg, axis=1)
    reference = base.reference_water_mass_kg+base.reference_bed_mass_kg
    residual = current-reference-boundary
    return current, reference, boundary, residual


def method_identity_v79() -> dict:
    return {
        "scope_id": SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "builds_additively_on": [
            "braids-q4-multifraction-suspended-lifecycle-v78-v1",
            "braids-suspended-wetdry-amr-ownership-v1",
            "resident-amr-transfer-and-two-slot-reflux",
        ],
        "fraction_scope": "2..3 noncohesive sand/gravel classes",
        "integrator": INTEGRATOR,
        "strict_accepted_step_receipt": True,
        "amr_scope": "one interior 2:1 parent/four-child group",
        "recursive_scope": "not admitted; two caller-supplied fine-integral register slots only",
        "authenticated_fine_substeps": AUTHENTICATED_FINE_SUBSTEPS,
        "recursive_subcycling_gap": (
            "two accepted fine implicit_fwave_midpoint step receipts bound to "
            "their flux integrals plus level-local lifecycle execution are absent"),
        "arbitrary_depth_recursive_claimed": False,
        "open_boundary_memory": (
            "root_owned_stable_face_IDs_outside_regrid_group_no_averaging"),
        "shared_face_owner": (
            "one physical advective_plus_diffusive integral per interface face"),
        "reflux": (
            "two-slot caller-supplied fine advective_plus_diffusive register "
            "replaces the coarse contribution"),
        "transaction": (
            "topology_state_register_boundary_memory_ledgers_atomic_exact_restart"),
        "accepted_update_limit": MAX_ACCEPTED_UPDATES,
        "automatic_cpu_fallback": False,
        "delft3d_source_modified": False,
        "long_run_claimed": False,
        "production_admission_claimed": False,
    }


FROZEN_V78_SHA256 = {
    # Filled from the accepted v78 bytes; changing any value requires a new
    # composition identity rather than silently moving the prerequisite.
    "braids_core/sediment/multifraction_suspended_lifecycle_v78.py":
        "eaba4861d38369eaf740803c43a722e10a0ee70bb9a2e1dcf56bddba2a7e10dd",
    "braids_gpu/multifraction_suspended_lifecycle_v78.py":
        "e4fdd4aa5244d0ceff7f7b5479cdfbe4bc2cd9c5344f731cecf5351180facc6b",
    "tests/test_q4_multifraction_suspended_lifecycle_v78.py":
        "d8bb11b79f20f624e10a37e910b50e41768a6b9168060320ba4354e716916c74",
    "tests/test_q4_multifraction_suspended_cuda_v78.py":
        "a21879699851eb9d325f34cb85f04fd00d32e466e1bc65e7353d9124f1fad91e",
}


AMR_DEPENDENCY_SHA256 = {
    "braids_core/sediment/suspended_amr_contract.py":
        "fd531c3688458b6cc4391847e00919b95f54c8ddd5f7f662bb18591a69e9c601",
    "braids_support/gpu_fwave/amr_transfer.py":
        "5c14ce754490506ccc22c887182be0b47dbf46d8046e95bd05db6bce2c5362a9",
    "braids_support/gpu_fwave/amr_transfer.cu":
        "f787de4437da9067eb4a7a07a3870d7772376331c9d060d847b9ec836f6b8e12",
    "braids_support/gpu_fwave/amr_reflux.py":
        "f001ea7a88c5a4fc277292c322a4e56a16a1ceb311de3f9f10361a6c84c8e64f",
    "braids_support/gpu_fwave/amr_reflux.cu":
        "318ef70fa402e6c42ff3b72ddedee59ec104033de2a4dcde0204559df38b53a5",
}


def _verify_hashes(root: str | Path, expected: Mapping[str, str], label: str):
    root = Path(root)
    result = {}
    for relative, digest in expected.items():
        path = root/relative
        if not path.is_file():
            raise MultifractionSuspendedAMRV79Error(
                f"v79 {label} source is missing: {relative}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != digest:
            raise MultifractionSuspendedAMRV79Error(
                f"v79 {label} source changed: {relative}")
        result[relative] = actual
    return result


def verify_frozen_v78_v79(root: str | Path):
    return _verify_hashes(root, FROZEN_V78_SHA256, "frozen v78")


def verify_amr_dependencies_v79(root: str | Path):
    return _verify_hashes(root, AMR_DEPENDENCY_SHA256, "AMR dependency")


__all__ = [
    "AMR_DEPENDENCY_SHA256", "AUTHENTICATED_FINE_SUBSTEPS",
    "FINE_INTEGRAL_SLOTS", "FROZEN_V78_SHA256",
    "MAX_ACCEPTED_UPDATES", "MultifractionSuspendedAMRExecutionV79",
    "MultifractionSuspendedAMRStateV79", "MultifractionSuspendedAMRTrialV79",
    "MultifractionSuspendedAMRV79Error", "REFLUX_LEVELS", "SCOPE_ID",
    "SEMANTIC_ID", "TOPOLOGY_COARSE", "TOPOLOGY_REFINED",
    "begin_coarsen_cpu_oracle_v79", "begin_refine_cpu_oracle_v79",
    "begin_transport_cpu_oracle_v79", "cancel_multifraction_suspended_amr_trial_v79",
    "classwise_combined_ledger_v79", "commit_multifraction_suspended_amr_trial_v79",
    "execution_document_v79", "load_multifraction_suspended_amr_checkpoint_v79",
    "make_implicit_fwave_midpoint_amr_receipt_v79", "method_identity_v79",
    "multifraction_suspended_amr_checkpoint_v79", "native_rows_v79",
    "resolve_multifraction_suspended_amr_execution_v79",
    "resolve_multifraction_suspended_amr_state_v79",
    "rollback_multifraction_suspended_amr_trial_v79",
    "validate_multifraction_suspended_amr_state_v79", "verify_amr_dependencies_v79",
    "verify_frozen_v78_v79",
]
