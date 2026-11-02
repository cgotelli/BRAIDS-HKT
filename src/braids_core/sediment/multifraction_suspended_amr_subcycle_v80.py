"""Bounded two-fine-step suspended AMR composition (Q4/v80).

This additive validation layer closes exactly one gap left open by v79.  For
one 2:1 parent/four-child group it executes two distinct level-local v78
lifecycle trials, binds each trial's advective and diffusive interface
integrals to its accepted ``implicit_fwave_midpoint`` receipt, and replaces
the provisional coarse integral once.  It is not a recursive or arbitrary-
depth AMR implementation.

The implementation here is the isolated CPU oracle.  The actual-CUDA owner
is :mod:`braids_gpu.multifraction_suspended_amr_subcycle_v80`.
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
    INTEGRATOR,
    ROLE_CPU_ORACLE,
    ROLE_CUDA,
    MultifractionSuspendedExecutionV78,
    MultifractionSuspendedStateV78,
    MultifractionSuspendedTrialV78,
    _validate_trial as _validate_trial_v78,
    begin_multifraction_cpu_oracle_trial_v78,
    commit_multifraction_trial_v78,
    execution_document_v78,
    make_implicit_fwave_midpoint_step_receipt_v78,
    resolve_multifraction_suspended_state_v78,
    validate_implicit_fwave_midpoint_step_receipt_v78,
    validate_multifraction_suspended_state_v78,
)
from .multifraction_suspended_amr_v79 import (
    TOPOLOGY_REFINED,
    MultifractionSuspendedAMRExecutionV79,
    MultifractionSuspendedAMRStateV79,
    _restrict_children,
    _wrap_state as _wrap_state_v79,
    execution_document_v79,
    load_multifraction_suspended_amr_checkpoint_v79,
    multifraction_suspended_amr_checkpoint_v79,
    native_rows_v79,
    validate_multifraction_suspended_amr_state_v79,
)


SCHEMA = 1
SCOPE_ID = "braids-q4-multifraction-suspended-amr-subcycle-v80-v1"
CHECKPOINT_SCOPE_ID = SCOPE_ID + ":checkpoint"
RECEIPT_SCOPE_ID = SCOPE_ID + ":accepted-two-fine-step"
SEMANTIC_ID = hashlib.sha256(SCOPE_ID.encode("ascii")).hexdigest()
FINE_SUBSTEPS = 2
REFLUX_LEVELS = 1
MAX_ACCEPTED_UPDATES = 4
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class MultifractionSuspendedAMRSubcycleV80Error(ValueError):
    """Fail-closed v80 receipt, lifecycle, reflux, or transaction error."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError, OverflowError) as error:
        raise MultifractionSuspendedAMRSubcycleV80Error(
            "v80 value is not finite canonical JSON") from error


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _array_digest(value: object) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(_canonical(list(array.shape)))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise MultifractionSuspendedAMRSubcycleV80Error(
            f"v80 {name} must be a lowercase SHA-256 digest")
    return value


def _frozen(value: object, shape: tuple[int, ...], name: str,
            *, nonnegative: bool = False) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise MultifractionSuspendedAMRSubcycleV80Error(
            f"v80 {name} shape or finite values differ")
    if nonnegative and np.any(array < 0.0):
        raise MultifractionSuspendedAMRSubcycleV80Error(
            f"v80 {name} must be nonnegative")
    result = np.ascontiguousarray(array)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class MultifractionSuspendedAMRSubcycleExecutionV80:
    role: str
    amr_execution: MultifractionSuspendedAMRExecutionV79
    fine_lifecycle_execution: MultifractionSuspendedExecutionV78
    fine_interface_faces: tuple[int, int]
    fine_to_global_orientation: tuple[int, int]
    fingerprint: str

    @property
    def fraction_count(self) -> int:
        return self.amr_execution.fraction_count


def _execution_payload(
    value: MultifractionSuspendedAMRSubcycleExecutionV80,
) -> dict:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "role": value.role,
        "v79_execution": execution_document_v79(value.amr_execution),
        "fine_v78_execution": execution_document_v78(
            value.fine_lifecycle_execution),
        "fine_interface_faces": list(value.fine_interface_faces),
        "fine_to_global_orientation": list(
            value.fine_to_global_orientation),
        "reflux_levels": REFLUX_LEVELS,
        "fine_substeps": FINE_SUBSTEPS,
        "refinement_ratio": 2,
        "arbitrary_depth": False,
        "automatic_cpu_fallback": False,
        "accepted_update_limit": MAX_ACCEPTED_UPDATES,
    }


def resolve_multifraction_suspended_amr_subcycle_execution_v80(
    amr_execution: MultifractionSuspendedAMRExecutionV79,
    fine_lifecycle_execution: MultifractionSuspendedExecutionV78, *,
    fine_interface_faces: tuple[int, int],
    fine_to_global_orientation: tuple[int, int] = (-1, 1),
) -> MultifractionSuspendedAMRSubcycleExecutionV80:
    """Resolve the sole admitted one-level, two-fine-step composition."""
    execution_document_v79(amr_execution)
    execution_document_v78(fine_lifecycle_execution)
    if (amr_execution.role != fine_lifecycle_execution.role or
            amr_execution.role not in (ROLE_CPU_ORACLE, ROLE_CUDA)):
        raise MultifractionSuspendedAMRSubcycleV80Error(
            "v80 global and fine lifecycle roles differ")
    global_v78 = amr_execution.lifecycle_execution
    if (fine_lifecycle_execution.fraction_count !=
            amr_execution.fraction_count or
            fine_lifecycle_execution.cell_count != 4 or
            fine_lifecycle_execution.base_execution.profile.fingerprint !=
            global_v78.base_execution.profile.fingerprint):
        raise MultifractionSuspendedAMRSubcycleV80Error(
            "v80 fine level must contain the same fractions on four children")
    faces = tuple(fine_interface_faces)
    if (len(faces) != 2 or len(set(faces)) != 2 or
            any(type(face) is not int or not 0 <= face <
                fine_lifecycle_execution.face_count for face in faces) or
            tuple(fine_to_global_orientation) != (-1, 1)):
        raise MultifractionSuspendedAMRSubcycleV80Error(
            "v80 requires the two ordered fine interface faces with (-1,+1) orientation")
    base = fine_lifecycle_execution.base_execution
    if (tuple(base.boundary_index[list(faces)]) != (0, 1) or
            tuple(base.owner[list(faces)]) != (0, 2) or
            np.any(base.neighbour[list(faces)] != -1) or
            set(np.flatnonzero(base.boundary_index >= 0).tolist()) != set(faces) or
            base.boundary_count != 2 or
            np.any(base.boundary_return_time_s != 0.0)):
        raise MultifractionSuspendedAMRSubcycleV80Error(
            "v80 fine interface boundaries or zero-memory ownership differ")
    if not np.array_equal(
            base.cell_area_m2, np.asarray(amr_execution.child_area_m2)):
        raise MultifractionSuspendedAMRSubcycleV80Error(
            "v80 fine lifecycle areas differ from the v79 child partition")
    placeholder = MultifractionSuspendedAMRSubcycleExecutionV80(
        amr_execution.role, amr_execution, fine_lifecycle_execution,
        faces, tuple(fine_to_global_orientation), "")
    return MultifractionSuspendedAMRSubcycleExecutionV80(
        **{**placeholder.__dict__,
           "fingerprint": _fingerprint(_execution_payload(placeholder))})


def execution_document_v80(
    value: MultifractionSuspendedAMRSubcycleExecutionV80,
) -> dict:
    if not isinstance(value, MultifractionSuspendedAMRSubcycleExecutionV80):
        raise MultifractionSuspendedAMRSubcycleV80Error(
            "v80 execution object differs")
    body = _execution_payload(value)
    if value.fingerprint != _fingerprint(body):
        raise MultifractionSuspendedAMRSubcycleV80Error(
            "v80 execution fingerprint differs")
    return {**body, "fingerprint": value.fingerprint}


@dataclass(frozen=True)
class MultifractionSuspendedAMRSubcycleStateV80:
    execution_fingerprint: str
    base_state: MultifractionSuspendedAMRStateV79
    last_macro_receipt_fingerprint: str | None
    last_fine_receipt_fingerprints: tuple[str, ...]
    last_fine_trial_sha256: tuple[str, ...]
    last_fine_advective_slots_sha256: str | None
    last_fine_diffusive_slots_sha256: str | None
    state_sha256: str

    @property
    def accepted_updates(self) -> int:
        return self.base_state.accepted_updates

    @property
    def time_s(self) -> float:
        return self.base_state.time_s


def _state_token(value: MultifractionSuspendedAMRSubcycleStateV80) -> str:
    return _fingerprint({
        "execution_fingerprint": value.execution_fingerprint,
        "base_state_sha256": value.base_state.state_sha256,
        "last_macro_receipt_fingerprint":
            value.last_macro_receipt_fingerprint,
        "last_fine_receipt_fingerprints":
            list(value.last_fine_receipt_fingerprints),
        "last_fine_trial_sha256": list(value.last_fine_trial_sha256),
        "last_fine_advective_slots_sha256":
            value.last_fine_advective_slots_sha256,
        "last_fine_diffusive_slots_sha256":
            value.last_fine_diffusive_slots_sha256,
    })


def _wrap_state(
    execution: MultifractionSuspendedAMRSubcycleExecutionV80, *,
    base_state: MultifractionSuspendedAMRStateV79,
    last_macro_receipt_fingerprint: str | None = None,
    last_fine_receipt_fingerprints: tuple[str, ...] = (),
    last_fine_trial_sha256: tuple[str, ...] = (),
    last_fine_advective_slots_sha256: str | None = None,
    last_fine_diffusive_slots_sha256: str | None = None,
) -> MultifractionSuspendedAMRSubcycleStateV80:
    execution_document_v80(execution)
    validate_multifraction_suspended_amr_state_v79(
        base_state, execution.amr_execution)
    if base_state.accepted_updates > MAX_ACCEPTED_UPDATES:
        raise MultifractionSuspendedAMRSubcycleV80Error(
            "v80 accepted update limit was exceeded")
    empty = last_macro_receipt_fingerprint is None
    values = (last_fine_receipt_fingerprints, last_fine_trial_sha256)
    hashes = (last_fine_advective_slots_sha256,
              last_fine_diffusive_slots_sha256)
    if empty:
        if any(values) or any(value is not None for value in hashes):
            raise MultifractionSuspendedAMRSubcycleV80Error(
                "v80 initial evidence inventory differs")
    else:
        _require_sha256(last_macro_receipt_fingerprint,
                        "macro receipt fingerprint")
        if any(len(value) != 2 for value in values):
            raise MultifractionSuspendedAMRSubcycleV80Error(
                "v80 accepted state requires two fine receipt/trial identities")
        for value in (*last_fine_receipt_fingerprints,
                      *last_fine_trial_sha256, *hashes):
            _require_sha256(value, "accepted fine evidence digest")
    placeholder = MultifractionSuspendedAMRSubcycleStateV80(
        execution.fingerprint, base_state, last_macro_receipt_fingerprint,
        tuple(last_fine_receipt_fingerprints), tuple(last_fine_trial_sha256),
        last_fine_advective_slots_sha256,
        last_fine_diffusive_slots_sha256, "")
    return MultifractionSuspendedAMRSubcycleStateV80(
        **{**placeholder.__dict__, "state_sha256": _state_token(placeholder)})


def resolve_multifraction_suspended_amr_subcycle_state_v80(
    execution: MultifractionSuspendedAMRSubcycleExecutionV80,
    base_state: MultifractionSuspendedAMRStateV79,
) -> MultifractionSuspendedAMRSubcycleStateV80:
    return _wrap_state(execution, base_state=base_state)


def validate_multifraction_suspended_amr_subcycle_state_v80(
    state: MultifractionSuspendedAMRSubcycleStateV80,
    execution: MultifractionSuspendedAMRSubcycleExecutionV80,
) -> None:
    if (not isinstance(state, MultifractionSuspendedAMRSubcycleStateV80) or
            state.execution_fingerprint != execution.fingerprint or
            state.state_sha256 != _state_token(state)):
        raise MultifractionSuspendedAMRSubcycleV80Error(
            "v80 state is stale, mutated, or belongs to another execution")
    _wrap_state(
        execution, base_state=state.base_state,
        last_macro_receipt_fingerprint=
            state.last_macro_receipt_fingerprint,
        last_fine_receipt_fingerprints=
            state.last_fine_receipt_fingerprints,
        last_fine_trial_sha256=state.last_fine_trial_sha256,
        last_fine_advective_slots_sha256=
            state.last_fine_advective_slots_sha256,
        last_fine_diffusive_slots_sha256=
            state.last_fine_diffusive_slots_sha256)


def make_implicit_fwave_midpoint_subcycle_receipt_v80(
    execution: MultifractionSuspendedAMRSubcycleExecutionV80,
    parent: MultifractionSuspendedAMRSubcycleStateV80, *,
    coarse_step_sha256: str,
    fine_step_sha256: tuple[str, str],
    dt_s: float,
    coarse_advective_integral_kg: object,
    coarse_diffusive_integral_kg: object,
) -> dict:
    validate_multifraction_suspended_amr_subcycle_state_v80(parent, execution)
    if parent.base_state.topology != TOPOLOGY_REFINED:
        raise MultifractionSuspendedAMRSubcycleV80Error(
            "v80 receipt requires the refined accepted topology")
    if (not isinstance(dt_s, (int, float)) or isinstance(dt_s, bool) or
            not math.isfinite(float(dt_s)) or float(dt_s) <= 0.0 or
            parent.accepted_updates + 2 > MAX_ACCEPTED_UPDATES):
        raise MultifractionSuspendedAMRSubcycleV80Error(
            "v80 receipt timestep or bounded update count differs")
    if (not isinstance(fine_step_sha256, tuple) or
            len(fine_step_sha256) != 2):
        raise MultifractionSuspendedAMRSubcycleV80Error(
            "v80 requires exactly two fine hydraulic step digests")
    coarse_sha = _require_sha256(coarse_step_sha256, "coarse step digest")
    fine_sha = tuple(_require_sha256(value, "fine step digest")
                     for value in fine_step_sha256)
    if fine_sha[0] == fine_sha[1]:
        raise MultifractionSuspendedAMRSubcycleV80Error(
            "v80 fine hydraulic step digests must be distinct")
    ng = execution.fraction_count
    ca = _frozen(coarse_advective_integral_kg, (2, ng),
                 "coarse advective integral")
    cd = _frozen(coarse_diffusive_integral_kg, (2, ng),
                 "coarse diffusive integral")
    dt = float(dt_s)
    half = 0.5*dt
    fine_receipts = []
    for step in range(2):
        fine_receipts.append(make_implicit_fwave_midpoint_step_receipt_v78(
            hydraulic_execution_sha256=
                execution.fine_lifecycle_execution.hydraulic_execution_sha256,
            accepted_step_sha256=fine_sha[step],
            accepted_update=parent.accepted_updates+step+1,
            time_before_s=parent.time_s+step*half, dt_s=half))
    body = {
        "schema": SCHEMA,
        "scope_id": RECEIPT_SCOPE_ID,
        "execution_fingerprint": execution.fingerprint,
        "parent_state_sha256": parent.state_sha256,
        "integrator": INTEGRATOR,
        "hydraulic_backend": "implicit_fwave_midpoint_gpu",
        "amr": True,
        "accepted": True,
        "automatic_cpu_fallback": False,
        "reflux_levels": REFLUX_LEVELS,
        "refinement_ratio": 2,
        "fine_substeps": FINE_SUBSTEPS,
        "arbitrary_depth": False,
        "accepted_update_before": parent.accepted_updates,
        "accepted_update_after": parent.accepted_updates+2,
        "time_before_s": parent.time_s,
        "dt_s": dt,
        "time_after_s": parent.time_s+dt,
        "coarse_step_sha256": coarse_sha,
        "coarse_advective_integral_sha256": _array_digest(ca),
        "coarse_diffusive_integral_sha256": _array_digest(cd),
        "fine_step_receipts": fine_receipts,
    }
    return {**body, "fingerprint": _fingerprint(body)}


def _validate_receipt(execution, parent, receipt, dt_s, coarse_adv,
                      coarse_diff) -> dict:
    expected = {
        "schema", "scope_id", "execution_fingerprint",
        "parent_state_sha256", "integrator", "hydraulic_backend", "amr",
        "accepted", "automatic_cpu_fallback", "reflux_levels",
        "refinement_ratio", "fine_substeps", "arbitrary_depth",
        "accepted_update_before", "accepted_update_after", "time_before_s",
        "dt_s", "time_after_s", "coarse_step_sha256",
        "coarse_advective_integral_sha256",
        "coarse_diffusive_integral_sha256", "fine_step_receipts",
        "fingerprint",
    }
    if not isinstance(receipt, Mapping) or set(receipt) != expected:
        raise MultifractionSuspendedAMRSubcycleV80Error(
            "v80 hydraulic receipt inventory differs")
    body = {key: receipt[key] for key in receipt if key != "fingerprint"}
    if (receipt["fingerprint"] != _fingerprint(body) or
            receipt["schema"] != SCHEMA or
            receipt["scope_id"] != RECEIPT_SCOPE_ID or
            receipt["execution_fingerprint"] != execution.fingerprint or
            receipt["parent_state_sha256"] != parent.state_sha256 or
            receipt["integrator"] != INTEGRATOR or
            receipt["hydraulic_backend"] != "implicit_fwave_midpoint_gpu" or
            receipt["amr"] is not True or receipt["accepted"] is not True or
            receipt["automatic_cpu_fallback"] is not False or
            receipt["reflux_levels"] != 1 or
            receipt["refinement_ratio"] != 2 or
            receipt["fine_substeps"] != 2 or
            receipt["arbitrary_depth"] is not False or
            receipt["accepted_update_before"] != parent.accepted_updates or
            receipt["accepted_update_after"] != parent.accepted_updates+2 or
            receipt["time_before_s"] != parent.time_s or
            receipt["dt_s"] != float(dt_s) or
            receipt["time_after_s"] != parent.time_s+float(dt_s) or
            receipt["coarse_advective_integral_sha256"] !=
            _array_digest(coarse_adv) or
            receipt["coarse_diffusive_integral_sha256"] !=
            _array_digest(coarse_diff)):
        raise MultifractionSuspendedAMRSubcycleV80Error(
            "v80 requires the current accepted two-fine-step receipt")
    _require_sha256(receipt["coarse_step_sha256"], "coarse step digest")
    nested = receipt["fine_step_receipts"]
    if not isinstance(nested, list) or len(nested) != 2:
        raise MultifractionSuspendedAMRSubcycleV80Error(
            "v80 fine receipt count differs")
    half = 0.5*float(dt_s)
    checked = []
    for step in range(2):
        checked.append(validate_implicit_fwave_midpoint_step_receipt_v78(
            nested[step], hydraulic_execution_sha256=
                execution.fine_lifecycle_execution.hydraulic_execution_sha256,
            parent_accepted_updates=parent.accepted_updates+step,
            parent_time_s=parent.time_s+step*half, dt_s=half))
    if checked[0]["accepted_step_sha256"] == \
            checked[1]["accepted_step_sha256"]:
        raise MultifractionSuspendedAMRSubcycleV80Error(
            "v80 fine hydraulic receipts are not distinct")
    return dict(receipt)


def _extract_fine_initial(execution, parent):
    global_state = parent.base_state
    cells = np.asarray(execution.amr_execution.child_cells)
    lifecycle = global_state.lifecycle.lifecycle
    return resolve_multifraction_suspended_state_v78(
        execution.fine_lifecycle_execution,
        suspended_mass_kg_m2=lifecycle.suspended_mass_kg_m2[:, cells],
        bed_mass_kg_m2=lifecycle.bed_mass_kg_m2[:, cells],
        accepted_updates=parent.accepted_updates, time_s=parent.time_s)


def _derive_fine_integrals(execution, trials, dt_s):
    ng = execution.fraction_count
    adv = np.empty((2, 2, ng), dtype=np.float64)
    diff = np.empty_like(adv)
    length = execution.fine_lifecycle_execution.base_execution.face_length_m
    half = 0.5*float(dt_s)
    for step, trial in enumerate(trials):
        for local, (face, orientation) in enumerate(zip(
                execution.fine_interface_faces,
                execution.fine_to_global_orientation)):
            adv[step, local] = (
                orientation*trial.advective_face_flux_kg_m_s[:, face]*
                length[face]*half)
            diff[step, local] = (
                orientation*trial.diffusive_face_mass_owner_out_kg[:, face])
    return np.ascontiguousarray(adv), np.ascontiguousarray(diff)


def _expanded_integrals(execution, ca, cd, fa, fd):
    ng, ns = execution.fraction_count, 3+2*execution.fraction_count
    coarse = np.zeros((2, 2, ns), dtype=np.float64)
    fine = np.zeros((2, 2, 2, ns), dtype=np.float64)
    coarse[:, :, 3:3+ng] = (ca+cd)[:, None, :]
    fine[:, :, :, 3:3+ng] = (fa+fd)[:, :, None, :]
    return coarse, fine


def _merge_global_lifecycle(execution, parent, fine_initial, fine_final,
                            rows, accepted_updates, time_s):
    global_v78 = execution.amr_execution.lifecycle_execution
    source = parent.base_state.lifecycle.lifecycle
    start = fine_initial.lifecycle
    final = fine_final.lifecycle
    ng = execution.fraction_count
    fields = {
        "suspended_mass_kg_m2": rows[:, 3:3+ng].T,
        "bed_mass_kg_m2": rows[:, 3+ng:3+2*ng].T,
        "boundary_last_outflow_concentration_kg_m3":
            source.boundary_last_outflow_concentration_kg_m3,
        "boundary_return_clock_s": source.boundary_return_clock_s,
        "advective_boundary_ledger_kg": source.advective_boundary_ledger_kg,
        "diffusive_boundary_ledger_kg": source.diffusive_boundary_ledger_kg,
        "water_exchange_ledger_kg": source.water_exchange_ledger_kg +
            final.water_exchange_ledger_kg-start.water_exchange_ledger_kg,
        "bed_exchange_ledger_kg": source.bed_exchange_ledger_kg +
            final.bed_exchange_ledger_kg-start.bed_exchange_ledger_kg,
        "wetdry_water_ledger_kg": source.wetdry_water_ledger_kg +
            final.wetdry_water_ledger_kg-start.wetdry_water_ledger_kg,
        "wetdry_bed_ledger_kg": source.wetdry_bed_ledger_kg +
            final.wetdry_bed_ledger_kg-start.wetdry_bed_ledger_kg,
        "reference_water_mass_kg": source.reference_water_mass_kg,
        "reference_bed_mass_kg": source.reference_bed_mass_kg,
        "accepted_updates": accepted_updates,
        "time_s": time_s,
    }
    return resolve_multifraction_suspended_state_v78(global_v78, **fields)


def _compose_candidate(execution, parent, receipt, fine_initial,
                       fine_trials, ca, cd, fa, fd, final_depth,
                       final_momentum):
    e = execution.amr_execution
    ng, ns = execution.fraction_count, 3+2*execution.fraction_count
    parent_rows = native_rows_v79(
        e, parent.base_state.lifecycle, parent.base_state.depth_m,
        parent.base_state.momentum_m2_s)
    rows = parent_rows.copy()
    cells = np.asarray(e.child_cells)
    fine_final = fine_trials[1].candidate
    rows[cells, 0] = final_depth
    rows[cells, 1:3] = final_momentum.T
    rows[cells, 3:3+ng] = fine_final.suspended_mass_kg_m2.T
    rows[cells, 3+ng:3+2*ng] = fine_final.bed_mass_kg_m2.T
    coarse, fine = _expanded_integrals(execution, ca, cd, fa, fd)
    subset = np.asarray(e.reflux_cells)
    before = np.ascontiguousarray(parent_rows[subset].copy())
    post = np.ascontiguousarray(rows[subset].copy())
    area = e.lifecycle_execution.base_execution.cell_area_m2[e.reflux_cells[0]]
    post[0] -= coarse[0, 0]/area
    post[0] += coarse[1, 1]/area
    register = np.sum(fine, axis=0, dtype=np.float64)
    after = post.copy()
    for face, side in enumerate(e.interface_coarse_sides):
        delta = coarse[face, side]-register[face, side]
        after[0] += (delta if side == 0 else -delta)/area
    if np.any(after[:, 0] < 0.0) or np.any(after[:, 3:] < 0.0):
        raise MultifractionSuspendedAMRSubcycleV80Error(
            "v80 reflux would require positivity repair")
    rows[subset] = after
    lifecycle = _merge_global_lifecycle(
        execution, parent, fine_initial, fine_final, rows,
        parent.accepted_updates+2, parent.time_s+float(receipt["dt_s"]))
    base = _wrap_state_v79(
        e, topology=TOPOLOGY_REFINED, lifecycle=lifecycle,
        depth_m=rows[:, 0], momentum_m2_s=rows[:, 1:3].T,
        coarse_parent_row=_restrict_children(e, rows),
        accepted_fine_flux_register_kg=register,
        topology_events=parent.base_state.topology_events+
            ("two_authenticated_fine_steps_reflux",))
    candidate = _wrap_state(
        execution, base_state=base,
        last_macro_receipt_fingerprint=receipt["fingerprint"],
        last_fine_receipt_fingerprints=tuple(
            item["fingerprint"] for item in receipt["fine_step_receipts"]),
        last_fine_trial_sha256=tuple(
            trial.trial_sha256 for trial in fine_trials),
        last_fine_advective_slots_sha256=_array_digest(fa),
        last_fine_diffusive_slots_sha256=_array_digest(fd))
    return candidate, before, post, after, register


@dataclass(frozen=True)
class MultifractionSuspendedAMRSubcycleTrialV80:
    execution_fingerprint: str
    parent_state_sha256: str
    candidate: MultifractionSuspendedAMRSubcycleStateV80
    macro_receipt: dict
    fine_initial_state: MultifractionSuspendedStateV78
    fine_trials: tuple[MultifractionSuspendedTrialV78,
                       MultifractionSuspendedTrialV78]
    coarse_advective_integral_kg: np.ndarray
    coarse_diffusive_integral_kg: np.ndarray
    fine_advective_slots_kg: np.ndarray
    fine_diffusive_slots_kg: np.ndarray
    fine_register_kg: np.ndarray
    reflux_before_rows: np.ndarray
    reflux_post_level_rows: np.ndarray
    reflux_after_rows: np.ndarray
    backend: str
    trial_sha256: str


def _trial_token(value: MultifractionSuspendedAMRSubcycleTrialV80) -> str:
    return _fingerprint({
        "execution_fingerprint": value.execution_fingerprint,
        "parent_state_sha256": value.parent_state_sha256,
        "candidate_state_sha256": value.candidate.state_sha256,
        "macro_receipt_fingerprint": value.macro_receipt.get("fingerprint"),
        "fine_initial_state_sha256": value.fine_initial_state.state_sha256,
        "fine_trial_sha256": [trial.trial_sha256
                              for trial in value.fine_trials],
        "arrays_sha256": [_array_digest(array) for array in (
            value.coarse_advective_integral_kg,
            value.coarse_diffusive_integral_kg,
            value.fine_advective_slots_kg,
            value.fine_diffusive_slots_kg,
            value.fine_register_kg,
            value.reflux_before_rows,
            value.reflux_post_level_rows,
            value.reflux_after_rows,
        )],
        "backend": value.backend,
    })


def _build_trial(execution, parent, candidate, receipt, fine_initial,
                 fine_trials, ca, cd, fa, fd, register, before, post, after,
                 backend):
    ng, ns = execution.fraction_count, 3+2*execution.fraction_count
    arrays = (
        _frozen(ca, (2, ng), "coarse advective integral"),
        _frozen(cd, (2, ng), "coarse diffusive integral"),
        _frozen(fa, (2, 2, ng), "fine advective slots"),
        _frozen(fd, (2, 2, ng), "fine diffusive slots"),
        _frozen(register, (2, 2, ns), "fine register"),
        _frozen(before, (4, ns), "reflux before rows"),
        _frozen(post, (4, ns), "reflux post-level rows"),
        _frozen(after, (4, ns), "reflux after rows"),
    )
    placeholder = MultifractionSuspendedAMRSubcycleTrialV80(
        execution.fingerprint, parent.state_sha256, candidate,
        dict(receipt), fine_initial, tuple(fine_trials), *arrays, backend, "")
    return MultifractionSuspendedAMRSubcycleTrialV80(
        **{**placeholder.__dict__, "trial_sha256": _trial_token(placeholder)})


def _input_chain_host(execution, parent, inputs, dt_s):
    if not isinstance(inputs, Sequence) or len(inputs) != 2:
        raise MultifractionSuspendedAMRSubcycleV80Error(
            "v80 requires exactly two fine lifecycle input bundles")
    cells = np.asarray(execution.amr_execution.child_cells)
    expected_before = parent.base_state.depth_m[cells]
    final_depth = None
    final_momentum = None
    half = 0.5*float(dt_s)
    for step, bundle in enumerate(inputs):
        if not isinstance(bundle, Mapping) or float(bundle.get("dt_s", -1.0)) != half:
            raise MultifractionSuspendedAMRSubcycleV80Error(
                "v80 fine lifecycle timestep differs from the receipt")
        before = np.asarray(bundle.get("depth_before_m"), dtype=np.float64)
        after = np.asarray(bundle.get("depth_after_m"), dtype=np.float64)
        momentum = np.asarray(bundle.get("endpoint_momentum_m2_s"),
                              dtype=np.float64)
        if (before.shape != (4,) or after.shape != (4,) or
                momentum.shape != (2, 4) or
                not np.array_equal(before, expected_before)):
            raise MultifractionSuspendedAMRSubcycleV80Error(
                "v80 fine hydraulic states are not temporally contiguous")
        expected_before = after
        final_depth, final_momentum = after, momentum
    return final_depth, final_momentum


def begin_multifraction_suspended_amr_subcycle_cpu_oracle_v80(
    execution: MultifractionSuspendedAMRSubcycleExecutionV80,
    parent: MultifractionSuspendedAMRSubcycleStateV80, *,
    accepted_subcycle_receipt: Mapping[str, object],
    coarse_advective_integral_kg: object,
    coarse_diffusive_integral_kg: object,
    fine_step_inputs: Sequence[Mapping[str, object]],
) -> MultifractionSuspendedAMRSubcycleTrialV80:
    validate_multifraction_suspended_amr_subcycle_state_v80(parent, execution)
    if (execution.role != ROLE_CPU_ORACLE or
            parent.base_state.topology != TOPOLOGY_REFINED):
        raise MultifractionSuspendedAMRSubcycleV80Error(
            "v80 CPU oracle requires the refined isolated CPU state")
    ng = execution.fraction_count
    ca = _frozen(coarse_advective_integral_kg, (2, ng),
                 "coarse advective integral")
    cd = _frozen(coarse_diffusive_integral_kg, (2, ng),
                 "coarse diffusive integral")
    dt_s = float(accepted_subcycle_receipt.get("dt_s", float("nan")))
    receipt = _validate_receipt(
        execution, parent, accepted_subcycle_receipt, dt_s, ca, cd)
    final_depth, final_momentum = _input_chain_host(
        execution, parent, fine_step_inputs, dt_s)
    initial = _extract_fine_initial(execution, parent)
    trial0 = begin_multifraction_cpu_oracle_trial_v78(
        execution.fine_lifecycle_execution, initial,
        accepted_step_receipt=receipt["fine_step_receipts"][0],
        **dict(fine_step_inputs[0]))
    state1 = commit_multifraction_trial_v78(
        execution.fine_lifecycle_execution, initial, trial0)
    trial1 = begin_multifraction_cpu_oracle_trial_v78(
        execution.fine_lifecycle_execution, state1,
        accepted_step_receipt=receipt["fine_step_receipts"][1],
        **dict(fine_step_inputs[1]))
    trials = (trial0, trial1)
    fa, fd = _derive_fine_integrals(execution, trials, dt_s)
    candidate, before, post, after, register = _compose_candidate(
        execution, parent, receipt, initial, trials, ca, cd, fa, fd,
        final_depth, final_momentum)
    return _build_trial(
        execution, parent, candidate, receipt, initial, trials,
        ca, cd, fa, fd, register, before, post, after, ROLE_CPU_ORACLE)


def _validate_trial(execution, parent, trial):
    validate_multifraction_suspended_amr_subcycle_state_v80(parent, execution)
    if (not isinstance(trial, MultifractionSuspendedAMRSubcycleTrialV80) or
            trial.execution_fingerprint != execution.fingerprint or
            trial.parent_state_sha256 != parent.state_sha256 or
            trial.trial_sha256 != _trial_token(trial)):
        raise MultifractionSuspendedAMRSubcycleV80Error(
            "v80 trial is stale, mutated, or belongs to another parent")
    validate_multifraction_suspended_amr_subcycle_state_v80(
        trial.candidate, execution)
    if trial.backend != execution.role:
        raise MultifractionSuspendedAMRSubcycleV80Error(
            "v80 trial backend differs from the execution role")
    receipt = _validate_receipt(
        execution, parent, trial.macro_receipt,
        float(trial.macro_receipt.get("dt_s", float("nan"))),
        trial.coarse_advective_integral_kg,
        trial.coarse_diffusive_integral_kg)
    expected_initial = _extract_fine_initial(execution, parent)
    if trial.fine_initial_state.state_sha256 != expected_initial.state_sha256:
        raise MultifractionSuspendedAMRSubcycleV80Error(
            "v80 fine initial state differs from the accepted children")
    if len(trial.fine_trials) != 2:
        raise MultifractionSuspendedAMRSubcycleV80Error(
            "v80 fine trial count differs")
    _validate_trial_v78(execution.fine_lifecycle_execution,
                        trial.fine_initial_state, trial.fine_trials[0])
    _validate_trial_v78(execution.fine_lifecycle_execution,
                        trial.fine_trials[0].candidate,
                        trial.fine_trials[1])
    for nested, nested_receipt in zip(
            trial.fine_trials, receipt["fine_step_receipts"]):
        if nested.accepted_step_receipt_sha256 != nested_receipt["fingerprint"]:
            raise MultifractionSuspendedAMRSubcycleV80Error(
                "v80 fine trial is not bound to its accepted receipt")
    fa, fd = _derive_fine_integrals(
        execution, trial.fine_trials, receipt["dt_s"])
    if (not np.array_equal(fa, trial.fine_advective_slots_kg) or
            not np.array_equal(fd, trial.fine_diffusive_slots_kg)):
        raise MultifractionSuspendedAMRSubcycleV80Error(
            "v80 fine integrals differ from the exact lifecycle trials")
    final = trial.fine_trials[1].candidate.lifecycle
    cells = np.asarray(execution.amr_execution.child_cells)
    candidate, before, post, after, register = _compose_candidate(
        execution, parent, receipt, trial.fine_initial_state,
        trial.fine_trials, trial.coarse_advective_integral_kg,
        trial.coarse_diffusive_integral_kg, fa, fd,
        trial.candidate.base_state.depth_m[cells],
        trial.candidate.base_state.momentum_m2_s[:, cells])
    if (candidate.state_sha256 != trial.candidate.state_sha256 or
            not np.array_equal(before, trial.reflux_before_rows) or
            not np.array_equal(post, trial.reflux_post_level_rows) or
            not np.array_equal(after, trial.reflux_after_rows) or
            not np.array_equal(register, trial.fine_register_kg) or
            not np.array_equal(
                final.suspended_mass_kg_m2,
                trial.candidate.base_state.lifecycle.suspended_mass_kg_m2[:, cells]) or
            not np.array_equal(
                final.bed_mass_kg_m2,
                trial.candidate.base_state.lifecycle.bed_mass_kg_m2[:, cells])):
        raise MultifractionSuspendedAMRSubcycleV80Error(
            "v80 candidate or reflux evidence differs from recomputation")


def commit_multifraction_suspended_amr_subcycle_trial_v80(
    execution, parent, trial,
):
    _validate_trial(execution, parent, trial)
    return trial.candidate


def rollback_multifraction_suspended_amr_subcycle_trial_v80(
    execution, parent, trial,
):
    _validate_trial(execution, parent, trial)
    return parent


def cancel_multifraction_suspended_amr_subcycle_trial_v80(
    execution, parent, trial,
):
    return rollback_multifraction_suspended_amr_subcycle_trial_v80(
        execution, parent, trial)


def multifraction_suspended_amr_subcycle_checkpoint_v80(
    execution, state,
) -> dict:
    validate_multifraction_suspended_amr_subcycle_state_v80(state, execution)
    body = {
        "schema": SCHEMA,
        "scope_id": CHECKPOINT_SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "execution_fingerprint": execution.fingerprint,
        "integrator": INTEGRATOR,
        "transaction_schema":
            "accepted_two_fine_receipts_trials_register_root_memory_v1",
        "v79_checkpoint": multifraction_suspended_amr_checkpoint_v79(
            execution.amr_execution, state.base_state),
        "last_macro_receipt_fingerprint":
            state.last_macro_receipt_fingerprint,
        "last_fine_receipt_fingerprints":
            list(state.last_fine_receipt_fingerprints),
        "last_fine_trial_sha256": list(state.last_fine_trial_sha256),
        "last_fine_advective_slots_sha256":
            state.last_fine_advective_slots_sha256,
        "last_fine_diffusive_slots_sha256":
            state.last_fine_diffusive_slots_sha256,
        "state_sha256": state.state_sha256,
    }
    return {**body, "fingerprint": _fingerprint(body)}


def load_multifraction_suspended_amr_subcycle_checkpoint_v80(
    execution, document,
):
    expected = {
        "schema", "scope_id", "semantic_id", "execution_fingerprint",
        "integrator", "transaction_schema", "v79_checkpoint",
        "last_macro_receipt_fingerprint", "last_fine_receipt_fingerprints",
        "last_fine_trial_sha256", "last_fine_advective_slots_sha256",
        "last_fine_diffusive_slots_sha256", "state_sha256", "fingerprint",
    }
    if not isinstance(document, Mapping) or set(document) != expected:
        raise MultifractionSuspendedAMRSubcycleV80Error(
            "v80 checkpoint inventory differs")
    body = {key: document[key] for key in document if key != "fingerprint"}
    if (document["fingerprint"] != _fingerprint(body) or
            document["schema"] != SCHEMA or
            document["scope_id"] != CHECKPOINT_SCOPE_ID or
            document["semantic_id"] != SEMANTIC_ID or
            document["execution_fingerprint"] != execution.fingerprint or
            document["integrator"] != INTEGRATOR or
            document["transaction_schema"] !=
            "accepted_two_fine_receipts_trials_register_root_memory_v1"):
        raise MultifractionSuspendedAMRSubcycleV80Error(
            "v80 checkpoint identity differs")
    base = load_multifraction_suspended_amr_checkpoint_v79(
        execution.amr_execution, document["v79_checkpoint"])
    state = _wrap_state(
        execution, base_state=base,
        last_macro_receipt_fingerprint=
            document["last_macro_receipt_fingerprint"],
        last_fine_receipt_fingerprints=tuple(
            document["last_fine_receipt_fingerprints"]),
        last_fine_trial_sha256=tuple(document["last_fine_trial_sha256"]),
        last_fine_advective_slots_sha256=
            document["last_fine_advective_slots_sha256"],
        last_fine_diffusive_slots_sha256=
            document["last_fine_diffusive_slots_sha256"])
    if document["state_sha256"] != state.state_sha256:
        raise MultifractionSuspendedAMRSubcycleV80Error(
            "v80 checkpoint accepted state digest differs")
    return state


def classwise_combined_ledger_v80(execution, state):
    validate_multifraction_suspended_amr_subcycle_state_v80(state, execution)
    base = state.base_state.lifecycle.lifecycle
    area = execution.amr_execution.lifecycle_execution.base_execution \
        .cell_area_m2[None, :]
    current = np.sum(
        (base.suspended_mass_kg_m2+base.bed_mass_kg_m2)*area, axis=1)
    boundary = np.sum(base.advective_boundary_ledger_kg+
                      base.diffusive_boundary_ledger_kg, axis=1)
    reference = base.reference_water_mass_kg+base.reference_bed_mass_kg
    return current, reference, boundary, current-reference-boundary


def method_identity_v80() -> dict:
    return {
        "scope_id": SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "builds_additively_on": [
            "braids-q4-multifraction-suspended-amr-v79-v1",
            "braids-q4-multifraction-suspended-lifecycle-v78-v1",
        ],
        "fraction_scope": "2..3 noncohesive sand/gravel classes",
        "integrator": INTEGRATOR,
        "amr_scope": "one 2:1 parent/four-child group, exactly two fine substeps",
        "authenticated_fine_substeps": 2,
        "fine_receipts_bound_to_start_end_and_step_digest": True,
        "fine_advective_diffusive_integrals_derived_from_exact_trials": True,
        "coarse_integral_replaced_once": True,
        "root_boundary_memory_preserved": True,
        "recursive_or_arbitrary_depth_claimed": False,
        "accepted_update_limit": MAX_ACCEPTED_UPDATES,
        "automatic_cpu_fallback": False,
        "cpu_scope": "isolated oracle only",
    }


FROZEN_V79_SHA256 = {
    "braids_core/sediment/multifraction_suspended_amr_v79.py":
        "2fe9d2e2417fd043dafddd6d49b98d93f6d76f0164eae2fa3d7645a22c9eaaa8",
    "braids_gpu/multifraction_suspended_amr_v79.py":
        "347666d59efd0ad87bb2763c2109584cd878b1c9e51a7647deda649259b98685",
    "tests/test_q4_multifraction_suspended_amr_v79.py":
        "2046f672ca871bc72e45806b710714d65b32811974ad8451fa3100bc74019a74",
    "tests/test_q4_multifraction_suspended_amr_cuda_v79.py":
        "2240efe8c3e097ddc17a6678bef50c1840ad75f10798d858ea5d74715c8d9083",
    "braids_support/sediment/delft3d_current/q4_multifraction_suspended_amr_v79/build_report.py":
        "2af910e17e4ea6e7925bf30fd3dd260459e24cf7515d453c0f2a62e52e1a0d28",
    "braids_support/sediment/delft3d_current/q4_multifraction_suspended_amr_v79/REPORT.md":
        "56c569893ad2d46867f1d4e1e2892759bb46893f1c40ea884012fe70d982fe32",
    "braids_support/sediment/delft3d_current/q4_multifraction_suspended_amr_v79/metrics.json":
        "e6745b7a8d8a1bccfc5654c48bdd962a1f8bbe23ae496e8eaec9877f29a8a467",
    "braids_support/sediment/delft3d_current/q4_multifraction_suspended_amr_v79/Q4_MULTIFRACTION_SUSPENDED_AMR_V79.png":
        "f89b8f9e8c5dcd3866e9fef65ec368d20503d78b74b11792bbbbe92aeb5d2aaa",
}


def verify_frozen_v79_v80(root: str | Path) -> dict[str, str]:
    base = Path(root)
    actual = {}
    for relative, expected in FROZEN_V79_SHA256.items():
        path = base/relative
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise MultifractionSuspendedAMRSubcycleV80Error(
                f"frozen v79 dependency changed: {relative}")
        actual[relative] = digest
    return actual


__all__ = [
    "MAX_ACCEPTED_UPDATES",
    "MultifractionSuspendedAMRSubcycleExecutionV80",
    "MultifractionSuspendedAMRSubcycleStateV80",
    "MultifractionSuspendedAMRSubcycleTrialV80",
    "MultifractionSuspendedAMRSubcycleV80Error",
    "begin_multifraction_suspended_amr_subcycle_cpu_oracle_v80",
    "cancel_multifraction_suspended_amr_subcycle_trial_v80",
    "classwise_combined_ledger_v80",
    "commit_multifraction_suspended_amr_subcycle_trial_v80",
    "execution_document_v80",
    "load_multifraction_suspended_amr_subcycle_checkpoint_v80",
    "make_implicit_fwave_midpoint_subcycle_receipt_v80",
    "method_identity_v80",
    "multifraction_suspended_amr_subcycle_checkpoint_v80",
    "resolve_multifraction_suspended_amr_subcycle_execution_v80",
    "resolve_multifraction_suspended_amr_subcycle_state_v80",
    "rollback_multifraction_suspended_amr_subcycle_trial_v80",
    "validate_multifraction_suspended_amr_subcycle_state_v80",
    "verify_frozen_v79_v80",
]
