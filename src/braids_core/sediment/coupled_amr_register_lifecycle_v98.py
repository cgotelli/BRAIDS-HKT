"""Backend-neutral scheduled register lifecycle for Q4/v98.

The numerical values supplied here are already authenticated, canonically
oriented face-time integrals extracted from real level trials.  This module
owns only the deterministic two-fine-substep replacement schedule and its
small transactional clock.  It owns no hydraulic/scalar/Exner state and is
not, by itself, an AMR solver qualification.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Callable, Mapping, Sequence

import numpy as np


SCHEMA = 1
SCOPE_ID = "braids.coupled_amr_register.lifecycle.v98"
PHASE_HYDRAULIC = "hydraulic_water_momentum"
PHASE_SCALAR_BED = "suspended_fluid_bed"
PHASES = (PHASE_HYDRAULIC, PHASE_SCALAR_BED)
FINE_SUBSTEPS = 2
FINE_FACES = 2
MAX_MACRO_UPDATES = 2
_SHA = re.compile(r"[0-9a-f]{64}")


class CoupledAMRRegisterLifecycleV98Error(ValueError):
    """Fail-closed scheduled-register error."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _digest(*values: object) -> str:
    digest = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(_canonical(list(array.shape)))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise CoupledAMRRegisterLifecycleV98Error(
            f"v98 {name} must be a lowercase SHA-256 digest")
    return value


def _finite(value: object, name: str, *, positive: bool = False) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(float(value)) or
            (positive and float(value) <= 0.0)):
        qualifier = "finite and positive" if positive else "finite"
        raise CoupledAMRRegisterLifecycleV98Error(
            f"v98 {name} must be {qualifier}")
    return float(value)


def _frozen(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value)
    if (array.shape != shape or array.dtype.kind not in "fiu" or
            not np.isfinite(array).all()):
        raise CoupledAMRRegisterLifecycleV98Error(
            f"v98 {name} must be finite with shape {shape}")
    result = np.array(array, dtype=np.float64, order="C", copy=True)
    result.setflags(write=False)
    return result


def _cancelled(value: bool | Callable[[], bool]) -> bool:
    if type(value) is bool:
        return value
    if not callable(value):
        raise TypeError("cancelled must be Boolean or callable")
    return bool(value())


@dataclass(frozen=True)
class ScheduledRegisterPhaseV98:
    phase: str
    interface_fingerprint: str
    lane_ids: tuple[str, ...]
    coarse_source_sha256: str
    fine_source_sha256: tuple[str, str]
    time_start_s: float
    coarse_dt_s: float
    fine_dt_s: tuple[float, float]
    coarse_cell_area_m2: float
    coarse_integral: np.ndarray
    fine_integral_slots: np.ndarray
    accepted_fine_integral: np.ndarray
    coarse_minus_fine_integral: np.ndarray
    coarse_cell_correction: np.ndarray
    coarse_replacement_count: int
    fine_slot_consumption_count: np.ndarray
    receipt_sha256: str


def _phase_payload(value: ScheduledRegisterPhaseV98) -> dict:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID + ":phase",
        "phase": value.phase,
        "interface": value.interface_fingerprint,
        "lane_ids": list(value.lane_ids),
        "coarse_source": value.coarse_source_sha256,
        "fine_source": list(value.fine_source_sha256),
        "time_start_s": value.time_start_s,
        "coarse_dt_s": value.coarse_dt_s,
        "fine_dt_s": list(value.fine_dt_s),
        "coarse_cell_area_m2": value.coarse_cell_area_m2,
        "arrays_sha256": _digest(
            value.coarse_integral, value.fine_integral_slots,
            value.accepted_fine_integral,
            value.coarse_minus_fine_integral,
            value.coarse_cell_correction,
            value.fine_slot_consumption_count),
        "coarse_replacement_count": value.coarse_replacement_count,
        "schedule": "two_substeps_times_two_fine_faces_each_exactly_once",
    }


def resolve_scheduled_register_phase_v98(
    *, phase: str, interface_fingerprint: str,
    lane_ids: Sequence[str], coarse_source_sha256: str,
    fine_source_sha256: Sequence[str], time_start_s: float,
    coarse_dt_s: float, fine_dt_s: Sequence[float],
    coarse_cell_area_m2: float, coarse_integral: object,
    fine_integral_slots: object,
    cancelled: bool | Callable[[], bool] = False,
) -> ScheduledRegisterPhaseV98:
    """Resolve one canonical coarse-minus-fine replacement phase."""
    if _cancelled(cancelled):
        raise InterruptedError("v98 register phase cancelled before work")
    if phase not in PHASES:
        raise CoupledAMRRegisterLifecycleV98Error(
            "v98 register phase identity differs")
    interface = _sha(interface_fingerprint, "interface fingerprint")
    lanes = tuple(lane_ids)
    if (not lanes or len(set(lanes)) != len(lanes) or
            any(not isinstance(value, str) or not value for value in lanes)):
        raise CoupledAMRRegisterLifecycleV98Error(
            "v98 lane IDs must be nonempty and unique")
    coarse_source = _sha(coarse_source_sha256, "coarse source")
    fine_sources = tuple(fine_source_sha256)
    if len(fine_sources) != FINE_SUBSTEPS:
        raise CoupledAMRRegisterLifecycleV98Error(
            "v98 requires exactly two fine source receipts")
    fine_sources = tuple(_sha(value, "fine source") for value in fine_sources)
    if fine_sources[0] == fine_sources[1]:
        raise CoupledAMRRegisterLifecycleV98Error(
            "v98 fine substep source receipts must be distinct")
    start = _finite(time_start_s, "phase start time")
    if start < 0.0:
        raise CoupledAMRRegisterLifecycleV98Error(
            "v98 phase start time must be nonnegative")
    coarse_dt = _finite(coarse_dt_s, "coarse dt", positive=True)
    fine_dt = tuple(_finite(value, "fine dt", positive=True)
                    for value in fine_dt_s)
    if (len(fine_dt) != FINE_SUBSTEPS or
            fine_dt[0] != fine_dt[1] or coarse_dt != sum(fine_dt)):
        raise CoupledAMRRegisterLifecycleV98Error(
            "v98 requires two equal fine steps spanning one coarse step")
    area = _finite(coarse_cell_area_m2, "coarse cell area", positive=True)
    nl = len(lanes)
    coarse = _frozen(coarse_integral, (nl,), "coarse integral")
    slots = _frozen(
        fine_integral_slots, (FINE_SUBSTEPS, FINE_FACES, nl),
        "fine integral slots")
    accepted = _frozen(np.sum(slots, axis=(0, 1), dtype=np.float64),
                       (nl,), "accepted fine integral")
    difference = _frozen(coarse-accepted, (nl,),
                         "coarse-minus-fine integral")
    correction = _frozen(difference/area, (nl,),
                         "coarse cell correction")
    counts = np.ones((FINE_SUBSTEPS, FINE_FACES), dtype=np.int32)
    counts.setflags(write=False)
    # The provisional coarse conservative contribution is ``-coarse``.
    # Applying ``coarse-fine`` once must leave exactly ``-fine`` up to the
    # roundoff of the same accepted operations; this is not a mass repair.
    reconstructed = -coarse+difference
    tolerance = 64.0*np.finfo(np.float64).eps*np.maximum(
        1.0, np.maximum(np.abs(reconstructed), np.abs(accepted)))
    if np.any(np.abs(reconstructed+accepted) > tolerance):
        raise CoupledAMRRegisterLifecycleV98Error(
            "v98 register replacement identity does not close")
    placeholder = ScheduledRegisterPhaseV98(
        phase, interface, lanes, coarse_source, fine_sources, start,
        coarse_dt, fine_dt, area, coarse, slots, accepted, difference,
        correction, 1, counts, "")
    result = ScheduledRegisterPhaseV98(
        **{**placeholder.__dict__,
           "receipt_sha256": _hash(_phase_payload(placeholder))})
    validate_scheduled_register_phase_v98(result)
    if _cancelled(cancelled):
        raise InterruptedError("v98 register phase cancelled after work")
    return result


def validate_scheduled_register_phase_v98(
    value: ScheduledRegisterPhaseV98,
) -> None:
    if type(value) is not ScheduledRegisterPhaseV98:
        raise CoupledAMRRegisterLifecycleV98Error(
            "v98 register phase type differs")
    nl = len(value.lane_ids)
    coarse = _frozen(value.coarse_integral, (nl,), "coarse integral")
    slots = _frozen(
        value.fine_integral_slots, (FINE_SUBSTEPS, FINE_FACES, nl),
        "fine integral slots")
    _frozen(value.accepted_fine_integral, (nl,), "accepted fine integral")
    _frozen(value.coarse_minus_fine_integral, (nl,),
            "coarse-minus-fine integral")
    _frozen(value.coarse_cell_correction, (nl,),
            "coarse cell correction")
    accepted = np.sum(slots, axis=(0, 1), dtype=np.float64)
    difference = coarse-accepted
    if (value.phase not in PHASES or not value.lane_ids or
            len(set(value.lane_ids)) != nl or
            any(not isinstance(item, str) or not item
                for item in value.lane_ids) or
            len(value.fine_source_sha256) != FINE_SUBSTEPS or
            value.fine_source_sha256[0] == value.fine_source_sha256[1] or
            not math.isfinite(value.time_start_s) or
            value.time_start_s < 0.0 or
            not math.isfinite(value.coarse_dt_s) or
            value.coarse_dt_s <= 0.0 or
            len(value.fine_dt_s) != FINE_SUBSTEPS or
            any(not math.isfinite(item) or item <= 0.0
                for item in value.fine_dt_s) or
            value.coarse_replacement_count != 1 or
            not np.array_equal(value.fine_slot_consumption_count,
                               np.ones((2, 2), dtype=np.int32)) or
            value.fine_dt_s[0] != value.fine_dt_s[1] or
            value.coarse_dt_s != sum(value.fine_dt_s) or
            not math.isfinite(value.coarse_cell_area_m2) or
            value.coarse_cell_area_m2 <= 0.0 or
            not np.array_equal(value.accepted_fine_integral, accepted) or
            not np.array_equal(value.coarse_minus_fine_integral, difference) or
            not np.array_equal(
                value.coarse_cell_correction,
                difference/value.coarse_cell_area_m2) or
            value.receipt_sha256 != _hash(_phase_payload(value))):
        raise CoupledAMRRegisterLifecycleV98Error(
            "v98 register phase schedule, arrays, or identity differs")
    _sha(value.interface_fingerprint, "interface fingerprint")
    _sha(value.coarse_source_sha256, "coarse source")
    for source in value.fine_source_sha256:
        _sha(source, "fine source")


@dataclass(frozen=True)
class CoupledAMRTwoPhaseRegisterV98:
    interface_fingerprint: str
    parent_state_sha256: str
    hydraulic: ScheduledRegisterPhaseV98
    scalar_bed: ScheduledRegisterPhaseV98
    phase_order: tuple[str, str]
    child_state_publication_count: int
    density_reflux_count: int
    receipt_sha256: str


def _two_phase_payload(value: CoupledAMRTwoPhaseRegisterV98) -> dict:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID + ":two-phase",
        "interface": value.interface_fingerprint,
        "parent": value.parent_state_sha256,
        "hydraulic": value.hydraulic.receipt_sha256,
        "scalar_bed": value.scalar_bed.receipt_sha256,
        "phase_order": list(value.phase_order),
        "child_state_publication_count": value.child_state_publication_count,
        "density_reflux_count": value.density_reflux_count,
    }


def compose_two_phase_register_v98(
    parent_state_sha256: str,
    hydraulic: ScheduledRegisterPhaseV98,
    scalar_bed: ScheduledRegisterPhaseV98,
) -> CoupledAMRTwoPhaseRegisterV98:
    validate_scheduled_register_phase_v98(hydraulic)
    validate_scheduled_register_phase_v98(scalar_bed)
    parent = _sha(parent_state_sha256, "parent state")
    if (hydraulic.phase != PHASE_HYDRAULIC or
            scalar_bed.phase != PHASE_SCALAR_BED or
            hydraulic.interface_fingerprint !=
                scalar_bed.interface_fingerprint or
            hydraulic.time_start_s != scalar_bed.time_start_s or
            hydraulic.coarse_dt_s != scalar_bed.coarse_dt_s or
            hydraulic.fine_dt_s != scalar_bed.fine_dt_s):
        raise CoupledAMRRegisterLifecycleV98Error(
            "v98 hydraulic and scalar/bed register schedules differ")
    placeholder = CoupledAMRTwoPhaseRegisterV98(
        hydraulic.interface_fingerprint, parent, hydraulic, scalar_bed,
        PHASES, 0, 0, "")
    return CoupledAMRTwoPhaseRegisterV98(
        **{**placeholder.__dict__,
           "receipt_sha256": _hash(_two_phase_payload(placeholder))})


def validate_two_phase_register_v98(
    value: CoupledAMRTwoPhaseRegisterV98,
) -> None:
    if type(value) is not CoupledAMRTwoPhaseRegisterV98:
        raise CoupledAMRRegisterLifecycleV98Error(
            "v98 two-phase register type differs")
    validate_scheduled_register_phase_v98(value.hydraulic)
    validate_scheduled_register_phase_v98(value.scalar_bed)
    rebuilt = compose_two_phase_register_v98(
        value.parent_state_sha256, value.hydraulic, value.scalar_bed)
    if (value.phase_order != PHASES or
            value.child_state_publication_count != 0 or
            value.density_reflux_count != 0 or
            value.receipt_sha256 != rebuilt.receipt_sha256):
        raise CoupledAMRRegisterLifecycleV98Error(
            "v98 two-phase register identity or ownership differs")


@dataclass(frozen=True)
class CoupledAMRRegisterClockV98:
    execution_fingerprint: str
    accepted_macro_updates: int
    time_s: float
    accepted_chain_sha256: str
    last_register_sha256: str | None
    state_sha256: str


def _clock_token(value: CoupledAMRRegisterClockV98) -> str:
    return _hash({
        "schema": SCHEMA,
        "scope_id": SCOPE_ID + ":clock",
        "execution": value.execution_fingerprint,
        "accepted_macro_updates": value.accepted_macro_updates,
        "time_s": value.time_s,
        "accepted_chain": value.accepted_chain_sha256,
        "last_register": value.last_register_sha256,
    })


def resolve_initial_register_clock_v98(
    execution_fingerprint: str,
) -> CoupledAMRRegisterClockV98:
    execution = _sha(execution_fingerprint, "execution fingerprint")
    chain = _hash({"scope_id": SCOPE_ID, "execution": execution})
    placeholder = CoupledAMRRegisterClockV98(
        execution, 0, 0.0, chain, None, "")
    return CoupledAMRRegisterClockV98(
        **{**placeholder.__dict__, "state_sha256": _clock_token(placeholder)})


def validate_register_clock_v98(
    value: CoupledAMRRegisterClockV98,
    execution_fingerprint: str,
) -> None:
    execution = _sha(execution_fingerprint, "execution fingerprint")
    if (type(value) is not CoupledAMRRegisterClockV98 or
            value.execution_fingerprint != execution or
            value.accepted_macro_updates < 0 or
            value.accepted_macro_updates > MAX_MACRO_UPDATES or
            not math.isfinite(value.time_s) or value.time_s < 0.0 or
            value.state_sha256 != _clock_token(value)):
        raise CoupledAMRRegisterLifecycleV98Error(
            "v98 register clock is stale or cross-execution")
    _sha(value.accepted_chain_sha256, "accepted chain")
    if value.last_register_sha256 is not None:
        _sha(value.last_register_sha256, "last register")


@dataclass(frozen=True)
class CoupledAMRRegisterTrialV98:
    execution_fingerprint: str
    parent_state_sha256: str
    register: CoupledAMRTwoPhaseRegisterV98
    candidate: CoupledAMRRegisterClockV98
    trial_sha256: str


def _trial_token(value: CoupledAMRRegisterTrialV98) -> str:
    return _hash({
        "schema": SCHEMA,
        "scope_id": SCOPE_ID + ":trial",
        "execution": value.execution_fingerprint,
        "parent": value.parent_state_sha256,
        "register": value.register.receipt_sha256,
        "candidate": value.candidate.state_sha256,
    })


def begin_register_trial_v98(
    execution_fingerprint: str,
    parent: CoupledAMRRegisterClockV98,
    register: CoupledAMRTwoPhaseRegisterV98, *,
    cancelled: bool | Callable[[], bool] = False,
) -> CoupledAMRRegisterTrialV98:
    validate_register_clock_v98(parent, execution_fingerprint)
    validate_two_phase_register_v98(register)
    if _cancelled(cancelled):
        raise InterruptedError("v98 register transaction cancelled")
    if parent.accepted_macro_updates >= MAX_MACRO_UPDATES:
        raise CoupledAMRRegisterLifecycleV98Error(
            "v98 bounded macro-update limit was reached")
    if register.hydraulic.time_start_s != parent.time_s:
        raise CoupledAMRRegisterLifecycleV98Error(
            "v98 register does not start at the accepted macro time")
    chain = _hash({
        "previous": parent.accepted_chain_sha256,
        "register": register.receipt_sha256,
    })
    candidate0 = CoupledAMRRegisterClockV98(
        parent.execution_fingerprint, parent.accepted_macro_updates+1,
        parent.time_s+register.hydraulic.coarse_dt_s, chain,
        register.receipt_sha256, "")
    candidate = CoupledAMRRegisterClockV98(
        **{**candidate0.__dict__, "state_sha256": _clock_token(candidate0)})
    placeholder = CoupledAMRRegisterTrialV98(
        parent.execution_fingerprint, parent.state_sha256, register,
        candidate, "")
    result = CoupledAMRRegisterTrialV98(
        **{**placeholder.__dict__, "trial_sha256": _trial_token(placeholder)})
    validate_register_trial_v98(parent, result)
    if _cancelled(cancelled):
        raise InterruptedError("v98 register transaction cancelled after work")
    return result


def validate_register_trial_v98(
    parent: CoupledAMRRegisterClockV98,
    trial: CoupledAMRRegisterTrialV98,
) -> None:
    validate_register_clock_v98(parent, parent.execution_fingerprint)
    if type(trial) is not CoupledAMRRegisterTrialV98:
        raise CoupledAMRRegisterLifecycleV98Error(
            "v98 register trial type differs")
    validate_two_phase_register_v98(trial.register)
    validate_register_clock_v98(
        trial.candidate, parent.execution_fingerprint)
    if (trial.execution_fingerprint != parent.execution_fingerprint or
            trial.parent_state_sha256 != parent.state_sha256 or
            trial.candidate.accepted_macro_updates !=
                parent.accepted_macro_updates+1 or
            trial.candidate.time_s !=
                parent.time_s+trial.register.hydraulic.coarse_dt_s or
            trial.candidate.last_register_sha256 !=
                trial.register.receipt_sha256 or
            trial.trial_sha256 != _trial_token(trial)):
        raise CoupledAMRRegisterLifecycleV98Error(
            "v98 register trial is stale or cross-parent")


def commit_register_trial_v98(
    parent: CoupledAMRRegisterClockV98,
    trial: CoupledAMRRegisterTrialV98,
) -> CoupledAMRRegisterClockV98:
    validate_register_trial_v98(parent, trial)
    return trial.candidate


def rollback_register_trial_v98(
    parent: CoupledAMRRegisterClockV98,
    trial: CoupledAMRRegisterTrialV98,
) -> CoupledAMRRegisterClockV98:
    validate_register_trial_v98(parent, trial)
    return parent


def register_clock_checkpoint_v98(
    value: CoupledAMRRegisterClockV98,
) -> dict[str, object]:
    validate_register_clock_v98(value, value.execution_fingerprint)
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID + ":checkpoint",
        "execution_fingerprint": value.execution_fingerprint,
        "accepted_macro_updates": value.accepted_macro_updates,
        "time_s": value.time_s,
        "accepted_chain_sha256": value.accepted_chain_sha256,
        "last_register_sha256": value.last_register_sha256,
        "state_sha256": value.state_sha256,
    }


def load_register_clock_checkpoint_v98(
    execution_fingerprint: str,
    document: Mapping[str, object],
) -> CoupledAMRRegisterClockV98:
    expected = {
        "schema", "scope_id", "execution_fingerprint",
        "accepted_macro_updates", "time_s", "accepted_chain_sha256",
        "last_register_sha256", "state_sha256",
    }
    if not isinstance(document, Mapping) or set(document) != expected:
        raise CoupledAMRRegisterLifecycleV98Error(
            "v98 register checkpoint inventory differs")
    if (document["schema"] != SCHEMA or
            document["scope_id"] != SCOPE_ID + ":checkpoint"):
        raise CoupledAMRRegisterLifecycleV98Error(
            "v98 register checkpoint schema or scope differs")
    value = CoupledAMRRegisterClockV98(
        str(document["execution_fingerprint"]),
        int(document["accepted_macro_updates"]), float(document["time_s"]),
        str(document["accepted_chain_sha256"]),
        (None if document["last_register_sha256"] is None else
         str(document["last_register_sha256"])),
        str(document["state_sha256"]))
    validate_register_clock_v98(value, execution_fingerprint)
    return value


__all__ = [
    "CoupledAMRRegisterClockV98", "CoupledAMRRegisterLifecycleV98Error",
    "CoupledAMRRegisterTrialV98", "CoupledAMRTwoPhaseRegisterV98",
    "FINE_FACES", "FINE_SUBSTEPS", "MAX_MACRO_UPDATES",
    "PHASE_HYDRAULIC", "PHASE_SCALAR_BED", "PHASES", "SCHEMA",
    "SCOPE_ID", "ScheduledRegisterPhaseV98", "begin_register_trial_v98",
    "commit_register_trial_v98", "compose_two_phase_register_v98",
    "load_register_clock_checkpoint_v98", "register_clock_checkpoint_v98",
    "resolve_initial_register_clock_v98",
    "resolve_scheduled_register_phase_v98", "rollback_register_trial_v98",
    "validate_register_clock_v98", "validate_register_trial_v98",
    "validate_scheduled_register_phase_v98",
    "validate_two_phase_register_v98",
]
