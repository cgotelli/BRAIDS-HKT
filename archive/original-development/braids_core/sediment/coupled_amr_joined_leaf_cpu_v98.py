"""Atomic first joined-leaf 2:1 coupled AMR macro owner (CPU v98).

This scope is intentionally one authenticated coarse/fine interface.  It is
not a general hierarchy/regrid admission.  One still-open coarse v97 trial and
two private v98 fine substeps are reduced to one register correction, one
corrected coarse scientific state, and zero-time numerical parents for the
next bounded macro update.  Fine children never publish through the source
provider; the coarse Q3 provider is the only public transaction.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Callable

import numpy as np

from .coupled_amr_corrected_coarse_cpu_v98 import (
    CoupledAMRCorrectedLeafClosedCPUV98,
    CoupledAMRCorrectedLeafStateCPUV98,
    CoupledAMRCorrectedLeafTrialCPUV98,
    abort_closed_coupled_amr_corrected_coarse_cpu_v98,
    begin_coupled_amr_corrected_coarse_cpu_v98,
    checkpoint_coupled_amr_corrected_coarse_cpu_v98,
    close_coupled_amr_corrected_coarse_cpu_v98,
    publish_coupled_amr_corrected_coarse_cpu_v98,
    load_coupled_amr_corrected_coarse_checkpoint_v98,
    rollback_coupled_amr_corrected_coarse_cpu_v98,
    validate_coupled_amr_corrected_coarse_state_v98,
    validate_coupled_amr_corrected_coarse_trial_v98,
)
from .coupled_amr_corrected_images_native_v98 import (
    NativeAMRCorrectedImagesRuntimeV98,
)
from .coupled_amr_fine_substep_cpu_v98 import (
    CoupledAMRFineSubstepStateCPUV98,
    validate_coupled_amr_fine_substep_state_v98,
)
from .coupled_amr_numerical_parent_cpu_v98 import (
    AMRRebasedNumericalParentReceiptCPUV98,
    rebase_corrected_leaf_to_private_v97_parent_v98,
    validate_amr_rebased_numerical_parent_cpu_v98,
)
from .coupled_amr_q3_trace_cpu_v98 import CoupledAMRQ3TraceCPUV98
from .coupled_amr_register_cpu_v98 import (
    CoupledAMRCoarseCorrectionCPUV98,
    CoupledAMRInterfaceMapCPUV98,
    CoupledFaceTimeIntegralsCPUV98,
    build_coupled_amr_coarse_correction_cpu_v98,
    extract_coupled_face_time_integrals_cpu_v98,
    validate_coupled_amr_coarse_correction_cpu_v98,
    validate_coupled_amr_interface_map_cpu_v98,
)
from .coupled_amr_scalar_trace_cpu_v98 import CoupledAMRScalarTraceCPUV98
from .coupled_amr_trace_cpu_v98 import CoupledAMRInternalTraceCPUV98
from .coupled_amr_turbulence_trace_cpu_v98 import (
    CoupledAMRTurbulenceTraceCPUV98,
)
from .coupled_wetdry_owner_cpu_v97 import (
    CoupledWetDryCPUExecutionV97,
    CoupledWetDryCPUStateV97,
    CoupledWetDryCPUTrialV97,
    coupled_wetdry_cpu_checkpoint_v97,
    load_coupled_wetdry_cpu_checkpoint_v97,
)
from .hles_native_cpu_v97 import NativeHlesRuntimeV97
from .q3_wetdry_adapter_cpu_v97 import (
    CPUQ3V83ActiveWetDryExnerAdapterV97,
)


SCHEMA = 1
SCOPE_ID = "braids.coupled_amr.joined_leaf.cpu.v98"
CHECKPOINT_SCOPE_ID = SCOPE_ID + ":checkpoint"
SEMANTIC_ID = "one_joined_2to1_leaf_interface_two_fine_substeps"
BACKEND = "native_cpu_private_fine_single_public_coarse"
MAX_MACRO_UPDATES = 2
_SHA = re.compile(r"[0-9a-f]{64}\Z")


class CoupledAMRJoinedLeafCPUV98Error(ValueError):
    """The bounded joined-leaf macro transaction differs."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("ascii")


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


def _poll(value: bool | Callable[[], bool]) -> Callable[[], bool]:
    if isinstance(value, (bool, np.bool_)):
        return lambda: bool(value)
    if not callable(value):
        raise TypeError("cancelled must be Boolean or callable")
    return value


@dataclass(frozen=True)
class CoupledAMRJoinedLeafCountsCPUV98:
    coarse_provisional_update_count: int = 1
    private_fine_substep_count: int = 2
    hydraulic_register_application_count: int = 1
    scalar_bed_register_application_count: int = 1
    corrected_coarse_derived_rebuild_count: int = 1
    coarse_numerical_rebase_count: int = 1
    fine_source_provider_mutation_count: int = 0
    covered_parent_accepted_row_count: int = 0
    physical_boundary_surrogate_count: int = 0
    public_coarse_provider_publication_count: int = 1


@dataclass(frozen=True)
class CoupledAMRJoinedLeafStateCPUV98:
    coarse_execution_fingerprint: str
    fine_execution_fingerprint: str
    interface_fingerprint: str
    joined_geometry_receipt_sha256: str
    corrected_coarse: CoupledAMRCorrectedLeafStateCPUV98
    coarse_numerical_parent: CoupledWetDryCPUStateV97
    coarse_rebase_receipt: AMRRebasedNumericalParentReceiptCPUV98
    fine_final_state_sha256: str
    fine_numerical_parent: CoupledWetDryCPUStateV97
    coarse_provider_state_sha256: str
    fine_provider_state_sha256: str
    cumulative_water_volume_m3: float
    cumulative_momentum_m4_s: np.ndarray
    cumulative_suspended_mass_kg: np.ndarray
    cumulative_fluid_integral: np.ndarray
    cumulative_bed_mass_kg: np.ndarray
    accepted_macro_updates: int
    time_s: float
    previous_state_sha256: str | None
    counts: CoupledAMRJoinedLeafCountsCPUV98
    semantic_id: str
    backend: str
    state_sha256: str


def _state_payload(value: CoupledAMRJoinedLeafStateCPUV98) -> dict:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID+":state",
        "executions": [value.coarse_execution_fingerprint,
                       value.fine_execution_fingerprint],
        "interface": value.interface_fingerprint,
        "joined_geometry": value.joined_geometry_receipt_sha256,
        "corrected_coarse": value.corrected_coarse.state_sha256,
        "coarse_numerical_parent":
            value.coarse_numerical_parent.state_sha256,
        "coarse_rebase": value.coarse_rebase_receipt.receipt_sha256,
        "fine_final": value.fine_final_state_sha256,
        "fine_numerical_parent": value.fine_numerical_parent.state_sha256,
        "provider_states": [value.coarse_provider_state_sha256,
                            value.fine_provider_state_sha256],
        "cumulative_internal_ledgers_sha256": _digest(
            np.atleast_1d(value.cumulative_water_volume_m3),
            value.cumulative_momentum_m4_s,
            value.cumulative_suspended_mass_kg,
            value.cumulative_fluid_integral,
            value.cumulative_bed_mass_kg),
        "accepted_macro_updates": value.accepted_macro_updates,
        "time_s": value.time_s,
        "previous": value.previous_state_sha256,
        "counts": list(value.counts.__dict__.values()),
        "semantic_id": value.semantic_id,
        "backend": value.backend,
        "scope_limit": "one_joined_leaf_interface_not_full_regrid",
        "global_physical_boundary_ownership_qualified": False,
    }


@dataclass(frozen=True)
class CoupledAMRJoinedLeafTrialCPUV98:
    coarse_parent_state_sha256: str
    fine_parent_state_sha256: str
    coarse_provisional_trial_sha256: str
    fine_substep_state_sha256: tuple[str, str]
    fine_step1: CoupledAMRFineSubstepStateCPUV98
    fine_step2: CoupledAMRFineSubstepStateCPUV98
    coarse_integrals: CoupledFaceTimeIntegralsCPUV98
    correction: CoupledAMRCoarseCorrectionCPUV98
    corrected_trial: CoupledAMRCorrectedLeafTrialCPUV98
    candidate: CoupledAMRJoinedLeafStateCPUV98
    trial_sha256: str


def _trial_payload(value: CoupledAMRJoinedLeafTrialCPUV98) -> dict:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID+":trial",
        "coarse_parent": value.coarse_parent_state_sha256,
        "fine_parent": value.fine_parent_state_sha256,
        "coarse_provisional": value.coarse_provisional_trial_sha256,
        "fine_substeps": list(value.fine_substep_state_sha256),
        "fine_step_objects": [value.fine_step1.state_sha256,
                              value.fine_step2.state_sha256],
        "coarse_integrals": value.coarse_integrals.receipt_sha256,
        "correction": value.correction.receipt_sha256,
        "corrected_trial": value.corrected_trial.trial_sha256,
        "candidate": value.candidate.state_sha256,
        "publication_count": 0,
    }


def _provider_clone(
    execution: CoupledWetDryCPUExecutionV97,
    source: CPUQ3V83ActiveWetDryExnerAdapterV97,
    state: object,
) -> CPUQ3V83ActiveWetDryExnerAdapterV97:
    result = CPUQ3V83ActiveWetDryExnerAdapterV97(
        outer_execution=execution.active_execution,
        binding=source.binding, state=state,
        native_factcr_runtime=source.native_factcr_runtime,
        native_q3_face_runtime=source.native_q3_face_runtime,
        native_threads=source.native_factcr_threads)
    if (result.state is not state or result._open is not None or
            result._prepared is not None or result._closed is not None):
        raise CoupledAMRJoinedLeafCPUV98Error(
            "v98 joined-leaf provider clone differs")
    return result


def _readonly(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    raw = np.asarray(value)
    if (raw.shape != shape or raw.dtype.kind not in "fiu" or
            not np.isfinite(raw).all()):
        raise CoupledAMRJoinedLeafCPUV98Error(
            f"v98 joined-leaf {name} differs")
    result = np.array(raw, dtype=np.float64, order="C", copy=True)
    result.setflags(write=False)
    return result


def _prepare_coupled_amr_joined_leaf_cpu_v98(
    coarse_execution: CoupledWetDryCPUExecutionV97,
    coarse_parent: CoupledWetDryCPUStateV97,
    coarse_provider: CPUQ3V83ActiveWetDryExnerAdapterV97,
    coarse_provisional_trial: CoupledWetDryCPUTrialV97,
    fine_execution: CoupledWetDryCPUExecutionV97,
    fine_parent: CoupledWetDryCPUStateV97,
    fine_source_provider: CPUQ3V83ActiveWetDryExnerAdapterV97,
    fine_step1: CoupledAMRFineSubstepStateCPUV98,
    fine_provider1: CPUQ3V83ActiveWetDryExnerAdapterV97,
    fine_step2: CoupledAMRFineSubstepStateCPUV98,
    fine_provider2: CPUQ3V83ActiveWetDryExnerAdapterV97,
    interface: CoupledAMRInterfaceMapCPUV98, *,
    internal_trace: CoupledAMRInternalTraceCPUV98,
    scalar_trace: CoupledAMRScalarTraceCPUV98,
    q3_trace: CoupledAMRQ3TraceCPUV98,
    turbulence_trace: CoupledAMRTurbulenceTraceCPUV98,
    hles_runtime: NativeHlesRuntimeV97,
    native_corrected_images_runtime: NativeAMRCorrectedImagesRuntimeV98,
    closure_chezy_mhalf_s: object,
    previous: CoupledAMRJoinedLeafStateCPUV98 | None = None,
    cancelled: bool | Callable[[], bool] = False,
) -> CoupledAMRJoinedLeafTrialCPUV98:
    """Prepare every fallible macro result without public publication."""
    poll = _poll(cancelled)
    validate_coupled_amr_interface_map_cpu_v98(
        interface, coarse_execution, fine_execution)
    if (internal_trace.joined_geometry_receipt_sha256 is None or
            internal_trace.physical_boundary_distance_reuse_count != 0):
        raise CoupledAMRJoinedLeafCPUV98Error(
            "v98 joined-leaf macro requires authenticated global geometry")
    validate_coupled_amr_fine_substep_state_v98(
        fine_step1, fine_execution, fine_parent, fine_source_provider,
        internal_trace, scalar_trace, q3_trace, turbulence_trace,
        fine_provider1)
    validate_coupled_amr_fine_substep_state_v98(
        fine_step2, fine_execution, fine_step1.numerical_parent,
        fine_provider1, internal_trace, scalar_trace, q3_trace,
        turbulence_trace, fine_provider2)
    if (fine_step1.substep_index != 0 or fine_step2.substep_index != 1 or
            fine_step1.face_time_integrals.candidate_state_sha256 !=
                fine_step2.face_time_integrals.parent_state_sha256):
        raise CoupledAMRJoinedLeafCPUV98Error(
            "v98 joined-leaf private fine sequence differs")
    if (fine_step2.parent_state_sha256 !=
            fine_step1.numerical_parent.state_sha256 or
            fine_step2.source_provider_state_sha256 !=
            fine_provider1.state.state_sha256 or
            fine_source_provider.accepted_updates !=
                fine_parent.accepted_updates or
            fine_source_provider.time_s != fine_parent.time_s):
        raise CoupledAMRJoinedLeafCPUV98Error(
            "v98 joined-leaf fine parent/provider chain differs")
    if poll():
        raise InterruptedError("v98 joined-leaf cancelled before register")
    coarse_integrals = extract_coupled_face_time_integrals_cpu_v98(
        coarse_execution, coarse_parent, coarse_provider,
        coarse_provisional_trial)
    fine_integrals = (fine_step1.face_time_integrals,
                      fine_step2.face_time_integrals)
    correction = build_coupled_amr_coarse_correction_cpu_v98(
        interface, coarse_integrals, fine_integrals,
        coarse_execution=coarse_execution,
        fine_execution=fine_execution)
    corrected_trial = begin_coupled_amr_corrected_coarse_cpu_v98(
        coarse_execution, coarse_parent, coarse_provider,
        coarse_provisional_trial, interface, coarse_integrals,
        fine_integrals, correction, fine_execution=fine_execution,
        hles_runtime=hles_runtime,
        native_corrected_images_runtime=native_corrected_images_runtime,
        closure_chezy_mhalf_s=closure_chezy_mhalf_s,
        cancelled=poll)
    corrected = corrected_trial.candidate
    clone = _provider_clone(
        coarse_execution, coarse_provider, corrected.q3_state)
    rebased, rebase_receipt = \
        rebase_corrected_leaf_to_private_v97_parent_v98(
            coarse_execution, corrected, clone, interface, correction)
    if poll():
        raise InterruptedError(
            "v98 joined-leaf cancelled before candidate freeze")

    ledger = correction.internal_interface_ledger
    ng = coarse_execution.fraction_count
    if previous is None:
        macro_updates = 1
        prior_sha = None
        water = float(ledger.water_volume_m3)
        momentum = np.asarray(ledger.momentum_m4_s)
        suspended = np.asarray(ledger.suspended_mass_kg)
        fluid = np.asarray(ledger.fluid_integral)
        bed = np.asarray(ledger.bed_mass_kg)
    else:
        validate_coupled_amr_joined_leaf_state_v98(
            previous, coarse_execution, fine_execution, interface,
            coarse_provider=None, fine_provider=None)
        if (previous.accepted_macro_updates >= MAX_MACRO_UPDATES or
                coarse_parent.state_sha256 !=
                    previous.coarse_numerical_parent.state_sha256 or
                fine_parent.state_sha256 !=
                    previous.fine_numerical_parent.state_sha256 or
                coarse_parent.time_s != previous.time_s or
                fine_parent.time_s != previous.time_s):
            raise CoupledAMRJoinedLeafCPUV98Error(
                "v98 joined-leaf continuation parent differs")
        macro_updates = previous.accepted_macro_updates+1
        prior_sha = previous.state_sha256
        water = (previous.cumulative_water_volume_m3+
                 float(ledger.water_volume_m3))
        momentum = previous.cumulative_momentum_m4_s+ledger.momentum_m4_s
        suspended = (previous.cumulative_suspended_mass_kg+
                     ledger.suspended_mass_kg)
        fluid = previous.cumulative_fluid_integral+ledger.fluid_integral
        bed = previous.cumulative_bed_mass_kg+ledger.bed_mass_kg
    counts = CoupledAMRJoinedLeafCountsCPUV98()
    placeholder = CoupledAMRJoinedLeafStateCPUV98(
        coarse_execution.fingerprint, fine_execution.fingerprint,
        interface.fingerprint,
        internal_trace.joined_geometry_receipt_sha256,
        corrected, rebased, rebase_receipt,
        fine_step2.state_sha256, fine_step2.numerical_parent,
        corrected.q3_state.state_sha256,
        fine_provider2.state.state_sha256,
        water, _readonly(momentum, (2,), "cumulative momentum"),
        _readonly(suspended, (ng,), "cumulative suspended"),
        _readonly(fluid, (2,), "cumulative fluid"),
        _readonly(bed, (ng,), "cumulative bed"),
        macro_updates, corrected.time_s, prior_sha, counts,
        SEMANTIC_ID, BACKEND, "")
    candidate = CoupledAMRJoinedLeafStateCPUV98(
        **{**placeholder.__dict__,
           "state_sha256": _hash(_state_payload(placeholder))})
    trial0 = CoupledAMRJoinedLeafTrialCPUV98(
        coarse_parent.state_sha256, fine_parent.state_sha256,
        coarse_provisional_trial.trial_sha256,
        (fine_step1.state_sha256, fine_step2.state_sha256),
        fine_step1, fine_step2,
        coarse_integrals, correction, corrected_trial, candidate, "")
    trial = CoupledAMRJoinedLeafTrialCPUV98(
        **{**trial0.__dict__,
           "trial_sha256": _hash(_trial_payload(trial0))})
    validate_coupled_amr_joined_leaf_trial_v98(
        trial, coarse_execution, coarse_parent, coarse_provider,
        coarse_provisional_trial, fine_execution, fine_parent,
        fine_source_provider, fine_step1, fine_provider1, fine_step2,
        fine_provider2, interface, internal_trace=internal_trace,
        scalar_trace=scalar_trace, q3_trace=q3_trace,
        turbulence_trace=turbulence_trace, previous=previous)
    return trial


def prepare_coupled_amr_joined_leaf_cpu_v98(
    coarse_execution: CoupledWetDryCPUExecutionV97,
    coarse_parent: CoupledWetDryCPUStateV97,
    coarse_provider: CPUQ3V83ActiveWetDryExnerAdapterV97,
    coarse_provisional_trial: CoupledWetDryCPUTrialV97,
    *args, **kwargs,
) -> CoupledAMRJoinedLeafTrialCPUV98:
    """Prepare atomically; failure consumes no public coarse transaction."""
    try:
        provider_snapshot = (
            coarse_provisional_trial.active_trial.base_v93_trial.
            q3_parent_checkpoint)
    except (AttributeError, TypeError):
        provider_snapshot = None
    try:
        return _prepare_coupled_amr_joined_leaf_cpu_v98(
            coarse_execution, coarse_parent, coarse_provider,
            coarse_provisional_trial, *args, **kwargs)
    except Exception as original_error:
        try:
            open_trial = (coarse_provisional_trial.active_trial.
                          base_v93_trial.q3_open_trial)
        except (AttributeError, TypeError):
            open_trial = None
        cleanup_error = None
        if (type(coarse_provider) is
                CPUQ3V83ActiveWetDryExnerAdapterV97 and
                coarse_provider._open is open_trial):
            try:
                rollback_coupled_amr_corrected_coarse_cpu_v98(
                    coarse_execution, coarse_parent, coarse_provider,
                    coarse_provisional_trial)
            except Exception as error:
                cleanup_error = error
        if provider_snapshot is not None:
            try:
                coarse_provider.restore_checkpoint(provider_snapshot)
            except Exception as error:  # pragma: no cover - double failure
                cleanup_error = error
        if cleanup_error is not None:
            raise RuntimeError(
                "v98 joined-leaf prepare failed and provider restore also "
                "failed") from cleanup_error
        raise original_error


def validate_coupled_amr_joined_leaf_state_v98(
    value: CoupledAMRJoinedLeafStateCPUV98,
    coarse_execution: CoupledWetDryCPUExecutionV97,
    fine_execution: CoupledWetDryCPUExecutionV97,
    interface: CoupledAMRInterfaceMapCPUV98, *,
    coarse_provider: CPUQ3V83ActiveWetDryExnerAdapterV97 | None,
    fine_provider: CPUQ3V83ActiveWetDryExnerAdapterV97 | None,
) -> None:
    validate_coupled_amr_interface_map_cpu_v98(
        interface, coarse_execution, fine_execution)
    if (type(value) is not CoupledAMRJoinedLeafStateCPUV98 or
            value.coarse_execution_fingerprint !=
                coarse_execution.fingerprint or
            value.fine_execution_fingerprint != fine_execution.fingerprint or
            value.interface_fingerprint != interface.fingerprint or
            _SHA.fullmatch(value.joined_geometry_receipt_sha256) is None or
            value.accepted_macro_updates not in (1, 2) or
            value.corrected_coarse.accepted_updates !=
                value.accepted_macro_updates or
            value.fine_numerical_parent.accepted_updates !=
                2*value.accepted_macro_updates or
            value.time_s != value.corrected_coarse.time_s or
            value.time_s != value.fine_numerical_parent.time_s or
            value.coarse_numerical_parent.time_s != value.time_s or
            value.coarse_provider_state_sha256 !=
                value.corrected_coarse.q3_state.state_sha256 or
            _SHA.fullmatch(value.fine_provider_state_sha256) is None or
            _SHA.fullmatch(value.fine_final_state_sha256) is None or
            value.counts != CoupledAMRJoinedLeafCountsCPUV98() or
            value.semantic_id != SEMANTIC_ID or value.backend != BACKEND or
            value.state_sha256 != _hash(_state_payload(value))):
        raise CoupledAMRJoinedLeafCPUV98Error(
            "v98 joined-leaf state identity or clocks differ")
    ng = coarse_execution.fraction_count
    arrays = ((value.cumulative_momentum_m4_s, (2,)),
              (value.cumulative_suspended_mass_kg, (ng,)),
              (value.cumulative_fluid_integral, (2,)),
              (value.cumulative_bed_mass_kg, (ng,)))
    if (not np.isfinite(value.cumulative_water_volume_m3) or
            any(array.shape != shape or array.flags.writeable or
                not np.isfinite(array).all() for array, shape in arrays)):
        raise CoupledAMRJoinedLeafCPUV98Error(
            "v98 joined-leaf cumulative internal ledgers differ")
    if (coarse_provider is not None and
            coarse_provider.state.state_sha256 !=
                value.coarse_provider_state_sha256):
        raise CoupledAMRJoinedLeafCPUV98Error(
            "v98 joined-leaf coarse provider state differs")
    if (fine_provider is not None and
            fine_provider.state.state_sha256 !=
                value.fine_provider_state_sha256):
        raise CoupledAMRJoinedLeafCPUV98Error(
            "v98 joined-leaf fine provider state differs")


def validate_coupled_amr_joined_leaf_trial_v98(
    value: CoupledAMRJoinedLeafTrialCPUV98,
    coarse_execution: CoupledWetDryCPUExecutionV97,
    coarse_parent: CoupledWetDryCPUStateV97,
    coarse_provider: CPUQ3V83ActiveWetDryExnerAdapterV97,
    coarse_provisional_trial: CoupledWetDryCPUTrialV97,
    fine_execution: CoupledWetDryCPUExecutionV97,
    fine_parent: CoupledWetDryCPUStateV97,
    fine_source_provider: CPUQ3V83ActiveWetDryExnerAdapterV97,
    fine_step1: CoupledAMRFineSubstepStateCPUV98,
    fine_provider1: CPUQ3V83ActiveWetDryExnerAdapterV97,
    fine_step2: CoupledAMRFineSubstepStateCPUV98,
    fine_provider2: CPUQ3V83ActiveWetDryExnerAdapterV97,
    interface: CoupledAMRInterfaceMapCPUV98, *,
    internal_trace: CoupledAMRInternalTraceCPUV98,
    scalar_trace: CoupledAMRScalarTraceCPUV98,
    q3_trace: CoupledAMRQ3TraceCPUV98,
    turbulence_trace: CoupledAMRTurbulenceTraceCPUV98,
    previous: CoupledAMRJoinedLeafStateCPUV98 | None,
) -> None:
    validate_coupled_amr_fine_substep_state_v98(
        fine_step1, fine_execution, fine_parent, fine_source_provider,
        internal_trace, scalar_trace, q3_trace, turbulence_trace,
        fine_provider1)
    validate_coupled_amr_fine_substep_state_v98(
        fine_step2, fine_execution, fine_step1.numerical_parent,
        fine_provider1, internal_trace, scalar_trace, q3_trace,
        turbulence_trace, fine_provider2)
    validate_coupled_amr_coarse_correction_cpu_v98(
        value.correction, interface, value.coarse_integrals,
        (fine_step1.face_time_integrals, fine_step2.face_time_integrals),
        coarse_execution=coarse_execution,
        fine_execution=fine_execution)
    validate_coupled_amr_corrected_coarse_trial_v98(
        coarse_execution, coarse_parent, coarse_provider,
        coarse_provisional_trial, interface, value.coarse_integrals,
        (fine_step1.face_time_integrals, fine_step2.face_time_integrals),
        value.corrected_trial, fine_execution=fine_execution)
    clone = _provider_clone(
        coarse_execution, coarse_provider,
        value.corrected_trial.candidate.q3_state)
    validate_amr_rebased_numerical_parent_cpu_v98(
        value.candidate.coarse_rebase_receipt, coarse_execution,
        value.corrected_trial.candidate,
        value.candidate.coarse_numerical_parent, clone, interface,
        value.correction)
    if (type(value) is not CoupledAMRJoinedLeafTrialCPUV98 or
            value.coarse_parent_state_sha256 != coarse_parent.state_sha256 or
            value.fine_parent_state_sha256 != fine_parent.state_sha256 or
            value.coarse_provisional_trial_sha256 !=
                coarse_provisional_trial.trial_sha256 or
            value.fine_substep_state_sha256 !=
                (fine_step1.state_sha256, fine_step2.state_sha256) or
            value.fine_step1 is not fine_step1 or
            value.fine_step2 is not fine_step2 or
            value.candidate.corrected_coarse is not
                value.corrected_trial.candidate or
            value.candidate.fine_final_state_sha256 !=
                fine_step2.state_sha256 or
            value.candidate.joined_geometry_receipt_sha256 !=
                internal_trace.joined_geometry_receipt_sha256 or
            value.trial_sha256 != _hash(_trial_payload(value))):
        raise CoupledAMRJoinedLeafCPUV98Error(
            "v98 joined-leaf trial identity differs")
    validate_coupled_amr_joined_leaf_state_v98(
        value.candidate, coarse_execution, fine_execution, interface,
        coarse_provider=None, fine_provider=fine_provider2)
    expected_updates = 1 if previous is None else \
        previous.accepted_macro_updates+1
    if value.candidate.accepted_macro_updates != expected_updates:
        raise CoupledAMRJoinedLeafCPUV98Error(
            "v98 joined-leaf accepted macro clock differs")


def _commit_coupled_amr_joined_leaf_cpu_v98(
    trial: CoupledAMRJoinedLeafTrialCPUV98,
    coarse_execution: CoupledWetDryCPUExecutionV97,
    coarse_parent: CoupledWetDryCPUStateV97,
    coarse_provider: CPUQ3V83ActiveWetDryExnerAdapterV97,
    coarse_provisional_trial: CoupledWetDryCPUTrialV97,
    fine_execution: CoupledWetDryCPUExecutionV97,
    interface: CoupledAMRInterfaceMapCPUV98,
) -> CoupledAMRJoinedLeafStateCPUV98:
    """Close and publish the only public child after all work is frozen."""
    fine_integrals = (trial.fine_step1.face_time_integrals,
                      trial.fine_step2.face_time_integrals)
    if (type(trial) is not CoupledAMRJoinedLeafTrialCPUV98 or
            interface.fingerprint != trial.candidate.interface_fingerprint or
            tuple(item.receipt_sha256 for item in fine_integrals) !=
                trial.correction.fine_integrals_sha256 or
            trial.trial_sha256 != _hash(_trial_payload(trial)) or
            trial.candidate.state_sha256 !=
                _hash(_state_payload(trial.candidate))):
        raise CoupledAMRJoinedLeafCPUV98Error(
            "v98 joined-leaf commit handle differs")
    closed: CoupledAMRCorrectedLeafClosedCPUV98 | None = None
    try:
        closed = close_coupled_amr_corrected_coarse_cpu_v98(
            coarse_execution, coarse_parent, coarse_provider,
            coarse_provisional_trial, interface, trial.coarse_integrals,
            fine_integrals, trial.corrected_trial,
            fine_execution=fine_execution)
        publish_coupled_amr_corrected_coarse_cpu_v98(
            coarse_provider, trial.corrected_trial, closed)
        return trial.candidate
    except Exception:
        if (closed is not None and coarse_provider._closed is
                trial.corrected_trial.corrected_q3_prepared):
            abort_closed_coupled_amr_corrected_coarse_cpu_v98(
                coarse_provider, coarse_provisional_trial,
                trial.corrected_trial, closed)
        if coarse_provider._open is not None:
            rollback_coupled_amr_corrected_coarse_cpu_v98(
                coarse_execution, coarse_parent, coarse_provider,
                coarse_provisional_trial)
        raise


def commit_coupled_amr_joined_leaf_cpu_v98(
    trial: CoupledAMRJoinedLeafTrialCPUV98,
    coarse_execution: CoupledWetDryCPUExecutionV97,
    coarse_parent: CoupledWetDryCPUStateV97,
    coarse_provider: CPUQ3V83ActiveWetDryExnerAdapterV97,
    coarse_provisional_trial: CoupledWetDryCPUTrialV97,
    fine_execution: CoupledWetDryCPUExecutionV97,
    interface: CoupledAMRInterfaceMapCPUV98,
) -> CoupledAMRJoinedLeafStateCPUV98:
    """Commit atomically, including failures before corrected close."""
    try:
        provider_snapshot = (
            coarse_provisional_trial.active_trial.base_v93_trial.
            q3_parent_checkpoint)
    except (AttributeError, TypeError):
        provider_snapshot = None
    try:
        return _commit_coupled_amr_joined_leaf_cpu_v98(
            trial, coarse_execution, coarse_parent, coarse_provider,
            coarse_provisional_trial, fine_execution, interface)
    except Exception as original_error:
        try:
            open_trial = (coarse_provisional_trial.active_trial.
                          base_v93_trial.q3_open_trial)
        except (AttributeError, TypeError):
            open_trial = None
        cleanup_error = None
        if (type(coarse_provider) is
                CPUQ3V83ActiveWetDryExnerAdapterV97 and
                coarse_provider._open is open_trial):
            try:
                rollback_coupled_amr_corrected_coarse_cpu_v98(
                    coarse_execution, coarse_parent, coarse_provider,
                    coarse_provisional_trial)
            except Exception as error:
                cleanup_error = error
        if provider_snapshot is not None:
            try:
                coarse_provider.restore_checkpoint(provider_snapshot)
            except Exception as error:  # pragma: no cover - double failure
                cleanup_error = error
        if cleanup_error is not None:
            raise RuntimeError(
                "v98 joined-leaf commit failed and provider restore also "
                "failed") from cleanup_error
        raise original_error


def checkpoint_coupled_amr_joined_leaf_cpu_v98(
    state: CoupledAMRJoinedLeafStateCPUV98,
    coarse_execution: CoupledWetDryCPUExecutionV97,
    fine_execution: CoupledWetDryCPUExecutionV97,
    coarse_provider: CPUQ3V83ActiveWetDryExnerAdapterV97,
    fine_provider: CPUQ3V83ActiveWetDryExnerAdapterV97,
    interface: CoupledAMRInterfaceMapCPUV98,
    correction: CoupledAMRCoarseCorrectionCPUV98,
) -> dict[str, object]:
    """Serialize the bounded joined-leaf state without a covered row."""
    validate_coupled_amr_joined_leaf_state_v98(
        state, coarse_execution, fine_execution, interface,
        coarse_provider=coarse_provider, fine_provider=fine_provider)
    validate_coupled_amr_corrected_coarse_state_v98(
        state.corrected_coarse, coarse_execution, interface, correction,
        provider=coarse_provider)
    clone = _provider_clone(
        coarse_execution, coarse_provider,
        state.corrected_coarse.q3_state)
    validate_amr_rebased_numerical_parent_cpu_v98(
        state.coarse_rebase_receipt, coarse_execution,
        state.corrected_coarse, state.coarse_numerical_parent, clone,
        interface, correction)
    body = {
        "schema": SCHEMA,
        "scope_id": CHECKPOINT_SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "backend": BACKEND,
        "coarse_execution_fingerprint": coarse_execution.fingerprint,
        "fine_execution_fingerprint": fine_execution.fingerprint,
        "interface_fingerprint": interface.fingerprint,
        "joined_geometry_receipt_sha256":
            state.joined_geometry_receipt_sha256,
        "correction_receipt_sha256": correction.receipt_sha256,
        "coarse_corrected_checkpoint":
            checkpoint_coupled_amr_corrected_coarse_cpu_v98(
                coarse_execution, state.corrected_coarse,
                coarse_provider, interface, correction),
        "fine_numerical_checkpoint": coupled_wetdry_cpu_checkpoint_v97(
            fine_execution, state.fine_numerical_parent, fine_provider),
        "fine_final_state_sha256": state.fine_final_state_sha256,
        "cumulative_water_volume_m3":
            state.cumulative_water_volume_m3,
        "cumulative_momentum_m4_s":
            state.cumulative_momentum_m4_s.tolist(),
        "cumulative_suspended_mass_kg":
            state.cumulative_suspended_mass_kg.tolist(),
        "cumulative_fluid_integral":
            state.cumulative_fluid_integral.tolist(),
        "cumulative_bed_mass_kg":
            state.cumulative_bed_mass_kg.tolist(),
        "accepted_macro_updates": state.accepted_macro_updates,
        "time_s": state.time_s,
        "previous_state_sha256": state.previous_state_sha256,
        "counts": state.counts.__dict__,
        "state_sha256": state.state_sha256,
        "covered_parent_accepted_row_count": 0,
    }
    return {**body, "fingerprint": _hash(body)}


def load_coupled_amr_joined_leaf_checkpoint_v98(
    document: object,
    coarse_execution: CoupledWetDryCPUExecutionV97,
    fine_execution: CoupledWetDryCPUExecutionV97,
    coarse_provider: CPUQ3V83ActiveWetDryExnerAdapterV97,
    fine_provider: CPUQ3V83ActiveWetDryExnerAdapterV97,
    interface: CoupledAMRInterfaceMapCPUV98,
    correction: CoupledAMRCoarseCorrectionCPUV98,
    *, native_corrected_images_runtime: NativeAMRCorrectedImagesRuntimeV98,
) -> CoupledAMRJoinedLeafStateCPUV98:
    """Restore one bounded joined leaf and both exact provider images."""
    keys = {
        "schema", "scope_id", "semantic_id", "backend",
        "coarse_execution_fingerprint", "fine_execution_fingerprint",
        "interface_fingerprint", "joined_geometry_receipt_sha256",
        "correction_receipt_sha256", "coarse_corrected_checkpoint",
        "fine_numerical_checkpoint", "fine_final_state_sha256",
        "cumulative_water_volume_m3", "cumulative_momentum_m4_s",
        "cumulative_suspended_mass_kg", "cumulative_fluid_integral",
        "cumulative_bed_mass_kg", "accepted_macro_updates", "time_s",
        "previous_state_sha256", "counts", "state_sha256",
        "covered_parent_accepted_row_count", "fingerprint",
    }
    if not isinstance(document, dict) or set(document) != keys:
        raise CoupledAMRJoinedLeafCPUV98Error(
            "v98 joined-leaf checkpoint inventory differs")
    body = {key: value for key, value in document.items()
            if key != "fingerprint"}
    if (document["fingerprint"] != _hash(body) or
            document["schema"] != SCHEMA or
            document["scope_id"] != CHECKPOINT_SCOPE_ID or
            document["semantic_id"] != SEMANTIC_ID or
            document["backend"] != BACKEND or
            document["coarse_execution_fingerprint"] !=
                coarse_execution.fingerprint or
            document["fine_execution_fingerprint"] !=
                fine_execution.fingerprint or
            document["interface_fingerprint"] != interface.fingerprint or
            document["correction_receipt_sha256"] !=
                correction.receipt_sha256 or
            document["counts"] !=
                CoupledAMRJoinedLeafCountsCPUV98().__dict__ or
            document["accepted_macro_updates"] not in (1, 2) or
            document["covered_parent_accepted_row_count"] != 0 or
            _SHA.fullmatch(
                str(document["joined_geometry_receipt_sha256"])) is None or
            _SHA.fullmatch(str(document["fine_final_state_sha256"])) is None):
        raise CoupledAMRJoinedLeafCPUV98Error(
            "v98 joined-leaf checkpoint identity differs")
    coarse_before = coarse_provider.checkpoint()
    fine_before = fine_provider.checkpoint()
    try:
        corrected = load_coupled_amr_corrected_coarse_checkpoint_v98(
            coarse_execution, document["coarse_corrected_checkpoint"],
            coarse_provider, interface, correction,
            native_corrected_images_runtime=native_corrected_images_runtime)
        coarse_parent, rebase_receipt = \
            rebase_corrected_leaf_to_private_v97_parent_v98(
                coarse_execution, corrected, coarse_provider,
                interface, correction)
        fine_parent = load_coupled_wetdry_cpu_checkpoint_v97(
            fine_execution, document["fine_numerical_checkpoint"],
            fine_provider)
        ng = coarse_execution.fraction_count
        placeholder = CoupledAMRJoinedLeafStateCPUV98(
            coarse_execution.fingerprint, fine_execution.fingerprint,
            interface.fingerprint,
            str(document["joined_geometry_receipt_sha256"]),
            corrected, coarse_parent, rebase_receipt,
            str(document["fine_final_state_sha256"]), fine_parent,
            coarse_provider.state.state_sha256,
            fine_provider.state.state_sha256,
            float(document["cumulative_water_volume_m3"]),
            _readonly(document["cumulative_momentum_m4_s"], (2,),
                      "checkpoint cumulative momentum"),
            _readonly(document["cumulative_suspended_mass_kg"], (ng,),
                      "checkpoint cumulative suspended"),
            _readonly(document["cumulative_fluid_integral"], (2,),
                      "checkpoint cumulative fluid"),
            _readonly(document["cumulative_bed_mass_kg"], (ng,),
                      "checkpoint cumulative bed"),
            int(document["accepted_macro_updates"]),
            float(document["time_s"]),
            document["previous_state_sha256"],
            CoupledAMRJoinedLeafCountsCPUV98(), SEMANTIC_ID, BACKEND, "")
        result = CoupledAMRJoinedLeafStateCPUV98(
            **{**placeholder.__dict__,
               "state_sha256": _hash(_state_payload(placeholder))})
        validate_coupled_amr_joined_leaf_state_v98(
            result, coarse_execution, fine_execution, interface,
            coarse_provider=coarse_provider, fine_provider=fine_provider)
        if (result.state_sha256 != document["state_sha256"] or
                result.time_s != document["time_s"]):
            raise CoupledAMRJoinedLeafCPUV98Error(
                "v98 joined-leaf checkpoint state differs")
        return result
    except Exception as original_error:
        cleanup_error = None
        for provider, snapshot in (
                (fine_provider, fine_before),
                (coarse_provider, coarse_before)):
            try:
                provider.restore_checkpoint(snapshot)
            except Exception as error:  # pragma: no cover - double failure
                cleanup_error = error
        if cleanup_error is not None:
            raise RuntimeError(
                "v98 joined-leaf restart failed and provider restore also "
                "failed") from cleanup_error
        raise original_error


__all__ = [
    "BACKEND", "CHECKPOINT_SCOPE_ID", "MAX_MACRO_UPDATES", "SCHEMA",
    "SCOPE_ID", "SEMANTIC_ID",
    "CoupledAMRJoinedLeafCPUV98Error",
    "CoupledAMRJoinedLeafCountsCPUV98",
    "CoupledAMRJoinedLeafStateCPUV98",
    "CoupledAMRJoinedLeafTrialCPUV98",
    "commit_coupled_amr_joined_leaf_cpu_v98",
    "checkpoint_coupled_amr_joined_leaf_cpu_v98",
    "load_coupled_amr_joined_leaf_checkpoint_v98",
    "prepare_coupled_amr_joined_leaf_cpu_v98",
    "validate_coupled_amr_joined_leaf_state_v98",
    "validate_coupled_amr_joined_leaf_trial_v98",
]
