"""Authenticated lagged-v91 viscosity trace for the minimum CPU AMR gate.

The donor viscosity is not interpolated or recapped.  V91 evaluates ``chkvic``
once from the parent HLES handoff and holds that capped field for the complete
coarse update.  This adapter selects the exact coarse-cell value from that
accepted cap receipt and exposes it at both fine-substep midpoint traces.  It
owns no cap, clock, boundary memory, momentum operator, or accepted state.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re

import numpy as np

from .coupled_amr_trace_cpu_v98 import (
    ALPHAS,
    CoupledAMRInternalTraceCPUV98,
    validate_coupled_amr_internal_trace_structure_cpu_v98,
)
from .coupled_wetdry_owner_cpu_v97 import (
    CoupledWetDryCPUExecutionV97,
    CoupledWetDryCPUStateV97,
    CoupledWetDryCPUTrialV97,
    _validate_trial as _validate_v97_trial,
    _v92_execution,
)
from .shared_turbulence_consumers_v91 import (
    ChkvicUsedTotalReceiptV91,
    _cap_token,
    validate_chkvic_receipt_v91,
)
from .shared_turbulence_native_cpu_v97 import NativeMomentumRateV97


SCHEMA = 1
SCOPE_ID = "braids.coupled_amr.turbulence_trace.cpu.v98"
_SHA = re.compile(r"[0-9a-f]{64}\Z")


class CoupledAMRTurbulenceTraceCPUV98Error(ValueError):
    """Fail-closed lagged coarse viscosity-trace error."""


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
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(_canonical(list(array.shape)))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _readonly(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    raw = np.asarray(value)
    if (raw.shape != shape or raw.dtype.kind not in "fiu" or
            not np.isfinite(raw).all() or np.any(raw < 0.0)):
        raise CoupledAMRTurbulenceTraceCPUV98Error(
            f"v98 turbulence trace {name} differs")
    result = np.array(raw, dtype=np.float64, order="C", copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class CoupledAMRTurbulenceTraceCPUV98:
    base_trace_receipt_sha256: str
    coarse_execution_fingerprint: str
    fine_execution_fingerprint: str
    coarse_parent_state_sha256: str
    coarse_provisional_trial_sha256: str
    coarse_cap_receipt_sha256: str
    parent_hles_state_sha256: str
    current_handoff_receipt_sha256: str
    coarse_cell: int
    fine_faces: tuple[int, int]
    substep_alpha: np.ndarray
    trace_capped_vicuv_m2_s: np.ndarray
    trace_capped_dicuv_m2_s: np.ndarray
    upstream_cap_application_count: int
    trace_cap_application_count: int
    accepted_clock_increment_count: int
    physical_boundary_owner_count: int
    receipt_sha256: str


def _payload(value: CoupledAMRTurbulenceTraceCPUV98) -> dict:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "base_trace": value.base_trace_receipt_sha256,
        "coarse_execution": value.coarse_execution_fingerprint,
        "fine_execution": value.fine_execution_fingerprint,
        "coarse_parent": value.coarse_parent_state_sha256,
        "coarse_provisional": value.coarse_provisional_trial_sha256,
        "coarse_cap": value.coarse_cap_receipt_sha256,
        "parent_hles": value.parent_hles_state_sha256,
        "current_handoff": value.current_handoff_receipt_sha256,
        "coarse_cell": value.coarse_cell,
        "fine_faces": list(value.fine_faces),
        "arrays_sha256": _digest(
            value.substep_alpha, value.trace_capped_vicuv_m2_s,
            value.trace_capped_dicuv_m2_s),
        "upstream_cap_application_count":
            value.upstream_cap_application_count,
        "trace_cap_application_count": value.trace_cap_application_count,
        "accepted_clock_increment_count":
            value.accepted_clock_increment_count,
        "physical_boundary_owner_count":
            value.physical_boundary_owner_count,
        "selection": "source_ordered_lagged_parent_v91_cap",
        "interpolation_count": 0,
        "physical_boundary_surrogate": False,
    }


def _source_cap(
    coarse_execution: CoupledWetDryCPUExecutionV97,
    coarse_parent: CoupledWetDryCPUStateV97,
    coarse_provisional: CoupledWetDryCPUTrialV97,
) -> ChkvicUsedTotalReceiptV91:
    result = coarse_provisional.hydraulic_evidence.wetdry_trial.\
        shared_turbulence_result
    if (type(result) is not NativeMomentumRateV97 or
            type(result.cap_receipt) is not ChkvicUsedTotalReceiptV91):
        raise CoupledAMRTurbulenceTraceCPUV98Error(
            "v98 turbulence trace requires the actual coarse v91 result")
    cap = result.cap_receipt
    v91_execution = _v92_execution(
        coarse_execution.active_execution).base_execution
    validate_chkvic_receipt_v91(
        cap, v91_execution, coarse_parent.hles_state,
        coarse_provisional.current_handoff)
    if (cap.receipt_sha256 != _cap_token(cap) or
            cap.receipt_sha256 !=
                coarse_provisional.hydraulic_evidence.chkvic_receipt_sha256 or
            result.parent_hles_state_sha256 !=
                coarse_parent.hles_state.state_sha256 or
            cap.dt_s !=
                coarse_provisional.hydraulic_evidence.wetdry_trial.dt_s):
        raise CoupledAMRTurbulenceTraceCPUV98Error(
            "v98 coarse cap/result/hydraulic identity differs")
    return cap


def _resolve(
    coarse_execution: CoupledWetDryCPUExecutionV97,
    coarse_parent: CoupledWetDryCPUStateV97,
    coarse_provider: object,
    coarse_provisional: CoupledWetDryCPUTrialV97,
    fine_execution: CoupledWetDryCPUExecutionV97,
    base_trace: CoupledAMRInternalTraceCPUV98,
    *, validate_source: bool,
) -> CoupledAMRTurbulenceTraceCPUV98:
    if (type(coarse_execution) is not CoupledWetDryCPUExecutionV97 or
            type(coarse_parent) is not CoupledWetDryCPUStateV97 or
            type(coarse_provisional) is not CoupledWetDryCPUTrialV97 or
            type(fine_execution) is not CoupledWetDryCPUExecutionV97 or
            type(base_trace) is not CoupledAMRInternalTraceCPUV98):
        raise CoupledAMRTurbulenceTraceCPUV98Error(
            "v98 turbulence trace source type differs")
    if validate_source:
        _validate_v97_trial(
            coarse_execution, coarse_parent, coarse_provider,
            coarse_provisional)
        # The base trace was independently reconstructed (including its
        # joined global geometry) by its sole resolver.  This derived cap
        # trace receives no geometry objects, so it authenticates the frozen
        # hand-off plus the exact coarse owner identities instead of
        # replaying an unqualified physical-boundary half-distance trace.
        validate_coupled_amr_internal_trace_structure_cpu_v98(
            base_trace, fine_execution)
        if (base_trace.coarse_execution_fingerprint !=
                coarse_execution.fingerprint or
                base_trace.coarse_parent_state_sha256 !=
                    coarse_parent.state_sha256 or
                base_trace.coarse_provisional_trial_sha256 !=
                    coarse_provisional.trial_sha256 or
                base_trace.coarse_provisional_state_sha256 !=
                    coarse_provisional.candidate.state_sha256):
            raise CoupledAMRTurbulenceTraceCPUV98Error(
                "v98 turbulence trace coarse source identity differs")
    else:
        validate_coupled_amr_internal_trace_structure_cpu_v98(
            base_trace, fine_execution)
    cap = _source_cap(coarse_execution, coarse_parent, coarse_provisional)
    cell = base_trace.coarse_cell
    capped_vicuv = np.full(
        (2, 2), float(cap.capped_total_vicuv_m2_s[cell]),
        dtype=np.float64)
    capped_dicuv = np.full(
        (2, 2), float(cap.capped_total_dicuv_m2_s[cell]),
        dtype=np.float64)
    alpha = _readonly(base_trace.substep_alpha, (2,), "substep alpha")
    capped_vicuv = _readonly(
        capped_vicuv, (2, 2), "capped vicuv")
    capped_dicuv = _readonly(
        capped_dicuv, (2, 2), "capped dicuv")
    placeholder = CoupledAMRTurbulenceTraceCPUV98(
        base_trace.receipt_sha256, coarse_execution.fingerprint,
        fine_execution.fingerprint, coarse_parent.state_sha256,
        coarse_provisional.trial_sha256, cap.receipt_sha256,
        cap.parent_hles_state_sha256,
        cap.current_handoff_receipt_sha256, cell,
        base_trace.fine_faces, alpha, capped_vicuv, capped_dicuv,
        1, 0, 0, 0, "")
    return CoupledAMRTurbulenceTraceCPUV98(
        **{**placeholder.__dict__,
           "receipt_sha256": _hash(_payload(placeholder))})


def resolve_coupled_amr_turbulence_trace_cpu_v98(
    coarse_execution: CoupledWetDryCPUExecutionV97,
    coarse_parent: CoupledWetDryCPUStateV97,
    coarse_provider: object,
    coarse_provisional: CoupledWetDryCPUTrialV97,
    fine_execution: CoupledWetDryCPUExecutionV97,
    base_trace: CoupledAMRInternalTraceCPUV98,
) -> CoupledAMRTurbulenceTraceCPUV98:
    result = _resolve(
        coarse_execution, coarse_parent, coarse_provider,
        coarse_provisional, fine_execution, base_trace,
        validate_source=True)
    validate_coupled_amr_turbulence_trace_cpu_v98(
        result, coarse_execution, coarse_parent, coarse_provider,
        coarse_provisional, fine_execution, base_trace)
    return result


def validate_coupled_amr_turbulence_trace_structure_cpu_v98(
    value: CoupledAMRTurbulenceTraceCPUV98,
    fine_execution: CoupledWetDryCPUExecutionV97,
    base_trace: CoupledAMRInternalTraceCPUV98,
) -> None:
    if (type(value) is not CoupledAMRTurbulenceTraceCPUV98 or
            type(fine_execution) is not CoupledWetDryCPUExecutionV97 or
            type(base_trace) is not CoupledAMRInternalTraceCPUV98):
        raise CoupledAMRTurbulenceTraceCPUV98Error(
            "v98 turbulence trace structure type differs")
    validate_coupled_amr_internal_trace_structure_cpu_v98(
        base_trace, fine_execution)
    if (value.receipt_sha256 != _hash(_payload(value)) or
            value.base_trace_receipt_sha256 != base_trace.receipt_sha256 or
            value.fine_execution_fingerprint != fine_execution.fingerprint or
            value.coarse_cell != base_trace.coarse_cell or
            value.fine_faces != base_trace.fine_faces or
            not np.array_equal(value.substep_alpha,
                               base_trace.substep_alpha) or
            value.substep_alpha.flags.writeable or
            value.trace_capped_vicuv_m2_s.shape != (2, 2) or
            value.trace_capped_vicuv_m2_s.flags.writeable or
            not np.isfinite(value.trace_capped_vicuv_m2_s).all() or
            np.any(value.trace_capped_vicuv_m2_s < 0.0) or
            value.trace_capped_dicuv_m2_s.shape != (2, 2) or
            value.trace_capped_dicuv_m2_s.flags.writeable or
            not np.isfinite(value.trace_capped_dicuv_m2_s).all() or
            np.any(value.trace_capped_dicuv_m2_s < 0.0) or
            value.upstream_cap_application_count != 1 or
            value.trace_cap_application_count != 0 or
            value.accepted_clock_increment_count != 0 or
            value.physical_boundary_owner_count != 0):
        raise CoupledAMRTurbulenceTraceCPUV98Error(
            "v98 turbulence trace structure/owner identity differs")
    for digest in (
            value.base_trace_receipt_sha256,
            value.coarse_execution_fingerprint,
            value.fine_execution_fingerprint,
            value.coarse_parent_state_sha256,
            value.coarse_provisional_trial_sha256,
            value.coarse_cap_receipt_sha256,
            value.parent_hles_state_sha256,
            value.current_handoff_receipt_sha256,
            value.receipt_sha256):
        if _SHA.fullmatch(digest) is None:
            raise CoupledAMRTurbulenceTraceCPUV98Error(
                "v98 turbulence trace SHA-256 identity differs")


def validate_coupled_amr_turbulence_trace_cpu_v98(
    value: CoupledAMRTurbulenceTraceCPUV98,
    coarse_execution: CoupledWetDryCPUExecutionV97,
    coarse_parent: CoupledWetDryCPUStateV97,
    coarse_provider: object,
    coarse_provisional: CoupledWetDryCPUTrialV97,
    fine_execution: CoupledWetDryCPUExecutionV97,
    base_trace: CoupledAMRInternalTraceCPUV98,
) -> None:
    validate_coupled_amr_turbulence_trace_structure_cpu_v98(
        value, fine_execution, base_trace)
    expected = _resolve(
        coarse_execution, coarse_parent, coarse_provider,
        coarse_provisional, fine_execution, base_trace,
        validate_source=True)
    if (value.receipt_sha256 != expected.receipt_sha256 or
            not np.array_equal(value.trace_capped_vicuv_m2_s,
                               expected.trace_capped_vicuv_m2_s) or
            not np.array_equal(value.trace_capped_dicuv_m2_s,
                               expected.trace_capped_dicuv_m2_s)):
        raise CoupledAMRTurbulenceTraceCPUV98Error(
            "v98 turbulence trace is stale, mutated, or source-mismatched")


__all__ = [
    "CoupledAMRTurbulenceTraceCPUV98",
    "CoupledAMRTurbulenceTraceCPUV98Error",
    "SCHEMA", "SCOPE_ID",
    "resolve_coupled_amr_turbulence_trace_cpu_v98",
    "validate_coupled_amr_turbulence_trace_cpu_v98",
    "validate_coupled_amr_turbulence_trace_structure_cpu_v98",
]
