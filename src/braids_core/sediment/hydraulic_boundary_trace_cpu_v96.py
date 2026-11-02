"""Authenticated adapter for the sole hydraulic open-boundary trace owner.

The production cut-cell/GPU boundary machinery resolves a physical open-face
state and its corrected physical flux from the current nonlinear iterate.  It
is *not* a ghost state for a second Riemann solve.  This module gives the v96
CPU research owner a narrow, typed adapter to that existing owner without
reimplementing any inlet/outlet law.

The adapter is intentionally algebra/evidence infrastructure.  A caller may
use the ``manufactured_algebra_delegate`` kind in focused tests, but that does
not qualify any physical boundary mode.  Product qualification requires the
``cutcell_solver_trace_adapter`` or exact
``implicit_midpoint_boundary_trace_adapter`` kind backed by the actually
loaded CPU/GPU boundary runtime and source identities.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Callable, Mapping

import numpy as np


SCHEMA = 1
SCOPE_ID = "braids-q4-hydraulic-boundary-trace-cpu-v96-v1"
SEMANTIC_ID = SCOPE_ID + ":sole-owner-physical-trace-and-flux"
KIND_INTERNAL = 1
KIND_CLOSED = 2
KIND_OPEN = 3
OWNER_CUTCELL = "cutcell_solver_trace_adapter"
OWNER_MANUFACTURED = "manufactured_algebra_delegate"
OWNER_ACTUAL_CUDA = "implicit_midpoint_boundary_trace_adapter"
OWNER_KINDS = (OWNER_CUTCELL, OWNER_ACTUAL_CUDA, OWNER_MANUFACTURED)
_SHA256 = __import__("re").compile(r"[0-9a-f]{64}\Z")


class HydraulicBoundaryTraceV96Error(ValueError):
    """The v96 boundary adapter execution or receipt is invalid."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError, OverflowError) as error:
        raise HydraulicBoundaryTraceV96Error(
            "v96 boundary value is not finite canonical JSON") from error


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


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise HydraulicBoundaryTraceV96Error(
            f"v96 boundary {name} must be a lowercase SHA-256")
    return value


def _integer_array(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != shape or raw.dtype.kind not in "iu":
        raise HydraulicBoundaryTraceV96Error(
            f"v96 boundary {name} must be integer with shape {shape}")
    result = np.ascontiguousarray(raw, dtype=np.int32)
    result.setflags(write=False)
    return result


def _float_array(value: object, shape: tuple[int, ...], name: str, *,
                 positive: bool = False) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != shape or raw.dtype.kind not in "fiub":
        raise HydraulicBoundaryTraceV96Error(
            f"v96 boundary {name} must be numeric with shape {shape}")
    result = np.ascontiguousarray(raw, dtype=np.float64)
    if not np.isfinite(result).all() or (positive and np.any(result <= 0.0)):
        raise HydraulicBoundaryTraceV96Error(
            f"v96 boundary {name} contains invalid values")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class HydraulicBoundaryTraceExecutionV96:
    owner: np.ndarray
    neighbour: np.ndarray
    boundary_index: np.ndarray
    boundary_mode: np.ndarray
    face_distance_m: np.ndarray
    hydraulic_owner_kind: str
    hydraulic_owner_execution_sha256: str
    hydraulic_owner_source_sha256: str
    hydraulic_owner_runtime_sha256: str
    semantic_id: str
    fingerprint: str

    @property
    def face_count(self) -> int:
        return int(self.owner.size)

    @property
    def cell_count(self) -> int:
        interior = self.neighbour[self.neighbour >= 0]
        maximum = int(np.max(self.owner))
        if interior.size:
            maximum = max(maximum, int(np.max(interior)))
        return maximum+1

    @property
    def open_mask(self) -> np.ndarray:
        value = np.ascontiguousarray(
            (self.neighbour < 0) & (self.boundary_index >= 0))
        value.setflags(write=False)
        return value


def _execution_payload(value: HydraulicBoundaryTraceExecutionV96) -> dict:
    kind = np.where(value.neighbour >= 0, KIND_INTERNAL,
                    np.where(value.boundary_index >= 0,
                             KIND_OPEN, KIND_CLOSED)).astype(np.int32)
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "semantic_id": value.semantic_id,
        "backend": "delegated_exact_hydraulic_boundary_owner",
        "topology_sha256": _digest(
            value.owner, value.neighbour, value.boundary_index,
            value.boundary_mode, value.face_distance_m, kind),
        "hydraulic_owner_kind": value.hydraulic_owner_kind,
        "hydraulic_owner_execution_sha256":
            value.hydraulic_owner_execution_sha256,
        "hydraulic_owner_source_sha256": value.hydraulic_owner_source_sha256,
        "hydraulic_owner_runtime_sha256": value.hydraulic_owner_runtime_sha256,
        "open_mask": "(neighbour<0)&(boundary_index>=0)",
        "closed_mask": "(neighbour<0)&(boundary_index<0)",
        "boundary_mode_role": "audit_only_not_open_mask",
        "trace_role": "physical_open_face_state_not_riemann_right_state",
        "flux_role": "authoritative_corrected_physical_open_face_flux",
        "physical_boundary_qualification":
            value.hydraulic_owner_kind in (OWNER_CUTCELL, OWNER_ACTUAL_CUDA),
    }


def resolve_hydraulic_boundary_trace_execution_v96(
    *, owner: object, neighbour: object, boundary_index: object,
    boundary_mode: object, face_distance_m: object,
    hydraulic_owner_kind: str,
    hydraulic_owner_execution_sha256: str,
    hydraulic_owner_source_sha256: str,
    hydraulic_owner_runtime_sha256: str,
) -> HydraulicBoundaryTraceExecutionV96:
    raw_owner = np.asarray(owner)
    if raw_owner.ndim != 1:
        raise HydraulicBoundaryTraceV96Error(
            "v96 boundary owner must be one-dimensional")
    nf = int(raw_owner.size)
    own = _integer_array(raw_owner, (nf,), "owner")
    nei = _integer_array(neighbour, (nf,), "neighbour")
    bid = _integer_array(boundary_index, (nf,), "boundary index")
    mode = _integer_array(boundary_mode, (nf,), "boundary mode")
    distance = _float_array(
        face_distance_m, (nf,), "owner-to-trace distance", positive=True)
    if hydraulic_owner_kind not in OWNER_KINDS:
        raise HydraulicBoundaryTraceV96Error(
            "v96 boundary owner kind is unsupported")
    if (np.any(own < 0) or np.any(nei < -1) or
            np.any((nei >= 0) & (bid >= 0)) or
            np.any((nei >= 0) & (mode != 0)) or
            np.any((nei < 0) & (bid < 0) & (mode != 0)) or
            np.any((nei < 0) & (bid >= 0) & (mode <= 0)) or
            not np.any((nei < 0) & (bid >= 0))):
        raise HydraulicBoundaryTraceV96Error(
            "v96 boundary topology/open mask or mode differs")
    placeholder = HydraulicBoundaryTraceExecutionV96(
        own, nei, bid, mode, distance, hydraulic_owner_kind,
        _sha(hydraulic_owner_execution_sha256, "owner execution"),
        _sha(hydraulic_owner_source_sha256, "owner source"),
        _sha(hydraulic_owner_runtime_sha256, "owner runtime"),
        SEMANTIC_ID, "")
    return HydraulicBoundaryTraceExecutionV96(
        **{**placeholder.__dict__,
           "fingerprint": _fingerprint(_execution_payload(placeholder))})


def execution_document_hydraulic_boundary_trace_v96(
    value: HydraulicBoundaryTraceExecutionV96,
) -> dict:
    if type(value) is not HydraulicBoundaryTraceExecutionV96:
        raise HydraulicBoundaryTraceV96Error(
            "v96 boundary execution type differs")
    payload = _execution_payload(value)
    if (value.semantic_id != SEMANTIC_ID or
            value.hydraulic_owner_kind not in OWNER_KINDS or
            value.fingerprint != _fingerprint(payload)):
        raise HydraulicBoundaryTraceV96Error(
            "v96 boundary execution fingerprint differs")
    return {**payload, "fingerprint": value.fingerprint}


@dataclass(frozen=True)
class HydraulicBoundaryTraceReceiptV96:
    execution_fingerprint: str
    hydraulic_request_sha256: str
    parent_state_sha256: str
    evaluation_index: int
    evaluation_time_s: float
    midpoint_q_sha256: str
    bed_elevation_sha256: str
    trace_q: np.ndarray
    physical_open_face_flux: np.ndarray
    physical_boundary_flux_owned_mask: np.ndarray
    exterior_densin_active_mask: np.ndarray
    reconstruction_receipt_sha256: str
    boundary_branch_receipt_sha256: str
    owner_evaluation_sha256: str
    owner_state_before_sha256: str
    owner_state_after_sha256: str
    open_face_count: int
    exterior_densin_active_face_count: int
    inflow_face_count: int
    outflow_face_count: int
    stagnant_face_count: int
    owner_call_count: int
    receipt_sha256: str


def _receipt_token(value: HydraulicBoundaryTraceReceiptV96) -> str:
    return _fingerprint({
        "scope_id": SCOPE_ID + ":trace-receipt",
        "execution_fingerprint": value.execution_fingerprint,
        "hydraulic_request_sha256": value.hydraulic_request_sha256,
        "parent_state_sha256": value.parent_state_sha256,
        "evaluation_index": value.evaluation_index,
        "evaluation_time_s": value.evaluation_time_s,
        "midpoint_q_sha256": value.midpoint_q_sha256,
        "bed_elevation_sha256": value.bed_elevation_sha256,
        "arrays_sha256": _digest(
            value.trace_q, value.physical_open_face_flux,
            value.physical_boundary_flux_owned_mask,
            value.exterior_densin_active_mask),
        "reconstruction_receipt_sha256":
            value.reconstruction_receipt_sha256,
        "boundary_branch_receipt_sha256":
            value.boundary_branch_receipt_sha256,
        "owner_evaluation_sha256": value.owner_evaluation_sha256,
        "owner_state_before_sha256": value.owner_state_before_sha256,
        "owner_state_after_sha256": value.owner_state_after_sha256,
        "counts": [value.open_face_count,
                   value.exterior_densin_active_face_count,
                   value.inflow_face_count,
                   value.outflow_face_count, value.stagnant_face_count,
                   value.owner_call_count],
        "call_order": (
            "current_midpoint_to_sole_boundary_owner_then_shared_trace_flux"),
    })


class HydraulicBoundaryTraceProviderV96:
    """Exact adapter around an established, externally owned trace resolver."""

    __slots__ = ("execution", "_resolver", "_state_probe")

    def __init__(
        self, execution: HydraulicBoundaryTraceExecutionV96, *,
        resolver: Callable[..., Mapping[str, object]],
        state_probe: Callable[[], object],
    ) -> None:
        execution_document_hydraulic_boundary_trace_v96(execution)
        if not callable(resolver) or not callable(state_probe):
            raise HydraulicBoundaryTraceV96Error(
                "v96 boundary adapter requires callable owner hooks")
        self.execution = execution
        self._resolver = resolver
        self._state_probe = state_probe

    def _state_sha256(self) -> str:
        return _fingerprint(self._state_probe())

    def resolve(
        self, midpoint_q: object, bed_elevation_m: object, *,
        hydraulic_request_sha256: str, parent_state_sha256: str,
        evaluation_index: int, evaluation_time_s: float,
        evaluation_dt_s: float | None = None,
    ) -> HydraulicBoundaryTraceReceiptV96:
        execution_document_hydraulic_boundary_trace_v96(self.execution)
        nf = self.execution.face_count
        q = _float_array(
            midpoint_q, (3, self.execution.cell_count), "midpoint state")
        bed = _float_array(
            bed_elevation_m, (q.shape[1],), "bed elevation")
        request = _sha(hydraulic_request_sha256, "hydraulic request")
        parent = _sha(parent_state_sha256, "parent state")
        if type(evaluation_index) is not int or evaluation_index < 1:
            raise HydraulicBoundaryTraceV96Error(
                "v96 boundary evaluation index is invalid")
        time = float(evaluation_time_s)
        dt = None if evaluation_dt_s is None else float(evaluation_dt_s)
        if (not math.isfinite(time) or time < 0.0 or
                (dt is not None and
                 (not math.isfinite(dt) or dt <= 0.0))):
            raise HydraulicBoundaryTraceV96Error(
                "v96 boundary evaluation time/timestep is invalid")
        before = self._state_sha256()
        q_view = np.array(q, copy=True, order="C")
        bed_view = np.array(bed, copy=True, order="C")
        q_view.setflags(write=False)
        bed_view.setflags(write=False)
        output = self._resolver(
            midpoint_q=q_view, bed_elevation_m=bed_view,
            evaluation_index=evaluation_index, evaluation_time_s=time,
            evaluation_dt_s=dt,
            hydraulic_request_sha256=request,
            parent_state_sha256=parent,
            execution=self.execution)
        after = self._state_sha256()
        if before != after:
            raise HydraulicBoundaryTraceV96Error(
                "v96 boundary owner mutated accepted state during resolution")
        if not isinstance(output, Mapping) or set(output) != {
                "trace_q", "physical_open_face_flux",
                "physical_boundary_flux_owned_mask",
                "exterior_densin_active_mask",
                "reconstruction_receipt_sha256",
                "boundary_branch_receipt_sha256",
                "owner_evaluation_sha256"}:
            raise HydraulicBoundaryTraceV96Error(
                "v96 boundary owner output contract differs")
        trace = _float_array(output["trace_q"], (3, nf), "physical trace")
        flux = _float_array(
            output["physical_open_face_flux"], (3, nf),
            "physical open-face flux")
        physical_raw = np.asarray(
            output["physical_boundary_flux_owned_mask"])
        densin_raw = np.asarray(output["exterior_densin_active_mask"])
        if (physical_raw.shape != (nf,) or
                physical_raw.dtype.kind not in "biu" or
                densin_raw.shape != (nf,) or
                densin_raw.dtype.kind not in "biu" or
                np.any((physical_raw != 0) & (physical_raw != 1)) or
                np.any((densin_raw != 0) & (densin_raw != 1))):
            raise HydraulicBoundaryTraceV96Error(
                "v96 boundary physical/density masks must be Boolean face data")
        physical = np.ascontiguousarray(physical_raw, dtype=np.int32)
        densin = np.ascontiguousarray(densin_raw, dtype=np.int32)
        physical.setflags(write=False)
        densin.setflags(write=False)
        reconstruction = _sha(
            output["reconstruction_receipt_sha256"], "reconstruction")
        branch = _sha(
            output["boundary_branch_receipt_sha256"], "boundary branch")
        owner_evaluation = _sha(
            output["owner_evaluation_sha256"], "owner evaluation")
        open_mask = self.execution.open_mask
        physical_mask = physical.astype(bool)
        densin_mask = densin.astype(bool)
        if (not np.array_equal(physical_mask, open_mask) or
                np.any(densin_mask & ~physical_mask) or
                not np.all(trace[:, ~physical_mask] == 0.0) or
                not np.all(flux[:, ~physical_mask] == 0.0) or
                np.any(trace[0, physical_mask] < 0.0) or
                np.any(trace[0, densin_mask] <= 0.0)):
            raise HydraulicBoundaryTraceV96Error(
                "v96 boundary owner physical-flux/density mask differs")
        water = flux[0, physical_mask]
        scale = np.maximum(1.0, np.abs(water))
        stationary = np.abs(water) <= 32*np.finfo(float).eps*scale
        open_count = int(np.count_nonzero(open_mask))
        densin_count = int(np.count_nonzero(densin_mask))
        inflow = int(np.count_nonzero((water < 0.0) & ~stationary))
        outflow = int(np.count_nonzero((water > 0.0) & ~stationary))
        stagnant = int(np.count_nonzero(stationary))
        placeholder = HydraulicBoundaryTraceReceiptV96(
            self.execution.fingerprint, request, parent, evaluation_index,
            time, _digest(q), _digest(bed), trace, flux, physical, densin,
            reconstruction, branch, owner_evaluation,
            before, after, open_count, densin_count,
            inflow, outflow, stagnant, 1, "")
        return HydraulicBoundaryTraceReceiptV96(
            **{**placeholder.__dict__,
               "receipt_sha256": _receipt_token(placeholder)})


def validate_hydraulic_boundary_trace_receipt_v96(
    value: HydraulicBoundaryTraceReceiptV96,
    execution: HydraulicBoundaryTraceExecutionV96, *,
    midpoint_q: object, bed_elevation_m: object,
    hydraulic_request_sha256: str, parent_state_sha256: str,
    evaluation_index: int, evaluation_time_s: float,
) -> None:
    execution_document_hydraulic_boundary_trace_v96(execution)
    if type(value) is not HydraulicBoundaryTraceReceiptV96:
        raise HydraulicBoundaryTraceV96Error(
            "v96 boundary trace receipt type differs")
    nc = execution.cell_count
    q = _float_array(midpoint_q, (3, nc), "receipt midpoint")
    bed = _float_array(bed_elevation_m, (nc,), "receipt bed")
    trace = _float_array(value.trace_q, (3, execution.face_count),
                         "receipt trace")
    flux = _float_array(value.physical_open_face_flux,
                        (3, execution.face_count), "receipt flux")
    physical_raw = np.asarray(value.physical_boundary_flux_owned_mask)
    densin_raw = np.asarray(value.exterior_densin_active_mask)
    if (physical_raw.shape != (execution.face_count,) or
            physical_raw.dtype.kind not in "biu" or
            densin_raw.shape != (execution.face_count,) or
            densin_raw.dtype.kind not in "biu" or
            np.any((physical_raw != 0) & (physical_raw != 1)) or
            np.any((densin_raw != 0) & (densin_raw != 1))):
        raise HydraulicBoundaryTraceV96Error(
            "v96 boundary receipt physical/density masks differ")
    physical = np.ascontiguousarray(physical_raw, dtype=np.int32)
    densin = np.ascontiguousarray(densin_raw, dtype=np.int32)
    open_mask = execution.open_mask
    physical_mask = physical.astype(bool)
    densin_mask = densin.astype(bool)
    water = flux[0, physical_mask]
    scale = np.maximum(1.0, np.abs(water))
    stationary = np.abs(water) <= 32*np.finfo(float).eps*scale
    expected_counts = (
        int(np.count_nonzero(open_mask)),
        int(np.count_nonzero(densin_mask)),
        int(np.count_nonzero((water < 0.0) & ~stationary)),
        int(np.count_nonzero((water > 0.0) & ~stationary)),
        int(np.count_nonzero(stationary)), 1)
    if (value.execution_fingerprint != execution.fingerprint or
            value.hydraulic_request_sha256 !=
                _sha(hydraulic_request_sha256, "hydraulic request") or
            value.parent_state_sha256 !=
                _sha(parent_state_sha256, "parent state") or
            value.evaluation_index != evaluation_index or
            value.evaluation_time_s != float(evaluation_time_s) or
            value.midpoint_q_sha256 != _digest(q) or
            value.bed_elevation_sha256 != _digest(bed) or
            value.owner_state_before_sha256 !=
                value.owner_state_after_sha256 or
            tuple((value.open_face_count,
                   value.exterior_densin_active_face_count,
                   value.inflow_face_count, value.outflow_face_count,
                   value.stagnant_face_count,
                   value.owner_call_count)) != expected_counts or
            _SHA256.fullmatch(value.reconstruction_receipt_sha256) is None or
            _SHA256.fullmatch(value.boundary_branch_receipt_sha256) is None or
            _SHA256.fullmatch(value.owner_evaluation_sha256) is None or
            _SHA256.fullmatch(value.owner_state_before_sha256) is None or
            _SHA256.fullmatch(value.owner_state_after_sha256) is None or
            not np.array_equal(physical_mask, open_mask) or
            np.any(densin_mask & ~physical_mask) or
            not np.all(trace[:, ~physical_mask] == 0.0) or
            not np.all(flux[:, ~physical_mask] == 0.0) or
            np.any(trace[0, physical_mask] < 0.0) or
            np.any(trace[0, densin_mask] <= 0.0) or
            value.receipt_sha256 != _receipt_token(value)):
        raise HydraulicBoundaryTraceV96Error(
            "v96 boundary trace receipt is stale or mismatched")


def method_identity_hydraulic_boundary_trace_v96() -> dict[str, object]:
    return {
        "scope_id": SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "open_mask": "(neighbour<0)&(boundary_index>=0)",
        "trace": "current_iterate_physical_open_face_state",
        "flux": "same_owner_corrected_physical_open_face_flux",
        "not_a_riemann_ghost": True,
        "manufactured_delegate_qualifies_only_algebra": True,
        "physical_mode_qualification_requires": [
            OWNER_CUTCELL, OWNER_ACTUAL_CUDA],
        "accepted_state_mutation_during_resolution": False,
    }


__all__ = [
    "KIND_CLOSED", "KIND_INTERNAL", "KIND_OPEN", "OWNER_CUTCELL",
    "OWNER_ACTUAL_CUDA",
    "OWNER_MANUFACTURED", "HydraulicBoundaryTraceExecutionV96",
    "HydraulicBoundaryTraceProviderV96", "HydraulicBoundaryTraceReceiptV96",
    "HydraulicBoundaryTraceV96Error",
    "execution_document_hydraulic_boundary_trace_v96",
    "method_identity_hydraulic_boundary_trace_v96",
    "resolve_hydraulic_boundary_trace_execution_v96",
    "validate_hydraulic_boundary_trace_receipt_v96",
]
