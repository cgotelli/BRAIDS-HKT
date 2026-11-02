"""Resident authenticated v67 MaximumWaterdepth direct-vector lifecycle.

This wrapper adds a content-addressed support topology to the existing v66
direct-vector lifecycle.  It owns no alternative constitutive or Exner
implementation: after enforcing the v67 receipt it delegates to that bounded
resident lifecycle and injects only the authenticated dwnvel support arrays.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping

import numpy as np

from braids_core.sediment.maximum_waterdepth_live_v67 import (
    MaximumWaterdepthAdmissionV67Error,
    _validate_runtime_binding,
    load_maximum_waterdepth_live_execution_v67,
    state_checkpoint_document_v67,
    unwrap_state_checkpoint_v67,
    verify_maximum_waterdepth_sources_v67,
)
from .direct_vector_live_v66 import (
    ResidentDirectVectorChannelsV66,
    ResidentDirectVectorLiveV66,
    ResidentDirectVectorStateV66,
    ResidentDirectVectorTrialV66,
)


BACKEND = "implicit_fwave_midpoint_gpu"
INTEGRATOR = "implicit_fwave_midpoint"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _fingerprint(value: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


@dataclass(frozen=True)
class ResidentMaximumWaterdepthChannelsV67:
    execution_fingerprint: str
    support_kind: str
    base: ResidentDirectVectorChannelsV66
    receipt_sha256: str
    device_id: int


@dataclass(frozen=True)
class ResidentMaximumWaterdepthStateV67:
    execution_fingerprint: str
    base: ResidentDirectVectorStateV66
    state_sha256: str
    device_id: int


@dataclass(frozen=True)
class ResidentMaximumWaterdepthTrialV67:
    execution_fingerprint: str
    parent_state_sha256: str
    base: ResidentDirectVectorTrialV66
    candidate: ResidentMaximumWaterdepthStateV67
    trial_sha256: str
    device_id: int


class ResidentMaximumWaterdepthDirectVectorLiveV67:
    """One-device v67 admission with no CPU or MaximumWaterdepth fallback."""

    backend = BACKEND
    integrator = INTEGRATOR

    def __init__(
        self, *, execution: Mapping[str, object], topology_fingerprint: str,
        cell_support_offset=None, cell_support_ids=None,
        trace_support_offset=None, trace_support_ids=None,
        verify_sources: bool = True,
    ):
        import cupy as cp

        self.cp = cp
        self.device_id = int(cp.cuda.Device().id)
        candidate = load_maximum_waterdepth_live_execution_v67(execution)
        if (candidate["backend"] != BACKEND or
                candidate["integrator"] != INTEGRATOR or
                candidate["automatic_backend_fallback"] is not False):
            raise MaximumWaterdepthAdmissionV67Error(
                "Q3-H6b/v67 GPU execution backend/integrator differs")
        enabled = bool(candidate["maximum_waterdepth"])
        arrays = (cell_support_offset, cell_support_ids,
                  trace_support_offset, trace_support_ids)
        if not enabled and any(value is not None for value in arrays):
            raise MaximumWaterdepthAdmissionV67Error(
                "Q3-H6b/v67 MaximumWaterdepth-off rejects support arrays")
        if enabled and any(value is None for value in arrays):
            raise MaximumWaterdepthAdmissionV67Error(
                "Q3-H6b/v67 MaximumWaterdepth-on requires complete support arrays")

        if enabled:
            cell_offset = self._resident_int64(
                "cell support offset", cell_support_offset)
            cell_ids = self._resident_int64("cell support IDs", cell_support_ids)
            trace_offset = self._resident_int64(
                "trace support offset", trace_support_offset)
            trace_ids = self._resident_int64(
                "trace support IDs", trace_support_ids)
            # Topology authentication is a setup operation.  Only these small,
            # immutable integer descriptors cross the host boundary; dynamic
            # depths and every trial/candidate remain resident.
            self.execution = _validate_runtime_binding(
                candidate, topology_fingerprint=topology_fingerprint,
                cell_support_offset=np.ascontiguousarray(cp.asnumpy(cell_offset)),
                cell_support_ids=np.ascontiguousarray(cp.asnumpy(cell_ids)),
                trace_support_offset=np.ascontiguousarray(cp.asnumpy(trace_offset)),
                trace_support_ids=np.ascontiguousarray(cp.asnumpy(trace_ids)))
            self._offset = {"cell": cell_offset, "trace": trace_offset}
            self._link_count = {
                "cell": int(cell_ids.size), "trace": int(trace_ids.size),
            }
            # Hold the ID buffers so a caller cannot recycle their allocation
            # while this authenticated runtime is alive.
            self._support_ids = {"cell": cell_ids, "trace": trace_ids}
        else:
            self.execution = _validate_runtime_binding(
                candidate, topology_fingerprint=topology_fingerprint,
                cell_support_offset=None, cell_support_ids=None,
                trace_support_offset=None, trace_support_ids=None)
            self._offset = {"cell": None, "trace": None}
            self._link_count = {"cell": 0, "trace": 0}
            self._support_ids = {"cell": None, "trace": None}
        self.enabled = enabled
        self.source_receipt = (verify_maximum_waterdepth_sources_v67(
            self.execution) if verify_sources else None)
        self.base = ResidentDirectVectorLiveV66(
            execution=self.execution["parent_execution"],
            verify_sources=False)
        if self.base.device_id != self.device_id:
            raise MaximumWaterdepthAdmissionV67Error(
                "Q3-H6b/v67 parent runtime selected another CUDA device")

    def _resident_int64(self, name: str, value):
        cp = self.cp
        if (not isinstance(value, cp.ndarray) or value.dtype != cp.int64 or
                value.ndim != 1 or not value.flags.c_contiguous):
            raise MaximumWaterdepthAdmissionV67Error(
                f"Q3-H6b/v67 {name} must be resident contiguous int64")
        if int(value.device.id) != self.device_id:
            raise MaximumWaterdepthAdmissionV67Error(
                f"Q3-H6b/v67 {name} is resident on another CUDA device")
        return value

    def _dynamic_support(self, kind: str, depth, active):
        cp = self.cp
        if kind not in {"cell", "trace"}:
            raise AssertionError(kind)
        if not self.enabled:
            if depth is not None or active is not None:
                raise MaximumWaterdepthAdmissionV67Error(
                    "Q3-H6b/v67 MaximumWaterdepth-off rejects support arrays")
            return {}
        if depth is None or active is None:
            raise MaximumWaterdepthAdmissionV67Error(
                f"Q3-H6b/v67 enabled {kind} evaluation lacks support data")
        count = self._link_count[kind]
        if (not isinstance(depth, cp.ndarray) or depth.dtype != cp.float64 or
                depth.ndim != 1 or not depth.flags.c_contiguous or
                int(depth.size) != count or int(depth.device.id) != self.device_id):
            raise MaximumWaterdepthAdmissionV67Error(
                f"Q3-H6b/v67 {kind} support depth differs from receipt")
        if (not isinstance(active, cp.ndarray) or active.dtype != cp.int32 or
                active.ndim != 1 or not active.flags.c_contiguous or
                int(active.size) != count or int(active.device.id) != self.device_id):
            raise MaximumWaterdepthAdmissionV67Error(
                f"Q3-H6b/v67 {kind} support activity differs from receipt")
        return {
            "maximum_waterdepth": True,
            "maximum_waterdepth_offset": self._offset[kind],
            "maximum_waterdepth_support_depth": depth,
            "maximum_waterdepth_support_active": active,
        }

    @staticmethod
    def _reject_override(values: Mapping[str, object]) -> None:
        forbidden = {
            "maximum_waterdepth", "maximum_waterdepth_offset",
            "maximum_waterdepth_support_depth",
            "maximum_waterdepth_support_active",
        }
        if forbidden.intersection(values):
            raise MaximumWaterdepthAdmissionV67Error(
                "Q3-H6b/v67 caller may not override authenticated support")

    def _wrap_channels(
        self, kind: str, base: ResidentDirectVectorChannelsV66,
    ) -> ResidentMaximumWaterdepthChannelsV67:
        token = _fingerprint({
            "execution_fingerprint": self.execution["fingerprint"],
            "support_kind": kind,
            "parent_channel_digest": base.digest,
        })
        return ResidentMaximumWaterdepthChannelsV67(
            self.execution["fingerprint"], kind, base, token, self.device_id)

    def _validate_channels(
        self, value: ResidentMaximumWaterdepthChannelsV67, kind: str,
    ) -> ResidentDirectVectorChannelsV66:
        if (not isinstance(value, ResidentMaximumWaterdepthChannelsV67) or
                value.device_id != self.device_id or
                value.execution_fingerprint != self.execution["fingerprint"] or
                value.support_kind != kind):
            raise MaximumWaterdepthAdmissionV67Error(
                "Q3-H6b/v67 channel receipt differs")
        expected = _fingerprint({
            "execution_fingerprint": value.execution_fingerprint,
            "support_kind": value.support_kind,
            "parent_channel_digest": value.base.digest,
        })
        if value.receipt_sha256 != expected:
            raise MaximumWaterdepthAdmissionV67Error(
                "Q3-H6b/v67 channel receipt is stale or forged")
        return value.base

    def cells(
        self, *, maximum_waterdepth_support_depth=None,
        maximum_waterdepth_support_active=None, **values,
    ) -> ResidentMaximumWaterdepthChannelsV67:
        self._reject_override(values)
        support = self._dynamic_support(
            "cell", maximum_waterdepth_support_depth,
            maximum_waterdepth_support_active)
        return self._wrap_channels("cell", self.base.cells(**values, **support))

    def traces(
        self, *, maximum_waterdepth_support_depth=None,
        maximum_waterdepth_support_active=None, **values,
    ) -> ResidentMaximumWaterdepthChannelsV67:
        self._reject_override(values)
        support = self._dynamic_support(
            "trace", maximum_waterdepth_support_depth,
            maximum_waterdepth_support_active)
        return self._wrap_channels("trace", self.base.traces(**values, **support))

    def _wrap_state(
        self, base: ResidentDirectVectorStateV66,
    ) -> ResidentMaximumWaterdepthStateV67:
        token = _fingerprint({
            "execution_fingerprint": self.execution["fingerprint"],
            "parent_state_sha256": base.state_sha256,
        })
        return ResidentMaximumWaterdepthStateV67(
            self.execution["fingerprint"], base, token, self.device_id)

    def _validate_state(
        self, state: ResidentMaximumWaterdepthStateV67,
    ) -> ResidentDirectVectorStateV66:
        if (not isinstance(state, ResidentMaximumWaterdepthStateV67) or
                state.device_id != self.device_id or
                state.execution_fingerprint != self.execution["fingerprint"]):
            raise MaximumWaterdepthAdmissionV67Error(
                "Q3-H6b/v67 state receipt differs")
        expected = _fingerprint({
            "execution_fingerprint": state.execution_fingerprint,
            "parent_state_sha256": state.base.state_sha256,
        })
        if state.state_sha256 != expected:
            raise MaximumWaterdepthAdmissionV67Error(
                "Q3-H6b/v67 state receipt is stale or forged")
        # The parent performs the array-content digest check at its next
        # mutation boundary; do it now as well to fail before unwrapping.
        self.base._validate_state(state.base)
        return state.base

    def resolve_state(self, **values) -> ResidentMaximumWaterdepthStateV67:
        return self._wrap_state(self.base.resolve_state(**values))

    def begin_trial(
        self, *, state: ResidentMaximumWaterdepthStateV67,
        cells: ResidentMaximumWaterdepthChannelsV67,
        traces: ResidentMaximumWaterdepthChannelsV67, **values,
    ) -> ResidentMaximumWaterdepthTrialV67:
        base_state = self._validate_state(state)
        base_cells = self._validate_channels(cells, "cell")
        base_traces = self._validate_channels(traces, "trace")
        base_trial = self.base.begin_trial(
            state=base_state, cells=base_cells, traces=base_traces, **values)
        candidate = self._wrap_state(base_trial.candidate)
        token = _fingerprint({
            "execution_fingerprint": self.execution["fingerprint"],
            "parent_state_sha256": state.state_sha256,
            "parent_trial_sha256": base_trial.trial_sha256,
            "candidate_state_sha256": candidate.state_sha256,
        })
        return ResidentMaximumWaterdepthTrialV67(
            self.execution["fingerprint"], state.state_sha256, base_trial,
            candidate, token, self.device_id)

    def _validate_trial(
        self, state: ResidentMaximumWaterdepthStateV67,
        trial: ResidentMaximumWaterdepthTrialV67,
    ) -> tuple[ResidentDirectVectorStateV66, ResidentDirectVectorTrialV66]:
        base_state = self._validate_state(state)
        if (not isinstance(trial, ResidentMaximumWaterdepthTrialV67) or
                trial.device_id != self.device_id or
                trial.execution_fingerprint != self.execution["fingerprint"] or
                trial.parent_state_sha256 != state.state_sha256):
            raise MaximumWaterdepthAdmissionV67Error(
                "Q3-H6b/v67 trial receipt differs")
        expected = _fingerprint({
            "execution_fingerprint": trial.execution_fingerprint,
            "parent_state_sha256": trial.parent_state_sha256,
            "parent_trial_sha256": trial.base.trial_sha256,
            "candidate_state_sha256": trial.candidate.state_sha256,
        })
        if trial.trial_sha256 != expected:
            raise MaximumWaterdepthAdmissionV67Error(
                "Q3-H6b/v67 trial receipt is stale or forged")
        if trial.candidate.base is not trial.base.candidate:
            raise MaximumWaterdepthAdmissionV67Error(
                "Q3-H6b/v67 candidate ownership differs")
        return base_state, trial.base

    def commit_trial(
        self, state: ResidentMaximumWaterdepthStateV67,
        trial: ResidentMaximumWaterdepthTrialV67,
    ) -> ResidentMaximumWaterdepthStateV67:
        base_state, base_trial = self._validate_trial(state, trial)
        accepted = self.base.commit_trial(base_state, base_trial)
        if accepted is not base_trial.candidate:
            raise MaximumWaterdepthAdmissionV67Error(
                "Q3-H6b/v67 parent committed an unexpected candidate")
        return trial.candidate

    def rollback_trial(
        self, state: ResidentMaximumWaterdepthStateV67,
        trial: ResidentMaximumWaterdepthTrialV67,
    ) -> ResidentMaximumWaterdepthStateV67:
        base_state, base_trial = self._validate_trial(state, trial)
        if self.base.rollback_trial(base_state, base_trial) is not base_state:
            raise MaximumWaterdepthAdmissionV67Error(
                "Q3-H6b/v67 parent rollback changed the accepted state")
        return state

    def cancel_trial(
        self, state: ResidentMaximumWaterdepthStateV67,
        trial: ResidentMaximumWaterdepthTrialV67,
    ) -> ResidentMaximumWaterdepthStateV67:
        base_state, base_trial = self._validate_trial(state, trial)
        if self.base.cancel_trial(base_state, base_trial) is not base_state:
            raise MaximumWaterdepthAdmissionV67Error(
                "Q3-H6b/v67 parent cancellation changed the accepted state")
        return state

    def state_checkpoint_document(
        self, state: ResidentMaximumWaterdepthStateV67,
    ) -> dict[str, object]:
        base_state = self._validate_state(state)
        # This explicit checkpoint is the sole accepted-state host crossing.
        portable_document = self.base.state_checkpoint_document(base_state)
        from braids_core.sediment.direct_vector_live_v66 import (
            load_state_checkpoint_v66,
        )
        portable_state = load_state_checkpoint_v66(
            portable_document, execution=self.execution["parent_execution"])
        return state_checkpoint_document_v67(
            portable_state, execution=self.execution)

    def load_state_checkpoint(
        self, value: Mapping[str, object],
    ) -> ResidentMaximumWaterdepthStateV67:
        base_document = unwrap_state_checkpoint_v67(
            value, execution=self.execution)
        return self._wrap_state(self.base.load_state_checkpoint(base_document))


__all__ = [
    "BACKEND", "INTEGRATOR", "ResidentMaximumWaterdepthChannelsV67",
    "ResidentMaximumWaterdepthDirectVectorLiveV67",
    "ResidentMaximumWaterdepthStateV67",
    "ResidentMaximumWaterdepthTrialV67",
]
