"""Resident CUDA v69 ISlope=2 extension of the v67 direct-vector family.

The wrapper delegates every constitutive evaluation to the authenticated v67
family and every inventory/Exner update to its law-neutral consumer.  Its only
new operation is the already authenticated H4 face-direction composer with an
internally allocated exact-zero ``taurat`` field.  No CPU fallback exists.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

from braids_core.sediment.direct_vector_positive_taurat_v69 import (
    DirectVectorPositiveTauratV69Error,
    load_direct_vector_positive_taurat_execution_v69,
    verify_direct_vector_positive_taurat_sources_v69,
)
from .direct_vector_face_exner_v67 import (
    ResidentDirectVectorFaceExnerV67,
    ResidentDirectVectorSlopeFieldsV69,
    ResidentDirectVectorStateV67,
    ResidentDirectVectorTrialV67,
)
from .direct_vector_family_live_v67 import (
    ResidentDirectVectorFamilyChannelsV67,
    ResidentDirectVectorFamilyLiveV67,
    ResidentDirectVectorFamilyTopologyV67,
)
from .flow_bed_slope_composition import ResidentFlowBedSlopeFaceComposer


SCOPE_ID = "braids-q3h7-gpu-direct-vector-positive-taurat-v69-v1"
CHECKPOINT_SCOPE_ID = SCOPE_ID + ":checkpoint"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


@dataclass(frozen=True)
class ResidentDirectVectorPositiveTauratTrialV69:
    execution_fingerprint: str
    parent_state_sha256: str
    topology_sha256: str
    cell_channel_sha256: str
    trace_channel_sha256: str
    slope_sha256: str
    face_trial: ResidentDirectVectorTrialV67
    trial_sha256: str
    device_id: int


class ResidentDirectVectorPositiveTauratV69:
    """One positive-only v67 producer/ISlope2/inventory/Exner lifecycle."""

    def __init__(
        self,
        *,
        execution: Mapping[str, object],
        family: ResidentDirectVectorFamilyLiveV67,
        slope_composer: ResidentFlowBedSlopeFaceComposer,
    ):
        if not isinstance(family, ResidentDirectVectorFamilyLiveV67):
            raise DirectVectorPositiveTauratV69Error(
                "Q3-H7/v69 requires the authenticated v67 CUDA family"
            )
        self.family = family
        self.root = Path(__file__).resolve().parents[1]
        self.execution = load_direct_vector_positive_taurat_execution_v69(
            execution, repository_root=self.root,
        )
        if (self.execution["backend"] != "implicit_fwave_midpoint_gpu" or
                self.execution["integrator"] != "implicit_fwave_midpoint" or
                self.execution["automatic_backend_fallback"] is not False or
                self.execution["parent_execution"] != family.execution):
            raise DirectVectorPositiveTauratV69Error(
                "Q3-H7/v69 CUDA wrapper parent/backend differs"
            )
        if (not isinstance(slope_composer, ResidentFlowBedSlopeFaceComposer) or
                slope_composer.device_id != family.device_id or
                slope_composer.identity !=
                    self.execution["bed_slope_execution"]):
            raise DirectVectorPositiveTauratV69Error(
                "Q3-H7/v69 resident slope composer identity differs"
            )
        self.source_receipt = verify_direct_vector_positive_taurat_sources_v69(
            self.execution, repository_root=self.root,
        )
        self.cp = family.cp
        self.device_id = family.device_id
        self.face_consumer = ResidentDirectVectorFaceExnerV67(
            execution_fingerprint=family.execution["fingerprint"],
            bulk_density_kg_m3=family.bulk_density,
            accepted_step_limit=int(self.execution["accepted_step_limit"]),
            slope_composer=slope_composer,
            slope_extension_fingerprint=self.execution["fingerprint"],
        )
        if self.face_consumer.device_id != self.device_id:
            raise DirectVectorPositiveTauratV69Error(
                "Q3-H7/v69 components selected different CUDA devices"
            )

    def cells(self, **values) -> ResidentDirectVectorFamilyChannelsV67:
        return self.family.cells(**values)

    def traces(self, **values) -> ResidentDirectVectorFamilyChannelsV67:
        return self.family.traces(**values)

    def prepare_topology(self, **values) -> ResidentDirectVectorFamilyTopologyV67:
        topology = self.family.prepare_topology(**values)
        self.face_consumer._validate_topology_receipt(topology.face_topology)
        return topology

    def resolve_state(self, **values) -> ResidentDirectVectorStateV67:
        state = self.family.resolve_state(**values)
        self.face_consumer._validate_state(state)
        return state

    def resolve_slope_fields(
        self,
        *,
        topology: ResidentDirectVectorFamilyTopologyV67,
        downhill: object,
        depth: object,
        mean_diameter: object,
        hiding: object,
    ) -> ResidentDirectVectorSlopeFieldsV69:
        self.family._validate_topology(topology)
        result = self.face_consumer.resolve_positive_taurat_slope_fields_v69(
            topology=topology.face_topology,
            downhill=downhill,
            depth=depth,
            d50=self.family.d50,
            mean_diameter=mean_diameter,
            hiding=hiding,
        )
        if not self.family._flag(self.cp.array_equal(
                result.d50, self.family.d50)):
            raise DirectVectorPositiveTauratV69Error(
                "Q3-H7/v69 slope D50 differs from the family execution"
            )
        return result

    def _trial_token(
        self,
        *,
        state: ResidentDirectVectorStateV67,
        topology: ResidentDirectVectorFamilyTopologyV67,
        cells: ResidentDirectVectorFamilyChannelsV67,
        traces: ResidentDirectVectorFamilyChannelsV67,
        slope: ResidentDirectVectorSlopeFieldsV69,
        face_trial: ResidentDirectVectorTrialV67,
    ) -> str:
        return _fingerprint({
            "scope_id": SCOPE_ID,
            "execution_fingerprint": self.execution["fingerprint"],
            "parent_execution_fingerprint":
                self.family.execution["fingerprint"],
            "parent_state_sha256": state.state_sha256,
            "candidate_state_sha256": face_trial.candidate.state_sha256,
            "topology_sha256": topology.topology_sha256,
            "cell_channel_sha256": cells.channel_sha256,
            "trace_channel_sha256": traces.channel_sha256,
            "slope_sha256": slope.slope_sha256,
            "face_trial_sha256": face_trial.trial_sha256,
            "taurat_zero_application_count": 1,
            "bed_slope_application_count": 1,
            "active_fraction_application_count": 1,
            "inventory_fill_application_count": 1,
        })

    def begin_trial(
        self,
        *,
        state: ResidentDirectVectorStateV67,
        cells: ResidentDirectVectorFamilyChannelsV67,
        traces: ResidentDirectVectorFamilyChannelsV67,
        topology: ResidentDirectVectorFamilyTopologyV67,
        slope: ResidentDirectVectorSlopeFieldsV69,
        dt_s: float,
        morphological_factor: float = 1.0,
        left_bed: object | None = None,
        right_bed: object | None = None,
        cancelled=lambda: False,
    ) -> ResidentDirectVectorPositiveTauratTrialV69:
        self.family._validate_channels(cells, point_role="cells")
        self.family._validate_channels(traces, point_role="traces")
        self.family._validate_topology(topology)
        face_trial = self.face_consumer.begin_trial(
            state=state, cells=cells.face_channels,
            traces=traces.face_channels, topology=topology.face_topology,
            slope=slope, dt_s=dt_s,
            morphological_factor=morphological_factor,
            left_bed=left_bed, right_bed=right_bed,
            cancelled=cancelled,
        )
        token = self._trial_token(
            state=state, topology=topology, cells=cells, traces=traces,
            slope=slope, face_trial=face_trial,
        )
        return ResidentDirectVectorPositiveTauratTrialV69(
            self.execution["fingerprint"], state.state_sha256,
            topology.topology_sha256, cells.channel_sha256,
            traces.channel_sha256, slope.slope_sha256, face_trial,
            token, self.device_id,
        )

    def _validate_trial(
        self,
        state: ResidentDirectVectorStateV67,
        trial: ResidentDirectVectorPositiveTauratTrialV69,
    ) -> None:
        if (not isinstance(trial,
                           ResidentDirectVectorPositiveTauratTrialV69) or
                trial.execution_fingerprint != self.execution["fingerprint"] or
                trial.parent_state_sha256 != state.state_sha256 or
                trial.device_id != self.device_id or
                trial.face_trial.slope_receipt_sha256 != trial.slope_sha256):
            raise DirectVectorPositiveTauratV69Error(
                "Q3-H7/v69 trial parent or slope receipt differs"
            )
        self.face_consumer._validate_trial(state, trial.face_trial)
        expected = _fingerprint({
            "scope_id": SCOPE_ID,
            "execution_fingerprint": self.execution["fingerprint"],
            "parent_execution_fingerprint":
                self.family.execution["fingerprint"],
            "parent_state_sha256": state.state_sha256,
            "candidate_state_sha256": trial.face_trial.candidate.state_sha256,
            "topology_sha256": trial.topology_sha256,
            "cell_channel_sha256": trial.cell_channel_sha256,
            "trace_channel_sha256": trial.trace_channel_sha256,
            "slope_sha256": trial.slope_sha256,
            "face_trial_sha256": trial.face_trial.trial_sha256,
            "taurat_zero_application_count": 1,
            "bed_slope_application_count": 1,
            "active_fraction_application_count": 1,
            "inventory_fill_application_count": 1,
        })
        if expected != trial.trial_sha256:
            raise DirectVectorPositiveTauratV69Error(
                "Q3-H7/v69 trial receipt is stale or mutated"
            )

    def commit_trial(
        self, state: ResidentDirectVectorStateV67,
        trial: ResidentDirectVectorPositiveTauratTrialV69,
    ) -> ResidentDirectVectorStateV67:
        self._validate_trial(state, trial)
        return self.face_consumer.commit_trial(state, trial.face_trial)

    def rollback_trial(
        self, state: ResidentDirectVectorStateV67,
        trial: ResidentDirectVectorPositiveTauratTrialV69,
    ) -> ResidentDirectVectorStateV67:
        self._validate_trial(state, trial)
        return self.face_consumer.rollback_trial(state, trial.face_trial)

    def cancel_trial(
        self, state: ResidentDirectVectorStateV67,
        trial: ResidentDirectVectorPositiveTauratTrialV69,
    ) -> ResidentDirectVectorStateV67:
        self._validate_trial(state, trial)
        return self.face_consumer.cancel_trial(state, trial.face_trial)

    def state_checkpoint_document(
        self, state: ResidentDirectVectorStateV67,
    ) -> dict[str, object]:
        nested = self.family.state_checkpoint_document(state)
        body = {
            "scope_id": CHECKPOINT_SCOPE_ID,
            "execution_fingerprint": self.execution["fingerprint"],
            "parent_execution_fingerprint":
                self.family.execution["fingerprint"],
            "family_checkpoint": nested,
        }
        return {**body, "fingerprint": _fingerprint(body)}

    def load_state_checkpoint(
        self, value: Mapping[str, object],
    ) -> ResidentDirectVectorStateV67:
        if not isinstance(value, Mapping):
            raise DirectVectorPositiveTauratV69Error(
                "Q3-H7/v69 checkpoint must be an object"
            )
        try:
            body = {key: value[key] for key in value if key != "fingerprint"}
            valid = (
                set(value) == {
                    "scope_id", "execution_fingerprint",
                    "parent_execution_fingerprint", "family_checkpoint",
                    "fingerprint",
                }
                and value["scope_id"] == CHECKPOINT_SCOPE_ID
                and value["execution_fingerprint"] ==
                    self.execution["fingerprint"]
                and value["parent_execution_fingerprint"] ==
                    self.family.execution["fingerprint"]
                and value["fingerprint"] == _fingerprint(body)
            )
        except (KeyError, TypeError, ValueError):
            valid = False
        if not valid:
            raise DirectVectorPositiveTauratV69Error(
                "Q3-H7/v69 checkpoint identity differs"
            )
        state = self.family.load_state_checkpoint(value["family_checkpoint"])
        self.face_consumer._validate_state(state)
        return state


__all__ = [
    "ResidentDirectVectorPositiveTauratTrialV69",
    "ResidentDirectVectorPositiveTauratV69", "SCOPE_ID",
]
