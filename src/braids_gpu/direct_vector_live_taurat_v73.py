"""Resident CUDA lifecycle carrying Van Rijn 1993's live ``taurat``.

The established v66 CUDA producer returns ``diagnostics[2]`` from the same
kernel invocation that produces the bed vector.  This module preserves that
row in an authenticated v73 receipt, applies Delft's cell-to-face projection,
and then delegates slope correction, inventory limiting and Exner to the
existing resident CUDA owners.  It never recomputes the ratio from an
unrelated shear diagnostic and has no CPU fallback.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from braids_core.sediment.channel_composition import TRA_BEDLOAD
from braids_core.sediment.classic_flow_taurat_v69 import (
    TAURAT_POSITIVE_ZERO,
    TAURAT_TRAM_LIVE,
)
from braids_core.sediment.direct_vector_family_live_v67 import (
    METHOD_VAN_RIJN_1993_CURRENT,
)
from braids_core.sediment.direct_vector_live_taurat_v73 import (
    DirectVectorLiveTauratV73Error,
    STRESS_SEMANTICS,
    load_direct_vector_live_taurat_execution_v73,
    verify_direct_vector_live_taurat_sources_v73,
)

from .direct_vector_face_exner_v67 import (
    FACE_CLOSED,
    FACE_INTERNAL,
    ResidentDirectVectorFaceExnerV67,
    ResidentDirectVectorSlopeFieldsV69,
    ResidentDirectVectorStateV67,
    ResidentDirectVectorTrialV67,
)
from .direct_vector_family_live_v67 import (
    SCOPE_ID as PARENT_SCOPE_ID,
    ResidentDirectVectorFamilyChannelsV67,
    ResidentDirectVectorFamilyLiveV67,
    ResidentDirectVectorFamilyTopologyV67,
    ResidentDirectVectorProducerGroupV67,
)
from .flow_bed_slope_composition import ResidentFlowBedSlopeFaceComposer


SCOPE_ID = "braids-q3h7-gpu-vr1993-live-taurat-v73-v1"
CHECKPOINT_SCOPE_ID = SCOPE_ID + ":checkpoint"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


@dataclass(frozen=True)
class ResidentDirectVectorLiveTauratGroupV73(
        ResidentDirectVectorProducerGroupV67):
    """A v67 producer group plus same-invocation selector-owned stress."""

    producer_taurat: object
    stress_semantics: str
    taurat_receipt_sha256: str


@dataclass(frozen=True)
class ResidentDirectVectorLiveTauratChannelsV73:
    """Topology-independent cell/trace channel carrying live ``taurat``."""

    parent: ResidentDirectVectorFamilyChannelsV67
    producer_taurat: object
    stress_semantics: tuple[str, ...]
    taurat_receipt_sha256: str
    device_id: int

    def __getattr__(self, name: str):
        return getattr(self.parent, name)


@dataclass(frozen=True)
class ResidentDirectVectorLiveTauratTrialV73:
    execution_fingerprint: str
    parent_state_sha256: str
    topology_sha256: str
    cell_taurat_receipt_sha256: str
    trace_taurat_receipt_sha256: str
    slope_sha256: str
    face_trial: ResidentDirectVectorTrialV67
    trial_sha256: str
    device_id: int

    @property
    def candidate(self) -> ResidentDirectVectorStateV67:
        return self.face_trial.candidate


class ResidentDirectVectorLiveTauratFaceExnerV73(
        ResidentDirectVectorFaceExnerV67):
    """v67 face/Exner owner with authenticated nonnegative live stress."""

    def _slope_token(
        self, *, topology, downhill, stress, depth, d50, mean_diameter, hiding,
    ) -> str:
        assert self.slope_composer is not None
        return _fingerprint({
            "scope_id": SCOPE_ID,
            "extension_fingerprint": self.slope_extension_fingerprint,
            "slope_execution_fingerprint":
                self.slope_composer.identity["fingerprint"],
            "topology_sha256": topology.topology_sha256,
            "stress_semantics": STRESS_SEMANTICS,
            "resident_arrays_sha256": self._resident_digest(
                downhill, stress, depth, d50, mean_diameter, hiding,
                label="direct_vector_live_taurat_slope_fields_v73",
            ),
        })

    def resolve_live_taurat_slope_fields_v73(
        self, *, topology, cell_taurat, downhill, depth, d50,
        mean_diameter, hiding,
    ) -> ResidentDirectVectorSlopeFieldsV69:
        cp = self.cp
        if self.slope_composer is None:
            raise DirectVectorLiveTauratV73Error(
                "Q3/v73 requires the resident ISlope=2 composer"
            )
        self._validate_topology_receipt(topology)
        cell = self._array(
            "cell producer taurat", cell_taurat, 2,
            (self.nclass, topology.ncell), copy=True,
        )
        down = self._array(
            "slope downhill", downhill, 2, (2, topology.nface), copy=True,
        )
        water_depth = self._array(
            "slope depth", depth, 2,
            (self.nclass, topology.nface), copy=True,
        )
        diameter = self._array(
            "slope d50", d50, 1, (self.nclass,), copy=True,
        )
        mean = self._array(
            "slope mean diameter", mean_diameter, 2,
            (self.nclass, topology.nface), copy=True,
        )
        hide = self._array(
            "slope hiding", hiding, 2,
            (self.nclass, topology.ncell), copy=True,
        )
        if (not self._all(cp.isfinite(cell) & (cell >= 0.0)) or
                not self._all(cp.isfinite(down)) or
                not self._all(cp.isfinite(water_depth) &
                              (water_depth > 0.0)) or
                not self._all(cp.isfinite(diameter) & (diameter > 0.0)) or
                not self._all(cp.isfinite(mean) & (mean > 0.0)) or
                not self._all(cp.isfinite(hide) & (hide >= 0.0))):
            raise DirectVectorLiveTauratV73Error(
                "Q3/v73 slope or producer-taurat fields are outside domain"
            )
        safe_owner = cp.clip(topology.owner, 0, topology.ncell-1)
        safe_other = cp.clip(topology.neighbour, 0, topology.ncell-1)
        stress = cp.ascontiguousarray(cell[:, safe_owner])
        internal = topology.role == FACE_INTERNAL
        stress = cp.ascontiguousarray(cp.where(
            internal[None, :],
            0.5*(cell[:, safe_owner] + cell[:, safe_other]),
            stress,
        ))
        stress = cp.ascontiguousarray(cp.where(
            (topology.role == FACE_CLOSED)[None, :], 0.0, stress,
        ))
        token = self._slope_token(
            topology=topology, downhill=down, stress=stress,
            depth=water_depth, d50=diameter, mean_diameter=mean,
            hiding=hide,
        )
        return ResidentDirectVectorSlopeFieldsV69(
            self.slope_extension_fingerprint,
            str(self.slope_composer.identity["fingerprint"]),
            topology.topology_sha256, STRESS_SEMANTICS,
            down, stress, water_depth, diameter, mean, hide, token,
            self.device_id,
        )

    def _validate_slope_fields_v69(self, slope, topology) -> None:
        cp = self.cp
        if (not isinstance(slope, ResidentDirectVectorSlopeFieldsV69) or
                self.slope_composer is None or
                slope.extension_fingerprint !=
                    self.slope_extension_fingerprint or
                slope.slope_execution_fingerprint !=
                    self.slope_composer.identity["fingerprint"] or
                slope.topology_sha256 != topology.topology_sha256 or
                slope.stress_semantics != STRESS_SEMANTICS or
                slope.device_id != self.device_id):
            raise DirectVectorLiveTauratV73Error(
                "Q3/v73 live-taurat slope receipt differs"
            )
        arrays = (
            ("downhill", slope.downhill, 2, (2, topology.nface)),
            ("stress", slope.stress, 2, (self.nclass, topology.nface)),
            ("depth", slope.depth, 2, (self.nclass, topology.nface)),
            ("d50", slope.d50, 1, (self.nclass,)),
            ("mean diameter", slope.mean_diameter, 2,
             (self.nclass, topology.nface)),
            ("hiding", slope.hiding, 2,
             (self.nclass, topology.ncell)),
        )
        for name, value, ndim, shape in arrays:
            self._array(f"slope {name}", value, ndim, shape)
        if (not self._all(cp.isfinite(slope.downhill)) or
                not self._all(cp.isfinite(slope.stress) &
                              (slope.stress >= 0.0)) or
                not self._all(cp.isfinite(slope.depth) &
                              (slope.depth > 0.0)) or
                not self._all(cp.isfinite(slope.d50) & (slope.d50 > 0.0)) or
                not self._all(cp.isfinite(slope.mean_diameter) &
                              (slope.mean_diameter > 0.0)) or
                not self._all(cp.isfinite(slope.hiding) &
                              (slope.hiding >= 0.0))):
            raise DirectVectorLiveTauratV73Error(
                "Q3/v73 live-taurat slope arrays are outside domain"
            )
        expected = self._slope_token(
            topology=topology, downhill=slope.downhill,
            stress=slope.stress, depth=slope.depth, d50=slope.d50,
            mean_diameter=slope.mean_diameter, hiding=slope.hiding,
        )
        if expected != slope.slope_sha256:
            raise DirectVectorLiveTauratV73Error(
                "Q3/v73 live-taurat slope arrays are stale or mutated"
            )


class ResidentDirectVectorLiveTauratV73(ResidentDirectVectorFamilyLiveV67):
    """Six-method CUDA family retaining the live TraFrm=-1 stress channel."""

    def __init__(
        self, *, live_taurat_execution: Mapping[str, object],
        catalogue_entry_ids, maximum_waterdepth_execution,
        slope_composer: ResidentFlowBedSlopeFaceComposer,
        cell_support_offset=None, cell_support_ids=None,
        trace_support_offset=None, trace_support_ids=None,
    ):
        self.live_execution = load_direct_vector_live_taurat_execution_v73(
            live_taurat_execution,
        )
        parent = self.live_execution["parent_execution"]
        super().__init__(
            execution=parent, catalogue_entry_ids=catalogue_entry_ids,
            maximum_waterdepth_execution=maximum_waterdepth_execution,
            cell_support_offset=cell_support_offset,
            cell_support_ids=cell_support_ids,
            trace_support_offset=trace_support_offset,
            trace_support_ids=trace_support_ids,
        )
        if (self.live_execution["backend"] !=
                "implicit_fwave_midpoint_gpu" or
                self.live_execution["integrator"] !=
                "implicit_fwave_midpoint" or
                self.live_execution["parent_execution"] != self.execution or
                self.live_execution["automatic_backend_fallback"] is not False):
            raise DirectVectorLiveTauratV73Error(
                "Q3/v73 requires the selected implicit_fwave_midpoint GPU "
                "parent without fallback"
            )
        if (not isinstance(slope_composer,
                           ResidentFlowBedSlopeFaceComposer) or
                slope_composer.device_id != self.device_id or
                slope_composer.identity !=
                    self.live_execution["bed_slope_execution"]):
            raise DirectVectorLiveTauratV73Error(
                "Q3/v73 resident slope composer identity differs"
            )
        self.live_source_receipt = verify_direct_vector_live_taurat_sources_v73(
            self.live_execution, repository_root=self.root,
        )
        self.face_consumer = ResidentDirectVectorLiveTauratFaceExnerV73(
            execution_fingerprint=self.execution["fingerprint"],
            bulk_density_kg_m3=self.bulk_density,
            accepted_step_limit=int(self.live_execution["accepted_step_limit"]),
            slope_composer=slope_composer,
            slope_extension_fingerprint=self.live_execution["fingerprint"],
        )
        self._verify_consumer_receipt()
        self.runtime_receipt = self._runtime_receipt_v73()

    def _runtime_receipt_v73(self):
        path = Path(__file__).resolve()
        body = {
            "scope_id": SCOPE_ID,
            "execution_fingerprint": self.live_execution["fingerprint"],
            "parent_execution_fingerprint": self.execution["fingerprint"],
            "backend": "implicit_fwave_midpoint_gpu",
            "integrator": "implicit_fwave_midpoint",
            "automatic_backend_fallback": False,
            "device_id": self.device_id,
            "producer_output_channel": "bedbc1993_taubcw_over_taucr1",
            "unrelated_shear_diagnostic_substitution_allowed": False,
            "maximum_waterdepth_execution_fingerprint":
                self.maximum_waterdepth_execution["fingerprint"],
            "maximum_waterdepth": self.maximum_waterdepth.enabled,
            "source_receipt": self.live_source_receipt,
            "high_level_source": {
                "path": str(path.relative_to(self.root)).replace("\\", "/"),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            },
        }
        return {**body, "fingerprint": _fingerprint(body)}

    def _taurat_group_token(self, parent, taurat, semantics: str) -> str:
        return _fingerprint({
            "scope_id": SCOPE_ID,
            "execution_fingerprint": self.live_execution["fingerprint"],
            "parent_group_sha256": parent.group_sha256,
            "method_id": parent.producer_profile_method_id,
            "fraction_indices": list(parent.fraction_indices),
            "stress_semantics": semantics,
            "resident_arrays_sha256": self._digest(
                taurat, label="direct_vector_producer_taurat_v73",
            ),
        })

    @staticmethod
    def _base_values(parent: ResidentDirectVectorProducerGroupV67) -> dict:
        return {
            field.name: getattr(parent, field.name)
            for field in fields(ResidentDirectVectorProducerGroupV67)
        }

    def _extend_group(self, parent, taurat, semantics):
        token = self._taurat_group_token(parent, taurat, semantics)
        return ResidentDirectVectorLiveTauratGroupV73(
            **self._base_values(parent), producer_taurat=taurat,
            stress_semantics=semantics, taurat_receipt_sha256=token,
        )

    def _produce_group(self, group, fields_value, *, point_role, owner=None,
                       role=None):
        cp = self.cp
        if group.method_id != METHOD_VAN_RIJN_1993_CURRENT:
            parent = super()._produce_group(
                group, fields_value, point_role=point_role,
                owner=owner, role=role,
            )
            taurat = cp.zeros(
                (len(group.indices), int(fields_value["npoint"])),
                dtype=cp.float64,
            )
            return self._extend_group(parent, taurat, TAURAT_POSITIVE_ZERO)

        npoint = int(fields_value["npoint"])
        nclass = len(group.indices)
        count = nclass*npoint
        active_mask = fields_value["active_mask"]
        method_fields = self.maximum_waterdepth.producer_fields(
            group.method_id, fields_value["hydraulics"], fields_value,
        )
        producer_fields = self._vr1993_fields(group, method_fields)
        adapted = group.runtime.evaluate_for_face_lifecycle(
            producer_fields=producer_fields,
            bed_factor=cp.full(nclass, self.bed_factor, dtype=cp.float64),
            suspended_factor=cp.full(
                nclass, self.suspended_factor, dtype=cp.float64,
            ),
            speed_epsilon=cp.full(
                (nclass, npoint), self.speed_epsilon, dtype=cp.float64,
            ),
            reject_invalid=False,
        )
        adapted.require_deferred_surface_fraction()
        raw = cp.ascontiguousarray(adapted.raw_values.reshape(5, count))
        composed = cp.ascontiguousarray(
            adapted.composed_values.reshape(3, count)
        )
        producer_status = cp.ascontiguousarray(
            adapted.producer_status.reshape(count)
        )
        route_status = cp.ascontiguousarray(adapted.route_status.reshape(count))
        active_points = cp.ascontiguousarray(
            self._broadcast_point(
                active_mask.astype(cp.float64), nclass,
            ).reshape(count) != 0.0
        )
        raw = cp.ascontiguousarray(cp.where(active_points[None, :], raw, 0.0))
        composed = cp.ascontiguousarray(cp.where(
            active_points[None, :], composed, 0.0,
        ))
        producer_status = cp.ascontiguousarray(cp.where(
            active_points, producer_status, 0,
        ).astype(cp.int32, copy=False))
        route_status = cp.ascontiguousarray(cp.where(
            active_points, route_status, 0,
        ).astype(cp.int32, copy=False))
        taurat = cp.ascontiguousarray(cp.where(
            active_points.reshape(nclass, npoint),
            adapted.producer_auxiliary[2].reshape(nclass, npoint), 0.0,
        ))
        producer_ids, parameter_ids = self._group_identity_rows(group)
        parent = ResidentDirectVectorProducerGroupV67(
            self.execution["fingerprint"],
            self.maximum_waterdepth_execution["fingerprint"],
            group.catalogue_entry_id, group.method_id, "current_only",
            group.route, group.indices, producer_ids, parameter_ids,
            raw, composed, producer_status, route_status,
            0, 0, 1,
            super()._group_token(
                group, raw, composed, producer_status, route_status,
            ),
            self.device_id,
        )
        return self._extend_group(parent, taurat, TAURAT_TRAM_LIVE)

    def _wrap_channels(self, parent):
        cp = self.cp
        taurat = cp.zeros(
            (self.nclass, parent.bed_mass_vector.shape[2]), dtype=cp.float64,
        )
        semantics = [""]*self.nclass
        receipts = []
        for group in parent.producer_groups:
            if not isinstance(group, ResidentDirectVectorLiveTauratGroupV73):
                raise DirectVectorLiveTauratV73Error(
                    "Q3/v73 producer group lost its taurat receipt"
                )
            for local, global_index in enumerate(group.fraction_indices):
                taurat[global_index] = group.producer_taurat[local]
                semantics[global_index] = group.stress_semantics
            receipts.append(group.taurat_receipt_sha256)
        taurat = cp.ascontiguousarray(taurat)
        token = _fingerprint({
            "scope_id": SCOPE_ID,
            "execution_fingerprint": self.live_execution["fingerprint"],
            "parent_channel_sha256": parent.channel_sha256,
            "point_role": parent.point_role,
            "stress_semantics": semantics,
            "group_taurat_receipts": receipts,
            "resident_arrays_sha256": self._digest(
                taurat, label=f"direct_vector_{parent.point_role}_taurat_v73",
            ),
        })
        return ResidentDirectVectorLiveTauratChannelsV73(
            parent, taurat, tuple(semantics), token, self.device_id,
        )

    def cells(self, **values):
        return self._wrap_channels(super().cells(**values))

    def traces(self, **values):
        return self._wrap_channels(super().traces(**values))

    def _validate_live_channels(self, channels, *, point_role):
        if (not isinstance(channels,
                           ResidentDirectVectorLiveTauratChannelsV73) or
                channels.device_id != self.device_id):
            raise DirectVectorLiveTauratV73Error(
                f"Q3/v73 {point_role} taurat receipt differs"
            )
        super()._validate_channels(channels.parent, point_role=point_role)
        cp = self.cp
        self._array(
            f"{point_role} producer taurat", channels.producer_taurat, 2,
            (self.nclass, channels.parent.bed_mass_vector.shape[2]),
        )
        if not self._flag(cp.all(
                cp.isfinite(channels.producer_taurat) &
                (channels.producer_taurat >= 0.0))):
            raise DirectVectorLiveTauratV73Error(
                f"Q3/v73 {point_role} producer taurat is outside domain"
            )
        receipts = [
            group.taurat_receipt_sha256
            for group in channels.parent.producer_groups
        ]
        expected = _fingerprint({
            "scope_id": SCOPE_ID,
            "execution_fingerprint": self.live_execution["fingerprint"],
            "parent_channel_sha256": channels.parent.channel_sha256,
            "point_role": point_role,
            "stress_semantics": list(channels.stress_semantics),
            "group_taurat_receipts": receipts,
            "resident_arrays_sha256": self._digest(
                channels.producer_taurat,
                label=f"direct_vector_{point_role}_taurat_v73",
            ),
        })
        if expected != channels.taurat_receipt_sha256:
            raise DirectVectorLiveTauratV73Error(
                f"Q3/v73 {point_role} taurat arrays are stale or mutated"
            )

    def resolve_slope_fields(
        self, *, topology, cells, traces, downhill, depth, mean_diameter,
        hiding,
    ):
        self._validate_topology(topology)
        self._validate_live_channels(cells, point_role="cells")
        self._validate_live_channels(traces, point_role="traces")
        return self.face_consumer.resolve_live_taurat_slope_fields_v73(
            topology=topology.face_topology,
            cell_taurat=cells.producer_taurat, downhill=downhill,
            depth=depth, d50=self.d50, mean_diameter=mean_diameter,
            hiding=hiding,
        )

    def _trial_token_v73(self, state, topology, cells, traces, slope,
                         face_trial):
        return _fingerprint({
            "scope_id": SCOPE_ID,
            "execution_fingerprint": self.live_execution["fingerprint"],
            "parent_execution_fingerprint": self.execution["fingerprint"],
            "parent_state_sha256": state.state_sha256,
            "candidate_state_sha256": face_trial.candidate.state_sha256,
            "topology_sha256": topology.topology_sha256,
            "cell_taurat_receipt_sha256": cells.taurat_receipt_sha256,
            "trace_taurat_receipt_sha256": traces.taurat_receipt_sha256,
            "slope_sha256": slope.slope_sha256,
            "face_trial_sha256": face_trial.trial_sha256,
            "producer_taurat_application_count": 1,
            "bed_slope_application_count": 1,
            "active_fraction_application_count": 1,
            "inventory_fill_application_count": 1,
        })

    def begin_trial(
        self, *, state, cells, traces, topology, slope, dt_s,
        morphological_factor=1.0, left_bed=None, right_bed=None,
        cancelled=lambda: False,
    ):
        if (self.execution["accepted_state_admitted"] is not True or
                any(value != TRA_BEDLOAD for value in self.transport_class)):
            raise DirectVectorLiveTauratV73Error(
                "Q3/v73 requires the accepted bedload-only route"
            )
        self._validate_live_channels(cells, point_role="cells")
        self._validate_live_channels(traces, point_role="traces")
        self._validate_topology(topology)
        self.face_consumer._validate_slope_fields_v69(
            slope, topology.face_topology,
        )
        face_trial = self.face_consumer.begin_trial(
            state=state, cells=cells.parent.face_channels,
            traces=traces.parent.face_channels,
            topology=topology.face_topology, slope=slope, dt_s=dt_s,
            morphological_factor=morphological_factor,
            left_bed=left_bed, right_bed=right_bed, cancelled=cancelled,
        )
        token = self._trial_token_v73(
            state, topology, cells, traces, slope, face_trial,
        )
        return ResidentDirectVectorLiveTauratTrialV73(
            self.live_execution["fingerprint"], state.state_sha256,
            topology.topology_sha256, cells.taurat_receipt_sha256,
            traces.taurat_receipt_sha256, slope.slope_sha256, face_trial,
            token, self.device_id,
        )

    def _validate_live_trial(self, state, trial):
        if (not isinstance(trial, ResidentDirectVectorLiveTauratTrialV73) or
                trial.execution_fingerprint !=
                    self.live_execution["fingerprint"] or
                trial.parent_state_sha256 != state.state_sha256 or
                trial.device_id != self.device_id or
                trial.face_trial.slope_receipt_sha256 != trial.slope_sha256):
            raise DirectVectorLiveTauratV73Error(
                "Q3/v73 trial parent or source receipt differs"
            )
        self.face_consumer._validate_trial(state, trial.face_trial)
        expected = _fingerprint({
            "scope_id": SCOPE_ID,
            "execution_fingerprint": self.live_execution["fingerprint"],
            "parent_execution_fingerprint": self.execution["fingerprint"],
            "parent_state_sha256": state.state_sha256,
            "candidate_state_sha256": trial.face_trial.candidate.state_sha256,
            "topology_sha256": trial.topology_sha256,
            "cell_taurat_receipt_sha256":
                trial.cell_taurat_receipt_sha256,
            "trace_taurat_receipt_sha256":
                trial.trace_taurat_receipt_sha256,
            "slope_sha256": trial.slope_sha256,
            "face_trial_sha256": trial.face_trial.trial_sha256,
            "producer_taurat_application_count": 1,
            "bed_slope_application_count": 1,
            "active_fraction_application_count": 1,
            "inventory_fill_application_count": 1,
        })
        if expected != trial.trial_sha256:
            raise DirectVectorLiveTauratV73Error(
                "Q3/v73 trial receipt is stale or mutated"
            )

    def commit_trial(self, state, trial):
        self._validate_live_trial(state, trial)
        return self.face_consumer.commit_trial(state, trial.face_trial)

    def rollback_trial(self, state, trial):
        self._validate_live_trial(state, trial)
        return self.face_consumer.rollback_trial(state, trial.face_trial)

    def cancel_trial(self, state, trial):
        self._validate_live_trial(state, trial)
        return self.face_consumer.cancel_trial(state, trial.face_trial)

    def state_checkpoint_document(self, state):
        nested = super().state_checkpoint_document(state)
        body = {
            "scope_id": CHECKPOINT_SCOPE_ID,
            "execution_fingerprint": self.live_execution["fingerprint"],
            "parent_execution_fingerprint": self.execution["fingerprint"],
            "family_checkpoint": nested,
        }
        return {**body, "fingerprint": _fingerprint(body)}

    def load_state_checkpoint(self, value):
        if not isinstance(value, Mapping):
            raise DirectVectorLiveTauratV73Error(
                "Q3/v73 checkpoint must be an object"
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
                    self.live_execution["fingerprint"]
                and value["parent_execution_fingerprint"] ==
                    self.execution["fingerprint"]
                and value["fingerprint"] == _fingerprint(body)
            )
        except (KeyError, TypeError, ValueError):
            valid = False
        if not valid:
            raise DirectVectorLiveTauratV73Error(
                "Q3/v73 checkpoint identity differs"
            )
        state = super().load_state_checkpoint(value["family_checkpoint"])
        self.face_consumer._validate_state(state)
        return state


__all__ = [
    "ResidentDirectVectorLiveTauratChannelsV73",
    "ResidentDirectVectorLiveTauratFaceExnerV73",
    "ResidentDirectVectorLiveTauratGroupV73",
    "ResidentDirectVectorLiveTauratTrialV73",
    "ResidentDirectVectorLiveTauratV73", "SCOPE_ID",
]
