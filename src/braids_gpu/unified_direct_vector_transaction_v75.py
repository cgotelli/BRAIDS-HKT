"""Actual-CUDA Q3/v75 direct-vector transaction composer.

This wrapper is intentionally additive.  It reuses the accepted v73 producer
and live-``taurat`` face owner, the optional v72 derived solve, the v71
wet/dry/AMR state owner, and the v70 supported boundary selector.  It launches
exactly one accepted v73 face/inventory/Exner transaction and never selects or
falls back to a CPU backend.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Mapping

import numpy as np

from braids_core.sediment.bedload_boundary_live_v70 import (
    SOURCE_EQUILIBRIUM_TRACE,
    SOURCE_NONE,
    SOURCE_OWNER_FREE_EXPORT,
    SOURCE_PRESCRIBED_EXTERNAL,
)
from braids_core.sediment.direct_vector_wetdry_amr_live_v71 import (
    ConservativeTransferPlanV71,
    FACE_CLOSED,
    FACE_OWNER_OUTLET,
    FACE_TRACE_INLET,
)
from braids_core.sediment.unified_direct_vector_transaction_v75 import (
    CHECKPOINT_SCOPE_ID,
    MAX_ACCEPTED_STEPS,
    SCOPE_ID,
    UnifiedDirectVectorV75Error,
    _canonical,
    _fingerprint,
    load_unified_direct_vector_execution_v75,
)
from .bedload_adaptation_live_v72 import (
    ResidentAdaptedCapacityV72,
    ResidentBedloadAdaptationV72,
)
from .bedload_boundary_live_v70 import (
    ResidentBedloadBoundaryCandidateV70,
    ResidentBedloadBoundaryTopologyV70,
    ResidentBedloadBoundaryV70,
)
from .direct_vector_face_exner_v67 import (
    ResidentDirectVectorChannelsV67,
    ResidentDirectVectorSlopeFieldsV69,
    ResidentDirectVectorTrialV67,
)
from .direct_vector_live_taurat_v73 import (
    ResidentDirectVectorLiveTauratChannelsV73,
    ResidentDirectVectorLiveTauratV73,
)
from .direct_vector_wetdry_amr_live_v71 import (
    ResidentDirectVectorWetDryAMRV71,
    ResidentWetDryAMRStateV71,
    ResidentWetDryRegridTrialV71,
    ResidentWetDryTopologyBindingV71,
)


@dataclass(frozen=True)
class ResidentUnifiedBindingV75:
    topology_id: str
    wetdry_binding: ResidentWetDryTopologyBindingV71
    boundary_runtime: ResidentBedloadBoundaryV70
    boundary_topology: ResidentBedloadBoundaryTopologyV70
    family_boundary_faces: object
    adaptation_runtime: ResidentBedloadAdaptationV72 | None
    binding_sha256: str
    device_id: int


@dataclass(frozen=True)
class ResidentUnifiedTrialV75:
    parent_state_sha256: str
    topology_id: str
    cells: ResidentDirectVectorLiveTauratChannelsV73
    traces: ResidentDirectVectorLiveTauratChannelsV73
    adaptation: ResidentAdaptedCapacityV72 | None
    adapted_cell_channels: ResidentDirectVectorChannelsV67
    slope: ResidentDirectVectorSlopeFieldsV69
    boundary_candidate: ResidentBedloadBoundaryCandidateV70
    face_trial: ResidentDirectVectorTrialV67
    candidate: ResidentWetDryAMRStateV71
    named_boundary_transfer_kg: object
    named_boundary_absolute_kg: object
    producer_taurat_application_count: int
    adaptation_application_count: int
    wetdry_owner_application_count: int
    face_reconstruction_application_count: int
    bed_slope_application_count: int
    boundary_owner_application_count: int
    active_fraction_application_count: int
    inventory_fill_application_count: int
    exner_application_count: int
    trial_sha256: str
    device_id: int


@dataclass(frozen=True)
class ResidentUnifiedRegridTrialV75:
    parent_state_sha256: str
    old_topology_id: str
    new_topology_id: str
    transfer_plan_id: str
    wetdry_trial: ResidentWetDryRegridTrialV71
    trial_sha256: str
    device_id: int


class ResidentUnifiedDirectVectorV75:
    """One selected-device, bounded unified v75 transaction owner."""

    backend = "implicit_fwave_midpoint_gpu"
    integrator = "implicit_fwave_midpoint"

    def __init__(
        self, *, execution: Mapping[str, object],
        wetdry_runtime: ResidentDirectVectorWetDryAMRV71,
    ):
        import cupy as cp

        self.cp = cp
        self.device_id = int(cp.cuda.Device().id)
        self.execution = load_unified_direct_vector_execution_v75(execution)
        if (self.execution["backend"] != self.backend or
                self.execution["integrator"] != self.integrator or
                self.execution["automatic_backend_fallback"] is not False):
            raise UnifiedDirectVectorV75Error(
                "Q3/v75 CUDA requires explicit implicit_fwave_midpoint_gpu"
            )
        if (not isinstance(wetdry_runtime, ResidentDirectVectorWetDryAMRV71) or
                wetdry_runtime.device_id != self.device_id or
                wetdry_runtime.execution != self.execution["wetdry_execution"]):
            raise UnifiedDirectVectorV75Error(
                "Q3/v75 CUDA v71 runtime/execution differs"
            )
        rows = {row["topology_id"]: row
                for row in self.execution["topologies"]}
        if set(rows) != set(wetdry_runtime.runtimes):
            raise UnifiedDirectVectorV75Error(
                "Q3/v75 CUDA v71 runtime topology inventory differs"
            )
        for topology_id, runtime in wetdry_runtime.runtimes.items():
            if (not isinstance(runtime, ResidentDirectVectorLiveTauratV73) or
                    runtime.live_execution !=
                        rows[topology_id]["live_taurat_execution"] or
                    runtime.device_id != self.device_id):
                raise UnifiedDirectVectorV75Error(
                    "Q3/v75 CUDA topology runtime is not its v73 owner"
                )
        self.wetdry = wetdry_runtime
        self.rows = rows
        self.nclass = len(self.execution["fraction_ids"])
        self._open_parent: str | None = None

    def _device_digest(self, *values, label: str) -> str:
        cp = self.cp
        digest = hashlib.sha256()
        digest.update(label.encode("ascii"))
        for raw in values:
            if not isinstance(raw, cp.ndarray) or int(raw.device.id) != self.device_id:
                raise UnifiedDirectVectorV75Error(
                    "Q3/v75 CUDA receipt contains a nonresident array"
                )
            host = np.ascontiguousarray(cp.asnumpy(raw))
            digest.update(str(host.dtype).encode("ascii"))
            digest.update(_canonical(list(host.shape)))
            digest.update(host.tobytes(order="C"))
        return digest.hexdigest()

    def prepare_binding(
        self, *, wetdry_binding: ResidentWetDryTopologyBindingV71,
        boundary_runtime: ResidentBedloadBoundaryV70,
        boundary_topology: ResidentBedloadBoundaryTopologyV70,
        family_boundary_faces,
        adaptation_runtime: ResidentBedloadAdaptationV72 | None = None,
    ) -> ResidentUnifiedBindingV75:
        cp = self.cp
        self.wetdry._validate_binding(wetdry_binding)
        topology_id = wetdry_binding.topology.topology_id
        row = self.rows[topology_id]
        if (not isinstance(boundary_runtime, ResidentBedloadBoundaryV70) or
                boundary_runtime.device_id != self.device_id or
                boundary_runtime.execution != row["boundary_execution"] or
                not isinstance(boundary_topology,
                               ResidentBedloadBoundaryTopologyV70)):
            raise UnifiedDirectVectorV75Error(
                "Q3/v75 CUDA v70 runtime/topology differs"
            )
        boundary_runtime._validate_topology(boundary_topology)
        mapping = cp.ascontiguousarray(
            cp.asarray(family_boundary_faces, dtype=cp.int32),
        )
        if (mapping.shape != (boundary_topology.nface,) or
                not bool(cp.all((mapping >= 0) &
                                (mapping < wetdry_binding.topology.nface)).item()) or
                int(cp.unique(mapping).size) != int(mapping.size)):
            raise UnifiedDirectVectorV75Error(
                "Q3/v75 CUDA family boundary-face map differs"
            )
        face = wetdry_binding.family_topology.face_topology
        if (not bool(cp.array_equal(
                    face.owner[mapping], boundary_topology.owner).item()) or
                not bool(cp.array_equal(
                    face.normal[:, mapping], boundary_topology.normal).item()) or
                not bool(cp.array_equal(
                    face.face_length[mapping],
                    boundary_topology.face_length_m).item()) or
                not bool(cp.array_equal(
                    face.cell_area, boundary_topology.cell_area_m2).item()) or
                not bool(cp.all(face.neighbour[mapping] == -1).item())):
            raise UnifiedDirectVectorV75Error(
                "Q3/v75 CUDA v70/family boundary topology differs"
            )
        adaptation_execution = row["adaptation_execution"]
        if adaptation_execution is None:
            if adaptation_runtime is not None:
                raise UnifiedDirectVectorV75Error(
                    "Q3/v75 CUDA adaptation runtime supplied while disabled"
                )
        else:
            if (not isinstance(adaptation_runtime,
                               ResidentBedloadAdaptationV72) or
                    adaptation_runtime.device_id != self.device_id or
                    adaptation_runtime.execution != adaptation_execution):
                raise UnifiedDirectVectorV75Error(
                    "Q3/v75 CUDA enabled adaptation runtime differs"
                )
            adaptation_runtime._match_direct_topology(
                wetdry_binding.family_topology,
            )
        token = _fingerprint({
            "scope_id": SCOPE_ID,
            "execution_fingerprint": self.execution["fingerprint"],
            "topology_id": topology_id,
            "wetdry_binding_sha256": wetdry_binding.binding_sha256,
            "boundary_topology_sha256": boundary_topology.topology_sha256,
            "adaptation_topology_sha256": (
                None if adaptation_runtime is None else
                adaptation_runtime.topology.topology_sha256
            ),
            "family_boundary_faces": cp.asnumpy(mapping).tolist(),
            "device_id": self.device_id,
        })
        return ResidentUnifiedBindingV75(
            topology_id, wetdry_binding, boundary_runtime, boundary_topology,
            mapping, adaptation_runtime, token, self.device_id,
        )

    def _validate_binding(self, binding: ResidentUnifiedBindingV75) -> None:
        cp = self.cp
        if (not isinstance(binding, ResidentUnifiedBindingV75) or
                binding.device_id != self.device_id or
                binding.topology_id not in self.rows):
            raise UnifiedDirectVectorV75Error(
                "Q3/v75 CUDA binding differs"
            )
        self.wetdry._validate_binding(binding.wetdry_binding)
        binding.boundary_runtime._validate_topology(binding.boundary_topology)
        token = _fingerprint({
            "scope_id": SCOPE_ID,
            "execution_fingerprint": self.execution["fingerprint"],
            "topology_id": binding.topology_id,
            "wetdry_binding_sha256": binding.wetdry_binding.binding_sha256,
            "boundary_topology_sha256":
                binding.boundary_topology.topology_sha256,
            "adaptation_topology_sha256": (
                None if binding.adaptation_runtime is None else
                binding.adaptation_runtime.topology.topology_sha256
            ),
            "family_boundary_faces":
                cp.asnumpy(binding.family_boundary_faces).tolist(),
            "device_id": self.device_id,
        })
        if token != binding.binding_sha256:
            raise UnifiedDirectVectorV75Error(
                "Q3/v75 CUDA binding arrays are stale"
            )

    def initial_state(self, binding: ResidentUnifiedBindingV75, *,
                      bed_mass_kg_m2) -> ResidentWetDryAMRStateV71:
        self._validate_binding(binding)
        direct = binding.wetdry_binding.runtime.resolve_state(
            bed_mass_kg_m2=bed_mass_kg_m2,
        )
        return self.wetdry.initial_state(binding.wetdry_binding, direct)

    def _boundary_candidate(self, binding, *, hydraulic_outward_flux,
                            cells, traces):
        cp = self.cp
        mapping = binding.family_boundary_faces
        candidate = binding.boundary_runtime.candidate(
            binding.boundary_topology,
            hydraulic_outward_flux=hydraulic_outward_flux,
            owner_bed_mass_vector=cells,
            trace_bed_mass_vector=cp.ascontiguousarray(traces[:, :, mapping]),
        )
        source = cp.asnumpy(candidate.source)
        roles = cp.asnumpy(
            binding.wetdry_binding.family_topology.face_topology.role[mapping]
        )
        for local, role in enumerate(roles):
            sources = set(source[:, local].tolist())
            if SOURCE_PRESCRIBED_EXTERNAL in sources:
                raise UnifiedDirectVectorV75Error(
                    "Q3/v75 prescribed external feed reached CUDA composition"
                )
            expected = None
            if sources == {SOURCE_EQUILIBRIUM_TRACE}:
                expected = FACE_TRACE_INLET
            elif sources == {SOURCE_OWNER_FREE_EXPORT}:
                expected = FACE_OWNER_OUTLET
            elif sources == {SOURCE_NONE}:
                expected = FACE_CLOSED
            if expected is None or int(role) != expected:
                raise UnifiedDirectVectorV75Error(
                    "Q3/v75 CUDA face role differs from v70 boundary owner"
                )
        return candidate

    def begin_trial(
        self, *, state: ResidentWetDryAMRStateV71,
        binding: ResidentUnifiedBindingV75, q, trace,
        roughness_parameter, kinematic_viscosity,
        settling_velocity_cells, settling_velocity_traces,
        downhill, depth, mean_diameter, hiding,
        hydraulic_outward_flux, dry_tolerance: float, dx_m: float,
        dy_m: float, dt_s: float, morphological_factor: float = 1.0,
        cancelled=lambda: False,
    ) -> ResidentUnifiedTrialV75:
        if not callable(cancelled):
            raise TypeError("cancelled must be callable")
        if cancelled():
            raise InterruptedError("Q3/v75 CUDA transaction cancelled")
        if self._open_parent is not None:
            raise RuntimeError("Q3/v75 CUDA already has an open transaction")
        self._validate_binding(binding)
        self.wetdry._validate_state(state, binding.wetdry_binding)
        if state.direct_state.accepted_step >= MAX_ACCEPTED_STEPS:
            raise UnifiedDirectVectorV75Error(
                "Q3/v75 CUDA bounded accepted-step limit reached"
            )
        runtime = binding.wetdry_binding.runtime
        cells, traces = self.wetdry.prepare_transport_inputs(
            binding.wetdry_binding, q=q, trace=trace,
            roughness_parameter=roughness_parameter,
            kinematic_viscosity=kinematic_viscosity,
            settling_velocity_cells=settling_velocity_cells,
            settling_velocity_traces=settling_velocity_traces,
            dry_tolerance=dry_tolerance,
        )
        if (not isinstance(cells, ResidentDirectVectorLiveTauratChannelsV73) or
                not isinstance(traces,
                               ResidentDirectVectorLiveTauratChannelsV73)):
            raise UnifiedDirectVectorV75Error(
                "Q3/v75 CUDA v71 preparation lost v73 taurat receipts"
            )
        adaptation = None
        adapted_vectors = cells.bed_mass_vector
        if binding.adaptation_runtime is not None:
            cp = self.cp
            feed = cp.zeros(
                (self.nclass, binding.adaptation_runtime.nboundary),
                dtype=cp.float64,
            )
            adaptation = binding.adaptation_runtime.relax(
                q=q, equilibrium_capacity=cells.bed_mass_vector,
                boundary_feed_kg_m_s=feed,
                dry_tolerance=dry_tolerance, dx_m=dx_m, dy_m=dy_m,
                cancelled=cancelled,
            )
            binding.adaptation_runtime._validate_result(
                adaptation, cells.bed_mass_vector,
            )
            adapted_vectors = adaptation.adapted_capacity
        adapted_channels = runtime.face_consumer.prepare_channels(
            point_role="cells", bed_mass_vector=adapted_vectors,
            status=cells.status,
            producer_fingerprint=(
                cells.channel_sha256 if adaptation is None else
                adaptation.result_sha256
            ),
            surface_fraction_application_count=0,
        )
        slope = runtime.resolve_slope_fields(
            topology=binding.wetdry_binding.family_topology,
            cells=cells, traces=traces, downhill=downhill, depth=depth,
            mean_diameter=mean_diameter, hiding=hiding,
        )
        boundary = self._boundary_candidate(
            binding, hydraulic_outward_flux=hydraulic_outward_flux,
            cells=adapted_vectors,
            traces=traces.bed_mass_vector,
        )
        face_trial = runtime.face_consumer.begin_trial(
            state=state.direct_state, cells=adapted_channels,
            traces=traces.parent.face_channels,
            topology=binding.wetdry_binding.family_topology.face_topology,
            slope=slope, dt_s=dt_s,
            morphological_factor=morphological_factor,
            cancelled=cancelled,
        )
        cp = self.cp
        topology = binding.wetdry_binding.topology
        dry_face = binding.wetdry_binding.family_topology.face_topology.role == \
            FACE_CLOSED
        if (bool(cp.any(face_trial.faces.flux[:, dry_face] != 0.0).item()) or
                (bool(cp.any(binding.wetdry_binding.wet == 0).item()) and
                 not bool(cp.array_equal(
                     state.direct_state.bed_mass_kg_m2[:,
                         binding.wetdry_binding.wet == 0],
                     face_trial.candidate.bed_mass_kg_m2[:,
                         binding.wetdry_binding.wet == 0],
                 ).item()))):
            raise UnifiedDirectVectorV75Error(
                "Q3/v75 CUDA wet/dry isolation differs"
            )
        candidate = self.wetdry._make_state(
            binding.wetdry_binding, face_trial.candidate,
            mesh_epoch=state.mesh_epoch,
            accepted_face_flux=cp.ascontiguousarray(face_trial.faces.flux.copy()),
            accepted_cell_availability=cp.ascontiguousarray(cp.where(
                binding.wetdry_binding.wet[None, :] == 1,
                face_trial.accepted_face_availability, 0.0,
            )),
        )
        nboundary = len(binding.boundary_runtime.execution["boundaries"])
        named = cp.zeros((self.nclass, nboundary), dtype=cp.float64)
        named_abs = cp.zeros_like(named)
        mapped_flux = face_trial.faces.flux[:, binding.family_boundary_faces]
        amounts = (
            mapped_flux*binding.boundary_topology.face_length_m[None, :]
            * np.float64(face_trial.dt_s)
            * np.float64(face_trial.morphological_factor)
        )
        for local in range(binding.boundary_topology.nface):
            boundary_index = int(
                binding.boundary_topology.boundary_index[local].item()
            )
            named[:, boundary_index] += amounts[:, local]
            named_abs[:, boundary_index] += cp.abs(amounts[:, local])
        if not bool(cp.allclose(
            cp.sum(named, axis=1), face_trial.boundary_transfer_kg,
            rtol=0.0, atol=64.0*np.finfo(np.float64).eps,
        ).item()):
            raise UnifiedDirectVectorV75Error(
                "Q3/v75 CUDA named v70 ledger differs from one Exner ledger"
            )
        counts = {
            "producer_taurat_application_count": 1,
            "adaptation_application_count": int(adaptation is not None),
            "wetdry_owner_application_count": 1,
            "face_reconstruction_application_count": 1,
            "bed_slope_application_count": 1,
            "boundary_owner_application_count": 1,
            "active_fraction_application_count":
                face_trial.active_fraction_application_count,
            "inventory_fill_application_count":
                face_trial.inventory_fill_application_count,
            "exner_application_count": 1,
        }
        body = {
            "scope_id": SCOPE_ID,
            "execution_fingerprint": self.execution["fingerprint"],
            "binding_sha256": binding.binding_sha256,
            "parent_state_sha256": state.state_sha256,
            "candidate_state_sha256": candidate.state_sha256,
            "cell_taurat_receipt_sha256": cells.taurat_receipt_sha256,
            "trace_taurat_receipt_sha256": traces.taurat_receipt_sha256,
            "adaptation_result_sha256": (
                None if adaptation is None else adaptation.result_sha256
            ),
            "adapted_cell_channel_sha256": adapted_channels.channel_sha256,
            "slope_sha256": slope.slope_sha256,
            "boundary_candidate_sha256": boundary.candidate_sha256,
            "face_trial_sha256": face_trial.trial_sha256,
            **counts,
            "resident_arrays_sha256": self._device_digest(
                named, named_abs, label="unified_v75_named_boundary_ledger",
            ),
        }
        trial = ResidentUnifiedTrialV75(
            state.state_sha256, binding.topology_id, cells, traces,
            adaptation, adapted_channels, slope, boundary, face_trial,
            candidate, named, named_abs, *counts.values(),
            _fingerprint(body), self.device_id,
        )
        if cancelled():
            raise InterruptedError(
                "Q3/v75 CUDA transaction cancelled before publication"
            )
        self._open_parent = state.state_sha256
        return trial

    def _validate_trial(self, state, binding, trial):
        self._validate_binding(binding)
        self.wetdry._validate_state(state, binding.wetdry_binding)
        runtime = binding.wetdry_binding.runtime
        if (not isinstance(trial, ResidentUnifiedTrialV75) or
                trial.device_id != self.device_id or
                trial.parent_state_sha256 != state.state_sha256 or
                trial.topology_id != binding.topology_id or
                self._open_parent != state.state_sha256):
            raise UnifiedDirectVectorV75Error(
                "Q3/v75 CUDA trial parent differs"
            )
        runtime._validate_live_channels(trial.cells, point_role="cells")
        runtime._validate_live_channels(trial.traces, point_role="traces")
        if binding.adaptation_runtime is not None:
            binding.adaptation_runtime._validate_result(
                trial.adaptation, trial.cells.bed_mass_vector,
            )
        runtime.face_consumer._validate_slope_fields_v69(
            trial.slope, binding.wetdry_binding.family_topology.face_topology,
        )
        runtime.face_consumer._validate_trial(
            state.direct_state, trial.face_trial,
        )
        self.wetdry._validate_state(trial.candidate, binding.wetdry_binding)
        expected_counts = (1, int(binding.adaptation_runtime is not None),
                           1, 1, 1, 1, 1, 1, 1)
        actual_counts = (
            trial.producer_taurat_application_count,
            trial.adaptation_application_count,
            trial.wetdry_owner_application_count,
            trial.face_reconstruction_application_count,
            trial.bed_slope_application_count,
            trial.boundary_owner_application_count,
            trial.active_fraction_application_count,
            trial.inventory_fill_application_count,
            trial.exner_application_count,
        )
        if actual_counts != expected_counts:
            raise UnifiedDirectVectorV75Error(
                "Q3/v75 CUDA once-only counters differ"
            )
        counts = dict(zip((
            "producer_taurat_application_count",
            "adaptation_application_count", "wetdry_owner_application_count",
            "face_reconstruction_application_count",
            "bed_slope_application_count", "boundary_owner_application_count",
            "active_fraction_application_count",
            "inventory_fill_application_count", "exner_application_count",
        ), actual_counts, strict=True))
        body = {
            "scope_id": SCOPE_ID,
            "execution_fingerprint": self.execution["fingerprint"],
            "binding_sha256": binding.binding_sha256,
            "parent_state_sha256": state.state_sha256,
            "candidate_state_sha256": trial.candidate.state_sha256,
            "cell_taurat_receipt_sha256":
                trial.cells.taurat_receipt_sha256,
            "trace_taurat_receipt_sha256":
                trial.traces.taurat_receipt_sha256,
            "adaptation_result_sha256": (
                None if trial.adaptation is None else
                trial.adaptation.result_sha256
            ),
            "adapted_cell_channel_sha256":
                trial.adapted_cell_channels.channel_sha256,
            "slope_sha256": trial.slope.slope_sha256,
            "boundary_candidate_sha256":
                trial.boundary_candidate.candidate_sha256,
            "face_trial_sha256": trial.face_trial.trial_sha256,
            **counts,
            "resident_arrays_sha256": self._device_digest(
                trial.named_boundary_transfer_kg,
                trial.named_boundary_absolute_kg,
                label="unified_v75_named_boundary_ledger",
            ),
        }
        if trial.trial_sha256 != _fingerprint(body):
            raise UnifiedDirectVectorV75Error(
                "Q3/v75 CUDA trial arrays are stale or mutated"
            )

    def commit(self, state, binding, trial):
        self._validate_trial(state, binding, trial)
        binding.wetdry_binding.runtime.face_consumer.commit_trial(
            state.direct_state, trial.face_trial,
        )
        self._open_parent = None
        return trial.candidate

    def rollback(self, state, binding, trial):
        self._validate_trial(state, binding, trial)
        binding.wetdry_binding.runtime.face_consumer.rollback_trial(
            state.direct_state, trial.face_trial,
        )
        self._open_parent = None
        return state

    def cancel(self, state, binding, trial):
        self._validate_trial(state, binding, trial)
        binding.wetdry_binding.runtime.face_consumer.cancel_trial(
            state.direct_state, trial.face_trial,
        )
        self._open_parent = None
        return state

    def begin_regrid(self, *, state, old_binding, new_binding,
                     plan: ConservativeTransferPlanV71, cancelled=lambda: False):
        if self._open_parent is not None:
            raise RuntimeError("Q3/v75 CUDA already has an open transaction")
        self._validate_binding(old_binding)
        self._validate_binding(new_binding)
        nested = self.wetdry.begin_regrid_trial(
            state, old_binding.wetdry_binding, new_binding.wetdry_binding,
            plan, cancelled=cancelled,
        )
        body = {
            "scope_id": SCOPE_ID, "kind": "regrid",
            "parent_state_sha256": state.state_sha256,
            "old_topology_id": old_binding.topology_id,
            "new_topology_id": new_binding.topology_id,
            "transfer_plan_id": plan.plan_id,
            "wetdry_trial_sha256": nested.trial_sha256,
            "transition_kind": nested.transition_kind,
            "wet_state_change_count": nested.wet_state_change_count,
            "transfer_application_count": nested.transfer_application_count,
            "face_register_cleanup_count": nested.face_register_cleanup_count,
            "availability_cleanup_count": nested.availability_cleanup_count,
            "exner_application_count": nested.exner_application_count,
        }
        trial = ResidentUnifiedRegridTrialV75(
            state.state_sha256, old_binding.topology_id,
            new_binding.topology_id, plan.plan_id, nested,
            _fingerprint(body), self.device_id,
        )
        self._open_parent = state.state_sha256
        return trial

    def _validate_regrid(self, state, old_binding, new_binding, plan, trial):
        if (not isinstance(trial, ResidentUnifiedRegridTrialV75) or
                trial.device_id != self.device_id or
                trial.parent_state_sha256 != state.state_sha256 or
                trial.old_topology_id != old_binding.topology_id or
                trial.new_topology_id != new_binding.topology_id or
                trial.transfer_plan_id != plan.plan_id or
                self._open_parent != state.state_sha256):
            raise UnifiedDirectVectorV75Error(
                "Q3/v75 CUDA regrid parent differs"
            )
        self.wetdry._validate_regrid_trial(
            state, old_binding.wetdry_binding, new_binding.wetdry_binding,
            plan, trial.wetdry_trial,
        )
        nested = trial.wetdry_trial
        body = {
            "scope_id": SCOPE_ID, "kind": "regrid",
            "parent_state_sha256": state.state_sha256,
            "old_topology_id": old_binding.topology_id,
            "new_topology_id": new_binding.topology_id,
            "transfer_plan_id": plan.plan_id,
            "wetdry_trial_sha256": nested.trial_sha256,
            "transition_kind": nested.transition_kind,
            "wet_state_change_count": nested.wet_state_change_count,
            "transfer_application_count": nested.transfer_application_count,
            "face_register_cleanup_count": nested.face_register_cleanup_count,
            "availability_cleanup_count": nested.availability_cleanup_count,
            "exner_application_count": nested.exner_application_count,
        }
        if trial.trial_sha256 != _fingerprint(body):
            raise UnifiedDirectVectorV75Error(
                "Q3/v75 CUDA regrid receipt is stale"
            )

    def commit_regrid(self, state, old_binding, new_binding, plan, trial):
        self._validate_regrid(state, old_binding, new_binding, plan, trial)
        result = self.wetdry.commit_regrid_trial(
            state, old_binding.wetdry_binding, new_binding.wetdry_binding,
            plan, trial.wetdry_trial,
        )
        self._open_parent = None
        return result

    def rollback_regrid(self, state, old_binding, new_binding, plan, trial):
        self._validate_regrid(state, old_binding, new_binding, plan, trial)
        result = self.wetdry.rollback_regrid_trial(
            state, old_binding.wetdry_binding, new_binding.wetdry_binding,
            plan, trial.wetdry_trial,
        )
        self._open_parent = None
        return result

    def cancel_regrid(self, state, old_binding, new_binding, plan, trial):
        self._validate_regrid(state, old_binding, new_binding, plan, trial)
        result = self.wetdry.cancel_regrid_trial(
            state, old_binding.wetdry_binding, new_binding.wetdry_binding,
            plan, trial.wetdry_trial,
        )
        self._open_parent = None
        return result

    def checkpoint(self, state, binding) -> dict[str, object]:
        self._validate_binding(binding)
        nested = self.wetdry.state_checkpoint_document(
            state, binding.wetdry_binding,
        )
        body = {
            "schema": 1, "scope_id": CHECKPOINT_SCOPE_ID,
            "execution_fingerprint": self.execution["fingerprint"],
            "binding_sha256": binding.binding_sha256,
            "topology_id": binding.topology_id,
            "wetdry_checkpoint": nested,
            "adaptation_execution_fingerprint": (
                None if binding.adaptation_runtime is None else
                binding.adaptation_runtime.execution["fingerprint"]
            ),
            "derived_adaptation_arrays_serialized": False,
        }
        return {**body, "fingerprint": _fingerprint(body)}

    def restart(self, value: Mapping[str, object], binding):
        self._validate_binding(binding)
        if not isinstance(value, Mapping):
            raise UnifiedDirectVectorV75Error(
                "Q3/v75 CUDA checkpoint must be an object"
            )
        body = {key: value[key] for key in value if key != "fingerprint"}
        adaptation = (
            None if binding.adaptation_runtime is None else
            binding.adaptation_runtime.execution["fingerprint"]
        )
        if (value.get("fingerprint") != _fingerprint(body) or
                value.get("schema") != 1 or
                value.get("scope_id") != CHECKPOINT_SCOPE_ID or
                value.get("execution_fingerprint") !=
                    self.execution["fingerprint"] or
                value.get("binding_sha256") != binding.binding_sha256 or
                value.get("topology_id") != binding.topology_id or
                value.get("adaptation_execution_fingerprint") != adaptation or
                value.get("derived_adaptation_arrays_serialized") is not False):
            raise UnifiedDirectVectorV75Error(
                "Q3/v75 CUDA checkpoint identity differs"
            )
        return self.wetdry.load_state_checkpoint(
            value["wetdry_checkpoint"], binding.wetdry_binding,
        )


__all__ = [
    "ResidentUnifiedBindingV75", "ResidentUnifiedDirectVectorV75",
    "ResidentUnifiedRegridTrialV75", "ResidentUnifiedTrialV75",
]
