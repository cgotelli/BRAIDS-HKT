"""Actual-CUDA composition of BRAIDS bed-load adaptation with Q3 lifecycles.

The numerical kernel is the already exercised class/cell Jacobi operator in
``validation.gpu_fwave.bedload_adaptation``.  This additive wrapper removes
its historical MPM/SSPRK33 admission assumption and binds it to an explicit
v72 execution receipt.  It then hands the adapted cell vector to exactly one
existing scalar-v65 or direct-vector-v67 face/inventory/Exner owner.

Ordering is intentional: constitutive capacity and permitted cell-direction
steering -> stage-local adaptation -> face reconstruction/global face-slope
correction -> one availability projection -> one Exner transaction.  A
cell-centred streamline PDE cannot consume a face-slope result without
inventing a second inverse reconstruction, so the global FLOW face correction
remains downstream.  This is the same ordering as BRAIDS' qualified native
extension and is not claimed to be a Delft3D operator.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Mapping

import numpy as np

from braids_core.sediment.bedload_adaptation_live_v72 import (
    BACKEND_GPU,
    BOUNDARY_EQUILIBRIUM,
    CAPACITY_DIRECT,
    CAPACITY_SCALAR,
    INLET_HYDRAULIC_MODES,
    MAX_ACCEPTED_STEPS,
    SCOPE_ID,
    BedloadAdaptationTopologyV72,
    BedloadAdaptationV72Error,
    _fingerprint,
    load_execution_v72,
    restart_document_v72,
    validate_topology_v72,
)


@dataclass(frozen=True)
class ResidentAdaptedCapacityV72:
    execution_fingerprint: str
    topology_sha256: str
    parent_capacity_sha256: str
    adapted_capacity: object
    iterations: int
    residual: float
    tolerance: float
    result_sha256: str
    device_id: int


@dataclass(frozen=True)
class ResidentDirectAdaptationTrialV72:
    execution_fingerprint: str
    parent_state_sha256: str
    parent_cell_channel_sha256: str
    adaptation: ResidentAdaptedCapacityV72
    adapted_cell_channel_sha256: str
    face_trial: object
    trial_sha256: str
    device_id: int

    @property
    def candidate(self):
        return self.face_trial.candidate


@dataclass(frozen=True)
class ResidentScalarAdaptationTrialV72:
    execution_fingerprint: str
    parent_state_sha256: str
    parent_production_sha256: str
    adaptation: ResidentAdaptedCapacityV72
    adapted_cell_channel_sha256: str
    face_trial: object
    trial_sha256: str
    device_id: int

    @property
    def candidate(self):
        return self.face_trial.candidate


class ResidentBedloadAdaptationV72:
    """One selected-GPU, stage-local adaptation operator and outer composer."""

    backend = BACKEND_GPU
    integrator = "implicit_fwave_midpoint"

    def __init__(self, execution: Mapping[str, object],
                 topology: BedloadAdaptationTopologyV72, *,
                 max_iterations: int = 4096):
        import cupy as cp
        from validation.gpu_fwave.bedload_adaptation import (
            ResidentBedloadAdaptation,
        )

        self.cp = cp
        self.device_id = int(cp.cuda.Device().id)
        self.execution = load_execution_v72(execution)
        if (self.execution["backend"] != self.backend or
                self.execution["integrator"] != self.integrator or
                self.execution["automatic_backend_fallback"] is not False):
            raise BedloadAdaptationV72Error(
                "Q3/v72 CUDA requires explicit implicit_fwave_midpoint_gpu")
        self.topology = validate_topology_v72(self.execution, topology)
        self.nclass = len(self.execution["fraction_ids"])
        self.ncell = topology.ncell
        self.nface = topology.nface
        self.nboundary = topology.nboundary
        if (isinstance(max_iterations, bool) or type(max_iterations) is not int
                or max_iterations < 1):
            raise BedloadAdaptationV72Error(
                "Q3/v72 CUDA max_iterations must be a positive integer")
        root = Path(__file__).resolve().parents[1]
        source = root / "validation/gpu_fwave/bedload_adaptation.cu"
        if not source.is_file():
            raise BedloadAdaptationV72Error(
                "Q3/v72 reused CUDA adaptation kernel is missing")
        self.cuda_source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
        # ResidentBedloadAdaptation historically admitted only MPM.  Its CUDA
        # kernel itself is law-neutral and consumes only q, q_eq, D50, feed,
        # and topology.  The v72 receipt above owns the real family identity;
        # this private compatibility value merely reaches the reused kernel.
        data = {
            "q": np.zeros((3, topology.ncell), dtype=np.float64),
            "xy": topology.xy, "mid": topology.mid,
            "normal": topology.normal, "length": topology.face_length_m,
            "area": topology.cell_area_m2, "owner": topology.owner,
            "neighbour": topology.neighbour, "bid": topology.boundary_index,
            "offset": topology.cell_face_offset, "faces": topology.cell_faces,
            "signs": topology.signs, "mode": topology.hydraulic_mode,
            "bed_boundary_mode": topology.bed_boundary_mode,
            "d50_fraction": np.asarray(self.execution["d50_m"]),
            "bedlaw": 2, "bedload_adaptation": 1,
            "bedload_adaptation_length_diameters":
                self.execution["length_diameters"],
            "dry_tolerance": 1.0e-8, "dx": 1.0, "dy": 1.0,
        }
        self._kernel = ResidentBedloadAdaptation(
            data, source_path=source, max_iterations=max_iterations)
        self._dry = None
        self._dx = None
        self._dy = None

    def _digest(self, value) -> str:
        array = self.cp.asnumpy(value)
        digest = hashlib.sha256()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(tuple(array.shape)).encode("ascii"))
        digest.update(np.ascontiguousarray(array).tobytes(order="C"))
        return digest.hexdigest()

    def _array(self, value, shape, name):
        cp = self.cp
        if (not isinstance(value, cp.ndarray) or value.dtype != cp.float64 or
                value.shape != shape or not value.flags.c_contiguous or
                int(value.device.id) != self.device_id):
            raise BedloadAdaptationV72Error(
                f"Q3/v72 CUDA {name} must be resident contiguous FP64 {shape}")
        return value

    def _configure_geometry_scale(self, dry_tolerance: float,
                                  dx_m: float, dy_m: float) -> None:
        values = (float(dry_tolerance), float(dx_m), float(dy_m))
        if (not np.isfinite(values).all() or values[0] < 0.0 or
                values[1] <= 0.0 or values[2] <= 0.0):
            raise BedloadAdaptationV72Error(
                "Q3/v72 CUDA dry tolerance and grid scales are invalid")
        self._kernel.dry, self._kernel.dx, self._kernel.dy = values
        self._dry, self._dx, self._dy = values

    def relax(self, *, q, equilibrium_capacity, boundary_feed_kg_m_s,
              dry_tolerance: float, dx_m: float, dy_m: float,
              cancelled=lambda: False) -> ResidentAdaptedCapacityV72:
        """Return an authenticated device-resident derived capacity."""
        cp = self.cp
        if not callable(cancelled):
            raise TypeError("cancelled must be callable")
        if cancelled():
            raise InterruptedError("Q3/v72 CUDA adaptation cancelled before launch")
        q = self._array(q, (3, self.ncell), "hydraulic state")
        equilibrium = self._array(
            equilibrium_capacity, (2, self.nclass, self.ncell),
            "equilibrium capacity")
        feed = self._array(
            boundary_feed_kg_m_s, (self.nclass, self.nboundary),
            "boundary feed")
        if bool(cp.any(feed < 0.0).item()):
            raise BedloadAdaptationV72Error(
                "Q3/v72 CUDA boundary feed must be nonnegative")
        self._configure_geometry_scale(dry_tolerance, dx_m, dy_m)
        adapted = equilibrium.copy(order="C")
        diagnostics = self._kernel.relax(q, adapted, feed)
        if cancelled():
            # The result is derived work only.  Do not publish its receipt.
            raise InterruptedError(
                "Q3/v72 CUDA adaptation cancelled before publication")
        parent_digest = self._digest(equilibrium)
        adapted_digest = self._digest(adapted)
        token = _fingerprint({
            "scope_id": SCOPE_ID,
            "execution_fingerprint": self.execution["fingerprint"],
            "topology_sha256": self.topology.topology_sha256,
            "parent_capacity_sha256": parent_digest,
            "adapted_capacity_sha256": adapted_digest,
            "iterations": diagnostics["iterations"],
            "residual": diagnostics["residual"],
            "tolerance": diagnostics["tolerance"],
            "cuda_source_sha256": self.cuda_source_sha256,
        })
        return ResidentAdaptedCapacityV72(
            self.execution["fingerprint"], self.topology.topology_sha256,
            parent_digest, adapted, int(diagnostics["iterations"]),
            float(diagnostics["residual"]), float(diagnostics["tolerance"]),
            token, self.device_id,
        )

    def _validate_result(self, result: ResidentAdaptedCapacityV72,
                         parent_capacity) -> None:
        if (not isinstance(result, ResidentAdaptedCapacityV72) or
                result.execution_fingerprint != self.execution["fingerprint"] or
                result.topology_sha256 != self.topology.topology_sha256 or
                result.parent_capacity_sha256 != self._digest(parent_capacity) or
                result.device_id != self.device_id or
                result.adapted_capacity.shape !=
                    (2, self.nclass, self.ncell)):
            raise BedloadAdaptationV72Error(
                "Q3/v72 CUDA adapted-capacity receipt differs")
        expected = _fingerprint({
            "scope_id": SCOPE_ID,
            "execution_fingerprint": self.execution["fingerprint"],
            "topology_sha256": self.topology.topology_sha256,
            "parent_capacity_sha256": result.parent_capacity_sha256,
            "adapted_capacity_sha256": self._digest(result.adapted_capacity),
            "iterations": result.iterations, "residual": result.residual,
            "tolerance": result.tolerance,
            "cuda_source_sha256": self.cuda_source_sha256,
        })
        if expected != result.result_sha256:
            raise BedloadAdaptationV72Error(
                "Q3/v72 CUDA adapted capacity is stale or mutated")

    def _require_live_equilibrium_inlets(self, feed) -> None:
        """Prevent a second/incorrect prescribed-feed boundary owner."""
        cp = self.cp
        if bool(cp.any(feed != 0.0).item()):
            raise BedloadAdaptationV72Error(
                "Q3/v72 live Exner composition rejects prescribed external "
                "feed until the Q6/v70 single-owner boundary merger")
        for mode, bed_mode in zip(self.topology.hydraulic_mode,
                                  self.topology.bed_boundary_mode, strict=True):
            if (int(mode) in INLET_HYDRAULIC_MODES and
                    int(bed_mode) != BOUNDARY_EQUILIBRIUM):
                raise BedloadAdaptationV72Error(
                    "Q3/v72 live inlet must use equilibrium capacity; "
                    "prescribed feed remains fail-closed")

    def _match_direct_topology(self, topology) -> None:
        cp = self.cp
        face = topology.face_topology
        checks = (
            cp.array_equal(face.owner, cp.asarray(self.topology.owner)),
            cp.array_equal(face.neighbour, cp.asarray(self.topology.neighbour)),
            cp.array_equal(face.normal, cp.asarray(self.topology.normal)),
            cp.array_equal(face.face_length,
                           cp.asarray(self.topology.face_length_m)),
            cp.array_equal(face.cell_area,
                           cp.asarray(self.topology.cell_area_m2)),
            cp.array_equal(face.offset,
                           cp.asarray(self.topology.cell_face_offset)),
            cp.array_equal(face.adjacency_faces,
                           cp.asarray(self.topology.cell_faces)),
            cp.array_equal(face.signs, cp.asarray(self.topology.signs)),
        )
        if not all(bool(value.item()) for value in checks):
            raise BedloadAdaptationV72Error(
                "Q3/v72 adaptation and direct-family topology differ")

    def begin_direct_trial(
        self, *, parent, state, cells, traces, topology, q,
        boundary_feed_kg_m_s, dry_tolerance: float, dx_m: float, dy_m: float,
        dt_s: float, morphological_factor: float = 1.0,
        left_bed=None, right_bed=None, cancelled=lambda: False,
    ) -> ResidentDirectAdaptationTrialV72:
        """Compose direct-v67 capacity -> adaptation -> one v67 Exner owner."""
        if self.execution["capacity_family"] != CAPACITY_DIRECT:
            raise BedloadAdaptationV72Error(
                "Q3/v72 execution is not a direct-vector family profile")
        if (getattr(parent, "execution", {}).get("fingerprint") !=
                self.execution["parent_execution_fingerprint"] or
                getattr(parent, "backend", None) != self.backend):
            raise BedloadAdaptationV72Error(
                "Q3/v72 direct parent execution/backend differs")
        if state.accepted_step >= MAX_ACCEPTED_STEPS:
            raise BedloadAdaptationV72Error(
                "Q3/v72 bounded accepted-step limit reached")
        parent._validate_channels(cells, point_role="cells")
        parent._validate_channels(traces, point_role="traces")
        parent._validate_topology(topology)
        self._match_direct_topology(topology)
        feed = self._array(
            boundary_feed_kg_m_s, (self.nclass, self.nboundary),
            "boundary feed")
        self._require_live_equilibrium_inlets(feed)
        adapted = self.relax(
            q=q, equilibrium_capacity=cells.bed_mass_vector,
            boundary_feed_kg_m_s=feed, dry_tolerance=dry_tolerance,
            dx_m=dx_m, dy_m=dy_m, cancelled=cancelled)
        self._validate_result(adapted, cells.bed_mass_vector)
        cell_channels = parent.face_consumer.prepare_channels(
            point_role="cells", bed_mass_vector=adapted.adapted_capacity,
            status=cells.status, producer_fingerprint=adapted.result_sha256,
            surface_fraction_application_count=0)
        face_trial = parent.face_consumer.begin_trial(
            state=state, cells=cell_channels,
            traces=traces.face_channels, topology=topology.face_topology,
            dt_s=dt_s, morphological_factor=morphological_factor,
            left_bed=left_bed, right_bed=right_bed, cancelled=cancelled)
        token = _fingerprint({
            "scope_id": SCOPE_ID,
            "execution_fingerprint": self.execution["fingerprint"],
            "parent_state_sha256": state.state_sha256,
            "parent_cell_channel_sha256": cells.channel_sha256,
            "adaptation_result_sha256": adapted.result_sha256,
            "adapted_cell_channel_sha256": cell_channels.channel_sha256,
            "face_trial_sha256": face_trial.trial_sha256,
            "availability_owner_count":
                face_trial.active_fraction_application_count,
            "inventory_owner_count":
                face_trial.inventory_fill_application_count,
            "exner_owner_count": 1,
        })
        return ResidentDirectAdaptationTrialV72(
            self.execution["fingerprint"], state.state_sha256,
            cells.channel_sha256, adapted, cell_channels.channel_sha256,
            face_trial, token, self.device_id)

    def _validate_direct_trial(self, parent, state, trial) -> None:
        if (not isinstance(trial, ResidentDirectAdaptationTrialV72) or
                trial.execution_fingerprint != self.execution["fingerprint"] or
                trial.parent_state_sha256 != state.state_sha256 or
                trial.device_id != self.device_id):
            raise BedloadAdaptationV72Error(
                "Q3/v72 direct trial parent differs")
        parent.face_consumer._validate_trial(state, trial.face_trial)
        token = _fingerprint({
            "scope_id": SCOPE_ID,
            "execution_fingerprint": self.execution["fingerprint"],
            "parent_state_sha256": state.state_sha256,
            "parent_cell_channel_sha256": trial.parent_cell_channel_sha256,
            "adaptation_result_sha256": trial.adaptation.result_sha256,
            "adapted_cell_channel_sha256": trial.adapted_cell_channel_sha256,
            "face_trial_sha256": trial.face_trial.trial_sha256,
            "availability_owner_count":
                trial.face_trial.active_fraction_application_count,
            "inventory_owner_count":
                trial.face_trial.inventory_fill_application_count,
            "exner_owner_count": 1,
        })
        if token != trial.trial_sha256:
            raise BedloadAdaptationV72Error(
                "Q3/v72 direct trial receipt is stale or mutated")

    def commit_direct_trial(self, parent, state, trial):
        self._validate_direct_trial(parent, state, trial)
        return parent.face_consumer.commit_trial(state, trial.face_trial)

    def rollback_direct_trial(self, parent, state, trial):
        self._validate_direct_trial(parent, state, trial)
        return parent.face_consumer.rollback_trial(state, trial.face_trial)

    def cancel_direct_trial(self, parent, state, trial):
        self._validate_direct_trial(parent, state, trial)
        return parent.face_consumer.cancel_trial(state, trial.face_trial)

    def begin_scalar_trial(
        self, *, parent, state, production, faces, q,
        boundary_feed_kg_m_s, dry_tolerance: float, dx_m: float, dy_m: float,
        cell_area_m2, dt_s: float, morphological_factor: float = 1.0,
        lateral_stencil=None, cancelled=lambda: False,
    ) -> ResidentScalarAdaptationTrialV72:
        """Compose scalar-v65 capacity -> adaptation -> one v65 Exner owner."""
        from braids_gpu.scalar_flow_live_v65 import (
            ResidentScalarFlowV65CellChannels,
        )

        if self.execution["capacity_family"] != CAPACITY_SCALAR:
            raise BedloadAdaptationV72Error(
                "Q3/v72 execution is not a scalar-v65 profile")
        if (getattr(parent, "execution", {}).get("fingerprint") !=
                self.execution["parent_execution_fingerprint"] or
                getattr(parent, "backend", None) != self.backend):
            raise BedloadAdaptationV72Error(
                "Q3/v72 scalar parent execution/backend differs")
        if state.accepted_step >= MAX_ACCEPTED_STEPS:
            raise BedloadAdaptationV72Error(
                "Q3/v72 bounded accepted-step limit reached")
        nfrac, ncell = map(int, state.bed_mass_kg_m2.shape)
        parent._validate_production(production, nfrac, ncell)
        feed = self._array(
            boundary_feed_kg_m_s, (self.nclass, self.nboundary),
            "boundary feed")
        self._require_live_equilibrium_inlets(feed)
        adapted = self.relax(
            q=q, equilibrium_capacity=production.channels.composed_bed_mass,
            boundary_feed_kg_m_s=feed, dry_tolerance=dry_tolerance,
            dx_m=dx_m, dy_m=dy_m, cancelled=cancelled)
        self._validate_result(adapted, production.channels.composed_bed_mass)
        retained = production.channels.retained_suspended_mass
        total = self.cp.ascontiguousarray(adapted.adapted_capacity + retained)
        arrays = (
            adapted.adapted_capacity, retained,
            production.channels.signed_suspended_mass_rate,
            production.channels.equilibrium_concentration, total,
        )
        channels = ResidentScalarFlowV65CellChannels(
            parent.execution["fingerprint"],
            parent.producer_execution["fingerprint"], *arrays,
            parent._digest(*arrays, label="cell_channels"), self.device_id)
        face_trial = parent._begin_bed_exner_trial_impl(
            state=state, channels=channels, faces=faces,
            cell_area_m2=cell_area_m2, dt_s=dt_s,
            morphological_factor=morphological_factor,
            lateral_stencil=lateral_stencil, cancelled=cancelled,
            source_production_sha256=production.production_sha256)
        token = _fingerprint({
            "scope_id": SCOPE_ID,
            "execution_fingerprint": self.execution["fingerprint"],
            "parent_state_sha256": state.state_sha256,
            "parent_production_sha256": production.production_sha256,
            "adaptation_result_sha256": adapted.result_sha256,
            "adapted_cell_channel_sha256": channels.digest,
            "face_trial_sha256": face_trial.trial_sha256,
            "availability_owner_count":
                face_trial.active_fraction_application_count,
            "inventory_owner_count":
                face_trial.inventory_fill_application_count,
            "exner_owner_count": 1,
        })
        return ResidentScalarAdaptationTrialV72(
            self.execution["fingerprint"], state.state_sha256,
            production.production_sha256, adapted, channels.digest,
            face_trial, token, self.device_id)

    def _validate_scalar_trial(self, parent, state, trial) -> None:
        if (not isinstance(trial, ResidentScalarAdaptationTrialV72) or
                trial.execution_fingerprint != self.execution["fingerprint"] or
                trial.parent_state_sha256 != state.state_sha256 or
                trial.device_id != self.device_id):
            raise BedloadAdaptationV72Error(
                "Q3/v72 scalar trial parent differs")
        parent._validate_trial(state, trial.face_trial)
        token = _fingerprint({
            "scope_id": SCOPE_ID,
            "execution_fingerprint": self.execution["fingerprint"],
            "parent_state_sha256": state.state_sha256,
            "parent_production_sha256": trial.parent_production_sha256,
            "adaptation_result_sha256": trial.adaptation.result_sha256,
            "adapted_cell_channel_sha256": trial.adapted_cell_channel_sha256,
            "face_trial_sha256": trial.face_trial.trial_sha256,
            "availability_owner_count":
                trial.face_trial.active_fraction_application_count,
            "inventory_owner_count":
                trial.face_trial.inventory_fill_application_count,
            "exner_owner_count": 1,
        })
        if token != trial.trial_sha256:
            raise BedloadAdaptationV72Error(
                "Q3/v72 scalar trial receipt is stale or mutated")

    def commit_scalar_trial(self, parent, state, trial):
        self._validate_scalar_trial(parent, state, trial)
        return parent.commit_trial(state, trial.face_trial)

    def rollback_scalar_trial(self, parent, state, trial):
        self._validate_scalar_trial(parent, state, trial)
        return parent.rollback_trial(state, trial.face_trial)

    def cancel_scalar_trial(self, parent, state, trial):
        self._validate_scalar_trial(parent, state, trial)
        return parent.cancel_trial(state, trial.face_trial)

    def checkpoint_document(self, parent_checkpoint: Mapping[str, object],
                            *, accepted_step: int) -> dict[str, object]:
        """Wrap a parent checkpoint without serialising derived work arrays."""
        if (isinstance(accepted_step, bool) or type(accepted_step) is not int or
                accepted_step < 0 or accepted_step > MAX_ACCEPTED_STEPS):
            raise BedloadAdaptationV72Error(
                "Q3/v72 checkpoint accepted step is invalid")
        body = {
            "schema": 1, "scope_id": SCOPE_ID,
            "execution_fingerprint": self.execution["fingerprint"],
            "accepted_step": accepted_step,
            "adaptation_restart": restart_document_v72(self.execution),
            "parent_checkpoint": dict(parent_checkpoint),
            "derived_device_arrays_serialized": False,
        }
        return {**body, "fingerprint": _fingerprint(body)}


__all__ = [
    "ResidentAdaptedCapacityV72", "ResidentBedloadAdaptationV72",
    "ResidentDirectAdaptationTrialV72", "ResidentScalarAdaptationTrialV72",
]
