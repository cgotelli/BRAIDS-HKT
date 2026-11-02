"""Resident CUDA composition of v67 bed load with v71 wet/dry AMR.

The caller explicitly supplies an authenticated v67 runtime for every
selected topology.  No backend selection or CPU fallback occurs here.  A
v71 regrid moves only committed bed inventory and cumulative boundary
ledgers; topology-local face data are re-created as exact zero.  Transport
trials remain v67 trials, so formula evaluation, active fractions, finite
inventory and Exner are neither copied nor reimplemented.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np

from braids_core.sediment.direct_vector_wetdry_amr_live_v71 import (
    ConservativeTransferPlanV71,
    DirectVectorWetDryAMRV71Error,
    FACE_CLOSED,
    INTEGRATOR,
    SCOPE_ID,
    WetDryTopologyV71,
    execution_document_v71,
)
from .direct_vector_family_live_v67 import (
    ResidentDirectVectorFamilyLiveV67,
    ResidentDirectVectorFamilyTopologyV67,
    ResidentDirectVectorFamilyTrialV67,
)
from .direct_vector_face_exner_v67 import ResidentDirectVectorStateV67


CUDA_OPTIONS = (
    "--std=c++11", "--fmad=false", "--prec-div=true",
    "--prec-sqrt=true", "--ftz=false",
)
CUDA_SOURCE_NAME = "direct_vector_wetdry_amr_live_v71.cu"
KERNEL_NAMES = (
    "v71_transfer_bed_mass", "v71_validate_wetdry_faces",
    "v71_clear_face_register",
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class ResidentWetDryTopologyBindingV71:
    topology: WetDryTopologyV71
    runtime: ResidentDirectVectorFamilyLiveV67
    family_topology: ResidentDirectVectorFamilyTopologyV67
    node_ids: object
    cell_area_m2: object
    wet: object
    binding_sha256: str
    device_id: int


@dataclass(frozen=True)
class ResidentWetDryAMRStateV71:
    topology_id: str
    mesh_epoch: int
    direct_state: ResidentDirectVectorStateV67
    node_ids: object
    cell_area_m2: object
    wet: object
    accepted_face_flux: object
    accepted_cell_availability: object
    state_sha256: str
    device_id: int


@dataclass(frozen=True)
class ResidentWetDryTransportTrialV71:
    parent_state_sha256: str
    topology_id: str
    family_trial: ResidentDirectVectorFamilyTrialV67
    candidate: ResidentWetDryAMRStateV71
    dry_face_flux_max_abs: float
    trial_sha256: str
    device_id: int


@dataclass(frozen=True)
class ResidentWetDryRegridTrialV71:
    parent_state_sha256: str
    old_topology_id: str
    new_topology_id: str
    transfer_plan_id: str
    candidate: ResidentWetDryAMRStateV71
    conservation_residual_kg: object
    transition_kind: str
    wet_state_change_count: int
    transfer_application_count: int
    face_register_cleanup_count: int
    availability_cleanup_count: int
    exner_application_count: int
    trial_sha256: str
    device_id: int


class ResidentDirectVectorWetDryAMRV71:
    """One user-selected, synchronous actual-CUDA v71 transaction owner."""

    backend = "implicit_fwave_midpoint_gpu"
    integrator = INTEGRATOR

    def __init__(
        self,
        *,
        runtimes: Mapping[str, ResidentDirectVectorFamilyLiveV67],
        maximum_accepted_steps: int = 4,
        source_path: str | Path | None = None,
    ):
        import cupy as cp

        if not isinstance(runtimes, Mapping) or not runtimes:
            raise DirectVectorWetDryAMRV71Error(
                "Q3-H7/v71 requires explicit topology-specific GPU runtimes"
            )
        self.cp = cp
        self.device_id = int(cp.cuda.Device().id)
        checked: dict[str, ResidentDirectVectorFamilyLiveV67] = {}
        physics = None
        for topology_id, runtime in runtimes.items():
            if (
                not isinstance(topology_id, str) or len(topology_id) != 64
                or not isinstance(runtime, ResidentDirectVectorFamilyLiveV67)
                or runtime.backend != self.backend
                or runtime.integrator != INTEGRATOR
                or runtime.device_id != self.device_id
                or runtime.execution["dependencies"]["topology"] != topology_id
                or runtime.runtime_receipt["automatic_backend_fallback"] is not False
            ):
                raise DirectVectorWetDryAMRV71Error(
                    "Q3-H7/v71 runtime/backend/topology binding differs"
                )
            current = runtime.execution["physical_profile_fingerprint"]
            if physics is None:
                physics = current
            elif current != physics:
                raise DirectVectorWetDryAMRV71Error(
                    "Q3-H7/v71 topology runtimes do not share one physical profile"
                )
            checked[topology_id] = runtime
        assert physics is not None
        self.runtimes = MappingProxyType(checked)
        self.execution = execution_document_v71(
            backend=self.backend, physics_fingerprint=physics,
            topology_ids=tuple(checked),
            maximum_accepted_steps=maximum_accepted_steps,
        )
        self.maximum_accepted_steps = maximum_accepted_steps

        path = Path(source_path or Path(__file__).with_name(CUDA_SOURCE_NAME)).resolve()
        if not path.is_file() or path.is_symlink():
            raise DirectVectorWetDryAMRV71Error(
                "Q3-H7/v71 CUDA source is absent or linked"
            )
        source = path.read_text(encoding="utf-8")
        with cp.cuda.Device(self.device_id):
            module = cp.RawModule(
                code=source, options=CUDA_OPTIONS, name_expressions=KERNEL_NAMES,
            )
            self.kernels = {
                name: module.get_function(name) for name in KERNEL_NAMES
            }
        self.source_receipt = MappingProxyType({
            "scope_id": SCOPE_ID,
            "python_source_sha256": _sha(Path(__file__).resolve()),
            "cuda_source_sha256": _sha(path),
            "cuda_source_name": path.name,
            "kernel_names": list(KERNEL_NAMES),
            "device_id": self.device_id,
            "automatic_backend_fallback": False,
        })
        self._open_parent: str | None = None

    def _resident(self, value: object, name: str, dtype, ndim: int, shape=None):
        cp = self.cp
        if (
            not isinstance(value, cp.ndarray) or value.dtype != dtype
            or value.ndim != ndim or not value.flags.c_contiguous
            or int(value.device.id) != self.device_id
            or (shape is not None and tuple(value.shape) != tuple(shape))
        ):
            raise DirectVectorWetDryAMRV71Error(
                f"Q3-H7/v71 {name} is not the required resident array"
            )
        return value

    def _state_token(
        self, binding: ResidentWetDryTopologyBindingV71,
        direct_state: ResidentDirectVectorStateV67,
        mesh_epoch: int,
        accepted_face_flux: object,
        accepted_cell_availability: object,
    ) -> str:
        cp = self.cp
        return _fingerprint({
            "scope_id": SCOPE_ID,
            "execution_fingerprint": self.execution["fingerprint"],
            "topology_id": binding.topology.topology_id,
            "binding_sha256": binding.binding_sha256,
            "mesh_epoch": mesh_epoch,
            "direct_state_sha256": direct_state.state_sha256,
            "accepted_step": direct_state.accepted_step,
            "time_s": direct_state.time_s,
            "accepted_face_flux_exact_zero": bool(
                cp.all(accepted_face_flux == 0.0).item()
            ),
            "face_register_shape": list(accepted_face_flux.shape),
            "accepted_cell_availability_exact_zero": bool(
                cp.all(accepted_cell_availability == 0.0).item()
            ),
            "availability_register_shape": list(
                accepted_cell_availability.shape
            ),
        })

    def _make_state(
        self,
        binding: ResidentWetDryTopologyBindingV71,
        direct_state: ResidentDirectVectorStateV67,
        *,
        mesh_epoch: int,
        accepted_face_flux: object,
        accepted_cell_availability: object,
    ) -> ResidentWetDryAMRStateV71:
        cp = self.cp
        runtime = binding.runtime
        runtime.face_consumer._validate_state(direct_state)
        if direct_state.accepted_step > self.maximum_accepted_steps:
            raise DirectVectorWetDryAMRV71Error(
                "Q3-H7/v71 bounded accepted-step limit exceeded"
            )
        flux = self._resident(
            accepted_face_flux, "accepted face register", cp.float64, 2,
            (runtime.nclass, binding.topology.nface),
        )
        availability = self._resident(
            accepted_cell_availability, "accepted availability register",
            cp.float64, 2, (runtime.nclass, binding.topology.ncell),
        )
        if not bool(cp.all(cp.isfinite(availability) &
                           (availability >= 0.0)).item()):
            raise DirectVectorWetDryAMRV71Error(
                "Q3-H7/v71 accepted availability register is invalid"
            )
        token = self._state_token(
            binding, direct_state, mesh_epoch, flux, availability,
        )
        return ResidentWetDryAMRStateV71(
            binding.topology.topology_id, mesh_epoch, direct_state,
            binding.node_ids, binding.cell_area_m2, binding.wet, flux,
            availability, token, self.device_id,
        )

    def _validate_binding(self, binding: ResidentWetDryTopologyBindingV71):
        cp = self.cp
        if (
            not isinstance(binding, ResidentWetDryTopologyBindingV71)
            or binding.device_id != self.device_id
            or binding.topology.topology_id not in self.runtimes
            or binding.runtime is not self.runtimes[binding.topology.topology_id]
        ):
            raise DirectVectorWetDryAMRV71Error(
                "Q3-H7/v71 selected topology binding differs"
            )
        binding.runtime._validate_topology(binding.family_topology)
        expected = _fingerprint({
            "scope_id": SCOPE_ID,
            "execution_fingerprint": self.execution["fingerprint"],
            "topology_id": binding.topology.topology_id,
            "family_topology_sha256": binding.family_topology.topology_sha256,
            "node_ids": cp.asnumpy(binding.node_ids).tolist(),
            "cell_area_m2": cp.asnumpy(binding.cell_area_m2).tolist(),
            "wet": cp.asnumpy(binding.wet).astype(bool).tolist(),
        })
        if expected != binding.binding_sha256:
            raise DirectVectorWetDryAMRV71Error(
                "Q3-H7/v71 selected topology binding is stale"
            )

    def _validate_state(
        self, state: ResidentWetDryAMRStateV71,
        binding: ResidentWetDryTopologyBindingV71,
    ):
        self._validate_binding(binding)
        if (
            not isinstance(state, ResidentWetDryAMRStateV71)
            or state.device_id != self.device_id
            or state.topology_id != binding.topology.topology_id
            or state.mesh_epoch < 0
        ):
            raise DirectVectorWetDryAMRV71Error(
                "Q3-H7/v71 accepted state/topology differs"
            )
        expected = self._state_token(
            binding, state.direct_state, state.mesh_epoch,
            state.accepted_face_flux, state.accepted_cell_availability,
        )
        if expected != state.state_sha256:
            raise DirectVectorWetDryAMRV71Error(
                "Q3-H7/v71 accepted state arrays are stale or mutated"
            )

    def prepare_binding(
        self,
        topology: WetDryTopologyV71,
        **family_topology_values,
    ) -> ResidentWetDryTopologyBindingV71:
        cp = self.cp
        if (
            not isinstance(topology, WetDryTopologyV71)
            or topology.topology_id not in self.runtimes
        ):
            raise DirectVectorWetDryAMRV71Error(
                "Q3-H7/v71 topology is not in the execution inventory"
            )
        runtime = self.runtimes[topology.topology_id]
        family = runtime.prepare_topology(**family_topology_values)
        face = family.face_topology
        if (
            not np.array_equal(cp.asnumpy(face.owner), topology.owner)
            or not np.array_equal(cp.asnumpy(face.neighbour), topology.neighbour)
            or not np.array_equal(cp.asnumpy(face.role), topology.role)
            or not np.array_equal(cp.asnumpy(face.cell_area), topology.cell_area_m2)
        ):
            raise DirectVectorWetDryAMRV71Error(
                "Q3-H7/v71 core and v67 face topology differ"
            )
        nodes = cp.ascontiguousarray(cp.asarray(topology.node_ids, dtype=cp.int64))
        area = cp.ascontiguousarray(cp.asarray(topology.cell_area_m2, dtype=cp.float64))
        wet = cp.ascontiguousarray(cp.asarray(topology.wet, dtype=cp.int32))
        status = cp.zeros(topology.nface, dtype=cp.int32)
        self.kernels["v71_validate_wetdry_faces"](
            ((topology.nface+255)//256,), (256,), (
                np.int32(topology.ncell), np.int32(topology.nface), wet,
                face.owner, face.neighbour, face.role, status,
            ),
        )
        if bool(cp.any(status != 0).item()):
            raise DirectVectorWetDryAMRV71Error(
                "Q3-H7/v71 CUDA wet/dry topology validation failed"
            )
        token = _fingerprint({
            "scope_id": SCOPE_ID,
            "execution_fingerprint": self.execution["fingerprint"],
            "topology_id": topology.topology_id,
            "family_topology_sha256": family.topology_sha256,
            "node_ids": topology.node_ids.tolist(),
            "cell_area_m2": topology.cell_area_m2.tolist(),
            "wet": topology.wet.tolist(),
        })
        return ResidentWetDryTopologyBindingV71(
            topology, runtime, family, nodes, area, wet, token, self.device_id,
        )

    def initial_state(
        self,
        binding: ResidentWetDryTopologyBindingV71,
        direct_state: ResidentDirectVectorStateV67,
    ) -> ResidentWetDryAMRStateV71:
        self._validate_binding(binding)
        cp = self.cp
        zero = cp.zeros(
            (binding.runtime.nclass, binding.topology.nface), dtype=cp.float64,
        )
        availability = cp.zeros(
            (binding.runtime.nclass, binding.topology.ncell), dtype=cp.float64,
        )
        return self._make_state(
            binding, direct_state, mesh_epoch=0, accepted_face_flux=zero,
            accepted_cell_availability=availability,
        )

    def prepare_transport_inputs(
        self,
        binding: ResidentWetDryTopologyBindingV71,
        *,
        q: object,
        trace: object,
        roughness_parameter: object,
        kinematic_viscosity: object,
        settling_velocity_cells: object,
        settling_velocity_traces: object,
        dry_tolerance: float,
    ):
        """Build v67 receipts while quarantining dry-cell producer work.

        v67 remains intentionally fully wet.  Dry cells therefore receive a
        finite inert producer state, but the v71 topology proves that no
        transport face can consume those values.  This is an explicit
        composition rule, not a fallback or a disabled constitutive law.
        """
        self._validate_binding(binding)
        cp = self.cp
        ncell, nface = binding.topology.ncell, binding.topology.nface
        q_value = self._resident(q, "physical q", cp.float64, 2, (3, ncell))
        trace_value = self._resident(
            trace, "physical trace", cp.float64, 2, (3, nface),
        )
        dry = float(dry_tolerance)
        if not np.isfinite(dry) or dry < 0.0:
            raise DirectVectorWetDryAMRV71Error(
                "Q3-H7/v71 dry tolerance differs"
            )
        physical_wet = q_value[0] > dry
        if not bool(cp.all(physical_wet == (binding.wet == 1)).item()):
            raise DirectVectorWetDryAMRV71Error(
                "Q3-H7/v71 hydraulic wet mask differs from selected topology"
            )
        safe = q_value.copy()
        minimum = max(
            1.0,
            float(binding.runtime.execution["physical_profile"][
                "minimum_work_depth_m"
            ]),
        )
        safe[0] = cp.where(binding.wet == 1, safe[0], minimum)
        safe[1] = cp.where(binding.wet == 1, safe[1], 0.0)
        safe[2] = cp.where(binding.wet == 1, safe[2], 0.0)
        cells = binding.runtime.cells(
            q=cp.ascontiguousarray(safe),
            roughness_parameter=roughness_parameter,
            kinematic_viscosity=kinematic_viscosity,
            active=cp.ones(ncell, dtype=cp.int32),
            dry_tolerance=dry,
            settling_velocity=settling_velocity_cells,
        )
        face = binding.family_topology.face_topology
        traces = binding.runtime.traces(
            trace=trace_value, owner=face.owner, role=face.role,
            roughness_parameter=roughness_parameter,
            kinematic_viscosity=kinematic_viscosity,
            dry_tolerance=dry,
            settling_velocity=settling_velocity_traces,
        )
        return cells, traces

    def _open(self, parent: str):
        if self._open_parent is not None:
            raise RuntimeError("Q3-H7/v71 already has an open transaction")
        self._open_parent = parent

    def _close(self, parent: str):
        if self._open_parent != parent:
            raise RuntimeError("Q3-H7/v71 transaction parent differs")
        self._open_parent = None

    def begin_transport_trial(
        self,
        state: ResidentWetDryAMRStateV71,
        binding: ResidentWetDryTopologyBindingV71,
        *, cells, traces, dt_s: float, morphological_factor: float = 1.0,
        cancelled=lambda: False,
    ) -> ResidentWetDryTransportTrialV71:
        self._validate_state(state, binding)
        if cancelled():
            raise RuntimeError("Q3-H7/v71 transport trial cancelled")
        family = binding.runtime.begin_trial(
            state=state.direct_state, cells=cells, traces=traces,
            topology=binding.family_topology, dt_s=dt_s,
            morphological_factor=morphological_factor, cancelled=cancelled,
        )
        cp = self.cp
        flux = family.face_trial.faces.flux
        dry_face = binding.family_topology.face_topology.role == FACE_CLOSED
        dry_max = (
            float(cp.max(cp.abs(flux[:, dry_face])).item())
            if bool(cp.any(dry_face).item()) else 0.0
        )
        if dry_max != 0.0:
            raise DirectVectorWetDryAMRV71Error(
                "Q3-H7/v71 closed/dry face acquired nonzero bed load"
            )
        old_mass = state.direct_state.bed_mass_kg_m2
        new_mass = family.face_trial.candidate.bed_mass_kg_m2
        dry_cell = binding.wet == 0
        if bool(cp.any(dry_cell).item()) and not bool(
            cp.array_equal(old_mass[:, dry_cell], new_mass[:, dry_cell]).item()
        ):
            raise DirectVectorWetDryAMRV71Error(
                "Q3-H7/v71 dry-cell bed inventory changed"
            )
        candidate = self._make_state(
            binding, family.face_trial.candidate, mesh_epoch=state.mesh_epoch,
            accepted_face_flux=cp.ascontiguousarray(flux.copy()),
            accepted_cell_availability=cp.ascontiguousarray(
                cp.where(
                    binding.wet[None, :] == 1,
                    family.face_trial.accepted_face_availability,
                    0.0,
                )
            ),
        )
        body = {
            "scope_id": SCOPE_ID,
            "kind": "transport_trial",
            "parent_state_sha256": state.state_sha256,
            "topology_id": state.topology_id,
            "family_trial_sha256": family.trial_sha256,
            "candidate_state_sha256": candidate.state_sha256,
            "dry_face_flux_max_abs": dry_max,
        }
        trial = ResidentWetDryTransportTrialV71(
            state.state_sha256, state.topology_id, family, candidate,
            dry_max, _fingerprint(body), self.device_id,
        )
        self._open(state.state_sha256)
        return trial

    def _validate_transport_trial(self, state, binding, trial):
        self._validate_state(state, binding)
        if (
            not isinstance(trial, ResidentWetDryTransportTrialV71)
            or trial.device_id != self.device_id
            or trial.parent_state_sha256 != state.state_sha256
            or trial.topology_id != state.topology_id
        ):
            raise DirectVectorWetDryAMRV71Error(
                "Q3-H7/v71 transport trial parent differs"
            )
        binding.runtime._validate_trial(state.direct_state, trial.family_trial)
        body = {
            "scope_id": SCOPE_ID, "kind": "transport_trial",
            "parent_state_sha256": state.state_sha256,
            "topology_id": state.topology_id,
            "family_trial_sha256": trial.family_trial.trial_sha256,
            "candidate_state_sha256": trial.candidate.state_sha256,
            "dry_face_flux_max_abs": trial.dry_face_flux_max_abs,
        }
        if trial.trial_sha256 != _fingerprint(body):
            raise DirectVectorWetDryAMRV71Error(
                "Q3-H7/v71 transport trial receipt is stale"
            )

    def commit_transport_trial(self, state, binding, trial):
        self._validate_transport_trial(state, binding, trial)
        binding.runtime.commit_trial(state.direct_state, trial.family_trial)
        self._close(state.state_sha256)
        return trial.candidate

    def rollback_transport_trial(self, state, binding, trial):
        self._validate_transport_trial(state, binding, trial)
        binding.runtime.rollback_trial(state.direct_state, trial.family_trial)
        self._close(state.state_sha256)
        return state

    def cancel_transport_trial(self, state, binding, trial):
        self._validate_transport_trial(state, binding, trial)
        binding.runtime.cancel_trial(state.direct_state, trial.family_trial)
        self._close(state.state_sha256)
        return state

    def begin_regrid_trial(
        self,
        state: ResidentWetDryAMRStateV71,
        old_binding: ResidentWetDryTopologyBindingV71,
        new_binding: ResidentWetDryTopologyBindingV71,
        plan: ConservativeTransferPlanV71,
        *,
        cancelled=lambda: False,
    ) -> ResidentWetDryRegridTrialV71:
        self._validate_state(state, old_binding)
        self._validate_binding(new_binding)
        if (
            not isinstance(plan, ConservativeTransferPlanV71)
            or plan.old_topology_id != old_binding.topology.topology_id
            or plan.new_topology_id != new_binding.topology.topology_id
            or state.direct_state.accepted_step > self.maximum_accepted_steps
        ):
            raise DirectVectorWetDryAMRV71Error(
                "Q3-H7/v71 transfer plan/state differs"
            )
        if cancelled():
            raise RuntimeError("Q3-H7/v71 regrid trial cancelled")
        cp = self.cp
        nclass = old_binding.runtime.nclass
        if new_binding.runtime.nclass != nclass:
            raise DirectVectorWetDryAMRV71Error(
                "Q3-H7/v71 fraction count changed across regrid"
            )
        old_index = cp.ascontiguousarray(cp.asarray(plan.old_index, dtype=cp.int32))
        offsets = cp.ascontiguousarray(cp.asarray(plan.new_offsets, dtype=cp.int32))
        overlap = cp.ascontiguousarray(
            cp.asarray(plan.overlap_area_m2, dtype=cp.float64)
        )
        mass = cp.zeros((nclass, new_binding.topology.ncell), dtype=cp.float64)
        status = cp.zeros(new_binding.topology.ncell, dtype=cp.int32)
        count = nclass*new_binding.topology.ncell
        self.kernels["v71_transfer_bed_mass"](
            ((count+255)//256,), (256,), (
                np.int32(nclass), np.int32(old_binding.topology.ncell),
                np.int32(new_binding.topology.ncell), offsets, old_index,
                overlap, new_binding.cell_area_m2,
                state.direct_state.bed_mass_kg_m2, mass, status,
            ),
        )
        if bool(cp.any(status != 0).item()):
            raise DirectVectorWetDryAMRV71Error(
                "Q3-H7/v71 CUDA conservative transfer failed"
            )
        before = cp.sum(
            state.direct_state.bed_mass_kg_m2*old_binding.cell_area_m2[None, :],
            axis=1,
        )
        after = cp.sum(mass*new_binding.cell_area_m2[None, :], axis=1)
        residual = cp.ascontiguousarray(after-before)
        scale = cp.maximum(1.0, cp.maximum(cp.abs(before), cp.abs(after)))
        if bool(cp.any(cp.abs(residual) > 16384.0*np.finfo(np.float64).eps*scale).item()):
            raise DirectVectorWetDryAMRV71Error(
                "Q3-H7/v71 CUDA transfer is not conservative"
            )
        old = state.direct_state
        candidate_direct = new_binding.runtime.resolve_state(
            bed_mass_kg_m2=cp.ascontiguousarray(mass),
            boundary_signed_kg=cp.ascontiguousarray(old.boundary_signed_kg.copy()),
            boundary_absolute_kg=cp.ascontiguousarray(
                old.boundary_absolute_kg.copy()
            ),
            accepted_step=old.accepted_step,
            time_s=old.time_s,
        )
        register = cp.empty(
            (nclass, new_binding.topology.nface), dtype=cp.float64,
        )
        register.fill(np.nan)
        register_count = int(register.size)
        self.kernels["v71_clear_face_register"](
            ((register_count+255)//256,), (256,), (
                np.int64(register_count), register,
            ),
        )
        if not bool(cp.all(register == 0.0).item()):
            raise DirectVectorWetDryAMRV71Error(
                "Q3-H7/v71 topology-local face cleanup failed"
            )
        # Availability is a topology-local face/Exner work register.  It is
        # never inherited by a new wet/dry owner or a new AMR cover.
        availability = cp.zeros(
            (nclass, new_binding.topology.ncell), dtype=cp.float64,
        )
        same_cover = (
            plan.classification == "identity_or_relabel"
            and old_binding.topology.ncell == new_binding.topology.ncell
            and np.array_equal(
                old_binding.topology.node_ids, new_binding.topology.node_ids
            )
            and np.array_equal(
                old_binding.topology.cell_area_m2,
                new_binding.topology.cell_area_m2,
            )
        )
        wet_state_change_count = (
            int(np.count_nonzero(
                old_binding.topology.wet != new_binding.topology.wet
            )) if same_cover else 0
        )
        if same_cover and wet_state_change_count > 0:
            transition_kind = "wetdry"
        elif plan.classification == "refine":
            transition_kind = "refine"
        elif plan.classification == "coarsen":
            transition_kind = "coarsen"
        else:
            transition_kind = "identity_or_relabel"
        candidate = self._make_state(
            new_binding, candidate_direct, mesh_epoch=state.mesh_epoch+1,
            accepted_face_flux=register,
            accepted_cell_availability=availability,
        )
        body = {
            "scope_id": SCOPE_ID, "kind": "regrid_trial",
            "parent_state_sha256": state.state_sha256,
            "old_topology_id": old_binding.topology.topology_id,
            "new_topology_id": new_binding.topology.topology_id,
            "transfer_plan_id": plan.plan_id,
            "candidate_state_sha256": candidate.state_sha256,
            "transition_kind": transition_kind,
            "wet_state_change_count": wet_state_change_count,
            "transfer_application_count": 1,
            "face_register_cleanup_count": 1,
            "availability_cleanup_count": 1,
            "exner_application_count": 0,
            "conservation_residual_kg": cp.asnumpy(residual).tolist(),
        }
        trial = ResidentWetDryRegridTrialV71(
            state.state_sha256, old_binding.topology.topology_id,
            new_binding.topology.topology_id, plan.plan_id, candidate,
            residual, transition_kind, wet_state_change_count, 1, 1, 1, 0,
            _fingerprint(body), self.device_id,
        )
        self._open(state.state_sha256)
        return trial

    def _validate_regrid_trial(self, state, old_binding, new_binding, plan, trial):
        self._validate_state(state, old_binding)
        self._validate_binding(new_binding)
        if (
            not isinstance(trial, ResidentWetDryRegridTrialV71)
            or trial.device_id != self.device_id
            or trial.parent_state_sha256 != state.state_sha256
            or trial.old_topology_id != old_binding.topology.topology_id
            or trial.new_topology_id != new_binding.topology.topology_id
            or trial.transfer_plan_id != plan.plan_id
            or trial.transfer_application_count != 1
            or trial.face_register_cleanup_count != 1
            or trial.availability_cleanup_count != 1
            or trial.exner_application_count != 0
            or trial.transition_kind not in {
                "wetdry", "refine", "coarsen", "identity_or_relabel",
            }
            or trial.wet_state_change_count < 0
        ):
            raise DirectVectorWetDryAMRV71Error(
                "Q3-H7/v71 regrid trial parent/receipt differs"
            )
        body = {
            "scope_id": SCOPE_ID, "kind": "regrid_trial",
            "parent_state_sha256": state.state_sha256,
            "old_topology_id": trial.old_topology_id,
            "new_topology_id": trial.new_topology_id,
            "transfer_plan_id": trial.transfer_plan_id,
            "candidate_state_sha256": trial.candidate.state_sha256,
            "transition_kind": trial.transition_kind,
            "wet_state_change_count": trial.wet_state_change_count,
            "transfer_application_count": 1,
            "face_register_cleanup_count": 1,
            "availability_cleanup_count": 1,
            "exner_application_count": 0,
            "conservation_residual_kg": self.cp.asnumpy(
                trial.conservation_residual_kg
            ).tolist(),
        }
        if trial.trial_sha256 != _fingerprint(body):
            raise DirectVectorWetDryAMRV71Error(
                "Q3-H7/v71 regrid trial arrays are stale"
            )

    def commit_regrid_trial(self, state, old_binding, new_binding, plan, trial):
        self._validate_regrid_trial(
            state, old_binding, new_binding, plan, trial,
        )
        self._close(state.state_sha256)
        return trial.candidate

    def rollback_regrid_trial(self, state, old_binding, new_binding, plan, trial):
        self._validate_regrid_trial(
            state, old_binding, new_binding, plan, trial,
        )
        self._close(state.state_sha256)
        return state

    def cancel_regrid_trial(self, state, old_binding, new_binding, plan, trial):
        self._validate_regrid_trial(
            state, old_binding, new_binding, plan, trial,
        )
        self._close(state.state_sha256)
        return state

    def state_checkpoint_document(
        self, state: ResidentWetDryAMRStateV71,
        binding: ResidentWetDryTopologyBindingV71,
    ) -> dict[str, object]:
        self._validate_state(state, binding)
        cp = self.cp
        body = {
            "schema": 1, "scope_id": SCOPE_ID,
            "execution_fingerprint": self.execution["fingerprint"],
            "backend": self.backend, "integrator": self.integrator,
            "topology_id": state.topology_id,
            "binding_sha256": binding.binding_sha256,
            "mesh_epoch": state.mesh_epoch,
            "node_ids": cp.asnumpy(state.node_ids).tolist(),
            "cell_area_m2": cp.asnumpy(state.cell_area_m2).tolist(),
            "wet": cp.asnumpy(state.wet).astype(bool).tolist(),
            "accepted_face_flux": cp.asnumpy(state.accepted_face_flux).tolist(),
            "accepted_cell_availability": cp.asnumpy(
                state.accepted_cell_availability
            ).tolist(),
            "state_sha256": state.state_sha256,
            "direct_state": binding.runtime.state_checkpoint_document(
                state.direct_state
            ),
            "source_receipt": dict(self.source_receipt),
        }
        return {**body, "fingerprint": _fingerprint(body)}

    def load_state_checkpoint(
        self,
        value: Mapping[str, object],
        binding: ResidentWetDryTopologyBindingV71,
    ) -> ResidentWetDryAMRStateV71:
        self._validate_binding(binding)
        if not isinstance(value, Mapping):
            raise DirectVectorWetDryAMRV71Error(
                "Q3-H7/v71 checkpoint must be an object"
            )
        body = {key: value[key] for key in value if key != "fingerprint"}
        if (
            value.get("fingerprint") != _fingerprint(body)
            or value.get("schema") != 1 or value.get("scope_id") != SCOPE_ID
            or value.get("execution_fingerprint") != self.execution["fingerprint"]
            or value.get("backend") != self.backend
            or value.get("integrator") != self.integrator
            or value.get("topology_id") != binding.topology.topology_id
            or value.get("binding_sha256") != binding.binding_sha256
            or value.get("source_receipt") != dict(self.source_receipt)
        ):
            raise DirectVectorWetDryAMRV71Error(
                "Q3-H7/v71 checkpoint header/source/topology differs"
            )
        cp = self.cp
        nodes = cp.ascontiguousarray(cp.asarray(value.get("node_ids"), dtype=cp.int64))
        area = cp.ascontiguousarray(
            cp.asarray(value.get("cell_area_m2"), dtype=cp.float64)
        )
        wet = cp.ascontiguousarray(cp.asarray(value.get("wet"), dtype=cp.int32))
        if (
            not bool(cp.array_equal(nodes, binding.node_ids).item())
            or not bool(cp.array_equal(area, binding.cell_area_m2).item())
            or not bool(cp.array_equal(wet, binding.wet).item())
        ):
            raise DirectVectorWetDryAMRV71Error(
                "Q3-H7/v71 checkpoint selected arrays differ"
            )
        direct = binding.runtime.load_state_checkpoint(value.get("direct_state"))
        flux = cp.ascontiguousarray(
            cp.asarray(value.get("accepted_face_flux"), dtype=cp.float64)
        )
        availability = cp.ascontiguousarray(cp.asarray(
            value.get("accepted_cell_availability"), dtype=cp.float64,
        ))
        mesh_epoch = value.get("mesh_epoch")
        if isinstance(mesh_epoch, bool) or not isinstance(mesh_epoch, int):
            raise DirectVectorWetDryAMRV71Error(
                "Q3-H7/v71 checkpoint mesh epoch differs"
            )
        state = self._make_state(
            binding, direct, mesh_epoch=mesh_epoch, accepted_face_flux=flux,
            accepted_cell_availability=availability,
        )
        if state.state_sha256 != value.get("state_sha256"):
            raise DirectVectorWetDryAMRV71Error(
                "Q3-H7/v71 checkpoint state token differs"
            )
        return state


__all__ = [
    "CUDA_OPTIONS", "CUDA_SOURCE_NAME", "KERNEL_NAMES",
    "ResidentDirectVectorWetDryAMRV71", "ResidentWetDryAMRStateV71",
    "ResidentWetDryRegridTrialV71", "ResidentWetDryTopologyBindingV71",
    "ResidentWetDryTransportTrialV71",
]
