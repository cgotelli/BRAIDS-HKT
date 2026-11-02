"""Actual-CUDA Q4-owned Q3 face/availability/Exner provider.

This is the resident peer of the bounded CPU v84 provider.  It owns a real
v67 CUDA constitutive call, the pure v74 CUDA route bridge, one H4 slope/face
composition, one finite-inventory projection, and one conservative Exner
candidate.  It never starts v81/v78, never commits a child Exner state, and
never falls back to a CPU numerical operator.  The outer v93 owner remains the
sole authoritative bed inventory.

Until the accepted hydraulic transaction publishes boundary trace states,
this bounded provider admits closed boundaries and owner free-export only.
Equilibrium/prescribed inflow is rejected rather than synthesized from owner
cells.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np

from braids_core.sediment.bedload_boundary_live_v70 import (
    SOURCE_EQUILIBRIUM_TRACE,
    SOURCE_NONE,
    SOURCE_OWNER_FREE_EXPORT,
    SOURCE_PRESCRIBED_EXTERNAL,
)
from braids_core.sediment.direct_vector_wetdry_amr_live_v71 import (
    FACE_CLOSED,
    FACE_INTERNAL,
    FACE_OWNER_OUTLET,
)
from braids_core.sediment.q3_exner_adapter_v93 import (
    MORPHOLOGICAL_FACTOR,
    Q3ExnerReceiptV93,
    _array_digest,
    _base_topology,
    _fingerprint,
    _host_array,
    _receipt_token,
    _shared_topology_sha256,
    _v92_physical_sha256,
    _validate_outer_state,
    validate_q3_v75_exner_receipt_v93,
)
from braids_core.sediment.q3_mixed_route_exner_adapter_v93 import (
    BED_SLOPE_SCOPE,
    BOUNDARY_SCOPE,
    CHECKPOINT_SCOPE_ID,
    CPUQ3V84MixedBindingV93,
    CPUQ3V84MixedStateV93,
    FRACTION_COUNT_SCOPE,
    Q3V84MixedExnerAdapterV93Error,
    resolve_cpu_q3_v84_mixed_state_v93,
    validate_cpu_q3_v84_mixed_state_v93,
)
from braids_core.sediment.q3_producer_lifecycle_binding_v84 import (
    _digest as _v84_array_digest,
)
from braids_core.sediment.route_lifecycle_composition_v81 import (
    ROUTE_COMBINE,
    ROUTE_OFF,
    ROUTE_TOTAL,
)
from braids_core.sediment.recursive_surface_feedback_v114 import (
    provider_surface_fraction_v114,
    provider_surface_slope_fields_v114,
)
from braids_core.sediment.transported_fluid_settling_v92 import (
    TransportedFluidTrialV92,
    _validate_trial as _validate_v92_trial,
)
from braids_gpu.bedload_boundary_live_v70 import (
    ResidentBedloadBoundaryTopologyV70,
    ResidentBedloadBoundaryV70,
)
from braids_gpu.flow_bed_slope_composition import (
    DONOR_NEIGHBOUR,
    DONOR_NONE,
    DONOR_OWNER,
    DONOR_TRACE,
    ResidentFlowBedSlopeFaceComposer,
)
from braids_gpu.q3_producer_lifecycle_binding_v84 import (
    ResidentQ3BoundRouteLifecycleV84,
    ResidentQ3RouteHandoffV84,
)


SCHEMA = 1
SCOPE_ID = "braids-q4-v84-mixed-face-exner-adapter-cuda-v93-v1"
PROVIDER_KIND = "q4_owned_v84_mixed_face_no_scalar_actual_cuda"
BACKEND = "actual_cuda"
CUDA_OPTIONS = (
    "--std=c++11",
    "--fmad=false",
    "--prec-div=true",
    "--prec-sqrt=true",
    "--ftz=false",
)
DETERMINISTIC_KERNEL_NAMES = (
    "q3_v93_outgoing_cell_gather",
    "q3_v93_delta_cell_gather",
    "q3_v93_named_boundary_gather",
)


def _resident_digest(cp, *values) -> str:
    return _array_digest(*(np.ascontiguousarray(cp.asnumpy(value))
                           for value in values))


@dataclass(frozen=True)
class ResidentQ3V84MixedBindingV93:
    core: CPUQ3V84MixedBindingV93
    composer: ResidentFlowBedSlopeFaceComposer
    boundary_runtime: ResidentBedloadBoundaryV70
    boundary_topology: ResidentBedloadBoundaryTopologyV70
    owner: object
    neighbour: object
    base_role: object
    normal: object
    face_length_m: object
    cell_area_m2: object
    cell_face_offset: object
    cell_faces: object
    cell_face_sign: object
    downhill: object
    family_boundary_faces: object
    binding_sha256: str
    device_id: int

    @property
    def topology_id(self) -> str:
        return self.core.topology_id

    @property
    def boundary_names(self) -> tuple[str, ...]:
        return self.core.boundary_names


def resolve_resident_q3_v84_mixed_binding_v93(
    core: CPUQ3V84MixedBindingV93,
) -> ResidentQ3V84MixedBindingV93:
    """Mirror one exact host topology into its selected-device owner."""
    if type(core) is not CPUQ3V84MixedBindingV93:
        raise Q3V84MixedExnerAdapterV93Error(
            "v93 mixed CUDA binding requires the exact host identity")
    composer = ResidentFlowBedSlopeFaceComposer(core.bed_slope_execution)
    cp = composer.cp
    device_id = composer.device_id
    boundary = ResidentBedloadBoundaryV70(core.boundary_execution)
    if boundary.device_id != device_id:
        raise Q3V84MixedExnerAdapterV93Error(
            "v93 mixed CUDA H4 and v70 selected different devices")
    topology = boundary.prepare_topology(
        owner=core.boundary_topology.owner,
        boundary_index=core.boundary_topology.boundary_index,
        normal=core.boundary_topology.normal,
        face_length_m=core.boundary_topology.face_length_m,
        cell_area_m2=core.boundary_topology.cell_area_m2)
    arrays = tuple(cp.ascontiguousarray(cp.asarray(value, dtype=dtype))
                   for value, dtype in (
                       (core.topology.owner, cp.int32),
                       (core.topology.neighbour, cp.int32),
                       (core.topology.role, cp.int32),
                       (core.normal, cp.float64),
                       (core.face_length_m, cp.float64),
                       (core.topology.cell_area_m2, cp.float64),
                       (core.cell_face_offset, cp.int32),
                       (core.cell_faces, cp.int32),
                       (core.signs, cp.int32),
                       (core.downhill, cp.float64),
                       (core.family_boundary_faces, cp.int32)))
    token = _fingerprint({
        "scope_id": SCOPE_ID+":binding",
        "host_binding_sha256": core.binding_sha256,
        "topology_id": core.topology_id,
        "boundary_topology_sha256": topology.topology_sha256,
        "bed_slope_execution_fingerprint":
            core.bed_slope_execution["fingerprint"],
        "boundary_execution_fingerprint":
            core.boundary_execution["fingerprint"],
        "resident_arrays_sha256": _resident_digest(cp, *arrays),
        "device_id": device_id,
        "trace_scope": "closed_or_owner_free_export_only",
    })
    return ResidentQ3V84MixedBindingV93(
        core, composer, boundary, topology, *arrays, token, device_id)


@dataclass(frozen=True)
class ResidentQ3V84MixedStateV93:
    core: CPUQ3V84MixedStateV93
    bed_mass_kg_m2: object
    active_fraction: object
    accepted_face_flux_kg_m_s: object
    accepted_cell_availability: object
    state_sha256: str
    device_id: int

    @property
    def authoritative_bed_mass_kg_m2(self) -> np.ndarray:
        """Canonical immutable host evidence for the outer Exner owner."""
        return self.core.authoritative_bed_mass_kg_m2


def _resident_state_token(
    value: ResidentQ3V84MixedStateV93,
    binding: ResidentQ3V84MixedBindingV93,
) -> str:
    return _fingerprint({
        "scope_id": SCOPE_ID+":state",
        "host_state_sha256": value.core.state_sha256,
        "binding_sha256": binding.binding_sha256,
        "resident_arrays_sha256": _resident_digest(
            binding.composer.cp, value.bed_mass_kg_m2,
            value.active_fraction, value.accepted_face_flux_kg_m_s,
            value.accepted_cell_availability),
        "device_id": value.device_id,
    })


def resolve_resident_q3_v84_mixed_state_v93(
    *, execution_fingerprint: str,
    binding: ResidentQ3V84MixedBindingV93,
    authoritative_bed_mass_kg_m2: object,
    bulk_density_kg_m3: object,
    retained_fraction: object | None = None,
    accepted_face_flux_kg_m_s: object | None = None,
    accepted_cell_availability: object | None = None,
    named_boundary_positive_in_kg: object | None = None,
    named_boundary_absolute_kg: object | None = None,
    depletion_roundoff_positive_in_kg: object | None = None,
    accepted_updates: int = 0, time_s: float = 0.0,
) -> ResidentQ3V84MixedStateV93:
    # Build immutable host evidence first, but do not use its derived mixture
    # as the live numerical input.  The accepted CUDA path independently
    # derives positive-stock fractions on device and uses the supplied prior
    # identity only in depleted cells.
    core = resolve_cpu_q3_v84_mixed_state_v93(
        execution_fingerprint=execution_fingerprint,
        binding=binding.core,
        authoritative_bed_mass_kg_m2=authoritative_bed_mass_kg_m2,
        bulk_density_kg_m3=bulk_density_kg_m3,
        retained_fraction=retained_fraction,
        accepted_face_flux_kg_m_s=accepted_face_flux_kg_m_s,
        accepted_cell_availability=accepted_cell_availability,
        named_boundary_positive_in_kg=named_boundary_positive_in_kg,
        named_boundary_absolute_kg=named_boundary_absolute_kg,
        depletion_roundoff_positive_in_kg=
            depletion_roundoff_positive_in_kg,
        accepted_updates=accepted_updates, time_s=time_s)
    cp = binding.composer.cp
    bed = cp.ascontiguousarray(cp.asarray(
        core.authoritative_bed_mass_kg_m2, dtype=cp.float64))
    bulk = cp.ascontiguousarray(cp.asarray(
        core.bulk_density_kg_m3, dtype=cp.float64))
    if retained_fraction is None:
        retained_host = np.zeros_like(core.authoritative_bed_mass_kg_m2)
    else:
        retained_host = _host_array(
            retained_fraction, core.active_fraction.shape,
            "CUDA retained active fraction", nonnegative=True)
    retained = cp.ascontiguousarray(cp.asarray(
        retained_host, dtype=cp.float64))
    volume = bed/bulk[:, None]
    total = cp.sum(volume, axis=0)
    positive = total > 0.0
    safe_total = cp.where(positive, total, 1.0)
    fraction = cp.ascontiguousarray(cp.where(
        positive[None, :], volume/safe_total[None, :], retained))
    tolerance = 512.0*np.finfo(np.float64).eps
    if not np.allclose(
            cp.asnumpy(fraction), core.active_fraction,
            rtol=tolerance, atol=tolerance):
        raise Q3V84MixedExnerAdapterV93Error(
            "v93 mixed CUDA surface-mixture rebase differs from CPU oracle")
    flux = cp.ascontiguousarray(cp.asarray(
        core.accepted_face_flux_kg_m_s, dtype=cp.float64))
    availability = cp.ascontiguousarray(cp.asarray(
        core.accepted_cell_availability, dtype=cp.float64))
    arrays = (bed, fraction, flux, availability)
    placeholder = ResidentQ3V84MixedStateV93(
        core, *arrays, "", binding.device_id)
    return ResidentQ3V84MixedStateV93(
        core, *arrays, _resident_state_token(placeholder, binding),
        binding.device_id)


def validate_resident_q3_v84_mixed_state_v93(
    value: ResidentQ3V84MixedStateV93, *, execution_fingerprint: str,
    binding: ResidentQ3V84MixedBindingV93,
) -> None:
    if (not isinstance(value, ResidentQ3V84MixedStateV93) or
            value.device_id != binding.device_id or
            value.state_sha256 != _resident_state_token(value, binding)):
        raise Q3V84MixedExnerAdapterV93Error(
            "v93 mixed CUDA state is stale or on another device")
    validate_cpu_q3_v84_mixed_state_v93(
        value.core, execution_fingerprint=execution_fingerprint,
        binding=binding.core)
    cp = binding.composer.cp
    exact_pairs = (
        (value.bed_mass_kg_m2,
         value.core.authoritative_bed_mass_kg_m2),
        (value.accepted_face_flux_kg_m_s,
         value.core.accepted_face_flux_kg_m_s),
        (value.accepted_cell_availability,
         value.core.accepted_cell_availability),
    )
    if any(not bool(cp.array_equal(device, cp.asarray(host)).item())
           for device, host in exact_pairs):
        raise Q3V84MixedExnerAdapterV93Error(
            "v93 mixed CUDA resident state differs from host evidence")
    tolerance = 512.0*np.finfo(np.float64).eps
    if not np.allclose(
            cp.asnumpy(value.active_fraction), value.core.active_fraction,
            rtol=tolerance, atol=tolerance):
        raise Q3V84MixedExnerAdapterV93Error(
            "v93 mixed CUDA resident mixture differs from host oracle")
    bulk = cp.asarray(value.core.bulk_density_kg_m3, dtype=cp.float64)
    volume = value.bed_mass_kg_m2/bulk[:, None]
    total = cp.sum(volume, axis=0)
    positive = total > 0.0
    derived = volume/cp.where(positive, total, 1.0)[None, :]
    if bool(cp.any(cp.abs(
            value.active_fraction[:, positive]-derived[:, positive]) >
            tolerance).item()):
        raise Q3V84MixedExnerAdapterV93Error(
            "v93 mixed CUDA positive-stock mixture was not device-derived")


@dataclass(frozen=True)
class ResidentQ3V84FaceTrialV93:
    parent_state_sha256: str
    route: ResidentQ3RouteHandoffV84
    accepted_endpoint_sha256: str
    hydraulic_face_receipt_sha256: str
    dt_s: float
    morphological_factor: float
    face_flux_kg_m_s: object
    cell_delta_kg_m2: object
    inventory_fill: object
    accepted_cell_availability: object
    named_boundary_transfer_kg: object
    named_boundary_absolute_kg: object
    depletion_roundoff_positive_in_kg: object
    conservation_residual_kg: object
    trial_sha256: str
    device_id: int


@dataclass(frozen=True)
class ResidentQ3V84OpenTrialV93:
    parent_state: ResidentQ3V84MixedStateV93
    face_trial: ResidentQ3V84FaceTrialV93
    receipt: Q3ExnerReceiptV93
    handle_sha256: str


@dataclass(frozen=True)
class ResidentQ3V84PreparedCommitV93:
    open_trial: ResidentQ3V84OpenTrialV93
    rebound_state: ResidentQ3V84MixedStateV93
    authoritative_bed_sha256: str
    preparation_sha256: str

    @property
    def rebound_state_sha256(self) -> str:
        return self.rebound_state.state_sha256

    @property
    def rebound_bed_mass_kg_m2(self) -> np.ndarray:
        return self.rebound_state.core.authoritative_bed_mass_kg_m2


def _face_token(value: ResidentQ3V84FaceTrialV93, cp) -> str:
    return _fingerprint({
        "scope_id": SCOPE_ID+":face-trial",
        "parent_state_sha256": value.parent_state_sha256,
        "route_sha256": value.route.resident_sha256,
        "accepted_endpoint_sha256": value.accepted_endpoint_sha256,
        "hydraulic_face_receipt_sha256":
            value.hydraulic_face_receipt_sha256,
        "dt_s": value.dt_s,
        "morphological_factor": value.morphological_factor,
        "resident_arrays_sha256": _resident_digest(
            cp, value.face_flux_kg_m_s, value.cell_delta_kg_m2,
            value.inventory_fill, value.accepted_cell_availability,
            value.named_boundary_transfer_kg,
            value.named_boundary_absolute_kg,
            value.depletion_roundoff_positive_in_kg,
            value.conservation_residual_kg),
        "constitutive_producer_count": "active",
        "route_handoff_count": "active",
        "face_exner_count": "active",
        "extra_scalar_lifecycle_count": "zero",
        "child_exner_commit": False,
        "device_id": value.device_id,
    })


def _open_token(value: ResidentQ3V84OpenTrialV93) -> str:
    return _fingerprint({
        "scope_id": SCOPE_ID+":open",
        "parent": value.parent_state.state_sha256,
        "face": value.face_trial.trial_sha256,
        "receipt": value.receipt.receipt_sha256,
    })


def _prepared_token(value: ResidentQ3V84PreparedCommitV93) -> str:
    return _fingerprint({
        "scope_id": SCOPE_ID+":prepared",
        "open": value.open_trial.handle_sha256,
        "rebound": value.rebound_state.state_sha256,
        "authoritative_bed": value.authoritative_bed_sha256,
    })


class ResidentQ3V84MixedExnerAdapterV93:
    """Identity-held actual-CUDA mixed-route Q3 provider."""

    backend = BACKEND
    provider_kind = PROVIDER_KIND
    bed_slope_scope = BED_SLOPE_SCOPE
    boundary_scope = BOUNDARY_SCOPE
    fraction_count_scope = FRACTION_COUNT_SCOPE

    def __init__(
        self, *, outer_execution,
        runtime: ResidentQ3BoundRouteLifecycleV84,
        binding: ResidentQ3V84MixedBindingV93,
        state: ResidentQ3V84MixedStateV93,
    ):
        from braids_core.sediment.suspended_exner_owner_v93 import (
            execution_document_v93,
        )
        execution_document_v93(outer_execution)
        if (outer_execution.backend != "implicit_fwave_midpoint_gpu" or
                outer_execution.fraction_count != FRACTION_COUNT_SCOPE or
                type(runtime) is not ResidentQ3BoundRouteLifecycleV84 or
                type(binding) is not ResidentQ3V84MixedBindingV93):
            raise Q3V84MixedExnerAdapterV93Error(
                "v93 mixed CUDA adapter requires the exact three-fraction "
                "resident objects")
        if (runtime.device_id != binding.device_id or
                runtime.execution["fingerprint"] !=
                    outer_execution.q3_execution_fingerprint or
                binding.topology_id != outer_execution.q3_topology_id or
                binding.binding_sha256 != outer_execution.q3_binding_sha256 or
                binding.boundary_names !=
                    tuple(outer_execution.q3_boundary_names)):
            raise Q3V84MixedExnerAdapterV93Error(
                "v93 mixed CUDA execution, device, mesh, or names differ")
        routes = tuple(runtime.lifecycle_execution.route_execution["routes"])
        if routes != tuple(np.asarray(outer_execution.routes).tolist()):
            raise Q3V84MixedExnerAdapterV93Error(
                "v93 mixed CUDA v74 and outer routes differ")
        physical = runtime.producer.execution["physical_profile"]
        ids = tuple(physical["fraction_ids"])
        if (ids != tuple(outer_execution.ordered_fraction_ids) or
                tuple(binding.core.bed_slope_execution[
                    "ordered_fraction_ids"]) != ids or
                not np.array_equal(
                    state.core.bulk_density_kg_m3,
                    np.asarray(binding.core.boundary_execution[
                        "bulk_density_kg_m3"], dtype=np.float64))):
            raise Q3V84MixedExnerAdapterV93Error(
                "v93 mixed CUDA fraction or bulk-density identity differs")
        outer_v78 = (outer_execution.base_execution.base_execution.
                     base_execution)
        if runtime.lifecycle_execution.lifecycle_execution.fingerprint != \
                outer_v78.fingerprint:
            raise Q3V84MixedExnerAdapterV93Error(
                "v93 mixed CUDA v84/v92 suspension identities differ")
        self.outer_execution = outer_execution
        self.runtime = runtime
        self.binding = binding
        self.cp = runtime.cp
        self.device_id = runtime.device_id
        self.boundary_names = binding.boundary_names
        source_path = Path(__file__).with_suffix(".cu").resolve()
        source = source_path.read_text(encoding="utf-8")
        with self.cp.cuda.Device(self.device_id):
            module = self.cp.RawModule(
                code=source, options=CUDA_OPTIONS,
                name_expressions=DETERMINISTIC_KERNEL_NAMES)
            self._deterministic_kernels = {
                name: module.get_function(name)
                for name in DETERMINISTIC_KERNEL_NAMES
            }
        self.deterministic_source_sha256 = hashlib.sha256(
            source.encode("utf-8")).hexdigest()
        self._deterministic_kernel_launch_counts = {
            name: 0 for name in DETERMINISTIC_KERNEL_NAMES
        }
        self._validate_topology_and_profile()
        validate_resident_q3_v84_mixed_state_v93(
            state, execution_fingerprint=self.execution_fingerprint,
            binding=binding)
        self._state = state
        self._open: ResidentQ3V84OpenTrialV93 | None = None
        self._prepared: ResidentQ3V84PreparedCommitV93 | None = None
        self._closed: ResidentQ3V84PreparedCommitV93 | None = None

    @property
    def execution_fingerprint(self) -> str:
        return self.runtime.execution["fingerprint"]

    @property
    def binding_sha256(self) -> str:
        return self.binding.binding_sha256

    @property
    def topology_id(self) -> str:
        return self.binding.topology_id

    @property
    def ordered_fraction_ids(self) -> tuple[str, ...]:
        return tuple(self.outer_execution.ordered_fraction_ids)

    @property
    def state_sha256(self) -> str:
        return self._state.state_sha256

    @property
    def accepted_updates(self) -> int:
        return self._state.core.accepted_updates

    @property
    def time_s(self) -> float:
        return self._state.core.time_s

    @property
    def authoritative_bed_mass_kg_m2(self) -> np.ndarray:
        return self._state.core.authoritative_bed_mass_kg_m2

    @property
    def deterministic_kernel_launch_counts(self) -> dict[str, int]:
        """Measured launches owned by this provider, never estimates."""
        return dict(self._deterministic_kernel_launch_counts)

    @property
    def state(self) -> ResidentQ3V84MixedStateV93:
        return self._state

    def _validate_topology_and_profile(self) -> None:
        base = _base_topology(self.outer_execution)
        core = self.binding.core
        mapping = np.asarray(core.family_boundary_faces)
        local = np.asarray(core.boundary_topology.boundary_index)
        expected = np.flatnonzero(base.boundary_index >= 0)
        checks = (
            core.topology.ncell == base.cell_count,
            core.topology.nface == base.face_count,
            np.array_equal(core.topology.owner, base.owner),
            np.array_equal(core.topology.neighbour, base.neighbour),
            np.array_equal(core.topology.cell_area_m2, base.cell_area_m2),
            np.array_equal(core.normal, base.normal),
            np.array_equal(core.face_length_m, base.face_length_m),
            np.array_equal(core.cell_face_offset, base.cell_face_offset),
            np.array_equal(core.cell_faces, base.cell_faces),
            np.array_equal(core.signs, base.cell_face_sign),
            base.boundary_count == len(self.boundary_names),
            np.array_equal(mapping, expected),
            np.array_equal(local, base.boundary_index[mapping]),
            bool(np.all(core.topology.wet)),
        )
        if not all(checks):
            raise Q3V84MixedExnerAdapterV93Error(
                "v93 mixed CUDA topology/boundary map differs from v92")
        physical = self.runtime.producer.execution["physical_profile"]
        profile = base.profile
        if (tuple(physical["fraction_ids"]) != self.ordered_fraction_ids or
                not np.array_equal(np.asarray(physical["d50_m"]),
                                   np.asarray(profile.sediment_d50_m)) or
                not np.array_equal(
                    np.asarray(physical["solid_density_kg_m3"]),
                    np.asarray(profile.sediment_density_kg_m3)) or
                float(physical["water_density_kg_m3"]) !=
                    float(profile.water_density_kg_m3) or
                float(physical["gravity_m_s2"]) !=
                    float(profile.gravity_m_s2)):
            raise Q3V84MixedExnerAdapterV93Error(
                "v93 mixed CUDA physical profile differs from v92")

    def validate_current_bed_clock(self, outer_parent) -> None:
        _validate_outer_state(outer_parent, self.outer_execution)
        validate_resident_q3_v84_mixed_state_v93(
            self._state, execution_fingerprint=self.execution_fingerprint,
            binding=self.binding)
        if (not np.array_equal(self.authoritative_bed_mass_kg_m2,
                               outer_parent.bed_mass_kg_m2) or
                self.accepted_updates != outer_parent.accepted_updates or
                self.time_s != outer_parent.time_s):
            raise Q3V84MixedExnerAdapterV93Error(
                "v93 mixed CUDA provider is not on the outer bed/clock")

    def _dynamic_roles(self, *, hydraulic_boundary, cell_vector):
        cp = self.cp
        mapping = self.binding.family_boundary_faces
        trace = cp.zeros(
            (2, self.outer_execution.fraction_count,
             self.binding.core.topology.nface), dtype=cp.float64)
        candidate = self.binding.boundary_runtime.candidate(
            self.binding.boundary_topology,
            hydraulic_outward_flux=hydraulic_boundary,
            owner_bed_mass_vector=cell_vector,
            trace_bed_mass_vector=cp.ascontiguousarray(trace[:, :, mapping]))
        sources = cp.asnumpy(candidate.source)
        role = self.binding.base_role.copy()
        for local, family in enumerate(cp.asnumpy(mapping).tolist()):
            selected = set(sources[:, local].tolist())
            if (SOURCE_PRESCRIBED_EXTERNAL in selected or
                    SOURCE_EQUILIBRIUM_TRACE in selected):
                raise Q3V84MixedExnerAdapterV93Error(
                    "v93 mixed CUDA inlet/external trace is not authenticated")
            if selected == {SOURCE_OWNER_FREE_EXPORT}:
                role[family] = FACE_OWNER_OUTLET
            elif selected == {SOURCE_NONE}:
                role[family] = FACE_CLOSED
            else:
                raise Q3V84MixedExnerAdapterV93Error(
                    "v93 mixed CUDA boundary classes chose different owners")
        return cp.ascontiguousarray(role), trace

    def _face_trial(self, *, route: ResidentQ3RouteHandoffV84,
                    v92_trial: TransportedFluidTrialV92):
        cp = self.cp
        ng = self.outer_execution.fraction_count
        nc = self.outer_execution.cell_count
        nf = self.binding.core.topology.nface
        active = np.asarray(self.outer_execution.routes) != ROUTE_OFF
        handoff = route.receipt.handoff
        if (not np.array_equal(handoff.active_fraction_application_count,
                               active.astype(np.int32)) or
                not np.array_equal(
                    handoff.bed_suspended_owner_count,
                    (np.asarray(self.outer_execution.routes) == ROUTE_TOTAL).
                    astype(np.int32)) or
                not np.array_equal(
                    handoff.water_suspended_owner_count,
                    (np.asarray(self.outer_execution.routes) == ROUTE_COMBINE).
                    astype(np.int32))):
            raise Q3V84MixedExnerAdapterV93Error(
                "v93 mixed CUDA route ownership differs")
        cell_vector = cp.ascontiguousarray(cp.stack((
            route.resident.bed_capacity_x,
            route.resident.bed_capacity_y), axis=0))
        inactive = cp.asarray(~active)
        if bool(cp.any(cell_vector[:, inactive] != 0.0).item()):
            raise Q3V84MixedExnerAdapterV93Error(
                "v93 mixed CUDA OFF route acquired capacity")
        q = cp.ascontiguousarray(cp.asarray(
            v92_trial.candidate.q, dtype=cp.float64))
        water = cp.ascontiguousarray(cp.asarray(
            v92_trial.hydraulic_face_receipt.accepted_water_flux_m2_s,
            dtype=cp.float64))
        mapping = self.binding.family_boundary_faces
        role, trace = self._dynamic_roles(
            hydraulic_boundary=cp.ascontiguousarray(water[mapping]),
            cell_vector=cell_vector)
        owner = self.binding.owner
        neighbour = self.binding.neighbour
        safe_other = cp.clip(neighbour, 0, nc-1)
        left = cp.ascontiguousarray(cell_vector[:, :, owner])
        right = cp.ascontiguousarray(cell_vector[:, :, safe_other])
        zero_stress = cp.zeros((ng, nf), dtype=cp.float64)
        neutral = cp.ones((ng, nf), dtype=cp.float64)
        hiding, surface_mean = provider_surface_slope_fields_v114(
            self, self.binding.core.topology, cp)
        d50 = cp.ascontiguousarray(cp.asarray(
            _base_topology(self.outer_execution).profile.sediment_d50_m,
            dtype=cp.float64))
        common = dict(
            left_bed=left, right_bed=right, owner_bed=left,
            trace_bed=trace, owner=owner, neighbour=neighbour, role=role,
            normal=self.binding.normal, downhill=self.binding.downhill,
            stress=zero_stress, depth=neutral, d50=d50,
            mean_diameter=surface_mean, hiding=hiding)
        ones = cp.ones((ng, nc), dtype=cp.float64)
        provisional = self.binding.composer.compose(
            availability=ones, **common)
        dt = float(v92_trial.hydraulic_receipt.dt_s)
        outgoing = cp.zeros((ng, nc), dtype=cp.float64)
        item_count = ng*nc
        threads = 128
        kernel_name = "q3_v93_outgoing_cell_gather"
        self._deterministic_kernels[kernel_name](
            ((item_count+threads-1)//threads,), (threads,), (
                np.int32(ng), np.int32(nc), np.int32(nf),
                self.binding.cell_face_offset,
                self.binding.cell_faces,
                provisional.donor, owner, neighbour, role,
                provisional.flux, self.binding.face_length_m,
                np.float64(dt), self.binding.cell_area_m2,
                np.int32(DONOR_OWNER), np.int32(DONOR_NEIGHBOUR),
                np.int32(DONOR_TRACE), np.int32(FACE_INTERNAL),
                np.int32(FACE_OWNER_OUTLET), outgoing))
        self._deterministic_kernel_launch_counts[kernel_name] += 1
        fill = cp.where(self._state.bed_mass_kg_m2 > 0.0, 1.0, 0.0)
        fill = cp.where(
            outgoing > 0.0,
            cp.minimum(1.0, self._state.bed_mass_kg_m2 /
                       cp.where(outgoing > 0.0, outgoing, 1.0)), fill)
        fill = cp.ascontiguousarray(fill)
        accepted = self.binding.composer.compose(
            availability=fill, **common)
        flux = cp.ascontiguousarray(accepted.flux)
        amount = flux*self.binding.face_length_m[None, :]*dt
        delta = cp.zeros((ng, nc), dtype=cp.float64)
        kernel_name = "q3_v93_delta_cell_gather"
        self._deterministic_kernels[kernel_name](
            ((item_count+threads-1)//threads,), (threads,), (
                np.int32(ng), np.int32(nc), np.int32(nf),
                self.binding.cell_face_offset,
                self.binding.cell_faces,
                self.binding.cell_face_sign,
                amount, self.binding.cell_area_m2, delta))
        self._deterministic_kernel_launch_counts[kernel_name] += 1
        nb = len(self.boundary_names)
        named = cp.zeros((ng, nb), dtype=cp.float64)
        absolute = cp.zeros_like(named)
        boundary_items = ng*nb
        if boundary_items:
            kernel_name = "q3_v93_named_boundary_gather"
            self._deterministic_kernels[kernel_name](
                ((boundary_items+threads-1)//threads,), (threads,), (
                    np.int32(ng), np.int32(nb),
                    np.int32(self.binding.boundary_topology.nface),
                    np.int32(nf), mapping,
                    self.binding.boundary_topology.boundary_index,
                    amount, named, absolute))
            self._deterministic_kernel_launch_counts[kernel_name] += 1
        candidate = self._state.bed_mass_kg_m2+delta
        tolerance = (4096.0*np.finfo(np.float64).eps *
                     cp.maximum(1.0, cp.abs(self._state.bed_mass_kg_m2)))
        if bool(cp.any(candidate < -tolerance).item()):
            raise Q3V84MixedExnerAdapterV93Error(
                "v93 mixed CUDA inventory limiter produced negative bed")
        before = delta.copy()
        negative = candidate < 0.0
        delta = cp.where(negative, -self._state.bed_mass_kg_m2, delta)
        delta = cp.ascontiguousarray(delta)
        depletion = cp.ascontiguousarray(cp.sum(
            (delta-before)*self.binding.cell_area_m2[None, :], axis=1))
        residual = cp.ascontiguousarray(
            cp.sum(delta*self.binding.cell_area_m2[None, :], axis=1)+
            cp.sum(named, axis=1)-depletion)
        scale = cp.maximum(
            1.0, cp.sum(cp.abs(delta*self.binding.cell_area_m2[None, :]),
                        axis=1)+cp.sum(absolute, axis=1)+depletion)
        if bool(cp.any(cp.abs(residual) >
                       8192.0*np.finfo(np.float64).eps*scale).item()):
            raise Q3V84MixedExnerAdapterV93Error(
                "v93 mixed CUDA face/Exner ledger does not close")
        endpoint = _v84_array_digest(
            np.ascontiguousarray(v92_trial.candidate.q[0]),
            np.ascontiguousarray(v92_trial.candidate.q[1:]))
        placeholder = ResidentQ3V84FaceTrialV93(
            self._state.state_sha256, route, endpoint,
            v92_trial.hydraulic_face_receipt.receipt_sha256,
            dt, MORPHOLOGICAL_FACTOR, flux, delta, fill, fill,
            named, absolute, depletion, residual, "", self.device_id)
        return ResidentQ3V84FaceTrialV93(
            **{**placeholder.__dict__,
               "trial_sha256": _face_token(placeholder, cp)})

    def _validate_face_trial(
        self, value: ResidentQ3V84FaceTrialV93,
    ) -> None:
        if (not isinstance(value, ResidentQ3V84FaceTrialV93) or
                value.device_id != self.device_id or
                value.parent_state_sha256 != self._state.state_sha256 or
                value.dt_s <= 0.0 or
                value.morphological_factor != MORPHOLOGICAL_FACTOR or
                value.trial_sha256 != _face_token(value, self.cp) or
                value.route.receipt.scalar_lifecycle_application_count != 0 or
                value.route.receipt.accepted_endpoint_sha256 !=
                    value.accepted_endpoint_sha256 or
                value.route.receipt.active_fraction_sha256 !=
                    _v84_array_digest(self.cp.asnumpy(
                        provider_surface_fraction_v114(self)))):
            raise Q3V84MixedExnerAdapterV93Error(
                "v93 mixed CUDA face trial is stale or cross-device")
        self.runtime.validate_route_handoff(
            value.route,
            accepted_endpoint_sha256=value.accepted_endpoint_sha256,
            active_fraction_sha256=
                _v84_array_digest(self.cp.asnumpy(
                    provider_surface_fraction_v114(self))))

    def _make_receipt(self, *, outer_parent, v92_trial,
                      trial: ResidentQ3V84FaceTrialV93):
        cp = self.cp
        ng = self.outer_execution.fraction_count
        active = np.asarray(self.outer_execution.routes) != ROUTE_OFF
        delta = _host_array(
            cp.asnumpy(trial.cell_delta_kg_m2),
            (ng, self.outer_execution.cell_count), "CUDA bed delta")
        positive = _host_array(
            -cp.asnumpy(trial.named_boundary_transfer_kg),
            (ng, len(self.boundary_names)), "CUDA named positive-in")
        absolute = _host_array(
            cp.asnumpy(trial.named_boundary_absolute_kg), positive.shape,
            "CUDA named absolute", nonnegative=True)
        depletion = _host_array(
            cp.asnumpy(trial.depletion_roundoff_positive_in_kg), (ng,),
            "CUDA depletion", nonnegative=True)
        boundary = _host_array(
            np.sum(positive, axis=1)+depletion, (ng,),
            "CUDA global boundary")
        routes = _host_array(
            self.outer_execution.routes, (ng,), "CUDA routes", integer=True)
        owner = _host_array(
            active.astype(np.int32), (ng,), "CUDA owner count", integer=True)
        zero = _host_array(
            np.zeros(ng, dtype=np.int32), (ng,),
            "CUDA extra scalar count", integer=True)
        placeholder = Q3ExnerReceiptV93(
            self.outer_execution.fingerprint, outer_parent.state_sha256,
            v92_trial.trial_sha256,
            v92_trial.hydraulic_receipt.receipt_sha256,
            v92_trial.hydraulic_face_receipt.receipt_sha256,
            self.execution_fingerprint, self.binding.binding_sha256,
            self.binding.topology_id, self._state.state_sha256,
            trial.trial_sha256, trial.trial_sha256,
            _resident_digest(cp, trial.face_flux_kg_m_s),
            _shared_topology_sha256(_base_topology(self.outer_execution)),
            self.ordered_fraction_ids,
            _v92_physical_sha256(self.outer_execution),
            trial.dt_s, MORPHOLOGICAL_FACTOR, self.boundary_names,
            routes, delta, positive, absolute, depletion, boundary, owner,
            self.backend, self.provider_kind, owner, owner, zero, "")
        return Q3ExnerReceiptV93(
            **{**placeholder.__dict__,
               "receipt_sha256": _receipt_token(placeholder)})

    def begin(
        self, *, outer_parent, parent_hles, handoff,
        v92_trial: TransportedFluidTrialV92,
        roughness_parameter, kinematic_viscosity_m2_s,
        settling_velocity_m_s,
        cancelled: Callable[[], bool] = lambda: False,
    ) -> ResidentQ3V84OpenTrialV93:
        if not callable(cancelled):
            raise TypeError("cancelled must be callable")
        if self._open is not None or self._prepared is not None or \
                self._closed is not None:
            raise RuntimeError("v93 mixed CUDA provider already has a transaction")
        _validate_outer_state(outer_parent, self.outer_execution)
        _validate_v92_trial(
            self.outer_execution.base_execution, outer_parent.base_state,
            parent_hles, handoff, v92_trial)
        if v92_trial.backend != BACKEND:
            raise Q3V84MixedExnerAdapterV93Error(
                "v93 mixed CUDA provider requires the actual-CUDA v92 trial")
        self.validate_current_bed_clock(outer_parent)
        if cancelled():
            raise InterruptedError("v93 mixed CUDA cancelled before Q3")
        cp = self.cp
        q = cp.ascontiguousarray(cp.asarray(
            v92_trial.candidate.q, dtype=cp.float64))
        roughness = self.runtime._device(
            roughness_parameter, (self.outer_execution.cell_count,),
            "adapter Manning roughness")
        viscosity = self.runtime._device(
            kinematic_viscosity_m2_s,
            (self.outer_execution.cell_count,), "adapter viscosity")
        settling = self.runtime._device(
            settling_velocity_m_s,
            (self.outer_execution.fraction_count,
             self.outer_execution.cell_count), "adapter settling velocity")
        try:
            surface_fraction = provider_surface_fraction_v114(self)
            route = ResidentQ3BoundRouteLifecycleV84.produce_route_handoff(
                self.runtime, depth_after_m=q[0],
                endpoint_momentum_m2_s=q[1:],
                roughness_parameter=roughness,
                kinematic_viscosity_m2_s=viscosity,
                settling_velocity_m_s=settling,
                active_fraction=surface_fraction)
            endpoint = _v84_array_digest(
                np.ascontiguousarray(v92_trial.candidate.q[0]),
                np.ascontiguousarray(v92_trial.candidate.q[1:]))
            fraction = _v84_array_digest(
                self.cp.asnumpy(surface_fraction))
            ResidentQ3BoundRouteLifecycleV84.validate_route_handoff(
                self.runtime, route, accepted_endpoint_sha256=endpoint,
                active_fraction_sha256=fraction)
            face = self._face_trial(route=route, v92_trial=v92_trial)
            self._validate_face_trial(face)
            receipt = self._make_receipt(
                outer_parent=outer_parent, v92_trial=v92_trial,
                trial=face)
            handle = ResidentQ3V84OpenTrialV93(
                self._state, face, receipt, "")
            handle = ResidentQ3V84OpenTrialV93(
                handle.parent_state, handle.face_trial, handle.receipt,
                _open_token(handle))
            if cancelled():
                raise InterruptedError(
                    "v93 mixed CUDA cancelled before publication")
            self._open = handle
            return handle
        except Exception:
            self._open = None
            self._prepared = None
            raise

    def _validate_open(self, value, *, outer_parent=None,
                       v92_trial=None) -> None:
        if (not isinstance(value, ResidentQ3V84OpenTrialV93) or
                value is not self._open or
                value.parent_state is not self._state or
                value.handle_sha256 != _open_token(value)):
            raise Q3V84MixedExnerAdapterV93Error(
                "v93 mixed CUDA open handle differs")
        self._validate_face_trial(value.face_trial)
        if outer_parent is not None and v92_trial is not None:
            expected_endpoint = _v84_array_digest(
                np.ascontiguousarray(v92_trial.candidate.q[0]),
                np.ascontiguousarray(v92_trial.candidate.q[1:]))
            if value.face_trial.accepted_endpoint_sha256 != expected_endpoint:
                raise Q3V84MixedExnerAdapterV93Error(
                    "v93 mixed CUDA endpoint differs from v92")
            validate_q3_v75_exner_receipt_v93(
                value.receipt, outer_execution=self.outer_execution,
                outer_parent=outer_parent, v92_trial=v92_trial,
                q3_binding_sha256=self.binding.binding_sha256,
                q3_topology_id=self.binding.topology_id,
                backend=self.backend)
            expected = self._make_receipt(
                outer_parent=outer_parent, v92_trial=v92_trial,
                trial=value.face_trial)
            if expected.receipt_sha256 != value.receipt.receipt_sha256:
                raise Q3V84MixedExnerAdapterV93Error(
                    "v93 mixed CUDA receipt differs from live resident trial")

    def validate_open(self, value, *, outer_parent, v92_trial):
        self._validate_open(
            value, outer_parent=outer_parent, v92_trial=v92_trial)
        return value.receipt

    def prepare_rebound(
        self, value: ResidentQ3V84OpenTrialV93, *,
        authoritative_bed_mass_kg_m2: object,
    ) -> ResidentQ3V84PreparedCommitV93:
        self._validate_open(value)
        if self._prepared is not None:
            raise RuntimeError("v93 mixed CUDA rebound is already prepared")
        bed = _host_array(
            authoritative_bed_mass_kg_m2,
            (self.outer_execution.fraction_count,
             self.outer_execution.cell_count), "CUDA authoritative bed",
            nonnegative=True)
        face = value.face_trial
        rebound = resolve_resident_q3_v84_mixed_state_v93(
            execution_fingerprint=self.execution_fingerprint,
            binding=self.binding, authoritative_bed_mass_kg_m2=bed,
            bulk_density_kg_m3=self._state.core.bulk_density_kg_m3,
            retained_fraction=self.cp.asnumpy(
                provider_surface_fraction_v114(self)),
            accepted_face_flux_kg_m_s=self.cp.asnumpy(
                face.face_flux_kg_m_s),
            accepted_cell_availability=self.cp.asnumpy(
                face.accepted_cell_availability),
            named_boundary_positive_in_kg=(
                self._state.core.named_boundary_positive_in_kg+
                value.receipt.named_boundary_positive_in_kg),
            named_boundary_absolute_kg=(
                self._state.core.named_boundary_absolute_kg+
                value.receipt.named_boundary_absolute_kg),
            depletion_roundoff_positive_in_kg=(
                self._state.core.depletion_roundoff_positive_in_kg+
                value.receipt.depletion_roundoff_positive_in_kg),
            accepted_updates=self.accepted_updates+1,
            time_s=self.time_s+face.dt_s)
        prepared = ResidentQ3V84PreparedCommitV93(
            value, rebound, _array_digest(bed), "")
        prepared = ResidentQ3V84PreparedCommitV93(
            prepared.open_trial, prepared.rebound_state,
            prepared.authoritative_bed_sha256, _prepared_token(prepared))
        self._prepared = prepared
        return prepared

    def validate_prepared(self, value, *, authoritative_bed_mass_kg_m2):
        if (not isinstance(value, ResidentQ3V84PreparedCommitV93) or
                value is not self._prepared or
                value.preparation_sha256 != _prepared_token(value)):
            raise Q3V84MixedExnerAdapterV93Error(
                "v93 mixed CUDA prepared commit differs")
        bed = _host_array(
            authoritative_bed_mass_kg_m2,
            (self.outer_execution.fraction_count,
             self.outer_execution.cell_count), "CUDA authoritative bed",
            nonnegative=True)
        if (value.authoritative_bed_sha256 != _array_digest(bed) or
                not np.array_equal(
                    value.rebound_state.core.authoritative_bed_mass_kg_m2,
                    bed)):
            raise Q3V84MixedExnerAdapterV93Error(
                "v93 mixed CUDA prepared bed differs")
        validate_resident_q3_v84_mixed_state_v93(
            value.rebound_state,
            execution_fingerprint=self.execution_fingerprint,
            binding=self.binding)
        return value.rebound_state

    def close_commit(self, value):
        if value is not self._prepared:
            raise Q3V84MixedExnerAdapterV93Error(
                "v93 mixed CUDA prepared commit differs")
        self._validate_open(value.open_trial)
        validate_resident_q3_v84_mixed_state_v93(
            value.rebound_state,
            execution_fingerprint=self.execution_fingerprint,
            binding=self.binding)
        self._open = None
        self._prepared = None
        self._closed = value
        return value.rebound_state

    def publish_commit(self, value):
        if value is not self._closed:
            raise Q3V84MixedExnerAdapterV93Error(
                "v93 mixed CUDA closed commit differs")
        self._state = value.rebound_state
        self._closed = None
        return self._state

    def abort_closed_commit(self, value) -> None:
        if value is not self._closed:
            raise Q3V84MixedExnerAdapterV93Error(
                "v93 mixed CUDA closed commit differs")
        self._closed = None

    def rollback(self, value):
        self._validate_open(value)
        self._open = None
        self._prepared = None
        return self._state

    def cancel(self, value):
        return self.rollback(value)

    def commit(self, value, *, authoritative_bed_mass_kg_m2):
        prepared = self.prepare_rebound(
            value,
            authoritative_bed_mass_kg_m2=authoritative_bed_mass_kg_m2)
        self.close_commit(prepared)
        return self.publish_commit(prepared)

    def checkpoint(self) -> dict[str, object]:
        if self._open is not None or self._prepared is not None or \
                self._closed is not None:
            raise RuntimeError(
                "v93 mixed CUDA checkpoint requires a committed point")
        state = self._state.core
        body = {
            "schema": SCHEMA,
            "scope_id": CHECKPOINT_SCOPE_ID+":actual-cuda",
            "outer_execution_fingerprint": self.outer_execution.fingerprint,
            "execution_fingerprint": self.execution_fingerprint,
            "binding_sha256": self.binding.binding_sha256,
            "topology_id": self.binding.topology_id,
            "boundary_names": list(self.boundary_names),
            "device_id": self.device_id,
            "deterministic_source_sha256":
                self.deterministic_source_sha256,
            "state": {
                "authoritative_bed_mass_kg_m2":
                    state.authoritative_bed_mass_kg_m2.tolist(),
                "bulk_density_kg_m3": state.bulk_density_kg_m3.tolist(),
                "active_fraction": state.active_fraction.tolist(),
                "resident_active_fraction": self.cp.asnumpy(
                    self._state.active_fraction).tolist(),
                "accepted_face_flux_kg_m_s":
                    state.accepted_face_flux_kg_m_s.tolist(),
                "accepted_cell_availability":
                    state.accepted_cell_availability.tolist(),
                "named_boundary_positive_in_kg":
                    state.named_boundary_positive_in_kg.tolist(),
                "named_boundary_absolute_kg":
                    state.named_boundary_absolute_kg.tolist(),
                "depletion_roundoff_positive_in_kg":
                    state.depletion_roundoff_positive_in_kg.tolist(),
                "accepted_updates": state.accepted_updates,
                "time_s": state.time_s,
                "host_state_sha256": state.state_sha256,
                "resident_state_sha256": self._state.state_sha256,
            },
            "extra_scalar_lifecycle_application_count": 0,
            "automatic_cpu_fallback": False,
        }
        return {**body, "fingerprint": _fingerprint(body)}

    def restore_checkpoint(self, value: Mapping[str, object]):
        if self._open is not None or self._prepared is not None or \
                self._closed is not None:
            raise RuntimeError(
                "v93 mixed CUDA restart cannot replace an open transaction")
        if not isinstance(value, Mapping):
            raise Q3V84MixedExnerAdapterV93Error(
                "v93 mixed CUDA checkpoint is missing")
        body = {key: item for key, item in value.items()
                if key != "fingerprint"}
        if (value.get("fingerprint") != _fingerprint(body) or
                value.get("schema") != SCHEMA or
                value.get("scope_id") !=
                    CHECKPOINT_SCOPE_ID+":actual-cuda" or
                value.get("outer_execution_fingerprint") !=
                    self.outer_execution.fingerprint or
                value.get("execution_fingerprint") !=
                    self.execution_fingerprint or
                value.get("binding_sha256") !=
                    self.binding.binding_sha256 or
                value.get("topology_id") != self.binding.topology_id or
                value.get("boundary_names") != list(self.boundary_names) or
                value.get("device_id") != self.device_id or
                value.get("deterministic_source_sha256") !=
                    self.deterministic_source_sha256 or
                value.get("extra_scalar_lifecycle_application_count") != 0 or
                value.get("automatic_cpu_fallback") is not False):
            raise Q3V84MixedExnerAdapterV93Error(
                "v93 mixed CUDA checkpoint identity differs")
        fields = value.get("state")
        if not isinstance(fields, Mapping):
            raise Q3V84MixedExnerAdapterV93Error(
                "v93 mixed CUDA checkpoint state is missing")
        state = resolve_resident_q3_v84_mixed_state_v93(
            execution_fingerprint=self.execution_fingerprint,
            binding=self.binding,
            authoritative_bed_mass_kg_m2=
                fields["authoritative_bed_mass_kg_m2"],
            bulk_density_kg_m3=fields["bulk_density_kg_m3"],
            retained_fraction=fields["resident_active_fraction"],
            accepted_face_flux_kg_m_s=
                fields["accepted_face_flux_kg_m_s"],
            accepted_cell_availability=
                fields["accepted_cell_availability"],
            named_boundary_positive_in_kg=
                fields["named_boundary_positive_in_kg"],
            named_boundary_absolute_kg=
                fields["named_boundary_absolute_kg"],
            depletion_roundoff_positive_in_kg=
                fields["depletion_roundoff_positive_in_kg"],
            accepted_updates=fields["accepted_updates"],
            time_s=fields["time_s"])
        if (state.core.state_sha256 != fields.get("host_state_sha256") or
                state.state_sha256 != fields.get("resident_state_sha256")):
            raise Q3V84MixedExnerAdapterV93Error(
                "v93 mixed CUDA checkpoint state differs")
        self._state = state
        return state


__all__ = [
    "BACKEND", "BED_SLOPE_SCOPE", "BOUNDARY_SCOPE", "CUDA_OPTIONS",
    "DETERMINISTIC_KERNEL_NAMES",
    "FRACTION_COUNT_SCOPE", "PROVIDER_KIND", "ResidentQ3V84FaceTrialV93",
    "ResidentQ3V84MixedBindingV93", "ResidentQ3V84MixedExnerAdapterV93",
    "ResidentQ3V84MixedStateV93", "ResidentQ3V84OpenTrialV93",
    "ResidentQ3V84PreparedCommitV93", "SCHEMA", "SCOPE_ID",
    "resolve_resident_q3_v84_mixed_binding_v93",
    "resolve_resident_q3_v84_mixed_state_v93",
    "validate_resident_q3_v84_mixed_state_v93",
]
