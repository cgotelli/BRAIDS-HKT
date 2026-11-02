"""Q4-owned mixed-route Q3 face/availability/Exner provider.

The v84 public ``begin_trial`` continues through v81/v78 and would duplicate
the scalar/FCT/exchange lifecycle already owned by v92.  This adapter instead
owns v84's pure constitutive-producer + v74 route-handoff seam, then performs
one bounded, authenticated H4/v70 face update.  It deliberately has no v67
or v81 child Exner commit: the outer v93 state is the only bed inventory.

The accepted route semantics are exact: OFF contributes zero, TOTAL sends the
v74 algebraic bed-plus-nonbed vector to Exner once, and COMBINE sends only the
v74 bed vector while v92 owns its water-column term.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import math

import numpy as np

from .bed_slope_execution import validate_bed_slope_execution_document
from .bedload_boundary_live_v70 import (
    SOURCE_EQUILIBRIUM_TRACE,
    SOURCE_NONE,
    SOURCE_OWNER_FREE_EXPORT,
    SOURCE_PRESCRIBED_EXTERNAL,
    BedloadBoundaryTopologyV70,
    load_bedload_boundary_execution_v70,
    resolve_bedload_boundary_candidate_v70,
)
from .direct_vector_wetdry_amr_live_v71 import (
    FACE_CLOSED,
    FACE_INTERNAL,
    FACE_OWNER_OUTLET,
    FACE_TRACE_INLET,
    WetDryTopologyV71,
)
from .q3_exner_adapter_v93 import (
    BACKEND_CPU,
    MORPHOLOGICAL_FACTOR,
    Q3ExnerReceiptV93,
    Q3V75ExnerAdapterV93Error,
    _array_digest,
    _base_topology,
    _fingerprint,
    _finite,
    _host_array,
    _receipt_token,
    _shared_topology_sha256,
    _v92_physical_sha256,
    _validate_outer_state,
    _surface_fraction,
    validate_q3_v75_exner_receipt_v93,
)
from .q3_producer_lifecycle_binding_v84 import (
    NativeQ3BoundRouteLifecycleV84,
    Q3RouteHandoffReceiptV84,
    _digest as _v84_array_digest,
    validate_route_handoff_receipt_v84,
)
from .route_lifecycle_composition_v81 import (
    ROUTE_COMBINE,
    ROUTE_OFF,
    ROUTE_TOTAL,
)
from .recursive_surface_feedback_v114 import (
    provider_surface_fraction_v114,
    provider_surface_slope_fields_v114,
)
from .transported_fluid_settling_v92 import (
    TransportedFluidTrialV92,
    _validate_trial as _validate_v92_trial,
)


SCHEMA = 1
SCOPE_ID = "braids-q4-v84-mixed-face-exner-adapter-v93-v1"
CHECKPOINT_SCOPE_ID = SCOPE_ID + ":checkpoint"
PROVIDER_KIND = "q4_owned_v84_mixed_face_no_scalar"
BED_SLOPE_SCOPE = "exact_zero_inert_no_live_slope_producer"
BOUNDARY_SCOPE = "closed_or_owner_free_export_only"
FRACTION_COUNT_SCOPE = 3


class Q3V84MixedExnerAdapterV93Error(Q3V75ExnerAdapterV93Error):
    """A pure v84 handoff or Q4-owned face transaction is invalid."""


def _readonly(value: object, shape: tuple[int, ...], name: str, *,
              integer: bool = False, nonnegative: bool = False) -> np.ndarray:
    try:
        return _host_array(
            value, shape, name, integer=integer, nonnegative=nonnegative)
    except Q3V75ExnerAdapterV93Error as error:
        raise Q3V84MixedExnerAdapterV93Error(str(error)) from error


@dataclass(frozen=True)
class CPUQ3V84MixedBindingV93:
    topology_id: str
    topology: WetDryTopologyV71
    normal: np.ndarray
    face_length_m: np.ndarray
    cell_face_offset: np.ndarray
    cell_faces: np.ndarray
    signs: np.ndarray
    downhill: np.ndarray
    bed_slope_execution: Mapping[str, object]
    boundary_execution: Mapping[str, object]
    boundary_topology: BedloadBoundaryTopologyV70
    family_boundary_faces: np.ndarray
    boundary_names: tuple[str, ...]
    binding_sha256: str


def resolve_cpu_q3_v84_mixed_binding_v93(
    *, topology: WetDryTopologyV71, normal: object,
    face_length_m: object, cell_face_offset: object, cell_faces: object,
    signs: object, downhill: object,
    bed_slope_execution: Mapping[str, object],
    boundary_execution: Mapping[str, object],
    boundary_topology: BedloadBoundaryTopologyV70,
    family_boundary_faces: object,
) -> CPUQ3V84MixedBindingV93:
    """Authenticate the fixed-grid H4/v70 topology used by the new owner."""
    if type(topology) is not WetDryTopologyV71 or not bool(np.all(topology.wet)):
        raise Q3V84MixedExnerAdapterV93Error(
            "v93 mixed CPU binding requires one fully-wet v71 topology")
    nc, nf = topology.ncell, topology.nface
    norm = _readonly(normal, (2, nf), "face normal")
    length = _readonly(face_length_m, (nf,), "face length", nonnegative=True)
    offset = _readonly(
        cell_face_offset, (nc+1,), "cell-face offset", integer=True)
    faces_raw = np.asarray(cell_faces)
    faces = _readonly(
        faces_raw, (faces_raw.size,), "cell faces", integer=True)
    signs_a = _readonly(signs, faces.shape, "cell-face signs", integer=True)
    down = _readonly(downhill, (2, nf), "downhill")
    # v93 does not yet receive a live, producer-authenticated slope field.
    # A caller-provided nonzero vector would therefore be configuration data,
    # not evidence from the accepted Q3 slope owner.  Admit only the exactly
    # inert case until that receipt is available.
    if np.any(down != 0.0):
        raise Q3V84MixedExnerAdapterV93Error(
            "v93 mixed owner admits only exact-zero downhill until a live "
            "bed-slope producer is bound")
    if (np.any(length <= 0.0) or offset[0] != 0 or
            offset[-1] != faces.size or np.any(np.diff(offset) < 0) or
            np.any((faces < 0) | (faces >= nf)) or
            np.any(~np.isin(signs_a, (-1, 1))) or
            np.any(np.abs(np.sum(norm*norm, axis=0)-1.0) > 1.0e-12)):
        raise Q3V84MixedExnerAdapterV93Error(
            "v93 mixed CPU face geometry or incidence is invalid")
    cell = np.repeat(np.arange(nc, dtype=np.int32), np.diff(offset))
    if np.any(~(((cell == topology.owner[faces]) & (signs_a == 1)) |
                ((cell == topology.neighbour[faces]) & (signs_a == -1)))):
        raise Q3V84MixedExnerAdapterV93Error(
            "v93 mixed CPU incidence ownership differs")
    slope = validate_bed_slope_execution_document(bed_slope_execution)
    # The bounded v84 profile contains positive classic TraFrm routes.  Their
    # accepted slope insertion is ISlope=2 with exact-zero taurat; later slope
    # modes need their own authenticated stress producer rather than a raw ABI.
    if int(slope["runtime_mode"]) != 2:
        raise Q3V84MixedExnerAdapterV93Error(
            "v93 mixed v84 owner currently requires authenticated ISlope=2")
    boundary = load_bedload_boundary_execution_v70(boundary_execution)
    if (type(boundary_topology) is not BedloadBoundaryTopologyV70 or
            boundary_topology.execution_fingerprint != boundary["fingerprint"]):
        raise Q3V84MixedExnerAdapterV93Error(
            "v93 mixed CPU v70 boundary topology differs")
    mapping_raw = np.asarray(family_boundary_faces)
    mapping = _readonly(
        mapping_raw, (boundary_topology.nface,), "family boundary faces",
        integer=True)
    if (np.any((mapping < 0) | (mapping >= nf)) or
            len(set(mapping.tolist())) != mapping.size or
            np.any(topology.neighbour[mapping] != -1) or
            not np.array_equal(topology.owner[mapping],
                               boundary_topology.owner) or
            not np.array_equal(norm[:, mapping], boundary_topology.normal) or
            not np.array_equal(length[mapping],
                               boundary_topology.face_length_m) or
            not np.array_equal(topology.cell_area_m2,
                               boundary_topology.cell_area_m2)):
        raise Q3V84MixedExnerAdapterV93Error(
            "v93 mixed CPU family/v70 boundary topology differs")
    names = tuple(row["name"] for row in boundary["boundaries"])
    if not names or len(set(names)) != len(names):
        raise Q3V84MixedExnerAdapterV93Error(
            "v93 mixed CPU logical boundary names are invalid")
    token = _fingerprint({
        "scope_id": SCOPE_ID + ":binding",
        "topology_id": topology.topology_id,
        "bed_slope_execution_fingerprint": slope["fingerprint"],
        "boundary_execution_fingerprint": boundary["fingerprint"],
        "boundary_topology_sha256": boundary_topology.topology_sha256,
        "boundary_names": list(names),
        "arrays_sha256": _array_digest(
            norm, length, offset, faces, signs_a, down, mapping),
        "trace_policy": BOUNDARY_SCOPE,
        "bed_slope_scope": BED_SLOPE_SCOPE,
        "fully_wet_fixed_grid": True,
    })
    return CPUQ3V84MixedBindingV93(
        topology.topology_id, topology, norm, length, offset, faces,
        signs_a, down, slope, boundary, boundary_topology, mapping, names,
        token)


@dataclass(frozen=True)
class CPUQ3V84MixedStateV93:
    execution_fingerprint: str
    binding_sha256: str
    topology_id: str
    authoritative_bed_mass_kg_m2: np.ndarray
    bulk_density_kg_m3: np.ndarray
    active_fraction: np.ndarray
    accepted_face_flux_kg_m_s: np.ndarray
    accepted_cell_availability: np.ndarray
    named_boundary_positive_in_kg: np.ndarray
    named_boundary_absolute_kg: np.ndarray
    depletion_roundoff_positive_in_kg: np.ndarray
    accepted_updates: int
    time_s: float
    state_sha256: str


def _state_token(value: CPUQ3V84MixedStateV93) -> str:
    return _fingerprint({
        "scope_id": SCOPE_ID + ":state",
        "execution_fingerprint": value.execution_fingerprint,
        "binding_sha256": value.binding_sha256,
        "topology_id": value.topology_id,
        "accepted_updates": value.accepted_updates,
        "time_s": value.time_s,
        "arrays_sha256": _array_digest(
            value.authoritative_bed_mass_kg_m2,
            value.bulk_density_kg_m3, value.active_fraction,
            value.accepted_face_flux_kg_m_s,
            value.accepted_cell_availability,
            value.named_boundary_positive_in_kg,
            value.named_boundary_absolute_kg,
            value.depletion_roundoff_positive_in_kg),
        "bed_owner": "outer_v93_projection_only",
    })


def resolve_cpu_q3_v84_mixed_state_v93(
    *, execution_fingerprint: str, binding: CPUQ3V84MixedBindingV93,
    authoritative_bed_mass_kg_m2: object, bulk_density_kg_m3: object,
    retained_fraction: object | None = None,
    accepted_face_flux_kg_m_s: object | None = None,
    accepted_cell_availability: object | None = None,
    named_boundary_positive_in_kg: object | None = None,
    named_boundary_absolute_kg: object | None = None,
    depletion_roundoff_positive_in_kg: object | None = None,
    accepted_updates: int = 0, time_s: float = 0.0,
) -> CPUQ3V84MixedStateV93:
    ng = len(binding.bed_slope_execution["ordered_fraction_ids"])
    nc, nf, nb = (binding.topology.ncell, binding.topology.nface,
                  len(binding.boundary_names))
    bed = _readonly(
        authoritative_bed_mass_kg_m2, (ng, nc), "projected bed",
        nonnegative=True)
    bulk = _readonly(
        bulk_density_kg_m3, (ng,), "bulk density", nonnegative=True)
    if np.any(bulk <= 0.0):
        raise Q3V84MixedExnerAdapterV93Error(
            "v93 mixed CPU bulk density must be positive")
    # There is one surface-mixture owner: positive stock is derived from the
    # authoritative outer bed, while a depleted cell retains its last
    # normalized surface identity.  This is the same contract as v67 and
    # prevents a zero-stock rebase/restart from silently losing grain order.
    volume = bed/bulk[:, None]
    positive = np.sum(volume, axis=0) > 0.0
    if retained_fraction is None:
        if np.any(~positive):
            raise Q3V84MixedExnerAdapterV93Error(
                "v93 mixed depleted cells require an explicit retained "
                "surface identity")
        retained = np.zeros_like(bed)
    else:
        retained = _readonly(
            retained_fraction, (ng, nc), "retained active fraction",
            nonnegative=True)
        tolerance = 512.0*np.finfo(np.float64).eps
        if (np.any(retained > 1.0+tolerance) or
                not np.allclose(np.sum(retained, axis=0), 1.0,
                                rtol=0.0, atol=tolerance*ng)):
            raise Q3V84MixedExnerAdapterV93Error(
                "v93 mixed retained active fractions must be normalized")
    fraction = _surface_fraction(bed, bulk, retained)
    flux = _readonly(
        np.zeros((ng, nf)) if accepted_face_flux_kg_m_s is None else
        accepted_face_flux_kg_m_s, (ng, nf), "accepted face flux")
    availability = _readonly(
        np.zeros((ng, nc)) if accepted_cell_availability is None else
        accepted_cell_availability, (ng, nc), "accepted availability",
        nonnegative=True)
    if np.any(availability > 1.0):
        raise Q3V84MixedExnerAdapterV93Error(
            "v93 mixed accepted availability exceeds one")
    named = _readonly(
        np.zeros((ng, nb)) if named_boundary_positive_in_kg is None else
        named_boundary_positive_in_kg, (ng, nb), "cumulative named ledger")
    absolute = _readonly(
        np.zeros((ng, nb)) if named_boundary_absolute_kg is None else
        named_boundary_absolute_kg, (ng, nb), "cumulative absolute ledger",
        nonnegative=True)
    depletion = _readonly(
        np.zeros(ng) if depletion_roundoff_positive_in_kg is None else
        depletion_roundoff_positive_in_kg, (ng,), "cumulative depletion",
        nonnegative=True)
    if np.any(absolute < np.abs(named)):
        raise Q3V84MixedExnerAdapterV93Error(
            "v93 mixed absolute boundary ledger does not bound signed")
    if (type(accepted_updates) is not int or accepted_updates < 0 or
            accepted_updates > 4):
        raise Q3V84MixedExnerAdapterV93Error(
            "v93 mixed accepted update is invalid")
    time = _finite(time_s, "accepted time")
    if time < 0.0:
        raise Q3V84MixedExnerAdapterV93Error(
            "v93 mixed accepted time is negative")
    placeholder = CPUQ3V84MixedStateV93(
        execution_fingerprint, binding.binding_sha256, binding.topology_id,
        bed, bulk, fraction, flux, availability, named, absolute, depletion,
        accepted_updates, time, "")
    return CPUQ3V84MixedStateV93(
        **{**placeholder.__dict__, "state_sha256": _state_token(placeholder)})


def validate_cpu_q3_v84_mixed_state_v93(
    value: CPUQ3V84MixedStateV93, *, execution_fingerprint: str,
    binding: CPUQ3V84MixedBindingV93,
) -> None:
    if (not isinstance(value, CPUQ3V84MixedStateV93) or
            value.execution_fingerprint != execution_fingerprint or
            value.binding_sha256 != binding.binding_sha256 or
            value.topology_id != binding.topology_id or
            value.state_sha256 != _state_token(value)):
        raise Q3V84MixedExnerAdapterV93Error(
            "v93 mixed CPU state is stale or cross-execution")
    rebuilt = resolve_cpu_q3_v84_mixed_state_v93(
        execution_fingerprint=execution_fingerprint, binding=binding,
        authoritative_bed_mass_kg_m2=value.authoritative_bed_mass_kg_m2,
        bulk_density_kg_m3=value.bulk_density_kg_m3,
        retained_fraction=value.active_fraction,
        accepted_face_flux_kg_m_s=value.accepted_face_flux_kg_m_s,
        accepted_cell_availability=value.accepted_cell_availability,
        named_boundary_positive_in_kg=value.named_boundary_positive_in_kg,
        named_boundary_absolute_kg=value.named_boundary_absolute_kg,
        depletion_roundoff_positive_in_kg=
            value.depletion_roundoff_positive_in_kg,
        accepted_updates=value.accepted_updates, time_s=value.time_s)
    if rebuilt.state_sha256 != value.state_sha256:
        raise Q3V84MixedExnerAdapterV93Error(
            "v93 mixed CPU state cannot be reconstructed")


@dataclass(frozen=True)
class CPUQ3V84FaceTrialV93:
    parent_state_sha256: str
    v84_route_handoff: Q3RouteHandoffReceiptV84
    accepted_endpoint_q_sha256: str
    accepted_hydraulic_face_receipt_sha256: str
    dt_s: float
    morphological_factor: float
    face_flux_kg_m_s: np.ndarray
    cell_delta_kg_m2: np.ndarray
    inventory_fill: np.ndarray
    accepted_cell_availability: np.ndarray
    named_boundary_transfer_kg: np.ndarray
    named_boundary_absolute_kg: np.ndarray
    depletion_roundoff_positive_in_kg: np.ndarray
    conservation_residual_kg: np.ndarray
    constitutive_producer_count: np.ndarray
    route_handoff_count: np.ndarray
    face_reconstruction_count: np.ndarray
    bed_slope_count: np.ndarray
    boundary_owner_count: np.ndarray
    inventory_fill_count: np.ndarray
    face_exner_count: np.ndarray
    extra_scalar_lifecycle_count: np.ndarray
    trial_sha256: str


def _face_trial_token(value: CPUQ3V84FaceTrialV93) -> str:
    return _fingerprint({
        "scope_id": SCOPE_ID + ":face-trial",
        "parent_state_sha256": value.parent_state_sha256,
        "v84_route_handoff_sha256": value.v84_route_handoff.receipt_sha256,
        "accepted_endpoint_q_sha256": value.accepted_endpoint_q_sha256,
        "accepted_hydraulic_face_receipt_sha256":
            value.accepted_hydraulic_face_receipt_sha256,
        "dt_s": value.dt_s,
        "morphological_factor": value.morphological_factor,
        "arrays_sha256": _array_digest(
            value.face_flux_kg_m_s, value.cell_delta_kg_m2,
            value.inventory_fill, value.accepted_cell_availability,
            value.named_boundary_transfer_kg,
            value.named_boundary_absolute_kg,
            value.depletion_roundoff_positive_in_kg,
            value.conservation_residual_kg,
            value.constitutive_producer_count, value.route_handoff_count,
            value.face_reconstruction_count, value.bed_slope_count,
            value.boundary_owner_count, value.inventory_fill_count,
            value.face_exner_count, value.extra_scalar_lifecycle_count),
        "single_outer_face_owner": True,
        "child_exner_commit": False,
    })


@dataclass(frozen=True)
class CPUQ3V84MixedOpenTrialV93:
    parent_state: CPUQ3V84MixedStateV93
    face_trial: CPUQ3V84FaceTrialV93
    receipt: Q3ExnerReceiptV93
    handle_sha256: str


@dataclass(frozen=True)
class CPUQ3V84MixedPreparedCommitV93:
    open_trial: CPUQ3V84MixedOpenTrialV93
    rebound_state: CPUQ3V84MixedStateV93
    authoritative_bed_sha256: str
    preparation_sha256: str

    @property
    def rebound_state_sha256(self) -> str:
        return self.rebound_state.state_sha256

    @property
    def rebound_bed_mass_kg_m2(self) -> np.ndarray:
        return self.rebound_state.authoritative_bed_mass_kg_m2


def _open_token(value: CPUQ3V84MixedOpenTrialV93) -> str:
    return _fingerprint({
        "scope_id": SCOPE_ID + ":open",
        "parent": value.parent_state.state_sha256,
        "face_trial": value.face_trial.trial_sha256,
        "receipt": value.receipt.receipt_sha256,
    })


def _prepared_token(value: CPUQ3V84MixedPreparedCommitV93) -> str:
    return _fingerprint({
        "scope_id": SCOPE_ID + ":prepared",
        "open": value.open_trial.handle_sha256,
        "rebound": value.rebound_state.state_sha256,
        "authoritative_bed": value.authoritative_bed_sha256,
    })


class CPUQ3V84MixedExnerAdapterV93:
    """Bounded accepted CPU owner for v84 mixed-route face transport."""

    backend = BACKEND_CPU
    provider_kind = PROVIDER_KIND
    bed_slope_scope = BED_SLOPE_SCOPE
    boundary_scope = BOUNDARY_SCOPE
    fraction_count_scope = FRACTION_COUNT_SCOPE

    def __init__(self, *, outer_execution,
                 runtime: NativeQ3BoundRouteLifecycleV84,
                 binding: CPUQ3V84MixedBindingV93,
                 state: CPUQ3V84MixedStateV93):
        from .suspended_exner_owner_v93 import execution_document_v93
        execution_document_v93(outer_execution)
        if (outer_execution.backend != "cpu" or
                outer_execution.fraction_count != FRACTION_COUNT_SCOPE or
                type(runtime) is not NativeQ3BoundRouteLifecycleV84 or
                type(binding) is not CPUQ3V84MixedBindingV93):
            raise Q3V84MixedExnerAdapterV93Error(
                "v93 mixed CPU adapter requires the exact three-fraction "
                "CPU v84 objects")
        if (runtime.execution["fingerprint"] !=
                outer_execution.q3_execution_fingerprint or
                binding.topology_id != outer_execution.q3_topology_id or
                binding.binding_sha256 != outer_execution.q3_binding_sha256 or
                binding.boundary_names !=
                    tuple(outer_execution.q3_boundary_names)):
            raise Q3V84MixedExnerAdapterV93Error(
                "v93 mixed CPU execution, topology, binding, or names differ")
        routes = tuple(runtime.lifecycle_execution.route_execution["routes"])
        if routes != tuple(np.asarray(outer_execution.routes).tolist()):
            raise Q3V84MixedExnerAdapterV93Error(
                "v93 mixed CPU v74 and outer routes differ")
        fraction_ids = tuple(runtime.producer.producer_execution[
            "physical_profile"]["fraction_ids"])
        if (fraction_ids != tuple(outer_execution.ordered_fraction_ids) or
                tuple(binding.bed_slope_execution["ordered_fraction_ids"]) !=
                    fraction_ids):
            raise Q3V84MixedExnerAdapterV93Error(
                "v93 mixed CPU fraction order differs")
        if not np.array_equal(
                state.bulk_density_kg_m3,
                np.asarray(binding.boundary_execution[
                    "bulk_density_kg_m3"], dtype=np.float64)):
            raise Q3V84MixedExnerAdapterV93Error(
                "v93 mixed CPU bulk density differs from bound v70 physics")
        # v84's lifecycle execution is inspected only for physical/profile
        # identity.  Its scalar begin_trial is never invoked by this adapter.
        outer_v78 = (outer_execution.base_execution.base_execution.
                     base_execution)
        if runtime.lifecycle_execution.lifecycle_execution.fingerprint != \
                outer_v78.fingerprint:
            raise Q3V84MixedExnerAdapterV93Error(
                "v93 mixed CPU v84/v92 suspension discretisations differ")
        self.outer_execution = outer_execution
        self.runtime = runtime
        self.binding = binding
        self.boundary_names = binding.boundary_names
        self._validate_topology_and_profile()
        validate_cpu_q3_v84_mixed_state_v93(
            state, execution_fingerprint=self.execution_fingerprint,
            binding=binding)
        self._state = state
        self._open: CPUQ3V84MixedOpenTrialV93 | None = None
        self._prepared: CPUQ3V84MixedPreparedCommitV93 | None = None
        self._closed: CPUQ3V84MixedPreparedCommitV93 | None = None

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
    def state(self) -> CPUQ3V84MixedStateV93:
        return self._state

    @property
    def state_sha256(self) -> str:
        return self._state.state_sha256

    @property
    def accepted_updates(self) -> int:
        return self._state.accepted_updates

    @property
    def time_s(self) -> float:
        return self._state.time_s

    @property
    def authoritative_bed_mass_kg_m2(self) -> np.ndarray:
        return self._state.authoritative_bed_mass_kg_m2

    def _validate_topology_and_profile(self) -> None:
        base = _base_topology(self.outer_execution)
        topology = self.binding.topology
        mapping = np.asarray(self.binding.family_boundary_faces)
        local_boundary = np.asarray(
            self.binding.boundary_topology.boundary_index)
        expected_faces = np.flatnonzero(base.boundary_index >= 0)
        checks = (
            topology.ncell == base.cell_count,
            topology.nface == base.face_count,
            np.array_equal(topology.owner, base.owner),
            np.array_equal(topology.neighbour, base.neighbour),
            np.array_equal(topology.cell_area_m2, base.cell_area_m2),
            np.array_equal(self.binding.normal, base.normal),
            np.array_equal(self.binding.face_length_m, base.face_length_m),
            np.array_equal(self.binding.cell_face_offset,
                           base.cell_face_offset),
            np.array_equal(self.binding.cell_faces, base.cell_faces),
            np.array_equal(self.binding.signs, base.cell_face_sign),
            base.boundary_count == len(self.boundary_names),
            np.array_equal(mapping, expected_faces),
            np.array_equal(local_boundary, base.boundary_index[mapping]),
            bool(np.all(topology.wet)),
        )
        if not all(checks):
            raise Q3V84MixedExnerAdapterV93Error(
                "v93 mixed CPU topology/boundary map differs from v92")
        physical = self.runtime.producer.producer_execution[
            "physical_profile"]
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
                "v93 mixed CPU physical profile differs from v92")

    def validate_current_bed_clock(self, outer_parent) -> None:
        _validate_outer_state(outer_parent, self.outer_execution)
        validate_cpu_q3_v84_mixed_state_v93(
            self._state, execution_fingerprint=self.execution_fingerprint,
            binding=self.binding)
        if (not np.array_equal(self._state.authoritative_bed_mass_kg_m2,
                               outer_parent.bed_mass_kg_m2) or
                self._state.accepted_updates != outer_parent.accepted_updates or
                self._state.time_s != outer_parent.time_s):
            raise Q3V84MixedExnerAdapterV93Error(
                "v93 mixed CPU provider was not rebound to the outer bed/clock")

    def _dynamic_roles(self, *, hydraulic_boundary: np.ndarray,
                       cell_vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        topology = self.binding.topology
        mapping = self.binding.family_boundary_faces
        # v92 authenticates accepted face water flux, but does not yet expose
        # an authenticated boundary hydraulic trace state.  The placeholder
        # below cannot enter accepted work because every trace/external source
        # is rejected before H4 composition.
        trace = np.zeros(
            (2, cell_vector.shape[1], topology.nface), dtype=np.float64)
        candidate = resolve_bedload_boundary_candidate_v70(
            self.binding.boundary_execution,
            self.binding.boundary_topology,
            hydraulic_outward_flux=hydraulic_boundary,
            owner_bed_mass_vector=cell_vector,
            trace_bed_mass_vector=np.ascontiguousarray(trace[:, :, mapping]))
        role = np.array(topology.role, dtype=np.int32, copy=True, order="C")
        for local, family in enumerate(mapping):
            sources = set(candidate.source[:, local].tolist())
            if (SOURCE_PRESCRIBED_EXTERNAL in sources or
                    SOURCE_EQUILIBRIUM_TRACE in sources):
                raise Q3V84MixedExnerAdapterV93Error(
                    "v93 mixed inlet/external trace capacity is not authenticated")
            if sources == {SOURCE_OWNER_FREE_EXPORT}:
                role[family] = FACE_OWNER_OUTLET
            elif sources == {SOURCE_NONE}:
                role[family] = FACE_CLOSED
            else:
                raise Q3V84MixedExnerAdapterV93Error(
                    "v93 mixed boundary classes selected different source owners")
        role = np.ascontiguousarray(role)
        role.setflags(write=False)
        return role, trace

    def _face_trial(
        self, *, route: Q3RouteHandoffReceiptV84,
        v92_trial: TransportedFluidTrialV92,
    ) -> CPUQ3V84FaceTrialV93:
        from braids_gpu.flow_bed_slope_composition import (
            DONOR_NEIGHBOUR, DONOR_NONE, DONOR_OWNER, DONOR_TRACE,
            PortableFlowBedSlopeFaceComposer,
        )
        ng = self.outer_execution.fraction_count
        nc, nf = self.binding.topology.ncell, self.binding.topology.nface
        active = np.asarray(self.outer_execution.routes) != ROUTE_OFF
        handoff = route.handoff
        if (not np.array_equal(handoff.active_fraction_application_count,
                               active.astype(np.int32)) or
                not np.array_equal(
                    handoff.bed_suspended_owner_count,
                    (np.asarray(self.outer_execution.routes) ==
                     ROUTE_TOTAL).astype(np.int32)) or
                not np.array_equal(
                    handoff.water_suspended_owner_count,
                    (np.asarray(self.outer_execution.routes) ==
                     ROUTE_COMBINE).astype(np.int32))):
            raise Q3V84MixedExnerAdapterV93Error(
                "v93 mixed v74 route ownership differs")
        cell_vector = np.ascontiguousarray(np.stack((
            handoff.bed_capacity_x, handoff.bed_capacity_y), axis=0))
        if np.any(cell_vector[:, ~active] != 0.0):
            raise Q3V84MixedExnerAdapterV93Error(
                "v93 mixed OFF route acquired bed capacity")
        mapping = np.asarray(self.binding.family_boundary_faces)
        hydraulic_boundary = np.ascontiguousarray(
            v92_trial.hydraulic_face_receipt.
            accepted_water_flux_m2_s[mapping])
        role, trace = self._dynamic_roles(
            hydraulic_boundary=hydraulic_boundary,
            cell_vector=cell_vector)
        topology = self.binding.topology
        safe_other = np.clip(topology.neighbour, 0, nc-1)
        left = np.ascontiguousarray(cell_vector[:, :, topology.owner])
        right = np.ascontiguousarray(cell_vector[:, :, safe_other])
        owner_bed = left
        trace_bed = np.ascontiguousarray(trace)
        zeros_stress = np.zeros((ng, nf), dtype=np.float64)
        neutral = np.ones((ng, nf), dtype=np.float64)
        hiding, surface_mean = provider_surface_slope_fields_v114(
            self, topology, np)
        d50 = np.ascontiguousarray(np.asarray(
            _base_topology(self.outer_execution).profile.sediment_d50_m,
            dtype=np.float64))
        composer = PortableFlowBedSlopeFaceComposer(
            self.binding.bed_slope_execution)
        common = dict(
            left_bed=left, right_bed=right, owner_bed=owner_bed,
            trace_bed=trace_bed,
            owner=np.ascontiguousarray(topology.owner, dtype=np.int32),
            neighbour=np.ascontiguousarray(topology.neighbour,
                                            dtype=np.int32),
            role=np.ascontiguousarray(role, dtype=np.int32),
            normal=np.ascontiguousarray(self.binding.normal),
            downhill=np.ascontiguousarray(self.binding.downhill),
            stress=zeros_stress, depth=neutral, d50=d50,
            mean_diameter=surface_mean, hiding=hiding)
        provisional = composer.compose(
            availability=np.ones((ng, nc), dtype=np.float64), **common)
        dt = _finite(v92_trial.hydraulic_receipt.dt_s, "accepted dt",
                     positive=True)
        donor_cell = np.full((ng, nf), -1, dtype=np.int32)
        donor_cell[provisional.donor == DONOR_OWNER] = np.broadcast_to(
            topology.owner, donor_cell.shape)[
                provisional.donor == DONOR_OWNER]
        donor_cell[provisional.donor == DONOR_NEIGHBOUR] = np.broadcast_to(
            safe_other, donor_cell.shape)[
                provisional.donor == DONOR_NEIGHBOUR]
        donor_cell[provisional.donor == DONOR_TRACE] = np.broadcast_to(
            topology.owner, donor_cell.shape)[
                provisional.donor == DONOR_TRACE]
        donor_cell[provisional.donor == DONOR_NONE] = -1
        outgoing = np.zeros((ng, nc), dtype=np.float64)
        for face in range(nf):
            for grain in range(ng):
                cell = int(donor_cell[grain, face])
                value = float(provisional.flux[grain, face])
                is_outgoing = (
                    cell >= 0 and
                    ((int(role[face]) == FACE_INTERNAL and value != 0.0) or
                     (int(role[face]) == FACE_OWNER_OUTLET and value > 0.0)))
                if is_outgoing:
                    outgoing[grain, cell] += (
                        abs(value)*self.binding.face_length_m[face]*dt /
                        topology.cell_area_m2[cell])
        fill = np.ones((ng, nc), dtype=np.float64)
        fill[self._state.authoritative_bed_mass_kg_m2 == 0.0] = 0.0
        demand = outgoing > 0.0
        fill[demand] = np.minimum(
            1.0, self._state.authoritative_bed_mass_kg_m2[demand] /
            outgoing[demand])
        accepted = composer.compose(
            availability=np.ascontiguousarray(fill), **common)
        flux = np.ascontiguousarray(accepted.flux)
        delta = np.zeros((ng, nc), dtype=np.float64)
        named = np.zeros((ng, len(self.boundary_names)), dtype=np.float64)
        absolute = np.zeros_like(named)
        inverse = {int(face): local for local, face in enumerate(mapping)}
        for face in range(nf):
            amount = flux[:, face]*self.binding.face_length_m[face]*dt
            owner = int(topology.owner[face])
            delta[:, owner] -= amount/topology.cell_area_m2[owner]
            if int(role[face]) == FACE_INTERNAL:
                other = int(topology.neighbour[face])
                delta[:, other] += amount/topology.cell_area_m2[other]
            else:
                local = inverse.get(face)
                if local is None:
                    raise Q3V84MixedExnerAdapterV93Error(
                        "v93 mixed boundary face lacks a logical owner")
                boundary = int(
                    self.binding.boundary_topology.boundary_index[local])
                named[:, boundary] += amount
                absolute[:, boundary] += np.abs(amount)
        candidate = self._state.authoritative_bed_mass_kg_m2+delta
        tolerance = 4096.0*np.finfo(np.float64).eps*np.maximum(
            1.0, np.abs(self._state.authoritative_bed_mass_kg_m2))
        if np.any(candidate < -tolerance):
            raise Q3V84MixedExnerAdapterV93Error(
                "v93 mixed inventory limiter produced a negative bed")
        before = np.array(delta, copy=True)
        negative = candidate < 0.0
        delta[negative] = -self._state.authoritative_bed_mass_kg_m2[negative]
        correction = delta-before
        depletion = np.sum(
            correction*topology.cell_area_m2[None, :], axis=1)
        if np.any(self._state.authoritative_bed_mass_kg_m2+delta < 0.0):
            raise Q3V84MixedExnerAdapterV93Error(
                "v93 mixed depletion correction did not close exactly")
        residual = np.sum(
            delta*topology.cell_area_m2[None, :], axis=1) + \
            np.sum(named, axis=1)-depletion
        scale = np.maximum(
            1.0, np.sum(np.abs(delta*topology.cell_area_m2[None, :]),
                        axis=1)+np.sum(absolute, axis=1)+depletion)
        if np.any(np.abs(residual) >
                  8192.0*np.finfo(np.float64).eps*scale):
            raise Q3V84MixedExnerAdapterV93Error(
                "v93 mixed face/Exner ledger does not close")
        counts = active.astype(np.int32)
        zeros = np.zeros(ng, dtype=np.int32)
        arrays = []
        for value in (flux, delta, fill, fill, named, absolute, depletion,
                      residual,
                      counts, counts, counts, zeros, counts, counts, counts,
                      zeros):
            array = np.ascontiguousarray(value)
            array.setflags(write=False)
            arrays.append(array)
        placeholder = CPUQ3V84FaceTrialV93(
            self._state.state_sha256, route,
            _v84_array_digest(
                v92_trial.candidate.q[0], v92_trial.candidate.q[1:]),
            v92_trial.hydraulic_face_receipt.receipt_sha256,
            dt, MORPHOLOGICAL_FACTOR, *arrays, "")
        return CPUQ3V84FaceTrialV93(
            **{**placeholder.__dict__,
               "trial_sha256": _face_trial_token(placeholder)})

    def _make_receipt(self, *, outer_parent, v92_trial,
                      face_trial: CPUQ3V84FaceTrialV93) -> Q3ExnerReceiptV93:
        ng = self.outer_execution.fraction_count
        active = np.asarray(self.outer_execution.routes) != ROUTE_OFF
        delta = _readonly(
            face_trial.cell_delta_kg_m2,
            (ng, self.outer_execution.cell_count), "bed delta")
        positive = _readonly(
            -np.asarray(face_trial.named_boundary_transfer_kg),
            (ng, len(self.boundary_names)), "named positive-in")
        absolute = _readonly(
            face_trial.named_boundary_absolute_kg, positive.shape,
            "named absolute", nonnegative=True)
        depletion = _readonly(
            face_trial.depletion_roundoff_positive_in_kg, (ng,),
            "depletion", nonnegative=True)
        boundary = _readonly(
            np.sum(positive, axis=1)+depletion, (ng,), "global boundary")
        routes = _readonly(
            self.outer_execution.routes, (ng,), "routes", integer=True)
        owner = _readonly(active.astype(np.int32), (ng,), "owner count",
                          integer=True)
        zero = _readonly(np.zeros(ng, dtype=np.int32), (ng,),
                         "extra scalar lifecycle", integer=True)
        placeholder = Q3ExnerReceiptV93(
            self.outer_execution.fingerprint, outer_parent.state_sha256,
            v92_trial.trial_sha256,
            v92_trial.hydraulic_receipt.receipt_sha256,
            v92_trial.hydraulic_face_receipt.receipt_sha256,
            self.execution_fingerprint, self.binding.binding_sha256,
            self.binding.topology_id, self._state.state_sha256,
            face_trial.trial_sha256, face_trial.trial_sha256,
            _array_digest(face_trial.face_flux_kg_m_s),
            _shared_topology_sha256(_base_topology(self.outer_execution)),
            self.ordered_fraction_ids,
            _v92_physical_sha256(self.outer_execution),
            face_trial.dt_s, MORPHOLOGICAL_FACTOR, self.boundary_names,
            routes, delta, positive, absolute, depletion, boundary, owner,
            self.backend, self.provider_kind, owner, owner, zero, "")
        return Q3ExnerReceiptV93(
            **{**placeholder.__dict__,
               "receipt_sha256": _receipt_token(placeholder)})

    def _validate_face_trial(self, value: CPUQ3V84FaceTrialV93) -> None:
        if (not isinstance(value, CPUQ3V84FaceTrialV93) or
                value.parent_state_sha256 != self._state.state_sha256 or
                value.dt_s <= 0.0 or
                value.morphological_factor != MORPHOLOGICAL_FACTOR or
                value.trial_sha256 != _face_trial_token(value) or
                value.v84_route_handoff.accepted_endpoint_sha256 !=
                    value.accepted_endpoint_q_sha256 or
                value.v84_route_handoff.active_fraction_sha256 !=
                    _v84_array_digest(provider_surface_fraction_v114(self)) or
                value.v84_route_handoff.scalar_lifecycle_application_count !=
                    0):
            raise Q3V84MixedExnerAdapterV93Error(
                "v93 mixed CPU face trial is stale or duplicated scalar work")
        validate_route_handoff_receipt_v84(
            value.v84_route_handoff,
            execution_fingerprint=self.execution_fingerprint,
            route_execution=self.runtime.lifecycle_execution.route_execution,
            backend="cpu")

    def begin(
        self, *, outer_parent, parent_hles, handoff,
        v92_trial: TransportedFluidTrialV92,
        roughness_parameter, kinematic_viscosity_m2_s,
        settling_velocity_m_s,
        cancelled: Callable[[], bool] = lambda: False,
    ) -> CPUQ3V84MixedOpenTrialV93:
        if not callable(cancelled):
            raise TypeError("cancelled must be callable")
        if self._open is not None or self._prepared is not None or \
                self._closed is not None:
            raise RuntimeError("v93 mixed CPU provider already has a transaction")
        _validate_outer_state(outer_parent, self.outer_execution)
        _validate_v92_trial(
            self.outer_execution.base_execution, outer_parent.base_state,
            parent_hles, handoff, v92_trial)
        if v92_trial.backend != BACKEND_CPU:
            raise Q3V84MixedExnerAdapterV93Error(
                "v93 mixed CPU provider requires the CPU v92 trial")
        self.validate_current_bed_clock(outer_parent)
        if cancelled():
            raise InterruptedError("v93 mixed CPU cancelled before Q3")
        route = None
        try:
            surface_fraction = provider_surface_fraction_v114(self)
            route = NativeQ3BoundRouteLifecycleV84.produce_route_handoff(
                self.runtime,
                depth_after_m=v92_trial.candidate.q[0],
                endpoint_momentum_m2_s=v92_trial.candidate.q[1:],
                roughness_parameter=roughness_parameter,
                kinematic_viscosity_m2_s=kinematic_viscosity_m2_s,
                settling_velocity_m_s=settling_velocity_m_s,
                active_fraction=surface_fraction)
            endpoint_sha = _v84_array_digest(
                v92_trial.candidate.q[0], v92_trial.candidate.q[1:])
            fraction_sha = _v84_array_digest(surface_fraction)
            validate_route_handoff_receipt_v84(
                route, execution_fingerprint=self.execution_fingerprint,
                route_execution=
                    self.runtime.lifecycle_execution.route_execution,
                backend="cpu", accepted_endpoint_sha256=endpoint_sha,
                active_fraction_sha256=fraction_sha)
            face = self._face_trial(route=route, v92_trial=v92_trial)
            self._validate_face_trial(face)
            receipt = self._make_receipt(
                outer_parent=outer_parent, v92_trial=v92_trial,
                face_trial=face)
            handle = CPUQ3V84MixedOpenTrialV93(
                self._state, face, receipt, "")
            handle = CPUQ3V84MixedOpenTrialV93(
                handle.parent_state, handle.face_trial, handle.receipt,
                _open_token(handle))
            if cancelled():
                raise InterruptedError(
                    "v93 mixed CPU cancelled before trial publication")
            self._open = handle
            return handle
        except Exception:
            # Pure v84 production and H4 composition do not mutate a runtime;
            # clearing the unpublished handle is therefore complete rollback.
            self._open = None
            self._prepared = None
            raise

    def _validate_open(self, value: CPUQ3V84MixedOpenTrialV93, *,
                       outer_parent=None, v92_trial=None) -> None:
        if (not isinstance(value, CPUQ3V84MixedOpenTrialV93) or
                value is not self._open or value.parent_state is not self._state or
                value.handle_sha256 != _open_token(value)):
            raise Q3V84MixedExnerAdapterV93Error(
                "v93 mixed CPU open handle differs")
        self._validate_face_trial(value.face_trial)
        if outer_parent is not None and v92_trial is not None:
            if value.face_trial.accepted_endpoint_q_sha256 != _v84_array_digest(
                    v92_trial.candidate.q[0], v92_trial.candidate.q[1:]):
                raise Q3V84MixedExnerAdapterV93Error(
                    "v93 mixed CPU constitutive endpoint differs from v92")
            validate_q3_v75_exner_receipt_v93(
                value.receipt, outer_execution=self.outer_execution,
                outer_parent=outer_parent, v92_trial=v92_trial,
                q3_binding_sha256=self.binding.binding_sha256,
                q3_topology_id=self.binding.topology_id,
                backend=self.backend)
            expected = self._make_receipt(
                outer_parent=outer_parent, v92_trial=v92_trial,
                face_trial=value.face_trial)
            if expected.receipt_sha256 != value.receipt.receipt_sha256:
                raise Q3V84MixedExnerAdapterV93Error(
                    "v93 mixed CPU receipt no longer matches its live trial")

    def validate_open(self, value, *, outer_parent, v92_trial):
        self._validate_open(
            value, outer_parent=outer_parent, v92_trial=v92_trial)
        return value.receipt

    def prepare_rebound(
        self, value: CPUQ3V84MixedOpenTrialV93, *,
        authoritative_bed_mass_kg_m2: object,
    ) -> CPUQ3V84MixedPreparedCommitV93:
        self._validate_open(value)
        if self._prepared is not None:
            raise RuntimeError("v93 mixed CPU rebound is already prepared")
        bed = _readonly(
            authoritative_bed_mass_kg_m2,
            (self.outer_execution.fraction_count,
             self.outer_execution.cell_count), "authoritative bed",
            nonnegative=True)
        trial = value.face_trial
        rebound = resolve_cpu_q3_v84_mixed_state_v93(
            execution_fingerprint=self.execution_fingerprint,
            binding=self.binding, authoritative_bed_mass_kg_m2=bed,
            bulk_density_kg_m3=self._state.bulk_density_kg_m3,
            retained_fraction=provider_surface_fraction_v114(self),
            accepted_face_flux_kg_m_s=trial.face_flux_kg_m_s,
            accepted_cell_availability=trial.accepted_cell_availability,
            named_boundary_positive_in_kg=(
                self._state.named_boundary_positive_in_kg+
                value.receipt.named_boundary_positive_in_kg),
            named_boundary_absolute_kg=(
                self._state.named_boundary_absolute_kg+
                value.receipt.named_boundary_absolute_kg),
            depletion_roundoff_positive_in_kg=(
                self._state.depletion_roundoff_positive_in_kg+
                value.receipt.depletion_roundoff_positive_in_kg),
            accepted_updates=self._state.accepted_updates+1,
            time_s=self._state.time_s+trial.dt_s)
        prepared = CPUQ3V84MixedPreparedCommitV93(
            value, rebound, _array_digest(bed), "")
        prepared = CPUQ3V84MixedPreparedCommitV93(
            prepared.open_trial, prepared.rebound_state,
            prepared.authoritative_bed_sha256, _prepared_token(prepared))
        self._prepared = prepared
        return prepared

    def validate_prepared(self, value, *, authoritative_bed_mass_kg_m2):
        if (not isinstance(value, CPUQ3V84MixedPreparedCommitV93) or
                value is not self._prepared or
                value.preparation_sha256 != _prepared_token(value)):
            raise Q3V84MixedExnerAdapterV93Error(
                "v93 mixed CPU prepared commit differs")
        bed = _readonly(
            authoritative_bed_mass_kg_m2,
            (self.outer_execution.fraction_count,
             self.outer_execution.cell_count), "authoritative bed",
            nonnegative=True)
        if (value.authoritative_bed_sha256 != _array_digest(bed) or
                not np.array_equal(
                    value.rebound_state.authoritative_bed_mass_kg_m2, bed)):
            raise Q3V84MixedExnerAdapterV93Error(
                "v93 mixed CPU prepared bed differs")
        validate_cpu_q3_v84_mixed_state_v93(
            value.rebound_state,
            execution_fingerprint=self.execution_fingerprint,
            binding=self.binding)
        return value.rebound_state

    def close_commit(self, value):
        if value is not self._prepared:
            raise Q3V84MixedExnerAdapterV93Error(
                "v93 mixed CPU prepared commit differs")
        self._validate_open(value.open_trial)
        validate_cpu_q3_v84_mixed_state_v93(
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
                "v93 mixed CPU closed commit differs")
        self._state = value.rebound_state
        self._closed = None
        return self._state

    def abort_closed_commit(self, value) -> None:
        if value is not self._closed:
            raise Q3V84MixedExnerAdapterV93Error(
                "v93 mixed CPU closed commit differs")
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
                "v93 mixed CPU checkpoint requires a committed point")
        state = self._state
        body = {
            "schema": SCHEMA, "scope_id": CHECKPOINT_SCOPE_ID,
            "outer_execution_fingerprint": self.outer_execution.fingerprint,
            "execution_fingerprint": self.execution_fingerprint,
            "binding_sha256": self.binding.binding_sha256,
            "topology_id": self.binding.topology_id,
            "boundary_names": list(self.boundary_names),
            "state": {
                "authoritative_bed_mass_kg_m2":
                    state.authoritative_bed_mass_kg_m2.tolist(),
                "bulk_density_kg_m3": state.bulk_density_kg_m3.tolist(),
                "active_fraction": state.active_fraction.tolist(),
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
                "state_sha256": state.state_sha256,
            },
            "extra_scalar_lifecycle_application_count": 0,
        }
        return {**body, "fingerprint": _fingerprint(body)}

    def restore_checkpoint(self, value: Mapping[str, object]):
        if self._open is not None or self._prepared is not None or \
                self._closed is not None:
            raise RuntimeError(
                "v93 mixed CPU restart cannot replace an open transaction")
        if not isinstance(value, Mapping):
            raise Q3V84MixedExnerAdapterV93Error(
                "v93 mixed CPU checkpoint is missing")
        body = {key: item for key, item in value.items()
                if key != "fingerprint"}
        if (value.get("fingerprint") != _fingerprint(body) or
                value.get("schema") != SCHEMA or
                value.get("scope_id") != CHECKPOINT_SCOPE_ID or
                value.get("outer_execution_fingerprint") !=
                    self.outer_execution.fingerprint or
                value.get("execution_fingerprint") !=
                    self.execution_fingerprint or
                value.get("binding_sha256") != self.binding.binding_sha256 or
                value.get("topology_id") != self.binding.topology_id or
                value.get("boundary_names") != list(self.boundary_names) or
                value.get("extra_scalar_lifecycle_application_count") != 0):
            raise Q3V84MixedExnerAdapterV93Error(
                "v93 mixed CPU checkpoint identity differs")
        fields = value.get("state")
        if not isinstance(fields, Mapping):
            raise Q3V84MixedExnerAdapterV93Error(
                "v93 mixed CPU checkpoint state is missing")
        state = resolve_cpu_q3_v84_mixed_state_v93(
            execution_fingerprint=self.execution_fingerprint,
            binding=self.binding,
            authoritative_bed_mass_kg_m2=
                fields["authoritative_bed_mass_kg_m2"],
            bulk_density_kg_m3=fields["bulk_density_kg_m3"],
            retained_fraction=fields["active_fraction"],
            accepted_face_flux_kg_m_s=fields["accepted_face_flux_kg_m_s"],
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
        if state.state_sha256 != fields.get("state_sha256"):
            raise Q3V84MixedExnerAdapterV93Error(
                "v93 mixed CPU checkpoint state differs")
        self._state = state
        return state


__all__ = [
    "BED_SLOPE_SCOPE", "BOUNDARY_SCOPE", "CHECKPOINT_SCOPE_ID",
    "CPUQ3V84FaceTrialV93",
    "CPUQ3V84MixedBindingV93", "CPUQ3V84MixedExnerAdapterV93",
    "CPUQ3V84MixedOpenTrialV93", "CPUQ3V84MixedPreparedCommitV93",
    "CPUQ3V84MixedStateV93", "FRACTION_COUNT_SCOPE", "PROVIDER_KIND",
    "Q3V84MixedExnerAdapterV93Error", "SCHEMA", "SCOPE_ID",
    "resolve_cpu_q3_v84_mixed_binding_v93",
    "resolve_cpu_q3_v84_mixed_state_v93",
    "validate_cpu_q3_v84_mixed_state_v93",
]
