"""Authenticated Q3/v75 bed-step adapter for the Q4/v93 Exner owner.

This module is deliberately additive.  It does not change the accepted v75
execution, trial, or state schemas.  Instead it owns one real v75 invocation,
keeps that child transaction open while the outer Q4 transaction is assessed,
and publishes a typed bed-step receipt derived from the authenticated child
trial.  The only admitted morphology factor is one; the later morphology-clock
owner will widen that contract.

The adapter also provides a projection boundary for the next accepted update.
After the outer owner has formed its authoritative bed, the committed v75 child
is rebuilt on that bed while retaining its own cumulative boundary ledgers,
accepted clock, topology epoch, and diagnostic registers.  The original v75
trial and candidate remain untouched.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from collections.abc import Callable, Mapping

import numpy as np

from .direct_vector_family_live_v67 import (
    resolve_direct_vector_family_state_v67,
)
from .direct_vector_wetdry_amr_live_v71 import FACE_INTERNAL
from .route_lifecycle_composition_v81 import ROUTE_OFF, ROUTE_TOTAL
from .transported_fluid_settling_v92 import (
    TransportedFluidTrialV92,
    _validate_trial as _validate_v92_trial,
)
from .unified_direct_vector_transaction_v75 import (
    CHECKPOINT_SCOPE_ID as V75_CHECKPOINT_SCOPE_ID,
    CPUUnifiedBindingV75,
    CPUUnifiedStateV75,
    CPUUnifiedTrialV75,
    PortableUnifiedDirectVectorV75,
)


SCHEMA = 1
SCOPE_ID = "braids-q4-q3-v75-exner-adapter-v93-v1"
CHECKPOINT_SCOPE_ID = SCOPE_ID + ":checkpoint"
BACKEND_CPU = "cpu_oracle"
MORPHOLOGICAL_FACTOR = 1.0


class Q3V75ExnerAdapterV93Error(ValueError):
    """A live v75 child, outer binding, or projection is invalid."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, OverflowError) as error:
        raise Q3V75ExnerAdapterV93Error(
            "v93 Q3/v75 value is not finite canonical JSON") from error


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _array_digest(*values: object) -> str:
    digest = hashlib.sha256()
    for raw in values:
        value = np.ascontiguousarray(np.asarray(raw))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(_canonical(list(value.shape)))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _finite(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)):
        raise Q3V75ExnerAdapterV93Error(
            f"v93 Q3/v75 {name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise Q3V75ExnerAdapterV93Error(
            f"v93 Q3/v75 {name} is outside its finite domain")
    return result


def _host_array(value: object, shape: tuple[int, ...], name: str, *,
                integer: bool = False, nonnegative: bool = False) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != shape or raw.dtype.kind not in "fiu":
        raise Q3V75ExnerAdapterV93Error(
            f"v93 Q3/v75 {name} must be numeric with shape {shape}")
    result = np.array(
        raw, dtype=np.int32 if integer else np.float64,
        copy=True, order="C")
    if not np.isfinite(result).all():
        raise Q3V75ExnerAdapterV93Error(
            f"v93 Q3/v75 {name} must be finite")
    if nonnegative and np.any(result < 0.0):
        raise Q3V75ExnerAdapterV93Error(
            f"v93 Q3/v75 {name} must be nonnegative")
    result.setflags(write=False)
    return result


def _base_topology(outer_execution):
    """Return the authenticated v76 topology nested below v92."""
    try:
        return (outer_execution.base_execution.base_execution.base_execution.
                base_execution)
    except AttributeError as error:
        raise Q3V75ExnerAdapterV93Error(
            "v93 Q3/v75 outer execution does not expose the v92 topology") \
            from error


def _shared_topology_sha256(base) -> str:
    return _array_digest(
        base.cell_area_m2, base.face_length_m, base.owner, base.neighbour,
        base.boundary_index, base.cell_face_offset, base.cell_faces,
        base.cell_face_sign, base.normal,
    )


def _v92_physical_sha256(outer_execution) -> str:
    base = _base_topology(outer_execution)
    profile = base.profile
    return _fingerprint({
        "ordered_fraction_ids": list(outer_execution.ordered_fraction_ids),
        "d50_m": list(profile.sediment_d50_m),
        "solid_density_kg_m3": list(profile.sediment_density_kg_m3),
        "water_density_kg_m3": profile.water_density_kg_m3,
        "gravity_m_s2": profile.gravity_m_s2,
    })


def _outer_execution_document(outer_execution) -> Mapping[str, object]:
    # Local import keeps this additive module importable by a future v93
    # coordinator without creating an import cycle.
    from .suspended_exner_owner_v93 import execution_document_v93
    return execution_document_v93(outer_execution)


def _validate_outer_state(outer_state, outer_execution) -> None:
    from .suspended_exner_owner_v93 import validate_suspended_exner_state_v93
    validate_suspended_exner_state_v93(outer_state, outer_execution)


def _surface_fraction(
    bed_mass_kg_m2: np.ndarray,
    bulk_density_kg_m3: np.ndarray,
    retained_fraction: np.ndarray,
) -> np.ndarray:
    volume = bed_mass_kg_m2 / bulk_density_kg_m3[:, None]
    total = np.sum(volume, axis=0)
    positive = total > 0.0
    result = np.array(retained_fraction, dtype=np.float64,
                      copy=True, order="C")
    result[:, positive] = volume[:, positive] / total[positive][None, :]
    result.setflags(write=False)
    return result


def _receipt_token(value: "Q3V75ExnerReceiptV93") -> str:
    return _fingerprint({
        "scope_id": SCOPE_ID,
        "execution_fingerprint": value.execution_fingerprint,
        "parent_state_sha256": value.parent_state_sha256,
        "accepted_v92_trial_sha256": value.accepted_v92_trial_sha256,
        "accepted_hydraulic_receipt_sha256":
            value.accepted_hydraulic_receipt_sha256,
        "accepted_hydraulic_face_receipt_sha256":
            value.accepted_hydraulic_face_receipt_sha256,
        "q3_execution_fingerprint": value.q3_execution_fingerprint,
        "q3_binding_sha256": value.q3_binding_sha256,
        "q3_topology_id": value.q3_topology_id,
        "q3_parent_state_sha256": value.q3_parent_state_sha256,
        "q3_candidate_state_sha256": value.q3_candidate_state_sha256,
        "source_trial_sha256": value.source_trial_sha256,
        "source_face_flux_sha256": value.source_face_flux_sha256,
        "shared_topology_sha256": value.shared_topology_sha256,
        "ordered_fraction_ids": list(value.ordered_fraction_ids),
        "physical_profile_sha256": value.physical_profile_sha256,
        "dt_s": value.dt_s,
        "morphological_factor": value.morphological_factor,
        "boundary_names": list(value.boundary_names),
        "arrays_sha256": _array_digest(
            value.routes, value.bed_delta_kg_m2,
            value.named_boundary_positive_in_kg,
            value.named_boundary_absolute_kg,
            value.depletion_roundoff_positive_in_kg,
            value.boundary_ledger_kg, value.q3_owner_count),
        "backend": value.backend,
        "provider_kind": value.provider_kind,
        "positive_boundary_sign": "into_domain",
        "depletion_is_not_a_named_boundary": True,
        "ownership_arrays_sha256": _array_digest(
            value.constitutive_producer_count, value.face_exner_count,
            value.extra_scalar_lifecycle_count),
    })


@dataclass(frozen=True)
class Q3V75ExnerReceiptV93:
    """Immutable producer-owned v75 contribution consumed by outer Exner."""

    execution_fingerprint: str
    parent_state_sha256: str
    accepted_v92_trial_sha256: str
    accepted_hydraulic_receipt_sha256: str
    accepted_hydraulic_face_receipt_sha256: str
    q3_execution_fingerprint: str
    q3_binding_sha256: str
    q3_topology_id: str
    q3_parent_state_sha256: str
    q3_candidate_state_sha256: str
    source_trial_sha256: str
    source_face_flux_sha256: str
    shared_topology_sha256: str
    ordered_fraction_ids: tuple[str, ...]
    physical_profile_sha256: str
    dt_s: float
    morphological_factor: float
    boundary_names: tuple[str, ...]
    routes: np.ndarray
    bed_delta_kg_m2: np.ndarray
    named_boundary_positive_in_kg: np.ndarray
    named_boundary_absolute_kg: np.ndarray
    depletion_roundoff_positive_in_kg: np.ndarray
    boundary_ledger_kg: np.ndarray
    q3_owner_count: np.ndarray
    backend: str
    provider_kind: str
    constitutive_producer_count: np.ndarray
    face_exner_count: np.ndarray
    extra_scalar_lifecycle_count: np.ndarray
    receipt_sha256: str

    @property
    def additional_scalar_lifecycle_count(self) -> np.ndarray:
        """Coordinator spelling for the same zero-owner counter."""
        return self.extra_scalar_lifecycle_count


# Neutral consumer ABI used by the outer Q4 owner.  The historical class name
# remains an alias so the all-TOTAL v75 provider and the mixed v84 provider can
# share one immutable receipt without pretending they share a child lifecycle.
Q3ExnerReceiptV93 = Q3V75ExnerReceiptV93


def validate_q3_v75_exner_receipt_v93(
    value: Q3V75ExnerReceiptV93, *, outer_execution,
    outer_parent, v92_trial: TransportedFluidTrialV92,
    q3_binding_sha256: str, q3_topology_id: str, backend: str,
) -> None:
    """Validate immutable evidence; live child validation remains adapter-owned."""
    _outer_execution_document(outer_execution)
    _validate_outer_state(outer_parent, outer_execution)
    ng = int(outer_execution.fraction_count)
    nc = int(outer_execution.cell_count)
    if not isinstance(value, Q3V75ExnerReceiptV93):
        raise Q3V75ExnerAdapterV93Error(
            "v93 Q3/v75 typed receipt is missing")
    arrays = (
        (value.routes, (ng,), True),
        (value.bed_delta_kg_m2, (ng, nc), False),
        (value.named_boundary_positive_in_kg,
         (ng, len(value.boundary_names)), False),
        (value.named_boundary_absolute_kg,
         (ng, len(value.boundary_names)), False),
        (value.depletion_roundoff_positive_in_kg, (ng,), False),
        (value.boundary_ledger_kg, (ng,), False),
        (value.q3_owner_count, (ng,), True),
        (value.constitutive_producer_count, (ng,), True),
        (value.face_exner_count, (ng,), True),
        (value.extra_scalar_lifecycle_count, (ng,), True),
    )
    for array, shape, integer in arrays:
        raw = np.asarray(array)
        if (raw.shape != shape or raw.dtype.kind not in ("iu" if integer else
                                                          "fiu") or
                not np.isfinite(raw).all()):
            raise Q3V75ExnerAdapterV93Error(
                "v93 Q3/v75 typed receipt arrays differ")
    expected_common = (
        outer_execution.fingerprint, outer_parent.state_sha256,
        v92_trial.trial_sha256,
        v92_trial.hydraulic_receipt.receipt_sha256,
        v92_trial.hydraulic_face_receipt.receipt_sha256,
        outer_execution.q3_execution_fingerprint,
        q3_binding_sha256, q3_topology_id, backend,
    )
    actual_common = (
        value.execution_fingerprint, value.parent_state_sha256,
        value.accepted_v92_trial_sha256,
        value.accepted_hydraulic_receipt_sha256,
        value.accepted_hydraulic_face_receipt_sha256,
        value.q3_execution_fingerprint, value.q3_binding_sha256,
        value.q3_topology_id, value.backend,
    )
    active = np.asarray(outer_execution.routes) != ROUTE_OFF
    expected_boundary = (
        np.sum(value.named_boundary_positive_in_kg, axis=1) +
        value.depletion_roundoff_positive_in_kg)
    expected_producer = (
        np.zeros(ng, dtype=np.int32)
        if value.provider_kind == "portable_oracle_v75_all_total"
        else active.astype(np.int32)
    )
    if (actual_common != expected_common or
            value.dt_s != v92_trial.hydraulic_receipt.dt_s or
            value.morphological_factor != MORPHOLOGICAL_FACTOR or
            value.receipt_sha256 != _receipt_token(value) or
            value.shared_topology_sha256 !=
                _shared_topology_sha256(_base_topology(outer_execution)) or
            value.ordered_fraction_ids !=
                tuple(outer_execution.ordered_fraction_ids) or
            value.physical_profile_sha256 !=
                _v92_physical_sha256(outer_execution) or
            value.boundary_names != tuple(outer_execution.q3_boundary_names) or
            len(set(value.boundary_names)) != len(value.boundary_names) or
            not np.array_equal(value.routes, outer_execution.routes) or
            not np.array_equal(value.q3_owner_count,
                               active.astype(np.int32)) or
            not np.array_equal(value.constitutive_producer_count,
                               expected_producer) or
            not np.array_equal(value.face_exner_count,
                               active.astype(np.int32)) or
            np.any(value.extra_scalar_lifecycle_count != 0) or
            not np.array_equal(value.boundary_ledger_kg,
                               expected_boundary) or
            np.any(value.depletion_roundoff_positive_in_kg < 0.0) or
            np.any(value.named_boundary_absolute_kg <
                   np.abs(value.named_boundary_positive_in_kg)) or
            np.any(value.bed_delta_kg_m2[~active] != 0.0) or
            np.any(value.named_boundary_positive_in_kg[~active] != 0.0) or
            np.any(value.named_boundary_absolute_kg[~active] != 0.0) or
            np.any(value.depletion_roundoff_positive_in_kg[~active] != 0.0)):
        raise Q3V75ExnerAdapterV93Error(
            "v93 Q3/v75 typed receipt is stale or cross-trial")


@dataclass(frozen=True)
class CPUQ3V75OpenTrialV93:
    child_parent: CPUUnifiedStateV75
    child_trial: CPUUnifiedTrialV75
    receipt: Q3V75ExnerReceiptV93
    handle_sha256: str


@dataclass(frozen=True)
class CPUQ3V75PreparedCommitV93:
    open_trial: CPUQ3V75OpenTrialV93
    rebound_state: CPUUnifiedStateV75
    authoritative_bed_sha256: str
    preparation_sha256: str

    @property
    def rebound_state_sha256(self) -> str:
        return self.rebound_state.state_sha256

    @property
    def rebound_bed_mass_kg_m2(self) -> np.ndarray:
        return self.rebound_state.direct_state.bed_mass_kg_m2


def _open_token(value: CPUQ3V75OpenTrialV93) -> str:
    return _fingerprint({
        "scope_id": SCOPE_ID + ":cpu-open",
        "child_parent": value.child_parent.state_sha256,
        "child_trial": value.child_trial.trial_sha256,
        "receipt": value.receipt.receipt_sha256,
    })


def _prepared_token(value: CPUQ3V75PreparedCommitV93) -> str:
    return _fingerprint({
        "scope_id": SCOPE_ID + ":cpu-prepared",
        "open": value.open_trial.handle_sha256,
        "rebound": value.rebound_state.state_sha256,
        "authoritative_bed": value.authoritative_bed_sha256,
    })


class CPUQ3V75ExnerAdapterV93:
    """Own one live portable-v75 child transaction at a time."""

    backend = BACKEND_CPU
    provider_kind = "portable_oracle_v75_all_total"

    def __init__(
        self, *, outer_execution,
        runtime: PortableUnifiedDirectVectorV75,
        binding: CPUUnifiedBindingV75,
        state: CPUUnifiedStateV75,
    ):
        _outer_execution_document(outer_execution)
        if (outer_execution.backend != "cpu" or
                type(runtime) is not PortableUnifiedDirectVectorV75 or
                type(binding) is not CPUUnifiedBindingV75 or
                type(state) is not CPUUnifiedStateV75):
            raise Q3V75ExnerAdapterV93Error(
                "v93 Q3/v75 CPU adapter requires the explicit CPU objects")
        fraction_ids = tuple(runtime.execution["fraction_ids"])
        if (runtime.execution["fingerprint"] !=
                outer_execution.q3_execution_fingerprint or
                binding.topology_id != outer_execution.q3_topology_id or
                binding.binding_sha256 != outer_execution.q3_binding_sha256 or
                state.topology_id != binding.topology_id or
                fraction_ids != tuple(outer_execution.ordered_fraction_ids) or
                np.any(np.asarray(outer_execution.routes) != ROUTE_TOTAL)):
            raise Q3V75ExnerAdapterV93Error(
                "v93 Q3/v75 CPU requires the exact all-TOTAL execution, "
                "topology, binding, and fraction order")
        PortableUnifiedDirectVectorV75._validate_state(
            runtime, state, binding)
        self.outer_execution = outer_execution
        self.runtime = runtime
        self.binding = binding
        self.boundary_names = tuple(
            row["name"] for row in binding.boundary_execution["boundaries"])
        if (self.boundary_names != tuple(outer_execution.q3_boundary_names) or
                len(set(self.boundary_names)) != len(self.boundary_names)):
            raise Q3V75ExnerAdapterV93Error(
                "v93 Q3/v75 CPU logical boundary names differ")
        self._validate_topology()
        self._state = state
        self._open: CPUQ3V75OpenTrialV93 | None = None
        self._prepared: CPUQ3V75PreparedCommitV93 | None = None
        self._closed: CPUQ3V75PreparedCommitV93 | None = None

    @property
    def state(self) -> CPUUnifiedStateV75:
        return self._state

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
        return tuple(self.runtime.execution["fraction_ids"])

    @property
    def state_sha256(self) -> str:
        return self._state.state_sha256

    @property
    def accepted_updates(self) -> int:
        return self._state.direct_state.accepted_step

    @property
    def time_s(self) -> float:
        return self._state.direct_state.time_s

    @property
    def authoritative_bed_mass_kg_m2(self) -> np.ndarray:
        return self._state.direct_state.bed_mass_kg_m2

    def validate_current_bed_clock(self, outer_parent) -> None:
        _validate_outer_state(outer_parent, self.outer_execution)
        self._validate_child_parent(outer_parent)

    def _validate_topology(self) -> None:
        base = _base_topology(self.outer_execution)
        topology = self.binding.wetdry_topology
        family_faces = np.asarray(
            self.binding.family_boundary_faces, dtype=np.int64)
        local_boundary = np.asarray(
            self.binding.boundary_topology.boundary_index, dtype=np.int64)
        expected_boundary_faces = np.flatnonzero(base.boundary_index >= 0)
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
            np.array_equal(family_faces, expected_boundary_faces),
            np.array_equal(local_boundary,
                           base.boundary_index[family_faces]),
            bool(np.all(topology.wet)),
        )
        if not all(checks):
            raise Q3V75ExnerAdapterV93Error(
                "v93 Q3/v75 CPU topology differs from the v92 owner")
        physical = self.runtime.rows[self.binding.topology_id][
            "live_taurat_execution"]["parent_execution"]["physical_profile"]
        profile = base.profile
        if (tuple(physical["fraction_ids"]) !=
                tuple(self.outer_execution.ordered_fraction_ids) or
                not np.array_equal(np.asarray(physical["d50_m"]),
                                   np.asarray(profile.sediment_d50_m)) or
                not np.array_equal(
                    np.asarray(physical["solid_density_kg_m3"]),
                    np.asarray(profile.sediment_density_kg_m3)) or
                float(physical["water_density_kg_m3"]) !=
                    float(profile.water_density_kg_m3) or
                float(physical["gravity_m_s2"]) !=
                    float(profile.gravity_m_s2)):
            raise Q3V75ExnerAdapterV93Error(
                "v93 Q3/v75 CPU physical fraction profile differs from v92")

    def _validate_child_parent(self, outer_parent) -> None:
        direct = self._state.direct_state
        if (not np.array_equal(direct.bed_mass_kg_m2,
                               outer_parent.bed_mass_kg_m2) or
                direct.accepted_step != outer_parent.accepted_updates or
                direct.time_s != outer_parent.time_s):
            raise Q3V75ExnerAdapterV93Error(
                "v93 Q3/v75 CPU child was not rebound to the authoritative bed")

    def _reconstruct(
        self, trial: CPUUnifiedTrialV75, dt_s: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        flux = np.asarray(trial.face_flux_kg_m_s)
        topology = self.binding.wetdry_topology
        ng, nf = flux.shape
        delta = np.zeros((ng, topology.ncell), dtype=np.float64)
        named = np.zeros((ng, len(self.boundary_names)), dtype=np.float64)
        absolute = np.zeros_like(named)
        inverse = {
            int(family): local
            for local, family in enumerate(self.binding.family_boundary_faces)
        }
        for face in range(nf):
            amount = (flux[:, face]*self.binding.face_length_m[face]*dt_s)
            owner = int(topology.owner[face])
            delta[:, owner] -= amount/topology.cell_area_m2[owner]
            if int(topology.role[face]) == FACE_INTERNAL:
                neighbour = int(topology.neighbour[face])
                delta[:, neighbour] += amount/topology.cell_area_m2[neighbour]
            else:
                local = inverse.get(face)
                if local is None:
                    raise Q3V75ExnerAdapterV93Error(
                        "v93 Q3/v75 CPU boundary face lacks a named owner")
                boundary = int(
                    self.binding.boundary_topology.boundary_index[local])
                named[:, boundary] += amount
                absolute[:, boundary] += np.abs(amount)
        return tuple(np.ascontiguousarray(item)
                     for item in (delta, named, absolute))

    def _make_receipt(
        self, outer_parent, v92_trial: TransportedFluidTrialV92,
        trial: CPUUnifiedTrialV75,
    ) -> Q3V75ExnerReceiptV93:
        dt = _finite(v92_trial.hydraulic_receipt.dt_s, "accepted dt",
                     positive=True)
        direct_parent = self._state.direct_state
        direct_candidate = trial.candidate.direct_state
        if (direct_candidate.accepted_step != direct_parent.accepted_step+1 or
                direct_candidate.time_s != direct_parent.time_s+dt):
            raise Q3V75ExnerAdapterV93Error(
                "v93 Q3/v75 CPU child clock differs from v92")
        reconstructed, named_out, absolute = self._reconstruct(trial, dt)
        if (not np.array_equal(reconstructed, trial.cell_delta_kg_m2) or
                not np.array_equal(named_out,
                                   trial.named_boundary_transfer_kg) or
                not np.array_equal(absolute,
                                   trial.named_boundary_absolute_kg)):
            raise Q3V75ExnerAdapterV93Error(
                "v93 Q3/v75 CPU trial is not an MF=1 face/Exner result")
        routes = _host_array(
            self.outer_execution.routes,
            (self.outer_execution.fraction_count,), "routes", integer=True)
        delta = _host_array(
            trial.cell_delta_kg_m2,
            (self.outer_execution.fraction_count,
             self.outer_execution.cell_count), "bed delta")
        positive_in = _host_array(
            -np.asarray(trial.named_boundary_transfer_kg),
            (self.outer_execution.fraction_count, len(self.boundary_names)),
            "named positive-in ledger")
        named_absolute = _host_array(
            trial.named_boundary_absolute_kg, positive_in.shape,
            "named absolute ledger", nonnegative=True)
        # The portable v75 oracle does not snap and ledger numerical depletion;
        # its authoritative outer contribution is the authenticated delta.
        depletion = _host_array(
            np.zeros(self.outer_execution.fraction_count),
            (self.outer_execution.fraction_count,),
            "depletion roundoff positive-in ledger")
        boundary = _host_array(
            np.sum(positive_in, axis=1)+depletion,
            (self.outer_execution.fraction_count,), "global boundary ledger")
        active = routes != ROUTE_OFF
        owner = _host_array(active.astype(np.int32), routes.shape,
                            "Q3 owner count", integer=True)
        area = _base_topology(self.outer_execution).cell_area_m2
        integrated = np.sum(delta*area[None, :], axis=1)
        tolerance = 8192.0*np.finfo(np.float64).eps*np.maximum(
            1.0, np.abs(integrated)+np.abs(boundary))
        if (np.any(np.abs(integrated-boundary) > tolerance) or
                np.any(delta[~active] != 0.0) or
                np.any(positive_in[~active] != 0.0)):
            raise Q3V75ExnerAdapterV93Error(
                "v93 Q3/v75 CPU route or positive-in ledger differs")
        placeholder = Q3V75ExnerReceiptV93(
            self.outer_execution.fingerprint, outer_parent.state_sha256,
            v92_trial.trial_sha256,
            v92_trial.hydraulic_receipt.receipt_sha256,
            v92_trial.hydraulic_face_receipt.receipt_sha256,
            self.runtime.execution["fingerprint"],
            self.binding.binding_sha256, self.binding.topology_id,
            self._state.state_sha256, trial.candidate.state_sha256,
            trial.trial_sha256, _array_digest(trial.face_flux_kg_m_s),
            _shared_topology_sha256(_base_topology(self.outer_execution)),
            tuple(self.outer_execution.ordered_fraction_ids),
            _v92_physical_sha256(self.outer_execution),
            dt, MORPHOLOGICAL_FACTOR, self.boundary_names, routes, delta,
            positive_in, named_absolute, depletion, boundary, owner,
            self.backend, self.provider_kind,
            _host_array(np.zeros_like(owner), owner.shape,
                        "constitutive producer count", integer=True), owner,
            _host_array(np.zeros_like(owner), owner.shape,
                        "extra scalar lifecycle count", integer=True), "")
        return Q3V75ExnerReceiptV93(
            **{**placeholder.__dict__,
               "receipt_sha256": _receipt_token(placeholder)})

    def _validate_open(
        self, value: CPUQ3V75OpenTrialV93, *, outer_parent=None,
        v92_trial: TransportedFluidTrialV92 | None = None,
    ) -> None:
        if (not isinstance(value, CPUQ3V75OpenTrialV93) or
                value is not self._open or
                value.child_parent is not self._state or
                value.handle_sha256 != _open_token(value)):
            raise Q3V75ExnerAdapterV93Error(
                "v93 Q3/v75 CPU open-trial handle differs")
        PortableUnifiedDirectVectorV75._validate_trial(
            self.runtime, value.child_parent, self.binding,
            value.child_trial)
        if outer_parent is not None and v92_trial is not None:
            validate_q3_v75_exner_receipt_v93(
                value.receipt, outer_execution=self.outer_execution,
                outer_parent=outer_parent, v92_trial=v92_trial,
                q3_binding_sha256=self.binding.binding_sha256,
                q3_topology_id=self.binding.topology_id,
                backend=self.backend)
            expected = self._make_receipt(
                outer_parent, v92_trial, value.child_trial)
            if expected.receipt_sha256 != value.receipt.receipt_sha256:
                raise Q3V75ExnerAdapterV93Error(
                    "v93 Q3/v75 CPU open handle no longer matches its live child")

    def begin(
        self, *, outer_parent, parent_hles, handoff,
        v92_trial: TransportedFluidTrialV92,
        capacity, downhill, depth, mean_diameter, hiding,
        dry_tolerance: float, dx_m: float, dy_m: float,
        cancelled: Callable[[], bool] = lambda: False,
    ) -> CPUQ3V75OpenTrialV93:
        """Open the real v75 child using v92-owned q, flux, dt, and MF=1."""
        if not callable(cancelled):
            raise TypeError("cancelled must be callable")
        if self._open is not None or self._prepared is not None or \
                self._closed is not None:
            raise RuntimeError("v93 Q3/v75 CPU adapter already has a transaction")
        _validate_outer_state(outer_parent, self.outer_execution)
        _validate_v92_trial(
            self.outer_execution.base_execution, outer_parent.base_state,
            parent_hles, handoff, v92_trial)
        if v92_trial.backend != BACKEND_CPU:
            raise Q3V75ExnerAdapterV93Error(
                "v93 Q3/v75 CPU adapter requires the CPU v92 trial")
        self._validate_child_parent(outer_parent)
        dt = _finite(v92_trial.hydraulic_receipt.dt_s, "accepted dt",
                     positive=True)
        mapping = np.asarray(self.binding.family_boundary_faces,
                             dtype=np.int64)
        hydraulic_boundary = np.ascontiguousarray(
            v92_trial.hydraulic_face_receipt.accepted_water_flux_m2_s[mapping])
        child = None
        try:
            child = PortableUnifiedDirectVectorV75.begin_trial(
                self.runtime,
                state=self._state, binding=self.binding,
                # Q3 constitutive production and its wet/adaptation gates are
                # endpoint consumers.  The midpoint belongs to the implicit
                # hydraulic residual, whereas candidate.q is the endpoint
                # authenticated by both the v91 hydraulic receipt and v92's
                # accepted face-flux receipt.
                capacity=capacity, q=v92_trial.candidate.q,
                downhill=downhill, depth=depth,
                mean_diameter=mean_diameter, hiding=hiding,
                hydraulic_outward_flux=hydraulic_boundary,
                dry_tolerance=dry_tolerance, dx_m=dx_m, dy_m=dy_m,
                dt_s=dt, morphological_factor=MORPHOLOGICAL_FACTOR,
                cancelled=cancelled)
            PortableUnifiedDirectVectorV75._validate_trial(
                self.runtime, self._state, self.binding, child)
            receipt = self._make_receipt(outer_parent, v92_trial, child)
            handle = CPUQ3V75OpenTrialV93(
                self._state, child, receipt, "")
            handle = CPUQ3V75OpenTrialV93(
                handle.child_parent, handle.child_trial, handle.receipt,
                _open_token(handle))
            self._open = handle
            return handle
        except Exception:
            if child is not None and self.runtime._open_parent == \
                    self._state.state_sha256:
                try:
                    PortableUnifiedDirectVectorV75.rollback(
                        self.runtime, self._state, self.binding, child)
                except Exception as cleanup_error:
                    raise RuntimeError(
                        "v93 Q3/v75 CPU failed and child rollback also failed") \
                        from cleanup_error
            raise

    def validate_open(
        self, value: CPUQ3V75OpenTrialV93, *, outer_parent,
        v92_trial: TransportedFluidTrialV92,
    ) -> Q3V75ExnerReceiptV93:
        self._validate_open(
            value, outer_parent=outer_parent, v92_trial=v92_trial)
        return value.receipt

    def prepare_rebound(
        self, value: CPUQ3V75OpenTrialV93, *,
        authoritative_bed_mass_kg_m2: object,
    ) -> CPUQ3V75PreparedCommitV93:
        """Validate and build the next child state without closing its trial."""
        self._validate_open(value)
        if self._prepared is not None:
            raise RuntimeError("v93 Q3/v75 CPU rebound is already prepared")
        ng = self.outer_execution.fraction_count
        nc = self.outer_execution.cell_count
        bed = _host_array(authoritative_bed_mass_kg_m2, (ng, nc),
                          "authoritative bed", nonnegative=True)
        committed = value.child_trial.candidate
        direct = committed.direct_state
        fraction = _surface_fraction(
            np.asarray(bed), np.asarray(direct.bulk_density_kg_m3),
            np.asarray(direct.active_fraction))
        parent_execution = self.runtime.rows[self.binding.topology_id][
            "live_taurat_execution"]["parent_execution"]
        rebound_direct = resolve_direct_vector_family_state_v67(
            execution=parent_execution, bed_mass_kg_m2=bed,
            active_fraction=fraction,
            boundary_signed_kg=direct.boundary_signed_kg,
            boundary_absolute_kg=direct.boundary_absolute_kg,
            accepted_step=direct.accepted_step, time_s=direct.time_s)
        rebound = self.runtime._make_state(
            self.binding, rebound_direct, mesh_epoch=committed.mesh_epoch,
            accepted_face_flux=committed.accepted_face_flux,
            accepted_cell_availability=committed.accepted_cell_availability)
        PortableUnifiedDirectVectorV75._validate_state(
            self.runtime, rebound, self.binding)
        prepared = CPUQ3V75PreparedCommitV93(
            value, rebound, _array_digest(bed), "")
        prepared = CPUQ3V75PreparedCommitV93(
            prepared.open_trial, prepared.rebound_state,
            prepared.authoritative_bed_sha256, _prepared_token(prepared))
        self._prepared = prepared
        return prepared

    def validate_prepared(
        self, value: CPUQ3V75PreparedCommitV93, *,
        authoritative_bed_mass_kg_m2: object,
    ) -> CPUUnifiedStateV75:
        if (not isinstance(value, CPUQ3V75PreparedCommitV93) or
                value is not self._prepared or
                value.preparation_sha256 != _prepared_token(value)):
            raise Q3V75ExnerAdapterV93Error(
                "v93 Q3/v75 CPU prepared commit differs")
        expected = _host_array(
            authoritative_bed_mass_kg_m2,
            (self.outer_execution.fraction_count,
             self.outer_execution.cell_count),
            "authoritative bed", nonnegative=True)
        if (value.authoritative_bed_sha256 != _array_digest(expected) or
                not np.array_equal(
                    value.rebound_state.direct_state.bed_mass_kg_m2,
                    expected)):
            raise Q3V75ExnerAdapterV93Error(
                "v93 Q3/v75 CPU prepared bed differs from the outer owner")
        PortableUnifiedDirectVectorV75._validate_state(
            self.runtime, value.rebound_state, self.binding)
        return value.rebound_state

    def close_commit(
        self, value: CPUQ3V75PreparedCommitV93,
    ) -> CPUUnifiedStateV75:
        """Close v75 after prevalidation; publication remains a separate step."""
        if (not isinstance(value, CPUQ3V75PreparedCommitV93) or
                value is not self._prepared or
                value.preparation_sha256 != _prepared_token(value)):
            raise Q3V75ExnerAdapterV93Error(
                "v93 Q3/v75 CPU prepared commit differs")
        self._validate_open(value.open_trial)
        if value.rebound_state.state_sha256 != \
                self.runtime._state_token(
                    self.binding, value.rebound_state.direct_state,
                    value.rebound_state.mesh_epoch,
                    value.rebound_state.accepted_face_flux,
                    value.rebound_state.accepted_cell_availability):
            raise Q3V75ExnerAdapterV93Error(
                "v93 Q3/v75 CPU rebound candidate differs before commit")
        committed = PortableUnifiedDirectVectorV75.commit(
            self.runtime, self._state, self.binding,
            value.open_trial.child_trial)
        self._open = None
        self._prepared = None
        if committed.state_sha256 != \
                value.open_trial.child_trial.candidate.state_sha256:
            self._closed = None
            raise Q3V75ExnerAdapterV93Error(
                "v93 Q3/v75 CPU child commit returned another candidate")
        self._closed = value
        return value.rebound_state

    def publish_commit(
        self, value: CPUQ3V75PreparedCommitV93,
    ) -> CPUUnifiedStateV75:
        """Publish only after the outer v92/v93 commit has also succeeded."""
        if value is not self._closed:
            raise Q3V75ExnerAdapterV93Error(
                "v93 Q3/v75 CPU closed commit differs")
        self._state = value.rebound_state
        self._closed = None
        return self._state

    def abort_closed_commit(self, value: CPUQ3V75PreparedCommitV93) -> None:
        """Discard an unpublished child candidate after an outer commit error."""
        if value is not self._closed:
            raise Q3V75ExnerAdapterV93Error(
                "v93 Q3/v75 CPU closed commit differs")
        # v75 stores accepted state in the caller, not the runtime.  The old
        # self._state is therefore still authoritative and no mutation needs
        # undoing after the child open marker was cleared.
        self._closed = None

    def commit(
        self, value: CPUQ3V75OpenTrialV93, *,
        authoritative_bed_mass_kg_m2: object,
    ) -> CPUUnifiedStateV75:
        """Convenience atomic child-only commit for focused tests."""
        prepared = self.prepare_rebound(
            value,
            authoritative_bed_mass_kg_m2=authoritative_bed_mass_kg_m2)
        self.close_commit(prepared)
        return self.publish_commit(prepared)

    def rollback(self, value: CPUQ3V75OpenTrialV93) -> CPUUnifiedStateV75:
        self._validate_open(value)
        rolled = PortableUnifiedDirectVectorV75.rollback(
            self.runtime, self._state, self.binding, value.child_trial)
        if rolled is not self._state:
            raise Q3V75ExnerAdapterV93Error(
                "v93 Q3/v75 CPU rollback changed its parent")
        self._open = None
        self._prepared = None
        return self._state

    def cancel(self, value: CPUQ3V75OpenTrialV93) -> CPUUnifiedStateV75:
        return self.rollback(value)

    def checkpoint(self) -> dict[str, object]:
        if self._open is not None or self._prepared is not None or \
                self._closed is not None:
            raise RuntimeError(
                "v93 Q3/v75 CPU checkpoint requires a committed point")
        child = self.runtime.checkpoint(self._state, self.binding)
        body = {
            "schema": SCHEMA,
            "scope_id": CHECKPOINT_SCOPE_ID,
            "outer_execution_fingerprint":
                self.outer_execution.fingerprint,
            "q3_execution_fingerprint": self.runtime.execution["fingerprint"],
            "q3_binding_sha256": self.binding.binding_sha256,
            "q3_topology_id": self.binding.topology_id,
            "boundary_names": list(self.boundary_names),
            "v75_checkpoint_scope_id": V75_CHECKPOINT_SCOPE_ID,
            "v75_checkpoint": child,
            "state_sha256": self._state.state_sha256,
            "authoritative_rebind_required_each_update": True,
        }
        return {**body, "fingerprint": _fingerprint(body)}

    def restore_checkpoint(self, value: Mapping[str, object]) -> CPUUnifiedStateV75:
        if self._open is not None or self._prepared is not None or \
                self._closed is not None:
            raise RuntimeError(
                "v93 Q3/v75 CPU restart cannot replace an open transaction")
        if not isinstance(value, Mapping):
            raise Q3V75ExnerAdapterV93Error(
                "v93 Q3/v75 CPU checkpoint is missing")
        body = {key: item for key, item in value.items()
                if key != "fingerprint"}
        if (value.get("fingerprint") != _fingerprint(body) or
                value.get("schema") != SCHEMA or
                value.get("scope_id") != CHECKPOINT_SCOPE_ID or
                value.get("outer_execution_fingerprint") !=
                    self.outer_execution.fingerprint or
                value.get("q3_execution_fingerprint") !=
                    self.runtime.execution["fingerprint"] or
                value.get("q3_binding_sha256") !=
                    self.binding.binding_sha256 or
                value.get("q3_topology_id") != self.binding.topology_id or
                value.get("boundary_names") != list(self.boundary_names) or
                value.get("v75_checkpoint_scope_id") !=
                    V75_CHECKPOINT_SCOPE_ID or
                value.get("authoritative_rebind_required_each_update") is not
                    True):
            raise Q3V75ExnerAdapterV93Error(
                "v93 Q3/v75 CPU checkpoint identity differs")
        state = self.runtime.restart(value["v75_checkpoint"], self.binding)
        if state.state_sha256 != value.get("state_sha256"):
            raise Q3V75ExnerAdapterV93Error(
                "v93 Q3/v75 CPU checkpoint state differs")
        self._state = state
        return state


__all__ = [
    "BACKEND_CPU", "CHECKPOINT_SCOPE_ID", "CPUQ3V75ExnerAdapterV93",
    "CPUQ3V75OpenTrialV93", "CPUQ3V75PreparedCommitV93",
    "MORPHOLOGICAL_FACTOR", "Q3V75ExnerAdapterV93Error",
    "Q3ExnerReceiptV93", "Q3V75ExnerReceiptV93", "SCHEMA", "SCOPE_ID",
    "validate_q3_v75_exner_receipt_v93",
]
