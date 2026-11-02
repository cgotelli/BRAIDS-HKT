"""Bounded Q3/v75 direct-vector transaction composer.

This additive module joins the accepted v67/v69/v70/v71/v72/v73 gates
without changing any of their implementations.  The ordering is fixed:

``v73 producer + live taurat -> optional v72 derived capacity -> v71
wet/dry ownership -> one v73 face/slope/inventory/Exner -> commit``.

The v70 boundary candidate owns the supported closed, equilibrium-inlet and
free-owner-export selection.  Prescribed external feed is deliberately
rejected: the accepted v67 face lifecycle multiplies boundary vectors by
owner availability, while v70 prescribed feed must bypass that operation.
Until one face merger owns both semantics, accepting it would create either
the wrong flux or a second Exner owner.

The portable class below is the CPU oracle for the composition receipt.  The
production CUDA implementation lives in :mod:`braids_gpu` and the native
OpenMP proof in ``solver/wave``.  No object in this module writes Delft3D.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Mapping, Sequence

import numpy as np

from .bedload_adaptation_live_v72 import (
    BACKEND_CPU as ADAPTATION_CPU,
    BACKEND_GPU as ADAPTATION_GPU,
    CAPACITY_DIRECT,
    BedloadAdaptationResultV72,
    BedloadAdaptationTopologyV72,
    load_execution_v72,
    relax_capacity_v72,
    restart_document_v72,
    validate_topology_v72,
)
from .bedload_boundary_live_v70 import (
    INTERNAL_CLOSED,
    INTERNAL_EQUILIBRIUM,
    INTERNAL_FREE_EXPORT,
    INTERNAL_PRESCRIBED,
    SOURCE_EQUILIBRIUM_TRACE,
    SOURCE_NONE,
    SOURCE_OWNER_FREE_EXPORT,
    SOURCE_PRESCRIBED_EXTERNAL,
    BedloadBoundaryCandidateV70,
    BedloadBoundaryTopologyV70,
    load_bedload_boundary_execution_v70,
    resolve_bedload_boundary_candidate_v70,
)
from .classic_flow_taurat_v69 import (
    project_cell_taurat_to_faces,
    resolve_cell_taurat,
)
from .direct_vector_family_live_v67 import (
    DirectVectorFamilyStateV67,
    advance_direct_vector_family_state_v67,
    direct_vector_family_state_checkpoint_v67,
    load_direct_vector_family_state_checkpoint_v67,
    resolve_direct_vector_family_state_v67,
    validate_direct_vector_family_state_v67,
)
from .direct_vector_live_taurat_v73 import (
    STRESS_SEMANTICS,
    load_direct_vector_live_taurat_execution_v73,
)
from .direct_vector_wetdry_amr_live_v71 import (
    ConservativeTransferPlanV71,
    FACE_CLOSED,
    FACE_INTERNAL,
    FACE_OWNER_OUTLET,
    FACE_TRACE_INLET,
    WetDryTopologyV71,
    execution_document_v71,
    transfer_bed_mass_reference_v71,
)


SCHEMA = 1
ENVELOPE_VERSION = 75
SCOPE_ID = "braids-q3-unified-direct-vector-transaction-v75-v1"
CHECKPOINT_SCOPE_ID = "braids-q3-unified-direct-vector-checkpoint-v75-v1"
MAX_ACCEPTED_STEPS = 4
BACKENDS = frozenset(("cpu", "implicit_fwave_midpoint_gpu"))
_SHA = re.compile(r"[0-9a-f]{64}\Z")


class UnifiedDirectVectorV75Error(ValueError):
    """The v75 execution, topology, transaction, or restart is invalid."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, OverflowError) as error:
        raise UnifiedDirectVectorV75Error(
            "Q3/v75 data are not finite canonical JSON"
        ) from error


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _array_digest(*values: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in values:
        data = np.ascontiguousarray(value)
        digest.update(str(data.dtype).encode("ascii"))
        digest.update(_canonical(list(data.shape)))
        digest.update(data.tobytes(order="C"))
    return digest.hexdigest()


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise UnifiedDirectVectorV75Error(
            f"Q3/v75 {name} must be a lowercase SHA-256 identity"
        )
    return value


def _finite(value: object, name: str, *, positive: bool = False,
            nonnegative: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise UnifiedDirectVectorV75Error(f"Q3/v75 {name} must be numeric")
    result = float(value)
    if (not math.isfinite(result) or (positive and result <= 0.0) or
            (nonnegative and result < 0.0)):
        raise UnifiedDirectVectorV75Error(
            f"Q3/v75 {name} is outside its finite domain"
        )
    return 0.0 if result == 0.0 else result


def _backend_child(backend: str) -> tuple[str, str, str]:
    if backend == "cpu":
        return "cpu", "cpu_openmp", ADAPTATION_CPU
    if backend == "implicit_fwave_midpoint_gpu":
        return backend, backend, ADAPTATION_GPU
    raise UnifiedDirectVectorV75Error(
        "Q3/v75 backend must be native CPU/OpenMP or explicit implicit GPU"
    )


def _mapping(value: object, name: str) -> dict[str, Mapping[str, object]]:
    if not isinstance(value, Mapping) or not value:
        raise UnifiedDirectVectorV75Error(
            f"Q3/v75 {name} must be a nonempty topology mapping"
        )
    result: dict[str, Mapping[str, object]] = {}
    for key, row in value.items():
        result[_sha(key, f"{name} topology")] = row
    return result


def unified_direct_vector_execution_document_v75(
    *,
    backend: str,
    live_taurat_executions: Mapping[str, Mapping[str, object]],
    wetdry_execution: Mapping[str, object],
    boundary_executions: Mapping[str, Mapping[str, object]],
    adaptation_executions: Mapping[str, Mapping[str, object]] | None = None,
    repository_root: str | Path | None = None,
) -> dict[str, object]:
    """Bind all child receipts before a v75 runtime may be constructed."""
    live_backend, boundary_backend, adaptation_backend = _backend_child(backend)
    live_rows = _mapping(live_taurat_executions, "live-taurat executions")
    boundary_rows = _mapping(boundary_executions, "boundary executions")
    adaptation_rows = (
        {} if adaptation_executions is None else
        _mapping(adaptation_executions, "adaptation executions")
    )
    topology_ids = tuple(live_rows)
    if set(boundary_rows) != set(topology_ids):
        raise UnifiedDirectVectorV75Error(
            "Q3/v75 every selected topology requires one v70 boundary receipt"
        )
    if adaptation_rows and set(adaptation_rows) != set(topology_ids):
        raise UnifiedDirectVectorV75Error(
            "Q3/v75 adaptation must be absent or bound on every topology"
        )

    try:
        wetdry = dict(wetdry_execution)
        rebuilt_wetdry = execution_document_v71(
            backend=wetdry["backend"],
            physics_fingerprint=wetdry["physics_fingerprint"],
            topology_ids=wetdry["topology_ids"],
            maximum_accepted_steps=wetdry["maximum_accepted_steps"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise UnifiedDirectVectorV75Error(
            "Q3/v75 v71 execution receipt is invalid"
        ) from error
    if wetdry != rebuilt_wetdry or wetdry["backend"] != live_backend or \
            wetdry["maximum_accepted_steps"] != MAX_ACCEPTED_STEPS or \
            set(wetdry["topology_ids"]) != set(topology_ids):
        raise UnifiedDirectVectorV75Error(
            "Q3/v75 v71 backend/topology/bounded-step identity differs"
        )

    root = (Path(repository_root) if repository_root is not None else
            Path(__file__).resolve().parents[2])
    physical_fingerprint: str | None = None
    fraction_ids: tuple[str, ...] | None = None
    resolved_rows: list[dict[str, object]] = []
    for topology_id in topology_ids:
        try:
            live = load_direct_vector_live_taurat_execution_v73(
                live_rows[topology_id], repository_root=root,
            )
            boundary = load_bedload_boundary_execution_v70(
                boundary_rows[topology_id], root=root,
            )
            adaptation = (
                None if not adaptation_rows else
                load_execution_v72(adaptation_rows[topology_id])
            )
        except (TypeError, ValueError) as error:
            raise UnifiedDirectVectorV75Error(
                f"Q3/v75 child receipt is invalid for {topology_id}"
            ) from error
        parent = live["parent_execution"]
        physical = parent["physical_profile_fingerprint"]
        ids = tuple(parent["physical_profile"]["fraction_ids"])
        if (live["backend"] != live_backend or
                parent["dependencies"]["topology"] != topology_id or
                live["accepted_step_limit"] != MAX_ACCEPTED_STEPS or
                live["stress_semantics"] != STRESS_SEMANTICS or
                live["automatic_backend_fallback"] is not False):
            raise UnifiedDirectVectorV75Error(
                "Q3/v75 live-taurat backend/topology/ownership differs"
            )
        if physical_fingerprint is None:
            physical_fingerprint, fraction_ids = physical, ids
        elif physical != physical_fingerprint or ids != fraction_ids:
            raise UnifiedDirectVectorV75Error(
                "Q3/v75 topology runtimes do not share one physical profile"
            )
        if (boundary["backend"] != boundary_backend or
                boundary["topology_fingerprint"] != topology_id or
                tuple(boundary["fraction_ids"]) != ids or
                boundary["parent_execution_fingerprint"] not in {
                    live["fingerprint"], parent["fingerprint"],
                } or boundary["accepted_step_limit"] != MAX_ACCEPTED_STEPS or
                boundary["automatic_backend_fallback"] is not False):
            raise UnifiedDirectVectorV75Error(
                "Q3/v75 v70 parent/backend/topology/fraction identity differs"
            )
        prescribed = [
            row["name"] for row in boundary["boundaries"]
            if row["internal_mode"] == INTERNAL_PRESCRIBED
        ]
        if prescribed:
            raise UnifiedDirectVectorV75Error(
                "Q3/v75 prescribed external feed is fail-closed until one "
                "face/availability/Exner owner can bypass owner availability: "
                + ", ".join(prescribed)
            )
        if any(row["internal_mode"] not in {
                INTERNAL_CLOSED, INTERNAL_EQUILIBRIUM,
                INTERNAL_FREE_EXPORT,
        } for row in boundary["boundaries"]):
            raise UnifiedDirectVectorV75Error(
                "Q3/v75 contains an unsupported v70 boundary mode"
            )
        if adaptation is not None and (
            adaptation["backend"] != adaptation_backend or
            adaptation["capacity_family"] != CAPACITY_DIRECT or
            adaptation["parent_execution_fingerprint"] != parent["fingerprint"] or
            adaptation["topology_fingerprint"] != topology_id or
            adaptation["wetdry_amr_execution_fingerprint"] !=
                wetdry["fingerprint"] or
            tuple(adaptation["fraction_ids"]) != ids or
            adaptation["accepted_step_limit"] != MAX_ACCEPTED_STEPS or
            adaptation["automatic_backend_fallback"] is not False
        ):
            raise UnifiedDirectVectorV75Error(
                "Q3/v75 v72 parent/backend/topology/fraction identity differs"
            )
        resolved_rows.append({
            "topology_id": topology_id,
            "live_taurat_execution": live,
            "boundary_execution": boundary,
            "adaptation_execution": adaptation,
        })
    assert physical_fingerprint is not None and fraction_ids is not None
    if wetdry["physics_fingerprint"] != physical_fingerprint:
        raise UnifiedDirectVectorV75Error(
            "Q3/v75 v71 physics fingerprint differs from v73"
        )

    body: dict[str, object] = {
        "schema": SCHEMA,
        "envelope_version": ENVELOPE_VERSION,
        "scope_id": SCOPE_ID,
        "backend": backend,
        "integrator": (
            "implicit_fwave_midpoint" if backend.endswith("_gpu")
            else "native_openmp"
        ),
        "automatic_backend_fallback": False,
        "maximum_accepted_steps": MAX_ACCEPTED_STEPS,
        "physical_profile_fingerprint": physical_fingerprint,
        "fraction_ids": list(fraction_ids),
        "wetdry_execution": wetdry,
        "adaptation_enabled": bool(adaptation_rows),
        "topologies": resolved_rows,
        "source_order": [
            "authenticated_v67_direct_vector_capacity_and_same_invocation_v73_taurat",
            "optional_v72_derived_stage_local_capacity",
            "v71_selected_wetdry_and_amr_owner",
            "one_v73_face_and_islope2_and_active_fraction_and_inventory",
            "supported_v70_boundary_selection_inside_the_same_face_owner",
            "one_exner_candidate_then_commit_or_rollback_or_cancel",
        ],
        "once_only_owners": {
            "producer_taurat": 1,
            "adaptation": "0_or_1_derived_not_accepted_state",
            "face_reconstruction": 1,
            "bed_slope": 1,
            "active_fraction": 1,
            "inventory_fill": 1,
            "exner": 1,
        },
        "prescribed_external_feed": {
            "admitted": False,
            "reason": (
                "v70 prescribed feed bypasses owner availability; the current "
                "v67/v73 face owner cannot merge that exception without a "
                "second or incorrect Exner path"
            ),
        },
        "normal_product_admission": False,
        "nonclaims": [
            "prescribed_external_feed_composition",
            "suspended_load_or_stratigraphy",
            "recursive_subcycling",
            "long_run_morphological_validation",
            "QGIS_or_product_admission",
        ],
    }
    return {**body, "fingerprint": _fingerprint(body)}


def load_unified_direct_vector_execution_v75(
    value: Mapping[str, object], *, repository_root: str | Path | None = None,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise UnifiedDirectVectorV75Error("Q3/v75 execution must be an object")
    try:
        rows = value["topologies"]
        rebuilt = unified_direct_vector_execution_document_v75(
            backend=value["backend"],
            live_taurat_executions={
                row["topology_id"]: row["live_taurat_execution"]
                for row in rows
            },
            wetdry_execution=value["wetdry_execution"],
            boundary_executions={
                row["topology_id"]: row["boundary_execution"]
                for row in rows
            },
            adaptation_executions=(
                {
                    row["topology_id"]: row["adaptation_execution"]
                    for row in rows
                } if value["adaptation_enabled"] else None
            ),
            repository_root=repository_root,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise UnifiedDirectVectorV75Error(
            "Q3/v75 execution identity is incomplete"
        ) from error
    if dict(value) != rebuilt or _canonical(value) != _canonical(rebuilt):
        raise UnifiedDirectVectorV75Error(
            "Q3/v75 execution identity is stale or forged"
        )
    return rebuilt


def verify_unified_direct_vector_sources_v75(
    execution: Mapping[str, object], *, repository_root: str | Path | None = None,
) -> dict[str, object]:
    """Hash only additive v75 files; child receipts authenticate their sources."""
    resolved = load_unified_direct_vector_execution_v75(
        execution, repository_root=repository_root,
    )
    root = (Path(repository_root) if repository_root is not None else
            Path(__file__).resolve().parents[2])
    paths = (
        "braids_core/sediment/unified_direct_vector_transaction_v75.py",
        "braids_gpu/unified_direct_vector_transaction_v75.py",
        "solver/wave/braids_unified_direct_vector_v75_module.f90",
    )
    rows = []
    for relative in paths:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise UnifiedDirectVectorV75Error(
                f"Q3/v75 additive implementation source is absent/linked: {relative}"
            )
        rows.append({
            "path": relative,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    body = {
        "scope_id": SCOPE_ID,
        "execution_fingerprint": resolved["fingerprint"],
        "sources": rows,
        "delft3d_source_access": "read_only_child_hash_authentication",
    }
    return {**body, "fingerprint": _fingerprint(body)}


@dataclass(frozen=True)
class DirectVectorCapacityV75:
    """Portable oracle input sealed to one v73 producer identity."""

    execution_fingerprint: str
    cell_capacity: np.ndarray
    trace_capacity: np.ndarray
    cell_taurat: np.ndarray
    trace_taurat: np.ndarray
    same_invocation_receipt_sha256: str


def seal_cpu_capacity_v75(
    *, live_taurat_execution: Mapping[str, object], cell_capacity: object,
    trace_capacity: object, live_cell_taurat: object,
    live_trace_taurat: object,
) -> DirectVectorCapacityV75:
    """Seal CPU-oracle producer arrays and enforce v69/v73 taurat semantics."""
    live = load_direct_vector_live_taurat_execution_v73(live_taurat_execution)
    nclass = len(live["fraction_methods"])
    cell = np.ascontiguousarray(np.asarray(cell_capacity, dtype=np.float64))
    trace = np.ascontiguousarray(np.asarray(trace_capacity, dtype=np.float64))
    if (cell.ndim != 3 or trace.ndim != 3 or
            cell.shape[:2] != (2, nclass) or
            trace.shape[:2] != (2, nclass) or
            not np.isfinite(cell).all() or not np.isfinite(trace).all()):
        raise UnifiedDirectVectorV75Error(
            "Q3/v75 CPU capacities must be finite (2,nclass,npoint) arrays"
        )
    selectors = np.asarray(
        [row["reference_selector"] for row in live["fraction_methods"]],
        dtype=np.int64,
    )
    cell_stress = resolve_cell_taurat(
        selectors, ncell=cell.shape[2], live_stress_ratio=live_cell_taurat,
    )
    trace_stress = resolve_cell_taurat(
        selectors, ncell=trace.shape[2], live_stress_ratio=live_trace_taurat,
    )
    frozen = []
    for array in (cell, trace, cell_stress, trace_stress):
        value = np.array(array, copy=True, order="C")
        value.setflags(write=False)
        frozen.append(value)
    token = _fingerprint({
        "scope_id": SCOPE_ID,
        "live_taurat_execution_fingerprint": live["fingerprint"],
        "producer_output_channel": live["producer_output_channel"],
        "stress_semantics": live["stress_semantics"],
        "same_invocation_capacity_and_taurat": True,
        "arrays_sha256": _array_digest(*frozen),
    })
    return DirectVectorCapacityV75(
        live["fingerprint"], *frozen, token,
    )


@dataclass(frozen=True)
class CPUUnifiedBindingV75:
    topology_id: str
    wetdry_topology: WetDryTopologyV71
    normal: np.ndarray
    face_length_m: np.ndarray
    cell_face_offset: np.ndarray
    cell_faces: np.ndarray
    signs: np.ndarray
    boundary_execution: Mapping[str, object]
    boundary_topology: BedloadBoundaryTopologyV70
    family_boundary_faces: np.ndarray
    adaptation_execution: Mapping[str, object] | None
    adaptation_topology: BedloadAdaptationTopologyV72 | None
    binding_sha256: str


@dataclass(frozen=True)
class CPUUnifiedStateV75:
    topology_id: str
    mesh_epoch: int
    direct_state: DirectVectorFamilyStateV67
    accepted_face_flux: np.ndarray
    accepted_cell_availability: np.ndarray
    state_sha256: str


@dataclass(frozen=True)
class CPUUnifiedTrialV75:
    parent_state_sha256: str
    topology_id: str
    capacity_receipt_sha256: str
    adaptation: BedloadAdaptationResultV72 | None
    consumed_cell_capacity_sha256: str
    face_taurat_sha256: str
    boundary_candidate: BedloadBoundaryCandidateV70
    face_flux_kg_m_s: np.ndarray
    inventory_fill: np.ndarray
    accepted_cell_availability: np.ndarray
    cell_delta_kg_m2: np.ndarray
    named_boundary_transfer_kg: np.ndarray
    named_boundary_absolute_kg: np.ndarray
    conservation_residual_kg: np.ndarray
    candidate: CPUUnifiedStateV75
    producer_taurat_application_count: int
    adaptation_application_count: int
    face_reconstruction_application_count: int
    bed_slope_application_count: int
    boundary_owner_application_count: int
    active_fraction_application_count: int
    inventory_fill_application_count: int
    exner_application_count: int
    trial_sha256: str


@dataclass(frozen=True)
class CPUUnifiedRegridTrialV75:
    parent_state_sha256: str
    old_topology_id: str
    new_topology_id: str
    transfer_plan_id: str
    transition_kind: str
    wet_state_change_count: int
    conservation_residual_kg: np.ndarray
    cleared_face_register: np.ndarray
    cleared_cell_availability: np.ndarray
    candidate: CPUUnifiedStateV75
    transfer_application_count: int
    face_register_cleanup_count: int
    availability_cleanup_count: int
    exner_application_count: int
    trial_sha256: str


def _readonly(value: object, dtype, shape: tuple[int, ...], name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != shape:
        raise UnifiedDirectVectorV75Error(
            f"Q3/v75 {name} must have shape {shape}"
        )
    result = np.ascontiguousarray(raw, dtype=dtype)
    if result.dtype.kind == "f" and not np.isfinite(result).all():
        raise UnifiedDirectVectorV75Error(f"Q3/v75 {name} is nonfinite")
    result.setflags(write=False)
    return result


class PortableUnifiedDirectVectorV75:
    """Portable NumPy oracle for one bounded v75 execution."""

    backend = "cpu"
    integrator = "native_openmp"

    def __init__(self, execution: Mapping[str, object]):
        self.execution = load_unified_direct_vector_execution_v75(execution)
        if self.execution["backend"] != self.backend:
            raise UnifiedDirectVectorV75Error(
                "Q3/v75 portable oracle requires the CPU execution receipt"
            )
        self.rows = {
            row["topology_id"]: row for row in self.execution["topologies"]
        }
        self.nclass = len(self.execution["fraction_ids"])
        self._open_parent: str | None = None

    def prepare_binding(
        self, *, topology: WetDryTopologyV71, normal: object,
        face_length_m: object, cell_face_offset: object, cell_faces: object,
        signs: object, boundary_topology: BedloadBoundaryTopologyV70,
        family_boundary_faces: object,
        adaptation_topology: BedloadAdaptationTopologyV72 | None = None,
    ) -> CPUUnifiedBindingV75:
        if (not isinstance(topology, WetDryTopologyV71) or
                topology.topology_id not in self.rows):
            raise UnifiedDirectVectorV75Error(
                "Q3/v75 CPU selected topology is outside the execution"
            )
        row = self.rows[topology.topology_id]
        normal_array = _readonly(
            normal, np.float64, (2, topology.nface), "face normals",
        )
        length = _readonly(
            face_length_m, np.float64, (topology.nface,), "face lengths",
        )
        offset = _readonly(
            cell_face_offset, np.int32, (topology.ncell + 1,),
            "cell-face offsets",
        )
        face_raw = np.asarray(cell_faces)
        faces = _readonly(
            face_raw, np.int32, (face_raw.size,), "cell faces",
        )
        sign = _readonly(signs, np.int32, (faces.size,), "incidence signs")
        if (np.any(length <= 0.0) or
                np.any(np.abs(np.sum(normal_array**2, axis=0)-1.0) > 1e-12) or
                offset[0] != 0 or offset[-1] != faces.size or
                np.any(np.diff(offset) < 0) or np.any(faces < 0) or
                np.any(faces >= topology.nface) or
                np.any(~np.isin(sign, (-1, 1)))):
            raise UnifiedDirectVectorV75Error(
                "Q3/v75 CPU face geometry/adjacency is invalid"
            )
        cell = np.repeat(np.arange(topology.ncell, dtype=np.int32), np.diff(offset))
        own = topology.owner[faces]
        other = topology.neighbour[faces]
        if np.any(~(((cell == own) & (sign == 1)) |
                    ((cell == other) & (sign == -1)))):
            raise UnifiedDirectVectorV75Error(
                "Q3/v75 CPU adjacency ownership differs"
            )
        boundary_execution = row["boundary_execution"]
        if (not isinstance(boundary_topology, BedloadBoundaryTopologyV70) or
                boundary_topology.execution_fingerprint !=
                    boundary_execution["fingerprint"]):
            raise UnifiedDirectVectorV75Error(
                "Q3/v75 CPU v70 topology/execution differs"
            )
        mapping = _readonly(
            family_boundary_faces, np.int32, (boundary_topology.nface,),
            "family boundary-face map",
        )
        if (np.any(mapping < 0) or np.any(mapping >= topology.nface) or
                len(set(mapping.tolist())) != mapping.size or
                np.any(topology.neighbour[mapping] != -1) or
                not np.array_equal(topology.owner[mapping],
                                   boundary_topology.owner) or
                not np.array_equal(normal_array[:, mapping],
                                   boundary_topology.normal) or
                not np.array_equal(length[mapping],
                                   boundary_topology.face_length_m) or
                not np.array_equal(topology.cell_area_m2,
                                   boundary_topology.cell_area_m2)):
            raise UnifiedDirectVectorV75Error(
                "Q3/v75 CPU v70/family boundary topology differs"
            )
        adaptation_execution = row["adaptation_execution"]
        if adaptation_execution is None:
            if adaptation_topology is not None:
                raise UnifiedDirectVectorV75Error(
                    "Q3/v75 adaptation topology supplied while disabled"
                )
        else:
            if adaptation_topology is None:
                raise UnifiedDirectVectorV75Error(
                    "Q3/v75 enabled adaptation requires its topology"
                )
            validate_topology_v72(adaptation_execution, adaptation_topology)
            if (adaptation_topology.ncell != topology.ncell or
                    adaptation_topology.nface != topology.nface or
                    not np.array_equal(adaptation_topology.owner,
                                       topology.owner) or
                    not np.array_equal(adaptation_topology.neighbour,
                                       topology.neighbour) or
                    not np.array_equal(adaptation_topology.normal,
                                       normal_array) or
                    not np.array_equal(adaptation_topology.face_length_m,
                                       length) or
                    not np.array_equal(adaptation_topology.cell_area_m2,
                                       topology.cell_area_m2)):
                raise UnifiedDirectVectorV75Error(
                    "Q3/v75 adaptation/family topology differs"
                )
        token = _fingerprint({
            "scope_id": SCOPE_ID,
            "execution_fingerprint": self.execution["fingerprint"],
            "topology_id": topology.topology_id,
            "boundary_topology_sha256": boundary_topology.topology_sha256,
            "adaptation_topology_sha256": (
                None if adaptation_topology is None else
                adaptation_topology.topology_sha256
            ),
            "arrays_sha256": _array_digest(
                normal_array, length, offset, faces, sign, mapping,
            ),
        })
        return CPUUnifiedBindingV75(
            topology.topology_id, topology, normal_array, length, offset,
            faces, sign, boundary_execution, boundary_topology, mapping,
            adaptation_execution, adaptation_topology, token,
        )

    def _state_token(self, binding: CPUUnifiedBindingV75,
                     direct: DirectVectorFamilyStateV67, mesh_epoch: int,
                     flux: np.ndarray, availability: np.ndarray) -> str:
        return _fingerprint({
            "scope_id": SCOPE_ID,
            "execution_fingerprint": self.execution["fingerprint"],
            "binding_sha256": binding.binding_sha256,
            "topology_id": binding.topology_id,
            "mesh_epoch": mesh_epoch,
            "direct_state_sha256": direct.state_sha256,
            "register_arrays_sha256": _array_digest(flux, availability),
        })

    def _make_state(self, binding: CPUUnifiedBindingV75,
                    direct: DirectVectorFamilyStateV67, *, mesh_epoch: int,
                    accepted_face_flux: object,
                    accepted_cell_availability: object) -> CPUUnifiedStateV75:
        parent_execution = self.rows[binding.topology_id][
            "live_taurat_execution"]["parent_execution"]
        validate_direct_vector_family_state_v67(
            direct, execution=parent_execution,
        )
        if direct.accepted_step > MAX_ACCEPTED_STEPS:
            raise UnifiedDirectVectorV75Error(
                "Q3/v75 CPU accepted-step limit exceeded"
            )
        flux = _readonly(
            accepted_face_flux, np.float64,
            (self.nclass, binding.wetdry_topology.nface),
            "accepted face register",
        )
        availability = _readonly(
            accepted_cell_availability, np.float64,
            (self.nclass, binding.wetdry_topology.ncell),
            "accepted availability register",
        )
        if np.any(availability < 0.0):
            raise UnifiedDirectVectorV75Error(
                "Q3/v75 CPU availability register is negative"
            )
        token = self._state_token(binding, direct, mesh_epoch, flux, availability)
        return CPUUnifiedStateV75(
            binding.topology_id, mesh_epoch, direct, flux, availability, token,
        )

    def initial_state(self, binding: CPUUnifiedBindingV75, *,
                      bed_mass_kg_m2: object) -> CPUUnifiedStateV75:
        parent = self.rows[binding.topology_id][
            "live_taurat_execution"]["parent_execution"]
        direct = resolve_direct_vector_family_state_v67(
            execution=parent, bed_mass_kg_m2=bed_mass_kg_m2,
        )
        return self._make_state(
            binding, direct, mesh_epoch=0,
            accepted_face_flux=np.zeros(
                (self.nclass, binding.wetdry_topology.nface), dtype=np.float64,
            ),
            accepted_cell_availability=np.zeros(
                (self.nclass, binding.wetdry_topology.ncell), dtype=np.float64,
            ),
        )

    def _validate_state(self, state: CPUUnifiedStateV75,
                        binding: CPUUnifiedBindingV75) -> None:
        if (not isinstance(state, CPUUnifiedStateV75) or
                state.topology_id != binding.topology_id or
                state.mesh_epoch < 0 or
                state.state_sha256 != self._state_token(
                    binding, state.direct_state, state.mesh_epoch,
                    state.accepted_face_flux,
                    state.accepted_cell_availability,
                )):
            raise UnifiedDirectVectorV75Error(
                "Q3/v75 CPU accepted state is stale or topology-mismatched"
            )

    def _boundary_candidate(
        self, binding: CPUUnifiedBindingV75, *, hydraulic_outward_flux: object,
        cells: np.ndarray, traces: np.ndarray,
    ) -> BedloadBoundaryCandidateV70:
        water = _readonly(
            hydraulic_outward_flux, np.float64,
            (binding.boundary_topology.nface,), "boundary hydraulic flux",
        )
        mapped = binding.family_boundary_faces
        candidate = resolve_bedload_boundary_candidate_v70(
            binding.boundary_execution, binding.boundary_topology,
            hydraulic_outward_flux=water,
            owner_bed_mass_vector=cells,
            trace_bed_mass_vector=np.ascontiguousarray(traces[:, :, mapped]),
        )
        for local, family_face in enumerate(mapped):
            sources = set(candidate.source[:, local].tolist())
            if SOURCE_PRESCRIBED_EXTERNAL in sources:
                raise UnifiedDirectVectorV75Error(
                    "Q3/v75 prescribed external feed reached the live composer"
                )
            expected = None
            if sources == {SOURCE_EQUILIBRIUM_TRACE}:
                expected = FACE_TRACE_INLET
            elif sources == {SOURCE_OWNER_FREE_EXPORT}:
                expected = FACE_OWNER_OUTLET
            elif sources == {SOURCE_NONE}:
                expected = FACE_CLOSED
            if expected is None or int(binding.wetdry_topology.role[family_face]) != expected:
                raise UnifiedDirectVectorV75Error(
                    "Q3/v75 selected face role differs from the v70 boundary owner"
                )
        return candidate

    def begin_trial(
        self, *, state: CPUUnifiedStateV75, binding: CPUUnifiedBindingV75,
        capacity: DirectVectorCapacityV75, q: object, downhill: object,
        depth: object, mean_diameter: object, hiding: object,
        hydraulic_outward_flux: object, dry_tolerance: float,
        dx_m: float, dy_m: float, dt_s: float,
        morphological_factor: float = 1.0, cancelled=lambda: False,
    ) -> CPUUnifiedTrialV75:
        """Build one uncommitted CPU oracle candidate in the required order."""
        if not callable(cancelled):
            raise TypeError("cancelled must be callable")
        if cancelled():
            raise InterruptedError("Q3/v75 CPU transaction cancelled before work")
        if self._open_parent is not None:
            raise RuntimeError("Q3/v75 CPU already has an open transaction")
        self._validate_state(state, binding)
        if state.direct_state.accepted_step >= MAX_ACCEPTED_STEPS:
            raise UnifiedDirectVectorV75Error(
                "Q3/v75 CPU bounded accepted-step limit reached"
            )
        row = self.rows[binding.topology_id]
        live = row["live_taurat_execution"]
        if (not isinstance(capacity, DirectVectorCapacityV75) or
                capacity.execution_fingerprint != live["fingerprint"] or
                capacity.cell_capacity.shape !=
                    (2, self.nclass, binding.wetdry_topology.ncell) or
                capacity.trace_capacity.shape !=
                    (2, self.nclass, binding.wetdry_topology.nface)):
            raise UnifiedDirectVectorV75Error(
                "Q3/v75 CPU capacity receipt differs"
            )
        sealed = seal_cpu_capacity_v75(
            live_taurat_execution=live,
            cell_capacity=capacity.cell_capacity,
            trace_capacity=capacity.trace_capacity,
            live_cell_taurat=capacity.cell_taurat,
            live_trace_taurat=capacity.trace_taurat,
        )
        if sealed.same_invocation_receipt_sha256 != \
                capacity.same_invocation_receipt_sha256:
            raise UnifiedDirectVectorV75Error(
                "Q3/v75 CPU capacity/taurat arrays are stale or separated"
            )
        topology = binding.wetdry_topology
        hydraulic = _readonly(q, np.float64, (3, topology.ncell), "hydraulics")
        dry = _finite(dry_tolerance, "dry tolerance", nonnegative=True)
        physical_wet = hydraulic[0] > dry
        if not np.array_equal(physical_wet, topology.wet):
            raise UnifiedDirectVectorV75Error(
                "Q3/v75 CPU hydraulic wet mask differs from v71"
            )
        cell_capacity = np.ascontiguousarray(capacity.cell_capacity)
        adaptation = None
        if binding.adaptation_execution is not None:
            assert binding.adaptation_topology is not None
            adaptation = relax_capacity_v72(
                binding.adaptation_execution, binding.adaptation_topology,
                q=hydraulic, equilibrium_capacity=cell_capacity,
                boundary_feed_kg_m_s=np.zeros(
                    (self.nclass, binding.adaptation_topology.nboundary),
                    dtype=np.float64,
                ),
                dry_tolerance=dry, dx_m=dx_m, dy_m=dy_m,
                cancelled=cancelled,
            )
            cell_capacity = np.ascontiguousarray(adaptation.adapted_capacity)
        consumed_digest = _array_digest(cell_capacity)

        face_taurat = project_cell_taurat_to_faces(
            capacity.cell_taurat, owner=topology.owner,
            neighbour=topology.neighbour, role=topology.role,
        )
        boundary_candidate = self._boundary_candidate(
            binding, hydraulic_outward_flux=hydraulic_outward_flux,
            cells=cell_capacity, traces=capacity.trace_capacity,
        )
        down = _readonly(
            downhill, np.float64, (2, topology.nface), "downhill",
        )
        water_depth = _readonly(
            depth, np.float64, (self.nclass, topology.nface), "slope depth",
        )
        mean = _readonly(
            mean_diameter, np.float64,
            (self.nclass, topology.nface), "mean diameter",
        )
        hide = _readonly(
            hiding, np.float64, (self.nclass, topology.ncell), "hiding",
        )
        d50 = np.ascontiguousarray(np.asarray(
            live["parent_execution"]["physical_profile"]["d50_m"],
            dtype=np.float64,
        ))
        safe_other = np.clip(topology.neighbour, 0, topology.ncell-1)
        left = np.ascontiguousarray(cell_capacity[:, :, topology.owner])
        right = np.ascontiguousarray(cell_capacity[:, :, safe_other])
        owner_bed = left
        trace_bed = np.ascontiguousarray(capacity.trace_capacity)

        # Use the accepted portable H4 operator.  The first evaluation is the
        # inventory demand preview; the second is the sole accepted face/slope
        # application represented by the public counters below.
        from braids_gpu.flow_bed_slope_composition import (
            DONOR_NEIGHBOUR, DONOR_NONE, DONOR_OWNER, DONOR_TRACE,
            PortableFlowBedSlopeFaceComposer,
        )
        composer = PortableFlowBedSlopeFaceComposer(
            live["bed_slope_execution"],
        )
        common = dict(
            left_bed=left, right_bed=right, owner_bed=owner_bed,
            trace_bed=trace_bed,
            owner=np.ascontiguousarray(topology.owner, dtype=np.int32),
            neighbour=np.ascontiguousarray(topology.neighbour, dtype=np.int32),
            role=np.ascontiguousarray(topology.role, dtype=np.int32),
            normal=np.ascontiguousarray(binding.normal), downhill=down,
            stress=np.ascontiguousarray(face_taurat), depth=water_depth,
            d50=d50, mean_diameter=mean, hiding=hide,
        )
        provisional = composer.compose(
            availability=np.ascontiguousarray(state.direct_state.active_fraction),
            **common,
        )
        dt = _finite(dt_s, "trial dt", positive=True)
        morfac = _finite(
            morphological_factor, "morphological factor", nonnegative=True,
        )
        donor_cell = np.full(
            (self.nclass, topology.nface), -1, dtype=np.int32,
        )
        donor_cell[provisional.donor == DONOR_OWNER] = np.broadcast_to(
            topology.owner, donor_cell.shape,
        )[provisional.donor == DONOR_OWNER]
        donor_cell[provisional.donor == DONOR_NEIGHBOUR] = np.broadcast_to(
            safe_other, donor_cell.shape,
        )[provisional.donor == DONOR_NEIGHBOUR]
        donor_cell[provisional.donor == DONOR_TRACE] = np.broadcast_to(
            topology.owner, donor_cell.shape,
        )[provisional.donor == DONOR_TRACE]
        donor_cell[provisional.donor == DONOR_NONE] = -1
        outgoing = np.zeros_like(state.direct_state.bed_mass_kg_m2)
        for face in range(topology.nface):
            role = int(topology.role[face])
            for grain in range(self.nclass):
                cell = int(donor_cell[grain, face])
                value = float(provisional.flux[grain, face])
                is_outgoing = (
                    cell >= 0 and
                    ((role == FACE_INTERNAL and value != 0.0) or
                     (role == FACE_OWNER_OUTLET and value > 0.0))
                )
                if is_outgoing:
                    outgoing[grain, cell] += (
                        abs(value)*binding.face_length_m[face]*dt*morfac /
                        topology.cell_area_m2[cell]
                    )
        fill = np.ones_like(outgoing)
        fill[state.direct_state.bed_mass_kg_m2 == 0.0] = 0.0
        demand = outgoing > 0.0
        fill[demand] = np.minimum(
            1.0, state.direct_state.bed_mass_kg_m2[demand]/outgoing[demand],
        )
        availability = np.ascontiguousarray(
            state.direct_state.active_fraction*fill,
        )
        accepted = composer.compose(availability=availability, **common)
        flux = np.ascontiguousarray(accepted.flux)
        if np.any(flux[:, topology.role == FACE_CLOSED] != 0.0):
            raise UnifiedDirectVectorV75Error(
                "Q3/v75 CPU closed/dry face acquired transport"
            )
        delta = np.zeros_like(state.direct_state.bed_mass_kg_m2)
        named = np.zeros(
            (self.nclass, len(binding.boundary_execution["boundaries"])),
            dtype=np.float64,
        )
        named_absolute = np.zeros_like(named)
        inverse_boundary = {
            int(family): local
            for local, family in enumerate(binding.family_boundary_faces)
        }
        for face in range(topology.nface):
            transfer = flux[:, face]*binding.face_length_m[face]*dt*morfac
            owner = int(topology.owner[face])
            delta[:, owner] -= transfer/topology.cell_area_m2[owner]
            if int(topology.role[face]) == FACE_INTERNAL:
                other = int(topology.neighbour[face])
                delta[:, other] += transfer/topology.cell_area_m2[other]
            else:
                local = inverse_boundary.get(face)
                if local is None:
                    raise UnifiedDirectVectorV75Error(
                        "Q3/v75 CPU boundary face lacks its v70 owner"
                    )
                boundary = int(binding.boundary_topology.boundary_index[local])
                named[:, boundary] += transfer
                named_absolute[:, boundary] += np.abs(transfer)
        candidate_mass = np.ascontiguousarray(
            state.direct_state.bed_mass_kg_m2 + delta,
        )
        tolerance = 2048.0*np.finfo(np.float64).eps*np.maximum(
            1.0, np.abs(state.direct_state.bed_mass_kg_m2),
        )
        if np.any(candidate_mass < -tolerance):
            raise UnifiedDirectVectorV75Error(
                "Q3/v75 CPU finite inventory projection failed"
            )
        candidate_mass = np.maximum(candidate_mass, 0.0)
        residual = np.sum(
            delta*topology.cell_area_m2[None, :], axis=1,
        ) + np.sum(named, axis=1)
        scale = np.maximum(
            1.0,
            np.sum(np.abs(delta*topology.cell_area_m2[None, :]), axis=1)
            + np.sum(named_absolute, axis=1),
        )
        if np.any(np.abs(residual) > 4096.0*np.finfo(np.float64).eps*scale):
            raise UnifiedDirectVectorV75Error(
                "Q3/v75 CPU conservation ledger does not close"
            )
        dry_cells = ~topology.wet
        if np.any(dry_cells) and not np.array_equal(
            candidate_mass[:, dry_cells],
            state.direct_state.bed_mass_kg_m2[:, dry_cells],
        ):
            raise UnifiedDirectVectorV75Error(
                "Q3/v75 CPU dry-cell inventory changed"
            )
        candidate_direct = advance_direct_vector_family_state_v67(
            execution=live["parent_execution"], parent=state.direct_state,
            bed_mass_kg_m2=candidate_mass, dt_s=dt,
            boundary_signed_delta_kg=np.sum(named, axis=1),
            boundary_absolute_delta_kg=np.sum(named_absolute, axis=1),
        )
        candidate = self._make_state(
            binding, candidate_direct, mesh_epoch=state.mesh_epoch,
            accepted_face_flux=flux,
            accepted_cell_availability=np.where(
                topology.wet[None, :], availability, 0.0,
            ),
        )
        counts = {
            "producer_taurat_application_count": 1,
            "adaptation_application_count": int(adaptation is not None),
            "face_reconstruction_application_count": 1,
            "bed_slope_application_count": 1,
            "boundary_owner_application_count": 1,
            "active_fraction_application_count": 1,
            "inventory_fill_application_count": 1,
            "exner_application_count": 1,
        }
        arrays = (
            flux, fill, availability, delta, named, named_absolute, residual,
        )
        body = {
            "scope_id": SCOPE_ID,
            "execution_fingerprint": self.execution["fingerprint"],
            "parent_state_sha256": state.state_sha256,
            "candidate_state_sha256": candidate.state_sha256,
            "topology_id": binding.topology_id,
            "capacity_receipt_sha256": capacity.same_invocation_receipt_sha256,
            "adaptation_result_sha256": (
                None if adaptation is None else adaptation.result_sha256
            ),
            "consumed_cell_capacity_sha256": consumed_digest,
            "face_taurat_sha256": _array_digest(face_taurat),
            "boundary_candidate_sha256": boundary_candidate.candidate_sha256,
            **counts,
            "arrays_sha256": _array_digest(*arrays),
        }
        trial = CPUUnifiedTrialV75(
            state.state_sha256, binding.topology_id,
            capacity.same_invocation_receipt_sha256, adaptation,
            consumed_digest, _array_digest(face_taurat), boundary_candidate,
            *[_readonly(value, value.dtype, value.shape, "trial array")
              for value in arrays],
            candidate, *counts.values(), _fingerprint(body),
        )
        if cancelled():
            raise InterruptedError(
                "Q3/v75 CPU transaction cancelled before publication"
            )
        self._open_parent = state.state_sha256
        return trial

    def _validate_trial(self, state: CPUUnifiedStateV75,
                        binding: CPUUnifiedBindingV75,
                        trial: CPUUnifiedTrialV75) -> None:
        self._validate_state(state, binding)
        if (not isinstance(trial, CPUUnifiedTrialV75) or
                trial.parent_state_sha256 != state.state_sha256 or
                trial.topology_id != binding.topology_id or
                self._open_parent != state.state_sha256 or
                (trial.producer_taurat_application_count,
                 trial.face_reconstruction_application_count,
                 trial.bed_slope_application_count,
                 trial.boundary_owner_application_count,
                 trial.active_fraction_application_count,
                 trial.inventory_fill_application_count,
                 trial.exner_application_count) != (1, 1, 1, 1, 1, 1, 1) or
                trial.adaptation_application_count !=
                    int(binding.adaptation_execution is not None)):
            raise UnifiedDirectVectorV75Error(
                "Q3/v75 CPU trial parent or once-only counters differ"
            )
        counts = {
            "producer_taurat_application_count": 1,
            "adaptation_application_count": trial.adaptation_application_count,
            "face_reconstruction_application_count": 1,
            "bed_slope_application_count": 1,
            "boundary_owner_application_count": 1,
            "active_fraction_application_count": 1,
            "inventory_fill_application_count": 1,
            "exner_application_count": 1,
        }
        body = {
            "scope_id": SCOPE_ID,
            "execution_fingerprint": self.execution["fingerprint"],
            "parent_state_sha256": state.state_sha256,
            "candidate_state_sha256": trial.candidate.state_sha256,
            "topology_id": binding.topology_id,
            "capacity_receipt_sha256": trial.capacity_receipt_sha256,
            "adaptation_result_sha256": (
                None if trial.adaptation is None else
                trial.adaptation.result_sha256
            ),
            "consumed_cell_capacity_sha256":
                trial.consumed_cell_capacity_sha256,
            "face_taurat_sha256": trial.face_taurat_sha256,
            "boundary_candidate_sha256":
                trial.boundary_candidate.candidate_sha256,
            **counts,
            "arrays_sha256": _array_digest(
                trial.face_flux_kg_m_s, trial.inventory_fill,
                trial.accepted_cell_availability, trial.cell_delta_kg_m2,
                trial.named_boundary_transfer_kg,
                trial.named_boundary_absolute_kg,
                trial.conservation_residual_kg,
            ),
        }
        if trial.trial_sha256 != _fingerprint(body):
            raise UnifiedDirectVectorV75Error(
                "Q3/v75 CPU trial arrays are stale or mutated"
            )

    def commit(self, state: CPUUnifiedStateV75, binding: CPUUnifiedBindingV75,
               trial: CPUUnifiedTrialV75) -> CPUUnifiedStateV75:
        self._validate_trial(state, binding, trial)
        self._open_parent = None
        return trial.candidate

    def rollback(self, state: CPUUnifiedStateV75,
                 binding: CPUUnifiedBindingV75,
                 trial: CPUUnifiedTrialV75) -> CPUUnifiedStateV75:
        self._validate_trial(state, binding, trial)
        self._open_parent = None
        return state

    def cancel(self, state: CPUUnifiedStateV75,
               binding: CPUUnifiedBindingV75,
               trial: CPUUnifiedTrialV75) -> CPUUnifiedStateV75:
        return self.rollback(state, binding, trial)

    def begin_regrid(
        self, *, state: CPUUnifiedStateV75,
        old_binding: CPUUnifiedBindingV75,
        new_binding: CPUUnifiedBindingV75,
        plan: ConservativeTransferPlanV71, cancelled=lambda: False,
    ) -> CPUUnifiedRegridTrialV75:
        if not callable(cancelled):
            raise TypeError("cancelled must be callable")
        if cancelled():
            raise InterruptedError("Q3/v75 CPU regrid cancelled")
        if self._open_parent is not None:
            raise RuntimeError("Q3/v75 CPU already has an open transaction")
        self._validate_state(state, old_binding)
        transferred, residual = transfer_bed_mass_reference_v71(
            state.direct_state.bed_mass_kg_m2,
            old_binding.wetdry_topology, new_binding.wetdry_topology, plan,
        )
        target_parent = self.rows[new_binding.topology_id][
            "live_taurat_execution"]["parent_execution"]
        direct = resolve_direct_vector_family_state_v67(
            execution=target_parent, bed_mass_kg_m2=transferred,
            boundary_signed_kg=state.direct_state.boundary_signed_kg,
            boundary_absolute_kg=state.direct_state.boundary_absolute_kg,
            accepted_step=state.direct_state.accepted_step,
            time_s=state.direct_state.time_s,
        )
        cleared_flux = np.zeros(
            (self.nclass, new_binding.wetdry_topology.nface), dtype=np.float64,
        )
        cleared_availability = np.zeros(
            (self.nclass, new_binding.wetdry_topology.ncell), dtype=np.float64,
        )
        candidate = self._make_state(
            new_binding, direct, mesh_epoch=state.mesh_epoch+1,
            accepted_face_flux=cleared_flux,
            accepted_cell_availability=cleared_availability,
        )
        same_cover = (
            plan.classification == "identity_or_relabel" and
            np.array_equal(old_binding.wetdry_topology.node_ids,
                           new_binding.wetdry_topology.node_ids) and
            np.array_equal(old_binding.wetdry_topology.cell_area_m2,
                           new_binding.wetdry_topology.cell_area_m2)
        )
        wet_changes = (
            int(np.count_nonzero(
                old_binding.wetdry_topology.wet !=
                new_binding.wetdry_topology.wet,
            )) if same_cover else 0
        )
        transition = (
            "wetdry" if same_cover and wet_changes else plan.classification
        )
        body = {
            "scope_id": SCOPE_ID, "kind": "regrid",
            "parent_state_sha256": state.state_sha256,
            "candidate_state_sha256": candidate.state_sha256,
            "old_topology_id": old_binding.topology_id,
            "new_topology_id": new_binding.topology_id,
            "transfer_plan_id": plan.plan_id,
            "transition_kind": transition,
            "wet_state_change_count": wet_changes,
            "transfer_application_count": 1,
            "face_register_cleanup_count": 1,
            "availability_cleanup_count": 1,
            "exner_application_count": 0,
            "arrays_sha256": _array_digest(
                residual, cleared_flux, cleared_availability,
            ),
        }
        trial = CPUUnifiedRegridTrialV75(
            state.state_sha256, old_binding.topology_id,
            new_binding.topology_id, plan.plan_id, transition, wet_changes,
            _readonly(residual, np.float64, residual.shape, "regrid residual"),
            _readonly(cleared_flux, np.float64, cleared_flux.shape,
                      "cleared face register"),
            _readonly(cleared_availability, np.float64,
                      cleared_availability.shape,
                      "cleared availability register"),
            candidate, 1, 1, 1, 0, _fingerprint(body),
        )
        self._open_parent = state.state_sha256
        return trial

    def _validate_regrid(self, state, old_binding, new_binding, plan, trial):
        self._validate_state(state, old_binding)
        if (not isinstance(trial, CPUUnifiedRegridTrialV75) or
                self._open_parent != state.state_sha256 or
                trial.parent_state_sha256 != state.state_sha256 or
                trial.old_topology_id != old_binding.topology_id or
                trial.new_topology_id != new_binding.topology_id or
                trial.transfer_plan_id != plan.plan_id or
                (trial.transfer_application_count,
                 trial.face_register_cleanup_count,
                 trial.availability_cleanup_count,
                 trial.exner_application_count) != (1, 1, 1, 0)):
            raise UnifiedDirectVectorV75Error(
                "Q3/v75 CPU regrid receipt differs"
            )
        body = {
            "scope_id": SCOPE_ID, "kind": "regrid",
            "parent_state_sha256": state.state_sha256,
            "candidate_state_sha256": trial.candidate.state_sha256,
            "old_topology_id": old_binding.topology_id,
            "new_topology_id": new_binding.topology_id,
            "transfer_plan_id": plan.plan_id,
            "transition_kind": trial.transition_kind,
            "wet_state_change_count": trial.wet_state_change_count,
            "transfer_application_count": 1,
            "face_register_cleanup_count": 1,
            "availability_cleanup_count": 1,
            "exner_application_count": 0,
            "arrays_sha256": _array_digest(
                trial.conservation_residual_kg,
                trial.cleared_face_register,
                trial.cleared_cell_availability,
            ),
        }
        if trial.trial_sha256 != _fingerprint(body):
            raise UnifiedDirectVectorV75Error(
                "Q3/v75 CPU regrid arrays are stale"
            )

    def commit_regrid(self, state, old_binding, new_binding, plan, trial):
        self._validate_regrid(state, old_binding, new_binding, plan, trial)
        self._open_parent = None
        return trial.candidate

    def rollback_regrid(self, state, old_binding, new_binding, plan, trial):
        self._validate_regrid(state, old_binding, new_binding, plan, trial)
        self._open_parent = None
        return state

    def cancel_regrid(self, state, old_binding, new_binding, plan, trial):
        return self.rollback_regrid(
            state, old_binding, new_binding, plan, trial,
        )

    def checkpoint(self, state: CPUUnifiedStateV75,
                   binding: CPUUnifiedBindingV75) -> dict[str, object]:
        self._validate_state(state, binding)
        body = {
            "schema": SCHEMA,
            "scope_id": CHECKPOINT_SCOPE_ID,
            "execution_fingerprint": self.execution["fingerprint"],
            "topology_id": binding.topology_id,
            "binding_sha256": binding.binding_sha256,
            "mesh_epoch": state.mesh_epoch,
            "accepted_face_flux": state.accepted_face_flux.tolist(),
            "accepted_cell_availability":
                state.accepted_cell_availability.tolist(),
            "direct_state": direct_vector_family_state_checkpoint_v67(
                state.direct_state,
            ),
            "adaptation_restart": (
                None if binding.adaptation_execution is None else
                restart_document_v72(binding.adaptation_execution)
            ),
            "derived_adaptation_arrays_serialized": False,
            "state_sha256": state.state_sha256,
        }
        return {**body, "fingerprint": _fingerprint(body)}

    def restart(self, value: Mapping[str, object],
                binding: CPUUnifiedBindingV75) -> CPUUnifiedStateV75:
        if not isinstance(value, Mapping):
            raise UnifiedDirectVectorV75Error(
                "Q3/v75 CPU checkpoint must be an object"
            )
        body = {key: value[key] for key in value if key != "fingerprint"}
        if (value.get("fingerprint") != _fingerprint(body) or
                value.get("schema") != SCHEMA or
                value.get("scope_id") != CHECKPOINT_SCOPE_ID or
                value.get("execution_fingerprint") !=
                    self.execution["fingerprint"] or
                value.get("topology_id") != binding.topology_id or
                value.get("binding_sha256") != binding.binding_sha256 or
                value.get("derived_adaptation_arrays_serialized") is not False):
            raise UnifiedDirectVectorV75Error(
                "Q3/v75 CPU checkpoint identity differs"
            )
        parent = self.rows[binding.topology_id][
            "live_taurat_execution"]["parent_execution"]
        direct = load_direct_vector_family_state_checkpoint_v67(
            value["direct_state"], execution=parent,
        )
        state = self._make_state(
            binding, direct, mesh_epoch=value["mesh_epoch"],
            accepted_face_flux=value["accepted_face_flux"],
            accepted_cell_availability=value["accepted_cell_availability"],
        )
        if state.state_sha256 != value.get("state_sha256"):
            raise UnifiedDirectVectorV75Error(
                "Q3/v75 CPU checkpoint state digest differs"
            )
        return state


__all__ = [
    "BACKENDS", "CHECKPOINT_SCOPE_ID", "CPUUnifiedBindingV75",
    "CPUUnifiedRegridTrialV75", "CPUUnifiedStateV75", "CPUUnifiedTrialV75",
    "DirectVectorCapacityV75", "ENVELOPE_VERSION", "MAX_ACCEPTED_STEPS",
    "PortableUnifiedDirectVectorV75", "SCHEMA", "SCOPE_ID",
    "UnifiedDirectVectorV75Error", "load_unified_direct_vector_execution_v75",
    "seal_cpu_capacity_v75", "unified_direct_vector_execution_document_v75",
    "verify_unified_direct_vector_sources_v75",
]
