"""Authenticated Q3 noncohesive bed-transport boundary owner.

This additive component captures the classic-FLOW ordering needed by the
scalar and direct-vector capacity families without claiming the later Q6 bed
level boundary implementation.  Constitutive vectors are already available
when this owner is called.  A hydraulically incoming face either uses its
equilibrium trace vector or an external prescribed feed.  A reversed face is
always a free, owner-state export; a closed face is exactly zero.

The public prescribed modes retain the three classic ``IBedCond`` units.  The
transaction converts them to mass per width once.  Prescribed external mass
is not multiplied by local bed fractions or finite availability.  Equilibrium
inflow and owner export are, matching ``adjust_bedload`` before the later
``bott3d`` prescribed-transport overwrite.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np


SCHEMA = 1
ENVELOPE_VERSION = 70
SCOPE_ID = "braids-q3-noncohesive-bed-boundary-live-v70-v1"
MAX_ACCEPTED_STEPS = 4

MODE_CLOSED = "bed_transport.closed"
MODE_EQUILIBRIUM = "bed_transport.equilibrium_capacity"
MODE_PRESCRIBED_PORE_VOLUME = "bed_transport.prescribed_volume_with_pores"
MODE_PRESCRIBED_SOLID_VOLUME = "bed_transport.prescribed_volume_without_pores"
MODE_PRESCRIBED_MASS = "bed_transport.prescribed_mass"
MODE_FREE_EXPORT = "bed_transport.free_export"

INTERNAL_CLOSED = 0
INTERNAL_EQUILIBRIUM = 1
INTERNAL_PRESCRIBED = 2
INTERNAL_FREE_EXPORT = 3

SOURCE_NONE = 0
SOURCE_PRESCRIBED_EXTERNAL = 1
SOURCE_EQUILIBRIUM_TRACE = 2
SOURCE_OWNER_FREE_EXPORT = 3

PUBLIC_MODE_DOCUMENT = MappingProxyType({
    MODE_CLOSED: {
        "classic_ibedcond": None,
        "internal_mode": INTERNAL_CLOSED,
        "rate_unit": "exact_zero",
    },
    MODE_EQUILIBRIUM: {
        "classic_ibedcond": None,
        "internal_mode": INTERNAL_EQUILIBRIUM,
        "rate_unit": "computed_capacity",
    },
    MODE_PRESCRIBED_PORE_VOLUME: {
        "classic_ibedcond": 4,
        "internal_mode": INTERNAL_PRESCRIBED,
        "rate_unit": "m3_bulk_bed_with_pores_per_m_s",
    },
    MODE_PRESCRIBED_SOLID_VOLUME: {
        "classic_ibedcond": 5,
        "internal_mode": INTERNAL_PRESCRIBED,
        "rate_unit": "m3_solid_per_m_s",
    },
    MODE_PRESCRIBED_MASS: {
        "classic_ibedcond": 8,
        "internal_mode": INTERNAL_PRESCRIBED,
        "rate_unit": "kg_per_m_s",
    },
    MODE_FREE_EXPORT: {
        # This is the transport-face consequence of FLOW's automatic
        # outflow/free resolution, not a claim that Q6 IBedCond=0 bed-level
        # copying has been implemented.
        "classic_ibedcond": None,
        "internal_mode": INTERNAL_FREE_EXPORT,
        "rate_unit": "computed_owner_capacity",
    },
})

LOCKED_SOURCES = MappingProxyType({
    "resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/"
    "flow2d3d_kernel/src/compute_sediment/erosed.f90":
        "4571fd7912233db03f597161af1a18702ee974483676004cda8186ccf24308c0",
    "resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/"
    "flow2d3d_kernel/src/compute_sediment/adjust_bedload.f90":
        "a3dae8226082977ec1ee78748e0d439f0f7356d267a9f33cba58636a61afdcba",
    "resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/"
    "flow2d3d_kernel/src/compute_sediment/bott3d.f90":
        "5abfee8230f66ba42b3eb87ff89948f65a0bfccc915a015c25f8c1412f4e47b1",
    "resources/Delft3D-main/src/utils_gpl/morphology/packages/"
    "morphology_io/src/rdmor.f90":
        "ab4e3a1bdf1290e790412375ea5a91cddf9039670948307169c38194314e0d49",
    "resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/"
    "flow2d3d_io/src/input/morbndfill.f90":
        "dbb243de1f153fc020ddeab75302a089706f99b1cbf986070eeeb927db89133c",
})

_SHA = re.compile(r"[0-9a-f]{64}\Z")


class BedloadBoundaryV70Error(ValueError):
    """The authenticated boundary contract or transaction is invalid."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise BedloadBoundaryV70Error(
            "Q3/v70 boundary data are not canonical JSON") from error


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise BedloadBoundaryV70Error(
            f"Q3/v70 {name} must be a lowercase SHA-256 identity")
    return value


def _finite(value: object, name: str, *, positive: bool = False,
            nonnegative: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)):
        raise BedloadBoundaryV70Error(f"Q3/v70 {name} must be numeric")
    result = float(value)
    if (not math.isfinite(result) or (positive and result <= 0.0) or
            (nonnegative and result < 0.0)):
        raise BedloadBoundaryV70Error(
            f"Q3/v70 {name} is outside its finite domain")
    return 0.0 if result == 0.0 else result


def _array(value: object, shape: tuple[int, ...], name: str, *,
           dtype=np.float64, integer: bool = False) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != shape or (integer and raw.dtype.kind not in "iu"):
        raise BedloadBoundaryV70Error(
            f"Q3/v70 {name} must have shape {shape}")
    result = np.array(raw, dtype=dtype, copy=True, order="C")
    if not integer and not np.all(np.isfinite(result)):
        raise BedloadBoundaryV70Error(f"Q3/v70 {name} must be finite")
    result.setflags(write=False)
    return result


def _array_digest(*values: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in values:
        data = np.ascontiguousarray(value)
        digest.update(str(data.dtype).encode("ascii"))
        digest.update(_canonical(list(data.shape)))
        digest.update(data.tobytes(order="C"))
    return digest.hexdigest()


def verify_bedload_boundary_sources_v70(root: Path | str = ".") -> dict[str, str]:
    """Authenticate the locked, unmodified classic-FLOW source bytes."""
    base = Path(root)
    actual: dict[str, str] = {}
    for relative, expected in LOCKED_SOURCES.items():
        path = base / relative
        if not path.is_file():
            raise BedloadBoundaryV70Error(
                f"Q3/v70 locked source is missing: {relative}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise BedloadBoundaryV70Error(
                f"Q3/v70 locked source differs: {relative}")
        actual[relative] = digest
    return actual


def bedload_boundary_semantic_document_v70() -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "envelope_version": ENVELOPE_VERSION,
        "scope_id": SCOPE_ID,
        "transport_scope": "noncohesive_bed_vector_only_not_Q4_suspension",
        "public_modes": {key: dict(value)
                         for key, value in PUBLIC_MODE_DOCUMENT.items()},
        "outward_sign_convention": "positive_mass_leaves_domain",
        "inward_prescription": "nonnegative_input_becomes_negative_face_flux",
        "source_order": [
            "erosed_constitutive_capacity",
            "upwbed_and_adjust_bedload",
            "active_fraction_and_fixfac_for_computed_capacity",
            "sum_current_wave_channels",
            "bott3d_hydraulic_direction_test_per_boundary_point",
            "incoming_ibedcond_4_5_8_prescribed_transport_overwrite",
            "outgoing_reversal_retains_free_computed_transport",
            "interior_only_transport_divergence",
            "bed_level_boundary_condition_after_layer_update",
        ],
        "availability": {
            "equilibrium_inflow": "owner_active_fraction_times_global_fill",
            "free_export": "owner_active_fraction_times_global_fill",
            "prescribed_external_feed": "bypasses_owner_fraction_and_fill",
            "application_count": 1,
            "q3_equilibrium_composition_rationale": (
                "adjust_bedload applies frc*fixf before bott3d and keeps "
                "fixf on the owner side at open inflow; classic frc may be "
                "upwind/exterior and belongs to the unimplemented Q6 "
                "composition boundary. Q3 therefore uses owner composition "
                "explicitly as its bounded abstraction, not as a Q6 claim"
            ),
        },
        "flow_reversal": (
            "hydraulic_outward_sign_selects_owner_free_export; prescribed "
            "or equilibrium feed is used only for hydraulic inflow"
        ),
        "named_boundary_ledger": {
            "signed": "outward_positive_kg",
            "absolute": "absolute_kg",
            "cardinality": "fraction_by_named_boundary",
        },
        "bounded_accepted_steps": MAX_ACCEPTED_STEPS,
        "locked_sources": dict(LOCKED_SOURCES),
        "source_spans": {
            "erosed.f90": "1412-1491",
            "adjust_bedload.f90": "223-261,445-462,479-527",
            "bott3d.f90": "628-725,727-856,1144-1234",
            "rdmor.f90": "790-806,1990-2125",
            "morbndfill.f90": "122-235",
        },
        "explicit_nonclaims": [
            "Q4_suspended_concentration_or_Thatcher_Harleman",
            "Q6_time_scheduled_bed_boundary_modes_0_through_8",
            "Q6_bed_level_or_composition_boundary_updates",
            "wet_dry_or_AMR_integration",
            "product_or_QGIS_admission",
        ],
    }


SEMANTIC_ID = _fingerprint(bedload_boundary_semantic_document_v70())


def _mode_rates(mode: str, values: Sequence[object], bulk: np.ndarray,
                solid: np.ndarray, nclass: int, name: str) -> tuple[list[float], list[float]]:
    if mode not in PUBLIC_MODE_DOCUMENT:
        raise BedloadBoundaryV70Error(
            f"Q3/v70 boundary {name!r} has an unknown public mode")
    if len(values) != nclass:
        raise BedloadBoundaryV70Error(
            f"Q3/v70 boundary {name!r} must define one rate per fraction")
    source = [_finite(value, f"{name} prescribed rate", nonnegative=True)
              for value in values]
    if mode not in (MODE_PRESCRIBED_PORE_VOLUME,
                    MODE_PRESCRIBED_SOLID_VOLUME, MODE_PRESCRIBED_MASS):
        if any(source):
            raise BedloadBoundaryV70Error(
                f"Q3/v70 boundary {name!r} is not a prescribed-feed mode")
        return source, [0.0] * nclass
    factor = (bulk if mode == MODE_PRESCRIBED_PORE_VOLUME else
              solid if mode == MODE_PRESCRIBED_SOLID_VOLUME else
              np.ones(nclass, dtype=np.float64))
    return source, [float(source[index] * factor[index])
                    for index in range(nclass)]


def _build_bedload_boundary_execution_v70(
    *, backend: str, parent_execution_fingerprint: str,
    topology_fingerprint: str, fraction_ids: Sequence[str],
    bulk_density_kg_m3: Sequence[object], solid_density_kg_m3: Sequence[object],
    boundaries: Sequence[Mapping[str, object]], root: Path | str = ".",
    verify_sources: bool,
) -> dict[str, object]:
    allowed_backends = {
        "cpu_openmp": "parent_selected_cpu_openmp",
        "implicit_fwave_midpoint_gpu": "parent_selected_actual_cuda",
    }
    if backend not in allowed_backends:
        raise BedloadBoundaryV70Error(
            "Q3/v70 backend must be cpu_openmp or implicit_fwave_midpoint_gpu")
    parent = _sha(parent_execution_fingerprint, "parent execution")
    topology = _sha(topology_fingerprint, "topology dependency")
    fractions = tuple(fraction_ids)
    if (not fractions or any(not isinstance(item, str) or not item.strip()
                             for item in fractions) or
            len(set(fractions)) != len(fractions)):
        raise BedloadBoundaryV70Error(
            "Q3/v70 fraction IDs must be unique nonempty strings")
    nclass = len(fractions)
    bulk = _array(bulk_density_kg_m3, (nclass,), "bulk density")
    solid = _array(solid_density_kg_m3, (nclass,), "solid density")
    if np.any(bulk <= 0.0) or np.any(solid <= 0.0) or np.any(solid < bulk):
        raise BedloadBoundaryV70Error(
            "Q3/v70 densities must be positive with solid >= bulk")
    rows = []
    names: set[str] = set()
    for raw in boundaries:
        if not isinstance(raw, Mapping):
            raise BedloadBoundaryV70Error("Q3/v70 boundary row must be a mapping")
        name = raw.get("name")
        mode = raw.get("mode")
        if (not isinstance(name, str) or not name.strip() or name in names or
                not isinstance(mode, str)):
            raise BedloadBoundaryV70Error(
                "Q3/v70 named boundaries must be unique nonempty strings")
        names.add(name)
        source, mass = _mode_rates(
            mode, raw.get("rates", [0.0] * nclass), bulk, solid, nclass, name)
        document = PUBLIC_MODE_DOCUMENT[mode]
        rows.append({
            "name": name,
            "public_mode_id": mode,
            "classic_ibedcond": document["classic_ibedcond"],
            "internal_mode": document["internal_mode"],
            "source_rate_unit": document["rate_unit"],
            "source_rates": source,
            "inward_mass_flux_kg_m_s": mass,
            "schedule": "constant_Q3_only",
        })
    if not rows:
        raise BedloadBoundaryV70Error("Q3/v70 requires at least one named boundary")
    if verify_sources:
        verify_bedload_boundary_sources_v70(root)
    body = {
        "schema": SCHEMA,
        "envelope_version": ENVELOPE_VERSION,
        "scope_id": SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "backend": backend,
        "backend_selection": allowed_backends[backend],
        "integrator": ("implicit_fwave_midpoint" if backend.endswith("_gpu")
                       else "parent_cpu_integrator"),
        "automatic_backend_fallback": False,
        "parent_execution_fingerprint": parent,
        "topology_fingerprint": topology,
        "fraction_ids": list(fractions),
        "bulk_density_kg_m3": bulk.tolist(),
        "solid_density_kg_m3": solid.tolist(),
        "boundaries": rows,
        "locked_sources": dict(LOCKED_SOURCES),
        "accepted_step_limit": MAX_ACCEPTED_STEPS,
        "normal_product_admission": False,
    }
    body["fingerprint"] = _fingerprint(body)
    return body


def build_bedload_boundary_execution_v70(
    *, backend: str, parent_execution_fingerprint: str,
    topology_fingerprint: str, fraction_ids: Sequence[str],
    bulk_density_kg_m3: Sequence[object], solid_density_kg_m3: Sequence[object],
    boundaries: Sequence[Mapping[str, object]], root: Path | str = ".",
) -> dict[str, object]:
    """Build a receipt after authenticating the locked reference checkout."""
    return _build_bedload_boundary_execution_v70(
        backend=backend,
        parent_execution_fingerprint=parent_execution_fingerprint,
        topology_fingerprint=topology_fingerprint,
        fraction_ids=fraction_ids,
        bulk_density_kg_m3=bulk_density_kg_m3,
        solid_density_kg_m3=solid_density_kg_m3,
        boundaries=boundaries, root=root, verify_sources=True)


def load_bedload_boundary_execution_v70(
    value: Mapping[str, object], *, root: Path | str = ".",
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise BedloadBoundaryV70Error("Q3/v70 execution receipt is missing")
    document = json.loads(_canonical(dict(value)).decode("ascii"))
    fingerprint = document.pop("fingerprint", None)
    if fingerprint != _fingerprint(document):
        raise BedloadBoundaryV70Error("Q3/v70 execution fingerprint differs")
    if (document.get("schema") != SCHEMA or
            document.get("envelope_version") != ENVELOPE_VERSION or
            document.get("scope_id") != SCOPE_ID or
            document.get("semantic_id") != SEMANTIC_ID or
            document.get("automatic_backend_fallback") is not False or
            document.get("accepted_step_limit") != MAX_ACCEPTED_STEPS or
            document.get("normal_product_admission") is not False or
            document.get("locked_sources") != dict(LOCKED_SOURCES)):
        raise BedloadBoundaryV70Error("Q3/v70 execution semantics differ")
    rebuilt = _build_bedload_boundary_execution_v70(
        backend=document["backend"],
        parent_execution_fingerprint=document["parent_execution_fingerprint"],
        topology_fingerprint=document["topology_fingerprint"],
        fraction_ids=document["fraction_ids"],
        bulk_density_kg_m3=document["bulk_density_kg_m3"],
        solid_density_kg_m3=document["solid_density_kg_m3"],
        boundaries=[{
            "name": row["name"], "mode": row["public_mode_id"],
            "rates": row["source_rates"],
        } for row in document["boundaries"]],
        root=root, verify_sources=False,
    )
    if rebuilt != dict(value):
        raise BedloadBoundaryV70Error("Q3/v70 execution receipt differs")
    return rebuilt


@dataclass(frozen=True)
class BedloadBoundaryTopologyV70:
    execution_fingerprint: str
    dependency_fingerprint: str
    ncell: int
    nface: int
    owner: np.ndarray
    boundary_index: np.ndarray
    mode: np.ndarray
    normal: np.ndarray
    face_length_m: np.ndarray
    cell_area_m2: np.ndarray
    prescribed_mass_kg_m_s: np.ndarray
    topology_sha256: str


@dataclass(frozen=True)
class BedloadBoundaryCandidateV70:
    execution_fingerprint: str
    topology_sha256: str
    potential_kg_m_s: np.ndarray
    requires_owner_availability: np.ndarray
    source: np.ndarray
    candidate_sha256: str


@dataclass(frozen=True)
class BedloadBoundaryStateV70:
    execution_fingerprint: str
    bed_mass_kg_m2: np.ndarray
    active_fraction: np.ndarray
    boundary_signed_kg: np.ndarray
    boundary_absolute_kg: np.ndarray
    accepted_step: int
    time_s: float
    state_sha256: str


@dataclass(frozen=True)
class BedloadBoundaryTrialV70:
    execution_fingerprint: str
    parent_state_sha256: str
    topology_sha256: str
    dt_s: float
    morphological_factor: float
    candidate_flux: BedloadBoundaryCandidateV70
    provisional_flux_kg_m_s: np.ndarray
    outgoing_demand_kg_m2: np.ndarray
    inventory_fill: np.ndarray
    accepted_face_availability: np.ndarray
    face_flux_kg_m_s: np.ndarray
    cell_delta_kg_m2: np.ndarray
    boundary_transfer_kg: np.ndarray
    boundary_absolute_transfer_kg: np.ndarray
    conservation_residual_kg: np.ndarray
    candidate: BedloadBoundaryStateV70
    active_fraction_application_count: int
    inventory_fill_application_count: int
    trial_sha256: str


def resolve_bedload_boundary_topology_v70(
    execution: Mapping[str, object], *, owner: object, boundary_index: object,
    normal: object, face_length_m: object, cell_area_m2: object,
) -> BedloadBoundaryTopologyV70:
    resolved = load_bedload_boundary_execution_v70(execution)
    owner_raw = np.asarray(owner)
    if owner_raw.ndim != 1 or owner_raw.dtype.kind not in "iu":
        raise BedloadBoundaryV70Error("Q3/v70 owner must be a 1-D integer array")
    nface = owner_raw.size
    area_raw = np.asarray(cell_area_m2)
    if area_raw.ndim != 1:
        raise BedloadBoundaryV70Error("Q3/v70 cell area must be one-dimensional")
    ncell = area_raw.size
    owners = _array(owner_raw, (nface,), "owner", dtype=np.int64, integer=True)
    bindex = _array(boundary_index, (nface,), "boundary index",
                    dtype=np.int64, integer=True)
    normals = _array(normal, (2, nface), "outward normal")
    lengths = _array(face_length_m, (nface,), "face length")
    areas = _array(area_raw, (ncell,), "cell area")
    nboundary = len(resolved["boundaries"])
    if (np.any(owners < 0) or np.any(owners >= ncell) or
            np.any(bindex < 0) or np.any(bindex >= nboundary) or
            np.any(lengths < 0.0) or np.any(areas <= 0.0)):
        raise BedloadBoundaryV70Error("Q3/v70 topology indices/geometry differ")
    norm = np.sqrt(np.sum(normals * normals, axis=0))
    if np.any(np.abs(norm - 1.0) > 1.0e-12):
        raise BedloadBoundaryV70Error("Q3/v70 outward normals must be unit length")
    modes_by_boundary = np.asarray(
        [row["internal_mode"] for row in resolved["boundaries"]],
        dtype=np.int32,
    )
    rates_by_boundary = np.asarray(
        [row["inward_mass_flux_kg_m_s"] for row in resolved["boundaries"]],
        dtype=np.float64,
    ).T
    modes = np.ascontiguousarray(modes_by_boundary[bindex])
    rates = np.ascontiguousarray(rates_by_boundary[:, bindex])
    modes.setflags(write=False)
    rates.setflags(write=False)
    token = _fingerprint({
        "scope_id": SCOPE_ID,
        "execution_fingerprint": resolved["fingerprint"],
        "dependency_fingerprint": resolved["topology_fingerprint"],
        "arrays_sha256": _array_digest(
            owners, bindex, modes, normals, lengths, areas, rates),
    })
    return BedloadBoundaryTopologyV70(
        resolved["fingerprint"], resolved["topology_fingerprint"], ncell,
        nface, owners, bindex, modes, normals, lengths, areas, rates, token)


def _validate_topology(topology: BedloadBoundaryTopologyV70,
                       execution: Mapping[str, object]) -> dict[str, object]:
    resolved = load_bedload_boundary_execution_v70(execution)
    if (not isinstance(topology, BedloadBoundaryTopologyV70) or
            topology.execution_fingerprint != resolved["fingerprint"] or
            topology.dependency_fingerprint != resolved["topology_fingerprint"]):
        raise BedloadBoundaryV70Error("Q3/v70 topology receipt differs")
    expected = _fingerprint({
        "scope_id": SCOPE_ID,
        "execution_fingerprint": resolved["fingerprint"],
        "dependency_fingerprint": resolved["topology_fingerprint"],
        "arrays_sha256": _array_digest(
            topology.owner, topology.boundary_index, topology.mode,
            topology.normal, topology.face_length_m, topology.cell_area_m2,
            topology.prescribed_mass_kg_m_s),
    })
    if expected != topology.topology_sha256:
        raise BedloadBoundaryV70Error("Q3/v70 topology arrays are stale")
    return resolved


def resolve_bedload_boundary_state_v70(
    execution: Mapping[str, object], *, bed_mass_kg_m2: object,
    active_fraction: object, boundary_signed_kg: object | None = None,
    boundary_absolute_kg: object | None = None, accepted_step: int = 0,
    time_s: float = 0.0,
) -> BedloadBoundaryStateV70:
    resolved = load_bedload_boundary_execution_v70(execution)
    nclass = len(resolved["fraction_ids"])
    raw = np.asarray(bed_mass_kg_m2)
    if raw.ndim != 2 or raw.shape[0] != nclass:
        raise BedloadBoundaryV70Error(
            "Q3/v70 bed mass must have shape (nfraction,ncell)")
    ncell = raw.shape[1]
    mass = _array(raw, (nclass, ncell), "bed mass")
    fraction = _array(active_fraction, (nclass, ncell), "active fraction")
    nboundary = len(resolved["boundaries"])
    signed = _array(
        np.zeros((nclass, nboundary)) if boundary_signed_kg is None
        else boundary_signed_kg,
        (nclass, nboundary), "signed boundary ledger")
    absolute = _array(
        np.zeros((nclass, nboundary)) if boundary_absolute_kg is None
        else boundary_absolute_kg,
        (nclass, nboundary), "absolute boundary ledger")
    if (np.any(mass < 0.0) or np.any(fraction < 0.0) or
            np.any(fraction > 1.0) or np.any(absolute < 0.0)):
        raise BedloadBoundaryV70Error("Q3/v70 state values are outside domain")
    volume = mass / np.asarray(resolved["bulk_density_kg_m3"])[:, None]
    positive = np.sum(volume, axis=0) > 0.0
    if np.any(np.abs(np.sum(fraction[:, positive], axis=0) - 1.0) > 1e-12):
        raise BedloadBoundaryV70Error(
            "Q3/v70 active fractions must sum to one in stocked cells")
    if (isinstance(accepted_step, bool) or not isinstance(accepted_step, int) or
            accepted_step < 0 or accepted_step > MAX_ACCEPTED_STEPS):
        raise BedloadBoundaryV70Error("Q3/v70 accepted step is invalid")
    time = _finite(time_s, "state time", nonnegative=True)
    token = _fingerprint({
        "scope_id": SCOPE_ID,
        "execution_fingerprint": resolved["fingerprint"],
        "accepted_step": accepted_step,
        "time_s": time,
        "arrays_sha256": _array_digest(mass, fraction, signed, absolute),
    })
    return BedloadBoundaryStateV70(
        resolved["fingerprint"], mass, fraction, signed, absolute,
        accepted_step, time, token)


def _validate_state(state: BedloadBoundaryStateV70,
                    execution: Mapping[str, object]) -> dict[str, object]:
    resolved = load_bedload_boundary_execution_v70(execution)
    if (not isinstance(state, BedloadBoundaryStateV70) or
            state.execution_fingerprint != resolved["fingerprint"]):
        raise BedloadBoundaryV70Error("Q3/v70 state receipt differs")
    rebuilt = resolve_bedload_boundary_state_v70(
        resolved, bed_mass_kg_m2=state.bed_mass_kg_m2,
        active_fraction=state.active_fraction,
        boundary_signed_kg=state.boundary_signed_kg,
        boundary_absolute_kg=state.boundary_absolute_kg,
        accepted_step=state.accepted_step, time_s=state.time_s)
    if rebuilt.state_sha256 != state.state_sha256:
        raise BedloadBoundaryV70Error("Q3/v70 state arrays are stale")
    return resolved


def resolve_bedload_boundary_candidate_v70(
    execution: Mapping[str, object], topology: BedloadBoundaryTopologyV70, *,
    hydraulic_outward_flux: object, owner_bed_mass_vector: object,
    trace_bed_mass_vector: object,
) -> BedloadBoundaryCandidateV70:
    resolved = _validate_topology(topology, execution)
    nclass = len(resolved["fraction_ids"])
    water = _array(hydraulic_outward_flux, (topology.nface,),
                   "hydraulic outward flux")
    owner_vector = _array(owner_bed_mass_vector,
                          (2, nclass, topology.ncell), "owner bed vector")
    trace_vector = _array(trace_bed_mass_vector,
                          (2, nclass, topology.nface), "trace bed vector")
    potential = np.zeros((nclass, topology.nface), dtype=np.float64)
    requires = np.zeros((nclass, topology.nface), dtype=np.int32)
    source = np.zeros((nclass, topology.nface), dtype=np.int32)
    for face in range(topology.nface):
        mode = int(topology.mode[face])
        owner = int(topology.owner[face])
        if mode == INTERNAL_CLOSED or water[face] == 0.0:
            continue
        for fraction in range(nclass):
            if water[face] < 0.0:
                if mode == INTERNAL_PRESCRIBED:
                    potential[fraction, face] = -float(
                        topology.prescribed_mass_kg_m_s[fraction, face])
                    source[fraction, face] = SOURCE_PRESCRIBED_EXTERNAL
                elif mode == INTERNAL_EQUILIBRIUM:
                    projected = float(np.dot(
                        trace_vector[:, fraction, face],
                        topology.normal[:, face]))
                    potential[fraction, face] = min(0.0, projected)
                    requires[fraction, face] = 1
                    source[fraction, face] = SOURCE_EQUILIBRIUM_TRACE
            else:
                projected = float(np.dot(
                    owner_vector[:, fraction, owner],
                    topology.normal[:, face]))
                potential[fraction, face] = max(0.0, projected)
                requires[fraction, face] = 1
                source[fraction, face] = SOURCE_OWNER_FREE_EXPORT
    for value in (potential, requires, source):
        value.setflags(write=False)
    token = _fingerprint({
        "scope_id": SCOPE_ID,
        "execution_fingerprint": resolved["fingerprint"],
        "topology_sha256": topology.topology_sha256,
        "arrays_sha256": _array_digest(potential, requires, source),
    })
    return BedloadBoundaryCandidateV70(
        resolved["fingerprint"], topology.topology_sha256, potential,
        requires, source, token)


def begin_bedload_boundary_trial_v70(
    execution: Mapping[str, object], *, state: BedloadBoundaryStateV70,
    topology: BedloadBoundaryTopologyV70, hydraulic_outward_flux: object,
    owner_bed_mass_vector: object, trace_bed_mass_vector: object,
    dt_s: float, morphological_factor: float = 1.0,
    cancelled=lambda: False,
) -> BedloadBoundaryTrialV70:
    if not callable(cancelled):
        raise TypeError("cancelled must be callable")
    if cancelled():
        raise InterruptedError("Q3/v70 boundary transaction cancelled before launch")
    resolved = _validate_state(state, execution)
    _validate_topology(topology, resolved)
    if state.accepted_step >= MAX_ACCEPTED_STEPS:
        raise BedloadBoundaryV70Error("Q3/v70 accepted-step limit reached")
    if state.bed_mass_kg_m2.shape[1] != topology.ncell:
        raise BedloadBoundaryV70Error("Q3/v70 state/topology cell count differs")
    dt = _finite(dt_s, "trial dt", positive=True)
    morfac = _finite(morphological_factor, "morphological factor",
                     nonnegative=True)
    candidate_flux = resolve_bedload_boundary_candidate_v70(
        resolved, topology, hydraulic_outward_flux=hydraulic_outward_flux,
        owner_bed_mass_vector=owner_bed_mass_vector,
        trace_bed_mass_vector=trace_bed_mass_vector)
    nclass, ncell = state.bed_mass_kg_m2.shape
    provisional = np.array(candidate_flux.potential_kg_m_s, copy=True)
    mask = candidate_flux.requires_owner_availability.astype(bool)
    for face in range(topology.nface):
        provisional[mask[:, face], face] *= state.active_fraction[
            mask[:, face], topology.owner[face]]
    demand = np.zeros((nclass, ncell), dtype=np.float64)
    for face in range(topology.nface):
        owner = int(topology.owner[face])
        outward = np.maximum(provisional[:, face], 0.0)
        demand[:, owner] += (
            outward * topology.face_length_m[face] * dt * morfac /
            topology.cell_area_m2[owner])
    fill = np.ones((nclass, ncell), dtype=np.float64)
    fill[state.bed_mass_kg_m2 == 0.0] = 0.0
    positive = demand > 0.0
    fill[positive] = np.minimum(
        1.0, state.bed_mass_kg_m2[positive] / demand[positive])
    availability = state.active_fraction * fill
    flux = np.array(candidate_flux.potential_kg_m_s, copy=True)
    for face in range(topology.nface):
        flux[mask[:, face], face] *= availability[
            mask[:, face], topology.owner[face]]
    delta = np.zeros((nclass, ncell), dtype=np.float64)
    nboundary = len(resolved["boundaries"])
    transfer = np.zeros((nclass, nboundary), dtype=np.float64)
    absolute = np.zeros_like(transfer)
    for face in range(topology.nface):
        owner = int(topology.owner[face])
        boundary = int(topology.boundary_index[face])
        amount = flux[:, face] * topology.face_length_m[face] * dt * morfac
        delta[:, owner] -= amount / topology.cell_area_m2[owner]
        transfer[:, boundary] += amount
        absolute[:, boundary] += np.abs(amount)
    candidate_mass = state.bed_mass_kg_m2 + delta
    tolerance = 2048.0 * np.finfo(np.float64).eps * np.maximum(
        1.0, np.maximum(state.bed_mass_kg_m2, np.abs(delta)))
    if np.any(candidate_mass < -tolerance):
        raise BedloadBoundaryV70Error(
            "Q3/v70 finite-inventory projection failed")
    candidate_mass = np.maximum(candidate_mass, 0.0)
    bulk = np.asarray(resolved["bulk_density_kg_m3"])[:, None]
    volume = candidate_mass / bulk
    total = np.sum(volume, axis=0)
    fractions = np.array(state.active_fraction, copy=True)
    stocked = total > 0.0
    fractions[:, stocked] = volume[:, stocked] / total[stocked]
    residual = np.sum(delta * topology.cell_area_m2[None, :], axis=1) + \
        np.sum(transfer, axis=1)
    residual_tolerance = 4096.0 * np.finfo(np.float64).eps * np.maximum(
        1.0, np.sum(np.abs(delta * topology.cell_area_m2[None, :]), axis=1)
        + np.sum(absolute, axis=1))
    if np.any(np.abs(residual) > residual_tolerance):
        raise BedloadBoundaryV70Error("Q3/v70 conservation ledger does not close")
    next_state = resolve_bedload_boundary_state_v70(
        resolved, bed_mass_kg_m2=candidate_mass,
        active_fraction=fractions,
        boundary_signed_kg=state.boundary_signed_kg + transfer,
        boundary_absolute_kg=state.boundary_absolute_kg + absolute,
        accepted_step=state.accepted_step + 1, time_s=state.time_s + dt)
    arrays = (provisional, demand, fill, availability, flux, delta, transfer,
              absolute, residual)
    for value in arrays:
        value.setflags(write=False)
    token = _fingerprint({
        "scope_id": SCOPE_ID,
        "execution_fingerprint": resolved["fingerprint"],
        "parent_state_sha256": state.state_sha256,
        "topology_sha256": topology.topology_sha256,
        "candidate_flux_sha256": candidate_flux.candidate_sha256,
        "candidate_state_sha256": next_state.state_sha256,
        "dt_s": dt,
        "morphological_factor": morfac,
        "active_fraction_application_count": 1,
        "inventory_fill_application_count": 1,
        "arrays_sha256": _array_digest(*arrays),
    })
    trial = BedloadBoundaryTrialV70(
        resolved["fingerprint"], state.state_sha256,
        topology.topology_sha256, dt, morfac, candidate_flux, provisional,
        demand, fill, availability, flux, delta, transfer, absolute, residual,
        next_state, 1, 1, token)
    if cancelled():
        raise InterruptedError(
            "Q3/v70 boundary transaction cancelled before publication")
    return trial


def _validate_trial(execution: Mapping[str, object], state: BedloadBoundaryStateV70,
                    trial: BedloadBoundaryTrialV70) -> dict[str, object]:
    resolved = _validate_state(state, execution)
    if (not isinstance(trial, BedloadBoundaryTrialV70) or
            trial.execution_fingerprint != resolved["fingerprint"] or
            trial.parent_state_sha256 != state.state_sha256 or
            trial.active_fraction_application_count != 1 or
            trial.inventory_fill_application_count != 1):
        raise BedloadBoundaryV70Error("Q3/v70 trial parent/count differs")
    candidate_token = _fingerprint({
        "scope_id": SCOPE_ID,
        "execution_fingerprint": resolved["fingerprint"],
        "topology_sha256": trial.topology_sha256,
        "arrays_sha256": _array_digest(
            trial.candidate_flux.potential_kg_m_s,
            trial.candidate_flux.requires_owner_availability,
            trial.candidate_flux.source),
    })
    if candidate_token != trial.candidate_flux.candidate_sha256:
        raise BedloadBoundaryV70Error("Q3/v70 candidate arrays are stale")
    _validate_state(trial.candidate, resolved)
    token = _fingerprint({
        "scope_id": SCOPE_ID,
        "execution_fingerprint": resolved["fingerprint"],
        "parent_state_sha256": state.state_sha256,
        "topology_sha256": trial.topology_sha256,
        "candidate_flux_sha256": trial.candidate_flux.candidate_sha256,
        "candidate_state_sha256": trial.candidate.state_sha256,
        "dt_s": trial.dt_s,
        "morphological_factor": trial.morphological_factor,
        "active_fraction_application_count": 1,
        "inventory_fill_application_count": 1,
        "arrays_sha256": _array_digest(
            trial.provisional_flux_kg_m_s, trial.outgoing_demand_kg_m2,
            trial.inventory_fill, trial.accepted_face_availability,
            trial.face_flux_kg_m_s, trial.cell_delta_kg_m2,
            trial.boundary_transfer_kg,
            trial.boundary_absolute_transfer_kg,
            trial.conservation_residual_kg),
    })
    if token != trial.trial_sha256:
        raise BedloadBoundaryV70Error("Q3/v70 trial arrays are stale")
    return resolved


def commit_bedload_boundary_trial_v70(
    execution: Mapping[str, object], state: BedloadBoundaryStateV70,
    trial: BedloadBoundaryTrialV70,
) -> BedloadBoundaryStateV70:
    _validate_trial(execution, state, trial)
    return trial.candidate


def rollback_bedload_boundary_trial_v70(
    execution: Mapping[str, object], state: BedloadBoundaryStateV70,
    trial: BedloadBoundaryTrialV70,
) -> BedloadBoundaryStateV70:
    _validate_trial(execution, state, trial)
    return state


def cancel_bedload_boundary_trial_v70(
    execution: Mapping[str, object], state: BedloadBoundaryStateV70,
    trial: BedloadBoundaryTrialV70,
) -> BedloadBoundaryStateV70:
    return rollback_bedload_boundary_trial_v70(execution, state, trial)


def bedload_boundary_checkpoint_v70(
    execution: Mapping[str, object], topology: BedloadBoundaryTopologyV70,
    state: BedloadBoundaryStateV70,
) -> dict[str, object]:
    resolved = _validate_state(state, execution)
    _validate_topology(topology, resolved)
    body = {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "execution_fingerprint": resolved["fingerprint"],
        "topology_sha256": topology.topology_sha256,
        "topology_dependency_fingerprint": topology.dependency_fingerprint,
        "boundary_names": [row["name"] for row in resolved["boundaries"]],
        "bed_mass_kg_m2": state.bed_mass_kg_m2.tolist(),
        "active_fraction": state.active_fraction.tolist(),
        "boundary_signed_kg": state.boundary_signed_kg.tolist(),
        "boundary_absolute_kg": state.boundary_absolute_kg.tolist(),
        "accepted_step": state.accepted_step,
        "time_s": state.time_s,
        "state_sha256": state.state_sha256,
    }
    body["fingerprint"] = _fingerprint(body)
    return body


def load_bedload_boundary_checkpoint_v70(
    execution: Mapping[str, object], topology: BedloadBoundaryTopologyV70,
    value: Mapping[str, object],
) -> BedloadBoundaryStateV70:
    resolved = _validate_topology(topology, execution)
    if not isinstance(value, Mapping):
        raise BedloadBoundaryV70Error("Q3/v70 checkpoint is missing")
    document = json.loads(_canonical(dict(value)).decode("ascii"))
    fingerprint = document.pop("fingerprint", None)
    if fingerprint != _fingerprint(document):
        raise BedloadBoundaryV70Error("Q3/v70 checkpoint fingerprint differs")
    if (document.get("schema") != SCHEMA or
            document.get("scope_id") != SCOPE_ID or
            document.get("execution_fingerprint") != resolved["fingerprint"] or
            document.get("topology_sha256") != topology.topology_sha256 or
            document.get("topology_dependency_fingerprint") !=
                topology.dependency_fingerprint or
            document.get("boundary_names") !=
                [row["name"] for row in resolved["boundaries"]]):
        raise BedloadBoundaryV70Error(
            "Q3/v70 checkpoint execution/topology/boundary identity differs")
    state = resolve_bedload_boundary_state_v70(
        resolved, bed_mass_kg_m2=document["bed_mass_kg_m2"],
        active_fraction=document["active_fraction"],
        boundary_signed_kg=document["boundary_signed_kg"],
        boundary_absolute_kg=document["boundary_absolute_kg"],
        accepted_step=document["accepted_step"], time_s=document["time_s"])
    if state.state_sha256 != document.get("state_sha256"):
        raise BedloadBoundaryV70Error("Q3/v70 checkpoint state differs")
    return state


__all__ = [
    "BedloadBoundaryCandidateV70", "BedloadBoundaryStateV70",
    "BedloadBoundaryTopologyV70", "BedloadBoundaryTrialV70",
    "BedloadBoundaryV70Error", "INTERNAL_CLOSED", "INTERNAL_EQUILIBRIUM",
    "INTERNAL_FREE_EXPORT", "INTERNAL_PRESCRIBED", "LOCKED_SOURCES",
    "MAX_ACCEPTED_STEPS", "MODE_CLOSED", "MODE_EQUILIBRIUM",
    "MODE_FREE_EXPORT", "MODE_PRESCRIBED_MASS",
    "MODE_PRESCRIBED_PORE_VOLUME", "MODE_PRESCRIBED_SOLID_VOLUME",
    "PUBLIC_MODE_DOCUMENT", "SEMANTIC_ID", "SOURCE_EQUILIBRIUM_TRACE",
    "SOURCE_NONE", "SOURCE_OWNER_FREE_EXPORT", "SOURCE_PRESCRIBED_EXTERNAL",
    "bedload_boundary_checkpoint_v70", "bedload_boundary_semantic_document_v70",
    "begin_bedload_boundary_trial_v70", "build_bedload_boundary_execution_v70",
    "cancel_bedload_boundary_trial_v70", "commit_bedload_boundary_trial_v70",
    "load_bedload_boundary_checkpoint_v70", "load_bedload_boundary_execution_v70",
    "resolve_bedload_boundary_candidate_v70", "resolve_bedload_boundary_state_v70",
    "resolve_bedload_boundary_topology_v70", "rollback_bedload_boundary_trial_v70",
    "verify_bedload_boundary_sources_v70",
]
