"""Zero-time 2:1 refine/coarsen transfer for the bounded v98 gate.

This is the conservative transfer operator used before real v97 leaf-state
construction.  It neither advances a clock nor creates a solver state.  Child
bed elevations supplied here must already be derived from the proposed child
classwise bed mass and the fine execution's reference geometry; derived EOS,
DSS and HLES are intentionally outside this routine.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Sequence

import numpy as np


SCHEMA = 1
SCOPE_ID = "braids.coupled_amr.zero_time_transfer.v98"
CHILD_COUNT = 4
_SHA = re.compile(r"[0-9a-f]{64}")


class CoupledAMRZeroTimeTransferV98Error(ValueError):
    """Fail-closed zero-time transfer error."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _digest(*values: object) -> str:
    digest = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(_canonical(list(array.shape)))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise CoupledAMRZeroTimeTransferV98Error(
            f"v98 {name} must be a lowercase SHA-256 digest")
    return value


def _finite(value: object, name: str, *, nonnegative: bool = False,
            positive: bool = False) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(float(value)) or
            (nonnegative and float(value) < 0.0) or
            (positive and float(value) <= 0.0)):
        raise CoupledAMRZeroTimeTransferV98Error(
            f"v98 {name} is outside its finite physical range")
    return float(value)


def _array(value: object, shape: tuple[int, ...], name: str, *,
           nonnegative: bool = False) -> np.ndarray:
    raw = np.asarray(value)
    if (raw.shape != shape or raw.dtype.kind not in "fiu" or
            not np.isfinite(raw).all() or
            (nonnegative and np.any(raw < 0.0))):
        raise CoupledAMRZeroTimeTransferV98Error(
            f"v98 {name} must be finite with shape {shape}")
    result = np.array(raw, dtype=np.float64, order="C", copy=True)
    result.setflags(write=False)
    return result


def _tolerance(*values: object) -> float:
    scale = 1.0
    for value in values:
        array = np.asarray(value, dtype=np.float64)
        if array.size:
            scale = max(scale, float(np.max(np.abs(array))))
    return 2048.0*np.finfo(np.float64).eps*scale


def _exact_integral_children(
    parent_value: np.ndarray,
    parent_area: float,
    child_area: np.ndarray,
    child_value: np.ndarray,
    *, nonnegative: bool,
    correction_index: int = -1,
) -> np.ndarray:
    """Close each lane by correcting one child without changing other lanes."""
    result = np.array(child_value, dtype=np.float64, order="C", copy=True)
    target = np.asarray(parent_value, dtype=np.float64)*parent_area
    current = np.sum(result*child_area, axis=-1, dtype=np.float64)
    result[..., correction_index] += (
        (target-current)/child_area[correction_index])
    current = np.sum(result*child_area, axis=-1, dtype=np.float64)
    result[..., correction_index] += (
        (target-current)/child_area[correction_index])
    if nonnegative:
        tolerance = _tolerance(result, target)
        if np.any(result < -tolerance):
            raise CoupledAMRZeroTimeTransferV98Error(
                "v98 exact child closure would make a conserved lane negative")
        result[result < 0.0] = 0.0
    residual = np.sum(result*child_area, axis=-1, dtype=np.float64)-target
    if np.any(np.abs(residual) > _tolerance(target, current)):
        raise CoupledAMRZeroTimeTransferV98Error(
            "v98 child conserved integral does not close")
    result.setflags(write=False)
    return result


def _stage_fill(
    parent_volume_m3: float,
    child_bed_elevation_m: np.ndarray,
    child_area_m2: np.ndarray,
    dry_tolerance_m: float,
) -> np.ndarray:
    """Fill by one stage over a dry-aware active child set."""
    volume = float(parent_volume_m3)
    if volume == 0.0:
        result = np.zeros(CHILD_COUNT, dtype=np.float64)
        result.setflags(write=False)
        return result
    bed = np.asarray(child_bed_elevation_m)
    area = np.asarray(child_area_m2)
    threshold = bed+dry_tolerance_m
    order = np.argsort(threshold, kind="stable")
    selected = None
    eta = None
    for count in range(1, CHILD_COUNT+1):
        active = order[:count]
        candidate = (volume+np.sum(area[active]*bed[active],
                                   dtype=np.float64))/np.sum(area[active])
        lower_ok = np.all(candidate > threshold[active])
        upper_ok = (count == CHILD_COUNT or
                    candidate <= threshold[order[count]])
        if lower_ok and upper_ok:
            selected, eta = active, float(candidate)
            break
    if selected is None or eta is None:
        raise CoupledAMRZeroTimeTransferV98Error(
            "v98 parent volume has no dry-aware child stage fill")
    depth = np.zeros(CHILD_COUNT, dtype=np.float64)
    depth[selected] = eta-bed[selected]
    residual = volume-np.sum(depth*area, dtype=np.float64)
    deepest = int(selected[np.argmin(bed[selected])])
    depth[deepest] += residual/area[deepest]
    residual = volume-np.sum(depth*area, dtype=np.float64)
    depth[deepest] += residual/area[deepest]
    positive = depth > 0.0
    if (np.any(depth < 0.0) or
            np.any(positive & (depth <= dry_tolerance_m)) or
            abs(np.sum(depth*area, dtype=np.float64)-volume) >
                float(_tolerance(volume))):
        raise CoupledAMRZeroTimeTransferV98Error(
            "v98 dry-aware stage fill is negative, sub-dry, or nonconservative")
    depth.setflags(write=False)
    return depth


@dataclass(frozen=True)
class CoupledAMRZeroTimeTransferV98:
    hierarchy_fingerprint: str
    parent_state_sha256: str
    fraction_count: int
    parent_area_m2: float
    child_area_m2: np.ndarray
    child_bed_elevation_m: np.ndarray
    dry_tolerance_m: float
    parent_depth_m: float
    parent_momentum_m2_s: np.ndarray
    parent_suspended_kg_m2: np.ndarray
    parent_bed_kg_m2: np.ndarray
    parent_fluid_integral: np.ndarray
    child_depth_m: np.ndarray
    child_momentum_m2_s: np.ndarray
    child_suspended_kg_m2: np.ndarray
    child_bed_kg_m2: np.ndarray
    child_fluid_integral: np.ndarray
    restricted_depth_m: float
    restricted_momentum_m2_s: np.ndarray
    restricted_suspended_kg_m2: np.ndarray
    restricted_bed_kg_m2: np.ndarray
    restricted_fluid_integral: np.ndarray
    boundary_source_ids: tuple[int, ...]
    boundary_memory_sha256: str
    accepted_updates: int
    time_s: float
    refine_event_count: int
    coarsen_event_count: int
    clock_increment_count: int
    constitutive_evaluation_count: int
    receipt_sha256: str


def _payload(value: CoupledAMRZeroTimeTransferV98) -> dict:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "hierarchy": value.hierarchy_fingerprint,
        "parent": value.parent_state_sha256,
        "fraction_count": value.fraction_count,
        "parent_area_m2": value.parent_area_m2,
        "dry_tolerance_m": value.dry_tolerance_m,
        "arrays_sha256": _digest(
            value.child_area_m2, value.child_bed_elevation_m,
            np.asarray(value.parent_depth_m),
            value.parent_momentum_m2_s,
            value.parent_suspended_kg_m2, value.parent_bed_kg_m2,
            value.parent_fluid_integral,
            value.child_depth_m, value.child_momentum_m2_s,
            value.child_suspended_kg_m2, value.child_bed_kg_m2,
            value.child_fluid_integral,
            np.asarray(value.restricted_depth_m),
            value.restricted_momentum_m2_s,
            value.restricted_suspended_kg_m2,
            value.restricted_bed_kg_m2,
            value.restricted_fluid_integral),
        "boundary_source_ids": list(value.boundary_source_ids),
        "boundary_memory_sha256": value.boundary_memory_sha256,
        "accepted_updates": value.accepted_updates,
        "time_s": value.time_s,
        "refine_event_count": value.refine_event_count,
        "coarsen_event_count": value.coarsen_event_count,
        "clock_increment_count": value.clock_increment_count,
        "constitutive_evaluation_count": value.constitutive_evaluation_count,
        "transfer": "stage_fill_refine_then_area_restrict_coarsen",
    }


def resolve_zero_time_refine_coarsen_v98(
    *, hierarchy_fingerprint: str, parent_state_sha256: str,
    parent_area_m2: float, child_area_m2: object,
    child_bed_elevation_m: object, dry_tolerance_m: float,
    parent_depth_m: float, parent_momentum_m2_s: object,
    parent_suspended_kg_m2: object, parent_bed_kg_m2: object,
    parent_fluid_integral: object,
    boundary_source_ids_before: Sequence[int],
    boundary_source_ids_after: Sequence[int],
    boundary_memory_before_sha256: str,
    boundary_memory_after_sha256: str,
    accepted_updates: int, time_s: float,
) -> CoupledAMRZeroTimeTransferV98:
    """Refine then coarsen one parent without advancing state or time."""
    hierarchy = _sha(hierarchy_fingerprint, "hierarchy fingerprint")
    parent_sha = _sha(parent_state_sha256, "parent state")
    area0 = _finite(parent_area_m2, "parent area", positive=True)
    area = _array(child_area_m2, (CHILD_COUNT,), "child area",
                  nonnegative=True)
    if np.any(area <= 0.0):
        raise CoupledAMRZeroTimeTransferV98Error(
            "v98 child areas must be positive")
    geometry_residual = float(np.sum(area, dtype=np.float64)-area0)
    if abs(geometry_residual) > float(np.max(_tolerance(area0, area))):
        raise CoupledAMRZeroTimeTransferV98Error(
            "v98 child areas do not partition the parent")
    bed_elevation = _array(
        child_bed_elevation_m, (CHILD_COUNT,), "child bed elevation")
    dry = _finite(dry_tolerance_m, "dry tolerance", positive=True)
    depth0 = _finite(parent_depth_m, "parent depth", nonnegative=True)
    momentum0 = np.asarray(parent_momentum_m2_s)
    if momentum0.shape != (2,) or not np.isfinite(momentum0).all():
        raise CoupledAMRZeroTimeTransferV98Error(
            "v98 parent momentum must have two finite components")
    suspended0 = np.asarray(parent_suspended_kg_m2)
    bed0 = np.asarray(parent_bed_kg_m2)
    if (suspended0.ndim != 1 or bed0.shape != suspended0.shape or
            suspended0.size < 1 or not np.isfinite(suspended0).all() or
            not np.isfinite(bed0).all() or np.any(suspended0 < 0.0) or
            np.any(bed0 < 0.0)):
        raise CoupledAMRZeroTimeTransferV98Error(
            "v98 parent class inventories differ")
    ng = int(suspended0.size)
    fluid0 = np.asarray(parent_fluid_integral)
    if fluid0.shape != (2,) or not np.isfinite(fluid0).all():
        raise CoupledAMRZeroTimeTransferV98Error(
            "v98 parent hT/hS inventory differs")
    if fluid0[1] < 0.0:
        raise CoupledAMRZeroTimeTransferV98Error(
            "v98 parent salinity inventory must be nonnegative")
    if depth0 <= dry and (np.any(momentum0 != 0.0) or
                          np.any(suspended0 != 0.0) or
                          np.any(fluid0 != 0.0)):
        raise CoupledAMRZeroTimeTransferV98Error(
            "v98 dry parent carries momentum, suspension, or fluid inventory")
    ids_before = tuple(boundary_source_ids_before)
    ids_after = tuple(boundary_source_ids_after)
    if (not ids_before or ids_before != ids_after or
            len(set(ids_before)) != len(ids_before) or
            any(type(value) is not int or value < 0 for value in ids_before)):
        raise CoupledAMRZeroTimeTransferV98Error(
            "v98 physical-boundary source IDs are not a one-to-one remap")
    memory_before = _sha(
        boundary_memory_before_sha256, "boundary memory before")
    memory_after = _sha(
        boundary_memory_after_sha256, "boundary memory after")
    if memory_before != memory_after:
        raise CoupledAMRZeroTimeTransferV98Error(
            "v98 zero-time regrid changed physical-boundary memory")
    if (type(accepted_updates) is not int or accepted_updates < 0 or
            accepted_updates > 4):
        raise CoupledAMRZeroTimeTransferV98Error(
            "v98 accepted update count is outside the bounded owner")
    time = _finite(time_s, "accepted time", nonnegative=True)

    child_depth = _stage_fill(
        depth0*area0, bed_elevation, area, dry)
    wet_child = child_depth > dry
    correction_index = int(np.flatnonzero(wet_child)[0]) \
        if np.any(wet_child) else 0
    if depth0 > dry:
        velocity = momentum0/depth0
        momentum_seed = velocity[:, None]*child_depth[None, :]
        suspended_seed = (suspended0/depth0)[:, None]*child_depth[None, :]
        fluid_seed = (fluid0/depth0)[:, None]*child_depth[None, :]
    else:
        momentum_seed = np.zeros((2, CHILD_COUNT))
        suspended_seed = np.zeros((ng, CHILD_COUNT))
        fluid_seed = np.zeros((2, CHILD_COUNT))
    child_momentum = _exact_integral_children(
        momentum0, area0, area, momentum_seed, nonnegative=False,
        correction_index=correction_index)
    child_suspended = _exact_integral_children(
        suspended0, area0, area, suspended_seed, nonnegative=True,
        correction_index=correction_index)
    bed_seed = np.broadcast_to(bed0[:, None], (ng, CHILD_COUNT)).copy()
    child_bed = _exact_integral_children(
        bed0, area0, area, bed_seed, nonnegative=True)
    child_fluid = _exact_integral_children(
        fluid0, area0, area, fluid_seed, nonnegative=False,
        correction_index=correction_index)
    dry_child = child_depth <= dry
    moved = np.array(child_suspended[:, dry_child], copy=True)
    if moved.size:
        suspended_mutable = np.array(child_suspended, copy=True)
        bed_mutable = np.array(child_bed, copy=True)
        suspended_mutable[:, dry_child] = 0.0
        bed_mutable[:, dry_child] += moved
        child_suspended = _array(
            suspended_mutable, (ng, CHILD_COUNT), "dry-clean child suspension",
            nonnegative=True)
        child_bed = _array(
            bed_mutable, (ng, CHILD_COUNT), "dry-clean child bed",
            nonnegative=True)
    restricted_depth = float(np.sum(child_depth*area, dtype=np.float64)/area0)
    restricted_momentum = _array(
        np.sum(child_momentum*area, axis=1, dtype=np.float64)/area0,
        (2,), "restricted momentum")
    restricted_suspended = _array(
        np.sum(child_suspended*area, axis=1, dtype=np.float64)/area0,
        (ng,), "restricted suspension", nonnegative=True)
    restricted_bed = _array(
        np.sum(child_bed*area, axis=1, dtype=np.float64)/area0,
        (ng,), "restricted bed", nonnegative=True)
    restricted_fluid = _array(
        np.sum(child_fluid*area, axis=1, dtype=np.float64)/area0,
        (2,), "restricted fluid")
    if (abs(restricted_depth-depth0) > float(_tolerance(depth0)) or
            np.any(np.abs(restricted_momentum-momentum0) >
                   _tolerance(restricted_momentum, momentum0)) or
            np.any(np.abs(
                (restricted_suspended+restricted_bed)-
                (suspended0+bed0)) > _tolerance(
                    restricted_suspended, restricted_bed,
                    suspended0, bed0)) or
            np.any(np.abs(restricted_fluid-fluid0) >
                   _tolerance(restricted_fluid, fluid0))):
        raise CoupledAMRZeroTimeTransferV98Error(
            "v98 zero-time refine/coarsen conservation differs")
    placeholder = CoupledAMRZeroTimeTransferV98(
        hierarchy, parent_sha, ng, area0, area, bed_elevation, dry, depth0,
        _array(momentum0, (2,), "parent momentum"),
        _array(suspended0, (ng,), "parent suspension", nonnegative=True),
        _array(bed0, (ng,), "parent bed", nonnegative=True),
        _array(fluid0, (2,), "parent fluid"), child_depth,
        child_momentum, child_suspended, child_bed, child_fluid,
        restricted_depth, restricted_momentum, restricted_suspended,
        restricted_bed, restricted_fluid, ids_before, memory_before,
        accepted_updates, time, 1, 1, 0, 0, "")
    result = CoupledAMRZeroTimeTransferV98(
        **{**placeholder.__dict__,
           "receipt_sha256": _hash(_payload(placeholder))})
    validate_zero_time_refine_coarsen_v98(result)
    return result


def validate_zero_time_refine_coarsen_v98(
    value: CoupledAMRZeroTimeTransferV98,
) -> None:
    if type(value) is not CoupledAMRZeroTimeTransferV98:
        raise CoupledAMRZeroTimeTransferV98Error(
            "v98 zero-time transfer type differs")
    ng = value.fraction_count
    arrays = (
        (value.child_area_m2, (4,), "child area", True),
        (value.child_bed_elevation_m, (4,), "child bed", False),
        (value.parent_momentum_m2_s, (2,), "parent momentum", False),
        (value.parent_suspended_kg_m2, (ng,),
         "parent suspension", True),
        (value.parent_bed_kg_m2, (ng,), "parent bed mass", True),
        (value.parent_fluid_integral, (2,), "parent fluid", False),
        (value.child_depth_m, (4,), "child depth", True),
        (value.child_momentum_m2_s, (2, 4), "child momentum", False),
        (value.child_suspended_kg_m2, (ng, 4), "child suspension", True),
        (value.child_bed_kg_m2, (ng, 4), "child bed mass", True),
        (value.child_fluid_integral, (2, 4), "child fluid", False),
        (value.restricted_momentum_m2_s, (2,), "restricted momentum", False),
        (value.restricted_suspended_kg_m2, (ng,),
         "restricted suspension", True),
        (value.restricted_bed_kg_m2, (ng,), "restricted bed", True),
        (value.restricted_fluid_integral, (2,), "restricted fluid", False),
    )
    for array, shape, name, nonnegative in arrays:
        checked = _array(array, shape, name, nonnegative=nonnegative)
        if (not np.array_equal(checked, array) or array.flags.writeable or
                not array.flags.c_contiguous):
            raise CoupledAMRZeroTimeTransferV98Error(
                f"v98 {name} changed during validation")
    if (ng < 1 or not math.isfinite(value.parent_area_m2) or
            value.parent_area_m2 <= 0.0 or
            not math.isfinite(value.dry_tolerance_m) or
            value.dry_tolerance_m <= 0.0 or
            not math.isfinite(value.parent_depth_m) or
            value.parent_depth_m < 0.0 or
            not math.isfinite(value.restricted_depth_m) or
            value.restricted_depth_m < 0.0 or
            not value.boundary_source_ids or
            len(set(value.boundary_source_ids)) !=
                len(value.boundary_source_ids) or
            value.refine_event_count != 1 or value.coarsen_event_count != 1 or
            value.clock_increment_count != 0 or
            value.constitutive_evaluation_count != 0 or
            value.receipt_sha256 != _hash(_payload(value))):
        raise CoupledAMRZeroTimeTransferV98Error(
            "v98 zero-time transfer identity or owner counts differ")
    if value.parent_fluid_integral[1] < 0.0:
        raise CoupledAMRZeroTimeTransferV98Error(
            "v98 parent salinity inventory differs")
    tolerance = _tolerance(
        value.parent_area_m2, value.child_area_m2,
        value.parent_depth_m, value.child_depth_m)
    if abs(np.sum(value.child_area_m2)-value.parent_area_m2) > \
            float(np.max(tolerance)):
        raise CoupledAMRZeroTimeTransferV98Error(
            "v98 child areas do not partition the parent")
    wet = value.child_depth_m > value.dry_tolerance_m
    dry = ~wet
    if np.any((value.child_depth_m > 0.0) & dry):
        raise CoupledAMRZeroTimeTransferV98Error(
            "v98 child depth lies inside the dry tolerance")
    if np.any(wet):
        stage = value.child_bed_elevation_m[wet]+value.child_depth_m[wet]
        if np.ptp(stage) > float(np.max(_tolerance(stage))):
            raise CoupledAMRZeroTimeTransferV98Error(
                "v98 wet children do not share one refinement stage")
    if (np.any(value.child_momentum_m2_s[:, dry] != 0.0) or
            np.any(value.child_suspended_kg_m2[:, dry] != 0.0) or
            np.any(value.child_fluid_integral[:, dry] != 0.0)):
        raise CoupledAMRZeroTimeTransferV98Error(
            "v98 dry children retain transported inventory")
    area = value.child_area_m2
    area0 = value.parent_area_m2
    depth_restricted = float(np.sum(value.child_depth_m*area)/area0)
    momentum_restricted = np.sum(
        value.child_momentum_m2_s*area, axis=1)/area0
    suspended_restricted = np.sum(
        value.child_suspended_kg_m2*area, axis=1)/area0
    bed_restricted = np.sum(
        value.child_bed_kg_m2*area, axis=1)/area0
    fluid_restricted = np.sum(
        value.child_fluid_integral*area, axis=1)/area0
    if (abs(depth_restricted-value.parent_depth_m) >
            float(np.max(_tolerance(depth_restricted,
                                    value.parent_depth_m))) or
            np.any(np.abs(momentum_restricted-value.parent_momentum_m2_s) >
                   _tolerance(momentum_restricted,
                              value.parent_momentum_m2_s)) or
            np.any(np.abs(
                suspended_restricted+bed_restricted-
                value.parent_suspended_kg_m2-value.parent_bed_kg_m2) >
                _tolerance(suspended_restricted, bed_restricted,
                           value.parent_suspended_kg_m2,
                           value.parent_bed_kg_m2)) or
            np.any(np.abs(fluid_restricted-value.parent_fluid_integral) >
                   _tolerance(fluid_restricted,
                              value.parent_fluid_integral)) or
            depth_restricted != value.restricted_depth_m or
            not np.array_equal(momentum_restricted,
                               value.restricted_momentum_m2_s) or
            not np.array_equal(suspended_restricted,
                               value.restricted_suspended_kg_m2) or
            not np.array_equal(bed_restricted,
                               value.restricted_bed_kg_m2) or
            not np.array_equal(fluid_restricted,
                               value.restricted_fluid_integral)):
        raise CoupledAMRZeroTimeTransferV98Error(
            "v98 independently recomputed transfer conservation differs")
    _sha(value.hierarchy_fingerprint, "hierarchy fingerprint")
    _sha(value.parent_state_sha256, "parent state")
    _sha(value.boundary_memory_sha256, "boundary memory")


__all__ = [
    "CHILD_COUNT", "CoupledAMRZeroTimeTransferV98",
    "CoupledAMRZeroTimeTransferV98Error", "SCHEMA", "SCOPE_ID",
    "resolve_zero_time_refine_coarsen_v98",
    "validate_zero_time_refine_coarsen_v98",
]
