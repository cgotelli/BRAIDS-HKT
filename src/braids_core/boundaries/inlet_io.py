"""Export the explicitly limited fixed-grid native BRAIDS inlet format v1.

Numeric solver IDs are assigned deterministically from sorted string IDs. A
sidecar preserves those strings, whole-face keys, geometry and target diagnostics.
Profiles must describe one convex basin. Their active interval is clipped at
the Section's declared maximum stage; exterior dry aprons are not hydraulic
walls. The native table has no per-face roughness or arbitrary oblique mapping.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from .inlet import (
    Face, InletError, InletSpec, _face_map, _number, build_inlet_target,
    register_face_ownership,
)
from .section import Section


FORMAT_VERSION = 1
MODE_CODES = {"normal": 1, "critical": 2, "depth": 3, "stage": 4}
ALLOCATION_CODES = {"uniform": 1, "conveyance": 2}
ENDPOINT_CODES = {"hold": 1, "zero": 2, "error": 3}
POLICY_CODES = {"error": 0, "target": 1}
BACKFLOW_CODES = {"error": 0, "allow": 1}
SIDE_NORMALS = {1: (1., 0.), 2: (-1., 0.), 3: (0., 1.), 4: (0., -1.)}


@dataclass(frozen=True)
class NativeGrid:
    nx: int
    ny: int
    xlower_m: float
    ylower_m: float
    dx_m: float
    dy_m: float

    def __post_init__(self) -> None:
        for name in ("nx", "ny"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise InletError("Native nx and ny must be positive integers")
        for name in ("xlower_m", "ylower_m", "dx_m", "dy_m"):
            object.__setattr__(self, name, _number(getattr(self, name), name, positive=name in ("dx_m", "dy_m")))


@dataclass(frozen=True)
class NativeFace:
    """A whole Face and its zero-based index on one exterior grid side.

    Sides are west/east/south/north = 1/2/3/4. The index increases with y on
    west/east, and with x on south/north, independently of inward-normal sign.
    """

    face: Face
    side: int
    global_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.face, Face):
            raise InletError("NativeFace requires a Face record")
        if isinstance(self.side, bool) or not isinstance(self.side, int) or self.side not in SIDE_NORMALS:
            raise InletError("Native side must be west/east/south/north = 1/2/3/4")
        if isinstance(self.global_index, bool) or not isinstance(self.global_index, int) or self.global_index < 0:
            raise InletError("Native transverse face index must be a nonnegative integer")
        if self.face.inward_normal != SIDE_NORMALS[self.side]:
            raise InletError("Native face side and inward normal disagree")


def _policy(value: str | Mapping[str, str], bc_id: str) -> str:
    name = value.get(bc_id) if isinstance(value, Mapping) else value
    if name not in POLICY_CODES:
        raise InletError(f"Boundary {bc_id!r} needs an explicit error/target regime policy")
    return name


def _convex_profile(section: Section) -> tuple[list[float], list[float]]:
    geometry = section.geometry(section.max_stage_m)
    if len(geometry.wet_intervals) != 1 or geometry.excluded_wet_area_m2 > 1e-12:
        raise InletError("Native v1 requires one connected convex basin")
    interval = geometry.wet_intervals[0]
    a, b = interval.start_station_m, interval.end_station_m
    # Keep the actual clipped intersection endpoints. This preserves a lowered
    # declared maximum stage and trims only terrain that is dry throughout.
    stations = [a] + [float(s) for s in section.stations_m if a < s < b] + [b]
    bed = [section.max_stage_m] + [float(z) for s, z in zip(section.stations_m, section.bed_elevations_m)
                                  if a < s < b] + [section.max_stage_m]
    slopes = np.diff(bed) / np.diff(stations)
    for left, right in zip(slopes, slopes[1:]):
        if left - right > 1e-10 * max(1., abs(left), abs(right)):
            raise InletError("Native v1 rejects nonconvex or compound active profiles")
    if min(bed) != min(section.bed_elevations_m):
        raise InletError("Native v1 requires the section's global lowest basin")
    return stations, bed


def write_native_inlets(path: str | Path, grid: NativeGrid, specs: Iterable[InletSpec],
                         sections: Mapping[str, Section],
                         native_faces_by_id: Mapping[str, Iterable[NativeFace]], *,
                         dry_policy: str | Mapping[str, str] = "error",
                         supercritical_policy: str | Mapping[str, str] = "error",
                         backflow_policy: str | Mapping[str, str] = "error") -> dict:
    """Write numeric v1 data and ``<path>.json``; return the sidecar metadata.

    This exporter requires gravity 9.81 m/s²; the surrounding native GeoClaw
    configuration must use that same value. NativeFace indices/lengths must
    match the supplied fixed grid. Native beds may differ from the continuous
    profile (e.g. cell-average topography), but must form one unimodal basin and
    lie at or above profile midpoint elevations within 1e-9 m. Thus positive
    native depth implies a connected wet midpoint on the convex profile. This
    preserves the explicit continuous/discrete area distinction.

    Policies govern otherwise unsupported dry/supercritical incoming states.
    A target policy authorizes use of the requested state there. Wet subcritical
    stage remains subject to native outgoing-characteristic compatibility.
    Local reverse flow is rejected by default. Explicit ``backflow_policy=allow``
    permits signed subcritical face discharges while retaining the prescribed
    positive section total and outgoing characteristic information.
    """
    if not isinstance(grid, NativeGrid):
        raise InletError("An explicit NativeGrid is required")
    requests = tuple(specs)
    if not requests or any(not isinstance(spec, InletSpec) for spec in requests):
        raise InletError("Native export requires nonempty InletSpec requests")
    by_id = {spec.bc_id: spec for spec in requests}
    if len(by_id) != len(requests):
        raise InletError("Native inlet IDs must be unique")
    ids = sorted(by_id)
    if set(native_faces_by_id) != set(ids) or any(bc_id not in sections for bc_id in ids):
        raise InletError("Every native inlet needs its Section and exactly one native face map")
    numeric_ids = {bc_id: index + 1 for index, bc_id in enumerate(ids)}
    maps, profiles, diagnostics, policies = {}, {}, {}, {}
    physical_owners = {}
    for bc_id in ids:
        spec, section = by_id[bc_id], sections[bc_id]
        if not isinstance(section, Section) or section.bc_id != bc_id:
            raise InletError("Native request and Section IDs must agree")
        if spec.gravity_m_s2 != 9.81:
            raise InletError("Native v1 export requires gravity 9.81 m/s² in inlet and solver settings")
        records = tuple(native_faces_by_id[bc_id])
        if not records or any(not isinstance(record, NativeFace) for record in records):
            raise InletError("Native maps must contain nonempty NativeFace records")
        ordered_faces = _face_map(section, [record.face for record in records])
        records_by_key = {record.face.global_key: record for record in records}
        ordered = tuple(records_by_key[face.global_key] for face in ordered_faces)
        sides = {record.side for record in ordered}
        if len(sides) != 1:
            raise InletError("A native inlet must occupy one straight exterior side")
        side = ordered[0].side
        spacing = grid.dy_m if side <= 2 else grid.dx_m
        count = grid.ny if side <= 2 else grid.nx
        indices = [record.global_index for record in ordered]
        if any(index >= count for index in indices):
            raise InletError("Native face index lies outside the supplied grid")
        if any(right != left + 1 for left, right in zip(indices, indices[1:])):
            raise InletError("Native face indices must increase contiguously with canonical stations")
        increasing = False
        for left, right in zip(ordered, ordered[1:]):
            difference = right.face.bed_elevation_m - left.face.bed_elevation_m
            if difference > 1e-9:
                increasing = True
            elif difference < -1e-9 and increasing:
                raise InletError("Native face beds must form a single unimodal basin")
        for record in ordered:
            face = record.face
            physical_key = record.side, record.global_index
            if physical_key in physical_owners:
                raise InletError(f"Duplicate native face ownership at side/index {physical_key}")
            physical_owners[physical_key] = bc_id
            if not math.isclose(face.length_m, spacing, rel_tol=1e-12, abs_tol=1e-12):
                raise InletError("Whole native face length does not match grid spacing; cut faces are unsupported")
            if face.manning_n is not None and face.manning_n != spec.manning_n:
                raise InletError("Native v1 cannot serialize differing per-face Manning coefficients")
            midpoint = face.station_start_m + (face.station_end_m - face.station_start_m) / 2
            expected_bed = float(np.interp(midpoint, section.stations_m, section.bed_elevations_m))
            if face.bed_elevation_m < expected_bed - 1e-9:
                raise InletError("Native bed lies below the convex profile midpoint; an explicit activation mapping is required")
        maps[bc_id] = ordered
        profiles[bc_id] = _convex_profile(section)
        backflow = backflow_policy.get(bc_id) if isinstance(backflow_policy, Mapping) else backflow_policy
        if backflow not in BACKFLOW_CODES:
            raise InletError(f"Boundary {bc_id!r} needs an explicit error/allow backflow policy")
        policies[bc_id] = (_policy(dry_policy, bc_id), _policy(supercritical_policy, bc_id), backflow)
        diagnostics[bc_id] = []
        for time in spec.hydrograph.times_s:
            target = build_inlet_target(spec, section, ordered_faces, time)
            diagnostics[bc_id].append({
                "time_s": time, "requested_discharge_m3_s": target.requested_discharge_m3_s,
                "target_stage_m": target.target_stage_m, "state_discharge_m3_s": target.state_discharge_m3_s,
                "continuous_area_m2": target.continuous_area_m2, "discrete_area_m2": target.discrete_area_m2,
                "continuous_to_discrete_area_ratio": target.continuous_to_discrete_area_ratio})
    register_face_ownership({bc_id: [record.face for record in records] for bc_id, records in maps.items()})

    def numbers(*values) -> str:
        return " ".join(str(value) if isinstance(value, int) else format(value, ".17g") for value in values)

    lines = [str(FORMAT_VERSION), numbers(grid.nx, grid.ny, grid.xlower_m, grid.ylower_m, grid.dx_m, grid.dy_m),
             numbers(len(ids), sum(len(records) for records in maps.values()))]
    for bc_id in ids:
        spec, section = by_id[bc_id], sections[bc_id]
        stations, bed = profiles[bc_id]
        dry, supercritical, backflow = policies[bc_id]
        lines.append(numbers(numeric_ids[bc_id], MODE_CODES[spec.mode], ALLOCATION_CODES[spec.allocation],
                             ENDPOINT_CODES[spec.hydrograph.endpoint_policy], POLICY_CODES[dry], POLICY_CODES[supercritical],
                             BACKFLOW_CODES[backflow],
                             len(stations), len(spec.hydrograph.times_s), spec.value_m if spec.value_m is not None else 0.,
                             section.datum_m, spec.manning_n, spec.energy_slope))
        lines.extend(numbers(station, elevation) for station, elevation in zip(stations, bed))
        lines.extend(numbers(time, discharge) for time, discharge in zip(spec.hydrograph.times_s, spec.hydrograph.discharges_m3_s))
    for bc_id in ids:
        lines.extend(numbers(numeric_ids[bc_id], record.side, record.global_index,
                             record.face.bed_elevation_m, record.face.length_m) for record in maps[bc_id])
    payload = "\n".join(lines) + "\n"
    path = Path(path)
    sidecar = path.with_name(path.name + ".json")
    metadata = {"format_version": FORMAT_VERSION, "data_file": path.name,
                "data_sha256": hashlib.sha256(payload.encode()).hexdigest(),
                "grid": asdict(grid), "required_solver_gravity_m_s2": 9.81,
                "string_to_numeric_id": numeric_ids, "numeric_to_string_id": {str(value): key for key, value in numeric_ids.items()},
                "boundary_header_columns": ["numeric_id", "mode", "allocation", "endpoint", "dry_policy", "supercritical_policy", "backflow_policy",
                                            "profile_points", "hydrograph_knots", "value_m", "datum_m", "manning_n", "slope"],
                "mode_codes": MODE_CODES, "allocation_codes": ALLOCATION_CODES, "endpoint_codes": ENDPOINT_CODES,
                "regime_policy_codes": POLICY_CODES,
                "backflow_policy_codes": BACKFLOW_CODES,
                "limitations": ["Fixed-grid axis-aligned whole native faces; no cut faces or AMR.",
                                "One convex active basin per ID; unimodal native beds at or above continuous midpoint elevations.",
                                "Per-face roughness differing from the scalar inlet value is unsupported.",
                                "Requested target allocation is distinct from wet subcritical characteristic compatibility.",
                                "State-integrated Q is not accepted numerical boundary flux."],
                "boundaries": {bc_id: {"spec": asdict(by_id[bc_id]), "datum_m": sections[bc_id].datum_m,
                                        "max_stage_m": sections[bc_id].max_stage_m,
                                        "dry_policy": policies[bc_id][0], "supercritical_policy": policies[bc_id][1],
                                        "backflow_policy": policies[bc_id][2],
                                        "exported_profile": {"stations_m": profiles[bc_id][0], "bed_elevations_m": profiles[bc_id][1]},
                                        "native_faces": [asdict(record) for record in maps[bc_id]],
                                        "knot_target_diagnostics": diagnostics[bc_id]}
                               for bc_id in ids}}
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded_metadata = json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.write_text(payload)
    sidecar.write_text(encoded_metadata)
    return json.loads(encoded_metadata)
