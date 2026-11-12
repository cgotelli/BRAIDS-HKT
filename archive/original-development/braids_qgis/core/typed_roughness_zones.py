"""Typed spatial C/n/ks zones with deterministic raw-parameter identity.

This is the schema used by new projects.  The older ``roughness.py`` API is
kept unchanged for historical Manning projects.  A value in this module is
always the raw parameter selected by ``mode``; no equivalent Manning value is
ever manufactured.

Whole-cell mapping can make two geometrically disjoint polygons touch the same
coarse solver cell.  Numeric max/min is not a valid generic policy because
larger ``n``/``ks`` means more resistance while larger Chezy ``C`` means less.
Each zone therefore carries an explicit integer priority.  The highest
priority wins; equal-priority, different-value ambiguity is rejected.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
from osgeo import ogr, osr

from braids_core.boundaries.cut_cells import CutCellMesh, cut_cell_polygon


ROUGHNESS_LAYER_NAME = "braids_roughness_zones"
ROUGHNESS_MODES = {
    "chezy": {"parameter_name": "chezy_C", "units": "m^0.5/s"},
    "manning": {"parameter_name": "manning_n", "units": "s/m^(1/3)"},
    "white_colebrook": {"parameter_name": "nikuradse_ks", "units": "m"},
}
SCHEMA = 1


def _require_mode(mode: str) -> dict[str, str]:
    try:
        return ROUGHNESS_MODES[mode]
    except (KeyError, TypeError) as error:
        raise ValueError("Roughness mode must be chezy, manning, or white_colebrook") from error


def _valid_parameter(mode: str, value: object) -> float:
    _require_mode(mode)
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("Roughness parameter cannot be Boolean")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("Roughness parameter must be numeric") from error
    if not math.isfinite(result) or result < 0.0 or (mode != "manning" and result == 0.0):
        rule = "nonnegative" if mode == "manning" else "positive"
        raise ValueError(f"{mode} roughness parameter must be finite and {rule}")
    return result


@dataclass(frozen=True)
class TypedRoughnessZone:
    zone_id: str
    mode: str
    parameter: float
    priority: int
    geometry_wkb: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.zone_id, str) or not self.zone_id.strip():
            raise ValueError("Typed roughness zone_id is required")
        _valid_parameter(self.mode, self.parameter)
        if isinstance(self.priority, bool) or not isinstance(self.priority, int):
            raise ValueError("Typed roughness priority must be an integer")
        if not isinstance(self.geometry_wkb, bytes) or not self.geometry_wkb:
            raise ValueError("Typed roughness geometry WKB is required")

    def geometry(self) -> ogr.Geometry:
        geometry = ogr.CreateGeometryFromWkb(self.geometry_wkb)
        if geometry is None:
            raise ValueError(f"Cannot restore roughness zone {self.zone_id!r}")
        return geometry


def create_typed_roughness_dataset(path: str | Path, crs_wkt: str) -> Path:
    """Create a mode-explicit, priority-resolved editable GeoPackage layer."""
    path = Path(path).expanduser().resolve()
    if path.exists():
        raise ValueError(f"Roughness-zone dataset already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    driver = ogr.GetDriverByName("GPKG")
    dataset = driver.CreateDataSource(str(path))
    if dataset is None:
        raise ValueError(f"Could not create roughness-zone dataset: {path}")
    spatial = osr.SpatialReference()
    spatial.ImportFromWkt(crs_wkt)
    layer = dataset.CreateLayer(ROUGHNESS_LAYER_NAME, spatial, ogr.wkbPolygon)
    identifier = ogr.FieldDefn("zone_id", ogr.OFTString)
    identifier.SetWidth(64)
    mode = ogr.FieldDefn("mode", ogr.OFTString)
    mode.SetWidth(32)
    parameter = ogr.FieldDefn("parameter", ogr.OFTReal)
    parameter.SetWidth(24)
    parameter.SetPrecision(15)
    priority = ogr.FieldDefn("priority", ogr.OFTInteger64)
    for field in (identifier, mode, parameter, priority):
        if layer.CreateField(field) != ogr.OGRERR_NONE:
            dataset = None
            raise ValueError("Could not create typed roughness-zone fields")
    dataset = None
    return path


def load_typed_roughness_zones(
    path: str | Path,
    crs_wkt: str,
    selected_mode: str,
    layer_name: str = ROUGHNESS_LAYER_NAME,
) -> tuple[TypedRoughnessZone, ...]:
    """Read zones and reject mixed or stale physical parameter types."""
    _require_mode(selected_mode)
    path = Path(path).expanduser().resolve()
    dataset = ogr.Open(str(path), 0)
    if dataset is None:
        raise ValueError(f"Cannot open roughness-zone dataset: {path}")
    layer = dataset.GetLayerByName(layer_name)
    if layer is None:
        raise ValueError(f"Roughness layer {layer_name!r} is missing")
    definition = layer.GetLayerDefn()
    fields = {definition.GetFieldDefn(index).GetName()
              for index in range(definition.GetFieldCount())}
    required = {"zone_id", "mode", "parameter", "priority"}
    if not required <= fields:
        raise ValueError("Typed roughness layer has lost required fields")
    expected = osr.SpatialReference()
    expected.ImportFromWkt(crs_wkt)
    spatial = layer.GetSpatialRef()
    if spatial is None or not spatial.IsSame(expected):
        raise ValueError("Roughness zones and domain must use the same CRS")
    zones: list[TypedRoughnessZone] = []
    identifiers: set[str] = set()
    for feature in layer:
        zone_id = str(feature["zone_id"] or "").strip()
        if not zone_id or zone_id in identifiers:
            reason = "requires a zone_id" if not zone_id else f"duplicates zone_id {zone_id!r}"
            raise ValueError(f"Typed roughness feature {reason}")
        identifiers.add(zone_id)
        mode = str(feature["mode"] or "").strip()
        if mode != selected_mode:
            raise ValueError(
                f"Roughness zone {zone_id!r} stores {mode!r}, not selected mode {selected_mode!r}"
            )
        for name in ("parameter", "priority"):
            if not feature.IsFieldSetAndNotNull(feature.GetFieldIndex(name)):
                raise ValueError(f"Roughness zone {zone_id!r} has no {name}")
        parameter = _valid_parameter(mode, feature["parameter"])
        priority_value = feature["priority"]
        if isinstance(priority_value, bool) or int(priority_value) != priority_value:
            raise ValueError(f"Roughness zone {zone_id!r} priority must be an integer")
        geometry = feature.GetGeometryRef()
        if geometry is None or geometry.IsEmpty():
            raise ValueError(f"Roughness zone {zone_id!r} has empty geometry")
        kind = ogr.GT_Flatten(geometry.GetGeometryType())
        if kind not in (ogr.wkbPolygon, ogr.wkbMultiPolygon) or not geometry.IsValid():
            raise ValueError(f"Roughness zone {zone_id!r} must be a valid polygon or multipolygon")
        zones.append(TypedRoughnessZone(
            zone_id, mode, parameter, int(priority_value),
            bytes(geometry.ExportToWkb()),
        ))
    dataset = None
    return tuple(zones)


def _array_identity(values: np.ndarray) -> dict[str, object]:
    raw = np.ascontiguousarray(values, dtype=np.float64)
    return {
        "shape": [int(raw.size)],
        "dtype": raw.dtype.str,
        "sha256": hashlib.sha256(raw.tobytes(order="C")).hexdigest(),
    }


def _fingerprinted(document: dict[str, object]) -> dict[str, object]:
    preimage = json.dumps(document, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False).encode("ascii")
    return {**document, "fingerprint": hashlib.sha256(preimage).hexdigest()}


def _validate_zones(mode: str, zones: Iterable[TypedRoughnessZone]) -> tuple[TypedRoughnessZone, ...]:
    _require_mode(mode)
    result = tuple(zones)
    if any(not isinstance(zone, TypedRoughnessZone) for zone in result):
        raise ValueError("TypedRoughnessZone entries are required")
    identifiers = [zone.zone_id for zone in result]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Typed roughness zone IDs must be unique")
    if any(zone.mode != mode for zone in result):
        raise ValueError("Every spatial roughness zone must use the selected mode")
    return result


def _resolve_candidates(candidates: list[TypedRoughnessZone], context: str):
    if not candidates:
        return None, []
    maximum = max(zone.priority for zone in candidates)
    leaders = sorted((zone for zone in candidates if zone.priority == maximum),
                     key=lambda zone: zone.zone_id)
    distinct = {zone.parameter for zone in leaders}
    if len(distinct) != 1:
        names = ", ".join(zone.zone_id for zone in leaders)
        raise ValueError(
            f"Ambiguous equal-priority roughness zones {names} at {context}; "
            "assign distinct priorities or the same raw parameter"
        )
    return leaders[0], [zone.zone_id for zone in leaders]


def map_typed_roughness_to_cells(
    mesh: CutCellMesh,
    mode: str,
    default_parameter: float,
    zones: Iterable[TypedRoughnessZone],
) -> tuple[np.ndarray, dict[str, object]]:
    """Map raw C/n/ks values without mode-dependent numeric substitution."""
    if not isinstance(mesh, CutCellMesh):
        raise ValueError("CutCellMesh is required")
    specification = _require_mode(mode)
    default = _valid_parameter(mode, default_parameter)
    zones = _validate_zones(mode, zones)
    values = np.full(len(mesh.cells), default, dtype=np.float64)
    cell_shapes = [cut_cell_polygon(cell) for cell in mesh.cells]
    envelopes = [shape.GetEnvelope() for shape in cell_shapes]
    candidates: list[list[TypedRoughnessZone]] = [[] for _ in mesh.cells]
    touched_counts: dict[str, int] = {}
    for zone in zones:
        geometry = zone.geometry()
        xmin, xmax, ymin, ymax = geometry.GetEnvelope()
        touched = 0
        for cell_id, (cell, envelope) in enumerate(zip(cell_shapes, envelopes)):
            cxmin, cxmax, cymin, cymax = envelope
            if cxmax < xmin or cxmin > xmax or cymax < ymin or cymin > ymax:
                continue
            intersection = cell.Intersection(geometry)
            if intersection is None or intersection.IsEmpty() or intersection.GetArea() <= 0.0:
                continue
            candidates[cell_id].append(zone)
            touched += 1
        if touched == 0:
            raise ValueError(
                f"Roughness zone {zone.zone_id!r} has no positive-area contact with a retained solver cell"
            )
        touched_counts[zone.zone_id] = touched
    selected: list[list[str]] = [[] for _ in mesh.cells]
    competing: list[int] = []
    overridden = 0
    for cell_id, options in enumerate(candidates):
        winner, leaders = _resolve_candidates(options, f"cell {cell_id}")
        if winner is None:
            continue
        values[cell_id] = winner.parameter
        selected[cell_id] = leaders
        overridden += 1
        if len(options) > 1:
            competing.append(cell_id)
    zone_rows = [{
        "zone_id": zone.zone_id,
        "mode": zone.mode,
        "parameter": float(zone.parameter),
        "priority": int(zone.priority),
        "geometry_sha256": hashlib.sha256(zone.geometry_wkb).hexdigest(),
    } for zone in sorted(zones, key=lambda item: item.zone_id)]
    binding = _fingerprinted({
        "schema": SCHEMA,
        "mode": mode,
        "parameter_name": specification["parameter_name"],
        "parameter_units": specification["units"],
        "default_parameter": default,
        "array": _array_identity(values),
        "resolution": "highest integer priority; equal-priority unequal raw values reject",
        "zones": zone_rows,
    })
    audit: dict[str, object] = {
        "rule": "whole retained cell on positive-area polygon intersection",
        "overlap_policy": binding["resolution"],
        "mode": mode,
        "parameter_name": specification["parameter_name"],
        "parameter_units": specification["units"],
        "default_parameter": default,
        "minimum_parameter": float(values.min()) if len(values) else default,
        "maximum_parameter": float(values.max()) if len(values) else default,
        "zone_count": len(zones),
        "overridden_cell_count": overridden,
        "default_cell_count": len(values) - overridden,
        "zone_touched_cell_counts": touched_counts,
        "competing_zone_cell_count": len(competing),
        "competing_zone_cells": competing,
        "selected_zone_ids_by_cell": selected,
        "roughness_field_binding": binding,
        "legacy_manning_conversion": False,
    }
    return values, audit


def boundary_typed_roughness(
    coordinates,
    mode: str,
    default_parameter: float,
    zones: Iterable[TypedRoughnessZone],
) -> tuple[float, dict[str, object]]:
    """Resolve a boundary raw parameter with the same explicit priority rule."""
    specification = _require_mode(mode)
    default = _valid_parameter(mode, default_parameter)
    zones = _validate_zones(mode, zones)
    line = ogr.Geometry(ogr.wkbLineString)
    for x, y in coordinates:
        line.AddPoint_2D(float(x), float(y))
    tolerance = max(1.0e-9, 1.0e-10 * line.Length())
    touching = []
    for zone in zones:
        intersection = line.Intersection(zone.geometry())
        if (intersection is not None and not intersection.IsEmpty() and
                intersection.Length() > tolerance):
            touching.append(zone)
    winner, leaders = _resolve_candidates(touching, "boundary edge")
    value = default if winner is None else winner.parameter
    return value, {
        "source": "default typed roughness parameter" if winner is None else "typed roughness zone on boundary edge",
        "mode": mode,
        "parameter_name": specification["parameter_name"],
        "parameter_units": specification["units"],
        "parameter": float(value),
        "zone_ids": leaders,
        "touching_zone_ids": sorted(zone.zone_id for zone in touching),
        "resolution": "highest integer priority; equal-priority unequal raw values reject",
        "legacy_manning_conversion": False,
    }


__all__ = [
    "ROUGHNESS_LAYER_NAME", "ROUGHNESS_MODES", "SCHEMA",
    "TypedRoughnessZone", "create_typed_roughness_dataset",
    "load_typed_roughness_zones", "map_typed_roughness_to_cells",
    "boundary_typed_roughness",
]
