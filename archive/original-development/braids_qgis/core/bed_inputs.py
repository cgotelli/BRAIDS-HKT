"""Spatial erodible-bed inputs and deterministic whole-cell mapping."""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
from osgeo import ogr, osr

from braids_core.boundaries.cut_cells import CutCellMesh, cut_cell_polygon


ERODIBLE_LAYER_NAME = "braids_erodible_zones"


@dataclass(frozen=True)
class ErodibleZone:
    zone_id: str
    thickness_m: float
    geometry_wkb: bytes

    def geometry(self) -> ogr.Geometry:
        geometry = ogr.CreateGeometryFromWkb(self.geometry_wkb)
        if geometry is None:
            raise ValueError(f"Cannot restore erodible zone {self.zone_id!r}")
        return geometry


def create_erodible_dataset(path: str | Path, crs_wkt: str) -> Path:
    """Create an empty editable polygon GeoPackage for erodible thickness."""
    path = Path(path).expanduser().resolve()
    if path.exists():
        raise ValueError(f"Erodible-zone dataset already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset = ogr.GetDriverByName("GPKG").CreateDataSource(str(path))
    if dataset is None:
        raise ValueError(f"Could not create erodible-zone dataset: {path}")
    spatial = osr.SpatialReference(); spatial.ImportFromWkt(crs_wkt)
    layer = dataset.CreateLayer(ERODIBLE_LAYER_NAME, spatial, ogr.wkbPolygon)
    identifier = ogr.FieldDefn("zone_id", ogr.OFTString); identifier.SetWidth(64)
    layer.CreateField(identifier)
    thickness = ogr.FieldDefn("thickness_m", ogr.OFTReal); thickness.SetWidth(20); thickness.SetPrecision(8)
    layer.CreateField(thickness)
    dataset = None
    return path


def load_erodible_zones(path: str | Path, crs_wkt: str,
                         layer_name: str = ERODIBLE_LAYER_NAME) -> tuple[ErodibleZone, ...]:
    """Read and validate saved erodible-thickness polygons."""
    path = Path(path).expanduser().resolve()
    dataset = ogr.Open(str(path), 0)
    if dataset is None:
        raise ValueError(f"Cannot open erodible-zone dataset: {path}")
    layer = dataset.GetLayerByName(layer_name)
    if layer is None:
        raise ValueError(f"Erodible layer {layer_name!r} is missing")
    fields = {layer.GetLayerDefn().GetFieldDefn(index).GetName()
              for index in range(layer.GetLayerDefn().GetFieldCount())}
    if not {"zone_id", "thickness_m"} <= fields:
        raise ValueError("Erodible-zone layer has lost required zone_id/thickness_m fields")
    expected = osr.SpatialReference(); expected.ImportFromWkt(crs_wkt)
    spatial = layer.GetSpatialRef()
    if spatial is None or not spatial.IsSame(expected):
        raise ValueError("Erodible zones and domain must use the same CRS")
    zones = []
    identifiers = set()
    for feature in layer:
        zone_id = str(feature["zone_id"] or "").strip()
        if not zone_id:
            raise ValueError("Every erodible polygon requires a zone_id")
        if zone_id in identifiers:
            raise ValueError(f"Duplicate erodible zone_id {zone_id!r}")
        identifiers.add(zone_id)
        if not feature.IsFieldSetAndNotNull(feature.GetFieldIndex("thickness_m")):
            raise ValueError(f"Erodible zone {zone_id!r} has no thickness")
        value = feature["thickness_m"]
        if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"Erodible zone {zone_id!r} needs a nonnegative finite thickness")
        geometry = feature.GetGeometryRef()
        if geometry is None or geometry.IsEmpty():
            raise ValueError(f"Erodible zone {zone_id!r} has empty geometry")
        kind = ogr.GT_Flatten(geometry.GetGeometryType())
        if kind not in (ogr.wkbPolygon, ogr.wkbMultiPolygon) or not geometry.IsValid():
            raise ValueError(f"Erodible zone {zone_id!r} must be a valid polygon or multipolygon")
        zones.append(ErodibleZone(zone_id, float(value), bytes(geometry.ExportToWkb())))
    dataset = None
    return tuple(zones)


def _cell_geometry(vertices) -> ogr.Geometry:
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for x, y in vertices:
        ring.AddPoint_2D(float(x), float(y))
    polygon = ogr.Geometry(ogr.wkbPolygon); polygon.AddGeometry(ring)
    return polygon


def map_erodible_to_cells(mesh: CutCellMesh, default_thickness_m: float,
                           zones: tuple[ErodibleZone, ...] | list[ErodibleZone]) -> tuple[np.ndarray, dict]:
    """Assign a thickness to a whole retained cell on positive-area contact."""
    if not isinstance(mesh, CutCellMesh):
        raise ValueError("CutCellMesh is required")
    if isinstance(default_thickness_m, bool) or not math.isfinite(float(default_thickness_m)) or float(default_thickness_m) < 0:
        raise ValueError("Default erodible thickness must be nonnegative and finite")
    values = np.full(len(mesh.cells), float(default_thickness_m))
    assigned_by: list[list[str]] = [[] for _ in mesh.cells]
    cell_shapes = [cut_cell_polygon(cell) for cell in mesh.cells]
    cell_envelopes = [shape.GetEnvelope() for shape in cell_shapes]
    zone_counts = {}
    for zone in zones:
        if not isinstance(zone, ErodibleZone):
            raise ValueError("ErodibleZone entries are required")
        geometry = zone.geometry(); xmin, xmax, ymin, ymax = geometry.GetEnvelope()
        touched = []
        for cell_id, (cell_shape, envelope) in enumerate(zip(cell_shapes, cell_envelopes)):
            cxmin, cxmax, cymin, cymax = envelope
            if cxmax < xmin or cxmin > xmax or cymax < ymin or cymin > ymax:
                continue
            intersection = cell_shape.Intersection(geometry)
            if intersection is None or intersection.IsEmpty() or intersection.GetArea() <= 0.:
                continue
            if assigned_by[cell_id] and not math.isclose(values[cell_id], zone.thickness_m,
                                                          rel_tol=0., abs_tol=1e-15):
                names = ", ".join([*assigned_by[cell_id], zone.zone_id])
                raise ValueError(
                    f"Erodible zones {names} assign different values to cell {cell_id}; "
                    "move the zone boundary to a cell edge or use the same value")
            values[cell_id] = zone.thickness_m
            assigned_by[cell_id].append(zone.zone_id); touched.append(cell_id)
        if not touched:
            raise ValueError(f"Erodible zone {zone.zone_id!r} has no positive-area contact with a retained solver cell")
        zone_counts[zone.zone_id] = len(touched)
    overridden = sum(bool(names) for names in assigned_by)
    return values, {
        "mode": "polygon thickness zones",
        "rule": "whole retained cell on positive-area polygon intersection",
        "overlap_policy": "different values on one cell rejected",
        "default_thickness_m": float(default_thickness_m),
        "minimum_thickness_m": float(values.min()) if len(values) else float(default_thickness_m),
        "maximum_thickness_m": float(values.max()) if len(values) else float(default_thickness_m),
        "zone_count": len(zones),
        "overridden_cell_count": overridden,
        "default_cell_count": len(values) - overridden,
        "zone_touched_cell_counts": zone_counts,
    }
