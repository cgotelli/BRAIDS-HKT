"""Explicit internal erodible-bank zones and whole-cell mapping."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from osgeo import ogr, osr

from braids_core.boundaries.cut_cells import CutCellMesh, cut_cell_polygon


BANK_LAYER_NAME = "braids_erodible_bank_zones"


@dataclass(frozen=True)
class BankZone:
    zone_id: str
    geometry_wkb: bytes

    def geometry(self) -> ogr.Geometry:
        geometry = ogr.CreateGeometryFromWkb(self.geometry_wkb)
        if geometry is None:
            raise ValueError(f"Cannot restore erodible-bank zone {self.zone_id!r}")
        return geometry


def create_bank_dataset(path: str | Path, crs_wkt: str) -> Path:
    """Create an editable polygon GeoPackage for internal bank donors."""
    path = Path(path).expanduser().resolve()
    if path.exists():
        raise ValueError(f"Erodible-bank dataset already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset = ogr.GetDriverByName("GPKG").CreateDataSource(str(path))
    if dataset is None:
        raise ValueError(f"Could not create erodible-bank dataset: {path}")
    spatial = osr.SpatialReference(); spatial.ImportFromWkt(crs_wkt)
    layer = dataset.CreateLayer(BANK_LAYER_NAME, spatial, ogr.wkbPolygon)
    identifier = ogr.FieldDefn("zone_id", ogr.OFTString); identifier.SetWidth(64)
    layer.CreateField(identifier)
    dataset = None
    return path


def load_bank_zones(path: str | Path, crs_wkt: str,
                    layer_name: str = BANK_LAYER_NAME) -> tuple[BankZone, ...]:
    """Read and validate explicitly erodible internal bank polygons."""
    path = Path(path).expanduser().resolve()
    dataset = ogr.Open(str(path), 0)
    if dataset is None:
        raise ValueError(f"Cannot open erodible-bank dataset: {path}")
    layer = dataset.GetLayerByName(layer_name)
    if layer is None:
        raise ValueError(f"Erodible-bank layer {layer_name!r} is missing")
    fields = {layer.GetLayerDefn().GetFieldDefn(index).GetName()
              for index in range(layer.GetLayerDefn().GetFieldCount())}
    if "zone_id" not in fields:
        raise ValueError("Erodible-bank layer has lost its required zone_id field")
    expected = osr.SpatialReference(); expected.ImportFromWkt(crs_wkt)
    spatial = layer.GetSpatialRef()
    if spatial is None or not spatial.IsSame(expected):
        raise ValueError("Erodible-bank zones and domain must use the same CRS")
    zones = []
    identifiers = set()
    for feature in layer:
        zone_id = str(feature["zone_id"] or "").strip()
        if not zone_id:
            raise ValueError("Every erodible-bank polygon requires a zone_id")
        if zone_id in identifiers:
            raise ValueError(f"Duplicate erodible-bank zone_id {zone_id!r}")
        identifiers.add(zone_id)
        geometry = feature.GetGeometryRef()
        if geometry is None or geometry.IsEmpty():
            raise ValueError(f"Erodible-bank zone {zone_id!r} has empty geometry")
        kind = ogr.GT_Flatten(geometry.GetGeometryType())
        if kind not in (ogr.wkbPolygon, ogr.wkbMultiPolygon) or not geometry.IsValid():
            raise ValueError(f"Erodible-bank zone {zone_id!r} must be a valid polygon or multipolygon")
        zones.append(BankZone(zone_id, bytes(geometry.ExportToWkb())))
    dataset = None
    return tuple(zones)


def _cell_geometry(vertices) -> ogr.Geometry:
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for x, y in vertices:
        ring.AddPoint_2D(float(x), float(y))
    polygon = ogr.Geometry(ogr.wkbPolygon); polygon.AddGeometry(ring)
    return polygon


def map_bank_zones_to_cells(mesh: CutCellMesh,
                            zones: tuple[BankZone, ...] | list[BankZone]) -> tuple[np.ndarray, dict]:
    """Mark a complete retained cell when a zone intersects it with area."""
    if not isinstance(mesh, CutCellMesh):
        raise ValueError("CutCellMesh is required")
    zones=tuple(zones)
    mask = np.zeros(len(mesh.cells), dtype=bool)
    if not zones:
        return mask,{"mode":"explicit internal erodible-bank zones",
            "rule":"whole retained cell on positive-area polygon intersection",
            "overlap_policy":"union; a cell is marked erodible once","zone_count":0,
            "erodible_cell_count":0,"zone_touched_cell_counts":{},"cell_zone_ids":{}}
    assigned_by: list[list[str]] = [[] for _ in mesh.cells]
    cell_shapes = [cut_cell_polygon(cell) for cell in mesh.cells]
    cell_envelopes = [shape.GetEnvelope() for shape in cell_shapes]
    counts = {}
    for zone in zones:
        if not isinstance(zone, BankZone):
            raise ValueError("BankZone entries are required")
        geometry = zone.geometry(); xmin, xmax, ymin, ymax = geometry.GetEnvelope()
        touched = []
        for cell_id, (cell_shape, envelope) in enumerate(zip(cell_shapes, cell_envelopes)):
            cxmin, cxmax, cymin, cymax = envelope
            if cxmax < xmin or cxmin > xmax or cymax < ymin or cymin > ymax:
                continue
            intersection = cell_shape.Intersection(geometry)
            if intersection is None or intersection.IsEmpty() or intersection.GetArea() <= 0.:
                continue
            mask[cell_id] = True
            assigned_by[cell_id].append(zone.zone_id); touched.append(cell_id)
        if not touched:
            raise ValueError(f"Erodible-bank zone {zone.zone_id!r} has no positive-area contact with a retained solver cell")
        counts[zone.zone_id] = len(touched)
    return mask, {
        "mode": "explicit internal erodible-bank zones",
        "rule": "whole retained cell on positive-area polygon intersection",
        "overlap_policy": "union; a cell is marked erodible once",
        "zone_count": len(zones),
        "erodible_cell_count": int(mask.sum()),
        "zone_touched_cell_counts": counts,
        "cell_zone_ids": {str(index): names for index, names in enumerate(assigned_by) if names},
    }
