"""Editable Manning zones and their deterministic whole-cell mapping."""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
from osgeo import ogr, osr

from braids_core.boundaries.cut_cells import CutCellMesh, cut_cell_polygon


MANNING_LAYER_NAME = "braids_manning_zones"


@dataclass(frozen=True)
class ManningZone:
    zone_id: str
    manning_n: float
    geometry_wkb: bytes

    def geometry(self) -> ogr.Geometry:
        geometry = ogr.CreateGeometryFromWkb(self.geometry_wkb)
        if geometry is None:
            raise ValueError(f"Cannot restore Manning zone {self.zone_id!r}")
        return geometry


def create_manning_dataset(path: str | Path, crs_wkt: str) -> Path:
    """Create an empty editable polygon GeoPackage for spatial roughness."""
    path = Path(path).expanduser().resolve()
    if path.exists():
        raise ValueError(f"Manning-zone dataset already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    driver = ogr.GetDriverByName("GPKG")
    dataset = driver.CreateDataSource(str(path))
    if dataset is None:
        raise ValueError(f"Could not create Manning-zone dataset: {path}")
    spatial = osr.SpatialReference(); spatial.ImportFromWkt(crs_wkt)
    layer = dataset.CreateLayer(MANNING_LAYER_NAME, spatial, ogr.wkbPolygon)
    identifier = ogr.FieldDefn("zone_id", ogr.OFTString); identifier.SetWidth(64)
    layer.CreateField(identifier)
    coefficient = ogr.FieldDefn("manning_n", ogr.OFTReal); coefficient.SetWidth(16); coefficient.SetPrecision(8)
    layer.CreateField(coefficient)
    dataset = None
    return path


def load_manning_zones(path: str | Path, crs_wkt: str,
                       layer_name: str = MANNING_LAYER_NAME) -> tuple[ManningZone, ...]:
    """Read and validate saved roughness polygons without importing QGIS."""
    path = Path(path).expanduser().resolve()
    dataset = ogr.Open(str(path), 0)
    if dataset is None:
        raise ValueError(f"Cannot open Manning-zone dataset: {path}")
    layer = dataset.GetLayerByName(layer_name)
    if layer is None:
        raise ValueError(f"Manning layer {layer_name!r} is missing")
    fields = {layer.GetLayerDefn().GetFieldDefn(index).GetName()
              for index in range(layer.GetLayerDefn().GetFieldCount())}
    if not {"zone_id", "manning_n"} <= fields:
        raise ValueError("Manning-zone layer has lost required zone_id/manning_n fields")
    expected = osr.SpatialReference(); expected.ImportFromWkt(crs_wkt)
    spatial = layer.GetSpatialRef()
    if spatial is None or not spatial.IsSame(expected):
        raise ValueError("Manning zones and domain must use the same CRS")
    zones = []
    identifiers = set()
    for feature in layer:
        zone_id = str(feature["zone_id"] or "").strip()
        if not zone_id:
            raise ValueError("Every Manning polygon requires a zone_id")
        if zone_id in identifiers:
            raise ValueError(f"Duplicate Manning zone_id {zone_id!r}")
        identifiers.add(zone_id)
        if not feature.IsFieldSetAndNotNull(feature.GetFieldIndex("manning_n")):
            raise ValueError(f"Manning zone {zone_id!r} has no coefficient")
        value = feature["manning_n"]
        if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"Manning zone {zone_id!r} needs a nonnegative finite coefficient")
        geometry = feature.GetGeometryRef()
        if geometry is None or geometry.IsEmpty():
            raise ValueError(f"Manning zone {zone_id!r} has empty geometry")
        kind = ogr.GT_Flatten(geometry.GetGeometryType())
        if kind not in (ogr.wkbPolygon, ogr.wkbMultiPolygon) or not geometry.IsValid():
            raise ValueError(f"Manning zone {zone_id!r} must be a valid polygon or multipolygon")
        zones.append(ManningZone(zone_id, float(value), bytes(geometry.ExportToWkb())))
    dataset = None
    return tuple(zones)


def _cell_geometry(vertices) -> ogr.Geometry:
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for x, y in vertices:
        ring.AddPoint_2D(float(x), float(y))
    polygon = ogr.Geometry(ogr.wkbPolygon); polygon.AddGeometry(ring)
    return polygon


def boundary_manning(coordinates, default_manning_n: float,
                     zones: tuple[ManningZone, ...] | list[ManningZone]):
    """Resolve one automatic section coefficient from polygons on its edge."""
    default=float(default_manning_n)
    if not math.isfinite(default) or default<0:
        raise ValueError("Default Manning coefficient must be nonnegative and finite")
    line=ogr.Geometry(ogr.wkbLineString)
    for x,y in coordinates:line.AddPoint_2D(float(x),float(y))
    tolerance=max(1.e-9,1.e-10*line.Length())
    touching=[]
    for zone in zones:
        intersection=line.Intersection(zone.geometry())
        if intersection is not None and not intersection.IsEmpty() and intersection.Length()>tolerance:
            touching.append(zone)
    if not touching:
        return default,{"source":"default Manning","manning_n":default,"zone_ids":[]}
    value=max(zone.manning_n for zone in touching)
    selected=sorted(zone.zone_id for zone in touching if zone.manning_n==value)
    return value,{"source":"Manning zone on boundary edge","manning_n":value,
                  "zone_ids":selected,"touching_zone_ids":sorted(zone.zone_id for zone in touching)}


def map_manning_to_cells(mesh: CutCellMesh, default_manning_n: float,
                         zones: tuple[ManningZone, ...] | list[ManningZone]) -> tuple[np.ndarray, dict]:
    """Assign a polygon value to every whole cell with positive-area contact.

    Cells are never split and values are never area weighted. When snapped
    zones meet at a node or edge and both touch one whole cell, the cell takes
    the highest coefficient. Interior polygon overlap and disconnected zones
    occupying one cell are rejected.
    """
    if not isinstance(mesh, CutCellMesh):
        raise ValueError("CutCellMesh is required")
    if isinstance(default_manning_n, bool) or not math.isfinite(float(default_manning_n)) or float(default_manning_n) < 0:
        raise ValueError("Default Manning coefficient must be nonnegative and finite")
    zones=tuple(zones)
    if any(not isinstance(zone,ManningZone) for zone in zones):
        raise ValueError("ManningZone entries are required")
    identifiers=[zone.zone_id for zone in zones]
    if len(identifiers)!=len(set(identifiers)):
        raise ValueError("Manning zone IDs must be unique")
    if not zones:
        values=np.full(len(mesh.cells),float(default_manning_n))
        return values,{"rule":"whole retained cell on positive-area polygon intersection",
            "overlap_policy":"shared-node/edge cell takes highest value; geometric overlap or disconnected co-occupancy rejected",
            "default_manning_n":float(default_manning_n),"minimum_manning_n":float(default_manning_n),
            "maximum_manning_n":float(default_manning_n),"zone_count":0,"overridden_cell_count":0,
            "default_cell_count":len(values),"zone_touched_cell_counts":{},
            "shared_node_or_edge_cell_count":0,"shared_node_or_edge_cells":[]}
    geometries={zone.zone_id:zone.geometry() for zone in zones}
    shared_boundaries={}
    for left_index,left in enumerate(zones):
        for right in zones[left_index+1:]:
            intersection=geometries[left.zone_id].Intersection(geometries[right.zone_id])
            if intersection is not None and not intersection.IsEmpty():
                if intersection.GetArea()>0.:
                    raise ValueError(f"Manning zones {left.zone_id}, {right.zone_id} overlap away from shared nodes/edges")
                shared_boundaries[frozenset((left.zone_id,right.zone_id))]=intersection
    values = np.full(len(mesh.cells), float(default_manning_n))
    assigned_by: list[list[str]] = [[] for _ in mesh.cells]
    cell_shapes = [cut_cell_polygon(cell) for cell in mesh.cells]
    cell_envelopes = [shape.GetEnvelope() for shape in cell_shapes]  # xmin,xmax,ymin,ymax
    zone_counts = {}
    for zone in zones:
        geometry = geometries[zone.zone_id]; xmin, xmax, ymin, ymax = geometry.GetEnvelope()
        touched = []
        for cell_id, (cell_shape, envelope) in enumerate(zip(cell_shapes, cell_envelopes)):
            cxmin, cxmax, cymin, cymax = envelope
            if cxmax < xmin or cxmin > xmax or cymax < ymin or cymin > ymax:
                continue
            intersection = cell_shape.Intersection(geometry)
            if intersection is None or intersection.IsEmpty() or intersection.GetArea() <= 0.:
                continue
            if assigned_by[cell_id]:
                pairs=[shared_boundaries.get(frozenset((name,zone.zone_id))) for name in assigned_by[cell_id]]
                if not all(shared is not None and shared.Intersects(cell_shape) for shared in pairs):
                    names = ", ".join([*assigned_by[cell_id], zone.zone_id])
                    raise ValueError(f"Manning zones {names} occupy cell {cell_id} without shared snapped nodes/edges")
                values[cell_id] = max(values[cell_id],zone.manning_n)
            else:values[cell_id] = zone.manning_n
            assigned_by[cell_id].append(zone.zone_id); touched.append(cell_id)
        if not touched:
            raise ValueError(f"Manning zone {zone.zone_id!r} has no positive-area contact with a retained solver cell")
        zone_counts[zone.zone_id] = len(touched)
    overridden = sum(bool(names) for names in assigned_by)
    shared_cells=[index for index,names in enumerate(assigned_by) if len(names)>1]
    audit = {
        "rule": "whole retained cell on positive-area polygon intersection",
        "overlap_policy": "shared-node/edge cell takes highest value; geometric overlap or disconnected co-occupancy rejected",
        "default_manning_n": float(default_manning_n),
        "minimum_manning_n": float(values.min()) if len(values) else float(default_manning_n),
        "maximum_manning_n": float(values.max()) if len(values) else float(default_manning_n),
        "zone_count": len(zones),
        "overridden_cell_count": overridden,
        "default_cell_count": len(values) - overridden,
        "zone_touched_cell_counts": zone_counts,
        "shared_node_or_edge_cell_count":len(shared_cells),
        "shared_node_or_edge_cells":shared_cells,
    }
    return values, audit
