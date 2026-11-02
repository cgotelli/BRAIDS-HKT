"""Polygon-domain geometry and explicit boundary assignments, without a solver map.

Every unassigned perimeter segment is a wall. IDs derive from exact geometry,
never feature order or polygon vertex numbering; changing an edge invalidates
its old ID. This module does not rasterize a domain or create native cut faces.
"""
from __future__ import annotations

from dataclasses import dataclass, field, InitVar
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

from osgeo import gdal, ogr, osr

from .gis import _projected_metres


def _identifier(value, label):
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be a nonempty string without surrounding whitespace")
    return value


def _digest(values):
    return hashlib.sha256(json.dumps(values, separators=(",", ":"), allow_nan=False).encode()).hexdigest()[:24]


def _polygon(vertices):
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for x, y in vertices:
        ring.AddPoint_2D(x, y)
    polygon = ogr.Geometry(ogr.wkbPolygon)
    polygon.AddGeometry(ring)
    return polygon


def _canonical_ring(vertices):
    points = []
    try:
        for vertex in vertices:
            if len(vertex) != 2:
                raise ValueError
            point = tuple(float(value) for value in vertex)
            if not all(math.isfinite(value) for value in point):
                raise ValueError
            points.append(tuple(0. if value == 0 else value for value in point))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Domain vertices require finite 2D coordinates") from error
    for index, (first, second) in enumerate(zip(points, points[1:]), 1):
        if first == second:
            raise ValueError(f"Consecutive duplicate domain vertex at source index {index} creates a zero-length edge; remove it explicitly in a derived geometry and retain the source mapping")
    if len(points) > 1 and points[0] == points[-1]:
        points.pop()
    if len(points) < 3 or len(set(points)) != len(points):
        raise ValueError("Domain requires at least three distinct vertices without repeated nonclosing vertices")
    polygon = _polygon(points+[points[0]])
    if (not polygon.IsValid() or not polygon.IsSimple()
            or not math.isfinite(polygon.GetArea()) or polygon.GetArea() <= 0):
        raise ValueError("Domain must be a valid simple polygon with positive area")
    # Translate before summing to avoid cancellation at large projected origins.
    x0, y0 = points[0]
    signed_area = math.fsum((x-x0)*(v-y0)-(u-x0)*(y-y0)
                           for (x, y), (u, v) in zip(points, points[1:]+points[:1]))
    if signed_area < 0:
        points.reverse()
    first = min(range(len(points)), key=points.__getitem__)
    points = points[first:]+points[:first]
    return tuple(points+[points[0]])


@dataclass(frozen=True)
class DomainEdge:
    edge_id: str
    start_xy: tuple[float, float]
    end_xy: tuple[float, float]
    length_m: float
    inward_normal: tuple[float, float]


@dataclass(frozen=True)
class BoundaryAssignment:
    """One whole edge or interval measured along its canonical CCW direction."""
    edge_id: str
    bc_id: str
    kind: str
    start_m: float = 0.
    end_m: float | None = None

    def __post_init__(self):
        _identifier(self.edge_id, "edge_id")
        _identifier(self.bc_id, "bc_id")
        if self.kind not in ("wall", "inlet", "outlet"):
            raise ValueError("Boundary kind must be wall, inlet or outlet")
        for name in ("start_m", "end_m"):
            value = getattr(self, name)
            if value is None and name == "end_m":
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite metres")
            object.__setattr__(self, name, float(value))


@dataclass(frozen=True)
class DomainBoundarySegment:
    segment_id: str
    edge_id: str
    start_m: float
    end_m: float
    start_xy: tuple[float, float]
    end_xy: tuple[float, float]
    inward_normal: tuple[float, float]
    kind: str
    bc_id: str

    @property
    def length_m(self):
        return self.end_m-self.start_m


@dataclass(frozen=True)
class PolygonDomain:
    domain_id: str
    vertices_xy: Sequence[Sequence[float]]
    spatial_reference: InitVar[osr.SpatialReference | str]
    crs_wkt: str = field(init=False)
    edges: tuple[DomainEdge, ...] = field(init=False)
    area_m2: float = field(init=False)

    def __post_init__(self, spatial_reference):
        _identifier(self.domain_id, "domain_id")
        spatial = _projected_metres(spatial_reference, "Domain")
        vertices = _canonical_ring(self.vertices_xy)
        edges = []
        for start, end in zip(vertices, vertices[1:]):
            dx, dy = end[0]-start[0], end[1]-start[1]
            length = math.hypot(dx, dy)
            if not math.isfinite(length) or length <= 0:
                raise ValueError("Domain edge lengths must be finite and positive")
            edge_id = self.domain_id+":edge:"+_digest(sorted((start, end)))
            edges.append(DomainEdge(edge_id, start, end, length, (-dy/length, dx/length)))
        object.__setattr__(self, "vertices_xy", vertices)
        object.__setattr__(self, "edges", tuple(edges))
        object.__setattr__(self, "crs_wkt", spatial.ExportToWkt())
        object.__setattr__(self, "area_m2", float(_polygon(vertices).GetArea()))

    def boundary_segments(self, assignments: Sequence[BoundaryAssignment] = ()) -> tuple[DomainBoundarySegment, ...]:
        """Partition the entire perimeter; gaps are walls and overlaps are errors.

        The same bc_id may label several pieces of one logical boundary only
        when their kinds agree. No automatic snapping, outlet selection, or
        conversion of interior lines to domain boundaries is performed.
        """
        default_id = self.domain_id+":wall"
        kinds = {default_id: "wall"}
        by_edge = {edge.edge_id: [] for edge in self.edges}
        for assignment in assignments:
            if not isinstance(assignment, BoundaryAssignment):
                raise ValueError("Assignments must be BoundaryAssignment records")
            if assignment.edge_id not in by_edge:
                raise ValueError("Unknown/stale edge_id; reselect the changed domain edge explicitly")
            if assignment.bc_id in kinds and kinds[assignment.bc_id] != assignment.kind:
                raise ValueError("A bc_id cannot mix wall, inlet and outlet kinds")
            kinds[assignment.bc_id] = assignment.kind
            by_edge[assignment.edge_id].append(assignment)
        segments = []
        for edge in self.edges:
            intervals = []
            for assignment in by_edge[edge.edge_id]:
                start = assignment.start_m
                end = edge.length_m if assignment.end_m is None else assignment.end_m
                if not 0 <= start < end <= edge.length_m:
                    raise ValueError("Assigned interval must have positive length within its edge; no clipping or snapping is implicit")
                intervals.append((start, end, assignment.kind, assignment.bc_id))
            intervals.sort()
            if any(next_start < end for (_, end, _, _), (next_start, _, _, _) in zip(intervals, intervals[1:])):
                raise ValueError("Boundary assignments overlap on one edge")
            cuts = sorted({0., edge.length_m, *(x for start, end, _, _ in intervals for x in (start, end))})
            for start, end in zip(cuts, cuts[1:]):
                owners = [(kind, bc_id) for left, right, kind, bc_id in intervals if left <= start and end <= right]
                kind, bc_id = owners[0] if owners else ("wall", default_id)

                def coordinate(station):
                    fraction = station/edge.length_m
                    return tuple(a+fraction*(b-a) for a, b in zip(edge.start_xy, edge.end_xy))

                segments.append(DomainBoundarySegment(
                    edge.edge_id+":segment:"+_digest((start, end)), edge.edge_id, start, end,
                    coordinate(start), coordinate(end), edge.inward_normal, kind, bc_id))
        return tuple(segments)


def boundary_polyline(domain: PolygonDomain, assignments: Sequence[BoundaryAssignment],
                      bc_id: str, *, tolerance_m: float=1e-8):
    """Return one contiguous collinear boundary section and its inward normal."""
    if not isinstance(domain,PolygonDomain):raise ValueError("PolygonDomain is required")
    _identifier(bc_id,"bc_id")
    segments=[row for row in domain.boundary_segments(assignments)
              if row.bc_id==bc_id and row.kind in ("inlet","outlet")]
    if not segments:raise ValueError(f"Boundary ID {bc_id!r} has no selected polygon faces")
    normal=segments[0].inward_normal;tangent=(normal[1],-normal[0]);origin=segments[0].start_xy
    intervals=[]
    for segment in segments:
        if any(abs(a-b)>1e-10 for a,b in zip(segment.inward_normal,normal)):
            raise ValueError(f"Boundary ID {bc_id!r} is noncollinear and cannot define one cross section")
        points=(segment.start_xy,segment.end_xy)
        offset=[((point[0]-origin[0])*normal[0]+(point[1]-origin[1])*normal[1]) for point in points]
        if max(abs(value) for value in offset)>tolerance_m:
            raise ValueError(f"Boundary ID {bc_id!r} is noncollinear and cannot define one cross section")
        station=[(point[0]-origin[0])*tangent[0]+(point[1]-origin[1])*tangent[1] for point in points]
        order=sorted(range(2),key=station.__getitem__)
        intervals.append((station[order[0]],station[order[1]],points[order[0]],points[order[1]]))
    intervals.sort(key=lambda row:row[0])
    if any(abs(right[0]-left[1])>tolerance_m for left,right in zip(intervals,intervals[1:])):
        raise ValueError(f"Boundary ID {bc_id!r} has disconnected faces and cannot define one cross section")
    coordinates=[intervals[0][2],*[row[3] for row in intervals]]
    if coordinates[-1]<coordinates[0]:coordinates.reverse()
    return tuple(coordinates),normal


def load_polygon_domain(path: str | Path, *, domain_id: str | None = None,
                        layer_name: str | None = None, id_field: str = "domain_id",
                        dem_path: str | Path | None = None) -> PolygonDomain:
    """Read one 2D Polygon without holes; optionally check DEM CRS/footprint.

    An explicit domain_id overrides the string ID field. Feature numbers and
    numeric shapefile IDs are never silently promoted to persistent IDs.
    Raster checks are horizontal only: elevation units/NoData need the separate
    terrain/section validation before any hydraulic use.
    """
    try:
        dataset = gdal.OpenEx(str(path), gdal.OF_VECTOR | gdal.OF_READONLY)
    except RuntimeError as error:
        raise ValueError(f"Cannot open domain vector: {path}") from error
    if dataset is None:
        raise ValueError(f"Cannot open domain vector: {path}")
    if layer_name is None and dataset.GetLayerCount() != 1:
        raise ValueError("Select layer_name explicitly in a multilayer domain dataset")
    layer = dataset.GetLayerByName(layer_name) if layer_name else dataset.GetLayer(0)
    if layer is None or layer.GetFeatureCount() != 1:
        raise ValueError("Domain layer must contain exactly one polygon feature")
    feature = layer.GetNextFeature()
    geometry = feature.GetGeometryRef()
    if geometry is None or geometry.GetGeometryType() != ogr.wkbPolygon or geometry.GetGeometryCount() != 1:
        raise ValueError("Domain must be one 2D Polygon without holes; multipart/Z/M geometries are unsupported")
    if domain_id is None:
        index = feature.GetFieldIndex(id_field)
        if index < 0 or layer.GetLayerDefn().GetFieldDefn(index).GetType() != ogr.OFTString:
            raise ValueError("Supply an explicit string domain_id or a string domain_id field")
        domain_id = feature.GetField(index)
    ring = geometry.GetGeometryRef(0)
    domain = PolygonDomain(domain_id, [(ring.GetX(i), ring.GetY(i)) for i in range(ring.GetPointCount())],
                           layer.GetSpatialRef())
    if dem_path is not None:
        try:
            dem = gdal.OpenEx(str(dem_path), gdal.OF_RASTER | gdal.OF_READONLY)
        except RuntimeError as error:
            raise ValueError(f"Cannot open DEM: {dem_path}") from error
        if dem is None:
            raise ValueError(f"Cannot open DEM: {dem_path}")
        spatial = _projected_metres(dem.GetSpatialRef(), "DEM")
        if not spatial.IsSame(_projected_metres(domain.crs_wkt, "Domain")):
            raise ValueError("Domain and DEM CRS must match; reproject explicitly")
        gt = dem.GetGeoTransform(can_return_null=True)
        if gt is None or not all(math.isfinite(value) for value in gt) or gt[1]*gt[5]-gt[2]*gt[4] == 0:
            raise ValueError("DEM requires a finite invertible affine geotransform")
        corners = [(gt[0]+i*gt[1]+j*gt[2], gt[3]+i*gt[4]+j*gt[5]) for i, j in
                   ((0, 0), (dem.RasterXSize, 0), (dem.RasterXSize, dem.RasterYSize), (0, dem.RasterYSize), (0, 0))]
        if not _polygon(domain.vertices_xy).Difference(_polygon(corners)).IsEmpty():
            raise ValueError("Domain extends outside the DEM footprint")
    return domain
