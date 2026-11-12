"""Persistent editable polygon-edge layer used by the QGIS workflow."""
from __future__ import annotations

from pathlib import Path

from osgeo import ogr, osr

from braids_core.boundaries.domain import BoundaryAssignment, PolygonDomain, boundary_polyline

EDGE_LAYER_NAME = "braids_boundary_edges"


def reassign_boundary_faces(domain: PolygonDomain,
                            assignments: list[BoundaryAssignment] | tuple[BoundaryAssignment, ...],
                            selected_edge_ids, kind: str, bc_id: str = "") -> tuple[BoundaryAssignment, ...]:
    """Return a validated, replacement-style boundary assignment.

    Reusing an active boundary ID replaces all faces formerly owned by that ID.
    The candidate is validated before the editable GIS layer is touched, so a
    disconnected or noncollinear selection cannot leave persistent bad state.
    """
    if not isinstance(domain, PolygonDomain):
        raise ValueError("PolygonDomain is required")
    boundary_kind=str(kind).strip().lower();identifier=str(bc_id).strip()
    if boundary_kind not in {"inlet","outlet","wall"}:
        raise ValueError("Boundary type must be inlet, outlet or wall")
    if boundary_kind!="wall" and not identifier:
        raise ValueError("Inlets and outlets require a boundary ID")
    if boundary_kind=="wall":identifier=""
    selected=tuple(dict.fromkeys(str(value) for value in selected_edge_ids))
    if not selected:raise ValueError("Select one or more polygon faces")
    if boundary_kind=="inlet" and len(selected)!=1:
        raise ValueError("An inlet must use exactly one selected polygon face")
    edges={edge.edge_id:edge for edge in domain.edges}
    unknown=sorted(set(selected)-set(edges))
    if unknown:raise ValueError(f"Selected boundary face does not belong to the calculation domain: {unknown[0]}")

    # Selected faces are taken from any previous owner. Reusing a logical ID
    # also clears every old face carrying that ID, making Apply a replacement
    # rather than an accidental append operation.
    retained=[value for value in assignments if value.edge_id not in selected and
              (boundary_kind=="wall" or value.bc_id!=identifier)]
    if boundary_kind!="wall":
        retained.extend(BoundaryAssignment(edge_id,identifier,boundary_kind,0.,edges[edge_id].length_m)
                        for edge_id in selected)
    candidate=tuple(retained)
    domain.boundary_segments(candidate)
    tolerance=1e-7*max(1.,max(edge.length_m for edge in domain.edges))
    for active_id in sorted({value.bc_id for value in candidate
                             if value.kind in {"inlet","outlet"}}):
        boundary_polyline(domain,candidate,active_id,tolerance_m=tolerance)
    return candidate


def create_edge_dataset(domain: PolygonDomain, path: str | Path) -> Path:
    """Write one editable GeoPackage feature per canonical polygon edge."""
    path=Path(path).expanduser().resolve()
    if path.exists():raise ValueError(f"Boundary edge dataset already exists: {path}")
    path.parent.mkdir(parents=True,exist_ok=True)
    driver=ogr.GetDriverByName("GPKG");dataset=driver.CreateDataSource(str(path))
    if dataset is None:raise ValueError(f"Could not create boundary edge dataset: {path}")
    spatial=osr.SpatialReference();spatial.ImportFromWkt(domain.crs_wkt)
    layer=dataset.CreateLayer(EDGE_LAYER_NAME,spatial,ogr.wkbLineString)
    for name,kind,width in (("edge_id",ogr.OFTString,96),("bc_id",ogr.OFTString,64),("kind",ogr.OFTString,12)):
        field=ogr.FieldDefn(name,kind);field.SetWidth(width);layer.CreateField(field)
    for name in ("start_m","end_m","length_m"):
        layer.CreateField(ogr.FieldDefn(name,ogr.OFTReal))
    layer.CreateField(ogr.FieldDefn("domain_id",ogr.OFTString))
    for edge in domain.edges:
        feature=ogr.Feature(layer.GetLayerDefn());feature.SetField("edge_id",edge.edge_id)
        feature.SetField("bc_id","");feature.SetField("kind","wall")
        feature.SetField("start_m",0.);feature.SetField("end_m",edge.length_m);feature.SetField("length_m",edge.length_m)
        feature.SetField("domain_id",domain.domain_id)
        line=ogr.Geometry(ogr.wkbLineString);line.AddPoint_2D(*edge.start_xy);line.AddPoint_2D(*edge.end_xy)
        feature.SetGeometry(line)
        if layer.CreateFeature(feature)!=ogr.OGRERR_NONE:raise ValueError("Could not write polygon edge")
    dataset=None
    return path


def load_assignments(path: str | Path, domain: PolygonDomain,
                     layer_name: str = EDGE_LAYER_NAME) -> list[BoundaryAssignment]:
    dataset=ogr.Open(str(Path(path).expanduser().resolve()),0)
    if dataset is None:raise ValueError(f"Cannot open boundary edge dataset: {path}")
    layer=dataset.GetLayerByName(layer_name)
    if layer is None:raise ValueError(f"Boundary layer {layer_name!r} is missing")
    required={"edge_id","bc_id","kind","start_m","end_m","domain_id"}
    fields={layer.GetLayerDefn().GetFieldDefn(i).GetName() for i in range(layer.GetLayerDefn().GetFieldCount())}
    if not required<=fields:raise ValueError("Boundary edge layer has lost required BRAIDS fields")
    assignments=[]
    for feature in layer:
        if feature["domain_id"]!=domain.domain_id:raise ValueError("Boundary edge layer belongs to another domain")
        kind=str(feature["kind"] or "").strip().lower();bc_id=str(feature["bc_id"] or "").strip()
        if kind=="wall":
            if bc_id:raise ValueError("Wall edges must have an empty bc_id")
            continue
        if kind not in {"inlet","outlet"} or not bc_id:raise ValueError("Each active edge needs inlet/outlet kind and bc_id")
        assignments.append(BoundaryAssignment(str(feature["edge_id"]),bc_id,kind,float(feature["start_m"]),float(feature["end_m"])))
    # This also rejects stale IDs, overlap, mixed kinds, and invalid intervals.
    domain.boundary_segments(assignments)
    return assignments
