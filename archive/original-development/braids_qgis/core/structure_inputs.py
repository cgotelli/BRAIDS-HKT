"""Fixed polygon obstacles represented by reflective internal cut-cell walls."""
from __future__ import annotations

from pathlib import Path

from osgeo import ogr, osr

from braids_core.boundaries.cut_cells import SolidObstacle


OBSTACLE_LAYER_NAME = "braids_solid_obstacles"


def create_obstacle_dataset(path: str | Path, crs_wkt: str) -> Path:
    """Create an editable GeoPackage layer for solid internal objects."""
    path = Path(path).expanduser().resolve()
    if path.exists():
        raise ValueError(f"Solid-obstacle dataset already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset = ogr.GetDriverByName("GPKG").CreateDataSource(str(path))
    if dataset is None:
        raise ValueError(f"Could not create solid-obstacle dataset: {path}")
    spatial = osr.SpatialReference();spatial.ImportFromWkt(crs_wkt)
    layer = dataset.CreateLayer(OBSTACLE_LAYER_NAME, spatial, ogr.wkbPolygon)
    identifier = ogr.FieldDefn("obstacle_id", ogr.OFTString);identifier.SetWidth(64)
    name = ogr.FieldDefn("name", ogr.OFTString);name.SetWidth(128)
    if layer is None or layer.CreateField(identifier) != ogr.OGRERR_NONE or \
            layer.CreateField(name) != ogr.OGRERR_NONE:
        dataset=None
        raise ValueError("Could not create solid-obstacle fields")
    dataset=None
    return path


def _polygons(geometry):
    kind=ogr.GT_Flatten(geometry.GetGeometryType())
    if kind==ogr.wkbPolygon:
        yield geometry
    elif kind==ogr.wkbMultiPolygon:
        for index in range(geometry.GetGeometryCount()):
            yield geometry.GetGeometryRef(index)


def load_solid_obstacles(path: str | Path, crs_wkt: str,
                         layer_name: str | None = None) -> tuple[SolidObstacle, ...]:
    """Load managed or ordinary polygon layers as fixed solid objects."""
    path=Path(path).expanduser().resolve();dataset=ogr.Open(str(path),0)
    if dataset is None:raise ValueError(f"Cannot open solid-obstacle dataset: {path}")
    layer=(dataset.GetLayerByName(layer_name) if layer_name else dataset.GetLayer(0))
    if layer is None:raise ValueError(f"Solid-obstacle layer {layer_name!r} is missing")
    expected=osr.SpatialReference();expected.ImportFromWkt(crs_wkt);spatial=layer.GetSpatialRef()
    if spatial is None or not spatial.IsSame(expected):
        raise ValueError("Solid obstacles and calculation domain must use the same CRS")
    definition=layer.GetLayerDefn();fields={definition.GetFieldDefn(i).GetName()
                                            for i in range(definition.GetFieldCount())}
    id_field=next((name for name in ("obstacle_id","structure_id","name") if name in fields),None)
    obstacles=[];identifiers=set()
    for feature in layer:
        geometry=feature.GetGeometryRef()
        if geometry is None or geometry.IsEmpty() or not geometry.IsValid() or \
                ogr.GT_Flatten(geometry.GetGeometryType()) not in (ogr.wkbPolygon,ogr.wkbMultiPolygon):
            raise ValueError(f"Solid-obstacle feature {feature.GetFID()} must be a valid polygon or multipolygon")
        base=str(feature[id_field] or "").strip() if id_field else ""
        base=base or f"obstacle_{feature.GetFID()}"
        parts=list(_polygons(geometry))
        for part_index,polygon in enumerate(parts,1):
            if polygon.GetGeometryCount()!=1:
                raise ValueError(f"Solid obstacle {base!r} must not contain holes")
            identifier=base if len(parts)==1 else f"{base}_{part_index}"
            if identifier in identifiers:raise ValueError(f"Duplicate solid-obstacle ID {identifier!r}")
            identifiers.add(identifier)
            ring=polygon.GetGeometryRef(0)
            vertices=tuple((ring.GetX(i),ring.GetY(i)) for i in range(ring.GetPointCount()))
            obstacles.append(SolidObstacle(identifier,vertices))
    dataset=None
    if not obstacles:raise ValueError("Solid-obstacle layer contains no polygons")
    return tuple(obstacles)
