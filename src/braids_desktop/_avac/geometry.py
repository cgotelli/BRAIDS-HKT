"""Validated project-owned release geometry; imported files stay read only."""
import json
import numpy as np
from osgeo import ogr


def display_mask_from_rings(rings, x, y):
    """Presentation-only mask; solver input geometry is never changed."""
    from shapely import Polygon, contains_xy
    xx, yy = np.meshgrid(x, y)
    mask = np.zeros(xx.shape, dtype=bool)
    for exterior, holes in rings:
        mask |= contains_xy(Polygon(exterior, holes), xx, yy)
    return mask


def export_release_polygons(path, rings, crs):
    """Stage current polygon geometry before replacing the chosen export."""
    from pathlib import Path
    import os
    import tempfile
    from .geo import spatial_reference
    path = Path(path).resolve()
    driver_name = {".geojson": "GeoJSON", ".json": "GeoJSON", ".gpkg": "GPKG", ".shp": "ESRI Shapefile"}.get(path.suffix.lower())
    if not driver_name:
        raise ValueError("Choose a GeoJSON, GeoPackage or shapefile filename.")
    payload = json.loads(release_geojson(rings, crs))
    if not rings:
        raise ValueError("There are no release polygons to export.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".avac-export-", dir=path.parent) as temporary:
        if not Path(temporary).resolve().is_relative_to(path.parent):
            raise ValueError("Export staging directory is outside the chosen folder.")
        staged = Path(temporary)/path.name
        dataset = ogr.GetDriverByName(driver_name).CreateDataSource(str(staged))
        options = ["ENCODING=UTF-8"] if driver_name == "ESRI Shapefile" else []
        layer = dataset.CreateLayer(path.stem, spatial_reference(crs), ogr.wkbPolygon, options=options)
        for item in payload["features"]:
            feature = ogr.Feature(layer.GetLayerDefn())
            feature.SetGeometry(ogr.CreateGeometryFromJson(json.dumps(item["geometry"])))
            if layer.CreateFeature(feature) != ogr.OGRERR_NONE:
                raise ValueError("Could not export release polygons.")
        feature = layer = dataset = None
        if driver_name == "GeoJSON" and not crs:
            # GeoJSON without a crs member otherwise defaults to WGS84 in OGR.
            staged.write_text(release_geojson(rings, crs), encoding="utf-8")
        for item in Path(temporary).iterdir():
            os.replace(item, path.parent/item.name)
        if path.suffix.lower() == ".shp":
            for suffix in (".qix", ".sbn", ".sbx", ".qpj"):
                stale = path.with_suffix(suffix)
                if stale.is_file():
                    stale.unlink()
    return path


def release_geojson(rings, crs):
    features = []
    for exterior, holes in rings:
        contours = []
        for ring in (exterior, *holes):
            coords = np.asarray(ring, dtype=float)
            if coords.ndim != 2 or coords.shape[1] != 2 or len(coords) < 3 or not np.all(np.isfinite(coords)):
                raise ValueError("A release polygon needs at least three finite vertices.")
            if not np.array_equal(coords[0], coords[-1]):
                coords = np.vstack((coords, coords[0]))
            if len(np.unique(coords[:-1], axis=0)) < 3:
                raise ValueError("A release polygon needs three distinct vertices.")
            contours.append(coords.tolist())
        geometry = dict(type="Polygon", coordinates=contours)
        polygon = ogr.CreateGeometryFromJson(json.dumps(geometry))
        if polygon is None or not polygon.IsValid() or polygon.GetArea() <= 0:
            raise ValueError("Invalid polygon. Avoid crossing edges and keep holes inside their exterior.")
        features.append(dict(type="Feature", properties={}, geometry=geometry))
    return json.dumps(dict(type="FeatureCollection", name="AVAC releases",
                           crs=dict(type="name", properties=dict(name=crs)) if crs else None, features=features),
                      indent=2, allow_nan=False) + "\n"
