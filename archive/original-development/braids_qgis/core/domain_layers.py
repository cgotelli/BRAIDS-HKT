"""Creation of the editable QGIS calculation-domain dataset."""
from __future__ import annotations

from pathlib import Path

from osgeo import ogr, osr


DOMAIN_LAYER_NAME = "braids_calculation_domain"


def create_domain_dataset(path: str | Path, crs_wkt: str) -> Path:
    """Create an empty one-feature polygon GeoPackage for native QGIS capture."""
    path = Path(path).expanduser().resolve()
    if path.exists():
        raise ValueError(f"Calculation-domain dataset already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset = ogr.GetDriverByName("GPKG").CreateDataSource(str(path))
    if dataset is None:
        raise ValueError(f"Could not create calculation-domain dataset: {path}")
    spatial = osr.SpatialReference(); spatial.ImportFromWkt(crs_wkt)
    layer = dataset.CreateLayer(DOMAIN_LAYER_NAME, spatial, ogr.wkbPolygon)
    identifier = ogr.FieldDefn("domain_id", ogr.OFTString); identifier.SetWidth(64)
    name = ogr.FieldDefn("name", ogr.OFTString); name.SetWidth(96)
    layer.CreateField(identifier); layer.CreateField(name)
    dataset = None
    return path
