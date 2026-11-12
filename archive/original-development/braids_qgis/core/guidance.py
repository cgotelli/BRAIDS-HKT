"""Editable drafting lines; these are not hydraulic boundary inputs."""
from pathlib import Path
from osgeo import ogr,osr

GUIDE_LAYER_NAME="braids_inlet_guides"
GUIDE_PROPERTY="braids/inlet_guidance"


def create_guidance_dataset(path,crs_wkt):
    path=Path(path).expanduser().resolve()
    if path.exists():raise ValueError(f"Guidance dataset already exists: {path}")
    path.parent.mkdir(parents=True,exist_ok=True)
    dataset=ogr.GetDriverByName("GPKG").CreateDataSource(str(path))
    if dataset is None:raise ValueError(f"Could not create guidance dataset: {path}")
    spatial=osr.SpatialReference();spatial.ImportFromWkt(crs_wkt)
    layer=dataset.CreateLayer(GUIDE_LAYER_NAME,spatial,ogr.wkbLineString)
    if layer is None:raise ValueError("Could not create the inlet guidance layer")
    for name in ("guide_id","name"):
        field=ogr.FieldDefn(name,ogr.OFTString);field.SetWidth(96)
        if layer.CreateField(field)!=ogr.OGRERR_NONE:raise ValueError(f"Could not create guide field {name}")
    dataset=None
    return path
