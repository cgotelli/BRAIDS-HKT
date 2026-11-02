"""Small synthetic, non-private hydrodynamic example for the desktop UI."""
from pathlib import Path
import tempfile

import numpy as np
from osgeo import gdal, osr

from .services import default_document, assign_boundary, save_project_folder
from braids_core.boundaries.domain import PolygonDomain


def create_river_demo(root, source=None):
    """Create a new synthetic channel without external project assets."""
    return create_demo(root)


def create_demo(root, backend="cpu", *, end_time_s=5.):
    """Save a new 20 x 8 m channel; no existing project or output is replaced."""
    root = Path(root).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Choose a new empty folder for the synthetic example")
    with tempfile.TemporaryDirectory(prefix="braids-desktop-demo-") as temporary:
        dem = Path(temporary) / "synthetic_channel.tif"
        raster = gdal.GetDriverByName("GTiff").Create(str(dem), 88, 40, 1, gdal.GDT_Float64)
        raster.SetGeoTransform((-1., .25, 0., 9., 0., -.25))
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(2056)
        raster.SetProjection(srs.ExportToWkt())
        xs = -1. + (np.arange(88) + .5) * .25
        raster.GetRasterBand(1).WriteArray(np.broadcast_to(.1 - .001 * xs, (40, 88)).copy())
        raster.FlushCache()
        raster = None
        doc = default_document(dem, [(0., 0.), (20., 0.), (20., 8.), (0., 8.)], "EPSG:2056",
            name="Synthetic channel â€” hydrodynamics", backend=backend,
            minimum_size_m=1., levels=2, end_time_s=end_time_s)
        doc["initial"]["stage_m"] = -1.
        doc["initial"]["velocity_m_s"] = [0., 0.]
        doc["settings"]["dt_max_s"] = min(.1, end_time_s)
        doc["settings"]["output_interval_s"] = min(.5, end_time_s / 2)
        doc["settings"]["controls"]["threads"] = 2
        domain = PolygonDomain(**doc["settings"]["domain"])
        left = [e.edge_id for e in domain.edges if e.start_xy[0] == e.end_xy[0] == 0.]
        right = [e.edge_id for e in domain.edges if e.start_xy[0] == e.end_xy[0] == 20.]
        doc = assign_boundary(doc, left, "Upstream", "inlet",
                              dict(method="inlet.critical_discharge", discharge_m3_s=.32))
        doc = assign_boundary(doc, right, "Downstream", "outlet", dict(method="outlet.stage", stage_m=.5))
        return save_project_folder(root, doc)
