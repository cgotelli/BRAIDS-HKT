"""Small synthetic, non-private hydrodynamic example for the desktop UI."""
from pathlib import Path
import tempfile

import numpy as np
from osgeo import gdal, osr

from .services import default_document, assign_boundary, save_project_folder
from braids_core.boundaries.domain import PolygonDomain


def create_river_demo(root,source=None):
    """Copy the supplied natural-river setup, never its simulation outputs."""
    from copy import deepcopy
    from . import workspace,project_file,services
    root=Path(root).resolve()
    if root.exists() and any(root.iterdir()):raise FileExistsError('Choose a new empty demo folder')
    source=Path(source or Path(__file__).resolve().parents[1]/'desktop/examples/river_demo182').resolve()
    draft=workspace.read_editor(source)
    document=deepcopy(draft['document']) if draft else services.load_project_folder(source)
    # A demo is a fresh case from the packaged physical inputs, not an import
    # of the author's inactive UI drafts or history. Do not migrate old editors.
    document.pop('editor_state',None)
    document['name']='Natural river demonstration'
    saved=save_project_folder(root,document,session={'desktop':{'guides':(draft or {}).get('session',{}).get('desktop',{}).get('guides',[]),'live_results':True}})
    project_file.write(project_file.default_file(root),root,saved)
    return saved


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
            name="Synthetic channel — hydrodynamics", backend=backend,
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
