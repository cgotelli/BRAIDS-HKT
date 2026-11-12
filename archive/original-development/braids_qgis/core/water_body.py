"""Initial constant-level water-body polygons and whole-cell mapping."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from osgeo import gdal, ogr, osr

from braids_core.boundaries.cut_cells import CutCellMesh, cut_cell_polygon


WATER_BODY_LAYER_NAME = "braids_initial_water_body"


@dataclass(frozen=True)
class WaterBodyRaster:
    """A DEM window stored south-to-north with its outer cell edges."""
    z: np.ndarray
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    cell_x: float
    cell_y: float
    crs_wkt: str


def water_body_seed_cell(raster: WaterBodyRaster, x: float, y: float) -> tuple[int, int]:
    """Return the south-to-north DEM cell containing a clicked point."""
    if min(raster.cell_x, raster.cell_y) <= 0 or not np.isfinite(
            [raster.cell_x, raster.cell_y]).all():
        raise ValueError("DEM cell sizes must be positive and finite")
    if not (raster.xmin <= float(x) < raster.xmax
            and raster.ymin <= float(y) < raster.ymax):
        raise ValueError("The selected point is outside the terrain DEM")
    column = int(np.floor((float(x) - raster.xmin) / raster.cell_x))
    row = int(np.floor((float(y) - raster.ymin) / raster.cell_y))
    return (min(max(row, 0), raster.z.shape[0] - 1),
            min(max(column, 0), raster.z.shape[1] - 1))


def connected_water_body_mask(elevation: np.ndarray, seed: tuple[int, int],
                              water_level_m: float, *, require_closed: bool = True) -> np.ndarray:
    """Flood the four-connected DEM basin at or below a horizontal level."""
    values = np.asarray(elevation, dtype=float)
    if values.ndim != 2 or not values.size:
        raise ValueError("Terrain DEM must be a non-empty two-dimensional grid")
    level = float(water_level_m)
    if not np.isfinite(level):
        raise ValueError("Initial water level must be finite")
    row, column = map(int, seed)
    if not (0 <= row < values.shape[0] and 0 <= column < values.shape[1]):
        raise ValueError("The selected point is outside the terrain grid")
    eligible = np.isfinite(values) & (values <= level)
    if not eligible[row, column]:
        selected = values[row, column]
        detail = "NoData" if not np.isfinite(selected) else f"{selected:g} m"
        raise ValueError(
            f"The selected DEM cell is {detail}, above the {level:g} m water level. "
            "Click inside the inundated basin")

    connected = np.zeros(values.shape, dtype=bool)
    stack = [(row, column)]
    rows, columns = values.shape
    while stack:
        current_row, current_column = stack.pop()
        if connected[current_row, current_column] or not eligible[current_row, current_column]:
            continue
        left = current_column
        while left > 0 and eligible[current_row, left - 1] and not connected[current_row, left - 1]:
            left -= 1
        right = current_column
        while right + 1 < columns and eligible[current_row, right + 1] and not connected[current_row, right + 1]:
            right += 1
        connected[current_row, left:right + 1] = True
        for neighbour_row in (current_row - 1, current_row + 1):
            if not 0 <= neighbour_row < rows:
                continue
            candidates = eligible[neighbour_row, left:right + 1] \
                & ~connected[neighbour_row, left:right + 1]
            starts = np.flatnonzero(candidates & np.r_[True, ~candidates[:-1]])
            stack.extend((neighbour_row, left + int(offset)) for offset in starts)

    if require_closed and (connected[0].any() or connected[-1].any()
                           or connected[:, 0].any() or connected[:, -1].any()):
        raise ValueError(
            "The connected water body reaches the terrain DEM edge, so its contour is not closed. "
            "Use a DEM that encloses the water body or lower the water level")
    return connected


def write_connected_water_body(path: str | Path, raster: WaterBodyRaster,
                               mask: np.ndarray, *, water_level_m: float,
                               seed_x: float, seed_y: float,
                               water_id: str = "water_1") -> Path:
    """Polygonize one connected wet DEM component into a BRAIDS GeoPackage."""
    path = Path(path).expanduser().resolve()
    values = np.asarray(mask, dtype=np.uint8)
    if values.shape != np.asarray(raster.z).shape or not np.any(values):
        raise ValueError("Water-body mask must match the DEM window and contain wet cells")
    create_water_body_dataset(path, raster.crs_wkt)
    source = gdal.GetDriverByName("MEM").Create(
        "", values.shape[1], values.shape[0], 1, gdal.GDT_Byte)
    if source is None:
        path.unlink(missing_ok=True)
        raise RuntimeError("GDAL could not create the water-body mask raster")
    source.SetGeoTransform((raster.xmin, raster.cell_x, 0., raster.ymax, 0., -raster.cell_y))
    source.SetProjection(raster.crs_wkt)
    band = source.GetRasterBand(1)
    band.WriteArray(np.flipud(values))
    dataset = ogr.Open(str(path), 1)
    layer = None if dataset is None else dataset.GetLayerByName(WATER_BODY_LAYER_NAME)
    if layer is None or gdal.Polygonize(band, band, layer, -1, []) != gdal.CE_None \
            or layer.GetFeatureCount() != 1:
        dataset = None
        source = None
        path.unlink(missing_ok=True)
        raise ValueError("GDAL could not create one polygon from the connected water body")
    feature = layer.GetNextFeature()
    feature.SetField("water_id", str(water_id))
    feature.SetField("name", f"Initial Water at {float(water_level_m):g} m")
    feature.SetField("stage_m", float(water_level_m))
    feature.SetField("seed_x", float(seed_x))
    feature.SetField("seed_y", float(seed_y))
    feature.SetField("cells", int(np.count_nonzero(values)))
    if layer.SetFeature(feature) != ogr.OGRERR_NONE:
        dataset = None
        source = None
        path.unlink(missing_ok=True)
        raise ValueError("Could not save the derived water-body attributes")
    feature = None
    dataset = None
    source = None
    return path


def create_water_body_dataset(path: str | Path, crs_wkt: str) -> Path:
    """Create an empty editable polygon layer for initial water extent."""
    path = Path(path).expanduser().resolve()
    if path.exists():
        raise ValueError(f"Initial-water dataset already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset = ogr.GetDriverByName("GPKG").CreateDataSource(str(path))
    if dataset is None:
        raise ValueError(f"Could not create initial-water dataset: {path}")
    spatial = osr.SpatialReference(); spatial.ImportFromWkt(crs_wkt)
    layer = dataset.CreateLayer(WATER_BODY_LAYER_NAME, spatial, ogr.wkbPolygon)
    identifier = ogr.FieldDefn("water_id", ogr.OFTString); identifier.SetWidth(64)
    name = ogr.FieldDefn("name", ogr.OFTString); name.SetWidth(96)
    stage = ogr.FieldDefn("stage_m", ogr.OFTReal); stage.SetWidth(20); stage.SetPrecision(8)
    seed_x = ogr.FieldDefn("seed_x", ogr.OFTReal); seed_x.SetWidth(24); seed_x.SetPrecision(8)
    seed_y = ogr.FieldDefn("seed_y", ogr.OFTReal); seed_y.SetWidth(24); seed_y.SetPrecision(8)
    cells = ogr.FieldDefn("cells", ogr.OFTInteger64)
    layer.CreateField(identifier); layer.CreateField(name); layer.CreateField(stage)
    layer.CreateField(seed_x); layer.CreateField(seed_y); layer.CreateField(cells)
    dataset = None
    return path


def _cell_polygon(vertices) -> ogr.Geometry:
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for x, y in vertices:
        ring.AddPoint_2D(float(x), float(y))
    ring.CloseRings()
    polygon = ogr.Geometry(ogr.wkbPolygon); polygon.AddGeometry(ring)
    return polygon


def map_water_body_to_cells(mesh: CutCellMesh, path: str | Path, crs_wkt: str,
                            layer_name: str | None = None) -> tuple[np.ndarray, dict]:
    """Select whole retained cells with positive-area polygon intersection."""
    if not isinstance(mesh, CutCellMesh):
        raise ValueError("CutCellMesh is required")
    path = Path(path).expanduser().resolve()
    dataset = ogr.Open(str(path), 0)
    if dataset is None:
        raise ValueError(f"Cannot open initial-water polygon dataset: {path}")
    layer = dataset.GetLayerByName(layer_name) if layer_name else dataset.GetLayer(0)
    if layer is None:
        raise ValueError(f"Initial-water polygon layer {layer_name!r} is missing")
    expected = osr.SpatialReference(); expected.ImportFromWkt(crs_wkt)
    spatial = layer.GetSpatialRef()
    if spatial is None or not spatial.IsSame(expected):
        raise ValueError("Initial-water polygons and calculation domain must use the same CRS")
    geometries = []
    for feature in layer:
        geometry = feature.GetGeometryRef()
        if geometry is None or geometry.IsEmpty():
            raise ValueError("Initial-water polygon layer contains empty geometry")
        kind = ogr.GT_Flatten(geometry.GetGeometryType())
        if kind not in (ogr.wkbPolygon, ogr.wkbMultiPolygon) or not geometry.IsValid():
            raise ValueError("Initial-water geometry must be a valid polygon or multipolygon")
        geometries.append(geometry.Clone())
    dataset = None
    if not geometries:
        raise ValueError("Initial-water polygon layer has no features")

    selected = np.zeros(len(mesh.cells), dtype=bool)
    cell_shapes = [cut_cell_polygon(cell) for cell in mesh.cells]
    cell_envelopes = [shape.GetEnvelope() for shape in cell_shapes]
    for geometry in geometries:
        xmin, xmax, ymin, ymax = geometry.GetEnvelope()
        for cell_id, (cell, envelope) in enumerate(zip(cell_shapes, cell_envelopes)):
            if selected[cell_id]:
                continue
            cxmin, cxmax, cymin, cymax = envelope
            if cxmax < xmin or cxmin > xmax or cymax < ymin or cymin > ymax:
                continue
            intersection = cell.Intersection(geometry)
            if intersection is not None and not intersection.IsEmpty() and intersection.GetArea() > 0.:
                selected[cell_id] = True
    if not selected.any():
        raise ValueError("Initial-water polygons do not overlap any retained calculation cell")
    return selected, {
        "mode": "constant-level polygon",
        "path": str(path),
        "layer": layer_name,
        "polygon_count": len(geometries),
        "selected_cell_count": int(selected.sum()),
        "cell_rule": "whole retained cell on positive-area polygon intersection",
    }
