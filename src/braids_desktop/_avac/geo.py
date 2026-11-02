"""GDAL/OGR adapters for the existing scientific raster and ring contracts."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from osgeo import gdal, ogr, osr

@dataclass(frozen=True)
class AvacRaster:
    """Display-only data contract, not an AVAC or BRAIDS solver state."""
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    metadata: dict
    crs_authid: str
    source_band: int = 1

gdal.UseExceptions()
ogr.UseExceptions()
osr.UseExceptions()

Rings = list[tuple[np.ndarray, list[np.ndarray]]]


@dataclass(frozen=True)
class Terrain:
    source: Path
    raster: AvacRaster
    wkt: str
    source_hash: str


def sha256_file(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def spatial_reference(text: str) -> osr.SpatialReference | None:
    if not text or not text.strip():
        return None
    srs = osr.SpatialReference()
    srs.SetFromUserInput(text)
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return srs


def has_geographic_reference(terrain: Terrain) -> bool:
    """Local/unreferenced coordinates cannot locate online background tiles."""
    srs = spatial_reference(terrain.wkt or terrain.raster.crs_authid)
    return srs is not None and not srs.IsLocal()


def geometry_transform(source, target_crs: str):
    """Use raw XY in an unreferenced project; otherwise transform known CRSs."""
    target = spatial_reference(target_crs)
    if target is None or target.IsLocal():
        return None
    if source is None:
        raise ValueError("Geometry has no CRS. Provide its CRS before importing into a georeferenced project.")
    source = spatial_reference(source.ExportToWkt())
    return None if source.IsSame(target) else osr.CoordinateTransformation(source, target)


def read_terrain(path: str | Path, band: int = 1) -> Terrain:
    """Read exact native samples, flip rows, retain outer edges and centres.

    Reject unsupported geometry rather than silently reproject/resample inputs.
    This first reader bounds allocation to 100 million cells; display previews
    are independent of the immutable full-resolution scientific array.
    """
    path = Path(path).expanduser().resolve(strict=True)
    with gdal.Open(str(path), gdal.GA_ReadOnly) as dataset:
        if not 1 <= band <= dataset.RasterCount:
            raise ValueError(f"Raster band {band} is unavailable.")
        width, height = dataset.RasterXSize, dataset.RasterYSize
        if width * height > 100_000_000:
            raise ValueError("DEM exceeds the current 100 million cell memory limit. Select an explicit smaller input.")
        transform = dataset.GetGeoTransform(can_return_null=True)
        if transform is None:
            raise ValueError("DEM has no georeferencing transform.")
        xmin, dx, rx, ymax, ry, dy = transform
        if rx != 0 or ry != 0 or dx <= 0 or dy >= 0:
            raise ValueError("DEM must be north-up with unrotated cells; explicitly convert this input first.")
        srs = dataset.GetSpatialRef()
        if srs is not None and (not (srs.IsProjected() or srs.IsLocal()) or not np.isclose(srs.GetLinearUnits(), 1.0, rtol=0, atol=1e-9)):
            raise ValueError("DEM must use a projected or local CRS in metres; geographic degrees are not solver distances.")
        source_band = dataset.GetRasterBand(band)
        north = source_band.ReadAsArray().astype(np.float64)
        nodata = source_band.GetNoDataValue()
        if nodata is not None:
            # Exact GDAL nodata equality: adjacent legitimate elevations must
            # not disappear through a relative-tolerance comparison.
            north[north == nodata] = np.nan
        mask = source_band.GetMaskBand().ReadAsArray()
        north[mask == 0] = np.nan
        north[~np.isfinite(north)] = np.nan
        if not np.any(np.isfinite(north)):
            raise ValueError("DEM contains no finite elevation samples.")
        # Keep an absent CRS absent, including in saved inputs and exports.
        # The UI warns that these local XY/elevation values are assumed metres.
        wkt = srs.ExportToWkt() if srs is not None else ""
        if srs is not None:
            try:srs.AutoIdentifyEPSG()
            except RuntimeError:pass
        authority, code = (srs.GetAuthorityName(None), srs.GetAuthorityCode(None)) if srs is not None else (None, None)
        crs = f"{authority}:{code}" if authority and code else ('Custom projected CRS' if wkt else '')
    ymin, xmax = ymax + height * dy, xmin + width * dx
    z = np.ascontiguousarray(north[::-1])
    z.setflags(write=False)
    metadata = dict(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, ncols=width,
                    nrows=height, cellsize=dx, cellsize_y=-dy,
                    nodata_value=nodata if nodata is not None else -9999.0)
    x = xmin + (np.arange(width) + .5) * dx
    y = ymin + (np.arange(height) + .5) * (-dy)
    raster = AvacRaster(x, y, z, metadata, crs, band)
    return Terrain(path, raster, wkt, sha256_file(path))


def read_polygons(path: str | Path, target_wkt: str, layer_name: str | None = None) -> Rings:
    """Transform valid Polygon/MultiPolygon features, preserving every hole."""
    dataset = ogr.Open(str(Path(path).expanduser().resolve(strict=True)), 0)
    if dataset is None:
        raise ValueError("Cannot open the polygon dataset.")
    if layer_name is None and dataset.GetLayerCount() != 1:
        raise ValueError("Dataset contains multiple layers; choose a single polygon layer explicitly.")
    layer = dataset.GetLayerByName(layer_name) if layer_name else dataset.GetLayer(0)
    if layer is None:
        raise ValueError("Polygon layer is unavailable.")
    transformation = geometry_transform(layer.GetSpatialRef(), target_wkt)
    rings: Rings = []
    for feature in layer:
        geometry = feature.GetGeometryRef()
        if geometry is None or geometry.IsEmpty():
            continue
        geometry = geometry.Clone()
        if not geometry.IsValid():
            raise ValueError(f"Release feature {feature.GetFID()} has invalid geometry. Repair it explicitly before importing.")
        if transformation is not None:
            geometry.Transform(transformation)
        kind = ogr.GT_Flatten(geometry.GetGeometryType())
        if kind not in (ogr.wkbPolygon, ogr.wkbMultiPolygon):
            raise ValueError(f"Feature {feature.GetFID()} is not a Polygon or MultiPolygon.")
        polygons = [geometry] if kind == ogr.wkbPolygon else [geometry.GetGeometryRef(i) for i in range(geometry.GetGeometryCount())]
        for polygon in polygons:
            contours = [np.asarray(polygon.GetGeometryRef(i).GetPoints(), dtype=float)[:, :2]
                        for i in range(polygon.GetGeometryCount())]
            if not contours or any(not np.all(np.isfinite(r)) or len(r) < 4 for r in contours):
                raise ValueError(f"Release feature {feature.GetFID()} has unusable rings.")
            rings.append((contours[0], contours[1:]))
    if not rings:
        raise ValueError("No valid release polygons found.")
    return rings


def probe(raster: AvacRaster, x: float, y: float) -> tuple[int, int, float] | None:
    meta = raster.metadata
    column = int(np.floor((x - float(meta["xmin"])) / float(meta["cellsize"])))
    row = int(np.floor((y - float(meta["ymin"])) / float(meta.get("cellsize_y", meta["cellsize"])) ))
    if 0 <= row < raster.z.shape[0] and 0 <= column < raster.z.shape[1]:
        return row, column, float(raster.z[row, column])
    return None
