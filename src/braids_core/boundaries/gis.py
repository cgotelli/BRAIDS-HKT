"""Read hydraulic sections from projected/local metre rasters and polylines.

The DEM's first band supplies elevations, including any GDAL scale and offset.
Elevations are bilinearly interpolated between pixel centres. In the outer half
pixel, the nearest centre value is extended constantly to the raster footprint;
coordinates beyond that footprint are errors. This convention matters on sloping
channels: an edge section samples the first/last column, not the analytic bed at
the outer edge. No hydraulic assumption is used to extrapolate the terrain.

Every section has an explicitly supplied inward unit normal, independent of line
digitization. Station zero is the lexicographically smaller endpoint (x, then y).
Optional ``seed_station_m`` attributes use this canonical station convention.
They retain provenance; the default section includes every submerged interval,
so a legacy seed does not exclude another wet portion of an assigned inlet.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from pathlib import Path
from typing import Sequence

import numpy as np
from osgeo import gdal, ogr, osr

from .section import Section


_METRE_UNITS = {"m", "meter", "meters", "metre", "metres"}


def _projected_metres(spatial_reference: osr.SpatialReference | str | None,
                      label: str) -> osr.SpatialReference:
    if spatial_reference is None:
        raise ValueError(f"{label} requires an explicit projected or local metre CRS")
    if isinstance(spatial_reference, osr.SpatialReference):
        spatial = spatial_reference.Clone()
    elif isinstance(spatial_reference, str) and spatial_reference.strip():
        spatial = osr.SpatialReference()
        try:
            result = spatial.SetFromUserInput(spatial_reference)
        except RuntimeError as error:
            raise ValueError(f"{label} has an invalid CRS") from error
        if result != 0:
            raise ValueError(f"{label} has an invalid CRS")
    else:
        raise ValueError(f"{label} requires an explicit projected or local metre CRS")
    if not (spatial.IsProjected() or spatial.IsLocal()) or not math.isclose(
            spatial.GetLinearUnits(), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{label} CRS must be projected or local with metre horizontal units")
    spatial.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return spatial


def _identifier(bc_id: object) -> str:
    if not isinstance(bc_id, str) or not bc_id.strip() or bc_id != bc_id.strip():
        raise ValueError("bc_id must be a nonempty string without surrounding whitespace")
    return bc_id


def _normal(inward_normal: Sequence[float]) -> tuple[float, float]:
    try:
        if len(inward_normal) != 2:
            raise ValueError
        nx, ny = (float(value) for value in inward_normal)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("inward_normal requires two finite unit-vector components") from error
    if not math.isfinite(nx) or not math.isfinite(ny) or not math.isclose(
            math.hypot(nx, ny), 1.0, rel_tol=0.0, abs_tol=1e-10):
        raise ValueError("inward_normal must be a finite, nonzero unit vector")
    return nx, ny


def _line_coordinates(coordinates: Sequence[Sequence[float]]) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    try:
        for point in coordinates:
            if len(point) != 2:
                raise ValueError
            x, y = (float(value) for value in point)
            if not math.isfinite(x) or not math.isfinite(y):
                raise ValueError
            points.append((x, y))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("A boundary polyline requires finite (x, y) coordinates") from error
    if len(points) < 2:
        raise ValueError("A boundary polyline requires at least two points")
    if any(first == second for first, second in zip(points, points[1:])):
        raise ValueError("A boundary polyline contains a zero-length segment")
    if points[0] == points[-1]:
        raise ValueError("A boundary polyline must be open, not closed")
    geometry = ogr.Geometry(ogr.wkbLineString)
    for x, y in points:
        geometry.AddPoint_2D(x, y)
    if not geometry.IsValid() or not geometry.IsSimple():
        raise ValueError("A boundary polyline must be valid and simple (no self-intersections)")
    if points[-1] < points[0]:
        points.reverse()
    return points


class _RasterSampler:
    def __init__(self, path: str | Path, *, allow_undeclared_metres: bool = False):
        try:
            self.dataset = gdal.OpenEx(str(path), gdal.OF_RASTER | gdal.OF_READONLY)
        except RuntimeError as error:
            raise ValueError(f"Cannot open DEM: {path}") from error
        if self.dataset is None or self.dataset.RasterCount < 1:
            raise ValueError(f"Cannot open DEM with an elevation band: {path}")
        self.spatial = _projected_metres(self.dataset.GetSpatialRef(), "DEM")
        self.band = self.dataset.GetRasterBand(1)
        declared_units = self.band.GetUnitType().strip()
        units = declared_units.lower()
        if units not in _METRE_UNITS and not (allow_undeclared_metres and not units):
            raise ValueError("DEM band must explicitly declare metre elevation units")
        self.declared_vertical_units = declared_units or None
        self.vertical_units_source = ("DEM band metadata" if declared_units else
                                      "BRAIDS project metre-elevation convention")
        self.width = self.dataset.RasterXSize
        self.height = self.dataset.RasterYSize
        self.transform = self.dataset.GetGeoTransform(can_return_null=True)
        if self.transform is None or not all(math.isfinite(v) for v in self.transform):
            raise ValueError("DEM requires a finite affine geotransform")
        _, a, b, _, c, d = self.transform
        determinant = a * d - b * c
        if determinant == 0 or not math.isfinite(determinant):
            raise ValueError("DEM geotransform must be invertible")
        self.inverse = (d / determinant, -b / determinant,
                        -c / determinant, a / determinant)
        # The smallest singular value also bounds sampling on skewed rasters.
        self.max_spacing = float(np.linalg.svd([[a, b], [c, d]], compute_uv=False)[-1]) / 2
        if self.max_spacing <= 0 or not math.isfinite(self.max_spacing):
            raise ValueError("DEM pixel dimensions must be positive and finite")
        self.nodata = self.band.GetNoDataValue()
        self.scale = self.band.GetScale()
        self.offset = self.band.GetOffset()
        self.scale = 1.0 if self.scale is None else self.scale
        self.offset = 0.0 if self.offset is None else self.offset
        if not math.isfinite(self.scale) or not math.isfinite(self.offset):
            raise ValueError("DEM elevation scale and offset must be finite")
        self.mask = None if self.band.GetMaskFlags() & gdal.GMF_ALL_VALID else self.band.GetMaskBand()
        self.cache: dict[tuple[int, int], float] = {}
        block_width, block_height = self.band.GetBlockSize()
        self.block_width = max(1, int(block_width))
        self.block_height = max(1, int(block_height))
        # Reading one GDAL pixel at a time dominates dense section sampling.
        # Keep a small LRU of native raster blocks: scanline GeoTIFFs remain
        # memory bounded and tiled rasters avoid thousands of driver calls.
        self.block_cache: OrderedDict[
            tuple[int, int], tuple[np.ndarray, np.ndarray | None]
        ] = OrderedDict()
        self.block_cache_limit = 32

    def _block(self, row: int, col: int) -> tuple[np.ndarray, np.ndarray | None, int, int]:
        block_row = row // self.block_height
        block_col = col // self.block_width
        key = block_row, block_col
        cached = self.block_cache.get(key)
        xoff = block_col * self.block_width
        yoff = block_row * self.block_height
        if cached is None:
            width = min(self.block_width, self.width - xoff)
            height = min(self.block_height, self.height - yoff)
            values = self.band.ReadAsArray(xoff, yoff, width, height)
            if values is None:
                raise ValueError(
                    f"Cannot read DEM block at row {yoff}, column {xoff}"
                )
            mask = None if self.mask is None else self.mask.ReadAsArray(
                xoff, yoff, width, height
            )
            if self.mask is not None and mask is None:
                raise ValueError(
                    f"Cannot read DEM mask block at row {yoff}, column {xoff}"
                )
            cached = np.asarray(values), None if mask is None else np.asarray(mask)
            self.block_cache[key] = cached
            if len(self.block_cache) > self.block_cache_limit:
                self.block_cache.popitem(last=False)
        else:
            self.block_cache.move_to_end(key)
        return cached[0], cached[1], yoff, xoff

    def _pixel(self, row: int, col: int) -> float:
        key = row, col
        if key not in self.cache:
            values, mask, yoff, xoff = self._block(row, col)
            raw = float(values[row - yoff, col - xoff])
            if (not math.isfinite(raw) or
                    (self.nodata is not None and raw == self.nodata) or
                    (mask is not None and mask[row - yoff, col - xoff] == 0)):
                raise ValueError(f"DEM interpolation touches NoData/invalid pixel at row {row}, column {col}")
            value = raw * self.scale + self.offset
            if not math.isfinite(value):
                raise ValueError("DEM elevation is not finite after applying scale and offset")
            self.cache[key] = value
        return self.cache[key]

    def sample(self, x: float, y: float) -> float:
        x0, _, _, y0, _, _ = self.transform
        a, b, c, d = self.inverse
        px, py = a * (x - x0) + b * (y - y0), c * (x - x0) + d * (y - y0)
        # Tolerate only affine arithmetic roundoff at the closed outer footprint.
        coordinate_roundoff = 8 * max(math.ulp(x), math.ulp(y), math.ulp(x0), math.ulp(y0))
        tolerance = coordinate_roundoff * max(abs(a) + abs(b), abs(c) + abs(d))
        if not (-tolerance <= px <= self.width + tolerance and
                -tolerance <= py <= self.height + tolerance):
            raise ValueError(f"Boundary coordinate ({x}, {y}) lies outside the DEM footprint")
        col = min(max(px - 0.5, 0.0), self.width - 1.0)
        row = min(max(py - 0.5, 0.0), self.height - 1.0)
        col0, row0 = math.floor(col), math.floor(row)
        dc, dr = col - col0, row - row0
        result = 0.0
        for r, wr in ((row0, 1.0 - dr), (min(row0 + 1, self.height - 1), dr)):
            for cc, wc in ((col0, 1.0 - dc), (min(col0 + 1, self.width - 1), dc)):
                weight = wr * wc
                if weight > 0:
                    result += weight * self._pixel(r, cc)
        return result


def _section(sampler: _RasterSampler, coordinates: Sequence[Sequence[float]], *,
             bc_id: str, inward_normal: Sequence[float],
             sample_spacing_m: float | None, datum_m: float | None,
             max_stage_m: float | None, seed_station_m: float | None) -> Section:
    bc_id = _identifier(bc_id)
    normal = _normal(inward_normal)
    points = _line_coordinates(coordinates)
    spacing = sampler.max_spacing if sample_spacing_m is None else float(sample_spacing_m)
    if not math.isfinite(spacing) or spacing <= 0:
        raise ValueError("sample_spacing_m must be positive and finite")
    if spacing > sampler.max_spacing * (1 + 1e-12):
        raise ValueError(f"sample_spacing_m must be at most half a DEM pixel ({sampler.max_spacing:g} m)")
    stations, elevations = [], []
    station = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        length = math.hypot(x1 - x0, y1 - y0)
        if not math.isfinite(length):
            raise ValueError("Boundary polyline length must be finite")
        count = math.ceil(length / spacing)
        for index in range(count):
            fraction = index / count
            stations.append(station + fraction * length)
            elevations.append(sampler.sample(x0 + fraction * (x1 - x0), y0 + fraction * (y1 - y0)))
        station += length
    stations.append(station)
    elevations.append(sampler.sample(*points[-1]))
    return Section(bc_id=bc_id, stations_m=stations, bed_elevations_m=elevations,
                   datum_m=min(elevations) if datum_m is None else datum_m,
                   inward_normal=normal, horizontal_units="m", vertical_units="m",
                   max_stage_m=max_stage_m, seed_station_m=seed_station_m)


def section_from_polyline(dem_path: str | Path,
                          coordinates: Sequence[Sequence[float]], *,
                          bc_id: str, inward_normal: Sequence[float],
                          spatial_reference: osr.SpatialReference | str,
                          sample_spacing_m: float | None = None,
                          datum_m: float | None = None,
                          max_stage_m: float | None = None,
                          seed_station_m: float | None = None,
                          allow_undeclared_metres: bool = False) -> Section:
    """Sample one open XY polyline, requiring its declared CRS to match the DEM.

    Vertices are retained; each segment is subdivided into equal intervals no
    longer than ``sample_spacing_m`` (default: half the smallest pixel scale).
    Larger requested spacing is rejected. Datum defaults to the sampled minimum;
    supply the surveyed thalweg datum when absolute depth must refer to it.
    """
    sampler = _RasterSampler(
        dem_path, allow_undeclared_metres=allow_undeclared_metres)
    spatial = _projected_metres(spatial_reference, "Boundary")
    if not sampler.spatial.IsSame(spatial):
        raise ValueError("DEM and boundary CRS must match; reproject the data explicitly")
    return _section(sampler, coordinates, bc_id=bc_id, inward_normal=inward_normal,
                    sample_spacing_m=sample_spacing_m, datum_m=datum_m,
                    max_stage_m=max_stage_m, seed_station_m=seed_station_m)


def load_boundary_sections(dem_path: str | Path, gpkg_path: str | Path,
                           layer_name: str = "boundaries",
                           sample_spacing_m: float | None = None,
                           datum_m: float | None = None) -> dict[str, Section]:
    """Read uniquely identified sections from a GeoPackage LineString layer.

    Required fields are string ``bc_id`` and numeric ``normal_x``, ``normal_y``.
    The normals point into the model domain and are never inferred from vertex
    order. Optional numeric fields are ``datum_m``, ``max_stage_m``, and
    ``seed_station_m``. A function-level datum overrides feature datums. Neither
    CRS nor vertical units are guessed or silently converted. The result is
    sorted by ID, independent of feature storage order.
    """
    sampler = _RasterSampler(dem_path)
    try:
        dataset = gdal.OpenEx(str(gpkg_path), gdal.OF_VECTOR | gdal.OF_READONLY)
    except RuntimeError as error:
        raise ValueError(f"Cannot open boundary dataset: {gpkg_path}") from error
    if dataset is None:
        raise ValueError(f"Cannot open boundary dataset: {gpkg_path}")
    layer = dataset.GetLayerByName(layer_name)
    if layer is None:
        raise ValueError(f"Boundary layer {layer_name!r} does not exist")
    spatial = _projected_metres(layer.GetSpatialRef(), "Boundary")
    if not sampler.spatial.IsSame(spatial):
        raise ValueError("DEM and boundary CRS must match; reproject the data explicitly")
    definition = layer.GetLayerDefn()
    indices = {name: definition.GetFieldIndex(name) for name in
               ("bc_id", "normal_x", "normal_y", "datum_m", "max_stage_m", "seed_station_m")}
    for name in ("bc_id", "normal_x", "normal_y"):
        if indices[name] < 0:
            raise ValueError(f"Boundary layer is missing required field {name!r}")
    if definition.GetFieldDefn(indices["bc_id"]).GetType() != ogr.OFTString:
        raise ValueError("Boundary bc_id field must have string type")
    for name in ("normal_x", "normal_y", "datum_m", "max_stage_m", "seed_station_m"):
        if indices[name] >= 0 and definition.GetFieldDefn(indices[name]).GetType() not in (
                ogr.OFTReal, ogr.OFTInteger, ogr.OFTInteger64):
            raise ValueError(f"Boundary {name} field must have numeric type")

    sections = {}
    for feature in layer:
        bc_id = _identifier(feature.GetField(indices["bc_id"]))
        if bc_id in sections:
            raise ValueError(f"Duplicate boundary bc_id: {bc_id!r}")
        geometry = feature.GetGeometryRef()
        if (geometry is None or geometry.IsEmpty() or
                ogr.GT_Flatten(geometry.GetGeometryType()) != ogr.wkbLineString):
            raise ValueError(f"Boundary {bc_id!r} must contain one nonempty LineString")

        def optional(name: str) -> float | None:
            if indices[name] < 0 or not feature.IsFieldSetAndNotNull(indices[name]):
                return None
            return float(feature.GetField(indices[name]))

        for name in ("normal_x", "normal_y"):
            if not feature.IsFieldSetAndNotNull(indices[name]):
                raise ValueError(f"Boundary {bc_id!r} requires an explicit {name}")
        sections[bc_id] = _section(
            sampler, [(geometry.GetX(i), geometry.GetY(i)) for i in range(geometry.GetPointCount())],
            bc_id=bc_id, inward_normal=(feature.GetField(indices["normal_x"]), feature.GetField(indices["normal_y"])),
            sample_spacing_m=sample_spacing_m,
            datum_m=datum_m if datum_m is not None else optional("datum_m"),
            max_stage_m=optional("max_stage_m"), seed_station_m=optional("seed_station_m"))
    if not sections:
        raise ValueError("Boundary layer contains no sections")
    return dict(sorted(sections.items()))
