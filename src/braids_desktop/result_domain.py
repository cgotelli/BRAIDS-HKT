"""Export a reusable calculation-domain polygon from a saved wet-result frame.

This is intentionally a *geometry* export rather than a new numerical wetting
method.  It turns a warm-up frame into the one simple polygon accepted by the
Desktop domain importer.  The threshold avoids exporting a numerical wet film;
the closing/opening operation removes cut-cell sized crenellations without
changing the saved solution.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from shapely import union_all
from shapely.geometry import Polygon


DEFAULT_WET_DEPTH_M = 0.005


def _polygonal_parts(geometry):
    if geometry.is_empty:
        return []
    if geometry.geom_type == 'Polygon':
        return [geometry]
    if geometry.geom_type == 'MultiPolygon':
        return list(geometry.geoms)
    return [part for part in geometry.geoms if part.geom_type == 'Polygon']


def flooded_domain(frame, *, wet_depth_m=DEFAULT_WET_DEPTH_M):
    """Return one smooth, hole-free polygon and an explicit export audit.

    A BRAIDS domain is one simple exterior ring.  Disconnected puddles are not
    useful as an imported channel domain, so only the largest connected wet
    component is retained and this is reported rather than hidden.  Holes are
    filled for the same importer contract; solids can be drawn again later.
    """
    wet_depth_m = float(wet_depth_m)
    if not np.isfinite(wet_depth_m) or wet_depth_m < 0:
        raise ValueError('Wet-depth threshold must be finite and non-negative')
    depth = np.asarray(frame.q[0], dtype=float)
    wet = np.isfinite(depth) & (depth > wet_depth_m)
    if not np.any(wet):
        raise ValueError(f'No cells are deeper than {wet_depth_m:g} m in this result frame')
    cells = [frame.polygons[i] for i in np.flatnonzero(wet)]
    merged = union_all(cells)
    parts = _polygonal_parts(merged)
    if not parts:
        raise ValueError('Flooded cells did not form a polygonal envelope')
    # An AMR cell scale is a defensible, result-specific smoothing scale.  A
    # simplification first allows the exported edge to cross cell boundaries;
    # the round closing/opening then turns its remaining angular joins into a
    # smooth channel envelope.  Neither operation touches solver fields.
    areas = np.asarray([cell.area for cell in cells], dtype=float)
    cell_scale_m = float(np.sqrt(np.median(areas)))
    generalization_m = 1.5 * cell_scale_m
    smoothing_m = cell_scale_m
    source = max(parts, key=lambda item: item.area)
    # A 1.5-cell tolerance removes raster/AMR crenellations but retains bends
    # and channel-scale bars.  Buffer out then in closes narrow notches and
    # retains rounded joins in the saved GeoJSON, rather than snapping the
    # output back to a sequence of mesh-aligned straight segments.
    smooth = source.simplify(generalization_m, preserve_topology=True).buffer(
        smoothing_m, join_style='round', quad_segs=4).buffer(-smoothing_m, join_style='round', quad_segs=4)
    candidates = _polygonal_parts(smooth)
    if not candidates:
        raise ValueError('Smoothing removed the flooded envelope; choose a frame with a wider wet area')
    smooth = max(candidates, key=lambda item: item.area)
    # The importer deliberately takes a simple ring.  Retain a single exterior
    # and report any filled islands/holes so this is never mistaken for a
    # solver representation of islands or solids.
    holes_removed = len(smooth.interiors)
    result = Polygon(smooth.exterior.coords)
    if not result.is_valid:
        result = result.buffer(0)
    candidates = _polygonal_parts(result)
    if not candidates:
        raise ValueError('The smoothed flooded envelope is invalid')
    result = max(candidates, key=lambda item: item.area)
    audit = dict(
        schema='braids.result-domain-export.v1',
        frame_index=int(frame.index), time_s=float(frame.time_s), wet_depth_threshold_m=wet_depth_m,
        wet_cells=int(wet.sum()), wet_area_m2=float(merged.area), selected_component_area_m2=float(source.area),
        exported_area_m2=float(result.area), cell_scale_m=cell_scale_m,
        generalization_tolerance_m=generalization_m, smoothing_radius_m=smoothing_m,
        disconnected_components_discarded=max(0, len(parts)-1), holes_filled=int(holes_removed),
        vertices=int(len(result.exterior.coords)-1),
    )
    return result, audit


def export_result_domain(frame, path, *, crs_wkt='', wet_depth_m=DEFAULT_WET_DEPTH_M):
    """Write an importable GeoJSON Polygon and an adjacent transparent audit."""
    from osgeo import ogr
    from ._avac.geo import spatial_reference

    path = Path(path)
    if path.suffix.lower() not in ('.geojson', '.json'):
        path = path.with_suffix('.geojson')
    if path.exists():
        raise FileExistsError('Choose a new filename; exported domains are never overwritten')
    geometry, audit = flooded_domain(frame, wet_depth_m=wet_depth_m)
    driver = ogr.GetDriverByName('GeoJSON')
    dataset = driver.CreateDataSource(str(path))
    if dataset is None:
        raise ValueError('Cannot create the domain GeoJSON file')
    layer = dataset.CreateLayer('BRAIDS flooded domain', srs=spatial_reference(crs_wkt), geom_type=ogr.wkbPolygon)
    layer.CreateField(ogr.FieldDefn('frame_time_s', ogr.OFTReal))
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for x, y in geometry.exterior.coords:
        ring.AddPoint_2D(float(x), float(y))
    polygon = ogr.Geometry(ogr.wkbPolygon); polygon.AddGeometry(ring)
    feature = ogr.Feature(layer.GetLayerDefn()); feature.SetField('frame_time_s', float(frame.time_s)); feature.SetGeometry(polygon)
    if layer.CreateFeature(feature) != 0:
        dataset = None
        raise ValueError('Cannot write the domain polygon')
    feature = None; dataset = None
    audit_path = path.with_suffix(path.suffix + '.audit.json')
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return path, audit_path, audit
