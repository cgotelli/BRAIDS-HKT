"""Content-owned GEOS geometry shared by construction and cell-order checks.

Only immutable Shapely geometries and scalar tuples are retained. Coordinates
and their order are the key, not object IDs, timestamps or a caller's approval
flag. Area/overlap/order policies remain with their respective callers.
"""
from functools import lru_cache
import numpy as np
from shapely.geometry import Polygon


@lru_cache(maxsize=32768)
def polygon_record(vertex_bytes):
    points = np.frombuffer(vertex_bytes, dtype=np.float64).reshape(-1, 2)
    polygon = Polygon(points)
    return (polygon, polygon.is_valid, polygon.area, polygon.bounds, polygon.length)


def polygon_records(polygons):
    records = []
    for points in polygons:
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError('Invalid polygon vertices')
        if len(points) > 1 and np.array_equal(points[0], points[-1]):
            points = points[:-1]
        records.append(polygon_record(points.tobytes()))
    return records
