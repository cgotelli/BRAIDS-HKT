"""Compact persisted cut-cell geometry with legacy JSON compatibility.

The native text deck remains the authoritative solver input.  These sidecars
exist for QGIS result rasterization, mesh display and audit.  Large cases use
an uncompressed NumPy archive so writing does not construct multi-gigabyte JSON
object graphs.  Small and historical JSON artifacts remain readable.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from braids_core.windows_paths import filesystem_path


COMPACT_FORMAT = "braids-packed-cell-geometry-v1"


def geometry_artifact(root: str | Path, stem: str) -> Path:
    """Return the current packed artifact or a legacy JSON artifact."""
    root = filesystem_path(root)
    packed = root / f"{stem}.npz"
    legacy = root / f"{stem}.json"
    if packed.is_file():
        return packed
    if legacy.is_file():
        return legacy
    raise FileNotFoundError(f"Missing {stem}.npz or {stem}.json in {root}")


def _value(row: Any, name: str, default=None):
    return row.get(name, default) if isinstance(row, Mapping) else getattr(row, name, default)


def _regular_ring(row: Any, grid: Mapping[str, Any]) -> tuple[tuple[float, float], ...] | None:
    holes = _value(row, "holes_xy", ()) or ()
    members = _value(row, "source_member_indices", _value(row, "member_indices", ())) or ()
    if holes or members:
        return None
    scale = int(_value(row, "member_scale", 1))
    x0 = float(grid["xlower_m"]) + int(_value(row, "i")) * float(grid["dx_m"])
    y0 = float(grid["ylower_m"]) + int(_value(row, "j")) * float(grid["dy_m"])
    x1 = x0 + scale * float(grid["dx_m"])
    y1 = y0 + scale * float(grid["dy_m"])
    expected = ((x0, y0), (x0, y1), (x1, y1), (x1, y0), (x0, y0))
    actual = _value(row, "vertices_xy")
    if len(actual) != 5 or any(float(actual[k][axis]) != expected[k][axis]
                               for k in range(5) for axis in range(2)):
        return None
    return expected


def _cell_counts(rows: Iterable[Any], grid: Mapping[str, Any], cell_count: int
                 ) -> tuple[int, int, int, np.ndarray]:
    vertices = holes = members = 0
    regular = np.empty(cell_count, dtype=np.uint8)
    for index,row in enumerate(rows):
        is_regular = _regular_ring(row, grid) is not None
        regular[index] = is_regular
        if not is_regular:
            vertices += len(_value(row, "vertices_xy"))
        rings = _value(row, "holes_xy", ()) or ()
        holes += sum(len(ring) for ring in rings)
        members += len(_value(row, "source_member_indices",
                             _value(row, "member_indices", ())) or ())
    return vertices, holes, members, regular


def write_packed_geometry(
    path: str | Path,
    *,
    cell_count: int,
    rows: Callable[[], Iterable[Any]],
    metadata: Mapping[str, Any],
    physical_faces: Sequence[Any] = (),
    extra_arrays: Mapping[str, np.ndarray] | None = None,
) -> Path:
    """Write exact cell polygons and compact scalar/index arrays atomically."""
    path = filesystem_path(path)
    if path.suffix != ".npz":
        raise ValueError("Packed geometry path must end in .npz")
    if isinstance(cell_count, bool) or cell_count < 1:
        raise ValueError("Packed geometry requires at least one cell")
    grid = metadata.get("grid")
    if not isinstance(grid, Mapping):
        raise ValueError("Packed geometry metadata requires a grid")
    vertex_count, hole_vertex_count, member_count, regular_cell = _cell_counts(
        rows(), grid, cell_count)
    integers = np.empty((cell_count, 6), dtype=np.int32)
    scalars = np.empty((cell_count, 4), dtype=np.float64)
    member_scale = np.empty(cell_count, dtype=np.int8)
    vertex_offsets = np.empty(cell_count + 1, dtype=np.int64)
    vertices = np.empty((vertex_count, 2), dtype=np.float64)
    cell_hole_offsets = np.empty(cell_count + 1, dtype=np.int64)
    hole_vertex_offsets = [0]
    hole_vertices = np.empty((hole_vertex_count, 2), dtype=np.float64)
    member_offsets = np.empty(cell_count + 1, dtype=np.int64)
    members = np.empty((member_count, 3), dtype=np.int32)
    vertex_cursor = hole_cursor = member_cursor = ring_cursor = 0
    vertex_offsets[0] = cell_hole_offsets[0] = member_offsets[0] = 0
    for index, row in enumerate(rows()):
        if index >= cell_count:
            raise ValueError("Packed geometry yielded more cells than declared")
        integers[index] = (
            int(_value(row, "cell_id")), int(_value(row, "i")), int(_value(row, "j")),
            int(_value(row, "component")), int(_value(row, "level", 0)),
            int(_value(row, "hierarchy_node_id", -1)),
        )
        centroid = _value(row, "centroid_xy")
        scalars[index] = (float(_value(row, "area_m2")), float(centroid[0]),
                          float(centroid[1]), float(_value(row, "nominal_area_m2", 0.)))
        member_scale[index] = int(_value(row, "member_scale", 1))
        regular = bool(regular_cell[index])
        if not regular:
            ring = np.asarray(_value(row, "vertices_xy"), dtype=np.float64)
            vertices[vertex_cursor:vertex_cursor + len(ring)] = ring
            vertex_cursor += len(ring)
        vertex_offsets[index + 1] = vertex_cursor
        for hole in (_value(row, "holes_xy", ()) or ()):
            hole = np.asarray(hole, dtype=np.float64)
            hole_vertices[hole_cursor:hole_cursor + len(hole)] = hole
            hole_cursor += len(hole)
            ring_cursor += 1
            hole_vertex_offsets.append(hole_cursor)
        cell_hole_offsets[index + 1] = ring_cursor
        source = _value(row, "source_member_indices",
                        _value(row, "member_indices", ())) or ()
        if source:
            source = np.asarray(source, dtype=np.int32)
            members[member_cursor:member_cursor + len(source)] = source
            member_cursor += len(source)
        member_offsets[index + 1] = member_cursor
    if index + 1 != cell_count:
        raise ValueError("Packed geometry yielded fewer cells than declared")
    face_rows = [vars(face).copy() if not isinstance(face, Mapping) else dict(face)
                 for face in physical_faces if int(_value(face, "neighbour")) < 0]
    packed_metadata = dict(metadata)
    packed_metadata.update(format=COMPACT_FORMAT, cell_count=cell_count,
                           faces_scope="physical boundaries only", faces=face_rows)
    arrays = {
        "metadata_json": np.asarray(json.dumps(packed_metadata, allow_nan=False,
                                                separators=(",", ":"))),
        "cell_integer": integers,
        "cell_scalar": scalars,
        "member_scale": member_scale,
        "regular_cell": regular_cell,
        "vertex_offsets": vertex_offsets,
        "vertices_xy": vertices,
        "cell_hole_offsets": cell_hole_offsets,
        "hole_vertex_offsets": np.asarray(hole_vertex_offsets, dtype=np.int64),
        "hole_vertices_xy": hole_vertices,
        "member_offsets": member_offsets,
        "source_member_indices": members,
    }
    if extra_arrays:
        overlap = set(arrays) & set(extra_arrays)
        if overlap:
            raise ValueError(f"Duplicate packed array names: {sorted(overlap)}")
        arrays.update({str(name): np.asarray(value) for name, value in extra_arrays.items()})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    temporary.replace(path)
    return path


def _expanded_members(source: np.ndarray, scale: int, i: int, j: int,
                      component: int, cover_cell: bool):
    if len(source):
        return tuple((int(mi) * scale + di, int(mj) * scale + dj, int(comp))
                     for mi, mj, comp in source
                     for dj in range(scale) for di in range(scale))
    if cover_cell:
        return tuple((i + di, j + dj, component)
                     for dj in range(scale) for di in range(scale))
    return ()


def load_geometry(path: str | Path, *, cell_ids: Sequence[int] | np.ndarray | None = None,
                  include_faces: bool = True) -> dict[str, Any]:
    """Load selected cell rows from packed geometry or a historical JSON file."""
    path = filesystem_path(path)
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if cell_ids is not None:
            payload["cells"] = [payload["cells"][int(index)] for index in cell_ids]
        if not include_faces:
            payload["faces"] = []
        return payload
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        integers = data["cell_integer"]
        scalars = data["cell_scalar"]
        selected = (np.arange(len(integers), dtype=np.int64) if cell_ids is None
                    else np.asarray(cell_ids, dtype=np.int64))
        if np.any(selected < 0) or np.any(selected >= len(integers)):
            raise IndexError("Packed geometry cell ID is outside the artifact")
        cells = []
        vertex_offsets = data["vertex_offsets"]
        cell_hole_offsets = data["cell_hole_offsets"]
        hole_vertex_offsets = data["hole_vertex_offsets"]
        member_offsets = data["member_offsets"]
        cover_cell = bool(metadata.get("members_cover_cell", False))
        for packed_index in selected:
            packed_index = int(packed_index)
            cell_id, i, j, component, level, hierarchy_node_id = map(
                int, integers[packed_index])
            v0, v1 = map(int, vertex_offsets[packed_index:packed_index + 2])
            h0, h1 = map(int, cell_hole_offsets[packed_index:packed_index + 2])
            holes = []
            for ring in range(h0, h1):
                p0, p1 = map(int, hole_vertex_offsets[ring:ring + 2])
                holes.append(tuple(map(tuple, data["hole_vertices_xy"][p0:p1].tolist())))
            m0, m1 = map(int, member_offsets[packed_index:packed_index + 2])
            scale = int(data["member_scale"][packed_index])
            if "regular_cell" in data.files and bool(data["regular_cell"][packed_index]):
                grid = metadata["grid"]
                x0 = float(grid["xlower_m"]) + i * float(grid["dx_m"])
                y0 = float(grid["ylower_m"]) + j * float(grid["dy_m"])
                x1 = x0 + scale * float(grid["dx_m"])
                y1 = y0 + scale * float(grid["dy_m"])
                cell_vertices = ((x0,y0),(x0,y1),(x1,y1),(x1,y0),(x0,y0))
            else:
                cell_vertices = tuple(map(tuple, data["vertices_xy"][v0:v1].tolist()))
            cells.append({
                "cell_id": cell_id, "i": i, "j": j, "component": component,
                "area_m2": float(scalars[packed_index, 0]),
                "centroid_xy": tuple(map(float, scalars[packed_index, 1:3])),
                "vertices_xy": cell_vertices,
                "member_indices": _expanded_members(data["source_member_indices"][m0:m1],
                                                     scale, i, j, component, cover_cell),
                "source_member_count": int(m1 - m0) if m1 > m0 else 1,
                "holes_xy": tuple(holes), "level": level,
                "hierarchy_node_id": hierarchy_node_id,
                "nominal_area_m2": float(scalars[packed_index, 3]),
            })
        result = {key: value for key, value in metadata.items() if key != "faces"}
        result["cells"] = cells
        result["faces"] = metadata.get("faces", []) if include_faces else []
        return result


def load_hierarchy_index(path: str | Path) -> dict[str, np.ndarray | dict[str, Any]]:
    """Load only tree arrays and areas, without expanding millions of cell dicts."""
    path = filesystem_path(path)
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {
            "metadata": {key: value for key, value in payload.items()
                         if key not in {"cells", "parent", "node_level", "maximum_level", "offsets"}},
            "parent": np.asarray(payload["parent"], dtype=np.int64),
            "node_level": np.asarray(payload["node_level"], dtype=np.int8),
            "maximum_level": np.asarray(payload["maximum_level"], dtype=np.int8),
            "offsets": np.asarray(payload["offsets"], dtype=np.int64),
            "area_m2": np.asarray([cell["area_m2"] for cell in payload["cells"]], dtype=float),
        }
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        return {"metadata": {key: value for key, value in metadata.items() if key != "faces"},
                "parent": data["parent"].copy(), "node_level": data["node_level"].copy(),
                "maximum_level": data["maximum_level"].copy(),
                "offsets": data["offsets"].copy(), "area_m2": data["cell_scalar"][:, 0].copy()}
