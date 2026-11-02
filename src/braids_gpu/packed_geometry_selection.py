"""Direct packed-array selection for zero-step AMR viewer geometry.

The legacy path expands every selected cell into Python dictionaries and then
packs the same arrays again.  This helper preserves that file contract while
selecting directly from the authenticated packed hierarchy.  It does not build
solver faces, choose cells, or advance numerical state.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from braids_core.boundaries.geometry_io import COMPACT_FORMAT


def _expanded(source, scale, i, j, component, cover):
    if len(source):
        return np.asarray([(int(mi) * scale + di, int(mj) * scale + dj,
                            int(comp))
                           for mi, mj, comp in source
                           for dj in range(scale) for di in range(scale)],
                          dtype=np.int32)
    if cover:
        return np.asarray([(i + di, j + dj, component)
                           for dj in range(scale) for di in range(scale)],
                          dtype=np.int32)
    return np.empty((0, 3), dtype=np.int32)


def write_selected_packed_geometry(source, output, selected_ids):
    """Write the exact legacy selected geometry without object expansion."""
    source, output = Path(source).resolve(), Path(output).resolve()
    ids = np.asarray(selected_ids, dtype=np.int64)
    if (source.suffix.lower() != '.npz' or output.suffix.lower() != '.npz' or
            ids.ndim != 1 or not len(ids) or len(np.unique(ids)) != len(ids)):
        raise ValueError('Packed AMR geometry selection differs')
    with np.load(source, allow_pickle=False) as data:
        required = {'metadata_json', 'cell_integer', 'cell_scalar',
            'member_scale', 'regular_cell', 'vertex_offsets', 'vertices_xy',
            'cell_hole_offsets', 'hole_vertex_offsets', 'hole_vertices_xy',
            'member_offsets', 'source_member_indices'}
        if not required <= set(data.files):
            raise ValueError('Packed AMR hierarchy geometry is incomplete')
        source_integer = data['cell_integer']
        if np.any(ids < 0) or np.any(ids >= len(source_integer)):
            raise IndexError('Selected AMR geometry cell is outside hierarchy')
        metadata = json.loads(str(data['metadata_json'].item()))
        grid = metadata.get('grid')
        if not isinstance(grid, dict):
            raise ValueError('Packed AMR hierarchy lacks Cartesian grid metadata')
        integers = np.ascontiguousarray(source_integer[ids], dtype=np.int32)
        integers[:, 0] = np.arange(len(ids), dtype=np.int32)
        scalars = np.ascontiguousarray(data['cell_scalar'][ids], dtype=np.float64)
        # ``load_geometry`` historically omitted this field from its expanded
        # row, so ``write_packed_geometry`` used its documented default of one.
        member_scale = np.ones(len(ids), dtype=np.int8)
        regular = np.empty(len(ids), dtype=np.uint8)
        vertex_offsets = np.empty(len(ids) + 1, dtype=np.int64)
        cell_hole_offsets = np.empty(len(ids) + 1, dtype=np.int64)
        member_offsets = np.empty(len(ids) + 1, dtype=np.int64)
        vertex_offsets[0] = cell_hole_offsets[0] = member_offsets[0] = 0
        source_vertex_offsets = data['vertex_offsets']
        source_hole_offsets = data['cell_hole_offsets']
        source_ring_offsets = data['hole_vertex_offsets']
        source_member_offsets = data['member_offsets']
        source_scale = data['member_scale']
        source_regular = data['regular_cell']
        source_vertices = data['vertices_xy']
        source_hole_vertices = data['hole_vertices_xy']
        source_members = data['source_member_indices']
        cover = bool(metadata.get('members_cover_cell', False))
        dx, dy = float(grid['dx_m']), float(grid['dy_m'])
        xlower, ylower = float(grid['xlower_m']), float(grid['ylower_m'])
        vertices, holes, members = [], [], []
        ring_lengths = []
        for local, node_value in enumerate(ids):
            node = int(node_value)
            _, i, j, component, _, _ = map(int, source_integer[node])
            scale = int(source_scale[node])
            h0, h1 = map(int, source_hole_offsets[node:node + 2])
            selected_holes = []
            for ring in range(h0, h1):
                p0, p1 = map(int, source_ring_offsets[ring:ring + 2])
                selected_holes.append(np.asarray(
                    source_hole_vertices[p0:p1], dtype=np.float64))
            cell_hole_offsets[local + 1] = (
                cell_hole_offsets[local] + len(selected_holes))
            holes.extend(selected_holes)
            ring_lengths.extend(len(value) for value in selected_holes)
            m0, m1 = map(int, source_member_offsets[node:node + 2])
            selected_members = _expanded(source_members[m0:m1], scale,
                                         i, j, component, cover)
            members.append(selected_members)
            member_offsets[local + 1] = member_offsets[local] + len(selected_members)
            if bool(source_regular[node]):
                x0, y0 = xlower + i * dx, ylower + j * dy
                x1, y1 = x0 + scale * dx, y0 + scale * dy
                ring = np.asarray(((x0, y0), (x0, y1), (x1, y1),
                                   (x1, y0), (x0, y0)), dtype=np.float64)
            else:
                v0, v1 = map(int, source_vertex_offsets[node:node + 2])
                ring = np.asarray(source_vertices[v0:v1], dtype=np.float64)
            # The repacker tests regularity using the expanded row's default
            # member scale (one), not the hierarchy's source scale.
            expected = np.asarray(((xlower + i * dx, ylower + j * dy),
                (xlower + i * dx, ylower + (j + 1) * dy),
                (xlower + (i + 1) * dx, ylower + (j + 1) * dy),
                (xlower + (i + 1) * dx, ylower + j * dy),
                (xlower + i * dx, ylower + j * dy)), dtype=np.float64)
            is_regular = (not selected_holes and not len(selected_members) and
                          ring.shape == (5, 2) and np.array_equal(ring, expected))
            regular[local] = int(is_regular)
            if not is_regular:
                vertices.append(ring)
            vertex_offsets[local + 1] = vertex_offsets[local] + (
                0 if is_regular else len(ring))
        vertices_xy = (np.concatenate(vertices).astype(np.float64, copy=False)
                       if vertices else np.empty((0, 2), dtype=np.float64))
        hole_vertices_xy = (np.concatenate(holes).astype(np.float64, copy=False)
                            if holes else np.empty((0, 2), dtype=np.float64))
        hole_vertex_offsets = np.empty(len(ring_lengths) + 1, dtype=np.int64)
        hole_vertex_offsets[0] = 0
        if ring_lengths:
            np.cumsum(ring_lengths, out=hole_vertex_offsets[1:])
        member_rows = (np.concatenate(members).astype(np.int32, copy=False)
                       if members else np.empty((0, 3), dtype=np.int32))
    packed_metadata = {key: value for key, value in metadata.items()
                       if key != 'faces'}
    packed_metadata['semantic_format'] = 'braids-cut-cell-mesh-v1'
    packed_metadata.update(format=COMPACT_FORMAT, cell_count=len(ids),
        faces_scope='physical boundaries only', faces=[])
    arrays = {
        'metadata_json': np.asarray(json.dumps(packed_metadata, allow_nan=False,
                                               separators=(',', ':'))),
        'cell_integer': integers, 'cell_scalar': scalars,
        'member_scale': member_scale, 'regular_cell': regular,
        'vertex_offsets': vertex_offsets, 'vertices_xy': vertices_xy,
        'cell_hole_offsets': cell_hole_offsets,
        'hole_vertex_offsets': hole_vertex_offsets,
        'hole_vertices_xy': hole_vertices_xy,
        'member_offsets': member_offsets,
        'source_member_indices': member_rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.partial')
    with temporary.open('wb') as handle:
        np.savez(handle, **arrays)
    temporary.replace(output)
    return output
