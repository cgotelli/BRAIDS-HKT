"""Configurable nested geometry and conservative state contract for adaptive runs.

Geometry is immutable: every possible leaf is clipped against exactly the same
domain and obstacles. Selecting leaves changes topology, never physical walls.
The native driver consumes this tree and performs adaptation without Python in
the time loop. Python selection/transfer routines are preparation and reference
implementations used to verify the compiled operators independently.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from functools import cached_property
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np
from osgeo import ogr

from .cut_cells import CutCellMesh, cut_cell_polygon, map_cut_cells
from .inlet_io import NativeGrid

AMR_LEVELS = 4
AMR_RATIO = 2
AMR_MIN_LEVELS = 2
AMR_MAX_LEVELS = 8


@dataclass(frozen=True)
class AMRSettings:
    enabled: bool = False
    levels: int = AMR_LEVELS
    refinement_ratio: int = AMR_RATIO
    regrid_interval_s: float = 1.
    coarsen_delay: int = 3
    wet_level: int = 1
    buffer_cells: int = 3
    wet_depth_m: float = 1e-6
    surface_tolerance_m: float = .02
    velocity_tolerance_m_s: float = .2
    bed_tolerance_m: float = .02
    concentration_tolerance_kg_m3: float = .01
    composition_tolerance: float = .05
    bed_change_tolerance_m: float = .002
    refine_wet_dry: bool = True
    refine_walls: bool = True
    refine_bathymetry: bool = True
    refine_hydraulics: bool = True
    refine_sediment: bool = True
    subcycling: bool = True

    def __post_init__(self):
        if (isinstance(self.levels,bool) or not isinstance(self.levels,int) or
                not AMR_MIN_LEVELS<=self.levels<=AMR_MAX_LEVELS):
            raise ValueError(f'AMR levels must be an integer from {AMR_MIN_LEVELS} to {AMR_MAX_LEVELS}')
        if self.refinement_ratio != AMR_RATIO or isinstance(self.refinement_ratio,bool):
            raise ValueError('AMR refinement ratio is fixed at 2:1')
        for name in ('enabled', 'refine_wet_dry', 'refine_walls', 'refine_bathymetry',
                     'refine_hydraulics', 'refine_sediment', 'subcycling'):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f'{name} must be boolean')
        for name, low, high in (('coarsen_delay', 1, 1000), ('wet_level', 1, self.levels-1),
                                ('buffer_cells', 2, 32)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                raise ValueError(f'{name} must be an integer between {low} and {high}')
        for name in ('regrid_interval_s', 'wet_depth_m', 'surface_tolerance_m',
                     'velocity_tolerance_m_s', 'bed_tolerance_m',
                     'concentration_tolerance_kg_m3', 'composition_tolerance',
                     'bed_change_tolerance_m'):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be positive and finite')
        if self.composition_tolerance > 1:
            raise ValueError('Composition tolerance must not exceed one')


@dataclass(frozen=True)
class AMRHierarchy:
    meshes: tuple[CutCellMesh, ...]
    offsets: tuple[int, ...]
    parent: np.ndarray
    children: tuple[tuple[int, ...], ...]
    node_level: np.ndarray
    maximum_level: np.ndarray
    refinement_ratio: int

    @property
    def count(self):
        return len(self.parent)

    @property
    def cells(self):
        return tuple(cell for mesh in self.meshes for cell in mesh.cells)

    @property
    def finest(self):
        return self.meshes[-1]

    @property
    def levels(self):
        return len(self.meshes)

    @property
    def finest_level(self):
        return self.levels-1

    @property
    def roots(self):
        return tuple(range(self.offsets[1]))

    def leaf_cover(self, leaf_ids):
        """Map descendants to the active ancestor; refined parents map to -1."""
        leaves = tuple(int(n) for n in leaf_ids)
        if len(set(leaves)) != len(leaves) or any(n < 0 or n >= self.count for n in leaves):
            raise ValueError('Leaf IDs must be unique valid hierarchy nodes')
        cover = np.full(self.count, -1, dtype=np.int64)
        cover[list(leaves)] = leaves
        for node in range(self.count):
            p = self.parent[node]
            if p >= 0 and cover[p] >= 0:
                if cover[node] >= 0:
                    raise ValueError('An active AMR cell overlaps an active ancestor')
                cover[node] = cover[p]
        if np.any(cover[self.offsets[-2]:] < 0):
            raise ValueError('AMR leaves must cover the complete flowing domain')
        return cover

    def select(self, leaf_ids, *, require_balance=True):
        """Build one non-overlapping mesh with one flux per coarse/fine segment."""
        leaves = tuple(sorted(int(n) for n in leaf_ids))
        cover = self.leaf_cover(leaves)
        local = {node: i for i, node in enumerate(leaves)}
        cells, faces = [], []
        for node in leaves:
            level = int(self.node_level[node])
            old = self.meshes[level].cells[node-self.offsets[level]]
            stride = self.refinement_ratio**(self.finest_level-level)
            members = tuple((i*stride+di, j*stride+dj, component)
                            for i,j,component in (old.member_indices or ((old.i,old.j,old.component),))
                            for dj in range(stride) for di in range(stride))
            cells.append(replace(old, cell_id=local[node], i=old.i*stride, j=old.j*stride,
                                 level=level, hierarchy_node_id=node,
                                 member_indices=members,
                                 nominal_area_m2=old.nominal_area_m2 or self.meshes[level].grid.dx_m*self.meshes[level].grid.dy_m))
        for level, mesh in enumerate(self.meshes):
            base = self.offsets[level]
            for face in mesh.faces:
                o = base+face.owner
                n = -1 if face.neighbour < 0 else base+face.neighbour
                a = int(cover[o])
                b = -1 if n < 0 else int(cover[n])
                if a < 0 or (n >= 0 and b < 0) or a == b:
                    continue
                if (n < 0 and o != a) or (n >= 0 and o != a and n != b):
                    continue  # only the finer active side owns this subface
                if n >= 0 and require_balance and abs(int(self.node_level[a])-int(self.node_level[b])) > 1:
                    raise ValueError('Adjacent AMR leaves differ by more than one level')
                face_id=len(faces);face_owner=local[a];face_neighbour=-1 if b < 0 else local[b]
                faces.append(face if (face.face_id==face_id and face.owner==face_owner and
                    face.neighbour==face_neighbour) else replace(face,face_id=face_id,
                    owner=face_owner,neighbour=face_neighbour))
        result = replace(self.finest, cells=tuple(cells), faces=tuple(faces))
        expected = self.meshes[0].metrics()['area_m2']
        audit = result.metrics()
        tolerance = 1e-9*max(self.meshes[0].grid.dx_m, self.meshes[0].grid.dy_m, 1.)
        if not math.isclose(audit['area_m2'], expected, rel_tol=1e-11, abs_tol=1e-11):
            raise ValueError('Adaptive leaf areas do not partition the domain')
        if audit['closure_linf_m'] > tolerance:
            raise ValueError('Adaptive face closure failed')
        return result

    @cached_property
    def _level_neighbours(self):
        # Immutable geometry; reuse these graph edges for successive decisions.
        return tuple(np.array([(f.owner,f.neighbour) for f in m.faces if f.neighbour>=0],
                              dtype=np.int64).reshape(-1,2).T for m in self.meshes)

    def balanced_leaves(self, requested_levels, *, transition_layers=1):
        """Take the maximum request in each subtree, then enforce face 2:1 balance."""
        if type(transition_layers) is not int or not 1<=transition_layers<=32:
            raise ValueError('Transition layers must be an integer from 1 to 32')
        wanted = np.asarray(requested_levels)
        if wanted.shape != (self.count,) or not np.isfinite(wanted).all() or \
                np.any(wanted != wanted.astype(int)) or np.any((wanted < 0) | (wanted >= self.levels)):
            raise ValueError(f'Each hierarchy node needs a requested level from zero to {self.finest_level}')
        wanted = wanted.astype(np.int64).copy()
        # A one-level hierarchy has exactly one admissible complete partition.
        # No face graph, neighbour grading or replacement mesh is needed to
        # prove 2:1 balance. Validate the request above even in this case.
        if self.levels==1:return self.roots
        # Breaklines are geometric constraints at the finest requested scale.
        # Pin both sides; their faces remain ordinary permeable interior faces.
        # Internal solid walls pin the adjoining fluid cells too. Ordinary
        # exterior domain walls are NOT a mandatory finest-refinement request.
        for face in self.finest.faces:
            if face.breakline_ids or face.boundary_role=='solid_obstacle':
                wanted[self.offsets[-2]+face.owner]=self.finest_level
                if face.neighbour>=0:wanted[self.offsets[-2]+face.neighbour]=self.finest_level
        for node in range(self.count):
            p = self.parent[node]
            if p >= 0:
                wanted[node] = max(wanted[node], wanted[p])
        for node in range(self.count-1, -1, -1):
            p = self.parent[node]
            if p >= 0:
                wanted[p] = max(wanted[p], wanted[node])
        if transition_layers>1:
            # Add bands once around this assessment's requests. At the finest
            # level the requested cells already supply the first row. Each
            # coarser level gets N face-neighbour rows at its OWN cell size.
            # Cropped/merged cells use true adjacency, never Cartesian indices.
            for level in range(self.finest_level,0,-1):
                lo,hi=self.offsets[level:level+2];local=wanted[lo:hi]
                active=local>=level if level==self.finest_level else local>level
                a,b=self._level_neighbours[level]
                rings=transition_layers-1 if level==self.finest_level else transition_layers
                for _ in range(rings):
                    expanded=active.copy();expanded[a[active[b]]]=True;expanded[b[active[a]]]=True
                    active=expanded
                local[active]=np.maximum(local[active],level)
                np.maximum.at(wanted,self.parent[lo:hi],local)
        wanted = np.minimum(wanted, self.maximum_level)
        leaves = set()

        def visit(node):
            if self.node_level[node] < wanted[node] and self.children[node]:
                for child in self.children[node]:
                    visit(child)
            else:
                leaves.add(node)

        for node in self.roots:
            visit(node)
        for _ in range(self.count+1):
            cover = self.leaf_cover(sorted(leaves))
            refine = set()
            for level, (owner,neighbour) in enumerate(self._level_neighbours):
                base = self.offsets[level]
                a,b=cover[base+owner],cover[base+neighbour]
                keep=(a>=0)&(b>=0)&(a!=b);a,b=a[keep],b[keep]
                la,lb=self.node_level[a],self.node_level[b];jump=abs(la-lb)>1
                for coarse in np.unique(np.where(la<lb,a,b)[jump]):
                    coarse=int(coarse)
                    if self.node_level[coarse]>=self.maximum_level[coarse] or not self.children[coarse]:
                        raise ValueError('AMR safety cap is incompatible with 2:1 balance')
                    refine.add(coarse)
            if not refine:
                return tuple(sorted(leaves))
            for node in sorted(refine):
                leaves.remove(node)
                leaves.update(self.children[node])
        raise ValueError('AMR 2:1 balance did not converge')

    def restrict(self, finest_values):
        """Area-weighted inventories; no concentration or bed-elevation averaging."""
        return self.restrict_many((finest_values,))[0]

    def restrict_many(self, fields):
        """Share geometry work, retaining each field's original dot/divide."""
        fields=tuple(np.asarray(v,dtype=float) for v in fields)
        if any(v.ndim<1 or v.shape[0]!=len(self.finest.cells) or not np.isfinite(v).all() for v in fields):
            raise ValueError('Finite data on every finest cell is required')
        results=tuple(np.empty((self.count,*v.shape[1:]),dtype=float) for v in fields)
        for result,value in zip(results,fields):result[self.offsets[-2]:]=value
        cells = self.cells
        for node in range(self.offsets[-2]-1, -1, -1):
            child = self.children[node]
            weights = np.array([cells[c].area_m2 for c in child])
            # Normalize by the sum of child areas to preserve constants at map-coordinate roundoff.
            ids=list(child);denominator=math.fsum(weights)
            for result in results:
                result[node]=np.tensordot(weights,result[ids],axes=(0,0))/denominator
        return results

    def write_geometry(self, path):
        """Stable node polygons for output visualization and native state mapping."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == '.npz':
            from .geometry_io import write_packed_geometry

            def rows():
                for level, mesh in enumerate(self.meshes):
                    stride = self.refinement_ratio**(self.finest_level-level)
                    for cell in mesh.cells:
                        node = self.offsets[level]+cell.cell_id
                        yield dict(vars(cell), cell_id=node, hierarchy_node_id=node,
                                   level=level, i=cell.i*stride, j=cell.j*stride,
                                   source_member_indices=cell.member_indices,
                                   member_scale=stride,
                                   nominal_area_m2=cell.nominal_area_m2 or mesh.grid.dx_m*mesh.grid.dy_m)

            metadata = dict(semantic_format='braids-amr-hierarchy-v2', levels=self.levels,
                refinement_ratio=self.refinement_ratio,
                grid=asdict(self.finest.grid), base_grid=asdict(self.meshes[0].grid),
                domain_id=self.finest.domain_id, crs_wkt=self.finest.crs_wkt,
                members_cover_cell=True)
            return write_packed_geometry(path, cell_count=self.count, rows=rows,
                metadata=metadata, extra_arrays={
                    'offsets':np.asarray(self.offsets,dtype=np.int64),
                    'parent':np.asarray(self.parent,dtype=np.int32),
                    'node_level':np.asarray(self.node_level,dtype=np.int8),
                    'maximum_level':np.asarray(self.maximum_level,dtype=np.int8)})
        node_cells=[]
        for level,mesh in enumerate(self.meshes):
            stride=self.refinement_ratio**(self.finest_level-level)
            for cell in mesh.cells:
                node=self.offsets[level]+cell.cell_id
                members=tuple((i*stride+di,j*stride+dj,component)
                              for i,j,component in (cell.member_indices or ((cell.i,cell.j,cell.component),))
                              for dj in range(stride) for di in range(stride))
                row=vars(cell).copy()
                row.update(cell_id=node,hierarchy_node_id=node,level=level,
                    i=cell.i*stride,j=cell.j*stride,member_indices=members,
                    nominal_area_m2=cell.nominal_area_m2 or mesh.grid.dx_m*mesh.grid.dy_m)
                node_cells.append(row)
        payload = dict(format='braids-amr-hierarchy-v2', levels=self.levels,
                       refinement_ratio=self.refinement_ratio,
                       grid=asdict(self.finest.grid), base_grid=asdict(self.meshes[0].grid),
                       domain_id=self.finest.domain_id, crs_wkt=self.finest.crs_wkt,
                       offsets=list(self.offsets), parent=self.parent.tolist(),
                       node_level=self.node_level.tolist(),
                       maximum_level=self.maximum_level.tolist(),
                       cells=node_cells)
        path.write_text(json.dumps(payload, allow_nan=False, separators=(',', ':'))+'\n')
        return payload


def build_hierarchy(domain, assignments, base_grid, obstacles=(), *, progress: Callable | None = None,
                    minimum_area_fraction: float = 0., cancelled: Callable | None = None,
                    levels: int = AMR_LEVELS, refinement_ratio: int = AMR_RATIO):
    """Clip nested Cartesian levels and establish component ancestry.

    ``progress`` receives a monotonic percentage. Work is weighted by the
    expected number of cells at each 2:1 level so a fine level does not leave
    the progress bar apparently stuck near completion.
    """
    settings=AMRSettings(enabled=True,levels=levels,refinement_ratio=refinement_ratio)
    level_count=settings.levels;refinement_ratio=settings.refinement_ratio
    assignments=tuple(assignments)
    meshes, audits, offsets = [], [], [0]
    level_weights=np.asarray([refinement_ratio**(2*level)
                              for level in range(level_count)],dtype=float)
    level_weights/=level_weights.sum()
    level_starts=np.concatenate(([0.],np.cumsum(level_weights)))
    clipping_share=90.
    for level in range(level_count):
        if cancelled and cancelled():
            raise InterruptedError("BRAIDS AMR mesh preparation cancelled")
        ratio = refinement_ratio**level
        grid = NativeGrid(base_grid.nx*ratio, base_grid.ny*ratio,
                          base_grid.xlower_m, base_grid.ylower_m,
                          base_grid.dx_m/ratio, base_grid.dy_m/ratio)
        level_start=clipping_share*level_starts[level]
        level_span=clipping_share*level_weights[level]
        if progress:progress(level_start)
        mesh = map_cut_cells(
            domain, assignments, grid, obstacles, cancelled=cancelled,
            progress=(lambda value,start=level_start,span=level_span:
                      progress(start+.7*span*value/100.)) if progress else None,
        )
        if minimum_area_fraction > 0:
            from .agglomeration import agglomerate_small_cells
            mesh, _, audit = agglomerate_small_cells(mesh, minimum_area_fraction,
                                                     preserve_parent_boundaries=True,
                                                     consume_input=True)
        else:
            audit = {'unresolved_small_cells': []}
        meshes.append(mesh)
        audits.append(audit)
        offsets.append(offsets[-1]+len(mesh.cells))
        if progress:progress(level_start+level_span)
    parent = np.full(offsets[-1], -1, dtype=np.int64)
    node_levels = np.concatenate([np.full(len(m.cells), l, dtype=np.int64) for l, m in enumerate(meshes)])
    children = [[] for _ in parent]
    origin = (base_grid.xlower_m, base_grid.ylower_m)
    for level in range(1, level_count):
        coarse = meshes[level-1]
        by_index = {}
        for c in coarse.cells:
            for i,j,_ in (c.member_indices or ((c.i,c.j,c.component),)):
                row=by_index.setdefault((i,j), [])
                if c.cell_id not in row:
                    row.append(c.cell_id)
        shapes = {}
        for cell in meshes[level].cells:
            candidates = by_index[(cell.i//refinement_ratio, cell.j//refinement_ratio)]
            if len(candidates) == 1:
                match = candidates[0]
            else:
                piece = cut_cell_polygon(cell, origin)
                point = piece.PointOnSurface()
                matches = []
                for c in candidates:
                    if c not in shapes:
                        shapes[c] = cut_cell_polygon(coarse.cells[c], origin)
                    if shapes[c].Intersects(point):
                        matches.append(c)
                if len(matches) != 1:
                    raise ValueError('Cannot identify unique parent of disconnected cut-cell component')
                match = matches[0]
            node = offsets[level]+cell.cell_id
            p = offsets[level-1]+match
            parent[node] = p
            children[p].append(node)
        if progress:progress(clipping_share+5.*level/(level_count-1))
    maximum = np.full(offsets[-1], level_count-1, dtype=np.int64)
    # The finest level is optional. If a finest fragment cannot be merged
    # without crossing its parent boundary, keep
    # the safe parent active. Lower-level fragments are unavoidable physical
    # boundary cells and must not cause large regions to lose resolution.
    parent_nominal = meshes[-2].grid.dx_m*meshes[-2].grid.dy_m
    for cell_id in audits[-1]['unresolved_small_cells']:
        node = offsets[-2]+cell_id
        safe = int(parent[node])
        if meshes[-2].cells[safe-offsets[-2]].area_m2 >= minimum_area_fraction*parent_nominal-1e-14:
            maximum[safe] = level_count-2
    # Open hydraulic boundaries always retain the finest available geometry.
    # Clear any safety cap along their ancestry so every native regrid can
    # descend to the minimum configured cell size at the inlet/outlet.
    open_boundary_ids={assignment.bc_id for assignment in assignments
                       if assignment.kind in ('inlet','outlet')}
    for face in meshes[-1].faces:
        if face.bc_id not in open_boundary_ids:
            continue
        node=offsets[-2]+face.owner
        while node>=0:
            maximum[node]=level_count-1
            node=int(parent[node])
    hierarchy = AMRHierarchy(tuple(meshes), tuple(offsets), parent,
                             tuple(tuple(c) for c in children),node_levels,maximum,refinement_ratio)
    cells = hierarchy.cells
    for node in range(offsets[-2]):
        total = math.fsum(cells[c].area_m2 for c in children[node])
        if not children[node] or not math.isclose(total, cells[node].area_m2,
                                                 rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError(f'Children of AMR node {node} do not conserve clipped area')
        if progress and (node%4096==0 or node+1==offsets[-2]):
            progress(95.+5.*(node+1)/max(1,offsets[-2]))
    if progress:progress(100.)
    return hierarchy


def conservative_water_children(volume, beds, areas):
    """Fill children at one stage with exactly the parent's retained water volume."""
    beds, areas = np.asarray(beds, dtype=float), np.asarray(areas, dtype=float)
    if beds.ndim != 1 or beds.shape != areas.shape or not len(beds) or \
            not np.isfinite(beds).all() or not np.isfinite(areas).all() or \
            np.any(areas <= 0) or not math.isfinite(volume) or volume < 0:
        raise ValueError('Finite positive child areas and nonnegative water volume are required')
    order = np.argsort(beds, kind='stable')
    wetted_area = 0.
    bed_moment = 0.
    stage = float(beds[order[0]])
    for k, index in enumerate(order):
        wetted_area += areas[index]
        bed_moment += areas[index]*beds[index]
        stage = (volume+bed_moment)/wetted_area
        if k == len(order)-1 or stage <= beds[order[k+1]]:
            break
    depth = np.maximum(stage-beds, 0.)
    # Correct only roundoff, using the largest wet inventory to retain positivity.
    residual = volume-float(depth@areas)
    largest = int(np.argmax(depth*areas))
    depth[largest] += residual/areas[largest]
    if depth[largest] < 0 and depth[largest] > -1e-14:
        depth[largest] = 0.
    return depth


def iter_native_hierarchy_lines(hierarchy, settings, boundary_ids, minimum_levels=None):
    """Yield the deterministic native tree without retaining millions of rows."""
    from .cut_cell_io import _canonical_section_endpoints
    if not isinstance(hierarchy, AMRHierarchy) or not isinstance(settings, AMRSettings) or not settings.enabled:
        raise ValueError('An enabled AMRSettings and AMRHierarchy are required')
    if settings.levels!=hierarchy.levels or settings.refinement_ratio!=hierarchy.refinement_ratio:
        raise ValueError('AMR settings and hierarchy levels/ratio disagree')
    minimum = np.zeros(hierarchy.count, dtype=int) if minimum_levels is None else np.asarray(minimum_levels)
    if minimum.shape != (hierarchy.count,) or np.any(minimum != minimum.astype(int)) or \
            np.any((minimum < 0) | (minimum >= hierarchy.levels)):
        raise ValueError(f'Minimum refinement levels must contain one integer L0–L{hierarchy.finest_level} per node')
    controls = [settings.regrid_interval_s, settings.coarsen_delay, settings.wet_level,
                settings.buffer_cells, settings.wet_depth_m, settings.surface_tolerance_m,
                settings.velocity_tolerance_m_s, settings.bed_tolerance_m,
                settings.concentration_tolerance_kg_m3, settings.composition_tolerance,
                settings.bed_change_tolerance_m, int(settings.refine_wet_dry),
                int(settings.refine_walls), int(settings.refine_bathymetry),
                int(settings.refine_hydraulics), int(settings.refine_sediment), int(settings.subcycling)]
    yield f'5 {hierarchy.levels} {hierarchy.refinement_ratio} {hierarchy.count} {sum(len(m.faces) for m in hierarchy.meshes)} {hierarchy.offsets[-2]}'
    yield ' '.join(format(v, '.17g') for v in controls)
    for level, mesh in enumerate(hierarchy.meshes):
        stride = hierarchy.refinement_ratio**(hierarchy.finest_level-level)
        for cell in mesh.cells:
            node = hierarchy.offsets[level]+cell.cell_id
            row = [hierarchy.parent[node]+1, level, cell.i*stride+1, cell.j*stride+1,
                   stride, cell.area_m2, *cell.centroid_xy, int(minimum[node]),
                   int(hierarchy.maximum_level[node])]
            yield ' '.join(format(v, '.17g') for v in row)
    for level, mesh in enumerate(hierarchy.meshes):
        base = hierarchy.offsets[level]
        for face in mesh.faces:
            sampled = int(face.bed_elevation_m is not None)
            left, right = _canonical_section_endpoints(face)
            row = [base+face.owner+1, 0 if face.neighbour < 0 else base+face.neighbour+1,
                   boundary_ids.get(face.bc_id, 0), level,
                   face.face_id+1 if level == hierarchy.finest_level else 0,
                   int(face.boundary_role == 'solid_obstacle'), *face.outward_normal,
                   face.length_m, *((np.array(face.start_xy)+face.end_xy)/2), sampled,
                   0. if face.bed_elevation_m is None else face.bed_elevation_m,
                   0. if left is None else left, 0. if right is None else right]
            yield ' '.join(format(v, '.17g') for v in row)


def native_hierarchy_lines(hierarchy, settings, boundary_ids, minimum_levels=None):
    """Compatibility materialization for callers that explicitly need a list."""
    return list(iter_native_hierarchy_lines(hierarchy, settings, boundary_ids, minimum_levels))
