"""Explicit inlet/outlet alignment; walls retain polygonal cut-cell geometry.

Open sections are aligned to their dominant Cartesian tangent, with a common
grid-line normal coordinate. Only vertices belonging to those sections move.
Other wall vertices remain exact; adjacent wall ends reconnect to moved section
ends. There is no centre mask, polygon rasterization or staircase wall.
The returned polygon and assignments are NEW geometry and must be presented
and saved by the caller. No flux, hydraulic method or time integrator changes.
"""
from dataclasses import asdict
import math
import numpy as np

from .domain import PolygonDomain, BoundaryAssignment, boundary_polyline, _polygon
from .inlet_io import NativeGrid


def finest_grid(grid, levels):
    """The actual nested Cartesian lattice, including its saved origin."""
    if type(levels) is not int or not 1 <= levels <= 6:
        raise ValueError('Expected 1..6 nested 2:1 grid levels')
    ratio = 2 ** (levels - 1)
    return NativeGrid(grid.nx*ratio, grid.ny*ratio, grid.xlower_m,
                      grid.ylower_m, grid.dx_m/ratio, grid.dy_m/ratio)


def validate_open_grid(domain, assignments, grid):
    """Reject off-lattice open sections without changing input geometry.

    ``grid`` is the actual finest lattice, not just a nominal resolution.
    An axis-aligned line halfway between grid lines still cuts every inlet
    owner in its normal direction. Transverse endpoints may intersect cropped
    walls between nodes; those wall intersections remain valid geometry.
    """
    assignments = tuple(assignments)
    segments = domain.boundary_segments(assignments)
    origin = np.array([grid.xlower_m, grid.ylower_m])
    spacing = np.array([grid.dx_m, grid.dy_m])
    audit = {}
    for name in sorted({s.bc_id for s in segments if s.kind != 'wall'}):
        line, _ = boundary_polyline(domain, assignments, name)
        points = np.asarray(line, dtype=float)
        tangent = int(np.argmax(np.abs(points[-1]-points[0])))
        cross = 1-tangent
        # Large GIS coordinates need floating-point tolerance, not a fraction
        # of a metre/cell that could admit a real cropped boundary row.
        scale = max(1., abs(origin[cross]), float(np.max(np.abs(points[:,cross]))))
        tolerance = max(1e-10*spacing[cross], 16*np.spacing(scale))
        nearest = origin[cross]+np.rint((points[:,cross]-origin[cross])/spacing[cross])*spacing[cross]
        axis = 'xy'[cross]
        if np.ptp(points[:,cross]) > tolerance:
            raise ValueError(f'Boundary {name!r} is not axis-aligned. Use Mesh > Align inlet/outlet sections '
                             'and review the proposed geometry before preparing.')
        if float(np.max(np.abs(points[:,cross]-nearest))) > tolerance:
            raise ValueError(f'Boundary {name!r} is axis-aligned but {axis}={points[0,cross]:.12g} m '
                f'is not on the actual finest grid (origin {origin[cross]:.12g} m, '
                f'spacing {spacing[cross]:.12g} m; nearest line {nearest[0]:.12g} m). '
                'Use Mesh > Align inlet/outlet sections and review the proposed geometry after changing cell size or levels.')
        audit[name] = dict(normal_axis=axis, coordinate_m=float(points[0,cross]),
                           grid_line_m=float(nearest[0]), tolerance_m=float(tolerance))
    return audit


def derive_axis_aligned(domain, assignments, grid, *, cancelled=lambda: False):
    """Return (new domain, new assignments, audit) on the supplied finest grid.

    Reject invalid polygons, intersecting sections and conflicting endpoints.
    Never repair them by filling holes or rasterizing walls.
    A new open section retains its ID, kind and orientation, NOT its old width.
    Its width and position change are explicitly returned for review.
    """
    def check():
        if cancelled():
            raise InterruptedError('Axis-aligned domain preparation cancelled')
    check()
    assignments = tuple(assignments)
    segments = domain.boundary_segments(assignments)
    origin = np.array([grid.xlower_m, grid.ylower_m])
    spacing = np.array([grid.dx_m, grid.dy_m])
    points = np.array([s.start_xy for s in segments]);original_points=points.copy()
    moved = {}
    sections = {}
    for name in sorted({s.bc_id for s in segments if s.kind != 'wall'}):
        check()
        line, normal = boundary_polyline(domain, assignments, name)
        delta = np.array(line[-1])-line[0]
        tangent = int(np.argmax(abs(delta)))
        cross = 1-tangent
        selected = [i for i,s in enumerate(segments) if s.bc_id == name and s.kind != 'wall']
        vertices = sorted({i for k in selected for i in (k,(k+1)%len(points))})
        centre = .5*(np.array(line[0])+line[-1])
        position = origin[cross]+round((centre[cross]-origin[cross])/spacing[cross])*spacing[cross]
        for i in vertices:
            new = origin+np.rint((points[i]-origin)/spacing)*spacing
            new[cross] = position
            if i in moved and not np.array_equal(moved[i], new):
                raise ValueError('Adjacent open boundaries require conflicting endpoint alignment')
            moved[i] = new
        # The ring can wrap through vertex zero; use extrema, not ring ordering.
        lo = min(float(moved[i][tangent]) for i in vertices)
        hi = max(float(moved[i][tangent]) for i in vertices)
        if hi <= lo:
            raise ValueError(f'Boundary {name} disappears at the selected resolution')
        sections[name] = dict(kind=segments[selected[0]].kind, tangent=tangent,
            cross=cross, position=position, low=lo, high=hi,
            inward_sign=math.copysign(1., normal[cross]),
            original_width_m=float(np.linalg.norm(delta)), aligned_width_m=hi-lo,
            original_coordinates=[list(v) for v in line],
            maximum_endpoint_shift_m=max(float(np.linalg.norm(moved[i]-points[i])) for i in vertices))
    for i, point in moved.items():
        points[i] = point
    # Canonical validation catches self-crossing introduced by rotating sections.
    result = PolygonDomain(domain.domain_id+'_axis',points.tolist(),domain.crs_wkt)
    local = np.array(result.vertices_xy)-origin
    extent = np.array([grid.nx,grid.ny])*spacing
    if np.any(local < -1e-8) or np.any(local > extent+1e-8):
        raise ValueError('Aligned domain extends outside the preparation grid')
    # Canonicalization may rotate the ring, so map by endpoint provenance, not
    # list position or coordinate proximity. Partial assignments become explicit
    # polygon edges. Preserve named walls as well as open-section identities.
    by_endpoints={(e.start_xy,e.end_xy):e for e in result.edges}
    mapped=[]
    for i,s in enumerate(segments):
        check()
        ends=tuple(points[i]),tuple(points[(i+1)%len(points)])
        if ends not in by_endpoints:
            raise ValueError('Alignment reverses the domain; edit the open sections')
        edge=by_endpoints[ends]
        if s.kind!='wall' or s.bc_id!=domain.domain_id+':wall':
            mapped.append(BoundaryAssignment(edge.edge_id,s.bc_id,s.kind))
    for name,s in sections.items():
        line,_=boundary_polyline(result,mapped,name)
        width=float(np.linalg.norm(np.array(line[-1])-line[0]))
        if not math.isclose(width,s['aligned_width_m'],rel_tol=1e-10,abs_tol=1e-8):
            raise ValueError(f'Boundary {name} lost part of its section during alignment')
        if abs(boundary_polyline(result,mapped,name)[1][s['cross']]-s['inward_sign'])>1e-10:
            raise ValueError(f'Boundary {name} reversed during alignment')
        s['aligned_coordinates']=[list(v) for v in line]
    check()
    validate_open_grid(result,mapped,grid)
    before=_polygon((np.array(domain.vertices_xy)-origin).tolist())
    after=_polygon((np.array(result.vertices_xy)-origin).tolist())
    symmetric=before.SymmetricDifference(after).GetArea()
    changes=[dict(vertex=i,original=original_points[i].tolist(),aligned=points[i].tolist())
             for i in moved if not np.array_equal(original_points[i],points[i])]
    wall_changes=[dict(segment=i,bc_id=s.bc_id,original=[original_points[i].tolist(),original_points[(i+1)%len(points)].tolist()],
        aligned=[points[i].tolist(),points[(i+1)%len(points)].tolist()])
        for i,s in enumerate(segments) if s.kind=='wall' and
        (not np.array_equal(original_points[i],points[i]) or
         not np.array_equal(original_points[(i+1)%len(points)],points[(i+1)%len(points)]))]
    audit=dict(method='axis_open_sections_cut_walls.v2',grid=asdict(grid),
        source_domain_id=domain.domain_id,derived_domain_id=result.domain_id,
        original_area_m2=domain.area_m2,derived_area_m2=result.area_m2,
        symmetric_difference_m2=symmetric,sections=sections,
        wall_treatment='polygon_cut_cells',wall_rasterization=False,
        unchanged_wall_segments=sum(s.kind=='wall' for s in segments)-len(wall_changes),
        reconnected_wall_segments=wall_changes,changed_vertices=changes,
        note='Only inlet/outlet vertices aligned. Other wall vertices unchanged; shared wall endpoints reconnect. Cropped wall cells remain permitted. Hydraulic methods unchanged.')
    return result,tuple(mapped),audit
