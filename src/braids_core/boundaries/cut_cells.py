"""Conforming polygon/cell geometry for a future embedded-boundary timestepper.

Preparation only: this mesh is NOT a ``braids_boundary.data`` deck. Areas,
Cartesian apertures and true polygon fragments are retained without snapping,
cell merging or replacing the polygon by a staircase. Disconnected pieces in
one Cartesian cell have separate control volumes. Coordinates are translated
before GEOS operations to retain precision in projected map coordinates.
"""
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path

import numpy as np
from osgeo import ogr

from .domain import (PolygonDomain, BoundaryAssignment, _canonical_ring,
                     _identifier, _polygon)
from .inlet_io import NativeGrid


def _parts(geometry, wanted):
    if geometry is None or geometry.IsEmpty():
        return
    if ogr.GT_Flatten(geometry.GetGeometryType()) == wanted:
        yield geometry
    elif geometry.GetGeometryCount() and ogr.GT_Flatten(geometry.GetGeometryType()) != ogr.wkbPolygon:
        for k in range(geometry.GetGeometryCount()):
            yield from _parts(geometry.GetGeometryRef(k), wanted)


def _line(a, b):
    result = ogr.Geometry(ogr.wkbLineString)
    result.AddPoint_2D(*a)
    result.AddPoint_2D(*b)
    return result


def _ring_center_mask(vertices, grid, origin):
    """Vectorized point-in-ring test at Cartesian cell centres.

    Cells touched by a physical edge are clipped exactly later. This mask is
    used only to recognize rectangles that cannot be cut by any edge.
    """
    ring = np.asarray(vertices, dtype=float)-np.asarray(origin, dtype=float)
    x = (np.arange(grid.nx, dtype=float)+.5)*grid.dx_m
    result = np.zeros((grid.ny, grid.nx), dtype=bool)
    for j in range(grid.ny):
        y = (j+.5)*grid.dy_m
        row = np.zeros(grid.nx, dtype=bool)
        for first, second in zip(ring, ring[1:]):
            if (first[1] > y) == (second[1] > y):
                continue
            crossing = first[0]+(y-first[1])*(second[0]-first[0])/(second[1]-first[1])
            row ^= x < crossing
        result[j] = row
    return result


def _mark_edge_cells(mask, vertices, grid, origin):
    """Mark every background cell traversed by a polygon edge."""
    local = np.asarray(vertices, dtype=float)-np.asarray(origin, dtype=float)
    for a, b in zip(local, local[1:]):
        delta = b-a
        cuts = [0., 1.]
        for axis, spacing, count in ((0, grid.dx_m, grid.nx),
                                     (1, grid.dy_m, grid.ny)):
            if delta[axis] == 0:
                continue
            first = max(1, math.floor(min(a[axis], b[axis])/spacing)+1)
            last = min(count-1, math.ceil(max(a[axis], b[axis])/spacing)-1)
            cuts.extend(t for k in range(first, last+1)
                        if 0 < (t := (k*spacing-a[axis])/delta[axis]) < 1)
        ordered = sorted(set(cuts))
        for lo, hi in zip(ordered, ordered[1:]):
            mid = a+.5*(lo+hi)*delta
            i, j = math.floor(mid[0]/grid.dx_m), math.floor(mid[1]/grid.dy_m)
            if 0 <= i < grid.nx and 0 <= j < grid.ny:
                mask[j, i] = True


@dataclass(frozen=True)
class CutCell:
    cell_id: int                 # zero-based, deterministic j/i/component order
    i: int
    j: int
    component: int
    area_m2: float
    centroid_xy: tuple[float, float]
    vertices_xy: tuple[tuple[float, float], ...]
    member_indices: tuple[tuple[int, int, int], ...] = ()  # original i/j/component after optional agglomeration
    holes_xy: tuple[tuple[tuple[float, float], ...], ...] = ()
    level: int = 0              # active AMR level; fixed-grid cells retain level zero
    hierarchy_node_id: int = -1 # stable ID across regrids, distinct from frame-local cell_id
    nominal_area_m2: float = 0. # adaptive background-cell area, before polygon clipping


@dataclass(frozen=True)
class CutFace:
    face_id: int
    owner: int                  # normal points OUT of owner
    neighbour: int              # -1 for physical polygon boundary
    start_xy: tuple[float, float]
    end_xy: tuple[float, float]
    outward_normal: tuple[float, float]
    length_m: float
    edge_id: str = ""
    segment_id: str = ""
    bc_id: str = ""
    kind: str = "interior"
    edge_start_m: float | None = None
    edge_end_m: float | None = None
    bed_elevation_m: float | None = None  # optional DEM sample ON the boundary, not owner average
    section_bed_start_m: float | None = None  # physical profile endpoints, canonicalized by writer
    section_bed_end_m: float | None = None
    structure_id: str = ""
    boundary_role: str = "external"
    breakline_ids: tuple[str, ...] = ()  # permeable mesh constraints, not boundary conditions


@dataclass(frozen=True)
class SolidObstacle:
    """One fixed impermeable polygon strictly inside the calculation domain."""
    obstacle_id: str
    vertices_xy: tuple[tuple[float, float], ...]

    def __post_init__(self):
        _identifier(self.obstacle_id, "Obstacle ID")
        try:
            vertices = _canonical_ring(self.vertices_xy)
        except ValueError as exc:
            raise ValueError(f"Invalid solid obstacle {self.obstacle_id!r}: {exc}") from exc
        object.__setattr__(self, "vertices_xy", vertices)


def cut_cell_polygon(cell: CutCell, origin=(0., 0.)) -> ogr.Geometry:
    """Return the complete retained cell polygon, including solid holes."""
    shift = np.asarray(origin, dtype=float)
    polygon = ogr.Geometry(ogr.wkbPolygon)
    for vertices in (cell.vertices_xy, *cell.holes_xy):
        ring = ogr.Geometry(ogr.wkbLinearRing)
        for point in vertices:
            x, y = np.asarray(point, dtype=float)-shift
            ring.AddPoint_2D(float(x), float(y))
        polygon.AddGeometry(ring)
    return polygon


@dataclass(frozen=True)
class CutCellMesh:
    domain_id: str
    crs_wkt: str
    grid: NativeGrid
    cells: tuple[CutCell, ...]
    faces: tuple[CutFace, ...]
    obstacle_ids: tuple[str, ...] = ()
    solid_area_m2: float = 0.
    _metrics_cache: dict | None = field(default=None, init=False, repr=False,
                                        compare=False, hash=False)

    def incidence(self):
        """CSR gather arrays, one-based for Fortran; no shared-cell writes.

        Positive sign is outward from the cell. Every internal aperture has
        one flux and two incidences with opposite signs, even for split cells.
        """
        count=len(self.cells)
        owners=np.fromiter((face.owner for face in self.faces),dtype=np.int64,
                           count=len(self.faces))
        neighbours=np.fromiter((face.neighbour for face in self.faces),dtype=np.int64,
                               count=len(self.faces))
        face_ids=np.arange(1,len(self.faces)+1,dtype=np.int64)
        internal=neighbours>=0
        cells=np.concatenate((owners,neighbours[internal]))
        gathered_faces=np.concatenate((face_ids,face_ids[internal]))
        signs=np.concatenate((np.ones(len(owners),dtype=np.int8),
                              -np.ones(int(internal.sum()),dtype=np.int8)))
        order=np.argsort(cells,kind='stable')
        counts=np.bincount(cells,minlength=count)
        offsets=np.empty(count+1,dtype=np.int64);offsets[0]=1
        np.cumsum(counts,out=offsets[1:]);offsets[1:]+=1
        return offsets,gathered_faces[order],signs[order]

    def metrics(self):
        """Discrete divergence-theorem checks, including first moments.

        The closure check is also the geometric requirement for uniform
        hydrostatic pressure to balance on a flat bed. It does not verify a
        variable-bed f-wave source or reconstructed boundary trace.
        """
        if self._metrics_cache is not None:
            return self._metrics_cache.copy()
        count=len(self.cells);face_count=len(self.faces)
        owners=np.fromiter((face.owner for face in self.faces),dtype=np.int64,count=face_count)
        neighbours=np.fromiter((face.neighbour for face in self.faces),dtype=np.int64,count=face_count)
        lengths=np.fromiter((face.length_m for face in self.faces),dtype=float,count=face_count)
        normals=np.fromiter((value for face in self.faces for value in face.outward_normal),
                            dtype=float,count=2*face_count).reshape(face_count,2)
        mids=np.fromiter(((a+b)*.5 for face in self.faces
                          for a,b in zip(face.start_xy,face.end_xy)),
                         dtype=float,count=2*face_count).reshape(face_count,2)
        centers=np.fromiter((value for cell in self.cells for value in cell.centroid_xy),
                            dtype=float,count=2*count).reshape(count,2)
        vector=lengths[:,None]*normals;internal=neighbours>=0
        closure=np.empty((count,2));moments=np.empty((count,2,2))
        for axis in range(2):
            closure[:,axis]=np.bincount(owners,weights=vector[:,axis],minlength=count)
            closure[:,axis]+=np.bincount(neighbours[internal],weights=-vector[internal,axis],
                                         minlength=count)
        for first in range(2):
            owner_delta=mids[:,first]-centers[owners,first]
            neighbour_delta=mids[internal,first]-centers[neighbours[internal],first]
            for second in range(2):
                moments[:,first,second]=np.bincount(
                    owners,weights=owner_delta*vector[:,second],minlength=count)
                moments[:,first,second]+=np.bincount(
                    neighbours[internal],weights=-neighbour_delta*vector[internal,second],
                    minlength=count)
        perimeter=np.bincount(owners,weights=lengths,minlength=count)
        perimeter+=np.bincount(neighbours[internal],weights=lengths[internal],minlength=count)
        areas = np.array([c.area_m2 for c in self.cells])
        error = moments-areas[:, None, None]*np.eye(2)
        result = dict(area_m2=math.fsum(areas), cell_count=len(self.cells), face_count=len(self.faces),
                    boundary_fragment_count=sum(f.neighbour < 0 for f in self.faces),
                    internal_wall_fragment_count=sum(f.boundary_role == "solid_obstacle" for f in self.faces),
                    internal_wall_length_m=math.fsum(f.length_m for f in self.faces
                                                     if f.boundary_role == "solid_obstacle"),
                    obstacle_count=len(self.obstacle_ids), solid_area_m2=self.solid_area_m2,
                    partial_cell_count=int(np.count_nonzero(areas < np.array([
                        c.nominal_area_m2 or self.grid.dx_m*self.grid.dy_m for c in self.cells])*(1-1e-12))),
                    minimum_area_fraction=float(min(areas/np.array([
                        c.nominal_area_m2 or self.grid.dx_m*self.grid.dy_m for c in self.cells]))),
                    closure_linf_m=float(abs(closure).max()),
                    moment_linf_m2=float(abs(error).max()),
                    minimum_area_over_perimeter_m=float(min(areas/perimeter)))
        object.__setattr__(self, "_metrics_cache", result)
        return result.copy()

    def write_geometry(self, path):
        """Inspection/interchange artifact; deliberately separate from solver inputs."""
        path = Path(path)
        if path.suffix == ".npz":
            from .geometry_io import write_packed_geometry
            metadata = dict(semantic_format="braids-cut-cell-geometry", version=2,
                            native_timestep_ready=False, domain_id=self.domain_id,
                            crs_wkt=self.crs_wkt, grid=asdict(self.grid),
                            obstacle_ids=list(self.obstacle_ids), solid_area_m2=self.solid_area_m2,
                            metrics=self.metrics(), members_cover_cell=False)
            return write_packed_geometry(path, cell_count=len(self.cells),
                rows=lambda: iter(self.cells), metadata=metadata,
                physical_faces=self.faces)
        payload = dict(format="braids-cut-cell-geometry", version=1, native_timestep_ready=False,
                       domain_id=self.domain_id, crs_wkt=self.crs_wkt, grid=asdict(self.grid),
                       obstacle_ids=list(self.obstacle_ids), solid_area_m2=self.solid_area_m2,
                       cells=[vars(c) for c in self.cells], faces=[vars(f) for f in self.faces],
                       metrics=self.metrics())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, allow_nan=False, separators=(",", ":"))+"\n")


def map_cut_cells(domain: PolygonDomain, assignments: list[BoundaryAssignment],
                  grid: NativeGrid, obstacles=(), *, progress=None, cancelled=None) -> CutCellMesh:
    """Clip a simple metre-CRS polygon and preserve all positive-area pieces.

    The explicit grid is in the domain CRS and must contain the entire domain.
    No terrain, bed interpolation or hydraulic allocation is inferred here.
    Every physical boundary fragment keeps its edge/segment/logical IDs;
    splitting an inlet does not introduce another Q specification.
    """
    if not isinstance(domain, PolygonDomain) or not isinstance(grid, NativeGrid):
        raise ValueError("PolygonDomain and NativeGrid are required")
    segments = domain.boundary_segments(assignments)
    obstacles = tuple(obstacles)
    if not all(isinstance(value, SolidObstacle) for value in obstacles):
        raise ValueError("Obstacles must be SolidObstacle records")
    if len({value.obstacle_id for value in obstacles}) != len(obstacles):
        raise ValueError("Solid obstacle IDs must be unique")
    origin = np.array([grid.xlower_m, grid.ylower_m])
    local = np.array(domain.vertices_xy)-origin
    width, height = grid.nx*grid.dx_m, grid.ny*grid.dy_m
    if (local < 0).any() or (local[:, 0] > width).any() or (local[:, 1] > height).any():
        raise ValueError("The grid must contain the entire polygon; clipping the domain is forbidden")
    outer_polygon = _polygon(local)
    obstacle_shapes = []
    tolerance = 128*np.finfo(float).eps*max(width,height,1.)
    for obstacle in obstacles:
        shape = _polygon(np.asarray(obstacle.vertices_xy)-origin)
        if not outer_polygon.Contains(shape) or outer_polygon.GetBoundary().Distance(shape) <= tolerance:
            raise ValueError(f"Solid obstacle {obstacle.obstacle_id!r} must lie strictly inside the calculation domain")
        for other, other_shape in obstacle_shapes:
            if shape.Distance(other_shape) <= tolerance:
                raise ValueError(f"Solid obstacles {other.obstacle_id!r} and {obstacle.obstacle_id!r} overlap or touch")
        obstacle_shapes.append((obstacle, shape))
    solid = None
    for _, shape in obstacle_shapes:
        solid = shape.Clone() if solid is None else solid.Union(shape)
    polygon = outer_polygon if solid is None else outer_polygon.Difference(solid)
    # Almost every cell in a large natural-domain hierarchy is a complete
    # Cartesian rectangle. Locate those cells with a scanline mask and reserve
    # GEOS clipping for the few cells actually traversed by a physical edge.
    inside = _ring_center_mask(domain.vertices_xy, grid, origin)
    boundary_cells = np.zeros_like(inside)
    _mark_edge_cells(boundary_cells, domain.vertices_xy, grid, origin)
    for obstacle, _ in obstacle_shapes:
        inside &= ~_ring_center_mask(obstacle.vertices_xy, grid, origin)
        _mark_edge_cells(boundary_cells, obstacle.vertices_xy, grid, origin)
    cells, shapes, full_cells, by_grid = [], [], [], {}
    candidates = np.argwhere(inside | boundary_cells)
    progress_stride = max(1, len(candidates) // 100)
    origin_x, origin_y = origin
    for candidate_index, (j_value, i_value) in enumerate(candidates):
        if cancelled and candidate_index % 4096 == 0 and cancelled():
            raise InterruptedError("BRAIDS mesh preparation cancelled")
        j, i = int(j_value), int(i_value)
        x, y = i * grid.dx_m, j * grid.dy_m
        # Adjacent cells must use the identical floating-point grid plane.
        # x+dx and (i+1)*dx can differ by an ulp for nonbinary spacing,
        # creating gaps/overlaps and losing the shared face in GEOS.
        xnext, ynext = (i + 1) * grid.dx_m, (j + 1) * grid.dy_m
        if not boundary_cells[j, i]:
            cid = len(cells)
            x0, x1 = x + origin_x, xnext + origin_x
            y0, y1 = y + origin_y, ynext + origin_y
            vertices = ((x0, y0), (x0, y1), (x1, y1), (x1, y0), (x0, y0))
            centroid_xy = ((x + xnext) * .5 + origin_x,
                           (y + ynext) * .5 + origin_y)
            cells.append(CutCell(cid, i, j, 0, grid.dx_m * grid.dy_m,
                                 centroid_xy, vertices))
            shapes.append(None)
            full_cells.append(True)
            by_grid.setdefault((i, j), []).append(cid)
        else:
            box = _polygon([(x, y), (xnext, y), (xnext, ynext),
                            (x, ynext), (x, y)])
            pieces = sorted(
                _parts(polygon.Intersection(box), ogr.wkbPolygon),
                key=lambda p: (p.Centroid().GetX(), p.Centroid().GetY(),
                               p.ExportToWkt()),
            )
            for component, piece in enumerate(pieces):
                if piece.GetArea() <= 0:
                    continue
                ring = piece.GetGeometryRef(0)
                centroid = piece.Centroid()
                if not centroid.Within(piece):
                    centroid = piece.PointOnSurface()
                cid = len(cells)
                vertices = tuple(
                    (ring.GetX(k) + origin_x, ring.GetY(k) + origin_y)
                    for k in range(ring.GetPointCount())
                )
                holes = tuple(
                    tuple((inner.GetX(k) + origin_x, inner.GetY(k) + origin_y)
                          for k in range(inner.GetPointCount()))
                    for inner in (piece.GetGeometryRef(index)
                                  for index in range(1, piece.GetGeometryCount()))
                )
                cells.append(CutCell(
                    cid, i, j, component, piece.GetArea(),
                    (centroid.GetX() + origin_x, centroid.GetY() + origin_y),
                    vertices, holes_xy=holes,
                ))
                shapes.append(piece.Clone())
                full_cells.append(
                    not holes and math.isclose(
                        piece.GetArea(), grid.dx_m * grid.dy_m,
                        rel_tol=2e-12,
                        abs_tol=2e-12 * grid.dx_m * grid.dy_m,
                    )
                )
                by_grid.setdefault((i, j), []).append(cid)
        if progress and (candidate_index % progress_stride == 0 or
                         candidate_index + 1 == len(candidates)):
            progress(55 * (candidate_index + 1) / max(1, len(candidates)))
    faces = []

    def cell_shape(cell_id):
        shape = shapes[cell_id]
        if shape is None:
            cell = cells[cell_id]
            x, y = cell.i*grid.dx_m, cell.j*grid.dy_m
            xnext, ynext = (cell.i+1)*grid.dx_m, (cell.j+1)*grid.dy_m
            shape = _polygon([(x,y), (xnext,y), (xnext,ynext),
                              (x,ynext), (x,y)])
            shapes[cell_id] = shape
        return shape

    def append(owner, neighbour, a, b, normal, **labels):
        length = math.hypot(a[0]-b[0],a[1]-b[1])
        if length <= 0:
            raise ValueError("Zero-length cut face")
        faces.append(CutFace(len(faces), owner, neighbour,
                             (a[0]+origin[0],a[1]+origin[1]),
                             (b[0]+origin[0],b[1]+origin[1]),normal,length,**labels))

    # Each internal Cartesian opening is constructed ONCE from both neighbours.
    # Intersections may have several line pieces; do not connect them across a wall.
    grid_rows=list(by_grid.items())
    for row_index,((i,j), owners) in enumerate(grid_rows):
        if cancelled and cancelled():
            raise InterruptedError("BRAIDS mesh preparation cancelled")
        for di,dj,normal in ((1,0,(1.,0.)), (0,1,(0.,1.))):
            for owner in owners:
                for neighbour in by_grid.get((i+di,j+dj), []):
                    if full_cells[owner] and full_cells[neighbour]:
                        if di:
                            x_global=(i+1)*grid.dx_m+origin[0]
                            faces.append(CutFace(
                                len(faces),owner,neighbour,
                                (x_global,(j+1)*grid.dy_m+origin[1]),
                                (x_global,j*grid.dy_m+origin[1]),
                                normal,grid.dy_m))
                        else:
                            y_global=(j+1)*grid.dy_m+origin[1]
                            faces.append(CutFace(
                                len(faces),owner,neighbour,
                                (i*grid.dx_m+origin[0],y_global),
                                ((i+1)*grid.dx_m+origin[0],y_global),
                                normal,grid.dx_m))
                        continue
                    common = cell_shape(owner).Intersection(cell_shape(neighbour))
                    lines = sorted(_parts(common, ogr.wkbLineString), key=lambda p: p.ExportToWkt())
                    for line in lines:
                        for k in range(line.GetPointCount()-1):
                            a,b = line.GetPoint(k)[:2], line.GetPoint(k+1)[:2]
                            if math.dist(a,b) > 0:
                                append(owner, neighbour, a,b,normal)
        if progress and (row_index % max(1,len(grid_rows)//100) == 0 or row_index+1 == len(grid_rows)):
            progress(55+30*(row_index+1)/len(grid_rows))

    # Split physical segments at exact grid intersections AND assignment ends.
    # Midpoint ownership uses the inward side when the edge is grid-coincident.
    # No inward buffer removes narrow cells or changes an opening length.
    edges = {edge.edge_id: edge for edge in domain.edges}
    for segment_index,s in enumerate(segments):
        if cancelled and cancelled():
            raise InterruptedError("BRAIDS mesh preparation cancelled")
        # Interpolate assignment endpoints AFTER translating to local space.
        # Interpolating in Swiss map coordinates first can round a partial
        # segment off its own polygon edge by more than the ownership tolerance.
        # Full-edge endpoints retain their original arithmetic and geometry.
        edge = edges[s.edge_id]
        start, end = np.array(edge.start_xy)-origin, np.array(edge.end_xy)-origin
        a = start if s.start_m == 0 else start+(s.start_m/edge.length_m)*(end-start)
        b = end if s.end_m == edge.length_m else start+(s.end_m/edge.length_m)*(end-start)
        delta = b-a
        cuts = [0.,1.]
        for axis,spacing,count in ((0,grid.dx_m,grid.nx), (1,grid.dy_m,grid.ny)):
            if delta[axis] != 0:
                first=max(1, math.floor(min(a[axis],b[axis])/spacing)+1)
                last=min(count-1, math.ceil(max(a[axis],b[axis])/spacing)-1)
                cuts.extend(t for k in range(first,last+1) if 0 < (t:=(k*spacing-a[axis])/delta[axis]) < 1)
        # Mathematically coincident corner intersections can differ by one ulp.
        # Unresolvable fragments trigger a diagnostic, never an area deletion.
        cuts = sorted(set(cuts))
        for lo,hi in zip(cuts,cuts[1:]):
            p,q = a+lo*delta, a+hi*delta
            if math.dist(p,q) == 0:
                continue
            mid = (p+q)/2
            indices=[]
            for axis,spacing in ((0,grid.dx_m),(1,grid.dy_m)):
                coordinate = mid[axis]/spacing
                if delta[axis] == 0 and coordinate == round(coordinate):
                    index = round(coordinate)-(1 if s.inward_normal[axis] < 0 else 0)
                else:
                    index = math.floor(coordinate)
                indices.append(index)
            candidates = by_grid.get(tuple(indices), [])
            point = ogr.Geometry(ogr.wkbPoint); point.AddPoint_2D(*mid)
            owners = [c for c in candidates if cell_shape(c).Distance(point) <= tolerance]
            if len(owners) != 1:
                raise ValueError(f"Ambiguous/unresolvable boundary fragment at {tuple(mid+origin)}; refine or shift grid")
            append(owners[0], -1, p,q,tuple(-v for v in s.inward_normal), edge_id=s.edge_id,
                   segment_id=s.segment_id, bc_id=s.bc_id, kind=s.kind,
                   edge_start_m=s.start_m+lo*s.length_m, edge_end_m=s.start_m+hi*s.length_m,
                   boundary_role="external")
        if progress:
            progress(85+10*(segment_index+1)/max(1,len(segments)))

    # Split every obstacle perimeter at background-grid intersections. The
    # normal points out of the retained water and into the solid polygon.
    for obstacle, _ in obstacle_shapes:
        if cancelled and cancelled():
            raise InterruptedError("BRAIDS mesh preparation cancelled")
        for edge_index, (start_global, end_global) in enumerate(
                zip(obstacle.vertices_xy, obstacle.vertices_xy[1:])):
            a, b = np.asarray(start_global)-origin, np.asarray(end_global)-origin
            delta = b-a;length = float(np.linalg.norm(delta))
            normal = np.array([-delta[1], delta[0]])/length  # obstacle is canonical CCW
            cuts = [0., 1.]
            for axis, spacing, count in ((0, grid.dx_m, grid.nx), (1, grid.dy_m, grid.ny)):
                if delta[axis] != 0:
                    first=max(1, math.floor(min(a[axis],b[axis])/spacing)+1)
                    last=min(count-1, math.ceil(max(a[axis],b[axis])/spacing)-1)
                    cuts.extend(t for k in range(first,last+1)
                                if 0 < (t := (k*spacing-a[axis])/delta[axis]) < 1)
            for lo, hi in zip(sorted(set(cuts)), sorted(set(cuts))[1:]):
                p, q = a+lo*delta, a+hi*delta
                if math.dist(p, q) == 0:
                    continue
                mid = .5*(p+q)
                probe = mid-normal*max(1024*tolerance, 1.e-9*min(grid.dx_m, grid.dy_m))
                indices=(min(grid.nx-1,max(0,math.floor(probe[0]/grid.dx_m))),
                         min(grid.ny-1,max(0,math.floor(probe[1]/grid.dy_m))))
                point=ogr.Geometry(ogr.wkbPoint);point.AddPoint_2D(*probe)
                owners=[candidate for candidate in by_grid.get(indices,())
                        if cell_shape(candidate).Contains(point) or
                        cell_shape(candidate).Distance(point)<=tolerance]
                if len(owners)!=1:
                    raise ValueError(f"Ambiguous solid-obstacle fragment at {tuple(mid+origin)}; refine or shift grid")
                edge_id=f"{obstacle.obstacle_id}:edge:{edge_index}"
                append(owners[0],-1,p,q,tuple(normal),edge_id=edge_id,
                       segment_id=edge_id,kind="wall",structure_id=obstacle.obstacle_id,
                       boundary_role="solid_obstacle")
    solid_area=math.fsum(shape.GetArea() for _,shape in obstacle_shapes)
    mesh = CutCellMesh(domain.domain_id, domain.crs_wkt, grid, tuple(cells), tuple(faces),
                       tuple(value.obstacle_id for value in obstacles),solid_area)
    metrics = mesh.metrics()
    if not math.isclose(metrics['area_m2'], polygon.GetArea(), rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("Cut-cell areas do not partition the polygon")
    # Input vertices and physical boundary fragments are represented in global
    # FP64 coordinates. At a million-metre origin their endpoint uncertainty
    # exceeds a tolerance based only on the local grid width. This guard must
    # admit coordinate roundoff without changing any face, area or ownership.
    coordinate_scale=max(1.,float(np.max(np.abs(np.asarray(domain.vertices_xy)))),
                         *(float(np.max(np.abs(np.asarray(o.vertices_xy)))) for o in obstacles))
    closure_limit=max(32*tolerance,8*float(np.spacing(coordinate_scale)))
    if metrics['closure_linf_m'] > closure_limit:
        raise ValueError(f"Cut-cell face closure failed; incomplete or duplicate ownership (residual {metrics['closure_linf_m']:.6g} m, limit {closure_limit:.6g} m)")
    if progress:
        progress(100.)
    return mesh
