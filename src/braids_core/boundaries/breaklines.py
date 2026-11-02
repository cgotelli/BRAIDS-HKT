"""Conforming breakline control volumes with an absolute minimum area.

Breaklines are permeable mesh constraints, never walls or bed elevations. Split
the finest Cartesian cells, then union small pieces without swallowing any
constraint. Build parents from these actual children (not independently clipped
rectangles). Both backends consume the resulting shared-face finite volumes.
"""
from dataclasses import replace
import heapq
from itertools import combinations
import math
import numpy as np
from shapely.geometry import Polygon, LineString, Point
from shapely.geometry.polygon import orient
from shapely.ops import unary_union, polygonize
from shapely.strtree import STRtree
from shapely import (constrained_delaunay_triangles,points as geometry_points,
                     linestrings,centroid,get_coordinates,line_interpolate_point,length as geometry_length)
from .cut_cells import CutCell, CutFace, CutCellMesh, map_cut_cells
from .inlet_io import NativeGrid

def validate_breaklines(rows, domain=None):
    if not isinstance(rows, list):raise ValueError('Breaklines must be a list of named geometries')
    names=set()
    for row in rows:
        if not isinstance(row,dict) or set(row)!={'id','closed','points_xy'}:
            raise ValueError('Each breakline needs id, closed and points_xy')
        name=row['id']
        if not isinstance(name,str) or not name.strip() or name in names:
            raise ValueError('Breakline names must be nonempty and unique')
        names.add(name)
        if type(row['closed']) is not bool:raise ValueError(name+': explicit closed switch required')
        points=np.asarray(row['points_xy'],float)
        minimum=3 if row['closed'] else 2
        if points.ndim!=2 or points.shape[1]!=2 or len(points)<minimum or not np.isfinite(points).all():
            raise ValueError(name+': finite coordinates and enough distinct vertices required')
        line=LineString(_points(row))
        if not line.is_simple or line.length<=0 or np.any(np.linalg.norm(np.diff(points,axis=0),axis=1)==0):
            raise ValueError(name+': self-intersections or repeated consecutive vertices are not allowed')
        if row['closed'] and (not Polygon(line).is_valid or Polygon(line).area<=0):
            raise ValueError(name+': a simple polygon outline is required')
        if domain is not None and not line.intersects(Polygon(domain.vertices_xy)):
            raise ValueError(name+': breakline does not intersect the calculation domain')
    return rows


def _points(row):
    points=list(map(tuple,row['points_xy']))
    if row['closed'] and points[0]!=points[-1]:points.append(points[0])
    return points


def _parts(geometry,kind):
    if geometry.is_empty:return
    if geometry.geom_type==kind:yield geometry
    elif hasattr(geometry,'geoms'):
        for part in geometry.geoms:yield from _parts(part,kind)


def _usable(poly,tol):
    if poly.geom_type!='Polygon' or not poly.is_valid or poly.interiors:return False
    vertices=np.asarray(orient(poly,sign=1).exterior.coords)[:-1]
    if not 3<=len(vertices)<=64:return False
    center=np.asarray(poly.centroid.coords[0]);edge=np.roll(vertices,-1,axis=0)-vertices
    # The area centroid must lie inside every inward face half-plane. The
    # native reconstruction and normal distances require this star-shaped CV.
    return bool(np.all(edge[:,0]*(center-vertices)[:,1]-edge[:,1]*(center-vertices)[:,0]>=-tol*np.linalg.norm(edge,axis=1)))


def _clean(poly):
    # Remove exactly collinear vertices; never move a physical/breakline edge.
    return orient(poly.simplify(0,preserve_topology=True),sign=1)


def _merge(polys,faces,barriers,area_floor,tol,stop,keys=None,*,area_roundoff=0.):
    """Deterministic adjacent unions; keys optionally restrict parent bins."""
    floors=np.broadcast_to(np.asarray(area_floor,dtype=float),(len(polys),)).copy()
    variable_floor=bool(len(floors) and np.ptp(floors)>0)
    def minimum(i):return floors[i]-max(floors[i]*1e-11,area_roundoff)
    count=len(polys);parent=list(range(count));groups={i:[i] for i in range(count)}
    shapes=dict(enumerate(polys));adj=[{} for _ in polys];protected=[set() for _ in polys]
    for face in faces:
        if face.neighbour>=0 and face.breakline_ids:
            protected[face.owner].add(face.neighbour);protected[face.neighbour].add(face.owner)
        if face.neighbour>=0 and not face.breakline_ids:
            a,b=face.owner,face.neighbour
            adj[a][b]=adj[a].get(b,0.)+face.length_m;adj[b][a]=adj[b].get(a,0.)+face.length_m
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    queue=[(p.area,i) for i,p in shapes.items() if keys is not None or p.area<minimum(i)]
    heapq.heapify(queue)
    while queue:
        stop();old,i=heapq.heappop(queue)
        if root(i)!=i or shapes[i].area!=old:continue
        if keys is None and old>=minimum(i):continue
        neighbors={}
        for member in groups[i]:
            for j,length in adj[member].items():
                other=root(j)
                if other!=i and (keys is None or keys[i]==keys[other]):neighbors[other]=neighbors.get(other,0.)+length
        candidates=sorted(neighbors,key=lambda j:(-neighbors[j],shapes[j].area,j))
        for j in candidates:
            if any(root(other)==j for member in groups[i] for other in protected[member]):continue
            shape=_clean(shapes[i].union(shapes[j]))
            if not _usable(shape,tol):continue
            # Protected face incidences also forbid swallowing a constraint
            # through an untagged path around its endpoint; no fuzzy re-overlay.
            parent[j]=i;groups[i]+=groups.pop(j);shapes.pop(j);shapes[i]=shape;floors[i]=min(floors[i],floors[j])
            if keys is not None or shape.area<minimum(i):heapq.heappush(queue,(shape.area,i))
            break
    if keys is None:
        # A bend can leave a small piece whose union with ANY single neighbor
        # has its centroid outside the reconstruction kernel. Intermediate
        # binary unions need not be usable when the final connected union is.
        # Repair only stalled pieces, with atomic local unions; never relax the
        # polygon predicate or cross a protected face (including between donors).
        while True:
            changed=False
            for i in sorted(shapes,key=lambda k:(shapes[k].area,k)):
                if i not in shapes or shapes[i].area>=minimum(i):continue
                stop();neighbors={}
                for member in groups[i]:
                    for j,length in adj[member].items():
                        other=root(j)
                        if other!=i:neighbors[other]=neighbors.get(other,0.)+length
                candidates=sorted(neighbors,key=lambda j:(-neighbors[j],shapes[j].area,j))
                accepted=False
                # Local 2:1 transitions can have more incident fragments than
                # a uniform grid. A valid atomic union may need four or more
                # donors even though each intermediate L-shaped union fails.
                for size in range(1,min(6 if variable_floor else 3,len(candidates))+1):
                    for others in combinations(candidates,size):
                        stop();selected={i,*others}
                        if any(root(other) in selected for j in selected for member in groups[j] for other in protected[member]):continue
                        shape=_clean(unary_union([shapes[j] for j in sorted(selected)]))
                        floor=min(floors[j] for j in selected)
                        if shape.area<floor-max(floor*1e-11,area_roundoff) or not _usable(shape,tol):continue
                        for j in others:
                            parent[j]=i;groups[i]+=groups.pop(j);shapes.pop(j)
                        shapes[i]=shape;floors[i]=floor;changed=accepted=True;break
                    if accepted:break
            if not changed:break
    ordered=sorted(shapes,key=lambda i:min(groups[i]));mapping=np.empty(count,np.int64)
    for new,i in enumerate(ordered):mapping[groups[i]]=new
    return [shapes[i] for i in ordered],mapping,[groups[i] for i in ordered]


def _faces(polys,template,lines,names,origin,tol):
    """Node all boundaries once; every interior segment has exactly two owners."""
    boundaries=[p.boundary for p in polys];index=STRtree(boundaries)
    centers=get_coordinates(centroid(polys));boundary=[f for f in template.faces if f.neighbour<0]
    exterior=[LineString(np.asarray([f.start_xy,f.end_xy])-origin) for f in boundary]
    exterior_index=STRtree(exterior);line_index=STRtree(lines)
    noded=unary_union(boundaries);faces=[]
    pairs=[]
    for chain in _parts(noded,'LineString'):
        xy=list(chain.coords);pairs.extend(zip(xy,xy[1:]))
    segments=linestrings(pairs)
    # Bulk exact-distance predicates replace three temporary buffers and
    # repeated Python/GEOS searches per face. Keep noding/order/coordinates.
    candidates={}
    for k,i in exterior_index.query(segments,predicate='dwithin',distance=tol).T:
        candidates.setdefault(int(k),[]).append(int(i))
    split=[]
    for k,(a,b) in enumerate(pairs):
        # Project assignment endpoints onto the retained polygon edge. Unioning
        # a second GIS round-trip of the same oblique wall can create duplicate
        # near-coincident segments; it is not a valid way of noding these edges.
        if k not in candidates:split.append((a,b));continue
        line=segments[k];length=line.length;cuts={0.,length}
        for i in candidates[k]:
            for point in exterior[i].coords:
                p=Point(point);s=line.project(p)
                if tol<s<length-tol and line.distance(p)<=tol:cuts.add(s)
        ordered=sorted(cuts)
        for lo,hi in zip(ordered,ordered[1:]):
            split.append((a if lo==0 else line.interpolate(lo).coords[0],b if hi==length else line.interpolate(hi).coords[0]))
    segments=linestrings(split);lengths=geometry_length(segments)
    middle=line_interpolate_point(segments,.5,normalized=True)
    incidence={};exterior_matches={};constraint_matches={}
    for destination,tree in [(incidence,index),(exterior_matches,exterior_index),(constraint_matches,line_index)]:
        for k,i in tree.query(middle,predicate='dwithin',distance=tol).T:
            destination.setdefault(int(k),[]).append(int(i))
    for k,(start,end) in enumerate(split):
        segment=segments[k];length=lengths[k]
        if length<=tol:continue
        mid=middle[k];owners=sorted(incidence.get(k,[]))
        if len(owners) not in (1,2):raise ValueError('Breakline mesh has an invalid face incidence')
        owner=owners[0];neighbor=owners[1] if len(owners)==2 else -1
        delta=np.asarray(end)-start;normal=np.array([delta[1],-delta[0]])/length
        if np.dot(np.asarray(mid.coords[0])-centers[owner],normal)<0:normal=-normal
        a=tuple(np.asarray(start)+origin);b=tuple(np.asarray(end)+origin)
        face=CutFace(len(faces),owner,neighbor,a,b,tuple(normal),length)
        if neighbor<0:
            matches=exterior_matches.get(k,[])
            if len(matches)!=1:raise ValueError('Breakline mesh lost the exterior boundary identity')
            i=matches[0];old=boundary[i];line=exterior[i]
            lo,hi=sorted((line.project(Point(start)),line.project(Point(end))))
            begin=None if old.edge_start_m is None else old.edge_start_m+lo
            finish=None if old.edge_start_m is None else old.edge_start_m+hi
            face=replace(old,face_id=len(faces),owner=owner,neighbour=-1,start_xy=a,end_xy=b,
                outward_normal=tuple(normal),length_m=length,edge_start_m=begin,edge_end_m=finish)
        matches=constraint_matches.get(k,[])
        if matches:face=replace(face,breakline_ids=tuple(sorted(names[i] for i in matches)))
        faces.append(face)
    return tuple(faces)


def _mesh(polys,faces,template,grid,origin,level,nominal_areas=None):
    cells=[];counts={}
    for number,poly in enumerate(polys):
        # Split/merge already returns clean, CCW polygons. Do not simplify and
        # orient the same geometry again at every hierarchy level.
        center=np.asarray(poly.centroid.coords[0])+origin
        i=math.floor((center[0]-grid.xlower_m)/grid.dx_m);j=math.floor((center[1]-grid.ylower_m)/grid.dy_m)
        component=counts.get((i,j),0);counts[i,j]=component+1
        cells.append(CutCell(number,i,j,component,poly.area,tuple(center),
            tuple(map(tuple,np.asarray(poly.exterior.coords)+origin)),level=level,
            nominal_area_m2=grid.dx_m*grid.dy_m if nominal_areas is None else float(nominal_areas[number])))
    result=CutCellMesh(template.domain_id,template.crs_wkt,grid,tuple(cells),tuple(faces),
        template.obstacle_ids,template.solid_area_m2)
    audit=result.metrics()
    if audit['closure_linf_m']>1e-8*max(grid.dx_m,grid.dy_m,1.):raise ValueError('Breakline face closure failed')
    return result


def _grid_aligned(rows,grid):
    """Exact stored-grid coincidence, not proximity to a physical constraint."""
    origin=(grid.xlower_m,grid.ylower_m);spacing=(grid.dx_m,grid.dy_m)
    for row in rows:
        xy=np.asarray(_points(row))
        if any(np.all(delta!=0) for delta in np.diff(xy,axis=0)):return False
        for p in xy:
            for axis in (0,1):
                k=round((p[axis]-origin[axis])/spacing[axis])
                if p[axis]!=origin[axis]+k*spacing[axis]:return False
    return True


def _tag_grid_faces(faces,lines,names,origin,tol):
    # A grid-coincident line adds labels/protection to existing complete faces.
    # It does not require global polygon boundary union and re-noding.
    middle=np.array([(np.asarray(f.start_xy)+f.end_xy)/2-origin for f in faces])
    matches=STRtree(lines).query(geometry_points(middle),predicate='dwithin',distance=tol)
    tagged={}
    for f,i in matches.T:
        face=faces[int(f)];segment=LineString(np.asarray([face.start_xy,face.end_xy])-origin)
        if lines[int(i)].covers(segment):tagged.setdefault(int(f),set()).add(names[int(i)])
    return tuple(replace(f,breakline_ids=tuple(sorted(tagged[k]))) if k in tagged else f for k,f in enumerate(faces))


def build_breakline_hierarchy(domain,assignments,base_grid,rows,*,levels,obstacles=(),cancelled=None,local_refinement=()):
    from .amr import AMRHierarchy
    validate_breaklines(rows,domain)
    def stop():
        if cancelled and cancelled():raise InterruptedError('Breakline preparation cancelled')
    stop();ratio=2**(levels-1)
    grid=NativeGrid(base_grid.nx*ratio,base_grid.ny*ratio,base_grid.xlower_m,base_grid.ylower_m,base_grid.dx_m/ratio,base_grid.dy_m/ratio)
    origin=np.array([grid.xlower_m,grid.ylower_m]);floor=grid.dx_m*grid.dy_m;tol=1e-10*max(grid.dx_m,grid.dy_m)
    # The template stores map coordinates, then this builder subtracts origin.
    # Decimal grid widths at projected coordinates lose a few coordinate ulps
    # in that round trip. A complete 0.1 m square must not become a sliver.
    # Bound only that representational area error; do not change vertices,
    # stored areas, constraint ownership, or the minimum for resolved cuts.
    extent=origin+np.array([grid.nx*grid.dx_m,grid.ny*grid.dy_m])
    coordinate_ulp=float(np.max(np.abs(np.spacing(np.r_[origin,extent]))))
    area_roundoff=8*(grid.dx_m+grid.dy_m)*coordinate_ulp
    face_tol=max(tol,8*coordinate_ulp)
    domain_shape=Polygon(np.asarray(domain.vertices_xy)-origin)
    if obstacles:
        domain_shape=domain_shape.difference(unary_union([Polygon(np.asarray(o.vertices_xy)-origin) for o in obstacles]))
    def represented_points(row):
        points=np.asarray(_points(row),float).copy()
        # A decimal GIS coordinate and origin+k*dx can differ by a few ulps.
        # Canonicalize only that representational neighbourhood, not a mesh-
        # sized tolerance; resolved subcell constraints remain unchanged.
        for axis,step in enumerate([grid.dx_m,grid.dy_m]):
            aligned=origin[axis]+np.rint((points[:,axis]-origin[axis])/step)*step
            close=np.abs(points[:,axis]-aligned)<=8*coordinate_ulp
            points[close,axis]=aligned[close]
        return points-origin
    lines=[LineString(represented_points(row)).intersection(domain_shape) for row in rows]
    barriers=unary_union(lines);index=STRtree(lines)
    template=map_cut_cells(domain,assignments,grid,obstacles=obstacles,cancelled=cancelled);polys=[]
    from .local_refinement import split_cells
    local_mesh=split_cells(template,local_refinement,origin,stop) if local_refinement else None
    seeds=([Polygon(np.asarray(c.vertices_xy)-origin,[np.asarray(r)-origin for r in c.holes_xy]) for c in template.cells],np.full(len(template.cells),floor)) if local_mesh is None else local_mesh
    nominal=[]
    coincident=_grid_aligned(rows,grid) and local_mesh is None;unchanged_faces=(not rows or coincident) and local_mesh is None
    for poly,cell_floor in zip(*seeds):
        stop();start=len(polys)
        local=[lines[int(i)].intersection(poly) for i in index.query(poly)] if rows and not coincident else []
        if not local:
            poly=_clean(poly)
            if _usable(poly,tol):polys.append(poly)
            else:
                unchanged_faces=False
                polys.extend(_clean(p) for p in constrained_delaunay_triangles(poly).geoms)
            nominal.extend([cell_floor]*(len(polys)-start));continue
        edges=[poly.boundary,*local]
        # Interior endpoints require extra ordinary mesh edges, otherwise GEOS
        # polygonization would drop a dangling constraint. They remain permeable.
        for line in local:
            for part in _parts(line,'LineString'):
                for p in (part.coords[0],part.coords[-1]):
                    if poly.boundary.distance(Point(p))>tol:
                        edges.extend(LineString([p,v]).intersection(poly) for v in poly.exterior.coords[:-1])
        pieces=[p for p in polygonize(unary_union(edges)) if poly.covers(p.representative_point())]
        for piece in pieces:
            piece=_clean(piece)
            if _usable(piece,tol):polys.append(piece)
            else:polys.extend(_clean(p) for p in constrained_delaunay_triangles(piece).geoms)
        nominal.extend([cell_floor]*(len(polys)-start))
    # Noding a decimal line through a stored grid vertex can produce a tiny
    # triangle entirely inside its coordinate-roundoff neighbourhood. It has
    # no resolved edge and cannot become a control volume or a merge barrier.
    keep=[i for i,p in enumerate(polys) if max(p.bounds[2]-p.bounds[0],p.bounds[3]-p.bounds[1])>face_tol]
    polys=[polys[i] for i in keep];nominal=np.asarray(nominal)[keep]
    if not math.isclose(sum(p.area for p in polys),domain_shape.area,rel_tol=1e-10):raise ValueError('Breakline split did not partition the domain')
    # Ordinary cropped walls/obstacles already have a conforming face ledger.
    # Keep it exactly when every cell survived without splitting (cleaning only
    # removes collinear vertices). Re-node only when a breakline or a hole/
    # non-star-shaped cut cell genuinely introduced new sub-cell edges.
    # Shared-face searches must also cover the round trip from stored map
    # coordinates. Sub-ulp intersection stubs are not resolved physical edges.
    faces=template.faces if unchanged_faces else _faces(polys,template,lines,[r['id'] for r in rows],origin,face_tol)
    if rows and coincident and unchanged_faces:faces=_tag_grid_faces(faces,lines,[r['id'] for r in rows],origin,tol)
    polys,mapping,groups=_merge(polys,faces,barriers,nominal,tol,stop,area_roundoff=area_roundoff)
    nominal=np.array([min(nominal[g]) for g in groups])
    faces=tuple(replace(f,face_id=k,owner=int(mapping[f.owner]),neighbour=-1 if f.neighbour<0 else int(mapping[f.neighbour]))
        for k,f in enumerate(f for f in faces if f.neighbour<0 or mapping[f.owner]!=mapping[f.neighbour]))
    small=[(p,a) for p,a in zip(polys,nominal) if p.area<a-max(a*1e-11,area_roundoff)]
    if small:
        p,local_floor=min(small,key=lambda v:v[0].area);xy=np.asarray(p.centroid.coords[0])+origin
        raise ValueError(f'Cannot preserve the walls/breaklines and minimum cell area {local_floor:g} m² near ({xy[0]:.6f}, {xy[1]:.6f}); remaining area {p.area:g} m². Move/widen the nearby constraints or reduce the smallest cell size.')
    meshes=[_mesh(polys,faces,template,grid,origin,levels-1,nominal)];maps=[]
    for level in range(levels-2,-1,-1):
        stop();stride=2**(levels-1-level)
        coarse=NativeGrid(grid.nx//stride,grid.ny//stride,grid.xlower_m,grid.ylower_m,grid.dx_m*stride,grid.dy_m*stride)
        keys=[(math.floor(p.centroid.x/coarse.dx_m),math.floor(p.centroid.y/coarse.dy_m)) for p in polys]
        polys,mapping,_=_merge(polys,faces,barriers,0.,tol,stop,keys=keys)
        faces=tuple(replace(f,face_id=k,owner=int(mapping[f.owner]),neighbour=-1 if f.neighbour<0 else int(mapping[f.neighbour]))
            for k,f in enumerate(f for f in faces if f.neighbour<0 or mapping[f.owner]!=mapping[f.neighbour]))
        meshes.append(_mesh(polys,faces,template,coarse,origin,level));maps.append(mapping)
    meshes=tuple(reversed(meshes));maps=list(reversed(maps));offsets=tuple(np.r_[0,np.cumsum([len(m.cells) for m in meshes])].tolist())
    parent=np.full(offsets[-1],-1,np.int64);children=[[] for _ in parent]
    for level,mapping in enumerate(maps,1):
        for child,p in enumerate(mapping):
            node=offsets[level]+child;ancestor=offsets[level-1]+int(p);parent[node]=ancestor;children[ancestor].append(node)
    node_level=np.concatenate([np.full(len(m.cells),l,np.int64) for l,m in enumerate(meshes)])
    tree=AMRHierarchy(meshes,offsets,parent,tuple(map(tuple,children)),node_level,np.full(len(parent),levels-1,np.int64),2)
    # Exact true-polygon ancestry and area partition, including frozen one-child
    # chains where no admissible coarse union is possible.
    cells=tree.cells
    for node in range(offsets[-2]):
        if not math.isclose(sum(cells[c].area_m2 for c in children[node]),cells[node].area_m2,rel_tol=1e-10):raise ValueError('Breakline hierarchy area mismatch')
    return tree
