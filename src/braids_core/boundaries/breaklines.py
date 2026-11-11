"""Conforming breakline control volumes with an absolute minimum area.

Breaklines are permeable mesh constraints, never walls or bed elevations. Split
the finest Cartesian cells, then union small pieces without swallowing any
constraint. Build parents from these actual children (not independently clipped
rectangles). Both backends consume the resulting shared-face finite volumes.
"""
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import heapq
from itertools import combinations
import math
import numpy as np
from shapely.geometry import Polygon, LineString, Point
from shapely.geometry.polygon import orient
from shapely.ops import unary_union, polygonize
from shapely.strtree import STRtree
from shapely import (area as geometry_area,constrained_delaunay_triangles,points as geometry_points,
                     linestrings,centroid,get_coordinates,get_parts,line_interpolate_point,length as geometry_length)
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


def _usable(poly,tol,*,oriented=False):
    if poly.geom_type!='Polygon' or not poly.is_valid or poly.interiors:return False
    vertices=np.asarray((poly if oriented else orient(poly,sign=1)).exterior.coords)[:-1]
    if not 3<=len(vertices)<=64:return False
    center=np.asarray(poly.centroid.coords[0])
    if len(vertices)==4:
        (x0,y0),(x1,y1),(x2,y2),(x3,y3)=vertices
        horizontal=y0==y1 and x1==x2 and y2==y3 and x3==x0
        vertical=x0==x1 and y1==y2 and x2==x3 and y3==y0
        ccw=(horizontal and ((x1>x0)==(y2>y1))) or (vertical and ((x2>x1)!=(y1>y0)))
        # Valid finite rectangles with a centroid inside their bounds satisfy
        # every original inward half-plane. Keep validity/holes/centroid checks
        # and fall back for CW, degenerate or extreme/unusual representations.
        if (ccw and x0!=x2 and y0!=y2 and math.isfinite(x2-x0) and math.isfinite(y2-y0)
            and min(x0,x2)<=center[0]<=max(x0,x2) and min(y0,y2)<=center[1]<=max(y0,y2)):
            return True
    edge=np.roll(vertices,-1,axis=0)-vertices
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
    # Shapes are immutable between accepted unions. Reuse each exact GEOS
    # area in the heap, ordering and threshold tests; refresh after a union.
    areas=dict(enumerate(map(float,geometry_area(polys))))
    for face in faces:
        if face.neighbour>=0 and face.breakline_ids:
            protected[face.owner].add(face.neighbour);protected[face.neighbour].add(face.owner)
        if face.neighbour>=0 and not face.breakline_ids:
            a,b=face.owner,face.neighbour
            adj[a][b]=adj[a].get(b,0.)+face.length_m;adj[b][a]=adj[b].get(a,0.)+face.length_m
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    queue=[(areas[i],i) for i in shapes if keys is not None or areas[i]<minimum(i)]
    heapq.heapify(queue)
    while queue:
        stop();old,i=heapq.heappop(queue)
        if root(i)!=i or areas[i]!=old:continue
        if keys is None and old>=minimum(i):continue
        neighbors={}
        for member in groups[i]:
            for j,length in adj[member].items():
                other=root(j)
                if other!=i and (keys is None or keys[i]==keys[other]):neighbors[other]=neighbors.get(other,0.)+length
        candidates=sorted(neighbors,key=lambda j:(-neighbors[j],areas[j],j))
        for j in candidates:
            if any(root(other)==j for member in groups[i] for other in protected[member]):continue
            shape=_clean(shapes[i].union(shapes[j]))
            if not _usable(shape,tol,oriented=True):continue
            # Protected face incidences also forbid swallowing a constraint
            # through an untagged path around its endpoint; no fuzzy re-overlay.
            parent[j]=i;groups[i]+=groups.pop(j);shapes.pop(j);shapes[i]=shape;floors[i]=min(floors[i],floors[j])
            areas.pop(j);areas[i]=shape.area
            if keys is not None or areas[i]<minimum(i):heapq.heappush(queue,(areas[i],i))
            break
    if keys is None:
        # A bend can leave a small piece whose union with ANY single neighbor
        # has its centroid outside the reconstruction kernel. Intermediate
        # binary unions need not be usable when the final connected union is.
        # Repair only stalled pieces, with atomic local unions; never relax the
        # polygon predicate or cross a protected face (including between donors).
        while True:
            changed=False
            for i in sorted(shapes,key=lambda k:(areas[k],k)):
                if i not in shapes or areas[i]>=minimum(i):continue
                stop();neighbors={}
                for member in groups[i]:
                    for j,length in adj[member].items():
                        other=root(j)
                        if other!=i:neighbors[other]=neighbors.get(other,0.)+length
                candidates=sorted(neighbors,key=lambda j:(-neighbors[j],areas[j],j))
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
                        area=shape.area
                        if area<floor-max(floor*1e-11,area_roundoff) or not _usable(shape,tol,oriented=True):continue
                        for j in others:
                            parent[j]=i;groups[i]+=groups.pop(j);shapes.pop(j);areas.pop(j)
                        shapes[i]=shape;areas[i]=area;floors[i]=floor;changed=accepted=True;break
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
    if noded.geom_type in ('LineString','MultiLineString'):
        # Preserve GEOS chain and vertex order while avoiding one Python
        # coordinate accessor for each of tens of thousands of chains.
        xy,chain_ids=get_coordinates(get_parts(noded),return_index=True)
        same=chain_ids[:-1]==chain_ids[1:]
        pairs=np.stack((xy[:-1][same],xy[1:][same]),axis=1)
    else:
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
    middle_xy=get_coordinates(middle);ends=np.asarray(split)
    delta=ends[:,1]-ends[:,0]
    normals=np.zeros_like(delta)
    np.divide(np.stack((delta[:,1],-delta[:,0]),axis=1),lengths[:,None],out=normals,where=(lengths>tol)[:,None])
    incidence={};exterior_matches={};constraint_matches={}
    for destination,tree in [(incidence,index),(exterior_matches,exterior_index),(constraint_matches,line_index)]:
        for k,i in tree.query(middle,predicate='dwithin',distance=tol).T:
            destination.setdefault(int(k),[]).append(int(i))
    for k,(start,end) in enumerate(split):
        length=lengths[k]
        if length<=tol:continue
        owners=sorted(incidence.get(k,[]))
        if len(owners) not in (1,2):raise ValueError('Breakline mesh has an invalid face incidence')
        owner=owners[0];neighbor=owners[1] if len(owners)==2 else -1
        normal=normals[k]
        if np.dot(middle_xy[k]-centers[owner],normal)<0:normal=-normal
        a=tuple(np.asarray(start)+origin);b=tuple(np.asarray(end)+origin)
        face=CutFace(len(faces),owner,neighbor,a,b,tuple(normal),length)
        if neighbor<0:
            matches=exterior_matches.get(k,[])
            if len(matches)!=1:raise ValueError('Breakline mesh lost the exterior boundary identity')
            i=matches[0];old=boundary[i];line=exterior[i]
            # An exterior fragment belongs to an already identified physical
            # edge. Keep that edge's normal: differentiating its short rounded
            # GIS endpoints amplifies coordinate roundoff by 1/fragment length
            # and falsely makes a straight section noncollinear. Check the
            # match in physical distance before inheriting the authoritative
            # normal. Interior faces retain their actual local edge normals.
            physical_normal=np.asarray(old.outward_normal)
            if (np.dot(normal,physical_normal)<=0 or
                np.max(np.abs((np.asarray([a,b])-old.start_xy)@physical_normal))>4*tol):
                raise ValueError('Re-noded exterior face leaves its physical boundary edge')
            lo,hi=sorted((line.project(Point(start)),line.project(Point(end))))
            begin=None if old.edge_start_m is None else old.edge_start_m+lo
            finish=None if old.edge_start_m is None else old.edge_start_m+hi
            face=replace(old,face_id=len(faces),owner=owner,neighbour=-1,start_xy=a,end_xy=b,
                outward_normal=old.outward_normal,length_m=length,edge_start_m=begin,edge_end_m=finish)
        matches=constraint_matches.get(k,[])
        if matches:face=replace(face,breakline_ids=tuple(sorted(names[i] for i in matches)))
        faces.append(face)
    return tuple(faces)


def _mesh(polys,faces,template,grid,origin,level,nominal_areas=None):
    cells=[];counts={}
    # GEOS performs the identical per-polygon calculations in its vector API.
    # Keep polygon order, ring starting vertices and map-coordinate arithmetic.
    centers=get_coordinates(centroid(polys))+origin
    areas=geometry_area(polys)
    coordinates,indices=get_coordinates(polys,return_index=True)
    offsets=np.r_[0,np.cumsum(np.bincount(indices,minlength=len(polys)))]
    coordinates+=origin
    for number,poly in enumerate(polys):
        # Split/merge already returns clean, CCW polygons. Do not simplify and
        # orient the same geometry again at every hierarchy level.
        center=centers[number]
        i=math.floor((center[0]-grid.xlower_m)/grid.dx_m);j=math.floor((center[1]-grid.ylower_m)/grid.dy_m)
        component=counts.get((i,j),0);counts[i,j]=component+1
        cells.append(CutCell(number,i,j,component,float(areas[number]),tuple(center),
            tuple(map(tuple,coordinates[offsets[number]:offsets[number+1]])),level=level,
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


def _constraint_hits(index,polys,threads,stop):
    """Read-only exact GEOS queries; concatenate in original seed order."""
    seeds=np.asarray(polys,dtype=object);workers=min(threads,len(seeds))
    if workers<=1:return index.query(seeds,predicate='intersects')
    cuts=np.linspace(0,len(seeds),workers+1,dtype=np.int64)
    def query(pair):
        start,end=pair;hit=index.query(seeds[start:end],predicate='intersects');hit[0]+=start;return hit
    with ThreadPoolExecutor(max_workers=workers) as pool:
        parts=[]
        for part in pool.map(query,zip(cuts[:-1],cuts[1:])):
            stop();parts.append(part)
    return np.concatenate(parts,axis=1)


def build_breakline_hierarchy(domain,assignments,base_grid,rows,*,levels,obstacles=(),cancelled=None,local_refinement=(),threads=4):
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
    simple=not obstacles and local_mesh is None
    flags=None
    if simple:
        from .uniform_geometry import simple_seeds
        ready,flags=simple_seeds(template,origin,threads,stop)
        for i,c in enumerate(template.cells):
            if ready[i] is None:ready[i]=Polygon(np.asarray(c.vertices_xy)-origin,[np.asarray(r)-origin for r in c.holes_xy])
        seeds=ready,np.full(len(template.cells),floor)
    else:
        seeds=([Polygon(np.asarray(c.vertices_xy)-origin,[np.asarray(r)-origin for r in c.holes_xy]) for c in template.cells],np.full(len(template.cells),floor)) if local_mesh is None else local_mesh
    local_flags=None
    if local_mesh is not None:
        from .uniform_geometry import polygon_rectangle_flags
        local_flags=polygon_rectangle_flags(seeds[0],threads)
    nominal=[]
    coincident=_grid_aligned(rows,grid) and local_mesh is None;unchanged_faces=(not rows or coincident) and local_mesh is None
    hits={}
    if rows and not coincident:
        # Bounding-box hits are not intersections. Long oblique polylines can
        # otherwise send most untouched cells through polygonize/unary_union.
        # GEOS tests the exact predicate in a batch before computing any cuts.
        for seed,line in _constraint_hits(index,seeds[0],threads,stop).T:
            hits.setdefault(int(seed),[]).append(int(line))
    for seed_index,(poly,cell_floor) in enumerate(zip(*seeds)):
        stop();start=len(polys)
        # Polygonization must see the original seed ring orientation. The
        # pre-oriented rectangle is only interchangeable after proving that
        # this cell is untouched; GEOS may choose another starting vertex for
        # cuts of a reversed ring, altering deterministic merge tie-breaks.
        if hits.get(seed_index) and simple and flags[seed_index]<0:
            poly=Polygon(np.asarray(template.cells[seed_index].vertices_xy)-origin)
        local=[lines[i].intersection(poly) for i in hits.get(seed_index,())]
        if simple and flags[seed_index] and not local:
            polys.append(poly);nominal.append(cell_floor);continue
        elif local_flags is not None and local_flags[seed_index] and not local:
            # Preserve raw orientation for actual cuts. Untouched proven
            # rectangles need only the same CCW orientation as _clean.
            polys.append(orient(poly,sign=1) if local_flags[seed_index]<0 else poly)
            nominal.append(cell_floor);continue
        elif not local:
            poly=_clean(poly)
            if _usable(poly,tol,oriented=True):polys.append(poly)
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
            if _usable(piece,tol,oriented=True):polys.append(piece)
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
        centers=get_coordinates(centroid(polys))
        keys=[(math.floor(x/coarse.dx_m),math.floor(y/coarse.dy_m)) for x,y in centers]
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
