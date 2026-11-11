"""Display-only interpolation and frozen-field particles; no solver mutations."""
import numpy as np
from collections import OrderedDict
from scipy.spatial import cKDTree,Delaunay
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import QhullError
import shapely
from PySide6.QtCore import QObject,QTimer,Qt,QRectF,QPointF
from PySide6.QtGui import QImage,QPainterPath,QPen,QColor,QPainter,QPolygonF
from PySide6.QtWidgets import QGraphicsItem


def wet_mask(frame):return frame.q[0]>1e-6


_mesh_display_cache=OrderedDict()
_plane_stencil_cache=OrderedDict()
_sparse_stencil_cache=OrderedDict()


def _display_geometry(frame):
    """Small geometry-only cache; AMR epochs never share changed polygons."""
    key=id(frame.polygons)
    cached=_mesh_display_cache.get(key)
    if cached is not None and cached[0] is frame.polygons:
        _mesh_display_cache.move_to_end(key);return cached[1:]
    polygons=np.array(frame.polygons,dtype=object)
    nodes=shapely.centroid(polygons)
    outside=~shapely.covers(polygons,nodes)
    # A merged concave cut cell may have its geometric centroid in a solid.
    # Display nodes must belong to fluid; no solver centre/state is changed.
    nodes[outside]=shapely.point_on_surface(polygons[outside])
    centers=np.column_stack((shapely.get_x(nodes),shapely.get_y(nodes)))
    bounds=shapely.bounds(polygons);diagonal=np.hypot(bounds[:,2]-bounds[:,0],bounds[:,3]-bounds[:,1])
    # ~0.35 cell widths for squares; thinner cropped polygons get a smaller
    # radius so a long/slender wet cell is not erased by its large area alone.
    radius=.5*shapely.area(polygons)/np.maximum(diagonal,1e-30)
    domain=shapely.union_all(polygons)
    _mesh_display_cache[key]=(frame.polygons,polygons,centers,radius,domain)
    while len(_mesh_display_cache)>3:_mesh_display_cache.popitem(last=False)
    return polygons,centers,radius,domain


def _plane_stencil(polygons, centres, domain):
    """Cache topology-only neighbours for terrain-display planes.

    Reconstructing bed slopes used to perform a GEOS line test for every cell
    on every frame.  The mesh topology does not change within an AMR epoch, so
    retain the visible nearest-neighbour stencil and recompute only the cheap
    numeric least-squares solve when the bed changes.
    """
    key=id(polygons);cached=_plane_stencil_cache.get(key)
    if cached is not None and cached[0] is polygons:
        _plane_stencil_cache.move_to_end(key);return cached[1:]
    count=len(centres);tree=cKDTree(centres);_,ids=tree.query(centres,k=min(9,count))
    ids=np.asarray(ids);ids=ids[:,None] if ids.ndim==1 else ids
    # First entry is the cell itself.  It has zero distance and is omitted.
    ids=ids[:,1:];delta=centres[ids]-centres[:,None,:]
    if ids.shape[1]:
        starts=np.broadcast_to(centres[:,None,:],delta.shape)
        visible=np.asarray(shapely.covers(domain,shapely.linestrings(np.stack((starts,centres[ids]),axis=2))),bool)
    else:visible=np.zeros(ids.shape,bool)
    _plane_stencil_cache[key]=(polygons,ids,delta,visible)
    while len(_plane_stencil_cache)>3:_plane_stencil_cache.popitem(last=False)
    return ids,delta,visible


def _sparse_stencil(frame):
    """Neighbour geometry for the shoreline-only renderer.

    This intentionally does not construct line-of-sight tests for every cell;
    those are evaluated only for the small set of possible shoreline cells.
    """
    key=id(frame.polygons);cached=_sparse_stencil_cache.get(key)
    if cached is not None and cached[0] is frame.polygons:
        _sparse_stencil_cache.move_to_end(key);return cached[1:]
    polygons=np.array(frame.polygons,dtype=object);nodes=shapely.centroid(polygons)
    outside=~shapely.covers(polygons,nodes)
    nodes[outside]=shapely.point_on_surface(polygons[outside])
    centres=np.column_stack((shapely.get_x(nodes),shapely.get_y(nodes)))
    count=len(centres);_,ids=cKDTree(centres).query(centres,k=min(9,count))
    ids=np.asarray(ids);ids=ids[:,None] if ids.ndim==1 else ids
    ids=ids[:,1:];delta=centres[ids]-centres[:,None,:]
    domain=shapely.union_all(polygons)
    _sparse_stencil_cache[key]=(frame.polygons,polygons,centres,ids,delta,domain)
    while len(_sparse_stencil_cache)>3:_sparse_stencil_cache.popitem(last=False)
    return polygons,centres,ids,delta,domain


def _convex_ring(polygon):
    """Return one convex exterior ring, or ``None`` for a safe flat fallback.

    This is deliberately a display helper.  It never changes a solver cell,
    and avoids inventing a planar interior for a merged concave/holed cell.
    """
    if polygon.is_empty or polygon.geom_type != 'Polygon' or len(polygon.interiors):
        return None
    xy=np.asarray(polygon.exterior.coords[:-1],float)
    if len(xy)<3:return None
    turn=np.cross(np.roll(xy,-1,axis=0)-xy,np.roll(xy,-2,axis=0)-np.roll(xy,-1,axis=0))
    good=turn[np.abs(turn)>1e-12]
    if len(good) and not (np.all(good>=0.) or np.all(good<=0.)):return None
    return xy


def _clip_positive_triangle(xy,depth):
    """Clip a linear-depth triangle to ``depth > 0`` and return its volume.

    On the retained polygon, depth remains linear; its integral is therefore
    polygon area times the mean vertex depth.  This makes the reconstruction
    exactly volume-conserving for every convex display cell.
    """
    vertices=[]
    for i in range(3):
        a,b=xy[i],xy[(i+1)%3];da,db=float(depth[i]),float(depth[(i+1)%3])
        if da>0.:vertices.append((a,da))
        if (da>0.) != (db>0.):
            fraction=da/(da-db);vertices.append((a+fraction*(b-a),0.))
    if len(vertices)<3:return None,0.
    points=np.asarray([v[0] for v in vertices],float);values=np.asarray([v[1] for v in vertices],float)
    shape=shapely.Polygon(points)
    if shape.is_empty or not shape.is_valid or shape.area<=0.:return None,0.
    return shape,float(shape.area*values.mean())


def _plane_volume_and_geometry(ring, bed, eta, centre):
    """Wet volume/geometry for a convex polygon and an inclined planar bed."""
    triangles=[];volume=0.
    for i in range(len(ring)):
        xy=np.vstack((centre,ring[i],ring[(i+1)%len(ring)]))
        depth=eta-(bed[0]+(xy-centre)@bed[1:])
        piece,part=_clip_positive_triangle(xy,depth)
        if piece is not None:triangles.append(piece);volume+=part
    return volume,(shapely.union_all(triangles) if triangles else shapely.GeometryCollection())


def _convex_plane_volume(ring, bed, eta, centre, *, geometry=False):
    """Fast exact wet volume of one convex planar display cell.

    Unlike the general triangulated helper, this avoids GEOS during the root
    iterations.  A clipped convex polygon retains a linear depth field, whose
    integral is its area times the mean vertex depth.
    """
    depth=eta-(bed[0]+(ring-centre)@bed[1:]);points=[];values=[]
    for i,a in enumerate(ring):
        b=ring[(i+1)%len(ring)];da=float(depth[i]);db=float(depth[(i+1)%len(ring)])
        if da>0.:points.append(a);values.append(da)
        if (da>0.) != (db>0.):
            fraction=da/(da-db);points.append(a+fraction*(b-a));values.append(0.)
    if len(points)<3:return 0.,(shapely.GeometryCollection() if geometry else None)
    xy=np.asarray(points,float);values=np.asarray(values,float)
    vector=xy[1:-1]-xy[0];following=xy[2:]-xy[0]
    area=.5*np.abs(vector[:,0]*following[:,1]-vector[:,1]*following[:,0])
    if not len(area) or not np.any(area>0.):return 0.,(shapely.GeometryCollection() if geometry else None)
    # Fan triangulation is exact for a linear depth field.  A simple mean of
    # polygon-vertex depths is not exact after clipping a non-triangular cell.
    volume=float(np.sum(area*(values[0]+values[1:-1]+values[2:])/3.))
    return volume,(shapely.Polygon(xy) if geometry else None)


def _bed_plane_gradients(polygons, centres, bed, domain):
    """Bounded least-squares DEM planes for *display only*.

    The plane intercept is the stored cell bed at the polygon centroid, so a
    fully wet cell preserves its cell-average volume.  Neighbours separated by
    a domain/solid wall are excluded.  These gradients are independent of the
    hydraulic reconstruction and cannot enter the numerical solve.
    """
    count=len(polygons);result=np.zeros((count,2));ids,delta,visible=_plane_stencil(polygons,centres,domain)
    if not ids.shape[1]:return result
    distance2=np.einsum('ijk,ijk->ij',delta,delta);weight=np.where(visible,1./np.maximum(distance2,1e-18),0.)
    difference=np.asarray(bed)[ids]-np.asarray(bed)[:,None]
    normal=np.einsum('ij,ijk,ijl->ikl',weight,delta,delta)
    right=np.einsum('ij,ijk,ij->ik',weight,delta,difference)
    # Per-cell pseudo-inverse retains a one-dimensional channel slope without
    # introducing a false transverse slope.
    result=np.einsum('ijk,ik->ij',np.linalg.pinv(normal,rcond=1e-12),right)
    span=np.max(np.where(visible,np.abs(difference),0.),axis=1)
    extent=np.sqrt(np.max(np.where(visible,distance2,0.),axis=1))
    magnitude=np.linalg.norm(result,axis=1);limit=2.*np.maximum(span,1e-12)/np.maximum(extent,1e-12)
    scale=np.minimum(1.,limit/np.maximum(magnitude,1e-30));result*=scale[:,None]
    return result


def terrain_wet_polygons(frame, *, dry=1e-6):
    """Return one volume-conserving display polygon per solver cell.

    The function reconstructs a planar bed only for rendering.  For each wet
    convex cell it finds one local water level whose clipped-planar volume is
    exactly ``h * cell_area``.  A dry, concave or holed cell is never turned
    into a newly wet patch.  Consequently this improves shoreline appearance
    without changing water, momentum, sediment, CFL or any saved result.
    """
    polygons,centres,_,domain=_display_geometry(frame)
    h=np.asarray(frame.q[0],float);bed=np.asarray(frame.z,float)
    gradients=_bed_plane_gradients(polygons,centres,bed,domain)
    # Most ordinary river cells are fully submerged.  Avoid polygon clipping
    # and 48 bisection iterations for them; only shallow cells close to their
    # reconstructed terrain range need the expensive display treatment.
    bounds=shapely.bounds(polygons);diagonal=np.hypot(bounds[:,2]-bounds[:,0],bounds[:,3]-bounds[:,1])
    relief=.5*np.linalg.norm(gradients,axis=1)*diagonal
    full=(h>dry)&(h>=relief+dry)
    display=np.empty(len(polygons),dtype=object)
    empty=shapely.GeometryCollection()
    display[:]=empty;display[full]=polygons[full]
    fractions=np.zeros(len(polygons));fractions[full]=1.
    for i,(polygon,centre) in enumerate(zip(polygons,centres)):
        if full[i] or not np.isfinite(h[i]) or h[i]<=dry:continue
        ring=_convex_ring(polygon)
        if ring is None:
            display[i]=polygon;fractions[i]=1.;continue
        local=bed[i]+(ring-centre)@gradients[i]
        target=float(h[i]*polygon.area)
        lo=float(np.min(local)-max(h[i],1.));hi=float(np.max(local)+max(h[i],1.))
        # Monotonic bisection of the exact clipped-planar volume.
        for _ in range(48):
            middle=.5*(lo+hi);volume,_=_plane_volume_and_geometry(ring,(bed[i],*gradients[i]),middle,centre)
            if volume<target:lo=middle
            else:hi=middle
        volume,shape=_plane_volume_and_geometry(ring,(bed[i],*gradients[i]),hi,centre)
        if not shape.is_empty:
            display[i]=shape;fractions[i]=min(1.,max(0.,shape.area/polygon.area))
    return tuple(display),fractions


def terrain_wet_edges(frame, *, dry=1e-6, iterations=20):
    """Return only the terrain-clipped wet/dry cell fragments for display.

    The normal finite-volume polygons remain the fast base layer.  This sparse
    companion is deliberately limited to cells at a possible shoreline: the
    raw polygon for such a cell can be hidden and its volume-conserving wedge
    drawn above it.  It gives the terrain-aware outline without rebuilding
    thousands of unchanged interior paths for every live frame.
    """
    polygons,centres,ids,delta,domain=_sparse_stencil(frame)
    h=np.asarray(frame.q[0],float);bed=np.asarray(frame.z,float)
    if not ids.shape[1]:return np.empty(0,np.int32),()
    difference=bed[ids]-bed[:,None]
    span=np.max(np.abs(difference),axis=1)
    # This intentionally generous screen only avoids fitting planes in cells
    # that are unquestionably deep. Cropped cells can be much longer than a
    # nearest-neighbour spacing, so the exact relief test below remains the
    # decision maker.
    rows=np.flatnonzero((h>dry)&(h<8.*span+dry))
    if not len(rows):return np.empty(0,np.int32),()
    local_delta=delta[rows];local_ids=ids[rows]
    # Join only candidate-to-neighbour paths. The union is still the exact
    # fluid domain, so a bank or solid cannot donate an artificial slope.
    starts=np.broadcast_to(centres[rows,None,:],local_delta.shape)
    local_visible=np.asarray(shapely.covers(domain,shapely.linestrings(np.stack((starts,centres[local_ids]),axis=2))),bool)
    distance2=np.einsum('ijk,ijk->ij',local_delta,local_delta)
    weight=np.where(local_visible,1./np.maximum(distance2,1e-18),0.)
    local_difference=bed[local_ids]-bed[rows,None]
    normal=np.einsum('ij,ijk,ijl->ikl',weight,local_delta,local_delta)
    right=np.einsum('ij,ijk,ij->ik',weight,local_delta,local_difference)
    gradients=np.einsum('ijk,ik->ij',np.linalg.pinv(normal,rcond=1e-12),right)
    local_span=np.max(np.where(local_visible,np.abs(local_difference),0.),axis=1)
    extent=np.sqrt(np.max(np.where(local_visible,distance2,0.),axis=1))
    magnitude=np.linalg.norm(gradients,axis=1);limit=2.*np.maximum(local_span,1e-12)/np.maximum(extent,1e-12)
    gradients*=np.minimum(1.,limit/np.maximum(magnitude,1e-30))[:,None]
    bounds=shapely.bounds(polygons[rows]);diagonal=np.hypot(bounds[:,2]-bounds[:,0],bounds[:,3]-bounds[:,1])
    partial=rows[h[rows]<.5*np.linalg.norm(gradients,axis=1)*diagonal+dry]
    if not len(partial):return np.empty(0,np.int32),()
    # Preserve the matching local gradient after the exact relief screen.
    gradients=gradients[np.isin(rows,partial)]
    kept=[];shapes=[]
    for row,gradient in zip(partial,gradients):
        ring=_convex_ring(polygons[row])
        if ring is None:continue
        lo=float(np.min(bed[row]+(ring-centres[row])@gradient)-max(h[row],1.))
        hi=float(np.max(bed[row]+(ring-centres[row])@gradient)+max(h[row],1.))
        target=float(h[row]*polygons[row].area)
        plane=(bed[row],*gradient)
        for _ in range(iterations):
            middle=.5*(lo+hi);volume,_=_convex_plane_volume(ring,plane,middle,centres[row])
            if volume<target:lo=middle
            else:hi=middle
        _,shape=_convex_plane_volume(ring,plane,hi,centres[row],geometry=True)
        if not shape.is_empty and shape.area<polygons[row].area*(1.-1e-10):
            kept.append(row);shapes.append(shape)
    return np.asarray(kept,np.int32),tuple(shapes)


def terrain_wet_geometry(frame, *, dry=1e-6):
    """Return the union of :func:`terrain_wet_polygons` for raster clients."""
    display,fractions=terrain_wet_polygons(frame,dry=dry)
    pieces=[shape for shape in display if not shape.is_empty]
    if not pieces:return shapely.GeometryCollection(),fractions
    # Cells/fragments are non-overlapping by construction.  Coverage union is
    # dramatically cheaper than a general overlay on large visible meshes.
    try:return shapely.coverage_union_all(pieces),fractions
    except Exception:return shapely.union_all(pieces),fractions


def _straight_ring(ring,tree,radius,wall,factor):
    """Join nearby wet tips across dry notches; pin physical-wall vertices.

    Only wet/dry edges may move. In particular the actual polygon/ellipse
    solid boundary is never simplified into a shortcut across the obstacle.
    Rings must have fluid on their left (CCW exterior, CW holes). A concave
    corner can be bridged, but a wet protrusion is not chopped off. Retain all
    original nodes in each error test so successive bridges cannot creep
    arbitrarily far from the original shoreline.
    """
    xy=np.asarray(ring.coords)[:-1];n=len(xy)
    if n<=3:return np.vstack((xy,xy[0]))
    _,near=tree.query(xy);tolerance=factor*2.25*radius[near]
    eps=max(1e-10,16*np.spacing(max(1.,float(np.abs(xy).max()))))
    pinned=shapely.distance(shapely.points(xy),wall)<=eps
    previous=(np.arange(n)-1)%n;following=(np.arange(n)+1)%n;alive=np.ones(n,bool);count=n
    from collections import deque
    pending=deque(range(n))
    while pending and count>3:
        i=pending.popleft()
        if not alive[i] or pinned[i]:continue
        a=previous[i];b=following[i];u=xy[i]-xy[a];v=xy[b]-xy[i]
        cross=u[0]*v[1]-u[1]*v[0]
        if cross>eps*max(np.linalg.norm(u),np.linalg.norm(v),eps):continue
        indices=np.arange(a+1,b if b>a else b+n)%n
        vector=xy[b]-xy[a]
        t=np.clip((xy[indices]-xy[a])@vector/max(float(vector@vector),1e-30),0.,1.)
        distance=np.linalg.norm(xy[indices]-(xy[a]+t[:,None]*vector),axis=1)
        if np.any(distance>tolerance[indices]+eps):continue
        alive[i]=False;count-=1;following[a]=b;previous[b]=a;pending.extend((a,b))
    result=xy[alive]
    return np.vstack((result,result[0]))


def _straight_shoreline(geometry,domain,centers,radius,strength=1.):
    """Piecewise-straight display outline, not a new computed flood extent.

    Topology and wet cell centres guard narrow channels and isolated patches.
    If a local shortcut is invalid, reduce its distance or retain that outline.
    Physical domain/solid walls remain exact, including at partial wetting.
    """
    oriented=shapely.orient_polygons(geometry)
    pieces=list(oriented.geoms) if oriented.geom_type=='MultiPolygon' else [oriented]
    tree=cKDTree(centers);wall=domain.boundary;result=[]
    for piece in pieces:
        cells=shapely.contains_xy(piece,centers[:,0],centers[:,1])
        selected=piece
        for factor in (strength,.5*strength,.25*strength):
            exterior=_straight_ring(piece.exterior,tree,radius,wall,factor)
            holes=[_straight_ring(ring,tree,radius,wall,factor) for ring in piece.interiors]
            candidate=shapely.Polygon(exterior,holes)
            if candidate.is_empty or not candidate.is_valid:continue
            candidate=candidate.intersection(domain)
            if candidate.geom_type!='Polygon' or len(candidate.interiors)!=len(piece.interiors):continue
            if not np.all(shapely.intersects_xy(candidate,centers[cells,0],centers[cells,1])):continue
            selected=candidate;break
        result.append(selected)
    combined=shapely.MultiPolygon(result) if len(result)>1 else result[0]
    # Do not merge disconnected puddles via competing display shortcuts.
    return combined if combined.is_valid else geometry


def _shoreline_coverage(geometry,domain,points,dx,dy):
    """Subpixel edge antialiasing only; no Gaussian mask or broad fade."""
    alpha=np.zeros(len(points))
    shapely.prepare(geometry);shapely.prepare(domain)
    for sx,sy in ((-.25,-.25),(.25,-.25),(-.25,.25),(.25,.25)):
        alpha+=.25*shapely.contains_xy(geometry,points[:,0]+sx*dx,points[:,1]+sy*dy)
    # Pixel centres inside a solid/outside the domain are always transparent.
    alpha[~shapely.contains_xy(domain,points[:,0],points[:,1])]=0.
    return alpha


def _linear_values(xy,data,sample,domain,polygons):
    """Bounded linear color facets with a continuous in-component fallback."""
    interpolated=np.full(len(sample),np.nan)
    if len(xy)>=3:
        try:
            tri=Delaunay(xy);simplex=tri.find_simplex(sample)
            used=np.unique(simplex[simplex>=0]);valid=np.zeros(len(tri.simplices),bool)
            # A Delaunay triangle can bridge a solid/concavity. Such triangles
            # are not allowed to mix opposite-side velocities or depths.
            valid[used]=shapely.covers(domain,shapely.polygons(xy[tri.simplices[used]]))
            good=(simplex>=0)&valid[np.maximum(simplex,0)]
            interpolated[good]=LinearNDInterpolator(tri,data)(sample[good])
        except QhullError:pass  # One-cell-wide/collinear components.
    outside=~np.isfinite(interpolated)
    if np.any(outside):
        query=sample[outside];tree=cKDTree(xy);_,nearest=tree.query(query)
        # Along walls/outside the centre hull retain the containing wet cell,
        # not a value from a geometrically close cell across a thin obstacle.
        matches=shapely.STRtree(polygons).query(shapely.points(query),predicate='within')
        nearest[matches[0]]=matches[1]
        owned=np.zeros(len(query),bool);owned[matches[0]]=True
        valid=np.ones(len(query),bool)
        unknown=np.flatnonzero(~owned)
        if len(unknown):
            # Newly filled shoreline notches have no original containing cell.
            # Select a visible local wet node, never one across a solid wall.
            _,choices=tree.query(query[unknown],k=min(8,len(xy)))
            if choices.ndim==1:choices=choices[:,None]
            valid[unknown]=False
            for column in range(choices.shape[1]):
                rows=unknown[~valid[unknown]]
                if not len(rows):break
                ids=choices[~valid[unknown],column]
                visible=shapely.covers(domain,shapely.linestrings(np.stack((query[rows],xy[ids]),axis=1)))
                nearest[rows[visible]]=ids[visible];valid[rows[visible]]=True
        # Every query passed here lies inside this wet connected component.
        # Retain the nearest component value as the last fallback, instead of
        # creating transparent pinholes at polygon boundaries or short
        # visibility-test degeneracies.
        interpolated[outside]=data[nearest]
    return np.clip(interpolated,float(data.min()),float(data.max()))


def raster_field(frame,values,variable,limits,to_scene,*,maximum=1000,palette=None,smooth_level=100,
                 terrain_aware_wetting=False,zero_transparent=True,domain_geometry=None):
    from .results import colour_values
    strength=max(0.,min(1.,float(smooth_level)/100.))
    hydraulic=variable not in ('bed_m','bed_change_m')
    use=np.isfinite(values)
    if hydraulic:use&=wet_mask(frame)
    if not np.any(use):return None
    polygons,centers,radius,domain=_display_geometry(frame)
    geometry=shapely.union_all(polygons[use])
    # The retained mesh can include display/cut-cell geometry at an exterior
    # edge.  The saved project domain is authoritative for a rendered result:
    # never interpolate or colour even a fraction of a pixel outside it.
    if domain_geometry is not None:
        domain=domain.intersection(domain_geometry)
        geometry=geometry.intersection(domain_geometry)
    # The optional terrain path reconstructs each stored wet volume against a
    # display-only planar bed.  Smooth results itself never changes the wet
    # outline: it is colour interpolation, not a Gaussian or new flood extent.
    terrain_geometry=None
    if terrain_aware_wetting and hydraulic:
        terrain_geometry,_=terrain_wet_geometry(frame)
        if domain_geometry is not None:terrain_geometry=terrain_geometry.intersection(domain_geometry)
        if not terrain_geometry.is_empty:geometry=terrain_geometry
    outline=geometry if terrain_geometry is None else terrain_geometry
    xmin,ymin,xmax,ymax=outline.bounds
    span=max(xmax-xmin,ymax-ymin);nx=max(2,round(maximum*(xmax-xmin)/span));ny=max(2,round(maximum*(ymax-ymin)/span))
    dx=(xmax-xmin)/nx;dy=(ymax-ymin)/ny
    x=xmin+(np.arange(nx)+.5)*dx;y=ymax-(np.arange(ny)+.5)*dy
    xx,yy=np.meshgrid(x,y);points=np.column_stack((xx.ravel(),yy.ravel()))
    mask=shapely.contains_xy(outline,points[:,0],points[:,1]);out=np.full(len(points),np.nan)
    _,nearest=cKDTree(centers[use]).query(points)
    # A result pixel is either in the retained wet footprint or absent.  Do
    # not introduce a semi-transparent fringe: it reads as dry/transparent
    # water even when the user explicitly turns zero transparency off.
    alpha=mask.astype(float)
    # Distinct wet components never exchange colors across a dry gap or wall.
    # Assign the new shoreline pixels to their nearest wet component, instead
    # of clipping them back to the staircase we have just reconstructed.
    pieces=list(geometry.geoms) if geometry.geom_type=='MultiPolygon' else [geometry]
    for piece in pieces:
        if terrain_geometry is not None:
            # A shallow wedge need not contain its parent cell centre.  Associate
            # it with every original wet cell whose footprint intersects it.
            cells=use&np.asarray(shapely.intersects(polygons,piece),bool)
            component=cells[use]
        else:
            component=shapely.intersects_xy(piece,centers[use,0],centers[use,1])
            cells=use&shapely.intersects_xy(piece,centers[:,0],centers[:,1])
        pixels=mask&component[nearest] if hydraulic else mask&shapely.intersects_xy(piece,points[:,0],points[:,1])
        if not np.any(cells) or not np.any(pixels):continue
        xy=centers[cells];data=np.asarray(values)[cells];sample=points[pixels]
        interpolated=_linear_values(xy,data,sample,outline,polygons[cells])
        if strength<1.:
            # Blend toward the value of the actual containing wet cell. This
            # cannot borrow colors across an obstacle or a disconnected patch.
            matches=shapely.STRtree(polygons[cells]).query(shapely.points(sample),predicate='within')
            if matches.size:
                rows,owners=matches
                raw=data[owners];smooth=np.where(np.isfinite(interpolated[rows]),interpolated[rows],raw)
                interpolated[rows]=(1.-strength)*raw+strength*smooth
        out[pixels]=interpolated
    rgba,_=colour_values(out,variable,limits=limits,palette=palette,
                         zero_transparent=zero_transparent)
    rgba[:,3]=np.rint(rgba[:,3]*alpha).astype(np.uint8)
    rgba=np.ascontiguousarray(rgba.reshape(ny,nx,4))
    image=QImage(rgba.data,nx,ny,rgba.strides[0],QImage.Format_RGBA8888).copy()
    return image,QRectF(to_scene(xmin,ymax),to_scene(xmax,ymin))


class ParticleTrailItem(QGraphicsItem):
    """Batch all tails by age, with a tapered fade and distinct luminous heads."""
    def __init__(self,view):
        super().__init__();self.view=view;self.tail_paths=[];self.head_points=QPolygonF();self._bounds=QRectF()
        self.setZValue(24)

    def boundingRect(self):return self._bounds

    def path(self):
        path=QPainterPath()
        for segment in self.tail_paths:path.addPath(segment)
        return path

    def set_tracks(self,history):
        # The map's affine world-to-scene mapping is applied in one array pass,
        # rather than once for every historical particle coordinate.
        origin=self.view.to_scene(0.,0.);ex=self.view.to_scene(1.,0.)-origin;ey=self.view.to_scene(0.,1.)-origin
        basis=np.array([[ex.x(),ex.y()],[ey.x(),ey.y()]])
        points=np.asarray(history)@basis+np.array([origin.x(),origin.y()])
        self.prepareGeometryChange();self.tail_paths=[]
        for first,last in zip(points[:-1],points[1:]):
            path=QPainterPath()
            for a,b in zip(first,last):path.moveTo(float(a[0]),float(a[1]));path.lineTo(float(b[0]),float(b[1]))
            self.tail_paths.append(path)
        self.head_points=QPolygonF([QPointF(float(x),float(y)) for x,y in points[-1]])
        minimum=points.min(axis=(0,1));maximum=points.max(axis=(0,1))
        margin=6./max(abs(self.view.transform().m11()),1e-9)
        self._bounds=QRectF(QPointF(*minimum),QPointF(*maximum)).adjusted(-margin,-margin,margin,margin)
        self.update()

    @staticmethod
    def tail_style(index,count):
        brightness=(index+1)/max(count,1)
        return round(230*brightness**2),.35+1.2*brightness

    def paint(self,painter,option,widget=None):
        painter.setRenderHint(QPainter.Antialiasing)
        for i,path in enumerate(self.tail_paths):
            alpha,width=self.tail_style(i,len(self.tail_paths))
            # Flat segment ends avoid bright overlapping beads along the tail.
            pen=QPen(QColor(210,244,255,alpha),width,Qt.SolidLine,Qt.FlatCap,Qt.RoundJoin);pen.setCosmetic(True)
            painter.setPen(pen);painter.drawPath(path)
        # One crisp tip only: no outer glow/halo, including when zoomed out.
        pen=QPen(QColor(250,255,255,255),2.6,Qt.SolidLine,Qt.RoundCap);pen.setCosmetic(True)
        painter.setPen(pen);painter.drawPoints(self.head_points)


def advect_frozen_particles(points, velocity, tree, radii, geometry, elapsed=.15):
    """Advance every track by the same display time, subcycling small cells.

    A cell-scale step limit controls spatial integration, never apparent speed.
    Every subsegment is checked against the wet domain, including solid holes.
    """
    after=np.asarray(points,float).copy();remaining=np.full(len(after),elapsed)
    good=np.ones(len(after),bool)
    while np.any(good & (remaining>1e-12)):
        ids=np.flatnonzero(good & (remaining>1e-12));before=after[ids]
        v=velocity(before);_,nearest=tree.query(before)
        step=np.minimum(remaining[ids],.2*radii[nearest]/np.maximum(np.linalg.norm(v,axis=1),1e-12))
        midpoint=before+.5*step[:,None]*v;vm=velocity(midpoint);_,middle=tree.query(midpoint)
        # Entering a refined region also constrains the current substep.
        step=np.minimum(step,.2*radii[middle]/np.maximum(np.linalg.norm(vm,axis=1),1e-12))
        trial=before+step[:,None]*velocity(before+.5*step[:,None]*v)
        finite=np.isfinite(trial).all(axis=1)&np.isfinite(step)&(step>0)
        safe=np.zeros(len(ids),bool)
        if np.any(finite):
            segments=shapely.linestrings(np.stack((before[finite],trial[finite]),axis=1))
            safe[finite]=shapely.covers(geometry,segments)
        after[ids[safe]]=trial[safe];remaining[ids[safe]]-=step[safe];good[ids[~safe]]=False
    return after,good


class ParticleTracks(QObject):
    """Advect decorative particles through one paused, immutable velocity frame."""
    def __init__(self,view):
        super().__init__(view);self.view=view;self.frame=None;self.item=None;self.enabled=False
        self.timer=QTimer(self);self.timer.setInterval(40);self.timer.timeout.connect(self.advance)
        self.positions=np.empty((0,2));self.rng=np.random.default_rng(1742);self.paths=[];self.age=None

    def clear(self):
        self.timer.stop();self.frame=None;self.positions=np.empty((0,2));self.paths=[]
        from shiboken6 import isValid
        if self.item is not None and isValid(self.item) and self.item.scene() is not None:self.view.scene().removeItem(self.item)
        self.item=None

    def set_frame(self,frame,density=1000):
        self.clear();self.frame=frame;polygons=np.array(frame.polygons,dtype=object);wet=wet_mask(frame)
        if not np.any(wet):return
        self.polygons=polygons[wet];self.geometry=shapely.union_all(self.polygons)
        self.centers=shapely.point_on_surface(self.polygons);self.xy=np.column_stack((shapely.get_x(self.centers),shapely.get_y(self.centers)))
        self.tree=cKDTree(self.xy);self.velocities=np.column_stack((frame.values['velocity_x_m_s'][wet],frame.values['velocity_y_m_s'][wet]))
        self.probability=shapely.area(self.polygons);self.probability/=self.probability.sum()
        self.radius=np.sqrt(shapely.area(self.polygons));self.positions=self._seed(density);self.age=np.zeros(density,dtype=int)
        self.paths=[self.positions.copy()];self.item=ParticleTrailItem(self.view);self.view.scene().addItem(self.item)
        self.item.set_tracks(self.paths)

    def _seed(self,count):
        ids=self.rng.choice(len(self.xy),count,p=self.probability);points=self.xy[ids].copy()
        candidate=points+self.rng.uniform(-.4,.4,(count,2))*self.radius[ids,None]
        good=shapely.contains_xy(self.geometry,candidate[:,0],candidate[:,1]);points[good]=candidate[good]
        return points

    def velocity(self,points):
        distance,indices=self.tree.query(points,k=min(4,len(self.xy)))
        if distance.ndim==1:distance=distance[:,None];indices=indices[:,None]
        weights=1/np.maximum(distance,1e-12)**2;weights/=weights.sum(axis=1)[:,None]
        return np.sum(self.velocities[indices]*weights[:,:,None],axis=1)

    def advance(self):
        if self.frame is None or not len(self.positions):return
        after,good=advect_frozen_particles(self.positions,self.velocity,self.tree,self.radius,self.geometry)
        self.age+=1;reset=(~good)|(self.age>140)
        after[reset]=self._seed(int(reset.sum()));self.age[reset]=0
        self.positions=after;self.paths.append(after.copy());self.paths=self.paths[-24:]
        # Never connect a reseeded particle to its former location.
        for previous in self.paths:previous[reset]=after[reset]
        self.item.set_tracks(self.paths)
