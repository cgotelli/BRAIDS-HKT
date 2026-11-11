"""Frozen normal-flow bed-slope proxy on the actual prepared finest mesh.

Normal flow assumes energy slope equals bed slope. Geometry cannot determine an
unsteady energy gradient. Fit signed inward falls, not their absolute values,
then take the boundary-width-weighted median of transverse-strip slopes. This
resists a minority of steep levee margins without searching for a desired sign.
Separate transverse-strip intercepts remove bank height/cross-sectional shape.
"""
from copy import deepcopy
import hashlib
from pathlib import Path
import numpy as np

AUTO_SLOPE='mesh'
SCHEMA='braids.normal-mesh-slopes.v1'
METHOD='normal_flow.initial_finest_section_frame_bed_fit.v2'


def _weighted_median(values,weights):
    order=np.argsort(values);v=np.asarray(values)[order];w=np.asarray(weights)[order]
    cumulative=np.cumsum(w);half=float(cumulative[-1]/2);at=int(np.searchsorted(cumulative,half))
    if at+1<len(v) and abs(cumulative[at]-half)<=8*np.finfo(float).eps*cumulative[-1]:
        return float((v[at]+v[at+1])/2)
    return float(v[at])


def inward_bed_slope(mesh,bed,name,*,rings=4):
    bed=np.asarray(bed,dtype=float);cells=mesh.cells;n=len(cells)
    if bed.shape!=(n,) or not np.isfinite(bed).all():
        raise ValueError('Normal depth needs finite prepared bed values on every finest mesh cell')
    faces=[f for f in mesh.faces if f.neighbour<0 and f.kind=='inlet' and f.bc_id==name]
    if not faces:raise ValueError(name+': normal depth has no inlet faces')
    outward=np.asarray(faces[0].outward_normal,float);inward=-outward
    tangent=np.array([-inward[1],inward[0]])
    origin=np.asarray(faces[0].start_xy,float)
    ends=np.asarray([v for f in faces for v in (f.start_xy,f.end_xy)],float)
    scale=max(1.,float(np.max(abs(ends))));tol=max(1e-10,64*np.spacing(scale))
    if (abs(np.linalg.norm(inward)-1)>1e-12 or
        any(not np.allclose(f.outward_normal,outward,rtol=0,atol=1e-12) for f in faces) or
        np.any(abs((ends-origin)@inward)>tol)):
        raise ValueError(name+': normal depth requires one collinear section with a consistent inward normal')
    xy=np.asarray([c.centroid_xy for c in cells],float);relative=xy-origin
    distance=relative@inward;station=relative@tangent
    h=float(np.hypot(inward[0]*mesh.grid.dx_m,inward[1]*mesh.grid.dy_m))
    if not np.isfinite(h) or h<=0:raise ValueError('Positive finest mesh spacing required')
    transverse=(ends-origin)@tangent;lo=float(transverse.min());hi=float(transverse.max())
    eligible=(distance>=-tol)&(distance<=(rings-.5)*h+tol)&(station>=lo-tol)&(station<=hi+tol)
    # Real shared faces determine connectivity; never bridge a wall, island,
    # dry-land gap or a nearby but disconnected reach using Euclidean proximity.
    rows=[[] for _ in cells]
    for f in mesh.faces:
        if f.neighbour>=0:
            rows[f.owner].append(f.neighbour);rows[f.neighbour].append(f.owner)
    reached=set(int(f.owner) for f in faces);front=list(reached)
    while front:
        next_front=[]
        for i in front:
            for j in rows[i]:
                if j not in reached and eligible[j]:reached.add(j);next_front.append(j)
        front=next_front
    ids=np.array(sorted(reached),dtype=np.int64)
    area=np.asarray([c.area_m2 for c in cells],float)
    groups={}
    widths={}
    # Strip identity is physical transverse station, never a background i/j
    # index. A merged inlet cell can own several face fragments, and a face
    # can span several strips. Partition its *length*, not its cell count.
    count=max(1,int(np.ceil((hi-lo)/h)))
    edges=lo+np.arange(count+1)*h;edges[-1]=hi
    for f in faces:
        ends_f=(np.asarray([f.start_xy,f.end_xy])-origin)@tangent
        a,b=sorted(ends_f)
        first=max(0,min(count-1,int(np.floor((a-lo)/h))))
        last=max(0,min(count-1,int(np.floor((b-lo)/h))))
        for key in range(first,last+1):
            overlap=max(0.,min(b,edges[key+1])-max(a,edges[key]))
            widths[key]=widths.get(key,0.)+overlap
    for i in ids:
        key=max(0,min(count-1,int(np.floor((station[i]-lo)/h))))
        groups.setdefault(key,[]).append(int(i))
    xs=[];ys=[];ts=[];weights=[];used=[];intercepts=[]
    for key,indices in groups.items():
        ix=np.asarray(indices,int);w=area[ix]
        if len(ix)<2 or np.ptp(distance[ix])<.5*h or widths.get(key,0.)<=0:continue
        average=lambda v:float(np.dot(v,w)/w.sum())
        xs.extend(distance[ix]-average(distance[ix]));ys.extend(bed[ix]-average(bed[ix]))
        ts.extend(station[ix]-average(station[ix]));weights.extend(w);used.extend(ix.tolist())
        intercepts.append(dict(cells=ix.tolist(),mean_bed_m=average(bed[ix]),boundary_width_m=widths[key]))
    if not xs:raise ValueError(name+': normal depth cannot resolve an inward mesh bed slope; extend the domain by at least two finest cells or choose critical depth / prescribed elevation / prescribed depth')
    x=np.asarray(xs);y=np.asarray(ys);t=np.asarray(ts);w=np.sqrt(np.asarray(weights))
    # Retained wall fragments can shift centroids transversely. Include their
    # transverse shift only when independently resolved; no rank-deficient
    # two-dimensional fit is allowed to invent a longitudinal gradient.
    X=np.column_stack((x,t));transverse_fit=np.linalg.matrix_rank(X*w[:,None],tol=1e-11*h)==2
    if not transverse_fit:X=X[:,:1]
    coefficient=np.linalg.lstsq(X*w[:,None],y*w,rcond=None)[0]
    mean_fit=-float(coefficient[0]);transverse_gradient=float(coefficient[1]) if transverse_fit else 0.
    cursor=0;strip_slopes=[];strip_widths=[]
    for row in intercepts:
        sl=slice(cursor,cursor+len(row['cells']));cursor+=len(row['cells'])
        weighted_x=x[sl]*w[sl]
        fall=-float(np.dot(weighted_x,(y[sl]-transverse_gradient*t[sl])*w[sl])/np.dot(weighted_x,weighted_x))
        row['inward_bed_fall_m_m']=fall;strip_slopes.append(fall);strip_widths.append(row['boundary_width_m'])
    slope=_weighted_median(strip_slopes,strip_widths)
    mad=_weighted_median(np.abs(np.asarray(strip_slopes)-slope),strip_widths)
    residual=y+slope*x-transverse_gradient*t
    uncertainty=64*np.finfo(float).eps*max(1.,float(np.max(abs(bed[used]))))/max(h,float(np.ptp(distance[used])))
    if not np.isfinite(slope) or slope<=uncertainty:
        raise ValueError(f'{name}: normal depth requires a resolved bed fall into the domain; prepared mesh slope is {slope:.8g} m/m. A flat/adverse bed has no positive uniform-flow normal depth. Choose critical depth, water elevation or water depth; no absolute-value or slope floor is applied.')
    return dict(method=METHOD,energy_slope=slope,assumption='uniform flow: energy slope equals initial inward bed slope',
        source='prepared finest-cell bed averages; frozen across AMR and restart',
        aggregation='boundary-width-weighted median of signed per-transverse-strip longitudinal fits; no absolute value, positive-only selection or extended search',
        strip_slope_range_m_m=[float(min(strip_slopes)),float(max(strip_slopes))],
        strip_slope_weighted_mad_m_m=mad,area_weighted_mean_fit_slope_m_m=mean_fit,strips=intercepts,
        normal_inward=inward.tolist(),finest_spacing_m=h,requested_inward_rows=rings,
        strip_frame='physical section normal/tangent; boundary-length-weighted transverse strips',
        fit_extent_m=[float(distance[used].min()),float(distance[used].max())],
        sample_cells=len(used),transverse_strips=len(intercepts),transverse_centroid_correction=bool(transverse_fit),
        bed_fit_rms_m=float(np.sqrt(np.average(residual**2,weights=np.asarray(weights)))),
        bed_sha256=hashlib.sha256(np.asarray(bed,dtype='<f8').tobytes()).hexdigest(),
        sample_cell_ids=used)


def prepare_slopes(hydraulics,mesh,bed):
    values={name:inward_bed_slope(mesh,bed,name) for name,row in hydraulics.items()
        if row['method']=='inlet.normal_discharge' and row.get('energy_slope')==AUTO_SLOPE}
    return dict(schema=SCHEMA,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),boundaries=values)


def resolved_hydraulics(hydraulics,document):
    """Resolve an authenticated prepared result, never reinterpret user data."""
    names={name for name,row in hydraulics.items() if row['method']=='inlet.normal_discharge'}
    if (set(document)!={'schema','source_sha256','boundaries'} or document['schema']!=SCHEMA or
        document['source_sha256']!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest() or
        set(document['boundaries'])!=names):raise ValueError('Prepared normal-depth slope identity differs; prepare again')
    result=deepcopy(hydraulics)
    for name in names:
        value=document['boundaries'][name]
        if (result[name]['energy_slope']!=AUTO_SLOPE or value.get('method')!=METHOD or
            type(value.get('energy_slope')) not in (int,float) or not np.isfinite(value['energy_slope']) or value['energy_slope']<=0):
            raise ValueError(name+': invalid prepared automatic normal-depth slope')
        result[name]['energy_slope']=float(value['energy_slope'])
    return result
