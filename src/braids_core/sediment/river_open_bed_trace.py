"""Bounded slope continuation for the sediment wave at free open boundaries.

The hydraulic face producer intentionally flattens some boundary-cell bed
slopes. That bed jump is not a physical step to diffuse into Exner. Extend
the bed limiter stencil through free openings with its one-sided LS plane.
Change only the private sediment-wave bed traces, preserving their depths.
Actual hydraulics, capacity reconstruction, prescribed beds and walls retain
their existing operators. NumPy/CuPy perform the same resident arithmetic.
"""
import hashlib
import numpy as np
from .bend_methods_v1 import digest
from .bed_columns_v101 import host


def prepared_face_bed(p,bed):
    """Initial physical-face elevation; prescribed elevations override it.

    A weighted one-sided plane uses distinct actual neighbours. Extrapolation
    is bounded by one observed neighbour elevation span. A single isolated
    cell has no supported slope and retains its own prepared elevation.
    This is setup geometry/data preparation on the host, not GPU stepping.
    """
    g={k:host(p.g[k]) for k in ('owner','neighbour','bid','offset','faces','xy','mid')}
    z=np.asarray(bed);out=np.zeros(len(g['owner']))
    opening=np.flatnonzero((g['neighbour']<0)&(g['bid']>=0))
    for i in np.unique(g['owner'][opening]):
        adjacent=g['faces'][g['offset'][i]:g['offset'][i+1]]
        neighbours=np.where(g['owner'][adjacent]==i,g['neighbour'][adjacent],g['owner'][adjacent])
        neighbours=np.unique(neighbours[neighbours>=0]);gradient=np.zeros(2);span=0.
        if len(neighbours):
            delta=(g['xy'][:,neighbours]-g['xy'][:,i,None]).T
            norm=np.linalg.norm(delta,axis=1);valid=norm>0.;delta=delta[valid];dv=z[neighbours[valid]]-z[i]
            if len(delta):
                weights=1./norm[valid];gradient=np.linalg.lstsq(delta*weights[:,None],dv*weights,rcond=None)[0]
                span=float(np.max(np.abs(dv)))
        faces=opening[g['owner'][opening]==i]
        increment=gradient@(g['mid'][:,faces]-g['xy'][:,i,None])
        out[faces]=z[i]+np.clip(increment,-span,span)
    return out


class OpenBedTrace:
    def __init__(self,p):
        self.xp=p.xp;g={k:host(v) for k,v in p.g.items()}
        mode=host(p.exterior_mode)
        eligible=np.flatnonzero((g['neighbour']<0)&(g['bid']>=0)&(mode==1))
        cells=np.unique(g['owner'][eligible]);self.active=bool(len(cells))
        self.identity=dict(method='free_open_bed_bounded_plane_continuation_v1',source_sha256=digest(__file__),
            bed='one-sided physical LS plane; limiter bounds include virtual cells mirrored across free openings',
            hydraulic_depth='preserve existing private wave-trace depth by shifting eta with bed',
            scope='interior sediment-wave sides owned by free-open-boundary cells; actual hydraulic flux unchanged',
            prescribed='fixed/elevation/rate boundary cells and ordinary wall-only cells unchanged',
            backend='resident NumPy/CuPy; host geometry only at construction',cells=int(len(cells)))
        if not self.active:return
        lookup={int(i):k for k,i in enumerate(cells)};neighbours=[];virtual=[];internal=[];target=[]
        for i in cells:
            neighbours.append([int(i)]);virtual.append([]);internal.append([])
            for k in range(g['offset'][i],g['offset'][i+1]):
                f=int(g['faces'][k]);own=int(g['owner'][f]);other=int(g['neighbour'][f])
                side=0 if own==i else 1;j=other if own==i else own
                dx,dy=g['displacement'][2*side:2*side+2,f]
                if j>=0:
                    neighbours[-1].append(j);internal[-1].append((dx,dy));target.append((lookup[int(i)],side,f,dx,dy))
                elif g['bid'][f]>=0 and mode[f]==1:
                    virtual[-1].append((2*dx,2*dy))
        def pad(rows,width,components=False):
            if components:
                a=np.zeros((len(rows),width,2))
                for k,row in enumerate(rows):
                    if row:a[k,:len(row)]=row
            else:
                a=np.empty((len(rows),width),np.int64)
                for k,row in enumerate(rows):a[k]=row[0];a[k,:len(row)]=row
            return a
        near=pad(neighbours,max(map(len,neighbours)))
        outside=pad(virtual,max(map(len,virtual)),True)
        faces=pad(internal,max(1,max(map(len,internal))),True)
        t=np.asarray(target,float).reshape(-1,5)
        self.cells=p.xp.asarray(cells);self.neighbours=p.xp.asarray(near)
        self.virtual=p.xp.asarray(outside);self.faces=p.xp.asarray(faces)
        self.rows=p.xp.asarray(t[:,0].astype(np.int64));self.sides=p.xp.asarray(t[:,1].astype(np.int64))
        self.indices=p.xp.asarray(t[:,2].astype(np.int64));self.displacement=p.xp.asarray(t[:,3:])
        self.identity['geometry_sha256']=hashlib.sha256(cells.tobytes()+near.tobytes()+outside.tobytes()+faces.tobytes()+t.tobytes()).hexdigest()

    def apply(self,z,gradient,trace):
        if not self.active:return
        xp=self.xp;v=z[self.cells];gx=gradient[4,self.cells];gy=gradient[5,self.cells]
        values=z[self.neighbours]
        extension=v[:,None]+gx[:,None]*self.virtual[:,:,0]+gy[:,None]*self.virtual[:,:,1]
        lo=xp.minimum(values.min(axis=1),extension.min(axis=1))
        hi=xp.maximum(values.max(axis=1),extension.max(axis=1))
        inc=gx[:,None]*self.faces[:,:,0]+gy[:,None]*self.faces[:,:,1]
        eps=64*np.finfo(float).eps*xp.maximum(xp.abs(v),xp.maximum(xp.abs(lo),xp.abs(hi)))
        positive=inc>eps[:,None];negative=inc<-eps[:,None]
        phi=xp.where(positive,(hi-v)[:,None]/xp.where(positive,inc,1.),
                     xp.where(negative,(lo-v)[:,None]/xp.where(negative,inc,1.),1.))
        phi=xp.maximum(0.,xp.minimum(1.,phi.min(axis=1)))
        dz=phi[self.rows]*(gx[self.rows]*self.displacement[:,0]+gy[self.rows]*self.displacement[:,1])
        bed=v[self.rows]+dz;old=trace[self.sides,3,self.indices].copy()
        trace[self.sides,0,self.indices]+=bed-old
        trace[self.sides,3,self.indices]=bed


def install(a):
    p=a.physics
    if (not hasattr(p,'exterior_sediment') or not hasattr(p,'hydraulic_wave_trace')
            or p.identity['consistent_hydraulic_wave_state']['pressure_only']):return
    if hasattr(p,'open_bed_trace'):raise ValueError('Open bed trace installed twice')
    p.open_bed_trace=OpenBedTrace(p)
    target=a
    while 'solver' in vars(target):target=target.solver
    target.identity['open_bed_trace']=p.open_bed_trace.identity
