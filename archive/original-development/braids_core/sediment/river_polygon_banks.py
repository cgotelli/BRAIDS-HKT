"""Conservative bank/layer transactions on conforming polygon leaf topology.

Exterior walls are absent from the transfer graph. Axis chains preserve the
old row/column ordering without inventing neighbors across walls or holes.
"""
import hashlib
from functools import lru_cache
from pathlib import Path
from types import MethodType
import numpy as np
from shapely.geometry import Polygon
from .river_polygon_geometry import polygon_record
from .river_mixtures import RiverMixtureSolver
from .bed_columns_v101 import host
from .bend_methods_v1 import BendTrialRejected,digest
from validation.sediment.river_morphodynamics.sediment_wave import WaveEngine
from validation.sediment.river_morphodynamics.wet_exposure import engine as gradient_engine


@lru_cache(maxsize=32768)
def _cell_geometry(vertex_bytes,area):
    """Validated immutable geometry, never hydraulic or morphological state.

    The key owns exact FP64 coordinate bytes and the authoritative area. An
    edited/reordered polygon or changed cell area therefore cannot reuse a
    previous validation result. Changing leaf patterns reuse their unchanged
    cells; returned mesh arrays are assembled separately from these objects.
    """
    points=np.frombuffer(vertex_bytes,dtype=np.float64).reshape(-1,2)
    poly,valid,poly_area,_,_=polygon_record(vertex_bytes)
    if not valid or not 3<=len(points)<=64:
        raise ValueError('a simple 3..64 vertex polygon is required')
    # Bound the area lost on the GIS-coordinate round trip, without replacing
    # the authoritative area or granting a blanket small-cell tolerance.
    span=np.roll(points,-1,axis=0)-np.roll(points,1,axis=0)
    ulp=np.abs(np.spacing(np.abs(points)))
    rounding=.5*np.sum(np.abs(span[:,1])*ulp[:,0]+np.abs(span[:,0])*ulp[:,1])
    rounding+=len(points)*float(np.max(ulp))**2
    tolerance=max(1e-13+1e-9*area,2*rounding)
    if abs(poly_area-area)>tolerance:
        raise ValueError(f'polygon area {poly_area:.17g} differs from actual cell area '
            f'{area:.17g} m2 by more than coordinate-rounding tolerance {tolerance:.3g} m2')
    anchor=points[0];relative=points-anchor
    local_center,local=_local_geometry(relative.tobytes(),area)
    center=anchor+local_center
    center.setflags(write=False)
    return center,local


@lru_cache(maxsize=32768)
def _local_geometry(relative_bytes,area):
    """Reuse identical cell shapes, not rounded global coordinates.

    Most Cartesian leaves share an exactly identical local rectangle. Cropped
    walls only share when every relative coordinate AND the physical area is
    identical. Global validity and coordinate-rounding checks remain above.
    """
    relative=np.frombuffer(relative_bytes,dtype=np.float64).reshape(-1,2)
    local_center=np.array(Polygon(relative).centroid.coords[0])
    local=relative-local_center
    signed=np.sum(local[:,0]*np.roll(local[:,1],-1)-local[:,1]*np.roll(local[:,0],-1))
    if signed<0:local=local[::-1]
    # The private affine-integration polygon must integrate a constant to the
    # physical cell area. Hydraulic faces and public polygons are unchanged.
    if abs(abs(signed)*.5-area)>32*np.finfo(float).eps*area:
        local*=np.sqrt(area/(abs(signed)*.5))
    # Byte-backed arrays cannot be made writable by a downstream caller.
    local_center=np.frombuffer(local_center.tobytes(),dtype=np.float64)
    local=np.frombuffer(local.tobytes(),dtype=np.float64).reshape(local.shape)
    return local_center,local


def topology(data,polygons,*,transfers=True):
    if type(transfers) is not bool:raise ValueError('Explicit polygon transfer construction required')
    n=len(data['area']);area=np.asarray(data['area'],float)
    if len(polygons)!=n or not np.isfinite(area).all() or np.any(area<=0):raise ValueError('Invalid polygon areas')
    centers=[];vertices=[];offset=[0]
    for i,points in enumerate(polygons):
        points=np.asarray(points,float)
        if points.ndim!=2 or points.shape[1]!=2 or not np.isfinite(points).all():raise ValueError('Invalid polygon vertices')
        if len(points)>1 and np.array_equal(points[0],points[-1]):points=points[:-1]
        try:center,local=_cell_geometry(points.tobytes(),area[i])
        except ValueError as error:raise ValueError(f'Cell {i}: {error}') from error
        centers.append(center);vertices.extend(local);offset.append(len(vertices))
    xy=np.array(centers);edges={};directions=[{},{}]
    normals=np.asarray(data['normal']);interior=np.asarray(data['neighbour'])>=0
    oblique=bool(np.any((np.abs(normals[0,interior])>1e-10)&(np.abs(normals[1,interior])>1e-10)))
    for f,(a,b) in enumerate(zip(data['owner'],data['neighbour'])):
        if b<0:continue
        a,b=int(a),int(b)
        if a==b or not 0<=min(a,b)<=max(a,b)<n:raise ValueError('Invalid internal face')
        normal=np.asarray(data['normal'])[:,f];length=float(data['length'][f])
        if not np.isfinite(normal).all() or not np.isfinite(length) or length<=0 or abs(np.linalg.norm(normal)-1)>1e-10:raise ValueError('Invalid shared face')
        axis=int(np.argmax(abs(normal)))
        left,right=(a,b) if normal[axis]>0 else (b,a);key=tuple(sorted((a,b)))
        # A stepped merged cell can meet the same neighbour along BOTH axes.
        # This is general adjacency, not an invalid hydraulic normal. Validate
        # that normal separately; never collapse or rotate the physical faces.
        separation=xy[b]-xy[a]
        roundoff=8*float(np.max(np.spacing(np.abs(xy[[a,b]]))))
        if np.dot(normal,separation)<-max(1e-12,roundoff):
            raise ValueError('Inconsistent shared-face orientation')
        if key in edges and edges[key]!=(axis,left,right):oblique=True
        edges[key]=(axis,left,right)
    if oblique:
        edges={key:(-1,*sorted(key,key=lambda c:(xy[c,0],xy[c,1],c))) for key in edges}
    if not transfers:
        # Geometry is still fully validated, and survey/remesh users still own
        # the exact polygons, area and centroids. A hydraulic/bedload-only owner
        # never consumes bank adjacency, reverse edges or collapse schedules.
        # Do not build those graphs (or their native operators) on every regrid.
        result=dict(area=area.copy(),xy=xy,vertex_offsets=np.array(offset,np.int32),
            vertices=np.ascontiguousarray(np.array(vertices).T),
            offset=np.zeros(n+1,np.int32),neighbour=np.empty(0,np.int32),
            reverse=np.empty(0,np.int32),chains=[],branching=False,oblique=oblique)
        hasher=hashlib.sha256(b'validated-polygon-geometry-without-transfer-operators-v1')
        for k,v in sorted(result.items()):
            if isinstance(v,np.ndarray):hasher.update(k.encode()+str(v.shape).encode()+str(v.dtype).encode()+v.tobytes())
        hasher.update(np.asarray(sorted(edges.values()),dtype=np.int64).tobytes())
        result['sha256']=hasher.hexdigest();return result
    adjacent=[[] for _ in range(n)]
    for a,b in sorted(edges):adjacent[a].append(b);adjacent[b].append(a)
    off=[0];nb=[];lookup={}
    for a,neighbors in enumerate(adjacent):
        for b in neighbors:lookup[a,b]=len(nb);nb.append(b)
        off.append(len(nb))
    rev=[0]*len(nb)
    for (a,b),e in lookup.items():rev[e]=lookup[b,a]
    chains=[];branching=False
    for axis in ((-1,) if oblique else range(2)):
        nxt={};prev={}
        pairs=sorted((a,b) for ax,a,b in edges.values() if ax==axis)
        has_branch=oblique or len({a for a,b in pairs})!=len(pairs) or len({b for a,b in pairs})!=len(pairs)
        if has_branch:
            # Oblique breaklines and coarse/fine leaves have general adjacency.
            # Matchings keep
            # each cell in at most one pair per launch, with deterministic
            # barriers between colors. No races or duplicate donor stocks.
            # Retain the historical chain order on non-branching axes.
            branching=True;groups=[];occupied=[]
            for a,b in pairs:
                color=next((j for j,s in enumerate(occupied) if a not in s and b not in s),len(groups))
                if color==len(groups):groups.append([]);occupied.append(set())
                groups[color].append((a,b));occupied[color].update((a,b))
            for group in groups:
                left=np.array([a for a,b in group],np.int32);right=np.array([b for a,b in group],np.int32)
                distance=np.linalg.norm(xy[right]-xy[left],axis=1)
                if not np.isfinite(distance).all() or np.any(distance<=0):raise ValueError('Coincident cell centers')
                chains.append(dict(offset=np.arange(len(group)+1,dtype=np.int32),left=left,right=right,distance=distance))
            continue
        for ax,a,b in edges.values():
            if ax!=axis:continue
            if a in nxt or b in prev:raise ValueError('Branching Cartesian chain not supported')
            nxt[a]=b;prev[b]=a
        starts=sorted(set(nxt)-set(prev),key=lambda c:(xy[c,1-axis],xy[c,axis]));used=set()
        co=[0];left=[];right=[];dist=[]
        for a in starts:
            while a in nxt:
                if a in used:raise ValueError('Cyclic Cartesian chain')
                used.add(a);b=nxt[a];distance=float(np.linalg.norm(xy[b]-xy[a]))
                if not np.isfinite(distance) or distance<=0:raise ValueError('Coincident cell centers')
                left.append(a);right.append(b);dist.append(distance);a=b
            co.append(len(left))
        if len(used)!=len(nxt):raise ValueError('Disconnected cyclic chain')
        chains.append(dict(offset=np.array(co,np.int32),left=np.array(left,np.int32),right=np.array(right,np.int32),distance=np.array(dist)))
    result=dict(area=area.copy(),xy=xy,vertex_offsets=np.array(offset,np.int32),vertices=np.ascontiguousarray(np.array(vertices).T),
        offset=np.array(off,np.int32),neighbour=np.array(nb,np.int32),reverse=np.array(rev,np.int32),chains=chains,branching=branching,oblique=oblique)
    hasher=hashlib.sha256()
    for group in [result,*chains]:
        for k,v in sorted(group.items()):
            if isinstance(v,np.ndarray):hasher.update(k.encode()+str(v.shape).encode()+str(v.dtype).encode()+v.tobytes())
    result['sha256']=hasher.hexdigest();return result


class PolygonProcesses:
    def __init__(self,backend,data,polygons,threads=4,*,transfers=True):
        if backend not in ('cpu','gpu'):raise ValueError('Explicit CPU/GPU required')
        self.xp=np
        if backend=='gpu':
            import cupy as cp
            self.xp=cp
        t=topology(data,polygons,transfers=transfers);self.n=len(t['area']);self.na=len(t['neighbour']);xp=self.xp
        self.transfers=transfers
        # Preserve the actual mesh polygons for AMR identity/intersections.
        # Reconstructing them from rounded centroids plus local integration
        # vertices can create spurious overlaps at projected coordinates.
        self.polygons=tuple(np.array(points,dtype=float,copy=True) for points in polygons)
        for points in self.polygons:points.setflags(write=False)
        t['sha256']=hashlib.sha256(t['sha256'].encode()+b''.join(p.tobytes() for p in self.polygons)).hexdigest()
        self.g={k:xp.asarray(v) for k,v in t.items() if isinstance(v,np.ndarray)}
        self.chains=[{k:xp.asarray(v) for k,v in ch.items()} for ch in t['chains']]
        # Private launch policy, not a physical option or CPU/GPU selector.
        # A fresh screen is required for each axis/color, never cached by time.
        self._screen_scour=backend=='gpu'
        self._scour_possible=xp.empty(max((len(c['left']) for c in self.chains),default=0),xp.int32)
        self.engine=(WaveEngine(backend,threads,source=Path(__file__).with_suffix('.h'),
            names=('polygon_wet','graph_bank_propose','graph_bank_scale','graph_bank_gather','graph_scour_screen','graph_scour'))
            if transfers else None)
        if backend=='gpu' and transfers:
            # Long chains have ordered, divergent work. Measurements on active
            # as well as inactive slopes favor one chain per block. Colored
            # one-edge groups retain the ordinary 128-thread launch instead.
            ordinary_launch=self.engine.launch
            def launch(name,count,*args):
                if name!='graph_scour' or not args[-3]:return ordinary_launch(name,count,*args)
                return self.engine.fn[name]((count,),(1,),tuple(np.int32(v) if isinstance(v,int)
                    else np.float64(v) if isinstance(v,float) else v for v in args))
            self.engine.launch=launch
        self.identity=dict(runtime=self.engine.identity if transfers else None,wrapper_sha256=digest(__file__),geometry_sha256=t['sha256'],
            geometry_helper_sha256=digest(Path(__file__).with_name('river_polygon_geometry.py')),
            area='actual polygon area',neighbors='unique internal shared-face pairs; exterior boundaries excluded',
            bank='FLOW net mass erosion, minimum eligible area, donor composition, shared finite-stock limit',
            collapse='legacy threshold/equalizing geometry; harmonic pair area prevents unequal-area overshoot',
            schedule=('shared-face conflict-free pair colors on oblique topology' if t['oblique'] else
                      'axis-ordered conflict-free pair colors on branching axes; historical chains elsewhere' if t['branching'] else
                      'independent x chains then y chains, ordered forward pairs inside each chain'),public_support=False)
        if not transfers:
            self.identity.update(transfer_operators=False,
                schedule='not constructed: bank sharing and collapse are explicitly off')

    def require_transfers(self):
        if not self.transfers:
            raise ValueError('Polygon transfer operators were not requested by this owner')

    def real(self,a,shape):
        xp=self.xp;a=xp.ascontiguousarray(a,dtype=xp.float64)
        if a.shape!=shape or not bool(xp.all(xp.isfinite(a))):raise ValueError('Invalid polygon process field')
        return a

    def wet(self,h,z,gradient,*,dry=1e-6,sedthr=.001):
        self.require_transfers()
        xp=self.xp;n=self.n;g=self.g
        h=self.real(h,(n,));z=self.real(z,(n,));grad=self.real(gradient,(2,n))
        if not 0<=dry<=sedthr or bool(xp.any(h<0)):raise ValueError('Invalid wet geometry depth')
        out=xp.zeros((4,n));active=xp.zeros(n,xp.int32)
        self.engine.launch('polygon_wet',n,n,g['vertices'].shape[1],float(dry),float(sedthr),g['vertex_offsets'],g['vertices'],g['area'],h,z,grad,out,active)
        if not bool(xp.all(xp.isfinite(out))) or float(xp.max(xp.abs(out[3])))>1e-10:
            raise BendTrialRejected('Polygon wet volume inversion failed')
        return out,active

    def bank(self,z,wet,erosion,fraction,stock,*,theta):
        self.require_transfers()
        xp=self.xp;n=self.n;ng=stock.shape[0];na=self.na;g=self.g
        if not np.isfinite(theta) or not 0<=theta<=1:raise ValueError('Invalid bank fraction')
        z=self.real(z,(n,));erosion=self.real(erosion,(ng,n));fraction=self.real(fraction,(ng,n));stock=self.real(stock,(ng,n))
        wet=xp.ascontiguousarray(wet,dtype=xp.int32)
        if wet.shape!=(n,) or bool(xp.any((wet!=0)&(wet!=1))) or bool(xp.any(stock<0)) or bool(xp.any(fraction<0)):
            raise ValueError('Invalid bank stock or wet mask')
        proposal=xp.zeros((ng,na));scale=xp.ones((ng,n));moved=xp.zeros((ng,n));e=self.engine
        e.launch('graph_bank_propose',n,n,ng,na,float(theta),g['offset'],g['neighbour'],g['area'],z,wet,erosion,fraction,stock,proposal)
        e.launch('graph_bank_scale',n*ng,n,ng,na,g['offset'],g['reverse'],g['area'],stock,proposal,scale)
        e.launch('graph_bank_gather',n*ng,n,ng,na,g['offset'],g['neighbour'],g['reverse'],proposal,scale,moved)
        if (not bool(xp.all(xp.isfinite(moved))) or bool(xp.any(stock+moved/g['area']<-1e-10))
                or float(xp.max(xp.abs(moved.sum(axis=1))))>1e-9):raise BendTrialRejected('Graph bank budget/stock failed')
        return moved

    def scour(self,mass,rock,*,slope,column_density=1590.,solid_fraction=None):
        self.require_transfers()
        xp=self.xp;n=self.n
        if mass.ndim!=3 or min(mass.shape)<1:raise ValueError('Expected layers, classes, cells')
        nl,ng,_=mass.shape;mass=self.real(mass,(nl,ng,n));rock=self.real(rock,(n,))
        density=xp.broadcast_to(xp.asarray(column_density,dtype=xp.float64),(ng,))
        density=self.real(density,(ng,))
        if bool(xp.any(density<=0)):raise ValueError('Positive class column densities required')
        if bool(xp.any(mass<0)) or not np.isfinite(slope) or slope<=0:raise ValueError('Invalid collapse stock/slope')
        packing=xp.ones((nl,n)) if solid_fraction is None else self.real(solid_fraction,(nl,n)).copy()
        if bool(xp.any((packing<=0)|(packing>1))):raise ValueError('Invalid collapse solid fraction')
        work=mass.copy();moved=xp.zeros((ng,n))
        for c in self.chains:
            nc=len(c['offset'])-1
            if nc:
                # One-edge color groups already have maximum pair parallelism.
                screened=int(self._screen_scour and len(c['left'])>nc)
                if screened:self.engine.launch('graph_scour_screen',len(c['left']),len(c['left']),n,ng,nl,float(slope),
                    c['left'],c['right'],c['distance'],rock,work,self._scour_possible,density,packing)
                self.engine.launch('graph_scour',nc,nc,n,ng,nl,float(slope),c['offset'],c['left'],c['right'],c['distance'],self.g['area'],rock,work,moved,density,screened,self._scour_possible,packing)
        closure=(work-mass).sum(axis=0)*self.g['area']-moved
        if (not bool(xp.all(xp.isfinite(work))) or bool(xp.any(work<0))
                or float(xp.max(xp.abs(moved.sum(axis=1))))>1e-9 or float(xp.max(xp.abs(closure)))>1e-9):
            raise BendTrialRejected('Graph layered collapse budget/stock failed')
        return (work,moved) if solid_fraction is None else (work,packing,moved)


class RiverPolygonBankSolver(RiverMixtureSolver):
    """The actual-geometry successor; full-cell constructors remain frozen evidence."""
    def __init__(self,*args,polygons,bank_theta=.5,collapse=False,collapse_slope=.5,repose_angle_deg=32.,**kwargs):
        if not np.isfinite(bank_theta) or not 0<=bank_theta<=1:raise ValueError('Invalid bank fraction')
        if type(collapse) is not bool or not np.isfinite(collapse_slope) or collapse_slope<=0:raise ValueError('Invalid collapse control')
        super().__init__(*args,**kwargs)
        self.bank_theta=float(bank_theta);self.collapse=collapse;self.collapse_slope=float(collapse_slope)
        p=self.physics;xp=self.xp;ng=len(self.capacity)
        self.polygon=PolygonProcesses(self.backend,self.data,polygons,p.flow.threads,
            transfers=bool(bank_theta or collapse))
        self.gradient_engine=gradient_engine(self.backend,p.flow.threads) if bank_theta else None
        p.polygon_gradient=xp.zeros((2,p.n));p.polygon_wet_geometry=xp.zeros((4,p.n));p.polygon_wet_active=xp.zeros(p.n,xp.int32)
        for name in ('bank','collapse'):
            for suffix in ('trial','cumulative'):setattr(p,'class_'+name+'_'+suffix,xp.zeros((ng,p.n)))
            setattr(p,'class_'+name+'_gross',xp.zeros(ng))
        for name in ('bank','relaxation'):
            for suffix in ('trial','cumulative'):setattr(p,name+'_'+suffix+'_mass',xp.zeros(p.n))
        original=p.commit
        def commit(owner):
            original()
            for name in ('bank','collapse'):
                trial=getattr(owner,'class_'+name+'_trial')
                getattr(owner,'class_'+name+'_cumulative')[:]+=trial
                getattr(owner,'class_'+name+'_gross')[:]+=xp.maximum(trial,0.).sum(axis=1)
            owner.bank_cumulative_mass+=owner.bank_trial_mass;owner.relaxation_cumulative_mass+=owner.relaxation_trial_mass
        p.commit=MethodType(commit,p)
        p.identity['polygon_bank_collapse']=dict(process=self.polygon.identity,
            gradient_runtime=self.gradient_engine.identity if self.gradient_engine is not None else None,
            bank_theta=bank_theta,collapse=collapse,collapse_slope=collapse_slope,
            wet_activity='affine bed clipped to actual polygon; conserved volume / wet area; existing physical cutoff',
            one_inventory=True,source_sha256=digest(__file__))
        p.identity['dry_bank']=dict(theta=bank_theta,hmaxth=0.,sedthr=p.sediment_depth,geometry='actual_internal_faces')
        p.identity['slope_relaxation']=dict(method='polygon_layered_scour' if collapse else 'off')
        self.identity['selected_river_bundle'].update(banks=bool(bank_theta),collapse=collapse,
            scope='Cartesian_cut_cell_classwise_bank_collapse_implicit_midpoint')

    def column_delta(self,qnew):
        """Bank supply depends on accepted stock and net transport, not on
        a provisional layer rebalance. Compose it before that expensive trial.
        """
        p=self.physics;xp=self.xp;c=self.columns;g=p.g
        if self.bank_theta>0:
            self.gradient_engine.launch('exposure_gradient',p.n,p.n,p.z0,g['offset'],g['faces'],g['owner'],g['neighbour'],g['dxdy'],g['inverse'],p.polygon_gradient)
            geom,wet=self.polygon.wet(qnew[0],p.z0,p.polygon_gradient,dry=p.parameters.dry,sedthr=p.sediment_depth)
            p.polygon_wet_geometry[:]=geom;p.polygon_wet_active[:]=wet
            active=c.available+p.class_delta
            if bool(xp.any(active<-1e-12)):raise BendTrialRejected('Negative post-transport bank stock')
            p.class_bank_trial[:]=self.polygon.bank(p.z0,wet,p.class_delta,c.surface_fraction,xp.maximum(active,0.),theta=self.bank_theta)
            return p.class_delta+p.class_bank_trial/g['area']
        return p.class_delta

    def post_transport_columns(self,qnew):
        """Compose separately identified transfers on the one unpublished stock."""
        pass

    def post_redistribution_columns(self,qnew):
        """Reconcile geometric histories after the final material redistribution."""
        pass

    def finish(self,qold,qnew,water,*,morphology=True):
        p=self.physics;xp=self.xp;c=self.columns;g=p.g
        p.class_bank_trial.fill(0);p.class_collapse_trial.fill(0);p.bank_trial_mass.fill(0);p.relaxation_trial_mass.fill(0)
        super().finish(qold,qnew,water,morphology=morphology)
        if not morphology:return p.last
        self.post_transport_columns(qnew)
        if self.collapse:
            mass,sv,ext,basal,dt,_=c.pending
            if getattr(self,'_repose_projection',None) is not None:
                work,packing,move=self._repose_projection.apply(mass,sv,c.rock)
            else:
                work,packing,move=self.polygon.scour(mass,c.rock,slope=self.collapse_slope,
                    column_density=self.material.column,solid_fraction=sv)
            if getattr(self,'_physical_toe_failure',None) is None:p.class_collapse_trial[:]=move
            else:p.class_collapse_trial[:]+=move
            if bool(xp.any(move!=0)):
                targets=c.runtime.targets(qnew[0])
                work,nextsv,base=c.runtime.evaluate(work,packing,xp.zeros_like(p.class_delta),targets,p._dt)
                if self.methods.base_update==1 and bool(xp.any(base!=0)):
                    raise BendTrialRejected('Independent finite base cannot import collapse stock')
                basal=basal+host((base*g['area']).sum(axis=1))
                c.pending_basal=c.pending_basal+base
                residual=host((work*g['area']).sum(axis=(0,2)))-host((c.mass*g['area']).sum(axis=(0,2)))-ext-basal
                if abs(residual).max()>1e-9:raise BendTrialRejected('Polygon class budget failed')
                c.pending=(work,nextsv,ext,basal,dt,residual);c.seal_pending()
        self.post_redistribution_columns(qnew)
        mass,sv,ext,basal,dt,residual=c.pending
        if self.methods.base_update==1 and np.any(basal):
            raise BendTrialRejected('Independent finite base cannot replenish sediment')
        p.ztrial[:]=self.material.bed(mass,sv,c.rock)
        p.bank_trial_mass[:]=p.class_bank_trial.sum(axis=0);p.relaxation_trial_mass[:]=p.class_collapse_trial.sum(axis=0)
        actual=(mass-c.mass).sum(axis=0)*g['area'];expected=(p.class_delta+c.pending_basal)*g['area']+p.class_bank_trial+p.class_collapse_trial
        local=float(xp.max(xp.abs(actual-expected)));internal=float(xp.max(xp.abs((p.class_bank_trial+p.class_collapse_trial).sum(axis=1))))
        if local>1e-9 or internal>1e-9:raise BendTrialRejected('Polygon local transfer closure failed')
        p.last.update(class_process_closure_max_kg=local,class_internal_budget_max_kg=internal,
            bank_transfer_mass_kg=float(xp.maximum(p.class_bank_trial,0.).sum()),
            collapse_transfer_mass_kg=float(xp.maximum(p.class_collapse_trial,0.).sum()),
            class_column_budget_max_kg=float(abs(residual).max()),
            basal_source_max_kg=float(abs(basal).max()),
            maximum_relative_total_bed_increment=float(xp.max(xp.abs(p.ztrial-p.z0)/xp.maximum(qnew[0],p.sediment_depth))))
        return p.last
