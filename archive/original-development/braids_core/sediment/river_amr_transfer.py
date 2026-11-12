"""Private zero-time regrid of the current combined river owner.

Conservative transfer, not a subcycling/reflux controller. Layers remain
index-aligned. Prepared finer rock supplies subcell relief while preserving
parent means. Geometry/physical boundaries must cover the same domain.
"""
from dataclasses import dataclass,asdict
from pathlib import Path
import copy
import hashlib
import numpy as np
import shapely
from shapely.geometry import Polygon
from shapely.strtree import STRtree
from .river_polygon_geometry import polygon_records
from .bed_columns_v101 import host,remap_columns_v101
from .river_columns import RiverColumnOwner
from validation.sediment.river_morphodynamics.sediment_wave import WaveEngine

CELL_DENSITY=('W','M','bedform_state','bedform_chezy')
CELL_TOTAL=('transport_cumulative_mass','class_bank_cumulative','class_collapse_cumulative',
    'bank_cumulative_mass','relaxation_cumulative_mass','suspended_exchange_cumulative',
    'suspended_acceleration','suspended_initial')
FACE_TOTAL=('boundary_ledger','secondary_boundary_ledger','class_boundary_ledger','suspended_boundary_ledger')
GLOBAL=('secondary_source_ledger','secondary_projection_ledger','class_bank_gross','class_collapse_gross','bedform_updates')
SCOPE='current-river-conservative-zero-time-transfer-v6-surveyed-bank-history'


def _coordinate_roundoff_overlap(left,right,area):
    """Recognize only boundary slivers within two coordinate ULPs.

    Projected vertices can round independently on opposite sides of a shared
    edge. This does not change the authoritative finite-volume cell areas.
    The area bound alone is insufficient: both polygon interiors must also
    be disjoint after removing that tiny boundary strip. Never use this to
    repair a geometrically unresolved cell or a genuine nonnested overlap.
    """
    width=2*np.spacing(max(1.,max(abs(v) for v in (*left.bounds,*right.bounds))))
    if (width>=1e-6*np.sqrt(min(left.area,right.area)) or
        area>width*(left.length+right.length)):
        return False
    a=left.buffer(-width);b=right.buffer(-width)
    return not a.is_empty and not b.is_empty and a.intersection(b).area==0.


def validate_wet_dry_transfer(q,secondary,suspended,mapping,dry,xp):
    # An exact one-to-one copy does not coarsen through the wet/dry tolerance.
    # Accepted inundation fronts can contain tiny positive depth and momentum;
    # retaining the finest cell must neither reject them nor zero their water.
    first=mapping.offsets[:-1];ids=mapping.indices[first]
    unchanged=((np.diff(mapping.offsets)==1)&(mapping.weights[first]==1.)&
               (mapping.target_area==mapping.source_area[ids]))
    refined=((np.diff(mapping.offsets)==1)&(mapping.weights[first]==1.)&
             (mapping.target_area<mapping.source_area[ids]))
    guarded=(q[0]<=dry)&~xp.asarray(unchanged)
    # Exact nested splitting of an accepted sub-dry parent retains its water
    # density. It carries no momentum or transported inventory. This exception
    # does not permit coarsening through the wet/dry threshold.
    if (bool(xp.any(q[0]<0)) or bool(xp.any((q[0]>0)&guarded&~xp.asarray(refined))) or
        bool(xp.any(q[1:,guarded])) or bool(xp.any(secondary[guarded])) or
        bool(xp.any(suspended[:,guarded]))):
        raise ValueError('Unsafe wet/dry coarsening; retain refinement')


@dataclass(frozen=True)
class CellMap:
    offsets: np.ndarray
    indices: np.ndarray
    weights: np.ndarray
    source_area: np.ndarray
    target_area: np.ndarray
    identity: str

    @classmethod
    def create(cls,offsets,indices,weights,source_area,target_area):
        off=np.array(offsets,np.int32,copy=True);idx=np.array(indices,np.int32,copy=True);w=np.array(weights,float,copy=True)
        sa=np.array(source_area,float,copy=True);ta=np.array(target_area,float,copy=True)
        if (sa.ndim!=1 or ta.ndim!=1 or not len(sa) or not len(ta) or np.any(sa<=0) or np.any(ta<=0)
            or off.shape!=(len(ta)+1,) or off[0]!=0 or off[-1]!=len(idx) or np.any(np.diff(off)<=0)
            or idx.ndim!=1 or w.shape!=idx.shape or np.any(idx<0) or np.any(idx>=len(sa)) or np.any(w<0)
            or not all(np.isfinite(v).all() for v in (w,sa,ta))):raise ValueError('Invalid conservative cell map')
        row=np.repeat(np.arange(len(ta)),np.diff(off))
        if (not np.allclose(np.bincount(row,weights=w),1,atol=1e-12,rtol=0)
            or not np.allclose(np.bincount(idx,weights=w*ta[row],minlength=len(sa)),sa,atol=1e-13,rtol=1e-11)):
            raise ValueError('Cell map does not partition the same area')
        arrays=(off,idx,w,sa,ta);digest=hashlib.sha256(b''.join(v.tobytes() for v in arrays)).hexdigest()
        for v in arrays:v.setflags(write=False)
        return cls(*arrays,digest)

    @classmethod
    def hierarchy(cls,tree,source,target):
        """Exact nested overlaps from the already validated immutable tree.

        Same ordering/weights as polygon intersections, without GEOS searches.
        Both leaf covers and authoritative area partitions are still checked.
        """
        source=tuple(source);target=tuple(target)
        sc=tree.leaf_cover(source);tc=tree.leaf_cover(target)
        sl={n:i for i,n in enumerate(source)};tl={n:i for i,n in enumerate(target)}
        cells=tree.cells
        sa=np.array([cells[n].area_m2 for n in source]);ta=np.array([cells[n].area_m2 for n in target])
        groups=[[] for _ in target]
        for i,n in enumerate(source):
            if tc[n]>=0:groups[tl[int(tc[n])]].append(i)
        offsets=[0];indices=[];weights=[]
        for j,n in enumerate(target):
            ids=[sl[int(sc[n])]] if sc[n]>=0 else groups[j]
            indices.extend(ids);weights.extend(min(sa[i],ta[j])/ta[j] for i in ids);offsets.append(len(indices))
        return cls.create(offsets,indices,weights,sa,ta)

    @classmethod
    def polygons(cls,source,target,source_area=None,target_area=None):
        source=tuple(source);target=tuple(target)
        # The initial leaf controller authenticates an already identical mesh.
        # Exact vertices AND authoritative areas permit an identity map, but
        # only after geometry/area validation and a cross-cell overlap check.
        # Equality of lists alone must never admit duplicated/overlapping cells.
        same=(source_area is not None and target_area is not None and
              len(source)==len(target) and np.array_equal(source_area,target_area) and
              all(np.array_equal(a,b) for a,b in zip(source,target)))
        sr=polygon_records(source);tr=sr if same else polygon_records(target)
        src=[r[0] for r in sr];dst=src if same else [r[0] for r in tr];tree=STRtree(src)
        off=[0];ids=[];weights=[]
        if not all(r[1] and r[2]>0 for r in sr+tr):raise ValueError('Invalid regrid polygon')
        sa=np.array([r[2] for r in sr]) if source_area is None else np.asarray(source_area)
        ta=np.array([r[2] for r in tr]) if target_area is None else np.asarray(target_area)
        authoritative=source_area is not None or target_area is not None
        if authoritative:
            # GIS vertices rounded at a million-metre origin cannot reproduce
            # subcell areas computed in the preparation's local coordinates.
            # Keep the authoritative solver areas; only admit nested overlaps.
            for records,areas in (((sr,sa),) if same else ((sr,sa),(tr,ta))):
                if areas.shape!=(len(records),):raise ValueError('Polygon authoritative area count differs')
                bounds=np.array([r[3] for r in records],float).reshape(-1,4)
                lengths=np.array([r[4] for r in records])
                tol=16*np.spacing(np.maximum(1.,np.max(np.abs(bounds),axis=1)))*lengths
                if np.any(np.abs(np.array([r[2] for r in records])-areas)>np.maximum(1e-12,tol)):
                    raise ValueError('Polygon differs from authoritative area')
        if same and len(src):
            # Bulk GEOS queries avoid a Python intersection loop over every
            # touching edge. Positive bounding-box overlap is necessary for
            # positive polygon area. Oblique/cropped cells still receive the
            # exact GEOS intersection test whenever their boxes overlap.
            pairs=tree.query(src);pairs=pairs[:,pairs[0]<pairs[1]]
            bounds=np.array([r[3] for r in sr],float)
            if pairs.shape[1]:
                left,right=pairs
                possible=np.all(np.minimum(bounds[left,2:],bounds[right,2:])>
                                np.maximum(bounds[left,:2],bounds[right,:2]),axis=1)
                pairs=pairs[:,possible]
            disjoint=True
            if pairs.shape[1]:
                shapes=np.asarray(src,dtype=object);left,right=pairs
                area=shapely.area(shapely.intersection(shapes[left],shapes[right]))
                actual=shapely.area(shapes)
                suspect=np.flatnonzero(area>1e-13*np.minimum(actual[left],actual[right]))
                disjoint=all(_coordinate_roundoff_overlap(src[left[k]],src[right[k]],area[k])
                             for k in suspect)
            if disjoint:
                n=len(src)
                return cls.create(np.arange(n+1),np.arange(n),np.ones(n),sa,ta)
        for i,p in enumerate(dst):
            for j in sorted(tree.query(p)):
                area=p.intersection(src[j]).area
                if area>1e-13*min(p.area,src[j].area):
                    if authoritative:
                        if _coordinate_roundoff_overlap(p,src[j],area):continue
                        tol=32*np.spacing(max(1.,max(abs(v) for v in (*p.bounds,*src[j].bounds))))*(p.length+src[j].length)
                        if abs(area-min(p.area,src[j].area))>max(1e-12,tol):raise ValueError('Authoritative-area map requires nested cells')
                        area=min(sa[j],ta[i])
                    ids.append(j);weights.append(area/ta[i])
            off.append(len(ids))
        return cls.create(off,ids,weights,sa,ta)


class TransferEngine:
    def __init__(self,backend,threads=4):
        if backend not in ('cpu','gpu'):raise ValueError('Explicit CPU/GPU required')
        self.xp=np
        if backend=='gpu':
            import cupy as cp
            self.xp=cp
        self.engine=WaveEngine(backend,threads,source=Path(__file__).with_suffix('.h'),names=('river_remap',))
        self.identity=self.engine.identity

    def apply(self,value,offsets,indices,weights,nt):
        xp=self.xp;ns=value.shape[-1];shape=(*value.shape[:-1],nt);nv=int(np.prod(value.shape[:-1]))
        data=xp.ascontiguousarray(value,dtype=xp.float64);out=xp.empty(shape,dtype=xp.float64)
        self.engine.launch('river_remap',nt*nv,nt,ns,nv,xp.asarray(offsets),xp.asarray(indices),xp.asarray(weights),data,out)
        return out

    def cells(self,value,mapping,*,extensive=False):
        xp=self.xp
        if value.shape[-1]!=len(mapping.source_area):raise ValueError('Transfer source shape differs')
        v=value/xp.asarray(mapping.source_area) if extensive else value
        out=self.apply(v,mapping.offsets,mapping.indices,mapping.weights,len(mapping.target_area))
        return out*xp.asarray(mapping.target_area) if extensive else out


def boundary_map(old,new):
    """Integrated exterior flux histories; never redistribute them to walls."""
    src=np.flatnonzero(old['neighbour']<0);dst=set(np.flatnonzero(new['neighbour']<0));off=[0];ids=[];w=[];coverage=np.zeros(old['nf']);target_coverage=np.zeros(new['nf'])
    # With optional inlet/outlet finest protection, unchanged faces still
    # survive as an exact permutation rather than requiring interpolation.
    # Keep that identity exact: interval intersections at large GIS coordinates
    # invent tiny overlaps between adjacent unchanged oblique segments.
    def key(data,i):
        return (int(data['bid'][i]),*data['normal'][:,i],*data['mid'][:,i],float(data['length'][i]))
    original={key(old,i):i for i in src};replacement={key(new,j):j for j in dst}
    if len(original)==len(src)==len(dst)==len(replacement) and original.keys()==replacement.keys():
        for j in range(new['nf']):
            if j in dst:ids.append(original[key(new,j)]);w.append(1.)
            off.append(len(ids))
        return np.array(off,np.int32),np.array(ids,np.int32),np.array(w,float)
    # Walls and optionally open sections may change their face partition.
    # Preserve each unchanged face, even when the WHOLE exterior is not a
    # permutation. Otherwise GIS-coordinate roundoff invents overlap with an
    # adjacent unchanged open face and repeatedly grows its integrated ledger.
    exact={j:original[key(new,j)] for j in dst if key(new,j) in original}
    preserved=set(exact.values())
    old_ulps=np.spacing(np.maximum(1.,np.abs(old['mid'])))
    new_ulps=np.spacing(np.maximum(1.,np.abs(new['mid'])))
    for j in range(new['nf']):
        if j in dst:
            if j in exact:
                i=exact[j];ids.append(i);w.append(1.);coverage[i]+=1.;target_coverage[j]=1.
                off.append(len(ids));continue
            for i in src:
                if i in preserved:continue
                # Bid identifies the same prescribed physical boundary; normals
                # must point outward on both representations. Walls have -1.
                if old['bid'][i]!=new['bid'][j] or np.dot(old['normal'][:,i],new['normal'][:,j])<1-1e-10:continue
                normal=old['normal'][:,i];shift=new['mid'][:,j]-old['mid'][:,i]
                # Exported oblique midpoints differ at FP roundoff. Exact GIS
                # line intersection incorrectly reports zero overlapping
                # length. Compare supporting lines, then intersect intervals.
                # Midpoints are serialized in global map coordinates. Rounding
                # each endpoint and then its midpoint can move two fragments of
                # the SAME supporting line by several coordinate ulps. A
                # length-only tolerance incorrectly rejects legal wall regrids
                # at e.g. Swiss LV95 origins. Bound that representation error,
                # not a physical displacement proportional to the map origin.
                # Local coordinates retain the old, much tighter bound.
                ulps=old_ulps[:,i]+new_ulps[:,j]
                line_tolerance=max(1e-10*max(1.,old['length'][i],new['length'][j]),
                                   4.*float(np.abs(normal)@ulps))
                if abs(np.dot(shift,normal))>line_tolerance:continue
                tangent=np.array([-normal[1],normal[0]]);center=np.dot(shift,tangent)
                lo=max(-.5*old['length'][i],center-.5*new['length'][j]);hi=min(.5*old['length'][i],center+.5*new['length'][j]);overlap=max(0.,hi-lo)
                # The same endpoint rounding also invents tiny overlap between
                # ADJACENT segments, or clips a fully nested short fragment.
                # Resolve only these ulp-ambiguous endpoint relationships. Keep
                # authoritative lengths, partial intersections and the strict
                # two-sided coverage check; never normalize a broken partition.
                endpoint_tolerance=max(4.*float(np.abs(tangent)@ulps),
                    32*np.finfo(float).eps*max(1.,abs(center),old['length'][i],new['length'][j]))
                if min(old['length'][i],new['length'][j])>2*endpoint_tolerance:
                    if abs(center)>=.5*(old['length'][i]+new['length'][j])-endpoint_tolerance:
                        overlap=0.
                    elif abs(center)<=.5*abs(old['length'][i]-new['length'][j])+endpoint_tolerance:
                        overlap=min(old['length'][i],new['length'][j])
                if overlap>0:
                    weight=overlap/old['length'][i];ids.append(i);w.append(weight);coverage[i]+=weight;target_coverage[j]+=overlap/new['length'][j]
        off.append(len(ids))
    if not np.allclose(coverage[src],1,atol=1e-9,rtol=0) or not np.allclose(target_coverage[list(dst)],1,atol=1e-9,rtol=0):raise ValueError('Physical boundary partition or identifier changed')
    return np.array(off,np.int32),np.array(ids,np.int32),np.array(w,float)


def exterior_contract(a):
    """Mesh-independent physics; mesh-specific initial/forcing hashes stay in restart.

    Accepted exterior history is transferred, not reinitialized. Spatial forcing
    is checked against physical overlaps separately, never omitted from checks.
    """
    omit={'initial_sha256','forcing_sha256','initial_mass_sha256','initial_sv_sha256','fractions_sha256'}
    def stable(value):
        if isinstance(value,dict):
            # These are indices into the current face array, not a user
            # setting.  A normal-depth outlet deliberately rebuilds its
            # connected face chains after every regrid; transfer below then
            # authenticates its physical-boundary overlap and state.
            return {k:stable(v) for k,v in value.items() if k not in omit|{'section_faces'}}
        if isinstance(value,list):return [stable(v) for v in value]
        if isinstance(value,tuple):return tuple(stable(v) for v in value)
        return value
    return {name:None if name not in a.identity else stable(a.identity[name])
            for name in ('exterior_bed','exterior_sediment','exterior_bedforms')}


def _contract_difference(left,right,path=''):
    """Return the first immutable-contract path that differs, for diagnostics."""
    if isinstance(left,dict) and isinstance(right,dict):
        for key in sorted(set(left)|set(right),key=str):
            if key not in left or key not in right:
                return path+'.'+str(key) if path else str(key)
            found=_contract_difference(left[key],right[key],path+'.'+str(key) if path else str(key))
            if found is not None:return found
        return None
    if left!=right:return path
    return None


def physical_contract(a):
    p=a.physics
    from .river_bank_remap import bank_contract
    return dict(methods=a.methods.fingerprint,controls=asdict(a.controls),bank_theta=a.bank_theta,
        physics_mode=a.identity.get('physics_mode','combined'),
        collapse=a.collapse,collapse_slope=a.collapse_slope,morphology=a.morphology,
        suspension={k:p.identity['river_suspension'][k] for k in ('bed_factor','sus_factor','diffusivity','temperature','cref','settling','formula')},
        suspension_reference=p.identity['river_suspension'].get('reference_parameters'),
        bedforms={k:p.identity['river_bedforms'][k] for k in ('methods','parameters')},
        resolved_model=p.identity.get('resolved_river_model'),
        selected_roughness=p.identity.get('river_roughness'),
        hydraulic_resistance=p.identity.get('hydraulic_resistance'),
        selected_corrections=p.identity.get('river_corrections'),
        public_boundaries=a.identity.get('public_boundaries'),
        material=a.material.identity,exterior=exterior_contract(a),surveyed_bank=bank_contract(a),
        geographic_bank_survey=a.identity.get('geographic_bank_survey'),turbine=a.identity.get('hkt_terrain'),
        sediment_schedule=a.identity.get('sediment_schedule'),
        composition=a.columns.composition.identity,
        dependent_layers=(None if 'river_dependent_layers' not in p.identity else
            {k:p.identity['river_dependent_layers'][k] for k in ('methods','binding','forcing')}),
        secondary=p.identity.get('method'),dry=p.parameters.dry,sediment_depth=p.sediment_depth)


def stage_explicit_clock(source,target):
    """Validate and detach the accepted mesh-independent explicit predictor.

    This is a hint, never a CFL exemption: every stage on the new mesh still
    recomputes its face frequencies and enforces the same hard CFL. No frequency
    array or spatial buffer is transferable. Historical non-explicit owners
    retain their existing remesh behavior.
    """
    from .river_explicit import RiverExplicit, METHOD, MAX_CFL
    from braids_core.hydraulics.river_hllc import RiverHLLC, METHOD as HLLC_METHOD
    methods={RiverExplicit:(METHOD,'river_explicit'),RiverHLLC:(HLLC_METHOD,'river_hllc')}
    old=getattr(source,'midpoint',None);new=getattr(target,'midpoint',None)
    explicit=lambda a,c: (isinstance(c,tuple(methods)) or
        a.identity.get('integrator') in (METHOD,HLLC_METHOD) or any(key in a.identity for _,key in methods.values()))
    if not (explicit(source,old) or explicit(target,new)):return None
    if type(old) not in methods or type(new) is not type(old):
        raise ValueError('Regrid explicit controller changed')
    method,key=methods[type(old)]
    for owner,controller in ((source,old),(target,new)):
        if (owner.identity.get('integrator')!=method or controller.identity.get('method')!=method or
            owner.identity.get(key)!=controller.identity):
            raise ValueError('Regrid explicit method or controller identity changed')
        cfl=controller.cfl
        if (isinstance(cfl,bool) or not np.isfinite(cfl) or not 0<cfl<=MAX_CFL or
            controller.identity.get('cfl')!=cfl):
            raise ValueError('Regrid explicit CFL identity changed')
        controller.validate_state(controller.state())
        if controller.easy_steps!=0:
            raise ValueError('Regrid explicit predictor state is invalid')
    if old.identity!=new.identity or old.cfl!=new.cfl:
        raise ValueError('Regrid explicit controller identity or CFL changed')
    # state() contains only immutable Python scalars. Detach them before any
    # candidate mutation; publication is postponed until transfer succeeds.
    return dict(step_hint=old.step_hint,easy_steps=0)


def verify_boundary_fields(source,target,engine,face):
    """Preserve prescribed intensive fields, not just integrated histories.

    Feeds are solid volume per width/time, concentrations are mass per water
    volume. Neither is a face-integrated inventory. Exterior identities and
    coverage were checked by boundary_map; use overlap / TARGET length here.
    Factories must supply the same physical forcing, never silently overwrite it.
    """
    xp=source.xp;old=source.data;new=target.data;off,ids,weights=face
    rows=np.repeat(np.arange(new['nf']),np.diff(off))
    intensive=weights*old['length'][ids]/new['length'][rows]
    oldbc=host(source.physics.bedbc);newbc=host(target.physics.bedbc)
    if not np.array_equal(oldbc[ids],newbc[rows]):raise ValueError('Sediment boundary kind changed during regrid')
    outside=new['neighbour']<0
    for label,left,right in (('class feed',source.feed,target.feed),
            ('suspended concentration',source.physics.suspended_boundary,target.physics.suspended_boundary)):
        if left.shape[:-1]!=right.shape[:-1]:raise ValueError('Boundary class count changed')
        mapped=engine.apply(left,off,ids,intensive,new['nf'])
        if not bool(xp.allclose(mapped[...,outside],right[...,outside],rtol=1e-12,atol=1e-15)):
            raise ValueError(label+' changed during regrid')
        for value,data in ((left,old),(right,new)):
            forbidden=(data['neighbour']>=0)|(data['bid']<0)
            if bool(xp.any(value[...,forbidden])):raise ValueError(label+' assigned to internal/closed faces')
    return dict(policy='same physical boundary kind and overlap-weighted intensive class forcing',
        nonzero_class_feed=bool(xp.any(source.feed)),
        nonzero_suspended_concentration=bool(xp.any(source.physics.suspended_boundary)))


def stage_exterior_history(source,target,engine,face):
    """Candidate-only physical-overlap transfer of zero-area boundary history.

    No physical inventory or time update occurs here. Preserve layer mass and
    bulk thickness along the boundary; averaging packing alone would not do so.
    All array arithmetic remains on the explicitly selected backend.
    """
    p,t,xp=source.physics,target.physics,source.xp
    for name in ('exterior_bed','exterior_sediment','exterior_bedforms'):
        if hasattr(p,name)!=hasattr(t,name):raise ValueError('Exterior state owner changed during regrid')
    if not hasattr(p,'exterior_bed'):return {},None
    old,new=source.data,target.data;off,ids,weights=face
    rows=np.repeat(np.arange(new['nf']),np.diff(off))
    intensive=weights*old['length'][ids]/new['length'][rows]
    if not np.array_equal(host(p.exterior_mode)[ids],host(t.exterior_mode)[rows]):
        raise ValueError('Exterior bed method changed during regrid')
    forcing=engine.apply(p.exterior_value,off,ids,intensive,new['nf'])
    if not bool(xp.allclose(forcing,t.exterior_value,rtol=1e-12,atol=1e-15)):
        raise ValueError('Exterior bed forcing changed during regrid')
    bed=engine.apply(p.exterior_bed,off,ids,intensive,new['nf'])
    if not bool(xp.all(xp.isfinite(bed))):raise ValueError('Nonfinite exterior bed remap')
    state=dict(exterior_bed=bed,exterior_trial=bed.copy())
    receipt=dict(policy='physical-boundary overlap; intensive zero-area history',
        physical_inventory_added_kg=0.,time_advance_s=0.,composition=False)
    if not hasattr(p,'exterior_sediment'):return state,receipt
    left,right=p.exterior_sediment,t.exterior_sediment
    sf,tf=host(left.faces),host(right.faces)
    reverse=np.full(old['nf'],-1,np.int32);reverse[sf]=np.arange(len(sf))
    co=[0];ci=[];cw=[]
    for j in tf:
        section=slice(off[j],off[j+1]);mapped=reverse[ids[section]]
        if not len(mapped) or np.any(mapped<0):raise ValueError('Exterior history crosses an internal/closed face')
        ci.extend(mapped);cw.extend(intensive[section]);co.append(len(ci))
    co=np.asarray(co,np.int32);ci=np.asarray(ci,np.int32);cw=np.asarray(cw,float)
    cr=np.repeat(np.arange(len(tf)),np.diff(co))
    if not np.array_equal(host(left.mode)[ci],host(right.mode)[cr]):
        raise ValueError('Exterior composition method changed during regrid')
    def mapped(v):return engine.apply(v,co,ci,cw,len(tf))
    if not bool(xp.allclose(mapped(left.fractions),right.fractions,rtol=1e-12,atol=1e-15)):
        raise ValueError('Exterior composition forcing changed during regrid')
    if p.exterior_mass.shape[:-1]!=t.exterior_mass.shape[:-1]:
        raise ValueError('Exterior layer/class count changed during regrid')
    rho=source.material.column.reshape(1,-1,1)
    mass=mapped(p.exterior_mass)
    oldsolid=(p.exterior_mass/rho).sum(1);solid=(mass/rho).sum(1)
    bulk=mapped(oldsolid/p.exterior_sv)
    sv=xp.where(bulk>0,solid/xp.maximum(bulk,1e-300),mapped(p.exterior_sv))
    if (bool(xp.any(~xp.isfinite(mass))) or bool(xp.any(mass<0))
            or bool(xp.any(~xp.isfinite(sv))) or bool(xp.any(sv<=0)) or bool(xp.any(sv>1+1e-12))):
        raise ValueError('Invalid exterior column remap')
    sv=xp.minimum(sv,1.) # Only a possible summation-roundoff excess above one.
    sl=xp.asarray(old['length'][sf]);tl=xp.asarray(new['length'][tf])
    mass_before=(p.exterior_mass*sl).sum(2);mass_after=(mass*tl).sum(2)
    volume_before=(oldsolid/p.exterior_sv*sl).sum(1);volume_after=(solid/sv*tl).sum(1)
    if (not bool(xp.allclose(mass_before,mass_after,rtol=1e-12,atol=1e-12))
            or not bool(xp.allclose(volume_before,volume_after,rtol=1e-12,atol=1e-14))):
        raise ValueError('Exterior boundary-history integral changed during regrid')
    state.update(exterior_mass=mass,exterior_mass_trial=mass.copy(),exterior_sv=sv,exterior_sv_trial=sv.copy())
    if hasattr(p,'exterior_bedforms'):
        state['exterior_bedform_state']=mapped(p.exterior_bedform_state)
        state['exterior_bedform_chezy']=mapped(p.exterior_bedform_chezy)
        state['exterior_bedform_updates']=p.exterior_bedform_updates.copy()
        receipt['bedforms']='physical-overlap intensive equilibrium components; no prediction during transfer'
    receipt.update(composition=True,source_faces=len(sf),target_faces=len(tf),
        layer_class_line_mass_residual_kg_m=host(mass_after-mass_before).tolist(),
        layer_bulk_line_volume_residual_m2=host(volume_after-volume_before).tolist())
    return state,receipt


def owner_polygons(a):
    return a.polygon.polygons


def refinement_subset(mapping):
    """Extract complete split parents, leaving unchanged/coarsened rows alone.

    A mixed transaction is nested only if every overlap contains either the
    entire source or the entire target cell. Crossing partitions are rejected.
    The returned submap independently partitions all selected parent areas.
    """
    m=mapping;row=np.repeat(np.arange(len(m.target_area)),np.diff(m.offsets))
    overlap=m.weights*m.target_area[row]
    if not np.all(np.isclose(overlap,m.source_area[m.indices],rtol=1e-10,atol=1e-13)|
                  np.isclose(overlap,m.target_area[row],rtol=1e-10,atol=1e-13)):
        raise ValueError('Regrid requires nested source/target cells')
    split=m.target_area[row]<m.source_area[m.indices]*(1-1e-10)
    targets=np.unique(row[split]);sources=np.unique(m.indices[split])
    if not len(targets):return None
    if np.any(np.diff(m.offsets)[targets]!=1):raise ValueError('Incomplete nested refinement group')
    parents=m.indices[m.offsets[targets]]
    sub=CellMap.create(np.arange(len(targets)+1),np.searchsorted(sources,parents),
        m.weights[m.offsets[targets]],m.source_area[sources],m.target_area[targets])
    return sources,targets,sub


def transfer(source,target,mapping,*,cancelled=lambda:False,reject_after_candidate=False,hierarchy_geometry=None):
    """Populate a fresh private target; source never mutated, no time advance.

    Callers publish the returned target only after success. Older checkpoints
    retain their original owners. This path requires the complete current
    split-capacity/layer/polygon-bank/bedform bundle, on the chosen backend.
    """
    if cancelled():raise InterruptedError('Regrid cancelled')
    verified=CellMap.create(mapping.offsets,mapping.indices,mapping.weights,mapping.source_area,mapping.target_area)
    if verified.identity!=mapping.identity:raise ValueError('Regrid mapping was mutated')
    if hierarchy_geometry is None:
        geometry=CellMap.polygons(owner_polygons(source),owner_polygons(target),host(source.columns.area),host(target.columns.area))
    else:
        tree,old_leaves,new_leaves=hierarchy_geometry;cells=tree.cells
        # Authenticate actual owner polygons against tree nodes, not a caller's
        # assertion that a cached map is trustworthy. No physical data cached.
        for owner,leaves in ((source,old_leaves),(target,new_leaves)):
            polygons=owner_polygons(owner)
            if len(polygons)!=len(leaves) or any(not np.array_equal(p,cells[n].vertices_xy) for p,n in zip(polygons,leaves)):
                raise ValueError('Regrid owner polygons differ from hierarchy geometry')
        geometry=CellMap.hierarchy(tree,old_leaves,new_leaves)
    if (not np.array_equal(geometry.offsets,mapping.offsets) or not np.array_equal(geometry.indices,mapping.indices)
        or not np.allclose(geometry.weights,mapping.weights,atol=1e-10,rtol=0)):
        raise ValueError('Regrid map crosses actual polygon geometry')
    source.columns.validate();target.columns.validate();p=source.physics;t=target.physics;xp=source.xp
    if source.backend!=target.backend:raise ValueError('Regrid cannot change CPU/GPU backend')
    if source.columns.pending is not None or p._open or target.columns.pending is not None or t._open:raise ValueError('Regrid requires accepted owners')
    if target.time or target.steps:raise ValueError('Regrid target must be fresh and unpublished')
    source_contract=physical_contract(source);target_contract=physical_contract(target)
    if source_contract!=target_contract:
        # A regrid must not be allowed to alter the selected physical model.
        # Name the differing immutable inputs: a bare failure here made a
        # preparation/runtime mismatch impossible to diagnose from a run log.
        changed=[name for name in source_contract if source_contract[name]!=target_contract.get(name)]
        detail=[_contract_difference(source_contract[name],target_contract.get(name),name) or name for name in changed]
        raise ValueError('Regrid physics changed: '+', '.join(detail))
    explicit_clock=stage_explicit_clock(source,target)
    if not np.allclose(host(source.columns.area),mapping.source_area,atol=1e-13,rtol=1e-11) or not np.allclose(host(target.columns.area),mapping.target_area,atol=1e-13,rtol=1e-11):raise ValueError('Regrid actual areas differ')
    # No material/forcing change may hide behind a mesh change.
    for name in ('forcing','mode','reversal_policy'):
        if not np.array_equal(source.data[name],target.data[name]):raise ValueError('Boundary forcing changed during regrid')
    engine=TransferEngine(source.backend,p.flow.threads)
    face=boundary_map(source.data,target.data)
    boundary_receipt=verify_boundary_fields(source,target,engine,face)
    exterior_state,exterior_receipt=stage_exterior_history(source,target,engine,face)
    columns=remap_columns_v101(source.columns,target.columns.runtime,mapping.offsets,mapping.indices,mapping.weights,mapping.target_area)
    # Reuse qualified native/CUDA mass-volume arithmetic, NOT its historical
    # bounded lifecycle. Preserve the current owner's clocks and global ledger.
    rebuilt=RiverColumnOwner(target.columns.runtime,columns.mass,columns.sv,columns.rock,columns.area,time_s=source.columns.time_s)
    rebuilt.updates=source.columns.updates
    for name in ('reference','external','basal'):setattr(rebuilt,name,getattr(source.columns,name).copy())
    rebuilt._seal()
    q=engine.cells(source.q,mapping);state={k:engine.cells(getattr(p,k),mapping) for k in CELL_DENSITY}
    state.update(exterior_state)
    state['suspended_mass']=engine.cells(p.suspended_mass,mapping)
    if hasattr(p,'suspended_diameter') != hasattr(t,'suspended_diameter'):
        raise ValueError('Remesh cannot change the suspended-diameter state owner')
    if hasattr(p,'suspended_diameter'):
        state['suspended_diameter']=engine.cells(p.suspended_diameter,mapping)
        if bool(xp.any(~xp.isfinite(state['suspended_diameter']))) or bool(xp.any(state['suspended_diameter']<=0.)):
            raise ValueError('Invalid suspended-diameter remesh')
        state['suspended_diameter_trial']=state['suspended_diameter'].copy()
    terrain=None;subset=refinement_subset(mapping)
    if subset is not None:
        from .river_terrain_transfer import TerrainEngine
        te=TerrainEngine(source.backend,p.flow.threads)
        si,ti,sub=subset;si=xp.asarray(si);ti=xp.asarray(ti)
        thickness=(rebuilt.bed-rebuilt.rock)[ti]
        carried=xp.concatenate((p.W.reshape(1,-1),p.suspended_mass))
        r,qt,ct,terrain=te.prolong(sub,source.columns.rock[si],target.columns.rock[ti],thickness,source.q[:,si],carried[:,si],dry=p.parameters.dry)
        rock=rebuilt.rock.copy();rock[ti]=r;q[:,ti]=qt
        state['W'][ti]=ct[0];state['suspended_mass'][:,ti]=ct[1:]
        terrain.update(refined_sources=len(si),refined_targets=len(ti),subset_map_sha256=sub.identity)
        candidate=RiverColumnOwner(target.columns.runtime,rebuilt.mass,rebuilt.sv,rock,rebuilt.area,time_s=rebuilt.time_s)
        candidate.updates=rebuilt.updates
        for name in ('reference','external','basal'):setattr(candidate,name,getattr(rebuilt,name).copy())
        candidate._seal();rebuilt=candidate
    state.update({k:engine.cells(getattr(p,k),mapping,extensive=True) for k in CELL_TOTAL})
    from .river_bank_remap import stage_bank_history
    bank_state,bank_receipt=stage_bank_history(source,target,engine,mapping,rebuilt,q)
    state.update(bank_state)
    for name in FACE_TOTAL:
        value=getattr(p,name)
        if bool(xp.any(value[...,source.data['neighbour']>=0])):raise ValueError('Boundary ledger has internal entries')
        state[name]=engine.apply(value,*face,target.data['nf'])
    water=engine.apply(source.water_ledger,*face,target.data['nf'])
    for name in GLOBAL:state[name]=getattr(p,name).copy()
    # Coarsening through the dry tolerance cannot silently discard momentum,
    # hI or suspension. A controller must retain refinement for such groups.
    validate_wet_dry_transfer(q,state['W'],state['suspended_mass'],mapping,p.parameters.dry,xp)
    before=host((source.columns.mass*source.columns.area).sum((0,2))+(p.suspended_mass*source.columns.area).sum(1))
    after=host((rebuilt.mass*rebuilt.area).sum((0,2))+(state['suspended_mass']*rebuilt.area).sum(1))
    water_res=float(xp.sum(q[0]*rebuilt.area)-xp.sum(source.q[0]*source.columns.area))
    mom_res=host((q[1:]*rebuilt.area).sum(1)-(source.q[1:]*source.columns.area).sum(1))
    secondary_res=float(xp.sum(state['W']*rebuilt.area)-xp.sum(p.W*source.columns.area))
    # The old and new totals are independent parallel reductions.  In a
    # terrain-aware split, the child-volume solve and two GPU reductions add
    # several legal rounding stages before the comparison.  This allowance is
    # still only roundoff-scale (2048 eps), not a physical mass tolerance.
    eps=2048*np.finfo(float).eps
    mass_tol=max(1e-9,eps*float(np.max(abs(before))))
    water_tol=max(1e-10,eps*float(xp.sum(xp.abs(source.q[0])*source.columns.area)))
    momentum_tol=max(1e-10,eps*float(xp.sum(xp.abs(source.q[1:])*source.columns.area)))
    secondary_tol=max(1e-10,eps*float(xp.sum(xp.abs(p.W)*source.columns.area)))
    if np.max(abs(after-before))>mass_tol or abs(water_res)>water_tol or np.max(abs(mom_res))>momentum_tol or abs(secondary_res)>secondary_tol:
        raise ValueError('Regrid conservation failed: mass='+str(float(np.max(abs(after-before))))+'/'+str(mass_tol)+
            ', water='+str(water_res)+'/'+str(water_tol)+', momentum='+str(host(mom_res).tolist())+'/'+str(momentum_tol)+
            ', secondary='+str(secondary_res)+'/'+str(secondary_tol))
    if cancelled():raise InterruptedError('Regrid cancelled before publication')
    if reject_after_candidate:raise RuntimeError('Injected private regrid rejection')
    # The existing column object is captured by suspension closures: replace
    # its contents, not the object, or exchange would use stale source stocks.
    target.columns.__dict__.clear();target.columns.__dict__.update(rebuilt.__dict__)
    target.q[:]=q;target.z[:]=rebuilt.bed;t.rock[:]=rebuilt.rock;t.z0[:]=target.z;t.ztrial[:]=target.z
    if target.backend=='gpu':target.hydraulics.rock[:]=t.rock
    for name,value in state.items():getattr(t,name)[...]=value
    target.water_ledger[:]=water
    target.solver.time=source.time;target.solver.steps=source.steps;t.time=p.time;t.steps=p.steps
    target.solver.last=copy.deepcopy(source.last);t.last=copy.deepcopy(p.last)
    receipt=dict(scope=SCOPE,backend=source.backend,source_cells=p.n,target_cells=t.n,map_sha256=mapping.identity,
        time_s=source.time,accepted_updates=source.steps,class_residual_kg=(after-before).tolist(),water_residual_m3=water_res,
        momentum_residual=mom_res.tolist(),secondary_residual=secondary_res,zero_time=True,constitutive_updates=0,
        column_owner='RiverColumnOwner',runtime=engine.identity,wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source_column_sha256=source.columns.state_sha256,
        prolongation=terrain,restriction='actual-area layer-aligned conservative average',boundary_forcing=boundary_receipt,
        exterior_history=exterior_receipt,surveyed_bank_history=bank_receipt)
    if explicit_clock is not None:
        receipt['explicit_clock']=dict(step_hint_s=explicit_clock['step_hint'],
            source='accepted source controller',stage_cfl_rechecked=True,spatial_buffers_transferred=False)
    target.identity['river_regrid']=dict(history=copy.deepcopy(source.identity.get('river_regrid',{}).get('history',[]))+[receipt])
    if explicit_clock is not None:target.midpoint.restore(explicit_clock)
    return target,receipt
