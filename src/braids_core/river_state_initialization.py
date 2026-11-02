"""Explicit conservative saved-state mapping, independent of exact restart."""
from pathlib import Path
from copy import deepcopy
import io
import numpy as np
import shapely
from shapely.geometry import Polygon
from shapely.strtree import STRtree
from . import river_state_snapshot as snapshot
from .river_project import identity,read_json

SCHEMA='braids.state-initialization.v1'


def _mesh_identity(polygons,areas):
    keys=[shapely.normalize(Polygon(p)).wkb for p in polygons]
    if len(set(keys))!=len(keys):raise ValueError('Duplicated saved-state cells')
    return keys


def match_cells(source,target,source_area,target_area):
    a=_mesh_identity(source,source_area);b=_mesh_identity(target,target_area)
    lookup={k:i for i,k in enumerate(a)}
    if len(a)!=len(b) or any(k not in lookup for k in b):return None
    order=np.array([lookup[k] for k in b],dtype=np.int64)
    if not np.allclose(np.asarray(source_area)[order],target_area,rtol=1e-12,atol=1e-12):return None
    return order


def overlap_map(source,target,source_area,target_area):
    """True polygon overlaps, balanced against authoritative solver areas.

    A common local origin and tiny row/column balancing handle rounded GIS
    coordinates without admitting uncovered cells or changing the fluid domain.
    """
    from .sediment.river_amr_transfer import CellMap
    origin=np.floor(np.min(np.concatenate(source+target),axis=0))
    src=np.array([Polygon(np.asarray(p)-origin) for p in source],object)
    dst=np.array([Polygon(np.asarray(p)-origin) for p in target],object)
    if any(not p.is_valid or p.area<=0 for p in [*src,*dst]):raise ValueError('Invalid mapping polygons')
    tree=STRtree(src);row,col=tree.query(dst,predicate='intersects')
    area=shapely.area(shapely.intersection(dst[row],src[col]));keep=area>1e-13*np.minimum(source_area[col],target_area[row])
    row=row[keep];col=col[keep];area=area[keep];ns=len(src);nt=len(dst)
    ra=np.bincount(row,area,minlength=nt);ca=np.bincount(col,area,minlength=ns)
    if (np.any(abs(ra/target_area-1)>1e-8) or np.any(abs(ca/source_area-1)>1e-8)):
        raise ValueError('Source and target cells must partition the same unchanged fluid domain; mapping has uncovered or overlapping area')
    # The corrections are at coordinate-roundoff scale, not a substitute for
    # real overlap. Each iteration preserves nonnegative overlap weights.
    for _ in range(500):
        area*=source_area[col]/np.bincount(col,area,minlength=ns)[col]
        area*=target_area[row]/np.bincount(row,area,minlength=nt)[row]
        if np.max(abs(np.bincount(col,area,minlength=ns)/source_area-1))<5e-12:break
    order=np.lexsort((col,row));row=row[order];col=col[order];area=area[order]
    offsets=np.r_[0,np.cumsum(np.bincount(row,minlength=nt))].astype(np.int32)
    return CellMap.create(offsets,col,area/target_area[row],source_area,target_area)


def apply_map(value,mapping):
    a=np.asarray(value);shape=(*a.shape[:-1],len(mapping.target_area));out=np.empty(shape)
    for j,(lo,hi) in enumerate(zip(mapping.offsets[:-1],mapping.offsets[1:])):
        out[...,j]=np.sum(a[...,mapping.indices[lo:hi]]*mapping.weights[lo:hi],axis=-1)
    return out


def _grains(project):
    c=project['settings']['columns']
    return tuple(c['diameter_m']),tuple(c['solid_density_kg_m3'])


def prepare(prepared,source_path,target_path,*,allow_mapping=False):
    """Return a mapping-required decision; never map without explicit opt-in."""
    from .river_worker import load_prepared
    from .sediment import adaptive_river_worker as core
    prepared=Path(prepared);project,receipt=load_prepared(prepared)
    raw=Path(source_path).read_bytes();meta,source=snapshot.decode(raw);old=meta['project']
    snapshot.same_domain(old,project)
    if project['settings']['end_time_s']<=meta['time_s']:raise ValueError(f"Set End time above the checkpoint time ({meta['time_s']:g} s). The saved time is retained.")
    tree=snapshot._tree(prepared)
    with np.load(prepared/'core/prepared/initial.npz',allow_pickle=False) as f:initial={k:f[k].copy() for k in f.files}
    profile=read_json((prepared/'core/prepared/profile.json').read_text(encoding='utf-8'))
    survey=read_json((prepared/'BANK_SURVEY.json').read_text(encoding='utf-8'));turbines=read_json((prepared/'TURBINES.json').read_text(encoding='utf-8'))
    pins=set(() if survey is None else survey['mapping']['required_leaves'])|set(() if turbines is None else turbines['required_leaves'])
    leaves=core.initial_leaves(profile,initial,tree,tuple(sorted(pins)));mesh=tree.select(leaves)
    target_polys=[np.asarray(c.vertices_xy) for c in mesh.cells];target_area=np.array([c.area_m2 for c in mesh.cells])
    source_polys=snapshot.polygons(source);order=match_cells(source_polys,target_polys,source['area'],target_area)
    decision=dict(schema=SCHEMA,project_id=identity(project),prepared_id=identity(receipt),source_sha256=snapshot.sha_bytes(raw),
        time_s=meta['time_s'],mesh_changed=order is None,source_cells=len(source_polys),target_cells=len(target_polys),leaves=list(leaves))
    target_path=Path(target_path)
    if target_path.is_file():
        existing,arrays=load(target_path,project,receipt)
        if all(existing[k]==v for k,v in decision.items()):return dict(existing,status='ready',path=str(target_path))
        raise ValueError('Existing initialized state belongs to another preparation/source; choose a new target')
    if order is None and not allow_mapping:return dict(decision,status='mapping_required',message='The prepared mesh differs. Click Map saved state to prepared mesh, then Run.')
    mapping=None if order is not None else overlap_map(source_polys,target_polys,source['area'],target_area)
    transform=(lambda v:np.asarray(v)[...,order].copy()) if order is not None else (lambda v:apply_map(v,mapping))
    loaded=['water depth','hydraulic discharge','bed elevation'];initialized=[];used={'q','z'}
    q=transform(source['q']);z=transform(source['z']);n=len(leaves)
    r={k:core.restrict(tree,v)[...,list(leaves)] for k,v in initial.items() if k!='fractions'}
    th=r['thickness'];vol=core.restrict(tree,initial['fractions']*initial['thickness'][:,None,:])[...,list(leaves)]
    ng=initial['fractions'].shape[1]
    fractions=np.divide(vol,th[:,None,:],out=np.full((len(th),ng,n),1./ng),where=th[:,None,:]>0)
    fields=dict(q=q,z=z,rock=z-th.sum(0),thickness=th,fractions=fractions,concentration=np.zeros((ng,n)),area=target_area)
    s=old['model']['physics']['sediment'];t=project['model']['physics']['sediment']
    same_grains=_grains(old)==_grains(project)
    bed_declaration_same=(old['settings']['columns']==project['settings']['columns']
                and old['initial']['layer_thickness_m']==project['initial']['layer_thickness_m']
                and old['initial']['layer_fractions']==project['initial']['layer_fractions']
                and old.get('material_zones',{}).get('erodible',[])==project.get('material_zones',{}).get('erodible',[]))
    columns_ok=(s is not None and t is not None and bed_declaration_same
                and s['fraction_basis']==t['fraction_basis'] and 'mass' in source and source['mass'].shape[:2]==(len(th),ng))
    if columns_ok:
        rho=np.asarray(project['settings']['columns']['solid_density_kg_m3'])
        mass=transform(source['mass']);source_bulk=(source['mass']/rho[None,:,None]).sum(1)/source['sv']
        thick=transform(source_bulk);solid=(mass/rho[None,:,None]).sum(1)
        sv=np.divide(solid,thick,out=transform(source['sv']),where=thick>0)
        fraction_volume=mass/rho[None,:,None]
        fractions=np.divide(fraction_volume,solid[:,None,:],out=np.full_like(mass,1./ng),where=solid[:,None,:]>0)
        fields.update(mass=mass,sv=sv,rock=z-thick.sum(0),thickness=thick,fractions=fractions)
        loaded.append('compatible bed layers and class stocks')
        used.update(('mass','sv','rock'))
    else:initialized.append('bed-layer thickness/composition from target settings; rock shifted to retain the saved bed')
    if s and t and same_grains and s['suspension']!='off' and t['suspension']!='off' and 'suspended' in source:
        suspended=transform(source['suspended']);fields['concentration']=np.divide(suspended,q[0],out=np.zeros_like(suspended),where=q[0]>0)
        loaded.append('suspended class mass')
        used.add('suspended')
    elif t and t['suspension']!='off':
        fields['concentration']=r['concentration'];initialized.append('suspended concentration from target initial settings')
    oh=old['model']['physics']['hydrodynamics'];nh=project['model']['physics']['hydrodynamics']
    from .river_jiang_cao import active as jc
    if not jc(old['model']) and not jc(project['model']) and oh['secondary_flow']==nh['secondary_flow'] and nh['secondary_flow']!='secondary.off' and 'W' in source:
        fields['W']=transform(source['W']);loaded.append('compatible secondary-current intensity')
        used.add('W')
    elif nh['secondary_flow']!='secondary.off':initialized.append('secondary-current state starts from zero')
    if jc(project['model']) and jc(old['model']) and same_grains and 'jc_state' in source:
        state=transform(source['jc_state']);state[:3]=q;state[5]=z;fields['jc_state']=state
        loaded.append('Jiang-Cao mobile sediment and secondary state')
        initialized=[v for v in initialized if v!='secondary-current state starts from zero']
        if bed_declaration_same:
            loaded.append('compatible Jiang-Cao bed inventory')
            initialized=[v for v in initialized if not v.startswith('bed-layer thickness/composition')]
        else:
            c=project['settings']['columns'];state[4]=c['dry_density_kg_m3'][0]/c['solid_density_kg_m3'][0]*fields['thickness'].sum(0)
        used.add('jc_state')
    elif jc(project['model']):initialized.append('Jiang-Cao mobile inventory starts at zero; bed stock uses target erodible depth')
    ignored=sorted(set(source)-{'area','vertices','offsets'}-used)
    basis='Jiang-Cao bulk mixture depth/discharge' if jc(old['model']) else 'Shallow-water depth/discharge'
    volume_name='mixture_column_volume_difference_m3' if jc(old['model']) else 'water_volume_difference_m3'
    decision.update(loaded=loaded,initialized=initialized,not_carried=ignored,
        histories='New budgets, numerical predictors, turbulence/bedform and bank-failure memory; exact restart preserves these when selected.',
        source_hydraulic_basis=basis,
        transition_assumption='Preserve free-surface depth and bulk velocity; phase inventories follow the target model.' if jc(old['model'])!=jc(project['model']) else 'Same hydraulic state definition.',
        mapping='exact cell permutation/copy' if mapping is None else 'conservative polygon overlaps',
        mapping_sha256=None if mapping is None else mapping.identity,
        conservation={volume_name:float(np.dot(q[0],target_area)-np.dot(source['q'][0],source['area'])),
            'momentum_difference':(q[1:]@target_area-source['q'][1:]@source['area']).tolist()})
    fields['leaves']=np.asarray(leaves,np.int64)
    # Geometry-dependent initial physics must see the imported bed, rather
    # than the DEM that happened to be used to prepare the target mesh.
    fine=finest_fields(tree,initial,fields);bed=fine['rock']+fine['thickness'].sum(0)
    from .river_normal_slope import prepare_slopes
    decision['normal_slopes']=prepare_slopes(project['settings']['hydraulics'],tree.finest,bed)
    decision['turbine_hub_elevations_m']={} if turbines is None else {
        r['name']:float(max(bed[r['hub_reference_finest_cells']])+r['hub_height_m']) for r in turbines['rotors']}
    raw=snapshot.encode(decision,fields);target_path.parent.mkdir(parents=True,exist_ok=True)
    from braids_gpu.atomic_output import write_bytes_atomic
    write_bytes_atomic(target_path,raw)
    return dict(decision,status='ready',path=str(target_path))


def load(path,project,receipt):
    raw=Path(path).read_bytes() if not isinstance(path,bytes) else path
    a=snapshot.archive(raw);m=read_json(str(a.pop('metadata')))
    if m.get('schema')!=SCHEMA or m.get('project_id')!=identity(project) or m.get('prepared_id')!=identity(receipt):raise ValueError('Initialized state differs from the current prepared setup; inspect/map the saved state again')
    if set(m['hashes'])!=set(a) or any(snapshot.array_hash(v)!=m['hashes'][k] for k,v in a.items()):raise ValueError('Initialized-state checksum differs')
    if any(not np.isfinite(v).all() for v in a.values()) or np.any(a['q'][0]<0):raise ValueError('Invalid initialized fields')
    n=m['target_cells'];nl=len(project['initial']['layer_thickness_m']);ng=len(project['settings']['columns']['diameter_m'])
    shapes=dict(q=(3,n),z=(n,),rock=(n,),thickness=(nl,n),fractions=(nl,ng,n),concentration=(ng,n),area=(n,),leaves=(n,))
    if any(k not in a or a[k].shape!=v for k,v in shapes.items()) or a['leaves'].dtype.kind not in 'iu' or len(set(a['leaves']))!=n:raise ValueError('Invalid initialized field shapes or leaves')
    if np.any(a['area']<=0) or np.any(a['thickness']<0) or np.any(a['fractions']<0) or np.any(a['concentration']<0) or not np.allclose(a['fractions'].sum(1),1.,rtol=0,atol=1e-10):raise ValueError('Invalid initialized stocks or fractions')
    if project['settings']['end_time_s']<=m['time_s']:raise ValueError('End time must exceed saved-state time')
    return m,a


def finest_fields(tree,initial,state):
    """Constant hierarchy prolongation, not an extra spatial interpolation."""
    leaves=tuple(map(int,state['leaves']));cover=tree.leaf_cover(leaves);lookup={node:i for i,node in enumerate(leaves)}
    order=np.array([lookup[int(cover[i])] for i in range(tree.offsets[-2],tree.count)])
    return {k:state[k][...,order].copy() for k in ('q','rock','thickness','fractions','concentration')}


def set_clock(owner,time_s,*,jiang_cao=False):
    owner.solver.time=float(time_s)
    if not jiang_cao:
        owner.physics.time=float(time_s);owner.columns.time_s=float(time_s);owner.columns._seal()
