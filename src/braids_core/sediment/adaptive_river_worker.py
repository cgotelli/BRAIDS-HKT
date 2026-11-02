"""Prepared-input common-clock river worker, explicit CPU/CUDA selection.

The combined split-capacity/layer/bank/suspension/bedform bundle uses its
shared explicit SSPRK22 f-wave owner on both backends. No method or backend
fallback. Input geometry and fields are not laboratory globals.
"""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from braids_gpu.atomic_output import save_arrays_atomic,write_json_atomic
from braids_core.windows_paths import filesystem_path
from braids_core.windows_native_io import native_directory
from .bed_columns_v101 import ColumnMethodsV101,host,initialize_columns_v101
from .current_river_solver import CurrentRiverControls
from .current_river_worker import _member
from .river_field_policy import AutomaticRiver,FixedRiver,FieldSettings
from .river_sediment_clock import next_hint
from .river_leaf_controller import RiverLeafController
from .river_leaf_io import PreparedLeafCache,MeshEpochWriter,sha

SCHEMA='braids.combined_adaptive_river.v1'
ROOT=Path(__file__).resolve().parents[2]


def check_stop(stop):
    if stop is not None and filesystem_path(stop).exists():raise InterruptedError('Adaptive river worker cancelled')


def wait_for_mesh_export(proc,stop,timeout_s=240.):
    """Wake at exporter completion, retaining bounded Stop/timeout checks.

    Sleeping after each poll imposed a full 50 ms delay even on tiny exports.
    A timed process wait returns immediately on completion; it does not bypass
    cancellation, exit-code checking or the caller's termination cleanup.
    """
    began=time.monotonic()
    while proc.poll() is None:
        check_stop(stop)
        remaining=timeout_s-(time.monotonic()-began)
        if remaining<=0:raise TimeoutError('Native mesh export timed out')
        try:proc.wait(timeout=min(.05,remaining))
        except subprocess.TimeoutExpired:pass


def validate_profile(p):
    expected={'schema','domain','assignments','grid','levels','controls','columns','hydraulics',
        'sediment_boundaries','bank','suspension','adaptation','adaptive','end_time_s','dt_max_s','output_interval_s'}
    if set(p)-{'physics_mode','breaklines','solids','sediment_start_time_s','local_refinement'}!=expected or p['schema']!=SCHEMA:raise ValueError('Adaptive river profile schema/fields differ')
    start=p.get('sediment_start_time_s',0.)
    if not np.isfinite(start) or not 0<=start<=p['end_time_s']:raise ValueError('Sediment start time must be between zero and end time')
    if 'solids' in p:
        from braids_core.boundaries.solids import validate_solids
        validate_solids(p['solids'])
    if 'breaklines' in p:
        from braids_core.boundaries.breaklines import validate_breaklines
        validate_breaklines(p['breaklines'])
    mode=p.get('physics_mode','combined')
    if mode not in ('hydrodynamics','bedload','combined'):raise ValueError('Unknown adaptive river physics mode')
    c=CurrentRiverControls(**p['controls']);c.validate();m=ColumnMethodsV101(**p['columns'])
    FieldSettings(**p['adaptation'])
    if type(p['adaptive']) is not bool:raise ValueError('Explicit adaptation switch required')
    if type(p['levels']) is not int or not 1<=p['levels']<=6:raise ValueError('Worker supports 1..6 nested 2:1 levels')
    clocks=[p[k] for k in ('end_time_s','dt_max_s','output_interval_s')]
    if not np.isfinite(clocks).all() or min(clocks)<=0:raise ValueError('Positive finite clocks required')
    if c.morfac!=1 and p['bank']['bank_theta']:
        from .bank_setup import validate_inventory
        validate_inventory(p['columns'])
    if (m.bed not in ('mixed','layered')
        or m.transport_layer not in ('constant','depth')):
        raise ValueError('Current finite mixed/layered material bundle required')
    from .river_material import RiverMaterial
    RiverMaterial.from_columns(m)
    if set(p['bank'])!={'bank_theta','collapse','collapse_slope','repose_angle_deg'}:raise ValueError('Explicit bank controls required')
    b=p['bank']
    if type(b['collapse']) is not bool or not np.isfinite([b['bank_theta'],b['collapse_slope'],b['repose_angle_deg']]).all() or not 0<=b['bank_theta']<=1 or b['collapse_slope']<=0 or not 0<b['repose_angle_deg']<90:
        raise ValueError('Invalid bank controls')
    s=p['suspension']
    if set(s)!={'bed_factor','sus_factor','diffusivity','temperature','cref'} or not np.isfinite(list(s.values())).all():raise ValueError('Explicit finite suspension controls required')
    if s['bed_factor'] not in (0,1) or s['sus_factor'] not in (0,1) or s['diffusivity']<0 or s['temperature']<=-20 or s['cref']<=0:raise ValueError('Invalid suspension controls')
    if set(p['hydraulics'])!=set(p['sediment_boundaries']):raise ValueError('Define sediment on every hydraulic boundary')
    if mode!='combined':
        # The hydrodynamics-only research route may compose the same
        # conservative horizontal-viscosity owner used by the public resolved
        # model.  Bedload-only remains an intentionally auxiliary-free route.
        if mode=='hydrodynamics' and c.viscosity and c.cfl>.5:
            raise ValueError('Simple mode above CFL 0.5 requires all auxiliary processes OFF')
        viscosity_invalid = bool(c.viscosity) and mode!='hydrodynamics'
        if (len(m.fraction_ids)!=1 or c.betac or viscosity_invalid or c.diffusivity
            or b['bank_theta'] or b['collapse'] or s['sus_factor'] or s['diffusivity']
            or s['bed_factor']!=1):raise ValueError('Simple mode requires one grain, bedload only, all auxiliary processes OFF')
        for row in p['sediment_boundaries'].values():
            if any(row['concentration_kg_m3']) or (mode=='hydrodynamics' and any(row.get('solid_flux_m2_s',[]))):
                raise ValueError('Disabled sediment process has nonzero boundary forcing')
    # Constant forcing only; never ignore a hydrograph or duplicate native grains.
    for row in p['hydraulics'].values():
        allowed={'mode','discharge_m3_s'} if row.get('mode')=='total_discharge' else {'mode','stage_m'}
        if set(row)!=allowed or row.get('mode') not in ('total_discharge','stage'):raise ValueError('Worker currently admits constant total discharge / stage boundaries only')
    return c,m


def hierarchy(p,stop=None):
    from braids_core.boundaries.domain import PolygonDomain,BoundaryAssignment
    from braids_core.boundaries.inlet_io import NativeGrid
    domain=PolygonDomain(**p['domain']);assignments=[BoundaryAssignment(**a) for a in p['assignments']];grid=NativeGrid(**p['grid'])
    # One area-constrained geometry route, including domains with no internal
    # constraints. The minimum is the finest FULL cell area, not the coarse
    # nominal area. Parents are unions of actual children, never re-clipped
    # independently into a new collection of arbitrarily small wall slivers.
    from braids_core.boundaries.breaklines import build_breakline_hierarchy
    from braids_core.boundaries.solids import validate_solids
    return build_breakline_hierarchy(domain,assignments,grid,p.get('breaklines',[]),levels=p['levels'],
        local_refinement=p.get('local_refinement',()),
        obstacles=validate_solids(p.get('solids',[]),domain),
        cancelled=lambda:stop is not None and filesystem_path(stop).exists())


def arrays_valid(a,t,m):
    n=len(t.finest.cells);ng=len(m.fraction_ids)
    if set(a)!={'q','rock','thickness','fractions','concentration'}:raise ValueError('Initial field schema differs')
    if a['q'].shape!=(3,n) or a['rock'].shape!=(n,) or a['concentration'].shape!=(ng,n):raise ValueError('Finest field shapes differ')
    if a['thickness'].ndim!=2 or a['thickness'].shape[1]!=n:raise ValueError('Layer/cell thickness array required')
    m.targets(np.zeros(1),a['thickness'].shape[0])
    if not all(np.isfinite(v).all() for v in a.values()):raise ValueError('Nonfinite initial fields')
    if np.any(a['thickness']<0) or np.any(a['q'][0]<0) or np.any(a['concentration']<0):raise ValueError('Invalid initial stock/depth/concentration')
    dry=a['q'][0]<=1e-6
    if np.any(a['q'][1:,dry]) or np.any(a['concentration'][:,dry]):raise ValueError('Dry initial cells cannot carry momentum/concentration')
    initialize_columns_v101(m,a['thickness'],a['fractions'])


def restrict(t,v):return np.moveaxis(t.restrict(np.moveaxis(v,-1,0)),0,-1)


def initial_leaves(p,initial,t,required=()):
    """Depth-only dynamic or terrain-only static, with physical geometry pins."""
    fine=tuple(range(t.offsets[-2],t.count))
    settings=FieldSettings(**p['adaptation'])
    wanted=np.zeros(t.count,dtype=int)
    if p['adaptive']:
        active=initial['q'][0]>0
        for f in t.finest.faces:
            if f.neighbour<0 and f.kind!='wall':active[f.owner]=True
        for _ in range(settings.buffer_cells):
            expanded=active.copy()
            for f in t.finest.faces:
                if f.neighbour>=0 and (active[f.owner] or active[f.neighbour]):
                    expanded[f.owner]=expanded[f.neighbour]=True
            active=expanded
        wanted[np.asarray(fine)[active]]=t.finest_level
    from .river_terrain_refinement import terrain_features
    features=terrain_features(t,initial['rock']+initial['thickness'].sum(0),
        settings,threads=p['controls']['threads'])
    wanted[features['nodes']]=t.finest_level
    from .river_field_policy import refinement_leaves
    leaves=refinement_leaves(t,wanted,settings,required)
    if not set(features['nodes']).issubset(leaves):
        raise ValueError('Hierarchy refinement cap prevents finest terrain protection')
    return leaves


def boundary_arrays(d,mesh,p,ng):
    active=sorted({f.bc_id for f in mesh.faces if f.neighbour<0 and f.kind!='wall'})
    if set(active)!=set(p['sediment_boundaries']):raise ValueError('Physical boundary IDs differ')
    bc=np.zeros(d['nf'],np.int32);feed=np.zeros((ng,d['nf']));conc=np.zeros_like(feed)
    for bid,name in enumerate(active):
        row=p['sediment_boundaries'][name];kind=row.get('kind')
        expected={'kind','concentration_kg_m3'}|({'solid_flux_m2_s'} if kind=='feed' else set())
        if kind not in ('feed','outflow') or set(row)!=expected:raise ValueError('Explicit class feed/outflow definition required')
        faces=(d['neighbour']<0)&(d['bid']==bid)
        if not np.any(faces):raise ValueError('Prepared boundary missing: '+name)
        bc[faces]=1 if kind=='feed' else 2
        for field,out in [('concentration_kg_m3',conc)]+([('solid_flux_m2_s',feed)] if kind=='feed' else []):
            value=np.asarray(row[field],float)
            if value.shape!=(ng,) or not np.isfinite(value).all() or np.any(value<0):raise ValueError('Ordered nonnegative per-class forcing required')
            out[:,faces]=value[:,None]
    return bc,feed,conc


def cache_for(root,p,a,t,stop=None,progress=lambda phase:None,*,initial_leaf_state=None):
    from braids_core.boundaries.cut_cell_io import write_cut_case,CutBoundary,CutSediment
    from braids_gpu.graded_native_exporter import EXPORTER_SHA256
    from . import river_mesh_records
    from .river_runtime_cache import prepare_shared
    c,m=validate_profile(p);ng=len(m.fraction_ids)
    # Restrict layer volumes, NOT fractions independently from their stock.
    r={k:restrict(t,v) for k,v in a.items() if k!='fractions'}
    r['fraction_volume']=restrict(t,a['fractions']*a['thickness'][:,None,:])
    r['suspended']=restrict(t,a['concentration']*a['q'][0])
    def prepare(folder,leaves,mesh):
        check_stop(stop);progress('mesh_preparation');ids=list(leaves);q=r['q'][:,ids];th=r['thickness'][:,ids];rock=r['rock'][ids]
        frac=np.divide(r['fraction_volume'][:,:,ids],th[:,None,:],out=np.full((len(th),ng,len(ids)),1./ng),where=th[:,None,:]>0);z=rock+th.sum(0)
        ci=np.divide(r['suspended'][:,ids],q[0],out=np.zeros((ng,len(ids))),where=q[0]>1e-6)
        special=initial_leaf_state if (initial_leaf_state is not None and tuple(leaves)==tuple(initial_leaf_state['leaves'])) else None
        if special is not None:
            q=special['q'].copy();rock=special['rock'].copy();th=special['thickness'].copy()
            frac=special['fractions'].copy();ci=special['concentration'].copy();z=special['z'].copy()
        # Geometry-only native exporter carrier: sediment is disabled here.
        # Its fixed dry density reconstructs the requested bulk bed elevation;
        # the class owner below replaces this placeholder with explicit stock.
        initial=np.zeros((len(ids),5));initial[:,:3]=q.T;initial[:,4]=1590*th.sum(0)
        # This legacy deck is a zero-time geometry/state carrier. Its exporter
        # admits the old native CFL range; that unused time-loop setting is not
        # the public SSPRK22 controller. The requested CFL remains in PROFILE,
        # the prepared signature and RiverExplicit's actual runtime identity.
        write_cut_case(folder/'case.data',mesh,initial,rock,{k:CutBoundary(**v) for k,v in p['hydraulics'].items()},
            end_time_s=p['end_time_s'],dt_max_s=p['dt_max_s'],output_interval_s=p['output_interval_s'],cfl=min(.5,c.cfl),scheme=6,
            dry_tolerance_m=1e-6,bedload=0,exchange=0,manning=.125**(1/6)/c.chezy,
            sediment=CutSediment(d50_m=c.diameter,d90_m=1.5*c.diameter,minimum_bedload_depth_m=.001,roughness_mode='manning'),
            compact_sidecars=True,source_sha256={'adaptive_worker':sha(__file__)})
        # The same native initializer computes geometry/reconstruction records,
        # but returns them through memory callbacks rather than a subprocess
        # and binary-file round trip. CPU hydraulics still uses the exact deck.
        d,record_identity=river_mesh_records.initialize(folder/'case.data',
            threads=8,cancelled=lambda:check_stop(stop))
        check_stop(stop);bc,feed,conc=boundary_arrays(d,mesh,p,ng)
        fields=dict(q=q,rock=rock,z=z,thickness=th,fractions=frac,concentration=ci,
            bedbc=bc,feed=feed,boundary_concentration=conc)
        if special is not None:
            for name in ('mass','sv','W','jc_state'):
                if name in special:fields[name]=special[name].copy()
        write_json_atomic(folder/'RECORDS.json',record_identity)
        prepare_shared(root,folder,d,cancelled=lambda:check_stop(stop));check_stop(stop)
        # The cache seals private detached copies and serializes exactly those
        # copies once for fresh-process restart; this process does not reread.
        return dict(data=d,fields=fields)
    signature=dict(schema=SCHEMA,worker_sha256=sha(__file__),runtime_cache_sha256=sha(Path(__file__).with_name('river_runtime_cache.py')),profile=p,
        fields_sha256={k:__import__('hashlib').sha256(v.tobytes()).hexdigest() for k,v in a.items()},
        record_initializer_sha256=sha(river_mesh_records.__file__),exporter_reference=EXPORTER_SHA256)
    if initial_leaf_state is not None:signature['initialized_leaf_fields']={k:__import__('hashlib').sha256(v.tobytes()).hexdigest() for k,v in initial_leaf_state.items()}
    return PreparedLeafCache(root,signature,prepare,cancelled=lambda:check_stop(stop))


def prepare(profile,fields,output,*,stop_file=None,required_leaves=()):
    check_stop(stop_file);p=json.loads(json.dumps(profile));c,m=validate_profile(p)
    t=hierarchy(p,stop_file);a={k:np.asarray(v,float) for k,v in fields.items()};arrays_valid(a,t,m)
    # Validate boundary semantics before any expensive native construction.
    from braids_gpu.atomic_output import write_json_atomic
    output=filesystem_path(output);output.mkdir(parents=True,exist_ok=False)
    write_json_atomic(output/'profile.json',p);save_arrays_atomic(output/'initial.npz',**a)
    cache=cache_for(output/'cache',p,a,t,stop_file,lambda phase:write_json_atomic(output/'PROGRESS.json',dict(phase=phase,percent=50)))
    fine=initial_leaves(p,a,t,required_leaves);cache.get(fine,t.select(fine));check_stop(stop_file)
    write_json_atomic(output/'PREPARED.json',dict(schema=SCHEMA,worker_sha256=sha(__file__),
        files={n:sha(output/n) for n in ('profile.json','initial.npz')},cells=len(fine),QGIS_admitted=False))
    write_json_atomic(output/'PROGRESS.json',dict(phase='prepared',percent=100))
    return output


def load_prepared(folder,stop=None):
    folder=filesystem_path(folder);receipt=json.loads((folder/'PREPARED.json').read_text())
    if receipt.get('schema')!=SCHEMA or receipt.get('worker_sha256')!=sha(__file__):raise ValueError('Prepared worker identity differs')
    if set(receipt.get('files',{}))!={'profile.json','initial.npz'}:raise ValueError('Prepared receipt differs')
    for n,h in receipt['files'].items():
        if sha(_member(folder,n))!=h:raise ValueError('Prepared artifact changed: '+n)
    p=json.loads((folder/'profile.json').read_text());c,m=validate_profile(p);check_stop(stop)
    with np.load(folder/'initial.npz',allow_pickle=False) as f:a={k:f[k] for k in f.files}
    t=hierarchy(p,stop);arrays_valid(a,t,m)
    return p,a,t


def factory(backend,p,t,cache,work,stop=None,*,resolved_model=None,process_parameters=None):
    from braids_gpu import general_live_transport as transport
    from .river_transport import build_transport_river
    from .river_polygon_banks import RiverPolygonBankSolver
    from .river_columns import RiverColumnOwner
    from .river_suspension import install_suspension
    from .river_bedforms import install_bedforms
    from .river_native_lifecycle import release_loaded_search_directories
    from .river_runtime_cache import source_folder
    if backend not in ('cpu','gpu'):raise ValueError('Explicit CPU/GPU backend required')
    selected=None
    if resolved_model is not None:
        if process_parameters is None:
            raise ValueError('Explicit slope and hiding parameters required for the current river project')
        from braids_core.river_project import validate_binding
        resolved_model=validate_binding(p,resolved_model)
        if resolved_model['request']['execution']['backend']!=backend:
            raise ValueError('Backend differs from the explicit project selection')
        selected=resolved_model['physics']['sediment']
    c,m=validate_profile(p);count=[0];survey=None
    if process_parameters is not None and process_parameters.get('bank_survey') is not None:
        from braids_core.river_bank_survey import BankSurveyMap
        from braids_core.river_model import SURVEY_BANKS
        if selected is None or selected['banks'] not in SURVEY_BANKS:
            raise ValueError('Survey input supplied without its explicit bank method')
        doc=process_parameters['bank_survey'];survey=BankSurveyMap.from_document(doc,doc['definition'])
    if selected is not None:
        from braids_core.river_model import SURVEY_BANKS
        if selected['banks'] in SURVEY_BANKS and survey is None:raise ValueError('Prepared geographic bank survey required')
    if resolved_model is None and (any(r!=2650 for r in m.solid_density_kg_m3)
                                  or any(r!=1590 for r in m.dry_density_kg_m3)):
        raise ValueError('Explicit resolved capacity/material binding required for nonstandard material')
    turbines=None if process_parameters is None else process_parameters.get('turbines')
    def make(leaves,mesh):
        check_stop(stop);inp=cache.get(leaves,mesh);count[0]+=1;folder=filesystem_path(work)/f'owner_{count[0]:04d}'
        d,f=cache.load_payload(inp)
        sources=source_folder(inp)
        if backend=='cpu':transport.verify_sources(sources)
        ctx=nullcontext(None) if backend=='cpu' else transport.constructors(sources,d,integrator='ssp33')
        from . import river_turbine_flow as turbine_flow
        with (turbine_flow.native_context(backend) if turbines else nullcontext()),ctx as Solver:
            a=build_transport_river(inp/'case.data',d,backend=backend,controls=c,bedbc=f['bedbc'],feed=np.zeros(d['nf']),
                work_directory=folder,gpu_solver_type=Solver,source_folder=sources if backend=='gpu' else None)
        a=RiverPolygonBankSolver(a,m,f['thickness'],f['fractions'],f['feed'],polygons=[cell.vertices_xy for cell in mesh.cells],**p['bank'])
        col=a.columns;rebuilt=RiverColumnOwner(col.runtime,f.get('mass',col.mass),f.get('sv',col.sv),f['rock'],col.area)
        col.__dict__.clear();col.__dict__.update(rebuilt.__dict__)
        a.q[:]=a.xp.asarray(f['q']);a.z[:]=col.bed;a.physics.rock[:]=col.rock;a.physics.z0[:]=a.z;a.physics.ztrial[:]=a.z
        if backend=='gpu':a.hydraulics.rock[:]=col.rock
        if selected is not None and selected['avalanche']=='redistribution.jiang_cao_repose_exner223_v1':
            from .river_repose import install as install_repose
            install_repose(a,p['bank']['repose_angle_deg'])
        mode=p.get('physics_mode','combined')
        suspension_on=(mode=='combined' if resolved_model is None else
            selected is not None and selected['suspension']!='off')
        bedforms_on=(mode=='combined' if resolved_model is None else
            selected is not None and selected['bedforms']!='off')
        install_suspension(a,folder,initial_concentration=f['concentration'],boundary_concentration=f['boundary_concentration'],
            enabled=suspension_on,capacity=None if selected is None else selected['capacity'],
            hiding='none' if selected is None else selected['capacity_hiding'].removeprefix('hiding.'),
            corrections=None if selected is None else (selected,process_parameters),
            **p['suspension'])
        if resolved_model is not None:
            from .river_roughness import install as install_roughness
            from braids_core.river_material_zones import cell_values
            method=resolved_model['physics']['hydrodynamics']['roughness']
            from .river_roughness import KEYS
            zones=process_parameters.get('material_zones',{}).get('roughness',[])
            spatial=cell_values(mesh,zones,process_parameters['roughness'][KEYS[method]],method=method) if zones else None
            install_roughness(a,folder,method=resolved_model['physics']['hydrodynamics']['roughness'],
                parameters=process_parameters['roughness'],bedforms_enabled=bedforms_on,spatial=spatial,zones=zones)
            from .river_hydraulic_drag import install as install_hydraulic_drag
            install_hydraulic_drag(a,folder)
        else:
            install_bedforms(a,folder,enabled=bedforms_on)
        a.morphology=True
        if resolved_model is not None:
            if selected is not None:
                from .river_dependent_layers import install as install_dependent_layers
                install_dependent_layers(a,selected,process_parameters['dependent_layers'])
            from .river_selected_physics import install
            install(a,resolved_model)
            from .river_wet_avalanche import install as install_wet_avalanche
            install_wet_avalanche(a,selected,process_parameters['wet_avalanche'])
            from braids_core.river_boundary import install as install_transport_boundaries
            install_transport_boundaries(a,process_parameters['transport_boundaries'],d)
            from braids_core.river_schedule import install as install_schedules
            install_schedules(a,process_parameters['sediment_schedules'],d)
            from braids_core.river_open_boundary import install as install_open_boundaries
            install_open_boundaries(a,process_parameters['boundaries'],folder/'exterior')
        elif mode!='combined':
            from .river_simple_physics import install
            install(a,mode)
        if survey is not None:
            survey.install(a,leaves,mesh)
            a._physical_toe.identity['public_support']=True
            if getattr(a,'_physical_toe_failure',None) is not None:
                a._physical_toe_failure.identity['public_support']=True
        if turbines:
            if selected is not None:raise ValueError('Turbines require hydrodynamics only')
            a.morphology=False
            turbine_flow.install(a,mesh.cells,turbines['rotors'],folder/'turbine')
            a._survey_required_leaves=tuple(sorted(set(getattr(a,'_survey_required_leaves',()))|set(turbines['required_leaves'])))
        a.cancelled=lambda:stop is not None and filesystem_path(stop).exists()
        from braids_gpu.graded_native_exporter import EXPORTER_SHA256
        a.identity['adaptive_worker']=dict(schema=SCHEMA,worker_sha256=sha(__file__),runtime_cache_sha256=sha(Path(__file__).with_name('river_runtime_cache.py')),profile=p,fields_sha256=sha(inp/'fields.npz'),
            scope='explicit_fwave_common_clock',QGIS_admitted=False,
            carrier_role='zero_time_geometry_and_spatial_inputs',
            carrier_cfl=float(d['cfl_target']),executed_face_sum_cfl=c.cfl,
            native_exporter_sha256=EXPORTER_SHA256,
            native_records_identity_sha256=sha(inp/'RECORDS.json'))
        if 'W' in f:a.physics.W[:]=a.xp.asarray(f['W'])
        release_loaded_search_directories(a);check_stop(stop)
        return a
    make.required_leaves=tuple(sorted(set(() if survey is None else survey.required_leaves)|set(() if turbines is None else turbines['required_leaves'])))
    return make


def fields(a):
    p=a.physics
    result={k:host(v).copy() for k,v in dict(q=a.q,z=a.z,mass=a.columns.mass,rock=a.columns.rock,sv=a.columns.sv,W=p.W,
        suspended=p.suspended_mass,bedforms=p.bedform_state,bank=p.class_bank_cumulative,collapse=p.class_collapse_cumulative,
        bank_gross=p.class_bank_gross,collapse_gross=p.class_collapse_gross,water=a.water_ledger,bed_boundary=p.class_boundary_ledger,
        sus_boundary=p.suspended_boundary_ledger,suspended_initial=p.suspended_initial,acceleration=p.suspended_acceleration,area=a.columns.area).items()}
    if getattr(a,'_physical_toe_failure',None) is not None:
        for name in ('bank_failure_history','bank_failure_count','bank_failure_class_cumulative'):
            result[name]=host(getattr(p,name)).copy()
    # Output-only least-squares reconstruction of the LAST accepted, limited
    # face transport. Does not reevaluate capacity or modify solver state.
    d=a.data;o=np.asarray(d['owner']);j=np.asarray(d['neighbour']);normal=host(p.g['normal']);length=host(p.g['length']);n=p.n
    owners=np.r_[o,j[j>=0]];nx=np.r_[normal[0],normal[0,j>=0]];ny=np.r_[normal[1],normal[1,j>=0]];w=np.r_[length,length[j>=0]]
    xx=np.bincount(owners,w*nx*nx,minlength=n);xy=np.bincount(owners,w*nx*ny,minlength=n);yy=np.bincount(owners,w*ny*ny,minlength=n)
    dt=float(getattr(p,'_dt',0.));flux=host(p.class_face_integral).sum(0)/(dt*p.morfac) if dt>0 and a.morphology else np.zeros(p.nf)
    f=np.r_[flux,flux[j>=0]];bx=np.bincount(owners,w*nx*f,minlength=n);by=np.bincount(owners,w*ny*f,minlength=n);det=xx*yy-xy*xy
    result['bedload_vector_m2_s']=np.vstack((np.divide(yy*bx-xy*by,det,out=np.zeros(n),where=det>0),np.divide(xx*by-xy*bx,det,out=np.zeros(n),where=det>0)))
    result['bedload_sample_dt_s']=np.array(dt)
    result.update(time=a.time,steps=a.steps,basal_ledger=a.columns.basal.copy());return result


def diagnostics(a,water_initial):
    xp=a.xp;p=a.physics;area=a.columns.area
    residual=((a.columns.mass.sum(0)+p.suspended_mass)*area).sum(1)-p.suspended_initial.sum(1)-a.material.solid*p.class_boundary_ledger.sum(1)-p.suspended_boundary_ledger.sum(1)-p.suspended_acceleration.sum(1)-xp.asarray(a.columns.basal)
    water=float(xp.sum(a.q[0]*area)-water_initial-xp.sum(a.water_ledger))
    wall=(a.data['neighbour']<0)&(a.data['bid']<0)
    walls=[float(xp.max(xp.abs(v[...,wall]))) if np.any(wall) else 0. for v in (a.water_ledger,p.class_boundary_ledger,p.suspended_boundary_ledger)]
    r=dict(class_residual_kg=host(residual).tolist(),water_residual_m3=water,wall_integrals=walls,
        minimum_stock=float(xp.min(a.columns.mass)),minimum_suspension=float(xp.min(p.suspended_mass)),bank_gross_kg=host(p.class_bank_gross).tolist(),
        basal_source_kg=a.columns.basal.tolist(),
        morphological_factor=float(p.morfac),
        suspended_acceleration_kg=host(xp.sum(p.suspended_acceleration,axis=1)).tolist())
    eps=64*np.finfo(float).eps
    class_tol=max(1e-8,eps*float(xp.max(xp.sum(xp.abs(p.suspended_initial),axis=1))))
    from .river_budget_roundoff import cumulative_water_roundoff
    current_water=float(xp.sum(a.q[0]*area));boundary_work=float(xp.sum(xp.abs(a.water_ledger)))
    single_water_tol=max(1e-9,eps*max(abs(water_initial),abs(current_water),boundary_work))
    water_tol=cumulative_water_roundoff(abs(water_initial),abs(current_water),boundary_work,a.steps)
    r.update(class_roundoff_limit_kg=class_tol,water_roundoff_limit_m3=water_tol,
             water_single_reduction_limit_m3=single_water_tol,
             water_arithmetic_accepted_steps=int(a.steps))
    if getattr(a,'_physical_toe_failure',None) is not None:
        r['retained_bank_failure']=dict(block_count=int(xp.sum(p.bank_failure_count)),
            transferred_kg=float(xp.sum(xp.maximum(p.bank_failure_class_cumulative,0.))))
    if getattr(a,'_repose_projection',None) is not None:
        r['repose']=dict(method=a._repose_projection.identity['method'],
            angle_deg=float(np.rad2deg(a._repose_projection.material.repose_angle_rad)),
            last_projection={k:p.last[k] for k in ('repose_stock_constrained','repose_maximum_slope','repose_iterations') if k in p.last})
    if max(abs(host(residual)))>class_tol or abs(water)>water_tol or any(walls) or min(r['minimum_stock'],r['minimum_suspension'])<0:raise RuntimeError('Combined worker budget/wall/stock check failed: '+str(r))
    return r


def read_live_frame(output):
    """Read only a fully published frame and its exact geometry epoch."""
    root=filesystem_path(output)/'frames';doc=json.loads((root/'INDEX.json').read_text());row=doc['frames'][-1]
    fp=_member(root,row['file']);mp=_member(root,row['epoch'])
    if sha(fp)!=row['sha256'] or sha(mp)!=doc['epochs'][row['epoch']]:raise ValueError('Live frame/mesh checksum differs')
    with np.load(fp,allow_pickle=False) as f:values={k:f[k] for k in f.files}
    with np.load(mp,allow_pickle=False) as f:mesh={k:f[k] for k in f.files}
    if str(values.pop('mesh_epoch'))!=row['epoch'] or values['q'].shape[-1]!=len(mesh['leaves']):raise ValueError('Live frame geometry differs')
    return row,mesh,values


from .river_runtime_lease import leased_runtime


def output_clock_tolerance(time):
    """Resolve administrative clocks at floating-point, not fixed, precision.

    A 1e-12 s output tail at t=210 s is not a meaningful sediment update:
    it removes stock limiting from the last-step flux sampled for plotting.
    Never change the physical CFL or advance the state without integration.
    Publish the accepted state within roundoff of the requested output time.
    """
    return max(1e-12,64*np.finfo(float).eps*max(1.,abs(float(time))))


def output_horizon(dt,time,event):
    """Integrate to the exact event on the last meaningful step, not a tail."""
    remaining=event-time
    return remaining if 0<remaining<=dt+output_clock_tolerance(event) else dt


class AcceptedCheckpointClock:
    """Administrative publication, never a timestep or AMR event.

    Mutating transactions invalidate an in-process generation before they
    start. This also covers a cancelled assessment and zero-time remeshing;
    accepted step count alone cannot identify the checkpoint's policy state.
    A failed outer publication can retry without reserializing the raw state.
    """
    def __init__(self,interval,path,on_checkpoint=None):
        if type(interval) not in (int,float) or not np.isfinite(interval) or interval<=0:
            raise ValueError('Positive finite checkpoint interval required')
        self.interval=float(interval);self.path=filesystem_path(path)
        self.on_checkpoint=on_checkpoint;self.generation=0
        self.raw_generation=None;self.notified_generation=None
        self.next_time=None;self.rows=[];self.receipt=None

    def invalidate(self):
        self.generation+=1

    def ensure(self,driver,*,force=False):
        a=driver.controller.owner
        if not force and self.next_time is not None and a.time+1e-12<self.next_time:
            return False
        if self.raw_generation!=self.generation:
            driver.checkpoint(self.path)
            self.raw_generation=self.generation
            self.receipt=dict(time_s=a.time,steps=a.steps,generation=self.generation,
                next_adaptation_s=driver.next_time,adaptation_events=len(driver.events))
            self.rows.append(dict(self.receipt))
        if self.notified_generation!=self.raw_generation:
            if self.on_checkpoint is not None:self.on_checkpoint(self.path,dict(self.receipt))
            self.notified_generation=self.raw_generation
        self.next_time=(np.floor((a.time+1e-12)/self.interval)+1)*self.interval
        return True


@leased_runtime
def run(prepared,output,*,backend,restart=None,until=None,stop_file=None,on_frame=None,resolved_model=None,checkpoint_path=None,on_created=None,process_parameters=None,
        checkpoint_interval_s=60.,on_checkpoint=None,on_checkpoint_state=None,initial_state=None):
    from .bend_methods_v1 import BendTrialRejected
    if backend not in ('cpu','gpu'):raise ValueError('Explicit CPU/GPU backend required')
    output=filesystem_path(output);output.mkdir(parents=True,exist_ok=False);began=time.perf_counter();driver=None;rows=[];rejected=[];laststep=-1;status='failed';error=None
    if on_created is not None:on_created()
    checkpoint_file=output/'checkpoint.npz' if checkpoint_path is None else filesystem_path(checkpoint_path)
    if not checkpoint_file.resolve().is_relative_to(output.resolve()):raise ValueError('Checkpoint must remain in this run')
    checkpoint_file.parent.mkdir(parents=True,exist_ok=True)
    def checkpoint_notice(path,row):
        if on_checkpoint_state is not None:
            controller=driver.controller
            on_checkpoint_state(path,row,controller.owner,controller._mesh_cache[controller.leaves])
        if on_checkpoint is not None:on_checkpoint(path,row)
    checkpoints=AcceptedCheckpointClock(checkpoint_interval_s,checkpoint_file,checkpoint_notice)
    now=[0.];end=[1.]
    hydraulic=dict(accepted_substeps=0,internal_retries=0,accuracy_retries=0,nonlinear_retries=0,cfl_retries=0,spatial_retries=0,accepted_cfl_max=0.,
        fwave_evaluations=0,minimum_substep_s=None,next_substep_hint_s=None,maximum_temporal_error_ratio=0.)
    def progress(phase):write_json_atomic(output/'PROGRESS.json',dict(status='running',phase=phase,backend=backend,time_s=now[0],
        hydraulic=dict(hydraulic),
        percent=5 if phase=='loading' else 15 if phase=='initial_owner' else min(100,20+80*now[0]/end[0])))
    def publish():
        nonlocal laststep
        a=driver.controller.owner
        if a.steps!=laststep:
            saved=fields(a);saved['initial_bed_m']=reference[list(driver.controller.leaves)].copy()
            writer.write(driver.controller,saved);laststep=a.steps
            row=dict(time_s=a.time,steps=a.steps,cells=len(driver.controller.leaves),**diagnostics(a,initial_water));rows.append(row)
            write_json_atomic(output/'DIAGNOSTICS.json',rows)
            if hasattr(a,'hkt_diagnose'):
                import csv,io
                from braids_gpu.atomic_output import write_bytes_atomic
                previous=getattr(publish,'turbine_rows',[])
                for diagnostic in a.hkt_diagnose():
                    turbine=dict(time_s=a.time,**diagnostic)
                    turbine['thrust_N']=diagnostic['force_per_density_m4_s2']*1000.
                    turbine['useful_power_W']=diagnostic['useful_power_per_density_m5_s3']*1000.
                    turbine['hydraulic_work_W']=diagnostic['hydraulic_power_removed_per_density_m5_s3']*1000.
                    previous.append(turbine)
                publish.turbine_rows=previous
                write_json_atomic(output/'TURBINE_PERFORMANCE.json',previous)
                stream=io.StringIO(newline='');csvwriter=csv.DictWriter(stream,fieldnames=list(turbine));csvwriter.writeheader();csvwriter.writerows(previous)
                write_bytes_atomic(output/'TURBINE_PERFORMANCE.csv',stream.getvalue().encode('utf-8'))
                write_json_atomic(output/'TURBINE_PERFORMANCE_INDEX.json',dict(schema='braids.turbine-performance.v1',rows=len(previous),
                    files={name:sha(output/name) for name in ['TURBINE_PERFORMANCE.json','TURBINE_PERFORMANCE.csv']}))
            if on_frame:on_frame(output,row)
    try:
        check_stop(stop_file);progress('loading');p,initial,t=load_prepared(prepared,stop_file);end[0]=p['end_time_s']
        initial_meta,initial_leaf_state=(None,None) if initial_state is None else initial_state
        if initial_state is not None:
            from braids_core.river_state_initialization import finest_fields
            initial=finest_fields(t,initial,initial_leaf_state)
        target=p['end_time_s'] if until is None else float(until)
        if not np.isfinite(target) or target<=0 or (target>p['end_time_s'] and restart is None):raise ValueError('Extended target requires a restart')
        end[0]=target
        cache=cache_for(filesystem_path(prepared)/'cache',p,initial,t,stop_file,progress,
            initial_leaf_state=initial_leaf_state)
        build=factory(backend,p,t,cache,output/'runtime',stop_file,resolved_model=resolved_model,
            process_parameters=process_parameters)
        def make(leaves,mesh):
            owner=build(leaves,mesh)
            if initial_state is not None and driver is None:
                from braids_core.river_state_initialization import set_clock
                set_clock(owner,initial_meta['time_s'])
            # Activation is an accepted clock state, not a change of physical
            # model. Construct regrid targets in the source's current phase so
            # the unmodified strict transfer contract can compare them. Restart
            # subsequently restores its authenticated clock/phase from payload.
            if resolved_model is not None:
                owner.morphology=(driver.controller.owner.morphology if driver is not None else
                    bool(resolved_model['physics']['sediment'] is not None and p.get('sediment_start_time_s',0.)==0.))
            return owner
        make.required_leaves=build.required_leaves
        settings=FieldSettings(**p['adaptation'])
        reference=restrict(t,initial['rock']+initial['thickness'].sum(0));progress('initial_owner')
        policy=AutomaticRiver if p['adaptive'] else FixedRiver
        if restart:
            driver=policy.restart(restart,t,make,reference,settings)
            # Cancellation may follow an accepted step but precede publication
            # of its due regrid. Finish that transaction before advancing time.
            if p['adaptive'] and driver.next_time<=driver.controller.owner.time+1e-12:
                checkpoints.invalidate()
                driver.assess(make,cancelled=lambda:stop_file is not None and filesystem_path(stop_file).exists())
        else:
            leaves=initial_leaves(p,initial,t,getattr(make,'required_leaves',())) if initial_state is None else tuple(map(int,initial_leaf_state['leaves']))
            controller=RiverLeafController.from_factory(t,leaves,make,
                cancelled=lambda:stop_file is not None and filesystem_path(stop_file).exists())
            driver=policy(controller,reference,settings)
            if p['adaptive']:
                checkpoints.invalidate()
                driver.assess(make,cancelled=lambda:stop_file is not None and filesystem_path(stop_file).exists())
        a=driver.controller.owner;now[0]=a.time
        if target<=a.time:raise ValueError('Restart target must advance time')
        initial_water=float(np.sum(initial['q'][0]*np.array([c.area_m2 for c in t.finest.cells])))
        writer=MeshEpochWriter(output/'frames');checkpoints.ensure(driver,force=True);publish();interval=p['output_interval_s']
        nextout=(np.floor((a.time+output_clock_tolerance(a.time))/interval)+1)*interval
        while driver.controller.owner.time<target-output_clock_tolerance(target):
            check_stop(stop_file);a=driver.controller.owner
            # Shared physical-time gate on the conservative Exner transaction.
            # Geometry/restarts retain the full selected-model identity. No
            # sediment flux, stock update or bed-boundary ledger during warm-up.
            start=p.get('sediment_start_time_s',0.)
            if resolved_model is not None:
                a.morphology=bool(resolved_model['physics']['sediment'] is not None and a.time>=start)
            # End steps exactly at output/adaptation clocks; no delayed decisions.
            clocks=[p['dt_max_s'],target-a.time,nextout-a.time]
            if a.morphology and driver.sediment_step_hint is not None:
                clocks.append(driver.sediment_step_hint)
            if a.time<start:clocks.append(start-a.time)
            if hasattr(a,'next_forcing_time'):clocks.append(a.next_forcing_time(a.time)-a.time)
            if p['adaptive'] and driver.next_time>a.time+1e-12:clocks.append(driver.next_time-a.time)
            dt=min(clocks)
            dt=output_horizon(dt,a.time,min(nextout,target))
            for attempt in range(10):
                try:
                    checkpoints.invalidate()
                    accepted=a.step(dt)
                    d=accepted['diagnostics']
                    if a.morphology and 'maximum_relative_bed_increment' in d:
                        driver.sediment_step_hint=next_hint(driver.sediment_step_hint,dt,p['dt_max_s'],
                            d['maximum_relative_bed_increment'],attempt>0)
                    hydraulic['accepted_substeps']+=d.get('hydraulic_substeps',1)
                    hydraulic['internal_retries']+=len(d.get('hydraulic_rejections',()))
                    hydraulic['accuracy_retries']+=sum(r.get('kind')=='accuracy' for r in d.get('hydraulic_rejections',()))
                    hydraulic['nonlinear_retries']+=sum(r.get('kind') not in ('accuracy','cfl','spatial_or_positivity') for r in d.get('hydraulic_rejections',()))
                    hydraulic['cfl_retries']+=sum(r.get('kind')=='cfl' for r in d.get('hydraulic_rejections',()))
                    hydraulic['spatial_retries']+=sum(r.get('kind')=='spatial_or_positivity' for r in d.get('hydraulic_rejections',()))
                    hydraulic['accepted_cfl_max']=max(hydraulic['accepted_cfl_max'],d.get('accepted_cfl_max',0.))
                    hydraulic['maximum_temporal_error_ratio']=max(hydraulic['maximum_temporal_error_ratio'],d.get('temporal_error_ratio_max',0.))
                    hydraulic['fwave_evaluations']+=d.get('fwave_evaluations',0)
                    value=d.get('hydraulic_minimum_dt',dt)
                    old=hydraulic['minimum_substep_s']
                    hydraulic['minimum_substep_s']=value if old is None else min(old,value)
                    hydraulic['next_substep_hint_s']=d.get('hydraulic_next_substep_hint_s')
                    break
                except BendTrialRejected as exc:
                    rejected.append(dict(time_s=a.time,dt_s=dt,reason=str(exc)))
                    if attempt==9:raise
                    dt*=.5;check_stop(stop_file)
            now[0]=a.time
            if p['adaptive']:
                if driver.next_time<=a.time+1e-12:checkpoints.invalidate()
                driver.assess(make,cancelled=lambda:stop_file is not None and filesystem_path(stop_file).exists())
            else:driver.next_time=a.time
            checkpoints.ensure(driver)
            if a.time>=nextout-output_clock_tolerance(nextout):publish();nextout+=interval
            progress('running')
        status='complete' if target>=p['end_time_s'] else 'segment_complete'
    except InterruptedError as exc:status='cancelled';error=str(exc)
    except Exception as exc:
        import traceback
        traceback.print_exc();status='failed';error=f'{type(exc).__name__}: {exc}'
    if driver is not None:
        now[0]=driver.controller.owner.time
        checkpoints.ensure(driver,force=True)
        if 'writer' in locals():publish()
        write_json_atomic(output/'ADAPTATION.json',driver.events)
        write_json_atomic(output/'CACHE.json',cache.records)
    summary=dict(status=status,error=error,backend=backend,time_s=now[0],wall_s=time.perf_counter()-began,
        hydraulic=hydraulic,
        sediment_next_step_hint_s=driver.sediment_step_hint if driver else None,
        checkpoint_interval_s=checkpoints.interval,checkpoints=checkpoints.rows,
        accepted_steps=driver.controller.owner.steps if driver else 0,rejected_trials=rejected,
        integrator=driver.controller.owner.solver.identity['integrator'] if driver else None,
        scope='explicit_fwave_common_clock',QGIS_admitted=False)
    write_json_atomic(output/'STATUS.json',summary);write_json_atomic(output/'PROGRESS.json',dict(summary,phase=status,percent=100 if status=='complete' else 20+80*now[0]/end[0]))
    if status=='failed':raise RuntimeError(error)
    return summary


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='action',required=True)
    a=sub.add_parser('prepare');a.add_argument('--profile',required=True);a.add_argument('--fields',required=True);a.add_argument('--output',required=True);a.add_argument('--stop-file')
    a=sub.add_parser('run');a.add_argument('--input',required=True);a.add_argument('--output',required=True);a.add_argument('--backend',choices=('cpu','gpu'),required=True)
    a.add_argument('--restart');a.add_argument('--until',type=float);a.add_argument('--stop-file')
    args=p.parse_args()
    if args.action=='prepare':
        with np.load(args.fields,allow_pickle=False) as f:fields_={k:f[k] for k in f.files}
        print(prepare(json.loads(Path(args.profile).read_text()),fields_,args.output,stop_file=args.stop_file))
    else:print(json.dumps(run(args.input,args.output,backend=args.backend,restart=args.restart,until=args.until,stop_file=args.stop_file)))


if __name__=='__main__':main()
