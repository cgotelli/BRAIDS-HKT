"""Desktop worker adapter for the qualified seven-field research composition.

Shares project preparation, polygon geometry, hydraulic BCs and result archives.
Never routes a JC selection into capacity/Exner or drops its mobile/I inventory.
"""
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np

from braids_core.river_jiang_cao import BUNDLE, COLLAPSE, SECONDARY, material
from braids_core.windows_paths import filesystem_path
from braids_gpu.atomic_output import write_json_atomic
from . import adaptive_river_worker as core
from .bend_methods_v1 import BendTrialRejected, digest
from .jiang_cao_wetdry import JiangCaoShoreline, GrainScaleBedloadTreatment
from .jiang_cao_roughness import RoughnessParameters
from .jiang_cao_remesh import JiangCaoRemesh, validate
from .jiang_cao_adaptive import JiangCaoAdaptive
from .river_leaf_io import MeshEpochWriter


OUTER_CFL_SAFETY=.98
OUTER_HINT_GROWTH=1.25


def outer_cfl_hint(previous_hint,trial_dt,observed_cfl,cfl_limit,dt_max):
    """Predict the coupled step from waves already measured by the last trial."""
    raw=(previous_hint,trial_dt,observed_cfl,cfl_limit,dt_max)
    if any(isinstance(v,(bool,np.bool_)) for v in raw):
        raise ValueError('Finite outer CFL hint inputs required')
    try:values=tuple(float(v) for v in raw)
    except (TypeError,ValueError) as exc:
        raise ValueError('Finite outer CFL hint inputs required') from exc
    if any(not np.isfinite(v) for v in values):
        raise ValueError('Finite outer CFL hint inputs required')
    previous_hint,trial_dt,observed_cfl,cfl_limit,dt_max=values
    if previous_hint<=0 or trial_dt<=0 or observed_cfl<0 or cfl_limit<=0 or dt_max<=0:
        raise ValueError('Positive outer CFL hint clocks required')
    if observed_cfl==0:return min(float(dt_max),OUTER_HINT_GROWTH*float(previous_hint))
    predicted=OUTER_CFL_SAFETY*float(cfl_limit)*float(trial_dt)/float(observed_cfl)
    return min(float(dt_max),OUTER_HINT_GROWTH*float(previous_hint),predicted)


class FreeExterior:
    """No independent exterior history: all values rebuilt from current q,z.

Only the explicit free-bed, constant-input contract is admitted to JC restart.
All fixed boundary arrays are checked; mutable flux/trace/section buffers are
scratch and rebuilt by the common exterior engine, not checkpoint state.
"""
    def __init__(self, s, ext):
        from .jiang_cao_checkpoint import array_hash, canonical
        self.s,self.ext=s,ext;self._evaluate=ext.evaluate
        self.forcing=ext.forcing.copy()
        self._fixed=canonical(self.configuration())
        ext.evaluate=self.evaluate

    def configuration(self):
        from .jiang_cao_checkpoint import array_hash
        e=self.ext;p=e.p;host=self.s.a.host
        if e.schedule.tables or e.bed_schedule.tables or e.bedforms is not None:
            raise ValueError('JC free exterior forbids independent histories/tables')
        if bool(e.xp.any((p.exterior_mode!=0)&(p.exterior_mode!=1))):
            raise ValueError('JC exterior bed must be free')
        return dict(adapter=digest(__file__),hydraulic=e.owner.identity['exterior_bed'],
            arrays={k:array_hash(host(v)) for k,v in dict(mode=p.exterior_mode,
                hydraulic_mode=e.hydraulic_mode,selected=e.selected,stage_policy=e.stage_policy,
                forcing=self.forcing,offset=e.offset,faces=e.faces).items()},
            roughness_mode=e.roughness_mode,floor=e.floor,waqua=e.waqua)

    def verify(self):
        from .jiang_cao_checkpoint import canonical
        if self._fixed!=canonical(self.configuration()):raise ValueError('JC exterior configuration changed')
        return self.configuration()

    def evaluate(self,q,z,*,time=None):
        s=self.s;e=self.ext;p=e.p
        p.exterior_bed[:]=s.xp.where(e.face_selected,z[p.g['owner']],0.)
        e.roughness[:]=s.shallow.resistance(s.roughness,q)
        e.forcing[:]=self.forcing
        return self._evaluate(q,z,time=time)


def factory(project, p, cache, work, boundaries, stop=None,*,initial_clock_s=0.):
    from braids_gpu import general_live_transport as transport
    from .river_transport import build_transport_river
    from .river_runtime_cache import source_folder
    from .river_native_lifecycle import release_loaded_search_directories
    from .river_exterior_bed import ExteriorBed
    from braids_core.river_project import ROUGHNESS_PARAMETERS
    backend=project['model']['request']['execution']['backend']
    selected=project['model']['physics']['sediment']; mat=material(project)
    controls,_=core.validate_profile(p); count=0

    def make(leaves, mesh):
        nonlocal count
        core.check_stop(stop); inp=cache.get(leaves,mesh); d,f=cache.load_payload(inp)
        sources=source_folder(inp); count+=1; folder=Path(work)/f'owner_{count:04d}'
        if backend=='cpu':transport.verify_sources(sources)
        ctx=nullcontext(None) if backend=='cpu' else transport.constructors(sources,d,integrator='ssp33')
        with ctx as Solver:
            a=build_transport_river(inp/'case.data',d,backend=backend,controls=controls,
                bedbc=f['bedbc'],feed=np.zeros(d['nf']),work_directory=folder,
                gpu_solver_type=Solver,source_folder=sources if backend=='gpu' else None)
        a.q[:]=a.xp.asarray(f['q']);a.z[:]=a.xp.asarray(f['z'])
        outside=(d['neighbour']<0)&(d['bid']>=0)
        kind=np.where(d['neighbour']>=0,0,np.where(outside,3,1)).astype(np.int32)
        cin=np.zeros(d['nf'])
        for bid,name in enumerate(sorted(p['hydraulics'])):
            faces=outside&(d['bid']==bid)
            row=project['settings']['sediment_boundaries'][name]
            if row['kind']!='outflow':
                kind[faces]=2
                ratio=row['solid_water_ratio'][0]
                # Q is mixture volume flux; qs/q_water -> c_solid = r/(1+r).
                cin[faces]=ratio/(1+ratio)
        s=JiangCaoShoreline(a,mat,mobile_solid=0.,
            shallow=GrainScaleBedloadTreatment(),
            bed_solid=(1-mat.porosity)*f['thickness'].sum(0),boundary_kind=kind,
            roughness_parameters=RoughnessParameters(mat,project['model']['physics']['sediment']['capacity']['grain_sizes_m']['d90']),
            collapse_enabled=selected['avalanche']==COLLAPSE,secondary_adjustment=SECONDARY)
        s.initial_clock_origin_s=float(initial_clock_s)
        if 'jc_state' in f:
            s.state[:]=s.xp.asarray(f['jc_state'])
            a.q[:]=s.state[:3];a.z[:]=s.state[5];a.physics.rock[:]=s.state[5]-s.state[4]/s.B
            s.initial_solid,s.initial_water=s.totals(s.state)
        a.cancelled=lambda:stop is not None and filesystem_path(stop).exists()
        names=sorted(boundaries['hydraulics'])
        if names:
            # The common exterior engine consumes a Manning value; the adapter
            # updates it from the SAME bundle resistance at every residual.
            a.physics.identity['river_roughness']=dict(method='roughness.manning',
                parameters=dict(ROUGHNESS_PARAMETERS),total_resistance_owner=BUNDLE)
            states=boundaries['states']; z=np.where(outside,f['z'][d['owner']],0.)
            ext=ExteriorBed(a,methods={b:'bed.free' for b in range(len(names))},initial=z,value=np.zeros(d['nf']),
                hydraulics={b:boundaries['hydraulics'][name] for b,name in enumerate(names)},
                stage_policies={b:states[name]['stage_policy'] for b,name in enumerate(names)
                                if boundaries['hydraulics'][name]['method']=='outlet.stage'},
                hydraulic_schedules={b:states[name]['hydraulic_schedule'] for b,name in enumerate(names)},
                bed_schedules={b:states[name]['bed_schedule'] for b,name in enumerate(names)})
            s.desktop_exterior=FreeExterior(s,ext)
        s.identity['desktop_bundle']=dict(method=BUNDLE,adapter=digest(__file__),
            resolved=project['model'],boundaries=boundaries,feed='c_solid = qs/q_water / (1 + qs/q_water)',
            hydraulic_discharge='mixture-volume discharge; water-volume flux = (1-c)*mixture flux',
            exterior='free bed and local bundle Manning rebuilt every residual; zero incoming I')
        result=JiangCaoRemesh(s,polygons=[c.vertices_xy for c in mesh.cells],inlet_concentration=cin)
        release_loaded_search_directories(result);core.check_stop(stop)
        return result
    return make


def fields(owner, reference):
    """Saved cell inventory and named instantaneous mobile flux, not Exner flux."""
    a=owner.a; s=owner.solver; st=a.host(s.state).copy();h=st[0]
    saved={k:getattr(s,k).copy() if hasattr(s,k) else None for k in ('secondary_fields','dispersion_flux')}
    try:
        _,cap=s.target(s.state)
        drift=s.relative_discharge(s.state,cap)
        velocity=s.state[1:3]/s.xp.where(s.state[0]>s.dry,s.state[0],1.)
        concentration=s.state[3]/s.xp.where(s.state[0]>s.dry,s.state[0],1.)
        # drift is relative MIXTURE discharge; the face transport multiplies
        # (Q + drift.n) by mobile solid concentration. Use the same units here.
        flux=a.host(velocity*s.state[3]+concentration*drift).copy()
        capacity=a.host(cap.values[:3]).copy()
    finally:
        for k,v in saved.items():
            if v is None and hasattr(s,k):delattr(s,k)
            elif v is not None:setattr(s,k,v)
    rho=s.exchange.material.capacity_material.solid_density_kg_m3
    return dict(q=st[:3],z=st[5],rock=st[5]-st[4]/s.B,
        mass=(rho*st[4])[None,None,:],initial_bed_m=reference,
        mobile_bedload_solid_m=st[3],mobile_bedload_concentration=np.divide(st[3],h,out=np.zeros_like(h),where=h>0),
        secondary_hI_m2_s=st[6],secondary_intensity_m_s=np.divide(st[6],h,out=np.zeros_like(h),where=h>0),
        bedload_vector_m2_s=flux,bedload_capacity_m2_s=capacity[0],
        water=-a.host(s.water_integral).copy(),bed_boundary=-rho*a.host(s.solid_integral)[None,:].copy(),
        water_vector_m2_s=st[1:3]-flux,
        bedload_flux_sampling=np.array('instantaneous cell mobile advection plus relative drift; not last accepted face flux'),
        time=np.array(s.time),steps=np.array(s.steps),state=st)


def run(prepared, output, *, project, receipt, boundaries, restart=None, until=None,
        stop_file=None, on_frame=None, checkpoint_path=None, on_created=None,
        checkpoint_interval_s=60., on_checkpoint=None,on_checkpoint_state=None,initial_state=None):
    """Same public entry, lifecycle publications and explicit backend as Desktop."""
    root=filesystem_path(output);root.mkdir(parents=True,exist_ok=False)
    if on_created:on_created()
    began=time.perf_counter(); driver=None; rows=[]; rejected=[]; status='failed';error=None
    outer_hint=None;outer_preemptions=0
    checkpoint_path=filesystem_path(checkpoint_path or root/'checkpoint.npz')
    checkpoint_path.parent.mkdir(parents=True,exist_ok=True)
    check=lambda:core.check_stop(stop_file)
    def progress(phase):
        at=0. if driver is None else driver.owner.time
        write_json_atomic(root/'PROGRESS.json',dict(status='running',phase=phase,time_s=at,
            percent=5 if driver is None else min(100,20+80*at/target),backend=backend))
    def save_checkpoint():
        # JC archives refuse overwrite. Publish a fresh accepted archive and
        # pass it to the existing authenticated project-envelope publisher.
        raw=root/'internal'/f'accepted_{driver.owner.steps:08d}_{len(driver.events):06d}.npz'
        if not raw.exists():driver.checkpoint(raw,check=lambda:None)
        if on_checkpoint:on_checkpoint(raw,dict(time_s=driver.owner.time,steps=driver.owner.steps,
            next_adaptation_s=driver.next_time,adaptation_events=len(driver.events)))
        if on_checkpoint_state:on_checkpoint_state(raw,dict(time_s=driver.owner.time,steps=driver.owner.steps,
            next_adaptation_s=driver.next_time,adaptation_events=len(driver.events)),driver.owner,driver._meshes[driver.leaves])
    def publish():
        owner=driver.owner
        if writer.frames and writer.frames[-1]['steps']==owner.steps:return
        validate(owner)
        adapter=SimpleNamespace(leaves=driver.leaves,owner=owner,hierarchy=t,
            geometry_sha256=driver.identity['geometry_sha256'],_mesh_cache=driver._meshes)
        writer.write(adapter,fields(owner,reference[list(driver.leaves)]))
        solid,water=owner.totals(owner.state)
        row=dict(time_s=owner.time,steps=owner.steps,cells=len(driver.leaves),
            solid_residual_m3=solid+float(owner.xp.sum(owner.solid_integral))-owner.initial_solid,
            water_plus_pore_residual_m3=water+float(owner.xp.sum(owner.water_integral))-owner.initial_water,
            cumulative_collapse_gross_solid_m3=owner.collapse_gross)
        rows.append(row);write_json_atomic(root/'DIAGNOSTICS.json',rows)
        if on_frame:on_frame(root,row)
    backend=project['model']['request']['execution']['backend']
    target=project['settings']['end_time_s'] if until is None else float(until)
    try:
        check();progress('loading')
        p,initial,t=core.load_prepared(prepared,stop_file)
        initial_meta,initial_leaf_state=(None,None) if initial_state is None else initial_state
        if initial_state is not None:
            from braids_core.river_state_initialization import finest_fields
            initial=finest_fields(t,initial,initial_leaf_state)
        cache=core.cache_for(filesystem_path(prepared)/'cache',p,initial,t,stop_file,progress,
            initial_leaf_state=initial_leaf_state)
        make=factory(project,p,cache,root/'runtime',boundaries,stop_file,
            initial_clock_s=0. if initial_state is None else initial_meta['time_s'])
        settings=core.FieldSettings(**p['adaptation'])
        fine=tuple(range(t.offsets[-2],t.count));pins=()
        params=project['model']['physics']['sediment']['capacity']['parameters']
        if params['ProtectInitialDryBank']:
            from .jiang_cao_coverage import initial_repose_patch
            protection=initial_repose_patch(t,make(fine,t.finest),check=check)
            pins=tuple(protection['protected_finest']);write_json_atomic(root/'DRY_BANK_PROTECTION.json',protection)
        leaves=core.initial_leaves(p,initial,t,pins) if initial_state is None else tuple(map(int,initial_leaf_state['leaves']))
        # Fixed meshes use the same epoch/checkpoint owner with no scheduled
        # regrids. The immutable large clock does not change on continuation.
        policy=replace(settings,terrain_enabled=False,domain_border_enabled=False,
                       interval=settings.interval if p['adaptive'] else 1.e30)
        driver=JiangCaoAdaptive(t,leaves,make,prepared_identity=receipt,settings=policy,protected_finest=pins,
            start_time_s=0. if initial_state is None else initial_meta['time_s'])
        reference=core.restrict(t,initial['rock']+initial['thickness'].sum(0))
        if restart:driver.restart(restart,check=check)
        if target<=driver.owner.time:raise ValueError('Restart target must advance time')
        writer=MeshEpochWriter(root/'frames');check();save_checkpoint();publish()
        interval=p['output_interval_s'];nextout=(np.floor((driver.owner.time+1e-12)/interval)+1)*interval
        nextck=(np.floor((driver.owner.time+1e-12)/checkpoint_interval_s)+1)*checkpoint_interval_s
        last=driver.owner.last
        outer_hint=float(last.get('outer_next_step_hint_s',p['dt_max_s'])) if last else float(p['dt_max_s'])
        if not np.isfinite(outer_hint) or not 0<outer_hint<=p['dt_max_s']:
            raise ValueError('Restart outer timestep hint differs')
        while driver.owner.time<target-core.output_clock_tolerance(target):
            check()
            if p['adaptive'] and driver.owner.time>=driver.next_time-1e-12:driver.assess(check=check)
            at=driver.owner.time;clock=min(nextout-at,target-at,driver.next_time-at)
            dt=min(p['dt_max_s'],outer_hint,clock)
            dt=core.output_horizon(dt,at,min(nextout,target,driver.next_time))
            for attempt in range(20):
                try:
                    driver.step(dt)
                    observed=float(driver.owner.last['hydraulic_cfl'])
                    outer_hint=outer_cfl_hint(outer_hint,dt,observed,
                        driver.owner.a.midpoint.cfl,p['dt_max_s'])
                    outer_preemptions+=int(outer_hint<p['dt_max_s']*(1-1e-12))
                    driver.owner.last['outer_next_step_hint_s']=outer_hint
                    break
                except BendTrialRejected as exc:
                    rejected.append(dict(time_s=at,dt_s=dt,error=str(exc)))
                    peak=float(driver.owner.a.midpoint.frequency_peak)
                    if str(exc).startswith('Explicit stage CFL') and peak>0:
                        observed=dt*peak
                        outer_hint=min(outer_hint,outer_cfl_hint(outer_hint,dt,observed,
                            driver.owner.a.midpoint.cfl,p['dt_max_s']))
                        dt=min(clock,outer_hint)
                    else:
                        dt*=.5;outer_hint=min(outer_hint,dt)
                    if attempt==19:raise
            if p['adaptive'] and driver.owner.time>=driver.next_time-1e-12:driver.assess(check=check)
            if driver.owner.time>=nextck-1e-12:
                save_checkpoint();nextck=(np.floor((driver.owner.time+1e-12)/checkpoint_interval_s)+1)*checkpoint_interval_s
            if driver.owner.time>=nextout-core.output_clock_tolerance(nextout):
                publish();nextout+=interval
            if driver.owner.steps%20==0:progress('hydraulics and mobile bedload')
        save_checkpoint();publish();status='complete'
    except InterruptedError as exc:
        status='cancelled';error=str(exc)
        if driver is not None:save_checkpoint()
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}'
    result=dict(status=status,error=error,time_s=0. if driver is None else driver.owner.time,
        steps=0 if driver is None else driver.owner.steps,backend=backend,method=BUNDLE,
        wall_seconds=time.perf_counter()-began,rejections=rejected,
        outer_cfl_preemptions=outer_preemptions,
        outer_next_step_hint_s=outer_hint,
        adaptation_events=[] if driver is None else driver.events)
    write_json_atomic(root/'STATUS.json',result)
    write_json_atomic(root/'PROGRESS.json',dict(result,percent=100 if status=='complete' else 0,phase=status))
    if status=='failed':raise RuntimeError(error)
    return result
