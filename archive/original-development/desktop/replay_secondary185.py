"""Isolated common-field operator replay; never a cross-backend restart route."""
import argparse,json,sys,os,time
from pathlib import Path
from stage_secondary185 import ROOT,TARGET,OUT,save,sha
if '--runtime' in sys.argv:
    TARGET=Path(sys.argv[sys.argv.index('--runtime')+1]).resolve()
os.environ['QT_QPA_PLATFORM']='offscreen'
sys.path[:0]=[str(TARGET),str(ROOT)]
import numpy as np
from braids_core.river_launcher import verify
from braids_core.sediment.portable_river_runtime import install
from braids_core import river_worker
from braids_core.sediment.bend_methods_v1 import BendTrialRejected
from braids_desktop.results import ResultsStore


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--backend',required=True)
    ap.add_argument('--state',default='cpu');ap.add_argument('--time',type=float,default=200.)
    ap.add_argument('--seconds',type=float,default=.2);ap.add_argument('--dt',type=float,default=.2)
    ap.add_argument('--tag',required=True);ap.add_argument('--trace',action='store_true')
    ap.add_argument('--history',action='store_true');ap.add_argument('--hydro-only',action='store_true')
    ap.add_argument('--trace-limit',type=int,default=6)
    ap.add_argument('--hydro-cfl',type=float)
    ap.add_argument('--sensitivity',action='store_true')
    ap.add_argument('--outlet-extension',action='store_true')
    ap.add_argument('--bed-trace-time',type=float)
    ap.add_argument('--continuous-face',action='store_true')
    ap.add_argument('--trace-after',type=float,default=0.)
    ap.add_argument('--face-bounds',action='store_true')
    ap.add_argument('--no-feedback',action='store_true')
    ap.add_argument('--no-direction',action='store_true')
    ap.add_argument('--slope-contrast',action='store_true')
    ap.add_argument('--prepared',type=Path)
    ap.add_argument('--source',type=Path)
    ap.add_argument('--output-root',type=Path)
    ap.add_argument('--runtime',type=Path)
    ap.add_argument('--wall-audit',action='store_true')
    ap.add_argument('--perturb-depth',type=float,default=0.)
    ap.add_argument('--operator-peer',type=Path)
    ap.add_argument('--steered-wave',action='store_true')
    ap.add_argument('--hydraulic-inputs',type=Path)
    ap.add_argument('--hydraulic-sensitivity',type=Path)
    ap.add_argument('--projected-gradient',action='store_true')
    ap.add_argument('--projected-walls',action='store_true')
    ap.add_argument('--projected-secant',action='store_true')
    args=ap.parse_args();verify();install(json.loads((TARGET/'PRECOMPILED.json').read_text()),TARGET)
    folder=(args.output_root or OUT/'replay')/args.tag;folder.mkdir(parents=True,exist_ok=False)
    captured=[];factory=river_worker.core.factory
    def instrument(*a,**kw):
        build=factory(*a,**kw)
        def traced(*aa,**kk):
            owner=build(*aa,**kk);captured.append(owner);return owner
        traced.__dict__.update(build.__dict__);return traced
    river_worker.core.factory=instrument
    stop=folder/'STOP'
    prepared=args.prepared or OUT/f'signed_{args.backend}_v6/prepared'
    source_path=args.source or OUT/f'qualified_{args.state}_v6/to_220'
    result=river_worker.run(prepared,folder/'construction',
        until=.001,stop_file=stop,on_frame=lambda *unused:stop.touch())
    assert result['status']=='cancelled' and result['time_s']==0.,result
    assert len(captured)==1
    a=captured[0];p=a.physics;xp=a.xp;a.cancelled=lambda:False
    source=ResultsStore(source_path)
    frame=source.read(next(i for i,t in enumerate(source.times) if abs(t-args.time)<1e-8))
    f=frame.arrays
    constructed=ResultsStore(folder/'construction').read(0)
    for key in ('vertices','offsets','levels'):
        np.testing.assert_array_equal(frame.mesh[key],constructed.mesh[key])
    # Explicit common-field initialisation. Historical cumulative ledgers are
    # deliberately not migrated; this compares operators with zero new budgets.
    a.q[:]=xp.asarray(f['q']);a.z[:]=xp.asarray(f['z']);p.W[:]=xp.asarray(f['W'])
    p.z0[:]=a.z;p.ztrial[:]=a.z;p.rock[:]=xp.asarray(f['rock'])
    c=a.columns
    rebuilt=type(c)(c.runtime,f['mass'],f['sv'],f['rock'],f['area'],time_s=args.time)
    c.__dict__.clear();c.__dict__.update(rebuilt.__dict__)
    core=a
    while hasattr(core,'solver'):core=core.solver
    core.time=args.time;core.steps=0;p.time=args.time;p.steps=0
    a.midpoint.step_hint=.005;a.midpoint.easy_steps=0;a.morphology=not args.hydro_only
    if args.hydro_cfl is not None:a.midpoint.cfl=args.hydro_cfl
    if args.outlet_extension:
        if args.backend=='gpu':a.hydraulics.reconstruction.outlet_extension=1
        else:
            # Isolated diagnosis of an existing native switch, not a public ABI.
            import ctypes
            ctypes.c_int.in_dll(a.hydraulics.lib,'__cutcell_solver_MOD_cpu_outlet_extension').value=1
    if args.backend=='gpu':a.hydraulics.rock[:]=p.rock
    if args.steered_wave:
        from steered_wave188 import install as install_steered_wave
        install_steered_wave(a,folder)
    if args.projected_gradient or args.projected_walls or args.projected_secant:
        from projected_gradient188 import install as install_projected
        install_projected(a,folder,walls=args.projected_walls or args.projected_secant,secant=args.projected_secant)
    audit=None
    if args.wall_audit or args.perturb_depth:
        from audit_wall188 import WallAudit
        audit=WallAudit(a,args.perturb_depth)
    if args.continuous_face:
        from validation.sediment.river_morphodynamics.sediment_wave import WaveEngine
        face=WaveEngine.__new__(WaveEngine)
        WaveEngine.__init__.__wrapped__(face,args.backend,p.flow.threads,
            source=ROOT/'desktop/continuous_face185.h',names=('capacity_reconstructed_faces',))
        p.reconstruction.fn['capacity_reconstructed_faces']=face.fn['capacity_reconstructed_faces']
        p._diagnostic_face=face
    if args.face_bounds:
        from validation.sediment.river_morphodynamics.sediment_wave import WaveEngine
        face=WaveEngine.__new__(WaveEngine)
        WaveEngine.__init__.__wrapped__(face,args.backend,p.flow.threads,
            source=ROOT/'desktop/face_bounds185.h',names=('capacity_gradients','capacity_reconstructed_faces'))
        p._diagnostic_face=face;p._diagnostic_bounds=xp.zeros((4,p.n))
        p.reconstruction.launch=lambda name,count,*aa:face.launch(name,count,*aa,p._diagnostic_bounds)
    if args.no_feedback:p.add_momentum=lambda *unused:None
    if args.no_direction:
        for capacity in a.capacity:
            old_evaluate=capacity.evaluate
            capacity.evaluate=lambda q,hi,_old=old_evaluate,**kw:_old(q,xp.zeros_like(hi),**kw)
    if args.slope_contrast:
        from braids_core.sediment.flow2dh_eh_operator import ClassicFlowEH
        for capacity in a.capacity:
            capacity.slope=ClassicFlowEH.slope.__get__(capacity)
    save(folder/'INPUT.json',dict(runtime=sha(TARGET/'RUNTIME.json'),backend=args.backend,
        source_backend=args.state,source_time=args.time,meaning='common physical fields, fresh history; not restart',
        wrapper=type(a).__name__,core=type(core).__name__,hydraulic_cfl=a.midpoint.cfl,
        isolated_outlet_extension=args.outlet_extension,isolated_continuous_face=args.continuous_face,
        isolated_face_bounds=args.face_bounds,isolated_no_feedback=args.no_feedback,
        isolated_no_direction=args.no_direction,isolated_slope_contrast=args.slope_contrast,
        dt=args.dt,seconds=args.seconds,perturb_depth_m=args.perturb_depth,wall_audit=args.wall_audit,
        isolated_steered_wave=args.steered_wave,
        isolated_projected_gradient=args.projected_gradient,isolated_projected_walls=args.projected_walls,
        isolated_projected_secant=args.projected_secant,
        source_path=str(source_path),prepared=str(prepared)))
    if args.operator_peer:
        from operator_wall188 import run
        peer=ResultsStore(args.operator_peer)
        other=peer.read(next(i for i,t in enumerate(peer.times) if abs(t-args.time)<1e-8))
        run(a,f,other.arrays,folder)
        return
    if args.hydraulic_inputs:
        from hydraulic_wall188 import run
        run(a,args.hydraulic_inputs,folder)
        return
    if args.hydraulic_sensitivity:
        from hydraulic_wall188 import sensitivity
        try:sensitivity(a,args.hydraulic_sensitivity,folder)
        except BaseException:
            import traceback
            (folder/'FAILURE.txt').write_text(traceback.format_exc());raise
        return
    if args.sensitivity:
        assert args.backend=='gpu'
        rec=a.hydraulics.reconstruction;captures={};rows=[]
        for stage in (76,104,120,136,145):
            paired=[]
            for source in ('cpu','gpu'):
                with np.load(OUT/f'replay/stages_{source}/FIELDS.npz') as fs:water=fs[f'hydraulic_{stage:02}_q'].copy()
                with np.load(OUT/f'replay/history_{source}/FIELDS.npz') as fs:bed=fs['history'][0,3].copy()
                rec.geometry['bed'][:]=xp.asarray(bed)
                buffers={k:a.host(v).copy() for k,v in rec.reconstruct(xp.asarray(water)).items()}
                paired.append(buffers)
                for k,v in buffers.items():captures[f'{stage}_{source}_{k}']=v
            rows.append(dict(stage=stage,**{k:float(np.max(np.abs(paired[0][k]-paired[1][k]))) for k in paired[0]}))
        captures.update({f'g_{k}':a.host(v) for k,v in rec.geometry.items() if isinstance(v,xp.ndarray)})
        np.savez(folder/'RECONSTRUCTION.npz',**captures);save(folder/'SENSITIVITY.json',rows)
        print(json.dumps(rows,indent=2));return
    trace={};counts={}
    def capture(label,inputs):
        if p.time<args.trace_after-1e-8:return
        k=counts.get(label,0);counts[label]=k+1
        if k<args.trace_limit:
            for key,v in inputs.items():
                if isinstance(v,xp.ndarray):trace[f'{label}_{k:02}_{key}']=a.host(v).copy()
    if args.bed_trace_time is not None:
        def selected():return abs(p.time-args.bed_trace_time)<1e-7
        def arrays():return {k:v for k,v in vars(p).items() if isinstance(v,xp.ndarray)}
        old_bed=a.bed_faces
        def bed(*aa,**kw):
            out=old_bed(*aa,**kw)
            if selected():capture('bed',arrays())
            return out
        a.bed_faces=bed
        for capacity in a.capacity:
            old_slope=capacity.slope
            def slope(fields,_old=old_slope):
                out=_old(fields)
                if selected():capture('slope',dict(fields=fields,result=out))
                return out
            capacity.slope=slope
        old_reconstruction=p.reconstruction.launch
        def reconstructed(name,count,*aa):
            out=old_reconstruction(name,count,*aa)
            if selected():capture(name,dict(bedcell=p.bedcell,bedface=p.bedface,
                capacity_gradient=p.capacity_gradient,capacity_limiter=p.capacity_limiter))
            return out
        p.reconstruction.launch=reconstructed
    if args.trace:
        for name in ('effective_inputs','refresh_secondary_momentum','bed_faces'):
            original=getattr(p,name)
            def wrap(*aa,_name=name,_original=original,**kw):
                out=_original(*aa,**kw)
                keys=('inputs','outputs','directional_B','directional_source','geometry_Bgrad','hold') if _name=='refresh_secondary_momentum' else (
                    'bedcell','bedface','bedcorrected','capacity_gradient','capacity_limiter','class_flux','effective_fields','effective_q','wave_diagnostic','shallow_vector')
                capture(_name,{k:getattr(p,k) for k in keys})
                return out
            setattr(p,name,wrap)
        rate=a.midpoint.rate
        def traced_rate(q,dt,t):
            r,w=rate(q,dt,t);capture('rate',dict(q=q,r=r,w=w));return r,w
        a.midpoint.rate=traced_rate
        momentum=p.add_momentum
        def traced_momentum(q,water,rate):
            capture('hydraulic',dict(q=q,r=rate,water=water))
            momentum(q,water,rate)
            capture('momentum',dict(r=rate))
        p.add_momentum=traced_momentum
    records=[];history=[];target=args.time+args.seconds;start=time.perf_counter()
    while a.time<target-1e-10:
        dt=min(args.dt,target-a.time);failures=[]
        for attempt in range(10):
            try:r=a.step(dt);break
            except BendTrialRejected as e:failures.append(str(e));dt*=.5
        else:raise RuntimeError(failures)
        records.append(dict(time=a.time,dt=dt,rejected=failures,**r['diagnostics']))
        if args.history:history.append(np.concatenate((a.host(a.q),a.host(a.z)[None,:],a.host(p.W)[None,:]),axis=0))
    trace.update({f'final_{k}':a.host(v).copy() for k,v in vars(p).items() if isinstance(v,xp.ndarray)})
    trace.update(final_q=a.host(a.q),final_z=a.host(a.z),final_mass=a.host(c.mass),initial_q=f['q'],initial_z=f['z'])
    trace.update({f'geometry_{k}':a.host(v) for k,v in p.g.items() if isinstance(v,xp.ndarray)})
    if history:trace['history']=np.asarray(history)
    np.savez(folder/'FIELDS.npz',**trace)
    save(folder/'STEPS.json',records);save(folder/'CHECK.json',dict(time=a.time,wall_s=time.perf_counter()-start,counts=counts))
    if audit is not None:audit.save(folder)
    print(args.tag,'complete',a.time,flush=True)
if __name__=='__main__':main()
