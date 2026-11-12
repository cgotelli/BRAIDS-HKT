"""Cold paired inlet qualification, immutable candidate and public workers."""
import argparse,io,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'validation/_inlet187'
TARGET=OUT/'prescribed_candidate'
sys.path[:0]=[str(TARGET),str(ROOT)]
os.environ['QT_QPA_PLATFORM']='offscreen';os.environ['OMP_NUM_THREADS']='4'
import numpy as np
from stage_secondary185 import save,sha

def checkpoint_arrays(path):
    data=path.read_bytes()
    while True:
        with np.load(io.BytesIO(data),allow_pickle=False) as f:
            key=next((k for k in ('payload','owner','river') if k in f),None)
            if key is None:return {k:f[k].copy() for k in f if k!='metadata'}
            data=f[key].tobytes()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--backend',required=True,choices=('cpu','gpu'))
    ap.add_argument('--dry',action='store_true');ap.add_argument('--restart',action='store_true')
    ap.add_argument('--until',type=float,default=220.)
    args=ap.parse_args();b=args.backend;folder=OUT/(('dry_prescribed' if args.dry else 'wet_prescribed')+'_'+b)
    from braids_core.river_launcher import verify
    from braids_core.sediment.portable_river_runtime import install
    from braids_core import river_worker
    from braids_desktop import services
    from braids_desktop.preparation import prepare
    from braids_desktop.results import ResultsStore
    verify();install(json.loads((TARGET/'PRECOMPILED.json').read_text()),TARGET)
    assert sha(TARGET/'RUNTIME.json')==json.loads((OUT/'PRESCRIBED_STAGE.json').read_text())['candidate']
    owners=[];records=[];factory=river_worker.core.factory
    def instrument(*aa,**kk):
        build=factory(*aa,**kk)
        def construct(*a,**k):
            owner=build(*a,**k);owners.append(owner)
            morph=owner.morphology;owner.morphology=False
            before=owner.checkpoint_bytes();owner.step(.01,commit=False)
            assert owner.checkpoint_bytes()==before,'Rollback must be byte exact'
            owner.morphology=morph;step=owner.step
            def recorded(*sa,**sk):
                r=step(*sa,**sk)
                if sk.get('commit',True):records.append(dict(time=owner.time,
                    **{key:owner.physics.last[key] for key in ('secondary_residual','secondary_cfl_max','secondary_substeps','solid_volume_residual_m3')}))
                return r
            owner.step=recorded;return owner
        construct.__dict__.update(build.__dict__);return construct
    river_worker.core.factory=instrument
    if args.restart:
        out=folder/'restart';out.mkdir(exist_ok=False);stop=out/'STOP'
        def cancel(path,row):
            if row['time_s']>=210.:stop.touch()
        # Restart from the candidate's own 200 s checkpoint, cancel after bed
        # evolution, then resume: never reinterpret the v6 historical state.
        candidates=list((folder/'checkpoints').glob('t_200_step_*.npz'))
        assert len(candidates)==1
        checkpoint=candidates[0]
        prefix=river_worker.run(folder/'prepared',out/'cancelled',until=220.,restart=checkpoint,
            stop_file=stop,on_frame=cancel)
        assert prefix['status']=='cancelled' and prefix['time_s']==210.,prefix
        result=river_worker.run(folder/'prepared',out/'resumed',until=220.,restart=out/'cancelled/checkpoint.npz')
        expected=ResultsStore(folder/'run').read(-1);actual=ResultsStore(out/'resumed').read(-1)
        exact={key:bool(np.array_equal(actual.arrays[key],expected.arrays[key])) for key in actual.arrays}
        left=checkpoint_arrays(folder/'run/checkpoint.npz');right=checkpoint_arrays(out/'resumed/checkpoint.npz')
        state={key:bool(np.array_equal(left[key],right[key])) for key in left}
        save(out/'CHECK.json',dict(status=result,frame_exact=exact,checkpoint_arrays_exact=state,
            cancellation=prefix,rollback_byte_exact=True))
        assert all(exact.values()) and all(state.values()),'Restart differs'
        print('RESTART EXACT',b,flush=True);return
    folder.mkdir(exist_ok=False)
    source=ROOT/f'validation/_mesh186/refined_{b}/project/braids_project.json'
    doc=json.loads(source.read_text());doc.pop('editor_state',None)
    # Explicit new validation input, not a supported old-project migration.
    from braids_core.river_project import SCHEMA
    from braids_core.river_open_boundary import resolve
    doc['schema']=SCHEMA
    for row in doc['boundary_states'].values():row['secondary_intensity_m_s']=0.
    kinds={name:('inlet' if row['method'].startswith('inlet.') else 'outlet') for name,row in doc['settings']['hydraulics'].items()}
    doc['boundary_resolution']=resolve(doc['settings']['hydraulics'],doc['boundary_states'],kinds,
        sediment=doc['model']['physics']['sediment'],ng=1,end_time=doc['settings']['end_time_s'])
    doc['terrain']['dem_path']=str((source.parent/doc['terrain']['dem_path']).resolve())
    if args.dry:doc['initial']['stage_m']=-.3220077144
    assert doc['model']['physics']['sediment']['bed_slope']=='bed_slope.off'
    services.save_project_folder(folder/'project',doc)
    save(folder/'INPUT.json',dict(runtime=sha(TARGET/'RUNTIME.json'),source=str(source),
        initial=doc['initial'],backend=b,bed_slope='off',until=args.until,
        boundary='prescribed_incoming_secondary_intensity_v1',incoming_secondary_m_s=0.,sediment_start=200.))
    start=time.perf_counter();prepare(folder/'project/braids_project.json',folder/'prepared',asset_cache=folder/'cache')
    print(b,'PREPARED',flush=True)
    def progress(path,row):
        if row['time_s']%20==0 or row['time_s']>=199:print(b,'time',row['time_s'],'wall',round(time.perf_counter()-start,1),flush=True)
    result=river_worker.run(folder/'prepared',folder/'run',until=args.until,
        checkpoint_dir=folder/'checkpoints',on_frame=progress)
    save(folder/'STEPS.json',records)
    frames=ResultsStore(folder/'run');first,last=frames.read(0),frames.read(-1)
    d=json.loads((folder/'run/DIAGNOSTICS.json').read_text())
    assert all(abs(r['water_residual_m3'])<=r['water_roundoff_limit_m3'] and
        max(abs(v) for v in r['class_residual_kg'])<=r['class_roundoff_limit_kg'] and not any(r['wall_integrals']) for r in d)
    assert all(abs(r['secondary_residual'])<1e-10 for r in records)
    assert np.isfinite(last.q).all() and last.q[0].min()>=0 and last.arrays['mass'].min()>=0
    save(folder/'CHECK.json',dict(status=result,elapsed=time.perf_counter()-start,rollback_byte_exact=True,
        water_residual=max(abs(r['water_residual_m3']) for r in d),
        sediment_residual=max(max(abs(v) for v in r['class_residual_kg']) for r in d),
        secondary_residual=max(abs(r['secondary_residual']) for r in records),
        bed_min_mm=float(1000*(last.z-first.z).min()),bed_max_mm=float(1000*(last.z-first.z).max())))
    print('DONE',b,flush=True)

if __name__=='__main__':main()
