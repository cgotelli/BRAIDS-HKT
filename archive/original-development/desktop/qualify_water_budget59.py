"""Package the retained scale-aware water arithmetic check, without physics changes."""
import argparse,hashlib,json,shutil,subprocess,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'desktop/runtime'
OUT=ROOT/'desktop/qualification/water_budget59';TARGET=OUT/'candidate'
REPORT=ROOT/'validation/hydrokinetic_turbines/reports/desktop_validation59'
PARENT='1dcf63c3b753ebbbe012313f7ee05932461aeb38d9526f22d41af6c478aa097a'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,d):Path(p).write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def stop():
    records=[]
    for p in REPORT.glob('*/STARTED.json'):
        r=read(p);output=Path(r['output']).resolve();home=Path.home()/'Desktop/BRAIDS_Turbine_Validation59'
        assert output.is_relative_to(home.resolve())
        history=output.parent.parent/'history'/(output.name+'.log')
        first=json.loads(history.open(encoding='utf-8').readline());assert Path(first['output']).resolve()==output
        args=first['arguments'];flag=Path(args[args.index('--stop-file')+1]).resolve()
        assert flag.parent==output.parent and flag.name.startswith('.braids_stop_')
        flag.open('x',encoding='utf-8').close()
        records.append(dict(case=r['case'],output=str(output),stop_file=str(flag)))
    save(REPORT/'INCIDENT_STOP.json',dict(reason='Investigate packaged fixed 1e-10m3 arithmetic check; preserve accepted state and all rejected work',runs=records))
    print('Requested accepted-state stops for',len(records),'own workers',flush=True)


def stage():
    assert sha(BASE/'RUNTIME.json')==PARENT
    old=read(BASE/'RUNTIME.json');TARGET.mkdir(parents=True,exist_ok=False)
    for n,h in old['files'].items():
        assert sha(BASE/n)==h,n
        p=TARGET/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(BASE/n,p)
    path=TARGET/'braids_core/sediment/shoreline_solver.py';text=path.read_text(encoding='utf-8')
    retained=(ROOT/'braids_core/sediment/shoreline_solver.py').read_text(encoding='utf-8')
    function=retained[retained.index('def water_increment_roundoff_limit('):retained.index('\ndef build_native():')]
    assert 'water_increment_roundoff_limit' not in text
    text=text.replace('\ndef build_native():','\n'+function+'\ndef build_native():',1)
    oldcheck="            if abs(residual)>1e-10:raise BendTrialRejected('Shoreline water budget failed')"
    newcheck="""            inventory=float(xp.sum(xp.maximum(self.q[0],nextq[0])*p.g['area']))
            roundoff=water_increment_roundoff_limit(inventory,float(xp.sum(xp.abs(ledger))))
            if not np.isfinite(residual) or abs(residual)>roundoff:
                raise BendTrialRejected(f'Shoreline water budget failed: residual={residual:.6e} m3, roundoff={roundoff:.6e} m3')"""
    assert text.count(oldcheck)==1;text=text.replace(oldcheck,newcheck)
    olddiag='diagnostics=dict(p.last,water_residual_m3=residual,**integration))'
    assert text.count(olddiag)==1
    text=text.replace(olddiag,'diagnostics=dict(p.last,water_residual_m3=residual,water_increment_roundoff_limit_m3=roundoff,**integration))')
    path.write_text(text,encoding='utf-8')
    p=TARGET/'braids_desktop/__init__.py';t=p.read_text();assert 'preview57' in t;p.write_text(t.replace('preview57','preview58'))
    new=dict(old,files={n:sha(TARGET/n) for n in old['files']});save(TARGET/'RUNTIME.json',new)
    changed=[n for n in old['files'] if old['files'][n]!=new['files'][n]]
    assert set(changed)=={'braids_core/sediment/shoreline_solver.py','braids_desktop/__init__.py'}
    save(OUT/'STAGE.json',dict(parent=PARENT,runtime_sha256=sha(TARGET/'RUNTIME.json'),changed=changed,
        numerical_change='Arithmetic rejection bound only; no flux/source, CFL, timestep formula, reconstruction, native code or physical-law changes.'))
    print('Staged candidate',sha(TARGET/'RUNTIME.json'),flush=True)


def handoff():
    current=read(OUT/'STAGE.json');assert sha(TARGET/'RUNTIME.json')==current['runtime_sha256']
    incident=REPORT/'fixed_floor_trials';incident.mkdir(exist_ok=False)
    summary={}
    for name in ['nash_cpu','nash_gpu','zero_cpu']:
        source=REPORT/name;record=read(source/'STARTED.json');status=read(Path(record['output'])/'STATUS.json')
        assert status['status']=='cancelled'
        summary[name]=dict(output=record['output'],time_s=status['time_s'],outer_rejections=len(status.get('rejected_trials',[])),
            reasons=sorted({r['reason'] for r in status.get('rejected_trials',[])}),hydraulic=status['hydraulic'])
        assert source.resolve().is_relative_to(REPORT.resolve());source.rename(incident/name)
        for file in ['STATUS.json','DIAGNOSTICS.json','PROJECT.json','PROGRESS.json']:
            shutil.copy2(Path(record['output'])/file,incident/name/file)
    save(incident/'SUMMARY.json',summary)
    original=read(REPORT/'LOCKED_SPEC.json');(REPORT/'LOCKED_SPEC.json').rename(incident/'LOCKED_SPEC.json')
    new=dict(original,runtime_sha256=current['runtime_sha256'],revision='Same physical inputs; scale-aware arithmetic-bound candidate. Initial fixed-floor runs stopped and retained.',
        prior_runtime=PARENT)
    save(REPORT/'LOCKED_SPEC.json',new)
    for name,r in new['projects'].items():
        spec=Path(r['folder'])/'VALIDATION_SPEC.json';d=read(spec)
        shutil.copy2(spec,incident/(name+'_VALIDATION_SPEC.json'))
        d.update(runtime_sha256=current['runtime_sha256'],arithmetic='max(1e-10, 64*epsilon*max(1,inventory,absolute boundary transfer)) m3; no physical or CFL changes')
        save(spec,d)
    runner=(ROOT/'desktop/run_turbine_validation58_ui.py').read_text(encoding='utf-8')
    runner=runner.replace("RUNTIME=ROOT/'desktop/runtime'","RUNTIME=ROOT/'desktop/qualification/water_budget59/candidate'")
    runner=runner.replace('BRAIDS_Turbine_Validation58','BRAIDS_Turbine_Validation59').replace('reports/desktop_validation58','reports/desktop_validation59')
    (ROOT/'desktop/run_turbine_validation59_ui.py').write_text(runner,encoding='utf-8')
    check=(ROOT/'desktop/check_breaklines57.py').read_text(encoding='utf-8').replace("OUT=ROOT/'desktop/qualification/breaklines57'","OUT=ROOT/'desktop/qualification/water_budget59'")
    (ROOT/'desktop/check_water_budget59.py').write_text(check,encoding='utf-8')
    print('Preserved rejected-work incident and prepared corrected candidate runners',flush=True)


def regress():
    import numpy as np
    records=[]
    for backend in ['cpu','gpu']:
        for mode in ['hydro','bedload']:
            name=backend+'_'+mode
            completed=subprocess.run([sys.executable,'-B',str(ROOT/'desktop/check_water_budget59.py'),backend,mode],cwd=ROOT,capture_output=True,text=True,timeout=180)
            (OUT/(name+'.log')).write_text(completed.stdout+'\n'+completed.stderr,encoding='utf-8')
            assert completed.returncode==0,completed.stdout+completed.stderr
            oldroot=ROOT/'desktop/qualification/breaklines57'/name/'run';newroot=OUT/name/'run'
            def last(root):
                index=read(root/'frames/INDEX.json');r=index['frames'][-1]
                with np.load(root/'frames'/r['file']) as a:return {k:a[k].copy() for k in a.files}
            old=last(oldroot);new=last(newroot)
            exact={k:bool(np.array_equal(old[k],new[k])) for k in ['q','z','water','bedload_vector_m2_s']}
            assert all(exact.values()),(name,exact)
            records.append(dict(case=name,exact_accepted_preview57_fields=exact,check=read(OUT/name/'CHECK.json')))
            print('AMR/restart/reference check passed:',name,flush=True)
    save(OUT/'REGRESSION.json',dict(status='passed',cases=records))


def bounded():
    import numpy as np
    sys.path.insert(0,str(TARGET))
    from braids_desktop.results import ResultsStore
    root=REPORT/'bounded_cpu';root.mkdir(exist_ok=True)
    for name,gpu_name in [('nash_cpu','nash_gpu'),('zero_cpu','zero_gpu')]:
        if (root/(name+'.json')).exists():continue
        record=read(REPORT/name/'STARTED.json');output=Path(record['output']);progress=read(output/'PROGRESS.json')
        if progress['time_s']<100.:
            print(name,'not yet at 100s',flush=True);continue
        history=output.parent.parent/'history'/(output.name+'.log');first=json.loads(history.open(encoding='utf-8').readline())
        args=first['arguments'];flag=Path(args[args.index('--stop-file')+1]).resolve()
        assert flag.parent==output.parent.resolve() and flag.name.startswith('.braids_stop_')
        store=ResultsStore(output);idx=int(np.flatnonzero(store.times==100.)[0]);frame=store.read(idx)
        index=store.index['frames'][idx];shutil.copy2(output/'frames'/index['file'],root/(name+'_100s.npz'))
        flag.open('x',encoding='utf-8').close()
        save(root/(name+'.json'),dict(status='bounded_accepted_state_stop_requested',case=name,comparison_time_s=100.,
            output=str(output),matched_gpu=gpu_name,frame_sha256=index['sha256'],stop_requested_after_s=progress['time_s'],
            note='Full 2000s wake average uses explicit GPU projects. This CPU companion verifies the same coupled trajectory at 100s; its final state/checkpoint remains resumable.'))
        print('Saved 100s CPU companion and requested accepted-state stop:',name,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stop','stage','handoff','regress','bounded']);globals()[p.parse_args().action]()
