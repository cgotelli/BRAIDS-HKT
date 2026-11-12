"""Compiler-free release checks against current, corrected checkout references.

Default smoke decks are the explicitly derived v19 release projects. Optional
natural references must contain PROJECT.json, QUALIFICATION.json and run/.
No old alpha126/127 result is treated as the corrected numerical reference.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def latest(folder):
    rows=json.loads((folder/'frames/INDEX.json').read_text())['frames']
    with np.load(folder/'frames'/rows[-1]['file'],allow_pickle=False) as f:
        return {k:f[k].copy() for k in f.files}


def compare_frames(current,reference):
    rows=json.loads((current/'frames/INDEX.json').read_text())['frames']
    before=json.loads((reference/'frames/INDEX.json').read_text())['frames']
    assert len(rows)==len(before),(len(rows),len(before))
    errors=[]
    for row,old in zip(rows,before):
        assert row['time_s']==old['time_s'] and row['steps']==old['steps']
        with np.load(current/'frames'/row['epoch'],allow_pickle=False) as a,np.load(reference/'frames'/old['epoch'],allow_pickle=False) as b:
            assert set(a.files)==set(b.files)
            for key in a.files:np.testing.assert_array_equal(a[key],b[key],err_msg='mesh '+key)
        with np.load(current/'frames'/row['file'],allow_pickle=False) as a,np.load(reference/'frames'/old['file'],allow_pickle=False) as b:
            assert set(a.files)==set(b.files)
            values={}
            for key in a.files:
                if a[key].dtype.kind in 'iufc':
                    values[key]=float(np.max(np.abs(a[key]-b[key]))) if a[key].size else 0.
                    np.testing.assert_allclose(a[key],b[key],atol=1e-10,rtol=0,err_msg=key)
                else:np.testing.assert_array_equal(a[key],b[key])
            errors.append(dict(time_s=row['time_s'],maximum_absolute_error=values))
    return errors


def main():
    p=argparse.ArgumentParser();p.add_argument('--plugin',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--projects',type=Path,default=ROOT/'validation/_inlet129/release_projects')
    p.add_argument('--reference-folder',type=Path)
    p.add_argument('--reference-case',action='append',default=[],help='Optional exact subdirectory name; default all completed cases')
    p.add_argument('--natural-only',action='store_true')
    p.add_argument('--python',type=Path,default=Path(sys.executable))
    a=p.parse_args();plugin=a.plugin.resolve();runtime=plugin/'river_runtime';out=a.output.resolve()
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    executable=a.python.resolve()
    env=dict(os.environ,PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(runtime),
        PATH=os.pathsep.join([str(executable.parent/'Library/bin'),str(executable.parent),
            os.environ.get('SystemRoot','C:/Windows')+'/System32']))
    for key in ('PYTHONHOME','QGIS_PREFIX_PATH','QT_PLUGIN_PATH','BRAIDS_TEST_TOOLCHAIN'):
        env.pop(key,None)
    launcher=runtime/'braids_core/river_launcher.py';checks={};runs={};last_project=None;last_prepared=None
    def call(name,arguments,success=True):
        start=time.perf_counter()
        result=subprocess.run([str(executable),str(launcher),*map(str,arguments)],cwd=out,env=env,
            capture_output=True,text=True,timeout=900)
        (out/(name+'.log')).write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
        if success and result.returncode:raise RuntimeError(name+' failed:\n'+result.stdout+'\n'+result.stderr)
        return result.returncode,time.perf_counter()-start
    def save():
        (out/'CHECKS.json').write_text(json.dumps(dict(plugin=str(plugin),checks=checks,runs=runs,
            compiler_in_path=False,scope='Deployment equivalence/restart checks; physical reference qualification remains separate'),indent=2),encoding='utf-8')
    def run_case(source,name,reference=None,restart=False):
        nonlocal last_project,last_prepared
        raw=source.read_bytes();doc=json.loads(raw);case=out/name;case.mkdir()
        prep=case/'prepared';run=case/'run';times={}
        _,times['prepare']=call(name+'_prepare',['prepare','--project',source,'--output',prep])
        _,times['run']=call(name+'_run',['run','--input',prep,'--output',run])
        status=json.loads((run/'STATUS.json').read_text());context=json.loads((run/'RUN_CONTEXT.json').read_text())
        checks[name+'_complete']=status['status']=='complete'
        checks[name+'_same_backend']=context['model']['request']['execution']['backend']==doc['model']['request']['execution']['backend']
        checks[name+'_explicit']=status['integrator']=='explicit_fwave_ssprk22_river_v1' and status['hydraulic']['nonlinear_retries']==0
        checks[name+'_results']=(run/'frames/INDEX.json').is_file()
        checks[name+'_checkpoint']=(run/'checkpoint.npz').is_file()
        rows=json.loads((run/'DIAGNOSTICS.json').read_text())
        checks[name+'_water_budget']=all(abs(r['water_residual_m3'])<=r['water_roundoff_limit_m3'] for r in rows)
        checks[name+'_wall_flux']=all(not any(r['wall_integrals']) for r in rows)
        result=dict(project=str(source),source_sha256=hashlib.sha256(raw).hexdigest(),run=str(run),status=status,
            times_s=times,timing_scope='Packaged deployment check; not an isolated speed benchmark')
        if reference is not None:
            result['reference']=str(reference)
            result['frame_comparison']=compare_frames(run,reference/'run')
            checks[name+'_reference_geometry_and_all_fields']=True
            expected=json.loads((reference/'run/STATUS.json').read_text())
            checks[name+'_reference_hydraulic_counts']=status['hydraulic']==expected['hydraulic']
        if restart:
            half=case/'half';resume=case/'resume';midpoint=doc['settings']['end_time_s']/2
            call(name+'_half',['run','--input',prep,'--output',half,'--until',midpoint])
            call(name+'_resume',['run','--input',prep,'--output',resume,'--restart',half/'checkpoint.npz'])
            checks[name+'_restart_complete']=json.loads((resume/'STATUS.json').read_text())['status']=='complete'
            original,continued=latest(run),latest(resume)
            checks[name+'_restart_all_fields_exact']=set(original)==set(continued) and all(np.array_equal(v,continued[k]) for k,v in original.items())
        checks[name+'_source_unchanged']=source.read_bytes()==raw
        runs[name]=result;last_project=source;last_prepared=prep;save()
        if not all(checks.values()):raise AssertionError({k:v for k,v in checks.items() if not v})
        print('Packaged qualification passed',name,flush=True)
    if not a.natural_only:
        for mode in ('dynamic','static'):
            for backend in ('cpu','gpu'):
                for physics in ('hydro','bedload'):
                    name=f'{mode}_{backend}_{physics}'
                    run_case(a.projects.resolve()/f'check_{name}.json','smoke_'+name,
                        restart=mode=='dynamic' and physics=='bedload')
    if a.reference_folder:
        root=a.reference_folder.resolve()
        references=[root/name for name in a.reference_case] if a.reference_case else sorted(p for p in root.iterdir() if p.is_dir())
        if not references:raise ValueError('No reference cases')
        for reference in references:
            if not (reference/'QUALIFICATION.json').is_file():raise ValueError('Missing completed current qualification: '+str(reference))
            qualification=json.loads((reference/'QUALIFICATION.json').read_text())
            if qualification['status']['status']!='complete':raise ValueError('Reference is incomplete: '+str(reference))
            run_case(reference/'PROJECT.json','natural_'+reference.name,reference=reference)
    if last_project is None:raise ValueError('No qualification cases selected')
    stop=out/'STOP';stop.touch()
    rc,_=call('cancel_prepare',['prepare','--project',last_project,'--output',out/'cancel_prepare','--stop-file',stop],success=False)
    checks['cancel_before_prepare']=rc!=0 and not (out/'cancel_prepare/PREPARED.json').exists()
    rc,_=call('cancel_run',['run','--input',last_prepared,'--output',out/'cancel_run','--stop-file',stop],success=False)
    cancelled=out/'cancel_run/STATUS.json'
    if cancelled.is_file():
        state=json.loads(cancelled.read_text());checks['cancel_before_run']=state['status']=='cancelled' and state['time_s']==0.
    else:checks['cancel_before_run']=rc!=0
    old=json.loads(last_project.read_text());old['schema']='braids.river-project.v18'
    old_path=out/'unsupported_v18.json';old_path.write_text(json.dumps(old),encoding='utf-8')
    rc,_=call('reject_old',['prepare','--project',old_path,'--output',out/'unsupported_prepared'],success=False)
    checks['reject_old_project']=rc!=0 and not (out/'unsupported_prepared').exists()
    save()
    if not all(checks.values()):raise AssertionError(checks)
    print(len(checks),'compiler-free checks passed',flush=True)


if __name__=='__main__':main()
