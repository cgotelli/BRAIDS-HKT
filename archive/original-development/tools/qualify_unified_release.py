"""Short compiler-free public launcher checks on the staged/installed payload."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--plugin',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--project',type=Path,action='append',required=True)
    a=p.parse_args();plugin=a.plugin.resolve();runtime=plugin/'river_runtime';out=a.output.resolve()
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    # No developer compiler, checkout, PYTHONPATH or QGIS DLLs in the worker.
    executable=Path(sys.executable).resolve()
    env=dict(os.environ,PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(runtime),
             PATH=os.pathsep.join([str(executable.parent/'Library/bin'),str(executable.parent),
                                   os.environ.get('SystemRoot','C:/Windows')+'/System32']))
    for key in ('PYTHONHOME','QGIS_PREFIX_PATH','QT_PLUGIN_PATH','BRAIDS_TEST_TOOLCHAIN'):
        env.pop(key,None)
    launcher=runtime/'braids_core/river_launcher.py';checks={};runs={}
    def call(name,args,success=True):
        done=subprocess.run([str(executable),str(launcher),*map(str,args)],cwd=out,env=env,
                            capture_output=True,text=True,timeout=240)
        (out/(name+'.log')).write_text(done.stdout+'\n'+done.stderr,encoding='utf-8')
        if success and done.returncode:raise RuntimeError(name+' failed: '+done.stdout+'\n'+done.stderr)
        return done.returncode
    for i,source in enumerate(a.project):
        doc=json.loads(source.read_text());name=f'{i:02d}_{source.stem}'
        # A project file is kept intact: no changed physics to pass packaging.
        project=out/(name+'.json');project.write_text(json.dumps(doc,indent=2))
        prep=out/(name+'_prepared');run=out/(name+'_run')
        call(name+'_prepare',['prepare','--project',project,'--output',prep])
        call(name+'_run',['run','--input',prep,'--output',run])
        status=json.loads((run/'STATUS.json').read_text());checks[name+'_complete']=status['status']=='complete'
        checks[name+'_results']=(run/'frames/INDEX.json').is_file()
        checks[name+'_checkpoint']=(run/'checkpoint.npz').is_file()
        runs[name]=str(run)
        # A split trajectory, not merely loading a completed checkpoint.
        half=out/(name+'_half')
        midpoint=doc['settings']['end_time_s']/2
        call(name+'_half',['run','--input',prep,'--output',half,'--until',midpoint])
        resume=out/(name+'_resume')
        call(name+'_resume',['run','--input',prep,'--output',resume,'--restart',half/'checkpoint.npz'])
        checks[name+'_restart']=json.loads((resume/'STATUS.json').read_text())['status']=='complete'
        def frame(folder):
            index=json.loads((folder/'frames/INDEX.json').read_text())
            return np.load(folder/'frames'/index['frames'][-1]['file'],allow_pickle=False)
        with frame(run) as original,frame(resume) as continued:
            for key in ('q','z','mass','W'):
                checks[name+'_restart_exact_'+key]=np.array_equal(original[key],continued[key])
        print('Qualified',name,flush=True)
    stop=out/'STOP';stop.touch()
    rc=call('cancel_prepare',['prepare','--project',project,'--output',out/'cancelled','--stop-file',stop],False)
    checks['cancel_before_prepare']=rc!=0 and not (out/'cancelled/PREPARED.json').exists()
    old=out/'old.json';old.write_text('{"schema":"braids.river-project.v11"}')
    rc=call('reject_old',['prepare','--project',old,'--output',out/'old_prepared'],False)
    checks['reject_old']=rc!=0 and not (out/'old_prepared').exists()
    result=dict(plugin=str(plugin),checks=checks,runs=runs,compiler_in_path=False,
                scope='short deployment/restart checks, not new physical validation')
    (out/'CHECKS.json').write_text(json.dumps(result,indent=2))
    if not all(checks.values()):raise AssertionError(checks)
    print(json.dumps(checks),flush=True)


if __name__=='__main__':main()
