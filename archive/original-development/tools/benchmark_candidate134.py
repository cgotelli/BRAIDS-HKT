"""Sequential ABBA checkpoint-pipeline trial against immutable alpha133.

Both variants use the same fresh child bootstrap, inputs and public worker.
The only candidate difference is the explicitly identified validation hook.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def child(args):
    started = time.perf_counter()
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(args.runtime.resolve()))
    sys.dont_write_bytecode = True
    from braids_core.river_launcher import verify
    manifest = verify()
    from braids_core.sediment.portable_river_runtime import install
    install(json.loads((args.runtime/'PRECOMPILED.json').read_text()), args.runtime)
    from braids_core import river_worker as worker
    candidate = None
    if args.variant == 'memory':
        from tools.output_candidate134 import install as hook
        candidate = hook()
    loaded = time.perf_counter()
    result = worker.run(args.prepared, args.output/'run')
    completed = time.perf_counter()
    record = dict(variant=args.variant, startup_s=loaded-started, worker_s=completed-loaded,
                  through_run_s=completed-started, status=result,
                  candidate=None if candidate is None else dict(identity=candidate.identity,stats=candidate.stats),
                  runtime=str(args.runtime), source_identity=worker.source_identity())
    if verify() != manifest: raise AssertionError('Immutable runtime changed')
    if candidate is not None: candidate.close()
    (args.output/'CHILD.json').write_text(json.dumps(record,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--prepared',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--variant',choices=['baseline','memory'])
    p.add_argument('--inputs',type=Path,default=ROOT/'validation/_performance134/matched')
    p.add_argument('--cases',nargs='+',default=['cpu8_output1','gpu_output1'])
    a=p.parse_args(); a.runtime=a.runtime.resolve(); a.output=a.output.resolve()
    a.output.mkdir(parents=True,exist_ok=False)
    if a.variant:
        child(a); return
    exe=Path(sys.executable).resolve()
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',PYTHONPATH=str(a.runtime),
             PATH=os.pathsep.join([str(exe.parent/'Library/bin'),str(exe.parent),
                                  os.environ.get('SystemRoot','C:/Windows')+'/System32']))
    for key in ('PYTHONHOME','QGIS_PREFIX_PATH','QT_PLUGIN_PATH','BRAIDS_TEST_TOOLCHAIN'):env.pop(key,None)
    rows=[]
    for case in a.cases:
        source=(a.inputs/(case+'.json')).resolve(); prepared=a.output/(case+'_prepared')
        with (a.output/(case+'_prepare.log')).open('w') as log:
            subprocess.run([str(exe),str(a.runtime/'braids_core/river_launcher.py'),'prepare',
                            '--project',str(source),'--output',str(prepared)],env=env,cwd=a.output,
                           stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
        for index,variant in enumerate(('baseline','memory','memory','baseline')):
            out=a.output/f'{case}_{index}_{variant}'
            print('START',out.name,flush=True); began=time.perf_counter()
            with out.with_suffix('.log').open('w') as log:
                subprocess.run([str(exe),str(Path(__file__).resolve()),'--runtime',str(a.runtime),
                                '--prepared',str(prepared),'--output',str(out),'--variant',variant],
                               env=env,cwd=a.output,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
            record=json.loads((out/'CHILD.json').read_text())
            if record['status']['status']!='complete':raise AssertionError(record)
            rows.append(dict(record,case=case,order=index,process_s=time.perf_counter()-began,
                             run=str(out/'run'),project_sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
            (a.output/'BENCHMARK.json').write_text(json.dumps(dict(rows=rows,scope='Fresh-process ABBA, no timing overlap'),indent=2)+'\n')
            print('PASS',case,variant,round(record['worker_s'],4),flush=True)
    sys.path.insert(0,str(ROOT))
    from tools.profile_execution130 import compare_frames
    comparisons=[]
    for case in a.cases:
        subset=[r for r in rows if r['case']==case]; reference=Path(subset[0]['run'])
        for row in subset[1:]:
            match=compare_frames(Path(row['run']),reference)
            comparisons.append(dict(case=case,variant=row['variant'],order=row['order'],**match))
    (a.output/'EXACT.json').write_text(json.dumps(comparisons,indent=2)+'\n')
    if not all(r['exact'] for r in comparisons):raise AssertionError('Candidate altered saved fields')


if __name__=='__main__':main()
