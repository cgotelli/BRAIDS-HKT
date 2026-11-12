"""Fresh prepare+run ABBA comparisons of immutable before/after releases."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--before',type=Path,required=True)
    p.add_argument('--after',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cases',nargs='+',default=['cpu8_clock1_output1','gpu_clock1_output1'])
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    runtimes={k:getattr(a,k).resolve() for k in ('before','after')}
    inputs={'before':ROOT/'validation/_performance134/matched','after':ROOT/'validation/_performance134/current_inputs'}
    exe=Path(sys.executable).resolve();rows=[]
    def execute(runtime,name,args):
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',PYTHONPATH=str(runtime),
                 PATH=os.pathsep.join([str(exe.parent/'Library/bin'),str(exe.parent),
                                      os.environ.get('SystemRoot','C:/Windows')+'/System32']))
        for key in ('PYTHONHOME','QGIS_PREFIX_PATH','QT_PLUGIN_PATH','BRAIDS_TEST_TOOLCHAIN'):env.pop(key,None)
        began=time.perf_counter()
        with (out/(name+'.log')).open('w') as log:
            subprocess.run([str(exe),str(runtime/'braids_core/river_launcher.py'),*map(str,args)],
                env=env,cwd=out,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
        return time.perf_counter()-began
    for case in a.cases:
        for index,variant in enumerate(('before','after','after','before')):
            name=f'{case}_{index}_{variant}';runtime=runtimes[variant];source=inputs[variant]/(case+'.json')
            digest=hashlib.sha256(source.read_bytes()).hexdigest();prepared=out/(name+'_prepared');run=out/(name+'_run')
            print('START',name,flush=True)
            preparation=execute(runtime,name+'_prepare',['prepare','--project',source,'--output',prepared])
            elapsed=execute(runtime,name+'_run',['run','--input',prepared,'--output',run])
            status=json.loads((run/'STATUS.json').read_text())
            if status['status']!='complete':raise AssertionError(status)
            if digest!=hashlib.sha256(source.read_bytes()).hexdigest():raise AssertionError('Input changed')
            rows.append(dict(case=case,index=index,variant=variant,runtime=str(runtime),
                runtime_manifest_sha256=hashlib.sha256((runtime/'RUNTIME.json').read_bytes()).hexdigest(),
                source=str(source),source_sha256=digest,preparation_process_s=preparation,
                run_process_s=elapsed,core_wall_s=status['wall_s'],total_process_s=preparation+elapsed,
                status=status,run=str(run)))
            (out/'BENCHMARK.json').write_text(json.dumps(dict(rows=rows,scope='Sequential fresh prepare+run ABBA; no profiler'),indent=2)+'\n')
            print('PASS',name,'prepare',round(preparation,3),'run',round(elapsed,3),flush=True)
    sys.path.insert(0,str(ROOT))
    from tools.profile_execution130 import compare_frames
    comparisons=[];summary={}
    for case in a.cases:
        selected=[r for r in rows if r['case']==case];ref=selected[0]
        for r in selected[1:]:
            match=compare_frames(Path(r['run']),Path(ref['run']))
            same_clock=r['status']['hydraulic']==ref['status']['hydraulic']
            same_budget=json.loads((Path(r['run'])/'DIAGNOSTICS.json').read_text())==json.loads((Path(ref['run'])/'DIAGNOSTICS.json').read_text())
            comparisons.append(dict(case=case,variant=r['variant'],index=r['index'],same_clock=same_clock,same_budgets=same_budget,**match))
        summary[case]={variant:{key:statistics.median(r[key] for r in selected if r['variant']==variant)
                        for key in ('preparation_process_s','run_process_s','core_wall_s','total_process_s')}
                       for variant in ('before','after')}
    result=dict(summary=summary,comparisons=comparisons)
    (out/'EXACT_AND_TIMING.json').write_text(json.dumps(result,indent=2)+'\n')
    if not all(r['exact'] and r['same_clock'] and r['same_budgets'] for r in comparisons):
        raise AssertionError('I/O-only release changed numerical results or clocks')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
