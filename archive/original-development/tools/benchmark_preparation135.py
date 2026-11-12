"""Same-input packaged preparation/run ABBA, with roundoff-level terrain checks."""
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
    runtime={k:getattr(a,k).resolve() for k in ('before','after')};rows=[]
    exe=Path(sys.executable).resolve()
    def execute(root,name,args):
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',PYTHONPATH=str(root),
            PATH=os.pathsep.join([str(exe.parent/'Library/bin'),str(exe.parent),
                                 os.environ.get('SystemRoot','C:/Windows')+'/System32']))
        for key in ('PYTHONHOME','QGIS_PREFIX_PATH','QT_PLUGIN_PATH','BRAIDS_TEST_TOOLCHAIN'):env.pop(key,None)
        began=time.perf_counter()
        with (out/(name+'.log')).open('w') as log:
            subprocess.run([str(exe),str(root/'braids_core/river_launcher.py'),*map(str,args)],
                env=env,cwd=out,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
        return time.perf_counter()-began
    for case in a.cases:
        for index,variant in enumerate(('before','after','after','before')):
            name=f'{case}_{index}_{variant}';r=runtime[variant]
            source=ROOT/'validation/_performance134/current_inputs'/(case+'.json')
            digest=hashlib.sha256(source.read_bytes()).hexdigest()
            prepared=out/(name+'_prepared');run=out/(name+'_run');print('START',name,flush=True)
            preparation=execute(r,name+'_prepare',['prepare','--project',source,'--output',prepared])
            elapsed=execute(r,name+'_run',['run','--input',prepared,'--output',run])
            status=json.loads((run/'STATUS.json').read_text())
            if status['status']!='complete':raise AssertionError(status)
            budgets=json.loads((run/'DIAGNOSTICS.json').read_text())
            conserved=all(abs(b['water_residual_m3'])<=b['water_roundoff_limit_m3']
                and all(abs(x)<=b['class_roundoff_limit_kg'] for x in b['class_residual_kg'])
                and not any(b['wall_integrals']) for b in budgets)
            if not conserved:raise AssertionError('Conservation or impermeable-wall check failed')
            if digest!=hashlib.sha256(source.read_bytes()).hexdigest():raise AssertionError('Input changed')
            rows.append(dict(case=case,index=index,variant=variant,runtime=str(r),
                runtime_manifest_sha256=hashlib.sha256((r/'RUNTIME.json').read_bytes()).hexdigest(),
                source=str(source),source_sha256=digest,preparation_process_s=preparation,
                run_process_s=elapsed,total_process_s=preparation+elapsed,core_wall_s=status['wall_s'],
                status=status,prepared=str(prepared),run=str(run),conservation_pass=conserved,
                maximum_water_residual_m3=max(abs(b['water_residual_m3']) for b in budgets)))
            (out/'BENCHMARK.json').write_text(json.dumps(dict(rows=rows,scope='Sequential fresh-process same-input ABBA, no profiler'),indent=2)+'\n')
            print('PASS',name,'prepare',round(preparation,3),'run',round(elapsed,3),flush=True)
    sys.path.insert(0,str(ROOT))
    from tools.profile_execution130 import compare_frames
    comparisons=[];summary={}
    for case in a.cases:
        selected=[r for r in rows if r['case']==case];ref=selected[0]
        for r in selected[1:]:
            match=compare_frames(Path(r['run']),Path(ref['run']))
            differences=[d for f in match['frames'] for d in f['differences']]
            bounded=all(d.get('maximum') is not None and d['maximum']<=1e-7 and d['shape'][0]==d['shape'][1] for d in differences)
            counters=('accepted_substeps','fwave_evaluations','cfl_retries','internal_retries','accuracy_retries','nonlinear_retries')
            same_counts=all(r['status']['hydraulic'].get(k)==ref['status']['hydraulic'].get(k) for k in counters)
            same_budget=json.loads((Path(r['run'])/'DIAGNOSTICS.json').read_text())==json.loads((Path(ref['run'])/'DIAGNOSTICS.json').read_text())
            comparisons.append(dict(case=case,variant=r['variant'],index=r['index'],same_counts=same_counts,
                same_budgets=same_budget,roundoff_bound_pass=bounded,maximum_abs_field_difference=max([d.get('maximum',0) or 0 for d in differences],default=0.),**match))
        summary[case]={v:{k:statistics.median(r[k] for r in selected if r['variant']==v)
            for k in ('preparation_process_s','run_process_s','core_wall_s','total_process_s')} for v in ('before','after')}
    result=dict(summary=summary,comparisons=comparisons,absolute_field_limit=1e-7,
        note='Same exact bilinear surface with changed floating-point summation. Record exactness; do not claim byte identity when only roundoff bounds pass.')
    (out/'COMPARISON.json').write_text(json.dumps(result,indent=2)+'\n')
    if not all(c['roundoff_bound_pass'] and c['same_counts'] for c in comparisons):raise AssertionError('Preparation changed fields or hydraulic work beyond roundoff screen')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
