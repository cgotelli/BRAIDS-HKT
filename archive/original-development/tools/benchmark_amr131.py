"""Sequential compiler-free AMR ABBA and bounded public CFL1 runs.

No solver monkeypatches. Baseline is the authenticated milestone130 benchmark
copy; candidate is maintained131. Sources, outer clocks and CFL0.5 match for
the implementation-only comparison. CFL1 has a separate recorded input.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import hashlib

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from braids_gpu.atomic_output import write_json_atomic


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--cases',nargs='+',default=['cpu5m','gpu2m'])
    ap.add_argument('--profile',action='store_true')
    ap.add_argument('--profile-variant',choices=('baseline','candidate'),default='candidate')
    ap.add_argument('--candidate',default='candidate131')
    a=ap.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    records=[]
    for case in a.cases:
        source=ROOT/'validation/_speed130'/('production_cpu5m' if case=='cpu5m' else 'production_gpu2m')/'PROJECT.json'
        ref=source.parent/'run'
        sequence=[a.profile_variant] if a.profile else ['baseline','candidate','candidate','baseline','cfl1']
        for index,variant in enumerate(sequence):
            root=ROOT/'validation/_amr131'/('baseline130' if variant=='baseline' else a.candidate)/'braids_qgis/river_runtime'
            selected=source
            if variant=='cfl1':
                doc=json.loads(source.read_text());doc['settings']['controls']['cfl']=1.
                selected=out/(case+'_CFL1.json');write_json_atomic(selected,doc)
            target=out/f'{case}_{index}_{variant}'
            cmd=[sys.executable,str(ROOT/'tools/profile_installed_execution130.py'),
                '--runtime',str(root),'--source',str(selected),'--reference-run',str(ref),
                '--until','60' if case=='cpu5m' else '100','--output',str(target)]
            if not a.profile:cmd.append('--clean')
            print('START',target.name,flush=True)
            start=time.perf_counter()
            with (out/(target.name+'.log')).open('w') as log:
                result=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            elapsed=time.perf_counter()-start
            if result.returncode:raise RuntimeError('Failed '+str(target))
            p=json.loads((target/'PROFILE.json').read_text())
            provenance=json.loads((target/'INSTALLED_RUNTIME.json').read_text())
            if not provenance['all_solver_modules_installed'] or not provenance['source_integrity_after']:
                raise AssertionError('Runtime provenance failed')
            exact=p['exact_reference']['exact']
            if variant!='cfl1' and not exact:raise AssertionError('Implementation changed fields: '+str(target))
            row=dict(case=case,variant=variant,index=index,worker_s=p['worker_wall_s'],
                preparation_s=p['preparation_s'],process_s=elapsed,exact=exact,
                hydraulic=p['result']['hydraulic'],runtime=p['runtime'],
                source=str(selected),input_sha256=hashlib.sha256(selected.read_bytes()).hexdigest())
            records.append(row);write_json_atomic(out/'BENCHMARK.json',dict(records=records))
            print('PASS',target.name,row['worker_s'],'exact',exact,flush=True)
    summary={}
    for case in a.cases:
        d={v:[r['worker_s'] for r in records if r['case']==case and r['variant']==v] for v in ('baseline','candidate','cfl1')}
        means={k:sum(v)/len(v) for k,v in d.items() if v}
        summary[case]=dict(samples=d,means=means,
            speedup={k:means['baseline']/v for k,v in means.items()} if 'baseline' in means else {})
    write_json_atomic(out/'BENCHMARK.json',dict(records=records,summary=summary,
        scope='Compiler-free sequential ABBA whole worker; CFL1 is separate input/numerical comparison. No installed profile changed.'))


if __name__=='__main__':main()
