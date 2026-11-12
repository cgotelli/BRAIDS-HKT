"""Sequential compiler-free alpha131/132 comparison at unchanged CFL and inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from braids_gpu.atomic_output import write_json_atomic


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--candidate',type=Path,default=ROOT/'dist/unified-alpha132/braids_qgis')
    p.add_argument('--cases',nargs='+',choices=('cpu5m','gpu5m','gpu2m'),default=['cpu5m','gpu5m'])
    p.add_argument('--profile',choices=('baseline','candidate'))
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    records=[]
    for case in a.cases:
        old=ROOT/'validation/_amr131/clean_final'
        if case=='gpu2m':
            source=old/'gpu2m_CFL1.json';reference=old/'gpu2m_4_cfl1/run';until=100
        else:
            source=old/'cpu5m_CFL1.json';reference=old/'cpu5m_4_cfl1/run';until=60
        doc=json.loads(source.read_text())
        if case=='gpu5m':
            # Explicit matching GPU input, not an automatic backend choice.
            from braids_core.river_model import resolve_model
            doc['model']['request']['execution']['backend']='gpu'
            doc['model']=resolve_model(doc['model']['request']).document()
            retained=ROOT/'validation/_amr132/clean/gpu5m_0_baseline/run'
            reference=retained if retained.is_dir() else None
        selected=out/(case+'_PROJECT.json');write_json_atomic(selected,doc)
        order=[a.profile] if a.profile else ['baseline','candidate','candidate','baseline']
        for index,variant in enumerate(order):
            plugin=ROOT/'dist/unified-alpha131/braids_qgis' if variant=='baseline' else a.candidate.resolve()
            target=out/f'{case}_{index}_{variant}'
            cmd=[sys.executable,str(ROOT/'tools/profile_installed_execution130.py'),
                 '--runtime',str(plugin/'river_runtime'),'--source',str(selected),
                 '--until',str(until),'--output',str(target)]
            if reference is not None:cmd+=['--reference-run',str(reference)]
            if not a.profile:cmd.append('--clean')
            print('START',target.name,flush=True);start=time.perf_counter()
            with (out/(target.name+'.log')).open('w') as log:
                done=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            elapsed=time.perf_counter()-start
            if done.returncode:raise RuntimeError('Failed '+str(target))
            result=json.loads((target/'PROFILE.json').read_text())
            provenance=json.loads((target/'INSTALLED_RUNTIME.json').read_text())
            if not provenance['all_solver_modules_installed'] or not provenance['source_integrity_after']:
                raise AssertionError('Runtime provenance differs')
            exact=result['exact_reference']['exact'] if reference is not None else None
            if exact is False:raise AssertionError('Saved fields differ: '+str(target))
            if reference is None:reference=target/'run'
            row=dict(case=case,variant=variant,worker_s=result['worker_wall_s'],
                     preparation_s=result['preparation_s'],process_s=elapsed,exact=exact,
                     hydraulic=result['result']['hydraulic'],runtime=result['runtime'],
                     input_sha256=hashlib.sha256(selected.read_bytes()).hexdigest())
            records.append(row);write_json_atomic(out/'BENCHMARK.json',dict(records=records))
            print('PASS',target.name,row['worker_s'],flush=True)
    summary={}
    for case in a.cases:
        groups={v:[r['worker_s'] for r in records if r['case']==case and r['variant']==v]
                for v in ('baseline','candidate')}
        means={k:sum(v)/len(v) for k,v in groups.items() if v}
        summary[case]=dict(samples=groups,means=means,
            reduction_percent=100*(1-means['candidate']/means['baseline']) if len(means)==2 else None)
    write_json_atomic(out/'BENCHMARK.json',dict(records=records,summary=summary,
        scope='Unchanged physics, CFL and AMR schedule; sequential compiler-free ABBA; no installed profile changes.'))


if __name__=='__main__':main()
