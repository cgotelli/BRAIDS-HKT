"""Archive-preparation ABBA with strict unchanged-output qualification.

Reuses milestone135's measured fresh-process launcher and conservation checks.
The final COMPARISON.json is marked strict only after every output/geometry
array, hydraulic diagnostic, budget and repeated process passes exact checks.
This script launches jobs; run it only in an agreed clean timing slot.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def qualify_exact(root):
    # Delay scientific analysis imports until all timing subprocesses finish.
    from tools.analyze_execution134 import compare_runs
    from tools.analyze_preparation135 import read,sha
    raw=read(root/'BENCHMARK.json');receipt=read(root/'COMPARISON.json')
    checks=[]
    for name in receipt['summary']:
        rows=sorted((r for r in raw['rows'] if r['case']==name),key=lambda r:r['index'])
        if [r['variant'] for r in rows]!=['before','after','after','before']:
            raise ValueError('Incomplete before/after ABBA')
        if len({r['source_sha256'] for r in rows})!=1:raise ValueError('Input changed across releases')
        first=rows[0]
        for row in rows[1:]:
            result=compare_runs(first['run'],row['run'],scope='Milestone136 archive preparation only: every published array and geometry must remain exact, including metadata arrays.')
            a=read(Path(first['run'])/'frames/INDEX.json');b=read(Path(row['run'])/'frames/INDEX.json')
            same_times=[r['time_s'] for r in a['frames']]==[r['time_s'] for r in b['frames']]
            same_hydraulic=first['status']['hydraulic']==row['status']['hydraulic']
            same_outer=all(first['status'][key]==row['status'][key] for key in ('accepted_steps','rejected_trials','time_s'))
            same_budgets=read(Path(first['run'])/'DIAGNOSTICS.json')==read(Path(row['run'])/'DIAGNOSTICS.json')
            previous=next(c for c in receipt['comparisons'] if c['case']==name and c['index']==row['index'])
            numeric_bytes=previous['exact']
            passed=all((result['all_common_arrays_exact'],result['all_common_geometry_exact'],same_times,
                        same_hydraulic,same_outer,same_budgets,numeric_bytes))
            checks.append(dict(case=name,index=row['index'],variant=row['variant'],passed=passed,
                same_output_times=same_times,same_hydraulic_diagnostics=same_hydraulic,
                same_outer_transactions=same_outer,same_budgets=same_budgets,
                numeric_arrays_byte_exact=numeric_bytes,all_arrays_exact=result['all_common_arrays_exact'],
                all_geometry_exact=result['all_common_geometry_exact'],comparison=result))
    strict=dict(schema='braids.preparation136.strict-comparison.v1',
        benchmark_sha256=sha(root/'BENCHMARK.json'),all_pass=all(c['passed'] for c in checks),checks=checks)
    (root/'STRICT_COMPARISON.json').write_text(json.dumps(strict,indent=2)+'\n',encoding='utf-8')
    if not strict['all_pass']:raise AssertionError('Archive preparation changed output fields, geometry, clocks or budgets')
    receipt.update(schema='braids.preparation136.comparison.v1',absolute_field_limit=0.,
        strict_full_field_comparison=True,strict_receipt_sha256=sha(root/'STRICT_COMPARISON.json'),
        note='Source archive preparation optimization only. All published numerical arrays are byte-exact; all other published arrays, geometry, hydraulic diagnostics and budgets are unchanged. No roundoff allowance is used.')
    for check in receipt['comparisons']:
        check['roundoff_bound_pass']=check['exact']
    (root/'COMPARISON.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--before',type=Path,default=ROOT/'dist/unified-alpha135/braids_qgis/river_runtime')
    parser.add_argument('--after',type=Path,default=ROOT/'dist/unified-alpha136/braids_qgis/river_runtime')
    parser.add_argument('--output',type=Path,default=ROOT/'validation/_preparation136/release_abba')
    parser.add_argument('--cases',nargs='+',default=['cpu8_clock1_output1','gpu_clock1_output1'])
    args=parser.parse_args()
    from tools.benchmark_preparation135 import main as run_abba
    old_argv=sys.argv
    try:
        sys.argv=['benchmark_preparation135.py','--before',str(args.before),'--after',str(args.after),
                  '--output',str(args.output),'--cases',*args.cases]
        run_abba()
    finally:sys.argv=old_argv
    qualify_exact(args.output.resolve())
    print('PASS strict full-field milestone136 qualification',flush=True)


if __name__=='__main__':main()
