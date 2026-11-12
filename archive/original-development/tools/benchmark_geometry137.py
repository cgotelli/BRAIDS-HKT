"""Fresh-process whole preparation/run timings and strict geometry qualification."""
from pathlib import Path
import argparse
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def qualify(root):
    from tools.analyze_execution134 import compare_runs
    raw=json.loads((root/'BENCHMARK.json').read_text())
    checks=[]
    for case in sorted({r['case'] for r in raw['rows']}):
        rows=sorted((r for r in raw['rows'] if r['case']==case),key=lambda r:r['index'])
        assert [r['variant'] for r in rows]==['before','after','after','before']
        assert len({r['source_sha256'] for r in rows})==1
        first=rows[0]
        for row in rows[1:]:
            result=compare_runs(first['run'],row['run'],scope='Geometry record reuse only; zero tolerance on every array, geometry, hydraulic diagnostic and budget.')
            budgets=lambda r:json.loads((Path(r['run'])/'DIAGNOSTICS.json').read_text())
            clocks=lambda r:{k:r['status'][k] for k in ('time_s','hydraulic','accepted_steps','rejected_trials')}
            passed=(result['all_common_arrays_exact'] and result['all_common_geometry_exact']
                    and clocks(first)==clocks(row) and budgets(first)==budgets(row))
            checks.append(dict(case=case,index=row['index'],passed=passed,comparison=result))
    receipt=dict(all_pass=all(c['passed'] for c in checks),checks=checks)
    (root/'STRICT_COMPARISON.json').write_text(json.dumps(receipt,indent=2)+'\n')
    assert receipt['all_pass']


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--before',type=Path,default=ROOT/'desktop/runtime')
    parser.add_argument('--after',type=Path,default=ROOT/'validation/_geometry137/candidate3')
    parser.add_argument('--output',type=Path,default=ROOT/'validation/_geometry137/abba_final')
    args=parser.parse_args()
    from tools.benchmark_preparation135 import main
    sys.argv=['benchmark_preparation135.py','--before',str(args.before),'--after',str(args.after),'--output',str(args.output)]
    main()
    qualify(args.output.resolve())
