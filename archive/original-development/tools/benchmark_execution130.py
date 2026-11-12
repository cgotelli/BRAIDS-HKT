"""Sequential ABBA clean timings for isolated execution-only optimizations.

Run only after coordinating an otherwise idle CPU/GPU measurement window.
Both variants retain alpha129 numerical methods and exact output comparison.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--cases', nargs='+', choices=('cpu5m', 'gpu2m'), default=['cpu5m', 'gpu2m'])
    ap.add_argument('--sequence', nargs='+', choices=('none', 'both', 'mesh', 'toolchain'), default=['none', 'both', 'both', 'none'])
    ap.add_argument('--include-selected', action='store_true', help='After equivalent ABBA, two runs of qualified cap1/combined-clock candidate per case')
    ap.add_argument('--resume', action='store_true', help='Reuse only completed, already recorded fresh-worker measurements')
    args = ap.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=args.resume)
    records = json.loads((out/'BENCHMARK.json').read_text())['records'] if args.resume and (out/'BENCHMARK.json').exists() else []
    for case in args.cases:
        folder, until = ('natural5m_final/5m_critical_cpu', 60) if case == 'cpu5m' else ('natural2m_final/2m_critical_gpu', 100)
        ref = ROOT / 'validation/_inlet129' / folder
        for index, variant in enumerate(args.sequence):
            if any(r['case']==case and r['variant']==variant and r['sequence']==index for r in records):
                print('RETAIN', case, index, variant, flush=True)
                continue
            target = out / f'{case}_{index}_{variant}'
            command = [sys.executable, str(ROOT/'tools/profile_execution130.py'), '--clean', '--optimizations', variant,
                '--source', str(ref/'PROJECT.json'), '--reference-run', str(ref/'run'), '--until', str(until), '--output', str(target)]
            print('START', target.name, flush=True)
            began = time.perf_counter()
            with (out/f'{target.name}.log').open('w') as log:
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f'Failed {target.name}; inspect its log')
            profile = json.loads((target/'PROFILE.json').read_text())
            if not profile['exact_reference']['exact']:
                raise AssertionError(f'Numerical difference in {target.name}')
            records.append(dict(case=case, variant=variant, sequence=index, process_s=time.perf_counter()-began,
                preparation_s=profile['preparation_s'], worker_s=profile['worker_wall_s'], status_wall_s=profile['result']['wall_s'],
                hydraulic=profile['result']['hydraulic'], exact_reference=True, runtime=profile['runtime']))
            (out/'BENCHMARK.json').write_text(json.dumps(dict(records=records), indent=2)+'\n')
            print('PASS', target.name, profile['worker_wall_s'], flush=True)
        if args.include_selected:
            selected_ref = ROOT/'validation/_speed130'/('clock_matrix/cap100_outer1' if case == 'cpu5m' else 'natural_combined_outer2')
            for index in range(2):
                if any(r['case']==case and r['variant']=='selected' and r['sequence']==index for r in records):
                    continue
                target = out/f'{case}_selected_{index}'
                command = [sys.executable, str(ROOT/'tools/profile_execution130.py'), '--clean', '--optimizations', 'both',
                    '--numerics', 'selected', '--source', str(selected_ref/'PROJECT.json'),
                    '--reference-run', str(selected_ref/'run'), '--until', str(until), '--output', str(target)]
                print('START', target.name, flush=True)
                began = None
                if not (args.resume and (target/'PROFILE.json').exists()):
                    began = time.perf_counter()
                    with (out/f'{target.name}.log').open('w') as log:
                        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                    if result.returncode:
                        raise RuntimeError(f'Failed {target.name}; inspect its log')
                profile = json.loads((target/'PROFILE.json').read_text())
                from tools.analyze_inlet129 import load_run, compare_runs
                comparisons = {name: compare_runs(load_run(target), load_run(folder))
                    for name, folder in (('alpha129', ref), ('selected_research', selected_ref))}
                for comparison in comparisons.values():
                    comparison['depth_5mm_pass'] = (comparison['skipped_times'] == 0 and comparison['depth_max_m'] <= .005)
                if not all(c['depth_5mm_pass'] for c in comparisons.values()):
                    (target/'COMPARISONS.json').write_text(json.dumps(comparisons, indent=2)+'\n')
                    raise AssertionError(f'5mm numerical depth comparison failed for {target.name}')
                (target/'COMPARISONS.json').write_text(json.dumps(comparisons, indent=2)+'\n')
                records.append(dict(case=case, variant='selected', sequence=index, process_s=None if began is None else time.perf_counter()-began,
                    preparation_s=profile['preparation_s'], worker_s=profile['worker_wall_s'], status_wall_s=profile['result']['wall_s'],
                    hydraulic=profile['result']['hydraulic'], exact_reference=profile['exact_reference']['exact'],
                    runtime=profile['runtime'], comparisons={name:dict(depth_max_m=c['depth_max_m'], depth_rms_m=c['depth_rms_m'],
                        skipped_times=c['skipped_times'], depth_5mm_pass=c['depth_5mm_pass']) for name, c in comparisons.items()}))
                (out/'BENCHMARK.json').write_text(json.dumps(dict(records=records), indent=2)+'\n')
                print('PASS', target.name, profile['worker_wall_s'], flush=True)
    summary = {}
    for case in args.cases:
        means = {variant: float(np.mean([v['worker_s'] for v in records if v['case']==case and v['variant']==variant]))
                 for variant in {r['variant'] for r in records if r['case']==case}}
        summary[case] = dict(mean_worker_s=means,
            speedup=means['none']/means['both'] if 'none' in means and 'both' in means else None,
            reduction_percent=100*(1-means['both']/means['none']) if 'none' in means and 'both' in means else None,
            combined_speedup=means['none']/means['selected'] if 'none' in means and 'selected' in means else None,
            combined_reduction_percent=100*(1-means['selected']/means['none']) if 'none' in means and 'selected' in means else None)
    (out/'BENCHMARK.json').write_text(json.dumps(dict(records=records, summary=summary,
        note='Sequential clean fresh-worker ABBA; fixed numerical settings, exact saved fields and meshes; fresh preparation excluded from worker wall'), indent=2)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
