"""Untouched alpha129 installed-runtime profile and six sequential clean runs.

Only checkout research hooks change numerical candidates. Every production module
and precompiled native implementation comes from the authenticated QGIS runtime.
Single samples are a bounded installed-path check, not a precise speedup estimate.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--skip-instrumented', action='store_true')
    ap.add_argument('--variants', nargs='+', choices=('baseline','conservative','selected'),
                    default=['baseline','conservative','selected'])
    args = ap.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=args.resume)
    records = json.loads((out/'BENCHMARK.json').read_text())['records'] if args.resume and (out/'BENCHMARK.json').exists() else []
    cpu = ROOT/'validation/_inlet129/natural5m_final/5m_critical_cpu'
    gpu = ROOT/'validation/_inlet129/natural2m_final/2m_critical_gpu'
    selected = dict(cpu5m=ROOT/'validation/_speed130/clock_matrix/cap100_outer1',
                    gpu2m=ROOT/'validation/_speed130/natural_combined_outer2')
    jobs = ([] if args.skip_instrumented else [('cpu5m', 'baseline', False)]) + [(case, variant, True)
        for case in ('cpu5m', 'gpu2m') for variant in args.variants]
    for case, variant, clean in jobs:
        name = case+'_'+variant+('_clean' if clean else '_instrumented')
        if any(r['name']==name for r in records):
            print('RETAIN', name, flush=True)
            continue
        baseline = cpu if case == 'cpu5m' else gpu
        source = baseline if variant == 'baseline' else selected[case]
        ref = selected[case] if variant == 'selected' else baseline
        target = out/name
        command = [sys.executable, str(ROOT/'tools/profile_installed_execution130.py'),
            '--source', str(source/'PROJECT.json'), '--reference-run', str(ref/'run'),
            '--until', '60' if case=='cpu5m' else '100', '--output', str(target),
            '--optimizations', 'none' if variant=='baseline' else 'mesh', '--numerics', variant]
        if clean:
            command.append('--clean')
        print('START', name, flush=True)
        began = time.perf_counter()
        with (out/(name+'.log')).open('w') as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        process_s = time.perf_counter()-began
        if result.returncode:
            raise RuntimeError('Failed '+name+'; inspect log')
        profile = json.loads((target/'PROFILE.json').read_text())
        receipt = json.loads((target/'INSTALLED_RUNTIME.json').read_text())
        if not receipt['all_solver_modules_installed'] or not receipt['source_integrity_after']:
            raise AssertionError('Installed source/module provenance failed')
        from tools.analyze_inlet129 import load_run, compare_runs
        loaded = load_run(target)
        comparisons = {label:compare_runs(loaded, load_run(folder)) for label,folder in (
            ('alpha129', baseline), ('selected_research', selected[case]))}
        for compare in comparisons.values():
            compare['depth_5mm_pass'] = compare['skipped_times']==0 and compare['depth_max_m']<=.005
        (target/'COMPARISONS.json').write_text(json.dumps(comparisons, indent=2)+'\n')
        if variant in ('baseline', 'selected') and not profile['exact_reference']['exact']:
            raise AssertionError('Installed trajectory differs from its matching checkout reference')
        if not comparisons['alpha129']['depth_5mm_pass']:
            raise AssertionError('Installed candidate fails 5mm baseline screen')
        records.append(dict(name=name,case=case,variant=variant,clean=clean,process_s=process_s,
            preparation_s=profile['preparation_s'],worker_s=profile['worker_wall_s'],
            status_wall_s=profile['result']['wall_s'],hydraulic=profile['result']['hydraulic'],
            exact_reference=profile['exact_reference']['exact'],installed_runtime=receipt['installed_runtime'],
            runtime=profile['runtime'],all_solver_modules_installed=receipt['all_solver_modules_installed'],
            source_integrity_after=receipt['source_integrity_after'],
            budget=loaded['summary']['budget'],
            comparisons={k:dict(depth_max_m=v['depth_max_m'],depth_rms_m=v['depth_rms_m'],
                skipped_times=v['skipped_times'],depth_5mm_pass=v['depth_5mm_pass']) for k,v in comparisons.items()}))
        (out/'BENCHMARK.json').write_text(json.dumps(dict(records=records), indent=2)+'\n')
        print('PASS',name,profile['worker_wall_s'],flush=True)
    summary={}
    for case in ('cpu5m','gpu2m'):
        vals={r['variant']:r['worker_s'] for r in records if r['case']==case and r['clean']}
        summary[case]=dict(worker_s=vals,speedup={k:vals['baseline']/v for k,v in vals.items()},
            reduction_percent={k:100*(1-v/vals['baseline']) for k,v in vals.items()})
    (out/'BENCHMARK.json').write_text(json.dumps(dict(records=records,summary=summary,
        note='Single clean fresh-process samples per variant. Installed precompiled constructors; no toolchain caching. Preparation, worker and whole process recorded separately.'), indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
