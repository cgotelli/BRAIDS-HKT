"""Bounded isolated CFL/threshold experiment, not a speed benchmark."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = ROOT/'validation/_inlet129/canonical_inputs_final'
    rows = []
    experiments = [('cpu', cfl, 1e-6) for cfl in (.5, .75, 1., 1.5)]
    experiments += [('cpu', 1., dry) for dry in (1e-4, 1e-3)]
    experiments += [('gpu', 1., 1e-6)]
    for backend, cap, dry in experiments:
        for method in ('critical', 'normal'):
            label = f'{backend}_{method}_cfl{cap:g}_dry{dry:g}'
            dest = out/label
            command = [sys.executable, str(ROOT/'tools/research_wet_cfl130.py'),
                '--source', str(source/f'static_{backend}_{method}.json'),
                '--analytic', str(source/'INPUTS.json'), '--output', str(dest),
                '--cfl', str(cap), '--dry', str(dry)]
            if (cap, dry) != (.5, 1e-6):
                reference = out/(f'cpu_{method}_cfl1_dry1e-06' if backend == 'gpu'
                                else f'cpu_{method}_cfl0.5_dry1e-06')
                command += ['--reference', str(reference)]
            print('START', label, flush=True)
            with (out/(label+'.log')).open('x', encoding='utf-8') as log:
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            row = dict(label=label, returncode=result.returncode, command=command)
            rows.append(row)
            (out/'RUNS.json').write_text(json.dumps(dict(
                timing_scope='Concurrent instrumented qualification, NOT a benchmark', runs=rows), indent=2))
            print('FINISHED', label, result.returncode, flush=True)
            if result.returncode:
                print((out/(label+'.log')).read_text(encoding='utf-8')[-6000:], flush=True)
    if any(row['returncode'] for row in rows):
        raise SystemExit('One or more isolated research cases need review')


if __name__ == '__main__':
    main()
