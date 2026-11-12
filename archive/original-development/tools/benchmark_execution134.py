"""Sequential actual public-launcher timings, including prepare and process cost."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, default=ROOT/'validation/_performance134/matched')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', nargs='+', required=True)
    parser.add_argument('--repeats', type=int, default=2)
    args = parser.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    runtime = args.runtime.resolve(); executable = Path(sys.executable).resolve()
    env = dict(os.environ, PYTHONPATH=str(runtime), PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
        PATH=os.pathsep.join([str(executable.parent/'Library/bin'), str(executable.parent),
            os.environ.get('SystemRoot', 'C:/Windows')+'/System32']))
    for key in ('PYTHONHOME','QGIS_PREFIX_PATH','QT_PLUGIN_PATH','BRAIDS_TEST_TOOLCHAIN'):
        env.pop(key, None)
    rows = []; prepared = {}
    def call(name, command):
        began = time.perf_counter()
        with (out/(name+'.log')).open('w') as log:
            done = subprocess.run([str(executable), str(runtime/'braids_core/river_launcher.py'), *map(str,command)],
                cwd=out, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=600)
        elapsed = time.perf_counter()-began
        if done.returncode: raise RuntimeError('Failed '+name+'; see log')
        return elapsed
    for repeat in range(args.repeats):
        # Reverse repeat order to reduce monotonic drift; each process is fresh.
        for case in (args.cases if repeat%2==0 else list(reversed(args.cases))):
            source = (args.inputs/(case+'.json')).resolve()
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            if case not in prepared:
                path = out/(case+'_prepared')
                preparation = call(case+'_prepare', ['prepare','--project',source,'--output',path])
                prepared[case] = (path, preparation)
            path, preparation = prepared[case]
            target = out/f'{case}_r{repeat}'
            print('START',target.name,flush=True)
            elapsed = call(target.name, ['run','--input',path,'--output',target])
            status = json.loads((target/'STATUS.json').read_text())
            if status['status'] != 'complete': raise AssertionError(status)
            if hashlib.sha256(source.read_bytes()).hexdigest() != digest: raise AssertionError('Input changed')
            frames = json.loads((target/'frames/INDEX.json').read_text())['frames']
            row = dict(case=case, repeat=repeat, source=str(source), source_sha256=digest,
                preparation_process_s=preparation, run_process_s=elapsed,
                core_wall_s=status['wall_s'], process_outside_core_s=elapsed-status['wall_s'],
                status=status, frames=len(frames), run=str(target))
            rows.append(row)
            (out/'BENCHMARK.json').write_text(json.dumps(dict(rows=rows,
                runtime=str(runtime), runtime_manifest_sha256=hashlib.sha256((runtime/'RUNTIME.json').read_bytes()).hexdigest(),
                scope='Sequential fresh compiler-free actual launcher processes; preparation measured once per input, '
                      'reused for subsequent fresh run processes. Core timer excludes startup/public preflight and final wrapping.'),indent=2)+'\n')
            print('PASS',case,round(elapsed,4),'process s',round(status['wall_s'],4),'core s',flush=True)


if __name__ == '__main__': main()
