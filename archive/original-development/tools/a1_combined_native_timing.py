"""Three write-once unchanged-native A1 timing repetitions per frozen variant.

This records an independent CPU baseline for the predeclared combined-boundary
GPU full gate. It never edits an input, source file, or prior run directory.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import time

import numpy as np

from braids_qgis.core.results import read_frames
from validation.gpu_fwave import a1_combined_full_gate_preflight as gate
from validation.gpu_fwave import a1_combined_live_scheduled_native_case as native
from validation.gpu_fwave import coupled_reference


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / 'validation/gpu_fwave/runs/a1_combined_native_timing_20261004_v1'
COUNTS = re.compile(
    r'steps, rejected trials, CPU seconds:\s*(\d+)\s+(\d+)\s+([0-9.eE+-]+)'
)


def run(output: Path) -> dict:
    output = output.resolve()
    if output.exists():
        raise FileExistsError('Native timing output is write-once: ' + str(output))
    references = {name: gate.preflight(name) for name in native.VARIANTS}
    plan, plan_sha = native.load_plan()
    executable = Path(plan['native_executable']).resolve()
    if native.sha(executable) != gate.EXE_SHA or plan_sha != gate.PLAN_SHA:
        raise ValueError('Frozen native executable or plan differs')
    env = {key: value for key, value in os.environ.items()
           if not key.startswith('BRAIDS_')}
    env.update(native.ENV)
    if env['OMP_NUM_THREADS'] != '8' or env['BRAIDS_TIME_INTEGRATOR'] != 'ssp43':
        raise ValueError('Native timing environment differs from frozen gate')
    output.mkdir(parents=True)
    result = dict(schema=1, kind='unchanged_native_A1_three_repeat_timing',
                  full_gate_sha256=native.sha(gate.GATE),
                  native_plan_sha256=plan_sha,
                  native_executable_sha256=native.sha(executable),
                  timing_script_sha256=native.sha(__file__),
                  environment={key: env[key] for key in native.ENV},
                  cpu_model=platform.processor(),
                  process_pid=os.getpid(),
                  gpu_solver_executed=False,
                  Delft3D_source_modified=False,
                  variants={})
    for variant in native.VARIANTS:
        row = next(item for item in plan['cases'] if item['variant'] == variant)
        deck = Path(row['input']).resolve()
        baseline = references[variant][1]
        baseline_q = np.stack([frame['q'] for frame in baseline])
        baseline_ledger = np.stack([frame['ledger'] for frame in baseline])
        runs = []
        for repeat in range(1, 4):
            folder = output / variant / f'repeat_{repeat}'
            setup_started = time.perf_counter()
            origin = coupled_reference.copy_inputs(deck, folder)
            setup_s = time.perf_counter() - setup_started
            started = time.perf_counter()
            with (folder / 'native.log').open('x', encoding='utf-8') as log:
                with subprocess.Popen(
                    [str(executable), deck.name], cwd=folder, env=env,
                    stdout=log, stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
                ) as process:
                    try:
                        return_code = process.wait(timeout=240)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                        raise
                    native_pid = process.pid
            process_s = time.perf_counter() - started
            if return_code != 0:
                raise RuntimeError(f'Native {variant} repeat {repeat} failed: {return_code}')
            frames_path = folder / 'frames.txt'
            frames = read_frames(frames_path)
            if len(frames) != 9:
                raise ValueError('Native timing repetition has missing frames')
            q = np.stack([frame['q'] for frame in frames])
            ledger = np.stack([frame['ledger'] for frame in frames])
            log_text = (folder / 'native.log').read_text(encoding='utf-8')
            match = COUNTS.search(log_text)
            if match is None or (int(match.group(1)), int(match.group(2))) != (104, 0):
                raise ValueError('Native timing repetition step/rejection count differs')
            runs.append(dict(repeat=repeat, pid=native_pid, setup_s=setup_s,
                             process_wall_s=process_s, native_cpu_seconds=float(match.group(3)),
                             accepted_steps=104, rejected_trials=0,
                             origin=origin, frames_sha256=native.sha(frames_path),
                             log_sha256=native.sha(folder / 'native.log'),
                             maximum_q_difference=float(np.max(np.abs(q-baseline_q))),
                             maximum_ledger_difference=float(np.max(np.abs(ledger-baseline_ledger)))))
        walls = [row['process_wall_s'] for row in runs]
        result['variants'][variant] = dict(
            runs=runs, median_process_wall_s=statistics.median(walls),
            process_wall_range_s=[min(walls), max(walls)],
            native_baseline_frames_sha256=gate.PINNED[variant]['frames'],
            all_frames_bitwise=all(row['frames_sha256'] == gate.PINNED[variant]['frames']
                                   for row in runs),
        )
    native.save(output / 'TIMING.json', result)
    return dict(output=str(output), variants={name: dict(
        median_process_wall_s=value['median_process_wall_s'],
        all_frames_bitwise=value['all_frames_bitwise'])
        for name, value in result['variants'].items()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
