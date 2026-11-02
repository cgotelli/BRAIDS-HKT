"""Reproducible, isolated profiling of the completed MF25 bend candidate.

No production solver changes and no Delft3D writes. Every replay starts from
the same preserved checkpoint; only the input end time changes. Instrumentation
adds timers to a mechanically generated BRAIDS source copy, not new equations.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

import numpy as np

from . import run_case as r
from .regularized_mobile import locked_candidate


SOURCE = r.HERE / 'runs/fine_mf25_equilibrium20h_v1_braids'
RUNTIME = r.HERE / 'runs/fine_mf25_3600_v2_runtime'


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'Instrumentation anchor is not unique: {old[:100]}')
    return text.replace(old, new, 1)


def instrument(folder):
    original, identity = locked_candidate(RUNTIME)
    folder.mkdir(parents=True, exist_ok=False)
    source = (RUNTIME / 'cutcell_solver.f90').read_text()
    source = replace_once(source, ' logical :: profile_enabled=.false.', ''' logical :: profile_enabled=.false.
 real(8) :: diag_secondary_s=0.d0,diag_curvature_s=0.d0,diag_filter_s=0.d0
 integer :: diag_filter_calls=0,diag_filter_iterations=0,diag_filter_max_iterations=0''')
    source = replace_once(source,
        '  real(8) :: filter_raw(nc),filter_residual',
        '  real(8) :: filter_raw(nc),filter_residual,diag_begin,diag_phase')
    source = replace_once(source,
        '  call secondary_flow_curvatures(state,secondary_curvature)\n  filter_raw=secondary_curvature',
        '''  diag_begin=omp_get_wtime();diag_phase=diag_begin
  call secondary_flow_curvatures(state,secondary_curvature)
  diag_curvature_s=diag_curvature_s+omp_get_wtime()-diag_phase
  diag_phase=omp_get_wtime()
  filter_raw=secondary_curvature''')
    source = replace_once(source,
        "  if(.not.ok) error stop 'secondary curvature regularization failed'",
        '''  diag_filter_s=diag_filter_s+omp_get_wtime()-diag_phase
  diag_filter_calls=diag_filter_calls+1
  diag_filter_iterations=diag_filter_iterations+filter_iterations
  diag_filter_max_iterations=max(diag_filter_max_iterations,filter_iterations)
  if(.not.ok) error stop 'secondary curvature regularization failed' ''')
    source = replace_once(source,
        '   source_integral=source_integral+secondary_source_work(c)\n  enddo\n end subroutine',
        '''   source_integral=source_integral+secondary_source_work(c)
  enddo
  diag_secondary_s=diag_secondary_s+omp_get_wtime()-diag_begin
 end subroutine''')
    source = replace_once(source, "  write(*,'(a,f12.4)') 'profile: authoritative output wall seconds='", """  write(*,'(a,f12.6)') 'diagnostic: secondary relaxation wall seconds=',diag_secondary_s
  write(*,'(a,f12.6)') 'diagnostic: raw curvature wall seconds=',diag_curvature_s
  write(*,'(a,f12.6)') 'diagnostic: curvature filter wall seconds=',diag_filter_s
  write(*,'(a,i0)') 'diagnostic: curvature filter calls=',diag_filter_calls
  write(*,'(a,i0)') 'diagnostic: curvature filter total iterations=',diag_filter_iterations
  write(*,'(a,i0)') 'diagnostic: curvature filter max iterations=',diag_filter_max_iterations
  write(*,'(a,f12.4)') 'profile: authoritative output wall seconds='""")
    copied = folder / 'cutcell_solver.f90'
    copied.write_text(source, encoding='utf-8')
    command = identity['command'][:]
    exe = folder / 'xbraids_profile.exe'
    command[command.index('-o') + 1] = str(exe)
    command = [str(copied) if Path(x).name == 'cutcell_solver.f90' else x for x in command]
    started = time.perf_counter()
    result = subprocess.run(command, cwd=folder, capture_output=True, text=True, timeout=180)
    (folder / 'compile.log').write_text(result.stdout + result.stderr, encoding='utf-8')
    result.check_returncode()
    r.write_json(folder / 'build.json', dict(
        role='Timers-only diagnostic; no production or Delft3D changes',
        original_executable_sha256=r.sha256(original),
        original_build=identity, command=command,
        instrumented_source_sha256=r.sha256(copied), executable_sha256=r.sha256(exe),
        compile_wall_s=time.perf_counter()-started))
    return exe


def parse_profile(text):
    rows = {}
    for prefix, key, value in re.findall(r'^(profile|diagnostic): (.+?)=\s*([\d.Ee+-]+)', text, re.M):
        rows[prefix + ': ' + key] = float(value)
    steps = re.search(r'steps, rejected trials, CPU seconds:\s*(\d+)\s+(\d+)\s+([\d.Ee+-]+)', text)
    if steps:
        rows.update(steps=int(steps[1]), rejected=int(steps[2]), cpu_s=float(steps[3]))
    return rows


def read_ledger(folder):
    # The native ledger has a dimension header and time/variable-width records.
    return np.fromstring((folder / 'boundary_ledger.txt').read_text(), sep=' ')


def replay(root, label, exe, threads, duration, original_steps, extra_env=None):
    folder = root / label
    folder.mkdir(parents=True, exist_ok=False)
    shutil.copy2(SOURCE / 'case.data.bin', folder / 'case.data.bin')
    # Native fingerprint explicitly excludes end time, not dt or output cadence.
    deck = replace_once((SOURCE / 'case.data').read_text(),
                        '2880 0.012500000000000001 24',
                        f'{2880+duration:.17g} 0.012500000000000001 24')
    (folder / 'case.data').write_text(deck, encoding='utf-8')
    checkpoint = SOURCE / 'checkpoint.bin'
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), OMP_DYNAMIC='FALSE',
               OMP_WAIT_POLICY='PASSIVE', BRAIDS_PROFILE='1', BRAIDS_PROGRESS_STEPS='2000')
    env.update(extra_env or {})
    command = [str(exe), 'case.data', str(checkpoint)]
    start = time.perf_counter()
    with (folder / 'native.log').open('w') as log:
        proc = subprocess.run(command, cwd=folder, env=env, stdout=log,
                              stderr=subprocess.STDOUT, timeout=180)
    wall = time.perf_counter() - start
    proc.check_returncode()
    profile = parse_profile((folder / 'native.log').read_text())
    row = dict(label=label, threads=threads, wall_s=wall, profile=profile,
               replay_steps=profile['steps']-original_steps,
               simulated_seconds=duration, executable_sha256=r.sha256(exe),
               input_checkpoint_sha256=r.sha256(checkpoint),
               case_sha256=r.sha256(folder/'case.data'),
               checkpoint_sha256=r.sha256(folder/'checkpoint.bin'),
               command=command, environment={k: env[k] for k in env if k.startswith(('OMP_', 'BRAIDS_'))})
    r.write_json(folder / 'execution.json', row)
    print(json.dumps(dict(label=label,threads=threads,wall_s=wall,steps=row['replay_steps'])), flush=True)
    return row


def run(tag, duration):
    exe, identity = locked_candidate(RUNTIME)
    root = r.HERE / 'runs' / tag
    report = r.HERE / 'reports' / tag
    root.mkdir(parents=True, exist_ok=False)
    report.mkdir(parents=True, exist_ok=False)
    original_profile = parse_profile((SOURCE / 'native.log').read_text())
    diagnostic_exe = instrument(root / 'runtime')
    rows = []
    for repeat, order in enumerate(((1,4,16,32,8,2), (2,8,32,16,4,1)), 1):
        for threads in order:
            rows.append(replay(root, f'r{repeat}_t{threads}', exe, threads, duration, original_profile['steps']))
            r.write_json(report / 'timings.json', dict(status='running', rows=rows))
    rows.append(replay(root, 'diagnostic_t8', diagnostic_exe, 8, duration, original_profile['steps']))
    finalize(tag)


def finalize(tag):
    """Re-analyze existing receipts without rerunning any simulation."""
    _, identity = locked_candidate(RUNTIME)
    root = r.HERE / 'runs' / tag
    report = r.HERE / 'reports' / tag
    original_profile = parse_profile((SOURCE / 'native.log').read_text())
    labels = [f'r{repeat}_t{threads}' for repeat in (1,2) for threads in (1,2,4,8,16,32)]
    labels.append('diagnostic_t8')
    rows = [json.loads((root / label / 'execution.json').read_text()) for label in labels]
    reference = r.shared.read_frames(r.shared.native_frames_path(root / rows[0]['label']))[-1]['q']
    ledger = read_ledger(root / rows[0]['label'])
    for row in rows:
        q = r.shared.read_frames(r.shared.native_frames_path(root / row['label']))[-1]['q']
        other_ledger = read_ledger(root / row['label'])
        row['maximum_state_difference'] = float(np.max(np.abs(q-reference)))
        row['maximum_boundary_ledger_difference'] = float(np.max(np.abs(other_ledger-ledger)))
        row['checkpoint_bitwise_equal'] = row['checkpoint_sha256'] == rows[0]['checkpoint_sha256']
    receipt = report / 'timings.json'
    previous = json.loads(receipt.read_text()) if receipt.exists() else {}
    completed = dict(status='complete',
        utc=datetime.now(timezone.utc).isoformat(), rows=rows,
        source=str(SOURCE), original_profile=original_profile,
        runtime_identity=identity, duration_s=rows[0]['simulated_seconds'],
        limitations=['Short late-state replay, not a whole-run or large-domain scaling test',
                     'Existing desktop workload and scheduler/thermal variation are not controlled',
                     'No equations, output interval, MF, tolerances or mesh resolution changed'])
    if previous.get('wait_policy_rows'):
        completed['wait_policy_rows'] = previous['wait_policy_rows']
    r.write_json(receipt, completed)
    print('COMPLETE ' + str(report), flush=True)


def wait_policy_test(tag):
    """Process-local OpenMP policy experiments, after the main sweep finishes."""
    exe, _ = locked_candidate(RUNTIME)
    root = r.HERE / 'runs' / tag
    receipt = r.HERE / 'reports' / tag / 'timings.json'
    data = json.loads(receipt.read_text())
    if data['status'] != 'complete' or data.get('wait_policy_rows'):
        raise ValueError('Need a completed sweep without a previous policy test')
    rows = []
    for repeat, order in enumerate(((8,16), (16,8)), 1):
        for threads in order:
            rows.append(replay(root, f'active_r{repeat}_t{threads}', exe, threads,
                               data['duration_s'], data['original_profile']['steps'],
                               dict(OMP_WAIT_POLICY='ACTIVE')))
    reference = r.shared.read_frames(r.shared.native_frames_path(root / data['rows'][0]['label']))[-1]['q']
    ledger = read_ledger(root / data['rows'][0]['label'])
    for row in rows:
        q = r.shared.read_frames(r.shared.native_frames_path(root / row['label']))[-1]['q']
        other = read_ledger(root / row['label'])
        row['maximum_state_difference'] = float(np.max(np.abs(q-reference)))
        row['maximum_boundary_ledger_difference'] = float(np.max(np.abs(other-ledger)))
        row['checkpoint_bitwise_equal'] = row['checkpoint_sha256'] == data['rows'][0]['checkpoint_sha256']
    data['wait_policy_rows'] = rows
    r.write_json(receipt, data)
    print('POLICY TEST COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('tag')
    parser.add_argument('--duration', type=float, default=1.)
    parser.add_argument('--wait-policy-test', action='store_true')
    parser.add_argument('--analyze-existing', action='store_true')
    args = parser.parse_args()
    if not 0 < args.duration <= 10:
        raise ValueError('Diagnostic duration must be in (0,10] seconds')
    if args.analyze_existing:
        finalize(args.tag)
    elif args.wait_policy_test:
        wait_policy_test(args.tag)
    else:
        run(args.tag, args.duration)
