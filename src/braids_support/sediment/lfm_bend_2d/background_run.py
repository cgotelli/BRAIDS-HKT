"""One finite, unattended warm-up/mobile pipeline; no periodic monitoring."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
import re
import sys

from . import analyze_runs as a
from . import run_case as r


def hydraulic_gates(metrics):
    """Use the existing first-pair operational thresholds before seeding."""
    return dict(
        steady_stage=metrics['final_stage_change_over_output_interval_m'] < 1e-5,
        inlet_Q=metrics['inlet_discharge_relative_error'] < .005,
        outlet_Q=metrics['outlet_discharge_relative_error'] < .005,
        water_balance=abs(metrics['water_budget_residual_m3']) < 1e-6,
        positive_depth=metrics['min_depth_m'] > 0,
    )


def run(tag, engine, *, dx=.10625, end=3600., morfac=1., threads=8,
        ranks=4, timeout_s=86400):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', tag):
        raise ValueError('Use a simple unique run tag')
    if engine not in ('braids', 'delft') or min(dx, end, morfac, timeout_s) <= 0:
        raise ValueError('Invalid engine or nonpositive run setting')
    report = r.HERE/'reports'/tag/engine
    hydro = r.HERE/'runs'/f'{tag}_hydraulic_{engine}'
    mobile = r.HERE/'runs'/f'{tag}_mobile_{engine}'
    if hydro.exists() or mobile.exists():
        raise FileExistsError('Refusing to overwrite a previous run')
    report.mkdir(parents=True, exist_ok=False)
    dt = .025 * min(1., dx/.2125)
    phases = [
        ('hydraulic', hydro, dict(end=300., output=30., morfac=1., hydraulic=True, seed=None)),
        ('mobile', mobile, dict(end=end/morfac, output=min(600., end)/morfac,
                              morfac=morfac, hydraulic=False, seed=hydro)),
    ]
    r.write_json(report/'plan.json', dict(
        engine=engine, dx_m=dx, dt_max_s=dt, end_morphological_s=end,
        morfac=morfac, threads=threads if engine=='braids' else None,
        ranks=ranks if engine=='delft' else None, timeout_per_phase_s=timeout_s,
        hydraulic_folder=str(hydro), mobile_folder=str(mobile),
        runner_sha256=r.sha256(r.Path(__file__)), engine_source_modified=False,
        note='Fine-grid unaccelerated transient, not equilibrium or an isolated grid-convergence test. '
             'Resolution, timestep and acceleration differ from the coarse factor-5 run. '
             'Hydraulic warm-up is checked once on completion before mobile-bed initialization. '
             'Secondary intensity is reinitialized in both engines, as in the previous pair.'))

    def status(state, phase, folder, **extra):
        row=dict(status=state, phase=phase, folder=str(folder), pid=os.getpid(),
                 updated_utc=datetime.now(timezone.utc).isoformat(), **extra)
        r.write_json(report/'status.json', row)
        print(row, flush=True)

    phase, folder = 'startup', hydro
    try:
        for phase, folder, options in phases:
            status('preparing', phase, folder)
            prepare = r.prepare_native if engine=='braids' else r.prepare_delft
            prepare(folder, dx=dx, dt=dt, **options)
            status('running', phase, folder)
            if engine=='braids':
                execution = r.execute_native(folder, threads, timeout_s=timeout_s)
            elif ranks==1:
                execution = r.shared.delft_runtime.run_case(folder, timeout_s=timeout_s)
            else:
                sys.path.insert(0, str(r.ROOT/'braids_support/sediment/synthetic_expansion'))
                import screen_delft_parallel as mpi
                execution = mpi.execute(folder, ranks, timeout=timeout_s)
            # One end-of-phase check, not a progress-monitoring loop.
            metrics = a.audit(a.load(folder))
            r.write_json(report/f'{phase}_metrics.json', metrics)
            if phase=='hydraulic':
                gates = hydraulic_gates(metrics)
                r.write_json(report/'hydraulic_gates.json', gates)
                if not all(gates.values()):
                    raise RuntimeError(f'Warm-up failed operational gates: {gates}')
            status('phase_complete', phase, folder, execution=execution)
        status('complete', phase, folder, experimental_validation_passed=False)
    except Exception as exc:
        status('failed', phase, folder, error=f'{type(exc).__name__}: {exc}')
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('tag')
    p.add_argument('engine', choices=['braids', 'delft'])
    p.add_argument('--dx', type=float, default=.10625)
    p.add_argument('--end', type=float, default=3600., help='Morphological seconds')
    p.add_argument('--morfac', type=float, default=1.)
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--ranks', type=int, default=4)
    p.add_argument('--timeout-s', type=float, default=86400.)
    run(**vars(p.parse_args()))
