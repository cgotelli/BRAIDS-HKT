"""Distinct write-once V2 lock for native-AMR WENO GPU AMR candidate."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

from . import amr_hydro_native_fixture as base
from . import amr_mpm_native_fixture_v2 as case
from . import amr_mpm_provenance as prior


HERE = Path(__file__).resolve().parent
SOURCE = case.ROOT / 'gpu_source_v2'
LOCK = case.ROOT / 'gpu_source_lock_v2.json'
DIAGNOSTIC = case.ROOT / 'native_path_probe_v2'
MODULES = ('amr_mpm_twolevel_gpu_v2.py', 'amr_mpm_provenance_v2.py')


def _files():
    runtime = DIAGNOSTIC / 'runtime'
    sources = [path for path in runtime.iterdir()
               if path.suffix.lower() in ('.f', '.f90')]
    if len(sources) < 15:
        raise ValueError('Incomplete private copied-source AMR geometry export')
    paths = [HERE / name for name in MODULES]
    paths += [HERE / 'amr_mpm_native_path_probe.py', prior.LOCK,
              prior.GATE, DIAGNOSTIC / 'READBACK.json',
              DIAGNOSTIC / 'amr/coupled_input.bin',
              DIAGNOSTIC / 'amr/coupled_stage_pre.bin',
              DIAGNOSTIC / 'amr/coupled_stage_post.bin',
              runtime / 'xbraids_amr_stage_probe.exe',
              *sources]
    paths += [SOURCE / name for name in MODULES]
    return tuple(sorted({path.resolve() for path in paths}))


def freeze():
    prior_receipt = prior.preflight()
    if SOURCE.exists() or LOCK.exists():
        raise FileExistsError('AMR MPM GPU V2 source lock is write-once')
    report = json.loads((DIAGNOSTIC / 'READBACK.json').read_text(encoding='utf-8'))
    if (report['initial_state_max_abs'] != 0. or
            report['geometry']['weno_cells']['exactly_equal'] or
            report['geometry']['owner']['max_abs'] != 0. or
            report['geometry']['length']['max_abs'] != 0. or
            not report['diagnostic_export_guard_only_change']):
        raise ValueError('Native AMR WENO diagnosis has changed')
    SOURCE.mkdir(parents=True)
    for name in MODULES:
        shutil.copy2(HERE / name, SOURCE / name)
    payload = dict(schema=1, parent_lock_sha256=prior_receipt['sha256'],
        gate_sha256=base.sha(prior.GATE),
        files={str(path): base.sha(path) for path in _files()},
        numerical_change='Replace fixed-export WENO face map with native AMR level-stride map',
        unchanged='Augmented f-wave, MPM, Exner, SSPRK33, reflux, source limits and GATE_V2',
        CPU_role='Initial immutable stencil geometry only; no per-stage CPU numerical fallback',
        Delft3D_modified=False)
    base.save(LOCK, payload)
    return dict(lock=str(LOCK), sha256=base.sha(LOCK),
        file_count=len(payload['files']), parent_lock_sha256=prior_receipt['sha256'])


def preflight():
    prior_receipt = prior.preflight()
    if not LOCK.exists():
        raise FileNotFoundError('AMR MPM GPU V2 source lock not frozen')
    payload = json.loads(LOCK.read_text(encoding='utf-8'))
    if payload['parent_lock_sha256'] != prior_receipt['sha256'] or \
            payload['gate_sha256'] != base.sha(prior.GATE):
        raise ValueError('AMR MPM V2 prior source/gate changed')
    files = _files()
    if {str(path) for path in files} != set(payload['files']):
        raise ValueError('AMR MPM V2 source/input file set changed')
    for path, digest in payload['files'].items():
        if base.sha(path) != digest:
            raise ValueError('AMR MPM V2 frozen source/input changed: ' + path)
    return dict(lock=str(LOCK), sha256=base.sha(LOCK),
        file_count=len(payload['files']), parent_lock_sha256=prior_receipt['sha256'],
        verified=True, Delft3D_modified=False)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'preflight'))
    args = parser.parse_args()
    print(json.dumps(globals()[args.action](), indent=2))
