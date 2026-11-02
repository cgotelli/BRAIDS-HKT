"""Write-once pre-GPU source/input/gate lock for composed Gate A."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

from . import amr_composed_native_fixture_v2 as case
from . import amr_composed_one_step_native as one
from . import amr_hydro_native_fixture as base


HERE = Path(__file__).resolve().parent
SOURCE = case.ROOT / 'gpu_composed_source_v1'
LOCK = case.ROOT / 'gpu_composed_source_lock_v1.json'
GATE = HERE / 'reports/amr_composed_20261004_v1/GATE.md'
PYTHON_MODULES = (
    'amr_composed_one_step_gpu.py',
    'amr_composed_one_step_provenance.py',
    'coupled_reference.py', 'coupled_solver.py', 'coupled_time.py',
    'secondary_transport.py', 'secondary_curvature.py',
    'sediment_transport.py', 'sediment_stratigraphy.py',
    'exner_update.py', 'suspended_faces.py',
    'suspended_advection.py', 'suspended_exchange.py',
    'exchange_ledger_parallel.py', 'boundaries.py',
    'reconstruction.py', 'hydraulic_pipeline.py',
    'resident_update.py', 'time_integration.py')


def _files():
    native = json.loads((case.ROOT / 'plan.json').read_text(encoding='utf-8'))
    precursor = json.loads((one.ROOT / 'plan.json').read_text(encoding='utf-8'))
    paths = [HERE / name for name in PYTHON_MODULES]
    paths += [GATE, case.ROOT / 'plan.json', case.ROOT / 'cpu/frames.txt',
              case.ROOT / 'cpu/checkpoint.bin',
              case.ROOT / 'amr_full_state_export_v1/initial/coupled_input.bin',
              case.ROOT / 'amr_full_state_export_v2/RECEIPT.json',
              one.ROOT / 'plan.json', one.ROOT / 'cpu/frames.txt',
              one.ROOT / 'cpu/checkpoint.bin', one.ROOT / 'export_receipt.json']
    paths += [Path(name) for name in native['inputs']]
    paths += [Path(name) for name in precursor['inputs']]
    paths += [Path(native['runtime'])]
    for folder in (one.ROOT / 'mesh_0', one.ROOT / 'amr_final'):
        paths += [path for path in folder.iterdir() if path.is_file()]
    paths += list(SOURCE.glob('*.cu'))
    return tuple(sorted({path.resolve() for path in paths}))


def freeze():
    if LOCK.exists() or SOURCE.exists():
        raise FileExistsError('Composed GPU Gate A source lock is write-once')
    native = json.loads((case.ROOT / 'plan.json').read_text(encoding='utf-8'))
    precursor = json.loads((one.ROOT / 'plan.json').read_text(encoding='utf-8'))
    for plan in (native, precursor):
        for name, digest in {**plan['source_sha256'], **plan['inputs'],
                             plan['runtime']:plan['runtime_sha256']}.items():
            if base.sha(name) != digest:
                raise ValueError('Frozen composed source/input changed: '+name)
    SOURCE.mkdir(parents=True)
    for source in HERE.glob('*.cu'):
        shutil.copy2(source, SOURCE / source.name)
    payload = dict(schema=1, numerical_branch='CUDA augmented f-wave SSPRK33, MPM/Exner, suspended FCT+exchange, signed hI on native AMR initial WENO map',
        native_full_plan_sha256=base.sha(case.ROOT / 'plan.json'),
        native_one_step_plan_sha256=base.sha(one.ROOT / 'plan.json'),
        declared_gate_sha256=base.sha(GATE),
        files={str(path):base.sha(path) for path in _files()},
        scope='Gate A first synchronous wet/dry step, not Gate B subcycled AMR/regrid',
        Delft3D_modified=False)
    base.save(LOCK, payload)
    return dict(lock=str(LOCK), sha256=base.sha(LOCK),
        file_count=len(payload['files']), cuda_files=len(list(SOURCE.glob('*.cu'))))


def preflight():
    if not LOCK.exists():
        raise FileNotFoundError('Composed GPU Gate A source not frozen')
    payload = json.loads(LOCK.read_text(encoding='utf-8'))
    if payload['declared_gate_sha256'] != base.sha(GATE):
        raise ValueError('Original pre-GPU Gate A contract changed')
    paths = _files()
    if {str(path) for path in paths} != set(payload['files']):
        raise ValueError('Composed GPU source/input file set changed')
    for name, digest in payload['files'].items():
        if base.sha(name) != digest:
            raise ValueError('Composed GPU frozen file changed: '+name)
    return dict(lock=str(LOCK), sha256=base.sha(LOCK),
        file_count=len(payload['files']), verified=True,
        Delft3D_modified=False)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'preflight'))
    args = parser.parse_args()
    print(json.dumps(globals()[args.action](), indent=2))
