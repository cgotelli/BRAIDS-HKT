"""Write-once source/input lock for the bounded active-MPM GPU AMR gate."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

from . import amr_hydro_native_fixture as base
from . import amr_mpm_native_fixture_v2 as case
from . import amr_mpm_export as exported


HERE = Path(__file__).resolve().parent
SOURCE = case.ROOT / 'gpu_source_v1'
LOCK = case.ROOT / 'gpu_source_lock_v1.json'
GATE = HERE / 'reports/amr_mpm_subcycled_candidate_20261004_v1/GATE_V2.md'
ORIGINAL_GATE = HERE / 'reports/amr_mpm_subcycled_candidate_20261004_v1/GATE.md'

PYTHON_NAMES = (
    'amr_hydro_native_fixture.py', 'amr_mpm_native_fixture.py',
    'amr_mpm_native_fixture_v2.py', 'amr_mpm_export.py',
    'amr_mpm_stage_oracle.py',
    'amr_mpm_twolevel_gpu.py', 'amr_mpm_provenance.py',
    'amr_mpm_restart.py', 'amr_mpm_transaction_test.py',
    'coupled_reference.py', 'coupled_solver.py', 'coupled_time.py',
    'time_integration.py', 'reconstruction.py', 'hydraulic_pipeline.py',
    'boundaries.py', 'resident_update.py', 'sediment_transport.py',
    'sediment_stratigraphy.py', 'secondary_transport.py',
    'secondary_curvature.py', 'exner_update.py', 'amr_reflux.py',
    'amr_hydro_route.py', 'suspended_faces.py', 'suspended_advection.py',
    'suspended_exchange.py', 'exchange_ledger_parallel.py')

CUDA_NAMES = (
    'augmented_fwave.cu', 'resident_update.cu', 'reconstruction.cu',
    'boundaries.cu', 'sediment_transport.cu', 'sediment_stratigraphy.cu',
    'secondary_transport.cu', 'secondary_curvature.cu', 'exner_update.cu',
    'coupled_time.cu', 'time_integration.cu', 'amr_reflux.cu')


def _reference_files():
    plan = json.loads((case.ROOT / 'plan.json').read_text(encoding='utf-8'))
    paths = [HERE / name for name in PYTHON_NAMES]
    paths.append(HERE.parents[1] / 'tests/test_gpu_amr_mpm_adapter_contract.py')
    paths += [GATE, ORIGINAL_GATE, case.ROOT / 'plan.json', case.ROOT / 'cpu/frames.txt',
              case.ROOT / 'cpu/native.log', case.ROOT / 'cpu/execution.json',
              exported.ROOT / 'build.json', exported.ROOT / 'receipt.json',
              exported.ROOT / 'runtime/xbraids_coupled_export.exe']
    paths += [Path(name) for name in plan['inputs']]
    paths.append(Path(plan['runtime']))
    paths += [exported.ROOT / 'mesh_0' / name for name in (
        'case.data', 'case.data.bin', 'case.data.json', 'geometry.npz',
        'leaf_identity.npz', 'coupled_input.bin', 'export.log')]
    return tuple(sorted({path.resolve() for path in paths}))


def freeze():
    if LOCK.exists() or SOURCE.exists():
        raise FileExistsError('AMR MPM GPU source lock is write-once')
    plan = json.loads((case.ROOT / 'plan.json').read_text(encoding='utf-8'))
    for name, digest in {**plan['source_sha256'], **plan['inputs'],
                         plan['runtime']: plan['runtime_sha256']}.items():
        if base.sha(name) != digest:
            raise ValueError('Frozen native MPM source/input changed: ' + name)
    receipt = json.loads((exported.ROOT / 'receipt.json').read_text(encoding='utf-8'))
    if receipt['native_plan_sha256'] != base.sha(case.ROOT / 'plan.json') or \
            receipt['native_frames_sha256'] != base.sha(case.ROOT / 'cpu/frames.txt'):
        raise ValueError('Selected-leaf material export disagrees with native case')
    SOURCE.mkdir(parents=True)
    for name in CUDA_NAMES:
        shutil.copy2(HERE / name, SOURCE / name)
    for name in PYTHON_NAMES:
        shutil.copy2(HERE / name, SOURCE / name)
    files = {str(path): base.sha(path) for path in _reference_files()}
    files.update({str((SOURCE / name).resolve()): base.sha(SOURCE / name)
                  for name in CUDA_NAMES + PYTHON_NAMES})
    payload = dict(schema=1, native_reference='unchanged CPU21b7',
        declared_gate_sha256=base.sha(GATE), files=files,
        cuda_source_names=list(CUDA_NAMES),
        numerical_GPU_branch='Augmented f-wave + graded MPM/Exner SSPRK33 level subcycling and reflux',
        unsupported='No regrid, wet/dry, bank, suspension, plugin or physical validation',
        Delft3D_modified=False)
    base.save(LOCK, payload)
    return dict(lock=str(LOCK), files=len(files), sha256=base.sha(LOCK))


def preflight():
    if not LOCK.exists():
        raise FileNotFoundError('AMR MPM GPU source lock not frozen')
    payload = json.loads(LOCK.read_text(encoding='utf-8'))
    actual_paths = {str(path) for path in _reference_files()}
    actual_paths.update(str((SOURCE / name).resolve())
                        for name in CUDA_NAMES + PYTHON_NAMES)
    if actual_paths != set(payload['files']):
        raise ValueError('AMR MPM frozen source/input file set changed')
    for path, digest in payload['files'].items():
        if base.sha(path) != digest:
            raise ValueError('AMR MPM frozen source/input changed: ' + path)
    plan = json.loads((case.ROOT / 'plan.json').read_text(encoding='utf-8'))
    for name, digest in {**plan['source_sha256'], **plan['inputs'],
                         plan['runtime']: plan['runtime_sha256']}.items():
        if base.sha(name) != digest:
            raise ValueError('Native MPM plan source/input changed: ' + name)
    return dict(lock=str(LOCK), sha256=base.sha(LOCK),
        file_count=len(payload['files']), verified=True,
        Delft3D_modified=False)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'preflight'))
    args = parser.parse_args()
    print(json.dumps(globals()[args.action](), indent=2))
