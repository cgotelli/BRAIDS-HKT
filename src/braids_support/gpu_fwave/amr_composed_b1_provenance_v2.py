"""Write-once pre-GPU source/input lock for the bounded Gate B1 experiment."""
from __future__ import annotations

import json
from pathlib import Path

from . import amr_composed_native_fixture_v2 as case
from . import amr_composed_b1_native as b1
from . import amr_composed_one_step_provenance as gate_a
from . import amr_hydro_native_fixture as base
from . import amr_mpm_wetdry_export as identity


HERE = Path(__file__).resolve().parent
LOCK = case.ROOT / 'gpu_composed_source_lock_b1_v2.json'
GATE = HERE / 'reports/amr_composed_20261004_v1/GATE_B1.md'


def _files():
    plan = json.loads((b1.ROOT / 'plan.json').read_text(encoding='utf-8'))
    paths = [HERE / name for name in
        ('amr_composed_b1_gpu_v2.py', 'amr_composed_b1_provenance_v2.py',
         'amr_reflux.py', 'amr_reflux.cu', 'amr_hydro_route.py',
         'amr_composed_one_step_gpu.py')]
    paths += [GATE, gate_a.LOCK, b1.ROOT / 'plan.json',
              b1.ROOT / 'cpu/frames.txt', b1.ROOT / 'cpu/checkpoint.bin',
              b1.ROOT / 'amr_final/coupled_input.bin',
              identity.ROOT / 'mesh_0/leaf_identity.npz']
    paths += [Path(name) for name in plan['inputs']]
    paths += [Path(name) for name in plan['source_sha256']]
    paths += [Path(plan['runtime'])]
    paths += list(gate_a.SOURCE.glob('*.cu'))
    return tuple(sorted({path.resolve() for path in paths}))


def freeze():
    if LOCK.exists():
        raise FileExistsError('B1 source lock is write-once')
    gate_a.preflight()
    plan = json.loads((b1.ROOT / 'plan.json').read_text(encoding='utf-8'))
    for name, digest in {**plan['source_sha256'], **plan['inputs'],
                         plan['runtime']: plan['runtime_sha256']}.items():
        if base.sha(name) != digest:
            raise ValueError('B1 native source/input changed: '+name)
    payload = dict(schema=1, scope='Private 10-row AMR composed mixed-level prefix through 0.02 s only',
        gate_sha256=base.sha(GATE), gate_A_lock_sha256=base.sha(gate_a.LOCK),
        files={str(path): base.sha(path) for path in _files()},
        no_CPU_numerical_fallback=True, Delft3D_modified=False)
    base.save(LOCK, payload)
    return dict(lock=str(LOCK), sha256=base.sha(LOCK),
                files=len(payload['files']))


def preflight():
    gate_a.preflight()
    payload = json.loads(LOCK.read_text(encoding='utf-8'))
    if (payload['gate_sha256'] != base.sha(GATE) or
            payload['gate_A_lock_sha256'] != base.sha(gate_a.LOCK) or
            set(payload['files']) != {str(path) for path in _files()}):
        raise ValueError('B1 frozen gate/source set changed')
    for name, digest in payload['files'].items():
        if base.sha(name) != digest:
            raise ValueError('B1 frozen source/input changed: '+name)
    return dict(lock=str(LOCK), sha256=base.sha(LOCK),
                files=len(payload['files']), verified=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'preflight'))
    args = parser.parse_args()
    print(json.dumps(globals()[args.action](), indent=2))
