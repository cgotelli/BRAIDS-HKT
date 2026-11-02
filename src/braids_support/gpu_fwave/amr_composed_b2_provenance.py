"""Write-once pre-GPU source/input lock for composed AMR coarsening Gate B2."""
from __future__ import annotations

import json
from pathlib import Path

from . import amr_composed_native_fixture_v2 as case
from . import amr_composed_b2_native as native
from . import amr_composed_b2_export as export
from . import amr_composed_b1_provenance_v2 as b1
from . import amr_hydro_native_fixture as base


HERE = Path(__file__).resolve().parent
LOCK = case.ROOT/'gpu_composed_source_lock_b2_v1.json'
GATE = HERE/'reports/amr_composed_20261004_v1/GATE_B2.md'


def _files():
    plan = json.loads((native.ROOT/'plan.json').read_text(encoding='utf-8'))
    paths = [HERE/name for name in
        ('amr_composed_b2_gpu.py', 'amr_composed_b2_provenance.py',
         'amr_composed_b2_native.py', 'amr_composed_b2_export.py',
         'amr_transfer.py')]
    paths += [GATE, b1.LOCK, native.ROOT/'plan.json',
              native.ROOT/'cpu/frames.txt', native.ROOT/'cpu/checkpoint.bin',
              native.ROOT/'amr_final/coupled_input.bin',
              export.ROOT/'RECEIPT.json', export.ROOT/'coupled_input.bin',
              case.ROOT/'cpu/frames.txt',
              case.ROOT/'amr_full_state_export_v2/final/coupled_input.bin']
    paths += [Path(name) for name in plan['inputs']]
    paths += [Path(name) for name in plan['source_sha256']]
    paths += [Path(plan['runtime'])]
    paths += list(b1.gate_a.SOURCE.glob('*.cu'))
    return tuple(sorted({path.resolve() for path in paths}))


def freeze():
    if LOCK.exists():
        raise FileExistsError('B2 source lock is write-once')
    b1.preflight()
    plan = json.loads((native.ROOT/'plan.json').read_text(encoding='utf-8'))
    for name, digest in {**plan['source_sha256'], **plan['inputs'],
                         plan['runtime']:plan['runtime_sha256']}.items():
        if base.sha(name) != digest:
            raise ValueError('B2 native source/input changed: '+name)
    payload = dict(schema=1, scope='Private composed 10-row GPU AMR through 0.06 s coarsening',
        gate_sha256=base.sha(GATE), B1_source_lock_sha256=base.sha(b1.LOCK),
        files={str(path):base.sha(path) for path in _files()},
        no_CPU_numerical_fallback=True, Delft3D_modified=False)
    base.save(LOCK, payload)
    return dict(lock=str(LOCK), sha256=base.sha(LOCK),
                files=len(payload['files']))


def preflight():
    b1.preflight()
    payload = json.loads(LOCK.read_text(encoding='utf-8'))
    if (payload['gate_sha256'] != base.sha(GATE) or
            payload['B1_source_lock_sha256'] != base.sha(b1.LOCK) or
            set(payload['files']) != {str(path) for path in _files()}):
        raise ValueError('B2 frozen gate/source set changed')
    for name, digest in payload['files'].items():
        if base.sha(name) != digest:
            raise ValueError('B2 frozen source/input changed: '+name)
    return dict(lock=str(LOCK), sha256=base.sha(LOCK),
                files=len(payload['files']), verified=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze','preflight'))
    args = parser.parse_args()
    print(json.dumps(globals()[args.action](), indent=2))
