"""Write-once source/input lock for post-regrid composed Gate B3."""
from __future__ import annotations

import json
from pathlib import Path

from . import amr_composed_native_fixture_v2 as case
from . import amr_composed_b2_provenance as b2
from . import amr_hydro_native_fixture as base


HERE = Path(__file__).resolve().parent
LOCK = case.ROOT/'gpu_composed_source_lock_b3_v1.json'
GATE = HERE/'reports/amr_composed_20261004_v1/GATE_B3.md'


def _files():
    plan = json.loads((case.ROOT/'plan.json').read_text(encoding='utf-8'))
    paths = [HERE/'amr_composed_b3_gpu.py',
             HERE/'amr_composed_b3_provenance.py', GATE, b2.LOCK,
             case.ROOT/'gpu_composed_b2_candidate_v1/READBACK.json',
             case.ROOT/'gpu_composed_b2_transaction_v1/READBACK.json',
             case.ROOT/'plan.json', case.ROOT/'cpu/frames.txt',
             case.ROOT/'cpu/checkpoint.bin',
             case.ROOT/'amr_full_state_export_v2/final/coupled_input.bin']
    paths += [Path(name) for name in plan['inputs']]
    paths += [Path(name) for name in plan['source_sha256']]
    paths += [Path(plan['runtime'])]
    paths += list(b2.b1.gate_a.SOURCE.glob('*.cu'))
    return tuple(sorted({path.resolve() for path in paths}))


def freeze():
    if LOCK.exists():
        raise FileExistsError('B3 source lock is write-once')
    b2.preflight()
    for name in ('gpu_composed_b2_candidate_v1',
                 'gpu_composed_b2_transaction_v1'):
        record = json.loads((case.ROOT/name/'READBACK.json').read_text(
            encoding='utf-8'))
        if not (record.get('gate_B2_numerical_passed') or
                record.get('frozen_gate_B2_transaction_passed')):
            raise ValueError('Prior B2 gate did not pass: '+name)
    payload = dict(schema=1, scope='Private composed post-regrid 0.06-to-0.08 s GPU fixture',
        gate_sha256=base.sha(GATE), B2_source_lock_sha256=base.sha(b2.LOCK),
        files={str(path):base.sha(path) for path in _files()},
        no_CPU_numerical_fallback=True, Delft3D_modified=False)
    base.save(LOCK, payload)
    return dict(lock=str(LOCK), sha256=base.sha(LOCK),
                files=len(payload['files']))


def preflight():
    b2.preflight()
    payload = json.loads(LOCK.read_text(encoding='utf-8'))
    if (payload['gate_sha256'] != base.sha(GATE) or
            payload['B2_source_lock_sha256'] != base.sha(b2.LOCK) or
            set(payload['files']) != {str(path) for path in _files()}):
        raise ValueError('B3 frozen gate/source set changed')
    for name, digest in payload['files'].items():
        if base.sha(name) != digest:
            raise ValueError('B3 frozen source/input changed: '+name)
    return dict(lock=str(LOCK), sha256=base.sha(LOCK),
                files=len(payload['files']), verified=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze','preflight'))
    args = parser.parse_args()
    print(json.dumps(globals()[args.action](), indent=2))
