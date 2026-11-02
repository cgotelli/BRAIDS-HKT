"""Write-once source lock for the all-five-frame Gate-B audit."""
from __future__ import annotations

import json
from pathlib import Path

from . import amr_composed_native_fixture_v2 as case
from . import amr_composed_b3_provenance as b3
from . import amr_hydro_native_fixture as base


HERE = Path(__file__).resolve().parent
LOCK = case.ROOT/'gpu_composed_b3_frames_source_lock_v1.json'
PATHS = (HERE/'amr_composed_b3_frames.py',
         HERE/'amr_composed_b3_frames_provenance.py', b3.LOCK,
         case.ROOT/'cpu/frames.txt',
         case.ROOT/'cpu/secondary_flow_budget.txt',
         case.ROOT/'mixed_composed_reference_b1/amr_final/coupled_input.bin',
         case.ROOT/'coarsened_composed_reference_b2/amr_final/coupled_input.bin',
         case.ROOT/'amr_full_state_export_v2/final/coupled_input.bin',
         case.ROOT/'gpu_composed_b3_candidate_v1/READBACK.json',
         case.ROOT/'gpu_composed_b3_transaction_v1/READBACK.json')


def freeze():
    if LOCK.exists():
        raise FileExistsError('All-frame source lock is write-once')
    b3.preflight()
    for name in ('gpu_composed_b3_candidate_v1',
                 'gpu_composed_b3_transaction_v1'):
        record = json.loads((case.ROOT/name/'READBACK.json').read_text(
            encoding='utf-8'))
        if not (record.get('gate_B3_numerical_passed') or
                record.get('frozen_gate_B3_transaction_passed')):
            raise ValueError('Prior B3 gate did not pass: '+name)
    payload = dict(schema=1, files={str(path.resolve()):base.sha(path)
                                    for path in PATHS},
        no_CPU_numerical_fallback=True, Delft3D_modified=False)
    base.save(LOCK, payload)
    return dict(lock=str(LOCK), sha256=base.sha(LOCK))


def preflight():
    b3.preflight()
    payload = json.loads(LOCK.read_text(encoding='utf-8'))
    if set(payload['files']) != {str(path.resolve()) for path in PATHS}:
        raise ValueError('All-frame frozen source set changed')
    for path, digest in payload['files'].items():
        if base.sha(path) != digest:
            raise ValueError('All-frame frozen source changed: '+path)
    return dict(lock=str(LOCK), sha256=base.sha(LOCK), verified=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze','preflight'))
    args = parser.parse_args()
    print(json.dumps(globals()[args.action](), indent=2))
