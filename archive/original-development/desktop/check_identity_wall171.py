"""An unchanged project must not restart the corrected state with old wall physics."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'desktop/runtime'),str(ROOT),str(ROOT/'desktop')]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--backend',required=True,choices=('cpu','gpu'));a=ap.parse_args()
    from braids_core.sediment.portable_river_runtime import install
    from braids_core import river_worker
    from check_bedload161 import save,sha
    install(json.loads((ROOT/'desktop/runtime/PRECOMPILED.json').read_text()),ROOT/'desktop/runtime')
    folder=ROOT/f'validation/_wall171/v1/reflection_17_{a.backend}'
    output=folder/'wrong_wall_method'
    try:river_worker.run(folder/'prepared',output,restart=folder/'cancelled/checkpoint.npz')
    except (ValueError,RuntimeError) as exc:message=str(exc)
    else:raise AssertionError('Old wall method accepted candidate checkpoint')
    status=json.loads((output/'STATUS.json').read_text())
    assert status['status']=='failed' and status['accepted_steps']==0,status
    assert 'identit' in message.lower() or 'runtime' in message.lower(),message
    save(folder/'IDENTITY_REJECTION.json',dict(error=message,status=status,script_sha256=sha(__file__)))
    print(message)


if __name__=='__main__':main()
