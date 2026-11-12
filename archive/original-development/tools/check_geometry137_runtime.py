"""Exercise the authenticated staged solver, never mixed checkout modules."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--runtime',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();runtime=args.runtime.resolve()
    sys.dont_write_bytecode=True
    sys.path.insert(0,str(ROOT));sys.path.insert(0,str(runtime))
    from braids_core.river_launcher import verify
    manifest=verify()
    from braids_core.sediment.portable_river_runtime import install
    install(json.loads((runtime/'PRECOMPILED.json').read_text()),runtime)
    os.environ['QT_QPA_PLATFORM']='offscreen'
    os.environ['BRAIDS_DESKTOP_TEST_RUNTIME']=str(runtime)
    import pytest
    result=pytest.main(['tests/test_desktop_services.py',
        'tests/test_cfl_public_execution131.py',
        '-q','--tb=short','--junitxml='+str(args.output)])
    origins={name:mod.__file__ for name,mod in sys.modules.items()
        if name.startswith(('braids_core.','braids_gpu.')) and getattr(mod,'__file__',None)}
    assert all(Path(p).resolve().is_relative_to(runtime) for p in origins.values()),origins
    assert verify()==manifest
    raise SystemExit(result)
