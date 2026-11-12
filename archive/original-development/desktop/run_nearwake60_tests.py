"""Force packaged-candidate imports before collecting focused tests."""
import json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];TARGET=ROOT/'desktop/qualification/nearwake60/candidate'
sys.path.insert(0,str(TARGET));sys.path.insert(1,str(ROOT));os.environ['QT_QPA_PLATFORM']='offscreen'
if os.name=='nt':dlls=[os.add_dll_directory(str(p)) for p in [Path(sys.base_prefix),Path(sys.base_prefix)/'Library/bin'] if p.is_dir()]
from braids_core.sediment import shoreline_solver
assert Path(shoreline_solver.__file__).resolve()==(TARGET/'braids_core/sediment/shoreline_solver.py').resolve()
from braids_core.sediment.portable_river_runtime import install
install(json.loads((TARGET/'PRECOMPILED.json').read_text()),TARGET)
import pytest
raise SystemExit(pytest.main(sys.argv[1:]))

