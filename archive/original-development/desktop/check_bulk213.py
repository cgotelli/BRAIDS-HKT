"""Packaged checks, with no native-source fallback or backend substitution."""
import json,os,sys,importlib
from stage_bulk213 import ROOT,TARGET,OUT
sys.path[:0]=[str(TARGET),str(ROOT),str(ROOT/'tests'),str(ROOT/'desktop')]
os.environ['QT_QPA_PLATFORM']='offscreen';os.environ['BRAIDS_DESKTOP_TEST_RUNTIME']=str(TARGET)
os.environ['OMP_NUM_THREADS']='4'
from braids_core.river_launcher import verify
from braids_core.sediment.portable_river_runtime import install
verify();install(json.loads((TARGET/'PRECOMPILED.json').read_text()),TARGET)
# Tests may use retained canonical validation input builders, never frozen source overlays.
from pathlib import Path
for name in ('validation','validation.sediment','validation.sediment.river_morphodynamics'):
    module=importlib.import_module(name);module.__path__.append(str(ROOT/Path(*name.split('.'))))
import pytest
files=sys.argv[1:] or ['tests/test_bulk213.py','tests/test_jiang_cao207.py',
    'tests/test_jiang_cao208.py','tests/test_jiang_cao209.py','tests/test_jiang_cao210.py',
    'tests/test_jiang_cao206_adjustment.py','tests/test_river_roughness.py',
    'tests/test_jiang_cao199.py']
raise SystemExit(pytest.main(['-q',*files,'--tb=short','--maxfail=4','--junitxml='+str(OUT/'REGRESSION.xml')]))
