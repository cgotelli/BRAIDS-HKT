"""Existing public transport/roughness/material regressions on new runtime."""
import json,os,sys,importlib
from pathlib import Path
from stage_formula190 import ROOT,TARGET,OUT
TARGET=Path(os.environ.get('BRAIDS_TEST_RUNTIME_OVERRIDE',TARGET)).resolve()
sys.path[:0]=[str(TARGET),str(ROOT),str(ROOT/'tests')]
os.environ['QT_QPA_PLATFORM']='offscreen';os.environ['BRAIDS_DESKTOP_TEST_RUNTIME']=str(TARGET)
os.environ['OMP_NUM_THREADS']='4'
from braids_core.river_launcher import verify
from braids_core.sediment.portable_river_runtime import install
verify();install(json.loads((TARGET/'PRECOMPILED.json').read_text()),TARGET)
for name in ('validation','validation.sediment','validation.sediment.river_morphodynamics'):
    module=importlib.import_module(name);module.__path__.append(str(ROOT/Path(*name.split('.'))))
import validation.sediment.river_morphodynamics.unified_additional_capacity_reference
sys.path[:0]=[str(TARGET),str(ROOT),str(ROOT/'tests')]
import validation.sediment.river_morphodynamics.sediment_wave as wave
wave.ROOT=OUT/'diagnostic_build';packaged=wave.WaveEngine.__init__
def diagnostic_only(self,*args,**kwargs):
    if args[0]=='cpu' and kwargs.get('names')==('matrix_bed_faces_cached','matrix_bed_faces'):
        return packaged.__wrapped__(self,*args,**kwargs)
    return packaged(self,*args,**kwargs)
wave.WaveEngine.__init__=diagnostic_only
import pytest
tests=['tests/test_smart_jaggi183.py','tests/test_desktop_bedload160.py',
    'tests/test_desktop_sediment158.py','tests/test_sediment_depletion158.py',
    'tests/test_river_capacity.py','tests/test_river_additional_capacity.py',
    'tests/test_river_additional_capacity_reference.py','tests/test_river_roughness.py',
    'tests/test_river_material_transport.py','tests/test_bedload_audit189.py',
    'tests/test_ancey_recking190.py','tests/test_recking190.py','tests/test_sediment_clock190.py']
raise SystemExit(pytest.main(['-q',*tests,'--junitxml='+str(OUT/os.environ.get('BRAIDS_TEST_RECEIPT','REGRESSION.xml'))]))
