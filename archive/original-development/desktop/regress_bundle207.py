"""Packaged old-law regressions plus new Desktop lifecycle admission tests."""
import os,runpy
from stage_bundle207 import ROOT,OUT,TARGET
os.environ['BRAIDS_TEST_RUNTIME_OVERRIDE']=str(TARGET)
os.environ['BRAIDS_TEST_RECEIPT']='PACKAGED_REGRESSION_v3.xml'
import stage_formula190
stage_formula190.OUT=OUT;stage_formula190.TARGET=TARGET
import pytest
original=pytest.main
def suite(args):
    return original(args+['tests/test_grain_diameter191.py','tests/test_camenen192.py','tests/test_wong193.py','tests/test_jiang_cao207.py'])
pytest.main=suite
runpy.run_path(str(ROOT/'desktop/regress_formula190.py'),run_name='__main__')
