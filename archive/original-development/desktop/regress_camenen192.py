"""Run the complete bedload regression screen against an isolated/installed runtime."""
import os,sys,runpy
from stage_camenen192 import ROOT,OUT,TARGET
os.environ.setdefault('BRAIDS_TEST_RUNTIME_OVERRIDE',str(TARGET))
import stage_formula190
stage_formula190.OUT=OUT;stage_formula190.TARGET=TARGET
import pytest
original=pytest.main
def all_tests(args):
    return original(args+['tests/test_grain_diameter191.py','tests/test_camenen192.py'])
pytest.main=all_tests
runpy.run_path(str(ROOT/'desktop/regress_formula190.py'),run_name='__main__')
