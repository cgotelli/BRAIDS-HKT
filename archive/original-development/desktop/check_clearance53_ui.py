import os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];TARGET=ROOT/'desktop/qualification/turbine_clearance53/candidate'
os.environ['QT_QPA_PLATFORM']='offscreen';os.environ['BRAIDS_DESKTOP_TEST_RUNTIME']=str(TARGET);sys.path.insert(0,str(TARGET))
from braids_desktop import forms,turbines
assert Path(forms.__file__).is_relative_to(TARGET)
import pytest
if __name__=='__main__':raise SystemExit(pytest.main([str(ROOT/'tests/test_hydrokinetic_clearance_ui_hkt4b.py'),'-q','--junitxml='+str(ROOT/'validation/hydrokinetic_turbines/reports/clearance_hkt4b/UI_PACKAGED.xml')]))
