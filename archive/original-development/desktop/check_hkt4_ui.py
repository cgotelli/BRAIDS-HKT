"""Use actual packaged widgets in the isolated Qt environment."""
import os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];TARGET=ROOT/'desktop/qualification/hkt4/candidate'
os.environ['QT_QPA_PLATFORM']='offscreen';os.environ['BRAIDS_DESKTOP_TEST_RUNTIME']=str(TARGET)
sys.path.insert(0,str(TARGET))
from braids_desktop import forms,shell,services
assert Path(forms.__file__).is_relative_to(TARGET)
import pytest
if __name__=='__main__':raise SystemExit(pytest.main([str(ROOT/'tests/test_hydrokinetic_desktop_ui_hkt4.py'),'-q','--junitxml='+str(ROOT/'validation/hydrokinetic_turbines/reports/desktop_hkt4/UI_PACKAGED.xml')]))
