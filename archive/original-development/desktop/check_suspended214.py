"""S1 candidate qualification, including unchanged CPU/GPU bedload kernels."""
import os,sys,json,importlib
from pathlib import Path
from stage_suspended214 import ROOT,OUT,TARGET

sys.path[:0]=[str(TARGET),str(ROOT),str(ROOT/'tests'),str(ROOT/'desktop')]
os.environ['QT_QPA_PLATFORM']='offscreen'
os.environ['BRAIDS_DESKTOP_TEST_RUNTIME']=str(TARGET)
os.environ['OMP_NUM_THREADS']='4'
from braids_core.river_launcher import verify
from braids_core.sediment.portable_river_runtime import install
verify();install(json.loads((TARGET/'PRECOMPILED.json').read_text()),TARGET)
for name in ('validation','validation.sediment','validation.sediment.river_morphodynamics'):
    module=importlib.import_module(name);module.__path__.append(str(ROOT/Path(*name.split('.'))))
import pytest
files=sys.argv[1:] or ['tests/test_suspended_setup214.py','tests/test_bulk213.py',
    'tests/test_jiang_cao207.py','tests/test_jiang_cao208.py','tests/test_jiang_cao209.py',
    'tests/test_jiang_cao210.py','tests/test_jiang_cao206_adjustment.py',
    'tests/test_river_roughness.py','tests/test_jiang_cao199.py']
# The retained test module's historical output directory must not be changed.
import test_bulk213
def backend_parity():
    import numpy as np
    assert len(test_bulk213.RUN_FIELDS)==16
    rows=[]
    for method in test_bulk213.ORDINARY:
        a,b=(test_bulk213.RUN_FIELDS[k,method] for k in ('cpu','gpu'))
        for field in a:np.testing.assert_allclose(a[field],b[field],rtol=2e-7,atol=1e-9)
        rows.append(dict(method=method,max_abs={k:float(np.max(abs(a[k]-b[k]))) for k in a}))
    from stage_suspended214 import save
    save(OUT/'COUPLED_PARITY.json',rows)
test_bulk213.test_coupled_backend_parity=backend_parity
raise SystemExit(pytest.main(['-q',*files,'--tb=short','--maxfail=4','--junitxml='+str(OUT/'REGRESSION.xml')]))
