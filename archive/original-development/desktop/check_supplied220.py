"""Compiler-free full release checks with separate S7 evidence."""
import os,sys,json,importlib
from stage_supplied220 import ROOT,OUT,TARGET,save
sys.path[:0]=[str(TARGET),str(ROOT),str(ROOT/'tests'),str(ROOT/'desktop')]
os.environ['QT_QPA_PLATFORM']='offscreen';os.environ['OMP_NUM_THREADS']='4'
os.environ['BRAIDS_DESKTOP_TEST_RUNTIME']=str(TARGET)
from braids_core.river_launcher import verify
from braids_core.sediment.portable_river_runtime import install
verify();install(json.loads((TARGET/'PRECOMPILED.json').read_text()),TARGET)
from pathlib import Path
check_dir=Path(os.environ.get('BRAIDS_S7_CHECK_DIR',str(OUT/'packaged_final'))).resolve()
if check_dir.exists():raise FileExistsError('Preserve evidence; set BRAIDS_S7_CHECK_DIR to a new folder')
xml=check_dir.with_suffix('.xml') if 'BRAIDS_S7_CHECK_DIR' in os.environ else OUT/'REGRESSION.xml'
for name in ('validation','validation.sediment','validation.sediment.river_morphodynamics'):
    module=importlib.import_module(name);module.__path__.append(str(ROOT/Path(*name.split('.'))))
import pytest,test_bulk213
def parity():
    import numpy as np
    assert len(test_bulk213.RUN_FIELDS)==16
    for method in test_bulk213.ORDINARY:
        a,b=(test_bulk213.RUN_FIELDS[k,method] for k in ('cpu','gpu'))
        for field in a:np.testing.assert_allclose(a[field],b[field],rtol=2e-7,atol=1e-9)
test_bulk213.test_coupled_backend_parity=parity
files=sys.argv[1:] or ['tests/test_supplied220.py','tests/test_suspended_classic219.py','tests/test_suspended_dl218.py','tests/test_suspended_gp217.py','tests/test_suspended_vr216.py','tests/test_suspended_setup214.py',
    'tests/test_suspended_core215.py','tests/test_suspended_exchange215.py',
    'tests/test_river_vr1993.py','tests/test_river_vr1993_binding.py','tests/test_bulk213.py',
    'tests/test_jiang_cao207.py','tests/test_jiang_cao208.py','tests/test_jiang_cao209.py',
    'tests/test_jiang_cao210.py','tests/test_jiang_cao206_adjustment.py',
    'tests/test_river_roughness.py','tests/test_jiang_cao199.py']
code=pytest.main(['-q',*files,'--tb=short','--maxfail=5','--basetemp='+str(check_dir),'--junitxml='+str(xml)])
if not code:
    import test_supplied220 as gp
    import numpy as np
    rows=[]
    for (case,bed,backend),(a,metrics) in gp.RUNS.items():
        if backend!='cpu':continue
        b,other=gp.RUNS[case,bed,'gpu']
        rows.append(dict(method=gp.METHOD,case=case,bed=bed,cpu=metrics,gpu=other,
            max_abs={k:float(np.max(abs(a[k]-b[k]))) for k in ('q','z','mass','suspended')}))
    if rows:save(OUT/'PARITY.json',rows)
raise SystemExit(code)


