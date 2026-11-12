"""S2 checkout regression; never modifies installed or sealed S1 evidence."""
import json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else ROOT/'validation/_suspended215/qualified'
if OUT.exists():raise SystemExit('Choose a fresh qualification output; preserve prior evidence')
OUT.mkdir(parents=True)
sys.path[:0]=[str(ROOT),str(ROOT/'tests'),str(ROOT/'desktop')]
os.environ['QT_QPA_PLATFORM']='offscreen';os.environ['OMP_NUM_THREADS']='4'
import pytest
import test_bulk213
def parity():
    import numpy as np
    rows=[]
    assert len(test_bulk213.RUN_FIELDS)==16
    for method in test_bulk213.ORDINARY:
        a,b=(test_bulk213.RUN_FIELDS[k,method] for k in ('cpu','gpu'))
        for field in a:np.testing.assert_allclose(a[field],b[field],rtol=2e-7,atol=1e-9)
        rows.append(dict(method=method,max_abs={k:float(np.max(abs(a[k]-b[k]))) for k in a}))
    (OUT/'BEDLOAD_PARITY.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
test_bulk213.test_coupled_backend_parity=parity
files=['tests/test_suspended_core215.py','tests/test_suspended_exchange215.py','tests/test_suspended_setup214.py','tests/test_bulk213.py',
       'tests/test_jiang_cao207.py','tests/test_jiang_cao208.py','tests/test_jiang_cao209.py',
       'tests/test_jiang_cao210.py','tests/test_jiang_cao206_adjustment.py',
       'tests/test_river_roughness.py','tests/test_jiang_cao199.py']
raise SystemExit(pytest.main(['-q',*files,'--tb=short','--maxfail=4',
    '--basetemp='+str(OUT/'tests'),'--junitxml='+str(OUT/'REGRESSION.xml')]))
