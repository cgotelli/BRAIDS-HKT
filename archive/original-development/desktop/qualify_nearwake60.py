"""Candidate regressions and identity checks; no tolerance changes to physics."""
import argparse,json,hashlib,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'desktop/qualification/nearwake60';REPORT=ROOT/'validation/hydrokinetic_turbines/reports/nearwake60'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,d):Path(p).write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
def last(p):
    r=read(p/'frames/INDEX.json')['frames'][-1]
    with np.load(p/'frames'/r['file'],allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}
def regress():
    result=[]
    for name in ['cpu_hydro','cpu_bedload','gpu_hydro','gpu_bedload']:
        check=read(OUT/name/'CHECK.json');assert check['status']=='passed' and check['restart_exact']
        old=last(ROOT/'desktop/qualification/water_budget59'/name/'run');new=last(OUT/name/'run')
        exact={k:bool(np.array_equal(old[k],new[k])) for k in ['q','z','water','bedload_vector_m2_s']}
        assert all(exact.values()),(name,exact)
        result.append(dict(case=name,exact_accepted_preview58_fields=exact,check=check))
    save(REPORT/'REGRESSION.json',dict(status='passed',cases=result));print('Four exact-field AMR/restart controls passed')
def control():
    sys.path.insert(0,str(OUT/'candidate'))
    from braids_desktop.results import ResultsStore
    old=ResultsStore(Path(read(REPORT/'zero8_gpu/STARTED.json')['output']))
    new=ResultsStore(Path(read(REPORT/'zero8_corrected/STARTED.json')['output']))
    assert new.times[-1]>=250
    rows=[]
    for i,t in enumerate(old.times):
        j=int(np.flatnonzero(new.times==t)[0]);a,b=old.read(i),new.read(j)
        exact={k:bool(np.array_equal(getattr(a,k),getattr(b,k))) for k in ['q','z']}
        assert all(exact.values()),(t,exact)
        rows.append(dict(time_s=float(t),exact=exact))
    save(REPORT/'CONTROL_ARITHMETIC_EXACT.json',dict(status='passed',times=rows,
        interpretation='Independent cold runs are field-identical through the old cumulative-check failure; no changed-runtime restart.'))
    print('Cold zero-load runs are field-identical through 250 s')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['regress','control']);globals()[p.parse_args().action]()
