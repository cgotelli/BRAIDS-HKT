"""Fresh ambient CPU/GPU runs and authenticated power estimates."""
import argparse,copy,time
from pathlib import Path
from check_power_planning75 import document,CASES,ROOT,TARGET,REPORT,read,save
from braids_desktop import preparation,power_planning
from braids_desktop.results import ResultsStore
from braids_core import river_worker

HOME=ROOT/'desktop/qualification/power_planning75'
def run(case,backend):
    root=HOME/(case+'_'+backend);root.mkdir(exist_ok=False)
    d=document(CASES.get(case,'staggered_high'),backend);d['settings']['controls']['threads']=4
    if case.startswith('variable'):
        from braids_desktop.services import _resolve_boundary_state
        d['boundary_states']['Upstream']['hydraulic_schedule']=dict(method='forcing.linear_physical_time',time_s=[0.,4.,8.,12.,16.,20.],discharge_m3_s=[6.496,6.496,6.2,6.2,6.7,6.7] if case=='variable_qualified' else [6.496,6.496,5.6,5.6,6.9,6.9])
        d=_resolve_boundary_state(d)
    save(root/'input.json',d);start=time.monotonic()
    prepared=preparation.prepare(root/'input.json',root/'prepared',asset_cache=HOME/'assets')
    # Every reference starts fresh; an imported coupled checkpoint is rejected
    # before field loading or creating a run directory.
    try:river_worker.run(prepared,root/'rejected',initial_state=root/'does_not_exist.npz')
    except ValueError as e:assert 'fresh turbine-free' in str(e)
    else:raise AssertionError('Disturbed-state guard absent')
    assert not (root/'rejected').exists()
    result=river_worker.run(prepared,root/'run');assert result['status']=='complete'
    store=ResultsStore(root/'run');records=power_planning.calculate(store)
    saved,csv=power_planning.read(store);assert records==saved and b'power.finite_section_gaussian_linear.v1' in csv
    save(root/'DONE.json',dict(result=result,elapsed_s=time.monotonic()-start,power_rows=len(records),last=records[-len(d['power_plan']['layout']['turbines']):],restart_guard=True))
    print('DONE',case,backend,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('case',choices=['all','variable','variable_qualified',*CASES]);p.add_argument('--backend',choices=['cpu','gpu'],default='cpu');a=p.parse_args()
    for case in CASES if a.case=='all' else [a.case]:run(case,a.backend)
