"""Fresh coupled Desktop examples and paired restart checks; no reference edits."""
import argparse,time
from coupled_wake76 import ROOT,TARGET,REPORT,read,save,environment
environment()
from braids_core import river_worker
from braids_core.river_model import resolve_model
from braids_desktop import preparation,services
from braids_desktop.results import ResultsStore
import numpy as np

HOME=TARGET.parent
CASES={'single':'single_high','tandem':'tandem_high6','staggered':'staggered_high'}

def run(case,backend,short=False):
    folder=HOME/(case+'_'+backend+('_short' if short else ''))
    if folder.exists():
        if (folder/'DONE.json').exists():return
        result=read(folder/'run/PROGRESS.json');assert result['status']=='complete' and not short
        rows=read(folder/'run/TURBINE_PERFORMANCE.json')
        assert all(r['useful_power_W']<=r['hydraulic_work_W']+1e-9 for r in rows)
        d=read(folder/'input.json')
        save(folder/'DONE.json',dict(result=result,elapsed_s=result['wall_s'],exact_restart=None,
             water_residual_m3=max(abs(r['water_residual_m3']) for r in read(folder/'run/DIAGNOSTICS.json')),
             last=rows[-len(d['turbines']['turbines']):],recovered_postprocessing=True))
        return
    folder.mkdir(exist_ok=False)
    d=read(ROOT/'desktop/qualification/selection74'/CASES[case]/'input.json')
    d.pop('power_plan',None);d.pop('editor_state',None)
    for r in d['turbines']['turbines']:r['control']['wake']['method']='gaussian_zero_mean_linear_overlap'
    d['model']['request']['execution']['backend']=backend
    d['model']=resolve_model(d['model']['request']).document()
    end=2. if short else 40.
    d['settings'].update(end_time_s=end,checkpoint_interval_s=1. if short else end,output_interval_s=.5 if short else 2.)
    d['settings']['controls']['threads']=4
    d=services.validate_document(d);save(folder/'input.json',d)
    start=time.monotonic();prepared=preparation.prepare(folder/'input.json',folder/'prepared',asset_cache=HOME/'assets')
    result=river_worker.run(prepared,folder/'run');assert result['status']=='complete'
    exact=None
    if short:
        river_worker.run(prepared,folder/'part',until=1.)
        river_worker.run(prepared,folder/'restart',restart=folder/'part/checkpoint.npz')
        a=ResultsStore(folder/'run').read(-1);b=ResultsStore(folder/'restart').read(-1)
        np.testing.assert_array_equal(a.q,b.q);np.testing.assert_array_equal(a.z,b.z)
        assert read(folder/'run/TURBINE_PERFORMANCE.json')[-3:]==read(folder/'restart/TURBINE_PERFORMANCE.json')[-3:]
        exact=True
    rows=read(folder/'run/TURBINE_PERFORMANCE.json')
    assert all(r['useful_power_W']<=r['hydraulic_work_W']+1e-9 for r in rows)
    save(folder/'DONE.json',dict(result=result,elapsed_s=time.monotonic()-start,exact_restart=exact,
         water_residual_m3=max(abs(r['water_residual_m3']) for r in read(folder/'run/DIAGNOSTICS.json')),
         last=rows[-len(d['turbines']['turbines']):]))
    print('DONE',case,backend,'short' if short else '',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('case',choices=['all',*CASES]);p.add_argument('--backend',default='gpu');p.add_argument('--short',action='store_true');a=p.parse_args()
    for case in CASES if a.case=='all' else [a.case]:run(case,a.backend,a.short)
