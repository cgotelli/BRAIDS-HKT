"""Actual matched hydraulic CPU/GPU and exact restart under guarded loading."""
import os,sys,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];RUNTIME=ROOT/'desktop/qualification/power_array62/candidate';REPORT=ROOT/'validation/hydrokinetic_turbines/reports/power_array62'
sys.path.insert(0,str(RUNTIME));sys.path.insert(1,str(ROOT));sys.dont_write_bytecode=True
if os.name=='nt':dlls=[os.add_dll_directory(str(p)) for p in [Path(sys.base_prefix),Path(sys.base_prefix)/'Library/bin'] if p.is_dir()]
from braids_desktop import services,preparation
from braids_desktop.results import ResultsStore
from braids_core import river_worker
from braids_core.river_model import resolve_model
from braids_core.sediment.portable_river_runtime import install
def save(p,d):Path(p).write_text(json.dumps(d,indent=2)+'\n')
def main():
    install(json.loads((RUNTIME/'PRECOMPILED.json').read_text()),RUNTIME)
    original=services.load_project_folder(Path.home()/'Desktop/BRAIDS_Turbine_Power62/tandem_5D/Project_data')
    states={};metrics={}
    out=RUNTIME.parent/'coupled_checks';out.mkdir(exist_ok=False)
    for backend in ['cpu','gpu']:
        root=out/backend;root.mkdir();doc=json.loads(json.dumps(original));request=doc['model']['request'];request['execution']['backend']=backend;doc['model']=resolve_model(request).document()
        doc['settings'].update(end_time_s=.8,output_interval_s=.2,checkpoint_interval_s=.4)
        doc=services.validate_document(doc);save(root/'input.json',doc)
        prepared=preparation.prepare(root/'input.json',root/'prepared',asset_cache=out/'assets')
        river_worker.run(prepared,root/'full')
        river_worker.run(prepared,root/'part',until=.4)
        river_worker.run(prepared,root/'continued',restart=root/'part/checkpoint.npz')
        f=ResultsStore(root/'full').read(-1);r=ResultsStore(root/'continued').read(-1)
        np.testing.assert_array_equal(f.q,r.q);np.testing.assert_array_equal(f.z,r.z)
        full=json.loads((root/'full/TURBINE_PERFORMANCE.json').read_text());continued=json.loads((root/'continued/TURBINE_PERFORMANCE.json').read_text())
        for name in ['Upstream rotor','Downstream rotor']:
            a=[v for v in full if v['turbine_id']==name][-1];b=[v for v in continued if v['turbine_id']==name][-1];assert a==b
        metrics[backend]=dict(exact_restart=True,final_time=f.time_s,cells=len(f.polygons),root=str(root))
        states[backend]=f.q.copy()
    error=float(np.max(abs(states['cpu']-states['gpu'])));assert error<1e-10,error
    save(REPORT/'COUPLED_CHECKS.json',dict(status='passed',backends=metrics,maximum_conserved_field_difference=error))
    print('Paired source, coupled hydraulics and restart passed',error)
if __name__=='__main__':main()
