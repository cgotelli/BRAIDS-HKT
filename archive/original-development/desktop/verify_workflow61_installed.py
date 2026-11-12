"""Installed namespace, executable restart, legacy data-only read and identities."""
import os,sys,json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];RUNTIME=ROOT/'desktop/runtime';REPORT=ROOT/'validation/hydrokinetic_turbines/reports/desktop_workflow61'
sys.path.insert(0,str(RUNTIME));sys.path.insert(1,str(ROOT));sys.dont_write_bytecode=True
if os.name=='nt':dlls=[os.add_dll_directory(str(p)) for p in [Path(sys.base_prefix),Path(sys.base_prefix)/'Library/bin'] if p.is_dir()]
from braids_core import river_worker as worker,river_state_snapshot as snapshot
from braids_core.river_launcher import verify
from braids_core.river_project import load_project
from braids_core.sediment.portable_river_runtime import install
from braids_desktop.results import ResultsStore
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def main():
    verify();install(read(RUNTIME/'PRECOMPILED.json'),RUNTIME)
    assert Path(worker.__file__).resolve()==(RUNTIME/'braids_core/river_worker.py').resolve()
    rows=[]
    for backend in ['cpu','gpu']:
        r=read(REPORT/f'CHECK_{backend}_mapped.json');folder=Path(r['root']);prepared=folder/'target/prepared';project,receipt=worker.load_prepared(prepared)
        checkpoint=next((folder/'checkpoints').glob('t_0.18_step_*.npz'));output=folder/'installed_resume'
        result=worker.run(prepared,output,restart=checkpoint);assert result['status']=='complete'
        a=ResultsStore(folder/'run');a=a.read(len(a.times)-1);b=ResultsStore(output);b=b.read(len(b.times)-1)
        exact={k:bool(np.array_equal(a.arrays[k],b.arrays[k])) for k in ['q','z','mass']};assert all(exact.values())
        rows.append(dict(backend=backend,exact_fields=exact,output=str(output)))
    oldhome=Path.home()/'Desktop/BRAIDS_Turbine_NearWake60/wake4_gpu';old=oldhome/'Project_data/runs/run_20261101_011052_99b4793d/checkpoint.npz'
    raw=snapshot.source_from_checkpoint(old);m,a=snapshot.decode(raw);assert m['time_s']==2000. and len(a['area'])==38736
    _,reference=snapshot.decode((REPORT/'source_v4_test.npz').read_bytes())
    for k in ['q','z']:np.testing.assert_array_equal(a[k],reference[k])
    try:load_project(oldhome/'Project_data/braids_project.json')
    except ValueError as e:rejection=str(e)
    else:raise AssertionError('Old project must not execute')
    try:worker.read_checkpoint(old,worker.checkpoint_identity(project,receipt))
    except ValueError as e:checkpoint_rejection=str(e)
    else:raise AssertionError('Old checkpoint must not execute')
    result=dict(status='passed',runtime_sha256=hashlib.sha256((RUNTIME/'RUNTIME.json').read_bytes()).hexdigest(),source_identity=worker.source_identity(),installed_exact_restarts=rows,
        historical_v4_data=dict(time_s=m['time_s'],cells=len(a['area']),original_fields_exact=True),old_project_execution_rejected=rejection,old_checkpoint_execution_rejected=checkpoint_rejection)
    (REPORT/'INSTALLED_CHECK.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print('Installed CPU/GPU exact restarts, v4 data import and strict legacy execution rejection passed',flush=True)
if __name__=='__main__':main()
