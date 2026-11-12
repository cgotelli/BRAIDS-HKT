"""Build and run an isolated, authenticated research source from saved states."""
import argparse,hashlib,json,os,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'desktop/qualification/power_array62/candidate'
CANDIDATE=ROOT/'desktop/qualification/coupled_inflow65/candidate'
REPORT=ROOT/'validation/hydrokinetic_turbines/reports/coupled_inflow65'
HOME=ROOT/'desktop/qualification/coupled_inflow65/runs'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def stage():
    REPORT.mkdir(parents=True,exist_ok=True);CANDIDATE.mkdir(parents=True,exist_ok=False)
    old=read(BASE/'RUNTIME.json');files=dict(old['files'])
    edited='braids_core/sediment/river_turbine_flow.py'
    for n,h in files.items():
        source=BASE/n;assert sha(source)==h,n
        target=CANDIDATE/n;target.parent.mkdir(parents=True,exist_ok=True)
        # Shared files are immutable. Only the changed wrapper gets a private copy.
        if n==edited:shutil.copy2(source,target)
        else:os.link(source,target)
    with (CANDIDATE/edited).open('a',encoding='utf-8') as f:
        f.write('\n# Isolated research identity; never installed into Desktop.\nfrom .coupled_inflow65 import make_array_class\nRotorArray = make_array_class(RotorArray)\n')
    files[edited]=sha(CANDIDATE/edited)
    for n in ['coupled_inflow65.py','rotor_inflow64.py']:
        rel='braids_core/sediment/'+n;shutil.copy2(ROOT/'validation/hydrokinetic_turbines'/n,CANDIDATE/rel);files[rel]=sha(CANDIDATE/rel)
    save(CANDIDATE/'RUNTIME.json',dict(old,files=files))
    save(REPORT/'STAGE.json',dict(parent=sha(BASE/'RUNTIME.json'),candidate=sha(CANDIDATE/'RUNTIME.json'),
        installed_runtime_sha256=sha(ROOT/'desktop/runtime/RUNTIME.json'),native_unchanged=True,
        linked_immutable_files=True,scope='research aligned pair, flat bed, fixed RPM, Mycek TI3 calibration'))


def run(spacing,backend,reference=False,check=False):
    runtime=BASE if reference else CANDIDATE
    sys.path.insert(0,str(runtime));sys.path.insert(1,str(ROOT));sys.dont_write_bytecode=True
    os.environ['QT_QPA_PLATFORM']='offscreen'
    if os.name=='nt':dlls=[os.add_dll_directory(str(p)) for p in [Path(sys.base_prefix),Path(sys.base_prefix)/'Library/bin'] if p.is_dir()]
    from braids_core.sediment import portable_river_runtime
    from braids_core import river_worker,river_state_initialization as init,river_state_snapshot as snap
    from braids_core.river_model import resolve_model
    from braids_desktop import services,preparation
    from braids_desktop.results import ResultsStore
    import numpy as np
    portable_river_runtime.install(read(runtime/'PRECOMPILED.json'),runtime)
    original=Path.home()/'Desktop/BRAIDS_Turbine_Power62'/('tandem_6D_rpm4_v2' if spacing==6 else 'tandem_10D_rpm4')
    name=f'{spacing}D_{backend}_'+('reference' if reference else 'research')+('_check_binary' if check else '')
    folder=HOME/name;folder.mkdir(parents=True,exist_ok=False)
    doc=services.load_project_folder(original/'Project_data');doc['model']['request']['execution']['backend']=backend
    doc['model']=resolve_model(doc['model']['request']).document()
    doc['settings'].update(end_time_s=61. if check else 80.,output_interval_s=.25 if check else 1.,checkpoint_interval_s=.5 if check else 20.)
    doc['settings']['controls']['threads']=8
    doc=services.validate_document(doc);save(folder/'input.json',doc)
    prepared=preparation.prepare(folder/'input.json',folder/'prepared',asset_cache=HOME/'assets')
    source=folder/'source.npz';source.write_bytes(snap.source_from_checkpoint(original/'run/checkpoint.npz'))
    decision=init.prepare(prepared,source,folder/'initialized.npz')
    assert decision['status']=='ready' and not decision['mesh_changed']
    save(folder/'INITIALIZATION.json',decision)
    save(folder/'SPEC.json',dict(runtime=sha(runtime/'RUNTIME.json'),parent_checkpoint=sha(original/'run/checkpoint.npz'),
         backend=backend,spacing_D=spacing,reference=reference,check=check,start_s=60.,end_s=doc['settings']['end_time_s'],
         mapping='exact cell copy/permutation; no interpolation',changes=['turbine source only'] if not reference else []))
    start=time.monotonic();print('Running',name,flush=True)
    result=river_worker.run(prepared,folder/'run',initial_state=folder/'initialized.npz')
    if check:
        river_worker.run(prepared,folder/'part',until=60.5,initial_state=folder/'initialized.npz')
        river_worker.run(prepared,folder/'restart',restart=folder/'part/checkpoint.npz')
        a=ResultsStore(folder/'run').read(-1);b=ResultsStore(folder/'restart').read(-1)
        np.testing.assert_array_equal(a.q,b.q);np.testing.assert_array_equal(a.z,b.z)
        da=read(folder/'run/TURBINE_PERFORMANCE.json')[-2:];db=read(folder/'restart/TURBINE_PERFORMANCE.json')[-2:]
        assert da==db
    save(folder/'DONE.json',dict(result=result,wall_seconds=time.monotonic()-start,exact_restart=check))
    print('Complete',name,time.monotonic()-start,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','run']);p.add_argument('--spacing',type=int,choices=[6,10],default=6)
    p.add_argument('--backend',choices=['cpu','gpu'],default='gpu');p.add_argument('--reference',action='store_true');p.add_argument('--check',action='store_true');a=p.parse_args()
    stage() if a.action=='stage' else run(a.spacing,a.backend,a.reference,a.check)
