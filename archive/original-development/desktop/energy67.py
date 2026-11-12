"""Build a device-configured candidate and short speed-surface/energy trials."""
import argparse,csv,hashlib,json,os,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'desktop/runtime';CANDIDATE=Path(os.environ.get('BRAIDS_ENERGY_RUNTIME',str(ROOT/'desktop/qualification/energy67/candidate')))
REPORT=ROOT/'validation/hydrokinetic_turbines/reports/energy67';HOME=ROOT/'desktop/qualification/energy67/runs'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def stage():
    REPORT.mkdir(parents=True,exist_ok=True);CANDIDATE.mkdir(parents=True,exist_ok=False)
    old=read(BASE/'RUNTIME.json');files=dict(old['files'])
    edits=['braids_core/river_turbines.py','braids_core/sediment/river_turbine_flow.py','braids_core/boundaries/breaklines.py']
    additions=['braids_core/river_turbine_surface.py','braids_core/sediment/river_turbine_inflow.py']
    for n,h in files.items():
        s=BASE/n;t=CANDIDATE/n;t.parent.mkdir(parents=True,exist_ok=True);assert sha(s)==h,n
        if n in edits:shutil.copy2(ROOT/n,t)
        else:os.link(s,t)
    for n in additions:
        t=CANDIDATE/n;t.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,t)
    for n in edits+additions:files[n]=sha(CANDIDATE/n)
    save(CANDIDATE/'RUNTIME.json',dict(old,files=files));save(REPORT/'STAGE.json',dict(parent=sha(BASE/'RUNTIME.json'),candidate=sha(CANDIDATE/'RUNTIME.json'),edits=edits+additions,native_unchanged=True,installed_unchanged=True))

def control(ti,rpm):
    from validation.hydrokinetic_turbines.performance66 import extract,coefficient
    import numpy as np
    rows=extract();speeds=sorted({r['inflow_m_s'] for r in rows if r['TI_percent']==ti})
    tables=[dict(inflow_m_s=u,curve=[dict(tip_speed_ratio=float(t),thrust_coefficient=coefficient(rows,ti,u,t,'Ct'),power_coefficient=coefficient(rows,ti,u,t,'Cp')) for t in np.arange(1.,7.51,.25)]) for u in speeds]
    if ti==15:a=read(REPORT.parent/'performance66/AUDIT.json');k=a['growth'];eps=a['initial_width']
    else:k=.017872630937080202;eps=.2806331261614108
    return dict(mode='fixed_rpm_surface',model_name=f'Mycek isolated model rotor, TI{ti}%, original speed curves',rotational_speed_rpm=rpm,
      performance=tables,outside_surface='error',wake=dict(method='gaussian_zero_mean_single_wake',growth=k,initial_width=eps,minimum_distance_D=3.,provenance=f'Mycek TI{ti}% single rotor 4D/8D fit; not a universal river calibration'))

def run(case,backend,check=False):
    sys.path[:0]=[str(CANDIDATE),str(ROOT)];sys.dont_write_bytecode=True;os.environ['QT_QPA_PLATFORM']='offscreen'
    if os.name=='nt':dlls=[os.add_dll_directory(str(p)) for p in [Path(sys.base_prefix),Path(sys.base_prefix)/'Library/bin'] if p.is_dir()]
    from braids_core.sediment import portable_river_runtime
    from braids_core import river_worker,river_state_initialization as init,river_state_snapshot as snap
    from braids_core.river_model import resolve_model
    from braids_core.river_open_boundary import resolve as resolve_boundaries
    from braids_core.river_turbine_surface import integrate_energy
    from braids_desktop import services,preparation
    from braids_desktop.results import ResultsStore
    import numpy as np
    portable_river_runtime.install(read(CANDIDATE/'PRECOMPILED.json'),CANDIDATE)
    ti=3 if case.startswith('low') else 15;spacing=10 if case.endswith('10') else 6
    prior=ROOT/f'desktop/qualification/{"coupled_inflow65" if ti==3 else "performance66"}/runs/{spacing}D_{"cpu_research" if ti==3 else "cpu_nu1"}'
    suffix=CANDIDATE.name.removeprefix('candidate')
    folder=HOME/(f'{case}_{backend}'+('_check' if check else '')+suffix)
    folder.mkdir(parents=True,exist_ok=False);doc=read(prior/'input.json');doc['model']['request']['execution']['backend']=backend;doc['model']=resolve_model(doc['model']['request']).document()
    start=80. if ti==3 else 60.;end=start+(.5 if check else 20.)
    doc['settings'].update(end_time_s=end,output_interval_s=.125 if check else .5,checkpoint_interval_s=.25 if check else 20.)
    if check:doc['settings']['dt_max_s']=.125
    for r in doc['turbines']['turbines']:r['control']=control(ti,r['control']['rotational_speed_rpm'])
    if case=='variable':
        doc['boundary_states']['Upstream']['hydraulic_schedule']=dict(method='forcing.linear_physical_time',time_s=[0.,60.,65.,70.,75.,80.],discharge_m3_s=[6.64,6.64,5.6,5.6,6.64,6.64])
        doc['boundary_resolution']=resolve_boundaries(doc['settings']['hydraulics'],doc['boundary_states'],dict(Upstream='inlet',Downstream='outlet'),sediment=None,ng=1,end_time=end)
    # Use the user's decimal representation that failed in66, to exercise fix.
    if spacing==6:doc['turbines']['turbines'][1]['center_xy'][0]=9.2
    doc=services.validate_document(doc);save(folder/'input.json',doc)
    prepared=preparation.prepare(folder/'input.json',folder/'prepared',asset_cache=HOME/'assets')
    source=folder/'source.npz';source.write_bytes(snap.source_from_checkpoint(prior/'run/checkpoint.npz'))
    decision=init.prepare(prepared,source,folder/'initialized.npz')
    if decision['status']=='mapping_required':
        save(folder/'MAPPING_DETECTED.json',decision)
        decision=init.prepare(prepared,source,folder/'initialized.npz',allow_mapping=True)
    assert decision['status']=='ready';save(folder/'INITIALIZATION.json',decision)
    save(folder/'SPEC.json',dict(runtime=sha(CANDIDATE/'RUNTIME.json'),prior_checkpoint=sha(prior/'run/checkpoint.npz'),case=case,backend=backend,ti=ti,spacing_D=spacing,start_s=start,end_s=end,check=check,changes='device speed/TSR surface and rotor moments; explicit source wake data; decimal mesh fix; variable case changes inlet only'))
    clock=time.monotonic();print('Running',folder.name,flush=True)
    result=river_worker.run(prepared,folder/'run',initial_state=folder/'initialized.npz')
    if check:
        river_worker.run(prepared,folder/'part',initial_state=folder/'initialized.npz',until=start+.25)
        river_worker.run(prepared,folder/'restart',restart=folder/'part/checkpoint.npz')
        a=ResultsStore(folder/'run').read(-1);b=ResultsStore(folder/'restart').read(-1);np.testing.assert_array_equal(a.q,b.q);np.testing.assert_array_equal(a.z,b.z)
        assert read(folder/'run/TURBINE_PERFORMANCE.json')[-2:]==read(folder/'restart/TURBINE_PERFORMANCE.json')[-2:]
        before=read(folder/'part/TURBINE_PERFORMANCE.json');after=read(folder/'restart/TURBINE_PERFORMANCE.json')
        # Explicit join: shared checkpoint samples must agree before dropping
        # the duplicate, rather than integrating two copies of its timestamp.
        lookup={(r['time_s'],r['turbine_id']):r for r in before}
        for r in after:
            key=(r['time_s'],r['turbine_id'])
            if key in lookup:assert r['useful_power_W']==lookup[key]['useful_power_W']
            else:lookup[key]=r
        key=lambda r:(r['turbine_id'],r['time_s'])
        assert sorted(integrate_energy(list(lookup.values())),key=key)==sorted(integrate_energy(read(folder/'run/TURBINE_PERFORMANCE.json')),key=key)
    records=read(folder/'run/TURBINE_PERFORMANCE.json');energy=integrate_energy(records)
    with (folder/'TURBINE_ENERGY.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(energy[0]));w.writeheader();w.writerows(energy)
    save(folder/'DONE.json',dict(result=result,wall_seconds=time.monotonic()-clock,exact_restart=check));print('Complete',folder.name,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','run']);p.add_argument('--case',choices=['high6','high10','low6','low10','variable'],default='high6');p.add_argument('--backend',choices=['cpu','gpu'],default='cpu');p.add_argument('--check',action='store_true');a=p.parse_args()
    stage() if a.action=='stage' else run(a.case,a.backend,a.check)
