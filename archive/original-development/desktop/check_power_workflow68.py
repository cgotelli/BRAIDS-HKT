"""Current UI -> folder -> preparation -> CPU/GPU -> restart/energy checks."""
import argparse,csv,json,os,sys,time
from pathlib import Path
from power_workflow68 import ROOT,TARGET,REPORT,read,save,sha
sys.path[:0]=[str(TARGET),str(ROOT)];sys.dont_write_bytecode=True;os.environ['QT_QPA_PLATFORM']='offscreen'
if os.name=='nt':dlls=[os.add_dll_directory(str(p)) for p in [Path(sys.base_prefix),Path(sys.base_prefix)/'Library/bin'] if p.is_dir()]

def run(backend,check=False):
    import numpy as np
    from PySide6.QtWidgets import QApplication
    from braids_core.sediment import portable_river_runtime
    from braids_core import river_worker,river_state_snapshot as snap,river_state_initialization as init
    from braids_desktop import services,preparation
    from braids_desktop.forms import SettingsPanel
    from braids_desktop.results import ResultsStore
    from braids_core.river_turbine_surface import integrate_energy
    from braids_core.river_model import resolve_model
    from braids_core.river_open_boundary import resolve as resolve_boundaries
    portable_river_runtime.install(read(TARGET/'PRECOMPILED.json'),TARGET)
    app=QApplication.instance() or QApplication([])
    home=ROOT/'desktop/qualification/power_workflow68'/('restart_'+backend if check else 'variable_'+backend)
    home.mkdir(parents=True,exist_ok=False)
    source=ROOT/'desktop/qualification/energy67/runs/variable_cpu/input.json';doc=read(source)
    doc['settings'].update(end_time_s=60.5 if check else 80.,output_interval_s=.125 if check else .5,checkpoint_interval_s=.25 if check else 20.,dt_max_s=.125 if check else .2)
    doc['name']='Power workflow 68 '+backend
    doc['model']['request']['execution']['backend']=backend;doc['model']=resolve_model(doc['model']['request']).document()
    if check:doc['boundary_states']['Upstream']['hydraulic_schedule']=dict(method='forcing.constant')
    doc['boundary_resolution']=resolve_boundaries(doc['settings']['hydraulics'],doc['boundary_states'],dict(Upstream='inlet',Downstream='outlet'),sediment=None,ng=1,end_time=doc['settings']['end_time_s'])
    # Exercise actual UI load/edit/save, not an alternative input schema.
    panel=SettingsPanel();panel.load_document(doc)
    e=panel.turbines
    for i in range(len(e.rows)):
        e.selection.setCurrentIndex(i);e.surface.operation.setChecked(True);e.surface.cut_in.setValue(.64);e.surface.cut_out.setValue(1.1);e.surface.efficiency.setValue(.85)
        e.surface.operating_source.setText('Verification only: local-plane speed window 0.64–1.1 m/s; constant 85% conversion. Not manufacturer data.')
    doc=panel.document(doc);panel.close();app.processEvents()
    doc=services.save_project_folder(home/'project',doc,source_base=source.parent);save(home/'input.json',doc)
    prepared=preparation.prepare(home/'input.json',home/'prepared',asset_cache=home.parent/'assets')
    checkpoint=ROOT/'desktop/qualification/performance66/runs/6D_cpu_nu1/run/checkpoint.npz'
    src=home/'source.npz';src.write_bytes(snap.source_from_checkpoint(checkpoint));initial=home/'initialized.npz'
    decision=init.prepare(prepared,src,initial);assert decision['status']=='ready' and not decision['mesh_changed'];save(home/'INITIALIZATION.json',decision)
    started=time.monotonic();result=river_worker.run(prepared,home/'run',initial_state=initial)
    data=read(home/'run/TURBINE_PERFORMANCE.json');energy=integrate_energy(data)
    for row in data:
        assert abs(row['electrical_power_W']-.85*row['useful_power_W'])<1e-10
        if not row['active']:assert row['useful_power_W']==row['thrust_N']==0
    if check:
        river_worker.run(prepared,home/'part',initial_state=initial,until=60.25)
        river_worker.run(prepared,home/'restart',restart=home/'part/checkpoint.npz')
        a,b=(ResultsStore(home/n).read(-1) for n in ['run','restart']);np.testing.assert_array_equal(a.q,b.q);np.testing.assert_array_equal(a.z,b.z)
        assert data[-2:]==read(home/'restart/TURBINE_PERFORMANCE.json')[-2:]
        joined={}
        for row in read(home/'part/TURBINE_PERFORMANCE.json')+read(home/'restart/TURBINE_PERFORMANCE.json'):
            key=row['turbine_id'],row['time_s']
            if key in joined:assert joined[key]==row
            joined[key]=row
        key=lambda r:(r['turbine_id'],r['time_s'])
        assert sorted(integrate_energy(list(joined.values())),key=key)==sorted(energy,key=key)
    with (home/'TURBINE_ENERGY.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(energy[0]));w.writeheader();w.writerows(energy)
    save(home/'DONE.json',dict(result=result,wall_seconds=time.monotonic()-started,exact_restart=check,runtime=sha(TARGET/'RUNTIME.json')));print(home,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--backend',choices=['cpu','gpu'],required=True);p.add_argument('--check',action='store_true');a=p.parse_args();run(a.backend,a.check)
