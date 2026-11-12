"""Fresh MPM bank-attached groin demonstration and warm-up/restart checks."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RUNTIME=Path(os.environ.get('BRAIDS_DESKTOP_TEST_RUNTIME',ROOT/'validation/_bedload157/candidate'))
sys.path[:0]=[str(RUNTIME),str(ROOT)]
os.environ['QT_QPA_PLATFORM']='offscreen'
from PySide6.QtWidgets import QApplication
import numpy as np
from braids_desktop.demo import create_demo
from braids_desktop import services
from braids_desktop.forms import SettingsPanel
from braids_desktop.river_physics import set_choice
from braids_core.boundaries.domain import PolygonDomain
from braids_core.sediment.adaptive_river_worker import read_live_frame


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--backend',choices=('cpu','gpu'),required=True)
    parser.add_argument('--end',type=float,default=120.);parser.add_argument('--start',type=float,default=40.)
    parser.add_argument('--cell',type=float,default=.5);parser.add_argument('--control',action='store_true')
    parser.add_argument('--restart-check',action='store_true')
    parser.add_argument('--adaptive',action='store_true')
    args=parser.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    app=QApplication([]);folder=root/'project';create_demo(folder,args.backend,end_time_s=args.end)
    doc=services.load_project_folder(folder)
    # A bank-attached obstruction is a notch in the exterior wall, not an
    # interior island with an artificial narrow gap against the bank.
    domain=[(0,0),(8,0),(8,3),(10,3),(10,0),(20,0),(20,8),(0,8)]
    if args.control:domain=[(0,0),(20,0),(20,8),(0,8)]
    doc=services.replace_domain(doc,domain,'EPSG:2056',for_editing=True)
    edges=PolygonDomain(**doc['settings']['domain']).edges
    left=[e.edge_id for e in edges if e.start_xy[0]==e.end_xy[0]==0]
    right=[e.edge_id for e in edges if e.start_xy[0]==e.end_xy[0]==20]
    doc=services.assign_boundary(doc,left,'Upstream','inlet',dict(method='inlet.stage_discharge',discharge_m3_s=4.,stage_m=.6),for_editing=True)
    doc=services.assign_boundary(doc,right,'Downstream','outlet',dict(method='outlet.stage',stage_m=.58),for_editing=True)
    f=SettingsPanel();f.load_document(doc);f._dry_stage_m=-1.
    f.minimum_size.setValue(args.cell);f.levels.setValue(1);f.cfl.setValue(.45)
    set_choice(f.mesh_mode,'dynamic_depth' if args.adaptive else 'static_terrain')
    if args.adaptive:f.levels.setValue(2);f.adaptation['interval'].setValue(1.)
    set_choice(f.roughness,'roughness.chezy');f.chezy.setValue(45.)
    f.sediment_enabled.setChecked(True);f.physics.bedload_on.setChecked(True)
    f.physics.grains.item(0,1).setText('1');f.physics.layers.item(0,0).setText('.5')
    f.dt_max.setValue(.1);f.end_time.setValue(args.end);f.output_interval.setValue(min(5.,args.start/2))
    f.sediment_start.setValue(args.start)
    from braids_core.river_boundary import FLOW_FREE
    for name in ('Upstream','Downstream'):
        set_choice(f.forcing.boundary,name);set_choice(f.forcing.method,FLOW_FREE)
    project=f.document();project['name']='MPM groin — '+args.backend
    services.save_project_folder(folder,project);f.close()
    env=os.environ.copy();env.update(PYTHONPATH=str(RUNTIME),PYTHONNOUSERSITE='1',PYTHONIOENCODING='utf-8')
    launcher=RUNTIME/'braids_core/river_launcher.py'
    def worker(label,action,*arguments):
        with (root/(label+'.log')).open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,'-B',str(ROOT/'desktop/worker.py'),str(launcher),action,*map(str,arguments)],
                cwd=RUNTIME,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=2400)
    started=time.perf_counter();prepared=root/'prepared';out=root/'run'
    worker('prepare','prepare','--project',folder/'braids_project.json','--output',prepared,'--asset-cache',root.parent/'asset_cache')
    worker('run','run','--input',prepared,'--output',out)
    row,mesh,state=read_live_frame(out);diagnostics=json.loads((out/'DIAGNOSTICS.json').read_text())
    assert np.isfinite(state['q']).all() and state['q'][0].min()>=0
    assert all(abs(r['water_residual_m3'])<=r['water_roundoff_limit_m3'] for r in diagnostics)
    assert all(max(map(abs,r['class_residual_kg']))<=r['class_roundoff_limit_kg'] for r in diagnostics)
    assert all(not any(r['wall_integrals']) for r in diagnostics)
    result=dict(time_s=row['time_s'],cells=len(mesh['leaves']),steps=row['steps'],wall_s=time.perf_counter()-started,
                water_residual_m3=max(abs(r['water_residual_m3']) for r in diagnostics),
                sediment_residual_kg=max(max(map(abs,r['class_residual_kg'])) for r in diagnostics))
    if args.restart_check:
        # Cross activation from an authenticated pre-activation checkpoint.
        worker('segment','run','--input',prepared,'--output',root/'segment','--until',args.start/2)
        worker('restart','run','--input',prepared,'--output',root/'resumed','--restart',root/'segment/checkpoint.npz')
        other=read_live_frame(root/'resumed')[2]
        result['restart_max_abs']={k:float(np.max(np.abs(state[k]-other[k]))) for k in ('q','z','mass','bed_boundary')}
        assert max(result['restart_max_abs'].values())<1e-8,result
    (root/'CHECK.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
