"""Fresh one-grain groin and authenticated end-time extension qualification."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUNTIME=ROOT/'validation/_sediment158/candidate'
sys.path[:0]=[str(RUNTIME),str(ROOT)];os.environ['QT_QPA_PLATFORM']='offscreen'
import numpy as np
from PySide6.QtWidgets import QApplication
from braids_desktop.demo import create_demo
from braids_desktop import services
from braids_desktop.forms import SettingsPanel
from braids_desktop.river_physics import set_choice
from braids_core.boundaries.domain import PolygonDomain
from braids_core.sediment.adaptive_river_worker import read_live_frame
from braids_core.river_boundary import FLOW_FREE,TOTAL,RATIO

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--backend',choices=('cpu','gpu'),required=True)
    p.add_argument('--end',type=float,default=120);p.add_argument('--start',type=float,default=40);p.add_argument('--cfl',type=float,default=1.)
    p.add_argument('--stock',type=float,default=0.);p.add_argument('--mf',type=float,default=1);p.add_argument('--zones',action='store_true')
    p.add_argument('--method',choices=('capacity','total','ratio'),default='capacity');p.add_argument('--roughness',choices=('chezy','manning','white_colebrook'),default='chezy')
    p.add_argument('--auxiliary',action='store_true');p.add_argument('--adaptive',action='store_true');p.add_argument('--extend',type=float)
    args=p.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=False);app=QApplication([])
    folder=root/'project';create_demo(folder,args.backend,end_time_s=args.end);doc=services.load_project_folder(folder)
    doc=services.replace_domain(doc,[(0,0),(8,0),(8,3),(10,3),(10,0),(20,0),(20,8),(0,8)],'EPSG:2056',for_editing=True)
    edges=PolygonDomain(**doc['settings']['domain']).edges
    doc=services.assign_boundary(doc,[e.edge_id for e in edges if e.start_xy[0]==e.end_xy[0]==0],'Upstream','inlet',dict(method='inlet.stage_discharge',discharge_m3_s=4.,stage_m=.6),for_editing=True)
    doc=services.assign_boundary(doc,[e.edge_id for e in edges if e.start_xy[0]==e.end_xy[0]==20],'Downstream','outlet',dict(method='outlet.stage',stage_m=.58),for_editing=True)
    f=SettingsPanel();f.load_document(doc);f._dry_stage_m=-1.;f.minimum_size.setValue(.5);f.levels.setValue(2 if args.adaptive else 1);f.cfl.setValue(args.cfl)
    set_choice(f.mesh_mode,'dynamic_depth' if args.adaptive else 'static_terrain');f.adaptation['interval'].setValue(1.)
    set_choice(f.roughness,'roughness.'+args.roughness);f.chezy.setValue(45.)
    f.sediment_enabled.setChecked(True);f.physics.bedload_on.setChecked(True);f.physics.grains.item(0,1).setText('1');f.physics.layers.item(0,0).setText(str(args.stock))
    f.end_time.setValue(args.end);f.dt_max.setValue(.1);f.output_interval.setValue(min(5.,args.end/4));f.sediment_start.setValue(args.start);f.morfac.setValue(args.mf)
    if args.auxiliary:f.secondary.setChecked(True);f.turbulence.setChecked(True);f.viscosity.setValue(.01);f.secondary_diffusion.setValue(.01)
    set_choice(f.forcing.boundary,'Upstream');set_choice(f.forcing.method,{'capacity':FLOW_FREE,'total':TOTAL,'ratio':RATIO}[args.method])
    if args.method!='capacity':f.forcing.table.item(0,2).setText('.0001' if args.method=='total' else '.000025')
    if args.zones:
        rv={'chezy':30.,'manning':.05,'white_colebrook':.01}[args.roughness]
        f.zones['roughness'].load([dict(id='rough bank',points_xy=[[6,0],[16,0],[16,4],[6,4]],value=rv)])
        f.zones['erodible'].load([dict(id='finite patch',points_xy=[[4,4],[12,4],[12,7],[4,7]],value=.05)])
    project=f.document();project['name']='Groin bare-base MPM '+args.backend;services.save_project_folder(folder,project);f.close()
    env=os.environ.copy();env.update(PYTHONPATH=str(RUNTIME),PYTHONNOUSERSITE='1',PYTHONIOENCODING='utf-8');launcher=RUNTIME/'braids_core/river_launcher.py'
    def worker(label,action,*arguments):
        with (root/(label+'.log')).open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,'-B',str(ROOT/'desktop/worker.py'),str(launcher),action,*map(str,arguments)],cwd=RUNTIME,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
    prepared=folder/'prepared/base';out=folder/'runs/first';began=time.perf_counter()
    worker('prepare','prepare','--project',folder/'braids_project.json','--output',prepared,'--asset-cache',root.parent/'asset_cache')
    worker('run','run','--input',prepared,'--output',out,'--checkpoint-dir',folder/'checkpoints/first')
    row,mesh,state=read_live_frame(out);diagnostics=json.loads((out/'DIAGNOSTICS.json').read_text());status=json.loads((out/'STATUS.json').read_text())
    result=dict(time_s=row['time_s'],cells=len(mesh['leaves']),wall_s=time.perf_counter()-began,hydraulic=status['hydraulic'],
        water_residual_m3=max(abs(r['water_residual_m3']) for r in diagnostics),sediment_residual_kg=max(max(map(abs,r['class_residual_kg'])) for r in diagnostics),
        minimum_stock=float(state['mass'].min()),maximum_deposit_m=float((state['z']-state['rock']).max()))
    assert result['water_residual_m3']<1e-7 and result['sediment_residual_kg']<1e-7 and result['minimum_stock']>=0
    if args.extend:
        checkpoint=max((folder/'checkpoints/first').glob('*.json'),key=lambda p:json.loads(p.read_text())['time_s']).with_suffix('.npz')
        current=json.loads((folder/'braids_project.json').read_text());current['settings']['end_time_s']=args.extend
        current['settings']['checkpoint_interval_s']=args.extend;current.pop('editor_state',None)
        extension=root/'EXTENSION.json';extension.write_text(json.dumps(current))
        worker('resumed','run','--input',prepared,'--output',folder/'runs/resumed','--restart',checkpoint,'--continuation-project',extension,'--checkpoint-dir',folder/'checkpoints/resumed')
        # Same physical setup and clock splitting: uninterrupted control through
        # the planned first horizon, then exact selected-checkpoint continuation.
        control=json.loads(json.dumps(project));control['settings']['end_time_s']=args.extend;control['settings']['checkpoint_interval_s']=args.extend;control.pop('editor_state',None)
        cp=root/'CONTROL.json';cp.write_text(json.dumps(control));control['terrain']['dem_path']=str((folder/project['terrain']['dem_path']).resolve());cp.write_text(json.dumps(control))
        worker('control_prepare','prepare','--project',cp,'--output',root/'control_prepared','--asset-cache',root.parent/'asset_cache')
        worker('control','run','--input',root/'control_prepared','--output',root/'control_run')
        resumed=read_live_frame(folder/'runs/resumed')[2];continuous=read_live_frame(root/'control_run')[2]
        result['restart_max_abs']={k:float(np.max(np.abs(resumed[k]-continuous[k]))) for k in ('q','z','mass','bed_boundary')}
        assert max(result['restart_max_abs'].values())<1e-8,result
        result['continued_time_s']=read_live_frame(folder/'runs/resumed')[0]['time_s']
    (root/'CHECK.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
