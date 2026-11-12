"""Small self-contained Desktop supplied-suspension example, no erodible stock."""
import os,sys,json,subprocess
from stage_supplied220 import ROOT,OUT,TARGET,REPORT,sha,save
sys.path[:0]=[str(TARGET),str(ROOT),str(ROOT/'tests')]
os.environ['QT_QPA_PLATFORM']='offscreen'
from PySide6.QtWidgets import QApplication
from test_supplied220 import document
from braids_desktop import services
from braids_desktop.forms import SettingsPanel
from braids_core.sediment.adaptive_river_worker import read_live_frame
import numpy as np

def main():
    folder=ROOT/'desktop/examples/supplied220';assert not folder.exists(),folder
    app=QApplication([]);doc=document(OUT/'example_input','cpu',bed=None,dt=.02)
    doc['name']='Supplied suspension on an initially non-erodible bed'
    doc['settings'].update(end_time_s=4.,output_interval_s=.5,checkpoint_interval_s=2.)
    doc['editor_state']['checkpoint_clock']=dict(mode='custom',custom_interval_s=2.)
    doc['initial'].update(layer_thickness_m=[0.],concentration_kg_m3=[0.])
    # Initial water is fully wet; inlet supplies 0.4 kg/m3, pure outflow downstream.
    doc=services._resolve_boundary_state(doc)
    f=SettingsPanel();f.load_document(doc);doc=f.document(doc);f.close()
    services.save_project_folder(folder,doc)
    prefix=[sys.executable,'-X','utf8',str(ROOT/'desktop/worker.py'),str(TARGET/'braids_core/river_launcher.py')]
    env=dict(os.environ,OMP_NUM_THREADS='4',PYTHONPATH=str(TARGET)+os.pathsep+str(ROOT))
    (folder/'logs').mkdir()
    for label,args in (
        ('prepare',['prepare','--project',str(folder/'braids_project.json'),'--output',str(folder/'prepared/example'),'--asset-cache',str(folder/'cache')]),
        ('run',['run','--input',str(folder/'prepared/example'),'--output',str(folder/'runs/example'),'--checkpoint-dir',str(folder/'checkpoints/example')])):
        with (folder/'logs'/f'{label}.log').open('x',encoding='utf-8') as stream:
            result=subprocess.run(prefix+args,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=180)
        assert result.returncode==0,(label,folder/'logs'/f'{label}.log')
    services.update_project_session(folder,prepared='prepared/example',output='runs/example',log='logs/run.log')
    field=read_live_frame(folder/'runs/example')[2]
    diag=json.loads((folder/'runs/example/DIAGNOSTICS.json').read_text())
    assert np.max(field['mass'])>0 and np.min(field['mass'])>=0
    budget=max(abs(x) for r in diag for x in r['class_residual_kg']);assert budget<1e-8
    again=services.load_project_folder(folder);assert again['model']==doc['model']
    save(OUT/'EXAMPLE.json',dict(runtime=sha(TARGET/'RUNTIME.json'),project=str(folder),
        seconds=4.,backend='cpu',initial_bed_stock=0.,initial_concentration_kg_m3=0.,
        inlet_concentration_kg_m3=.4,Ds_m=.0001,sediment_residual_kg=budget,
        max_deposited_kg_m2=float(np.max(field['mass'])),folder_roundtrip=True,
        scope='Synthetic numerical demonstration, not experiment'))
    print('Desktop example ready',folder,flush=True)

if __name__=='__main__':main()
