"""Short portable bank example through actual Desktop subprocesses."""
import os,sys,json,subprocess
from stage_repose223 import ROOT,OUT,TARGET,BASE,REPORT,sha,save
RUNTIME=BASE if '--installed' in sys.argv else TARGET
sys.path[:0]=[str(RUNTIME),str(ROOT),str(ROOT/'tests')]
os.environ['QT_QPA_PLATFORM']='offscreen'
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFontDatabase
from test_repose223 import document
from braids_desktop.river_physics import set_choice
from braids_core.sediment.collapse_setup import ORDINARY
from braids_desktop import services
from braids_desktop.forms import SettingsPanel
from braids_desktop._avac.theme import apply_theme
from braids_core.sediment.adaptive_river_worker import read_live_frame
import numpy as np

def main():
    app=QApplication([]);QFontDatabase.addApplicationFont('C:/Windows/Fonts/segoeui.ttf');apply_theme(app,'light')
    folder=ROOT/'desktop/examples/repose223';rows=[];installed='--installed' in sys.argv
    prefix=[sys.executable,'-X','utf8',str(ROOT/'desktop/worker.py'),str(RUNTIME/'braids_core/river_launcher.py')]
    env=dict(os.environ,OMP_NUM_THREADS='4',PYTHONPATH=str(RUNTIME)+os.pathsep+str(ROOT))
    for backend,mf,on in [('cpu',25,False),('cpu',25,'equalization'),('cpu',1,True),('cpu',25,True),('gpu',25,True)]:
        name=f'{backend}_MF{mf}_'+('equalization' if on=='equalization' else 'repose' if on else 'no_collapse');case=folder/name
        def invoke(label,args):
            path=case/'logs'/(('installed_' if installed else '')+label+'.log')
            with path.open('x',encoding='utf-8') as stream:
                result=subprocess.run(prefix+args,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=300)
            assert result.returncode==0,(label,path)
        if not installed:
            assert not case.exists()
            d=document(OUT/'example_inputs'/name,backend,mf=mf,case='cropped',dt=.002,end=.4,collapse=bool(on))
            if on!='equalization' and not on:
                panel=SettingsPanel();panel.load_document(d);panel.physics.avalanche_on.setChecked(False);d=panel.document(d);panel.close()
            if on=='equalization':
                panel=SettingsPanel();panel.load_document(d);set_choice(panel.physics.avalanche,ORDINARY);panel.physics.collapse_slope.setValue(float(np.tan(np.deg2rad(20.))));d=panel.document(d);panel.close()
            d['name']=f'Finite bank: MPM + Garcia-Parker, {backend.upper()}, MF {mf}, collapse {on}'
            services.save_project_folder(case,d);(case/'logs').mkdir()
            f=SettingsPanel();f.resize(410,950);f.load_document(d);f.show()
            if backend=='cpu' and mf==25 and on:
                REPORT.mkdir(parents=True,exist_ok=True)
                for step,name_ui in [('Morpho evolution','UI_COLLAPSE'),('Sediment transport','UI_RESOLVED')]:
                    f.set_step(f.step_names.index(step))
                    if step=='Sediment transport':f.sediment_tabs.setCurrentIndex(3)
                    for _ in range(6):app.processEvents()
                    assert f.grab().save(str(REPORT/(name_ui+'.png')))
                    assert f.stack.currentWidget().horizontalScrollBar().maximum()==0
            f.close()
            invoke('prepare',['prepare','--project',str(case/'braids_project.json'),'--output',str(case/'prepared/example'),'--asset-cache',str(case/'cache')])
            invoke('run',['run','--input',str(case/'prepared/example'),'--output',str(case/'runs/example'),'--checkpoint-dir',str(case/'checkpoints/example')])
            services.update_project_session(case,prepared='prepared/example',output='runs/example',log='logs/run.log')
        checkpoint=next((case/'checkpoints/example').glob('t_0.2_*.npz'))
        restart=case/'runs'/('installed_restart' if installed else 'restart')
        invoke('restart',['run','--input',str(case/'prepared/example'),'--output',str(restart),'--restart',str(checkpoint)])
        f=read_live_frame(case/'runs/example')[2];r=read_live_frame(restart)[2]
        for k in f:np.testing.assert_array_equal(f[k],r[k])
        diag=json.loads((case/'runs/example/DIAGNOSTICS.json').read_text())
        rows.append(dict(case=name,backend=backend,mf=mf,collapse=on,exact_restart=True,
            mass_residual=max(abs(x) for d in diag for x in d['class_residual_kg']),
            water_residual=max(abs(d['water_residual_m3']) for d in diag),
            gross_collapse_kg=float(f['collapse_gross'].sum())))
    save(OUT/('INSTALLED_LAUNCH.json' if installed else 'EXAMPLE.json'),dict(runtime=sha(RUNTIME/'RUNTIME.json'),cases=rows))
    print('Portable bank examples and child-process restarts complete',flush=True)
if __name__=='__main__':main()
