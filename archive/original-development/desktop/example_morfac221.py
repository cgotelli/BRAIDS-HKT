"""Portable short MF comparison; actual Desktop child process and restart."""
import os,sys,json,subprocess
from stage_morfac221 import ROOT,OUT,TARGET,BASE,REPORT,sha,save
RUNTIME=BASE if '--installed' in sys.argv else TARGET
sys.path[:0]=[str(RUNTIME),str(ROOT),str(ROOT/'tests')]
os.environ['QT_QPA_PLATFORM']='offscreen'
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFontDatabase
from test_morfac221 import doc
from braids_desktop import services
from braids_desktop.forms import SettingsPanel
from braids_desktop._avac.theme import apply_theme
from braids_core.sediment.adaptive_river_worker import read_live_frame
import numpy as np

def main():
    app=QApplication([]);QFontDatabase.addApplicationFont('C:/Windows/Fonts/segoeui.ttf');apply_theme(app,'light')
    folder=ROOT/'desktop/examples/morfac221';rows=[]
    prefix=[sys.executable,'-X','utf8',str(ROOT/'desktop/worker.py'),str(RUNTIME/'braids_core/river_launcher.py')]
    env=dict(os.environ,OMP_NUM_THREADS='4',PYTHONPATH=str(RUNTIME)+os.pathsep+str(ROOT))
    installed='--installed' in sys.argv
    for backend,mf in [('cpu',1),('cpu',5),('cpu',25),('gpu',25)]:
        case=folder/f'{backend}_MF{mf}'
        def invoke(label,args):
            path=case/'logs'/(('installed_' if installed else '')+label+'.log')
            with path.open('x',encoding='utf-8') as stream:
                result=subprocess.run(prefix+args,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=300)
            assert result.returncode==0,(label,path)
        if not installed and not case.exists():
            d=doc(OUT/'example_inputs'/f'{backend}_{mf}',backend,'suspended.garcia_parker_1991',bed='capacity.mpm',mf=mf,dt=.02)
            d['name']=f'MF {mf}: MPM + Garcia-Parker, {backend.upper()}'
            d['settings'].update(end_time_s=4.,output_interval_s=.5,checkpoint_interval_s=2.)
            d['editor_state']['checkpoint_clock']=dict(mode='custom',custom_interval_s=2.)
            d['initial'].update(layer_thickness_m=[0.],concentration_kg_m3=[0.])
            d=services._resolve_boundary_state(d)
            f=SettingsPanel();f.resize(390,900);f.load_document(d);d=f.document(d)
            f.set_step(f.step_names.index('Sediment transport'));f.show()
            if backend=='cpu' and mf==25:
                for i in (0,3):
                    f.sediment_tabs.setCurrentIndex(i)
                    for _ in range(6):app.processEvents()
                    REPORT.mkdir(parents=True,exist_ok=True)
                    assert f.grab().save(str(REPORT/f'UI_{i}.png'))
                    assert f.stack.currentWidget().horizontalScrollBar().maximum()==0
            assert 'MF = '+str(float(mf)) in f.method_summary();f.close()
            services.save_project_folder(case,d);(case/'logs').mkdir()
            invoke('prepare',['prepare','--project',str(case/'braids_project.json'),'--output',str(case/'prepared/example'),'--asset-cache',str(case/'cache')])
            invoke('run',['run','--input',str(case/'prepared/example'),'--output',str(case/'runs/example'),'--checkpoint-dir',str(case/'checkpoints/example')])
            services.update_project_session(case,prepared='prepared/example',output='runs/example',log='logs/run.log')
            again=services.load_project_folder(case);assert again['settings']==d['settings']
        checkpoint=next((case/'checkpoints/example').glob('t_2_*.npz'))
        restart=case/'runs'/('installed_restart' if installed else 'restart')
        if not restart.exists():
            invoke('restart',['run','--input',str(case/'prepared/example'),'--output',str(restart),'--restart',str(checkpoint)])
        else:
            assert json.loads((restart/'status.json').read_text())['status']=='complete'
            assert services.load_project_folder(case)['settings']['controls']['morfac']==mf
        field=read_live_frame(case/'runs/example')[2];resumed=read_live_frame(restart)[2]
        for k in field:np.testing.assert_array_equal(field[k],resumed[k])
        diag=json.loads((case/'runs/example/DIAGNOSTICS.json').read_text())
        budget=max(abs(x) for r in diag for x in r['class_residual_kg']);assert budget<1e-8
        assert np.max(field['mass'])>0 and np.min(field['mass'])>=0
        rows.append(dict(backend=backend,mf=mf,hydraulic_seconds=4.,bed_clock_seconds=4*mf,
            sediment_residual_kg=budget,exact_restart=True,folder=str(case),
            acceleration_kg=float(field['acceleration'].sum()),max_bed_mass_kg_m2=float(field['mass'].max())))
    save(OUT/('INSTALLED_LAUNCH.json' if installed else 'EXAMPLE.json'),dict(runtime=sha(RUNTIME/'RUNTIME.json'),cases=rows,
        scope='Synthetic supplied sand deposition with active MPM and GP; initially zero erodible stock. Not experimental validation or proof of exact time compression.'))
    print('MF projects and child-process restart checks complete',flush=True)
if __name__=='__main__':main()
