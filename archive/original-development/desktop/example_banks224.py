"""Portable matched bank controls through actual Desktop worker subprocesses."""
import os,sys,json,subprocess
from stage_banks224 import ROOT,OUT,TARGET,BASE,REPORT,sha,save
RUNTIME=BASE if '--installed' in sys.argv else TARGET
sys.path[:0]=[str(RUNTIME),str(ROOT),str(ROOT/'tests')]
os.environ['QT_QPA_PLATFORM']='offscreen'
from PySide6.QtWidgets import QApplication
from test_banks224 import document
from braids_core.sediment.bank_setup import METHODS
from braids_core.sediment.collapse_setup import JC,ORDINARY
from braids_desktop import services
from braids_desktop.forms import SettingsPanel
from braids_core.sediment.adaptive_river_worker import read_live_frame
import numpy as np

CASES=[('off','cpu',25.,'off','off'),('sharing','cpu',25.,METHODS[0],'off'),
    ('toe','cpu',25.,METHODS[1],'off'),('block','cpu',25.,METHODS[2],'off'),
    ('block_equalization','cpu',25.,METHODS[2],ORDINARY),
    ('block_repose','cpu',25.,METHODS[2],JC),
    ('gpu_block_repose','gpu',25.,METHODS[2],JC),
    ('mf1_block_repose','cpu',1.,METHODS[2],JC)]

def main():
    app=QApplication([]);folder=ROOT/'desktop/examples/banks224';rows=[]
    installed='--installed' in sys.argv
    prefix=[sys.executable,'-X','utf8',str(ROOT/'desktop/worker.py'),str(RUNTIME/'braids_core/river_launcher.py')]
    env=dict(os.environ,OMP_NUM_THREADS='4',PYTHONPATH=str(RUNTIME)+os.pathsep+str(ROOT))
    for name,backend,mf,method,collapse in CASES:
        case=folder/name
        def invoke(label,args):
            path=case/'logs'/(('installed_' if installed else '')+label+'.log')
            with path.open('x',encoding='utf-8') as stream:
                result=subprocess.run(prefix+args,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=300)
            assert result.returncode==0,(label,path)
        if not installed:
            assert not case.exists()
            d=document(OUT/'example_inputs'/name,backend,'capacity.mpm','suspended.garcia_parker_1991',
                method=method,collapse=collapse,mf=mf,case='cropped',dt=.002,end=.4)
            # Small laboratory-scale numerical demonstration, not a calibrated
            # bank-strength parameter. The unit tests use 1e-8 to stress history.
            if method==METHODS[2]:
                panel=SettingsPanel();panel.load_document(d);panel.physics.bank_critical.setValue(.0001)
                d=panel.document(d);panel.close()
            d['name']='Finite bank comparison: '+name
            services.save_project_folder(case,d);(case/'logs').mkdir()
            invoke('prepare',['prepare','--project',str(case/'braids_project.json'),'--output',str(case/'prepared/example'),'--asset-cache',str(case/'cache')])
            invoke('run',['run','--input',str(case/'prepared/example'),'--output',str(case/'runs/example'),'--checkpoint-dir',str(case/'checkpoints/example')])
            services.update_project_session(case,prepared='prepared/example',output='runs/example',log='logs/run.log')
        checkpoint=next((case/'checkpoints/example').glob('t_0.2_*.npz'))
        restart=case/'runs'/('installed_restart' if installed else 'restart')
        invoke('restart',['run','--input',str(case/'prepared/example'),'--output',str(restart),'--restart',str(checkpoint)])
        f=read_live_frame(case/'runs/example')[2];r=read_live_frame(restart)[2]
        for k in f:np.testing.assert_array_equal(f[k],r[k])
        diag=json.loads((case/'runs/example/DIAGNOSTICS.json').read_text())
        rows.append(dict(case=name,backend=backend,mf=mf,method=method,collapse=collapse,exact_restart=True,
            mass_residual=max(abs(x) for d in diag for x in d['class_residual_kg']),
            water_residual=max(abs(d['water_residual_m3']) for d in diag),
            bank_gross_kg=float(f['bank_gross'].sum()),collapse_gross_kg=float(f['collapse_gross'].sum()),
            block_count=int(f.get('bank_failure_count',np.zeros(1)).sum())))
        print(name+' passed',flush=True)
    save(OUT/('INSTALLED_LAUNCH.json' if installed else 'EXAMPLE.json'),dict(runtime=sha(RUNTIME/'RUNTIME.json'),cases=rows))

if __name__=='__main__':main()
