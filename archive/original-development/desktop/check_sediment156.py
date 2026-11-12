"""Short authenticated Desktop worker checks for the independent process UI.

Creates only fresh validation outputs. No private project or solver source changes.
Run with desktop/.venv/Scripts/python.exe -B desktop/check_sediment156.py --output ...
"""
import argparse
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
RUNTIME=ROOT/'desktop/runtime'
sys.path[:0]=[str(RUNTIME),str(ROOT)]
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFontDatabase
import numpy as np
from braids_desktop.forms import SettingsPanel
from braids_desktop.river_physics import set_choice
from braids_desktop.demo import create_demo
from braids_desktop import services
from braids_core.sediment.adaptive_river_worker import read_live_frame


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--wet',action='store_true',help='Test-only identical prewetted initial fields, not a change to Desktop dry-start defaults')
    args=parser.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    app=QApplication([]);QFontDatabase.addApplicationFont('C:/Windows/Fonts/segoeui.ttf')
    from braids_desktop._avac.theme import apply_theme
    apply_theme(app,'light');results={}
    environment=os.environ.copy();environment.update(PYTHONPATH=str(RUNTIME),PYTHONNOUSERSITE='1',PYTHONIOENCODING='utf-8')
    launcher=RUNTIME/'braids_core/river_launcher.py'
    def worker(name,action,arguments):
        with (root/(name+'_'+action+'.log')).open('w',encoding='utf-8') as stream:
            subprocess.run([sys.executable,'-B',str(ROOT/'desktop/worker.py'),str(launcher),action,*map(str,arguments)],
                cwd=RUNTIME,env=environment,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=300)
    modes=('bedload','suspended','combined') if args.wet else ('bedload','suspended','combined','collapse','combined_banks')
    for mode in modes:
        for backend in ('cpu','gpu'):
            name=mode+'_'+backend;folder=root/name
            create_demo(folder,backend,end_time_s=.1)
            original=services.load_project_folder(folder);f=SettingsPanel();f.load_document(original)
            f.sediment_enabled.setChecked(True);p=f.physics
            p.bedload_on.setChecked(mode in ('bedload','combined','combined_banks'))
            p.suspension_on.setChecked(mode in ('suspended','combined','combined_banks'))
            p.avalanche_on.setChecked(mode in ('collapse','combined_banks'))
            p.banks_on.setChecked(mode=='combined_banks')
            set_choice(p.capacity,'capacity.general');f.cfl.setValue(.5);f.dt_max.setValue(.01)
            f.output_interval.setValue(.05);f.checkpoint_interval.setValue(.1)
            if p.suspension_on.isChecked():
                set_choice(f.forcing.boundary,'Upstream');f.forcing.table.item(0,3).setText('.2')
                set_choice(f.forcing.boundary,'Downstream');f.forcing.table.item(0,3).setText('.2')
            doc=f.document()
            if args.wet:doc['initial'].update(stage_m=.5,velocity_m_s=[1.,0.])
            services.save_project_folder(folder,doc)
            f.resize(420,850);f.show();f.set_step('Sediment transport')
            if name=='combined_cpu':
                for step in ('Sediment transport','Morpho evolution','Sediment transport BC'):
                    f.set_step(step);app.processEvents();app.processEvents()
                    f.grab().save(str(root/(step.replace(' ','_')+'.png')))
            f.close();f.deleteLater();app.processEvents()
            started=time.perf_counter()
            prepared=folder/'check_prepared';output=folder/'check_run'
            worker(name,'prepare',['--project',folder/'braids_project.json','--output',prepared,'--asset-cache',root/'asset_cache'])
            worker(name,'run',['--input',prepared,'--output',output])
            row,mesh,values=read_live_frame(output)
            diagnostics=json.loads((output/'DIAGNOSTICS.json').read_text())
            assert abs(row['time_s']-.1)<1e-12 and np.isfinite(values['q']).all()
            assert np.min(values['q'][0])>=0
            assert all(abs(d['water_residual_m3'])<=d['water_roundoff_limit_m3'] for d in diagnostics)
            assert all(max(abs(v) for v in d['class_residual_kg'])<=d['class_roundoff_limit_kg'] for d in diagnostics)
            assert all(not any(d['wall_integrals']) for d in diagnostics)
            with np.load(output/'internal/checkpoint.npz') as outer:
                with np.load(BytesIO(outer['payload'].tobytes())) as adaptive:
                    with np.load(BytesIO(adaptive['owner'].tobytes())) as column:
                        with np.load(BytesIO(column['river'].tobytes())) as river:
                            bed_flux_max=float(np.max(np.abs(river['class_flux'])))
                            bed_boundary_max=float(np.max(np.abs(river['class_boundary_ledger'])))
            if not doc['settings']['suspension']['bed_factor']:
                assert bed_flux_max==bed_boundary_max==0.
            results[name]=dict(time_s=row['time_s'],steps=row['steps'],cells=len(mesh['leaves']),
                wall_seconds=time.perf_counter()-started,bed_factor=doc['settings']['suspension']['bed_factor'],
                bed_boundary_max=bed_boundary_max,bed_flux_max=bed_flux_max,
                max_suspended_kg_m2=float(np.max(values['suspended'])),
                water_residual_max_m3=max(abs(d['water_residual_m3']) for d in diagnostics),
                class_residual_max_kg=max(max(abs(x) for x in d['class_residual_kg']) for d in diagnostics))
            print(name,results[name],flush=True)
    comparisons={}
    for mode in modes:
        a=read_live_frame(root/(mode+'_cpu')/'check_run');b=read_live_frame(root/(mode+'_gpu')/'check_run')
        np.testing.assert_array_equal(a[1]['leaves'],b[1]['leaves'])
        comparisons[mode]={key:float(np.max(np.abs(a[2][key]-b[2][key]))) for key in ('q','z','mass','suspended')}
        assert comparisons[mode]['z']<1e-6 and comparisons[mode]['q']<1e-6
    report=dict(runtime_sha256=hashlib.sha256((RUNTIME/'RUNTIME.json').read_bytes()).hexdigest(),
        cases=results,backend_max_abs=comparisons,
        scope='0.1 s '+('prewetted mobile-channel' if args.wet else 'cold dry-start')+' configuration/execution smoke, not physical validation')
    (root/'CHECK.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':main()
