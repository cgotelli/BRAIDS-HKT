"""Instrument the frozen preview24 worker without editing its runtime."""
import os,sys,json,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUNTIME=Path(os.environ.get('BRAIDS_AUDIT_RUNTIME',ROOT/'desktop/runtime'))
sys.path[:0]=[str(RUNTIME),str(ROOT)]
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from braids_core.sediment import adaptive_river_worker as worker
from braids_core.sediment.bed_columns_v101 import host
import numpy as np
from braids_core.sediment.portable_river_runtime import install
install(json.loads((RUNTIME/'PRECOMPILED.json').read_text()),RUNTIME)
from braids_core.river_worker import run
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
p.add_argument('--backend',choices=['cpu','gpu'],default='cpu');p.add_argument('--formula',choices=['mpm','ashida_michiue_generalized','wang_fredsoe'],default='mpm')
p.add_argument('--end',type=float,default=220.);p.add_argument('--stock',type=float,default=0.);args=p.parse_args()
original=worker.fields;rows=[]
def fields(a):
    result=original(a)
    if 190<a.time<221:
        rows.append(dict(time=a.time,dt=a.physics._dt,maximum=float(np.linalg.norm(result['bedload_vector_m2_s'],axis=0).max())))
        print(rows[-1],flush=True)
    return result
worker.fields=fields
from PySide6.QtWidgets import QApplication
from check_boundaries159 import document,services
app=QApplication.instance() or QApplication([])
doc=document(args.output/'project',args.backend,'bedload_normal_critical')
from braids_desktop.forms import SettingsPanel
f=SettingsPanel();f.load_document(doc);f.physics.enabled.setChecked(True);f.physics.bedload_on.setChecked(True)
from braids_desktop.river_physics import set_choice
set_choice(f.physics.capacity,'capacity.'+args.formula)
f.end_time.setValue(args.end)
f.physics.layers.item(0,0).setText(str(args.stock))
f.output_interval.setValue(5.)
doc=f.document()
services.save_project_folder(args.output/'project',doc)
from braids_desktop.preparation import prepare
prepared=args.output/'prepared';prepare(args.output/'project/braids_project.json',prepared,asset_cache=args.output/'assets')
status=run(prepared,args.output/'run',until=args.end)
args.output.joinpath('CLOCK.json').write_text(json.dumps(rows,indent=2))
args.output.joinpath('CHECK.json').write_text(json.dumps(dict(status=status,formula=args.formula,backend=args.backend),indent=2))
