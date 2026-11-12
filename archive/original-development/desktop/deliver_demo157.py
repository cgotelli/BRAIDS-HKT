"""Create a fresh self-contained groin project and a current natural-river demo."""
import json,shutil,sys
from copy import deepcopy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'desktop/runtime'),str(ROOT)]
from braids_desktop import services,project_file,workspace
from braids_core import river_folder
from braids_core.river_project import SCHEMA

source=ROOT/'validation/_bedload157/final_cpu'
destination=Path.home()/'Desktop/BRAIDS MPM groin';destination.mkdir(exist_ok=False)
root=destination/'Groin_data';doc=services.load_project_folder(source/'project')
saved=services.save_project_folder(root,doc)
for backend,name in [('cpu','final_cpu'),('gpu','final_gpu')]:
    run=root/'runs'/backend
    shutil.copytree(ROOT/'validation/_bedload157'/name/'run',run,ignore=shutil.ignore_patterns('runtime'))
    shutil.copy2(ROOT/'validation/_bedload157'/name/'run.log',run/'execution.log')
    river_folder.record_event(root,'qualified_demo',output=f'runs/{backend}',backend=backend,status='complete',
                              reference=str(ROOT/'validation/_bedload157'/name))
river_folder.update_session(root,dict(output='runs/cpu',desktop=dict(live_results=False,model_step='Sediment transport',result_step='Probes')))
project_file.write(destination/'Groin.braids.json',root,saved)
figures=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_bedload157'
for name in ('BED_EVOLUTION.png','CPU_GPU_CONTROL.png','HYDRAULICS.png','COMPARISON.json'):
    shutil.copy2(figures/name,destination/name)
# Reissue only the bundled demonstration (not user projects) under the new
# explicit schema. Preserve the former package as a frozen versioned template.
old=ROOT/'desktop/examples/river_demo';new=ROOT/'desktop/examples/river_demo157'
raw=json.loads((old/'braids_project.json').read_text());raw.pop('editor_state',None)
assert raw['model']['physics']['sediment'] is None
raw['schema']=SCHEMA;raw['settings']['sediment_start_time_s']=0.
raw['terrain']['dem_path']=str(old/raw['terrain']['dem_path'])
services.save_project_folder(new,raw)
project_file.write(project_file.default_file(new),new,services.load_project_folder(new))
print(destination/'Groin.braids.json')
