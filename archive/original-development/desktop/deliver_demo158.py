"""Retain complete portable inputs, results and selected checkpoints per backend."""
import json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'desktop/runtime'),str(ROOT)]
from braids_desktop import services,project_file
from braids_core import river_folder
from braids_core.river_project import SCHEMA
destination=Path.home()/'Desktop/BRAIDS bare-bed groin';destination.mkdir(exist_ok=True)
for backend in ('cpu','gpu'):
    source=ROOT/'validation/_sediment158'/(backend+'_final')/'project'
    folder=destination/backend.upper()/'Groin_data';folder.parent.mkdir(exist_ok=True)
    shutil.copytree(source,folder)
    doc=services.load_project_folder(folder)
    river_folder.update_session(folder,dict(output='runs/resumed',prepared='prepared/base',
        desktop=dict(live_results=False,model_step='Sediment transport',result_step='Profiles')))
    project_file.write(folder.parent/'Groin.braids.json',folder,doc)
for name in ('BARE_BED_EVOLUTION.png','CPU_GPU.png','SEDIMENT_BUDGET.png','COMPARISON.json'):
    shutil.copy2(ROOT/'validation/sediment/river_morphodynamics/reports/desktop_sediment158'/name,destination/name)
# Explicitly reissue the bundled demo; never migrate or overwrite user projects.
old=ROOT/'desktop/examples/river_demo157';new=ROOT/'desktop/examples/river_demo158'
raw=json.loads((old/'braids_project.json').read_text());raw.pop('editor_state',None)
assert raw['model']['physics']['sediment'] is None
raw['schema']=SCHEMA;raw['material_zones']={'roughness':[],'erodible':[]}
raw['terrain']['dem_path']=str(old/raw['terrain']['dem_path'])
services.save_project_folder(new,raw);project_file.write(project_file.default_file(new),new,services.load_project_folder(new))
print(destination)
