"""New explicitly documented Test1 validation projects, leaving the user file intact."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from braids_core.river_model import resolve_model
from braids_core.river_project import SCHEMA,validate_project
from braids_gpu.atomic_output import write_json_atomic

out=ROOT/'validation/_test1_amr_optimization/release_test1';out.mkdir(exist_ok=False)
source=Path('C:/Users/STREEM/Desktop/Test1/runs/prepare_20261021_204137_af80f2')
for backend in ('cpu','gpu'):
    doc=json.loads((source/'PROJECT.json').read_text());before=doc['settings']['adaptation']
    doc['schema']=SCHEMA;doc['settings']['adaptation']={k:before[k] for k in ('interval','coarsen_delay','buffer_cells')}
    doc['terrain']['dem_path']=json.loads((source/'core/RECIPE.json').read_text())['dem_path']
    doc['model']['request']['execution']['backend']=backend;doc['model']=resolve_model(doc['model']['request']).document()
    validate_project(doc)
    write_json_atomic(out/f'Test1_5m_alpha112_{backend}.json',doc)
    doc['settings']['end_time_s']=8.
    validate_project(doc);write_json_atomic(out/f'check_{backend}.json',doc)
write_json_atomic(out/'CHANGES.json',dict(source=str(source),changes=['Project schema v13',
    'Depth-only AMR: retained interval, delay and buffer', 'Explicit CPU/GPU copies',
    'check_* validation files end at 8 seconds; Test1_* files retain original duration'],
    original_user_inputs_modified=False))
