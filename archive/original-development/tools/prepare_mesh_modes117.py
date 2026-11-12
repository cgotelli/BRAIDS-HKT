"""New v15 qualification/delivery inputs; never edits old user projects."""
from pathlib import Path
from dataclasses import asdict
import json,sys,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from braids_core.river_project import SCHEMA,validate_project
from braids_core.river_model import resolve_model,new_bedload_selection
from braids_core.sediment.river_field_policy import FieldSettings
from braids_core.river_open_boundary import new_state,resolve
from braids_gpu.atomic_output import write_json_atomic

def current(doc,mode):
    doc=json.loads(json.dumps(doc));doc['schema']=SCHEMA
    doc['settings']['adaptive']=mode=='dynamic'
    doc['settings']['adaptation']={**asdict(FieldSettings()),**doc['settings']['adaptation'],
        'terrain_enabled':mode=='static'}
    doc['model']=resolve_model(doc['model']['request']).document()
    validate_project(doc);return doc

def main():
    out=ROOT/'validation/_mesh_modes117/projects';out.mkdir(parents=True,exist_ok=False)
    source=ROOT/'validation/_test1_amr_interval/cpu_installed_1s/prepared'
    original=json.loads((source/'PROJECT.json').read_text())
    original['terrain']['dem_path']=json.loads((source/'core/RECIPE.json').read_text())['dem_path']
    for mode in ('dynamic','static'):
        for backend in ('cpu','gpu'):
            doc=current(original,mode);doc['model']['request']['execution']['backend']=backend
            doc['model']=resolve_model(doc['model']['request']).document()
            write_json_atomic(out/f'Test1_{mode}_{backend}.json',doc)
            for bedload in (False,True):
                d=json.loads(json.dumps(doc));d['settings']['end_time_s']=8.
                if bedload:
                    s=new_bedload_selection();s['grains'][0]['id']=d['settings']['columns']['fraction_ids'][0]
                    s['bed']=dict(composition='bed.layered',transport_layer='transport_layer.constant',coarse_layer=False,sorting='sorting.off')
                    d['model']['request']['sediment']=dict(enabled=True,selection=s)
                    d['model']=resolve_model(d['model']['request']).document();d['settings']['suspension']['bed_factor']=1.
                    d['boundary_states']={n:new_state(r['method'],sediment=True) for n,r in d['settings']['hydraulics'].items()}
                    d['boundary_resolution']=resolve(d['settings']['hydraulics'],d['boundary_states'],
                        {r['bc_id']:r['kind'] for r in d['settings']['assignments']},sediment=d['model']['physics']['sediment'],ng=1,end_time=8.)
                validate_project(d)
                write_json_atomic(out/f'check_{mode}_{backend}_{"bedload" if bedload else "hydro"}.json',d)
    write_json_atomic(out/'INPUTS.json',dict(source=str(source),source_sha256=hashlib.sha256((source/'PROJECT.json').read_bytes()).hexdigest(),
        changes=['Explicit v15 mesh policy, dynamic depth-only or static slope-only','CPU/GPU copies','check files stop at 8 seconds; Test1 copies retain original end time','Bedload checks explicitly enable only one-grain MPM, MF1'],originals_modified=False))

if __name__=='__main__':main()
