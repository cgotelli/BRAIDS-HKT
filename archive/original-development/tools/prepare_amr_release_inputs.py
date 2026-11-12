"""Explicit audit copies of alpha111 release decks for the new AMR schema.

Never invoked by the project loader. Old user projects remain rejected; only
validation inputs are copied, with every schema change recorded.
"""
import json,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from braids_core.river_project import SCHEMA,validate_project
from braids_gpu.atomic_output import write_json_atomic

def main():
    out=ROOT/'validation/_test1_amr_optimization/release_inputs';out.mkdir(exist_ok=False)
    old=json.loads((ROOT/'validation/_test1_gpu_fix/release_record/RECORD.json').read_text())
    records=[]
    for i,row in enumerate(old['projects']):
        source=ROOT/row['project'];raw=source.read_bytes()
        assert hashlib.sha256(raw).hexdigest()==row['sha256']
        doc=json.loads(raw);before=doc['settings']['adaptation'].copy()
        doc['schema']=SCHEMA
        doc['settings']['adaptation']={k:before[k] for k in ('interval','coarsen_delay','buffer_cells')}
        dem=Path(doc['terrain']['dem_path'])
        if not dem.is_absolute():doc['terrain']['dem_path']=str((source.parent/dem).resolve())
        validate_project(doc)
        dest=out/f'{i:03d}_{source.name}';write_json_atomic(dest,doc)
        records.append(dict(source=str(source),before_sha256=row['sha256'],destination=str(dest),
            original_adaptation=before,current_adaptation=doc['settings']['adaptation'],physics_changed=False))
    write_json_atomic(out/'CHANGES.json',dict(scope='validation-only input copies; not project migration',records=records))
    print(len(records),'release inputs copied and validated')
if __name__=='__main__':main()
