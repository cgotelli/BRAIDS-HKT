"""Explicit Test1 copies for a slower AMR clock; no user project mutation."""
import json,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from braids_core.river_model import resolve_model
from braids_core.river_project import validate_project
from braids_gpu.atomic_output import write_json_atomic

def main():
    out=ROOT/'validation/_test1_amr_interval/release_test1';out.mkdir(exist_ok=False)
    source=ROOT/'validation/_test1_amr_optimization/release_test1/Test1_5m_alpha112_cpu.json'
    raw=source.read_bytes()
    for backend in ('cpu','gpu'):
        doc=json.loads(raw);doc['settings']['adaptation']['interval']=1.
        doc['model']['request']['execution']['backend']=backend
        doc['model']=resolve_model(doc['model']['request']).document();validate_project(doc)
        write_json_atomic(out/f'Test1_5m_alpha113_{backend}.json',doc)
        doc['settings']['end_time_s']=8.;validate_project(doc)
        write_json_atomic(out/f'check_{backend}.json',doc)
    write_json_atomic(out/'CHANGES.json',dict(source=str(source),source_sha256=hashlib.sha256(raw).hexdigest(),
        changes=['AMR assessment interval 0.05 to 1 second','Explicit CPU and GPU copies',
                 'check_* files stop at 8 seconds; Test1_* retains 600 seconds'],original_user_files_modified=False))
if __name__=='__main__':main()
