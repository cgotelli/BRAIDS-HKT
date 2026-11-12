"""Re-record all prior release bindings on explicit current mesh-policy inputs."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.prepare_mesh_modes117 import current
from tools.build_unified_qgis import record
from braids_gpu.atomic_output import write_json_atomic

def main():
    base=ROOT/'validation/_mesh_modes117';out=base/'release_inputs';out.mkdir(exist_ok=False)
    previous=ROOT/'validation/_test1_amr_interval/release_record/RECORD.json'
    docs=json.loads(previous.read_text());paths=[];changes=[]
    for i,row in enumerate(docs['projects']):
        src=Path(row['project']);doc=json.loads(src.read_text())
        mode='dynamic' if doc['settings']['adaptive'] else 'static'
        doc=current(doc,mode);target=out/f'cpu_{i:03d}_{src.stem}.json'
        write_json_atomic(target,doc);paths.append(target)
        changes.append(dict(source=str(src),copy=str(target),mode=mode,changes=['v15 schema','Explicit exclusive mesh criteria; physical inputs unchanged']))
    for mode in ('dynamic','static'):
        for physics in ('hydro','bedload'):
            paths.append(base/'projects'/f'check_{mode}_cpu_{physics}.json')
    write_json_atomic(out/'CHANGES.json',changes)
    record(base/'release_record',paths)

if __name__=='__main__':main()
