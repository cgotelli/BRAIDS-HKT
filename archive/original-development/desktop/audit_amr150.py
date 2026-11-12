"""Read-only reproduction of the reported dynamic AMR boundary failure."""
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'validation/_amr150'
runtime=ROOT/'desktop/runtime';sys.path.insert(0,str(ROOT));sys.path.insert(0,str(runtime))
from braids_core.river_launcher import verify
verify()
from braids_core.sediment.portable_river_runtime import install
install(json.loads((runtime/'PRECOMPILED.json').read_text()),runtime)
from braids_desktop import services,preparation
from braids_core import river_worker
import braids_core.sediment.river_amr_transfer as remap

source=Path('C:/Users/STREEM/Desktop/0101ttt/River_data/prepared/prepare_20261024_163840_37d6b0b0/PROJECT.json')
raw=source.read_bytes();doc=json.loads(raw)
case=OUT/'baseline';case.mkdir(parents=True,exist_ok=True)
(case/'SOURCE.json').write_bytes(raw)
project=case/'project'
services.save_project_folder(project,doc,source_base=source.parents[2])
prepared=preparation.prepare(project/'braids_project.json',case/'prepared/current',asset_cache=case/'model_layers/preparation_cache')
original=remap.boundary_map
calls=[]
def traced(old,new):
    start=time.perf_counter()
    try:
        result=original(old,new);calls.append(dict(seconds=time.perf_counter()-start,source_faces=old['nf'],target_faces=new['nf'],passed=True));return result
    except ValueError:
        np.savez(case/'FAILED_FACES.npz',**{prefix+k:np.asarray(data[k]) for prefix,data in [('old_',old),('new_',new)] for k in ('nf','neighbour','bid','normal','mid','length')})
        calls.append(dict(seconds=time.perf_counter()-start,source_faces=old['nf'],target_faces=new['nf'],passed=False));raise
remap.boundary_map=traced
try:river_worker.run(prepared,case/'run',until=5.)
except RuntimeError as exc:print('REPRODUCED',exc)
(case/'AUDIT.json').write_text(json.dumps(dict(source=str(source),source_sha256=hashlib.sha256(raw).hexdigest(),calls=calls),indent=2)+'\n')
assert source.read_bytes()==raw
print(calls)
