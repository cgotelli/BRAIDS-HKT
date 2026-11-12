"""Regression of the actual QGIS 0.05 s output clock on current CPU/GPU."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.build_unified_qgis import record
from braids_core import river_worker
from braids_core.river_project import validate_project
from braids_gpu.atomic_output import write_json_atomic

def main():
    base=ROOT/'validation/_mesh_modes117';out=base/'clock_projects';out.mkdir(exist_ok=False);paths=[]
    for source in sorted((base/'projects').glob('check_*.json')):
        d=json.loads(source.read_text());d['settings'].update(end_time_s=.1,output_interval_s=.05)
        validate_project(d);target=out/source.name;write_json_atomic(target,d);paths.append(target)
    record(base/'clock_record',paths)
    for path in paths:
        if '_gpu_' not in path.stem:continue
        root=base/'clock_gpu'/path.stem
        prep=river_worker.prepare(path,root/'prepared');result=river_worker.run(prep,root/'run')
        assert result['status']=='complete';print('Clock GPU passed',path.name,flush=True)

if __name__=='__main__':main()
