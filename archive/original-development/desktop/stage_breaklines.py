"""Build the isolated breakline candidate from the retained Desktop release."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
CHANGED=('braids_core/boundaries/breaklines.py','braids_core/boundaries/amr.py',
         'braids_core/boundaries/cut_cells.py','braids_core/boundaries/agglomeration.py',
         'braids_core/sediment/adaptive_river_worker.py','braids_core/river_project.py',
         'braids_core/sediment/river_polygon_banks.py','braids_core/sediment/river_leaf_controller.py')

def stage(target):
    target=Path(target).resolve();base=ROOT/'desktop/runtime'
    if target==base or not target.is_relative_to(ROOT/'validation/_breaklines147'):
        raise ValueError('Stage only to the isolated breaklines147 workspace')
    if not target.exists():shutil.copytree(base,target)
    for name in CHANGED:
        destination=target/name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,destination)
    path=target/'RUNTIME.json';manifest=json.loads(path.read_text(encoding='utf-8'))
    for name in CHANGED:manifest['files'][name]=hashlib.sha256((target/name).read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(target)

if __name__=='__main__':stage(sys.argv[1])
