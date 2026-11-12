"""Stage area-constrained walls and solid preparation over frozen preview16."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
FILES=('braids_core/boundaries/breaklines.py','braids_core/boundaries/amr.py',
       'braids_core/sediment/adaptive_river_worker.py','braids_core/sediment/river_polygon_banks.py')
def stage():
    target=ROOT/'validation/_mesh152/candidate';base=ROOT/'validation/_solids151/candidate'
    if not target.exists():shutil.copytree(base,target)
    path=target/'RUNTIME.json';data=json.loads(path.read_text())
    for name in FILES:
        shutil.copy2(ROOT/name,target/name)
        data['files'][name]=hashlib.sha256((target/name).read_bytes()).hexdigest()
    path.write_text(json.dumps(data,indent=2)+'\n')
    print(hashlib.sha256(path.read_bytes()).hexdigest())
if __name__=='__main__':stage()
