"""Stage warm-up on the frozen Desktop runtime; native libraries unchanged."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'validation/_bedload157'
FILES=('braids_core/river_project.py','braids_core/sediment/adaptive_river_worker.py')

def stage():
    base=ROOT/'desktop/runtime';target=OUT/'candidate'
    if not target.exists():shutil.copytree(base,target)
    path=target/'RUNTIME.json';data=json.loads(path.read_text(encoding='utf-8'))
    for name in FILES:
        shutil.copy2(ROOT/name,target/name)
        data['files'][name]=hashlib.sha256((target/name).read_bytes()).hexdigest()
    path.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
    print(hashlib.sha256(path.read_bytes()).hexdigest())

if __name__=='__main__':stage()
