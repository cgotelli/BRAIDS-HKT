"""Stage only the merge repair over the authenticated preview13 runtime."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
base=ROOT/'desktop/runtime';target=ROOT/'validation/_merge149/candidate'
if not target.exists():shutil.copytree(base,target)
name='braids_core/boundaries/breaklines.py';shutil.copy2(ROOT/name,target/name)
path=target/'RUNTIME.json';data=json.loads(path.read_text(encoding='utf-8'))
data['files'][name]=hashlib.sha256((target/name).read_bytes()).hexdigest()
path.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
print(hashlib.sha256(path.read_bytes()).hexdigest())
