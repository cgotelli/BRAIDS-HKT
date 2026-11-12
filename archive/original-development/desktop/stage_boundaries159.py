"""Stage the boundary-choice consolidation; numerical kernels are unchanged."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
base=ROOT/'desktop/runtime';target=ROOT/'validation/_boundaries159/candidate'
if target.exists():raise FileExistsError(target)
shutil.copytree(base,target)
manifest=json.loads((target/'RUNTIME.json').read_text())
for name in ('braids_core/river_open_boundary.py',):
    shutil.copy2(ROOT/name,target/name)
    manifest['files'][name]=hashlib.sha256((target/name).read_bytes()).hexdigest()
(target/'RUNTIME.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(hashlib.sha256((target/'RUNTIME.json').read_bytes()).hexdigest())
