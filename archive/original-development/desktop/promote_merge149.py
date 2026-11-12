"""Promote the qualified one-file runtime repair with a retained prior source."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];base=ROOT/'desktop/runtime';candidate=ROOT/'validation/_merge149/candidate'
backup=ROOT/'validation/_merge149/preview13_source';name='braids_core/boundaries/breaklines.py'
old='f59ec400c94ae2d05afa1b5201a5d5e0333479f59e1e0c88d7708fcad66f1fed'
new='588c8124dace5afcc3b695cff5d0af03d0e3884393cc2b8e16eec82a715e98cf'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
assert sha(base/'RUNTIME.json')==old and sha(candidate/'RUNTIME.json')==new
a=json.loads((base/'RUNTIME.json').read_text());b=json.loads((candidate/'RUNTIME.json').read_text())
assert set(a['files'])==set(b['files'])
assert {k for k in a['files'] if a['files'][k]!=b['files'][k]}=={name}
for root,manifest in ((base,a),(candidate,b)):
    for k,digest in manifest['files'].items():assert sha(root/k)==digest,k
assert sha(base/'PRECOMPILED.json')==sha(candidate/'PRECOMPILED.json')
for item in (name,'RUNTIME.json'):
    target=backup/item;target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():assert sha(target)==sha(base/item)
    else:shutil.copy2(base/item,target)
shutil.copy2(candidate/name,base/name);shutil.copy2(candidate/'RUNTIME.json',base/'RUNTIME.json')
for k,digest in b['files'].items():assert sha(base/k)==digest,k
receipt=dict(previous_runtime=old,installed_runtime=new,changed=[name],native_libraries_unchanged=True,backup=str(backup))
(ROOT/'desktop/evidence/merge149/DELIVERY.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt,indent=2))
