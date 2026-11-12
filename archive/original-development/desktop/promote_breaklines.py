"""Install the qualified local breakline candidate, preserving prior sources."""
from pathlib import Path
import hashlib
import json
import shutil
from stage_breaklines import CHANGED

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'desktop/runtime'
CANDIDATE=ROOT/'validation/_breaklines147/candidate'
BACKUP=ROOT/'validation/_breaklines147/preview11_runtime_sources'
OLD='741c15534703fcbd6207d845af2811681a8f9f96da05f8724d012da686e2ccac'
NEW='f59ec400c94ae2d05afa1b5201a5d5e0333479f59e1e0c88d7708fcad66f1fed'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

assert sha(BASE/'RUNTIME.json')==OLD,'Installed runtime changed; do not overwrite it'
assert sha(CANDIDATE/'RUNTIME.json')==NEW,'Candidate changed after qualification'
old=json.loads((BASE/'RUNTIME.json').read_text());new=json.loads((CANDIDATE/'RUNTIME.json').read_text())
assert set(new['files'])-set(old['files'])=={'braids_core/boundaries/breaklines.py'}
assert not set(old['files'])-set(new['files'])
changed={name for name in new['files'] if new['files'][name]!=old['files'].get(name)}
assert changed==set(CHANGED)
for name,checksum in old['files'].items():assert sha(BASE/name)==checksum,name
for name,checksum in new['files'].items():assert sha(CANDIDATE/name)==checksum,name
assert sha(BASE/'PRECOMPILED.json')==sha(CANDIDATE/'PRECOMPILED.json')
# Back up only superseded maintained Python sources and the manifest. Native
# binaries, frozen reference runs and user projects are never changed here.
for name in (*CHANGED,'RUNTIME.json'):
    source=BASE/name
    if source.is_file():
        destination=BACKUP/name;destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists():assert sha(destination)==sha(source),'Existing backup is different'
        else:shutil.copy2(source,destination)
for name in CHANGED:shutil.copy2(CANDIDATE/name,BASE/name)
shutil.copy2(CANDIDATE/'RUNTIME.json',BASE/'RUNTIME.json')
for name,checksum in new['files'].items():assert sha(BASE/name)==checksum,name
receipt=dict(previous_manifest=OLD,installed_manifest=NEW,changed=sorted(changed),
             previous_source_backup=str(BACKUP),native_manifest_unchanged=True)
(ROOT/'desktop/evidence/breaklines147/DELIVERY.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt,indent=2))
