"""Install the qualified boundary-map-only repair, preserving prior source."""
import hashlib,json,shutil
from pathlib import Path
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'validation/_amr150'
base=ROOT/'desktop/runtime';candidate=out/'candidate';backup=out/'preview14_source'
name='braids_core/sediment/river_amr_transfer.py'
old='588c8124dace5afcc3b695cff5d0af03d0e3884393cc2b8e16eec82a715e98cf'
new='ef2ec1e43184f884644e9f4fff717d175df5507a72f213d8b53081f5afe501d1'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
suite=ET.parse(out/'FINAL_QUALIFIED.xml').getroot().find('testsuite')
assert int(suite.attrib['tests'])==215 and int(suite.attrib['errors'])==int(suite.attrib['failures'])==0
comparison=json.loads((out/'COMPARISON.json').read_text())
assert max(comparison['cpu_restart'].values())==0 and len(json.loads((out/'SWEEP.json').read_text()))==18
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
receipt=dict(previous_runtime=old,installed_runtime=new,changed=[name],native_libraries_unchanged=True,
    backup=str(backup),regression_tests=215,full_boundary_partition_pairs=18)
evidence=ROOT/'desktop/evidence/amr150';evidence.mkdir(parents=True,exist_ok=True)
(evidence/'DELIVERY.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt,indent=2))
