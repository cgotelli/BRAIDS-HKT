"""Deliver the tested menu consolidation, not a new hydraulic algorithm."""
import hashlib,json,shutil
from pathlib import Path
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];base=ROOT/'desktop/runtime'
folder=ROOT/'validation/_boundaries159';candidate=folder/'candidate';backup=folder/'preview23_runtime'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(base/'RUNTIME.json')=='060223af890c56183c5c4c7118a986ae013c0e483b4d1dc4bec20fdb4f37d7b4'
for name in ('DESKTOP.xml','PUBLIC2.xml','OPERATORS.xml'):
    suite=ET.parse(folder/name).getroot().find('testsuite')
    assert int(suite.attrib['failures'])==int(suite.attrib['errors'])==0,name
for backend in ('cpu','gpu'):
    rows=json.loads((folder/'candidate_runs'/(backend+'_SUMMARY.json')).read_text())
    assert len(rows)==14 and not any(r['exit_code'] for r in rows),backend
    r=json.loads((folder/'candidate_runs'/('normal_'+backend)/'RESTART.json').read_text())
    assert max(r['maximum_absolute_difference'].values())==0
old=json.loads((base/'RUNTIME.json').read_text());new=json.loads((candidate/'RUNTIME.json').read_text())
changed=[k for k,v in new['files'].items() if old['files'].get(k)!=v]
assert changed==['braids_core/river_open_boundary.py'],changed
for source,manifest in ((base,old),(candidate,new)):
    for name,digest in manifest['files'].items():assert sha(source/name)==digest,name
for name in [*changed,'RUNTIME.json']:
    saved=backup/name;saved.parent.mkdir(parents=True,exist_ok=True)
    if saved.exists():raise FileExistsError(saved)
    shutil.copy2(base/name,saved);shutil.copy2(candidate/name,base/name)
for name,digest in new['files'].items():assert sha(base/name)==digest,name
receipt=dict(previous_runtime=sha(backup/'RUNTIME.json'),installed_runtime=sha(base/'RUNTIME.json'),
    changed=changed,backup=str(backup),numerical_kernels_changed=False,
    retired_public_methods=['inlet.depth_discharge','outlet.supercritical'],
    open_issue='Free-weir outlet rejects local upstream reversal in the groin wake; not general river-outlet qualification.',
    Delft3D_modified=False,QGIS_modified=False)
out=ROOT/'desktop/evidence/boundaries159';out.mkdir(parents=True,exist_ok=True)
(out/'DELIVERY.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))
