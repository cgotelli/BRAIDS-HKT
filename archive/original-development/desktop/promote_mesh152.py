"""Install qualified preview17 sources only, preserving the previous runtime."""
import hashlib,json,shutil
from pathlib import Path
import xml.etree.ElementTree as ET
from stage_mesh152 import ROOT,FILES
OUT=ROOT/'validation/_mesh152';base=ROOT/'desktop/runtime';candidate=OUT/'candidate'
backup=OUT/'preview16_source'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
suite=ET.parse(OUT/'QUALIFIED.xml').getroot().find('testsuite')
assert int(suite.attrib['tests'])==246 and int(suite.attrib['errors'])==int(suite.attrib['failures'])==0
for case in ('project','lake','wetfront','bedload'):
    for backend in ('cpu','gpu'):
        row=json.loads((OUT/(case+'_'+backend)/'CHECK.json').read_text())
        assert row['run']['status']=='complete' and row['maximum_wall_integral']==0
        assert row['maximum_water_residual_m3']<1e-7 and row['maximum_class_residual_kg']<1e-7
        if case=='bedload':assert max(row['restart_maximum_difference'].values())==0
        if case=='wetfront':assert sum(e['changed'] for e in row['adaptations'])>1
comparison=json.loads((OUT/'COMPARISON.json').read_text())
assert comparison['project']['q']<1e-9 and comparison['wetfront']['q']<1e-9
assert comparison['bedload']['depth_m']<1e-4 and comparison['bedload']['z']<1e-4 and comparison['bedload']['q']<1e-3
assert sha(base/'RUNTIME.json')=='d5017b52fca9c42cfe5645a03345967921e3f2f0481c662d917d127085f02b02'
a=json.loads((base/'RUNTIME.json').read_text());b=json.loads((candidate/'RUNTIME.json').read_text())
assert {k for k in b['files'] if a['files'].get(k)!=b['files'][k]}==set(FILES)
assert set(a['files'])==set(b['files'])
for root,manifest in ((base,a),(candidate,b)):
    for k,digest in manifest['files'].items():assert sha(root/k)==digest,k
assert sha(base/'PRECOMPILED.json')==sha(candidate/'PRECOMPILED.json')
for name in (*FILES,'RUNTIME.json'):
    target=backup/name;target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():assert sha(target)==sha(base/name)
    else:shutil.copy2(base/name,target)
for name in (*FILES,'RUNTIME.json'):shutil.copy2(candidate/name,base/name)
for k,digest in b['files'].items():assert sha(base/k)==digest,k
receipt=dict(previous_runtime=sha(backup/'RUNTIME.json'),installed_runtime=sha(base/'RUNTIME.json'),
    changed=list(FILES),native_libraries_unchanged=True,backup=str(backup),candidate_tests=246)
evidence=ROOT/'desktop/evidence/mesh152';evidence.mkdir(parents=True,exist_ok=True)
(evidence/'DELIVERY.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
