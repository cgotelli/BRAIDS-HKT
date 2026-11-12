"""Install only the qualified preview16 runtime changes; preserve preview15."""
import hashlib,json,shutil
from pathlib import Path
import xml.etree.ElementTree as ET
from stage_solids151 import ROOT,FILES
OUT=ROOT/'validation/_solids151';base=ROOT/'desktop/runtime';candidate=OUT/'candidate'
backup=OUT/'preview15_source'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
suite=ET.parse(OUT/'FINAL_QUALIFIED.xml').getroot().find('testsuite')
assert int(suite.attrib['tests'])>=233 and int(suite.attrib['errors'])==int(suite.attrib['failures'])==0
extra=ET.parse(OUT/'UI_EXTRA.xml').getroot().find('testsuite')
assert int(extra.attrib['tests'])==6 and int(extra.attrib['errors'])==int(extra.attrib['failures'])==0
for case in ('lake','wetfront','bedload'):
    for backend in ('cpu','gpu'):
        row=json.loads((OUT/(case+'_'+backend)/'CHECK.json').read_text())
        assert row['run']['status']=='complete' and row['maximum_wall_integral']==0
        assert row['maximum_water_residual_m3']<1e-8 and row['maximum_class_residual_kg']<1e-7
        if case=='bedload':assert max(row['restart_maximum_difference'].values())==0
        if case=='wetfront':assert len(row['adaptations'])==25 and sum(e['changed'] for e in row['adaptations'])>5
comparison=json.loads((OUT/'COMPARISON.json').read_text())
assert max(row['q'] for row in comparison.values())<1e-9
assert sha(base/'RUNTIME.json')=='ef2ec1e43184f884644e9f4fff717d175df5507a72f213d8b53081f5afe501d1'
a=json.loads((base/'RUNTIME.json').read_text());b=json.loads((candidate/'RUNTIME.json').read_text())
assert {k for k in b['files'] if a['files'].get(k)!=b['files'][k]}==set(FILES)
assert set(a['files'])-set(b['files'])==set()
for root,manifest in ((base,a),(candidate,b)):
    for k,digest in manifest['files'].items():assert sha(root/k)==digest,k
assert sha(base/'PRECOMPILED.json')==sha(candidate/'PRECOMPILED.json')
for name in (*FILES,'RUNTIME.json'):
    if (base/name).is_file():
        target=backup/name;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert sha(target)==sha(base/name)
        else:shutil.copy2(base/name,target)
for name in (*FILES,'RUNTIME.json'):shutil.copy2(candidate/name,base/name)
for k,digest in b['files'].items():assert sha(base/k)==digest,k
receipt=dict(previous_runtime=sha(backup/'RUNTIME.json'),installed_runtime=sha(base/'RUNTIME.json'),
    changed=list(FILES),native_libraries_unchanged=True,backup=str(backup),candidate_tests=int(suite.attrib['tests']),extra_ui_tests=6)
evidence=ROOT/'desktop/evidence/solids151';evidence.mkdir(parents=True,exist_ok=True)
(evidence/'DELIVERY.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');print(json.dumps(receipt,indent=2))
