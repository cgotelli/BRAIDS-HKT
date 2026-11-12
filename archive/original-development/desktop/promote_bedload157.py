"""Promote qualified Desktop changes only; retain superseded source evidence."""
import hashlib,json,shutil
from pathlib import Path
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'validation/_bedload157'
base=ROOT/'desktop/runtime';candidate=out/'candidate';backup=out/'preview21_source'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
suite=ET.parse(out/'FINAL.xml').getroot().find('testsuite')
assert int(suite.attrib['tests'])>=225 and int(suite.attrib['failures'])==int(suite.attrib['errors'])==0
for name in ('final_cpu','final_gpu','control_cpu','warmup_cpu','warmup_gpu2','adaptive_cpu2','adaptive_gpu2'):
    check=json.loads((out/name/'CHECK.json').read_text())
    assert check['sediment_residual_kg']<1e-7 and check['water_residual_m3']<1e-7
    assert json.loads((out/name/'run/STATUS.json').read_text())['status']=='complete'
    if name.startswith('warmup'):assert max(check['restart_max_abs'].values())==0.
assert sha(base/'RUNTIME.json')=='4d5ebbf4cea1b8cdb275113e139567559ffcfe6c1abdce208135aed4d58f4bd7'
a=json.loads((base/'RUNTIME.json').read_text());b=json.loads((candidate/'RUNTIME.json').read_text())
for root,data in ((base,a),(candidate,b)):
    for name,digest in data['files'].items():assert sha(root/name)==digest,name
assert set(a['files'])<=set(b['files'])
changed=[name for name,digest in b['files'].items() if a['files'].get(name)!=digest]
for name in (*changed,'RUNTIME.json'):
    target=base/name
    if target.exists():
        old=backup/name;old.parent.mkdir(parents=True,exist_ok=True)
        if old.exists():assert sha(old)==sha(target)
        else:shutil.copy2(target,old)
    target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(candidate/name,target)
for name,digest in b['files'].items():assert sha(base/name)==digest,name
receipt=dict(previous_runtime=sha(backup/'RUNTIME.json'),installed_runtime=sha(base/'RUNTIME.json'),
    tests=int(suite.attrib['tests']),changed=changed,backup=str(backup),Delft3D_modified=False,QGIS_modified=False)
evidence=ROOT/'desktop/evidence/bedload157';evidence.mkdir(parents=True,exist_ok=True)
(evidence/'DELIVERY.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps(receipt,indent=2))
