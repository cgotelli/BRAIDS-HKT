"""Install qualified preview25 runtime without touching QGIS or user projects."""
import hashlib,json,shutil
from pathlib import Path
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];base=ROOT/'desktop/runtime'
folder=ROOT/'validation/_boundaries160';candidate=folder/'candidate3';backup=folder/'preview24_runtime'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(base/'RUNTIME.json')=='398367e1cc1e663c4367dd8d62a81db69054429e48eafef7c4eece1ded177356'
for name in ('OPERATORS.xml','DESKTOP3.xml','SERVICES.xml'):
    suite=ET.parse(folder/name).getroot().find('testsuite')
    assert int(suite.attrib['failures'])==int(suite.attrib['errors'])==0,name
for group,count in (('weir_runs',4),('bedload_runs',3),('finite_runs',2)):
    for backend in ('cpu','gpu'):
        rows=json.loads((folder/group/(backend+'_SUMMARY.json')).read_text())
        assert len(rows)==count and not any(r['exit_code'] for r in rows)
for backend in ('cpu','gpu'):
    r=json.loads((folder/'weir_runs'/('bedload_stage_weir_'+backend)/'RESTART.json').read_text())
    assert max(r['maximum_absolute_difference'].values())==0
    for case in ('jump','dry'):
        assert json.loads((folder/'general'/(case+'_'+backend)/'CHECK.json').read_text())['status']=='complete'
summary=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_boundaries160/COMPARISON.json'
assert summary.is_file()
old=json.loads((base/'RUNTIME.json').read_text());new=json.loads((candidate/'RUNTIME.json').read_text())
changed=[k for k,v in new['files'].items() if old['files'].get(k)!=v]
# Candidate3 has the same numerical source, freshly compiled Windows DLLs,
# and updated public explanatory text. It also passed fresh dry/jump/finite
# coupled runs and real Desktop worker tests; do not claim binary identity.
measured=json.loads((folder/'candidate2/RUNTIME.json').read_text())
delta=[k for k,v in new['files'].items() if measured['files'].get(k)!=v]
assert [k for k in delta if k.endswith(('.py','.h'))]==['braids_core/river_open_boundary.py']
assert all(k in ('PRECOMPILED.json','braids_core/river_open_boundary.py') or k.startswith('precompiled/') for k in delta)
before=json.loads((folder/'candidate2/PRECOMPILED.json').read_text());after=json.loads((candidate/'PRECOMPILED.json').read_text())
for key,row in after.items():
    if row!=before.get(key):
        assert 'exterior_critical_outlet' in key or 'exterior_free_weir' in key
        for field in ('source_sha256','compiler','flags'):assert row['identity'][field]==before[key]['identity'][field]
for source,manifest in ((base,old),(candidate,new)):
    for name,digest in manifest['files'].items():assert sha(source/name)==digest,name
for name in [*changed,'RUNTIME.json']:
    saved=backup/name;saved.parent.mkdir(parents=True,exist_ok=True)
    if saved.exists():raise FileExistsError(saved)
    if (base/name).exists():shutil.copy2(base/name,saved)
    (base/name).parent.mkdir(parents=True,exist_ok=True);shutil.copy2(candidate/name,base/name)
shutil.copy2(ROOT/'desktop/PROVENANCE.json',folder/'preview24_PROVENANCE.json')
for name,digest in new['files'].items():assert sha(base/name)==digest,name
receipt=dict(previous_runtime=sha(backup/'RUNTIME.json'),installed_runtime=sha(base/'RUNTIME.json'),
    changed=changed,backup=str(backup),numerical_kernels_changed=True,
    changes=['shock-compatible subcritical outward-only free weir','output-clock roundoff-tail removal','MPM plus Ashida-Michiue and Wang-Fredsoe bedload UI'],
    old_results_modified=False,Delft3D_modified=False,QGIS_modified=False)
out=ROOT/'desktop/evidence/boundaries160';out.mkdir(parents=True,exist_ok=True)
(out/'DELIVERY.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))
