"""Install the gated local Desktop preview; preserve every replaced payload."""
import hashlib
import json
import shutil
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
BASE=ROOT/'desktop/runtime'
FOLDER=ROOT/'validation/_shoal165'
CANDIDATE=FOLDER/'candidate3'
BACKUP=FOLDER/'preview25_backup'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if sys.argv[1:]==['--receipt-only']:
        # Recovery after a post-install reporting failure: authenticate the
        # whole installed payload and retained prior manifest, without writes
        # to any runtime/library or backup file.
        assert sha(BASE/'RUNTIME.json')==sha(CANDIDATE/'RUNTIME.json')
        assert sha(BACKUP/'runtime/RUNTIME.json')=='8e72c97e1782ba3cca50c32189c9da47cbb278f175d754a6dc831e620d6f0e5e'
        old=json.loads((BACKUP/'runtime/RUNTIME.json').read_text())
        new=json.loads((CANDIDATE/'RUNTIME.json').read_text())
        for name,digest in new['files'].items():assert sha(BASE/name)==digest,name
        for name,digest in old['files'].items():
            saved=BACKUP/'runtime'/name
            assert sha(saved if saved.exists() else BASE/name)==digest,name
        changed=[name for name,digest in new['files'].items() if old['files'].get(name)!=digest]
        publish_receipt(changed)
        return
    assert sha(BASE/'RUNTIME.json')=='8e72c97e1782ba3cca50c32189c9da47cbb278f175d754a6dc831e620d6f0e5e'
    if BACKUP.exists():raise FileExistsError(BACKUP)
    for name,count in (('REGRESSIONS.xml',179),('PACKAGE3.xml',27),('SHOAL.xml',5)):
        suite=ET.parse(FOLDER/name).getroot().find('testsuite')
        assert int(suite.attrib['tests'])==count
        assert int(suite.attrib['failures'])==int(suite.attrib['errors'])==int(suite.attrib['skipped'])==0
    audit=json.loads((ROOT/'validation/sediment/river_morphodynamics/reports/desktop_shoal165/AUDIT.json').read_text())
    assert sha(CANDIDATE/'RUNTIME.json')==audit['candidate3_runtime_sha256']
    for row in audit['groin_release_regression'].values():assert all(row['exact'].values())
    old=json.loads((BASE/'RUNTIME.json').read_text());new=json.loads((CANDIDATE/'RUNTIME.json').read_text())
    for base,manifest in ((BASE,old),(CANDIDATE,new)):
        for name,digest in manifest['files'].items():
            path=(base/name).resolve()
            assert path.is_relative_to(base.resolve())
            assert sha(path)==digest,name
    changed=[name for name,digest in new['files'].items() if old['files'].get(name)!=digest]
    BACKUP.mkdir(parents=True)
    for name in changed+['RUNTIME.json']:
        target=BASE/name
        if target.exists():
            saved=BACKUP/'runtime'/name;saved.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(target,saved)
    for name in ('desktop/PROVENANCE.json','desktop/README.md','braids_desktop/__init__.py'):
        saved=BACKUP/name;saved.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,saved)
    # All checks precede the first installed replacement; manifest is last.
    for name in changed+['RUNTIME.json']:
        target=BASE/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(CANDIDATE/name,target)
    for name,digest in new['files'].items():assert sha(BASE/name)==digest,name
    publish_receipt(changed)


def publish_receipt(changed):
    receipt=dict(previous_runtime=sha(BACKUP/'runtime/RUNTIME.json'),installed_runtime=sha(BASE/'RUNTIME.json'),
        changed=changed,backup=str(BACKUP),QGIS_modified=False,Delft3D_modified=False,user_projects_modified=False,
        qualification='validation/sediment/river_morphodynamics/reports/desktop_shoal165/REPORT.md')
    from braids_gpu.atomic_output import write_json_atomic
    folder=ROOT/'desktop/evidence/shoal165';folder.mkdir(parents=True,exist_ok=True)
    write_json_atomic(folder/'DELIVERY.json',receipt)
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
