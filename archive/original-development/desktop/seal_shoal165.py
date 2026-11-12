"""Freeze Desktop165 reproducibility receipts after installed verification."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from braids_gpu.atomic_output import write_json_atomic
OUT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_shoal165'
RAW=ROOT/'validation/_shoal165'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if (OUT/'MANIFEST.json').exists():raise FileExistsError(OUT/'MANIFEST.json')
    audit=json.loads((OUT/'AUDIT.json').read_text())
    assert sha(ROOT/'desktop/runtime/RUNTIME.json')==audit['candidate3_runtime_sha256']
    manifest=json.loads((ROOT/'desktop/runtime/RUNTIME.json').read_text())
    for name,digest in manifest['files'].items():assert sha(ROOT/'desktop/runtime'/name)==digest
    receipts={}
    for name,count in (('REGRESSIONS.xml',179),('PACKAGE3.xml',27),('SHOAL.xml',5),('INSTALLED.xml',27)):
        path=RAW/name;suite=ET.parse(path).getroot().find('testsuite')
        assert int(suite.attrib['tests'])==count
        assert all(int(suite.attrib[key])==0 for key in ('failures','errors','skipped'))
        receipts[name]=dict(sha256=sha(path),passed=count,failed=0,skipped=0)
    files=['desktop/'+name for name in ('check_shoal165.py','stage_shoal165.py','check_groin165.py',
        'review_shoal165.py','promote_shoal165.py','seal_shoal165.py','README.md','PROVENANCE.json')]
    files+=['tests/test_desktop_shoal165.py','braids_desktop/__init__.py',
            'validation/RIVER_MORPHODYNAMICS_STATUS.md','desktop/evidence/shoal165/DELIVERY.json']
    files+=json.loads((RAW/'STAGE.json').read_text())['source_files']
    for name in files:
        target=OUT/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,target)
    data=dict(installed_runtime_sha256=sha(ROOT/'desktop/runtime/RUNTIME.json'),
        version='0.1.0-preview26',test_receipts=receipts,
        sources_sha256={name:sha(ROOT/name) for name in files},
        artifacts_sha256={p.name:sha(p) for p in OUT.iterdir() if p.is_file()},
        QGIS_modified=False,Delft3D_modified=False,
        acceptance='Short three-law current-worker wet/dry/transcritical release; not EH or experimental reproduction',
        old_runtime_checkpoint_migration=False)
    write_json_atomic(OUT/'MANIFEST.json',data)
    print(json.dumps(dict(version=data['version'],runtime=data['installed_runtime_sha256'],receipts=receipts),indent=2))


if __name__=='__main__':main()
