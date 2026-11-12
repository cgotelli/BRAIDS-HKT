"""Seal bounded qualification evidence, retaining failed candidate outcomes."""
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_corner167'
RAW=ROOT/'validation/_corner167'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    target=OUT/'MANIFEST.json'
    if target.exists():raise FileExistsError(target)
    audit=json.loads((OUT/'AUDIT.json').read_text());runtime=ROOT/'desktop/runtime'
    assert sha(runtime/'RUNTIME.json')==audit['runtime_manifest_sha256']
    for name,digest in json.loads((runtime/'RUNTIME.json').read_text())['files'].items():
        assert sha(runtime/name)==digest,name
    for name,digest in audit['native_sources_read_only'].items():assert sha(ROOT/name)==digest
    tests=RAW/'checks_qualified.xml';suite=ET.parse(tests).getroot().find('testsuite')
    assert int(suite.attrib['tests'])==37
    assert all(int(suite.attrib[k])==0 for k in ('errors','failures','skipped'))
    files=[str(p.relative_to(ROOT)) for p in (ROOT/'desktop').glob('*167.py')]
    files+=['tests/test_desktop_corner167.py','validation/RIVER_MORPHODYNAMICS_STATUS.md']
    for name in files:
        dest=OUT/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    receipts={};outputs={}
    for path in RAW.glob('final_*/*/AUDIT.json'):
        row=json.loads(path.read_text())
        for name,digest in row['outputs'].items():assert sha(path.parent/name)==digest
        receipts[str(path.relative_to(ROOT))]=sha(path)
    assert len(receipts)==8
    for path in (RAW/'runs').glob('final_*/*/STATUS.json'):
        row=json.loads(path.read_text());assert row['status']['status']=='complete'
        assert row['script_sha256']==sha(ROOT/'desktop/run_corner167.py')
        assert row['candidate_sha256']==sha(ROOT/'desktop/corner_candidate167.py')
        receipts[str(path.relative_to(ROOT))]=sha(path)
        for p in (path.parent/'run').rglob('*'):
            if p.is_file():outputs[str(p.relative_to(ROOT))]=sha(p)
    assert len(receipts)==16
    for label in ('native_momentum','native_momentum_fine'):
        path=RAW/label
        for name,digest in json.loads((path/'INPUTS.json').read_text())['inputs'].items():assert sha(path/name)==digest
        for row in json.loads((path/'execution.json').read_text())['outputs']:assert sha(path/row['name'])==row['sha256']
        for name in ('INPUTS.json','execution.json'):receipts[str((path/name).relative_to(ROOT))]=sha(path/name)
    builds={str(p.relative_to(ROOT)):sha(p) for p in (RAW/'build').glob('*/IDENTITY.json')}
    for name in builds:
        p=ROOT/name;row=json.loads(p.read_text());assert sha(p.parent/'hydraulics.dll')==row['library_sha256']
        # Build identity hashes canonical compiler input text; Windows writes
        # CRLF on disk. Preserve that identity and separately hash file bytes.
        assert hashlib.sha256((p.parent/'corner.f90').read_text().encode()).hexdigest()==row['source_sha256']
    payload=dict(installed_runtime_changed=False,Delft3D_modified=False,improvement_gate_met=False,
        scope='Corner diagnosis and bounded rejection of candidates; no new solver release',
        tests=dict(passed=37,failures=0,skipped=0,receipt=str(tests.relative_to(ROOT)),sha256=sha(tests)),
        receipts=receipts,run_outputs=outputs,builds=builds,sources={name:sha(ROOT/name) for name in files},
        artifacts={p.name:sha(p) for p in OUT.iterdir() if p.is_file()})
    target.write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
    print('Desktop167 bounded qualification sealed; improved-jet gate remains open; preview26 unchanged.')


if __name__=='__main__':main()
