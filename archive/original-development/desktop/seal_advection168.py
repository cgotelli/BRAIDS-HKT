"""Seal the conservative cross-advection decision without promoting a solver."""
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'validation/_advection168'
OUT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_advection168'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    target=OUT/'MANIFEST.json'
    if target.exists():raise FileExistsError(target)
    audit=json.loads((OUT/'AUDIT.json').read_text())
    runtime=ROOT/'desktop/runtime'
    assert sha(runtime/'RUNTIME.json')==audit['runtime_sha256']
    for name,digest in audit['manifest_installed']['files'].items():assert sha(runtime/name)==digest
    for name,digest in audit['native_sources_read_only'].items():assert sha(ROOT/name)==digest
    for name,digest in audit['scripts'].items():assert sha(ROOT/'desktop'/name)==digest
    suite=ET.parse(RAW/'checks.xml').getroot().find('testsuite')
    assert int(suite.attrib['tests'])==51
    assert all(int(suite.attrib[k])==0 for k in ('failures','errors','skipped'))
    receipts={};data={}
    for backend in ('cpu','gpu'):
        p=RAW/'probes'/backend/'AUDIT.json';row=json.loads(p.read_text())
        assert row['status']['status']=='segment_complete'
        assert row['script_sha256']==sha(ROOT/'desktop/check_advection168.py')
        receipts[str(p.relative_to(ROOT))]=sha(p)
        for name,digest in row['outputs'].items():
            assert sha(p.parent/name)==digest
            data[str((p.parent/name).relative_to(ROOT))]=digest
    folder=RAW/'native_cyclic'
    for name in ('INPUTS.json','execution.json'):
        receipts[str((folder/name).relative_to(ROOT))]=sha(folder/name)
    for name,digest in json.loads((folder/'INPUTS.json').read_text())['inputs'].items():assert sha(folder/name)==digest
    for row in json.loads((folder/'execution.json').read_text())['outputs']:
        assert sha(folder/row['name'])==row['sha256']
        data[str((folder/row['name']).relative_to(ROOT))]=row['sha256']
    files=list((ROOT/'desktop').glob('*168.py'))+[ROOT/'tests/test_desktop_advection168.py',ROOT/'validation/RIVER_MORPHODYNAMICS_STATUS.md']
    for p in files:
        dest=OUT/'source_snapshot'/p.relative_to(ROOT)
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    payload=dict(scope='Suppression decision and control-volume accounting; improved groin jet remains open',
        production_changed=False,Delft3D_modified=False,improved_jet_qualified=False,
        tests=dict(passed=51,failures=0,skipped=0,receipt=str((RAW/'checks.xml').relative_to(ROOT)),sha256=sha(RAW/'checks.xml')),
        receipts=receipts,outputs=data,sources={str(p.relative_to(ROOT)):sha(p) for p in files},
        artifacts={p.name:sha(p) for p in OUT.iterdir() if p.is_file()},
        manual=dict(sha256=sha(RAW/'FLOW_User_Manual.pdf'),url=audit['manual']['url']))
    target.write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
    print('Desktop168 audit sealed; no suppression or new solver installed.')


if __name__=='__main__':main()
