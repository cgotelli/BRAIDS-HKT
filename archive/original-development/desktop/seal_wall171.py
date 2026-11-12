"""Seal bounded wall qualification; no production or reference replacement."""
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_reflection171'
RAW=ROOT/'validation/_wall171'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    path=OUT/'MANIFEST.json'
    if path.exists():raise FileExistsError(path)
    a=json.loads((OUT/'AUDIT.json').read_text())
    for key in ('inputs','scripts','source_reference'):
        for name,digest in a[key].items():assert sha(ROOT/name)==digest,name
    runtime=ROOT/'desktop/runtime'
    assert sha(runtime/'RUNTIME.json')==a['runtime_manifest_sha256']
    for name,digest in json.loads((runtime/'RUNTIME.json').read_text())['files'].items():assert sha(runtime/name)==digest,name
    receipt=RAW/'checks.xml';suite=ET.parse(receipt).getroot().find('testsuite')
    assert int(suite.attrib['tests'])==348
    assert all(int(suite.attrib[k])==0 for k in ('failures','errors','skipped'))
    sources=list((ROOT/'desktop').glob('*171.py'))+[
        ROOT/'desktop/wall_pressure170.py',ROOT/'tests/test_desktop_reflection171.py',
        ROOT/'validation/RIVER_MORPHODYNAMICS_STATUS.md']
    for p in sources:
        dst=OUT/'source_snapshot'/p.relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
    result=dict(scope='Exact wall-pressure general-worker numerical qualification; isolated CPU/GPU runtime',
        complete_trajectories=24,paired_operator_states=50,cancel_resume_pairs=16,
        qualified=True,installed=False,production_changed=False,Delft3D_modified=False,groin_jet_agreement_closed=False,
        tests=dict(passed=348,new=106,retained=242,failed=0,skipped=0,receipt=str(receipt.relative_to(ROOT)),sha256=sha(receipt)),
        runtime_manifest_sha256=a['runtime_manifest_sha256'],
        sources={str(p.relative_to(ROOT)):sha(p) for p in sources},
        artifacts={p.name:sha(p) for p in OUT.iterdir() if p.is_file()},
        figures_visually_checked=['WALL_TRAJECTORIES.png','CONSERVATION_AND_PARITY.png','BEDLOAD_RESTART.png'],
        acceptance_note='Groin mixed-q preliminary 1e-6 screen failed (6.06e-6); depth 0.004 mm accepted under user millimetre allowance. Closed controls remain roundoff.',
        next_milestone='Canonical Desktop integration and compiler-free paired package/boundary/three-bedload-law qualification before installation')
    path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print('Desktop171 sealed: 348 checks; 24 full trajectories; preview26 unchanged.')


if __name__=='__main__':main()
