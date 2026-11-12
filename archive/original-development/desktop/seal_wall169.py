"""Seal the wall component and scour-accounting milestone, without deployment."""
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_wall169'
RAW=ROOT/'validation/_wall169'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    target=OUT/'MANIFEST.json'
    if target.exists():raise FileExistsError(target)
    audit=json.loads((OUT/'AUDIT.json').read_text())
    for group in (audit['wall']['files'],audit['chain']['inputs'],audit['native_sources_read_only']):
        for name,digest in group.items():assert sha(ROOT/name)==digest,name
    for name,digest in audit['scripts'].items():assert sha(ROOT/'desktop'/name)==digest
    runtime=ROOT/'desktop/runtime'
    assert sha(runtime/'RUNTIME.json')==audit['runtime_manifest_sha256']
    for name,digest in json.loads((runtime/'RUNTIME.json').read_text())['files'].items():
        assert sha(runtime/name)==digest,name
    prior=OUT.parent/'desktop_corner167'
    pm=json.loads((prior/'MANIFEST.json').read_text())
    assert sha(prior/'AUDIT.json')==pm['artifacts']['AUDIT.json']==audit['prior_audit_sha256']
    # Retained CPU/GPU candidate receipts are still the same completed runs.
    for backend in ('cpu','gpu'):
        p=ROOT/f'validation/_corner167/runs/final_tangent/{backend}/STATUS.json'
        assert sha(p)==audit['retained_coupled']['tangent'][backend+'_status_sha256']
    suite=ET.parse(RAW/'checks.xml').getroot().find('testsuite')
    assert int(suite.attrib['tests'])==87
    assert all(int(suite.attrib[k])==0 for k in ('errors','failures','skipped'))
    files=list((ROOT/'desktop').glob('*169.py'))+[ROOT/'tests/test_desktop_wall169.py',ROOT/'validation/RIVER_MORPHODYNAMICS_STATUS.md']
    for p in files:
        dest=OUT/'source_snapshot'/p.relative_to(ROOT)
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    result=dict(scope='Rotated-wall component qualification and complete local scour accounting; no improved-groin release',
        production_changed=False,Delft3D_modified=False,groin_improvement_gate=False,
        checks=dict(passed=87,failed=0,skipped=0,receipt=str((RAW/'checks.xml').relative_to(ROOT)),sha256=sha(RAW/'checks.xml')),
        sources={str(p.relative_to(ROOT)):sha(p) for p in files},
        evidence={**audit['wall']['files'],**audit['chain']['inputs']},
        artifacts={p.name:sha(p) for p in OUT.iterdir() if p.is_file()},
        harness_preflight_failures=['Missing diagnostic CUDA-source path; corrected before geometry construction.',
            'Zero Manning rejected by carrier roughness validator; corrected to positive inert carrier value before operator checks.'],
        figures_visually_checked=['ROTATED_WALLS.png','SCOUR_CHAIN.png','WALL_AND_COUPLED.png'])
    target.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print('Desktop169 sealed: 87 checks; preview26 unchanged; no Delft3D changes.')


if __name__=='__main__':main()
