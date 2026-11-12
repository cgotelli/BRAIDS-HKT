"""Seal the isolated pressure qualification without changing Desktop delivery."""
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_interface170'
RAW=ROOT/'validation/_interface170'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    target=OUT/'MANIFEST.json'
    if target.exists():raise FileExistsError(target)
    a=json.loads((OUT/'AUDIT.json').read_text())
    for group in (a['inputs'],a['native_sources_read_only'],a['waves']['sources']):
        for name,digest in group.items():assert sha(ROOT/name)==digest,name
    for name,digest in a['scripts'].items():assert sha(ROOT/'desktop'/name)==digest,name
    runtime=ROOT/'desktop/runtime'
    assert sha(runtime/'RUNTIME.json')==a['runtime_manifest_sha256']
    for name,digest in json.loads((runtime/'RUNTIME.json').read_text())['files'].items():
        assert sha(runtime/name)==digest,name
    suite=ET.parse(RAW/'checks.xml').getroot().find('testsuite')
    assert int(suite.attrib['tests'])==242
    assert all(int(suite.attrib[k])==0 for k in ('errors','failures','skipped'))
    files=list((ROOT/'desktop').glob('*170.py'))+[ROOT/'tests/test_desktop_interface170.py',ROOT/'validation/RIVER_MORPHODYNAMICS_STATUS.md']
    for p in files:
        dst=OUT/'source_snapshot'/p.relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
    result=dict(scope='Wave audit and pressure-only candidate, one bounded paired groin comparison',
        production_changed=False,Delft3D_modified=False,groin_improvement_gate=False,
        tests=dict(passed=242,failed=0,skipped=0,receipt=str((RAW/'checks.xml').relative_to(ROOT)),sha256=sha(RAW/'checks.xml')),
        sources={str(p.relative_to(ROOT)):sha(p) for p in files},
        artifacts={p.name:sha(p) for p in OUT.iterdir() if p.is_file()},
        evidence=a['inputs'],
        figures_visually_checked=['WAVE_PARTS.png','WALL_PRESSURE.png','GROIN_COMPARISON.png'],
        retained_harness_failures=[
            'v1 wave audit assumed an unused Euler theta buffer was initialized; actual residual bypasses that buffer.',
            'Pressure harness initially omitted import path and compiler DLL directory; corrected before completed controls.',
            'runs/pressure_v1 GPU constructor hook was superseded by final face activity installation; baseline output, NOT corrected GPU evidence.'],
        candidate_selection='Exact wall Riemann pressure, pending general wet/dry/cropped-wall and restart delivery gates')
    target.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print('Desktop170 sealed: 242 checks; pressure prototype only; preview26 unchanged.')


if __name__=='__main__':main()
