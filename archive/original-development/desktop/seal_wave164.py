"""Retain only this milestone's BRAIDS sources and evidence hashes."""
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_wave164'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    suite=ROOT/'validation/_wave164/current_regressions_final.xml'
    xml=ET.parse(suite);cases=list(xml.iter('testcase'))
    assert len(cases)==179
    assert not list(xml.iter('failure')) and not list(xml.iter('error')) and not list(xml.iter('skipped'))
    paths=[ROOT/'braids_core/sediment'/name for name in ('bed_wave_flux.py','river_capacity.py',
        'river_suspension.py','wall_velocity_bounds.py','wall_bed_hydraulics.py')]
    paths += [ROOT/'tests/test_bed_wave_flux164.py']
    paths += list((ROOT/'desktop').glob('*164.py'))
    hashes={}
    for p in paths:
        target=OUT/'source_snapshot'/p.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert sha(target)==sha(p),target
        else:shutil.copy2(p,target)
        hashes[p.relative_to(ROOT).as_posix()]=sha(p)
    artifacts={p.relative_to(OUT).as_posix():sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='MANIFEST.json'}
    runtime=sha(ROOT/'desktop/runtime/RUNTIME.json')
    assert runtime=='8e72c97e1782ba3cca50c32189c9da47cbb278f175d754a6dc831e620d6f0e5e'
    receipt=dict(sources_sha256=hashes,artifacts_sha256=artifacts,
        installed_runtime_sha256=runtime,installed_desktop_changed=False,Delft3D_source_modified=False,
        test_receipt=dict(path=str(suite.relative_to(ROOT)),sha256=sha(suite),passed=len(cases),failed=0,skipped=0),
        scope='Current public groin correction; historical shoal temporal limitation explicitly retained; no universal accuracy or package claim')
    (OUT/'MANIFEST.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt['test_receipt'],indent=2))


if __name__=='__main__':main()
