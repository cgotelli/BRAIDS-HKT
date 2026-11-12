"""Seal bounded research180; preserve engines, prior experiments and failed gates."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'desktop')]
from stage_wall172 import save
from seal_pulse177 import sha
OUT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_pair180'
BASE=ROOT/'validation/_pair180'


def main():
    runtime=ROOT/'desktop/runtime';identity=sha(runtime/'RUNTIME.json')
    assert identity=='8bb07b4e4877411e0035060842a47feecc41b1d5e04f14cb389407c38c9349c9'
    for name,digest in json.loads((runtime/'RUNTIME.json').read_text())['files'].items():assert sha(runtime/name)==digest,name
    receipt=ET.parse(BASE/'VERIFIED_REGRESSION.xml').getroot().find('testsuite')
    assert receipt.attrib['tests']=='652' and receipt.attrib['failures']=='0' and receipt.attrib['errors']=='0'
    sources={};folder=ROOT/'resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/flow2d3d_kernel/src'
    with zipfile.ZipFile(ROOT/'resources/Delft3D-main.zip') as z:
        for name in ('compute/mom_fls.f90','compute/ulim.f90','compute/uzd.f90','compute/upwhu.f90','main/trisol.f90','inichk/init_mom_output.f90'):
            p=folder/name;raw=z.read(p.relative_to(ROOT/'resources').as_posix())
            if p.exists():assert p.read_bytes()==raw,name
            sources[p.relative_to(ROOT).as_posix()]=hashlib.sha256(raw).hexdigest()
    files={}
    for p in [*BASE.rglob('*'),*(ROOT/'desktop').glob('*180.py'),ROOT/'tests/test_desktop_pair180.py',*OUT.iterdir()]:
        if not p.is_file() or '__pycache__' in p.parts or p.name=='MANIFEST.json':continue
        files[p.relative_to(ROOT).as_posix()]=sha(p)
    refs={}
    for name in ('desktop_corner174','desktop_zero175','desktop_corner176','desktop_pulse177','desktop_tip178','desktop_trace179'):
        p=OUT.parent/name/'MANIFEST.json';refs[p.relative_to(ROOT).as_posix()]=sha(p)
    for name in ('validation/_corner174/delft/trim-sim_P057.nc','validation/_corner174/delft/execution.json'):
        refs[name]=sha(ROOT/name)
    # Check every reused baseline byte against the sealed milestone179 evidence.
    prior=json.loads((OUT.parent/'desktop_trace179/MANIFEST.json').read_text())
    for name,digest in prior['files'].items():
        if name.startswith('validation/_trace179/run/baseline') or name.startswith('desktop/') or name.startswith('tests/'):
            assert sha(ROOT/name)==digest,name
    save(OUT/'MANIFEST.json',dict(candidate_assessment_complete=True,candidate_adopted=False,
        zero_viscosity_groin_fixed=False,production_changed=False,Delft3D_source_modified=False,
        Delft3D_runtime_modified=False,runtime_manifest_sha256=identity,Delft3D_sources=sources,
        reference_manifests=refs,files=files,test_receipts=dict(focused='validation/_pair180/FOCUSED_REGRESSION.xml',
        initial='validation/_pair180/CURRENT_REGRESSION.xml',verified='validation/_pair180/VERIFIED_REGRESSION.xml'),
        tests=652,historical_live_install_assertions_deselected=2,
        scope='Four new 20 s paired runs and eight 4 s wall controls; sealed current-solver controls reused. '
              'No new native run, no source patch, no physical-viscosity change; candidate not adopted.',
        source_hash_note='Research generated source IDs hash normalized text; manifest entries hash actual file bytes.'))
    print('Verified unchanged engines and reused baseline; sealed',len(files),'files; no promotion',flush=True)


if __name__=='__main__':main()
