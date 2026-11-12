"""Hash the groin audit and verify the installed/reference engines stayed intact."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'desktop')]
from stage_wall172 import save
from seal_pulse177 import sha
OUT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_tip178'
BASE=ROOT/'validation/_tip178'


def main():
    runtime=ROOT/'desktop/runtime';identity=sha(runtime/'RUNTIME.json')
    assert identity=='8bb07b4e4877411e0035060842a47feecc41b1d5e04f14cb389407c38c9349c9'
    for name,digest in json.loads((runtime/'RUNTIME.json').read_text())['files'].items():
        assert sha(runtime/name)==digest,name
    sources={};folder=ROOT/'resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/flow2d3d_kernel/src'
    with zipfile.ZipFile(ROOT/'resources/Delft3D-main.zip') as z:
        for name in ('compute/mom_fls.f90','compute/ulim.f90','compute/uzd.f90',
                     'compute/upwhu.f90','main/trisol.f90','inichk/init_mom_output.f90'):
            p=folder/name;member=p.relative_to(ROOT/'resources').as_posix();raw=z.read(member)
            if p.exists():assert p.read_bytes()==raw,name
            sources[p.relative_to(ROOT).as_posix()]=hashlib.sha256(raw).hexdigest()
    files={}
    for p in [*BASE.rglob('*'),*(ROOT/'desktop').glob('*178.py'),ROOT/'tests/test_desktop_tip178.py',*OUT.iterdir()]:
        if not p.is_file() or '__pycache__' in p.parts or p.name=='MANIFEST.json':continue
        files[p.relative_to(ROOT).as_posix()]=sha(p)
    refs={}
    for name in ('desktop_corner174','desktop_zero175','desktop_corner176','desktop_pulse177'):
        p=OUT.parent/name/'MANIFEST.json';refs[p.relative_to(ROOT).as_posix()]=sha(p)
    for name in ('validation/_corner174/delft/trim-sim_P057.nc','validation/_corner174/delft/execution.json'):
        refs[name]=sha(ROOT/name)
    save(OUT/'MANIFEST.json',dict(audit_complete=True,groin_accuracy_qualified=False,
        production_changed=False,Delft3D_source_modified=False,Delft3D_runtime_modified=False,
        runtime_manifest_sha256=identity,Delft3D_sources=sources,reference_manifests=refs,files=files,
        source_scope='Read-only original files or archive contents; no source overlays or native rebuild.',
        scope='43 same-state replays per backend and two 2 ms manufactured FLOW controls; no long run.'))
    print('Verified unchanged engines; sealed',len(files),'files',flush=True)


if __name__=='__main__':main()
