"""Seal diagnostic evidence without promoting an unqualified hydraulic change."""
import json
from pathlib import Path
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'desktop')]
from stage_wall172 import save,sha
BASE=ROOT/'validation/_zero175'
OUT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_zero175'


def main():
    runtime=ROOT/'desktop/runtime'
    identity=sha(runtime/'RUNTIME.json')
    assert identity=='8bb07b4e4877411e0035060842a47feecc41b1d5e04f14cb389407c38c9349c9'
    for name,digest in json.loads((runtime/'RUNTIME.json').read_text())['files'].items():
        assert sha(runtime/name)==digest,name
    sources={}
    folder=ROOT/'resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/flow2d3d_kernel/src'
    import hashlib
    with zipfile.ZipFile(ROOT/'resources/Delft3D-main.zip') as archive:
        for name in ('compute/mom_fls.f90','compute/ulim.f90','compute/slim.f90',
                     'compute/upwhu.f90','compute/cucbp.f90','timedep/incbc.f90','main/trisol.f90'):
            p=folder/name;digest=sha(p);suffix=p.relative_to(ROOT/'resources').as_posix()
            matches=[n for n in archive.namelist() if n==suffix or n.endswith('/'+suffix)]
            assert len(matches)==1,(name,matches)
            assert hashlib.sha256(archive.read(matches[0])).hexdigest()==digest,name
            sources[p.relative_to(ROOT).as_posix()]=digest
    files={p.relative_to(ROOT).as_posix():sha(p) for p in BASE.rglob('*')
           if p.is_file() and '__pycache__' not in p.parts}
    for pattern in ('*zero175.py','reconstruction175.py'):
        for p in (ROOT/'desktop').glob(pattern):files[p.relative_to(ROOT).as_posix()]=sha(p)
    for p in [ROOT/'tests/test_desktop_zero175.py',*OUT.glob('*.png'),OUT/'REPORT.md',OUT/'DIAGNOSTICS.json']:
        files[p.relative_to(ROOT).as_posix()]=sha(p)
    save(OUT/'MANIFEST.json',dict(diagnostic_complete=True,hydraulic_fix_qualified=False,
        production_changed=False,Delft3D_source_modified=False,Delft3D_runtime_modified=False,
        runtime_manifest_sha256=identity,Delft3D_source= sources,files=files,
        caveat='Final analysis/harness sources retained; individual CHECK files record executed script hashes. Later harness additions did not alter earlier run inputs or generated operator sources.'))
    print('Sealed',len(files),'files; no solver promotion',flush=True)


if __name__=='__main__':main()
