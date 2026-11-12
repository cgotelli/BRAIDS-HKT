"""Seal boundary diagnostic evidence; neither engine is promoted or modified."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'desktop')]
from stage_wall172 import save
BASE=ROOT/'validation/_pulse177'
OUT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_pulse177'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    runtime=ROOT/'desktop/runtime';identity=sha(runtime/'RUNTIME.json')
    assert identity=='8bb07b4e4877411e0035060842a47feecc41b1d5e04f14cb389407c38c9349c9'
    for name,digest in json.loads((runtime/'RUNTIME.json').read_text())['files'].items():
        assert sha(runtime/name)==digest,name
    print('Installed runtime hashes unchanged',flush=True)
    sources={}
    folder=ROOT/'resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/flow2d3d_kernel/src'
    with zipfile.ZipFile(ROOT/'resources/Delft3D-main.zip') as archive:
        for name in ('compute/cucbp.f90','compute/upwhu.f90','compute/mom_fls.f90',
                     'compute/ulim.f90','compute/slim.f90','compute/calrou.f90','compute/calksc.f90'):
            p=folder/name;suffix=p.relative_to(ROOT/'resources').as_posix()
            matches=[n for n in archive.namelist() if n==suffix or n.endswith('/'+suffix)]
            assert len(matches)==1
            digest=sha(p);assert hashlib.sha256(archive.read(matches[0])).hexdigest()==digest
            sources[p.relative_to(ROOT).as_posix()]=digest
    print('Inspected Delft3D sources match original archive',flush=True)
    files={}
    paths=[p for p in BASE.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    paths.extend((ROOT/'desktop').glob('*177.py'))
    paths.extend([ROOT/'tests/test_desktop_pulse177.py',*OUT.glob('*.png'),
                  OUT/'REPORT.md',OUT/'DIAGNOSTICS.json',OUT/'GROIN_WAVES.json',OUT/'SERIES.npz'])
    for i,p in enumerate(paths):
        files[p.relative_to(ROOT).as_posix()]=sha(p)
        if i and i%2500==0:print('Hashed',i,'evidence files',flush=True)
    references={}
    for name in ('desktop_corner174','desktop_zero175','desktop_corner176'):
        p=OUT.parent/name/'MANIFEST.json';references[p.relative_to(ROOT).as_posix()]=sha(p)
    save(OUT/'MANIFEST.json',dict(boundary_diagnostic_complete=True,
        zero_viscosity_groin_fixed=False,production_changed=False,
        Delft3D_source_modified=False,Delft3D_runtime_modified=False,
        runtime_manifest_sha256=identity,Delft3D_source=sources,
        reference_manifests=references,files=files,
        accepted_run_root='validation/_pulse177/matched',
        excluded_runs='Initial Chezy 1e6 native controls fail uniform-state gate; retained outside matched. '
                      'Chezy45 drag control only diagnoses that input sensitivity.',
        harness_versions='Individual CHECK/INPUTS files record executed inputs and source hashes. '
                         'Later harness additions affect reporting or input-control labels only.'))
    print('Sealed',len(files),'files; no solver or boundary change',flush=True)


if __name__=='__main__':main()
