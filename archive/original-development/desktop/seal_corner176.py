"""Seal the isolated corner audit without promoting its failed candidates."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'desktop')]
from stage_wall172 import save, sha

BASE = ROOT / 'validation/_quadrature176'
OUT = ROOT / 'validation/sediment/river_morphodynamics/reports/desktop_corner176'


def main():
    runtime = ROOT / 'desktop/runtime'
    identity = sha(runtime / 'RUNTIME.json')
    assert identity == '8bb07b4e4877411e0035060842a47feecc41b1d5e04f14cb389407c38c9349c9'
    for name, digest in json.loads((runtime / 'RUNTIME.json').read_text())['files'].items():
        assert sha(runtime / name) == digest, name
    sources = {}
    folder = ROOT / 'resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/flow2d3d_kernel/src'
    with zipfile.ZipFile(ROOT / 'resources/Delft3D-main.zip') as archive:
        for name in ('compute/mom_fls.f90', 'compute/ulim.f90', 'compute/slim.f90',
                     'compute/upwhu.f90', 'compute/cucbp.f90', 'timedep/incbc.f90',
                     'main/trisol.f90'):
            path = folder / name
            digest = sha(path)
            suffix = path.relative_to(ROOT / 'resources').as_posix()
            matches = [n for n in archive.namelist() if n == suffix or n.endswith('/' + suffix)]
            assert len(matches) == 1, (name, matches)
            assert hashlib.sha256(archive.read(matches[0])).hexdigest() == digest, name
            sources[path.relative_to(ROOT).as_posix()] = digest
    files = {p.relative_to(ROOT).as_posix(): sha(p) for p in BASE.rglob('*')
             if p.is_file() and '__pycache__' not in p.parts}
    for path in (ROOT / 'desktop').glob('*176.py'):
        files[path.relative_to(ROOT).as_posix()] = sha(path)
    for path in [ROOT / 'tests/test_desktop_corner176.py',
                 ROOT / 'desktop/corner_candidate167.py',
                 ROOT / 'desktop/corner_exact174.py',
                 *OUT.glob('*.png'), OUT / 'REPORT.md', OUT / 'DIAGNOSTICS.json']:
        files[path.relative_to(ROOT).as_posix()] = sha(path)
    references = {}
    for name in ('desktop_corner174', 'desktop_zero175'):
        path = OUT.parent / name / 'MANIFEST.json'
        references[path.relative_to(ROOT).as_posix()] = sha(path)
    save(OUT / 'MANIFEST.json', dict(
        diagnostic_complete=True, hydraulic_fix_qualified=False,
        production_changed=False, Delft3D_source_modified=False,
        Delft3D_runtime_modified=False, runtime_manifest_sha256=identity,
        Delft3D_source=sources, reference_manifests=references, files=files,
        caveat='Final harness sources retained; individual CHECK files identify executed '
               'versions. regular_v2 adds tracing and reproduces regular rates exactly. '
               'All candidates remain isolated research; no installed runtime changes.'))
    print('Sealed', len(files), 'files; installed runtime and inspected Delft3D sources unchanged', flush=True)


if __name__ == '__main__':
    main()
