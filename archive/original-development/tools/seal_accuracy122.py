"""Deliver new example files and seal the completed alpha122 evidence."""
from pathlib import Path
import argparse,hashlib,json,shutil,zipfile,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
BASE=ROOT/'validation/_accuracy122'
REPORT=ROOT/'validation/sediment/river_morphodynamics/reports/unified_qgis_v1/accuracy122'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2),encoding='utf-8')

def deliver():
    from braids_core.river_project import load_project
    original=Path('C:/Users/STREEM/Documents/BRAIDS/1m.json')
    expected=sha(BASE.parent/'_axis_boundary120/ORIGINAL.json')
    assert sha(original)==expected,'Original user project changed; inspect before delivery'
    assert json.loads((BASE/'INSTALL.json').read_text())['status']=='installed'
    checks=json.loads((BASE/'qgis_installed/CHECKS.json').read_text())['checks'];assert all(checks.values())
    target=original.parent/'alpha122';target.mkdir(exist_ok=False)
    files=[]
    for b in ('cpu','gpu'):
        source=BASE/f'projects/River1m_{b}.json';load_project(source)
        dest=target/f'1m_axis_aligned_{b}.json';shutil.copy2(source,dest)
        files.append(dict(path=str(dest),sha256=sha(dest)))
    shutil.copy2(BASE.parent/'_axis_boundary120/GEOMETRY.json',target/'GEOMETRY_CHANGES.json')
    write(target/'READ_ME.json',dict(version='0.1.0-alpha.122',schema='braids.river-project.v16',
        scope='New explicitly derived axis-aligned projects; not migration of the original',
        sediment_enabled=False,mesh='Static terrain-only 4/2/1 m, whole-cell polygon',
        original_project=str(original),original_sha256=expected,
        instructions=['Open the CPU or GPU JSON in the BRAIDS tab-based dock and prepare again.',
                      'These retain the 60 s run clock. Only the first 3.75 s has been paired against the fine temporal control.',
                      'The geometry is different: read GEOMETRY_CHANGES.json. Original file remains untouched.',
                      'Local temporal tolerances control step-pair differences, not global accumulated error.',
                      'Keep the existing external Python/CuPy environment for GPU execution.'],
        files=files,report=str(REPORT/'REPORT.md')))
    write(BASE/'DELIVERY.json',dict(directory=str(target),files=files,original_sha256=sha(original)))
    print(target)

def seal():
    paths=['README.md','validation/RIVER_MORPHODYNAMICS_STATUS.md',
        'validation/sediment/river_morphodynamics/UNIFIED_QGIS_PLAN.md',
        'braids_core/sediment/river_midpoint.py','braids_core/sediment/river_midpoint_iteration.py',
        'braids_core/sediment/river_midpoint_accuracy.py','braids_core/sediment/shoreline_solver.py',
        'braids_core/sediment/current_river_solver.py','braids_core/sediment/adaptive_river_worker.py',
        'braids_core/sediment/river_capacity.py','braids_core/sediment/river_suspension.h',
        'braids_core/river_project.py','braids_core/boundaries/axis_aligned.py',
        'braids_qgis/gui/river_dock.py','braids_qgis/core/river_runner.py','braids_qgis/metadata.txt',
        'braids_qgis/FIRST_RELEASE.md','braids_qgis/MODEL_SETUP.md',
        'tools/accuracy122.py','tools/nonlinear122.py','tools/deliver_accuracy122.py',
        'tools/analyze_accuracy122.py','tools/run_qgis_accuracy122.py','tools/seal_accuracy122.py',
        'tools/build_unified_qgis.py','tools/install_unified_qgis.py','tools/qualify_unified_release.py',
        'tests/test_accuracy122.py','tests/test_river_midpoint.py','tests/test_river_midpoint_retry_cap.py',
        'tests/test_river_project.py','tests/test_river_public_hydraulic_matrix.py',
        'tests/test_river_public_boundaries.py','tests/support/qgis_accuracy122.py']
    with zipfile.ZipFile(REPORT/'SOURCES.zip','x',zipfile.ZIP_DEFLATED) as z:
        for rel in paths:z.write(ROOT/rel,rel)
    evidence=['BEFORE.zip','MAINTAINED_TESTS.xml','BOUNDARY_TESTS.xml','PUBLIC_BOUNDARY_TESTS.xml','PACKAGER_TESTS.xml',
        'INPUT_CHANGES.json','release_record/RECORD.json','release_record/BINDINGS.json','release_record/PACKAGE.json',
        'portable_staged/CHECKS.json','qgis_staged/CHECKS.json','qgis_staged/LAUNCHER.json',
        'qgis_installed/CHECKS.json','qgis_installed/LAUNCHER.json','INSTALL.json','DELIVERY.json']
    checks={}
    for relative in ('portable_staged','qgis_staged','qgis_installed'):
        d=json.loads((BASE/relative/'CHECKS.json').read_text());assert all(d['checks'].values())
        checks[relative]=len(d['checks'])
    for relative in ('qgis_staged','qgis_installed'):
        assert json.loads((BASE/relative/'LAUNCHER.json').read_text())['returncode']==0
    write(REPORT/'MANIFEST.json',dict(
        schema='braids.accuracy122-delivery.v1',source_files={p:sha(ROOT/p) for p in paths},
        evidence={str(BASE/p):sha(BASE/p) for p in evidence},checks=checks,
        report_files={p.name:sha(p) for p in REPORT.iterdir() if p.is_file() and p.name!='MANIFEST.json'},
        release=json.loads((BASE/'INSTALL.json').read_text()),delivery=json.loads((BASE/'DELIVERY.json').read_text())))
    print(REPORT/'MANIFEST.json')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['deliver','seal']);a=p.parse_args()
    (deliver if a.action=='deliver' else seal)()
