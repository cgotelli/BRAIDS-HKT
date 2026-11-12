"""Seal the bounded settled-flow/BC diagnostic, retaining failed screens."""
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'desktop/runtime'),str(ROOT),str(ROOT/'desktop')]
from stage_wall172 import sha,save
from review_hydraulics162 import verify_native
BASE=ROOT/'validation/_corner174'
OUT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_corner174'


def main():
    e=json.loads((OUT/'MIXING_AND_BOUNDARIES.json').read_text())
    runtime=ROOT/'desktop/runtime';identity=sha(runtime/'RUNTIME.json')
    assert identity==e['runtime_sha256']=='8bb07b4e4877411e0035060842a47feecc41b1d5e04f14cb389407c38c9349c9'
    for name,digest in json.loads((runtime/'RUNTIME.json').read_text())['files'].items():assert sha(runtime/name)==digest,name
    references={};read_only={}
    for folder in (BASE/'delft',BASE/'long/delft',BASE/'mixing/delft'):
        references[str(folder.relative_to(ROOT))]=verify_native(folder)
        read_only[str((folder/'trih-sim_P057.nc').relative_to(ROOT))]=sha(folder/'trih-sim_P057.nc')
    names=('incbc.f90','rdbcg.f90','cucbp.f90','upwhu.f90','uzd.f90','vihrov.f90','mom_fls.f90','slim.f90')
    with zipfile.ZipFile(ROOT/'resources/Delft3D-main.zip') as archive:
        for name in names:
            paths=[p for p in archive.namelist() if p.endswith('/'+name)];assert len(paths)==1,(name,paths)
            p=ROOT/'resources'/paths[0];content=archive.read(paths[0]);assert p.read_bytes()==content
            read_only[str(p.relative_to(ROOT))]=hashlib.sha256(content).hexdigest()
    suites=ET.parse(BASE/'CHECKS.xml').getroot().findall('testsuite')
    tests=sum(int(s.attrib['tests']) for s in suites)
    assert all(int(s.attrib[k])==0 for s in suites for k in ('failures','errors','skipped'))
    assert tests>=226
    assert all(r['passed'] for rows in e['mix300']['settling'].values() for r in rows)
    assert not e['zero600']['settling']['BRAIDS'][-1]['passed']
    outputs={}
    for folder in (BASE,BASE/'long',BASE/'mixing',BASE/'inlet_total'):
        for backend in ('cpu','gpu'):
            parent=folder/backend
            for name in ('project.json','WARMUP.json','MOBILE.json'):
                p=parent/name
                if p.exists():outputs[str(p.relative_to(ROOT))]=sha(p)
            for kind in ('warmup','mobile'):
                run=parent/kind
                if not run.exists():continue
                for name in ('PROJECT.json','RUN_CONTEXT.json','PROGRESS.json','DIAGNOSTICS.json','checkpoint.npz'):
                    p=run/name
                    if p.exists():outputs[str(p.relative_to(ROOT))]=sha(p)
                for p in (run/'frames').rglob('*'):
                    if p.is_file():outputs[str(p.relative_to(ROOT))]=sha(p)
    for category in ('probes_v2','potential','boundary_probes_v2'):
        for backend in ('cpu','gpu'):
            for p in (BASE/category/backend).iterdir():
                if p.is_file():outputs[str(p.relative_to(ROOT))]=sha(p)
    sources=list((ROOT/'desktop').glob('*174.py'))+[ROOT/'tests/test_desktop_corner174.py',ROOT/'validation/RIVER_MORPHODYNAMICS_STATUS.md']
    data=dict(scope='Known-answer corner checks, matched mixing, measured hydraulic settling, boundary audit, short post-spinup MPM',
        settled_comparison_completed=True,jet_scour_equivalence=False,production_changed=False,Desktop_runtime_changed=False,
        Delft3D_source_modified=False,Delft3D_runtime_modified=False,QGIS_modified=False,
        installed_version='0.1.0-preview27',runtime_manifest_sha256=identity,
        tests=dict(passed=tests,new=tests-185,retained=185,failures=0,errors=0,skipped=0,receipt_sha256=sha(BASE/'CHECKS.xml')),
        failed_screens=dict(zero_viscosity_steady_after_600s=False,zero_viscosity_backend_depth_1mm_at_600s=False,
                            singular_corner_accuracy=False,settled_scour_extrema_equivalence=False),
        native_references=references,native_read_only=read_only,outputs=outputs,
        sources={str(p.relative_to(ROOT)):sha(p) for p in sources},
        artifacts={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='MANIFEST.json'},
        figures_visually_checked=['SETTLING.png','WAKE_HISTORY.png','MIXING_SETTLING.png','SETTLED_SECTIONS.png',
            'SETTLED_BEDLOAD.png','BEDLOAD_TRENDS.png','BOUNDARY_AUDIT.png','INLET_CONTRAST.png'],
        next_milestone='Resolve the settled sharp-corner turning/head-loss and erosion-concentration discrepancy; qualify any demonstrated correction on CPU/GPU.',
        immediate_next_step='Freeze the 300 s matched-viscosity states and compare volume-consistent pressure/advection/viscosity budgets across the groin nose, including native staggering; then one short paired confirmation, without MPM tuning.')
    save(OUT/'MANIFEST.json',data)
    print(json.dumps(dict(tests=data['tests'],settled_comparison_completed=True,production_changed=False,
                         jet_scour_equivalence=False,authenticated_output_files=len(outputs)),indent=2))


if __name__=='__main__':main()
