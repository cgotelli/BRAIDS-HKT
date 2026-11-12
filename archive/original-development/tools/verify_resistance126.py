"""Verify bounded qualification receipts and unchanged delivered runtime."""
from pathlib import Path
import sys,json,hashlib,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from braids_core import river_worker
from braids_gpu.atomic_output import write_json_atomic


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    base=ROOT/'validation/_resistance126'
    expected='90e50015d3dc356c60731789c561af8fdace3eb967722edd5bcc139350dcda74'
    assert river_worker.source_identity()==expected
    installed=Path('C:/Users/STREEM/AppData/Roaming/QGIS/QGIS3/profiles/default/python/plugins/braids_qgis')
    manifest=json.loads((installed/'MANIFEST.json').read_text())
    for name,digest in manifest['files'].items():assert sha(installed/name)==digest,name
    archive=ROOT/'dist/unified-alpha123-final/BRAIDS-0.1.0-alpha.123-QGIS-windows-x86_64.zip'
    assert sha(archive)=='78c763e1ee9a7c9e9e4a0ea03a98680b0cbbd04032cd0590b3b0138551329261'
    tests={}
    for name in ('COUPLED_v2.xml','PUBLIC.xml','ROUGHNESS_v2.xml','REGRESSION.xml'):
        cases=list(ET.parse(base/name).iter('testcase'))
        if name=='PUBLIC.xml':cases=[c for c in cases if c.get('classname','').endswith('test_river_public_hydraulic_matrix')]
        assert cases,name
        for c in cases:assert not list(c), (name,c.attrib,ET.tostring(c).decode())
        tests[name]=len(cases)
    assert sum(tests.values())==67,tests
    sources=[*sorted((ROOT/'validation/hydraulics/resistance126').glob('*.py')),
        ROOT/'validation/hydraulics/resistance126/cell_drag.h',
        ROOT/'tools/profile_resistance126.py',ROOT/'tools/analyze_resistance126.py',Path(__file__),
        ROOT/'tests/test_resistance126.py',ROOT/'tests/test_resistance126_coupled.py',
        ROOT/'tests/test_river_roughness_integration.py',
        ROOT/'resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/flow2d3d_kernel/src/compute/taubot.f90']
    data=dict(tests=tests,passing_checks=sum(tests.values()),production_runtime_identity=expected,
        production_unchanged=True,plugin_unchanged=True,installed_version=manifest['version'],
        verified_installed_files=len(manifest['files']),release_archive_sha256=sha(archive),
        source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in sources},
        retained_failed_attempts={'COUPLED.xml':'duplicate research native speed-export installation; corrected test harness',
            'PUBLIC.xml':'18 boundary checks passed; roughness fixture then failed obsolete cell-count assertion; final geometry-area check rerun in ROUGHNESS_v2.xml'},
        scope='Qualified research candidate. No public method/default or QGIS replacement in milestone 126.')
    write_json_atomic(base/'INTEGRITY.json',data);print(json.dumps(data,indent=2))


if __name__=='__main__':main()
