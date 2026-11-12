"""Seal the installed release only after its own public runs and UI tests pass."""
import json,sys,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'validation/_wall188'
RUNTIME=ROOT/'desktop/runtime'
REPORT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_wall188'
sys.path[:0]=[str(RUNTIME),str(ROOT)]
from stage_secondary185 import sha,save
from braids_desktop.results import ResultsStore
import braids_desktop


def main():
    install=json.loads((OUT/'INSTALL.json').read_text())
    manifest=json.loads((RUNTIME/'RUNTIME.json').read_text())
    assert sha(RUNTIME/'RUNTIME.json')==install['installed']
    for name,digest in manifest['files'].items():assert sha(RUNTIME/name)==digest,name
    assert Path(braids_desktop.__file__).resolve().is_relative_to(RUNTIME.resolve())
    assert braids_desktop.__version__=='0.1.0-preview30'
    for name in ('forms.py','river_boundaries.py','__init__.py'):
        assert (ROOT/'braids_desktop'/name).read_text(encoding='utf-8')==(RUNTIME/'braids_desktop'/name).read_text(encoding='utf-8'),name
    runs={}
    for backend in ('cpu','gpu'):
        root=OUT/f'installed/public/smart_jaggi_{backend}'
        check=json.loads((root/'CHECK.json').read_text())
        assert check['runtime']==install['installed'] and check['status']['status']=='complete'
        assert all(check['restart']['exact'].values()) and check['compiler_free']
        actual=ResultsStore(root/'full').read(-1)
        reference=ResultsStore(OUT/f'smoke/fixed/smart_jaggi_{backend}/full').read(-1)
        assert actual.arrays.keys()==reference.arrays.keys()
        exact={k:bool(np.array_equal(actual.arrays[k],reference.arrays[k])) for k in actual.arrays}
        assert all(exact.values()),(backend,exact)
        runs[backend]=dict(status='complete',candidate_arrays_exact=exact,restart_exact=check['restart']['exact'],
            water_residual_m3=check['water_residual_m3'],sediment_residual_kg=check['sediment_residual_kg'])
    suite=ET.parse(OUT/'TESTS_INSTALLED.xml').getroot().find('testsuite')
    assert int(suite.attrib['tests'])==130 and all(int(suite.attrib[k])==0 for k in ('errors','failures','skipped'))
    save(REPORT/'DELIVERY.json',dict(version=braids_desktop.__version__,runtime=install['installed'],installed=True,
        payloads_verified=len(manifest['files']),installed_tests=130,installed_runs=runs,
        old_release_backup=install['backup'],user_projects_modified=False,QGIS_modified=False,Delft3D_modified=False,
        distribution_scope='Local Windows preview; not a clean-PC installer qualification'))
    print('INSTALLED RELEASE VERIFIED',braids_desktop.__version__,len(manifest['files']),'payloads, 130 tests, paired exact runs')


if __name__=='__main__':main()
