"""Gate the local release on retained scientific and packaged-runtime checks."""
import json,sys,xml.etree.ElementTree as ET
from pathlib import Path
from stage_formula190 import ROOT,TARGET,OUT
from stage_secondary185 import save,sha
from release_formula190 import RELEASE,verify
sys.path[:0]=[str(RELEASE),str(ROOT)]
import numpy as np
from braids_desktop.results import ResultsStore
REPORT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_transport190'

def main():
    verify(RELEASE)
    suite=ET.parse(OUT/'REGRESSION_RELEASE.xml').getroot().find('testsuite')
    assert int(suite.attrib['tests'])==180 and int(suite.attrib['failures'])==0 and int(suite.attrib['errors'])==0
    matched={}
    for law in ('ancey_recking_2023','recking_2010'):
        for backend in ('cpu','gpu'):
            label=law+'_'+backend
            candidate=ResultsStore(OUT/'runs/wet'/label/'full')
            packaged=ResultsStore(OUT/'release_qualification/runs/wet'/label/'full')
            exact={k:True for k in ('q','z','mass','water','bed_boundary')}
            for i in range(len(candidate.times)):
                a,b=candidate.read(i),packaged.read(i)
                for k in exact:exact[k]&=bool(np.array_equal(a.arrays[k],b.arrays[k]))
            assert all(exact.values());matched[label]=exact
    for backend in ('cpu','gpu'):
        smoke=json.loads((OUT/'final_smoke_v2'/(backend+'_CHECK.json')).read_text())
        assert smoke['runtime']==sha(RELEASE/'RUNTIME.json') and smoke['old_checkpoint_rejected']
        assert all(all(r.values()) for r in smoke['fresh_prepare_and_6_to_8s_restart_exact'].values())
    comparison=json.loads((REPORT/'COMPARISON.json').read_text())
    for r in comparison['cases'].values():
        assert r['maximum_backend_bed_mm']<1 and r['maximum_backend_depth_mm']<1
        for c in r['checks']:
            assert c['status']['status']=='complete' and all(c['restart_exact'].values())
            assert c['compiler_free'] and c['inlet_capacity_oracle']
    legacy=json.loads((OUT/'LEGACY_COMPARISON.json').read_text())
    assert len(legacy)==8 and all(all(v==0 for v in row.values()) for row in legacy.values())
    assert len(json.loads((OUT/'restart_rejection/CHECK.json').read_text())['checks'])==8
    proof=dict(qualified=True,scope='single-grain bedload; explicit CPU/GPU; short wet/dry AMR and restart',
        experimental_validation=False,candidate_runtime=sha(TARGET/'RUNTIME.json'),release_runtime=sha(RELEASE/'RUNTIME.json'),
        regression_tests=180,new_law_coupled_cases=8,legacy_coupled_cases=8,
        fresh_release_coupled_cases=4,release_fields_exact=matched,final_release_smoke='fresh prepare, 0..6 then 6..8 s, both laws/backends',
        legacy_fields_exact=True,sediment_guard_unchanged=True,source_controlled_metadata_only_release=True,
        figures={p.name:sha(p) for p in REPORT.glob('*.png')},Delft3D_modified=False,QGIS_changed=False)
    save(REPORT/'QUALIFICATION.json',proof)
    sources=[]
    for sub in ('ancey_recking2023','recking2010'):
        for p in (ROOT/'validation/vendor/bedload'/sub).rglob('*'):
            if p.is_file() and (p.suffix in ('.pdf','.gif') or p.name=='figure3-4.nb'):
                sources.append(dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p)))
    save(REPORT/'SOURCES.json',sources)
    print('QUALIFIED',proof['release_runtime'])
if __name__=='__main__':main()
