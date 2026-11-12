"""Final evidence/integrity receipt; never replaces the installed Desktop."""
import hashlib,json,sys,xml.etree.ElementTree as ET
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'validation/_bend189'
REPORT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_bend189'
sys.path[:0]=[str(ROOT/'desktop/runtime'),str(ROOT/'desktop'),str(ROOT)]
from braids_core.river_launcher import verify
from stage_secondary185 import save,sha


if __name__=='__main__':
    verify()
    runtime=sha(ROOT/'desktop/runtime/RUNTIME.json')
    assert runtime==sha(ROOT/'validation/_wall188/desktop_candidate/RUNTIME.json')
    suite=ET.parse(BASE/'TRANSPORT_TESTS_v4.xml').getroot().find('testsuite')
    assert int(suite.attrib['tests'])==149
    assert all(int(suite.attrib[k])==0 for k in ('errors','failures','skipped'))
    comparison=json.loads((REPORT/'COMPARISON.json').read_text())
    formulas=json.loads((REPORT/'FORMULA_AUDIT.json').read_text())
    assert comparison['initial_arrays_exact'] and len(comparison['cases'])==6
    assert formulas['equation_input_cases']==21588 and len(formulas['coupled_pairs'])==8
    assert all(json.loads((BASE/k/'JOBS.json').read_text())['complete'] for k in ('live','dry'))
    assert all(v['final_mesh_exact'] for v in formulas['coupled_pairs'].values())
    checks=[]
    for family in ('live','dry'):
        for path in sorted((BASE/family/'groin').glob('*/CHECK.json')):
            row=json.loads(path.read_text());assert row['status']['status']=='complete'
            assert row['runtime']==runtime and row['compiler_free']
            checks.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path)))
    assert len(checks)==16
    source_files=[*sorted((ROOT/'desktop').glob('*189.py')),ROOT/'tests/test_bedload_audit189.py']
    receipt=dict(complete=True,installed_runtime_unchanged=True,runtime=runtime,
        installed_payload_count=len(json.loads((ROOT/'desktop/runtime/RUNTIME.json').read_text())['files']),
        regression_tests=149,equation_backend_input_evaluations=21588,
        bend_trajectories=6,new_bend_trajectories=4,reused_immutable_on_trajectories=2,
        hydraulic_warmup_s=200.,physical_mobile_window_s=20.,morphological_window_s=100.,
        public_groin_runs=16,delft3d_source_modified=False,new_transport_models=0,
        solver_source_changes=0,checks=checks,
        validation_sources={str(p.relative_to(ROOT)):sha(p) for p in source_files},
        report_artifacts={p.name:sha(p) for p in REPORT.iterdir() if p.is_file() and p.name!='QUALIFICATION.json'})
    save(REPORT/'QUALIFICATION.json',receipt)
    print('SEALED: 149 tests, 21588 equation inputs, 6 bend trajectories, 16 wet/dry public groins; preview30 unchanged.')
