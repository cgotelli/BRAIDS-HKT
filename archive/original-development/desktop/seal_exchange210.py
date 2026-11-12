"""Seal the exchange milestone and verify previous evidence without modifying it."""
import hashlib,json,shutil,xml.etree.ElementTree as ET
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT/'validation/_exchange210'
REPORT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_exchange210'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    old_manifest=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_shallow209/ARTIFACTS.json'
    old=json.loads(old_manifest.read_text())['files']
    for i,(n,h) in enumerate(old.items(),1):
        assert (ROOT/n).is_file() and sha(ROOT/n)==h,n
        if i%2000==0:print(f'Verified {i}/{len(old)} previous209 artifacts',flush=True)
    installed=ROOT/'desktop/runtime';m=json.loads((installed/'RUNTIME.json').read_text())
    assert sha(installed/'RUNTIME.json')=='f88145dcf039c589557e9bb1acf7c6dacaf8b1afedec7952e3a533aa9f6ccb6b'
    for n,h in m['files'].items():assert sha(installed/n)==h,n
    print(f'Verified {len(m["files"])} installed payloads unchanged',flush=True)
    candidate=WORK/'candidate';cm=json.loads((candidate/'RUNTIME.json').read_text())
    assert sha(candidate/'RUNTIME.json')=='b240c08d228679e9c89e33b52e111d2edef8265bb3d6de2115bd46bbc91b9a34'
    for n,h in cm['files'].items():assert sha(candidate/n)==h,n
    files=('braids_core/sediment/jiang_cao_exchange.h','braids_core/sediment/jiang_cao_exchange.py',
           'braids_core/sediment/jiang_cao_wetdry.py','braids_core/river_jiang_cao.py','braids_core/sediment/jiang_cao_desktop.py')
    for n in files:
        if n not in files[-2:]:assert sha(ROOT/n)==sha(candidate/n),n
        else:assert sha(ROOT/n)==sha(ROOT/'validation/_startup208/candidate'/n),n
        dest=REPORT/'source'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(candidate/n,dest)
    tests={}
    for p in sorted(WORK.glob('*.xml')):
        suites=list(ET.parse(p).getroot().iter('testsuite'))
        counts={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
        if p.name in ('COMPONENT_V2.xml','REGRESSION_CHECKOUT.xml','REGRESSION_PACKAGED_FINAL.xml'):
            assert counts['failures']==counts['errors']==counts['skipped']==0
        tests[p.name]=counts;shutil.copy2(p,REPORT/p.name)
    paths=['tests/test_jiang_cao196.py','tests/test_jiang_cao209.py','tests/test_jiang_cao210.py','validation/RIVER_MORPHODYNAMICS_STATUS.md']
    paths += [p.relative_to(ROOT).as_posix() for p in (ROOT/'desktop').glob('*exchange210.py')]
    for n in paths:
        dest=REPORT/'source'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,dest)
    result=json.loads((REPORT/'COMPARISON.json').read_text())
    for case in ('strong','flume','front'):
        for backend in ('cpu','gpu'):
            check=result[case+'_'+backend]['checks']
            assert check['status']=='complete' and check['exact_restart'] and check['rejected_trials']==0
            if case!='strong':assert result[case+'_previous_'+backend]['all_saved_arrays_exact']
    traces={b:json.loads((WORK/('trace_'+b)/'CHECK.json').read_text()) for b in ('cpu','gpu')}
    assert all(v['calls_above_packing']>0 and v['actual_max']<.6 and v['old_checkpoint_rejected'] for v in traces.values())
    closure=dict(milestone=210,decision='finite-stock exchange through packing target qualified; research candidate not installed',
        candidate_runtime=sha(candidate/'RUNTIME.json'),installed=False,public_selection_unchanged=True,
        installed_runtime=sha(installed/'RUNTIME.json'),verified_installed_files=len(m['files']),
        previous209_manifest=sha(old_manifest),verified_previous209_files=len(old),tests=tests,traces=traces,
        next_milestone='Desktop delivery of qualified208-210 corrections in one named research bundle',
        immediate_next_step='promote verified resolver/adapter, build new package, qualify lifecycle and ordinary formulas before installation')
    (REPORT/'CLOSURE.json').write_text(json.dumps(closure,indent=2))
    entries={}
    for folder in (WORK,REPORT):
        for p in folder.rglob('*'):
            if p.is_file() and p.name!='ARTIFACTS.json' and '__pycache__' not in p.parts and p.suffix!='.pyc':
                entries[p.relative_to(ROOT).as_posix()]=sha(p)
                if len(entries)%2000==0:print(f'Sealed {len(entries)} current artifacts',flush=True)
    (REPORT/'ARTIFACTS.json').write_text(json.dumps(dict(schema=1,milestone=210,path_base='repository_root',files=entries),indent=2))
    print(json.dumps(dict(verified_previous=len(old),sealed=len(entries),tests=tests)),flush=True)
if __name__=='__main__':main()
