"""Freeze candidate evidence, retaining failures and verifying previous inputs."""
import hashlib,json,shutil,xml.etree.ElementTree as ET
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT/'validation/_shallow209'
REPORT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_shallow209'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    previous=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_startup208/ARTIFACTS.json'
    old=json.loads(previous.read_text())['files']
    for i,(n,h) in enumerate(old.items(),1):
        assert (ROOT/n).is_file() and sha(ROOT/n)==h,n
        if i%2000==0:print(f'Verified {i}/{len(old)} previous208 artifacts',flush=True)
    installed=ROOT/'desktop/runtime';m=json.loads((installed/'RUNTIME.json').read_text())
    assert sha(installed/'RUNTIME.json')=='f88145dcf039c589557e9bb1acf7c6dacaf8b1afedec7952e3a533aa9f6ccb6b'
    for n,h in m['files'].items():assert sha(installed/n)==h,n
    # Public selector/adapter are restored, not promoted from the research trial.
    for n in ('braids_core/river_jiang_cao.py','braids_core/sediment/jiang_cao_desktop.py'):
        assert sha(ROOT/n)==sha(ROOT/'validation/_startup208/candidate'/n),n
    for n in ('braids_core/river_jiang_cao.py','braids_core/sediment/jiang_cao_desktop.py','braids_core/sediment/jiang_cao_wetdry.py'):
        assert sha(REPORT/'source'/n)==sha(WORK/'candidate_v3'/n),n
    tests={}
    for name in ('REGRESSION_CHECKOUT_FINAL.xml','REGRESSION_PACKAGED_V3.xml','REGRESSION_PACKAGED.xml','REGRESSION_PACKAGED_V2.xml','REGRESSION_NEW.xml'):
        suites=list(ET.parse(WORK/name).getroot().iter('testsuite'))
        counts={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
        if name in ('REGRESSION_CHECKOUT_FINAL.xml','REGRESSION_PACKAGED_V3.xml'):
            assert counts['failures']==counts['errors']==counts['skipped']==0
        tests[name]=counts;shutil.copy2(WORK/name,REPORT/name)
    for n in ['tests/test_jiang_cao209.py']+[p.relative_to(ROOT).as_posix() for p in (ROOT/'desktop').glob('*shallow209.py')]:
        dest=REPORT/'source'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,dest)
    result=json.loads((REPORT/'COMPARISON.json').read_text())
    for name in ('cpu','gpu'):
        assert result['final_candidate_flume'][name]['checks']['exact_restart']
        assert result['front'][name]['exact_restart'] and result['front'][name]['regrids']==5
        assert result['oblique_'+name]['status']=='failed'
    closure=dict(milestone=209,decision='bounded flume treatment qualified; general promotion withheld for strong dry-inlet target-domain restriction',
        installed=False,public_selection_unchanged=True,installed_runtime=sha(installed/'RUNTIME.json'),
        verified_installed_files=len(m['files']),verified_previous208_files=len(old),tests=tests,
        candidate_runtime=sha(WORK/'candidate_v3/RUNTIME.json'),ordinary_native_kernels_changed=False,
        next_milestone='finite-stock source integration across ce=B; no clipping of capacity or actual concentration',
        immediate_next_step='frozen-state ODE controls below, at and above B; alpha transitions and stock exhaustion')
    (REPORT/'CLOSURE.json').write_text(json.dumps(closure,indent=2))
    entries={}
    for folder in (WORK,REPORT):
        for p in folder.rglob('*'):
            if p.is_file() and p.name!='ARTIFACTS.json' and '__pycache__' not in p.parts and p.suffix!='.pyc':
                entries[p.relative_to(ROOT).as_posix()]=sha(p)
                if len(entries)%3000==0:print(f'Sealed {len(entries)} current artifacts',flush=True)
    (REPORT/'ARTIFACTS.json').write_text(json.dumps(dict(schema=1,milestone=209,path_base='repository_root',files=entries),indent=2))
    print(json.dumps(dict(verified_previous=len(old),sealed=len(entries),tests=tests)),flush=True)
if __name__=='__main__':main()
