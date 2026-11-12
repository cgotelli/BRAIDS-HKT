"""Seal bounded evidence and verify prior references/installed payloads unchanged."""
import hashlib,json,shutil,xml.etree.ElementTree as ET
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT/'validation/_startup208'
REPORT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_startup208'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    previous=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_bundle207/ARTIFACTS.json'
    files=json.loads(previous.read_text())['files'];bad=[n for n,h in files.items() if not (ROOT/n).is_file() or sha(ROOT/n)!=h]
    assert not bad,bad
    runtime=ROOT/'desktop/runtime';m=json.loads((runtime/'RUNTIME.json').read_text())
    assert sha(runtime/'RUNTIME.json')=='f88145dcf039c589557e9bb1acf7c6dacaf8b1afedec7952e3a533aa9f6ccb6b'
    for n,h in m['files'].items():assert sha(runtime/n)==h,n
    tests={}
    for n in ('REGRESSION_FOCUSED.xml','REGRESSION_OPERATORS.xml','REGRESSION_PACKAGED_FINAL.xml'):
        root=ET.parse(WORK/n).getroot();suites=list(root.iter('testsuite'))
        counts={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
        assert counts['failures']==counts['errors']==counts['skipped']==0,counts
        tests[n]=counts;shutil.copy2(WORK/n,REPORT/n)
    shutil.copy2(WORK/'REGRESSION_PACKAGED.xml',REPORT/'INITIAL_PACKAGED_COLLECTION_ERRORS.xml')
    source=['braids_core/sediment/jiang_cao_adjustment.py','braids_core/sediment/jiang_cao_desktop.py',
            'tests/test_jiang_cao208.py']+[p.relative_to(ROOT).as_posix() for p in (ROOT/'desktop').glob('*startup208.py')]
    for n in source:
        dest=REPORT/'source'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,dest)
    (REPORT/'CLOSURE.json').write_text(json.dumps(dict(previous_207_files_verified=len(files),
        installed_runtime_unchanged=sha(runtime/'RUNTIME.json'),installed_files_verified=len(m['files']),tests=tests,
        numerical_candidate=json.loads((WORK/'STAGE.json').read_text()),installed=False,
        milestone='bounded startup audit and targeted numerical correction complete; long braiding admission NOT passed'),indent=2))
    entries={}
    for directory in (REPORT,WORK):
        for p in directory.rglob('*'):
            if p.is_file() and p.name!='ARTIFACTS.json' and '__pycache__' not in p.parts and p.suffix!='.pyc':
                entries[p.relative_to(ROOT).as_posix()]=sha(p)
    (REPORT/'ARTIFACTS.json').write_text(json.dumps(dict(schema=1,milestone=208,path_base='repository_root',files=entries),indent=2))
    print(json.dumps(dict(verified_previous=len(files),sealed=len(entries),tests=tests)))
if __name__=='__main__':main()
