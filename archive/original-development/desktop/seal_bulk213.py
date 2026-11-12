"""Record installed identity and preserve the exact qualification sources."""
import shutil,json
from stage_bulk213 import ROOT,OUT,BASE,FILES,sha,save,verify
REPORT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_bulk213'
verify(BASE)
record=json.loads((REPORT/'INSTALL.json').read_text())
assert sha(BASE/'RUNTIME.json')==record['runtime']
editor=json.loads((OUT/'INSTALLED_EDITOR.json').read_text())
assert editor['runtime']==record['runtime'] and editor['project_roundtrip']
for name in ('INSTALLED_EDITOR.json','EDITOR.json'):
    shutil.copy2(OUT/name,REPORT/name)
for p in (OUT/'figures').glob('*.png'):shutil.copy2(p,REPORT/'figures'/p.name)
for name in FILES:
    target=OUT/'source'/name;target.parent.mkdir(parents=True,exist_ok=True)
    assert sha(ROOT/name)==sha(BASE/name)
    shutil.copy2(BASE/name,target)
sources={name:sha(BASE/name) for name in FILES}
save(REPORT/'SOURCES.json',sources)
shutil.copy2(BASE/'RUNTIME.json',REPORT/'RUNTIME.json')
shutil.copy2(ROOT/'desktop/PROVENANCE.json',REPORT/'PROVENANCE.json')
artifacts={p.relative_to(REPORT).as_posix():sha(p) for p in REPORT.rglob('*')
           if p.is_file() and p.name!='ARTIFACTS.json'}
save(REPORT/'ARTIFACTS.json',artifacts)
assert all(sha(REPORT/n)==h for n,h in artifacts.items())
print('Sealed',len(artifacts),'artifacts; installed preview38 verified',flush=True)
