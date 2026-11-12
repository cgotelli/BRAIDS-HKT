"""Retain final UI qualification and hash the bounded report artifacts."""
import shutil
from stage_grains212 import ROOT,OUT,sha,save
REPORT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_grains212'
for n in ('INSTALLED_EDITOR_CHECK.json','EDITOR_DIAGNOSTIC.json'):
    shutil.copy2(OUT/n,REPORT/n)
files={p.relative_to(REPORT).as_posix():sha(p) for p in sorted(REPORT.rglob('*'))
       if p.is_file() and p.name!='ARTIFACTS.json'}
save(REPORT/'ARTIFACTS.json',dict(files=files))
print('Sealed',len(files),'report artifacts')
