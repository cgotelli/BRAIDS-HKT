"""Seal reproducible integration evidence and example lifecycle records."""
import sys,json,shutil,xml.etree.ElementTree as ET
from stage_bundle207 import ROOT,OUT,TARGET,FILES,sha,save,verify
REPORT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_bundle207'
runtime=ROOT/'desktop/runtime';verify(runtime);assert sha(runtime/'RUNTIME.json')==sha(TARGET/'RUNTIME.json')
sys.path[:0]=[str(runtime),str(ROOT)]
from braids_core.river_folder import record_event
from braids_desktop.services import update_project_session
for name in ('moving_front','protected_bank'):
    folder=ROOT/'desktop/examples/jiang_cao207'/name
    for i,action in enumerate(('prepare','run')):
        logfile=folder/'history'/f'qualification207_{action}.log';assert not logfile.exists()
        shutil.copy2(OUT/'logs'/f'installed_{name}_{i}.log',logfile)
        record_event(folder,'desktop_worker',action=action,status='complete',prepared='prepared/qualification207',
            output=('prepared/' if action=='prepare' else 'runs/')+'qualification207',log=logfile.relative_to(folder).as_posix(),
            qualification='bounded research integration207')
    update_project_session(folder,prepared='prepared/qualification207',results='runs/qualification207')
old=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_startup206'
oldfiles=json.loads((old/'ARTIFACTS.json').read_text())['files']
for name,h in oldfiles.items():assert sha(old/name)==h,name
tests=['tests/test_jiang_cao207.py','tests/test_bedload_audit189.py','tests/test_desktop_bedload160.py','tests/test_grain_diameter191.py','tests/test_jiang_cao195.py']
scripts=[p.relative_to(ROOT).as_posix() for p in (ROOT/'desktop').glob('*bundle207.py')]
for name in (*FILES,*tests,*scripts):
    dst=REPORT/'source'/name;dst.parent.mkdir(parents=True,exist_ok=True);assert not dst.exists();shutil.copy2(ROOT/name,dst)
for src,dst in [(OUT/'REGRESSION_FINAL.xml','REGRESSION_FINAL.xml'),(OUT/'PACKAGED_REGRESSION_v3.xml','PACKAGED_REGRESSION.xml'),
        (OUT/'REGRESSION.xml','INITIAL_CATALOG_FAILURES.xml'),(OUT/'ADAPTER_v3.xml','INITIAL_VECTOR_ROUNDOFF_FAILURES.xml'),
        (TARGET/'RUNTIME.json','RUNTIME.json'),(OUT/'STAGE.json','STAGE.json')]:
    assert not (REPORT/dst).exists();shutil.copy2(src,REPORT/dst)
closure=dict(runtime=sha(runtime/'RUNTIME.json'),verified206_files=len(oldfiles),
    tests=894,packaged_tests=353,public_backend_controls=16,hydraulic_boundary_controls=24,
    cases=8,failed_oblique_dry_controls=['oblique_cpu','oblique_gpu','oblique_dry_fine_cpu'],
    installed_examples=2,experimental_validation=False,unrestricted_dry_start=False)
save(REPORT/'CLOSURE.json',closure)
# Retain identities for the actual inputs, numerical output, failed controls and
# logs in the validation workspace, without copying duplicate runtime payloads.
evidence={}
for folder in ('release_v3','release_controls_v3','boundaries_v3','baseline','logs'):
    for p in sorted((OUT/folder).rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc':evidence[p.relative_to(ROOT).as_posix()]=sha(p)
for p in sorted(REPORT.rglob('*')):
    if p.is_file() and p.name!='ARTIFACTS.json':evidence[p.relative_to(ROOT).as_posix()]=sha(p)
save(REPORT/'ARTIFACTS.json',dict(schema=1,milestone=207,path_base='repository_root',files=evidence))
print('Sealed',len(evidence),'artifacts; historical206',len(oldfiles),'verified',flush=True)
