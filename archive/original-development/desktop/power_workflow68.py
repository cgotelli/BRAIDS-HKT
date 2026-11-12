"""Stage isolated power-workflow runtime, preserving every older reference."""
import hashlib,json,os,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'desktop/qualification/energy67/candidate_v3'
TARGET=ROOT/'desktop/qualification/power_workflow68/candidate_v3'
REPORT=ROOT/'validation/hydrokinetic_turbines/reports/power_workflow68'
CHANGED=['braids_core/river_turbine_surface.py','braids_core/sediment/river_turbine_inflow.py','braids_core/river_turbines.py','braids_core/river_project.py','braids_desktop/turbines.py','braids_desktop/turbine_surface.py','braids_desktop/results.py','braids_desktop/__init__.py']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def stage():
    TARGET.mkdir(parents=True,exist_ok=False);REPORT.mkdir(parents=True,exist_ok=True)
    old=read(BASE/'RUNTIME.json');files=dict(old['files'])
    for n,h in files.items():
        s=BASE/n;t=TARGET/n;t.parent.mkdir(parents=True,exist_ok=True);assert sha(s)==h,n
        if n in CHANGED:shutil.copy2(ROOT/n,t)
        else:os.link(s,t)
    for n in CHANGED:
        t=TARGET/n;t.parent.mkdir(parents=True,exist_ok=True)
        if not t.exists():shutil.copy2(ROOT/n,t)
        files[n]=sha(t)
    save(TARGET/'RUNTIME.json',dict(old,files=files));save(REPORT/('STAGE_'+TARGET.name+'.json'),dict(parent=sha(BASE/'RUNTIME.json'),candidate=sha(TARGET/'RUNTIME.json'),changed=CHANGED,installed_parent=sha(ROOT/'desktop/runtime/RUNTIME.json')))
    print(TARGET)
if __name__=='__main__':stage()
