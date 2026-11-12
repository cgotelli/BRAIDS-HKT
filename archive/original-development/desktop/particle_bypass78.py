"""Stage the qualified power-planning extension without touching frozen files."""
import hashlib,json,os,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'desktop/runtime'
TARGET=ROOT/'desktop/qualification/particle_bypass78/candidate'
REPORT=ROOT/'validation/hydrokinetic_turbines/reports/particle_bypass78'
CHANGED=['braids_desktop/visual_fields.py','braids_desktop/__init__.py']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,v):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def stage():
    TARGET.mkdir(parents=True,exist_ok=False);old=read(BASE/'RUNTIME.json');files=dict(old['files'])
    for name,h in files.items():
        s=BASE/name;t=TARGET/name;t.parent.mkdir(parents=True,exist_ok=True)
        if name in CHANGED:shutil.copy2(ROOT/name,t)
        else:os.link(s,t)
    for name in CHANGED:
        t=TARGET/name;t.parent.mkdir(parents=True,exist_ok=True)
        if not t.exists():shutil.copy2(ROOT/name,t)
        files[name]=sha(t)
    save(TARGET/'RUNTIME.json',dict(old,files=files));save(REPORT/'STAGE.json',dict(installed_parent=sha(BASE/'RUNTIME.json'),candidate=sha(TARGET/'RUNTIME.json'),changed=CHANGED))
def environment():
    sys.path[:0]=[str(TARGET),str(ROOT)];sys.dont_write_bytecode=True
    global dlls
    dlls=[os.add_dll_directory(str(p)) for p in [Path(sys.base_prefix),Path(sys.base_prefix)/'Library/bin'] if p.is_dir()] if os.name=='nt' else []
    from braids_core.sediment import portable_river_runtime
    portable_river_runtime.install(read(TARGET/'PRECOMPILED.json'),TARGET)
if __name__=='__main__':stage()
