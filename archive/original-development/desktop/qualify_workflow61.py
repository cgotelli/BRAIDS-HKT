"""Isolate the wake/profile/saved-state update from retained release evidence."""
import argparse,hashlib,json,os,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'desktop/runtime';OUT=ROOT/'desktop/qualification/workflow61';TARGET=OUT/'candidate'
REPORT=ROOT/'validation/hydrokinetic_turbines/reports/desktop_workflow61'
FILES=['braids_core/boundaries/cut_cells.py','braids_core/river_turbines.py','braids_core/river_project.py','braids_core/river_worker.py',
 'braids_core/river_state_snapshot.py','braids_core/river_state_initialization.py',
 'braids_core/sediment/adaptive_river_worker.py','braids_core/sediment/jiang_cao_desktop.py','braids_core/sediment/jiang_cao_adaptive.py','braids_core/sediment/jiang_cao_checkpoint.py',
 'braids_desktop/turbines.py','braids_desktop/forms.py','braids_desktop/results.py','braids_desktop/profile_interpolation.py',
 'braids_desktop/result_annotations.py','braids_desktop/state_initialization.py','braids_desktop/execution.py','braids_desktop/shell.py']
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,d):Path(p).write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stage(refresh=False):
    REPORT.mkdir(parents=True,exist_ok=True);old=read(BASE/'RUNTIME.json')
    if not refresh:
        TARGET.mkdir(parents=True,exist_ok=False)
        for n,h in old['files'].items():
            assert sha(BASE/n)==h,n
            dest=TARGET/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(BASE/n,dest)
    for n in FILES:
        (TARGET/n).parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,TARGET/n)
    version=TARGET/'braids_desktop/__init__.py';version.write_text('"""BRAIDS standalone desktop application."""\n__version__ = "0.1.0-preview60"\n',encoding='utf-8')
    # The isolated base was verified when copied; refresh only declared edits.
    # A complete manifest verification is performed again before installation.
    files=dict(old['files']);files.update({n:sha(TARGET/n) for n in [*FILES,'braids_desktop/__init__.py']});save(TARGET/'RUNTIME.json',dict(old,files=files))
    changed=[n for n,h in files.items() if h!=old['files'].get(n)]
    assert set(changed)<=set(FILES)|{'braids_desktop/__init__.py'}
    save(REPORT/'STAGE.json',dict(parent_runtime=sha(BASE/'RUNTIME.json'),runtime_sha256=sha(TARGET/'RUNTIME.json'),changed=sorted(changed),native_kernels_unchanged=True))
    print(TARGET,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--refresh',action='store_true');stage(p.parse_args().refresh)
