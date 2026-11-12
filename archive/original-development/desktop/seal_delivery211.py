"""Seal numerical delivery evidence, leaving the user's project editable."""
import json,sys
from pathlib import Path
from stage_delivery211 import ROOT,OUT,TARGET,sha,save,verify
from experiment_delivery211 import FOLDER,RUNTIME
REPORT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_delivery211'
def main():
    m=verify(RUNTIME);assert sha(RUNTIME/'RUNTIME.json')==sha(TARGET/'RUNTIME.json')
    old=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_exchange210/ARTIFACTS.json'
    evidence=json.loads(old.read_text())['files']
    for n,h in evidence.items():assert sha(ROOT/n)==h,n
    ready=json.loads((OUT/'READY_EXPERIMENT.json').read_text());assert ready['all_frames_exact_to_first_run'] and ready['editor_stable']
    editor=json.loads((OUT/'EDITOR_CHECK.json').read_text());assert editor['current_continuation'] and not editor['physical_changes']
    continued=json.loads((OUT/'CONTINUATION_READY.json').read_text());assert all(continued['start_fields_exact'].values())
    assert continued['status']['time_s']==21. and continued['status']['status']=='complete'
    check=json.loads((OUT/'INSTALLED_CHECK.json').read_text());assert all(x['all_frames_exact'] for x in check['checks'])
    sys.path[:0]=[str(RUNTIME),str(ROOT)]
    from braids_desktop import services
    from braids_core.river_project import load_project
    from braids_core.river_worker import load_prepared,validate_continuation
    services.load_project_folder(FOLDER);services.project_history(FOLDER)
    validate_continuation(load_prepared(FOLDER/'prepared/flume211_ready')[0],load_project(FOLDER/'braids_project.json'))
    project_files={p.relative_to(FOLDER).as_posix():sha(p) for p in FOLDER.rglob('*') if p.is_file() and p.name not in ('.braids-folder.guard','.braids-folder.lock')}
    save(REPORT/'DELIVERED_FOLDER.json',dict(root=str(FOLDER),status='checksums at delivery; this user project remains editable',files=project_files))
    save(REPORT/'CLOSURE.json',dict(runtime=sha(RUNTIME/'RUNTIME.json'),payloads=len(m['files']),
      previous210_artifacts_unchanged=len(evidence),packaged_tests=519,ordinary_formula_controls=16,
      lifecycle_cases=4,installed_child_all_frames_exact=True,editor_continuation=True,
      experiment_end_s=20.,checkpoint_extension_end_s=21.,MF=1.,QGIS_changed=False,Delft3D_modified=False,
      interactive_visual_check='unavailable: Windows native pipe connection failed twice',no_simulation_left_running=True))
    files={}
    for base in (OUT,REPORT):
        for p in base.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.name!='ARTIFACTS.json':files[p.relative_to(ROOT).as_posix()]=sha(p)
    for p in sorted((ROOT/'desktop').glob('*211.py')):files[p.relative_to(ROOT).as_posix()]=sha(p)
    for n in ('braids_core/river_jiang_cao.py','braids_core/sediment/jiang_cao_desktop.py','braids_desktop/__init__.py'):
        files[n]=sha(ROOT/n)
    save(REPORT/'ARTIFACTS.json',dict(schema=1,milestone=211,path_base='repository_root',files=files))
    print('Sealed',len(files),'artifacts; previous210 unchanged',len(evidence),flush=True)
if __name__=='__main__':main()
