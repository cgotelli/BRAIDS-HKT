"""Qualify the geometric repose criterion with ordinary Exner inventory."""
from stage_grains212 import ROOT,sha,save,verify
import shutil
BASE=ROOT/'desktop/runtime'
BASE_HASH='d9e01d1ff72d2910fae9d24b97082dc71f4947fa3893dfbd02decda572d43694'
OUT=ROOT/'desktop/qualification/repose223';TARGET=OUT/'candidate_final'
REPORT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_repose223'
FILES=('braids_core/river_project.py','braids_core/river_model.py',
    'braids_core/sediment/collapse_setup.py','braids_core/sediment/adaptive_river_worker.py',
    'braids_core/sediment/river_repose.py','braids_core/sediment/river_polygon_banks.py',
    'braids_core/sediment/jiang_cao_collapse.py','braids_desktop/__init__.py',
    'braids_desktop/forms.py','braids_desktop/river_physics.py','braids_desktop/services.py',
    'braids_desktop/sediment_workflow.py')
def main():
    assert sha(BASE/'RUNTIME.json')==BASE_HASH
    m=verify(BASE)
    if TARGET.exists():verify(TARGET)
    else:
        for n in m['files']:
            dst=TARGET/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(BASE/n,dst)
    for n in FILES:
        dst=TARGET/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,dst);m['files'][n]=sha(dst)
    save(TARGET/'RUNTIME.json',m);verify(TARGET)
    assert sha(TARGET/'PRECOMPILED.json')==sha(BASE/'PRECOMPILED.json')
    save(OUT/'STAGE.json',dict(runtime=sha(TARGET/'RUNTIME.json'),base=BASE_HASH,changed=FILES,native_bindings_unchanged=True))
    print('Staged preview47',sha(TARGET/'RUNTIME.json'),flush=True)
if __name__=='__main__':main()
