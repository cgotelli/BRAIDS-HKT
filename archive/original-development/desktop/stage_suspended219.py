"""Stage S6 over the immutable preview42 delivery."""
from stage_grains212 import ROOT,sha,save,verify
import shutil
BASE=ROOT/'desktop/runtime'
BASE_HASH='79efb9da5ae8801ab7b4d8f38886b87464a7025e955ddcde1c46ec9daca3437c'
OUT=ROOT/'validation/_suspended219';TARGET=OUT/'candidate'
REPORT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_suspension219'
FILES=('braids_core/river_project.py','braids_core/river_model.py','braids_core/river_worker.py',
    'braids_core/sediment/transport_setup.py','braids_core/sediment/river_suspension.py',
    'braids_core/sediment/river_classic_suspension.py','braids_core/sediment/river_classic_suspension.h',
    'braids_desktop/__init__.py','braids_desktop/river_physics.py','braids_desktop/forms.py',
    'braids_desktop/sediment_workflow.py')

def main():
    assert sha(BASE/'RUNTIME.json')==BASE_HASH
    original=verify(BASE)
    if TARGET.exists():m=verify(TARGET)
    else:
        m=original
        for n in m['files']:
            dst=TARGET/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(BASE/n,dst)
    for n in FILES:
        dst=TARGET/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,dst);m['files'][n]=sha(dst)
    save(TARGET/'RUNTIME.json',m);verify(TARGET)
    save(OUT/'STAGE.json',dict(runtime=sha(TARGET/'RUNTIME.json'),base=BASE_HASH,changed=FILES))
    print('Staged S6',sha(TARGET/'RUNTIME.json'),flush=True)

if __name__=='__main__':main()

