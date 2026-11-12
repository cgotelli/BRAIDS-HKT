"""Stage S2/S3 shared numerical sources and Desktop over immutable preview39."""
from stage_grains212 import ROOT,sha,save,verify
import shutil
BASE=ROOT/'desktop/runtime'
BASE_HASH='0eea57c99b1f6e6818c96de222826d492900356df5141e66709b63f92ad9c114'
OUT=ROOT/'validation/_suspended216';TARGET=OUT/'candidate'
REPORT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_suspension216'
FILES=('braids_core/river_project.py','braids_core/sediment/transport_setup.py',
    'braids_core/sediment/grain_diameter_contract.py','braids_core/sediment/river_vr1993.py',
    'braids_core/sediment/river_suspension.py','braids_core/sediment/river_suspension.h',
    'braids_core/sediment/river_capacity.py','braids_core/sediment/river_amr_transfer.py',
    'braids_desktop/__init__.py','braids_desktop/river_physics.py','braids_desktop/forms.py',
    'braids_desktop/sediment_workflow.py','braids_desktop/river_forcing.py','braids_desktop/bedload_catalog.py')

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
    print('Staged S3',sha(TARGET/'RUNTIME.json'),flush=True)

if __name__=='__main__':main()
