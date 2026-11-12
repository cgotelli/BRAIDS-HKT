"""Stage measured bulk diameters over immutable preview37, never edit its inputs."""
from stage_grains212 import ROOT,sha,save,verify
import shutil,json
BASE=ROOT/'desktop/runtime'
BASE_HASH='8bc9e502ee15600e197161aa29b62a2e3b58b900416909fc4c5f28719eaf119c'
OUT=ROOT/'validation/_bulk213';TARGET=OUT/'candidate'
FILES=('braids_core/river_model.py','braids_core/river_project.py','braids_core/river_jiang_cao.py',
    'braids_core/sediment/grain_diameter_contract.py','braids_core/sediment/river_capacity.py',
    'braids_core/sediment/river_scalar_law.h','braids_core/sediment/river_corrections.py',
    'braids_core/sediment/river_exterior_sediment.py','braids_core/sediment/river_smart_jaggi.py',
    'braids_core/sediment/jiang_cao_roughness.py','braids_core/sediment/jiang_cao_wetdry.py',
    'braids_core/sediment/jiang_cao_desktop.py','braids_desktop/__init__.py',
    'braids_desktop/bedload_catalog.py','braids_desktop/river_physics.py',
    'braids_desktop/sediment_workflow.py','braids_desktop/forms.py')

def main():
    assert sha(BASE/'RUNTIME.json')==BASE_HASH
    original=verify(BASE)
    if not TARGET.exists():
        for n in original['files']:
            dest=TARGET/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(BASE/n,dest)
        save(TARGET/'RUNTIME.json',original)
    manifest=json.loads((TARGET/'RUNTIME.json').read_text())
    for n,h in manifest['files'].items():
        if n not in FILES:assert sha(TARGET/n)==h,n
    for n in FILES:
        shutil.copy2(ROOT/n,TARGET/n);manifest['files'][n]=sha(TARGET/n)
    save(TARGET/'RUNTIME.json',manifest);verify(TARGET)
    save(OUT/'STAGE.json',dict(runtime=sha(TARGET/'RUNTIME.json'),base=BASE_HASH,changed=FILES))
    print('Staged preview38',sha(TARGET/'RUNTIME.json'),flush=True)

if __name__=='__main__':main()
