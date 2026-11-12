"""Twenty-second mobile-bed window on the matched longer hydraulic control."""
import argparse
import json
import shutil
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'desktop'))
import check_bedload161 as old
from check_domain162 import BASE as HYDRO

BASE=ROOT/'validation/_hydraulics162/mobile'


def delft(formula):
    import runtime
    parent=HYDRO/'long/delft';out=BASE/('delft_'+formula)
    out.mkdir(parents=True,exist_ok=False)
    inputs=json.loads((parent/'INPUTS.json').read_text())
    for name,checksum in inputs['inputs'].items():
        assert old.sha(parent/name)==checksum
        shutil.copy2(parent/name,out/name)
    mdf=(out/'sim_P057.mdf').read_text()
    for key,value in dict(Tstop='60',Flmap='0 5 60',Flhis='0 5 60').items():mdf=old.replace(mdf,key,value)
    (out/'sim_P057.mdf').write_text(mdf)
    (out/'mor.mor').write_text(old.replace((out/'mor.mor').read_text(),'MorStt','40'))
    number,parameters=old.SPECS[formula]
    (out/'tra.tra').write_text('[TransportFormulaFileInformation]\nFileVersion=01.00\n[TransportFormula]\nNumber='+str(number)+'\n'+''.join(f'{k}={v:.17g}\n' for k,v in parameters.items()))
    old.save(out/'INPUTS.json',dict(parent=str(parent),parent_manifest_sha256=old.sha(parent/'INPUTS.json'),
        changes=dict(end_s=60,morphology_starts_s=40,formula=formula,parameters=parameters),
        Delft3D_source_modified=False,Delft3D_runtime_modified=False,
        inputs={p.name:old.sha(p) for p in out.iterdir() if p.is_file()}))
    print(runtime.run_case(out,configuration='config_flow2d3d.xml',timeout_s=180),flush=True)


def braids(formula,backend):
    from PySide6.QtWidgets import QApplication
    from braids_desktop import services
    from braids_desktop.forms import SettingsPanel
    from braids_desktop.river_physics import set_choice
    from braids_desktop.preparation import prepare
    from braids_core.river_open_boundary import resolve
    from braids_core.river_boundary import TOTAL
    from braids_core.sediment.portable_river_runtime import install
    from braids_core.river_worker import run
    install(json.loads((old.RUNTIME/'PRECOMPILED.json').read_text()),old.RUNTIME)
    app=QApplication.instance() or QApplication([])
    out=BASE/('braids_'+formula+'_'+backend);out.mkdir(parents=True,exist_ok=False)
    parent=HYDRO/'long/braids_cpu/project'
    doc=services.load_project_folder(parent)
    dem=Path(doc['terrain']['dem_path'])
    if not dem.is_absolute():doc['terrain']['dem_path']=str((parent/dem).resolve())
    f=SettingsPanel();f.load_document(doc)
    set_choice(f.backend,backend)
    f.sediment_enabled.setChecked(True);f.physics.bedload_on.setChecked(True)
    f.physics.grains.item(0,1).setText('1');f.physics.layers.item(0,0).setText('.1')
    set_choice(f.physics.capacity,'capacity.'+formula)
    f.end_time.setValue(60.);f.sediment_start.setValue(40.);f.morfac.setValue(1.)
    set_choice(f.forcing.method,TOTAL);f.forcing.table.item(0,2).setText('.0001')
    doc=f.document();f.close();doc['initial'].update(stage_m=.6,velocity_m_s=[0.,0.])
    doc['boundary_states']['Downstream']['stage_policy']='stage.subcritical_backflow'
    doc.pop('editor_state',None)
    doc['boundary_resolution']=resolve(doc['settings']['hydraulics'],doc['boundary_states'],
        dict(Upstream='inlet',Downstream='outlet'),sediment=doc['model']['physics']['sediment'],ng=1,end_time=60.)
    services.save_project_folder(out/'project',doc)
    prepare(out/'project/braids_project.json',out/'prepared',asset_cache=BASE/'assets')
    status=run(out/'prepared',out/'run',until=60.)
    old.save(out/'STATUS.json',dict(status=status,script_sha256=old.sha(__file__),
        runtime_manifest_sha256=old.sha(old.RUNTIME/'RUNTIME.json')))
    print(status,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('engine',choices=['braids','delft'])
    p.add_argument('formula',choices=old.SPECS);p.add_argument('--backend',choices=['cpu','gpu'],default='cpu')
    a=p.parse_args()
    if a.engine=='delft':delft(a.formula)
    else:braids(a.formula,a.backend)
