"""Two rotors through a real accepted-time dry-bank regrid and restart."""
import argparse,importlib.util,json,shutil,sys
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'desktop/qualification/turbine_array55';TARGET=OUT/'candidate';sys.path.insert(0,str(TARGET))
from braids_core.river_launcher import verify
from braids_core.sediment.portable_river_runtime import install
from braids_core import river_worker
from braids_core.river_project import validate_project
from braids_core.sediment import adaptive_river_worker as core
from braids_core.sediment.river_field_policy import AutomaticRiver
from braids_desktop import preparation
spec=importlib.util.spec_from_file_location('_hkt4_fixture',ROOT/'validation/hydrokinetic_turbines/desktop_hkt4_case.py')
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture);create=fixture.create


def last(folder):
    row=json.loads((folder/'frames/INDEX.json').read_text())['frames'][-1]
    with np.load(folder/'frames'/row['file']) as a:return {k:a[k].copy() for k in a.files}


def run(backend):
    verify();install(json.loads((TARGET/'PRECOMPILED.json').read_text()),TARGET);shutil.which=lambda _:None
    root=OUT/('remesh_'+backend);source=create(root,backend,end=1.)
    doc=json.loads(source.read_text());doc['settings']['checkpoint_interval_s']=.5
    second=deepcopy(doc['turbines']['turbines'][0]);second.update(id='Second rotor',center_xy=[5.03,2.]);doc['turbines']['turbines'].append(second)
    doc=validate_project(doc);source.write_text(json.dumps(doc,indent=2)+'\n')
    prepared=preparation.prepare(source,root/'prepared',asset_cache=OUT/'remesh_assets')
    original=AutomaticRiver.assess
    def assess(self,factory,**kw):
        if self.controller.owner.steps==0:self.next_time=self.settings.interval;return
        return original(self,factory,**kw)
    def finest(profile,initial,tree,required=()):return tuple(range(tree.offsets[-2],tree.count))
    with patch.object(core,'initial_leaves',finest),patch.object(AutomaticRiver,'assess',assess):
        result=river_worker.run(prepared,root/'run',checkpoint_dir=root/'checkpoints');assert result['status']=='complete'
        events=json.loads((root/'run/ADAPTATION.json').read_text());changed=[e for e in events if e['changed']]
        assert changed and changed[0]['time_s']>0
        checkpoint=next((root/'checkpoints').glob('t_0.5_step_*.npz'))
        resumed=river_worker.run(prepared,root/'resume',restart=checkpoint);assert resumed['status']=='complete'
    for field in ['q','z','water']:np.testing.assert_array_equal(last(root/'run')[field],last(root/'resume')[field])
    rows=json.loads((root/'run/TURBINE_PERFORMANCE.json').read_text());assert {r['turbine_id'] for r in rows}=={'Turbine 1','Second rotor'}
    diagnostics=json.loads((root/'run/DIAGNOSTICS.json').read_text());residual=max(abs(r['water_residual_m3']) for r in diagnostics);assert residual<1e-8
    receipt=dict(backend=backend,end_s=1.,restart_after_remesh_exact=True,forced_initial_finest_for_qualification=True,
        changed=[{k:e[k] for k in ['time_s','from_cells','to_cells','transfer']} for e in changed],maximum_water_residual_m3=residual,rotor_ids=sorted({r['turbine_id'] for r in rows}))
    (root/'CHECK.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--backend',choices=['cpu','gpu'],required=True);run(p.parse_args().backend)
