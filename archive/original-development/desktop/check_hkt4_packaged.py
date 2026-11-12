"""Cold packaged CPU/GPU route, no runtime compiler, restart and cancellation."""
import argparse,hashlib,json,shutil,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];TARGET=ROOT/'desktop/qualification/hkt4/candidate'
sys.path.insert(0,str(TARGET))
from braids_core.river_launcher import verify
from braids_core.sediment.portable_river_runtime import install
from braids_core import river_worker
from braids_core.sediment import adaptive_river_worker as core
from braids_core.sediment.river_field_policy import AutomaticRiver
from braids_desktop import preparation
from braids_desktop.results import ResultsStore
from validation.hydrokinetic_turbines.desktop_hkt4_case import create
import numpy as np


def fields(folder):
    store=ResultsStore(folder);return store.read(-1).arrays


def main(backend):
    manifest=verify();install(json.loads((TARGET/'PRECOMPILED.json').read_text()),TARGET)
    # A fresh process must load every CPU binding without compiler discovery.
    shutil.which=lambda name:None
    root=ROOT/'desktop/qualification/hkt4'/('public_release_'+backend);source=create(root,backend,end=3.)
    prepared=root/'prepared';preparation.prepare(source,prepared,asset_cache=root/'assets')
    p,_=river_worker.load_prepared(prepared);assert p['turbines']
    full=river_worker.run(prepared,root/'full',checkpoint_dir=root/'checkpoints');assert full['status']=='complete'
    one=next((root/'checkpoints').glob('t_1_step_*.npz'))
    resumed=river_worker.run(prepared,root/'restart',restart=one);assert resumed['status']=='complete'
    for name in ['q','z','water']:np.testing.assert_array_equal(fields(root/'full')[name],fields(root/'restart')[name])
    stop=root/'stop.request'
    def cancel(folder,row):
        if row['time_s']>=.5:stop.write_text('stop after accepted half-second')
    cancelled=river_worker.run(prepared,root/'cancel',stop_file=stop,on_frame=cancel);assert cancelled['status']=='cancelled' and cancelled['time_s']==.5
    resumed_cancel=river_worker.run(prepared,root/'resume_cancel',restart=root/'cancel/checkpoint.npz');assert resumed_cancel['status']=='complete'
    np.testing.assert_array_equal(fields(root/'full')['q'],fields(root/'resume_cancel')['q'])
    # Qualification-only starting partition forces a genuine accepted-time
    # terrain/dry-bank regrid. This does not change public AMR policy.
    from unittest.mock import patch
    original=AutomaticRiver.assess
    def assess(self,factory,**kw):
        if self.controller.owner.steps==0:self.next_time=self.settings.interval;return
        return original(self,factory,**kw)
    def finest(profile,initial,tree,required=()):return tuple(range(tree.offsets[-2],tree.count))
    with patch.object(core,'initial_leaves',finest),patch.object(AutomaticRiver,'assess',assess):
        adaptive=river_worker.run(prepared,root/'remesh');assert adaptive['status']=='complete'
    events=json.loads((root/'remesh/ADAPTATION.json').read_text());changed=[e for e in events if e['changed']]
    assert changed and changed[0]['time_s']>0
    receipt=dict(backend=backend,runtime=hashlib.sha256((TARGET/'RUNTIME.json').read_bytes()).hexdigest(),compiler_discovery_disabled=True,
        complete=full['status'],restart_exact=True,cancelled_time_s=cancelled['time_s'],resume_cancel_exact=True,
        actual_remesh=[dict(time_s=e['time_s'],from_cells=e['from_cells'],to_cells=e['to_cells'],transfer=e['transfer']) for e in changed])
    (root/'CHECK.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--backend',choices=['cpu','gpu'],required=True);main(a.parse_args().backend)
