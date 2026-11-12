"""Exact paired restart plus the unchanged user case through sediment activation."""
import argparse,json,sys,time,os,io
from stage_secondary185 import ROOT,TARGET,OUT,save,sha
os.environ['QT_QPA_PLATFORM']='offscreen'
sys.path[:0]=[str(TARGET),str(ROOT)]
import numpy as np
from braids_core.river_launcher import verify
from braids_core.sediment.portable_river_runtime import install
from braids_core import river_worker
from braids_desktop.results import ResultsStore


def physics_checkpoint(path):
    data=path.read_bytes()
    while True:
        with np.load(io.BytesIO(data),allow_pickle=False) as archive:
            key=next((k for k in ('payload','owner','river') if k in archive),None)
            if key is None:return {k:archive[k].copy() for k in archive if k!='metadata'}
            data=archive[key].tobytes()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--backend',required=True,choices=['cpu','gpu']);args=ap.parse_args()
    verify();install(json.loads((TARGET/'PRECOMPILED.json').read_text()),TARGET)
    b=args.backend;base=OUT/f'signed_{b}_v6';prep=base/'prepared'
    folder=OUT/f'qualified_{b}_v6';folder.mkdir(exist_ok=False)
    owners=[];records=[];factory=river_worker.core.factory
    def instrument(*a,**kw):
        build=factory(*a,**kw)
        def traced(*aa,**kk):
            owner=build(*aa,**kk);owners.append(owner)
            # Detached snapshot and commit=False must restore all arrays,
            # including secondary state and accumulated face/source ledgers.
            was_morphology=owner.morphology;owner.morphology=False
            original=owner.checkpoint_bytes()
            owner.step(.01,commit=False)
            assert owner.checkpoint_bytes()==original,'Trial rollback changed checkpoint'
            owner.morphology=was_morphology
            step=owner.step
            def recorded(*sa,**sk):
                result=step(*sa,**sk)
                if sk.get('commit',True):
                    records.append(dict(time_s=owner.time,**{k:owner.physics.last[k] for k in
                        ('secondary_residual','secondary_cfl_max','secondary_substeps','solid_volume_residual_m3')}))
                return result
            owner.step=recorded
            return owner
        traced.__dict__.update(build.__dict__);return traced
    river_worker.core.factory=instrument
    stop=folder/'STOP'
    def cancel(output,row):
        if row['time_s']>=10.:stop.touch()
    prefix=river_worker.run(prep,folder/'cancelled',until=20.,stop_file=stop,on_frame=cancel)
    assert prefix['status']=='cancelled' and prefix['time_s']==10.,prefix
    resumed=river_worker.run(prep,folder/'resumed',until=20.,restart=folder/'cancelled/checkpoint.npz')
    expected=ResultsStore(base/'run').read(-1);actual=ResultsStore(folder/'resumed').read(-1)
    exact={k:bool(np.array_equal(expected.arrays[k],actual.arrays[k])) for k in
           ('q','z','W','mass','water','bed_boundary','secondary_boundary','secondary_source') if k in actual.arrays}
    assert all(exact.values()),exact
    expected_checkpoint=physics_checkpoint(base/'run/checkpoint.npz')
    actual_checkpoint=physics_checkpoint(folder/'resumed/checkpoint.npz')
    ledger_keys=('secondary_boundary_ledger','secondary_source_ledger','secondary_projection_ledger')
    exact.update({k:bool(np.array_equal(expected_checkpoint[k],actual_checkpoint[k])) for k in ledger_keys})
    assert all(exact.values()),exact
    save(folder/'RESTART.json',dict(runtime=sha(TARGET/'RUNTIME.json'),exact=exact,
        cancellation=prefix['status'],rollback_byte_exact=True))
    print(b,'RESTART PASS',exact,flush=True)
    # Preserve the input's 200-second warm-up and original 0.2 s outer step.
    def progress(output,row):print(b,'physical time',row['time_s'],flush=True)
    start=time.perf_counter()
    status=river_worker.run(prep,folder/'to_220',until=220.,restart=base/'run/checkpoint.npz',on_frame=progress)
    last=ResultsStore(folder/'to_220').read(-1)
    assert status['status']=='segment_complete' and status['time_s']==220.,status
    assert np.isfinite(last.arrays['q']).all() and last.arrays['q'][0].min()>=0
    assert np.isfinite(last.arrays['W']).all() and last.arrays['mass'].min()>=0
    initial=ResultsStore(base/'run').read(0);dz=last.arrays['z']-initial.arrays['z']
    assert dz.min()<0. and dz.max()>0.,(dz.min(),dz.max())
    diagnostics=json.loads((folder/'to_220/DIAGNOSTICS.json').read_text())
    save(folder/'STEPS.json',records)
    values={k:max(abs(float(row[k])) for row in records) for k in
            ('secondary_residual','solid_volume_residual_m3','secondary_cfl_max')}
    values['water_residual_m3']=max(abs(row['water_residual_m3']) for row in diagnostics)
    values['class_residual_kg']=max(max(abs(v) for v in row['class_residual_kg']) for row in diagnostics)
    for row in diagnostics:
        assert abs(row['water_residual_m3'])<=row['water_roundoff_limit_m3']
        assert max(abs(v) for v in row['class_residual_kg'])<=row['class_roundoff_limit_kg']
    assert values['secondary_residual']<1e-10
    assert values['secondary_cfl_max']<=.8*(1+1e-12)
    save(folder/'CHECK.json',dict(status=status,runtime=sha(TARGET/'RUNTIME.json'),
        continuation_wall_s=time.perf_counter()-start,restart_exact=exact,
        extrema=dict(bed_min_m=float(dz.min()),bed_max_m=float(dz.max()),W_abs_max=float(abs(last.arrays['W']).max())),
        diagnostics=values,initial_state='user saved dry start',sediment_start_time_s=200.,outer_dt_max_s=.2))
    print(b,'220 s PASS',flush=True)
if __name__=='__main__':main()
