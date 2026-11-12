"""Reject old runtime and corrupted current checkpoints before output creation."""
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'validation/_wall188';TARGET=OUT/'desktop_candidate'
sys.path[:0]=[str(TARGET),str(ROOT)]
from braids_core import river_worker
from braids_core.river_launcher import verify
from stage_secondary185 import save,sha


def main():
    verify();rows={};folder=OUT/'identity';folder.mkdir(exist_ok=False)
    for backend in ('cpu','gpu'):
        prep=OUT/f'release/wet_prescribed_{backend}/prepared'
        old=ROOT/f'validation/_inlet187/wet_prescribed_{backend}/run/checkpoint.npz'
        current=OUT/f'release/wet_prescribed_{backend}/run/checkpoint.npz'
        with np.load(current,allow_pickle=False) as f:identity=f['identity'].copy();payload=f['payload'].copy()
        payload[len(payload)//2]^=1
        corrupt=folder/(backend+'_corrupt.npz');np.savez(corrupt,identity=identity,payload=payload)
        for label,path in (('old',old),('corrupt',corrupt)):
            output=folder/(backend+'_'+label+'_must_not_run')
            try:river_worker.run(prep,output,restart=path,until=220.)
            except ValueError as exc:
                assert 'checkpoint' in str(exc).lower() or 'identity' in str(exc).lower(),str(exc)
                assert not output.exists()
                rows[backend+'_'+label]=dict(rejected=True,error=str(exc),output_created=False)
            else:raise AssertionError('Checkpoint unexpectedly accepted')
    save(OUT/'IDENTITY.json',dict(runtime=sha(TARGET/'RUNTIME.json'),checks=rows))
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
