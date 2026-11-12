"""Short natural-river AMR regression for unchanged open-boundary histories."""
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from braids_core import river_worker
from braids_core.river_model import resolve_model
from braids_core.sediment.adaptive_river_worker import read_live_frame
from braids_gpu.atomic_output import write_json_atomic


def main():
    out=ROOT/'validation/_delivery127/amr_history_fixed';out.mkdir(parents=True,exist_ok=False);runs={}
    source=ROOT/'validation/_delivery127/release_record/record_041/prepared/PROJECT.json'
    for backend in ('cpu','gpu'):
        doc=json.loads(source.read_text());doc['model']['request']['execution']['backend']=backend
        doc['model']=resolve_model(doc['model']['request']).document();path=out/(backend+'.json')
        write_json_atomic(path,doc);prep=river_worker.prepare(path,out/(backend+'_prepared'))
        run=out/(backend+'_run');status=river_worker.run(prep,run)
        assert status['status']=='complete'
        row,mesh,fields=read_live_frame(run);diag=json.loads((run/'DIAGNOSTICS.json').read_text())
        assert max(abs(r['water_residual_m3']) for r in diag)<1e-10
        assert all(not any(r['wall_integrals']) for r in diag)
        np.testing.assert_allclose(fields['q'][0]@fields['area'],88.,rtol=0,atol=1e-10)
        runs[backend]=dict(status=status,maximum_water_residual_m3=max(abs(r['water_residual_m3']) for r in diag),
            cells=len(mesh['leaves']),water_m3=float(fields['q'][0]@fields['area']))
        if backend=='cpu':reference=fields
        else:
            for k in ('q','z','mass'):np.testing.assert_allclose(fields[k],reference[k],rtol=0,atol=3e-9)
        write_json_atomic(out/'CHECKS.json',runs);print(backend,runs[backend],flush=True)


if __name__=='__main__':main()
