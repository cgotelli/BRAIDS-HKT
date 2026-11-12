"""Run the isolated Jiang--Cao odd/even candidate on the paper flume."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
CANDIDATE=ROOT/'validation/_checkerboard227/candidate_v3'
CASE=ROOT.parent/'BRAIDS_JiangCao_Experiment225'
OUT=ROOT/'validation/_checkerboard227/runs'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('label');parser.add_argument('--backend',choices=('cpu','gpu'),required=True)
    parser.add_argument('--end',type=float,required=True);parser.add_argument('--threads',type=int,default=8)
    parser.add_argument('--restart',type=Path);parser.add_argument('--prepared',type=Path)
    parser.add_argument('--diagnose-capacity',action='store_true')
    args=parser.parse_args()
    folder=OUT/args.label
    if folder.exists():raise FileExistsError(f'Preserve existing qualification: {folder}')
    folder.mkdir(parents=True)
    sys.path[:0]=[str(CANDIDATE),str(ROOT)]
    os.environ.update(OMP_NUM_THREADS=str(args.threads),OMP_DYNAMIC='FALSE',OPENBLAS_NUM_THREADS='1',
        MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',QT_QPA_PLATFORM='offscreen')
    from braids_core.river_launcher import verify
    from braids_core.sediment.portable_river_runtime import install
    verify();install(json.loads((CANDIDATE/'PRECOMPILED.json').read_text()),CANDIDATE)
    from braids_desktop import services
    from braids_desktop.preparation import prepare
    from braids_core.river_project import validate_project
    from braids_core.river_model import resolve_model
    from braids_core import river_worker
    if args.diagnose_capacity:
        from braids_core.sediment.jiang_cao_capacity import JiangCaoCapacity, STATUSES
        original_evaluate=JiangCaoCapacity.evaluate
        def evaluate_with_evidence(self,q,grad_z,manning,*,concentration=0.,strict=True):
            result=original_evaluate(self,q,grad_z,manning,concentration=concentration,strict=False)
            bad=self.xp.flatnonzero(result.status>=3)
            if strict and int(bad.size):
                index=int(bad[0]);host=(self.xp.asnumpy if self.backend=='gpu' else lambda a: np.asarray(a))
                state=host(q)[:,index];gradient=host(grad_z)[:,index]
                rough=float(host(self.xp.broadcast_to(self.xp.asarray(manning),(q.shape[1],)))[index])
                conc=float(host(self.xp.broadcast_to(self.xp.asarray(concentration),(q.shape[1],)))[index])
                row=dict(index=index,status=int(host(result.status)[index]),message=STATUSES[int(host(result.status)[index])],
                    q=state.tolist(),gradient=gradient.tolist(),gradient_magnitude=float(np.hypot(*gradient)),
                    manning=rough,concentration=conc)
                save(folder/'CAPACITY_FAILURE.json',row)
                raise ValueError(f"Jiang-Cao capacity cell {index}: {row['message']}")
            return result
        JiangCaoCapacity.evaluate=evaluate_with_evidence
    from copy import deepcopy
    document=services.load_project_folder(CASE)
    request=deepcopy(document['model']['request']);request['execution']['backend']=args.backend
    document['model']=resolve_model(request).document()
    document['settings']['controls'].update(threads=args.threads,cfl=.8)
    document['settings'].update(dt_max_s=.02,end_time_s=max(60.,args.end),output_interval_s=5.,checkpoint_interval_s=5.)
    document['editor_state']['checkpoint_clock']=dict(mode='custom',custom_interval_s=5.)
    project=folder/'project.json';save(project,validate_project(document))
    started=time.perf_counter()
    prepared=args.prepared.resolve() if args.prepared else prepare(project,folder/'prepared',asset_cache=CASE/'cache')
    preparation_s=time.perf_counter()-started
    started=time.perf_counter()
    result=river_worker.run(prepared,folder/'run',restart=args.restart,until=args.end,
        checkpoint_dir=folder/'checkpoints')
    execution_s=time.perf_counter()-started
    row=dict(label=args.label,backend=args.backend,threads=args.threads,end_s=args.end,
        preparation_s=preparation_s,execution_s=execution_s,result=result,
        runtime_sha256=sha(CANDIDATE/'RUNTIME.json'),prepared=str(prepared),restart=str(args.restart) if args.restart else None)
    save(folder/'RESULT.json',row);print(json.dumps(row,indent=2),flush=True)


if __name__=='__main__':main()
