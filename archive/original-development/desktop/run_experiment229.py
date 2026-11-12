"""Run the configured preview50 MF1 experiment to the first morphology gate."""
from __future__ import annotations

import hashlib,json,os
from pathlib import Path
import sys,time


ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT.parent/'BRAIDS_JiangCao_Experiment228'
RUNTIME=PROJECT/'runtime';PREPARED=PROJECT/'prepared/experiment229';OUTPUT=PROJECT/'runs/experiment229_600s'
CHECKPOINTS=PROJECT/'checkpoints/experiment229_600s';OUT=ROOT/'validation/_experiment229'
MANIFEST='63b330057b70c20f8066ebeded4b113495fec82e97c734b5bf97a9a9973572f9'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf8')


def main():
    if any(p.exists() for p in (PREPARED,OUTPUT,CHECKPOINTS)):raise FileExistsError('Preserve existing experiment229 artifacts')
    sys.path[:0]=[str(RUNTIME),str(ROOT)]
    os.environ.update(OMP_NUM_THREADS='8',OMP_DYNAMIC='FALSE',OPENBLAS_NUM_THREADS='1',
        MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',QT_QPA_PLATFORM='offscreen')
    from braids_core.river_launcher import verify
    from braids_core.sediment.portable_river_runtime import install
    verify();install(json.loads((RUNTIME/'PRECOMPILED.json').read_text()),RUNTIME)
    if sha(RUNTIME/'RUNTIME.json')!=MANIFEST:raise ValueError('Project runtime differs')
    from braids_desktop import services
    from braids_desktop.preparation import prepare
    from braids_core import river_worker
    document=services.load_project_folder(PROJECT)
    controls=document['settings']['controls'];settings=document['settings']
    if (controls['cfl'],settings['dt_max_s'],controls['threads'])!=(.5,.03,8):raise ValueError('Selected controls differ')
    began=time.perf_counter();prepared=prepare(PROJECT/'braids_project.json',PREPARED,asset_cache=PROJECT/'cache')
    preparation_s=time.perf_counter()-began;began=time.perf_counter()
    result=river_worker.run(prepared,OUTPUT,until=600.,checkpoint_dir=CHECKPOINTS)
    execution_s=time.perf_counter()-began
    services.update_project_session(PROJECT,prepared=str(PREPARED.relative_to(PROJECT)),
        output=str(OUTPUT.relative_to(PROJECT)),log='runs/experiment229_600s/STATUS.json')
    record=dict(runtime=MANIFEST,project=str(PROJECT),prepared=str(PREPARED),output=str(OUTPUT),
        checkpoints=str(CHECKPOINTS),preparation_s=preparation_s,execution_s=execution_s,
        controls=dict(cfl=.5,dt_max_s=.03,threads=8),result=result)
    save(OUT/'RUN_600S.json',record);save(PROJECT/'RUN_600S.json',record)
    print(json.dumps(record,indent=2),flush=True)


if __name__=='__main__':main()
