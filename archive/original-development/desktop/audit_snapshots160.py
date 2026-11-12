"""Read-only audit of the user's actual groin runs, not just the supplied example."""
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'desktop/runtime'),str(ROOT)]
from braids_desktop.results import ResultsStore
base=Path('C:/Users/STREEM/Desktop/BRAIDS bare-bed groin/CPU/Groin_data/runs')
for run in base.iterdir():
    if not (run/'frames/INDEX.json').exists():continue
    s=ResultsStore(run);print(run.name,s.times[-1],flush=True)
    if s.times[-1]<210:continue
    for t in [195,200,205,210,215]:
        f=s.read(int(np.argmin(abs(s.times-t))));v=np.linalg.norm(f.arrays['bedload_vector_m2_s'],axis=0)
        print(f.time_s,len(v),'bedload',np.quantile(v,[0,.5,.9,1]),'z',np.quantile(f.z-f.arrays['initial_bed_m'],[0,.5,1]),flush=True)
