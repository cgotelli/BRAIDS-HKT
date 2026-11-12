"""Capture first invalid transport inputs in the automatic-controller regression."""
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from braids_core.sediment.river_suspension import SplitCapacity
from braids_core.sediment.bed_columns_v101 import host
original=SplitCapacity.evaluate
def evaluate(self,q,hi,**kw):
    try:return original(self,q,hi,**kw)
    except ValueError:
        arrays=dict(q=host(q),hi=host(hi),capacity_q=host(kw.get('capacity_q',q)),chezy=host(kw['chezy']))
        folder=ROOT/'validation/_accuracy122/diagnostic';folder.mkdir(exist_ok=True)
        np.savez(folder/'INVALID.npz',**arrays)
        for k,a in arrays.items():print(k,'nonfinite',np.count_nonzero(~np.isfinite(a)),'minmax',np.nanmin(a),np.nanmax(a),flush=True)
        raise
SplitCapacity.evaluate=evaluate
from tools.accuracy122 import main
main()
