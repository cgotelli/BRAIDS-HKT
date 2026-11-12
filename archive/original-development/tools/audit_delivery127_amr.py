"""Read-only execution instrumentation of the large-coordinate AMR budget."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from braids_core import river_worker
from braids_core.sediment.bed_columns_v101 import host
from braids_core.sediment.adaptive_river_worker import AutomaticRiver
import numpy as np


def main():
    source=ROOT/'validation/_delivery127/release_record/record_041/prepared'
    output=ROOT/'validation/_delivery127/amr_budget_audit';records=[]
    original=AutomaticRiver.assess
    def totals(driver):
        a=driver.controller.owner
        return dict(time=a.time,cells=a.physics.n,water=float(np.sum(host(a.q[0])*host(a.columns.area))),
            ledger=float(np.sum(host(a.water_ledger))),area=float(np.sum(host(a.columns.area))))
    def assess(self,*args,**kw):
        before=totals(self);r=original(self,*args,**kw);after=totals(self)
        if before['cells']!=after['cells'] or before['water']!=after['water']:
            records.append(dict(before=before,after=after,water_change=after['water']-before['water']))
        return r
    AutomaticRiver.assess=assess
    try:river_worker.run(source,output)
    finally:
        (output/'AMR_AUDIT.json').write_text(json.dumps(records,indent=2));print(json.dumps(records,indent=2))


if __name__=='__main__':main()
