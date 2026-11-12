"""Record turbine CPU bindings using a fresh cold coupled preparation/run."""
import json
from pathlib import Path
from braids_core.sediment import portable_river_runtime as portable
from validation.hydrokinetic_turbines.desktop_hkt4_case import run

ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    portable.install()
    run(ROOT/'desktop/qualification/hkt4/record_cpu','cpu',.5)
    p=ROOT/'desktop/qualification/hkt4/BINDINGS.json';p.write_text(json.dumps(portable.RECORDS,indent=2)+'\n');print(p)
