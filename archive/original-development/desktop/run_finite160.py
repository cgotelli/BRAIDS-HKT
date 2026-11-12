"""Short erodible-bed controls for the newly exposed scalar formulas."""
from concurrent.futures import ThreadPoolExecutor
import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'validation/_boundaries160/finite_runs';OUT.mkdir(exist_ok=False)
def campaign(backend):
    rows=[]
    for formula in ('ashida_michiue_generalized','wang_fredsoe'):
        name=formula+'_'+backend
        with (OUT/(name+'.log')).open('w',encoding='utf-8') as log:
            done=subprocess.run([sys.executable,'-B',str(ROOT/'desktop/probe_clock160.py'),'--output',str(OUT/name),
                '--backend',backend,'--formula',formula,'--end','60','--stock','.05'],stdout=log,stderr=subprocess.STDOUT,timeout=600)
        rows.append(dict(case=name,exit_code=done.returncode));print(rows[-1],flush=True)
    (OUT/(backend+'_SUMMARY.json')).write_text(json.dumps(rows,indent=2));return rows
with ThreadPoolExecutor(max_workers=2) as pool:rows=list(pool.map(campaign,('cpu','gpu')))
assert not any(r['exit_code'] for group in rows for r in group)
