"""Run CPU and GPU groin campaigns; timings are not clean speed benchmarks."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'desktop'))
from check_boundaries159 import CASES

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cases',nargs='+',default=list(CASES));args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    def campaign(backend):
        summary=[]
        for case in args.cases:
            name=case+'_'+backend;began=time.perf_counter()
            with (args.output/(name+'.log')).open('w',encoding='utf-8') as log:
                try:
                    result=subprocess.run([sys.executable,'-B',str(ROOT/'desktop/check_boundaries159.py'),
                        '--case',case,'--backend',backend,'--output',str(args.output/name)],
                        stdout=log,stderr=subprocess.STDOUT,timeout=300)
                    code=result.returncode
                except subprocess.TimeoutExpired:code=-100
            row=dict(case=case,backend=backend,exit_code=code,wall_s=time.perf_counter()-began)
            summary.append(row);print(json.dumps(row),flush=True)
            (args.output/(backend+'_SUMMARY.json')).write_text(json.dumps(summary,indent=2))
        return summary
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(campaign,('cpu','gpu')))
    if any(r['exit_code'] for group in results for r in group):raise SystemExit(1)
if __name__=='__main__':main()
