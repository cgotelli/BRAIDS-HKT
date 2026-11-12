"""Independent CPU, GPU and locked FLOW validation sequences, not hybrid solving."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'validation/_hydraulics162/mobile'
LAWS=('mpm','ashida_michiue_generalized','wang_fredsoe')


def sequence(engine,backend=None):
    for formula in LAWS:
        name=engine+'_'+formula+('_'+backend if backend else '')
        receipt=BASE/name/('STATUS.json' if engine=='braids' else 'execution.json')
        if receipt.exists():
            data=json.loads(receipt.read_text())
            assert data.get('success',data.get('status',{}).get('status')=='complete')
            continue
        command=[sys.executable,str(ROOT/'desktop/check_mobile162.py'),engine,formula]
        if backend:command+=['--backend',backend]
        log=BASE/'logs'/f'{name}.log';log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('w') as f:
            result=subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=240)
        if result.returncode:raise RuntimeError(str(log))
        print(name+' complete',flush=True)


if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=3) as pool:
        tasks=[pool.submit(sequence,'braids',b) for b in ('cpu','gpu')]
        tasks.append(pool.submit(sequence,'delft'))
        for task in tasks:task.result()
