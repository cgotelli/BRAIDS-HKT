"""Keep the distinct hydraulic control and wet/dry/reversal checks bounded."""
from concurrent.futures import ThreadPoolExecutor
import json
from run_wall188 import execute,OUT,save
CASES=('total','critical','normal','stage','dry_critical','dry_normal','dry_stage',
       'out_critical','weir_valid','free_supercritical','backflow','schedule')


def sequence(backend):
    for case in CASES:
        receipt=OUT/'boundaries'/(case+'_'+backend)/'CHECK.json'
        if receipt.exists():
            row=json.loads(receipt.read_text());assert row['status']=='complete'
            continue
        execute('boundary_'+case+'_'+backend,'boundaries_wall188.py',
            ['--backend',backend,'--case',case,'--output',str(OUT/'boundaries'/(case+'_'+backend))])


if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=2) as pool:
        for job in [pool.submit(sequence,b) for b in ('cpu','gpu')]:job.result()
    save(OUT/'BOUNDARY_JOBS.json',dict(cases=CASES,backends=['cpu','gpu'],passed=2*len(CASES)))
