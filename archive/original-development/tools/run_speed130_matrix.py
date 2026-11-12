"""Bounded sequential numerical screens, not concurrent speed benchmarks."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--group',choices=['wet_cfl','clock'],required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    if a.group=='wet_cfl':
        source=ROOT/'validation/_inlet129/natural2m_final/2m_critical_gpu/PROJECT.json'
        specs=[('cap050',.5,1e-6,'baseline',.5),('cap075',.75,1e-6,'baseline',.5),
               ('cap100',1.,1e-6,'baseline',.5),('cap150',1.5,1e-6,'baseline',.5),
               ('dry010mm',1.,1e-4,'baseline',.5),('dry100mm',1.,1e-3,'baseline',.5)]
        backend='gpu';until=100.
    else:
        source=ROOT/'validation/_inlet129/natural5m_final/5m_critical_cpu/PROJECT.json'
        specs=[('carry',.5,1e-6,'carry',.5),('near_limit',.5,1e-6,'near_limit',.5),
               ('combined',.5,1e-6,'combined',.5),('outer1',.5,1e-6,'combined',1.),
               ('cap100_outer1',1.,1e-6,'combined',1.)]
        backend='cpu';until=60.
    results=[]
    for name,cap,dry,clock,outer in specs:
        output=a.output/name
        command=[sys.executable,str(ROOT/'tools/run_speed130_case.py'),
                 '--source',str(source),'--output',str(output),'--backend',backend,
                 '--until',str(until),'--cfl-cap',str(cap),'--dry',str(dry),
                 '--clock',clock,'--outer-dt',str(outer)]
        print('START',name,flush=True)
        with (a.output/(name+'.log')).open('x',encoding='utf-8') as log:
            result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        results.append(dict(name=name,returncode=result.returncode,command=command))
        (a.output/'MATRIX.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
        print('PASS' if result.returncode==0 else 'FAILED',name,flush=True)
    if any(r['returncode'] for r in results):
        raise SystemExit('Some isolated candidates failed; failures retained in MATRIX.json and logs')


if __name__=='__main__':main()
