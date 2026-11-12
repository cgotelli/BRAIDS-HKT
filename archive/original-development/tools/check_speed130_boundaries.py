"""Short matched natural-case boundary/backend qualification of selected controls."""
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.analyze_inlet129 import load_run,compare_runs


def main():
    output=ROOT/'validation/_speed130/selected_boundaries'
    output.mkdir(parents=True,exist_ok=False)
    records=[]
    # Critical GPU is already covered by natural_combined_outer2. The CPU pair
    # also covers the same fixed2m support, not a misleading coarse/fine pair.
    for method,backend in [('normal','gpu'),('stage','gpu'),('depth','gpu'),('critical','cpu')]:
        name=method+'_'+backend;target=output/name
        reference=ROOT/f'validation/_inlet129/natural2m_final/2m_{method}_{backend}'
        source=reference/'PROJECT.json'
        command=[sys.executable,str(ROOT/'tools/run_speed130_case.py'),'--source',str(source),
                 '--output',str(target),'--backend',backend,'--method',method,
                 '--until','100','--outer-dt','2','--cfl-cap','1','--clock','combined']
        print('START',name,flush=True)
        with (output/(name+'.log')).open('x') as log:
            result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        row=dict(case=name,command=command,returncode=result.returncode)
        if not result.returncode:
            run=load_run(target);row['summary']=run['summary']
            row['baseline_comparison']=compare_runs(run,load_run(reference))
            if backend=='gpu':
                fine=ROOT/f'validation/_inlet129/natural2m_final_fine/2m_{method}_gpu'
                row['fine_comparison']=compare_runs(run,load_run(fine))
            else:
                row['paired_gpu_comparison']=compare_runs(run,load_run(ROOT/'validation/_speed130/natural_combined_outer2'))
        records.append(row)
        (output/'SUMMARY.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
        print('PASS' if not result.returncode else 'FAILED',name,flush=True)
    if any(r['returncode'] for r in records):raise SystemExit('Failed research cases retained')


if __name__=='__main__':main()
