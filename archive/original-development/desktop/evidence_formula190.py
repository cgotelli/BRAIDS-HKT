"""Summarize sequential timing and preserved old-law coupled trajectories."""
import json,sys
from stage_formula190 import ROOT,TARGET,OUT
from stage_secondary185 import save
sys.path[:0]=[str(TARGET),str(ROOT)]
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from braids_desktop.results import ResultsStore
REPORT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_transport190'

def main():
    timings=json.loads((ROOT/'validation/_step190/clean_timing/TIMINGS.json').read_text())
    perf={};fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
    for i,backend in enumerate(('cpu','gpu')):
        old,new=[next(r['status'] for r in timings if r['backend']==backend and r['label']==l)
                 for l in ('baseline','candidate')]
        values=[old['wall_s'],new['wall_s']]
        for x,v,color in zip((i*3,i*3+1),values,('gray','steelblue')):
            ax.bar(x,v,color=color);ax.text(x,v+3,f'{v:.1f} s',ha='center')
        perf[backend]=dict(before_s=values[0],after_s=values[1],speedup=values[0]/values[1],
            reduction_percent=100*(1-values[1]/values[0]),old_rejected=len(old['rejected_trials']),
            new_rejected=len(new['rejected_trials']))
    ax.set_xticks([0,1,3,4],['CPU before','CPU after','GPU before','GPU after'])
    ax.set_ylabel('Worker wall time (s)');ax.set_ylim(0,290)
    ax.set_title('Wang–Fredsøe dry-AMR groin: 16 s simulation\nSequential cold workers, one run per configuration')
    fig.savefig(REPORT/'PERFORMANCE.png',dpi=170);plt.close(fig)
    old_laws={}
    for law in ('mpm','ashida_michiue_generalized','wang_fredsoe','smart_jaggi'):
        for backend in ('cpu','gpu'):
            label=law+'_'+backend
            old=ResultsStore(ROOT/'validation/_bend189/live/groin'/label/'full')
            new=ResultsStore(OUT/'legacy_regression/wet'/label/'full')
            np.testing.assert_array_equal(old.times,new.times)
            fields={k:0. for k in ('q','z','mass','water','bed_boundary')}
            for i in range(len(old.times)):
                a,b=old.read(i),new.read(i)
                for k in ('vertices','offsets','levels'):np.testing.assert_array_equal(a.mesh[k],b.mesh[k])
                for k in fields:fields[k]=max(fields[k],float(np.max(np.abs(a.arrays[k]-b.arrays[k]),initial=0)))
                if i==0:
                    for k in ('q','z','mass'):np.testing.assert_array_equal(a.arrays[k],b.arrays[k])
            assert fields['z']<1e-7 and fields['q']<1e-6,fields
            old_laws[label]=fields
    save(REPORT/'PERFORMANCE.json',dict(timings=perf,solver_threads=2,OMP_NUM_THREADS='4',
        cpu='AMD Ryzen 9 9950X3D, 16 cores / 32 logical processors',gpu='NVIDIA GeForce RTX 5080',
        driver='610.88',repetitions=1,preparation_included=False,construction_step_amr_output_included=True))
    save(OUT/'LEGACY_COMPARISON.json',old_laws)
    print('PERFORMANCE',perf);print('OLD LAW ALL-FRAME DIFFERENCES',old_laws)
if __name__=='__main__':main()
