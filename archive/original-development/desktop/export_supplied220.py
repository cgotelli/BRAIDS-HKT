"""Analytical pulse export: independent space/time integration of characteristics."""
import sys,os,json
from stage_supplied220 import ROOT,OUT,TARGET,REPORT,sha,save
sys.path[:0]=[str(TARGET),str(ROOT),str(ROOT/'tests')]
os.environ['QT_QPA_PLATFORM']='offscreen';os.environ['OMP_NUM_THREADS']='4'
from braids_core.river_launcher import verify
from braids_core.sediment.portable_river_runtime import install
verify();install(json.loads((TARGET/'PRECOMPILED.json').read_text()),TARGET)
from test_supplied220 import pulse
from test_suspended_core215 import settling_speed
from scipy.special import erf
from scipy.integrate import quad
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    end=30.;u=.2;length=6.;center=.9;sigma=.3;peak=.1;k=settling_speed(.0001,1.65,20.)/.5
    def integral(hi):
        if hi<=0:return 0.
        return peak*sigma*np.sqrt(np.pi/2)*(erf((hi-center)/(np.sqrt(2)*sigma))-erf(-center/(np.sqrt(2)*sigma)))
    initial=integral(length);suspended=np.exp(-k*end)*integral(length-u*end)
    deposited=quad(lambda t:k*np.exp(-k*t)*integral(length-u*t),0,end,epsabs=1e-14)[0]
    exported=initial-suspended-deposited
    expected=dict(initial_inventory=initial,final_inventory=suspended,bed_inventory=deposited,exported=exported)
    rows=[];fine={}
    for backend in ('cpu','gpu'):
        for n in (120,240):
            a=pulse(backend,n,end);budget={key:a[key] for key in expected}
            error=max(abs(budget[key]-value) for key,value in expected.items())
            rows.append(dict(backend=backend,cells=n,budget=budget,budget_max_error=error,relative_initial=error/initial,residual=a['residual']))
            print(json.dumps(rows[-1]),flush=True)
            # Donor-cell advection leaves a diffuse tail at the open boundary.
            # Separate that spatial error from conservation: <=3% coarse,
            # <=1% fine, with improvement required below. No coefficient tuning.
            assert a['exported']>.5*initial and error/initial<(.03 if n==120 else .01) and abs(a['residual'])<2e-14
            if n==240:fine[backend]=a
    for backend in ('cpu','gpu'):
        pair=[r for r in rows if r['backend']==backend]
        assert pair[1]['budget_max_error']<pair[0]['budget_max_error']
    for key in ('mass','deposited'):np.testing.assert_allclose(fine['cpu'][key],fine['gpu'][key],rtol=1e-12,atol=1e-14)
    fig,ax=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    keys=('final_inventory','bed_inventory','exported');labels=('Suspended','Deposited','Exported');x=np.arange(3)
    for i,(label,values) in enumerate((('Analytical',expected),('CPU',fine['cpu']),('GPU',fine['gpu']))):
        ax[0].bar(x+(i-1)*.25,[values[key] for key in keys],width=.25,label=label)
    ax[0].set(xticks=x,xticklabels=labels,ylabel='Mass (kg)',title='Pulse has exited after 30 s');ax[0].legend()
    ax[1].plot(fine['cpu']['x'],fine['cpu']['deposited'],label='CPU deposited')
    ax[1].plot(fine['gpu']['x'][::8],fine['gpu']['deposited'][::8],'.',label='GPU deposited')
    ax[1].set(xlabel='Distance (m)',ylabel='Deposited mass (kg/m²)',title='Deposition left along the transport path');ax[1].legend()
    for a in ax:a.set_ylim(bottom=0);a.grid(alpha=.2)
    fig.suptitle('Frozen-flow analytical export control • one bulk material • no entrainment')
    REPORT.mkdir(parents=True,exist_ok=True);fig.savefig(REPORT/'PULSE_EXPORT.png',dpi=160);plt.close(fig)
    receipt=dict(runtime=sha(TARGET/'RUNTIME.json'),analytic=expected,cases=rows,passed=True)
    save(OUT/'EXPORT.json',receipt);save(REPORT/'EXPORT.json',receipt);print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
