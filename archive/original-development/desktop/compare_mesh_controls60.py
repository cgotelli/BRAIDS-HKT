"""Same-time causal comparison; retain unsuccessful rotor-only refinement."""
from finalize_nearwake60 import *


def compare(include_eight=False):
    old=Path.home()/'Desktop/BRAIDS_Turbine_Validation59'
    baseline=read(old/'02_Nash_Equivalent_Load_GPU/LAST_VALIDATION_RUN.json')['output']
    zero=read(old/'04_Matched_Zero_Thrust_GPU/LAST_VALIDATION_RUN.json')['output']
    control=saved_case(zero,'zero',500.,600.)
    cases={'Original D/2':saved_case(baseline,'original',500.,600.)}
    for label,name in [('D/4 rotor only','rotor4_gpu'),('D/4 rotor + wake','wake4_gpu')]+([('D/8 rotor + wake','wake8_gpu')] if include_eight else []):
        output=read(REPORT/name/'STARTED.json')['output'];cases[label]=saved_case(output,name,500.,600.)
    fig,axes=plt.subplots(1,3,figsize=(13,4.3),layout='constrained');data={};xx=np.linspace(2,6,161)
    for label,c in cases.items():
        axes[0].plot(xx,100*deficit(c,control,xx,0.),label=label)
        ys=np.linspace(-2,2,161)
        for ax,x in zip(axes[1:],[2.,4.]):ax.plot(100*deficit(c,control,x,ys),ys,label=label)
        stations=np.array([2.,2.5,3.,3.5,4.])
        data[label]=dict(times_s=c['times'].tolist(),stations_D=stations.tolist(),deficit_percent=(100*deficit(c,control,stations,0.)).tolist())
    axes[0].set(xlabel='x/D',ylabel='Velocity deficit (%)',title='Same-time centreline');axes[0].legend(fontsize=8)
    for ax,x in zip(axes[1:],[2.,4.]):ax.set(xlabel='Velocity deficit (%)',ylabel='y/D',title=f'{x:g}D, unsmoothed profiles')
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('500–600 s in every case; causal mesh control, not final far-wake acceptance')
    name='SAME_TIME_CONTROLS_D8' if include_eight else 'SAME_TIME_CONTROLS'
    fig.savefig(REPORT/(name+'.png'),dpi=170);plt.close(fig);save(REPORT/(name+'.json'),data);print(json.dumps(data,indent=2))


if __name__=='__main__':compare('--eight' in sys.argv)
