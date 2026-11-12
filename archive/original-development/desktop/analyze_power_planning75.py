"""Functional CPU/GPU, varying-flow and delivery figures; no new calibration."""
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from power_planning75 import ROOT,REPORT,read,save
HOME=ROOT/'desktop/qualification/power_planning75'
def main():
    cpu=read(HOME/'staggered_cpu/run/POWER_PLANNING.json');gpu=read(HOME/'staggered_gpu/run/POWER_PLANNING.json')
    assert [(r['time_s'],r['turbine_id']) for r in cpu]==[(r['time_s'],r['turbine_id']) for r in gpu]
    gaps={key:max(abs(a[key]-b[key]) for a,b in zip(cpu,gpu)) for key in ['useful_power_W','upstream_velocity_m_s','power_coefficient','thrust_coefficient']}
    assert gaps['useful_power_W']<.01,gaps
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained');summary=[]
    for ax,case in zip(axes.flat,['single','tandem','staggered','variable_qualified']):
        backend='gpu' if case.startswith('variable') else 'cpu';rows=read(HOME/(case+'_'+backend)/'run/POWER_PLANNING.json')
        ids=list(dict.fromkeys(r['turbine_id'] for r in rows))
        for name in ids:
            rr=[r for r in rows if r['turbine_id']==name];ax.plot([r['time_s'] for r in rr],[r['useful_power_W'] for r in rr],label=name)
            summary.append(dict(case=case,turbine=name,first_W=rr[0]['useful_power_W'],last_W=rr[-1]['useful_power_W'],minimum_W=min(r['useful_power_W'] for r in rr),maximum_W=max(r['useful_power_W'] for r in rr)))
        ax.set(title=('Varying discharge' if case.startswith('variable') else case.capitalize())+' — planning estimate',xlabel='Hydraulic time (s)',ylabel='Mechanical shaft power (W)');ax.set_ylim(0,1.15*max(r['useful_power_W'] for r in rows));ax.grid(alpha=.2);ax.legend()
    fig.savefig(REPORT/'POWER_HISTORIES.png',dpi=160);plt.close(fig)
    variable=[r for r in summary if r['case']=='variable_qualified'];assert all(r['maximum_W']-r['minimum_W']>.5 for r in variable)
    water={name:max(abs(r['water_residual_m3']) for r in read(HOME/name/'run/DIAGNOSTICS.json')) for name in ['single_cpu','tandem_cpu','staggered_cpu','staggered_gpu','variable_qualified_gpu']}
    save(REPORT/'METRICS.json',dict(passed=True,backend_max_absolute_difference=gaps,power=summary,maximum_water_residual_m3=water,scope='Fresh 20-second turbine-free hydraulic runs and quasi-steady power estimates, not new transient experimental validation.'))
    print(gaps)
if __name__=='__main__':main()
