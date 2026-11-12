"""Trace the first Exner update and short hydraulic inlet contrasts."""
import json
import numpy as np
from stripes_mesh186 import ROOT,REPORT,frame,map_field
from stage_secondary185 import save
from braids_desktop.results import ResultsStore
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=ROOT/'validation/_mesh186'


def inlet_series(path):
    store=ResultsStore(path);f0=store.read(0)
    xy=np.array([p.centroid.coords[0] for p in f0.polygons]).T;ids=np.flatnonzero(xy[0]<5)
    rows=[]
    for t in range(21):
        f=store.read(int(np.flatnonzero(np.isclose(store.times,t,rtol=0,atol=1e-8))[0]))
        v=f.q[2]/f.q[0];I=f.arrays['W']/f.q[0];k=ids[np.argmax(abs(I[ids]))]
        rows.append(dict(time_s=t,v_max=float(abs(v[ids]).max()),I_max=float(abs(I[ids]).max()),
                         I_peak_xy=xy[:,k].tolist()))
    return rows


def main():
    stats=dict(first_update={},hydraulic_controls={})
    fig,axs=plt.subplots(2,3,figsize=(17,7),layout='constrained')
    for r,mesh in enumerate(('coarse','refined')):
        f=np.load(OUT/f'replay/{mesh}_first/FIELDS.npz');xy=f['geometry_xy'];mask=xy[0]<5
        o=f['geometry_owner'];j=f['geometry_neighbour'];L=f['geometry_length'];A=f['geometry_area'];N=f['geometry_normal'];n=len(A);inter=j>=0
        flux=f['bed_00_class_flux'][0]
        def div(v):
            return (-np.bincount(o,weights=L*v,minlength=n)+np.bincount(j[inter],weights=(L*v)[inter],minlength=n))/A
        # dt=0.2, MF=5, solid fraction=0.6: this is the actual first-step dz.
        total=div(flux)/.6
        xpart=div(np.where(abs(N[0])>.5,flux,0))/.6
        ypart=div(np.where(abs(N[1])>.5,flux,0))/.6
        actual=f['final_z']-f['initial_z'];closure=float(abs(actual-total).max())
        assert closure<1e-12
        constant=float(max(abs(div(N[0])[mask]).max(),abs(div(N[1])[mask]).max()))
        assert constant<1e-10
        row=dict(Exner_reconstruction_error_m=closure,constant_vector_divergence_max=constant,components={})
        for c,(name,a) in enumerate((('Total',total),('x-face contribution',xpart),('y-face contribution',ypart))):
            ids=np.flatnonzero(mask);k=ids[np.argmax(abs(a[mask]))]
            row['components'][name]=dict(min_mm=float(a[mask].min()*1000),max_mm=float(a[mask].max()*1000),peak_xy=xy[:,k].tolist())
            base=frame(OUT/f'{mesh}_cpu/run',200)
            pc=map_field(axs[r,c],base,a*1000,-5,5,f'{mesh.title()} | {name}, first 0.2 s (mm)')
        stats['first_update'][mesh]=row
    fig.colorbar(pc,ax=axs.ravel().tolist(),shrink=.7,label='Bed change (mm)')
    fig.suptitle('Exact first Exner update: the refined stripes are dominated by transverse sediment flux')
    fig.savefig(REPORT/'FIRST_BED_UPDATE.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(13,5.5))
    for tag,label in [('refined_cpu','Refined, unchanged baseline'),('coarse_cpu','Coarse, unchanged baseline'),
            ('betac_zero_cpu','Refined, secondary momentum OFF (viscosity retained)'),
            ('matched_energy_slope_cpu','Refined, same energy slope as coarse'),
            ('zero_incoming_secondary_cpu','Refined, incoming intensity zero, CPU'),
            ('zero_incoming_secondary_gpu','Refined, incoming intensity zero, GPU')]:
        rows=inlet_series(OUT/tag/'run');stats['hydraulic_controls'][tag]=rows
        for ax,key in zip(axs,('v_max','I_max')):
            ax.plot([r['time_s'] for r in rows],[r[key] for r in rows],label=label,
                    ls='--' if tag.endswith('gpu') else '-')
    for ax,title in zip(axs,('Maximum transverse velocity in inlet reach','Maximum secondary intensity in inlet reach')):
        ax.set(title=title,xlabel='Physical time (s)',ylabel='m/s');ax.grid(alpha=.2)
    handles,labels=axs[1].get_legend_handles_labels()
    fig.legend(handles,labels,fontsize=8,loc='lower center',ncol=2)
    fig.suptitle('Short hydraulic controls: bed fixed, same refined mesh and wet initial condition')
    fig.tight_layout(rect=(0,.19,1,.94))
    fig.savefig(REPORT/'INLET_FEEDBACK.png',dpi=170);plt.close(fig)
    a=frame(OUT/'zero_incoming_secondary_cpu/run',20);b=frame(OUT/'zero_incoming_secondary_gpu/run',20)
    stats['zero_incoming_backend']=dict(depth_max_m=float(abs(a.q[0]-b.q[0]).max()),
        momentum_max_m2_s=float(abs(a.q[1:]-b.q[1:]).max()),W_max_m2_s=float(abs(a.arrays['W']-b.arrays['W']).max()))
    save(REPORT/'STRIPE_TRACE.json',stats)
    print(json.dumps({k:v[-1] for k,v in stats['hydraulic_controls'].items()},indent=2))
    print(json.dumps(stats['first_update'],indent=2))
    print(stats['zero_incoming_backend'])


if __name__=='__main__':main()
