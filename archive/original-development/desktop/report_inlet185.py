"""Compact evidence for the held release's common-state inlet diagnosis."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from braids_desktop.results import ResultsStore
from stage_secondary185 import OUT,BASE,TARGET,sha,save

REPORT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_secondary185'
PAIRS=(
 ('Common fields, unchanged v6','common_cpu_20','common_gpu_20'),
 ('Outlet bounds only','outlet_cpu','outlet_gpu'),
 ('Outlet bounds and continuous face flux','continuous_cpu','continuous_gpu'),
 ('Same corrections, outer dt 0.05 s','continuous_dt05_cpu','continuous_dt05_gpu'),
 ('Face-local limiter prototype','facebounds_cpu','facebounds_gpu'),
 ('Secondary momentum feedback off contrast','feedbackoff_cpu','feedbackoff_gpu'),
 ('Secondary bed steering off contrast','directionoff_cpu','directionoff_gpu'),
 ('Koch-Flokstra on contrast','slopeon_cpu','slopeon_gpu'),
 ('Koch-Flokstra on, independent warm-up fields','slopeon_cpu','slopeon_own_gpu'),
 ('Koch-Flokstra on, original numerical operators','originalslope_cpu','originalslope_gpu'),
)


def data(tag):return np.load(OUT/'replay'/tag/'FIELDS.npz')
def read(tag,name):return json.loads((OUT/'replay'/tag/name).read_text())


def main():
    rows=[]
    for label,c,g in PAIRS:
        a,b=data(c),data(g);xy=data('slopeon_cpu')['geometry_xy'];r=dict(label=label,cpu=c,gpu=g,fields={})
        assert a['final_z'].shape==b['final_z'].shape==xy[0].shape
        r['common_initial_fields']=all(np.array_equal(a[k],b[k]) for k in ('initial_q','initial_z'))
        for name,x,y in [('bed_m',a['final_z'],b['final_z']),('depth_m',a['final_q'][0],b['final_q'][0]),
                         ('momentum_m2_s',a['final_q'][1:],b['final_q'][1:]),('hI_m2_s',a['final_W'],b['final_W'])]:
            d=abs(x-y);idx=np.unravel_index(d.argmax(),d.shape)
            r['fields'][name]=dict(max_abs=float(d.max()),rms=float(np.sqrt(np.mean(d*d))),
                peak_xy_m=xy[:,idx[-1]].tolist())
        inlet=xy[0]<5.
        r['inlet_x_below_5m']={name:dict(max_abs=float(abs(x-y).max()),
            rms=float(np.sqrt(np.mean((x-y)**2)))) for name,x,y in
            [('bed_m',a['final_z'][inlet],b['final_z'][inlet]),
             ('depth_m',a['final_q'][0,inlet],b['final_q'][0,inlet])]}
        r['budgets']={}
        for tag in (c,g):
            steps=read(tag,'STEPS.json')
            r['budgets'][tag]={k:max(abs(s.get(k,0.)) for s in steps) for k in
                ('water_residual_m3','secondary_residual','solid_volume_residual_m3','class_flux_budget_max_kg')}
            assert all(v<1e-9 for v in r['budgets'][tag].values()),r['budgets']
        rows.append(r)
    save(REPORT/'INLET_DIAGNOSIS.json',dict(installed_unchanged=sha(BASE/'RUNTIME.json'),
        staged_unchanged=sha(TARGET/'RUNTIME.json'),replays=rows,
        scope='200 to 220 s physical-field replay, fresh ledgers, no checkpoint migration; MF5',
        installation='held by user; diagnostic changes not staged or installed'))
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for prefix,label in [('outlet','Outlet bounds only'),('continuous','+ Continuous face flux'),
                         ('facebounds','Face-local limiter prototype'),('slopeon','+ Bed-slope physics')]:
        a,b=data(prefix+'_cpu'),data(prefix+'_gpu')
        times=[s['time'] for s in read(prefix+'_cpu','STEPS.json')]
        d=abs(a['history']-b['history'])
        for ax,component in zip(axes,(3,0)):
            ax.semilogy(times,np.maximum(1e-10,1000*d[:,component].max(axis=1)),label=label)
    for ax,title in zip(axes,('Maximum bed difference','Maximum water-depth difference')):
        ax.axhline(1.,color='black',ls='--',lw=1,label='1 mm release screen')
        ax.set(title=title,xlabel='Physical time (s)',ylabel='CPU/GPU difference (mm)',ylim=(1e-10,100.))
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Common starting fields: numerical corrections versus an explicit physics contrast')
    fig.savefig(REPORT/'INLET_GROWTH.png',dpi=160);plt.close(fig)
    frame=ResultsStore(OUT/'qualified_cpu_v6/to_220').read(-1)
    polys=[np.asarray(p.exterior.coords) for p in frame.polygons]
    cases=[('continuous_cpu','Original bed-slope setting: OFF'),('slopeon_cpu','Diagnostic bed-slope setting: ON')]
    fig,axes=plt.subplots(2,2,figsize=(13,7),layout='constrained')
    for row,(tag,label) in enumerate(cases):
        a=data(tag);b=data('continuous_gpu' if row==0 else 'slopeon_own_gpu')
        fields=[1000*(a['final_z']-a['initial_z']),1000*(b['final_z']-a['final_z'])]
        for col,(field,limit) in enumerate(zip(fields,(120.,8.))):
            pc=PolyCollection(polys,array=field,cmap='RdBu_r',edgecolors='#666666',linewidths=.15)
            pc.set_clim(-limit,limit);ax=axes[row,col];ax.add_collection(pc)
            ax.set(xlim=(0,5),ylim=(-2.1,2.1),xlabel='x (m)',ylabel='y (m)',
                title=label+(' | CPU bed change (mm)' if col==0 else ' | GPU minus CPU bed (mm)'))
            ax.set_aspect('equal');fig.colorbar(pc,ax=ax,shrink=.8)
    fig.suptitle('Raw inlet cells after 20 s of bedload at MF5; bottom comparison uses independent warm-up fields')
    fig.savefig(REPORT/'INLET_CELLS.png',dpi=170);plt.close(fig)
    print(json.dumps([{r['label']:{k:1000*r['fields'][k]['max_abs'] for k in ('bed_m','depth_m')}} for r in rows],indent=2))


if __name__=='__main__':main()
