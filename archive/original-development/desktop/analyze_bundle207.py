"""Raw-cell Desktop backend/lifecycle comparisons; no smoothing or tuning."""
import sys,json
from pathlib import Path
from stage_bundle207 import ROOT,OUT,TARGET,sha,save
sys.path[:0]=[str(TARGET),str(ROOT)]
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize,TwoSlopeNorm
from braids_desktop.results import ResultsStore
REPORT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_bundle207'
REPORT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
records={};stores={}
for scenario in ('front','bank','open','oblique_wet'):
    stores[scenario]={b:ResultsStore(OUT/'release_v3'/(scenario+'_'+b)/'run') for b in ('cpu','gpu')}
    cpu,gpu=stores[scenario].values();assert np.array_equal(cpu.times,gpu.times)
    maximum={k:0. for k in ('depth_mm','bed_mm','mobile_mm','intensity_m_s')}
    for i in range(len(cpu.times)):
        a=cpu.read(i);b=gpu.read(i);np.testing.assert_array_equal(a.mesh['leaves'],b.mesh['leaves'])
        for key,av,bv,scale in [('depth_mm',a.q[0],b.q[0],1000),('bed_mm',a.z,b.z,1000),
            ('mobile_mm',a.arrays['mobile_bedload_solid_m'],b.arrays['mobile_bedload_solid_m'],1000),
            ('intensity_m_s',a.values['secondary_intensity_m_s'],b.values['secondary_intensity_m_s'],1)]:
            maximum[key]=max(maximum[key],float(np.max(abs(av-bv)))*scale)
    assert maximum['depth_mm']<.01 and maximum['bed_mm']<.01,maximum
    runs={b:json.loads((OUT/'release_v3'/(scenario+'_'+b)/'run/STATUS.json').read_text()) for b in ('cpu','gpu')}
    restart={b:json.loads((OUT/'release_v3'/(scenario+'_'+b)/'RESTART.json').read_text()) for b in ('cpu','gpu')}
    assert all(all(r['exact'].values()) for r in restart.values())
    records[scenario]=dict(maximum_backend=maximum,runs=runs,restart=restart)

def paint(ax,frame,values,title,cmap,limit):
    verts=[np.asarray(p.exterior.coords) for p in frame.polygons]
    coll=PolyCollection(verts,array=values,cmap=cmap,norm=Normalize(*limit),edgecolors='#465260',linewidths=.25)
    ax.add_collection(coll);ax.autoscale_view();ax.set_aspect('equal');ax.set_title(title);ax.set_xlabel('x (m)');ax.set_ylabel('y (m)')
    return coll

for scenario,filename in [('front','FRONT_COMPARISON.png'),('bank','BANK_COMPARISON.png'),('oblique_wet','OBLIQUE_COMPARISON.png')]:
    a,b=(s.read(-1) for s in stores[scenario].values());fig,axes=plt.subplots(2,3,figsize=(13,6),layout='constrained')
    for row,(va,vb,label,cmap) in enumerate([(a.q[0]*1000,b.q[0]*1000,'depth (mm)','Blues'),
            ((a.z-a.arrays['initial_bed_m'])*1000,(b.z-b.arrays['initial_bed_m'])*1000,'bed change (mm)','RdBu_r')]):
        if row==0:limit=(0,max(1e-9,float(max(va.max(),vb.max()))))
        else:
            bound=max(1e-9,float(max(abs(va).max(),abs(vb).max())));limit=(-bound,bound)
        for col,(f,v,backend) in enumerate([(a,va,'CPU'),(b,vb,'GPU')]):
            c=paint(axes[row,col],f,v,backend+' '+label,cmap,limit);fig.colorbar(c,ax=axes[row,col],shrink=.7)
        diff=vb-va;bound=max(1e-12,float(abs(diff).max()))
        c=paint(axes[row,2],a,diff,'GPU − CPU (mm)','RdBu_r',(-bound,bound));fig.colorbar(c,ax=axes[row,2],shrink=.7)
    fig.suptitle(f'Desktop research bundle — {scenario}, t = {a.time_s:g} s\nRaw polygons; backend error has its own much smaller colour scale')
    fig.savefig(REPORT/filename,dpi=170);plt.close(fig)

fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
for backend,style in [('cpu','-'),('gpu','--')]:
    front=stores['front'][backend]
    axes[0,0].step(front.times,[len(front.read(i).polygons) for i in range(len(front.times))],style,where='post',label=backend.upper())
    bank=OUT/'release_v3'/('bank_'+backend);diag=json.loads((bank/'run/DIAGNOSTICS.json').read_text())
    axes[0,1].plot([r['time_s'] for r in diag],[r['cumulative_collapse_gross_solid_m3']*1e6 for r in diag],style,label=backend.upper())
    for kind,color in [('solid_residual_m3','tab:brown'),('water_plus_pore_residual_m3','tab:blue')]:
        diag=json.loads((OUT/'release_v3'/('front_'+backend)/'run/DIAGNOSTICS.json').read_text())
        axes[1,0].plot([r['time_s'] for r in diag],[r[kind] for r in diag],style,color=color,label=backend+' '+kind.split('_')[0])
    water=front.boundary_flow('inlet',magnitude=False);solid=front.boundary_flow('inlet',quantity='bedload',magnitude=False)
    axes[1,1].plot(water.time_s,water.values+solid.values/2650,style,label=backend.upper()+' mixture Q')
axes[0,0].axhline(64,color='grey',ls=':',label='uniform finest');axes[0,0].set_title('Real AMR mesh changes');axes[0,0].set_ylabel('Cells')
axes[0,1].set_title('Selected repose collapse (closed bank)');axes[0,1].set_ylabel('Gross redistributed solid (cm³)')
axes[1,0].set_title('Open-front phase-budget residuals');axes[1,0].set_ylabel('m³')
axes[1,1].axhline(.024,color='grey',ls=':',label='prescribed Q');axes[1,1].set_title('Accepted inlet: water + solid');axes[1,1].set_ylabel('m³/s');axes[1,1].set_ylim(.02399,.02401)
for ax in axes.flat:ax.set_xlabel('Time (s)');ax.legend(fontsize=8);ax.grid(alpha=.15)
fig.savefig(REPORT/'LIFECYCLE.png',dpi=170);plt.close(fig)
save(REPORT/'COMPARISON.json',dict(runtime=sha(TARGET/'RUNTIME.json'),cases=records))
print(json.dumps({k:v['maximum_backend'] for k,v in records.items()},indent=2))
