"""Assess wall-only incident reconstruction without equating quietness to accuracy."""
import json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'desktop/runtime'),str(ROOT),str(ROOT/'desktop')]
from stage_wall172 import save,sha
from review_pair180 import face_ledger
from review_trace179 import metrics
from review_zero175 import load
from review_hydraulics162 import norm
from review_mixing174 import pair,budget
from trace179 import POINTS
from tip178 import exact_rate,velocity
from regular_corner176 import rate as regular_rate,integral
from braids_desktop.results import ResultsStore
from probe_zero175 import energy
from wallpoly181 import geometry
BASE=ROOT/'validation/_incident181'
OUT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_incident181'
SOURCES=dict(bounded='validation/_tip178/probes/gpu/bounded.npz',
 regular='validation/_quadrature176/regular_v2/gpu/RESIDUAL.npz',
 potential='validation/_corner174/potential/gpu/RESIDUAL.npz',
 affine='validation/_corner174/probes_v2/gpu/affine_+0.5.npz',
 developed='validation/_tip178/probes/gpu/BRAIDS200.npz')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    original=dict(np.load(ROOT/SOURCES['bounded']));xy=original['xy']
    ids=np.array([np.flatnonzero(np.all(xy==p,axis=1))[0] for p in POINTS])
    known={};operator={};ledgers={};current={};changed={}
    for name,path in SOURCES.items():
        a=dict(np.load(ROOT/path));b=dict(a);b.update(dict(np.load(BASE/f'probe/wall/gpu/{name}.npz')))
        c=np.load(BASE/f'probe/wall/cpu/{name}.npz');current[name]=a;changed[name]=b
        operator[name]={k:norm(c[k]-b[k]) for k in ('rate','water')}
        interior=a['neighbour']>=0;opened=(a['neighbour']<0)&(a['bid']>=0)
        np.testing.assert_array_equal(a['rec_trace'][:,interior|opened],b['rec_trace'][:,interior|opened])
        np.testing.assert_array_equal(a['water'],b['water'])
        np.testing.assert_array_equal(a['rate'][0],b['rate'][0])
        if name in ('bounded','regular'):
            answer=exact_rate if name=='bounded' else regular_rate
            truth=np.array([answer(*(np.array(p)-[8,3])) for p in POINTS]).T
            row=dict(exact=truth.T.tolist())
            for label,v in [('baseline',a),('wall',b)]:
                e=v['rate'][:,ids]-truth
                row[label]=dict(mass_rms=float(np.sqrt(np.mean(e[0]**2))),
                    momentum_rms=float(np.sqrt(np.mean(e[1:]**2))),rates=v['rate'][:,ids].T.tolist())
            known[name]=row
        if name in ('bounded','developed'):
            ledgers[name]={k:face_ledger(v,name=='bounded') for k,v in [('baseline',a),('wall',b)]}
    data={};hist={};budgets={};parity={};runs={}
    for variant in ('baseline','baseline_half','wall','wall_half'):
        root=ROOT/'validation/_trace179' if variant.startswith('baseline') else BASE
        for backend in ('cpu','gpu'):
            name=variant+'_'+backend;folder=root/'run'/variant/backend
            data[name]=load(folder/'run');hist[name]=metrics(data[name],xy);budgets[name]=budget(folder,'run')
            runs[name]=json.loads((folder/'CHECK.json').read_text())['status']
            f=ResultsStore(folder/'run').read(0);np.testing.assert_array_equal(f.q,current['developed']['q'])
            np.testing.assert_allclose(f.z,current['developed']['z'],atol=2e-15,rtol=0)
        parity[variant]={k:norm(data[variant+'_cpu'][k]-data[variant+'_gpu'][k]) for k in ('h','uv','z')}
    temporal={v:{k:norm(data[v+'_cpu'][k]-data[v+'_half_cpu'][k]) for k in ('h','uv','z')} for v in ('baseline','wall')}
    change={k:norm(data['wall_cpu'][k]-data['baseline_cpu'][k]) for k in ('h','uv','z')}
    controls={};control_parity={};impact={}
    for label,root in [('cropped_fallback',BASE),('active_aligned',BASE/'active_controls')]:
        controls[label]={};control_parity[label]={}
        for case in ('reflection_0','reflection_37','wetting_37','lake_37'):
            for backend in ('cpu','gpu'):
                folder=root/f'controls/{backend}/{case}'
                v=json.loads((folder/'CHECK.json').read_text());v['candidate']=json.loads((folder/'OWNER.json').read_text())['wallstate181']
                controls[label][case+'_'+backend]=v
            c=ResultsStore(root/f'controls/cpu/{case}/full');g=ResultsStore(root/f'controls/gpu/{case}/full')
            control_parity[label][case]=max(float(abs(c.read(j).q-g.read(j).q).max()) for j in range(len(c.times)))
            if case.startswith('reflection'):
                rows=[]
                for p in (root/f'controls/gpu/{case}/probes').glob('*.npz'):
                    a=np.load(p);wall=a['neighbour']<0
                    expected=np.sum(a['q'][1:,a['owner'][wall]]*a['normal'][:,wall],axis=0)
                    e=float(abs(a['local'][1,wall]-expected).max());assert e<1e-14
                    rows.append(dict(name=p.name,incident_momentum_error=e))
                impact[label+'_'+case]=rows
    waves={};waveparity={};wavefields={}
    for variant in ('baseline','wall'):
        for backend in ('cpu','gpu'):
            folder=BASE/f'waves_v5/{variant}/{backend}';name=variant+'_'+backend
            waves[name]=json.loads((folder/'CHECK.json').read_text());waves[name]['budget']=budget(folder,'run')
            wavefields[name]=dict(np.load(folder/'COMPARISON.npz'))
            f=ResultsStore(folder/'run').read(0);initial=np.load(folder/'INITIAL.npz')
            np.testing.assert_array_equal(f.q,initial['q']);assert len(initial['xy'])==80
            np.testing.assert_allclose(f.arrays['area'],.25,atol=1e-13,rtol=0)
        waveparity[variant]={k:norm(wavefields[variant+'_cpu'][k]-wavefields[variant+'_gpu'][k]) for k in ('eta','velocity')}
    _,native,_,_,_,_,native_info=pair(ROOT/'validation/_corner174')
    selected=(native['t']>=200)&(native['t']<=220);native={k:v[selected] for k,v in native.items()};native['t']-=200
    common={k:metrics({f:v[np.isin(data[k]['t'],native['t'])] for f,v in data[k].items()},xy) for k in ('baseline_cpu','wall_cpu')}
    common['FLOW_independent_trajectory']=metrics(native,xy)
    result=dict(known=known,operator=operator,face_ledger=ledgers,history=hist,budget=budgets,backend=parity,
        temporal=temporal,change=change,run_status=runs,controls=controls,control_backend=control_parity,
        uniform_impact=impact,waves=waves,wave_backend=waveparity,same_output_times=common,
        energy={k:energy(a)[0] for k,a in [('baseline',current['developed']),('wall',changed['developed'])]},
        eligible_groin_cells=int((geometry(original)[0]>=0).sum()),
        prototype_reflected_LS=json.loads((BASE/'PROTOTYPE.json').read_text()),
        prototype_all_face_polynomial=json.loads((BASE/'polynomial/CHECK.json').read_text()),
        runtime_sha256=sha(ROOT/'desktop/runtime/RUNTIME.json'),initial_conditions_identical=True,
        physical_viscosity=0.,native_max_viscosity=native_info['max_viscosity'],candidate_adopted=False,
        production_changed=False,Delft3D_changed=False,groin_fixed=False)
    save(OUT/'DIAGNOSTICS.json',result)

    fig,axs=plt.subplots(1,3,figsize=(14,4.8),layout='constrained');c=ids[2];r=int(geometry(original)[0][c])
    s=np.linspace(0,1,201);h=original['q'][0,c];p0=original['q'][2,c];p1=original['q'][2,r]
    exact=np.array([integral(lambda x:velocity(x,float(y))[1],0,.5)/.5 for y in s*.5])
    wallface=np.flatnonzero((original['owner']==c)&(original['neighbour']<0))[0]
    theta=1-changed['bounded']['rec_trace'][2,wallface]/p0
    polynomial=((3.5*p0-.5*p1)*s+(.75*p1-2.25*p0)*s*s)/h
    axs[0].plot(s*.5,exact,'k',label='Analytic cross-strip average')
    axs[0].plot(s*.5,p0/h+original['rec_grad'][2,1,c]*(s-.5)*.5,color='#c76d28',label='Current limited linear fit')
    axs[0].plot(s*.5,p0/h+theta*(polynomial-p0/h),color='#2368ab',label='Bounded quadratic candidate')
    axs[0].axhline(p0/h,color='.5',ls=':',label='Cell average')
    axs[0].set(xlabel='Distance from wall (m)',ylabel='Normal-axis velocity v (m/s)',title='First wall cell: subcell minimum');axs[0].legend(fontsize=7)
    for ax,k,title in [(axs[1],0,'Mass-rate error is unchanged'),(axs[2],2,'Wall momentum error is reduced')]:
        truth=np.array(known['bounded']['exact'])[:,k]
        for offset,label,color in [(-.17,'baseline','#c76d28'),(.17,'wall','#2368ab')]:
            ax.bar(np.arange(5)+offset,np.array(known['bounded'][label]['rates'])[:,k]-truth,.34,label=label,color=color)
        ax.set_xticks(np.arange(5),[f'{p[0]}, {p[1]}' for p in POINTS],rotation=35,fontsize=8)
        ax.set(title=title,ylabel='m/s' if k==0 else 'm²/s²',xlabel='Cell centre x, y (m)');ax.legend(fontsize=8)
    for ax in axs:ax.grid(alpha=.2);ax.axhline(0,color='k',lw=.5)
    fig.suptitle('Known-field limitation: a local wall-pressure improvement is not a complete flux correction')
    fig.savefig(OUT/'INCIDENT_STATE.png',dpi=160);plt.close(fig)

    fig,axs=plt.subplots(1,3,figsize=(14,4.3),layout='constrained');a=wavefields['baseline_cpu'];j=int(np.argmin(abs(a['t']-3.5)))
    mask=a['xy'][:,1]==4.25;order=np.argsort(a['xy'][mask,0]);x=a['xy'][mask,0][order]
    axs[0].plot(x,a['exact_eta'][j,mask][order]*1000,'k',label='Linear analytic wave')
    for variant,color in [('baseline','#c76d28'),('wall','#2368ab')]:
        v=wavefields[variant+'_cpu'];axs[0].plot(x,v['eta'][j,mask][order]*1000,'.-',color=color,label=variant)
        pc=int(np.flatnonzero(np.all(v['xy']==[11.75,4.25],axis=1))[0])
        axs[1].plot(v['t'],v['eta'][:,pc]*1000,color=color,label=variant)
        axs[2].plot(v['t'],np.sqrt(np.mean((v['eta']-v['exact_eta'])**2,axis=1))*1000,color=color,label=variant)
    axs[1].plot(a['t'],a['exact_eta'][:,pc]*1000,'k:',label='Linear analytic wave')
    axs[0].set(xlabel='x (m)',ylabel='Surface perturbation (mm)',title='Reflected pulse at 3.5 s')
    axs[1].set(xlabel='Time (s)',ylabel='Surface perturbation (mm)',title='Last cell before the closed end')
    axs[2].set(xlabel='Time (s)',ylabel='Elevation RMS error (mm)',title='Weakly nonlinear numerical vs linear reference')
    for ax in axs:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Independent wave control: no lost reflection; no wave-accuracy gain either')
    fig.savefig(OUT/'WALL_WAVE.png',dpi=160);plt.close(fig)

    fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for name,color in [('baseline_cpu','#c76d28'),('wall_cpu','#2368ab')]:
        d=data[name];axs[0,0].plot(d['t']+200,d['h'][:,c]+d['z'][:,c],label=name,color=color)
        axs[0,1].plot(d['t']+200,d['uv'][:,1,c],label=name,color=color)
    axs[0,0].plot(native['t']+200,native['h'][:,c]+native['z'][:,c],'k:',label='FLOW independent trajectory')
    axs[0,1].plot(native['t']+200,native['uv'][:,1,c],'k:',label='FLOW independent trajectory')
    axs[0,0].set(ylabel='Water surface (m)',title='First cell above the leading tip')
    axs[0,1].set(ylabel='Transverse velocity (m/s)',title='Same cell: no consistent coupled benefit')
    for ax in axs[0]:ax.set_xlabel('Equivalent original time (s)');ax.legend(fontsize=8);ax.grid(alpha=.2)
    regions=('domain','upstream','groin','wake','outlet');x=np.arange(5)
    for i,label in enumerate(('baseline_cpu','wall_cpu','baseline_half_cpu','wall_half_cpu')):
        axs[1,0].bar(x+(i-1.5)*.19,[hist[label]['regions'][r]['linearly_detrended_eta_rms_mm'] for r in regions],.19,label=label)
    axs[1,0].set_xticks(x,regions);axs[1,0].set(ylabel='Detrended elevation RMS (mm)',title='Regional response at two timesteps');axs[1,0].legend(fontsize=7)
    f=ResultsStore(BASE/'run/wall/cpu/run').read(0);polys=[np.asarray(p.exterior.coords) for p in f.polygons]
    values=(data['wall_cpu']['h'][-1]-data['baseline_cpu']['h'][-1])*1000;lim=abs(values).max()
    p=PolyCollection(polys,array=values,cmap='RdBu_r',clim=(-lim,lim),linewidth=.1,edgecolor='gray')
    ax=axs[1,1];ax.add_collection(p);ax.set(xlim=(4,20),ylim=(0,8),aspect='equal',xlabel='x (m)',ylabel='y (m)',title='Candidate minus current depth at 220 s (mm)')
    ax.fill([8,10,10,8],[0,0,3,3],color='.7');fig.colorbar(p,ax=ax,shrink=.8)
    fig.suptitle('Identical starting fields, 20 s continuation, zero physical viscosity')
    fig.savefig(OUT/'GROIN_COMPARISON.png',dpi=160);plt.close(fig)
    print(json.dumps({k:result[k] for k in ('known','temporal','backend','change','energy','wave_backend')},indent=2))

if __name__=='__main__':main()
