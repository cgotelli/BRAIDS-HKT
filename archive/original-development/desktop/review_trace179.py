"""Known-answer and short coupled evidence; never rank wake quality by quietness alone."""
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'desktop/runtime'),str(ROOT),str(ROOT/'desktop')]
from stage_wall172 import save,sha
from trace179 import BASE,POINTS
from tip178 import exact_rate
from regular_corner176 import rate as regular_rate
from review_zero175 import load
from review_mixing174 import pair,budget
from review_hydraulics162 import norm
from braids_desktop.results import ResultsStore
from probe_zero175 import energy
OUT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_trace179'


def metrics(d,xy):
    eta=d['h']+d['z'];std=np.std(eta,axis=0)
    basis=np.c_[np.ones(len(d['t'])),d['t']-d['t'][0]]
    detrended=eta-basis@np.linalg.lstsq(basis,eta,rcond=None)[0]
    masks=dict(domain=np.ones(len(xy),bool),upstream=xy[:,0]<7,
        groin=(xy[:,0]>=7)&(xy[:,0]<11),wake=(xy[:,0]>=11)&(xy[:,0]<30),outlet=xy[:,0]>=30)
    return dict(regions={k:dict(eta_std_rms_mm=float(np.sqrt(np.mean(std[m]**2))*1000),
        linearly_detrended_eta_rms_mm=float(np.sqrt(np.mean(detrended[:,m]**2))*1000),
        velocity_std_rms_m_s=float(np.sqrt(np.var(d['uv'][:,:,m],axis=0).mean()))) for k,m in masks.items()},
        minimum_depth_m=float(d['h'].min()),maximum_depth_m=float(d['h'].max()),
        maximum_speed_m_s=float(np.linalg.norm(d['uv'],axis=1).max()),
        bed_change_max_m=float(abs(d['z']-d['z'][0]).max()))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    original=dict(np.load(ROOT/'validation/_tip178/probes/gpu/bounded.npz'))
    xy=original['xy'];ids=np.array([np.flatnonzero(np.all(xy==p,axis=1))[0] for p in POINTS])
    known={};operator={}
    for name in ('bounded','regular','potential','affine','developed'):
        c=np.load(BASE/f'probe/facewise/cpu/{name}.npz');g=np.load(BASE/f'probe/facewise/gpu/{name}.npz')
        operator[name]={k:norm(c[k]-g[k]) for k in ('rate','water')}
    for name,ref,answer in [('bounded',original,exact_rate),
            ('regular',dict(np.load(ROOT/'validation/_quadrature176/regular_v2/gpu/RESIDUAL.npz')),regular_rate)]:
        truth=np.array([answer(*(np.array(p)-[8,3])) for p in POINTS]).T
        candidate=np.load(BASE/f'probe/facewise/gpu/{name}.npz')['rate']
        rows={}
        for label,r in [('baseline',ref['rate']),('facewise',candidate)]:
            e=r[:,ids]-truth
            rows[label]=dict(mass_rate_rms_m_s=float(np.sqrt(np.mean(e[0]**2))),
                momentum_rate_rms_m2_s2=float(np.sqrt(np.mean(e[1:]**2))),rates=r[:,ids].T.tolist())
        known[name]=dict(exact=truth.T.tolist(),**rows)
    datasets={};histories={};parity={};budgets={};checks={}
    for variant in ('baseline','facewise','baseline_half','facewise_half'):
        for backend in ('cpu','gpu'):
            label=variant+'_'+backend;folder=BASE/'run'/variant/backend
            datasets[label]=load(folder/'run');histories[label]=metrics(datasets[label],xy)
            budgets[label]=budget(folder,'run');checks[label]=json.loads((folder/'CHECK.json').read_text())
            f=ResultsStore(folder/'run').read(0)
            source=np.load(BASE/'probe/facewise/cpu/developed.npz')
            assert np.array_equal(f.q,source['q']) and np.max(abs(f.z-source['z']))<2e-15
        parity[variant]={k:norm(datasets[variant+'_cpu'][k]-datasets[variant+'_gpu'][k]) for k in ('h','uv','z')}
    temporal={v:{k:norm(datasets[v+'_cpu'][k]-datasets[v+'_half_cpu'][k]) for k in ('h','uv','z')} for v in ('baseline','facewise')}
    changes={k:norm(datasets['facewise_cpu'][k]-datasets['baseline_cpu'][k]) for k in ('h','uv','z')}
    _,native,_,_,_,_,native_info=pair(ROOT/'validation/_corner174')
    m=(native['t']>=200)&(native['t']<=220);native={k:v[m] for k,v in native.items()}
    native['t']=native['t']-200
    # Compare on identical one-second sample times, without pretending FLOW has
    # the same saved velocity state as the BRAIDS continuation.
    sampled={}
    for name in ('baseline_cpu','facewise_cpu'):
        d=datasets[name];m=np.isin(d['t'],native['t']);sampled[name]=metrics({k:v[m] for k,v in d.items()},xy)
    sampled['FLOW_independent_trajectory']=metrics(native,xy)
    controls={};control_parity={}
    for case in ('reflection_0','reflection_37','wetting_37','lake_37'):
        for b in ('cpu','gpu'):controls[case+'_'+b]=json.loads((BASE/f'controls/{b}/{case}/CHECK.json').read_text())
        c=ResultsStore(BASE/f'controls/cpu/{case}/full');g=ResultsStore(BASE/f'controls/gpu/{case}/full')
        control_parity[case]=dict(q_max=max(float(abs(c.read(j).q-g.read(j).q).max()) for j in range(len(c.times))))
    old=dict(np.load(ROOT/'validation/_tip178/probes/gpu/BRAIDS200.npz'))
    fresh=dict(np.load(BASE/'probe/facewise/gpu/developed.npz'));new=dict(old);new.update(fresh)
    opened=(old['neighbour']<0)&(old['bid']>=0)
    assert np.array_equal(old['fl'][:,opened],new['fl'][:,opened])
    energies={k:energy(a)[0] for k,a in [('baseline',old),('facewise',new)]}
    save(OUT/'DIAGNOSTICS.json',dict(known=known,operator_backend=operator,points_xy=POINTS,
        history=histories,same_output_times=sampled,backend=parity,budget=budgets,
        temporal=temporal,change=changes,controls=controls,control_backend=control_parity,
        same_state_energy=energies,run_status={k:v['status'] for k,v in checks.items()},
        initial_conditions_identical=True,production_changed=False,Delft3D_changed=False,
        native_max_viscosity=native_info['max_viscosity'],runtime_sha256=sha(ROOT/'desktop/runtime/RUNTIME.json')))
    fig,axs=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for row,(name,a) in enumerate(known.items()):
        for col in range(3):
            ax=axs[row,col];x=np.arange(len(ids));ax.plot(x,np.array(a['exact'])[:,col],'k.-',label='Exact face-integrated derivative')
            for label,color in [('baseline','#c76d28'),('facewise','#2368ab')]:ax.plot(x,np.array(a[label]['rates'])[:,col],'o--',color=color,label=label)
            ax.set_xticks(x,[f'{p[0]:g}, {p[1]:g}' for p in POINTS],rotation=30,fontsize=8)
            ax.set(title=name+' / '+('depth rate','x-momentum rate','y-momentum rate')[col],
                ylabel='m/s' if col==0 else 'm²/s²',xlabel='Cell centre x, y (m)');ax.grid(alpha=.2)
            if row==0 and col==0:ax.legend(fontsize=7)
    fig.suptitle('Known-answer corner controls: lower error, but not exact reproduction')
    fig.savefig(OUT/'KNOWN_CORNER.png',dpi=160);plt.close(fig)
    s=ResultsStore(BASE/'run/baseline/cpu/run');polys=[np.asarray(p.exterior.coords) for p in s.read(0).polygons]
    fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained');colors={'baseline_cpu':'#c76d28','facewise_cpu':'#2368ab'}
    point=int(ids[2])
    for name,color in colors.items():
        d=datasets[name];axs[0,0].plot(d['t']+200,d['h'][:,point]+d['z'][:,point],color=color,label=name)
        axs[0,1].plot(d['t']+200,d['uv'][:,1,point],color=color,label=name)
    axs[0,0].plot(native['t']+200,native['h'][:,point]+native['z'][:,point],'k:',label='FLOW independent trajectory')
    axs[0,1].plot(native['t']+200,native['uv'][:,1,point],'k:',label='FLOW independent trajectory')
    for ax,title,ylabel in [(axs[0,0],'First cell above leading tip','Water elevation (m)'),(axs[0,1],'Same raw cell; zero added viscosity','Transverse velocity (m/s)')]:
        ax.set(title=title,xlabel='Equivalent original time (s)',ylabel=ylabel);ax.grid(alpha=.2);ax.legend(fontsize=8)
    for ax,key,title in [(axs[1,0],'variance','Change in water-surface standard deviation (mm)'),(axs[1,1],'depth','Candidate minus baseline depth at 220 s (mm)')]:
        b=datasets['baseline_cpu']['h'];a=datasets['facewise_cpu']['h']
        val=(np.std(a,axis=0)-np.std(b,axis=0))*1000 if key=='variance' else (a[-1]-b[-1])*1000
        lim=float(abs(val).max());pc=PolyCollection(polys,array=val,cmap='RdBu_r',clim=(-lim,lim),linewidth=.1,edgecolor='gray')
        ax.add_collection(pc);ax.set(xlim=(4,20),ylim=(0,8),aspect='equal',xlabel='x (m)',ylabel='y (m)',title=title)
        ax.fill([8,10,10,8],[0,0,3,3],color='.7');fig.colorbar(pc,ax=ax,shrink=.8)
    fig.suptitle('20 s continuation from exactly the same BRAIDS state — not a settled-flow proof')
    fig.savefig(OUT/'SHORT_CONTINUATION.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    regions=('domain','upstream','groin','wake','outlet');x=np.arange(len(regions))
    for i,label in enumerate(('baseline_cpu','facewise_cpu','baseline_half_cpu','facewise_half_cpu')):
        axs[0].bar(x+(i-1.5)*.19,[histories[label]['regions'][r]['eta_std_rms_mm'] for r in regions],.19,label=label)
    axs[0].set_xticks(x,regions);axs[0].set(ylabel='RMS of cell temporal standard deviations (mm)',title='Spatial pattern is robust to halving CFL');axs[0].legend(fontsize=7)
    for label,color in colors.items():
        base=label.removesuffix('_cpu');a=datasets[label]['h'];b=datasets[base+'_half_cpu']['h']
        axs[1].plot(datasets[label]['t']+200,abs(a-b).max(axis=1)*1000,color=color,label=base+' maximum')
        axs[1].plot(datasets[label]['t']+200,np.sqrt(np.mean((a-b)**2,axis=1))*1000,color=color,ls=':',label=base+' RMS')
    axs[1].set(xlabel='Equivalent original time (s)',ylabel='Depth difference (mm)',title='CFL 1 versus 0.5, same model and initial state');axs[1].legend(fontsize=8)
    for ax in axs:ax.grid(axis='y',alpha=.2)
    fig.savefig(OUT/'SENSITIVITY.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(known=known,temporal=temporal,change=changes,backend=parity,energy=energies),indent=2))


if __name__=='__main__':main()
