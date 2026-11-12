"""Face-by-face corner balance and bounded CPU/GPU candidate assessment."""
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
from pair180 import BASE
from trace179 import POINTS
from tip178 import velocity,exact_rate
from regular_corner176 import integral,rate as regular_rate
from review_zero175 import load
from review_trace179 import metrics
from review_hydraulics162 import norm
from review_mixing174 import pair,budget
from braids_desktop.results import ResultsStore
from quadrature176 import gather
from probe_zero175 import energy
OUT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_pair180'


def exact_flux(a,f):
    """Independent face average of the compact known field, oriented owner-out."""
    p=a['mid'][:,f]-[8.,3.];n=a['normal'][:,f];axis=int(np.argmax(abs(n)))
    other=1-axis;length=a['length'][f]
    def value(s):
        r=p.copy();r[other]=s;v=velocity(*r);un=np.dot(v,n)
        return .5*un*np.r_[1.,v]+np.r_[0.,.5*9.81*.5**2*n]
    return np.array([integral(lambda s:value(s)[k],p[other]-length/2,p[other]+length/2)/length for k in range(3)])


def face_ledger(a,known=False):
    rows={}
    for point in ([8.25,3.25],[8.25,3.75]):
        c=int(np.flatnonzero(np.all(a['xy']==point,axis=1))[0]);rr=[]
        for f in np.flatnonzero((a['owner']==c)|(a['neighbour']==c)):
            owner=a['owner'][f]==c;sign=-1 if owner else 1
            t=a['rec_trace'][:,f];n=a['normal'][:,f];h=t[[0,4]]
            uv=np.array([t[1:3]/h[0],t[5:7]/h[1]])
            un=uv@n;opened=a['neighbour'][f]<0 and a['bid'][f]>=0
            wall=a['neighbour'][f]<0 and a['bid'][f]<0
            actual=a['fl' if owner else 'fr'][:,f]
            adv=np.mean([h[s]*un[s]*np.r_[1.,uv[s]] for s in (0,1)],axis=0)
            if wall:adv[:]=0.
            pressure=np.r_[0.,.25*9.81*np.sum(h*h)*n]
            side=0 if owner else 1
            bed=np.r_[0.,.5*9.81*(a['q'][0,c]+t[8+side])*(t[3+4*side]-a['z'][c])*n]
            remainder=actual-adv-pressure-bed
            coefficient=sign*a['length'][f]/a['area'][c]
            row=dict(face=int(f),mid=a['mid'][:,f].tolist(),normal=n.tolist(),
                boundary='wall' if wall else 'open' if opened else 'fluid',
                normal_velocities=un.tolist(),velocity=uv.tolist(),depths=h.tolist(),
                flux=actual.tolist(),rate_contribution=(coefficient*actual).tolist(),
                central_advection_rate=(coefficient*adv).tolist(),
                central_pressure_rate=(coefficient*pressure).tolist(),
                fwave_remainder_rate=(coefficient*remainder).tolist(),bed_rate=(coefficient*bed).tolist(),
                right_trace_role='unused storage copy; wall pressure uses left incident state' if wall else 'neighbour state')
            if known:
                ex=exact_flux(a,f)
                row.update(exact_midpoint_velocity=velocity(*(a['mid'][:,f]-[8,3])).tolist(),
                    exact_flux=ex.tolist(),exact_rate_contribution=(coefficient*ex).tolist(),
                    error_contribution=(coefficient*(actual-ex)).tolist())
            rr.append(row)
        total=np.sum([r['rate_contribution'] for r in rr],axis=0)
        np.testing.assert_allclose(total,a['rate'][:,c],rtol=0,atol=2e-12)
        if known:np.testing.assert_allclose(np.sum([r['exact_rate_contribution'] for r in rr],axis=0),exact_rate(*(np.array(point)-[8,3])),atol=2e-12,rtol=0)
        rows[str(point)]=dict(faces=rr,rate=total.tolist())
    return rows


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    reference=dict(np.load(ROOT/'validation/_tip178/probes/gpu/bounded.npz'))
    xy=reference['xy'];ids=np.array([np.flatnonzero(np.all(xy==p,axis=1))[0] for p in POINTS])
    operators={};known={};ledgers={};datasets={};histories={};budgets={};parity={};checks={}
    for name in ('bounded','regular','potential','affine','developed'):
        c=np.load(BASE/f'probe/normal/cpu/{name}.npz');g=np.load(BASE/f'probe/normal/gpu/{name}.npz')
        operators[name]={k:norm(c[k]-g[k]) for k in ('rate','water')}
    for name,base,answer in [('bounded',reference,exact_rate),('regular',dict(np.load(ROOT/'validation/_quadrature176/regular_v2/gpu/RESIDUAL.npz')),regular_rate)]:
        truth=np.array([answer(*(np.array(p)-[8,3])) for p in POINTS]).T;rows={}
        for label,a in [('baseline',base),('normal',np.load(BASE/f'probe/normal/gpu/{name}.npz'))]:
            error=a['rate'][:,ids]-truth
            rows[label]=dict(mass_rate_rms_m_s=float(np.sqrt(np.mean(error[0]**2))),
                momentum_rate_rms_m2_s2=float(np.sqrt(np.mean(error[1:]**2))),rates=a['rate'][:,ids].T.tolist())
        known[name]=dict(exact=truth.T.tolist(),**rows)
    for kind,oldfile,newfile in [('known','bounded','bounded'),('developed','BRAIDS200','developed')]:
        a=dict(np.load(ROOT/f'validation/_tip178/probes/gpu/{oldfile}.npz'));b=dict(a)
        b.update(dict(np.load(BASE/f'probe/normal/gpu/{newfile}.npz')))
        ledgers[kind]={name:face_ledger(state,kind=='known') for name,state in [('baseline',a),('normal',b)]}
        if kind=='known':
            ledgers[kind]['oracle_interior_normal']=face_ledger(dict(np.load(BASE/'known_oracle_interior_normal.npz')),True)
    for variant in ('baseline','normal','baseline_half','normal_half'):
        root=BASE if variant.startswith('normal') else ROOT/'validation/_trace179'
        for backend in ('cpu','gpu'):
            label=variant+'_'+backend;folder=root/'run'/variant/backend
            datasets[label]=load(folder/'run');histories[label]=metrics(datasets[label],xy)
            budgets[label]=budget(folder,'run');checks[label]=json.loads((folder/'CHECK.json').read_text())
            frame=ResultsStore(folder/'run').read(0);source=np.load(BASE/'probe/normal/cpu/developed.npz')
            np.testing.assert_array_equal(frame.q,source['q']);np.testing.assert_allclose(frame.z,source['z'],rtol=0,atol=2e-15)
        parity[variant]={k:norm(datasets[variant+'_cpu'][k]-datasets[variant+'_gpu'][k]) for k in ('h','uv','z')}
    temporal={v:{k:norm(datasets[v+'_cpu'][k]-datasets[v+'_half_cpu'][k]) for k in ('h','uv','z')} for v in ('baseline','normal')}
    changes={k:norm(datasets['normal_cpu'][k]-datasets['baseline_cpu'][k]) for k in ('h','uv','z')}
    _,native,_,_,_,_,native_info=pair(ROOT/'validation/_corner174')
    m=(native['t']>=200)&(native['t']<=220);native={k:v[m] for k,v in native.items()};native['t']-=200
    sampled={}
    for name in ('baseline_cpu','normal_cpu'):
        d=datasets[name];m=np.isin(d['t'],native['t']);sampled[name]=metrics({k:v[m] for k,v in d.items()},xy)
    sampled['FLOW_independent_trajectory']=metrics(native,xy)
    controls={};control_parity={}
    for case in ('reflection_0','reflection_37','wetting_37','lake_37'):
        for b in ('cpu','gpu'):controls[case+'_'+b]=json.loads((BASE/f'controls/{b}/{case}/CHECK.json').read_text())
        c=ResultsStore(BASE/f'controls/cpu/{case}/full');g=ResultsStore(BASE/f'controls/gpu/{case}/full')
        control_parity[case]=dict(q_max=max(float(abs(c.read(j).q-g.read(j).q).max()) for j in range(len(c.times))))
    old=dict(np.load(ROOT/'validation/_tip178/probes/gpu/BRAIDS200.npz'));new=dict(old)
    new.update(dict(np.load(BASE/'probe/normal/gpu/developed.npz')))
    opened=(old['neighbour']<0)&(old['bid']>=0)
    np.testing.assert_array_equal(old['fl'][:,opened],new['fl'][:,opened])
    energies={k:energy(a)[0] for k,a in [('baseline',old),('normal',new)]}
    result=dict(known=known,operator_backend=operators,points_xy=POINTS,face_ledger=ledgers,
        history=histories,same_output_times=sampled,backend=parity,budget=budgets,temporal=temporal,
        change=changes,controls=controls,control_backend=control_parity,same_state_energy=energies,
        run_status={k:v['status'] for k,v in checks.items()},initial_conditions_identical=True,
        production_changed=False,Delft3D_changed=False,candidate_adopted=False,groin_fixed=False,
        native_max_viscosity=native_info['max_viscosity'],runtime_sha256=sha(ROOT/'desktop/runtime/RUNTIME.json'),
        diagnostic_jump_blending=json.loads((BASE/'ATTRIBUTION.json').read_text()))
    save(OUT/'DIAGNOSTICS.json',result)

    fig,axs=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for row,(name,a) in enumerate(known.items()):
        for col in range(3):
            ax=axs[row,col];x=np.arange(len(ids));ax.plot(x,np.array(a['exact'])[:,col],'k.-',label='Independent exact derivative')
            for label,color in [('baseline','#c76d28'),('normal','#2368ab')]:ax.plot(x,np.array(a[label]['rates'])[:,col],'o--',color=color,label=label)
            ax.set_xticks(x,[f'{p[0]:g}, {p[1]:g}' for p in POINTS],rotation=30,fontsize=8)
            ax.set(title=name+' / '+('depth rate','x-momentum rate','y-momentum rate')[col],
                ylabel='m/s' if col==0 else 'm²/s²',xlabel='Cell centre x, y (m)');ax.grid(alpha=.2)
            if row==0 and col==0:ax.legend(fontsize=7)
    fig.suptitle('Normal-only reflected reconstruction: improved local derivatives, not a complete fix')
    fig.savefig(OUT/'KNOWN_CORNER.png',dpi=160);plt.close(fig)

    fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for row,point in enumerate(('[8.25, 3.25]','[8.25, 3.75]')):
        for col,k in enumerate((0,2)):
            ax=axs[row,col]
            for j,(label,color) in enumerate([('baseline','#c76d28'),('normal','#2368ab'),('oracle_interior_normal','#888888')]):
                faces=ledgers['known'][label][point]['faces'];x=np.arange(len(faces))
                ax.bar(x+(j-1)*.24,[r['error_contribution'][k] for r in faces],.24,label=label,color=color)
            ax.set_xticks(x,[f"{r['mid'][0]:g},{r['mid'][1]:g}\n{r['boundary']}" for r in faces],fontsize=8)
            ax.set(title=point+' / '+('depth-rate error' if k==0 else 'y-momentum-rate error'),
                xlabel='Face midpoint x, y (m)',ylabel='m/s' if k==0 else 'm²/s²');ax.axhline(0,color='k',lw=.5);ax.grid(axis='y',alpha=.2)
            if row==0 and col==0:ax.legend(fontsize=7)
    fig.suptitle('Per-face error ledger: an exact interior normal velocity does not repair the wall trace')
    fig.savefig(OUT/'FACE_LEDGER.png',dpi=160);plt.close(fig)

    s=ResultsStore(ROOT/'validation/_trace179/run/baseline/cpu/run');polys=[np.asarray(p.exterior.coords) for p in s.read(0).polygons]
    fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained');colors={'baseline_cpu':'#c76d28','normal_cpu':'#2368ab'}
    point=int(ids[2])
    for name,color in colors.items():
        d=datasets[name];axs[0,0].plot(d['t']+200,d['h'][:,point]+d['z'][:,point],color=color,label=name)
        axs[0,1].plot(d['t']+200,d['uv'][:,1,point],color=color,label=name)
    axs[0,0].plot(native['t']+200,native['h'][:,point]+native['z'][:,point],'k:',label='FLOW independent trajectory')
    axs[0,1].plot(native['t']+200,native['uv'][:,1,point],'k:',label='FLOW independent trajectory')
    for ax,title,ylabel in [(axs[0,0],'First cell above leading tip','Water elevation (m)'),(axs[0,1],'Same raw cell; zero added viscosity','Transverse velocity (m/s)')]:
        ax.set(title=title,xlabel='Equivalent original time (s)',ylabel=ylabel);ax.grid(alpha=.2);ax.legend(fontsize=8)
    for ax,key,title in [(axs[1,0],'variance','Change in water-surface temporal std. (mm)'),(axs[1,1],'depth','Candidate minus baseline depth at 220 s (mm)')]:
        b=datasets['baseline_cpu']['h'];a=datasets['normal_cpu']['h']
        val=(np.std(a,axis=0)-np.std(b,axis=0))*1000 if key=='variance' else (a[-1]-b[-1])*1000
        lim=float(abs(val).max());pc=PolyCollection(polys,array=val,cmap='RdBu_r',clim=(-lim,lim),linewidth=.1,edgecolor='gray')
        ax.add_collection(pc);ax.set(xlim=(4,20),ylim=(0,8),aspect='equal',xlabel='x (m)',ylabel='y (m)',title=title)
        ax.fill([8,10,10,8],[0,0,3,3],color='.7');fig.colorbar(pc,ax=ax,shrink=.8)
    fig.suptitle('Same BRAIDS starting state, 20 s continuation — mixed coupled benefit')
    fig.savefig(OUT/'SHORT_CONTINUATION.png',dpi=160);plt.close(fig)

    fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained');regions=('domain','upstream','groin','wake','outlet');x=np.arange(len(regions))
    for i,label in enumerate(('baseline_cpu','normal_cpu','baseline_half_cpu','normal_half_cpu')):
        axs[0].bar(x+(i-1.5)*.19,[histories[label]['regions'][r]['linearly_detrended_eta_rms_mm'] for r in regions],.19,label=label)
    axs[0].set_xticks(x,regions);axs[0].set(ylabel='Linearly detrended elevation RMS (mm)',title='Mixed outcome persists at half CFL');axs[0].legend(fontsize=7)
    for label,color in colors.items():
        v=label.removesuffix('_cpu');a=datasets[label]['h'];b=datasets[v+'_half_cpu']['h']
        axs[1].plot(datasets[label]['t']+200,abs(a-b).max(axis=1)*1000,color=color,label=v+' maximum')
        axs[1].plot(datasets[label]['t']+200,np.sqrt(np.mean((a-b)**2,axis=1))*1000,color=color,ls=':',label=v+' RMS')
    axs[1].set(xlabel='Equivalent original time (s)',ylabel='Depth difference (mm)',title='CFL 1 versus 0.5, same model');axs[1].legend(fontsize=8)
    for ax in axs:ax.grid(axis='y',alpha=.2)
    fig.savefig(OUT/'SENSITIVITY.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(known=known,temporal=temporal,backend=parity,change=changes,same_state_energy=energies),indent=2))


if __name__=='__main__':main()
