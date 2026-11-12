"""Corner flux error attribution and bounded zero-viscosity history figures."""
import json
from pathlib import Path
import sys
import numpy as np
from scipy.integrate import quad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'desktop/runtime'),str(ROOT),str(ROOT/'desktop')]
from stage_wall172 import save,sha
from quadrature176 import BASE,gather,exact_face_table
from corner_exact174 import mean_rate,face_flux
from review_mixing174 import pair
from review_zero175 import load

OUT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_corner176'
POINTS=np.array([[7.75,2.75],[7.75,3.25],[8.25,3.25],[8.75,3.25],[8.25,3.75]])


def acceleration(rate,q):
    return (rate[1:]-q[1:]/q[0]*rate[0])/q[0]


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    a=dict(np.load(ROOT/'validation/_corner174/potential/gpu/RESIDUAL.npz'))
    ids=np.array([np.flatnonzero(np.all(a['xy']==p,axis=1))[0] for p in POINTS]);c=ids[2]
    exact=np.array([mean_rate(*(p-[8,3])) for p in POINTS]).T
    variants={'Current':a['rate'],
        'Two-point face integration':np.load(BASE/'gpu/potential.npz')['quadrature'],
        'Reflected-normal fit':np.load(BASE/'wall_fit/probe/gpu/potential.npz')['rate'],
        'Axis-wall stencils':np.load(BASE/'axis_stencil/probe/gpu/potential.npz')['rate']}
    rows={}
    for name,rate in variants.items():
        diff=rate[:,ids]-exact
        rows[name]=dict(mass_rate_rms_error_m_s=float(np.sqrt(np.mean(diff[0]**2))),
            momentum_rate_rms_error_m2_s2=float(np.sqrt(np.mean(diff[1:]**2))),
            first_cell_rate=rate[:,c].tolist(),first_cell_acceleration_m_s2=acceleration(rate[:,c],a['q'][:,c]).tolist(),
            five_cells_rate=rate[:,ids].T.tolist())
    quadrature=[]
    for n in (1,2,4,8,16,32,64,None):
        fl=exact_face_table(a,n);rate=gather(a,fl,fl)[:,c]
        quadrature.append(dict(nodes=n,first_cell_rate=rate.tolist(),
            first_cell_acceleration_m_s2=acceleration(rate,a['q'][:,c]).tolist()))
    # Independent polar volume integration of -(u.grad)u, not reuse of
    # the Cartesian face integral implementation.
    dx=.5;A=.5;h=.5
    polar=np.array([h*A*A/dx**2*sum(quad(lambda t:fun(t)*(dx/max(np.cos(t),np.sin(t)))**(1/3),
        lo,hi,epsabs=1e-12,epsrel=1e-12)[0] for lo,hi in ((0,np.pi/4),(np.pi/4,np.pi/2)))
        for fun in (np.cos,np.sin)])
    assert np.max(abs(polar-exact[1:,2]))<1e-12
    faces=[]
    for f in np.flatnonzero((a['owner']==c)|(a['neighbour']==c)):
        axis=int(np.argmax(abs(a['normal'][:,f])));sign=a['normal'][axis,f]
        actual=a['fl'][:,f] if a['owner'][f]==c else a['fr'][:,f]
        true=sign*face_flux(*(a['mid'][:,f]-[8,3]),axis)
        faces.append(dict(face=int(f),mid=a['mid'][:,f].tolist(),normal=a['normal'][:,f].tolist(),
            into_cell_sign=int(-1 if a['owner'][f]==c else 1),length=float(a['length'][f]),
            current_flux=actual.tolist(),exact_integrated_flux=true.tolist()))
    f=next(r['face'] for r in faces if r['mid']==[8.,3.25])
    maxbound=float(max(a['rec_vhi'][1,a['owner'][f]],a['rec_vhi'][1,c]))
    mean_face_u=float(face_flux(0,.25,0)[0]/.5)
    parity={}
    for case in ('potential','base40','base120','base300'):
        cpu=np.load(BASE/'cpu'/(case+'.npz'));gpu=np.load(BASE/'gpu'/(case+'.npz'))
        parity[case]={k:float(abs(cpu[k]-gpu[k]).max()) for k in ('quadrature','fl','fr')}
    wall_parity={}
    for case in ('potential','affine','base40'):
        cpu=np.load(BASE/'wall_fit/probe/cpu'/(case+'.npz'));gpu=np.load(BASE/'wall_fit/probe/gpu'/(case+'.npz'))
        wall_parity[case]={k:float(abs(cpu[k]-gpu[k]).max()) for k in ('rate','water')}
    b,n,xy,area,*_=pair(ROOT/'validation/_corner174/long')
    gpu=load(ROOT/'validation/_corner174/long/gpu/warmup')
    history={};series={}
    for name,d in [('BRAIDS CPU',b),('BRAIDS GPU',gpu),('FLOW Flood',n)]:
        eta=d['h']+d['z'];ke=.5*np.sum(d['h']*np.sum(d['uv']**2,axis=1)*area,axis=1)
        speed=np.sqrt(np.sum(d['uv']**2,axis=1));records=[]
        for lo in (100,200,300,400,500):
            sel=(d['t']>=lo)&(d['t']<=lo+100)
            records.append(dict(from_s=lo,to_s=lo+100,eta_std_rms_mm=float(np.sqrt(np.var(eta[sel],axis=0).mean())*1000),
                speed_max_m_s=float(speed[sel].max()),depth_min_m=float(d['h'][sel].min()),depth_max_m=float(d['h'][sel].max()),
                kinetic_energy_over_density_min_m5_s2=float(ke[sel].min()),kinetic_energy_over_density_max_m5_s2=float(ke[sel].max())))
        history[name]=records
        series[name]=(d['t'],ke,speed.max(axis=1),np.array([np.sqrt(np.var(eta[max(0,j-20):j+1],axis=0).mean())*1000 for j in range(len(d['t']))]))
    backends={k:float(abs(b[k]-gpu[k]).max()) for k in ('h','uv','z')}
    regular={}
    import regular_corner176 as rc
    rb=np.load(BASE/'regular_v2/cpu/RESIDUAL.npz');ra=np.load(BASE/'axis_stencil/probe/cpu/regular.npz')
    rexact=np.array([rc.rate(*(p-[8,3])) for p in POINTS]).T
    for name,values in [('Current',rb['rate']),('Axis-wall stencils',ra['rate'])]:
        error=values[:,ids]-rexact
        regular[name]=dict(mass_rate_rms_error_m_s=float(np.sqrt(np.mean(error[0]**2))),
            momentum_rate_rms_error_m2_s2=float(np.sqrt(np.mean(error[1:]**2))),rate=values[:,ids].T.tolist())
    ab=load(BASE/'axis_stencil/run300/cpu/run');ag=load(BASE/'axis_stencil/run300/gpu/run')
    coupled={};short,flow,*_=pair(ROOT/'validation/_corner174')
    for name,d in [('Current',short),('Axis-wall stencils',ab),('FLOW Flood',flow)]:
        eta=d['h']+d['z'];std=np.std(eta[-21:],axis=0);tip=(xy[:,0]>=7)&(xy[:,0]<11)
        coupled[name]=dict(eta_std_rms_mm=float(np.sqrt(np.mean(std**2))*1000),
            groin_eta_std_rms_mm=float(np.sqrt(np.mean(std[tip]**2))*1000),
            upstream_mean_eta_m=float(eta[-21:,xy[:,0]==4.25].mean()),
            tip_mean_u_m_s=float(d['uv'][-21:,0,c].mean()))
    coupled['backend_max_difference']={k:float(abs(ab[k]-ag[k]).max()) for k in ('h','uv','z')}
    for backend in ('cpu','gpu'):
        d=json.loads((BASE/f'axis_stencil/run300/{backend}/run/DIAGNOSTICS.json').read_text())
        coupled[backend+'_budget']=dict(water_residual_max_m3=max(abs(r['water_residual_m3']) for r in d),
            solid_residual_max_kg=max(abs(v) for r in d for v in r['class_residual_kg']),
            wall_integral_max=max(abs(v) for r in d for v in r['wall_integrals']))
    save(OUT/'DIAGNOSTICS.json',dict(corner=dict(points_xy=POINTS.tolist(),exact_rate=exact.T.tolist(),
        polar_volume_momentum_rate=polar.tolist(),variants=rows,exact_field_quadrature=quadrature,first_cell_faces=faces,
        exact_west_face_mean_u_m_s=mean_face_u,maximum_reconstruction_bound_u_m_s=maxbound),
        quadrature_backend=parity,wall_fit_backend=wall_parity,
        regular_corner=regular,regular_exact_rate=rexact.T.tolist(),axis_stencil_coupled=coupled,
        history_100s_windows=history,retained_600s_backend_max_difference=backends,
        physical_viscosity=0.,candidate_qualified=False,production_changed=False,
        runtime_sha256=sha(ROOT/'desktop/runtime/RUNTIME.json')))

    labels=['Current','Two-point\nintegration','Wall-normal\nfit','Axis-wall\nstencils','Exact field\nat face centre','Exact face\nintegrals']
    rates=[variants[k][:,c] for k in variants]+[np.array(quadrature[0]['first_cell_rate']),exact[:,2]]
    accs=np.array([acceleration(r,a['q'][:,c]) for r in rates])
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    for ax,values,title,unit in zip(axes,[np.array(rates)[:,0],accs[:,0],accs[:,1]],
        ['Instantaneous mass rate','Streamwise acceleration','Transverse acceleration'],['m/s','m/s²','m/s²']):
        ax.bar(np.arange(6),values,color=['#d95f02','#e6ab02','#7570b3','#e7298a','#66a61e','#1b9e77'])
        ax.axhline(0,color='k',lw=.7);ax.set_xticks(np.arange(6),labels,fontsize=8)
        ax.set(title=title,ylabel=unit);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Known-answer sharp corner: cell (8.25, 3.25) m — instantaneous residual, not steady flow')
    fig.savefig(OUT/'CORNER_RESIDUAL.png',dpi=160);plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
    for k,ax in enumerate(axes):
        s=np.array([r['into_cell_sign'] for r in faces])*2
        actual=np.array([r['current_flux'][k] for r in faces])*s
        truth=np.array([r['exact_integrated_flux'][k] for r in faces])*s
        x=np.arange(4);ax.bar(x-.18,actual,.36,label='Current');ax.bar(x+.18,truth,.36,label='Exact integrated')
        ax.set_xticks(x,['West','East','North','Wall']);ax.axhline(0,color='k',lw=.7);ax.grid(axis='y',alpha=.2)
        ax.set(title=['Mass balance','x momentum balance','y momentum balance'][k],ylabel=['m/s','m²/s²','m²/s²'][k]);ax.legend()
    fig.suptitle('Face-by-face contributions to the first corner cell; all four contributions sum to its rate')
    fig.savefig(OUT/'FACE_CONTRIBUTIONS.png',dpi=160);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    x=[r['nodes'] for r in quadrature[:-1]]
    for k in (0,1):axes[0].semilogx(x,[r['first_cell_acceleration_m_s2'][k] for r in quadrature[:-1]],'o-',label=['a_x','a_y'][k])
    axes[0].axhline(exact[1,2]/.5,color='k',ls='--',label='Exact volume mean')
    axes[0].set(xlabel='Gauss points per face, using exact field',ylabel='Acceleration (m/s²)',ylim=(0,1),title='Even exact point samples converge slowly at the singularity')
    axes[1].bar(['Largest allowed\ncell-based u','Exact incoming\nface-average u'],[maxbound,mean_face_u],color=['#d95f02','#1b9e77'])
    axes[1].set(ylabel='Velocity (m/s)',ylim=(0,None),title='Coarse reconstruction bounds\ncannot recover the incoming face mean')
    for ax in axes:ax.grid(axis='y',alpha=.2)
    axes[0].legend();fig.savefig(OUT/'CORNER_ACCURACY_LIMIT.png',dpi=160);plt.close(fig)

    fig,axes=plt.subplots(3,1,figsize=(12,10),layout='constrained')
    for name,(t,ke,speed,variance) in series.items():
        for ax,y in zip(axes,(variance,ke,speed)):ax.plot(t,y,label=name,lw=1)
    for ax,title,unit in zip(axes,['Trailing 20 s water-surface fluctuations','Integrated kinetic energy / density','Maximum speed'],
        ['RMS fluctuation (mm)','m⁵/s²','m/s']):
        ax.set(title=title,ylabel=unit,xlabel='Time (s)',xlim=(0,600),ylim=(0,None));ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Existing zero-viscosity histories: bounded over 600 s, but not steady or backend-qualified')
    fig.savefig(OUT/'ZERO_VISCOSITY_BOUNDS.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    for k,ax in enumerate(axes):
        ax.plot(range(5),rexact[k],'ko-',label='Exact integrated rate')
        ax.plot(range(5),rb['rate'][k,ids],'s-',label='Current')
        ax.plot(range(5),ra['rate'][k,ids],'d-',label='Axis-wall stencil')
        ax.set_xticks(range(5),[f'({p[0]:g}, {p[1]:g})' for p in POINTS],rotation=30,ha='right',fontsize=8)
        ax.axhline(0,color='k',lw=.6);ax.grid(alpha=.2)
        ax.set(title=['Mass','x momentum','y momentum'][k],ylabel=['m/s','m²/s²','m²/s²'][k],xlabel='Cell centre (m)')
    fig.suptitle('Bounded manufactured corner, epsilon=1 m; lower error is not a complete fix')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=3)
    fig.savefig(OUT/'BOUNDED_CORNER.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(16,4.5),layout='constrained')
    for name,d in [('Current',short),('Axis-wall stencils',ab),('FLOW Flood',flow)]:
        eta=d['h']+d['z'];var=[np.sqrt(np.var(eta[max(0,j-20):j+1],axis=0).mean())*1000 for j in range(len(d['t']))]
        axes[0].plot(d['t'],var,label=name)
        sel=xy[:,0]==8.25;axes[1].plot(xy[sel,1],d['uv'][-21:,0,sel].mean(0),'.-',label=name)
        sel=xy[:,0]==4.25;axes[2].plot(xy[sel,1],eta[-21:,sel].mean(0),'.-',label=name)
    axes[0].set(title='Zero-viscosity wake',xlabel='Time (s)',ylabel='20 s surface fluctuation RMS (mm)',ylim=(0,None))
    axes[1].set(title='Jet at x=8.25 m',xlabel='y (m)',ylabel='Mean u, 280–300 s (m/s)',ylim=(0,None))
    axes[2].set(title='Upstream x=4.25 m',xlabel='y (m)',ylabel='Mean surface, 280–300 s (m)')
    for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('Axis-wall geometry trial rejected as an oscillation fix')
    fig.savefig(OUT/'GROIN_CANDIDATE.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(regular=regular,coupled=coupled,backend=backends),indent=2))


if __name__=='__main__':main()
