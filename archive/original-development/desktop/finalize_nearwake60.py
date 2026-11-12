"""Full-window near/far comparison with independently verified zero-load controls."""
from analyze_nearwake60 import *
from matplotlib.collections import PolyCollection


def positive_halfwidth(y,deficits):
    ids=np.argsort(y);y=np.asarray(y)[ids];d=np.asarray(deficits)[ids]
    keep=y>=0;y=y[keep];d=d[keep]
    half=np.max(d)*.5
    crossing=np.flatnonzero(d<=half)
    if not len(crossing) or crossing[0]==0:return None
    i=int(crossing[0]);return float(y[i-1]+(half-d[i-1])*(y[i]-y[i-1])/(d[i]-d[i-1]))


def saved_case(output,name,start,end):
    # Use the same loader for historical and new fields, without modifying history.
    store=ResultsStore(Path(output));ids=np.flatnonzero((store.times>=start)&(store.times<=end))
    assert len(ids)>=3 and abs(store.times[ids[-1]]-end)<1e-8
    states=[]
    for idx in ids:
        f=store.read(int(idx));states.append(np.array([f.q[0],f.q[1]/f.q[0],f.q[2]/f.q[0]]))
    xy=np.array([p.centroid.coords[0] for p in f.polygons]);q=np.mean(states,axis=0)
    return dict(name=name,xy=xy,tri=Delaunay(xy),state=q,
        speed=np.mean([np.hypot(s[1],s[2]) for s in states],axis=0),
        states=states,times=store.times[ids],polygons=f.polygons,area=f.mesh['area'],output=str(output))


def finish():
    assert (REPORT/'CONTROL_CHECK.json').is_file()
    oldhome=Path.home()/'Desktop/BRAIDS_Turbine_Validation59'
    cases={};controls={}
    for name,sub in [('nash_gpu','02_Nash_Equivalent_Load_GPU'),('zero_gpu','04_Matched_Zero_Thrust_GPU')]:
        row=read(oldhome/sub/'LAST_VALIDATION_RUN.json')
        c=saved_case(row['output'],name,1500.,2000.)
        if name=='nash_gpu':cases['D/2']=c
        else:controls['D/2']=c
    for n in (4,8):
        control_name='zero8_corrected' if n==8 else 'zero4_gpu'
        a=read(REPORT/f'wake{n}_gpu'/'RECEIPT.json');b=read(REPORT/control_name/'RECEIPT.json')
        assert a['status']==b['status']=='complete'
        cases[f'D/{n}']=saved_case(a['output'],f'wake{n}_gpu',1500.,2000.)
        controls[f'D/{n}']=saved_case(b['output'],control_name,400.,500.)
        np.testing.assert_array_equal(cases[f'D/{n}']['xy'],controls[f'D/{n}']['xy'])
    references=list(csv.DictReader((ROOT/'validation/hydrokinetic_turbines/measurements/nash_2015_figure6b.csv').open()))
    refs={k:np.array([[float(r['x_over_d']),float(r['velocity_deficit'])] for r in references if r['series']==k]) for k in ['published_2d_model','experiment_depth_average']}
    result=dict(wake_runtime_sha256=read(REPORT/'LOCKED_SPEC.json')['runtime_sha256'],
        corrected_control_runtime_sha256=read(REPORT/'ARITHMETIC_STAGE.json')['runtime_sha256'],physics_changed=False,
        numerical_correction='Resolve unchanged smooth force and retain transverse wake resolution with explicit mesh-only breaklines.',
        control_check=read(REPORT/'CONTROL_CHECK.json'),backend=read(REPORT/'CPU_GPU.json'),cases={})
    colors={'D/2':'#7e8c99','D/4':'#d98926','D/8':'#156fa8'};rows=[]
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
    xx=np.linspace(2.5,20,351)
    for key,c in cases.items():
        zero=controls[key];pred=deficit(c,zero,xx,0.)
        for ax in axes:ax.plot(xx,100*pred,color=colors[key],lw=2,label=f'BRAIDS {key}')
        r=dict(cells=len(c['xy']),output=c['output'],averaging_times_s=c['times'].tolist(),comparison={})
        r['maximum_normalization_sensitivity_pp']=float(100*np.max(abs(pred-deficit(c,controls['D/2'],xx,0.))))
        r['normalization']='Own refined-mesh 400–500s zero-thrust average; sensitivity to historical 1500–2000s control reported' if key!='D/2' else 'Original matched 1500–2000s zero-thrust average'
        for ref,data in refs.items():
            predref=deficit(c,zero,data[:,0],0.);r['comparison'][ref]={}
            for part,mask in [('near_2p5_to4D',data[:,0]<=4),('far_ge4D',data[:,0]>=4),('all',data[:,0]>=0)]:
                stats=metric(predref[mask],data[mask,1]);stats['relative_L2']=float(np.linalg.norm(predref[mask]-data[mask,1])/np.linalg.norm(data[mask,1]));r['comparison'][ref][part]=stats
            rows.extend(dict(resolution=key,reference=ref,x_D=float(x),prediction=float(p),target=float(t)) for x,p,t in zip(data[:,0],predref,data[:,1]))
        mid=len(c['states'])//2
        first=dict(c,speed=np.mean([np.hypot(s[1],s[2]) for s in c['states'][:mid+1]],axis=0))
        last=dict(c,speed=np.mean([np.hypot(s[1],s[2]) for s in c['states'][mid+1:]],axis=0))
        r['maximum_centreline_halfwindow_change_pp']=float(100*np.max(abs(deficit(first,zero,xx,0.)-deficit(last,zero,xx,0.))))
        output=Path(c['output']);diags=read(output/'DIAGNOSTICS.json');status=read(output/'STATUS.json')
        r['maximum_water_residual_m3']=max(abs(d['water_residual_m3']) for d in diags)
        r['outer_rejections']=len(status.get('rejected_trials',[]));r['hydraulic']=status['hydraulic']
        # Raw exported values are inputs/diagnostics, not independent Ct/Cp validation.
        table=list(csv.DictReader((output/'TURBINE_PERFORMANCE.csv').open()))
        if table:
            table=[d for d in table if 1500<=float(d['time_s'])<=2000]
            r['mean_thrust_N']=float(np.mean([float(d['thrust_N']) for d in table]))
            r['mean_reconstructed_approach_m_s']=float(np.mean([float(d['upstream_velocity_m_s']) for d in table]))
            r['mean_requested_Ct']=float(np.mean([float(d['requested_Ct']) for d in table]))
            r['mean_useful_power_W']=float(np.mean([float(d['useful_power_W']) for d in table]))
            r['inactive_turbine_samples']=sum(d['active'].lower()!='true' for d in table)
            r['minimum_surface_clearance_m']=min(float(d['minimum_coverage_margin_m']) for d in table)
            assert r['inactive_turbine_samples']==0
            r['Ct_against_nominal_inlet']=r['mean_thrust_N']/(.5*1000*(np.pi*10**2/4)*.2**2)
            r['Ct_against_reconstructed_approach']=r['mean_thrust_N']/(.5*1000*(np.pi*10**2/4)*r['mean_reconstructed_approach_m_s']**2)
        result['cases'][key]=r
    for key,label,col in [('published_2d_model','Nash published 2D','#222222'),('experiment_depth_average','Experiment: published depth mean','#329256')]:
        d=refs[key]
        for ax in axes:ax.plot(d[:,0],100*d[:,1],'--',color=col,label=label)
    axes[0].set(xlim=(2.5,4.1),ylim=(10,25),title='Near wake: 2.5–4D')
    axes[1].set(xlim=(4,20),ylim=(0,20),title='Far wake: ≥4D (same frozen interval)')
    for ax in axes:ax.set(xlabel='Downstream distance / D',ylabel='Velocity deficit (%)');ax.grid(alpha=.2)
    axes[1].legend(fontsize=8);fig.suptitle('Same Ct, viscosity and solver; matched zero-thrust normalization; 1500–2000 s wake averages')
    fig.savefig(REPORT/'NEAR_FAR_COMPARISON.png',dpi=170);plt.close(fig)

    lateral=list(csv.DictReader((ROOT/'validation/hydrokinetic_turbines/measurements/nash_2015_figure7.csv').open()))
    fig,axes=plt.subplots(1,4,figsize=(14,4.8),layout='constrained')
    for ax,x in zip(axes,[2.,4.,10.,20.]):
        ys=np.linspace(-3,3,241)
        for key,c in cases.items():
            ax.plot(100*deficit(c,controls[key],x,ys),ys,color=colors[key],label=f'BRAIDS {key}',lw=1.6)
            # Actual CV means at the nearest axial section, not a spline.
            xc=c['xy'][:,0];near=abs(xc-(500+10*x));minimum=np.min(near)
            ids=np.flatnonzero((near<=minimum+1e-8)&(abs(c['xy'][:,1]-295)<15))
            y=(c['xy'][ids,1]-295)/10;xp=(xc[ids]-500)/10
            d=1-c['speed'][ids]/sample(controls[key],xp,y)
            ax.scatter(100*d,y,s=8,color=colors[key],zorder=4)
        data=[r for r in lateral if float(r['x_over_d'])==x]
        yy=np.array([float(r['y_over_d']) for r in data]);dd=np.array([float(r['velocity_deficit']) for r in data])
        ax.plot(100*dd,yy,'k--',label='Nash published 2D')
        ax.set(xlabel='Velocity deficit (%)',ylabel='Transverse position / D',title=f'{x:g}D',ylim=(-3,3));ax.grid(alpha=.2)
        for key,c in cases.items():
            pp=deficit(c,controls[key],x,yy)
            result['cases'][key].setdefault('lateral',{})[str(x)]=dict(**metric(pp,dd),
                prediction_peak_percent=float(100*np.max(pp)),reference_peak_percent=float(100*np.max(dd)),
                prediction_halfwidth_D=positive_halfwidth(ys,deficit(c,controls[key],x,ys)),
                reference_halfwidth_D=positive_halfwidth(yy,dd),
                prediction_deficit_integral_D=float(np.trapezoid(pp,yy)),reference_deficit_integral_D=float(np.trapezoid(dd,yy)))
    axes[0].legend(fontsize=7);fig.suptitle('Raw cell values (dots) and linear interpolation (lines): no clipping, spline or color smoothing')
    fig.savefig(REPORT/'LATERAL_RAW_CELLS.png',dpi=170);plt.close(fig)

    map_max=0.
    for key,c in cases.items():
        mask=(c['xy'][:,0]>480)&(c['xy'][:,0]<710)&(abs(c['xy'][:,1]-295)<30)
        x=(c['xy'][mask,0]-500)/10;y=(c['xy'][mask,1]-295)/10
        map_max=max(map_max,float(np.max(100*(1-c['speed'][mask]/sample(controls[key],x,y)))))
    map_limit=float(5*np.ceil(map_max/5));result['map_maximum_deficit_percent']=map_max
    fig,axes=plt.subplots(3,1,figsize=(12,8),layout='constrained')
    for ax,(key,c) in zip(axes,cases.items()):
        mask=(c['xy'][:,0]>480)&(c['xy'][:,0]<710)&(abs(c['xy'][:,1]-295)<30)
        ids=np.flatnonzero(mask);polys=[(np.array(c['polygons'][i].exterior.coords)-[500,295])/10 for i in ids]
        x=(c['xy'][ids,0]-500)/10;y=(c['xy'][ids,1]-295)/10
        data=100*(1-c['speed'][ids]/sample(controls[key],x,y))
        artist=PolyCollection(polys,array=data,cmap='viridis',clim=(0,map_limit),edgecolors='none')
        ax.add_collection(artist);ax.plot([0,0],[-.5,.5],color='red',lw=2)
        ax.set(xlim=(-2,21),ylim=(-3,3),aspect='equal',xlabel='x/D',ylabel='y/D',title=f'{key}: unsmoothed control-volume deficit')
        fig.colorbar(artist,ax=ax,label='Deficit (%)')
    fig.savefig(REPORT/'WAKE_MAPS.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    positions=np.array([2.5,4.,10.,20.]);result['temporal']={}
    for key,c in cases.items():
        store=ResultsStore(Path(c['output']));history=[]
        for idx,t in enumerate(store.times):
            f=store.read(idx);state=dict(c,speed=np.hypot(f.q[1],f.q[2])/f.q[0])
            history.append(100*deficit(state,controls[key],positions,0.))
        history=np.array(history);result['temporal'][key]=dict(times_s=store.times.tolist(),x_D=positions.tolist(),deficit_percent=history.tolist())
        for j,ax in enumerate(axes.flat):ax.plot(store.times,history[:,j],color=colors[key],label=f'BRAIDS {key}')
    for j,ax in enumerate(axes.flat):
        for ref,label,col in [('published_2d_model','Published 2D','#222222'),('experiment_depth_average','Experimental depth mean','#329256')]:
            data=refs[ref];target=float(np.interp(positions[j],data[:,0],100*data[:,1]))
            ax.axhline(target,ls='--',lw=1,color=col,label=label)
        ax.axvspan(1500,2000,color='gray',alpha=.1);ax.grid(alpha=.2)
        ax.set(xlabel='Time (s)',ylabel='Deficit (%)',title=f'{positions[j]:g}D',xlim=(0,2000),ylim=(0,None))
    axes[1,1].legend(fontsize=7);fig.suptitle('Development of the wake; final comparison window shaded')
    fig.savefig(REPORT/'TEMPORAL_DEFICIT.png',dpi=170);plt.close(fig)
    result['aims']={}
    for ref in refs:
        r=result['cases']['D/8']['comparison'][ref]
        result['aims'][ref]=dict(near_RMS_le3pp=r['near_2p5_to4D']['RMS_pp']<=3,far_RMS_le3pp=r['far_ge4D']['RMS_pp']<=3,
            original_far_max_le6pp=r['far_ge4D']['max_pp']<=6,original_far_relative_L2_le25pct=r['far_ge4D']['relative_L2']<=.25)
    save(REPORT/'METRICS.json',result)
    with (REPORT/'COMPARISON.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(result,indent=2))


if __name__=='__main__':finish()
