"""Execute the installed spatial operator on a saved warm state; no time advance."""
from nearwake60 import *
import numpy as np
from braids_core.sediment.portable_river_runtime import install as portable
portable(read(RUNTIME/'PRECOMPILED.json'),RUNTIME)
from braids_core import river_worker as public
from braids_core.sediment import adaptive_river_worker as core
from braids_core.sediment import river_turbine_flow as turbine
from braids_desktop.results import ResultsStore

def gather(face,g):
    out=np.zeros((3,len(g['area'])));o=g['owner'];j=g['neighbour'];inside=j>=0
    for k in (0,1,2):
        np.add.at(out[k],o,-face[k]*g['length']/g['area'][o])
        np.add.at(out[k],j[inside],face[k,inside]*g['length'][inside]/g['area'][j[inside]])
    return out

def run():
    record=read(PARENT/'nash_gpu/RECEIPT.json');output=Path(record['output']);context=read(output/'RUN_CONTEXT.json')
    prep=(output/context['input']).resolve();captured={};original=core.run
    def intercept(*args,**kwargs):captured.update(prepared=args[0],kwargs=kwargs);return None
    core.run=intercept
    try:public.run(prep,REPORT/'unused_replay_output')
    finally:core.run=original
    p,initial,t=core.load_prepared(captured['prepared']);kwargs=captured['kwargs']
    cache=core.cache_for(Path(captured['prepared'])/'cache',p,initial,t)
    make=core.factory('gpu',p,t,cache,REPORT/'replay_work',resolved_model=kwargs['resolved_model'],process_parameters=kwargs['process_parameters'])
    leaves=core.initial_leaves(p,initial,t,make.required_leaves);mesh=t.select(leaves);a=make(leaves,mesh)
    f=ResultsStore(output).read(-1);a.q[:]=a.xp.asarray(f.q);a.z[:]=a.xp.asarray(f.z)
    assert np.max(abs(f.z))<1e-12
    physics=a.physics;xp=a.xp;g={k:a.host(v).copy() for k,v in physics.g.items() if hasattr(v,'shape')}
    terms={};original_launch=turbine.FullStressEngine.launch
    def launch(engine,name,*args):
        result=original_launch(engine,name,*args)
        if name in ('hkt_full_strain_faces','hkt_balanced_pressure'):
            face=a.host(args[-1] if name=='hkt_full_strain_faces' else args[-2]).copy()
            terms[name]=-gather(np.vstack((np.zeros(physics.nf),face)),g)
            if name=='hkt_balanced_pressure':
                # Save exact trace/source fields passed into the correction.
                terms['trace']=a.host(args[7]).copy();terms['source']=a.host(args[9]).copy()
        return result
    turbine.FullStressEngine.launch=launch
    try:
        a.time=2000.;physics.time=2000.
        physics.begin(a.q,a.z,.001,2000.);m=a.midpoint;m.interval_base=a.q;m.minimum_theta=1.;m.frequency_peak=0.
        rate,water=m.rate(a.q,.001,2000.)
    finally:turbine.FullStressEngine.launch=original_launch
    flux=a.hydraulics.flux;local=a.host(flux.buffers['local']);fl=a.host(flux.buffers['fl']);fr=a.host(flux.buffers['fr'])
    hl,unl,utl,_,hr,unr,utr,_=local;nx,ny=g['normal']
    safe_l=np.maximum(hl,1e-30);safe_r=np.maximum(hr,1e-30)
    fn=.5*(unl*unl/safe_l+unr*unr/safe_r);ft=.5*(unl*utl/safe_l+unr*utr/safe_r)
    adv=np.array([.5*(unl+unr),fn*nx-ft*ny,fn*ny+ft*nx]);pressure=.25*9.81*(hl*hl+hr*hr)
    pf=np.array([np.zeros(physics.nf),pressure*nx,pressure*ny]);terms['central_advection']=gather(adv,g);terms['central_face_pressure']=gather(pf,g)
    # Actual hydraulic gather uses separate left/right pressure corrections.
    raw=np.zeros_like(f.q);o=g['owner'];j=g['neighbour'];inside=j>=0
    for k in range(3):
        np.add.at(raw[k],o,-fl[k]*g['length']/g['area'][o]);np.add.at(raw[k],j[inside],fr[k,inside]*g['length'][inside]/g['area'][j[inside]])
    terms['native_remainder']=raw-terms['central_advection']-terms['central_face_pressure']
    source=xp.zeros_like(a.q);a.turbine_source.apply(a.q,a.z,a.hkt_gradient,source);terms['actuator']=a.host(source)
    trace=terms.pop('trace');source=terms.pop('source')
    ids=np.flatnonzero(g['neighbour']>=0);aa=g['owner'][ids];bb=g['neighbour'][ids]
    da,db=g['distance'][:,ids];normal=g['normal'][:,ids]
    source_face=np.zeros((3,physics.nf));lowfr_face=np.zeros_like(source_face)
    source_delta=-da*db/(da+db)*np.sum((source[1:,aa]-source[1:,bb])*normal,axis=0)
    speed=np.sqrt(9.81*.5*(trace[0,ids]+trace[1,ids]));theta=np.minimum(1.,np.maximum(trace[4,ids],trace[5,ids])/speed)
    low_delta=-.5*(1-theta)*speed*(2*trace[0,ids]*trace[1,ids]/(trace[0,ids]+trace[1,ids]))*(trace[3,ids]-trace[2,ids])
    active=a.host(a.hkt_status)[ids]==0
    source_face[1:,ids]=normal*(source_delta*active);lowfr_face[1:,ids]=normal*(low_delta*active)
    source_pressure=-gather(source_face,g);lowfr=-gather(lowfr_face,g)
    total=a.host(rate).copy();terms['other']=total-sum(terms.values());terms['total']=total
    xy=np.array([c.centroid_xy for c in mesh.cells]);region=(xy[:,0]>480)&(xy[:,0]<550)&(abs(xy[:,1]-295)<30)
    metrics={}
    for name,r in terms.items():
        assert np.isfinite(r).all(),name
        metrics[name]=dict(integrated_momentum_rate=(r[1:]@g['area']).tolist(),maximum_local_rate=float(np.max(abs(r[1:,region]))))
    pieces=dict(source_pressure=source_pressure,low_froude=lowfr,pressure_interpolation=terms['hkt_balanced_pressure']-source_pressure-lowfr)
    for name,r in pieces.items():metrics[name]=dict(integrated_momentum_rate=(r[1:]@g['area']).tolist(),maximum_local_rate=float(np.max(abs(r[1:,region]))),subset_of='hkt_balanced_pressure')
    save(REPORT/'WARM_OPERATOR_LEDGER.json',dict(backend='gpu',time_s=2000.,source_runtime=sha(RUNTIME/'RUNTIME.json'),
        input_run=str(output),no_time_advance=True,terms=metrics,
        pressure_status={str(i):int(np.sum(a.host(a.hkt_status)==i)) for i in np.unique(a.host(a.hkt_status))},
        source=a.hkt_diagnose(),maximum_recomposition_error=float(np.max(abs(total-sum(v for k,v in terms.items() if k!='total'))))))
    np.savez_compressed(REPORT/'WARM_OPERATOR_LEDGER.npz',xy=xy,area=g['area'],q=f.q,**terms,**pieces)
    print(json.dumps(metrics,indent=2))
if __name__=='__main__':run()
