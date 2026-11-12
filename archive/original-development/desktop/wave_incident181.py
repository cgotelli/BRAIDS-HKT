"""Small-amplitude closed-wall reflection versus the analytic SWE answer."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'desktop/runtime'),str(ROOT),str(ROOT/'desktop')]
from stage_wall172 import save,sha
from check_jet173 import setup
from check_wall171 import document
from check_incident181 import setup_factory
H=.5;G=9.81;C=np.sqrt(G*H);AMP=.001;SIGMA=.75;XC=6.;RIGHT=12.;LEFT=2.;DX=.5

def gaussian(x):return AMP*np.exp(-.5*((np.asarray(x)-XC)/SIGMA)**2)

def exact(x,t):
    # The left mirror is exponentially small during this window; include it
    # and its second reflected image for the finite closed interval.
    inc=gaussian(x-C*t);right=gaussian(2*RIGHT-x-C*t)
    left=gaussian(2*LEFT-x-C*t);twice=gaussian(x+2*(RIGHT-LEFT)-C*t)
    return inc+right+left+twice,C/H*(inc-right-left+twice)

def main(backend,variant):
    from braids_core import river_worker
    from braids_desktop.results import ResultsStore
    if variant=='wall':setup_factory(setup,backend)
    else:setup()
    out=ROOT/'validation/_incident181/waves_v5'/variant/backend;out.mkdir(parents=True,exist_ok=False)
    document(out/'inputs',backend,0,'reflection');path=out/'inputs/project/braids_project.json'
    from braids_desktop import services
    d=services.default_document(out/'inputs/terrain.tif',[[2.,4.],[12.,4.],[12.,6.],[2.,6.]],
        'EPSG:2056',backend=backend,minimum_size_m=DX,levels=1,end_time_s=4.,
        name=f'Wall incident pulse {variant} {backend}')
    d['initial'].update(stage_m=H,velocity_m_s=[0.,0.])
    d['settings'].update(output_interval_s=.05,end_time_s=4.,checkpoint_interval_s=4.,dt_max_s=.05)
    d['settings']['controls'].update(cfl=1.,threads=4)
    d['settings']['controls']['chezy']=1000.;d['roughness_parameters']['chezy_m_half_s']=1000.
    # Default hydraulic roughness choice is explicit in saved document.
    from braids_core.river_model import resolve_model
    request=d['model']['request'];request['hydrodynamics']['roughness']='roughness.chezy'
    d['model']=resolve_model(request).document()
    save(path,d)
    original=river_worker.prepare_recipe
    def recipe(*args,**kw):
        previous=kw.get('on_fields')
        def fields(tree,a):
            if previous:previous(tree,a)
            xy=np.array([cell.centroid_xy for cell in tree.finest.cells]);nodes,weights=np.polynomial.legendre.leggauss(16)
            perturb=gaussian(xy[:,0,None]+DX/2*nodes);h=H+perturb;u=C/H*perturb
            a['q'][:]=np.vstack([h@weights/2,(h*u)@weights/2,np.zeros(len(xy))])
            np.savez(out/'INITIAL.npz',xy=xy,q=a['q'],z=a['rock']+a['thickness'].sum(0))
        kw['on_fields']=fields;return original(*args,**kw)
    river_worker.prepare_recipe=recipe;prepared=out/'prepared';river_worker.prepare(path,prepared)
    status=river_worker.run(prepared,out/'run');s=ResultsStore(out/'run');xy=np.load(out/'INITIAL.npz')['xy']
    nodes,weights=np.polynomial.legendre.leggauss(16);eta=[];uv=[];expected=[];velocity=[]
    for j,t in enumerate(s.times):
        frame=s.read(j);eta.append(frame.q[0]-H);uv.append(frame.q[1]/frame.q[0])
        e,v=exact(xy[:,0,None]+DX/2*nodes,t);expected.append(e@weights/2);velocity.append(v@weights/2)
    eta=np.array(eta);expected=np.array(expected);uv=np.array(uv);velocity=np.array(velocity)
    np.savez(out/'COMPARISON.npz',t=s.times,xy=xy,eta=eta,exact_eta=expected,velocity=uv,exact_velocity=velocity)
    save(out/'CHECK.json',dict(status=status,backend=backend,variant=variant,depth=H,amplitude=AMP,sigma=SIGMA,
        cell_size=DX,chezy=1000.,reflection='closed wall, theoretical elevation reflection +1',
        eta_error_rms_m=float(np.sqrt(np.mean((eta-expected)**2))),eta_error_max_m=float(abs(eta-expected).max()),
        velocity_error_rms=float(np.sqrt(np.mean((uv-velocity)**2))),minimum_depth=float(H+eta.min()),
        source_sha256=sha(__file__),candidate_sha256=sha(ROOT/'desktop/wallstate181.py'),production_changed=False))
    print('Completed closed-wall pulse',variant,backend,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('backend',choices=('cpu','gpu'));p.add_argument('variant',choices=('baseline','wall'))
    a=p.parse_args();main(a.backend,a.variant)
