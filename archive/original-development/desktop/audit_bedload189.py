"""Sweep physical forcing and coefficient mappings on the delivered CPU/CUDA code."""
import json,os,sys
from copy import deepcopy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUNTIME=ROOT/'validation/_wall188/desktop_candidate'
OUT=ROOT/'validation/_bend189/formulas';sys.path[:0]=[str(RUNTIME),str(ROOT),str(ROOT/'desktop')]
os.environ['OMP_NUM_THREADS']='4';os.environ['QT_QPA_PLATFORM']='offscreen'
import numpy as np
from bedload_oracle189 import formula,field
from stage_secondary185 import save,sha


def main():
    from braids_desktop.bedload_catalog import CATALOG
    from braids_core.river_launcher import verify
    from braids_core.sediment.portable_river_runtime import install
    from braids_core.sediment.river_capacity import ResolvedRiverCapacity
    from braids_core.sediment.bed_columns_v101 import host
    verify();install(json.loads((RUNTIME/'PRECOMPILED.json').read_text()),RUNTIME)
    rows=[];curves={}
    for method,entry in CATALOG.items():
        configs=[deepcopy(entry['parameters'])]
        for key,value in entry['parameters'].items():
            changed=deepcopy(entry['parameters']);changed[key]=value*.73;configs.append(changed)
        for ci,params in enumerate(configs):
            for di,(d,rs,rw) in enumerate(((.0002,2650.,1000.),(.005,2800.,1025.),(.02,1600.,998.))):
                n=257;h=np.geomspace(.002,4.,n);c=np.linspace(12.,70.,n)
                u=np.r_[0.,np.geomspace(.00001,5.,n-1)];d90=np.full(n,1.5*d);hide=np.linspace(.3,1.7,n)
                shear=u*(.04+.1*np.arange(n)/n)
                expected=formula(method,params,u,h,c,d,d90,hide,shear,rs,rw)
                results=[]
                for backend in ('cpu','gpu'):
                    op=ResolvedRiverCapacity(backend,diameter=d,chezy=30.,capacity=dict(method=method,parameters=params),
                        solid_density=rs,water_density=rw,folder=OUT/'operators'/f'{method}_{ci}_{di}_{backend}')
                    xp=op.xp
                    def probe(speed):
                        out=xp.zeros((3,n));ratio=np.divide(speed,u,out=np.ones_like(u),where=u>0)
                        op.engine.launch('scalar_forcing_probe',n,n,d,xp.asarray(speed),xp.asarray(c),xp.asarray(h),
                            xp.asarray(d90),xp.asarray(hide),xp.asarray(shear*ratio),op.wire,out)
                        return host(out)
                    got=probe(u);np.testing.assert_allclose(got[0],expected,rtol=2e-11,atol=2e-16)
                    eps=np.maximum(u*1e-6,1e-8);valid=(u>eps)&(expected>1e-12)
                    fd=(probe(u+eps)[0]-probe(np.maximum(0,u-eps))[0])/(2*eps)
                    np.testing.assert_allclose(got[2,valid],fd[valid],rtol=3e-6,atol=2e-10)
                    angle=np.linspace(-np.pi,np.pi,n);q=np.ascontiguousarray([h,h*u*np.cos(angle),h*u*np.sin(angle)])
                    z0=h/(np.e*np.expm1(.41*c/np.sqrt(9.81)))
                    cs=np.maximum(1e-6,np.sqrt(9.81)/.41*np.log1p(np.maximum(.01,h)/np.maximum(1e-8,np.e*z0)))
                    fac=.41*cs/(np.sqrt(9.81)*np.log1p(h*float(np.float32(.368))/z0))
                    cq=q.copy();cq[1:]*=fac;hi=h*np.linspace(-.03,.03,n)
                    op.z0=xp.asarray(z0);op.d90=xp.asarray(d90);op.hiding=xp.asarray(hide);op.hydraulic_chezy=xp.asarray(c)
                    g=host(op.evaluate(xp.asarray(q),xp.asarray(hi),capacity_q=xp.asarray(cq),chezy=xp.asarray(cs)))
                    ref=field(op,q,hi,cq,cs);np.testing.assert_allclose(g,ref,rtol=2e-11,atol=1e-12)
                    results.append(g)
                    rows.append(dict(method=method,backend=backend,parameters=params,diameter_m=d,solid_density=rs,water_density=rw,
                        points=n,scalar_max_m2_s=float(abs(got[0]-expected).max()),field_max=float(abs(g-ref).max()),
                        derivative_max=float(abs(got[2,valid]-fd[valid]).max()),finite=bool(np.isfinite(g).all())))
                np.testing.assert_allclose(*results,rtol=2e-10,atol=1e-11)
        speed=np.linspace(0,4,401);curves[method]=dict(speed=speed.tolist(),bedload=formula(method,entry['parameters'],speed,
            np.full_like(speed,.3),np.full_like(speed,30.),.005,np.full_like(speed,.0075),np.ones_like(speed),speed*np.sqrt(9.81)/30).tolist())
    save(OUT/'AUDIT.json',dict(runtime=sha(RUNTIME/'RUNTIME.json'),rows=rows,curves=curves,
        complete=True,exposed_models=list(CATALOG),scope='Four public single-fraction bedload laws; not inactive research formulas'))
    print('PASS',len(rows),'backend/configuration batches,',sum(r['points'] for r in rows),'scalar and field inputs')


if __name__=='__main__':main()
