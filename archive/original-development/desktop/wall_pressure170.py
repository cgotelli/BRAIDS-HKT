"""Isolated exact reflecting-wall pressure; augmented interior solver unchanged.

The scalar wall Riemann problem has zero normal star velocity. Rarefaction
depth follows its invariant; compression depth solves the shock jump relation.
No changes to Delft3D, production, reconstruction or sediment formulas.
"""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'desktop/runtime'),str(ROOT),str(ROOT/'desktop')]
METHOD='exact_reflecting_wall_pressure170'
CUDA_HELPER=r'''
__device__ double wall_pressure170(double h,double momentum,double g,double dry) {
    if(h<=dry) return 0.;
    const double c=sqrt(g*h), v=momentum/h/c;
    double ratio;
    if(v<=0.) {const double a=fmax(0.,1.+.5*v);ratio=a*a;}
    else {
        double lo=1.,hi=1.+sqrt(2.)*v;
        double r=1.+v;
        for(int k=0;k<32;++k) {
            const double s=sqrt(.5*(1.+1./r));
            const double f=(r-1.)*s-v;
            if(fabs(f)<=2.e-15*fmax(1.,v)) break;
            if(f>0.) hi=r;else lo=r;
            const double df=s-(r-1.)/(4.*s*r*r);
            const double trial=r-f/df;
            r=(trial>lo && trial<hi) ? trial : .5*(lo+hi);
        }
        ratio=r;
    }
    const double star=h*ratio;return .5*g*star*star;
}
'''
FORTRAN_HELPER=r'''
 real(c_double) function wall_pressure170(h,momentum,g,dry) bind(C)
  use iso_c_binding, only:c_double
  real(c_double),value::h,momentum,g,dry
  real(c_double)::c,v,ratio,a,lo,hi,r,s,ff,df,trial,star
  integer::k
  wall_pressure170=0.d0
  if(h<=dry)return
  c=sqrt(g*h);v=momentum/h/c
  if(v<=0.d0)then
   a=max(0.d0,1.d0+.5d0*v);ratio=a*a
  else
   lo=1.d0;hi=1.d0+sqrt(2.d0)*v;r=1.d0+v
   do k=1,32
    s=sqrt(.5d0*(1.d0+1.d0/r));ff=(r-1.d0)*s-v
    if(abs(ff)<=2.d-15*max(1.d0,v))exit
    if(ff>0.d0)then
     hi=r
    else
     lo=r
    endif
    df=s-(r-1.d0)/(4.d0*s*r*r);trial=r-ff/df
    if(trial>lo.and.trial<hi)then
     r=trial
    else
     r=.5d0*(lo+hi)
    endif
   enddo
   ratio=r
  endif
  star=h*ratio;wall_pressure170=.5d0*g*star*star
 end function
'''


def native_source(code):
    old='fl(1,f)=0;fl(2:3,f)=dot_product(fl(2:3,f),normal(:,f))*normal(:,f);fr(:,f)=fl(:,f)'
    assert code.count(old)==2
    new='fl(1,f)=0;fl(2:3,f)=wall_pressure170(l(1),dot_product(l(2:3),normal(:,f)),grav,dry_tolerance)*normal(:,f);fr(:,f)=fl(:,f)'
    code=code.replace(old,new)
    end=code.rindex('\nend module')
    return code[:end]+FORTRAN_HELPER+code[end:]


def cuda_source(code):
    old='const double pressure=resident_dot(l[1],l[2],nx,ny);'
    assert code.count(old)==1
    return CUDA_HELPER+'\n'+code.replace(old,'const double pressure=wall_pressure170(hL,local[nf+f],grav,dry);')


def build():
    from check_bedload161 import sha,save
    table=json.loads((ROOT/'desktop/runtime/PRECOMPILED.json').read_text())
    row=next(v for k,v in table.items() if 'shoreline_hydraulic_probe.build' in k and
        v['native']['file']=='precompiled/d440a06059c179ec44ed/hydraulics.dll')
    source=ROOT/'desktop/runtime/precompiled/d440a06059c179ec44ed/shoreline_probe.f90'
    assert sha(source)==row['identity']['source_sha256']
    code=native_source(source.read_text());digest=hashlib.sha256(code.encode()).hexdigest()
    out=ROOT/'validation/_interface170/build'/digest[:20];target=out/'hydraulics.dll'
    if not target.exists():
        out.mkdir(parents=True,exist_ok=True);generated=out/'wall_pressure.f90';generated.write_text(code)
        cmd=list(row['identity']['command']);old=next(v for v in cmd if v.endswith('shoreline_probe.f90'))
        cmd[cmd.index(old)]=str(generated);cmd[cmd.index('-o')+1]=str(target)
        result=subprocess.run(cmd,cwd=out,capture_output=True,text=True,timeout=180)
        (out/'BUILD.log').write_text(result.stdout+result.stderr)
        if result.returncode:raise RuntimeError(result.stderr)
        save(out/'IDENTITY.json',dict(parent=row,command=cmd,source_sha256=digest,
            library_sha256=sha(target),method=METHOD,production_modified=False,Delft3D_modified=False))
    identity=json.loads((out/'IDENTITY.json').read_text());assert sha(target)==identity['library_sha256']
    return target,identity,row['identity']['command'][0]


def install(backend):
    from validation.sediment.river_morphodynamics import shoreline_hydraulic_probe as probe
    if backend=='cpu':
        built=build();probe.build=lambda mode:built
    elif backend=='gpu':
        # The complete owner installs its wet-face/HR assembly after the bare
        # ResidentFlux constructor. Transform that FINAL source, preserving HR.
        from validation.sediment.river_morphodynamics import face_activity
        original=face_activity.cuda_transform
        face_activity.cuda_transform=lambda code:cuda_source(original(code))
    else:raise ValueError(backend)


def analytic_pressure(h,momentum,g=9.81,dry=1e-6):
    """Independent scalar reference using a bracketed SciPy root, not Newton."""
    from scipy.optimize import brentq
    if h<=dry:return 0.
    u=momentum/h
    if u<=0:star=max(0.,np.sqrt(g*h)+u/2)**2/g
    else:
        fun=lambda star:(star-h)*np.sqrt(.5*g*(1/star+1/h))-u
        star=brentq(fun,h,h+np.sqrt(2*h/g)*u,xtol=1e-15,rtol=1e-14)
    return .5*g*star**2


def qualify(out):
    import cupy as cp
    from validation.gpu_fwave.reconstruction import OPTIONS
    from check_interface170 import cpu_solver,gpu_solver,physical
    from check_bedload161 import save,sha
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    library,identity,compiler=build()
    dll_dir=os.add_dll_directory(str(Path(compiler).parent))
    lib=ctypes.CDLL(str(library))
    f=lib.wall_pressure170;f.argtypes=[ctypes.c_double]*4;f.restype=ctypes.c_double
    code=CUDA_HELPER+r'''
extern "C" __global__ void probe170(const double* h,const double* m,int n,double* out) {
 int i=blockDim.x*blockIdx.x+threadIdx.x;
 if(i<n)out[i]=wall_pressure170(h[i],m[i],9.81,1.e-6);
}
'''
    module=cp.RawModule(code=code,options=OPTIONS);kernel=module.get_function('probe170')
    rows=[(h,fr) for h in (0.,1e-7,1e-6,1e-5,.1,.333,.5,2.)
          for fr in (-3.,-2.,-1.5,-.5,-.01,0.,.01,.5,1.,3.,20.)]
    h=np.array([r[0] for r in rows]);fr=np.array([r[1] for r in rows]);m=h*fr*np.sqrt(9.81*h)
    native=np.array([f(a,b,9.81,1e-6) for a,b in zip(h,m)])
    device=cp.empty(len(h));kernel(((len(h)+255)//256,),(256,),(cp.asarray(h),cp.asarray(m),np.int32(len(h)),device))
    gpu=cp.asnumpy(device);exact=np.array([analytic_pressure(a,b) for a,b in zip(h,m)])
    scale=np.maximum(1.,exact)
    assert np.max(abs(native-exact)/scale)<2e-13
    assert np.max(abs(native-gpu)/scale)<2e-13
    wet=h>1e-6;x=np.ascontiguousarray(np.vstack([h[wet],m[wet],h[wet]*2,np.zeros(wet.sum()),h[wet],-m[wet],h[wet]*2,np.zeros(wet.sum())]))
    old=cpu_solver(x);cg=gpu_solver(x);np.testing.assert_allclose(old,cg,atol=1e-10,rtol=1e-12)
    old_pressure=np.array([(physical(x[:4,i])+old[:3,i])[1] for i in range(x.shape[1])])
    np.savez(out/'PRESSURE.npz',h=h,fr=fr,momentum=m,exact=exact,cpu=native,gpu=gpu,
             wet=wet,old_pressure=old_pressure,old_cpu=old,old_gpu=cg)
    save(out/'AUDIT.json',dict(count=len(h),native_relative_error=float(max(abs(native-exact)/scale)),
        backend_relative_error=float(max(abs(native-gpu)/scale)),old_minimum_pressure=float(old_pressure.min()),
        build=identity,script_sha256=sha(__file__),outputs={'PRESSURE.npz':sha(out/'PRESSURE.npz')}))
    return identity


if __name__=='__main__':qualify(ROOT/'validation/_interface170/pressure_v3')
