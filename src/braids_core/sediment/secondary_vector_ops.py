"""Stage-local GPU arithmetic; identical clocks, scalar fluxes and guards.

No data are cached across trials. FMA contraction is disabled to retain the
operation ordering of the separate float64 array expressions.
"""
import numpy as np
_kernels={}

def kernel(xp,name,inputs,outputs,code):
    if name not in _kernels:
        _kernels[name]=xp.ElementwiseKernel(inputs,outputs,code,'braids_secondary_'+name,
                                           options=('--fmad=false',))
    return _kernels[name]

def state(xp,q0,r0,r1,dt,f):
    if xp is np:return q0+dt*((f-.5*f*f)*r0+.5*f*f*r1)
    return kernel(xp,'state','float64 q,float64 a,float64 b,float64 dt,float64 c0,float64 c1',
                  'float64 out','double x=c0*a; double y=c1*b; out=q+dt*(x+y);')(q0,r0,r1,dt,f-.5*f*f,.5*f*f)

def blend(xp,a,b,f):
    if xp is np:return (1.-f)*a+f*b
    return kernel(xp,'blend','float64 a,float64 b,float64 left,float64 right','float64 out',
                  'out=left*a+right*b;')(a,b,1.-f,f)

def stage(xp,start,r0,r1,dt,f):
    if xp is np:return start+dt*((1.-f)*r0+f*r1)
    return kernel(xp,'stage','float64 q,float64 a,float64 b,float64 dt,float64 left,float64 right',
                  'float64 out','out=q+dt*(left*a+right*b);')(start,r0,r1,dt,1.-f,f)

def update(xp,work,work2,depth,production,decay,dt):
    if xp is np:
        transported=.5*(work+work2)
        return transported,(transported+dt*depth*production)/(1.+dt*decay)
    return kernel(xp,'update','float64 a,float64 b,float64 h,float64 p,float64 d,float64 dt',
                  'float64 transported,float64 out',
                  'transported=.5*(a+b); out=(transported+(dt*h)*p)/(1.+dt*d);')(work,work2,depth,production,decay,dt)


def safe(xp,q,threshold):
    if xp is np:
        wet=q[0]>threshold;result=q.copy()
        result[0]=xp.where(wet,q[0],.001001)
        result[1:]=xp.where(wet,q[1:],0.)
        return result,wet
    result=xp.empty_like(q);wet=xp.empty(q.shape[1],dtype=xp.bool_)
    kernel(xp,'safe','float64 h,float64 u,float64 v,float64 threshold',
           'float64 oh,float64 ou,float64 ov,bool wet',
           'wet=h>threshold; oh=wet?h:.001001; ou=wet?u:0.; ov=wet?v:0.;')(
               q[0],q[1],q[2],threshold,result[0],result[1],result[2],wet)
    return result,wet
