"""Independent dimensional equations for the four public bedload identities.

MPM: locked FLOW tranb2; Ashida: trab14; Wang: tran9t + shld;
Smart-Jaeggi: BASEHPC 4.0.2 equations 1.29-1.31. No production-law calls.
"""
import numpy as np


def formula(method,p,u,h,c,d,d90,hide,shear,rhos=2650.,rhow=1000.):
    u,h,c,d90,hide,shear=np.broadcast_arrays(u,h,c,d90,hide,shear)
    relative=(rhos-rhow)/rhow;theta=(u/c)**2/(relative*d)
    unit=np.sqrt(9.81*relative*d**3)
    if method=='capacity.mpm':
        cg=18*np.log10(12*np.maximum(h/d90,1.))
        mu=np.minimum(1.,(c/cg)**1.5)
        return p['ACal']*8*unit*np.maximum(mu*theta-.047*hide,0.)**1.5
    if method=='capacity.ashida_michiue_generalized':
        r=np.divide(hide*p['ThetaC'],theta,out=np.full_like(theta,np.inf),where=theta>0)
        valid=(r<1)&(u>=float(np.float32(1e-6)))&(c>=float(np.float32(1e-6)))
        result=np.zeros_like(theta)
        result[valid]=p['ACal']*unit*theta[valid]**p['PowerM']*(1-r[valid])**p['PowerP']*(1-np.sqrt(r[valid]))**p['PowerQ']
        return result
    if method=='capacity.wang_fredsoe':
        # Stock tran9t's dfr iteration has no returning branch for positive
        # physical inputs: cgrain uses h. Preserve its native real constants.
        ds=d*(relative*9.81/p['VicMol']**2)**float(np.float32(.33333333))
        tc=(.240/ds if ds<=4 else .140/ds**.64 if ds<=10 else
            .040/ds**.10 if ds<=20 else .013*ds**.29 if ds<=150 else .055)
        cg=6+2.5*np.log(h/(2.5*d90));theta=(u/cg)**2/(relative*d)
        excess=theta-tc;result=np.zeros_like(theta);valid=(excess>0)&(u>0)&(c>0)
        result[valid]=(p['ACal']*float(np.float32(4.8695))*d*shear[valid]*
            (1+(float(np.float32(.2668))/excess[valid])**4)**(-.25)*
            (1-float(np.float32(.7))*np.sqrt(tc/theta[valid])))
        return result
    if method=='capacity.smart_jaggi':
        j=np.divide(u*u,c*c*h,out=np.zeros_like(u),where=h>0)
        critical=np.divide(p['ThetaC']*relative*d,h,out=np.zeros_like(h),where=h>0)
        return p['ACal']/relative*j**.6*h*u*np.maximum(j-critical,0.)
    raise ValueError(method)


def field(op,q,hi,cq,c):
    from braids_core.sediment.bed_columns_v101 import host
    q,cq,c,hi=map(host,(q,cq,c,hi));h=q[0];u,v=q[1:]/h;speed=np.hypot(u,v)
    ue,ve=cq[1:]/cq[0];effective=np.hypot(ue,ve)
    z0=host(op.z0);shear=speed*.41/np.log1p(h*float(np.float32(.368))/z0)
    method=op.identity['method'];parameters=op.identity['parameters']
    cs=c;depth=np.maximum(.01,cq[0]);forcing=effective
    if method=='capacity.smart_jaggi':cs=host(op.hydraulic_chezy);depth=h;forcing=speed;ue,ve=u,v
    qb=formula(method,parameters,forcing,depth,cs,op.d,host(op.d90),host(op.hiding),shear,op.solid_density,op.water_density)
    theta=shear**2/(9.81*((op.solid_density-op.water_density)/op.water_density)*op.d)
    if method=='capacity.smart_jaggi':theta=speed**2/(cs*cs*((op.solid_density-op.water_density)/op.water_density)*op.d)
    alpha=np.divide(12.5*(1-np.sqrt(9.81)/(.4*c))*hi/h,speed,out=np.zeros_like(h),where=speed>0)
    bx,by=u+alpha*v,v-alpha*u;norm=np.hypot(bx,by);mag=qb*op.solid_density*op.bed_factor
    return np.array([np.divide(mag*ue,forcing,out=np.zeros_like(h),where=forcing>0),
        np.divide(mag*ve,forcing,out=np.zeros_like(h),where=forcing>0),theta,
        np.divide(mag*bx,norm,out=np.zeros_like(h),where=norm>0),
        np.divide(mag*by,norm,out=np.zeros_like(h),where=norm>0),alpha])
