"""Compact fixed-axis support/coverage gather. No operating policy here."""
import numpy as np
_kernel=None

def sample(rotor,q,z,grad):
    global _kernel
    xp=rotor.xp
    if _kernel is None:
        _kernel=xp.RawKernel(r'''
        __device__ double minprop(double a,double b){return isnan(a)?a:(isnan(b)?b:fmin(a,b));}
        __device__ double maxprop(double a,double b){return isnan(a)?a:(isnan(b)?b:fmax(a,b));}
        extern "C" __global__ void gather(int n,int ns,int np,double dry,double nx,double ny,
          const long long* ids,const long long* plane,const double* delta,const double* weights,
          const double* q,const double* z,const double* grad,double* h,double* w,double* un,double* ut,double* stats){
          if(blockIdx.x || threadIdx.x)return;
          double lo=1./0.,hi=-1./0.,is_dry=0.,su=0.,sh=0.,st=0.,minu=1./0.;
          for(int k=0;k<ns;k++){
            int c=ids[k];double hh=q[c],den=maxprop(hh,dry),u=q[n+c]/den,v=q[2*n+c]/den;
            h[k]=hh;w[k]=weights[c];un[k]=u*nx+v*ny;ut[k]=-u*ny+v*nx;
            is_dry=fmax(is_dry,hh<=dry?1.:0.);minu=minprop(minu,un[k]);
            su+=w[k]*un[k];sh+=w[k]*hh;st+=w[k]*ut[k];
          }
          for(int k=0;k<np;k++){
            int c=plane[k];double eta=q[c]+z[c];
            double surface=(eta+grad[4*n+c]*delta[k])+grad[5*n+c]*delta[np+k];
            double bed=(z[c]+grad[6*n+c]*delta[k])+grad[7*n+c]*delta[np+k];
            lo=minprop(lo,minprop(surface,eta));hi=maxprop(hi,maxprop(bed,z[c]));
          }
          stats[0]=lo;stats[1]=hi;stats[2]=is_dry;stats[3]=su;stats[4]=sh;stats[5]=st;stats[6]=minu;
        }''','gather',options=('--fmad=false',))
    # Ownership-local scratch: result lifetime is one synchronous sample call.
    buffers=getattr(rotor,'_sample_pack',None)
    if buffers is None:
        buffers=(xp.asarray(rotor.ids,dtype=xp.int64),xp.asarray(rotor.plane,dtype=xp.int64),
                 xp.empty((4,len(rotor.ids)),dtype=xp.float64),xp.empty(7,dtype=xp.float64),
                 xp.ascontiguousarray(rotor.delta))
        rotor._sample_pack=buffers
    ids,plane,values,stats,delta=buffers
    nx,ny=rotor.normal
    _kernel((1,),(1,),(np.int32(q.shape[1]),np.int32(len(ids)),np.int32(len(plane)),
        np.float64(rotor.dry),np.float64(nx),np.float64(ny),ids,plane,delta,rotor.normalized,
        q,z,grad,*values,stats))
    return (*values,stats.get())
