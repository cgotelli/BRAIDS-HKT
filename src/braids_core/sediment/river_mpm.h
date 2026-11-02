// Shared OpenMP/CUDA MPM, default FLOW grain distributions and Egiazaroff.
#ifdef __CUDACC__
#define ENTRY extern "C" __global__
#define LOOP(N) for(int i=blockDim.x*blockIdx.x+threadIdx.x;i<N;i+=blockDim.x*gridDim.x)
#else
#include <cmath>
#include <omp.h>
#define ENTRY extern "C" __declspec(dllexport)
#define LOOP(N) _Pragma("omp parallel for schedule(static)") for(int i=0;i<N;++i)
extern "C" __declspec(dllexport) void mobile_threads(int n){omp_set_dynamic(0);omp_set_num_threads(n);}
#endif
ENTRY void mixture(int n,int ng,int hiding,const double*mass,const double*diam,
 double*factor,double*d90,double*mean){
 LOOP(n){double total=0.,dm=0.,low=1.e100,high=-1.e100;
  double logd[32],lo[32],hi[32];
  for(int k=0;k<ng;++k){total+=mass[k*n+i];dm+=mass[k*n+i]*diam[k];
   logd[k]=log(diam[k]);lo[k]=logd[k]+1.25*log(.75);hi[k]=logd[k]+1.25*log(1.5);
   low=fmin(low,lo[k]);high=fmax(high,hi[k]);}
  // A depleted column carries zero transport via the stock limiter. Retain
  // finite, neutral constitutive scratch without inventing surface material.
  if(total<=0.){d90[i]=1.5*diam[0];mean[i]=diam[0];for(int k=0;k<ng;++k)factor[k*n+i]=1.;continue;}
  mean[i]=dm/total;
  for(int k=0;k<ng;++k){double r=diam[k]/fmax(mean[i],1.e-4),v=1.27875360095283;
   factor[k*n+i]=hiding?pow(v/(v+log10(r)),2.):1.;}
  for(int it=0;it<60;++it){double mid=.5*(low+high),cdf=0.;
   for(int k=0;k<ng;++k){double f=0.;if(mid>=hi[k])f=1.;else if(mid>logd[k])f=.5+.5*(mid-logd[k])/(hi[k]-logd[k]);
    else if(mid>lo[k])f=.5*(mid-lo[k])/(logd[k]-lo[k]);cdf+=mass[k*n+i]/total*f;}
   if(cdf<.9)low=mid;else high=mid;
  }d90[i]=exp(.5*(low+high));
 }
}
ENTRY void mpm(int n,double d,const double*q,const double*cq,const double*c,
 const double*hi,const double*d90,const double*hide,double*out){
 LOOP(n){double h=q[i],u=q[n+i]/h,v=q[2*n+i]/h,ue=cq[n+i]/cq[i],ve=cq[2*n+i]/cq[i];
  double speed=hypot(ue,ve),theta=speed*speed/(c[i]*c[i]*1.65*d);
  double grain=18.*log10(12.*fmax(fmax(.01,cq[i])/d90[i],1.)),mu=fmin(1.,pow(c[i]/grain,1.5));
  double excess=fmax(0.,mu*theta-.047*hide[i]);
  double mag=2650.*8.*sqrt(9.81*1.65*d*excess)*d*excess;
  double x=speed>0.?mag*ue/speed:0.,y=speed>0.?mag*ve/speed:0.;
  double us=hypot(u,v),alpha=us>0.&&mag>0.?12.5*(1.-sqrt(9.81)/(.4*c[i]))*(hi[i]/h)/us:0.;
  // FLOW bed-direction rotation, using primary steering velocity and signed I.
  double bx=u+alpha*v,by=v-alpha*u;
  double bs=hypot(bx,by);
  out[i]=x;out[n+i]=y;out[2*n+i]=theta;
  out[3*n+i]=bs>0.?mag*bx/bs:0.;out[4*n+i]=bs>0.?mag*by/bs:0.;out[5*n+i]=alpha;
 }
}
