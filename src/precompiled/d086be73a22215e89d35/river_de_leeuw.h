// de Leeuw et al. (2020), doi:10.5194/esurf-8-485-2020 (CC BY 4.0).
// Eq.26a as PRINTED (denominator factor 3 => limit 1/3, not prose 0.3).
// Section 3.3 preferred concentration P, Eq.8 settling, section 4.2 below-a.
// BRAIDS local friction-slope, bounded skin and shoreline extensions are named.
#ifdef __CUDACC__
#define ENTRY extern "C" __global__
#define LOCAL __device__
#define LOOP(N) for(int i=blockDim.x*blockIdx.x+threadIdx.x;i<N;i+=blockDim.x*gridDim.x)
#else
#include <cmath>
#include <omp.h>
#define ENTRY extern "C" __declspec(dllexport)
#define LOCAL inline
#define LOOP(N) _Pragma("omp parallel for schedule(static)") for(int i=0;i<N;++i)
extern "C" __declspec(dllexport) void mobile_threads(int n){omp_set_dynamic(0);omp_set_num_threads(n);}
#endif
LOCAL double dl_profile(double p,const double*quad){
 // y=z/h, t=logit(y)-logit(.1). Below .1h: c=1.66 ca.
 double top=43./(1.+p),sum=0.;
 for(int j=0;j<48;++j){double t=top*quad[j],b=9.*exp(-t);
  sum+=quad[48+j]*exp(-p*t)*b/((1.+b)*(1.+b));}
 return fmin(1.066,fmax(.166,.166+top*sum));
}
LOCAL void dl_values(double h,double speed,double z0,double d,double d84,double r,double temp,
 const double*quad,double*v){
 double nu=4.e-5/(20.+temp),ws=r*9.81*d*d/(18.*nu+sqrt(.75*r*9.81*d*d*d));
 double u=fabs(speed),ut=u*.41/log1p(.36800000071525574*h/z0),fr=u/sqrt(9.81*h);
 // Eq.7 solves Hsk, NOT H: us=(U/8.1)^.75 * ut^.25 * (3D84/H)^.125.
 // Sf=ut^2/(gH) from actual local hydraulic resistance, not a signed DEM slope.
 double raw=ut>0.?exp(.75*log(u/8.1)+.25*log(ut)+.125*log(3.*d84/h)):0.;
 double us=fmin(ut,raw),es=0.;
 if(us>0. && fr>0.){
  double lm=1.5*log(us/ws)+log(fr);
  if(lm>log(.015)){
   double le=log(4.74e-4)+1.18*(lm+log1p(-.015*exp(-lm)));
   es=le>=0.?1./(3.+exp(-le)):exp(le)/(1.+3.*exp(le));
  }
 }
 double p=ut>0.?pow(ws/ut,.45):1.e100,j=dl_profile(p,quad);
 if(h<fmax(10.*d,3.*d84)){es=0.;j=1.;}
 v[0]=ws;v[1]=es;v[2]=j;v[3]=ut;v[4]=us;v[5]=fr;v[6]=p;v[7]=raw;
}
ENTRY void dl_probe(int n,double d,double d84,double r,double temp,const double*h,
 const double*speed,const double*z0,const double*quad,double*out){
 LOOP(n){double v[8];dl_values(h[i],speed[i],z0[i],d,d84,r,temp,quad,v);
  for(int k=0;k<8;++k)out[k*n+i]=v[k];}
}
ENTRY void dl_exchange(int n,double dt,double dry,double d,double d84,double rho,double r,double temp,
 const double*q,const double*z0,const double*mass,const double*stock,const double*quad,
 double*next,double*delta,double*diag){
 LOOP(n){double h=q[i];
  if(h<=dry){next[i]=0.;delta[i]=mass[i];diag[i]=diag[n+i]=diag[2*n+i]=0.;continue;}
  double v[8];dl_values(h,hypot(q[n+i],q[2*n+i])/h,z0[i],d,d84,r,temp,quad,v);
  double rate=v[0]/(h*v[2]),equilibrium=rho*v[1]*h*v[2];
  double change=(equilibrium-mass[i])*(-expm1(-dt*rate));
  change=fmax(-mass[i],fmin(change,fmax(stock[i],0.)*(1.-16.*2.22044604925031308085e-16)));
  next[i]=mass[i]+change;delta[i]=-change;
  diag[i]=v[0];diag[n+i]=rho*v[1]*v[2];diag[2*n+i]=rate;
 }
}
