// Sun (2015) Eq.1d + explicitly reconstructed van Rijn (1984) equilibrium dunes.
// Shared CPU/CUDA arithmetic. No FLOW form-drag multiplier on JC capacity.
#ifdef __CUDACC__
#define JC_ENTRY extern "C" __global__
#define JC_LOOP(N) for(int i=blockDim.x*blockIdx.x+threadIdx.x;i<N;i+=blockDim.x*gridDim.x)
#else
#include <cmath>
#include <omp.h>
#define JC_ENTRY extern "C" __declspec(dllexport)
#define JC_LOOP(N) _Pragma("omp parallel for schedule(static)") for(int i=0;i<N;++i)
extern "C" __declspec(dllexport) void mobile_threads(int n){omp_set_dynamic(0);omp_set_num_threads(n);}
#endif

// p: D50,D90,g,critical ustar^2. Inputs are validated before launch.
// out: grain C, grain ustar^2,T,height,length,grain ks,form ks,total ks,C,n,total ustar^2.
JC_ENTRY void jc_roughness(int n,int bedforms,const double*p,const double*q,double*out){
 JC_LOOP(n){
  double h=q[i],U=hypot(q[n+i],q[2*n+i])/h,kg=3.*p[1];
  double cg=18.*log10(12.*h/kg),ug2=p[2]*(U/cg)*(U/cg),T=ug2/p[3]-1.;
  double height=0.;
  if(bedforms && T>0. && T<25.)height=.11*h*pow(p[0]/h,.3)*(-expm1(-.5*T))*(25.-T);
  double length=7.3*h,kf=1.1*height*(-expm1(-25.*height/length)),ks=kg+kf;
  double C=18.*log10(12.*h/ks),manning=pow(h,1./6.)/C;
  out[i]=cg;out[n+i]=ug2;out[2*n+i]=T;out[3*n+i]=height;out[4*n+i]=length;
  out[5*n+i]=kg;out[6*n+i]=kf;out[7*n+i]=ks;out[8*n+i]=C;out[9*n+i]=manning;
  out[10*n+i]=p[2]*(U/C)*(U/C);
 }
}
