// Isolated source-guided drag contrasts, not production physics.
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
ENTRY void front_drag(int n,double dt,double g,double chezy,double floor_depth,
 double dry,int law,double*q){
 // taubot sigma kmax=1, no-wave law-of-wall conversion. Unlike FLOW this
 // diagnostic is CELL centred and uses the exact local quadratic decay.
 double c=chezy;
 if(law)c=sqrt(g)/.41*log1p(exp(1.)*expm1(.41*chezy/sqrt(g)));
 LOOP(n){double h=q[i];
  if(h<=dry){q[n+i]=0.;q[2*n+i]=0.;continue;}
  double u=q[n+i]/h,v=q[2*n+i]/h;
  double factor=1./(1.+dt*g*sqrt(u*u+v*v)/(c*c*fmax(h,floor_depth)));
  q[n+i]*=factor;q[2*n+i]*=factor;
 }
}
