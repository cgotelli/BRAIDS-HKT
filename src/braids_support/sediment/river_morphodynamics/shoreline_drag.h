// Isolated FLOW momentum working-depth control, not a water-depth floor.
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
ENTRY void shoreline_drag(int n,double dt,double gravity,double chezy,double dry,double floor,double*q){
 LOOP(n){double h=q[i];
  if(h<=dry){q[n+i]=0.;q[2*n+i]=0.;continue;}
  double speed=hypot(q[n+i],q[2*n+i])/h;
  double f=1./(1.+dt*gravity*speed/(chezy*chezy*fmax(h,floor)));
  q[n+i]*=f;q[2*n+i]*=f;
 }
}
