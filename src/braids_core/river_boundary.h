// Shared OpenMP/CUDA boundary policy. Input is already class solid volume.
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

ENTRY void river_prescribed_boundary(int nf,const int*selected,const int*active,
 const double*water,const double*feed,double*flux){
 LOOP(nf){
  if(!selected[i])continue;
  if(!active[i])flux[i]=0.;
  else if(selected[i]==1 && water[i]<=0.)flux[i]=-feed[i];
  else if(selected[i]==3)flux[i]=water[i]>0.?fmax(0.,flux[i]):0.;
  else if(selected[i]==4 && water[i]<=0.)flux[i]=feed[i]*water[i];
  // Selection 2 is free equilibrium: computed signed capacity on either direction.
  // Outward hydraulic flow keeps computed signed transport, including slopes.
 }
}
