// Research bank eligibility only; no transport-capacity or hydraulic change.
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

ENTRY void handoff_depth(int n,double dry,double sedthr,const double*h,
 const double*geometry,double*wetdepth,int*eligible){
 LOOP(n){double area=geometry[n+i];
  double value=area>0. ? h[i]/area : 0.;wetdepth[i]=value;
  eligible[i]=(h[i]>dry && value>sedthr)?1:0;
 }
}
