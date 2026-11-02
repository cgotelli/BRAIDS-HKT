// Zero-time conservative sparse transfer, shared OpenMP/CUDA arithmetic.
#ifdef __CUDACC__
#define ENTRY extern "C" __global__
#define LOOP(N) for(int i=blockDim.x*blockIdx.x+threadIdx.x;i<N;i+=blockDim.x*gridDim.x)
#else
#include <omp.h>
#define ENTRY extern "C" __declspec(dllexport)
#define LOOP(N) _Pragma("omp parallel for schedule(static)") for(int i=0;i<N;++i)
extern "C" __declspec(dllexport) void mobile_threads(int n){omp_set_dynamic(0);omp_set_num_threads(n);}
#endif
ENTRY void river_remap(int nt,int ns,int nv,const int*offset,const int*index,const double*weight,
 const double*source,double*target){
 LOOP(nt*nv){int c=i%nt,k=i/nt;double s=0.;
  for(int e=offset[c];e<offset[c+1];++e)s+=weight[e]*source[k*ns+index[e]];
  target[i]=s;
 }
}
