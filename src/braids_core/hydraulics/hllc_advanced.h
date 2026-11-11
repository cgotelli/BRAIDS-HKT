// Explicit nonorthogonal viscous stability bound on the same CPU/CUDA body.
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

ENTRY void hllc_advanced_frequency(int n,int nf,double dry,double nu,
 const double*q,const double*metric,const int*owner,const int*nb,
 const int*off,const int*faces,const double*length,const double*area,double*out){
 LOOP(n){double sum=0.;
  if(q[i]>dry)for(int k=off[i];k<off[i+1];++k){int f=faces[k],j=nb[f],o=owner[f];
   if(j>=0 && q[o]>dry && q[j]>dry)sum+=nu*length[f]*metric[f];
  }
  out[i]=sum/area[i];
 }
}
