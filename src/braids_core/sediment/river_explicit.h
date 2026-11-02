// Shared OpenMP/CUDA CFL reduction input. Actual reconstructed interior speed;
// prescribed exterior flux uses its actual boundary and cell water states.
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
ENTRY void explicit_frequency(int n,int nf,double grav,double dry,const double*q,
 const double*speed,const double*trace,const int*mask,const int*owner,
 const double*normal,const double*area,const double*length,const int*off,
 const int*faces,double*out){
 LOOP(n){double sum=0.;for(int k=off[i];k<off[i+1];++k){int f=faces[k];double s=speed[f];
  if(mask[f]){int o=owner[f];double h=q[o],hb=trace[f],nx=normal[f],ny=normal[nf+f];
   double a=h>dry?fabs((q[n+o]*nx+q[2*n+o]*ny)/h)+sqrt(grav*h):0.;
   double b=hb>dry?fabs((trace[nf+f]*nx+trace[2*nf+f]*ny)/hb)+sqrt(grav*hb):0.;
   s=fmax(a,b);}
  sum+=length[f]*s;}
  out[i]=sum/area[i];
 }
}
