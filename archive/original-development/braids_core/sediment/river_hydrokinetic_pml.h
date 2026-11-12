// Algebraic depth-averaged Prandtl mixing length: nu = a u_* H.
#ifdef __CUDACC__
#define ENTRY extern "C" __global__
#define LOOP(N) for(int i=blockDim.x*blockIdx.x+threadIdx.x;i<N;i+=blockDim.x*gridDim.x)
#else
#include <cmath>
#include <omp.h>
#define ENTRY extern "C" __declspec(dllexport)
#define LOOP(N) _Pragma("omp parallel for schedule(static)") for(int i=0;i<N;++i)
extern "C" __declspec(dllexport) void hkt_stress_threads(int n){omp_set_dynamic(0);omp_set_num_threads(n);}
#endif

ENTRY void hkt_pml_nu(int n,double dry,double gravity,double chezy,double a,
 const double*q,double*out){
 LOOP(n){
  if(q[i]<=dry){out[i]=0.;continue;}
  double u=q[n+i]/q[i],v=q[2*n+i]/q[i];
  out[i]=a*sqrt(gravity)/chezy*sqrt(u*u+v*v)*q[i];
 }
}

ENTRY void hkt_pml_walls(int nf,int n,double dry,const double*q,const double*nu,
 const int*owner,const int*nb,const int*bid,const double*normal,const double*distance,
 double*out){
 LOOP(nf){
  if(nb[i]>=0 || bid[i]>=0)continue;
  int o=owner[i];if(q[o]<=dry)continue;
  double u=q[n+o]/q[o],v=q[2*n+o]/q[o],nx=normal[i],ny=normal[nf+i];
  double un=u*nx+v*ny,scale=-nu[o]*q[o]/distance[i];
  out[i]=scale*(u+un*nx);out[nf+i]=scale*(v+un*ny);
 }
}
