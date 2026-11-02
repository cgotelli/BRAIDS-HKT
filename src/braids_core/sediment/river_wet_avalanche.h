// Classic FLOW wet-slope flux, mapped to true finite-volume faces.
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

ENTRY void wet_avalanche_geometry(int nf,int n,const int*owner,const int*nb,
 const int*offset,const int*faces,const double*normal,const double*length,
 const double*area,const double*xy,double*weight){
 LOOP(nf){
  weight[i]=0.;int o=owner[i],j=nb[i];if(j<0)continue;
  double nx=normal[i],ny=normal[nf+i],span[2]={0.,0.};
  int cells[2]={o,j};
  for(int s=0;s<2;++s)for(int k=offset[cells[s]];k<offset[cells[s]+1];++k){
   int f=faces[k];span[s]+=.5*length[f]*fabs(nx*normal[f]+ny*normal[nf+f]);
  }
  // Allocate the cell area to each segment of its projected side. Otherwise
  // two fine neighbours each receive the entire coarse-cell area.
  double ao=area[o]*length[i]/span[0],aj=area[j]*length[i]/span[1];
  double dn=(xy[j]-xy[o])*nx+(xy[n+j]-xy[n+o])*ny;
  weight[i]=(ao/(ao+aj))*aj*dn/length[i];
 }
}

ENTRY void wet_avalanche_flux(int nf,int n,double dry,double slope,double time,
 const int*owner,const int*nb,const int*active,const double*q,
 const double*normal,const double*bedface,const double*weight,double*out){
 LOOP(nf){
  out[i]=0.;int o=owner[i],j=nb[i];
  if(j<0||!active[i]||q[o]<=dry||q[j]<=dry)continue;
  // bedface contains positive-down, nonorthogonally corrected bed gradient.
  double gx=bedface[2*nf+i],gy=bedface[3*nf+i],s=sqrt(gx*gx+gy*gy);
  if(s>slope)out[i]=weight[i]*(gx*normal[i]+gy*normal[nf+i])*(1.-slope/s)/time;
 }
}
