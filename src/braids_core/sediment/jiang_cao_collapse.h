// Conservative geometric repose relaxation, shared OpenMP/CUDA arithmetic.
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

// One solid-volume proposal per actual shared face. No exterior transfer.
// Full vector LS slope augments the pair secant: catches diagonal steepness.
ENTRY void jc_repose_propose(int n,int nf,double B,double slope,
 const int*owner,const int*nb,const double*area,const double*distance,
 const double*weight,const double*z,const double*grad,double*F,double*excess){
 LOOP(nf){int o=owner[i],j=nb[i];F[i]=0.;excess[i]=0.;if(j<0)continue;
  double dz=z[o]-z[j],sec=fabs(dz)/distance[i];
  double s=fmax(sec,fmax(hypot(grad[o],grad[n+o]),hypot(grad[j],grad[n+j])));
  excess[i]=fmax(0.,s-slope);
  if(s>slope){double harmonic=area[o]*(area[j]/(area[o]+area[j]));
   F[i]=B*weight[i]*harmonic*dz*(1.-slope/s);}
 }
}

ENTRY void jc_repose_donors(int n,const int*offset,const int*faces,const int*owner,
 const double*area,const double*stock,const double*F,double*scale){
 LOOP(n){double outgoing=0.;for(int k=offset[i];k<offset[i+1];++k){int f=faces[k];
  outgoing+=fmax(0.,(owner[f]==i?1.:-1.)*F[f]);}
  double available=stock[i]*area[i];scale[i]=outgoing>available?(available/outgoing)*(1.-32.*2.2204460492503131e-16):1.;
 }
}

ENTRY void jc_repose_scale(int nf,const int*owner,const int*nb,const double*scale,
 double*F,double*ledger){
 LOOP(nf){if(nb[i]<0){F[i]=0.;continue;}
  F[i]*=scale[F[i]>=0.?owner[i]:nb[i]];ledger[i]+=F[i];}
}

ENTRY void jc_repose_gather(int n,double B,const int*offset,const int*faces,
 const int*owner,const double*area,const double*F,double*stock,double*z){
 LOOP(n){double volume=0.;for(int k=offset[i];k<offset[i+1];++k){int f=faces[k];
  volume-=(owner[f]==i?1.:-1.)*F[f];}
  if(volume!=0.){double db=volume/area[i];stock[i]+=db;z[i]+=db/B;}
 }
}
