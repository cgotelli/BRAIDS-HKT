// Conservative endpoint positivity for the SAME midpoint residual on CPU/CUDA.
// Available water is the accepted interval-start inventory, not midpoint water.
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
ENTRY void midpoint_inventory(int n,double dt,const double*base,const double*area,const double*length,
 const int*off,const int*faces,const int*signs,const double*water,double*theta){
 LOOP(n){double out=0.;
  for(int k=off[i];k<off[i+1];++k){int f=faces[k];out+=fmax(signs[k]*water[f],0.)*length[f];}
  theta[i]=1.;
  if(out>0. && dt*out>=base[i]*area[i])
   theta[i]=fmin(1.,base[i]*area[i]/(dt*out))*(1.-32.*2.2204460492503131e-16);
 }
}
ENTRY void midpoint_faces(int nf,int n,double dry,const double*q,const double*theta,
 const int*o,const int*j,const int*bid,double*water,double*correction,int*status){
 LOOP(nf){status[i]=0;correction[i]=0.;correction[nf+i]=0.;
  if(water[i]==0.)continue;
  int donor=water[i]<0.?j[i]:o[i];
  if(donor<0 || theta[donor]==1.)continue;
  // Prescribed boundary discharge must not be silently reduced.
  if(j[i]<0 && bid[i]>=0){status[i]=1;continue;}
  double delta=(theta[donor]-1.)*water[i];water[i]*=theta[donor];
  if(q[donor]>dry){correction[i]=delta*q[n+donor]/q[donor];correction[nf+i]=delta*q[2*n+donor]/q[donor];}
 }
}
ENTRY void midpoint_gather(int n,int nf,const double*area,const double*length,
 const int*off,const int*faces,const int*signs,const double*water,const double*correction,double*rate){
 LOOP(n){double mass=0.,mx=0.,my=0.;
  for(int k=off[i];k<off[i+1];++k){int f=faces[k];double w=signs[k]*length[f]/area[i];
   mass-=w*water[f];mx-=w*correction[f];my-=w*correction[nf+f];}
  rate[i]=mass;rate[n+i]+=mx;rate[2*n+i]+=my;
 }
}
