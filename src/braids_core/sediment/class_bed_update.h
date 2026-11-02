// Classwise conservative stock limiting. Shared FP64 OpenMP/CUDA arithmetic.
// Each cell traverses its existing adjacency order; each face has one writer.
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

ENTRY void class_donor(int nc,int nf,int ng,double dt,const double*rho,const double*available,
 const double*flux,const double*length,const double*area,const int*off,
 const int*faces,const int*signs,double*out,double*last){
 LOOP(ng*nc){int grain=i/nc,c=i%nc;double outgoing=0.;
  for(int k=off[c];k<off[c+1];++k){int f=faces[k];
   outgoing+=fmax(0.,signs[k]*flux[grain*nf+f])*length[f];}
  // A binding donor quotient can round upward, then scaling and gathering
  // can spend a few ulps more than the inventory. Reserve only arithmetic
  // round-off at that face limiter, never clip the conservative cell update.
  // Count adjacency contributions so cropped/AMR cells have the same guard.
  const double eps=2.22044604925031308085e-16;
  double reserve=1.-(16.+2.*double(off[c+1]-off[c]))*eps;
  double factor=outgoing>0.?fmin(1.,fmax(0.,available[i]/rho[grain])*area[c]/(dt*outgoing)*reserve):1.;
  // At extreme depletion, division by density and face scaling enter the
  // subnormal range: the relative-epsilon reserve above no longer bounds
  // their absolute rounding error. Retain this vanishing stock, rather than
  // clipping a negative cell update (which would manufacture sediment).
  // sqrt(DBL_MIN) leaves ample exponent headroom for the volume/flux chain;
  // this is a numerical-range guard, not a physical sediment-depth cutoff.
  if(outgoing>0. && available[i]<1.4916681462400413e-154)factor=0.;
  out[i]=factor;if(grain==ng-1)last[c]=factor;
 }
}

ENTRY void class_scale(int nc,int nf,int ng,double dt,const int*owner,
 const int*nb,const double*fac,double*flux,double*integral,double*aux){
 LOOP(ng*nf){int grain=i/nf,f=i%nf;double value=flux[i];
  int donor=value>=0.?owner[f]:nb[f];
  if(donor>=0)value*=fac[grain*nc+donor];
  flux[i]=value;integral[i]=dt*value;
  if(grain==ng-1)aux[2*nf+f]=value;
 }
}

ENTRY void class_divergence(int nc,int nf,int ng,double dt,const double*rho,
 const double*flux,const double*length,const double*area,const int*off,
 const int*faces,const int*signs,double*delta,double*last){
 LOOP(ng*nc){int grain=i/nc,c=i%nc;double r=0.;
  for(int k=off[c];k<off[c+1];++k)
   r-=signs[k]*length[faces[k]]*flux[grain*nf+faces[k]];
  double rate=r/area[c];delta[i]=(rho[grain]*dt)*rate;
  if(grain==ng-1)last[c]=rate;
 }
}

ENTRY void class_ledger(int nf,int ng,const int*nb,const int*bid,
 const double*length,const double*integral,double*ledger){
 LOOP(ng*nf){int f=i%nf;ledger[i]-=integral[i]*length[f]*double(nb[f]<0&&bid[f]>=0);}
}
