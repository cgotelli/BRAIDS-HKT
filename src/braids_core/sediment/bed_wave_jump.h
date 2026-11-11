// Limited true-face bed values for the existing bed-diagonal viscosity.
// One writer per cell/face; shared OpenMP/CUDA body, no atomic reductions.
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

ENTRY void bed_jump_limits(int n,int nf,const double*z,const double*gradient,
 const int*off,const int*faces,const int*owner,const int*nb,
 const double*delta,double*out){
 LOOP(n){
  double lo=z[i],hi=z[i],predlo=0.,predhi=0.;int na=off[n];
  double gx=gradient[i],gy=gradient[n+i];
  for(int k=off[i];k<off[i+1];++k){int f=faces[k],j=owner[f]==i?nb[f]:owner[f];
   if(j>=0){lo=fmin(lo,z[j]);hi=fmax(hi,z[j]);
    double change=gx*delta[k]+gy*delta[na+k];predlo=fmin(predlo,change);predhi=fmax(predhi,change);}}
  double phi=1.;
  const double eps=64.*2.220446049250313e-16*fmax(fabs(z[i]),fmax(fabs(lo),fabs(hi)));
  // Limiting at neighbour centroids preserves affine fields on skewed cells.
  // A true face need not lie inside the neighbours' centroid-value range.
  // Use stencil extrema rather than noisy near-orthogonal pair quotients.
  if(predhi>eps)phi=fmin(phi,fmax(0.,(hi-z[i])/predhi));
  if(predlo<-eps)phi=fmin(phi,fmax(0.,(lo-z[i])/predlo));
  out[i]=phi*gx;out[n+i]=phi*gy;
 }
}

// Separate numerical jump: never modify hydraulic traces or HR opening.
ENTRY void bed_jump_faces(int nf,int n,const double*z,const double*gradient,
 const int*owner,const int*nb,const double*disp,double*out){
 LOOP(nf){int o=owner[i],j=nb[i];out[i]=0.;if(j<0)continue;
  double left=z[o]+gradient[o]*disp[i]+gradient[n+o]*disp[nf+i];
  double right=z[j]+gradient[j]*disp[2*nf+i]+gradient[n+j]*disp[3*nf+i];
  out[i]=right-left;
 }
}
