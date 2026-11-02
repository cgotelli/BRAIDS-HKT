// Geometry-corrected auxiliary faces for Cartesian cut/merged cells.
// Same OpenMP/CUDA bodies. Interior corrections only; physical coefficients,
// boundary treatment, bed reconstruction and f-wave hydraulics are unchanged.
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

ENTRY void geometry_scalar_correction(int nf,int n,double diffusion,const double*q,const double*hi,
 const double*grad,const int*owner,const int*nb,const double*normal,
 const double*distance,const double*xy,double*flux){
 LOOP(nf){int o=owner[i],j=nb[i];if(j<0)continue;
  double d=distance[i]+distance[nf+i],wo=distance[nf+i]/d,wj=distance[i]/d;
  double nx=normal[i],ny=normal[nf+i],dx=xy[j]-xy[o],dy=xy[n+j]-xy[n+o],dn=dx*nx+dy*ny;
  double gx=wo*grad[6*n+o]+wj*grad[6*n+j],gy=wo*grad[7*n+o]+wj*grad[7*n+j];
  double jump=hi[j]/q[j]-hi[o]/q[o];
  double derivative=(jump-gx*(dx-dn*nx)-gy*(dy-dn*ny))/dn;
  // Replace only the existing diffusive contribution; leave advection intact.
  flux[i]+=diffusion*.5*(q[o]+q[j])*(jump/d-derivative);
 }
}

ENTRY void geometry_momentum_faces(int nf,int n,int correct_stress,int correct_viscosity,
 double nu,const double*q,const double*B,const double*grad,const double*gradB,
 const int*owner,const int*nb,const double*normal,const double*distance,
 const double*xy,const double*displace,double*out){
 LOOP(nf){int o=owner[i],j=nb[i];double nx=normal[i],ny=normal[nf+i],b=B[o],vx=0.,vy=0.;
  if(j>=0){
   double d=distance[i]+distance[nf+i],wo=distance[nf+i]/d,wj=distance[i]/d;
   if(correct_stress){
    double bo=B[o]+gradB[6*n+o]*displace[i]+gradB[7*n+o]*displace[nf+i];
    double bj=B[j]+gradB[6*n+j]*displace[2*nf+i]+gradB[7*n+j]*displace[3*nf+i];
    b=wo*bo+wj*bj;
   }else b=wo*B[o]+wj*B[j];
   double du=q[n+j]/q[j]-q[n+o]/q[o],dv=q[2*n+j]/q[j]-q[2*n+o]/q[o];
   if(correct_viscosity){
    double dx=xy[j]-xy[o],dy=xy[n+j]-xy[n+o],dn=dx*nx+dy*ny;
    double ux=wo*grad[o]+wj*grad[j],uy=wo*grad[n+o]+wj*grad[n+j];
    double vxg=wo*grad[2*n+o]+wj*grad[2*n+j],vyg=wo*grad[3*n+o]+wj*grad[3*n+j];
    vx=nu*(du-ux*(dx-dn*nx)-uy*(dy-dn*ny))/dn;
    vy=nu*(dv-vxg*(dx-dn*nx)-vyg*(dy-dn*ny))/dn;
   }else{vx=nu*du/d;vy=nu*dv/d;}
  }
  out[i]=b*ny+vx;out[nf+i]=-b*nx+vy;
 }
}
