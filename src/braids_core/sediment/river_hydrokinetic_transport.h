// Depth-integrated standard k-epsilon transport and variable-viscosity
// momentum exchange.  One source is compiled for OpenMP and CUDA so the
// selected backend executes identical face arithmetic.
#ifdef __CUDACC__
#define ENTRY extern "C" __global__
#define LOOP(N) for(int i=blockDim.x*blockIdx.x+threadIdx.x;i<N;i+=blockDim.x*gridDim.x)
#else
#include <cmath>
#include <omp.h>
#define ENTRY extern "C" __declspec(dllexport)
#define LOOP(N) _Pragma("omp parallel for schedule(static)") for(int i=0;i<N;++i)
extern "C" __declspec(dllexport) void hkt_threads(int n){omp_set_dynamic(0);omp_set_num_threads(n);}
#endif

ENTRY void hkt_scalar_faces(int nf,int n,double dry,double sigma,double molecular,
 const double*q,const double*inventory,const double*nu,const double*grad,
 const double*water,double incoming,const int*owner,const int*nb,const int*bid,
 const double*distance,const double*displace,const double*diffusion_open,double*out){
 LOOP(nf){
  int o=owner[i],j=nb[i];out[i]=0.;
  if(j<0){
   if(bid[i]<0 || q[o]<=dry)continue;
   double co=inventory[o]/q[o];out[i]=water[i]*(water[i]<0.?incoming:co);continue;
  }
  bool owet=q[o]>dry,jwet=q[j]>dry;
  if(!owet && !jwet)continue;
  // Hydraulic wetting owns transfer into a dry cell; turbulence has no
  // independent diffusion across that interface.
  if(!owet || !jwet){
   double donor=water[i]>=0.?(owet?inventory[o]/q[o]:0.):(jwet?inventory[j]/q[j]:0.);
   out[i]=water[i]*donor;continue;
  }
  double co=inventory[o]/q[o],cj=inventory[j]/q[j];int up=water[i]>=0.?o:j;
  // Positivity-preserving first-order transport is intentional here.  Mean
  // velocity remains second order; this auxiliary RANS state must not create
  // negative k or epsilon at the compact disc source.
  double adv=up==o?co:cj;
  double d=distance[i]+distance[nf+i];
  double wo=distance[nf+i]/d,wj=distance[i]/d;
  double alpha=(molecular+(wo*nu[o]+wj*nu[j])/sigma)*diffusion_open[i];
  double h=fmin(q[o],q[j]);
  out[i]=water[i]*adv-alpha*h*(cj-co)/d;
 }
}

ENTRY void hkt_momentum_faces(int nf,int n,double dry,const double*q,const double*z,
 const double*grad,const double*nu,const int*owner,const int*nb,const double*normal,
 const double*distance,const double*xy,double*out){
 LOOP(nf){
  out[i]=0.;out[nf+i]=0.;int o=owner[i],j=nb[i];
  if(j<0 || q[o]<=dry || q[j]<=dry)continue;
  double h=fmin(q[o],q[j]);
  h=fmin(h,fmax(0.,.5*(q[o]+z[o]+q[j]+z[j])-fmax(z[o],z[j])));
  double d=distance[i]+distance[nf+i],wo=distance[nf+i]/d,wj=distance[i]/d;
  double visc=wo*nu[o]+wj*nu[j];
  double nx=normal[i],ny=normal[nf+i],dx=xy[j]-xy[o],dy=xy[n+j]-xy[n+o];
  double dn=dx*nx+dy*ny,tx=dx-dn*nx,ty=dy-dn*ny;
  double du=q[n+j]/q[j]-q[n+o]/q[o],dv=q[2*n+j]/q[j]-q[2*n+o]/q[o];
  double ux=wo*grad[o]+wj*grad[j],uy=wo*grad[n+o]+wj*grad[n+j];
  double vx=wo*grad[2*n+o]+wj*grad[2*n+j],vy=wo*grad[3*n+o]+wj*grad[3*n+j];
  out[i]=visc*h*(du-ux*tx-uy*ty)/dn;
  out[nf+i]=visc*h*(dv-vx*tx-vy*ty)/dn;
 }
}
