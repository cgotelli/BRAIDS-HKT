// Objective symmetric-strain Reynolds stress. One FP64 OpenMP/CUDA body.
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

ENTRY void hkt_full_strain_faces(int nf,int n,double dry,const double*q,const double*z,
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
  double dn=dx*nx+dy*ny;
  double du=q[n+j]/q[j]-q[n+o]/q[o],dv=q[2*n+j]/q[j]-q[2*n+o]/q[o];
  double ux=wo*grad[o]+wj*grad[j],uy=wo*grad[n+o]+wj*grad[n+j];
  double vx=wo*grad[2*n+o]+wj*grad[2*n+j],vy=wo*grad[3*n+o]+wj*grad[3*n+j];
  // Recover the observed inter-centre difference without losing tangential
  // strain or confusing nonorthogonal displacement with face-normal distance.
  double cu=(du-ux*dx-uy*dy)/dn,cv=(dv-vx*dx-vy*dy)/dn;
  ux+=cu*nx;uy+=cu*ny;vx+=cv*nx;vy+=cv*ny;
  out[i]=visc*h*(2.*ux*nx+(uy+vx)*ny);
  out[nf+i]=visc*h*((uy+vx)*nx+2.*vy*ny);
 }
}
