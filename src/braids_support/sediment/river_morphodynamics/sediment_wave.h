// Isolated Mahdizadeh--Sharifi secant-wave direction experiment.
// Physical capacity, objective MUSCL gradients and bed-slope operator unchanged.
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

ENTRY void sediment_wave_faces(int nf,int n,double acceleration,const double*z,
 const double*cell,const double*grad,const int*owner,const int*nb,
 const double*normal,const double*disp,double*face,double*diagnostic){
 LOOP(nf){int o=owner[i],j=nb[i];diagnostic[i]=diagnostic[nf+i]=0.;if(j<0)continue;
  double nx=normal[i],ny=normal[nf+i],l[2],r[2];
  for(int c=0;c<2;++c){
   l[c]=cell[(3+c)*n+o]+grad[2*c*n+o]*disp[i]+grad[(2*c+1)*n+o]*disp[nf+i];
   r[c]=cell[(3+c)*n+j]+grad[2*c*n+j]*disp[2*nf+i]+grad[(2*c+1)*n+j]*disp[3*nf+i];
  }
  double dz=z[j]-z[o];
  double dq=(cell[3*n+j]-cell[3*n+o])*nx+(cell[4*n+j]-cell[4*n+o])*ny;
  // Roundoff-level bed jumps carry no usable secant information. The original
  // FLOW weak-interaction upwind closure remains for a locally flat bed.
  double eps=64.*2.2204460492503131e-16*fmax(fabs(z[o]),fabs(z[j]));
  if(fabs(dz)<=eps||dq==0.)continue;
  double speed=acceleration*dq/(2650.*dz);
  diagnostic[i]=speed;
  double previous=face[i]*nx+face[nf+i]*ny;
  double next=speed>0.?l[0]*nx+l[1]*ny:r[0]*nx+r[1]*ny;
  face[i]+=(next-previous)*nx;face[nf+i]+=(next-previous)*ny;
  diagnostic[nf+i]=next-previous;
 }
}
