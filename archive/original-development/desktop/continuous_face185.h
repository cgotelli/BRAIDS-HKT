// Isolated continuous normal capacity assembly. Reconstruction is unchanged.
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
ENTRY void capacity_reconstructed_faces(int nf,int n,const double*cell,
 const double*grad,const int*owner,const int*nb,const double*normal,
 const double*distance,const double*disp,double*out){
 LOOP(nf){int o=owner[i],j=nb[i];if(j<0)continue;
  double nx=normal[i],ny=normal[nf+i],left[2],right[2];
  for(int c=0;c<2;++c){
   left[c]=cell[(3+c)*n+o]+grad[2*c*n+o]*disp[i]+grad[(2*c+1)*n+o]*disp[nf+i];
   right[c]=cell[(3+c)*n+j]+grad[2*c*n+j]*disp[2*nf+i]+grad[(2*c+1)*n+j]*disp[3*nf+i];
  }
  double a=left[0]*nx+left[1]*ny,b=right[0]*nx+right[1]*ny;
  // Split incoming contributions continuously. Identical to the old donor
  // branch when both traces have the same sign; no zero-sign jump otherwise.
  double un=fmax(a,0.)+fmin(b,0.);
  double d=distance[i]+distance[nf+i],wo=distance[nf+i]/d,wj=distance[i]/d;
  double ut=wo*(-left[0]*ny+left[1]*nx)+wj*(-right[0]*ny+right[1]*nx);
  out[i]=un*nx-ut*ny;out[nf+i]=un*ny+ut*nx;
 }
}
