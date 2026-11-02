// Paired OpenMP/CUDA normal-to-transverse face transport reconstruction.
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
ENTRY void normal_cell_vector(int n,int nf,const int*off,const int*faces,
 const double*normal,const double*length,const double*flux,double*cell){
 LOOP(n){double xx=0.,xy=0.,yy=0.,bx=0.,by=0.;
  for(int k=off[i];k<off[i+1];++k){int f=faces[k];double nx=normal[f],ny=normal[nf+f],w=length[f];
   xx+=w*nx*nx;xy+=w*nx*ny;yy+=w*ny*ny;bx+=w*nx*flux[f];by+=w*ny*flux[f];
  }
  double det=xx*yy-xy*xy;
  cell[i]=(yy*bx-xy*by)/det;cell[n+i]=(xx*by-xy*bx)/det;
 }
}
// Geometry is immutable for this owner. Keep the original summation order and
// determinant, not a reciprocal approximation. Rebuild on each new leaf owner.
ENTRY void normal_cell_geometry(int n,int nf,const int*off,const int*faces,
 const double*normal,const double*length,double*geometry){
 LOOP(n){double xx=0.,xy=0.,yy=0.;
  for(int k=off[i];k<off[i+1];++k){int f=faces[k];double nx=normal[f],ny=normal[nf+f],w=length[f];
   xx+=w*nx*nx;xy+=w*nx*ny;yy+=w*ny*ny;
  }
  geometry[i]=xx;geometry[n+i]=xy;geometry[2*n+i]=yy;geometry[3*n+i]=xx*yy-xy*xy;
 }
}
ENTRY void normal_cell_vector_cached(int n,int nf,const int*off,const int*faces,
 const double*normal,const double*length,const double*geometry,const double*flux,double*cell){
 LOOP(n){double bx=0.,by=0.;
  for(int k=off[i];k<off[i+1];++k){int f=faces[k];double nx=normal[f],ny=normal[nf+f],w=length[f];
   bx+=w*nx*flux[f];by+=w*ny*flux[f];
  }
  double xx=geometry[i],xy=geometry[n+i],yy=geometry[2*n+i],det=geometry[3*n+i];
  cell[i]=(yy*bx-xy*by)/det;cell[n+i]=(xx*by-xy*bx)/det;
 }
}
ENTRY void transverse_face_vector(int nf,int n,const int*owner,const int*nb,
 const double*normal,const double*cell,double*face){
 LOOP(nf){int o=owner[i],j=nb[i];if(j<0)continue;
  double nx=normal[i],ny=normal[nf+i];
  double target=-ny*.5*(cell[o]+cell[j])+nx*.5*(cell[n+o]+cell[n+j]);
  double old=-ny*face[i]+nx*face[nf+i];
  face[i]-=ny*(target-old);face[nf+i]+=nx*(target-old);
 }
}
