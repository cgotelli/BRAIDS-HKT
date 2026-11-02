// Research-only shared OpenMP/CUDA hydraulic opening classifier.
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
ENTRY void hr_face_activity(int nf,double dry,const int*nb,const int*bid,
 const double*trace,int*active,double*opening){
 LOOP(nf){
  double e0=trace[i],e1=trace[4*nf+i],b0=trace[3*nf+i],b1=trace[7*nf+i];
  double top=fmax(b0,b1),h=fmax(0.,fmax(e0,e1)-top);
  double scale=fmax(dry,fmax(fabs(top),fmax(fabs(e0),fabs(e1))));
  double tol=16.*2.2204460492503131e-16*scale;
  opening[i]=h;
  // This tolerance resolves subtraction roundoff at the EXISTING dry depth.
  // Exterior boundary contracts are deliberately not replaced here.
  active[i]=nb[i]<0?(bid[i]>=0):(h>dry+tol);
 }
}
