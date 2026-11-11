// Conservative low-Froude acoustic correction on actual HLLC face states.
#ifdef __CUDACC__
#define ENTRY extern "C" __global__
#define LOOP(N) for(int i=blockDim.x*blockIdx.x+threadIdx.x;i<N;i+=blockDim.x*gridDim.x)
#else
#include <math.h>
#include <omp.h>
#define ENTRY extern "C" __declspec(dllexport)
#define LOOP(N) _Pragma("omp parallel for schedule(static)") for(int i=0;i<N;++i)
extern "C" __declspec(dllexport) void mobile_threads(int n){omp_set_dynamic(0);omp_set_num_threads(n);}
#endif

ENTRY void hllc_turbine_pressure(int n,int nf,double gravity,double dry,
 const double*half,const double*grad,const double*xy,const double*mid,
 const double*normal,const int*owner,const int*nb,double*fl,double*fr){
 LOOP(nf){int a=owner[i],b=nb[i];if(b<0)continue;
  if(half[a]-half[3*n+a]<=dry || half[b]-half[3*n+b]<=dry)continue;
  double L[4],R[4];
  for(int m=0;m<4;++m){
   L[m]=half[m*n+a]+grad[2*m*n+a]*(mid[i]-xy[a])+grad[(2*m+1)*n+a]*(mid[nf+i]-xy[n+a]);
   R[m]=half[m*n+b]+grad[2*m*n+b]*(mid[i]-xy[b])+grad[(2*m+1)*n+b]*(mid[nf+i]-xy[n+b]);
  }
  double bed=fmax(L[3],R[3]),hl=L[0]-bed,hr=R[0]-bed,h=.5*(hl+hr);
  if(fmin(hl,hr)<=dry || fabs(hr-hl)>.005*h)continue;
  double c=sqrt(gravity*h),theta=fmin(1.,fmax(hypot(L[1],L[2]),hypot(R[1],R[2]))/c);
  double nx=normal[i],ny=normal[nf+i],du=(R[1]-L[1])*nx+(R[2]-L[2])*ny;
  double delta=.5*(1.-theta)*c*(2.*hl*hr/(hl+hr))*du;
  fl[nf+i]+=delta*nx;fr[nf+i]+=delta*nx;
  fl[2*nf+i]+=delta*ny;fr[2*nf+i]+=delta*ny;
 }
}
