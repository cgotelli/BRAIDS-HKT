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
ENTRY void jc206_relax(int n,double dt,double dry,double hr,const double*q,
 const double*raw,const double*a,double*state){
 LOOP(n){double h=q[i],H=state[i];if(H<=dry){state[6*n+i]=0.;continue;}
  double U=h>dry?hypot(q[n+i],q[2*n+i])/h:0.;
  double L=fmax(h,hr)*(1.-2.*a[i])/(.32*a[i]),rate=U/L;
  double target=h>hr?U*h*raw[6*n+i]:0.;
  double I=state[6*n+i]/H;
  state[6*n+i]=H*(I+(target-I)*(-expm1(-rate*dt)));
 }
}
ENTRY void jc206_scalar_faces(int n,int nf,double incoming,const int*o,const int*j,
 const int*kind,const double*xy,const double*mid,const double*Q,const double*c,
 const double*grad,double*low,double*high){
 LOOP(nf){low[i]=high[i]=0.;if(kind[i]==1)continue;
  int donor=Q[i]>=0.?o[i]:j[i];
  if(donor<0){low[i]=high[i]=Q[i]*incoming;continue;}
  low[i]=Q[i]*c[donor];
  double value=c[donor]+grad[donor]*(mid[i]-xy[donor])+grad[n+donor]*(mid[nf+i]-xy[n+donor]);
  high[i]=Q[i]*value;
 }
}
