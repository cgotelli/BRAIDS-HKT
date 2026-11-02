// Named research direction-resolution regularization, identical CPU/CUDA.
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
ENTRY void jc206_direction(int n,const int*off,const int*nb,const double*xy,
 const int*active,const double*v,double*out){
 LOOP(n){
  for(int c=0;c<7;++c)out[c*n+i]=0.;
  if(!active[i])continue;
  double xx=0.,yy=0.,xyv=0.,bx[2]={0.,0.},by[2]={0.,0.},var=0.;int count=0;
  for(int l=off[i];l<off[i+1];++l){int j=nb[l];if(!active[j])continue;
   double dx=xy[j]-xy[i],dy=xy[n+j]-xy[n+i],w=1./(dx*dx+dy*dy);
   xx+=w*dx*dx;yy+=w*dy*dy;xyv+=w*dx*dy;++count;
   for(int c=0;c<2;++c){double d=v[c*n+j]-v[c*n+i];bx[c]+=w*dx*d;by[c]+=w*dy*d;var+=d*d;}
  }
  double tr=xx+yy,det=xx*yy-xyv*xyv;
  for(int c=0;c<2;++c){double gx=0.,gy=0.;
   if(det>1.e-13*tr*tr){gx=(yy*bx[c]-xyv*by[c])/det;gy=(xx*by[c]-xyv*bx[c])/det;}
   else if(tr>0.){gx=(xx*bx[c]+xyv*by[c])/(tr*tr);gy=(xyv*bx[c]+yy*by[c])/(tr*tr);}
   out[2*c*n+i]=gx;out[(2*c+1)*n+i]=gy;
  }
  double U=hypot(v[i],v[n+i]),sigma=count?sqrt(var/count):0.;
  double r=sigma>0.?fmin(1.,U/sigma):1.,weight=r*r*(3.-2.*r),k=0.;
  if(U>0.){double tx=v[i]/U,ty=v[n+i]/U;
   k=(tx*tx*out[2*n+i]+tx*ty*(out[3*n+i]-out[i])-ty*ty*out[n+i])/U;}
  out[4*n+i]=sigma;out[5*n+i]=weight;out[6*n+i]=k;
 }
}
