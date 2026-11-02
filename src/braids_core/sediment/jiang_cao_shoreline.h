// Research wet/dry geometry and two-phase flux correction, CPU/CUDA shared.
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

// Fresh masked LS in physical coordinates. Rank-one stencil uses its
// Moore-Penrose inverse, not a preferred x/y direction. Dry samples excluded.
ENTRY void jc201_gradient(int n,int nf,int limited,const int*off,const int*faces,
 const int*owner,const int*nb,const double*xy,const double*mid,const int*active,
 const double*v,double*out){
 LOOP(n){out[i]=out[n+i]=0.;if(!active[i])continue;
  double xx=0.,yy=0.,xyv=0.,bx=0.,by=0.,lo=v[i],hi=v[i];
  for(int k=off[i];k<off[i+1];++k){int f=faces[k],j=owner[f]==i?nb[f]:owner[f];
   if(j<0||!active[j])continue;
   double dx=xy[j]-xy[i],dy=xy[n+j]-xy[n+i],w=1./(dx*dx+dy*dy),dv=v[j]-v[i];
   xx+=w*dx*dx;yy+=w*dy*dy;xyv+=w*dx*dy;bx+=w*dx*dv;by+=w*dy*dv;
   lo=fmin(lo,v[j]);hi=fmax(hi,v[j]);}
  double tr=xx+yy,det=xx*yy-xyv*xyv,gx=0.,gy=0.;
  if(det>1.e-13*tr*tr){gx=(yy*bx-xyv*by)/det;gy=(xx*by-xyv*bx)/det;}
  else if(tr>0.){gx=(xx*bx+xyv*by)/(tr*tr);gy=(xyv*bx+yy*by)/(tr*tr);}
  double phi=1.;
  if(limited)for(int k=off[i];k<off[i+1];++k){int f=faces[k];
   double d=gx*(mid[f]-xy[i])+gy*(mid[nf+f]-xy[n+i]);
   if(d>0.)phi=fmin(phi,(hi-v[i])/d);else if(d<0.)phi=fmin(phi,(lo-v[i])/d);}
  out[i]=phi*gx;out[n+i]=phi*gy;
 }
}

ENTRY void jc201_low(int n,int nf,const int*o,const int*j,const int*kind,
 const double*h,const double*m,const double*Q,const double*cin,double*F){
 LOOP(nf){int a=o[i],b=j[i];F[i]=0.;if(kind[i]==1)continue;
  if(Q[i]>=0.)F[i]=h[a]>0.?Q[i]*(m[a]/h[a]):0.;
  else if(b>=0)F[i]=h[b]>0.?Q[i]*(m[b]/h[b]):0.;
  else if(kind[i]==2)F[i]=Q[i]*cin[i];
 }
}

// Zalesak-style shared-face antidiffusive flux bounds. Protect BOTH m>=0
// and packing complement B*h-m>=0 with the actual hydraulic endpoint hnew.
ENTRY void jc201_bounds(int n,double dt,double B,const int*off,const int*faces,
 const int*o,const double*area,const double*h,const double*m,const double*Q,
 const double*low,const double*high,double*hn,double*ml,double*plus,double*minus,int*bad){
 LOOP(n){double sq=0.,sf=0.,p=0.,r=0.,outq=0.;
  for(int k=off[i];k<off[i+1];++k){int f=faces[k];double sign=o[f]==i?1.:-1.;
   sq+=sign*Q[f];sf+=sign*low[f];outq+=fmax(0.,sign*Q[f]);
   double df=dt*sign*(high[f]-low[f]);p+=fmax(0.,-df);r+=fmax(0.,df);}
  double H=h[i]-dt*sq/area[i],M=m[i]-dt*sf/area[i];hn[i]=H;ml[i]=M;
  bad[i]=(H<0.||M<0.||M>B*H)?1:0;
  if(dt*outq>h[i]*area[i]*(1.+1.e-12))bad[i]=2;
  double available=fmax(0.,M*area[i]),room=fmax(0.,(B*H-M)*area[i]);
  plus[i]=p>0.?fmin(1.,(1.-32.*2.2204460492503131e-16)*room/p):1.;
  minus[i]=r>0.?fmin(1.,(1.-32.*2.2204460492503131e-16)*available/r):1.;
 }
}

ENTRY void jc201_correct(int nf,const int*o,const int*j,const double*low,
 const double*high,const double*plus,const double*minus,double*F,double*theta){
 LOOP(nf){double d=high[i]-low[i];int a=o[i],b=j[i];double t=1.;
  if(d>=0.){t=minus[a];if(b>=0)t=fmin(t,plus[b]);}
  else{t=plus[a];if(b>=0)t=fmin(t,minus[b]);}
  F[i]=low[i]+t*d;theta[i]=t;
 }
}

ENTRY void jc201_endpoint(int n,double dt,const int*off,const int*faces,
 const int*o,const double*area,const double*m,const double*F,double*out){
 LOOP(n){double sum=0.;for(int k=off[i];k<off[i+1];++k){int f=faces[k];sum+=(o[f]==i?1.:-1.)*F[f];}
  out[i]=m[i]-dt*sum/area[i];}
}

ENTRY void jc201_manning(int n,double dt,double gravity,double dry,
 const double*rough,double*q){
 LOOP(n){double h=q[i];if(h<=dry){q[n+i]=q[2*n+i]=0.;continue;}
  double speed=hypot(q[n+i],q[2*n+i])/h;
  double factor=1./(1.+dt*gravity*rough[i]*rough[i]*speed/pow(h,4./3.));
  q[n+i]*=factor;q[2*n+i]*=factor;}
}
