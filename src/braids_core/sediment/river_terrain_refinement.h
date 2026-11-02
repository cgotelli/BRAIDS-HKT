// Geometry-only slope-break detector; identical OpenMP/CUDA arithmetic.
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
ENTRY void terrain_gradient(int n,const int*off,const int*adj,const double*xy,
 const double*z,double*grad){
 LOOP(n){double xx=0.,yx=0.,yy=0.,xz=0.,yz=0.;
  for(int e=off[i];e<off[i+1];++e){int j=adj[e];
   double dx=xy[2*j]-xy[2*i],dy=xy[2*j+1]-xy[2*i+1],r=dx*dx+dy*dy;
   if(r<=0.)continue;double w=1./r,dz=z[j]-z[i];
   xx+=w*dx*dx;yx+=w*dx*dy;yy+=w*dy*dy;xz+=w*dx*dz;yz+=w*dy*dz;}
  double det=xx*yy-yx*yx,tr=xx+yy;
  if(det>1e-12*tr*tr){grad[2*i]=(yy*xz-yx*yz)/det;grad[2*i+1]=(xx*yz-yx*xz)/det;}
  else if(tr>0.){grad[2*i]=(xx*xz+yx*yz)/(tr*tr);grad[2*i+1]=(yx*xz+yy*yz)/(tr*tr);}
  else{grad[2*i]=0.;grad[2*i+1]=0.;}
 }
}
ENTRY void terrain_tags(int n,double angle_limit,double detail_limit,const int*off,
 const int*adj,const double*xy,const double*z,const double*grad,double*angle,double*detail,int*tag){
 LOOP(n){double amax=0.,emax=0.;int hit=0;
  for(int e=off[i];e<off[i+1];++e){int j=adj[e];
   double dx=xy[2*j]-xy[2*i],dy=xy[2*j+1]-xy[2*i+1],r=hypot(dx,dy);
   if(r<=0.)continue;
   double si=(grad[2*i]*dx+grad[2*i+1]*dy)/r,sj=(grad[2*j]*dx+grad[2*j+1]*dy)/r;
   double sec=(z[j]-z[i])/r;
   double a=fmax(fabs(atan(si)-atan(sj)),fmax(fabs(atan(si)-atan(sec)),fabs(atan(sj)-atan(sec))));
   double d=r*fmax(.5*fabs(si-sj),fmax(fabs(si-sec),fabs(sj-sec)));
   amax=fmax(amax,a);emax=fmax(emax,d);
   if(a>=angle_limit&&d>=detail_limit)hit=1;
  }angle[i]=amax;detail[i]=emax;tag[i]=hit;
 }
}
ENTRY void terrain_halo(int n,const int*off,const int*adj,const int*tag,int*out){
 LOOP(n){int value=tag[i];for(int e=off[i];e<off[i+1]&&!value;++e)value=tag[adj[e]];out[i]=value;}
}
