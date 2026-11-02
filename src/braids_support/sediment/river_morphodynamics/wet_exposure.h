// Research geometry only. Same arithmetic compiled for OpenMP and CUDA.
#ifdef __CUDACC__
#define ENTRY extern "C" __global__
#define LOCAL __device__
#define LOOP(N) for(int i=blockDim.x*blockIdx.x+threadIdx.x;i<N;i+=blockDim.x*gridDim.x)
#else
#include <cmath>
#include <omp.h>
#define ENTRY extern "C" __declspec(dllexport)
#define LOCAL inline
#define LOOP(N) _Pragma("omp parallel for schedule(static)") for(int i=0;i<N;++i)
extern "C" __declspec(dllexport) void mobile_threads(int n){omp_set_dynamic(0);omp_set_num_threads(n);}
#endif

// Exact volume / full cell area and wet-area fraction of an affine bed.
// a,b are its positive directional reliefs; s is stage above lowest corner.
LOCAL void measure(double a,double b,double s,double&v,double&area){
 double r=a+b;if(s<=0.){v=area=0.;return;}
 if(s>=r){v=s-.5*r;area=1.;return;}
 if(b>a){double t=a;a=b;b=t;}
 if(b<=1.e-14*a){v=s*s/(2.*a);area=s/a;return;}
 if(s<b){v=s*s*s/(6.*a*b);area=s*s/(2.*a*b);}
 else if(s<=a){v=(s*s-s*b+b*b/3.)/(2.*a);area=(s-.5*b)/a;}
 else{double t=r-s;v=s-.5*r+t*t*t/(6.*a*b);area=1.-t*t/(2.*a*b);}
}

ENTRY void exposure_gradient(int n,const double*z,const int*off,const int*faces,
 const int*owner,const int*nb,const double*delta,const double*inv,double*grad){
 LOOP(n){int na=off[n];double bx=0.,by=0.;
  for(int k=off[i];k<off[i+1];++k){int f=faces[k],j=owner[f]==i?nb[f]:owner[f];if(j<0)continue;
   double dx=delta[k],dy=delta[na+k],w=1./(dx*dx+dy*dy),dv=z[j]-z[i];bx+=w*dx*dv;by+=w*dy*dv;
  }
  grad[i]=inv[i]*bx+inv[n+i]*by;grad[n+i]=inv[n+i]*bx+inv[2*n+i]*by;
 }
}

ENTRY void exposure_cells(int n,double dx,double dy,double sedthr,const double*h,
 const double*z,const double*grad,double*out){
 LOOP(n){double a=fabs(grad[i])*dx,b=fabs(grad[n+i])*dy,r=a+b,v,area,s;
  if(h[i]<=0.)s=0.;
  else if(h[i]>=.5*r)s=h[i]+.5*r;
  else{double lo=0.,hi=r;for(int k=0;k<52;++k){double mid=.5*(lo+hi);measure(a,b,mid,v,area);if(v<h[i])lo=mid;else hi=mid;}s=.5*(lo+hi);}
  measure(a,b,s,v,area);out[i]=z[i]-.5*r+s;out[n+i]=area;out[3*n+i]=v-h[i];
  measure(a,b,s-sedthr,v,area);out[2*n+i]=area;
 }
}

LOCAL void restrict_wet(double stage,double midbed,double slope,double&lo,double&hi){
 if(fabs(slope)<1.e-30){if(!(stage>midbed))hi=lo;return;}
 double crossing=(stage-midbed)/slope;
 if(slope>0.)hi=fmin(hi,crossing);else lo=fmax(lo,crossing);
}

ENTRY void exposure_faces(int nf,int n,const int*owner,const int*nb,const double*normal,
 const double*length,const double*disp,const double*z,const double*grad,const double*cell,double*out){
 LOOP(nf){int o=owner[i],j=nb[i];if(j<0){out[i]=out[nf+i]=1.;continue;}
  double lo=-.5*length[i],hi=.5*length[i],stage=fmax(cell[o],cell[j]);
  double tx=-normal[nf+i],ty=normal[i];
  for(int side=0;side<2;++side){int c=side?j:o;
   double bed=z[c]+grad[c]*disp[2*side*nf+i]+grad[n+c]*disp[(2*side+1)*nf+i];
   restrict_wet(stage,bed,grad[c]*tx+grad[n+c]*ty,lo,hi);
  }
  out[i]=fmin(1.,fmax(0.,(hi-lo)/length[i]));
  // Stored ONLY as the rejected area-as-face-weight diagnostic.
  out[nf+i]=cell[n+(z[o]>=z[j]?o:j)];
 }
}

ENTRY void exposure_scale(int nf,const double*factor,double*flux){LOOP(nf){flux[i]*=factor[i];}}
