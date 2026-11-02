// Boundary-only exact SWE Riemann flux, common virtual reservoir velocity and
// exact net section Q. Interior augmented f-waves are unchanged.
#ifdef __CUDACC__
#define ENTRY extern "C" __global__
#define LOCAL __device__ __forceinline__
#define LEADER if(threadIdx.x==0)
#define LANES for(int lane=threadIdx.x;lane<128;lane+=blockDim.x)
#define SYNC __syncthreads()
#else
#include <cmath>
#include <omp.h>
using std::isfinite;
#define ENTRY extern "C" __declspec(dllexport)
#define LOCAL inline
#define LEADER _Pragma("omp single nowait")
#define LANES _Pragma("omp for schedule(static) nowait") for(int lane=0;lane<128;++lane)
#define SYNC _Pragma("omp barrier")
extern "C" __declspec(dllexport) void mobile_threads(int n){omp_set_dynamic(0);omp_set_num_threads(n);}
#endif

struct InletRiemann {double h,u,derivative;int iterations,error;};
struct InletWork {
 double part[6][128],lo,hi,velocity,rate,gross,derivative,area,eta,reference_area,scale;
 int error,iterations,max_inner;
};

LOCAL void inlet_wave(double h,double hk,double g,double &f,double &df){
 if(h>hk){double a=sqrt(.5*g*(1./h+1./hk));f=(h-hk)*a;df=a-.25*g*(h-hk)/(a*h*h);}
 else{double c=sqrt(g*h);f=2.*(c-sqrt(g*hk));df=sqrt(g/h);}
}

LOCAL InletRiemann inlet_exact(double hl,double ul,double hr,double ur,double g){
 InletRiemann r={0.,0.,0.,0,0};double cl=sqrt(g*hl),cr=sqrt(g*hr);
 if(hl==0. && hr==0.)return r;
 if(hr==0. || hl==0. || ur-ul>=2.*(cl+cr)){
  if(hl>0. && ul-cl>=0.){r.h=hl;r.u=ul;r.derivative=hl;}
  else if(hl>0. && ul+2.*cl>0.){r.u=(ul+2.*cl)/3.;r.h=r.u*r.u/g;r.derivative=r.h;}
  else if(hr>0. && ur+cr<=0.){r.h=hr;r.u=ur;}
  else if(hr>0. && ur-2.*cr<0.){r.u=(ur-2.*cr)/3.;r.h=r.u*r.u/g;}
  return r;
 }
 // Exact shock/rarefaction intersection, safeguarded Newton in positive h.
 double guess=.5*(cl+cr)-.25*(ur-ul);guess=guess*guess/g;
 double lo=0.,hi=fmax(fmax(hl,hr),guess),fl,fr,dl,dr;
 inlet_wave(hi,hl,g,fl,dl);inlet_wave(hi,hr,g,fr,dr);
 for(int j=0;j<60 && fl+fr+ur-ul<0.;++j){hi*=2.;inlet_wave(hi,hl,g,fl,dl);inlet_wave(hi,hr,g,fr,dr);}
 if(!isfinite(hi) || fl+fr+ur-ul<0.){r.error=1;return r;}
 double h=fmax(fmin(guess,hi),1e-30),residual=0.;
 for(int j=0;j<70;++j){
  inlet_wave(h,hl,g,fl,dl);inlet_wave(h,hr,g,fr,dr);residual=fl+fr+ur-ul;r.iterations=j+1;
  if(fabs(residual)<=4e-14*fmax(1.,cl+cr+fabs(ul)+fabs(ur)))break;
  if(residual<0.)lo=h;else hi=h;
  double next=h-residual/(dl+dr);if(!(next>lo && next<hi))next=lo+.5*(hi-lo);h=next;
 }
 inlet_wave(h,hl,g,fl,dl);inlet_wave(h,hr,g,fr,dr);residual=fl+fr+ur-ul;
 if(!isfinite(h) || fabs(residual)>2e-11*fmax(1.,cl+cr+fabs(ul)+fabs(ur))){r.error=2;return r;}
 double u=.5*(ul+ur+fr-fl),c=sqrt(g*h);
 r.h=h;r.u=u;r.derivative=fmax(0.,(u+h*dr)/(dl+dr));
 if(u>=0.){
  if(h>hl){double shock=ul-sqrt(.5*g*h*(h+hl)/hl);if(shock>=0.){r.h=hl;r.u=ul;r.derivative=hl;}}
  else if(ul-cl>=0.){r.h=hl;r.u=ul;r.derivative=hl;}
  else if(u-c>0.){r.u=(ul+2.*cl)/3.;r.h=r.u*r.u/g;r.derivative=r.h;}
 }else{
  if(h>hr){double shock=ur+sqrt(.5*g*h*(h+hr)/hr);if(shock<=0.){r.h=hr;r.u=ur;r.derivative=0.;}}
  else if(ur+cr<=0.){r.h=hr;r.u=ur;r.derivative=0.;}
  else if(u+c<0.){r.u=(ur-2.*cr)/3.;r.h=r.u*r.u/g;r.derivative=0.;}
 }
 return r;
}

LOCAL void inlet_rate(InletWork&w,int b,int nf,int n,double g,double dry,
 const int*offset,const int*faces,const int*owner,const double*q,const double*z,
 const double*bed,const double*normal,const double*length,double requested){
 LANES {
  double rate=0.,derivative=0.,area=0.,inner=0.,error=0.,gross=0.;
  for(int k=offset[b]+lane;k<offset[b+1];k+=128){int f=faces[k],o=owner[f];
   double h0=q[o],hi=h0>dry?fmax(h0+z[o]-bed[f],0.):0.;if(hi<=dry)hi=0.;
   double ui=h0>dry?-(q[n+o]*normal[f]+q[2*n+o]*normal[nf+f])/h0:0.;
   double hr=fmax(requested-bed[f],0.);if(hr<=dry)hr=0.;
   InletRiemann r=inlet_exact(hr,w.velocity,hi,ui,g);
   rate+=r.h*r.u*length[f];derivative+=r.derivative*length[f];area+=r.h*length[f];
   gross+=fabs(r.h*r.u*length[f]);
   inner=fmax(inner,r.iterations);error=fmax(error,r.error);
  }
  w.part[0][lane]=rate;w.part[1][lane]=derivative;w.part[2][lane]=area;w.part[3][lane]=inner;w.part[4][lane]=error;
  w.part[5][lane]=gross;
 }
 SYNC;
 LEADER {
  w.rate=w.derivative=w.area=w.gross=0.;
  for(int lane=0;lane<128;++lane){w.rate+=w.part[0][lane];w.derivative+=w.part[1][lane];w.area+=w.part[2][lane];
   w.gross+=w.part[5][lane];w.max_inner=int(fmax(w.max_inner,w.part[3][lane]));if(w.part[4][lane])w.error=5;}
 }
 SYNC;
}

LOCAL void inlet_group(InletWork&w,int b,int nf,int n,int nb,double gravity,double dry,double speed_limit,
 const int*offset,const int*faces,const int*owner,const double*q,const double*z,const double*bed,
 const double*normal,const double*length,const double*forcing,int*branch,double*invariant,
 double*stage,double*target_stage,double*metrics,double*exterior_velocity,int*iterations,double*mass_balance,
 double*trace,double*flux,int*face_status,int*section_status){
 const double Q=forcing[b],requested=forcing[nb+b];
 LEADER {w.error=0;w.iterations=w.max_inner=0;w.rate=w.gross=w.area=w.eta=0.;}
 LANES {
  double area=0.,scale=1.;
  for(int k=offset[b]+lane;k<offset[b+1];k+=128){int f=faces[k],o=owner[f];face_status[f]=0;branch[f]=0;
   double h0=q[o],hi=h0>dry?fmax(h0+z[o]-bed[f],0.):0.;if(hi<=dry)hi=0.;
   double ui=h0>dry?-(q[n+o]*normal[f]+q[2*n+o]*normal[nf+f])/h0:0.;
   invariant[f]=ui-2.*sqrt(gravity*hi);
   double hr=fmax(requested-bed[f],0.);if(hr>dry)area+=hr*length[f];
   scale=fmax(scale,fabs(ui)+2.*sqrt(gravity*hi)+2.*sqrt(gravity*hr));
  }
  w.part[0][lane]=area;w.part[1][lane]=scale;
 }
 SYNC;
 LEADER {
  w.reference_area=0.;w.scale=1.;
  for(int lane=0;lane<128;++lane){w.reference_area+=w.part[0][lane];w.scale=fmax(w.scale,w.part[1][lane]);}
  if(!isfinite(Q) || Q<0. || !isfinite(requested) || (Q>0. && w.reference_area<=0.))w.error=1;
  w.velocity=w.reference_area>0.?Q/w.reference_area:0.;w.scale=fmax(w.scale,1.+fabs(w.velocity));
 }
 SYNC;
 if(Q>0. && !w.error){
  LEADER {w.lo=w.velocity-w.scale;w.hi=w.velocity+w.scale;w.velocity=w.lo;}
  SYNC;inlet_rate(w,b,nf,n,gravity,dry,offset,faces,owner,q,z,bed,normal,length,requested);
  for(int j=0;j<60 && w.rate>Q && !w.error;++j){LEADER {w.lo-=w.scale;w.scale*=2.;w.velocity=w.lo;} SYNC;
   inlet_rate(w,b,nf,n,gravity,dry,offset,faces,owner,q,z,bed,normal,length,requested);}
  if(w.rate>Q){LEADER {w.error=2;} SYNC;}
  LEADER {w.velocity=w.hi;}
  SYNC;inlet_rate(w,b,nf,n,gravity,dry,offset,faces,owner,q,z,bed,normal,length,requested);
  for(int j=0;j<60 && w.rate<Q && !w.error;++j){LEADER {w.hi+=w.scale;w.scale*=2.;w.velocity=w.hi;} SYNC;
   inlet_rate(w,b,nf,n,gravity,dry,offset,faces,owner,q,z,bed,normal,length,requested);}
  if(w.rate<Q){LEADER {w.error=2;} SYNC;}
  LEADER {w.velocity=fmin(w.hi,fmax(w.lo,Q/w.reference_area));}
  SYNC;
  for(int j=0;j<80 && !w.error;++j){
   inlet_rate(w,b,nf,n,gravity,dry,offset,faces,owner,q,z,bed,normal,length,requested);
   LEADER {w.iterations=j+1;}
   SYNC;
   // Net Q may be much smaller than signed face throughflow. Include an
   // explicit double-precision summation/local-root budget, never a Q floor.
   if(fabs(w.rate-Q)<=3e-13*Q+256.*2.220446049250313e-16*w.gross)break;
   LEADER {
    if(w.rate<Q)w.lo=w.velocity;else w.hi=w.velocity;
    double next=w.derivative>0.?w.velocity+(Q-w.rate)/w.derivative:w.lo+.5*(w.hi-w.lo);
    if(!(next>w.lo && next<w.hi))next=w.lo+.5*(w.hi-w.lo);w.velocity=next;
   }
   SYNC;
  }
  inlet_rate(w,b,nf,n,gravity,dry,offset,faces,owner,q,z,bed,normal,length,requested);
  if(!isfinite(w.rate) || fabs(w.rate-Q)>1e-10*Q+1024.*2.220446049250313e-16*w.gross){LEADER {w.error=3;} SYNC;}
 }
 LEADER {section_status[b]=w.error;target_stage[b]=requested;exterior_velocity[b]=w.velocity;
  iterations[b]=w.iterations;iterations[nb+b]=w.max_inner;}
 SYNC;if(w.error)return;
 LANES {
  double area=0.,eta=0.,subcritical=0.,incoming=0.,rate=0.,gross=0.;
  for(int k=offset[b]+lane;k<offset[b+1];k+=128){int f=faces[k],o=owner[f];double nx=normal[f],ny=normal[nf+f];
   double h0=q[o],hi=h0>dry?fmax(h0+z[o]-bed[f],0.):0.;if(hi<=dry)hi=0.;
   double ui=h0>dry?-(q[n+o]*nx+q[2*n+o]*ny)/h0:0.;
   double vt=h0>dry?(-q[n+o]*ny+q[2*n+o]*nx)/h0:0.;
   double hr=fmax(requested-bed[f],0.);if(hr<=dry)hr=0.;
   InletRiemann r=Q==0.?inlet_exact(h0,-ui,h0,ui,gravity):inlet_exact(hr,w.velocity,hi,ui,gravity);
   if(Q==0.)r.u=0.;
   if(r.error || !isfinite(r.h) || !isfinite(r.u) || fabs(r.u)>speed_limit){face_status[f]=3;continue;}
   double wave=sqrt(gravity*r.h),water=-r.h*r.u,tangent=r.u<0.?vt:0.;
   int kind=(Q==0. || r.h==0.)?0:(r.u<0.?4:(fabs(r.u-wave)<=1e-10*fmax(1.,wave)?3:(r.u>wave?2:1)));
   branch[f]=kind;subcritical+=kind==1;incoming+=kind==2 || kind==3;
   trace[f]=r.h;trace[nf+f]=water*nx-r.h*tangent*ny;trace[2*nf+f]=water*ny+r.h*tangent*nx;
   double pressure=Q==0.?0.:.5*gravity*(h0*h0-hi*hi);
   double momentum=r.h*r.u*r.u+.5*gravity*r.h*r.h+pressure;
   flux[f]=water;flux[nf+f]=momentum*nx-water*tangent*ny;flux[2*nf+f]=momentum*ny+water*tangent*nx;
   rate+=r.h*r.u*length[f];gross+=fabs(r.h*r.u*length[f]);area+=r.h*length[f];eta+=r.h*length[f]*(r.h+bed[f]);
  }
  w.part[0][lane]=rate;w.part[1][lane]=area;w.part[2][lane]=subcritical;w.part[3][lane]=incoming;w.part[4][lane]=eta;
  w.part[5][lane]=gross;
 }
 SYNC;
 LEADER {
  double eta=0.,gross=0.;for(int j=0;j<4;++j)metrics[j*nb+b]=0.;
  for(int lane=0;lane<128;++lane){for(int j=0;j<4;++j)metrics[j*nb+b]+=w.part[j][lane];eta+=w.part[4][lane];gross+=w.part[5][lane];}
  stage[b]=metrics[nb+b]>0.?eta/metrics[nb+b]:requested;
  mass_balance[b]=metrics[b]-Q;mass_balance[nb+b]=gross;
  mass_balance[2*nb+b]=1e-10*Q+1024.*2.220446049250313e-16*gross;
 }
 SYNC;
}

ENTRY void exterior_compatible_inlet(int nf,int n,int nb,double gravity,double dry,double speed_limit,
 const int*control,const int*offset,const int*faces,const int*owner,const double*q,const double*z,const double*bed,
 const double*normal,const double*length,const double*forcing,int*branch,double*invariant,
 double*stage,double*target_stage,double*metrics,double*exterior_velocity,int*iterations,double*mass_balance,
 double*trace,double*flux,int*face_status,int*section_status){
#ifdef __CUDACC__
 int b=blockIdx.x;if(b>=nb || !control[b])return;__shared__ InletWork work;
 inlet_group(work,b,nf,n,nb,gravity,dry,speed_limit,offset,faces,owner,q,z,bed,normal,length,forcing,
  branch,invariant,stage,target_stage,metrics,exterior_velocity,iterations,mass_balance,trace,flux,face_status,section_status);
#else
 for(int b=0;b<nb;++b){if(!control[b])continue;InletWork work;
  #pragma omp parallel shared(work)
  {inlet_group(work,b,nf,n,nb,gravity,dry,speed_limit,offset,faces,owner,q,z,bed,normal,length,forcing,
   branch,invariant,stage,target_stage,metrics,exterior_velocity,iterations,mass_balance,trace,flux,face_status,section_status);}
 }
#endif
}
