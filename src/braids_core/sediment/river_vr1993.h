// One current-only FLOW 2DH van Rijn 1993 producer for OpenMP and CUDA.
// This is a constitutive operator, NOT yet a coupled/public river binding.
// f rows: h,u,v,zvel,z0cur,z0rou,d50,d90,rhos,rhow,g,nu,ws,Dstar,
//         taucr_class,taucr_bed,fraction,mud_fraction,DSS,tauadd,kappa,eps,camax,sigmol.
// p: IopSus,AksFac,RWave,RDC,RDW,IopKCW,EpsPar,BetaM,iTauCr.
// o: raw mass bed x/y, fraction-owned ceq, TAURAT, fraction-owned caks,
//    aks,new DSS,ustarc,muc,tauc,d(bed magnitude)/d(speed),bed magnitude.
#ifndef RIVER_VR1993_EMBEDDED
#ifdef __CUDACC__
#define ENTRY extern "C" __global__
#define LOCAL __device__ __forceinline__
#define LOOP(N) for(int i=blockDim.x*blockIdx.x+threadIdx.x;i<N;i+=blockDim.x*gridDim.x)
#else
#include <cmath>
#include <omp.h>
using std::isfinite;
#define ENTRY extern "C" __declspec(dllexport)
#define LOCAL inline
#define LOOP(N) _Pragma("omp parallel for schedule(static)") for(int i=0;i<N;++i)
extern "C" __declspec(dllexport) void mobile_threads(int n){omp_set_dynamic(0);omp_set_num_threads(n);}
#endif
#endif

LOCAL int river_vr1993_point(const double*f,int n,int i,const double*p,double*r){
 for(int k=0;k<12;++k)r[k]=0.;
 for(int k=0;k<24;++k)if(!isfinite(f[k*n+i]))return 1;
 const double h=f[i],u=f[n+i],v=f[2*n+i],zvel=f[3*n+i],
  z0=f[4*n+i],zrough=f[5*n+i],d=f[6*n+i],d90=f[7*n+i],
  rhos=f[8*n+i],rho=f[9*n+i],g=f[10*n+i],nu=f[11*n+i],
  ws=f[12*n+i],ds=f[13*n+i],tau0=f[14*n+i],taubed=f[15*n+i],
  frac=f[16*n+i],mud=f[17*n+i],dss=f[18*n+i],tauadd=f[19*n+i],
  kappa=f[20*n+i],eps=f[21*n+i],camax=f[22*n+i],sigmol=f[23*n+i];
 if(!(h>0.&&zvel>0.&&z0>0.&&zrough>0.&&d>0.&&d90>0.&&rhos>rho&&rho>0.
    &&g>0.&&nu>0.&&ws>=0.&&ds>0.&&tau0>=0.&&taubed>=0.
    &&frac>=0.&&frac<=1.&&mud>=0.&&mud<=1.&&dss>0.&&tauadd>=0.
    &&kappa>0.&&eps>0.&&camax>0.&&camax<=1.&&sigmol>0.))return 1;
 const double critical=p[8]==1.?tau0*pow(1.+mud,p[7]):taubed;
 const double rc=p[5]==1.?30.*z0:p[3];
 if(!(critical>0.&&12.*h>rc&&12.*h>3.*d90))return 2;
 const double speed=hypot(u,v),a=kappa/log1p(zvel/z0);
 double ust=a*speed,tauc=rho*ust*ust;
 // bedbc1993: additional stress is a quadrature addition, not a sum.
 if(tauadd>0.){tauc=hypot(tauc,tauadd);ust=sqrt(tauc/rho);}
 const double f1c=.24/pow(log10(12.*h/(3.*d90)),2.);
 const double fc=.24/pow(log10(12.*h/rc),2.),muc=f1c/fc;
 const double ratio=muc*tauc/critical,stage=fmax(0.,ratio-1.);
 const double aks=fmin(fmax(p[1]*rc,.01*h),.2*h);
 double caks=stage>eps?fmin(.015*d*pow(stage,1.5)/(aks*pow(ds,.3)),camax):0.;
 double nextdss=dss;
 if(p[0]==1.)nextdss=d*(stage<=1.?.64:stage>=25.?1.:1.+.015*(stage-25.));
 // tram1 applies surface fraction BEFORE the concentration threshold.
 caks*=frac;
 const double beta=ust>eps?fmin(fmax(1.+2.*pow(ws/ust,2.),1.),1.5):1.;
 const double epsmax=.25*kappa*beta*ust*h;
 // FLOW erosed DATA arrays are default-real literals promoted to fp.
 const double sig[20]={-.0874f,-.2472f,-.3797f,-.4897f,-.5809f,-.6565f,-.7193f,-.7713f,-.8145f,-.8503f,
  -.8800f,-.9046f,-.9250f,-.9419f,-.9560f,-.9676f,-.9773f,-.9854f,-.9920f,-.9975f};
 const double thick[20]={.1747f,.1449f,.1202f,.0997f,.0827f,.0686f,.0569f,.0472f,.0391f,.0325f,
  .0269f,.0223f,.0185f,.0154f,.0127f,.0106f,.0088f,.0073f,.0060f,.0050f};
 double dif[20],profile[20];
 for(int k=0;k<20;++k){
  double zi=(1.+sig[k]-.5*thick[k])*h;
  dif[k]=zi>=.5*h?epsmax:epsmax*(1.-pow(1.-2.*zi/h,2.));profile[k]=0.;
 }
 int bottom=0;
 for(int k=18;k>=0;--k)if((1.+sig[k]-.5*thick[k])*h>=aks){bottom=k;break;}
 const double background=nu/sigmol;
 if(caks>1.e-6){
  double dz=h*(1.+sig[bottom])-aks,df=fmax(dif[bottom]+background,.1*ws*dz);
  profile[bottom]=caks/(1.+dz*ws/df)*rhos;
  for(int k=bottom-1;k>=0;--k){
   dz=h*(sig[k]-sig[k+1]);df=fmax(dif[k]+background,.1*ws*dz);
   profile[k]=profile[k+1]/(1.+dz*ws/df);
  }
  for(int k=bottom+1;k<20;++k)profile[k]=profile[k-1];
 }
 double avgu=0.,avgcu=0.;
 for(int k=0;k<20;++k){
  double ul=log1p((1.+sig[k])*h/zrough);
  avgu+=ul*thick[k];avgcu+=ul*profile[k]*thick[k];
 }
 const double transport_stage=fmax(1.e-10,ratio-1.);
 const double constant=.5*rhos*d*sqrt(muc)*pow(ds,-.3);
 const double mag=constant*ust*transport_stage;
 // Analytic frozen-depth/material derivative, including tauadd quadrature.
 const double dust=ust>0.?rho*pow(a,4.)*pow(speed,3.)/(ust*tauc):0.;
 const double dratio=ust>0.?2.*muc*rho*ust*dust/critical:0.;
 const double dmag=constant*(dust*transport_stage+(ratio-1.>1.e-10?ust*dratio:0.));
 r[0]=speed>eps?mag*u/speed:0.;r[1]=speed>eps?mag*v/speed:0.;
 r[2]=avgcu/fmax(avgu,eps);r[3]=ratio;r[4]=caks;r[5]=aks;r[6]=nextdss;
 r[7]=ust;r[8]=muc;r[9]=tauc;r[10]=speed>eps?dmag:0.;r[11]=speed>eps?mag:0.;
 for(int k=0;k<12;++k)if(!isfinite(r[k]))return 3;
 return 0;
}
#ifndef RIVER_VR1993_EMBEDDED
ENTRY void vr1993_producer(int n,const double*f,const double*p,double*out,int*status){
 LOOP(n){double row[12];status[i]=river_vr1993_point(f,n,i,p,row);
  for(int k=0;k<12;++k)out[k*n+i]=status[i]?0.:row[k];
 }
}
#endif
