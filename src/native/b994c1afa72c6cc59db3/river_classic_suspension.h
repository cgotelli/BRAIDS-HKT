// S6 current-only equilibrium closures, with explicitly named 2DH extensions.
// Van Rijn 1984 II eq11-13,19,22,33,34,37,38. ZF1994 A=.331,n=1.75,Cm=.46.
// CL2008 eq4,9,11,12,16,17. Threshold ratio cr/theta follows authors' 2007
// eq185 and Camenen's 2018 Alpine-river application eq5. The supplied 2008
// PDF eq16 prints theta/theta_cr; the named bundle uses the corrected ratio.
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
LOCAL double classic_ws(double d,double r,double nu){
 if(d<.0001)return r*9.81*d*d/(18.*nu);
 if(d<=.001){double x=.01*r*9.81*d*d*d/(nu*nu);return 10.*nu/d*x/(sqrt(1.+x)+1.);}
 return 1.1*sqrt(r*9.81*d);
}
LOCAL double classic_zf_skin(double h,double u,double ut,double d){
 // ZF1994 eq9-11, kN=2.5D50. Dprime/h=(us/ut)^2.
 // Solve in log(us/ut); bound skin shear by local total physical shear.
 if(ut<=0.||u<=0.)return 0.;
 double c=6.+2.5*log(h/(2.5*d)),r=u/ut;
 if(r>=c)return ut;
 double lo=-c/5.,hi=0.,t=0.;
 for(int k=0;k<32;++k){
  double e=exp(t),f=e*(c+5.*t)-r;
  if(f>0.)hi=t;else lo=t;
  if(fabs(f)<=2.e-14*fmax(r,1.))break;
  double candidate=t-f/(e*(c+5.*t+5.));
  t=(candidate>lo&&candidate<hi)?candidate:.5*(lo+hi);
 }
 return ut*exp(t);
}
LOCAL double classic_threshold(double d){
 if(d<=4.)return .24/d;
 if(d<=10.)return .14*pow(d,-.64);
 if(d<=20.)return .04*pow(d,-.1);
 if(d<=150.)return .013*pow(d,.29);
 return .055;
}
LOCAL double classic_profile(double alpha,double p,int parabolic_constant,const double*quad){
 // Integral only ABOVE a. Logit coordinates resolve steep near-bed profiles.
 double end=parabolic_constant?log((1.-alpha)/alpha):50.;
 double top=fmin(end,50./fmax(p,1.));double sum=0.;
 for(int k=0;k<64;++k){double t=top*quad[k],b=(1.-alpha)/alpha*exp(-t);
  sum+=quad[64+k]*exp(-p*t)*b/((1.+b)*(1.+b));}
 sum*=top;
 if(parabolic_constant){double c=exp(-p*end);sum+=p>1.e-10?c*(-expm1(-2.*p))/(4.*p):.5*c;}
 return fmax(1.e-300,fmin(1.-alpha,sum));
}
LOCAL void classic_values(int method,double h,double u,double z0,double ds,double d50,double d90,double r,double temp,const double*quad,double*v){
 double nu=4.e-5/(20.+temp),ws=classic_ws(ds,r,nu),ut=fabs(u)*.41/log1p(.36800000071525574*h/z0);
 // ZF1994 used Rubey (1933), not Van Rijn, for its suspended median.
 if(method==1){double visc=6.*nu/ds;ws=(2./3.)*r*9.81*ds/(sqrt(visc*visc+(2./3.)*r*9.81*ds)+visc);}
 double dstar=d50*cbrt(r*9.81/(nu*nu)),us=ut,ca=0.,j=1.,a=0.,p=0.,mix=1.,theta=0.,crit=0.;
 bool shallow=h<20.*fmax(ds,d50);
 if(method==0){
  a=fmax(30.*z0,.01*h);shallow=shallow||a>=.5*h||h<=d90;
  if(!shallow){
   us=fmin(ut,fabs(u)*sqrt(9.81)/(18.*log10(12.*h/(3.*d90))));
   theta=us*us/(r*9.81*d50);crit=classic_threshold(dstar);
   ca=fmin(.65,.015*d50/a*pow(fmax(theta/crit-1.,0.),1.5)/pow(dstar,.3));
   double ratio=ut>0.?ws/ut:1.e100,bounded=fmin(ratio,1.);
   mix=1.+2.*bounded*bounded;
   p=ratio/(.4*mix)+2.5*pow(bounded,.8)*pow(ca/.65,.4);
   j=classic_profile(a/h,p,1,quad);
  }
 }else if(method==1){
  a=2.*d50;shallow=shallow||a>=.5*h;
  if(!shallow){
   us=classic_zf_skin(h,fabs(u),ut,d50);
   theta=us*us/(r*9.81*d50);crit=.045;double excess=fmax(theta-crit,0.);
   if(excess>0.){double lx=log(.331)+1.75*log(excess);ca=lx>0.?1./(1./.46+exp(-lx)):exp(lx)/(1.+exp(lx)/.46);}
   p=ut>0.?ws/(.4*ut):1.e100;j=classic_profile(a/h,p,0,quad);
  }
 }else{
  theta=ut*ut/(r*9.81*d50);crit=.30/(1.+1.2*dstar)+.055*(-expm1(-.02*dstar));
  if(!shallow){
   if(theta>0.)ca=fmin(.65,.0015*exp(-.2*dstar)*theta*exp(-4.5*crit/theta));
   double ratio=ut>0.?ws/ut:1.e100;
   double s=sin(1.57079632679489661923*(ratio<=1.?ratio:1./ratio));
   mix=ratio<=1.?.4+3.5*s*s:1.+2.9*s*s;
   p=6.*ratio/(.4*mix);j=p>1.e-8?(-expm1(-p))/p:1.-.5*p+p*p/6.;
  }
 }
 if(shallow){ca=0.;j=1.;}
 v[0]=ws;v[1]=ca;v[2]=j;v[3]=ut;v[4]=us;v[5]=a;v[6]=p;v[7]=mix;v[8]=theta;v[9]=crit;
}
ENTRY void classic_probe(int n,int method,double ds,double d50,double d90,double r,double temp,const double*h,const double*u,const double*z0,const double*quad,double*out){
 LOOP(n){double v[10];classic_values(method,h[i],u[i],z0[i],ds,d50,d90,r,temp,quad,v);for(int k=0;k<10;++k)out[k*n+i]=v[k];}
}
ENTRY void classic_exchange(int n,double dt,double dry,int method,double ds,double d50,double d90,double rho,double r,double temp,const double*q,const double*z0,const double*mass,const double*stock,const double*quad,double*next,double*delta,double*diag){
 LOOP(n){double h=q[i];
  if(h<=dry){next[i]=0.;delta[i]=mass[i];diag[i]=diag[n+i]=diag[2*n+i]=0.;continue;}
  double v[10];classic_values(method,h,hypot(q[n+i],q[2*n+i])/h,z0[i],ds,d50,d90,r,temp,quad,v);
  double rate=v[0]/(h*v[2]),eq=rho*v[1]*h*v[2];
  double change=(eq-mass[i])*(-expm1(-dt*rate));
  change=fmax(-mass[i],fmin(change,fmax(stock[i],0.)*(1.-16.*2.22044604925031308085e-16)));
  next[i]=mass[i]+change;delta[i]=-change;diag[i]=v[0];diag[n+i]=rho*v[1]*v[2];diag[2*n+i]=rate;
 }
}
