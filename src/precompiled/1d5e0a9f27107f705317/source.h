// split_capacity_bed_diagonal_v1
// Research bed-row f-wave splitting of a frozen-coefficient SWE--Exner
// Jacobian. Augmented hydraulic flux is NOT replaced. Physical EH/KF unchanged.
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
LOCAL double value(int c,int i,int n,const double*q,const double*z){
 return c==0?q[i]:(c==3?z[i]:q[c*n+i]/q[i]);
}
// Current-only FLOW scalar constitutive laws (1/2/4/7/13/14/16/17/18) in solid volume.
// One body serves cell capacity, exchange and the frozen-coefficient bed row.
// Explicit class relative density; g=9.81. Material is checked by the caller.
#ifdef __CUDACC__
#define RIVER_LOCAL __device__
#else
#define RIVER_LOCAL inline
using std::isfinite;
using std::nan;
#endif
RIVER_LOCAL double river_shields_curve(double dstar){
 if(dstar<=4.)return .240/dstar;
 if(dstar<=10.)return .140/pow(dstar,.64);
 if(dstar<=20.)return .040/pow(dstar,.10);
 if(dstar<=150.)return .013*pow(dstar,.29);
 return .055;
}
// Noncohesive FLOW tranb7, tran9t and trab14. Constants written as float
// reproduce the unsuffixed default-real literals in the reference routines.
// The analytic bed derivative freezes depth, material, roughness and hiding;
// supplied RP_USTAR scales with effective speed at fixed roughness.
RIVER_LOCAL void river_additional_scalar(const double*p,double speed,double chezy,double h,
 double d,double d90,double hide,double shear,double&bed,double&sus,double&der){
 const int law=int(p[0]);const double relative=p[14],acal=p[1],nu=p[15];
 bed=0.;sus=0.;der=0.;
 if(law==7){
  const double a=p[16],ws=p[17];
  if(h/a<double(1.33f)||speed<double(1.e-3f))return;
  double ds=d*pow(relative*9.81/nu/nu,double(1.f/3.f));
  double lr=log10(12.*h/a),mu=pow(lr/log10(12.*h/3./d90),2.);
  double fc=double(.24f)*pow(lr,-2.),tau=.125*fc*speed*speed;
  // All public fractions are noncohesive: mud fraction is explicitly zero.
  double critical=relative*9.81*d*river_shields_curve(ds);
  double raw=(mu*tau-critical)/critical,t=fmax(raw,double(.000001f));
  double ca=double(.015f)*acal*d/a*pow(t,1.5)/pow(ds,double(.3f));
  double ust=sqrt(.125*fc)*speed,ratio=ws/ust;
  double beta=fmin(1.+2.*ratio*ratio,1.5);
  double psi=2.5*pow(ratio,double(.8f))*pow(ca/double(.65f),double(.4f));
  double zc=fmin(ws/double(.41f)/ust/beta+psi,20.);
  double ah=a/h,integral;
  if(fabs(zc-double(1.2f))>double(1.e-4f))
   integral=(pow(ah,zc)-pow(ah,double(1.2f)))/pow(1.-ah,zc)/(double(1.2f)-zc);
  else integral=-pow(ah/(1.-ah),double(1.2f))*log(ah);
  sus=integral*speed*h*ca;
  const double exponent=t<3.?double(2.1f):1.5;
  const double coefficient=t<3.?double(.053f):double(.100f);
  bed=coefficient*sqrt(relative)*sqrt(9.81)*pow(d,1.5)*pow(ds,-double(.3f))*pow(t,exponent);
  // ACal belongs to ca (suspension), not sbot, in stock tranb7.
  if(raw>double(.000001f))der=bed*exponent*2.*(raw+1.)/(t*speed);
 }else if(law==13){
  if(speed==0.||chezy<=0.)return;
  double total=double(.05f)*speed*speed*pow(shear,3.)/d/pow(9.81*relative,2.);
  double cg=6.+2.5*log(h/(2.5*d90));
  double theta=9.81*pow(speed/cg,2.)/relative/9.81/d;
  double ds=d*pow(relative*9.81/(nu*nu),double(.33333333f));
  double critical=river_shields_curve(ds),ex=theta-critical;
  if(ex>0.){
   double a=double(.2668f),f,df;
   // Algebraically identical resistance, with a finite threshold derivative.
   if(ex<=a){double r=ex/a,w=1.+pow(r,4.);f=r/pow(w,.25);df=1./(a*pow(w,1.25));}
   else{double r=a/ex,w=1.+pow(r,4.);f=1./pow(w,.25);df=pow(r,4.)/(ex*pow(w,1.25));}
   double root=sqrt(critical/theta),g=1.-double(.7f)*root;
   double scale=acal*double(4.8695f)*d*shear;
   bed=scale*f*g;
   der=scale/speed*(f*g+df*2.*theta*g+f*double(.7f)*root);
  }
  // Preserve the native signed residual. A negative suspended equilibrium
  // is rejected by the live exchange owner, not silently clipped here.
  sus=total-bed;
 }else if(law==14){
  if(speed<double(1.e-6f)||chezy<double(1.e-6f))return;
  double sgd=relative*9.81*d,ust=sqrt(9.81)*speed/chezy;
  double theta=ust*ust/sgd,r=hide*p[6]/theta;
  if(r>=1.)return;
  double m=p[19],pp=p[20],qq=p[21],root=sqrt(r);
  double t=pow(theta,m),u=pow(1.-r,pp),v=pow(1.-root,qq);
  double scale=acal*sqrt(sgd*d*d);
  bed=scale*t*u*v;
  if(m!=0.)der=2.*m*bed/speed;
  if(pp!=0.)der+=scale*t*pp*pow(1.-r,pp-1.)*v*2.*r/speed;
  if(qq!=0.)der+=scale*t*u*qq*pow(1.-root,qq-1.)*root/speed;
 }
}
// FLOW trabwc/trabwc2/trabg. TAUB is hydraulic stress, NOT the stress
// reconstructed from the floored sediment Chezy. Dg/DgSD/sand describe the
// current surface mixture; the donor's fraction is applied once downstream.
RIVER_LOCAL void river_intrinsic_scalar(const double*p,double speed,double chezy,double d,
 double dg,double dgsd,double sand,double taub,double&bed,double&sus,double&der,
 double stress_elasticity=2.){
 bed=0.;sus=0.;der=0.;
 if(chezy<1.e-6||dg<1.e-6||speed==0.)return;
 const int law=int(p[0]);
 const double relative=p[14],water=p[13]/(1.+relative);
 double b,critical,scale=1.;
 if(law==16){
  b=.67/(1.+exp(1.5-d/dg));
  critical=.021+.015*exp(-20.*sand);scale=p[1];
 }else{
  double sigma=dgsd==0.?0.:pow(log(dgsd)/log(2.),2.);
  double exponent=(law==17?10.1:7.1)*sigma-(law==17?14.14:11.786);
  // Stable logistic evaluation, including FLOW's spatial-D50 DgSD=0 limit.
  double inv=dgsd==0.?0.:(exponent>0.?exp(-exponent)/(1.+exp(-exponent)):1./(1.+exp(exponent)));
  critical=p[22]+(law==17?.015:.022)*inv;
  b=(1.-p[23])/(1.+exp(law==17?1.5-d/dg:1.9-d/(3.*dg)));
 }
 double tauri=critical*(p[13]-water)*9.81*dg*pow(d/dg,b);
 if(!(tauri>0.)||!isfinite(tauri)){bed=nan("");der=bed;return;}
 double phi=taub/tauri;
 if(phi==0.)return;
 double w=phi<1.35?.002*pow(phi,7.5):14.*pow(1.-.894/sqrt(phi),4.5);
 bed=scale*w*pow(sqrt(9.81)*speed/chezy,3.)/(relative*9.81);
 // Freeze roughness/mixture and the physical/effective speed ratio, but not
 // the Reynolds dependence of the selected skin stress. Retain the original
 // quadratic arithmetic exactly for total drag and the rough-skin branch.
 der=stress_elasticity==2.?bed/speed*(phi<1.35?18.:3.+4.5*.894/(sqrt(phi)-.894)):
     bed/speed*(3.+stress_elasticity*(phi<1.35?7.5:2.25*.894/(sqrt(phi)-.894)));
}
RIVER_LOCAL void river_scalar_law(const double*parameters,double speed,double chezy,double depth,
 double diameter,double d90,double hide,double&bed,double&sus,double&dspeed,double shear_velocity=0.,
 double dg=0.,double dgsd=0.,double sand=0.,double taub=0.,double stress_elasticity=2.){
 const int law=int(parameters[0]);
 if(law==104){
  // Camenen & Larson (2005), steady current. Authors' CR-07-1 Eq81.
  // Eq49 critical Shields is prepared once per D50/density/viscosity in slot6.
  // No MPM excess-stress cutoff, ripple factor, hiding or suspended channel.
  bed=0.;sus=0.;dspeed=0.;
  if(!(speed>0.)||!(depth>0.)||parameters[1]==0.)return;
  double theta=speed*speed/(chezy*chezy*parameters[14]*diameter);
  if(!(theta>0.))return;
  double attenuation=4.5*parameters[6]/theta;
  if(!isfinite(attenuation))return; // exact zero in FP64 at vanishing speed
  double logbed=log(12.*parameters[1])+log(diameter)+.5*log(9.81*parameters[14]*diameter)
                +1.5*log(theta)-attenuation;
  bed=exp(logbed);
  // Log evaluation preserves the derivative even when bed itself underflows.
  dspeed=exp(logbed-log(speed)+log(3.+2.*attenuation));
  return;
 }
 if(law==103){
  // Recking (2010), WRR W03518 Eqs 4,12-14. One grain => D84/D50=1.
  // The reach slope is explicit, NOT a local instantaneous friction slope.
  bed=0.;sus=0.;dspeed=0.;
  if(!(speed>0.)||!(depth>0.)||parameters[1]==0.)return;
  double theta=speed*speed/(chezy*chezy*parameters[14]*diameter);
  if(!(theta>0.))return;
  double critical=1.32*parameters[3]+.037;
  double low=log(.0005)+6.5*log(theta/critical);
  double high=log(14.)+2.45*log(theta);
  // Exact intersection of Eq12 branches (Eq13 before printed rounding).
  // This avoids a spurious tiny capacity jump from rounded 12.53/1.605.
  bool partial=low<high;
  bed=parameters[1]*diameter*sqrt(9.81*parameters[14]*diameter)*exp(partial?low:high);
  dspeed=2.*(partial?6.5:2.45)*bed/speed;
  return;
 }
 if(law==102){
  // Ancey & Recking (2023), Earth-Science Reviews 246, Eq. (42).
  // Surface d50; physical hydraulic stress; solid-volume transport per width.
  // Evaluate the smooth master curve in logarithms, including its low tail.
  bed=0.;sus=0.;dspeed=0.;
  if(!(speed>0.)||!(depth>0.)||parameters[1]==0.)return;
  double theta=speed*speed/(chezy*chezy*parameters[14]*diameter);
  if(!(theta>0.))return;
  double x=1.5*log(theta/.40),y=8.*log(theta/.078);
  double lx=fmax(x,0.)+log1p(exp(-fabs(x)));
  double ly=fmax(y,0.)+log1p(exp(-fabs(y)));
  double phi=exp(16.*log(10.*theta)-(8./9.)*lx-(13./8.)*ly);
  double sx=x>=0.?1./(1.+exp(-x)):exp(x)/(1.+exp(x));
  double sy=y>=0.?1./(1.+exp(-y)):exp(y)/(1.+exp(y));
  bed=parameters[1]*diameter*sqrt(9.81*parameters[14]*diameter)*phi;
  dspeed=2.*bed/speed*(16.-(4./3.)*sx-13.*sy);
  return;
 }
 if(law==101){
  // Smart & Jaeggi energy-slope form, BASEHPC 4.0.2 Eqs 1.29-1.31.
  // One grain: d90=d30=dm=diameter, grading factor exactly one.
  // Physical depth, physical velocity and hydraulic Chezy (m^0.5/s).
  // Output solid volume / width / time (m^2/s), no porosity or density factor.
  bed=0.;sus=0.;dspeed=0.;
  if(!(depth>0.)||!(speed>0.))return;
  const double relative=parameters[14],head=speed*speed/(chezy*chezy);
  const double excess=head-parameters[6]*relative*diameter;
  if(!(excess>0.))return;
  const double pref=parameters[1]/relative*pow(head/depth,.6);
  bed=pref*speed*excess;
  // Fixed depth and resistance; explicit partial wrt depth = -0.6*bed/depth.
  dspeed=pref*(2.2*excess+2.*head);
  return;
 }
 if(law>=16&&law<=18){
  river_intrinsic_scalar(parameters,speed,chezy,diameter,dg,dgsd,sand,taub,bed,sus,dspeed,stress_elasticity);
  return;
 }
 if(law==7||law==13||law==14){
  river_additional_scalar(parameters,speed,chezy,depth,diameter,d90,hide,shear_velocity,bed,sus,dspeed);
  return;
 }
 const double acal=parameters[1],share=parameters[2],powerb=parameters[3],powerc=parameters[4];
 const double ripple=parameters[5],critical=parameters[6],susacal=parameters[7];
 const double suspowerb=parameters[8],suspowerc=parameters[9],susripple=parameters[10],suscritical=parameters[11];
 const double relative=parameters[14];
 double theta=speed*speed/(chezy*chezy*relative*diameter);
 double scale=diameter*sqrt(9.81*relative*diameter);
 bed=0.;sus=0.;dspeed=0.;
 if(law==1){
  double total=.05*acal*(chezy*chezy/9.81)*scale*pow(theta,2.5);
  bed=(1.-share)*total;sus=share*total;
  if(speed>0.)dspeed=5.*bed/speed;
 }else if(law==2){
  double grain=18.*log10(12.*fmax(depth/d90,1.));
  double mu=fmin(1.,pow(chezy/grain,1.5));
  double ex=fmax(mu*theta-.047*hide,0.);
  bed=acal*8.*scale*ex*sqrt(ex);
  // Do not divide by excess: the derivative is finite at incipient motion.
  if(speed>0.)dspeed=acal*24.*scale*sqrt(ex)*mu*theta/speed;
 }else{
  double ex=fmax(ripple*theta-critical*hide,0.);
  bed=acal*scale*pow(theta,powerb)*pow(ex,powerc);
  double sx=fmax(susripple*theta-suscritical*hide,0.);
  sus=susacal*scale*pow(theta,suspowerb)*pow(sx,suspowerc);
  if(speed>0.){
   if(powerb!=0.)dspeed=2.*powerb*bed/speed;
   if(ex>0.&&powerc!=0.)
    dspeed+=acal*scale*pow(theta,powerb)*powerc*pow(ex,powerc-1.)*ripple*2.*theta/speed;
  }
 }
}

// Current-only, noncohesive compbsskin (FLOW): replace RP_TAUB only.
// Explicit sand roughness and molecular viscosity; no inferred grain multiplier,
// hydraulic drag mutation, mud/fluff mixture or wave closure.
// The analytic elasticity is d(log(tau))/d(log(physical speed)), at fixed depth.
RIVER_LOCAL int river_skin_stress(double speed,double depth,double rho,double ks,double nu,
 double&tau,double&elasticity,double&rough,double&smooth){
 tau=0.;elasticity=2.;rough=0.;smooth=0.;
 if(!isfinite(speed)||!isfinite(depth)||!isfinite(rho)||!isfinite(ks)||!isfinite(nu)
    ||speed<0.||depth<=0.||rho<=0.||ks<0.||nu<=0.)return 1;
 double z0=fmax(ks/30.,1.e-12),den=log(depth/z0)-1.;
 if(den==0.||!isfinite(den))return 1;
 double re=fmax(speed,1.e-12)*depth/nu,term=pow(re,-.08);
 rough=pow(.40/den,2.);
 smooth=1.615e-4*exp(6.*term);
 if(!isfinite(rough)||!isfinite(smooth))return 1;
 if(speed>=1.e-6){
  bool is_rough=rough>=smooth;
  tau=rho*(is_rough?rough:smooth)*speed*speed;
  elasticity=is_rough?2.:2.-.48*term;
 }
 return (!isfinite(tau)||!isfinite(elasticity))?1:0;
}
ENTRY void skin_stress_probe(int n,const double*speed,const double*depth,const double*rho,
 const double*ks,const double*nu,double*out,int*status){
 LOOP(n){status[i]=river_skin_stress(speed[i],depth[i],rho[i],ks[i],nu[i],
  out[i],out[n+i],out[2*n+i],out[3*n+i]);}
}

// Current-only FLOW trab11/trab12, solid-volume transport.
// Components are along the effective GLOBAL velocity. The capacity owns its
// direction: eqtran does not apply its scalar spiral correction to these laws.
// Gradient is d|qbed|/d(effective ux,uy) at frozen depth/material/roughness.
RIVER_LOCAL void river_current_vector_law(const double*p,double u,double v,double h,double d,
 double&bed,double&sus,double&gx,double&gy,double&theta){
 bed=0.;sus=0.;gx=0.;gy=0.;theta=0.;
 const int law=int(p[0]);const double speed=hypot(u,v),rel=p[14],nu=p[15];
 if(law==11){
  if(speed<double(1.e-6f)||h>200.||h<double(.01f))return;
  const double d90=p[24]*d,ds=pow(9.81*rel/(nu*nu),double(1.f/3.f))*d;
  const double critical=d<=double(.0005f)?
   double(.19f)*pow(d,double(.1f))*log10(4.*h/d90):
   8.5*pow(d,double(.6f))*log10(4.*h/d90);
  if(speed<=critical)return;
  const double asb=double(.005f)*h*pow(d/h/(rel*9.81*d),double(1.2f));
  const double ass=double(.012f)*d*pow(ds,-double(.6f))/pow(rel*9.81*d,double(1.2f));
  const double ex=speed-critical,power=double(2.4f),term=pow(ex,power);
  bed=p[1]*asb*speed*term;sus=p[1]*ass*speed*term;
  const double radial=p[1]*asb*(term+speed*power*pow(ex,power-1.));
  gx=radial*u/speed;gy=radial*v/speed;
  // RouZ0 and AlfaUrms are wave-related here: zero wave forcing makes their
  // contribution zero. They remain explicit saved method parameters.
 }else if(law==12){
  if(speed<1.e-6||h>200.||h<.01)return;
  const double z0=d/p[26],drag=pow(double(.41f)/(1.+log(z0/h)),2.);
  const double ds=pow(9.81*rel/(nu*nu),1./3.)*d;
  const double critical=.30/(1.+1.2*ds)+.055*(1.-exp(-.020*ds));
  theta=drag*speed*speed/(rel*9.81*d);
  if(theta<=critical)return;
  const double eps=double(1.e-4f),root=sqrt(theta+eps);
  const double x1=12.*root*(theta-critical);
  // Stock trab12 retains WAVEPS and phi even with waves off. The tiny
  // near-threshold branch is therefore angular; do NOT evaluate in face axes.
  const double x2=12.*(.95+.19*cos(-2.*atan2(v,u)))*sqrt(eps)*theta;
  const double scale=p[1]*sqrt(9.81*rel*d*d*d);
  bed=scale*fmax(x1,x2);
  if(x1>=x2){
   const double radial=scale*12.*theta*((theta-critical)/root+2.*root)/speed;
   gx=radial*u/speed;gy=radial*v/speed;
  }else{
   const double k=scale*12.*sqrt(eps)*drag/(rel*9.81*d);
   gx=2.*k*(.95+.19)*u;gy=2.*k*(.95-.19)*v;
  }
 }else{
  bed=nan("");sus=nan("");gx=nan("");gy=nan("");
 }
}
RIVER_LOCAL void river_vector_bed_row(double acceleration,double h,double u,double v,
 double nx,double ny,double velocity_factor,double mag,double gx,double gy,
 double&d0,double&d1,double&d2){
 const double sp2=u*u+v*v,speed=sqrt(sp2),eps=double(1.e-6f);
 const double factor=acceleration*mag/(speed*h);
 const double gn=gx*nx+gy*ny,gt=-gx*ny+gy*nx;
 // erosed EFU/EFV divide by UMOD+eps. This mapping is NOT a frozen
 // constant: include its radial derivative at fixed log-law conversion.
 const double ratio=speed/(speed+eps);
 const double radial=eps/(speed+eps)/(speed+eps)*(gn*u+gt*v)/speed;
 const double beta_n=acceleration*velocity_factor/h*(ratio*gn+radial*u);
 const double beta_t=acceleration*velocity_factor/h*(ratio*gt+radial*v);
 d1=factor*(1.-u*u/sp2)+beta_n*u/sqrt(sp2);
 d2=-factor*u*v/sp2+beta_t*u/sqrt(sp2);
 d0=-d1*u-d2*v;
}

#define RIVER_VR1993_EMBEDDED
#define LOCAL RIVER_LOCAL
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

RIVER_LOCAL int river_vr_current(const double*p,double h,double u,double v,double d,
 double d90,double z0,double fraction,double dss,double ws,double*out,double sigmol_override=0.){
 double ds=d*pow(p[14]*9.81/(p[15]*p[15]),.3333);
 double theta=ds<=4.?.24/ds:ds<=10.?.14/pow(ds,.64):
  ds<=20.?.04/pow(ds,.1):ds<=150.?.013*pow(ds,.29):.055;
 double rho=p[13]/(1.+p[14]);
 double forcing[24]={fmax(.01,h),u,v,h*double(.368f),z0,z0,
  d,d90,p[13],rho,9.81,p[15],ws,ds,p[38]*rho*p[14]*9.81*d*theta,p[41],
  fraction,0.,dss,0.,double(.41f),double(1.e-6f),p[40],sigmol_override>0.?sigmol_override:p[39]};
 return river_vr1993_point(forcing,1,0,p+29,out);
}

// Algebraic equilibrium suspended flux, added ONCE to reconstructed bedload.
// FLOW eqtran non-suspfrac branch. No resident suspended inventory.
// Inputs distinguish physical characteristic velocity from EFVLM/EFUMN.
// Return signed suspended magnitude (kg/m/s), its Cartesian vector and DSS.
// Wang-Fredsoe's signed residual is intentional in this algebraic total.
RIVER_LOCAL int river_total_suspended(const double*p,double h,double u,double v,
 double ue,double ve,double c,double d,double d90,double hide,double z0,
 double fraction,double dss,double&mag,double&x,double&y,double&nextdss){
 mag=0.;x=0.;y=0.;nextdss=dss;
 int law=int(p[0]);
 if(law==2||law==12||law==14||law>=16)return 0;
 const double physical=hypot(u,v),effective=hypot(ue,ve),eps=double(1.e-6f);
 const double factor=physical/(physical+eps),efx=ue*factor,efy=ve*factor;
 if(law==-1){
  // erosed NON-suspfrac: nominal DSS, zero fall-speed profile, sigmol=1.
  // Unlike the split route, no accepted suspended-size update is consumed.
  // Explicitly named in the resolver/UI; not physical settling in disguise.
  double r[12];
  int bad=river_vr_current(p,h,u,v,d,d90,z0,fraction,d,0.,r,1.);
  if(bad)return bad;
  // tram1's fraction-dependent cutoff must precede undoing the fraction.
  // The shared face owner later applies the donor's fraction exactly once.
  if(fraction>0.){double a=r[2]*fmax(.01,h)/fraction;x=a*efx;y=a*efy;}
  mag=hypot(x,y);
 }else if(law==11){
  double b,s,gx,gy,t;river_current_vector_law(p,efx,efy,fmax(.01,h),d,b,s,gx,gy,t);
  double speed=hypot(efx,efy);mag=p[13]*s;
  if(speed>0.){x=mag*efx/speed;y=mag*efy/speed;}
 }else{
  double b,s,der,shear=physical*.41/log1p(h*double(.368f)/z0);
  river_scalar_law(p,effective,c,fmax(.01,h),d,d90,hide,b,s,der,shear);
  double a=p[13]*s/(effective+eps);
  x=a*efx;y=a*efy;mag=a*hypot(efx,efy);
 }
 return isfinite(mag)&&isfinite(x)&&isfinite(y)&&isfinite(nextdss)?0:4;
}
// Frozen-depth/material/roughness principal part, not a new physical law.
// Differentiate the SAME algebraic assembly, including both native epsilon
// denominators. A symmetric radial perturbation avoids axis-dependent noise.
RIVER_LOCAL int river_total_suspended_radial(const double*p,double h,double u,double v,
 double velocity_factor,double c,double d,double d90,double hide,double z0,
 double fraction,double dss,double&mag,double&der){
 double x,y,nd;
 int bad=river_total_suspended(p,h,u,v,u*velocity_factor,v*velocity_factor,c,
  d,d90,hide,z0,fraction,dss,mag,x,y,nd);
 if(bad)return bad;
 double speed=hypot(u,v);der=0.;
 if(speed<=1.e-12)return 0;
 double delta=fmin(.25*speed,1.e-5*fmax(speed,1.e-3)),plus,minus;
 double rp=(speed+delta)/speed,rm=(speed-delta)/speed;
 bad=river_total_suspended(p,h,u*rp,v*rp,u*rp*velocity_factor,v*rp*velocity_factor,c,
  d,d90,hide,z0,fraction,dss,plus,x,y,nd);
 if(bad)return bad;
 bad=river_total_suspended(p,h,u*rm,v*rm,u*rm*velocity_factor,v*rm*velocity_factor,c,
  d,d90,hide,z0,fraction,dss,minus,x,y,nd);
 der=(plus-minus)/(2.*delta);
 return bad;
}

ENTRY void wave_state_gradients(int n,int nf,const double*q,const double*z,
 const int*off,const int*faces,const int*owner,const int*nb,const double*delta,
 const double*inv,const double*disp,double*grad){
 LOOP(n){int na=off[n];
  for(int c=0;c<4;++c){double v=value(c,i,n,q,z),lo=v,hi=v,bx=0.,by=0.;
   for(int k=off[i];k<off[i+1];++k){int f=faces[k],j=owner[f]==i?nb[f]:owner[f];if(j<0)continue;
    double dx=delta[k],dy=delta[na+k],w=1./(dx*dx+dy*dy),dv=value(c,j,n,q,z);
    lo=fmin(lo,dv);hi=fmax(hi,dv);bx+=w*dx*(dv-v);by+=w*dy*(dv-v);
   }
   double gx=inv[i]*bx+inv[n+i]*by,gy=inv[n+i]*bx+inv[2*n+i]*by,phi=1.;
   double eps=64.*2.2204460492503131e-16*fmax(fabs(v),fmax(fabs(lo),fabs(hi)));
   for(int k=off[i];k<off[i+1];++k){int f=faces[k];if(nb[f]<0)continue;
    int side=owner[f]==i?0:2;double d=gx*disp[side*nf+f]+gy*disp[(side+1)*nf+f];
    if(d>eps)phi=fmin(phi,(hi-v)/d);else if(d<-eps)phi=fmin(phi,(lo-v)/d);
   }
   grad[2*c*n+i]=fmax(0.,phi)*gx;grad[(2*c+1)*n+i]=fmax(0.,phi)*gy;
  }
 }
}
ENTRY void matrix_bed_faces(int nf,int n,double acceleration_base,const double*q,const double*z,
 const double*cell,const double*capgrad,const double*stategrad,const int*owner,const int*nb,
 const double*normal,const double*disp,double*face,double*diagnostic,const double*law_params,const double*packing,const double*total_frac,const double*total_dss,const double*hide,const double*d90,const double*rough,double diameter){
 LOOP(nf){int o=owner[i],j=nb[i];for(int k=0;k<3;++k)diagnostic[k*nf+i]=0.;if(j<0)continue;
  // Freeze the accepted active-layer packing at this face, just like
  // roughness/hiding. This is not a full mixture/porosity Jacobian.
  double acceleration=acceleration_base/(.5*(packing[o]+packing[j]));
  double nx=normal[i],ny=normal[nf+i],s[2][4],qs[2],der[3]={0.,0.,0.};
  for(int side=0;side<2;++side){int a=side?j:o;
   double dx=disp[2*side*nf+i],dy=disp[(2*side+1)*nf+i];
   // stategrad is the fresh hydraulic primitive face array (side,eta,u,v,z).
   s[side][0]=fmax(1.e-6,stategrad[(4*side)*nf+i]-stategrad[(4*side+3)*nf+i]);
   s[side][1]=stategrad[(4*side+1)*nf+i];s[side][2]=stategrad[(4*side+2)*nf+i];
   s[side][3]=stategrad[(4*side+3)*nf+i];
   double vx=s[side][1],vy=s[side][2];s[side][1]=vx*nx+vy*ny;s[side][2]=-vx*ny+vy*nx;
   double x=cell[3*n+a]+capgrad[a]*dx+capgrad[n+a]*dy;
   double y=cell[4*n+a]+capgrad[2*n+a]*dx+capgrad[3*n+a]*dy;
   qs[side]=acceleration*(x*nx+y*ny)/law_params[13];
  }
  double h=.5*(s[0][0]+s[1][0]),rt0=sqrt(s[0][0]),rt1=sqrt(s[1][0]);
  double u=(rt0*s[0][1]+rt1*s[1][1])/(rt0+rt1),v=(rt0*s[0][2]+rt1*s[1][2])/(rt0+rt1),gh=9.81*h;

  // The common bed determines ONLY the hydraulic opening. Never replace
  // physical z or physical depth in Exner/capacity by the common bed.
  double top=fmax(s[0][3],s[1][3]);
  double eta0=s[0][0]+s[0][3],eta1=s[1][0]+s[1][3];
  double hs0=fmax(0.,eta0-top),hs1=fmax(0.,eta1-top);
  // Stable pressure divided difference, including a dry/active crossing.
  double deta=eta1-eta0, opening;
  if(hs0>0. && hs1>0.)opening=.5*(hs0+hs1);
  else if(deta!=0.)opening=.5*(hs0+hs1)*(hs1-hs0)/deta;
  else opening=0.;
  double pressure_jump=.5*9.81*(hs0+hs1)*(hs1-hs0);
  gh=9.81*opening;
  // Positive EH principal-part linearization at ONE mean state. Averaging
  // cell Jacobians evaluated at different speeds violates the homogeneous
  // relation dh+u*dm_n+v*dm_t=0 and can produce complex roots. Steering and
  // effective shear remain in the physical flux; only the dissipative wave
  // matrix uses a frozen mean magnitude and unrotated EH velocity derivative.
  double sp2=u*u+v*v;
  // Evaluate the frozen constitutive coefficient at the SAME mean h,u,v.
  // Working roughness depth is floored; characteristic elevation is physical.
  // Endpoint capacity (a fifth power) cannot serve as a mean-state magnitude.
  double zrough=.5*(rough[2*n+o]+rough[2*n+j]);
  double cmean=sqrt(9.81)/.41*log1p(fmax(.01,h)/fmax(1.e-8,exp(1.)*zrough));
  double velocity_factor=.41*cmean/(sqrt(9.81)*log1p(h*double(float(.368))/zrough));
  double cwork=fmax(1.e-6,cmean),ve=sqrt(sp2)*velocity_factor;
  double dm90=.5*(d90[o]+d90[j]);
  double mag=0.,sus=0.,derivative=0.;
  double shear=law_params[0]==13?sqrt(sp2)*.41/log1p(h*double(float(.368))/zrough):0.;
  river_scalar_law(law_params,ve,cwork,fmax(.01,h),diameter,dm90,.5*(hide[o]+hide[j]),mag,sus,derivative,shear);
  mag*=law_params[12];derivative*=law_params[12];

  double sm,sd;
  // Use GLOBAL components for formula direction, including Soulsby angular terms.
  double gu=u*nx-v*ny,gv=u*ny+v*nx;
  // VR bedload uses physical speed, but eqtran folds concentration with EFUMN.
  double vf=.41*cmean/(sqrt(9.81)*log1p(h*double(.368f)/zrough));
  if(river_total_suspended_radial(law_params,h,gu,gv,vf,cwork,diameter,dm90,
      .5*(hide[o]+hide[j]),zrough,.5*(total_frac[o]+total_frac[j]),
      .5*(total_dss[o]+total_dss[j]),sm,sd)){diagnostic[2*nf+i]=8.;continue;}
  sm/=law_params[13];sd/=law_params[13];
  mag+=sm;

  // Positive radial principal part uses the same summed physical flux.
  // Actual secondary steering remains in the reconstructed physical vector.
  derivative=derivative*velocity_factor+sd;velocity_factor=1.;
  if(!(h>0.)||sp2<=1.e-20||mag<=0.)continue;
  double factor=acceleration*mag/(sqrt(sp2)*h);
  // Finite at the mobility threshold: do not divide by excess.
  double radial=acceleration*derivative*velocity_factor/h;
  der[0]=-radial*u;der[1]=factor+(radial-factor)*u*u/sp2;
  der[2]=(radial-factor)*u*v/sp2;
  double b=-2.*u,c=u*u-gh*(1.+der[1]),d=-gh*(der[0]+v*der[2]);
  double p=c-b*b/3.,qq=2.*b*b*b/27.-b*c/3.+d;
  if(p>=0.){diagnostic[2*nf+i]=1.;continue;}
  double radius=2.*sqrt(-p/3.),arg=(3.*qq/(2.*p))*sqrt(-3./p);
  if(fabs(arg)>1.+1.e-10){diagnostic[2*nf+i]=2.;continue;}
  double theta=acos(fmax(-1.,fmin(1.,arg)))/3.,roots[3],speed=fabs(u);
  for(int k=0;k<3;++k){roots[k]=radius*cos(theta-2.*3.14159265358979323846*k/3.)-b/3.;speed=fmax(speed,fabs(roots[k]));}
  // General Newton/Hermite polynomial of sign(A). Unlike the native MPM
  // shortcut, allow two eigenvalues on either side of zero at oblique faces.
  // Repeated same-sign roots have zero divided differences, not a pole.
  double node[4]={roots[0],roots[1],roots[2],u},coef[4],sgn[4];
  for(int k=0;k<4;++k)for(int l=k+1;l<4;++l)if(node[l]<node[k]){double tmp=node[k];node[k]=node[l];node[l]=tmp;}
  for(int k=0;k<4;++k)coef[k]=sgn[k]=fabs(node[k])<=1.e-13*speed?0.:(node[k]>0.?1.:-1.);
  // Numerically coincident stationary waves: a cubic's roundoff can place
  // one root just across zero. Treat the entire machine-scale cluster as
  // stationary; never divide a sign jump by a roundoff-size eigenvalue gap.
  // This changes waves only within 1e-12 of the spectral radius, not a
  // tunable physical velocity/transport threshold.
  for(int first=0;first<4;){int last=first;
   while(last<3&&node[last+1]-node[last]<=1.e-12*speed)++last;
   if(sgn[first]!=sgn[last])for(int k=first;k<=last;++k)coef[k]=sgn[k]=0.;
   first=last+1;
  }
  bool bad=false;
  for(int order=1;order<4;++order)for(int k=3;k>=order;--k){
   double den=node[k]-node[k-order];
   if(fabs(den)<=1.e-12*speed){
    if(sgn[k]!=sgn[k-order]){bad=true;break;}
    coef[k]=0.;
   }else coef[k]=(coef[k]-coef[k-1])/den;
  }
  if(bad){diagnostic[2*nf+i]=4.;continue;}
  double a[4][4]={{0.,1.,0.,0.},{gh-u*u,2.*u,0.,gh},{-u*v,v,u,0.},{der[0],der[1],der[2],0.}};
  double row[4]={0.,0.,0.,coef[3]};
  for(int k=2;k>=0;--k){double next[4]={0.,0.,0.,coef[k]};
   for(int r=0;r<4;++r)for(int col=0;col<4;++col)next[col]+=row[r]*(a[r][col]-(r==col?node[k]:0.));
   for(int col=0;col<4;++col)row[col]=next[col];
  }
  // Preserve the physical MUSCL flux; do not inject independent hydraulic jumps.
  const double bed_speed=fmax(0.,row[1]*gh);
  double next=face[i]*nx+face[nf+i]*ny-.5*bed_speed*(s[1][3]-s[0][3])*law_params[13]/acceleration;
  double old=face[i]*nx+face[nf+i]*ny;
  face[i]+=(next-old)*nx;face[nf+i]+=(next-old)*ny;
  diagnostic[i]=speed;diagnostic[nf+i]=next-old;
 }
}
