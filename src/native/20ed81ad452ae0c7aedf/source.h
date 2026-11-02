// Shared OpenMP/CUDA current-only general split transport and mass exchange.
#ifdef __CUDACC__
#define ENTRY extern "C" __global__
#define LOOP(N) for(int i=blockDim.x*blockIdx.x+threadIdx.x;i<N;i+=blockDim.x*gridDim.x)
#else
#include <cmath>
#include <omp.h>
using std::isfinite;
#define ENTRY extern "C" __declspec(dllexport)
#define LOOP(N) _Pragma("omp parallel for schedule(static)") for(int i=0;i<N;++i)
extern "C" __declspec(dllexport) void mobile_threads(int n){omp_set_dynamic(0);omp_set_num_threads(n);}
#endif
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
 if(law==105){
  // Wong & Parker (2006), authors preprint Eq24: 3.97*(theta-.0495)^1.5.
  // Arithmetic mean Dm; local hydraulic BED shear, no MPM form-drag factor.
  // Solid volume / width / time; one uniform grain and no suspension.
  bed=0.;sus=0.;dspeed=0.;
  if(!(speed>0.)||!(depth>0.)||parameters[1]==0.)return;
  double theta=speed*speed/(chezy*chezy*parameters[14]*diameter);
  double excess=theta-.0495;
  if(!(excess>0.))return;
  double pref=3.97*parameters[1]*diameter*sqrt(9.81*parameters[14]*diameter);
  bed=pref*excess*sqrt(excess);
  dspeed=3.*pref*sqrt(excess)*theta/speed;
  return;
 }
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

ENTRY void split_capacity(int n,double d,double bed_factor,const double*q,const double*cq,const double*c,
 const double*hi,const double*d90,const double*hide,const double*law_params,const double*z0,double*out,int*status,double*taurat){
 LOOP(n){bool valid=isfinite(c[i])&&isfinite(hi[i])&&c[i]>=1.e-6&&q[i]>0.&&cq[i]>0.;
  for(int k=0;k<3;++k)valid=valid&&isfinite(q[k*n+i])&&isfinite(cq[k*n+i]);
  valid=valid&&isfinite(d90[i])&&isfinite(hide[i])&&d90[i]>0.&&hide[i]>0.;
  valid=valid&&isfinite(z0[i])&&z0[i]>0.;
  status[i]=valid?0:1;taurat[i]=0.;if(!valid)continue;
  double h=q[i],u=q[n+i]/h,v=q[2*n+i]/h,ue=cq[n+i]/cq[i],ve=cq[2*n+i]/cq[i];
  double speed=hypot(ue,ve),theta=speed*speed/(c[i]*c[i]*law_params[14]*d),bed,sus,derivative;
  double shear=hypot(u,v)*.41/log1p(h*double(float(.368))/z0[i]);
  theta=shear*shear/(9.81*law_params[14]*d);
  if(law_params[0]==-1){double vr[12];
   if(river_vr_current(law_params,h,u,v,d,d90[i],z0[i],1.,d,0.,vr)){status[i]=3;continue;}
   bed=vr[11]/law_params[13];sus=0.;derivative=vr[10]/law_params[13];
   taurat[i]=vr[3];theta=shear*shear/(9.81*law_params[14]*d);ue=u;ve=v;speed=hypot(u,v);
  }else if(law_params[0]==11||law_params[0]==12){double gx,gy,own_theta;
   double physical_speed=hypot(u,v),factor=physical_speed/(physical_speed+double(1.e-6f));
   ue*=factor;ve*=factor;speed=hypot(ue,ve);
   river_current_vector_law(law_params,ue,ve,fmax(.01,cq[i]),d,bed,sus,gx,gy,own_theta);
   derivative=hypot(gx,gy);theta=shear*shear/(9.81*law_params[14]*d);
  }else river_scalar_law(law_params,speed,c[i],fmax(.01,cq[i]),d,d90[i],hide[i],bed,sus,derivative,shear);
  if(!isfinite(bed)||!isfinite(sus)||!isfinite(derivative)){status[i]=2;continue;}
  double mag=bed_factor*law_params[13]*bed;
  double us=hypot(u,v),alpha=us>0.?12.5*(1.-sqrt(9.81)/(.4*c[i]))*(hi[i]/h)/us:0.;
  bool vector_owned=law_params[0]==-1||law_params[0]==11||law_params[0]==12;
  if(vector_owned)alpha=0.;
  double bx=vector_owned?ue:u+alpha*v,by=vector_owned?ve:v-alpha*u,bs=hypot(bx,by);
  out[i]=speed>0.?mag*ue/speed:0.;out[n+i]=speed>0.?mag*ve/speed:0.;out[2*n+i]=theta;
  out[3*n+i]=bs>0.?mag*bx/bs:0.;out[4*n+i]=bs>0.?mag*by/bs:0.;out[5*n+i]=alpha;
 }
}
ENTRY void sus_faces(int nf,int n,int ng,double diff,double dry,const int*o,const int*j,const int*bid,
 const double*length,const double*distance,const double*h,const double*mass,const double*water,
 const double*bc,double*flux){
 LOOP(nf){int a=o[i],b=j[i];bool wall=b<0&&bid[i]<0;
  for(int k=0;k<ng;++k){double ca=h[a]>dry?mass[k*n+a]/h[a]:0.;
   double cb=b>=0?(h[b]>dry?mass[k*n+b]/h[b]:0.):bc[k*nf+i];
   double f=wall?0.:water[i]*(water[i]>=0.?ca:cb);
   // Diffusion uses only true internal wet faces. Open boundary is prescribed
   // advective inflow / zero diffusive gradient outflow, not a hidden reservoir.
   if(b>=0&&h[a]>dry&&h[b]>dry)f+=diff*.5*(h[a]+h[b])*(ca-cb)/(distance[i]+distance[nf+i]);
   flux[k*nf+i]=f;
  }
 }
}
ENTRY void sus_limit(int n,int nf,int ng,double dt,const int*offset,const int*faces,const int*o,
 const double*length,const double*area,const double*mass,const double*flux,double*scale){
 LOOP(n){for(int k=0;k<ng;++k){double loss=0.;for(int e=offset[i];e<offset[i+1];++e){int f=faces[e];
    double signed_flux=(o[f]==i?1.:-1.)*flux[k*nf+f];loss+=fmax(0.,signed_flux)*length[f];}
   scale[k*n+i]=loss>0.?fmin(1.,mass[k*n+i]*area[i]/(dt*loss)):1.;}}
}
ENTRY void sus_scale(int nf,int n,int ng,const int*o,const int*j,const double*scale,double*flux){
 LOOP(nf){for(int k=0;k<ng;++k){int donor=flux[k*nf+i]>=0.?o[i]:j[i];
  if(donor>=0)flux[k*nf+i]*=scale[k*n+donor];}}
}
ENTRY void sus_gather(int n,int nf,int ng,double dt,const int*offset,const int*faces,const int*o,
 const double*length,const double*area,const double*mass,const double*flux,double*out){
 LOOP(n){for(int k=0;k<ng;++k){double sum=0.;for(int e=offset[i];e<offset[i+1];++e){int f=faces[e];
  sum+=(o[f]==i?-1.:1.)*flux[k*nf+f]*length[f];}out[k*n+i]=mass[k*n+i]+dt*sum/area[i];}}
}
// Same predicates as the original separate NumPy/CuPy scans. Cell-parallel,
// with one writer per status entry and no changes to exchange arithmetic.
ENTRY void sus_check(int n,int ng,const double*next,const double*available,
 const double*delta,int*status){
 LOOP(n){int bad=0;for(int k=0;k<ng;++k){int a=k*n+i;
  if(!isfinite(next[a])||next[a]<0.||available[a]+delta[a]<-1e-10)bad=1;}
  status[i]=bad;
 }
}
ENTRY void vr_sus_exchange(int n,int ng,double dt,double mf,double dry,double sedthr,double cref,double temp,
 double sus_factor,const double*diam,const double*h0,const double*q,const double*cq,const double*c,const double*z0,
 const double*mass,const double*stock,const double*frac,double*next,double*delta,double*diagnostic,const double*d90,const double*hide,const double*law_params,const double*dss,double*dss_next){
 LOOP(n){double h=q[i],hs=fmax(.01,h),speed=h>dry?hypot(q[n+i],q[2*n+i])/h:0.;
  double ve=h>dry?hypot(cq[n+i],cq[2*n+i])/cq[i]:0.;
  // erosed: USTAR = UMOD*kappa/log(1+ZUMOD/z0rou). Capacity Chezy is
  // separately supplied and must not be substituted in this shear. erosed's
  // 1e-8 denominator floor belongs to Chezy, not to USTAR's z0 argument.
  double shear=h>dry?speed*.41/log1p(h*double(float(.368))/z0[i]):0.;
  double total=0.;for(int k=0;k<ng;++k)total+=h>dry?mass[k*n+i]/h:0.;
  double hinder=pow(fmax(0.,1.-total/cref),5.),nu=4.e-5/(20.+temp);
  for(int k=0;k<ng;++k){const double*law=law_params+46*k;
   int a=k*n+i;double d=diam[k],ws;dss_next[a]=dss[a];
   double transport_d=d; if(law[0]==-1)d=dss[a];
   if(d<1.5*64.e-6)ws=law[14]*9.81*d*d/(18.*nu);
   else if(d<.001){double coef=d<128.e-6?(-2.9912/64.e-6)*d+15.9824:10.;
    ws=coef*nu/d*(sqrt(1.+law[14]*9.81*d*d*d/(100.*nu*nu))-1.);}
   else ws=1.1*sqrt(law[14]*9.81*d);
   ws*=hinder;d=transport_d;double wsl=fmax(.001,ws),theta=ve*ve/(c[i]*c[i]*law[14]*d);
   double bed,qs,derivative;
   if(law[0]==11||law[0]==12){double gx,gy,own_theta,factor=speed/(speed+double(1.e-6f));
    river_current_vector_law(law,h>dry?factor*cq[n+i]/cq[i]:0.,h>dry?factor*cq[2*n+i]/cq[i]:0.,hs,d,bed,qs,gx,gy,own_theta);
   }else if(law[0]==-1){bed=0.;qs=0.;derivative=0.;}
   else river_scalar_law(law,ve,c[i],hs,d,d90[i],hide[a],bed,qs,derivative,shear);
   double cap=sus_factor*law[13]*qs;
   double ceq=cap/(ve+1.e-6)/hs*frac[a],hots=0.,rate=0.,source=0.;
   if(law[0]==-1&&h>sedthr){double vr[12];
    int bad=river_vr_current(law,h,q[n+i]/h,q[2*n+i]/h,d,d90[i],z0[i],frac[a],dss[a],ws,vr);
    if(bad){next[a]=nan("");delta[a]=nan("");continue;}
    ceq=sus_factor*vr[2];dss_next[a]=vr[6];
   }
   if(h>sedthr){if(speed>1.e-6&&shear>1.e-6){double u=fmin(shear/speed,.15),x=fmin(wsl/shear,1.);
      double ts=x*exp(((double)1.547f-(double)20.12f*u)*x*x*x+
       ((double)326.832f*pow(u,(double)2.2047f)-(double).2f)*x*x+
       ((double).1385f*log(u)-(double)6.4061f)*x+((double).5467f*u+(double)2.1963f));
      hots=wsl/ts;source=ceq*hots*h0[i]/fmax(.01,h0[i]);rate=hots/hs;
    }else rate=wsl/hs;}
   double candidate=(mass[a]+dt*source)/(1.+dt*rate);
   // MF accelerates bed response, not scalar time. Bound erosion against
   // available stock / MF before making either side of the transaction.
   candidate=fmax(0.,fmin(candidate,mass[a]+stock[a]/mf));
   double change=candidate-mass[a];delta[a]=-mf*change;next[a]=candidate;
   // A truly dry cell deposits its stranded scalar ONCE on physical time.
   // This transfer is not multiplied by MF or confused with entrainment.
   if(h<=dry){delta[a]=mass[a];next[a]=0.;}
   diagnostic[a]=ws;diagnostic[n*ng+a]=ceq;diagnostic[2*n*ng+a]=hots;
  }
 }
}

ENTRY void mixture(int n,int ng,int hiding,const double*mass,const double*diam,
 double*factor,double*d90,double*mean){
 LOOP(n){double total=0.,dm=0.,low=1.e100,high=-1.e100;
  double logd[32],lo[32],hi[32];
  for(int k=0;k<ng;++k){total+=mass[k*n+i];dm+=mass[k*n+i]*diam[k];
   logd[k]=log(diam[k]);lo[k]=logd[k]+1.25*log(.75);hi[k]=logd[k]+1.25*log(1.5);
   low=fmin(low,lo[k]);high=fmax(high,hi[k]);}
  // A depleted column carries zero transport via the stock limiter. Retain
  // finite, neutral constitutive scratch without inventing surface material.
  if(total<=0.){d90[i]=1.5*diam[0];mean[i]=diam[0];for(int k=0;k<ng;++k)factor[k*n+i]=1.;continue;}
  mean[i]=dm/total;
  for(int k=0;k<ng;++k){double r=diam[k]/fmax(mean[i],1.e-4),v=1.27875360095283;
   factor[k*n+i]=hiding?pow(v/(v+log10(r)),2.):1.;}
  for(int it=0;it<60;++it){double mid=.5*(low+high),cdf=0.;
   for(int k=0;k<ng;++k){double f=0.;if(mid>=hi[k])f=1.;else if(mid>logd[k])f=.5+.5*(mid-logd[k])/(hi[k]-logd[k]);
    else if(mid>lo[k])f=.5*(mid-lo[k])/(logd[k]-lo[k]);cdf+=mass[k*n+i]/total*f;}
   if(cdf<.9)low=mid;else high=mid;
  }d90[i]=exp(.5*(low+high));
 }
}

ENTRY void intrinsic_capacity(int n,double d,double bed_factor,const double*q,const double*cq,const double*c,
 const double*hi,const double*d90,const double*hide,const double*law_params,const double*z0,const double*grain,const double*hydraulic_c,double*out,int*status,double*taurat){
 LOOP(n){bool valid=isfinite(c[i])&&isfinite(hi[i])&&c[i]>=1.e-6&&q[i]>0.&&cq[i]>0.;
  for(int k=0;k<3;++k)valid=valid&&isfinite(q[k*n+i])&&isfinite(cq[k*n+i]);
  valid=valid&&isfinite(d90[i])&&isfinite(hide[i])&&d90[i]>0.&&hide[i]>0.;
  valid=valid&&isfinite(z0[i])&&z0[i]>0.;
  valid=valid&&isfinite(hydraulic_c[i])&&hydraulic_c[i]>0.&&isfinite(grain[i])&&grain[i]>=0.
   &&isfinite(grain[n+i])&&grain[n+i]>=0.&&isfinite(grain[2*n+i])&&grain[2*n+i]>=0.&&grain[2*n+i]<=1.;
  status[i]=valid?0:1;taurat[i]=0.;if(!valid)continue;
  double h=q[i],u=q[n+i]/h,v=q[2*n+i]/h,ue=cq[n+i]/cq[i],ve=cq[2*n+i]/cq[i];
  double speed=hypot(ue,ve),theta=speed*speed/(c[i]*c[i]*law_params[14]*d),bed,sus,derivative;
  double shear=hypot(u,v)*.41/log1p(h*double(float(.368))/z0[i]);
  theta=shear*shear/(9.81*law_params[14]*d);
  if(law_params[0]==-1){double vr[12];
   if(river_vr_current(law_params,h,u,v,d,d90[i],z0[i],1.,d,0.,vr)){status[i]=3;continue;}
   bed=vr[11]/law_params[13];sus=0.;derivative=vr[10]/law_params[13];
   taurat[i]=vr[3];theta=shear*shear/(9.81*law_params[14]*d);ue=u;ve=v;speed=hypot(u,v);
  }else if(law_params[0]==11||law_params[0]==12){double gx,gy,own_theta;
   double physical_speed=hypot(u,v),factor=physical_speed/(physical_speed+double(1.e-6f));
   ue*=factor;ve*=factor;speed=hypot(ue,ve);
   river_current_vector_law(law_params,ue,ve,fmax(.01,cq[i]),d,bed,sus,gx,gy,own_theta);
   derivative=hypot(gx,gy);theta=shear*shear/(9.81*law_params[14]*d);
  }else {double tau=law_params[13]/(1.+law_params[14])*9.81*(u*u+v*v)/(hydraulic_c[i]*hydraulic_c[i]);
   double elasticity=2.,cdr,cds;
   if(law_params[43]&&river_skin_stress(hypot(u,v),fmax(.01,h),law_params[13]/(1.+law_params[14]),
     law_params[44],law_params[45],tau,elasticity,cdr,cds)){status[i]=4;continue;}
   river_scalar_law(law_params,speed,c[i],fmax(.01,cq[i]),d,d90[i],hide[i],bed,sus,derivative,shear,
    grain[i],grain[n+i],grain[2*n+i],tau,elasticity);}
  if(!isfinite(bed)||!isfinite(sus)||!isfinite(derivative)){status[i]=2;continue;}
  // FLOW ust2 for adjust_bedload is characteristic shear, not TAUB.
  // TAUB above belongs only to the intrinsic capacity law.
  theta=shear*shear/(9.81*law_params[14]*d);
  double mag=bed_factor*law_params[13]*bed;
  double us=hypot(u,v),alpha=us>0.?12.5*(1.-sqrt(9.81)/(.4*c[i]))*(hi[i]/h)/us:0.;
  bool vector_owned=law_params[0]==-1||law_params[0]==11||law_params[0]==12;
  if(vector_owned)alpha=0.;
  double bx=vector_owned?ue:u+alpha*v,by=vector_owned?ve:v-alpha*u,bs=hypot(bx,by);
  out[i]=speed>0.?mag*ue/speed:0.;out[n+i]=speed>0.?mag*ve/speed:0.;out[2*n+i]=theta;
  out[3*n+i]=bs>0.?mag*bx/bs:0.;out[4*n+i]=bs>0.?mag*by/bs:0.;out[5*n+i]=alpha;
 }
}

// Noncohesive constant-per-class branch of FLOW compdiam/compsandfrac.
// Caller resolves mixed/layered composition weights. No within-class spread
// enters DgSD in this branch. The <2mm sand test has no lower size cutoff.
ENTRY void intrinsic_grains(int n,int ng,const double*weight,const double*diam,
 double*out,int*status){
 LOOP(n){
  double total=0.,logmean=0.,sand=0.;bool valid=true;
  out[i]=out[n+i]=out[2*n+i]=0.;
  for(int k=0;k<ng;++k){
   double w=weight[k*n+i],d=diam[k];
   if(!isfinite(w)||w<0.||!isfinite(d)||d<=0.){valid=false;break;}
   total+=w;logmean+=w*log(d);if(d<.002)sand+=w;
  }
  status[i]=valid&&isfinite(total)&&isfinite(logmean)?0:1;
  if(status[i]||total==0.)continue;
  logmean/=total;
  double variance=0.;
  for(int k=0;k<ng;++k){double v=log(diam[k])-logmean;variance+=weight[k*n+i]/total*v*v;}
  out[i]=exp(logmean);out[n+i]=exp(sqrt(variance));out[2*n+i]=sand/total;
  if(!isfinite(out[i])||!isfinite(out[n+i])||!isfinite(out[2*n+i]))status[i]=1;
 }
}

ENTRY void scalar_probe(int n,double d,const double*speed,const double*c,const double*h,
 const double*d90,const double*hide,const double*law_params,double*out){
 LOOP(n){double b,s,der;river_scalar_law(law_params,speed[i],c[i],h[i],d,d90[i],hide[i],b,s,der,sqrt(9.81)*speed[i]/c[i]);
  out[i]=b;out[n+i]=s;out[2*n+i]=der;}
}
ENTRY void scalar_forcing_probe(int n,double d,const double*speed,const double*c,const double*h,
 const double*d90,const double*hide,const double*shear,const double*law_params,double*out){
 LOOP(n){double b,s,der;river_scalar_law(law_params,speed[i],c[i],h[i],d,d90[i],hide[i],b,s,der,shear[i]);
  out[i]=b;out[n+i]=s;out[2*n+i]=der;}
}
ENTRY void intrinsic_forcing_probe(int n,double d,const double*speed,const double*c,
 const double*grain,const double*taub,const double*law_params,double*out){
 LOOP(n){double b,s,der;river_scalar_law(law_params,speed[i],c[i],.1,d,d,1.,b,s,der,0.,
  grain[i],grain[n+i],grain[2*n+i],taub[i]);out[i]=b;out[n+i]=s;out[2*n+i]=der;}
}
ENTRY void intrinsic_skin_probe(int n,double d,const double*speed,const double*c,
 const double*grain,const double*physical_speed,const double*depth,const double*law_params,double*out){
 LOOP(n){double b,s,der,tau,elasticity,cdr,cds;
  if(river_skin_stress(physical_speed[i],depth[i],law_params[13]/(1.+law_params[14]),
    law_params[44],law_params[45],tau,elasticity,cdr,cds)){
   out[i]=nan("");out[n+i]=nan("");out[2*n+i]=nan("");continue;}
  river_scalar_law(law_params,speed[i],c[i],depth[i],d,d,1.,b,s,der,0.,
   grain[i],grain[n+i],grain[2*n+i],tau,elasticity);out[i]=b;out[n+i]=s;out[2*n+i]=der;}
}
ENTRY void vector_forcing_probe(int n,double d,const double*u,const double*v,const double*h,
 const double*law_params,double*out){
 LOOP(n){double b,s,gx,gy,theta;river_current_vector_law(law_params,u[i],v[i],h[i],d,b,s,gx,gy,theta);
  double speed=hypot(u[i],v[i]);out[i]=speed>0.?b*u[i]/speed:0.;out[n+i]=speed>0.?b*v[i]/speed:0.;
  out[2*n+i]=speed>0.?s*u[i]/speed:0.;out[3*n+i]=speed>0.?s*v[i]/speed:0.;
  out[4*n+i]=gx;out[5*n+i]=gy;out[6*n+i]=theta;}
}
ENTRY void vector_wave_probe(int n,double d,const double*u,const double*v,const double*h,
 const double*angle,const double*k,const double*law_params,double*out){
 LOOP(n){double nx=cos(angle[i]),ny=sin(angle[i]),b,s,gx,gy,t,speed=hypot(u[i],v[i]);
  double kv=k[i]*speed/(speed+double(1.e-6f));
  river_current_vector_law(law_params,kv*(u[i]*nx-v[i]*ny),kv*(u[i]*ny+v[i]*nx),h[i],d,b,s,gx,gy,t);
  double a=0.,c=0.,e=0.;
  if(u[i]*u[i]+v[i]*v[i]>0.)river_vector_bed_row(1.,h[i],u[i],v[i],nx,ny,k[i],b,gx,gy,a,c,e);
  out[i]=a;out[n+i]=c;out[2*n+i]=e;}
}

ENTRY void smart_capacity(int n,double d,double bed_factor,const double*q,const double*cq,const double*c,
 const double*hi,const double*d90,const double*hide,const double*law_params,const double*z0,const double*hydraulic_c,double*out,int*status,double*taurat){
 LOOP(n){bool valid=isfinite(c[i])&&isfinite(hi[i])&&c[i]>=1.e-6&&q[i]>0.&&cq[i]>0.;
  for(int k=0;k<3;++k)valid=valid&&isfinite(q[k*n+i])&&isfinite(cq[k*n+i]);
  valid=valid&&isfinite(d90[i])&&isfinite(hide[i])&&d90[i]>0.&&hide[i]>0.;
  valid=valid&&isfinite(z0[i])&&z0[i]>0.;
  status[i]=valid?0:1;taurat[i]=0.;if(!valid)continue;
  double h=q[i],u=q[n+i]/h,v=q[2*n+i]/h,ue=cq[n+i]/cq[i],ve=cq[2*n+i]/cq[i];
  // Consume the real hydraulic drag, not erosed's depth-floor/log-law mapping.
  double speed=hypot(u,v),bed,sus,derivative;
  if(!(hydraulic_c[i]>0.)||!isfinite(hydraulic_c[i])){status[i]=2;continue;}
  double theta=speed*speed/(hydraulic_c[i]*hydraulic_c[i]*law_params[14]*d);
  river_scalar_law(law_params,speed,hydraulic_c[i],h,d,d,1.,bed,sus,derivative);
  if(!isfinite(bed)||!isfinite(derivative)){status[i]=2;continue;}
  ue=u;ve=v;
  double mag=bed_factor*law_params[13]*bed;
  double us=hypot(u,v),alpha=us>0.?12.5*(1.-sqrt(9.81)/(.4*c[i]))*(hi[i]/h)/us:0.;
  bool vector_owned=law_params[0]==-1||law_params[0]==11||law_params[0]==12;
  if(vector_owned)alpha=0.;
  double bx=vector_owned?ue:u+alpha*v,by=vector_owned?ve:v-alpha*u,bs=hypot(bx,by);
  out[i]=speed>0.?mag*ue/speed:0.;out[n+i]=speed>0.?mag*ve/speed:0.;out[2*n+i]=theta;
  out[3*n+i]=bs>0.?mag*bx/bs:0.;out[4*n+i]=bs>0.?mag*by/bs:0.;out[5*n+i]=alpha;
 }
}

ENTRY void sus_exchange(int n,int ng,double dt,double mf,double dry,double sedthr,double cref,double temp,
 double sus_factor,const double*diam,const double*h0,const double*q,const double*cq,const double*c,const double*z0,
 const double*mass,const double*stock,const double*frac,double*next,double*delta,double*diagnostic,const double*d90,const double*hide,const double*law_params){
 LOOP(n){double h=q[i],hs=fmax(.01,h),speed=h>dry?hypot(q[n+i],q[2*n+i])/h:0.;
  double ve=h>dry?hypot(cq[n+i],cq[2*n+i])/cq[i]:0.;
  // erosed: USTAR = UMOD*kappa/log(1+ZUMOD/z0rou). Capacity Chezy is
  // separately supplied and must not be substituted in this shear. erosed's
  // 1e-8 denominator floor belongs to Chezy, not to USTAR's z0 argument.
  double shear=h>dry?speed*.41/log1p(h*double(float(.368))/z0[i]):0.;
  double total=0.;for(int k=0;k<ng;++k)total+=h>dry?mass[k*n+i]/h:0.;
  double hinder=pow(fmax(0.,1.-total/cref),5.),nu=4.e-5/(20.+temp);
  for(int k=0;k<ng;++k){const double*law=law_params+46*k;
   int a=k*n+i;if(law[0]==-1){next[a]=nan("");delta[a]=nan("");continue;}double d=diam[k],ws;
   if(d<1.5*64.e-6)ws=law[14]*9.81*d*d/(18.*nu);
   else if(d<.001){double coef=d<128.e-6?(-2.9912/64.e-6)*d+15.9824:10.;
    ws=coef*nu/d*(sqrt(1.+law[14]*9.81*d*d*d/(100.*nu*nu))-1.);}
   else ws=1.1*sqrt(law[14]*9.81*d);
   ws*=hinder;double wsl=fmax(.001,ws),theta=ve*ve/(c[i]*c[i]*law[14]*d);
   double bed,qs,derivative;
   if(law[0]==11||law[0]==12){double gx,gy,own_theta,factor=speed/(speed+double(1.e-6f));
    river_current_vector_law(law,h>dry?factor*cq[n+i]/cq[i]:0.,h>dry?factor*cq[2*n+i]/cq[i]:0.,hs,d,bed,qs,gx,gy,own_theta);
   }else river_scalar_law(law,ve,c[i],hs,d,d90[i],hide[a],bed,qs,derivative,shear);
   double cap=sus_factor*law[13]*qs;
   double ceq=cap/(ve+1.e-6)/hs*frac[a],hots=0.,rate=0.,source=0.;
   if(h>sedthr){if(speed>1.e-6&&shear>1.e-6){double u=fmin(shear/speed,.15),x=fmin(wsl/shear,1.);
      double ts=x*exp(((double)1.547f-(double)20.12f*u)*x*x*x+
       ((double)326.832f*pow(u,(double)2.2047f)-(double).2f)*x*x+
       ((double).1385f*log(u)-(double)6.4061f)*x+((double).5467f*u+(double)2.1963f));
      hots=wsl/ts;source=ceq*hots*h0[i]/fmax(.01,h0[i]);rate=hots/hs;
    }else rate=wsl/hs;}
   double candidate=(mass[a]+dt*source)/(1.+dt*rate);
   // MF accelerates bed response, not scalar time. Bound erosion against
   // available stock / MF before making either side of the transaction.
   candidate=fmax(0.,fmin(candidate,mass[a]+stock[a]/mf));
   double change=candidate-mass[a];delta[a]=-mf*change;next[a]=candidate;
   // A truly dry cell deposits its stranded scalar ONCE on physical time.
   // This transfer is not multiplied by MF or confused with entrainment.
   if(h<=dry){delta[a]=mass[a];next[a]=0.;}
   diagnostic[a]=ws;diagnostic[n*ng+a]=ceq;diagnostic[2*n*ng+a]=hots;
  }
 }
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

ENTRY void total_capacity(int n,double d,double bed_factor,const double*q,const double*cq,const double*c,
 const double*hi,const double*d90,const double*hide,const double*law_params,const double*z0,double*out,int*status,double*taurat,const double*fraction,const double*dss){
 LOOP(n){bool valid=isfinite(c[i])&&isfinite(hi[i])&&c[i]>=1.e-6&&q[i]>0.&&cq[i]>0.;
  for(int k=0;k<3;++k)valid=valid&&isfinite(q[k*n+i])&&isfinite(cq[k*n+i]);
  valid=valid&&isfinite(d90[i])&&isfinite(hide[i])&&d90[i]>0.&&hide[i]>0.;
  valid=valid&&isfinite(z0[i])&&z0[i]>0.;
  status[i]=valid?0:1;taurat[i]=0.;if(!valid)continue;
  double h=q[i],u=q[n+i]/h,v=q[2*n+i]/h,ue=cq[n+i]/cq[i],ve=cq[2*n+i]/cq[i];
  double speed=hypot(ue,ve),theta=speed*speed/(c[i]*c[i]*law_params[14]*d),bed,sus,derivative;
  double shear=hypot(u,v)*.41/log1p(h*double(float(.368))/z0[i]);
  theta=shear*shear/(9.81*law_params[14]*d);
  if(law_params[0]==-1){double vr[12];
   if(river_vr_current(law_params,h,u,v,d,d90[i],z0[i],1.,d,0.,vr)){status[i]=3;continue;}
   bed=vr[11]/law_params[13];sus=0.;derivative=vr[10]/law_params[13];
   taurat[i]=vr[3];theta=shear*shear/(9.81*law_params[14]*d);ue=u;ve=v;speed=hypot(u,v);
  }else if(law_params[0]==11||law_params[0]==12){double gx,gy,own_theta;
   double physical_speed=hypot(u,v),factor=physical_speed/(physical_speed+double(1.e-6f));
   ue*=factor;ve*=factor;speed=hypot(ue,ve);
   river_current_vector_law(law_params,ue,ve,fmax(.01,cq[i]),d,bed,sus,gx,gy,own_theta);
   derivative=hypot(gx,gy);theta=shear*shear/(9.81*law_params[14]*d);
  }else river_scalar_law(law_params,speed,c[i],fmax(.01,cq[i]),d,d90[i],hide[i],bed,sus,derivative,shear);
  if(!isfinite(bed)||!isfinite(sus)||!isfinite(derivative)){status[i]=2;continue;}
  double mag=bed_factor*law_params[13]*bed;
  double us=hypot(u,v),alpha=us>0.?12.5*(1.-sqrt(9.81)/(.4*c[i]))*(hi[i]/h)/us:0.;
  bool vector_owned=law_params[0]==-1||law_params[0]==11||law_params[0]==12;
  if(vector_owned)alpha=0.;
  double bx=vector_owned?ue:u+alpha*v,by=vector_owned?ve:v-alpha*u,bs=hypot(bx,by);
  out[i]=speed>0.?mag*ue/speed:0.;out[n+i]=speed>0.?mag*ve/speed:0.;out[2*n+i]=theta;
  out[3*n+i]=bs>0.?mag*bx/bs:0.;out[4*n+i]=bs>0.?mag*by/bs:0.;out[5*n+i]=alpha;
  double sm,sx,sy,nextdss;
  if(river_total_suspended(law_params,h,u,v,cq[n+i]/cq[i],cq[2*n+i]/cq[i],
     c[i],d,d90[i],hide[i],z0[i],fraction[i],dss[i],sm,sx,sy,nextdss)){
   status[i]=4;continue;
  }
  out[i]+=sx;out[n+i]+=sy;out[3*n+i]+=sx;out[4*n+i]+=sy;

 }
}

ENTRY void total_suspended_probe(int n,double d,const double*q,const double*cq,
 const double*c,const double*d90,const double*hide,const double*z0,const double*fraction,
 const double*dss,const double*law,double*out,int*status){
 LOOP(n){double h=q[i],u=q[n+i]/h,v=q[2*n+i]/h,sm,sx,sy,nd;
  status[i]=river_total_suspended(law,h,u,v,cq[n+i]/cq[i],cq[2*n+i]/cq[i],c[i],
    d,d90[i],hide[i],z0[i],fraction[i],dss[i],sm,sx,sy,nd);
  out[i]=sx;out[n+i]=sy;out[2*n+i]=nd;out[3*n+i]=sm;
 }
}
ENTRY void total_radial_probe(int n,double d,const double*q,const double*factor,
 const double*c,const double*d90,const double*hide,const double*z0,const double*fraction,
 const double*dss,const double*law,double*out,int*status){
 LOOP(n){double h=q[i],sm,der;
  status[i]=river_total_suspended_radial(law,h,q[n+i]/h,q[2*n+i]/h,factor[i],c[i],
    d,d90[i],hide[i],z0[i],fraction[i],dss[i],sm,der);
  out[i]=sm;out[n+i]=der;
 }
}

// BRAIDS deposition-only extension, not a FLOW Galappatti exchange selector.
// One cell owns every fraction; settling uses the original total concentration.
// Exponential exchange is exact for depth and hindered velocity frozen during
// this source step. There is no equilibrium capacity or entrainment.
ENTRY void settling_only_exchange(int n,int ng,int stride,double dt,double mf,double dry,
 double cref,double temp,const double*diam,const double*q,const double*mass,
 const double*law_params,double*next,double*delta,double*diagnostic){
 LOOP(n){double h=q[i],total=0.;
  for(int k=0;k<ng;++k)total+=h>dry?mass[k*n+i]/h:0.;
  double hinder=pow(fmax(0.,1.-total/cref),5.),nu=4.e-5/(20.+temp);
  for(int k=0;k<ng;++k){int a=k*n+i;double d=diam[k],relative=law_params[k*stride+14],ws;
   // FLOW SettleFrm2 (noncohesive, no salinity/flocculation), nominal class
   // diameter. Do not import Galappatti's 1 mm/s floor or SedThr gate.
   if(d<1.5*64.e-6)ws=relative*9.81*d*d/(18.*nu);
   else if(d<.001){double coef=d<128.e-6?(-2.9912/64.e-6)*d+15.9824:10.;
    ws=coef*nu/d*(sqrt(1.+relative*9.81*d*d*d/(100.*nu*nu))-1.);}
   else ws=1.1*sqrt(relative*9.81*d);
   ws*=hinder;
   // expm1 preserves a small transfer without subtracting nearly equal masses.
   double deposited=h>dry?-mass[a]*expm1(-dt*ws/h):mass[a];
   next[a]=mass[a]-deposited;
   delta[a]=(h>dry?mf:1.)*deposited;
   diagnostic[a]=ws;diagnostic[n*ng+a]=0.;diagnostic[2*n*ng+a]=h>dry?ws:0.;
  }
 }
}
