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
 // Explicit bulk D90 replaces class-CDF estimates at every cell/face/boundary.
 if(law==2||law==13)d90=parameters[46];
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
  // Recking (2010), WRR W03518 Eqs 4,12-14, explicit bulk D84/D50.
  // The reach slope is explicit, NOT a local instantaneous friction slope.
  bed=0.;sus=0.;dspeed=0.;
  if(!(speed>0.)||!(depth>0.)||parameters[1]==0.)return;
  double theta=speed*speed/(chezy*chezy*parameters[14]*diameter);
  if(!(theta>0.))return;
  double ratio=parameters[48];
  double critical=(1.32*parameters[3]+.037)*pow(ratio,-.93);
  double low=log(.0005)-18.*sqrt(parameters[3])*log(ratio)+6.5*log(theta/critical);
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
  // Mean diameter plus independently measured D90/D30 grading.
  // Physical depth, physical velocity and hydraulic Chezy (m^0.5/s).
  // Output solid volume / width / time (m^2/s), no porosity or density factor.
  bed=0.;sus=0.;dspeed=0.;
  if(!(depth>0.)||!(speed>0.))return;
  const double relative=parameters[14],head=speed*speed/(chezy*chezy);
  const double excess=head-parameters[6]*relative*diameter;
  if(!(excess>0.))return;
  const double pref=parameters[1]*parameters[47]/relative*pow(head/depth,.6);
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
