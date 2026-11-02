// Fixed-grid bed-load sediment pipeline. Native alpha82 reconstruction is unchanged.
// Bedlaw 2 is the qualified MPM path. Bedlaws 3 and 4 are isolated weak-route
// GPU prototypes of the existing CPU Ashida--Michiue and Soulsby scalar
// closures. Bedlaw 8 is the Delft3D-compatible Engelund--Hansen capacity used
// by the Struiksma T2 validation case.
// All arrays are FP64 SoA; no flux smoothing or lower-order substitute is used.
// braids_riemanntype is supplied by the qualified augmented f-wave CUDA source.
#define BRAIDS_SEDIMENT_STAGE_GUARD_ABI 1
__device__ double st_weno(const double* v){
 double p0=(2.*v[0]-7.*v[1]+11.*v[2])/6.,p1=(-v[1]+5.*v[2]+2.*v[3])/6.,p2=(2.*v[2]+5.*v[3]-v[4])/6.;
 double a0=v[0]-2.*v[1]+v[2],c0=v[0]-4.*v[1]+3.*v[2],a1=v[1]-2.*v[2]+v[3],c1=v[1]-v[3];
 double a2=v[2]-2.*v[3]+v[4],c2=3.*v[2]-4.*v[3]+v[4];
 double b0=13./12.*a0*a0+.25*c0*c0,b1=13./12.*a1*a1+.25*c1*c1,b2=13./12.*a2*a2+.25*c2*c2;
 double mx=v[0]*v[0];for(int k=1;k<5;++k)mx=fmax(mx,v[k]*v[k]);
 double tau=fabs(b0-b2),eps=1.e-40+1.e-14*mx,r0=tau/(b0+eps),r1=tau/(b1+eps),r2=tau/(b2+eps);
 double w0=.1*(1.+r0*r0),w1=.6*(1.+r1*r1),w2=.3*(1.+r2*r2);
 return ((w0*p0+w1*p1)+w2*p2)/((w0+w1)+w2);
}

extern "C" __global__ void st_capacity(const int nc,const int ng,const double* q,const double* layers,
 const double* manning,const double* d50,const double* d90,const double* rhos,const double* bulk,
 const double* min_depth,const int* active,double rho,double grav,double dry,double critical,double target,
 double cf,int friction_mode,
 double* capacity,double* availability,double* mixture_d90,int* status){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
 status[c]=0;double volume=0.,maxd90=d90[0];
 for(int g=0;g<ng;++g){volume+=layers[g*nc+c]/bulk[g];maxd90=fmax(maxd90,d90[g]);
  capacity[g*nc+c]=0.;capacity[(ng+g)*nc+c]=0.;}
 for(int g=0;g<ng;++g)availability[g*nc+c]=volume>0. ? (layers[g*nc+c]/bulk[g])/volume*fmin(1.,volume/target):0.;
 mixture_d90[c]=0.;if(!active[c]||q[c]<=dry)return;
 double mix=volume>0.?exp(log(d50[0])+1.25*log(1.5)*(2.*.9-1.)):maxd90;mixture_d90[c]=mix;
 double h=q[c],hu=q[nc+c],hv=q[2*nc+c],minimum=fmax(min_depth[0],dry),wet=1.;
 if(h<minimum&&minimum>dry){double xi=fmax(0.,fmin(1.,(h-dry)/(minimum-dry)));wet=xi*xi*(3.-2.*xi);
  h=minimum*(1.+128.*2.2204460492503131e-16);hu=hu*h/q[c];hv=hv*h/q[c];}
 double u=hu/h,v=hv/h,speed=sqrt(u*u+v*v);
 double chezy=0.,grain=0.,ripple=0.;
 if(friction_mode==1){chezy=pow(h,1./6.)/manning[c];grain=18.*log10(12.*fmax(h/mix,1.));
  if(!(isfinite(chezy)&&chezy>0.&&grain>0.)){status[c]=1;return;}
  ripple=fmin(pow(chezy/grain,1.5),1.);}
 if(!(isfinite(h)&&isfinite(speed))){status[c]=1;return;}
 if(speed==0.)return;
 for(int g=0;g<ng;++g){double relative=rhos[g]/rho-1.,excess;
  if(friction_mode==1){double a=speed/chezy;
   double theta=(a*a)/(relative*d50[g]);excess=fmax(ripple*theta-critical,0.);}
  else{double theta=fmax(0.,cf*speed*speed/(grav*relative*d50[g])-critical);excess=theta;}
  double magnitude=8.*sqrt(grav*relative*(d50[g]*d50[g]*d50[g]))*pow(excess,1.5);
  double x=wet*(rhos[g]*(magnitude*u/speed)),y=wet*(rhos[g]*(magnitude*v/speed));
  capacity[g*nc+c]=x;capacity[(ng+g)*nc+c]=y;if(!isfinite(x)||!isfinite(y))status[c]=1;}
}

// The mixed-grain path mirrors native mixture_statistics/hiding_factors and
// bed_capacity. Keep the original identical-label kernel above for its frozen
// operation order and measured bend performance.
extern "C" __global__ void st_capacity_graded(const int nc,const int ng,const double* q,const double* layers,
 const double* manning,const double* d50,const double* d90,const double* rhos,const double* bulk,
 const double* min_depth,const int* active,double rho,double grav,double dry,double critical,double target,
 double cf,int friction_mode,
 int hiding_mode,double parker_alpha,double wu_exponent,double* capacity,double* availability,
 double* mixture_d90,double* hiding,int* status){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
 status[c]=0;double volume=0.,maxd90=d90[0];bool same=true;
 for(int g=0;g<ng;++g){double vg=layers[g*nc+c]/bulk[g];
  if(!isfinite(vg)||vg<0.)status[c]=1;
  volume+=vg;maxd90=fmax(maxd90,d90[g]);if(d50[g]!=d50[0])same=false;
  capacity[g*nc+c]=0.;capacity[(ng+g)*nc+c]=0.;hiding[g*nc+c]=1.;}
 mixture_d90[c]=0.;
 if(status[c]||!isfinite(volume)||volume<0.){status[c]=1;return;}
 for(int g=0;g<ng;++g){double vg=layers[g*nc+c]/bulk[g];
  availability[g*nc+c]=volume>0.?vg/volume*fmin(1.,volume/target):0.;}
 if(active[c]&&hiding_mode&&volume>0.){
  double mean=0.;for(int g=0;g<ng;++g)mean+=((layers[g*nc+c]/bulk[g])/volume)*d50[g];
  double local_mean=fmax(mean,1.e-4);
  for(int g=0;g<ng;++g){double ratio=d50[g]/local_mean,factor=1.;
   if(hiding_mode==2||hiding_mode==3){
    if(hiding_mode==3&&ratio<(double)0.38889f)factor=(double)0.8429f/ratio;
    else{double denominator=1.27875360095283+log10(ratio);
     if(denominator==0.){status[c]=1;return;}
     factor=pow(1.27875360095283/denominator,2.);}
   }else if(hiding_mode==4)factor=pow(ratio,-parker_alpha);
   else if(hiding_mode==5){double phi=0.;
    for(int other=0;other<ng;++other)phi+=((layers[other*nc+c]/bulk[other])/volume)*d50[other]/(d50[g]+d50[other]);
    double pei=1.-phi;if(phi<=0.||pei<=0.){status[c]=1;return;}
    factor=pow(pei/phi,wu_exponent);
   }
   if(!isfinite(factor)||factor<=0.){status[c]=1;return;}
   hiding[g*nc+c]=factor;
  }
 }
 if(!active[c]||q[c]<=dry)return;
 double mix=maxd90;
 if(volume>0.){
  if(same)mix=exp(log(d50[0])+1.25*log(1.5)*(2.*.9-1.));
  else{
   // Native piecewise-loguniform mixture D90: D10=.75 D50, D90=1.5 D50.
   double log_d50[32],lower_bound[32],upper_bound[32];
   double lo=1.e300,hi=-1.e300;
   for(int g=0;g<ng;++g){log_d50[g]=log(d50[g]);
    lower_bound[g]=log_d50[g]+1.25*log(.75);
    upper_bound[g]=log_d50[g]+1.25*log(1.5);
    lo=fmin(lo,lower_bound[g]);hi=fmax(hi,upper_bound[g]);}
   for(int iteration=0;iteration<60;++iteration){double mid=.5*(lo+hi),cumulative=0.;
    for(int g=0;g<ng;++g){double share=(layers[g*nc+c]/bulk[g])/volume;
     double lower=lower_bound[g],upper=upper_bound[g],x=log_d50[g];
     if(mid<=lower)continue;
     else if(mid<x)cumulative+=share*.5*(mid-lower)/(x-lower);
     else if(mid<upper)cumulative+=share*(.5+.5*(mid-x)/(upper-x));
     else cumulative+=share;}
    if(cumulative<.9)lo=mid;else hi=mid;}
   mix=exp(.5*(lo+hi));
  }
 }
 mixture_d90[c]=mix;if(!isfinite(mix)||mix<=0.){status[c]=1;return;}
 double original_h=q[c],original_hu=q[nc+c],original_hv=q[2*nc+c];
 for(int g=0;g<ng;++g){double h=original_h,hu=original_hu,hv=original_hv,wet=1.;
  double minimum=fmax(min_depth[g],dry);
  if(h<minimum&&minimum>dry){double xi=fmax(0.,fmin(1.,(h-dry)/(minimum-dry)));
   wet=xi*xi*(3.-2.*xi);h=minimum*(1.+128.*2.2204460492503131e-16);
   hu=original_hu*h/original_h;hv=original_hv*h/original_h;}
  if(wet<=0.)continue;
  double u=hu/h,v=hv/h,speed=sqrt(u*u+v*v);
  double chezy=0.,grain=0.,ripple=0.;
  if(friction_mode==1){chezy=pow(h,1./6.)/manning[c];grain=18.*log10(12.*fmax(h/mix,1.));
   if(!(isfinite(chezy)&&chezy>0.&&grain>0.)){status[c]=1;return;}
   ripple=fmin(pow(chezy/grain,1.5),1.);}
  if(!(isfinite(h)&&isfinite(speed))){status[c]=1;return;}
  if(speed==0.)continue;
  double relative=rhos[g]/rho-1.,excess;
  if(friction_mode==1){double a=speed/chezy;
   double theta=(a*a)/(relative*d50[g]);excess=fmax(ripple*theta-hiding[g*nc+c]*critical,0.);}
  else{double theta=fmax(0.,cf*speed*speed/(grav*relative*d50[g])-hiding[g*nc+c]*critical);excess=theta;}
  double magnitude=8.*sqrt(grav*relative*(d50[g]*d50[g]*d50[g]))*pow(excess,1.5);
  double x=wet*(rhos[g]*(magnitude*u/speed)),y=wet*(rhos[g]*(magnitude*v/speed));
  capacity[g*nc+c]=x;capacity[(ng+g)*nc+c]=y;if(!isfinite(x)||!isfinite(y))status[c]=1;
 }
}

// Isolated current-only Delft3D trab14 scalar with its generic hiding factor.
// Production law-3 dispatch still passes no hiding option; this device routine
// exists for bounded CPU/GPU qualification before any compatibility admission.
__device__ bool st_ashida_michiue_current(double h,double u,double v,double manning,
 double d50,double rhos,double rho,double grav,double calibration,double hiding,
 double& x,double& y,double& tau_ratio){
 x=0.;y=0.;tau_ratio=0.;
 if(!(isfinite(h)&&isfinite(u)&&isfinite(v)&&isfinite(manning)&&isfinite(d50)&&
      isfinite(rhos)&&isfinite(rho)&&isfinite(grav)&&isfinite(calibration)&&
      isfinite(hiding))||h<0.||manning<0.||d50<=0.||rhos<=rho||rho<=0.||
    grav<=0.||calibration<0.||hiding<0.)return false;
 if(h<=0.||calibration==0.||manning==0.)return true;
 double speed=sqrt(u*u+v*v),chezy=pow(h,1./6.)/manning;
 if(speed<1.e-6||chezy<1.e-6)return true;
 double relative=rhos/rho-1.,a=speed/chezy;
 double theta=(a*a)/(relative*d50),threshold=.05*hiding;
 tau_ratio=theta/fmax(threshold,2.2250738585072014e-308);
 double magnitude=0.;
 if(theta>threshold)magnitude=17.*calibration*sqrt(grav*relative*d50*d50*d50)*
  pow(theta,1.5)*(1.-threshold/theta)*(1.-sqrt(threshold/theta));
 x=rhos*magnitude*u/speed;y=rhos*magnitude*v/speed;
 return isfinite(x)&&isfinite(y)&&isfinite(tau_ratio)&&tau_ratio>=0.;
}

// Existing CPU laws 3/4 evaluated on resident cell states. This kernel only
// supplies scalar capacity to the established reconstructed-capacity (weak)
// face route: it does not claim a coupled SWE--Exner Jacobian. For Soulsby,
// roughness_mode 0 retains configured z0 while mode 1 mirrors FLOW's
// Manning-to-log-law conversion in cutcell_solver.f90:sediment_z0.
extern "C" __global__ void st_capacity_current34(const int nc,const int ng,const double* q,const double* layers,
 const double* manning,const double* d50,const double* d90,const double* rhos,const double* bulk,
 const double* min_depth,const double* z0,const int* roughness_mode,const int* active,int law,
 double rho,double grav,double dry,double target,double calibration,double* capacity,double* availability,
 double* mixture_d90,double* hiding,int* status){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
 status[c]=0;double volume=0.,mix=0.;
 for(int g=0;g<ng;++g){double vg=layers[g*nc+c]/bulk[g];
  if(!isfinite(vg)||vg<0.)status[c]=1;
  volume+=vg;mix=fmax(mix,d90[g]);capacity[g*nc+c]=0.;capacity[(ng+g)*nc+c]=0.;
  hiding[g*nc+c]=1.;}
 mixture_d90[c]=mix;
 if(status[c]||!isfinite(volume)||volume<0.){status[c]=1;return;}
 for(int g=0;g<ng;++g){double vg=layers[g*nc+c]/bulk[g];
  availability[g*nc+c]=volume>0.?vg/volume*fmin(1.,volume/target):0.;}
 if((law!=3&&law!=4)||!active[c]||q[c]<=dry){if(law!=3&&law!=4)status[c]=1;return;}
 double original_h=q[c],original_hu=q[nc+c],original_hv=q[2*nc+c];
 for(int g=0;g<ng;++g){double h=original_h,hu=original_hu,hv=original_hv,wet=1.;
  double minimum=fmax(min_depth[g],dry);
  if(h<minimum&&minimum>dry){double xi=fmax(0.,fmin(1.,(h-dry)/(minimum-dry)));
   wet=xi*xi*(3.-2.*xi);h=minimum*(1.+128.*2.2204460492503131e-16);
   hu=original_hu*h/original_h;hv=original_hv*h/original_h;}
  if(wet<=0.)continue;
  double u=hu/h,v=hv/h,speed=sqrt(u*u+v*v),relative=rhos[g]/rho-1.;
  if(!(isfinite(h)&&isfinite(speed)&&relative>0.&&d50[g]>0.&&grav>0.)){status[c]=1;return;}
  if(speed<=1.e-12)continue;
  double theta=0.,theta_c=0.,magnitude=0.;
  double dstar=pow(grav*relative/1.e-12,1./3.)*d50[g];
  theta_c=.30/(1.+1.2*dstar)+.055*(1.-exp(-.020*dstar));
  if(law==3){
   if(!(isfinite(manning[c])&&manning[c]>0.)){status[c]=1;return;}
   double chezy=pow(h,1./6.)/manning[c];if(speed<1.e-6||chezy<1.e-6)continue;
   theta=(speed/chezy)*(speed/chezy)/(relative*d50[g]);
   if(theta>.05)magnitude=17.*calibration*sqrt(grav*relative*d50[g]*d50[g]*d50[g])*
    pow(theta,1.5)*(1.-.05/theta)*(1.-sqrt(.05/theta));
  }else{
   if(speed<1.e-6||h<.01||h>200.)continue;
   if(roughness_mode[g]!=0&&roughness_mode[g]!=1){status[c]=1;return;}
   double roughness=z0[g];
   if(roughness_mode[g]==1){
    if(!(isfinite(manning[c])&&manning[c]>0.)){status[c]=1;return;}
    double x=.41*pow(h,1./6.)/(sqrt(grav)*manning[c]),e=exp(-x);
    roughness=fmax(2.2250738585072014e-308,h*exp(-1.)*e/(1.-e));
   }
   if(!(isfinite(roughness)&&roughness>0.)){status[c]=1;return;}
   double denominator=1.+log(roughness/h),cdrag=(.41/denominator)*(.41/denominator);
   theta=rho*cdrag*speed*speed/((rhos[g]-rho)*grav*d50[g]);
   if(theta>theta_c)magnitude=calibration*sqrt(grav*relative*d50[g]*d50[g]*d50[g])*
    12.*sqrt(theta+1.e-4)*(theta-theta_c);
  }
  double x=wet*rhos[g]*magnitude*u/speed,y=wet*rhos[g]*magnitude*v/speed;
  capacity[g*nc+c]=x;capacity[(ng+g)*nc+c]=y;
  if(!isfinite(theta)||theta<0.||!isfinite(x)||!isfinite(y)){status[c]=1;return;}
 }
}

extern "C" __global__ void st_capacity_eh(const int nc,const int ng,const double* q,const double* layers,
 const double* manning,const double* d50,const double* d90,const double* rhos,const double* bulk,
 const double* min_depth,const int* active,double rho,double grav,double dry,double target,
 double calibration,double constant_chezy,double* capacity,double* availability,double* mixture_d90,double* hiding,int* status){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
 status[c]=0;double volume=0.,mix=0.;
 for(int g=0;g<ng;++g){double vg=layers[g*nc+c]/bulk[g];
  if(!isfinite(vg)||vg<0.)status[c]=1;
  volume+=vg;mix=fmax(mix,d90[g]);capacity[g*nc+c]=0.;capacity[(ng+g)*nc+c]=0.;
  hiding[g*nc+c]=1.;}
 mixture_d90[c]=mix;
 if(status[c]||!isfinite(volume)||volume<0.){status[c]=1;return;}
 for(int g=0;g<ng;++g){double vg=layers[g*nc+c]/bulk[g];
  availability[g*nc+c]=volume>0.?vg/volume*fmin(1.,volume/target):0.;}
 if(!active[c]||q[c]<=dry)return;
 double original_h=q[c],original_hu=q[nc+c],original_hv=q[2*nc+c];
 if(!(isfinite(constant_chezy)&&constant_chezy>0.)){status[c]=1;return;}
 for(int g=0;g<ng;++g){double h=original_h,hu=original_hu,hv=original_hv,wet=1.;
  double minimum=fmax(min_depth[g],dry);
  if(h<minimum&&minimum>dry){double xi=fmax(0.,fmin(1.,(h-dry)/(minimum-dry)));
   wet=xi*xi*(3.-2.*xi);h=minimum*(1.+128.*2.2204460492503131e-16);
   hu=original_hu*h/original_h;hv=original_hv*h/original_h;}
  if(wet<=0.)continue;
  double u=hu/h,v=hv/h,speed=sqrt(u*u+v*v);if(!isfinite(speed)){status[c]=1;return;}
  if(speed==0.)continue;
  double relative=rhos[g]/rho-1.,chezy=constant_chezy;
  if(!(isfinite(chezy)&&chezy>0.&&relative>0.)){status[c]=1;return;}
  double a=speed/chezy,theta=(a*a)/(relative*d50[g]);
  double magnitude=.05*calibration*(chezy*chezy/grav)*pow(d50[g],1.5)*
    sqrt(grav*relative)*pow(theta,2.5);
  double x=wet*rhos[g]*magnitude*u/speed,y=wet*rhos[g]*magnitude*v/speed;
  capacity[g*nc+c]=x;capacity[(ng+g)*nc+c]=y;
  if(!isfinite(x)||!isfinite(y)){status[c]=1;return;}
 }
}

// TELEMAC-GAIA v8p5r1 formula 3 is intentionally separate from the frozen
// constant-Chezy law-8 kernel above.  These gradients are unlimited LSQ
// source-equation inputs: down=-grad(z), eta_grad=grad(h+z).
extern "C" __global__ void st_gaia_gradients(const int nc,const int na,const double* q,const double* bed,
 const double* matrix,const double* delta,const double* weight,const int* owner,const int* neighbour,
 const int* offset,const int* faces,const int* signs,const int* active,double* down,double* eta_grad){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
 down[c]=0.;down[nc+c]=0.;eta_grad[c]=0.;eta_grad[nc+c]=0.;if(!active[c])return;
 double a11=matrix[c],a12=matrix[nc+c],a22=matrix[2*nc+c],det=matrix[3*nc+c];
 double bz1=0.,bz2=0.,be1=0.,be2=0.;
 for(int k=offset[c];k<offset[c+1];++k){int f=faces[k],n=signs[k]>0?neighbour[f]:owner[f];if(n<0)continue;
  double rx=delta[k],ry=delta[na+k],w=weight[k];
  bz1+=w*rx*(bed[n]-bed[c]);bz2+=w*ry*(bed[n]-bed[c]);
  be1+=w*rx*((bed[n]+q[n])-(bed[c]+q[c]));be2+=w*ry*((bed[n]+q[n])-(bed[c]+q[c]));}
 if(det<=1.e-12*(a11+a22)*(a11+a22))return;
 double dzx=(a22*bz1-a12*bz2)/det,dzy=(a11*bz2-a12*bz1)/det;
 down[c]=-dzx;down[nc+c]=-dzy;
 eta_grad[c]=(a22*be1-a12*be2)/det;eta_grad[nc+c]=(a11*be2-a12*be1)/det;
}

__device__ bool st_gaia_engel_cc(double tau,double cf,double h,double d,double rhos,double rho,
 double grav,double skin_ratio,double& qmass,double& theta_cc,double& mu){
 qmass=0.;theta_cc=0.;mu=0.;
 if(!(isfinite(tau)&&isfinite(cf)&&isfinite(h)&&isfinite(d)&&isfinite(rhos)&&isfinite(rho)&&
      isfinite(grav)&&isfinite(skin_ratio))||tau<0.||cf<=0.||h<0.||d<=0.||rhos<=rho||
      rho<=0.||grav<=0.||skin_ratio<=0.)return false;
 double ksp=skin_ratio*d;
 if(h>ksp){double aa=2.5*log(11.036*fmax(h,ksp)/ksp);if(!(isfinite(aa)&&aa>0.))return false;
  double cfskin=2./(aa*aa);mu=cfskin/cf;}
 double tetap=tau*mu/((rhos-rho)*grav*d)+1.e-6;
 if(tetap<=.06)theta_cc=0.;
 else if(tetap<.384)theta_cc=sqrt(2.5*(tetap-.06));
 else if(tetap<1.080)theta_cc=1.066*pow(tetap,.176);
 else theta_cc=tetap;
 qmass=.1*rhos*sqrt((rhos/rho-1.)*grav*d*d*d)*pow(theta_cc,2.5)/fmax(cf,1.e-6);
 return isfinite(qmass)&&isfinite(theta_cc)&&isfinite(mu)&&qmass>=0.&&theta_cc>=0.&&mu>=0.;
}

__device__ void st_gaia_mass_jacobian(double h,double u,double v,double n,double d,
 double rhos,double rho,double g,double skin_ratio,double hmin,double hdry,double* derivative){
 derivative[0]=0.;derivative[1]=0.;derivative[2]=0.;
 if(h<=hdry||n<=0.||d<=0.||rhos<=rho||rho<=0.||g<=0.||skin_ratio<=0.)return;
 double velocity=sqrt(u*u+v*v);if(velocity<=0.)return;
 double heff=h;if(h<hmin&&hmin>hdry)heff=hmin*(1.+128.*2.2204460492503131e-16);
 double ksp=skin_ratio*d;if(heff<=ksp)return;
 double aa=2.5*log(11.036*heff/ksp);if(aa<=0.)return;
 double cf=2.*g*n*n/pow(heff,1./3.),cfskin=2./(aa*aa);
 double tetap=.5*rho*cfskin*velocity*velocity/((rhos-rho)*g*d)+1.e-6;
 if(tetap<=.06)return;
 double theta,dlogtheta;
 if(tetap<.384){theta=sqrt(2.5*(tetap-.06));dlogtheta=.5/(tetap-.06);}
 else if(tetap<1.080){theta=1.066*pow(tetap,.176);dlogtheta=.176/tetap;}
 else{theta=tetap;dlogtheta=1./tetap;}
 double capacity=.1*rhos*sqrt((rhos/rho-1.)*g*d*d*d)*pow(theta,2.5)/fmax(cf,1.e-6);
 double capacity_speed=capacity*2.5*dlogtheta*2.*(tetap-1.e-6)/velocity;
 double capacity_h=0.;
 if(h>=hmin||hmin<=hdry){capacity_h=capacity*2.5*dlogtheta*(-5.*(tetap-1.e-6)/(aa*heff));
  if(cf>1.e-6)capacity_h+=capacity/(3.*heff);}
 double wet=1.,dwet=0.;
 if(h<hmin&&hmin>hdry){double xi=fmax(0.,fmin(1.,(h-hdry)/(hmin-hdry)));
  wet=xi*xi*(3.-2.*xi);dwet=6.*xi*(1.-xi)/(hmin-hdry);}
 double total=wet*capacity,total_h=wet*capacity_h+dwet*capacity,total_speed=wet*capacity_speed;
 double normal_fraction=u/velocity;
 derivative[0]=normal_fraction*total_h-u*total_speed/h;
 derivative[1]=(total_speed*normal_fraction*normal_fraction+total*v*v/(velocity*velocity*velocity))/h;
 derivative[2]=u*v*(total_speed/(velocity*velocity)-total/(velocity*velocity*velocity))/h;
}

__device__ int st_coupled_upwind_sign(double mean_u,double selected_lambda){
 // The selected cubic root is only the opposite-going wave when it actually
 // opposes the tangential/mean-flow eigenvalue.  If even the minimum
 // (positive flow) or maximum (negative flow) root has the same sign, every
 // characteristic is one-way and sign(A) is exactly +/-I.
 if(mean_u>=0.&&selected_lambda>=0.)return 1;
 if(mean_u<0.&&selected_lambda<=0.)return -1;
 return 0;
}

__device__ bool st_gaia_open_capacity(double h,double hu,double hv,double manning,double d,
 double rhos,double rho,double grav,double dry,double minimum,double skin_ratio,double secondary_alpha,
 double eta_x,double eta_y,double& x,double& y,double& tau){
 x=0.;y=0.;tau=0.;if(h<=dry)return true;double wet=1.;minimum=fmax(minimum,dry);
 if(h<minimum&&minimum>dry){double xi=fmax(0.,fmin(1.,(h-dry)/(minimum-dry)));wet=xi*xi*(3.-2.*xi);
  double next=minimum*(1.+128.*2.2204460492503131e-16);hu*=next/h;hv*=next/h;h=next;}
 double u=hu/h,v=hv/h,speed=sqrt(u*u+v*v);if(!(isfinite(speed)&&manning>0.&&secondary_alpha>0.))return false;
 if(speed<=1.e-12)return true;double chezy=pow(h,1./6.)/manning,cf=2.*grav/(chezy*chezy);
 tau=.5*rho*cf*speed*speed;double qmass=0.,theta=0.,mu=0.;
 if(!st_gaia_engel_cc(tau,cf,h,d,rhos,rho,grav,skin_ratio,qmass,theta,mu))return false;
 double sx=tau*u/speed,sy=tau*v/speed,cross=(u*eta_y-v*eta_x)/(speed*speed);
 double cc=cross*(7./secondary_alpha*rho*grav)*h*cf/2.;sx+=cc*v;sy-=cc*u;
 if(tau>0.){x=wet*qmass*sx/tau;y=wet*qmass*sy/tau;}
 return isfinite(x)&&isfinite(y)&&isfinite(tau);
}

extern "C" __global__ void st_capacity_gaia(const int nc,const int ng,const double* q,const double* layers,
 const double* manning,const double* d50,const double* d90,const double* rhos,const double* bulk,
 const double* min_depth,const int* active,const double* down,const double* eta_grad,
 double rho,double grav,double dry,double target,double skin_ratio,double slope_beta,
 double deviation_beta2,double secondary_alpha,double* base_capacity,double* capacity,double* availability,
 double* mixture_d90,double* hiding,int* status){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
 status[c]=0;double volume=0.,mix=0.;
 for(int g=0;g<ng;++g){double vg=layers[g*nc+c]/bulk[g];if(!isfinite(vg)||vg<0.)status[c]=1;
   volume+=vg;mix=fmax(mix,d90[g]);base_capacity[g*nc+c]=0.;base_capacity[(ng+g)*nc+c]=0.;
   capacity[g*nc+c]=0.;capacity[(ng+g)*nc+c]=0.;hiding[g*nc+c]=1.;}
 mixture_d90[c]=mix;if(status[c]||!isfinite(volume)||volume<0.){status[c]=1;return;}
 for(int g=0;g<ng;++g){double vg=layers[g*nc+c]/bulk[g];
  availability[g*nc+c]=volume>0.?vg/volume*fmin(1.,volume/target):0.;}
 if(!active[c]||q[c]<=dry)return;
 if(!(isfinite(manning[c])&&manning[c]>0.&&skin_ratio>0.&&slope_beta>=0.&&
      deviation_beta2>0.&&secondary_alpha>0.)){status[c]=1;return;}
 double original_h=q[c],original_hu=q[nc+c],original_hv=q[2*nc+c];
 for(int g=0;g<ng;++g){double h=original_h,hu=original_hu,hv=original_hv,wet=1.;
  double minimum=fmax(min_depth[g],dry);if(h<minimum&&minimum>dry){double xi=fmax(0.,fmin(1.,(h-dry)/(minimum-dry)));
   wet=xi*xi*(3.-2.*xi);h=minimum*(1.+128.*2.2204460492503131e-16);hu=original_hu*h/original_h;hv=original_hv*h/original_h;}
  if(wet<=0.)continue;double u=hu/h,v=hv/h,speed=sqrt(u*u+v*v);if(!isfinite(speed)){status[c]=1;return;}
  if(speed<=1.e-12)continue;double chezy=pow(h,1./6.)/manning[c];
  if(!(isfinite(chezy)&&chezy>0.)){status[c]=1;return;}
  double cf=2.*grav/(chezy*chezy),tau=.5*rho*cf*speed*speed,qmass=0.,theta=0.,mu=0.;
  if(!st_gaia_engel_cc(tau,cf,h,d50[g],rhos[g],rho,grav,skin_ratio,qmass,theta,mu)){status[c]=1;return;}
   if(qmass<=0.||tau<=0.)continue;double fx=u/speed,fy=v/speed;
   base_capacity[g*nc+c]=wet*qmass*fx;base_capacity[(ng+g)*nc+c]=wet*qmass*fy;
   double gzx=-down[c],gzy=-down[nc+c];
  double tt1=sqrt(((rhos[g]-rho)*grav*d50[g]/(deviation_beta2*deviation_beta2))/fmax(tau,1.e-10));
  double sx=fx-tt1*gzx,sy=fy-tt1*gzy,snorm=sqrt(sx*sx+sy*sy);if(snorm<=0.){status[c]=1;return;}
  double coef=fmax(0.,1.-slope_beta*(gzx*fx+gzy*fy));sx=tau*coef*sx/snorm;sy=tau*coef*sy/snorm;
  double cross=(u*eta_grad[nc+c]-v*eta_grad[c])/(speed*speed);
  double cc=cross*(7./secondary_alpha*rho*grav)*h*cf/2.;sx+=cc*v;sy-=cc*u;
  double x=wet*qmass*sx/tau,y=wet*qmass*sy/tau;capacity[g*nc+c]=x;capacity[(ng+g)*nc+c]=y;
  if(!isfinite(x)||!isfinite(y)){status[c]=1;return;}
 }
}

// Accepted-output-only diagnostic for the GAIA model-3 stress composition.
// The production capacity kernel above remains free of diagnostic writes.  This
// kernel consumes its retained base/corrected capacities and independently
// reconstructs the constitutive stress decomposition at an output clock.
// Field order is declared by GAIA_DIAGNOSTIC_FIELDS in sediment_transport.py.
extern "C" __global__ void st_gaia_diagnostic(const int nc,const int ng,const double* q,
 const double* manning,const double* d50,const double* rhos,const double* min_depth,
 const int* active,const double* down,const double* eta_grad,double rho,double grav,double dry,
 double slope_beta,double deviation_beta2,double secondary_alpha,const double* base_capacity,
 const double* corrected_capacity,double* diagnostic,int* status){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
 status[c]=0;
 const int fields=19;
 for(int k=0;k<fields;++k)for(int g=0;g<ng;++g)diagnostic[(k*ng+g)*nc+c]=0.;
 if(!(isfinite(q[c])&&isfinite(q[nc+c])&&isfinite(q[2*nc+c])&&
      isfinite(down[c])&&isfinite(down[nc+c])&&isfinite(eta_grad[c])&&
      isfinite(eta_grad[nc+c])&&isfinite(rho)&&isfinite(grav)&&isfinite(dry)&&
      isfinite(slope_beta)&&isfinite(deviation_beta2)&&isfinite(secondary_alpha))||
      rho<=0.||grav<=0.||dry<0.||slope_beta<0.||deviation_beta2<=0.||secondary_alpha<=0.){
  status[c]=1;return;
 }
 double original_h=q[c],original_hu=q[nc+c],original_hv=q[2*nc+c];
 for(int g=0;g<ng;++g){
  #define GD(field) diagnostic[((field)*ng+g)*nc+c]
  GD(1)=eta_grad[c];GD(2)=eta_grad[nc+c];GD(3)=-down[c];GD(4)=-down[nc+c];
  double bx=base_capacity[g*nc+c],by=base_capacity[(ng+g)*nc+c];
  double qx=corrected_capacity[g*nc+c],qy=corrected_capacity[(ng+g)*nc+c];
  GD(14)=bx;GD(15)=by;GD(16)=qx;GD(17)=qy;
  if(!(isfinite(bx)&&isfinite(by)&&isfinite(qx)&&isfinite(qy)&&isfinite(manning[c])&&
       isfinite(d50[g])&&isfinite(rhos[g])&&isfinite(min_depth[g]))||manning[c]<=0.||
       d50[g]<=0.||rhos[g]<=rho||min_depth[g]<0.){status[c]=1;continue;}
  if(!active[c]||original_h<=dry)continue;
  double h=original_h,hu=original_hu,hv=original_hv,minimum=fmax(min_depth[g],dry);
  if(h<minimum&&minimum>dry){double next=minimum*(1.+128.*2.2204460492503131e-16);
   hu*=next/h;hv*=next/h;h=next;}
  double u=hu/h,v=hv/h,speed2=u*u+v*v,speed=sqrt(speed2);
  if(!(isfinite(speed)&&isfinite(h))){status[c]=1;continue;}
  if(speed<=1.e-12)continue;
  double chezy=pow(h,1./6.)/manning[c];
  if(!(isfinite(chezy)&&chezy>0.)){status[c]=1;continue;}
  double cf=2.*grav/(chezy*chezy),tau=.5*rho*cf*speed2;GD(5)=tau;
  if(bx*bx+by*by<=0.||tau<=0.)continue;
  double fx=u/speed,fy=v/speed,gzx=-down[c],gzy=-down[nc+c];
  double tt1=sqrt(((rhos[g]-rho)*grav*d50[g]/(deviation_beta2*deviation_beta2))/fmax(tau,1.e-10));
  double sx=fx-tt1*gzx,sy=fy-tt1*gzy,snorm=sqrt(sx*sx+sy*sy);
  if(!(isfinite(snorm)&&snorm>0.)){status[c]=1;continue;}
  double factor=fmax(0.,1.-slope_beta*(gzx*fx+gzy*fy));
  sx=tau*factor*sx/snorm;sy=tau*factor*sy/snorm;GD(6)=sx;GD(7)=sy;GD(18)=factor;
  double cross=(u*eta_grad[nc+c]-v*eta_grad[c])/speed2;
  double cc=cross*(7./secondary_alpha*rho*grav)*h*cf/2.;
  double tx=sx+cc*v,ty=sy-cc*u,dx=tx-sx,dy=ty-sy;
  GD(8)=dx;GD(9)=dy;GD(10)=tx;GD(11)=ty;
  double stress2=tx*tx+ty*ty,slope2=sx*sx+sy*sy;
  if(stress2>0.&&slope2>0.)GD(12)=atan2(sx*ty-sy*tx,sx*tx+sy*ty);
  if(stress2>0.)GD(13)=atan2(u*ty-v*tx,u*tx+v*ty);
  GD(0)=1.;
  double qmag=sqrt(bx*bx+by*by),rx=qmag*tx/tau,ry=qmag*ty/tau;
  double scale=fmax(1.e-20,fmax(fmax(fabs(rx),fabs(ry)),fmax(fabs(qx),fabs(qy))));
  if(fmax(fabs(rx-qx),fabs(ry-qy))>256.*2.2204460492503131e-16*scale)status[c]=1;
  if(!isfinite(GD(12))||!isfinite(GD(13))||!isfinite(rx)||!isfinite(ry))status[c]=1;
  #undef GD
 }
}

extern "C" __global__ void st_gradients(const int nc,const int nf,const int na,const int ng,
 const double* capacity,const double* bed,const double* matrix,const double* delta,const double* weight,
 const double* xy,const double* mid,const int* owner,const int* neighbour,const int* offset,
 const int* faces,const int* signs,const int* active,double* gradient,double* down){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
 for(int j=0;j<4*ng;++j)gradient[j*nc+c]=0.;down[c]=0.;down[nc+c]=0.;if(!active[c])return;
 double a11=matrix[c],a12=matrix[nc+c],a22=matrix[2*nc+c],det=matrix[3*nc+c],b1=0.,b2=0.;
 for(int k=offset[c];k<offset[c+1];++k){int f=faces[k],n=signs[k]>0?neighbour[f]:owner[f];if(n<0)continue;
  b1+=weight[k]*delta[k]*(bed[n]-bed[c]);b2+=weight[k]*delta[na+k]*(bed[n]-bed[c]);}
 if(det<=1.e-12*(a11+a22)*(a11+a22))return;
 down[c]=-(a22*b1-a12*b2)/det;down[nc+c]=-(a11*b2-a12*b1)/det;
 for(int g=0;g<ng;++g)for(int m=0;m<2;++m){int a=(m*ng+g)*nc;double lo=capacity[a+c],hi=lo; b1=0.;b2=0.;
  for(int k=offset[c];k<offset[c+1];++k){int f=faces[k],n=signs[k]>0?neighbour[f]:owner[f];if(n<0)continue;
   b1+=weight[k]*delta[k]*(capacity[a+n]-capacity[a+c]);b2+=weight[k]*delta[na+k]*(capacity[a+n]-capacity[a+c]);
   lo=fmin(lo,capacity[a+n]);hi=fmax(hi,capacity[a+n]);}
  double gx=(a22*b1-a12*b2)/det,gy=(a11*b2-a12*b1)/det,theta=1.;
  for(int k=offset[c];k<offset[c+1];++k){int f=faces[k],n=signs[k]>0?neighbour[f]:owner[f];if(n<0)continue;
   double d=gx*(mid[f]-xy[c])+gy*(mid[nf+f]-xy[nc+c]);
   if(d>0.)theta=fmin(theta,(hi-capacity[a+c])/d);if(d<0.)theta=fmin(theta,(lo-capacity[a+c])/d);}
  theta=fmax(0.,fmin(1.,theta));gradient[((m*2)*ng+g)*nc+c]=gx*theta;gradient[((m*2+1)*ng+g)*nc+c]=gy*theta;
 }
}

__device__ void st_face_value(int c,int n,int side,int f,int g,int nc,int nf,int ng,
 const double* cap,const double* grad,const double* disp,const int* weno,const int* obdist,
 bool consistent,const double* xy,const double* mid,const double* width,
 const int* owner,const int* neighbour,const int* offset,const int* faces,const int* signs,
 double nx,double ny,double& x,double& y){
 x=cap[g*nc+c];y=cap[(ng+g)*nc+c];if(n<0)return;
 if(obdist[c]<=2&&!consistent){x=.5*(x+cap[g*nc+n]);y=.5*(y+cap[(ng+g)*nc+n]);return;}
 double value[2]={x,y};
 for(int m=0;m<2;++m){if(weno[side*nf+f]>=0){double stencil[5];
   for(int k=0;k<5;++k)stencil[k]=cap[(m*ng+g)*nc+weno[(k*2+side)*nf+f]];value[m]=st_weno(stencil);
  }else{value[m]+=grad[((m*2)*ng+g)*nc+c]*disp[side*nf+f]+grad[((m*2+1)*ng+g)*nc+c]*disp[(2+side)*nf+f];}}
 if(consistent&&width[c]>0.&&width[nc+c]>0.){
  double dx=mid[f]-xy[c],dy=mid[nf+f]-xy[nc+c];int a=-1,bb=-1;
  double r1x=0.,r1y=0.;
  for(int k=offset[c];k<offset[c+1];++k){int ff=faces[k],other=signs[k]>0?neighbour[ff]:owner[ff];
   if(other<0||obdist[other]<=obdist[c])continue;
   double rx=xy[other]-xy[c],ry=xy[nc+other]-xy[nc+c],rr=rx*rx+ry*ry;
   if(fabs(rx*dy-ry*dx)>1.e-8*rr)continue;
   if(fmax(fabs(width[other]-width[c]),fabs(width[nc+other]-width[nc+c]))>
      1.e-8*fmax(width[c],width[nc+c]))continue;
   a=other;r1x=rx;r1y=ry;break;}
  if(a>=0)for(int k=offset[a];k<offset[a+1];++k){int ff=faces[k],other=signs[k]>0?neighbour[ff]:owner[ff];
   if(other<0)continue;
   double rx=xy[other]-xy[a],ry=xy[nc+other]-xy[nc+a];
   if(fmax(fabs(rx-r1x),fabs(ry-r1y))>1.e-8*sqrt(r1x*r1x+r1y*r1y))continue;
   if(fmax(fabs(width[other]-width[c]),fabs(width[nc+other]-width[nc+c]))>
      1.e-8*fmax(width[c],width[nc+c]))continue;
   bb=other;break;}
  if(bb>=0){double s=(dx*r1x+dy*r1y)/(r1x*r1x+r1y*r1y);
   for(int m=0;m<2;++m){int base=(m*ng+g)*nc;
    double curvature=.5*(cap[base+bb]-2.*cap[base+a]+cap[base+c]);
    value[m]=cap[base+c]+s*(cap[base+a]-cap[base+c]-curvature)+(s*s-1./12.)*curvature;}}
 }
 double an=cap[g*nc+c]*nx+cap[(ng+g)*nc+c]*ny,bn=cap[g*nc+n]*nx+cap[(ng+g)*nc+n]*ny;
 double at=-cap[g*nc+c]*ny+cap[(ng+g)*nc+c]*nx,bt=-cap[g*nc+n]*ny+cap[(ng+g)*nc+n]*nx;
 double vn=fmax(fmin(an,bn),fmin(fmax(an,bn),value[0]*nx+value[1]*ny));
 double vt=fmax(fmin(at,bt),fmin(fmax(at,bt),-value[0]*ny+value[1]*nx));
 x=vn*nx-vt*ny;y=vn*ny+vt*nx;
}

extern "C" __global__ void st_gaia_base_faces(int nc,int nf,int ng,const double* capacity,
 const double* gradient,const double* xy,const double* mid,const double* width,
 const double* normal,const double* disp,const int* weno,const int* obdist,
 const int* owner,const int* neighbour,const int* offset,const int* faces,const int* signs,
 double* face_left,double* face_right){
 int f=blockIdx.x*blockDim.x+threadIdx.x;if(f>=nf)return;
 int c=owner[f],n=neighbour[f];if(n<0)return;
 double nx=normal[f],ny=normal[nf+f];
 for(int g=0;g<ng;++g){double lx,ly,rx,ry;
  st_face_value(c,n,0,f,g,nc,nf,ng,capacity,gradient,disp,weno,obdist,false,
   xy,mid,width,owner,neighbour,offset,faces,signs,nx,ny,lx,ly);
  st_face_value(n,c,1,f,g,nc,nf,ng,capacity,gradient,disp,weno,obdist,false,
   xy,mid,width,owner,neighbour,offset,faces,signs,nx,ny,rx,ry);
  int at=g*nf+f;face_left[at]=lx;face_left[ng*nf+at]=ly;
  face_right[at]=rx;face_right[ng*nf+at]=ry;
 }
}

__device__ bool st_mpm_tau_ratio(double h,double hu,double hv,double manning,
 double d50,double mix_d90,double rhos,double rho,double grav,double critical,
 double hiding,double cf,int friction_mode,double min_depth,double dry,double& ratio){
 ratio=0.;if(h<=dry)return true;
 double minimum=fmax(min_depth,dry);
 if(h<minimum&&minimum>dry){double next=minimum*(1.+128.*2.2204460492503131e-16);
  hu*=next/h;hv*=next/h;h=next;}
 double u=hu/h,v=hv/h,speed=sqrt(u*u+v*v);if(!isfinite(h)||!isfinite(speed))return false;
 if(speed==0.)return true;
 double relative=rhos/rho-1.,shear=0.;
 if(friction_mode==1){double chezy=pow(h,1./6.)/manning;
  double grain=18.*log10(12.*fmax(h/mix_d90,1.));
  if(!(isfinite(chezy)&&chezy>0.&&grain>0.))return false;
  double ripple=fmin(pow(chezy/grain,1.5),1.);
  double a=speed/chezy;shear=ripple*(a*a)/(relative*d50);
 }else shear=cf*speed*speed/(grav*relative*d50);
 double threshold=critical*hiding;
 ratio=threshold>0.?shear/threshold:(shear>0.?1.7976931348623157e308:0.);
 return isfinite(ratio)&&ratio>=0.;
}

__device__ void st_slope(double& x,double& y,double sx,double sy,double tau_ratio,
 double alfabs,double alfabn,double angle){
 double magnitude=sqrt(x*x+y*y);if(magnitude<=1.e-6)return;
 double ax=x/magnitude,ay=y/magnitude,dx=-ay,dy=ax,ds=sx*ax+sy*ay,dp=sx*dx+sy*dy;
 double phi=tan(angle*acos(-1.)/180.);ds=fmin(.9*phi,ds);
 double bagnold=phi/(cos(atan(ds))*(phi-ds));double along=fmax(0.,1.+alfabs*(bagnold-1.));
 double transverse=alfabn*dp;if(tau_ratio>=1.)transverse/=sqrt(tau_ratio);
 x=along*magnitude*(ax+transverse*dx);y=along*magnitude*(ay+transverse*dy);
}

__device__ bool st_eh_shields(double h,double hu,double hv,double chezy,
 double d50,double rhos,double rho,double dry,double& theta){
 theta=0.;if(h<=dry)return true;
 if(!(isfinite(h)&&isfinite(hu)&&isfinite(hv)&&isfinite(chezy))||chezy<=0.)return false;
 double u=hu/h,v=hv/h,speed=sqrt(u*u+v*v),relative=rhos/rho-1.;
 if(!(isfinite(speed)&&isfinite(chezy)&&chezy>0.&&relative>0.&&d50>0.))return false;
 double a=speed/chezy;theta=(a*a)/(relative*d50);
 return isfinite(theta)&&theta>=0.;
}

__device__ bool st_slope_stress(double h,double hu,double hv,double manning,
 double d50,double mix_d90,double rhos,double rho,double grav,double critical,
 double hiding,double cf,int friction_mode,double min_depth,double dry,int slope,
 double capacity_x,double capacity_y,double& stress,
 double steering_hu=0.,double steering_hv=0.){
 // FLOW's positive EH/MPM eqtran branches retain erosed's zero taurat.
 // Mode 5 records that contract explicitly; legacy mode 2 is unchanged.
 stress=0.;if(slope==5||h<=dry)return true;
 if(slope==2&&friction_mode!=2)return st_mpm_tau_ratio(h,hu,hv,manning,
   d50,mix_d90,rhos,rho,grav,critical,hiding,cf,friction_mode,min_depth,dry,stress);
 double minimum=fmax(min_depth,dry);
 if(h<minimum&&minimum>dry){double next=minimum*(1.+128.*2.2204460492503131e-16);
  hu*=next/h;hv*=next/h;h=next;}
 double chezy=0.;
 if(friction_mode==2)chezy=cf;
 else if(friction_mode==1)chezy=pow(h,1./6.)/manning;
 else if(cf>0.)chezy=sqrt(grav/cf);
 else return true;
 if(!st_eh_shields(h,hu,hv,chezy,d50,rhos,rho,dry,stress))return false;
 if(slope==2){
  // EH has no capacity threshold. The explicitly chosen legacy ratio uses
  // FLOW initsedtra's physical grain critical stress, not MPM critical.
  double dstar=d50*pow((rhos/rho-1.)*grav/1.e-12,.3333),threshold;
  if(dstar<=4.)threshold=.24/dstar;
  else if(dstar<=10.)threshold=.14/pow(dstar,.64);
  else if(dstar<=20.)threshold=.04/pow(dstar,.1);
  else if(dstar<=150.)threshold=.013*pow(dstar,.29);
  else threshold=.055;
  stress/=threshold;
 }else if(slope==3){
  // eqtran's spiral rotation preserves capacity magnitude but increases
  // ust2 by sqrt(1+alphaspir^2). Recover that factor from the resident
  // cell capacity direction and velocity without a second curvature solve.
  if(steering_hu==0.&&steering_hv==0.){steering_hu=hu;steering_hv=hv;}
  double speed=sqrt(steering_hu*steering_hu+steering_hv*steering_hv),magnitude=sqrt(capacity_x*capacity_x+capacity_y*capacity_y);
  if(speed>0.&&magnitude>0.){
   double cosine=(capacity_x*steering_hu+capacity_y*steering_hv)/(magnitude*speed);
   if(!(isfinite(cosine)&&cosine>0.))return false;
   stress/=fmin(cosine,1.);
  }
 }
 return isfinite(stress)&&stress>=0.;
}

__device__ bool st_eh_capacity(double h,double hu,double hv,double chezy,double d50,
 double rhos,double rho,double grav,double dry,double minimum,double& x,double& y,double& theta){
 x=0.;y=0.;theta=0.;if(h<=dry)return true;
 if(!(isfinite(h)&&isfinite(hu)&&isfinite(hv)&&isfinite(chezy))||chezy<=0.)return false;
 double wet=1.;if(h<minimum&&minimum>dry){double xi=fmax(0.,fmin(1.,(h-dry)/(minimum-dry)));
  wet=xi*xi*(3.-2.*xi);}
 double u=hu/h,v=hv/h,speed=sqrt(u*u+v*v),relative=rhos/rho-1.;
 if(!(isfinite(speed)&&relative>0.&&d50>0.&&grav>0.))return false;if(speed==0.)return true;
 double a=speed/chezy;theta=(a*a)/(relative*d50);
 double magnitude=.05*(chezy*chezy/grav)*pow(d50,1.5)*sqrt(grav*relative)*pow(theta,2.5);
 x=wet*rhos*magnitude*u/speed;y=wet*rhos*magnitude*v/speed;
 return isfinite(x)&&isfinite(y)&&isfinite(theta)&&theta>=0.;
}

__device__ void st_koch_flokstra(double& x,double& y,double sx,double sy,double shield,
 double alfabs,double ashld,double bshld){
 // Delft3D ISlope=3 with CShld=DShld=0, the standard Struiksma T2 setup.
 // sx/sy are positive downhill, matching dzduu/dzdvv in adjust_bedload.
 double magnitude=sqrt(x*x+y*y);if(magnitude<=1.e-6)return;
 double ax=x/magnitude,ay=y/magnitude,dzds=sx*ax+sy*ay;
 double ftheta=shield>0.?ashld*pow(shield,bshld):0.;
 double vx=ax,vy=ay;
 if(sx!=0.||sy!=0.){vx=ftheta*ax+sx;vy=ftheta*ay+sy;}
 double norm=sqrt(vx*vx+vy*vy);if(norm<=0.)return;
 magnitude*=1.+alfabs*dzds;
 x=magnitude*vx/norm;y=magnitude*vy/norm;
}

// Isolated Delft3D FLOW ISlope=4 constitutive operator.  This is deliberately
// not connected to the public bed_slope dispatch: BRAIDS mode 4 remains the
// qualified GAIA Talmon bundle until the full Parker-Andrews route is admitted.
// sx/sy are dimensionless bed gradients positive downhill, matching the other
// FLOW slope operators in this file.  parker_alpha is FLOW's ALFPA, not the
// Parker-KLM hiding exponent.
__device__ bool st_parker_andrews(double& x,double& y,double sx,double sy,
 double shield,double hiding,double alfabs,double parker_alpha,double critical){
 if(!(isfinite(x)&&isfinite(y)&&isfinite(sx)&&isfinite(sy)&&isfinite(shield)&&
      isfinite(hiding)&&isfinite(alfabs)&&isfinite(parker_alpha)&&isfinite(critical))||
    shield<0.||hiding<0.||alfabs<0.||parker_alpha<0.||critical<0.)return false;
 double magnitude=sqrt(x*x+y*y);if(magnitude<=1.e-6)return true;
 double ax=x/magnitude,ay=y/magnitude,dzds=sx*ax+sy*ay;
 // Locked adjust_bedload sets phi=30 degrees and caps dzds for every ISlope.
 dzds=fmin((double).9f*tan((double)(30.f/180.f)*acos(-1.)),dzds);
 if(sx==0.&&sy==0.)return true;
 double ftheta=0.;
 if(shield!=0.)ftheta=parker_alpha*sqrt(shield/fmax(.1*shield,hiding*critical));
 double vx=ftheta*ax+sx,vy=ftheta*ay+sy,norm=sqrt(vx*vx+vy*vy);
 if(norm<=2.2250738585072014e-308)return false;
 magnitude*=1.+alfabs*dzds;x=magnitude*vx/norm;y=magnitude*vy/norm;
 return isfinite(x)&&isfinite(y);
}

// Native model-0 coupled SWE/Exner stage outlet uses the *hydraulic face
// trace*, not lagged cell-centre capacity, for local-equilibrium export.
// The MPM formula is the same as st_capacity, evaluated for one grain at the
// already selected boundary hydraulic trace. This routine is not called by
// prognostic/equilibrium secondary-flow branches.
__device__ bool st_mpm_trace_capacity(double h,double hu,double hv,double manning,
 double d50,double mix_d90,double rhos,double rho,double grav,double critical,
 double hiding,double cf,int friction_mode,double min_depth,double dry,double& x,double& y,
 double& tau_ratio){
 x=0.;y=0.;tau_ratio=0.;if(h<=dry)return true;
 if(!st_mpm_tau_ratio(h,hu,hv,manning,d50,mix_d90,rhos,rho,grav,critical,
     hiding,cf,friction_mode,min_depth,dry,tau_ratio))return false;
 double minimum=fmax(min_depth,dry),wet=1.;
 if(h<minimum&&minimum>dry){double xi=fmax(0.,fmin(1.,(h-dry)/(minimum-dry)));
  wet=xi*xi*(3.-2.*xi);double new_h=minimum*(1.+128.*2.2204460492503131e-16);
  hu*=new_h/h;hv*=new_h/h;h=new_h;}
 if(wet<=0.)return true;
 double u=hu/h,v=hv/h,speed=sqrt(u*u+v*v),chezy=0.,grain=0.,ripple=0.;
 if(friction_mode==1){chezy=pow(h,1./6.)/manning;grain=18.*log10(12.*fmax(h/mix_d90,1.));
  if(!(isfinite(chezy)&&chezy>0.&&grain>0.))return false;
  ripple=fmin(pow(chezy/grain,1.5),1.);}
 if(!(isfinite(h)&&isfinite(speed)))return false;
 if(speed==0.)return true;
 double relative=rhos/rho-1.,excess;
 if(friction_mode==1){double a=speed/chezy;
  double theta=(a*a)/(relative*d50);excess=fmax(ripple*theta-critical*hiding,0.);}
 else excess=fmax(0.,cf*speed*speed/(grav*relative*d50)-critical*hiding);
 double magnitude=8.*sqrt(grav*relative*(d50*d50*d50))*pow(excess,1.5);
 x=wet*rhos*magnitude*u/speed;y=wet*rhos*magnitude*v/speed;
 return isfinite(x)&&isfinite(y);
}

__device__ void st_mpm_mass_jacobian(double h,double u,double v,double n,double d50,
 double d90,double relative,double rhos,double g,double critical,double hiding,
 double cf,int friction_mode,double hmin,double hdry,double* derivative){
 derivative[0]=0.;derivative[1]=0.;derivative[2]=0.;if(h<=hdry)return;
 double heff=fmax(h,hmin),velocity=sqrt(u*u+v*v);if(velocity<=0.)return;
 double alpha,dalpha;
 if(friction_mode==1){double chezy=pow(heff,1./6.)/n,grain=18.*log10(12.*fmax(heff/d90,1.));
  double mu=fmin(pow(chezy/grain,1.5),1.);alpha=mu/(chezy*chezy*relative*d50);
  if(mu>=1.)dalpha=-alpha/(3.*heff);
  else{dalpha=-alpha/(12.*heff);
   if(heff>d90)dalpha-=1.5*alpha*18./(log(10.)*heff*grain);}}
 else{alpha=cf/(g*relative*d50);dalpha=0.;}
 if(h<hmin)dalpha=0.;
 double excess=alpha*velocity*velocity-critical*hiding;if(excess<=0.)return;
 double root=sqrt(excess),k=rhos*8.*sqrt(g*relative*d50*d50*d50);
 double capacity=k*excess*root;
 derivative[0]=1.5*k*root*velocity*u*(dalpha-2.*alpha/h);
 derivative[1]=3.*k*alpha*root*u*u/(h*velocity)+capacity*(1.-u*u/(velocity*velocity))/(h*velocity);
 derivative[2]=3.*k*alpha*root*u*v/(h*velocity)-capacity*u*v/(h*velocity*velocity*velocity);
 if(h<hmin&&hmin>hdry){double xi=(h-hdry)/(hmin-hdry),weight=xi*xi*(3.-2.*xi),dweight=6.*xi*(1.-xi)/(hmin-hdry);
  for(int k=0;k<3;++k)derivative[k]*=weight;
  derivative[0]+=dweight*capacity*u/velocity;}
}

extern "C" __global__ void st_coupled_stage(int nc,int nf,int ng,int nb,const double* q,
 const double* face_value,const double* bed,const double* normal,const int* owner,
 const int* neighbour,const int* bid,const int* mode,const double* bchydro,
 const double* manning,const double* d50,const double* rhos,const double* bulk,
 const double* mindepth,const double* mixture_d90,const double* hiding,
 const double* availability,double rho,double grav,double critical,double cf,int friction_mode,
 double dry,double morfac,double* trace,int* status,int* applied){
 int f=blockIdx.x*blockDim.x+threadIdx.x;if(f>=nf)return;status[f]=0;applied[f]=0;
 int c=owner[f],b=bid[f];if(neighbour[f]>=0||b<0||mode[b]!=2||q[c]<=dry)return;
 double nx=normal[f],ny=normal[nf+f],qn=q[nc+c]*nx+q[2*nc+c]*ny;
 if(qn<=0.||qn>=q[c]*sqrt(grav*q[c]))return;
 double h=trace[8*nf+f],zleft=face_value[6*nf+f];if(h<=dry||morfac<=0.)return;
 double hu=h*face_value[2*nf+f],hv=h*face_value[4*nf+f];
 double u=(hu*nx+hv*ny)/h,v=(-hu*ny+hv*nx)/h;
 if(u<=0.||u>=sqrt(grav*h))return;
 double derivative[3]={0.,0.,0.};
 for(int g=0;g<ng;++g){double jac[3];st_mpm_mass_jacobian(h,u,v,manning[c],d50[g],
   mixture_d90[c],rhos[g]/rho-1.,rhos[g],grav,critical,hiding[g*nc+c],
   cf,friction_mode,mindepth[g],dry,jac);
  double factor=morfac*availability[g*nc+c]/bulk[g];
  for(int k=0;k<3;++k)derivative[k]+=factor*jac[k];}
 double bb=-2.*u,cc=u*u-grav*h*(1.+derivative[1]);
 double dd=-grav*h*(derivative[0]+v*derivative[2]);
 double p=cc-bb*bb/3.,qq=2.*bb*bb*bb/27.-bb*cc/3.+dd;if(p>=0.)return;
 double argument=3.*qq/(2.*p)*sqrt(-3./p);if(fabs(argument)>1.+1.e-10)return;
 double theta=acos(fmax(-1.,fmin(1.,argument)))/3.,radius=2.*sqrt(-p/3.),lambda=1.e300;
 for(int k=0;k<3;++k)lambda=fmin(lambda,radius*cos(theta-2.*acos(-1.)*k/3.)-bb/3.);
 if(lambda>=0.)return;
 double ratio=(lambda-u)*(lambda-u)/(grav*h),dh=(bchydro[nb+b]-h-zleft)/ratio;
 double next_h=h+dh,next_qn=h*u+lambda*dh,next_qt=h*v+v*dh;
 if(next_h<=dry||next_qn<=0.)return;
 double next_hu=next_qn*nx-next_qt*ny,next_hv=next_qn*ny+next_qt*nx;
 double next_z=zleft+(ratio-1.)*dh;
 if(!(isfinite(next_h)&&isfinite(next_hu)&&isfinite(next_hv)&&isfinite(next_z))){status[f]=1;return;}
 trace[f]=next_h;trace[nf+f]=next_hu;trace[2*nf+f]=next_hv;
 trace[3*nf+f]=next_z;trace[8*nf+f]=next_h;applied[f]=1;
}

extern "C" __global__ void st_faces(int nc,int nf,int ng,int nb,const double* cap,const double* grad,
 const double* bed,const double* down,const double* eta_grad,const double* availability,const double* xy,const double* mid,
 const double* width,const double* normal,const double* disp,const int* weno,const int* obdist,
 const int* ubdist,const int* owner,const int* neighbour,const int* bid,
 const int* offset,const int* faces,const int* signs,
 const int* active,const int* mode,const int* bedmode,const double* feed,const double* bchydro,
 const double* trace,const double* water,const int* closed,double dry,double grav,double dx,double dy,
 int slope,double alfabs,double alfabn,double angle,int coupled_kind,const double* q,
 const double* manning,const double* d50,const double* mix_d90,const double* rhos,
 const double* hiding,const double* min_depth,double rho,double critical,double cf,int friction_mode,
 double gaia_skin_ratio,double gaia_secondary_alpha,
 double* face_left,double* face_right,double* raw,int* status){
 int f=blockIdx.x*blockDim.x+threadIdx.x;if(f>=nf)return;int c=owner[f],n=neighbour[f],b=bid[f];
 status[f]=0;for(int g=0;g<ng;++g)raw[g*nf+f]=0.;
 if(!active[c]&&(n<0||!active[n]))return;if(n<0&&b<0)return;
 double nx=normal[f],ny=normal[nf+f],sx=0.,sy=0.;
 if(slope==2||slope==3||slope==5){double sn=0.,st;
  if(n>=0){double dn=(xy[n]-xy[c])*nx+(xy[nc+n]-xy[nc+c])*ny;
   if(!isfinite(dn)||dn<=1.e-12*fmax(dx,dy)){status[f]=1;return;}
   sn=-(bed[n]-bed[c])/dn;st=.5*(-(down[c]+down[n])*ny+(down[nc+c]+down[nc+n])*nx);
  }else st=-down[c]*ny+down[nc+c]*nx;sx=sn*nx-st*ny;sy=sn*ny+st*nx;
 }
 bool open=true;
 if(n>=0){double hL=trace[f],hR=trace[4*nf+f];if(closed[f])return;
  if(hL<=dry&&hR<=dry)open=false;
  else if(hR<=dry){double u=(trace[nf+f]*nx+trace[2*nf+f]*ny)/hL;
   RiemannType t=braids_riemanntype(hL,hL,u,-u,1,dry,grav);open=fmax(hL,t.hm)+trace[3*nf+f]>=trace[7*nf+f];}
  else if(hL<=dry){double u=(trace[5*nf+f]*nx+trace[6*nf+f]*ny)/hR;
   RiemannType t=braids_riemanntype(hR,hR,-u,u,1,dry,grav);open=fmax(hR,t.hm)+trace[7*nf+f]>=trace[3*nf+f];}
 }
 for(int g=0;g<ng;++g){double x=0.,y=0.,value=0.;int donor=c;
  if(n>=0){double lx,ly,rx,ry;
   double hl=trace[f],hr=trace[4*nf+f];
    bool consistent=coupled_kind==1&&fmin(hl,hr)>dry&&
     min(obdist[c],obdist[n])<=2&&min(ubdist[c],ubdist[n])>2&&
     !(fabs(trace[nf+f]*nx+trace[2*nf+f]*ny)>=hl*sqrt(grav*hl)&&
       fabs(trace[5*nf+f]*nx+trace[6*nf+f]*ny)>=hr*sqrt(grav*hr));
   st_face_value(c,n,0,f,g,nc,nf,ng,cap,grad,disp,weno,obdist,consistent,
     xy,mid,width,owner,neighbour,offset,faces,signs,nx,ny,lx,ly);
   st_face_value(n,c,1,f,g,nc,nf,ng,cap,grad,disp,weno,obdist,consistent,
     xy,mid,width,owner,neighbour,offset,faces,signs,nx,ny,rx,ry);
    if(coupled_kind!=0){int at=g*nf+f;face_left[at]=lx;face_left[ng*nf+at]=ly;
    face_right[at]=rx;face_right[ng*nf+at]=ry;}
   double l=lx*nx+ly*ny,r=rx*nx+ry*ny;
   if(l>0.&&r>0.){x=lx;y=ly;}else if(l<0.&&r<0.){x=rx;y=ry;donor=n;}
   else{x=.5*(lx+rx);y=.5*(ly+ry);if(x*nx+y*ny<0.)donor=n;}
   if(slope==2||slope==3||slope==5){double trl=0.,trr=0.;
    if(!st_slope_stress(q[c],q[nc+c],q[2*nc+c],manning[c],d50[g],mix_d90[c],rhos[g],rho,
       grav,critical,hiding[g*nc+c],cf,friction_mode,min_depth[g],dry,slope,
       cap[g*nc+c],cap[(ng+g)*nc+c],trl)||
       !st_slope_stress(q[n],q[nc+n],q[2*nc+n],manning[n],d50[g],mix_d90[n],rhos[g],rho,
       grav,critical,hiding[g*nc+n],cf,friction_mode,min_depth[g],dry,slope,
       cap[g*nc+n],cap[(ng+g)*nc+n],trr)){status[f]=1;return;}
    if(slope==3)st_koch_flokstra(x,y,sx,sy,.5*(trl+trr),alfabs,alfabn,angle);
    else st_slope(x,y,sx,sy,.5*(trl+trr),alfabs,alfabn,angle);
   }
   donor=(x*nx+y*ny)>=0.?c:n;
   x*=availability[g*nc+donor];y*=availability[g*nc+donor];value=open?x*nx+y*ny:0.;
  }else if(b>=0){int md=mode[b];
   if((md==1||md==6||md==7||md==8||md==9)&&water[f]<0.){
    if(bedmode[b]==1){if(friction_mode==3){double tr=0.;
      if(!st_gaia_open_capacity(trace[f],trace[nf+f],trace[2*nf+f],manning[c],d50[g],rhos[g],rho,
         grav,dry,min_depth[g],gaia_skin_ratio,gaia_secondary_alpha,eta_grad[c],eta_grad[nc+c],x,y,tr)){
       status[f]=1;return;}
     }else{x=cap[g*nc+c];y=cap[(ng+g)*nc+c];}
     if(slope==2||slope==3||slope==5){double tr=0.;
      if(!st_slope_stress(q[c],q[nc+c],q[2*nc+c],manning[c],d50[g],mix_d90[c],rhos[g],rho,
         grav,critical,hiding[g*nc+c],cf,friction_mode,min_depth[g],dry,slope,
         cap[g*nc+c],cap[(ng+g)*nc+c],tr)){status[f]=1;return;}
      if(slope==3)st_koch_flokstra(x,y,sx,sy,tr,alfabs,alfabn,angle);
      else st_slope(x,y,sx,sy,tr,alfabs,alfabn,angle);
     }
     x*=availability[g*nc+c];y*=availability[g*nc+c];value=fmin(0.,x*nx+y*ny);
    }else if(bedmode[b]==0||bedmode[b]==2){
     if(bchydro[b]>0.)value=feed[g*nb+b]*water[f]/bchydro[b];else value=-feed[g*nb+b];
    }else{status[f]=1;return;}
   }
   if(md>1&&md!=6&&md!=7&&md!=8&&md!=9&&water[f]>0.){
    x=cap[g*nc+c];y=cap[(ng+g)*nc+c];
    double stress_h=q[c],stress_hu=q[nc+c],stress_hv=q[2*nc+c];
    if((friction_mode==2||friction_mode==3)&&md==2&&q[c]>dry){double qn=q[nc+c]*nx+q[2*nc+c]*ny;
     if(fabs(qn)<q[c]*sqrt(grav*q[c])){double rx=0.,ry=0.,tr=0.;
      if(friction_mode==3){
       if(!st_gaia_open_capacity(trace[f],trace[nf+f],trace[2*nf+f],manning[c],d50[g],rhos[g],rho,
          grav,dry,min_depth[g],gaia_skin_ratio,gaia_secondary_alpha,eta_grad[c],eta_grad[nc+c],rx,ry,tr)){
        status[f]=1;return;}x=rx;y=ry;
      }else{
       if(!st_eh_capacity(trace[f],trace[nf+f],trace[2*nf+f],cf,d50[g],rhos[g],rho,
          grav,dry,fmax(min_depth[g],dry),rx,ry,tr)){status[f]=1;return;}
       double oldmag=sqrt(x*x+y*y),newmag=sqrt(rx*rx+ry*ry);
       if(oldmag>0.){x*=newmag/oldmag;y*=newmag/oldmag;}else{x=rx;y=ry;}
       stress_h=trace[f];stress_hu=trace[nf+f];stress_hv=trace[2*nf+f];
      }
     }
    }
    if(slope==2||slope==3||slope==5){double tr=0.;
     if(!st_slope_stress(stress_h,stress_hu,stress_hv,manning[c],d50[g],mix_d90[c],rhos[g],rho,
        grav,critical,hiding[g*nc+c],cf,friction_mode,min_depth[g],dry,slope,
        cap[g*nc+c],cap[(ng+g)*nc+c],tr,q[nc+c],q[2*nc+c])){status[f]=1;return;}
     if(slope==3)st_koch_flokstra(x,y,sx,sy,tr,alfabs,alfabn,angle);
     else st_slope(x,y,sx,sy,tr,alfabs,alfabn,angle);
    }
    x*=availability[g*nc+c];y*=availability[g*nc+c];value=fmax(0.,x*nx+y*ny);
   }
    if(coupled_kind==1&&md==2&&water[f]>0.&&q[c]>dry){
    double h=q[c],qn=q[nc+c]*nx+q[2*nc+c]*ny;
    if(fabs(qn)<h*sqrt(grav*h)){
     double tr=0.;if(!st_mpm_trace_capacity(trace[f],trace[nf+f],trace[2*nf+f],manning[c],
         d50[g],mix_d90[c],rhos[g],rho,grav,critical,hiding[g*nc+c],cf,friction_mode,
         min_depth[g],dry,x,y,tr)){status[f]=1;return;}
     if(slope==2||slope==3||slope==5){
      if(!st_slope_stress(trace[f],trace[nf+f],trace[2*nf+f],manning[c],
         d50[g],mix_d90[c],rhos[g],rho,grav,critical,hiding[g*nc+c],cf,friction_mode,
         min_depth[g],dry,slope,x,y,tr)){status[f]=1;return;}
      if(slope==3)st_koch_flokstra(x,y,sx,sy,tr,alfabs,alfabn,angle);
      else st_slope(x,y,sx,sy,tr,alfabs,alfabn,angle);
     }
     x*=availability[g*nc+c];y*=availability[g*nc+c];value=fmax(0.,x*nx+y*ny);
    }
   }
  }
  raw[g*nf+f]=value;if(!isfinite(value))status[f]=1;
 }
}

// Native coupled_mpm_face: Exner waves modify hydraulic flux before draining.
// Only the exact model-0 MPM trigger enters this kernel.
extern "C" __global__ void st_coupled_faces(int nc,int nf,int ng,const double* local,
 const double* riemann,const int* kind,const int* closed,const int* owner,
 const int* neighbour,const int* ubdist,const int* obdist,const double* normal,const double* availability,
 const double* face_left,const double* face_right,const double* manning,
 const double* d50,const double* rhos,const double* bulk,const double* mindepth,
 const double* mix_d90,const double* hiding,double rho,double grav,double critical,
 double cf,int friction_mode,
 double dry,double morfac,double* fl,double* fr,double* speed,double* raw,int* status){
 int f=blockIdx.x*blockDim.x+threadIdx.x;if(f>=nf)return;status[f]=0;
 int c=owner[f],n=neighbour[f];
 if(kind[f]!=1||n<0||closed[f]||min(ubdist[c],ubdist[n])<=2)return;
 double hl=local[f],hr=local[4*nf+f];if(fmin(hl,hr)<=dry||morfac<=0.)return;
 // Native coupled_mpm_face: a stage condition on supercritical outflow is
 // free release. Retain the already-computed augmented f-wave flux/speed.
 // This is an applicability rule, not an alteration of either flux solver.
 if(min(obdist[c],obdist[n])<=2&&
    fabs(local[nf+f])>=hl*sqrt(grav*hl)&&
    fabs(local[5*nf+f])>=hr*sqrt(grav*hr))return;
 double nx=normal[f],ny=normal[nf+f];
 double left[3]={hl,local[nf+f],local[2*nf+f]};
 double right[3]={hr,local[5*nf+f],local[6*nf+f]};
 double h=.5*(hl+hr),u=.5*(left[1]+right[1])/h,v=.5*(left[2]+right[2])/h;
 double derivative[3]={0.,0.,0.},qsleft=0.,qsright=0.,weightsum=0.;
 for(int g=0;g<ng;++g){int at=g*nf+f;
  double plus=face_left[at]*nx+face_left[ng*nf+at]*ny;
  double minus=face_right[at]*nx+face_right[ng*nf+at]*ny;
  double wc=availability[g*nc+c],wn=availability[g*nc+n],weight=.5*(wc+wn);
  weightsum+=weight;qsleft+=plus*wc/bulk[g]*morfac;qsright+=minus*wn/bulk[g]*morfac;
  double jac[3];st_mpm_mass_jacobian(h,u,v,.5*(manning[c]+manning[n]),d50[g],
   .5*(mix_d90[c]+mix_d90[n]),rhos[g]/rho-1.,rhos[g],grav,critical,
   .5*(hiding[g*nc+c]+hiding[g*nc+n]),cf,friction_mode,mindepth[g],dry,jac);
  double factor=morfac*weight/bulk[g];for(int k=0;k<3;++k)derivative[k]+=factor*jac[k];}
 if(weightsum<=0.||fmax(fabs(derivative[0]),fmax(fabs(derivative[1]),fabs(derivative[2])))<1.e-16)return;
 // Native Jacobian uses arithmetic mean state above, but its coupled flux
 // matrix uses Roe-averaged normal/tangential velocity at unequal depths.
 double rootl=sqrt(hl),rootr=sqrt(hr);
 u=(left[1]/rootl+right[1]/rootr)/(rootl+rootr);
 v=(left[2]/rootl+right[2]/rootr)/(rootl+rootr);
 double bb=-2.*u,cc=u*u-grav*h*(1.+derivative[1]);
 double dd=-grav*h*(derivative[0]+v*derivative[2]);
 double p=cc-bb*bb/3.,qq=2.*bb*bb*bb/27.-bb*cc/3.+dd;if(p>=0.)return;
 double argument=3.*qq/(2.*p)*sqrt(-3./p);if(fabs(argument)>1.+1.e-10)return;
 double theta=acos(fmax(-1.,fmin(1.,argument)))/3.,radius=2.*sqrt(-p/3.),roots[3];
 for(int k=0;k<3;++k)roots[k]=radius*cos(theta-2.*acos(-1.)*k/3.)-bb/3.;
 double newspeed=fmax(fabs(u),fmax(fabs(roots[0]),fmax(fabs(roots[1]),fabs(roots[2]))));
 int index=0;for(int k=1;k<3;++k)if((u>=0.&&roots[k]<roots[index])||(u<0.&&roots[k]>roots[index]))index=k;
 double lambda=roots[index],other[3];int j=0,upwind=st_coupled_upwind_sign(u,lambda);
 double den=1.;
 if(!upwind){
  for(int k=0;k<3;++k)if(k!=index)other[j++]=roots[k];other[2]=u;
  for(int k=0;k<3;++k)if((u>=0.&&other[k]<-1.e-10*newspeed)||
    (u<0.&&other[k]>1.e-10*newspeed))return;
  for(int k=0;k<3;++k)den*=lambda-other[k];
  if(fabs(den)<1.e-12*fmax(1.e-30,newspeed*newspeed*newspeed))return;
 }
 double pl[4]={left[1],left[1]*left[1]/hl+.5*grav*hl*hl,left[1]*left[2]/hl,qsleft};
 double pr[4]={right[1],right[1]*right[1]/hr+.5*grav*hr*hr,right[1]*right[2]/hr,qsright};
 double jump[4],project[4],next[4];
 for(int k=0;k<4;++k)jump[k]=pr[k]-pl[k];jump[1]+=grav*h*(local[7*nf+f]-local[3*nf+f]);
 if(!upwind){for(int k=0;k<4;++k)project[k]=jump[k];
  for(int k=0;k<3;++k){double ev=other[k];
   next[0]=project[1]-ev*project[0];
   next[1]=(grav*h-u*u)*project[0]+(2.*u-ev)*project[1]+grav*h*project[3];
   next[2]=-u*v*project[0]+v*project[1]+(u-ev)*project[2];
   next[3]=derivative[0]*project[0]+derivative[1]*project[1]+derivative[2]*project[2]-ev*project[3];
   for(int m=0;m<4;++m)project[m]=next[m];}}
 double fluxl[4],fluxr[4];for(int k=0;k<4;++k){double pj=upwind?0.:project[k]/den;
  double diss=upwind?upwind*jump[k]:(u>=0.?jump[k]-2.*pj:-jump[k]+2.*pj);
  fluxl[k]=pl[k]+.5*(jump[k]-diss);fluxr[k]=pr[k]-.5*(jump[k]+diss);}
 fluxl[0]=fluxr[0]=.5*(fluxl[0]+fluxr[0]);
 for(int k=2;k<4;++k)fluxl[k]=fluxr[k]=.5*(fluxl[k]+fluxr[k]);
 for(int k=0;k<4;++k)if(!isfinite(fluxl[k])||!isfinite(fluxr[k]))return;
 // Augmented f-wave flux prior to source/hydrostatic correction.
 double oldl[3],oldr[3];for(int k=0;k<3;++k){
  oldl[k]=pl[k]+riemann[k*nf+f];oldr[k]=pr[k]-riemann[(3+k)*nf+f];}
 oldl[0]=oldr[0]=.5*(oldl[0]+oldr[0]);
 double dl[3]={fluxl[0]-oldl[0],fluxl[1]-oldl[1],fluxl[2]-oldl[2]};
 double dr[3]={fluxr[0]-oldr[0],fluxr[1]-oldr[1],fluxr[2]-oldr[2]};
 fl[f]+=dl[0];fr[f]+=dr[0];
 fl[nf+f]+=dl[1]*nx-dl[2]*ny;fl[2*nf+f]+=dl[1]*ny+dl[2]*nx;
 fr[nf+f]+=dr[1]*nx-dr[2]*ny;fr[2*nf+f]+=dr[1]*ny+dr[2]*nx;
 speed[f]=fmax(speed[f],newspeed);
 double value=(fluxl[3]-.5*(qsleft+qsright))/morfac;
 for(int g=0;g<ng;++g){int at=g*nf+f;
  double plus=face_left[at]*nx+face_left[ng*nf+at]*ny;
  double minus=face_right[at]*nx+face_right[ng*nf+at]*ny;
  double wc=availability[g*nc+c],wn=availability[g*nc+n],weight=.5*(wc+wn);
  double base;
  if(plus>0.&&minus>0.)base=plus*wc;
  else if(plus<0.&&minus<0.)base=minus*wn;
  else{base=.5*(plus+minus);base*=base<0.?wn:wc;}
  double slope=raw[at]-base;
  raw[at]=.5*(plus*wc+minus*wn)+value*weight/weightsum*bulk[g]+slope;
  if(!isfinite(raw[at]))status[f]=1;}
}

// Formula-3 counterpart of coupled_mpm_face.  Only the uncorrected
// Engelund--Chollet--Cunge capacity enters the SWE--Exner Jacobian.  The
// Talmon/SLOPEFF and GAIA secondary-current contribution already present in
// raw is retained as an additive physical residual.
extern "C" __global__ void st_coupled_gaia_faces(int nc,int nf,int ng,const double* local,
 const double* riemann,const int* kind,const int* closed,const int* owner,
 const int* neighbour,const int* ubdist,const int* obdist,const double* normal,
 const double* availability,const double* base_left,const double* base_right,
 const double* manning,const double* d50,const double* rhos,const double* bulk,
 const double* mindepth,double rho,double grav,double skin_ratio,double dry,double morfac,
 double* fl,double* fr,double* speed,double* raw,int* status){
 int f=blockIdx.x*blockDim.x+threadIdx.x;if(f>=nf)return;status[f]=0;
 int c=owner[f],n=neighbour[f];
 if(kind[f]!=1||n<0||closed[f]||min(ubdist[c],ubdist[n])<=2)return;
 double hl=local[f],hr=local[4*nf+f];if(fmin(hl,hr)<=dry||morfac<=0.)return;
 if(min(obdist[c],obdist[n])<=2&&
    fabs(local[nf+f])>=hl*sqrt(grav*hl)&&
    fabs(local[5*nf+f])>=hr*sqrt(grav*hr))return;
 double nx=normal[f],ny=normal[nf+f];
 double left[3]={hl,local[nf+f],local[2*nf+f]};
 double right[3]={hr,local[5*nf+f],local[6*nf+f]};
 double h=.5*(hl+hr),u=.5*(left[1]+right[1])/h,v=.5*(left[2]+right[2])/h;
 double derivative[3]={0.,0.,0.},qsleft=0.,qsright=0.,weightsum=0.;
 for(int g=0;g<ng;++g){int at=g*nf+f;
  double plus=base_left[at]*nx+base_left[ng*nf+at]*ny;
  double minus=base_right[at]*nx+base_right[ng*nf+at]*ny;
  double wc=availability[g*nc+c],wn=availability[g*nc+n],weight=.5*(wc+wn);
  weightsum+=weight;qsleft+=plus*wc/bulk[g]*morfac;qsright+=minus*wn/bulk[g]*morfac;
  double jac[3];st_gaia_mass_jacobian(h,u,v,.5*(manning[c]+manning[n]),d50[g],
   rhos[g],rho,grav,skin_ratio,mindepth[g],dry,jac);
  double factor=morfac*weight/bulk[g];for(int k=0;k<3;++k)derivative[k]+=factor*jac[k];
 }
 if(weightsum<=0.||fmax(fabs(derivative[0]),fmax(fabs(derivative[1]),fabs(derivative[2])))<1.e-16)return;
 double rootl=sqrt(hl),rootr=sqrt(hr);
 u=(left[1]/rootl+right[1]/rootr)/(rootl+rootr);
 v=(left[2]/rootl+right[2]/rootr)/(rootl+rootr);
 double bb=-2.*u,cc=u*u-grav*h*(1.+derivative[1]);
 double dd=-grav*h*(derivative[0]+v*derivative[2]);
 double p=cc-bb*bb/3.,qq=2.*bb*bb*bb/27.-bb*cc/3.+dd;if(p>=0.)return;
 double argument=3.*qq/(2.*p)*sqrt(-3./p);if(fabs(argument)>1.+1.e-10)return;
 double theta=acos(fmax(-1.,fmin(1.,argument)))/3.,radius=2.*sqrt(-p/3.),roots[3];
 for(int k=0;k<3;++k)roots[k]=radius*cos(theta-2.*acos(-1.)*k/3.)-bb/3.;
 double newspeed=fmax(fabs(u),fmax(fabs(roots[0]),fmax(fabs(roots[1]),fabs(roots[2]))));
 int index=0;for(int k=1;k<3;++k)if((u>=0.&&roots[k]<roots[index])||(u<0.&&roots[k]>roots[index]))index=k;
 double lambda=roots[index],other[3];int j=0,upwind=st_coupled_upwind_sign(u,lambda);
 double den=1.;
 if(!upwind){
  for(int k=0;k<3;++k)if(k!=index)other[j++]=roots[k];other[2]=u;
  for(int k=0;k<3;++k)if((u>=0.&&other[k]<-1.e-10*newspeed)||
    (u<0.&&other[k]>1.e-10*newspeed))return;
  for(int k=0;k<3;++k)den*=lambda-other[k];
  if(fabs(den)<1.e-12*fmax(1.e-30,newspeed*newspeed*newspeed))return;
 }
 double pl[4]={left[1],left[1]*left[1]/hl+.5*grav*hl*hl,left[1]*left[2]/hl,qsleft};
 double pr[4]={right[1],right[1]*right[1]/hr+.5*grav*hr*hr,right[1]*right[2]/hr,qsright};
 double jump[4],project[4],next[4];
 for(int k=0;k<4;++k)jump[k]=pr[k]-pl[k];jump[1]+=grav*h*(local[7*nf+f]-local[3*nf+f]);
 if(!upwind){for(int k=0;k<4;++k)project[k]=jump[k];
  for(int k=0;k<3;++k){double ev=other[k];
   next[0]=project[1]-ev*project[0];
   next[1]=(grav*h-u*u)*project[0]+(2.*u-ev)*project[1]+grav*h*project[3];
   next[2]=-u*v*project[0]+v*project[1]+(u-ev)*project[2];
   next[3]=derivative[0]*project[0]+derivative[1]*project[1]+derivative[2]*project[2]-ev*project[3];
   for(int m=0;m<4;++m)project[m]=next[m];}}
 double fluxl[4],fluxr[4];for(int k=0;k<4;++k){double pj=upwind?0.:project[k]/den;
  double diss=upwind?upwind*jump[k]:(u>=0.?jump[k]-2.*pj:-jump[k]+2.*pj);
  fluxl[k]=pl[k]+.5*(jump[k]-diss);fluxr[k]=pr[k]-.5*(jump[k]+diss);}
 fluxl[0]=fluxr[0]=.5*(fluxl[0]+fluxr[0]);
 for(int k=2;k<4;++k)fluxl[k]=fluxr[k]=.5*(fluxl[k]+fluxr[k]);
 for(int k=0;k<4;++k)if(!isfinite(fluxl[k])||!isfinite(fluxr[k]))return;
 double oldl[3],oldr[3];for(int k=0;k<3;++k){
  oldl[k]=pl[k]+riemann[k*nf+f];oldr[k]=pr[k]-riemann[(3+k)*nf+f];}
 oldl[0]=oldr[0]=.5*(oldl[0]+oldr[0]);
 double dl[3]={fluxl[0]-oldl[0],fluxl[1]-oldl[1],fluxl[2]-oldl[2]};
 double dr[3]={fluxr[0]-oldr[0],fluxr[1]-oldr[1],fluxr[2]-oldr[2]};
 fl[f]+=dl[0];fr[f]+=dr[0];
 fl[nf+f]+=dl[1]*nx-dl[2]*ny;fl[2*nf+f]+=dl[1]*ny+dl[2]*nx;
 fr[nf+f]+=dr[1]*nx-dr[2]*ny;fr[2*nf+f]+=dr[1]*ny+dr[2]*nx;
 speed[f]=fmax(speed[f],newspeed);
 double value=(fluxl[3]-.5*(qsleft+qsright))/morfac;
 for(int g=0;g<ng;++g){int at=g*nf+f;
  double plus=base_left[at]*nx+base_left[ng*nf+at]*ny;
  double minus=base_right[at]*nx+base_right[ng*nf+at]*ny;
  double wc=availability[g*nc+c],wn=availability[g*nc+n],weight=.5*(wc+wn),base;
  if(plus>0.&&minus>0.)base=plus*wc;
  else if(plus<0.&&minus<0.)base=minus*wn;
  else{base=.5*(plus+minus);base*=base<0.?wn:wc;}
  double residual=raw[at]-base;
  raw[at]=.5*(plus*wc+minus*wn)+value*weight/weightsum*bulk[g]+residual;
  if(!isfinite(raw[at]))status[f]=1;}
}

extern "C" __global__ void st_rebalance(int nc,int ng,double* layers,const double* bulk,double desired,int* status){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;double a=0.,s=0.;status[c]=0;
 for(int g=0;g<ng;++g){a+=layers[g*nc+c]/bulk[g];s+=layers[(ng+g)*nc+c]/bulk[g];}
 double target=fmin(desired,a+s);
 if(a>target&&a>0.){double factor=fmin(1.,(a-target)/a);for(int g=0;g<ng;++g){double amount=layers[g*nc+c]*factor;
  layers[g*nc+c]-=amount;layers[(ng+g)*nc+c]+=amount;}}
 else if(a<target&&s>0.){double factor=fmin(1.,(target-a)/s);for(int g=0;g<ng;++g){double amount=layers[(ng+g)*nc+c]*factor;
  layers[(ng+g)*nc+c]-=amount;layers[g*nc+c]+=amount;}}
 for(int j=0;j<2*ng;++j){double value=layers[j*nc+c];
  if(value<0.&&fabs(value)<=128.*2.2250738585072014e-308)layers[j*nc+c]=0.;
  else if(!isfinite(value)||value<0.)status[c]=1;}
}

extern "C" __global__ void st_bed(int nc,int ng,const double* layers,const double* rock,const double* bulk,double* bed){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;double z=rock[c];
 for(int g=0;g<ng;++g){z+=layers[g*nc+c]/bulk[g];z+=layers[(ng+g)*nc+c]/bulk[g];}bed[c]=z;
}
