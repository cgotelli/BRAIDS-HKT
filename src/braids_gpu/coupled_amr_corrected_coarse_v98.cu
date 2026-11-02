// Pure corrected coarse image. No accepted owner, clock or constitutive replay.
extern "C" __global__ void amr_corrected_images_v98(
 int nc,int ng,int cell,double dry,int transport_t,int transport_s,double eqt,double eqs,
 const double* q0,const double* sus0,const double* bed0,const double* fluid0,
 const double* dq,const double* dsus,const double* dbed,const double* dfluid,
 const double* initial_z,const double* reference_bed,const double* bulk,
 double* q,double* sus,double* bed,double* fluid,double* concentration,
 double* temperature,double* salinity,double* z,double* eta,int* wet,
 double* dry_transfer,double* momentum_cleanup,double* fluid_cleanup,int* status){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
 bool changed=c==cell,bad=false;status[c]=0;
 for(int k=0;k<3;k++){
  double value=q0[(long long)k*nc+c]+(changed?dq[k]:0.);
  q[(long long)k*nc+c]=value;if(!isfinite(value))bad=true;
 }
 double h=q[c];if(h<0.)bad=true;bool active=h>dry;wet[c]=active?1:0;
 for(int k=0;k<2;k++){
  double value=fluid0[(long long)k*nc+c]+(changed?dfluid[k]:0.);
  if(!isfinite(value)||(k==1&&value<0.))bad=true;
  fluid_cleanup[(long long)k*nc+c]=active?0.:-value;
  fluid[(long long)k*nc+c]=active?value:0.;
  double momentum=q[(long long)(k+1)*nc+c];
  momentum_cleanup[(long long)k*nc+c]=active?0.:-momentum;
  if(!active)q[(long long)(k+1)*nc+c]=0.;
 }
 double elevation=initial_z[c];
 for(int k=0;k<ng;k++){
  long long i=(long long)k*nc+c;
  double s=sus0[i]+(changed?dsus[k]:0.),b=bed0[i]+(changed?dbed[k]:0.);
  double tolerance=32768.*2.2204460492503131e-16*fmax(1.,fmax(fabs(s),fabs(b)));
  if(fabs(s)<=tolerance)s=fmax(s,0.);if(fabs(b)<=tolerance)b=fmax(b,0.);
  if(!isfinite(s)||!isfinite(b)||s<0.||b<0.)bad=true;
  dry_transfer[i]=active?0.:s;
  if(!active){b+=s;s=0.;}
  sus[i]=s;bed[i]=b;concentration[i]=active?s/h:0.;
  elevation+=(b-reference_bed[i])/bulk[k];
  if(!isfinite(b)||!isfinite(concentration[i]))bad=true;
 }
 double t=active&&transport_t?fluid[c]/h:eqt;
 double s=active&&transport_s?fluid[nc+c]/h:eqs;
 double sal_tol=4096.*2.2204460492503131e-16*fmax(1.,fabs(s));
 if(!isfinite(t)||!isfinite(s)||t<=-20.||s < -sal_tol)bad=true;
 temperature[c]=t;salinity[c]=fmax(s,0.);z[c]=elevation;eta[c]=h+elevation;
 if(!isfinite(elevation)||!isfinite(eta[c]))bad=true;
 status[c]=bad?1:0;
}
