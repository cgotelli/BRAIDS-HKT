// Current-state equilibrium bedform roughness; no host per-cell fallback.
// NVRTC does not provide the host C headers on the packaged Windows runtime.
#define BF_DBL_MAX 1.7976931348623157e308
#define BF_DBL_MIN 2.2250738585072014e-308
#define BF_DBL_EPSILON 2.2204460492503131e-16

__device__ bool bf_percentile(const double* volume,const double* d50,int ng,
                              double percentile,double* diameter){
 double total=0.;for(int g=0;g<ng;++g)total+=volume[g];
 if(!(total>0.)||!isfinite(total))return false;
 bool same=true;for(int g=1;g<ng;++g)if(d50[g]!=d50[0])same=false;
 if(same){double x=log(d50[0]);
  if(percentile<.5)*diameter=exp(x+1.25*log(.75)*(1.-2.*percentile));
  else *diameter=exp(x+1.25*log(1.5)*(2.*percentile-1.));
  return isfinite(*diameter)&&*diameter>0.;}
 double log_d50[32],lower_bound[32],upper_bound[32],lo=BF_DBL_MAX,hi=-BF_DBL_MAX;
 for(int g=0;g<ng;++g){log_d50[g]=log(d50[g]);
  lower_bound[g]=log_d50[g]+1.25*log(.75);
  upper_bound[g]=log_d50[g]+1.25*log(1.5);
  lo=fmin(lo,lower_bound[g]);hi=fmax(hi,upper_bound[g]);}
 for(int iteration=0;iteration<60;++iteration){double mid=.5*(lo+hi),cumulative=0.;
  for(int g=0;g<ng;++g){double share=volume[g]/total;
   double lower=lower_bound[g],upper=upper_bound[g],x=log_d50[g];
   if(mid<=lower)continue;
   else if(mid<x)cumulative=cumulative+share*.5*(mid-lower)/(x-lower);
   else if(mid<upper)cumulative=cumulative+share*(.5+.5*(mid-x)/(upper-x));
   else cumulative=cumulative+share;}
  if(cumulative<percentile)lo=mid;else hi=mid;}
 *diameter=exp(.5*(lo+hi));return isfinite(*diameter)&&*diameter>0.;
}

extern "C" __global__ void bf_properties(int nc,int ng,const double* q,const double* layers,
 const double* seed,const double* d50,const double* rhos,const double* bulk,
 double rho,double dry,double critical,double* result,int* iterations,int* status){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
 result[c]=seed[c];for(int k=1;k<6;++k)result[k*nc+c]=0.;
 iterations[c]=0;status[c]=0;
 double h=q[c];if(h<=dry)return;
 if(!isfinite(h)||!isfinite(q[nc+c])||!isfinite(q[2*nc+c])){status[c]=1;return;}
 double volume[32],total=0.,relative_sum=0.;
 for(int g=0;g<ng;++g){double mass=layers[g*nc+c];
  if(!isfinite(mass)){status[c]=1;return;}
  volume[g]=fmax(0.,mass)/bulk[g];total+=volume[g];
  relative_sum+=volume[g]*(rhos[g]/rho-1.);}
 if(total<=0.)return;
 if(!isfinite(total)){status[c]=1;return;}
 double d50_mix,d90_mix;
 if(!bf_percentile(volume,d50,ng,.5,&d50_mix)||
    !bf_percentile(volume,d50,ng,.9,&d90_mix)){status[c]=1;return;}
 double relative_density=relative_sum/total,u=q[nc+c]/h,v=q[2*nc+c]/h;
 double n=seed[c],height=0.,length=7.3*h,grain_ks=3.*d90_mix,dune_ks=0.,total_ks=0.;
 if(!(isfinite(h)&&isfinite(u)&&isfinite(v)&&isfinite(n)&&isfinite(d50_mix)&&
      isfinite(d90_mix)&&isfinite(relative_density)&&h>0.&&n>0.&&
      d50_mix>0.&&d90_mix>0.&&relative_density>0.&&critical>=0.)){
  status[c]=1;return;}
 double speed=sqrt(u*u+v*v);int count=0;
 for(int iteration=1;iteration<=100;++iteration){
  double chezy=pow(h,1./6.)/n,grain_chezy=18.*log10(12.*h/grain_ks);
  if(!(fmin(chezy,grain_chezy)>0.)){status[c]=1;return;}
  double mu=pow(chezy/grain_chezy,1.5);
  double theta=speed*speed/(chezy*chezy*relative_density*d50_mix);
  if(theta*mu>BF_DBL_MIN)height=fmax((24./63.)*(1.-critical/(theta*mu))*h,0.);
  dune_ks=1.1*height*(1.-exp(-25.*height/length));
  total_ks=fmin(.5*h,sqrt(grain_ks*grain_ks+dune_ks*dune_ks));
  total_ks=fmax(total_ks,BF_DBL_MIN);
  chezy=18.*log10(12.*h/total_ks);
  if(!(chezy>0.)){status[c]=1;return;}
  double next_n=pow(h,1./6.)/chezy;
  double scale=fmax(1.,fmax(fabs(next_n),fabs(n)));
  if(fabs(next_n-n)<=32.*BF_DBL_EPSILON*scale){n=next_n;count=iteration;break;}
  n=.5*(n+next_n);}
 if(count==0||!isfinite(n)||!isfinite(height)||!isfinite(length)||
    !isfinite(grain_ks)||!isfinite(dune_ks)||!isfinite(total_ks)||
    n<=0.||fmin(fmin(height,length),fmin(grain_ks,fmin(dune_ks,total_ks)))<0.){
  status[c]=1;return;}
 result[c]=n;result[nc+c]=height;result[2*nc+c]=length;
 result[3*nc+c]=grain_ks;result[4*nc+c]=dune_ks;result[5*nc+c]=total_ks;
 iterations[c]=count;
}
