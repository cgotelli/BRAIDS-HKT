// BRAIDS_SAMPLED_LEGACY_DISCHARGE_ABI 1
// Native production-section mode1. Augmented f-wave is invoked unchanged in a
// separate resident launch; no new Riemann solver or hydraulic reconstruction.
// NVRTC supplies device math builtins without platform C-library headers.
// Keep the kernel portable to the packaged Windows CUDA runtime.
#define DBL_MAX 1.7976931348623157e308
#define DBL_EPSILON 2.2204460492503131e-16

extern "C" __global__ void sampled_mode1_prepare(
 const double*q,const double*bed,const double*normal,const double*length,const int*owner,
 const int*offset,const int*faces,const double*forcing,const double*delta,
 const int*selected,const int*slot,long long nc,long long nf,int nb,long long nfan,
 double grav,double dry,int*branch,double*invariant,double*bounds,double*fan,int*needed,int*status){
 const int b=blockIdx.x,lane=threadIdx.x;if(b>=nb||!selected[b])return;
 __shared__ double sums[128],lows[128],highs[128];__shared__ int flags[128];
 double area=0.,lo=DBL_MAX,hi=-DBL_MAX;int invalid=0;
 for(int k=offset[b]+lane;k<offset[b+1];k+=128){
  int f=faces[k],c=owner[f],s=slot[f];double h=q[c],u=0.;
  if(h>dry)u=-(q[nc+c]*normal[f]+q[2*nc+c]*normal[nf+f])/h;
  bool incoming=h<=dry||u>=sqrt(grav*h)*(1.-1.e-12);
  double projected=h>dry?fmax(h-delta[f],0.):h,cwave=sqrt(grav*projected);
  invariant[f]=u-2.*cwave;
  branch[f]=(incoming||projected<=dry||u>=cwave*(1.-1.e-12))?2:1;
  if(projected>dry&&u<=-cwave)invalid=1;
  area+=fmax(forcing[nb+b]-bed[c]-delta[f],0.)*length[f];
  lo=fmin(lo,bed[c]+delta[f]);hi=fmax(hi,fmax(bed[c]+delta[f]+h+1.,forcing[nb+b]));
  needed[s]=0;for(int row=0;row<8;++row)fan[row*nfan+s]=0.;
 }
 sums[lane]=area;lows[lane]=lo;highs[lane]=hi;flags[lane]=invalid;__syncthreads();
 for(int span=64;span;span/=2){if(lane<span){sums[lane]+=sums[lane+span];
  lows[lane]=fmin(lows[lane],lows[lane+span]);highs[lane]=fmax(highs[lane],highs[lane+span]);
  flags[lane]|=flags[lane+span];}__syncthreads();}
 if(lane==0){
  status[b]=flags[0]||!isfinite(sums[0])||(forcing[b]>0.&&sums[0]<=0.);
  bounds[b]=lows[0];bounds[nb+b]=highs[0];bounds[2*nb+b]=sums[0]>0.?forcing[b]/sums[0]:0.;
  bounds[3*nb+b]=sums[0];
 }__syncthreads();
 if(status[b])return;
 const double target=bounds[2*nb+b];
 for(int k=offset[b]+lane;k<offset[b+1];k+=128){
  int f=faces[k],c=owner[f],s=slot[f];double h=fmax(forcing[nb+b]-bed[c]-delta[f],0.);
  if(branch[f]!=1||h<=dry||target<=sqrt(grav*h))continue;
  needed[s]=1;double nx=normal[f],ny=normal[nf+f];
  fan[s]=q[c];fan[nfan+s]=q[nc+c]*nx+q[2*nc+c]*ny;
  fan[2*nfan+s]=-q[nc+c]*ny+q[2*nc+c]*nx;fan[3*nfan+s]=bed[c];
  fan[4*nfan+s]=h;fan[5*nfan+s]=-h*target;
  fan[6*nfan+s]=0.;fan[7*nfan+s]=bed[c]+delta[f];
 }
}

__device__ double sampled_mode1_flux(double stage,int b,int lane,const double*bed,
 const double*length,const int*owner,const int*offset,const int*faces,const double*delta,
 const int*branch,const double*invariant,double grav,double target,double*values){
 double sum=0.;for(int k=offset[b]+lane;k<offset[b+1];k+=128){int f=faces[k];
  double h=fmax(stage-bed[owner[f]]-delta[f],0.);
  double u=branch[f]==1?invariant[f]+2.*sqrt(grav*h):target;sum+=h*u*length[f];}
 values[lane]=sum;__syncthreads();for(int span=64;span;span/=2){
  if(lane<span)values[lane]+=values[lane+span];__syncthreads();}return values[0];
}

extern "C" __global__ void sampled_mode1_solve(
 const double*q,const double*bed,const double*normal,const double*length,const int*owner,
 const int*offset,const int*faces,const double*forcing,const double*delta,
 const int*selected,const int*slot,long long nc,long long nf,int nb,long long nfan,
 double grav,double dry,double speed_limit,int*branch,const double*invariant,const double*bounds,
 const double*fan,const int*needed,const int*fan_status,double*stages,
 double*trace,int*status,int*face_status){
 const int b=blockIdx.x,lane=threadIdx.x;if(b>=nb||!selected[b]||status[b])return;
 __shared__ double values[128],lo,hi,stage;__shared__ int ready,invalid;
 const double target=bounds[2*nb+b],desired=forcing[b];
 for(int k=offset[b]+lane;k<offset[b+1];k+=128){
  int f=faces[k],c=owner[f],s=slot[f];if(!needed[s])continue;
  if(fan_status[s]){atomicOr(status+b,4);continue;}
  double largest=-DBL_MAX,speed=0.;for(int j=6;j<9;++j){largest=fmax(largest,fan[j*nfan+s]);
   speed=fmax(speed,fabs(fan[j*nfan+s]));}
  double h=fmax(forcing[nb+b]-bed[c]-delta[f],0.);
  double pl=q[c]>dry?q[nc+c]*normal[f]+q[2*nc+c]*normal[nf+f]:0.;
  double flux=.5*(pl+fan[s]-h*target-fan[3*nfan+s]);
  if(largest<=64.*DBL_EPSILON*fmax(1.,speed)&&fabs(flux+h*target)<=1.e-10*fmax(1.,h*target))branch[f]=2;
 }__syncthreads();if(status[b])return;
 if(lane==0){lo=bounds[b];hi=bounds[nb+b];ready=0;invalid=0;}__syncthreads();
 for(int iteration=0;iteration<60;++iteration){
  double flux=sampled_mode1_flux(hi,b,lane,bed,length,owner,offset,faces,delta,branch,invariant,grav,target,values);
  if(lane==0){if(isfinite(flux)&&flux>=desired)ready=1;else hi=lo+2.*(hi-lo);
   if(!isfinite(hi))invalid=1;}__syncthreads();if(ready||invalid)break;
 }
 if(!ready||invalid){if(lane==0)status[b]=8;return;}
 for(int iteration=0;iteration<90;++iteration){
  if(lane==0){stage=.5*(lo+hi);ready=stage==lo||stage==hi;}__syncthreads();if(ready)break;
  double flux=sampled_mode1_flux(stage,b,lane,bed,length,owner,offset,faces,delta,branch,invariant,grav,target,values);
  if(lane==0){if(!isfinite(flux))invalid=1;else if(flux<desired)lo=stage;else hi=stage;}
  __syncthreads();if(invalid)break;
 }
 if(invalid){if(lane==0)status[b]=8;return;}
 if(lane==0){stage=.5*(lo+hi);stages[b]=stage;}__syncthreads();
 for(int k=offset[b]+lane;k<offset[b+1];k+=128){int f=faces[k],c=owner[f];
  double h=fmax(stage-bed[c]-delta[f],0.);
  double u=branch[f]==1?invariant[f]+2.*sqrt(grav*h):target;
  trace[f]=0.;trace[nf+f]=0.;trace[2*nf+f]=0.;
  bool bad=u< -1.e-12||(branch[f]==1&&u>sqrt(grav*h)*(1.+1.e-12));
  if(h<=dry&&bad)continue;
  if(bad||(h>dry&&fabs(u)>speed_limit)||!isfinite(h)||!isfinite(u)){
   face_status[f]=3;atomicOr(status+b,2);continue;}
  trace[f]=h;trace[nf+f]=-h*u*normal[f];trace[2*nf+f]=-h*u*normal[nf+f];
 }
}
