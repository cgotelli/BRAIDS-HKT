// FP64 mode4 owner-characteristic outlet. New validation component only.
#define DBL_EPSILON 2.2204460492503131e-16
#define DBL_MAX 1.7976931348623157e308
// NVRTC provides device math builtins; no external SDK headers are required.

extern "C" __global__ void lco_inputs(const double* bed,const double* q,int nc,int* invalid)
{
 int c=blockIdx.x*blockDim.x+threadIdx.x;
 if(c<nc && (!isfinite(bed[c]) || !isfinite(q[c]) || q[c]<0. ||
     !isfinite(q[1LL*nc+c]) || !isfinite(q[2LL*nc+c])))atomicOr(invalid,1);
}

__device__ void lco_close(int f,int c,int nc,int nf,const double* q,const double* normal,
 double dry,double* trace)
{
 const double h=q[c];
 if(h<=dry){trace[f]=trace[1LL*nf+f]=trace[2LL*nf+f]=0.;return;}
 const double un=(q[1LL*nc+c]*normal[f]+q[2LL*nc+c]*normal[1LL*nf+f])/h;
 const double ux=q[1LL*nc+c]/h-un*normal[f],uy=q[2LL*nc+c]/h-un*normal[1LL*nf+f];
 trace[f]=h;trace[1LL*nf+f]=h*ux;trace[2LL*nf+f]=h*uy;
}

// Same A/T/Q functions as native, with deterministic parallel summation order.
__device__ void lco_metrics(int first,int last,const int* owner,const int* faces,const int* branch,
 const double* bed,const double* length,const double* invariant,double eta,double g,double dry,
 double* aa,double* tt,double* qq,double* values,int* flags,int* evaluations)
{
 int lane=threadIdx.x;double a=0.,t=0.,q=0.;
 for(int p=first+lane;p<last;p+=blockDim.x){
  int f=faces[p];if(branch[f]!=1)continue;
  double h=fmax(eta-bed[owner[f]],0.);if(h<=dry)continue;
  double u=invariant[f]+2.*sqrt(g*h);
  a+=h*length[f];t+=length[f];q-=h*u*length[f];
 }
 aa[lane]=a;tt[lane]=t;qq[lane]=q;__syncthreads();
 for(int d=blockDim.x/2;d;d/=2){if(lane<d){aa[lane]+=aa[lane+d];tt[lane]+=tt[lane+d];qq[lane]+=qq[lane+d];}__syncthreads();}
 if(lane==0){
  values[0]=aa[0];values[1]=tt[0];values[2]=qq[0];
  values[3]=tt[0]>0.?sqrt(g*((aa[0]*aa[0])*aa[0])/tt[0]):0.;
  values[4]=qq[0]-values[3];++*evaluations;
  for(int j=0;j<5;j++)if(!isfinite(values[j]))*flags|=32;
 }
 __syncthreads();
}

extern "C" __global__ void lco_apply(const int* owner,const int* faces,const int* offsets,
 const double* length,const double* normal,const double* bed,const double* q,const int* invalid,
 int nc,int nf,int nb,double g,double dry,double limit,double* trace,int* branch,int* initial_branch,
 double* invariant,int* removal_pass,int* removal_reason,double* face_speed,int* status,int* face_status,
 double* stage,double* area,double* top_width,double* characteristic_Q,double* critical_Q,double* residual,
 double* integrated,double* lower_residual,double* upper_residual,int* active_passes,int* expansions,
 int* bisections,int* metric_evaluations,int* failed_face,double* failed_speed)
{
 const int b=blockIdx.x,lane=threadIdx.x;if(b>=nb)return;
 const int first=offsets[b],last=offsets[b+1],count=last-first;
 const double tol=128.*DBL_EPSILON;
 __shared__ double aa[256],tt[256],qq[256],v[5],lower,upper,eta,reslo,reshi;
 __shared__ int flags,done,changed,bracket,pass_count,expand_count,bisect_count,eval_count,first_speed;
 if(lane==0){
  flags=invalid[0];done=0;changed=0;pass_count=expand_count=bisect_count=eval_count=0;first_speed=last;
  stage[b]=nan("");area[b]=top_width[b]=characteristic_Q[b]=critical_Q[b]=residual[b]=0.;
  lower_residual[b]=upper_residual[b]=nan("");failed_face[b]=-1;failed_speed[b]=nan("");
 }
 __syncthreads();
 if(!flags)for(int p=first+lane;p<last;p+=blockDim.x){
  int f=faces[p],c=owner[f];double h=q[c];
  if(h<=dry){removal_reason[f]=1;continue;}
  double qn=q[1LL*nc+c]*normal[f]+q[2LL*nc+c]*normal[1LL*nf+f];
  if(!isfinite(qn)){atomicOr(&flags,32);continue;}
  if(qn < -tol*fmax(1.,fabs(qn))){lco_close(f,c,nc,nf,q,normal,dry,trace);removal_reason[f]=2;continue;}
  double wave=sqrt(g*h),ub=-qn/h;
  if(!isfinite(wave)||!isfinite(ub)){atomicOr(&flags,32);continue;}
  if(ub < -wave*(1.+1.e-10)){
   trace[f]=h;trace[1LL*nf+f]=q[1LL*nc+c];trace[2LL*nf+f]=q[2LL*nc+c];branch[f]=initial_branch[f]=2;continue;
  }
  branch[f]=initial_branch[f]=1;invariant[f]=ub-2.*wave;
  if(!isfinite(invariant[f]))atomicOr(&flags,32);
 }
 __syncthreads();
 for(int pass=1;pass<=count+1 && !flags && !done;pass++){
  double lo=DBL_MAX,hi=-DBL_MAX;
  for(int p=first+lane;p<last;p+=blockDim.x){int f=faces[p],c=owner[f];if(branch[f]==1){lo=fmin(lo,bed[c]);hi=fmax(hi,bed[c]+q[c]);}}
  aa[lane]=lo;tt[lane]=hi;__syncthreads();
  for(int d=blockDim.x/2;d;d/=2){if(lane<d){aa[lane]=fmin(aa[lane],aa[lane+d]);tt[lane]=fmax(tt[lane],tt[lane+d]);}__syncthreads();}
  if(lane==0){
   pass_count=pass;lower=aa[0];upper=tt[0];
   if(lower==DBL_MAX)done=1;
   else{lower+=fmax(2.*dry,tol*fmax(1.,fabs(lower)));if(!isfinite(lower)||!isfinite(upper))flags|=32;}
  }
  __syncthreads();if(flags||done)break;
  lco_metrics(first,last,owner,faces,branch,bed,length,invariant,lower,g,dry,aa,tt,qq,v,&flags,&eval_count);
  if(lane==0){reslo=v[4];lower_residual[b]=reslo;}__syncthreads();
  lco_metrics(first,last,owner,faces,branch,bed,length,invariant,upper,g,dry,aa,tt,qq,v,&flags,&eval_count);
  if(lane==0){reshi=v[4];upper_residual[b]=reshi;}__syncthreads();if(flags)break;
  if(reslo<0.){
   for(int p=first+lane;p<last;p+=blockDim.x){int f=faces[p];if(branch[f]==1){
    lco_close(f,owner[f],nc,nf,q,normal,dry,trace);branch[f]=0;removal_pass[f]=pass;removal_reason[f]=3;
   }}
   if(lane==0)done=1;__syncthreads();break;
  }
  if(lane==0)bracket=0;__syncthreads();
  for(int it=0;it<60 && !flags;it++){
   if(reshi<=0.){if(lane==0)bracket=1;__syncthreads();break;}
   if(lane==0){upper=lower+2.*(upper-lower);++expand_count;if(!isfinite(upper))flags|=32;}
   __syncthreads();if(flags)break;
   lco_metrics(first,last,owner,faces,branch,bed,length,invariant,upper,g,dry,aa,tt,qq,v,&flags,&eval_count);
   if(lane==0){reshi=v[4];upper_residual[b]=reshi;}__syncthreads();
  }
  if(lane==0 && !flags && !bracket)flags|=2;__syncthreads();if(flags)break;
  for(int it=0;it<90 && !flags;it++){
   if(lane==0){eta=.5*(lower+upper);if(!isfinite(eta))flags|=32;}__syncthreads();if(flags)break;
   lco_metrics(first,last,owner,faces,branch,bed,length,invariant,eta,g,dry,aa,tt,qq,v,&flags,&eval_count);
   if(lane==0){if(v[4]>0.)lower=eta;else upper=eta;++bisect_count;}__syncthreads();
  }
  if(flags)break;
  if(lane==0){eta=.5*(lower+upper);changed=0;if(!isfinite(eta))flags|=32;}__syncthreads();if(flags)break;
  for(int p=first+lane;p<last;p+=blockDim.x){
   int f=faces[p];if(branch[f]!=1)continue;double h=fmax(eta-bed[owner[f]],0.);
   int reason=h<=dry?4:0;
   if(!reason){double ub=invariant[f]+2.*sqrt(g*h);if(!isfinite(ub))atomicOr(&flags,32);else if(ub>tol)reason=5;}
   if(reason){lco_close(f,owner[f],nc,nf,q,normal,dry,trace);branch[f]=0;removal_pass[f]=pass;removal_reason[f]=reason;atomicAdd(&changed,1);}
  }
  __syncthreads();if(flags)break;if(changed)continue;
  lco_metrics(first,last,owner,faces,branch,bed,length,invariant,eta,g,dry,aa,tt,qq,v,&flags,&eval_count);
  if(lane==0){
   stage[b]=eta;area[b]=v[0];top_width[b]=v[1];characteristic_Q[b]=v[2];critical_Q[b]=v[3];residual[b]=v[4];
   if(v[0]<=0.||v[1]<=0.)flags|=4;
   if(fabs(v[4])>1.e-10*fmax(v[3],1.))flags|=8;
  }
  __syncthreads();if(flags)break;
  for(int p=first+lane;p<last;p+=blockDim.x){
   int f=faces[p],c=owner[f];if(branch[f]!=1)continue;
   double h=fmax(eta-bed[c],0.),ub=invariant[f]+2.*sqrt(g*h),un=0.,ux=0.,uy=0.;
   if(q[c]>dry){un=(q[1LL*nc+c]*normal[f]+q[2LL*nc+c]*normal[1LL*nf+f])/q[c];
    ux=q[1LL*nc+c]/q[c]-un*normal[f];uy=q[2LL*nc+c]/q[c]-un*normal[1LL*nf+f];}
   double speed=sqrt(ub*ub+(ux*ux+uy*uy));face_speed[f]=speed;
   if(!isfinite(speed)){atomicOr(&flags,32);continue;}
   if(speed>limit){atomicOr(&flags,16);atomicMin(&first_speed,p);continue;}
   trace[f]=h;trace[1LL*nf+f]=h*(-ub*normal[f]+ux);trace[2LL*nf+f]=h*(-ub*normal[1LL*nf+f]+uy);
  }
  if(lane==0)done=1;__syncthreads();
 }
 if(lane==0){
  if(!flags&&!done)flags|=64;
  if(first_speed<last){failed_face[b]=faces[first_speed];failed_speed[b]=face_speed[faces[first_speed]];}
 }
 __syncthreads();
 double total=0.;
 for(int p=first+lane;p<last;p+=blockDim.x){int f=faces[p];
  double h=trace[f],hu=trace[1LL*nf+f],hv=trace[2LL*nf+f];
  double flow=(hu*normal[f]+hv*normal[1LL*nf+f])*length[f];
  if(!isfinite(h)||!isfinite(hu)||!isfinite(hv)||!isfinite(flow))atomicOr(&flags,32);
  total+=flow;
 }
 qq[lane]=total;__syncthreads();
 for(int d=blockDim.x/2;d;d/=2){if(lane<d)qq[lane]+=qq[lane+d];__syncthreads();}
 if(lane==0){
  if(!isfinite(qq[0]))flags|=32;
  status[b]=flags;integrated[b]=flags?nan(""):qq[0];
  active_passes[b]=pass_count;expansions[b]=expand_count;bisections[b]=bisect_count;metric_evaluations[b]=eval_count;
 }
 __syncthreads();
 for(int p=first+lane;p<last;p+=blockDim.x){int f=faces[p];face_status[f]=flags;
  if(flags)trace[f]=trace[1LL*nf+f]=trace[2LL*nf+f]=0.;}
}
