// Typed C/W mode-9 normal-stage solver.  This is additive: the frozen
// Manning root remains in braids_support/gpu_fwave/live_normal_section.cu.
struct TNSShared {
 double a[128],p[128],w[128],k[128];int group[128];
 double carry_a,carry_p,area,width,conveyance;
 int carry_group,flag,nlevels,branch,kind,bisect,expand,evaluations,done;
 double minimum,maximum,lo,hi,left_rate,right_rate,stage,residual;
};

__device__ double tns_inf(){return __longlong_as_double(0x7ff0000000000000LL);}

// All 128 threads enter every call. Contiguous equal raw parameters form one
// subsection. The selected C/W conversion is evaluated at each subsection's
// hydraulic radius, exactly like native roughness_conveyance().
__device__ void tns_metric(int b,double eta,const int* offsets,const int* faces,
 const int* group_id,const double* group_parameter,const double* length,
 const double* nodes,int roughness_mode,double gravity,double minimum_work_depth,
 int waqua_white_colebrook,TNSShared* s,int* status)
{
 const int lane=threadIdx.x,first=offsets[b],last=offsets[b+1];
 double total_a=0.,total_w=0.,total_k=0.;
 if(lane==0){s->carry_a=0.;s->carry_p=0.;s->carry_group=-1;s->evaluations++;}
 __syncthreads();
 for(int tile=first;tile<last;tile+=128){
  const int pos=tile+lane,valid=pos<last;
  double a=0.,p=0.,w=0.;int g=-1;
  if(valid){
   const int f=faces[pos],node=pos+b;g=group_id[pos];
   const double z0=nodes[node],z1=nodes[node+1],h0=eta-z0,h1=eta-z1;
   if(!isfinite(h0)||!isfinite(h1)||!isfinite(z1-z0))atomicOr(status+b,16);
   if(pos==first && eta>z0)p+=eta-z0;
   if(fmax(h0,h1)>0.){
    double fraction;
    if(fmin(h0,h1)>0.)fraction=1.;
    else if(h0>0.)fraction=h0/(h0-h1);
    else fraction=h1/(h1-h0);
    w=length[f]*fraction;
    a=.5*(fmax(h0,0.)+fmax(h1,0.))*w;
    p+=hypot(length[f],z1-z0)*fraction;
   }
   if(!isfinite(a)||!isfinite(p)||!isfinite(w)||a<0.||p<0.||w<0.)
    atomicOr(status+b,16);
  }
  total_a+=a;total_w+=w;s->a[lane]=a;s->p[lane]=p;s->group[lane]=g;
  __syncthreads();
  for(int distance=1;distance<128;distance*=2){
   const bool join=valid && lane>=distance && g==s->group[lane-distance];
   const double prior_a=join?s->a[lane-distance]:0.;
   const double prior_p=join?s->p[lane-distance]:0.;
   __syncthreads();
   if(join){s->a[lane]+=prior_a;s->p[lane]+=prior_p;}
   __syncthreads();
  }
  const double ga=s->a[lane]+(valid&&g==s->carry_group?s->carry_a:0.);
  double gp=s->p[lane]+(valid&&g==s->carry_group?s->carry_p:0.);
  if(valid && (pos==last-1 || group_id[pos+1]!=g)){
   if(pos==last-1 && eta>nodes[pos+b+1])gp+=eta-nodes[pos+b+1];
   if(!isfinite(ga)||!isfinite(gp)||ga<0.||gp<0.)atomicOr(status+b,16);
   else if(ga>0. && gp>0.){
    const double radius=ga/gp;
    double chezy=0.,z0=0.,drag=0.;
    const bool valid_roughness=br_roughness_state(
     roughness_mode,radius,group_parameter[g],gravity,minimum_work_depth,
     waqua_white_colebrook,&chezy,&z0,&drag);
    const double value=valid_roughness?ga*chezy*sqrt(radius):nan("");
    if(!valid_roughness||!isfinite(value)||value<0.)atomicOr(status+b,16);
    else total_k+=value;
   }
  }
  __syncthreads();
  if(valid && (lane==127 || pos==last-1)){
   s->carry_group=g;s->carry_a=ga;s->carry_p=gp;
  }
  __syncthreads();
 }
 s->a[lane]=total_a;s->w[lane]=total_w;s->k[lane]=total_k;
 __syncthreads();
 for(int distance=64;distance;distance/=2){
  if(lane<distance){s->a[lane]+=s->a[lane+distance];
   s->w[lane]+=s->w[lane+distance];s->k[lane]+=s->k[lane+distance];}
  __syncthreads();
 }
 if(lane==0){
  s->area=s->a[0];s->width=s->w[0];s->conveyance=s->k[0];
  if(!isfinite(s->area)||!isfinite(s->width)||!isfinite(s->conveyance)||
     s->area<0.||s->width<0.||s->conveyance<0.)atomicOr(status+b,16);
 }
 __syncthreads();
}

__device__ void tns_nodes(int b,const int* owner,const int* faces,
 const int* offsets,const double* left,const double* right,const double* bed,
 double* nodes,int* status)
{
 for(int pos=offsets[b]+threadIdx.x;pos<offsets[b+1];pos+=128){
  const int f=faces[pos],node=pos+b;
  double value=bed[owner[f]]+left[f];
  if(pos>offsets[b]){
   const int old=faces[pos-1];
   const double previous=bed[owner[old]]+right[old];
   value=.5*(previous+value);
  }
  nodes[node]=value;if(!isfinite(value))atomicOr(status+b,16);
  if(pos==offsets[b+1]-1){value=bed[owner[f]]+right[f];nodes[node+1]=value;
   if(!isfinite(value))atomicOr(status+b,16);}
 }
 __syncthreads();
}

extern "C" __global__ void tns_inputs(const double* bed,const double* value,
 int nc,int nb,int nonnegative,int* global_status,int* status)
{
 const int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<nc && !isfinite(bed[i]))atomicOr(global_status,1);
 if(i<nb && (!isfinite(value[i])||(nonnegative&&value[i]<0.)))
  atomicOr(status+i,1);
}

extern "C" __global__ void tns_metrics(
 const int* owner,const int* faces,const int* offsets,const int* group_id,
 const double* group_parameter,const double* length,const double* left,
 const double* right,const double* bed,const double* eta,
 const int* global_status,int nb,int roughness_mode,double gravity,
 double minimum_work_depth,int waqua_white_colebrook,double* nodes,
 double* area,double* width,double* conveyance,int* status)
{
 const int b=blockIdx.x;if(b>=nb)return;
 __shared__ TNSShared shared;TNSShared* s=&shared;
 if(threadIdx.x==0){status[b]|=global_status[0];s->evaluations=0;}
 __syncthreads();
 tns_nodes(b,owner,faces,offsets,left,right,bed,nodes,status);
 if(status[b]==0)tns_metric(b,eta[b],offsets,faces,group_id,group_parameter,
  length,nodes,roughness_mode,gravity,minimum_work_depth,
  waqua_white_colebrook,s,status);
 if(threadIdx.x==0){area[b]=status[b]?nan(""):s->area;
  width[b]=status[b]?nan(""):s->width;
  conveyance[b]=status[b]?nan(""):s->conveyance;}
}

extern "C" __global__ void tns_solve(
 const int* owner,const int* faces,const int* offsets,const int* group_id,
 const double* group_parameter,const double* length,const double* left,
 const double* right,const double* slope,const double* bed,
 const double* discharge,const int* global_status,const int* level_offsets,
 const int* level_capacity,int nb,int roughness_mode,double gravity,
 double minimum_work_depth,int waqua_white_colebrook,
 double* nodes,double* levels,int* cache_valid,double* cached_bed,
 double* cached_q,double* cached_stage,double* cached_area,double* cached_width,
 double* cached_k,double* cached_residual,int* cached_branch,int* cached_kind,
 double* stage,double* area,double* width,double* conveyance,double* residual,
 int* status,int* cache_hit,int* branch,int* kind,int* bisections,
 int* expansions,int* evaluations)
{
#define BRAIDS_TYPED_NORMAL_ENDPOINT_ROOT_ABI 1
 const int b=blockIdx.x,lane=threadIdx.x;if(b>=nb)return;
 const int first=offsets[b],last=offsets[b+1],count=last-first,nodebase=first+b;
 __shared__ TNSShared shared;TNSShared* s=&shared;
 if(lane==0){
  status[b]|=global_status[0];stage[b]=0.;
  area[b]=width[b]=conveyance[b]=residual[b]=nan("");
  cache_hit[b]=0;branch[b]=-1;kind[b]=-1;
  bisections[b]=expansions[b]=evaluations[b]=0;
  s->evaluations=0;s->expand=0;s->bisect=0;s->branch=-1;s->kind=-1;s->done=0;
 }
 __syncthreads();
 if(status[b]){if(lane==0)cache_valid[b]=0;
  for(int j=lane;j<=count;j+=128)nodes[nodebase+j]=nan("");return;}
 int changed=!(cache_valid[b]&&cached_q[b]==discharge[b]);
 if(!changed)for(int pos=first+lane;pos<last;pos+=128)
  changed|=cached_bed[pos]!=bed[owner[faces[pos]]];
 s->group[lane]=changed;__syncthreads();
 for(int d=64;d;d/=2){if(lane<d)s->group[lane]|=s->group[lane+d];
  __syncthreads();}
 if(!s->group[0]){
  if(lane==0){cache_hit[b]=1;stage[b]=cached_stage[b];
   area[b]=cached_area[b];width[b]=cached_width[b];conveyance[b]=cached_k[b];
   residual[b]=cached_residual[b];branch[b]=cached_branch[b];
   kind[b]=cached_kind[b];}
  return;
 }
 if(lane==0)cache_valid[b]=0;
 tns_nodes(b,owner,faces,offsets,left,right,bed,nodes,status);
 if(status[b])return;
 double low=tns_inf(),high=-tns_inf();int index=2147483647;
 for(int j=lane;j<=count;j+=128){const double value=nodes[nodebase+j];
  if(value<low||(value==low&&j<index)){low=value;index=j;}high=fmax(high,value);}
 s->a[lane]=low;s->p[lane]=high;s->group[lane]=index;__syncthreads();
 for(int d=64;d;d/=2){if(lane<d){
  if(s->a[lane+d]<s->a[lane]||
     (s->a[lane+d]==s->a[lane]&&s->group[lane+d]<s->group[lane])){
   s->a[lane]=s->a[lane+d];s->group[lane]=s->group[lane+d];}
  s->p[lane]=fmax(s->p[lane],s->p[lane+d]);}__syncthreads();}
 if(lane==0){s->minimum=s->a[0];s->maximum=s->p[0];
  if(s->maximum==s->minimum)s->maximum=s->minimum+1.;}
 __syncthreads();if(status[b])return;
 const double Q=discharge[b],root_slope=sqrt(slope[b]);
 while(true){
  tns_metric(b,s->maximum,offsets,faces,group_id,group_parameter,length,nodes,
   roughness_mode,gravity,minimum_work_depth,waqua_white_colebrook,s,status);
  if(lane==0){
   const double rate=root_slope*s->conveyance;
   if(!isfinite(rate))status[b]|=16;
   s->done=status[b]||!(rate<Q&&s->expand<60);
   if(!s->done){s->maximum=s->minimum+2.*(s->maximum-s->minimum);s->expand++;
    if(!isfinite(s->maximum)){status[b]|=16;s->done=1;}}
  }
  __syncthreads();if(s->done)break;
 }
 if(lane==0 && s->maximum<=s->minimum)status[b]|=2;
 __syncthreads();
 if(status[b]){if(lane==0){expansions[b]=s->expand;
  evaluations[b]=s->evaluations;}return;}
 if(Q==0.){
  if(lane==0){s->stage=s->minimum;s->branch=-1;s->kind=2;s->done=1;}
 }else{
  const int base=level_offsets[b],capacity=level_capacity[b];
  for(int j=lane;j<capacity;j+=128){
   double value=tns_inf();
   if(j==0)value=s->minimum;else if(j==1)value=s->maximum;
   else if(j<=count){const double z=nodes[nodebase+j-1];
    if(z>s->minimum&&z<s->maximum)value=z;}
   levels[base+j]=value;
  }
  __syncthreads();
  for(int size=2;;size*=2){
   for(int distance=size/2;distance;distance/=2){
    for(int j=lane;j<capacity;j+=128){const int other=j^distance;
     if(other>j){const double a=levels[base+j],c=levels[base+other];
      if(((j&size)==0&&a>c)||((j&size)!=0&&a<c)){
       levels[base+j]=c;levels[base+other]=a;}}}
    __syncthreads();
   }
   if(size==capacity)break;
  }
  if(lane==0){int n=0;for(int j=0;j<capacity;j++){
   const double value=levels[base+j];if(!isfinite(value))break;
   if(n==0||value!=levels[base+n-1])levels[base+n++]=value;
  }s->nlevels=n;s->done=0;}
  __syncthreads();
  for(int k=0;k<s->nlevels-1;k++){
   if(lane==0){const double zlo=levels[base+k],zhi=levels[base+k+1];
    const double tol=fmax(1.e-12,128.*2.220446049250313080847263336181640625e-16*
     fmax(1.,fmax(fabs(zlo),fabs(zhi))));
    s->lo=zlo+tol;s->hi=zhi-tol;s->bisect=0;s->branch=k;}
   __syncthreads();
   if(s->lo<s->hi){
    tns_metric(b,s->lo,offsets,faces,group_id,group_parameter,length,nodes,
     roughness_mode,gravity,minimum_work_depth,waqua_white_colebrook,s,status);
    if(lane==0)s->left_rate=root_slope*s->conveyance-Q;__syncthreads();
    tns_metric(b,s->hi,offsets,faces,group_id,group_parameter,length,nodes,
     roughness_mode,gravity,minimum_work_depth,waqua_white_colebrook,s,status);
    if(lane==0)s->right_rate=root_slope*s->conveyance-Q;__syncthreads();
    if(status[b])break;
    if(s->left_rate*s->right_rate<=0.){
     for(int iteration=1;iteration<=80;iteration++){
      if(lane==0)s->stage=.5*(s->lo+s->hi);__syncthreads();
      tns_metric(b,s->stage,offsets,faces,group_id,group_parameter,length,nodes,
       roughness_mode,gravity,minimum_work_depth,waqua_white_colebrook,s,status);
      if(lane==0){
       s->bisect=iteration;s->residual=root_slope*s->conveyance-Q;
       s->done=status[b]||fabs(s->residual)<=1.e-10*fmax(1.,Q)||
        s->hi-s->lo<=1.e-10;
       if(!s->done){
        if((s->left_rate<=0.&&s->residual>=0.)||
           (s->left_rate>=0.&&s->residual<=0.)){
         s->hi=s->stage;s->right_rate=s->residual;
        }else{s->lo=s->stage;s->left_rate=s->residual;}
       }
      }
      __syncthreads();if(s->done)break;
     }
     if(lane==0){s->stage=.5*(s->lo+s->hi);s->kind=0;s->done=1;}
     __syncthreads();break;
    }
   }
   tns_metric(b,levels[base+k+1],offsets,faces,group_id,group_parameter,
    length,nodes,roughness_mode,gravity,minimum_work_depth,
    waqua_white_colebrook,s,status);
   if(lane==0){s->residual=root_slope*s->conveyance-Q;
    if(fabs(s->residual)<=1.e-10*fmax(1.,Q)){
     s->stage=levels[base+k+1];s->kind=1;s->done=1;}}
   __syncthreads();if(s->done||status[b])break;
  }
  if(lane==0&&!s->done&&!status[b])status[b]|=8;
 }
 __syncthreads();
 if(!status[b]){
  tns_metric(b,s->stage,offsets,faces,group_id,group_parameter,length,nodes,
   roughness_mode,gravity,minimum_work_depth,waqua_white_colebrook,s,status);
  if(lane==0){s->residual=root_slope*s->conveyance-Q;
   if(!isfinite(s->stage)||!isfinite(s->residual))status[b]|=16;}
  __syncthreads();
 }
 if(lane==0){
  expansions[b]=s->expand;bisections[b]=s->bisect;
  evaluations[b]=s->evaluations;
  if(!status[b]){
   stage[b]=cached_stage[b]=s->stage;area[b]=cached_area[b]=s->area;
   width[b]=cached_width[b]=s->width;conveyance[b]=cached_k[b]=s->conveyance;
   residual[b]=cached_residual[b]=s->residual;
   branch[b]=cached_branch[b]=s->branch;kind[b]=cached_kind[b]=s->kind;
   cached_q[b]=Q;cache_valid[b]=1;
  }
 }
 __syncthreads();
 if(!status[b])for(int pos=first+lane;pos<last;pos+=128)
  cached_bed[pos]=bed[owner[faces[pos]]];
}
