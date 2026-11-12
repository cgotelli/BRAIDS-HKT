// Appended to the SHA-pinned, byte-identical qualified normal metric prefix.
// Critical INLET only. One cooperative block/section; no host root iterations.
struct LCShared {
 int nlevels,nterrain,nroots,selected,kind,branch,bisections,done;
 double minimum,maximum,stage,lo,hi,left_rate,right_rate,residual;
 double mid,quarter,a0,t0,lower_t,upper_t,best_energy;
};
struct LCWork {
 double *levels,*interval_roots,*interval_lo,*interval_hi,*roots,*energies;
 int *candidate_valid,*knot_valid,*interval_reason,*interval_iterations,*root_kind,*root_branch;
};

__device__ double lc_rate(double area,double width,double gravity){
 return area>0. && width>0. ? sqrt(gravity*(area*area*area)/width) : 0.;
}

__device__ void lc_sort_unique(double* levels,int capacity,LCShared* c){
 const int lane=threadIdx.x;
 for(int size=2;;size*=2){
  for(int distance=size/2;distance;distance/=2){
   for(int j=lane;j<capacity;j+=128){const int other=j^distance;
    if(other>j){const double a=levels[j],b=levels[other];
     if(((j&size)==0 && a>b)||((j&size)!=0 && a<b)){levels[j]=b;levels[other]=a;}}
   }
   __syncthreads();
  }
  if(size==capacity)break;
 }
 if(lane==0){int n=0;
  for(int j=0;j<capacity;j++){const double value=levels[j];if(!isfinite(value))break;
   if(n==0 || value!=levels[n-1])levels[n++]=value;}
  c->nlevels=n;
 }
 __syncthreads();
 for(int j=c->nlevels+lane;j<capacity;j+=128)levels[j]=ln_inf();
 __syncthreads();
}

__device__ void lc_find(int b,int count,int capacity,int root_capacity,double Q,double gravity,
 const int* offsets,const int* faces,const int* group_id,const double* group_n,
 const double* length,const double* nodes,LCWork w,LNShared* s,LCShared* c,int* status)
{
 const int lane=threadIdx.x,nodebase=offsets[b]+b;
 double minimum=ln_inf();int index=2147483647;
 for(int j=lane;j<=count;j+=128){const double value=nodes[nodebase+j];
  if(value<minimum || (value==minimum && j<index)){minimum=value;index=j;}}
 s->a[lane]=minimum;s->group[lane]=index;__syncthreads();
 for(int d=64;d;d/=2){if(lane<d){
  if(s->a[lane+d]<s->a[lane] || (s->a[lane+d]==s->a[lane] && s->group[lane+d]<s->group[lane])){
   s->a[lane]=s->a[lane+d];s->group[lane]=s->group[lane+d];}}
  __syncthreads();}
 if(lane==0){
  c->minimum=s->a[0];c->maximum=fmin(nodes[nodebase],nodes[nodebase+count]);
  if(s->group[0]==0 || s->group[0]==count || c->maximum<=c->minimum)status[b]|=2;
 }
 __syncthreads();if(status[b])return;
 if(Q==0.){if(lane==0){c->stage=c->minimum;c->kind=2;}return;}

 for(int j=lane;j<capacity;j+=128){double value=ln_inf();
  if(j==0)value=c->minimum;else if(j==1)value=c->maximum;
  else if(j<=count){const double z=nodes[nodebase+j-1];if(z>c->minimum && z<c->maximum)value=z;}
  w.levels[j]=value;
 }
 __syncthreads();lc_sort_unique(w.levels,capacity,c);
 if(lane==0)c->nterrain=c->nlevels;
 __syncthreads();
 // Native stationary-rate knots: initial terrain intervals only; append,
 // then sort once, preserving native turn order and exact duplicate tests.
 for(int k=0;k<c->nterrain-1;k++){
  if(lane==0){c->mid=0.5*(w.levels[k]+w.levels[k+1]);c->quarter=0.25*(w.levels[k+1]-w.levels[k]);}
  __syncthreads();
  if(c->quarter<=0. || c->mid-c->quarter==c->mid+c->quarter)continue;
  ln_metric(b,c->mid,offsets,faces,group_id,group_n,length,nodes,s,status);
  if(lane==0){c->a0=s->area;c->t0=s->width;}__syncthreads();
  ln_metric(b,c->mid-c->quarter,offsets,faces,group_id,group_n,length,nodes,s,status);
  if(lane==0)c->lower_t=s->width;__syncthreads();
  ln_metric(b,c->mid+c->quarter,offsets,faces,group_id,group_n,length,nodes,s,status);
  if(lane==0){
   c->upper_t=s->width;
   const double derivative=(c->upper_t-c->lower_t)/(2*c->quarter);
   if(!(fabs(derivative)<1.e-12*fmax(c->t0,1.)/(4*c->quarter))){
    double qa=2.5*(derivative*derivative),qb=5*c->t0*derivative,qc=3*(c->t0*c->t0)-c->a0*derivative;
    const double scale=fmax(fabs(qa),fmax(fabs(qb),fabs(qc)));
    qa/=scale;qb/=scale;qc/=scale;
    const double discriminant=qb*qb-4*qa*qc;
    if(discriminant>=0.){
     const double stable=-0.5*(qb+copysign(sqrt(discriminant),qb));
     const double turns[2]={stable==0. ? -qb/(2*qa) : stable/qa,
                            stable==0. ? -qb/(2*qa) : qc/stable};
     for(int turn=0;turn<2;turn++){
      const double value=c->mid+turns[turn];
      if(!isfinite(value)||value<=w.levels[k]||value>=w.levels[k+1])continue;
      bool duplicate=false;for(int j=0;j<c->nlevels;j++)duplicate|=w.levels[j]==value;
      if(duplicate)continue;
      if(c->nlevels>=capacity){status[b]|=32;break;}
      w.levels[c->nlevels++]=value;
     }
    }
   }
  }
  __syncthreads();if(status[b])return;
 }
 lc_sort_unique(w.levels,capacity,c);

 // Evaluate all increasing brackets. Native OpenMP executes these independent
 // intervals in parallel; this baseline cooperates over section segments and
 // retains interval ordering when compacting the eventual candidates.
 for(int k=0;k<c->nlevels-1;k++){
  if(lane==0){
   const double low=w.levels[k],high=w.levels[k+1];
   const double tol=fmax(1.e-12,128.*2.220446049250313080847263336181640625e-16*fmax(1.,fmax(fabs(low),fabs(high))));
   c->lo=low+tol;c->hi=high-tol;c->done=0;w.interval_lo[k]=c->lo;w.interval_hi[k]=c->hi;
   if(c->lo>=c->hi)w.interval_reason[k]=1;
  }
  __syncthreads();if(c->lo>=c->hi)continue;
  ln_metric(b,c->lo,offsets,faces,group_id,group_n,length,nodes,s,status);
  if(lane==0)c->left_rate=lc_rate(s->area,s->width,gravity)-Q;__syncthreads();
  ln_metric(b,c->hi,offsets,faces,group_id,group_n,length,nodes,s,status);
  if(lane==0){
   c->right_rate=lc_rate(s->area,s->width,gravity)-Q;
   if(!isfinite(c->left_rate)||!isfinite(c->right_rate))status[b]|=16;
   if(c->right_rate<c->left_rate)w.interval_reason[k]=2;
   else if(c->left_rate*c->right_rate>0.)w.interval_reason[k]=3;
  }
  __syncthreads();if(status[b])return;
  if(w.interval_reason[k])continue;
  for(int iteration=1;iteration<=80;iteration++){
   if(lane==0)c->mid=0.5*(c->lo+c->hi);__syncthreads();
   ln_metric(b,c->mid,offsets,faces,group_id,group_n,length,nodes,s,status);
   if(lane==0){
    c->residual=lc_rate(s->area,s->width,gravity)-Q;
    if(!isfinite(c->residual))status[b]|=16;
    c->bisections++;w.interval_iterations[k]=iteration;
    c->done=status[b] || fabs(c->residual)<=1.e-10*fmax(1.,Q) || c->hi-c->lo<=1.e-10;
    if(!c->done){
     if((c->left_rate<=0. && c->residual>=0.) || (c->left_rate>=0. && c->residual<=0.)){
      c->hi=c->mid;c->right_rate=c->residual;
     }else{c->lo=c->mid;c->left_rate=c->residual;}
    }
   }
   __syncthreads();if(c->done)break;
  }
  if(lane==0){
   w.interval_lo[k]=c->lo;w.interval_hi[k]=c->hi;
   if(!status[b]){w.interval_roots[k]=0.5*(c->lo+c->hi);w.candidate_valid[k]=1;w.interval_reason[k]=4;}
  }
  __syncthreads();if(status[b])return;
 }
 if(lane==0){
  for(int k=0;k<c->nlevels-1;k++)if(w.candidate_valid[k]){
   if(c->nroots>=root_capacity){status[b]|=32;break;}
   const int n=c->nroots++;w.roots[n]=w.interval_roots[k];w.root_kind[n]=0;w.root_branch[n]=k;
  }
 }
 __syncthreads();if(status[b])return;

 // Append admissible EXACT upper knots after all interval candidates. A
 // nextafter probe, not a fixed stage delta, checks native connectivity jumps.
 for(int k=1;k<c->nlevels;k++){
  const double value=w.levels[k];if(value<=c->minimum)continue;
  ln_metric(b,value,offsets,faces,group_id,group_n,length,nodes,s,status);
  if(lane==0){c->residual=lc_rate(s->area,s->width,gravity)-Q;
   if(!isfinite(c->residual))status[b]|=16;
   c->done=fabs(c->residual)<=1.e-10*fmax(1.,Q);}
  __syncthreads();if(status[b])return;
  if(!c->done)continue;
  if(value<c->maximum){
   ln_metric(b,value,offsets,faces,group_id,group_n,length,nodes,s,status);
   if(lane==0)c->a0=s->area;__syncthreads();
   ln_metric(b,fmin(c->maximum,nextafter(value,ln_inf())),offsets,faces,group_id,group_n,length,nodes,s,status);
   if(lane==0 && s->area-c->a0>64.*2.220446049250313080847263336181640625e-16*fmax(1.,fmax(fabs(c->a0),fabs(s->area))))c->done=0;
   __syncthreads();if(status[b])return;
  }
  if(lane==0 && c->done){
   if(c->nroots>=root_capacity)status[b]|=32;
   else{const int n=c->nroots++;w.roots[n]=value;w.root_kind[n]=1;w.root_branch[n]=k;w.knot_valid[k]=1;}
  }
  __syncthreads();if(status[b])return;
 }
 if(lane==0){
  if(c->nroots==0)status[b]|=8;
  else{c->stage=w.roots[0];c->selected=0;c->best_energy=1.7976931348623157e308;}
 }
 __syncthreads();if(status[b])return;
 for(int k=0;k<c->nroots;k++){
  ln_metric(b,w.roots[k],offsets,faces,group_id,group_n,length,nodes,s,status);
  if(lane==0 && s->area>0.){
   const double energy=w.roots[k]+(Q*Q)/(2*gravity*(s->area*s->area));w.energies[k]=energy;
   if(!isfinite(energy))status[b]|=16;
   else if(energy<c->best_energy){c->best_energy=energy;c->stage=w.roots[k];c->selected=k;}
  }
  __syncthreads();if(status[b])return;
 }
 if(lane==0){c->kind=w.root_kind[c->selected];c->branch=w.root_branch[c->selected];}
 __syncthreads();
}

extern "C" __global__ void lc_solve(
 const int* owner,const int* faces,const int* offsets,const int* group_id,
 const double* group_n,const double* length,const double* left,const double* right,
 const double* bed,const double* discharge,const int* global_status,double gravity,
 const int* level_offsets,const int* level_capacity,const int* root_offsets,const int* root_capacity,int nb,
 double* nodes,int* cache_valid,double* cached_bed,double* cached_q,
 double* stage,double* area,double* width,double* conveyance,double* residual,
 int* status,int* cache_hit,int* selected_branch,int* selected_kind,int* selected_index,
 int* nlevels,int* nterrain,int* nroots,int* bisections,int* evaluations,
 double* levels,double* interval_roots,double* interval_lo,double* interval_hi,
 double* roots,double* energies,int* candidate_valid,int* knot_valid,
 int* interval_reason,int* interval_iterations,int* root_kind,int* root_branch)
{
 const int b=blockIdx.x,lane=threadIdx.x;if(b>=nb)return;
 const int first=offsets[b],last=offsets[b+1],count=last-first,nodebase=first+b;
 const int base=level_offsets[b],capacity=level_capacity[b],rbase=root_offsets[b],rcap=root_capacity[b];
 __shared__ LNShared shared;__shared__ LCShared critical;LNShared* s=&shared;LCShared* c=&critical;
 if(lane==0){status[b]|=global_status[0];cache_hit[b]=0;bisections[b]=evaluations[b]=0;s->evaluations=0;}
 __syncthreads();
 int changed=status[b] || !(cache_valid[b] && cached_q[b]==discharge[b]);
 if(!changed)for(int pos=first+lane;pos<last;pos+=128)changed|=cached_bed[pos]!=bed[owner[faces[pos]]];
 s->group[lane]=changed;__syncthreads();
 for(int d=64;d;d/=2){if(lane<d)s->group[lane]|=s->group[lane+d];__syncthreads();}
 if(!s->group[0]){if(lane==0)cache_hit[b]=1;return;}
 if(lane==0){
  cache_valid[b]=0;stage[b]=0.;area[b]=width[b]=conveyance[b]=residual[b]=nan("");
  selected_branch[b]=selected_kind[b]=selected_index[b]=-1;nlevels[b]=nterrain[b]=nroots[b]=0;
  c->nlevels=c->nterrain=c->nroots=c->bisections=0;c->selected=c->kind=c->branch=-1;c->stage=0.;
 }
 for(int j=lane;j<capacity;j+=128){
  levels[base+j]=ln_inf();interval_roots[base+j]=interval_lo[base+j]=interval_hi[base+j]=nan("");
  candidate_valid[base+j]=knot_valid[base+j]=interval_reason[base+j]=interval_iterations[base+j]=0;
 }
 for(int j=lane;j<rcap;j+=128){roots[rbase+j]=energies[rbase+j]=nan("");root_kind[rbase+j]=root_branch[rbase+j]=-1;}
 __syncthreads();
 if(status[b]){for(int j=lane;j<=count;j+=128)nodes[nodebase+j]=nan("");return;}
 ln_nodes(b,owner,faces,offsets,left,right,bed,nodes,status);
 LCWork w={levels+base,interval_roots+base,interval_lo+base,interval_hi+base,roots+rbase,energies+rbase,
           candidate_valid+base,knot_valid+base,interval_reason+base,interval_iterations+base,root_kind+rbase,root_branch+rbase};
 if(!status[b])lc_find(b,count,capacity,rcap,discharge[b],gravity,offsets,faces,group_id,group_n,length,nodes,w,s,c,status);
 __syncthreads();
 if(!status[b]){
  ln_metric(b,c->stage,offsets,faces,group_id,group_n,length,nodes,s,status);
  if(lane==0){c->residual=lc_rate(s->area,s->width,gravity)-discharge[b];if(!isfinite(c->stage)||!isfinite(c->residual))status[b]|=16;}
 }
 __syncthreads();
 if(lane==0){
  nlevels[b]=c->nlevels;nterrain[b]=c->nterrain;nroots[b]=c->nroots;
  bisections[b]=c->bisections;evaluations[b]=s->evaluations;
  if(!status[b]){
   stage[b]=c->stage;area[b]=s->area;width[b]=s->width;conveyance[b]=s->conveyance;residual[b]=c->residual;
   selected_index[b]=c->selected;selected_kind[b]=c->kind;selected_branch[b]=c->branch;
   cached_q[b]=discharge[b];cache_valid[b]=1;
  }
 }
 __syncthreads();
 if(!status[b])for(int pos=first+lane;pos<last;pos+=128)cached_bed[pos]=bed[owner[faces[pos]]];
}
