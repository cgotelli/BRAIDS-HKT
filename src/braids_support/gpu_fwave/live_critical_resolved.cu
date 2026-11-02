// Appended to the SHA-pinned, byte-identical qualified normal metric prefix.
// RESOLVED critical INLET candidate: analytic derivative, true turning bracket,
// relative-Q convergence and explicit representable-stage quality. Original
// frozen critical shader is unchanged. One block/section; no CPU fallback.
struct LCShared {
 int nlevels,nterrain,nroots,selected,kind,branch,bisections,done,quality;
 double minimum,maximum,stage,lo,hi,left_rate,right_rate,residual;
 double mid,quarter,a0,t0,lower_t,upper_t,best_energy,derivative,interval_stage;
};
struct LCWork {
 double *levels,*interval_roots,*interval_lo,*interval_hi,*roots,*energies;
 int *candidate_valid,*knot_valid,*interval_reason,*interval_iterations,*root_kind,*root_branch;
 int *interval_quality,*candidate_quality;
 double *interval_rate_lo,*interval_rate_hi,*candidate_residual;
};

__device__ double lc_rate(double area,double width,double gravity){
 return area>0. && width>0. ? sqrt(gravity*(area*area*area)/width) : 0.;
}

__device__ double lcr_target(double Q,double rate){
 return 1.e-10*Q+32.*2.220446049250313080847263336181640625e-16*fmax(Q,fabs(rate));
}

__device__ double lcr_froude_error(double area,double width,double Q,double gravity){
 if(!(area>0. && width>0.))return ln_inf();
 const double u=Q/area,fr2=(u*u)*width/(gravity*area);
 return isfinite(fr2)?fabs(fr2-1.):ln_inf();
}

// Exact piecewise-linear T' on a terrain interval: no sampled subtraction.
// Same independent segment tiling/tree as the qualified A/T helper.
__device__ void lcr_derivative(int b,double eta,const int* offsets,const int* faces,
 const double* length,const double* nodes,LNShared* s,LCShared* c,int* status){
 const int lane=threadIdx.x;double sum=0.;
 for(int pos=offsets[b]+lane;pos<offsets[b+1];pos+=128){
  const int node=pos+b;const double lo=fmin(nodes[node],nodes[node+1]),hi=fmax(nodes[node],nodes[node+1]);
  if(eta>lo && eta<hi)sum+=length[faces[pos]]/(hi-lo);
 }
 s->p[lane]=sum;__syncthreads();
 for(int d=64;d;d/=2){if(lane<d)s->p[lane]+=s->p[lane+d];__syncthreads();}
 if(lane==0){c->derivative=s->p[0];if(!isfinite(c->derivative)||c->derivative<0.)status[b]|=16;}
 __syncthreads();
}

__device__ double lcr_turn_sign(double area,double width,double derivative){
 // Division by positive T^2 rescales D without changing its sign. The dry
 // edge has D=0 by convention and can never supply a false negative bracket.
 if(!(width>0.))return 0.;
 if(area==0.)return 3.; // Avoid the indeterminate 0*Inf at underflowed wet area.
 return 3.-(area/width)*(derivative/width);
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
 if(Q==0.){if(lane==0){c->stage=c->minimum;c->kind=2;c->quality=2;}return;}

 for(int j=lane;j<capacity;j+=128){double value=ln_inf();
  if(j==0)value=c->minimum;else if(j==1)value=c->maximum;
  else if(j<=count){const double z=nodes[nodebase+j-1];if(z>c->minimum && z<c->maximum)value=z;}
  w.levels[j]=value;
 }
 __syncthreads();lc_sort_unique(w.levels,capacity,c);
 if(lane==0)c->nterrain=c->nlevels;
 __syncthreads();
 // Analytic derivative and a strict actual sign bracket exclude the spurious
 // near-dry quadratic roots present in the legacy sampled-T' calculation.
 for(int k=0;k<c->nterrain-1;k++){
  if(lane==0){
   c->mid=0.5*(w.levels[k]+w.levels[k+1]);
   c->lo=nextafter(w.levels[k],w.levels[k+1]);
   c->hi=nextafter(w.levels[k+1],w.levels[k]);
  }
  __syncthreads();
  if(!(c->lo<c->hi) || !(c->mid>w.levels[k] && c->mid<w.levels[k+1]))continue;
  ln_metric(b,c->mid,offsets,faces,group_id,group_n,length,nodes,s,status);
  if(lane==0){c->a0=s->area;c->t0=s->width;}__syncthreads();
  lcr_derivative(b,c->mid,offsets,faces,length,nodes,s,c,status);
  if(status[b])return;
  if(c->derivative==0.)continue;
  // Before the first terrain knot A=b*h+Tprime*h*h/2, with b>=0,h>0.
  // D=3*b*b+5*b*Tprime*h+2.5*Tprime*Tprime*h*h is nonnegative:
  // no interior turn exists, and a subnormal nextafter dry probe is unnecessary.
  if(k==0)continue;
  ln_metric(b,c->lo,offsets,faces,group_id,group_n,length,nodes,s,status);
  if(lane==0){c->lower_t=lcr_turn_sign(s->area,s->width,c->derivative);if(!isfinite(c->lower_t))status[b]|=16;}__syncthreads();
  ln_metric(b,c->hi,offsets,faces,group_id,group_n,length,nodes,s,status);
  if(lane==0){c->upper_t=lcr_turn_sign(s->area,s->width,c->derivative);if(!isfinite(c->upper_t))status[b]|=16;}__syncthreads();
  if(status[b])return;
  if(!(c->lower_t<0. && c->upper_t>0.))continue;
  if(lane==0){
   double qa=2.5*(c->derivative*c->derivative),qb=5*c->t0*c->derivative,qc=3*(c->t0*c->t0)-c->a0*c->derivative;
   const double scale=fmax(fabs(qa),fmax(fabs(qb),fabs(qc)));
   qa/=scale;qb/=scale;qc/=scale;
   const double discriminant=qb*qb-4*qa*qc;
   c->interval_stage=nan("");
   if(discriminant>=0.){
    const double stable=-0.5*(qb+copysign(sqrt(discriminant),qb));
    const double turns[2]={stable==0. ? -qb/(2*qa) : stable/qa,stable==0. ? -qb/(2*qa) : qc/stable};
    for(int turn=0;turn<2;turn++){
     const double value=c->mid+turns[turn];
     if(isfinite(value) && value>c->lo && value<c->hi){c->interval_stage=value;break;}
    }
   }
   // A failed/cancellation-sensitive closed-form estimate starts at the
   // bracket midpoint. Its bracket was established from independent geometry.
   if(!isfinite(c->interval_stage))c->interval_stage=0.5*(c->lo+c->hi);
  }
  __syncthreads();
  for(int iteration=0;iteration<128;iteration++){
   ln_metric(b,c->interval_stage,offsets,faces,group_id,group_n,length,nodes,s,status);
   if(lane==0){
    const double sign=lcr_turn_sign(s->area,s->width,c->derivative);
    if(!isfinite(sign))status[b]|=16;
    c->done=status[b] || fabs(sign)<=64.*2.220446049250313080847263336181640625e-16
                || nextafter(c->lo,c->hi)>=c->hi;
    if(!c->done){
     if(sign<0.)c->lo=c->interval_stage;else c->hi=c->interval_stage;
     const double middle=0.5*(c->lo+c->hi);
     if(middle==c->lo || middle==c->hi)c->done=1;else c->interval_stage=middle;
    }
   }
   __syncthreads();if(c->done)break;
  }
  if(lane==0){
   if(!c->done && !status[b])status[b]|=64;
   const double value=c->interval_stage;
   if(!(isfinite(value)&&value>w.levels[k]&&value<w.levels[k+1]))status[b]|=16;
   else{
    bool duplicate=false;for(int j=0;j<c->nlevels;j++)duplicate|=w.levels[j]==value;
    if(!duplicate){if(c->nlevels>=capacity)status[b]|=32;else w.levels[c->nlevels++]=value;}
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
   else if((c->left_rate>0. && c->right_rate>0.) || (c->left_rate<0. && c->right_rate<0.))w.interval_reason[k]=3;
  }
  __syncthreads();if(status[b])return;
  if(w.interval_reason[k])continue;
  if(lane==0){c->quality=-1;c->interval_stage=nan("");}
  __syncthreads();
  for(int iteration=1;iteration<=128;iteration++){
   if(lane==0)c->mid=0.5*(c->lo+c->hi);__syncthreads();
   ln_metric(b,c->mid,offsets,faces,group_id,group_n,length,nodes,s,status);
   if(lane==0){
    const double rate=lc_rate(s->area,s->width,gravity);c->residual=rate-Q;
    if(!isfinite(c->residual))status[b]|=16;
    c->bisections++;w.interval_iterations[k]=iteration;
    c->done=status[b] || fabs(c->residual)<=lcr_target(Q,rate);
    if(!status[b] && c->done){c->quality=0;c->interval_stage=c->mid;}
    else if(!c->done && (nextafter(c->lo,c->hi)>=c->hi || c->mid==c->lo || c->mid==c->hi))c->done=2;
   }
   __syncthreads();
   if(c->done==2){
    ln_metric(b,c->lo,offsets,faces,group_id,group_n,length,nodes,s,status);
    if(lane==0){c->a0=s->area;c->t0=s->width;w.interval_rate_lo[k]=lc_rate(s->area,s->width,gravity);c->left_rate=w.interval_rate_lo[k]-Q;}__syncthreads();
    ln_metric(b,c->hi,offsets,faces,group_id,group_n,length,nodes,s,status);
    if(lane==0){
     w.interval_rate_hi[k]=lc_rate(s->area,s->width,gravity);c->right_rate=w.interval_rate_hi[k]-Q;
     const bool choose_low=fabs(c->left_rate)<=fabs(c->right_rate);
     const double A=choose_low?c->a0:s->area,T=choose_low?c->t0:s->width;
     c->interval_stage=choose_low?c->lo:c->hi;
     c->residual=choose_low?c->left_rate:c->right_rate;
     if(!isfinite(c->residual))status[b]|=16;
     else if(fabs(c->residual)<=lcr_target(Q,c->residual+Q))c->quality=0;
     else if(lcr_froude_error(A,T,Q,gravity)<=1.e-6)c->quality=1;
     else status[b]|=64;
     c->done=1;
    }
    __syncthreads();
   }
   if(c->done)break;
   if(lane==0){
    if((c->left_rate<=0. && c->residual>=0.) || (c->left_rate>=0. && c->residual<=0.)){
     c->hi=c->mid;c->right_rate=c->residual;
    }else{c->lo=c->mid;c->left_rate=c->residual;}
   }
   __syncthreads();
  }
  if(lane==0){
   w.interval_lo[k]=c->lo;w.interval_hi[k]=c->hi;
   if(!status[b] && c->quality<0)status[b]|=64;
   if(!status[b]){
    w.interval_roots[k]=c->interval_stage;w.candidate_valid[k]=1;w.interval_reason[k]=c->quality==1?5:4;
    w.interval_quality[k]=c->quality;
   }else if(status[b]&64)w.interval_reason[k]=6;
  }
  __syncthreads();if(status[b])return;
 }
 if(lane==0){
  for(int k=0;k<c->nlevels-1;k++)if(w.candidate_valid[k]){
   if(c->nroots>=root_capacity){status[b]|=32;break;}
   const int n=c->nroots++;w.roots[n]=w.interval_roots[k];w.root_kind[n]=0;w.root_branch[n]=k;w.candidate_quality[n]=w.interval_quality[k];
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
   c->done=fabs(c->residual)<=lcr_target(Q,c->residual+Q);}
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
   else{const int n=c->nroots++;w.roots[n]=value;w.root_kind[n]=1;w.root_branch[n]=k;w.knot_valid[k]=1;w.candidate_quality[n]=0;}
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
   w.candidate_residual[k]=lc_rate(s->area,s->width,gravity)-Q;
   const double energy=w.roots[k]+(Q*Q)/(2*gravity*(s->area*s->area));w.energies[k]=energy;
   if(!isfinite(energy))status[b]|=16;
   else if(energy<c->best_energy){c->best_energy=energy;c->stage=w.roots[k];c->selected=k;}
  }
  __syncthreads();if(status[b])return;
 }
 if(lane==0){c->kind=w.root_kind[c->selected];c->branch=w.root_branch[c->selected];c->quality=w.candidate_quality[c->selected];}
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
 int* interval_reason,int* interval_iterations,int* root_kind,int* root_branch,
 int* quality,int* interval_quality,int* candidate_quality,
 double* interval_rate_lo,double* interval_rate_hi,double* candidate_residual)
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
  selected_branch[b]=selected_kind[b]=selected_index[b]=-1;quality[b]=-1;nlevels[b]=nterrain[b]=nroots[b]=0;
  c->nlevels=c->nterrain=c->nroots=c->bisections=0;c->selected=c->kind=c->branch=c->quality=-1;c->stage=0.;
 }
 for(int j=lane;j<capacity;j+=128){
  levels[base+j]=ln_inf();interval_roots[base+j]=interval_lo[base+j]=interval_hi[base+j]=nan("");
  candidate_valid[base+j]=knot_valid[base+j]=interval_reason[base+j]=interval_iterations[base+j]=0;interval_quality[base+j]=-1;
  interval_rate_lo[base+j]=interval_rate_hi[base+j]=nan("");
 }
 for(int j=lane;j<rcap;j+=128){roots[rbase+j]=energies[rbase+j]=candidate_residual[rbase+j]=nan("");root_kind[rbase+j]=root_branch[rbase+j]=candidate_quality[rbase+j]=-1;}
 __syncthreads();
 if(status[b]){for(int j=lane;j<=count;j+=128)nodes[nodebase+j]=nan("");return;}
 ln_nodes(b,owner,faces,offsets,left,right,bed,nodes,status);
 LCWork w={levels+base,interval_roots+base,interval_lo+base,interval_hi+base,roots+rbase,energies+rbase,
           candidate_valid+base,knot_valid+base,interval_reason+base,interval_iterations+base,root_kind+rbase,root_branch+rbase,
           interval_quality+base,candidate_quality+rbase,interval_rate_lo+base,interval_rate_hi+base,candidate_residual+rbase};
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
   selected_index[b]=c->selected;selected_kind[b]=c->kind;selected_branch[b]=c->branch;quality[b]=c->quality;
   cached_q[b]=discharge[b];cache_valid[b]=1;
  }
 }
 __syncthreads();
 if(!status[b])for(int pos=first+lane;pos<last;pos+=128)cached_bed[pos]=bed[owner[faces[pos]]];
}
