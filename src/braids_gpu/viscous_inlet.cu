// Resident viscosity-equilibrated total-discharge inlet for typed C/M/W drag.
// One CUDA block owns one independent boundary.  The nonlinear section solve
// is deliberately one thread: inlet sections are small, the tridiagonal solve
// is sequential, and keeping it resident avoids a host round-trip during each
// implicit midpoint residual evaluation.  Different inlets run concurrently.

__device__ __forceinline__ double br_profile_merit(
 int begin,int count,const double* width,const double* beta,const double* weight,
 const double* edge,const double* u,const double* delta,double alpha,
 double driving,double delta_driving,double discharge,double* constraint_out,
 double* scale_out,double* acceleration_max_out)
{
 const double candidate_driving=driving+alpha*delta_driving;
 double constraint=-discharge,scale=fabs(candidate_driving),sum_square=0.0;
 double acceleration_max=0.0;
 for(int j=0;j<count;++j){
  const int p=begin+j;
  const double value=u[p]+alpha*delta[p];
  double diffusion=0.0;
  if(j>0)diffusion+=edge[p-1]*(value-(u[p-1]+alpha*delta[p-1]));
  if(j+1<count)diffusion+=edge[p]*(value-(u[p+1]+alpha*delta[p+1]));
  const double acceleration=diffusion/width[p]+beta[p]*value*value-candidate_driving;
  scale=fmax(scale,fmax(beta[p]*value*value,fabs(diffusion/width[p])));
  acceleration_max=fmax(acceleration_max,fabs(acceleration));
  constraint+=weight[p]*value;
 }
 scale=fmax(scale,2.2250738585072014e-308);
 const double qscale=fmax(fmax(discharge,fabs(constraint+discharge)),
                          2.2250738585072014e-308);
 for(int j=0;j<count;++j){
  const int p=begin+j;
  const double value=u[p]+alpha*delta[p];
  double diffusion=0.0;
  if(j>0)diffusion+=edge[p-1]*(value-(u[p-1]+alpha*delta[p-1]));
  if(j+1<count)diffusion+=edge[p]*(value-(u[p+1]+alpha*delta[p+1]));
  const double acceleration=diffusion/width[p]+beta[p]*value*value-candidate_driving;
  const double normalized=acceleration/scale;
  sum_square+=normalized*normalized;
 }
 const double normalized_constraint=constraint/qscale;
 *constraint_out=constraint;*scale_out=scale;*acceleration_max_out=acceleration_max;
 return sqrt(sum_square/(double)count+normalized_constraint*normalized_constraint);
}

__device__ __forceinline__ bool br_profile_tridiagonal(
 int begin,int count,const double* lower,const double* diagonal,const double* upper,
 const double* rhs,double* cprime,double* dprime,double* solution)
{
 double scale=1.0;
 for(int j=0;j<count;++j){const int p=begin+j;
  scale=fmax(scale,fmax(fabs(lower[p]),fmax(fabs(diagonal[p]),fabs(upper[p]))));}
 const double tolerance=128.0*2.2250738585072014e-308*scale;
 double pivot=diagonal[begin];
 if(!isfinite(pivot)||pivot<=tolerance)return false;
 cprime[begin]=upper[begin]/pivot;dprime[begin]=rhs[begin]/pivot;
 for(int j=1;j<count;++j){const int p=begin+j;
  pivot=diagonal[p]-lower[p]*cprime[p-1];
  if(!isfinite(pivot)||pivot<=tolerance)return false;
  cprime[p]=upper[p]/pivot;
  dprime[p]=(rhs[p]-lower[p]*dprime[p-1])/pivot;
 }
 solution[begin+count-1]=dprime[begin+count-1];
 for(int j=count-2;j>=0;--j){const int p=begin+j;
  solution[p]=dprime[p]-cprime[p]*solution[p+1];}
 for(int j=0;j<count;++j)if(!isfinite(solution[begin+j]))return false;
 return true;
}

extern "C" __global__ void viscosity_equilibrated_profiles(
 const double* q,const double* parameter,const int* owner_slot,const int* face_slot,
 const double* slot_width,const double* slot_center,const int* slot_touching,
 const int* offsets,const int* active,const double* discharge,long long nc,long long nf,
 int nb,int roughness_mode,int waqua_white_colebrook,double viscosity,double gravity,
 double dry_tolerance,double work_depth,double* face_speed,double* driving_output,
 double* discharge_residual,double* acceleration_residual,int* iterations,int* status,
 int* face_component,int* wet_slot,double* width,double* depth,double* beta,
 double* weight,double* edge,double* u,double* rhs,double* diagonal,double* lower,
 double* upper,double* cprime,double* dprime,double* primary,double* sensitivity,
 double* delta)
{
 const int b=(int)blockIdx.x;
 if(b>=nb||threadIdx.x!=0)return;
 status[b]=0;iterations[b]=0;driving_output[b]=0.0;
 discharge_residual[b]=0.0;acceleration_residual[b]=0.0;
 const int first=offsets[b],last=offsets[b+1];
 for(int k=first;k<last;++k){const int f=face_slot[k];
  if(f>=0&&f<nf){face_speed[f]=0.0;face_component[f]=-1;}}
 if(!active[b])return;
 const double target=discharge[b];
 if(first>=last||!isfinite(target)||target<0.0||!isfinite(viscosity)||viscosity<0.0||
    !isfinite(gravity)||gravity<=0.0||!isfinite(dry_tolerance)||dry_tolerance<=0.0||
    !isfinite(work_depth)||work_depth<0.0){status[b]=1;return;}
 int count=0,component=-1,previous_slot=-2;
 bool component_has_drag=false;
 for(int k=first;k<last;++k){
  const int c=owner_slot[k],f=face_slot[k];
  if(c<0||c>=nc||f<0||f>=nf||!isfinite(slot_width[k])||slot_width[k]<=0.0||
     !isfinite(slot_center[k])){status[b]=1;return;}
  const double h=q[c];
  if(!isfinite(h)||h<0.0){status[b]=1;return;}
  if(h<=dry_tolerance)continue;
  const int p=first+count;wet_slot[p]=k;width[p]=slot_width[k];depth[p]=h;
  const double raw=parameter[c];double chezy=0.0,z0=0.0,drag=0.0;
  bool valid=false;
  if(roughness_mode==2&&raw==0.0){drag=0.0;valid=true;}
  else valid=br_roughness_state(roughness_mode,h,raw,gravity,work_depth,
                                waqua_white_colebrook,&chezy,&z0,&drag);
  if(!valid){status[b]=1;return;}
  beta[p]=drag/h;weight[p]=slot_width[k]*h;
  const bool connected=(count>0&&k==previous_slot+1&&slot_touching[k-1]);
  if(!connected){
   if(count>0&&!component_has_drag){status[b]=3;return;}
   if(count>0)edge[p-1]=0.0;
   ++component;component_has_drag=false;
  }else{
   const double distance=slot_center[k]-slot_center[k-1];
   if(distance<=0.0||!isfinite(distance)){status[b]=1;return;}
   edge[p-1]=viscosity/distance;
  }
  component_has_drag=component_has_drag||beta[p]>0.0;
  face_component[f]=component;previous_slot=k;++count;
 }
 if(target>0.0&&count==0){status[b]=2;return;}
 if(target==0.0||count==0)return;
 if(!component_has_drag){status[b]=3;return;}
 double wet_area=0.0,width_sum=0.0;
 for(int j=0;j<count;++j){const int p=first+j;wet_area+=weight[p];width_sum+=width[p];}
 if(wet_area<=0.0||width_sum<=0.0||!isfinite(wet_area)){status[b]=2;return;}
 const double initial=target/wet_area;
 double driving=0.0;
 for(int j=0;j<count;++j){const int p=first+j;u[p]=initial;
  driving+=width[p]*beta[p]*initial*initial;}
 driving/=width_sum;
 if(driving<=0.0||!isfinite(driving)){status[b]=3;return;}
 const double relative_tolerance=2.0e-11;
 const double absolute_acceleration_tolerance=2.0e-14;
 bool converged=false;
 double final_constraint=0.0,final_scale=0.0,final_acceleration=0.0;
 // delta=0 lets the shared evaluator inspect the current iterate.
 for(int j=0;j<count;++j)delta[first+j]=0.0;
 for(int iteration=0;iteration<=80;++iteration){
  iterations[b]=iteration;
  const double merit=br_profile_merit(first,count,width,beta,weight,edge,u,delta,0.0,
      driving,0.0,target,&final_constraint,&final_scale,&final_acceleration);
  const double qscale=fmax(fmax(target,fabs(final_constraint+target)),
                           2.2250738585072014e-308);
  const double qtolerance=128.0*2.2204460492503131e-16*qscale;
  const double atolerance=absolute_acceleration_tolerance+relative_tolerance*final_scale;
  if(final_acceleration<=atolerance&&fabs(final_constraint)<=qtolerance){converged=true;break;}
  if(iteration==80)break;
  for(int j=0;j<count;++j){const int p=first+j;
   double diffusion=0.0;
   if(j>0)diffusion+=edge[p-1]*(u[p]-u[p-1]);
   if(j+1<count)diffusion+=edge[p]*(u[p]-u[p+1]);
   rhs[p]=-(diffusion+width[p]*beta[p]*u[p]*u[p]-width[p]*driving);
   diagonal[p]=2.0*width[p]*beta[p]*u[p];lower[p]=0.0;upper[p]=0.0;
   if(j>0){lower[p]=-edge[p-1];diagonal[p]+=edge[p-1];}
   if(j+1<count){upper[p]=-edge[p];diagonal[p]+=edge[p];}
  }
  if(!br_profile_tridiagonal(first,count,lower,diagonal,upper,rhs,cprime,dprime,primary)){
   status[b]=4;return;}
  for(int j=0;j<count;++j)rhs[first+j]=width[first+j];
  if(!br_profile_tridiagonal(first,count,lower,diagonal,upper,rhs,cprime,dprime,sensitivity)){
   status[b]=4;return;}
  double denominator=0.0,numerator=-final_constraint;
  for(int j=0;j<count;++j){const int p=first+j;
   denominator+=weight[p]*sensitivity[p];numerator-=weight[p]*primary[p];}
  if(denominator<=0.0||!isfinite(denominator)){status[b]=4;return;}
  const double delta_driving=numerator/denominator;
  double alpha=1.0;
  for(int j=0;j<count;++j){const int p=first+j;
   delta[p]=primary[p]+sensitivity[p]*delta_driving;
   if(!isfinite(delta[p])){status[b]=4;return;}
   if(delta[p]<0.0)alpha=fmin(alpha,0.99*(-u[p]/delta[p]));
  }
  if(!isfinite(delta_driving)){status[b]=4;return;}
  if(delta_driving<0.0)alpha=fmin(alpha,0.99*(-driving/delta_driving));
  bool accepted=false;
  for(int line_search=0;line_search<60;++line_search){
   bool positive=driving+alpha*delta_driving>0.0;
   for(int j=0;j<count&&positive;++j){const int p=first+j;
    positive=u[p]+alpha*delta[p]>0.0;}
   double constraint=0.0,scale=0.0,acceleration=0.0;
   const double candidate=positive?br_profile_merit(first,count,width,beta,weight,edge,
       u,delta,alpha,driving,delta_driving,target,&constraint,&scale,&acceleration):merit;
   if(positive&&candidate<merit){
    for(int j=0;j<count;++j){const int p=first+j;u[p]+=alpha*delta[p];}
    driving+=alpha*delta_driving;accepted=true;break;
   }
   alpha*=0.5;
  }
  if(!accepted){status[b]=4;return;}
 }
 if(!converged){status[b]=4;return;}
 for(int j=0;j<count;++j){const int p=first+j,k=wet_slot[p];face_speed[face_slot[k]]=u[p];}
 driving_output[b]=driving;discharge_residual[b]=final_constraint;
 acceleration_residual[b]=final_acceleration;
}

extern "C" __global__ void viscosity_equilibrated_traces(
 const double* q,const double* face_speed,const int* owner,const int* bid,
 const int* active,const double* normal,long long nc,long long nf,int nb,
 double dry_tolerance,double speed_limit,double* trace,int* boundary_status,
 int* face_status)
{
 const long long f=(long long)blockIdx.x*blockDim.x+threadIdx.x;
 if(f>=nf)return;
 const int b=bid[f];
 if(b<0||b>=nb||!active[b])return;
 const int c=owner[f];
 if(c<0||c>=nc){face_status[f]=20;atomicOr(boundary_status+b,8);return;}
 const double h=q[c];
 trace[f]=0.0;trace[nf+f]=0.0;trace[2*nf+f]=0.0;
 if(h<=dry_tolerance)return;
 const double nx=normal[f],ny=normal[nf+f],ub=face_speed[f];
 const double vx=q[nc+c]/h,vy=q[2*nc+c]/h;
 const double un=vx*nx+vy*ny;
 const double tx=vx-un*nx,ty=vy-un*ny;
 const double speed=sqrt(ub*ub+tx*tx+ty*ty);
 if(!isfinite(h)||!isfinite(ub)||!isfinite(tx)||!isfinite(ty)||speed>speed_limit){
  face_status[f]=21;atomicOr(boundary_status+b,16);return;
 }
 trace[f]=h;trace[nf+f]=h*(-ub*nx+tx);trace[2*nf+f]=h*(-ub*ny+ty);
}
