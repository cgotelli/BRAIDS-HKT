// Isolated hydraulic GPU port: constant nonsampled modes 2, 3, 6 and 8.
// Boundary sections use deterministic block reductions, not floating atomics.
// Spatially varying roughness is retained; no viscosity-equilibrated profile.
extern "C" __global__ void gpu_section_traces(
 const double* q,const double* bed,const double* roughness,const double* normals,
 const double* length,const int* owner,const int* offsets,const int* faces,
 const int* modes,const double* forcing,long long nc,long long nf,int nb,
 double dry,double speed_limit,double chezy_floor,double* trace,int* status)
{
 const int b=blockIdx.x,lane=threadIdx.x;
 if(b>=nb)return;
 if(lane==0)status[b]=0;
 __syncthreads();
 const int mode=modes[b];
 if(mode!=6 && mode!=8)return;
 __shared__ double sums[128], widths[128];
 double zsum=0.0,wsum=0.0;
 for(int k=offsets[b]+lane;k<offsets[b+1];k+=128) {
  const int f=faces[k],c=owner[f];
  if(q[c]>dry){zsum+=(bed[c]+q[c])*length[f];wsum+=length[f];}
 }
 sums[lane]=zsum;widths[lane]=wsum;__syncthreads();
 for(int span=64;span;span/=2){if(lane<span){sums[lane]+=sums[lane+span];widths[lane]+=widths[lane+span];}__syncthreads();}
 const double zavg=widths[0]>0.0?sums[0]/widths[0]:0.0;
 const double desired=forcing[b],target_depth=forcing[nb+b];
 if(lane==0 && desired>0.0 && ((mode==6 && widths[0]<=0.0)||(mode==8 && target_depth<=dry)))status[b]=1;
 double local_weight=0.0;
 for(int k=offsets[b]+lane;k<offsets[b+1];k+=128){
  const int f=faces[k],c=owner[f];const double h=mode==8?target_depth:q[c];
  if(h<=dry)continue;
  double w=pow(mode==8?h:fmax(zavg-bed[c],0.0),1.5)*pow(fmax(h,chezy_floor),1.0/6.0);
  if(roughness[c]>0.0)w/=roughness[c];local_weight+=w*length[f];
 }
 // All lanes must finish reading zavg before overwriting reduction storage.
 __syncthreads();sums[lane]=local_weight;__syncthreads();
 for(int span=64;span;span/=2){if(lane<span)sums[lane]+=sums[lane+span];__syncthreads();}
 const double total=sums[0];
 if(lane==0 && desired>0.0 && total<=0.0)status[b]=1;
 __syncthreads();
 for(int k=offsets[b]+lane;k<offsets[b+1];k+=128){
  const int f=faces[k],c=owner[f];const double h=mode==8?target_depth:q[c];
  const double nx=normals[f],ny=normals[nf+f];
  double tx=0.0,ty=0.0,ub=0.0;
  if(mode==6 && q[c]>dry){
   const double un=(q[nc+c]*nx+q[2*nc+c]*ny)/q[c];
   tx=q[nc+c]/q[c]-un*nx;ty=q[2*nc+c]/q[c]-un*ny;
  }
  if(h<=dry){trace[f]=0.0;trace[nf+f]=0.0;trace[2*nf+f]=0.0;continue;}
  if(desired>0.0 && total>0.0){
   double w=pow(mode==8?h:fmax(zavg-bed[c],0.0),1.5)*pow(fmax(h,chezy_floor),1.0/6.0);
   if(roughness[c]>0.0)w/=roughness[c];
   ub=desired*(w*length[f])/(total*h*length[f]);
  }
  if(sqrt(ub*ub+tx*tx+ty*ty)>speed_limit)atomicOr(status+b,2);
  trace[f]=h;trace[nf+f]=h*(-ub*nx+tx);trace[2*nf+f]=h*(-ub*ny+ty);
 }
}

extern "C" __global__ void gpu_stage_traces(
 const double* q,const double* bed,const double* normal,const int* owner,
 const int* bid,const int* modes,const double* forcing,long long nc,long long nf,
 int nb,double grav,double dry,double* trace,int* face_status)
{
 const long long f=(long long)blockIdx.x*blockDim.x+threadIdx.x;
 if(f>=nf)return;face_status[f]=0;
 const int b=bid[f];if(b<0 || (modes[b]!=2 && modes[b]!=3))return;
 const int c=owner[f];const double nx=normal[f],ny=normal[nf+f];
 double edge_h=q[c],edge_x=q[nc+c],edge_y=q[2*nc+c],edge_b=bed[c];
 if(modes[b]==2 && q[c]>dry && fabs(trace[3*nf+f]-bed[c])>64.0*2.2204460492503131e-16*fmax(1.0,fabs(bed[c])) &&
    fabs(q[nc+c]*nx+q[2*nc+c]*ny)<q[c]*sqrt(grav*q[c])) {
  edge_b=trace[3*nf+f];edge_h=trace[8*nf+f];
  // Reconstructed conservative momenta use the same raw face depth for a wet owner.
  edge_x=trace[nf+f];edge_y=trace[2*nf+f];
 }
 double h=edge_h,u=0.0;if(h>dry)u=-(edge_x*nx+edge_y*ny)/h;
 trace[f]=0.0;trace[nf+f]=0.0;trace[2*nf+f]=0.0;
 if(h<=dry)return;
 const double wave=sqrt(grav*h),invariant=u-2.0*wave;
 if(u<=-wave){trace[f]=edge_h;trace[nf+f]=edge_x;trace[2*nf+f]=edge_y;return;}
 if(modes[b]==3){face_status[f]=1;return;}
 if(u>=wave){trace[f]=edge_h;return;}
 h=fmax(forcing[nb+b]-edge_b,0.0);
 double ub;
 if(h<=dry || invariant<=-3.0*sqrt(grav*h)){
  h=(invariant/3.0)*(invariant/3.0)/grav;
  if(h<=dry){trace[f]=edge_h;return;}
  ub=-sqrt(grav*h);
 }else{
  ub=invariant+2.0*sqrt(grav*h);
  if(ub>0.0){trace[f]=edge_h;return;}
  if(fabs(ub)>=sqrt(grav*h)){face_status[f]=1;return;}
 }
 const double tx=edge_x/edge_h+u*nx,ty=edge_y/edge_h+u*ny;
 trace[f]=h;trace[nf+f]=h*(-ub*nx+tx);trace[2*nf+f]=h*(-ub*ny+ty);
 if(!isfinite(trace[f]) || !isfinite(trace[nf+f]) || !isfinite(trace[2*nf+f]))face_status[f]=2;
}

// Stock native mode-1 total-Q inlet on unsampled, zero-offset owner-bed faces.
// One boundary owns a block; section sums/root evaluations are reduced across
// its faces. The native shared-stage characteristic equation is solved rather
// than substituting mode 6's free-stage Manning conveyance distribution.
__device__ double gpu_mode1_flux(double stage,int b,int lane,const double* q,
 const double* bed,const double* normal,const double* length,const int* owner,
 const int* offsets,const int* faces,long long nc,long long nf,double grav,double dry,
 double target_u,double* reduction){
 double partial=0.;
 for(int k=offsets[b]+lane;k<offsets[b+1];k+=128){
  const int f=faces[k],c=owner[f];const double depth=q[c];
  double speed=0.,wave=0.;
  if(depth>dry){speed=-(q[nc+c]*normal[f]+q[2*nc+c]*normal[nf+f])/depth;
   wave=sqrt(grav*depth);}
  const bool characteristic=depth>dry&&speed<wave*(1.-1.e-12);
  const double h=fmax(stage-bed[c],0.);
  const double ub=characteristic?speed-2.*wave+2.*sqrt(grav*h):target_u;
  partial+=h*ub*length[f];
 }
 reduction[lane]=partial;__syncthreads();
 for(int span=64;span;span/=2){if(lane<span)reduction[lane]+=reduction[lane+span];__syncthreads();}
 return reduction[0];
}

extern "C" __global__ void gpu_mode1_traces(
 const double* q,const double* bed,const double* normal,const double* length,
 const int* owner,const int* offsets,const int* faces,const int* modes,
 const double* forcing,long long nc,long long nf,int nb,double grav,double dry,
 double speed_limit,double* trace,int* status,int* face_status){
 const int b=blockIdx.x,lane=threadIdx.x;
 if(b>=nb||modes[b]!=1)return;
 __shared__ double values[128],lows[128],highs[128],lo,hi,target_u,stage;
 __shared__ int ready,invalid;
 const double desired=forcing[b],reference_stage=forcing[nb+b];
 double local_area=0.,local_lo=1.7976931348623157e308;
 double local_hi=-1.7976931348623157e308;int local_invalid=0;
 for(int k=offsets[b]+lane;k<offsets[b+1];k+=128){
  const int f=faces[k],c=owner[f];const double h=q[c];
  local_area+=fmax(reference_stage-bed[c],0.)*length[f];
  local_lo=fmin(local_lo,bed[c]);
  local_hi=fmax(local_hi,fmax(bed[c]+h+1.,reference_stage));
  if(h>dry){
   const double u=-(q[nc+c]*normal[f]+q[2*nc+c]*normal[nf+f])/h;
   if(u<=-sqrt(grav*h))local_invalid=1;
  }
 }
 values[lane]=local_area;lows[lane]=local_lo;highs[lane]=local_hi;
 __shared__ int flags[128];flags[lane]=local_invalid;
 __syncthreads();
 for(int span=64;span;span/=2){
  if(lane<span){values[lane]+=values[lane+span];lows[lane]=fmin(lows[lane],lows[lane+span]);
   highs[lane]=fmax(highs[lane],highs[lane+span]);flags[lane]|=flags[lane+span];}
  __syncthreads();
 }
 if(lane==0){
  invalid=flags[0]||(offsets[b]==offsets[b+1])||!isfinite(values[0])||
   !isfinite(lows[0])||!isfinite(highs[0])||desired<0.||
   (desired>0.&&values[0]<=0.);
  lo=lows[0];hi=highs[0];target_u=values[0]>0.?desired/values[0]:0.;ready=0;
 }
 __syncthreads();
 if(invalid){if(lane==0)status[b]=1;return;}
 for(int iteration=0;iteration<60;++iteration){
  const double flux=gpu_mode1_flux(hi,b,lane,q,bed,normal,length,owner,
      offsets,faces,nc,nf,grav,dry,target_u,values);
  if(lane==0){
   if(isfinite(flux)&&flux>=desired)ready=1;
   else hi=lo+2.*(hi-lo);
   if(!isfinite(hi))invalid=1;
  }
  __syncthreads();if(ready||invalid)break;
 }
 if(!ready||invalid){if(lane==0)status[b]=1;return;}
 for(int iteration=0;iteration<90;++iteration){
  if(lane==0){stage=.5*(lo+hi);ready=stage==lo||stage==hi;}
  __syncthreads();if(ready)break;
  const double flux=gpu_mode1_flux(stage,b,lane,q,bed,normal,length,owner,
      offsets,faces,nc,nf,grav,dry,target_u,values);
  if(lane==0){
   if(!isfinite(flux))invalid=1;
   else if(flux<desired)lo=stage;
   else hi=stage;
  }
  __syncthreads();if(invalid)break;
 }
 if(invalid){if(lane==0)status[b]=1;return;}
 if(lane==0)stage=.5*(lo+hi);
 __syncthreads();
 for(int k=offsets[b]+lane;k<offsets[b+1];k+=128){
  const int f=faces[k],c=owner[f];const double depth=q[c];
  double u=0.,wave=0.;
  if(depth>dry){u=-(q[nc+c]*normal[f]+q[2*nc+c]*normal[nf+f])/depth;
   wave=sqrt(grav*depth);}
  const bool characteristic=depth>dry&&u<wave*(1.-1.e-12);
  const double h=fmax(stage-bed[c],0.);
  const double ub=characteristic?u-2.*wave+2.*sqrt(grav*h):target_u;
  trace[f]=0.;trace[nf+f]=0.;trace[2*nf+f]=0.;
  if(h<=dry&&(ub< -1.e-12||(characteristic&&ub>sqrt(grav*h)*(1.+1.e-12))))continue;
  if(ub< -1.e-12||(characteristic&&ub>sqrt(grav*h)*(1.+1.e-12))||
     (h>dry&&fabs(ub)>speed_limit)||!isfinite(h)||!isfinite(ub)){
   face_status[f]=3;atomicOr(status+b,2);continue;
  }
  trace[f]=h;trace[nf+f]=-h*ub*normal[f];trace[2*nf+f]=-h*ub*normal[nf+f];
 }
}

extern "C" __global__ void gpu_boundary_ledger(
 const double* flux,const double* length,const int* offsets,const int* faces,
 int nb,double dt,double* ledger)
{
 const int b=blockIdx.x,lane=threadIdx.x;if(b>=nb)return;
 __shared__ double values[128];double sum=0.0;
 for(int k=offsets[b]+lane;k<offsets[b+1];k+=128){const int f=faces[k];sum+=flux[f]*length[f];}
 values[lane]=sum;__syncthreads();
 for(int span=64;span;span/=2){if(lane<span)values[lane]+=values[lane+span];__syncthreads();}
 if(lane==0)ledger[b]=-dt*values[0];
}

extern "C" __global__ void gpu_pipeline_status(
 const int* primitive,const int* reconstruction,const int* augmented,
 const int* face,const int* cell,const int* boundary,const int* boundary_face,
 long long nc,long long nf,int nb,int* status)
{
 const long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
 if(i>=nc && i>=nf && i>=nb)return;
 int value=0;
 if(i<nc){value|=(primitive[i]!=0);value|=(cell[i]!=0);}
 if(i<nf){value|=(reconstruction[i]!=0);value|=(augmented[i]!=0);
  value|=(face[i]!=0);value|=(boundary_face[i]!=0);}
 if(i<nb)value|=(boundary[i]!=0);status[i]=value;
}
