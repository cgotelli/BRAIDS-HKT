// Isolated all-pool mode9 trace closure, downstream of a successful live root.
// FP64 field arithmetic. No Riemann solve, host reduction or accepted-state write.
extern "C" __global__ void lnt_inputs(const double* bed,const double* q,int nc,int* invalid)
{
 const int c=blockIdx.x*blockDim.x+threadIdx.x;
 if(c<nc && (!isfinite(bed[c]) || !isfinite(q[c]) || !isfinite(q[1LL*nc+c])
             || !isfinite(q[2LL*nc+c]) || q[c]<0.))atomicOr(invalid,1);
}

extern "C" __global__ void lnt_apply(const int* owner,const int* faces,const int* offsets,
 const double* length,const double* mid,const double* normal,const double* bed,
 const double* stage,const int* root_status,const double* discharge,
 const int* invalid,int nf,int nb,double dry,double limit,
 double* trace,int* status,int* face_status,double* wet_area,double* speed,double* integrated)
{
 const int b=blockIdx.x,lane=threadIdx.x;if(b>=nb)return;
 __shared__ double sum[128],area,velocity;__shared__ int flags;
 if(lane==0){
  flags=invalid[0];area=nan("");velocity=nan("");
  if(root_status[b]!=0)flags|=2;
  if(!isfinite(stage[b]) || !isfinite(discharge[b]) || discharge[b]<0.)flags|=1;
 }
 __syncthreads();
 double value=0.;
 if(!flags)for(int p=offsets[b]+lane;p<offsets[b+1];p+=128){
  const int f=faces[p];const double raw=stage[b]-bed[owner[f]]-mid[f];
  if(!isfinite(raw))atomicOr(&flags,16);
  else {const double h=fmax(raw,0.);if(h>dry)value+=h*length[f];}
 }
 sum[lane]=value;__syncthreads();
 for(int d=64;d;d/=2){if(lane<d)sum[lane]+=sum[lane+d];__syncthreads();}
 if(lane==0 && !flags){
  area=sum[0];
  if(!isfinite(area) || area<0.)flags|=16;
  else if(discharge[b]>0. && area<=0.)flags|=4;
  else{
   velocity=discharge[b]>0.?discharge[b]/area:0.;
   if(!isfinite(velocity))flags|=16;
   else if(fabs(velocity)>limit)flags|=8;
  }
 }
 __syncthreads();
 value=0.;
 if(!flags)for(int p=offsets[b]+lane;p<offsets[b+1];p+=128){
  const int f=faces[p];const double h=fmax(stage[b]-bed[owner[f]]-mid[f],0.);
  if(h>dry){
   const double hu=(-h*velocity)*normal[f],hv=(-h*velocity)*normal[1LL*nf+f];
   const double flow=-(hu*normal[f]+hv*normal[1LL*nf+f])*length[f];
   if(!isfinite(hu)||!isfinite(hv)||!isfinite(flow))atomicOr(&flags,16);
   else {trace[f]=h;trace[1LL*nf+f]=hu;trace[2LL*nf+f]=hv;value+=flow;}
  }
 }
 sum[lane]=value;__syncthreads();
 for(int d=64;d;d/=2){if(lane<d)sum[lane]+=sum[lane+d];__syncthreads();}
 if(lane==0){
  if(!isfinite(sum[0]))flags|=16;
  status[b]=flags;wet_area[b]=area;speed[b]=velocity;
  integrated[b]=flags?nan(""):sum[0];
 }
 __syncthreads();
 for(int p=offsets[b]+lane;p<offsets[b+1];p+=128){
  const int f=faces[p];face_status[f]=flags;
  if(flags)trace[f]=trace[1LL*nf+f]=trace[2LL*nf+f]=0.;
 }
}
