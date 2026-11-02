// Versioned normal-section boundary quadrature. The root and this closure
// consume the same current piecewise-linear section nodes. No persistent
// subcell state, CPU root, f-wave change, or sediment reconstruction change.
__device__ void int_moments(double eta,double z0,double z1,
                           double* m1,double* m2,double* hm) {
 const double h0=eta-z0,h1=eta-z1;
 *m1=0.;*m2=0.;*hm=fmax(fmax(h0,h1),0.);
 if(*hm<=0.)return;
 const double a=fmax(h0,0.),b=fmax(h1,0.);
 const double fraction=fmin(h0,h1)>0.?1.:(h0>0.?h0/(h0-h1):h1/(h1-h0));
 *m1=0.5*(a+b)*fraction;
 *m2=(a*a+a*b+b*b)*(fraction/3.);
}

extern "C" __global__ void lnt_inputs(const double* bed,const double* q,int nc,int* invalid) {
 const int c=blockIdx.x*blockDim.x+threadIdx.x;
 if(c<nc && (!isfinite(bed[c]) || !isfinite(q[c]) || q[c]<0. ||
             !isfinite(q[1LL*nc+c]) || !isfinite(q[2LL*nc+c])))atomicOr(invalid,1);
}

extern "C" __global__ void lnt_apply(
 const int* owner,const int* faces,const int* offsets,const double* length,
 const double* normal,const double* mid_offset,const double* nodes,const double* bed,const double* q,
 const double* stage,const int* root_status,const double* discharge,const int* invalid,
 int nc,int nf,int nb,int face_policy,double dry,double limit,
 double* trace,int* status,int* face_status,double* wet_area,double* speed,double* integrated,
 double* mean,double* moment2,double* maximum,double* owner_moment2,double* correction) {
 const int b=blockIdx.x,lane=threadIdx.x;if(b>=nb)return;
 __shared__ double sum[128],area,velocity;__shared__ int flags;
 if(lane==0){
  flags=invalid[0];area=nan("");velocity=nan("");
  if(root_status[b])flags|=2;
  if(!isfinite(stage[b]) || !isfinite(discharge[b]) || discharge[b]<0.)flags|=1;
 }
 __syncthreads();
 double value=0.;
 // All lanes enter; flags may change atomically inside, never guard a barrier.
 for(int p=offsets[b]+lane;p<offsets[b+1];p+=128){
  const int f=faces[p],c=owner[f];const double z0=nodes[p+b],z1=nodes[p+b+1];
  double h,h2,hm,ch,ch2,chm;
  int_moments(stage[b],z0,z1,&h,&h2,&hm);
  int_moments(bed[c]+q[c],z0,z1,&ch,&ch2,&chm);
  // Policy 1 reproduces the native mode-9 face closure exactly.  The live
  // section root remains the same piecewise-linear surveyed root, but the
  // native trace assigns wetness and section Q from the sampled midpoint
  // depth.  Policy 0 retains the previously qualified subface-integrated
  // closure for legacy runs.
  if(face_policy==1){
   h=fmax(stage[b]-bed[c]-mid_offset[f],0.);
   // Native mode 9 zeroes the complete sampled trace when its midpoint
   // depth does not clear dry.  Consequently that face contributes neither
   // h^2 pressure nor wave speed.  The owner-column bed-path correction below
   // still uses (h_owner+0)*offset, exactly as native FLOW does.
   if(h<=dry)h=0.;
   h2=h*h;hm=h;
   ch2=q[c]*q[c];
  }
  mean[f]=h;moment2[f]=h2;maximum[f]=hm;owner_moment2[f]=ch2;
  correction[f]=q[c]>dry?(face_policy==1?(q[c]+h)*mid_offset[f]:q[c]*q[c]-ch2):0.;
  if(!isfinite(z0)||!isfinite(z1)||!isfinite(h)||!isfinite(h2)||!isfinite(hm)||
     !isfinite(ch2)||!isfinite(correction[f]))atomicOr(&flags,16);
  if(h>dry)value+=h*length[f];
 }
 sum[lane]=value;__syncthreads();
 for(int d=64;d;d/=2){if(lane<d)sum[lane]+=sum[lane+d];__syncthreads();}
 if(lane==0 && !flags){
  area=sum[0];
  if(!isfinite(area)||area<0.)flags|=16;
  else if(discharge[b]>0. && area<=0.)flags|=4;
  else {velocity=discharge[b]>0.?discharge[b]/area:0.;
   if(!isfinite(velocity))flags|=16;
   else if(velocity>limit)flags|=8;
  }
 }
 __syncthreads();
 value=0.;
 if(!flags)for(int p=offsets[b]+lane;p<offsets[b+1];p+=128){
  const int f=faces[p];const double h=mean[f];
  if(h>dry){
   const double hu=-h*velocity*normal[f],hv=-h*velocity*normal[1LL*nf+f];
   const double flow=-(hu*normal[f]+hv*normal[1LL*nf+f])*length[f];
   if(!isfinite(hu)||!isfinite(hv)||!isfinite(flow))atomicOr(&flags,16);
   else {trace[f]=h;trace[1LL*nf+f]=hu;trace[2LL*nf+f]=hv;value+=flow;}
  }
 }
 sum[lane]=value;__syncthreads();
 for(int d=64;d;d/=2){if(lane<d)sum[lane]+=sum[lane+d];__syncthreads();}
 if(lane==0){
  if(!isfinite(sum[0]))flags|=16;
  status[b]=flags;wet_area[b]=area;speed[b]=velocity;integrated[b]=flags?nan(""):sum[0];
 }
 __syncthreads();
 for(int p=offsets[b]+lane;p<offsets[b+1];p+=128){
  const int f=faces[p];face_status[f]=flags;
  if(flags)trace[f]=trace[1LL*nf+f]=trace[2LL*nf+f]=0.;
 }
}

extern "C" __global__ void lnt_flux(
 const int* faces,const int* offsets,const double* normal,const double* trace,
 const int* status,const double* moment2,const double* maximum,const double* correction,
 const double* velocity,int nb,int nf,double grav,double dry,
 double* fl,double* fr,double* speed,int* face_status) {
 const int b=blockIdx.x,lane=threadIdx.x;if(b>=nb)return;
 for(int p=offsets[b]+lane;p<offsets[b+1];p+=128){
  const int f=faces[p];
  if(status[b]){face_status[f]|=status[b];continue;}
  const double nx=normal[f],ny=normal[1LL*nf+f],h=trace[f];
  const double hu=trace[1LL*nf+f],hv=trace[2LL*nf+f];
  const double mass=hu*nx+hv*ny;
  // Exact hydrostatic face moment plus the interior-column bed path.
  // At eta_boundary=eta_owner this sums to g*h_owner^2/2, including
  // partially wetted segments. No static midpoint bed offset is reused.
  const double pressure=0.5*grav*(moment2[f]+correction[f]);
  const double fx=(h>dry?mass*hu/h:0.)+pressure*nx;
  const double fy=(h>dry?mass*hv/h:0.)+pressure*ny;
  fl[f]=fr[f]=mass;fl[1LL*nf+f]=fr[1LL*nf+f]=fx;fl[2LL*nf+f]=fr[2LL*nf+f]=fy;
  speed[f]=sqrt(grav*maximum[f])+(h>dry?fabs(velocity[b]):0.);
  if(!isfinite(mass)||!isfinite(fx)||!isfinite(fy)||!isfinite(speed[f]))face_status[f]|=16;
 }
}
