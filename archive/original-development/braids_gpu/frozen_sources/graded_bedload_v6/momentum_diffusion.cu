// FP64 native v35 velocity-Laplacian diffusion, fixed qualified wet mask.
// Same recovered flux, dissipation correction, SSPRK2 and energy rejection.
extern "C" __global__ void md_mask(const double* q,const double* bed,const double* xy,
 const double* normal,const double* length,const int* owner,const int* neighbour,
 const double* conductance,long long nc,long long nf,double dry,double nu,int* status){
 long long f=blockIdx.x*(long long)blockDim.x+threadIdx.x;if(f>=nf)return;
 status[f]=0;int l=owner[f],r=neighbour[f];if(r<0)return;
 double h1=q[l],h2=q[r],d0=xy[r]-xy[l],d1=xy[nc+r]-xy[nc+l];
 double dn=d0*normal[f]+d1*normal[nf+f],actual=0.;
 if(h1>dry&&h2>dry){double hf=.5*((bed[l]+h1)+(bed[r]+h2))-fmax(bed[l],bed[r]);
  if(hf>.5*dry){double act=hf<dry?fmax(0.,fmin(1.,(hf-.5*dry)/(.5*dry))):1.;actual=nu*length[f]*act/dn;}}
 if(!isfinite(actual)||actual!=conductance[f])status[f]=1;
}
extern "C" __global__ void md_velocity(const double* momentum,const double* depth,long long nc,
 double dry,double* velocity,int* status){
 long long c=blockIdx.x*(long long)blockDim.x+threadIdx.x;if(c>=nc)return;
 status[c]=0;velocity[c]=0.;velocity[nc+c]=0.;
 if(!isfinite(depth[c])||!isfinite(momentum[c])||!isfinite(momentum[nc+c])){status[c]=1;return;}
 if(depth[c]>dry){velocity[c]=momentum[c]/depth[c];velocity[nc+c]=momentum[nc+c]/depth[c];}
}
extern "C" __global__ void md_recover(const double* velocity,const double* xy,const double* area,
 const int* cells,const int* order,const int* valid,const int* stencils,const int* counts,
 const double* inverse,const double* affine,long long nc,long long nr,double* polynomial,int* status){
 long long s=blockIdx.x*(long long)blockDim.x+threadIdx.x;if(s>=nr)return;
 for(int k=0;k<10;++k)polynomial[k*nr+s]=0.;status[s]=0;if(!valid[s])return;
 int c=cells[s],n=order[s]==2?5:2;double h=sqrt(area[c]),rhs[10]={0.};
 const double rt=1.414213562373095048801688724209698;
 for(int k=0;k<counts[s];++k){int r=stencils[k*nr+s];double zx=(xy[r]-xy[c])/h,zy=(xy[nc+r]-xy[nc+c])/h;
  double w=1./(zx*zx+zy*zy),row[5]={zx,zy,zx*zx/rt,zx*zy,zy*zy/rt};
  double d0=velocity[r]-velocity[c],d1=velocity[nc+r]-velocity[nc+c];
  for(int i=0;i<n;++i){rhs[i]+=w*row[i]*d0;rhs[5+i]+=w*row[i]*d1;}}
 for(int component=0;component<2;++component)for(int i=0;i<n;++i){double value=0.;
  for(int j=0;j<n;++j)value+=(n==5?inverse[(i*5+j)*nr+s]:affine[(i*2+j)*nr+s])*rhs[component*5+j];
  polynomial[(component*5+i)*nr+s]=value;if(!isfinite(value))status[s]=1;}
}
__device__ void md_trace(int c,long long f,int slot,const int* order,const double* p,
 const double* area,const double* xy,const double* mid,const double* normal,long long nc,long long nf,
 long long nr,double* out){
 const double rt=1.414213562373095048801688724209698;
 double h=sqrt(area[c]),zx=(mid[f]-xy[c])/h,zy=(mid[nf+f]-xy[nc+c])/h;
 double nx=normal[f],ny=normal[nf+f];
 double row[5]={nx/h,ny/h,rt*nx*zx/h,(nx*zy+ny*zx)/h,rt*ny*zy/h};
 int n=order[slot]==2?5:2;
 for(int component=0;component<2;++component){out[component]=0.;
  for(int i=0;i<n;++i)out[component]+=row[i]*p[(component*5+i)*nr+slot];}
}
extern "C" __global__ void md_flux(const double* velocity,const double* area,const double* xy,
 const double* mid,const double* normal,const int* owner,const int* neighbour,
 const double* conductance,const double* diffusivity,const int* corrected,const int* slot,
 const int* order,const int* valid,const double* polynomial,long long nc,long long nf,long long nr,
 double* primary,double* correction,double* d0,double* dc,int* status){
 long long f=blockIdx.x*(long long)blockDim.x+threadIdx.x;if(f>=nf)return;
 primary[f]=primary[nf+f]=correction[f]=correction[nf+f]=d0[f]=dc[f]=0.;status[f]=0;
 if(conductance[f]<=0.)return;int l=owner[f],r=neighbour[f];
 double du[2]={velocity[r]-velocity[l],velocity[nc+r]-velocity[nc+l]};
 for(int k=0;k<2;++k)primary[k*nf+f]=conductance[f]*du[k];
 if(corrected[f]&&slot[l]>=0&&slot[r]>=0&&valid[slot[l]]&&valid[slot[r]]){
  double left[2],right[2];md_trace(l,f,slot[l],order,polynomial,area,xy,mid,normal,nc,nf,nr,left);
  md_trace(r,f,slot[r],order,polynomial,area,xy,mid,normal,nc,nf,nr,right);
  for(int k=0;k<2;++k)correction[k*nf+f]=.5*diffusivity[f]*(left[k]+right[k])-primary[k*nf+f];}
 d0[f]=du[0]*primary[f]+du[1]*primary[nf+f];dc[f]=du[0]*correction[f]+du[1]*correction[nf+f];
 if(!isfinite(d0[f])||!isfinite(dc[f]))status[f]=1;
}
extern "C" __global__ void md_factor(const double* d0,const double* dc,double* factor,int* status){
 if(blockIdx.x||threadIdx.x)return;status[0]=0;factor[0]=1.;
 if(!isfinite(d0[0])||!isfinite(dc[0])||d0[0]<0.){status[0]=1;return;}
 if(dc[0]<-.95*d0[0])factor[0]=d0[0]>0.?.95*d0[0]/(-dc[0]):0.;
 if(!isfinite(factor[0])||factor[0]<0.||factor[0]>1.)status[0]=1;
}
extern "C" __global__ void md_gather(const double* stage,const double* depth,const double* area,
 const double* primary,const double* correction,const double* factor,const int* offset,
 const int* faces,const int* signs,long long nc,long long nf,double dry,double ds,double* candidate){
 long long c=blockIdx.x*(long long)blockDim.x+threadIdx.x;if(c>=nc)return;
 for(int component=0;component<2;++component){double change=0.;
  if(depth[c]>dry)for(int k=offset[c];k<offset[c+1];++k){int f=faces[k];
   double flux=primary[component*nf+f]+factor[0]*correction[component*nf+f];change+=signs[k]*flux;}
  candidate[component*nc+c]=stage[component*nc+c]+ds*depth[c]*change/area[c];}
}
extern "C" __global__ void md_energy(const double* momentum,const double* depth,const double* area,
 long long nc,double dry,double* energy){
 long long c=blockIdx.x*(long long)blockDim.x+threadIdx.x;if(c>=nc)return;
 energy[c]=0.;if(depth[c]>dry){double u=momentum[c]/depth[c],v=momentum[nc+c]/depth[c];energy[c]=.5*area[c]*(u*u+v*v);}
}

// Graph variant: dt is a resident parameter, never frozen in a captured graph.
extern "C" __global__ void md_gather_dynamic(const double* stage,const double* depth,const double* area,
 const double* primary,const double* correction,const double* factor,const int* offset,
 const int* faces,const int* signs,long long nc,long long nf,double dry,const double* ds,double* candidate){
 long long c=blockIdx.x*(long long)blockDim.x+threadIdx.x;if(c>=nc)return;
 for(int component=0;component<2;++component){double change=0.;
  if(depth[c]>dry)for(int k=offset[c];k<offset[c+1];++k){int f=faces[k];
   double flux=primary[component*nf+f]+factor[0]*correction[component*nf+f];change+=signs[k]*flux;}
  candidate[component*nc+c]=stage[component*nc+c]+ds[0]*depth[c]*change/area[c];}
}
extern "C" __global__ void md_trial_begin(const int* mask_status,double* summary){
 if(blockIdx.x||threadIdx.x)return;
 summary[0]=mask_status[0];summary[1]=0.;summary[2]=0.;summary[3]=1.;
}
extern "C" __global__ void md_trial_operator_check(const int* cell_status,const int* face_status,
 const int* recovery_status,const int* scalar_status,const double* factor,double* summary){
 if(blockIdx.x||threadIdx.x)return;
 // The native controller stops after a failed Euler energy test. The graph
 // still runs its private second stage, but that unreachable stage must not
 // turn a retry into a fatal error or contribute a diagnostic correction.
 if(summary[2])return;
 if(cell_status[0]||face_status[0]||recovery_status[0]||scalar_status[0])summary[1]=1.;
 summary[3]=fmin(summary[3],factor[0]);
}
extern "C" __global__ void md_trial_energy_check(const double* before,const double* after,double* summary){
 if(blockIdx.x||threadIdx.x)return;
 const double eps=2.220446049250313080847263336181640625e-16;
 const double tiny=2.225073858507201383090232717332404064e-308;
 if(!isfinite(before[0])||!isfinite(after[0])||after[0]>before[0]+128.*eps*fmax(before[0],tiny))summary[2]=1.;
}
