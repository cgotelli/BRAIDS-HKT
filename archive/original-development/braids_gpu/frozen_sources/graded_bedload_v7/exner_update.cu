// FP64 fixed-grid bed-solid donor limiting and Exner inventory update.
// Input fluxes are reconstructed physical bedload mass fluxes, kg/(m s).
// This does NOT calculate sediment face reconstruction or coupled SWE fluxes.
extern "C" __global__ void bed_scale(const double* raw,const int* owner,const int* neighbour,
 const int* bid,const int* active,long long nf,int ng,double mf,double* flux,int* status){
 long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
 if(i>=nf*ng)return;const int f=i%nf,n=neighbour[f];double v=raw[i];
 status[i]=(!isfinite(v) || (v!=0. && ((n<0 && bid[f]<0)||!active[owner[f]]||(n>=0 && !active[n]))));
 flux[i]=mf==1.0?v:mf*v;if(!isfinite(flux[i]))status[i]=1;
}
extern "C" __global__ void bed_donors(const double* mass,const double* area,const double* length,
 const int* offset,const int* faces,const int* signs,const int* active,
 const double* flux,long long nc,long long nf,int ng,double dt,double* theta){
 long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=nc*ng)return;
 const int c=i%nc,g=i/nc;double outgoing=0.;theta[i]=1.;if(!active[c])return;
 for(int k=offset[c];k<offset[c+1];++k){int f=faces[k];outgoing+=fmax(signs[k]*flux[g*nf+f],0.)*length[f];}
 if(dt*outgoing>mass[i]*area[c])theta[i]=mass[i]*area[c]/(dt*outgoing)*(1.-16.*2.2204460492503131e-16);
}
extern "C" __global__ void bed_limit(const int* owner,const int* neighbour,
 const double* theta,long long nc,long long nf,int ng,double* flux){
 long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=nf*ng)return;
 const int f=i%nf,g=i/nf,donor=flux[i]<0.?neighbour[f]:owner[f];
 if(donor>=0)flux[i]*=theta[g*nc+donor];
}
extern "C" __global__ void bed_gather(const double* mass,const double* area,const double* length,
 const int* offset,const int* faces,const int* signs,const int* active,const double* flux,
 const double* rock,const double* bulk,const int* face_status,long long nc,long long nf,int ng,double dt,
 double* next,double* bed,int* status){
 long long c=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
 status[c]=0;double z=rock[c];
 for(int g=0;g<ng;++g){
  double sumflux=0.;
  if(!isfinite(mass[g*nc+c])||mass[g*nc+c]<0.)status[c]=1;
  for(int k=offset[c];k<offset[c+1];++k){int f=faces[k];
   if(face_status[g*nf+f])status[c]=1;
   if(active[c])sumflux+=signs[k]*(length[f]/area[c])*flux[g*nf+f];}
  double m=active[c]?mass[g*nc+c]-dt*sumflux:mass[g*nc+c];
  if(m<0. && fabs(m)<=128.*2.2250738585072014e-308)m=0.;
  if(!isfinite(m)||m<0.)status[c]=1;
  next[g*nc+c]=m;z+=m/bulk[g];
 }bed[c]=z;if(!isfinite(z))status[c]=1;
}
extern "C" __global__ void bed_ledger(const double* flux,const double* length,const int* offsets,const int* faces,
 long long nf,int ng,int nb,double dt,double* ledger){
 const int b=blockIdx.x,g=blockIdx.y,lane=threadIdx.x;if(b>=nb||g>=ng)return;
 __shared__ double work[128];double v=0.;
 for(int k=offsets[b]+lane;k<offsets[b+1];k+=128){const int f=faces[k];v-=dt*length[f]*flux[g*nf+f];}
 work[lane]=v;__syncthreads();
 for(int n=64;n;n/=2){if(lane<n)work[lane]+=work[lane+n];__syncthreads();}
 if(lane==0)ledger[g*nb+b]=work[0];
}
