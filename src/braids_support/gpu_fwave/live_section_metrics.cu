// Isolated production-section geometry/conveyance; no boundary root or trace.
// All dynamic inputs and outputs are resident FP64. No atomics in numeric sums.
extern "C" __global__ void lsm_nodes(
 const int* owner,const int* faces,const int* offsets,const int* boundary,
 const double* left,const double* right,const double* bed,const double* stage,
 int nc,int nbf,int nb,double* nodes,int* status,int* global_status)
{
 const int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<nc && !isfinite(bed[i]))atomicOr(global_status,1);
 if(i<nb && !isfinite(stage[i]))atomicOr(status+i,1);
 if(i>=nbf)return;
 const int b=boundary[i],f=faces[i],node=i+b;
 const double zleft=bed[owner[f]]+left[f];
 double z=zleft;
 if(i>offsets[b]){
  const int previous=faces[i-1];
  const double previous_right=bed[owner[previous]]+right[previous];
  z=0.5*(previous_right+zleft);
 }
 nodes[node]=z;if(!isfinite(z))atomicOr(status+b,2);
 if(i==offsets[b+1]-1){
  const double zr=bed[owner[f]]+right[f];nodes[node+1]=zr;
  if(!isfinite(zr))atomicOr(status+b,2);
 }
}

extern "C" __global__ void lsm_segments(
 const int* faces,const int* offsets,const int* boundary,const double* length,
 const double* nodes,const double* stage,int nbf,
 double* area,double* width,double* perimeter,int* status)
{
 const int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=nbf)return;
 const int b=boundary[k],node=k+b,f=faces[k];
 const double eta=stage[b],z0=nodes[node],z1=nodes[node+1];
 const double h0=eta-z0,h1=eta-z1;
 double a=0.,w=0.,p=0.;
 if(!isfinite(h0)||!isfinite(h1)||!isfinite(z1-z0)){
  atomicOr(status+b,2);area[k]=width[k]=perimeter[k]=0.;return;
 }
 if(k==offsets[b] && eta>z0)p=p+eta-z0;
 if(fmax(h0,h1)>0.){
  double fraction;
  if(fmin(h0,h1)>0.)fraction=1.;
  else if(h0>0.)fraction=h0/(h0-h1);
  else fraction=h1/(h1-h0);
  w=length[f]*fraction;
  a=0.5*(fmax(h0,0.)+fmax(h1,0.))*w;
  const double segment_perimeter=hypot(length[f],z1-z0)*fraction;
  p=p+segment_perimeter;
 }
 if(k==offsets[b+1]-1 && eta>z1)p=p+eta-z1;
 if(!isfinite(a)||!isfinite(w)||!isfinite(p)||a<0.||w<0.||p<0.)atomicOr(status+b,2);
 area[k]=a;width[k]=w;perimeter[k]=p;
}

extern "C" __global__ void lsm_groups(
 const int* offsets,const int* boundary,const double* manning,
 const double* segment_area,const double* segment_perimeter,
 int ng,double* group_conveyance,int* status)
{
 const int g=blockIdx.x,lane=threadIdx.x;if(g>=ng)return;
 __shared__ double aa[128],pp[128];double a=0.,p=0.;
 for(int k=offsets[g]+lane;k<offsets[g+1];k+=128){a+=segment_area[k];p+=segment_perimeter[k];}
 aa[lane]=a;pp[lane]=p;__syncthreads();
 for(int span=64;span;span/=2){if(lane<span){aa[lane]+=aa[lane+span];pp[lane]+=pp[lane+span];}__syncthreads();}
 if(lane==0){
  double value=0.;
  if(!isfinite(aa[0])||!isfinite(pp[0])||aa[0]<0.||pp[0]<0.)atomicOr(status+boundary[g],2);
  else if(aa[0]>0. && pp[0]>0.)value=aa[0]*pow(aa[0]/pp[0],2.0/3.0)/manning[g];
  if(!isfinite(value)||value<0.)atomicOr(status+boundary[g],2);
  group_conveyance[g]=value;
 }
}

extern "C" __global__ void lsm_totals(
 const int* offsets,const int* group_offsets,const double* segment_area,
 const double* segment_width,const double* group_conveyance,int nb,
 const int* global_status,int* status,double* area,double* width,double* conveyance)
{
 const int b=blockIdx.x,lane=threadIdx.x;if(b>=nb)return;
 __shared__ double aa[128],ww[128],kk[128];double a=0.,w=0.,k=0.;
 for(int i=offsets[b]+lane;i<offsets[b+1];i+=128){a+=segment_area[i];w+=segment_width[i];}
 for(int i=group_offsets[b]+lane;i<group_offsets[b+1];i+=128)k+=group_conveyance[i];
 aa[lane]=a;ww[lane]=w;kk[lane]=k;__syncthreads();
 for(int span=64;span;span/=2){if(lane<span){aa[lane]+=aa[lane+span];ww[lane]+=ww[lane+span];kk[lane]+=kk[lane+span];}__syncthreads();}
 if(lane==0){
  int flag=status[b]|global_status[0];
  if(!isfinite(aa[0])||!isfinite(ww[0])||!isfinite(kk[0])||aa[0]<0.||ww[0]<0.||kk[0]<0.)flag|=2;
  status[b]=flag;
  area[b]=flag?nan(""):aa[0];width[b]=flag?nan(""):ww[0];conveyance[b]=flag?nan(""):kk[0];
 }
}
