// Exact law-9 mapping of TELEMAC-GAIA LIEBOR=KENT at either a free-capacity
// or prescribed-feed inlet. Only the boundary face bed is fixed. Owner-cell
// water, momentum, bed inventory and free-surface stage remain live.

extern "C" __global__ void gpu_gaia_fixed_inlet_traces(
 const double* q,const double* bed,const double* roughness,const double* normal,
 const double* length,const int* owner,const int* offsets,const int* faces,
 const int* modes,const double* forcing,const int* fixed_mode,
 const double* fixed_bed,long long nc,long long nf,int nb,double dry,
 double speed_limit,double chezy_floor,double* trace,int* status,
 int* face_status)
{
 const int b=blockIdx.x,lane=threadIdx.x;
 if(b>=nb || fixed_mode[b]!=1)return;
 __shared__ double values[128],widths[128],stage,total;
 __shared__ int flags[128],invalid;
 double zsum=0.0,wet_width=0.0;int local_invalid=(modes[b]!=6);
 if(lane==0){status[b]=0;invalid=0;stage=0.0;total=0.0;}
 for(int k=offsets[b]+lane;k<offsets[b+1];k+=128){
  const int f=faces[k],c=owner[f];
  if(c<0 || c>=nc || !isfinite(q[c]) || !isfinite(bed[c]) ||
     !isfinite(length[f]) || length[f]<0.0 || !isfinite(fixed_bed[f]) ||
     !isfinite(roughness[c]) || roughness[c]<=0.0)local_invalid=1;
  if(c>=0 && c<nc && q[c]>dry){zsum+=(bed[c]+q[c])*length[f];wet_width+=length[f];}
 }
 values[lane]=zsum;widths[lane]=wet_width;flags[lane]=local_invalid;__syncthreads();
 for(int span=64;span;span/=2){
  if(lane<span){values[lane]+=values[lane+span];widths[lane]+=widths[lane+span];flags[lane]|=flags[lane+span];}
  __syncthreads();
 }
 if(lane==0){
  invalid=flags[0] || !isfinite(forcing[b]) || forcing[b]<0.0 ||
          !isfinite(values[0]) || !isfinite(widths[0]) ||
          (forcing[b]>0.0 && widths[0]<=0.0);
  stage=widths[0]>0.0?values[0]/widths[0]:0.0;
 }
 __syncthreads();
 double local_weight=0.0;local_invalid=0;
 for(int k=offsets[b]+lane;k<offsets[b+1];k+=128){
  const int f=faces[k],c=owner[f];
  const double h=fmax(stage-fixed_bed[f],0.0);
  const double weight=h>dry?pow(h,1.5)*pow(fmax(h,chezy_floor),1.0/6.0)/roughness[c]:0.0;
  if(!isfinite(h) || !isfinite(weight))local_invalid=1;
  local_weight+=weight*length[f];
 }
 values[lane]=local_weight;flags[lane]=local_invalid;__syncthreads();
 for(int span=64;span;span/=2){
  if(lane<span){values[lane]+=values[lane+span];flags[lane]|=flags[lane+span];}
  __syncthreads();
 }
 if(lane==0){
  total=values[0];
  invalid|=flags[0] || !isfinite(total) || (forcing[b]>0.0 && total<=0.0);
  if(invalid)status[b]=4;
 }
 __syncthreads();
 for(int k=offsets[b]+lane;k<offsets[b+1];k+=128){
  const int f=faces[k],c=owner[f];const double nx=normal[f],ny=normal[nf+f];
  const double h=fmax(stage-fixed_bed[f],0.0);
  double tx=0.0,ty=0.0,ub=0.0;
  face_status[f]=0;trace[f]=0.0;trace[nf+f]=0.0;trace[2*nf+f]=0.0;
  if(invalid){face_status[f]=8;continue;}
  if(q[c]>dry){
   const double un=(q[nc+c]*nx+q[2*nc+c]*ny)/q[c];
   tx=q[nc+c]/q[c]-un*nx;ty=q[2*nc+c]/q[c]-un*ny;
  }
  const double weight=h>dry?pow(h,1.5)*pow(fmax(h,chezy_floor),1.0/6.0)/roughness[c]:0.0;
  if(h<=dry)continue;
  if(forcing[b]>0.0)ub=forcing[b]*weight/(total*h);
  if(!isfinite(tx) || !isfinite(ty) || !isfinite(ub) ||
     sqrt(ub*ub+tx*tx+ty*ty)>speed_limit){
   face_status[f]=16;atomicOr(status+b,8);continue;
  }
  trace[f]=h;trace[nf+f]=h*(-ub*nx+tx);trace[2*nf+f]=h*(-ub*ny+ty);
 }
}

extern "C" __global__ void gpu_gaia_fixed_inlet_pressure(
 const double* q,const double* bed,const double* trace,const double* normal,
 const int* owner,const int* bid,const int* fixed_mode,const double* fixed_bed,
 long long nc,long long nf,int nb,double grav,double dry,double* fl,double* fr,
 int* face_status)
{
 const long long f=(long long)blockIdx.x*blockDim.x+threadIdx.x;
 if(f>=nf)return;
 const int b=bid[f];if(b<0 || b>=nb || fixed_mode[b]!=1)return;
 const int c=owner[f];const double h=q[c],face_h=trace[f];
 if(c<0 || c>=nc || !isfinite(h) || !isfinite(face_h) || face_h<0.0 ||
    !isfinite(bed[c]) || !isfinite(fixed_bed[f]) || !isfinite(grav) || grav<0.0){
  atomicOr(face_status+f,32);return;
 }
 if(h>dry){
  const double dz=fixed_bed[f]-bed[c];
  const double hgeom=fmax(h-dz,0.0);
  const double correction=0.5*grav*(h*h-hgeom*hgeom);
  if(!isfinite(correction)){atomicOr(face_status+f,32);return;}
  fl[nf+f]+=correction*normal[f];fl[2*nf+f]+=correction*normal[nf+f];
 }
 fl[0*nf+f]=fl[f];
 for(int j=0;j<3;++j)fr[j*nf+f]=fl[j*nf+f];
}
