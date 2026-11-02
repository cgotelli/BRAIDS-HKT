// Typed C/M/W section weighting for stock total-Q/depth inlets (modes 6/8).
// This translation unit is compiled after the archived hydraulic_roughness.cu
// so br_roughness_state remains the one resident constitutive implementation.
extern "C" __global__ void typed_section_traces(
 const double* q,const double* bed,const double* parameter,const double* normals,
 const double* length,const int* owner,const int* offsets,const int* faces,
 const int* modes,const double* forcing,long long nc,long long nf,int nb,
 double dry,double speed_limit,int roughness_mode,double gravity,
 double minimum_work_depth,int waqua_white_colebrook,double* trace,int* status)
{
 const int b=blockIdx.x,lane=threadIdx.x;
 if(b>=nb)return;
 if(lane==0)status[b]=0;
 __syncthreads();
 const int mode=modes[b];
 if(mode!=6 && mode!=8)return;
 __shared__ double sums[128],widths[128];
 double zsum=0.0,wsum=0.0;
 for(int k=offsets[b]+lane;k<offsets[b+1];k+=128){
  const int f=faces[k],c=owner[f];
  if(q[c]>dry){zsum+=(bed[c]+q[c])*length[f];wsum+=length[f];}
 }
 sums[lane]=zsum;widths[lane]=wsum;__syncthreads();
 for(int span=64;span;span/=2){
  if(lane<span){sums[lane]+=sums[lane+span];widths[lane]+=widths[lane+span];}
  __syncthreads();
 }
 const double zavg=widths[0]>0.0?sums[0]/widths[0]:0.0;
 const double desired=forcing[b],target_depth=forcing[nb+b];
 if(lane==0 && desired>0.0 &&
    ((mode==6 && widths[0]<=0.0)||(mode==8 && target_depth<=dry)))status[b]=1;
 double local_weight=0.0;
 for(int k=offsets[b]+lane;k<offsets[b+1];k+=128){
  const int f=faces[k],c=owner[f];const double h=mode==8?target_depth:q[c];
  const double section_depth=mode==8?h:fmax(zavg-bed[c],0.0);
  if(h<=dry||section_depth<=0.0)continue;
  double chezy=0.0,z0=0.0,drag=0.0;
  const bool valid=br_roughness_state(roughness_mode,h,parameter[c],gravity,
      minimum_work_depth,waqua_white_colebrook,&chezy,&z0,&drag);
  if(!valid){atomicOr(status+b,4);continue;}
  local_weight+=pow(section_depth,1.5)*chezy*length[f];
 }
 __syncthreads();sums[lane]=local_weight;__syncthreads();
 for(int span=64;span;span/=2){
  if(lane<span)sums[lane]+=sums[lane+span];__syncthreads();
 }
 const double total=sums[0];
 if(lane==0 && desired>0.0 && total<=0.0)status[b]|=1;
 __syncthreads();
 for(int k=offsets[b]+lane;k<offsets[b+1];k+=128){
  const int f=faces[k],c=owner[f];const double h=mode==8?target_depth:q[c];
  const double section_depth=mode==8?h:fmax(zavg-bed[c],0.0);
  const double nx=normals[f],ny=normals[nf+f];
  double tx=0.0,ty=0.0,ub=0.0;
  if(mode==6 && q[c]>dry){
   const double un=(q[nc+c]*nx+q[2*nc+c]*ny)/q[c];
   tx=q[nc+c]/q[c]-un*nx;ty=q[2*nc+c]/q[c]-un*ny;
  }
  if(h<=dry||section_depth<=0.0){
   trace[f]=0.0;trace[nf+f]=0.0;trace[2*nf+f]=0.0;continue;
  }
  double chezy=0.0,z0=0.0,drag=0.0;
  const bool valid=br_roughness_state(roughness_mode,h,parameter[c],gravity,
      minimum_work_depth,waqua_white_colebrook,&chezy,&z0,&drag);
  if(!valid){atomicOr(status+b,4);trace[f]=0.0;trace[nf+f]=0.0;
   trace[2*nf+f]=0.0;continue;}
  if(desired>0.0 && total>0.0){
   const double weight=pow(section_depth,1.5)*chezy;
   ub=desired*(weight*length[f])/(total*h*length[f]);
  }
  if(sqrt(ub*ub+tx*tx+ty*ty)>speed_limit)atomicOr(status+b,2);
  trace[f]=h;trace[nf+f]=h*(-ub*nx+tx);trace[2*nf+f]=h*(-ub*ny+ty);
 }
}
