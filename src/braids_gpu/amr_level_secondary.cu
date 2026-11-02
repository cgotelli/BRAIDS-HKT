// Level-local publication/masking around the unchanged secondary operators.
#define BRAIDS_AMR_LEVEL_SECONDARY_ABI 2
extern "C" __global__ void als_validate(int nc,int model,const double* q,const double* mass,int* status){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
 status[c]=(!isfinite(q[c])||q[c]<0.||!isfinite(q[nc+c])||!isfinite(q[2*nc+c])||
            !isfinite(mass[c])||(model!=2&&mass[c]!=0.))?1:0;
}
extern "C" __global__ void als_active(int nc,const int* cells,const int* active,int* selected){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;selected[c]=cells[c]&&active[c];
}
extern "C" __global__ void als_mask_flux(int nf,const int* incident,const double* input,double* masked){
 int f=blockIdx.x*blockDim.x+threadIdx.x;if(f>=nf)return;masked[f]=incident[f]?input[f]:0.;
}
extern "C" __global__ void als_mask_low(int nf,const int* incident,double* low,double* anti,double* boundary,int* status){
 int f=blockIdx.x*blockDim.x+threadIdx.x;if(f>=nf||incident[f])return;
 low[f]=0.;anti[f]=0.;boundary[f]=0.;status[f]=0;
}
extern "C" __global__ void als_bound_depth(int nc,const int* cells,const double* q,
 const double* actual_next,const double* area,const double* length,const int* offset,
 const int* faces,const int* signs,const double* water,double dt,double* bound_next,int* status){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
 double h=actual_next[c];
 if(!cells[c]){
  h=q[c];
  // Virtual donor volume for FCT only, original native per-cell CSR order.
  // Ghost physical q is NOT advanced, and no floor/tolerance is introduced.
  for(int k=offset[c];k<offset[c+1];++k){int f=faces[k];
   double z=-dt*signs[k]*length[f]/area[c];h+=z*water[f];}
 }
 bound_next[c]=h;status[c]=isfinite(h)?0:6;
}
extern "C" __global__ void als_source_publish(int nc,const int* cells,const double* original,
 const double* candidate,const double* source,const int* candidate_status,double* next,
 double* work,int* status){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
 next[c]=original[c];work[c]=0.;status[c]=0;
 if(!cells[c])return;
 next[c]=candidate[c];work[c]=source[c];status[c]=candidate_status[c];
 if(!isfinite(next[c])||!isfinite(work[c]))status[c]=1;
}
extern "C" __global__ void als_scoped_status(int n,const int* selected,const int* input,int* output){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)output[i]=selected[i]?input[i]:0;
}
extern "C" __global__ void als_total_status(const double* total,int* status){
 if(blockIdx.x==0&&threadIdx.x==0&&!isfinite(total[0]))status[0]=1;
}
