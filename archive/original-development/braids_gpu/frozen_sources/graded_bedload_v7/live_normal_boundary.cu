// Private NORMAL-boundary bridge; original kernels and flux bodies unchanged.
extern "C" __global__ void lnb_gather(const double* forcing,const int* ids,int n,double* Q){
 const int k=blockIdx.x*blockDim.x+threadIdx.x;if(k<n)Q[k]=forcing[ids[k]];
}
extern "C" __global__ void lnb_stage(const double* stage,const int* root_status,
 const int* ids,int n,int nb,double* forcing,double* sediment){
 const int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=n)return;const int b=ids[k];
 // Failure cannot become usable stage. Downstream status rejects the Euler.
 const double z=root_status[k]?0.:stage[k];forcing[nb+b]=z;sediment[nb+b]=z;
 sediment[b]=forcing[b];
}
extern "C" __global__ void lnb_scatter(const int* faces,int count,int nf,
 const double* trace,const int* source_status,double* destination,int* face_status){
 const int p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=count)return;const int f=faces[p];
 destination[f]=trace[f];destination[1LL*nf+f]=trace[1LL*nf+f];destination[2LL*nf+f]=trace[2LL*nf+f];
 face_status[f]|=source_status[f];
}
extern "C" __global__ void lnb_merge(const int* ids,int n,const int* root_status,
 const int* trace_status,int* boundary_status,int* latch){
 const int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=n)return;const int b=ids[k];
 const int flag=trace_status[k]|(root_status[k]?2:0);latch[k]|=flag;boundary_status[b]|=latch[k];
}
extern "C" __global__ void lnb_record(const int* faces,int count,int nf,
 const double* trace,double* saved){
 const int p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=count)return;const int f=faces[p];
 saved[p]=trace[f];saved[1LL*count+p]=trace[1LL*nf+f];saved[2LL*count+p]=trace[2LL*nf+f];
}
