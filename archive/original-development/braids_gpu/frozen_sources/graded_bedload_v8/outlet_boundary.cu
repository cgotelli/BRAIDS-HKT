// Data-only private mode4 bridge. No hydraulic or sediment formula here.
extern "C" __global__ void ob_scatter(const int* faces,int count,int nf,const double* src,
 const int* src_status,double* trace,int* face_status){
 int p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=count)return;int f=faces[p];
 trace[f]=src[f];trace[1LL*nf+f]=src[1LL*nf+f];trace[2LL*nf+f]=src[2LL*nf+f];face_status[f]|=src_status[f];
}
extern "C" __global__ void ob_merge(const int* ids,int n,const int* flags,int* status,int* latch){
 int p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=n)return;latch[p]|=flags[p];status[ids[p]]|=latch[p];
}
extern "C" __global__ void ob_record_trace(const int* faces,int n,int nf,const double* trace,double* saved){
 int p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=n)return;int f=faces[p];
 saved[p]=trace[f];saved[1LL*n+p]=trace[1LL*nf+f];saved[2LL*n+p]=trace[2LL*nf+f];
}
extern "C" __global__ void ob_record_float(const int* faces,int n,const double* src,double* saved){
 int p=blockIdx.x*blockDim.x+threadIdx.x;if(p<n)saved[p]=src[faces[p]];
}
extern "C" __global__ void ob_record_int(const int* faces,int n,const int* src,int* saved){
 int p=blockIdx.x*blockDim.x+threadIdx.x;if(p<n)saved[p]=src[faces[p]];
}
