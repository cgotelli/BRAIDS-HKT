// Transactional coupled SSPRK(4,3) and explicitly selected SSPRK(3,3).
// Native arithmetic, FP64, no fast math.
extern "C" __global__ void coupled_capture(const double* value,const double* base,
 const int* flag,long long n,int nonnegative,double* out,int* status) {
 long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 bool valid=isfinite(value[i])&&(!nonnegative||value[i]>=0.0);
 status[i]=valid?0:1;out[i]=(valid&&flag[0]==0)?value[i]:base[i];
}
extern "C" __global__ void coupled_combine(const double* base,const double* third,
 long long n,double* out) {
 long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
 if(i<n)out[i]=2.0*base[i]/3.0+third[i]/3.0;
}
extern "C" __global__ void coupled_ssp33_combine_second(const double* base,const double* second,
 long long n,double* out) {
 long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
 if(i<n)out[i]=.75*base[i]+.25*second[i];
}
extern "C" __global__ void coupled_ssp33_combine_final(const double* base,const double* third,
 long long n,double* out) {
 long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
 if(i<n)out[i]=base[i]/3.0+2.0*third[i]/3.0;
}
extern "C" __global__ void coupled_ssp33_ledger(const double* incoming,double* accumulated,
 long long n,int stage,int* status) {
 long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
 if(i>=n)return;
 const double flux=incoming[i];
 const double value=stage<2?flux/6.0:2.0*flux/3.0;
 const double sum=accumulated[i]+value;
 status[i]=isfinite(flux)&&isfinite(value)&&isfinite(sum)?0:1;
 if(status[i]==0)accumulated[i]=sum;
}
extern "C" __global__ void coupled_commit(const double* trial,double* accepted,
 long long n,const int* flag) {
 long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
 if(i<n&&flag[0]==0)accepted[i]=trial[i];
}
extern "C" __global__ void coupled_sum_source(const double* first,const double* second,
 double* total,int* flag) {
 if(blockIdx.x||threadIdx.x)return;total[0]=first[0]+second[0];
 if(!isfinite(total[0]))flag[0]=1;
}
