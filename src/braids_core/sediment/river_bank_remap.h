// Integer transaction markers stay integer. They are diagnostics, not physics.
#ifdef __CUDACC__
#define ENTRY extern "C" __global__
#define LOOP(N) for(int c=blockDim.x*blockIdx.x+threadIdx.x;c<N;c+=blockDim.x*gridDim.x)
#else
#include <omp.h>
#define ENTRY extern "C" __declspec(dllexport)
#define LOOP(N) _Pragma("omp parallel for schedule(static)") for(int c=0;c<N;++c)
extern "C" __declspec(dllexport) void mobile_threads(int n){omp_set_dynamic(0);omp_set_num_threads(n);}
#endif
ENTRY void bank_counts(int n,const int*off,const int*ids,const long long*old,long long*out,int*status){
 LOOP(n){long long sum=0;status[c]=0;
  for(int e=off[c];e<off[c+1];++e){long long v=old[ids[e]];
   if(v<0||v>9223372036854775807LL-sum){status[c]=1;break;}sum+=v;}
  out[c]=sum;
 }
}
