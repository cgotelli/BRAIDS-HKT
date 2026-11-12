// Research: the same coarse V-cycle in one cooperating GPU thread block.
// FP64 and per-cell arithmetic order are retained. All original arrays stay on
// device. Fine grids still use the original multi-block parallel kernels.
__device__ const int* cb_i(const unsigned long long* p,int l,int k){return (const int*)p[14*l+k];}
__device__ double* cb_d(const unsigned long long* p,int l,int k){return (double*)p[14*l+k];}
extern "C" __global__ void cb_cycle(int first,int levels,const int* sizes,
 const unsigned long long* p,const double* factor){
 int tid=threadIdx.x;
 for(int l=first;l<levels-1;++l){
  int nc=sizes[l];const int* row=cb_i(p,l,0);const int* edge=cb_i(p,l,1);
  const int* other=cb_i(p,l,2);const int* owner=cb_i(p,l,3);
  const double* mass=cb_d(p,l,4);const double* w=cb_d(p,l,5);const double* diagonal=cb_d(p,l,6);
  const double* b=cb_d(p,l,7);double* x=cb_d(p,l,8);double* work=cb_d(p,l,9);double* res=cb_d(p,l,10);
  for(int c=tid;c<nc;c+=blockDim.x)work[c]=(2./3.)*b[c]/diagonal[c];
  __syncthreads();
  for(int c=tid;c<nc;c+=blockDim.x){double a=b[c];
   for(int k=row[c];k<row[c+1];++k)a+=w[edge[k]]*work[other[k]];
   x[c]=work[c]/3.+(2./3.)*a/diagonal[c];}
  __syncthreads();
  for(int c=tid;c<nc;c+=blockDim.x){double a=mass[c]*x[c];
   for(int k=row[c];k<row[c+1];++k){int f=edge[k],n=other[k];if(w[f]==0.)continue;
    if(owner[f]==c){double flux=w[f]*(x[c]-x[n]);a+=flux;}
    else{double flux=w[f]*(x[n]-x[c]);a-=flux;}}res[c]=a;}
  __syncthreads();
  const int* child_row=cb_i(p,l,11);const int* child=cb_i(p,l,12);double* coarse=cb_d(p,l+1,7);
  for(int c=tid;c<sizes[l+1];c+=blockDim.x){double a=0.;
   for(int k=child_row[c];k<child_row[c+1];++k){int i=child[k];a=a+b[i]-res[i];}coarse[c]=a;}
  __syncthreads();
 }
 // Exactly the original 32-lane coarse triangular arithmetic.
 int last=levels-1,nc=sizes[last];const double* b=cb_d(p,last,7);double* x=cb_d(p,last,8);
 if(tid<32){
  for(int c=0;c<nc;++c){double value=tid<c?factor[c*nc+tid]*x[tid]:0.;
   for(int offset=16;offset>0;offset/=2)value+=__shfl_down_sync(0xffffffff,value,offset);
   if(tid==0)x[c]=(b[c]-value)/factor[c*nc+c];__syncwarp();}
  for(int c=nc-1;c>=0;--c){double value=tid>c&&tid<nc?factor[tid*nc+c]*x[tid]:0.;
   for(int offset=16;offset>0;offset/=2)value+=__shfl_down_sync(0xffffffff,value,offset);
   if(tid==0)x[c]=(x[c]-value)/factor[c*nc+c];__syncwarp();}
 }
 __syncthreads();
 for(int l=levels-2;l>=first;--l){
  nc=sizes[l];const int* row=cb_i(p,l,0);const int* edge=cb_i(p,l,1);const int* other=cb_i(p,l,2);
  const int* parent=cb_i(p,l,13);const double* coarse=cb_d(p,l+1,8);
  const double* w=cb_d(p,l,5);const double* diagonal=cb_d(p,l,6);b=cb_d(p,l,7);
  x=cb_d(p,l,8);double* work=cb_d(p,l,9);
  for(int c=tid;c<nc;c+=blockDim.x)x[c]+=coarse[parent[c]];
  __syncthreads();
  for(int c=tid;c<nc;c+=blockDim.x){double a=b[c];
   for(int k=row[c];k<row[c+1];++k)a+=w[edge[k]]*x[other[k]];
   work[c]=x[c]/3.+(2./3.)*a/diagonal[c];}
  __syncthreads();
  for(int c=tid;c<nc;c+=blockDim.x){double a=b[c];
   for(int k=row[c];k<row[c+1];++k)a+=w[edge[k]]*work[other[k]];
   x[c]=work[c]/3.+(2./3.)*a/diagonal[c];}
  __syncthreads();
 }
}
