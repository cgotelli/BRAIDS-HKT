// Isolated FP64 port of BRAIDS' graph Helmholtz curvature operator and
// symmetric weighted-Jacobi multilevel preconditioner. No physical closure
// or convergence tolerance is changed; reductions may have different order.
extern "C" {
__global__ void sc_coeff(int nf,const int* owner,const int* other,
 const double* h,const double* length,const double* projection,const double* distance2,
 double dry,double factor,double* w){
 int f=blockIdx.x*blockDim.x+threadIdx.x;if(f>=nf)return;
 int c=owner[f],n=other[f];w[f]=0.;if(n<0||fmin(h[c],h[n])<=dry)return;
 double a=(factor*h[c])*(factor*h[c]),b=(factor*h[n])*(factor*h[n]);
 if(factor==0.)return;
 w[f]=2.*a*b/(a+b)*length[f]*projection[f]/distance2[f];
}
__global__ void sc_reduce_edges(int nf,const int* row,const int* child,const double* fine,double* w){
 int f=blockIdx.x*blockDim.x+threadIdx.x;if(f>=nf)return;double a=0.;
 for(int k=row[f];k<row[f+1];++k)a+=fine[child[k]];w[f]=a;
}
__global__ void sc_diagonal(int nc,const int* row,const int* edge,const double* mass,const double* w,double* diagonal){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;double a=mass[c];
 for(int k=row[c];k<row[c+1];++k)a+=w[edge[k]];diagonal[c]=a;
}
__global__ void sc_multiply(int nc,const int* row,const int* edge,const int* other,const int* owner,
 const double* mass,const double* w,const double* x,double* y){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;double a=mass[c]*x[c];
 for(int k=row[c];k<row[c+1];++k){int f=edge[k],n=other[k];if(w[f]==0.)continue;
 if(owner[f]==c){double flux=w[f]*(x[c]-x[n]);a+=flux;}
 else{double flux=w[f]*(x[n]-x[c]);a-=flux;}}y[c]=a;
}
__global__ void sc_smooth(int nc,const int* row,const int* edge,const int* other,
 const double* w,const double* diagonal,const double* b,const double* x,double* out,int zero){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
 if(zero){out[c]=(2./3.)*b[c]/diagonal[c];return;}
 double a=b[c];for(int k=row[c];k<row[c+1];++k)a+=w[edge[k]]*x[other[k]];
 out[c]=x[c]/3.+(2./3.)*a/diagonal[c];
}
__global__ void sc_restrict(int nc,const int* row,const int* child,const double* b,const double* ax,double* coarse){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;double a=0.;
 for(int k=row[c];k<row[c+1];++k){int i=child[k];a=a+b[i]-ax[i];}coarse[c]=a;
}
__global__ void sc_prolong(int nc,const int* parent,const double* coarse,double* x){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c<nc)x[c]+=coarse[parent[c]];
}
// Coarsest grid has at most 32 unknowns. A single thread avoids launch and
// transfer overhead for this bounded triangular calculation, not large grids.
__global__ void sc_factor(int nc,const int* row,const int* edge,const int* other,
 const double* w,const double* diagonal,double* a,int* status){
 if(blockIdx.x||threadIdx.x)return;*status=0;
 for(int i=0;i<nc*nc;++i)a[i]=0.;
 for(int c=0;c<nc;++c){a[c*nc+c]=diagonal[c];
 for(int k=row[c];k<row[c+1];++k)a[c*nc+other[k]]-=w[edge[k]];}
 for(int j=0;j<nc;++j){double sum=0.;for(int k=0;k<j;++k)sum+=a[j*nc+k]*a[j*nc+k];
 double value=a[j*nc+j]-sum;if(!(value>0.)||!isfinite(value)){*status=1;return;}
 a[j*nc+j]=sqrt(value);
 for(int i=j+1;i<nc;++i){sum=0.;for(int k=0;k<j;++k)sum+=a[i*nc+k]*a[j*nc+k];
 a[i*nc+j]=(a[i*nc+j]-sum)/a[j*nc+j];}}
}
__global__ void sc_coarse_solve(int nc,const double* a,const double* b,double* x){
 if(blockIdx.x||threadIdx.x)return;
 for(int c=0;c<nc;++c){double sum=0.;for(int k=0;k<c;++k)sum+=a[c*nc+k]*x[k];x[c]=(b[c]-sum)/a[c*nc+c];}
 for(int c=nc-1;c>=0;--c){double sum=0.;for(int k=c+1;k<nc;++k)sum+=a[k*nc+c]*x[k];x[c]=(x[c]-sum)/a[c*nc+c];}
}
__global__ void sc_factor_parallel(int nc,const int* row,const int* edge,const int* other,
 const double* w,const double* diagonal,double* a,int* status){
 int i=threadIdx.x;if(i==0)*status=0;
 if(i<nc){for(int j=0;j<nc;++j)a[i*nc+j]=0.;a[i*nc+i]=diagonal[i];
  for(int k=row[i];k<row[i+1];++k)a[i*nc+other[k]]-=w[edge[k]];}
 __syncthreads();
 for(int j=0;j<nc;++j){
  if(i==j){double sum=0.;for(int k=0;k<j;++k)sum+=a[j*nc+k]*a[j*nc+k];
   double value=a[j*nc+j]-sum;if(!(value>0.)||!isfinite(value))*status=1;else a[j*nc+j]=sqrt(value);}
  __syncthreads();if(*status)return;
  if(i>j&&i<nc){double sum=0.;for(int k=0;k<j;++k)sum+=a[i*nc+k]*a[j*nc+k];
   a[i*nc+j]=(a[i*nc+j]-sum)/a[j*nc+j];}__syncthreads();
 }
}
__global__ void sc_coarse_solve_parallel(int nc,const double* a,const double* b,double* x){
 int lane=threadIdx.x;
 for(int c=0;c<nc;++c){double value=lane<c?a[c*nc+lane]*x[lane]:0.;
  for(int offset=16;offset>0;offset/=2)value+=__shfl_down_sync(0xffffffff,value,offset);
  if(lane==0)x[c]=(b[c]-value)/a[c*nc+c];__syncwarp();}
 for(int c=nc-1;c>=0;--c){double value=lane>c&&lane<nc?a[lane*nc+c]*x[lane]:0.;
  for(int offset=16;offset>0;offset/=2)value+=__shfl_down_sync(0xffffffff,value,offset);
  if(lane==0)x[c]=(x[c]-value)/a[c*nc+c];__syncwarp();}
}
__global__ void sc_pcg_update(int nc,double* x,double* r,const double* p,const double* ap,
 const double* rho,const double* denominator,int* status,const int* done){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc||*status||*done)return;
 if(!(*denominator>0.)||!isfinite(*denominator)){atomicExch(status,1);return;}
 double alpha=*rho/ *denominator;x[c]+=alpha*p[c];r[c]-=alpha*ap[c];
}
__global__ void sc_pcg_direction(int nc,double* p,const double* z,const double* rho,const double* next,int* status,const int* done){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc||*status||*done)return;p[c]=z[c]+(*next/ *rho)*p[c];
}
__global__ void sc_convergence(const double* norm2,const double* tolerance,int* done,int* count,const int* status){
 if(blockIdx.x||threadIdx.x||*done)return;++*count;
 if(*status||!isfinite(*norm2)){*done=2;return;}
 if(sqrt(*norm2)<=*tolerance)*done=1;
}
}
