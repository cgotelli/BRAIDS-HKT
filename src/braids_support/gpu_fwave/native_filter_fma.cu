// Isolated FP64 FMA triangular application, reciprocal pivots; all other operations unchanged. FP64, no FMA contraction or CPU fallback.
// One persistent block traverses dependency levels; independent rows per level
// run concurrently. Gather lists retain the native scatter subtraction order.
extern "C" __global__ void nf_coeff(int nf,const int* owner,const int* other,
 const double* h,const double* length,const double* projection,const double* distance2,
 double dry,double depth_factor,double* weight){
 int f=blockIdx.x*blockDim.x+threadIdx.x;if(f>=nf)return;weight[f]=0.;
 int c=owner[f],n=other[f];if(n<0||fmin(h[c],h[n])<=dry||depth_factor==0.)return;
 double a=(depth_factor*h[c])*(depth_factor*h[c]),b=(depth_factor*h[n])*(depth_factor*h[n]);
 weight[f]=2.*a*b/(a+b)*length[f]*projection[f]/distance2[f];
}

__device__ void nf_multiply(int nc,const int* row,const int* edge,const int* other,
 const int* owner,const double* mass,const double* weight,const double* x,double* y){
 for(int c=threadIdx.x;c<nc;c+=blockDim.x){double value=mass[c]*x[c];
  for(int k=row[c];k<row[c+1];++k){int f=edge[k],n=other[k];if(weight[f]==0.)continue;
   if(owner[f]==c){double flux=weight[f]*(x[c]-x[n]);value+=flux;}
   else{double flux=weight[f]*(x[n]-x[c]);value-=flux;}}
  y[c]=value;}
 __syncthreads();
}

__device__ double nf_dot(int nc,const double* x,const double* y,int arithmetic,double* scratch){
 // Variant0 follows serial native SUM element order. Variant1 uses a fixed
 // parallel tree and is a separately reported arithmetic experiment.
 if(arithmetic==0){if(threadIdx.x==0){double v=0.;for(int c=0;c<nc;++c)v+=x[c]*y[c];scratch[0]=v;}}
 else{double v=0.;for(int c=threadIdx.x;c<nc;c+=blockDim.x)v+=x[c]*y[c];scratch[threadIdx.x]=v;
  __syncthreads();for(int stride=blockDim.x/2;stride;stride/=2){if(threadIdx.x<stride)scratch[threadIdx.x]+=scratch[threadIdx.x+stride];__syncthreads();}}
 __syncthreads();double result=scratch[0];__syncthreads();return result;
}

__device__ void nf_precondition(int nc,int nlevel,const int* level_row,const int* level_cells,
 const int* row,const int* column,const int* upper_row,const int* upper_slot,const int* upper_owner,
 const double* factor,const double* pivot,const double* diagonal,const double* r,double* z,int use_ic){
 if(!use_ic){for(int c=threadIdx.x;c<nc;c+=blockDim.x)z[c]=r[c]/diagonal[c];__syncthreads();return;}
 for(int level=0;level<nlevel;++level){
  for(int j=level_row[level]+threadIdx.x;j<level_row[level+1];j+=blockDim.x){int c=level_cells[j];double value=r[c];
   for(int k=row[c];k<row[c+1];++k)value=__fma_rn(-factor[k],z[column[k]],value);
   z[c]=value*pivot[c];}
  __syncthreads();}
 for(int level=nlevel-1;level>=0;--level){
  for(int j=level_row[level]+threadIdx.x;j<level_row[level+1];j+=blockDim.x){int c=level_cells[j];double value=z[c];
   // Native backward scatter visits the higher global cell IDs descending.
   for(int k=upper_row[c+1]-1;k>=upper_row[c];--k)value=__fma_rn(-factor[upper_slot[k]],z[upper_owner[k]],value);
   z[c]=value*pivot[c];}
  __syncthreads();}
}

extern "C" __global__ void nf_solve(int nc,int nz,int nlevel,
 const int* adj_row,const int* adj_edge,const int* adj_other,const int* owner,
 const int* row,const int* column,const int* upper_row,const int* upper_slot,const int* upper_owner,
 const int* level_row,const int* level_cells,const int* slot_face_row,const int* slot_faces,
 const int* diag_row,const int* diag_left,const int* diag_right,
 const int* factor_row,const int* factor_left,const int* factor_right,
 const double* mass,const double* weight,const double* raw,double rtol,double atol,int maxiter,int arithmetic,
 double* diagonal,double* pivot_global,double* factor,double* filtered,double* rhs,double* r,double* z_global,double* p,double* ap,
 double* metrics,int* info,int workspace_mode){
 // Only storage changes: all scalar arithmetic and dependency levels are retained.
 extern __shared__ double workspace[];
 double* z=workspace;
 double* pivot=workspace_mode==2?workspace+nc:pivot_global;
 __shared__ double scratch[256],rho,beta,alpha,tolerance,residual,true_residual;
 __shared__ int use_ic,done;
 if(threadIdx.x==0){use_ic=1;done=0;info[0]=0;info[1]=0;info[2]=1;info[3]=0;}
 for(int c=threadIdx.x;c<nc;c+=blockDim.x){double value=mass[c];
  for(int k=adj_row[c];k<adj_row[c+1];++k)value+=weight[adj_edge[k]];
  diagonal[c]=value;rhs[c]=mass[c]*raw[c];filtered[c]=raw[c];}
 for(int k=threadIdx.x;k<nz;k+=blockDim.x){double value=0.;
  for(int j=slot_face_row[k];j<slot_face_row[k+1];++j)value-=weight[slot_faces[j]];factor[k]=value;}
 __syncthreads();
 for(int level=0;level<nlevel;++level){
  for(int j=level_row[level]+threadIdx.x;j<level_row[level+1];j+=blockDim.x){int c=level_cells[j];double value=diagonal[c];
   for(int k=diag_row[c];k<diag_row[c+1];++k)value-=factor[diag_left[k]]*factor[diag_right[k]];
   if(!(value>0.)||!isfinite(value)){atomicExch(&use_ic,0);pivot[c]=1.;}else pivot[c]=sqrt(value);
   for(int up=upper_row[c];up<upper_row[c+1];++up){int k=upper_slot[up];double entry=factor[k];
    for(int a=factor_row[k];a<factor_row[k+1];++a)entry-=factor[factor_left[a]]*factor[factor_right[a]];
    factor[k]=entry/pivot[c];}}
  __syncthreads();if(!use_ic)break;
 }
 if(threadIdx.x==0)info[2]=use_ic;
 // Isolated arithmetic variant: reciprocal once, multiply in each triangular sweep.
 if(use_ic){for(int c=threadIdx.x;c<nc;c+=blockDim.x)pivot[c]=1./pivot[c];}
 __syncthreads();
 double scale=sqrt(nf_dot(nc,rhs,rhs,arithmetic,scratch));
 double area_norm=sqrt(nf_dot(nc,mass,mass,arithmetic,scratch));
 if(threadIdx.x==0)tolerance=rtol*scale+atol*area_norm;
 nf_multiply(nc,adj_row,adj_edge,adj_other,owner,mass,weight,filtered,ap);
 for(int c=threadIdx.x;c<nc;c+=blockDim.x)r[c]=rhs[c]-ap[c];__syncthreads();
 double norm=sqrt(nf_dot(nc,r,r,arithmetic,scratch));
 if(threadIdx.x==0){residual=norm;done=residual<=tolerance;}
 __syncthreads();
 if(!done){
  nf_precondition(nc,nlevel,level_row,level_cells,row,column,upper_row,upper_slot,upper_owner,factor,pivot,diagonal,r,z,use_ic);
  for(int c=threadIdx.x;c<nc;c+=blockDim.x)p[c]=z[c];__syncthreads();
  double rz=nf_dot(nc,r,z,arithmetic,scratch);if(threadIdx.x==0)rho=rz;__syncthreads();
  for(int iteration=1;iteration<=maxiter;++iteration){
   nf_multiply(nc,adj_row,adj_edge,adj_other,owner,mass,weight,p,ap);
   double denominator=nf_dot(nc,p,ap,arithmetic,scratch);
   if(threadIdx.x==0){if(!(denominator>0.)||!isfinite(denominator)){info[0]=1;done=2;}else alpha=rho/denominator;}
   __syncthreads();if(done==2)break;
   for(int c=threadIdx.x;c<nc;c+=blockDim.x){filtered[c]+=alpha*p[c];r[c]-=alpha*ap[c];}__syncthreads();
   norm=sqrt(nf_dot(nc,r,r,arithmetic,scratch));
   if(threadIdx.x==0){info[1]=iteration;residual=norm;done=residual<=tolerance;}
   __syncthreads();if(done)break;
   nf_precondition(nc,nlevel,level_row,level_cells,row,column,upper_row,upper_slot,upper_owner,factor,pivot,diagonal,r,z,use_ic);
   rz=nf_dot(nc,r,z,arithmetic,scratch);if(threadIdx.x==0){beta=rz/rho;rho=rz;}__syncthreads();
   for(int c=threadIdx.x;c<nc;c+=blockDim.x)p[c]=z[c]+beta*p[c];__syncthreads();
  }
 }
 // Retain the native final true-residual <=10*tolerance acceptance gate.
 nf_multiply(nc,adj_row,adj_edge,adj_other,owner,mass,weight,filtered,ap);
 for(int c=threadIdx.x;c<nc;c+=blockDim.x){ap[c]=rhs[c]-ap[c];if(!isfinite(filtered[c]))atomicExch(info,2);}__syncthreads();
 double truth=sqrt(nf_dot(nc,ap,ap,arithmetic,scratch));
 if(threadIdx.x==0){true_residual=truth;metrics[0]=residual;metrics[1]=true_residual;metrics[2]=tolerance;
  metrics[3]=scale;info[3]=(info[0]==0&&residual<=tolerance&&true_residual<=10.*tolerance);
  if(!info[3]&&info[0]==0)info[0]=3;}
}
