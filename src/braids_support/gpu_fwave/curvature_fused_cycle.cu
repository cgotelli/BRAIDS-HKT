// Isolated implementation experiment: preserve FP64 weighted-Jacobi and
// residual summation order; fuse temporary-producing kernels, not equations.
extern "C" {
__global__ void cf_presmooth(int nc,const int* row,const int* edge,const int* other,
 const double* w,const double* diagonal,const double* b,double* out){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
 double x=(2./3.)*b[c]/diagonal[c],a=b[c];
 for(int k=row[c];k<row[c+1];++k){int n=other[k];
  double xn=(2./3.)*b[n]/diagonal[n];a+=w[edge[k]]*xn;}
 out[c]=x/3.+(2./3.)*a/diagonal[c];
}
__global__ void cf_restrict(int nc,const int* child_row,const int* child,
 const int* row,const int* edge,const int* other,const int* owner,
 const double* mass,const double* w,const double* b,const double* x,double* coarse){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;double sum=0.;
 for(int j=child_row[c];j<child_row[c+1];++j){int i=child[j];double a=mass[i]*x[i];
  for(int k=row[i];k<row[i+1];++k){int f=edge[k],n=other[k];if(w[f]==0.)continue;
   if(owner[f]==i){double flux=w[f]*(x[i]-x[n]);a+=flux;}
   else{double flux=w[f]*(x[n]-x[i]);a-=flux;}}
  sum=sum+b[i]-a;
 }coarse[c]=sum;
}
__global__ void cf_prolong_smooth(int nc,const int* row,const int* edge,const int* other,
 const int* parent,const double* coarse,const double* w,const double* diagonal,
 const double* b,const double* x,double* out){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
 double xc=x[c]+coarse[parent[c]],a=b[c];
 for(int k=row[c];k<row[c+1];++k){int n=other[k];
  double xn=x[n]+coarse[parent[n]];a+=w[edge[k]]*xn;}
 out[c]=xc/3.+(2./3.)*a/diagonal[c];
}
}
