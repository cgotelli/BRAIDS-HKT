// Pure actual-CUDA counterpart of the native v97 wet-active projection.
extern "C" __global__ void density_active_projection_v97(
 long long nc,long long nf,const double* depth,const double* accel,
 const double* normal,const double* length,const int* offset,const int* faces,
 const int* signs,const int* wet,const int* active,double rank_tol,double residual_tol,
 double* cell_accel,double* rate,int* ranks,double* residual,int* status)
{
 const long long c=(long long)blockDim.x*blockIdx.x+threadIdx.x;
 if(c>=nc) return;
 cell_accel[c]=cell_accel[nc+c]=rate[c]=rate[nc+c]=residual[c]=0.;
 ranks[c]=status[c]=0;
 if(!wet[c]) return;
 double a=0.,b=0.,d=0.,bx=0.,by=0.; int nactive=0; bool allzero=true;
 for(int slot=offset[c];slot<offset[c+1];++slot) {
  const int f=faces[slot]; if(!active[f]) continue;
  ++nactive;
  const double nx=signs[slot]*normal[f],ny=signs[slot]*normal[nf+f];
  const double an=signs[slot]*accel[f],w=length[f];
  a+=w*nx*nx; b+=w*nx*ny; d+=w*ny*ny;
  bx+=w*nx*an; by+=w*ny*an;
  if(an!=0.) allzero=false;
 }
 if(!nactive) return;
 if(allzero) { ranks[c]=-1; return; } // Exact zero; no numerical-rank rejection.
 const double det=a*d-b*b,trace=a+d,gap=hypot(a-d,2.*b),hi=.5*(trace+gap);
 if(!isfinite(hi)||hi<=0.||!isfinite(det)||!isfinite(bx)||!isfinite(by)) {
  status[c]=1; return;
 }
 const double lo=det/hi,ratio=lo/hi;
 if(ratio< -rank_tol||(ratio>.5*rank_tol&&ratio<2.*rank_tol)) {
  status[c]=2; return;
 }
 double ax,ay;
 if(ratio>=2.*rank_tol) {
  ranks[c]=2;
  ax=(d*bx-b*by)/det; ay=(a*by-b*bx)/det;
 } else {
  ranks[c]=1;
  ax=((a-lo)*bx+b*by)/((hi-lo)*hi);
  ay=(b*bx+(d-lo)*by)/((hi-lo)*hi);
 }
 const double rx=a*ax+b*ay-bx,ry=b*ax+d*ay-by;
 const double scale=fmax(2.2250738585072014e-308,
  fmax(fabs(bx)+fabs(a*ax)+fabs(b*ay),fabs(by)+fabs(b*ax)+fabs(d*ay)));
 residual[c]=fmax(fabs(rx),fabs(ry))/scale;
 if(!isfinite(ax)||!isfinite(ay)||!isfinite(residual[c])||residual[c]>residual_tol) {
  status[c]=3; return;
 }
 cell_accel[c]=ax; cell_accel[nc+c]=ay;
 rate[c]=depth[c]*ax; rate[nc+c]=depth[c]*ay;
 if(!isfinite(rate[c])||!isfinite(rate[nc+c])) status[c]=4;
}
