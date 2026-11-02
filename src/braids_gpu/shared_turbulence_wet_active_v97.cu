// Additive v91 wet-active momentum kernels; no state/clock/cap ownership.
extern "C" __global__ void v97_momentum_face_flux(
 long long nc,long long nf,const double* q,const double* vic,const int* owner,
 const int* neighbour,const double* length,const double* distance,const int* wet,
 const int* active,double* flux,int* status)
{
 const long long f=(long long)blockDim.x*blockIdx.x+threadIdx.x;
 if(f>=nf) return;
 flux[f]=flux[nf+f]=0.; status[f]=0;
 if(!active[f]) return;
 const int l=owner[f],r=neighbour[f];
 if(r<0||!wet[l]||!wet[r]||q[l]<=0.||q[r]<=0.) { status[f]=1; return; }
 const double coefficient=.5*(q[l]+q[r])*.5*(vic[l]+vic[r])*length[f]/distance[f];
 for(int k=1;k<3;++k) {
  const double value=coefficient*(q[k*nc+r]/q[r]-q[k*nc+l]/q[l]);
  if(!isfinite(value)) status[f]=2;
  else flux[(k-1)*nf+f]=value;
 }
}

extern "C" __global__ void v97_momentum_gather(
 long long nc,long long nf,const double* area,const int* offset,const int* faces,
 const int* signs,const int* wet,const double* flux,double* rate,int* status)
{
 const long long c=(long long)blockDim.x*blockIdx.x+threadIdx.x;
 if(c>=nc) return;
 rate[c]=rate[nc+c]=rate[2*nc+c]=0.; status[c]=0;
 if(!wet[c]) return;
 for(int k=1;k<3;++k) {
  double sum=0.;
  for(int slot=offset[c];slot<offset[c+1];++slot) sum+=signs[slot]*flux[(k-1)*nf+faces[slot]];
  const double value=sum/area[c];
  if(!isfinite(value)) status[c]=1;
  else rate[k*nc+c]=value;
 }
}
