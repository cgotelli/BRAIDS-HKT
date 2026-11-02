// Pure v91 coarse/fine viscosity contribution; caps are accepted inputs.
extern "C" __global__ void momentum_trace_faces_v98(
    long long nc,long long nf,long long nt,const int* owner,const int* lookup,
    const double* q,const double* vic,const double* length,const double* distance,
    const int* wet,const double* tq,const double* tv,const int* tw,
    double* flux,int* status){
  long long f=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(f>=nf)return;
  flux[f]=0.;flux[nf+f]=0.;status[f]=0;
  int t=lookup[f];if(t<0)return;int c=owner[f];if(!wet[c]||!tw[t])return;
  double coefficient=.5*(q[c]+tq[t])*.5*(vic[c]+tv[t])*length[f]/distance[f];
  for(int k=1;k<=2;k++){
    double value=coefficient*(tq[(long long)k*nt+t]/tq[t]-q[(long long)k*nc+c]/q[c]);
    flux[(long long)(k-1)*nf+f]=value;if(!isfinite(value))status[f]=1;
  }
}
extern "C" __global__ void momentum_trace_cells_v98(
    long long nc,long long nf,const int* offset,const int* faces,const int* signs,
    const int* wet,const double* area,const double* flux,double* rate,int* status){
  long long c=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
  status[c]=0;rate[c]=0.;rate[nc+c]=0.;if(!wet[c])return;
  for(int k=0;k<2;k++){
    double total=0.;for(int s=offset[c];s<offset[c+1];s++)
      total+=signs[s]*flux[(long long)k*nf+faces[s]];
    double value=total/area[c];rate[(long long)k*nc+c]=value;
    if(!isfinite(value))status[c]=1;
  }
}
