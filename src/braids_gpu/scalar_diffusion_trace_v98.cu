// Trace faces enter the existing donor limiter with all ordinary face amounts.
extern "C" __global__ void diffusion_trace_faces_v98(
 int nc,int nf,int nl,int nt,double dry,double dt,const double* depth,const double* mass,
 const double* dic,const double* exchange,const double* distance,const int* active,
 const int* owner,const int* lookup,const double* th,const double* tc,const double* td,
 const int* tw,double* effective,double* amount,int* active_out,int* status){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=nl*nf)return;
 int k=i/nf,f=i%nf,t=lookup[f];if(t<0)return;
 effective[i]=0.;amount[i]=0.;active_out[i]=0;status[i]=0;
 if(!active[f]||!tw[t])return;int c=owner[f];
 double hl=depth[c],hr=th[t],deff=.5*(dic[c]+td[t])/.7;
 effective[i]=deff;if(fmin(hl,hr)<=dry)return;
 double value=dt*exchange[f]*deff*(mass[(long long)k*nc+c]/hl-tc[(long long)k*nt+t])/distance[f];
 amount[i]=value;active_out[i]=1;if(!isfinite(value))status[i]=1;
}
extern "C" __global__ void diffusion_trace_ledger_v98(
 int nf,int nl,const int* lookup,const double* amount,const int* face_status,
 double* ledger,int* status){
 int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=nl)return;
 double total=0.;status[k]=0;
 for(int f=0;f<nf;f++)if(lookup[f]>=0){
  int i=k*nf+f;total-=amount[i];if(face_status[i])status[k]=1;
 }
 ledger[k]=total;if(!isfinite(total))status[k]=1;
}
