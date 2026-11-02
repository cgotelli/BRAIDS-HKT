// Only the density mask is augmented. No physical-boundary or v91 owner.
extern "C" __global__ void density_trace_faces_v98(
 const long long nc,const long long nf,const long long nt,
 const int* owner,const int* lookup,const double* depth,const double* density,
 const double* distance,const double* base_acceleration,const int* base_mask,
 const int* wet,const double* trace_depth,const double* trace_density,
 const int* trace_wet,const double rho_ref,const double gravity,
 double* acceleration,int* mask,int* status) {
 const long long f=blockIdx.x*blockDim.x+threadIdx.x;
 if(f>=nf)return;
 acceleration[f]=base_acceleration[f];mask[f]=base_mask[f];status[f]=0;
 const int t=lookup[f];
 if(t<0)return;
 const int c=owner[f];
 acceleration[f]=0.;mask[f]=0;
 if(!wet[c]||!trace_wet[t])return;
 acceleration[f]=-gravity*(trace_density[t]-density[c])*
  (trace_depth[t]+depth[c])/(4.*rho_ref*distance[f]);
 mask[f]=1;
 if(!isfinite(acceleration[f]))status[f]=1;
}
