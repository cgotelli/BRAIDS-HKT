// Isolated level-local orchestration; the augmented f-wave and reconstruction
// kernels are reused unchanged. Nonincident faces are exactly zero BEFORE
// donor limiting. Geometry and per-level masks are immutable resident inputs.
#define BRAIDS_AMR_LEVEL_EULER_ABI 2
extern "C" __global__ void ale_kinds(const int* incident,const int* source,
 int nf,int* kind){int f=blockIdx.x*blockDim.x+threadIdx.x;if(f<nf)kind[f]=incident[f]?source[f]:0;}
// Native coupled_mpm_face retains pure f-wave free release on supercritical
// stage-adjacent faces. This mask affects ONLY the optional SWE-Exner split;
// the hydraulic face kind, flux, pressure and augmented solver stay unchanged.
extern "C" __global__ void ale_coupled_kinds(const int* kind,const int* owner,
 const int* neighbour,const int* open_distance,const double* local,int nf,
 double gravity,int* output){
 int f=blockIdx.x*blockDim.x+threadIdx.x;if(f>=nf)return;
 int value=kind[f],c=owner[f],n=neighbour[f];
 if(value==1&&n>=0&&min(open_distance[c],open_distance[n])<=2){
  double hl=local[f],hr=local[4*nf+f];
  if(fabs(local[nf+f])>=hl*sqrt(gravity*hl)&&
     fabs(local[5*nf+f])>=hr*sqrt(gravity*hr))value=0;
 }
 output[f]=value;
}
extern "C" __global__ void ale_active(const int* level,const int* physical,
 int nc,int* active){int c=blockIdx.x*blockDim.x+threadIdx.x;if(c<nc)active[c]=level[c]&&physical[c];}
extern "C" __global__ void ale_mask_flux(const int* incident,const double* bedraw,
 int nf,int ng,double* fl,double* fr,double* speed,double* bed){
 int f=blockIdx.x*blockDim.x+threadIdx.x;if(f>=nf)return;
 if(!incident[f]){speed[f]=0.;for(int j=0;j<3;++j){fl[j*nf+f]=0.;fr[j*nf+f]=0.;}}
 for(int g=0;g<ng;++g)bed[g*nf+f]=incident[f]?bedraw[g*nf+f]:0.;
}
extern "C" __global__ void ale_scoped_status(const int* mask,const int* status,
 int n,int components,int* output){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n*components)output[i]=mask[i%n]?status[i]:0;
}
extern "C" __global__ void ale_copy_level(const int* selected,const double* source,
 long long size,int nc,double* destination){
 long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
 if(i<size&&selected[i%nc])destination[i]=source[i];
}
