// Concatenated after braids_dynamic_bedforms_v104.h by the resident runtime.
extern "C" __global__ void braids_dynamic_diagnostics_v104(int nc,const int* mode,const double* p,
    const double* depth,const double* u,const double* v,const double* bx,const double* by,
    const double* height,const double* equilibrium,const double* length,const double* ripple,
    const double* mega,const double* static_dune,double* out,int* status) {
    int c=blockDim.x*blockIdx.x+threadIdx.x;
    if(c<nc) b104_diagnostic_cell(c,nc,mode,p,depth,u,v,bx,by,height,equilibrium,
        length,ripple,mega,static_dune,out,status);
}
extern "C" __global__ void braids_dynamic_roughness_relax_v104(int nc,const double* p,double dt,
    const double* ripple,const double* mega,const double* dune,const double* target_ripple,
    const double* target_mega,const double* target_dune,double* out,int* status) {
    int c=blockDim.x*blockIdx.x+threadIdx.x;
    if(c<nc) b104_relax_roughness_cell(c,nc,p,dt,ripple,mega,dune,target_ripple,
        target_mega,target_dune,out,status);
}
extern "C" __global__ void braids_dynamic_step_v104(int nx,int ny,const int* mode,
    const int* boundary,double dt,double dx,double dy,const double* initial,
    const double* equilibrium,const double* cx,const double* cy,const double* trelax,
    const int* wet,double* result,int* status) {
    int c=blockDim.x*blockIdx.x+threadIdx.x;
    if(c<nx*ny) b104_update_cell(c,nx,ny,mode,boundary,dt,dx,dy,initial,equilibrium,
        cx,cy,trelax,wet,result,status);
}
