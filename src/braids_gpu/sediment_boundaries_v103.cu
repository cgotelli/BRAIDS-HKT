extern "C" __global__ void braids_q6b_bed_v103(int nc,int mode,int inward,double dt,
    const double* computed,const double* interior,const double* depth,
    const double* prescribed,double* result,int* status) {
    int c=blockDim.x*blockIdx.x+threadIdx.x;
    if(c<nc) braids_q6b_bed_cell_v103(c,nc,mode,inward,dt,computed,interior,
        depth,prescribed,result,status);
}
extern "C" __global__ void braids_q6b_composition_v103(int nc,int ng,int nl,int mode,int coarse,
    const double* rho,const double* interior,const double* values,double* boundary,
    int* status) {
    int c=blockDim.x*blockIdx.x+threadIdx.x;
    if(c<nc) braids_q6b_composition_cell_v103(c,nc,ng,nl,mode,coarse,rho,interior,
        values,boundary,status);
}
extern "C" __global__ void braids_q6b_load_v103(int nc,int ng,double dt,
    const double* discharge,const double* concentration,const double* initial_volume,
    const double* current_volume,const double* area,const int* wet,double* mass,
    double* delta,int* status) {
    int c=blockDim.x*blockIdx.x+threadIdx.x;
    if(c<nc) braids_q6b_load_cell_v103(c,nc,ng,dt,discharge,concentration,
        initial_volume,current_volume,area,wet,mass,delta,status);
}
