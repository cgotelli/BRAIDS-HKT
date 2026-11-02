// Concatenated after braids_dependent_layers_v103.h by the resident runtime.
extern "C" __global__ void braids_dependent_targets_v103(int nc,int nl,int mode,int coarse,
    int exchange,int nlag,double active,double alpha,double minimum,double coarse_h,
    double exchange_h,double lag_h,double eu_h,const double* depth,const double* dune,double* targets) {
    int c=blockDim.x*blockIdx.x+threadIdx.x;
    if(c<nc) braids_dependent_targets_cell_v103(c,nc,nl,mode,coarse,exchange,nlag,
        active,alpha,minimum,coarse_h,exchange_h,lag_h,eu_h,depth,dune,targets);
}
extern "C" __global__ void braids_dependent_sort_v103(int nc,int ng,int nl,int mode,int topmode,
    int pormode,double dt,double rhow,double gravity,double viscosity,double amax,double sinkmax,
    double asfm,double bsfm,double sigma,double sv0,double svsand,double svmud,
    const double* tau,const double* dune_length,
    const double* active_target,const double* rho,const double* rhosol,const double* diameter,
    const double* hidexp,const double* bedload,const double* initial,const double* initial_sv,
    const double* delta,const double* phi,const double* sig,const int* mud,double* mass,double* sv,
    double* mobility,double* transfer,int* status) {
    int c=blockDim.x*blockIdx.x+threadIdx.x;
    if(c<nc) braids_q5b_prepare_sort_cell_v103(c,nc,ng,nl,mode,topmode,pormode,dt,
        rhow,gravity,viscosity,amax,sinkmax,asfm,bsfm,sigma,sv0,svsand,svmud,tau,
        dune_length,active_target,rho,rhosol,diameter,hidexp,bedload,initial,
        initial_sv,delta,phi,sig,mud,mass,sv,mobility,transfer,status);
}
