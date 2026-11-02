// The shared header is concatenated by the resident runtime before this entry point.
extern "C" __global__ void braids_bed_targets_v101(int nc,int nl,int method,int nlag,
    double active,double alpha,double minimum,double eulerian,double exchange,
    double lagrangian,const double* depth,double* target) {
    int c=blockDim.x*blockIdx.x+threadIdx.x;
    if(c<nc) bed_targets_v101(c,nc,nl,method,nlag,active,alpha,minimum,
        eulerian,exchange,lagrangian,depth,target);
}
extern "C" __global__ void braids_bed_integrate_v101(int nc,int ng,double dt,
    const int* enabled,const double* rate,double* delta) {
    int i=blockDim.x*blockIdx.x+threadIdx.x;
    if(i<nc*ng) delta[i]=enabled[i/nc]?dt*rate[i]:0.;
}
extern "C" __global__ void braids_bed_columns_v101(int nc,int ng,int nl,
    int top,int base,int por,int nlag,int exchange,double dt,const double* initial,const double* initial_sv,
    const double* delta,const double* target,const double* rho,const double* phi,
    const double* sig,const int* mud,double sv0,double svsand,double svmud,
    int nd,const double* zd,const double* kd,double* mass,double* sv,
    double* source,double* work,int* status) {
    int c=blockDim.x*blockIdx.x+threadIdx.x;
    if(c<nc) bed_column_v101(c,nc,ng,nl,top,base,por,nlag,exchange,dt,initial,initial_sv,
        delta,target,rho,phi,sig,mud,sv0,svsand,svmud,nd,zd,kd,mass,sv,source,work,status);
}
extern "C" __global__ void braids_bed_columns_coarse_v101(int nc,int ng,int nl,
    int top,int base,int por,int nlag,int exchange,double dt,const double* initial,const double* initial_sv,
    const double* delta,const double* target,const double* rho,const double* phi,
    const double* sig,const int* mud,double sv0,double svsand,double svmud,
    int nd,const double* zd,const double* kd,double* mass,double* sv,
    double* source,double* work,int* status,const double* mobility) {
    int c=blockDim.x*blockIdx.x+threadIdx.x;
    if(c<nc) bed_column_v101(c,nc,ng,nl,top,base,por,nlag,exchange,dt,initial,initial_sv,
        delta,target,rho,phi,sig,mud,sv0,svsand,svmud,nd,zd,kd,mass,sv,source,work,status,mobility);
}
extern "C" __global__ void braids_bed_limits_v101(int nc,int ng,double mf,
    const double* active,const double* water,const double* depth,
    const double* bedrate,const double* waterrate,const double* rho,
    double maxbed,double maxconc,double entrainment,double* limits) {
    int c=blockDim.x*blockIdx.x+threadIdx.x;
    if(c<nc) bed_limits_v101(c,nc,ng,mf,active,water,depth,bedrate,waterrate,rho,maxbed,maxconc,entrainment,limits);
}
extern "C" __global__ void braids_bed_remap_v101(int nc,int oldnc,int ng,int nl,
    const int* offsets,const int* indices,const double* weights,
    const double* mass,const double* sv,const double* rock,const double* rho,
    double* out,double* outsv,double* outrock) {
    int c=blockDim.x*blockIdx.x+threadIdx.x;
    if(c<nc) bed_remap_v101(c,nc,oldnc,ng,nl,offsets,indices,weights,mass,sv,rock,rho,out,outsv,outrock);
}
extern "C" __global__ void braids_bed_characteristics_v101(int nc,int ng,int np,int nq,
    const double* fractions,const double* medians,const double* means,const double* nodes,
    const double* cdf,const double* quantiles,double* out) {
    int c=blockDim.x*blockIdx.x+threadIdx.x;
    if(c<nc) bed_characteristics_v101(c,nc,ng,np,nq,fractions,medians,means,nodes,cdf,quantiles,out);
}
extern "C" __global__ void braids_bed_fixfac_v101(int nc,int ng,int nl,int mode,double threshold,
    const double* mass,const double* sv,const double* rho,const double* target,double* out) {
    int c=blockDim.x*blockIdx.x+threadIdx.x;
    if(c<nc) bed_fixfac_v101(c,nc,ng,nl,mode,threshold,mass,sv,rho,target,out);
}
extern "C" __global__ void braids_bed_tracer_v101(int nc,int ng,const double* mass,
    const double* rho,const double* diameter,const double* calibration,
    const double* fractions,const double* fixfac,double* outfrac,double* outfix) {
    int c=blockDim.x*blockIdx.x+threadIdx.x;
    if(c<nc) bed_tracer_v101(c,nc,ng,mass,rho,diameter,calibration,fractions,fixfac,outfrac,outfix);
}
