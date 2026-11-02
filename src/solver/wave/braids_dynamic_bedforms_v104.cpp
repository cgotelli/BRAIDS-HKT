#include <cmath>
#include <vector>
using std::isfinite;
#include "braids_dynamic_bedforms_v104.h"

extern "C" void braids_dynamic_diagnostics_v104(int nc,int nt,const int* mode,const double* p,
    const double* depth,const double* u,const double* v,const double* bx,const double* by,
    const double* height,const double* equilibrium,const double* length,const double* ripple,
    const double* mega,const double* static_dune,double* out,int* status) {
    #pragma omp parallel for schedule(static) num_threads(nt)
    for(int c=0;c<nc;c++) b104_diagnostic_cell(c,nc,mode,p,depth,u,v,bx,by,height,
        equilibrium,length,ripple,mega,static_dune,out,status);
}

extern "C" void braids_dynamic_roughness_relax_v104(int nc,int nt,const double* p,double dt,
    const double* ripple,const double* mega,const double* dune,const double* target_ripple,
    const double* target_mega,const double* target_dune,double* out,int* status) {
    #pragma omp parallel for schedule(static) num_threads(nt)
    for(int c=0;c<nc;c++) b104_relax_roughness_cell(c,nc,p,dt,ripple,mega,dune,
        target_ripple,target_mega,target_dune,out,status);
}

extern "C" void braids_dynamic_step_v104(int nx,int ny,int nt,int nsub,const int* mode,
    const int* boundary,double dt,double dx,double dy,const double* initial,
    const double* equilibrium,const double* cx,const double* cy,const double* trelax,
    const int* wet,double* result,int* status) {
    const int nc=nx*ny;
    std::vector<double> a(initial,initial+nc), b(nc);
    std::vector<int> local(nc);
    const double ds=dt/nsub;
    for(int step=0;step<nsub;step++) {
        #pragma omp parallel for schedule(static) num_threads(nt)
        for(int c=0;c<nc;c++) b104_update_cell(c,nx,ny,mode,boundary,ds,dx,dy,
            a.data(),equilibrium,cx,cy,trelax,wet,b.data(),local.data());
        for(int c=0;c<nc;c++) if(local[c]) { status[c]=local[c]; return; }
        a.swap(b);
    }
    #pragma omp parallel for schedule(static) num_threads(nt)
    for(int c=0;c<nc;c++) { result[c]=a[c]; status[c]=0; }
}
