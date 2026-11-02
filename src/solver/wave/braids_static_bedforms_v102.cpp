#include <cmath>
using std::isfinite;
#include "braids_static_bedforms_v102.h"
extern "C" void braids_static_bedforms_v102(int nc,int nt,const int* mode,const double* p,
    const double* depth,const double* u,const double* v,const double* chezy,
    const double* d50,const double* d90,double* out,int* status) {
    #pragma omp parallel for schedule(static) num_threads(nt)
    for(int c=0;c<nc;c++) b102_bedform_cell(c,nc,mode,p,depth,u,v,chezy,d50,d90,out,status);
}
extern "C" void braids_static_capacity_v102(int nc,int nt,const int* mode,const double* p,
    const double* depth,const double* u,const double* v,const double* d10,const double* d50,
    const double* d90,const double* rks,const double* skin,const double* mud,const double* concentration,
    const double* salinity,const double* sx,const double* sy,double* out,int* status) {
    #pragma omp parallel for schedule(static) num_threads(nt)
    for(int c=0;c<nc;c++) b102_capacity_cell(c,nc,mode,p,depth,u,v,d10,d50,d90,rks,skin,mud,
        concentration,salinity,sx,sy,out,status);
}
