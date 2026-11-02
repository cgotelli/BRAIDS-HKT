#include "braids_recursive_surface_feedback_v114.h"

extern "C" void braids_surface_hiding_v114(
    int nc, int ng, int threads, int mode, const double* active_volume,
    const double* d50, double parker_alpha, double wu_exponent,
    double* factors, double* mean_diameter, int* status)
{
    #pragma omp parallel for num_threads(threads) schedule(static)
    for(int cell=0;cell<nc;++cell)
        braids_surface_hiding_cell_v114(
            cell,nc,ng,mode,active_volume,d50,parker_alpha,wu_exponent,
            factors,mean_diameter,status);
}
