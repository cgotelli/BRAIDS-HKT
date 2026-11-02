// Q4/v83 locked-FLOW FactCr threshold formation.
//
// One CUDA thread owns one (fraction, cell) point.  The kernel implements
// initsedtra/erosed taucr0 formation only; the frozen v66 current-only
// Van Rijn producer consumes the resulting effective threshold.

extern "C" __global__ void factcr_threshold_v83(
    const long long ncell, const int nclass, const double factcr,
    const double *d50, const double *solid_density,
    const double *water_density, const double *gravity,
    const double *kinematic_viscosity,
    double *dstar, double *critical_shields,
    double *unfactored_taucr, double *effective_taucr, int *status)
{
    const long long linear=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    const long long count=(long long)nclass*ncell;
    if(linear>=count) return;
    const int grain=(int)(linear/ncell);
    const long long cell=linear-(long long)grain*ncell;
    const double dg=d50[grain];
    const double rhos=solid_density[grain];
    const double rho=water_density[cell];
    const double grav=gravity[cell];
    const double nu=kinematic_viscosity[cell];
    bool valid=isfinite(factcr) && factcr>0.0 && isfinite(dg) && dg>0.0 &&
        isfinite(rhos) && isfinite(rho) && rhos>rho && rho>0.0 &&
        isfinite(grav) && grav>0.0 && isfinite(nu) && nu>0.0;
    if(!valid) {
        dstar[linear]=0.0; critical_shields[linear]=0.0;
        unfactored_taucr[linear]=0.0; effective_taucr[linear]=0.0;
        status[linear]=1; return;
    }
    const double relative=(rhos-rho)/rho;
    const double dst=dg*pow(relative*grav/(nu*nu),0.3333);
    double theta;
    if(dst<=4.0) theta=0.24/dst;
    else if(dst<=10.0) theta=0.14/pow(dst,0.64);
    else if(dst<=20.0) theta=0.04/pow(dst,0.10);
    else if(dst<=150.0) theta=0.013*pow(dst,0.29);
    else theta=0.055;
    const double base=(rhos-rho)*grav*dg*theta;
    const double effective=factcr*base;
    valid=isfinite(dst) && dst>0.0 && isfinite(theta) && theta>0.0 &&
        isfinite(base) && base>0.0 && isfinite(effective) && effective>0.0;
    dstar[linear]=dst;
    critical_shields[linear]=theta;
    unfactored_taucr[linear]=base;
    effective_taucr[linear]=effective;
    status[linear]=valid ? 0 : 1;
}
