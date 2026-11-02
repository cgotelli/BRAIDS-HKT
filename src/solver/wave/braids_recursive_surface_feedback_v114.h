#pragma once
#include <cmath>

inline void braids_surface_hiding_cell_v114(
    int cell, int nc, int ng, int mode, const double* active_volume,
    const double* d50, double parker_alpha, double wu_exponent,
    double* factors, double* mean_diameter, int* status)
{
    const double log10_19=1.27875360095283;
    const double ashida_limit=(double)(float)0.38889;
    const double ashida_scale=(double)(float)0.8429;
    double total=0.0, mean=0.0;
    status[cell]=0;
    for(int grain=0;grain<ng;++grain) {
        factors[grain*nc+cell]=1.0;
        const double volume=active_volume[grain*nc+cell];
        if(!std::isfinite(volume) || volume<0.0 ||
           !std::isfinite(d50[grain]) || d50[grain]<=0.0) {
            status[cell]=1; return;
        }
        total+=volume;
    }
    if(!std::isfinite(total) || total<=0.0) {status[cell]=2;return;}
    for(int grain=0;grain<ng;++grain)
        mean+=(active_volume[grain*nc+cell]/total)*d50[grain];
    if(!std::isfinite(mean) || mean<0.0) {status[cell]=4;return;}
    mean_diameter[cell]=mean;
    const double reference=std::fmax(mean,1.0e-4);
    for(int grain=0;grain<ng;++grain) {
        const double ratio=d50[grain]/reference;
        double factor=1.0;
        if(mode==2 || mode==3) {
            if(mode==3 && ratio<ashida_limit) factor=ashida_scale/ratio;
            else {
                const double denominator=log10_19+std::log10(ratio);
                if(denominator==0.0) {status[cell]=3;return;}
                const double quotient=log10_19/denominator;
                factor=quotient*quotient;
            }
        } else if(mode==4) factor=std::pow(ratio,-parker_alpha);
        else if(mode==5) {
            double phi=0.0;
            for(int other=0;other<ng;++other)
                phi+=(active_volume[other*nc+cell]/total)*d50[other]/
                     (d50[grain]+d50[other]);
            const double pei=1.0-phi;
            if(phi<=0.0 || pei<=0.0) {status[cell]=3;return;}
            factor=std::pow(pei/phi,wu_exponent);
        }
        if(!std::isfinite(factor) || factor<=0.0) {status[cell]=4;return;}
        factors[grain*nc+cell]=factor;
    }
}
