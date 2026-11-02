// Resident FP64 CUDA transcription of BRAIDS/Delft3D hiding/exposure.
//
// The arithmetic order follows braids_hiding_exposure_module.f90, which in
// turn follows locked Delft3D4 comphidexp.f90 and compdiam.f90.  Input/output
// matrices are fraction-major: value(grain,state) is value[grain*nstate+state].

enum HidingExposureStatus {
    HE_OK = 0,
    HE_INVALID_INPUT = 1,
    HE_DEPLETED = 2,
    HE_SINGULAR_FORMULA = 3,
    HE_INVALID_OUTPUT = 4
};

__device__ __forceinline__ void he_neutralize(
    const long long state, const long long nstate, const int nclass,
    double *factors, double *mean_diameter)
{
    for (int grain=0; grain<nclass; ++grain)
        factors[(long long)grain*nstate+state]=1.0;
    mean_diameter[state]=0.0;
}

extern "C" __global__ void hiding_exposure_factors(
    const long long nstate,
    const int nclass,
    const int mode,
    const double *active_volume,
    const double *d50,
    const double parker_alpha,
    const double wu_exponent,
    double *factors,
    double *mean_diameter,
    int *status)
{
    const long long state=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(state>=nstate) return;
    he_neutralize(state,nstate,nclass,factors,mean_diameter);

    if(nclass<1 || !(mode==0 || mode==2 || mode==3 || mode==4 || mode==5) ||
       !isfinite(parker_alpha) || !isfinite(wu_exponent)) {
        status[state]=HE_INVALID_INPUT;
        return;
    }

    // Match the native SUM order and validate the complete mixture before
    // exposing any constitutive result.
    double total_volume=0.0;
    for(int grain=0; grain<nclass; ++grain) {
        const double volume=active_volume[(long long)grain*nstate+state];
        if(!isfinite(volume) || volume<0.0 ||
           !isfinite(d50[grain]) || d50[grain]<=0.0) {
            status[state]=HE_INVALID_INPUT;
            return;
        }
        total_volume=total_volume+volume;
    }
    if(!isfinite(total_volume)) {
        status[state]=HE_INVALID_INPUT;
        return;
    }
    if(total_volume<=0.0) {
        status[state]=HE_DEPLETED;
        return;
    }

    // Native compdiam first materializes fraction=volume/total and then uses
    // DOT_PRODUCT(fraction,d50).  Preserve those operations and their order.
    double mean=0.0;
    for(int grain=0; grain<nclass; ++grain) {
        const double fraction=
            active_volume[(long long)grain*nstate+state]/total_volume;
        mean=mean+fraction*d50[grain];
    }
    if(!isfinite(mean) || mean<0.0) {
        status[state]=HE_INVALID_OUTPUT;
        return;
    }
    const double local_mean=fmax(mean,1.0e-4);

    // Delft's literal is high precision.  The two Ashida literals below are
    // default-real in the source and are intentionally promoted from float.
    const double log10_19=1.27875360095283;
    const double ashida_limit=(double)0.38889f;
    const double ashida_scale=(double)0.8429f;

    for(int grain=0; grain<nclass; ++grain) {
        const double diameter_ratio=d50[grain]/local_mean;
        double factor=1.0;
        if(mode==2 || mode==3) {
            if(mode==3 && diameter_ratio<ashida_limit) {
                factor=ashida_scale/diameter_ratio;
            } else {
                const double denominator=log10_19+log10(diameter_ratio);
                if(denominator==0.0) {
                    he_neutralize(state,nstate,nclass,factors,mean_diameter);
                    status[state]=HE_SINGULAR_FORMULA;
                    return;
                }
                const double quotient=log10_19/denominator;
                factor=quotient*quotient;
            }
        } else if(mode==4) {
            factor=pow(diameter_ratio,-parker_alpha);
        } else if(mode==5) {
            double phi=0.0;
            for(int other=0; other<nclass; ++other) {
                const double fraction=
                    active_volume[(long long)other*nstate+state]/total_volume;
                phi=phi+fraction*d50[other]/
                    (d50[grain]+d50[other]);
            }
            const double pei=1.0-phi;
            if(phi<=0.0 || pei<=0.0) {
                he_neutralize(state,nstate,nclass,factors,mean_diameter);
                status[state]=HE_SINGULAR_FORMULA;
                return;
            }
            factor=pow(pei/phi,wu_exponent);
        }
        if(!isfinite(factor) || factor<=0.0) {
            he_neutralize(state,nstate,nclass,factors,mean_diameter);
            status[state]=HE_INVALID_OUTPUT;
            return;
        }
        factors[(long long)grain*nstate+state]=factor;
    }
    mean_diameter[state]=mean;
    status[state]=HE_OK;
}
