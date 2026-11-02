// Isolated resident-FP64 CUDA transcription of Delft3D4 compbsskin.
//
// This is the current-only Soulsby/Clarke sediment skin-shear closure.  Its
// result is the sediment RP_TAUB value.  It must never replace hydraulic
// Chezy, z0, drag, friction, velocity, or momentum-source state.  In
// particular, the formula-specific von Karman coefficient is 0.40, distinct
// from the 0.41 used by FLOW's hydraulic roughness conversion.
__device__ __forceinline__ void ss_zero(
    const long long i, const long long count, double *output)
{
    #pragma unroll
    for(int field=0;field<4;++field)
        output[(long long)field*count+i]=0.0;
}

// Output rows: sediment RP_TAUB, skin z0, rough drag, smooth drag.
extern "C" __global__ void sediment_soulsby_clarke_skin_shear(
    const long long count,
    const double *depth,
    const double *speed,
    const double *kinematic_viscosity,
    const double *water_density,
    const double *ks_sand,
    const double *ks_silt,
    const double *mud_factor,
    const double *critical_mud_1,
    const double *critical_mud_2,
    const double *fluff_fraction,
    double *output,
    int *status)
{
    const long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=count) return;
    ss_zero(i,count,output);

    bool valid=isfinite(depth[i]) && isfinite(speed[i]) &&
        isfinite(kinematic_viscosity[i]) && isfinite(water_density[i]) &&
        isfinite(ks_sand[i]) && isfinite(ks_silt[i]) &&
        isfinite(mud_factor[i]) && isfinite(critical_mud_1[i]) &&
        isfinite(critical_mud_2[i]) && isfinite(fluff_fraction[i]);
    valid=valid && depth[i]>0.0 && speed[i]>=0.0 &&
        kinematic_viscosity[i]>0.0 && water_density[i]>0.0 &&
        ks_sand[i]>=0.0 && ks_silt[i]>=0.0 &&
        critical_mud_2[i]>critical_mud_1[i] &&
        fluff_fraction[i]>=0.0 && fluff_fraction[i]<=1.0;
    if(!valid) { status[i]=1; return; }

    double alpha;
    if(mud_factor[i]>critical_mud_2[i]) {
        alpha=1.0;
    } else if(mud_factor[i]>critical_mud_1[i]) {
        const double position=(mud_factor[i]-critical_mud_1[i])/
            (critical_mud_2[i]-critical_mud_1[i]);
        alpha=3.0*position*position-
            2.0*position*position*position;
    } else {
        alpha=0.0;
    }

    const double ks_bed=alpha*ks_silt[i]+(1.0-alpha)*ks_sand[i];
    const double ks_effective=fluff_fraction[i]*ks_silt[i]+
        (1.0-fluff_fraction[i])*ks_bed;
    const double z0_skin=fmax(ks_effective/30.0,1.0e-12);

    // compbsskin floors speed only while evaluating the Reynolds number.
    const double reynolds=fmax(speed[i],1.0e-12)*depth[i]/
        kinematic_viscosity[i];
    const double cd_smooth=1.615e-4*exp(6.0*pow(reynolds,-0.08));
    const double denominator=log(depth[i]/z0_skin)-1.0;
    if(!isfinite(reynolds) || !isfinite(cd_smooth) ||
       !isfinite(denominator) || denominator==0.0) {
        status[i]=1; return;
    }
    // Formula-specific 0.40 from locked compbsskin; do not change to 0.41.
    const double cd_rough=pow(0.40/denominator,2.0);
    double rp_taub=0.0;
    if(speed[i]>=1.0e-6)
        rp_taub=water_density[i]*fmax(cd_rough,cd_smooth)*
            speed[i]*speed[i];

    valid=isfinite(rp_taub) && isfinite(z0_skin) &&
        isfinite(cd_rough) && isfinite(cd_smooth) &&
        rp_taub>=0.0 && z0_skin>=0.0 &&
        cd_rough>=0.0 && cd_smooth>=0.0;
    if(!valid) { status[i]=1; ss_zero(i,count,output); return; }
    output[i]=rp_taub;
    output[count+i]=z0_skin;
    output[2*count+i]=cd_rough;
    output[3*count+i]=cd_smooth;
    status[i]=0;
}
