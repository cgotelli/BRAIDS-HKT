// Isolated resident FP64 CUDA operators for BRAIDS bedlaw 1 and 5/6/7.
//
// The Van Rijn algebra is a clean implementation of the current CPU
// braids_bedload_vr1993_module and the already qualified device equations in
// braids_support/gpu_fwave/suspended_exchange.cu.  No frozen source is included or
// modified.  The Wilcock-family equations follow braids_bedload_current_module
// and locked Delft3D4 trabwc/trabwc2/trabg.
//
// One CUDA thread owns one cell.  It loops over sediment classes so mixture
// statistics are formed once from a coherent surface state; there is no nested
// thread team and no host constitutive fallback.

__device__ __forceinline__ long long rc_class_index(
    const int grain, const long long cell, const long long ncell)
{
    return (long long)grain*ncell+cell;
}

__device__ __forceinline__ void rc_zero_cell(
    const long long cell, const long long ncell, const int nclass,
    double *capacity, double *diagnostics)
{
    for(int grain=0; grain<nclass; ++grain) {
        const long long point=rc_class_index(grain,cell,ncell);
        capacity[point]=0.0;
        capacity[(long long)nclass*ncell+point]=0.0;
        #pragma unroll
        for(int row=0; row<5; ++row)
            diagnostics[((long long)row*nclass+grain)*ncell+cell]=0.0;
    }
}

__device__ __forceinline__ bool rc_finite_outputs(
    const long long cell, const long long ncell, const int nclass,
    const double *capacity, const double *diagnostics)
{
    for(int grain=0; grain<nclass; ++grain) {
        const long long point=rc_class_index(grain,cell,ncell);
        if(!isfinite(capacity[point]) ||
           !isfinite(capacity[(long long)nclass*ncell+point])) return false;
        #pragma unroll
        for(int row=0; row<5; ++row)
            if(!isfinite(diagnostics[((long long)row*nclass+grain)*ncell+cell]))
                return false;
    }
    return true;
}

extern "C" __global__ void river_total_shear(
    const long long ncell, const double *depth, const double *u,
    const double *v, const double *manning_n, const double *water_density,
    const double *gravity, double *output, int *status)
{
    const long long cell=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(cell>=ncell) return;
    output[cell]=0.0;
    output[ncell+cell]=0.0;
    bool valid=isfinite(depth[cell]) && isfinite(u[cell]) && isfinite(v[cell]) &&
        isfinite(manning_n[cell]) && isfinite(water_density[cell]) &&
        isfinite(gravity[cell]);
    valid=valid && depth[cell]>=0.0 && manning_n[cell]>=0.0 &&
        water_density[cell]>0.0 && gravity[cell]>0.0;
    if(!valid) { status[cell]=1; return; }
    if(depth[cell]==0.0 || manning_n[cell]==0.0) {
        status[cell]=0; return;
    }
    const double speed=sqrt(u[cell]*u[cell]+v[cell]*v[cell]);
    const double chezy=pow(depth[cell],1.0/6.0)/manning_n[cell];
    const double shear=water_density[cell]*gravity[cell]*speed*speed/(chezy*chezy);
    valid=isfinite(speed) && isfinite(chezy) && isfinite(shear) && chezy>0.0 &&
        shear>=0.0;
    if(valid) {
        output[cell]=shear;
        output[ncell+cell]=chezy;
    }
    status[cell]=valid ? 0 : 1;
}

__device__ __forceinline__ void rc_vr_grain_parameters(
    const double d50, const double solid_density, const double water_density,
    const double gravity, const double viscosity, double *dstar,
    double *critical_shear)
{
    const double relative=(solid_density-water_density)/water_density;
    // Delft3D initsedtra and the BRAIDS CPU transcription use 0.3333 exactly.
    *dstar=d50*pow(relative*gravity/(viscosity*viscosity),0.3333);
    double shields;
    if(*dstar<=4.0) shields=0.24/(*dstar);
    else if(*dstar<=10.0) shields=0.14/pow(*dstar,0.64);
    else if(*dstar<=20.0) shields=0.04/pow(*dstar,0.10);
    else if(*dstar<=150.0) shields=0.013*pow(*dstar,0.29);
    else shields=0.055;
    *critical_shear=(solid_density-water_density)*gravity*d50*shields;
}

extern "C" __global__ void river_vr1993_capacity(
    const long long ncell, const int nclass,
    const double *depth, const double *u, const double *v,
    const double *observation_depth, const double *water_density,
    const double *gravity, const double *kinematic_viscosity,
    const double *d50, const double *d90, const double *solid_density,
    const double *minimum_depth, const double *roughness_z0,
    double *capacity, double *diagnostics, int *status)
{
    const long long cell=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(cell>=ncell) return;
    rc_zero_cell(cell,ncell,nclass,capacity,diagnostics);
    bool valid=isfinite(depth[cell]) && isfinite(u[cell]) && isfinite(v[cell]) &&
        isfinite(observation_depth[cell]) && isfinite(water_density[cell]) &&
        isfinite(gravity[cell]) && isfinite(kinematic_viscosity[cell]);
    valid=valid && depth[cell]>=0.0 && observation_depth[cell]>0.0 &&
        water_density[cell]>0.0 && gravity[cell]>0.0 &&
        kinematic_viscosity[cell]>0.0;
    for(int grain=0; grain<nclass; ++grain) {
        const long long point=rc_class_index(grain,cell,ncell);
        valid=valid && isfinite(d50[grain]) && isfinite(d90[grain]) &&
            isfinite(solid_density[grain]) && isfinite(minimum_depth[grain]) &&
            isfinite(roughness_z0[point]) && d50[grain]>0.0 &&
            d90[grain]>=d50[grain] && solid_density[grain]>water_density[cell] &&
            minimum_depth[grain]>=0.0 && roughness_z0[point]>0.0;
    }
    if(!valid) { status[cell]=1; return; }

    const double speed=sqrt(u[cell]*u[cell]+v[cell]*v[cell]);
    if(!isfinite(speed)) { status[cell]=1; return; }
    for(int grain=0; grain<nclass; ++grain) {
        const long long point=rc_class_index(grain,cell,ncell);
        double dstar,critical_shear;
        rc_vr_grain_parameters(
            d50[grain],solid_density[grain],water_density[cell],gravity[cell],
            kinematic_viscosity[cell],&dstar,&critical_shear);
        diagnostics[((long long)3*nclass+grain)*ncell+cell]=dstar;
        diagnostics[((long long)4*nclass+grain)*ncell+cell]=critical_shear;
        if(!isfinite(dstar) || !isfinite(critical_shear) || dstar<=0.0 ||
           critical_shear<=0.0) { valid=false; break; }
        if(depth[cell]<=minimum_depth[grain]) continue;

        const double rc=30.0*roughness_z0[point];
        if(12.0*depth[cell]<=fmax(rc,3.0*d90[grain])) {
            valid=false; break;
        }
        const double ustar=speed*0.41/log(
            1.0+(observation_depth[cell]/exp(1.0))/roughness_z0[point]);
        const double total_shear=water_density[cell]*ustar*ustar;
        const double f1c=0.24*pow(log10(
            12.0*depth[cell]/(3.0*d90[grain])),-2.0);
        const double fc=0.24*pow(log10(12.0*depth[cell]/rc),-2.0);
        const double muc=f1c/fc;
        const double tau_ratio=muc*total_shear/critical_shear;
        const double stage=fmax(1.0e-10,tau_ratio-1.0);
        const double skin_ustar=ustar*sqrt(muc);
        const double magnitude=0.5*solid_density[grain]*d50[grain]*skin_ustar*
            pow(dstar,-0.3)*stage;
        diagnostics[((long long)0*nclass+grain)*ncell+cell]=tau_ratio;
        diagnostics[((long long)1*nclass+grain)*ncell+cell]=ustar;
        diagnostics[((long long)2*nclass+grain)*ncell+cell]=muc;
        if(speed>1.0e-6) {
            capacity[point]=magnitude*u[cell]/speed;
            capacity[(long long)nclass*ncell+point]=magnitude*v[cell]/speed;
        }
        if(!isfinite(ustar) || !isfinite(muc) || !isfinite(tau_ratio) ||
           !isfinite(magnitude) || ustar<0.0 || muc<0.0 || tau_ratio<0.0) {
            valid=false; break;
        }
    }
    valid=valid && rc_finite_outputs(cell,ncell,nclass,capacity,diagnostics);
    status[cell]=valid ? 0 : 1;
    if(!valid) rc_zero_cell(cell,ncell,nclass,capacity,diagnostics);
}

__device__ __forceinline__ double rc_wilcock_rate(const double phi)
{
    if(phi<1.35) return 0.002*pow(phi,7.5);
    return 14.0*pow(1.0-0.894/sqrt(phi),4.5);
}

extern "C" __global__ void river_wilcock_capacity(
    const long long ncell, const int nclass, const int law,
    const double *depth, const double *u, const double *v,
    const double *manning_n, const double *bed_shear,
    const double *water_density, const double *gravity,
    const double *parameter_1, const double *parameter_2, const double *d50,
    const double *solid_density, const double *bulk_density,
    const double *surface_mass, const int *sediment_type,
    double *capacity, double *diagnostics, double *mixture, int *status)
{
    const long long cell=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(cell>=ncell) return;
    rc_zero_cell(cell,ncell,nclass,capacity,diagnostics);
    mixture[cell]=0.0;
    mixture[ncell+cell]=0.0;
    mixture[2*ncell+cell]=0.0;
    bool valid=(law==5 || law==6 || law==7) && isfinite(depth[cell]) &&
        isfinite(u[cell]) && isfinite(v[cell]) && isfinite(manning_n[cell]) &&
        isfinite(bed_shear[cell]) && isfinite(water_density[cell]) &&
        isfinite(gravity[cell]) && isfinite(parameter_1[cell]) &&
        isfinite(parameter_2[cell]);
    valid=valid && depth[cell]>=0.0 && manning_n[cell]>=0.0 &&
        bed_shear[cell]>=0.0 && water_density[cell]>0.0 && gravity[cell]>0.0 &&
        ((law==5 && parameter_1[cell]>=0.0) ||
         (law!=5 && parameter_1[cell]>=0.0));

    double fallback_log_sum=0.0;
    double total_volume=0.0;
    for(int grain=0; grain<nclass; ++grain) {
        const long long point=rc_class_index(grain,cell,ncell);
        valid=valid && isfinite(d50[grain]) && isfinite(solid_density[grain]) &&
            isfinite(bulk_density[grain]) && isfinite(surface_mass[point]) &&
            d50[grain]>0.0 && solid_density[grain]>water_density[cell] &&
            bulk_density[grain]>0.0 &&
            (sediment_type[grain]==0 || sediment_type[grain]==1);
        if(d50[grain]>0.0 && isfinite(d50[grain])) fallback_log_sum+=log(d50[grain]);
        if(sediment_type[grain]==0 && isfinite(surface_mass[point]) &&
           isfinite(bulk_density[grain]) && bulk_density[grain]>0.0)
            total_volume+=fmax(surface_mass[point],0.0)/bulk_density[grain];
    }
    valid=valid && isfinite(fallback_log_sum) && isfinite(total_volume) &&
        total_volume>=0.0;
    if(!valid) { status[cell]=1; return; }

    // Exact volume-share statistic family in braids_surface_statistics.
    double mean_log=fallback_log_sum/(double)nclass;
    double sand_fraction=0.0;
    if(total_volume>0.0) {
        mean_log=0.0;
        for(int grain=0; grain<nclass; ++grain) {
            if(sediment_type[grain]!=0) continue;
            const long long point=rc_class_index(grain,cell,ncell);
            const double share=(fmax(surface_mass[point],0.0)/bulk_density[grain])/
                total_volume;
            mean_log+=share*log(d50[grain]);
            if(d50[grain]<2.0e-3) sand_fraction+=share;
        }
    }
    const double dg=exp(mean_log);
    double spread_sum=0.0;
    if(total_volume>0.0) {
        for(int grain=0; grain<nclass; ++grain) {
            if(sediment_type[grain]!=0) continue;
            const long long point=rc_class_index(grain,cell,ncell);
            const double share=(fmax(surface_mass[point],0.0)/bulk_density[grain])/
                total_volume;
            const double delta=log(d50[grain])-mean_log;
            spread_sum+=share*delta*delta;
        }
    }
    const double dgsd=exp(sqrt(fmax(0.0,spread_sum)));
    mixture[cell]=dg;
    mixture[ncell+cell]=dgsd;
    mixture[2*ncell+cell]=sand_fraction;
    valid=isfinite(dg) && isfinite(dgsd) && isfinite(sand_fraction) && dg>0.0 &&
        dgsd>=1.0 && sand_fraction>=0.0 && sand_fraction<=1.0;
    if(!valid) {
        status[cell]=1;
        mixture[cell]=mixture[ncell+cell]=mixture[2*ncell+cell]=0.0;
        return;
    }

    if(depth[cell]==0.0 || (law==5 && parameter_1[cell]==0.0) ||
       manning_n[cell]==0.0) {
        status[cell]=0; return;
    }
    const double speed=sqrt(u[cell]*u[cell]+v[cell]*v[cell]);
    const double chezy=pow(depth[cell],1.0/6.0)/manning_n[cell];
    if(!isfinite(speed) || !isfinite(chezy)) {
        status[cell]=1;
        rc_zero_cell(cell,ncell,nclass,capacity,diagnostics);
        mixture[cell]=mixture[ncell+cell]=mixture[2*ncell+cell]=0.0;
        return;
    }
    if(speed<=1.0e-12 || chezy<=1.0e-12 || chezy<1.0e-6 || dg<1.0e-6) {
        status[cell]=0; return;
    }

    // RP_TAUB enters phi.  In contrast, ustar deliberately stays on the total
    // hydraulic Chezy chain even when RP_TAUB was supplied by a skin closure.
    const double total_ustar=sqrt(gravity[cell])*speed/chezy;
    for(int grain=0; grain<nclass; ++grain) {
        if(sediment_type[grain]!=0) continue;
        const long long point=rc_class_index(grain,cell,ncell);
        double b,taurm;
        if(law==5) {
            b=0.67/(1.0+exp(1.5-d50[grain]/dg));
            taurm=(0.021+0.015*exp(-20.0*sand_fraction))*
                (solid_density[grain]-water_density[cell])*gravity[cell]*dg;
        } else {
            const double sigma_phi=pow(log(dgsd)/log(2.0),2.0);
            if(law==6) {
                b=(1.0-parameter_2[cell])/(1.0+exp(1.5-d50[grain]/dg));
                taurm=(parameter_1[cell]+0.015/
                    (1.0+exp(10.1*sigma_phi-14.14)))*
                    (solid_density[grain]-water_density[cell])*gravity[cell]*dg;
            } else {
                b=(1.0-parameter_2[cell])/
                    (1.0+exp(1.9-d50[grain]/(3.0*dg)));
                taurm=(parameter_1[cell]+0.022/
                    (1.0+exp(7.1*sigma_phi-11.786)))*
                    (solid_density[grain]-water_density[cell])*gravity[cell]*dg;
            }
        }
        const double tauri=taurm*pow(d50[grain]/dg,b);
        const double phi=fmax(0.0,bed_shear[cell]/tauri);
        const double wistar=rc_wilcock_rate(phi);
        const double relative=solid_density[grain]/water_density[cell]-1.0;
        const double formula_scale=law==5 ? parameter_1[cell] : 1.0;
        const double magnitude=formula_scale*wistar*pow(total_ustar,3.0)/
            (relative*gravity[cell]);
        capacity[point]=solid_density[grain]*magnitude*u[cell]/speed;
        capacity[(long long)nclass*ncell+point]=solid_density[grain]*magnitude*v[cell]/speed;
        diagnostics[((long long)0*nclass+grain)*ncell+cell]=phi;
        diagnostics[((long long)1*nclass+grain)*ncell+cell]=total_ustar;
        diagnostics[((long long)2*nclass+grain)*ncell+cell]=wistar;
        diagnostics[((long long)3*nclass+grain)*ncell+cell]=tauri;
        diagnostics[((long long)4*nclass+grain)*ncell+cell]=taurm;
        if(!isfinite(b) || !isfinite(taurm) || !isfinite(tauri) ||
           !isfinite(phi) || !isfinite(wistar) || !isfinite(magnitude) ||
           taurm<=0.0 || tauri<=0.0 || phi<0.0 || wistar<0.0 || magnitude<0.0) {
            valid=false; break;
        }
    }
    valid=valid && rc_finite_outputs(cell,ncell,nclass,capacity,diagnostics);
    status[cell]=valid ? 0 : 1;
    if(!valid) {
        rc_zero_cell(cell,ncell,nclass,capacity,diagnostics);
        mixture[cell]=mixture[ncell+cell]=mixture[2*ncell+cell]=0.0;
    }
}
