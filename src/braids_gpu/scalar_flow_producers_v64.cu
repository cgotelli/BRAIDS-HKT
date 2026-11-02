// SPDX-License-Identifier: GPL-3.0-only
//
// Resident producer closure for the bounded v64 scalar FLOW seam.  The
// mixture kernel mirrors the classic FLOW compdiam/compsandfrac call site
// used by erosed (seddm == sedd50).  The hydraulic kernel consumes the
// accepted implicit-midpoint state and implements the current-only 2DH
// dwnvel/erosed scalar producer arithmetic.  Neither kernel owns faces,
// boundaries, Exner, or a CPU fallback.

#define V64_MAX_CLASSES 32

__device__ __forceinline__ void v64_zero_mixture(
    const long long cell, const long long ncell, const int nclass,
    double *fraction, double *mixture)
{
    for(int grain=0;grain<nclass;++grain)
        fraction[(long long)grain*ncell+cell]=0.0;
    #pragma unroll
    for(int row=0;row<6;++row)
        mixture[(long long)row*ncell+cell]=0.0;
}

// mixture rows: arithmetic mean Dm, geometric mean Dg, true geometric DgSD,
// D90, sand fraction, mud fraction.  Distribution CDF values are fractions
// in [0,1], exactly as logseddia is stored after rdsed's final /100 conversion.
extern "C" __global__ void scalar_flow_mixture_v64(
    const long long ncell, const int nclass, const int max_points,
    const int spatial_d50_mode,
    const double *surface_mass, const double *bulk_density,
    const double *retained_fraction, const int retain_depleted,
    const double *class_d50, const double *log_sigma,
    const int *is_mud, const int *distribution_count,
    const double *distribution_cdf, const double *distribution_log_diameter,
    const double *spatial_d50,
    double *fraction, double *resolved_d50, double *mixture, int *status)
{
    const long long cell=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(cell>=ncell) return;
    v64_zero_mixture(cell,ncell,nclass,fraction,mixture);
    for(int grain=0;grain<nclass;++grain)
        resolved_d50[(long long)grain*ncell+cell]=0.0;

    bool valid=nclass>0 && nclass<=V64_MAX_CLASSES && max_points>=1;
    double total_volume=0.0;
    for(int grain=0;grain<nclass;++grain) {
        const double mass=surface_mass[(long long)grain*ncell+cell];
        valid=valid && isfinite(mass) && mass>=0.0 &&
            isfinite(bulk_density[grain]) && bulk_density[grain]>0.0 &&
            (is_mud[grain]==0 || is_mud[grain]==1);
        if(valid) total_volume+=mass/bulk_density[grain];
    }
    valid=valid && isfinite(total_volume) && total_volume>=0.0;
    if(!valid) { status[cell]=1; return; }

    if(total_volume>0.0) {
        for(int grain=0;grain<nclass;++grain) {
            const double volume=surface_mass[(long long)grain*ncell+cell]/
                bulk_density[grain];
            fraction[(long long)grain*ncell+cell]=volume/total_volume;
        }
    } else if(retain_depleted!=0) {
        double fraction_sum=0.0;
        for(int grain=0;grain<nclass;++grain) {
            const double share=retained_fraction[(long long)grain*ncell+cell];
            valid=valid && isfinite(share) && share>=0.0;
            fraction_sum+=share;
            fraction[(long long)grain*ncell+cell]=share;
        }
        valid=valid && isfinite(fraction_sum) &&
            fabs(fraction_sum-1.0)<=512.0*2.2204460492503131e-16*nclass;
        if(!valid) { status[cell]=1; return; }
    }

    if(spatial_d50_mode!=0) {
        valid=nclass==1 && isfinite(spatial_d50[cell]) &&
            spatial_d50[cell]>0.0 && isfinite(log_sigma[0]) &&
            log_sigma[0]>=0.0;
        if(!valid) {
            v64_zero_mixture(cell,ncell,nclass,fraction,mixture);
            status[cell]=1; return;
        }
        const double d50=spatial_d50[cell];
        resolved_d50[cell]=d50;
        mixture[5*ncell+cell]=is_mud[0] ? fraction[cell] : 0.0;
        if(!is_mud[0]) {
            // compdiam's spatial branch intentionally returns DgSD=0.
            mixture[cell]=d50*exp(0.5*log_sigma[0]*log_sigma[0]);
            mixture[ncell+cell]=d50;
            mixture[2*ncell+cell]=0.0;
            // xx=0.90 lands on the tabulated FLOW 90th-percentile value.
            mixture[3*ncell+cell]=d50*exp(1.2816*log_sigma[0]);
            mixture[4*ncell+cell]=(d50>=64.0e-6 && d50<2.0e-3) ? 1.0 : 0.0;
        }
        for(int row=0;row<6;++row)
            valid=valid && isfinite(mixture[(long long)row*ncell+cell]);
        status[cell]=valid ? 0 : 1;
        if(!valid) v64_zero_mixture(cell,ncell,nclass,fraction,mixture);
        return;
    }

    double fracdiam=0.0;
    double mud_fraction=0.0;
    for(int grain=0;grain<nclass;++grain) {
        const double share=fraction[(long long)grain*ncell+cell];
        resolved_d50[(long long)grain*ncell+cell]=class_d50[grain];
        if(is_mud[grain]) mud_fraction+=share;
        else fracdiam+=share;
    }
    double dm=0.0;
    double dg=1.0;
    double dgsd=0.0;
    double sand_fraction=0.0;
    if(fracdiam>0.0) {
        for(int grain=0;grain<nclass;++grain) if(!is_mud[grain]) {
            const double share=fraction[(long long)grain*ncell+cell]/fracdiam;
            // Classic erosed passes sedd50 for both seddm and sedd50.
            dm+=share*class_d50[grain];
            dg*=pow(class_d50[grain],share);
            if(class_d50[grain]<2.0e-3) sand_fraction+=share;
        }
        for(int grain=0;grain<nclass;++grain) if(!is_mud[grain]) {
            const double share=fraction[(long long)grain*ncell+cell]/fracdiam;
            if(fraction[(long long)grain*ncell+cell]>0.0)
                dgsd+=share*pow(log(class_d50[grain])-log(dg),2.0);
        }
        dgsd=exp(sqrt(dgsd));
    } else {
        dg=0.0;
        // dgsd, dm and sand_fraction deliberately remain zero.
    }

    // One requested percentile, D90.  This is the merge scan in compdiam,
    // including its source-defined depleted/mud-only behaviour.
    int stage[V64_MAX_CLASSES];
    for(int grain=0;grain<nclass;++grain) stage[grain]=0;
    double log_previous=0.0;
    double accumulated=0.0;
    double d90=0.0;
    const double required=0.90;
    int iterations=0;
    while(required>accumulated) {
        double log_diameter=999.999;
        int trigger=-1;
        double density=0.0;
        for(int grain=0;grain<nclass;++grain) {
            const int s=stage[grain];
            const int count=distribution_count[grain];
            if(s<count) {
                if(s>0) {
                    const long long previous=(long long)grain*max_points+s-1;
                    const long long next=previous+1;
                    const double fraction_factor=fracdiam>0.0 ?
                        fraction[(long long)grain*ncell+cell]/fracdiam : 1.0;
                    density+=fraction_factor*
                        (distribution_cdf[next]-distribution_cdf[previous])/
                        (distribution_log_diameter[next]-
                         distribution_log_diameter[previous]);
                }
                const double candidate=distribution_log_diameter[
                    (long long)grain*max_points+s];
                if(candidate<log_diameter) {
                    log_diameter=candidate;
                    trigger=grain;
                }
            }
        }
        if(trigger<0) {
            d90=exp(log_previous);
            break;
        }
        const double interval=density*(log_diameter-log_previous);
        if(interval>=required-accumulated) {
            log_previous+=(log_diameter-log_previous)*
                (required-accumulated)/interval;
            accumulated=required;
            d90=exp(log_previous);
            break;
        }
        accumulated+=interval;
        log_previous=log_diameter;
        if(stage[trigger]==0) {
            const int count=distribution_count[trigger];
            const double first=distribution_log_diameter[
                (long long)trigger*max_points];
            const double last=distribution_log_diameter[
                (long long)trigger*max_points+count-1];
            if(last<=first) {
                // Exact transcription of compdiam's stage=5 Dirac branch.
                // The locked routine uses the raw surface fraction here (not
                // fracdiam-normalized density) and emits every percentile
                // inside that point mass at the exact diameter.
                stage[trigger]=count>5 ? count : 5;
                const double point_mass=fraction[
                    (long long)trigger*ncell+cell];
                  // Right-continuous inverse CDF.  The locked compdiam strict
                  // comparison leaves an exact Dirac-edge Dxx unassigned.
                  if(point_mass>=required-accumulated) {
                    d90=exp(log_diameter);
                    accumulated=required;
                    break;
                }
                accumulated+=point_mass;
            } else {
                stage[trigger]=1;
            }
        } else {
            stage[trigger]+=1;
        }
        ++iterations;
        if(iterations>nclass*max_points+nclass) {
            valid=false; break;
        }
    }

    mixture[cell]=dm;
    mixture[ncell+cell]=dg;
    mixture[2*ncell+cell]=dgsd;
    mixture[3*ncell+cell]=d90;
    mixture[4*ncell+cell]=sand_fraction;
    mixture[5*ncell+cell]=mud_fraction;
    valid=valid && isfinite(dm) && isfinite(dg) && isfinite(dgsd) &&
        isfinite(d90) && isfinite(sand_fraction) && isfinite(mud_fraction) &&
        dm>=0.0 && dg>=0.0 && dgsd>=0.0 && d90>=0.0 &&
        sand_fraction>=0.0 && sand_fraction<=1.0 &&
        mud_fraction>=0.0 && mud_fraction<=1.0;
    status[cell]=valid ? 0 : 1;
    if(!valid) v64_zero_mixture(cell,ncell,nclass,fraction,mixture);
}

__device__ __forceinline__ void v64_zero_hydraulic(
    const long long cell, const long long ncell, double *output)
{
    #pragma unroll
    for(int row=0;row<15;++row)
        output[(long long)row*ncell+cell]=0.0;
}

// hydraulic rows: raw depth, erosed working depth, UCHAR, VCHAR, VELCH,
// ZVLCH, EFUMN, EFVMN, USTAR, total RP_TAUB, pre-secondary USTAR^2,
// post-secondary USTAR^2, bed dir x/y, sediment Chezy.  dwnvel's observation depth stays raw;
// erosed's roughness/RP_DEPTH role is independently floored to 0.01 m.
extern "C" __global__ void scalar_flow_hydraulic_v64(
    const long long ncell, const double *q, const int *active,
    const double *chezy, const double *z0, const double *drag,
    const double *spiral_intensity, const int secondary_enabled,
    const double espir, const double water_density, const double gravity,
    const double direction_epsilon, double *output, int *status)
{
    const long long cell=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(cell>=ncell) return;
    v64_zero_hydraulic(cell,ncell,output);
    if(active[cell]==0) { status[cell]=0; return; }
    bool valid=active[cell]==1 && isfinite(q[cell]) && q[cell]>0.0 &&
        isfinite(q[ncell+cell]) && isfinite(q[2*ncell+cell]) &&
        isfinite(chezy[cell]) && chezy[cell]>0.0 &&
        isfinite(z0[cell]) && z0[cell]>0.0 &&
        isfinite(drag[cell]) && drag[cell]>0.0 &&
        isfinite(water_density) && water_density>0.0 &&
        isfinite(gravity) && gravity>0.0 &&
        isfinite(direction_epsilon) && direction_epsilon>0.0 &&
        (secondary_enabled==0 || secondary_enabled==1) &&
        isfinite(espir) && isfinite(spiral_intensity[cell]);
    if(!valid) { status[cell]=1; return; }

    const double raw_depth=q[cell];
    const double working_depth=fmax(0.01,raw_depth);
    double u=q[ncell+cell]/raw_depth;
    double v=q[2*ncell+cell]/raw_depth;
    if(fabs(u)<direction_epsilon) u=0.0;
    if(fabs(v)<direction_epsilon) v=0.0;
    const double speed=sqrt(u*u+v*v);
    const double observation_height=(double)0.368f*raw_depth;
    const double logarithm=log1p(observation_height/z0[cell]);
    valid=isfinite(speed) && isfinite(logarithm) && logarithm>0.0;
    if(!valid) { status[cell]=1; return; }
    // erosed recomputes its sediment Chezy from total z0rou and its distinct
    // 0.01 m working depth.  The upstream hydraulic Chezy is not substituted.
    const double euler_number=2.7182818284590452353602874713526625;
    const double von_karman=(double)0.41f;
    const double sediment_chezy=sqrt(gravity)*
        log1p(working_depth/fmax(1.0e-8,euler_number*z0[cell]))/von_karman;
    const double ustar=speed*von_karman/logarithm;
    const double effective_speed=ustar*sediment_chezy/sqrt(gravity);
    const double effective_u=effective_speed*u/(speed+direction_epsilon);
    const double effective_v=effective_speed*v/(speed+direction_epsilon);
    const double total_shear=water_density*drag[cell]*speed*speed;
    const double ust2=ustar*ustar;

    double post_ust2=0.0;
    double direction_x=0.0;
    double direction_y=0.0;
    if(speed>0.0) {
        double alpha=0.0;
        if(secondary_enabled) {
            alpha=sqrt(gravity)/0.4/sediment_chezy;
            alpha=12.5*espir*(1.0-alpha)*spiral_intensity[cell]/speed;
        }
        const double tx=ust2*(u+alpha*v)/speed;
        const double ty=ust2*(v-alpha*u)/speed;
        post_ust2=sqrt(tx*tx+ty*ty);
        if(post_ust2>direction_epsilon) {
            direction_x=tx/post_ust2;
            direction_y=ty/post_ust2;
        }
    }
    const double local[15]={raw_depth,working_depth,u,v,speed,
        observation_height,effective_u,effective_v,ustar,total_shear,ust2,
        post_ust2,direction_x,direction_y,sediment_chezy};
    #pragma unroll
    for(int row=0;row<15;++row) {
        output[(long long)row*ncell+cell]=local[row];
        valid=valid && isfinite(local[row]);
    }
    valid=valid && ustar>=0.0 && total_shear>=0.0 && ust2>=0.0 &&
        post_ust2>=0.0 && sediment_chezy>0.0;
    status[cell]=valid ? 0 : 1;
    if(!valid) v64_zero_hydraulic(cell,ncell,output);
}
