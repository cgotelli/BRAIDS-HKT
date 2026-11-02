// Resident composition kernels for the Delft3D-current M1 capacity ABI.
//
// These kernels fill the gaps between the already-qualified isolated CUDA
// operators.  They do not reconstruct faces or update Exner state.  All
// arrays use class-major layout: point = grain*ncell + cell and vector rows
// use (component*nclass + grain)*ncell + cell.

__device__ __forceinline__ long long m1_vec(
    const int component, const int grain, const long long cell,
    const int nclass, const long long ncell)
{
    return ((long long)component*nclass+grain)*ncell+cell;
}

__device__ __forceinline__ void m1_zero_point(
    const long long point, const long long points,
    double *bed, double *suspended, double *absolute_shields,
    double *critical_shields, double *stress_ratio)
{
    bed[point]=0.0;
    bed[points+point]=0.0;
    suspended[point]=0.0;
    suspended[points+point]=0.0;
    absolute_shields[point]=0.0;
    critical_shields[point]=0.0;
    stress_ratio[point]=0.0;
}

// Exact current-only Delft3D tranb2 (MPM) and tranb1
// (Engelund--Hansen) scalar capacities.  Formula parameters and hiding are
// class/cell matrices even when a configured run keeps them spatially
// constant.  This avoids host-side expansion in a future runtime adapter.
extern "C" __global__ void m1_basic_capacity(
    const long long ncell, const int nclass, const int law,
    const double *depth, const double *u, const double *v,
    const double *chezy, const double *d50, const double *mixture_d90,
    const double *solid_density, const double water_density,
    const double gravity, const double *acal, const double *susfac,
    const double *hiding, double *bed, double *suspended,
    double *absolute_shields, double *critical_shields,
    double *stress_ratio, int *status)
{
    const long long point=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    const long long points=ncell*(long long)nclass;
    if(point>=points) return;
    const int grain=(int)(point/ncell);
    const long long cell=point-(long long)grain*ncell;
    m1_zero_point(point,points,bed,suspended,absolute_shields,
                  critical_shields,stress_ratio);
    bool valid=(law==2 || law==8) && isfinite(depth[cell]) &&
        isfinite(u[cell]) && isfinite(v[cell]) && isfinite(chezy[cell]) &&
        isfinite(d50[grain]) &&
        isfinite(solid_density[grain]) && isfinite(water_density) &&
        isfinite(gravity) && isfinite(acal[point]) &&
        isfinite(susfac[point]) && isfinite(hiding[point]);
    valid=valid && depth[cell]>=0.0 && chezy[cell]>=0.0 &&
        d50[grain]>0.0 &&
        solid_density[grain]>water_density && water_density>0.0 &&
        gravity>0.0 && acal[point]>=0.0 && hiding[point]>=0.0;
    // tranb2 consumes one local RP_D90MX mixture statistic per FLOW point,
    // not a separate roughness D90 for each transported fraction.  The EH
    // branch does not consume D90 at all.
    if(law==2) valid=valid && isfinite(mixture_d90[cell]) &&
        mixture_d90[cell]>0.0;
    if(!valid) { status[point]=1; return; }
    if(depth[cell]==0.0 || acal[point]==0.0) {
        status[point]=0; return;
    }
    if(chezy[cell]<=0.0) { status[point]=1; return; }
    const double speed=sqrt(u[cell]*u[cell]+v[cell]*v[cell]);
    if(!isfinite(speed)) { status[point]=1; return; }
    if(speed==0.0) { status[point]=0; return; }
    const double relative=solid_density[grain]/water_density-1.0;
    const double theta=pow(speed/chezy[cell],2.0)/(relative*d50[grain]);
    absolute_shields[point]=theta;
    double bed_volume=0.0;
    double suspended_volume=0.0;
    if(law==2) {
        const double grain_chezy=18.0*log10(
            12.0*fmax(depth[cell]/mixture_d90[cell],1.0));
        if(!isfinite(grain_chezy) || grain_chezy<=0.0) {
            status[point]=1; m1_zero_point(point,points,bed,suspended,
                absolute_shields,critical_shields,stress_ratio); return;
        }
        const double ripple=fmin(pow(chezy[cell]/grain_chezy,1.5),1.0);
        const double threshold=hiding[point]*0.047;
        const double excess=fmax(ripple*theta-threshold,0.0);
        bed_volume=acal[point]*8.0*
            sqrt(gravity*relative*d50[grain]*excess)*d50[grain]*excess;
        critical_shields[point]=threshold;
        stress_ratio[point]=threshold>0.0 ? ripple*theta/threshold :
            (ripple*theta>0.0 ? 1.7976931348623157e308 : 0.0);
    } else {
        // Delft's SusFac deliberately remains a signed split coefficient;
        // the dispatcher never clamps or discards either channel.
        const double total=0.05*acal[point]*(chezy[cell]*chezy[cell]/gravity)*
            pow(d50[grain],1.5)*sqrt(gravity*relative)*pow(theta,2.5);
        bed_volume=(1.0-susfac[point])*total;
        suspended_volume=susfac[point]*total;
        // EH has no critical threshold.  Leave stress_ratio explicitly zero;
        // absolute_shields is the diagnostic used by Koch--Flokstra.
    }
    const double direction_x=u[cell]/speed;
    const double direction_y=v[cell]/speed;
    bed[point]=solid_density[grain]*bed_volume*direction_x;
    bed[points+point]=solid_density[grain]*bed_volume*direction_y;
    suspended[point]=solid_density[grain]*suspended_volume*direction_x;
    suspended[points+point]=solid_density[grain]*suspended_volume*direction_y;
    valid=isfinite(theta) && theta>=0.0 && isfinite(bed[point]) &&
        isfinite(bed[points+point]) && isfinite(suspended[point]) &&
        isfinite(suspended[points+point]) &&
        isfinite(critical_shields[point]) &&
        isfinite(stress_ratio[point]);
    status[point]=valid ? 0 : 1;
    if(!valid) m1_zero_point(point,points,bed,suspended,
                             absolute_shields,critical_shields,stress_ratio);
}

__device__ __forceinline__ double m1_wilcock_rate(const double phi)
{
    if(phi<1.35) return 0.002*pow(phi,7.5);
    return 14.0*pow(1.0-0.894/sqrt(phi),4.5);
}

// Wilcock--Crowe, modified Wilcock--Crowe, and Gaeuman using the selected
// C/M/W hydraulic Chezy for the total-current transport scale.  RP_TAUB is a
// separate input and may be total or Soulsby/Clarke skin shear.
extern "C" __global__ void m1_wilcock_capacity_chezy(
    const long long ncell, const int nclass, const int law,
    const double *depth, const double *u, const double *v,
    const double *chezy, const double *sediment_shear,
    const double water_density, const double gravity,
    const double *d50, const double *solid_density,
    const double *bulk_density, const double *surface_mass,
    const int *sediment_type, const double *parameter_1,
    const double *parameter_2, double *bed, double *suspended,
    double *absolute_shields, double *critical_shields,
    double *stress_ratio, double *mixture, int *status)
{
    const long long cell=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(cell>=ncell) return;
    const long long points=ncell*(long long)nclass;
    for(int grain=0;grain<nclass;++grain) {
        const long long point=(long long)grain*ncell+cell;
        m1_zero_point(point,points,bed,suspended,absolute_shields,
                      critical_shields,stress_ratio);
    }
    mixture[cell]=0.0;
    mixture[ncell+cell]=0.0;
    mixture[2*ncell+cell]=0.0;
    bool valid=(law==5 || law==6 || law==7) && isfinite(depth[cell]) &&
        isfinite(u[cell]) && isfinite(v[cell]) && isfinite(chezy[cell]) &&
        isfinite(sediment_shear[cell]) && isfinite(water_density) &&
        isfinite(gravity) && depth[cell]>=0.0 && chezy[cell]>=0.0 &&
        sediment_shear[cell]>=0.0 && water_density>0.0 && gravity>0.0;
    double total_volume=0.0;
    double fallback_log_sum=0.0;
    for(int grain=0;grain<nclass;++grain) {
        const long long point=(long long)grain*ncell+cell;
        valid=valid && isfinite(d50[grain]) &&
            isfinite(solid_density[grain]) && isfinite(bulk_density[grain]) &&
            isfinite(surface_mass[point]) && isfinite(parameter_1[point]) &&
            isfinite(parameter_2[point]) && d50[grain]>0.0 &&
            solid_density[grain]>water_density && bulk_density[grain]>0.0 &&
            (sediment_type[grain]==0 || sediment_type[grain]==1) &&
            parameter_1[point]>=0.0 && parameter_2[point]>=0.0;
        if(d50[grain]>0.0 && isfinite(d50[grain]))
            fallback_log_sum+=log(d50[grain]);
        if(sediment_type[grain]==0 && isfinite(surface_mass[point]) &&
           surface_mass[point]>=0.0 && bulk_density[grain]>0.0)
            total_volume+=surface_mass[point]/bulk_density[grain];
        else if(surface_mass[point]<0.0) valid=false;
    }
    if(!valid || !isfinite(total_volume)) { status[cell]=1; return; }
    double mean_log=fallback_log_sum/(double)nclass;
    double sand_fraction=0.0;
    if(total_volume>0.0) {
        mean_log=0.0;
        for(int grain=0;grain<nclass;++grain) {
            if(sediment_type[grain]!=0) continue;
            const long long point=(long long)grain*ncell+cell;
            const double share=(surface_mass[point]/bulk_density[grain])/
                total_volume;
            mean_log+=share*log(d50[grain]);
            if(d50[grain]<2.0e-3) sand_fraction+=share;
        }
    }
    const double dg=exp(mean_log);
    double variance=0.0;
    if(total_volume>0.0) {
        for(int grain=0;grain<nclass;++grain) {
            if(sediment_type[grain]!=0) continue;
            const long long point=(long long)grain*ncell+cell;
            const double share=(surface_mass[point]/bulk_density[grain])/
                total_volume;
            const double delta=log(d50[grain])-mean_log;
            variance+=share*delta*delta;
        }
    }
    const double dgsd=exp(sqrt(fmax(0.0,variance)));
    mixture[cell]=dg;
    mixture[ncell+cell]=dgsd;
    mixture[2*ncell+cell]=sand_fraction;
    valid=isfinite(dg) && isfinite(dgsd) && isfinite(sand_fraction) &&
        dg>0.0 && dgsd>=1.0 && sand_fraction>=0.0 && sand_fraction<=1.0;
    if(!valid) { status[cell]=1; return; }
    if(depth[cell]==0.0 || chezy[cell]==0.0) {
        status[cell]=0; return;
    }
    const double speed=sqrt(u[cell]*u[cell]+v[cell]*v[cell]);
    if(!isfinite(speed)) { status[cell]=1; return; }
    if(speed<=1.0e-12 || chezy[cell]<1.0e-6 || dg<1.0e-6) {
        status[cell]=0; return;
    }
    const double total_ustar=sqrt(gravity)*speed/chezy[cell];
    for(int grain=0;grain<nclass;++grain) {
        if(sediment_type[grain]!=0) continue;
        const long long point=(long long)grain*ncell+cell;
        double b=0.0,taurm=0.0,scale=1.0;
        if(law==5) {
            b=0.67/(1.0+exp(1.5-d50[grain]/dg));
            taurm=(0.021+0.015*exp(-20.0*sand_fraction))*
                (solid_density[grain]-water_density)*gravity*dg;
            scale=parameter_1[point];
        } else {
            const double sigma_phi=pow(log(dgsd)/log(2.0),2.0);
            if(law==6) {
                b=(1.0-parameter_2[point])/
                    (1.0+exp(1.5-d50[grain]/dg));
                taurm=(parameter_1[point]+0.015/
                    (1.0+exp(10.1*sigma_phi-14.14)))*
                    (solid_density[grain]-water_density)*gravity*dg;
            } else {
                b=(1.0-parameter_2[point])/
                    (1.0+exp(1.9-d50[grain]/(3.0*dg)));
                taurm=(parameter_1[point]+0.022/
                    (1.0+exp(7.1*sigma_phi-11.786)))*
                    (solid_density[grain]-water_density)*gravity*dg;
            }
        }
        const double tauri=taurm*pow(d50[grain]/dg,b);
        if(!isfinite(tauri) || tauri<=0.0) { valid=false; break; }
        const double phi=fmax(0.0,sediment_shear[cell]/tauri);
        const double wistar=m1_wilcock_rate(phi);
        const double relative=solid_density[grain]/water_density-1.0;
        const double magnitude=scale*wistar*pow(total_ustar,3.0)/
            (relative*gravity);
        bed[point]=solid_density[grain]*magnitude*u[cell]/speed;
        bed[points+point]=solid_density[grain]*magnitude*v[cell]/speed;
        const double denominator=(solid_density[grain]-water_density)*
            gravity*d50[grain];
        absolute_shields[point]=sediment_shear[cell]/denominator;
        critical_shields[point]=tauri/denominator;
        stress_ratio[point]=phi;
        valid=isfinite(b) && isfinite(taurm) && isfinite(phi) &&
            isfinite(wistar) && isfinite(magnitude) &&
            isfinite(bed[point]) && isfinite(bed[points+point]) &&
            taurm>0.0 && phi>=0.0 && wistar>=0.0 && magnitude>=0.0;
        if(!valid) break;
    }
    status[cell]=valid ? 0 : 1;
    if(!valid) {
        for(int grain=0;grain<nclass;++grain) {
            const long long point=(long long)grain*ncell+cell;
            m1_zero_point(point,points,bed,suspended,absolute_shields,
                          critical_shields,stress_ratio);
        }
        mixture[cell]=mixture[ncell+cell]=mixture[2*ncell+cell]=0.0;
    }
}

// Rotate only the BED channel for formulas that Delft3D declares as scalar
// bed-direction outputs. Delft scalar suspended transport remains in the
// depth-averaged flow direction; it must not inherit secondary-flow steering.
extern "C" __global__ void m1_rotate_scalar_capacity(
    const long long ncell, const int nclass, const double *u, const double *v,
    const double *direction_x, const double *direction_y,
    double *bed, int *status)
{
    const long long point=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    const long long points=ncell*(long long)nclass;
    if(point>=points) return;
    const long long cell=point%ncell;
    if(status[point]!=0) return;
    const double dx=direction_x[cell],dy=direction_y[cell];
    const double flow=sqrt(u[cell]*u[cell]+v[cell]*v[cell]);
    const double target=sqrt(dx*dx+dy*dy);
    bool valid=isfinite(dx) && isfinite(dy) && isfinite(flow) &&
        isfinite(target);
    const double bx=bed[point],by=bed[points+point];
    valid=valid && isfinite(bx) && isfinite(by);
    if(!valid) { status[point]=1; return; }
    const double magnitude=fabs(bx)+fabs(by);
    if(magnitude==0.0) return;
    if(flow<=0.0 || target<=0.0) {
        status[point]=1;
        bed[point]=bed[points+point]=0.0;
        return;
    }
    const double fx=u[cell]/flow,fy=v[cell]/flow;
    const double tx=dx/target,ty=dy/target;
    const double bed_scalar=bx*fx+by*fy;
    bed[point]=bed_scalar*tx;
    bed[points+point]=bed_scalar*ty;
}
