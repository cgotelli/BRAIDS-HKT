// Isolated CUDA scalar operators for four current-only Delft3D4 capacities.
//
// The equations mirror the clean BRAIDS CPU transcriptions in
// braids_delft_capacity_module.f90 and braids_soulsby_van_rijn_module.f90.
// This source is deliberately not wired into the production sediment method
// catalogue.  Every input and output is a resident FP64 array; invalid states
// are zeroed and marked in the per-point status array.
__device__ __forceinline__ double dc_shields_curve(const double dstar)
{
    if(!isfinite(dstar) || dstar<=0.0) return 0.0;
    if(dstar<=4.0) return 0.240/dstar;
    if(dstar<=10.0) return 0.140/pow(dstar,0.64);
    if(dstar<=20.0) return 0.040/pow(dstar,0.10);
    if(dstar<=150.0) return 0.013*pow(dstar,0.29);
    return 0.055;
}

__device__ __forceinline__ void dc_zero(
    const long long i,const long long count,double *output)
{
    #pragma unroll
    for(int field=0;field<7;++field) output[(long long)field*count+i]=0.0;
}

__device__ __forceinline__ bool dc_finite7(
    const long long i,const long long count,const double *output)
{
    #pragma unroll
    for(int field=0;field<7;++field)
        if(!isfinite(output[(long long)field*count+i])) return false;
    return true;
}

// output rows: bed x/y, suspended x/y, Shields, auxiliary 1, auxiliary 2.
extern "C" __global__ void delft_general_capacity(
    const long long count,const double *u,const double *v,const double *d50,
    const double *solid_density,const double *water_density,const double *gravity,
    const double *chezy,const double *hiding_factor,const double *bed_calibration,
    const double *bed_shields_power,const double *bed_excess_power,
    const double *bed_ripple_factor,const double *bed_critical_shields,
    const double *sus_calibration,const double *sus_shields_power,
    const double *sus_excess_power,const double *sus_ripple_factor,
    const double *sus_critical_shields,double *output,int *status)
{
    const long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=count) return;
    dc_zero(i,count,output);
    bool valid=isfinite(u[i]) && isfinite(v[i]) && isfinite(d50[i]) &&
        isfinite(solid_density[i]) && isfinite(water_density[i]) &&
        isfinite(gravity[i]) && isfinite(chezy[i]) && isfinite(hiding_factor[i]) &&
        isfinite(bed_calibration[i]) && isfinite(bed_shields_power[i]) &&
        isfinite(bed_excess_power[i]) && isfinite(bed_ripple_factor[i]) &&
        isfinite(bed_critical_shields[i]) && isfinite(sus_calibration[i]) &&
        isfinite(sus_shields_power[i]) && isfinite(sus_excess_power[i]) &&
        isfinite(sus_ripple_factor[i]) && isfinite(sus_critical_shields[i]);
    valid=valid && d50[i]>0.0 && solid_density[i]>water_density[i] &&
        water_density[i]>0.0 && gravity[i]>0.0 && chezy[i]>0.0 &&
        hiding_factor[i]>=0.0 && bed_calibration[i]>=0.0 &&
        sus_calibration[i]>=0.0 && bed_ripple_factor[i]>=0.0 &&
        sus_ripple_factor[i]>=0.0 && bed_critical_shields[i]>=0.0 &&
        sus_critical_shields[i]>=0.0;
    if(!valid) { status[i]=1; return; }

    const double speed=sqrt(u[i]*u[i]+v[i]*v[i]);
    if(!isfinite(speed)) { status[i]=1; return; }
    if(speed==0.0) { status[i]=0; return; }
    const double relative_density=solid_density[i]/water_density[i]-1.0;
    const double shields=pow(speed/chezy[i],2.0)/(relative_density*d50[i]);
    const double bed_excess=fmax(
        bed_ripple_factor[i]*shields-hiding_factor[i]*bed_critical_shields[i],0.0);
    const double sus_excess=fmax(
        sus_ripple_factor[i]*shields-hiding_factor[i]*sus_critical_shields[i],0.0);
    const double scale=pow(d50[i],1.5)*sqrt(gravity[i]*relative_density);
    const double bed_volume=bed_calibration[i]*scale*
        pow(shields,bed_shields_power[i])*pow(bed_excess,bed_excess_power[i]);
    const double sus_volume=sus_calibration[i]*scale*
        pow(shields,sus_shields_power[i])*pow(sus_excess,sus_excess_power[i]);
    output[i]=solid_density[i]*bed_volume*u[i]/speed;
    output[count+i]=solid_density[i]*bed_volume*v[i]/speed;
    output[2*count+i]=solid_density[i]*sus_volume*u[i]/speed;
    output[3*count+i]=solid_density[i]*sus_volume*v[i]/speed;
    output[4*count+i]=shields;
    valid=dc_finite7(i,count,output) && shields>=0.0;
    status[i]=valid ? 0 : 1;
    if(!valid) dc_zero(i,count,output);
}

extern "C" __global__ void delft_modified_van_rijn_1984(
    const long long count,const double *depth,const double *u,const double *v,
    const double *d50,const double *d90,const double *solid_density,
    const double *water_density,const double *gravity,
    const double *kinematic_viscosity,const double *calibration,
    const double *reference_height,const double *settling_velocity,
    const double *mud_power,const double *mud_fraction,double *output,int *status)
{
    const long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=count) return;
    dc_zero(i,count,output);
    bool valid=isfinite(depth[i]) && isfinite(u[i]) && isfinite(v[i]) &&
        isfinite(d50[i]) && isfinite(d90[i]) && isfinite(solid_density[i]) &&
        isfinite(water_density[i]) && isfinite(gravity[i]) &&
        isfinite(kinematic_viscosity[i]) && isfinite(calibration[i]) &&
        isfinite(reference_height[i]) && isfinite(settling_velocity[i]) &&
        isfinite(mud_power[i]) && isfinite(mud_fraction[i]);
    valid=valid && depth[i]>=0.0 && d50[i]>0.0 && d90[i]>0.0 &&
        solid_density[i]>water_density[i] && water_density[i]>0.0 &&
        gravity[i]>0.0 && kinematic_viscosity[i]>0.0 && calibration[i]>=0.0 &&
        reference_height[i]>0.0 && settling_velocity[i]>=0.0 && mud_fraction[i]>=0.0;
    if(!valid) { status[i]=1; return; }

    const double speed=sqrt(u[i]*u[i]+v[i]*v[i]);
    if(!isfinite(speed)) { status[i]=1; return; }
    if(depth[i]/reference_height[i]<(double)1.33f || speed<(double)1.0e-3f) {
        status[i]=0; return;
    }
    const double relative_density=solid_density[i]/water_density[i]-1.0;
    const double dstar=d50[i]*pow(relative_density*gravity[i]/
        kinematic_viscosity[i]/kinematic_viscosity[i],(double)(1.0f/3.0f));
    const double log_reference=log10((double)12.0f*depth[i]/reference_height[i]);
    const double roughness_factor=pow(log_reference/
        log10((double)12.0f*depth[i]/(double)3.0f/d90[i]),2.0);
    const double friction_factor=(double)0.24f*pow(log_reference,-2.0);
    const double shear=(double)0.125f*water_density[i]*friction_factor*speed*speed;
    const double shields=roughness_factor*shear/
        ((solid_density[i]-water_density[i])*gravity[i]*d50[i]);
    const double critical_shields=dc_shields_curve(dstar);
    const double total_critical_shear=(solid_density[i]-water_density[i])*
        gravity[i]*d50[i]*critical_shields*pow(1.0+mud_fraction[i],mud_power[i]);
    const double transport_stage=fmax(
        (roughness_factor*shear-total_critical_shear)/total_critical_shear,
        (double)0.000001f);
    const double concentration=(double)0.015f*calibration[i]*d50[i]/
        reference_height[i]*pow(transport_stage,(double)1.5f)/
        pow(dstar,(double)0.3f);
    const double shear_velocity=sqrt((double)0.125f*friction_factor)*speed;
    const double settling_ratio=settling_velocity[i]/shear_velocity;
    const double beta=fmin((double)1.0f+(double)2.0f*settling_ratio*settling_ratio,
        (double)1.5f);
    const double psi=(double)2.5f*pow(settling_ratio,(double)0.8f)*
        pow(concentration/(double)0.65f,(double)0.4f);
    // FLOW's iniphy assigns unsuffixed 0.41 before passing vonkar to tranb7.
    const double zc=fmin(settling_velocity[i]/(double)0.41f/shear_velocity/beta+psi,20.0);
    const double ah=reference_height[i]/depth[i];
    double integral_factor;
    if(fabs(zc-(double)1.2f)>(double)1.0e-4f)
        integral_factor=(pow(ah,zc)-pow(ah,(double)1.2f))/
            pow((double)1.0f-ah,zc)/((double)1.2f-zc);
    else
        integral_factor=-pow(ah/((double)1.0f-ah),(double)1.2f)*log(ah);
    const double sus_volume=integral_factor*speed*depth[i]*concentration;
    double bed_volume;
    if(transport_stage<(double)3.0f)
        bed_volume=(double)0.053f*pow(relative_density,(double)0.5f)*sqrt(gravity[i])*
            pow(d50[i],(double)1.5f)*pow(dstar,-(double)0.3f)*
            pow(transport_stage,(double)2.1f);
    else
        bed_volume=(double)0.100f*pow(relative_density,(double)0.5f)*sqrt(gravity[i])*
            pow(d50[i],(double)1.5f)*pow(dstar,-(double)0.3f)*
            pow(transport_stage,(double)1.5f);
    output[i]=solid_density[i]*bed_volume*u[i]/speed;
    output[count+i]=solid_density[i]*bed_volume*v[i]/speed;
    output[2*count+i]=solid_density[i]*sus_volume*u[i]/speed;
    output[3*count+i]=solid_density[i]*sus_volume*v[i]/speed;
    output[4*count+i]=shields;
    output[5*count+i]=critical_shields;
    output[6*count+i]=transport_stage;
    valid=dc_finite7(i,count,output);
    status[i]=valid ? 0 : 1;
    if(!valid) dc_zero(i,count,output);
}

extern "C" __global__ void delft_soulsby_van_rijn(
    const long long count,const double *depth,const double *u,const double *v,
    const double *d50,const double *solid_density,const double *water_density,
    const double *gravity,const double *kinematic_viscosity,
    const double *calibration,const double *d90_d50,const double *roughness_z0,
    double *output,int *status)
{
    const long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=count) return;
    dc_zero(i,count,output);
    bool valid=isfinite(depth[i]) && isfinite(u[i]) && isfinite(v[i]) &&
        isfinite(d50[i]) && isfinite(solid_density[i]) && isfinite(water_density[i]) &&
        isfinite(gravity[i]) && isfinite(kinematic_viscosity[i]) &&
        isfinite(calibration[i]) && isfinite(d90_d50[i]) && isfinite(roughness_z0[i]);
    valid=valid && depth[i]>=0.0 && d50[i]>0.0 &&
        solid_density[i]>water_density[i] && water_density[i]>0.0 && gravity[i]>0.0 &&
        kinematic_viscosity[i]>0.0 && calibration[i]>=0.0 && d90_d50[i]>0.0 &&
        roughness_z0[i]>0.0;
    if(!valid) { status[i]=1; return; }

    // These values are promoted binary32 literals in locked trab11 and in the
    // CPU transcription.  Keep their source evaluation, including 1.f/3.f.
    const double minimum_depth=(double)0.01f;
    const double maximum_depth=(double)200.0f;
    const double minimum_speed=(double)0.000001f;
    const double grain_branch=(double)0.0005f;
    const double exponent_one_third=(double)(1.0f/3.0f);
    const double exponent_one_tenth=(double)0.1f;
    const double exponent_three_fifths=(double)0.6f;
    const double exponent_six_fifths=(double)1.2f;
    const double exponent_twelve_fifths=(double)2.4f;
    const double exponent_one_half=(double)0.5f;
    const double von_karman=(double)0.41f;

    const double speed=sqrt(u[i]*u[i]+v[i]*v[i]);
    if(!isfinite(speed)) { status[i]=1; return; }
    if(speed<minimum_speed || depth[i]>maximum_depth || depth[i]<minimum_depth) {
        status[i]=0; return;
    }
    const double relative_density=solid_density[i]/water_density[i]-1.0;
    const double d90=d90_d50[i]*d50[i];
    const double dstar=pow(gravity[i]*relative_density/
        (kinematic_viscosity[i]*kinematic_viscosity[i]),exponent_one_third)*d50[i];
    if(!isfinite(relative_density) || !isfinite(d90) || !isfinite(dstar) ||
       relative_density<=0.0 || d90<=0.0 || dstar<=0.0) {
        status[i]=1; return;
    }
    double critical_speed;
    if(d50[i]<=grain_branch)
        critical_speed=(double)0.19f*pow(d50[i],exponent_one_tenth)*
            log10((double)4.0f*depth[i]/d90);
    else
        critical_speed=(double)8.5f*pow(d50[i],exponent_three_fifths)*
            log10((double)4.0f*depth[i]/d90);
    const double drag_denominator=log(depth[i]/roughness_z0[i])-1.0;
    if(drag_denominator==0.0) { status[i]=1; return; }
    const double drag=pow(von_karman/drag_denominator,2.0);
    const double shields=drag*speed*speed/(relative_density*gravity[i]*d50[i]);
    const double critical_shields=drag*critical_speed*critical_speed/
        (relative_density*gravity[i]*d50[i]);
    valid=isfinite(critical_speed) && isfinite(drag) && isfinite(shields) &&
        isfinite(critical_shields) && critical_speed>=0.0 && drag>=0.0;
    if(!valid) { status[i]=1; return; }
    output[4*count+i]=shields;
    output[5*count+i]=critical_shields;
    output[6*count+i]=critical_speed;
    const double effective_speed=pow(speed*speed,exponent_one_half);
    if(effective_speed<=critical_speed) { status[i]=0; return; }
    const double bed_coefficient=(double)0.005f*depth[i]*pow(
        d50[i]/depth[i]/(relative_density*gravity[i]*d50[i]),exponent_six_fifths);
    const double suspended_coefficient=(double)0.012f*d50[i]*
        pow(dstar,-exponent_three_fifths)/
        pow(relative_density*gravity[i]*d50[i],exponent_six_fifths);
    const double excess_power=pow(effective_speed-critical_speed,exponent_twelve_fifths);
    output[i]=calibration[i]*bed_coefficient*u[i]*excess_power*solid_density[i];
    output[count+i]=calibration[i]*bed_coefficient*v[i]*excess_power*solid_density[i];
    output[2*count+i]=calibration[i]*suspended_coefficient*u[i]*excess_power*
        solid_density[i];
    output[3*count+i]=calibration[i]*suspended_coefficient*v[i]*excess_power*
        solid_density[i];
    valid=dc_finite7(i,count,output);
    status[i]=valid ? 0 : 1;
    if(!valid) dc_zero(i,count,output);
}

extern "C" __global__ void delft_wang_fredsoe_capacity(
    const long long count,const double *depth,const double *u,const double *v,
    const double *d50,const double *d90,const double *solid_density,
    const double *water_density,const double *gravity,
    const double *kinematic_viscosity,const double *chezy,
    const double *current_shear_velocity,const double *calibration,
    double *output,int *status)
{
    const long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=count) return;
    dc_zero(i,count,output);
    bool valid=isfinite(depth[i]) && isfinite(u[i]) && isfinite(v[i]) &&
        isfinite(d50[i]) && isfinite(d90[i]) && isfinite(solid_density[i]) &&
        isfinite(water_density[i]) && isfinite(gravity[i]) &&
        isfinite(kinematic_viscosity[i]) && isfinite(chezy[i]) &&
        isfinite(current_shear_velocity[i]) && isfinite(calibration[i]);
    valid=valid && depth[i]>0.0 && d50[i]>0.0 && d90[i]>0.0 &&
        solid_density[i]>water_density[i] && water_density[i]>0.0 && gravity[i]>0.0 &&
        kinematic_viscosity[i]>0.0 && chezy[i]>=0.0 &&
        current_shear_velocity[i]>=0.0 && calibration[i]>=0.0;
    if(!valid) { status[i]=1; return; }

    const double speed=sqrt(u[i]*u[i]+v[i]*v[i]);
    if(!isfinite(speed)) { status[i]=1; return; }
    if(speed==0.0 || chezy[i]<=0.0) { status[i]=0; return; }
    const double relative_density=solid_density[i]/water_density[i]-1.0;
    const double total_volume=(double)0.05f*speed*speed*pow(current_shear_velocity[i],3.0)/
        d50[i]/pow(gravity[i]*relative_density,2.0);
    const double cgrain=(double)6.0f+(double)2.5f*
        log(depth[i]/((double)2.5f*d90[i]));
    if(cgrain==0.0) { status[i]=1; return; }
    const double shields=gravity[i]*pow(speed/cgrain,2.0)/relative_density/
        gravity[i]/d50[i];
    const double dstar=d50[i]*pow(relative_density*gravity[i]/
        (kinematic_viscosity[i]*kinematic_viscosity[i]),(double)0.33333333f);
    const double critical_shields=dc_shields_curve(dstar);
    const double excess=shields-critical_shields;
    double bed_volume=0.0;
    if(excess>0.0)
        bed_volume=calibration[i]*(double)4.8695f*d50[i]*current_shear_velocity[i]/
            pow((double)1.0f+pow((double)0.2668f/excess,4.0),(double)0.25f)*
            ((double)1.0f-(double)0.7f*sqrt(critical_shields/shields));
    const double suspended_volume=total_volume-bed_volume;
    output[i]=solid_density[i]*bed_volume*u[i]/speed;
    output[count+i]=solid_density[i]*bed_volume*v[i]/speed;
    output[2*count+i]=solid_density[i]*suspended_volume*u[i]/speed;
    output[3*count+i]=solid_density[i]*suspended_volume*v[i]/speed;
    output[4*count+i]=shields;
    output[5*count+i]=critical_shields;
    output[6*count+i]=solid_density[i]*total_volume;
    valid=dc_finite7(i,count,output);
    status[i]=valid ? 0 : 1;
    if(!valid) dc_zero(i,count,output);
}
