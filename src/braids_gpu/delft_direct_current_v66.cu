// Isolated resident-CUDA foundations for Delft3D4 FLOW TraFrm 5, 19 and 20.
//
// This is the current-only classic-FLOW subset.  Wave inputs and classic FLOW
// state fields that are zero at the locked call site are explicit and rejected
// when nonzero.  Output rows are bed x/y mass, explicit suspended x/y mass,
// concentration mass, and three formula diagnostics.  TraFrm 19/20 leave the
// explicit suspended rows zero because their public suspended output is an
// equilibrium concentration.

enum DirectStatus {
    DIRECT_OK = 0,
    DIRECT_INVALID = 1,
    DIRECT_WAVE_UNSUPPORTED = 2,
    DIRECT_CLASSIC_STATE_UNSUPPORTED = 3,
    DIRECT_OPTION_UNSUPPORTED = 4
};

__device__ __forceinline__ void dd_zero(
    const long long i, const long long count, double *output)
{
    #pragma unroll
    for(int row=0; row<8; ++row) output[(long long)row*count+i]=0.0;
}

__device__ __forceinline__ bool dd_finite8(
    const long long i, const long long count, const double *output)
{
    #pragma unroll
    for(int row=0; row<8; ++row)
        if(!isfinite(output[(long long)row*count+i])) return false;
    return true;
}

__device__ __forceinline__ bool dd_wave_active(
    const double height, const double period, const double wavelength,
    const double angle, const double orbital, const double breaker)
{
    return height!=0.0 || period!=0.0 || wavelength!=0.0 || angle!=0.0 ||
           orbital!=0.0 || breaker!=0.0;
}

__device__ __forceinline__ double dd_bijker_term_f(
    const int n, const double y, const double z)
{
    const double sign=((n+1)%2==0) ? 1.0 : -1.0;
    if(fabs((double)n-z)<=1.0e-10) return sign*z*log(y);
    if(n==1) return pow(y,1.0-z)/(1.0-z);
    double numerator=1.0;
    double denominator=1.0;
    for(int k=2;k<=n;++k) {
        numerator*=z-(double)k+2.0;
        denominator*=(double)k;
    }
    denominator/=(double)n;
    return sign*numerator*pow(y,(double)n-z)/(denominator*((double)n-z));
}

__device__ __forceinline__ double dd_bijker_term_g(
    const int n, const double y, const double z)
{
    const double sign=((n+1)%2==0) ? 1.0 : -1.0;
    if(fabs((double)n-z)<=1.0e-10) {
        const double ly=log(y);
        return sign*z*ly*ly*0.5;
    }
    if(n==1) {
        const double exponent=1.0-z;
        return pow(y,exponent)*(exponent*log(y)-1.0)/(exponent*exponent);
    }
    double numerator=1.0;
    double denominator=1.0;
    for(int k=2;k<=n;++k) {
        numerator*=z-(double)k+2.0;
        denominator*=(double)k;
    }
    denominator/=(double)n;
    const double exponent=(double)n-z;
    return sign*numerator*pow(y,exponent)/denominator*
           (exponent*log(y)-1.0)/(exponent*exponent);
}

__device__ __forceinline__ double dd_bijker_integral(
    const double a, const double b, const double z, const bool use_g)
{
    double sum=0.0;
    for(int n=1;n<=100;++n) {
        const double rest=use_g ?
            dd_bijker_term_g(n,b,z)-dd_bijker_term_g(n,a,z) :
            dd_bijker_term_f(n,b,z)-dd_bijker_term_f(n,a,z);
        sum+=rest;
        if(rest==0.0 || sum==0.0) break;
        if(fabs(rest/sum)<0.001) break;
    }
    return sum;
}

extern "C" __global__ void delft_bijker_current_v66(
    const long long count,
    const double *depth,const double *u,const double *v,
    const double *d50,const double *d90,
    const double *solid_density,const double *water_density,
    const double *gravity,const double *chezy,const double *settling_velocity,
    const double *cal_bs,const double *cal_bd,
    const double *crit_cs,const double *crit_cd,
    const double *roughness_ks,const double *formula_settling_velocity,
    const double *porosity,const double *formula_wave_period,
    const double *fac_a,const double *fac_u,const double *eps_sl,
    const double *wave_height,const double *wave_period,
    const double *wavelength,const double *wave_angle,
    const double *orbital_velocity,const double *breaker_turbulence,
    const double *bed_slope_x,const double *bed_slope_y,
    double *output,int *status)
{
    const long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=count) return;
    dd_zero(i,count,output);
    bool valid=isfinite(depth[i]) && isfinite(u[i]) && isfinite(v[i]) &&
        isfinite(d50[i]) && isfinite(d90[i]) && isfinite(solid_density[i]) &&
        isfinite(water_density[i]) && isfinite(gravity[i]) && isfinite(chezy[i]) &&
        isfinite(settling_velocity[i]) && isfinite(cal_bs[i]) && isfinite(cal_bd[i]) &&
        isfinite(crit_cs[i]) && isfinite(crit_cd[i]) && isfinite(roughness_ks[i]) &&
        isfinite(formula_settling_velocity[i]) && isfinite(porosity[i]) &&
        isfinite(formula_wave_period[i]) && isfinite(fac_a[i]) && isfinite(fac_u[i]) &&
        isfinite(eps_sl[i]) && isfinite(wave_height[i]) && isfinite(wave_period[i]) &&
        isfinite(wavelength[i]) && isfinite(wave_angle[i]) &&
        isfinite(orbital_velocity[i]) && isfinite(breaker_turbulence[i]) &&
        isfinite(bed_slope_x[i]) && isfinite(bed_slope_y[i]);
    if(!valid) { status[i]=DIRECT_INVALID; return; }
    if(dd_wave_active(wave_height[i],wave_period[i],wavelength[i],wave_angle[i],
                      orbital_velocity[i],breaker_turbulence[i])) {
        status[i]=DIRECT_WAVE_UNSUPPORTED; return;
    }
    if(bed_slope_x[i]!=0.0 || bed_slope_y[i]!=0.0) {
        status[i]=DIRECT_CLASSIC_STATE_UNSUPPORTED; return;
    }
    if(eps_sl[i]>0.0) { status[i]=DIRECT_OPTION_UNSUPPORTED; return; }
    valid=depth[i]>=0.0 && d50[i]>0.0 && d90[i]>0.0 &&
        solid_density[i]>water_density[i] && water_density[i]>0.0 &&
        gravity[i]>0.0 && chezy[i]>0.0 && settling_velocity[i]>=0.0 &&
        porosity[i]>=0.0 && porosity[i]<1.0 && formula_wave_period[i]>0.0;
    if(!valid) { status[i]=DIRECT_INVALID; return; }
    status[i]=DIRECT_OK;
    if(depth[i]==0.0) return;

    double roughness=roughness_ks[i];
    if(roughness<0.0) roughness=12.0*depth[i]*pow(10.0,-chezy[i]/18.0);
    double fall=formula_settling_velocity[i];
    if(fall<0.0) fall=settling_velocity[i];
    if(!isfinite(roughness) || !isfinite(fall) || roughness<=0.0 || fall<0.0) {
        status[i]=DIRECT_INVALID; return;
    }
    if(depth[i]/roughness<=1.33 || depth[i]>200.0) return;
    const double speed=sqrt(u[i]*u[i]+v[i]*v[i]);
    if(!isfinite(speed)) { status[i]=DIRECT_INVALID; return; }
    const double delta=solid_density[i]/water_density[i]-1.0;
    const double relhrms=0.0;
    double bfactor;
    if(crit_cd[i]<relhrms && relhrms<crit_cs[i]) {
        const double factor=(cal_bs[i]-cal_bd[i])/(crit_cs[i]-crit_cd[i]);
        bfactor=factor*relhrms+cal_bd[i]-factor*crit_cd[i];
    } else if(relhrms<=crit_cd[i]) bfactor=cal_bd[i];
    else bfactor=cal_bs[i];
    const double c90=18.0*log10(12.0*depth[i]/d90[i]);
    double rmu=chezy[i]/c90;
    if(!isfinite(rmu) || rmu<0.0) { status[i]=DIRECT_INVALID; return; }
    rmu*=sqrt(rmu);
    double current_sq=0.0;
    if(speed>1.0e-10) current_sq=speed*speed;
    double exponent_arg;
    double z;
    if(current_sq>1.0e-20) {
        exponent_arg=-0.27*delta*d50[i]*chezy[i]*chezy[i]/(rmu*current_sq);
        exponent_arg=fmax(-50.0,fmin(50.0,exponent_arg));
        const double shear_velocity=sqrt(gravity[i]/(chezy[i]*chezy[i]))*
                                    sqrt(current_sq);
        // FLOW initializes von Karman from an unsuffixed .41 literal.
        z=fmin(fall/(double)0.41f/shear_velocity,8.0);
    } else {
        exponent_arg=-50.0;
        z=8.0;
    }
    const double bed_volume_scale=bfactor*d50[i]/chezy[i]*sqrt(gravity[i])*
        exp(exponent_arg)*(1.0-porosity[i]);
    const double rkh=roughness/depth[i];
    const double common=0.216*pow(rkh,z-1.0)/pow(1.0-rkh,z);
    const double integral_f=common*dd_bijker_integral(rkh,1.0,z,false);
    const double integral_g=common*dd_bijker_integral(rkh,1.0,z,true);
    const double concentration_volume=1.83*bed_volume_scale*
        (integral_f*log(33.0/rkh)+integral_g);
    const double concentration_mass=solid_density[i]*concentration_volume;
    output[4*count+i]=concentration_mass;
    output[5*count+i]=z;
    output[6*count+i]=solid_density[i]*bed_volume_scale;
    output[7*count+i]=speed;
    if(speed>1.0e-10) {
        output[i]=solid_density[i]*bed_volume_scale*u[i];
        output[count+i]=solid_density[i]*bed_volume_scale*v[i];
        output[2*count+i]=concentration_mass*u[i];
        output[3*count+i]=concentration_mass*v[i];
    }
    valid=dd_finite8(i,count,output);
    status[i]=valid ? DIRECT_OK : DIRECT_INVALID;
    if(!valid) dd_zero(i,count,output);
}

__device__ __forceinline__ void dd_formula_1920(
    const bool xbeach,const long long i,const long long count,
    const double *depth,const double *u,const double *v,
    const double *d15,const double *d50,const double *d90,
    const double *solid_density,const double *water_density,
    const double *gravity,const double *kinematic_viscosity,const double *chezy,
    const double *facua,const double *facas,const double *facsk,
    const int *waveform,const int *sws,const int *lws,const int *dilatancy,
    const double *rheea,const double *pormax,const int *bedslpini,
    const double *smax,const double *reposeangle,const double *cmax,
    const double *z0_input,const double *alfad50,
    const double *wave_height,const double *wave_period,
    const double *wavelength,const double *wave_angle,
    const double *orbital_velocity,const double *breaker_turbulence,
    const double *bed_change_rate,const double *bed_slope_x,const double *bed_slope_y,
    double *output,int *status)
{
    dd_zero(i,count,output);
    bool valid=isfinite(depth[i]) && isfinite(u[i]) && isfinite(v[i]) &&
        isfinite(d15[i]) && isfinite(d50[i]) && isfinite(d90[i]) &&
        isfinite(solid_density[i]) && isfinite(water_density[i]) &&
        isfinite(gravity[i]) && isfinite(kinematic_viscosity[i]) &&
        isfinite(chezy[i]) && isfinite(facua[i]) && isfinite(facas[i]) &&
        isfinite(facsk[i]) && isfinite(rheea[i]) && isfinite(pormax[i]) &&
        isfinite(smax[i]) && isfinite(reposeangle[i]) && isfinite(cmax[i]) &&
        isfinite(z0_input[i]) && isfinite(alfad50[i]) &&
        isfinite(wave_height[i]) && isfinite(wave_period[i]) &&
        isfinite(wavelength[i]) && isfinite(wave_angle[i]) &&
        isfinite(orbital_velocity[i]) && isfinite(breaker_turbulence[i]) &&
        isfinite(bed_change_rate[i]) && isfinite(bed_slope_x[i]) &&
        isfinite(bed_slope_y[i]);
    if(!valid) { status[i]=DIRECT_INVALID; return; }
    if(dd_wave_active(wave_height[i],wave_period[i],wavelength[i],wave_angle[i],
                      orbital_velocity[i],breaker_turbulence[i])) {
        status[i]=DIRECT_WAVE_UNSUPPORTED; return;
    }
    if(bed_change_rate[i]!=0.0 || bed_slope_x[i]!=0.0 || bed_slope_y[i]!=0.0) {
        status[i]=DIRECT_CLASSIC_STATE_UNSUPPORTED; return;
    }
    if(dilatancy[i]==1 || bedslpini[i]==1 || bedslpini[i]==2) {
        status[i]=DIRECT_OPTION_UNSUPPORTED; return;
    }
    valid=depth[i]>=0.0 && d15[i]>=0.0 && d50[i]>0.0 && d90[i]>0.0 &&
        solid_density[i]>water_density[i] && water_density[i]>0.0 &&
        gravity[i]>0.0 && kinematic_viscosity[i]>0.0 && chezy[i]>0.0;
    if(!valid) { status[i]=DIRECT_INVALID; return; }
    const double speed=sqrt(u[i]*u[i]+v[i]*v[i]);
    if(!isfinite(speed)) { status[i]=DIRECT_INVALID; return; }
    status[i]=DIRECT_OK;
    if(speed<1.0e-6 || depth[i]>200.0 || depth[i]<0.01) return;

    // waveform/sws/lws and facua/facas/facsk are retained in the typed ABI.
    // Their wave terms are exactly zero in this admitted current-only subset.
    (void)facua;(void)facas;(void)facsk;(void)waveform;(void)sws;(void)lws;
    (void)rheea;(void)pormax;(void)reposeangle;(void)d15;(void)kinematic_viscosity;
    const double delta=solid_density[i]/water_density[i]-1.0;
    const double dstar=pow(delta*gravity[i]/1.0e-12,1.0/3.0)*d50[i];
    double current_critical;
    if(d50[i]<=0.0005) {
        current_critical=0.19*pow(d50[i],0.1)*log10(4.0*depth[i]/d90[i]);
    } else if(d50[i]<=0.002) {
        current_critical=8.5*pow(d50[i],0.6)*log10(4.0*depth[i]/d90[i]);
    } else if(xbeach) {
        current_critical=8.5*pow(d50[i],0.6)*log10(4.0*depth[i]/d90[i]);
    } else {
        current_critical=1.3*sqrt(delta*gravity[i]*d50[i])*
                         pow(depth[i]/d50[i],0.5/3.0);
    }
    double critical=current_critical;
    if(!xbeach) {
        const double b2=speed/fmax(speed,5.0e-3);
        critical=b2*current_critical; // zero-period current-only Ucrw is zero.
    }
    const double cf=gravity[i]/(chezy[i]*chezy[i]);
    double smax_value=fmax(-1.0,fmin(3.0,smax[i]));
    if(smax_value<0.0) smax_value=1.7976931348623157e308*1.0e-20;
    const double cmax_value=fmax(0.0,fmin(1.0,cmax[i]));
    const double alfa=fmax(0.0,fmin(1.5,alfad50[i]));
    double asb,ass,effective_sq;
    if(xbeach) {
        const double z0=fmax(0.0001,fmin(0.05,z0_input[i]));
        const double drag=pow(0.40/(log(fmax(depth[i],10.0*z0)/z0)-1.0),2.0);
        asb=0.005*depth[i]*pow(
            d50[i]/depth[i]/(delta*gravity[i]*d50[i]),1.2);
        ass=0.012*d50[i]*pow(dstar,-0.6)/
            pow(delta*gravity[i]*d50[i],1.2);
        effective_sq=fmin(speed*speed,
            smax_value*gravity[i]/cf*d50[i]*delta);
        // Retain drag evaluation because locked trab20 validates/uses it in
        // its wave coefficient even though urms2 is zero here.
        if(!isfinite(drag)) { status[i]=DIRECT_INVALID; return; }
    } else {
        asb=0.015*depth[i]*pow(d50[i]/depth[i],1.2)/
            pow(delta*gravity[i]*d50[i],0.75);
        ass=0.012*d50[i]*pow(dstar,-0.6)/
            pow(delta*gravity[i]*d50[i],1.2);
        effective_sq=fmin(speed*speed,
            smax_value*gravity[i]/fmax(cf,1.0e-10)*d50[i]*delta);
    }
    const double effective=sqrt(effective_sq);
    double bed_eq=0.0;
    double sus_eq=0.0;
    if(effective>critical) {
        bed_eq=asb*pow(effective-critical,xbeach ? 2.4 : 1.5);
        sus_eq=ass*pow(effective-critical,2.4);
    }
    if(alfa>0.0) {
        const double multiplier=pow(0.000225/d50[i],alfa);
        bed_eq*=multiplier;
        sus_eq*=multiplier;
    }
    const double maximum=cmax_value*depth[i]/2.0;
    bed_eq=fmin(bed_eq,maximum);
    const double concentration_mass=solid_density[i]*fmin(sus_eq,maximum)/depth[i];
    output[i]=solid_density[i]*bed_eq*u[i];
    output[count+i]=solid_density[i]*bed_eq*v[i];
    // rows 2/3 remain zero: this formula owns equilibrium concentration.
    output[4*count+i]=concentration_mass;
    output[5*count+i]=critical;
    output[6*count+i]=critical;
    output[7*count+i]=effective;
    valid=dd_finite8(i,count,output);
    status[i]=valid ? DIRECT_OK : DIRECT_INVALID;
    if(!valid) dd_zero(i,count,output);
}

#define DIRECT_1920_ARGUMENTS \
    const long long count, \
    const double *depth,const double *u,const double *v, \
    const double *d15,const double *d50,const double *d90, \
    const double *solid_density,const double *water_density, \
    const double *gravity,const double *kinematic_viscosity,const double *chezy, \
    const double *facua,const double *facas,const double *facsk, \
    const int *waveform,const int *sws,const int *lws,const int *dilatancy, \
    const double *rheea,const double *pormax,const int *bedslpini, \
    const double *smax,const double *reposeangle,const double *cmax, \
    const double *z0_input,const double *alfad50, \
    const double *wave_height,const double *wave_period, \
    const double *wavelength,const double *wave_angle, \
    const double *orbital_velocity,const double *breaker_turbulence, \
    const double *bed_change_rate,const double *bed_slope_x,const double *bed_slope_y, \
    double *output,int *status

#define DIRECT_1920_VALUES \
    depth,u,v,d15,d50,d90,solid_density,water_density,gravity,kinematic_viscosity,chezy, \
    facua,facas,facsk,waveform,sws,lws,dilatancy,rheea,pormax,bedslpini,smax, \
    reposeangle,cmax,z0_input,alfad50,wave_height,wave_period,wavelength,wave_angle, \
    orbital_velocity,breaker_turbulence,bed_change_rate,bed_slope_x,bed_slope_y, \
    output,status

extern "C" __global__ void delft_van_thiel_van_rijn_current_v66(
    DIRECT_1920_ARGUMENTS)
{
    const long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count) dd_formula_1920(false,i,count,DIRECT_1920_VALUES);
}

extern "C" __global__ void delft_xbeach_soulsby_van_rijn_current_v66(
    DIRECT_1920_ARGUMENTS)
{
    const long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count) dd_formula_1920(true,i,count,DIRECT_1920_VALUES);
}

#undef DIRECT_1920_ARGUMENTS
#undef DIRECT_1920_VALUES
