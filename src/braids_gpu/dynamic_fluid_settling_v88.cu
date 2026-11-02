// Q4/v88 resident Delft3D4 FLOW 2DH fluid properties and SettleFrm=2 exchange.
// One cell owns all fractions.  The EOS algebra follows dens_eck,
// dens_unes and dens_nacl; viscosity follows eqsettle exactly.

extern "C" __global__ void v88_fluid_properties(
    const int nc, const double *temperature_c, const double *salinity_ppt,
    const int eos_code, const int constituents_active,
    const double reference_water_density,
    double *water_density, double *molecular_viscosity, int *status)
{
    const int cell=blockIdx.x*blockDim.x+threadIdx.x;
    if(cell>=nc) return;
    const double t=temperature_c[cell], s=salinity_ppt[cell];
    bool bad=!isfinite(t) || t<=-20.0 || !isfinite(s) || s<0.0 ||
             !isfinite(reference_water_density) ||
             reference_water_density<=0.0 || eos_code<0 || eos_code>2 ||
             (constituents_active!=0 && constituents_active!=1);
    double rho=reference_water_density;
    if(!bad && constituents_active) {
        if(eos_code==0) {
            const double cp0=5890.0+38.0*t-0.3750*t*t;
            const double clam=1779.5+11.25*t-0.0745*t*t;
            const double clam0=3.8+0.01*t;
            const double cp1=cp0+3.0*s;
            rho=1000.0*cp1/(0.698*cp1+clam-clam0*s);
        } else if(eos_code==1) {
            const double t2=t*t, t3=t2*t, t4=t3*t, t5=t4*t;
            const double sq=sqrt(fmax(0.0,s));
            const double rhwa=999.842594+6.793952e-2*t-
                9.095290e-3*t2+1.001685e-4*t3-
                1.120083e-6*t4+6.536332e-9*t5;
            const double asal=8.24493e-1-4.0899e-3*t+
                7.6438e-5*t2-8.2467e-7*t3+5.3875e-9*t4;
            const double bsal=-5.72466e-3+1.0227e-4*t-1.6546e-6*t2;
            rho=rhwa+(asal+bsal*sq+4.8314e-4*s)*s;
        } else {
            const double t2=t*t, t3=t2*t;
            rho=999.904+4.8292e-2*t-7.2312e-3*t2+
                2.9963e-5*t3+7.6427e-1*s-3.1490e-3*s*t+
                3.1273e-5*s*t2;
        }
    }
    const double nu=4.0e-5/(20.0+t);
    bad=bad || !isfinite(rho) || rho<=0.0 || !isfinite(nu) || nu<=0.0;
    water_density[cell]=bad ? 0.0 : rho;
    molecular_viscosity[cell]=bad ? 0.0 : nu;
    status[cell]=bad ? 1 : 0;
}

extern "C" __global__ void v88_dynamic_fluid_exchange(
    const int nc, const int ng, const double *depth_before,
    const double *depth_after, const double *endpoint_momentum,
    const double *suspended, const double *active_bed,
    const double *equilibrium_capacity, const double *surface_fraction,
    const double *reference_total_shear, const int *wet_mask,
    const double *sediment_d50, const double *settling_fac_dss,
    const double *sediment_density, const double *sediment_threshold,
    const double *water_density, const double *temperature_c,
    const double reference_water_density,
    const double settling_reference_concentration,
    const double capacity_speed_epsilon, const double gravity,
    const double dry, const double factsd,
    const double reference_depth_floor, const double dt,
    double *next_suspended, double *next_active,
    double *equilibrium_concentration, double *adaptation_velocity,
    double *dimensionless_timescale, double *clear_settling_velocity,
    double *settling_velocity, double *hindered_factor, double *delta,
    double *limiter_correction, int *sediment_active, int *status)
{
    const int cell=blockIdx.x*blockDim.x+threadIdx.x;
    if(cell>=nc) return;
    const double h0=depth_before[cell], h1=depth_after[cell];
    const double hu=endpoint_momentum[cell], hv=endpoint_momentum[nc+cell];
    const double rho=water_density[cell], temperature=temperature_c[cell];
    const bool endpoint_wet=wet_mask[cell]!=0;
    const bool exchange_active=
        h1*(endpoint_wet ? 1.0 : 0.0)>sediment_threshold[0];
    const double safe_h0=exchange_active ? fmax(reference_depth_floor,h0) : 1.0;
    const double safe_h1=exchange_active ? fmax(reference_depth_floor,h1) : 1.0;
    const double speed=endpoint_wet ? hypot(hu,hv)/h1 : 0.0;
    // The frozen v78 handoff constructs tau with its reference rho. Recover
    // the authenticated upstream shear velocity; local rho belongs only to
    // the settling relative-density term in this bounded component.
    const double tau=reference_total_shear[cell];
    const double shear=(isfinite(tau) && tau>=0.0 &&
                        reference_water_density>0.0) ?
        sqrt(tau/reference_water_density) : nan("");
    bool bad=!isfinite(h0) || !isfinite(h1) || !isfinite(hu) ||
        !isfinite(hv) || !isfinite(speed) || !isfinite(shear) ||
        h0<0.0 || h1<0.0 || !isfinite(rho) || rho<=0.0 ||
        !isfinite(reference_water_density) || reference_water_density<=0.0 ||
        endpoint_wet!=(h1>dry) || !isfinite(temperature) ||
        temperature<=-20.0 ||
        !isfinite(settling_reference_concentration) ||
        settling_reference_concentration<=0.0 ||
        !isfinite(capacity_speed_epsilon) || capacity_speed_epsilon<=0.0 ||
        !isfinite(gravity) || gravity<=0.0 || !isfinite(dry) || dry<0.0 ||
        !isfinite(factsd) || factsd<=0.0 ||
        !isfinite(reference_depth_floor) || reference_depth_floor<=0.0 ||
        !isfinite(sediment_threshold[0]) || sediment_threshold[0]<0.0 ||
        !isfinite(dt) || dt<=0.0;
    sediment_active[cell]=exchange_active ? 1 : 0;
    for(int grain=0;grain<ng;++grain) {
        const int point=grain*nc+cell;
        if(!isfinite(suspended[point]) || !isfinite(active_bed[point]) ||
           suspended[point]<0.0 || active_bed[point]<0.0 ||
           !isfinite(equilibrium_capacity[point]) ||
           !isfinite(surface_fraction[point]) ||
           surface_fraction[point]<0.0 || surface_fraction[point]>1.0 ||
           !isfinite(sediment_d50[grain]) || sediment_d50[grain]<=0.0 ||
           !isfinite(settling_fac_dss[grain]) ||
           settling_fac_dss[grain]<=0.0 ||
           !isfinite(sediment_density[grain]) ||
           sediment_density[grain]<=rho) bad=true;
    }
    if(bad) {
        status[cell]=1;
        for(int grain=0;grain<ng;++grain) {
            const int point=grain*nc+cell;
            next_suspended[point]=suspended[point];
            next_active[point]=active_bed[point];
            equilibrium_concentration[point]=0.0;
            adaptation_velocity[point]=0.0;
            dimensionless_timescale[point]=0.0;
            clear_settling_velocity[point]=0.0;
            settling_velocity[point]=0.0;
            delta[point]=0.0;
            limiter_correction[point]=0.0;
        }
        hindered_factor[cell]=0.0;
        return;
    }
    double total_concentration=0.0;
    if(endpoint_wet)
        for(int grain=0;grain<ng;++grain)
            total_concentration+=suspended[grain*nc+cell]/h1;
    const double hindered=endpoint_wet ?
        pow(fmax(0.0,1.0-fmax(0.0,total_concentration)/
                 settling_reference_concentration),5.0) : 0.0;
    const double viscosity=4.0e-5/(20.0+temperature);
    hindered_factor[cell]=hindered;
    for(int grain=0;grain<ng;++grain) {
        const int point=grain*nc+cell;
        const double old_s=suspended[point], old_b=active_bed[point];
        const double total=old_s+old_b;
        const double equilibrium=
            equilibrium_capacity[point]*surface_fraction[point];
        const double dss=sediment_d50[grain]*settling_fac_dss[grain];
        const double relative_density=sediment_density[grain]/rho;
        double clear_ws;
        if(dss<1.5*64.0e-6) {
            clear_ws=(relative_density-1.0)*gravity*dss*dss/
                (18.0*viscosity);
        } else if(dss<0.5*2.0e-3) {
            const double coefficient=dss<2.0*64.0e-6 ?
                -2.9912*dss/(64.0e-6)+15.9824 : 10.0;
            clear_ws=coefficient*viscosity/dss*
                (sqrt(1.0+(relative_density-1.0)*gravity*dss*dss*dss/
                 (100.0*viscosity*viscosity))-1.0);
        } else {
            clear_ws=1.1*sqrt((relative_density-1.0)*gravity*dss);
        }
        const double ws=endpoint_wet ? clear_ws*hindered : 0.0;
        const double wsl=fmax(1.0e-3,ws), eps=capacity_speed_epsilon;
        double timescale=0.0, hots=0.0;
        if(exchange_active && speed>eps && shear>eps) {
            const double u=fmin(shear/speed,0.15);
            const double w=shear>wsl ? wsl/shear : 1.0;
            const double x=w, x2=x*x, x3=x2*x;
            const double exponent=
                (((double)1.547f-(double)20.12f*u)*x3)+
                (((double)326.832f*pow(u,(double)2.2047f)-(double)0.2f)*x2)+
                (((double)0.1385f*log(u)-(double)6.4061f)*x)+
                ((double)0.5467f*u+(double)2.1963f);
            timescale=x*exp(exponent);
            hots=wsl/(timescale*factsd);
        }
        const double explicit_source=(exchange_active && speed>eps && shear>eps) ?
            equilibrium*hots/safe_h0 : 0.0;
        const double total_sink=!exchange_active ? 0.0 :
            ((speed>eps && shear>eps) ? hots/safe_h1 : wsl/safe_h1);
        const double unbounded=exchange_active ?
            (old_s+dt*h0*explicit_source)/(1.0+dt*total_sink) : old_s;
        const double candidate=fmin(fmax(unbounded,0.0),total);
        const double bed_after=total-candidate;
        if(!isfinite(equilibrium) || !isfinite(timescale) ||
           !isfinite(hots) || !isfinite(total_sink) || !isfinite(candidate) ||
           !isfinite(bed_after) || candidate<0.0 || bed_after<0.0 ||
           !isfinite(unbounded) || !isfinite(clear_ws) || clear_ws<0.0 ||
           !isfinite(ws) || ws<0.0 ||
           ((exchange_active && speed>eps && shear>eps) && timescale<=0.0)) {
            bad=true;
            next_suspended[point]=old_s;
            next_active[point]=old_b;
            equilibrium_concentration[point]=0.0;
            adaptation_velocity[point]=0.0;
            dimensionless_timescale[point]=0.0;
            clear_settling_velocity[point]=0.0;
            settling_velocity[point]=0.0;
            delta[point]=0.0;
            limiter_correction[point]=0.0;
        } else {
            next_suspended[point]=candidate;
            next_active[point]=bed_after;
            equilibrium_concentration[point]=equilibrium;
            adaptation_velocity[point]=exchange_active ? hots : 0.0;
            dimensionless_timescale[point]=exchange_active ? timescale : 0.0;
            clear_settling_velocity[point]=clear_ws;
            settling_velocity[point]=ws;
            delta[point]=candidate-old_s;
            limiter_correction[point]=fabs(candidate-unbounded);
        }
    }
    status[cell]=bad ? 1 : 0;
}
