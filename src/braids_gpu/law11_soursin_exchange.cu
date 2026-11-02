// Resident law-11 FLOW 2DH Galappatti exchange.
//
// The algebra follows locked eqtran.f90/soursin_2d.f90 source order.  FLOW's
// default-real literals are explicitly rounded to binary32 before promotion.
// One thread owns a cell and all fractions, so the finite-bed cap is local and
// every water gain has the exact opposite active-bed change.  The 0.01 m
// h0/h1 values are reference depths only: the assembled source numerator
// deliberately retains actual h0, including when h0 is below that floor.

extern "C" __global__ void law11_soursin_2d_exchange(
    const int nc, const int ng, const double *depth_before,
    const double *depth_after, const double *endpoint_momentum,
    const double *suspended, const double *active_bed,
    const double *equilibrium_capacity, const double *surface_fraction,
    const double *total_shear, const int *wet_mask,
    const double *sediment_d50, const double *settling_fac_dss,
    const double *sediment_density, const double *sediment_threshold,
    const double water_density, const double temperature_c,
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
    const double hu=endpoint_momentum[cell];
    const double hv=endpoint_momentum[nc+cell];
    const bool endpoint_wet=wet_mask[cell]!=0;
    const bool exchange_active=
        h1*(endpoint_wet ? 1.0 : 0.0)>sediment_threshold[0];
    const double safe_h0=exchange_active ? fmax(reference_depth_floor,h0) : 1.0;
    const double safe_h1=exchange_active ? fmax(reference_depth_floor,h1) : 1.0;
    const double speed=endpoint_wet ? hypot(hu,hv)/h1 : 0.0;
    const double tau=total_shear[cell];
    const double shear=(isfinite(tau) && tau>=0.0 && water_density>0.0) ?
        sqrt(tau/water_density) : nan("");
    bool bad=!isfinite(h0) || !isfinite(h1) || !isfinite(hu) ||
        !isfinite(hv) || !isfinite(speed) || !isfinite(shear) ||
        h0<0.0 || h1<0.0 || !isfinite(water_density) || water_density<=0.0 ||
        endpoint_wet!=(h1>dry) || !isfinite(temperature_c) ||
        temperature_c<=-20.0 ||
        !isfinite(settling_reference_concentration) ||
        settling_reference_concentration<=0.0 ||
        !isfinite(capacity_speed_epsilon) ||
        capacity_speed_epsilon<=0.0 ||
        !isfinite(gravity) || gravity<=0.0 ||
        !isfinite(dry) || dry<0.0 || !isfinite(factsd) || factsd<=0.0 ||
        !isfinite(reference_depth_floor) || reference_depth_floor<=0.0 ||
        !isfinite(sediment_threshold[0]) || sediment_threshold[0]<0.0 ||
        !isfinite(dt) || dt<=0.0;
    sediment_active[cell]=exchange_active ? 1 : 0;

    for(int grain=0;grain<ng;++grain) {
        const int point=grain*nc+cell;
        const double old_s=suspended[point];
        const double old_b=active_bed[point];
        const double equilibrium=equilibrium_capacity[point];
        const double fraction=surface_fraction[point];
        if(!isfinite(old_s) || !isfinite(old_b) || old_s<0.0 || old_b<0.0 ||
           !isfinite(equilibrium) || !isfinite(fraction) ||
           fraction<0.0 || fraction>1.0 ||
           !isfinite(sediment_d50[grain]) || sediment_d50[grain]<=0.0 ||
           !isfinite(settling_fac_dss[grain]) ||
           settling_fac_dss[grain]<=0.0 ||
           !isfinite(sediment_density[grain]) ||
           sediment_density[grain]<=water_density)
            bad=true;
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

    const double eps=capacity_speed_epsilon;
    double total_concentration=0.0;
    if(endpoint_wet) {
        for(int grain=0;grain<ng;++grain)
            total_concentration+=suspended[grain*nc+cell]/h1;
    }
    const double hindered=endpoint_wet ?
        pow(fmax(0.0,1.0-fmax(0.0,total_concentration)/
                 settling_reference_concentration),5.0) : 0.0;
    const double viscosity=4.0e-5/(20.0+temperature_c);
    hindered_factor[cell]=hindered;
    for(int grain=0;grain<ng;++grain) {
        const int point=grain*nc+cell;
        const double old_s=suspended[point];
        const double old_b=active_bed[point];
        const double total=old_s+old_b;
        const double fraction=surface_fraction[point];
        // The common post-density composer has already applied global Sus,
        // retained the sign and divided by (speed+eps)*h1.  Active surface
        // fraction belongs here and is applied exactly once.
        const double equilibrium=equilibrium_capacity[point]*fraction;
        const double dss=sediment_d50[grain]*settling_fac_dss[grain];
        const double relative_density=sediment_density[grain]/water_density;
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
        const double wsl=fmax(1.0e-3,ws);
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
