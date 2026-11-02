// Q4/v92 transported temperature/salinity and parent-timed settling.
//
// Temperature is transported through the accepted nonnegative scalar FCT as
// theta=T-T_floor, T_floor=-20 C.  The caller authenticates discrete water
// continuity before recovering hT=h*theta+T_floor*h.  This affine change of
// variable is exactly conservative and permits signed Celsius hT without
// weakening the positive scalar limiter.  Salinity is transported directly.

extern "C" __global__ void v92_pack_affine_fluid(
    const int nc, const double dry, const double temperature_floor,
    const double *depth, const double *h_temperature,
    const double *h_salinity, double *shifted_mass,
    double *actual_concentration, int *status)
{
    const int cell=blockIdx.x*blockDim.x+threadIdx.x;
    if(cell>=nc) return;
    const double h=depth[cell], ht=h_temperature[cell], hs=h_salinity[cell];
    bool bad=!isfinite(h)||h<0.0||!isfinite(ht)||!isfinite(hs);
    double t=0.0,s=0.0,theta=0.0;
    if(!bad && h>dry) {
        t=ht/h;s=hs/h;theta=t-temperature_floor;
        bad=!isfinite(t)||!isfinite(s)||!isfinite(theta)||
            t<=temperature_floor||s<0.0||theta<=0.0;
    } else if(!bad) {
        const double scale=fmax(1.0,fmax(fabs(ht),fabs(hs)));
        bad=fabs(ht)>4096.0*2.2204460492503131e-16*scale||
            fabs(hs)>4096.0*2.2204460492503131e-16*scale;
        // Moving-wet/dry states retain their conservative sub-tolerance
        // water depth while publishing exact-zero physical hT/hS.  The
        // affine temperature mass is therefore hT-T_floor*h, not zero.
        // Keeping this water-column offset prevents a dry-parent/dry-endpoint
        // update from silently deleting -T_floor*h on every new scalar pass.
        theta=-temperature_floor;
    }
    shifted_mass[cell]=bad?0.0:h*theta;
    shifted_mass[nc+cell]=bad?0.0:hs;
    actual_concentration[cell]=bad?0.0:t;
    actual_concentration[nc+cell]=bad?0.0:s;
    status[cell]=bad?1:0;
}

extern "C" __global__ void v92_shift_fluid_values(
    const int count, const double temperature_floor,
    const double *actual, double *shifted, int *status)
{
    const int point=blockIdx.x*blockDim.x+threadIdx.x;
    if(point>=2*count) return;
    const int scalar=point/count;
    const double value=actual[point];
    const bool bad=!isfinite(value)||
        (scalar==0?value<=temperature_floor:value<0.0);
    shifted[point]=bad?0.0:(scalar==0?value-temperature_floor:value);
    status[point]=bad?1:0;
}

extern "C" __global__ void v92_unshift_boundary_memory(
    const int nf, const double temperature_floor,
    const double *shifted, double *actual, int *status)
{
    const int point=blockIdx.x*blockDim.x+threadIdx.x;
    if(point>=2*nf) return;
    const int scalar=point/nf;
    const double value=shifted[point];
    const double result=scalar==0?value+temperature_floor:value;
    const bool bad=!isfinite(value)||!isfinite(result)||
        (scalar==0?result<=temperature_floor:result<0.0);
    actual[point]=bad?0.0:result;
    status[point]=bad?1:0;
}

extern "C" __global__ void v92_water_continuity(
    const int nc, const double dt, const double *depth_before,
    const double *depth_after, const double *water_flux,
    const double *face_length, const double *cell_area,
    const int *offset, const int *faces, const int *signs,
    double *residual, double *tolerance, int *status)
{
    const int cell=blockIdx.x*blockDim.x+threadIdx.x;
    if(cell>=nc) return;
    double divergence=0.0,absolute_sum=0.0;
    bool bad=!isfinite(dt)||dt<=0.0||!isfinite(depth_before[cell])||
        !isfinite(depth_after[cell])||depth_before[cell]<0.0||
        depth_after[cell]<0.0||!isfinite(cell_area[cell])||
        cell_area[cell]<=0.0;
    for(int slot=offset[cell];slot<offset[cell+1];++slot) {
        const int face=faces[slot];
        const double term=dt*((double)signs[slot])*face_length[face]*
                          water_flux[face]/cell_area[cell];
        if(!isfinite(term)) bad=true;
        divergence+=term;absolute_sum+=fabs(term);
    }
    const double value=depth_after[cell]-depth_before[cell]+divergence;
    const double scale=fmax(1.0,fmax(fabs(depth_before[cell]),
                       fmax(fabs(depth_after[cell]),absolute_sum)));
    const double tol=32768.0*2.2204460492503131e-16*scale;
    residual[cell]=value;tolerance[cell]=tol;
    status[cell]=(bad||!isfinite(value)||fabs(value)>tol)?1:0;
}

extern "C" __global__ void v92_unpack_affine_fluid(
    const int nc, const double dry, const double temperature_floor,
    const double equilibrium_temperature, const double equilibrium_salinity,
    const double *depth_after, const double *shifted_mass,
    double *h_temperature, double *h_salinity,
    double *temperature, double *salinity,
    double *wetdry_removed_actual_mass, int *status)
{
    const int cell=blockIdx.x*blockDim.x+threadIdx.x;
    if(cell>=nc) return;
    const double h=depth_after[cell];
    const double theta_mass=shifted_mass[cell];
    const double salt_mass=shifted_mass[nc+cell];
    double ht=theta_mass+temperature_floor*h,hs=salt_mass;
    bool bad=!isfinite(h)||h<0.0||!isfinite(theta_mass)||theta_mass<0.0||
             !isfinite(salt_mass)||salt_mass<0.0;
    if(!bad && h>dry) {
        const double t=ht/h,s=hs/h;
        bad=!isfinite(t)||t<=temperature_floor||!isfinite(s)||s<0.0;
        h_temperature[cell]=bad?0.0:ht;
        h_salinity[cell]=bad?0.0:hs;
        temperature[cell]=bad?0.0:t;
        salinity[cell]=bad?0.0:s;
        wetdry_removed_actual_mass[cell]=0.0;
        wetdry_removed_actual_mass[nc+cell]=0.0;
    } else {
        h_temperature[cell]=0.0;h_salinity[cell]=0.0;
        temperature[cell]=equilibrium_temperature;
        salinity[cell]=equilibrium_salinity;
        wetdry_removed_actual_mass[cell]=bad?0.0:ht;
        wetdry_removed_actual_mass[nc+cell]=bad?0.0:hs;
    }
    status[cell]=bad?1:0;
}

extern "C" __global__ void v92_fluid_properties(
    const int nc, const double *temperature_c, const double *salinity_ppt,
    const int eos_code, const int constituents_active,
    const double reference_water_density,
    double *water_density, double *molecular_viscosity, int *status)
{
    const int cell=blockIdx.x*blockDim.x+threadIdx.x;
    if(cell>=nc) return;
    const double t=temperature_c[cell],s=salinity_ppt[cell];
    bool bad=!isfinite(t)||t<=-20.0||!isfinite(s)||s<0.0||
        !isfinite(reference_water_density)||reference_water_density<=0.0||
        eos_code<0||eos_code>2||
        (constituents_active!=0&&constituents_active!=1);
    double rho=reference_water_density;
    if(!bad&&constituents_active) {
        if(eos_code==0) {
            const double cp0=5890.0+38.0*t-0.3750*t*t;
            const double clam=1779.5+11.25*t-0.0745*t*t;
            const double clam0=3.8+0.01*t;
            const double cp1=cp0+3.0*s;
            rho=1000.0*cp1/(0.698*cp1+clam-clam0*s);
        } else if(eos_code==1) {
            const double t2=t*t,t3=t2*t,t4=t3*t,t5=t4*t;
            const double sq=sqrt(fmax(0.0,s));
            const double rhwa=999.842594+6.793952e-2*t-
                9.095290e-3*t2+1.001685e-4*t3-
                1.120083e-6*t4+6.536332e-9*t5;
            const double asal=8.24493e-1-4.0899e-3*t+
                7.6438e-5*t2-8.2467e-7*t3+5.3875e-9*t4;
            const double bsal=-5.72466e-3+1.0227e-4*t-1.6546e-6*t2;
            rho=rhwa+(asal+bsal*sq+4.8314e-4*s)*s;
        } else {
            const double t2=t*t,t3=t2*t;
            rho=999.904+4.8292e-2*t-7.2312e-3*t2+
                2.9963e-5*t3+7.6427e-1*s-3.1490e-3*s*t+
                3.1273e-5*s*t2;
        }
    }
    const double nu=4.0e-5/(20.0+t);
    bad=bad||!isfinite(rho)||rho<=0.0||!isfinite(nu)||nu<=0.0;
    water_density[cell]=bad?0.0:rho;
    molecular_viscosity[cell]=bad?0.0:nu;
    status[cell]=bad?1:0;
}

extern "C" __global__ void v92_boundary_water_ledger(
    const int nb, const double dt, const int *boundary_offset,
    const int *boundary_faces, const double *water_flux,
    const double *face_length, double *ledger, int *status)
{
    const int boundary=blockIdx.x*blockDim.x+threadIdx.x;
    if(boundary>=nb) return;
    double value=0.0;bool bad=!isfinite(dt)||dt<=0.0;
    for(int slot=boundary_offset[boundary];
        slot<boundary_offset[boundary+1];++slot) {
        const int face=boundary_faces[slot];
        const double term=-dt*face_length[face]*water_flux[face];
        if(!isfinite(term)) bad=true;
        value+=term;
    }
    ledger[boundary]=value;status[boundary]=(bad||!isfinite(value))?1:0;
}

// Compute the current SettleFrm=2 velocities before scalar transport.  This
// is the locked FLOW fallve timing: ctot, rho and temperature are all from
// the accepted parent state.  The subsequent exchange kernel consumes these
// arrays and never recomputes them from transported concentration.
extern "C" __global__ void v92_parent_settling(
    const int nc, const int ng, const double dry,
    const double gravity, const double reference_concentration,
    const double *depth_before, const double *parent_suspended,
    const double *parent_temperature, const double *parent_water_density,
    const double *sediment_d50, const double *settling_fac_dss,
    const double *sediment_density, double *clear_settling_velocity,
    double *settling_velocity, double *hindered_factor, int *status)
{
    const int cell=blockIdx.x*blockDim.x+threadIdx.x;
    if(cell>=nc) return;
    const double h=depth_before[cell],temperature=parent_temperature[cell];
    const double rho=parent_water_density[cell];
    const bool parent_wet=h>dry;
    bool bad=!isfinite(h)||h<0.0||!isfinite(temperature)||temperature<=-20.0||
        !isfinite(rho)||rho<=0.0||!isfinite(dry)||dry<0.0||
        !isfinite(gravity)||gravity<=0.0||
        !isfinite(reference_concentration)||reference_concentration<=0.0;
    double total_concentration=0.0;
    for(int grain=0;grain<ng;++grain) {
        const int point=grain*nc+cell;
        const double mass=parent_suspended[point];
        if(!isfinite(mass)||mass<0.0||!isfinite(sediment_d50[grain])||
           sediment_d50[grain]<=0.0||!isfinite(settling_fac_dss[grain])||
           settling_fac_dss[grain]<=0.0||
           !isfinite(sediment_density[grain])||
           sediment_density[grain]<=rho) bad=true;
        if(parent_wet) total_concentration+=mass/h;
    }
    // Parent concentration is lagged, but hydraulic wetness belongs to the
    // accepted endpoint and is applied by the caller after this preparation.
    // A dry parent can rewet during the implicit solve; its exact-zero parent
    // concentration therefore has hindered factor one and a valid clear-water
    // settling speed, which the endpoint gate either retains or zeros.
    const double hindered=
        pow(fmax(0.0,1.0-fmax(0.0,total_concentration)/
                 reference_concentration),5.0);
    const double viscosity=4.0e-5/(20.0+temperature);
    hindered_factor[cell]=bad?0.0:hindered;
    for(int grain=0;grain<ng;++grain) {
        const int point=grain*nc+cell;
        const double dss=sediment_d50[grain]*settling_fac_dss[grain];
        const double relative_density=sediment_density[grain]/rho;
        double clear_ws=0.0;
        if(!bad) {
            if(dss<1.5*64.0e-6)
                clear_ws=(relative_density-1.0)*gravity*dss*dss/
                         (18.0*viscosity);
            else if(dss<0.5*2.0e-3) {
                const double coefficient=dss<2.0*64.0e-6?
                    -2.9912*dss/(64.0e-6)+15.9824:10.0;
                clear_ws=coefficient*viscosity/dss*
                    (sqrt(1.0+(relative_density-1.0)*gravity*dss*dss*dss/
                     (100.0*viscosity*viscosity))-1.0);
            } else clear_ws=1.1*sqrt((relative_density-1.0)*gravity*dss);
        }
        const double ws=clear_ws*hindered;
        if(!isfinite(clear_ws)||clear_ws<0.0||!isfinite(ws)||ws<0.0)
            bad=true;
        clear_settling_velocity[point]=bad?0.0:clear_ws;
        settling_velocity[point]=bad?0.0:ws;
    }
    status[cell]=bad?1:0;
}

// Parent-timed peer used only when an authenticated owner supplies the
// effective suspended diameter for every fraction/cell.  This preserves the
// FLOW fallve order: the incoming (accepted) Dss is consumed before scalar
// transport, while any Dss produced by the current transport call belongs to
// the next accepted update.  The established static-profile kernel above is
// retained byte-for-byte for its historical path.
extern "C" __global__ void v92_parent_settling_diameter(
    const int nc, const int ng, const double dry,
    const double gravity, const double reference_concentration,
    const double *depth_before, const double *parent_suspended,
    const double *parent_temperature, const double *parent_water_density,
    const double *settling_diameter, const double *sediment_density,
    double *clear_settling_velocity, double *settling_velocity,
    double *hindered_factor, int *status)
{
    const int cell=blockIdx.x*blockDim.x+threadIdx.x;
    if(cell>=nc) return;
    const double h=depth_before[cell],temperature=parent_temperature[cell];
    const double rho=parent_water_density[cell];
    const bool parent_wet=h>dry;
    bool bad=!isfinite(h)||h<0.0||!isfinite(temperature)||temperature<=-20.0||
        !isfinite(rho)||rho<=0.0||!isfinite(dry)||dry<0.0||
        !isfinite(gravity)||gravity<=0.0||
        !isfinite(reference_concentration)||reference_concentration<=0.0;
    double total_concentration=0.0;
    for(int grain=0;grain<ng;++grain) {
        const int point=grain*nc+cell;
        const double mass=parent_suspended[point];
        const double dss=settling_diameter[point];
        if(!isfinite(mass)||mass<0.0||!isfinite(dss)||dss<=0.0||
           !isfinite(sediment_density[grain])||
           sediment_density[grain]<=rho) bad=true;
        if(parent_wet) total_concentration+=mass/h;
    }
    const double hindered=
        pow(fmax(0.0,1.0-fmax(0.0,total_concentration)/
                 reference_concentration),5.0);
    const double viscosity=4.0e-5/(20.0+temperature);
    hindered_factor[cell]=bad?0.0:hindered;
    for(int grain=0;grain<ng;++grain) {
        const int point=grain*nc+cell;
        const double dss=settling_diameter[point];
        const double relative_density=sediment_density[grain]/rho;
        double clear_ws=0.0;
        if(!bad) {
            if(dss<1.5*64.0e-6)
                clear_ws=(relative_density-1.0)*gravity*dss*dss/
                         (18.0*viscosity);
            else if(dss<0.5*2.0e-3) {
                const double coefficient=dss<2.0*64.0e-6?
                    -2.9912*dss/(64.0e-6)+15.9824:10.0;
                clear_ws=coefficient*viscosity/dss*
                    (sqrt(1.0+(relative_density-1.0)*gravity*dss*dss*dss/
                     (100.0*viscosity*viscosity))-1.0);
            } else clear_ws=1.1*sqrt((relative_density-1.0)*gravity*dss);
        }
        const double ws=clear_ws*hindered;
        if(!isfinite(clear_ws)||clear_ws<0.0||!isfinite(ws)||ws<0.0)
            bad=true;
        clear_settling_velocity[point]=bad?0.0:clear_ws;
        settling_velocity[point]=bad?0.0:ws;
    }
    status[cell]=bad?1:0;
}

// Dynamic Galappatti exchange with the already prepared parent settling
// arrays.  Exchange itself acts on the transported suspended mass.
extern "C" __global__ void v92_parent_timed_dynamic_exchange(
    const int nc, const int ng, const double *depth_before,
    const double *depth_after, const double *endpoint_momentum,
    const double *transported_suspended,
    const double *active_bed, const double *equilibrium_capacity,
    const double *surface_fraction, const double *reference_total_shear,
    const int *wet_mask, const double *sediment_threshold,
    const double *prepared_clear_settling_velocity,
    const double *prepared_settling_velocity,
    const double *prepared_hindered_factor,
    const double reference_water_density,
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
    const double h0=depth_before[cell],h1=depth_after[cell];
    const double hu=endpoint_momentum[cell],hv=endpoint_momentum[nc+cell];
    const bool endpoint_wet=wet_mask[cell]!=0;
    const bool exchange_active=
        h1*(endpoint_wet?1.0:0.0)>sediment_threshold[0];
    const double safe_h0=exchange_active?fmax(reference_depth_floor,h0):1.0;
    const double safe_h1=exchange_active?fmax(reference_depth_floor,h1):1.0;
    const double speed=endpoint_wet?hypot(hu,hv)/h1:0.0;
    const double tau=reference_total_shear[cell];
    const double shear=(isfinite(tau)&&tau>=0.0&&
                        reference_water_density>0.0)?
        sqrt(tau/reference_water_density):nan("");
    bool bad=!isfinite(h0)||!isfinite(h1)||!isfinite(hu)||!isfinite(hv)||
        !isfinite(speed)||!isfinite(shear)||h0<0.0||h1<0.0||
        !isfinite(reference_water_density)||
        reference_water_density<=0.0||endpoint_wet!=(h1>dry)||
        !isfinite(capacity_speed_epsilon)||capacity_speed_epsilon<=0.0||
        !isfinite(gravity)||gravity<=0.0||!isfinite(dry)||dry<0.0||
        !isfinite(factsd)||factsd<=0.0||
        !isfinite(reference_depth_floor)||reference_depth_floor<=0.0||
        !isfinite(sediment_threshold[0])||sediment_threshold[0]<0.0||
        !isfinite(dt)||dt<=0.0;
    sediment_active[cell]=exchange_active?1:0;
    const double hindered=prepared_hindered_factor[cell];
    if(!isfinite(hindered)||hindered<0.0||hindered>1.0) bad=true;
    for(int grain=0;grain<ng;++grain) {
        const int point=grain*nc+cell;
        if(!isfinite(transported_suspended[point])||
           transported_suspended[point]<0.0||
           !isfinite(active_bed[point])||active_bed[point]<0.0||
           !isfinite(equilibrium_capacity[point])||
           !isfinite(surface_fraction[point])||
           surface_fraction[point]<0.0||surface_fraction[point]>1.0||
           !isfinite(prepared_clear_settling_velocity[point])||
           prepared_clear_settling_velocity[point]<0.0||
           !isfinite(prepared_settling_velocity[point])||
           prepared_settling_velocity[point]<0.0) bad=true;
    }
    hindered_factor[cell]=bad?0.0:hindered;
    for(int grain=0;grain<ng;++grain) {
        const int point=grain*nc+cell;
        const double old_s=transported_suspended[point];
        const double old_b=active_bed[point],total=old_s+old_b;
        const double equilibrium=equilibrium_capacity[point]*
                                 surface_fraction[point];
        const double clear_ws=prepared_clear_settling_velocity[point];
        const double ws=endpoint_wet?prepared_settling_velocity[point]:0.0;
        const double wsl=fmax(1.0e-3,ws),eps=capacity_speed_epsilon;
        double timescale=0.0,hots=0.0;
        if(!bad&&exchange_active&&speed>eps&&shear>eps) {
            const double u=fmin(shear/speed,0.15);
            const double w=shear>wsl?wsl/shear:1.0;
            const double x=w,x2=x*x,x3=x2*x;
            const double exponent=
                (((double)1.547f-(double)20.12f*u)*x3)+
                (((double)326.832f*pow(u,(double)2.2047f)-(double)0.2f)*x2)+
                (((double)0.1385f*log(u)-(double)6.4061f)*x)+
                ((double)0.5467f*u+(double)2.1963f);
            timescale=x*exp(exponent);hots=wsl/(timescale*factsd);
        }
        const double explicit_source=(exchange_active&&speed>eps&&shear>eps)?
            equilibrium*hots/safe_h0:0.0;
        const double total_sink=!exchange_active?0.0:
            ((speed>eps&&shear>eps)?hots/safe_h1:wsl/safe_h1);
        const double unbounded=exchange_active?
            (old_s+dt*h0*explicit_source)/(1.0+dt*total_sink):old_s;
        const double candidate=fmin(fmax(unbounded,0.0),total);
        const double bed_after=total-candidate;
        bool local_bad=bad||!isfinite(equilibrium)||!isfinite(timescale)||
            !isfinite(hots)||!isfinite(total_sink)||!isfinite(candidate)||
            !isfinite(bed_after)||candidate<0.0||bed_after<0.0||
            !isfinite(unbounded)||!isfinite(clear_ws)||clear_ws<0.0||
            !isfinite(ws)||ws<0.0||
            ((exchange_active&&speed>eps&&shear>eps)&&timescale<=0.0);
        if(local_bad) {
            bad=true;next_suspended[point]=old_s;next_active[point]=old_b;
            equilibrium_concentration[point]=0.0;
            adaptation_velocity[point]=0.0;
            dimensionless_timescale[point]=0.0;
            clear_settling_velocity[point]=0.0;
            settling_velocity[point]=0.0;delta[point]=0.0;
            limiter_correction[point]=0.0;
        } else {
            next_suspended[point]=candidate;next_active[point]=bed_after;
            equilibrium_concentration[point]=equilibrium;
            adaptation_velocity[point]=exchange_active?hots:0.0;
            dimensionless_timescale[point]=exchange_active?timescale:0.0;
            clear_settling_velocity[point]=clear_ws;
            settling_velocity[point]=ws;delta[point]=candidate-old_s;
            limiter_correction[point]=fabs(candidate-unbounded);
        }
    }
    status[cell]=bad?1:0;
}
