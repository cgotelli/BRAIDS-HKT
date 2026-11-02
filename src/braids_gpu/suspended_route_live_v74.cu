// Bounded Q4 v74 split-route owner for actual CUDA execution.
//
// Inputs are already mass-valued Q3 constitutive outputs.  This kernel owns
// only off/total/combine partition and combine-only soursin_2d exchange.  It
// neither copies nor modifies Delft3D source and performs no CPU fallback.

extern "C" __global__ void suspended_route_update_v74(
    const int nlane, const int ncell, const int *output_kind,
    const int *route, const double *depth, const double *velocity_x,
    const double *velocity_y, const double *raw_bed_x,
    const double *raw_bed_y, const double *raw_suspended_x,
    const double *raw_suspended_y, const double *raw_equilibrium,
    const double *active_fraction, const double *old_bed,
    const double *old_suspended, const double *depth_before,
    const double *depth_after, const double *speed,
    const double *shear_velocity, const double *settling_velocity,
    const double *cell_area, const double bed_factor,
    const double suspended_factor, const double speed_epsilon,
    const double fac_tsd, const double sediment_threshold,
    const double minimum_reference_depth, const double dt,
    double *bed_capacity_x, double *bed_capacity_y,
    double *equilibrium_concentration, double *suspended_equivalent_x,
    double *suspended_equivalent_y, double *next_bed,
    double *next_suspended, double *transferred_to_suspension,
    double *dimensionless_timescale, double *adaptation_velocity,
    double *limiter_area, double *mass_residual, int *status)
{
    const int point=blockIdx.x*blockDim.x+threadIdx.x;
    const int count=nlane*ncell;
    if(point>=count) return;
    const int lane=point/ncell;
    const int cell=point-lane*ncell;
    const int kind=output_kind[lane];
    const int selected_route=route[lane];
    const double h=depth[cell];
    const double ux=velocity_x[cell];
    const double uy=velocity_y[cell];
    const double flow_speed=hypot(ux,uy);
    const double fraction=active_fraction[point];
    const double source_bed_x=raw_bed_x[point];
    const double source_bed_y=raw_bed_y[point];
    const double source_suspended_x=raw_suspended_x[point];
    const double source_suspended_y=raw_suspended_y[point];
    const double source_equilibrium=raw_equilibrium[point];
    const double parent_bed=old_bed[point];
    const double parent_suspended=old_suspended[point];
    const double h0=depth_before[cell];
    const double h1=depth_after[cell];
    const double umod=speed[cell];
    const double ustar=shear_velocity[cell];
    const double ws=settling_velocity[point];
    const double area=cell_area[cell];

    bool bad=!isfinite(h) || h<=0.0 || !isfinite(ux) || !isfinite(uy) ||
        !isfinite(flow_speed) || !isfinite(fraction) || fraction<0.0 ||
        fraction>1.0 || !isfinite(source_bed_x) ||
        !isfinite(source_bed_y) || !isfinite(source_suspended_x) ||
        !isfinite(source_suspended_y) || !isfinite(source_equilibrium) ||
        source_equilibrium<0.0 || !isfinite(parent_bed) || parent_bed<0.0 ||
        !isfinite(parent_suspended) || parent_suspended<0.0 ||
        !isfinite(h0) || h0<=0.0 || !isfinite(h1) || h1<=0.0 ||
        !isfinite(umod) || umod<0.0 || !isfinite(ustar) || ustar<0.0 ||
        !isfinite(ws) || ws<0.0 || !isfinite(area) || area<=0.0 ||
        !isfinite(bed_factor) || bed_factor<0.0 ||
        !isfinite(suspended_factor) || suspended_factor<0.0 ||
        !isfinite(speed_epsilon) || speed_epsilon<=0.0 ||
        !isfinite(fac_tsd) || fac_tsd<=0.0 ||
        !isfinite(sediment_threshold) || sediment_threshold<0.0 ||
        !isfinite(minimum_reference_depth) || minimum_reference_depth<=0.0 ||
        !isfinite(dt) || dt<=0.0 ||
        (kind!=1 && kind!=2) ||
        (selected_route!=0 && selected_route!=1 && selected_route!=3) ||
        (kind==1 && source_equilibrium!=0.0) ||
        (kind==2 && (source_suspended_x!=0.0 || source_suspended_y!=0.0));

    double scaled_nonbed_x=0.0,scaled_nonbed_y=0.0,derived_equilibrium=0.0;
    if(kind==1) {
        scaled_nonbed_x=suspended_factor*source_suspended_x;
        scaled_nonbed_y=suspended_factor*source_suspended_y;
        derived_equilibrium=suspended_factor*
            hypot(source_suspended_x,source_suspended_y)/
            (flow_speed+speed_epsilon)/h;
    } else {
        derived_equilibrium=suspended_factor*source_equilibrium;
        scaled_nonbed_x=derived_equilibrium*h*ux;
        scaled_nonbed_y=derived_equilibrium*h*uy;
    }

    double route_bed_x=0.0,route_bed_y=0.0,route_equilibrium=0.0;
    double equivalent_x=0.0,equivalent_y=0.0;
    if(selected_route!=0) {
        equivalent_x=fraction*scaled_nonbed_x;
        equivalent_y=fraction*scaled_nonbed_y;
        if(selected_route==1) {
            route_bed_x=fraction*(bed_factor*source_bed_x+scaled_nonbed_x);
            route_bed_y=fraction*(bed_factor*source_bed_y+scaled_nonbed_y);
        } else {
            route_bed_x=fraction*bed_factor*source_bed_x;
            route_bed_y=fraction*bed_factor*source_bed_y;
            route_equilibrium=fraction*derived_equilibrium;
        }
    }

    const bool active=selected_route==3 && h1>sediment_threshold;
    const bool moving=active && umod>1.0e-6 && ustar>1.0e-6;
    const double wsl=fmax(1.0e-3,ws);
    double timescale=0.0,hots=0.0,explicit_source=0.0,total_sink=0.0;
    if(moving) {
        const double ratio=fmin(ustar/umod,0.15);
        const double w=ustar>wsl ? wsl/ustar : 1.0;
        const double x2=w*w,x3=x2*w;
        const double exponent=
            (((double)1.547f-(double)20.12f*ratio)*x3)+
            (((double)326.832f*pow(ratio,(double)2.2047f)-(double)0.2f)*x2)+
            (((double)0.1385f*log(ratio)-(double)6.4061f)*w)+
            ((double)0.5467f*ratio+(double)2.1963f);
        timescale=w*exp(exponent);
        hots=wsl/(timescale*fac_tsd);
        explicit_source=route_equilibrium*hots/fmax(minimum_reference_depth,h0);
        total_sink=hots/fmax(minimum_reference_depth,h1);
    } else if(active) {
        total_sink=wsl/fmax(minimum_reference_depth,h1);
    }
    const double total=parent_bed+parent_suspended;
    const double raw_candidate=active ?
        (parent_suspended+dt*h0*explicit_source)/(1.0+dt*total_sink) :
        parent_suspended;
    const double candidate_suspended=active ?
        fmin(fmax(raw_candidate,0.0),total) : parent_suspended;
    const double candidate_bed=total-candidate_suspended;
    const double transfer=candidate_suspended-parent_suspended;
    const double local_residual=
        ((candidate_suspended+candidate_bed)-total)*area;
    bad=bad || !isfinite(route_bed_x) || !isfinite(route_bed_y) ||
        !isfinite(route_equilibrium) || !isfinite(timescale) ||
        !isfinite(hots) || !isfinite(candidate_suspended) ||
        !isfinite(candidate_bed) || candidate_suspended<0.0 ||
        candidate_bed<0.0 || (moving && timescale<=0.0);

    if(bad) {
        bed_capacity_x[point]=0.0;
        bed_capacity_y[point]=0.0;
        equilibrium_concentration[point]=0.0;
        suspended_equivalent_x[point]=0.0;
        suspended_equivalent_y[point]=0.0;
        next_bed[point]=parent_bed;
        next_suspended[point]=parent_suspended;
        transferred_to_suspension[point]=0.0;
        dimensionless_timescale[point]=0.0;
        adaptation_velocity[point]=0.0;
        limiter_area[point]=0.0;
        mass_residual[point]=0.0;
        status[point]=1;
        return;
    }
    bed_capacity_x[point]=route_bed_x;
    bed_capacity_y[point]=route_bed_y;
    equilibrium_concentration[point]=route_equilibrium;
    suspended_equivalent_x[point]=equivalent_x;
    suspended_equivalent_y[point]=equivalent_y;
    next_bed[point]=candidate_bed;
    next_suspended[point]=candidate_suspended;
    transferred_to_suspension[point]=transfer;
    dimensionless_timescale[point]=timescale;
    adaptation_velocity[point]=hots;
    limiter_area[point]=fabs(candidate_suspended-raw_candidate)*area;
    mass_residual[point]=local_residual;
    status[point]=0;
}
