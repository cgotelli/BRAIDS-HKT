// Additive Q4/v81 bridge from raw v74 split rows to the v78 lifecycle ABI.
//
// This kernel performs route algebra only.  It never advances phase mass.
// The returned pre-fraction capacity is consumed exactly once by the frozen
// v78 CUDA lifecycle, which owns settling/exchange/advection/diffusion,
// boundary memory and wet/dry cleanup.

extern "C" __global__ void route_lifecycle_bridge_v81(
    const int nlane, const int ncell, const int *output_kind,
    const int *route, const double *depth_after,
    const double *endpoint_momentum, const double *raw_bed_x,
    const double *raw_bed_y, const double *raw_suspended_x,
    const double *raw_suspended_y, const double *raw_equilibrium,
    const double *active_fraction, const double bed_factor,
    const double suspended_factor, const double speed_epsilon,
    const double dry_tolerance, const double sediment_threshold,
    const double reference_depth, double *bed_capacity_x,
    double *bed_capacity_y, double *equilibrium_target,
    double *suspended_equivalent_x, double *suspended_equivalent_y,
    double *lifecycle_capacity, int *status)
{
    const int point=blockIdx.x*blockDim.x+threadIdx.x;
    const int count=nlane*ncell;
    if(point>=count) return;
    const int lane=point/ncell;
    const int cell=point-lane*ncell;
    const int kind=output_kind[lane];
    const int selected_route=route[lane];
    const double h1=depth_after[cell];
    const bool wet=h1>dry_tolerance;
    const bool sediment_active=wet && h1>sediment_threshold;
    const double exchange_depth=sediment_active ?
        fmax(reference_depth,h1) : 1.0;
    const double mx=endpoint_momentum[cell];
    const double my=endpoint_momentum[ncell+cell];
    const double ux=wet ? mx/h1 : 0.0;
    const double uy=wet ? my/h1 : 0.0;
    const double speed=hypot(ux,uy);
    const double fraction=active_fraction[point];
    const double source_bed_x=raw_bed_x[point];
    const double source_bed_y=raw_bed_y[point];
    const double source_sus_x=raw_suspended_x[point];
    const double source_sus_y=raw_suspended_y[point];
    const double source_eq=raw_equilibrium[point];

    bool bad=!isfinite(h1) || h1<0.0 || !isfinite(mx) || !isfinite(my) ||
        !isfinite(ux) || !isfinite(uy) || !isfinite(speed) ||
        !isfinite(fraction) || fraction<0.0 || fraction>1.0 ||
        !isfinite(source_bed_x) || !isfinite(source_bed_y) ||
        !isfinite(source_sus_x) || !isfinite(source_sus_y) ||
        !isfinite(source_eq) || source_eq<0.0 ||
        !isfinite(bed_factor) || bed_factor<0.0 ||
        !isfinite(suspended_factor) || suspended_factor<0.0 ||
        !isfinite(speed_epsilon) || speed_epsilon<=0.0 ||
        !isfinite(dry_tolerance) || dry_tolerance<0.0 ||
        !isfinite(sediment_threshold) || sediment_threshold<0.0 ||
        !isfinite(reference_depth) || reference_depth<=0.0 ||
        (kind!=1 && kind!=2) ||
        (selected_route!=0 && selected_route!=1 && selected_route!=3) ||
        (kind==1 && source_eq!=0.0) ||
        (kind==2 && (source_sus_x!=0.0 || source_sus_y!=0.0));
    if(!sediment_active &&
       (source_bed_x!=0.0 || source_bed_y!=0.0 || source_sus_x!=0.0 ||
        source_sus_y!=0.0 || source_eq!=0.0)) bad=true;

    double nonbed_x=0.0,nonbed_y=0.0,derived_eq=0.0;
    double pre_fraction_capacity=0.0;
    if(kind==1) {
        const double magnitude=hypot(source_sus_x,source_sus_y);
        nonbed_x=suspended_factor*source_sus_x;
        nonbed_y=suspended_factor*source_sus_y;
        derived_eq=suspended_factor*magnitude/
            (speed+speed_epsilon)/exchange_depth;
        pre_fraction_capacity=suspended_factor*magnitude;
    } else {
        derived_eq=suspended_factor*source_eq;
        nonbed_x=derived_eq*exchange_depth*ux;
        nonbed_y=derived_eq*exchange_depth*uy;
        pre_fraction_capacity=derived_eq*(speed+speed_epsilon)*
            exchange_depth;
    }

    double bx=0.0,by=0.0,target=0.0,ex=0.0,ey=0.0,capacity=0.0;
    if(selected_route!=0) {
        ex=fraction*nonbed_x;
        ey=fraction*nonbed_y;
        if(selected_route==1) {
            bx=fraction*(bed_factor*source_bed_x+nonbed_x);
            by=fraction*(bed_factor*source_bed_y+nonbed_y);
        } else {
            bx=fraction*bed_factor*source_bed_x;
            by=fraction*bed_factor*source_bed_y;
            target=fraction*derived_eq;
            capacity=pre_fraction_capacity;
        }
    }
    bad=bad || !isfinite(bx) || !isfinite(by) || !isfinite(target) ||
        !isfinite(ex) || !isfinite(ey) || !isfinite(capacity) || capacity<0.0;
    if(bad) {
        bed_capacity_x[point]=0.0;
        bed_capacity_y[point]=0.0;
        equilibrium_target[point]=0.0;
        suspended_equivalent_x[point]=0.0;
        suspended_equivalent_y[point]=0.0;
        lifecycle_capacity[point]=0.0;
        status[point]=1;
        return;
    }
    bed_capacity_x[point]=bx;
    bed_capacity_y[point]=by;
    equilibrium_target[point]=target;
    suspended_equivalent_x[point]=ex;
    suspended_equivalent_y[point]=ey;
    lifecycle_capacity[point]=capacity;
    status[point]=0;
}
