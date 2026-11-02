// Resident post-formula FLOW route composition for direct-vector methods.
//
// Inputs are already solid-mass-valued.  The kernel performs no density
// conversion and no direction rotation.  ``output_kind`` is the dedicated
// ownership enum resolved from stable method strings by the host wrapper; it
// is deliberately unrelated to historical Delft3D numeric law IDs.

enum DirectVectorOutputKindV66 {
    DV_ZERO_NONBED = 0,
    DV_EXPLICIT_SUSPENDED_VECTOR = 1,
    DV_EQUILIBRIUM_CONCENTRATION = 2
};

enum DirectVectorRouteV66 {
    DV_TRA_BEDLOAD = 0,
    DV_TRA_COMBINE = 1
};

enum DirectVectorRouteStatusV66 {
    DV_ROUTE_OK = 0,
    DV_ROUTE_INVALID_METHOD = 1,
    DV_ROUTE_INVALID_ROUTE = 2,
    DV_ROUTE_NONFINITE = 3,
    DV_ROUTE_INVALID_ARGUMENT = 4,
    DV_ROUTE_SOURCE_KIND_MISMATCH = 5,
    DV_ROUTE_DRY_SOURCE_NONZERO = 6,
    DV_ROUTE_ARITHMETIC = 7
};

__device__ __forceinline__ void dv_zero_output(
    const long long point, const long long count, double *output)
{
    output[point] = 0.0;
    output[count + point] = 0.0;
    output[2*count + point] = 0.0;
}

extern "C" __global__ void compose_direct_vector_route_v66(
    const long long count,
    const int *output_kind,
    const int *route,
    const double *depth,
    const double *velocity_x,
    const double *velocity_y,
    const double *raw_bed_mass_x,
    const double *raw_bed_mass_y,
    const double *raw_suspended_mass_x,
    const double *raw_suspended_mass_y,
    const double *raw_equilibrium_mass_concentration,
    const double *bed_factor,
    const double *suspended_factor,
    const double *active_fraction,
    const double *speed_epsilon,
    double *output,
    int *status)
{
    const long long i = (long long)blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= count) return;
    dv_zero_output(i, count, output);
    status[i] = DV_ROUTE_NONFINITE;

    const int kind = output_kind[i];
    const int selected_route = route[i];
    if (kind < DV_ZERO_NONBED || kind > DV_EQUILIBRIUM_CONCENTRATION) {
        status[i] = DV_ROUTE_INVALID_METHOD;
        return;
    }
    if (selected_route != DV_TRA_BEDLOAD &&
            selected_route != DV_TRA_COMBINE) {
        status[i] = DV_ROUTE_INVALID_ROUTE;
        return;
    }

    const double h = depth[i];
    const double ux = velocity_x[i];
    const double uy = velocity_y[i];
    const double bx = raw_bed_mass_x[i];
    const double by = raw_bed_mass_y[i];
    const double sx = raw_suspended_mass_x[i];
    const double sy = raw_suspended_mass_y[i];
    const double ceq = raw_equilibrium_mass_concentration[i];
    const double bed = bed_factor[i];
    const double sus = suspended_factor[i];
    const double fraction = active_fraction[i];
    const double epsilon = speed_epsilon[i];
    if (!isfinite(h) || !isfinite(ux) || !isfinite(uy) ||
            !isfinite(bx) || !isfinite(by) || !isfinite(sx) ||
            !isfinite(sy) || !isfinite(ceq) || !isfinite(bed) ||
            !isfinite(sus) || !isfinite(fraction) || !isfinite(epsilon)) {
        return;
    }
    if (h < 0.0 || fraction < 0.0 || fraction > 1.0 ||
            epsilon <= 0.0 || ceq < 0.0) {
        status[i] = DV_ROUTE_INVALID_ARGUMENT;
        return;
    }

    const bool vector_nonzero = sx != 0.0 || sy != 0.0;
    const bool concentration_nonzero = ceq != 0.0;
    if ((kind == DV_ZERO_NONBED &&
            (vector_nonzero || concentration_nonzero)) ||
        (kind == DV_EXPLICIT_SUSPENDED_VECTOR && concentration_nonzero) ||
        (kind == DV_EQUILIBRIUM_CONCENTRATION && vector_nonzero)) {
        status[i] = DV_ROUTE_SOURCE_KIND_MISMATCH;
        return;
    }

    const bool source_nonzero = bx != 0.0 || by != 0.0 ||
                                vector_nonzero || concentration_nonzero;
    if (h == 0.0) {
        if (source_nonzero) {
            status[i] = DV_ROUTE_DRY_SOURCE_NONZERO;
            return;
        }
        status[i] = DV_ROUTE_OK;
        return;
    }

    // These are the only Bed/Sus applications in the operator.
    const double scaled_bx = bed*bx;
    const double scaled_by = bed*by;
    double nonbed_x = 0.0;
    double nonbed_y = 0.0;
    double scaled_concentration = 0.0;
    if (kind == DV_EXPLICIT_SUSPENDED_VECTOR) {
        nonbed_x = sus*sx;
        nonbed_y = sus*sy;
    } else if (kind == DV_EQUILIBRIUM_CONCENTRATION) {
        scaled_concentration = sus*ceq;
        nonbed_x = scaled_concentration*h*ux;
        nonbed_y = scaled_concentration*h*uy;
    }

    double result_x;
    double result_y;
    double q4_concentration = 0.0;
    if (selected_route == DV_TRA_BEDLOAD) {
        // Algebraic total first; active fraction exactly once afterwards.
        result_x = fraction*(scaled_bx + nonbed_x);
        result_y = fraction*(scaled_by + nonbed_y);
    } else {
        result_x = fraction*scaled_bx;
        result_y = fraction*scaled_by;
        if (kind == DV_EXPLICIT_SUSPENDED_VECTOR) {
            const double suspended_magnitude = hypot(sx, sy);
            const double speed = hypot(ux, uy);
            q4_concentration = fraction*sus*suspended_magnitude/
                               (speed + epsilon)/h;
        } else if (kind == DV_EQUILIBRIUM_CONCENTRATION) {
            q4_concentration = fraction*scaled_concentration;
        }
    }

    if (!isfinite(result_x) || !isfinite(result_y) ||
            !isfinite(q4_concentration)) {
        status[i] = DV_ROUTE_ARITHMETIC;
        return;
    }
    output[i] = result_x;
    output[count + i] = result_y;
    output[2*count + i] = q4_concentration;
    status[i] = DV_ROUTE_OK;
}
