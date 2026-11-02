// Bounded resident post-density channel composition.
//
// Inputs are already mass-valued transport channels in class-major SoA
// layout.  The bed vector has already received any bed-only secondary-flow
// rotation.  The suspended vector is checked against, and projected onto, the
// unrotated depth-averaged base-flow direction.  No density conversion and no
// direction rotation occur in this kernel.

extern "C" __global__ void compose_post_density_channels(
    int ncell,
    int nclass,
    double bed_factor,
    double suspended_factor,
    double speed_epsilon,
    double alignment_rtol,
    double alignment_atol,
    const double* raw_bed_mass,
    const double* raw_suspended_mass,
    const double* base_velocity,
    const double* depth,
    const int* transport_class,
    double* composed_bed_mass,
    double* retained_suspended_mass,
    double* equilibrium_suspended_mass_concentration,
    double* signed_suspended_mass_rate,
    int* status)
{
    const int point = blockIdx.x * blockDim.x + threadIdx.x;
    const int npoint = nclass * ncell;
    if (point >= npoint) return;

    const int sediment_class = point / ncell;
    const int cell = point - sediment_class * ncell;
    const int y = npoint + point;
    const int velocity_y = ncell + cell;

    // Fail-safe outputs: every rejected point remains exactly zero.
    composed_bed_mass[point] = 0.0;
    composed_bed_mass[y] = 0.0;
    retained_suspended_mass[point] = 0.0;
    retained_suspended_mass[y] = 0.0;
    equilibrium_suspended_mass_concentration[point] = 0.0;
    signed_suspended_mass_rate[point] = 0.0;
    status[point] = 0;

    const double bx = raw_bed_mass[point];
    const double by = raw_bed_mass[y];
    const double sx = raw_suspended_mass[point];
    const double sy = raw_suspended_mass[y];
    const double u = base_velocity[cell];
    const double v = base_velocity[velocity_y];
    const double h = depth[cell];
    const int kind = transport_class[sediment_class];

    if (!isfinite(bx) || !isfinite(by) || !isfinite(sx) || !isfinite(sy) ||
        !isfinite(u) || !isfinite(v) || !isfinite(h)) {
        status[point] = 1;  // nonfinite point input
        return;
    }
    if (kind == 2) {
        // FLOW TRA_ADVDIFF uses erosilt and leaves the fraction loop before
        // eqtran; treating it as a formula-channel combination would invent
        // a mapping that the locked source does not contain.
        status[point] = 3;
        return;
    }
    if (kind != 1 && kind != 3) {
        status[point] = 2;  // unsupported transport class
        return;
    }
    if (kind == 3 && h <= 0.0) {
        status[point] = 4;  // combined concentration needs positive depth
        return;
    }

    const double speed = hypot(u, v);
    const double suspended_norm = hypot(sx, sy);
    if (!isfinite(speed) || !isfinite(suspended_norm)) {
        status[point] = 7;  // finite inputs overflowed a derived magnitude
        return;
    }

    double flow_x = 0.0;
    double flow_y = 0.0;
    double signed_raw_suspended = 0.0;
    if (speed == 0.0) {
        if (sx != 0.0 || sy != 0.0) {
            status[point] = 6;  // nonzero suspended vector has no flow axis
            return;
        }
    } else {
        flow_x = u / speed;
        flow_y = v / speed;
        if (suspended_norm != 0.0) {
            signed_raw_suspended = sx * flow_x + sy * flow_y;
            const double residual_x = sx - signed_raw_suspended * flow_x;
            const double residual_y = sy - signed_raw_suspended * flow_y;
            const double residual = hypot(residual_x, residual_y);
            if (!isfinite(signed_raw_suspended) || !isfinite(residual)) {
                status[point] = 7;
                return;
            }
            if (residual > alignment_atol +
                           alignment_rtol * suspended_norm) {
                status[point] = 5;  // not parallel or antiparallel to base flow
                return;
            }
        }
    }

    const double scaled_bx = bed_factor * bx;
    const double scaled_by = bed_factor * by;
    const double scaled_suspended = suspended_factor * signed_raw_suspended;
    if (!isfinite(scaled_bx) || !isfinite(scaled_by) ||
        !isfinite(scaled_suspended)) {
        status[point] = 7;
        return;
    }

    double out_x = scaled_bx;
    double out_y = scaled_by;
    double retained_x = 0.0;
    double retained_y = 0.0;
    double equilibrium = 0.0;
    if (kind == 1) {
        // Bedload-only FLOW fractions fold the formula's suspended channel
        // back into bed transport along the unrotated base-flow direction.
        out_x += scaled_suspended * flow_x;
        out_y += scaled_suspended * flow_y;
    } else {
        // Combined fractions retain bed transport separately and feed the
        // suspended channel to advection-diffusion as equilibrium mass
        // concentration.  The additive epsilon follows locked eqtran order.
        retained_x = scaled_suspended * flow_x;
        retained_y = scaled_suspended * flow_y;
        // Keep FLOW's operation order: eqtran executes two divisions, then
        // applies the active-bed fraction downstream when forming conc2d.
        equilibrium = scaled_suspended / (speed + speed_epsilon) / h;
    }

    if (!isfinite(out_x) || !isfinite(out_y) ||
        !isfinite(retained_x) || !isfinite(retained_y) ||
        !isfinite(equilibrium)) {
        status[point] = 7;
        return;
    }
    composed_bed_mass[point] = out_x;
    composed_bed_mass[y] = out_y;
    retained_suspended_mass[point] = retained_x;
    retained_suspended_mass[y] = retained_y;
    equilibrium_suspended_mass_concentration[point] = equilibrium;
    signed_suspended_mass_rate[point] = scaled_suspended;
}
