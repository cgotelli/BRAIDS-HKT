// Additive live-selected Q3-H3/v59 kernels.
//
// The accepted v58 controller remains the sole macrostep owner.  These
// kernels only replace its legacy Betac face injection with Delft3D's
// depth-weighted velocity-source mapping and expose a deterministic
// once-per-accepted-step face register.  Open faces are deliberately rejected
// by the Python admission gate until O1 supplies their numerical operator.

enum V59LiveStatus {
    V59_LIVE_OK = 0,
    V59_LIVE_INVALID_INPUT = 1,
    V59_LIVE_INVALID_TOPOLOGY = 2,
    V59_LIVE_INVALID_OUTPUT = 3
};

extern "C" __global__ void v59_live_signed_hi_register(
    const int ncell, const int nface, const int *owner,
    const int *neighbour, const double *length,
    const double *advective_flux, const double *diffusion_conductance,
    const double *diffused_intensity, const double dt,
    double *signed_hi_register, int *status)
{
    const int face = blockIdx.x*blockDim.x + threadIdx.x;
    if (face >= nface) return;
    status[face] = V59_LIVE_OK;
    const int left = owner[face], right = neighbour[face];
    const double ds = length[face];
    const double fa = advective_flux[face];
    const double g = diffusion_conductance[face];
    if (left < 0 || left >= ncell || right >= ncell || !(ds > 0.0) ||
        !isfinite(ds) || !isfinite(dt) || dt < 0.0 || !isfinite(fa) ||
        !isfinite(g) || g < 0.0) {
        status[face] = V59_LIVE_INVALID_INPUT;
        return;
    }
    if (right < 0) {
        // This first slice has closed exterior faces.  Their advective and
        // diffusive physical transfer must therefore be identically zero.
        if (fa != 0.0 || g != 0.0)
            status[face] = V59_LIVE_INVALID_TOPOLOGY;
        signed_hi_register[face] = 0.0;
        signed_hi_register[nface + face] = 0.0;
        return;
    }
    const double il = diffused_intensity[left];
    const double ir = diffused_intensity[right];
    if (!isfinite(il) || !isfinite(ir)) {
        status[face] = V59_LIVE_INVALID_INPUT;
        return;
    }
    // Both entries are positive owner-to-neighbour integrated transfers.
    // The advection flux is the exact accepted FCT face flux driven by the
    // converged implicit hydraulic water flux.  Diffusion uses the converged
    // BE intensity and its already assembled conductance.
    signed_hi_register[face] = dt*ds*fa;
    signed_hi_register[nface + face] = dt*g*(il-ir);
    if (!isfinite(signed_hi_register[face]) ||
        !isfinite(signed_hi_register[nface + face]))
        status[face] = V59_LIVE_INVALID_OUTPUT;
}

extern "C" __global__ void v59_live_add_depth_weighted_betac(
    const int ncell, const int nface, const int *owner,
    const int *neighbour, const double *accepted_q,
    const double *stress_flux, double *flux_left, double *flux_right,
    int *status)
{
    const int face = blockIdx.x*blockDim.x + threadIdx.x;
    if (face >= nface) return;
    status[face] = V59_LIVE_OK;
    const int left = owner[face], right = neighbour[face];
    if (left < 0 || left >= ncell || right >= ncell) {
        status[face] = V59_LIVE_INVALID_TOPOLOGY;
        return;
    }
    const double hl = accepted_q[left];
    const double sx = stress_flux[face];
    const double sy = stress_flux[nface + face];
    if (right < 0) {
        // Numerical open faces are rejected by the host admission gate.  The
        // only remaining exterior segments are closed walls and v58 freezes
        // their tensor traction to exact zero.
        if (sx != 0.0 || sy != 0.0) status[face] = V59_LIVE_INVALID_TOPOLOGY;
        return;
    }
    const double hr = accepted_q[right];
    if (!(hl > 0.0) || !(hr > 0.0) || !isfinite(hl) || !isfinite(hr) ||
        !isfinite(sx) || !isfinite(sy)) {
        status[face] = V59_LIVE_INVALID_INPUT;
        return;
    }
    // The f-wave owner subtracts ``flux_left`` from the owner and adds
    // ``flux_right`` to the neighbour.  Delft3D's T-divergence is a velocity
    // source, hence the two depth-integrated momentum contributions use the
    // receiving cell's accepted-start depth and need not cancel when hl!=hr.
    flux_left[nface + face] += hl*sx;
    flux_left[2*nface + face] += hl*sy;
    flux_right[nface + face] += hr*sx;
    flux_right[2*nface + face] += hr*sy;
}

extern "C" __global__ void v59_live_betac_register(
    const int ncell, const int nface, const int *owner,
    const int *neighbour, const double *length, const double *stress_flux,
    const double dt, double *velocity_register, int *status)
{
    const int face = blockIdx.x*blockDim.x + threadIdx.x;
    if (face >= nface) return;
    status[face] = V59_LIVE_OK;
    const int left = owner[face], right = neighbour[face];
    const double ds = length[face];
    const double sx = stress_flux[face];
    const double sy = stress_flux[nface + face];
    if (left < 0 || left >= ncell || right >= ncell ||
        !(ds > 0.0) || !isfinite(ds) || !isfinite(dt) || dt < 0.0 ||
        !isfinite(sx) || !isfinite(sy)) {
        status[face] = V59_LIVE_INVALID_INPUT;
        return;
    }
    if (right < 0) {
        if (sx != 0.0 || sy != 0.0) status[face] = V59_LIVE_INVALID_TOPOLOGY;
        velocity_register[face] = 0.0;
        velocity_register[nface + face] = 0.0;
        return;
    }
    velocity_register[face] = dt*ds*sx;
    velocity_register[nface + face] = dt*ds*sy;
}

extern "C" __global__ void v59_live_gather_betac_increment(
    const int ncell, const int nface, const int *offset, const int *faces,
    const int *owner, const int *neighbour, const double *accepted_q,
    const double *velocity_register, double *cell_increment, int *status)
{
    const int cell = blockIdx.x*blockDim.x + threadIdx.x;
    if (cell >= ncell) return;
    status[cell] = V59_LIVE_OK;
    if (offset[cell] < 0 || offset[cell + 1] < offset[cell] ||
        !isfinite(accepted_q[cell]) || !(accepted_q[cell] > 0.0)) {
        status[cell] = V59_LIVE_INVALID_INPUT;
        return;
    }
    double x = 0.0, y = 0.0;
    for (int item = offset[cell]; item < offset[cell + 1]; ++item) {
        const int face = faces[item];
        if (face < 0 || face >= nface) {
            status[cell] = V59_LIVE_INVALID_TOPOLOGY;
            return;
        }
        if (owner[face] == cell) {
            x -= velocity_register[face];
            y -= velocity_register[nface + face];
        } else if (neighbour[face] == cell) {
            x += velocity_register[face];
            y += velocity_register[nface + face];
        } else {
            status[cell] = V59_LIVE_INVALID_TOPOLOGY;
            return;
        }
    }
    x *= accepted_q[cell];
    y *= accepted_q[cell];
    if (!isfinite(x) || !isfinite(y)) {
        status[cell] = V59_LIVE_INVALID_OUTPUT;
        return;
    }
    cell_increment[cell] = x;
    cell_increment[ncell + cell] = y;
}
