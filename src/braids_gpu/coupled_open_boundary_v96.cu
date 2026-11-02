// Actual-CUDA exterior KMAX=1 DensIn face operator for the v96 owner.
//
// Interior faces remain owned by the unchanged v85 kernel.  This kernel
// overwrites only active physical-open faces using the trace already resolved
// by the sole hydraulic boundary owner; it never treats that trace as a
// Riemann right state or launches another boundary solver.
extern "C" __global__ void v96_open_densin_face(
    const double* cell_depth,
    const double* cell_mixture_density,
    const double* boundary_mixture_density,
    const double* hydraulic_trace_q,
    const int* active_open_face,
    const int* owner,
    const double* owner_to_trace_distance,
    long long nf,
    double gravity,
    double reference_density,
    int ibaroc_open_boundary,
    double* face_acceleration,
    int* face_status)
{
    const long long face = (long long)blockDim.x*blockIdx.x + threadIdx.x;
    if (face >= nf || active_open_face[face] == 0) return;

    const int left = owner[face];
    const double h_left = cell_depth[left];
    const double h_boundary = hydraulic_trace_q[face];
    const double rho_left = cell_mixture_density[left];
    const double rho_boundary = boundary_mixture_density[face];
    const double distance = owner_to_trace_distance[face];
    if (!(h_left > 0.0) || !(h_boundary > 0.0) ||
            !(rho_left > 0.0) || !(rho_boundary > 0.0) ||
            !(distance > 0.0) || !isfinite(h_left) ||
            !isfinite(h_boundary) || !isfinite(rho_left) ||
            !isfinite(rho_boundary) || !isfinite(distance)) {
        face_status[face] = 11;
        return;
    }
    double value = 0.0;
    if (ibaroc_open_boundary != 0) {
        value = -gravity*(rho_boundary-rho_left)*(h_boundary+h_left)/
            (4.0*reference_density*distance);
    }
    if (!isfinite(value)) {
        face_status[face] = 12;
        return;
    }
    face_acceleration[face] = value;
    face_status[face] = 0;
}
