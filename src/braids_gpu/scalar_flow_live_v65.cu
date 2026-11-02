// Resident CUDA kernels for the bounded Q3-H5 scalar FLOW v65 transaction.
//
// Physical arrays stay on the selected device.  The kernels deliberately do
// not implement a second capacity or slope law: capacity comes from the v64
// producer, channel folding comes from channel_composition.cu, and optional
// ISlope 3/4 direction correction comes from flow_bed_slope_runtime.cu.

extern "C" __global__ void scalar_v65_lateral_421(
    long long ncell,
    int nfrac,
    const long long* x_offsets,
    const long long* x_indices,
    const double* x_weights,
    const long long* y_offsets,
    const long long* y_indices,
    const double* y_weights,
    const double* bed,
    double* averaged)
{
    const long long point = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long count = (long long)nfrac * ncell;
    if (point >= count) return;
    const int fraction = (int)(point / ncell);
    const long long cell = point - (long long)fraction * ncell;
    double x = 0.0;
    double y = 0.0;
    for (long long cursor=x_offsets[cell]; cursor<x_offsets[cell+1]; ++cursor) {
        x += x_weights[cursor] * bed[(long long)fraction*ncell+x_indices[cursor]];
    }
    const long long component_stride = (long long)nfrac*ncell;
    for (long long cursor=y_offsets[cell]; cursor<y_offsets[cell+1]; ++cursor) {
        y += y_weights[cursor] *
             bed[component_stride+(long long)fraction*ncell+y_indices[cursor]];
    }
    averaged[point] = x;
    averaged[component_stride+point] = y;
}

extern "C" __global__ void scalar_v65_face_candidates(
    long long ncell,
    int nfrac,
    long long nface,
    int upwind,
    const int* owner,
    const int* neighbour,
    const int* role,
    const double* normal,
    const double* cell_bed,
    const double* total_direction,
    const double* trace_bed,
    double* raw,
    int* raw_donor)
{
    const long long point = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long count = (long long)nfrac*nface;
    if (point >= count) return;
    const int fraction = (int)(point/nface);
    const long long face = point-(long long)fraction*nface;
    const int left_cell = owner[face];
    const int right_cell = neighbour[face];
    const int face_role = role[face];
    const double nx = normal[face];
    const double ny = normal[nface+face];
    const long long cell_stride = (long long)nfrac*ncell;
    const long long face_stride = (long long)nfrac*nface;
    const long long left = (long long)fraction*ncell+left_cell;
    const long long out = (long long)fraction*nface+face;
    double qx = 0.0;
    double qy = 0.0;
    int selected = 0;

    if (face_role == 0) { // internal
        const long long right = (long long)fraction*ncell+right_cell;
        const double lx = cell_bed[left];
        const double ly = cell_bed[cell_stride+left];
        const double rx = cell_bed[right];
        const double ry = cell_bed[cell_stride+right];
        if (upwind) {
            const double left_sign = total_direction[left]*nx+
                                     total_direction[cell_stride+left]*ny;
            const double right_sign = total_direction[right]*nx+
                                      total_direction[cell_stride+right]*ny;
            if (left_sign > 0.0 && right_sign > 0.0) {
                qx = lx; qy = ly; selected = 1;
            } else if (left_sign < 0.0 && right_sign < 0.0) {
                qx = rx; qy = ry; selected = 2;
            } else {
                qx = 0.5*(lx+rx); qy = 0.5*(ly+ry);
                selected = qx*nx+qy*ny < 0.0 ? 2 : 1;
            }
        } else {
            qx = 0.5*(lx+rx); qy = 0.5*(ly+ry);
            selected = qx*nx+qy*ny < 0.0 ? 2 : 1;
        }
    } else if (face_role == 1 || face_role == 2) { // trace in/out
        qx = trace_bed[out];
        qy = trace_bed[face_stride+out];
        selected = 3;
    } else if (face_role == 3) { // owner outlet
        qx = cell_bed[left];
        qy = cell_bed[cell_stride+left];
        selected = 1;
    } // role 4 is closed and retains exact zero/NONE

    raw[out] = qx;
    raw[face_stride+out] = qy;
    raw_donor[out] = selected;
}

extern "C" __global__ void scalar_v65_face_potential(
    int nfrac,
    long long nface,
    const int* owner,
    const int* neighbour,
    const int* role,
    const double* normal,
    const double* corrected,
    double* potential,
    int* donor,
    int* donor_cell)
{
    const long long point = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long count = (long long)nfrac*nface;
    if (point >= count) return;
    const long long face = point%nface;
    const long long stride = (long long)nfrac*nface;
    const int face_role = role[face];
    double value = corrected[point]*normal[face] +
                   corrected[stride+point]*normal[nface+face];
    int selected = 0;
    int cell = -1;
    if (face_role == 0) {
        if (value < 0.0) { selected = 2; cell = neighbour[face]; }
        else { selected = 1; cell = owner[face]; }
    } else if (face_role == 1) {
        value = fmin(0.0, value);
        selected = 3; cell = owner[face];
    } else if (face_role == 2) {
        value = fmax(0.0, value);
        selected = 3; cell = owner[face];
    } else if (face_role == 3) {
        value = fmax(0.0, value);
        selected = 1; cell = owner[face];
    } else {
        value = 0.0;
    }
    potential[point] = value;
    donor[point] = selected;
    donor_cell[point] = cell;
}

extern "C" __global__ void scalar_v65_inventory_demand(
    int nfrac,
    long long ncell,
    long long nface,
    double dt,
    double morfac,
    const int* role,
    const int* donor_cell,
    const double* potential,
    const double* active_fraction,
    const double* face_length,
    const double* cell_area,
    double* outgoing)
{
    const long long point = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long count = (long long)nfrac*nface;
    if (point >= count) return;
    const int fraction = (int)(point/nface);
    const long long face = point-(long long)fraction*nface;
    const int cell = donor_cell[point];
    if (cell < 0) return;
    const double preliminary = potential[point]*
        active_fraction[(long long)fraction*ncell+cell];
    const int face_role = role[face];
    const bool leaves_inventory =
        (face_role == 0 && preliminary != 0.0) ||
        ((face_role == 2 || face_role == 3) && preliminary > 0.0);
    if (!leaves_inventory) return;
    const double amount = fabs(preliminary)*face_length[face]*dt*morfac/
                          cell_area[cell];
    atomicAdd(&outgoing[(long long)fraction*ncell+cell], amount);
}

extern "C" __global__ void scalar_v65_inventory_fill(
    long long count,
    const double* mass,
    const double* active_fraction,
    const double* outgoing,
    double* fill,
    double* availability)
{
    const long long point = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (point >= count) return;
    double value = 1.0;
    if (mass[point] == 0.0) value = 0.0;
    else if (outgoing[point] > 0.0)
        value = fmin(1.0, mass[point]/outgoing[point]);
    fill[point] = value;
    availability[point] = active_fraction[point]*value;
}

extern "C" __global__ void scalar_v65_atomic_exner(
    int nfrac,
    long long ncell,
    long long nface,
    double dt,
    double morfac,
    const int* owner,
    const int* neighbour,
    const int* role,
    const int* donor_cell,
    const double* potential,
    const double* availability,
    const double* face_length,
    const double* cell_area,
    double* flux,
    double* delta,
    double* boundary_signed,
    double* boundary_absolute)
{
    const long long point = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long count = (long long)nfrac*nface;
    if (point >= count) return;
    const int fraction = (int)(point/nface);
    const long long face = point-(long long)fraction*nface;
    const int cell = donor_cell[point];
    double value = 0.0;
    if (cell >= 0)
        value = potential[point]*availability[(long long)fraction*ncell+cell];
    flux[point] = value;
    const double transfer = dt*morfac*face_length[face]*value;
    const int left = owner[face];
    atomicAdd(&delta[(long long)fraction*ncell+left],
              -transfer/cell_area[left]);
    if (role[face] == 0) {
        const int right = neighbour[face];
        atomicAdd(&delta[(long long)fraction*ncell+right],
                  transfer/cell_area[right]);
    } else {
        atomicAdd(&boundary_signed[fraction], transfer);
        atomicAdd(&boundary_absolute[fraction], fabs(transfer));
    }
}

__device__ __forceinline__ unsigned long long scalar_v65_mix64(
    unsigned long long value)
{
    value ^= value >> 30;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27;
    value *= 0x94d049bb133111ebULL;
    value ^= value >> 31;
    return value;
}

extern "C" __global__ void scalar_v65_digest_fp64(
    long long count,
    unsigned long long seed,
    const double* values,
    unsigned long long* digest)
{
    const long long point = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (point >= count) return;
    const unsigned long long bits = (unsigned long long)__double_as_longlong(values[point]);
    const unsigned long long index = (unsigned long long)point;
    atomicXor(&digest[0], scalar_v65_mix64(bits ^ seed ^
              (index+1ULL)*0x9e3779b97f4a7c15ULL));
    atomicXor(&digest[1], scalar_v65_mix64(bits ^ ~seed ^
              (index+1ULL)*0xd6e8feb86659fd93ULL));
}
