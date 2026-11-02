// Q3-H7/v71 resident conservative bed-inventory transfer and cleanup.
//
// Constitutive transport and Exner remain in the authenticated v67 family.
// These kernels are topology/ownership operators only.  Links are grouped by
// target cell and consumed in their canonical old-index order so CPU/OpenMP
// and CUDA use the same summation order per output value.

extern "C" __global__ void v71_transfer_bed_mass(
    const int nclass,
    const int nold,
    const int nnew,
    const int *new_offsets,
    const int *old_index,
    const double *overlap_area,
    const double *new_area,
    const double *old_mass,
    double *new_mass,
    int *status)
{
    const long long flat = (long long)blockDim.x*blockIdx.x + threadIdx.x;
    const long long count = (long long)nclass*nnew;
    if (flat >= count) return;
    const int fraction = (int)(flat/nnew);
    const int target = (int)(flat - (long long)fraction*nnew);
    if (nclass <= 0 || nold <= 0 || nnew <= 0 ||
        !(new_area[target] > 0.0)) {
        status[target] = 1;
        return;
    }
    const int start = new_offsets[target];
    const int finish = new_offsets[target+1];
    if (start < 0 || finish <= start) {
        status[target] = 2;
        return;
    }
    double total = 0.0;
    for (int link=start; link<finish; ++link) {
        const int source = old_index[link];
        const double overlap = overlap_area[link];
        if (source < 0 || source >= nold || !(overlap > 0.0)) {
            status[target] = 3;
            return;
        }
        const double value = old_mass[(long long)fraction*nold+source];
        if (!isfinite(value) || value < 0.0) {
            status[target] = 4;
            return;
        }
        total += value*overlap;
    }
    const double value = total/new_area[target];
    if (!isfinite(value) || value < 0.0) {
        status[target] = 5;
        return;
    }
    new_mass[flat] = value;
}

extern "C" __global__ void v71_validate_wetdry_faces(
    const int ncell,
    const int nface,
    const int *wet,
    const int *owner,
    const int *neighbour,
    const int *role,
    int *status)
{
    const int face = blockDim.x*blockIdx.x + threadIdx.x;
    if (face >= nface) return;
    const int own = owner[face];
    const int other = neighbour[face];
    const int face_role = role[face];
    int local = 0;
    if (ncell <= 0 || own < 0 || own >= ncell ||
        face_role < 0 || face_role > 4) {
        local = 1;
    } else if (face_role == 0) {
        if (other < 0 || other >= ncell || other == own ||
            wet[own] != 1 || wet[other] != 1) local = 2;
    } else {
        if (other != -1) local = 3;
        if (wet[own] != 1 && face_role != 4) local = 4;
    }
    status[face] = local;
}

extern "C" __global__ void v71_clear_face_register(
    const long long count,
    double *values)
{
    const long long item = (long long)blockDim.x*blockIdx.x + threadIdx.x;
    if (item < count) values[item] = 0.0;
}

