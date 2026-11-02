// Deterministic resident reductions for the Q4-owned v93 Q3 adapter.
//
// Each output is owned by exactly one thread and every incident face is
// visited in the authenticated CSR order.  This deliberately avoids
// floating-point atomics: independent cold starts and checkpoint restarts
// must form the same bed-state identity on the same CUDA implementation.

extern "C" __global__ void q3_v93_outgoing_cell_gather(
    const int grain_count,
    const int cell_count,
    const int face_count,
    const int *cell_face_offset,
    const int *cell_faces,
    const int *donor,
    const int *owner,
    const int *neighbour,
    const int *role,
    const double *flux,
    const double *face_length,
    const double dt,
    const double *cell_area,
    const int donor_owner,
    const int donor_neighbour,
    const int donor_trace,
    const int face_internal,
    const int face_owner_outlet,
    double *outgoing)
{
    const long long item =
        (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long item_count =
        (long long)grain_count * (long long)cell_count;
    if (item >= item_count) return;

    const int grain = (int)(item / cell_count);
    const int cell = (int)(item - (long long)grain * cell_count);
    double value = 0.0;
    for (int slot = cell_face_offset[cell];
         slot < cell_face_offset[cell + 1]; ++slot) {
        const int face = cell_faces[slot];
        const int code = donor[(long long)grain * face_count + face];
        int donor_cell = -1;
        if (code == donor_owner || code == donor_trace) {
            donor_cell = owner[face];
        } else if (code == donor_neighbour) {
            donor_cell = neighbour[face];
        }
        if (donor_cell != cell) continue;

        const double face_flux =
            flux[(long long)grain * face_count + face];
        const int face_role = role[face];
        const bool leaves_inventory =
            (face_role == face_internal && face_flux != 0.0) ||
            (face_role == face_owner_outlet && face_flux > 0.0);
        if (leaves_inventory) {
            const double magnitude =
                face_flux < 0.0 ? -face_flux : face_flux;
            value += magnitude * face_length[face] * dt /
                     cell_area[cell];
        }
    }
    outgoing[item] = value;
}


extern "C" __global__ void q3_v93_delta_cell_gather(
    const int grain_count,
    const int cell_count,
    const int face_count,
    const int *cell_face_offset,
    const int *cell_faces,
    const int *cell_face_sign,
    const double *face_amount,
    const double *cell_area,
    double *cell_delta)
{
    const long long item =
        (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long item_count =
        (long long)grain_count * (long long)cell_count;
    if (item >= item_count) return;

    const int grain = (int)(item / cell_count);
    const int cell = (int)(item - (long long)grain * cell_count);
    double value = 0.0;
    for (int slot = cell_face_offset[cell];
         slot < cell_face_offset[cell + 1]; ++slot) {
        const int face = cell_faces[slot];
        value += -(double)cell_face_sign[slot] *
                 face_amount[(long long)grain * face_count + face] /
                 cell_area[cell];
    }
    cell_delta[item] = value;
}


extern "C" __global__ void q3_v93_named_boundary_gather(
    const int grain_count,
    const int boundary_count,
    const int boundary_face_count,
    const int face_count,
    const int *family_boundary_faces,
    const int *boundary_index,
    const double *face_amount,
    double *signed_transfer,
    double *absolute_transfer)
{
    const long long item =
        (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long item_count =
        (long long)grain_count * (long long)boundary_count;
    if (item >= item_count) return;

    const int grain = (int)(item / boundary_count);
    const int boundary =
        (int)(item - (long long)grain * boundary_count);
    double signed_value = 0.0;
    double absolute_value = 0.0;
    for (int local = 0; local < boundary_face_count; ++local) {
        if (boundary_index[local] != boundary) continue;
        const int face = family_boundary_faces[local];
        const double value =
            face_amount[(long long)grain * face_count + face];
        signed_value += value;
        absolute_value += value < 0.0 ? -value : value;
    }
    signed_transfer[item] = signed_value;
    absolute_transfer[item] = absolute_value;
}
