extern "C" __global__ void v90_lpfluc_face(
    const int nf,
    const double closure_interval_s,
    const double hturrt_min,
    const double* __restrict__ previous_filtered,
    const double* __restrict__ mean_velocity,
    const int* __restrict__ active_face,
    double* __restrict__ filtered,
    double* __restrict__ fluctuation,
    int* __restrict__ status)
{
    const int face = blockDim.x * blockIdx.x + threadIdx.x;
    if (face >= nf) return;
    int code = 0;
    const double old_value = previous_filtered[face];
    const double mean_value = mean_velocity[face];
    const int active = active_face[face];
    if (!(isfinite(old_value) && isfinite(mean_value)) ||
        !(active == 0 || active == 1) || closure_interval_s <= 0.0 ||
        !isfinite(closure_interval_s) ||
        !(hturrt_min > 0.0 || hturrt_min == -1.0)) {
        code = 1;
    }
    double low = 0.0;
    if (code == 0 && hturrt_min > 0.0 && active == 1) {
        const double a = exp(-closure_interval_s / (60.0 * hturrt_min));
        low = a * old_value + (1.0 - a) * mean_value;
    }
    filtered[face] = low;
    fluctuation[face] = code == 0 ? mean_value - low : 0.0;
    status[face] = code;
}

extern "C" __global__ void v90_reconstruct_cell_vector_nface(
    const int nc,
    const int nf,
    const int* __restrict__ cell_face_offset,
    const int* __restrict__ cell_faces,
    const double* __restrict__ normal,
    const double* __restrict__ face_length,
    const double* __restrict__ centre_distance,
    const int* __restrict__ active_face,
    const double* __restrict__ face_value,
    double* __restrict__ cell_vector,
    int* __restrict__ complete,
    int* __restrict__ status)
{
    const int cell = blockDim.x * blockIdx.x + threadIdx.x;
    if (cell >= nc) return;
    double axx = 0.0, axy = 0.0, ayy = 0.0, bx = 0.0, by = 0.0;
    int count = 0;
    int code = 0;
    for (int slot = cell_face_offset[cell];
         slot < cell_face_offset[cell + 1]; ++slot) {
        const int face = cell_faces[slot];
        if (face < 0 || face >= nf) { code = 1; continue; }
        if (active_face[face] == 0) continue;
        const double nx = normal[face];
        const double ny = normal[nf + face];
        const double length = face_length[face];
        const double distance = centre_distance[face];
        const double q = face_value[face];
        if (!(isfinite(nx) && isfinite(ny) && isfinite(length) &&
              isfinite(distance) && isfinite(q)) || length <= 0.0 ||
            distance <= 0.0) {
            code = 2;
            continue;
        }
        const double weight = length / distance;
        axx += weight * nx * nx;
        axy += weight * nx * ny;
        ayy += weight * ny * ny;
        bx += weight * nx * q;
        by += weight * ny * q;
        ++count;
    }
    const double determinant = axx * ayy - axy * axy;
    const double trace = axx + ayy;
    const double scale = fmax(1.0, trace * trace);
    int good = code == 0 && count >= 2 &&
               determinant > 256.0 * 2.2204460492503131e-16 * scale;
    double u = 0.0, v = 0.0;
    if (good) {
        u = (ayy * bx - axy * by) / determinant;
        v = (axx * by - axy * bx) / determinant;
        if (!(isfinite(u) && isfinite(v))) {
            code = 3;
            good = 0;
            u = 0.0;
            v = 0.0;
        }
    }
    cell_vector[cell] = u;
    cell_vector[nc + cell] = v;
    complete[cell] = good;
    status[cell] = code;
}

extern "C" __global__ void v90_protke_green_gauss(
    const int nc,
    const int nf,
    const int* __restrict__ cell_face_offset,
    const int* __restrict__ cell_faces,
    const int* __restrict__ cell_face_sign,
    const int* __restrict__ owner,
    const int* __restrict__ neighbour,
    const double* __restrict__ normal,
    const double* __restrict__ face_length,
    const double* __restrict__ cell_area,
    const double* __restrict__ fluctuation_cell,
    const int* __restrict__ complete,
    const int* __restrict__ active_cell,
    double* __restrict__ ptke,
    int* __restrict__ boundary_cell,
    int* __restrict__ status)
{
    const int cell = blockDim.x * blockIdx.x + threadIdx.x;
    if (cell >= nc) return;
    double dudx = 0.0, dudy = 0.0, dvdx = 0.0, dvdy = 0.0;
    int valid = active_cell[cell] == 1 && complete[cell] == 1;
    int boundary = 0;
    int code = 0;
    for (int slot = cell_face_offset[cell];
         slot < cell_face_offset[cell + 1]; ++slot) {
        const int face = cell_faces[slot];
        if (face < 0 || face >= nf) { code = 1; valid = 0; continue; }
        const int other = owner[face] == cell ? neighbour[face] : owner[face];
        if (other < 0) {
            boundary = 1;
            valid = 0;
            continue;
        }
        if (other >= nc || complete[other] == 0 || active_cell[other] == 0) {
            valid = 0;
            continue;
        }
        const double sign = (double)cell_face_sign[slot];
        const double nx = sign * normal[face];
        const double ny = sign * normal[nf + face];
        const double length = face_length[face];
        const double uf = 0.5 * (fluctuation_cell[cell] +
                                 fluctuation_cell[other]);
        const double vf = 0.5 * (fluctuation_cell[nc + cell] +
                                 fluctuation_cell[nc + other]);
        dudx += uf * nx * length;
        dudy += uf * ny * length;
        dvdx += vf * nx * length;
        dvdy += vf * ny * length;
    }
    double value = 0.0;
    if (valid && code == 0 && cell_area[cell] > 0.0) {
        const double inverse_area = 1.0 / cell_area[cell];
        dudx *= inverse_area;
        dudy *= inverse_area;
        dvdx *= inverse_area;
        dvdy *= inverse_area;
        value = 2.0 * (dudx * dudx + dvdy * dvdy) +
                (dudy + dvdx) * (dudy + dvdx);
        if (!isfinite(value) || value < 0.0) {
            code = 2;
            value = 0.0;
        }
    }
    ptke[cell] = value;
    boundary_cell[cell] = boundary;
    status[cell] = code;
}
