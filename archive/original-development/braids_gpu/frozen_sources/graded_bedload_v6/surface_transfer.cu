#define SURFACE_DBL_EPSILON 2.2204460492503131e-16

extern "C" __global__ void surface_donor_scales(
    int nc, int nf, int ng, const double* active, const double* proposed,
    const double* area, const int* offset, const int* faces, const int* signs,
    double* scale, int* status) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nc * ng) return;
    const int g = idx / nc, c = idx - g * nc;
    const double available = active[idx] * area[c];
    double outgoing = 0.0;
    int bad = !isfinite(active[idx]) || active[idx] < 0.0;
    for (int k = offset[c]; k < offset[c + 1]; ++k) {
        const int f = faces[k];
        const double candidate = proposed[g * nf + f];
        if (!isfinite(candidate)) { bad = 1; continue; }
        const double signed_mass = signs[k] * candidate;
        outgoing += fmax(signed_mass, 0.0);
    }
    if (!isfinite(outgoing) || !isfinite(available)) bad = 1;
    double value = 1.0;
    if (!bad && outgoing > available && outgoing > 0.0)
        value = fmax(0.0, available / outgoing) * (1.0 - 64.0 * SURFACE_DBL_EPSILON);
    scale[idx] = bad ? 0.0 : value;
    status[idx] = bad;
}

extern "C" __global__ void surface_limit_faces(
    int nc, int nf, int ng, const double* proposed, const int* owner,
    const int* neighbour, const double* scale, double* limited, int* status) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nf * ng) return;
    const int g = idx / nf, f = idx - g * nf;
    const double value = proposed[idx];
    const int donor = value > 0.0 ? owner[f] : neighbour[f];
    const int bad = !isfinite(value) || (neighbour[f] < 0 && value != 0.0) ||
                    (value != 0.0 && (donor < 0 || donor >= nc));
    limited[idx] = bad ? 0.0 : (value == 0.0 ? 0.0 : value * scale[g * nc + donor]);
    status[idx] = bad;
}

extern "C" __global__ void surface_gather(
    int nc, int nf, int ng, const double* active, const double* area,
    const int* offset, const int* faces, const int* signs,
    const double* limited, int* status, double* next) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nc * ng) return;
    const int g = idx / nc, c = idx - g * nc;
    double change = 0.0;
    for (int k = offset[c]; k < offset[c + 1]; ++k)
        change -= signs[k] * limited[g * nf + faces[k]];
    double value = active[idx] + change / area[c];
    if (value < 0.0 && fabs(value) * area[c] <= 128.0 * SURFACE_DBL_EPSILON) value = 0.0;
    const int bad = !isfinite(value) || value < 0.0;
    next[idx] = bad ? 0.0 : value;
    status[idx] |= bad;
}
