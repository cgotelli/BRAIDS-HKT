// Additive v98 validation binding: turn authenticated physical kind-3 faces
// into ordinary reflective kind-2 faces before the sole augmented f-wave call.
// The production boundary owner is deliberately not called by this route.

extern "C" __global__ void v98_close_physical_faces(
    const int *faces, const long long count, const long long nf,
    int *kind, double *trace, int *face_status)
{
    const long long row = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= count) return;
    const int face = faces[row];
    if (face < 0 || face >= nf || kind[face] != 3) {
        if (face >= 0 && face < nf) face_status[face] = 1;
        return;
    }
    kind[face] = 2;
    // A kind-2 face consumes the reconstructed reflective owner state, not a
    // physical boundary trace.  Clear the unused image so stale live forcing
    // cannot be mistaken for accepted evidence.
    for (int component = 0; component < 10; ++component)
        trace[(long long)component * nf + face] = 0.0;
}

extern "C" __global__ void v98_validate_reflective_faces(
    const int *faces, const long long count, const long long nf,
    const int *kind, int *face_status)
{
    const long long row = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= count) return;
    const int face = faces[row];
    if (face < 0 || face >= nf || kind[face] != 2)
        if (face >= 0 && face < nf) face_status[face] = 1;
}
