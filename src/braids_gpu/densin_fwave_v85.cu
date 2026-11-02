// Actual-CUDA Q4/v85 DensIn density and midpoint momentum-source kernels.
extern "C" __global__ void densin_mixture_v85(
    const double* concentration, const double* water_density,
    const double* solid_density, long long ng, long long nc,
    double* mixture, int* status)
{
    const long long cell = (long long)blockDim.x*blockIdx.x + threadIdx.x;
    if (cell >= nc) return;
    double rho = water_density[cell];
    if (!(rho > 0.0) || !isfinite(rho)) { status[cell] = 1; return; }
    for (long long grain=0; grain<ng; ++grain) {
        const double c = concentration[grain*nc + cell];
        const double rhos = solid_density[grain];
        if (!(c >= 0.0) || !isfinite(c) ||
                !(rhos > water_density[cell]) || !isfinite(rhos)) {
            status[cell] = 2; return;
        }
        rho += c*(1.0-water_density[cell]/rhos);
    }
    if (!(rho > 0.0) || !isfinite(rho)) { status[cell] = 3; return; }
    mixture[cell] = rho;
    status[cell] = 0;
}

extern "C" __global__ void densin_face_v85(
    const double* depth, const double* mixture,
    const int* owner, const int* neighbour, const double* distance,
    long long nf, double gravity, double reference_density,
    double* face_acceleration, int* status)
{
    const long long face = (long long)blockDim.x*blockIdx.x + threadIdx.x;
    if (face >= nf) return;
    const int left = owner[face];
    const int right = neighbour[face];
    if (right < 0) {
        face_acceleration[face] = 0.0;
        status[face] = 0;
        return;
    }
    const double value = -gravity*(mixture[right]-mixture[left]) *
        (depth[right]+depth[left])/(4.0*reference_density*distance[face]);
    if (!isfinite(value)) { status[face] = 1; return; }
    face_acceleration[face] = value;
    status[face] = 0;
}

extern "C" __global__ void densin_project_stride_v85(
    const double* face_acceleration, const double* normal,
    const double* face_length, const int* offset, const int* faces,
    const int* signs, const double* midpoint_depth, long long nc,
    long long nf, double* cell_acceleration, double* momentum_rate,
    int* status)
{
    const long long cell = (long long)blockDim.x*blockIdx.x + threadIdx.x;
    if (cell >= nc) return;
    double m00=0.0, m01=0.0, m11=0.0, b0=0.0, b1=0.0;
    for (int slot=offset[cell]; slot<offset[cell+1]; ++slot) {
        const int face=faces[slot];
        const double sign=(double)signs[slot];
        const double nx=sign*normal[face];
        const double ny=sign*normal[nf + face];
        const double directional=sign*face_acceleration[face];
        const double weight=face_length[face];
        m00 += weight*nx*nx;
        m01 += weight*nx*ny;
        m11 += weight*ny*ny;
        b0 += weight*nx*directional;
        b1 += weight*ny*directional;
    }
    const double determinant=m00*m11-m01*m01;
    const double scale=fmax(1.0, fmax(fabs(m00), fmax(fabs(m01), fabs(m11))));
    if (!(determinant > 5.684341886080802e-14*scale*scale) ||
            !isfinite(determinant)) {
        status[cell]=1; return;
    }
    const double ax=(m11*b0-m01*b1)/determinant;
    const double ay=(-m01*b0+m00*b1)/determinant;
    const double h=midpoint_depth[cell];
    if (!(h > 0.0) || !isfinite(h) || !isfinite(ax) || !isfinite(ay)) {
        status[cell]=2; return;
    }
    cell_acceleration[cell]=ax;
    cell_acceleration[nc+cell]=ay;
    momentum_rate[cell]=h*ax;
    momentum_rate[nc+cell]=h*ay;
    status[cell]=0;
}

extern "C" __global__ void densin_compose_v85(
    const double* parent_q, const double* base_rate,
    const double* density_rate, long long nc, double dt,
    double dry_tolerance, double* candidate_q, int* status)
{
    const long long cell = (long long)blockDim.x*blockIdx.x + threadIdx.x;
    if (cell >= nc) return;
    const double h=parent_q[cell]+dt*base_rate[cell];
    const double hu=parent_q[nc+cell]+dt*(base_rate[nc+cell]+density_rate[cell]);
    const double hv=parent_q[2*nc+cell]+dt*(base_rate[2*nc+cell]+density_rate[nc+cell]);
    if (!(h > dry_tolerance) || !isfinite(h) || !isfinite(hu) || !isfinite(hv)) {
        status[cell]=1; return;
    }
    candidate_q[cell]=h;
    candidate_q[nc+cell]=hu;
    candidate_q[2*nc+cell]=hv;
    status[cell]=0;
}
