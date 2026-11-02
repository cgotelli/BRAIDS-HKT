extern "C" __global__ void v87_detvic_current_only(
    const int nc,
    const int hles_enabled,
    const int elder_enabled,
    const double gamma,
    const double sigmat,
    const double flp,
    const double vicmol,
    const double dicmol,
    const double von_karman,
    const double gravity,
    const double dry_tolerance,
    const double* __restrict__ cell_area,
    const double* __restrict__ depth,
    const double* __restrict__ velocity,
    const double* __restrict__ chezy,
    const double* __restrict__ ptke,
    const int* __restrict__ active,
    double* __restrict__ hles_vicuv,
    double* __restrict__ elder_vicuv,
    double* __restrict__ dynamic_vicuv,
    double* __restrict__ dynamic_dicuv,
    int* __restrict__ status)
{
    const int cell = blockDim.x * blockIdx.x + threadIdx.x;
    if (cell >= nc) return;

    int code = 0;
    const double h = depth[cell];
    const double u = velocity[cell];
    const double v = velocity[nc + cell];
    const double c = chezy[cell];
    const double production = ptke[cell];
    const double area = cell_area[cell];
    if (!(isfinite(h) && isfinite(u) && isfinite(v) && isfinite(c) &&
          isfinite(production) && isfinite(area)) || h < 0.0 || c < 0.0 ||
          production < 0.0 || area <= 0.0 ||
          !(active[cell] == 0 || active[cell] == 1)) {
        code = 1;
    }

    double hles = 0.0;
    double elder = 0.0;
    double visc = 0.0;
    double diff = 0.0;
    if (code == 0 && hles_enabled != 0 && active[cell] != 0) {
        const double safe_depth = fmax(dry_tolerance, h);
        const double safe_chezy = fmax(1.0, c);
        const double speed = hypot(u, v);
        const double b = 0.75 * gravity * speed /
                         (safe_depth * safe_chezy * safe_chezy);
        const double ks = 3.141592653589793238462643383279502884 * flp /
                          sqrt(area);
        hles = (sqrt(gamma * gamma * sigmat * sigmat * production + b * b)
                - b) / (ks * ks);
        if (elder_enabled != 0) {
            const double ustar = speed * sqrt(gravity) / safe_chezy;
            elder = von_karman * ustar * safe_depth / 6.0;
        }
        visc = hles + elder + vicmol;
        diff = hles + elder + dicmol;
        if (!(isfinite(hles) && isfinite(elder) && isfinite(visc) &&
              isfinite(diff)) || hles < 0.0 || elder < 0.0 || visc < 0.0 ||
              diff < 0.0) {
            code = 2;
            hles = 0.0;
            elder = 0.0;
            visc = 0.0;
            diff = 0.0;
        }
    }
    hles_vicuv[cell] = hles;
    elder_vicuv[cell] = elder;
    dynamic_vicuv[cell] = visc;
    dynamic_dicuv[cell] = diff;
    status[cell] = code;
}
