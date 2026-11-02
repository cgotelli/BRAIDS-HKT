extern "C" __global__ void braids_amr_bed_selector_v100(
    const int nc, const int ng, const int mode, const int* combine,
    const double* bed, const double* water, const double* diffused,
    const double* transfer, const double* cleanup, double* exchange,
    double* divergence, double* selected, double* final_bed, int* status) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i >= nc*ng) return;
    exchange[i] = combine[i/nc] ? -transfer[i] : 0.0;
    divergence[i] = (combine[i/nc] && mode == 1) ? diffused[i]-water[i] : 0.0;
    selected[i] = mode == 0 ? exchange[i] : (mode == 1 ? divergence[i] : 0.0);
    final_bed[i] = bed[i]+selected[i]+cleanup[i];
    status[i] = final_bed[i] < 0.0;
}
