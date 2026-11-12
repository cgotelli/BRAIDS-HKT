// Isolated FP64 SSPRK(4,3) and hydraulic-only split Manning helpers.
// Not a full solver. No fast-math; compile with --fmad=false.
// NVRTC supplies device math builtins; avoid requiring host CRT headers.

extern "C" __global__ void ti_merge_flag(const int* other, int* flag) {
    if (threadIdx.x == 0 && blockIdx.x == 0 && other[0] > flag[0]) flag[0] = other[0];
}

extern "C" __global__ void ti_merge_cfl(const double* other, double* peak, int* flag) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (!isfinite(other[0]) || other[0] < 0.0) flag[0] = 4;
        else peak[0] = fmax(peak[0], other[0]);
    }
}

extern "C" __global__ void ti_prepare(double* q, long long n, double dry,
                                      double speed_limit, int* status) {
    const long long c = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (c >= n) return;
    const double h=q[c], x=q[n+c], y=q[2*n+c];
    status[c]=0;
    if (!isfinite(h) || !isfinite(x) || !isfinite(y) || h<0.0) {status[c]=2; return;}
    if (h<=dry) {q[n+c]=0.0; q[2*n+c]=0.0; return;}
    const double speed=sqrt(x*x+y*y)/h;
    if (!isfinite(speed)) {status[c]=2; return;}
    if (speed>speed_limit) {q[n+c]=x*(speed_limit/speed); q[2*n+c]=y*(speed_limit/speed);}
}

extern "C" __global__ void ti_capture(const double* incoming, const double* base,
                                      const int* flag, long long n, double* output, int* status) {
    const long long c = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (c>=n) return;
    const double h=incoming[c], x=incoming[n+c], y=incoming[2*n+c];
    const bool valid=isfinite(h) && isfinite(x) && isfinite(y) && h>=0.0;
    status[c]=valid ? 0 : 2;
    const bool keep=valid && flag[0]==0;
    output[c]=keep ? h : base[c];
    output[n+c]=keep ? x : base[n+c];
    output[2*n+c]=keep ? y : base[2*n+c];
}

extern "C" __global__ void ti_combine(const double* base, const double* third,
                                      long long n, double* output) {
    const long long i=blockIdx.x*(long long)blockDim.x+threadIdx.x;
    if (i<3*n) output[i]=2.0*base[i]/3.0+third[i]/3.0;
}

extern "C" __global__ void ti_ledger(const double* incoming, double* accumulated,
                                     long long nb, int stage, int* status) {
    const long long i=blockIdx.x*(long long)blockDim.x+threadIdx.x;
    if (i>=nb) return;
    const double flux=incoming[i];
    const double value=stage<3 ? flux/3.0 : flux;
    const double total=stage==0 ? value : accumulated[i]+value;
    status[i]=isfinite(flux) && isfinite(total) ? 0 : 3;
    accumulated[i]=status[i]==0 ? total : 0.0;
}

extern "C" __global__ void ti_check_ledger(const double* ledger, long long nb, int* status) {
    const long long i=blockIdx.x*(long long)blockDim.x+threadIdx.x;
    if (i<nb) status[i]=isfinite(ledger[i]) ? 0 : 3;
}

extern "C" __global__ void ti_check_ledger_sum(const double* accepted, const double* trial,
                                              long long nb, int* status) {
    const long long i=blockIdx.x*(long long)blockDim.x+threadIdx.x;
    if (i<nb) status[i]=isfinite(accepted[i]+trial[i]) ? 0 : 3;
}

extern "C" __global__ void ti_manning(double* q, const double* roughness, long long n,
                                      double dt, double grav, double dry, double chezy_min,
                                      int* status) {
    const long long c=blockIdx.x*(long long)blockDim.x+threadIdx.x;
    if (c>=n) return;
    const double h=q[c], x=q[n+c], y=q[2*n+c], rough=roughness[c];
    status[c]=0;
    if (!isfinite(h) || !isfinite(x) || !isfinite(y) || h<0.0 || !isfinite(rough) || rough<0.0) {
        status[c]=2; return;
    }
    if (h<=dry) {q[n+c]=0.0; q[2*n+c]=0.0; return;}
    const double speed=sqrt(x*x+y*y)/h;
    // source() applies this backward-Euler-like exact uniform-drag map ONCE
    // after all hydraulic stages. Preserve operation order, including pow.
    const double denominator=chezy_min>0.0 && h<chezy_min
        ? h*pow(chezy_min,1.0/3.0) : pow(h,4.0/3.0);
    const double factor=1.0+dt*grav*(rough*rough)*speed/denominator;
    if (!isfinite(speed) || !isfinite(factor) || factor<1.0) {status[c]=2; return;}
    q[n+c]=x/factor; q[2*n+c]=y/factor;
}

extern "C" __global__ void ti_commit_q(const double* trial, double* accepted,
                                       long long n, const int* flag) {
    const long long i=blockIdx.x*(long long)blockDim.x+threadIdx.x;
    if (i<3*n && flag[0]==0) accepted[i]=trial[i];
}

extern "C" __global__ void ti_commit_ledger(const double* trial, double* accepted,
                                            long long nb, const int* flag) {
    const long long i=blockIdx.x*(long long)blockDim.x+threadIdx.x;
    if (i<nb && flag[0]==0) accepted[i]=accepted[i]+trial[i];
}
