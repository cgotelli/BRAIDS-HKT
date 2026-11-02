// Delft3D-2DH C/M/W roughness conversion for resident CUDA operators.
// Keep arithmetic paired with braids_hydraulic_roughness_module.f90; CuPy
// builds this source with --fmad=false and precise division/square root.
__device__ __forceinline__ bool br_roughness_state(
    int mode, double depth, double parameter, double gravity,
    double minimum_work_depth, int waqua_white_colebrook,
    double *chezy, double *z0, double *drag)
{
    // FLOW hydraulic roughness uses 0.41; Soulsby separately uses 0.40.
    // Locked INIPHY uses an unsuffixed default-real literal before promotion.
    const double von_karman=(double)0.41f;
    const double euler_number=2.7182818284590452353602874713526625;
    *chezy=0.0; *z0=0.0; *drag=0.0;
    if(!isfinite(depth) || !isfinite(parameter) || !isfinite(gravity) ||
       !isfinite(minimum_work_depth) || depth<0.0 || parameter<=0.0 ||
       gravity<=0.0 || minimum_work_depth<0.0) return false;
    if(depth==0.0) return true;
    const double work_depth=fmax(depth,minimum_work_depth);
    if(mode==1) {
        *chezy=parameter;
    } else if(mode==2) {
        *chezy=pow(work_depth,1.0/6.0)/parameter;
    } else if(mode==3) {
        if(waqua_white_colebrook) {
            const double ratio=fmax(0.5,work_depth/parameter);
            *chezy=18.0*log10(12.0*ratio);
        } else {
            *chezy=sqrt(gravity)*log1p(work_depth/(euler_number*parameter/30.0))/von_karman;
        }
    } else {
        return false;
    }
    if(!isfinite(*chezy) || *chezy<=0.0) return false;
    const double x=von_karman*(*chezy)/sqrt(gravity);
    const double negative_exp=exp(-x);
    // Avoid a host-header dependency in the CuPy/NVRTC build path.
    const double smallest_normal=2.2250738585072014e-308;
    *z0=fmax(smallest_normal,work_depth*negative_exp/(euler_number*(1.0-negative_exp)));
    *drag=gravity/((*chezy)*(*chezy));
    return isfinite(*z0) && isfinite(*drag) && *z0>0.0 && *drag>0.0;
}

extern "C" __global__ void hydraulic_roughness_state(
    long long count, int mode, const double *depth, const double *parameter,
    double gravity, double minimum_work_depth, int waqua_white_colebrook,
    double *chezy, double *z0, double *drag, int *status)
{
    const long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=count) return;
    status[i]=br_roughness_state(mode,depth[i],parameter[i],gravity,
        minimum_work_depth,waqua_white_colebrook,&chezy[i],&z0[i],&drag[i]) ? 0 : 1;
}

extern "C" __global__ void hydraulic_roughness_shear(
    long long count, int mode, const double *depth, const double *speed,
    const double *parameter, double density, double gravity,
    double minimum_work_depth, int waqua_white_colebrook,
    double *shear, double *chezy, double *z0, int *status)
{
    const long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=count) return;
    double drag=0.0;
    const bool rough=br_roughness_state(mode,depth[i],parameter[i],gravity,
        minimum_work_depth,waqua_white_colebrook,&chezy[i],&z0[i],&drag);
    const bool valid=rough && isfinite(speed[i]) && speed[i]>=0.0 &&
                     isfinite(density) && density>0.0;
    shear[i]=valid ? density*drag*speed[i]*speed[i] : 0.0;
    status[i]=(valid && isfinite(shear[i]) && shear[i]>=0.0) ? 0 : 1;
}

extern "C" __global__ void hydraulic_roughness_friction_rate(
    long long count, int mode, const double *depth, const double *speed,
    const double *parameter, double gravity, double minimum_work_depth,
    int waqua_white_colebrook, double *rate, int *status)
{
    const long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=count) return;
    double chezy=0.0,z0=0.0,drag=0.0;
    const bool rough=br_roughness_state(mode,depth[i],parameter[i],gravity,
        minimum_work_depth,waqua_white_colebrook,&chezy,&z0,&drag);
    const bool valid=rough && depth[i]>0.0 && isfinite(speed[i]) && speed[i]>=0.0;
    rate[i]=valid ? drag*speed[i]/depth[i] : 0.0;
    status[i]=(valid && isfinite(rate[i]) && rate[i]>=0.0) ? 0 : 1;
}
