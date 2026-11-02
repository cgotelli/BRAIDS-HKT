// Small lifecycle-only kernels for the authenticated Q3-H6a direct-vector path.
// Constitutive transport, face reconstruction and Exner remain in their
// independently authenticated modules; this file owns only deterministic
// inventory fill and resident content tokens.

__device__ __forceinline__ unsigned long long direct_v66_mix64(
    unsigned long long value)
{
    value ^= value >> 30;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27;
    value *= 0x94d049bb133111ebULL;
    value ^= value >> 31;
    return value;
}

extern "C" __global__ void direct_v66_digest_fp64(
    const long long count,const unsigned long long seed,
    const double *values,unsigned long long *digest)
{
    const long long point=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(point>=count) return;
    const unsigned long long bits=(unsigned long long)__double_as_longlong(values[point]);
    const unsigned long long index=(unsigned long long)point;
    atomicXor(&digest[0],direct_v66_mix64(bits^seed^
              (index+1ULL)*0x9e3779b97f4a7c15ULL));
    atomicXor(&digest[1],direct_v66_mix64(bits^~seed^
              (index+1ULL)*0xd6e8feb86659fd93ULL));
}

extern "C" __global__ void direct_v66_inventory_fill(
    const int ncell,const int nface,const int nadjacency,const int nclass,
    const int *offset,const int *adjacency_faces,
    const int *owner,const int *neighbour,const int *role,
    const int *donor,const double *provisional_flux,const double *face_length,
    const double *area,const double *old_mass,const double *active_fraction,
    const double dt,const double morphological_factor,
    double *fill,double *availability,int *status)
{
    const long long point=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    const long long count=(long long)nclass*ncell;
    if(point>=count) return;
    const int grain=(int)(point/ncell);
    const int cell=(int)(point-(long long)grain*ncell);
    fill[point]=0.0;availability[point]=0.0;
    if(offset[cell]<0 || offset[cell]>offset[cell+1] ||
       offset[cell+1]>nadjacency || !isfinite(area[cell]) || area[cell]<=0.0 ||
       !isfinite(old_mass[point]) || old_mass[point]<0.0 ||
       !isfinite(active_fraction[point]) || active_fraction[point]<0.0 ||
       !isfinite(dt) || dt<=0.0 || !isfinite(morphological_factor) ||
       morphological_factor<0.0) {
        status[cell]=1;return;
    }
    double demand=0.0;
    for(int item=offset[cell];item<offset[cell+1];++item) {
        const int face=adjacency_faces[item];
        if(face<0 || face>=nface) { status[cell]=1;return; }
        const int face_role=role[face];
        const int selected=donor[(long long)grain*nface+face];
        int donor_cell=-1;
        if(selected==1 || selected==3) donor_cell=owner[face];
        else if(selected==2) donor_cell=neighbour[face];
        const bool removes=(face_role==0 || face_role==2 || face_role==3);
        if(removes && donor_cell==cell) {
            const double flux=provisional_flux[(long long)grain*nface+face];
            if(!isfinite(flux) || !isfinite(face_length[face]) ||
               face_length[face]<0.0) { status[cell]=1;return; }
            demand+=fabs(flux)*face_length[face]*dt*morphological_factor/area[cell];
        }
    }
    double factor=old_mass[point]>0.0 ? 1.0 : 0.0;
    if(demand>0.0) factor=fmin(factor,old_mass[point]/demand);
    if(!isfinite(factor) || factor<0.0 || factor>1.0) {
        status[cell]=1;return;
    }
    fill[point]=factor;
    availability[point]=active_fraction[point]*factor;
}
