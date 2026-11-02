// Resident current-only FLOW MaximumWaterdepth characteristic-velocity stage.
// The CSR support table generalises the four structured velocity-point depths
// without changing the source max/divide operation.

extern "C" __global__ void maximum_waterdepth_v66(
    const long long ncell,const long long nlink,
    const double *cell_depth,const double *transport_u,const double *transport_v,
    const int *active,const long long *support_offset,
    const double *support_depth,const int *support_active,
    const int enabled,const double direction_epsilon,
    double *observation_depth,double *characteristic_u,double *characteristic_v,
    double *characteristic_speed,double *observation_height,int *status)
{
    const long long cell=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(cell>=ncell) return;
    observation_depth[cell]=0.0;
    characteristic_u[cell]=0.0;
    characteristic_v[cell]=0.0;
    characteristic_speed[cell]=0.0;
    observation_height[cell]=0.0;
    status[cell]=0;
    const long long first=support_offset[cell];
    const long long last=support_offset[cell+1];
    if(first<0 || first>last || last>nlink) { status[cell]=1; return; }
    if(active[cell]==0) return;
    if(active[cell]!=1 || !isfinite(cell_depth[cell]) || cell_depth[cell]<=0.0 ||
       !isfinite(transport_u[cell]) || !isfinite(transport_v[cell]) ||
       !isfinite(direction_epsilon) || direction_epsilon<=0.0) {
        status[cell]=1; return;
    }
    double h=cell_depth[cell];
    if(enabled!=0) {
        for(long long item=first;item<last;++item) {
            if((support_active[item]!=0 && support_active[item]!=1) ||
               !isfinite(support_depth[item]) || support_depth[item]<0.0) {
                status[cell]=1; return;
            }
            if(support_active[item]==1) h=fmax(h,support_depth[item]);
        }
    }
    double u=transport_u[cell]/h;
    double v=transport_v[cell]/h;
    if(fabs(u)<direction_epsilon) u=0.0;
    if(fabs(v)<direction_epsilon) v=0.0;
    observation_depth[cell]=h;
    characteristic_u[cell]=u;
    characteristic_v[cell]=v;
    characteristic_speed[cell]=hypot(u,v);
    // Locked dwnvel uses default-real 0.368 promoted to fp.
    observation_height[cell]=0.3680000007152557373046875*h;
}
