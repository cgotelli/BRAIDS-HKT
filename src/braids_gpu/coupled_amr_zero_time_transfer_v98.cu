// Conservative four-child stage fill. One independent parent per CUDA thread.
// No constitutive solve, physical boundary, accepted-state or clock owner.
// NVRTC supplies device math intrinsics; avoid host C-runtime header dependencies.
#define DBL_EPSILON 2.2204460492503130808472633361816e-16

__device__ double v98_transfer_tol(double a, double b=0.0) {
    return 2048.0*DBL_EPSILON*fmax(1.0,fmax(fabs(a),fabs(b)));
}

__device__ void v98_close_lane(double parent, double area0, const double *area,
    double *values, int correction, bool positive, int *status) {
    double target=parent*area0, sum=0.0;
    for(int k=0;k<4;k++) sum+=values[k]*area[k];
    values[correction]+=(target-sum)/area[correction];
    sum=0.0;
    for(int k=0;k<4;k++) sum+=values[k]*area[k];
    values[correction]+=(target-sum)/area[correction];
    double scale=fmax(1.0,fabs(target));
    for(int k=0;k<4;k++) scale=fmax(scale,fabs(values[k]));
    for(int k=0;k<4;k++) {
        if(!isfinite(values[k]) || (positive && values[k]<-2048.0*DBL_EPSILON*scale))
            *status|=16;
        if(positive && values[k]<0.0) values[k]=0.0;
    }
    sum=0.0;
    for(int k=0;k<4;k++) sum+=values[k]*area[k];
    if(!isfinite(sum) || fabs(sum-target)>v98_transfer_tol(target,sum)) *status|=16;
}

extern "C" __global__ void amr_zero_time_transfer_v98(
    int np, int ng, double dry, const double *parent_area,
    const double *child_area, const double *child_bed,
    const double *parent_q, const double *parent_sus,
    const double *parent_stock, const double *parent_fluid,
    double *child_q, double *child_sus, double *child_stock,
    double *child_fluid, double *restricted_q, double *restricted_sus,
    double *restricted_stock, double *restricted_fluid, int *parent_status) {
    int p=blockDim.x*blockIdx.x+threadIdx.x;
    if(p>=np) return;
    int status=0, order[4]={0,1,2,3};
    double area[4], bed[4], h[4]={0,0,0,0}, area_sum=0.0;
    double area0=parent_area[p], h0=parent_q[p];
    for(int k=0;k<4;k++) {
        area[k]=child_area[p*4+k]; bed[k]=child_bed[p*4+k];
        if(!isfinite(area[k]) || area[k]<=0.0 || !isfinite(bed[k])) status|=1;
        area_sum+=area[k];
    }
    if(!isfinite(area0) || area0<=0.0 ||
        fabs(area_sum-area0)>v98_transfer_tol(area0,area_sum)) status|=1;
    if(!isfinite(h0) || h0<0.0) status|=2;
    for(int lane=1;lane<3;lane++) {
        double q=parent_q[lane*np+p];
        if(!isfinite(q) || (h0<=dry && q!=0.0)) status|=2;
    }
    for(int g=0;g<ng;g++) {
        double s=parent_sus[g*np+p], b=parent_stock[g*np+p];
        if(!isfinite(s) || !isfinite(b) || s<0.0 || b<0.0 ||
            (h0<=dry && s!=0.0)) status|=2;
    }
    for(int lane=0;lane<2;lane++) {
        double f=parent_fluid[lane*np+p];
        if(!isfinite(f) || (h0<=dry && f!=0.0)) status|=2;
        if(lane==1 && f<0.0) status|=32;
    }
    parent_status[p]=status;
    if(status) return;
    // Stable insertion sort by the wetting threshold (dry is shared).
    for(int k=1;k<4;k++) {
        int value=order[k], j=k;
        while(j>0 && bed[order[j-1]]+dry>bed[value]+dry) {
            order[j]=order[j-1]; --j;
        }
        order[j]=value;
    }
    double volume=h0*area0;
    if(!isfinite(volume)) status|=2;
    if(volume>0.0 && !status) {
        bool found=false;
        for(int count=1;count<=4;count++) {
            double weighted_bed=0.0, active_area=0.0;
            for(int j=0;j<count;j++) {
                int k=order[j]; weighted_bed+=area[k]*bed[k]; active_area+=area[k];
            }
            double eta=(volume+weighted_bed)/active_area;
            bool lower=isfinite(eta);
            for(int j=0;j<count;j++) lower=lower && eta>bed[order[j]]+dry;
            bool upper=count==4 || eta<=bed[order[count]]+dry;
            if(lower && upper) {
                int deepest=order[0];
                for(int j=0;j<count;j++) h[order[j]]=eta-bed[order[j]];
                for(int repeat=0;repeat<2;repeat++) {
                    double sum=0.0;
                    for(int k=0;k<4;k++) sum+=h[k]*area[k];
                    h[deepest]+=(volume-sum)/area[deepest];
                }
                found=true; break;
            }
        }
        if(!found) status|=4;
    }
    double filled=0.0;
    int correction=0;
    bool have_wet=false;
    for(int k=0;k<4;k++) {
        if(!isfinite(h[k]) || h[k]<0.0 || (h[k]>0.0 && h[k]<=dry)) status|=8;
        if(!have_wet && h[k]>dry) { correction=k; have_wet=true; }
        filled+=h[k]*area[k]; child_q[p*4+k]=h[k];
    }
    if(fabs(filled-volume)>v98_transfer_tol(volume,filled)) status|=8;
    restricted_q[p]=filled/area0;
    // Momentum and hT/hS preserve velocity/concentration in wet children.
    for(int lane=0;lane<4;lane++) {
        const double *source=lane<2 ? parent_q : parent_fluid;
        int row=lane<2 ? lane+1 : lane-2;
        double value=source[row*np+p], values[4];
        for(int k=0;k<4;k++) values[k]=h0>dry ? (value/h0)*h[k] : 0.0;
        v98_close_lane(value,area0,area,values,correction,false,&status);
        double sum=0.0;
        for(int k=0;k<4;k++) {
            if(h[k]<=dry && values[k]!=0.0) status|=16;
            if(lane<2) child_q[(row*np+p)*4+k]=values[k];
            else child_fluid[(row*np+p)*4+k]=values[k];
            sum+=values[k]*area[k];
        }
        if(lane<2) restricted_q[row*np+p]=sum/area0;
        else restricted_fluid[row*np+p]=sum/area0;
    }
    for(int g=0;g<ng;g++) {
        double s0=parent_sus[g*np+p], b0=parent_stock[g*np+p], s[4], b[4];
        for(int k=0;k<4;k++) { s[k]=h0>dry ? (s0/h0)*h[k] : 0.0; b[k]=b0; }
        v98_close_lane(s0,area0,area,s,correction,true,&status);
        v98_close_lane(b0,area0,area,b,3,true,&status);
        double sum_s=0.0, sum_b=0.0;
        for(int k=0;k<4;k++) {
            if(h[k]<=dry) { b[k]+=s[k]; s[k]=0.0; }
            child_sus[(g*np+p)*4+k]=s[k]; child_stock[(g*np+p)*4+k]=b[k];
            sum_s+=s[k]*area[k]; sum_b+=b[k]*area[k];
        }
        restricted_sus[g*np+p]=sum_s/area0;
        restricted_stock[g*np+p]=sum_b/area0;
        if(fabs((sum_s+sum_b)/area0-(s0+b0))>v98_transfer_tol(s0+b0)) status|=16;
    }
    parent_status[p]=status;
}
