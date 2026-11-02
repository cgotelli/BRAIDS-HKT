// SPDX-License-Identifier: GPL-3.0-only
// Resident CUDA surface identity/statistics kernels for bounded Q3/v69.

#define V69_MAX_CLASSES 32
#define V69_MAX_POINTS 101

extern "C" __global__ void surface_resolve_v69(
    const long long ncell, const int nclass,
    const double *mass, const double *density, const double *retained,
    const int retained_supplied, double *identity, int *has_stock, int *status)
{
    const long long cell=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(cell>=ncell) return;
    bool valid=nclass>0 && nclass<=V69_MAX_CLASSES;
    double total=0.0;
    for(int grain=0;grain<nclass;++grain) {
        const double value=mass[(long long)grain*ncell+cell];
        valid=valid && isfinite(value) && value>=0.0 &&
            isfinite(density[grain]) && density[grain]>0.0;
        if(valid) total+=value/density[grain];
    }
    if(!valid || !isfinite(total)) { status[cell]=1;has_stock[cell]=0;return; }
    if(total>0.0) {
        has_stock[cell]=1;
        for(int grain=0;grain<nclass;++grain)
            identity[(long long)grain*ncell+cell]=
                mass[(long long)grain*ncell+cell]/density[grain]/total;
    } else {
        has_stock[cell]=0;
        double sum=0.0;
        valid=retained_supplied!=0;
        for(int grain=0;grain<nclass;++grain) {
            const double value=retained[(long long)grain*ncell+cell];
            valid=valid && isfinite(value) && value>=0.0;
            sum+=value;
            identity[(long long)grain*ncell+cell]=value;
        }
        valid=valid && fabs(sum-1.0)<=
            512.0*2.2204460492503131e-16*nclass;
    }
    status[cell]=valid ? 0 : 1;
}

extern "C" __global__ void surface_availability_v69(
    const long long count, const double *mass, const double *identity,
    const double *outgoing, double *fill, double *availability, int *status)
{
    const long long index=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(index>=count) return;
    const double m=mass[index],f=identity[index],demand=outgoing[index];
    bool valid=isfinite(m) && m>=0.0 && isfinite(f) && f>=0.0 &&
        isfinite(demand) && demand>=0.0;
    double limiter=0.0;
    if(valid && m>0.0) limiter=demand>0.0 ? fmin(1.0,m/demand) : 1.0;
    fill[index]=limiter;
    availability[index]=f*limiter;
    status[index]=valid ? 0 : 1;
}

extern "C" __global__ void surface_statistics_v69(
    const long long ncell, const int nclass, const int maxpoint,
    const double *fraction, const double *mean_diameter,
    const double *median_diameter, const int *sediment_type,
    const int max_mud_type, const int min_dxx_type, const int *count,
    const double *probability, const double *log_table,
    double *statistics, int *has_nonmud, int *status)
{
    const long long cell=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(cell>=ncell) return;
    bool valid=nclass>0 && nclass<=V69_MAX_CLASSES &&
        maxpoint>0 && maxpoint<=V69_MAX_POINTS;
    double fracsum=0.0,fracdiam=0.0,fracnonmud=0.0,mud=0.0;
    for(int grain=0;grain<nclass;++grain) {
        const double share=fraction[(long long)grain*ncell+cell];
        valid=valid && isfinite(share) && share>=0.0;
        fracsum+=share;
        if(sediment_type[grain]>=min_dxx_type) fracdiam+=share;
        if(sediment_type[grain]>max_mud_type) fracnonmud+=share;
        else mud+=share;
    }
    valid=valid && fabs(fracsum-1.0)<=
        512.0*2.2204460492503131e-16*nclass;
    double dm=0.0,dg=0.0,dgsd=0.0,sand=0.0;
    if(fracdiam>0.0) {
        double logdg=0.0;
        for(int grain=0;grain<nclass;++grain)
            if(sediment_type[grain]>=min_dxx_type) {
                const double weight=fraction[(long long)grain*ncell+cell]/fracdiam;
                dm+=weight*mean_diameter[grain];
                logdg+=weight*log(median_diameter[grain]);
            }
        dg=exp(logdg);
        for(int grain=0;grain<nclass;++grain)
            if(sediment_type[grain]>=min_dxx_type &&
               fraction[(long long)grain*ncell+cell]>0.0) {
                const double weight=fraction[(long long)grain*ncell+cell]/fracdiam;
                const double delta=log(median_diameter[grain])-logdg;
                dgsd+=weight*delta*delta;
            }
        dgsd=exp(sqrt(dgsd));
    }
    if(fracnonmud>0.0)
        for(int grain=0;grain<nclass;++grain)
            if(sediment_type[grain]>max_mud_type && mean_diameter[grain]<2.0e-3)
                sand+=fraction[(long long)grain*ncell+cell]/fracnonmud;

    int stage[V69_MAX_CLASSES];
    for(int grain=0;grain<nclass;++grain) stage[grain]=0;
    const double requested[3]={0.10,0.50,0.90};
    double dxx[3]={0.0,0.0,0.0};
    int wanted=0,iterations=0;
    double required=requested[0],previous=0.0,accumulated=0.0;
    while(valid && wanted<3 && required>accumulated) {
        if(++iterations>nclass*maxpoint+nclass+12) { valid=false;break; }
        double logdiam=999.999,density=0.0;
        int trigger=-1;
        for(int grain=0;grain<nclass;++grain) {
            const int s=stage[grain],n=count[grain];
            valid=valid && n>=0 && n<=maxpoint;
            if(s<n) {
                if(s>0) {
                    const long long left=(long long)grain*maxpoint+s-1;
                    const long long right=left+1;
                    const double factor=fracdiam>0.0 ?
                        fraction[(long long)grain*ncell+cell]/fracdiam : 1.0;
                    density+=factor*(probability[right]-probability[left])/
                        (log_table[right]-log_table[left]);
                }
                const double candidate=log_table[(long long)grain*maxpoint+s];
                if(candidate<logdiam) { logdiam=candidate;trigger=grain; }
            }
        }
        if(trigger<0) {
            while(wanted<3) dxx[wanted++]=exp(previous);
            break;
        }
        double interval=density*(logdiam-previous);
        while(wanted<3 && interval>=required-accumulated) {
            if(interval>0.0)
                previous+=(logdiam-previous)*(required-accumulated)/interval;
            accumulated=required;dxx[wanted]=exp(previous);++wanted;
            if(wanted>=3) break;
            required=requested[wanted];interval=density*(logdiam-previous);
        }
        if(wanted>=3) break;
        accumulated+=interval;previous=logdiam;
        if(stage[trigger]==0) {
            const int n=count[trigger];
            const double first=log_table[(long long)trigger*maxpoint];
            const double last=log_table[(long long)trigger*maxpoint+n-1];
            if(last<=first) {
                stage[trigger]=n>5 ? n : 5;
                const double point=fraction[(long long)trigger*ncell+cell];
                // Right-continuous inverse CDF.  The locked compdiam strict
                // comparison leaves an exact Dirac-edge Dxx unassigned.
                while(wanted<3 && point>=required-accumulated) {
                    dxx[wanted]=exp(logdiam);++wanted;
                    if(wanted<3) required=requested[wanted];
                }
                accumulated+=point;
            } else stage[trigger]=1;
        } else ++stage[trigger];
    }
    statistics[cell]=dm;
    statistics[ncell+cell]=dg;
    statistics[2*ncell+cell]=dxx[0];
    statistics[3*ncell+cell]=dxx[1];
    statistics[4*ncell+cell]=dxx[2];
    statistics[5*ncell+cell]=dgsd;
    statistics[6*ncell+cell]=sand;
    statistics[7*ncell+cell]=mud;
    has_nonmud[cell]=fracnonmud>0.0 ? 1 : 0;
    for(int row=0;row<8;++row)
        valid=valid && isfinite(statistics[(long long)row*ncell+cell]) &&
            statistics[(long long)row*ncell+cell]>=0.0;
    status[cell]=valid ? 0 : 1;
}
