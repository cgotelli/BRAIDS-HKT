// Fixed-grid suspended-sediment transaction support.  All trial writes remain
// separate from accepted storage until the caller's global status is known.
extern "C" __global__ void sr_stage_ledger(const double *incoming,double *trial,
                                             long long count,int stage,int *status) {
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=count)return;
    double value=stage<3?incoming[i]/3.:incoming[i];
    double total=stage==0?value:trial[i]+value;
    status[i]=isfinite(incoming[i])&&isfinite(total)?0:1;
    trial[i]=status[i]==0?total:0.;
}

// The two columns are signed gains in the suspended and active reservoirs.
// Their sum must vanish for each grain, including mixed settling/erosion.
extern "C" __global__ void sr_exchange_ledger(int nc,int ng,const double *area,
        const double *before_s,const double *before_b,const double *after_s,
        const double *after_b,double *ledger,int *status) {
    int g=blockIdx.x*blockDim.x+threadIdx.x;if(g>=ng)return;
    double water=0.,bed=0.,scale=0.;int bad=0;
    for(int c=0;c<nc;++c){
        int i=g*nc+c;double a=area[c],s0=before_s[i],b0=before_b[i];
        double s1=after_s[i],b1=after_b[i];
        if(!isfinite(a)||a<=0.||!isfinite(s0)||!isfinite(b0)||
           !isfinite(s1)||!isfinite(b1)||s0<0.||b0<0.||s1<0.||b1<0.)bad=1;
        water+=(s1-s0)*a;bed+=(b1-b0)*a;
        scale+=(fabs(s0)+fabs(b0)+fabs(s1)+fabs(b1))*a;
    }
    if(!isfinite(water)||!isfinite(bed)||
       fabs(water+bed)>512.*2.2204460492503131e-16*fmax(scale,1.))bad=1;
    ledger[2*g]=water;ledger[2*g+1]=bed;status[g]=bad;
}

extern "C" __global__ void sr_commit_field(const double *trial,double *accepted,
                                               long long count,const int *failure) {
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count&&failure[0]==0)accepted[i]=trial[i];
}

extern "C" __global__ void sr_commit_ledger(const double *trial,double *accepted,
                                                long long count,const int *failure) {
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count&&failure[0]==0)accepted[i]+=trial[i];
}
