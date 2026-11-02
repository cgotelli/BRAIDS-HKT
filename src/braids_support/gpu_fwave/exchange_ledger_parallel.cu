// Deterministic cell-parallel replacement for suspension_resident.cu:sr_exchange_ledger.
// The coupled GPU solver uses these kernels for active noncohesive exchange.
#define TILE 256

__device__ __forceinline__ void exchange_cell(int c,int g,int nc,
        const double *area,const double *before_s,const double *before_b,
        const double *after_s,const double *after_b,
        double *water,double *bed,double *scale,int *bad) {
    int i=g*nc+c;
    double a=area[c],s0=before_s[i],b0=before_b[i];
    double s1=after_s[i],b1=after_b[i];
    if(!isfinite(a)||a<=0.||!isfinite(s0)||!isfinite(b0)||
       !isfinite(s1)||!isfinite(b1)||s0<0.||b0<0.||s1<0.||b1<0.)*bad=1;
    *water+=(s1-s0)*a;
    *bed+=(b1-b0)*a;
    *scale+=(fabs(s0)+fabs(b0)+fabs(s1)+fabs(b1))*a;
}

__device__ __forceinline__ void exchange_reduce(double water,double bed,
        double scale,int bad,double *out_water,double *out_bed,
        double *out_scale,int *out_bad) {
    __shared__ double w[TILE],b[TILE],s[TILE];
    __shared__ int flags[TILE];
    int lane=threadIdx.x;
    w[lane]=water;b[lane]=bed;s[lane]=scale;flags[lane]=bad;
    __syncthreads();
    for(int stride=TILE/2;stride>0;stride>>=1) {
        if(lane<stride) {
            w[lane]+=w[lane+stride];
            b[lane]+=b[lane+stride];
            s[lane]+=s[lane+stride];
            flags[lane]|=flags[lane+stride];
        }
        __syncthreads();
    }
    if(lane==0) {
        *out_water=w[0];*out_bed=b[0];*out_scale=s[0];*out_bad=flags[0];
    }
}

// One block per grain: parallel cell striding and deterministic tree reduction.
extern "C" __global__ void sr_exchange_direct(int nc,int ng,const double *area,
        const double *before_s,const double *before_b,const double *after_s,
        const double *after_b,double *ledger,int *status) {
    int g=blockIdx.x;if(g>=ng)return;
    double water=0.,bed=0.,scale=0.;int bad=0;
    for(int c=threadIdx.x;c<nc;c+=TILE)
        exchange_cell(c,g,nc,area,before_s,before_b,after_s,after_b,
                      &water,&bed,&scale,&bad);
    double wr,br,sr;int flag;
    exchange_reduce(water,bed,scale,bad,&wr,&br,&sr,&flag);
    if(threadIdx.x==0) {
        if(!isfinite(wr)||!isfinite(br)||
           fabs(wr+br)>512.*2.2204460492503131e-16*fmax(sr,1.))flag=1;
        ledger[2*g]=wr;ledger[2*g+1]=br;status[g]=flag;
    }
}

// Stage 1: many grain/tile blocks expose reach-wide parallelism without atomics.
extern "C" __global__ void sr_exchange_partial(int nc,int ng,int nt,
        const double *area,const double *before_s,const double *before_b,
        const double *after_s,const double *after_b,
        double *partial_water,double *partial_bed,double *partial_scale,int *partial_bad) {
    int t=blockIdx.x,g=blockIdx.y,c=t*TILE+threadIdx.x;
    if(t>=nt||g>=ng)return;
    double water=0.,bed=0.,scale=0.;int bad=0;
    if(c<nc)exchange_cell(c,g,nc,area,before_s,before_b,after_s,after_b,
                          &water,&bed,&scale,&bad);
    double wr,br,sr;int flag;
    exchange_reduce(water,bed,scale,bad,&wr,&br,&sr,&flag);
    if(threadIdx.x==0) {
        int i=g*nt+t;
        partial_water[i]=wr;partial_bed[i]=br;
        partial_scale[i]=sr;partial_bad[i]=flag;
    }
}

// Stage 2: ordered partial sum per grain; same signed convention and status gate.
extern "C" __global__ void sr_exchange_finish(int nt,int ng,
        const double *partial_water,const double *partial_bed,
        const double *partial_scale,const int *partial_bad,
        double *ledger,int *status) {
    int g=blockIdx.x;if(g>=ng)return;
    double water=0.,bed=0.,scale=0.;int bad=0;
    for(int t=threadIdx.x;t<nt;t+=TILE) {
        int i=g*nt+t;
        water+=partial_water[i];bed+=partial_bed[i];
        scale+=partial_scale[i];bad|=partial_bad[i];
    }
    double wr,br,sr;int flag;
    exchange_reduce(water,bed,scale,bad,&wr,&br,&sr,&flag);
    if(threadIdx.x==0) {
        if(!isfinite(wr)||!isfinite(br)||
           fabs(wr+br)>512.*2.2204460492503131e-16*fmax(sr,1.))flag=1;
        ledger[2*g]=wr;ledger[2*g+1]=br;status[g]=flag;
    }
}
