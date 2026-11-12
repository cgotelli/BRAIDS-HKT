// Native endpoint holding and first knot at/above interior evaluation time.
__device__ double bs_value(const double* t, const double* v, int n,
                           int stride, int offset, double time, double fallback) {
    if (!n) return fallback;
    if (time <= t[offset]) return v[offset];
    if (time >= t[(n-1)*stride+offset]) return v[(n-1)*stride+offset];
    int lo=1, hi=n-1;
    while (lo<hi) {
        int mid=lo+(hi-lo)/2;
        if (time<=t[mid*stride+offset]) hi=mid; else lo=mid+1;
    }
    int left=(lo-1)*stride+offset, right=lo*stride+offset;
    double weight=(time-t[left])/(t[right]-t[left]);
    return (1.0-weight)*v[left]+weight*v[right];
}

extern "C" __global__ void bs_evaluate(
    int nb, int ng, const double* clock, const int* mode, const double* forcing,
    const double* default_c, const double* default_feed,
    const int* nh, const int* ns, const int* nf,
    const double* ht, const double* hq, const double* hs,
    const double* st, const double* sc, const double* sf,
    const double* ft, const double* fc, const double* ff,
    double* times, double* hydraulic, double* concentration, double* feed, int* status) {
    int width=nb>0?nb:1, index=blockIdx.x*blockDim.x+threadIdx.x;
    if (index>=4*width) return;
    int stage=index/width, b=index%width;
    double current=clock[0], dt=clock[1];
    double fraction=stage==0?0.0:(stage==2?1.0:0.5);
    double time=current+fraction*dt;
    bool valid=isfinite(current)&&isfinite(dt)&&dt>0&&
        isfinite(current+dt)&&current+dt>current;
    if (b==0) times[stage]=time;
    if (!nb) return;
    if (!valid) {
        status[index]=4;
        hydraulic[(stage*2)*nb+b]=nan(""); hydraulic[(stage*2+1)*nb+b]=nan("");
        for (int g=0;g<ng;++g) {int j=(stage*ng+g)*nb+b; concentration[j]=nan("");feed[j]=nan("");}
        return;
    }
    double q=bs_value(ht,hq,nh[b],nb,b,time,forcing[b]);
    double level=bs_value(ht,hs,nh[b],nb,b,time,forcing[nb+b]);
    hydraulic[(stage*2)*nb+b]=q; hydraulic[(stage*2+1)*nb+b]=level;
    int flag=(!isfinite(q)||!isfinite(level))?1:0;
    bool positive=false;
    for (int g=0;g<ng;++g) {
        int position=g*nb+b;
        double c=default_c[position], f=default_feed[position];
        if (!g) {c=bs_value(st,sc,ns[b],nb,b,time,c);f=bs_value(st,sf,ns[b],nb,b,time,f);}
        c=bs_value(ft,fc,nf[position],ng*nb,position,time,c);
        f=bs_value(ft,ff,nf[position],ng*nb,position,time,f);
        concentration[stage*ng*nb+position]=c;feed[stage*ng*nb+position]=f;
        positive=positive||f>0;
        if (!isfinite(c)||!isfinite(f)) flag|=1;
    }
    int m=mode[b];
    if ((m==1||m==6||m==7||m==8||m==9)&&positive&&q<=0) flag|=2;
    status[index]=flag;
}
