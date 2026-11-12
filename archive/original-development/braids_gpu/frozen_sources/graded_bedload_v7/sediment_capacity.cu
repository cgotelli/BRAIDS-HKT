// Isolated FP64 transport constitutive kernels; no reconstruction or Exner update.
// Direct port of the frozen BRAIDS MPM and Bagnold--Ikeda implementations.
// The latter is AGPL-3.0-or-later, derived in BRAIDS from Deltares ISlope=2.
// NVRTC supplies device math builtins; no system C/C++ headers are required.

extern "C" __global__ void sediment_capacity(const int n, const double* x,
    const int* mode, const int* active, double* q, int* status) {
    const int i=blockDim.x*blockIdx.x+threadIdx.x;
    if(i>=n) return;
    q[i]=0.; q[n+i]=0.; status[i]=0;
    if(!active[i]) return;
    double a[14];
    for(int j=0;j<14;++j) a[j]=x[j*n+i];
    // h, hu, hv, n, D50, D90, relative density, g, critical Shields,
    // hiding factor, solid density, fixed Cf, minimum formula depth, dry depth.
    const double h=a[0];
    if(h<=a[13]) return;
    double hc=h, hu=a[1], hv=a[2], wet=1.;
    const double minimum=fmax(a[12],a[13]);
    if(h<minimum && minimum>a[13]) {
        const double xi=fmax(0.,fmin(1.,(h-a[13])/(minimum-a[13])));
        wet=xi*xi*(3.-2.*xi);
        hc=minimum*(1.+128.*2.220446049250313080847263336181640625e-16);
        hu=hu*hc/h; hv=hv*hc/h;
    }
    const double u=hu/hc, v=hv/hc;
    double magnitude=0., speed;
    if(mode[i]==1) {
        if(!(isfinite(hc)&&isfinite(u)&&isfinite(v)&&isfinite(a[3])&&isfinite(a[4])&&
             isfinite(a[5])&&isfinite(a[6])&&isfinite(a[7])&&isfinite(a[8])&&isfinite(a[9])) ||
           !(hc>0.&&a[3]>0.&&a[4]>0.&&a[5]>0.&&a[6]>0.&&a[7]>0.&&a[8]>=0.&&a[9]>0.)) {
            status[i]=1; return;
        }
        speed=sqrt(u*u+v*v);
        if(speed==0.) return;
        const double chezy=pow(hc,1./6.)/a[3];
        const double grain=18.*log10(12.*fmax(hc/a[5],1.));
        if(chezy<=0. || grain<=0.) {status[i]=1;return;}
        const double ripple=fmin(pow(chezy/grain,1.5),1.);
        const double sc=speed/chezy;
        const double theta=sc*sc/(a[6]*a[4]);
        const double excess=fmax(ripple*theta-a[9]*a[8],0.);
        magnitude=8.*sqrt(a[7]*a[6]*(a[4]*a[4]*a[4]))*pow(excess,1.5);
    } else if(mode[i]==0) {
        const double critical=a[8]*a[9];
        if(!(isfinite(u)&&isfinite(v)&&isfinite(a[11])&&isfinite(a[4])&&isfinite(a[6])&&
             isfinite(a[7])&&isfinite(critical)) ||
           !(a[4]>0.&&a[6]>0.&&a[7]>0.&&a[11]>=0.&&critical>=0.)) {
            status[i]=1; return;
        }
        speed=sqrt(u*u+v*v);
        if(speed==0.) return;
        const double theta=fmax(0.,a[11]*(speed*speed)/(a[7]*a[6]*a[4])-critical);
        magnitude=8.*sqrt(a[7]*a[6]*(a[4]*a[4]*a[4]))*pow(theta,1.5);
    } else {status[i]=2;return;}
    const double qx=magnitude*u/speed, qy=magnitude*v/speed;
    q[i]=wet*(a[10]*qx); q[n+i]=wet*(a[10]*qy);
    if(!(isfinite(q[i])&&isfinite(q[n+i]))) status[i]=1;
}

extern "C" __global__ void sediment_slope(const int n, const double* x,
    double* q, int* status) {
    const int i=blockDim.x*blockIdx.x+threadIdx.x;
    if(i>=n) return;
    double a[8]; bool valid=true;
    for(int j=0;j<8;++j){a[j]=x[j*n+i];valid=valid&&isfinite(a[j]);}
    q[i]=a[0];q[n+i]=a[1];status[i]=0;
    if(!valid || !(a[4]>=0.&&a[5]>=0.&&a[6]>=0.&&a[7]>0.&&a[7]<90.)) {status[i]=1;return;}
    const double magnitude=sqrt(a[0]*a[0]+a[1]*a[1]);
    if(magnitude<=1.e-6) return;
    const double ax=a[0]/magnitude,ay=a[1]/magnitude,nx=-ay,ny=ax;
    double ds=a[2]*ax+a[3]*ay;
    const double dp=a[2]*nx+a[3]*ny;
    const double phi=tan(a[7]*acos(-1.)/180.);
    ds=fmin(.9*phi,ds);
    const double bagnold=phi/(cos(atan(ds))*(phi-ds));
    const double along=fmax(0.,1.+a[5]*(bagnold-1.));
    const double normal=a[4]>=1. ? a[6]*dp/sqrt(a[4]) : a[6]*dp;
    q[i]=along*magnitude*(ax+normal*nx);
    q[n+i]=along*magnitude*(ay+normal*ny);
    if(!(isfinite(q[i])&&isfinite(q[n+i])))status[i]=1;
}
