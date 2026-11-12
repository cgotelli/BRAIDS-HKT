// Isolated noncohesive source exchange, evaluated after hydraulic friction.
// One thread owns one cell so all classes see the original water/bed state.

__device__ __constant__ double sx_sigma[20] = {
    -.0874,-.2472,-.3797,-.4897,-.5809,-.6565,-.7193,-.7713,-.8145,-.8503,
    -.8800,-.9046,-.9250,-.9419,-.9560,-.9676,-.9773,-.9854,-.9920,-.9975};
__device__ __constant__ double sx_thickness[20] = {
    .1747,.1449,.1202,.0997,.0827,.0686,.0569,.0472,.0391,.0325,
    .0269,.0223,.0185,.0154,.0127,.0106,.0088,.0073,.0060,.0050};

__device__ inline double sx_min(double a,double b) {return a<b?a:b;}
__device__ inline double sx_max(double a,double b) {return a>b?a:b;}

// Exact mathematical order of braids_vr1993_coefficients in the current CPU
// module. Default-real soursin constants are rounded to float before double.
__device__ void sx_vr1993(double h,double hu,double hv,double ws,double rhos,
                          double rho,double d50,double d90,double z0,double aks_factor,
                          double dstar,double taucr,double *eq,double *rate) {
    *eq=0.;*rate=0.;
    if(h<=0.)return;
    double umod=sqrt(hu*hu+hv*hv)/h;
    double zumod=h/exp(1.);
    double rc=30.*z0;
    double aks=sx_min(sx_max(aks_factor*rc,.01*h),.2*h);
    double ustar=umod*.41/log(1.+zumod/z0);
    double tauc=rho*ustar*ustar;
    double f1c=.24*pow(log10(12.*h/(3.*d90)),-2.);
    double fc=.24*pow(log10(12.*h/rc),-2.);
    double muc=f1c/fc;
    double tau_ratio=muc*tauc/taucr;
    double transport_stage=sx_max(0.,tau_ratio-1.);
    double caks=0.;
    if(transport_stage>1.e-6)
        caks=sx_min(.015*d50*pow(transport_stage,1.5)/(aks*pow(dstar,.3)),.65);

    int kmax=0;
    for(int k=18;k>=0;--k) {
        double zi=(1.+sx_sigma[k]-sx_thickness[k]/2.)*h;
        if(zi>=aks){kmax=k;break;}
    }
    double beta=1.;
    if(ustar>1.e-6)beta=sx_min(sx_max(1.+2.*pow(ws/ustar,2.),1.),1.5);
    double epsmxc=.25*.41*beta*ustar*h;
    double seddif[21];double profile[20];
    seddif[0]=epsmxc;
    for(int k=0;k<20;++k){
        double zi=(1.+sx_sigma[k]-sx_thickness[k]/2.)*h;
        if(zi>=.5*h)seddif[k+1]=epsmxc;
        else seddif[k+1]=epsmxc*(1.-pow(1.-2.*zi/h,2.));
        profile[k]=0.;
    }
    if(caks>1.e-6){
        double dz=h*(1.+sx_sigma[kmax])-aks;
        double diffbt=sx_max(seddif[kmax+1]+1.e-6,.1*ws*dz);
        double factor=1.+dz*ws/diffbt;
        profile[kmax]=caks*rhos/factor;
        for(int k=kmax-1;k>=0;--k){
            dz=h*(sx_sigma[k]-sx_sigma[k+1]);
            diffbt=sx_max(seddif[k+1]+1.e-6,.1*ws*dz);
            factor=1.+dz*ws/diffbt;
            profile[k]=profile[k+1]/factor;
        }
        for(int k=kmax+1;k<20;++k)profile[k]=profile[k-1];
    }
    double avgu=0.,avgcu=0.;
    if(zumod>0.){
        for(int k=0;k<20;++k){
            double z=(1.+sx_sigma[k])*h;
            double ulog=log(1.+z/z0);
            avgu=avgu+ulog*sx_thickness[k];
            avgcu=avgcu+ulog*profile[k]*sx_thickness[k];
        }
        *eq=avgcu/sx_max(avgu,1.e-6);
    }
    double wsl=sx_max(1.e-3,ws);
    if(wsl<=0.)return;
    if(umod>1.e-6&&ustar>1.e-6){
        double u=sx_min(ustar/umod,.15);
        double w=sx_min(wsl/ustar,1.);
        double x=w,x2=x*x,x3=x2*x;
        double tsd=x*exp(((double)1.547f-(double)20.12f*u)*x3+
                         ((double)326.832f*pow(u,(double)2.2047f)-(double).2f)*x2+
                         ((double).1385f*log(u)-(double)6.4061f)*x+
                         ((double).5467f*u+(double)2.1963f));
        double hots=wsl/tsd;
        *rate=hots/h;
    }else{
        *eq=0.;*rate=wsl/h;
    }
}

extern "C" __global__ void sx_exchange(
    int nc,int ng,int exchange_mode,int settling_mode,int deck_version,int active_mode,
    const double *depth,const double *momentum,const double *suspended,const double *active,
    const double *manning,const double *bulk,const double *ws,const double *d50,const double *d90,
    const double *rhos,const double *z0,const double *sedthr,const double *reference_height,
    const double *dstar,const double *taucr,const int *roughness_mode,
    double rho,double grav,double dry,double active_target,double depth_fraction,
    double minimum_target,double hindered_reference,double dt,
    double *next_suspended,double *next_active,double *delta,int *status) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c>=nc)return;
    double h=depth[c],hu=momentum[c],hv=momentum[nc+c],roughness=manning[c];
    int bad=(!isfinite(h)||h<0.||!isfinite(hu)||!isfinite(hv)||!isfinite(roughness)||roughness<0.);
    double concentration=0.,thickness=0.;
    for(int g=0;g<ng;++g){
        double s=suspended[g*nc+c],b=active[g*nc+c];
        if(!isfinite(s)||!isfinite(b)||s<0.||b<0.)bad=1;
        if(h>dry)concentration=concentration+sx_max(0.,s)/h;
        thickness=thickness+b/bulk[g];
    }
    if(bad){
        status[c]=1;
        for(int g=0;g<ng;++g){int i=g*nc+c;next_suspended[i]=suspended[i];next_active[i]=active[i];delta[i]=0.;}
        return;
    }
    double hindered=1.;
    if(settling_mode==1&&h>dry)
        hindered=pow(sx_max(0.,1.-concentration/hindered_reference),5.);
    double target=active_target;
    if(active_mode==1)target=sx_max(depth_fraction*sx_max(0.,h),minimum_target);
    double work_s[32],work_b[32];
    for(int g=0;g<ng;++g){
        int i=g*nc+c;
        double s=suspended[i],b=active[i],total=s+b;
        double effective_ws=ws[g]*hindered;
        double availability=1.;
        if(deck_version>=13){
            if(thickness<=0.)availability=0.;
            else availability=(b/bulk[g])/thickness*sx_min(1.,thickness/target);
        }
        if(h<=dry)s=0.;
        else if(exchange_mode==1)s=s*exp(-effective_ws*dt/h);
        else if(h>sedthr[g]){
            double rough_z0=z0[g];
            if(roughness_mode[g]!=0){
                double x=.41*pow(h,1./6.)/(sqrt(grav)*roughness);
                double e=exp(-x);
                rough_z0=sx_max(2.2250738585072014e-308,h*exp(-1.)*e/(1.-e));
            }
            double eq,rate;
            sx_vr1993(h,hu,hv,effective_ws,rhos[g],rho,d50[g],d90[g],rough_z0,
                      reference_height[g],dstar[g],taucr[g],&eq,&rate);
            eq=eq*availability;
            s=sx_min(total,sx_max(0.,h*eq+(s-h*eq)*exp(-rate*dt)));
        }
        work_s[g]=s;work_b[g]=total-s;
        if(!isfinite(s)||!isfinite(work_b[g])||s<0.||work_b[g]<0.)bad=2;
    }
    status[c]=bad;
    for(int g=0;g<ng;++g){
        int i=g*nc+c;
        next_suspended[i]=bad?suspended[i]:work_s[g];
        next_active[i]=bad?active[i]:work_b[g];
        delta[i]=bad?0.:work_s[g]-suspended[i];
    }
}
