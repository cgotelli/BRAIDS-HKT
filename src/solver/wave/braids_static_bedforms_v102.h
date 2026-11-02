// Q7a static Delft3D4-FLOW bedform/capacity equations for CPU and CUDA.
// The locked Delft3D checkout is a read-only source reference.  This is an
// independent BRAIDS implementation of the current-only 2DH operator contract.
#ifndef BRAIDS_HD
#define BRAIDS_HD
#endif

#define B102_DSILT 3.2e-5
#define B102_DSAND 6.4e-5
#define B102_DGRAVEL 2.0e-3

BRAIDS_HD inline double b102_clip(double x,double a,double b) {
    return fmin(b,fmax(a,x));
}
BRAIDS_HD inline double b102_shields(double dstar) {
    if(dstar<=4.) return .24/dstar;
    if(dstar<=10.) return .14*pow(dstar,-.64);
    if(dstar<=20.) return .04*pow(dstar,-.10);
    if(dstar<=150.) return .013*pow(dstar,.29);
    return .055;
}
BRAIDS_HD inline void b102_bedform_cell(int c,int nc,const int* mode,const double* p,
    const double* depth,const double* u,const double* v,const double* chezy,
    const double* d50,const double* d90,double* out,int* status) {
    status[c]=0;
    const double h=depth[c], speed=hypot(u[c],v[c]), g=p[11], rel=p[12], nu=p[13];
    if(!(h>0.) || !(d50[c]>0.) || !(d90[c]>0.) || !(chezy[c]>0.) ||
       !(g>0.) || !(rel>0.) || !(nu>0.)) { status[c]=1; return; }
    const double ds=d50[c]*pow(g*rel/(nu*nu),1./3.);
    const double tc=p[1]>=0.?p[1]:b102_shields(ds);
    if(mode[8]==0) {
        for(int k=0;k<9;k++) out[k*nc+c]=0.;
        out[6*nc+c]=tc; out[8*nc+c]=ds;
        return;
    }
    const double rgrain=3.*d90[c];
    const double cgrain=18.*log10(12.*h/rgrain);
    const double ub2=g*speed*speed/(cgrain*cgrain);
    const double ucr2=g*rel*d50[c]*tc;
    double height=0.;
    if(mode[0]==1) { // Van Rijn 1984
        const double t=(ub2-ucr2)/ucr2;
        if(t>0. && t<25.) height=p[0]*.11*h*pow(d50[c]/h,.3)*(1.-exp(-.5*t))*(25.-t);
    } else if(mode[0]==2) { // Fredsoe-MPM
        const double mu=pow(chezy[c]/cgrain,1.5);
        const double theta=speed*speed/(chezy[c]*chezy[c]*rel*d50[c]);
        height=p[0]*(24./63.)*fmax(1.-tc/fmax(theta*mu,2.220446049250313e-16),0.)*h;
    } else if(mode[0]==3) height=p[0]*h/5.; // Fredsoe-EH
    else if(mode[0]==4) height=p[2]*pow(fabs(h),p[3]);
    double length=mode[1]==2?p[4]*pow(fabs(h),p[5]):7.3*h;
    length=fmax(length,1.e-6);
    double rksr=0.,rksmr=0.,rksd=0.,psi=0.;
    if(mode[2]==1) { // Van Rijn 1984
        rksr=3.*d90[c];
        rksd=1.1*height*(1.-exp(-25.*height/length));
    } else if(mode[2]==2) { // power on dune height
        rksr=3.*d90[c]; rksd=p[6]*pow(fabs(height),p[7]);
    } else if(mode[2]==0) { // static Van Rijn 2007/2004 predictor
        const double fch2=b102_clip(d50[c]/(1.5*B102_DSAND),.3,1.);
        const double fcoarse=fmin(pow(.25*B102_DGRAVEL/d50[c],1.5),1.);
        const double dm=b102_clip(d50[c],.0001,.0005);
        psi=speed*speed/(1.65*g*dm);
        double rr=psi<=50.?150.*fcoarse*dm:(psi>=250.?20.*fcoarse*dm:
            (182.5-.65*psi)*fcoarse*dm);
        if(d50[c]<B102_DSILT) rr=20.*B102_DSILT;
        rksr=b102_clip(rr,d90[c],.02*h)*p[8];
        double mr=0.;
        if(p[9]>0.) mr=psi<=50.?.0002*fch2*psi*h:(psi<550.?
            (.011-.00002*psi)*fch2*h:0.);
        if(mr<.02) mr=d50[c]>=1.5*B102_DSAND?.02:
            (d50[c]>=B102_DSILT?200.*(d50[c]/(1.5*B102_DSAND))*d50[c]:0.);
        if(h<=1.) mr*=h;
        if(d50[c]<B102_DSILT) mr=0.;
        rksmr=fmin(.2,mr*p[9]);
        double rd=(h>1. && p[10]>0.)?(psi<=100.?.0004*psi*h:
            (psi<600.?(.048-.00008*psi)*h:0.)):0.;
        rksd=rd*p[10];
    }
    const double total=sqrt(rksr*rksr+rksmr*rksmr+rksd*rksd);
    out[c]=height; out[nc+c]=length; out[2*nc+c]=rksr;
    out[3*nc+c]=rksmr; out[4*nc+c]=rksd; out[5*nc+c]=total;
    out[6*nc+c]=tc; out[7*nc+c]=psi; out[8*nc+c]=ds;
    if(!isfinite(height+length+total+tc+psi+ds) || height<0. || total<0.) status[c]=2;
}

BRAIDS_HD inline double b102_settling(double dss,double d50,double rho,double rhos,
    double g,double temp,double sal,double salmax,double gamfloc,double ctot,double csoil) {
    const double s=rhos/rho, nu=4.e-5/(20.+temp);
    double ws;
    if(dss<1.5*B102_DSAND) ws=(s-1.)*g*dss*dss/(18.*nu);
    else if(dss<.5*B102_DGRAVEL) {
        const double coef=dss<2.*B102_DSAND?(-2.9912/B102_DSAND)*dss+15.9824:10.;
        ws=coef*nu/dss*(sqrt(1.+(s-1.)*g*dss*dss*dss/(100.*nu*nu))-1.);
    } else ws=1.1*sqrt((s-1.)*g*dss);
    double ffloc=1., hinset;
    if(d50<B102_DSAND && sal>0.) {
        const double cgel=.65*rhos*fmin(d50/B102_DSAND,1.);
        hinset=fmax(0.,1.-fmax(0.,.65*ctot)/cgel);
        if(sal>=.01 && salmax>0.) {
            const double fh=fmax(4.+log10(2.*fmax(1.e-6,ctot)/cgel),1.);
            const double ef=b102_clip(B102_DSAND/d50-1.,1.,3.);
            const double f0=b102_clip(pow(fh,ef),1.,10.);
            ffloc=b102_clip(((f0-1.)*fmin(1.,sal/salmax)+1.)*gamfloc,1.,10.);
        }
    } else hinset=fmax(0.,1.-fmax(0.,ctot)/csoil);
    return ffloc*ws*pow(hinset,5.);
}

BRAIDS_HD inline void b102_capacity_cell(int c,int nc,const int* mode,const double* p,
    const double* depth,const double* u,const double* v,const double* d10,
    const double* d50,const double* d90,const double* rks,const double* skin,const double* mud,
    const double* concentration,const double* salinity,const double* sx,
    const double* sy,double* out,int* status) {
    status[c]=0;
    const double h=depth[c], ux=u[c], uy=v[c], speed=hypot(ux,uy), g=p[11];
    const double rel=p[12], nu=p[13], rho=p[14], rhos=p[15], eps=2.220446049250313e-16;
    if(!(h>0.) || !(d10[c]>0.) || !(d50[c]>0.) || !(d90[c]>0.) || !(rks[c]>0.) ||
       !(rho>0.) || !(rhos>rho) || !(p[21]>0.)) {status[c]=1;return;}
    // FLOW initsedtra/erosed deliberately use exponent 0.3333 here.
    const double dstar=d50[c]*pow(g*rel/(nu*nu),.3333);
    const double thetacr=p[25]>=0.?p[25]:b102_shields(dstar);
    double dss=d50[c], qx=0.,qy=0.,caks=0.,taurat=0.,shear=0.,aks=0.;
    const double phi=30.*3.14159265358979323846/180.;
    // Common TRANSPOR2004 bed-boundary calculation.  FLOW executes this
    // before dispatching either TraFrm=-2 or -4, and SANTOSS transport is
    // gated by caks>0 even though SANTOSS owns the transported magnitude.
    double taucr0=thetacr*(rhos-rho)*g*d50[c];
    double taucr=taucr0;
    if(mode[5]==1) {
        double fpack=1.,fch=1.,fclay=1.;
        if(d50[c]<B102_DSAND) {
            const double cmaxs=.65, cm=b102_clip((d50[c]/B102_DSAND)*cmaxs,.05,cmaxs);
            fpack=fmin(cm/cmaxs,1.); fch=fmax(pow(B102_DSAND/d50[c],p[17]),1.);
        } else fclay=fmin(pow(1.+mud[c],p[18]),2.);
        taucr=fpack*fch*fclay*taucr0;
    } else if(p[23]>0.) taucr=p[23];
    const double rc=fmax(rks[c],1.e-9), z0=rc/30.;
    const double lg=log10(12.*h/rc), fc=.24/(lg*lg);
    const double tauc=.125*rho*fc*speed*speed;
    const double fc1=.24/pow(log10(12.*h/d90[c]),2.);
    const double taub=(fc1/fc)*tauc;
    taurat=taub/taucr;
    const double dzds=(sx[c]*ux+sy[c]*uy)/fmax(speed,1.e-4);
    const double dzdn=fabs(sx[c]*uy+sy[c]*ux)/fmax(speed,1.e-4);
    const double fsl=pow(fmax(1.-atan(dzds)/phi,.001),.75)*pow(fmax(1.-atan(dzdn)/phi,.001),.37);
    const double rr=b102_clip(.8+.2*((taub/(taucr*fsl)-.8)/1.2),.8,1.);
    const double ta=(taub-rr*taucr*fsl)/(taucr*fsl);
    aks=z0+fmax(.5*skin[c],.01);
    const double fsilt=fmax(1.,B102_DSAND/d50[c]);
    if(ta>eps) caks=fmin(.015*fsilt*d50[c]*pow(ta,1.5)/(aks*pow(dstar,.3)),p[16]);
    const double psi=speed*speed/(rel*g*b102_clip(d50[c],.0001,.0005));
    if(mode[4]==1) {
        dss=psi<550.?(1.+.0006*(d50[c]/d10[c]-1.)*(psi-550.))*d50[c]:d50[c];
        dss=fmax(d10[c],dss); if(d50[c]<B102_DSILT) dss=d50[c];
        dss=fmax(dss,.5*B102_DSILT);
    }
    if(mode[3]==1) { // current-only Van Rijn 2007 / TRANSPOR2004
        const double delm=.05, vrdelm=delm<z0?0.:speed*log(30.*delm/rc)/(-1.+log(30.*h/rc));
        const double vrra=vrdelm*log(30.*aks/rc)/log(30.*delm/rc);
        const double beta=.25*pow((-1.+log(30.*h/rc))/log(30.*aks/rc),2.);
        const double tau=.5*rho*(beta*fc1)*vrra*vrra;
        const double rr2=b102_clip(.8+.2*((tau/(fsl*taucr)-.8)/1.2),.8,1.);
        const double excess=fmax(.0001,(tau-rr2*fsl*taucr)/(fsl*taucr));
        const double sb=fsilt*(.5*d50[c]*rhos/pow(dstar,.3))*sqrt(tau/rho)*excess;
        if(speed>0.) { qx=sb*ux/speed; qy=sb*uy/speed; }
        shear=taub;
    } else if(mode[3]==2) { // current-only SANTOSS 2.08
        double ksc1=3.*d90[c], theta1=0.,theta=0.,fcc=0.;
        for(int it=0;it<100;it++) {
            double ksc=ksc1;
            if(it>0) {
                const double mu=d50[c]<=.00015?6.:(d50[c]>=.00020?1.:
                    6.+(d50[c]-.00015)*(1.-6.)/(.00020-.00015));
                ksc=fmax(ksc1,d50[c]*(mu+6.*(theta1-1.)));
            }
            const double z0=ksc/30.;
            fcc=speed==0.?0.:2.*pow(.4/(log(h/z0)-1.+z0/h),2.);
            theta=.5*fcc*speed*speed/(rel*g*d50[c]);
            if(it>0 && fabs(theta-theta1)<=.001) break;
            theta1=theta;
        }
        const double dstarS=pow(rel*g/(nu*nu),1./3.)*d50[c];
        const double scr=.3/(1.+1.2*dstarS)+.055*(1.-exp(-.02*dstarS));
        double fsl=1.;
        if(mode[6]==1 && speed>0.) {
            const double dzds=-(sx[c]*ux+sy[c]*uy)/speed;
            fsl=sin(phi+atan(dzds))/sin(phi);
        }
        const double coef=d50[c]<=.00015?25.08:(d50[c]>=.00020?13.:
            25.08+(d50[c]-.00015)*(13.-25.08)/(.00020-.00015));
        const double sflt=d50[c]*coef*theta;
        const double d50s=.8*d50[c], dstars=pow(rel*g/(nu*nu),1./3.)*d50s;
        const double wss=nu/d50s*(sqrt(10.36*10.36+1.049*dstars*dstars*dstars)-10.36);
        double pc=0.; if(mode[7]==1 && speed>0.) pc=8.2*sflt/(wss*2.*fmax(p[24],.01));
        const double load=10.9711*pow(fmax(theta-fsl*scr,0.),1.2);
        const double moved=pc<=1.?load:fmin(1./pc,1.)*load;
        // santoss_core uses scx/sqrt(sc): its magnitude is sqrt(sc), not one.
        const double q=(caks>0.?moved:0.)*sqrt(fmax(theta,0.))*sqrt(rel*g*d50[c]*d50[c]*d50[c])*rhos;
        if(speed>0.) {qx=q*ux/speed;qy=q*uy/speed;}
        shear=theta;
    }
    const double ws=b102_settling(dss,d50[c],rho,rhos,g,p[22],salinity[c],p[19],p[20],concentration[c],p[21]);
    out[c]=qx;out[nc+c]=qy;out[2*nc+c]=caks;out[3*nc+c]=dss;
    out[4*nc+c]=ws;out[5*nc+c]=taurat;out[6*nc+c]=shear;
    out[7*nc+c]=aks;out[8*nc+c]=hypot(qx,qy);out[9*nc+c]=thetacr;
    if(!isfinite(qx+qy+caks+dss+ws+taurat+shear+aks) || caks<0. || dss<=0. || ws<0.) status[c]=2;
}
