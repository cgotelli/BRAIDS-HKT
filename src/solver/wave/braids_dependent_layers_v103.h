// Shared Q5b dependent-layer sorting equations for native CPU and CUDA.
// Source semantics: locked Delft3D4 bedcomposition_module/updcrslyr.
#ifndef BRAIDS_HD
#define BRAIDS_HD
#endif
BRAIDS_HD inline void braids_dependent_targets_cell_v103(int c,int nc,int nl,
    int mode,int coarse,int exchange,int nlag,double active,double alpha,double minimum,
    double coarse_h,double exchange_h,double lag_h,double eu_h,const double* depth,
    const double* dune,double* targets) {
    int start=1+coarse+exchange;
    for(int k=0;k<nl;k++) {
        double h=eu_h;
        if(k==0) h=mode==1?active:fmax(alpha*(mode==2?depth[c]:dune[c]),minimum);
        else if(coarse && k==1) h=coarse_h;
        else if(exchange && k==1+coarse) h=exchange_h;
        else if(k>=start && k<start+nlag) h=lag_h;
        targets[k*nc+c]=h;
    }
}

BRAIDS_HD inline double braids_q5b_taucrit_v103(double d, double rhos,
    double rhow, double gravity, double viscosity) {
    double dstar=d*pow(((rhos-rhow)/rhow)*gravity/(viscosity*viscosity),1./3.);
    double theta=(dstar<=4.)?.240/dstar:
        (dstar<=10.)?.140/pow(dstar,.64):
        (dstar<=20.)?.040/pow(dstar,.10):
        (dstar<=150.)?.013*pow(dstar,.29):.055;
    return theta*(rhos-rhow)*gravity*d;
}

BRAIDS_HD inline void braids_q5b_sort_cell_v103(int c,int nc,int ng,int nl,
    int mode,double dt,double rhow,double gravity,double viscosity,
    double amax,double sinkmax,double asfm,double bsfm,double sigma,
    const double* tau,const double* dune_length,const double* active_target,
    const double* rho,const double* rhosol,const double* diameter,const double* hidexp,
    const double* bedload,double* mass,const double* sv,double* mobility,
    double* transfer,int* status) {
    if(nl<2 || mode<0 || mode>4 || !(dt>=0.) || !(amax>=1.) ||
       !(sinkmax>=0. && sinkmax<=1.) || !(sigma>0.) || !(rhow>0.) ||
       !(gravity>0.) || !(viscosity>0.) || !(tau[c]>=0.) ||
       !(dune_length[c]>0.) || !(active_target[c]>0.)) {status[c]=1;return;}
    double total=0.,mobile_mass=0.,immobile=0.,active_volume=0.,qs=0.;
    for(int g=0;g<ng;g++) {
        int i=g*nc+c; double m=mass[i],tc;
        if(!(m>=0.) || !(rho[g]>0.) || !(rhosol[g]>rhow) || !(diameter[g]>0.) ||
           !(hidexp[i]>0.) || !isfinite(bedload[i])) {status[c]=1;return;}
        if(mode==3) tc=asfm*pow(1000.*diameter[g],bsfm);
        else if(mode==4) tc=hidexp[i]*.047*(rhosol[g]-rhow)*gravity*diameter[g];
        else tc=braids_q5b_taucrit_v103(diameter[g],rhosol[g],rhow,gravity,viscosity);
        double ratio=tau[c]/fmax(tc,1e-300),p=1.;
        if(mode==1 || mode==4) p=ratio>1.?1.:0.;
        else if(mode==2 || mode==3)
            p=.5*(1.+erf(log(fmax(ratio,1e-300))/(sigma*sqrt(2.))));
        mobility[i]=p; total+=m; mobile_mass+=p*m; immobile+=(1.-p)*m;
        active_volume+=m/rho[g];
        qs+=fmax(0.,bedload[i])/rho[g];
    }
    if(mode==0 || total<=0. || immobile<=0. || dt==0.) {
        transfer[c]=0.;status[c]=0;return;
    }
    double active_thickness=active_volume/fmax(sv[c],1e-300);
    if(!(active_thickness>0.)) {transfer[c]=0.;status[c]=0;return;}
    double stretch=fmin(1.+(amax-1.)*active_thickness/active_target[c],amax);
    double length=fmax(dune_length[c]*stretch,1e-3);
    // updcrslyr converts solid transport to bulk transport with the active
    // layer's solid-volume fraction before deriving the dune migration speed.
    qs=qs/fmax(sv[c],1e-300);
    double cel=qs/active_thickness;
    double sink=fmin(cel*dt/fmax(length,1e-300),sinkmax);
    double moved=fmin(immobile,sink*(1.-mobile_mass/total)*total);
    for(int g=0;g<ng;g++) {
        int a=g*nc+c,co=(ng+g)*nc+c;
        double dm=moved*(1.-mobility[a])*mass[a]/immobile;
        mass[a]-=dm; mass[co]+=dm;
    }
    transfer[c]=moved;status[c]=0;
}

// Apply the accepted bed pulse and update active-layer porosity before sorting.
// bed_column_v101 then performs target-thickness restoration, base rules and
// diffusion exactly once on this prepared state.
BRAIDS_HD inline void braids_q5b_prepare_sort_cell_v103(int c,int nc,int ng,int nl,
    int mode,int topmode,int pormode,double dt,double rhow,double gravity,
    double viscosity,double amax,double sinkmax,double asfm,double bsfm,double sigma,
    double sv0,double svsand,double svmud,const double* tau,const double* dune_length,
    const double* active_target,const double* rho,const double* rhosol,
    const double* diameter,const double* hidexp,const double* bedload,
    const double* initial,const double* initial_sv,const double* delta,
    const double* phi,const double* sig,const int* mud,double* mass,double* sv,
    double* mobility,double* transfer,int* status) {
    status[c]=0;
    for(int k=0;k<nl;k++) {
        sv[k*nc+c]=initial_sv[k*nc+c];
        for(int g=0;g<ng;g++) mass[(k*ng+g)*nc+c]=initial[(k*ng+g)*nc+c];
    }
    double oldh=bed_thickness_v101(ng,nc,c,0,mass,sv,rho),eroded=0.,dep=0.;
    for(int g=0;g<ng;g++) {
        int i=g*nc+c;double d=delta[i];
        if(!isfinite(d) || mass[i]+d<0.) {status[c]=1;return;}
        eroded+=fmax(-d,0.)/rho[g];dep+=fmax(d,0.)/rho[g];mass[i]+=d;
        mobility[i]=fmax(d,0.); // temporary fresh-deposit composition scratch
    }
    double solid=0.;for(int g=0;g<ng;g++) solid+=mass[g*nc+c]/rho[g];
    // Same exact constant as bed_column_v101: no separate porosity when
    // packing is already part of the reference density.
    if(pormode==0 || pormode==5) sv[c]=1.;
    else if(topmode==1)
        sv[c]=1.-bed_porosity_v101(ng,nc,c,pormode,mass,phi,sig,sv0,svsand,svmud,mud);
    else {
        double fresh=1.-bed_porosity_v101(ng,nc,c,pormode,mobility,phi,sig,sv0,svsand,svmud,mud);
        double h=oldh-eroded/initial_sv[c]+dep/fresh;
        sv[c]=(solid>0. && h>0.)?solid/h:1.;
    }
    braids_q5b_sort_cell_v103(c,nc,ng,nl,mode,dt,rhow,gravity,viscosity,
        amax,sinkmax,asfm,bsfm,sigma,tau,dune_length,active_target,rho,rhosol,
        diameter,hidexp,bedload,mass,sv,mobility,transfer,status);
}
