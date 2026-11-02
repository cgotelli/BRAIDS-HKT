// Current-only Delft3D4 FLOW Van Rijn 1993 Q3 producer and ownership seam.
//
// One CUDA thread owns one (fraction, cell) point.  The constitutive ordering
// matches solver/wave/braids_vr1993_route_v66_module.f90.  No host fallback,
// wave physics, external secondary-flow rotation, or Q4 exchange is hidden in
// this kernel.

__device__ __constant__ double vr66_sigma[20] = {
    -.0874,-.2472,-.3797,-.4897,-.5809,-.6565,-.7193,-.7713,-.8145,-.8503,
    -.8800,-.9046,-.9250,-.9419,-.9560,-.9676,-.9773,-.9854,-.9920,-.9975};
__device__ __constant__ double vr66_thickness[20] = {
    .1747,.1449,.1202,.0997,.0827,.0686,.0569,.0472,.0391,.0325,
    .0269,.0223,.0185,.0154,.0127,.0106,.0088,.0073,.0060,.0050};

__device__ __forceinline__ long long vr66_point(
    const int grain, const long long cell, const long long ncell)
{
    return (long long)grain*ncell+cell;
}

extern "C" __global__ void vr1993_route_v66(
    const long long ncell, const int nclass, const int route,
    const double *depth, const double *velocity,
    const double *observation_height, const double *water_density,
    const double *gravity, const double *kinematic_viscosity,
    const double *additional_shear_stress,
    const double *d50, const double *d90, const double *solid_density,
    const double *molecular_schmidt, const double *aks_factor,
    const double *maximum_volume_concentration, const double *minimum_depth,
    const double *bed_factor, const double *suspended_factor,
    const double *settling_velocity, const double *roughness_z0,
    const double *dstar, const double *critical_shear_stress,
    const double *surface_fraction,
    double *raw_bed, double *raw_conc2d, double *bed_mass,
    double *equilibrium_concentration, double *diagnostics, int *status)
{
    const long long linear=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    const long long count=(long long)nclass*ncell;
    if(linear>=count) return;
    const int grain=(int)(linear/ncell);
    const long long cell=linear-(long long)grain*ncell;
    const long long point=vr66_point(grain,cell,ncell);
    const long long vector_stride=count;

    raw_bed[point]=0.0;
    raw_bed[vector_stride+point]=0.0;
    raw_conc2d[point]=0.0;
    bed_mass[point]=0.0;
    bed_mass[vector_stride+point]=0.0;
    equilibrium_concentration[point]=0.0;
    for(int row=0;row<6;++row)
        diagnostics[(long long)row*count+point]=0.0;

    const double h=depth[cell];
    const double ux=velocity[cell];
    const double uy=velocity[ncell+cell];
    const double zvel=observation_height[cell];
    const double rho=water_density[cell];
    const double grav=gravity[cell];
    const double nu=kinematic_viscosity[cell];
    const double tauadd=additional_shear_stress[cell];
    const double dg=d50[grain];
    const double d90g=d90[grain];
    const double rhos=solid_density[grain];
    const double sigmol=molecular_schmidt[grain];
    const double af=aks_factor[grain];
    const double camax=maximum_volume_concentration[grain];
    const double hmin=minimum_depth[grain];
    const double bedfac=bed_factor[grain];
    const double susfac=suspended_factor[grain];
    const double ws=settling_velocity[point];
    const double z0=roughness_z0[point];
    const double dst=dstar[point];
    const double taucr=critical_shear_stress[point];
    const double fraction=surface_fraction[point];

    bool valid=(route==1 || route==2) && isfinite(h) && isfinite(ux) &&
        isfinite(uy) && isfinite(zvel) && isfinite(rho) && isfinite(grav) &&
        isfinite(nu) && isfinite(tauadd) && isfinite(dg) && isfinite(d90g) &&
        isfinite(rhos) && isfinite(sigmol) && isfinite(af) && isfinite(camax) &&
        isfinite(hmin) && isfinite(bedfac) && isfinite(susfac) &&
        isfinite(ws) && isfinite(z0) && isfinite(dst) && isfinite(taucr) &&
        isfinite(fraction);
    valid=valid && h>=0.0 && hmin>=0.0 && zvel>=0.0 && dg>0.0 &&
        d90g>=dg && rho>0.0 && rhos>rho && grav>0.0 && nu>0.0 &&
        tauadd>=0.0 && sigmol>0.0 && af>0.0 && camax>0.0 && camax<=1.0 &&
        bedfac>=0.0 && susfac>=0.0 && ws>=0.0 && z0>0.0 && dst>0.0 &&
        taucr>0.0 && fraction>=0.0 && fraction<=1.0;
    if(!valid) { status[point]=1; return; }
    if(h<=hmin) { status[point]=0; return; }
    if(zvel<=0.0) { status[point]=1; return; }

    const double speed=sqrt(ux*ux+uy*uy);
    const double rc=30.0*z0;
    const double aks=fmin(fmax(af*rc,0.01*h),0.2*h);
    if(!isfinite(speed) || !isfinite(aks) || aks<=0.0 ||
       12.0*h<=fmax(rc,3.0*d90g)) {
        status[point]=1; return;
    }

    double ustar=speed*0.41/log(1.0+zvel/z0);
    double current_shear=rho*ustar*ustar;
    if(tauadd>0.0) {
        current_shear=sqrt(current_shear*current_shear+tauadd*tauadd);
        ustar=sqrt(current_shear/rho);
    }
    const double f1c=0.24*pow(log10(12.0*h/(3.0*d90g)),-2.0);
    const double fc=0.24*pow(log10(12.0*h/rc),-2.0);
    const double muc=f1c/fc;
    const double tau_ratio=muc*current_shear/taucr;
    double stage=fmax(0.0,tau_ratio-1.0);
    double caks=0.0;
    if(stage>1.0e-6)
        caks=fmin(0.015*dg*pow(stage,1.5)/(aks*pow(dst,0.3)),camax);

    double beta=1.0;
    if(ustar>1.0e-6)
        beta=fmin(fmax(1.0+2.0*pow(ws/ustar,2.0),1.0),1.5);
    const double epsmxc=0.25*0.41*beta*ustar*h;
    double seddif[20];
    double profile[20];
    for(int k=0;k<20;++k) {
        const double zi=(1.0+vr66_sigma[k]-vr66_thickness[k]/2.0)*h;
        if(zi>=0.5*h) seddif[k]=epsmxc;
        else seddif[k]=epsmxc*(1.0-pow(1.0-2.0*zi/h,2.0));
        profile[k]=0.0;
    }

    int kmax=0;
    for(int k=18;k>=0;--k) {
        const double zi=(1.0+vr66_sigma[k]-vr66_thickness[k]/2.0)*h;
        if(zi>=aks) { kmax=k; break; }
    }
    const double bakdif=nu/sigmol;
    if(caks>1.0e-6) {
        double dz=h*(1.0+vr66_sigma[kmax])-aks;
        double diffbt=fmax(seddif[kmax]+bakdif,0.1*ws*dz);
        double factor=1.0+dz*ws/diffbt;
        profile[kmax]=caks*rhos/factor;
        for(int k=kmax-1;k>=0;--k) {
            dz=h*(vr66_sigma[k]-vr66_sigma[k+1]);
            diffbt=fmax(seddif[k]+bakdif,0.1*ws*dz);
            factor=1.0+dz*ws/diffbt;
            profile[k]=profile[k+1]/factor;
        }
        for(int k=kmax+1;k<20;++k) profile[k]=profile[k-1];
    }

    double avgu=0.0,avgcu=0.0;
    for(int k=0;k<20;++k) {
        const double z=(1.0+vr66_sigma[k])*h;
        const double ulog=log(1.0+z/z0);
        avgu+=ulog*vr66_thickness[k];
        avgcu+=ulog*profile[k]*vr66_thickness[k];
    }
    const double conc2d=avgcu/fmax(avgu,1.0e-6);

    stage=fmax(1.0e-10,tau_ratio-1.0);
    const double skin_ustar=ustar*sqrt(muc);
    const double bed_magnitude=0.5*rhos*dg*skin_ustar*pow(dst,-0.3)*stage;
    double qx=0.0,qy=0.0;
    if(speed>1.0e-6) {
        qx=bed_magnitude*ux/speed;
        qy=bed_magnitude*uy/speed;
    }

    valid=isfinite(ustar) && isfinite(muc) && isfinite(tau_ratio) &&
        isfinite(caks) && isfinite(conc2d) && isfinite(qx) && isfinite(qy) &&
        isfinite(bed_magnitude) && ustar>=0.0 && muc>=0.0 && tau_ratio>=0.0 &&
        caks>=0.0 && conc2d>=0.0;
    if(!valid) { status[point]=1; return; }

    double composed_x,composed_y,equilibrium;
    if(route==1) {
        composed_x=fraction*(bedfac*qx+susfac*conc2d*h*ux);
        composed_y=fraction*(bedfac*qy+susfac*conc2d*h*uy);
        equilibrium=0.0;
    } else {
        composed_x=fraction*bedfac*qx;
        composed_y=fraction*bedfac*qy;
        equilibrium=fraction*susfac*conc2d;
    }
    valid=isfinite(composed_x) && isfinite(composed_y) &&
        isfinite(equilibrium) && equilibrium>=0.0;
    if(!valid) { status[point]=1; return; }

    raw_bed[point]=qx;
    raw_bed[vector_stride+point]=qy;
    raw_conc2d[point]=conc2d;
    bed_mass[point]=composed_x;
    bed_mass[vector_stride+point]=composed_y;
    equilibrium_concentration[point]=equilibrium;
    diagnostics[point]=ustar;
    diagnostics[count+point]=muc;
    diagnostics[2*count+point]=tau_ratio;
    diagnostics[3*count+point]=caks;
    diagnostics[4*count+point]=aks;
    diagnostics[5*count+point]=current_shear;
    status[point]=0;
}
