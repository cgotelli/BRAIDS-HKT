// SPDX-License-Identifier: GPL-3.0-only
//
// Additive point kernels for the Wilcock-Crowe family (FLOW TraFrm 16--18).
// TraFrm 7, 13 and 14 are reused through the already-qualified resident CUDA
// operators by delft_scalar_capacity_v63.py.  This source is intentionally not
// wired into the live dispatcher.

__device__ __forceinline__ void v63_zero(
    const long long point,const long long count,double *output)
{
    #pragma unroll
    for(int row=0;row<7;++row) output[(long long)row*count+point]=0.0;
}

extern "C" __global__ void delft_wilcock_scalar_v63(
    const long long count,const int selector,
    const double *u,const double *v,const double *d50,
    const double *solid_density,const double *water_density,
    const double *gravity,const double *chezy,const double *bed_shear,
    const double *dg,const double *dgsd,const double *sand_fraction,
    const double *acal,const double *theta_c0,const double *alpha0,
    double *output,int *status)
{
    const long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=count) return;
    v63_zero(i,count,output);
    bool valid=(selector==16 || selector==17 || selector==18) &&
        isfinite(u[i]) && isfinite(v[i]) && isfinite(d50[i]) &&
        isfinite(solid_density[i]) && isfinite(water_density[i]) &&
        isfinite(gravity[i]) && isfinite(chezy[i]) &&
        isfinite(bed_shear[i]) && isfinite(dg[i]) &&
        (selector==16 || isfinite(dgsd[i])) &&
        isfinite(sand_fraction[i]) && isfinite(acal[i]) &&
        isfinite(theta_c0[i]) && isfinite(alpha0[i]);
    valid=valid && d50[i]>0.0 && solid_density[i]>water_density[i] &&
        water_density[i]>0.0 && gravity[i]>0.0 && chezy[i]>=0.0 &&
        bed_shear[i]>=0.0 && dg[i]>=0.0 &&
        (selector==16 || dgsd[i]>=0.0) &&
        sand_fraction[i]>=0.0 && sand_fraction[i]<=1.0 && acal[i]>=0.0 &&
        theta_c0[i]>=0.0 && alpha0[i]>=0.0;
    if(!valid) { status[i]=1; return; }

    const double speed=sqrt(u[i]*u[i]+v[i]*v[i]);
    if(!isfinite(speed)) { status[i]=1; return; }
    // Locked trabwc/trabwc2/trabg return zero before evaluating a formula
    // whenever Chezy or Dg is below this threshold.  Direction is owned by
    // eqtran, which also produces zero for zero velocity.
    if(chezy[i]<1.0e-6 || dg[i]<1.0e-6 || speed==0.0) {
        status[i]=0; return;
    }

    double b,taurm,scale=1.0;
    if(selector==16) {
        b=0.67/(1.0+exp(1.5-d50[i]/dg[i]));
        taurm=(0.021+0.015*exp(-20.0*sand_fraction[i]))*
            (solid_density[i]-water_density[i])*gravity[i]*dg[i];
        scale=acal[i];
    } else {
        // FLOW's spatial-D50 branch deliberately supplies DgSD=0.  Evaluate
        // its infinite-sigma limiting value explicitly instead of relying on
        // log(0) and overflow behaviour varying between CUDA toolchains.
        const double sigma_phi=dgsd[i]==0.0 ? 0.0 :
            pow(log(dgsd[i])/log(2.0),2.0);
        if(selector==17) {
            b=(1.0-alpha0[i])/(1.0+exp(1.5-d50[i]/dg[i]));
            taurm=(theta_c0[i]+(dgsd[i]==0.0 ? 0.0 : 0.015/
                (1.0+exp(10.1*sigma_phi-14.14))))*
                (solid_density[i]-water_density[i])*gravity[i]*dg[i];
        } else {
            b=(1.0-alpha0[i])/
                (1.0+exp(1.9-d50[i]/(3.0*dg[i])));
            taurm=(theta_c0[i]+(dgsd[i]==0.0 ? 0.0 : 0.022/
                (1.0+exp(7.1*sigma_phi-11.786))))*
                (solid_density[i]-water_density[i])*gravity[i]*dg[i];
        }
    }
    const double tauri=taurm*pow(d50[i]/dg[i],b);
    if(!isfinite(b) || !isfinite(taurm) || !isfinite(tauri) ||
       taurm<=0.0 || tauri<=0.0) {
        status[i]=1; return;
    }
    const double phi=bed_shear[i]/tauri;
    const double wistar=phi<1.35 ? 0.002*pow(phi,7.5) :
        14.0*pow(1.0-0.894/sqrt(phi),4.5);
    const double total_ustar=sqrt(gravity[i])*speed/chezy[i];
    const double relative=solid_density[i]/water_density[i]-1.0;
    const double volume=scale*wistar*pow(total_ustar,3.0)/
        (relative*gravity[i]);
    const double denominator=(solid_density[i]-water_density[i])*
        gravity[i]*d50[i];
    output[i]=solid_density[i]*volume*u[i]/speed;
    output[count+i]=solid_density[i]*volume*v[i]/speed;
    // Rows 2/3 are the explicit suspended channel, zero for TraFrm 16--18.
    output[4*count+i]=bed_shear[i]/denominator;
    output[5*count+i]=tauri/denominator;
    output[6*count+i]=phi;
    valid=isfinite(wistar) && isfinite(total_ustar) && isfinite(volume) &&
        isfinite(output[i]) && isfinite(output[count+i]) &&
        isfinite(output[4*count+i]) && isfinite(output[5*count+i]) &&
        isfinite(output[6*count+i]) && wistar>=0.0 && volume>=0.0 &&
        output[4*count+i]>=0.0 && output[5*count+i]>=0.0 && phi>=0.0;
    status[i]=valid ? 0 : 1;
    if(!valid) v63_zero(i,count,output);
}
