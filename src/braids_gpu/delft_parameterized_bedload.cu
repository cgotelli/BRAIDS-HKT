// Isolated current-only Delft3D4 FLOW TraFrm=12 and TraFrm=14 operators.
//
// The locked reference routines are trab12.f90 and trab14.f90.  This source
// is intentionally not connected to the coupled/public GPU dispatch.  Every
// physical input and result remains resident on the selected CUDA device;
// invalid points are zeroed and marked in status.

__device__ __forceinline__ void dpb_zero(
    const long long i,const long long count,double *output)
{
    output[i]=0.0;
    output[count+i]=0.0;
    output[2*count+i]=0.0;
}

extern "C" __global__ void delft_ashida_michiue_parameterized(
    const long long count,const double *u,const double *v,const double *d50,
    const double *solid_density,const double *water_density,const double *gravity,
    const double *chezy,const double *hiding_factor,const double *acal,
    const double *critical_shields,const double *power_m,const double *power_p,
    const double *power_q,double *output,int *status)
{
    const long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=count) return;
    dpb_zero(i,count,output);
    bool valid=isfinite(u[i]) && isfinite(v[i]) && isfinite(d50[i]) &&
        isfinite(solid_density[i]) && isfinite(water_density[i]) &&
        isfinite(gravity[i]) && isfinite(chezy[i]) && isfinite(hiding_factor[i]) &&
        isfinite(acal[i]) && isfinite(critical_shields[i]) &&
        isfinite(power_m[i]) && isfinite(power_p[i]) && isfinite(power_q[i]);
    valid=valid && d50[i]>0.0 && solid_density[i]>water_density[i] &&
        water_density[i]>0.0 && gravity[i]>0.0 && chezy[i]>=0.0 &&
        hiding_factor[i]>=0.0 && acal[i]>=0.0 && critical_shields[i]>=0.0 &&
        power_m[i]>=0.0 && power_p[i]>=0.0 && power_q[i]>=0.0;
    if(!valid) { status[i]=1; return; }

    const double speed=sqrt(u[i]*u[i]+v[i]*v[i]);
    // trab14's unsuffixed comparison literal is binary32 before fp promotion.
    const double minimum_speed=(double)1.e-6f;
    if(!isfinite(speed)) { status[i]=1; return; }
    if(speed<minimum_speed || chezy[i]<minimum_speed) { status[i]=0; return; }
    const double relative=solid_density[i]/water_density[i]-1.0;
    const double sag=sqrt(gravity[i]);
    const double sgd=relative*gravity[i]*d50[i];
    const double scale=sqrt(sgd*d50[i]*d50[i]);
    const double shear_velocity=sag*speed/chezy[i];
    const double shields=(shear_velocity*shear_velocity)/sgd;
    const double effective_critical=hiding_factor[i]*critical_shields[i];
    const double ratio=effective_critical>0.0 ? shields/effective_critical :
        1.7976931348623157e308;
    const double critical_ratio=effective_critical/shields;
    double volume=0.0;
    if(critical_ratio<1.0)
        volume=acal[i]*scale*pow(shields,power_m[i])*
            pow(1.0-critical_ratio,power_p[i])*
            pow(1.0-sqrt(critical_ratio),power_q[i]);
    output[i]=solid_density[i]*volume*u[i]/speed;
    output[count+i]=solid_density[i]*volume*v[i]/speed;
    output[2*count+i]=ratio;
    valid=isfinite(output[i]) && isfinite(output[count+i]) &&
        isfinite(output[2*count+i]) && volume>=0.0 && ratio>=0.0;
    status[i]=valid ? 0 : 1;
    if(!valid) dpb_zero(i,count,output);
}

extern "C" __global__ void delft_soulsby_parameterized(
    const long long count,const double *u,const double *v,const double *depth,
    const double *d50,const double *solid_density,const double *water_density,
    const double *gravity,const double *kinematic_viscosity,const double *acal,
    const int *model_index,const double *ratio_d50_z0,const double *von_karman,
    double *output,int *status)
{
    const long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=count) return;
    dpb_zero(i,count,output);
    bool valid=isfinite(u[i]) && isfinite(v[i]) && isfinite(depth[i]) &&
        isfinite(d50[i]) && isfinite(solid_density[i]) &&
        isfinite(water_density[i]) && isfinite(gravity[i]) &&
        isfinite(kinematic_viscosity[i]) && isfinite(acal[i]) &&
        isfinite(ratio_d50_z0[i]) && isfinite(von_karman[i]);
    valid=valid && depth[i]>=0.0 && d50[i]>0.0 &&
        solid_density[i]>water_density[i] && water_density[i]>0.0 &&
        gravity[i]>0.0 && kinematic_viscosity[i]>0.0 && acal[i]>=0.0 &&
        model_index[i]>=1 && model_index[i]<=8 && ratio_d50_z0[i]>0.0 &&
        von_karman[i]>0.0;
    if(!valid) { status[i]=1; return; }

    const double speed=sqrt(u[i]*u[i]+v[i]*v[i]);
    if(!isfinite(speed)) { status[i]=1; return; }
    if(speed<1.e-6 || depth[i]>200.0 || depth[i]<0.01) {
        status[i]=0; return;
    }
    const double relative=solid_density[i]/water_density[i]-1.0;
    const double z0=d50[i]/ratio_d50_z0[i];
    const double drag_denominator=1.0+log(z0/depth[i]);
    if(drag_denominator==0.0) { status[i]=1; return; }
    const double drag=pow(von_karman[i]/drag_denominator,2.0);
    const double tau_current=water_density[i]*drag*speed*speed;
    const double facth=1.0/(water_density[i]*gravity[i]*relative*d50[i]);
    const double theta_mean=tau_current*facth;
    const double theta_wave=0.0;
    const double dstar=pow(gravity[i]*relative/
        (kinematic_viscosity[i]*kinematic_viscosity[i]),1.0/3.0)*d50[i];
    const double theta_critical=0.30/(1.0+1.2*dstar)+
        0.055*(1.0-exp(-0.020*dstar));
    const double phi=-atan2(v[i],u[i]);
    const double theta_max=sqrt(
        pow(theta_mean+theta_wave*cos(phi),2.0)+
        pow(theta_wave*sin(phi),2.0));
    const double ratio=theta_max/fmax(theta_critical,2.2250738585072014e-308);
    output[2*count+i]=ratio;
    if(theta_max<=theta_critical) { status[i]=0; return; }

    // trab12 declares WAVEPS from an unsuffixed default-real literal.
    const double wave_epsilon=(double)1.e-4f;
    const double phi_x1=12.0*sqrt(theta_mean+wave_epsilon)*
        (theta_mean-theta_critical);
    const double phi_x2=12.0*(0.95+0.19*cos(2.0*phi))*
        sqrt(theta_wave+wave_epsilon)*theta_mean;
    const double phi_x=fmax(phi_x1,phi_x2);
    const double phi_y=12.0*(0.19*theta_mean*theta_wave*theta_wave*
        sin(2.0*phi))/(pow(theta_wave+wave_epsilon,1.5)+
        1.5*pow(theta_mean+wave_epsilon,1.5));
    const double scale=sqrt(gravity[i]*relative*d50[i]*d50[i]*d50[i]);
    const double volume_x=acal[i]*scale/speed*(phi_x*u[i]-phi_y*v[i]);
    const double volume_y=acal[i]*scale/speed*(phi_x*v[i]+phi_y*u[i]);
    output[i]=solid_density[i]*volume_x;
    output[count+i]=solid_density[i]*volume_y;
    valid=isfinite(output[i]) && isfinite(output[count+i]) &&
        isfinite(output[2*count+i]) && ratio>=0.0;
    status[i]=valid ? 0 : 1;
    if(!valid) dpb_zero(i,count,output);
}
