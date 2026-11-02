// Resident FP64 CUDA transcription of locked Delft3D4 FLOW
// adjust_bedload.f90 ISlope=2/3/4.  This isolated kernel is not part of the
// accepted face path; it is the Q3-H operator foundation.

enum FlowBedSlopeStatus {
    FBS_OK = 0,
    FBS_INVALID_MODE = 1,
    FBS_INVALID_INPUT = 2,
    FBS_SINGULAR = 3,
    FBS_INVALID_OUTPUT = 4
};

extern "C" __global__ void flow_bed_slope_direction(
    const long long count,
    const int islope,
    const double *qx,
    const double *qy,
    const double *down_x,
    const double *down_y,
    const double *stress,
    const double *depth,
    const double *d50,
    const double *mean_diameter,
    const double *hiding_left,
    const double *hiding_right,
    const double alfabs,
    const double alfabn,
    const double ashld,
    const double bshld,
    const double cshld,
    const double dshld,
    const double alfpa,
    const double thcrpa,
    double *corrected_x,
    double *corrected_y,
    int *status)
{
    const long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=count) return;
    const double input_x=qx[i],input_y=qy[i];
    corrected_x[i]=input_x;
    corrected_y[i]=input_y;

    if(islope<2 || islope>4) {
        status[i]=FBS_INVALID_MODE;
        return;
    }
    if(!isfinite(input_x) || !isfinite(input_y) ||
       !isfinite(down_x[i]) || !isfinite(down_y[i]) ||
       !isfinite(stress[i]) || !isfinite(alfabs) || stress[i]<0.0) {
        status[i]=FBS_INVALID_INPUT;
        return;
    }
    if(islope==2) {
        if(!isfinite(alfabn)) {
            status[i]=FBS_INVALID_INPUT;
            return;
        }
    } else if(islope==3) {
        if(!isfinite(depth[i]) || !isfinite(d50[i]) ||
           !isfinite(mean_diameter[i]) || !isfinite(ashld) ||
           !isfinite(bshld) || !isfinite(cshld) || !isfinite(dshld) ||
           depth[i]<=0.0 || d50[i]<=0.0 || mean_diameter[i]<=0.0) {
            status[i]=FBS_INVALID_INPUT;
            return;
        }
    } else {
        if(!isfinite(hiding_left[i]) || !isfinite(hiding_right[i]) ||
           !isfinite(alfpa) || !isfinite(thcrpa) ||
           hiding_left[i]<0.0 || hiding_right[i]<0.0 || thcrpa<0.0) {
            status[i]=FBS_INVALID_INPUT;
            return;
        }
    }

    double magnitude=sqrt(input_x*input_x+input_y*input_y);
    if(!isfinite(magnitude)) {
        status[i]=FBS_INVALID_INPUT;
        return;
    }
    status[i]=FBS_OK;
    if(magnitude<=1.0e-6) return;

    double dzds=down_x[i]*input_x/magnitude+down_y[i]*input_y/magnitude;
    const double dzdn=-down_x[i]*input_y/magnitude+down_y[i]*input_x/magnitude;
    // Match FLOW's default-real literal rounding before promotion to FP64.
    const double tan_phi=tan((double)(30.0f/180.0f)*acos(-1.0));
    dzds=fmin((double)0.9f*tan_phi,dzds);

    if(islope==2) {
        const double bagnold=tan_phi/
            (cos(atan(dzds))*(tan_phi-dzds));
        const double along=fmax(0.0,1.0+alfabs*(bagnold-1.0));
        const double normal=stress[i]>=1.0 ?
            alfabn*pow(1.0/stress[i],0.5)*dzdn : alfabn*dzdn;
        corrected_x[i]=along*input_x-along*input_y*normal;
        corrected_y[i]=along*input_y+along*input_x*normal;
    } else {
        double ftheta=0.0;
        if(stress[i]!=0.0) {
            if(islope==3) {
                // Full FLOW A/B/C/D parameterization and operation order.
                ftheta=ashld*pow(stress[i],bshld)*
                    pow(d50[i]/depth[i],cshld)*
                    pow(d50[i]/mean_diameter[i],dshld);
            } else {
                // FLOW averages the already-selected generic hiding factor at
                // this point; it is not an extra capacity multiplier.
                const double local_hiding=
                    (hiding_left[i]+hiding_right[i])/2.0;
                const double denominator=fmax(
                    stress[i]*0.1,local_hiding*thcrpa);
                if(!isfinite(denominator) || denominator<=0.0) {
                    status[i]=FBS_SINGULAR;
                    return;
                }
                ftheta=alfpa*sqrt(stress[i]/denominator);
            }
        }
        double direction_x,direction_y;
        if(down_x[i]!=0.0 || down_y[i]!=0.0) {
            direction_x=ftheta*input_x/magnitude+down_x[i];
            direction_y=ftheta*input_y/magnitude+down_y[i];
        } else {
            direction_x=input_x/magnitude;
            direction_y=input_y/magnitude;
        }
        const double norm=sqrt(
            direction_x*direction_x+direction_y*direction_y);
        if(!isfinite(norm) || norm<=2.2250738585072014e-308) {
            status[i]=FBS_SINGULAR;
            return;
        }
        magnitude=magnitude*(1.0+alfabs*dzds);
        corrected_x[i]=magnitude*direction_x/norm;
        corrected_y[i]=magnitude*direction_y/norm;
    }

    if(!isfinite(corrected_x[i]) || !isfinite(corrected_y[i])) {
        corrected_x[i]=input_x;
        corrected_y[i]=input_y;
        status[i]=FBS_INVALID_OUTPUT;
    }
}
