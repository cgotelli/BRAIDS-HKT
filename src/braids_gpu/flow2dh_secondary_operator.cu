// Resident FP64 transcription of the classic Delft3D4 FLOW 2DH local
// secondary-flow operators.  Geometry differentiation and stress divergence
// are intentionally outside this component.
//
// Locked source anchors:
//   secrhs.f90:172-200  Rmincf/Coriolis/Chzmin/source/sink
//   secbou.f90:132-154  open-boundary equilibrium
//   eqtran.f90:713-747  scalar bed-vector rotation
//   vihsec.f90:137-207  Betac coefficient and stability clamp

enum Flow2DHSecondaryStatus {
    F2S_OK = 0,
    F2S_INVALID_INPUT = 1,
    F2S_INVALID_GEOMETRY = 2,
    F2S_INVALID_RELAXATION = 3,
    F2S_INVALID_OUTPUT = 4
};

extern "C" __global__ void flow2dh_secondary_operator(
    const long long count,
    const unsigned char *active,
    const double *depth_old,
    const double *depth_new,
    const double *u,
    const double *v,
    const double *raw_inverse_radius,
    const double *metric_ksi_sum,
    const double *metric_eta_sum,
    const double *coriolis,
    const double *raw_chezy,
    const double *spiral_intensity,
    const double *bed_x,
    const double *bed_y,
    const double dryflc,
    const double rmincf,
    const double chzmin,
    const double gravity,
    const double von_karman,
    const double espir,
    const double betac,
    const double epsilon_value,
    double *capped_inverse_radius,
    double *cap_limit,
    double *equilibrium_target,
    double *source_coefficient,
    double *sink_coefficient,
    double *boundary_equilibrium,
    double *rotated_x,
    double *rotated_y,
    double *spiral_alpha,
    double *effective_chezy,
    double *relaxation_alpha,
    double *betast,
    double *beta,
    double *stress_xx_basis,
    double *stress_xy_basis,
    unsigned char *was_capped,
    unsigned char *boundary_defined,
    unsigned char *betac_enabled,
    unsigned char *beta_was_clamped,
    int *status)
{
    const long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=count) return;

    capped_inverse_radius[i]=0.0;
    cap_limit[i]=0.0;
    equilibrium_target[i]=0.0;
    source_coefficient[i]=0.0;
    sink_coefficient[i]=0.0;
    boundary_equilibrium[i]=0.0;
    rotated_x[i]=0.0;
    rotated_y[i]=0.0;
    spiral_alpha[i]=0.0;
    effective_chezy[i]=0.0;
    relaxation_alpha[i]=0.0;
    betast[i]=0.0;
    beta[i]=0.0;
    stress_xx_basis[i]=0.0;
    stress_xy_basis[i]=0.0;
    was_capped[i]=0;
    boundary_defined[i]=0;
    betac_enabled[i]=0;
    beta_was_clamped[i]=0;
    status[i]=F2S_OK;

    // Non-finite state is always an explicit error, including masked/dry
    // entries.  A mask may suppress physics, but never conceal corrupt data.
    if(!isfinite(depth_old[i]) || !isfinite(depth_new[i]) ||
       !isfinite(u[i]) || !isfinite(v[i]) ||
       !isfinite(raw_inverse_radius[i]) ||
       !isfinite(metric_ksi_sum[i]) || !isfinite(metric_eta_sum[i]) ||
       !isfinite(coriolis[i]) || !isfinite(raw_chezy[i]) ||
       !isfinite(spiral_intensity[i]) ||
       !isfinite(bed_x[i]) || !isfinite(bed_y[i]) ||
       !isfinite(dryflc) || !isfinite(rmincf) || !isfinite(chzmin) ||
       !isfinite(gravity) || !isfinite(von_karman) || !isfinite(espir) ||
       !isfinite(betac) || !isfinite(epsilon_value) ||
       dryflc<0.0 || rmincf<=0.0 || chzmin<=0.0 || gravity<=0.0 ||
       von_karman<=0.0 || espir<0.0 || epsilon_value<0.0 ||
       raw_chezy[i]<=0.0 || active[i]>1) {
        status[i]=F2S_INVALID_INPUT;
        return;
    }
    if(active[i]==0) return;

    const double metric_scale=fmax(metric_ksi_sum[i],metric_eta_sum[i]);
    if(metric_scale<=0.0) {
        status[i]=F2S_INVALID_GEOMETRY;
        return;
    }
    const double local_limit=1.0/(rmincf*metric_scale);
    if(!isfinite(local_limit) || local_limit<=0.0) {
        status[i]=F2S_INVALID_GEOMETRY;
        return;
    }
    cap_limit[i]=local_limit;
    if(fabs(raw_inverse_radius[i])<local_limit) {
        capped_inverse_radius[i]=raw_inverse_radius[i];
    } else {
        capped_inverse_radius[i]=copysign(local_limit,raw_inverse_radius[i]);
        was_capped[i]=1;
    }

    const double htrsh=0.5*dryflc;
    const double h_old=fmax(htrsh,depth_old[i]);
    const double h_new=fmax(htrsh,depth_new[i]);
    const double speed=sqrt(u[i]*u[i]+v[i]*v[i]);
    const double chezy=fmax(raw_chezy[i],chzmin);
    const double alpha=sqrt(gravity)/(von_karman*chezy);
    effective_chezy[i]=chezy;
    relaxation_alpha[i]=alpha;
    if(!isfinite(speed) || !isfinite(alpha) || alpha<0.0 || alpha>=0.5) {
        status[i]=F2S_INVALID_RELAXATION;
        return;
    }
    const double tau_old=0.5*(1.0-2.0*alpha)*h_old;
    const double tau_new=0.5*(1.0-2.0*alpha)*h_new;
    if(!isfinite(tau_old) || !isfinite(tau_new) ||
       tau_old<=0.0 || tau_new<=0.0) {
        status[i]=F2S_INVALID_RELAXATION;
        return;
    }
    equilibrium_target[i]=
        (speed*capped_inverse_radius[i]-0.5*coriolis[i])*h_old;
    const double rate_old=
        (von_karman*von_karman*alpha)*speed/tau_old;
    source_coefficient[i]=equilibrium_target[i]*rate_old;
    sink_coefficient[i]=
        (von_karman*von_karman*alpha)*speed/tau_new;

    const double denominator=sink_coefficient[i]*depth_new[i];
    const double epsh=0.5*epsilon_value*dryflc;
    if(fabs(denominator)>epsh) {
        boundary_equilibrium[i]=
            source_coefficient[i]*depth_old[i]/denominator;
        boundary_defined[i]=1;
    }

    const double bed_magnitude=sqrt(bed_x[i]*bed_x[i]+bed_y[i]*bed_y[i]);
    if(!isfinite(bed_magnitude)) {
        status[i]=F2S_INVALID_OUTPUT;
        return;
    }
    if(speed>0.0 && bed_magnitude>0.0) {
        const double a=12.5*espir*
            (1.0-sqrt(gravity)/(0.4*raw_chezy[i]))*
            spiral_intensity[i]/speed;
        const double tx=u[i]+a*v[i];
        const double ty=v[i]-a*u[i];
        const double direction_norm=sqrt(tx*tx+ty*ty);
        spiral_alpha[i]=a;
        if(!isfinite(a) || !isfinite(direction_norm)) {
            status[i]=F2S_INVALID_OUTPUT;
            return;
        }
        if(direction_norm>2.2204460492503131e-16) {
            rotated_x[i]=bed_magnitude*tx/direction_norm;
            rotated_y[i]=bed_magnitude*ty/direction_norm;
        }
    }

    if(fabs(betac)>=1.0e-4) {
        betac_enabled[i]=1;
        const double stress_speed=fmax(1.0e-8,speed);
        const double bs=betac*
            (5.0*alpha-15.6*alpha*alpha+37.5*alpha*alpha*alpha);
        double b=bs*spiral_intensity[i]/stress_speed;
        const double beta_limit=fabs(0.1*bs);
        if(fabs(b)>beta_limit) {
            // This signed assignment is intentional and exactly follows FLOW.
            b=0.1*bs;
            beta_was_clamped[i]=1;
        }
        betast[i]=bs;
        beta[i]=b;
        stress_xx_basis[i]=2.0*b*u[i]*v[i];
        stress_xy_basis[i]=b*(u[i]*u[i]-v[i]*v[i]);
    }

    if(!isfinite(capped_inverse_radius[i]) || !isfinite(cap_limit[i]) ||
       !isfinite(equilibrium_target[i]) || !isfinite(source_coefficient[i]) ||
       !isfinite(sink_coefficient[i]) || !isfinite(boundary_equilibrium[i]) ||
       !isfinite(rotated_x[i]) || !isfinite(rotated_y[i]) ||
       !isfinite(spiral_alpha[i]) || !isfinite(effective_chezy[i]) ||
       !isfinite(relaxation_alpha[i]) || !isfinite(betast[i]) ||
       !isfinite(beta[i]) || !isfinite(stress_xx_basis[i]) ||
       !isfinite(stress_xy_basis[i])) {
        status[i]=F2S_INVALID_OUTPUT;
    }
}
