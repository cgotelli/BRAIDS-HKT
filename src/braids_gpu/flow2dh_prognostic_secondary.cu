// Resident fixed-grid Q3-H3 classic FLOW prognostic-secondary helpers.
//
// The constitutive source/sink and boundary-equilibrium coefficients are
// supplied by the authenticated classic FLOW point operator.  This file owns
// only the signed hI state update and the conservative-advection boundary
// trace.  It deliberately does not implement the historical exact-exponential
// BRAIDS relaxation or the filtered-curvature legacy route.

enum Flow2DHPrognosticStatus {
    F2P_OK = 0,
    F2P_INVALID_INPUT = 1,
    F2P_INVALID_SOURCE = 2,
    F2P_INVALID_OUTPUT = 3
};

extern "C" __global__ void flow2dh_prognostic_reaction(
    const int ncell,const double *depth_old,const double *depth_new,
    const unsigned char *active,const double *source_coefficient,
    const double *sink_coefficient,const double *area,const double dryflc,
    const double dt,double *mass,double *intensity,double *source_work,
    double *explicit_source_work,double *implicit_sink_work,int *status)
{
    const int cell=blockIdx.x*blockDim.x+threadIdx.x;
    if(cell>=ncell) return;
    status[cell]=F2P_OK;intensity[cell]=0.0;source_work[cell]=0.0;
    explicit_source_work[cell]=0.0;implicit_sink_work[cell]=0.0;
    const double h0=depth_old[cell],h1=depth_new[cell],m0=mass[cell];
    const double source=source_coefficient[cell],sink=sink_coefficient[cell];
    const double cell_area=area[cell];
    if(!isfinite(h0)||!isfinite(h1)||!isfinite(m0)||!isfinite(source)||
       !isfinite(sink)||!isfinite(cell_area)||!isfinite(dryflc)||
       !isfinite(dt)||h0<0.0||h1<0.0||sink<0.0||cell_area<=0.0||
       dryflc<0.0||dt<0.0||active[cell]>1) {
        status[cell]=F2P_INVALID_INPUT;return;
    }
    // chkdry owns this signed reset.  The removed mass is part of the source
    // budget so a dry transition cannot create an unexplained residual.
    if(active[cell]==0 || h1<=dryflc) {
        mass[cell]=0.0;source_work[cell]=-cell_area*m0;
        implicit_sink_work[cell]=source_work[cell];return;
    }
    const double explicit_increment=dt*h0*source;
    const double pre_sink=m0+explicit_increment;
    const double denominator=1.0+dt*sink;
    if(!isfinite(explicit_increment)||!isfinite(pre_sink)||
       !isfinite(denominator)||denominator<=0.0) {
        status[cell]=F2P_INVALID_SOURCE;return;
    }
    const double m1=pre_sink/denominator;
    const double sink_increment=m1-pre_sink;
    const double i1=m1/h1;
    if(!isfinite(m1)||!isfinite(i1)||!isfinite(sink_increment)) {
        status[cell]=F2P_INVALID_OUTPUT;return;
    }
    mass[cell]=m1;intensity[cell]=i1;
    explicit_source_work[cell]=cell_area*explicit_increment;
    implicit_sink_work[cell]=cell_area*sink_increment;
    source_work[cell]=cell_area*(m1-m0);
}

extern "C" __global__ void flow2dh_prognostic_primitive(
    const int ncell,const double *depth,const double *mass,
    const unsigned char *active,const double dryflc,double *intensity,
    int *status)
{
    const int cell=blockIdx.x*blockDim.x+threadIdx.x;
    if(cell>=ncell) return;
    status[cell]=F2P_OK;intensity[cell]=0.0;
    const double h=depth[cell],m=mass[cell];
    if(!isfinite(h)||!isfinite(m)||!isfinite(dryflc)||h<0.0||
       dryflc<0.0||active[cell]>1) {
        status[cell]=F2P_INVALID_INPUT;return;
    }
    if(active[cell]==0 || h<=dryflc) return;
    intensity[cell]=m/h;
    if(!isfinite(intensity[cell])) {
        intensity[cell]=0.0;status[cell]=F2P_INVALID_OUTPUT;
    }
}

extern "C" __global__ void flow2dh_prognostic_low_flux(
    const int ncell,const int nface,const double *cell_intensity,
    const double *face_intensity,const double *boundary_equilibrium,
    const unsigned char *boundary_defined,const int *owner,
    const int *neighbour,const double *water_flux,double *low_flux,
    double *anti_flux,double *boundary_intensity,int *status)
{
    const int face=blockIdx.x*blockDim.x+threadIdx.x;
    if(face>=nface) return;
    status[face]=F2P_OK;boundary_intensity[face]=0.0;
    const int left=owner[face],right=neighbour[face];
    if(left<0||left>=ncell||right>=ncell||boundary_defined[left]>1||
       !isfinite(water_flux[face])) {
        low_flux[face]=0.0;anti_flux[face]=0.0;
        status[face]=F2P_INVALID_INPUT;return;
    }
    if(right<0) {
        // secbou overwrites both open ends from the adjacent interior
        // equilibrium, independently of instantaneous flow direction.  With
        // one prescribed trace, low and high boundary fluxes are identical.
        const double value=boundary_defined[left]
            ? boundary_equilibrium[left] : 0.0;
        boundary_intensity[face]=value;
        low_flux[face]=water_flux[face]*value;anti_flux[face]=0.0;
    } else if(water_flux[face]>=0.0) {
        low_flux[face]=water_flux[face]*cell_intensity[left];
        anti_flux[face]=water_flux[face]*face_intensity[face]-low_flux[face];
    } else {
        low_flux[face]=water_flux[face]*cell_intensity[right];
        anti_flux[face]=water_flux[face]*face_intensity[nface+face]-low_flux[face];
    }
    if(!isfinite(low_flux[face])||!isfinite(anti_flux[face])||
       !isfinite(boundary_intensity[face])) {
        low_flux[face]=0.0;anti_flux[face]=0.0;
        boundary_intensity[face]=0.0;status[face]=F2P_INVALID_OUTPUT;
    }
}

extern "C" __global__ void flow2dh_prognostic_fct_bounds(
    const int ncell,const int nface,const double *intensity,
    const double *mass,const double *next_depth,const double *area,
    const double *length,const int *owner,const int *neighbour,const int *bid,
    const int *offset,const int *faces,const int *signs,const int *active,
    const double *low_flux,const double *anti_flux,
    const double *boundary_intensity,const double dt,double *lowstate,
    double *plus,double *minus,int *status)
{
    const int cell=blockIdx.x*blockDim.x+threadIdx.x;
    if(cell>=ncell) return;
    status[cell]=F2P_OK;lowstate[cell]=mass[cell];plus[cell]=1.0;minus[cell]=1.0;
    if(active[cell]==0) return;
    double pp=0.0,pm=0.0,value=mass[cell];
    double lower=intensity[cell],upper=lower;
    bool classic_open_trace=false;
    for(int item=offset[cell];item<offset[cell+1];++item) {
        const int face=faces[item];
        const int next=signs[item]<0 ? owner[face] : neighbour[face];
        if(next>=0) {
            lower=fmin(lower,intensity[next]);upper=fmax(upper,intensity[next]);
        } else if(bid[face]>=0) {
            // Classic FLOW prescribes the same adjacent-interior equilibrium
            // trace at an open boundary for both flow directions.
            lower=fmin(lower,boundary_intensity[face]);
            upper=fmax(upper,boundary_intensity[face]);
            classic_open_trace=true;
        }
        const double z=-dt*signs[item]*length[face]/area[cell];
        value+=z*low_flux[face];
        pp+=fmax(z*anti_flux[face],0.0);
        pm+=fmin(z*anti_flux[face],0.0);
    }
    if(next_depth[cell]<=0.0) { lower=0.0;upper=0.0; }
    else { lower*=next_depth[cell];upper*=next_depth[cell]; }
    if(classic_open_trace) {
        // secbou's sign-independent Dirichlet trace is also imposed at an
        // outflow.  Its monotone low-order flux can therefore move signed hI
        // outside the old neighbouring-intensity envelope.  This baseline is
        // admissible; only a nonzero antidiffusive correction is limited.
        // This is the exact native-v57 FCT ownership rule.
        lower=fmin(lower,value);upper=fmax(upper,value);
    }
    const double scale=512.0*2.2204460492503131e-16*
        fmax(1.0,fmax(fabs(value),fmax(fabs(lower),fabs(upper))));
    if(value<lower-scale||value>upper+scale) {
        status[cell]=6;return;
    }
    lowstate[cell]=value;
    if(pp>0.0) plus[cell]=fmin(1.0,fmax(0.0,upper-value)/pp)*
        (1.0-32.0*2.2204460492503131e-16);
    if(pm<0.0) minus[cell]=fmin(1.0,fmax(0.0,value-lower)/(-pm))*
        (1.0-32.0*2.2204460492503131e-16);
}

__device__ __forceinline__ bool flow2dh_prognostic_rotate_one(
    const double u,const double v,const double raw_chezy,const double espir,
    const double gravity,const double intensity,double *x,double *y)
{
    if(!isfinite(u)||!isfinite(v)||!isfinite(raw_chezy)||raw_chezy<=0.0||
       !isfinite(espir)||espir<0.0||!isfinite(gravity)||gravity<=0.0||
       !isfinite(intensity)||!isfinite(*x)||!isfinite(*y)) return false;
    const double speed=sqrt(u*u+v*v),magnitude=sqrt((*x)*(*x)+(*y)*(*y));
    if(!isfinite(speed)||!isfinite(magnitude)) return false;
    if(speed<=0.0||magnitude<=0.0) { *x=0.0;*y=0.0;return true; }
    const double alpha=12.5*espir*
        (1.0-sqrt(gravity)/(0.4*raw_chezy))*intensity/speed;
    const double tx=u+alpha*v,ty=v-alpha*u,norm=sqrt(tx*tx+ty*ty);
    if(!isfinite(alpha)||!isfinite(norm)) return false;
    if(norm>2.2204460492503131e-16) {
        *x=magnitude*tx/norm;*y=magnitude*ty/norm;
    } else { *x=0.0;*y=0.0; }
    return isfinite(*x)&&isfinite(*y);
}

extern "C" __global__ void flow2dh_prognostic_rotate_boundary_bed(
    const int ncell,const int nface,const int nclass,const double *trace,
    const int *owner,const int *role,const double *trace_raw_chezy,
    const int *trace_roughness_status,
    const double *owner_boundary_equilibrium,const double dryflc,
    const double espir,const double gravity,const double *rotated_cell_bed,
    double *raw_boundary_bed,
    int *status)
{
    const int face=blockIdx.x*blockDim.x+threadIdx.x;
    if(face>=nface) return;
    status[face]=F2P_OK;
    const int cell=owner[face],selected=role[face];
    if(cell<0||cell>=ncell||selected<0||selected>3) {
        status[face]=F2P_INVALID_INPUT;return;
    }
    for(int grain=0;grain<nclass;++grain) {
        const long long xat=(long long)grain*nface+face;
        const long long yat=((long long)nclass+grain)*nface+face;
        // Roles 1/2 are prescribed trace capacities and consume secbou's
        // equilibrium I.  Role 3 is owner outflow: copy the already-once-
        // rotated owner-cell capacity so it cannot be steered a second time.
        // Role zero carries no sediment transfer.
        if(selected==0) {
            raw_boundary_bed[xat]=0.0;raw_boundary_bed[yat]=0.0;continue;
        }
        if(selected==3) {
            const long long cx=(long long)grain*ncell+cell;
            const long long cy=((long long)nclass+grain)*ncell+cell;
            const double x=rotated_cell_bed[cx],y=rotated_cell_bed[cy];
            if(!isfinite(x)||!isfinite(y)) {
                raw_boundary_bed[xat]=0.0;raw_boundary_bed[yat]=0.0;
                status[face]=F2P_INVALID_INPUT;
            } else {
                raw_boundary_bed[xat]=x;raw_boundary_bed[yat]=y;
            }
            continue;
        }
        const double h=trace[face];
        if(!isfinite(h)||trace_roughness_status[face]!=0) {
            raw_boundary_bed[xat]=0.0;raw_boundary_bed[yat]=0.0;
            status[face]=F2P_INVALID_INPUT;continue;
        }
        if(h<=dryflc) {
            raw_boundary_bed[xat]=0.0;raw_boundary_bed[yat]=0.0;continue;
        }
        const double u=trace[nface+face]/h,v=trace[2*nface+face]/h;
        double x=raw_boundary_bed[xat],y=raw_boundary_bed[yat];
        if(!flow2dh_prognostic_rotate_one(
                u,v,trace_raw_chezy[face],espir,gravity,
                owner_boundary_equilibrium[cell],&x,&y)) {
            raw_boundary_bed[xat]=0.0;raw_boundary_bed[yat]=0.0;
            status[face]=F2P_INVALID_OUTPUT;
        } else { raw_boundary_bed[xat]=x;raw_boundary_bed[yat]=y; }
    }
}
