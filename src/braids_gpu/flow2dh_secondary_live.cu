// Fixed-grid live composition helpers for the classic Delft3D4 FLOW 2DH
// equilibrium-secondary operator.  Constitutive source/cap/rotation equations
// remain in flow2dh_secondary_operator.cu; this file owns only resident least-
// squares curvature and batched bed-vector routing around that point operator.

extern "C" __global__ void flow2dh_lsq_curvature(
    const int ncell,const int nface,const int nadjacency,
    const double *q,const int *active,const double *matrix,
    const double *delta,const double *weight,const int *owner,
    const int *neighbour,const int *offset,const int *faces,const int *signs,
    const double dry,double *u,double *v,double *curvature,
    unsigned char *operator_active,int *status)
{
    const int cell=blockIdx.x*blockDim.x+threadIdx.x;
    if(cell>=ncell) return;
    u[cell]=0.0;v[cell]=0.0;curvature[cell]=0.0;
    operator_active[cell]=0;status[cell]=0;
    if(active[cell]!=0 && active[cell]!=1) { status[cell]=1;return; }
    const double h=q[cell],hu=q[ncell+cell],hv=q[2*ncell+cell];
    if(!isfinite(h)||!isfinite(hu)||!isfinite(hv)||h<0.0) {
        status[cell]=1;return;
    }
    if(active[cell]==0 || h<dry) return;
    operator_active[cell]=1;
    u[cell]=hu/h;v[cell]=hv/h;
    const double speed=sqrt(u[cell]*u[cell]+v[cell]*v[cell]);
    if(!isfinite(speed)) { status[cell]=1;return; }
    if(speed<1.0e-3) return;
    if(offset[cell]<0 || offset[cell]>offset[cell+1] ||
       offset[cell+1]>nadjacency) { status[cell]=1;return; }
    const double a11=matrix[cell],a12=matrix[ncell+cell];
    const double a22=matrix[2*ncell+cell],det=matrix[3*ncell+cell];
    if(!isfinite(a11)||!isfinite(a12)||!isfinite(a22)||!isfinite(det)) {
        status[cell]=1;return;
    }
    double bx=0.0,by=0.0,cx=0.0,cy=0.0;
    for(int item=offset[cell];item<offset[cell+1];++item) {
        const int face=faces[item];
        if(face<0 || face>=nface || (signs[item]!=1 && signs[item]!=-1)) {
            status[cell]=1;return;
        }
        const int next=signs[item]<0 ? owner[face] : neighbour[face];
        if(next<0) continue;
        if(next>=ncell) { status[cell]=1;return; }
        const double hn=q[next],hun=q[ncell+next],hvn=q[2*ncell+next];
        const double rx=delta[item],ry=delta[nadjacency+item],w=weight[item];
        if(!isfinite(hn)||!isfinite(hun)||!isfinite(hvn)||hn<0.0||
           !isfinite(rx)||!isfinite(ry)||!isfinite(w)||w<0.0) {
            status[cell]=1;return;
        }
        double un=0.0,vn=0.0;
        if(hn>=dry) { un=hun/hn;vn=hvn/hn; }
        bx+=w*rx*(un-u[cell]);by+=w*ry*(un-u[cell]);
        cx+=w*rx*(vn-v[cell]);cy+=w*ry*(vn-v[cell]);
    }
    // Pair the existing CPU fixed-grid least-squares implementation.  FLOW's
    // first live route deliberately uses the same absolute singularity gate.
    if(det<1.0e-6) return;
    const double dudx=(bx*a22-a12*by)/det;
    const double dudy=(a11*by-bx*a12)/det;
    const double dvdx=(cx*a22-a12*cy)/det;
    const double dvdy=(a11*cy-cx*a12)/det;
    curvature[cell]=-(u[cell]*u[cell]*dvdx-v[cell]*v[cell]*dudy+
        u[cell]*v[cell]*(dvdy-dudx))/(speed*speed*speed);
    if(!isfinite(curvature[cell])) { curvature[cell]=0.0;status[cell]=1; }
}

__device__ __forceinline__ bool flow2dh_rotate_one(
    const double u,const double v,const double raw_chezy,const double espir,
    const double gravity,const double intensity,double *x,double *y)
{
    if(!isfinite(u)||!isfinite(v)||!isfinite(raw_chezy)||raw_chezy<=0.0||
       !isfinite(espir)||espir<0.0||!isfinite(gravity)||gravity<=0.0||
       !isfinite(intensity)||!isfinite(*x)||!isfinite(*y)) return false;
    const double speed=sqrt(u*u+v*v),magnitude=sqrt((*x)*(*x)+(*y)*(*y));
    if(!isfinite(speed)||!isfinite(magnitude)) return false;
    if(speed<=0.0 || magnitude<=0.0) { *x=0.0;*y=0.0;return true; }
    // Literal eqtran.f90:713-747 convention: source relaxation may floor C,
    // whereas rotation consumes the unfloored typed hydraulic Chezy.
    const double alpha=12.5*espir*(1.0-sqrt(gravity)/(0.4*raw_chezy))*
        intensity/speed;
    const double tx=u+alpha*v,ty=v-alpha*u;
    const double norm=sqrt(tx*tx+ty*ty);
    if(!isfinite(alpha)||!isfinite(norm)) return false;
    if(norm>2.2204460492503131e-16) {
        *x=magnitude*tx/norm;*y=magnitude*ty/norm;
    } else { *x=0.0;*y=0.0; }
    return isfinite(*x)&&isfinite(*y);
}

extern "C" __global__ void flow2dh_rotate_cell_bed(
    const int ncell,const int nclass,const double *u,const double *v,
    const double *raw_chezy,const double *intensity,const unsigned char *active,
    const double espir,const double gravity,double *raw_bed,int *status)
{
    const int cell=blockIdx.x*blockDim.x+threadIdx.x;
    if(cell>=ncell) return;
    status[cell]=0;
    if(active[cell]>1) { status[cell]=1;return; }
    for(int grain=0;grain<nclass;++grain) {
        const long long xat=(long long)grain*ncell+cell;
        const long long yat=((long long)nclass+grain)*ncell+cell;
        double x=raw_bed[xat],y=raw_bed[yat];
        if(active[cell]==0) { raw_bed[xat]=0.0;raw_bed[yat]=0.0;continue; }
        if(!flow2dh_rotate_one(u[cell],v[cell],raw_chezy[cell],espir,
                              gravity,intensity[cell],&x,&y)) {
            raw_bed[xat]=0.0;raw_bed[yat]=0.0;status[cell]=1;
        } else { raw_bed[xat]=x;raw_bed[yat]=y; }
    }
}

extern "C" __global__ void flow2dh_rotate_boundary_bed(
    const int ncell,const int nface,const int nclass,const double *trace,
    const int *owner,const int *role,const double *trace_raw_chezy,
    const int *trace_roughness_status,
    const double *owner_boundary_equilibrium,const double *rotated_cell_bed,
    const double dry,const double espir,
    const double gravity,double *raw_boundary_bed,int *status)
{
    const int face=blockIdx.x*blockDim.x+threadIdx.x;
    if(face>=nface) return;
    status[face]=0;
    const int cell=owner[face],selected=role[face];
    if(cell<0 || cell>=ncell || selected<0 || selected>3) {
        status[face]=1;return;
    }
    for(int grain=0;grain<nclass;++grain) {
        const long long xat=(long long)grain*nface+face;
        const long long yat=((long long)nclass+grain)*nface+face;
        if(selected==3) {
            // Owner-outflow role consumes the cell vector that was already
            // rotated once.  The downstream face projector uses this same
            // owner channel; mirroring it here keeps diagnostics unambiguous.
            raw_boundary_bed[xat]=rotated_cell_bed[(long long)grain*ncell+cell];
            raw_boundary_bed[yat]=rotated_cell_bed[((long long)nclass+grain)*ncell+cell];
        } else if(selected==1 || selected==2) {
            const double h=trace[face];
            if(!isfinite(h) || trace_roughness_status[face]!=0) {
                raw_boundary_bed[xat]=0.0;raw_boundary_bed[yat]=0.0;
                status[face]=1;continue;
            }
            // A wet inflow/stage trace may legitimately adjoin a currently
            // dry owner during wetting.  FLOW then uses zero owner intensity
            // but retains the trace-local scalar bed capacity and direction.
            if(h<=dry) {
                raw_boundary_bed[xat]=0.0;raw_boundary_bed[yat]=0.0;continue;
            }
            const double u=trace[nface+face]/h,v=trace[2*nface+face]/h;
            double x=raw_boundary_bed[xat],y=raw_boundary_bed[yat];
            if(!flow2dh_rotate_one(u,v,trace_raw_chezy[face],espir,gravity,
                    owner_boundary_equilibrium[cell],&x,&y)) {
                raw_boundary_bed[xat]=0.0;raw_boundary_bed[yat]=0.0;
                status[face]=1;
            } else { raw_boundary_bed[xat]=x;raw_boundary_bed[yat]=y; }
        } else {
            // Closed, wall and hydraulic-flow reversal roles carry no bed
            // transfer.  Never leave stale trace capacity in these slots.
            raw_boundary_bed[xat]=0.0;raw_boundary_bed[yat]=0.0;
        }
    }
}
