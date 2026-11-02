// Additive CUDA operators for the Q3-H3/v58 nonzero classic FLOW slice.
//
// The accepted v57 CUDA sources are intentionally not included or modified.
// This inventory owns only three additions: exact constant-Coriolis rotation,
// accepted-start-lagged Cartesian vihsec stress fluxes, and a conservative
// graph backward-Euler solve for signed spiral intensity.  Every candidate is
// private until the Python transaction owner explicitly commits it.

enum V58Status {
    V58_OK = 0,
    V58_INVALID_INPUT = 1,
    V58_INVALID_GEOMETRY = 2,
    V58_INVALID_LINEAR_SYSTEM = 3,
    V58_NOT_CONVERGED = 4,
    V58_CANCELLED = 5,
    V58_INVALID_OUTPUT = 6
};

__device__ __forceinline__ void v58_tensor(
    const double u,const double v,const double intensity,const double chezy,
    const double betac,const double gravity,const double von_karman,
    const double chzmin,double *txx,double *txy,double *tyy,int *status)
{
    *txx=0.0;*txy=0.0;*tyy=0.0;
    if(!isfinite(u)||!isfinite(v)||!isfinite(intensity)||!isfinite(chezy)||
       !isfinite(betac)||!isfinite(gravity)||!isfinite(von_karman)||
       !isfinite(chzmin)||chezy<=0.0||gravity<=0.0||von_karman<=0.0||
       chzmin<=0.0) {
        *status=V58_INVALID_INPUT;return;
    }
    if(fabs(betac)<1.0e-4) return;
    const double speed=fmax(1.0e-8,sqrt(u*u+v*v));
    const double alpha=sqrt(gravity)/(von_karman*fmax(chezy,chzmin));
    const double beta_star=betac*
        (5.0*alpha-15.6*alpha*alpha+37.5*alpha*alpha*alpha);
    double beta=beta_star*intensity/speed;
    // Literal locked vihsec assignment.  It is intentionally not a
    // sign-preserving saturation.
    if(fabs(beta)>fabs(0.1*beta_star)) beta=0.1*beta_star;
    *txx=2.0*beta*u*v;
    *txy=beta*(u*u-v*v);
    *tyy=-*txx;
    if(!isfinite(*txx)||!isfinite(*txy)||!isfinite(*tyy)) {
        *txx=0.0;*txy=0.0;*tyy=0.0;*status=V58_INVALID_OUTPUT;
    }
}

extern "C" __global__ void v58_constant_coriolis(
    const int ncell,const double coriolis,const double dt,const double *area,
    double *q,double *integrated_increment,int *status)
{
    const int cell=blockIdx.x*blockDim.x+threadIdx.x;
    if(cell>=ncell) return;
    status[cell]=V58_OK;
    integrated_increment[cell]=0.0;
    integrated_increment[ncell+cell]=0.0;
    const double qx=q[ncell+cell],qy=q[2*ncell+cell],a=area[cell];
    if(!isfinite(qx)||!isfinite(qy)||!isfinite(a)||a<=0.0||
       !isfinite(coriolis)||!isfinite(dt)||dt<0.0) {
        status[cell]=V58_INVALID_INPUT;return;
    }
    if(coriolis==0.0||dt==0.0) return;
    const double angle=coriolis*dt,co=cos(angle),si=sin(angle);
    const double nx=co*qx+si*qy,ny=-si*qx+co*qy;
    if(!isfinite(nx)||!isfinite(ny)) {
        status[cell]=V58_INVALID_OUTPUT;return;
    }
    q[ncell+cell]=nx;q[2*ncell+cell]=ny;
    integrated_increment[cell]=a*(nx-qx);
    integrated_increment[ncell+cell]=a*(ny-qy);
}

extern "C" __global__ void v58_betac_cells(
    const int ncell,const double *q,const double *h_i,const double *chezy,
    const int *active,const double dryflc,const double betac,
    const double gravity,const double von_karman,const double chzmin,
    double *tensor,int *status)
{
    const int cell=blockIdx.x*blockDim.x+threadIdx.x;
    if(cell>=ncell) return;
    status[cell]=V58_OK;
    tensor[cell]=0.0;tensor[ncell+cell]=0.0;tensor[2*ncell+cell]=0.0;
    const double h=q[cell];
    if(!isfinite(h)||!isfinite(h_i[cell])||!isfinite(dryflc)||dryflc<0.0||
       active[cell]<0||active[cell]>1) {
        status[cell]=V58_INVALID_INPUT;return;
    }
    if(active[cell]==0||h<=dryflc||fabs(betac)<1.0e-4) return;
    const double u=q[ncell+cell]/h,v=q[2*ncell+cell]/h;
    v58_tensor(u,v,h_i[cell]/h,chezy[cell],betac,gravity,von_karman,
               chzmin,&tensor[cell],&tensor[ncell+cell],
               &tensor[2*ncell+cell],&status[cell]);
}

extern "C" __global__ void v58_betac_faces(
    const int ncell,const int nface,const double *trace,const int *owner,
    const int *neighbour,const int *open_face,const double *normal,
    const double *cell_tensor,const double *boundary_equilibrium,
    const double *boundary_chezy,const double dryflc,const double betac,
    const double gravity,const double von_karman,const double chzmin,
    double *stress_flux,int *status)
{
    const int face=blockIdx.x*blockDim.x+threadIdx.x;
    if(face>=nface) return;
    status[face]=V58_OK;stress_flux[face]=0.0;stress_flux[nface+face]=0.0;
    const int left=owner[face],right=neighbour[face];
    if(left<0||left>=ncell||right>=ncell||open_face[face]<0||
       open_face[face]>1||!isfinite(normal[face])||
       !isfinite(normal[nface+face])) {
        status[face]=V58_INVALID_INPUT;return;
    }
    if(fabs(betac)<1.0e-4) return;
    double txx=0.0,txy=0.0,tyy=0.0;
    if(right>=0) {
        txx=0.5*(cell_tensor[left]+cell_tensor[right]);
        txy=0.5*(cell_tensor[ncell+left]+cell_tensor[ncell+right]);
        tyy=0.5*(cell_tensor[2*ncell+left]+cell_tensor[2*ncell+right]);
    } else if(open_face[face]) {
        const double h=trace[face];
        if(!isfinite(h)||h<=dryflc||!isfinite(boundary_equilibrium[left])) {
            status[face]=V58_INVALID_INPUT;return;
        }
        const double u=trace[nface+face]/h,v=trace[2*nface+face]/h;
        v58_tensor(u,v,boundary_equilibrium[left],boundary_chezy[face],
                   betac,gravity,von_karman,chzmin,&txx,&txy,&tyy,
                   &status[face]);
    }
    if(status[face]!=V58_OK) return;
    const double nx=normal[face],ny=normal[nface+face];
    // Conservative face flux -T.n.  The f-wave divergence therefore adds
    // +div(T) to momentum, matching the v58 CPU convention.
    stress_flux[face]=-(txx*nx+txy*ny);
    stress_flux[nface+face]=-(txy*nx+tyy*ny);
    if(!isfinite(stress_flux[face])||!isfinite(stress_flux[nface+face])) {
        stress_flux[face]=0.0;stress_flux[nface+face]=0.0;
        status[face]=V58_INVALID_OUTPUT;
    }
}

extern "C" __global__ void v58_add_frozen_stress(
    const int nface,const double *stress_flux,double *flux_left,
    double *flux_right)
{
    const int face=blockIdx.x*blockDim.x+threadIdx.x;
    if(face>=nface) return;
    flux_left[nface+face]+=stress_flux[face];
    flux_left[2*nface+face]+=stress_flux[nface+face];
    flux_right[nface+face]+=stress_flux[face];
    flux_right[2*nface+face]+=stress_flux[nface+face];
}

extern "C" __global__ void v58_stress_boundary_rate(
    const int nface,const int nboundary,const int *neighbour,
    const int *open_face,const int *boundary_id,const double *length,
    const double *stress_flux,double *boundary_rate,int *status)
{
    const int face=blockIdx.x*blockDim.x+threadIdx.x;
    if(face>=nface) return;
    if(neighbour[face]>=0||!open_face[face]) return;
    const int boundary=boundary_id[face];
    if(boundary<0||boundary>=nboundary||!isfinite(length[face])) {
        status[face]=V58_INVALID_INPUT;return;
    }
    // -flux is the domain momentum-rate contribution at an exterior face.
    atomicAdd(&boundary_rate[boundary],-length[face]*stress_flux[face]);
    atomicAdd(&boundary_rate[nboundary+boundary],
              -length[face]*stress_flux[nface+face]);
}

extern "C" __global__ void v58_diffusion_conductance(
    const int ncell,const int nface,const double *depth,
    const double *boundary_depth,const double diffusivity,
    const double *length,const double *distance,const int *owner,
    const int *neighbour,const int *open_face,double *conductance,int *status)
{
    const int face=blockIdx.x*blockDim.x+threadIdx.x;
    if(face>=nface) return;
    conductance[face]=0.0;status[face]=V58_OK;
    const int left=owner[face],right=neighbour[face];
    if(left<0||left>=ncell||right>=ncell||open_face[face]<0||
       open_face[face]>1||!isfinite(depth[left])||depth[left]<=0.0||
       !isfinite(diffusivity)||diffusivity<0.0||!isfinite(length[face])||
       length[face]<=0.0||!isfinite(distance[face])) {
        status[face]=V58_INVALID_INPUT;return;
    }
    if(diffusivity==0.0||(right<0&&!open_face[face])) return;
    if(distance[face]<=0.0) {
        status[face]=V58_INVALID_GEOMETRY;return;
    }
    double hface=depth[left];
    if(right>=0) {
        if(!isfinite(depth[right])||depth[right]<=0.0) {
            status[face]=V58_INVALID_INPUT;return;
        }
        hface=0.5*(depth[left]+depth[right]);
    } else {
        // Locked FLOW uses the authenticated hydraulic trace depth (its wet
        // face area), not the adjacent cell depth, on an open boundary.
        if(!isfinite(boundary_depth[face])||boundary_depth[face]<=0.0) {
            status[face]=V58_INVALID_INPUT;return;
        }
        hface=boundary_depth[face];
    }
    conductance[face]=length[face]*hface*diffusivity/(0.7*distance[face]);
    if(!isfinite(conductance[face])||conductance[face]<0.0) {
        conductance[face]=0.0;status[face]=V58_INVALID_OUTPUT;
    }
}

extern "C" __global__ void v58_diffusion_setup(
    const int ncell,const int nface,const int *offset,const int *faces,
    const int *owner,const int *neighbour,const int *open_face,
    const double *area,const double *depth,const double *conductance,
    const double *boundary_trace,const double *mass,const double dt,
    double *x,double *rhs,double *diagonal,int *status)
{
    const int cell=blockIdx.x*blockDim.x+threadIdx.x;
    if(cell>=ncell) return;
    status[cell]=V58_OK;
    const double a=area[cell],h=depth[cell],m=mass[cell];
    if(!isfinite(a)||!isfinite(h)||!isfinite(m)||!isfinite(dt)||
       a<=0.0||h<=0.0||dt<0.0||offset[cell]<0||
       offset[cell+1]<offset[cell]) {
        x[cell]=0.0;rhs[cell]=0.0;diagonal[cell]=1.0;
        status[cell]=V58_INVALID_INPUT;return;
    }
    x[cell]=m/h;rhs[cell]=a*m;diagonal[cell]=a*h;
    for(int item=offset[cell];item<offset[cell+1];++item) {
        const int face=faces[item];
        if(face<0||face>=nface) { status[cell]=V58_INVALID_GEOMETRY;continue; }
        const int left=owner[face],right=neighbour[face];
        if(left!=cell&&right!=cell) {
            status[cell]=V58_INVALID_GEOMETRY;continue;
        }
        const double g=conductance[face];
        if(!isfinite(g)||g<0.0) { status[cell]=V58_INVALID_INPUT;continue; }
        if(right>=0) diagonal[cell]+=dt*g;
        else if(open_face[face]&&left==cell) {
            if(!isfinite(boundary_trace[face])) {
                status[cell]=V58_INVALID_INPUT;continue;
            }
            diagonal[cell]+=dt*g;
            rhs[cell]+=dt*g*boundary_trace[face];
        }
    }
    if(!isfinite(x[cell])||!isfinite(rhs[cell])||
       !isfinite(diagonal[cell])||diagonal[cell]<=0.0)
        status[cell]=V58_INVALID_LINEAR_SYSTEM;
}

extern "C" __global__ void v58_diffusion_matvec(
    const int ncell,const int nface,const int *offset,const int *faces,
    const int *owner,const int *neighbour,const int *open_face,
    const double *area,const double *depth,const double *conductance,
    const double dt,const double *vector,double *result,const int *control,
    const int allow_inactive)
{
    const int cell=blockIdx.x*blockDim.x+threadIdx.x;
    if(cell>=ncell) return;
    if(!allow_inactive&&control[0]==0) return;
    double value=area[cell]*depth[cell]*vector[cell];
    for(int item=offset[cell];item<offset[cell+1];++item) {
        const int face=faces[item],left=owner[face],right=neighbour[face];
        const double g=conductance[face];
        if(right>=0) {
            const int other=left==cell?right:left;
            value+=dt*g*(vector[cell]-vector[other]);
        } else if(open_face[face]&&left==cell) value+=dt*g*vector[cell];
    }
    result[cell]=value;
}

extern "C" __global__ void v58_pcg_initialize_vectors(
    const int ncell,const double *rhs,const double *matvec,
    const double *diagonal,double *r,double *z,double *p)
{
    const int cell=blockIdx.x*blockDim.x+threadIdx.x;
    if(cell>=ncell) return;
    r[cell]=rhs[cell]-matvec[cell];z[cell]=r[cell]/diagonal[cell];p[cell]=z[cell];
}

extern "C" __global__ void v58_pcg_initialize_control(
    const double relative_tolerance,const double absolute_tolerance,
    const double *rhs2,const double *rr,const double *rz,const int *cancel,
    double *scalars,int *control)
{
    if(threadIdx.x||blockIdx.x) return;
    control[0]=1;control[1]=V58_OK;control[2]=0;control[3]=0;
    const double rhsnorm=sqrt(fmax(0.0,*rhs2));
    const double residual=sqrt(fmax(0.0,*rr));
    const double target=fmax(absolute_tolerance,
                             relative_tolerance*fmax(1.0,rhsnorm));
    scalars[0]=*rz;scalars[1]=rhsnorm;scalars[2]=residual;
    scalars[3]=target;scalars[4]=0.0;scalars[5]=0.0;
    if(*cancel) { control[0]=0;control[1]=V58_CANCELLED;return; }
    if(!isfinite(rhsnorm)||!isfinite(residual)||!isfinite(target)||
       !isfinite(*rz)||target<0.0) {
        control[0]=0;control[1]=V58_INVALID_LINEAR_SYSTEM;return;
    }
    if(residual<=target) { control[0]=0;control[3]=1; }
}

extern "C" __global__ void v58_pcg_alpha(
    const double *denominator,const int *cancel,double *scalars,int *control)
{
    if(threadIdx.x||blockIdx.x||control[0]==0) return;
    if(*cancel) { control[0]=0;control[1]=V58_CANCELLED;return; }
    if(!isfinite(*denominator)||*denominator<=0.0||!isfinite(scalars[0])) {
        control[0]=0;control[1]=V58_INVALID_LINEAR_SYSTEM;return;
    }
    scalars[4]=scalars[0]/(*denominator);
    if(!isfinite(scalars[4])) {
        control[0]=0;control[1]=V58_INVALID_LINEAR_SYSTEM;
    }
}

extern "C" __global__ void v58_pcg_update_xr(
    const int ncell,const double *p,const double *ap,const double *scalars,
    const int *control,double *x,double *r)
{
    const int cell=blockIdx.x*blockDim.x+threadIdx.x;
    if(cell>=ncell||control[0]==0) return;
    x[cell]+=scalars[4]*p[cell];r[cell]-=scalars[4]*ap[cell];
}

extern "C" __global__ void v58_pcg_precondition(
    const int ncell,const double *r,const double *diagonal,
    const int *control,double *z)
{
    const int cell=blockIdx.x*blockDim.x+threadIdx.x;
    if(cell>=ncell||control[0]==0) return;
    z[cell]=r[cell]/diagonal[cell];
}

extern "C" __global__ void v58_pcg_convergence_beta(
    const int iteration,const double *rr,const double *rz_new,
    const int *cancel,double *scalars,int *control)
{
    if(threadIdx.x||blockIdx.x||control[0]==0) return;
    control[2]=iteration;
    if(*cancel) { control[0]=0;control[1]=V58_CANCELLED;return; }
    const double residual=sqrt(fmax(0.0,*rr));scalars[2]=residual;
    if(!isfinite(residual)||!isfinite(*rz_new)) {
        control[0]=0;control[1]=V58_INVALID_LINEAR_SYSTEM;return;
    }
    if(residual<=scalars[3]) { control[0]=0;control[3]=1;return; }
    if(scalars[0]==0.0) {
        control[0]=0;control[1]=V58_INVALID_LINEAR_SYSTEM;return;
    }
    scalars[5]=(*rz_new)/scalars[0];scalars[0]=*rz_new;
    if(!isfinite(scalars[5])) {
        control[0]=0;control[1]=V58_INVALID_LINEAR_SYSTEM;
    }
}

extern "C" __global__ void v58_pcg_update_p(
    const int ncell,const double *z,const double *scalars,const int *control,
    double *p)
{
    const int cell=blockIdx.x*blockDim.x+threadIdx.x;
    if(cell>=ncell||control[0]==0) return;
    p[cell]=z[cell]+scalars[5]*p[cell];
}

extern "C" __global__ void v58_diffusion_finish(
    const int ncell,const double *depth,const double *x,const int *control,
    double *mass,int *status)
{
    const int cell=blockIdx.x*blockDim.x+threadIdx.x;
    if(cell>=ncell) return;
    if(control[1]!=V58_OK||control[3]==0) {
        status[cell]=control[1]==V58_OK?V58_NOT_CONVERGED:control[1];return;
    }
    const double value=depth[cell]*x[cell];
    if(!isfinite(value)) { status[cell]=V58_INVALID_OUTPUT;return; }
    mass[cell]=value;
}

extern "C" __global__ void v58_diffusion_boundary_ledger(
    const int nface,const int nboundary,const int *owner,
    const int *neighbour,const int *open_face,const int *boundary_id,
    const double *conductance,const double *boundary_trace,const double *x,
    const double dt,double *ledger,int *status)
{
    const int face=blockIdx.x*blockDim.x+threadIdx.x;
    if(face>=nface||neighbour[face]>=0||!open_face[face]) return;
    const int boundary=boundary_id[face];
    if(boundary<0||boundary>=nboundary) {
        status[face]=V58_INVALID_INPUT;return;
    }
    const double value=dt*conductance[face]*
        (boundary_trace[face]-x[owner[face]]);
    if(!isfinite(value)) { status[face]=V58_INVALID_OUTPUT;return; }
    atomicAdd(&ledger[boundary],value);
}
