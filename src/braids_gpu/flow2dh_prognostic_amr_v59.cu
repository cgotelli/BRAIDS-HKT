// Isolated Q3-H3/v59 CUDA operators for a synchronous selected AMR mesh.
//
// This source deliberately does not include or modify the accepted v57/v58
// sources.  Physical faces are canonical non-overlapping segments.  The
// signed secondary state is M=hI; its conserved inventory is A*M.

enum V59Status {
    V59_OK = 0,
    V59_INVALID_INPUT = 1,
    V59_INVALID_GEOMETRY = 2,
    V59_INVALID_LINEAR_SYSTEM = 3,
    V59_NOT_CONVERGED = 4,
    V59_CANCELLED = 5,
    V59_INVALID_OUTPUT = 6,
    V59_INVALID_TOPOLOGY = 7
};

__device__ __forceinline__ void v59_tensor(
    const double u,const double v,const double intensity,const double chezy,
    const double betac,const double gravity,const double von_karman,
    const double chzmin,double *txx,double *txy,double *tyy,int *status)
{
    *txx=0.0;*txy=0.0;*tyy=0.0;
    if(!isfinite(u)||!isfinite(v)||!isfinite(intensity)||!isfinite(chezy)||
       !isfinite(betac)||!isfinite(gravity)||!isfinite(von_karman)||
       !isfinite(chzmin)||chezy<=0.0||gravity<=0.0||von_karman<=0.0||
       chzmin<=0.0) {
        *status=V59_INVALID_INPUT;return;
    }
    if(fabs(betac)<1.0e-4) return;
    const double speed=fmax(1.0e-8,sqrt(u*u+v*v));
    const double alpha=sqrt(gravity)/(von_karman*fmax(chezy,chzmin));
    const double beta_star=betac*
        (5.0*alpha-15.6*alpha*alpha+37.5*alpha*alpha*alpha);
    double beta=beta_star*intensity/speed;
    // Literal locked FLOW/v58 clamp; this is not sign preserving.
    if(fabs(beta)>fabs(0.1*beta_star)) beta=0.1*beta_star;
    *txx=2.0*beta*u*v;
    *txy=beta*(u*u-v*v);
    *tyy=-*txx;
    if(!isfinite(*txx)||!isfinite(*txy)||!isfinite(*tyy)) {
        *txx=0.0;*txy=0.0;*tyy=0.0;*status=V59_INVALID_OUTPUT;
    }
}

extern "C" __global__ void v59_restrict_family(
    const int ncell,const int family,const int *family_parent,
    const int *child_offset,const int *children,const double *area,
    double *q,double *mass,int *active,const double dry,const int switch_selection,
    const int require_children_active,int *status)
{
    if(threadIdx.x||blockIdx.x) return;
    status[0]=V59_OK;
    const int parent=family_parent[family];
    const int first=child_offset[family],last=child_offset[family+1];
    if(parent<0||parent>=ncell||last-first!=4||first<0||last<first) {
        status[0]=V59_INVALID_TOPOLOGY;return;
    }
    double total_area=0.0,sum_q0=0.0,sum_q1=0.0,sum_q2=0.0,sum_m=0.0;
    for(int k=first;k<last;++k) {
        const int child=children[k];
        if(child<0||child>=ncell||child==parent||
           (require_children_active&&active[child]!=1)||
           !isfinite(area[child])||area[child]<=0.0||
           !isfinite(q[child])||q[child]<=dry||!isfinite(q[ncell+child])||
           !isfinite(q[2*ncell+child])||!isfinite(mass[child])) {
            status[0]=V59_INVALID_TOPOLOGY;return;
        }
        const double a=area[child];
        total_area+=a;sum_q0+=a*q[child];sum_q1+=a*q[ncell+child];
        sum_q2+=a*q[2*ncell+child];sum_m+=a*mass[child];
    }
    if(!isfinite(dry)||dry<=0.0||!isfinite(area[parent])||area[parent]<=0.0||
       fabs(total_area-area[parent])>
           2.0e-13*fmax(1.0,fabs(area[parent]))) {
        status[0]=V59_INVALID_TOPOLOGY;return;
    }
    q[parent]=sum_q0/area[parent];
    q[ncell+parent]=sum_q1/area[parent];
    q[2*ncell+parent]=sum_q2/area[parent];
    mass[parent]=sum_m/area[parent];
    if(!isfinite(q[parent])||q[parent]<=dry||!isfinite(q[ncell+parent])||
       !isfinite(q[2*ncell+parent])||!isfinite(mass[parent])) {
        status[0]=V59_INVALID_OUTPUT;return;
    }
    if(switch_selection) {
        for(int k=first;k<last;++k) active[children[k]]=0;
        active[parent]=1;
    }
}

extern "C" __global__ void v59_prolong_family(
    const int ncell,const int family,const int *family_parent,
    const int *child_offset,const int *children,const double *area,
    const double *centre,const double *child_depth,const double *gradient,
    const double intensity_min,const double intensity_max,const double dry,
    double *q,double *mass,int *active,int *status)
{
    if(threadIdx.x||blockIdx.x) return;
    status[0]=V59_OK;
    const int parent=family_parent[family];
    const int first=child_offset[family],last=child_offset[family+1];
    if(parent<0||parent>=ncell||last-first!=4||active[parent]!=1||
       !isfinite(intensity_min)||!isfinite(intensity_max)||
       intensity_min>intensity_max||!isfinite(dry)||dry<0.0||
       !isfinite(gradient[0])||!isfinite(gradient[1])) {
        status[0]=V59_INVALID_TOPOLOGY;return;
    }
    const double hp=q[parent],ap=area[parent];
    if(!isfinite(hp)||hp<=dry||!isfinite(ap)||ap<=0.0||
       !isfinite(mass[parent])||!isfinite(q[ncell+parent])||
       !isfinite(q[2*ncell+parent])) {
        status[0]=V59_INVALID_INPUT;return;
    }
    const double ip=mass[parent]/hp;
    if(ip<intensity_min-2.0e-14||ip>intensity_max+2.0e-14) {
        status[0]=V59_INVALID_INPUT;return;
    }
    double water=0.0,child_area=0.0;
    for(int k=first;k<last;++k) {
        const int child=children[k];
        const double h=child_depth[child];
        if(child<0||child>=ncell||child==parent||active[child]!=0||
           !isfinite(area[child])||area[child]<=0.0||
           !isfinite(h)||h<=dry||!isfinite(centre[child])||
           !isfinite(centre[ncell+child])) {
            status[0]=V59_INVALID_TOPOLOGY;return;
        }
        child_area+=area[child];water+=area[child]*h;
    }
    const double target_water=ap*hp;
    if(fabs(child_area-ap)>2.0e-13*fmax(1.0,fabs(ap))||
       fabs(water-target_water)>2.0e-13*fmax(1.0,fabs(target_water))) {
        status[0]=V59_INVALID_INPUT;return;
    }
    const double up=q[ncell+parent]/hp,vp=q[2*ncell+parent]/hp;
    double inventory=0.0;
    for(int k=first;k<last;++k) {
        const int child=children[k];
        const double h=child_depth[child];
        double intensity=ip+gradient[0]*(centre[child]-centre[parent])+
            gradient[1]*(centre[ncell+child]-centre[ncell+parent]);
        intensity=fmin(intensity_max,fmax(intensity_min,intensity));
        q[child]=h;q[ncell+child]=h*up;q[2*ncell+child]=h*vp;
        mass[child]=h*intensity;
        inventory+=area[child]*mass[child];
    }
    const double target=ap*mass[parent];
    double residual=target-inventory,total_capacity=0.0;
    for(int k=first;k<last;++k) {
        const int child=children[k];
        const double h=q[child],intensity=mass[child]/h;
        total_capacity+=area[child]*h*(residual>=0.0?
            intensity_max-intensity:intensity-intensity_min);
    }
    const double scale=fmax(1.0,fmax(fabs(target),fabs(inventory)));
    if(fabs(residual)>2.0e-14*scale) {
        if(!isfinite(total_capacity)||total_capacity<=0.0||
           fabs(residual)>total_capacity+2.0e-13*scale) {
            status[0]=V59_INVALID_OUTPUT;return;
        }
        for(int k=first;k<last;++k) {
            const int child=children[k];
            const double h=q[child],intensity=mass[child]/h;
            const double capacity=area[child]*h*(residual>=0.0?
                intensity_max-intensity:intensity-intensity_min);
            const double share=residual*capacity/total_capacity;
            mass[child]+=share/area[child];
        }
    }
    inventory=0.0;
    for(int k=first;k<last;++k) inventory+=area[children[k]]*mass[children[k]];
    residual=target-inventory;
    int best=-1;double best_capacity=-1.0;
    for(int k=first;k<last;++k) {
        const int child=children[k];
        const double h=q[child],intensity=mass[child]/h;
        const double capacity=area[child]*h*(residual>=0.0?
            intensity_max-intensity:intensity-intensity_min);
        if(capacity>best_capacity) { best_capacity=capacity;best=child; }
    }
    if(residual!=0.0) {
        if(best<0||fabs(residual)>best_capacity+2.0e-13*scale) {
            status[0]=V59_INVALID_OUTPUT;return;
        }
        mass[best]+=residual/area[best];
    }
    inventory=0.0;
    for(int k=first;k<last;++k) {
        const int child=children[k];
        const double intensity=mass[child]/q[child];
        if(!isfinite(intensity)||intensity<intensity_min-2.0e-13||
           intensity>intensity_max+2.0e-13) {
            status[0]=V59_INVALID_OUTPUT;return;
        }
        inventory+=area[child]*mass[child];
    }
    if(fabs(inventory-target)>2.0e-13*scale) {
        status[0]=V59_INVALID_OUTPUT;return;
    }
    for(int k=first;k<last;++k) active[children[k]]=1;
    active[parent]=0;
}

extern "C" __global__ void v59_advective_segments(
    const int ncell,const int nface,const int *owner,const int *neighbour,
    const double *length,const double *midpoint,const double *centre,
    const double *normal_discharge,const double *accepted_q,
    const double *accepted_mass,const double *gradient,const double *bounds,
    const int *active,const double dt,double *face_register,
    double *cell_increment,int *status)
{
    const int face=blockIdx.x*blockDim.x+threadIdx.x;
    if(face>=nface) return;
    status[face]=V59_OK;face_register[face]=0.0;
    const int left=owner[face],right=neighbour[face];
    if(left<0||left>=ncell||right<0||right>=ncell||active[left]!=1||
       active[right]!=1||!isfinite(length[face])||length[face]<=0.0||
       !isfinite(normal_discharge[face])||!isfinite(dt)||dt<0.0) {
        status[face]=V59_INVALID_TOPOLOGY;return;
    }
    const int source=normal_discharge[face]>=0.0?left:right;
    const double h=accepted_q[source];
    if(!isfinite(h)||h<=0.0||!isfinite(accepted_mass[source])||
       !isfinite(gradient[source])||!isfinite(gradient[ncell+source])||
       !isfinite(bounds[source])||!isfinite(bounds[ncell+source])||
       bounds[source]>bounds[ncell+source]) {
        status[face]=V59_INVALID_INPUT;return;
    }
    const double cell_intensity=accepted_mass[source]/h;
    const double bound_scale=fmax(1.0,fmax(fabs(cell_intensity),
        fmax(fabs(bounds[source]),fabs(bounds[ncell+source]))));
    if(!isfinite(cell_intensity)||
       cell_intensity<bounds[source]-2.0e-13*bound_scale||
       cell_intensity>bounds[ncell+source]+2.0e-13*bound_scale) {
        status[face]=V59_INVALID_INPUT;return;
    }
    double intensity=cell_intensity+
        gradient[source]*(midpoint[face]-centre[source])+
        gradient[ncell+source]*(midpoint[nface+face]-centre[ncell+source]);
    if(!isfinite(intensity)) { status[face]=V59_INVALID_OUTPUT;return; }
    intensity=fmin(bounds[ncell+source],fmax(bounds[source],intensity));
    const double transfer=dt*length[face]*normal_discharge[face]*intensity;
    if(!isfinite(transfer)) { status[face]=V59_INVALID_OUTPUT;return; }
    face_register[face]=transfer;
}

extern "C" __global__ void v59_gather_scalar_register(
    const int ncell,const int nface,const int *offset,const int *faces,
    const int *owner,const int *neighbour,const int *active,
    const double *face_register,double *cell_increment,int *status)
{
    const int cell=blockIdx.x*blockDim.x+threadIdx.x;
    if(cell>=ncell) return;
    cell_increment[cell]=0.0;
    if(status[cell]!=V59_OK||active[cell]!=1) return;
    if(offset[cell]<0||offset[cell+1]<offset[cell]) {
        status[cell]=V59_INVALID_TOPOLOGY;return;
    }
    double value=0.0;
    for(int item=offset[cell];item<offset[cell+1];++item) {
        const int face=faces[item];
        if(face<0||face>=nface) {
            status[cell]=V59_INVALID_TOPOLOGY;return;
        }
        const int left=owner[face],right=neighbour[face];
        if(left==cell) value-=face_register[face];
        else if(right==cell) value+=face_register[face];
        else { status[cell]=V59_INVALID_TOPOLOGY;return; }
    }
    if(!isfinite(value)) { status[cell]=V59_INVALID_OUTPUT;return; }
    cell_increment[cell]=value;
}

extern "C" __global__ void v59_gather_vector_register(
    const int ncell,const int nface,const int *offset,const int *faces,
    const int *owner,const int *neighbour,const int *active,
    const double *depth,const double *face_register,double *cell_increment,
    int *status)
{
    const int cell=blockIdx.x*blockDim.x+threadIdx.x;
    if(cell>=ncell) return;
    cell_increment[cell]=0.0;cell_increment[ncell+cell]=0.0;
    if(status[cell]!=V59_OK||active[cell]!=1) return;
    if(!isfinite(depth[cell])||depth[cell]<=0.0) {
        status[cell]=V59_INVALID_INPUT;return;
    }
    if(offset[cell]<0||offset[cell+1]<offset[cell]) {
        status[cell]=V59_INVALID_TOPOLOGY;return;
    }
    double x=0.0,y=0.0;
    for(int item=offset[cell];item<offset[cell+1];++item) {
        const int face=faces[item];
        if(face<0||face>=nface) {
            status[cell]=V59_INVALID_TOPOLOGY;return;
        }
        const int left=owner[face],right=neighbour[face];
        if(left==cell) {
            x-=face_register[face];y-=face_register[nface+face];
        } else if(right==cell) {
            x+=face_register[face];y+=face_register[nface+face];
        } else { status[cell]=V59_INVALID_TOPOLOGY;return; }
    }
    if(!isfinite(x)||!isfinite(y)) {
        status[cell]=V59_INVALID_OUTPUT;return;
    }
    cell_increment[cell]=depth[cell]*x;
    cell_increment[ncell+cell]=depth[cell]*y;
}

extern "C" __global__ void v59_apply_mass_increment(
    const int ncell,const double *area,const int *active,
    const double *increment,double *mass,int *status)
{
    const int cell=blockIdx.x*blockDim.x+threadIdx.x;
    if(cell>=ncell) return;
    status[cell]=V59_OK;
    if(active[cell]!=1) return;
    if(!isfinite(area[cell])||area[cell]<=0.0||!isfinite(increment[cell])||
       !isfinite(mass[cell])) { status[cell]=V59_INVALID_INPUT;return; }
    const double value=mass[cell]+increment[cell]/area[cell];
    if(!isfinite(value)) { status[cell]=V59_INVALID_OUTPUT;return; }
    mass[cell]=value;
}

extern "C" __global__ void v59_diffusion_conductance(
    const int ncell,const int nface,const double *depth,const double diffusivity,
    const double schmidt,const double *length,const double *distance,
    const int *owner,const int *neighbour,const int *active,
    double *conductance,int *status)
{
    const int face=blockIdx.x*blockDim.x+threadIdx.x;
    if(face>=nface) return;
    conductance[face]=0.0;status[face]=V59_OK;
    const int left=owner[face],right=neighbour[face];
    if(left<0||left>=ncell||right<0||right>=ncell||active[left]!=1||
       active[right]!=1||!isfinite(depth[left])||depth[left]<=0.0||
       !isfinite(depth[right])||depth[right]<=0.0||
       !isfinite(diffusivity)||diffusivity<0.0||!isfinite(schmidt)||
       schmidt<=0.0||!isfinite(length[face])||length[face]<=0.0||
       !isfinite(distance[face])||distance[face]<=0.0) {
        status[face]=V59_INVALID_GEOMETRY;return;
    }
    conductance[face]=length[face]*0.5*(depth[left]+depth[right])*
        diffusivity/(schmidt*distance[face]);
    if(!isfinite(conductance[face])||conductance[face]<0.0) {
        conductance[face]=0.0;status[face]=V59_INVALID_OUTPUT;
    }
}

extern "C" __global__ void v59_diffusion_setup(
    const int ncell,const int nface,const int *offset,const int *faces,
    const int *owner,const int *neighbour,const double *area,
    const double *depth,const double *conductance,const double *mass,
    const int *active,const double dt,double *x,double *rhs,double *diagonal,
    int *status)
{
    const int cell=blockIdx.x*blockDim.x+threadIdx.x;
    if(cell>=ncell) return;
    status[cell]=V59_OK;
    if(active[cell]!=1) { x[cell]=0.0;rhs[cell]=0.0;diagonal[cell]=1.0;return; }
    const double a=area[cell],h=depth[cell],m=mass[cell];
    if(!isfinite(a)||a<=0.0||!isfinite(h)||h<=0.0||!isfinite(m)||
       !isfinite(dt)||dt<0.0||offset[cell]<0||offset[cell+1]<offset[cell]) {
        x[cell]=0.0;rhs[cell]=0.0;diagonal[cell]=1.0;
        status[cell]=V59_INVALID_INPUT;return;
    }
    x[cell]=m/h;rhs[cell]=a*m;diagonal[cell]=a*h;
    for(int item=offset[cell];item<offset[cell+1];++item) {
        const int face=faces[item];
        if(face<0||face>=nface) { status[cell]=V59_INVALID_GEOMETRY;continue; }
        const int left=owner[face],right=neighbour[face];
        if((left!=cell&&right!=cell)||active[left]!=1||active[right]!=1) {
            status[cell]=V59_INVALID_TOPOLOGY;continue;
        }
        const double g=conductance[face];
        if(!isfinite(g)||g<0.0) { status[cell]=V59_INVALID_INPUT;continue; }
        diagonal[cell]+=dt*g;
    }
    if(!isfinite(x[cell])||!isfinite(rhs[cell])||!isfinite(diagonal[cell])||
       diagonal[cell]<=0.0) status[cell]=V59_INVALID_LINEAR_SYSTEM;
}

extern "C" __global__ void v59_diffusion_matvec(
    const int ncell,const int nface,const int *offset,const int *faces,
    const int *owner,const int *neighbour,const double *area,
    const double *depth,const double *conductance,const int *active,
    const double dt,const double *vector,double *result,const int *control,
    const int allow_inactive)
{
    const int cell=blockIdx.x*blockDim.x+threadIdx.x;
    if(cell>=ncell) return;
    if(!allow_inactive&&control[0]==0) return;
    if(active[cell]!=1) { result[cell]=vector[cell];return; }
    double value=area[cell]*depth[cell]*vector[cell];
    for(int item=offset[cell];item<offset[cell+1];++item) {
        const int face=faces[item],left=owner[face],right=neighbour[face];
        const int other=left==cell?right:left;
        value+=dt*conductance[face]*(vector[cell]-vector[other]);
    }
    result[cell]=value;
}

extern "C" __global__ void v59_pcg_initialize_vectors(
    const int ncell,const double *rhs,const double *matvec,
    const double *diagonal,double *r,double *z,double *p)
{
    const int cell=blockIdx.x*blockDim.x+threadIdx.x;
    if(cell>=ncell) return;
    r[cell]=rhs[cell]-matvec[cell];z[cell]=r[cell]/diagonal[cell];p[cell]=z[cell];
}

extern "C" __global__ void v59_pcg_initialize_control(
    const double relative_tolerance,const double absolute_tolerance,
    const double *rhs2,const double *rr,const double *rz,const int *cancel,
    double *scalars,int *control)
{
    if(threadIdx.x||blockIdx.x) return;
    control[0]=1;control[1]=V59_OK;control[2]=0;control[3]=0;
    const double rhsnorm=sqrt(fmax(0.0,*rhs2));
    const double residual=sqrt(fmax(0.0,*rr));
    const double target=fmax(absolute_tolerance,
                             relative_tolerance*rhsnorm);
    scalars[0]=*rz;scalars[1]=rhsnorm;scalars[2]=residual;
    scalars[3]=target;scalars[4]=0.0;scalars[5]=0.0;
    if(*cancel) { control[0]=0;control[1]=V59_CANCELLED;return; }
    if(!isfinite(rhsnorm)||!isfinite(residual)||!isfinite(target)||
       !isfinite(*rz)||target<0.0) {
        control[0]=0;control[1]=V59_INVALID_LINEAR_SYSTEM;return;
    }
    if(residual<=target) { control[0]=0;control[3]=1; }
}

extern "C" __global__ void v59_pcg_alpha(
    const double *denominator,const int *cancel,double *scalars,int *control)
{
    if(threadIdx.x||blockIdx.x||control[0]==0) return;
    if(*cancel) { control[0]=0;control[1]=V59_CANCELLED;return; }
    if(!isfinite(*denominator)||*denominator<=0.0||!isfinite(scalars[0])) {
        control[0]=0;control[1]=V59_INVALID_LINEAR_SYSTEM;return;
    }
    scalars[4]=scalars[0]/(*denominator);
    if(!isfinite(scalars[4])) {
        control[0]=0;control[1]=V59_INVALID_LINEAR_SYSTEM;
    }
}

extern "C" __global__ void v59_pcg_update_xr(
    const int ncell,const double *p,const double *ap,const double *scalars,
    const int *control,double *x,double *r)
{
    const int cell=blockIdx.x*blockDim.x+threadIdx.x;
    if(cell>=ncell||control[0]==0) return;
    x[cell]+=scalars[4]*p[cell];r[cell]-=scalars[4]*ap[cell];
}

extern "C" __global__ void v59_pcg_precondition(
    const int ncell,const double *r,const double *diagonal,
    const int *control,double *z)
{
    const int cell=blockIdx.x*blockDim.x+threadIdx.x;
    if(cell>=ncell||control[0]==0) return;
    z[cell]=r[cell]/diagonal[cell];
}

extern "C" __global__ void v59_pcg_convergence_beta(
    const int iteration,const double *rr,const double *rz_new,
    const int *cancel,double *scalars,int *control)
{
    if(threadIdx.x||blockIdx.x||control[0]==0) return;
    control[2]=iteration;
    if(*cancel) { control[0]=0;control[1]=V59_CANCELLED;return; }
    const double residual=sqrt(fmax(0.0,*rr));scalars[2]=residual;
    if(!isfinite(residual)||!isfinite(*rz_new)) {
        control[0]=0;control[1]=V59_INVALID_LINEAR_SYSTEM;return;
    }
    if(residual<=scalars[3]) { control[0]=0;control[3]=1;return; }
    if(scalars[0]==0.0) {
        control[0]=0;control[1]=V59_INVALID_LINEAR_SYSTEM;return;
    }
    scalars[5]=(*rz_new)/scalars[0];scalars[0]=*rz_new;
    if(!isfinite(scalars[5])) {
        control[0]=0;control[1]=V59_INVALID_LINEAR_SYSTEM;
    }
}

extern "C" __global__ void v59_pcg_update_p(
    const int ncell,const double *z,const double *scalars,const int *control,
    double *p)
{
    const int cell=blockIdx.x*blockDim.x+threadIdx.x;
    if(cell>=ncell||control[0]==0) return;
    p[cell]=z[cell]+scalars[5]*p[cell];
}

extern "C" __global__ void v59_diffusion_finish(
    const int ncell,const double *depth,const double *x,const int *active,
    const int *control,const double *input_mass,double *candidate,int *status)
{
    const int cell=blockIdx.x*blockDim.x+threadIdx.x;
    if(cell>=ncell) return;
    candidate[cell]=input_mass[cell];
    if(active[cell]!=1) return;
    if(control[1]!=V59_OK||control[3]==0) {
        status[cell]=control[1]==V59_OK?V59_NOT_CONVERGED:control[1];return;
    }
    const double value=depth[cell]*x[cell];
    if(!isfinite(value)) { status[cell]=V59_INVALID_OUTPUT;return; }
    candidate[cell]=value;
}

extern "C" __global__ void v59_diffusion_register(
    const int ncell,const int nface,const int *owner,const int *neighbour,
    const double *conductance,const double *intensity,const double dt,
    double *face_register,double *cell_increment,int *status)
{
    const int face=blockIdx.x*blockDim.x+threadIdx.x;
    if(face>=nface) return;
    const int left=owner[face],right=neighbour[face];
    const double transfer=dt*conductance[face]*
        (intensity[left]-intensity[right]);
    if(!isfinite(transfer)) { status[face]=V59_INVALID_OUTPUT;return; }
    face_register[face]=transfer;
}

extern "C" __global__ void v59_betac_cells(
    const int ncell,const double *q,const double *mass,const double *chezy,
    const int *active,const double dry,const double betac,
    const double gravity,const double von_karman,const double chzmin,
    double *tensor,int *status)
{
    const int cell=blockIdx.x*blockDim.x+threadIdx.x;
    if(cell>=ncell) return;
    status[cell]=V59_OK;
    tensor[cell]=0.0;tensor[ncell+cell]=0.0;tensor[2*ncell+cell]=0.0;
    if(active[cell]!=1) return;
    const double h=q[cell];
    if(!isfinite(h)||h<=dry||!isfinite(mass[cell])) {
        status[cell]=V59_INVALID_INPUT;return;
    }
    v59_tensor(q[ncell+cell]/h,q[2*ncell+cell]/h,mass[cell]/h,
               chezy[cell],betac,gravity,von_karman,chzmin,&tensor[cell],
               &tensor[ncell+cell],&tensor[2*ncell+cell],&status[cell]);
}

extern "C" __global__ void v59_betac_segments(
    const int ncell,const int nface,const int *owner,const int *neighbour,
    const double *normal,const double *length,const double *tensor,
    const int *active,const double dt,double *flux,double *face_register,
    double *cell_increment,int *status)
{
    const int face=blockIdx.x*blockDim.x+threadIdx.x;
    if(face>=nface) return;
    status[face]=V59_OK;
    const int left=owner[face],right=neighbour[face];
    if(left<0||left>=ncell||right<0||right>=ncell||active[left]!=1||
       active[right]!=1||!isfinite(normal[face])||
       !isfinite(normal[nface+face])||!isfinite(length[face])||
       length[face]<=0.0||!isfinite(dt)||dt<0.0) {
        status[face]=V59_INVALID_TOPOLOGY;return;
    }
    const double txx=0.5*(tensor[left]+tensor[right]);
    const double txy=0.5*(tensor[ncell+left]+tensor[ncell+right]);
    const double tyy=0.5*(tensor[2*ncell+left]+tensor[2*ncell+right]);
    const double nx=normal[face],ny=normal[nface+face];
    const double fx=-(txx*nx+txy*ny),fy=-(txy*nx+tyy*ny);
    const double jx=dt*length[face]*fx,jy=dt*length[face]*fy;
    if(!isfinite(jx)||!isfinite(jy)) {
        status[face]=V59_INVALID_OUTPUT;return;
    }
    flux[face]=fx;flux[nface+face]=fy;
    face_register[face]=jx;face_register[nface+face]=jy;
}

extern "C" __global__ void v59_reduce_vector_cell_increment_ordered(
    const int ncell,const double *cell_increment,double *ledger)
{
    if(threadIdx.x||blockIdx.x) return;
    double x=0.0,y=0.0;
    for(int cell=0;cell<ncell;++cell) {
        x+=cell_increment[cell];y+=cell_increment[ncell+cell];
    }
    ledger[0]=x;ledger[1]=y;
}

extern "C" __global__ void v59_coriolis_active_leaves(
    const int ncell,const double coriolis,const double dt,const double *area,
    const int *active,double *q,double *cell_increment,int *status)
{
    const int cell=blockIdx.x*blockDim.x+threadIdx.x;
    if(cell>=ncell) return;
    status[cell]=V59_OK;cell_increment[cell]=0.0;
    cell_increment[ncell+cell]=0.0;
    if(active[cell]!=1) return;
    const double qx=q[ncell+cell],qy=q[2*ncell+cell],a=area[cell];
    if(!isfinite(qx)||!isfinite(qy)||!isfinite(a)||a<=0.0||
       !isfinite(coriolis)||!isfinite(dt)||dt<0.0) {
        status[cell]=V59_INVALID_INPUT;return;
    }
    const double angle=coriolis*dt,co=cos(angle),si=sin(angle);
    const double nx=co*qx+si*qy,ny=-si*qx+co*qy;
    if(!isfinite(nx)||!isfinite(ny)) {
        status[cell]=V59_INVALID_OUTPUT;return;
    }
    q[ncell+cell]=nx;q[2*ncell+cell]=ny;
    cell_increment[cell]=a*(nx-qx);
    cell_increment[ncell+cell]=a*(ny-qy);
}
