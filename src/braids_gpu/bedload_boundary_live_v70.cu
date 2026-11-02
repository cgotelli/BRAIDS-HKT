// Q3/v70 noncohesive bed-vector boundary ownership.
// Positive flux is outward.  Prescribed feed is stored positive inward and
// therefore emitted with a negative sign.  Constitutive capacity is supplied
// by the authenticated scalar/direct producer family; this file contains no
// transport formula.

__device__ __forceinline__ long long bnd70_vec(
    const int component,const int grain,const int point,
    const int nclass,const int npoint)
{
    return ((long long)component*nclass+grain)*npoint+point;
}

extern "C" __global__ void bed_boundary_v70_candidate(
    const int ncell,const int nface,const int nclass,
    const int *owner,const int *mode,const double *normal,
    const double *hydraulic_outward,const double *owner_vector,
    const double *trace_vector,const double *prescribed_mass,
    double *potential,int *requires_availability,int *source,int *status)
{
    const long long point=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    const long long count=(long long)nclass*nface;
    if(point>=count) return;
    const int grain=(int)(point/nface);
    const int face=(int)(point-(long long)grain*nface);
    potential[point]=0.0;requires_availability[point]=0;source[point]=0;
    const int cell=owner[face];
    const double water=hydraulic_outward[face];
    if(cell<0 || cell>=ncell || mode[face]<0 || mode[face]>3 ||
       !isfinite(water) || !isfinite(normal[face]) ||
       !isfinite(normal[nface+face])) {
        status[face]=1;return;
    }
    if(mode[face]==0 || water==0.0) return;
    if(water<0.0) {
        if(mode[face]==2) {
            const double rate=prescribed_mass[point];
            if(!isfinite(rate) || rate<0.0) { status[face]=1;return; }
            potential[point]=-rate;source[point]=1;
        } else if(mode[face]==1) {
            const double x=trace_vector[bnd70_vec(0,grain,face,nclass,nface)];
            const double y=trace_vector[bnd70_vec(1,grain,face,nclass,nface)];
            const double value=x*normal[face]+y*normal[nface+face];
            if(!isfinite(value)) { status[face]=1;return; }
            potential[point]=fmin(0.0,value);
            requires_availability[point]=1;source[point]=2;
        }
    } else {
        const double x=owner_vector[bnd70_vec(0,grain,cell,nclass,ncell)];
        const double y=owner_vector[bnd70_vec(1,grain,cell,nclass,ncell)];
        const double value=x*normal[face]+y*normal[nface+face];
        if(!isfinite(value)) { status[face]=1;return; }
        potential[point]=fmax(0.0,value);
        requires_availability[point]=1;source[point]=3;
    }
}

extern "C" __global__ void bed_boundary_v70_demand_fill(
    const int ncell,const int nface,const int nclass,
    const int *owner,const int *requires_availability,
    const double *potential,const double *face_length,const double *cell_area,
    const double *bed_mass,const double *active_fraction,
    const double dt,const double morfac,
    double *demand,double *fill,double *availability,
    int *status)
{
    const long long point=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    const long long count=(long long)nclass*ncell;
    if(point>=count) return;
    const int grain=(int)(point/ncell);
    const int cell=(int)(point-(long long)grain*ncell);
    if(!isfinite(cell_area[cell]) || cell_area[cell]<=0.0 ||
       !isfinite(bed_mass[point]) || bed_mass[point]<0.0 ||
       !isfinite(active_fraction[point]) || active_fraction[point]<0.0 ||
       active_fraction[point]>1.0 || !isfinite(dt) || dt<=0.0 ||
       !isfinite(morfac) || morfac<0.0) {
        status[cell]=1;demand[point]=0.0;
        fill[point]=0.0;availability[point]=0.0;return;
    }
    double required=0.0;
    for(int face=0;face<nface;++face) {
        if(owner[face]!=cell) continue;
        const long long at=(long long)grain*nface+face;
        const double value=potential[at];
        if(!isfinite(value) || !isfinite(face_length[face]) ||
           face_length[face]<0.0) { status[cell]=1;continue; }
        if(value>0.0 && requires_availability[at]!=0)
            required+=value*active_fraction[point]*face_length[face]*dt*morfac/
                      cell_area[cell];
    }
    double factor=bed_mass[point]>0.0 ? 1.0 : 0.0;
    if(required>0.0) factor=fmin(factor,bed_mass[point]/required);
    if(!isfinite(factor) || factor<0.0 || factor>1.0) {
        status[cell]=1;factor=0.0;
    }
    demand[point]=required;fill[point]=factor;
    availability[point]=active_fraction[point]*factor;
}

extern "C" __global__ void bed_boundary_v70_apply(
    const int ncell,const int nface,const int nclass,
    const int *owner,const int *requires_availability,
    const double *potential,const double *active_fraction,const double *fill,
    double *face_flux,int *status)
{
    const long long point=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    const long long count=(long long)nclass*nface;
    if(point>=count) return;
    const int grain=(int)(point/nface);
    const int face=(int)(point-(long long)grain*nface);
    const int cell=owner[face];
    if(cell<0 || cell>=ncell) { status[face]=1;face_flux[point]=0.0;return; }
    double value=potential[point];
    if(requires_availability[point]!=0)
        value*=active_fraction[(long long)grain*ncell+cell]*
               fill[(long long)grain*ncell+cell];
    if(!isfinite(value)) { status[face]=1;value=0.0; }
    face_flux[point]=value;
}

extern "C" __global__ void bed_boundary_v70_cell_delta(
    const int ncell,const int nface,const int nclass,
    const int *owner,const double *face_length,const double *cell_area,
    const double *face_flux,const double *old_mass,
    const double dt,const double morfac,double *new_mass,double *delta,
    int *status)
{
    const long long point=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    const long long count=(long long)nclass*ncell;
    if(point>=count) return;
    const int grain=(int)(point/ncell);
    const int cell=(int)(point-(long long)grain*ncell);
    double integrated=0.0;
    for(int face=0;face<nface;++face) if(owner[face]==cell)
        integrated+=face_flux[(long long)grain*nface+face]*face_length[face];
    const double change=-dt*morfac*integrated/cell_area[cell];
    const double candidate=old_mass[point]+change;
    const double tolerance=2048.0*2.2204460492503131e-16*
        fmax(1.0,fmax(old_mass[point],fabs(change)));
    if(!isfinite(change) || !isfinite(candidate) || candidate < -tolerance) {
        status[cell]=1;new_mass[point]=old_mass[point];delta[point]=0.0;return;
    }
    new_mass[point]=fmax(0.0,candidate);
    delta[point]=new_mass[point]-old_mass[point];
}

extern "C" __global__ void bed_boundary_v70_ledger(
    const int nface,const int nclass,const int nboundary,
    const int *boundary_index,const double *face_length,const double *face_flux,
    const double dt,const double morfac,double *signed_transfer,
    double *absolute_transfer,int *status)
{
    const long long point=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    const long long count=(long long)nclass*nboundary;
    if(point>=count) return;
    const int grain=(int)(point/nboundary);
    const int boundary=(int)(point-(long long)grain*nboundary);
    double signed_value=0.0,absolute_value=0.0;
    for(int face=0;face<nface;++face) {
        if(boundary_index[face]<0 || boundary_index[face]>=nboundary) {
            status[boundary]=1;continue;
        }
        if(boundary_index[face]!=boundary) continue;
        const double amount=face_flux[(long long)grain*nface+face]*
                            face_length[face]*dt*morfac;
        if(!isfinite(amount)) { status[boundary]=1;continue; }
        signed_value+=amount;absolute_value+=fabs(amount);
    }
    signed_transfer[point]=signed_value;
    absolute_transfer[point]=absolute_value;
}
