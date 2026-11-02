// Common resident face and Exner consumers for qualified Q3 capacity vectors.
//
// Nonlinear cell and boundary-trace capacities are produced by
// ResidentM1CapacityDispatch.  This file deliberately contains no second
// constitutive implementation: every downstream consumer receives the exact
// same split ABI and explicit boundary role.  The face kernel consumes the
// returned vector directly; direct-vector laws are never externally rotated.

__device__ __forceinline__ long long q3_vec(
    const int component,const int grain,const int point,
    const int nclass,const int npoint)
{
    return ((long long)component*nclass+grain)*npoint+point;
}

extern "C" __global__ void q3_scalar_face_dispatch(
    const int ncell,const int nface,const int nclass,
    const int *owner,const int *neighbour,const int *role,
    const double *normal,const double *left_bed,const double *right_bed,
    const double *cell_bed,const double *trace_bed,
    const double *availability,double *face_flux,int *donor,int *status)
{
    const int face=blockIdx.x*blockDim.x+threadIdx.x;
    if(face>=nface) return;
    status[face]=0;
    const int left_cell=owner[face];
    const int right_cell=neighbour[face];
    if(left_cell<0 || left_cell>=ncell || right_cell < -1 || right_cell>=ncell) {
        status[face]=1; return;
    }
    const double nx=normal[face],ny=normal[nface+face];
    if(!isfinite(nx) || !isfinite(ny)) { status[face]=1; return; }
    for(int grain=0;grain<nclass;++grain) {
        const long long out=(long long)grain*nface+face;
        face_flux[out]=0.0;donor[out]=0;
        const double owner_available=availability[(long long)grain*ncell+left_cell];
        const double neighbour_available=right_cell>=0 ?
            availability[(long long)grain*ncell+right_cell] : 0.0;
        if(!isfinite(owner_available) || owner_available<0.0 ||
           !isfinite(neighbour_available) || neighbour_available<0.0) {
            status[face]=1; continue;
        }
        double x=0.0,y=0.0,scale=0.0;
        if(right_cell>=0) {
            if(role[face]!=0) { status[face]=1; continue; }
            const double lx=left_bed[q3_vec(0,grain,face,nclass,nface)];
            const double ly=left_bed[q3_vec(1,grain,face,nclass,nface)];
            const double rx=right_bed[q3_vec(0,grain,face,nclass,nface)];
            const double ry=right_bed[q3_vec(1,grain,face,nclass,nface)];
            const double ln=lx*nx+ly*ny,rn=rx*nx+ry*ny;
            if(ln>0.0 && rn>0.0) {
                x=lx;y=ly;scale=owner_available;donor[out]=1;
            } else if(ln<0.0 && rn<0.0) {
                x=rx;y=ry;scale=neighbour_available;donor[out]=2;
            } else {
                x=0.5*(lx+rx);y=0.5*(ly+ry);
                if(x*nx+y*ny<0.0) {
                    scale=neighbour_available;donor[out]=2;
                } else {
                    scale=owner_available;donor[out]=1;
                }
            }
            face_flux[out]=(x*nx+y*ny)*scale;
        } else if(role[face]==1 || role[face]==2) {
            x=trace_bed[q3_vec(0,grain,face,nclass,nface)];
            y=trace_bed[q3_vec(1,grain,face,nclass,nface)];
            const double projected=(x*nx+y*ny)*owner_available;
            face_flux[out]=role[face]==1 ? fmin(0.0,projected) :
                                           fmax(0.0,projected);
            donor[out]=3;
        } else if(role[face]==3) {
            x=cell_bed[q3_vec(0,grain,left_cell,nclass,ncell)];
            y=cell_bed[q3_vec(1,grain,left_cell,nclass,ncell)];
            face_flux[out]=fmax(0.0,(x*nx+y*ny)*owner_available);
            donor[out]=1;
        } else if(role[face]==4) {
            // Explicit closed face.  Keep exact zero flux and no donor.
            face_flux[out]=0.0;
            donor[out]=0;
        } else if(role[face]!=0) {
            status[face]=1;
        }
        if(!isfinite(face_flux[out])) {
            face_flux[out]=0.0;donor[out]=0;status[face]=1;
        }
    }
}

extern "C" __global__ void q3_scalar_exner_candidate(
    const int ncell,const int nface,const int nadjacency,const int nclass,
    const int *offset,const int *faces,const int *signs,
    const double *face_length,const double *area,const double *face_flux,
    const int *face_status,
    const double *old_mass,const double dt,const double morphological_factor,
    double *new_mass,double *delta_mass,int *status)
{
    const int cell=blockIdx.x*blockDim.x+threadIdx.x;
    if(cell>=ncell) return;
    status[cell]=0;
    // A rejected candidate must expose the unchanged accepted state, never
    // uninitialised device memory.  Initialise every class before validating
    // topology so even a malformed offset table is transaction-safe.
    for(int grain=0;grain<nclass;++grain) {
        const long long at=(long long)grain*ncell+cell;
        new_mass[at]=old_mass[at];delta_mass[at]=0.0;
    }
    if(offset[cell]<0 || offset[cell]>offset[cell+1] ||
       offset[cell+1]>nadjacency || !isfinite(area[cell]) || area[cell]<=0.0 ||
       !isfinite(dt) || dt<0.0 || !isfinite(morphological_factor) ||
       morphological_factor<0.0) {
        status[cell]=1;return;
    }
    for(int grain=0;grain<nclass;++grain) {
        const long long at=(long long)grain*ncell+cell;
        if(!isfinite(old_mass[at]) || old_mass[at]<0.0) {
            status[cell]=1;continue;
        }
        double integrated=0.0;
        bool valid=true;
        for(int item=offset[cell];item<offset[cell+1];++item) {
            const int face=faces[item],sign=signs[item];
            if(face<0 || face>=nface || (sign!=1 && sign!=-1) ||
               face_status[face]!=0 ||
               !isfinite(face_length[face]) || face_length[face]<0.0 ||
               !isfinite(face_flux[(long long)grain*nface+face])) {
                valid=false;break;
            }
            integrated+=(double)sign*face_flux[(long long)grain*nface+face]*
                face_length[face];
        }
        const double change=-dt*morphological_factor*integrated/area[cell];
        const double candidate=old_mass[at]+change;
        const double tolerance=128.0*2.2204460492503131e-16*
            fmax(1.0,fmax(old_mass[at],fabs(change)));
        valid=valid && isfinite(change) && isfinite(candidate) &&
            candidate>=-tolerance;
        if(valid) {
            new_mass[at]=fmax(0.0,candidate);
            delta_mass[at]=new_mass[at]-old_mass[at];
        } else {
            status[cell]=1;
        }
    }
    // A cell update is one transaction across all sediment classes.  If any
    // class rejects, expose the complete accepted vector and a zero delta;
    // never leave an earlier class committed beside a rejected later class.
    if(status[cell]!=0) {
        for(int grain=0;grain<nclass;++grain) {
            const long long at=(long long)grain*ncell+cell;
            new_mass[at]=old_mass[at];delta_mass[at]=0.0;
        }
    }
}
