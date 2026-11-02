// Law-11 boundary trace capacity for the authenticated M1 midpoint slice.
//
// Internal faces remain owned by the existing reconstructed-capacity kernel.
// This kernel evaluates raw open-boundary channels from the actual hydraulic
// trace.  The common resident composer is called by the host immediately
// afterwards; the second kernel below projects that composed bed vector to the
// face.  Composition algebra is deliberately not duplicated here.

// H4 needs the reconstructed vectors *before* donor inventory is applied.
// The historical st_faces kernel exposes those vectors only for its coupled
// MPM branch and combines candidate selection, slope and availability in one
// operation.  Keep that frozen kernel untouched: this reconstruction-only
// kernel is an exact copy of its scalar, ``consistent=false`` face operation.
// It deliberately has no selector, availability, slope, boundary or Exner
// side effect.  The authenticated H4 composer is the sole downstream owner of
// those operations.
__device__ __forceinline__ double m1_face_weno(const double *v)
{
    const double p0=(2.0*v[0]-7.0*v[1]+11.0*v[2])/6.0;
    const double p1=(-v[1]+5.0*v[2]+2.0*v[3])/6.0;
    const double p2=(2.0*v[2]+5.0*v[3]-v[4])/6.0;
    const double a0=v[0]-2.0*v[1]+v[2];
    const double c0=v[0]-4.0*v[1]+3.0*v[2];
    const double a1=v[1]-2.0*v[2]+v[3];
    const double c1=v[1]-v[3];
    const double a2=v[2]-2.0*v[3]+v[4];
    const double c2=3.0*v[2]-4.0*v[3]+v[4];
    const double b0=13.0/12.0*a0*a0+0.25*c0*c0;
    const double b1=13.0/12.0*a1*a1+0.25*c1*c1;
    const double b2=13.0/12.0*a2*a2+0.25*c2*c2;
    double mx=v[0]*v[0];
    for(int k=1;k<5;++k) mx=fmax(mx,v[k]*v[k]);
    const double tau=fabs(b0-b2),eps=1.0e-40+1.0e-14*mx;
    const double r0=tau/(b0+eps),r1=tau/(b1+eps),r2=tau/(b2+eps);
    const double w0=0.1*(1.0+r0*r0);
    const double w1=0.6*(1.0+r1*r1);
    const double w2=0.3*(1.0+r2*r2);
    return ((w0*p0+w1*p1)+w2*p2)/((w0+w1)+w2);
}

__device__ __forceinline__ void m1_reconstruct_side(
    const int c,const int n,const int side,const int f,const int g,
    const int nc,const int nf,const int ng,const double *capacity,
    const double *gradient,const double *displacement,const int *weno,
    const int *open_boundary_distance,const double *xy,const double *mid,
    const double *rectangle_width,const int *owner,const int *neighbour,
    const int *offset,const int *faces,const int *signs,
    const double nx,const double ny,double *x,double *y)
{
    *x=capacity[(long long)g*nc+c];
    *y=capacity[(long long)(ng+g)*nc+c];
    if(n<0) return;
    if(open_boundary_distance[c]<=2) {
        *x=0.5*(*x+capacity[(long long)g*nc+n]);
        *y=0.5*(*y+capacity[(long long)(ng+g)*nc+n]);
        return;
    }
    double value[2]={*x,*y};
    for(int m=0;m<2;++m) {
        if(weno[(long long)side*nf+f]>=0) {
            double stencil[5];
            for(int k=0;k<5;++k)
                stencil[k]=capacity[(long long)(m*ng+g)*nc+
                    weno[(long long)(k*2+side)*nf+f]];
            value[m]=m1_face_weno(stencil);
        } else {
            value[m]+=gradient[(long long)((m*2)*ng+g)*nc+c]*
                          displacement[(long long)side*nf+f]+
                      gradient[(long long)((m*2+1)*ng+g)*nc+c]*
                          displacement[(long long)(2+side)*nf+f];
        }
    }
    if(rectangle_width[c]>0.0 && rectangle_width[nc+c]>0.0) {
        const double dx=mid[f]-xy[c],dy=mid[nf+f]-xy[nc+c];
        int a=-1,b=-1; double r1x=0.0,r1y=0.0;
        for(int k=offset[c];k<offset[c+1];++k) {
            const int ff=faces[k];
            const int other=signs[k]>0?neighbour[ff]:owner[ff];
            if(other<0 || open_boundary_distance[other]<=
                    open_boundary_distance[c]) continue;
            const double rx=xy[other]-xy[c],ry=xy[nc+other]-xy[nc+c];
            const double rr=rx*rx+ry*ry;
            if(fabs(rx*dy-ry*dx)>1.0e-8*rr) continue;
            if(fmax(fabs(rectangle_width[other]-rectangle_width[c]),
                    fabs(rectangle_width[nc+other]-rectangle_width[nc+c]))>
               1.0e-8*fmax(rectangle_width[c],rectangle_width[nc+c])) continue;
            a=other;r1x=rx;r1y=ry;break;
        }
        if(a>=0) for(int k=offset[a];k<offset[a+1];++k) {
            const int ff=faces[k];
            const int other=signs[k]>0?neighbour[ff]:owner[ff];
            if(other<0) continue;
            const double rx=xy[other]-xy[a],ry=xy[nc+other]-xy[nc+a];
            if(fmax(fabs(rx-r1x),fabs(ry-r1y))>
               1.0e-8*sqrt(r1x*r1x+r1y*r1y)) continue;
            if(fmax(fabs(rectangle_width[other]-rectangle_width[c]),
                    fabs(rectangle_width[nc+other]-rectangle_width[nc+c]))>
               1.0e-8*fmax(rectangle_width[c],rectangle_width[nc+c])) continue;
            b=other;break;
        }
        if(b>=0) {
            const double s=(dx*r1x+dy*r1y)/(r1x*r1x+r1y*r1y);
            for(int m=0;m<2;++m) {
                const long long base=(long long)(m*ng+g)*nc;
                const double curvature=0.5*(capacity[base+b]-
                    2.0*capacity[base+a]+capacity[base+c]);
                value[m]=capacity[base+c]+s*(capacity[base+a]-
                    capacity[base+c]-curvature)+(s*s-1.0/12.0)*curvature;
            }
        }
    }
    const double an=capacity[(long long)g*nc+c]*nx+
                    capacity[(long long)(ng+g)*nc+c]*ny;
    const double bn=capacity[(long long)g*nc+n]*nx+
                    capacity[(long long)(ng+g)*nc+n]*ny;
    const double at=-capacity[(long long)g*nc+c]*ny+
                     capacity[(long long)(ng+g)*nc+c]*nx;
    const double bt=-capacity[(long long)g*nc+n]*ny+
                     capacity[(long long)(ng+g)*nc+n]*nx;
    const double vn=fmax(fmin(an,bn),fmin(fmax(an,bn),value[0]*nx+value[1]*ny));
    const double vt=fmax(fmin(at,bt),fmin(fmax(at,bt),-value[0]*ny+value[1]*nx));
    *x=vn*nx-vt*ny;*y=vn*ny+vt*nx;
}

extern "C" __global__ void m1_reconstruct_composed_faces(
    const int nc,const int nf,const int ng,const double *capacity,
    const double *gradient,const double *xy,const double *mid,
    const double *rectangle_width,const double *normal,
    const double *displacement,const int *weno,
    const int *open_boundary_distance,const int *owner,const int *neighbour,
    const int *offset,const int *faces,const int *signs,
    double *face_left,double *face_right)
{
    const int f=blockIdx.x*blockDim.x+threadIdx.x;
    if(f>=nf) return;
    const int c=owner[f],n=neighbour[f];
    for(int g=0;g<ng;++g) {
        const long long at=(long long)g*nf+f;
        face_left[at]=face_left[(long long)ng*nf+at]=0.0;
        face_right[at]=face_right[(long long)ng*nf+at]=0.0;
    }
    if(n<0) return;
    const double nx=normal[f],ny=normal[nf+f];
    for(int g=0;g<ng;++g) {
        double lx,ly,rx,ry;
        m1_reconstruct_side(c,n,0,f,g,nc,nf,ng,capacity,gradient,
            displacement,weno,open_boundary_distance,xy,mid,rectangle_width,
            owner,neighbour,offset,faces,signs,nx,ny,&lx,&ly);
        m1_reconstruct_side(n,c,1,f,g,nc,nf,ng,capacity,gradient,
            displacement,weno,open_boundary_distance,xy,mid,rectangle_width,
            owner,neighbour,offset,faces,signs,nx,ny,&rx,&ry);
        const long long at=(long long)g*nf+f;
        face_left[at]=lx;face_left[(long long)ng*nf+at]=ly;
        face_right[at]=rx;face_right[(long long)ng*nf+at]=ry;
    }
}
__device__ __forceinline__ bool m1_stage_trace_owned(
    const int mode,const double water,const int cell,const int nc,
    const int face,const int nf,const double *q,const double *normal,
    const double grav,const double dry)
{
    if(mode!=2 || water<=0.0) return false;
    const double h=q[cell];
    if(!(isfinite(h)&&h>dry)) return false;
    const double hu=q[nc+cell],hv=q[2*nc+cell];
    const double nx=normal[face],ny=normal[nf+face];
    if(!(isfinite(hu)&&isfinite(hv)&&isfinite(nx)&&isfinite(ny))) return false;
    // Native FLOW only replaces the owner channel at an imposed-stage
    // outlet while the owner is wet and subcritical in the outward-normal
    // direction.  A supercritical outflow owns its existing cell capacity.
    return fabs(hu*nx+hv*ny)<h*sqrt(grav*h);
}

extern "C" __global__ void m1_law11_boundary_faces(
    const int nc,const int nf,const int ng,const int nb,const double *trace,
    const double *q,const double *water,const int *closed,const int *owner,
    const int *neighbour,const int *bid,
    const int *mode,const int *bedmode,const double *normal,
    const double *roughness_parameter,const double *d50,const double *rhos,
    const double *minimum_bedload_depth,
    const double rho,const double grav,
    const double dry,const double minimum_work_depth,
    const int roughness_mode,const int waqua_white_colebrook,
    const double *acal,const double *power_b,
    const double *power_c,const double *ripfac,const double *theta_c,
    const double *sus_acal,const double *sus_power_b,
    const double *sus_power_c,const double *sus_ripfac,
    const double *sus_theta_c,double *bed_trace,double *suspended_trace,
    int *role,int *status)
{
    const int f=blockIdx.x*blockDim.x+threadIdx.x;
    if(f>=nf) return;
    const int c=owner[f],n=neighbour[f],b=bid[f];
    if(n>=0) return;
    role[f]=0;
    if(b<0 || b>=nb) return;
    for(int g=0;g<ng;++g) {
        const long long at=(long long)g*nf+f;
        bed_trace[at]=bed_trace[(long long)ng*nf+at]=0.0;
        suspended_trace[at]=suspended_trace[(long long)ng*nf+at]=0.0;
    }
    if(status[f] || closed[f]) return;
    const int md=mode[b];
    const bool inlet=(md==1||md==6||md==7||md==8||md==9);
    const bool equilibrium_inlet=inlet && water[f]<0.0 && bedmode[b]==1;
    // FLOW recomputes the constitutive channel only for an imposed-stage
    // outlet.  Critical/normal/free outlets retain the already composed
    // owner capacity (native diagnostic role 3); treating every outlet mode
    // as a stage trace can amplify law 11 by orders of magnitude.
    const bool stage_outlet=m1_stage_trace_owned(
        md,water[f],c,nc,f,nf,q,normal,grav,dry);
    const bool owner_outlet=(md==2||md==4) && water[f]>0.0 && !stage_outlet;
    role[f]=equilibrium_inlet ? 1 : (stage_outlet ? 2 :
            (owner_outlet ? 3 : 0));
    if(role[f]!=1 && role[f]!=2) return;
    const double h=trace[f],hu=trace[nf+f],hv=trace[2*nf+f];
    if(h<=dry) return;
    if(!(isfinite(h)&&isfinite(hu)&&isfinite(hv)&&
         isfinite(roughness_parameter[c])) ||
       roughness_parameter[c]<=0.0 || !(isfinite(rho)&&isfinite(grav)) ||
       rho<=0.0 || grav<=0.0) { status[f]=1; return; }
    const double u=hu/h,v=hv/h,speed=sqrt(u*u+v*v);
    if(!isfinite(speed)) { status[f]=1; return; }
    if(speed==0.0) return;
    for(int g=0;g<ng;++g) {
        const long long at=(long long)g*nf+f;
        bool valid=isfinite(d50[g])&&isfinite(rhos[g])&&d50[g]>0.0&&rhos[g]>rho&&
            isfinite(minimum_bedload_depth[g])&&minimum_bedload_depth[g]>=0.0&&
            isfinite(acal[g])&&isfinite(power_b[g])&&isfinite(power_c[g])&&
            isfinite(ripfac[g])&&isfinite(theta_c[g])&&isfinite(sus_acal[g])&&
            isfinite(sus_power_b[g])&&isfinite(sus_power_c[g])&&
            isfinite(sus_ripfac[g])&&isfinite(sus_theta_c[g])&&
            acal[g]>=0.0&&ripfac[g]>=0.0&&theta_c[g]>=0.0&&
            sus_acal[g]>=0.0&&sus_ripfac[g]>=0.0&&sus_theta_c[g]>=0.0;
        if(!valid) { status[f]=1; return; }
        // Match native bed_capacity's shallow transport regularisation.  The
        // constitutive depth is raised to the class threshold while the
        // resolved velocity is retained, then both raw channels receive the
        // same C1 wetting factor.  This is separate from channel composition.
        const double threshold=fmax(minimum_bedload_depth[g],dry);
        double evaluation_depth=h,wet_fraction=1.0;
        if(h<threshold && threshold>dry) {
            const double xi=fmax(0.0,fmin(1.0,(h-dry)/(threshold-dry)));
            wet_fraction=xi*xi*(3.0-2.0*xi);
            evaluation_depth=threshold*(1.0+128.0*2.2204460492503131e-16);
        }
        if(wet_fraction<=0.0) continue;
        double chezy=0.0,z0=0.0,drag=0.0;
        const bool rough=br_roughness_state(roughness_mode,evaluation_depth,
            roughness_parameter[c],grav,minimum_work_depth,
            waqua_white_colebrook,&chezy,&z0,&drag);
        if(!rough) { status[f]=1; return; }
        const double relative=rhos[g]/rho-1.0;
        const double shields=(speed/chezy)*(speed/chezy)/(relative*d50[g]);
        const double bed_excess=fmax(ripfac[g]*shields-theta_c[g],0.0);
        const double sus_excess=fmax(sus_ripfac[g]*shields-sus_theta_c[g],0.0);
        const double scale=pow(d50[g],1.5)*sqrt(grav*relative);
        const double bed_volume=wet_fraction*acal[g]*scale*
            pow(shields,power_b[g])*pow(bed_excess,power_c[g]);
        const double sus_volume=wet_fraction*sus_acal[g]*scale*
            pow(shields,sus_power_b[g])*pow(sus_excess,sus_power_c[g]);
        const double bx=rhos[g]*bed_volume*u/speed;
        const double by=rhos[g]*bed_volume*v/speed;
        const double sx=rhos[g]*sus_volume*u/speed;
        const double sy=rhos[g]*sus_volume*v/speed;
        if(!(isfinite(bx)&&isfinite(by)&&isfinite(sx)&&isfinite(sy))) {
            status[f]=1; return;
        }
        bed_trace[at]=bx;bed_trace[(long long)ng*nf+at]=by;
        suspended_trace[at]=sx;suspended_trace[(long long)ng*nf+at]=sy;
    }
}

// Apply only the existing face policy after common channel composition.
// Internal faces are untouched because st_faces already reconstructed them
// from the composed cell field.
extern "C" __global__ void m1_project_composed_boundary_faces(
    const int nc,const int nf,const int ng,const int nb,const double *water,
    const int *closed,const int *owner,const int *neighbour,const int *bid,
    const int *mode,const int *bedmode,const double *normal,
    const double *availability,const int *role,
    const double *cell_raw_bed,const double *cell_raw_suspended,
    const double *cell_bed,const double *cell_suspended,
    const double *cell_signed,const double *cell_equilibrium,
    const int *cell_composition_status,
    double *boundary_raw_bed,double *boundary_raw_suspended,
    double *boundary_bed,double *boundary_suspended,
    double *boundary_signed,double *boundary_equilibrium,
    int *boundary_composition_status,double *raw,int *status)
{
    const int f=blockIdx.x*blockDim.x+threadIdx.x;
    if(f>=nf || neighbour[f]>=0) return;
    const int c=owner[f],b=bid[f];
    if(b<0 || b>=nb) return;
    const int boundary_role=role[f];
    if(boundary_role==3) {
        // A native role-3 outlet owns the already composed cell channels.
        // Copy diagnostics only; do not pass them through the composer again
        // and, critically, do not alter the base reconstructed face flux.
        for(int g=0;g<ng;++g) {
            const long long cell_scalar=(long long)g*nc+c;
            const long long face_scalar=(long long)g*nf+f;
            const long long cell_y=(long long)ng*nc+cell_scalar;
            const long long face_y=(long long)ng*nf+face_scalar;
            boundary_raw_bed[face_scalar]=cell_raw_bed[cell_scalar];
            boundary_raw_bed[face_y]=cell_raw_bed[cell_y];
            boundary_raw_suspended[face_scalar]=cell_raw_suspended[cell_scalar];
            boundary_raw_suspended[face_y]=cell_raw_suspended[cell_y];
            boundary_bed[face_scalar]=cell_bed[cell_scalar];
            boundary_bed[face_y]=cell_bed[cell_y];
            boundary_suspended[face_scalar]=cell_suspended[cell_scalar];
            boundary_suspended[face_y]=cell_suspended[cell_y];
            boundary_signed[face_scalar]=cell_signed[cell_scalar];
            boundary_equilibrium[face_scalar]=cell_equilibrium[cell_scalar];
            boundary_composition_status[face_scalar]=
                cell_composition_status[cell_scalar];
            if(cell_composition_status[cell_scalar]!=0) status[f]=1;
        }
        return;
    }
    if(boundary_role!=1 && boundary_role!=2) return;
    // Only a trace-owned face is replaced.  Critical/normal/free outlets and
    // prescribed/reversed inlets retain the base reconstructed owner flux.
    for(int g=0;g<ng;++g) raw[(long long)g*nf+f]=0.0;
    if(status[f] || closed[f]) return;
    const double nx=normal[f],ny=normal[nf+f];
    for(int g=0;g<ng;++g) {
        const long long face=(long long)g*nf+f;
        const long long vector=(long long)g*nf+f;
        const double bx=boundary_bed[vector];
        const double by=boundary_bed[(long long)ng*nf+vector];
        const double fraction=availability[(long long)g*nc+c];
        const double projected=(bx*nx+by*ny)*fraction;
        if(!(isfinite(bx)&&isfinite(by)&&isfinite(fraction)&&
             isfinite(projected))) { status[f]=1; return; }
        raw[face]=boundary_role==1 ? fmin(0.0,projected) :
                                    fmax(0.0,projected);
    }
}
