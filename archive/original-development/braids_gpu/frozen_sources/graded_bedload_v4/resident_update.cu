// Research-only, FP64 resident hydraulic face/update pipeline.
// Concatenate AFTER the separately frozen augmented_fwave.cu source.
// This does not reconstruct WENO traces, solve boundary characteristics,
// integrate friction, advance a complete timestep, update AMR, or sediment.
// Cell arrays and face arrays are component-major (C-contiguous SoA).
// Ownership/CSR indices are zero-based; neighbour=-1 denotes exterior.

__device__ __forceinline__ double resident_dot(
    double x, double y, double nx, double ny) { return x * nx + y * ny; }

extern "C" __global__ void resident_prepare_faces(
    const double* trace, const double* q, const double* bed,
    const double* normal, const double* distance,
    const int* owner, const int* neighbour, const int* kind,
    long long nc, long long nf, double grav, double dry,
    double* local, int* closed)
{
    const long long f = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nf) return;
    for (int j=0; j<8; ++j) local[j*nf+f] = 0.0;
    closed[f] = 0;
    if (kind[f] == 0 || kind[f] == 3) return;
    const int c=owner[f], n=neighbour[f];
    const double nx=normal[f], ny=normal[nf+f];
    double hL=trace[f], xL=trace[nf+f], yL=trace[2*nf+f], bL=trace[3*nf+f];
    double hR=trace[4*nf+f], xR=trace[5*nf+f], yR=trace[6*nf+f], bR=trace[7*nf+f];
    if (kind[f] == 2) {
        hR=hL; bR=bL;
        const double dot=resident_dot(xL,yL,nx,ny);
        xR=xL-2.0*dot*nx; yR=yL-2.0*dot*ny;
    }
    const double unL=resident_dot(xL,yL,nx,ny), utL=resident_dot(xL,yL,-ny,nx);
    const double unR=resident_dot(xR,yR,nx,ny), utR=resident_dot(xR,yR,-ny,nx);
    local[f]=hL; local[nf+f]=unL; local[2*nf+f]=utL; local[3*nf+f]=bL;
    local[4*nf+f]=hR; local[5*nf+f]=unR; local[6*nf+f]=utR; local[7*nf+f]=bR;
    if (kind[f] != 1 || fmin(q[c],q[n])>dry) return;
    const double dc=distance[f], dn=distance[nf+f];
    double shared;
    if (dc+dn>2.2250738585072014e-308)
        shared=(dn*(bed[c]+q[c])+dc*(bed[n]+q[n]))/(dc+dn)-fmax(bed[c],bed[n]);
    else shared=0.5*((bed[c]+q[c])+(bed[n]+q[n]))-fmax(bed[c],bed[n]);
    bool blocked=shared<=dry;
    if (blocked) {
        bool opens=false;
        if (hL>dry && hR<=dry && unL/hL>0.0) {
            RiemannType type=braids_riemanntype(hL,hL,unL/hL,-unL/hL,1,dry,grav);
            opens=type.hm>hL+dry && type.hm+bL>bR+dry;
        } else if (hR>dry && hL<=dry && -unR/hR>0.0) {
            RiemannType type=braids_riemanntype(hR,hR,-unR/hR,unR/hR,1,dry,grav);
            opens=type.hm>hR+dry && type.hm+bR>bL+dry;
        }
        blocked=!opens;
    }
    closed[f]=blocked ? 1 : 0;
}

extern "C" __global__ void resident_assemble_flux(
    const double* trace, const double* local, const double* riemann,
    const double* q, const double* bed, const double* normal,
    const int* owner, const int* neighbour, const int* kind,
    const int* boundary_mode, const int* sampled, const double* bedoffset,
    const int* hydro_constant, const int* closed,
    long long nc, long long nf, double grav, double dry,
    double* fl, double* fr, double* speed, int* face_status)
{
    const long long f=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if (f>=nf) return;
    face_status[f]=0; speed[f]=0.0;
    for (int j=0;j<3;++j) { fl[j*nf+f]=0.0; fr[j*nf+f]=0.0; }
    if (kind[f]==0) return;
    const int c=owner[f], n=neighbour[f];
    const double nx=normal[f], ny=normal[nf+f], hL=trace[f];
    double hR=kind[f]==2 ? hL : trace[4*nf+f];
    double l[3]={0.0,0.0,0.0}, r[3]={0.0,0.0,0.0};
    if (kind[f]==3) {
        l[0]=resident_dot(trace[nf+f],trace[2*nf+f],nx,ny);
        if (hL>dry) {
            l[1]=l[0]*trace[nf+f]/hL+0.5*grav*(hL*hL)*nx;
            l[2]=l[0]*trace[2*nf+f]/hL+0.5*grav*(hL*hL)*ny;
        }
        speed[f]=sqrt(grav*hL);
        if (hL>dry) speed[f]+=fabs(l[0])/hL;
        for (int j=0;j<3;++j) r[j]=l[j];
    } else if (closed[f]) {
        l[1]=0.5*grav*(hL*hL)*nx; l[2]=0.5*grav*(hL*hL)*ny;
        r[1]=0.5*grav*(hR*hR)*nx; r[2]=0.5*grav*(hR*hR)*ny;
        if (hL>dry) speed[f]=fmax(speed[f],fabs(local[nf+f]/hL)+sqrt(grav*hL));
        if (hR>dry) speed[f]=fmax(speed[f],fabs(local[5*nf+f]/hR)+sqrt(grav*hR));
    } else {
        double pl[3]={0.0,0.0,0.0}, pr[3]={0.0,0.0,0.0};
        if (hL>dry) {
            const double hu=local[nf+f], hv=local[2*nf+f];
            pl[0]=hu; pl[1]=(hu*hu)/hL+0.5*grav*(hL*hL); pl[2]=hu*hv/hL;
        }
        if (hR>dry) {
            const double hu=local[5*nf+f], hv=local[6*nf+f];
            pr[0]=hu; pr[1]=(hu*hu)/hR+0.5*grav*(hR*hR); pr[2]=hu*hv/hR;
        }
        for (int j=0;j<3;++j) { pl[j]+=riemann[j*nf+f]; pr[j]-=riemann[(3+j)*nf+f]; }
        l[0]=pl[0]; l[1]=pl[1]*nx+pl[2]*(-ny); l[2]=pl[1]*ny+pl[2]*nx;
        r[0]=pr[0]; r[1]=pr[1]*nx+pr[2]*(-ny); r[2]=pr[1]*ny+pr[2]*nx;
        if (fabs(l[0]-r[0])>1.e-10*fmax(1.0,fmax(fabs(l[0]),fabs(r[0])))) face_status[f]=1;
        l[0]=0.5*(l[0]+r[0]); r[0]=l[0];
        speed[f]=fmax(fabs(riemann[6*nf+f]),fmax(fabs(riemann[7*nf+f]),fabs(riemann[8*nf+f])));
        if (kind[f]==2) {
            l[0]=0.0;
            const double pressure=resident_dot(l[1],l[2],nx,ny);
            l[1]=pressure*nx; l[2]=pressure*ny;
            for(int j=0;j<3;++j) r[j]=l[j];
        }
    }
    if (kind[f]!=3) {
        if(q[c]>dry) {
            const double correction=hydro_constant[c] ?
                0.5*grav*(q[c]*q[c]-hL*hL) :
                0.5*grav*(q[c]+trace[8*nf+f])*(trace[3*nf+f]-bed[c]);
            l[1]+=correction*nx; l[2]+=correction*ny;
        }
        if(n>=0) {
            if(q[n]>dry) {
                const double correction=hydro_constant[n] ?
                    0.5*grav*(q[n]*q[n]-hR*hR) :
                    0.5*grav*(q[n]+trace[9*nf+f])*(trace[7*nf+f]-bed[n]);
                r[1]+=correction*nx; r[2]+=correction*ny;
            }
        } else for(int j=0;j<3;++j) r[j]=l[j];
    } else if(boundary_mode[f]==2) {
        if(q[c]>dry && fabs(trace[3*nf+f]-bed[c])>64.0*2.2204460492503131e-16*fmax(1.0,fabs(bed[c])) &&
            fabs(resident_dot(q[nc+c],q[2*nc+c],nx,ny))<q[c]*sqrt(grav*q[c])) {
            const double depth=trace[8*nf+f];
            const double correction=hydro_constant[c] ?
                0.5*grav*(q[c]*q[c]-depth*depth) :
                0.5*grav*(q[c]+depth)*(trace[3*nf+f]-bed[c]);
            l[1]+=correction*nx; l[2]+=correction*ny;
        }
        for(int j=0;j<3;++j) r[j]=l[j];
    } else if(sampled[f]==1) {
        if(q[c]>dry) {
            const double correction=0.5*grav*(q[c]+hL)*bedoffset[f];
            l[1]+=correction*nx; l[2]+=correction*ny;
        }
        for(int j=0;j<3;++j) r[j]=l[j];
    }
    for(int j=0;j<3;++j) {
        fl[j*nf+f]=l[j]; fr[j*nf+f]=r[j];
        if(!isfinite(l[j]) || !isfinite(r[j])) face_status[f]|=4;
    }
}

extern "C" __global__ void resident_draining(
    const double* q,const double* area,const double* length,const double* fl,
    const int* offset,const int* faces,const int* signs,const int* active,
    long long nc,long long nf,double dt,double* theta)
{
    const long long c=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(c>=nc)return;
    theta[c]=1.0; if(!active[c])return;
    double outgoing=0.0;
    for(int k=offset[c];k<offset[c+1];++k) {
        const int f=faces[k]; outgoing+=fmax(signs[k]*fl[f],0.0)*length[f];
    }
    if(dt*outgoing>=q[c]*area[c] && outgoing>0.0)
        theta[c]=fmin(1.0,q[c]*area[c]/(dt*outgoing))*(1.0-32.0*2.2204460492503131e-16);
}

extern "C" __global__ void resident_limit_flux(
    const double* q,const double* theta,const int* owner,const int* neighbour,
    const int* kind,long long nc,long long nf,double dry,
    double* fl,double* fr,int* face_status)
{
    const long long f=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(f>=nf || fl[f]==0.0)return;
    const int donor=fl[f]<0.0 ? neighbour[f] : owner[f];
    if(donor<0 || theta[donor]==1.0)return;
    if(kind[f]==3) { face_status[f]|=2;return; }
    const double correction=(theta[donor]-1.0)*fl[f];
    fl[f]*=theta[donor];fr[f]=fl[f];
    for(int j=1;j<3;++j) {
        const double velocity=q[donor]>dry ? q[j*nc+donor]/q[donor] : 0.0;
        fl[j*nf+f]+=correction*velocity;fr[j*nf+f]+=correction*velocity;
    }
}

extern "C" __global__ void resident_gather_update(
    const double* q,const double* area,const double* length,const double* normal,
    const double* fl,const double* fr,const double* speed,const double* maxdepth,
    const int* offset,const int* faces,const int* signs,const int* active,
    long long nc,long long nf,double dt,double grav,double cfl_target,
    double* next,double* cfl_ratio,int* cell_status)
{
    const long long c=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(c>=nc)return;
    cell_status[c]=0;cfl_ratio[c]=0.0;
    for(int j=0;j<3;++j) next[j*nc+c]=q[j*nc+c];
    if(!active[c])return;
    double sum[3]={0.0,0.0,0.0},cfl=0.0;
    const double pressure=0.5*grav*(q[c]*q[c]);
    for(int k=offset[c];k<offset[c+1];++k) {
        const int f=faces[k]; const double weight=length[f]/area[c];
        cfl+=length[f]*speed[f];
        if(signs[k]>0) {
            sum[0]+=weight*fl[f];
            sum[1]+=weight*(fl[nf+f]-pressure*normal[f]);
            sum[2]+=weight*(fl[2*nf+f]-pressure*normal[nf+f]);
        } else {
            sum[0]-=weight*fr[f];
            sum[1]-=weight*(fr[nf+f]-pressure*normal[f]);
            sum[2]-=weight*(fr[2*nf+f]-pressure*normal[nf+f]);
        }
    }
    cfl_ratio[c]=dt*cfl/area[c];
    if(cfl_ratio[c]>cfl_target){cell_status[c]=2;return;}
    for(int j=0;j<3;++j) next[j*nc+c]=q[j*nc+c]-dt*sum[j];
    if(next[c]<0.0) {
        if(-next[c]<=64.0*2.2204460492503131e-16*fmax(maxdepth[0],1.0))
            for(int j=0;j<3;++j) next[j*nc+c]=0.0;
        else cell_status[c]=3;
    }
}
