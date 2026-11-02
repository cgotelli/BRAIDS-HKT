// Production kernels for the explicitly selected full-f-wave implicit midpoint.
//
// The nonlinear residual and accepted face flux are assembled by the existing
// augmented f-wave pipeline.  The linear operator below is only a frozen,
// conservative acoustic approximation used by simplified Newton.  Open-boundary
// traces are treated as frozen in that approximation; their complete nonlinear
// closure remains in every authoritative f-wave residual evaluation.

extern "C" __global__ void implicit_linear_acoustic_faces(
    const double* increment,const double* href,const int* owner,
    const int* neighbour,const int* boundary_mode,const double* normal,
    const double* length,long long nc,long long nf,double grav,
    double* divergence,double* gradx,double* grady,double* diffusion)
{
    const long long f=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(f>=nf)return;
    const int i=owner[f],j=neighbour[f];
    const double nx=normal[f],ny=normal[nf+f],ell=length[f];
    if(j<0){
        // Mode zero is the qualified slip wall.  For every prescribed open
        // closure the ghost/section solution is deliberately frozen in W;
        // the full boundary derivative is recovered by nonlinear iterations.
        if(boundary_mode[f]!=0)return;
        const double c=sqrt(grav*href[i]);
        const double mn=increment[nc+i]*nx+increment[2*nc+i]*ny;
        atomicAdd(diffusion+nc+i,-c*ell*mn*nx);
        atomicAdd(diffusion+2*nc+i,-c*ell*mn*ny);
        return;
    }
    const double hface=.5*(href[i]+href[j]);
    const double c=sqrt(grav*hface);
    const double dh=increment[j]-increment[i];
    const double dmx=increment[nc+j]-increment[nc+i];
    const double dmy=increment[2*nc+j]-increment[2*nc+i];
    const double dmn=dmx*nx+dmy*ny;
    const double discharge=.5*((increment[nc+i]+increment[nc+j])*nx+
                                (increment[2*nc+i]+increment[2*nc+j])*ny)-
                           .5*c*dh;
    const double integrated=ell*discharge;
    atomicAdd(divergence+i,integrated);atomicAdd(divergence+j,-integrated);
    const double gx=.5*ell*dh*nx,gy=.5*ell*dh*ny;
    atomicAdd(gradx+i,gx);atomicAdd(gradx+j,gx);
    atomicAdd(grady+i,gy);atomicAdd(grady+j,gy);
    const double k=.5*c*ell;
    const double dx=k*dmn*nx,dy=k*dmn*ny;
    atomicAdd(diffusion+nc+i,dx);atomicAdd(diffusion+nc+j,-dx);
    atomicAdd(diffusion+2*nc+i,dy);atomicAdd(diffusion+2*nc+j,-dy);
}

extern "C" __global__ void implicit_linear_acoustic_finish(
    const double* href,const double* area,const double* divergence,
    const double* gradx,const double* grady,const double* diffusion,
    long long nc,double grav,double* rate)
{
    const long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=nc)return;
    const double inv=1.0/area[i];
    rate[i]=-divergence[i]*inv;
    rate[nc+i]=(-grav*href[i]*gradx[i]+diffusion[nc+i])*inv;
    rate[2*nc+i]=(-grav*href[i]*grady[i]+diffusion[2*nc+i])*inv;
}

// Algebraically identical to resident_gather_update's unlimited derivative,
// but avoids forming q+dt*R and subtracting nearly equal FP64 states.
extern "C" __global__ void implicit_full_rate_from_faces(
    const double* q,const double* area,const double* length,
    const double* normal,const double* fl,const double* fr,
    const int* offset,const int* faces,const int* signs,const int* active,
    long long nc,long long nf,double grav,double* rate)
{
    const long long c=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(c>=nc)return;
    for(int j=0;j<3;++j)rate[j*nc+c]=0.0;
    if(!active[c])return;
    double sum[3]={0.0,0.0,0.0};
    const double pressure=.5*grav*q[c]*q[c];
    for(int k=offset[c];k<offset[c+1];++k){
        const int f=faces[k];const double weight=length[f]/area[c];
        if(signs[k]>0){
            sum[0]+=weight*fl[f];
            sum[1]+=weight*(fl[nf+f]-pressure*normal[f]);
            sum[2]+=weight*(fl[2*nf+f]-pressure*normal[nf+f]);
        }else{
            sum[0]-=weight*fr[f];
            sum[1]-=weight*(fr[nf+f]-pressure*normal[f]);
            sum[2]-=weight*(fr[2*nf+f]-pressure*normal[nf+f]);
        }
    }
    for(int j=0;j<3;++j)rate[j*nc+c]=-sum[j];
}

extern "C" __global__ void implicit_water_divergence_faces(
    const double* flux,const int* owner,const int* neighbour,
    const double* length,long long nf,double* divergence)
{
    const long long f=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(f>=nf)return;
    const int i=owner[f],j=neighbour[f];
    const double integrated=length[f]*flux[f];
    atomicAdd(divergence+i,integrated);
    if(j>=0)atomicAdd(divergence+j,-integrated);
}

extern "C" __global__ void implicit_depth_from_flux(
    const double* old_depth,const double* area,const double* divergence,
    long long nc,double dt,double wet_floor,double* depth,int* status)
{
    const long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=nc)return;
    const double value=old_depth[i]-dt*divergence[i]/area[i];
    depth[i]=value;
    status[i]=(!isfinite(value)||value<=wet_floor)?1:0;
}

extern "C" __global__ void implicit_advective_cfl_faces(
    const double* q,const double* area,const int* owner,const int* neighbour,
    const double* normal,const double* length,long long nc,long long nf,double dt,
    double* cell_metric)
{
    const long long f=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(f>=nf)return;
    const int i=owner[f],j=neighbour[f];
    const double ui=q[nc+i]/q[i],vi=q[2*nc+i]/q[i];
    double speed=fabs(ui*normal[f]+vi*normal[nf+f]);
    if(j>=0){
        const double uj=q[nc+j]/q[j],vj=q[2*nc+j]/q[j];
        speed=fmax(speed,fabs(uj*normal[f]+vj*normal[nf+f]));
    }
    const double value=dt*length[f]*speed;
    atomicAdd(cell_metric+i,value/area[i]);
    if(j>=0)atomicAdd(cell_metric+j,value/area[j]);
}

// Device-resident BiCGSTAB scalar recurrence. Scalar slots are
// rho,rho_old,alpha,omega,denominator,tt,ts,snorm2,rnorm2,tolerance.
// Control slots are done,status,iteration_count.
extern "C" __global__ void implicit_bicg_direction(
    long long n,double* p,const double* r,const double* v,
    const double* scalar,const int* control)
{
    const long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n||control[0]||control[1])return;
    if(control[2]==0)p[i]=r[i];
    else{
        const double beta=(scalar[0]/scalar[1])*(scalar[2]/scalar[3]);
        p[i]=r[i]+beta*(p[i]-scalar[3]*v[i]);
    }
}

extern "C" __global__ void implicit_bicg_alpha(double* scalar,int* control)
{
    if(blockIdx.x||threadIdx.x||control[0]||control[1])return;
    if(!isfinite(scalar[0])||!isfinite(scalar[4])||
       fabs(scalar[0])<=1.e-300||fabs(scalar[4])<=1.e-300){control[1]=1;return;}
    scalar[2]=scalar[0]/scalar[4];
}

extern "C" __global__ void implicit_bicg_residual_stage(
    long long n,const double* r,const double* v,const double* scalar,
    const int* control,double* s)
{
    const long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n)return;
    s[i]=(control[0]||control[1])?0.0:r[i]-scalar[2]*v[i];
}

extern "C" __global__ void implicit_bicg_omega(double* scalar,int* control)
{
    if(blockIdx.x||threadIdx.x||control[0]||control[1])return;
    if(!isfinite(scalar[5])||!isfinite(scalar[6])||!isfinite(scalar[7])){
        control[1]=2;return;
    }
    if(scalar[5]<=1.e-300){
        if(sqrt(fmax(scalar[7],0.0))<=scalar[9])scalar[3]=0.0;
        else control[1]=3;
        return;
    }
    scalar[3]=scalar[6]/scalar[5];
    if(!isfinite(scalar[3]))control[1]=4;
}

extern "C" __global__ void implicit_bicg_update(
    long long n,double* x,double* r,const double* p,const double* s,
    const double* t,const double* scalar,const int* control)
{
    const long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n||control[0]||control[1])return;
    x[i]+=scalar[2]*p[i]+scalar[3]*s[i];
    r[i]=s[i]-scalar[3]*t[i];
}

extern "C" __global__ void implicit_bicg_convergence(double* scalar,int* control)
{
    if(blockIdx.x||threadIdx.x||control[0]||control[1])return;
    if(!isfinite(scalar[8])){control[1]=5;return;}
    ++control[2];scalar[1]=scalar[0];
    if(sqrt(fmax(scalar[8],0.0))<=scalar[9])control[0]=1;
}
