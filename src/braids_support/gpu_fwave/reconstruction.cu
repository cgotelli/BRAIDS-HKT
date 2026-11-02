// Research-only fixed-grid, hydraulic scheme-6 reconstruction.
// FP64, no fast math; arithmetic follows the frozen alpha82 Fortran routines.
// This is not a sediment reconstruction or a boundary characteristic solver.
// Arrays are C-contiguous SoA, final axis cells/faces; missing indices are -1.

__device__ __forceinline__ double recon_weno(const double* v) {
    const double p0=(2.0*v[0]-7.0*v[1]+11.0*v[2])/6.0;
    const double p1=(-v[1]+5.0*v[2]+2.0*v[3])/6.0;
    const double p2=(2.0*v[2]+5.0*v[3]-v[4])/6.0;
    const double a0=v[0]-2.0*v[1]+v[2], c0=v[0]-4.0*v[1]+3.0*v[2];
    const double a1=v[1]-2.0*v[2]+v[3], c1=v[1]-v[3];
    const double a2=v[2]-2.0*v[3]+v[4], c2=3.0*v[2]-4.0*v[3]+v[4];
    const double b0=(13.0/12.0)*(a0*a0)+0.25*(c0*c0);
    const double b1=(13.0/12.0)*(a1*a1)+0.25*(c1*c1);
    const double b2=(13.0/12.0)*(a2*a2)+0.25*(c2*c2);
    double scale=v[0]*v[0];
    for(int k=1;k<5;++k) scale=fmax(scale,v[k]*v[k]);
    const double tau=fabs(b0-b2), eps=1.e-40+1.e-14*scale;
    const double r0=tau/(b0+eps),r1=tau/(b1+eps),r2=tau/(b2+eps);
    const double w0=0.1*(1.0+r0*r0),w1=0.6*(1.0+r1*r1),w2=0.3*(1.0+r2*r2);
    return ((w0*p0+w1*p1)+w2*p2)/((w0+w1)+w2);
}

extern "C" __global__ void reconstruction_primitive(
    const double* q,const double* bed,const int* owner,const int* neighbour,const int* bid,
    const int* offset,const int* faces,const int* signs,long long nc,double dry,
    double* primitive,int* active,int* status) {
    const long long c=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(c>=nc)return;
    const double h=q[c];
    primitive[c]=h+bed[c]; primitive[nc+c]=0.;primitive[2*nc+c]=0.;primitive[3*nc+c]=bed[c];
    if(h>dry) {primitive[nc+c]=q[nc+c]/h;primitive[2*nc+c]=q[2*nc+c]/h;}
    active[c]=h>dry?1:0;
    if(!active[c])for(int k=offset[c];k<offset[c+1];++k) {
        const int f=faces[k],n=signs[k]<0?owner[f]:neighbour[f];
        if(bid[f]>=0 || (n>=0 && q[n]>dry)) {active[c]=1;break;}
    }
    status[c]=h<0.0 || !isfinite(h) || !isfinite(bed[c]) || !isfinite(q[nc+c]) || !isfinite(q[2*nc+c]) ? 1:0;
}

// One cell owns all its gradient, wall and flattening work. Only primitive
// values and immutable geometry are read from other cells: no inter-cell race.
extern "C" __global__ void reconstruction_cells(
    const double* q,const double* v,const double* xy,const double* mid,const double* normal,
    const int* owner,const int* neighbour,const int* bid,const int* boundary_mode,
    const int* offset,const int* faces,const int* signs,const int* active,
    const double* ls_matrix,const double* ls_weight,const double* ls_delta,
    const double* rectangle,const int* weno,const double* displacement,
    const int* triangle,const int* wall_ids,const int* wall_kind,const double* stencil_normal,
    const int* wall_slot,const double* wall_normal,const double* wall_mean_distance,const double* wall_box_center,
    long long nc,long long nf,long long nadj,double dry,double grav,double dx,int outlet_extension,
    double* grad,double* vlo,double* vhi,double* wall_values,int* wall_ready,int* constant)
{
    const long long c=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(c>=nc)return;
    double centre[4],lo[4],hi[4],gx[4]={0.,0.,0.,0.},gy[4]={0.,0.,0.,0.};
    double values[4][4]; int ready[4]={0,0,0,0};
    for(int m=0;m<4;++m) {centre[m]=v[m*nc+c];lo[m]=centre[m];hi[m]=centre[m];for(int s=0;s<4;++s)values[m][s]=0.;}
    if(active[c]) {
        double bx[4]={0.,0.,0.,0.},by[4]={0.,0.,0.,0.};
        const double a11=ls_matrix[c],a12=ls_matrix[nc+c],a22=ls_matrix[2*nc+c],det=ls_matrix[3*nc+c];
        for(int k=offset[c];k<offset[c+1];++k) {
            const int f=faces[k], n=signs[k]<0 ? owner[f]:neighbour[f];
            if(n<0)continue; // Hydraulic bounds do not consume concentration BCs.
            const double rx=ls_delta[k],ry=ls_delta[nadj+k],w=ls_weight[k];
            for(int m=0;m<4;++m) {
                double adjacent=v[m*nc+n];
                if(m==0 && centre[0]-centre[3]>dry && v[n]-v[3*nc+n]<=dry && v[3*nc+n]>=centre[0])adjacent=centre[0];
                bx[m]+=w*rx*(adjacent-centre[m]);by[m]+=w*ry*(adjacent-centre[m]);
                lo[m]=fmin(lo[m],adjacent);hi[m]=fmax(hi[m],adjacent);
            }
        }
        if(det>1.e-12*((a11+a22)*(a11+a22))) {
            for(int m=0;m<4;++m) {gx[m]=(a22*bx[m]-a12*by[m])/det;gy[m]=(a11*by[m]-a12*bx[m])/det;}
            const double h=centre[0]-centre[3];
            if(outlet_extension && h>dry) {
                for(int k=offset[c];k<offset[c+1];++k) {
                    const int f=faces[k];
                    if(neighbour[f]>=0 || bid[f]<0 || boundary_mode[f]!=2)continue;
                    if(fabs(centre[1]*normal[f]+centre[2]*normal[nf+f])>=sqrt(grav*h))continue;
                    const double rx=mid[f]-xy[c],ry=mid[nf+f]-xy[nc+c];
                    if(fabs(gx[3]*rx+gy[3]*ry)<=64.0*2.2204460492503131e-16*fmax(1.0,fabs(centre[3])))continue;
                    for(int m=0;m<4;++m) {const double exterior=centre[m]+2.0*(gx[m]*rx+gy[m]*ry);lo[m]=fmin(lo[m],exterior);hi[m]=fmax(hi[m],exterior);}
                }
            }
            double theta[4]={1.,1.,1.,1.};
            for(int k=offset[c];k<offset[c+1];++k) {
                const int f=faces[k];const double rx=mid[f]-xy[c],ry=mid[nf+f]-xy[nc+c];
                if(neighbour[f]<0 && bid[f]>=0 && boundary_mode[f]==2 && h>dry &&
                   fabs(centre[1]*normal[f]+centre[2]*normal[nf+f])<sqrt(grav*h) &&
                   fabs(gx[3]*rx+gy[3]*ry)>64.0*2.2204460492503131e-16*fmax(1.0,fabs(centre[3])))continue;
                for(int m=0;m<4;++m) {const double delta=gx[m]*rx+gy[m]*ry;
                    // A roundoff-sized transverse increment must not erase a finite streamwise slope.
                    if(fabs(delta)<=64.0*2.2204460492503131e-16*fmax(1.0,fmax(fabs(centre[m]),fmax(fabs(lo[m]),fabs(hi[m])))))continue;
                    if(delta>0.)theta[m]=fmin(theta[m],(hi[m]-centre[m])/delta);
                    if(delta<0.)theta[m]=fmin(theta[m],(lo[m]-centre[m])/delta);
                }
            }
            for(int m=0;m<4;++m) {gx[m]*=fmax(0.,theta[m]);gy[m]*=fmax(0.,theta[m]);}
        }
    }
    bool walls=q[c]>dry;
    if(triangle[c])for(int a=0;a<2;++a)for(int k=0;k<5;++k)if(wall_ids[(k*2+a)*nc+c]<0)walls=false;
    if(walls) {
        const double tx=-wall_normal[nc+c],ty=wall_normal[c];
        double tlo=centre[1]*tx+centre[2]*ty,thi=tlo;
        if(triangle[c])for(int k=offset[c];k<offset[c+1];++k) {
            const int f=faces[k],other=signs[k]<0?owner[f]:neighbour[f];if(other<0)continue;
            const double ut=v[nc+other]*tx+v[2*nc+other]*ty;tlo=fmin(tlo,ut);thi=fmax(thi,ut);
        }
        for(int a=0;a<2;++a) {
            bool wet=true;
            for(int k=0;k<5;++k) {const int s=wall_ids[(k*2+a)*nc+c];if(s<0)wet=false;else if(q[s]<=dry)wet=false;}
            if(!wet)continue;
            for(int side=0;side<2;++side) {
                const int slot=2*a+side;double stencil[4][5],tangential[5];
                for(int k=0;k<5;++k) {
                    const int kk=side==0?4-k:k, s=wall_ids[(kk*2+a)*nc+c],kind=wall_kind[(kk*2+a)*nc+c];
                    for(int m=0;m<4;++m)stencil[m][k]=v[m*nc+s];
                    if(kind>0) {
                        const double nx=stencil_normal[(kk*2+a)*nc+c],ny=stencil_normal[(10+kk*2+a)*nc+c];
                        const double un=stencil[1][k]*nx+stencil[2][k]*ny;
                        stencil[1][k]-=(double)kind*un*nx;stencil[2][k]-=(double)kind*un*ny;
                    }
                    if(triangle[c])tangential[k]=stencil[1][k]*tx+stencil[2][k]*ty;
                }
                for(int m=0;m<4;++m)values[m][slot]=recon_weno(stencil[m]);
                values[0][slot]=fmax(lo[0],fmin(hi[0],values[0][slot]));values[3][slot]=fmax(lo[3],fmin(hi[3],values[3][slot]));
                if(triangle[c]) {const double ut=fmax(tlo,fmin(thi,recon_weno(tangential)));values[1][slot]=ut*tx;values[2][slot]=ut*ty;}
                else for(int m=1;m<3;++m)values[m][slot]=fmax(lo[m],fmin(hi[m],values[m][slot]));
                ready[slot]=1;
            }
        }
        if(triangle[c]) {
            bool all=true;for(int s=0;s<4;++s)if(!ready[s] || values[0][s]-values[3][s]<=dry)all=false;
            if(!all)for(int s=0;s<4;++s)ready[s]=0;
            else {
                const double nx=wall_normal[c],ny=wall_normal[nc+c],normal_q=q[nc+c]*nx+q[2*nc+c]*ny;
                for(int a=0;a<2;++a)for(int side=0;side<2;++side) {
                    const int slot=2*a+side;const double h=values[0][slot]-values[3][slot];
                    double px=wall_box_center[c],py=wall_box_center[nc+c];
                    if(a==0)px=px+(side-0.5)*dx;else py=py+(side-0.5)*dx;
                    const double ratio=((px-wall_box_center[c])*nx+(py-wall_box_center[nc+c])*ny)/wall_mean_distance[c];
                    values[1][slot]+=normal_q*ratio/h*nx;values[2][slot]+=normal_q*ratio/h*ny;
                }
            }
        }
    }
    bool flatten=q[c]<=dry;
    if(!flatten)for(int k=offset[c];k<offset[c+1];++k) {
        const int f=faces[k],side=c==owner[f]?0:1,slot=wall_slot[side*nf+f];double eta,zb;
        if(slot>=0 && ready[slot]) {eta=values[0][slot];zb=values[3][slot];}
        else {
            const double rx=displacement[side*nf+f],ry=displacement[(2+side)*nf+f];
            eta=(centre[0]+gx[0]*rx)+gy[0]*ry;zb=(centre[3]+gx[3]*rx)+gy[3]*ry;
            if(weno[side*nf+f]>=0) {
                double se[5],sb[5];for(int j=0;j<5;++j) {const int s=weno[(j*2+side)*nf+f];se[j]=v[s];sb[j]=v[3*nc+s];
                    if(centre[0]-centre[3]>dry && se[j]-sb[j]<=dry && sb[j]>=centre[0])se[j]=centre[0];}
                eta=fmax(lo[0],fmin(hi[0],recon_weno(se)));zb=fmax(lo[3],fmin(hi[3],recon_weno(sb)));
            }
        }
        if(eta-zb<=dry || ((rectangle[c]<=0. || rectangle[nc+c]<=0.) && fabs(zb-centre[3])>=q[c])) {flatten=true;break;}
    }
    constant[c]=flatten?1:0;
    for(int m=0;m<4;++m) {grad[(m*2)*nc+c]=gx[m];grad[(m*2+1)*nc+c]=gy[m];vlo[m*nc+c]=lo[m];vhi[m*nc+c]=hi[m];
        for(int s=0;s<4;++s)wall_values[(m*4+s)*nc+c]=values[m][s];}
    for(int s=0;s<4;++s)wall_ready[s*nc+c]=ready[s];
}

extern "C" __global__ void reconstruction_faces(
    const double* q,const double* v,const double* grad,const double* vlo,const double* vhi,
    const double* wall_values,const int* wall_ready,const int* constant,
    const int* owner,const int* neighbour,const int* bid,const int* wall_slot,const int* weno,
    const double* displacement,const double* boundary_trace,int has_boundary_trace,
    long long nc,long long nf,double dry,double* face_value,double* trace,int* kind,int* status)
{
    const long long f=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(f>=nf)return;
    status[f]=0;for(int j=0;j<10;++j)trace[j*nf+f]=0.;for(int j=0;j<8;++j)face_value[j*nf+f]=0.;
    const int c=owner[f],n=neighbour[f];
    int type=n>=0?1:(bid[f]<0?2:3);
    if(bid[f]<0 && q[c]<=dry && (n<0 || q[n]<=dry))type=0;
    kind[f]=type;
    // Face-value diagnostics are evaluated even on skipped dry faces. Flux
    // trace outputs retain the original Euler early-exit zero convention.
    for(int side=0;side<2;++side) {
        const int cell=side==0?c:n;if(cell<0)continue;
        const int slot=wall_slot[side*nf+f];double value[4];
        if(slot>=0 && wall_ready[slot*nc+cell])for(int m=0;m<4;++m)value[m]=wall_values[(m*4+slot)*nc+cell];
        else {
            const double rx=displacement[side*nf+f],ry=displacement[(2+side)*nf+f];
            for(int m=0;m<4;++m)value[m]=(v[m*nc+cell]+grad[(m*2)*nc+cell]*rx)+grad[(m*2+1)*nc+cell]*ry;
            if(weno[side*nf+f]>=0)for(int m=0;m<4;++m) {
                double stencil[5];for(int k=0;k<5;++k) {const int s=weno[(k*2+side)*nf+f];stencil[k]=v[m*nc+s];
                    if(m==0 && v[cell]-v[3*nc+cell]>dry && v[s]-v[3*nc+s]<=dry && v[3*nc+s]>=v[cell])stencil[k]=v[cell];}
                value[m]=fmax(vlo[m*nc+cell],fmin(vhi[m*nc+cell],recon_weno(stencil)));
            }
        }
        if(constant[cell]) {for(int m=0;m<3;++m)value[m]=v[m*nc+cell];if(v[cell]-v[3*nc+cell]<=dry)value[3]=v[3*nc+cell];}
        for(int m=0;m<4;++m) {face_value[(m*2+side)*nf+f]=value[m];if(!isfinite(value[m]))status[f]|=1;}
        if(type==0)continue;
        double depth=fmax(value[0]-value[3],0.);trace[(8+side)*nf+f]=depth;
        if(q[cell]<=dry || (constant[cell] && depth<=dry))depth=0.;
        trace[(4*side)*nf+f]=depth;trace[(4*side+1)*nf+f]=depth*value[1];trace[(4*side+2)*nf+f]=depth*value[2];trace[(4*side+3)*nf+f]=value[3];
    }
    if(type==2) {for(int j=0;j<4;++j)trace[(4+j)*nf+f]=trace[j*nf+f];trace[9*nf+f]=trace[8*nf+f];}
    if(type==3) {
        // With no boundary state supplied this deliberately retains the raw
        // owner reconstruction. The caller must apply a boundary solver before
        // the face-flux kernel. Kind 3 is the explicit pending-boundary mask.
        if(has_boundary_trace)for(int j=0;j<3;++j) {trace[j*nf+f]=boundary_trace[j*nf+f];if(!isfinite(trace[j*nf+f]))status[f]|=4;}
    }
}
