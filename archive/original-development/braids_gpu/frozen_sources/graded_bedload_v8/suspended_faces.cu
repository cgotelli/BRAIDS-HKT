// Scheme-6 suspended concentration face values, isolated from hydraulic flux.
// SoA C layout, 0-based cells/faces/boundaries, -1 denotes missing index.
#define SC_EPS 2.2204460492503131e-16

__device__ __forceinline__ double sc_weno(const double* v) {
    const double p0=(2.*v[0]-7.*v[1]+11.*v[2])/6.;
    const double p1=(-v[1]+5.*v[2]+2.*v[3])/6.;
    const double p2=(2.*v[2]+5.*v[3]-v[4])/6.;
    const double a0=v[0]-2.*v[1]+v[2],c0=v[0]-4.*v[1]+3.*v[2];
    const double a1=v[1]-2.*v[2]+v[3],c1=v[1]-v[3];
    const double a2=v[2]-2.*v[3]+v[4],c2=3.*v[2]-4.*v[3]+v[4];
    const double b0=(13./12.)*a0*a0+.25*c0*c0;
    const double b1=(13./12.)*a1*a1+.25*c1*c1;
    const double b2=(13./12.)*a2*a2+.25*c2*c2;
    double scale=v[0]*v[0];for(int k=1;k<5;++k)scale=fmax(scale,v[k]*v[k]);
    const double tau=fabs(b0-b2),eps=1.e-40+1.e-14*scale;
    const double r0=tau/(b0+eps),r1=tau/(b1+eps),r2=tau/(b2+eps);
    const double w0=.1*(1.+r0*r0),w1=.6*(1.+r1*r1),w2=.3*(1.+r2*r2);
    return ((w0*p0+w1*p1)+w2*p2)/((w0+w1)+w2);
}

extern "C" __global__ void sc_primitive(int nc,int ng,double dry,
 const double* depth,const double* suspended,double* primitive,int* status) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
    double h=depth[c];int bad=!isfinite(h)||h<0.;
    for(int g=0;g<ng;++g){double mass=suspended[g*nc+c];
        if(!isfinite(mass)||mass<0.)bad=1;
        primitive[g*nc+c]=h>dry?mass/h:0.;}
    status[c]=bad;
}

extern "C" __global__ void sc_gradients(int nc,int nf,int nadj,int ng,int nb,
 double dry,double grav,int outlet_extension,const double* v,const double* hydro,
 const double* boundary,const double* xy,const double* mid,const double* normal,
 const int* owner,const int* neighbour,const int* bid,const int* mode,
 const int* offset,const int* faces,const int* signs,const int* active,
 const double* matrix,const double* weight,const double* delta,
 double* grad,double* vlo,double* vhi,int* status) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;status[c]=0;
    const double a11=matrix[c],a12=matrix[nc+c],a22=matrix[2*nc+c],det=matrix[3*nc+c];
    double bed=hydro[3*nc+c],h=hydro[c]-bed;
    // Native limiter skips an outlet face only if the *unlimited* bed
    // gradient has a resolved slope there. Compute it from the same LS sums.
    double bbx=0.,bby=0.,bedgx=0.,bedgy=0.;
    if(active[c]&&det>1.e-12*(a11+a22)*(a11+a22)){
        for(int k=offset[c];k<offset[c+1];++k){int f=faces[k];
            int n=signs[k]<0?owner[f]:neighbour[f];if(n<0)continue;
            const double r0=delta[k],r1=delta[nadj+k],w=weight[k],d=hydro[3*nc+n]-bed;
            bbx+=w*r0*d;bby+=w*r1*d;}
        bedgx=(a22*bbx-a12*bby)/det;bedgy=(a11*bby-a12*bbx)/det;
    }
    for(int g=0;g<ng;++g){const int i=g*nc+c;double centre=v[i],lo=centre,hi=centre;
        double bx=0.,by=0.,gx=0.,gy=0.;
        if(active[c]){
            for(int k=offset[c];k<offset[c+1];++k){int f=faces[k];
                int n=signs[k]<0?owner[f]:neighbour[f];
                if(n<0){int b=bid[f];if(b>=0&&b<nb){double bc=boundary[g*nb+b];
                        lo=fmin(lo,bc);hi=fmax(hi,bc);}continue;}
                const double adjacent=v[g*nc+n],r0=delta[k],r1=delta[nadj+k],w=weight[k];
                bx+=w*r0*(adjacent-centre);by+=w*r1*(adjacent-centre);
                lo=fmin(lo,adjacent);hi=fmax(hi,adjacent);
            }
            if(det>1.e-12*(a11+a22)*(a11+a22)){
                gx=(a22*bx-a12*by)/det;gy=(a11*by-a12*bx)/det;
                if(outlet_extension&&h>dry){
                    for(int k=offset[c];k<offset[c+1];++k){int f=faces[k];
                        if(neighbour[f]>=0||bid[f]<0||mode[f]!=2)continue;
                        double un=hydro[nc+c]*normal[f]+hydro[2*nc+c]*normal[nf+f];
                        if(fabs(un)>=sqrt(grav*h))continue;
                        double r0=mid[f]-xy[c],r1=mid[nf+f]-xy[nc+c];
                        double db=bedgx*r0+bedgy*r1;
                        if(fabs(db)<=64.*SC_EPS*fmax(1.,fabs(bed)))continue;
                        double exterior=centre+2.*(gx*r0+gy*r1);
                        lo=fmin(lo,exterior);hi=fmax(hi,exterior);
                    }
                }
                double theta=1.;
                for(int k=offset[c];k<offset[c+1];++k){int f=faces[k];
                    double r0=mid[f]-xy[c],r1=mid[nf+f]-xy[nc+c];
                    if(neighbour[f]<0&&bid[f]>=0&&mode[f]==2&&h>dry){
                        double un=hydro[nc+c]*normal[f]+hydro[2*nc+c]*normal[nf+f];
                        double db=bedgx*r0+bedgy*r1;
                        if(fabs(un)<sqrt(grav*h)&&
                           fabs(db)>64.*SC_EPS*fmax(1.,fabs(bed)))continue;}
                    double d=gx*r0+gy*r1;
                    if(d>0.)theta=fmin(theta,(hi-centre)/d);
                    if(d<0.)theta=fmin(theta,(lo-centre)/d);
                }
                gx*=fmax(0.,theta);gy*=fmax(0.,theta);
            }
        }
        vlo[i]=lo;vhi[i]=hi;grad[(g*2)*nc+c]=gx;grad[(g*2+1)*nc+c]=gy;
        if(!isfinite(lo)||!isfinite(hi)||!isfinite(gx)||!isfinite(gy))status[c]=1;
    }
}

extern "C" __global__ void sc_wall_values(int nc,int ng,const double* v,
 const int* wall_ids,const int* wall_ready,double* wall_values) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
    for(int g=0;g<ng;++g)for(int slot=0;slot<4;++slot){int i=(g*4+slot)*nc+c;
        wall_values[i]=0.;if(!wall_ready[slot*nc+c])continue;
        int a=slot/2,side=slot%2;double stencil[5];bool valid=true;
        for(int k=0;k<5;++k){int kk=side==0?4-k:k;
            int s=wall_ids[(kk*2+a)*nc+c];
            if(s<0||s>=nc){valid=false;stencil[k]=0.;}
            else stencil[k]=v[g*nc+s];}
        wall_values[i]=valid?sc_weno(stencil):nan("");
    }
}

__device__ __forceinline__ double sc_face_value(int c,int f,int side,int g,
 int nc,int nf,const double* v,const double* grad,const double* wall_values,
 const int* wall_ready,const int* wall_slot,const int* weno,
 const double* displacement) {
    int slot=wall_slot[side*nf+f];
    if(slot>=0&&wall_ready[slot*nc+c])return wall_values[(g*4+slot)*nc+c];
    double value=v[g*nc+c];
    double r0=displacement[side*nf+f],r1=displacement[(2+side)*nf+f];
    value+=grad[(g*2)*nc+c]*r0+grad[(g*2+1)*nc+c]*r1;
    if(weno[(side)*nf+f]>=0){double stencil[5];
        for(int k=0;k<5;++k){int s=weno[(k*2+side)*nf+f];stencil[k]=v[g*nc+s];}
        value=sc_weno(stencil);
    }
    return value;
}

extern "C" __global__ void sc_faces(int nc,int nf,int ng,int nb,double dry,
 const double* depth,const double* v,const double* grad,
 const double* wall_values,const int* wall_ready,const double* boundary,
 const int* owner,const int* neighbour,const int* bid,const int* wall_slot,
 const int* weno,const double* displacement,double* left,double* right,int* status) {
    int f=blockIdx.x*blockDim.x+threadIdx.x;if(f>=nf)return;
    int c=owner[f],n=neighbour[f],b=bid[f],bad=0;
    bool zero=b<0&&depth[c]<=dry&&(n<0||depth[n]<=dry);
    for(int g=0;g<ng;++g){double cl=0.,cr=0.;
        if(!zero){
            if(n>=0){cl=sc_face_value(c,f,0,g,nc,nf,v,grad,wall_values,wall_ready,wall_slot,weno,displacement);
                cr=sc_face_value(n,f,1,g,nc,nf,v,grad,wall_values,wall_ready,wall_slot,weno,displacement);}
            else if(b<0){cl=sc_face_value(c,f,0,g,nc,nf,v,grad,wall_values,wall_ready,wall_slot,weno,displacement);cr=cl;}
            else {cl=v[g*nc+c];cr=boundary[g*nb+b];}
        }
        left[g*nf+f]=cl;right[g*nf+f]=cr;
        if(!isfinite(cl)||!isfinite(cr)||cr<0.&&b>=0)bad=1;
    }
    status[f]=bad;
}

extern "C" __global__ void sc_status_merge(int nc,int nf,
 const int* primitive_status,const int* gradient_status,
 const int* face_status,int* status) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    int bad=(i<nc&&(primitive_status[i]||gradient_status[i]))||
            (i<nf&&face_status[i]);
    if(bad)atomicExch(status,1);
}
