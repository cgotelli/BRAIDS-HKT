// One explicit general-avalanche substep, matching BRAIDS's current
// bed_downslopes / bank depth intercept / wet-face mobility arithmetic.
// The output is an un-limited signed (grain,face) total mass proposal;
// ResidentSurfaceTransfer owns donor limiting and conservative gathering.
#define AV_EPS 2.2204460492503131e-16

__device__ inline double av_positive_triangle(double x,double y,double z) {
    double lo=fmin(x,fmin(y,z)),hi=fmax(x,fmax(y,z)),md=x+y+z-lo-hi;
    if(lo>=0.)return (x+y+z)/3.;
    if(hi<=0.)return 0.;
    if(md<=0.)return hi*hi*hi/(3.*(hi-lo)*(hi-md));
    return (x+y+z)/3.+(-lo)*(-lo)*(-lo)/(3.*(md-lo)*(hi-lo));
}

__device__ inline double av_mobility(double slope,double dry_slope,double wet_slope,
                                      double transition,double dry,double a0,double b0,
                                      double a1,double b1) {
    if(slope<=0.)return 0.;
    if(dry_slope==wet_slope)return fmax(1.-dry_slope/slope,0.);
    double breaks[12],targets[3];int count=2,ntarget;
    breaks[0]=0.;breaks[1]=1.;
    if(transition>0.){
        targets[0]=0.;targets[1]=transition;
        targets[2]=transition*(slope-dry_slope)/(wet_slope-dry_slope);ntarget=3;
    }else{targets[0]=dry;ntarget=1;}
    double a[2]={a0,a1},b[2]={b0,b1};
    for(int i=0;i<2;++i){
        if(b[i]==0.)continue;
        for(int j=0;j<ntarget;++j){double root=(targets[j]-a[i])/b[i];
            if(root>0.&&root<1.)breaks[count++]=root;}
    }
    if(b[0]!=b[1]){double root=(a[1]-a[0])/(b[0]-b[1]);
        if(root>0.&&root<1.)breaks[count++]=root;}
    for(int i=1;i<count;++i){double tmp=breaks[i];int j=i-1;
        while(j>=0&&breaks[j]>tmp){breaks[j+1]=breaks[j];--j;}
        breaks[j+1]=tmp;}
    double alpha=0.;
    for(int k=0;k<count-1;++k){
        double t=.5*(breaks[k]+breaks[k+1]);
        double h=fmax(0.,fmax(a0+b0*t,a1+b1*t));
        double weight=transition>0.?fmin(1.,h/transition):(h>dry?1.:0.);
        double critical=dry_slope+weight*(wet_slope-dry_slope);
        alpha+=(breaks[k+1]-breaks[k])*fmax(1.-critical/slope,0.);
    }
    return alpha;
}

extern "C" __global__ void av_downslopes(int nc,int nf,int ni,const double* bed,
 const int* owner,const int* neighbour,const int* offset,const int* faces,
 const int* signs,const double* delta,const double* weight,const double* matrix,
 double* down,int* status) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
    double z=bed[c],b1=0.,b2=0.;int bad=!isfinite(z);
    for(int k=offset[c];k<offset[c+1];++k){int f=faces[k];
        int n=signs[k]>0?neighbour[f]:owner[f];if(n<0)continue;
        double w=weight[k],x=delta[k],y=delta[ni+k];
        b1+=w*x*(bed[n]-z);b2+=w*y*(bed[n]-z);
    }
    double a11=matrix[c],a12=matrix[nc+c],a22=matrix[2*nc+c],det=matrix[3*nc+c];
    double sx=0.,sy=0.;
    if(det>1.e-12*(a11+a22)*(a11+a22)){
        sx=-(a22*b1-a12*b2)/det;sy=-(a11*b2-a12*b1)/det;}
    bad|=!isfinite(sx)||!isfinite(sy);
    down[c]=bad?0.:sx;down[nc+c]=bad?0.:sy;status[c]=bad;
}

extern "C" __global__ void av_depth_intercept(int nc,int nf,const double* depth,
 const double* down,const double* xy,const double* mid,const double* normal,
 const double* length,const double* area,const int* offset,const int* faces,
 const int* signs,double* intercept,int* status) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
    double h=depth[c],sx=down[c],sy=down[nc+c],cx=xy[c],cy=xy[nc+c];
    int bad=!isfinite(h)||h<0.||!isfinite(sx)||!isfinite(sy);
    double minimum=0.,maximum=0.;
    for(int k=offset[c];k<offset[c+1];++k){int f=faces[k];
        double rx=mid[f]-cx,ry=mid[nf+f]-cy,nx=normal[f],ny=normal[nf+f];
        double tx=-ny,ty=nx,half=.5*length[f];
        double e0=sx*(rx-half*tx)+sy*(ry-half*ty);
        double e1=sx*(rx+half*tx)+sy*(ry+half*ty);
        minimum=fmin(minimum,fmin(e0,e1));maximum=fmax(maximum,fmax(e0,e1));
    }
    double a;
    if(h<=0.)a=-maximum;
    else if(h+minimum>=0.)a=h;
    else{
        double lo=-maximum,hi=h-minimum;
        for(int iteration=0;iteration<48;++iteration){
            a=.5*(lo+hi);double mean=0.;
            for(int k=offset[c];k<offset[c+1];++k){int f=faces[k];
                double rx=mid[f]-cx,ry=mid[nf+f]-cy,nx=normal[f],ny=normal[nf+f];
                double tx=-ny,ty=nx,half=.5*length[f];
                double e0=sx*(rx-half*tx)+sy*(ry-half*ty);
                double e1=sx*(rx+half*tx)+sy*(ry+half*ty);
                double w=.5*length[f]*signs[k]*(nx*rx+ny*ry)/area[c];
                mean+=w*av_positive_triangle(a,a+e0,a+e1);
            }
            if(mean>h)hi=a;else lo=a;
        }
        a=.5*(lo+hi);
    }
    bad|=!isfinite(a);intercept[c]=bad?0.:a;status[c]=bad;
}

extern "C" __global__ void av_face_rate(int nc,int nf,const double* bed,
 const double* down,const double* intercept,const double* xy,const double* mid,
 const double* normal,const double* length,const int* owner,const int* neighbour,
 const int* front_active,double diffusivity,double wet_slope,double dry_slope,
 double transition,double dry,double* rate,double* conductance,int* status) {
    int f=blockIdx.x*blockDim.x+threadIdx.x;if(f>=nf)return;
    int c=owner[f],n=neighbour[f];rate[f]=0.;conductance[f]=0.;status[f]=0;
    if(n<0||front_active[c]||front_active[n])return;
    double nx=normal[f],ny=normal[nf+f],tx=-ny,ty=nx;
    double dx=xy[n]-xy[c],dy=xy[nc+n]-xy[nc+c];
    double distance=fabs(dx*nx+dy*ny);
    if(distance<=AV_EPS)distance=sqrt(dx*dx+dy*dy);
    if(distance<=AV_EPS)return;
    double tangent_slope=.5*((down[c]+down[n])*tx+(down[nc+c]+down[nc+n])*ty);
    double normal_slope=(bed[c]-bed[n])/distance-tangent_slope*(dx*tx+dy*ty)/distance;
    if(normal_slope==0.)return;
    double sx=normal_slope*nx+tangent_slope*tx,sy=normal_slope*ny+tangent_slope*ty;
    double magnitude=sqrt(sx*sx+sy*sy);
    double rx=mid[f]-xy[c],ry=mid[nf+f]-xy[nc+c],half=.5*length[f];
    double a0=intercept[c]+down[c]*(rx-half*tx)+down[nc+c]*(ry-half*ty);
    double b0=2.*half*(down[c]*tx+down[nc+c]*ty);
    rx=mid[f]-xy[n];ry=mid[nf+f]-xy[nc+n];
    double a1=intercept[n]+down[n]*(rx-half*tx)+down[nc+n]*(ry-half*ty);
    double b1=2.*half*(down[n]*tx+down[nc+n]*ty);
    double alpha=av_mobility(magnitude,dry_slope,wet_slope,transition,dry,a0,b0,a1,b1);
    double coeff=diffusivity*length[f]*alpha/distance;
    double flux=diffusivity*length[f]*alpha*normal_slope;
    if(!isfinite(coeff)||!isfinite(flux)||!isfinite(alpha)||alpha<0.){status[f]=1;return;}
    if(coeff>0.){rate[f]=flux;conductance[f]=coeff;}
}

extern "C" __global__ void av_cell_rate(int nc,const double* conductance,
 const double* area,const int* offset,const int* faces,double* weight,int* status) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
    double sum=0.;for(int k=offset[c];k<offset[c+1];++k)sum+=conductance[faces[k]];
    double value=sum/area[c];status[c]=!isfinite(value)||value<0.;
    weight[c]=status[c]?0.:value;
}

extern "C" __global__ void av_subdt(const double* maximum,double remaining,
                                       double* subdt,int* status) {
    if(blockIdx.x||threadIdx.x)return;
    double value=maximum[0]>0.?fmin(remaining,.45/maximum[0]):0.;
    status[0]=!isfinite(maximum[0])||maximum[0]<0.||!isfinite(value)||value<0.;
    subdt[0]=status[0]?0.:value;
}

extern "C" __global__ void av_propose(int nc,int nf,int ng,const double* active,
 const double* bulk,const double* rate,const int* owner,const int* neighbour,
 const double* subdt,double* proposed,int* status) {
    int f=blockIdx.x*blockDim.x+threadIdx.x;if(f>=nf)return;
    double flow=rate[f];int c=owner[f],n=neighbour[f];status[f]=0;
    if(n<0||flow==0.){for(int g=0;g<ng;++g)proposed[g*nf+f]=0.;return;}
    int donor=flow>0.?c:n;double volume=0.;int bad=0;
    for(int g=0;g<ng;++g){double amount=active[g*nc+donor];
        if(!isfinite(amount)||amount<0.)bad=1;
        volume+=amount/bulk[g];}
    double moved=subdt[0]*fabs(flow);
    if(!isfinite(volume)||!isfinite(moved)||moved<0.)bad=1;
    for(int g=0;g<ng;++g){double value=0.;
        if(!bad&&volume>0.){
            double share=(active[g*nc+donor]/bulk[g])/volume;
            value=(flow>0.?1.:-1.)*moved*share*bulk[g];}
        proposed[g*nf+f]=value;
    }
    status[f]=bad;
}
