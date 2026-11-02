// Resident fixed-grid FLOW 2DH suspended-boundary and horizontal-diffusion
// operators.  All arrays are FP64 SoA; face water flux is positive from the
// owner towards the neighbour/exterior.  Boundary ledgers are positive inward.

#define M2B_PI 3.141592653589793238462643383279502884
#define M2B_SCHMIDT 0.7

extern "C" __global__ void m2b_resolve_boundary(
 int nc,int nf,int ng,int nb,double dry,double dt,
 const double* depth,const double* suspended,const double* prescribed,
 const double* water,const double* length,const int* owner,const int* neighbour,
 const int* bid,const int* neumann,const double* return_time,
 const double* accepted_last,const double* accepted_clock,
 double* resolved,double* advective_mass_in,double* trial_last,
 double* trial_clock,int* status)
{
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=ng*nf)return;
 int g=i/nf,f=i%nf,c=owner[f],n=neighbour[f],b=bid[f];
 double last=accepted_last[i],clock=accepted_clock[i];
 trial_last[i]=last;trial_clock[i]=clock;resolved[i]=0.;advective_mass_in[i]=0.;status[i]=0;
 if(n>=0||b<0)return;
 double h=depth[c],m=suspended[g*nc+c];
 double p=prescribed[g*nb+b];
 double t=return_time[i],qin=-water[f]*length[f];
 if(!isfinite(h)||h<0.||!isfinite(m)||m<0.||!isfinite(p)||p<0.||
    !isfinite(t)||!isfinite(qin)||!isfinite(last)||last<0.||
    !isfinite(clock)||clock<0.){status[i]=1;return;}
 double ci=h>dry?m/h:0.,bc=p;
 if(t<0.){bc=ci;trial_clock[i]=0.;}
 else if(t>0.){
   double tol=256.*2.2204460492503131e-16*fmax(1.,t);
   if(clock>t+tol){status[i]=1;return;}
   if(qin<0.){trial_last[i]=ci;trial_clock[i]=t;bc=ci;}
   else if(qin>0.){
     double next=fmax(clock-dt,0.);trial_clock[i]=next;
     double weight=.5*(cos(M2B_PI*next/t)+1.);
     bc=last+weight*(p-last);
   } else trial_clock[i]=0.;
 }
 if(neumann[g])bc=fmax(0.,ci);
 double ca=qin>0.?bc:ci;
 resolved[i]=bc;advective_mass_in[i]=dt*qin*ca;
 if(!isfinite(ci)||ci<0.||!isfinite(bc)||bc<0.||
    !isfinite(advective_mass_in[i]))status[i]=1;
}

extern "C" __global__ void m2b_expected_advective_ledger(
 int nf,int ng,int nb,const int* boundary_offset,const int* boundary_faces,
 const double* face_mass_in,const int* face_status,double* ledger,int* status)
{
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=ng*nb)return;
 int g=i/nb,b=i%nb;double value=0.;status[i]=0;
 for(int k=boundary_offset[b];k<boundary_offset[b+1];++k){int f=boundary_faces[k];
   if(face_status[g*nf+f])status[i]=1;value+=face_mass_in[g*nf+f];}
 ledger[i]=value;if(!isfinite(value))status[i]=1;
}

// Same limited least-squares scalar reconstruction as the generic scheme-6
// path, but every boundary face owns its resolved concentration.  This is
// required when faces under one boundary id have different flow directions.
extern "C" __global__ void m2b_sc_gradients(
 int nc,int nf,int nadj,int ng,double dry,double grav,int outlet_extension,
 const double* v,const double* hydro,const double* face_boundary,
 const double* xy,const double* mid,const double* normal,const int* owner,
 const int* neighbour,const int* bid,const int* mode,const int* offset,
 const int* faces,const int* signs,const int* active,const double* matrix,
 const double* weight,const double* delta,double* grad,double* vlo,double* vhi,
 int* status)
{
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;status[c]=0;
 const double a11=matrix[c],a12=matrix[nc+c],a22=matrix[2*nc+c],det=matrix[3*nc+c];
 double bed=hydro[3*nc+c],h=hydro[c]-bed,bbx=0.,bby=0.,bedgx=0.,bedgy=0.;
 if(active[c]&&det>1.e-12*(a11+a22)*(a11+a22)){
   for(int k=offset[c];k<offset[c+1];++k){int f=faces[k];
     int n=signs[k]<0?owner[f]:neighbour[f];if(n<0)continue;
     double r0=delta[k],r1=delta[nadj+k],w=weight[k],d=hydro[3*nc+n]-bed;
     bbx+=w*r0*d;bby+=w*r1*d;}
   bedgx=(a22*bbx-a12*bby)/det;bedgy=(a11*bby-a12*bbx)/det;
 }
 for(int g=0;g<ng;++g){int i=g*nc+c;double centre=v[i],lo=centre,hi=centre;
   double bx=0.,by=0.,gx=0.,gy=0.;
   if(active[c]){
     for(int k=offset[c];k<offset[c+1];++k){int f=faces[k];
       int n=signs[k]<0?owner[f]:neighbour[f];
       // Match native v48: open-boundary values do not participate in the
       // reconstruction extrema.  The resolved per-face value enters the
       // boundary flux and the subsequent FCT bounds instead.
       if(n<0)continue;
       double adjacent=v[g*nc+n],r0=delta[k],r1=delta[nadj+k],w=weight[k];
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
           double r0=mid[f]-xy[c],r1=mid[nf+f]-xy[nc+c],db=bedgx*r0+bedgy*r1;
           if(fabs(db)<=64.*2.2204460492503131e-16*fmax(1.,fabs(bed)))continue;
           double exterior=centre+2.*(gx*r0+gy*r1);lo=fmin(lo,exterior);hi=fmax(hi,exterior);
         }
       }
       double theta=1.;
       for(int k=offset[c];k<offset[c+1];++k){int f=faces[k];
         double r0=mid[f]-xy[c],r1=mid[nf+f]-xy[nc+c];
         if(neighbour[f]<0&&bid[f]>=0&&mode[f]==2&&h>dry){
           double un=hydro[nc+c]*normal[f]+hydro[2*nc+c]*normal[nf+f];
           double db=bedgx*r0+bedgy*r1;
           if(fabs(un)<sqrt(grav*h)&&fabs(db)>64.*2.2204460492503131e-16*fmax(1.,fabs(bed)))continue;
         }
         double d=gx*r0+gy*r1;if(d>0.)theta=fmin(theta,(hi-centre)/d);
         if(d<0.)theta=fmin(theta,(lo-centre)/d);
       }
       gx*=fmax(0.,theta);gy*=fmax(0.,theta);
     }
   }
   vlo[i]=lo;vhi[i]=hi;grad[(g*2)*nc+c]=gx;grad[(g*2+1)*nc+c]=gy;
   if(!isfinite(lo)||!isfinite(hi)||!isfinite(gx)||!isfinite(gy))status[c]=1;
 }
}

extern "C" __global__ void m2b_patch_boundary_faces(
 int nf,int ng,const int* neighbour,const int* bid,const double* face_boundary,
 double* right,int* status)
{
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=ng*nf)return;
 int f=i%nf;if(neighbour[f]>=0||bid[f]<0)return;
 double value=face_boundary[i];right[i]=value;
 if(!isfinite(value)||value<0.)status[f]=1;
}

extern "C" __global__ void m2b_sa_face_low(
 int nc,int nf,int ng,const double* water,const double* concentration,
 const double* left,const double* right,const double* face_boundary,
 const int* owner,const int* neighbour,const int* bid,const int* conc_status,
 double* low_flux,double* anti,int* status)
{
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=ng*nf)return;
 int g=i/nf,f=i%nf,c=owner[f],n=neighbour[f],b=bid[f];double w=water[f];
 status[i]=0;low_flux[i]=0.;anti[i]=0.;
 if(!isfinite(w)||conc_status[g*nc+c]||(n>=0&&conc_status[g*nc+n])){status[i]=1;return;}
 if(n<0&&b<0){if(w!=0.)status[i]=1;return;}
 if(n<0){if(w<0.){double value=face_boundary[i];if(!isfinite(value)||value<0.){status[i]=1;return;}low_flux[i]=w*value;}
   else low_flux[i]=w*concentration[g*nc+c];}
 else if(w>=0.){double value=left[i];if(!isfinite(value)){status[i]=1;return;}
   low_flux[i]=w*concentration[g*nc+c];anti[i]=w*value-low_flux[i];}
 else {double value=right[i];if(!isfinite(value)){status[i]=1;return;}
   low_flux[i]=w*concentration[g*nc+n];anti[i]=w*value-low_flux[i];}
 if(!isfinite(low_flux[i])||!isfinite(anti[i]))status[i]=1;
}

// Cell bound calculation copied from the generic FCT route with only the
// boundary lookup changed from per-id to per-face.
__device__ int m2b_extremum(const double* concentration,int nc,int g,const int ids[5],double& bound){
 double values[5],z[5],scale=0.;for(int k=0;k<5;++k){values[k]=concentration[g*nc+ids[k]];scale=fmax(scale,fabs(values[k]));}
 bound=values[2];if(scale==0.)return 1;double lo=1.e300,hi=-1.e300,tol=4096.*2.2204460492503131e-16;
 for(int k=0;k<5;++k){z[k]=values[k]/scale;lo=fmin(lo,z[k]);hi=fmax(hi,z[k]);}if(hi-lo<=tol)return 1;
 if((z[2]-z[1])*(z[3]-z[2])>0.)return 0;double d20=z[0]-2.*z[1]+z[2],d21=z[1]-2.*z[2]+z[3],d22=z[2]-2.*z[3]+z[4];
 double amin=fmin(fabs(d20),fmin(fabs(d21),fabs(d22))),amax=fmax(fabs(d20),fmax(fabs(d21),fabs(d22)));
 if(amin<=tol||!((d20>0.&&d21>0.&&d22>0.)||(d20<0.&&d21<0.&&d22<0.))||amin<.5*amax)return 0;
 double slope=.5*(z[3]-z[1]),vertex=fmax(-.5,fmin(.5,-slope/d21));
 double point=z[2]+slope*vertex+.5*d21*vertex*vertex-d21/24.;
 if(d21<0.){bound=scale*(z[2]+2.*fmax(point-z[2],0.));return 2;}
 bound=fmax(0.,scale*(z[2]-2.*fmax(z[2]-point,0.)));return 3;
}

__device__ void m2b_smooth_bound(int c,int g,int nc,int nf,const double* concentration,
 const double* normal,const double* rectangle,const int* hydro_constant,const int* weno,
 const int* owner,const int* neighbour,const int* bid,const int* offset,const int* faces,
 double& lower,double& upper){
 if(hydro_constant[c]||rectangle[c]<=0.||rectangle[nc+c]<=0.)return;
 double lo=lower,hi=upper,scale=fmax(fmax(fabs(lower),fabs(upper)),2.2250738585072014e-308);
 double tol=4096.*2.2204460492503131e-16*scale;int extkind=1;
 for(int k=offset[c];k<offset[c+1];++k)if(bid[faces[k]]>=0)return;
 for(int axis=0;axis<2;++axis){int ids[5]={-1,-1,-1,-1,-1};
   for(int k=offset[c];k<offset[c+1];++k){int f=faces[k];if(fabs(normal[axis*nf+f])!=1.)continue;
     int side=c==owner[f]?0:1;if(weno[side*nf+f]>=0)for(int j=0;j<5;++j)ids[j]=weno[(j*2+side)*nf+f];}
   if(ids[0]<0){for(int k=offset[c];k<offset[c+1];++k){int f=faces[k],n=c==owner[f]?neighbour[f]:owner[f];
       if(fabs(normal[axis*nf+f])!=1.||n<0)continue;
       if(hydro_constant[n]||fabs(concentration[g*nc+n]-concentration[g*nc+c])>tol)return;}continue;}
   for(int j=0;j<5;++j)if(ids[j]<0||ids[j]>=nc||hydro_constant[ids[j]])return;
   double candidate;int kind=m2b_extremum(concentration,nc,g,ids,candidate);if(kind==0)return;if(kind==1)continue;
   if(extkind!=1&&kind!=extkind)return;extkind=kind;if(kind==2)hi=fmax(hi,candidate);if(kind==3)lo=fmin(lo,candidate);
 }
 lower=fmax(0.,lo);upper=hi;
}

extern "C" __global__ void m2b_sa_cell_bounds(
 int nc,int nf,int ng,const double* mass,const double* next_depth,
 const double* concentration,const double* face_boundary,const double* water,
 const double* low_flux,const double* anti,const double* area,const double* length,
 const double* normal,const double* rectangle,const int* hydro_constant,
 const int* weno,const int* owner,const int* neighbour,const int* bid,
 const int* offset,const int* faces,const int* signs,const int* conc_status,
 const int* face_status,double dt,double* low_state,double* rplus,
 double* rminus,int* status)
{
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=ng*nc)return;
 int g=i/nc,c=i%nc;status[i]=0;low_state[i]=mass[i];rplus[i]=1.;rminus[i]=1.;
 double nh=next_depth[c];if(conc_status[i]||!isfinite(nh)||nh<0.){status[i]=1;return;}
 double pp=0.,pm=0.,low=mass[i],lower=concentration[i],upper=lower;
 for(int k=offset[c];k<offset[c+1];++k){int f=faces[k],n=signs[k]>0?neighbour[f]:owner[f],b=bid[f];
   if(face_status[g*nf+f]){status[i]=1;return;}
   if(n>=0){lower=fmin(lower,concentration[g*nc+n]);upper=fmax(upper,concentration[g*nc+n]);}
   else if(b>=0&&water[f]<0.){double value=face_boundary[g*nf+f];lower=fmin(lower,value);upper=fmax(upper,value);}
   double z=-dt*signs[k]*length[f]/area[c],a=anti[g*nf+f];low+=z*low_flux[g*nf+f];pp+=fmax(z*a,0.);pm+=fmin(z*a,0.);
 }
 if(!isfinite(low)||low<0.){status[i]=1;return;}
 m2b_smooth_bound(c,g,nc,nf,concentration,normal,rectangle,hydro_constant,weno,owner,neighbour,bid,offset,faces,lower,upper);
 low_state[i]=low;if(pp>0.)rplus[i]=fmin(1.,fmax(0.,upper*nh-low-64.*2.2250738585072014e-308)/pp);
 if(pm<0.)rminus[i]=fmin(1.,fmax(0.,low-lower*nh-64.*2.2250738585072014e-308)/(-pm));
 if(pp>0.)rplus[i]*=(1.-32.*2.2204460492503131e-16);if(pm<0.)rminus[i]*=(1.-32.*2.2204460492503131e-16);
 if(!isfinite(rplus[i])||!isfinite(rminus[i]))status[i]=1;
}

extern "C" __global__ void m2b_face_area_from_trace(
 int nf,const int* neighbour,const int* bid,const double* trace,
 const double* width,double* exchange_area,int* active,int* status)
{
 int f=blockIdx.x*blockDim.x+threadIdx.x;if(f>=nf)return;
 int n=neighbour[f];active[f]=0;status[f]=0;exchange_area[f]=0.;if(n<0&&bid[f]<0)return;
 double hl=trace[f],hr=n>=0?trace[4*nf+f]:hl;
 double hface=n>=0?.5*(hl+hr):hl;
 if(isfinite(hface)&&isfinite(width[f])&&hface>=0.&&width[f]>=0.){
   exchange_area[f]=hface*width[f];active[f]=1;
 } else status[f]=1;
}

extern "C" __global__ void m2b_diffusion_faces(
 int nc,int nf,int ng,double dry,double dt,const double* depth,
 const double* mass,const double* face_boundary,const double* dicuv,
 const double* exchange_area,const double* centre_distance,const int* face_active,
 const int* owner,const int* neighbour,const int* bid,double* effective,
 double* face_mass_owner_out,int* active_out,int* status)
{
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=ng*nf)return;
 int g=i/nf,f=i%nf,c=owner[f],n=neighbour[f],b=bid[f];status[i]=0;
 effective[i]=0.;face_mass_owner_out[i]=0.;active_out[i]=0;
 int fa=face_active[f];if(fa!=0&&fa!=1){status[i]=1;return;}
 if((n<0&&b<0)||!fa)return;
 double hl=depth[c],hr=n>=0?depth[n]:hl,dl=dicuv[c],dr=n>=0?dicuv[n]:dl;
 double area=exchange_area[f],dist=centre_distance[f];
 if(!isfinite(hl)||!isfinite(hr)||hl<0.||hr<0.||!isfinite(dl)||!isfinite(dr)||dl<0.||dr<0.||
    !isfinite(area)||area<0.||!isfinite(dist)||dist<=0.){status[i]=1;return;}
 double deff=.5*(dl+dr)/M2B_SCHMIDT;effective[i]=deff;
 if(fmin(hl,hr)<=dry)return;
 double cl=mass[g*nc+c]/hl,cr=n>=0?mass[g*nc+n]/hr:face_boundary[i];
 if(!isfinite(cl)||cl<0.||!isfinite(cr)||cr<0.){status[i]=1;return;}
 face_mass_owner_out[i]=dt*area*deff*(cl-cr)/dist;active_out[i]=1;
 if(!isfinite(face_mass_owner_out[i]))status[i]=1;
}

// Deterministic native-v48 donor bound.  face_mass already contains dt and
// is a signed whole-face amount, positive owner to neighbour/exterior.
extern "C" __global__ void m2b_diffusion_theta(
 int nc,int nf,int ng,const double* mass,const double* face_mass,
 const double* area,const int* offset,const int* faces,const int* signs,
 const int* face_status,double* theta,int* status)
{
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=ng*nc)return;
 int g=i/nc,c=i%nc;double outgoing=0.;theta[i]=1.;status[i]=0;
 double inventory=mass[i]*area[c];
 if(!isfinite(mass[i])||mass[i]<0.||!isfinite(area[c])||area[c]<=0.){
   status[i]=1;return;
 }
 for(int k=offset[c];k<offset[c+1];++k){int f=faces[k],j=g*nf+f;
   if(face_status[j]){status[i]=1;return;}
   outgoing+=fmax(signs[k]*face_mass[j],0.);
 }
 if(!isfinite(outgoing)){status[i]=1;return;}
 if(outgoing>0.&&outgoing>=inventory){
   theta[i]=fmin(1.,inventory/outgoing)*(1.-32.*2.2204460492503131e-16);
 }
 if(!isfinite(theta[i])||theta[i]<0.||theta[i]>1.)status[i]=1;
}

extern "C" __global__ void m2b_diffusion_limit(
 int nc,int nf,int ng,const int* owner,const int* neighbour,
 const double* theta,const int* theta_status,double* face_mass,int* status)
{
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=ng*nf)return;
 if(status[i])return;int g=i/nf,f=i%nf;double value=face_mass[i];
 int donor=value>0.?owner[f]:(value<0.?neighbour[f]:-1);
 if(donor>=0){int j=g*nc+donor;if(theta_status[j]){status[i]=1;return;}
   value*=theta[j];}
 face_mass[i]=value;if(!isfinite(value))status[i]=1;
}

extern "C" __global__ void m2b_diffusion_update(
 int nc,int nf,int ng,const double* mass,const double* face_mass,
 const double* area,const int* offset,const int* faces,const int* signs,
 const int* face_status,double* next_mass,int* status)
{
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=ng*nc)return;
 int g=i/nc,c=i%nc;double value=mass[i];status[i]=0;
 for(int k=offset[c];k<offset[c+1];++k){int f=faces[k];if(face_status[g*nf+f]){status[i]=1;return;}
   value-=signs[k]*face_mass[g*nf+f]/area[c];}
 double tol=256.*2.2204460492503131e-16*fmax(1.,fabs(mass[i]));
 if(!isfinite(value)||value < -tol){status[i]=1;return;}next_mass[i]=fmax(0.,value);
}

extern "C" __global__ void m2b_diffusion_ledger(
 int nf,int ng,int nb,const int* boundary_offset,const int* boundary_faces,
 const double* face_mass,const int* face_status,double* ledger,int* status)
{
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=ng*nb)return;
 int g=i/nb,b=i%nb;double value=0.;status[i]=0;
 for(int k=boundary_offset[b];k<boundary_offset[b+1];++k){int f=boundary_faces[k];
   if(face_status[g*nf+f])status[i]=1;value-=face_mass[g*nf+f];}
 ledger[i]=value;if(!isfinite(value))status[i]=1;
}
