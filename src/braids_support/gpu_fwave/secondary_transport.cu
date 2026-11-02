// Fixed-grid secondary-flow branches of the native CPU equations.
// Model 1 uses raw curvature and diagnostic equilibrium hI each Euler stage.
// Model 2 carries signed conserved M=h*I with unchanged WENO-Z and FCT.
#define BRAIDS_SECONDARY_OUTLET_ABI 1
__device__ __forceinline__ double st_weno(const double* v){
 double p0=(2.*v[0]-7.*v[1]+11.*v[2])/6.,p1=(-v[1]+5.*v[2]+2.*v[3])/6.,p2=(2.*v[2]+5.*v[3]-v[4])/6.;
 double a0=v[0]-2.*v[1]+v[2],c0=v[0]-4.*v[1]+3.*v[2],a1=v[1]-2.*v[2]+v[3],c1=v[1]-v[3];
 double a2=v[2]-2.*v[3]+v[4],c2=3.*v[2]-4.*v[3]+v[4];
 double b0=(13./12.)*(a0*a0)+.25*(c0*c0),b1=(13./12.)*(a1*a1)+.25*(c1*c1),b2=(13./12.)*(a2*a2)+.25*(c2*c2);
 double scale=v[0]*v[0];for(int k=1;k<5;++k)scale=fmax(scale,v[k]*v[k]);
 double tau=fabs(b0-b2),eps=1.e-40+1.e-14*scale,r0=tau/(b0+eps),r1=tau/(b1+eps),r2=tau/(b2+eps);
 double w0=.1*(1.+r0*r0),w1=.6*(1.+r1*r1),w2=.3*(1.+r2*r2);return ((w0*p0+w1*p1)+w2*p2)/((w0+w1)+w2);
}
extern "C" __global__ void st_raw_curvature(int nc,int nf,const double* q,const double* xy,const int* owner,
 const int* neighbour,const int* offset,const int* faces,const int* signs,double dry,double* raw,int* status){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;raw[c]=0.;status[c]=0;
 if(q[c]<dry)return;double u=q[nc+c]/q[c],v=q[2*nc+c]/q[c],speed=sqrt(u*u+v*v);if(speed<1.e-3)return;
 double a11=0.,a12=0.,a22=0.,bx=0.,by=0.,cx=0.,cy=0.;
 for(int k=offset[c];k<offset[c+1];++k){int f=faces[k],n=signs[k]<0?owner[f]:neighbour[f];if(n<0)continue;
  double rx=xy[n]-xy[c],ry=xy[nc+n]-xy[nc+c],dist=fmax(sqrt(rx*rx+ry*ry),1.e-6),w=1./(dist*dist);
  double un=0.,vn=0.;if(q[n]>=dry){un=q[nc+n]/q[n];vn=q[2*nc+n]/q[n];}
  a11+=w*rx*rx;a12+=w*rx*ry;a22+=w*ry*ry;bx+=w*rx*(un-u);by+=w*ry*(un-u);cx+=w*rx*(vn-v);cy+=w*ry*(vn-v);
 }
 double det=a11*a22-a12*a12;if(det<1.e-6)return;
 double dudx=(bx*a22-a12*by)/det,dudy=(a11*by-bx*a12)/det,dvdx=(cx*a22-a12*cy)/det,dvdy=(a11*cy-cx*a12)/det;
 raw[c]=-(u*u*dvdx-v*v*dudy+u*v*(dvdy-dudx))/(speed*speed*speed);if(!isfinite(raw[c])){raw[c]=0.;status[c]=1;}
}
extern "C" __global__ void st_equilibrium(int nc,const double* q,const double* curvature,double dry,
 double* intensity,double* mass,int* status){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;intensity[c]=0.;mass[c]=0.;status[c]=0;
 double h=q[c];if(!isfinite(h)||h<0.||!isfinite(curvature[c])){status[c]=1;return;}
 if(h<=dry)return;double u=q[nc+c]/h,v=q[2*nc+c]/h,speed=sqrt(u*u+v*v);
 if(!isfinite(speed)){status[c]=1;return;}if(speed<1.e-3)return;
 intensity[c]=h*curvature[c]*speed;
 mass[c]=h*intensity[c];if(!isfinite(intensity[c])||!isfinite(mass[c])){intensity[c]=0.;mass[c]=0.;status[c]=1;}
}
extern "C" __global__ void st_relax(int nc,const double* q,double* mass,const double* curvature,const double* manning,
 const double* area,double dry,double g,double dt,double* source,int* status){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;source[c]=0.;status[c]=0;double h=q[c];
 if(h<=dry){source[c]=area[c]*(-mass[c]);mass[c]=0.;return;}
 double u=q[nc+c]/h,v=q[2*nc+c]/h,speed=sqrt(u*u+v*v);if(!isfinite(speed)||!isfinite(curvature[c])){status[c]=1;return;}
 if(speed<1.e-3||dt<=0.)return;
 double eq=h*curvature[c]*speed,chezy=pow(h,1./6.)/manning[c],alpha=sqrt(g)/(.41*fmax(chezy,20.));
 if(!isfinite(eq)||!isfinite(chezy)||chezy<=0.||!isfinite(alpha)||alpha<=0.||alpha>=.5){status[c]=1;return;}
 double time_a=((1.-2.*alpha)*h/(2.*.41*.41*alpha))/speed;
 if(!isfinite(time_a)||time_a<=0.){status[c]=1;return;}
 double target=h*eq,updated=mass[c],increment=0.;if(!isfinite(target)){status[c]=1;return;}
 if(time_a<=dt/50.){increment=target-mass[c];updated=target;}
 else{double ratio=dt/time_a,fraction;
  if(ratio<=0.)fraction=0.;else if(ratio<1.e-5)fraction=ratio*(1.+ratio*(-.5+ratio*(1./6.+ratio*(-1./24.+ratio*(1./120.-ratio/720.)))));
  else if(ratio>=50.)fraction=1.;else fraction=1.-exp(-ratio);
  increment=(target-mass[c])*fraction;updated=mass[c]+increment;}
 if(!isfinite(updated)||!isfinite(increment)){status[c]=1;return;}mass[c]=updated;source[c]=area[c]*increment;
}
extern "C" __global__ void st_primitive(int nc,const double* q,const double* mass,double* intensity,int* status){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;intensity[c]=q[c]>0.?mass[c]/q[c]:0.;
 status[c]=!isfinite(intensity[c])||!isfinite(mass[c])||q[c]<0.?1:0;
}
extern "C" __global__ void st_reconstruct_cells(int nc,int nf,int nadj,const double* q,const double* bed,const double* intensity,
 const double* xy,const double* mid,const double* normal,const int* owner,const int* neighbour,const int* bid,const int* bmode,
 const int* offset,const int* faces,const int* signs,const int* active,const double* matrix,const double* weight,const double* delta,
 double dry,double g,int outlet_extension,double* grad){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;grad[c]=0.;grad[nc+c]=0.;if(!active[c])return;
 double a11=matrix[c],a12=matrix[nc+c],a22=matrix[2*nc+c],det=matrix[3*nc+c],bx=0.,by=0.,zx=0.,zy=0.;
 double lower=intensity[c],upper=lower;
 for(int k=offset[c];k<offset[c+1];++k){int f=faces[k],n=signs[k]<0?owner[f]:neighbour[f];if(n<0)continue;
  double w=weight[k],rx=delta[k],ry=delta[nadj+k];bx+=w*rx*(intensity[n]-intensity[c]);by+=w*ry*(intensity[n]-intensity[c]);
  zx+=w*rx*(bed[n]-bed[c]);zy+=w*ry*(bed[n]-bed[c]);lower=fmin(lower,intensity[n]);upper=fmax(upper,intensity[n]);}
 if(det<=1.e-12*(a11+a22)*(a11+a22))return;
 double gx=(a22*bx-a12*by)/det,gy=(a11*by-a12*bx)/det,bgx=(a22*zx-a12*zy)/det,bgy=(a11*zy-a12*zx)/det,theta=1.;
 // Qualified outlet-extension-1 limiter extrema for subcritical mode-2 faces.
 // This changes the secondary-flow limiter bounds, not the f-wave ghost state.
 if(outlet_extension==1&&q[c]>dry){
  double u=q[nc+c]/q[c],v=q[2*nc+c]/q[c];
  for(int k=offset[c];k<offset[c+1];++k){
   int f=faces[k];
   if(neighbour[f]>=0||bid[f]<0||bmode[f]!=2)continue;
   if(fabs(u*normal[f]+v*normal[nf+f])>=sqrt(g*q[c]))continue;
   double rx=mid[f]-xy[c],ry=mid[nf+f]-xy[nc+c];
   if(fabs(bgx*rx+bgy*ry)<=64.*2.2204460492503131e-16*fmax(1.,fabs(bed[c])))continue;
   double exterior=intensity[c]+2.*(gx*rx+gy*ry);
   lower=fmin(lower,exterior);upper=fmax(upper,exterior);
  }
 }
 for(int k=offset[c];k<offset[c+1];++k){int f=faces[k];double rx=mid[f]-xy[c],ry=mid[nf+f]-xy[nc+c];
  if(neighbour[f]<0&&bid[f]>=0&&bmode[f]==2&&q[c]>dry){double u=q[nc+c]/q[c],v=q[2*nc+c]/q[c];
   if(fabs(u*normal[f]+v*normal[nf+f])<sqrt(g*q[c])&&fabs(bgx*rx+bgy*ry)>64.*2.2204460492503131e-16*fmax(1.,fabs(bed[c])))continue;}
  double z=gx*rx+gy*ry;if(z>0.)theta=fmin(theta,(upper-intensity[c])/z);if(z<0.)theta=fmin(theta,(lower-intensity[c])/z);}
 grad[c]=gx*fmax(0.,theta);grad[nc+c]=gy*fmax(0.,theta);
}
extern "C" __global__ void st_reconstruct_faces(int nc,int nf,const double* intensity,const double* grad,const int* owner,
 const int* neighbour,const int* weno,const double* displacement,const int* wall_ids,const int* wall_slot,const int* wall_ready,
 double* traces,int* status){
 int f=blockIdx.x*blockDim.x+threadIdx.x;if(f>=nf)return;status[f]=0;
 for(int side=0;side<2;++side){int c=side?neighbour[f]:owner[f];traces[side*nf+f]=0.;if(c<0)continue;
  double value;int slot=wall_slot[side*nf+f];
  if(slot>=0&&wall_ready[slot*nc+c]){double v[5];int axis=slot/2,which=slot%2;
   for(int k=0;k<5;++k){int kk=which==0?4-k:k;v[k]=intensity[wall_ids[(kk*2+axis)*nc+c]];}value=st_weno(v);}
  else{value=(intensity[c]+grad[c]*displacement[side*nf+f])+grad[nc+c]*displacement[(2+side)*nf+f];
   if(weno[side*nf+f]>=0){double v[5];for(int k=0;k<5;++k)v[k]=intensity[weno[(k*2+side)*nf+f]];value=st_weno(v);}}
  traces[side*nf+f]=value;if(!isfinite(value))status[f]=1;
 }
}
__device__ void st_stress(double h,double hu,double hv,double n,double g,double beta,double intensity,double* tensor,double* signal,int* status){
 tensor[0]=0.;tensor[1]=0.;tensor[2]=0.;*signal=0.;double u=hu/h,v=hv/h,speed=sqrt(u*u+v*v);if(speed<1.e-3)return;
 double chezy=pow(h,1./6.)/n,a=sqrt(g)/(.41*chezy),coef=beta*(5.*a-15.6*(a*a)+37.5*(a*a*a)),b=coef*intensity/speed;
 tensor[0]=-2.*h*b*u*v;tensor[1]=h*b*(u*u-v*v);tensor[2]=-tensor[0];*signal=2.*fabs(coef*intensity);
 if(!isfinite(tensor[0])||!isfinite(tensor[1])||!isfinite(*signal))*status=1;
}
extern "C" __global__ void st_add_stress(int nc,int nf,const double* trace,const double* intensity,const double* manning,
 const double* normal,const int* owner,const int* neighbour,const int* kind,const int* closed,double dry,double g,double beta,
 double* fl,double* fr,double* speed,double* traction,int* status){
 int f=blockIdx.x*blockDim.x+threadIdx.x;if(f>=nf)return;status[f]=0;traction[f]=0.;traction[nf+f]=0.;if(kind[f]==0||closed[f]||beta<=0.)return;
 int c=owner[f],n=neighbour[f];double l[3]={0.,0.,0.},r[3]={0.,0.,0.},sl=0.,sr=0.;int code=0;
 if(trace[f]>dry)st_stress(trace[f],trace[nf+f],trace[2*nf+f],manning[c],g,beta,intensity[f],l,&sl,&code);
 if(n>=0&&trace[4*nf+f]>dry){st_stress(trace[4*nf+f],trace[5*nf+f],trace[6*nf+f],manning[n],g,beta,intensity[nf+f],r,&sr,&code);
  for(int j=0;j<3;++j)l[j]=.5*(l[j]+r[j]);}
 double x=l[0]*normal[f]+l[1]*normal[nf+f],y=l[1]*normal[f]+l[2]*normal[nf+f];traction[f]=x;traction[nf+f]=y;
 fl[nf+f]+=x;fl[2*nf+f]+=y;fr[nf+f]+=x;fr[2*nf+f]+=y;speed[f]+=fmax(sl,sr);status[f]=code;
}
extern "C" __global__ void st_low_flux(int nc,int nf,const double* q,const double* intensity,const double* face,
 const double* curvature,const int* owner,const int* neighbour,const double* water,double dry,double* low,double* anti,double* boundary,int* status){
 int f=blockIdx.x*blockDim.x+threadIdx.x;if(f>=nf)return;status[f]=0;boundary[f]=0.;int c=owner[f],n=neighbour[f];
 if(water[f]>=0.){low[f]=water[f]*intensity[c];anti[f]=water[f]*face[f]-low[f];}
 else if(n>=0){low[f]=water[f]*intensity[n];anti[f]=water[f]*face[nf+f]-low[f];}
 else{double eq=0.,h=q[c];if(h>dry){double u=q[nc+c]/h,v=q[2*nc+c]/h,s=sqrt(u*u+v*v);if(s>=1.e-3)eq=h*curvature[c]*s;}
  boundary[f]=eq;low[f]=water[f]*eq;anti[f]=0.;}
 if(!isfinite(low[f])||!isfinite(anti[f]))status[f]=1;
}
extern "C" __global__ void st_fct_bounds(int nc,int nf,const double* intensity,const double* mass,const double* next,
 const double* area,const double* length,const int* owner,const int* neighbour,const int* bid,const int* offset,const int* faces,const int* signs,
 const int* active,const double* water,const double* low,const double* anti,const double* boundary,double dt,
 double* lowstate,double* plus,double* minus,int* status){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;status[c]=0;lowstate[c]=mass[c];plus[c]=1.;minus[c]=1.;if(!active[c])return;
 double pp=0.,pm=0.,value=mass[c],lower=intensity[c],upper=lower;
 for(int k=offset[c];k<offset[c+1];++k){int f=faces[k],n=signs[k]<0?owner[f]:neighbour[f];
  if(n>=0){lower=fmin(lower,intensity[n]);upper=fmax(upper,intensity[n]);}
  else if(bid[f]>=0&&water[f]<0.){lower=fmin(lower,boundary[f]);upper=fmax(upper,boundary[f]);}
  double z=-dt*signs[k]*length[f]/area[c];value+=z*low[f];pp+=fmax(z*anti[f],0.);pm+=fmin(z*anti[f],0.);}
 if(next[c]<=0.){lower=0.;upper=0.;}else{lower*=next[c];upper*=next[c];}
 double scale=512.*2.2204460492503131e-16*fmax(1.,fmax(fabs(value),fmax(fabs(lower),fabs(upper))));
 if(value<lower-scale||value>upper+scale){status[c]=6;return;}lowstate[c]=value;
 if(pp>0.)plus[c]=fmin(1.,fmax(0.,upper-value)/pp)*(1.-32.*2.2204460492503131e-16);
 if(pm<0.)minus[c]=fmin(1.,fmax(0.,value-lower)/(-pm))*(1.-32.*2.2204460492503131e-16);
}
extern "C" __global__ void st_fct_faces(int nf,const int* owner,const int* neighbour,const double* low,const double* plus,const double* minus,double* anti,double* flux){
 int f=blockIdx.x*blockDim.x+threadIdx.x;if(f>=nf)return;int c=owner[f],n=neighbour[f];double scale=1.;
 if(anti[f]>0.){scale=minus[c];if(n>=0)scale=fmin(scale,plus[n]);}
 else if(anti[f]<0.){scale=plus[c];if(n>=0)scale=fmin(scale,minus[n]);}
 anti[f]*=scale;flux[f]=low[f]+anti[f];
}
extern "C" __global__ void st_fct_update(int nc,const double* nextq,const double* mass,const double* lowstate,const double* area,const double* length,
 const int* offset,const int* faces,const int* signs,const int* active,const double* anti,double dt,double* next,int* status){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;next[c]=mass[c];if(!active[c]||status[c])return;double value=lowstate[c];
 for(int k=offset[c];k<offset[c+1];++k){int f=faces[k];value=value-dt*signs[k]*length[f]*anti[f]/area[c];}
 double scale=512.*2.2204460492503131e-16*fmax(1.,fabs(value));if(nextq[c]<=0.){if(fabs(value)<=scale)value=0.;else status[c]=6;}
 next[c]=value;if(!isfinite(value))status[c]=6;
}
extern "C" __global__ void st_ledger(int nb,const int* offset,const int* faces,const double* length,const double* flux,double dt,double* ledger){
 int b=blockIdx.x;if(b>=nb)return;__shared__ double sum[128];double value=0.;
 for(int k=offset[b]+threadIdx.x;k<offset[b+1];k+=blockDim.x){int f=faces[k];value-=dt*length[f]*flux[f];}
 sum[threadIdx.x]=value;__syncthreads();for(int stride=64;stride>0;stride/=2){if(threadIdx.x<stride)sum[threadIdx.x]+=sum[threadIdx.x+stride];__syncthreads();}
 if(threadIdx.x==0)ledger[b]=sum[0];
}
extern "C" __global__ void st_rotate(int nc,int ng,const double* q,const double* mass,const double* manning,double dry,double g,double espir,double* capacity,int* status){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=nc*ng)return;int c=i%nc;status[i]=0;if(q[c]<=dry)return;
 double h=q[c],u=q[nc+c]/h,v=q[2*nc+c]/h,qx=capacity[i],qy=capacity[ng*nc+i],intensity=mass[c]/h;
 double speed=sqrt(u*u+v*v),magnitude=sqrt(qx*qx+qy*qy);if(speed<=0.||magnitude<=0.)return;
 double chezy=pow(h,1./6.)/manning[c],alpha=12.5*espir*(1.-sqrt(g)/(.4*chezy))*intensity/speed;
 double tx=u+alpha*v,ty=v-alpha*u,direction=sqrt(tx*tx+ty*ty);
 if(direction<=2.2204460492503131e-16){qx=0.;qy=0.;}else{qx=magnitude*tx/direction;qy=magnitude*ty/direction;}
 capacity[i]=qx;capacity[ng*nc+i]=qy;if(!isfinite(qx)||!isfinite(qy))status[i]=1;
}
