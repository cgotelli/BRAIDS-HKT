// Additive actual-CUDA scalar-FCT interface kernels for Q4/v98.
//
// The ordinary concentration/limiter/update/physical-boundary-ledger kernels
// remain the frozen v92 kernels.  These kernels only replace the two stages
// that need a negative-neighbour coarse/fine trace, plus the separate AMR
// interface ledger.  Selected faces are never physical boundaries.

__device__ int v98_sa_extremum(const double* concentration,int nc,int g,
 const int ids[5],double& bound){
 double values[5],z[5],scale=0.;
 for(int k=0;k<5;++k){values[k]=concentration[g*nc+ids[k]];scale=fmax(scale,fabs(values[k]));}
 bound=values[2];if(scale==0.)return 1;
 double lo=1.e300,hi=-1.e300,tol=4096.*2.2204460492503131e-16;
 for(int k=0;k<5;++k){z[k]=values[k]/scale;lo=fmin(lo,z[k]);hi=fmax(hi,z[k]);}
 if(hi-lo<=tol)return 1;
 if((z[2]-z[1])*(z[3]-z[2])>0.)return 0;
 double d20=z[0]-2.*z[1]+z[2],d21=z[1]-2.*z[2]+z[3],d22=z[2]-2.*z[3]+z[4];
 double amin=fmin(fabs(d20),fmin(fabs(d21),fabs(d22)));
 double amax=fmax(fabs(d20),fmax(fabs(d21),fabs(d22)));
 if(amin<=tol||!((d20>0.&&d21>0.&&d22>0.)||(d20<0.&&d21<0.&&d22<0.))||amin<.5*amax)return 0;
 double slope=.5*(z[3]-z[1]),vertex=fmax(-.5,fmin(.5,-slope/d21));
 double point=z[2]+slope*vertex+.5*d21*vertex*vertex-d21/24.;
 if(d21<0.){bound=scale*(z[2]+2.*fmax(point-z[2],0.));return 2;}
 bound=fmax(0.,scale*(z[2]-2.*fmax(z[2]-point,0.)));return 3;
}

__device__ void v98_sa_smooth_bound(int c,int g,int nc,int nf,
 const double* concentration,const double* normal,const double* rectangle,
 const int* hydro_constant,const int* weno,const int* owner,
 const int* neighbour,const int* bid,const int* offset,const int* faces,
 double& lower,double& upper){
 if(hydro_constant[c]||rectangle[c]<=0.||rectangle[nc+c]<=0.)return;
 double lo=lower,hi=upper,scale=fmax(fmax(fabs(lower),fabs(upper)),2.2250738585072014e-308);
 double tol=4096.*2.2204460492503131e-16*scale;int extkind=1;
 for(int k=offset[c];k<offset[c+1];++k)if(bid[faces[k]]>=0)return;
 for(int axis=0;axis<2;++axis){int ids[5]={-1,-1,-1,-1,-1};
  for(int k=offset[c];k<offset[c+1];++k){int f=faces[k];
   if(fabs(normal[axis*nf+f])!=1.)continue;
   int side=c==owner[f]?0:1;
   if(weno[side*nf+f]>=0)
    for(int j=0;j<5;++j)ids[j]=weno[(j*2+side)*nf+f];
  }
  if(ids[0]<0){
   for(int k=offset[c];k<offset[c+1];++k){int f=faces[k],n=c==owner[f]?neighbour[f]:owner[f];
    if(fabs(normal[axis*nf+f])!=1.||n<0)continue;
    if(hydro_constant[n]||fabs(concentration[g*nc+n]-concentration[g*nc+c])>tol)return;
   }
   continue;
  }
  for(int j=0;j<5;++j)if(ids[j]<0||ids[j]>=nc||hydro_constant[ids[j]])return;
  double candidate;int kind=v98_sa_extremum(concentration,nc,g,ids,candidate);
  if(kind==0)return;if(kind==1)continue;
  if(extkind!=1&&kind!=extkind)return;extkind=kind;
  if(kind==2)hi=fmax(hi,candidate);if(kind==3)lo=fmin(lo,candidate);
 }
 lower=fmax(0.,lo);upper=hi;
}

extern "C" __global__ void v98_sa_face_low(
 int nc,int nf,int ng,int nb,const double* water,
 const double* concentration,const double* left,const double* right,
 const double* boundary,const double* interface_state,
 const int* interface_active,const int* owner,const int* neighbour,
 const int* bid,const int* conc_status,double* low_flux,double* anti,
 int* status){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=ng*nf)return;
 int g=i/nf,f=i%nf,c=owner[f],n=neighbour[f],b=bid[f];double w=water[f];
 status[i]=0;low_flux[i]=0.;anti[i]=0.;
 if(!isfinite(w)||conc_status[g*nc+c]||(n>=0&&conc_status[g*nc+n])){status[i]=1;return;}
 if(n<0&&b<0){
  if(interface_active[f]){
   double value=interface_state[i];
   if(!isfinite(value)||value<0.){status[i]=2;return;}
   if(w<0.){low_flux[i]=w*value;anti[i]=w*right[i]-low_flux[i];}
   else{low_flux[i]=w*concentration[g*nc+c];anti[i]=w*left[i]-low_flux[i];}
  }else if(w!=0.)status[i]=1;
  return;
 }
 if(interface_active[f]){status[i]=2;return;}
 if(n<0&&b>=0){
  if(w<0.){double value=boundary[g*nb+b];if(!isfinite(value)||value<0.){status[i]=1;return;}
   low_flux[i]=w*value;}
  else low_flux[i]=w*concentration[g*nc+c];
 }else if(w>=0.){
  double value=left[i];if(!isfinite(value)){status[i]=1;return;}
  low_flux[i]=w*concentration[g*nc+c];anti[i]=w*value-low_flux[i];
 }else{
  double value=right[i];if(!isfinite(value)){status[i]=1;return;}
  low_flux[i]=w*concentration[g*nc+n];anti[i]=w*value-low_flux[i];
 }
 if(!isfinite(low_flux[i])||!isfinite(anti[i]))status[i]=1;
}

extern "C" __global__ void v98_sa_cell_bounds(
 int nc,int nf,int ng,int nb,const double* mass,const double* next_depth,
 const double* concentration,const double* boundary,const double* interface_state,
 const int* interface_active,const double* water,const double* low_flux,
 const double* anti,const double* area,const double* length,const double* normal,
 const double* rectangle,const int* hydro_constant,const int* weno,
 const int* owner,const int* neighbour,const int* bid,const int* offset,
 const int* faces,const int* signs,const int* conc_status,
 const int* face_status,double dt,double* low_state,double* rplus,
 double* rminus,int* status){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=ng*nc)return;
 int g=i/nc,c=i%nc;status[i]=0;low_state[i]=mass[i];rplus[i]=1.;rminus[i]=1.;
 double nh=next_depth[c];if(conc_status[i]||!isfinite(nh)||nh<0.){status[i]=1;return;}
 double pp=0.,pm=0.,low=mass[i],lower=concentration[i],upper=lower;
 for(int k=offset[c];k<offset[c+1];++k){int f=faces[k],n=signs[k]>0?neighbour[f]:owner[f],b=bid[f];
  if(face_status[g*nf+f]){status[i]=1;return;}
  if(n>=0){lower=fmin(lower,concentration[g*nc+n]);upper=fmax(upper,concentration[g*nc+n]);}
  else if(interface_active[f]&&water[f]<0.){double value=interface_state[g*nf+f];
   lower=fmin(lower,value);upper=fmax(upper,value);}
  else if(b>=0&&water[f]<0.){double value=boundary[g*nb+b];
   lower=fmin(lower,value);upper=fmax(upper,value);}
  double z=-dt*signs[k]*length[f]/area[c],a=anti[g*nf+f];
  low+=z*low_flux[g*nf+f];pp+=fmax(z*a,0.);pm+=fmin(z*a,0.);
 }
 if(!isfinite(low)||low<0.){status[i]=1;return;}
 v98_sa_smooth_bound(c,g,nc,nf,concentration,normal,rectangle,
  hydro_constant,weno,owner,neighbour,bid,offset,faces,lower,upper);
 low_state[i]=low;
 if(pp>0.)rplus[i]=fmin(1.,fmax(0.,upper*nh-low-64.*2.2250738585072014e-308)/pp);
 if(pm<0.)rminus[i]=fmin(1.,fmax(0.,low-lower*nh-64.*2.2250738585072014e-308)/(-pm));
 if(pp>0.)rplus[i]*=(1.-32.*2.2204460492503131e-16);
 if(pm<0.)rminus[i]*=(1.-32.*2.2204460492503131e-16);
 if(!isfinite(rplus[i])||!isfinite(rminus[i]))status[i]=1;
}

extern "C" __global__ void v98_sa_interface_ledger(
 int nf,int ng,int ni,const int* interface_faces,const double* flux,
 const double* length,const int* flux_status,double dt,double* ledger,
 int* status){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=ng*ni)return;
 int g=i/ni,s=i%ni,f=interface_faces[s];status[i]=0;
 if(f<0||f>=nf||flux_status[g*nf+f]){ledger[i]=0.;status[i]=1;return;}
 double value=-dt*length[f]*flux[g*nf+f];ledger[i]=value;
 if(!isfinite(value))status[i]=1;
}
