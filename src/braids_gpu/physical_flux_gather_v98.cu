// Physical replacement precedes one donor limiter and the pressure cell gather.
// No Riemann, density, viscosity, boundary-memory or accepted-clock owner.
extern "C" __global__ void physical_replace_v98(
 int nf,const double* left0,const double* right0,const double* physical,
 const int* physical_mask,const int* neighbour,double* left,double* right,
 int* consumed,int* status){
 int f=blockIdx.x*blockDim.x+threadIdx.x;if(f>=nf)return;
 bool bad=physical_mask[f]!=0&&physical_mask[f]!=1;
 if(physical_mask[f]&&neighbour[f]>=0)bad=true;
 for(int k=0;k<3;k++){
  long long i=(long long)k*nf+f;
  if(!isfinite(left0[i])||!isfinite(right0[i])||!isfinite(physical[i]))bad=true;
  left[i]=physical_mask[f]?physical[i]:left0[i];
  right[i]=physical_mask[f]?physical[i]:right0[i];
 }
 if(left[f]!=right[f])bad=true;
 consumed[f]=physical_mask[f]?1:0;status[f]=bad?1:0;
}

extern "C" __global__ void physical_draining_v98(
 int nc,double dt,const double* q,const int* wet,const int* offset,
 const int* faces,const int* signs,const double* length,const double* area,
 const double* left,double* theta,int* status){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
 bool bad=wet[c]!=0&&wet[c]!=1;
 for(int k=0;k<3;k++)if(!isfinite(q[(long long)k*nc+c]))bad=true;
 if(q[c]<0.)bad=true;
 double outgoing=0.;
 for(int j=offset[c];j<offset[c+1];j++){
  int f=faces[j];outgoing+=fmax(signs[j]*left[f],0.)*length[f];
 }
 double available=wet[c]?q[c]*area[c]:0.,factor=1.;
 if(outgoing>0.&&dt*outgoing>=available)
  factor=fmin(1.,available/(dt*outgoing))*(1.-32.*2.2204460492503131e-16);
 if(!isfinite(factor)||factor<0.||factor>1.)bad=true;
 theta[c]=factor;status[c]=bad?1:0;
}

extern "C" __global__ void physical_limit_v98(
 int nc,int nf,double dry,const double* q,const int* wet,const int* owner,
 const int* neighbour,const double* left,const double* right,const double* theta,
 double* limited_left,double* limited_right,double* water,int* limited){
 int f=blockIdx.x*blockDim.x+threadIdx.x;if(f>=nf)return;
 for(int k=0;k<3;k++){
  long long i=(long long)k*nf+f;limited_left[i]=left[i];limited_right[i]=right[i];
 }
 int donor=-1;limited[f]=0;
 if(left[f]>0.)donor=owner[f];
 if(left[f]<0.&&neighbour[f]>=0)donor=neighbour[f];
 if(donor>=0&&theta[donor]<1.){
  double factor=theta[donor],correction=(factor-1.)*left[f];limited[f]=1;
  limited_left[f]=factor*left[f];limited_right[f]=limited_left[f];
  double u=0.,v=0.;
  if(wet[donor]&&q[donor]>dry){u=q[nc+donor]/q[donor];v=q[2LL*nc+donor]/q[donor];}
  limited_left[nf+f]+=correction*u;limited_right[nf+f]+=correction*u;
  limited_left[2LL*nf+f]+=correction*v;limited_right[2LL*nf+f]+=correction*v;
 }
 water[f]=limited_left[f];
}

extern "C" __global__ void physical_pressure_gather_v98(
 int nc,int nf,double gravity,const double* q,const int* wet,const int* offset,
 const int* faces,const int* signs,const double* length,const double* area,
 const double* normal,const double* left,const double* right,double* rate,
 int* status){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
 bool bad=status[c]!=0;
 double h=wet[c]?q[c]:0.,pressure=.5*gravity*h*h;
 for(int k=0;k<3;k++)rate[(long long)k*nc+c]=0.;
 for(int j=offset[c];j<offset[c+1];j++){
  int f=faces[j];double s=signs[j],w=length[f]/area[c];
  for(int k=0;k<3;k++){
   long long i=(long long)k*nf+f;
   double contribution=signs[j]<0?-right[i]:left[i];
   if(k>0)contribution-=s*pressure*normal[(long long)(k-1)*nf+f];
   rate[(long long)k*nc+c]-=contribution*w;
  }
 }
 for(int k=0;k<3;k++)if(!isfinite(rate[(long long)k*nc+c]))bad=true;
 status[c]=bad?1:0;
}
