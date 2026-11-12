// Isolated GPU regeneration of native v35 diffusion geometry after wet/dry change.
// Fixed cell slots avoid host compaction; all stencil and inverse work is GPU.
extern "C" __global__ void mg_faces(const double* q,const double* bed,const double* xy,
 const double* mid,const double* normal,const double* length,const int* owner,const int* neighbour,
 long long nc,long long nf,double dry,double nu,double* conductance,double* diffusivity,
 double* tangent,double* asymmetry,int* status){
 long long f=blockIdx.x*(long long)blockDim.x+threadIdx.x;if(f>=nf)return;
 conductance[f]=diffusivity[f]=tangent[f]=asymmetry[f]=0.;status[f]=0;
 int l=owner[f],r=neighbour[f];if(r<0)return;
 const double eps=2.220446049250313080847263336181640625e-16,tiny=2.225073858507201383e-308;
 double dx=xy[r]-xy[l],dy=xy[nc+r]-xy[nc+l],dleng=sqrt(dx*dx+dy*dy);
 double nx=normal[f],ny=normal[nf+f],dn=dx*nx+dy*ny;
 if(!isfinite(dn)||dn<=64.*eps*fmax(dleng,tiny)){status[f]=1;return;}
 tangent[f]=-dx*ny+dy*nx;
 double dl=(mid[f]-xy[l])*nx+(mid[nf+f]-xy[nc+l])*ny;
 double dr=(xy[r]-mid[f])*nx+(xy[nc+r]-mid[nf+f])*ny;asymmetry[f]=dl-dr;
 double scale=fmax(1.,fmax(dleng,fabs(dl)+fabs(dr)));
 scale=fmax(scale,sqrt(xy[l]*xy[l]+xy[nc+l]*xy[nc+l]));
 scale=fmax(scale,sqrt(xy[r]*xy[r]+xy[nc+r]*xy[nc+r]));
 scale=fmax(scale,sqrt(mid[f]*mid[f]+mid[nf+f]*mid[nf+f]));
 if(fabs(tangent[f])<=512.*eps*scale)tangent[f]=0.;
 if(fabs(asymmetry[f])<=512.*eps*scale)asymmetry[f]=0.;
 double h1=q[l],h2=q[r];if(h1<=dry||h2<=dry)return;
 double hf=.5*((bed[l]+h1)+(bed[r]+h2))-fmax(bed[l],bed[r]);if(hf<=.5*dry)return;
 double activation=hf<dry?fmax(0.,fmin(1.,(hf-.5*dry)/(.5*dry))):1.;
 conductance[f]=nu*length[f]*activation/dn;diffusivity[f]=conductance[f]*dn;
 if(!isfinite(conductance[f]))status[f]=1;
}
extern "C" __global__ void mg_boundary(const double* q,const double* conductance,const int* owner,
 const int* neighbour,const int* offset,const int* faces,const int* signs,long long nc,double dry,int* boundary){
 long long c=blockIdx.x*(long long)blockDim.x+threadIdx.x;if(c>=nc)return;
 boundary[c]=0;if(q[c]<=dry)return;
 for(int k=offset[c];k<offset[c+1];++k){int f=faces[k],r=signs[k]<0?owner[f]:neighbour[f];
  if(r<0||conductance[f]<=0.){boundary[c]=1;return;}}
}
extern "C" __global__ void mg_corrected(const double* conductance,const double* tangent,const double* asymmetry,
 const int* owner,const int* neighbour,const int* boundary,long long nf,int* corrected){
 long long f=blockIdx.x*(long long)blockDim.x+threadIdx.x;if(f>=nf)return;
 corrected[f]=0;if(conductance[f]<=0.)return;
 corrected[f]=tangent[f]!=0.||asymmetry[f]!=0.||boundary[owner[f]]||boundary[neighbour[f]];
}
extern "C" __global__ void mg_slots(const int* corrected,const int* offset,const int* faces,long long nc,int* slot){
 long long c=blockIdx.x*(long long)blockDim.x+threadIdx.x;if(c>=nc)return;slot[c]=-1;
 for(int k=offset[c];k<offset[c+1];++k)if(corrected[faces[k]]){slot[c]=(int)c;return;}
}
__device__ bool mg_append(int center,int r,int* ids,int* count){
 if(r==center)return true;for(int i=0;i<*count;++i)if(ids[i]==r)return true;
 if(*count>=256)return false;ids[(*count)++]=r;return true;
}
__device__ bool mg_inverse(const double* matrix,int n,double* inverse){
 double a[50]={0.},diagonal[5],candidate[25],temporary[25];
 for(int i=0;i<n;++i){if(!isfinite(matrix[i*n+i])||matrix[i*n+i]<=0.)return false;diagonal[i]=sqrt(matrix[i*n+i]);}
 for(int i=0;i<n;++i)for(int j=0;j<n;++j){a[i*(2*n)+j]=matrix[i*n+j]/(diagonal[i]*diagonal[j]);if(!isfinite(a[i*(2*n)+j]))return false;}
 for(int i=0;i<n;++i)a[i*(2*n)+n+i]=1.;
 for(int i=0;i<n;++i){int pivot=i;for(int j=i+1;j<n;++j)if(fabs(a[j*(2*n)+i])>fabs(a[pivot*(2*n)+i]))pivot=j;
  double pv=fabs(a[pivot*(2*n)+i]);if(!isfinite(pv)||pv<=1e-12)return false;
  if(pivot!=i)for(int j=0;j<2*n;++j){double t=a[i*(2*n)+j];a[i*(2*n)+j]=a[pivot*(2*n)+j];a[pivot*(2*n)+j]=t;}
  double divisor=a[i*(2*n)+i];for(int j=0;j<2*n;++j)a[i*(2*n)+j]/=divisor;
  for(int j=0;j<n;++j){if(j==i)continue;double factor=a[j*(2*n)+i];for(int k=0;k<2*n;++k)a[j*(2*n)+k]-=factor*a[i*(2*n)+k];}}
 for(int i=0;i<n;++i)for(int j=0;j<n;++j)temporary[i*n+j]=a[i*(2*n)+n+j]/(diagonal[i]*diagonal[j]);
 for(int i=0;i<n;++i)for(int j=0;j<n;++j){candidate[i*n+j]=.5*(temporary[i*n+j]+temporary[j*n+i]);if(!isfinite(candidate[i*n+j]))return false;}
 for(int i=0;i<n;++i)for(int j=0;j<n;++j){double residual=0.;for(int k=0;k<n;++k)residual+=matrix[i*n+k]*candidate[k*n+j];
  if(i==j)residual-=1.;if(!isfinite(residual)||fabs(residual)>1e-8)return false;}
 for(int i=0;i<n*n;++i)inverse[i]=candidate[i];return true;
}
extern "C" __global__ void mg_recovery(const double* area,const double* xy,const double* mid,const double* normal,
 const double* length,const double* conductance,const int* owner,const int* neighbour,const int* offset,
 const int* faces,const int* signs,const int* slot,long long nc,long long nf,int* stencils,int* counts,
 double* inverse,double* affine,int* order,int* valid){
 long long c=blockIdx.x*(long long)blockDim.x+threadIdx.x;if(c>=nc)return;
 for(int k=0;k<256;++k)stencils[k*nc+c]=-1;
 for(int k=0;k<25;++k)inverse[k*nc+c]=0.;for(int k=0;k<4;++k)affine[k*nc+c]=0.;
 counts[c]=order[c]=valid[c]=0;if(slot[c]<0||area[c]<=0.)return;
 int ids[256],count=0;bool complete=true;
 for(int k=offset[c];k<offset[c+1];++k){int f=faces[k],r=signs[k]<0?owner[f]:neighbour[f];
  if(r>=0&&conductance[f]>0.)if(!mg_append((int)c,r,ids,&count)){complete=false;break;}}
 int first=count;
 for(int j=0;j<first&&complete;++j)for(int k=offset[ids[j]];k<offset[ids[j]+1];++k){int f=faces[k],r=signs[k]<0?owner[f]:neighbour[f];
  if(r>=0&&conductance[f]>0.)if(!mg_append((int)c,r,ids,&count)){complete=false;break;}}
 counts[c]=count;for(int k=0;k<count;++k)stencils[k*nc+c]=ids[k];if(!complete)return;
 const double rt=1.414213562373095048801688724209698;double h=sqrt(area[c]),matrix[25]={0.};
 for(int k=0;k<count;++k){int r=ids[k];double zx=(xy[r]-xy[c])/h,zy=(xy[nc+r]-xy[nc+c])/h,w=1./(zx*zx+zy*zy);
  double row[5]={zx,zy,zx*zx/rt,zx*zy,zy*zy/rt};
  for(int i=0;i<5;++i)for(int j=0;j<5;++j)matrix[i*5+j]+=w*row[i]*row[j];}
 for(int k=offset[c];k<offset[c+1];++k){int f=faces[k],r=signs[k]<0?owner[f]:neighbour[f];if(r>=0&&conductance[f]>0.)continue;
  double zx=(mid[f]-xy[c])/h,zy=(mid[nf+f]-xy[nc+c])/h,w=length[f]/h,nx=normal[f],ny=normal[nf+f];
  double row[5]={nx,ny,rt*nx*zx,nx*zy+ny*zx,rt*ny*zy};
  for(int i=0;i<5;++i)for(int j=0;j<5;++j)matrix[i*5+j]+=w*row[i]*row[j];}
 double answer[25];if(mg_inverse(matrix,5,answer)){order[c]=2;valid[c]=1;for(int k=0;k<25;++k)inverse[k*nc+c]=answer[k];return;}
 if(count<2)return;double m2[4]={matrix[0],matrix[1],matrix[5],matrix[6]};
 if(mg_inverse(m2,2,answer)){order[c]=1;valid[c]=1;for(int k=0;k<4;++k)affine[k*nc+c]=answer[k];}
}
__device__ double mg_l1(int c,int f,const double* area,const double* xy,const double* mid,const double* normal,
 const int* stencils,const int* counts,const double* inverse,const double* affine,const int* order,long long nc,long long nf){
 const double rt=1.414213562373095048801688724209698;double h=sqrt(area[c]),zx=(mid[f]-xy[c])/h,zy=(mid[nf+f]-xy[nc+c])/h;
 double nx=normal[f],ny=normal[nf+f],eval[5]={nx/h,ny/h,rt*nx*zx/h,(nx*zy+ny*zx)/h,rt*ny*zy/h};
 int n=order[c]==2?5:2;double value=0.,total=0.;
 for(int k=0;k<counts[c];++k){int r=stencils[k*nc+c];double x=(xy[r]-xy[c])/h,y=(xy[nc+r]-xy[nc+c])/h,w=1./(x*x+y*y);
  double row[5]={x,y,x*x/rt,x*y,y*y/rt},coefficient=0.;
  for(int i=0;i<n;++i){double product=0.;for(int j=0;j<n;++j)product+=(n==5?inverse[(i*5+j)*nc+c]:affine[(i*2+j)*nc+c])*row[j];coefficient+=eval[i]*product;}
  coefficient*=w;value+=fabs(coefficient);total+=coefficient;}
 return value+fabs(total);
}
extern "C" __global__ void mg_bound(const double* area,const double* xy,const double* mid,const double* normal,
 const double* conductance,const double* diffusivity,const int* owner,const int* neighbour,const int* slot,
 const int* stencils,const int* counts,const double* inverse,const double* affine,const int* order,const int* valid,
 long long nc,long long nf,int* corrected,double* bound){
 long long f=blockIdx.x*(long long)blockDim.x+threadIdx.x;if(f>=nf)return;bound[f]=0.;if(conductance[f]<=0.)return;
 bound[f]=2.*conductance[f];if(!corrected[f])return;int l=owner[f],r=neighbour[f];
 if(slot[l]<0||slot[r]<0)return;if(!valid[l]||!valid[r]){corrected[f]=0;return;}
 double sl=mg_l1(l,f,area,xy,mid,normal,stencils,counts,inverse,affine,order,nc,nf);
 double sr=mg_l1(r,f,area,xy,mid,normal,stencils,counts,inverse,affine,order,nc,nf);
 if(!isfinite(sl)||!isfinite(sr)){corrected[f]=0;return;}
 double recovery=.5*diffusivity[f]*(sl+sr);if(!isfinite(recovery)||recovery<0.){corrected[f]=0;return;}
 bound[f]=fmax(bound[f],recovery);
}
extern "C" __global__ void mg_rate(const double* q,const double* area,const double* bound,const int* offset,
 const int* faces,long long nc,double dry,double* rate){
 long long c=blockIdx.x*(long long)blockDim.x+threadIdx.x;if(c>=nc)return;rate[c]=0.;if(q[c]<=dry)return;
 for(int k=offset[c];k<offset[c+1];++k)rate[c]+=.5*bound[faces[k]];rate[c]/=area[c];
}
