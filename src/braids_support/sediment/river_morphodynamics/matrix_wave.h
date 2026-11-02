// Research bed-row f-wave splitting of a frozen-coefficient SWE--Exner
// Jacobian. Augmented hydraulic flux is NOT replaced. Physical EH/KF unchanged.
#ifdef __CUDACC__
#define ENTRY extern "C" __global__
#define LOCAL __device__
#define LOOP(N) for(int i=blockDim.x*blockIdx.x+threadIdx.x;i<N;i+=blockDim.x*gridDim.x)
#else
#include <cmath>
#include <omp.h>
#define ENTRY extern "C" __declspec(dllexport)
#define LOCAL inline
#define LOOP(N) _Pragma("omp parallel for schedule(static)") for(int i=0;i<N;++i)
extern "C" __declspec(dllexport) void mobile_threads(int n){omp_set_dynamic(0);omp_set_num_threads(n);}
#endif
LOCAL double value(int c,int i,int n,const double*q,const double*z){
 return c==0?q[i]:(c==3?z[i]:q[c*n+i]/q[i]);
}
ENTRY void wave_state_gradients(int n,int nf,const double*q,const double*z,
 const int*off,const int*faces,const int*owner,const int*nb,const double*delta,
 const double*inv,const double*disp,double*grad){
 LOOP(n){int na=off[n];
  for(int c=0;c<4;++c){double v=value(c,i,n,q,z),lo=v,hi=v,bx=0.,by=0.;
   for(int k=off[i];k<off[i+1];++k){int f=faces[k],j=owner[f]==i?nb[f]:owner[f];if(j<0)continue;
    double dx=delta[k],dy=delta[na+k],w=1./(dx*dx+dy*dy),dv=value(c,j,n,q,z);
    lo=fmin(lo,dv);hi=fmax(hi,dv);bx+=w*dx*(dv-v);by+=w*dy*(dv-v);
   }
   double gx=inv[i]*bx+inv[n+i]*by,gy=inv[n+i]*bx+inv[2*n+i]*by,phi=1.;
   double eps=64.*2.2204460492503131e-16*fmax(fabs(v),fmax(fabs(lo),fabs(hi)));
   for(int k=off[i];k<off[i+1];++k){int f=faces[k];if(nb[f]<0)continue;
    int side=owner[f]==i?0:2;double d=gx*disp[side*nf+f]+gy*disp[(side+1)*nf+f];
    if(d>eps)phi=fmin(phi,(hi-v)/d);else if(d<-eps)phi=fmin(phi,(lo-v)/d);
   }
   grad[2*c*n+i]=fmax(0.,phi)*gx;grad[(2*c+1)*n+i]=fmax(0.,phi)*gy;
  }
 }
}
ENTRY void matrix_bed_faces(int nf,int n,double acceleration,const double*q,const double*z,
 const double*cell,const double*capgrad,const double*stategrad,const int*owner,const int*nb,
 const double*normal,const double*disp,double*face,double*diagnostic){
 LOOP(nf){int o=owner[i],j=nb[i];for(int k=0;k<3;++k)diagnostic[k*nf+i]=0.;if(j<0)continue;
  double nx=normal[i],ny=normal[nf+i],s[2][4],qs[2],der[3]={0.,0.,0.};
  for(int side=0;side<2;++side){int a=side?j:o;
   double dx=disp[2*side*nf+i],dy=disp[(2*side+1)*nf+i];
   for(int c=0;c<4;++c)s[side][c]=value(c,a,n,q,z)+stategrad[2*c*n+a]*dx+stategrad[(2*c+1)*n+a]*dy;
   double vx=s[side][1],vy=s[side][2];s[side][1]=vx*nx+vy*ny;s[side][2]=-vx*ny+vy*nx;
   double x=cell[3*n+a]+capgrad[a]*dx+capgrad[n+a]*dy;
   double y=cell[4*n+a]+capgrad[2*n+a]*dx+capgrad[3*n+a]*dy;
   qs[side]=acceleration*(x*nx+y*ny)/2650.;
  }
  double h=.5*(s[0][0]+s[1][0]),rt0=sqrt(s[0][0]),rt1=sqrt(s[1][0]);
  double u=(rt0*s[0][1]+rt1*s[1][1])/(rt0+rt1),v=(rt0*s[0][2]+rt1*s[1][2])/(rt0+rt1),gh=9.81*h;
  // Positive EH principal-part linearization at ONE mean state. Averaging
  // cell Jacobians evaluated at different speeds violates the homogeneous
  // relation dh+u*dm_n+v*dm_t=0 and can produce complex roots. Steering and
  // effective shear remain in the physical flux; only the dissipative wave
  // matrix uses a frozen mean magnitude and unrotated EH velocity derivative.
  double sp2=u*u+v*v;
  double mag=.5*(hypot(cell[3*n+o],cell[4*n+o])+hypot(cell[3*n+j],cell[4*n+j]))/2650.;
  if(!(h>0.)||sp2<=1.e-20||mag<=0.)continue;
  double factor=acceleration*mag/(sqrt(sp2)*h);
  der[0]=-5.*factor*u;der[1]=factor*(1.+4.*u*u/sp2);der[2]=factor*4.*u*v/sp2;
  double b=-2.*u,c=u*u-gh*(1.+der[1]),d=-gh*(der[0]+v*der[2]);
  double p=c-b*b/3.,qq=2.*b*b*b/27.-b*c/3.+d;
  if(p>=0.){diagnostic[2*nf+i]=1.;continue;}
  double radius=2.*sqrt(-p/3.),arg=(3.*qq/(2.*p))*sqrt(-3./p);
  if(fabs(arg)>1.+1.e-10){diagnostic[2*nf+i]=2.;continue;}
  double theta=acos(fmax(-1.,fmin(1.,arg)))/3.,roots[3],speed=fabs(u);
  for(int k=0;k<3;++k){roots[k]=radius*cos(theta-2.*3.14159265358979323846*k/3.)-b/3.;speed=fmax(speed,fabs(roots[k]));}
  // General Newton/Hermite polynomial of sign(A). Unlike the native MPM
  // shortcut, allow two eigenvalues on either side of zero at oblique faces.
  // Repeated same-sign roots have zero divided differences, not a pole.
  double node[4]={roots[0],roots[1],roots[2],u},coef[4],sgn[4];
  for(int k=0;k<4;++k)for(int l=k+1;l<4;++l)if(node[l]<node[k]){double tmp=node[k];node[k]=node[l];node[l]=tmp;}
  for(int k=0;k<4;++k)coef[k]=sgn[k]=fabs(node[k])<=1.e-13*speed?0.:(node[k]>0.?1.:-1.);
  // Numerically coincident stationary waves: a cubic's roundoff can place
  // one root just across zero. Treat the entire machine-scale cluster as
  // stationary; never divide a sign jump by a roundoff-size eigenvalue gap.
  // This changes waves only within 1e-12 of the spectral radius, not a
  // tunable physical velocity/transport threshold.
  for(int first=0;first<4;){int last=first;
   while(last<3&&node[last+1]-node[last]<=1.e-12*speed)++last;
   if(sgn[first]!=sgn[last])for(int k=first;k<=last;++k)coef[k]=sgn[k]=0.;
   first=last+1;
  }
  bool bad=false;
  for(int order=1;order<4;++order)for(int k=3;k>=order;--k){
   double den=node[k]-node[k-order];
   if(fabs(den)<=1.e-12*speed){
    if(sgn[k]!=sgn[k-order]){bad=true;break;}
    coef[k]=0.;
   }else coef[k]=(coef[k]-coef[k-1])/den;
  }
  if(bad){diagnostic[2*nf+i]=4.;continue;}
  double a[4][4]={{0.,1.,0.,0.},{gh-u*u,2.*u,0.,gh},{-u*v,v,u,0.},{der[0],der[1],der[2],0.}};
  double row[4]={0.,0.,0.,coef[3]};
  for(int k=2;k>=0;--k){double next[4]={0.,0.,0.,coef[k]};
   for(int r=0;r<4;++r)for(int col=0;col<4;++col)next[col]+=row[r]*(a[r][col]-(r==col?node[k]:0.));
   for(int col=0;col<4;++col)row[col]=next[col];
  }
  double jump[4]={s[1][0]*s[1][1]-s[0][0]*s[0][1],
    s[1][0]*(s[1][1]*s[1][1]+.5*9.81*s[1][0])-s[0][0]*(s[0][1]*s[0][1]+.5*9.81*s[0][0])+gh*(s[1][3]-s[0][3]),
    s[1][0]*s[1][1]*s[1][2]-s[0][0]*s[0][1]*s[0][2],qs[1]-qs[0]};
  double diss=0.;for(int k=0;k<4;++k)diss+=row[k]*jump[k];
  double next=(.5*(qs[0]+qs[1]-diss))*2650./acceleration;
  double old=face[i]*nx+face[nf+i]*ny;
  face[i]+=(next-old)*nx;face[nf+i]+=(next-old)*ny;
  diagnostic[i]=speed;diagnostic[nf+i]=next-old;
 }
}
