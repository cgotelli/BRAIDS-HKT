// Explicit Juez Phi/theta_c^S/geometric-angle reconstruction. Research only.
// Both OpenMP and CUDA compile this exact arithmetic. No practical-Smart edits.
#ifdef __CUDACC__
#define ENTRY extern "C" __global__
#define LOCAL __device__
#define LOOP(N) for(int i=blockDim.x*blockIdx.x+threadIdx.x;i<N;i+=blockDim.x*gridDim.x)
#else
#include <cmath>
#include <omp.h>
using std::isfinite;
#define ENTRY extern "C" __declspec(dllexport)
#define LOCAL inline
#define LOOP(N) _Pragma("omp parallel for schedule(static)") for(int i=0;i<N;++i)
extern "C" __declspec(dllexport) void mobile_threads(int n){omp_set_dynamic(0);omp_set_num_threads(n);}
#endif
// Material: meanDm,D30,D90,rho_s,rho_w,Manning_n,tan(repose),transport_depth.
// Jacobian is d(qbx,qby)/d(h,hu,hv,z_x,z_y), solid m2/s, not mass or bulk.
LOCAL int cfbs_law(const double*p,double h,double mx,double my,double gx,double gy,
 double*flux,double*jac,double*diag){
 for(int k=0;k<2;++k)flux[k]=0.;for(int k=0;k<10;++k)jac[k]=0.;
 for(int k=0;k<5;++k)diag[k]=0.;
 if(!isfinite(h)||!isfinite(mx)||!isfinite(my)||!isfinite(gx)||!isfinite(gy)||h<0.)return 1;
 if(h<=p[7])return 0;
 double m=hypot(mx,my);if(m==0.)return 0;
 double ex=mx/m,ey=my/m,u=m/h,delta=p[3]/p[4]-1.,n2=p[5]*p[5];
 double sb=-gx*ex-gy*ey,sf=n2*u*u/pow(h,4./3.);
 double theta=n2*u*u/(delta*p[0]*cbrt(h));
 double tc=.047*(1.-sb/p[6])/sqrt(1.+sb*sb),s=sb>0.?sb:sf;
 diag[0]=sb;diag[1]=sf;diag[2]=theta;diag[3]=tc;diag[4]=s;
 // Super-repose states require explicit redistribution, never threshold clipping.
 if(fabs(sb)>p[6]*(1.+1.e-12))return 3;
 double excess=theta-tc;if(excess<=0.)return 0;
 double cstar=pow(h,1./6.)/(sqrt(9.81)*p[5]);
 double root=sqrt(theta),sp=pow(s,.6),k=4.*pow(p[2]/p[1],.2)*sqrt(9.81*delta*pow(p[0],3));
 double qb=k*sp*cstar*root*excess;
 double dsb[5]={0.,-(gx+sb*ex)/m,-(gy+sb*ey)/m,-ex,-ey};
 double dt[5]={-7.*theta/(3.*h),2.*theta*ex/m,2.*theta*ey/m,0.,0.};
 double dsf[5]={-10.*sf/(3.*h),2.*sf*ex/m,2.*sf*ey/m,0.,0.};
 double dtc=.047*(-1./p[6]-sb)/pow(1.+sb*sb,1.5);
 flux[0]=qb*ex;flux[1]=qb*ey;
 for(int c=0;c<5;++c){
  double ds=sb>0.?dsb[c]:dsf[c];
  double dqb=k*(.6*pow(s,-.4)*ds*cstar*root*excess+
    sp*cstar*((.5*excess/root+root)*dt[c]-root*dtc*dsb[c])+
    (c==0?sp*cstar*root*excess/(6.*h):0.));
  double dex=c==1?(1.-ex*ex)/m:c==2?-ex*ey/m:0.;
  double dey=c==1?-ex*ey/m:c==2?(1.-ey*ey)/m:0.;
  jac[c]=ex*dqb+qb*dex;jac[5+c]=ey*dqb+qb*dey;
 }
 if(!isfinite(qb))return 2;for(int c=0;c<10;++c)if(!isfinite(jac[c]))return 2;
 return 0;
}
ENTRY void cfbs_probe(int n,const double*p,const double*q,const double*gradient,
 double*field,double*jacobian,double*diagnostic,int*status){
 LOOP(n){double f[2],j[10],d[5];status[i]=cfbs_law(p,q[i],q[n+i],q[2*n+i],gradient[i],gradient[n+i],f,j,d);
  for(int k=0;k<6;++k)field[k*n+i]=0.;
  field[i]=field[3*n+i]=p[3]*f[0];field[n+i]=field[4*n+i]=p[3]*f[1];field[2*n+i]=d[2];
  for(int k=0;k<10;++k)jacobian[k*n+i]=j[k];for(int k=0;k<5;++k)diagnostic[k*n+i]=d[k];
 }
}
ENTRY void cfbs_gradient(int n,const int*off,const int*faces,const int*owner,const int*nb,
 const double*xy,const double*z,double*gradient,int*rank){
 LOOP(n){double aa=0.,ab=0.,bb=0.,rx=0.,ry=0.;
  for(int k=off[i];k<off[i+1];++k){int f=faces[k],b=owner[f]==i?nb[f]:owner[f];if(b<0)continue;
   // Multiple face pieces joining the same cells are one geometric sample.
   bool duplicate=false;for(int l=off[i];l<k;++l){int g=faces[l],c=owner[g]==i?nb[g]:owner[g];if(c==b){duplicate=true;break;}}
   if(duplicate)continue;
   double dx=xy[b]-xy[i],dy=xy[n+b]-xy[n+i],r2=dx*dx+dy*dy;if(r2<=0.)continue;
   double dv=z[b]-z[i];aa+=dx*dx/r2;ab+=dx*dy/r2;bb+=dy*dy/r2;rx+=dx*dv/r2;ry+=dy*dv/r2;
  }
  double trace=aa+bb,det=aa*bb-ab*ab;gradient[i]=gradient[n+i]=0.;rank[i]=0;
  if(det>128.*2.2204460492503131e-16*trace*trace){gradient[i]=(bb*rx-ab*ry)/det;gradient[n+i]=(aa*ry-ab*rx)/det;rank[i]=2;}
  else if(trace>0.){gradient[i]=rx/trace;gradient[n+i]=ry/trace;rank[i]=1;}
 }
}
LOCAL double cfbs_component(int n,int i,int c,double ax,double ay,const double*value){
 return c==1?ax*value[n+i]+ay*value[2*n+i]:value[c*n+i];
}
LOCAL double cfbs_phi(int n,int nf,int i,int c,double ax,double ay,double gx,double gy,
 const double*value,const int*off,const int*faces,const int*owner,const int*nb,const double*disp){
 double vi=cfbs_component(n,i,c,ax,ay,value),lo=vi,hi=vi;
  for(int k=off[i];k<off[i+1];++k){int f=faces[k],b=owner[f]==i?nb[f]:owner[f];if(b<0)continue;
   double v=cfbs_component(n,b,c,ax,ay,value);lo=fmin(lo,v);hi=fmax(hi,v);}
  double phi=1.;
  double eps=64.*2.2204460492503131e-16*fmax(fabs(vi),fmax(fabs(lo),fabs(hi)));
  for(int k=off[i];k<off[i+1];++k){int f=faces[k];if(nb[f]<0)continue;int side=owner[f]==i?0:2;
   double dv=gx*disp[side*nf+f]+gy*disp[(side+1)*nf+f];
   if(dv>eps)phi=fmin(phi,(hi-vi)/dv);else if(dv< -eps)phi=fmin(phi,(lo-vi)/dv);
  }
  return fmax(0.,phi);
}
// Scalars use physical LS. Vector traces are bounded in a covariant local
// flow frame; independent Cartesian bounds would introduce grid orientation.
ENTRY void cfbs_limit(int n,int nf,const double*value,const int*off,const int*faces,
 const int*owner,const int*nb,const double*disp,double*gradient){
 LOOP(n){
  for(int c=0;c<4;c+=3){double gx=gradient[2*c*n+i],gy=gradient[(2*c+1)*n+i];
   double phi=cfbs_phi(n,nf,i,c,0.,0.,gx,gy,value,off,faces,owner,nb,disp);
   gradient[2*c*n+i]=phi*gx;gradient[(2*c+1)*n+i]=phi*gy;}
  double ex=value[n+i],ey=value[2*n+i],speed=hypot(ex,ey);
  if(speed==0.)for(int k=off[i];k<off[i+1];++k){int f=faces[k],b=owner[f]==i?nb[f]:owner[f];if(b<0)continue;
   ex=value[n+b];ey=value[2*n+b];speed=hypot(ex,ey);if(speed>0.)break;}
  if(speed>0.){ex/=speed;ey/=speed;}else{ex=1.;ey=0.;}
  double sx=ex*gradient[2*n+i]+ey*gradient[4*n+i],sy=ex*gradient[3*n+i]+ey*gradient[5*n+i];
  double tx=-ey*gradient[2*n+i]+ex*gradient[4*n+i],ty=-ey*gradient[3*n+i]+ex*gradient[5*n+i];
  double ps=cfbs_phi(n,nf,i,1,ex,ey,sx,sy,value,off,faces,owner,nb,disp);
  double pt=cfbs_phi(n,nf,i,1,-ey,ex,tx,ty,value,off,faces,owner,nb,disp);
  gradient[2*n+i]=ex*ps*sx-ey*pt*tx;gradient[3*n+i]=ex*ps*sy-ey*pt*ty;
  gradient[4*n+i]=ey*ps*sx+ex*pt*tx;gradient[5*n+i]=ey*ps*sy+ex*pt*ty;
 }
}
// Shared-face Exner flux at true nonorthogonal gradients. No hydraulic jump
// is injected into bed flux; physical gradient dependence supplies diffusion.
// coeff stores normal diffusion conductance; frequency is assembled by cells.
ENTRY void cfbs_faces(int nf,int n,const double*p,const double*q,const double*z,const double*gradient,const double*tracegrad,
 const int*owner,const int*nb,const int*bedbc,const double*normal,const double*xy,
 const double*disp,const double*feed,double*flux,double*coeff,double*face_diag,int*status){
 LOOP(nf){int a=owner[i],b=nb[i];flux[i]=coeff[i]=0.;status[i]=0;for(int k=0;k<4;++k)face_diag[k*nf+i]=0.;
  if(b<0&&bedbc[i]==0)continue;
  if(b<0&&bedbc[i]==1){flux[i]=-feed[i];continue;}
  if(q[a]<=p[7]||(b>=0&&q[b]<=p[7]))continue;
  double nx=normal[i],ny=normal[nf+i],h=q[a],u=q[n+a]/h,v=q[2*n+a]/h,gx=gradient[a],gy=gradient[n+a],dn=1.;
  if(b>=0){
   double dx=xy[b]-xy[a],dy=xy[n+b]-xy[n+a];dn=dx*nx+dy*ny;
   if(!(dn>0.)){status[i]=4;continue;}
   gx=.5*(gx+gradient[b]);gy=.5*(gy+gradient[n+b]);
   double corr=(z[b]-z[a]-gx*dx-gy*dy)/dn;gx+=nx*corr;gy+=ny*corr;
   double trace[2][3];for(int side=0;side<2;++side){int cell=side?b:a;
    double prim[3]={q[cell],q[n+cell]/q[cell],q[2*n+cell]/q[cell]};
    for(int c=0;c<3;++c)trace[side][c]=prim[c]+tracegrad[2*c*n+cell]*disp[2*side*nf+i]+tracegrad[(2*c+1)*n+cell]*disp[(2*side+1)*nf+i];
   }
   if(trace[0][0]<=p[7]||trace[1][0]<=p[7])continue;
   double rt0=sqrt(trace[0][0]),rt1=sqrt(trace[1][0]);u=(rt0*trace[0][1]+rt1*trace[1][1])/(rt0+rt1);v=(rt0*trace[0][2]+rt1*trace[1][2])/(rt0+rt1);h=.5*(trace[0][0]+trace[1][0]);
  }
  double f[2],j[10],d[5];status[i]=cfbs_law(p,h,h*u,h*v,gx,gy,f,j,d);
  if(status[i])continue;
  flux[i]=nx*f[0]+ny*f[1];if(b<0)flux[i]=fmax(0.,flux[i]);
  double kn=-(nx*(j[3]*nx+j[4]*ny)+ny*(j[8]*nx+j[9]*ny));
  if(b>=0&&kn< -1.e-12){status[i]=5;continue;}
  coeff[i]=b>=0?fmax(0.,kn)/dn:0.;
  face_diag[i]=d[0];face_diag[nf+i]=kn;face_diag[2*nf+i]=h;face_diag[3*nf+i]=hypot(u,v);
 }
}
ENTRY void cfbs_frequency(int n,const int*off,const int*faces,const double*length,const double*area,
 const double*conductance,double packing,double*frequency){
 LOOP(n){double rate=0.;for(int k=off[i];k<off[i+1];++k){int f=faces[k];rate+=length[f]*conductance[f];}
  frequency[i]=rate/(packing*area[i]);
 }
}
ENTRY void cfbs_gather(int n,const int*off,const int*faces,const int*signs,const double*length,
 const double*area,const double*flux,double packing,double*rate){
 LOOP(n){double r=0.;for(int k=off[i];k<off[i+1];++k){int f=faces[k];r-=signs[k]*length[f]*flux[f];}rate[i]=r/(packing*area[i]);}
}
// Frozen first-order SWE/Exner symbol plus separate gradient diffusion.
// Non-real first-order roots are diagnosed, never clamped into a real model.
ENTRY void cfbs_wave_probe(int n,const double*p,double packing,const double*q,const double*gradient,
 const double*normal,double*out,int*status){
 LOOP(n){for(int k=0;k<6;++k)out[k*n+i]=0.;
  double f[2],j[10],di[5],h=q[i];status[i]=cfbs_law(p,h,q[n+i],q[2*n+i],gradient[i],gradient[n+i],f,j,di);
  if(status[i]||h<=p[7])continue;
  double nx=normal[i],ny=normal[n+i],u=(nx*q[n+i]+ny*q[2*n+i])/h,v=(-ny*q[n+i]+nx*q[2*n+i])/h,gh=9.81*h;
  double a=(nx*j[1]+ny*j[6])/packing,b=(nx*j[2]+ny*j[7])/packing;
  double der[3]={(nx*j[0]+ny*j[5])/packing,a*nx+b*ny,-a*ny+b*nx};
  out[2*n+i]=-(nx*(nx*j[3]+ny*j[8])+ny*(nx*j[4]+ny*j[9]))/packing;
  for(int k=0;k<3;++k)out[(3+k)*n+i]=der[k];
  double cb=-2.*u,cc=u*u-gh*(1.+der[1]),cd=-gh*(der[0]+v*der[2]);
  double pp=cc-cb*cb/3.,qq=2.*cb*cb*cb/27.-cb*cc/3.+cd;
  if(!(pp<0.)){status[i]=6;continue;}
  double radius=2.*sqrt(-pp/3.),arg=(3.*qq/(2.*pp))*sqrt(-3./pp);
  if(fabs(arg)>1.+1.e-10){status[i]=6;continue;}
  double theta=acos(fmax(-1.,fmin(1.,arg)))/3.,node[4],coef[4],sgn[4],speed=fabs(u);
  for(int k=0;k<3;++k){node[k]=radius*cos(theta-2.*3.14159265358979323846*k/3.)-cb/3.;speed=fmax(speed,fabs(node[k]));}
  node[3]=u;out[i]=speed;
  for(int k=0;k<4;++k)for(int l=k+1;l<4;++l)if(node[l]<node[k]){double t=node[k];node[k]=node[l];node[l]=t;}
  for(int k=0;k<4;++k)coef[k]=sgn[k]=fabs(node[k])<=1.e-13*speed?0.:(node[k]>0.?1.:-1.);
  for(int first=0;first<4;){int last=first;while(last<3&&node[last+1]-node[last]<=1.e-12*speed)++last;
   if(sgn[first]!=sgn[last])for(int k=first;k<=last;++k)coef[k]=sgn[k]=0.;first=last+1;}
  bool bad=false;
  for(int order=1;order<4;++order)for(int k=3;k>=order;--k){double den=node[k]-node[k-order];
   if(fabs(den)<=1.e-12*speed){if(sgn[k]!=sgn[k-order]){bad=true;break;}coef[k]=0.;}
   else coef[k]=(coef[k]-coef[k-1])/den;}
  if(bad){status[i]=7;continue;}
  double mat[4][4]={{0.,1.,0.,0.},{gh-u*u,2.*u,0.,gh},{-u*v,v,u,0.},{der[0],der[1],der[2],0.}};
  double row[4]={0.,0.,0.,coef[3]};
  for(int k=2;k>=0;--k){double next[4]={0.,0.,0.,coef[k]};for(int r=0;r<4;++r)for(int c=0;c<4;++c)next[c]+=row[r]*(mat[r][c]-(r==c?node[k]:0.));for(int c=0;c<4;++c)row[c]=next[c];}
  out[n+i]=fmax(0.,row[1]*gh);
 }
}
