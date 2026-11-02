// Bend bundle integration operators, shared FP64 CPU/OpenMP and CUDA bodies.
// Seeded from the frozen attached_pdf_v2 testbed; public defaults unchanged.
// Arrays are SoA. No reinterpretation of historical Delft3D secondary state.
#ifdef __CUDACC__
#define FN __device__ __forceinline__
#define ENTRY extern "C" __global__
#define LOOP(N) for(int i=blockDim.x*blockIdx.x+threadIdx.x;i<N;i+=blockDim.x*gridDim.x)
#else
#include <cmath>
#include <omp.h>
using std::isfinite;
#define FN inline
#define ENTRY extern "C" __declspec(dllexport)
#define LOOP(N) _Pragma("omp parallel for schedule(static)") for(int i=0;i<N;++i)
extern "C" __declspec(dllexport) void bend_threads(int n){omp_set_dynamic(0);omp_set_num_threads(n);}
#endif

FN double mm(double a,double b){return a*b<=0.?0.:(fabs(a)<fabs(b)?a:b);}
FN double norm2(double x,double y){return sqrt(x*x+y*y);}

// Least-squares gradients (velocity, bed, secondary velocity, mobile storage).
// dxdy is per cell adjacency and inverse least-squares matrices are precomputed.
ENTRY void gradients(int nc,const double*q,const double*z,const double*S,const double*M,
 const int*off,const int*faces,const int*owner,const int*neighbour,
 const double*dxdy,const double*inv,double*out){
 LOOP(nc){
  double x[5]={q[nc+i]/q[i],q[2*nc+i]/q[i],z[i],S[i]/q[i],M[i]};
  double bx[5]={0.,0.,0.,0.,0.},by[5]={0.,0.,0.,0.,0.};
  const int na=off[nc];
  for(int s=off[i];s<off[i+1];++s){
   int f=faces[s],j=owner[f]==i?neighbour[f]:owner[f]; if(j<0)continue;
   double dx=dxdy[s],dy=dxdy[na+s],wt=1./(dx*dx+dy*dy);
   double y[5]={q[nc+j]/q[j],q[2*nc+j]/q[j],z[j],S[j]/q[j],M[j]};
   for(int k=0;k<5;++k){bx[k]+=wt*dx*(y[k]-x[k]);by[k]+=wt*dy*(y[k]-x[k]);}
  }
  for(int k=0;k<5;++k){out[(2*k)*nc+i]=inv[i]*bx[k]+inv[nc+i]*by[k];
   out[(2*k+1)*nc+i]=inv[nc+i]*bx[k]+inv[2*nc+i]*by[k];}
 }
}

// p: g,dry,Manning,d50,relative_density,porosity,theta_c,Cg1,Cspi,
//    As,Ds,Lb,beta,capacity_longitudinal_multiplier,turbulence_Ct.
//    ripple_mu,LV_chi,wall_sand_roughness,molecular_viscosity.
// method 1 = Lai unit/prescribed-beta bedload; 2/3/4 = LV BS/Odgaard/KB.
// method 5 = explicit BRAIDS power-law torque reconstruction, not recovered
// author code. Keep the LV Table1 normalization and empirical decay, replace
// the linear-profile centrifugal moment by the exact Eq4 power-law moment.
// out: qbx,qby,Dxx,Dxy,Dyy,P,lambda,capacity,w,theta,curvature,fD,nu,beta,Lb.
// beta/Lb are populated by the separate Lai adaptation kernel.
FN double bend_constitutive_cell(int i,int nc,int method,int saturation,int instantaneous,const double*p,
 const double*q,const double*S,const double*grad,double*out,int*status,const double*curvature=0){
  status[i]=0;for(int k=0;k<13;++k)out[k*nc+i]=0.;
  double h=q[i],u=q[nc+i]/h,v=q[2*nc+i]/h,w=norm2(u,v);
  if(h<=p[1]||!isfinite(w)){status[i]=1;return 0.;}
  out[11*nc+i]=1.;out[8*nc+i]=w;if(w==0.)return 0.;
  double tx=u/w,ty=v/w,C=pow(h,1./6.)/p[2],cf=p[0]/(C*C),us=sqrt(cf)*w;
  double theta=us*us/((p[4]-1.)*p[0]*p[3]),m=.4*C/sqrt(p[0]);
  double ux=grad[i],uy=grad[nc+i],vx=grad[2*nc+i],vy=grad[3*nc+i];
  double curv=(v*(u*ux+v*uy)-u*(u*vx+v*vy))/(w*w*w);
  if(curvature)curv=curvature[i];
  double zx=grad[4*nc+i],zy=grad[5*nc+i],vns=S[i]/h;
  double fd=1.,kp=0.,kd=0.,x,y,scale,cap;
  if(method==1){
   scale=2.*p[8]/(.41*.41)*(1.-sqrt(p[0])/(.41*C))*h*curv;
   x=tx+scale*ty;y=ty-scale*tx;double r=norm2(x,y);x/=r;y/=r;
   scale=p[7]/(.85*sqrt(theta));x-=scale*zx;y-=scale*zy;
   cap=.05/cf*pow(theta,2.5)*sqrt((p[4]-1.)*p[0]*p[3]*p[3]*p[3]);
  }else{
   if(saturation==2){
    // Explicit BRAIDS research extension, not an attributed author policy.
    // beta^4 = h^2*(.41/m)^(-2.2)*k*(k + d_n(w)/w).
    // This is Eq12 on its domain, without division by curvature. Negative
    // interaction receives its continuous beta=0 limit (fD=1). Status 6
    // records EVERY extension cell; strict mode below remains unchanged.
    double wx=tx*ux+ty*vx,wy=tx*uy+ty*vy;
    double interaction=curv*(curv+(-ty*wx+tx*wy)/w);
    if(interaction<0.)status[i]=6;
    double b=sqrt(h)*pow(.41/m,-.55)*pow(fmax(0.,interaction),.25);
    double den=b*(1.05*b*b-.89*b+.5);if(den>0.)fd=-expm1(-.4/den);
   }else if(saturation&&fabs(curv)>1.e-14){
    double wx=tx*ux+ty*vx,wy=tx*uy+ty*vy;
    double alpha=(-ty*wx+tx*wy)/(w*curv);
    if(alpha< -1.-1.e-12){status[i]=2;return 0.;}
    // Only roundoff-level negative radicands are rounded to zero.
    double b=sqrt(h*fabs(curv))*pow(.41/m,-.55)*pow(fmax(0.,alpha+1.),.25);
    double den=b*(1.05*b*b-.89*b+.5);if(den>0.)fd=-expm1(-.4/den);
   }
   if(method==2){kp=sqrt(6.*p[9])*.4/(2.*m);kd=p[10]*.4/m;}
   if(method==3){kp=(2.*m+1.)/(m*(m+1.));kd=2.*.4*.4/(m+1.);}
   if(method==4){if(m<=2.){status[i]=3;return 0.;}kp=3./m;kd=2.*.4*.4/(m-2.);}
   if(method==5){
    // J = integral_0^1 (eta-.5)*[((m+1)/m)*eta^(1/m)]^2 d_eta
    //   = (m+1)/(2*m*(m+2)). For the old linear profile,
    // J_linear=sqrt(6*As)*kappa/(6*m), so Table1 has kp=3*J_linear.
    // This profile replacement is testable but does not fully match Fig2;
    // do not call it the authenticated LV2023 source or silently select it.
    kp=1.5*(m+1.)/(m*(m+2.));kd=p[10]*.4/m;
   }
   // LV Eq11: local adaptation uses the CURRENT velocity/depth/curvature.
   // The identical local velocity feeds both momentum dispersion and bedload
   // direction. There is no transported state or artificial relaxation time.
   if(instantaneous){
    if(kd<=0.){status[i]=7;return 0.;}
    vns=fd*(kp/kd)*w*h*curv;
   }
   double ss=h*w*w/(m*(m+2.)),sn=h*w*vns/(1.+2.*m),nn=h*vns*vns/3.;
   out[2*nc+i]=ss*tx*tx-2.*sn*tx*ty+nn*ty*ty;
   out[3*nc+i]=(ss-nn)*tx*ty+sn*(tx*tx-ty*ty);
   out[4*nc+i]=ss*ty*ty+2.*sn*tx*ty+nn*tx*tx;
   scale=vns/w;x=tx+scale*ty;y=ty-scale*tx;double r=norm2(x,y);x/=r;y/=r;
   scale=1./(9.*pow(p[3]/h,.3)*sqrt(theta));x-=scale*zx;y-=scale*zy;
   // Typeset Eq.35: theta_c^4 + mu*theta^4, NOT (mu*theta)^4.
   double ratio=theta/p[6],a4=p[15]*ratio*ratio*ratio*ratio;
   double T=p[6]*expm1(.25*log1p(a4));
   // Typeset Eq.34 MULTIPLIES chi*c_f. The paper's c_f=Ks^2*h^(4/3)/g
   // is not the dimensionless drag coefficient cf used for Shields above.
   // Preserve the published SI empirical expression; no guessed reciprocal.
   double longitudinal=1.-p[16]*pow(h,4./3.)/(p[2]*p[2]*p[0])*(tx*zx+ty*zy);
   if(longitudinal<0.){status[i]=5;return 0.;}
   cap=8.*sqrt((p[4]-1.)*p[0]*p[3]*p[3]*p[3])*pow(T,1.5)*p[13]*longitudinal;
  }
  double r=norm2(x,y);if(r==0.||!isfinite(cap)){status[i]=4;return 0.;}
  out[i]=cap*x/r;out[nc+i]=cap*y/r;
  out[5*nc+i]=fd*kp*w*w*curv;out[6*nc+i]=kd*w/h;
  out[7*nc+i]=cap;out[9*nc+i]=theta;out[10*nc+i]=curv;out[11*nc+i]=fd;
  out[12*nc+i]=p[14]*us*h;
  return vns;
}

// Preserve the transported entry point and its ABI for existing diagnostics.
ENTRY void constitutive(int nc,int method,int saturation,const double*p,
 const double*q,const double*S,const double*grad,double*out,int*status){
 LOOP(nc){bend_constitutive_cell(i,nc,method,saturation,0,p,q,S,grad,out,status);}
}

ENTRY void bend_constitutive_instantaneous(int nc,int method,int saturation,const double*p,
 const double*q,const double*S,const double*grad,double*out,int*status,double*localW){
 LOOP(nc){localW[i]=4.*bend_constitutive_cell(i,nc,method,saturation,1,p,q,S,grad,out,status);}
}

ENTRY void bend_constitutive_curvature(int nc,int method,int saturation,int instantaneous,const double*p,
 const double*q,const double*S,const double*grad,const double*curvature,double*out,int*status,double*localW){
 LOOP(nc){localW[i]=4.*bend_constitutive_cell(i,nc,method,saturation,instantaneous,p,q,S,grad,out,status,curvature);}
}

// Research Eq22 reconstruction on the native Cartesian/cut-cell graph.
// Backtrack with the current stage velocity. Locate a nearby donor by a
// connectivity-constrained nearest-centroid walk, then reconstruct its OLD
// velocity with LS gradients. No triangulation bridges opposite channel banks.
// status 1 explicitly records one-sided extension beyond an exterior face;
// >1 rejects the research footprint, rather than silently clipping velocity.
// This is not the unavailable 2DEF nodal interpolation/trajectory algorithm.
ENTRY void bend_backtracked_curvature(int nc,double tau,const double*q,const double*oldq,
 const double*oldgrad,const double*xy,const double*area,const double*mid,const double*normal,
 int nf,const int*off,const int*faces,const int*owner,const int*nb,double*out,int*status){
 LOOP(nc){
  double u=q[nc+i]/q[i],v=q[2*nc+i]/q[i],w=norm2(u,v);out[i]=0.;status[i]=0;
  if(w==0.)continue;
  if(tau<=0.||w*tau>1.5*sqrt(area[i])){status[i]=2;continue;}
  double px=xy[i]-tau*u,py=xy[nc+i]-tau*v;
  int donor=i;bool found=false;
  for(int it=0;it<16;++it){
   int best=donor;double dx=px-xy[donor],dy=py-xy[nc+donor],rr=dx*dx+dy*dy;
   for(int s=off[donor];s<off[donor+1];++s){
    int f=faces[s],j=owner[f]==donor?nb[f]:owner[f];if(j<0)continue;
    dx=px-xy[j];dy=py-xy[nc+j];double dist=dx*dx+dy*dy;
    if(dist<rr){rr=dist;best=j;}
   }
   if(best==donor){found=true;break;}donor=best;
  }
  if(!found){status[i]=3;continue;}
  double dx=px-xy[donor],dy=py-xy[nc+donor];
  if(dx*dx+dy*dy>2.25*area[donor]){status[i]=2;continue;}
  for(int s=off[donor];s<off[donor+1];++s){int f=faces[s];
   if(nb[f]<0 && (px-mid[f])*normal[f]+(py-mid[nf+f])*normal[nf+f]>1.e-12)
    status[i]=1;
  }
  double u0=oldq[nc+donor]/oldq[donor]+oldgrad[donor]*dx+oldgrad[nc+donor]*dy;
  double v0=oldq[2*nc+donor]/oldq[donor]+oldgrad[2*nc+donor]*dx+oldgrad[3*nc+donor]*dy;
  out[i]=(u*v0-v*u0)/(w*w*w*tau);
  if(!isfinite(out[i]))status[i]=4;
 }
}

ENTRY void secondary_source(int n,double dt,const double*q,const double*op,
 const double*old,double*next){
 LOOP(n){double lam=op[6*n+i],e=exp(-dt*lam);
  double phi=lam>0.?-expm1(-dt*lam)/lam:dt;
  next[i]=old[i]*e+q[i]*op[5*n+i]*phi;}
}

// Shared auxiliary face fluxes: full tensor, parabolic mixing, sediment,
// secondary advection. Wall water/sediment flux is exactly zero.
ENTRY void auxiliary_faces(int nf,int nc,int method,const double*p,const double*q,
 const double*S,const double*M,const double*op,const double*grad,const double*water,
 const int*owner,const int*nb,const double*normal,const double*distance,
 const double*displacement,double*out){
 LOOP(nf){
  int o=owner[i],j=nb[i];double nx=normal[i],ny=normal[nf+i];
  double dxx=op[2*nc+o],dxy=op[3*nc+o],dyy=op[4*nc+o];
  double fx=0.,fy=0.,sed=0.,sec=0.;
  if(j>=0){
   double lo=distance[i],lj=distance[nf+i],sum=lo+lj,wo=lj/sum,wj=lo/sum;
   dxx=wo*dxx+wj*op[2*nc+j];dxy=wo*dxy+wj*op[3*nc+j];dyy=wo*dyy+wj*op[4*nc+j];
   fx=dxx*nx+dxy*ny;fy=dxy*nx+dyy*ny;
   double nu=.5*(op[12*nc+o]+op[12*nc+j]),hh=fmin(q[o],q[j]);
   // Symmetric two-point turbulent shear; vector Laplacian, identified in report.
   fx-=hh*nu*(q[nc+j]/q[j]-q[nc+o]/q[o])/sum;
   fy-=hh*nu*(q[2*nc+j]/q[j]-q[2*nc+o]/q[o])/sum;
   double so=S[o]/q[o],sj=S[j]/q[j];int up=water[i]>=0.?o:j;
   double sr=up==o?so:sj;
   // Pair-limited MUSCL reconstruction, preserves the sign of secondary flow.
   double ds=(grad[6*nc+up]*displacement[(up==o?0:2)*nf+i]+
              grad[7*nc+up]*displacement[(up==o?1:3)*nf+i]);
   sr+=mm(ds,(up==o?sj-so:so-sj));sec=water[i]*sr;
   if(method==1){
    double co=op[7*nc+o],cj=op[7*nc+j];
    double ex=co>0.?op[o]/co:0.,ey=co>0.?op[nc+o]/co:0.;
    double jx=cj>0.?op[j]/cj:0.,jy=cj>0.?op[nc+j]/cj:0.;
    double bo=op[13*nc+o],bj=op[13*nc+j];
    double vel=(wo*bo*op[8*nc+o]*ex+wj*bj*op[8*nc+j]*jx)*nx+
               (wo*bo*op[8*nc+o]*ey+wj*bj*op[8*nc+j]*jy)*ny;
    up=vel>=0.?o:j;double val=M[up];
    ds=grad[8*nc+up]*displacement[(up==o?0:2)*nf+i]+grad[9*nc+up]*displacement[(up==o?1:3)*nf+i];
    val+=mm(ds,up==o?M[j]-M[o]:M[o]-M[j]);sed=vel*val;
   }else sed=(wo*op[o]+wj*op[j])*nx+(wo*op[nc+o]+wj*op[nc+j])*ny;
  }else{
   // Slip wall: only normal dispersion traction, no tangential momentum flux.
   double stress=dxx*nx*nx+2.*dxy*nx*ny+dyy*ny*ny;fx=stress*nx;fy=stress*ny;
   if(method!=1&&p[17]>0.){
    double u=q[nc+o]/q[o],v=q[2*nc+o]/q[o],w=norm2(u,v),ut=-u*ny+v*nx;
    if(w>0.&&ut!=0.){
     double C=pow(q[o],1./6.)/p[2],us=sqrt(p[0])*w/C;
     double dw=50.*p[18]/us;
     if(dw<=p[17]){for(int k=0;k<4;++k)out[k*nf+i]=nan("");continue;}
     double ratio=.4/log(dw/p[17]);
     // Eq.31 per density, integrated over the wet wall height. The face
     // length is applied once in gather_hydro, and force opposes slip.
     double traction=q[o]*w*w*ratio*ratio*(ut>0.?1.:-1.);
     fx-=traction*ny;fy+=traction*nx;
    }
   }
  }
  out[i]=fx;out[nf+i]=fy;out[2*nf+i]=sed;out[3*nf+i]=sec;
 }
}

// Linear-exact reconstruction of the dispersion tensor on skew/merged cells.
// Geometry is Cartesian in the interior, but cut-cell centroids need not be
// aligned with face normals. No transport capacity or sediment slope changes.
ENTRY void bend_tensor_gradients(int nc,const double*op,const int*off,const int*faces,
 const int*owner,const int*nb,const double*dxdy,const double*inv,double*out){
 LOOP(nc){double bx[3]={0.,0.,0.},by[3]={0.,0.,0.};int na=off[nc];
  for(int s=off[i];s<off[i+1];++s){int f=faces[s],j=owner[f]==i?nb[f]:owner[f];if(j<0)continue;
   double dx=dxdy[s],dy=dxdy[na+s],weight=1./(dx*dx+dy*dy);
   for(int k=0;k<3;++k){double delta=op[(k+2)*nc+j]-op[(k+2)*nc+i];
    bx[k]+=weight*dx*delta;by[k]+=weight*dy*delta;}}
  for(int k=0;k<3;++k){out[2*k*nc+i]=inv[i]*bx[k]+inv[nc+i]*by[k];
   out[(2*k+1)*nc+i]=inv[nc+i]*bx[k]+inv[2*nc+i]*by[k];}
 }
}

ENTRY void bend_linear_momentum_faces(int nf,int nc,const double*q,const double*op,
 const double*grad,const double*tgrad,const int*owner,const int*nb,const double*normal,
 const double*distance,const double*disp,double*out){
 LOOP(nf){int o=owner[i],j=nb[i];if(j<0)continue;
  double nx=normal[i],ny=normal[nf+i],lo=distance[i],lj=distance[nf+i],dn=lo+lj;
  double wo=lj/dn,wj=lo/dn,tensor[3];
  for(int k=0;k<3;++k){
   double left=op[(k+2)*nc+o]+tgrad[2*k*nc+o]*disp[i]+tgrad[(2*k+1)*nc+o]*disp[nf+i];
   double right=op[(k+2)*nc+j]+tgrad[2*k*nc+j]*disp[2*nf+i]+tgrad[(2*k+1)*nc+j]*disp[3*nf+i];
   tensor[k]=wo*left+wj*right;
  }
  double dx=disp[i]-disp[2*nf+i],dy=disp[nf+i]-disp[3*nf+i];
  double nu=.5*(op[12*nc+o]+op[12*nc+j]),hh=fmin(q[o],q[j]);
  double force[2]={tensor[0]*nx+tensor[1]*ny,tensor[1]*nx+tensor[2]*ny};
  for(int k=0;k<2;++k){
   double gx=wo*grad[2*k*nc+o]+wj*grad[2*k*nc+j];
   double gy=wo*grad[(2*k+1)*nc+o]+wj*grad[(2*k+1)*nc+j];
   // Over-relaxed nonorthogonal correction: exact for any affine velocity,
   // identical to the old two-point derivative when d is parallel to n.
   double difference=q[(k+1)*nc+j]/q[j]-q[(k+1)*nc+o]/q[o];
   double derivative=(difference-gx*(dx-dn*nx)-gy*(dy-dn*ny))/dn;
   out[k*nf+i]=force[k]-hh*nu*derivative;
  }
 }
}

// Published LV Eq29 tensor trace: convex centroid-to-face distance average.
// A covariance tensor must remain positive semidefinite. Unlike unlimited
// tensor extrapolation, this trace preserves that property on every face.
// Retain the linear-exact nonorthogonal VISCOUS derivative separately: the
// diagnostic option changes dispersion only, not turbulence or boundaries.
ENTRY void bend_eq29_momentum_faces(int nf,int nc,const double*q,const double*op,
 const double*grad,const int*owner,const int*nb,const double*normal,
 const double*distance,const double*disp,double*out){
 LOOP(nf){int o=owner[i],j=nb[i];if(j<0)continue;
  double nx=normal[i],ny=normal[nf+i],lo=distance[i],lj=distance[nf+i],dn=lo+lj;
  double wo=lj/dn,wj=lo/dn;
  double eo=norm2(disp[i],disp[nf+i]),ej=norm2(disp[2*nf+i],disp[3*nf+i]);
  double tensor[3];
  for(int k=0;k<3;++k)tensor[k]=(ej*op[(k+2)*nc+o]+eo*op[(k+2)*nc+j])/(eo+ej);
  double dx=disp[i]-disp[2*nf+i],dy=disp[nf+i]-disp[3*nf+i];
  double nu=.5*(op[12*nc+o]+op[12*nc+j]),hh=fmin(q[o],q[j]);
  double force[2]={tensor[0]*nx+tensor[1]*ny,tensor[1]*nx+tensor[2]*ny};
  for(int k=0;k<2;++k){
   double gx=wo*grad[2*k*nc+o]+wj*grad[2*k*nc+j];
   double gy=wo*grad[(2*k+1)*nc+o]+wj*grad[(2*k+1)*nc+j];
   double difference=q[(k+1)*nc+j]/q[j]-q[(k+1)*nc+o]/q[o];
   double derivative=(difference-gx*(dx-dn*nx)-gy*(dy-dn*ny))/dn;
   out[k*nf+i]=force[k]-hh*nu*derivative;
  }
 }
}

// Correct only LV's internal-face bed-slope direction. On a collocated mesh,
// averaging cell-centred gradients has a two-cell null mode. The secant term
// restores sensitivity to adjacent bed differences, including on skew faces.
// This is a numerical reconstruction, NOT a new physical slope coefficient.
// Capacity (including any longitudinal multiplier) stays cell-centred.
// Open/wall fluxes and Lai's distinct nonequilibrium transport are untouched.
ENTRY void bend_lv_face_slope(int nf,int nc,const double*p,const double*q,
 const double*z,const double*W,const double*op,const double*grad,
 const int*owner,const int*nb,const double*normal,const double*distance,
 const double*xy,double*out){
 LOOP(nf){int o=owner[i],j=nb[i];if(j<0)continue;
  double lo=distance[i],lj=distance[nf+i],sum=lo+lj,wo=lj/sum,wj=lo/sum;
  double dx=xy[j]-xy[o],dy=xy[nc+j]-xy[nc+o];
  double gx=wo*grad[4*nc+o]+wj*grad[4*nc+j];
  double gy=wo*grad[5*nc+o]+wj*grad[5*nc+j];
  double residual=(z[j]-z[o]-gx*dx-gy*dy)/(dx*dx+dy*dy);
  gx+=residual*dx;gy+=residual*dy;
  double flux=0.;int cells[2]={o,j};double weights[2]={wo,wj};
  for(int k=0;k<2;++k){int c=cells[k];double cap=op[7*nc+c],w=op[8*nc+c];
   if(cap==0.||w==0.)continue;
   double h=q[c],tx=q[nc+c]/(h*w),ty=q[2*nc+c]/(h*w),ratio=W[c]/(4.*w);
   double x=tx+ratio*ty,y=ty-ratio*tx,r=norm2(x,y);x/=r;y/=r;
   double f=9.*pow(p[3]/h,.3)*sqrt(op[9*nc+c]);
   x-=gx/f;y-=gy/f;r=norm2(x,y);
   if(r==0.||!isfinite(r)){flux=nan("");break;}
   flux+=weights[k]*cap*(x*normal[i]+y*normal[nf+i])/r;
  }
  out[2*nf+i]=flux;
 }
}

ENTRY void gather_hydro(int nc,int nf,const double*q,const double*p,const double*fl,
 const double*fr,const double*aux,const double*length,const double*area,
 const int*off,const int*faces,const int*signs,double*out){
 LOOP(nc){double r[3]={0.,0.,0.};
  for(int k=off[i];k<off[i+1];++k){int f=faces[k],s=signs[k];double a=s*length[f]/area[i];
   const double*flux=s>0?fl:fr;for(int m=0;m<3;++m)r[m]-=a*flux[m*nf+f];
   r[1]-=a*aux[f];r[2]-=a*aux[nf+f];}
  double h=q[i],w=norm2(q[nc+i],q[2*nc+i])/h,drag=p[0]*p[2]*p[2]*w/pow(h,4./3.);
  out[i]=r[0];out[nc+i]=r[1]-drag*q[nc+i];out[2*nc+i]=r[2]-drag*q[2*nc+i];
 }
}

// Isolated LV Eq26 research option. The printed approximation places nu_T
// OUTSIDE the Laplacian; the usual conservative face flux differentiates nu_T
// too. Replace ONLY the interior viscous rate, retaining wall traction and D.
// The optional factor two tests the literal Eq27 separately; it is not an
// inferred physical constant. Scalar mixing uses the unchanged viscosity.
// local_correction is an acceleration of depth-integrated momentum, and its
// volume sum need not vanish for variable nu. Record it, do not call it a
// conservative pair flux or silently make it the production default.
ENTRY void bend_lv_cell_viscosity(int nc,int nf,double factor,const double*q,
 const double*op,const double*grad,const int*off,const int*faces,const int*owner,
 const int*nb,const double*normal,const double*distance,const double*disp,
 const double*length,const double*area,double*correction){
 LOOP(nc){double value[2]={0.,0.};
  for(int a=off[i];a<off[i+1];++a){int f=faces[a],o=owner[f],j=nb[f];if(j<0)continue;
   double dn=distance[f]+distance[nf+f],wo=distance[nf+f]/dn,wj=distance[f]/dn;
   double nx=normal[f],ny=normal[nf+f];
   double dx=disp[f]-disp[2*nf+f],dy=disp[nf+f]-disp[3*nf+f];
   double wet=fmin(q[o],q[j]),nu_face=.5*(op[12*nc+o]+op[12*nc+j]);
   double scale=(factor*op[12*nc+i]-nu_face)*wet*length[f]/area[i]*(i==o?1.:-1.);
   for(int k=0;k<2;++k){
    double gx=wo*grad[2*k*nc+o]+wj*grad[2*k*nc+j];
    double gy=wo*grad[(2*k+1)*nc+o]+wj*grad[(2*k+1)*nc+j];
    double delta=q[(k+1)*nc+j]/q[j]-q[(k+1)*nc+o]/q[o];
    value[k]+=scale*(delta-gx*(dx-dn*nx)-gy*(dy-dn*ny))/dn;
   }
  }
  correction[i]=value[0];correction[nc+i]=value[1];
 }
}

ENTRY void divergence(int nc,int nf,int component,const double*flux,const double*length,
 const double*area,const int*off,const int*faces,const int*signs,double*out){
 LOOP(nc){double r=0.;for(int k=off[i];k<off[i+1];++k)r-=signs[k]*length[faces[k]]*flux[component*nf+faces[k]];
 out[i]=r/area[i];}
}

// Donor limiter is shared per face; never repair negative inventories by clipping cells.
ENTRY void donor_factors(int nc,int nf,int component,double dt,const double*available,
 const double*flux,const double*length,const double*area,const int*off,
 const int*faces,const int*signs,double*out){
 LOOP(nc){double outgoing=0.;for(int k=off[i];k<off[i+1];++k){int f=faces[k];
 outgoing+=fmax(0.,signs[k]*flux[component*nf+f])*length[f];}
 out[i]=outgoing>0.?fmin(1.,fmax(0.,available[i])*area[i]/(dt*outgoing)):1.;}
}
ENTRY void limit_flux(int nf,int component,const int*owner,const int*nb,const double*fac,double*flux){
 LOOP(nf){int j=flux[component*nf+i]>=0.?owner[i]:nb[i];if(j>=0)flux[component*nf+i]*=fac[j];}
}
ENTRY void exchange(int n,double dt,const double*p,const double*op,const double*M,
 const double*z,const double*rock,double*Mnew,double*znew,double*exchanged){
 LOOP(n){double speed=op[8*n+i],Lb=op[14*n+i],lambda=speed/Lb;
 double phi=lambda>0.?-expm1(-dt*lambda)/lambda:dt;
 double delta=phi*(op[7*n+i]/Lb-lambda*M[i]);
 delta=fmin(delta,(1.-p[5])*(z[i]-rock[i]));
 Mnew[i]=M[i]+delta;znew[i]=z[i]-delta/(1.-p[5]);exchanged[i]=delta;}
}
ENTRY void combine(int n,double a,const double*x,double b,const double*y,double*out){
 LOOP(n){out[i]=a*x[i]+b*y[i];}
}
ENTRY void available_bed(int n,double porosity,const double*z,const double*rock,double*out){
 LOOP(n){out[i]=(1.-porosity)*(z[i]-rock[i]);}
}

// LV physical inventory is h*omega = 4*v_ns. Constitutive S=h*v_ns is a
// derived view only. The default transports omega; the explicit Eq10 research
// route transports 4*v_ns. Both use this local fixed-depth source map. Neither
// interpretation is proof of the author's unspecified variable-depth extension.
ENTRY void bend_vorticity_source(int n,double dt,const double*op,const double*old,double*next){
 LOOP(n){double lam=op[6*n+i],phi=lam>0.?-expm1(-dt*lam)/lam:dt;
  next[i]=old[i]*exp(-dt*lam)+4.*op[5*n+i]*phi;}
}

// Optional 2DEF reference viscosity: Viero (2019), Eq4, cited by
// Lazzarin--Viero (2023), section2.6.2. Cs=.2 and CE=.07 are published
// reference values, not fitted LFM parameters. Ae is the actual cell area;
// square/merged BRAIDS cells are not the author's triangular mesh.
// One scalar viscosity feeds BOTH momentum mixing and scalar diffusivity.
ENTRY void bend_viero_eddy(int n,const double*p,const double*q,const double*area,
 const double*grad,double*op){
 LOOP(n){double h=q[i],w=op[8*n+i];
  double us=sqrt(p[0])*p[2]*w/pow(h,1./6.);
  double ux=grad[i],uy=grad[n+i],vx=grad[2*n+i],vy=grad[3*n+i];
  double strain2=2.*ux*ux+(uy+vx)*(uy+vx)+2.*vy*vy;
  double horizontal=2.*.2*.2*area[i],vertical=.07*us*h;
  op[12*n+i]=sqrt(horizontal*horizontal*strain2+vertical*vertical);
 }
}

// Open faces: prescribed incoming solid volume/width and secondary vorticity;
// outgoing sediment uses local capacity/adaptation, never an exterior zero bed.
// Only supplied open faces are overwritten; impermeable walls retain traction.
ENTRY void bend_open_faces(int nf,int nc,int method,const double*p,const double*q,
 const double*W,const double*M,const double*op,const double*water,const int*owner,
 const int*nb,const int*bid,const double*normal,const double*feed,const double*omega_in,
 double*out){
 LOOP(nf){if(nb[i]>=0||bid[i]<0)continue;
  int o=owner[i];double nx=normal[i],ny=normal[nf+i];
  out[i]=op[2*nc+o]*nx+op[3*nc+o]*ny;
  out[nf+i]=op[3*nc+o]*nx+op[4*nc+o]*ny;
  out[3*nf+i]=water[i]*(water[i]>=0.?W[o]/q[o]:omega_in[i]);
  if(water[i]<0.)out[2*nf+i]=-feed[i];
  else if(water[i]>0.){
   double cap=op[7*nc+o],projected=fmax(0.,op[o]*nx+op[nc+o]*ny);
   out[2*nf+i]=method==1?(cap>0.?op[13*nc+o]*op[8*nc+o]*M[o]*projected/cap:0.):projected;
  }else out[2*nf+i]=0.;
 }
}

// Equation 13: turbulent + anisotropic secondary-vorticity diffusion.
// Reconstruct the complete tensor gradient, with a centre-to-centre correction
// for nonorthogonal faces. Coefficients have units m2/s; flux has units m2/s2.
// Mask bits: 1 turbulent, 2 longitudinal shear, 4 transverse shear.
// Partial tensors are explicitly identified research controls, not defaults.
FN void bend_scalar_diffusion_face(int i,int nf,int nc,int components,int surface,double schmidt,const double*p,
 const double*q,const double*W,const double*op,const double*grad,const int*owner,
 const int*nb,const double*normal,const double*xy,double*out,double*conductance){
 conductance[i]=0.;int o=owner[i],j=nb[i];if(j<0)return;
  double Kxx=0.,Kxy=0.,Kyy=0.;int ids[2]={o,j};
  for(int k=0;k<2;++k){int c=ids[k];double h=q[c],w=op[8*nc+c];
   // Turbulent diffusion remains defined at a stagnation point (e.g. nonzero
   // strain-based viscosity). Only the flow-aligned shear tensor needs w>0.
   double tx=w>0.?q[nc+c]/h/w:0.,ty=w>0.?q[2*nc+c]/h/w:0.;
   double us=sqrt(p[0])*p[2]*w/pow(h,1./6.),vns=.25*W[c];
   double ds=5.93*us*h,dn=us>0.?vns*vns*h/(2.4*us):0.;
   double isotropic=op[12*nc+c]/schmidt;
   if(!(components&1))isotropic=0.;
   if(!(components&2))ds=0.;
   if(!(components&4))dn=0.;
   Kxx+=.5*h*(isotropic+ds*tx*tx+dn*ty*ty);
   Kxy+=.5*h*(ds-dn)*tx*ty;
   Kyy+=.5*h*(isotropic+ds*ty*ty+dn*tx*tx);
  }
  double dx=xy[j]-xy[o],dy=xy[nc+j]-xy[nc+o],rr=dx*dx+dy*dy;
  double gx=.5*(grad[6*nc+o]+grad[6*nc+j]);
  double gy=.5*(grad[7*nc+o]+grad[7*nc+j]);
  double jump=surface?W[j]-W[o]:W[j]/q[j]-W[o]/q[o];
  double correction=(jump-gx*dx-gy*dy)/rr;
  gx+=correction*dx;gy+=correction*dy;
  double nx=normal[i],ny=normal[nf+i];
  out[3*nf+i]-=nx*(Kxx*gx+Kxy*gy)+ny*(Kxy*gx+Kyy*gy);
  conductance[i]=(Kxx+Kyy)/sqrt(rr); // conservative diagnostic bound
}

// Preserve the historical ABI and arithmetic for the default vorticity route.
ENTRY void bend_scalar_diffusion(int nf,int nc,int components,double schmidt,const double*p,
 const double*q,const double*W,const double*op,const double*grad,const int*owner,
 const int*nb,const double*normal,const double*xy,double*out,double*conductance){
 LOOP(nf){bend_scalar_diffusion_face(i,nf,nc,components,0,schmidt,p,q,W,op,grad,owner,nb,normal,xy,out,conductance);}
}
ENTRY void bend_surface_velocity_diffusion(int nf,int nc,int components,double schmidt,const double*p,
 const double*q,const double*W,const double*op,const double*grad,const int*owner,
 const int*nb,const double*normal,const double*xy,double*out,double*conductance){
 LOOP(nf){bend_scalar_diffusion_face(i,nf,nc,components,1,schmidt,p,q,W,op,grad,owner,nb,normal,xy,out,conductance);}
}

// Eq10 contrast: advect 4*v_ns with the accepted water flux. Only the secondary
// component is overwritten; momentum and sediment face operators are unchanged.
ENTRY void bend_surface_velocity_faces(int nf,int nc,const double*q,const double*W,
 const double*grad,const double*water,const int*owner,const int*nb,const int*bid,
 const double*displacement,const double*omega_in,double*out){
 LOOP(nf){int o=owner[i],j=nb[i];double value=0.;
  if(j>=0){int up=water[i]>=0.?o:j;
   double ds=grad[6*nc+up]*displacement[(up==o?0:2)*nf+i]
            +grad[7*nc+up]*displacement[(up==o?1:3)*nf+i];
   value=W[up]+mm(ds,up==o?W[j]-W[o]:W[o]-W[j]);
  }else if(bid[i]>=0)value=water[i]>=0.?W[o]:q[o]*omega_in[i];
  out[3*nf+i]=water[i]*value;
 }
}

// Add auxiliary momentum only: no duplicate hydraulic flux or Manning drag.
ENTRY void bend_add_momentum(int nc,int nf,const double*aux,const double*length,
 const double*area,const int*off,const int*faces,const int*signs,double*rate){
 LOOP(nc){double rx=0.,ry=0.;for(int k=off[i];k<off[i+1];++k){int f=faces[k];
  double a=signs[k]*length[f]/area[i];rx-=a*aux[f];ry-=a*aux[nf+f];}
  rate[nc+i]+=rx;rate[2*nc+i]+=ry;}
}

ENTRY void bend_diffusion_metric(int nc,double dt,const double*h,const double*conductance,
 const double*length,const double*area,const int*off,const int*faces,double*metric){
 LOOP(nc){double sum=0.;for(int k=off[i];k<off[i+1];++k){int f=faces[k];sum+=length[f]*conductance[f];}
  metric[i]=dt*sum/(area[i]*h[i]);}
}

// Lai (2020), visually checked Eqs10a-c,11a,15. Fields13/14 are beta/Lb.
// Eq11 uses Vt*M, NOT beta*Vt*M. Thus uniform equilibrium through-flux is
// beta*q_capacity; incoming feed must state which quantity it prescribes.
// No invented threshold floor or extrapolation: status6=Phi,7=Z,8=Lb domain.
ENTRY void bend_lai_adaptation(int n,int velocity_mode,int length_mode,double ws,
 const double*p,const double*q,double*op,int*status){
 LOOP(n){
  op[13*n+i]=p[12];op[14*n+i]=p[11];if(status[i])continue;
  double w=op[8*n+i],theta=op[9*n+i],phi=theta/.045;
  if(velocity_mode){
   if(phi>=20.){status[i]=6;continue;}
   double ratio=sqrt(p[0])*p[2]/pow(q[i],1./6.);
   double beta=w>0.?ratio*1.1*pow(phi,.17)*(-expm1(-5.*phi))/sqrt(.045):0.;
   if(velocity_mode==2){
    double Z=w>0.?ws/(.41*ratio*w):1.;
    if(Z>=1.){status[i]=7;continue;}
    double bs=1.+ratio/(2.*.41)*(-expm1(2.7*Z));
    beta=fmax(beta,bs);
   }
   op[13*n+i]=beta;
  }
  if(length_mode){
   if(theta<=.045){status[i]=8;continue;}
   op[14*n+i]=4000.*(theta-.045)*p[3];
  }
 }
}

// CPU controller uses the same split drag map and low-Froude/advective bounds
// as the GPU midpoint. Independent cell gathers avoid atomics/nested teams.
ENTRY void bend_manning(int n,double dt,double chezy_min,const double*p,double*q){
 LOOP(n){double h=q[i],speed=norm2(q[n+i],q[2*n+i])/h;
  double den=chezy_min>0.&&h<chezy_min?h*pow(chezy_min,1./3.):pow(h,4./3.);
  double factor=1.+dt*p[0]*p[2]*p[2]*speed/den;
  q[n+i]/=factor;q[2*n+i]/=factor;}
}
ENTRY void bend_hydraulic_envelope(int n,int nf,double dt,const double*p,const double*q,
 const int*off,const int*faces,const int*owner,const int*nb,const double*normal,
 const double*length,const double*area,double*metric,int*status){
 LOOP(n){
  status[i]=0;metric[i]=0.;double h=q[i],u=q[n+i]/h,v=q[2*n+i]/h;
  if(!isfinite(h)||h<=20.*p[1]||!isfinite(u)||!isfinite(v)){status[i]=1;continue;}
  if(norm2(u,v)/sqrt(p[0]*h)>.65){status[i]=2;continue;}
  double sum=0.;
  for(int k=off[i];k<off[i+1];++k){int f=faces[k],j=owner[f]==i?nb[f]:owner[f];
   double nx=normal[f],ny=normal[nf+f],s=fabs(u*nx+v*ny);
   if(j>=0)s=fmax(s,fabs((q[n+j]*nx+q[2*n+j]*ny)/q[j]));
   sum+=length[f]*s;}
  metric[i]=dt*sum/area[i];if(metric[i]>.8)status[i]=3;
 }
}

// CPU cell gather of the GPU midpoint's frozen acoustic LINEARIZATION.
// It is a preconditioner/iteration matrix only; the nonlinear residual remains
// the complete reconstructed augmented f-wave + live bend momentum operator.
ENTRY void bend_linear_acoustic(int n,int nf,double grav,const double*href,const double*x,
 const int*off,const int*faces,const int*signs,const int*owner,const int*nb,const int*bid,
 const double*normal,const double*length,const double*area,double*out){
 LOOP(n){double r0=0.,gx=0.,gy=0.,dx=0.,dy=0.;
  for(int k=off[i];k<off[i+1];++k){int f=faces[k],o=owner[f],j=nb[f],s=signs[k];
   double nx=normal[f],ny=normal[nf+f],ell=length[f];
   if(j<0){if(bid[f]>=0)continue;
    double c=sqrt(grav*href[i]),mn=x[n+i]*nx+x[2*n+i]*ny;
    dx-=c*ell*mn*nx;dy-=c*ell*mn*ny;continue;}
   double c=sqrt(grav*.5*(href[o]+href[j])),dh=x[j]-x[o];
   double dmn=(x[n+j]-x[n+o])*nx+(x[2*n+j]-x[2*n+o])*ny;
   double discharge=.5*((x[n+o]+x[n+j])*nx+(x[2*n+o]+x[2*n+j])*ny)-.5*c*dh;
   r0-=s*ell*discharge;gx+=.5*ell*dh*nx;gy+=.5*ell*dh*ny;
   dx+=s*.5*c*ell*dmn*nx;dy+=s*.5*c*ell*dmn*ny;
  }
  out[i]=r0/area[i];out[n+i]=(-grav*href[i]*gx+dx)/area[i];
  out[2*n+i]=(-grav*href[i]*gy+dy)/area[i];
 }
}

