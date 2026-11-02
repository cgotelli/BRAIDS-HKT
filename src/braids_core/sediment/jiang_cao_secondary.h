// Research secondary.de_vriend1977_jc2026_cartesian198_v1.
// Positive curvature = turn LEFT; near-bed drift LEFT; surface secondary RIGHT.
#ifdef __CUDACC__
#define JC_ENTRY extern "C" __global__
#define JC_LOOP(N) for(int i=blockDim.x*blockIdx.x+threadIdx.x;i<N;i+=blockDim.x*gridDim.x)
#else
#include <cmath>
#include <omp.h>
#define JC_ENTRY extern "C" __declspec(dllexport)
#define JC_LOOP(N) _Pragma("omp parallel for schedule(static)") for(int i=0;i<N;++i)
extern "C" __declspec(dllexport) void mobile_threads(int n){omp_set_dynamic(0);omp_set_num_threads(n);}
#endif

// mom = <L*g0,g1,g2>, then row-major <gj*gk>; W=g0+a*g1+a^2*g2.
// out: k, h*k, a, Dxx,Dxy,Dyy, tan(phi), relative qx,qy.
JC_ENTRY void jc_secondary(int n,double gravity,int active,const double*q,
 const double*gradient,const double*manning,const double*cap,const double*mom,double*out){
 JC_LOOP(n){
  double h=q[i],u=q[n+i]/h,v=q[2*n+i]/h,U=hypot(u,v);
  double a=sqrt(gravity)*manning[i]/(.4*pow(h,1./6.));
  double k=0.,Dxx=0.,Dxy=0.,Dyy=0.;
  if(U>0. && active){
   double tx=u/U,ty=v/U;
   k=(tx*tx*gradient[2*n+i]+tx*ty*(gradient[3*n+i]-gradient[i])-ty*ty*gradient[n+i])/U;
   double A=U*h*k/.16,p[3]={1.,a,a*a},lw=0.,ww=0.;
   for(int j=0;j<3;++j){lw+=p[j]*mom[j];
    for(int l=0;l<3;++l)ww+=p[j]*p[l]*mom[3+3*j+l];}
   double ss=U*U*a*a,sn=-U*A*a*lw,nn=A*A*ww;
   Dxx=ss*tx*tx-2.*sn*tx*ty+nn*ty*ty;
   Dxy=(ss-nn)*tx*ty+sn*(tx*tx-ty*ty);
   Dyy=ss*ty*ty+2.*sn*tx*ty+nn*tx*tx;
  }
  double theta=cap[3*n+i],tc=cap[4*n+i];
  double r=7.*h*k+(theta>0.?-1.5*cap[8*n+i]*sqrt(tc/theta):0.);
  double norm=hypot(1.,r),mx=q[n+i],my=q[2*n+i];
  out[i]=k;out[n+i]=h*k;out[2*n+i]=a;
  out[3*n+i]=Dxx;out[4*n+i]=Dxy;out[5*n+i]=Dyy;out[6*n+i]=r;
  out[7*n+i]=(mx-r*my)/norm-mx;out[8*n+i]=(my+r*mx)/norm-my;
 }
}

// Conservative covariance flux h D.n at true face midpoints.
// Solid walls: velocity normal to wall is zero at every height => D.n=0.
// Open boundaries: one-sided extrapolated tensor; never reuse a wall ghost.
JC_ENTRY void jc_dispersion_faces(int n,int nf,const int*owner,const int*neighbour,
 const int*kind,const double*length,const double*normal,const double*xy,
 const double*mid,const double*tensor,const double*grad,double*flux){
 JC_LOOP(nf){
  if(kind[i]==1){flux[i]=0.;flux[nf+i]=0.;continue;}
  int o=owner[i],j=neighbour[i];double t[3];
  for(int c=0;c<3;++c){
   double vo=tensor[c*n+o]+grad[2*c*n+o]*(mid[i]-xy[o])+grad[(2*c+1)*n+o]*(mid[nf+i]-xy[n+o]);
   double vj=vo;
   if(j>=0)vj=tensor[c*n+j]+grad[2*c*n+j]*(mid[i]-xy[j])+grad[(2*c+1)*n+j]*(mid[nf+i]-xy[n+j]);
   t[c]=.5*(vo+vj);
  }
  double nx=normal[i],ny=normal[nf+i];
  flux[i]=length[i]*(t[0]*nx+t[1]*ny);
  flux[nf+i]=length[i]*(t[1]*nx+t[2]*ny);
 }
}

JC_ENTRY void jc_dispersion_divergence(int n,int nf,const int*offset,const int*faces,
 const int*owner,const double*area,const double*flux,double*rate){
 JC_LOOP(n){double rx=0.,ry=0.;for(int k=offset[i];k<offset[i+1];++k){
   int f=faces[k];double sign=owner[f]==i?1.:-1.;rx-=sign*flux[f];ry-=sign*flux[nf+f];}
  rate[n+i]+=rx/area[i];rate[2*n+i]+=ry/area[i];}
}
