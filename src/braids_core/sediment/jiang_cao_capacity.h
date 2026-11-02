// Isolated Jiang-Cao capacity reconstruction, exponent 3/2 chosen by user.
// Same double-precision constitutive arithmetic for OpenMP and CUDA.
// No Exner update, mobile inventory, secondary flow, collapse or public binding.
#ifdef __CUDACC__
#define JC_ENTRY extern "C" __global__
#define JC_LOCAL __device__ __forceinline__
#define JC_LOOP(N) for(int i=blockDim.x*blockIdx.x+threadIdx.x;i<N;i+=blockDim.x*gridDim.x)
#else
#include <cmath>
#include <omp.h>
using std::isfinite;
#define JC_ENTRY extern "C" __declspec(dllexport)
#define JC_LOCAL static inline
#define JC_LOOP(N) _Pragma("omp parallel for schedule(static)") for(int i=0;i<N;++i)
extern "C" __declspec(dllexport) void mobile_threads(int n){omp_set_dynamic(0);omp_set_num_threads(n);}
#endif

JC_LOCAL void jc_invalid(int n,int i,double*out,int*status,int code){
 status[i]=code;
 for(int k=0;k<23;++k)out[k*n+i]=nan("");
}

JC_ENTRY void jc_capacity(int n,const double*p,const double*q,const double*gradient,
 const double*manning,const double*concentration,double*out,int*status){
 JC_LOOP(n){
  for(int k=0;k<23;++k)out[k*n+i]=0.;
  status[i]=0;
  const double h=q[i],mx=q[n+i],my=q[2*n+i];
  const double gx=gradient[i],gy=gradient[n+i],rough=manning[i],c=concentration[i];
  if(!isfinite(h)||!isfinite(mx)||!isfinite(my)||!isfinite(gx)||!isfinite(gy)
     ||!isfinite(rough)||!isfinite(c)||h<0.||rough<=0.||c<0.||c>=1.){
   jc_invalid(n,i,out,status,3);continue;
  }
  // Dry hydraulic cells carry no bedload. Their actual momentum is not divided
  // by a surrogate wet depth. Collapse must have its own dry-bank owner.
  if(h<=p[5]){status[i]=1;continue;}
  const double tangent=p[6],slope=hypot(gx,gy);
  if(!isfinite(slope)){jc_invalid(n,i,out,status,5);continue;}
  // Above-repose terrain requires redistribution. Never clamp a negative
  // radicand, invent a positive threshold, or quietly accept a zero capacity.
  if(slope>=tangent){jc_invalid(n,i,out,status,4);continue;}
  const double speed=hypot(mx,my)/h;
  if(!isfinite(speed)){jc_invalid(n,i,out,status,5);continue;}
  const double hthird=cbrt(h),chezy=sqrt(hthird)/rough;
  out[9*n+i]=chezy;
  if(!isfinite(chezy)||chezy<=0.){jc_invalid(n,i,out,status,5);continue;}
  if(speed==0.){status[i]=2;continue;}
  const double ex=(mx/h)/speed,ey=(my/h)/speed;
  // e=(ex,ey) is streamwise; left normal=(-ey,ex).
  // a,b are UPHILL bed gradients. Downhill transverse steering is -b.
  const double a=gx*ex+gy*ey,b=-gx*ey+gy*ex;
  const double da=hypot(1.,a),db=hypot(1.,b),t=b/tangent;
  const double root=sqrt((1.-t)*(1.+t));
  const double ks=(1.+a/tangent)/da,kn=root/db;
  const double critical=.047*ks*kn;
  const double theta=rough*rough*speed*speed/(hthird*p[1]*p[0]);
  const double ustar2=p[4]*p[1]*p[0]*theta;
  const double rho=p[2]*c+p[3]*(1.-c);
  const double tau=rho*ustar2;
  if(!isfinite(chezy)||!isfinite(theta)||!isfinite(tau)||!isfinite(critical)
      ||chezy<=0.||critical<=0.||theta<0.){
   jc_invalid(n,i,out,status,5);continue;
  }
  out[3*n+i]=theta;out[4*n+i]=critical;
  out[5*n+i]=ks;out[6*n+i]=kn;out[7*n+i]=a;out[8*n+i]=b;
  out[10*n+i]=ustar2;out[11*n+i]=tau;
  const double excess=theta-critical;
  if(excess<=0.)continue;
  const double factor=1.5*p[7]*sqrt(excess),qb=p[7]*excess*sqrt(excess);
  // Heading derivatives of the slope threshold, at fixed |U| and depth.
  const double dks=(1./tangent-a)/(da*da*da);
  const double dkn=-b/(tangent*tangent*root*db)-b*root/(db*db*db);
  const double critical_psi=.047*(dks*b*kn-ks*dkn*a);
  // Full scalar Jacobian in (h,hu,hv), fixed n and Cartesian bed gradient.
  const double dh=-factor*(7./3.)*theta/h;
  const double dmx=factor*(2.*theta*ex+critical_psi*ey)/(h*speed);
  const double dmy=factor*(2.*theta*ey-critical_psi*ex)/(h*speed);
  // Eq.12 gravitational term only. A normalized rotation preserves the Eq.9
  // capacity magnitude. Curvature/de Vriend feedback is NOT implemented here.
  const double ratio=sqrt(critical/theta),r=-1.5*b*ratio;
  const double norm=hypot(1.,r),dx=(ex-r*ey)/norm,dy=(ey+r*ex)/norm;
  const double rpsi=-1.5*ratio*(-a+.5*b*critical_psi/critical);
  const double invangle=1./(1.+r*r);
  const double gamma_h=(7./6.)*r/h*invangle;
  const double gamma_mx=(-ey+(-rpsi*ey-r*ex)*invangle)/(h*speed);
  const double gamma_my=( ex+( rpsi*ex-r*ey)*invangle)/(h*speed);
  out[i]=qb;out[n+i]=qb*dx;out[2*n+i]=qb*dy;
  out[12*n+i]=factor*2.*theta/speed;
  out[13*n+i]=dh;out[14*n+i]=dmx;out[15*n+i]=dmy;
  out[16*n+i]=dh*dx-qb*dy*gamma_h;
  out[17*n+i]=dmx*dx-qb*dy*gamma_mx;
  out[18*n+i]=dmy*dx-qb*dy*gamma_my;
  out[19*n+i]=dh*dy+qb*dx*gamma_h;
  out[20*n+i]=dmx*dy+qb*dx*gamma_mx;
  out[21*n+i]=dmy*dy+qb*dx*gamma_my;
  out[22*n+i]=atan(r);
  for(int k=0;k<23;++k)if(!isfinite(out[k*n+i])){
   jc_invalid(n,i,out,status,5);break;
  }
 }
}
