// GP1991 Eqs 26b,37,38. Auxiliary depth-average assembly is BRAIDS S4,
// not a claim of a complete GP1991 solver. Shared CPU/OpenMP and CUDA.
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
LOCAL double gp_settle(double d,double r,double nu){
 if(d<96.e-6)return r*9.81*d*d/(18.*nu);
 if(d<.001){double k=d<128.e-6?(-2.9912/64.e-6)*d+15.9824:10.;
  double t=r*9.81*d*d*d/(100.*nu*nu);
  return k*nu/d*t/(sqrt(1.+t)+1.);}
 return 1.1*sqrt(r*9.81*d);
}
LOCAL double gp_profile(double p,const double*quad){
 // y=z/h; t=logit(y)-logit(.05). Integrate from reference to surface.
 // Constant c=ca below a, Rouse above: mean/reference ratio in [.05,1].
 double top=43./(1.+p),sum=0.;
 for(int j=0;j<48;++j){double t=top*quad[j],b=19.*exp(-t);
  sum+=quad[48+j]*exp(-p*t)*b/((1.+b)*(1.+b));}
 return fmin(1.,fmax(.05,.05+top*sum));
}
LOCAL void gp_values(double h,double speed,double z0,double d,double r,double temp,
 const double*quad,double*v){
 double nu=4.e-5/(20.+temp),ws=gp_settle(d,r,nu);
 // Total local hydraulic shear uses the physical log resistance, not the
 // capacity floor. GP's profile kappa=.4 is distinct from hydraulic kappa=.41.
 double ut=fabs(speed)*.41/log1p(.36800000071525574*h/z0);
 double theta=ut*ut/(r*9.81*d),skin=fmin(theta,.06+.4*theta*theta);
 double us=sqrt(skin*r*9.81*d),rep=sqrt(r*9.81*d)*d/nu;
 double es=0.;
 if(us>0.){double loga=log(1.3e-7/.3)+5.*(log(us/ws)+.6*log(rep));
  es=loga>=0.?.3/(1.+exp(-loga)):.3*exp(loga)/(1.+exp(loga));}
 double p=ut>0.?ws/(.4*ut):1.e100;
 double j=gp_profile(p,quad);
 if(h<20.*d){es=0.;j=1.;} // named shallow extension, not the published law
 v[0]=ws;v[1]=es;v[2]=j;v[3]=ut;v[4]=us;v[5]=rep;v[6]=p;
}
ENTRY void gp_probe(int n,double d,double r,double temp,const double*h,const double*speed,
 const double*z0,const double*quad,double*out){
 LOOP(n){double v[7];gp_values(h[i],speed[i],z0[i],d,r,temp,quad,v);
  for(int k=0;k<7;++k)out[k*n+i]=v[k];}
}
ENTRY void gp_exchange(int n,double dt,double dry,double d,double rho,double r,double temp,
 const double*q,const double*z0,const double*mass,const double*stock,const double*quad,
 double*next,double*delta,double*diag){
 LOOP(n){double h=q[i];
  if(h<=dry){next[i]=0.;delta[i]=mass[i];diag[i]=diag[n+i]=diag[2*n+i]=0.;continue;}
  double v[7];gp_values(h,hypot(q[n+i],q[2*n+i])/h,z0[i],d,r,temp,quad,v);
  double rate=v[0]/(h*v[2]),equilibrium=rho*v[1]*h*v[2];
  double change=(equilibrium-mass[i])*(-expm1(-dt*rate));
  // Keep the bounded transfer itself (not a subtraction of rounded totals).
  // Same arithmetic reserve as the S2 finite-stock exchange, not an erosion cutoff.
  change=fmax(-mass[i],fmin(change,fmax(stock[i],0.)*(1.-16.*2.22044604925031308085e-16)));
  next[i]=mass[i]+change;delta[i]=-change;
  diag[i]=v[0];diag[n+i]=rho*v[1]*v[2];diag[2*n+i]=rate;
 }
}
