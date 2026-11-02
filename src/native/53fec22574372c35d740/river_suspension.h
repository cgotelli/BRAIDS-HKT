// Shared OpenMP/CUDA current-only general split transport and mass exchange.
#ifdef __CUDACC__
#define ENTRY extern "C" __global__
#define LOOP(N) for(int i=blockDim.x*blockIdx.x+threadIdx.x;i<N;i+=blockDim.x*gridDim.x)
#else
#include <cmath>
#include <omp.h>
using std::isfinite;
#define ENTRY extern "C" __declspec(dllexport)
#define LOOP(N) _Pragma("omp parallel for schedule(static)") for(int i=0;i<N;++i)
extern "C" __declspec(dllexport) void mobile_threads(int n){omp_set_dynamic(0);omp_set_num_threads(n);}
#endif
ENTRY void split_capacity(int n,double d,double bed_factor,const double*q,const double*cq,const double*c,
 const double*hi,double*out,int*status){
 LOOP(n){bool valid=isfinite(c[i])&&isfinite(hi[i])&&c[i]>=1.e-6&&q[i]>0.&&cq[i]>0.;
  for(int k=0;k<3;++k)valid=valid&&isfinite(q[k*n+i])&&isfinite(cq[k*n+i]);
  status[i]=valid?0:1;if(!valid)continue;
  double h=q[i],u=q[n+i]/h,v=q[2*n+i]/h,ue=cq[n+i]/cq[i],ve=cq[2*n+i]/cq[i];
  double speed=hypot(ue,ve),theta=speed*speed/(c[i]*c[i]*1.65*d),ex=fmax(0.,theta-.047);
  double mag=bed_factor*2650.*8.*sqrt(9.81*1.65*d)*d*ex*sqrt(ex);
  double us=hypot(u,v),alpha=us>0.&&mag>0.?12.5*(1.-sqrt(9.81)/(.4*c[i]))*(hi[i]/h)/us:0.;
  double bx=u+alpha*v,by=v-alpha*u,bs=hypot(bx,by);
  out[i]=speed>0.?mag*ue/speed:0.;out[n+i]=speed>0.?mag*ve/speed:0.;out[2*n+i]=theta;
  out[3*n+i]=bs>0.?mag*bx/bs:0.;out[4*n+i]=bs>0.?mag*by/bs:0.;out[5*n+i]=alpha;
 }
}
ENTRY void sus_faces(int nf,int n,int ng,double diff,double dry,const int*o,const int*j,const int*bid,
 const double*length,const double*distance,const double*h,const double*mass,const double*water,
 const double*bc,double*flux){
 LOOP(nf){int a=o[i],b=j[i];bool wall=b<0&&bid[i]<0;
  for(int k=0;k<ng;++k){double ca=h[a]>dry?mass[k*n+a]/h[a]:0.;
   double cb=b>=0?(h[b]>dry?mass[k*n+b]/h[b]:0.):bc[k*nf+i];
   double f=wall?0.:water[i]*(water[i]>=0.?ca:cb);
   // Diffusion uses only true internal wet faces. Open boundary is prescribed
   // advective inflow / zero diffusive gradient outflow, not a hidden reservoir.
   if(b>=0&&h[a]>dry&&h[b]>dry)f+=diff*.5*(h[a]+h[b])*(ca-cb)/(distance[i]+distance[nf+i]);
   flux[k*nf+i]=f;
  }
 }
}
ENTRY void sus_limit(int n,int nf,int ng,double dt,const int*offset,const int*faces,const int*o,
 const double*length,const double*area,const double*mass,const double*flux,double*scale){
 LOOP(n){for(int k=0;k<ng;++k){double loss=0.;for(int e=offset[i];e<offset[i+1];++e){int f=faces[e];
    double signed_flux=(o[f]==i?1.:-1.)*flux[k*nf+f];loss+=fmax(0.,signed_flux)*length[f];}
   scale[k*n+i]=loss>0.?fmin(1.,mass[k*n+i]*area[i]/(dt*loss)):1.;}}
}
// Wet-neighbour weighted LS concentration gradients. Recompute the small
// metric when wetness changes; dry concentrations and walls are not samples.
// Singular one-dimensional stencils retain their resolved directional slope.
ENTRY void sus_gradients(int n,int ng,double dry,const int*offset,const int*faces,
 const int*o,const int*j,const double*xy,const double*h,const double*mass,double*grad){
 LOOP(n){for(int k=0;k<ng;++k){double aa=0.,ab=0.,bb=0.,rx=0.,ry=0.;
  if(h[i]>dry){double ci=mass[k*n+i]/h[i];
   for(int e=offset[i];e<offset[i+1];++e){int f=faces[e],b=o[f]==i?j[f]:o[f];
    if(b<0||h[b]<=dry)continue;
    double dx=xy[b]-xy[i],dy=xy[n+b]-xy[n+i],r2=dx*dx+dy*dy;
    if(r2<=0.)continue;
    double dc=mass[k*n+b]/h[b]-ci;
    aa+=dx*dx/r2;ab+=dx*dy/r2;bb+=dy*dy/r2;rx+=dx*dc/r2;ry+=dy*dc/r2;
   }
  }
  double det=aa*bb-ab*ab,tr=aa+bb,gx=0.,gy=0.;
  if(det>1.e-12*tr*tr){gx=(bb*rx-ab*ry)/det;gy=(aa*ry-ab*rx)/det;}
  else if(tr>0.){gx=rx/tr;gy=ry/tr;}
  grad[(2*k)*n+i]=gx;grad[(2*k+1)*n+i]=gy;
 }}
}
// Correct the base two-point flux for tangential centroid displacement.
// There is still only one shared face flux and one conservative donor limiter.
// For orthogonal centres this is exactly zero; boundaries remain impermeable
// or zero diffusive flux. This is not smoothing of the hydraulic solution.
ENTRY void sus_nonorthogonal(int nf,int n,int ng,double diff,double dry,const int*o,const int*j,
 const double*xy,const double*normal,const double*distance,const double*h,const double*grad,double*flux){
 LOOP(nf){int a=o[i],b=j[i];if(b<0||h[a]<=dry||h[b]<=dry)continue;
  double dn=distance[i]+distance[nf+i];
  double tx=xy[b]-xy[a]-dn*normal[i],ty=xy[n+b]-xy[n+a]-dn*normal[nf+i];
  for(int k=0;k<ng;++k){double gx=.5*(grad[2*k*n+a]+grad[2*k*n+b]);
   double gy=.5*(grad[(2*k+1)*n+a]+grad[(2*k+1)*n+b]);
   flux[k*nf+i]+=diff*.5*(h[a]+h[b])*(gx*tx+gy*ty)/dn;
  }
 }
}
ENTRY void sus_scale(int nf,int n,int ng,const int*o,const int*j,const double*scale,double*flux){
 LOOP(nf){for(int k=0;k<ng;++k){int donor=flux[k*nf+i]>=0.?o[i]:j[i];
  if(donor>=0)flux[k*nf+i]*=scale[k*n+donor];}}
}
ENTRY void sus_gather(int n,int nf,int ng,double dt,const int*offset,const int*faces,const int*o,
 const double*length,const double*area,const double*mass,const double*flux,double*out){
 LOOP(n){for(int k=0;k<ng;++k){double sum=0.;for(int e=offset[i];e<offset[i+1];++e){int f=faces[e];
  sum+=(o[f]==i?-1.:1.)*flux[k*nf+f]*length[f];}out[k*n+i]=mass[k*n+i]+dt*sum/area[i];}}
}
// Same predicates as the original separate NumPy/CuPy scans. Cell-parallel,
// with one writer per status entry and no changes to exchange arithmetic.
ENTRY void sus_check(int n,int ng,const double*next,const double*available,
 const double*delta,int*status){
 LOOP(n){int bad=0;for(int k=0;k<ng;++k){int a=k*n+i;
  if(!isfinite(next[a])||next[a]<0.||available[a]+delta[a]<-1e-10)bad=1;}
  status[i]=bad;
 }
}
ENTRY void sus_exchange(int n,int ng,double dt,double mf,double dry,double sedthr,double cref,double temp,
 double sus_factor,const double*diam,const double*h0,const double*q,const double*cq,const double*c,const double*z0,
 const double*mass,const double*stock,const double*frac,double*next,double*delta,double*diagnostic){
 LOOP(n){double h=q[i],hs=fmax(.01,h),speed=h>dry?hypot(q[n+i],q[2*n+i])/h:0.;
  double ve=h>dry?hypot(cq[n+i],cq[2*n+i])/cq[i]:0.;
  // erosed: USTAR = UMOD*kappa/log(1+ZUMOD/z0rou). Capacity Chezy is
  // separately supplied and must not be substituted in this shear. erosed's
  // 1e-8 denominator floor belongs to Chezy, not to USTAR's z0 argument.
  double shear=h>dry?speed*.41/log1p(h*double(float(.368))/z0[i]):0.;
  double total=0.;for(int k=0;k<ng;++k)total+=h>dry?mass[k*n+i]/h:0.;
  double hinder=pow(fmax(0.,1.-total/cref),5.),nu=4.e-5/(20.+temp);
  for(int k=0;k<ng;++k){int a=k*n+i;double d=diam[k],ws;
   if(d<1.5*64.e-6)ws=1.65*9.81*d*d/(18.*nu);
   else if(d<.001){double coef=d<128.e-6?(-2.9912/64.e-6)*d+15.9824:10.;
    ws=coef*nu/d*(sqrt(1.+1.65*9.81*d*d*d/(100.*nu*nu))-1.);}
   else ws=1.1*sqrt(1.65*9.81*d);
   ws*=hinder;double wsl=fmax(.001,ws),theta=ve*ve/(c[i]*c[i]*1.65*d);
   double cap=sus_factor*2650.*2.5*d*sqrt(9.81*1.65*d)*theta*theta*fmax(.9*theta-.035,0.);
   double ceq=cap/(ve+1.e-6)/hs*frac[a],hots=0.,rate=0.,source=0.;
   if(h>sedthr){if(speed>1.e-6&&shear>1.e-6){double u=fmin(shear/speed,.15),x=fmin(wsl/shear,1.);
      double ts=x*exp(((double)1.547f-(double)20.12f*u)*x*x*x+
       ((double)326.832f*pow(u,(double)2.2047f)-(double).2f)*x*x+
       ((double).1385f*log(u)-(double)6.4061f)*x+((double).5467f*u+(double)2.1963f));
      hots=wsl/ts;source=ceq*hots*h0[i]/fmax(.01,h0[i]);rate=hots/hs;
    }else rate=wsl/hs;}
   // Evaluate the increment directly. Forming (mass + available/MF) and
   // subtracting mass again can overdraw a nearly exhausted bed by round-off.
   // A relative arithmetic reserve also protects division/multiplication by
   // MF. It is not a physical erosion cutoff or a relaxed inventory check.
   const double reserve=1.-16.*2.22044604925031308085e-16;
   double change=dt*(source-rate*mass[a])/(1.+dt*rate);
   change=fmax(-mass[a],fmin(change,fmax(0.,stock[a])/mf*reserve));
   delta[a]=-mf*change;next[a]=mass[a]+change;
   // A truly dry cell deposits its stranded scalar ONCE on physical time.
   // This transfer is not multiplied by MF or confused with entrainment.
   if(h<=dry){delta[a]=mass[a];next[a]=0.;}
   diagnostic[a]=ws;diagnostic[n*ng+a]=ceq;diagnostic[2*n*ng+a]=hots;
  }
 }
}
