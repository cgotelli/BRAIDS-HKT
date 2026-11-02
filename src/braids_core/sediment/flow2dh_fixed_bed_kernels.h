// Shared OpenMP/CUDA Cartesian fixed-bed coupling, classic FLOW secondary I.
#ifdef __CUDACC__
#define ENTRY extern "C" __global__
#define LOOP(N) for(int i=blockDim.x*blockIdx.x+threadIdx.x;i<N;i+=blockDim.x*gridDim.x)
#else
#include <cmath>
#include <omp.h>
#define ENTRY extern "C" __declspec(dllexport)
#define LOOP(N) _Pragma("omp parallel for schedule(static)") for(int i=0;i<N;++i)
extern "C" __declspec(dllexport) void flow_threads(int n){omp_set_dynamic(0);omp_set_num_threads(n);}
#endif

ENTRY void flow_inputs(int n,double dx,double chezy,const double*q,const double*hold,
 const double*hi,const double*grad,double*x){
 LOOP(n){double h=q[i],u=q[n+i]/h,v=q[2*n+i]/h,w=sqrt(u*u+v*v);
  x[i]=hold[i];x[n+i]=h;x[2*n+i]=u;x[3*n+i]=v;
  // FLOW curvat: positive clockwise curvature, not channel centreline radius.
  x[4*n+i]=w>1.e-6?(v*(u*grad[i]+v*grad[n+i])-u*(u*grad[2*n+i]+v*grad[3*n+i]))/(w*w*w):0.;
  x[5*n+i]=2*dx;x[6*n+i]=2*dx;x[7*n+i]=0.;x[8*n+i]=chezy;
  x[9*n+i]=hi[i]/h;x[10*n+i]=0.;x[11*n+i]=0.;
 }
}

ENTRY void flow_drag(int n,double dt,double gravity,double chezy,double*q){
 LOOP(n){double h=q[i],u=q[n+i]/h,v=q[2*n+i]/h;
  double factor=1./(1.+dt*gravity*sqrt(u*u+v*v)/(chezy*chezy*h));
  q[n+i]*=factor;q[2*n+i]*=factor;
 }
}

// Cartesian tensor from FLOW manual (9.151--153):
// Txx=-2 beta u v, Txy=Tyx=beta(u*u-v*v), Tyy=+2 beta u v.
// The local operator returns +2 beta u v as a BASIS, not Cartesian Txx.
// Do not transplant directional vihsec sweep signs into a global tensor.
// Keep vihsec's velocity-acceleration form; gather multiplies by accepted h.
// FLOW vihsec evaluates rxy at lateral boundaries when either adjacent
// velocity pair is active; it does NOT impose zero secondary shear there.
// Extrapolate owner stress at the boundary. Viscous flux remains free-slip.
// This reproduces a stock-model limitation noted by Nabi et al.; it is not
// evidence of a physically resolved sidewall secondary cell.
ENTRY void flow_momentum_faces(int nf,int n,double nu,const double*q,const double*txx,
 const double*txy,const int*owner,const int*nb,const double*normal,const double*dist,double*out){
 LOOP(nf){int o=owner[i],j=nb[i];out[i]=0.;out[nf+i]=0.;
  if(j<0){out[i]=-txx[o]*normal[i]+txy[o]*normal[nf+i];
          out[nf+i]=txy[o]*normal[i]+txx[o]*normal[nf+i];continue;}
  double d=dist[i]+dist[nf+i],wo=dist[nf+i]/d,wj=dist[i]/d;
  double a=wo*txx[o]+wj*txx[j],b=wo*txy[o]+wj*txy[j];
  double nx=normal[i],ny=normal[nf+i];
  out[i]=-a*nx+b*ny+nu*(q[n+j]/q[j]-q[n+o]/q[o])/d;
  out[nf+i]=b*nx+a*ny+nu*(q[2*n+j]/q[j]-q[2*n+o]/q[o])/d;
 }
}

ENTRY void flow_momentum_gather(int n,int nf,const double*h,const double*flux,
 const int*off,const int*faces,const int*sign,const double*length,const double*area,double*rate){
 LOOP(n){double x=0.,y=0.;for(int k=off[i];k<off[i+1];++k){int f=faces[k];
  x+=sign[k]*length[f]*flux[f];y+=sign[k]*length[f]*flux[nf+f];}
  rate[n+i]+=h[i]*x/area[i];rate[2*n+i]+=h[i]*y/area[i];
 }
}

ENTRY void flow_scalar_faces(int nf,int n,double diffusivity,const double*q,const double*hi,
 const double*grad,const double*eq,const double*water,const int*owner,const int*nb,const int*bid,
 const double*dist,const double*displace,double*flux){
 LOOP(nf){int o=owner[i],j=nb[i];double co=hi[o]/q[o],value=co;
  if(j>=0){double cj=hi[j]/q[j];int up=water[i]>=0.?o:j;
   double ds=grad[6*n+up]*displace[(up==o?0:2)*nf+i]+grad[7*n+up]*displace[(up==o?1:3)*nf+i];
   double diff=up==o?cj-co:co-cj;
   double limited=ds*diff<=0.?0.:(fabs(ds)<fabs(diff)?ds:diff);
   value=(up==o?co:cj)+limited;
   flux[i]=water[i]*value-diffusivity*.5*(q[o]+q[j])*(cj-co)/(dist[i]+dist[nf+i]);
  }else flux[i]=bid[i]>=0?water[i]*(water[i]<0.?eq[o]:co):0.;
 }
}

ENTRY void flow_scalar_gather(int n,double dt,const double*old,const double*flux,
 const int*off,const int*faces,const int*sign,const double*length,const double*area,double*out){
 LOOP(n){double sum=0.;for(int k=off[i];k<off[i+1];++k){int f=faces[k];sum+=sign[k]*length[f]*flux[f];}
  out[i]=old[i]-dt*sum/area[i];
 }
}

// Same shoreline selection as the array implementation, one independent face
// per iteration. qsafe is division-safe scratch, raw depth decides wetness.
// Keep hi unmasked here: an incoming water face can inundate a dry receiver.
ENTRY void flow_shoreline_scalar_faces(int nf,double dry,const double*q,
 const double*qsafe,const double*hi,const double*eq,const double*water,
 const int*owner,const int*nb,const int*bid,double*flux){
 LOOP(nf){int o=owner[i],j=nb[i];
  if(j<0 && bid[i]<0){flux[i]=0.;continue;}
  if(q[o]>dry && (j<0 || q[j]>dry))continue;
  double donor=water[i]>=0.?hi[o]/qsafe[o]:(j>=0?hi[j]/qsafe[j]:eq[o]);
  flux[i]=water[i]*donor;
 }
}
