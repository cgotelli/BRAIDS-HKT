// Shared OpenMP/CUDA bedform feedback on the actual river face graph.
// Constitutive predictor b102_bedform_cell is prepended from the qualified
// BRAIDS static-bedform implementation; no Delft3D source is compiled here.
#ifdef __CUDACC__
#define ENTRY extern "C" __global__
#define LOOP(N) for(int i=blockDim.x*blockIdx.x+threadIdx.x;i<N;i+=blockDim.x*gridDim.x)
#else
#include <omp.h>
#define ENTRY extern "C" __declspec(dllexport)
#define LOOP(N) _Pragma("omp parallel for schedule(static)") for(int i=0;i<N;++i)
extern "C" __declspec(dllexport) void mobile_threads(int n){omp_set_dynamic(0);omp_set_num_threads(n);}
#endif
#ifdef __CUDACC__
#define BDF_LOCAL __device__ __forceinline__
#else
#define BDF_LOCAL inline
#endif
BDF_LOCAL double bdf_cdf(int j,int ng,const double*frac,const double*cdf){
 double value=0.;for(int k=0;k<ng;++k)value+=frac[k]*cdf[j*ng+k];return value;
}
ENTRY void bdf_grains(int n,int ng,int nk,const double*mass,const double*diam,
 const double*knots,const double*cdf,double*out){
 LOOP(n){double total=0.,frac[32];
  for(int k=0;k<ng;++k)total+=mass[k*n+i];
  for(int k=0;k<ng;++k)frac[k]=total>0.?mass[k*n+i]/total:0.;
  for(int p=0;p<2;++p){double target=p?.9:.5;
   if(total<=0.){out[p*n+i]=diam[0]*(p?1.5:1.);continue;}
   // Lower-bound search selects the leftmost inverse at a probability plateau.
   int lo=0,hi=nk-1;
   while(lo<hi){int j=(lo+hi)/2;if(bdf_cdf(j,ng,frac,cdf)<target)lo=j+1;else hi=j;}
   int j=lo;double x=knots[j];
   if(j>0){double a=bdf_cdf(j-1,ng,frac,cdf),b=bdf_cdf(j,ng,frac,cdf);
    if(b>a)x=knots[j-1]+fmax(0.,fmin(1.,(target-a)/(b-a)))*(knots[j]-knots[j-1]);
   }
   out[p*n+i]=exp(x);
  }
 }
}
ENTRY void bdf_predict(int n,double active,const int*mode,const double*params,
 const double*q,const double*c,const double*grains,double*out,int*status){
 LOOP(n){status[i]=0;if(q[i]<=active)continue;
  double h=q[i],u=q[n+i]/h,v=q[2*n+i]/h,local[9];int s=0;
  b102_bedform_cell(0,1,mode,params,&h,&u,&v,c+i,grains+i,grains+n+i,local,&s);
  for(int k=0;k<9;++k)out[k*n+i]=local[k];status[i]=s;
 }
}
ENTRY void bdf_faces(int nf,int n,double dry,const double*q,const double*z,const double*grad,
 const double*components,const int*owner,const int*nb,const int*bid,const double*dist,
 const double*disp,const double*weights,double*depth,double*wetweights,double*out){
 LOOP(nf){int o=owner[i],j=nb[i];
  double h=q[o],wo=1.,wj=0.;
  if(j>=0){double d=dist[i]+dist[nf+i];wo=dist[nf+i]/d;wj=dist[i]/d;
   double eo=q[o]+z[o]+grad[6*n+o]*disp[i]+grad[7*n+o]*disp[nf+i];
   double ej=q[j]+z[j]+grad[6*n+j]*disp[2*nf+i]+grad[7*n+j]*disp[3*nf+i];
   h=wo*eo+wj*ej-fmax(z[o],z[j]);
  }
  // Retain the old effective-face diagnostic's 1 cm floor, independently
  // of the physical-water depth used by trtrou/calrou below.
  depth[i]=(j<0&&bid[i]<0)?0.:fmax(.01,h);
  bool active=q[o]>dry&&(j>=0?q[j]>dry:bid[i]>=0);
  wetweights[i]=active?weights[i]:0.;wetweights[nf+i]=active?weights[nf+i]:0.;
  if(!active){for(int k=0;k<3;++k)out[k*nf+i]=0.;continue;}
  if(j>=0)h=fmax(dry,h);
  double ks2=0.;for(int k=2;k<5;++k){double r=wo*components[k*n+o]+(j>=0?wj*components[k*n+j]:0.);ks2+=r*r;}
  // FLOW trachytope 105, area averaging 1, ROUMET=C. Calrou applies
  // its 1 mm floor AFTER the half-depth cap in trtrou (do not reverse).
  double ks=fmax(.001,fmax(1e-6,fmin(sqrt(ks2),.5*h)));
  double c=18.*log10(fmax(12.*h/ks,1.0129));
  out[i]=c;out[nf+i]=h/(exp(1.)*expm1(.41*c/sqrt(9.81)));out[2*nf+i]=ks;
 }
}
ENTRY void bdf_gather(int n,int nf,double baseline,const double*q,const double*face,
 const double*weights,const double*face_depth,const int*off,const int*faces,const int*owner,
 double*hydro,double*out,double*cq){
 LOOP(n){double sum=0.,drag=0.,z0=0.,depth=0.,total=0.;
  for(int k=off[i];k<off[i+1];++k){int f=faces[k];if(face[f]<=0.)continue;
   double w=weights[(owner[f]==i?0:nf)+f];depth+=w*face_depth[f];total+=w;
   sum+=w;drag+=w/(face[f]*face[f]);z0+=w*face[nf+f];
  }
  hydro[i]=sum>0.?sqrt(sum/drag):baseline;
  z0=sum>0.?z0/sum:fmax(.01,q[i])/(exp(1.)*expm1(.41*baseline/sqrt(9.81)));
  double c=sqrt(9.81)/.41*log1p(fmax(.01,q[i])/fmax(1e-8,exp(1.)*z0));
  double factor=.41*c/(sqrt(9.81)*log1p(q[i]*double(float(.368))/z0));
  out[i]=fmax(1.e-6,c);out[n+i]=factor;out[2*n+i]=z0;out[3*n+i]=total>0.?depth/total:fmax(.01,q[i]);
  cq[i]=q[i];cq[n+i]=q[n+i]*factor;cq[2*n+i]=q[2*n+i]*factor;
 }
}
