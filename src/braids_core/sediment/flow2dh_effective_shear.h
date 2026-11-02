// Explicit FLOW sediment forcing adapter, not a change to f-wave hydraulics.
#ifdef __CUDACC__
#define ENTRY extern "C" __global__
#define LOOP(N) for(int i=blockDim.x*blockIdx.x+threadIdx.x;i<N;i+=blockDim.x*gridDim.x)
#else
#include <cmath>
#include <omp.h>
#define ENTRY extern "C" __declspec(dllexport)
#define LOOP(N) _Pragma("omp parallel for schedule(static)") for(int i=0;i<N;++i)
extern "C" __declspec(dllexport) void mobile_threads(int n){omp_set_dynamic(0);omp_set_num_threads(n);}
#endif

ENTRY void effective_face_depth(int nf,int n,const double*q,const double*z,const double*grad,
 const int*owner,const int*nb,const int*bid,const double*distance,const double*displace,double*out){
 LOOP(nf){int o=owner[i],j=nb[i];
  if(j<0){out[i]=bid[i]>=0?fmax(.01,q[o]):0.;continue;}
  double d=distance[i]+distance[nf+i],wo=distance[nf+i]/d,wj=distance[i]/d;
  double eo=q[o]+z[o]+grad[6*n+o]*displace[i]+grad[7*n+o]*displace[nf+i];
  double ej=q[j]+z[j]+grad[6*n+j]*displace[2*nf+i]+grad[7*n+j]*displace[3*nf+i];
  // MOR is mapped to MIN in FLOW: choose the shallower adjacent bed depth.
  // Reconstruct surface, not cell depth, to the actual non-orthogonal face.
  out[i]=fmax(.01,wo*eo+wj*ej-fmax(z[o],z[j]));
 }
}

ENTRY void effective_cell_shear(int n,int nf,double configured_chezy,const double*q,
 const double*face_depth,const double*weights,const int*off,const int*faces,
 const int*owner,const int*nb,const int*bid,double*out,double*cq){
 LOOP(n){double total=0.,sum=0.;
  for(int k=off[i];k<off[i+1];++k){int f=faces[k];
   if(nb[f]<0&&bid[f]<0)continue;
   double w=weights[(owner[f]==i?0:nf)+f];sum+=w*face_depth[f];total+=w;
  }
  double hf=total>0.?sum/total:fmax(.01,q[i]);
  double sg=sqrt(9.81),kap=.41,ee=exp(1.);
  double z0=hf/(ee*expm1(kap*configured_chezy/sg));
  double c=sg/kap*log1p(fmax(.01,q[i])/fmax(1.e-8,ee*z0));
  // erosed floors its roughness working depth at 1 cm, but dwnvel's
  // characteristic elevation is 0.368 * ACTUAL water depth. Applying the
  // roughness floor here suppresses shallow-cell shear/transport. This
  // entry receives positive division-safe scratch from the shoreline owner.
  // dwnvel uses default-real 0.368, rather than exactly exp(-1).
  double speed_factor=kap/log1p(q[i]*double(float(.368))/z0)*c/sg;
  out[i]=fmax(1.e-6,c);out[n+i]=speed_factor;out[2*n+i]=z0;out[3*n+i]=hf;
  cq[i]=q[i];cq[n+i]=q[n+i]*speed_factor;cq[2*n+i]=q[2*n+i]*speed_factor;
 }
}

// A dry neighbour supplies neither a water-surface slope nor hydraulic
// roughness. These shoreline-only entries consume physical depth, never
// the positive scratch depth used to evaluate sediment constitutive laws.
ENTRY void effective_wet_gradient(int n,double dry,const double*q,const double*z,
 const int*off,const int*faces,const int*owner,const int*nb,const double*xy,double*grad){
 LOOP(n){double xx=0.,xyv=0.,yy=0.,bx=0.,by=0.;
  if(q[i]>dry)for(int k=off[i];k<off[i+1];++k){int f=faces[k],j=owner[f]==i?nb[f]:owner[f];
   if(j<0||q[j]<=dry)continue;
   double dx=xy[j]-xy[i],dy=xy[n+j]-xy[n+i],d2=dx*dx+dy*dy;
   if(d2<=0.)continue;
   double w=1./d2,dh=q[j]+z[j]-q[i]-z[i];
   xx+=w*dx*dx;xyv+=w*dx*dy;yy+=w*dy*dy;bx+=w*dx*dh;by+=w*dy*dh;
  }
  double det=xx*yy-xyv*xyv,tr=xx+yy,gx=0.,gy=0.;
  if(det>1.e-14*tr*tr){gx=(yy*bx-xyv*by)/det;gy=(xx*by-xyv*bx)/det;}
  else if(tr>0.){gx=bx/tr;gy=by/tr;}
  grad[6*n+i]=gx;grad[7*n+i]=gy;
 }
}

ENTRY void effective_wet_weights(int nf,int n,double dry,const double*q,
 const int*owner,const int*nb,const int*bid,const double*weights,double*out){
 LOOP(nf){int o=owner[i],j=nb[i];
  bool active=q[o]>dry&&(j>=0?q[j]>dry:bid[i]>=0);
  out[i]=active?weights[i]:0.;out[nf+i]=active?weights[nf+i]:0.;
 }
}
