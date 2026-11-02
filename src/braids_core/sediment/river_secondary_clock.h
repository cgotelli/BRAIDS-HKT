// Shared OpenMP/CUDA stability bound for signed hI transport.
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

// Absolute row bound of the LS gradient projected onto each face's
// nonorthogonal displacement. Cached geometry only, rebuilt with each mesh.
ENTRY void secondary_metric(int n,int nf,int corrected,const int*owner,const int*nb,
 const int*off,const int*faces,const double*xy,const double*normal,
 const double*distance,const double*inv,double*metric){
 LOOP(nf){int o=owner[i],j=nb[i];metric[i]=0.;if(j<0)continue;
  double d=distance[i]+distance[nf+i];
  if(!corrected){metric[i]=2./d;continue;}
  double nx=normal[i],ny=normal[nf+i],dx=xy[j]-xy[o],dy=xy[n+j]-xy[n+o];
  double dn=dx*nx+dy*ny,tx=dx-dn*nx,ty=dy-dn*ny;
  double total=2.;
  for(int side=0;side<2;++side){int c=side?j:o;double bound=0.;
   for(int k=off[c];k<off[c+1];++k){int f=faces[k],a=owner[f]==c?nb[f]:owner[f];if(a<0)continue;
    double ax=xy[a]-xy[c],ay=xy[n+a]-xy[n+c],rr=ax*ax+ay*ay;
    double gx=(inv[c]*ax+inv[n+c]*ay)/rr,gy=(inv[n+c]*ax+inv[2*n+c]*ay)/rr;
    bound+=2.*fabs(tx*gx+ty*gy);
   }
   total+=(side?distance[i]:distance[nf+i])/d*bound;
  }
  metric[i]=total/fabs(dn);
 }
}

// Reconstruction can double a donor perturbation: include that factor,
// not just |u|/dx. Diffusion uses the actual nonorthogonal face operator.
ENTRY void secondary_frequency(int n,int nf,double dry,double diffusion,
 const double*q,const double*water,const double*metric,const int*owner,
 const int*nb,const int*bid,const int*off,const int*faces,const double*length,
 const double*area,double*out){
 LOOP(n){double h=q[i],sum=0.;if(h<=dry){out[i]=0.;continue;}
  for(int k=off[i];k<off[i+1];++k){int f=faces[k],o=owner[f],j=nb[f];
   if(j<0 && bid[f]<0)continue;
   double outward=o==i?water[f]:-water[f];
   sum+=2.*length[f]*fmax(outward,0.);
   if(j>=0 && q[o]>dry && q[j]>dry)
    sum+=length[f]*diffusion*.5*(q[o]+q[j])*metric[f];
  }
  out[i]=sum/(area[i]*h);
 }
}

// BRAIDS numerical guard, NOT a change to the locked FLOW oracle. FLOW's
// literal negative over-limit assignment reverses handedness. Preserve the
// magnitude limit while making beta odd and continuous in signed intensity.
ENTRY void secondary_signed_stress(int n,const double*inputs,double*outputs){
 LOOP(n){double u=inputs[2*n+i],v=inputs[3*n+i],I=inputs[9*n+i];
  double bs=outputs[11*n+i],speed=fmax(1.e-8,sqrt(u*u+v*v));
  double raw=bs*I/speed,limit=fabs(.1*bs);
  if(raw < -limit){double b=-limit;outputs[12*n+i]=b;
   outputs[13*n+i]=2.*b*u*v;outputs[14*n+i]=b*(u*u-v*v);}
 }
}

// Signed secondary state has a transported history. The equilibrium of the
// receiving cell is NOT an independent upstream forcing: importing it skips
// relaxation and closes a curvature -> I -> momentum -> curvature feedback.
// Prescribed signed upstream I is independent of the receiver, including
// inundation and reversal. Outgoing and interior/wall fluxes are unchanged.
ENTRY void secondary_open_faces(int nf,double dry,const double*q,const double*hi,
 const double*water,const int*owner,const int*nb,const int*bid,const double*incoming,double*flux){
 LOOP(nf){if(nb[i]<0 && bid[i]>=0 && water[i]<0.){
  flux[i]=water[i]*incoming[i];
 }}
}
