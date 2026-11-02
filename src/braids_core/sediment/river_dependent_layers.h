// Sorting forcing on the actual Cartesian/cropped/AMR face graph.
// FLOW references: taubot/caltmx current-only stress, bott3d face-flux sbot.
// Collocated face velocities use distance interpolation, not staggered indices.
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

ENTRY void sorting_face_stress(int nf,int n,double dry,double density,double gravity,
 const double*q,const double*z,const double*grad,const double*chezy,
 const int*owner,const int*nb,const int*bid,const double*dist,const double*disp,
 double*out){
 LOOP(nf){int o=owner[i],j=nb[i];out[i]=0.;out[nf+i]=0.;
  if(q[o]<=dry || (j<0?bid[i]<0:q[j]<=dry) || chezy[i]<=0.)continue;
  double h=q[o],u=q[n+o]/h,v=q[2*n+o]/h;
  if(j>=0){
   double d=dist[i]+dist[nf+i],wo=dist[nf+i]/d,wj=dist[i]/d;
   double eo=q[o]+z[o]+grad[6*n+o]*disp[i]+grad[7*n+o]*disp[nf+i];
   double ej=q[j]+z[j]+grad[6*n+j]*disp[2*nf+i]+grad[7*n+j]*disp[3*nf+i];
   h=wo*eo+wj*ej-fmax(z[o],z[j]);
   u=wo*u+wj*q[n+j]/q[j];v=wo*v+wj*q[2*n+j]/q[j];
  }
  if(h<=dry)continue;
  out[i]=density*gravity*(u*u+v*v)/(chezy[i]*chezy[i])*h;
  out[nf+i]=1.;
 }
}

ENTRY void sorting_cell_forcing(int n,int nf,int ng,const double*density,double dry,
 const double*q,const double*face,const double*normal,const double*length,
 const int*off,const int*faces,const double*flux,double*shear,double*bedload){
 LOOP(n){double xx=0.,xy=0.,yy=0.,sum=0.,stress=0.;
  for(int k=off[i];k<off[i+1];++k){int f=faces[k];if(face[nf+f]==0.)continue;
   double w=length[f],nx=normal[f],ny=normal[nf+f];
   xx+=w*nx*nx;xy+=w*nx*ny;yy+=w*ny*ny;sum+=w;stress+=w*face[f];
  }
  shear[i]=q[i]>dry&&sum>0.?stress/(sum*fmax(.01,q[i])):0.;
  double det=xx*yy-xy*xy,tr=xx+yy;
  for(int g=0;g<ng;++g){
   double bx=0.,by=0.;
   for(int k=off[i];k<off[i+1];++k){int f=faces[k];if(face[nf+f]==0.)continue;
    double value=length[f]*flux[g*nf+f];bx+=value*normal[f];by+=value*normal[nf+f];
   }
   double x=0.,y=0.;
   if(det>1.e-14*tr*tr){x=(yy*bx-xy*by)/det;y=(xx*by-xy*bx)/det;}
   else if(tr>0.){x=bx/tr;y=by/tr;}
   bedload[g*n+i]=q[i]>dry?density[g]*sqrt(x*x+y*y):0.;
  }
 }
}
