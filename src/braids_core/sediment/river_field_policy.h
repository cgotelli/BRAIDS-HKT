// Accepted-state AMR indicators, shared OpenMP/CUDA arithmetic. No physics update.
#ifdef __CUDACC__
#define ENTRY extern "C" __global__
#define LOOP(N) for(int i=blockDim.x*blockIdx.x+threadIdx.x;i<N;i+=blockDim.x*gridDim.x)
#else
#include <math.h>
#include <omp.h>
#define ENTRY extern "C" __declspec(dllexport)
#define LOOP(N) _Pragma("omp parallel for schedule(static)") for(int i=0;i<N;++i)
extern "C" __declspec(dllexport) void mobile_threads(int n){omp_set_dynamic(0);omp_set_num_threads(n);}
#endif
ENTRY void depth_tags(int n,double dry,const double*q,int*tag){
 LOOP(n){tag[i]=q[i]>dry?1:0;}
}
ENTRY void depth_buffer(int n,const int*off,const int*adj,const int*old,int*out){
 LOOP(n){int value=old[i];for(int e=off[i];e<off[i+1]&&!value;++e)value=old[adj[e]];out[i]=value;}
}
ENTRY void field_primitives(int n,int nc,double dry,const double*q,const double*z,
 const double*mass,const double*rho,const double*sus,const int*off,const int*adj,
 const double*xy,double*frac,double*conc,double*grad){
 LOOP(n){double total=0.;for(int k=0;k<nc;++k)total+=mass[k*n+i]/rho[k];
  for(int k=0;k<nc;++k){frac[k*n+i]=total>0.?mass[k*n+i]/rho[k]/total:0.;conc[k*n+i]=q[i]>dry?sus[k*n+i]/q[i]:0.;}
  double xx=0.,xyv=0.,yy=0.,xz=0.,yz=0.;
  for(int e=off[i];e<off[i+1];++e){int j=adj[e];double dx=xy[2*j]-xy[2*i],dy=xy[2*j+1]-xy[2*i+1],r=dx*dx+dy*dy,w=1./r,dz=z[j]-z[i];
   xx+=w*dx*dx;xyv+=w*dx*dy;yy+=w*dy*dy;xz+=w*dx*dz;yz+=w*dy*dz;}
  double det=xx*yy-xyv*xyv;
  grad[2*i]=det>1e-20?(yy*xz-xyv*yz)/det:0.;grad[2*i+1]=det>1e-20?(xx*yz-xyv*xz)/det:0.;
 }
}
ENTRY void field_tags(int n,int nc,double dry,double scale,double eta_tol,double vel_tol,
 double bed_tol,double change_tol,double mix_tol,double conc_tol,double interval,double buffer,
 const double*q,const double*z,const double*reference,const double*previous,const double*frac,
 const double*conc,const double*grad,const int*off,const int*adj,const double*xy,int*bits,double*reach){
 LOOP(n){double h=q[i],u=h>dry?q[n+i]/h:0.,v=h>dry?q[2*n+i]/h:0.;int b=0;
  if(fabs(z[i]-reference[i]-previous[i])>change_tol)b|=4;
  for(int e=off[i];e<off[i+1];++e){int j=adj[e];double hj=q[j],dx=xy[2*j]-xy[2*i],dy=xy[2*j+1]-xy[2*i+1],d=sqrt(dx*dx+dy*dy);
   if((h>dry)!=(hj>dry))b|=1;
   if(h>dry&&hj>dry){double uj=q[n+j]/hj,vj=q[2*n+j]/hj;
    if(fabs(h+z[i]-hj-z[j])*scale/d>eta_tol||hypot(u-uj,v-vj)*scale/d>vel_tol)b|=2;
    for(int k=0;k<nc;++k)if(fabs(conc[k*n+i]-conc[k*n+j])*scale/d>conc_tol)b|=16;
   }
   if(h>dry||hj>dry){
    if(fabs(z[j]-z[i]-grad[2*i]*dx-grad[2*i+1]*dy)>bed_tol)b|=32;
    for(int k=0;k<nc;++k)if(fabs(frac[k*n+i]-frac[k*n+j])*scale/d>mix_tol)b|=8;
   }
  }
  bits[i]=b;reach[i]=b?buffer+interval*(hypot(u,v)+2.*sqrt(9.81*fmax(h,0.))):-1.;
 }
}
ENTRY void field_buffer(int n,const int*off,const int*adj,const double*dist,const double*old,double*out){
 LOOP(n){double value=old[i];for(int e=off[i];e<off[i+1];++e)value=fmax(value,old[adj[e]]-dist[e]);out[i]=value;}
}
ENTRY void field_history(int nt,int ns,const int*off,const int*idx,const double*weight,
 const double*z,const double*reference,double*out){
 LOOP(nt){double value=0.;for(int e=off[i];e<off[i+1];++e){int j=idx[e];value+=weight[e]*(z[j]-reference[j]);}out[i]=value;}
}
