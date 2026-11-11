// Q(stage) rating, characteristic-coupled whole-outlet control.
#ifdef __CUDACC__
#define ENTRY extern "C" __global__
#define LOCAL __device__ __forceinline__
#define LOOP(N) for(int i=blockDim.x*blockIdx.x+threadIdx.x;i<N;i+=blockDim.x*gridDim.x)
#else
#include <cmath>
#include <omp.h>
using std::isfinite;
#define ENTRY extern "C" __declspec(dllexport)
#define LOCAL inline
#define LOOP(N) _Pragma("omp parallel for schedule(static)") for(int i=0;i<N;++i)
extern "C" __declspec(dllexport) void mobile_threads(int n){omp_set_dynamic(0);omp_set_num_threads(n);}
#endif

LOCAL double rating_velocity(double h,double hi,double u,double g){
 return h>hi?u-(h-hi)*sqrt(.5*g*(1./h+1./hi)):u+2.*(sqrt(g*hi)-sqrt(g*h));
}
LOCAL double rating_q(int b,double eta,const int*table_offsets,const double*table_q,const double*table_eta){
 int a=table_offsets[b],end=table_offsets[b+1];
 if(eta<=table_eta[a])return table_q[a];
 if(eta>=table_eta[end-1])return table_q[end-1];
 int lo=a,hi=end-1;
 while(hi-lo>1){int mid=(lo+hi)/2;if(table_eta[mid]<=eta)lo=mid;else hi=mid;}
 double w=(eta-table_eta[lo])/(table_eta[hi]-table_eta[lo]);
 return (1.-w)*table_q[lo]+w*table_q[hi];
}
LOCAL double rating_residual(int b,int nf,int n,double eta,double g,double dry,
 const int*offset,const int*faces,const int*owner,const double*q,const double*z,const double*bed,
 const double*normal,const double*length,const int*table_offsets,const double*table_q,const double*table_eta,double*actual){
 double Q=0.;
 for(int k=offset[b];k<offset[b+1];++k){int f=faces[k],c=owner[f];double hc=q[c],hi=fmax(hc+z[c]-bed[f],0.);
  if(hc<=dry || hi<=dry)continue;
  double u=(q[n+c]*normal[f]+q[2*n+c]*normal[nf+f])/hc;
  if(u>=sqrt(g*hi)){Q+=hi*u*length[f];continue;}
  double h=fmax(eta-bed[f],0.);if(h>dry)Q+=h*fmax(rating_velocity(h,hi,u,g),0.)*length[f];
 }
 *actual=Q;return Q-rating_q(b,eta,table_offsets,table_q,table_eta);
}

ENTRY void rating_outlet(int nf,int n,int nb,double g,double dry,double speed,
 const int*control,const int*offset,const int*faces,const int*owner,const double*q,const double*z,const double*bed,
 const double*normal,const double*length,const int*table_offsets,const double*table_q,const double*table_eta,
 double*stage,double*rate,double*trace,double*flux,int*face_status,int*section_status){
 LOOP(nb){if(!control[i])continue;section_status[i]=0;stage[i]=0.;rate[i]=0.;
  int first=table_offsets[i],last=table_offsets[i+1]-1,wet=0;double lo=table_eta[first],hi=table_eta[last];
  for(int k=offset[i];k<offset[i+1];++k){int f=faces[k],c=owner[f];double h0=q[c],h=fmax(h0+z[c]-bed[f],0.);
   if(h0<=dry || h<=dry)continue;
   double u=(q[n+c]*normal[f]+q[2*n+c]*normal[nf+f])/h0;
   if(u>=sqrt(g*h))continue;
   double left=dry,right=fmax(h,1.);
   for(int it=0;it<60 && rating_velocity(right,h,u,g)>sqrt(g*right);++it)right*=2.;
   for(int it=0;it<60;++it){double m=.5*(left+right);if(rating_velocity(m,h,u,g)>sqrt(g*m))left=m;else right=m;}
   lo=fmax(lo,bed[f]+right);++wet;
  }
  double level=0.,actual=0.;
  if(wet){
   if(!(lo<=hi)){section_status[i]=1;continue;}
   double rlo=rating_residual(i,nf,n,lo,g,dry,offset,faces,owner,q,z,bed,normal,length,table_offsets,table_q,table_eta,&actual);
   double rhi=rating_residual(i,nf,n,hi,g,dry,offset,faces,owner,q,z,bed,normal,length,table_offsets,table_q,table_eta,&actual);
   if(!isfinite(rlo)||!isfinite(rhi)||rlo<0.||rhi>0.){section_status[i]=1;continue;}
   for(int it=0;it<90;++it){double m=.5*(lo+hi);if(m==lo||m==hi)break;
    double r=rating_residual(i,nf,n,m,g,dry,offset,faces,owner,q,z,bed,normal,length,table_offsets,table_q,table_eta,&actual);
    if(r>0.)lo=m;else hi=m;
   }
   level=.5*(lo+hi);double r=rating_residual(i,nf,n,level,g,dry,offset,faces,owner,q,z,bed,normal,length,table_offsets,table_q,table_eta,&actual);
   if(!isfinite(r)||fabs(r)>1e-10*fmax(1.,fabs(actual))){section_status[i]=2;continue;}
   // A fictitious zero-flow root can occur at the lowest rating point when
   // every local characteristic wants backflow. Reflecting those faces would
   // give a different stage, so it is not a solution of the Q–H condition.
   if(actual<=1e-14){
    for(int k=offset[i];k<offset[i+1];++k){int f=faces[k],c=owner[f];double hc=q[c],hh=fmax(hc+z[c]-bed[f],0.);
     if(hc<=dry||hh<=dry)continue;
     double ui=(q[n+c]*normal[f]+q[2*n+c]*normal[nf+f])/hc,h=fmax(level-bed[f],0.);
     if(h>dry && ui<sqrt(g*hh) && rating_velocity(h,hh,ui,g)<-1e-12)section_status[i]=3;
    }
    if(section_status[i])continue;
   }
  }
  stage[i]=level;rate[i]=actual;
  for(int k=offset[i];k<offset[i+1];++k){int f=faces[k],c=owner[f];double h0=q[c],geom=fmax(h0+z[c]-bed[f],0.);
   double nx=normal[f],ny=normal[nf+f],ux=0.,uy=0.;if(h0>dry){ux=q[n+c]/h0;uy=q[2*n+c]/h0;}
   double un=ux*nx+uy*ny,tx=ux-un*nx,ty=uy-un*ny,h=geom;
   if(h0<=dry || geom<=dry){h=0.;un=0.;}
   else if(un>=sqrt(g*geom)){} // No downstream control for supercritical outflow.
   else {double incoming=un;h=fmax(level-bed[f],0.);un=h>dry?rating_velocity(h,geom,incoming,g):0.;
    if(un<=0.){double left=dry,right=fmax(geom,1.);
     for(int it=0;it<60 && rating_velocity(right,geom,incoming,g)>0.;++it)right*=2.;
     for(int it=0;it<80;++it){double m=.5*(left+right);if(rating_velocity(m,geom,incoming,g)>0.)left=m;else right=m;}
     h=.5*(left+right);un=0.;
    }
   }
   if(!isfinite(h)||!isfinite(un)||sqrt(un*un+tx*tx+ty*ty)>speed){face_status[f]=3;continue;}
   double water=0.,fx=0.,fy=0.;
   if(h>dry){trace[f]=h;trace[nf+f]=h*(un*nx+tx);trace[2*nf+f]=h*(un*ny+ty);
    water=h*un;fx=water*(un*nx+tx)+.5*g*h*h*nx;fy=water*(un*ny+ty)+.5*g*h*h*ny;}
   else trace[f]=trace[nf+f]=trace[2*nf+f]=0.;
   double pressure=h0>dry?.5*g*(h0*h0-geom*geom):0.;
   flux[f]=water;flux[nf+f]=fx+pressure*nx;flux[2*nf+f]=fy+pressure*ny;
  }
 }
}
