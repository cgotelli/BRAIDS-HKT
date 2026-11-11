// Section-local Manning/Chezy/Colebrook outlet research kernel.
// One thread per connected section; no host reductions or persisted state.
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

LOCAL double nout_velocity(double h,double hi,double u,double g){
 return h>hi?u-(h-hi)*sqrt(.5*g*(1./h+1./hi)):u+2.*(sqrt(g*hi)-sqrt(g*h));
}
LOCAL double nout_chezy(int law,double r,double value,double g){
 if(law==1)return value;
 if(law==2)return pow(r,1./6.)/value;
 if(law==3)return sqrt(g)/.41*log1p(30.*r/(exp(1.)*value));
 return 18.*log10(12.*fmax(.5,r/value));
}
LOCAL double nout_residual(int start,int end,int nf,int n,double eta,double g,double dry,int law,double slope,
 const int*faces,const int*owner,const double*q,const double*z,const double*bed,
 const double*normal,const double*length,const double*roughness,double*area_out,double*q_out){
 double area=0.,perimeter=0.,rate=0.,drag=0.,prev=0.;
 for(int k=start;k<end;++k){int f=faces[k],c=owner[f];double h=fmax(eta-bed[f],0.);
  area+=h*length[f];perimeter+=h>dry?length[f]:0.;
  if(k==start)perimeter+=h;else perimeter+=fabs(h-prev);
  prev=h;
 }
 perimeter+=prev;
 if(area<=0. || perimeter<=0.){*area_out=0.;*q_out=0.;return 0.;}
 double radius=area/perimeter;
 // The hydraulic radius is section-wide. Preserve the same division and
 // summation order, but evaluate the expensive Manning power only once.
 double manning_radius=law==2?pow(radius,1./6.):0.;
 for(int k=start;k<end;++k){int f=faces[k],c=owner[f];double h0=q[c],hi=fmax(h0+z[c]-bed[f],0.);
  double h=fmax(eta-bed[f],0.);double face_rate=0.;if(h0>dry && hi>dry){
   double ui=(q[n+c]*normal[f]+q[2*n+c]*normal[nf+f])/h0;
   if(ui>=sqrt(g*hi))face_rate=ui*hi;
   else {double invariant=fmax(ui+2.*sqrt(g*hi),0.),sonic=invariant*invariant/(9.*g);
    if(h<sonic)face_rate=sonic*sqrt(g*sonic);else if(h>dry)face_rate=h*fmax(nout_velocity(h,hi,ui,g),0.);
   }
  }
  rate+=face_rate*length[f];if(h<=dry)continue;
  double left=k==start?0.:fmax(eta-bed[faces[k-1]],0.);
  double right=k+1==end?0.:fmax(eta-bed[faces[k+1]],0.);
  double p=length[f]+fmax(h-left,0.)+fmax(h-right,0.);
  double C=law==2?manning_radius/roughness[c]:nout_chezy(law,radius,roughness[c],g);drag+=p/(C*C);
 }
 *area_out=area;*q_out=rate;
 return rate-(drag>0.?area*sqrt(area*slope/drag):0.);
}
ENTRY void normal_outlet(int nf,int n,int sections,double g,double dry,double speed,int law,
 const int*offset,const int*faces,const int*owner,const double*q,const double*z,const double*bed,
 const double*normal,const double*length,const double*roughness,const double*slopes,
 double*stage,double*metrics,double*trace,double*flux,int*face_status,int*section_status,int*transcritical){
 LOOP(sections){section_status[i]=0;transcritical[i]=0;stage[i]=0.;metrics[i]=0.;metrics[sections+i]=0.;
  int cursor=offset[i],stop=offset[i+1];
  while(cursor<stop){
   int firstface=faces[cursor],firstcell=owner[firstface];
   double firsth=q[firstcell],firstgeom=fmax(firsth+z[firstcell]-bed[firstface],0.);
   if(firsth<=dry || firstgeom<=dry){
    trace[firstface]=trace[nf+firstface]=trace[2*nf+firstface]=0.;flux[firstface]=0.;
    double pressure=firsth>dry?.5*g*firsth*firsth:0.;
    flux[nf+firstface]=pressure*normal[firstface];flux[2*nf+firstface]=pressure*normal[nf+firstface];
    ++cursor;continue;
   }
   int start=cursor;
   while(cursor<stop){int f=faces[cursor],c=owner[f];double hh=q[c];
    if(hh<=dry || hh+z[c]-bed[f]<=dry)break;
    ++cursor;
   }
   int end=cursor,wet=0;double lo=1e300,hi=-1e300,area=0.,rate=0.;
  for(int k=start;k<end;++k){int f=faces[k],c=owner[f];double h0=q[c],hf=fmax(h0+z[c]-bed[f],0.);
   if(h0<=dry || hf<=dry)continue;
   double u=(q[n+c]*normal[f]+q[2*n+c]*normal[nf+f])/h0;
   if(u>=sqrt(g*hf))continue;
   lo=fmin(lo,bed[f]+2.*dry);hi=fmax(hi,z[c]+h0);++wet;
  }
  double level=0.;bool fallback=false;
  if(wet){hi=fmax(hi,lo+1.);double r=nout_residual(start,end,nf,n,lo,g,dry,law,slopes[i],faces,owner,q,z,bed,normal,length,roughness,&area,&rate);
   if(!isfinite(r))section_status[i]=1;
   // The normal-flow relation is then entirely supercritical. A downstream
   // stage cannot control it, so preserve the outgoing interior trace.
   else if(r<0.){fallback=true;transcritical[i]=1;}
   if(!section_status[i]&&!fallback){
    double rh=nout_residual(start,end,nf,n,hi,g,dry,law,slopes[i],faces,owner,q,z,bed,normal,length,roughness,&area,&rate);
    for(int it=0;it<60 && rh>0.;++it){hi=lo+2.*(hi-lo);rh=nout_residual(start,end,nf,n,hi,g,dry,law,slopes[i],faces,owner,q,z,bed,normal,length,roughness,&area,&rate);}
    if(!isfinite(rh)||rh>0.)section_status[i]=2;
    for(int it=0;it<90 && !section_status[i];++it){double m=.5*(lo+hi);if(m==lo||m==hi)break;
     r=nout_residual(start,end,nf,n,m,g,dry,law,slopes[i],faces,owner,q,z,bed,normal,length,roughness,&area,&rate);
     if(r>0.)lo=m;else hi=m;
    }
    level=.5*(lo+hi);r=nout_residual(start,end,nf,n,level,g,dry,law,slopes[i],faces,owner,q,z,bed,normal,length,roughness,&area,&rate);
    if(!isfinite(r)||fabs(r)>1e-10*fmax(1.,fabs(rate)))section_status[i]=3;
   }
  }
  stage[i]=fallback?-1.:level;metrics[i]+=area;metrics[sections+i]+=rate;
  if(section_status[i])break;
  for(int k=start;k<end;++k){int f=faces[k],c=owner[f];double h0=q[c],geom=fmax(h0+z[c]-bed[f],0.);
   double nx=normal[f],ny=normal[nf+f],ux=0.,uy=0.;if(h0>dry){ux=q[n+c]/h0;uy=q[2*n+c]/h0;}
   double un=ux*nx+uy*ny,tx=ux-un*nx,ty=uy-un*ny,h=geom;
   if(h0<=dry || geom<=dry){h=0.;un=0.;}
   else if(fallback){} // normal flow is supercritical: preserve outgoing interior trace
   else if(un>=sqrt(g*geom)){} // outward supercritical trace is not controlled
   else {double incoming=un;h=fmax(level-bed[f],0.);
    double invariant=fmax(incoming+2.*sqrt(g*geom),0.),sonic=invariant*invariant/(9.*g);
    if(h<sonic){h=sonic;un=sqrt(g*sonic);}else un=h>dry?nout_velocity(h,geom,incoming,g):0.;
    if(un<=0.){
     // The outflow-only part of a section is a reflecting boundary, not a
     // zero-velocity trace at an arbitrary externally imposed stage.
     double left=dry,right=fmax(geom,1.);
     for(int it=0;it<60 && nout_velocity(right,geom,incoming,g)>0.;++it)right*=2.;
     for(int it=0;it<80;++it){double m=.5*(left+right);
      if(nout_velocity(m,geom,incoming,g)>0.)left=m;else right=m;}
     h=.5*(left+right);un=0.;
    }
   }
   if(!isfinite(h)||!isfinite(un)||sqrt(un*un+tx*tx+ty*ty)>speed){face_status[f]=3;continue;}
   double water=0.,fx=0.,fy=0.;
   if(h>dry){trace[f]=h;trace[nf+f]=h*(un*nx+tx);trace[2*nf+f]=h*(un*ny+ty);
    water=h*un;fx=water*(un*nx+tx)+.5*g*h*h*nx;fy=water*(un*ny+ty)+.5*g*h*h*ny;}
   else {trace[f]=trace[nf+f]=trace[2*nf+f]=0.;}
   double pressure=h0>dry?.5*g*(h0*h0-geom*geom):0.;
   flux[f]=water;flux[nf+f]=fx+pressure*nx;flux[2*nf+f]=fy+pressure*ny;
  }
  }
 }
}
