// Surface-perturbation turbine correction. Same FP64 OpenMP/CUDA arithmetic.
#ifdef __CUDACC__
#define ENTRY extern "C" __global__
#define LOOP(N) for(int i=blockDim.x*blockIdx.x+threadIdx.x;i<N;i+=blockDim.x*gridDim.x)
#else
#include <cmath>
#include <omp.h>
#define ENTRY extern "C" __declspec(dllexport)
#define LOOP(N) _Pragma("omp parallel for schedule(static)") for(int i=0;i<N;++i)
extern "C" __declspec(dllexport) void hkt_stress_threads(int n){omp_set_dynamic(0);omp_set_num_threads(n);}
#endif

ENTRY void hkt_wet_gradients(int n,double dry,const double*q,const double*z,
 const int*off,const int*faces,const int*owner,const int*nb,const double*dxdy,
 const double*inverse,double*out){
 LOOP(n){for(int k=0;k<10;k++)out[k*n+i]=0.;
  double xx=0.,xy=0.,yy=0.,bx[3]={0.,0.,0.},by[3]={0.,0.,0.};
  double bz[2]={0.,0.};int na=off[n];bool wet=q[i]>dry;
  for(int s=off[i];s<off[i+1];s++){
   int f=faces[s],j=owner[f]==i?nb[f]:owner[f];if(j<0)continue;
   double dx=dxdy[s],dy=dxdy[na+s],w=1./(dx*dx+dy*dy),dz=z[j]-z[i];
   bz[0]+=w*dx*dz;bz[1]+=w*dy*dz;
   if(!wet || q[j]<=dry)continue;
   xx+=w*dx*dx;xy+=w*dx*dy;yy+=w*dy*dy;
   double difference[3]={q[n+j]/q[j]-q[n+i]/q[i],q[2*n+j]/q[j]-q[2*n+i]/q[i],(q[j]+z[j])-(q[i]+z[i])};
   for(int k=0;k<3;k++){bx[k]+=w*dx*difference[k];by[k]+=w*dy*difference[k];}
  }
  out[6*n+i]=inverse[i]*bz[0]+inverse[n+i]*bz[1];
  out[7*n+i]=inverse[n+i]*bz[0]+inverse[2*n+i]*bz[1];
  double det=xx*yy-xy*xy;
  if(det>1e-12*fmax(1.,xx*yy)){
   for(int k=0;k<3;k++){out[2*k*n+i]=(yy*bx[k]-xy*by[k])/det;out[(2*k+1)*n+i]=(xx*by[k]-xy*bx[k])/det;}
  }
 }
}

ENTRY void hkt_surface_pack(int nf,int n,const double*local,const int*owner,
 const int*nb,const int*constant,double*out){
 LOOP(nf){for(int k=0;k<10;k++)out[k*nf+i]=0.;if(nb[i]<0)continue;
  double hl=local[i],hr=local[4*nf+i];
  out[i]=hl+local[3*nf+i];out[nf+i]=hr+local[7*nf+i];
  out[2*nf+i]=hl>0.?local[nf+i]/hl:0.;out[3*nf+i]=hr>0.?local[5*nf+i]/hr:0.;
  out[4*nf+i]=hl>0.?hypot(local[nf+i],local[2*nf+i])/hl:0.;
  out[5*nf+i]=hr>0.?hypot(local[5*nf+i],local[6*nf+i])/hr:0.;
  out[6*nf+i]=local[3*nf+i];out[7*nf+i]=local[7*nf+i];
  out[8*nf+i]=constant[owner[i]];out[9*nf+i]=constant[nb[i]];
 }
}

ENTRY void hkt_balanced_pressure(int nf,int n,double gravity,double dry,
 const double*q,const double*z,const double*trace,const double*grad,const double*source,
 const int*owner,const int*nb,const double*normal,const double*distance,
 const double*displacement,double*out,int*status){
 LOOP(nf){out[i]=0.;out[nf+i]=0.;status[i]=0;int a=owner[i],b=nb[i];if(b<0)continue;
  // Native augmented f-wave retains the shoreline/bore operator. This one
  // spatial correction is zero outside its fully wet small-surface-jump range.
  if(q[a]<=dry || q[b]<=dry){status[i]=1;continue;}
  double bed=fmax(fmax(z[a],z[b]),fmax(trace[6*nf+i],trace[7*nf+i]));
  double ha=fmax(0.,q[a]+z[a]-bed),hb=fmax(0.,q[b]+z[b]-bed);
  double hl=fmax(0.,trace[i]-bed),hr=fmax(0.,trace[nf+i]-bed);
  if(fmin(fmin(ha,hb),fmin(hl,hr))<=dry){status[i]=1;continue;}
  double h=.5*(hl+hr);
  if(fabs(hr-hl)>.005*h){status[i]=2;continue;}
  double old=.25*gravity*(hl*hl+hr*hr),po=old,pn=old;
  if(trace[8*nf+i]!=0.)po+=.5*gravity*(ha*ha-hl*hl);
  if(trace[9*nf+i]!=0.)pn+=.5*gravity*(hb*hb-hr*hr);
  if(fabs(po-pn)>128.*2.2204460492503131e-16*fmax(1.,fmax(fabs(po),fabs(pn)))){status[i]=3;continue;}
  double da=distance[i],db=distance[nf+i],d=da+db,wa=db/d,wb=da/d;
  double sx=wa*displacement[i]+wb*displacement[2*nf+i];
  double sy=wa*displacement[nf+i]+wb*displacement[3*nf+i];
  double pressure=.5*gravity*(wa*ha*ha+wb*hb*hb);
  pressure+=gravity*sx*(wa*ha*grad[4*n+a]+wb*hb*grad[4*n+b]);
  pressure+=gravity*sy*(wa*ha*grad[5*n+a]+wb*hb*grad[5*n+b]);
  double nx=normal[i],ny=normal[nf+i];
  pressure+=da*db/d*((source[n+a]-source[n+b])*nx+(source[2*n+a]-source[2*n+b])*ny);
  double delta=.5*(po+pn)-pressure;
  double speed=sqrt(gravity*h),theta=fmin(1.,fmax(trace[4*nf+i],trace[5*nf+i])/speed);
  // Frozen low-Froude normal velocity-jump correction, common-depth form.
  delta-=.5*(1.-theta)*speed*(2.*hl*hr/(hl+hr))*(trace[3*nf+i]-trace[2*nf+i]);
  out[i]=delta*nx;out[nf+i]=delta*ny;
 }
}
