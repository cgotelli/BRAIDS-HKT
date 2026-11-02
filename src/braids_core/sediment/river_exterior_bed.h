// Exterior bed state and hydraulic traces on the selected OpenMP/CUDA backend.
// No physical cell bed, water or sediment inventory is overwritten here.
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

ENTRY void exterior_update(int nf,double dt,const int*mode,const int*owner,
 const double*accepted,const double*oldz,const double*newz,const double*water,
 const double*value,double*trial){
 LOOP(nf){
  double z=accepted[i];int m=mode[i];
  if(m && (m==1 || water[i]>0.))z+=newz[owner[i]]-oldz[owner[i]];
  else if(m==3)z=value[i];
  else if(m==4)z+=dt*value[i];
  trial[i]=z;
 }
}

// First reduce blocks of 64 faces, then their partial sums. No floating atomics
// and no all-face serial section kernel; the same reduction tree runs on both.
ENTRY void exterior_partial(int count,int nf,int nb,int tiles,int fields,
 const int*offset,const int*faces,const double*input,double*partial){
 LOOP(count){
  int tile=i%tiles,b=(i/tiles)%nb,field=i/(tiles*nb);
  double sum=0.;int end=offset[b+1];
  for(int k=offset[b]+64*tile;k<end && k<offset[b]+64*(tile+1);++k)
   sum+=input[field*nf+faces[k]];
  partial[i]=sum;
 }
}
ENTRY void exterior_totals(int count,int tiles,const double*partial,double*total){
 LOOP(count){double s=0.;for(int k=0;k<tiles;++k)s+=partial[i*tiles+k];total[i]=s;}
}
ENTRY void exterior_stage_terms(int nf,int n,double dry,const int*owner,
 const int*mask,const double*q,const double*z,const double*length,double*terms){
 LOOP(nf){int c=owner[i];double w=mask[i] && q[c]>dry?length[i]:0.;
  terms[i]=(z[c]+q[c])*w;terms[nf+i]=w;
 }
}
LOCAL double exterior_chezy(int mode,double h,double parameter,double floor,double gravity,int waqua){
 double d=fmax(h,floor);
 if(mode==1)return parameter;
 if(mode==2)return pow(d,1./6.)/parameter;
 return waqua?18.*log10(12.*fmax(.5,d/parameter)):
  sqrt(gravity)/.41*log1p(30.*d/(exp(1.)*parameter));
}
ENTRY void exterior_weights(int nf,int n,int nb,double dry,double gravity,
 int roughness_mode,double floor,int waqua,const int*owner,const int*bid,
 const int*selected,const int*mode,const double*q,const double*z,const double*bed,
 const double*length,const double*roughness,const double*stage,const double*forcing,double*weights){
 LOOP(nf){weights[i]=0.;int b=bid[i],c=owner[i];
  if(!selected[i])continue;
  if(mode[b]==9){
   double h=fmax(forcing[nb+b]-bed[i],0.);
   if(h>dry)weights[i]=h*length[i];
   continue;
  }
  if(mode[b]!=6 || stage[nb+b]<=0.)continue;
  double h=fmax(q[c]+z[c]-bed[i],0.),section=fmax(stage[b]/stage[nb+b]-bed[i],0.);
  if(h>dry && section>0.)weights[i]=pow(section,1.5)*
   exterior_chezy(roughness_mode,h,roughness[c],floor,gravity,waqua)*length[i];
 }
}
ENTRY void exterior_flux(int nf,int n,int nb,double gravity,double dry,double speed_limit,
 const int*owner,const int*bid,const int*selected,const int*mode,const int*stage_policy,
 const double*q,const double*z,const double*bed,const double*normal,const double*length,
 const double*forcing,const double*weights,const double*total,double*trace,double*flux,int*status){
 LOOP(nf){
  status[i]=0;for(int k=0;k<3;++k){trace[k*nf+i]=0.;flux[k*nf+i]=0.;}
  if(!selected[i])continue;
  int b=bid[i],c=owner[i],m=mode[b];double h0=q[c],nx=normal[i],ny=normal[nf+i];
  if(m==4 || m==5)continue; // Resident characteristic section owner writes this outlet.
  double h=fmax(h0+z[c]-bed[i],0.),geometric=h,ux=0.,uy=0.;
  if(h0>dry){ux=q[n+c]/h0;uy=q[2*n+c]/h0;}
  double un=ux*nx+uy*ny,tx=ux-un*nx,ty=uy-un*ny,normal_speed=0.;
  if(m==9){
   // Two explicit inlet controls: Q and section water line. Unlike the
   // free-stage inlet, this state can admit water into a dry physical owner.
   h=fmax(forcing[nb+b]-bed[i],0.);tx=ty=0.;
   double Q=forcing[b];
   if(Q>0. && total[b]<=0.){status[i]=1;continue;}
   if(h>dry && Q>0.)normal_speed=-Q/total[b];
  }else if(m==6){
   double Q=forcing[b];
   if(Q>0. && total[b]<=0.){status[i]=1;continue;}
   if(h>dry && Q>0.)normal_speed=-Q*weights[i]/(total[b]*h*length[i]);
  }else if(m==3){
   // No downstream datum is inferred for a supercritical outflow. A wet
   // subcritical or inward state needs another explicitly selected control.
   if(h0<=dry)h=0.;
   else if(h>dry){
    if(un<sqrt(gravity*h)){status[i]=6;continue;}
    normal_speed=un;
   }
  }else if(m==2){
   double prescribed=fmax(forcing[nb+b]-bed[i],0.);
   if(stage_policy[b] && (h0<=dry || h<=dry) && prescribed>dry){status[i]=4;continue;}
   if(stage_policy[b] && h0<=dry)h=0.;
   // A dry exterior trace still exerts the interior hydrostatic bed-step
   // force below. Only a genuinely dry physical owner has zero pressure.
   if(h>dry){
    double u=-un,wave=sqrt(gravity*h),invariant=u-2.*wave;
    if(u<=-wave)normal_speed=un;
    else if(u>=wave){
     if(stage_policy[b]){status[i]=5;continue;}
     normal_speed=0.;tx=ty=0.;
    }
    else{
     double target=fmax(forcing[nb+b]-bed[i],0.),ub;
     if(target<=dry || invariant<=-3.*sqrt(gravity*target)){
      target=(invariant/3.)*(invariant/3.)/gravity;ub=-sqrt(gravity*target);
     }else ub=invariant+2.*sqrt(gravity*target);
     if(ub>0. && stage_policy[b]){
      // One outgoing characteristic plus prescribed water level. Incoming
      // tangential momentum is explicit clear transverse flow, not donor
      // extrapolation from an interior outflow. Stage alone cannot specify
      // both incoming gravity characteristics.
      if(ub>=sqrt(gravity*target)){status[i]=5;continue;}
      h=target;normal_speed=-ub;tx=ty=0.;
     }
     else if(target<=dry || ub>0.){normal_speed=0.;tx=ty=0.;}
     else{h=target;normal_speed=-ub;}
    }
   }
  }else{status[i]=2;continue;}
  if(!isfinite(h) || !isfinite(normal_speed) || h<0. ||
     sqrt(normal_speed*normal_speed+tx*tx+ty*ty)>speed_limit){status[i]=3;continue;}
  double water=0.,fx=0.,fy=0.;
  if(h>dry){
   trace[i]=h;trace[nf+i]=h*(normal_speed*nx+tx);trace[2*nf+i]=h*(normal_speed*ny+ty);
   water=trace[nf+i]*nx+trace[2*nf+i]*ny;
   fx=water*trace[nf+i]/h+.5*gravity*h*h*nx;
   fy=water*trace[2*nf+i]/h+.5*gravity*h*h*ny;
  }
  // Interior-to-exterior hydrostatic integration, capped at the shoreline.
  double pressure=h0>dry?.5*gravity*(h0*h0-geometric*geometric):0.;
  // An unwetted prescribed-inlet segment admits no water. Its wet interior
  // still needs the closed-face pressure reaction; zero flux in all three
  // equations is not a wall and produces a spurious outward acceleration.
  if(m==9 && h<=dry)pressure=h0>dry?.5*gravity*h0*h0:0.;
  flux[i]=water;flux[nf+i]=fx+pressure*nx;flux[2*nf+i]=fy+pressure*ny;
 }
}
