// Characteristic critical outlet on actual exterior faces. One resident team
// per section; logical 128-lane sums are identical on OpenMP and CUDA.
#ifdef __CUDACC__
#define ENTRY extern "C" __global__
#define LOCAL __device__ __forceinline__
#define LEADER if(threadIdx.x==0)
#define LANES for(int lane=threadIdx.x;lane<128;lane+=blockDim.x)
#define SYNC __syncthreads()
#define INF_VALUE __longlong_as_double(0x7ff0000000000000ULL)
#else
#include <cmath>
#include <omp.h>
using std::isfinite;
#define ENTRY extern "C" __declspec(dllexport)
#define LOCAL inline
#define LEADER _Pragma("omp single nowait")
#define LANES _Pragma("omp for schedule(static) nowait") for(int lane=0;lane<128;++lane)
#define SYNC _Pragma("omp barrier")
#define INF_VALUE HUGE_VAL
extern "C" __declspec(dllexport) void mobile_threads(int n){omp_set_dynamic(0);omp_set_num_threads(n);}
#endif

struct OutletWork {
 double part[3][128],lo,hi,eta,residual,area,width,rate;
 int count,changed,error,done,stagnant;
};

// A linear face extrapolation across a lower exterior bed is not resolvable
// when it implies a depth arbitrarily larger than the cell-average inventory.
// Bound downward bed extrapolation by local water depth in the thinnest
// wet-front cells. Blend continuously back to the unmodified face geometry
// between 0.1 and 1 mm of cell-average water. The bound tends to a dry face
// as the cell dries; it adds neither a water film nor water mass.
LOCAL double outlet_resolved_bed(int f,const int*owner,const double*q,const double*z,const double*bed){
 int c=owner[f];double h=fmax(q[c],0.);
 if(h>=1.e-3)return bed[f];
 double bounded=fmax(bed[f],z[c]-h),blend=fmin(1.,fmax(0.,(h-1.e-4)/9.e-4));
 blend=blend*blend*(3.-2.*blend);
 return bed[f]+(bounded-bed[f])*(1.-blend);
}

LOCAL void outlet_residual(OutletWork&w,int b,double eta,double gravity,double dry,
 const int*offset,const int*faces,const int*branch,const double*invariant,
 const double*bed,const double*length,const int*owner,const double*q,const double*z){
 LANES {
  double a=0.,width=0.,rate=0.;
  for(int k=offset[b]+lane;k<offset[b+1];k+=128){int f=faces[k];if(branch[f]!=1)continue;
   double h=fmax(eta-outlet_resolved_bed(f,owner,q,z,bed),0.);if(h<=dry)continue;
   a+=h*length[f];width+=length[f];rate+=h*(invariant[f]-2.*sqrt(gravity*h))*length[f];
  }
  w.part[0][lane]=a;w.part[1][lane]=width;w.part[2][lane]=rate;
 }
 SYNC;
 LEADER {
  double a=0.,width=0.,rate=0.;
  for(int lane=0;lane<128;++lane){a+=w.part[0][lane];width+=w.part[1][lane];rate+=w.part[2][lane];}
  w.area=a;w.width=width;w.rate=rate;w.residual=rate-(width>0.?sqrt(gravity*a*a*a/width):0.);
 }
 SYNC;
}

LOCAL void outlet_group(OutletWork&w,int b,int nf,int n,int nb,double gravity,double dry,double speed_limit,
 const int*offset,const int*faces,const int*owner,const double*q,const double*z,const double*bed,
 const double*normal,const double*length,int*branch,double*invariant,double*stage,double*metrics,
 double*trace,double*flux,int*face_status,int*section_status){
 const double tol=128.*2.2204460492503131e-16;
 LEADER {w.error=0;w.done=0;w.eta=0.;w.area=0.;w.width=0.;w.rate=0.;}
 LANES {
  for(int k=offset[b]+lane;k<offset[b+1];k+=128){int f=faces[k],c=owner[f];
   branch[f]=0;invariant[f]=0.;double h0=q[c],h=fmax(h0+z[c]-outlet_resolved_bed(f,owner,q,z,bed),0.);
   if(h0<=dry || h<=dry)continue;
   double un=(q[n+c]*normal[f]+q[2*n+c]*normal[nf+f])/h0,wave=sqrt(gravity*h);
   if(un < -tol*fmax(1.,fabs(un)))continue;
   if(un>wave*(1.+1e-10)){branch[f]=2;continue;}
   branch[f]=1;invariant[f]=un+2.*wave;
  }
 }
 SYNC;
 // Each repeated pass removes at least one controlled face. Supercritical
 // outflow is independent; it does not enter the controlled-section Fr.
 for(int active=0;active<=offset[b+1]-offset[b];++active){
  LANES {
   double lo=INF_VALUE,hi=-INF_VALUE,count=0.;
   for(int k=offset[b]+lane;k<offset[b+1];k+=128){int f=faces[k];if(branch[f]!=1)continue;
    lo=fmin(lo,outlet_resolved_bed(f,owner,q,z,bed));hi=fmax(hi,q[owner[f]]+z[owner[f]]);count+=1.;
   }
   w.part[0][lane]=lo;w.part[1][lane]=hi;w.part[2][lane]=count;
  }
  SYNC;
  LEADER {
   w.lo=INF_VALUE;w.hi=-INF_VALUE;w.count=0;
   for(int lane=0;lane<128;++lane){w.lo=fmin(w.lo,w.part[0][lane]);w.hi=fmax(w.hi,w.part[1][lane]);w.count+=int(w.part[2][lane]);}
   if(w.count)w.lo+=fmax(2.*dry,tol*fmax(1.,fabs(w.lo)));
   else{w.done=1;w.area=w.width=w.rate=0.;}
  }
  SYNC;
  if(w.done)break;
  outlet_residual(w,b,w.lo,gravity,dry,offset,faces,branch,invariant,bed,length,owner,q,z);
  if(w.residual<0.){
   // As in the native characteristic definition, unresolved controlled
   // depths close rather than adding an artificial film or halving forever.
   LANES {for(int k=offset[b]+lane;k<offset[b+1];k+=128){int f=faces[k];if(branch[f]==1)branch[f]=0;}}
   LEADER {w.done=1;w.area=w.width=w.rate=0.;w.eta=w.lo;}
   SYNC;break;
  }
  outlet_residual(w,b,w.hi,gravity,dry,offset,faces,branch,invariant,bed,length,owner,q,z);
  for(int j=0;j<60 && w.residual>0.;++j){
   LEADER {w.hi=w.lo+2.*(w.hi-w.lo);}
   SYNC;outlet_residual(w,b,w.hi,gravity,dry,offset,faces,branch,invariant,bed,length,owner,q,z);
  }
  if(w.residual>0. || !isfinite(w.residual)){
   LEADER {w.error=1;}
   SYNC;break;
  }
  for(int j=0;j<90;++j){
   LEADER {w.eta=w.lo+.5*(w.hi-w.lo);w.stagnant=(w.eta==w.lo || w.eta==w.hi);}
   SYNC;
   // The final midpoint and acceptance residual below are unchanged. Once
   // rounded to an endpoint, further bisections cannot improve the root.
   if(w.stagnant)break;
   outlet_residual(w,b,w.eta,gravity,dry,offset,faces,branch,invariant,bed,length,owner,q,z);
   LEADER {if(w.residual>0.)w.lo=w.eta;else w.hi=w.eta;}
   SYNC;
  }
  LEADER {w.eta=w.lo+.5*(w.hi-w.lo);}
  SYNC;
  LANES {
   int changed=0;
   for(int k=offset[b]+lane;k<offset[b+1];k+=128){int f=faces[k];if(branch[f]!=1)continue;
    double h=fmax(w.eta-outlet_resolved_bed(f,owner,q,z,bed),0.),un=invariant[f]-2.*sqrt(gravity*h);
    if(h<=dry || un < -tol){branch[f]=0;++changed;}
   }
   w.part[0][lane]=changed;
  }
  SYNC;
  LEADER {w.changed=0;for(int lane=0;lane<128;++lane)w.changed+=int(w.part[0][lane]);}
  SYNC;
  if(w.changed)continue;
  outlet_residual(w,b,w.eta,gravity,dry,offset,faces,branch,invariant,bed,length,owner,q,z);
  LEADER {
   double qc=w.width>0.?sqrt(gravity*w.area*w.area*w.area/w.width):0.;
   if(w.area<=0. || w.width<=0. || !isfinite(w.residual) || fabs(w.residual)>1e-10*fmax(qc,1.))w.error=2;
   w.done=1;
  }
  SYNC;break;
 }
 LEADER {
  if(!w.done && !w.error)w.error=3;
  section_status[b]=w.error;stage[b]=w.eta;metrics[b]=w.area;metrics[nb+b]=w.width;metrics[2*nb+b]=w.rate;
 }
 SYNC;
 if(w.error)return;
 LANES {
  for(int k=offset[b]+lane;k<offset[b+1];k+=128){int f=faces[k],c=owner[f];
   double h0=q[c],geom=fmax(h0+z[c]-outlet_resolved_bed(f,owner,q,z,bed),0.),h=geom,nx=normal[f],ny=normal[nf+f];
   double ux=0.,uy=0.;if(h0>dry){ux=q[n+c]/h0;uy=q[2*n+c]/h0;}
   else h=0.;
   double un=ux*nx+uy*ny,tx=ux-un*nx,ty=uy-un*ny;
   if(branch[f]==0)un=0.;
   else if(branch[f]==1){h=fmax(w.eta-outlet_resolved_bed(f,owner,q,z,bed),0.);un=invariant[f]-2.*sqrt(gravity*h);}
   if(!isfinite(h) || !isfinite(un) || sqrt(un*un+tx*tx+ty*ty)>speed_limit){face_status[f]=3;continue;}
   double water=0.,fx=0.,fy=0.;
   if(h>dry){
    trace[f]=h;trace[nf+f]=h*(un*nx+tx);trace[2*nf+f]=h*(un*ny+ty);
    water=trace[nf+f]*nx+trace[2*nf+f]*ny;
    fx=water*trace[nf+f]/h+.5*gravity*h*h*nx;fy=water*trace[2*nf+f]/h+.5*gravity*h*h*ny;
   }
   double pressure=h0>dry?.5*gravity*(h0*h0-geom*geom):0.;
   flux[f]=water;flux[nf+f]=fx+pressure*nx;flux[2*nf+f]=fy+pressure*ny;
  }
 }
 SYNC;
}

ENTRY void exterior_critical_outlet(int nf,int n,int nb,double gravity,double dry,double speed_limit,
 const int*control,const int*offset,const int*faces,const int*owner,const double*q,const double*z,const double*bed,
 const double*normal,const double*length,int*branch,double*invariant,double*stage,double*metrics,
 double*trace,double*flux,int*face_status,int*section_status){
#ifdef __CUDACC__
 int b=blockIdx.x;if(b>=nb || !control[b])return;
 __shared__ OutletWork work;
 outlet_group(work,b,nf,n,nb,gravity,dry,speed_limit,offset,faces,owner,q,z,bed,normal,length,
  branch,invariant,stage,metrics,trace,flux,face_status,section_status);
#else
 for(int b=0;b<nb;++b){if(!control[b])continue;OutletWork work;
  #pragma omp parallel shared(work)
  {outlet_group(work,b,nf,n,nb,gravity,dry,speed_limit,offset,faces,owner,q,z,bed,normal,length,
    branch,invariant,stage,metrics,trace,flux,face_status,section_status);}
 }
#endif
}

// Outgoing Lax curve: rarefaction below hi, Rankine-Hugoniot shock above it.
// A characteristic invariant alone is not valid across a backwater shock.
LOCAL double weir_velocity(double h,double hi,double ui,double gravity){
 return h>hi ? ui-(h-hi)*sqrt(.5*gravity*(1./h+1./hi))
             : ui+2.*(sqrt(gravity*hi)-sqrt(gravity*h));
}

LOCAL void weir_residual(OutletWork&w,int b,double eta,double gravity,double dry,double crest,double factor,
 const int*offset,const int*faces,const int*branch,const double*invariant,const double*bed,const double*length,
 const int*owner,const double*q,const double*z){
 LANES {
  double area=0.,width=0.,rate=0.;
  for(int k=offset[b]+lane;k<offset[b+1];k+=128){int f=faces[k],c=owner[f];if(!branch[f])continue;
   double h=fmax(eta-bed[f],0.),hi=fmax(q[c]+z[c]-bed[f],0.);
   if(h<=dry)continue;
   double u=fmax(weir_velocity(h,hi,invariant[f],gravity),0.);
   if(u>0.){area+=h*length[f];width+=length[f];rate+=h*u*length[f];}
  }
  w.part[0][lane]=area;w.part[1][lane]=width;w.part[2][lane]=rate;
 }
 SYNC;
 LEADER {
  w.area=w.width=w.rate=0.;
  for(int lane=0;lane<128;++lane){w.area+=w.part[0][lane];w.width+=w.part[1][lane];w.rate+=w.part[2][lane];}
 }
 SYNC;
 LEADER {double head=fmax(eta-crest,0.);w.residual=w.rate-factor*head*sqrt(head);}
 SYNC;
}

LOCAL void weir_group(OutletWork&w,int b,int nf,int n,int nb,double gravity,double dry,double speed_limit,
 const double*parameters,const int*offset,const int*faces,const int*owner,const double*q,const double*z,const double*bed,
 const double*normal,const double*length,int*branch,double*invariant,double*stage,double*metrics,
 double*trace,double*flux,int*face_status,int*section_status){
 const double tol=128.*2.2204460492503131e-16,crest=parameters[b],factor=parameters[nb+b]*parameters[2*nb+b];
 LEADER {w.error=0;w.done=0;w.eta=0.;w.area=w.width=w.rate=0.;}
 LANES {
  double level=0.,width=0.,bad=0.;
  for(int k=offset[b]+lane;k<offset[b+1];k+=128){int f=faces[k],c=owner[f];
   branch[f]=0;invariant[f]=0.;double h0=q[c],h=fmax(h0+z[c]-bed[f],0.);
   if(h0<=dry || h<=dry)continue;
   double un=(q[n+c]*normal[f]+q[2*n+c]*normal[nf+f])/h0,wave=sqrt(gravity*h);
   branch[f]=1;invariant[f]=un;level+=(h0+z[c])*length[f];width+=length[f];
  }
  w.part[0][lane]=level;w.part[1][lane]=width;w.part[2][lane]=bad;
 }
 SYNC;
 LEADER {
  double level=0.,width=0.,bad=0.;
  for(int lane=0;lane<128;++lane){level+=w.part[0][lane];width+=w.part[1][lane];bad+=w.part[2][lane];}
  if(width==0.)w.done=1;
 }
 SYNC;
 if(!w.error && w.done){
  LANES {for(int k=offset[b]+lane;k<offset[b+1];k+=128)branch[faces[k]]=0;}
  SYNC;
 }
 if(!w.error && !w.done){
  LANES {
   double lo=-INF_VALUE,hi=-INF_VALUE;
   for(int k=offset[b]+lane;k<offset[b+1];k+=128){int f=faces[k];if(branch[f]!=1)continue;
    // Start on the subcritical (decreasing-discharge) branch, independently
    // of initial Froude number. A high crest can send a shock upstream.
    double h0=fmax(q[owner[f]]+z[owner[f]]-bed[f],0.),u=invariant[f];
    double left=0.,right=fmax(h0,1.);
    for(int it=0;it<60 && weir_velocity(right,h0,u,gravity)>sqrt(gravity*right);++it)right*=2.;
    for(int it=0;it<70;++it){double mid=.5*(left+right);
     if(weir_velocity(mid,h0,u,gravity)>sqrt(gravity*mid))left=mid;else right=mid;}
    lo=fmax(lo,bed[f]+right);
    hi=fmax(hi,q[owner[f]]+z[owner[f]]);
   }
   w.part[0][lane]=lo;w.part[1][lane]=hi;
  }
  SYNC;
  LEADER {
   w.lo=-INF_VALUE;w.hi=-INF_VALUE;
   for(int lane=0;lane<128;++lane){w.lo=fmax(w.lo,w.part[0][lane]);w.hi=fmax(w.hi,w.part[1][lane]);}
   w.lo=fmax(w.lo,crest);
   w.hi=fmax(w.hi,w.lo+1.);
  }
  SYNC;
  weir_residual(w,b,w.lo,gravity,dry,crest,factor,offset,faces,branch,invariant,bed,length,owner,q,z);
  LEADER {if(w.residual<0. || !isfinite(w.residual))w.error=1;}
  SYNC;
  if(!w.error){
   weir_residual(w,b,w.hi,gravity,dry,crest,factor,offset,faces,branch,invariant,bed,length,owner,q,z);
   for(int j=0;j<60 && w.residual>0.;++j){
    LEADER {w.hi=w.lo+2.*(w.hi-w.lo);}
    SYNC;weir_residual(w,b,w.hi,gravity,dry,crest,factor,offset,faces,branch,invariant,bed,length,owner,q,z);
   }
   LEADER {if(w.residual>0. || !isfinite(w.residual))w.error=1;}
   SYNC;
  }
  if(!w.error){
   for(int j=0;j<90;++j){
    LEADER {w.eta=w.lo+.5*(w.hi-w.lo);w.stagnant=(w.eta==w.lo || w.eta==w.hi);}
    SYNC;
    if(w.stagnant)break;
    weir_residual(w,b,w.eta,gravity,dry,crest,factor,offset,faces,branch,invariant,bed,length,owner,q,z);
    LEADER {if(w.residual>0.)w.lo=w.eta;else w.hi=w.eta;}
    SYNC;
   }
   LEADER {w.eta=w.lo+.5*(w.hi-w.lo);}
   SYNC;weir_residual(w,b,w.eta,gravity,dry,crest,factor,offset,faces,branch,invariant,bed,length,owner,q,z);
   LEADER {
    double head=fmax(w.eta-crest,0.),requested=factor*head*sqrt(head);
    if(!isfinite(w.residual) || fabs(w.residual)>1e-10*fmax(requested,1.))w.error=2;
   }
   SYNC;
   LANES {
    double bad=0.;
    for(int k=offset[b]+lane;k<offset[b+1];k+=128){int f=faces[k];if(branch[f]!=1)continue;
     int c=owner[f];double hi=fmax(q[c]+z[c]-bed[f],0.),h=fmax(w.eta-bed[f],0.);
     double u=weir_velocity(h,hi,invariant[f],gravity);
     if(h<=dry || u<=0.){branch[f]=0;continue;}
     // No prescribed subcritical boundary may propagate a downstream-only
     // shock into the domain. Report an uncontrolled/too-low structure.
     if(h>hi && invariant[f]-h*sqrt(.5*gravity*(1./h+1./hi))>tol)bad+=1.;
    }
    w.part[0][lane]=bad;
   }
   SYNC;
   LEADER {for(int lane=0;lane<128;++lane)if(w.part[0][lane]>0.)w.error=5;}
   SYNC;
  }
 }
 LEADER {section_status[b]=w.error;stage[b]=w.eta;metrics[b]=w.area;metrics[nb+b]=w.width;metrics[2*nb+b]=w.rate;}
 SYNC;if(w.error)return;
 LANES {
  for(int k=offset[b]+lane;k<offset[b+1];k+=128){int f=faces[k],c=owner[f];
   double h0=q[c],geom=fmax(h0+z[c]-bed[f],0.),h=geom,nx=normal[f],ny=normal[nf+f];
   double ux=0.,uy=0.;if(h0>dry){ux=q[n+c]/h0;uy=q[2*n+c]/h0;}else h=0.;
   double un=ux*nx+uy*ny,tx=ux-un*nx,ty=uy-un*ny;
   if(branch[f]==0){
    // Reflect a blocked stream, including its stagnation/shock pressure.
    // Merely zeroing velocity at the interior depth loses that pressure.
    if(geom>dry && h0>dry && un!=0.){
     double lo=0.,hi=fmax(geom,1.);
     for(int it=0;it<60 && weir_velocity(hi,geom,un,gravity)>0.;++it)hi*=2.;
     for(int it=0;it<80;++it){double mid=.5*(lo+hi);
      if(weir_velocity(mid,geom,un,gravity)>0.)lo=mid;else hi=mid;}
     h=.5*(lo+hi);
    }
    un=0.;
   }else{h=fmax(w.eta-bed[f],0.);un=weir_velocity(h,geom,invariant[f],gravity);}
   if(!isfinite(h) || !isfinite(un) || sqrt(un*un+tx*tx+ty*ty)>speed_limit){face_status[f]=3;continue;}
   double water=0.,fx=0.,fy=0.;
   if(h>dry){
    trace[f]=h;trace[nf+f]=h*(un*nx+tx);trace[2*nf+f]=h*(un*ny+ty);
    water=trace[nf+f]*nx+trace[2*nf+f]*ny;
    fx=water*trace[nf+f]/h+.5*gravity*h*h*nx;fy=water*trace[2*nf+f]/h+.5*gravity*h*h*ny;
   }
   double pressure=h0>dry?.5*gravity*(h0*h0-geom*geom):0.;
   flux[f]=water;flux[nf+f]=fx+pressure*nx;flux[2*nf+f]=fy+pressure*ny;
  }
 }
 SYNC;
}

ENTRY void exterior_free_weir(int nf,int n,int nb,double gravity,double dry,double speed_limit,
 const int*control,const double*parameters,const int*offset,const int*faces,const int*owner,const double*q,const double*z,const double*bed,
 const double*normal,const double*length,int*branch,double*invariant,double*stage,double*metrics,
 double*trace,double*flux,int*face_status,int*section_status){
#ifdef __CUDACC__
 int b=blockIdx.x;if(b>=nb || !control[b])return;__shared__ OutletWork work;
 weir_group(work,b,nf,n,nb,gravity,dry,speed_limit,parameters,offset,faces,owner,q,z,bed,normal,length,
  branch,invariant,stage,metrics,trace,flux,face_status,section_status);
#else
 for(int b=0;b<nb;++b){if(!control[b])continue;OutletWork work;
  #pragma omp parallel shared(work)
  {weir_group(work,b,nf,n,nb,gravity,dry,speed_limit,parameters,offset,faces,owner,q,z,bed,normal,length,
    branch,invariant,stage,metrics,trace,flux,face_status,section_status);}
 }
#endif
}
