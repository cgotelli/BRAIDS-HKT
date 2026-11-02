// Standalone Jiang-Cao single-grain inventory/exchange qualification.
// One source for native OpenMP and CUDA; not a public river execution route.
#ifdef __CUDACC__
#define JE_ENTRY extern "C" __global__
#define JE_LOCAL __device__ __forceinline__
#define JE_LOOP(N) for(int i=blockDim.x*blockIdx.x+threadIdx.x;i<N;i+=blockDim.x*gridDim.x)
#else
#include <cmath>
#include <omp.h>
using std::isfinite;
#define JE_ENTRY extern "C" __declspec(dllexport)
#define JE_LOCAL static inline
#define JE_LOOP(N) _Pragma("omp parallel for schedule(static)") for(int i=0;i<N;++i)
extern "C" __declspec(dllexport) void mobile_threads(int n){omp_set_dynamic(0);omp_set_num_threads(n);}
#endif

JE_LOCAL void je_bad(int n,int i,int rows,double*out,int*status,int code){
 status[i]=code;for(int k=0;k<rows;++k)out[k*n+i]=nan("");
}

// Stable time integral for dh/dt=k*(H-h)/(h-shift).
// y is the fraction of the distance from h0 to equilibrium H traversed.
JE_LOCAL double je_elapsed(double y,double h0,double H,double shift,double k){
 if(y>=1.)return 1.e300;
 double remainder;
 if(y<1.e-4)remainder=y*y*(.5+y*(1./3.+y*(.25+y*(.2+y/6.))));
 else remainder=-log1p(-y)-y;
 return ((h0-shift)*y+(H-shift)*remainder)/k;
}

JE_LOCAL double je_relax(double h0,double H,double W,double omega,double B,double ce,double dt){
 double start=h0,remain=dt;
 // At most one crossing of c=B/2 (h=2W). alpha then switches continuously.
 for(int segment=0;segment<2;++segment){
  if(start==H||remain==0.)return start;
  const bool dense=start>2.*W||(start==2.*W&&H>start);
  const double shift=dense?W:0.,k=(dense?1.:2.)*omega*(B-ce)/B;
  const double cross=(2.*W-start)/(H-start);
  if(cross>0.&&cross<1.){
   const double tcross=je_elapsed(cross,start,H,shift,k);
   if(tcross<remain){start=2.*W;remain-=tcross;continue;}
  }
  double lo=0.,hi=1.;
  for(int j=0;j<56;++j){
   const double y=.5*(lo+hi);
   if(je_elapsed(y,start,H,shift,k)>remain)hi=y;else lo=y;
  }
  return start+(.5*(lo+hi))*(H-start);
 }
 return start;
}

// Integral of (h-shift)/(A*(a*h+b)), using displacement rather than the
// equilibrium H=b/(B-ce). H is singular at ce=B, but the source ODE is not.
JE_LOCAL double je_growth_time(double d,double h,double shift,double D,double a,double A){
 const double r=a*(d/D);
 if(r<=-1.)return 1./0.;
 double L,M;
 if(fabs(r)<1.e-3){
  L=1.+r*(-.5+r*(1./3.+r*(-.25+r*(.2+r*(-1./6.+r/7.)))));
  M=.5+r*(-1./3.+r*(.25+r*(-.2+r*(1./6.+r*(-1./7.+r/8.)))));
 }else{L=log1p(r)/r;M=(1.-L)/r;}
 return (d/D)*((h-shift)*L+d*M)/A;
}

JE_LOCAL double je_growth(double h,double W,double omega,double B,double ce,
 double dt,double stock,int*exhausted){
 double start=h,remain=dt,available=stock/B;
 const double a=ce-B,b=B*W;
 // Finite-stock monotone erosion. alpha changes once at c=B/2.
 for(int segment=0;segment<2;++segment){
  if(available==0.){*exhausted=1;return start;}
  if(remain==0.)return start;
  const bool dense=start>=2.*W;
  const double shift=dense?W:0.,A=(dense?1.:2.)*omega/B;
  const double D=a*start+b;
  if(!(D>0.))return start;
  double upper=fmin(available,remain*(A*(D/(start-shift))));
  if(a<0.)upper=fmin(upper,-D/a);
  const double cross=2.*W-start;
  if(!dense&&cross>0.&&cross<upper){
   const double t=je_growth_time(cross,start,shift,D,a,A);
   if(t<remain){start=2.*W;available-=cross;remain-=t;continue;}
  }
  if(je_growth_time(upper,start,shift,D,a,A)<=remain){
   if(upper==available)*exhausted=1;
   return start+upper;
  }
  double lo=0.,hi=upper;
  for(int j=0;j<56;++j){
   const double d=.5*(lo+hi);
   if(je_growth_time(d,start,shift,D,a,A)>remain)hi=d;else lo=d;
  }
  return start+.5*(lo+hi);
 }
 return start;
}

JE_ENTRY void jc_exchange(int n,const double*p,double dt,const double*state,
 const double*ce,double*out,int*status){
 JE_LOOP(n){
  // state: mixture h, hu, hv; mobile solid m=hc; erodible solid stock; bed z.
  const double h=state[i],hu=state[n+i],hv=state[2*n+i],m=state[3*n+i];
  const double stock=state[4*n+i],z=state[5*n+i],target=ce[i];
  const double B=p[0],omega=p[1],rw=p[2],rs=p[3];
  if(!isfinite(h)||!isfinite(hu)||!isfinite(hv)||!isfinite(m)||!isfinite(stock)
     ||!isfinite(z)||!isfinite(target)||h<0.||m<0.||stock<0.||target<0.
     ||(h==0.&&(m!=0.||hu!=0.||hv!=0.||target!=0.))||(h>0.&&m>=B*h)){
   je_bad(n,i,8,out,status,3);continue;
  }
  status[i]=0;
  for(int k=0;k<6;++k)out[k*n+i]=state[k*n+i];
  out[6*n+i]=0.;out[7*n+i]=h>0.?fmin(2.,B/(m/h)):2.;
  if(h==0.||dt==0.)continue;
  const double W=h-m/B;
  if(W<=0.){je_bad(n,i,8,out,status,4);continue;}
  double hn;int exhausted=0;
  if(target>=B*(1.-1.e-4)&&target>m/h)
   hn=je_growth(h,W,omega,B,target,dt,stock,&exhausted);
  else{
   const double H=B*W/(B-target);
   if(!isfinite(H)){je_bad(n,i,8,out,status,4);continue;}
   hn=je_relax(h,H,W,omega,B,target,dt);
  }
  // A finite bed stops NET erosion, but deposition can build new erodible stock.
  double delta=B*(hn-h);
  if(exhausted||(delta>=stock&&target>m/h)){delta=stock;status[i]=1;}
  if(delta< -m)delta=-m; // only endpoint floating-point roundoff, not a negative-state repair
  hn=h+delta/B;
  const double mn=m+delta,sn=stock-delta;
  const double mass0=rw*h+(rs-rw)*m,mass1=rw*hn+(rs-rw)*mn;
  const double factor=(hn/h)*(mass0/mass1);
  out[i]=hn;out[n+i]=hu*factor;out[2*n+i]=hv*factor;
  out[3*n+i]=mn;out[4*n+i]=sn;out[5*n+i]=z-delta/B;
  out[6*n+i]=delta;
  out[7*n+i]=mn>0.?fmin(2.,B*hn/mn):2.;
  if(hn<=0.||mn<0.||mn>=B*hn||sn<0.){je_bad(n,i,8,out,status,4);continue;}
  for(int k=0;k<8;++k)if(!isfinite(out[k*n+i])){je_bad(n,i,8,out,status,4);break;}
 }
}

// All face rates are owner-outward. feed is an inward solid-volume rate (m3/s).
JE_ENTRY void jc_inventory_faces(int nf,int n,const int*owner,const int*neighbour,
 const int*kind,const double*length,const double*vn,const double*feed,
 const double*volume_flux,const double*mobile,double*solid_flux){
 JE_LOOP(nf){
  int o=owner[i],j=neighbour[i];double a=vn[i]*length[i];
  if(j>=0)solid_flux[i]=a*(a>=0.?mobile[o]:mobile[j]);
  else if(kind[i]==1)solid_flux[i]=0.;
  else if(kind[i]==2&&volume_flux[i]<0.)solid_flux[i]=-feed[i];
  else solid_flux[i]=fmax(a,0.)*mobile[o]; // export only, including reversing outlets
 }
}

JE_ENTRY void jc_inventory_cells(int n,double dt,double B,const double*area,
 const int*offset,const int*faces,const int*owner,const int*neighbour,const int*kind,
 const double*length,const double*vn,const double*volume_flux,const double*solid_flux,
 const double*h,const double*mobile,double*out,int*status){
 JE_LOOP(n){
  double dv=0.,dm=0.,outgoing=0.;status[i]=0;
  for(int a=offset[i];a<offset[i+1];++a){
   int f=faces[a];const double sign=owner[f]==i?1.:-1.;
   dv+=sign*volume_flux[f];dm+=sign*solid_flux[f];
   if(neighbour[f]>=0||kind[f]!=1){
    // A feed replaces the advective law only for actual hydraulic inflow.
    if(!(neighbour[f]<0&&kind[f]==2&&volume_flux[f]<0.))
     outgoing+=fmax(sign*vn[f]*length[f],0.);
   }
  }
  const double cfl=dt*outgoing/area[i],hn=h[i]-dt*dv/area[i],mn=mobile[i]-dt*dm/area[i];
  if(cfl>1.+1.e-13){je_bad(n,i,3,out,status,5);continue;}
  if(!isfinite(hn)||!isfinite(mn)||hn<0.||mn<0.||(hn>0.&&mn>=B*hn)||(hn==0.&&mn!=0.)){
   je_bad(n,i,3,out,status,6);continue;
  }
  out[i]=hn;out[n+i]=mn;out[2*n+i]=cfl;
 }
}
