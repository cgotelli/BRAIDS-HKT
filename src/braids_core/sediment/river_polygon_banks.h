// Actual-connectivity bank sharing, polygon wet activity and layered collapse.
// One source for OpenMP/CUDA; no atomics, no external-wall transfer.
#ifdef __CUDACC__
#define ENTRY extern "C" __global__
#define LOCAL __device__
#define LOOP(N) for(int i=blockDim.x*blockIdx.x+threadIdx.x;i<N;i+=blockDim.x*gridDim.x)
#else
#include <cmath>
#include <omp.h>
#define ENTRY extern "C" __declspec(dllexport)
#define LOCAL inline
#define LOOP(N) _Pragma("omp parallel for schedule(static)") for(int i=0;i<N;++i)
extern "C" __declspec(dllexport) void mobile_threads(int n){omp_set_dynamic(0);omp_set_num_threads(n);}
#endif

LOCAL void polygon_measure(int nv,const double*x,const double*y,const double*d,
 double s,double&area,double&volume){
 double px[128],py[128],ph[128];int m=0;
 for(int k=0;k<nv;++k){int j=(k+1)%nv;double ha=s-d[k],hb=s-d[j];
  if(ha>0.){px[m]=x[k];py[m]=y[k];ph[m++]=ha;}
  if((ha>0.)!=(hb>0.)){double t=ha/(ha-hb);
   px[m]=x[k]+t*(x[j]-x[k]);py[m]=y[k]+t*(y[j]-y[k]);ph[m++]=0.;}
 }
 area=volume=0.;
 for(int k=1;k+1<m;++k){double a=.5*((px[k]-px[0])*(py[k+1]-py[0])-(py[k]-py[0])*(px[k+1]-px[0]));
  area+=a;volume+=a*(ph[0]+ph[k]+ph[k+1])/3.;}
 area=fmax(area,0.);volume=fmax(volume,0.);
}

ENTRY void polygon_wet(int n,int nvert,double dry,double sedthr,const int*off,
 const double*xy,const double*area,const double*h,const double*z,const double*grad,
 double*out,int*active){
 LOOP(n){double x[64],y[64],d[64],low=1.e100,high=-1.e100;int nv=off[i+1]-off[i];
  for(int k=0;k<nv;++k){int v=off[i]+k;x[k]=xy[v];y[k]=xy[nvert+v];d[k]=grad[i]*x[k]+grad[n+i]*y[k];
   low=fmin(low,d[k]);high=fmax(high,d[k]);}
  double s=low,a=0.,v=0.;
  if(h[i]>0.){
   if(h[i]>=high)s=h[i];
   else{high+=h[i];for(int k=0;k<52;++k){s=.5*(low+high);polygon_measure(nv,x,y,d,s,a,v);
      if(v<h[i]*area[i])low=s;else high=s;}s=.5*(low+high);}
   polygon_measure(nv,x,y,d,s,a,v);
  }
  out[i]=z[i]+s;out[n+i]=fmin(1.,a/area[i]);out[3*n+i]=v/area[i]-h[i];
  double wetdepth=a>0.?h[i]*area[i]/a:0.;
  active[i]=(h[i]>dry&&wetdepth>sedthr)?1:0;
  polygon_measure(nv,x,y,d,s-sedthr,a,v);out[2*n+i]=fmin(1.,a/area[i]);
 }
}

ENTRY void graph_bank_propose(int n,int ng,int na,double theta,const int*off,const int*nb,
 const double*area,const double*z,const int*wet,const double*erosion,const double*frac,
 const double*stock,double*proposal){
 LOOP(n){double net=0.,weight=0.,amin=area[i];
  for(int k=0;k<ng;++k)net-=erosion[k*n+i];
  for(int e=off[i];e<off[i+1];++e)for(int k=0;k<ng;++k)proposal[k*na+e]=0.;
  if(!wet[i]||net<=0.)continue;
  for(int e=off[i];e<off[i+1];++e){int j=nb[e];if(wet[j]||z[j]<=z[i])continue;
   double sum=0.;for(int k=0;k<ng;++k)sum+=stock[k*n+j];if(sum<=0.)continue;
   amin=fmin(amin,area[j]);for(int k=0;k<ng;++k)weight+=frac[k*n+j];}
  if(weight<=1.e-7)continue;
  double request=theta*amin*net/weight;
  for(int e=off[i];e<off[i+1];++e){int j=nb[e];if(wet[j]||z[j]<=z[i])continue;
   double sum=0.;for(int k=0;k<ng;++k)sum+=stock[k*n+j];if(sum<=0.)continue;
   for(int k=0;k<ng;++k)proposal[k*na+e]=request*frac[k*n+j];}
 }
}
ENTRY void graph_bank_scale(int n,int ng,int na,const int*off,const int*reverse,
 const double*area,const double*stock,const double*proposal,double*scale){
 LOOP(n*ng){int c=i%n,k=i/n;double request=0.;
  for(int e=off[c];e<off[c+1];++e)request+=proposal[k*na+reverse[e]];
  double available=stock[i]*area[c];scale[i]=request>available&&request>0.?fmax(0.,available/request)*(1.-64.*2.2204460492503131e-16):1.;}
}
ENTRY void graph_bank_gather(int n,int ng,int na,const int*off,const int*nb,const int*reverse,
 const double*proposal,const double*scale,double*moved){
 LOOP(n*ng){int c=i%n,k=i/n;double v=0.;for(int e=off[c];e<off[c+1];++e)
  v+=proposal[k*na+e]*scale[k*n+nb[e]]-proposal[k*na+reverse[e]]*scale[i];moved[i]=v;}
}

// An immutable pre-screen for ONE axis/color, recomputed after prior transfers.
// A whole chain is inactive iff every pair is initially below threshold. Once
// any pair is active, ALL pairs must run in order: transfers can activate the
// next pair. Independent edge flags avoid shared writes and atomic reductions.
ENTRY void graph_scour_screen(int edges,int n,int ng,int nl,double slope,
 const int*left,const int*right,const double*distance,const double*rock,
 const double*mass,int*possible,const double*density,const double*sv){
 LOOP(edges){int a=left[i],b=right[i];double sa=0.,sb=0.;
  for(int l=0;l<nl;++l)for(int k=0;k<ng;++k){sa+=mass[(l*ng+k)*n+a]/density[k]/sv[l*n+a];sb+=mass[(l*ng+k)*n+b]/density[k]/sv[l*n+b];}
  // Keep exactly the original arithmetic, including its parenthesization.
  double dz=rock[a]+sa-rock[b]-sb;possible[i]=!(fabs(dz)<=slope*distance[i]);
 }
}

ENTRY void graph_scour(int chains,int n,int ng,int nl,double slope,const int*offset,
 const int*left,const int*right,const double*distance,const double*area,const double*rock,
 double*mass,double*moved,const double*density,int screened,const int*possible,double*sv){
 LOOP(chains){
  if(screened){bool active=false;for(int e=offset[i];e<offset[i+1];++e)if(possible[e]){active=true;break;}
   if(!active)continue;}
  for(int e=offset[i];e<offset[i+1];++e){int a=left[e],b=right[e];double sa=0.,sb=0.;
  for(int l=0;l<nl;++l)for(int k=0;k<ng;++k){sa+=mass[(l*ng+k)*n+a]/density[k]/sv[l*n+a];sb+=mass[(l*ng+k)*n+b]/density[k]/sv[l*n+b];}
  double dz=rock[a]+sa-rock[b]-sb;if(fabs(dz)<=slope*distance[e])continue;
  int donor=dz>0.?a:b,recipient=dz>0.?b:a;
  // Equalize without overshoot on unequal areas; agrees with legacy Scour
  // on equal-area pairs. Native .25*dz*(Aa+Ab) is unsafe for a tiny cut cell.
  double request=fabs(dz)/(1./area[a]+1./area[b]);
  double available=(dz>0.?sa:sb)*area[donor];
  double remaining=fmin(request,available*(1.-64.*2.2204460492503131e-16));
  for(int l=0;l<nl&&remaining>0.;++l){double layer=0.;
   for(int k=0;k<ng;++k)layer+=mass[(l*ng+k)*n+donor]/density[k]/sv[l*n+donor]*area[donor];
   if(layer<=0.)continue;double take=fmin(remaining,layer),share=fmin(1.,take/layer);
   double recipient_solid=0.;
   for(int k=0;k<ng;++k)recipient_solid+=mass[k*n+recipient]/density[k];
   double recipient_volume=recipient_solid/sv[recipient];
   double added_solid=0.;
   for(int k=0;k<ng;++k){int ix=(l*ng+k)*n+donor;double loss=mass[ix]*share,amount=loss*area[donor];
    mass[ix]-=loss;mass[k*n+recipient]+=amount/area[recipient];
    added_solid+=amount/density[k]/area[recipient];
    moved[k*n+donor]-=amount;moved[k*n+recipient]+=amount;}
   // Move the donor's bulk volume with its mass. Recompute physical packing
   // only in the subsequent single column rebalance, never per grain.
   if(sv[recipient]!=1. || sv[l*n+donor]!=1.)
     sv[recipient]=(recipient_solid+added_solid)/(recipient_volume+added_solid/sv[l*n+donor]);
   remaining=fmax(0.,remaining-take);
  }
 }}
}
