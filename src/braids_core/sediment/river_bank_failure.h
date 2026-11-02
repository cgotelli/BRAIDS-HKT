// Common CPU/CUDA retained undercut and finite, geometry-bounded block release.
#ifdef __CUDACC__
#define ENTRY extern "C" __global__
#define LOCAL __device__
#define LOOP(N) for(int c=blockDim.x*blockIdx.x+threadIdx.x;c<N;c+=blockDim.x*gridDim.x)
#else
#include <cmath>
#include <omp.h>
using std::isfinite;
#define ENTRY extern "C" __declspec(dllexport)
#define LOCAL inline
#define LOOP(N) _Pragma("omp parallel for schedule(static)") for(int c=0;c<N;++c)
extern "C" __declspec(dllexport) void mobile_threads(int n){omp_set_dynamic(0);omp_set_num_threads(n);}
#endif
#define BF_EPS 2.2204460492503131e-16

ENTRY void failure_bed(int n,int ng,int nl,const double*mass,const double*sv,
 const double*density,const double*rock,const double*area,double*bed,double*volume,double*maxsv){
 LOOP(n){double v=0.,s=0.;for(int l=0;l<nl;++l){s=fmax(s,sv[l*n+c]);
   for(int k=0;k<ng;++k)v+=mass[(l*ng+k)*n+c]/density[k]/sv[l*n+c];}
  bed[c]=rock[c]+v;volume[c]=v*area[c];maxsv[c]=s;
 }
}

LOCAL int failure_exposed(int c,const int*off,const int*candidates,const int*toe,
 const double*bed,const double*depth,const double*stock,double dry){
 int t=toe[c];if(t<0)return -1;double base=bed[t]+depth[t];
 for(int e=off[c];e<off[c+1];++e){int s=candidates[e];
  if(depth[s]<=dry&&bed[s]>base&&stock[s]>0.)return s;}
 return -1;
}

LOCAL void failure_retire(int c,int n,double*history){
 for(int k=0;k<4;++k)history[k*n+c]=0.;
}

ENTRY void failure_history(int n,double dry,const int*off,const int*candidates,const int*toe,
 const int*boff,const int*bfaces,const double*surface,const double*length,const double*actual,
 const double*area,const double*bed,const double*depth,const double*stock,
 double*history,int*exposed,int*status){
 LOOP(n){status[c]=0;exposed[c]=-1;if(toe[c]<0)continue;
  double cumulative=history[4*n+c]+actual[c]/area[c];
  double denom=0.,width=0.;
  for(int e=boff[c];e<boff[c+1];++e){int f=bfaces[e];if(surface[f]>0.){denom+=surface[f];width+=length[f];}}
  if(!isfinite(cumulative)||!isfinite(denom)||!isfinite(width)||(actual[c]>0.&&denom<=0.)){status[c]=1;continue;}
  history[4*n+c]=cumulative;
  int s=failure_exposed(c,off,candidates,toe,bed,depth,stock,dry);exposed[c]=s;
  if(s<0){failure_retire(c,n,history);continue;}
  // A dry toe alone is not a vanished bank. Retain its existing geometry.
  if(denom<=0.)continue;
  history[c]+=actual[c]/denom;
  history[n+c]=bed[s]-bed[toe[c]];history[2*n+c]=depth[toe[c]];history[3*n+c]=width;
  if(!isfinite(history[c]))status[c]=1;
 }
}

ENTRY void failure_propose(int n,double critical,const int*toe,const int*exposed,
 const double*area,const double*bed,const double*depth,const double*stock,
 const double*history,double*proposal,double*above){
 LOOP(n){proposal[c]=0.;above[c]=0.;int s=exposed[c],t=toe[c];
  if(t<0||s<0||history[c]+64.*BF_EPS*critical<critical)continue;
  double height=fmax(bed[s]-bed[t]-depth[t],0.),width=history[3*n+c];
  if(height<=0.||width<=0.)continue;
  double cap=fmin(stock[s],area[s]*height),cross=width*height;
  proposal[c]=history[c]>=cap/cross?cap:history[c]*cross;above[c]=height;
 }
}

ENTRY void failure_shared_limit(int n,const int*off,const int*donors,const int*selected,
 const double*area,const double*proposal,const double*above,const double*sv,const double*maxsv,
 double*scale,int*status){
 LOOP(n){double incoming=0.,factor=1.;status[c]=0;
  for(int e=off[c];e<off[c+1];++e){int d=donors[e],s=selected[d];
   if(proposal[d]>0.)incoming+=proposal[d]*maxsv[s]/sv[c];}
  for(int e=off[c];e<off[c+1];++e){int d=donors[e],s=selected[d];if(proposal[d]<=0.)continue;
   double closure=proposal[d]/area[s]+incoming/area[c];factor=fmin(factor,above[d]/closure);}
  scale[c]=fmax(0.,factor);
  if(!isfinite(incoming)||!isfinite(factor))status[c]=1;
 }
}

ENTRY void failure_extract(int n,int ng,int nl,const int*selected,const int*toe,
 const double*area,const double*density,const double*sv,const double*proposal,
 const double*above,const double*scale,double*work,double*history,long long*count,
 double*removed,double*actual,double*moved,int*status){
 LOOP(n){status[c]=0;actual[c]=0.;for(int k=0;k<ng;++k)removed[k*n+c]=0.;
  int s=selected[c],t=toe[c];if(s<0||t<0||proposal[c]<=0.)continue;
  double remaining=proposal[c]*scale[t],taken=0.;
  for(int l=0;l<nl;++l){double v=0.;for(int k=0;k<ng;++k)v+=work[(l*ng+k)*n+s]*area[s]/density[k]/sv[l*n+s];
   double take=fmin(remaining,v),ratio=v>0.?fmin(1.,take/v):0.;
   for(int k=0;k<ng;++k){int i=(l*ng+k)*n+s;double m=work[i]*ratio;work[i]-=m;
    removed[k*n+c]+=m*area[s];moved[k*n+s]-=m*area[s];}
   remaining=fmax(0.,remaining-take);taken+=take;
  }
  actual[c]=taken;if(taken<=0.)continue;
  history[5*n+c]+=taken/area[c];
  history[c]=fmax(0.,history[c]-taken/(history[3*n+c]*above[c]));
  if(count[c]==9223372036854775807LL||!isfinite(history[c])||!isfinite(history[5*n+c]))status[c]=1;
  else ++count[c];
 }
}

ENTRY void failure_deposit(int n,int ng,const int*off,const int*donors,
 const double*removed,const double*area,double*work,double*moved){
 LOOP(n){for(int k=0;k<ng;++k){double gain=0.;
  for(int e=off[c];e<off[c+1];++e){double m=removed[k*n+donors[e]];gain+=m;work[k*n+c]+=m/area[c];}
  moved[k*n+c]+=gain;
 }}
}

ENTRY void failure_reconcile(int n,double dry,const int*off,const int*candidates,const int*toe,
 const double*bed,const double*depth,const double*stock,double*history){
 LOOP(n){if(toe[c]>=0&&failure_exposed(c,off,candidates,toe,bed,depth,stock,dry)<0)failure_retire(c,n,history);}
}
