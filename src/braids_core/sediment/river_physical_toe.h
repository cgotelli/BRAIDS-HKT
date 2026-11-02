// Accepted-Exner toe supply on actual internal faces. Shared OpenMP/CUDA.
// No independent inventory, capacity proxy, exterior transfer or atomics.
#ifdef __CUDACC__
#define ENTRY extern "C" __global__
#define LOOP(N) for(int c=blockDim.x*blockIdx.x+threadIdx.x;c<N;c+=blockDim.x*gridDim.x)
#else
#include <cmath>
#include <omp.h>
#define ENTRY extern "C" __declspec(dllexport)
#define LOOP(N) _Pragma("omp parallel for schedule(static)") for(int c=0;c<N;++c)
extern "C" __declspec(dllexport) void mobile_threads(int n){omp_set_dynamic(0);omp_set_num_threads(n);}
#endif

ENTRY void toe_state(int n,int ng,int nl,const double*mass,const double*sv,
 const double*density,const double*rock,const double*area,double*bed,double*stock,double*maxsv){
 LOOP(n){double v=0.,packing=0.;
  for(int l=0;l<nl;++l){packing=fmax(packing,sv[l*n+c]);
   for(int k=0;k<ng;++k)v+=mass[(l*ng+k)*n+c]/density[k]/sv[l*n+c];}
  bed[c]=rock[c]+v;stock[c]=v*area[c];maxsv[c]=packing;
 }
}

ENTRY void toe_request(int n,int ng,double fraction,double dry,const int*off,
 const int*faces,const int*bank,const double*length,const double*area,
 const double*delta,const double*density,const double*surface_sv,
 const double*z,const double*h,double*demand,double*request,double*surface){
 LOOP(n){double net=0.,total=0.;int last=-1;
  // Sum signed changes before taking the positive part. Replacement of one
  // class by equal-volume deposition of another is sorting, not toe scour.
  for(int k=0;k<ng;++k)net-=delta[k*n+c]/density[k];
  demand[c]=off[c]<off[c+1]?fmax(net,0.)*area[c]/surface_sv[c]:0.;
  for(int e=off[c];e<off[c+1];++e){int f=faces[e];
   double wet=h[c]>dry?fmin(h[c],fmax(z[bank[f]]-z[c],0.)):0.;
   surface[f]=wet*length[f];request[f]=0.;
   if(wet>0.){total+=length[f];last=f;}
  }
  double remaining=fraction*demand[c];
  if(last<0||remaining<=0.)continue;
  for(int e=off[c];e<off[c+1];++e){int f=faces[e];if(surface[f]<=0.)continue;
   double v=f==last?remaining:fmin(remaining,fraction*demand[c]*length[f]/total);
   request[f]=v;remaining=fmax(0.,remaining-v);
  }
 }
}

ENTRY void toe_stock_limit(int n,const int*off,const int*faces,const double*stock,double*request){
 LOOP(n){double total=0.;for(int e=off[c];e<off[c+1];++e)total+=request[faces[e]];
  double scale=total>stock[c]?stock[c]/total:1.;
  if(scale<1.)for(int e=off[c];e<off[c+1];++e)request[faces[e]]*=scale;
 }
}

ENTRY void toe_totals(int n,const int*boff,const int*bfaces,const int*toff,const int*tfaces,
 const int*bank,const double*request,const double*maxsv,const double*sv,
 double*outgoing,double*incoming){
 LOOP(n){double out=0.,in=0.;
  for(int e=boff[c];e<boff[c+1];++e)out+=request[bfaces[e]];
  // Bound recipient raising with the maximum donor-layer packing. Removing
  // a bulk volume from loose and dense layers does not deposit that same
  // bulk volume at a differently packed recipient.
  for(int e=toff[c];e<toff[c+1];++e){int f=tfaces[e];in+=request[f]*maxsv[bank[f]]/sv[c];}
  outgoing[c]=out;incoming[c]=in;
 }
}

ENTRY void toe_geometry_scales(int n,const int*bank,const int*toe,const int*boff,const int*bfaces,
 const int*toff,const int*tfaces,
 const double*area,const double*length,const double*surface,const double*bed,
 const double*outgoing,const double*incoming,double*donor_scale,double*toe_scale){
 LOOP(n){double ds=1.,ts=1.,wetarea=0.,width=0.;
  for(int e=boff[c];e<boff[c+1];++e){int f=bfaces[e],t=toe[f];if(surface[f]<=0.)continue;
   double closure=outgoing[c]/area[c]+incoming[t]/area[t];
   if(closure>0.)ds=fmin(ds,fmax(bed[c]-bed[t],0.)/closure);
   wetarea+=surface[f];width+=length[f];}
  // A donor's plan area belongs to the whole column, not to each face.
  if(outgoing[c]>0.&&width>0.)ds=fmin(ds,area[c]*(wetarea/width)/outgoing[c]);
  for(int e=toff[c];e<toff[c+1];++e){int f=tfaces[e],b=bank[f];if(surface[f]<=0.)continue;
   double closure=outgoing[b]/area[b]+incoming[c]/area[c];
   if(closure>0.)ts=fmin(ts,fmax(bed[b]-bed[c],0.)/closure);}
  donor_scale[c]=fmax(0.,ds);toe_scale[c]=fmax(0.,ts);
 }
}

ENTRY void toe_limit_requests(int nf,const int*bank,const int*toe,
 const double*donor_scale,const double*toe_scale,double*request){
 LOOP(nf){if(bank[c]>=0)request[c]*=fmin(donor_scale[bank[c]],toe_scale[toe[c]]);}
}

ENTRY void toe_extract(int n,int ng,int nl,const int*off,const int*faces,
 const double*request,const double*area,const double*density,const double*sv,
 const double*mass,double*work,double*removed,double*volume,double*total_request){
 LOOP(n){double remaining=0.,taken=0.;
  for(int e=off[c];e<off[c+1];++e)remaining+=request[faces[e]];
  total_request[c]=remaining;for(int k=0;k<ng;++k)removed[k*n+c]=0.;
  for(int l=0;l<nl;++l){double v=0.;
   for(int k=0;k<ng;++k)v+=mass[(l*ng+k)*n+c]*area[c]/density[k]/sv[l*n+c];
   double take=fmin(remaining,v),ratio=v>0.?fmin(1.,take/v):0.;
   for(int k=0;k<ng;++k){int i=(l*ng+k)*n+c;double debit=mass[i]*ratio;
    work[i]=mass[i]-debit;removed[k*n+c]+=debit*area[c];}
   taken+=take;remaining=fmax(0.,remaining-take);
  }
  volume[c]=taken;
 }
}

ENTRY void toe_route(int n,int nf,int ng,const int*off,const int*faces,const int*owner,
 const double*request,const double*total,const double*removed,double*face_mass){
 LOOP(n){int last=-1;
  for(int e=off[c];e<off[c+1];++e)if(request[faces[e]]>0.)last=faces[e];
  for(int k=0;k<ng;++k){double remaining=removed[k*n+c];
   for(int e=off[c];e<off[c+1];++e){int f=faces[e];double v=0.;
    if(request[f]>0.)v=f==last?remaining:fmin(remaining,removed[k*n+c]*request[f]/total[c]);
    remaining=fmax(0.,remaining-v);face_mass[k*nf+f]=owner[f]==c?v:-v;
   }
  }
 }
}

ENTRY void toe_deposit(int n,int nf,int ng,const int*off,const int*faces,
 const double*area,const double*face_mass,const double*removed,double*work,double*moved){
 LOOP(n){for(int k=0;k<ng;++k){double gain=0.;
   for(int e=off[c];e<off[c+1];++e)gain+=fabs(face_mass[k*nf+faces[e]]);
   work[k*n+c]+=gain/area[c];moved[k*n+c]=gain-removed[k*n+c];
  }
 }
}
