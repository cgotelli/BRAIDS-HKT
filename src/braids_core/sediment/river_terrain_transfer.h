// Nested terrain-guided prolongation. One independent parent per work item.
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
ENTRY void terrain_prolong(int ns,int nt,int nv,double dry,const int*offset,const int*child,
 const double*area,const double*parent_area,const double*parent_rock,const double*reference_rock,
 const double*thickness,const double*parent_q,const double*carried,double*rock,double*q,double*out,double*shift){
 LOOP(ns){double mean=0.;
  for(int e=offset[i];e<offset[i+1];++e){int j=child[e];mean+=reference_rock[j]*area[j];}
  mean/=parent_area[i];
  shift[i]=parent_rock[i]-mean;
  for(int e=offset[i];e<offset[i+1];++e){int j=child[e];rock[j]=parent_rock[i]+reference_rock[j]-mean;q[j]=1.;}
  double h=parent_q[i],eta=0.;
  if(h>dry){
   // Remove unsupported child films without discarding the parent's water.
   // Each removal leaves at least one wet child when h > dry. Re-solve the
   // volume equation on the retained support; no global or atomic reduction.
   for(int k=0;k<=offset[i+1]-offset[i];++k){double aw=0.,za=0.;int removed=0;
    for(int e=offset[i];e<offset[i+1];++e){int j=child[e];if(q[j]>0.){aw+=area[j];za+=(rock[j]+thickness[j])*area[j];}}
    eta=(h*parent_area[i]+za)/aw;
    for(int e=offset[i];e<offset[i+1];++e){int j=child[e];if(q[j]>0.&&eta-rock[j]-thickness[j]<=dry){q[j]=0.;removed=1;}}
    if(!removed)break;
   }
  }
  for(int e=offset[i];e<offset[i+1];++e){int j=child[e];double hj=h<=dry?h:(q[j]>0.?fmax(0.,eta-rock[j]-thickness[j]):0.);
   double ratio=h>dry?hj/h:0.;q[j]=hj;q[nt+j]=parent_q[ns+i]*ratio;q[2*nt+j]=parent_q[2*ns+i]*ratio;
   for(int k=0;k<nv;++k)out[k*nt+j]=carried[k*ns+i]*ratio;
  }
 }
}
