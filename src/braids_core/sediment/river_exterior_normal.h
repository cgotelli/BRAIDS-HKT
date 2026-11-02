// Normal section on actual exterior faces, with physical wetted sidewalls.
#ifdef __CUDACC__
#define ENTRY extern "C" __global__
#define LOCAL __device__ __forceinline__
#define INF_VALUE __longlong_as_double(0x7ff0000000000000ULL)
#define LOOP(N) for(int i=blockDim.x*blockIdx.x+threadIdx.x;i<N;i+=blockDim.x*gridDim.x)
#else
#include <cmath>
#include <omp.h>
using std::isfinite;
#define ENTRY extern "C" __declspec(dllexport)
#define LOCAL inline
#define INF_VALUE HUGE_VAL
#define LOOP(N) _Pragma("omp parallel for schedule(static)") for(int i=0;i<N;++i)
extern "C" __declspec(dllexport) void mobile_threads(int n){omp_set_dynamic(0);omp_set_num_threads(n);}
#endif

LOCAL double normal_chezy(int mode,double radius,double parameter,double floor,double gravity){
 double r=fmax(radius,floor);
 if(mode==1)return parameter;
 if(mode==2)return pow(r,1./6.)/parameter;
 if(mode==3)return sqrt(gravity)/.41*log1p(30.*r/(exp(1.)*parameter));
 return 18.*log10(12.*fmax(.5,r/parameter));
}
LOCAL double normal_perimeter(int f,double eta,double dry,const int*left,const int*right,
 const double*bed,const double*length){
 double h=eta-bed[f];if(h<=dry)return 0.;
 int l=left[f],r=right[f];double hl=l<0?0.:fmax(eta-bed[l],0.),hr=r<0?0.:fmax(eta-bed[r],0.);
 if(hl<=dry)hl=0.;if(hr<=dry)hr=0.;
 return length[f]+fmax(h-hl,0.)+fmax(h-hr,0.);
}
LOCAL double normal_rate(int b,double eta,double dry,double gravity,int law,double parameter,
 int feedback,int ne,const int*offset,const int*faces,const int*left,const int*right,const int*compact,
 const double*bed,const double*length,const double*state,double slope,double*metrics){
 double area=0.,perimeter=0.,drag=0.;
 for(int k=offset[b];k<offset[b+1];++k){int f=faces[k];double h=eta-bed[f];
  if(h>dry){area+=h*length[f];perimeter+=normal_perimeter(f,eta,dry,left,right,bed,length);}
 }
 double rate=0.;
 if(area>0. && perimeter>0.){
  double radius=area/perimeter,static_c=normal_chezy(law,radius,parameter,dry,gravity);
  for(int k=offset[b];k<offset[b+1];++k){int f=faces[k];double h=eta-bed[f];if(h<=dry)continue;
   double c=static_c;
   if(feedback){int i=compact[f];double ks2=0.;
    for(int j=2;j<5;++j){double v=state[j*ne+i];ks2+=v*v;}
    // Same frozen bedform components and local physical-depth cap as exterior
    // resistance. Convert them with section hydraulic radius for conveyance.
    double ks=fmax(.001,fmax(1e-6,fmin(sqrt(ks2),.5*h)));
    c=18.*log10(fmax(12.*fmax(radius,dry)/ks,1.0129));
   }
   drag+=normal_perimeter(f,eta,dry,left,right,bed,length)/(c*c);
  }
  if(drag>0.)rate=area*sqrt(area*slope/drag);
 }
 if(metrics){metrics[0]=area;metrics[1]=perimeter;metrics[2]=drag;metrics[3]=rate;}
 return rate;
}

ENTRY void section_normal_candidates(int nf,int nb,int ne,double gravity,double dry,int law,double parameter,int feedback,
 const int*selected,const int*bid,const int*control,const int*offset,const int*faces,
 const int*left,const int*right,const int*compact,const double*bed,const double*length,
 const double*state,const double*slope,const double*forcing,double*candidate,double*rank){
 LOOP(nf){
  candidate[i]=0.;rank[i]=INF_VALUE;int b=bid[i];if(!selected[i] || !control[b])continue;
  double Q=forcing[b],level=bed[i],minimum=level,next=INF_VALUE;bool duplicate=false;
  for(int k=offset[b];k<offset[b+1];++k){int f=faces[k];double z=bed[f];minimum=fmin(minimum,z);
   if(z>level)next=fmin(next,z);if(z==level && f<i)duplicate=true;
  }
  if(duplicate)continue;
  if(Q==0.){candidate[i]=minimum;rank[i]=minimum;continue;}
  double epsilon=8.*2.2204460492503131e-16*fmax(1.,fabs(level));
  double lo=level+dry+epsilon,hi=isfinite(next)?next+dry:level+fmax(1.,64.*dry);
  if(!(hi>lo))continue;
  double rlo=normal_rate(b,lo,dry,gravity,law,parameter,feedback,ne,offset,faces,left,right,compact,bed,length,state,slope[b],0)-Q;
  double rhi=normal_rate(b,hi,dry,gravity,law,parameter,feedback,ne,offset,faces,left,right,compact,bed,length,state,slope[b],0)-Q;
  if(!isfinite(next)){
   for(int k=0;k<60 && rhi<0.;++k){hi=level+2.*(hi-level);
    rhi=normal_rate(b,hi,dry,gravity,law,parameter,feedback,ne,offset,faces,left,right,compact,bed,length,state,slope[b],0)-Q;
   }
  }
  if(!isfinite(rlo) || !isfinite(rhi))continue;
  // Search each wet-width branch in ascending stage. Sub-bracketing also
  // covers a turning composite-resistance curve; never bisect across a dry
  // face jump and then call that discontinuity a converged normal root.
  double start=lo,end=hi,previous=rlo;bool found=false;
  for(int j=1;j<=32;++j){
   double at=start+(end-start)*(double(j)/32.);
   double value=j==32?rhi:normal_rate(b,at,dry,gravity,law,parameter,feedback,ne,offset,faces,left,right,compact,bed,length,state,slope[b],0)-Q;
   if(isfinite(value) && ((previous<=0. && value>=0.) || (previous>=0. && value<=0.))){hi=at;rhi=value;rlo=previous;found=true;break;}
   lo=at;previous=value;
  }
  if(!found)continue;
  for(int j=0;j<80;++j){
   double mid=lo+.5*(hi-lo);if(mid==lo || mid==hi)break;
   double value=normal_rate(b,mid,dry,gravity,law,parameter,feedback,ne,offset,faces,left,right,compact,bed,length,state,slope[b],0)-Q;
   if((rlo<=0. && value>=0.) || (rlo>=0. && value<=0.)){hi=mid;rhi=value;}
   else{lo=mid;rlo=value;}
  }
  double eta=fabs(rlo)<fabs(rhi)?lo:hi;
  double rate=normal_rate(b,eta,dry,gravity,law,parameter,feedback,ne,offset,faces,left,right,compact,bed,length,state,slope[b],0);
  double tolerance=1e-10+128.*2.2204460492503131e-16*fmax(1.,fabs(eta))/fmax(eta-minimum,dry);
  if(isfinite(rate) && fabs(rate/Q-1.)<=tolerance){candidate[i]=eta;rank[i]=eta;}
 }
}

ENTRY void section_normal_metrics(int nb,int ne,double gravity,double dry,int law,double parameter,int feedback,
 const int*control,const int*offset,const int*faces,const int*left,const int*right,const int*compact,
 const double*bed,const double*length,const double*state,const double*slope,const double*forcing,double*out){
 LOOP(nb){
  double values[4]={0.,0.,0.,0.};
  if(control[i])normal_rate(i,forcing[nb+i],dry,gravity,law,parameter,feedback,ne,offset,faces,left,right,compact,bed,length,state,slope[i],values);
  for(int j=0;j<4;++j)out[j*nb+i]=values[j];
 }
}
