// Section critical control on the actual flat exterior face geometry.
// Candidate branches and reduction tiles execute in parallel on OpenMP/CUDA.
#ifdef __CUDACC__
#define SECTION_INF __longlong_as_double(0x7ff0000000000000ULL)
#define ENTRY extern "C" __global__
#define LOOP(N) for(int i=blockDim.x*blockIdx.x+threadIdx.x;i<N;i+=blockDim.x*gridDim.x)
#else
#include <cmath>
#include <omp.h>
using std::isfinite;
#define SECTION_INF HUGE_VAL
#define ENTRY extern "C" __declspec(dllexport)
#define LOOP(N) _Pragma("omp parallel for schedule(static)") for(int i=0;i<N;++i)
extern "C" __declspec(dllexport) void mobile_threads(int n){omp_set_dynamic(0);omp_set_num_threads(n);}
#endif

ENTRY void section_critical_candidates(int nf,int nb,double gravity,double dry,
 const int*selected,const int*bid,const int*control,const int*offset,const int*faces,
 const double*bed,const double*length,const double*forcing,double*candidate,double*energy){
 LOOP(nf){
  candidate[i]=0.;energy[i]=SECTION_INF;
  int b=bid[i];if(!selected[i] || !control[b])continue;
  double Q=forcing[b],level=bed[i],minimum=level,next=SECTION_INF,width=0.,excess=0.;
  // Each distinct bed level starts one constant-wet-width branch. For that
  // branch A(eta)=width*(eta-level)+excess and Fr^2=Q^2*width/(g*A^3).
  // Enumerating all branches also handles compound/disconnected wet sections.
  for(int k=offset[b];k<offset[b+1];++k){
   int f=faces[k];double z=bed[f];minimum=fmin(minimum,z);
   if(z>level)next=fmin(next,z);
   else{width+=length[f];excess+=(level-z)*length[f];}
  }
  if(Q==0.){candidate[i]=minimum;energy[i]=0.;continue;}
  double area=cbrt(Q*Q*width/gravity),eta=level+(area-excess)/width;
  if(!isfinite(area) || area<=0. || !isfinite(eta) || !(eta>level+dry) || eta>next+dry)continue;
  // Recheck the actual dry-face active set after rounded stage arithmetic.
  double actual=0.,wet_width=0.;
  for(int k=offset[b];k<offset[b+1];++k){
   int f=faces[k];double h=eta-bed[f];
   if(h>dry){actual+=h*length[f];wet_width+=length[f];}
  }
  if(wet_width!=width || actual<=0.)continue;
  double residual=fabs(Q*Q*wet_width/(gravity*actual*actual*actual)-1.);
  // Large absolute datums can lose a few ulps when subtracting bed from eta.
  // Never accept an unresolved wet-width transition as a critical root.
  double tolerance=1e-10+64.*2.2204460492503131e-16*fmax(1.,fabs(eta))*width/actual;
  if(residual>tolerance)continue;
  candidate[i]=eta;energy[i]=eta-minimum+Q*Q/(2.*gravity*actual*actual);
 }
}

ENTRY void section_critical_partial(int count,int nb,int tiles,const int*offset,const int*faces,
 const double*candidate,const double*energy,double*stage_part,double*energy_part){
 LOOP(count){
  int b=i/tiles,tile=i%tiles;double best=SECTION_INF,stage=0.;
  for(int k=offset[b]+64*tile;k<offset[b+1] && k<offset[b]+64*(tile+1);++k){
   int f=faces[k];double e=energy[f],s=candidate[f];
   if(e<best || (e==best && s<stage)){best=e;stage=s;}
  }
  stage_part[i]=stage;energy_part[i]=best;
 }
}

ENTRY void section_critical_total(int nb,int tiles,const int*control,const double*stage_part,
 const double*energy_part,double*forcing,int*status){
 LOOP(nb){
  status[i]=0;if(!control[i])continue;
  double best=SECTION_INF,stage=0.;
  for(int k=0;k<tiles;++k){int j=i*tiles+k;double e=energy_part[j],s=stage_part[j];
   if(e<best || (e==best && s<stage)){best=e;stage=s;}
  }
  if(!isfinite(best)){status[i]=1;continue;}
  forcing[nb+i]=stage;
 }
}
