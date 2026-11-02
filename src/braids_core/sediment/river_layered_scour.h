// Conservative layered extension of the existing legacy-Scour geometry.
// Peel actual donor strata; never homogenize the whole column or invent stock.
// Parallel independent rows, then columns; pairs within each sweep are ordered.
#ifdef __CUDACC__
#define ENTRY extern "C" __global__
#define LOCAL __device__
#define LOOP(N) for(int row=blockDim.x*blockIdx.x+threadIdx.x;row<N;row+=blockDim.x*gridDim.x)
#else
#include <cmath>
#include <omp.h>
#define ENTRY extern "C" __declspec(dllexport)
#define LOCAL inline
#define LOOP(N) _Pragma("omp parallel for schedule(static)") for(int row=0;row<N;++row)
extern "C" __declspec(dllexport) void mobile_threads(int n){omp_set_dynamic(0);omp_set_num_threads(n);}
#endif

LOCAL void layered_pair(int a,int b,int nc,int ng,int nl,double dx,double slope,
 const double*area,const double*rock,double*mass,double*moved){
 double za=rock[a],zb=rock[b],stocka=0.,stockb=0.;
 for(int l=0;l<nl;++l)for(int k=0;k<ng;++k){
  stocka+=mass[(l*ng+k)*nc+a]/1590.;stockb+=mass[(l*ng+k)*nc+b]/1590.;
 }
 za+=stocka;zb+=stockb;double dz=za-zb;
 if(fabs(dz/dx)<=slope)return;
 int donor=dz>0.?a:b,recipient=dz>0.?b:a;
 double requested=.25*fabs(dz)*(area[a]+area[b]);
 double available=(dz>0.?stocka:stockb)*area[donor];
 double remaining=fmin(requested,available*(1.-64.*2.2204460492503131e-16));
 for(int l=0;l<nl&&remaining>0.;++l){
  double layer=0.;for(int k=0;k<ng;++k)layer+=mass[(l*ng+k)*nc+donor]/1590.*area[donor];
  if(layer<=0.)continue;
  double take=fmin(remaining,layer),share=fmin(1.,take/layer);
  for(int k=0;k<ng;++k){int ix=(l*ng+k)*nc+donor;
   double loss=mass[ix]*share,amount=loss*area[donor];
   mass[ix]-=loss;mass[k*nc+recipient]+=amount/area[recipient];
   moved[k*nc+donor]-=amount;moved[k*nc+recipient]+=amount;
  }
  remaining=fmax(0.,remaining-take);
 }
}
ENTRY void layered_scour_rows(int nx,int ny,int ng,int nl,double dx,double slope,
 const double*area,const double*rock,double*mass,double*moved){
 LOOP(ny){for(int i=0;i<nx-1;++i)layered_pair(row*nx+i,row*nx+i+1,nx*ny,ng,nl,dx,slope,area,rock,mass,moved);}
}
ENTRY void layered_scour_columns(int nx,int ny,int ng,int nl,double dy,double slope,
 const double*area,const double*rock,double*mass,double*moved){
 LOOP(nx){for(int j=0;j<ny-1;++j)layered_pair(j*nx+row,(j+1)*nx+row,nx*ny,ng,nl,dy,slope,area,rock,mass,moved);}
}
