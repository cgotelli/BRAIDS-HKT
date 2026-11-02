// Zero-area exterior material history. Same arithmetic on OpenMP and CUDA.
#ifdef __CUDACC__
#define ENTRY extern "C" __global__
#define LOOP(N) for(int i=blockDim.x*blockIdx.x+threadIdx.x;i<N;i+=blockDim.x*gridDim.x)
#else
#include <cmath>
#include <omp.h>
#define ENTRY extern "C" __declspec(dllexport)
#define LOOP(N) _Pragma("omp parallel for schedule(static)") for(int i=0;i<N;++i)
extern "C" __declspec(dllexport) void mobile_threads(int n){omp_set_dynamic(0);omp_set_num_threads(n);}
#endif

ENTRY void exterior_composition(int ne,int nc,int nl,int ng,const int*mode,
 const int*owner,const double*rho,const double*fractions,const double*mass,
 const double*sv,const double*cellmass,const double*cellsv,double*out,double*outsv){
 LOOP(ne){int m=mode[i],o=owner[i];double total=0.,denom=0.;
  for(int k=0;k<ng;++k){total+=mass[k*ne+i];
   denom+=fractions[k*ne+i]*(m==4?rho[k]:1.);}
  for(int l=0;l<nl;++l){
   outsv[l*ne+i]=m==1?cellsv[l*nc+o]:sv[l*ne+i];
   for(int k=0;k<ng;++k){int b=(l*ng+k)*ne+i;
    double v=mass[b];
    if(m==1)v=cellmass[(l*ng+k)*nc+o];
    else if(l==0&&(m==3||m==4))v=total*fractions[k*ne+i]*(m==4?rho[k]:1.)/denom;
    out[b]=v;
   }
  }
 }
}

ENTRY void exterior_sediment_face(int ne,int nf,int nc,int slope_mode,int vr,
 double dry,double seddepth,const int*faces,const int*owner,const double*normal,
 const double*disp,const double*q,const double*z,const double*bed,const double*grad,
 const double*cell,const double*external,const double*trace,const double*mean,
 const double*hide,const double*inside_hide,const double*taurat,double*fields){
 LOOP(ne){int f=faces[i],o=owner[i];double nx=normal[f],ny=normal[nf+f];
  double a=cell[3*nc+o]*nx+cell[4*nc+o]*ny;
  double b=external[3*ne+i]*nx+external[4*ne+i]*ny;
  double un=(a>0.&&b>0.)?a:((a<0.&&b<0.)?b:.5*(a+b));
  // Exterior state is AT the physical boundary face, not a fictitious cell
  // centre. Tangent and stress therefore use its directly evaluated state.
  double ut=-external[3*ne+i]*ny+external[4*ne+i]*nx;
  double h=trace[i];if(q[o]<=seddepth||h<=seddepth)un=0.;
  fields[i]=un*nx-ut*ny;fields[ne+i]=un*ny+ut*nx;
  double gx=grad[4*nc+o],gy=grad[5*nc+o],dx=disp[f],dy=disp[nf+f];
  double correction=(bed[f]-z[o]-gx*dx-gy*dy)/(dx*nx+dy*ny);
  gx+=correction*nx;gy+=correction*ny;
  fields[2*ne+i]=-gx;fields[3*ne+i]=-gy;
  fields[4*ne+i]=slope_mode==2?(vr?taurat[i]:0.):external[2*ne+i]*sqrt(1.+external[5*ne+i]*external[5*ne+i]);
  fields[5*ne+i]=fmax(h,dry);fields[7*ne+i]=mean[i];
  fields[8*ne+i]=inside_hide[o];fields[9*ne+i]=hide[i];
 }
}
