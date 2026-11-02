// Limited second-order capacity reconstruction for the isolated FLOW bundle.
// The historical donor-cell module and bed-slope secants remain unchanged.
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

ENTRY void capacity_gradients(int n,int nf,const double*cell,const int*off,
 const int*faces,const int*owner,const int*nb,const double*delta,const double*inv,
 const double*disp,double*grad,double*limiter){
 LOOP(n){
  double value[2]={cell[3*n+i],cell[4*n+i]};
  double lo[2]={value[0],value[1]},hi[2]={value[0],value[1]};
  double bx[2]={0.,0.},by[2]={0.,0.},gx[2],gy[2];
  int na=off[n];
  for(int k=off[i];k<off[i+1];++k){
   int f=faces[k],j=owner[f]==i?nb[f]:owner[f];if(j<0)continue;
   double dx=delta[k],dy=delta[na+k],w=1./(dx*dx+dy*dy);
   for(int c=0;c<2;++c){double v=cell[(3+c)*n+j];
    lo[c]=fmin(lo[c],v);hi[c]=fmax(hi[c],v);
    bx[c]+=w*dx*(v-value[c]);by[c]+=w*dy*(v-value[c]);}
  }
  for(int c=0;c<2;++c){gx[c]=inv[i]*bx[c]+inv[n+i]*by[c];
   gy[c]=inv[n+i]*bx[c]+inv[2*n+i]*by[c];}
  // Common Barth--Jespersen factor for both vector components. This keeps
  // a parallel capacity field parallel, including near restricted stencils.
  // Open/wall faces retain their historical boundary closure, so only the
  // reconstructed interior faces participate in these bounds.
  double phi=1.;
  for(int k=off[i];k<off[i+1];++k){int f=faces[k];if(nb[f]<0)continue;
   int b=owner[f]==i?0:2;double dx=disp[b*nf+f],dy=disp[(b+1)*nf+f];
   for(int c=0;c<2;++c){double increment=gx[c]*dx+gy[c]*dy;
    double eps=64.*2.2204460492503131e-16*fmax(fabs(value[c]),fmax(fabs(lo[c]),fabs(hi[c])));
    if(increment>eps)phi=fmin(phi,(hi[c]-value[c])/increment);
    else if(increment< -eps)phi=fmin(phi,(lo[c]-value[c])/increment);
   }
  }
  phi=fmax(0.,fmin(1.,phi));limiter[i]=phi;
  for(int c=0;c<2;++c){grad[2*c*n+i]=phi*gx[c];grad[(2*c+1)*n+i]=phi*gy[c];}
 }
}

ENTRY void capacity_reconstructed_faces(int nf,int n,const double*cell,
 const double*grad,const int*owner,const int*nb,const double*normal,
 const double*distance,const double*disp,double*out){
 LOOP(nf){int o=owner[i],j=nb[i];if(j<0)continue;
  double nx=normal[i],ny=normal[nf+i],left[2],right[2];
  for(int c=0;c<2;++c){
   left[c]=cell[(3+c)*n+o]+grad[2*c*n+o]*disp[i]+grad[(2*c+1)*n+o]*disp[nf+i];
   right[c]=cell[(3+c)*n+j]+grad[2*c*n+j]*disp[2*nf+i]+grad[(2*c+1)*n+j]*disp[3*nf+i];
  }
  double a=left[0]*nx+left[1]*ny,b=right[0]*nx+right[1]*ny;
  double un=(a>0.&&b>0.)?a:((a<0.&&b<0.)?b:.5*(a+b));
  double d=distance[i]+distance[nf+i],wo=distance[nf+i]/d,wj=distance[i]/d;
  double ut=wo*(-left[0]*ny+left[1]*nx)+wj*(-right[0]*ny+right[1]*nx);
  out[i]=un*nx-ut*ny;out[nf+i]=un*ny+ut*nx;
 }
}
