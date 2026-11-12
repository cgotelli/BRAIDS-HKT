// Isolated face-local bounds: a tiny increment at one face cannot flatten
// the capacity gradient towards every other face of the cell.
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
 const double*disp,double*grad,double*limiter,double*bounds){
 LOOP(n){
  double x=cell[3*n+i],y=cell[4*n+i],speed=hypot(x,y);
  double tx=speed>0.?x/speed:1.,ty=speed>0.?y/speed:0.;
  double lo[2]={tx*x+ty*y,-ty*x+tx*y},hi[2]={lo[0],lo[1]};
  double bx[2]={0.,0.},by[2]={0.,0.};int na=off[n];
  for(int k=off[i];k<off[i+1];++k){int f=faces[k],j=owner[f]==i?nb[f]:owner[f];if(j<0)continue;
   double dx=delta[k],dy=delta[na+k],w=1./(dx*dx+dy*dy);
   double vx=cell[3*n+j],vy=cell[4*n+j],v[2]={tx*vx+ty*vy,-ty*vx+tx*vy};
   for(int c=0;c<2;++c){lo[c]=fmin(lo[c],v[c]);hi[c]=fmax(hi[c],v[c]);
    double dv=cell[(3+c)*n+j]-cell[(3+c)*n+i];bx[c]+=w*dx*dv;by[c]+=w*dy*dv;}
  }
  for(int c=0;c<2;++c){grad[2*c*n+i]=speed>0.?inv[i]*bx[c]+inv[n+i]*by[c]:0.;
   grad[(2*c+1)*n+i]=speed>0.?inv[n+i]*bx[c]+inv[2*n+i]*by[c]:0.;
   bounds[2*c*n+i]=lo[c];bounds[(2*c+1)*n+i]=hi[c];}
  limiter[i]=1.;
 }
}
ENTRY void capacity_reconstructed_faces(int nf,int n,const double*cell,
 const double*grad,const int*owner,const int*nb,const double*normal,
 const double*distance,const double*disp,double*out,const double*bounds){
 LOOP(nf){int o=owner[i],j=nb[i];if(j<0)continue;
  double nx=normal[i],ny=normal[nf+i],value[2][2];
  for(int s=0;s<2;++s){int c=s==0?o:j;
   double x=cell[3*n+c],y=cell[4*n+c],speed=hypot(x,y);
   double tx=speed>0.?x/speed:1.,ty=speed>0.?y/speed:0.;
   double vx=x+grad[c]*disp[2*s*nf+i]+grad[n+c]*disp[(2*s+1)*nf+i];
   double vy=y+grad[2*n+c]*disp[2*s*nf+i]+grad[3*n+c]*disp[(2*s+1)*nf+i];
   double a=fmax(bounds[c],fmin(bounds[n+c],tx*vx+ty*vy));
   double b=fmax(bounds[2*n+c],fmin(bounds[3*n+c],-ty*vx+tx*vy));
   value[s][0]=tx*a-ty*b;value[s][1]=ty*a+tx*b;
  }
  double a=value[0][0]*nx+value[0][1]*ny,b=value[1][0]*nx+value[1][1]*ny;
  double un=fmax(a,0.)+fmin(b,0.);
  double d=distance[i]+distance[nf+i],wo=distance[nf+i]/d,wj=distance[i]/d;
  double ut=wo*(-value[0][0]*ny+value[0][1]*nx)+wj*(-value[1][0]*ny+value[1][1]*nx);
  out[i]=un*nx-ut*ny;out[nf+i]=un*ny+ut*nx;
 }
}
