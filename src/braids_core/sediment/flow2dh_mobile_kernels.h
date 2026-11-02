// New isolated classic-FLOW mobile coupling. No change to hydraulic reconstruction.
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

ENTRY void flow_bed_face_inputs(int nf,int n,double diameter,const double*q,
 const double*z,const double*cell,const double*grad,const int*owner,const int*nb,
 const double*normal,const double*distance,const double*xy,double*out){
 LOOP(nf){int o=owner[i],j=nb[i];double nx=normal[i],ny=normal[nf+i];
  double x=cell[3*n+o],y=cell[4*n+o],gx=grad[4*n+o],gy=grad[5*n+o];
  double stress=cell[2*n+o]*sqrt(1.+cell[5*n+o]*cell[5*n+o]),h=q[o];
  if(j>=0){
   double d=distance[i]+distance[nf+i],wo=distance[nf+i]/d,wj=distance[i]/d;
   double a=x*nx+y*ny,b=cell[3*n+j]*nx+cell[4*n+j]*ny;
   // FLOW upwbed sign rule in the actual face-normal basis. Tangential
   // transport is geometrically interpolated, not independently upwinded.
   double un=(a>0.&&b>0.)?a:((a<0.&&b<0.)?b:.5*(a+b));
   double ut=wo*(-x*ny+y*nx)+wj*(-cell[3*n+j]*ny+cell[4*n+j]*nx);
   x=un*nx-ut*ny;y=un*ny+ut*nx;
   gx=wo*gx+wj*grad[4*n+j];gy=wo*gy+wj*grad[5*n+j];
   double dx=xy[j]-xy[o],dy=xy[n+j]-xy[n+o],dn=dx*nx+dy*ny;
   // Non-orthogonal secant correction: affine exact, sees alternating bed
   // elevations which a centred cell-gradient-only method would miss.
   double correction=(z[j]-z[o]-gx*dx-gy*dy)/dn;
   gx+=correction*nx;gy+=correction*ny;
   stress=wo*stress+wj*cell[2*n+j]*sqrt(1.+cell[5*n+j]*cell[5*n+j]);
   h=wo*h+wj*q[j];
  }else{double normal_slope=gx*nx+gy*ny;gx-=normal_slope*nx;gy-=normal_slope*ny;}
  out[i]=x;out[nf+i]=y;out[2*nf+i]=-gx;out[3*nf+i]=-gy;
  out[4*nf+i]=stress;out[5*nf+i]=h;out[6*nf+i]=diameter;
  out[7*nf+i]=diameter;out[8*nf+i]=1.;out[9*nf+i]=1.;
 }
}

ENTRY void flow_bed_flux(int nf,double density,const double*corrected,
 const int*nb,const int*bedbc,const double*normal,const double*feed,double*aux){
 LOOP(nf){double f=(corrected[i]*normal[i]+corrected[nf+i]*normal[nf+i])/density;
  if(nb[i]<0){
   if(bedbc[i]==1)f=-feed[i];
   else if(bedbc[i]==2)f=fmax(0.,f); // open outflow, no unprescribed incoming stock
   else if(bedbc[i]==3){} // native policy: preserve signed capacity before local inflow replacement
   else f=0.;
  }
  aux[2*nf+i]=f;
 }
}

// Released FLOW directional response in an instantaneous streamline frame:
// Fs=d_n(B)+2*k_clockwise*B, Fn=-d_s(B), B=betast*I*speed.
// The minus in Fn is verified by beta-on/off native FLOW updates, not inferred
// from the manual's symmetric tensor. Mapping to Cartesian gives -J grad(B)
// plus the metric source 2*k*B*t. Geometry remains Cartesian and fixed.
ENTRY void flow_directional_faces(int nf,int n,double nu,const double*q,const double*B,
 const int*owner,const int*nb,const double*normal,const double*dist,double*out){
 LOOP(nf){int o=owner[i],j=nb[i];double b=B[o],vx=0.,vy=0.;
  if(j>=0){double d=dist[i]+dist[nf+i],wo=dist[nf+i]/d,wj=dist[i]/d;
   b=wo*b+wj*B[j];vx=nu*(q[n+j]/q[j]-q[n+o]/q[o])/d;
   vy=nu*(q[2*n+j]/q[j]-q[2*n+o]/q[o])/d;}
  out[i]=b*normal[nf+i]+vx;out[nf+i]=-b*normal[i]+vy;
 }
}

ENTRY void flow_directional_geometric(int n,const double*q,const double*op,double*B,double*source){
 LOOP(n){double u=q[n+i]/q[i],v=q[2*n+i]/q[i],speed=sqrt(u*u+v*v);
  B[i]=op[12*n+i]*speed*speed;
  double factor=speed>1.e-8?2.*op[i]*B[i]/speed:0.;
  source[i]=factor*u;source[n+i]=factor*v;
 }
}
