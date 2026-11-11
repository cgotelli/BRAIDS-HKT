// Fixed-mesh Jiang-Cao coupling: identical OpenMP/CUDA physical arithmetic.
#ifdef __CUDACC__
#define JC_ENTRY extern "C" __global__
#define JC_LOOP(N) for(int i=blockDim.x*blockIdx.x+threadIdx.x;i<N;i+=blockDim.x*gridDim.x)
#else
#include <cmath>
#include <omp.h>
using std::isfinite;
#define JC_ENTRY extern "C" __declspec(dllexport)
#define JC_LOOP(N) _Pragma("omp parallel for schedule(static)") for(int i=0;i<N;++i)
extern "C" __declspec(dllexport) void mobile_threads(int n){omp_set_dynamic(0);omp_set_num_threads(n);}
#endif

// Weighted least squares in physical Cartesian coordinates; no axis assumption.
JC_ENTRY void jc_gradient(int n,int limited,const int*offset,const int*adj,
 const double*weights,const double*mid_delta,const double*v,double*out){
 JC_LOOP(n){
  double gx=0.,gy=0.,lo=v[i],hi=v[i];
  for(int k=offset[i];k<offset[i+1];++k){int j=adj[k];double d=v[j]-v[i];
   gx+=weights[2*k]*d;gy+=weights[2*k+1]*d;lo=fmin(lo,v[j]);hi=fmax(hi,v[j]);}
  double phi=1.;
  if(limited)for(int k=offset[i];k<offset[i+1];++k){
   double d=gx*mid_delta[2*k]+gy*mid_delta[2*k+1];
   if(d>0.)phi=fmin(phi,(hi-v[i])/d);else if(d<0.)phi=fmin(phi,(lo-v[i])/d);}
  out[i]=phi*gx;out[n+i]=phi*gy;
 }
}

// Affine-exact one-sided capacity slope.  The centered least-squares
// transverse component is retained; only its odd/even-blind streamwise
// component is replaced.  Prefer meaningfully aligned forward neighbours;
// at an outlet corner use the strongly aligned backward side.  This avoids
// dividing by an almost perpendicular projection at a moving wet front.
JC_ENTRY void jc_capacity_gradient(int n,const int*offset,const int*adj,
 const double*xy,const double*v,const double*q,const double*base,double*out){
 JC_LOOP(n){
  const double mx=q[n+i],my=q[2*n+i],mag=hypot(mx,my);
  double gx=base[i],gy=base[n+i];
  if(mag>0.){
   const double ex=mx/mag,ey=my/mag,along=gx*ex+gy*ey;
   const double px=gx-along*ex,py=gy-along*ey;
   double maxf=0.,maxb=0.;
   for(int k=offset[i];k<offset[i+1];++k){
    int j=adj[k];double dx=xy[j]-xy[i],dy=xy[n+j]-xy[n+i];
    double r2=dx*dx+dy*dy,forward=dx*ex+dy*ey;
    double cosine=forward/sqrt(r2);maxf=fmax(maxf,cosine);maxb=fmax(maxb,-cosine);
   }
   double side=maxf>=.25?1.:(maxb>=.25?-1.:0.);
   double aligned=side>0.?maxf:maxb,sum=0.,weight=0.;
   if(side!=0.)for(int k=offset[i];k<offset[i+1];++k){
    int j=adj[k];double dx=xy[j]-xy[i],dy=xy[n+j]-xy[n+i];
    double r2=dx*dx+dy*dy,forward=dx*ex+dy*ey,projection=side*forward;
    if(projection>0. && projection/sqrt(r2)>=.5*aligned){
     double w=forward*forward/r2;
     sum+=w*((v[j]-v[i]-px*dx-py*dy)/forward);weight+=w;
    }
   }
   if(weight>0.){
    double corrected=sum/weight;
    gx=px+corrected*ex;gy=py+corrected*ey;
   }
  }
  out[i]=gx;out[n+i]=gy;
 }
}

// First pass of the capacity-only compact filter.  A pair average is the exact
// value of an affine field at the two-centroid midpoint, including on cropped
// and skew cells.  Projected shared-face width selects the local flow path.
JC_ENTRY void jc_capacity_average(int n,int nf,const int*offset,const int*adj,
 const int*face,const double*length,const double*normal,const double*xy,
 const double*q,double*average,double*location){
 JC_LOOP(n){
  const double mx=q[n+i],my=q[2*n+i],mag=hypot(mx,my);
  for(int c=0;c<3;++c)average[c*n+i]=q[c*n+i];
  location[i]=xy[i];location[n+i]=xy[n+i];
  if(!(q[i]>0.)||mag==0.)continue;
  const double ex=mx/mag,ey=my/mag;double total=0.,sum[3]={0.,0.,0.},sx=0.,sy=0.;
  for(int k=offset[i];k<offset[i+1];++k){
   int j=adj[k],f=face[k];if(!(q[j]>0.))continue;
   double projected=fabs(normal[f]*ex+normal[nf+f]*ey);
   double w=length[f]*projected;if(w<=0.)continue;
   for(int c=0;c<3;++c)sum[c]+=w*.5*(q[c*n+i]+q[c*n+j]);
   sx+=w*.5*(xy[i]+xy[j]);sy+=w*.5*(xy[n+i]+xy[n+j]);total+=w;
  }
  if(total>0.){for(int c=0;c<3;++c)average[c*n+i]=sum[c]/total;
   location[i]=sx/total;location[n+i]=sy/total;}
 }
}

// Recover the compact pair average from its actual sampling location to the
// cell centroid.  Dynamic-location WLS is affine-exact.  A pairwise odd/even
// mode has constant averages and therefore cannot be reintroduced here.
JC_ENTRY void jc_capacity_state(int n,const int*offset,const int*adj,const double*xy,
 const double*average,const double*location,const double*q,double*out){
 JC_LOOP(n){
  for(int c=0;c<3;++c){
   double xx=0.,yy=0.,xyv=0.,bx=0.,by=0.;
   for(int k=offset[i];k<offset[i+1];++k){int j=adj[k];
    double dx=location[j]-location[i],dy=location[n+j]-location[n+i],r2=dx*dx+dy*dy;
    if(r2<=1.e-28)continue;double w=1./r2,dv=average[c*n+j]-average[c*n+i];
    xx+=w*dx*dx;yy+=w*dy*dy;xyv+=w*dx*dy;bx+=w*dx*dv;by+=w*dy*dv;
   }
   double tr=xx+yy,det=xx*yy-xyv*xyv,gx=0.,gy=0.;
   if(det>1.e-13*tr*tr){gx=(yy*bx-xyv*by)/det;gy=(xx*by-xyv*bx)/det;}
   else if(tr>0.){gx=(xx*bx+xyv*by)/(tr*tr);gy=(xyv*bx+yy*by)/(tr*tr);}
   out[c*n+i]=average[c*n+i]+gx*(xy[i]-location[i])+gy*(xy[n+i]-location[n+i]);
  }
  if(!(q[i]>0.)||!(out[i]>0.)||!isfinite(out[i])||!isfinite(out[n+i])||!isfinite(out[2*n+i]))
   for(int c=0;c<3;++c)out[c*n+i]=q[c*n+i];
 }
}

JC_ENTRY void jc_manning(int n,double dt,double gravity,const double*rough,double*q){
 JC_LOOP(n){double h=q[i],speed=hypot(q[n+i],q[2*n+i])/h;
  double factor=1./(1.+dt*gravity*rough[i]*rough[i]*speed/pow(h,4./3.));
  q[n+i]*=factor;q[2*n+i]*=factor;}
}

// Density-gradient source from the target mixture momentum equation.
JC_ENTRY void jc_density_source(int n,double gravity,double rw,double rs,
 const double*q,const double*c,const double*grad,double*rate){
 JC_LOOP(n){double rho=rw+(rs-rw)*c[i],f=-.5*gravity*q[i]*q[i]*(rs-rw)/rho;
  rate[n+i]+=f*grad[i];rate[2*n+i]+=f*grad[n+i];}
}

// Relative bedload discharge for the paper's gravitational direction term.
// A sub-capacity mobile inventory still moves when the local capacity is zero.
JC_ENTRY void jc_relative(int n,const double*q,const double*cap,double*delta){
 JC_LOOP(n){double mx=q[n+i],my=q[2*n+i],theta=cap[3*n+i],tc=cap[4*n+i];
  double r=theta>0.?-1.5*cap[8*n+i]*sqrt(tc/theta):0.,norm=hypot(1.,r);
  delta[i]=(mx-r*my)/norm-mx;delta[n+i]=(my+r*mx)/norm-my;}
}

// Q is the accepted hydraulic stage mixture rate INCLUDING face length.
// Constant c and zero relative velocity give F_s=c*Q exactly.
JC_ENTRY void jc_mobile_faces(int n,int nf,const int*owner,const int*neighbour,
 const int*kind,const double*length,const double*normal,const double*xy,
 const double*mid,const double*Q,const double*c,const double*grad,
 const double*delta,const double*dgrad,const double*cin,double*F,double*transportQ){
 JC_LOOP(nf){int o=owner[i],j=neighbour[i];double flux=Q[i];
  transportQ[i]=flux;
  if(kind[i]==1){F[i]=0.;transportQ[i]=0.;continue;}
  if(j<0){F[i]=flux>=0.?flux*c[o]:(kind[i]==2?flux*cin[i]:0.);continue;}
  // Reconstruct relative discharge to the TRUE midpoint on each side.
  // Centroid averaging alone is not affine-exact on a cropped/skew face.
  double dox=mid[i]-xy[o],doy=mid[nf+i]-xy[n+o],djx=mid[i]-xy[j],djy=mid[nf+i]-xy[n+j];
  double rx=.5*(delta[o]+dgrad[o]*dox+dgrad[n+o]*doy+delta[j]+dgrad[j]*djx+dgrad[n+j]*djy);
  double ry=.5*(delta[n+o]+dgrad[2*n+o]*dox+dgrad[3*n+o]*doy+delta[n+j]+dgrad[2*n+j]*djx+dgrad[3*n+j]*djy);
  double rel=(rx*normal[i]+ry*normal[nf+i])*length[i];
  double total=flux+rel;transportQ[i]=total;int donor=total>=0.?o:j;
  double trace=c[donor]+grad[donor]*(mid[i]-xy[donor])+grad[n+donor]*(mid[nf+i]-xy[n+donor]);
  // Trace limiter guarantees nonnegative values analytically; only roundoff clamp.
  F[i]=total*fmax(0.,trace);
 }
}

// A shared-face donor bound prevents loss of nonexistent mobile inventory,
// including near-vacuum roundoff. No state clipping or mass creation.
JC_ENTRY void jc_mobile_bound(int n,double dt,const int*offset,const int*faces,
 const int*owner,const double*area,const double*h,const double*m,
 const double*F,const double*T,double*factor,double*frequency){
 JC_LOOP(n){double out=0.,speed=0.;for(int k=offset[i];k<offset[i+1];++k){
  int f=faces[k];double sign=owner[f]==i?1.:-1.;out+=fmax(0.,sign*F[f]);
  speed+=fmax(0.,sign*T[f]);}
  // Hold only outgoing subnormal inventory: area/divergence arithmetic cannot
  // resolve its sign reliably. Incoming flux still enters; nothing is deleted.
  factor[i]=out>0.?(m[i]*area[i]<1.4240472694446089e-306?0.:
      fmin(1.,(1.-2.e-15)*m[i]*area[i]/(dt*out))):1.;
  frequency[i]=speed/(h[i]*area[i]);}
}

JC_ENTRY void jc_mobile_limit(int nf,const int*owner,const int*neighbour,const double*factor,double*F){
 JC_LOOP(nf){int donor=F[i]>=0.?owner[i]:neighbour[i];if(donor>=0)F[i]*=factor[donor];}
}

JC_ENTRY void jc_mobile_divergence(int n,const int*offset,const int*faces,
 const int*owner,const double*area,const double*F,double*out){
 JC_LOOP(n){double sum=0.;for(int k=offset[i];k<offset[i+1];++k){int f=faces[k];
  sum+=(owner[f]==i?1.:-1.)*F[f];}out[i]=-sum/area[i];}
}
