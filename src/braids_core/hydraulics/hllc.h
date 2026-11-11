// Shared OpenMP/CUDA second-order polygonal MUSCL-Hancock shallow water.
#ifdef __CUDACC__
#define ENTRY extern "C" __global__
#define DEVICE __device__
#define LOOP(N) for(int i=blockDim.x*blockIdx.x+threadIdx.x;i<N;i+=blockDim.x*gridDim.x)
#else
#include <math.h>
#include <omp.h>
#define ENTRY extern "C" __declspec(dllexport)
#define DEVICE inline
#define LOOP(N) _Pragma("omp parallel for schedule(static)") for(int i=0;i<N;++i)
extern "C" __declspec(dllexport) void mobile_threads(int n){omp_set_dynamic(0);omp_set_num_threads(n);}
#endif

ENTRY void hllc_reconstruct(int n,int nf,double gravity,double dry,double dt,const double*q,const double*z,
 const double*xy,const double*mid,const double*normal,const int*owner,const int*neighbour,const int*selected,
 const int*off,const int*faces,const double*inverse,double*grad,double*half,double*qhalf,int*status){
 LOOP(n){double h=q[i],vi[4]={h+z[i],h>dry?q[n+i]/h:0.,h>dry?q[2*n+i]/h:0.,z[i]};
  double bx[4]={0.,0.,0.,0.},by[4]={0.,0.,0.,0.},lo[4],hi[4],G[8];
  for(int m=0;m<4;++m)lo[m]=hi[m]=vi[m];
  for(int k=off[i];k<off[i+1];++k){int f=faces[k],j=owner[f]==i?neighbour[f]:owner[f];
   double dx,dy,vj[4];
   if(j>=0){double hj=q[j];dx=xy[j]-xy[i];dy=xy[n+j]-xy[n+i];
    vj[0]=hj+z[j];vj[1]=hj>dry?q[n+j]/hj:0.;vj[2]=hj>dry?q[2*n+j]/hj:0.;vj[3]=z[j];
    if(vj[0]-vj[3]<=dry && vj[3]>=vi[0])vj[0]=vi[0];
   }else{if(selected[f])continue;double nx=normal[f],ny=normal[nf+f];
    double distance=(mid[f]-xy[i])*nx+(mid[nf+f]-xy[n+i])*ny;dx=2.*distance*nx;dy=2.*distance*ny;
    for(int m=0;m<4;++m)vj[m]=vi[m];double un=vi[1]*nx+vi[2]*ny;vj[1]-=2.*un*nx;vj[2]-=2.*un*ny;
   }
   double d2=dx*dx+dy*dy;if(d2<=0.)continue;
   for(int m=0;m<4;++m){double d=vj[m]-vi[m];bx[m]+=d*dx/d2;by[m]+=d*dy/d2;lo[m]=fmin(lo[m],vj[m]);hi[m]=fmax(hi[m],vj[m]);}
  }
  for(int m=0;m<4;++m){double gx=inverse[i]*bx[m]+inverse[n+i]*by[m];double gy=inverse[n+i]*bx[m]+inverse[2*n+i]*by[m],theta=1.;
   for(int k=off[i];k<off[i+1];++k){int f=faces[k];double d=gx*(mid[f]-xy[i])+gy*(mid[nf+f]-xy[n+i]);
    if(d>0.)theta=fmin(theta,(hi[m]-vi[m])/d);else if(d<0.)theta=fmin(theta,(lo[m]-vi[m])/d);}
   G[2*m]=theta*gx;G[2*m+1]=theta*gy;
  }
  h=vi[0]-vi[3];if(h<=dry)for(int k=0;k<6;++k)G[k]=0.;
  for(int k=0;k<8;++k)grad[k*n+i]=G[k];
  double u=vi[1],w=vi[2];double eta=vi[0]-.5*dt*(u*(G[0]-G[6])+w*(G[1]-G[7])+h*(G[2]+G[5]));
  double uh=u-.5*dt*(u*G[2]+w*G[3]+gravity*G[0]);double wh=w-.5*dt*(u*G[4]+w*G[5]+gravity*G[1]);
  if(eta<vi[3]){eta=vi[0];uh=u;wh=w;}if(h<=dry){eta=vi[0];uh=wh=0.;}
  half[i]=eta;half[n+i]=uh;half[2*n+i]=wh;half[3*n+i]=vi[3];h=fmax(0.,eta-vi[3]);
  qhalf[i]=h;qhalf[n+i]=h*uh;qhalf[2*n+i]=h*wh;status[i]=!isfinite(eta)||!isfinite(uh)||!isfinite(wh);
 }
}

DEVICE void hllc_flux(double hl,double ul,double vl,double hr,double ur,double vr,
 double g,double dry,double*out,double&speed){
 double cl=sqrt(g*hl),cr=sqrt(g*hr);
 double sl=fmin(ul-cl,ur-cr),sr=fmax(ul+cl,ur+cr);
 if(hl<=dry)sl=ur-2.*cr;if(hr<=dry)sr=ul+2.*cl;
 speed=fmax(fabs(sl),fabs(sr));
 double L[3]={hl*ul,hl*ul*ul+.5*g*hl*hl,hl*ul*vl};
 double R[3]={hr*ur,hr*ur*ur+.5*g*hr*hr,hr*ur*vr};
 if(hl<=dry && hr<=dry){out[0]=out[1]=out[2]=speed=0.;return;}
 if(sl>=0.){for(int k=0;k<3;++k)out[k]=L[k];return;}
 if(sr<=0.){for(int k=0;k<3;++k)out[k]=R[k];return;}
 double a=hl*(sl-ul),b=hr*(sr-ur),den=a-b;
 double sm=fabs(den)>1e-30?(.5*g*(hr*hr-hl*hl)+a*ul-b*ur)/den:0.;
 bool contact=hl>dry && hr>dry && sl<sm && sm<sr;
 if(contact){double hs=sm>=0.?a/(sl-sm):b/(sr-sm);
  if(hs>=0. && isfinite(hs)){
   if(sm>=0.){out[0]=L[0]+sl*(hs-hl);out[1]=L[1]+sl*(hs*sm-hl*ul);out[2]=L[2]+sl*(hs*vl-hl*vl);}
   else{out[0]=R[0]+sr*(hs-hr);out[1]=R[1]+sr*(hs*sm-hr*ur);out[2]=R[2]+sr*(hs*vr-hr*vr);}
   return;
  }
 }
 // HLLE at dry interfaces / inadmissible contact states, recorded method.
 double QL[3]={hl,hl*ul,hl*vl},QR[3]={hr,hr*ur,hr*vr};
 for(int k=0;k<3;++k)out[k]=(sr*L[k]-sl*R[k]+sl*sr*(QR[k]-QL[k]))/(sr-sl);
}

ENTRY void hllc_predict_source(int n,double dry,double dt,const double*rate,double*half,double*qhalf){
 LOOP(n){double h=qhalf[i];qhalf[n+i]+=.5*dt*rate[n+i];qhalf[2*n+i]+=.5*dt*rate[2*n+i];
  half[n+i]=h>dry?qhalf[n+i]/h:0.;half[2*n+i]=h>dry?qhalf[2*n+i]/h:0.;}
}

ENTRY void hllc_faces(int n,int nf,double gravity,double dry,const double*half,
 const double*grad,const double*xy,const double*mid,const double*normal,
 const int*owner,const int*neighbour,const int*selected,const double*external,const double*external_trace,
 double*fl,double*fr,double*speed){
 LOOP(nf){int l=owner[i],r=neighbour[i];double nx=normal[i],ny=normal[nf+i];
  if(selected[i]){double h=fmax(0.,half[l]-half[3*n+l]);
   fl[i]=fr[i]=external[i];
   fl[nf+i]=fr[nf+i]=external[nf+i]-.5*gravity*h*h*nx;
   fl[2*nf+i]=fr[2*nf+i]=external[2*nf+i]-.5*gravity*h*h*ny;
   double hb=external_trace[i],ub=hb>dry?(external_trace[nf+i]*nx+external_trace[2*nf+i]*ny)/hb:0.;
   speed[i]=fmax(fabs(half[n+l]*nx+half[2*n+l]*ny)+sqrt(gravity*h),fabs(ub)+sqrt(gravity*fmax(hb,0.)));continue;
  }
  double L[4],R[4];for(int m=0;m<4;++m){
   L[m]=half[m*n+l]+grad[2*m*n+l]*(mid[i]-xy[l])+grad[(2*m+1)*n+l]*(mid[nf+i]-xy[n+l]);
   R[m]=r<0?L[m]:half[m*n+r]+grad[2*m*n+r]*(mid[i]-xy[r])+grad[(2*m+1)*n+r]*(mid[nf+i]-xy[n+r]);
  }
  // Cell-average dry inventory cannot represent the partially wet wedge
  // implied by extrapolating a dry cell's bed. Retain the actual cell-bed
  // step at wet/dry interfaces; otherwise a static lake spuriously wets it.
  bool ldry=half[l]-half[3*n+l]<=dry,rdry=r>=0 && half[r]-half[3*n+r]<=dry;
  if(ldry || rdry){L[3]=half[3*n+l];R[3]=r<0?L[3]:half[3*n+r];}
  if(ldry){L[0]=L[3];L[1]=L[2]=0.;}
  if(rdry){R[0]=R[3];R[1]=R[2]=0.;}
  if(r<0){double un=R[1]*nx+R[2]*ny;R[1]-=2.*un*nx;R[2]-=2.*un*ny;}
  double z=fmax(L[3],R[3]),hl=fmax(0.,L[0]-L[3]),hr=fmax(0.,R[0]-R[3]);
  double hls=fmax(0.,L[0]-z),hrs=fmax(0.,R[0]-z),out[3],s;
  hllc_flux(hls,L[1]*nx+L[2]*ny,-L[1]*ny+L[2]*nx,
            hrs,R[1]*nx+R[2]*ny,-R[1]*ny+R[2]*nx,gravity,dry,out,s);
  double el=fmax(0.,half[l]-L[3]),er=r<0?el:fmax(0.,half[r]-R[3]);
  double pl=.5*gravity*(hl*hl-hls*hls-el*el),pr=.5*gravity*(hr*hr-hrs*hrs-er*er);
  fl[i]=fr[i]=r<0?0.:out[0];
  fl[nf+i]=(out[1]+pl)*nx-out[2]*ny;fl[2*nf+i]=(out[1]+pl)*ny+out[2]*nx;
  fr[nf+i]=(out[1]+pr)*nx-out[2]*ny;fr[2*nf+i]=(out[1]+pr)*ny+out[2]*nx;
  speed[i]=s;
 }
}

ENTRY void hllc_budget(int n,int nf,double dt,const double*q,const double*area,
 const double*length,const int*off,const int*faces,const int*signs,const double*fl,
 const double*speed,const int*selected,const int*status,double*theta,double*frequency){
 LOOP(n){double outgoing=0.,prescribed=0.,freq=0.;
  for(int k=off[i];k<off[i+1];++k){int f=faces[k];double debit=fmax(0.,signs[k]*fl[f])*length[f];
   if(selected[f])prescribed+=debit;else outgoing+=debit;freq+=speed[f]*length[f];}
  frequency[i]=status[i]?1./0.:freq/area[i];theta[i]=1.;
  // Named exterior fluxes must remain exact. Reserve their outward debit
  // before limiting internal transport, rather than rejecting every boundary
  // cell that needs any internal limiter. A negative sentinel denotes a truly
  // unfunded prescribed withdrawal and still rejects the entire trial.
  double available=q[i]*area[i]/dt;
  if(prescribed>available)theta[i]=-1.;
  else if(outgoing>0. && outgoing>=available-prescribed)
   theta[i]=fmin(1.,(available-prescribed)/outgoing)*(1.-32.*2.2204460492503131e-16);
 }
}

ENTRY void hllc_limit(int n,int nf,double dry,const double*q,const double*theta,
 const int*owner,const int*neighbour,const int*selected,double*fl,double*fr,int*status){
 LOOP(nf){status[i]=0;double water=fl[i];int donor=water<0.?neighbour[i]:owner[i];
  if(donor<0 || water==0.)continue;
  if(selected[i]){status[i]=theta[donor]<0.?1:0;continue;}
  if(theta[donor]>=1.)continue;
  double factor=fmax(0.,theta[donor]);double delta=(factor-1.)*water;fl[i]=fr[i]=water*factor;
  if(q[donor]>dry)for(int k=1;k<3;++k){double d=delta*q[k*n+donor]/q[donor];fl[k*nf+i]+=d;fr[k*nf+i]+=d;}
 }
}

ENTRY void hllc_update(int n,int nf,double dt,double dry,const double*q,
 const double*area,const double*length,const int*off,const int*faces,const int*signs,
 const double*fl,const double*fr,const int*face_status,double*next,double*status){
 LOOP(n){double v[3]={q[i],q[n+i],q[2*n+i]};bool bad=false;
  for(int k=off[i];k<off[i+1];++k){int f=faces[k],s=signs[k];double a=dt*s*length[f]/area[i];
   for(int m=0;m<3;++m)v[m]-=a*(s>0?fl[m*nf+f]:fr[m*nf+f]);bad=bad||face_status[f];}
  if(v[0]<=dry)v[1]=v[2]=0.;
  for(int m=0;m<3;++m){next[m*n+i]=v[m];bad=bad||!isfinite(v[m]);}
  status[i]=bad||v[0]<0.?1.:0.;
 }
}
