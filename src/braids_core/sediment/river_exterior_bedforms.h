// Appended to the shared bedform predictor source, on OpenMP or CUDA.
BDF_LOCAL double exterior_bdf_ks(double h,int n,int i,const double*state){
 double k2=0.;for(int k=2;k<5;++k){double r=state[k*n+i];k2+=r*r;}
 return fmax(.001,fmax(1e-6,fmin(sqrt(k2),.5*h)));
}
BDF_LOCAL double exterior_bdf_chezy(double h,int n,int i,const double*state){
 double ks=exterior_bdf_ks(h,n,i,state);
 return 18.*log10(fmax(12.*h/ks,1.0129));
}
ENTRY void exterior_bdf_weights(int ne,int nf,int nc,int nb,double dry,
 const int*faces,const int*owner,const int*bid,const int*mode,const double*q,const double*z,
 const double*bed,const double*length,const double*stage,const double*forcing,const double*state,double*weights){
 LOOP(ne){int f=faces[i],o=owner[i],b=bid[f];weights[f]=0.;
  if(mode[b]==9){
   double h=fmax(forcing[nb+b]-bed[f],0.);
   if(h>dry)weights[f]=h*length[f];
   continue;
  }
  if(mode[b]!=6 || stage[nb+b]<=0.)continue;
  double h=fmax(q[o]+z[o]-bed[f],0.),section=fmax(stage[b]/stage[nb+b]-bed[f],0.);
  if(h>dry && section>0.)weights[f]=pow(section,1.5)*exterior_bdf_chezy(h,ne,i,state)*length[f];
 }
}
ENTRY void exterior_bdf_resistance(int ne,double dry,const double*q,const double*state,double*out){
 LOOP(ne){double h=fmax(dry,q[i]),c=exterior_bdf_chezy(h,ne,i,state);
  out[i]=c;out[ne+i]=h/(exp(1.)*expm1(.41*c/sqrt(9.81)));
  out[2*ne+i]=exterior_bdf_ks(h,ne,i,state);
 }
}
ENTRY void exterior_bdf_faces(int ne,int nf,int nc,double dry,const int*faces,const int*owner,
 const double*q,const double*trace,const double*roughness,const double*weights,
 double*depth,double*wetweights,double*out){
 LOOP(ne){int f=faces[i];bool active=q[owner[i]]>dry && trace[i]>dry;
  depth[f]=fmax(.01,trace[i]);
  wetweights[f]=active?weights[f]:0.;wetweights[nf+f]=active?weights[nf+f]:0.;
  for(int k=0;k<3;++k)out[k*nf+f]=active?roughness[k*ne+i]:0.;
 }
}
