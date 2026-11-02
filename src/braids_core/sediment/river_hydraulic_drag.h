// Cell-local finite-volume source quadrature. Constitutive laws are prepended
// unchanged from river_roughness.source(); no cell/face averaging or wet halo.
ENTRY void river_cell_drag(int n,double dt,double gravity,double dry,int mode,
 double value,int feedback,const double*components,double*q,double*chezy){
 LOOP(n){double h=q[i];
  if(h<=dry){chezy[i]=river_chezy(mode,dry,value);q[n+i]=q[2*n+i]=0.;continue;}
  double c=river_chezy(mode,h,value);
  if(feedback){
   double ks2=0.;for(int k=2;k<5;++k){double r=components[k*n+i];ks2+=r*r;}
   double ks=fmax(.001,fmax(1e-6,fmin(sqrt(ks2),.5*h)));
   c=18.*log10(fmax(12.*h/ks,1.0129));
  }
  chezy[i]=c;double speed=sqrt(q[n+i]*q[n+i]+q[2*n+i]*q[2*n+i])/h;
  double factor=1./(1.+dt*gravity*speed/(c*c*h));
  q[n+i]*=factor;q[2*n+i]*=factor;
 }
}
