// BRAIDS deposition-only extension, not a FLOW Galappatti exchange selector.
// One cell owns every fraction; settling uses the original total concentration.
// Exponential exchange is exact for depth and hindered velocity frozen during
// this source step. There is no equilibrium capacity or entrainment.
ENTRY void settling_only_exchange(int n,int ng,int stride,double dt,double mf,double dry,
 double cref,double temp,const double*diam,const double*q,const double*mass,
 const double*law_params,double*next,double*delta,double*diagnostic){
 LOOP(n){double h=q[i],total=0.;
  for(int k=0;k<ng;++k)total+=h>dry?mass[k*n+i]/h:0.;
  double hinder=pow(fmax(0.,1.-total/cref),5.),nu=4.e-5/(20.+temp);
  for(int k=0;k<ng;++k){int a=k*n+i;double d=diam[k],relative=law_params[k*stride+14],ws;
   // FLOW SettleFrm2 (noncohesive, no salinity/flocculation), nominal class
   // diameter. Do not import Galappatti's 1 mm/s floor or SedThr gate.
   if(d<1.5*64.e-6)ws=relative*9.81*d*d/(18.*nu);
   else if(d<.001){double coef=d<128.e-6?(-2.9912/64.e-6)*d+15.9824:10.;
    ws=coef*nu/d*(sqrt(1.+relative*9.81*d*d*d/(100.*nu*nu))-1.);}
   else ws=1.1*sqrt(relative*9.81*d);
   ws*=hinder;
   // expm1 preserves a small transfer without subtracting nearly equal masses.
   double deposited=h>dry?-mass[a]*expm1(-dt*ws/h):mass[a];
   next[a]=mass[a]-deposited;
   delta[a]=(h>dry?mf:1.)*deposited;
   diagnostic[a]=ws;diagnostic[n*ng+a]=0.;diagnostic[2*n*ng+a]=h>dry?ws:0.;
  }
 }
}
