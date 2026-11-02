// Current-only, noncohesive compbsskin (FLOW): replace RP_TAUB only.
// Explicit sand roughness and molecular viscosity; no inferred grain multiplier,
// hydraulic drag mutation, mud/fluff mixture or wave closure.
// The analytic elasticity is d(log(tau))/d(log(physical speed)), at fixed depth.
RIVER_LOCAL int river_skin_stress(double speed,double depth,double rho,double ks,double nu,
 double&tau,double&elasticity,double&rough,double&smooth){
 tau=0.;elasticity=2.;rough=0.;smooth=0.;
 if(!isfinite(speed)||!isfinite(depth)||!isfinite(rho)||!isfinite(ks)||!isfinite(nu)
    ||speed<0.||depth<=0.||rho<=0.||ks<0.||nu<=0.)return 1;
 double z0=fmax(ks/30.,1.e-12),den=log(depth/z0)-1.;
 if(den==0.||!isfinite(den))return 1;
 double re=fmax(speed,1.e-12)*depth/nu,term=pow(re,-.08);
 rough=pow(.40/den,2.);
 smooth=1.615e-4*exp(6.*term);
 if(!isfinite(rough)||!isfinite(smooth))return 1;
 if(speed>=1.e-6){
  bool is_rough=rough>=smooth;
  tau=rho*(is_rough?rough:smooth)*speed*speed;
  elasticity=is_rough?2.:2.-.48*term;
 }
 return (!isfinite(tau)||!isfinite(elasticity))?1:0;
}
ENTRY void skin_stress_probe(int n,const double*speed,const double*depth,const double*rho,
 const double*ks,const double*nu,double*out,int*status){
 LOOP(n){status[i]=river_skin_stress(speed[i],depth[i],rho[i],ks[i],nu[i],
  out[i],out[n+i],out[2*n+i],out[3*n+i]);}
}
