// Algebraic equilibrium suspended flux, added ONCE to reconstructed bedload.
// FLOW eqtran non-suspfrac branch. No resident suspended inventory.
// Inputs distinguish physical characteristic velocity from EFVLM/EFUMN.
// Return signed suspended magnitude (kg/m/s), its Cartesian vector and DSS.
// Wang-Fredsoe's signed residual is intentional in this algebraic total.
RIVER_LOCAL int river_total_suspended(const double*p,double h,double u,double v,
 double ue,double ve,double c,double d,double d90,double hide,double z0,
 double fraction,double dss,double&mag,double&x,double&y,double&nextdss){
 mag=0.;x=0.;y=0.;nextdss=dss;
 int law=int(p[0]);
 if(law==2||law==12||law==14||law>=16)return 0;
 const double physical=hypot(u,v),effective=hypot(ue,ve),eps=double(1.e-6f);
 const double factor=physical/(physical+eps),efx=ue*factor,efy=ve*factor;
 if(law==-1){
  // erosed NON-suspfrac: nominal DSS, zero fall-speed profile, sigmol=1.
  // Unlike the split route, no accepted suspended-size update is consumed.
  // Explicitly named in the resolver/UI; not physical settling in disguise.
  double r[12];
  int bad=river_vr_current(p,h,u,v,d,d90,z0,fraction,d,0.,r,1.);
  if(bad)return bad;
  // tram1's fraction-dependent cutoff must precede undoing the fraction.
  // The shared face owner later applies the donor's fraction exactly once.
  if(fraction>0.){double a=r[2]*fmax(.01,h)/fraction;x=a*efx;y=a*efy;}
  mag=hypot(x,y);
 }else if(law==11){
  double b,s,gx,gy,t;river_current_vector_law(p,efx,efy,fmax(.01,h),d,b,s,gx,gy,t);
  double speed=hypot(efx,efy);mag=p[13]*s;
  if(speed>0.){x=mag*efx/speed;y=mag*efy/speed;}
 }else{
  double b,s,der,shear=physical*.41/log1p(h*double(.368f)/z0);
  river_scalar_law(p,effective,c,fmax(.01,h),d,d90,hide,b,s,der,shear);
  double a=p[13]*s/(effective+eps);
  x=a*efx;y=a*efy;mag=a*hypot(efx,efy);
 }
 return isfinite(mag)&&isfinite(x)&&isfinite(y)&&isfinite(nextdss)?0:4;
}
// Frozen-depth/material/roughness principal part, not a new physical law.
// Differentiate the SAME algebraic assembly, including both native epsilon
// denominators. A symmetric radial perturbation avoids axis-dependent noise.
RIVER_LOCAL int river_total_suspended_radial(const double*p,double h,double u,double v,
 double velocity_factor,double c,double d,double d90,double hide,double z0,
 double fraction,double dss,double&mag,double&der){
 double x,y,nd;
 int bad=river_total_suspended(p,h,u,v,u*velocity_factor,v*velocity_factor,c,
  d,d90,hide,z0,fraction,dss,mag,x,y,nd);
 if(bad)return bad;
 double speed=hypot(u,v);der=0.;
 if(speed<=1.e-12)return 0;
 double delta=fmin(.25*speed,1.e-5*fmax(speed,1.e-3)),plus,minus;
 double rp=(speed+delta)/speed,rm=(speed-delta)/speed;
 bad=river_total_suspended(p,h,u*rp,v*rp,u*rp*velocity_factor,v*rp*velocity_factor,c,
  d,d90,hide,z0,fraction,dss,plus,x,y,nd);
 if(bad)return bad;
 bad=river_total_suspended(p,h,u*rm,v*rm,u*rm*velocity_factor,v*rm*velocity_factor,c,
  d,d90,hide,z0,fraction,dss,minus,x,y,nd);
 der=(plus-minus)/(2.*delta);
 return bad;
}
