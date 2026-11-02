// Current-only FLOW trab11/trab12, solid-volume transport.
// Components are along the effective GLOBAL velocity. The capacity owns its
// direction: eqtran does not apply its scalar spiral correction to these laws.
// Gradient is d|qbed|/d(effective ux,uy) at frozen depth/material/roughness.
RIVER_LOCAL void river_current_vector_law(const double*p,double u,double v,double h,double d,
 double&bed,double&sus,double&gx,double&gy,double&theta){
 bed=0.;sus=0.;gx=0.;gy=0.;theta=0.;
 const int law=int(p[0]);const double speed=hypot(u,v),rel=p[14],nu=p[15];
 if(law==11){
  if(speed<double(1.e-6f)||h>200.||h<double(.01f))return;
  const double d90=p[24]*d,ds=pow(9.81*rel/(nu*nu),double(1.f/3.f))*d;
  const double critical=d<=double(.0005f)?
   double(.19f)*pow(d,double(.1f))*log10(4.*h/d90):
   8.5*pow(d,double(.6f))*log10(4.*h/d90);
  if(speed<=critical)return;
  const double asb=double(.005f)*h*pow(d/h/(rel*9.81*d),double(1.2f));
  const double ass=double(.012f)*d*pow(ds,-double(.6f))/pow(rel*9.81*d,double(1.2f));
  const double ex=speed-critical,power=double(2.4f),term=pow(ex,power);
  bed=p[1]*asb*speed*term;sus=p[1]*ass*speed*term;
  const double radial=p[1]*asb*(term+speed*power*pow(ex,power-1.));
  gx=radial*u/speed;gy=radial*v/speed;
  // RouZ0 and AlfaUrms are wave-related here: zero wave forcing makes their
  // contribution zero. They remain explicit saved method parameters.
 }else if(law==12){
  if(speed<1.e-6||h>200.||h<.01)return;
  const double z0=d/p[26],drag=pow(double(.41f)/(1.+log(z0/h)),2.);
  const double ds=pow(9.81*rel/(nu*nu),1./3.)*d;
  const double critical=.30/(1.+1.2*ds)+.055*(1.-exp(-.020*ds));
  theta=drag*speed*speed/(rel*9.81*d);
  if(theta<=critical)return;
  const double eps=double(1.e-4f),root=sqrt(theta+eps);
  const double x1=12.*root*(theta-critical);
  // Stock trab12 retains WAVEPS and phi even with waves off. The tiny
  // near-threshold branch is therefore angular; do NOT evaluate in face axes.
  const double x2=12.*(.95+.19*cos(-2.*atan2(v,u)))*sqrt(eps)*theta;
  const double scale=p[1]*sqrt(9.81*rel*d*d*d);
  bed=scale*fmax(x1,x2);
  if(x1>=x2){
   const double radial=scale*12.*theta*((theta-critical)/root+2.*root)/speed;
   gx=radial*u/speed;gy=radial*v/speed;
  }else{
   const double k=scale*12.*sqrt(eps)*drag/(rel*9.81*d);
   gx=2.*k*(.95+.19)*u;gy=2.*k*(.95-.19)*v;
  }
 }else{
  bed=nan("");sus=nan("");gx=nan("");gy=nan("");
 }
}
RIVER_LOCAL void river_vector_bed_row(double acceleration,double h,double u,double v,
 double nx,double ny,double velocity_factor,double mag,double gx,double gy,
 double&d0,double&d1,double&d2){
 const double sp2=u*u+v*v,speed=sqrt(sp2),eps=double(1.e-6f);
 const double factor=acceleration*mag/(speed*h);
 const double gn=gx*nx+gy*ny,gt=-gx*ny+gy*nx;
 // erosed EFU/EFV divide by UMOD+eps. This mapping is NOT a frozen
 // constant: include its radial derivative at fixed log-law conversion.
 const double ratio=speed/(speed+eps);
 const double radial=eps/(speed+eps)/(speed+eps)*(gn*u+gt*v)/speed;
 const double beta_n=acceleration*velocity_factor/h*(ratio*gn+radial*u);
 const double beta_t=acceleration*velocity_factor/h*(ratio*gt+radial*v);
 d1=factor*(1.-u*u/sp2)+beta_n*u/sqrt(sp2);
 d2=-factor*u*v/sp2+beta_t*u/sqrt(sp2);
 d0=-d1*u-d2*v;
}
