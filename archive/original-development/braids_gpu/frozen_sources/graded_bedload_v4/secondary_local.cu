// Fused FP64 batch of unchanged local BRAIDS secondary-flow helper equations.
// Inputs/outputs are component-major; no operator splitting or advection here.
extern "C" __global__ void secondary_local(int nc,const double* in,double* out,int* status,double g){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
 double h=in[c],u=in[nc+c],v=in[2*nc+c],dudx=in[3*nc+c],dudy=in[4*nc+c];
 double dvdx=in[5*nc+c],dvdy=in[6*nc+c],n=in[7*nc+c],curv=in[8*nc+c];
 double stored=in[9*nc+c],dt=in[10*nc+c],qx=in[11*nc+c],qy=in[12*nc+c];
 double espir=in[13*nc+c],beta_spiral=in[14*nc+c];
 double raw=0.,eq=0.,time_a=1.7976931348623157e308,updated=stored,increment=0.;
 double alpha=0.,xx=0.,xy=0.,yy=0.,signal=0.,speed=sqrt(u*u+v*v);
 int code=0;bool active_eq=false,active_adapt=false,active_stress=false;
 bool valid=isfinite(u)&&isfinite(v)&&isfinite(dudx)&&isfinite(dudy)&&isfinite(dvdx)&&isfinite(dvdy);
 if(valid&&speed>=1.e-3){raw=-(u*u*dvdx-v*v*dudy+u*v*(dvdy-dudx))/(speed*speed*speed);valid=isfinite(raw);if(!valid)raw=0.;}
 if(!valid)code|=1;
 valid=isfinite(h)&&isfinite(u)&&isfinite(v)&&isfinite(curv)&&h>=0.;
 if(valid&&h>0.){
  if(!isfinite(speed))valid=false;
  else if(speed>=1.e-3){eq=h*curv*speed;valid=isfinite(eq);if(valid)active_eq=true;else eq=0.;}
 }if(!valid)code|=2;
 valid=isfinite(h)&&isfinite(u)&&isfinite(v)&&isfinite(n)&&isfinite(g)&&h>=0.&&n>0.&&g>0.;
 if(valid&&h>0.){
  if(!isfinite(speed))valid=false;
  else if(speed>=1.e-3){double chezy=pow(h,1./6.)/n;
   valid=isfinite(chezy)&&chezy>0.;
   if(valid){double a=sqrt(g)/(.41*fmax(chezy,20.));valid=isfinite(a)&&a>0.&&a<.5;
    if(valid){double length=(1.-2.*a)*h/(2.*.41*.41*a);time_a=length/speed;
     valid=isfinite(time_a)&&time_a>0.;if(valid)active_adapt=true;else time_a=1.7976931348623157e308;}}
  }
 }if(!valid)code|=4;
 valid=isfinite(h)&&isfinite(stored)&&isfinite(eq)&&isfinite(time_a)&&isfinite(dt)&&h>=0.&&time_a>0.&&dt>=0.;
 if(valid&&dt>0.){
  double target=h*eq;valid=isfinite(target);
  if(valid){if(time_a<=dt/50.){increment=target-stored;updated=target;}
   else{double ratio=dt/time_a,fraction;
    if(ratio<=0.)fraction=0.;
    else if(ratio<1.e-5)fraction=ratio*(1.+ratio*(-.5+ratio*(1./6.+ratio*(-1./24.+ratio*(1./120.-ratio/720.)))));
    else if(ratio>=50.)fraction=1.;else fraction=1.-exp(-ratio);
    increment=(target-stored)*fraction;updated=stored+increment;}
   valid=isfinite(updated)&&isfinite(increment);if(!valid){updated=stored;increment=0.;}}
 }if(!valid)code|=8;
 double intensity=h>0.?updated/h:0.;
 valid=isfinite(h)&&isfinite(u)&&isfinite(v)&&isfinite(n)&&isfinite(g)&&isfinite(espir)&&isfinite(intensity)&&isfinite(qx)&&isfinite(qy);
 valid=valid&&h>=0.&&n>0.&&g>0.&&espir>=0.;
 if(valid&&h>0.){double magnitude=sqrt(qx*qx+qy*qy);
  if(speed>0.&&magnitude>0.){double chezy=pow(h,1./6.)/n;valid=isfinite(chezy)&&chezy>0.;
   if(valid){alpha=12.5*espir*(1.-sqrt(g)/(.4*chezy))*intensity/speed;
    double tx=u+alpha*v,ty=v-alpha*u,direction=sqrt(tx*tx+ty*ty);
    if(direction<=2.2204460492503131e-16){qx=0.;qy=0.;}else{qx=magnitude*tx/direction;qy=magnitude*ty/direction;}
    valid=isfinite(qx)&&isfinite(qy)&&isfinite(alpha);}}
 }if(!valid)code|=16;
 valid=isfinite(h)&&isfinite(u)&&isfinite(v)&&isfinite(n)&&isfinite(g)&&isfinite(beta_spiral)&&isfinite(intensity);
 valid=valid&&h>=0.&&n>0.&&g>0.&&beta_spiral>=0.;
 if(valid&&h>0.&&beta_spiral>0.){
  if(!isfinite(speed))valid=false;
  else if(speed>=1.e-3){double chezy=pow(h,1./6.)/n;valid=isfinite(chezy)&&chezy>0.;
   if(valid){double a=sqrt(g)/(.41*chezy),coefficient=beta_spiral*(5.*a-15.6*(a*a)+37.5*(a*a*a));
    double beta=coefficient*intensity/speed;xx=-2.*h*beta*u*v;xy=h*beta*(u*u-v*v);yy=-xx;
    signal=2.*fabs(coefficient*intensity);valid=isfinite(xx)&&isfinite(xy)&&isfinite(yy)&&isfinite(signal);
    if(!valid){xx=0.;xy=0.;yy=0.;signal=0.;}else active_stress=signal>0.;}}
 }if(!valid)code|=32;
 out[c]=raw;out[nc+c]=eq;out[2*nc+c]=time_a;out[3*nc+c]=updated;out[4*nc+c]=increment;
 out[5*nc+c]=qx;out[6*nc+c]=qy;out[7*nc+c]=alpha;out[8*nc+c]=xx;out[9*nc+c]=xy;
 out[10*nc+c]=yy;out[11*nc+c]=signal;out[12*nc+c]=active_eq;out[13*nc+c]=active_adapt;out[14*nc+c]=active_stress;
 status[c]=code;
}
