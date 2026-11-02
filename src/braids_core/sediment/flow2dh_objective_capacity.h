// Rotationally covariant common-factor capacity limiter.
// The physical transport vector defines the local axes, not geographic x/y.
// Reuse the accepted gradient stencil and face reconstruction without changes.
ENTRY void capacity_objective_gradients(int n,int nf,const double*cell,const int*off,
 const int*faces,const int*owner,const int*nb,const double*delta,const double*inv,
 const double*disp,double*grad,double*limiter){
 LOOP(n){
  double value[2]={cell[3*n+i],cell[4*n+i]};
  double speed=hypot(value[0],value[1]);
  // A zero vector has no frame. A zero reconstruction is objective and bounded.
  if(speed==0.){limiter[i]=0.;for(int c=0;c<4;++c)grad[c*n+i]=0.;continue;}
  double tx=value[0]/speed,ty=value[1]/speed;
  double centre[2]={tx*value[0]+ty*value[1],-ty*value[0]+tx*value[1]};
  double lo[2]={centre[0],centre[1]},hi[2]={centre[0],centre[1]};
  double bx[2]={0.,0.},by[2]={0.,0.},gx[2],gy[2];int na=off[n];
  for(int k=off[i];k<off[i+1];++k){
   int f=faces[k],j=owner[f]==i?nb[f]:owner[f];if(j<0)continue;
   double dx=delta[k],dy=delta[na+k],w=1./(dx*dx+dy*dy);
   double vx=cell[3*n+j],vy=cell[4*n+j],local[2]={tx*vx+ty*vy,-ty*vx+tx*vy};
   for(int c=0;c<2;++c){
    lo[c]=fmin(lo[c],local[c]);hi[c]=fmax(hi[c],local[c]);
    double dv=cell[(3+c)*n+j]-value[c];bx[c]+=w*dx*dv;by[c]+=w*dy*dv;
   }
  }
  for(int c=0;c<2;++c){gx[c]=inv[i]*bx[c]+inv[n+i]*by[c];
   gy[c]=inv[n+i]*bx[c]+inv[2*n+i]*by[c];}
  double phi=1.,eps=64.*2.2204460492503131e-16*speed;
  for(int k=off[i];k<off[i+1];++k){int f=faces[k];if(nb[f]<0)continue;
   int b=owner[f]==i?0:2;double dx=disp[b*nf+f],dy=disp[(b+1)*nf+f];
   double ix=gx[0]*dx+gy[0]*dy,iy=gx[1]*dx+gy[1]*dy;
   double increment[2]={tx*ix+ty*iy,-ty*ix+tx*iy};
   for(int c=0;c<2;++c){
    if(increment[c]>eps)phi=fmin(phi,(hi[c]-centre[c])/increment[c]);
    else if(increment[c]<-eps)phi=fmin(phi,(lo[c]-centre[c])/increment[c]);
   }
  }
  phi=fmax(0.,fmin(1.,phi));limiter[i]=phi;
  for(int c=0;c<2;++c){grad[2*c*n+i]=phi*gx[c];grad[(2*c+1)*n+i]=phi*gy[c];}
 }
}
