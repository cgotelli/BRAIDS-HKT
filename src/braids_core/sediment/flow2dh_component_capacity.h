// Rotationally covariant component limits in the local transport frame.
// A vanishing transverse component must not suppress a finite longitudinal
// gradient. Each local component retains its own Barth--Jespersen bounds;
// rotate the limited gradients back once for the unchanged face operator.
ENTRY void capacity_objective_gradients(int n,int nf,const double*cell,const int*off,
 const int*faces,const int*owner,const int*nb,const double*delta,const double*inv,
 const double*disp,double*grad,double*limiter){
 LOOP(n){
  double value[2]={cell[3*n+i],cell[4*n+i]};
  double speed=hypot(value[0],value[1]);
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
  for(int c=0;c<2;++c){gx[c]=inv[i]*bx[c]+inv[n+i]*by[c];gy[c]=inv[n+i]*bx[c]+inv[2*n+i]*by[c];}
  double lx[2]={tx*gx[0]+ty*gx[1],-ty*gx[0]+tx*gx[1]};
  double ly[2]={tx*gy[0]+ty*gy[1],-ty*gy[0]+tx*gy[1]};
  double phi[2]={1.,1.},eps=64.*2.2204460492503131e-16*speed;
  for(int k=off[i];k<off[i+1];++k){int f=faces[k];if(nb[f]<0)continue;
   int b=owner[f]==i?0:2;double dx=disp[b*nf+f],dy=disp[(b+1)*nf+f];
   for(int c=0;c<2;++c){double increment=lx[c]*dx+ly[c]*dy;
    if(increment>eps)phi[c]=fmin(phi[c],(hi[c]-centre[c])/increment);
    else if(increment<-eps)phi[c]=fmin(phi[c],(lo[c]-centre[c])/increment);
   }
  }
  for(int c=0;c<2;++c){phi[c]=fmax(0.,fmin(1.,phi[c]));lx[c]*=phi[c];ly[c]*=phi[c];}
  // Retain the diagnostic scalar ABI: minimum of two factors, not a common
  // factor applied to the physical vector. Restart identity states this.
  limiter[i]=fmin(phi[0],phi[1]);
  grad[i]=tx*lx[0]-ty*lx[1];grad[n+i]=tx*ly[0]-ty*ly[1];
  grad[2*n+i]=ty*lx[0]+tx*lx[1];grad[3*n+i]=ty*ly[0]+tx*ly[1];
 }
}
