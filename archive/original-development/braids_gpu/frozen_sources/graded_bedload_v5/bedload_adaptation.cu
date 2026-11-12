// Native fixed-grid classwise steady first-order streamline adaptation.
// Arrays are SoA and zero-based; adjacency order follows the prepared native
// snapshot. These kernels alter no accepted state and use no atomics.

extern "C" __global__ void adapt_prepare(
 const int nc,const int nf,const double* q,const double* xy,const double* mid,
 const double* normal,const double* length,const double* area,const int* owner,
 const int* neighbour,const int* bid,const int* offset,const int* faces,
 const int* signs,const int* mode,const double dry,const double dx,
 const double dy,double* face_weight,double* total_weight,double* distance){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
 double h=q[c],u=0.,v=0.;if(h>dry){u=q[nc+c]/h;v=q[2*nc+c]/h;}
 double speed=sqrt(u*u+v*v),total=0.,dsum=0.;
 for(int k=offset[c];k<offset[c+1];++k){
  int f=faces[k],n=signs[k]>0?neighbour[f]:owner[f];
  double outward_x=signs[k]*normal[f],outward_y=signs[k]*normal[nf+f];
  double weight=0.,segment=0.;
  if(speed>0.&&h>dry){
   if(n>=0){
    double hn=q[n],un=0.,vn=0.;if(hn>dry){un=q[nc+n]/hn;vn=q[2*nc+n]/hn;}
    double fu=.5*(u+un),fv=.5*(v+vn);
    double inflow=-(fu*outward_x+fv*outward_y);
    if(inflow>0.){
     weight=inflow*length[f];double fspeed=sqrt(fu*fu+fv*fv);
     double rx=xy[c]-xy[n],ry=xy[nc+c]-xy[nc+n];
     if(fspeed>0.)segment=fabs((rx*fu+ry*fv)/fspeed);
     if(segment<=64.*2.2204460492503131e-16*fmax(dx,dy))segment=sqrt(rx*rx+ry*ry);
    }
   }else{
    int b=bid[f];
    if(b>=0&&(mode[b]==1||mode[b]==6||mode[b]==7||mode[b]==8||mode[b]==9)){
     double inflow=-(u*outward_x+v*outward_y);
     if(inflow>0.){
      weight=inflow*length[f];
      double rx=xy[c]-mid[f],ry=xy[nc+c]-mid[nf+f];
      segment=fabs((rx*u+ry*v)/speed);
      if(segment<=64.*2.2204460492503131e-16*fmax(dx,dy))segment=sqrt(rx*rx+ry*ry);
     }
    }
   }
  }
  face_weight[k]=weight;total+=weight;dsum+=weight*segment;
 }
 total_weight[c]=total;double d=0.;
 if(total>0.){
  d=dsum/total;
  if(d<=64.*2.2204460492503131e-16*fmax(dx,dy))d=area[c]*speed/total;
 }
 distance[c]=d;
}

extern "C" __global__ void adapt_jacobi(
 const int nc,const int ng,const int nf,const int nb,const double* previous,
 const double* equilibrium,const double* d50,const double* normal,
 const int* owner,const int* neighbour,const int* bid,const int* offset,
 const int* faces,const int* signs,const int* bed_boundary_mode,
 const double* boundary_span,const double* feed,const double* face_weight,
 const double* total_weight,const double* distance,const double length_diameters,
 double* following,double* change){
 int index=blockIdx.x*blockDim.x+threadIdx.x;if(index>=nc*ng)return;
 int g=index/nc,c=index-g*nc,x=index,y=(ng+g)*nc+c;
 double qx=equilibrium[x],qy=equilibrium[y],total=total_weight[c];
 if(total>0.){
  double upstream_x=0.,upstream_y=0.;
  for(int k=offset[c];k<offset[c+1];++k){
   double weight=face_weight[k];if(weight<=0.)continue;
   int f=faces[k],n=signs[k]>0?neighbour[f]:owner[f];
   double source_x=0.,source_y=0.;
   if(n>=0){source_x=previous[g*nc+n];source_y=previous[(ng+g)*nc+n];}
   else{
    int b=bid[f];if(b<0||b>=nb)continue;
    if(bed_boundary_mode[b]==1){source_x=equilibrium[x];source_y=equilibrium[y];}
    else if(boundary_span[b]>0.){
     double density=feed[g*nb+b]/boundary_span[b];
     source_x=-signs[k]*normal[f]*density;
     source_y=-signs[k]*normal[nf+f]*density;
    }
   }
   upstream_x+=weight*source_x;upstream_y+=weight*source_y;
  }
  double attenuation=exp(-distance[c]/(length_diameters*d50[g]));
  qx=(1.-attenuation)*equilibrium[x]+attenuation*upstream_x/total;
  qy=(1.-attenuation)*equilibrium[y]+attenuation*upstream_y/total;
 }
 following[x]=qx;following[y]=qy;
 change[index]=(isfinite(qx)&&isfinite(qy))?
  fmax(fabs(qx-previous[x]),fabs(qy-previous[y])):
  __longlong_as_double(0x7ff0000000000000LL);
}
