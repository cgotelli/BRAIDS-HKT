// Research-only one-block device-resident Jacobi convergence candidate.
// One block supplies a global barrier at every iteration. All arithmetic
// inside each class/cell update duplicates adapt_jacobi; no atomics/history.

extern "C" __global__ void adapt_persistent(
 const int nc,const int ng,const int nf,const int nb,const double* equilibrium,
 const double* d50,const double* normal,const int* owner,const int* neighbour,
 const int* bid,const int* offset,const int* faces,const int* signs,
 const int* bed_boundary_mode,const double* boundary_span,const double* feed,
 const double* face_weight,const double* total_weight,const double* distance,
 const double length_diameters,const int max_iterations,double* first,
 double* second,double* capacity,double* diagnostic){
 const int tid=threadIdx.x;
 __shared__ double maxima[256];
 double local_scale=0.;
 for(int index=tid;index<2*ng*nc;index+=blockDim.x)
  local_scale=fmax(local_scale,fabs(equilibrium[index]));
 maxima[tid]=local_scale;__syncthreads();
 for(int stride=blockDim.x/2;stride>0;stride/=2){
  if(tid<stride)maxima[tid]=fmax(maxima[tid],maxima[tid+stride]);
  __syncthreads();
 }
 if(tid==0){diagnostic[0]=-2.;diagnostic[1]=0.;
  diagnostic[2]=1.e-11*fmax(maxima[0],1.e-20);}
 __syncthreads();
 const double* previous=equilibrium;
 double* following=first;
 for(int iteration=1;iteration<=max_iterations;++iteration){
  double local_change=0.;
  for(int index=tid;index<nc*ng;index+=blockDim.x){
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
   double change=(isfinite(qx)&&isfinite(qy))?
    fmax(fabs(qx-previous[x]),fabs(qy-previous[y])):
    __longlong_as_double(0x7ff0000000000000LL);
   local_change=fmax(local_change,change);
  }
  maxima[tid]=local_change;__syncthreads();
  for(int stride=blockDim.x/2;stride>0;stride/=2){
   if(tid<stride)maxima[tid]=fmax(maxima[tid],maxima[tid+stride]);
   __syncthreads();
  }
  if(tid==0){diagnostic[1]=maxima[0];
   if(!isfinite(maxima[0]))diagnostic[0]=-1.;
   else if(maxima[0]<=diagnostic[2])diagnostic[0]=(double)iteration;}
  __syncthreads();
  if(diagnostic[0]>0.){
   for(int index=tid;index<2*ng*nc;index+=blockDim.x)
    capacity[index]=following[index];
   return;
  }
  if(diagnostic[0]==-1.)return;
  previous=following;
  following=(following==first)?second:first;
  __syncthreads();
 }
}
