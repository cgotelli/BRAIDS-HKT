// Auxiliary accepted-state AMR indicators. No state/flux/time-step updates.
#define BRAIDS_AMR_GENERAL_INDICATORS_ABI 2
extern "C" __global__ void gi_restrict(int count,int ns,const int* nodes,
 const int* children,const double* area,double* rows){
 int item=blockIdx.x*blockDim.x+threadIdx.x;if(item>=count*ns)return;
 int node=nodes[item/ns],s=item%ns;double value=0.,total=0.;
 for(int k=0;k<4;++k){int c=children[4*node+k];value+=area[c]*rows[c*ns+s];total+=area[c];}
 rows[node*ns+s]=value/total;
}

extern "C" __global__ void gi_faces(int nf,const int* owner,const int* neighbour,
 const int* ids,const int* wet,const double* eta,const double* u,const double* v,
 const double* manning,const int* material,const int* bid,int finest,double etol,
 double vtol,int* raw,int* cause,int* front,int* moving){
 int f=blockIdx.x*blockDim.x+threadIdx.x;if(f>=nf)return;
 int a=owner[f],b=neighbour[f];
 if(b<0){if(bid[f]>=0){atomicMax(raw+ids[a],finest);atomicOr(cause+a,1<<8);}return;}
 if(wet[a]!=wet[b]){
  atomicMax(raw+ids[a],finest);atomicMax(raw+ids[b],finest);
  atomicOr(cause+a,1<<2);atomicOr(cause+b,1<<2);
  if(wet[a])atomicExch(front+a,1);if(wet[b])atomicExch(front+b,1);
  atomicExch(moving+a,1);atomicExch(moving+b,1);
 }
 double du=u[a]-u[b],dv=v[a]-v[b];
 if(wet[a]&&wet[b]&&(fabs(eta[a]-eta[b])>etol||sqrt(du*du+dv*dv)>vtol)){
  atomicMax(raw+ids[a],finest);atomicMax(raw+ids[b],finest);
  atomicOr(cause+a,1<<3);atomicOr(cause+b,1<<3);
  atomicExch(moving+a,1);atomicExch(moving+b,1);
 }
 if((wet[a]||wet[b])&&(manning[a]!=manning[b]||material[a]!=material[b])){
  atomicMax(raw+ids[a],min(2,finest));atomicMax(raw+ids[b],min(2,finest));
  atomicOr(cause+a,1<<7);atomicOr(cause+b,1<<7);
 }
}

extern "C" __global__ void gi_emerge(int n,int ns,int nb,const int* ids,
 const int* children,const int* bed_index,const double* bulk,const double* rock,
 const double* rows,const int* wet,const double* eta,double wetdepth,int finest,
 int* raw,int* cause,int* moving){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n||!wet[i])return;
 int node=ids[i];if(children[4*node]<0)return;
 for(int k=0;k<4;++k){int c=children[4*node+k];double bed=rock[c],sum=0.;
  for(int j=0;j<nb;++j)sum+=rows[c*ns+bed_index[j]]/bulk[j];
  bed+=sum;if(bed>=eta[i]-wetdepth){atomicMax(raw+node,finest);
   atomicOr(cause+i,1<<12);atomicExch(moving+i,1);return;}
 }
}

extern "C" __global__ void gi_reach(int n,const int* offsets,const int* neighbour,
 const double* distance,const double* old,double* next,int* changed){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;double best=old[i];
 for(int k=offsets[i];k<offsets[i+1];++k){double value=old[neighbour[k]]-distance[k];
  if(value>=0.&&value>best+1e-12)best=value;}
 next[i]=best;if(best>old[i])atomicExch(changed,1);
}

extern "C" __global__ void gi_veto(int ng,const int* groups,const int* ids,
 const int* levels,const double* depth,const double* eta,const int* material,
 double dry,double span_limit,int finest,int evolving,int* raw,int* veto,
 int* mixed_out,int* material_out,double* span_out){
 int p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=ng)return;
 int wet=0,mat=material[groups[4*p]],mixedmat=0;
 double lo=1.7976931348623157e308,hi=-1.7976931348623157e308;
 for(int j=0;j<4;++j){int i=groups[4*p+j];
  if(depth[i]>dry){++wet;lo=fmin(lo,eta[i]);hi=fmax(hi,eta[i]);}
  mixedmat=mixedmat||(material[i]!=mat);}
 int mixed=wet>0&&wet<4;double span=wet?hi-lo:0.;
 mixedmat=mixedmat&&evolving;int hydraulic=mixed||span>span_limit;
 veto[p]=hydraulic||mixedmat;mixed_out[p]=mixed;material_out[p]=mixedmat;span_out[p]=span;
 if(veto[p])for(int j=0;j<4;++j){int node=ids[groups[4*p+j]];
  atomicMax(raw+node,hydraulic?finest:levels[node]);}
}

// The remaining indicators use cell-owned original directed CSR traversal.
// In particular, near-water becomes true as neighbours are encountered, not
// from an order-independent prepass. All geometric fits gather in leaf order.
extern "C" __global__ void gi_tree_bed(int nn,int ns,int nb,const int* indices,
 const double* bulk,const double* rock,const double* rows,double* bed,int* status){
 int node=blockIdx.x*blockDim.x+threadIdx.x;if(node>=nn)return;
 double sum=0.;for(int k=0;k<nb;++k)sum+=rows[node*ns+indices[k]]/bulk[k];
 bed[node]=rock[node]+sum;if(!isfinite(bed[node]))atomicOr(status,1);
}

extern "C" __global__ void gi_sediment_primitives(int n,int ns,int ng,
 const int* suspended,const int* active,const double* bulk,const double* q,
 double dry,double* concentration,double* fraction,double* volume,int* status){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 double total=0.;for(int g=0;g<ng;++g)total+=q[i*ns+active[g]]/bulk[g];
 volume[i]=total;if(!isfinite(total)){atomicOr(status,2);return;}
 double h=q[i*ns];
 for(int g=0;g<ng;++g){
  double c=h>dry?q[i*ns+suspended[g]]/h:0.;
  double f=0.;
  if(total>0.){
   double denominator=total*bulk[g];
   if(!isfinite(denominator)||denominator<=0.){atomicOr(status,2);return;}
   f=q[i*ns+active[g]]/denominator;
  }
  concentration[i*ng+g]=c;fraction[i*ng+g]=f;
  if(!isfinite(c)||!isfinite(f))atomicOr(status,2);
 }
}

extern "C" __global__ void gi_cells(int n,int ng,const int* ids,const int* offset,
 const int* faces,const int* signs,const int* owner,const int* neighbour,
 const int* bid,const int* material,const int* children,const double* xy,
 const double* bed,const double* tree_bed,const int* wet,const double* concentration,
 const double* fraction,const double* volume,const double* last_bed,const double* gradient,
 int walls,int sediment,int bathy,int finest,double ctol,double mixtol,double dztol,
 double bedtol,int* raw,int* cause,int* moving,int* near_water,int* status){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 int node=ids[i],near=wet[i],bits=0;
 for(int k=offset[i];k<offset[i+1];++k){
  int f=faces[k],j=signs[k]<0?owner[f]:neighbour[f];
  if(j<0){if(walls&&wet[i]&&bid[f]<0)bits|=1<<9;continue;}
  if(wet[j])near=1;
  if(sediment&&near){
   for(int g=0;g<ng;++g){
    double dc=concentration[i*ng+g]-concentration[j*ng+g];
    if(!isfinite(dc))atomicOr(status,4);
    if(fabs(dc)>ctol)bits|=1<<4;
   }
   double dz=bed[i]-bed[j];if(!isfinite(dz))atomicOr(status,4);
   if(material[i]&&fabs(dz)>dztol)bits|=1<<5;
   if(volume[i]>0.&&volume[j]>0.)for(int g=0;g<ng;++g){
    double df=fraction[i*ng+g]-fraction[j*ng+g];
    if(!isfinite(df))atomicOr(status,4);
    if(fabs(df)>mixtol)bits|=1<<6;
   }
  }
 }
 near_water[i]=near;
 if(bathy&&near){
  double change=0.,gx=gradient[2*i],gy=gradient[2*i+1];
  for(int k=0;k<4;++k){int child=children[4*node+k];if(child<0)break;
   double dx=xy[2*child]-xy[2*node],dy=xy[2*child+1]-xy[2*node+1];
   double residual=tree_bed[child]-bed[i]-(gx*dx+gy*dy);
   if(!isfinite(residual))atomicOr(status,4);change=fmax(change,fabs(residual));
  }
  for(int k=offset[i];k<offset[i+1];++k){
   int f=faces[k],j=signs[k]<0?owner[f]:neighbour[f];if(j<0)continue;
   int other=ids[j];double dx=xy[2*other]-xy[2*node],dy=xy[2*other+1]-xy[2*node+1];
   double residual=bed[j]-bed[i]-(gx*dx+gy*dy);
   if(!isfinite(residual))atomicOr(status,4);change=fmax(change,fabs(residual));
  }
  if(change>bedtol)bits|=1<<10;
 }
 if(sediment){double change=bed[i]-last_bed[node];
  if(!isfinite(change))atomicOr(status,4);if(fabs(change)>dztol)bits|=1<<11;}
 if(bits)raw[node]=max(raw[node],finest);
 cause[i]|=bits;
 if(bits&((1<<4)|(1<<5)|(1<<6)|(1<<11)))moving[i]=1;
}

extern "C" __global__ void gi_ancestor_detail(int nn,const int* offsets,
 const int* descendants,const int* ids,const double* area,const double* xy,
 const double* bed,const double* tree_bed,const int* near_water,const int* level,
 double tolerance,int* raw,int* raised,int* status){
 int node=blockIdx.x*blockDim.x+threadIdx.x;if(node>=nn)return;
 double xx=0.,xyv=0.,yy=0.,xz=0.,yz=0.;int near=0;
 for(int k=offsets[node];k<offsets[node+1];++k){
  int i=descendants[k],leaf=ids[i];double dx=xy[2*leaf]-xy[2*node];
  double dy=xy[2*leaf+1]-xy[2*node+1],dz=bed[i]-tree_bed[node],w=area[leaf];
  xx+=w*(dx*dx);xyv+=w*(dx*dy);yy+=w*(dy*dy);
  xz+=w*(dx*dz);yz+=w*(dy*dz);near=near||near_water[i];
 }
 if(!near)return;
 double product=xx*yy,det=product-xyv*xyv,gx=0.,gy=0.;
 if(!isfinite(xx)||!isfinite(xyv)||!isfinite(yy)||!isfinite(xz)||!isfinite(yz)||
    !isfinite(product)||!isfinite(det)){atomicOr(status,8);return;}
 if(det>1.e-12*product){gx=(yy*xz-xyv*yz)/det;gy=(xx*yz-xyv*xz)/det;}
 else if(xx+yy>0.){gx=xz/(xx+yy);gy=yz/(xx+yy);}
 if(!isfinite(gx)||!isfinite(gy)){atomicOr(status,8);return;}
 int wanted=raw[node];
 for(int k=offsets[node];k<offsets[node+1];++k){
  int i=descendants[k],leaf=ids[i];
  double dx=xy[2*leaf]-xy[2*node],dy=xy[2*leaf+1]-xy[2*node+1];
  double residual=bed[i]-tree_bed[node]-(gx*dx+gy*dy);
  if(!isfinite(residual))atomicOr(status,8);
  if(fabs(residual)>tolerance)wanted=max(wanted,level[node]+1);
 }
 raised[node]=wanted>raw[node];raw[node]=wanted;
}

extern "C" __global__ void gi_ancestor_causes(int n,const int* ids,const int* parent,
 const int* raised,int* cause){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 for(int node=ids[i];node>=0;node=parent[node])if(raised[node]){cause[i]|=1<<13;return;}
}
