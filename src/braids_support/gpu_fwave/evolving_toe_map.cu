// Ordinary-cell channel and bank roles. No fractional front or solid area.
#define BRAIDS_EVOLVING_TOE_ABI 2
#define TM_HUGE 1.7976931348623157e308

extern "C" __global__ void tm_initial(int nc,int ng,int nl,const int* material,
 const double* bed,const double* depth,const double* layers,double dry,
 int* channel,int* inventory,int* status){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
 int has=0,bad=!isfinite(bed[c])||!isfinite(depth[c])||depth[c]<0.;
 for(int l=0;l<nl;++l)for(int g=0;g<ng;++g){double x=layers[(l*ng+g)*nc+c];
   if(!isfinite(x)||x<0.)bad=1;if(x>0.)has=1;}
 inventory[c]=has;channel[c]=!material[c]&&depth[c]>dry;status[c]=bad;
}

extern "C" __global__ void tm_reach(int nc,const int* material,
 const int* offset,const int* faces,const int* owner,const int* neighbour,
 const double* bed,const double* depth,double dry,const double* tolerance,
 const int* old,int* next,int* changed,int* status){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
 int value=old[c];status[c]=0;
 if(!value&&depth[c]>dry){
   for(int k=offset[c];k<offset[c+1];++k){int f=faces[k];
     int n=owner[f]==c?neighbour[f]:owner[f];if(n<0||!old[n])continue;
     double gap=bed[c]-bed[n];
     if(!isfinite(gap)){status[c]=1;continue;}
     if(!material[c]||gap<=tolerance[0]){value=1;break;}
   }
 }
 next[c]=value;if(value!=old[c])atomicExch(changed,1);
}

extern "C" __global__ void tm_bank(int nc,const int* material,
 const int* channel,const int* inventory,int* bank){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c<nc)bank[c]=material[c]&&!channel[c]&&inventory[c];
}

extern "C" __global__ void tm_faces(int nf,const int* owner,const int* neighbour,
 const int* bank,const int* channel,const double* bed,
 int* flag,int* donor,int* toe,int* status){
 int f=blockIdx.x*blockDim.x+threadIdx.x;if(f>=nf)return;
 flag[f]=0;donor[f]=-1;toe[f]=-1;status[f]=0;
 int c=owner[f],n=neighbour[f];if(n<0)return;
 int b=bank[c]&&channel[n]?c:(bank[n]&&channel[c]?n:-1);
 if(b<0)return;int t=b==c?n:c;double gap=bed[b]-bed[t];
 if(!isfinite(gap)){status[f]=1;return;}
 if(gap<=0.)return;flag[f]=1;donor[f]=b;toe[f]=t;
}

__device__ int tm_prefer(int d,int t,int f,int old_d,int old_t,int old_f,
 const double* bed,const double* length){
 if(old_d<0)return 1;
 if(bed[t]!=bed[old_t])return bed[t]<bed[old_t];
 if(length[f]!=length[old_f])return length[f]>length[old_f];
 if(t!=old_t)return t<old_t;if(d!=old_d)return d<old_d;return f<old_f;
}

extern "C" __global__ void tm_seed(int nc,const int* offset,const int* faces,
 const int* face_bank,const int* face_toe,const double* edge,const double* bed,
 const double* length,int* donor,int* toe,int* face,double* distance,
 int* is_donor,int* is_toe,int* ambiguous){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
 int d=-1,t=-1,fbest=-1,amb=0,recipient=0;double best=TM_HUGE;
 for(int k=offset[c];k<offset[c+1];++k){int f=faces[k];
   if(face_toe[f]==c)recipient=1;if(face_bank[f]!=c)continue;
   int target=face_toe[f];double dist=edge[f];
   if(dist<best){d=c;t=target;fbest=f;best=dist;amb=0;}
   else if(dist==best){if(t!=target)amb=1;
     if(tm_prefer(c,target,f,d,t,fbest,bed,length)){d=c;t=target;fbest=f;}}
 }
 donor[c]=d;toe[c]=t;face[c]=fbest;distance[c]=best;
 is_donor[c]=d>=0;is_toe[c]=recipient;ambiguous[c]=amb;
}

extern "C" __global__ void tm_relax(int nc,const int* bank,const int* roots,
 const int* offset,const int* faces,const int* owner,const int* neighbour,
 const double* bed,const double* length,const double* edge,const double* tol,
 const int* old_d,const int* old_t,const int* old_f,const double* old_dist,const int* old_amb,
 int* next_d,int* next_t,int* next_f,double* next_dist,int* next_amb,int* changed,int* status){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
 int d=old_d[c],t=old_t[c],fbest=old_f[c],amb=old_amb[c];double best=old_dist[c];
 status[c]=0;
 if(bank[c]&&!roots[c])for(int k=offset[c];k<offset[c+1];++k){int f=faces[k];
   int n=owner[f]==c?neighbour[f]:owner[f];
   if(n<0||!bank[n]||old_d[n]<0)continue;
   double gap=bed[n]-bed[c];if(!isfinite(gap)){status[c]=1;continue;}
   if(gap>tol[0])continue;
   double trial=old_dist[n]+edge[f];if(!isfinite(trial)){status[c]=1;continue;}
   if(trial<best){best=trial;d=old_d[n];t=old_t[n];fbest=old_f[n];amb=old_amb[n];}
   else if(trial==best){if(old_d[n]!=d||old_t[n]!=t||old_amb[n])amb=1;
     if(tm_prefer(old_d[n],old_t[n],old_f[n],d,t,fbest,bed,length)){
       d=old_d[n];t=old_t[n];fbest=old_f[n];}}
 }
 next_d[c]=d;next_t[c]=t;next_f[c]=fbest;next_dist[c]=best;next_amb[c]=amb;
 if(d!=old_d[c]||t!=old_t[c]||fbest!=old_f[c]||best!=old_dist[c]||amb!=old_amb[c])
   atomicExch(changed,1);
}

// Dormant support is NOT an active sediment-transfer face. A previously
// represented material interface remains meaningful while water temporarily
// disconnects/dries, provided its bank column and positive relief still exist.
extern "C" __global__ void tm_support_faces(int nf,const int* owner,
 const int* neighbour,const int* bank,const double* bed,
 const int* active_bank,const int* active_toe,const int* active_donor,const int* initialized,
 const int* old_bank,const int* old_toe,int* support_bank,int* support_toe,int* status){
 int f=blockIdx.x*blockDim.x+threadIdx.x;if(f>=nf)return;
 status[f]=0;support_bank[f]=active_bank[f];support_toe[f]=active_toe[f];
 if(active_bank[f]>=0||!initialized[0])return;
 int b=old_bank[f],t=old_toe[f];if(b<0||t<0)return;
 if(!((owner[f]==b&&neighbour[f]==t)||(owner[f]==t&&neighbour[f]==b))){status[f]=1;return;}
 if(!bank[b])return;
 // A former material-channel cell may become bank[] merely because it dries.
 // Preserve that dormancy. Actual ownership by a CURRENT active bank root,
 // however, proves a role return and must not leave a chained old support root.
 if(active_donor[t]>=0)return;
 double gap=bed[b]-bed[t];if(!isfinite(gap)){status[f]=1;return;}
 if(gap>0.){support_bank[f]=b;support_toe[f]=t;}
}

// Active roots must use the active preferred toe. Dormant support cannot cause
// inherited undercut from a different old toe to be used by current physics.
extern "C" __global__ void tm_support_seed(int nc,const int* active_root,
 const int* active_d,const int* active_t,const int* active_f,
 const double* active_distance,const int* active_amb,
 int* donor,int* toe,int* face,double* distance,int* ambiguous,int* root){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc||!active_root[c])return;
 donor[c]=active_d[c];toe[c]=active_t[c];face[c]=active_f[c];
 distance[c]=active_distance[c];ambiguous[c]=active_amb[c];root[c]=1;
}

extern "C" __global__ void tm_support_live(int nc,const int* donor,const int* toe,
 const int* inventory,const double* bed,const double* depth,double dry,
 int* live,int* status){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
 status[c]=0;int d=donor[c],t=toe[c];if(d<0)return;
 if(d>=nc||t<0||t>=nc){status[c]=1;return;}
 double base=bed[t]+depth[t];if(!isfinite(base)){status[c]=1;return;}
 if(inventory[c]&&depth[c]<=dry&&bed[c]>base)atomicExch(live+d,1);
}

// Each membership or interface change invalidates the geometric history of
// BOTH affected roots. This includes non-root candidates far from a root cell.
// Boolean atomic flags are order independent; no material/state is scattered.
extern "C" __global__ void tm_changed_support(int nc,int nf,const int* initialized,
 const int* old_d,const int* old_t,const int* old_f,
 const int* next_d,const int* next_t,const int* next_f,
 const int* old_bank,const int* old_toe,const int* next_bank,const int* next_toe,
 const int* live,int* changed){
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<nc){
   if(!live[i])atomicExch(changed+i,1);
   int d=initialized[0]?old_d[i]:-1;
   if(!initialized[0]||d!=next_d[i]||old_t[i]!=next_t[i]||old_f[i]!=next_f[i]){
     if(d>=0&&d<nc)atomicExch(changed+d,1);
     if(next_d[i]>=0&&next_d[i]<nc)atomicExch(changed+next_d[i],1);}
 }
 if(i<nf){int b=initialized[0]?old_bank[i]:-1;
   if(!initialized[0]||b!=next_bank[i]||old_toe[i]!=next_toe[i]){
     if(b>=0&&b<nc)atomicExch(changed+b,1);
     if(next_bank[i]>=0&&next_bank[i]<nc)atomicExch(changed+next_bank[i],1);}
 }
}

extern "C" __global__ void tm_retire(int nc,const int* changed,
 double* u,double* height,double* wet,double* length){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
 if(changed[c]){
   u[c]=0.;height[c]=0.;wet[c]=0.;length[c]=0.;}
}
