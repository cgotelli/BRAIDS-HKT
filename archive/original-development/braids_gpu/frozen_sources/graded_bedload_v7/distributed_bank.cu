// Accepted-flux distributed slope-bank transport (native bank mode 3).
// All kernels operate on private trial fields. Persistent history is committed
// only by the enclosing coupled transaction after every status check succeeds.
#define DB_EPS 2.2204460492503131e-16

extern "C" __global__ void db_capture(
    const double* flux, const double* length, long long nf, long long count,
    double dt, double weight, double* face_mass, int* failure) {
  const long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=count)return;
  const long long f=i%nf;
  const double next=face_mass[i]+weight*dt*length[f]*flux[i];
  if(!isfinite(flux[i])||!isfinite(length[f])||length[f]<=0.||
     !isfinite(dt)||dt<=0.||!isfinite(weight)||weight<0.||!isfinite(next)) {
    atomicExch(failure,1);return;
  }
  face_mass[i]=next;
}

extern "C" __global__ void db_reconstruct(
    int nc,int nf,int ng,const int* owner,const int* neighbour,
    const int* offset,const int* faces,const double* normal,
    const double* length,const double* mass,double dt,
    double* q,int* rank_deficient,int* status) {
  const int c=blockIdx.x*blockDim.x+threadIdx.x;
  if(c>=nc)return;
  double mxx=0.,mxy=0.,myy=0.;int bad=!isfinite(dt)||dt<=0.;
  for(int k=offset[c];k<offset[c+1];++k){
    const int f=faces[k];const double nx=normal[f],ny=normal[nf+f],l=length[f];
    if(f<0||f>=nf||!isfinite(nx)||!isfinite(ny)||!isfinite(l)||l<=0.){
      bad=1;continue;}
    mxx+=l*nx*nx;mxy+=l*nx*ny;myy+=l*ny*ny;
  }
  const double determinant=mxx*myy-mxy*mxy;
  const double scale=fmax(mxx*myy,1.);
  if(!isfinite(mxx)||!isfinite(mxy)||!isfinite(myy)||
     !isfinite(determinant)||!isfinite(scale))bad=1;
  const int deficient=bad||determinant<=1.e-14*scale;
  rank_deficient[c]=deficient;
  for(int g=0;g<ng;++g){
    double bx=0.,by=0.;
    for(int k=offset[c];k<offset[c+1];++k){
      const int f=faces[k];
      if(f<0||f>=nf)continue;
      const double r=mass[g*nf+f]/dt;
      bx+=normal[f]*r;by+=normal[nf+f]*r;
    }
    double x=0.,y=0.;
    if(!deficient){
      x=(myy*bx-mxy*by)/determinant;
      y=(mxx*by-mxy*bx)/determinant;
    }
    if(!isfinite(bx)||!isfinite(by)||!isfinite(x)||!isfinite(y))bad=1;
    q[(0*ng+g)*nc+c]=bad?0.:x;
    q[(1*ng+g)*nc+c]=bad?0.:y;
  }
  status[c]=bad;
}

extern "C" __global__ void db_propose(
    int nc,int nf,int ng,const int* owner,const int* neighbour,
    const double* xy,const double* normal,const double* length,
    const double* bed,const int* erodible,const double* q,
    const double* active,double alpha,double critical_slope,double dt,
    double* proposal,int* eligible,int* donor,int* recipient,
    double* slope,double* tangent_transport,int* status) {
  const int f=blockIdx.x*blockDim.x+threadIdx.x;
  if(f>=nf)return;
  eligible[f]=0;donor[f]=-1;recipient[f]=-1;
  slope[f]=0.;tangent_transport[f]=0.;status[f]=0;
  for(int g=0;g<ng;++g)proposal[g*nf+f]=0.;
  const int o=owner[f],n=neighbour[f];
  if(o<0||o>=nc||n<-1||n>=nc||n==o){status[f]=1;return;}
  if(n<0)return;
  int high=-1,low=-1,orientation=0;
  if(bed[o]>bed[n]){high=o;low=n;orientation=1;}
  else if(bed[n]>bed[o]){high=n;low=o;orientation=-1;}
  else return;
  const double dx=xy[high]-xy[low],dy=xy[nc+high]-xy[nc+low];
  const double distance=sqrt(dx*dx+dy*dy);
  const double s=(bed[high]-bed[low])/distance;
  if(!isfinite(distance)||distance<=0.||!isfinite(s)||
     !isfinite(length[f])||length[f]<=0.){status[f]=1;return;}
  slope[f]=s;
  if(!erodible[high]||s<=critical_slope)return;
  eligible[f]=1;donor[f]=high;recipient[f]=low;
  double total_x=0.,total_y=0.,active_total=0.;
  for(int g=0;g<ng;++g){
    total_x+=q[(0*ng+g)*nc+low];
    total_y+=q[(1*ng+g)*nc+low];
    active_total+=active[g*nc+high];
  }
  const double nx=normal[f],ny=normal[nf+f];
  const double normal_size=sqrt(nx*nx+ny*ny);
  if(!isfinite(normal_size)||normal_size<=0.||!isfinite(active_total)||
     active_total<0.||!isfinite(total_x)||!isfinite(total_y)){
    status[f]=1;return;
  }
  const double transport=fabs((-ny*total_x+nx*total_y)/normal_size);
  tangent_transport[f]=transport;
  if(active_total<=0.)return;
  const double amount=alpha*dt*length[f]*transport;
  if(!isfinite(amount)||amount<0.){status[f]=1;return;}
  for(int g=0;g<ng;++g){
    const double value=orientation*amount*active[g*nc+high]/active_total;
    if(!isfinite(value)){status[f]=1;return;}
    proposal[g*nf+f]=value;
  }
}

extern "C" __global__ void db_history(
    int nc,int nf,int ng,const double* area,const double* length,
    const double* bulk,const int* donor_offset,const int* donor_faces,
    const int* eligible,const int* donor,const int* recipient,
    const double* bed,const double* depth,const double* transfer,
    double dry,double* undercut,double* reference_height,
    double* toe_height,double* reference_length,double* toe_cumulative,
    int* status) {
  const int c=blockIdx.x*blockDim.x+threadIdx.x;
  if(c>=nc)return;
  int exposed=0,bad=0;
  double request=0.,weight=0.,down_length=0.,down_height=0.;
  double toe_max=toe_height[c],cumulative=toe_cumulative[c];
  const double tolerance=512.*DB_EPS*fmax(fabs(bed[c]),1.);
  for(int k=donor_offset[c];k<donor_offset[c+1];++k){
    const int f=donor_faces[k];
    if(!eligible[f]||donor[f]!=c)continue;
    const int r=recipient[f];
    if(r<0||r>=nc){bad=1;continue;}
    const double height=fmax(bed[c]-bed[r],0.);
    const double wet=fmin(fmax(depth[r],0.),height);
    const double above=fmax(height-wet,0.);
    if(above>tolerance)exposed=1;
    double volume=0.;
    for(int g=0;g<ng;++g)volume+=fabs(transfer[g*nf+f])/bulk[g];
    if(!isfinite(volume)||volume<0.){bad=1;continue;}
    if(volume<=0.)continue;
    cumulative+=volume/area[c];
    if(above<=tolerance||wet<=dry)continue;
    request+=volume;
    weight+=length[f]*wet;
    down_length+=length[f];
    down_height=fmax(down_height,height);
    toe_max=fmax(toe_max,wet);
  }
  if(!exposed){undercut[c]=0.;reference_height[c]=0.;
    toe_height[c]=0.;reference_length[c]=0.;}
  else if(request>0.){
    if(weight<=0.||!isfinite(weight))bad=1;
    else{
      undercut[c]+=request/weight;
      reference_height[c]=fmax(reference_height[c],down_height);
      reference_length[c]=fmax(reference_length[c],down_length);
      toe_height[c]=toe_max;
    }
  }
  toe_cumulative[c]=cumulative;
  if(!isfinite(undercut[c])||!isfinite(reference_height[c])||
     !isfinite(reference_length[c])||!isfinite(toe_height[c])||
     !isfinite(toe_cumulative[c])||undercut[c]<0.||
     reference_height[c]<0.||reference_length[c]<0.||
     toe_height[c]<0.||toe_cumulative[c]<0.)bad=1;
  status[c]=bad;
}

extern "C" __global__ void db_bed_after_transfer(
    int nc,int nf,int ng,const double* bed,const double* bulk,
    const double* prior_active,const double* next_active,
    double* next_bed,int* status) {
  const int c=blockIdx.x*blockDim.x+threadIdx.x;
  if(c>=nc)return;
  double value=bed[c];
  for(int g=0;g<ng;++g)
    value+=(next_active[g*nc+c]-prior_active[g*nc+c])/bulk[g];
  next_bed[c]=value;
  status[c]=!isfinite(value);
}

extern "C" __global__ void db_plan_collapse(
    int nc,const double* length,const int* donor_offset,
    const int* donor_faces,const int* eligible,const int* donor,
    const int* recipient,const double* tangent_transport,
    const double* undercut,double critical,int* candidate,
    int* pending,int* counters,int* status) {
  const int c=blockIdx.x*blockDim.x+threadIdx.x;
  if(c>=nc)return;
  candidate[c]=-1;pending[c]=0;status[c]=0;
  if(!isfinite(undercut[c])||undercut[c]<0.){status[c]=1;return;}
  if(undercut[c]<critical)return;
  double best=-1.;int best_face=-1;
  for(int k=donor_offset[c];k<donor_offset[c+1];++k){
    const int f=donor_faces[k];
    if(!eligible[f]||donor[f]!=c)continue;
    const double score=tangent_transport[f]*length[f];
    if(!isfinite(score)){status[c]=1;return;}
    if(score>best){best=score;best_face=f;}
  }
  if(best_face<0)return;
  const int r=recipient[best_face];
  if(r<0||r>=nc||r==c){status[c]=1;return;}
  candidate[c]=r;pending[c]=1;
  atomicAdd(counters,1);
}

extern "C" __global__ void db_wave_mark(
    int nc,const int* source_offset,const int* sources,
    const int* candidate,const int* pending,int* ready,
    int* counters,int* status) {
  const int c=blockIdx.x*blockDim.x+threadIdx.x;
  if(c>=nc)return;
  ready[c]=0;
  if(!pending[c])return;
  const int r=candidate[c];
  if(r<0||r>=nc){status[c]=1;return;}
  // The ascending native loop imposes an edge from an earlier pending donor
  // whenever the two collapse operations touch the same cell. The CSR lists
  // include all geometric neighbours, and candidate targets are neighbours.
  for(int k=source_offset[r];k<source_offset[r+1];++k){
    const int d=sources[k];
    if(d<c&&pending[d]&&candidate[d]==r)return; // shared recipient
  }
  for(int k=source_offset[c];k<source_offset[c+1];++k){
    const int d=sources[k];
    if(d<c&&pending[d]&&candidate[d]==c)return; // earlier donor into c
  }
  if(r<c&&pending[r])return; // c deposits into an earlier donor
  ready[c]=1;atomicAdd(counters+1,1);
}

extern "C" __global__ void db_collapse_wave(
    int nc,int nf,int ng,int nl,const double* area,const double* bulk,
    const int* candidate,const int* ready,int* pending,int* counters,
    double* bed,double* layers,
    double critical,double* undercut,double* reference_height,
    double* toe_height,double* reference_length,
    double* collapse_cumulative,int* collapse_count,
    double* removed,double* removed_volume,int* target,int* status) {
  const int c=blockIdx.x*blockDim.x+threadIdx.x;
  if(c>=nc)return;
  if(!ready[c])return;
  pending[c]=0;atomicSub(counters,1);
  const int r=candidate[c];
  if(r<0||r>=nc){status[c]=1;return;}
  const double height=fmax(bed[c]-bed[r],0.);
  const double above=fmax(height-toe_height[c],0.);
  if(above<=0.||reference_length[c]<=0.)return;
  const double event_real=undercut[c]/critical;
  if(!isfinite(event_real)||event_real>2147483647.){status[c]=1;return;}
  const int events=(int)event_real;
  double requested=events*critical*reference_length[c]*above;
  requested=fmin(requested,height/(1./area[c]+1./area[r]));
  if(!isfinite(requested)||requested<0.){status[c]=1;return;}
  double remaining=requested,actual=0.;
  for(int l=0;l<nl;++l){
    if(remaining<=64.*DB_EPS*fmax(requested,1.))break;
    double layer_volume=0.;
    for(int g=0;g<ng;++g)
      layer_volume+=layers[(l*ng+g)*nc+c]*area[c]/bulk[g];
    if(layer_volume<=0.)continue;
    const double take=fmin(remaining,layer_volume);
    const double factor=fmin(1.,take/layer_volume);
    for(int g=0;g<ng;++g){
      const int i=(l*ng+g)*nc+c;
      const double amount=layers[i]*factor;
      layers[i]-=amount;
      removed[g*nc+c]+=amount*area[c];
      if(layers[i]<0.&&fabs(layers[i])*area[c]<=128.*DB_EPS)layers[i]=0.;
    }
    actual+=take;remaining=fmax(0.,remaining-take);
  }
  if(actual<=0.)return;
  target[c]=r;
  removed_volume[c]=actual;
  bed[c]-=actual/area[c];
  collapse_cumulative[c]+=actual/area[c];
  if(collapse_count[c]>2147483647-events){status[c]=1;return;}
  collapse_count[c]+=events;
  undercut[c]=fmax(0.,undercut[c]-events*critical);
  if(undercut[c]<=64.*DB_EPS*critical){
    undercut[c]=0.;reference_height[c]=0.;toe_height[c]=0.;
    reference_length[c]=0.;
  }
  if(!isfinite(collapse_cumulative[c])||!isfinite(undercut[c])||
     !isfinite(bed[c])||collapse_cumulative[c]<0.||undercut[c]<0.)status[c]=1;
}

extern "C" __global__ void db_deposit_wave(
    int nc,int nf,int ng,const double* area,const int* source_offset,
    const int* sources,const int* ready,const int* target,
    const double* removed,const double* removed_volume,
    double* bed,double* layers,double* gain,int* status) {
  const int c=blockIdx.x*blockDim.x+threadIdx.x;
  if(c>=nc)return;
  int bad=0,incoming=0;
  double volume=0.;
  for(int k=source_offset[c];k<source_offset[c+1];++k){
    const int source=sources[k];
    if(ready[source]&&target[source]==c){
      ++incoming;volume+=removed_volume[source];
    }
  }
  if(incoming>1||incoming>0&&ready[c])bad=1;
  for(int g=0;g<ng;++g){
    double received=0.;
    for(int k=source_offset[c];k<source_offset[c+1];++k){
      const int source=sources[k];
      if(ready[source]&&target[source]==c)received+=removed[g*nc+source];
    }
    const int i=g*nc+c;
    const double value=layers[i]+received/area[c];
    gain[i]+=received;layers[i]=value;
    if(!isfinite(received)||received<0.||!isfinite(value)||value<0.)bad=1;
  }
  bed[c]+=volume/area[c];
  if(!isfinite(bed[c])||!isfinite(volume)||volume<0.)bad=1;
  if(bad)status[c]=1;
}

extern "C" __global__ void db_budget_partial(
    int nc,int ng,int chunks,const double* removed,const double* gain,
    double* partial) {
  const int g=blockIdx.y,chunk=blockIdx.x,lane=threadIdx.x;
  const int c=chunk*blockDim.x+lane;
  __shared__ double debit[256],credit[256];
  debit[lane]=c<nc?removed[g*nc+c]:0.;
  credit[lane]=c<nc?gain[g*nc+c]:0.;
  __syncthreads();
  for(int stride=blockDim.x/2;stride>0;stride/=2){
    if(lane<stride){debit[lane]+=debit[lane+stride];
                     credit[lane]+=credit[lane+stride];}
    __syncthreads();
  }
  if(lane==0){
    const int i=(g*chunks+chunk)*2;
    partial[i]=debit[0];partial[i+1]=credit[0];
  }
}

extern "C" __global__ void db_budget(
    int ng,int chunks,const double* partial,double* budget,int* status) {
  const int g=blockIdx.x*blockDim.x+threadIdx.x;
  if(g>=ng)return;
  double debit=0.,credit=0.;
  for(int chunk=0;chunk<chunks;++chunk){
    const int i=(g*chunks+chunk)*2;
    debit+=partial[i];credit+=partial[i+1];
  }
  budget[2*g]=debit;budget[2*g+1]=credit;
  status[g]=!isfinite(debit)||!isfinite(credit)||
    fabs(debit-credit)>512.*DB_EPS*fmax(fmax(fabs(debit),fabs(credit)),1.);
}

extern "C" __global__ void db_validate(
    int nc,int ng,int nl,const double* layers,const double* undercut,
    const double* reference_height,const double* toe_height,
    const double* reference_length,const double* toe_cumulative,
    const double* collapse_cumulative,const int* collapse_count,
    int* status) {
  const int c=blockIdx.x*blockDim.x+threadIdx.x;
  if(c>=nc)return;
  int bad=collapse_count[c]<0;
  const double* histories[6]={undercut,reference_height,toe_height,
    reference_length,toe_cumulative,collapse_cumulative};
  for(int i=0;i<6;++i)if(!isfinite(histories[i][c])||histories[i][c]<0.)bad=1;
  for(int l=0;l<nl;++l)for(int g=0;g<ng;++g){
    const double x=layers[(l*ng+g)*nc+c];
    if(!isfinite(x)||x<0.)bad=1;
  }
  status[c]=bad;
}
