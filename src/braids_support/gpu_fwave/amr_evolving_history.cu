// Isolated evolving-support AMR history proposal; no accepted-state writes.
// The history columns are u,H,h,L,toe_volume/area,collapse_volume/area.
// u is a width memory, not a stored cavity volume. All loops are independent
// cell/root threads; CSR row order makes reductions and integer ties stable.
#define EH_EPS 2.2204460492503131e-16
#define BRAIDS_AMR_EVOLVING_HISTORY_ABI 1

extern "C" __global__ void eh_validate(int no,int nn,int ne,int ns,
 const double* h,const int* count,const double* ao,const double* an,
 const int* co,const int* ci,const double* cw,const int* so,const int* si,
 const double* sf,const double* live,int have_exposure,const double* eo,
 const double* en,int* status){
 int i=blockIdx.x*blockDim.x+threadIdx.x,bad=0;
 int size=max(max(no,nn+1),max(ne,ns));if(i>=size)return;
 if(i<no){
  for(int j=0;j<6;++j)if(!isfinite(h[6*i+j])||h[6*i+j]<0.)bad|=1;
  if(count[i]<0||!isfinite(ao[i])||ao[i]<=0.||!isfinite(live[i])||live[i]<0.)bad|=1;
  if((h[6*i]>0.&&h[6*i+3]<=0.)||(h[6*i+3]>0.&&live[i]<=0.))bad|=1;
  if(have_exposure&&(!isfinite(eo[i])||eo[i]<0.))bad|=1;
 }
 if(i<nn){
  if(!isfinite(an[i])||an[i]<=0.)bad|=1;
  if(have_exposure&&(!isfinite(en[i])||en[i]<0.))bad|=1;
 }
 if(i<=nn){
  if(co[i]<0||co[i]>ne||so[i]<0||so[i]>ns)bad|=2;
  if(i<nn&&(co[i]>co[i+1]||so[i]>so[i+1]))bad|=2;
  if(i==0&&(co[i]!=0||so[i]!=0))bad|=2;
  if(i==nn&&(co[i]!=ne||so[i]!=ns))bad|=2;
 }
 if(i<ne&&(ci[i]<0||ci[i]>=no||!isfinite(cw[i])||cw[i]<=0.))bad|=2;
 if(i<ns&&(si[i]<0||si[i]>=no||!isfinite(sf[i])||sf[i]<=0.))bad|=2;
 status[i]=bad;
}

// Cell inverse CSR partitions each old area. It also assigns every actual
// event exactly once; a split does not manufacture a count for each child.
extern "C" __global__ void eh_old_partition(int no,const double* h,
 const int* count,const double* area,const int* off,const int* edges,
 const int* edge_new,const double* weight,const int* soff,const int* sedges,
 const double* fraction,long long* edge_count,int* status){
 int r=blockIdx.x*blockDim.x+threadIdx.x;if(r>=no)return;
 int bad=0,start=off[r],end=off[r+1];double total=0.,coverage=0.;
 for(int k=start;k<end;++k){int e=edges[k];total+=weight[e];
  if(k>start&&edge_new[e]==edge_new[edges[k-1]])bad|=4;
 }
 if(!isfinite(total)||total<=0.||
    fabs(total-area[r])>512.*EH_EPS*fmax(total,area[r]))bad|=4;
 for(int k=soff[r];k<soff[r+1];++k)coverage+=fraction[sedges[k]];
 bool memory=false;for(int j=0;j<4;++j)memory=memory||h[6*r+j]>0.;
 if(!isfinite(coverage)||coverage>1.+512.*EH_EPS||
    (memory&&fabs(coverage-1.)>512.*EH_EPS))bad|=8;
 if(bad){status[r]=bad;return;}
 long long assigned=0;
 for(int k=start;k<end;++k){int e=edges[k];
  double quota=(double)count[r]*(weight[e]/total);
  long long base=(long long)floor(quota);edge_count[e]=base;assigned+=base;
 }
 long long left=(long long)count[r]-assigned;
 if(left<0||left>=end-start){status[r]=16;return;}
 for(int k=start;k<end;++k){int e=edges[k],rank=0;
  double q=(double)count[r]*(weight[e]/total),rem=q-floor(q);
  for(int j=start;j<end;++j){int other=edges[j];
   double oq=(double)count[r]*(weight[other]/total),ore=oq-floor(oq);
   if(ore>rem||(ore==rem&&edge_new[other]<edge_new[e]))++rank;
  }
  if(rank<left)++edge_count[e];
 }
 status[r]=0;
}

extern "C" __global__ void eh_new_rows(int nn,const double* h,
 const double* old_area,const double* area,const int* co,const int* ci,
 const double* cw,const long long* edge_count,const int* so,const int* si,
 const double* sf,double critical,const int* prior,double* out,int* count,
 int* status){
 int n=blockIdx.x*blockDim.x+threadIdx.x;if(n>=nn)return;
 status[n]=0;if(prior[0]!=0)return;
 double a=0.,toe=0.,collapse=0.,length=0.,width=0.,height=0.,wet=0.;
 long long events=0;bool below=false,above=false;int bad=0;
 for(int k=co[n];k<co[n+1];++k){int r=ci[k];double w=cw[k];
  a+=w;toe+=w*h[6*r+4];collapse+=w*h[6*r+5];events+=edge_count[k];
 }
 if(!isfinite(a)||a<=0.||fabs(a-area[n])>512.*EH_EPS*fmax(a,area[n]))bad|=4;
 for(int k=so[n];k<so[n+1];++k){int r=si[k];double l=sf[k]*h[6*r+3];
  length+=l;width+=l*h[6*r];height+=l*h[6*r+1];wet+=l*h[6*r+2];
  if(l>0.){below=below||h[6*r]<critical;above=above||h[6*r]>=critical;}
 }
 if(below&&above)bad|=32;
 if(events<0||events>2147483647LL)bad|=16;
 if(!isfinite(toe)||!isfinite(collapse)||!isfinite(length)||
    !isfinite(width)||!isfinite(height)||!isfinite(wet))bad|=64;
 if(bad){status[n]=bad;return;}
 if(so[n+1]-so[n]==1&&sf[so[n]]==1.){
  int r=si[so[n]];for(int j=0;j<4;++j)out[6*n+j]=h[6*r+j];
 }else if(length>0.){
  out[6*n]=width/length;out[6*n+1]=height/length;
  out[6*n+2]=wet/length;out[6*n+3]=length;
 }
 if(co[n+1]-co[n]==1&&cw[co[n]]==area[n]&&
    old_area[ci[co[n]]]==area[n]){
  int r=ci[co[n]];out[6*n+4]=h[6*r+4];out[6*n+5]=h[6*r+5];
 }else{out[6*n+4]=toe/area[n];out[6*n+5]=collapse/area[n];}
 count[n]=(int)events;
 for(int j=0;j<6;++j)if(!isfinite(out[6*n+j])||out[6*n+j]<0.)bad|=64;
 status[n]=bad;
}
