// Geometry-only AMR support handoff. Accepted old mapping/history is read-only.
#define BRAIDS_AMR_EVOLVING_SUPPORT_ABI 1

extern "C" __global__ void ae_seed(int nf,const long long* offsets,
 const long long* old_face,const int* orientation,const int* old_owner,
 const int* old_bank,const int* old_toe,const int* owner,const int* neighbour,
 int* bank,int* toe,int* status){
 int f=blockIdx.x*blockDim.x+threadIdx.x;if(f>=nf)return;
 bank[f]=-1;toe[f]=-1;status[f]=0;
 for(long long k=offsets[f];k<offsets[f+1];++k){int a=old_face[k];
   if(old_bank[a]<0)continue;
   int side=(old_bank[a]==old_owner[a]?1:-1)*orientation[k];
   int b=side>0?owner[f]:neighbour[f],t=side>0?neighbour[f]:owner[f];
   if(b<0||t<0||(bank[f]>=0&&(bank[f]!=b||toe[f]!=t))){status[f]=1;continue;}
   bank[f]=b;toe[f]=t;
 }
}

extern "C" __global__ void ae_lengths(int nc,const int* offsets,
 const int* faces,const int* bank,const double* length,double* total){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;double value=0.;
 for(int k=offsets[c];k<offsets[c+1];++k){int f=faces[k];if(bank[f]==c)value+=length[f];}
 total[c]=value;
}

extern "C" __global__ void ae_edges(int ne,const long long* old_face,
 const long long* new_face,const double* fraction,const int* old_bank,
 const int* new_bank,const int* seed_bank,const double* length,const double* total,
 int* roots_old,int* roots_new,double* weight){
 int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=ne)return;
 int f=old_face[k],g=new_face[k],r=old_bank[f];
 roots_old[k]=-1;roots_new[k]=-1;weight[k]=0.;
 if(r<0||new_bank[g]<0||new_bank[g]!=seed_bank[g]||total[r]<=0.)return;
 roots_old[k]=r;roots_new[k]=new_bank[g];
 weight[k]=length[f]*fraction[k]/total[r];
}

extern "C" __global__ void ae_exposure(int nc,int ng,int nl,
 const int* offsets,const int* candidates,const int* toe,
 const double* layers,const double* bed,const double* depth,double dry,double* exposed){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
 exposed[c]=0.;int t=toe[c];if(t<0)return;double base=bed[t]+depth[t];
 for(int k=offsets[c];k<offsets[c+1];++k){int s=candidates[k];
   if(depth[s]>dry||bed[s]<=base)continue;int has=0;
   for(int l=0;l<nl;++l)for(int g=0;g<ng;++g)if(layers[(l*ng+g)*nc+s]>0.)has=1;
   if(has){exposed[c]=bed[s]-base;return;}
 }
}

extern "C" __global__ void ae_merge_guard(int nc,int nf,
 const int* offsets,const int* roots,const long long* face,
 const int* old_bank,const int* old_owner,const double* normal,
 const double* history,int* status){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;status[c]=0;
 for(int k=offsets[c];k<offsets[c+1];++k){int r=roots[k],f=face[k];
   if(history[r*6+3]<=0.)continue;double sign=old_bank[f]==old_owner[f]?1.:-1.;
   for(int j=offsets[c];j<k;++j){int s=roots[j],g=face[j];
     if(r==s||history[s*6+3]<=0.)continue;
     double other=old_bank[g]==old_owner[g]?1.:-1.;
     double dot=(normal[f]*normal[g]+normal[nf+f]*normal[nf+g])*sign*other;
     if(dot < -1e-12)status[c]=1;
   }
 }
}
