// Isolated AMR bank-history inheritance on a selected native four-child split.
// One CUDA thread owns one newly exposed tree node. The source is the immutable
// accepted old-covering leaf, never another trial node; no child-parent race.

extern "C" __global__ void amr_prolong_bank_history(
    const double* accepted,const int* accepted_count,
    const int* selected,const int* cover,const int* old_leaf,
    int nodes,double* trial,int* trial_count,int* status) {
    int n=blockIdx.x*blockDim.x+threadIdx.x;
    if(n>=nodes)return;
    status[n]=0;
    if(!selected[n])return;
    int p=cover[n];
    if(p<0 || p>=nodes){status[n]=1;return;}
    if(p==n){
        if(!old_leaf[n])status[n]=2;
        return;
    }
    if(!old_leaf[p] || old_leaf[n]){status[n]=3;return;}
    if(accepted_count[p]<0){status[n]=4;return;}
    for(int field=0;field<6;++field){
        double value=accepted[p*6+field];
        if(!(value>=0.0) || !isfinite(value)){status[n]=5;return;}
    }
    for(int field=0;field<6;++field)
        trial[n*6+field]=accepted[p*6+field];
    trial_count[n]=accepted_count[p];
}
