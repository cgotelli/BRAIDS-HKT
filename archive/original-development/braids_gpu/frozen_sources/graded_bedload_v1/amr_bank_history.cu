// Isolated AMR bank-failure history restriction. One parent per GPU thread.
// Exact native child order; no topology, regrid exposure or level stepping.

extern "C" __global__ void amr_restrict_bank_history(
    const double* child,const int* child_count,const double* area,
    int parents,double* parent,int* parent_count,int* status) {
    int p=blockIdx.x*blockDim.x+threadIdx.x;
    if(p>=parents)return;
    double maximum0=0.0,maximum1=0.0,maximum2=0.0;
    double sum3=0.0,sum4=0.0,sum5=0.0,total=0.0;
    int count=0;
    for(int k=0;k<4;++k){
        int row=(p*4+k)*6;
        double weight=area[p*4+k];
        if(!(weight>0.0) || !isfinite(weight) || child_count[p*4+k]<0){
            status[p]=1;return;
        }
        for(int j=0;j<6;++j){
            double value=child[row+j];
            if(!(value>=0.0) || !isfinite(value)){
                status[p]=2;return;
            }
        }
        total+=weight;
        maximum0=fmax(maximum0,child[row]);
        maximum1=fmax(maximum1,child[row+1]);
        maximum2=fmax(maximum2,child[row+2]);
        sum3+=weight*child[row+3];
        sum4+=weight*child[row+4];
        sum5+=weight*child[row+5];
        count=max(count,child_count[p*4+k]);
    }
    if(!(total>0.0) || !isfinite(total)){status[p]=3;return;}
    int out=p*6;
    parent[out]=maximum0;parent[out+1]=maximum1;parent[out+2]=maximum2;
    parent[out+3]=sum3/total;parent[out+4]=sum4/total;
    parent[out+5]=sum5/total;parent_count[p]=count;
    status[p]=0;
}
