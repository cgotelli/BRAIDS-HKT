// Private resident water + active-bed reflux repair, no phase exchange.
#define BRAIDS_BANK_REFLUX_REPAIR_ABI 2

__device__ void br_error(int c,int bit,int* cell_status,int* status){
    atomicOr(cell_status+c,bit);atomicOr(status,bit);
}
__device__ bool br_contains(const int* ids,int count,int value){
    for(int k=0;k<count;++k)if(ids[k]==value)return true;
    return false;
}
extern "C" __global__ void br_validate_input(
    const double* state,const double* area,const int* levels,const int* active,
    const int* suspended,int nc,int ns,int ng,int nsp,int level,int root,int secondary,
    int* cell_status,int* status){
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
    for(int j=0;j<ns;++j){
        double v=state[(long long)c*ns+j];
        if(!isfinite(v)||!isfinite(v*area[c])){br_error(c,1,cell_status,status);continue;}
        if(br_contains(suspended,nsp,j)&&v!=0.0)br_error(c,2,cell_status,status);
        if(v>=0.0||j==1||j==2||j==secondary)continue;
        if(j==0){if(!root&&levels[c]!=level)br_error(c,8,cell_status,status);}
        else if(!root&&br_contains(active,ng,j)){
            if(levels[c]!=level)br_error(c,8,cell_status,status);
        }else br_error(c,4,cell_status,status);
    }
}

__device__ void br_water_one(double* state,const double* area,const int* offset,
    const int* donors,int c,int nc,int ns,int ng,int secondary,bool root,
    int* cell_status,int* status,double* diagnostics,double* details){
    long long row=(long long)c*ns;
    if(state[row]>=0.0)return;
    double deficit=-state[row]*area[c],total=0.0;
    long long diag=((long long)c*(ng+1))*4,detail=((long long)c*(ng+1))*3;
    details[detail]=deficit;
    // A representable negative stock whose extensive inventory underflows is
    // rejected, not silently clipped to zero.
    if(!(deficit>0.0)||!isfinite(deficit)){br_error(c,64,cell_status,status);return;}
    for(int k=offset[c];k<offset[c+1];++k){
        int n=donors[k];total+=fmax(0.0,state[(long long)n*ns]*area[n]);
    }
    bool global=root&&total<deficit;
    if(global){total=0.0;for(int n=0;n<nc;++n)total+=fmax(0.0,state[(long long)n*ns]*area[n]);}
    details[detail+1]=total;
    if(!isfinite(total)){br_error(c,64,cell_status,status);return;}
    if(total<deficit){details[detail+2]=deficit-total;br_error(c,16,cell_status,status);return;}
    double px=state[row+1]*area[c],py=state[row+2]*area[c],remaining=deficit;
    double sec=secondary>=0?state[row+secondary]*area[c]:0.0;
    if(!isfinite(px)||!isfinite(py)||!isfinite(sec)){br_error(c,64,cell_status,status);return;}
    int begin=global?0:offset[c],end=global?nc:offset[c+1];
    for(int k=begin;k<end;++k){
        int n=global?k:donors[k];long long nr=(long long)n*ns;
        double available=fmax(0.0,state[nr]*area[n]);if(available<=0.0)continue;
        double take=fmin(available,remaining),fraction=take/deficit;
        // Difference in extensive space gives exactly zero on full depletion;
        // it never clips a negative inventory.
        state[nr]=(available-take)/area[n];
        state[nr+1]+=fraction*px/area[n];state[nr+2]+=fraction*py/area[n];
        if(secondary>=0)state[nr+secondary]+=fraction*sec/area[n];
        remaining-=take;
        if(!isfinite(state[nr])||!isfinite(state[nr+1])||!isfinite(state[nr+2])||
           (secondary>=0&&!isfinite(state[nr+secondary]))){
            br_error(c,64,cell_status,status);return;
        }
        if(remaining==0.0)break;
    }
    details[detail+2]=remaining;
    if(remaining!=0.0){br_error(c,128,cell_status,status);return;}
    state[row]=0.0;state[row+1]=0.0;state[row+2]=0.0;
    if(secondary>=0)state[row+secondary]=0.0;
    diagnostics[diag]=1.0;diagnostics[diag+1]=deficit;
    diagnostics[diag+2]=px;diagnostics[diag+3]=py;
}

extern "C" __global__ void br_water_wave(double* state,const double* area,
    const int* offset,const int* donors,const int* cells,const int* active,
    int nc,int ns,int ng,int count,int secondary,const int* gate,int* cell_status,int* status,
    double* diagnostics,double* details){
    int p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=count||*gate)return;
    br_water_one(state,area,offset,donors,cells[p],nc,ns,ng,secondary,false,
                 cell_status,status,diagnostics,details);
}

extern "C" __global__ void br_root_water(double* state,const double* area,
    const int* offset,const int* donors,int nc,int ns,int ng,int secondary,const int* gate,
    int* cell_status,int* status,double* diagnostics,double* details){
    if(blockIdx.x||threadIdx.x||*gate)return;
    // All-domain fallback couples every request; preserve native ascending-cell
    // order here rather than racing global withdrawals or copying state to CPU.
    for(int c=0;c<nc;++c){
        br_water_one(state,area,offset,donors,c,nc,ns,ng,secondary,true,
                     cell_status,status,diagnostics,details);
        if(*status)return;
    }
}

extern "C" __global__ void br_bed_wave(double* state,const double* area,
    const int* offset,const int* donors,const int* cells,const int* active,
    int nc,int ns,int ng,int count,const int* gate,int* cell_status,int* status,
    double* diagnostics,double* details){
    int p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=count*ng||*gate)return;
    int c=cells[p/ng],g=p%ng,j=active[g];long long row=(long long)c*ns+j;
    if(state[row]>=0.0)return;
    double deficit=-state[row]*area[c],total=0.0;
    long long diag=((long long)c*(ng+1)+g+1)*4,detail=((long long)c*(ng+1)+g+1)*3;
    details[detail]=deficit;
    if(!(deficit>0.0)||!isfinite(deficit)){br_error(c,64,cell_status,status);return;}
    for(int k=offset[c];k<offset[c+1];++k){
        int n=donors[k];total+=fmax(0.0,state[(long long)n*ns+j]*area[n]);
    }
    details[detail+1]=total;
    if(!isfinite(total)){br_error(c,64,cell_status,status);return;}
    if(total<deficit){details[detail+2]=deficit-total;br_error(c,32,cell_status,status);return;}
    double remaining=deficit;
    for(int k=offset[c];k<offset[c+1];++k){
        int n=donors[k];long long nr=(long long)n*ns+j;
        double available=fmax(0.0,state[nr]*area[n]);if(available<=0.0)continue;
        double take=fmin(available,remaining);state[nr]=(available-take)/area[n];remaining-=take;
        if(!isfinite(state[nr])){br_error(c,64,cell_status,status);return;}
        if(remaining==0.0)break;
    }
    details[detail+2]=remaining;
    if(remaining!=0.0){br_error(c,128,cell_status,status);return;}
    state[row]=0.0;diagnostics[diag]=1.0;diagnostics[diag+1]=deficit;
}

extern "C" __global__ void br_validate_result(const double* state,const int* suspended,
    int nc,int ns,int ng,int nsp,int secondary,int* cell_status,int* status){
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
    for(int j=0;j<ns;++j){
        double value=state[(long long)c*ns+j];
        if(!isfinite(value))br_error(c,64,cell_status,status);
        if((j==0||j>=3)&&j!=secondary&&value<0.0)br_error(c,4,cell_status,status);
        if(br_contains(suspended,nsp,j)&&value!=0.0)br_error(c,2,cell_status,status);
    }
}
