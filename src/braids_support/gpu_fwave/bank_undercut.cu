// Physical bank-toe mode-2 undercut history and discrete mapped collapse.
// Inputs/outputs are separate trial arrays. The preceding toe supply provides
// ACTUAL removed mass; raw bedload capacity is never used as an undercut proxy.
#define BU_EPS 2.2204460492503131e-16
#define BRAIDS_UNDERCUT_INTERFACE_ABI 3

// Immutable snapshots avoid reading another donor's concurrently debited column.
extern "C" __global__ void bu_current_bed(int nc,int ng,int nl,const double* bulk,
 const double* reference_layers,const double* layers,const double* reference_bed,
 const double* depth,double* bed,int* status) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
    double z=reference_bed[c];int bad=!isfinite(z)||!isfinite(depth[c])||depth[c]<0.;
    for(int l=0;l<nl;++l)for(int g=0;g<ng;++g){int i=(l*ng+g)*nc+c;
        double r=reference_layers[i],a=layers[i];
        if(!isfinite(r)||r<0.||!isfinite(a)||a<0.)bad=1;
        z+=(a-r)/bulk[g];}
    bed[c]=z;status[c]=bad||!isfinite(z);
}

__device__ int bu_exposed(int c,int nc,int ng,int nl,const double* layers,
 const double* bed,const double* depth,const int* toe_cell,
 const int* candidate_offset,const int* candidates,double dry,int* invalid) {
    int t=toe_cell[c];if(t<0||t>=nc){*invalid=1;return -1;}
    double base=bed[t]+depth[t];
    if(!isfinite(base)){*invalid=1;return -1;}
    for(int k=candidate_offset[c];k<candidate_offset[c+1];++k){int s=candidates[k];
        if(depth[s]>dry||!isfinite(depth[s])||depth[s]<0.||!(bed[s]>base))continue;
        int has=0;for(int l=0;l<nl;++l)for(int g=0;g<ng;++g)
            if(layers[(l*ng+g)*nc+s]>0.)has=1;
        if(has)return s;}
    return -1;
}

__device__ void bu_retire_geometry(int c,double* u,double* h,double* wet,double* len){
    u[c]=0.;h[c]=0.;wet[c]=0.;len[c]=0.;
}

extern "C" __global__ void bu_update_history(int nc,int ng,int nl,const double* area,
 const double* bulk,const double* reference_depth,const double* reference_bed,
 const double* toe_removed,const int* is_donor,const int* donor_offset,
 const int* donor_faces,const int* candidate_toe,const double* face_length,
 const double* layers,const double* current_bed,const double* current_depth,
 const int* toe_cell,const int* candidate_offset,const int* candidates,double dry,
 int* exposed_cell,
 double* undercut,double* ref_height,double* toe_height,double* ref_length,
 double* toe_cumulative,int* status) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
    status[c]=0;exposed_cell[c]=-1;if(!is_donor[c])return;
    double actual=0.;for(int g=0;g<ng;++g)actual+=toe_removed[g*nc+c]/bulk[g];
    if(!isfinite(actual)||actual<0.){status[c]=1;return;}
    double cumulative=toe_cumulative[c]+actual/area[c];
    if(!isfinite(cumulative)){status[c]=1;return;}toe_cumulative[c]=cumulative;
    int invalid=0;
    int selected=bu_exposed(c,nc,ng,nl,layers,current_bed,current_depth,
                            toe_cell,candidate_offset,candidates,dry,&invalid);
    if(invalid){status[c]=1;return;}
    exposed_cell[c]=selected;
    double denominator=0.,interface_length=0.;
    for(int k=donor_offset[c];k<donor_offset[c+1];++k){int f=donor_faces[k],toe=candidate_toe[f];
        double height=fmax(reference_bed[c]-reference_bed[toe],0.);
        double wet=fmin(reference_depth[toe],height),len=face_length[f];
        // Match toe-supply eligibility, including interface length. A vanished
        // face must not enlarge a later collapse block at another valid face.
        if(!isfinite(reference_bed[c]-reference_bed[toe])||
           !isfinite(reference_depth[toe])||reference_depth[toe]<0.){status[c]=1;return;}
        if(wet<=0.)continue;
        denominator+=len*wet;interface_length+=len;
    }
    if(!isfinite(denominator)||!isfinite(interface_length)||
       (actual>0.&&denominator<=0.)){
        status[c]=1;return;}
    if(selected<0){bu_retire_geometry(c,undercut,ref_height,toe_height,ref_length);return;}
    // A temporarily dry toe does not erase an existing exposed dry bank.
    if(denominator<=0.)return;
    double next=undercut[c]+(actual>0.?actual/denominator:0.);
    if(!isfinite(next)){status[c]=1;return;}
    int t=toe_cell[c];undercut[c]=next;
    ref_height[c]=current_bed[selected]-current_bed[t];
    toe_height[c]=current_depth[t];ref_length[c]=interface_length;
}

extern "C" __global__ void bu_propose(int nc,int ng,int nl,const double* area,
 const double* bulk,const double* current_depth,const double* current_bed,
 const double* layers,const int* exposed_cell,const int* toe_cell,
 const int* is_donor,double critical,const double* undercut,const double* ref_length,
 double* proposal,double* above_height,int* collapse_cell,int* collapse_target,int* status){
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
    collapse_cell[c]=-1;collapse_target[c]=-1;status[c]=0;
    proposal[c]=0.;above_height[c]=0.;
    if(!is_donor[c]||undercut[c]<critical)return;
    int toe=toe_cell[c],selected=exposed_cell[c];
    if(toe<0||toe>=nc){status[c]=1;return;}
    if(selected<0)return;
    double base=current_bed[toe]+current_depth[toe];
    if(!isfinite(base)){status[c]=1;return;}
    double above=fmax(current_bed[selected]-base,0.);
    if(above<=0.||ref_length[c]<=0.)return;
    // The threshold triggers one physical transaction; the count is not the
    // number of requested critical-width quanta. Never manufacture huge ints.
    double inventory=0.;for(int l=0;l<nl;++l)for(int g=0;g<ng;++g)
        inventory+=layers[(l*ng+g)*nc+selected]*area[selected]/bulk[g];
    double cap=fmin(inventory,area[selected]*above);
    double cross=ref_length[c]*above;
    if(!isfinite(cap)||cap<0.||!isfinite(cross)||cross<=0.||
       !isfinite(undercut[c])){status[c]=1;return;}
    // Comparing widths first avoids overflow from an unbounded old u*L*H.
    double requested=undercut[c]>=cap/cross?cap:undercut[c]*cross;
    if(requested<=0.)return;
    proposal[c]=requested;above_height[c]=above;
    collapse_cell[c]=selected;collapse_target[c]=toe;
}

extern "C" __global__ void bu_limit_shared_toe(int nc,const double* area,
 const double* proposal,const double* above,const int* selected,const int* target,
 const int* toe_donor_offset,const int* toe_donors,double* toe_scale,int* status){
    int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=nc)return;
    double total=0.,scale=1.;int bad=0;
    for(int k=toe_donor_offset[t];k<toe_donor_offset[t+1];++k){int d=toe_donors[k];
        if(target[d]==t)total+=proposal[d];}
    for(int k=toe_donor_offset[t];k<toe_donor_offset[t+1];++k){int d=toe_donors[k];
        if(target[d]!=t||proposal[d]<=0.)continue;
        int s=selected[d];double lowering=proposal[d]/area[s]+total/area[t];
        if(!isfinite(lowering)||lowering<=0.||!isfinite(above[d])){bad=1;continue;}
        scale=fmin(scale,above[d]/lowering);}
    toe_scale[t]=fmax(0.,scale);status[t]=bad||!isfinite(total)||!isfinite(scale);
}

extern "C" __global__ void bu_collapse(int nc,int ng,int nl,const double* area,
 const double* bulk,const double* proposal,const double* above_height,
 const double* toe_scale,const int* collapse_cell,const int* collapse_target,
 double* trial_layers,double* undercut,const double* ref_length,
 double* collapse_cumulative,int* collapse_count,double* removed_layers,
 double* removed_mass,double* actual_volume,int* status){
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
    status[c]=0;actual_volume[c]=0.;
    for(int g=0;g<ng;++g)removed_mass[g*nc+c]=0.;
    int selected=collapse_cell[c],toe=collapse_target[c];if(selected<0||toe<0)return;
    double requested=proposal[c]*toe_scale[toe],remaining=requested,actual=0.;
    if(!isfinite(requested)||requested<0.){status[c]=1;return;}
    for(int l=0;l<nl;++l){
        if(remaining<=0.)break;
        double layer_volume=0.;
        for(int g=0;g<ng;++g)layer_volume+=trial_layers[(l*ng+g)*nc+selected]*area[selected]/bulk[g];
        if(layer_volume<=0.)continue;
        double take=fmin(remaining,layer_volume),factor=fmin(1.,take/layer_volume);
        for(int g=0;g<ng;++g){int i=(l*ng+g)*nc+selected;
            double amount=trial_layers[i]*factor;trial_layers[i]-=amount;
            removed_layers[i]=amount*area[selected];removed_mass[g*nc+c]+=removed_layers[i];
            if(trial_layers[i]<0.&&fabs(trial_layers[i])*area[selected]<=128.*BU_EPS)
                trial_layers[i]=0.;
        }
        actual+=take;remaining=fmax(0.,remaining-take);
    }
    actual_volume[c]=actual;if(actual<=0.)return;
    collapse_cumulative[c]+=actual/area[c];
    if(collapse_count[c]==2147483647){status[c]=1;return;}
    ++collapse_count[c];
    undercut[c]=fmax(0.,undercut[c]-actual/(ref_length[c]*above_height[c]));
    if(!isfinite(collapse_cumulative[c])||!isfinite(undercut[c]))status[c]=1;
}

extern "C" __global__ void bu_reconcile(int nc,int ng,int nl,const double* layers,
 const double* bed,const double* depth,const int* toe_cell,const int* is_donor,
 const int* candidate_offset,const int* candidates,double dry,
 double* undercut,double* ref_height,double* toe_height,double* ref_length,int* status){
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
    status[c]=0;if(!is_donor[c])return;
    int invalid=0;
    int selected=bu_exposed(c,nc,ng,nl,layers,bed,depth,toe_cell,candidate_offset,candidates,dry,&invalid);
    if(invalid){status[c]=1;return;}
    if(selected<0)bu_retire_geometry(c,undercut,ref_height,toe_height,ref_length);
}

extern "C" __global__ void bu_deposit(int nc,int ng,const double* area,
 const double* removed_mass,const int* target,const int* toe_donor_offset,
 const int* toe_donors,double* trial_layers,
 double* recipient_gain,int* status) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;int bad=0;
    for(int g=0;g<ng;++g){int i=g*nc+c;double value=trial_layers[i],gain=0.;
        // Fixed donor order matches native's sequential increment for each
        // collapse event, even if multiple donors deposit at the same toe.
        for(int k=toe_donor_offset[c];k<toe_donor_offset[c+1];++k){
            int donor=toe_donors[k];if(target[donor]!=c)continue;
            double amount=removed_mass[g*nc+donor];gain+=amount;
            value=value+amount/area[c];}
        trial_layers[i]=value;recipient_gain[i]=gain;
        if(!isfinite(value)||value<0.||!isfinite(gain))bad=1;
    }
    status[c]=bad;
}

extern "C" __global__ void bu_budget(int nc,int ng,const double* removed,
 const double* gain,double* budget,int* status) {
    int g=blockIdx.x*blockDim.x+threadIdx.x;if(g>=ng)return;
    double debit=0.,credit=0.;for(int c=0;c<nc;++c){
        debit+=removed[g*nc+c];credit+=gain[g*nc+c];}
    budget[2*g]=debit;budget[2*g+1]=credit;
    status[g]=!isfinite(debit)||!isfinite(credit)||
        fabs(debit-credit)>512.*BU_EPS*fmax(fmax(fabs(debit),fabs(credit)),1.);
}

extern "C" __global__ void bu_validate(int nc,int ng,int nl,const double* reference_layers,
 const double* reference_depth,const double* reference_bed,
 const double* toe_removed,const double* layers,
 const double* undercut,const double* ref_height,const double* toe_height,
 const double* ref_length,const double* toe_cumulative,
 const double* collapse_cumulative,const int* collapse_count,int* status) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;int bad=0;
    if(!isfinite(reference_depth[c])||reference_depth[c]<0.||
       !isfinite(reference_bed[c]))bad=1;
    for(int g=0;g<ng;++g){double x=toe_removed[g*nc+c];
        if(!isfinite(x)||x<0.)bad=1;}
    for(int l=0;l<nl;++l)for(int g=0;g<ng;++g){
        double x=layers[(l*ng+g)*nc+c],r=reference_layers[(l*ng+g)*nc+c];
        if(!isfinite(x)||x<0.||!isfinite(r)||r<0.)bad=1;}
    const double* fields[6]={undercut,ref_height,toe_height,ref_length,
                              toe_cumulative,collapse_cumulative};
    for(int i=0;i<6;++i)if(!isfinite(fields[i][c])||fields[i][c]<0.)bad=1;
    if(collapse_count[c]<0)bad=1;status[c]=bad;
}
