// Physical bank-toe mode-2 undercut history and discrete mapped collapse.
// Inputs/outputs are separate trial arrays. The preceding toe supply provides
// ACTUAL removed mass; raw bedload capacity is never used as an undercut proxy.
#define BU_EPS 2.2204460492503131e-16

extern "C" __global__ void bu_update_history(int nc,int ng,const double* area,
 const double* bulk,const double* reference_depth,const double* reference_bed,
 const double* toe_removed,const int* is_donor,const int* donor_offset,
 const int* donor_faces,const int* candidate_toe,const double* face_length,
 double* undercut,double* ref_height,double* toe_height,double* ref_length,
 double* toe_cumulative,int* status) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
    status[c]=0;if(!is_donor[c])return;
    double actual=0.;for(int g=0;g<ng;++g)actual+=toe_removed[g*nc+c]/bulk[g];
    if(!isfinite(actual)||actual<0.){status[c]=1;return;}
    if(actual<=0.)return;
    double denominator=0.,interface_length=0.,height_max=ref_height[c],toe_max=toe_height[c];
    for(int k=donor_offset[c];k<donor_offset[c+1];++k){int f=donor_faces[k],toe=candidate_toe[f];
        double height=fmax(reference_bed[c]-reference_bed[toe],0.);
        double wet=fmin(reference_depth[toe],height),len=face_length[f];
        denominator+=len*wet;interface_length+=len;
        height_max=fmax(height_max,height);toe_max=fmax(toe_max,wet);
    }
    if(!isfinite(denominator)||denominator<=0.||!isfinite(interface_length)){
        status[c]=1;return;}
    double next=undercut[c]+actual/denominator;
    double cumulative=toe_cumulative[c]+actual/area[c];
    if(!isfinite(next)||!isfinite(cumulative)){status[c]=1;return;}
    undercut[c]=next;ref_height[c]=height_max;toe_height[c]=toe_max;
    ref_length[c]=fmax(ref_length[c],interface_length);toe_cumulative[c]=cumulative;
}

extern "C" __global__ void bu_collapse(int nc,int ng,int nl,const double* area,
 const double* bulk,const double* reference_depth,const double* reference_bed,
 const double* reference_layers,const int* candidate_offset,const int* candidates,
 const int* toe_cell,const int* is_donor,double dry,
 double critical,double* trial_layers,double* undercut,double* ref_height,
 double* toe_height,double* ref_length,double* collapse_cumulative,
 int* collapse_count,double* removed_layers,double* removed_mass,
 int* collapse_cell,int* collapse_target,int* status) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
    collapse_cell[c]=-1;collapse_target[c]=-1;status[c]=0;
    for(int g=0;g<ng;++g)removed_mass[g*nc+c]=0.;
    if(!is_donor[c]||undercut[c]<critical)return;
    double event_real=undercut[c]/critical;
    if(!isfinite(event_real)||event_real>2147483647.){status[c]=1;return;}
    int events=(int)event_real,toe=toe_cell[c],selected=-1;
    if(toe<0||toe>=nc){status[c]=1;return;}
    for(int k=candidate_offset[c];k<candidate_offset[c+1];++k){
        int candidate=candidates[k];
        if(reference_depth[candidate]>dry)continue;
        int has=0;for(int g=0;g<ng;++g){double inventory=0.;
            for(int l=0;l<nl;++l)inventory+=reference_layers[(l*ng+g)*nc+candidate];
            if(inventory>0.)has=1;}
        if(!has||reference_bed[candidate]<=reference_bed[toe])continue;
        if(selected<0)selected=candidate;
    }
    if(selected<0)return;
    double above=fmax(reference_bed[selected]-reference_bed[toe]-toe_height[c],0.);
    if(above<=0.||ref_length[c]<=0.)return;
    double requested=events*critical*ref_length[c]*above;
    if(!isfinite(requested)||requested<0.){status[c]=1;return;}
    double remaining=requested,actual=0.;
    for(int l=0;l<nl;++l){
        if(remaining<=64.*BU_EPS*fmax(requested,1.))break;
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
    collapse_cell[c]=selected;collapse_target[c]=toe;
    if(actual>0.)collapse_cumulative[c]+=actual/area[c];
    if(collapse_count[c]>2147483647-events){status[c]=1;return;}
    collapse_count[c]+=events;
    double residual=undercut[c]-events*critical;
    undercut[c]=fmax(0.,residual);
    if(undercut[c]<=64.*BU_EPS*critical){
        undercut[c]=0.;ref_height[c]=0.;toe_height[c]=0.;ref_length[c]=0.;}
    if(!isfinite(collapse_cumulative[c])||!isfinite(undercut[c]))status[c]=1;
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
