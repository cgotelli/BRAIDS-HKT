// Native physical toe mode-2 contract: accepted Exner demand, deterministic
// face allocation, surface-down multilayer extraction, routed face mass and
// recipient-only deposition. Donors are never debited by SurfaceTransfer.
#define TS_EPS 2.2204460492503131e-16
#define BRAIDS_TOE_INTERFACE_ABI 4

extern "C" __global__ void ts_demand(int nc,int ng,int nl,const double* before,
 const double* after,const double* area,const double* bulk,const int* toe_cell,
 double* demand_mass,double* demand_volume,int* status) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
    int bad=0;double volume=0.;
    for(int g=0;g<ng;++g){double old=0.,next=0.;
        for(int l=0;l<nl;++l){int i=(l*ng+g)*nc+c;
            double b=before[i],a=after[i];
            if(!isfinite(b)||!isfinite(a)||b<0.||a<0.)bad=1;
            old+=b;next+=a;}
        double mass=toe_cell[c]?fmax((old-next)*area[c],0.):0.;
        demand_mass[g*nc+c]=mass;volume+=mass/bulk[g];
    }
    status[c]=bad||!isfinite(volume);demand_volume[c]=status[c]?0.:volume;
}

extern "C" __global__ void ts_face_requests(int nc,const double* demand,
 const double* length,const int* toe_offset,const int* toe_faces,
 const int* candidate_bank,const double* reference_bed,const double* reference_depth,
 double capacity,double* request,int* eligible,double* interface_area,int* status) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
    int first=toe_offset[c],last=toe_offset[c+1];double total=0.;
    int final_face=-1,bad=!isfinite(reference_bed[c])||
        !isfinite(reference_depth[c])||reference_depth[c]<0.;
    // The same beginning-of-step reference geometry is used by undercut.
    // A deposited collapse block can bury a previously mapped interface.
    // Check it BEFORE extracting inventory; static mapping alone is not enough.
    for(int k=first;k<last;++k){int f=toe_faces[k],b=candidate_bank[f];
        double difference=reference_bed[b]-reference_bed[c];
        double wet=fmin(fmax(reference_depth[c],0.),fmax(difference,0.));
        double surface=length[f]*wet;
        int valid=isfinite(reference_bed[b])&&isfinite(difference)&&isfinite(surface);
        if(!valid)bad=1;
        eligible[f]=valid&&!bad&&wet>0.;
        interface_area[f]=valid?surface:0.;
        if(eligible[f]){total+=length[f];final_face=f;}
    }
    double remaining=capacity*demand[c];
    status[c]=bad||!isfinite(total)||!isfinite(remaining)||remaining<0.;
    // No submerged interface means no lateral supply, not a failed Exner step.
    if(status[c]||remaining<=0.||final_face<0)return;
    for(int k=first;k<last;++k){int f=toe_faces[k];double amount;
        if(!eligible[f])continue;
        if(f==final_face)amount=remaining;
        else{amount=capacity*demand[c]*length[f]/total;amount=fmin(amount,remaining);}
        request[f]=amount;remaining=fmax(0.,remaining-amount);
    }
}

extern "C" __global__ void ts_donor_requests(int nc,const int* donor_offset,
 const int* donor_faces,const double* face_request,double* donor_request,int* status) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
    double value=0.;
    for(int k=donor_offset[c];k<donor_offset[c+1];++k)value+=face_request[donor_faces[k]];
    status[c]=!isfinite(value)||value<0.;donor_request[c]=status[c]?0.:value;
}

// Limit allocated requests by the real multilayer donor column BEFORE shared
// geometry is gathered. Unavailable material cannot raise a recipient or lower
// a donor and must not suppress a healthy neighbour. Each donor owns its faces,
// so this proportional reduction requires neither atomics nor redistribution.
extern "C" __global__ void ts_inventory_limit(int nc,int ng,int nl,
 const double* after,const double* area,const double* bulk,
 const int* donor_offset,const int* donor_faces,const double* requested,
 double* request,double* available,double* inventory_scale,int* status) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
    double inventory=0.;int bad=0;
    for(int l=0;l<nl;++l){double volume=0.;
        for(int g=0;g<ng;++g){double value=after[(l*ng+g)*nc+c];
            if(!isfinite(value)||value<0.)bad=1;
            volume+=value*area[c]/bulk[g];}
        inventory+=volume;}
    double demand=requested[c],scale=1.;
    if(!isfinite(inventory)||inventory<0.||!isfinite(demand)||demand<0.)bad=1;
    if(bad)scale=0.;else if(demand>inventory)scale=inventory/demand;
    for(int k=donor_offset[c];k<donor_offset[c+1];++k)
        if(scale<1.)request[donor_faces[k]]*=scale;
    available[c]=bad?0.:inventory;inventory_scale[c]=scale;status[c]=bad;
}

// Geometry limiter: first assemble ALL potential donor lowering and recipient
// raising. Per-face independent caps are insufficient for shared recipients.
// Separate kernels provide global ordering without atomics or CPU transfers.
extern "C" __global__ void ts_geometry_state(int nc,int ng,int nl,
 const double* before,const double* after,const double* bulk,const double* bed,
 const int* toe_offset,const int* toe_faces,const double* request,
 double* post_bed,double* incoming,int* status) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
    double change=0.,gain=0.;
    for(int l=0;l<nl;++l)for(int g=0;g<ng;++g){int i=(l*ng+g)*nc+c;
        change+=(after[i]-before[i])/bulk[g];}
    for(int k=toe_offset[c];k<toe_offset[c+1];++k)gain+=request[toe_faces[k]];
    post_bed[c]=bed[c]+change;incoming[c]=gain;
    status[c]=!isfinite(post_bed[c])||!isfinite(gain)||gain<0.;
}

extern "C" __global__ void ts_geometry_scales(int nc,const double* area,
 const double* length,const int* eligible,const double* surface,
 const int* donor_offset,const int* donor_faces,const int* toe_offset,
 const int* toe_faces,const int* bank,const int* toe,const double* bed,
 const double* outgoing,const double* incoming,double* donor_scale,
 double* toe_scale,int* status) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
    double ds=1.,ts=1.,length_sum=0.,surface_sum=0.;int bad=0;
    for(int k=donor_offset[c];k<donor_offset[c+1];++k){int f=donor_faces[k];
        if(!eligible[f])continue;
        int t=toe[f];double gap=bed[c]-bed[t];
        double closure=outgoing[c]/area[c]+incoming[t]/area[t];
        if(!isfinite(gap)||!isfinite(closure))bad=1;
        if(closure>0.)ds=fmin(ds,fmax(gap,0.)/closure);
        length_sum+=length[f];surface_sum+=surface[f];
    }
    // Area belongs to one donor column, not independently to each of its faces.
    // This bounds per-step retreat by area / total eligible interface length.
    if(outgoing[c]>0.&&length_sum>0.)
        ds=fmin(ds,(area[c]*(surface_sum/length_sum))/outgoing[c]);
    for(int k=toe_offset[c];k<toe_offset[c+1];++k){int f=toe_faces[k];
        if(!eligible[f])continue;
        int b=bank[f];double gap=bed[b]-bed[c];
        double closure=outgoing[b]/area[b]+incoming[c]/area[c];
        if(!isfinite(gap)||!isfinite(closure))bad=1;
        if(closure>0.)ts=fmin(ts,fmax(gap,0.)/closure);
    }
    if(!isfinite(ds)||!isfinite(ts)||!isfinite(length_sum)||!isfinite(surface_sum))bad=1;
    donor_scale[c]=bad?0.:fmax(0.,fmin(1.,ds));
    toe_scale[c]=bad?0.:fmax(0.,fmin(1.,ts));status[c]=bad;
}

extern "C" __global__ void ts_limit_requests(int nf,const int* eligible,
 const int* bank,const int* toe,const double* donor_scale,const double* toe_scale,
 double* request) {
    int f=blockIdx.x*blockDim.x+threadIdx.x;if(f>=nf)return;
    if(eligible[f])request[f]*=fmin(donor_scale[bank[f]],toe_scale[toe[f]]);
}

extern "C" __global__ void ts_extract(int nc,int ng,int nl,const double* after,
 const double* area,const double* bulk,const double* donor_request,
 double* trial,double* removed_layers,double* removed_mass,double* actual,int* status) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
    double requested=donor_request[c],remaining=fmax(0.,requested),taken=0.;int bad=0;
    for(int g=0;g<ng;++g)removed_mass[g*nc+c]=0.;
    for(int l=0;l<nl;++l){
        double layer_volume=0.;
        for(int g=0;g<ng;++g){int i=(l*ng+g)*nc+c;double value=after[i];
            trial[i]=value;removed_layers[i]=0.;
            if(!isfinite(value)||value<0.)bad=1;
            layer_volume+=value*area[c]/bulk[g];}
        if(remaining<=0.||layer_volume<=0.)continue;
        double take=fmin(remaining,layer_volume),factor=fmin(1.,take/layer_volume);
        for(int g=0;g<ng;++g){int i=(l*ng+g)*nc+c;
            double amount=trial[i]*factor;trial[i]-=amount;
            removed_layers[i]=amount*area[c];removed_mass[g*nc+c]+=removed_layers[i];
            if(trial[i]<0.&&fabs(trial[i])*area[c]<=128.*TS_EPS)trial[i]=0.;}
        taken+=take;remaining=fmax(0.,remaining-take);
    }
    if(!isfinite(requested)||requested<0.||!isfinite(taken))bad=1;
    actual[c]=bad?0.:taken;status[c]=bad;
}

extern "C" __global__ void ts_route(int nc,int nf,int ng,const int* donor_offset,
 const int* donor_faces,const double* face_request,const double* donor_request,
 const double* removed,const int* owner,const int* candidate_bank,
 double* face_mass,int* status) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
    double requested=donor_request[c];int first=donor_offset[c],last=donor_offset[c+1];
    int final_face=-1;for(int k=first;k<last;++k){int f=donor_faces[k];
        if(face_request[f]>0.)final_face=f;}
    int bad=(requested<=0.&&final_face>=0)||(requested>0.&&final_face<0);
    double left[32];for(int g=0;g<ng;++g){left[g]=removed[g*nc+c];
        if(requested<=0.&&left[g]>0.)bad=1;}
    if(!bad&&requested>0.)for(int k=first;k<last;++k){int f=donor_faces[k];
        if(face_request[f]<=0.)continue;
        for(int g=0;g<ng;++g){double amount;
            if(f==final_face)amount=left[g];
            else{amount=removed[g*nc+c]*face_request[f]/requested;
                amount=fmin(amount,left[g]);}
            left[g]=fmax(0.,left[g]-amount);
            face_mass[g*nf+f]=(owner[f]==candidate_bank[f]?amount:-amount);
        }
    }
    status[c]=bad;
}

extern "C" __global__ void ts_deposit(int nc,int nf,int ng,const double* area,
 const int* offset,const int* faces,const int* signs,const double* face_mass,
 double* trial,double* recipient_gain,int* status) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
    int bad=0;
    for(int g=0;g<ng;++g){double incoming=0.;
        for(int k=offset[c];k<offset[c+1];++k){int f=faces[k];double amount=face_mass[g*nf+f];
            if(signs[k]>0){if(amount<0.)incoming-=amount;}
            else if(amount>0.)incoming+=amount;
        }
        recipient_gain[g*nc+c]=incoming;
        int i=g*nc+c;double value=trial[i]+incoming/area[c];
        if(!isfinite(value)||value<0.)bad=1;trial[i]=value;
    }
    status[c]=bad;
}

extern "C" __global__ void ts_budget(int nc,int ng,const double* removed,
 const double* recipient_gain,double* budget,int* status) {
    int g=blockIdx.x*blockDim.x+threadIdx.x;if(g>=ng)return;
    double debit=0.,credit=0.;
    for(int c=0;c<nc;++c){debit+=removed[g*nc+c];credit+=recipient_gain[g*nc+c];}
    budget[2*g]=debit;budget[2*g+1]=credit;
    double scale=fmax(fmax(fabs(debit),fabs(credit)),1.);
    status[g]=!isfinite(debit)||!isfinite(credit)||fabs(debit-credit)>512.*TS_EPS*scale;
}
