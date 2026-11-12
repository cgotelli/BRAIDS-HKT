// Native physical toe mode-2 contract: accepted Exner demand, deterministic
// face allocation, surface-down multilayer extraction, routed face mass and
// recipient-only deposition. Donors are never debited by SurfaceTransfer.
#define TS_EPS 2.2204460492503131e-16

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
 double capacity,double* request,int* status) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
    int first=toe_offset[c],last=toe_offset[c+1];double total=0.;
    for(int k=first;k<last;++k)total+=length[toe_faces[k]];
    double remaining=capacity*demand[c];
    status[c]=!isfinite(remaining)||(remaining>0.&&total<=0.);
    if(status[c]||remaining<=0.)return;
    for(int k=first;k<last;++k){int f=toe_faces[k];double amount;
        if(k==last-1)amount=remaining;
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
        if(remaining<=64.*TS_EPS*fmax(requested,1.)||layer_volume<=0.)continue;
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
